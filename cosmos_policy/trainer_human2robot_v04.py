# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extended Trainer for Cosmos Policy with epoch tracking.

This trainer extends the base ImaginaireTrainer to add:
- Epoch tracking and sampler epoch setting for proper distributed sampling
"""

import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import torch
import torch.utils.data

from cosmos_policy._src.imaginaire.model import ImaginaireModel
from cosmos_policy._src.imaginaire.trainer import ImaginaireTrainer
from cosmos_policy._src.imaginaire.utils import distributed, log, misc
from cosmos_policy._src.imaginaire.utils.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling


def _finite_training_value(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value.detach()).all())
    if isinstance(value, dict):
        return all(_finite_training_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_training_value(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(torch.isfinite(torch.tensor(float(value))))
    return True


def _training_scalars(value: Any, prefix: str = "loss") -> dict[str, float]:
    if isinstance(value, torch.Tensor):
        return {prefix: float(value.detach().float().mean().cpu())}
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for key, item in value.items():
            result.update(_training_scalars(item, f"{prefix}.{key}"))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(_training_scalars(item, f"{prefix}.{index}"))
        return result
    if isinstance(value, (int, float)):
        return {prefix: float(value)}
    return {}


def _write_progress(iteration: int, loss: Any, optimizer, started: float) -> None:
    progress_path = os.environ.get("HUMAN2ROBOT_V04_STAGE5_PROGRESS_PATH")
    if not progress_path or torch.distributed.get_rank() != 0:
        return
    path = Path(progress_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "human2robot-v04-stage5-training-progress-v1",
        "status": "RUNNING" if iteration < 7000 else "TRAINING_STEPS_COMPLETED",
        "method": os.environ.get("HUMAN2ROBOT_V04_STAGE5_METHOD"),
        "optimizer_step": int(iteration),
        "max_optimizer_steps": 7000,
        "loss": _training_scalars(loss),
        "learning_rate": [float(group["lr"]) for group in optimizer.param_groups],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
        "loss_finite_enforced": True,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Human2RobotV04Trainer(ImaginaireTrainer):
    """
    Extended Trainer for Cosmos Policy.

    Adds special handling for:
    - Epoch tracking to properly set dataloader sampler epochs (needed for distributed training)
    - Simplified initial validation check (removes run_validation_on_start requirement)
    """

    def __init__(self, config):
        super().__init__(config)

    def train(
        self,
        model: ImaginaireModel,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ) -> None:
        """The training function.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_train (torch.utils.data.DataLoader): The training data loader.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
        """
        # Leaving this for backward compability for now, but we can think about moving this to model.on_train_start for all models.
        model = model.to("cuda", memory_format=self.config.trainer.memory_format)  # type: ignore
        model.on_train_start(self.config.trainer.memory_format)

        # Initialize the optimizer, scheduler, and grad_scaler.
        self.callbacks.on_optimizer_init_start()
        optimizer, scheduler = model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
        self.callbacks.on_optimizer_init_end()
        # Load the model checkpoint and get the starting iteration number.
        iteration = self.checkpointer.load(model, optimizer, scheduler, grad_scaler)
        grad_accum_iter = 0
        log.critical(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        if self.config.trainer.distributed_parallelism == "ddp":
            # Create a DDP model wrapper.
            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            model_ddp = model
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")

        log.info("Starting training...")
        self.callbacks.on_train_start(model, iteration=iteration)
        # Initial validation.
        if self.config.trainer.run_validation and iteration == 0 and self.config.trainer.run_validation_on_start:
            self.validate(model, dataloader_val, iteration=iteration)
        _end_training = False
        training_started_monotonic = time.monotonic()
        with (
            maybe_enable_profiling(self.config, global_step=iteration) as torch_profiler,
            maybe_enable_memory_snapshot(self.config, global_step=iteration) as memory_profiler,
        ):
            epoch = 0
            while True:
                dataloader_train.sampler.set_epoch(epoch)
                dataloader_train_iter = iter(dataloader_train)
                while True:
                    self.callbacks.on_before_dataloading(iteration)
                    try:
                        with (
                            self.training_timer("dataloader_train"),
                            self.straggler_detector.profile_section(
                                "dataloading",
                                self.config.trainer.straggler_detection.analyze_dataloading,
                                profile_cuda=False,
                            ),
                        ):
                            data_batch = next(dataloader_train_iter)
                    except StopIteration:
                        break
                    finally:
                        self.callbacks.on_after_dataloading(iteration)
                    # If max_iter is reached, exit the training loop.
                    if iteration >= self.config.trainer.max_iter:
                        _end_training = True
                        break
                    # Move all tensors in the data batch to GPU device.
                    data_batch = misc.to(data_batch, device="cuda")
                    # The actual training step.
                    self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
                    self.callbacks.on_training_step_batch_start(model, data_batch, iteration=iteration)
                    if not model.training:
                        model_ddp.train()
                    assert model_ddp.training, "model_ddp is not in training mode."
                    assert model.training, "model is not in training mode."
                    output_batch, loss, grad_accum_iter = self.training_step(
                        model_ddp,
                        optimizer,
                        scheduler,
                        grad_scaler,
                        data_batch,
                        iteration=iteration,
                        grad_accum_iter=grad_accum_iter,
                    )
                    finite = torch.tensor(
                        int(_finite_training_value(loss)),
                        device="cuda",
                        dtype=torch.int32,
                    )
                    torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
                    if int(finite.item()) != 1:
                        raise FloatingPointError("Human2Robot v04 stage-5 loss became nonfinite")
                    self.callbacks.on_training_step_batch_end(
                        model, data_batch, output_batch, loss, iteration=iteration
                    )
                    # If the gradients are still being accumulated, continue to load the next training batch.
                    if grad_accum_iter != 0:
                        continue
                    # Do the following when an actual optimizer (update) step has been made.
                    iteration += 1
                    if iteration % 10 == 0 or iteration == 7000:
                        _write_progress(iteration, loss, optimizer, training_started_monotonic)
                    # Save checkpoint.
                    if iteration % self.config.checkpoint.save_iter == 0:
                        # Free fragmented allocator pages before DCP gather_object —
                        # the SavePlan pickle on rank 0 spikes GPU memory and we've
                        # seen NCCL Error 1 ("unhandled cuda error") here when one
                        # rank silently OOMs. Costs ~ms; reduces intermittent fail.
                        torch.cuda.empty_cache()
                        self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
                    self.callbacks.on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)
                    # Validation.
                    if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0:
                        self.validate(model, dataloader_val, iteration=iteration)
                    # This iteration is successful; reset the timeout signal.
                    signal.alarm(self.config.trainer.timeout_period)
                    self.straggler_detector.generate_report(iteration)
                    if torch_profiler:
                        torch_profiler.step()
                    if memory_profiler:
                        memory_profiler.step()
                epoch += 1
                if _end_training:
                    break
        log.success("Done with training.")
        if iteration % self.config.checkpoint.save_iter != 0:
            torch.cuda.empty_cache()
            self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
        self.callbacks.on_train_end(model, iteration=iteration)
        self.checkpointer.finalize()
        distributed.barrier()
        self.callbacks.on_app_end()
