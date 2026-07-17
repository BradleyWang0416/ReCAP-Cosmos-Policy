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
Cosmos Policy training script with manual DistributedSampler instantiation.

This script extends the base training script to manually create DistributedSampler
instead of using instantiate(), avoiding duplicate dataset creation.
"""

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from loguru import logger as logging
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.config import Config, load_config, pretty_print_overrides
from cosmos_policy._src.imaginaire.lazy_config import LazyConfig, instantiate
from cosmos_policy._src.imaginaire.serialization import to_yaml
from cosmos_policy._src.imaginaire.utils import distributed
from cosmos_policy._src.imaginaire.utils.context_managers import data_loader_init, distributed_init, model_init
from cosmos_policy._src.imaginaire.utils.launch import log_reproducible_setup


def _write_optional_human2robot_p2_runtime_binding(config: Config) -> None:
    """Hard-bind a formal Human2Robot P2 run to its actual distributed state.

    The hook is dormant for every non-P2 job.  The P2 orchestrator supplies the
    binding path and frozen expectations through environment variables.  All
    ranks validate the same values before model construction or optimizer
    steps; rank zero atomically writes the evidence file.
    """

    binding_path_value = os.environ.get("HUMAN2ROBOT_P2_RUNTIME_BINDING_PATH")
    if not binding_path_value:
        return

    required_env = (
        "HUMAN2ROBOT_P2_PROTOCOL_SHA256",
        "HUMAN2ROBOT_P2_FOUR_GPU_SUCCESSOR_SHA256",
        "HUMAN2ROBOT_P2_MEMORY_SUCCESSOR_SHA256",
        "HUMAN2ROBOT_P2_IO_SUCCESSOR_SHA256",
        "HUMAN2ROBOT_P2_LOGGING_SUCCESSOR_SHA256",
        "HUMAN2ROBOT_P2_CODE_SHA256",
        "HUMAN2ROBOT_P2_CELL_ID",
        "HUMAN2ROBOT_P2_EXPERIMENT_ID",
        "HUMAN2ROBOT_P2_VARIANT_ID",
        "HUMAN2ROBOT_P2_METHOD_ID",
        "HUMAN2ROBOT_P2_EXPECTED_WORLD_SIZE",
        "HUMAN2ROBOT_P2_EXPECTED_DP_WORLD_SIZE",
        "HUMAN2ROBOT_P2_EXPECTED_SEED",
        "HUMAN2ROBOT_P2_EXPECTED_MAX_ITER",
        "HUMAN2ROBOT_P2_EXPECTED_BATCH_PER_DP_RANK",
        "HUMAN2ROBOT_P2_EXPECTED_GRAD_ACCUM_STEPS",
        "HUMAN2ROBOT_P2_EXPECTED_FSDP_SHARD_SIZE",
        "HUMAN2ROBOT_P2_EXPECTED_EFFECTIVE_GLOBAL_BATCH",
        "HUMAN2ROBOT_P2_EXPECTED_SAVE_ITER",
        "HUMAN2ROBOT_P2_EXPECTED_H_STEPS",
        "HUMAN2ROBOT_P2_EXPECTED_K_STEPS",
        "HUMAN2ROBOT_P2_EXPECTED_TOP_K",
        "HUMAN2ROBOT_P2_EXPECTED_POOL_SIZE",
        "HUMAN2ROBOT_P2_EXPECTED_RETRIEVAL_MODALITY",
        "HUMAN2ROBOT_P2_EXPECTED_TIME_VIEW_ID",
        "HUMAN2ROBOT_P2_EXPECTED_QUERY_OFFSET",
        "HUMAN2ROBOT_P2_EXPECTED_TARGET_REPRESENTATION",
        "HUMAN2ROBOT_P2_EXPECTED_INIT_CKPT_PATH",
        "HUMAN2ROBOT_P2_EXPECTED_TOKENIZER_PATH",
        "HUMAN2ROBOT_P2_EXPECTED_PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_NCCL_TRACE_BUFFER_SIZE",
        "TORCH_NCCL_DUMP_ON_TIMEOUT",
        "TORCH_NCCL_DESYNC_DEBUG",
        "NCCL_DEBUG",
        "HUMAN2ROBOT_P2_SLOW_SAMPLE_SECONDS",
    )
    missing = [name for name in required_env if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing formal Human2Robot P2 environment bindings: {missing}")

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    dp_world_size = parallel_state.get_data_parallel_world_size()
    dp_rank = parallel_state.get_data_parallel_rank()
    dataset = config.dataloader_train.dataset
    model_config = config.model.config
    actual = {
        "experiment_id": str(dataset.experiment_id),
        "variant_id": str(dataset.variant_id),
        "method_id": str(dataset.method_id),
        "world_size": int(world_size),
        "data_parallel_world_size": int(dp_world_size),
        "seed": int(config.trainer.seed),
        "max_optimizer_steps": int(config.trainer.max_iter),
        "batch_size_per_data_parallel_rank": int(config.dataloader_train.batch_size),
        "checkpoint_save_every_steps": int(config.checkpoint.save_iter),
        "gradient_accumulation_steps": int(config.trainer.grad_accum_iter),
        "fsdp_shard_size": int(model_config.fsdp_shard_size),
        "effective_global_batch_size": int(dp_world_size)
        * int(config.dataloader_train.batch_size)
        * int(config.trainer.grad_accum_iter),
        "visible_cuda_device_count": int(torch.cuda.device_count()),
        "sampler_seed": int(config.dataloader_train.sampler.seed),
        "H_steps": int(dataset.h_steps),
        "K_steps": int(dataset.k_steps),
        "top_k": int(dataset.top_k),
        "pool_size": int(dataset.pool_size),
        "retrieval_modality": str(dataset.retrieval_modality),
        "time_view_id": str(dataset.time_view_id),
        "query_offset_view_steps": int(dataset.query_offset_view_steps),
        "target_representation": str(dataset.target_representation),
        "action_dim": int(model_config.action_dim),
        "proprio_dim": int(model_config.proprio_dim),
        "state_t": int(model_config.state_t),
        "min_num_conditional_frames": int(model_config.min_num_conditional_frames),
        "max_num_conditional_frames": int(model_config.max_num_conditional_frames),
        "tokenizer_chunk_duration": int(model_config.tokenizer.chunk_duration),
        "precision": str(model_config.precision),
        "num_duplicates_per_image": int(dataset.num_duplicates_per_image),
        "use_image_aug": bool(dataset.use_image_aug),
        "initialization_checkpoint_path": str(config.checkpoint.load_path),
        "tokenizer_checkpoint_path": str(model_config.tokenizer.vae_pth),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "torch_nccl_trace_buffer_size": os.environ.get("TORCH_NCCL_TRACE_BUFFER_SIZE"),
        "torch_nccl_dump_on_timeout": os.environ.get("TORCH_NCCL_DUMP_ON_TIMEOUT"),
        "torch_nccl_desync_debug": os.environ.get("TORCH_NCCL_DESYNC_DEBUG"),
        "nccl_debug": os.environ.get("NCCL_DEBUG"),
        "nccl_debug_subsys": os.environ.get("NCCL_DEBUG_SUBSYS"),
        "slow_sample_seconds": os.environ.get("HUMAN2ROBOT_P2_SLOW_SAMPLE_SECONDS"),
    }
    expected = {
        "experiment_id": os.environ["HUMAN2ROBOT_P2_EXPERIMENT_ID"],
        "variant_id": os.environ["HUMAN2ROBOT_P2_VARIANT_ID"],
        "method_id": os.environ["HUMAN2ROBOT_P2_METHOD_ID"],
        "world_size": int(os.environ["HUMAN2ROBOT_P2_EXPECTED_WORLD_SIZE"]),
        "data_parallel_world_size": int(os.environ["HUMAN2ROBOT_P2_EXPECTED_DP_WORLD_SIZE"]),
        "seed": int(os.environ["HUMAN2ROBOT_P2_EXPECTED_SEED"]),
        "max_optimizer_steps": int(os.environ["HUMAN2ROBOT_P2_EXPECTED_MAX_ITER"]),
        "batch_size_per_data_parallel_rank": int(
            os.environ["HUMAN2ROBOT_P2_EXPECTED_BATCH_PER_DP_RANK"]
        ),
        "checkpoint_save_every_steps": int(os.environ["HUMAN2ROBOT_P2_EXPECTED_SAVE_ITER"]),
        "gradient_accumulation_steps": int(
            os.environ["HUMAN2ROBOT_P2_EXPECTED_GRAD_ACCUM_STEPS"]
        ),
        "fsdp_shard_size": int(os.environ["HUMAN2ROBOT_P2_EXPECTED_FSDP_SHARD_SIZE"]),
        "effective_global_batch_size": int(
            os.environ["HUMAN2ROBOT_P2_EXPECTED_EFFECTIVE_GLOBAL_BATCH"]
        ),
        "visible_cuda_device_count": int(os.environ["HUMAN2ROBOT_P2_EXPECTED_WORLD_SIZE"]),
        "sampler_seed": int(os.environ["HUMAN2ROBOT_P2_EXPECTED_SEED"]),
        "H_steps": int(os.environ["HUMAN2ROBOT_P2_EXPECTED_H_STEPS"]),
        "K_steps": int(os.environ["HUMAN2ROBOT_P2_EXPECTED_K_STEPS"]),
        "top_k": int(os.environ["HUMAN2ROBOT_P2_EXPECTED_TOP_K"]),
        "pool_size": int(os.environ["HUMAN2ROBOT_P2_EXPECTED_POOL_SIZE"]),
        "retrieval_modality": os.environ["HUMAN2ROBOT_P2_EXPECTED_RETRIEVAL_MODALITY"],
        "time_view_id": os.environ["HUMAN2ROBOT_P2_EXPECTED_TIME_VIEW_ID"],
        "query_offset_view_steps": int(os.environ["HUMAN2ROBOT_P2_EXPECTED_QUERY_OFFSET"]),
        "target_representation": os.environ[
            "HUMAN2ROBOT_P2_EXPECTED_TARGET_REPRESENTATION"
        ],
        "action_dim": 10,
        "proprio_dim": 10,
        "state_t": 8 + int(os.environ["HUMAN2ROBOT_P2_EXPECTED_H_STEPS"]) // 4,
        "min_num_conditional_frames": 5
        + int(os.environ["HUMAN2ROBOT_P2_EXPECTED_H_STEPS"]) // 4,
        "max_num_conditional_frames": 5
        + int(os.environ["HUMAN2ROBOT_P2_EXPECTED_H_STEPS"]) // 4,
        "tokenizer_chunk_duration": 29
        + int(os.environ["HUMAN2ROBOT_P2_EXPECTED_H_STEPS"]),
        "precision": "bfloat16",
        "num_duplicates_per_image": 4,
        "use_image_aug": True,
        "initialization_checkpoint_path": os.environ[
            "HUMAN2ROBOT_P2_EXPECTED_INIT_CKPT_PATH"
        ],
        "tokenizer_checkpoint_path": os.environ[
            "HUMAN2ROBOT_P2_EXPECTED_TOKENIZER_PATH"
        ],
        "pytorch_cuda_alloc_conf": os.environ[
            "HUMAN2ROBOT_P2_EXPECTED_PYTORCH_CUDA_ALLOC_CONF"
        ],
        "torch_nccl_trace_buffer_size": "65536",
        "torch_nccl_dump_on_timeout": "1",
        "torch_nccl_desync_debug": "1",
        "nccl_debug": "WARN",
        "nccl_debug_subsys": None,
        "slow_sample_seconds": "5",
    }
    if actual != expected:
        raise RuntimeError(f"Formal Human2Robot P2 runtime mismatch: actual={actual}, expected={expected}")
    optimization_actual = {
        "optimizer": str(config.optimizer.optim_type).lower(),
        "learning_rate": float(config.optimizer.lr),
        "weight_decay": float(config.optimizer.weight_decay),
        "betas": [float(value) for value in config.optimizer.betas],
        "load_training_state": bool(config.checkpoint.load_training_state),
        "load_ema_to_reg": bool(config.checkpoint.load_ema_to_reg),
    }
    optimization_expected = {
        "optimizer": "adamw",
        "learning_rate": 0.0001,
        "weight_decay": 0.1,
        "betas": [0.9, 0.999],
        "load_training_state": False,
        "load_ema_to_reg": True,
    }
    if optimization_actual != optimization_expected:
        raise RuntimeError(
            "Formal Human2Robot P2 optimization mismatch: "
            f"actual={optimization_actual}, expected={optimization_expected}"
        )

    if rank == 0:
        payload = {
            "schema_version": "human2robot-m5b-p2-runtime-binding-v3",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "cell_id": os.environ["HUMAN2ROBOT_P2_CELL_ID"],
            "experiment_id": os.environ["HUMAN2ROBOT_P2_EXPERIMENT_ID"],
            "variant_id": os.environ["HUMAN2ROBOT_P2_VARIANT_ID"],
            "method_id": os.environ["HUMAN2ROBOT_P2_METHOD_ID"],
            "protocol_file_sha256": os.environ["HUMAN2ROBOT_P2_PROTOCOL_SHA256"],
            "four_gpu_successor_sha256": os.environ[
                "HUMAN2ROBOT_P2_FOUR_GPU_SUCCESSOR_SHA256"
            ],
            "memory_successor_sha256": os.environ[
                "HUMAN2ROBOT_P2_MEMORY_SUCCESSOR_SHA256"
            ],
            "io_successor_sha256": os.environ["HUMAN2ROBOT_P2_IO_SUCCESSOR_SHA256"],
            "logging_successor_sha256": os.environ[
                "HUMAN2ROBOT_P2_LOGGING_SUCCESSOR_SHA256"
            ],
            "code_sha256": os.environ["HUMAN2ROBOT_P2_CODE_SHA256"],
            "actual": actual,
            "distributed": {
                "global_rank": int(rank),
                "data_parallel_rank": int(dp_rank),
                "tensor_model_parallel_world_size": int(
                    parallel_state.get_tensor_model_parallel_world_size()
                ),
                "pipeline_model_parallel_world_size": int(
                    parallel_state.get_pipeline_model_parallel_world_size()
                ),
                "context_parallel_world_size": int(
                    parallel_state.get_context_parallel_world_size()
                ),
            },
            "data_contract": {
                "split": str(dataset.split),
                "method_id": str(dataset.method_id),
                "experiment_id": str(dataset.experiment_id),
                "variant_id": str(dataset.variant_id),
                "H_steps": int(dataset.h_steps),
                "K_steps": int(dataset.k_steps),
                "window_stride": int(dataset.window_stride),
                "top_k": int(dataset.top_k),
                "pool_size": int(dataset.pool_size),
                "retrieval_modality": str(dataset.retrieval_modality),
                "time_view_id": str(dataset.time_view_id),
                "query_offset_view_steps": int(dataset.query_offset_view_steps),
                "target_representation": str(dataset.target_representation),
                "canonical_root": str(dataset.canonical_root),
                "main_view_path": str(dataset.main_view_path),
                "protocol_path": str(dataset.protocol_path),
                "supplement_path": str(dataset.supplement_path),
                "p1_pool_root": str(dataset.p1_pool_root),
                "statistics_path": str(dataset.statistics_path),
                "retrieval_index_path": str(dataset.retrieval_index_path),
            },
            "optimization": optimization_actual,
            "initialization_checkpoint_path": str(config.checkpoint.load_path),
            "job": {
                "project": str(config.job.project),
                "group": str(config.job.group),
                "name": str(config.job.name),
            },
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "visible_cuda_device_count": int(torch.cuda.device_count()),
                "gpu_name": torch.cuda.get_device_name(0),
                "offline_auto_download_disabled": os.environ.get("COSMOS_SKIP_HF_AUTO_DOWNLOAD")
                == "1",
                "huggingface_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
                "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
                "wandb_disabled": os.environ.get("WANDB_MODE") == "disabled",
                "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                "torch_nccl_trace_buffer_size": os.environ.get(
                    "TORCH_NCCL_TRACE_BUFFER_SIZE"
                ),
                "torch_nccl_dump_on_timeout": os.environ.get("TORCH_NCCL_DUMP_ON_TIMEOUT"),
                "torch_nccl_desync_debug": os.environ.get("TORCH_NCCL_DESYNC_DEBUG"),
                "nccl_debug": os.environ.get("NCCL_DEBUG"),
                "nccl_debug_subsys": os.environ.get("NCCL_DEBUG_SUBSYS"),
                "slow_sample_seconds": os.environ.get(
                    "HUMAN2ROBOT_P2_SLOW_SAMPLE_SECONDS"
                ),
            },
        }
        binding_path = Path(binding_path_value)
        binding_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = binding_path.with_suffix(binding_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_path, binding_path)
    torch.distributed.barrier()


@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    # Need to initialize the distributed environment before calling config.validate() because it tries to synchronize
    # a buffer across ranks. If you don't do this, then you end up allocating a bunch of buffers on rank 0, and also that
    # check doesn't actually do anything.
    with distributed_init():
        distributed.init()

    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    trainer = config.trainer.type(config)
    # ImaginaireTrainer initializes Megatron model/data-parallel groups in its
    # constructor.  Bind the formal runtime only after those groups exist, but
    # still before model construction, dataloading, or any optimizer step.
    _write_optional_human2robot_p2_runtime_binding(config)
    # Setup the miscellaneous stuff for reproducibility.
    log_reproducible_setup(config, args)

    with model_init():
        model = instantiate(config.model)

    # Create the dataloaders.
    with data_loader_init():
        # NOTE (user): We manually instantiate the dataloader instead of using instantiate(config.dataloader_train),
        # since it is difficult to set up the DistributedSampler without creating two duplicates of the dataset.
        # We intentionally instantiate the dataloader on every process (rather than the rank 0 process only) to work with the DistributedSampler.
        dataset = instantiate(config.dataloader_train.dataset)
        sampler_seed = int(getattr(config.dataloader_train.sampler, "seed", config.trainer.seed))
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=parallel_state.get_data_parallel_world_size(),
            rank=parallel_state.get_data_parallel_rank(),
            shuffle=True,
            seed=sampler_seed,
        )
        dataloader_train = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=config.dataloader_train.batch_size,
            drop_last=config.dataloader_train.drop_last,
            num_workers=config.dataloader_train.num_workers,
            persistent_workers=config.dataloader_train.persistent_workers,
            pin_memory=config.dataloader_train.pin_memory,
            pin_memory_device=config.dataloader_train.pin_memory_device,
            timeout=config.dataloader_train.timeout,
        )

        dataloader_val = None
        if config.trainer.run_validation:
            # NOTE (user): Manually instantiate the val dataloader as well
            dataset_val = instantiate(config.dataloader_val.dataset)
            sampler_val_seed = int(getattr(config.dataloader_val.sampler, "seed", config.trainer.seed))
            sampler_val = DistributedSampler(
                dataset=dataset_val,
                num_replicas=parallel_state.get_data_parallel_world_size(),
                rank=parallel_state.get_data_parallel_rank(),
                shuffle=False,  # Do not shuffle the validation set
                seed=sampler_val_seed,
            )
            dataloader_val = DataLoader(
                dataset=dataset_val,
                sampler=sampler_val,
                batch_size=config.dataloader_val.batch_size,
                drop_last=config.dataloader_val.drop_last,
                num_workers=config.dataloader_val.num_workers,
                persistent_workers=config.dataloader_val.persistent_workers,
                pin_memory=config.dataloader_val.pin_memory,
                pin_memory_device=config.dataloader_val.pin_memory_device,
                timeout=config.dataloader_val.timeout,
            )

    # Start training
    trainer.train(
        model,
        dataloader_train,
        dataloader_val,
    )


if __name__ == "__main__":
    # Usage: torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train --config=cosmos_policy/config/experiment/your_config.py

    # Get the config file from the input arguments.
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", help="Path to the config file", required=False)
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.opts, enable_one_logger=True)

    if args.dryrun:
        logging.info(
            "Config:\n" + config.pretty_print(use_color=True) + "\n" + pretty_print_overrides(args.opts, use_color=True)
        )
        os.makedirs(config.job.path_local, exist_ok=True)
        try:
            to_yaml(config, f"{config.job.path_local}/config.yaml")
        except Exception:
            logging.error("to_yaml failed, falling back to LazyConfig.save_yaml:")
            logging.error(f"Traceback: {traceback.format_exc()}")
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
    else:
        # Launch the training job.
        launch(config, args)
