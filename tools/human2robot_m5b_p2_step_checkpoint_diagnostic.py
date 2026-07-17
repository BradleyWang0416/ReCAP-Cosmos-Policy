#!/usr/bin/env python3
"""Explicitly non-formal inference diagnostic for an intermediate M5B-P2 DCP.

This module is intentionally outside the frozen formal DAG.  It never writes to
the formal artifact root and its output is ineligible for acceptance evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cosmos_policy.datasets.human2robot_p2_contract import canonical_window_metrics
from cosmos_policy.datasets.human2robot_p2_dataset import build_human2robot_p2_dataset
from cosmos_policy.models.human2robot_adapter import validate_human2robot_batch
from tools import human2robot_m5b_p2_inference as formal
from tools.human2robot_m5b_p2_evaluation import RankPrediction, evaluate_ranked_query
from tools.human2robot_m5b_p2_matrix import CellBinding, load_execution_matrix


FORMAL_PARENT_CELL_ID = (
    "learned_training_checkpoint__M5B-MAIN-01__frozen_main__"
    "no_retrieval__seed20260711"
)
FORMAL_EVALUATION_VARIANT = "main_comparison_pool10"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _diagnostic_binding(workspace: Path) -> CellBinding:
    matrix = load_execution_matrix(workspace)
    matches = [
        binding
        for binding in matrix.bindings_by_id.values()
        if binding.evaluation is not None
        and binding.evaluation.parent_artifact_id == FORMAL_PARENT_CELL_ID
        and binding.evaluation.variant_id == FORMAL_EVALUATION_VARIANT
        and binding.evaluation.method_id == "no_retrieval"
    ]
    _require(len(matches) == 1, f"Expected one evaluation binding, found {len(matches)}")
    return matches[0]


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    _require(length > 0, "Cannot select from an empty query list")
    _require(1 <= count <= length, f"Bad query count {count} for length {length}")
    if count == 1:
        return [length // 2]
    return sorted({int(round(value)) for value in np.linspace(0, length - 1, count)})


def _select_query_indices(
    dataset: Any,
    *,
    queries_per_task: int,
    all_queries: bool,
    shard_index: int,
    shard_count: int,
) -> tuple[list[int], list[dict[str, Any]], list[str]]:
    tasks = sorted({str(query.task) for query in dataset.queries})
    _require(tasks, "Held-out dataset has no tasks")
    _require(0 <= shard_index < shard_count, "Shard index is out of range")
    shard_tasks = [task for index, task in enumerate(tasks) if index % shard_count == shard_index]
    selected_indices: list[int] = []
    provenance: list[dict[str, Any]] = []
    for task in shard_tasks:
        candidates = [
            (index, query)
            for index, query in enumerate(dataset.queries)
            if str(query.task) == task
        ]
        candidates.sort(
            key=lambda pair: (
                str(pair[1].episode_id),
                int(pair[1].current_row),
                str(pair[1].window_id),
            )
        )
        selected_positions = (
            list(range(len(candidates)))
            if all_queries
            else _evenly_spaced_indices(len(candidates), queries_per_task)
        )
        for selected_position in selected_positions:
            query_index, query = candidates[selected_position]
            selected_indices.append(query_index)
            provenance.append(
                {
                    "task": task,
                    "candidate_count": len(candidates),
                    "selection_rule": (
                        "sorted_episode_current_row_window_id_all_queries"
                        if all_queries
                        else "sorted_episode_current_row_window_id_evenly_spaced"
                    ),
                    "selected_position": selected_position,
                    "query_index": query_index,
                    "query_id": str(query.window_id),
                    "episode_id": str(query.episode_id),
                    "current_row": int(query.current_row),
                }
            )
    _require(selected_indices, f"Shard {shard_index} selected no queries")
    return selected_indices, provenance, shard_tasks


class IntermediateCheckpointBackend:
    """Formal sampler/model path with only the expected iteration relaxed."""

    def __init__(
        self,
        workspace: Path,
        binding: CellBinding,
        checkpoint_path: Path,
        expected_iteration: int,
    ) -> None:
        evaluation = binding.evaluation
        _require(
            evaluation is not None and evaluation.checkpoint_cell_id is not None,
            "Diagnostic model backend requires a checkpoint-linked binding",
        )
        _require(checkpoint_path.is_dir(), f"Checkpoint directory missing: {checkpoint_path}")
        sampler = formal.validate_formal_sampler_signature()
        _require(sampler["status"] == "passed", str(sampler["reason"]))
        # Hydra's Python-config loader expects a package-relative module path.
        # The exact frozen source tree is selected by PYTHONPATH at launch.
        config_path = Path("cosmos_policy/config/config.py")
        options = [
            "--",
            f"experiment={binding.training_spec.config_name}",
            f"checkpoint.load_path={checkpoint_path}",
            "checkpoint.load_training_state=False",
            "checkpoint.load_ema_to_reg=True",
        ]
        config = formal.load_config(str(config_path), options, enable_one_logger=True)
        config.checkpoint.load_path = str(checkpoint_path)
        config.checkpoint.load_training_state = False
        config.checkpoint.load_ema_to_reg = True
        with formal.distributed_init():
            formal.distributed.init()
        config.validate()
        self.trainer = config.trainer.type(config)
        with formal.model_init():
            model = formal.instantiate(config.model)
        model = model.to("cuda", memory_format=config.trainer.memory_format)
        model.on_train_start(config.trainer.memory_format)
        _require(float(model.config.shift) == formal.FORMAL_SHIFT, "Resolved model shift changed")
        _require(
            bool(model.config.use_kerras_sigma_at_inference)
            is formal.FORMAL_USE_KERRAS_SIGMA,
            "Resolved Karras-sigma setting changed",
        )
        optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
        checkpointer_iteration = self.trainer.checkpointer.load(
            model, optimizer, scheduler, grad_scaler
        )
        expected_dirname = f"iter_{expected_iteration:09d}"
        _require(
            checkpoint_path.name == expected_dirname,
            f"Checkpoint dirname is {checkpoint_path.name}, expected {expected_dirname}",
        )
        _require(
            int(checkpointer_iteration) in {0, expected_iteration},
            "Checkpointer returned an unexpected iteration "
            f"{checkpointer_iteration} for {checkpoint_path}",
        )
        model.eval()
        self.model = model
        self.k_steps = int(evaluation.k_steps)
        self.loaded_iteration = int(expected_iteration)
        self.checkpointer_reported_iteration = int(checkpointer_iteration)

    @torch.inference_mode()
    def __call__(self, item: Mapping[str, Any], seed: int) -> np.ndarray:
        batch = formal.misc.to(formal.default_collate([item]), device="cuda")
        # The shared sampler normalizes uint8 video before dispatching to the
        # Human2Robot subclass hook.  Validate the strict adapter contract before
        # that normalization, then use the immediate parent implementations for
        # the already-normalized internal sampler calls.
        validate_human2robot_batch(batch)
        parent = super(type(self.model), self.model)
        originals = {
            "get_data_and_condition": self.model.get_data_and_condition,
            "get_velocity_fn_from_batch": self.model.get_velocity_fn_from_batch,
        }
        self.model.get_data_and_condition = parent.get_data_and_condition
        self.model.get_velocity_fn_from_batch = parent.get_velocity_fn_from_batch
        try:
            # The DCP model parameters are BF16 while the rectified-flow sampler
            # deliberately initializes its noise/state in FP32.  Match the
            # mixed-precision execution contract used by model inference so
            # linear/attention inputs are cast to the checkpoint parameter dtype.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                generated = self.model.generate_samples_from_batch(
                    batch,
                    n_sample=1,
                    guidance=formal.FORMAL_GUIDANCE,
                    num_steps=formal.FORMAL_NUM_STEPS,
                    shift=formal.FORMAL_SHIFT,
                    seed=seed,
                    is_negative_prompt=False,
                    use_variance_scale=formal.FORMAL_VARIANCE_SCALE,
                )
        finally:
            for name, method in originals.items():
                setattr(self.model, name, method)
        direct = getattr(self.model, "_generated_action", None)
        if direct is not None:
            actions = direct.reshape(1, self.k_steps, 10)
            self.model._generated_action = None
        else:
            actions = formal.extract_action_chunk_from_latent_sequence(
                generated,
                action_shape=(self.k_steps, 10),
                action_indices=batch["action_latent_idx"],
                decoder=getattr(self.model, "action_decoder", None),
            )
        return actions[0].to(torch.float32).cpu().numpy()


def _metric_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted({name for record in records for name in record["metrics"]})
    return {
        name: {
            "mean": float(np.mean([float(record["metrics"][name]) for record in records])),
            "median": float(np.median([float(record["metrics"][name]) for record in records])),
        }
        for name in names
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    workspace = args.workspace.resolve()
    checkpoint_path = args.checkpoint.resolve()
    binding = _diagnostic_binding(workspace)
    evaluation = binding.evaluation
    _require(evaluation is not None, "Evaluation binding is missing")
    kwargs = formal.dataset_kwargs(workspace, binding)
    dataset = build_human2robot_p2_dataset(**kwargs)
    selected_query_indices, selection, shard_tasks = _select_query_indices(
        dataset,
        queries_per_task=args.queries_per_task,
        all_queries=args.all_queries,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    if args.all_queries and args.shard_count == 1:
        _require(
            len(selected_query_indices) == len(dataset.queries),
            "All-query mode did not select the complete held-out query set",
        )
    selected_query_index_set = set(selected_query_indices)
    selected_example_indices = [
        index
        for index, example in enumerate(dataset.examples)
        if int(example.query_index) in selected_query_index_set
    ]
    expected_examples = len(selected_query_indices) * int(evaluation.top_k)
    _require(
        len(selected_example_indices) == expected_examples,
        f"Expected {expected_examples} rank examples, found {len(selected_example_indices)}",
    )
    statistics = formal.read_json(Path(kwargs["statistics_path"]))
    bounds = formal.read_json(args.workspace_bounds.resolve())
    _require(bounds.get("status") == "frozen", "Workspace bounds are not frozen")

    backend_started = time.time()
    backend = IntermediateCheckpointBackend(
        workspace,
        binding,
        checkpoint_path,
        args.expected_iteration,
    )
    backend_load_seconds = time.time() - backend_started
    grouped: dict[str, list[RankPrediction]] = defaultdict(list)
    baselines: dict[str, dict[str, float]] = {}
    inference_receipts: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    for completed, dataset_index in enumerate(selected_example_indices, start=1):
        item = dataset[dataset_index]
        query_id = str(item["query_id"])
        inference_seed = formal.deterministic_inference_seed(
            evaluation.run_seed,
            evaluation.experiment_id,
            evaluation.variant_id,
            str(item["task"]),
            str(item["episode_id"]),
            int(item["current_row"]),
            int(item["retrieval_rank"]),
        )
        torch.cuda.synchronize()
        inference_started = time.time()
        prediction = backend(item, inference_seed)
        torch.cuda.synchronize()
        inference_seconds = time.time() - inference_started
        inference_receipts.append(
            {
                "completed": completed,
                "total": len(selected_example_indices),
                "dataset_index": dataset_index,
                "query_id": query_id,
                "task": str(item["task"]),
                "retrieval_rank": int(item["retrieval_rank"]),
                "inference_seed": int(inference_seed),
                "inference_seconds": inference_seconds,
            }
        )
        print(
            "[M5B-P2-NONFORMAL] "
            + json.dumps(inference_receipts[-1], sort_keys=True),
            flush=True,
        )
        grouped[query_id].append(
            RankPrediction(
                query_id=query_id,
                task=str(item["task"]),
                episode_id=str(item["episode_id"]),
                current_row=int(item["current_row"]),
                retrieval_rank=int(item["retrieval_rank"]),
                target_representation=str(item["target_representation"]),
                normalized_prediction=prediction,
                current_state_10d=np.asarray(item["raw_current_state"]),
                aligned_pool_10d=np.asarray(item["raw_aligned_pool"]),
                query_target_10d=np.asarray(item["raw_query_target"]),
                gap_crossing_count=int(item["gap_crossing_count"]),
                heldout_target_retrieval_feature_count=int(
                    item["heldout_target_retrieval_feature_count"]
                ),
            )
        )
        if query_id not in baselines:
            target = np.asarray(item["raw_query_target"], dtype=np.float64)
            current = np.asarray(item["raw_current_state"], dtype=np.float64)
            no_motion = np.repeat(current.reshape(1, 10), len(target), axis=0)
            baselines[query_id] = dict(canonical_window_metrics(no_motion, target))

    window_records = [
        evaluate_ranked_query(
            grouped[query_id],
            statistics=statistics,
            workspace_xyz_min=bounds["xyz_min"],
            workspace_xyz_max=bounds["xyz_max"],
        )
        for query_id in sorted(grouped)
    ]
    for record in window_records:
        record["diagnostic_no_motion_baseline_metrics"] = baselines[record["query_id"]]
    baseline_records = [
        {"metrics": record["diagnostic_no_motion_baseline_metrics"]}
        for record in window_records
    ]
    device = torch.cuda.get_device_properties(0)
    payload: dict[str, Any] = {
        "schema_version": "human2robot-m5b-p2-intermediate-diagnostic-v1",
        "status": "completed",
        "formal_result": False,
        "acceptance_eligible": False,
        "claim_boundary": (
            "Complete held-out step-checkpoint diagnostic, repeatedly inspected for training "
            "monitoring only; not a formal evaluation, checkpoint-selection result, or "
            "evidence for M5B-P2 acceptance."
            if args.all_queries
            else "Small representative step-checkpoint diagnostic only; not a formal "
            "evaluation, not a baseline comparison, and not evidence for M5B-P2 acceptance."
        ),
        "method_id": evaluation.method_id,
        "run_seed": evaluation.run_seed,
        "checkpoint": {
            "path": str(checkpoint_path),
            "expected_iteration": args.expected_iteration,
            "loaded_iteration": backend.loaded_iteration,
            "checkpointer_reported_iteration": backend.checkpointer_reported_iteration,
            "training_state_loaded": False,
            "formal_parent_cell_id": FORMAL_PARENT_CELL_ID,
        },
        "binding": asdict(evaluation),
        "selection": {
            "split": "heldout",
            "all_queries": args.all_queries,
            "queries_per_task": None if args.all_queries else args.queries_per_task,
            "diagnostic_scope": (
                "full_heldout_query_set" if args.all_queries else "representative_query_subset"
            ),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "shard_tasks": shard_tasks,
            "selected_query_count": len(selected_query_indices),
            "selected_rank_example_count": len(selected_example_indices),
            "query_provenance": selection,
        },
        "sampler": {
            "guidance": formal.FORMAL_GUIDANCE,
            "num_sampler_calls": formal.FORMAL_NUM_STEPS,
            "shift": formal.FORMAL_SHIFT,
            "use_karras_sigma": formal.FORMAL_USE_KERRAS_SIGMA,
            "use_variance_scale": formal.FORMAL_VARIANCE_SCALE,
            "autocast_dtype": "bfloat16",
            "adapter_validation_order": "strict_uint8_pre_normalization",
            "normalized_internal_dispatch": "immediate_parent_methods",
        },
        "metric_summary": _metric_summary(window_records),
        "diagnostic_no_motion_baseline_summary": _metric_summary(baseline_records),
        "window_records": window_records,
        "inference_receipts": inference_receipts,
        "runtime": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": device.name,
            "gpu_total_memory_bytes": int(device.total_memory),
            "backend_load_seconds": backend_load_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "total_wall_seconds": time.time() - started,
        },
        "guardrail_totals": {
            name: int(sum(int(record["guardrails"][name]) for record in window_records))
            for name in sorted(window_records[0]["guardrails"])
        },
        "completed_at_unix": time.time(),
    }
    _write_json_atomic(args.output.resolve(), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-iteration", type=int, default=1000)
    parser.add_argument("--workspace-bounds", type=Path, required=True)
    parser.add_argument("--queries-per-task", type=int, default=1)
    parser.add_argument("--all-queries", action="store_true")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = run(args)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    print(
        json.dumps(
            {
                "status": payload["status"],
                "formal_result": payload["formal_result"],
                "output": str(args.output.resolve()),
                "selection": payload["selection"],
                "metric_summary": payload["metric_summary"],
                "guardrail_totals": payload["guardrail_totals"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
