#!/usr/bin/env python3
"""Build commands for every frozen M5B-P2 DAG cell without launching them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from tools.human2robot_m5b_p2_matrix import (
    FOUR_GPU_BATCH_PER_DP_RANK,
    FOUR_GPU_DP_WORLD_SIZE,
    FOUR_GPU_EFFECTIVE_GLOBAL_BATCH_SIZE,
    FOUR_GPU_FSDP_SHARD_SIZE,
    FOUR_GPU_GRADIENT_ACCUMULATION_STEPS,
    FOUR_GPU_SUCCESSOR_SHA256,
    FOUR_GPU_WORLD_SIZE,
    IO_DIAGNOSTIC_ENV,
    IO_SUCCESSOR_SHA256,
    LAG_VIEW_MANIFEST_SHA256,
    MEMORY_SUCCESSOR_SHA256,
    LOGGING_SUCCESSOR_SHA256,
    PREPARED_MANIFEST_SHA256,
    PYTORCH_CUDA_ALLOC_CONF,
    SUPPLEMENT_SHA256,
    WORKSPACE_BOUNDS_SHA256,
    CellBinding,
    ExecutionMatrix,
    load_execution_matrix,
)

DEFAULT_ARTIFACT_ROOT = "/DATA1/wxs/ReCAP_M5B_P2_RUNS"
FORMAL_OFFLINE_ENV = {
    "COSMOS_SKIP_HF_AUTO_DOWNLOAD": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF,
    **IO_DIAGNOSTIC_ENV,
}


class HandlerContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class HandlerPlan:
    cell_id: str
    artifact_kind: str
    handler_kind: str
    parent_artifact_ids: tuple[str, ...]
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    gpu_count: int
    can_execute_before_formal_activation: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HandlerContractError(message)


def _command(binding: CellBinding) -> tuple[str, ...]:
    cell = binding.cell
    if cell.artifact_kind == "learned_training_checkpoint":
        _require(binding.training_spec is not None, f"Training spec missing: {cell.cell_id}")
        return (
            ".venv/bin/python",
            "-m",
            "tools.human2robot_m5b_p2",
            "--workspace",
            "/workspace",
            "--output-root",
            DEFAULT_ARTIFACT_ROOT,
            "run-cell",
            cell.cell_id,
            "--activation-path",
            f"{DEFAULT_ARTIFACT_ROOT}/launch_activation_v6.json",
        )
    if cell.artifact_kind in {"nonlearned_method_artifact", "checkpoint_linked_evaluation"}:
        return (
            ".venv/bin/python",
            "-m",
            "tools.human2robot_m5b_p2_inference",
            "--workspace",
            "/workspace",
            "--artifact-root",
            DEFAULT_ARTIFACT_ROOT,
            "run-cell",
            "--cell-id",
            cell.cell_id,
            "--activation-path",
            f"{DEFAULT_ARTIFACT_ROOT}/launch_activation_v6.json",
            "--workspace-bounds-path",
            "/workspace/方案/v03/M5B_P2_workspace_bounds_v1.json",
        )
    if cell.artifact_kind == "aggregate_report":
        return (
            ".venv/bin/python",
            "-m",
            "tools.human2robot_m5b_p2_reports",
            "--workspace",
            "/workspace",
            "--artifact-root",
            DEFAULT_ARTIFACT_ROOT,
            "build-cell",
            "--cell-id",
            cell.cell_id,
        )
    raise HandlerContractError(f"No command builder for {cell.cell_id}")


def build_handler_plans(matrix: ExecutionMatrix) -> dict[str, HandlerPlan]:
    result: dict[str, HandlerPlan] = {}
    for cell_id in matrix.topological_cell_ids:
        binding = matrix.bindings_by_id[cell_id]
        artifact_kind = binding.cell.artifact_kind
        gpu_count = FOUR_GPU_WORLD_SIZE if artifact_kind == "learned_training_checkpoint" else (
            0 if artifact_kind == "aggregate_report" or binding.cell.method_id == "retrieval_only" else 1
        )
        result[cell_id] = HandlerPlan(
            cell_id=cell_id,
            artifact_kind=artifact_kind,
            handler_kind=binding.handler_kind,
            parent_artifact_ids=binding.cell.parent_artifact_ids,
            command=_command(binding),
            environment=tuple(sorted(FORMAL_OFFLINE_ENV.items())),
            gpu_count=gpu_count,
            can_execute_before_formal_activation=False,
        )
    _require(len(result) == 203, "Handler plan does not cover all 203 cells")
    return result


def handler_coverage_manifest(matrix: ExecutionMatrix) -> dict[str, object]:
    plans = build_handler_plans(matrix)
    counts: dict[str, int] = {}
    for plan in plans.values():
        counts[plan.artifact_kind] = counts.get(plan.artifact_kind, 0) + 1
    return {
        "schema_version": "human2robot-m5b-p2-handler-coverage-v1",
        "cell_count": len(plans),
        "counts": counts,
        "all_cells_have_handlers": len(plans) == len(matrix.bindings_by_id) == 203,
        "formal_queue_open": False,
        "formal_readiness_blockers": list(matrix.formal_readiness_blockers),
        "plans": {
            cell_id: {
                "artifact_kind": plan.artifact_kind,
                "handler_kind": plan.handler_kind,
                "parent_artifact_ids": list(plan.parent_artifact_ids),
                "command": list(plan.command),
                "environment": dict(plan.environment),
                "gpu_count": plan.gpu_count,
                "can_execute_before_formal_activation": plan.can_execute_before_formal_activation,
            }
            for cell_id, plan in plans.items()
        },
    }


def require_formal_activation(
    activation: Mapping[str, object],
    matrix: ExecutionMatrix,
) -> None:
    expected = {
        "schema_version": "human2robot-m5b-p2-launch-activation-v6",
        "status": "approved",
        "launch_authorized": True,
        "formal_queue_allowed": True,
        "p2_acceptance_allowed": False,
        "registry_sha256": matrix.prepared_manifest["registry_file_sha256"],
        "supplement_sha256": SUPPLEMENT_SHA256,
        "four_gpu_successor_sha256": FOUR_GPU_SUCCESSOR_SHA256,
        "memory_successor_sha256": MEMORY_SUCCESSOR_SHA256,
        "io_successor_sha256": IO_SUCCESSOR_SHA256,
        "logging_successor_sha256": LOGGING_SUCCESSOR_SHA256,
        "indexed_hdf5_image_reads": True,
        "diagnostic_environment": IO_DIAGNOSTIC_ENV,
        "pytorch_cuda_alloc_conf": PYTORCH_CUDA_ALLOC_CONF,
        "prepared_manifest_sha256": PREPARED_MANIFEST_SHA256,
        "workspace_bounds_sha256": WORKSPACE_BOUNDS_SHA256,
        "lag_view_manifest_sha256": LAG_VIEW_MANIFEST_SHA256,
        "native_rectified_flow_contract_resolved": True,
        "all_147_evaluations_bound_to_terminal_report": True,
        "docker_full_suite_passed": True,
        "source_snapshot_frozen": True,
        "gpu_count": FOUR_GPU_WORLD_SIZE,
        "world_size": FOUR_GPU_WORLD_SIZE,
        "data_parallel_world_size": FOUR_GPU_DP_WORLD_SIZE,
        "fsdp_shard_size": FOUR_GPU_FSDP_SHARD_SIZE,
        "batch_size_per_data_parallel_rank": FOUR_GPU_BATCH_PER_DP_RANK,
        "gradient_accumulation_steps": FOUR_GPU_GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch_size": FOUR_GPU_EFFECTIVE_GLOBAL_BATCH_SIZE,
        "storage_probe_passed": True,
        "formal_output_mount_writable": True,
        "local_weight_hashes_passed": True,
    }
    mismatches = {
        key: {"actual": activation.get(key), "expected": value}
        for key, value in expected.items()
        if activation.get(key) != value
    }
    _require(not mismatches, f"Formal activation is incomplete: {mismatches}")
    _require(not matrix.formal_readiness_blockers, f"Matrix blockers remain: {matrix.formal_readiness_blockers}")


def require_final_acceptance(
    acceptance: Mapping[str, object],
    matrix: ExecutionMatrix,
    terminal_artifact: Mapping[str, object],
    *,
    terminal_report_sha256: str,
) -> None:
    expected = {
        "schema_version": "human2robot-m5b-p2-final-acceptance-v6",
        "status": "passed",
        "formal_queue_allowed": True,
        "p2_acceptance_allowed": True,
        "terminal_cell_id": "aggregate_report__M5B-QUAL-01__full_matrix_completion_acceptance",
        "terminal_report_status": "passed",
        "terminal_report_sha256": terminal_report_sha256,
        "completed_cell_count": 203,
        "registry_sha256": matrix.prepared_manifest["registry_file_sha256"],
        "supplement_sha256": SUPPLEMENT_SHA256,
        "four_gpu_successor_sha256": FOUR_GPU_SUCCESSOR_SHA256,
        "memory_successor_sha256": MEMORY_SUCCESSOR_SHA256,
        "io_successor_sha256": IO_SUCCESSOR_SHA256,
        "logging_successor_sha256": LOGGING_SUCCESSOR_SHA256,
        "indexed_hdf5_image_reads": True,
        "diagnostic_environment": IO_DIAGNOSTIC_ENV,
        "pytorch_cuda_alloc_conf": PYTORCH_CUDA_ALLOC_CONF,
        "world_size": FOUR_GPU_WORLD_SIZE,
        "data_parallel_world_size": FOUR_GPU_DP_WORLD_SIZE,
        "fsdp_shard_size": FOUR_GPU_FSDP_SHARD_SIZE,
        "batch_size_per_data_parallel_rank": FOUR_GPU_BATCH_PER_DP_RANK,
        "gradient_accumulation_steps": FOUR_GPU_GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch_size": FOUR_GPU_EFFECTIVE_GLOBAL_BATCH_SIZE,
        "prepared_manifest_sha256": PREPARED_MANIFEST_SHA256,
        "workspace_bounds_sha256": WORKSPACE_BOUNDS_SHA256,
        "lag_view_manifest_sha256": LAG_VIEW_MANIFEST_SHA256,
    }
    mismatches = {
        key: {"actual": acceptance.get(key), "expected": value}
        for key, value in expected.items()
        if acceptance.get(key) != value
    }
    _require(not mismatches, f"Final acceptance is incomplete: {mismatches}")
    _require(terminal_artifact.get("cell_id") == expected["terminal_cell_id"], "Terminal cell mismatch")
    _require(terminal_artifact.get("status") == "completed", "Terminal report is incomplete")
    _require(terminal_artifact.get("acceptance_status") == "passed", "Terminal report did not pass")


if __name__ == "__main__":
    print(json.dumps(handler_coverage_manifest(load_execution_matrix()), indent=2, sort_keys=True))
