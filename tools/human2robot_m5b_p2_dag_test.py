from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.human2robot_m5b_p2_dag import (
    DagContractError,
    build_plan,
    completed_artifact,
    inventory,
    run_registered_cell,
)
from tools.human2robot_m5b_p2 import source_manifest, source_paths
from tools.human2robot_m5b_p2_handlers import DEFAULT_ARTIFACT_ROOT
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
    file_sha256,
    load_execution_matrix,
)


WORKSPACE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def matrix():
    return load_execution_matrix(WORKSPACE, verify_prepared_artifact_hashes=False)


def test_empty_inventory_is_203_missing_and_only_root_cells_ready(matrix, tmp_path: Path) -> None:
    result = inventory(matrix, tmp_path)
    assert result["counts"] == {"completed": 0, "missing": 203, "invalid": 0}
    assert len(result["ready_cell_ids"]) == 51
    assert all(not matrix.cells_by_id[cell_id].parent_artifact_ids for cell_id in result["ready_cell_ids"])
    assert result["formal_queue_allowed"] is False


def test_completed_artifact_rejects_nonformal_payload(tmp_path: Path) -> None:
    cell_id = "cell"
    path = tmp_path / "cells" / cell_id / "artifact.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"cell_id": cell_id, "status": "completed", "formal_result": False}),
        encoding="utf-8",
    )
    with pytest.raises(DagContractError, match="not formal"):
        completed_artifact(tmp_path, cell_id)


def test_plan_opens_only_for_activation_bound_to_current_source(matrix, tmp_path: Path) -> None:
    source = source_manifest(WORKSPACE, source_paths(WORKSPACE))
    snapshot = tmp_path / "source_snapshots" / source["code_sha256"] / "source_snapshot_manifest.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps({**source, "created_at_utc": "2026-07-14T00:00:00+00:00"}), encoding="utf-8")
    receipt = tmp_path / "docker_suite_receipt_v6.json"
    receipt.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    activation = {
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
        "candidate_code_sha256": source["code_sha256"],
        "source_snapshot_manifest_path": str(snapshot),
        "docker_suite_receipt_path": str(receipt),
        "docker_suite_receipt_sha256": file_sha256(receipt),
    }
    activation_path = tmp_path / "launch_activation_v6.json"
    activation_path.write_text(json.dumps(activation), encoding="utf-8")
    plan = build_plan(WORKSPACE, tmp_path, activation_path, matrix)
    assert plan["formal_queue_allowed"] is True
    assert plan["launch_activation"]["status"] == "approved"

    activation["candidate_code_sha256"] = "0" * 64
    activation_path.write_text(json.dumps(activation), encoding="utf-8")
    plan = build_plan(WORKSPACE, tmp_path, activation_path, matrix)
    assert plan["formal_queue_allowed"] is False
    assert "different candidate code" in plan["launch_activation"]["error"]


def test_single_cell_dispatch_fails_before_subprocess_while_matrix_blocked(matrix, tmp_path: Path) -> None:
    activation = {
        "schema_version": "human2robot-m5b-p2-formal-activation-v1",
        "status": "approved",
        "formal_queue_allowed": True,
        "registry_sha256": matrix.prepared_manifest["registry_file_sha256"],
        "solver_contract_resolved": True,
        "workspace_bounds_frozen": True,
        "all_147_evaluations_bound_to_completion_report": True,
        "docker_full_suite_passed": True,
        "source_snapshot_frozen": True,
        "gpu_count": 8,
        "storage_probe_passed": True,
    }
    activation_path = tmp_path / "activation.json"
    activation_path.write_text(json.dumps(activation), encoding="utf-8")
    root_cell = matrix.topological_cell_ids[0]
    with pytest.raises(DagContractError, match="Formal activation is incomplete"):
        run_registered_cell(
            WORKSPACE,
            Path(DEFAULT_ARTIFACT_ROOT),
            activation_path,
            root_cell,
        )
