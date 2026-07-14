from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.human2robot_m5b_p2_dag import (
    DagContractError,
    completed_artifact,
    inventory,
    run_registered_cell,
)
from tools.human2robot_m5b_p2_handlers import DEFAULT_ARTIFACT_ROOT
from tools.human2robot_m5b_p2_matrix import load_execution_matrix


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
