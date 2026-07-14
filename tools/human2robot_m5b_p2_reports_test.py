from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.human2robot_m5b_p2_matrix import load_execution_matrix
from tools.human2robot_m5b_p2_reports import (
    ReportContractError,
    _inventory,
    artifact_path,
    build_registered_report,
    resolution_gate,
)


WORKSPACE = Path(__file__).resolve().parents[1]


def _write_parent(root: Path, cell_id: str, seed: int, method_offset: float) -> None:
    path = artifact_path(root, cell_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cell_id": cell_id,
                "status": "completed",
                "formal_result": True,
                "task_seed_records": [
                    {
                        "task": f"task_{task}",
                        "seed": seed,
                        "metrics": {"position_error_median_canonical": method_offset + task * 0.01},
                        "guardrails": {},
                    }
                    for task in range(4)
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_seed_report_binds_all_nine_parent_hashes_and_selects_cases(tmp_path: Path) -> None:
    matrix = load_execution_matrix(WORKSPACE, verify_prepared_artifact_hashes=False)
    cell_id = "aggregate_report__M5B-QUAL-01__seed_level_case_manifest__seed20260711"
    cell = matrix.cells_by_id[cell_id]
    for index, parent_id in enumerate(cell.parent_artifact_ids):
        _write_parent(tmp_path, parent_id, 20260711, float(index))
    report = build_registered_report(matrix, tmp_path, cell_id)
    assert report["status"] == "completed"
    assert report["formal_result"] is True
    assert len(report["parent_artifacts"]) == 9
    assert set(report["cases"]) == {"task_0", "task_1", "task_2", "task_3"}
    assert all(len(value["best"]) == len(value["worst"]) == 3 for value in report["cases"].values())


def test_registered_report_refuses_one_missing_parent(tmp_path: Path) -> None:
    matrix = load_execution_matrix(WORKSPACE, verify_prepared_artifact_hashes=False)
    cell_id = "aggregate_report__M5B-QUAL-01__seed_level_case_manifest__seed20260712"
    cell = matrix.cells_by_id[cell_id]
    for index, parent_id in enumerate(cell.parent_artifact_ids[:-1]):
        _write_parent(tmp_path, parent_id, 20260712, float(index))
    with pytest.raises(ReportContractError, match="Missing JSON"):
        build_registered_report(matrix, tmp_path, cell_id)


def test_full_inventory_never_imputes_missing_cells(tmp_path: Path) -> None:
    matrix = load_execution_matrix(WORKSPACE, verify_prepared_artifact_hashes=False)
    inventory = _inventory(matrix, tmp_path)
    assert inventory["expected_cell_count"] == 203
    assert inventory["completed_cell_count"] == 0
    assert len(inventory["missing_cell_ids"]) == 203
    assert inventory["all_203_complete"] is False


def test_resolution_gate_uses_mean_jaccard_and_reports_distribution() -> None:
    artifacts = {}
    for seed in (20260711, 20260712, 20260713):
        source_id = (
            "checkpoint_linked_evaluation__M5B-RES-01__source_240x426_then_resize_224__"
            f"recap_hand_ret__seed{seed}"
        )
        crop_id = (
            "checkpoint_linked_evaluation__M5B-RES-01__center_crop_240x424_then_resize_224__"
            f"recap_hand_ret__seed{seed}"
        )
        units_source = [
            {"task": f"task_{task}", "seed": seed, "metrics": {"position_error_median_canonical": 1.0}}
            for task in range(4)
        ]
        units_crop = [
            {"task": f"task_{task}", "seed": seed, "metrics": {"position_error_median_canonical": 1.04}}
            for task in range(4)
        ]
        left = {f"q{index}": ["a", "b", "c"] for index in range(5)}
        right = dict(left)
        if seed == 20260711:
            right["q0"] = ["a", "b", "d"]
        artifacts[source_id] = {"cell_id": source_id, "task_seed_records": units_source, "visual_topk_by_query": left}
        artifacts[crop_id] = {"cell_id": crop_id, "task_seed_records": units_crop, "visual_topk_by_query": right}
    result = resolution_gate(artifacts)
    assert result["status"] == "passed"
    assert result["mean_query_topk_jaccard"] >= 0.90
    assert result["minimum_query_topk_jaccard"] == 0.5
    assert result["median_query_topk_jaccard"] == 1.0
    assert result["identical_query_topk_ratio"] < 1.0
