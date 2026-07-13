from __future__ import annotations

from pathlib import Path

from tools.human2robot_m5b_p2_freeze_registry import (
    EXPECTED_COUNTS,
    STATUS,
    SUPPLEMENT_SHA256,
    build_frozen_registry,
)


def test_frozen_registry_binds_approved_supplement_and_all_cells() -> None:
    workspace = Path(__file__).resolve().parents[1]
    registry = build_frozen_registry(workspace)
    assert registry["status"] == STATUS
    assert registry["supplement_file_sha256"] == SUPPLEMENT_SHA256
    assert registry["counts"] == EXPECTED_COUNTS
    assert registry["cell_count"] == 202
    assert registry["formal_queue_allowed"] is False
    assert registry["p2_acceptance_allowed"] is False
    assert all(cell["status"] == "pending" for cell in registry["cells"])
    assert all(cell["formal_result"] is False for cell in registry["cells"])


def test_frozen_registry_cells_and_parent_edges_are_closed() -> None:
    workspace = Path(__file__).resolve().parents[1]
    registry = build_frozen_registry(workspace)
    cell_ids = {cell["cell_id"] for cell in registry["cells"]}
    assert len(cell_ids) == registry["cell_count"]
    assert all(
        parent_id in cell_ids
        for cell in registry["cells"]
        for parent_id in cell["parent_artifact_ids"]
    )
