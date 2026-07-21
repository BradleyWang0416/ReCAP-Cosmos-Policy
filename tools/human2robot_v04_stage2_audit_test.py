from __future__ import annotations

from pathlib import Path

import pytest

from tools import human2robot_v04_retrieval as stage2


def test_stage2_command_defaults_to_dry_run() -> None:
    args = stage2.build_parser().parse_args([])
    assert args.execute is False
    result = stage2.audit_retrieval_contract(
        workspace=Path("/workspace"),
        derived_root=Path("/workspace/data/Human2Robot/derived/v04"),
        execute=False,
    )
    assert result["status"] == "DRY_RUN"
    assert result["training_allowed"] is False
    assert "history/current-only feature provenance" in result["planned_checks"]


def test_immutable_stage2_artifact_cannot_be_replaced(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    stage2.write_json_atomic(path, {"status": "PASSED"}, immutable=True)
    with pytest.raises(stage2.Stage2AuditError, match="Refusing to replace"):
        stage2.write_json_atomic(path, {"status": "FAILED"}, immutable=True)
