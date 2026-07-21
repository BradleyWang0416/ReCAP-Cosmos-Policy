from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import human2robot_v04 as v04


def test_atomic_json_round_trip_and_no_partial(tmp_path: Path) -> None:
    output = tmp_path / "nested/receipt.json"
    payload = {"status": "passed", "unicode": "冻结", "values": [1, 2, 3]}
    v04.write_json_atomic(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.rglob("*.partial"))


def test_active_process_scan_ignores_unrelated_processes() -> None:
    rows = v04.active_v03_processes()
    assert all(any(token in row["command"] for token in v04.FORBIDDEN_PROCESS_TOKENS) for row in rows)


def test_mount_binding_has_auditable_shape(tmp_path: Path) -> None:
    binding = v04.mount_binding(tmp_path)
    assert binding["status"] in {"passed", "failed"}
    assert binding["path"] == str(tmp_path.resolve())


def test_preflight_is_blocked_without_formal_bindings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HUMAN2ROBOT_V04_IMAGE", raising=False)
    monkeypatch.delenv("HUMAN2ROBOT_V04_IMAGE_ID", raising=False)
    monkeypatch.setattr(v04, "RUN_ROOT", tmp_path / "runs")
    monkeypatch.setattr(v04, "verify_v03", lambda _: {"status": "blocked", "blockers": ["test"]})
    monkeypatch.setattr(v04, "_asset_checks", lambda _: [{"status": "failed"}])
    monkeypatch.setattr(v04, "_gpu_probe", lambda: {"status": "failed", "host_physical_indices": []})
    monkeypatch.setattr(v04, "_import_probe", lambda name: {"module": name, "status": "passed"})
    monkeypatch.setattr(v04, "bind_file", lambda path: {"path": str(path), "size_bytes": 1, "sha256": "0" * 64})
    monkeypatch.setattr(v04.shutil, "disk_usage", lambda _: v04.shutil._ntuple_diskusage(1024, 0, 1024))
    receipt = v04.build_preflight(tmp_path)
    assert receipt["status"] == "BLOCKED_ENVIRONMENT"
    assert receipt["formal_v04_allowed"] is False
    assert "docker_image_identity_not_bound" in receipt["blockers"]


def test_stage1_commands_default_to_dry_run() -> None:
    args = v04.build_parser().parse_args(["prepare-data"])
    assert args.execute is False
    assert args.source_root == Path("/DATA1/wxs/DATASETS/Human2Robot/data/v1")
    assert args.derived_root.name == "v04"


def test_replacement_assessment_parser_accepts_multiple_candidates() -> None:
    args = v04.build_parser().parse_args(
        [
            "assess-heldout-replacement",
            "--candidate-task",
            "push_plate_v1",
            "--candidate-task",
            "push_box_two_v1",
        ]
    )
    assert args.execute is False
    assert args.candidate_task == ["push_plate_v1", "push_box_two_v1"]
    assert args.replaced_heldout_task == "grab_pencil1_v1"
