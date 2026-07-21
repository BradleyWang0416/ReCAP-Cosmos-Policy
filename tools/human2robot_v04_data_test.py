from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from tools import human2robot_v04_data as stage1


def _write_source(path: Path, *, frames: int = 32, gap_at: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    step = np.arange(frames, dtype=np.int64)
    if gap_at is not None:
        step[gap_at:] += 1
    timestamp = np.arange(frames, dtype=np.int64) // 3
    with h5py.File(path, "w") as file:
        camera = file.create_group("cam_data")
        base = np.arange(frames * 2 * 3 * 3, dtype=np.uint8).reshape(frames, 2, 3, 3)
        camera.create_dataset("human_camera", data=base)
        camera.create_dataset("robot_camera", data=np.flip(base, axis=2))
        action = np.zeros((frames, 7), dtype=np.float32)
        action[:, 6] = np.arange(frames) % 2
        file.create_dataset("action", data=action)
        file.create_dataset("transformed_hand_coords", data=np.zeros((frames, 24, 3), dtype=np.float32))
        file.create_dataset("transformed_hand_frames", data=np.zeros((frames, 4, 3), dtype=np.float32))
        file.create_dataset("end_position", data=np.zeros((frames, 6), dtype=np.float32))
        file.create_dataset("gripper_state", data=(np.arange(frames) % 2).astype(np.float32))
        file.create_dataset("qpos", data=np.zeros((frames, 7), dtype=np.float32))
        file.create_dataset("qvel", data=np.zeros((frames, 7), dtype=np.float32))
        file.create_dataset("step", data=step)
        file.create_dataset("timestamp", data=timestamp)


def _record(source: Path, source_root: Path, *, partition: str, role: str, rank: int = 1) -> dict[str, object]:
    summary = stage1.inspect_source(source)
    return stage1._identity_record(
        source_root=source_root,
        path=source,
        source_sha=stage1.file_sha256(source),
        partition=partition,
        task=source.parent.name,
        role=role,
        summary=summary,
        partition_rank=rank,
    )


def test_source_rank_is_byte_deterministic() -> None:
    expected = stage1.source_rank("grab_cube2_v1/episode_12.hdf5")
    assert expected == stage1.source_rank("grab_cube2_v1/episode_12.hdf5")
    assert expected != stage1.source_rank("grab_cube2_v1/episode_13.hdf5")


def test_inspection_requires_gap_safe_H8_K8_windows(tmp_path: Path) -> None:
    valid = tmp_path / "task/episode_0.hdf5"
    _write_source(valid, frames=32, gap_at=16)
    summary = stage1.inspect_source(valid)
    assert summary["gap_count"] == 1
    assert summary["legal_window_count"] == 2

    invalid = tmp_path / "task/episode_1.hdf5"
    _write_source(invalid, frames=30, gap_at=15)
    with pytest.raises(stage1.Stage1DataError, match="No legal gap-safe"):
        stage1.inspect_source(invalid)


@pytest.mark.parametrize(
    ("partition", "role", "expected", "forbidden"),
    [
        ("v04_human_pool", "human", stage1.HUMAN_PROJECTION_DATASETS, "data/demo_0/robot/images"),
        ("v04_robot_dev", "robot", stage1.ROBOT_PROJECTION_DATASETS, "data/demo_0/human/images"),
    ],
)
def test_role_projection_enforces_physical_dataset_allowlist(
    tmp_path: Path,
    partition: str,
    role: str,
    expected: tuple[str, ...],
    forbidden: str,
) -> None:
    source_root = tmp_path / "raw"
    source = source_root / "task/episode_0.hdf5"
    _write_source(source)
    record = _record(source, source_root, partition=partition, role=role)
    projection = stage1.materialize_projection(source_root, tmp_path / "derived", record, "a" * 64)
    assert projection["dataset_paths"] == sorted(expected)
    assert forbidden not in projection["dataset_paths"]
    with h5py.File(projection["path"], "r") as file:
        demo = file["data/demo_0"]
        assert demo.attrs["contains_opposite_role_fields"] == np.bool_(False)
        assert demo.attrs["source_sha256"] == record["source_sha256"]


def _invariant_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    counter = 0

    def append(task: str, partition: str, role: str, rank: int) -> None:
        nonlocal counter
        source_sha = f"{counter:064x}"
        record: dict[str, object] = {
            "source_relative_path": f"{task}/episode_{counter}.hdf5",
            "source_sha256": source_sha,
            "source_partition": partition,
            "task": task,
            "episode_id": f"{task}/episode_{counter}",
            "role": role,
            "partition_rank": rank,
            "legal_window_count": 1,
            "window_identity": {"format": "test", "inherits": []},
        }
        if role in ("human", "paired"):
            record["human_content_sha256"] = f"h{source_sha[1:]}"
        if role in ("robot", "paired"):
            record["robot_content_sha256"] = f"r{source_sha[1:]}"
        records.append(record)
        counter += 1

    for task in stage1.SEEN_TASKS:
        for rank in range(1, 5):
            append(task, "seen_train", "paired", rank)
        append(task, "seen_validation", "paired", 1)
    for task in stage1.HELDOUT_TASKS:
        for partition, role, count in (
            ("legacy_quarantine", "paired", stage1.LEGACY_QUARANTINE_COUNTS[task]),
            ("v04_human_pool", "human", 10),
            ("v04_robot_dev", "robot", 5),
            ("v04_robot_final", "robot", 20),
            ("reserve", "paired", 1),
        ):
            for rank in range(1, count + 1):
                append(task, partition, role, rank)
    return records


def test_partition_invariants_are_pairwise_disjoint_and_exact() -> None:
    validation = stage1.validate_invariants(_invariant_records(), [])
    assert validation["pairwise_source_sha256_overlap_count"] == 0
    assert validation["partition_counts"]["v04_human_pool"] == 40
    assert validation["partition_counts"]["v04_robot_dev"] == 20
    assert validation["partition_counts"]["v04_robot_final"] == 80
    assert validation["partition_counts"]["legacy_quarantine"] == 30


def test_partition_invariants_allow_same_partition_duplicate_but_reject_cross_partition_source_sha() -> None:
    records = _invariant_records()
    records[1]["source_sha256"] = records[0]["source_sha256"]
    validation = stage1.validate_invariants(records, [])
    assert validation["within_partition_duplicate_sha256_count"]["seen_train"] == 1
    records[-1]["source_sha256"] = records[0]["source_sha256"]
    with pytest.raises(stage1.Stage1DataError, match="SHA overlap"):
        stage1.validate_invariants(records, [])


def test_rejected_sources_are_bound_into_raw_inventory_but_not_split() -> None:
    records = _invariant_records()
    records[0]["projection"] = {"path": "/derived/example.hdf5", "sha256": "f" * 64}
    rejected = [
        {
            "source_relative_path": "seen/rejected.hdf5",
            "source_sha256": "e" * 64,
            "task": "seen",
            "rejection_reason": "Stage1DataError: fewer than H+K frames",
        }
    ]
    validation = stage1.validate_invariants(records, rejected)
    assert validation["raw_candidate_count"] == len(records) + 1
    assert stage1._raw_inventory_payload(records, rejected)[-1] == {
        "path": "seen/rejected.hdf5",
        "sha256": "e" * 64,
    }
    assert "projection" not in stage1._split_payload(records)[0]

    rejected[0]["source_relative_path"] = records[0]["source_relative_path"]
    with pytest.raises(stage1.Stage1DataError, match="both accepted and rejected"):
        stage1.validate_invariants(records, rejected)


def test_legacy_quarantine_is_union_of_canonical_and_pool(tmp_path: Path) -> None:
    canonical_path = tmp_path / "data/Human2Robot/canonical/v3/preprocessing_manifest.json"
    pool_path = tmp_path / "data/Human2Robot/derived/m5b_v03/p1_human_only_pool/selection_manifest.json"
    canonical_path.parent.mkdir(parents=True)
    pool_path.parent.mkdir(parents=True)
    canonical_records = []
    pool_records = []
    source_index = 0
    for task in stage1.HELDOUT_TASKS:
        for index in range(stage1.LEGACY_QUARANTINE_COUNTS[task]):
            record = {"task": task, "source_sha256": f"{source_index:064x}"}
            source_index += 1
            pool_records.append(record)
            if index == 0:
                canonical_records.append(record)
    canonical_path.write_text(json.dumps({"episodes": canonical_records}), encoding="utf-8")
    pool_path.write_text(json.dumps({"episodes": pool_records}), encoding="utf-8")
    quarantine, provenance = stage1._legacy_quarantine(tmp_path)
    assert {task: len(values) for task, values in quarantine.items()} == stage1.LEGACY_QUARANTINE_COUNTS
    assert provenance["per_task_source_sha256_count"] == stage1.LEGACY_QUARANTINE_COUNTS


def test_approved_heldout_amendment_is_explicit() -> None:
    assert stage1.HELDOUT_TASKS == (
        "grab_cube2_v1",
        "push_plate_v1",
        "grab_to_plate1_v1",
        "push_box_random_v1",
    )
    assert "grab_pencil1_v1" not in stage1.HELDOUT_TASKS
    assert stage1.LEGACY_QUARANTINE_COUNTS["push_plate_v1"] == 0
    assert sum(stage1.LEGACY_QUARANTINE_COUNTS.values()) == 30


def test_replacement_assessment_is_read_only_and_does_not_authorize_protocol_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "raw"
    seen = source_root / "seen/episode_0.hdf5"
    _write_source(seen, frames=32)
    _write_source(source_root / "frozen_heldout/episode_0.hdf5", frames=33)
    _write_source(source_root / "candidate_good/episode_0.hdf5", frames=34)
    _write_source(source_root / "candidate_good/episode_1.hdf5", frames=35)
    (source_root / "candidate_overlap").mkdir(parents=True)
    shutil.copyfile(seen, source_root / "candidate_overlap/episode_0.hdf5")
    _write_source(source_root / "candidate_overlap/episode_1.hdf5", frames=36)

    canonical = tmp_path / "data/Human2Robot/canonical/v3/preprocessing_manifest.json"
    pool = tmp_path / "data/Human2Robot/derived/m5b_v03/p1_human_only_pool/selection_manifest.json"
    canonical.parent.mkdir(parents=True)
    pool.parent.mkdir(parents=True)
    canonical.write_text(json.dumps({"episodes": []}), encoding="utf-8")
    pool.write_text(json.dumps({"episodes": []}), encoding="utf-8")
    monkeypatch.setattr(stage1, "SEEN_TASKS", ("seen",))
    monkeypatch.setattr(stage1, "PRE_AMENDMENT_HELDOUT_TASKS", ("frozen_heldout",))

    result = stage1.assess_heldout_replacements(
        workspace=tmp_path,
        source_root=source_root,
        candidates=("candidate_good", "candidate_overlap"),
        replaced_task="frozen_heldout",
        required_new_count=2,
        execute=True,
    )
    assert result["status"] == "PASSED"
    assert result["stage1_status"] == "BLOCKED_PROTOCOL"
    assert result["protocol_change_authorized"] is False
    assert result["stage1_prepare_allowed"] is False
    assert result["data_quality_preference"] == "candidate_good"
    assert result["candidate_assessments"]["candidate_good"]["eligible_by_stage1_data_contract"] is True
    overlap = result["candidate_assessments"]["candidate_overlap"]
    assert overlap["baseline_overlap_unique_source_sha256_count"] == 1
    assert overlap["eligible_by_stage1_data_contract"] is False
