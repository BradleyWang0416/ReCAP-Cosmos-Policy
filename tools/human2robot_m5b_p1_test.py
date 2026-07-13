from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tools.human2robot_m5b_p1 import (
    ALLOWED_SOURCE_DATASETS,
    DERIVED_DATASETS,
    P1Error,
    _derived_dataset_paths,
    convert_human_episode,
    inspect_human_source,
    rank_source_candidates,
    validate_records,
)


def _source(path: Path, *, frames: int = 12, marker: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        camera = file.create_group("cam_data")
        camera.create_dataset(
            "human_camera",
            data=np.full((frames, 4, 6, 3), marker, dtype=np.uint16),
        )
        # Deliberately invalid robot-side data: P1 must never inspect it.
        camera.create_dataset("robot_camera", data=np.full((1,), np.nan, dtype=np.float32))
        action = np.zeros((frames, 7), dtype=np.float32)
        action[:, 0] = marker
        action[:, 3:6] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        action[:, 6] = np.linspace(0.0, 1.0, frames)
        file.create_dataset("action", data=action)
        file.create_dataset("transformed_hand_coords", data=np.full((frames, 24, 3), marker, dtype=np.float32))
        file.create_dataset("transformed_hand_frames", data=np.full((frames, 4, 3), marker, dtype=np.float32))
        file.create_dataset("step", data=np.arange(frames, dtype=np.int64))
        file.create_dataset("timestamp", data=np.arange(frames, dtype=np.int64))
        file.create_dataset("end_position", data=np.full((1,), np.nan, dtype=np.float32))
        file.create_dataset("qpos", data=np.full((1,), np.nan, dtype=np.float32))
        file.create_dataset("qvel", data=np.full((1,), np.nan, dtype=np.float32))
        file.create_dataset("gripper_state", data=np.full((1,), np.nan, dtype=np.float32))


def test_inspection_reads_only_human_allowlist(tmp_path: Path) -> None:
    source = tmp_path / "task/episode_0.hdf5"
    _source(source)
    summary = inspect_human_source(source)
    assert summary["source_datasets_read"] == list(ALLOWED_SOURCE_DATASETS)
    assert summary["forbidden_source_datasets_read"] == []
    assert summary["frame_count"] == 12
    assert summary["max_gap_safe_segment_frames"] == 12


def test_conversion_emits_human_only_schema(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = source_root / "heldout_task/episode_0.hdf5"
    output = tmp_path / "pool/episode_0.hdf5"
    _source(source)
    record = convert_human_episode(
        source,
        output,
        source_root=source_root,
        task="heldout_task",
        pool_rank=1,
        protocol_sha256="a" * 64,
        split_sha256="b" * 64,
        conversion_code_sha256="c" * 64,
        overwrite=False,
    )
    assert record["forbidden_source_datasets_read"] == []
    assert _derived_dataset_paths(output) == DERIVED_DATASETS
    with h5py.File(output, "r") as file:
        demo = file["data/demo_0"]
        assert bool(demo.attrs["human_only"]) is True
        assert bool(demo.attrs["contains_robot_observation_or_target"]) is False
        assert demo.attrs["conversion_code_sha256"] == "c" * 64
        assert demo["human/images"].dtype == np.uint8
        assert demo["human/hand_plan_10d"].shape == (12, 10)


def test_source_ranking_keeps_frozen_anchor_first(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    for index in range(12):
        _source(source_root / f"heldout_task/episode_{index}.hdf5", marker=index + 1)
    ranked = rank_source_candidates(
        source_root,
        "heldout_task",
        "heldout_task/episode_0.hdf5",
    )
    assert ranked[0].name == "episode_0.hdf5"
    assert len(ranked) == 12
    assert len(set(ranked)) == 12


def _records() -> tuple[list[dict[str, object]], list[str], dict[str, str]]:
    tasks = ["task_a", "task_b", "task_c", "task_d"]
    anchors = {task: f"{task}/episode_0.hdf5" for task in tasks}
    records: list[dict[str, object]] = []
    index = 0
    for task in tasks:
        for rank in range(1, 11):
            source = anchors[task] if rank == 1 else f"{task}/episode_{rank}.hdf5"
            records.append(
                {
                    "task": task,
                    "pool_rank": rank,
                    "source_relative_path": source,
                    "source_sha256": f"source-{index}",
                    "human_content_sha256": f"human-{index}",
                    "output_sha256": f"output-{index}",
                    "max_gap_safe_segment_frames": 8,
                    "forbidden_source_datasets_read": [],
                }
            )
            index += 1
    return records, tasks, anchors


def test_independence_validation_requires_unique_source_and_human_content() -> None:
    records, tasks, anchors = _records()
    result = validate_records(records, heldout_tasks=tasks, train_tasks=["train"], anchors=anchors)
    assert result["every_task_meets_requirement"] is True
    assert result["source_episode_identity_unique_count"] == 40
    records[1]["human_content_sha256"] = records[0]["human_content_sha256"]
    with pytest.raises(P1Error, match="Duplicate human demonstration content"):
        validate_records(records, heldout_tasks=tasks, train_tasks=["train"], anchors=anchors)
