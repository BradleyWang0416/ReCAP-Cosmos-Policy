from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from tools.human2robot_m2 import (
    LEGACY_FIXED_STRIDE3,
    PRESERVE_NATIVE,
    ConversionConfig,
    ValidationLimits,
    audit_timebase,
    compute_and_write_statistics,
    convert_dataset,
    convert_episode,
    frame_indices,
    legacy_resample_indices,
    poses_euler_to_10d,
    require_canonical_v3,
    run_m2_pipeline,
    validate_canonical_episode,
)


def _write_source(path: Path, frame_count: int = 31, image_dtype: object = np.uint8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(frame_count, dtype=np.float32)
    pose = np.stack(
        (
            100.0 + time,
            300.0 + time,
            20.0 + time * 0.1,
            np.zeros_like(time),
            np.zeros_like(time),
            np.zeros_like(time),
        ),
        axis=1,
    )
    action = np.concatenate((pose + np.array([1, 1, 1, 0, 0, 0]), (time % 2)[:, None]), axis=1)
    with h5py.File(path, "w") as file:
        file.create_dataset("action", data=action.astype(np.float32))
        camera = file.create_group("cam_data")
        images = np.zeros((frame_count, 8, 10, 3), dtype=image_dtype)
        images[:, :, :, 0] = np.arange(frame_count, dtype=image_dtype)[:, None, None]
        camera.create_dataset("human_camera", data=images)
        camera.create_dataset("robot_camera", data=images)
        file.create_dataset("end_position", data=pose.astype(np.float32))
        file.create_dataset("gripper_state", data=(time % 2).astype(np.float32))
        file.create_dataset("qpos", data=np.zeros((frame_count, 7), dtype=np.float32))
        file.create_dataset("qvel", data=np.zeros((frame_count, 7), dtype=np.float32))
        file.create_dataset("step", data=np.arange(frame_count, dtype=np.int64))
        file.create_dataset("timestamp", data=(1_700_000_000 + time // 30).astype(np.int64))
        file.create_dataset("transformed_hand_coords", data=np.zeros((frame_count, 24, 3), dtype=np.float32))
        frames = np.zeros((frame_count, 4, 3), dtype=np.float32)
        frames[:, 0, 0] = 1.0
        file.create_dataset("transformed_hand_frames", data=frames)


def _write_parent_split(path: Path, tasks: list[str], heldout_task_count: int) -> None:
    import hashlib

    ranked = sorted(tasks, key=lambda task: hashlib.sha256(f"20260711:{task}".encode()).hexdigest())
    heldout = set(ranked[:heldout_task_count])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "split_sha256": "test-v2-parent",
                "tasks": {task: ("heldout" if task in heldout else "train") for task in tasks},
            }
        )
    )


def test_native_policy_preserves_all_frames_and_legacy_is_explicit() -> None:
    np.testing.assert_array_equal(frame_indices(31, PRESERVE_NATIVE), np.arange(31))
    np.testing.assert_array_equal(legacy_resample_indices(31), np.arange(0, 31, 3))
    np.testing.assert_array_equal(frame_indices(31, LEGACY_FIXED_STRIDE3), np.arange(0, 31, 3))


def test_timebase_audit_segments_jumps_and_rollbacks() -> None:
    audit = audit_timebase(
        np.array([0, 1, 8, 9, 4, 5]),
        np.array([100, 100, 101, 101, 99, 99]),
    )
    assert audit["timebase_status"] == "discontinuous"
    assert audit["step_jump_count"] == 1
    assert audit["step_rollback_count"] == 1
    assert audit["timestamp_rollback_count"] == 1
    np.testing.assert_array_equal(audit["gap_mask"], [False, False, True, False, True, False])
    np.testing.assert_array_equal(audit["segment_id"], [0, 0, 1, 1, 2, 2])


def test_pose_conversion_identity_rotation() -> None:
    result = poses_euler_to_10d(
        np.array([[100.0, 200.0, 300.0, 0.0, 0.0, 0.0]]),
        np.array([1.0]),
        position_scale=0.001,
        euler_order="xyz",
    )
    np.testing.assert_allclose(result[0, :3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(result[0, 3:9], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert result[0, -1] == 1.0


def test_convert_validate_and_write_statistics(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = source_root / "task_a/episode_0.hdf5"
    output_root = tmp_path / "canonical"
    output = output_root / "pilot/demo_00000.hdf5"
    _write_source(source, image_dtype=np.uint16)
    config = ConversionConfig(
        source_root=source_root,
        output_root=output_root,
        report_root=tmp_path / "reports",
        episode_count=1,
        heldout_task_count=0,
    )
    record = convert_episode(
        source,
        output,
        config,
        demo_index=0,
        task_split="train",
        split_manifest_hash="test-split",
    )
    assert record["canonical_frames"] == 31
    assert record["canonical_frames"] == record["source_frames"]

    validation = validate_canonical_episode(output, ValidationLimits(), source_root=source_root)
    assert validation["status"] == "passed", validation["errors"]
    with h5py.File(output, "r") as file:
        assert file["data/demo_0/obs/robot_images"].shape == (31, 8, 10, 3)
        assert file["data/demo_0/obs/robot_images"].dtype == np.dtype("uint8")
        assert file["data/demo_0/metadata/human/images"].shape == (31, 8, 10, 3)
        assert file["data/demo_0/metadata"].attrs["human_image_conversion"] == "uint16_container_cast_to_uint8"
        assert file["data/demo_0/trajectories/human_hand_robot_frame_10d"].shape == (31, 10)
        assert file["data/demo_0/trajectories/robot_ee_observed_10d"].shape == (31, 10)
        assert "actions" not in file["data/demo_0"]
        assert "timestamps" not in file["data/demo_0/metadata"]
        np.testing.assert_array_equal(file["data/demo_0/metadata/source_indices"][:], np.arange(31))

    summary = compute_and_write_statistics(
        [output],
        output_root,
        required_split="train",
        split_manifest_hash="test-split",
    )
    assert summary["frame_count"] == 31
    for name in (
        "robot_observed_statistics.json",
        "human_hand_robot_frame_statistics.json",
        "dataset_statistics_post_norm_by_role.json",
        "data_quality_statistics.json",
    ):
        assert (output_root / name).is_file()
    statistics = json.loads((output_root / "human_hand_robot_frame_statistics.json").read_text())
    assert len(statistics["human_hand_robot_frame_10d_min"]) == 10
    assert statistics["_provenance"]["role"] == "human_hand_pose_in_robot_frame"
    assert statistics["_provenance"]["heldout_data_used"] is False
    assert not (output_root / "delta_dataset_statistics.json").exists()


def test_validator_rejects_nan(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = source_root / "task_a/episode_0.hdf5"
    output = tmp_path / "canonical/pilot/demo_00000.hdf5"
    _write_source(source)
    config = ConversionConfig(
        source_root=source_root,
        output_root=tmp_path / "canonical",
        report_root=tmp_path / "reports",
        episode_count=1,
        heldout_task_count=0,
    )
    convert_episode(
        source,
        output,
        config,
        demo_index=0,
        task_split="train",
        split_manifest_hash="test-split",
    )
    with h5py.File(output, "r+") as file:
        file["data/demo_0/trajectories/human_hand_robot_frame_10d"][0, 0] = np.nan
    report = validate_canonical_episode(output, ValidationLimits(), source_root=source_root)
    assert report["status"] == "failed"
    assert any("human_hand_robot_frame_10d contains NaN/Inf" in error for error in report["errors"])


def test_validator_rejects_artificial_timeline(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = source_root / "task_a/episode_0.hdf5"
    output = tmp_path / "canonical/pilot/demo_00000.hdf5"
    _write_source(source)
    config = ConversionConfig(
        source_root=source_root,
        output_root=tmp_path / "canonical",
        report_root=tmp_path / "reports",
        episode_count=1,
        heldout_task_count=0,
    )
    convert_episode(
        source,
        output,
        config,
        demo_index=0,
        task_split="train",
        split_manifest_hash="test-split",
    )
    with h5py.File(output, "r+") as file:
        file["data/demo_0/metadata"].create_dataset("timestamps", data=np.arange(31) / 10.0)
    report = validate_canonical_episode(output, ValidationLimits(), source_root=source_root)
    assert report["time_truth_validation"]["status"] == "failed"
    assert any("synthetic" in error for error in report["time_truth_validation"]["errors"])


def test_task_split_is_frozen_before_statistics(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_source(source_root / "task_a/episode_0.hdf5", frame_count=6)
    _write_source(source_root / "task_b/episode_0.hdf5", frame_count=6)
    parent = tmp_path / "parent-v2-split.json"
    _write_parent_split(parent, ["task_a", "task_b"], 1)
    config = ConversionConfig(
        source_root=source_root,
        output_root=tmp_path / "canonical",
        report_root=tmp_path / "reports",
        episode_count=2,
        heldout_task_count=1,
        parent_v2_split_manifest=parent,
    )
    manifest = convert_dataset(config)
    assert {record["task_split"] for record in manifest["episodes"]} == {"train", "heldout"}
    split = json.loads((config.output_root / "task_split_manifest.json").read_text())
    assert set(split["train_tasks"]).isdisjoint(split["heldout_tasks"])
    assert len(split["episodes"]) == 2


def test_small_native_pipeline_passes_both_validator_categories(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_source(source_root / "task_a/episode_0.hdf5", frame_count=6)
    _write_source(source_root / "task_b/episode_0.hdf5", frame_count=6)
    parent = tmp_path / "parent-v2-split.json"
    _write_parent_split(parent, ["task_a", "task_b"], 1)
    config = ConversionConfig(
        source_root=source_root,
        output_root=tmp_path / "canonical",
        report_root=tmp_path / "reports",
        legacy_v1_root=tmp_path / "missing-v1",
        legacy_v1_report=tmp_path / "missing-v1-report.md",
        episode_count=2,
        heldout_task_count=1,
        parent_v2_split_manifest=parent,
    )
    report = run_m2_pipeline(config, ValidationLimits(), visualization_count=0)
    assert report["status"] == "passed"
    validation = json.loads((config.output_root / "m2_validation_report.json").read_text())
    assert validation["status"] == "passed"
    assert validation["aggregate"]["source_one_to_one_verified_count"] == 2
    assert all(episode["structure_validation"]["status"] == "passed" for episode in validation["episodes"])
    assert all(episode["time_truth_validation"]["status"] == "passed" for episode in validation["episodes"])
    assert all(episode["evidence_role_validation"]["status"] == "passed" for episode in validation["episodes"])


def test_nominal_fps_requires_provenance_and_record_clock_stays_untrusted(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = source_root / "task_a/episode_0.hdf5"
    output = tmp_path / "canonical/pilot/demo_00000.hdf5"
    _write_source(source)
    config = ConversionConfig(source_root=source_root, output_root=tmp_path / "canonical")
    convert_episode(source, output, config, demo_index=0, task_split="train", split_manifest_hash="test")
    with h5py.File(output, "r") as file:
        metadata = file["data/demo_0/metadata"]
        assert metadata.attrs["nominal_camera_fps"] == 30.0
        assert metadata.attrs["nominal_camera_fps_status"] == "verified_upstream"
        assert not metadata.attrs["record_timebase_globally_trusted"]
    with h5py.File(output, "r+") as file:
        del file["data/demo_0/metadata"].attrs["nominal_camera_fps_source_url"]
    report = validate_canonical_episode(output, ValidationLimits(), source_root=source_root)
    assert report["status"] == "failed"
    assert any("nominal_camera_fps_source_url" in error for error in report["errors"])


def test_role_swap_and_generic_action_are_rejected(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = source_root / "task_a/episode_0.hdf5"
    output = tmp_path / "canonical/pilot/demo_00000.hdf5"
    _write_source(source)
    config = ConversionConfig(source_root=source_root, output_root=tmp_path / "canonical")
    convert_episode(source, output, config, demo_index=0, task_split="train", split_manifest_hash="test")
    with h5py.File(output, "r+") as file:
        demo = file["data/demo_0"]
        demo.create_dataset("actions", data=demo["trajectories/human_hand_robot_frame_10d"][:])
        robot = demo["trajectories/robot_ee_observed_10d"][:]
        human = demo["trajectories/human_hand_robot_frame_10d"][:]
        demo["trajectories/robot_ee_observed_10d"][:] = human
        demo["trajectories/human_hand_robot_frame_10d"][:] = robot
    report = validate_canonical_episode(output, ValidationLimits(), source_root=source_root)
    assert report["evidence_role_validation"]["status"] == "failed"
    assert any("generic actions" in error for error in report["errors"])
    assert any("not mapped" in error for error in report["errors"])


def test_formal_v3_reader_hard_rejects_legacy_schema(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.hdf5"
    with h5py.File(legacy, "w") as file:
        file.attrs["schema_version"] = "human2robot-canonical-hdf5-v2"
        demo = file.create_group("data/demo_0")
        demo.attrs["schema_version"] = "human2robot-canonical-hdf5-v2"
    try:
        require_canonical_v3(legacy)
    except Exception as exc:
        assert "require schema human2robot-canonical-hdf5-v3" in str(exc)
    else:
        raise AssertionError("legacy schema was not rejected")
