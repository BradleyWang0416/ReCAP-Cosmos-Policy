from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

try:
    from tools.human2robot_m2 import SCHEMA_VERSION
    from tools.human2robot_m3 import (
        ALIGNMENT_ID,
        POOL_ACTION_VIEW_RAW,
        QUERY_ACTION_VIEW,
        Episode,
        M3Config,
        TimeViewSpec,
        align_pool_chunk,
        build_retrieval_index,
        calibrate_alignment,
        rotation_6d_to_matrix,
        run_m3_pipeline,
        view_segment_indices,
    )
except ModuleNotFoundError:  # Direct pytest collection from the tools directory.
    from human2robot_m2 import SCHEMA_VERSION
    from human2robot_m3 import (
        ALIGNMENT_ID,
        POOL_ACTION_VIEW_RAW,
        QUERY_ACTION_VIEW,
        Episode,
        M3Config,
        TimeViewSpec,
        align_pool_chunk,
        build_retrieval_index,
        calibrate_alignment,
        rotation_6d_to_matrix,
        run_m3_pipeline,
        view_segment_indices,
    )


def _trajectory(length: int, offset: float = 0.0) -> np.ndarray:
    time = np.arange(length, dtype=np.float64)
    values = np.zeros((length, 10), dtype=np.float64)
    values[:, 0] = offset + time * 0.01
    values[:, 1] = 0.25 + time * 0.002
    values[:, 2] = 0.1
    values[:, 3] = 1.0
    values[:, 7] = 1.0
    values[:, 9] = (time >= length // 2).astype(np.float64)
    return values


def _episode(episode_id: str, split: str, *, offset: float = 0.0, length: int = 64) -> Episode:
    human = _trajectory(length, offset)
    robot = human.copy()
    robot[1:] = human[:-1]  # The strictly-future t+1 label is exactly aligned.
    segment_id = np.zeros(length, dtype=np.int64)
    return Episode(
        episode_id=episode_id,
        path=Path(f"{episode_id}.hdf5"),
        task=f"task_{episode_id}",
        split=split,
        source_relative_path=f"task_{episode_id}/episode_0.hdf5",
        human=human,
        robot=robot,
        segment_id=segment_id,
        gap_mask=np.zeros(length, dtype=bool),
    )


def test_time_view_indices_never_cross_segment() -> None:
    episode = _episode("demo", "train", length=12)
    episode.segment_id = np.array([0] * 5 + [1] * 7)
    episode.gap_mask[5] = True
    result = view_segment_indices(episode, TimeViewSpec("stride3", stride=3))
    assert [indices.tolist() for indices in result] == [[0, 3], [5, 8, 11]]
    assert all(len(np.unique(episode.segment_id[indices])) == 1 for indices in result)


def test_query_anchor_alignment_preserves_valid_rotation() -> None:
    pool = _trajectory(5)
    query = _trajectory(1, offset=0.4)[0]
    aligned = align_pool_chunk(pool, query)
    np.testing.assert_allclose(aligned[0, :3], query[:3])
    np.testing.assert_allclose(aligned[0, 3:9], query[3:9])
    matrices = rotation_6d_to_matrix(aligned[:, 3:9])
    np.testing.assert_allclose(
        matrices @ matrices.transpose(0, 2, 1), np.repeat(np.eye(3)[None], len(matrices), axis=0), atol=1e-7
    )
    np.testing.assert_allclose(np.linalg.det(matrices), 1.0, atol=1e-7)


def test_retrieval_index_uses_phase_only_and_has_top_k() -> None:
    episodes = [_episode("train0", "train"), _episode("train1", "train", offset=0.2), _episode("held", "heldout")]
    config = M3Config(horizon=4, window_stride=2, top_k=10)
    metrics, arrays = build_retrieval_index(
        episodes, TimeViewSpec("nominal_camera_30hz_segmented", nominal_hz=30.0), config
    )
    assert metrics["every_query_has_top_k"]
    assert metrics["gap_crossing_count"] == 0
    assert metrics["retrieval_feature_schema"] == ["normalized_segment_phase"]
    assert metrics["heldout_robot_trajectory_used_in_retrieval_feature"] is False
    assert arrays["candidate_episode_id"].shape[1] == 10


def test_train_only_lag_calibration_selects_strict_future_tplus1() -> None:
    episodes = [_episode("train", "train"), _episode("held", "heldout", offset=5.0)]
    config = M3Config(max_lag=4, wrong_lag=4)
    alignment = calibrate_alignment(episodes, config, {"split_sha256": "test"})
    assert alignment["calibration_scope"] == "train_tasks_only"
    assert alignment["heldout_tasks_used"] is False
    assert alignment["best_position_error_lag_source_rows"] == 1
    assert alignment["approved_query_action_view_id"] == QUERY_ACTION_VIEW
    assert alignment["approved_future_offset_view_steps"] == 1
    assert alignment["deployment_command_adapter_id"] is None


def _write_canonical_episode(path: Path, source_relative_path: str, offset: float) -> None:
    human = _trajectory(64, offset)
    robot = human.copy()
    robot[1:] = human[:-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        file.attrs["schema_version"] = SCHEMA_VERSION
        demo = file.create_group("data/demo_0")
        demo.attrs["schema_version"] = SCHEMA_VERSION
        demo.attrs["source_relative_path"] = source_relative_path
        trajectories = demo.create_group("trajectories")
        trajectories.create_dataset("human_hand_robot_frame_10d", data=human.astype(np.float32))
        trajectories.create_dataset("robot_ee_observed_10d", data=robot.astype(np.float32))
        metadata = demo.create_group("metadata")
        metadata.create_dataset("segment_id", data=np.zeros(64, dtype=np.int32))
        metadata.create_dataset("gap_mask", data=np.zeros(64, dtype=bool))
        metadata.attrs["source_action_role"] = "human_hand_pose_in_robot_frame"
        metadata.attrs["robot_trajectory_role"] = "observed_robot_ee_pose"


def test_small_end_to_end_pipeline_writes_passed_acceptance(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical/v3"
    records = [
        ("task_a/episode_0.hdf5", "task_a", "train", 0.0),
        ("task_b/episode_0.hdf5", "task_b", "train", 0.2),
        ("task_c/episode_0.hdf5", "task_c", "heldout", 0.4),
    ]
    for index, (source, _task, _split, offset) in enumerate(records):
        _write_canonical_episode(canonical / f"pilot/demo_{index:05d}.hdf5", source, offset)
    split_manifest = {
        "split_sha256": "test-split",
        "episodes": [
            {"source_relative_path": source, "task": task, "split": split}
            for source, task, split, _offset in records
        ],
    }
    (canonical / "task_split_manifest.json").write_text(json.dumps(split_manifest), encoding="utf-8")
    (canonical / "preprocessing_manifest.json").write_text(
        json.dumps({"episodes": [{"source_relative_path": item[0]} for item in records]}), encoding="utf-8"
    )
    evidence = tmp_path / "source_evidence_manifest_v3.json"
    evidence.write_text(
        json.dumps(
            {
                "source_action_role": "human_hand_pose_in_robot_frame",
                "source_action_role_status": "verified_upstream",
                "source_action_as_robot_command_status": "unknown",
                "sources": [{"url": "https://example.test", "version": "test", "accessed_at": "2026-07-11"}],
            }
        ),
        encoding="utf-8",
    )
    config = M3Config(
        canonical_root=canonical,
        derived_root=tmp_path / "derived",
        report_root=tmp_path / "reports",
        evidence_manifest=evidence,
        horizon=4,
        window_stride=2,
        top_k=2,
        max_lag=4,
        wrong_lag=4,
        phase_bins=16,
        expected_episode_count=3,
    )
    report = run_m3_pipeline(config)
    assert report["status"] == "passed"
    assert report["gates"] == {"A2a": "passed", "A2b": "passed", "B": "passed"}
    assert report["retrieval_sanity"]["heldout_robot_trajectory_used_in_retrieval_feature"] is False
    assert report["deployment_command_adapter_id"] is None
    main_path = (
        config.derived_root
        / "views/nominal_camera_30hz_segmented"
        / POOL_ACTION_VIEW_RAW
        / QUERY_ACTION_VIEW
        / ALIGNMENT_ID
    )
    assert (main_path / "retrieval_index.npz").is_file()
    manifest = json.loads((main_path / "view_manifest.json").read_text(encoding="utf-8"))
    assert manifest["H_steps"] == manifest["K_steps"] == 4
    assert manifest["implementation_code_sha256"]
    assert manifest["action_alignment_manifest_sha256"]
    assert (config.report_root / "M3_action_time_residual_验收报告.md").is_file()
