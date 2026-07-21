from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from cosmos_policy.datasets.human2robot_v04_retrieval import (
    CANDIDATE_PARTITION,
    HUMAN_DATASETS,
    ROBOT_DATASETS,
    FeatureProvenance,
    Human2RobotV04RetrievalError,
    P2Window,
    RetrievalFeature,
    assert_pool_growth_nested,
    build_retrieval_feature,
    candidate_rejection_reason,
    filter_candidates,
    rank_geometry_plus_visual,
    rank_oracle_phase,
    read_feature_inputs,
    validate_primary_config,
    window_from_manifest_record,
)


def _write_projection(
    path: Path,
    *,
    role: str,
    partition: str,
    source_sha: str,
    source_relative_path: str,
    task: str = "heldout",
    rank: int = 1,
) -> dict[str, object]:
    frames = 24
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        demo = file.create_group("data/demo_0")
        identity = {
            "source_sha256": source_sha,
            "source_relative_path": source_relative_path,
            "source_partition": partition,
            "task": task,
            "episode_id": source_relative_path.removesuffix(".hdf5"),
            "role": role,
        }
        for key, value in identity.items():
            demo.attrs[key] = value
        demo.attrs["frame_count"] = frames
        time = demo.create_group("time")
        time.create_dataset("gap_mask", data=np.zeros(frames, dtype=bool))
        time.create_dataset("legal_window_start", data=np.arange(9, dtype=np.int64))
        time.create_dataset("segment_id", data=np.zeros(frames, dtype=np.int32))
        time.create_dataset("source_step", data=np.arange(frames, dtype=np.int64))
        time.create_dataset("source_timestamp", data=np.arange(frames, dtype=np.int64))
        pose = np.zeros((frames, 6), dtype=np.float32)
        pose[:, 0] = np.arange(frames, dtype=np.float32)
        pose[:, 3] = np.arange(frames, dtype=np.float32) * 0.5
        images = np.zeros((frames, 4, 5, 3), dtype=np.uint8)
        images[:] = np.arange(frames, dtype=np.uint8)[:, None, None, None]
        if role == "human":
            human = demo.create_group("human")
            action = np.concatenate((pose, np.linspace(0.0, 1.0, frames)[:, None]), axis=1)
            human.create_dataset("hand_action_7d", data=action)
            human.create_dataset("hand_coords", data=np.zeros((frames, 24, 3), dtype=np.float32))
            human.create_dataset("hand_frames", data=np.zeros((frames, 4, 3), dtype=np.float32))
            human.create_dataset("images", data=images)
            content_key = "human_content_sha256"
        else:
            robot = demo.create_group("robot")
            robot.create_dataset("observed_eef_pose_6d", data=pose)
            robot.create_dataset("gripper_state", data=np.linspace(0.0, 1.0, frames, dtype=np.float32))
            robot.create_dataset("images", data=images)
            content_key = "robot_content_sha256"
    return {
        **identity,
        "partition_rank": rank,
        content_key: ("a" if role == "human" else "b") * 64,
        "projection": {"path": str(path)},
    }


def _windows(tmp_path: Path) -> tuple[P2Window, list[P2Window]]:
    query_record = _write_projection(
        tmp_path / "robot.hdf5",
        role="robot",
        partition="v04_robot_dev",
        source_sha="f" * 64,
        source_relative_path="heldout/robot.hdf5",
    )
    query = window_from_manifest_record(query_record, 0)
    candidates = []
    for rank in range(1, 11):
        record = _write_projection(
            tmp_path / f"human_{rank}.hdf5",
            role="human",
            partition=CANDIDATE_PARTITION,
            source_sha=f"{rank:064x}",
            source_relative_path=f"heldout/human_{rank}.hdf5",
            rank=rank,
        )
        record["human_content_sha256"] = f"{rank + 100:064x}"
        candidates.append(window_from_manifest_record(record, 0))
    return query, candidates


def _feature(window: P2Window, value: float) -> RetrievalFeature:
    geometry_datasets = (
        ("data/demo_0/human/hand_action_7d",)
        if window.role == "human"
        else ("data/demo_0/robot/observed_eef_pose_6d", "data/demo_0/robot/gripper_state")
    )
    visual_dataset = f"data/demo_0/{window.role}/images"
    provenance = FeatureProvenance(
        role=window.role,
        source_sha256=window.source_sha256,
        source_relative_path=window.source_relative_path,
        source_partition=window.source_partition,
        geometry_datasets=geometry_datasets,
        geometry_rows=window.history_rows,
        visual_dataset=visual_dataset,
        visual_row=window.current_row,
        visual_feature_kind="frozen_wan_latent",
    )
    return RetrievalFeature(
        geometry=np.asarray([1.0, value], dtype=np.float32),
        visual=np.asarray([1.0, -value], dtype=np.float32),
        provenance=provenance,
    )


def test_window_binds_source_identity_and_projection_allowlist(tmp_path: Path) -> None:
    query, candidates = _windows(tmp_path)
    assert query.source_partition == "v04_robot_dev"
    assert query.dataset_paths == tuple(sorted(ROBOT_DATASETS))
    assert candidates[0].source_partition == CANDIDATE_PARTITION
    assert candidates[0].dataset_paths == tuple(sorted(HUMAN_DATASETS))
    assert query.history_rows == tuple(range(8)) and query.future_rows == tuple(range(8, 16))


def test_same_sha_and_same_path_are_rejected_independently(tmp_path: Path) -> None:
    query, candidates = _windows(tmp_path)
    same_sha = replace(candidates[0], source_sha256=query.source_sha256)
    same_path = replace(candidates[1], source_relative_path=query.source_relative_path)
    assert candidate_rejection_reason(query, same_sha, 10) == "same_source_sha256"
    assert candidate_rejection_reason(query, same_path, 10) == "same_source_relative_path"


def test_wrong_partition_role_and_robot_fields_never_enter_pool(tmp_path: Path) -> None:
    query, candidates = _windows(tmp_path)
    wrong_partition = replace(candidates[0], source_partition="legacy_quarantine")
    wrong_role = replace(candidates[1], role="robot")
    robot_fields = replace(candidates[2], dataset_paths=tuple(sorted(ROBOT_DATASETS)))
    eligible, rejected = filter_candidates(query, [wrong_partition, wrong_role, robot_fields], pool_size=10)
    assert not eligible
    assert rejected == {
        "candidate_dataset_allowlist_violation": 1,
        "candidate_not_human_only": 1,
        "candidate_not_v04_human_pool": 1,
    }


def test_feature_reader_uses_only_history_and_current_frame(tmp_path: Path) -> None:
    query, candidates = _windows(tmp_path)
    for window in (query, candidates[0]):
        history, image, provenance = read_feature_inputs(window)
        assert history.shape == (8, 10)
        assert image.shape == (4, 5, 3)
        assert provenance.geometry_rows == window.history_rows
        assert provenance.visual_row == window.current_row
        assert not provenance.future_rows_read
        assert not provenance.target_datasets_read
        assert not provenance.opposite_role_datasets_read


def test_modifying_robot_future_target_leaves_feature_unchanged(tmp_path: Path) -> None:
    query, _ = _windows(tmp_path)
    before_history, before_image, before_provenance = read_feature_inputs(query)
    future_slice = slice(query.future_rows[0], query.future_rows[-1] + 1)
    with h5py.File(query.path, "r+") as file:
        file["data/demo_0/robot/observed_eef_pose_6d"][future_slice] = np.full((8, 6), 999999.0)
        file["data/demo_0/robot/gripper_state"][future_slice] = np.zeros(8)
    after_history, after_image, after_provenance = read_feature_inputs(query)
    np.testing.assert_array_equal(before_history, after_history)
    np.testing.assert_array_equal(before_image, after_image)
    assert before_provenance == after_provenance


def test_primary_feature_calls_encoder_once_with_current_frame(tmp_path: Path) -> None:
    query, _ = _windows(tmp_path)
    seen = []

    def encoder(image: np.ndarray) -> np.ndarray:
        seen.append(image.copy())
        return np.ones((4, 2, 2), dtype=np.float32)

    feature = build_retrieval_feature(
        query,
        geometry_mean_10d=np.zeros(10, dtype=np.float32),
        geometry_std_10d=np.ones(10, dtype=np.float32),
        frozen_wan_encoder=encoder,
    )
    assert len(seen) == 1
    assert bool(np.all(seen[0] == query.current_row))
    assert feature.provenance.visual_feature_kind == "frozen_wan_latent"


def test_primary_config_hard_fails_phase_and_noncanonical_pool() -> None:
    validate_primary_config({"retrieval_modality": "geometry_plus_visual", "top_k": 3, "pool_size": 10})
    with pytest.raises(Human2RobotV04RetrievalError, match="diagnostic-only"):
        validate_primary_config({"retrieval_modality": "phase", "top_k": 3, "pool_size": 10})
    with pytest.raises(Human2RobotV04RetrievalError, match="pool_size"):
        validate_primary_config({"retrieval_modality": "geometry_plus_visual", "top_k": 3, "pool_size": 8})


def test_pool_growth_is_strictly_nested_by_frozen_rank(tmp_path: Path) -> None:
    _, candidates = _windows(tmp_path)
    nested = assert_pool_growth_nested(candidates)
    assert nested[1]["heldout"] == (candidates[0].source_sha256,)
    for smaller, larger in zip((1, 2, 4, 8), (2, 4, 8, 10)):
        assert set(nested[smaller]["heldout"]) < set(nested[larger]["heldout"])


def test_geometry_plus_visual_ranking_records_full_provenance(tmp_path: Path) -> None:
    query, candidates = _windows(tmp_path)
    features = {query.window_id: _feature(query, 0.0)}
    features.update({candidate.window_id: _feature(candidate, rank / 10.0) for rank, candidate in enumerate(candidates, 1)})
    first = rank_geometry_plus_visual(query, candidates, features)
    repeat = rank_geometry_plus_visual(query, list(reversed(candidates)), features)
    assert [item.candidate_id for item in first] == [item.candidate_id for item in repeat]
    assert len(first) == 3
    assert all(item.query_source_sha256 == query.source_sha256 for item in first)
    assert all(item.candidate_partition == CANDIDATE_PARTITION for item in first)
    assert all(item.modality == "geometry_plus_visual" for item in first)
    assert all(item.to_dict()["candidate_feature_provenance"]["visual_row"] == 7 for item in first)


def test_oracle_phase_requires_completed_primary_receipt(tmp_path: Path) -> None:
    query, candidates = _windows(tmp_path)
    with pytest.raises(Human2RobotV04RetrievalError, match="primary completion"):
        rank_oracle_phase(query, candidates, primary_completion_receipt_sha256="")
    ranked = rank_oracle_phase(query, candidates, primary_completion_receipt_sha256="c" * 64)
    assert ranked and ranked[0][1] >= 0.0
