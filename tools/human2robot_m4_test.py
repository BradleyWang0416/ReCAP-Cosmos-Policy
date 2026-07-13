from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    from tools.human2robot_m3 import Episode, TimeViewSpec
    from tools.human2robot_m4 import (
        BASELINES,
        BridgeSample,
        M4Config,
        M4Error,
        build_bridge_samples,
        evaluate_pool_growth,
        fit_ridge,
        load_checkpoint,
        train_baselines,
        write_checkpoint,
    )
except ModuleNotFoundError:  # Direct pytest collection from the tools directory.
    from human2robot_m3 import Episode, TimeViewSpec
    from human2robot_m4 import (
        BASELINES,
        BridgeSample,
        M4Config,
        M4Error,
        build_bridge_samples,
        evaluate_pool_growth,
        fit_ridge,
        load_checkpoint,
        train_baselines,
        write_checkpoint,
    )


def _trajectory(length: int, offset: float = 0.0) -> np.ndarray:
    time = np.arange(length, dtype=np.float64)
    values = np.zeros((length, 10), dtype=np.float64)
    values[:, 0] = offset + time * 0.01
    values[:, 1] = 0.2 + time * 0.002
    values[:, 2] = 0.1
    values[:, 3] = 1.0
    values[:, 7] = 1.0
    values[:, 9] = (time >= length // 2).astype(np.float64)
    return values


def _episode(episode_id: str, task: str, split: str, offset: float = 0.0) -> Episode:
    human = _trajectory(48, offset)
    robot = human.copy()
    robot[1:] = human[:-1]
    return Episode(
        episode_id=episode_id,
        path=Path(f"{episode_id}.hdf5"),
        task=task,
        split=split,
        source_relative_path=f"{task}/episode_0.hdf5",
        human=human,
        robot=robot,
        segment_id=np.zeros(48, dtype=np.int64),
        gap_mask=np.zeros(48, dtype=bool),
    )


def test_bridge_samples_are_strict_future_and_do_not_cross_segments() -> None:
    episode = _episode("demo", "task", "train")
    episode.segment_id = np.asarray([0] * 24 + [1] * 24, dtype=np.int64)
    episode.gap_mask[24] = True
    heldout = _episode("held", "held_task", "heldout", offset=0.4)
    config = M4Config(horizon=4, window_stride=4, expected_episode_count=None)
    samples, summary = build_bridge_samples(
        [episode, heldout], TimeViewSpec("nominal", nominal_hz=30.0), config
    )
    assert summary["gap_crossing_count"] == 0
    assert samples
    for sample in (item for item in samples if item.episode_id == "demo"):
        np.testing.assert_allclose(sample.target[0], episode.robot[sample.current_row + 1])


def test_all_four_baselines_run_without_target_retrieval_features(tmp_path: Path) -> None:
    episodes = [
        _episode("train_a", "seen_a", "train", 0.0),
        _episode("train_b", "seen_b", "train", 0.2),
        _episode("held_a", "unseen", "heldout", 0.4),
    ]
    config = M4Config(
        horizon=4,
        window_stride=4,
        retrieval_top_k=2,
        pool_growth_sizes=(1, 0),
        output_root=tmp_path / "m4",
        expected_episode_count=None,
    )
    samples, summary = build_bridge_samples(
        episodes, TimeViewSpec("nominal_camera_30hz_segmented", nominal_hz=30.0), config
    )
    models = train_baselines(samples, config)
    results = evaluate_pool_growth(samples, models, config)
    assert summary["heldout_robot_trajectory_used_in_retrieval_feature"] is False
    assert all(set(item["methods"]) == set(BASELINES) for item in results)
    assert all(np.isfinite(item["methods"]["recap_hand_ret"]["position_error_median_canonical"]) for item in results)


def test_checkpoint_loader_hard_fails_on_view_mismatch(tmp_path: Path) -> None:
    model = fit_ridge(np.eye(4), np.eye(4), alpha=1e-2)
    models = {"no_retrieval": model, "co_training": model, "recap_hand_ret": model}
    bindings = {
        "canonical_schema": "v3",
        "canonical_manifest_sha256": "canonical",
        "source_evidence_manifest_sha256": "evidence",
        "split_sha256": "split",
        "time_view_id": "nominal_camera_30hz_segmented",
        "pool_action_view_id": "human_hand_robot_frame_raw",
        "query_action_view_id": "robot_ee_observed_t_plus_1_bc_proxy",
        "action_alignment_id": "alignment",
        "pool_action_role": "pool",
        "query_action_role": "query",
        "query_command_status": "unverified",
        "policy_coordinate": {"policy_dt": 1 / 30},
        "H_steps": 8,
        "H_seconds": 8 / 30,
        "K_steps": 8,
        "K_seconds": 8 / 30,
        "gap_policy": "never_cross_segment",
        "alignment_version": "alignment-hash",
        "view_id": "view",
        "retrieval_index_sha256": "index",
    }
    config = M4Config(output_root=tmp_path / "m4")
    checkpoint, manifest, _payload = write_checkpoint(models, bindings, config)
    loaded = load_checkpoint(checkpoint, manifest, bindings)
    assert set(loaded) == {"no_retrieval", "co_training", "recap_hand_ret"}
    changed = json.loads(json.dumps(bindings))
    changed["time_view_id"] = "wrong_view"
    with pytest.raises(M4Error, match="binding mismatch"):
        load_checkpoint(checkpoint, manifest, changed)
