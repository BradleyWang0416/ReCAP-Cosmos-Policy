from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

try:
    from tools.human2robot_m3 import Episode
    from tools.human2robot_m5a import (
        M5AConfig,
        M5AError,
        run_action_role_stress,
        run_resolution_stress,
        run_temporal_stress,
        run_time_view_matrix,
        validate_prerequisites,
    )
except ModuleNotFoundError:  # Direct pytest collection from the tools directory.
    from human2robot_m3 import Episode
    from human2robot_m5a import (
        M5AConfig,
        M5AError,
        run_action_role_stress,
        run_resolution_stress,
        run_temporal_stress,
        run_time_view_matrix,
        validate_prerequisites,
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


def _episode(path: Path, split: str = "train", length: int = 96) -> Episode:
    human = _trajectory(length)
    robot = human.copy()
    robot[1:] = human[:-1]
    return Episode(
        episode_id=path.stem,
        path=path,
        task="task",
        split=split,
        source_relative_path="task/episode_0.hdf5",
        human=human,
        robot=robot,
        segment_id=np.asarray([0] * (length // 2) + [1] * (length - length // 2), dtype=np.int64),
        gap_mask=np.zeros(length, dtype=bool),
    )


def test_action_role_stress_detects_all_required_perturbations(tmp_path: Path) -> None:
    report = run_action_role_stress(
        [_episode(tmp_path / "demo.hdf5")],
        M5AConfig(expected_episode_count=None, wrong_lag=8),
    )
    assert report["status"] == "passed"
    assert set(report["checks"]) == {"wrong_role", "same_frame_copy", "wrong_lag", "scale_x2"}
    assert all(item["detector_triggered"] for item in report["checks"].values())
    assert report["checks"]["wrong_lag"]["perturbed_residual_norm_median"] > 0
    assert report["checks"]["scale_x2"]["perturbed_residual_norm_median"] > 0


def test_temporal_stress_detects_drop_jitter_pause_and_step_jump(tmp_path: Path) -> None:
    report = run_temporal_stress(
        [_episode(tmp_path / "demo.hdf5")],
        M5AConfig(expected_episode_count=None, frame_drop_every=8),
    )
    assert report["status"] == "passed"
    assert report["baseline_detector_clear"] is True
    assert report["all_required_detectors_triggered"] is True
    assert report["gap_crossing_zero"] is True


def test_all_six_time_views_are_nonempty_and_segment_safe(tmp_path: Path) -> None:
    report = run_time_view_matrix(
        [_episode(tmp_path / "demo.hdf5")],
        M5AConfig(expected_episode_count=None),
    )
    assert report["status"] == "passed"
    assert len(report["results"]) == 6
    assert all(item["gap_crossing_count"] == 0 for item in report["results"])


def test_resolution_stress_is_read_only_and_phase_retrieval_invariant(tmp_path: Path) -> None:
    path = tmp_path / "demo.hdf5"
    episode = _episode(path, length=12)
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as file:
        demo = file.create_group("data/demo_0")
        metadata = demo.create_group("metadata")
        human = metadata.create_group("human")
        human.create_dataset(
            "images", data=rng.integers(0, 256, size=(12, 240, 426, 3), dtype=np.uint8)
        )
        obs = demo.create_group("obs")
        obs.create_dataset(
            "robot_images", data=rng.integers(0, 256, size=(12, 240, 426, 3), dtype=np.uint8)
        )
    report = run_resolution_stress(
        [episode],
        M5AConfig(expected_episode_count=None, sampled_frames_per_stream=2),
        ["normalized_segment_phase", "task_pool_membership"],
    )
    assert report["status"] == "passed"
    assert report["checks"]["action_contract_sha256_unchanged"] is True
    assert report["checks"]["current_phase_retrieval_exactly_image_independent"] is True
    assert report["claim_boundary"]["visual_retrieval_conclusion"].startswith("NEEDS_EXPERIMENT")


def test_prerequisites_reject_deployment_adapter(tmp_path: Path) -> None:
    bindings = {
        "time_view_id": "nominal_camera_30hz_segmented",
        "pool_action_view_id": "human_hand_robot_frame_raw",
        "query_action_view_id": "robot_ee_observed_t_plus_1_bc_proxy",
        "action_alignment_id": "train_only_tplus1_query_anchor_se3_identity_scale_v1",
        "query_command_status": "unverified",
    }
    m3_path = tmp_path / "m3.json"
    m4_path = tmp_path / "m4.json"
    config_path = tmp_path / "m4_config.json"
    m3_path.write_text(json.dumps({"status": "passed", "gates": {"B": "passed"}}))
    m4_path.write_text(
        json.dumps(
            {
                "status": "launched",
                "gate_c": "pending",
                "m6_rollout_approved": False,
                "deployment_command_adapter_id": "forbidden-adapter",
                "bindings": bindings,
            }
        )
    )
    config_path.write_text(
        json.dumps({"bindings": bindings, "deployment_command_adapter_id": None})
    )
    config = M5AConfig(
        m3_report_path=m3_path,
        m4_report_path=m4_path,
        m4_config_path=config_path,
        expected_episode_count=None,
    )
    with pytest.raises(M5AError, match="deployment command adapter"):
        validate_prerequisites(config)
