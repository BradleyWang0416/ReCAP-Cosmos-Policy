from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cosmos_policy.datasets.human2robot_dataset import Human2RobotContractError
from cosmos_policy.datasets.human2robot_p2_dataset import Human2RobotP2Dataset, P2Window
from cosmos_policy.models.human2robot_adapter import validate_human2robot_batch

ROOT = Path(__file__).resolve().parents[2]
VIEW = (
    ROOT
    / "data/Human2Robot/derived/views/nominal_camera_30hz_segmented"
    / "human_hand_robot_frame_raw/robot_ee_observed_t_plus_1_bc_proxy"
    / "train_only_tplus1_query_anchor_se3_identity_scale_v1"
)


def _dataset(**overrides):
    kwargs = {
        "canonical_root": ROOT / "data/Human2Robot/canonical/v3",
        "main_view_path": VIEW,
        "m3_report_path": ROOT / "data/Human2Robot/derived/m3_v03/m3_validation_report.json",
        "m4_report_path": ROOT / "data/Human2Robot/derived/m4_v03/m4_launch_report.json",
        "protocol_path": ROOT / "方案/v03/M5B_formal_acceptance_protocol_v1.json",
        "supplement_path": ROOT / "方案/v03/M5B_P2_execution_supplement_v2.json",
        "p1_pool_root": ROOT / "data/Human2Robot/derived/m5b_v03/p1_human_only_pool",
        "method_id": "recap_hand_ret",
        "experiment_id": "M5B-MAIN-01",
        "variant_id": "frozen_main",
        "seed": 20260711,
        "use_image_aug": False,
        "diagnostic_window_limit": 2,
    }
    kwargs.update(overrides)
    return Human2RobotP2Dataset(**kwargs)


def test_train_phase_retrieval_is_cross_window_and_query_weight_balanced() -> None:
    dataset = _dataset(split="train")
    assert len(dataset) == 6
    assert len({item.query_index for item in dataset.examples}) == 2
    assert all(item.candidate_index is not None for item in dataset.examples)
    for query_index in {item.query_index for item in dataset.examples}:
        assert sum(
            1.0 / item.effective_k
            for item in dataset.examples
            if item.query_index == query_index
        ) == pytest.approx(1.0)
    sample = dataset[0]
    assert sample["candidate_id"] != sample["query_id"]
    assert sample["sample_weight"] == pytest.approx(1.0 / 3.0)
    assert sample["heldout_target_retrieval_feature_count"] == 0
    assert validate_human2robot_batch(sample)["formal_shapes_valid"] is True


def test_heldout_retrieval_reads_p1_human_only_pool_and_never_robot_targets() -> None:
    dataset = _dataset(split="heldout", pool_size=1, top_k=3)
    sample = dataset[0]
    assert sample["candidate_id"].startswith("p1:")
    assert len(sample["candidate_human_content_sha256"]) == 64
    assert sample["heldout_target_retrieval_feature_count"] == 0
    assert dataset.contract_manifest()["p1_selection_id"] == (
        "48e0c0f5c283a5a7b9f3de8eb6535f13f5f760cc325a81413053015fd6299afd"
    )


def test_no_retrieval_replicates_same_query_budget_but_masks_pool() -> None:
    dataset = _dataset(
        split="train",
        method_id="no_retrieval",
        target_representation="absolute",
    )
    assert len(dataset) == 6
    sample = dataset[0]
    assert sample["has_ret_data"] == 0
    assert bool(torch.all(sample["retrieved_actions"] == 0))
    assert sample["target_representation"] == "absolute"


def test_h4_k4_changes_pixel_and_latent_layout_without_padding() -> None:
    dataset = _dataset(split="train", h_steps=4, k_steps=4)
    sample = dataset[0]
    assert sample["video"].shape == (3, 33, 224, 224)
    assert sample["retrieved_actions"].shape == (4, 10)
    assert sample["actions"].shape == (4, 10)
    assert sample["action_latent_idx"] == 6
    assert validate_human2robot_batch(sample)["latent_layout_valid"] is True


def test_lag_and_future_state_remain_variant_scoped() -> None:
    with pytest.raises(Human2RobotContractError, match="lag=5 is diagnostic-only"):
        _dataset(query_offset_view_steps=5)
    with pytest.raises(Human2RobotContractError, match="future_state is only registered"):
        _dataset(target_representation="future_state")


def test_geometry_caches_per_window_without_loading_images(monkeypatch) -> None:
    dataset = object.__new__(Human2RobotP2Dataset)
    dataset.retrieval_modality = "geometry"
    dataset.index_manifest = {
        "geometry_relative_10d_mean": [0.0] * 10,
        "geometry_relative_10d_std": [1.0] * 10,
    }
    dataset._geometry_cache = {}
    window = P2Window(
        window_id="candidate-1",
        episode_id="episode-1",
        path=Path("unused.hdf5"),
        source_kind="canonical",
        task="stack",
        split="train",
        segment_number=0,
        current_row=7,
        history_rows=np.arange(8, dtype=np.int64),
        future_rows=np.arange(8, 16, dtype=np.int64),
        phase=0.5,
        human_content_sha256="a" * 64,
        pool_rank=None,
    )
    history = np.arange(80, dtype=np.float64).reshape(8, 10)
    history_reads = 0

    def read_history(_window: P2Window, _role: str) -> np.ndarray:
        nonlocal history_reads
        history_reads += 1
        return history

    def fail_image_read(*_args, **_kwargs):
        raise AssertionError("geometry ranking must not read image arrays")

    monkeypatch.setattr(dataset, "_state_history", read_history, raising=False)
    monkeypatch.setattr(dataset, "_states_and_images", fail_image_read)

    first = dataset._geometry(window, "human")
    second = dataset._geometry(window, "human")

    assert history_reads == 1
    assert first is second
