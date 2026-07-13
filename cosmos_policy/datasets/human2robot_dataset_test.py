from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cosmos_policy.datasets.human2robot_dataset import Human2RobotFormalDataset
from cosmos_policy.models.human2robot_adapter import validate_human2robot_batch

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "data/Human2Robot/canonical/v3"
VIEW = (
    ROOT
    / "data/Human2Robot/derived/views/nominal_camera_30hz_segmented"
    / "human_hand_robot_frame_raw/robot_ee_observed_t_plus_1_bc_proxy"
    / "train_only_tplus1_query_anchor_se3_identity_scale_v1"
)
M3 = ROOT / "data/Human2Robot/derived/m3_v03/m3_validation_report.json"
M4 = ROOT / "data/Human2Robot/derived/m4_v03/m4_launch_report.json"
PROTOCOL = ROOT / "方案/v03/M5B_formal_acceptance_protocol_v1.json"


def _dataset(method: str = "recap_hand_ret", seed: int = 20260711, split: str = "train"):
    return Human2RobotFormalDataset(
        canonical_root=CANONICAL,
        main_view_path=VIEW,
        m3_report_path=M3,
        m4_report_path=M4,
        protocol_path=PROTOCOL,
        split=split,
        method_id=method,
        seed=seed,
        use_image_aug=False,
    )


def test_formal_train_dataset_matches_frozen_m4_windows() -> None:
    dataset = _dataset()
    assert len(dataset) == 968
    manifest = dataset.contract_manifest()
    assert manifest["window_count"] == 968
    assert manifest["H_steps"] == manifest["K_steps"] == 8
    assert manifest["query_command_status"] == "unverified"
    assert manifest["deployment_command_adapter_id"] is None


def test_formal_sample_has_real_37_frame_10d_contract() -> None:
    sample = _dataset()[0]
    assert sample["video"].shape == (3, 37, 224, 224)
    assert sample["video"].dtype == torch.uint8
    assert sample["actions"].shape == (8, 10)
    assert sample["retrieved_actions"].shape == (8, 10)
    assert sample["proprio"].shape == (10,)
    assert sample["target_representation"] == "residual"
    torch.testing.assert_close(sample["raw_residual"], sample["raw_query_target"] - sample["raw_aligned_pool"])
    assert all(validate_human2robot_batch(sample).values())


def test_default_collation_preserves_contract() -> None:
    batch = next(iter(DataLoader(_dataset(), batch_size=2, shuffle=False, num_workers=0)))
    assert batch["video"].shape == (2, 3, 37, 224, 224)
    assert batch["actions"].shape == (2, 8, 10)
    assert all(validate_human2robot_batch(batch).values())


def test_method_semantics_are_separate_and_no_retrieval_is_zeroed() -> None:
    recap = _dataset("recap_hand_ret")[0]
    co = _dataset("co_training")[0]
    no_ret = _dataset("no_retrieval")[0]
    assert recap["target_representation"] == "residual"
    assert co["target_representation"] == "absolute"
    assert no_ret["target_representation"] == "absolute"
    assert no_ret["has_ret_data"] == 0
    assert np.count_nonzero(no_ret["retrieved_actions"].numpy()) == 0
    assert not torch.equal(recap["actions"], co["actions"])


def test_heldout_dataset_is_eval_only_and_segment_safe() -> None:
    dataset = _dataset(split="heldout")
    assert len(dataset) == 153
    sample = dataset[0]
    assert sample["split"] == "heldout"
    assert sample["gap_crossing_count"] == 0
    assert sample["strict_future_offset_view_steps"] == 1


def test_diagnostic_overfit_mode_exposes_exactly_one_train_window() -> None:
    dataset = Human2RobotFormalDataset(
        canonical_root=CANONICAL,
        main_view_path=VIEW,
        m3_report_path=M3,
        m4_report_path=M4,
        protocol_path=PROTOCOL,
        split="train",
        method_id="recap_hand_ret",
        seed=20260711,
        use_image_aug=False,
        diagnostic_overfit_window_index=7,
    )
    assert len(dataset) == 1
    assert dataset.contract_manifest()["diagnostic_overfit_window_index"] == 7
    assert dataset[0]["diagnostic_overfit_mode"] == 1
