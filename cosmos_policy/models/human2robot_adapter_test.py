from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from cosmos_policy.datasets.human2robot_dataset import Human2RobotFormalDataset
from cosmos_policy.models.human2robot_adapter import (
    Human2RobotBatchError,
    run_one_batch_overfit_probe,
    validate_human2robot_batch,
)
from cosmos_policy.models.policy_video2world_model_human2robot_ret import (
    CosmosPolicyHuman2RobotRetModelRectifiedFlow,
)
from cosmos_policy.models.policy_video2world_model_pusht_ret import (
    CosmosPolicyPushTRetModelRectifiedFlow,
)

ROOT = Path(__file__).resolve().parents[2]
VIEW = (
    ROOT
    / "data/Human2Robot/derived/views/nominal_camera_30hz_segmented"
    / "human_hand_robot_frame_raw/robot_ee_observed_t_plus_1_bc_proxy"
    / "train_only_tplus1_query_anchor_se3_identity_scale_v1"
)


def _batch():
    dataset = Human2RobotFormalDataset(
        canonical_root=ROOT / "data/Human2Robot/canonical/v3",
        main_view_path=VIEW,
        m3_report_path=ROOT / "data/Human2Robot/derived/m3_v03/m3_validation_report.json",
        m4_report_path=ROOT / "data/Human2Robot/derived/m4_v03/m4_launch_report.json",
        protocol_path=ROOT / "方案/v03/M5B_formal_acceptance_protocol_v1.json",
        method_id="recap_hand_ret",
        seed=20260711,
        use_image_aug=False,
    )
    return next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))


def test_formal_model_is_real_retrieval_conditioned_2b_subclass() -> None:
    assert issubclass(
        CosmosPolicyHuman2RobotRetModelRectifiedFlow,
        CosmosPolicyPushTRetModelRectifiedFlow,
    )


def test_contract_hard_fails_wrong_action_shape() -> None:
    batch = _batch()
    batch["actions"] = torch.zeros(2, 7, 10)
    with pytest.raises(Human2RobotBatchError, match="Invalid action shape"):
        validate_human2robot_batch(batch)


def test_contract_hard_fails_pre_normalized_video() -> None:
    batch = _batch()
    batch["video"] = batch["video"].float() / 127.5 - 1.0
    with pytest.raises(Human2RobotBatchError, match="Invalid video dtype"):
        validate_human2robot_batch(batch)


def test_contract_hard_fails_command_upgrade() -> None:
    batch = _batch()
    batch["query_command_status"] = ["executable", "executable"]
    with pytest.raises(Human2RobotBatchError, match="Command status upgraded"):
        validate_human2robot_batch(batch)


def test_contract_hard_fails_role_target_mismatch() -> None:
    batch = _batch()
    batch["target_representation"] = ["absolute", "absolute"]
    with pytest.raises(Human2RobotBatchError, match="Method/target mismatch"):
        validate_human2robot_batch(batch)


def test_contract_accepts_frozen_h4_k4_dynamic_latent_layout() -> None:
    batch = _batch()
    batch["H_steps"] = torch.tensor([4, 4])
    batch["K_steps"] = torch.tensor([4, 4])
    batch["video"] = torch.zeros(2, 3, 33, 224, 224, dtype=torch.uint8)
    batch["actions"] = torch.zeros(2, 4, 10)
    batch["retrieved_actions"] = torch.zeros(2, 4, 10)
    for name, value in {
        "retrieved_state_latent_idx": 2,
        "retrieved_action_latent_idx": 3,
        "current_image_latent_idx": 4,
        "current_proprio_latent_idx": 5,
        "action_latent_idx": 6,
        "future_image_latent_idx": 7,
        "future_proprio_latent_idx": 8,
    }.items():
        batch[name] = torch.tensor([value, value])
    assert validate_human2robot_batch(batch)["latent_layout_valid"] is True


def test_future_state_and_lag_are_variant_scoped() -> None:
    batch = _batch()
    batch["target_representation"] = ["future_state", "future_state"]
    batch["experiment_id"] = ["M5B-REP-01", "M5B-REP-01"]
    batch["variant_id"] = ["future_state", "future_state"]
    assert validate_human2robot_batch(batch)["method_target_semantics_valid"] is True
    batch["experiment_id"] = ["M5B-MAIN-01", "M5B-MAIN-01"]
    with pytest.raises(Human2RobotBatchError, match="future_state is only registered"):
        validate_human2robot_batch(batch)

    batch = _batch()
    batch["strict_future_offset_view_steps"] = torch.tensor([5, 5])
    with pytest.raises(Human2RobotBatchError, match="lag=5 is only registered"):
        validate_human2robot_batch(batch)


def test_real_adapter_batch_overfits_on_cuda() -> None:
    assert torch.cuda.is_available(), "P0 must run inside the GPU Docker environment"
    result = run_one_batch_overfit_probe(_batch(), steps=500, device="cuda")
    assert result["passed"] is True
    assert result["final_loss"] < result["initial_loss"] * 0.01
