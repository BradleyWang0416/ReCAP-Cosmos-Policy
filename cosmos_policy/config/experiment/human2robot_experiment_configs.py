"""Formal 3-seed Human2Robot M5-B training configs.

These configs intentionally use only local paths.  They never fall back to a
Hugging Face URI and therefore cannot silently download weights at import or
launch time.
"""

from __future__ import annotations

import os

from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_policy.datasets.human2robot_dataset import build_human2robot_formal_dataset
from cosmos_policy.models.policy_video2world_model_human2robot_ret import (
    CosmosPolicyHuman2RobotRetModelRectifiedFlow,
)

WORKSPACE = os.environ.get("RECAP_WORKSPACE", "/workspace")
HUMAN2ROBOT_ROOT = os.environ.get(
    "HUMAN2ROBOT_ROOT", os.path.join(WORKSPACE, "data", "Human2Robot")
)
CANONICAL_ROOT = os.path.join(HUMAN2ROBOT_ROOT, "canonical", "v3")
MAIN_VIEW_PATH = os.path.join(
    HUMAN2ROBOT_ROOT,
    "derived",
    "views",
    "nominal_camera_30hz_segmented",
    "human_hand_robot_frame_raw",
    "robot_ee_observed_t_plus_1_bc_proxy",
    "train_only_tplus1_query_anchor_se3_identity_scale_v1",
)
M3_REPORT_PATH = os.path.join(HUMAN2ROBOT_ROOT, "derived", "m3_v03", "m3_validation_report.json")
M4_REPORT_PATH = os.path.join(HUMAN2ROBOT_ROOT, "derived", "m4_v03", "m4_launch_report.json")
PROTOCOL_PATH = os.path.join(WORKSPACE, "方案", "v03", "M5B_formal_acceptance_protocol_v1.json")
LOCAL_POSTTRAINED_CKPT = os.environ.get(
    "COSMOS_PREDICT2P5_POSTTRAINED_CKPT",
    "/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/base/post-trained/"
    "81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
)
LOCAL_TOKENIZER_CKPT = os.environ.get(
    "COSMOS_PREDICT2P5_TOKENIZER_CKPT",
    "/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth",
)
FORMAL_SEEDS = (20260711, 20260712, 20260713)
LEARNED_METHODS = ("no_retrieval", "co_training", "recap_hand_ret")


def _dataset(method_id: str, seed: int):
    return L(build_human2robot_formal_dataset)(
        canonical_root=CANONICAL_ROOT,
        main_view_path=MAIN_VIEW_PATH,
        m3_report_path=M3_REPORT_PATH,
        m4_report_path=M4_REPORT_PATH,
        protocol_path=PROTOCOL_PATH,
        split="train",
        method_id=method_id,
        seed=seed,
        horizon=8,
        window_stride=8,
        final_image_size=224,
        num_duplicates_per_image=4,
        use_image_aug=True,
        text_conditioning="disabled_zero_embedding",
        diagnostic_overfit_window_index=None,
    )


def _formal_config(method_id: str, seed: int) -> LazyDict:
    dataset = _dataset(method_id, seed)
    name = f"cosmos_predict2p5_2b_human2robot_{method_id}_seed{seed}"
    return LazyDict(
        dict(
            defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100", "_self_"],
            trainer=dict(
                seed=seed,
                max_iter=7000,
                run_validation=False,
            ),
            model=L(CosmosPolicyHuman2RobotRetModelRectifiedFlow)(
                config=dict(
                    state_t=10,
                    min_num_conditional_frames=7,
                    max_num_conditional_frames=7,
                    conditional_frames_probs={
                        0: 0.0,
                        1: 0.0,
                        2: 0.0,
                        3: 0.0,
                        4: 0.0,
                        5: 0.0,
                        6: 0.0,
                        7: 1.0,
                    },
                    tokenizer=dict(vae_pth=LOCAL_TOKENIZER_CKPT, chunk_duration=37),
                    text_encoder_class="T5",
                    resolution="224",
                    action_dim=10,
                    proprio_dim=10,
                    use_action_projection=False,
                    use_proprio_projection=False,
                    projection_hidden_dim=256,
                    action_loss_multiplier=16,
                    net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
                )
            ),
            optimizer=dict(lr=1e-4, weight_decay=0.1, betas=[0.9, 0.999]),
            scheduler=dict(
                cycle_lengths=[20000, 100000000000000],
                warm_up_steps=[500, 0],
                f_start=[1e-6, 0.06],
                f_max=[1.0, 0.06],
                f_min=[0.06, 0.06],
            ),
            checkpoint=dict(
                load_path=get_checkpoint_path(LOCAL_POSTTRAINED_CKPT),
                load_training_state=False,
                strict_resume=False,
                save_iter=1000,
                load_ema_to_reg=True,
            ),
            dataloader_train=L(DataLoader)(
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                dataset=dataset,
                sampler=L(DistributedSampler)(
                    dataset=dataset,
                    num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                    rank=L(parallel_state.get_data_parallel_rank)(),
                    shuffle=True,
                    seed=seed,
                ),
                batch_size=25,
                drop_last=True,
            ),
            job=dict(group="human2robot_m5b_formal", name=name),
            upload_reproducible_setup=False,
        )
    )


ALL_HUMAN2ROBOT_CONFIGS = [
    _formal_config(method_id, seed)
    for method_id in LEARNED_METHODS
    for seed in FORMAL_SEEDS
]
