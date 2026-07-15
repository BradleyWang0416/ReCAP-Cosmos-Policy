"""Frozen 48-checkpoint, 3-seed Human2Robot M5B-P2 training configs.

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
from cosmos_policy.datasets.human2robot_p2_dataset import build_human2robot_p2_dataset
from cosmos_policy.datasets.human2robot_p2_specs import (
    FORMAL_SEEDS,
    P2TrainingSpec,
    p2_training_specs,
)
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
SUPPLEMENT_PATH = os.path.join(WORKSPACE, "方案", "v03", "M5B_P2_execution_supplement_v2.json")
P1_POOL_ROOT = os.path.join(HUMAN2ROBOT_ROOT, "derived", "m5b_v03", "p1_human_only_pool")
P2_PREPARED_ROOT = os.path.join(HUMAN2ROBOT_ROOT, "derived", "m5b_v03", "p2_prepared_v2")
LOCAL_POSTTRAINED_CKPT = os.environ.get(
    "COSMOS_PREDICT2P5_POSTTRAINED_CKPT",
    "/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/base/post-trained/"
    "81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
)
LOCAL_TOKENIZER_CKPT = os.environ.get(
    "COSMOS_PREDICT2P5_TOKENIZER_CKPT",
    "/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth",
)
LEARNED_METHODS = ("no_retrieval", "co_training", "recap_hand_ret")


def _prepared_path(kind: str, spec: P2TrainingSpec, suffix: str) -> str:
    return os.path.join(P2_PREPARED_ROOT, kind, f"{spec.cell_id}.{suffix}")


def _dataset(spec: P2TrainingSpec):
    return L(build_human2robot_p2_dataset)(
        canonical_root=CANONICAL_ROOT,
        main_view_path=MAIN_VIEW_PATH,
        m3_report_path=M3_REPORT_PATH,
        m4_report_path=M4_REPORT_PATH,
        protocol_path=PROTOCOL_PATH,
        supplement_path=SUPPLEMENT_PATH,
        p1_pool_root=P1_POOL_ROOT,
        split="train",
        method_id=spec.method_id,
        experiment_id=spec.experiment_id,
        variant_id=spec.variant_id,
        seed=spec.seed,
        h_steps=spec.h_steps,
        k_steps=spec.k_steps,
        window_stride=8,
        top_k=spec.top_k,
        pool_size=spec.pool_size,
        retrieval_modality=spec.retrieval_modality,
        time_view_id=spec.time_view_id,
        query_offset_view_steps=spec.query_offset_view_steps,
        target_representation=spec.target_representation,
        statistics_path=_prepared_path("statistics", spec, "json"),
        retrieval_index_path=_prepared_path("indices", spec, "npz"),
        resolution_variant="center_crop_240x424_then_resize_224",
        num_duplicates_per_image=4,
        use_image_aug=True,
        text_conditioning="disabled_zero_embedding",
        diagnostic_window_limit=None,
    )


def _formal_config(spec: P2TrainingSpec) -> LazyDict:
    dataset = _dataset(spec)
    action_latent_idx = spec.action_latent_idx
    return LazyDict(
        dict(
            defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100", "_self_"],
            trainer=dict(
                seed=spec.seed,
                max_iter=7000,
                grad_accum_iter=2,
                run_validation=False,
            ),
            model=L(CosmosPolicyHuman2RobotRetModelRectifiedFlow)(
                config=dict(
                    fsdp_shard_size=4,
                    state_t=spec.state_t,
                    min_num_conditional_frames=action_latent_idx,
                    max_num_conditional_frames=action_latent_idx,
                    conditional_frames_probs={
                        index: float(index == action_latent_idx)
                        for index in range(action_latent_idx + 1)
                    },
                    tokenizer=dict(
                        vae_pth=LOCAL_TOKENIZER_CKPT,
                        chunk_duration=spec.tokenizer_chunk_duration,
                    ),
                    text_encoder_class="T5",
                    resolution="224",
                    action_dim=10,
                    proprio_dim=10,
                    use_action_projection=False,
                    use_proprio_projection=False,
                    projection_hidden_dim=256,
                    action_loss_multiplier=16,
                    shift=5,
                    use_dynamic_shift=False,
                    use_kerras_sigma_at_inference=True,
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
                    seed=spec.seed,
                ),
                batch_size=25,
                drop_last=True,
            ),
            job=dict(group="human2robot_m5b_p2_formal", name=spec.config_name, wandb_mode="disabled"),
            upload_reproducible_setup=False,
        )
    )


ALL_HUMAN2ROBOT_CONFIGS = [_formal_config(spec) for spec in p2_training_specs()]
MAIN_HUMAN2ROBOT_CONFIGS = [
    config
    for config, spec in zip(ALL_HUMAN2ROBOT_CONFIGS, p2_training_specs(), strict=True)
    if spec.experiment_id == "M5B-MAIN-01"
]
