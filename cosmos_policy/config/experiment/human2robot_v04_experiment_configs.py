"""Frozen single-seed Human2Robot v04 stage-5 training configs."""

from __future__ import annotations

import os

from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_policy.datasets.human2robot_v04_dataset import build_human2robot_v04_dataset
from cosmos_policy.models.policy_video2world_model_human2robot_ret import (
    CosmosPolicyHuman2RobotRetModelRectifiedFlow,
)
from cosmos_policy.trainer_human2robot_v04 import Human2RobotV04Trainer


WORKSPACE = os.environ.get("RECAP_WORKSPACE", "/workspace")
RUN_ROOT = os.environ.get("HUMAN2ROBOT_V04_RUN_ROOT", "/DATA1/wxs/ReCAP_M5B_V04_RUNS")
SOURCE_ROOT = "/DATA1/wxs/DATASETS/Human2Robot/data/v1"
INPUT_ROOT = os.environ.get("HUMAN2ROBOT_V04_STAGE5_INPUT_ROOT", os.path.join(RUN_ROOT, "stage5", "training_inputs"))
STAGE4_LOCK = os.path.join(WORKSPACE, "data", "Human2Robot", "derived", "v04", "stage4_protocol_lock.json")
LOCAL_POSTTRAINED_CKPT = os.environ.get(
    "COSMOS_PREDICT2P5_POSTTRAINED_CKPT",
    "/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/base/post-trained/"
    "81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
)
LOCAL_TOKENIZER_CKPT = os.environ.get(
    "COSMOS_PREDICT2P5_TOKENIZER_CKPT",
    "/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth",
)
METHODS = ("no_retrieval", "co_training", "recap_hand_ret")


def config_name(method: str) -> str:
    return f"cosmos_predict2p5_2b_human2robot_v04_{method}_seed20260711"


def _dataset(method: str):
    method_root = os.path.join(INPUT_ROOT, method)
    return L(build_human2robot_v04_dataset)(
        source_root=SOURCE_ROOT,
        training_index_path=os.path.join(method_root, "training_index.npz"),
        statistics_path=os.path.join(method_root, "action_statistics.json"),
        stage4_protocol_lock_path=STAGE4_LOCK,
        method_id=method,
        seed=20260711,
        h_steps=8,
        k_steps=8,
        top_k=3,
        pool_size=10,
        retrieval_modality="geometry_plus_visual",
        final_image_size=224,
        num_duplicates_per_image=4,
        use_image_aug=True,
        text_conditioning="disabled_zero_embedding",
    )


def _formal_config(method: str) -> LazyDict:
    dataset = _dataset(method)
    action_latent_idx = 7
    return LazyDict(
        dict(
            defaults=["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100", "_self_"],
            trainer=dict(
                type=Human2RobotV04Trainer,
                seed=20260711,
                max_iter=7000,
                grad_accum_iter=2,
                run_validation=False,
            ),
            model=L(CosmosPolicyHuman2RobotRetModelRectifiedFlow)(
                config=dict(
                    fsdp_shard_size=4,
                    state_t=10,
                    min_num_conditional_frames=action_latent_idx,
                    max_num_conditional_frames=action_latent_idx,
                    conditional_frames_probs={index: float(index == action_latent_idx) for index in range(8)},
                    tokenizer=dict(vae_pth=LOCAL_TOKENIZER_CKPT, chunk_duration=37),
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
                    shuffle=False,
                    seed=20260711,
                ),
                batch_size=25,
                drop_last=True,
            ),
            job=dict(group="human2robot_v04_stage5", name=config_name(method), wandb_mode="disabled"),
            upload_reproducible_setup=False,
        ),
        flags={"allow_objects": True},
    )


ALL_HUMAN2ROBOT_V04_CONFIGS = [_formal_config(method) for method in METHODS]
