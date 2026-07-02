# Data & Checkpoint Manifest (Hugging Face)

Two HF repos, both named `ReCAP-Cosmos2.5-pusht` (a model repo and a dataset repo can share
a name — they differ by `--repo-type`), owned by `Jeongeun`. Upload
scripts: `hf_upload_dataset.sh` (`EVAL_ONLY=1` to skip training data), `hf_upload_checkpoint.sh`
(both take `HF_USER=<org>`).

The dataset root is `$BASE_DATASETS_DIR/PushT-Cosmos-Policy/`. Source paths below are where
these files currently live on the training machine (`/mnt/ddn/dataset/PushT-Cosmos-Policy/`).

> **Cleanup:** the on-disk `success_only/` is ~2.5 GB and holds ~140 unused rotation/variant
> pools (rot±5/±10/±20/±25/±35/±40/±50/±55/±90/±105/±120/±135/±150/±165/±180, `tri_color_0`,
> `tri_default_0`, `tri_default_predict2_0`) plus 20 unused `.npz` and a 102 MB
> `reason1_embeddings.pkl`. Only the **57 pools (~655 MB)** below are used by the released
> model. The upload script stages just those.

---

## 1. Dataset repo — [`Jeongeun/ReCAP-Cosmos2.5-pusht`](https://huggingface.co/datasets/Jeongeun/ReCAP-Cosmos2.5-pusht)

### Required for EVALUATION (live retrieval pools + stats) — 45 dirs
`success_only/<pool>_{0,1,2,3,4}` for each pool, used per visual config:

| pool | visual_config |
|---|---|
| `base`          | tri_default |
| `goal_flipped`  | tri_goal_flipped |
| `rot0`          | tri_rot0 |
| `rot15` / `rot-15` | tri_rot15 / tri_rot-15 |
| `rot30` / `rot-30` | tri_rot30 / tri_rot-30 |
| `rot60` / `rot-60` | tri_rot60 / tri_rot-60 |

Plus `success_only/`: `t5_embeddings.pkl`, `dataset_statistics.json`,
`delta_dataset_statistics.json` (required for residual eval), `dataset_statistics_post_norm.json`.

### Additionally required to RETRAIN

Retrieval-source extra shards: `success_only/base_5`, `success_only/goal_flipped_5`.

Query splits (the exact suites referenced by the retrieval `.npz` / top-100 allowlists):
- `success_only/tri_default_predict2_1`
- `success_only/tri_default_predict2p5_distilled_0`, `…_distilled_1`
- `success_only/tri_default_predict2p5_no_pred_0`, `…_no_pred_1`
- `success_only/tri_goal_0` … `tri_goal_4`

Allowlists: `success_only/episode_action_error_ranking_tri_default_p.json`,
`success_only/episode_action_error_ranking_tri_goal.json`.

**Matching retrieval `.npz`** (repo root, NOT under `success_only/`):
- `retrieval_results_state_action_tri_default_p_base.npz`
  — 11 990 queries from the 5 `tri_default_predict2*` suites → matches in `base_0..5`
- `retrieval_results_state_action_tri_goal_goal_flipped.npz`
  — 14 681 queries from `tri_goal_0..4` → matches in `goal_flipped_0..5`

> These two are the configured npz for `pusht_ret_dataset_top100_residual`
> (`pusht_experiment_configs.py`). The other 20 `.npz` at the source root are from earlier
> retrieval experiments and are **not** used. Eval recomputes retrieval live and needs no npz.

---

## 2. Model repo — [`Jeongeun/ReCAP-Cosmos2.5-pusht`](https://huggingface.co/Jeongeun/ReCAP-Cosmos2.5-pusht)

Source: `/mnt/ddn/tmp/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_480p_pusht_ret_top100_residual/checkpoints/model_000007000.pt` (≈3.9 GB)

Upload: `model_000007000.pt` + `dataset_statistics.json` + `delta_dataset_statistics.json`
+ `t5_embeddings.pkl` (bundled for convenience).

The Predict2.5 video backbone weights are pulled automatically from the public
`nvidia/Cosmos-Predict2.5-2B` HF repo at load time — do not re-upload them.
