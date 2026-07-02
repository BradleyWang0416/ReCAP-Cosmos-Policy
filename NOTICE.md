# Attribution

This repository is a fork / derivative work and is distributed under the Apache
License 2.0 (see [LICENSE](LICENSE)). It combines code from the upstream projects below
with original additions (the PushT environment integration and the retrieval-augmented
"PushT-RAG" dataset, policy model, and evaluation).

## Upstream sources

- **Cosmos Policy** — https://github.com/NVlabs/cosmos-policy
  Policy framework (training/eval harness, base policy models, config system).

- **Cosmos-Predict2.5** — https://github.com/nvidia-cosmos/cosmos-predict2.5
  Video-world backbone under `cosmos_policy/_src/` (Predict2.5 rectified-flow models,
  tokenizers, schedulers). Backbone weights are downloaded from the public
  `nvidia/Cosmos-Predict2.5-2B` Hugging Face repo at runtime.

Copyright for the upstream code remains with NVIDIA CORPORATION & AFFILIATES.
See [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for third-party license texts.

## Original additions in this fork

- PushT environment + evaluation: `cosmos_policy/experiments/robot/pusht/`
- Retrieval-augmented (RAG) dataset / model / eval:
  - `cosmos_policy/datasets/pusht_dataset_ret.py`
  - `cosmos_policy/models/policy_video2world_model_pusht_ret.py`
  - `cosmos_policy/experiments/robot/pusht_ret/`
- PushT experiment configs: `cosmos_policy/config/experiment/pusht_experiment_configs.py`
