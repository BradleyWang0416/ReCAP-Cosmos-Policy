from __future__ import annotations

import json
from pathlib import Path

from cosmos_policy.config.experiment.human2robot_experiment_configs import (
    ALL_HUMAN2ROBOT_CONFIGS,
    FORMAL_SEEDS,
    LEARNED_METHODS,
    LOCAL_POSTTRAINED_CKPT,
    LOCAL_TOKENIZER_CKPT,
    MAIN_HUMAN2ROBOT_CONFIGS,
    P2_PREPARED_ROOT,
    WORKSPACE,
)
from cosmos_policy.datasets.human2robot_p2_dataset import build_human2robot_p2_dataset
from cosmos_policy.datasets.human2robot_p2_specs import p2_training_specs


def test_all_frozen_learned_cells_have_exactly_one_config() -> None:
    specs = p2_training_specs()
    assert len(ALL_HUMAN2ROBOT_CONFIGS) == len(specs) == 48
    names = [config["job"]["name"] for config in ALL_HUMAN2ROBOT_CONFIGS]
    assert names == [spec.config_name for spec in specs]
    assert len(set(names)) == 48


def test_main_config_names_remain_backward_compatible() -> None:
    assert len(MAIN_HUMAN2ROBOT_CONFIGS) == len(FORMAL_SEEDS) * len(LEARNED_METHODS) == 9
    names = {config["job"]["name"] for config in MAIN_HUMAN2ROBOT_CONFIGS}
    expected = {
        f"cosmos_predict2p5_2b_human2robot_{method}_seed{seed}"
        for method in LEARNED_METHODS
        for seed in FORMAL_SEEDS
    }
    assert names == expected


def test_formal_optimizer_checkpoint_and_dynamic_dimensions_are_frozen() -> None:
    for config, spec in zip(ALL_HUMAN2ROBOT_CONFIGS, p2_training_specs(), strict=True):
        assert config["defaults"] == ["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100", "_self_"]
        dataset = config["dataloader_train"]["dataset"]
        assert dataset["_target_"] is build_human2robot_p2_dataset
        assert dataset["experiment_id"] == spec.experiment_id
        assert dataset["variant_id"] == spec.variant_id
        assert dataset["method_id"] == spec.method_id
        assert dataset["seed"] == spec.seed
        assert dataset["h_steps"] == spec.h_steps
        assert dataset["k_steps"] == spec.k_steps
        assert dataset["statistics_path"] == f"{P2_PREPARED_ROOT}/statistics/{spec.cell_id}.json"
        assert dataset["retrieval_index_path"] == f"{P2_PREPARED_ROOT}/indices/{spec.cell_id}.npz"
        assert config["trainer"]["max_iter"] == 7000
        assert config["trainer"]["seed"] == spec.seed
        assert config["optimizer"]["lr"] == 1e-4
        assert config["checkpoint"]["save_iter"] == 1000
        assert config["dataloader_train"]["batch_size"] == 25
        assert config["dataloader_train"]["sampler"]["seed"] == config["trainer"]["seed"]
        assert config["job"]["wandb_mode"] == "disabled"
        model = config["model"]["config"]
        assert model["action_dim"] == model["proprio_dim"] == 10
        assert model["state_t"] == spec.state_t
        assert model["min_num_conditional_frames"] == spec.action_latent_idx
        assert model["max_num_conditional_frames"] == spec.action_latent_idx
        assert model["conditional_frames_probs"] == {
            index: float(index == spec.action_latent_idx)
            for index in range(spec.action_latent_idx + 1)
        }
        assert model["tokenizer"]["chunk_duration"] == spec.tokenizer_chunk_duration


def test_specs_match_frozen_learned_cell_registry_exactly() -> None:
    registry_path = Path(WORKSPACE) / "方案" / "v03" / "M5B_P2_cell_registry_v1.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_cells = {
        cell["cell_id"]
        for cell in registry["cells"]
        if cell["artifact_kind"] == "learned_training_checkpoint"
    }
    specs = p2_training_specs()
    assert registry_cells == {spec.cell_id for spec in specs}
    assert all(cell["status"] == "pending" and cell["formal_result"] is False for cell in registry["cells"])


def test_formal_config_uses_only_existing_local_weight_paths() -> None:
    assert LOCAL_POSTTRAINED_CKPT.startswith("/DATA1/")
    assert LOCAL_TOKENIZER_CKPT.startswith("/DATA1/")
    assert "hf://" not in LOCAL_POSTTRAINED_CKPT
    assert "hf://" not in LOCAL_TOKENIZER_CKPT
