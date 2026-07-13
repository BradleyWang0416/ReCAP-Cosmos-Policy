from __future__ import annotations

from cosmos_policy.config.experiment.human2robot_experiment_configs import (
    ALL_HUMAN2ROBOT_CONFIGS,
    FORMAL_SEEDS,
    LEARNED_METHODS,
    LOCAL_POSTTRAINED_CKPT,
    LOCAL_TOKENIZER_CKPT,
)
from cosmos_policy.datasets.human2robot_dataset import build_human2robot_formal_dataset


def test_all_three_seed_method_configs_exist() -> None:
    assert len(ALL_HUMAN2ROBOT_CONFIGS) == len(FORMAL_SEEDS) * len(LEARNED_METHODS) == 9
    names = {config["job"]["name"] for config in ALL_HUMAN2ROBOT_CONFIGS}
    expected = {
        f"cosmos_predict2p5_2b_human2robot_{method}_seed{seed}"
        for method in LEARNED_METHODS
        for seed in FORMAL_SEEDS
    }
    assert names == expected


def test_formal_optimizer_checkpoint_and_dimensions_are_frozen() -> None:
    for config in ALL_HUMAN2ROBOT_CONFIGS:
        assert config["defaults"] == ["/experiment/cosmos_predict2p5_2b_480p_pusht_ret_100", "_self_"]
        assert config["dataloader_train"]["dataset"]["_target_"] is build_human2robot_formal_dataset
        assert config["trainer"]["max_iter"] == 7000
        assert config["trainer"]["seed"] in FORMAL_SEEDS
        assert config["optimizer"]["lr"] == 1e-4
        assert config["checkpoint"]["save_iter"] == 1000
        assert config["dataloader_train"]["batch_size"] == 25
        assert config["dataloader_train"]["sampler"]["seed"] == config["trainer"]["seed"]
        model = config["model"]["config"]
        assert model["action_dim"] == model["proprio_dim"] == 10
        assert model["state_t"] == 10
        assert model["tokenizer"]["chunk_duration"] == 37


def test_formal_config_uses_only_existing_local_weight_paths() -> None:
    assert LOCAL_POSTTRAINED_CKPT.startswith("/DATA1/")
    assert LOCAL_TOKENIZER_CKPT.startswith("/DATA1/")
    assert "hf://" not in LOCAL_POSTTRAINED_CKPT
    assert "hf://" not in LOCAL_TOKENIZER_CKPT
