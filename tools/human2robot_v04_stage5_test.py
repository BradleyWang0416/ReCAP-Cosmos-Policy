from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from cosmos_policy.config.experiment.human2robot_v04_experiment_configs import (
    ALL_HUMAN2ROBOT_V04_CONFIGS,
    METHODS,
)
from cosmos_policy.datasets.human2robot_dataset import align_pool_chunk
from cosmos_policy.datasets.human2robot_v04_dataset import build_human2robot_v04_dataset
from cosmos_policy.datasets.human2robot_v04_dataset import V04_DATASET_KWARGS, _formal_dataset_kwargs
from cosmos_policy.config.experiment.pusht_experiment_configs import (
    cosmos_predict2p5_2b_480p_pusht_ret_100,
)
from tools import human2robot_v04_experiment as interface
from tools import human2robot_v04_stage5 as stage5


def test_three_method_configs_share_the_frozen_optimization_contract() -> None:
    assert len(ALL_HUMAN2ROBOT_V04_CONFIGS) == len(METHODS) == 3
    reference = ALL_HUMAN2ROBOT_V04_CONFIGS[0]
    for config, method in zip(ALL_HUMAN2ROBOT_V04_CONFIGS, METHODS, strict=True):
        assert config["trainer"] == reference["trainer"]
        assert config["model"] == reference["model"]
        assert config["optimizer"] == reference["optimizer"]
        assert config["scheduler"] == reference["scheduler"]
        assert config["checkpoint"] == reference["checkpoint"]
        assert config["dataloader_train"]["batch_size"] == 25
        assert config["dataloader_train"]["dataset"]["_target_"] is build_human2robot_v04_dataset
        assert config["dataloader_train"]["dataset"]["method_id"] == method
        assert config["job"]["name"] == stage5.CONFIG_BY_METHOD[method]


def test_vectorized_alignment_matches_the_frozen_scalar_alignment() -> None:
    rng = np.random.default_rng(20260711)
    pool = rng.normal(size=(5, 8, 10))
    current = rng.normal(size=(5, 10))
    actual = stage5._align_pool_batch(pool, current)
    expected = np.stack([align_pool_chunk(item, anchor) for item, anchor in zip(pool, current, strict=True)])
    assert np.allclose(actual, expected, atol=1e-10)


def test_v04_factory_quarantines_parent_dataset_fields_and_rejects_unknowns() -> None:
    dataset_config = dict(ALL_HUMAN2ROBOT_V04_CONFIGS[0]["dataloader_train"]["dataset"])
    dataset_config.pop("_target_")
    dataset_config["chunk_size"] = 8
    assert set(_formal_dataset_kwargs(dataset_config)) == set(dataset_config) - {"chunk_size"}
    dataset_config["unregistered_parent_field"] = True
    try:
        _formal_dataset_kwargs(dataset_config)
    except Exception as error:
        assert "unknown inherited dataset kwargs" in str(error)
    else:
        raise AssertionError("Unknown recursively merged dataset field was not rejected")


def test_resolved_parent_config_is_accepted_only_at_the_v04_factory_boundary() -> None:
    resolved = OmegaConf.merge(cosmos_predict2p5_2b_480p_pusht_ret_100, ALL_HUMAN2ROBOT_V04_CONFIGS[0])
    dataset_config = dict(resolved["dataloader_train"]["dataset"])
    dataset_config.pop("_target_")
    assert set(_formal_dataset_kwargs(dataset_config)) == V04_DATASET_KWARGS


def test_checkpoint_retention_keeps_two_rolling_points_and_final(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    for step in range(1000, 7001, 1000):
        (checkpoint_root / f"iter_{step:09d}").mkdir()
    (checkpoint_root / "latest_checkpoint.txt").write_text("iter_000007000\n", encoding="utf-8")
    removed = stage5.prune_checkpoints(checkpoint_root)
    assert set(removed) == {
        "iter_000001000",
        "iter_000002000",
        "iter_000003000",
        "iter_000004000",
    }
    assert {path.name for path in checkpoint_root.glob("iter_*")} == {
        "iter_000005000",
        "iter_000006000",
        "iter_000007000",
    }


def test_train_dispatch_enters_stage5_only_after_preflight(monkeypatch) -> None:
    args = interface.build_parser().parse_args(["train", "--method", "no_retrieval", "--execute"])
    monkeypatch.setattr(interface.frozen, "build_preflight", lambda workspace: {"status": "PASSED", "blockers": []})
    monkeypatch.setattr(
        stage5,
        "run_stage5",
        lambda method, **kwargs: {"status": "PASSED", "method": method, "training_started": True},
    )
    result = interface.dispatch(args, workspace=Path("/workspace"))
    assert result["status"] == "PASSED"
    assert result["method"] == "no_retrieval"
    assert result["training_started"] is True
