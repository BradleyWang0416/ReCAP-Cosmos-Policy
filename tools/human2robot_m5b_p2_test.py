from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.human2robot_m5b_p2 import (
    BATCH_PER_DP_RANK,
    FIXED_DP_WORLD_SIZE,
    FIXED_WORLD_SIZE,
    FORMAL_SEEDS,
    INITIALIZATION_CHECKPOINT_PATH,
    LEARNED_METHODS,
    MAIN_EXPERIMENT_ID,
    MAX_OPTIMIZER_STEPS,
    PROTOCOL_SHA256,
    REQUIRED_EXPERIMENT_IDS,
    SAVE_EVERY_STEPS,
    TOKENIZER_CHECKPOINT_PATH,
    P2Error,
    base_bindings,
    exclusive_execution_lock,
    main_training_cells,
    protocol_experiment_coverage,
    queue_implemented_main_subset,
    run_cell,
    update_master_acceptance,
    validate_binding_keys,
    validate_dcp_checkpoint,
    validate_execution_supplement_proposal,
    validate_frozen_cell_registry,
    validate_frozen_execution_supplement,
    verify_runtime_binding,
)


def test_main_training_cells_are_exactly_three_methods_by_three_seeds() -> None:
    cells = main_training_cells()
    assert len(cells) == len(LEARNED_METHODS) * len(FORMAL_SEEDS) == 9
    assert len({cell.cell_id for cell in cells}) == 9
    assert {(cell.method_id, cell.seed) for cell in cells} == {
        (method, seed) for method in LEARNED_METHODS for seed in FORMAL_SEEDS
    }
    assert {cell.experiment_id for cell in cells} == {MAIN_EXPERIMENT_ID}


def test_protocol_coverage_never_overclaims_main_subset() -> None:
    protocol = {
        "experiment_matrix": [
            {"experiment_id": experiment_id} for experiment_id in REQUIRED_EXPERIMENT_IDS
        ]
    }
    coverage = protocol_experiment_coverage(
        protocol,
        execution_spec_frozen=True,
        full_cell_registry_bound=True,
    )
    assert coverage["checkpoint_execution_implemented"] == [MAIN_EXPERIMENT_ID]
    assert coverage["full_protocol_matrix_implemented"] is False
    assert set(coverage["checkpoint_or_evaluation_execution_not_yet_implemented"]) == set(
        REQUIRED_EXPERIMENT_IDS[1:]
    )
    assert coverage["execution_supplement_status"] == "frozen_approved_execution_spec"
    assert coverage["full_execution_spec_frozen"] is True
    assert coverage["full_cell_registry_bound"] is True
    assert coverage["unresolved_execution_decisions"] == []
    assert coverage["resolved_execution_decision_ids"] == [
        "P2-SCOPE-01",
        "P2-NONLEARNED-01",
        "P2-REP-01",
        "P2-RET-01",
        "P2-VARIANTS-01",
        "P2-EVAL-01",
    ]


def test_execution_supplement_proposal_is_nonformal_and_self_consistent() -> None:
    workspace = Path(__file__).resolve().parents[1]
    evidence = validate_execution_supplement_proposal(workspace)
    assert evidence["status"] == "PROPOSED_UNAPPROVED_NOT_FORMAL_EVIDENCE"
    assert evidence["formal_queue_allowed"] is False
    assert evidence["p2_acceptance_allowed"] is False
    assert evidence["candidate_unique_learned_checkpoint_count"] == 48
    assert evidence["candidate_registry_cell_count"] == 202


def test_frozen_supplement_and_registry_are_bound_but_still_nonformal() -> None:
    workspace = Path(__file__).resolve().parents[1]
    supplement = validate_frozen_execution_supplement(workspace)
    registry = validate_frozen_cell_registry(workspace)
    assert supplement["status"] == "frozen_approved_execution_spec"
    assert supplement["formal_queue_allowed"] is False
    assert supplement["p2_acceptance_allowed"] is False
    assert supplement["cell_count"] == 202
    assert registry["status"] == "frozen_pending_execution"
    assert registry["formal_queue_allowed"] is False
    assert registry["p2_acceptance_allowed"] is False
    assert registry["cell_count"] == 202
    assert registry["counts"] == {
        "learned_training_checkpoint": 48,
        "nonlearned_method_artifact": 3,
        "checkpoint_linked_evaluation": 147,
        "aggregate_report": 4,
    }


def _synthetic_checkpoint(path: Path, world_size: int) -> None:
    for component in ("model", "optim", "scheduler", "trainer"):
        component_path = path / component
        component_path.mkdir(parents=True)
        (component_path / ".metadata").write_bytes(b"metadata")
        for rank in range(world_size):
            (component_path / f"__{rank}_0.distcp").write_bytes(f"rank={rank}".encode())


def test_dcp_checkpoint_requires_every_rank_and_component(tmp_path: Path) -> None:
    checkpoint = tmp_path / "iter_000007000"
    _synthetic_checkpoint(checkpoint, world_size=2)
    evidence = validate_dcp_checkpoint(checkpoint, expected_world_size=2)
    assert set(evidence["components"]) == {"model", "optim", "scheduler", "trainer"}
    (checkpoint / "model/__1_0.distcp").unlink()
    with pytest.raises(P2Error, match="rank files"):
        validate_dcp_checkpoint(checkpoint, expected_world_size=2)


def test_runtime_binding_checks_actual_world_size_seed_batch_and_steps(tmp_path: Path) -> None:
    cell = main_training_cells()[0]
    code_sha256 = "c" * 64
    binding_path = tmp_path / "runtime.json"
    binding = {
        "cell_id": cell.cell_id,
        "experiment_id": cell.experiment_id,
        "method_id": cell.method_id,
        "protocol_file_sha256": PROTOCOL_SHA256,
        "code_sha256": code_sha256,
        "actual": {
            "world_size": FIXED_WORLD_SIZE,
            "data_parallel_world_size": FIXED_DP_WORLD_SIZE,
            "seed": cell.seed,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "batch_size_per_data_parallel_rank": BATCH_PER_DP_RANK,
            "checkpoint_save_every_steps": SAVE_EVERY_STEPS,
            "gradient_accumulation_steps": 1,
            "sampler_seed": cell.seed,
            "H_steps": 8,
            "K_steps": 8,
            "initialization_checkpoint_path": str(INITIALIZATION_CHECKPOINT_PATH),
            "tokenizer_checkpoint_path": str(TOKENIZER_CHECKPOINT_PATH),
        },
        "optimization": {
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "weight_decay": 0.1,
            "betas": [0.9, 0.999],
            "load_training_state": False,
            "load_ema_to_reg": True,
        },
        "environment": {
            "offline_auto_download_disabled": True,
            "huggingface_offline": True,
            "transformers_offline": True,
            "wandb_disabled": True,
        },
    }
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    assert verify_runtime_binding(binding_path, cell, code_sha256)["actual"]["seed"] == cell.seed
    binding["actual"]["data_parallel_world_size"] = 1
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(P2Error, match="DP world size"):
        verify_runtime_binding(binding_path, cell, code_sha256)


def test_required_binding_order_and_full_gate_boundary() -> None:
    cell = main_training_cells()[0]
    bindings = base_bindings(cell, "d" * 64)
    validate_binding_keys(bindings)
    master = {
        "implemented_main_training_cells": [{"status": "completed"} for _ in range(9)],
        "protocol_experiment_coverage": {
            "full_protocol_matrix_implemented": False,
            "full_execution_spec_frozen": False,
            "full_cell_registry_bound": False,
        },
        "claim_boundary": {},
    }
    update_master_acceptance(master)
    assert master["acceptance"]["implemented_cells_complete"] is True
    assert master["acceptance"]["p2_gate_passed"] is False
    assert master["status"] == "pending"
    assert master["formal_result"] is False

    master["protocol_experiment_coverage"].update(
        {
            "full_protocol_matrix_implemented": True,
            "full_execution_spec_frozen": True,
        }
    )
    update_master_acceptance(master)
    assert master["acceptance"]["p2_gate_passed"] is False
    assert master["acceptance"]["full_cell_registry_bound"] is False


def test_partial_main_launch_requires_explicit_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(P2Error, match="only an implemented M5B-MAIN-01 subset"):
        run_cell(tmp_path, "any-cell")
    with pytest.raises(P2Error, match="cannot pass full P2"):
        queue_implemented_main_subset(tmp_path)


def test_execution_lock_rejects_a_second_fixed_gpu_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "p2_execution.lock"
    with exclusive_execution_lock(lock_path, "first"):
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        assert owner["purpose"] == "first"
        with pytest.raises(P2Error, match="Another M5B-P2 execution owns"):
            with exclusive_execution_lock(lock_path, "second"):
                raise AssertionError("unreachable")
