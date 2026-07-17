from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.human2robot_m5b_p2 import (
    BATCH_PER_DP_RANK,
    EFFECTIVE_GLOBAL_BATCH_SIZE,
    FSDP_SHARD_SIZE,
    FIXED_DP_WORLD_SIZE,
    FIXED_WORLD_SIZE,
    FORMAL_SEEDS,
    GRADIENT_ACCUMULATION_STEPS,
    INITIALIZATION_CHECKPOINT_PATH,
    LEARNED_METHODS,
    MAIN_EXPERIMENT_ID,
    MAX_OPTIMIZER_STEPS,
    PROTOCOL_SHA256,
    REQUIRED_EXPERIMENT_IDS,
    RUNTIME_DIAGNOSTIC_FIELDS,
    SAVE_EVERY_STEPS,
    TOKENIZER_CHECKPOINT_PATH,
    P2Error,
    attempt_log_path,
    base_bindings,
    exclusive_execution_lock,
    initial_cell_record,
    main_training_cells,
    protocol_experiment_coverage,
    queue_implemented_main_subset,
    require_four_gpu_runtime_container,
    run_cell,
    training_command,
    update_master_acceptance,
    validate_binding_keys,
    validate_dcp_checkpoint,
    validate_execution_supplement_proposal,
    validate_frozen_cell_registry,
    validate_frozen_execution_supplement,
    verify_runtime_binding,
)
from tools.human2robot_m5b_p2_matrix import (
    FOUR_GPU_SUCCESSOR_SHA256,
    IO_DIAGNOSTIC_ENV,
    IO_SUCCESSOR_SHA256,
    LOGGING_SUCCESSOR_SHA256,
    MEMORY_SUCCESSOR_SHA256,
    PYTORCH_CUDA_ALLOC_CONF,
)


def test_training_cells_exactly_cover_frozen_48_specs() -> None:
    cells = main_training_cells()
    assert len(cells) == 48
    assert len({cell.cell_id for cell in cells}) == 48
    assert {cell.seed for cell in cells} == set(FORMAL_SEEDS)
    assert sum(cell.experiment_id == MAIN_EXPERIMENT_ID for cell in cells) == 9
    assert {cell.method_id for cell in cells if cell.experiment_id == MAIN_EXPERIMENT_ID} == set(
        LEARNED_METHODS
    )


def test_training_record_reserves_common_registry_artifact_for_evaluators(tmp_path: Path) -> None:
    cell = main_training_cells()[0]
    record = initial_cell_record(cell, tmp_path, "0" * 64)
    assert record["cell_id"] == cell.cell_id
    assert record["registry_artifact_path"] == str(
        tmp_path / "cells" / cell.cell_id / "artifact.json"
    )


def test_protocol_coverage_reports_all_bound_handler_families() -> None:
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
    assert set(coverage["checkpoint_execution_implemented"]) == set(REQUIRED_EXPERIMENT_IDS[:6])
    assert coverage["evaluation_or_report_execution_implemented"] == list(REQUIRED_EXPERIMENT_IDS)
    assert coverage["full_protocol_matrix_implemented"] is True
    assert coverage["checkpoint_or_evaluation_execution_not_yet_implemented"] == []
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
    assert supplement["cell_count"] == 203
    assert registry["status"] == "frozen_pending_execution"
    assert registry["formal_queue_allowed"] is False
    assert registry["p2_acceptance_allowed"] is False
    assert registry["cell_count"] == 203
    assert registry["counts"] == {
        "learned_training_checkpoint": 48,
        "nonlearned_method_artifact": 3,
        "checkpoint_linked_evaluation": 147,
        "aggregate_report": 5,
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
        "schema_version": "human2robot-m5b-p2-runtime-binding-v3",
        "cell_id": cell.cell_id,
        "experiment_id": cell.experiment_id,
        "variant_id": cell.variant_id,
        "method_id": cell.method_id,
        "protocol_file_sha256": PROTOCOL_SHA256,
        "four_gpu_successor_sha256": FOUR_GPU_SUCCESSOR_SHA256,
        "memory_successor_sha256": MEMORY_SUCCESSOR_SHA256,
        "io_successor_sha256": IO_SUCCESSOR_SHA256,
        "logging_successor_sha256": LOGGING_SUCCESSOR_SHA256,
        "code_sha256": code_sha256,
        "actual": {
            "world_size": FIXED_WORLD_SIZE,
            "data_parallel_world_size": FIXED_DP_WORLD_SIZE,
            "seed": cell.seed,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "batch_size_per_data_parallel_rank": BATCH_PER_DP_RANK,
            "checkpoint_save_every_steps": SAVE_EVERY_STEPS,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "fsdp_shard_size": FSDP_SHARD_SIZE,
            "effective_global_batch_size": EFFECTIVE_GLOBAL_BATCH_SIZE,
            "visible_cuda_device_count": FIXED_WORLD_SIZE,
            "sampler_seed": cell.seed,
            "H_steps": 8,
            "K_steps": 8,
            "top_k": cell.top_k,
            "pool_size": cell.pool_size,
            "retrieval_modality": cell.retrieval_modality,
            "time_view_id": cell.time_view_id,
            "query_offset_view_steps": cell.query_offset_view_steps,
            "target_representation": cell.target_representation,
            "initialization_checkpoint_path": str(INITIALIZATION_CHECKPOINT_PATH),
            "tokenizer_checkpoint_path": str(TOKENIZER_CHECKPOINT_PATH),
            "pytorch_cuda_alloc_conf": PYTORCH_CUDA_ALLOC_CONF,
            **{
                field_name: IO_DIAGNOSTIC_ENV[environment_name]
                for environment_name, field_name in RUNTIME_DIAGNOSTIC_FIELDS.items()
            },
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
            "pytorch_cuda_alloc_conf": PYTORCH_CUDA_ALLOC_CONF,
            **{
                field_name: IO_DIAGNOSTIC_ENV[environment_name]
                for environment_name, field_name in RUNTIME_DIAGNOSTIC_FIELDS.items()
            },
        },
    }
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    assert verify_runtime_binding(binding_path, cell, code_sha256)["actual"]["seed"] == cell.seed
    binding["actual"]["data_parallel_world_size"] = 1
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(P2Error, match="DP world size"):
        verify_runtime_binding(binding_path, cell, code_sha256)

    binding["actual"]["data_parallel_world_size"] = FIXED_DP_WORLD_SIZE
    binding["actual"]["pytorch_cuda_alloc_conf"] = "max_split_size_mb:128"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(P2Error, match="CUDA allocator"):
        verify_runtime_binding(binding_path, cell, code_sha256)


def test_four_gpu_successor_preserves_effective_global_batch() -> None:
    assert FIXED_WORLD_SIZE == FIXED_DP_WORLD_SIZE == FSDP_SHARD_SIZE == 4
    assert BATCH_PER_DP_RANK == 25
    assert GRADIENT_ACCUMULATION_STEPS == 2
    assert EFFECTIVE_GLOBAL_BATCH_SIZE == 200
    assert (
        FIXED_DP_WORLD_SIZE * BATCH_PER_DP_RANK * GRADIENT_ACCUMULATION_STEPS
        == EFFECTIVE_GLOBAL_BATCH_SIZE
    )


def test_training_command_exposes_exactly_four_logical_gpus(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    snapshot = tmp_path / "snapshot"
    torchrun = workspace / ".venv/bin/torchrun"
    config_path = snapshot / "cosmos_policy/config/config.py"
    torchrun.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    torchrun.write_text("", encoding="utf-8")
    config_path.write_text("", encoding="utf-8")
    cell = main_training_cells()[0]
    record = {
        "runtime_binding_path": str(tmp_path / "runtime.json"),
        "bindings": {"code_sha256": "e" * 64},
    }
    command, environment = training_command(
        workspace, snapshot, tmp_path / "outputs", record, cell
    )
    assert "--nproc_per_node=4" in command
    assert "--config=cosmos_policy/config/config.py" in command
    assert all(str(snapshot) not in argument for argument in command)
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert environment["PYTORCH_CUDA_ALLOC_CONF"] == PYTORCH_CUDA_ALLOC_CONF
    assert environment["HUMAN2ROBOT_P2_MEMORY_SUCCESSOR_SHA256"] == MEMORY_SUCCESSOR_SHA256
    assert environment["HUMAN2ROBOT_P2_IO_SUCCESSOR_SHA256"] == IO_SUCCESSOR_SHA256
    assert (
        environment["HUMAN2ROBOT_P2_LOGGING_SUCCESSOR_SHA256"]
        == LOGGING_SUCCESSOR_SHA256
    )
    assert all(environment[key] == value for key, value in IO_DIAGNOSTIC_ENV.items())
    assert "NCCL_DEBUG_SUBSYS" not in environment
    assert (
        environment["HUMAN2ROBOT_P2_EXPECTED_PYTORCH_CUDA_ALLOC_CONF"]
        == PYTORCH_CUDA_ALLOC_CONF
    )
    assert environment["HUMAN2ROBOT_P2_EXPECTED_GRAD_ACCUM_STEPS"] == "2"
    assert environment["HUMAN2ROBOT_P2_EXPECTED_FSDP_SHARD_SIZE"] == "4"
    assert environment["HUMAN2ROBOT_P2_EXPECTED_EFFECTIVE_GLOBAL_BATCH"] == "200"


def test_formal_logging_avoids_per_collective_info_flood() -> None:
    assert IO_DIAGNOSTIC_ENV["NCCL_DEBUG"] == "WARN"
    assert "NCCL_DEBUG_SUBSYS" not in IO_DIAGNOSTIC_ENV
    assert IO_DIAGNOSTIC_ENV["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "65536"
    assert IO_DIAGNOSTIC_ENV["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "1"
    assert IO_DIAGNOSTIC_ENV["TORCH_NCCL_DESYNC_DEBUG"] == "1"


def test_formal_retries_get_distinct_immutable_attempt_logs(tmp_path: Path) -> None:
    record = {"attempt_count": 1, "log_directory": str(tmp_path / "cell")}
    first = attempt_log_path(record)
    record["attempt_count"] = 2
    second = attempt_log_path(record)
    assert first == tmp_path / "cell/attempt_0001.log"
    assert second == tmp_path / "cell/attempt_0002.log"
    assert first != second


def test_dispatch_rejects_historical_eight_gpu_container(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    with pytest.raises(P2Error, match="requires exactly 4 visible GPUs"):
        require_four_gpu_runtime_container()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    require_four_gpu_runtime_container()


def test_required_binding_order_and_full_gate_boundary() -> None:
    cell = main_training_cells()[0]
    bindings = base_bindings(cell, "d" * 64)
    validate_binding_keys(bindings)
    master = {
        "implemented_main_training_cells": [{"status": "completed"} for _ in range(48)],
        "all_registry_cells_complete": False,
        "protocol_experiment_coverage": {
            "full_protocol_matrix_implemented": False,
            "full_execution_spec_frozen": False,
            "full_cell_registry_bound": False,
        },
        "claim_boundary": {},
    }
    update_master_acceptance(master)
    assert master["acceptance"]["learned_training_cells_complete"] is True
    assert master["acceptance"]["all_203_registry_cells_complete"] is False
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


def test_launch_requires_formal_activation_artifact(tmp_path: Path) -> None:
    missing_activation = tmp_path / "missing_launch_activation_v6.json"
    with pytest.raises(P2Error, match="Formal activation artifact missing"):
        run_cell(tmp_path, "any-cell", activation_path=missing_activation)
    with pytest.raises(P2Error, match="Formal activation artifact missing"):
        queue_implemented_main_subset(tmp_path, activation_path=missing_activation)


def test_execution_lock_rejects_a_second_fixed_gpu_owner(tmp_path: Path) -> None:
    lock_path = tmp_path / "p2_execution.lock"
    with exclusive_execution_lock(lock_path, "first"):
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        assert owner["purpose"] == "first"
        with pytest.raises(P2Error, match="Another M5B-P2 execution owns"):
            with exclusive_execution_lock(lock_path, "second"):
                raise AssertionError("unreachable")
