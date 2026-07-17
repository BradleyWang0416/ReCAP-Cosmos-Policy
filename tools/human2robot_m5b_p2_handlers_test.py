from __future__ import annotations

from pathlib import Path

import pytest

from cosmos_policy.config.experiment.human2robot_experiment_configs import (
    ALL_HUMAN2ROBOT_CONFIGS,
)
from tools.human2robot_m5b_p2 import build_parser as build_training_parser
from tools.human2robot_m5b_p2_handlers import (
    HandlerContractError,
    build_handler_plans,
    handler_coverage_manifest,
    require_formal_activation,
)
from tools.human2robot_m5b_p2_inference import build_parser as build_inference_parser
from tools.human2robot_m5b_p2_matrix import load_execution_matrix
from tools.human2robot_m5b_p2_matrix import (
    FOUR_GPU_BATCH_PER_DP_RANK,
    FOUR_GPU_DP_WORLD_SIZE,
    FOUR_GPU_EFFECTIVE_GLOBAL_BATCH_SIZE,
    FOUR_GPU_FSDP_SHARD_SIZE,
    FOUR_GPU_GRADIENT_ACCUMULATION_STEPS,
    FOUR_GPU_SUCCESSOR_SHA256,
    FOUR_GPU_WORLD_SIZE,
    IO_DIAGNOSTIC_ENV,
    IO_SUCCESSOR_SHA256,
    LAG_VIEW_MANIFEST_SHA256,
    MEMORY_SUCCESSOR_SHA256,
    LOGGING_SUCCESSOR_SHA256,
    PREPARED_MANIFEST_SHA256,
    PYTORCH_CUDA_ALLOC_CONF,
    SUPPLEMENT_SHA256,
    WORKSPACE_BOUNDS_SHA256,
)
from tools.human2robot_m5b_p2_reports import build_parser as build_report_parser


WORKSPACE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def matrix():
    return load_execution_matrix(WORKSPACE, verify_prepared_artifact_hashes=False)


def test_every_frozen_cell_has_a_nonlaunching_handler_plan(matrix) -> None:
    plans = build_handler_plans(matrix)
    assert len(plans) == 203
    assert all(plan.command for plan in plans.values())
    assert all(dict(plan.environment)["HF_HUB_OFFLINE"] == "1" for plan in plans.values())
    assert all(
        dict(plan.environment)["PYTORCH_CUDA_ALLOC_CONF"] == PYTORCH_CUDA_ALLOC_CONF
        for plan in plans.values()
    )
    assert all(
        all(dict(plan.environment)[key] == value for key, value in IO_DIAGNOSTIC_ENV.items())
        for plan in plans.values()
    )
    assert all(plan.can_execute_before_formal_activation is False for plan in plans.values())
    assert sum(plan.artifact_kind == "learned_training_checkpoint" for plan in plans.values()) == 48
    assert sum(plan.artifact_kind == "nonlearned_method_artifact" for plan in plans.values()) == 3
    assert sum(plan.artifact_kind == "checkpoint_linked_evaluation" for plan in plans.values()) == 147
    assert sum(plan.artifact_kind == "aggregate_report" for plan in plans.values()) == 5


def test_training_eval_and_report_commands_are_separate(matrix) -> None:
    plans = build_handler_plans(matrix)
    training = next(plan for plan in plans.values() if plan.artifact_kind == "learned_training_checkpoint")
    evaluation = next(
        plan
        for plan in plans.values()
        if plan.artifact_kind == "checkpoint_linked_evaluation" and plan.gpu_count == 1
    )
    retrieval_only = next(
        plan
        for plan in plans.values()
        if plan.artifact_kind == "nonlearned_method_artifact"
    )
    report = next(plan for plan in plans.values() if plan.artifact_kind == "aggregate_report")
    assert "tools.human2robot_m5b_p2" in training.command
    assert training.gpu_count == 4
    assert any(value.endswith("/launch_activation_v6.json") for value in training.command)
    assert "--activation-path" in training.command
    assert "tools.human2robot_m5b_p2_inference" in evaluation.command
    assert "--activation-path" in evaluation.command
    assert "--workspace-bounds-path" in evaluation.command
    assert retrieval_only.gpu_count == 0
    assert "tools.human2robot_m5b_p2_reports" in report.command


def test_coverage_manifest_never_opens_queue(matrix) -> None:
    manifest = handler_coverage_manifest(matrix)
    assert manifest["all_cells_have_handlers"] is True
    assert manifest["formal_queue_open"] is False
    assert manifest["cell_count"] == 203
    assert manifest["formal_readiness_blockers"] == []


def test_activation_fails_closed_even_if_user_supplies_optimistic_flags(matrix) -> None:
    activation = {
        "schema_version": "human2robot-m5b-p2-formal-activation-v1",
        "status": "approved",
        "formal_queue_allowed": True,
        "registry_sha256": matrix.prepared_manifest["registry_file_sha256"],
        "solver_contract_resolved": True,
        "workspace_bounds_frozen": True,
        "all_147_evaluations_bound_to_completion_report": True,
        "docker_full_suite_passed": True,
        "source_snapshot_frozen": True,
        "gpu_count": 8,
        "storage_probe_passed": True,
    }
    with pytest.raises(HandlerContractError, match="Formal activation is incomplete"):
        require_formal_activation(activation, matrix)


def test_complete_launch_activation_opens_queue_but_not_p2_acceptance(matrix) -> None:
    activation = {
        "schema_version": "human2robot-m5b-p2-launch-activation-v6",
        "status": "approved",
        "launch_authorized": True,
        "formal_queue_allowed": True,
        "p2_acceptance_allowed": False,
        "registry_sha256": matrix.prepared_manifest["registry_file_sha256"],
        "supplement_sha256": SUPPLEMENT_SHA256,
        "four_gpu_successor_sha256": FOUR_GPU_SUCCESSOR_SHA256,
        "memory_successor_sha256": MEMORY_SUCCESSOR_SHA256,
        "io_successor_sha256": IO_SUCCESSOR_SHA256,
        "logging_successor_sha256": LOGGING_SUCCESSOR_SHA256,
        "indexed_hdf5_image_reads": True,
        "diagnostic_environment": IO_DIAGNOSTIC_ENV,
        "pytorch_cuda_alloc_conf": PYTORCH_CUDA_ALLOC_CONF,
        "prepared_manifest_sha256": PREPARED_MANIFEST_SHA256,
        "workspace_bounds_sha256": WORKSPACE_BOUNDS_SHA256,
        "lag_view_manifest_sha256": LAG_VIEW_MANIFEST_SHA256,
        "native_rectified_flow_contract_resolved": True,
        "all_147_evaluations_bound_to_terminal_report": True,
        "docker_full_suite_passed": True,
        "source_snapshot_frozen": True,
        "gpu_count": FOUR_GPU_WORLD_SIZE,
        "world_size": FOUR_GPU_WORLD_SIZE,
        "data_parallel_world_size": FOUR_GPU_DP_WORLD_SIZE,
        "fsdp_shard_size": FOUR_GPU_FSDP_SHARD_SIZE,
        "batch_size_per_data_parallel_rank": FOUR_GPU_BATCH_PER_DP_RANK,
        "gradient_accumulation_steps": FOUR_GPU_GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch_size": FOUR_GPU_EFFECTIVE_GLOBAL_BATCH_SIZE,
        "storage_probe_passed": True,
        "formal_output_mount_writable": True,
        "local_weight_hashes_passed": True,
    }
    require_formal_activation(activation, matrix)
    assert activation["p2_acceptance_allowed"] is False

    activation["pytorch_cuda_alloc_conf"] = "max_split_size_mb:128"
    with pytest.raises(HandlerContractError, match="Formal activation is incomplete"):
        require_formal_activation(activation, matrix)

    activation["pytorch_cuda_alloc_conf"] = PYTORCH_CUDA_ALLOC_CONF
    activation["schema_version"] = "human2robot-m5b-p2-launch-activation-v3"
    with pytest.raises(HandlerContractError, match="Formal activation is incomplete"):
        require_formal_activation(activation, matrix)


def test_all_203_handler_commands_parse_and_all_48_training_configs_resolve(matrix) -> None:
    plans = build_handler_plans(matrix)
    config_names = {config.job.name for config in ALL_HUMAN2ROBOT_CONFIGS}
    seen_training_configs = set()
    parsed_nontraining = 0
    for plan in plans.values():
        command = list(plan.command)
        if plan.artifact_kind == "learned_training_checkpoint":
            module_index = command.index("tools.human2robot_m5b_p2")
            parsed = build_training_parser().parse_args(command[module_index + 1 :])
            assert parsed.cell_id == plan.cell_id
            config_name = matrix.bindings_by_id[plan.cell_id].training_spec.config_name
            assert config_name in config_names
            seen_training_configs.add(config_name)
        elif plan.artifact_kind in {"nonlearned_method_artifact", "checkpoint_linked_evaluation"}:
            module_index = command.index("tools.human2robot_m5b_p2_inference")
            parsed = build_inference_parser().parse_args(command[module_index + 1 :])
            assert parsed.cell_id == plan.cell_id
            parsed_nontraining += 1
        else:
            module_index = command.index("tools.human2robot_m5b_p2_reports")
            parsed = build_report_parser().parse_args(command[module_index + 1 :])
            assert parsed.cell_id == plan.cell_id
            parsed_nontraining += 1
    assert len(seen_training_configs) == 48
    assert parsed_nontraining == 155
