from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from tools.human2robot_m5b_p2_matrix import (
    FOUR_GPU_SUCCESSOR_SHA256,
    IO_DIAGNOSTIC_ENV,
    IO_SUCCESSOR_SHA256,
    MEMORY_SUCCESSOR_SHA256,
    PYTORCH_CUDA_ALLOC_CONF,
    file_sha256,
    validate_four_gpu_successor,
    validate_io_successor,
    validate_memory_successor,
)


WORKSPACE = Path(__file__).resolve().parents[1]


def read(relative: str) -> dict:
    return json.loads((WORKSPACE / relative).read_text(encoding="utf-8"))


def test_v1_registry_is_preserved_and_v2_adds_one_terminal_cell() -> None:
    v1_path = WORKSPACE / "方案/v03/M5B_P2_cell_registry_v1.json"
    assert file_sha256(v1_path) == "4664d036bcf6bc41e8a44fac2afe04ff6de62c2a180a29d3433bd83e46604df5"
    v1 = read("方案/v03/M5B_P2_cell_registry_v1.json")
    v2 = read("方案/v03/M5B_P2_cell_registry_v2.json")
    assert v2["cells"][:202] == v1["cells"]
    terminal = v2["cells"][-1]
    assert terminal["cell_id"] == "aggregate_report__M5B-QUAL-01__full_matrix_completion_acceptance"
    assert len(terminal["parent_artifact_ids"]) == 202
    assert set(terminal["parent_artifact_ids"]) == {cell["cell_id"] for cell in v1["cells"]}
    assert v2["formal_queue_allowed"] is False
    assert v2["p2_acceptance_allowed"] is False


def test_workspace_bounds_are_train_only_five_percent_envelope_without_clipping() -> None:
    bounds = read("方案/v03/M5B_P2_workspace_bounds_v1.json")
    raw_min = np.asarray(bounds["raw_train_xyz_min"])
    raw_max = np.asarray(bounds["raw_train_xyz_max"])
    margin = 0.05 * (raw_max - raw_min)
    np.testing.assert_allclose(bounds["margin_xyz"], margin)
    np.testing.assert_allclose(bounds["xyz_min"], raw_min - margin)
    np.testing.assert_allclose(bounds["xyz_max"], raw_max + margin)
    assert bounds["heldout_data_used"] is False
    assert bounds["source_episode_count"] == 16
    assert bounds["inference_policy"].startswith("do_not_clip")


def test_only_three_lag_entries_are_rematerialized_against_offset5_view() -> None:
    old = read("data/Human2Robot/derived/m5b_v03/p2_prepared/prepared_manifest.json")
    new = read("data/Human2Robot/derived/m5b_v03/p2_prepared_v2/prepared_manifest.json")
    old_by_id = {entry["cell_id"]: entry for entry in old["entries"]}
    lag = [entry for entry in new["entries"] if entry["spec"]["query_offset_view_steps"] == 5]
    reused = [entry for entry in new["entries"] if entry["spec"]["query_offset_view_steps"] == 1]
    assert len(lag) == 3
    assert len(reused) == 45
    assert all(entry.get("successor_materialization", "").startswith("hash-identical") for entry in reused)
    for entry in reused:
        old_entry = old_by_id[entry["cell_id"]]
        assert entry["statistics_sha256"] == old_entry["statistics_sha256"]
        assert entry["retrieval_index_sha256"] == old_entry["retrieval_index_sha256"]
    for entry in lag:
        assert entry["train_contract"]["query_count"] == 943
        assert entry["heldout_contract"]["query_count"] == 147
        assert entry["train_contract"]["time_view_manifest_sha256"] == (
            "53ab59227f865767f07fd4b8c6cea52689b7c22ec1359cedb975308644fe806d"
        )


def test_two_stage_schemas_are_frozen_but_neither_is_an_activation_or_result() -> None:
    launch = read("方案/v03/M5B_P2_launch_activation_schema_v2.json")
    final = read("方案/v03/M5B_P2_final_acceptance_schema_v2.json")
    assert launch["status"] == "frozen_schema_not_activated"
    assert launch["required_exact_values"]["p2_acceptance_allowed"] is False
    assert final["status"] == "frozen_schema_not_accepted"
    assert final["required_exact_values"]["p2_acceptance_allowed"] is True
    assert final["required_exact_values"]["terminal_report_status"] == "passed"


def test_four_gpu_successor_is_a_locked_runtime_only_delta() -> None:
    evidence = validate_four_gpu_successor(WORKSPACE)
    successor = read("方案/v03/M5B_P2_4gpu_successor_v3.json")
    runtime = successor["frozen_runtime"]
    assert evidence["file_sha256"] == FOUR_GPU_SUCCESSOR_SHA256
    assert runtime["world_size"] == runtime["data_parallel_world_size"] == 4
    assert runtime["fsdp_shard_size"] == runtime["checkpoint_rank_count"] == 4
    assert runtime["batch_size_per_data_parallel_rank"] == 25
    assert runtime["gradient_accumulation_steps"] == 2
    assert runtime["effective_global_batch_size"] == 200
    assert successor["compatibility"]["prepared_manifest"]["sha256"] == (
        "15a1bd6cc378079b04a821fe691fe293739acc827e183caa44633b76b6a629cd"
    )
    assert successor["claim_boundary"]["runtime_performance_status"] == "NEEDS_EXPERIMENT"
    assert successor["claim_boundary"]["formal_queue_allowed"] is False


def test_v3_schemas_require_four_gpu_runtime_and_leave_v2_history_intact() -> None:
    launch_v2 = read("方案/v03/M5B_P2_launch_activation_schema_v2.json")
    launch_v3 = read("方案/v03/M5B_P2_launch_activation_schema_v3.json")
    final_v3 = read("方案/v03/M5B_P2_final_acceptance_schema_v3.json")
    assert launch_v2["required_exact_values"]["gpu_count"] == 8
    required = launch_v3["required_exact_values"]
    assert required["gpu_count"] == required["world_size"] == 4
    assert required["gradient_accumulation_steps"] == 2
    assert required["effective_global_batch_size"] == 200
    assert required["four_gpu_successor_sha256"] == FOUR_GPU_SUCCESSOR_SHA256
    assert final_v3["required_exact_values"]["four_gpu_successor_sha256"] == (
        FOUR_GPU_SUCCESSOR_SHA256
    )


def test_memory_successor_is_locked_to_the_single_allocator_delta() -> None:
    evidence = validate_memory_successor(WORKSPACE)
    successor = read("方案/v03/M5B_P2_memory_successor_v4.json")
    lock = read("方案/v03/M5B_P2_memory_successor_v4.lock.json")
    assert evidence["file_sha256"] == MEMORY_SUCCESSOR_SHA256
    delta = successor["frozen_runtime_delta"]
    assert delta["environment"] == {
        "PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF
    }
    assert delta["only_authorized_runtime_delta"] is True
    assert successor["inherited_exact_runtime"]["world_size"] == 4
    assert successor["inherited_exact_runtime"]["effective_global_batch_size"] == 200
    assert successor["observed_failure_basis"]["formal_result"] is False
    assert successor["observed_failure_basis"]["completed_optimizer_iterations"] == 2
    assert lock["contains_successful_cell_result"] is False
    assert lock["passes_p2"] is False


def test_v4_schemas_require_memory_successor_and_preserve_v3_history() -> None:
    launch_v3 = read("方案/v03/M5B_P2_launch_activation_schema_v3.json")
    launch_v4 = read("方案/v03/M5B_P2_launch_activation_schema_v4.json")
    final_v4 = read("方案/v03/M5B_P2_final_acceptance_schema_v4.json")
    assert "memory_successor_sha256" not in launch_v3["required_exact_values"]
    for schema in (launch_v4, final_v4):
        required = schema["required_exact_values"]
        assert required["memory_successor_sha256"] == MEMORY_SUCCESSOR_SHA256
        assert required["pytorch_cuda_alloc_conf"] == PYTORCH_CUDA_ALLOC_CONF


def test_io_successor_forbids_full_episode_reads_and_binds_diagnostics() -> None:
    evidence = validate_io_successor(WORKSPACE)
    successor = read("方案/v03/M5B_P2_io_successor_v5.json")
    lock = read("方案/v03/M5B_P2_io_successor_v5.lock.json")
    assert evidence["file_sha256"] == IO_SUCCESSOR_SHA256
    assert evidence["indexed_hdf5_image_reads"] is True
    assert successor["frozen_data_io_delta"]["no_full_episode_image_reads"] is True
    assert successor["frozen_data_io_delta"]["model_input_semantics_changed"] is False
    assert successor["frozen_diagnostic_environment"] == IO_DIAGNOSTIC_ENV
    assert successor["inherited_exact_runtime"]["effective_global_batch_size"] == 200
    assert successor["observed_failure_basis"]["failure_kind"] == (
        "ProcessGroupNCCLWatchdogTimeout"
    )
    assert lock["contains_successful_cell_result"] is False
    assert lock["passes_p2"] is False


def test_v5_schemas_require_io_successor_and_preserve_v4_history() -> None:
    launch_v4 = read("方案/v03/M5B_P2_launch_activation_schema_v4.json")
    launch_v5 = read("方案/v03/M5B_P2_launch_activation_schema_v5.json")
    final_v5 = read("方案/v03/M5B_P2_final_acceptance_schema_v5.json")
    assert "io_successor_sha256" not in launch_v4["required_exact_values"]
    for schema in (launch_v5, final_v5):
        required = schema["required_exact_values"]
        assert required["io_successor_sha256"] == IO_SUCCESSOR_SHA256
        assert required["indexed_hdf5_image_reads"] is True
        assert required["diagnostic_environment"] == IO_DIAGNOSTIC_ENV


def test_formal_docker_launcher_exposes_only_selected_four_host_gpus() -> None:
    launcher = (WORKSPACE / "start_m5b_p2_formal_docker.sh").read_text(encoding="utf-8")
    assert 'M5B_P2_GPU_DEVICES="${M5B_P2_GPU_DEVICES:-0,1,2,3}"' in launcher
    assert '--gpus "$M5B_P2_DOCKER_GPU_REQUEST"' in launcher
    assert "--gpus all" not in launcher
    assert '-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' in launcher
    for key, value in IO_DIAGNOSTIC_ENV.items():
        assert f"-e {key}={value}" in launcher


def test_runtime_binding_runs_after_megatron_parallel_groups_are_initialized() -> None:
    from cosmos_policy.scripts import train as train_script

    launch_source = inspect.getsource(inspect.unwrap(train_script.launch))
    trainer_init = launch_source.index("trainer = config.trainer.type(config)")
    runtime_binding = launch_source.index(
        "_write_optional_human2robot_p2_runtime_binding(config)"
    )
    model_init = launch_source.index("with model_init():")
    assert trainer_init < runtime_binding < model_init
