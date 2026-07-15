#!/usr/bin/env python3
"""Fail-closed binding for the frozen 203-cell M5B-P2 successor matrix.

This module does not launch training.  It turns the frozen registry and the
materialized 48 learned-cell inputs into an executable DAG contract.  Every
runtime handler must consume these bindings rather than reconstructing cell
semantics from ad-hoc command-line defaults.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from cosmos_policy.datasets.human2robot_p2_specs import P2TrainingSpec, p2_training_specs

REGISTRY_RELATIVE_PATH = Path("方案/v03/M5B_P2_cell_registry_v2.json")
SUPPLEMENT_RELATIVE_PATH = Path("方案/v03/M5B_P2_execution_supplement_v2.json")
PREPARED_MANIFEST_RELATIVE_PATH = Path(
    "data/Human2Robot/derived/m5b_v03/p2_prepared_v2/prepared_manifest.json"
)
WORKSPACE_BOUNDS_RELATIVE_PATH = Path("方案/v03/M5B_P2_workspace_bounds_v1.json")
LAG_VIEW_MANIFEST_RELATIVE_PATH = Path(
    "data/Human2Robot/derived/views/nominal_camera_30hz_segmented/"
    "human_hand_robot_frame_raw/robot_ee_observed_t_plus_5_lag_diagnostic/"
    "train_only_tplus5_query_anchor_se3_identity_scale_v1/view_manifest.json"
)
FOUR_GPU_SUCCESSOR_RELATIVE_PATH = Path("方案/v03/M5B_P2_4gpu_successor_v3.json")
FOUR_GPU_SUCCESSOR_LOCK_RELATIVE_PATH = Path(
    "方案/v03/M5B_P2_4gpu_successor_v3.lock.json"
)
LAUNCH_ACTIVATION_SCHEMA_V3_RELATIVE_PATH = Path(
    "方案/v03/M5B_P2_launch_activation_schema_v3.json"
)
LAUNCH_ACTIVATION_SCHEMA_V3_LOCK_RELATIVE_PATH = Path(
    "方案/v03/M5B_P2_launch_activation_schema_v3.lock.json"
)
FINAL_ACCEPTANCE_SCHEMA_V3_RELATIVE_PATH = Path(
    "方案/v03/M5B_P2_final_acceptance_schema_v3.json"
)
FINAL_ACCEPTANCE_SCHEMA_V3_LOCK_RELATIVE_PATH = Path(
    "方案/v03/M5B_P2_final_acceptance_schema_v3.lock.json"
)
MEMORY_SUCCESSOR_RELATIVE_PATH = Path("方案/v03/M5B_P2_memory_successor_v4.json")
MEMORY_SUCCESSOR_LOCK_RELATIVE_PATH = Path(
    "方案/v03/M5B_P2_memory_successor_v4.lock.json"
)
LAUNCH_ACTIVATION_SCHEMA_V4_RELATIVE_PATH = Path(
    "方案/v03/M5B_P2_launch_activation_schema_v4.json"
)
LAUNCH_ACTIVATION_SCHEMA_V4_LOCK_RELATIVE_PATH = Path(
    "方案/v03/M5B_P2_launch_activation_schema_v4.lock.json"
)
FINAL_ACCEPTANCE_SCHEMA_V4_RELATIVE_PATH = Path(
    "方案/v03/M5B_P2_final_acceptance_schema_v4.json"
)
FINAL_ACCEPTANCE_SCHEMA_V4_LOCK_RELATIVE_PATH = Path(
    "方案/v03/M5B_P2_final_acceptance_schema_v4.lock.json"
)
IO_SUCCESSOR_RELATIVE_PATH = Path("方案/v03/M5B_P2_io_successor_v5.json")
IO_SUCCESSOR_LOCK_RELATIVE_PATH = Path("方案/v03/M5B_P2_io_successor_v5.lock.json")
LAUNCH_ACTIVATION_SCHEMA_V5_RELATIVE_PATH = Path(
    "方案/v03/M5B_P2_launch_activation_schema_v5.json"
)
LAUNCH_ACTIVATION_SCHEMA_V5_LOCK_RELATIVE_PATH = Path(
    "方案/v03/M5B_P2_launch_activation_schema_v5.lock.json"
)
FINAL_ACCEPTANCE_SCHEMA_V5_RELATIVE_PATH = Path(
    "方案/v03/M5B_P2_final_acceptance_schema_v5.json"
)
FINAL_ACCEPTANCE_SCHEMA_V5_LOCK_RELATIVE_PATH = Path(
    "方案/v03/M5B_P2_final_acceptance_schema_v5.lock.json"
)

REGISTRY_SHA256 = "502cc57d41c7e4829e872ac95a258d7dc1e8d0d8a27ddfc3cf0315d4d31ef2d6"
REGISTRY_CELLS_SHA256 = "cea1bbc669ff02e7c22f3511b84a136a255ea27dae60a4356876d8cd74b3be12"
SUPPLEMENT_SHA256 = "17d9fc308c50b9b7899793a4c8d3bca1eeba217053fbacb368e2f9a2e390d7ab"
PREPARED_MANIFEST_SHA256 = (
    "15a1bd6cc378079b04a821fe691fe293739acc827e183caa44633b76b6a629cd"
)
WORKSPACE_BOUNDS_SHA256 = "29e0fd8d4b58beabcf7cea7ba50488a0775a79b6f429596a3573a0bbb007eb6a"
LAG_VIEW_MANIFEST_SHA256 = "53ab59227f865767f07fd4b8c6cea52689b7c22ec1359cedb975308644fe806d"
PROTOCOL_SHA256 = "7598dfa2ac2e129f5d21a295dad23b90f63c3c8e68811da73cbcc20eb95d5ce4"
SPLIT_SHA256 = "1d3ef2377aa19938b06646f6d5fc31ec9f275fc9f37e253e1e9aa5eecdc5a968"
POOL_MANIFEST_SHA256 = "47e87be5800194de6e0ac99b47dbe23ef96a91298edbff3e9996b1484b489299"
TOKENIZER_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
FOUR_GPU_SUCCESSOR_SHA256 = "6f333136b343cee87dca3c0328a73ffd441d3059633159d02e6f514573b809ab"
FOUR_GPU_SUCCESSOR_LOCK_SHA256 = "6b57c2abd5ab5a62f0eeaed9b5436880ab7403088aae1241aa4937548888e9cb"
LAUNCH_ACTIVATION_SCHEMA_V3_SHA256 = "40c9ed15ca2f49f91ca1b049948d2297f556c52c4a1611c76f27a1fdcff8f12e"
LAUNCH_ACTIVATION_SCHEMA_V3_LOCK_SHA256 = (
    "8459e7f6e4a5a2c5bb591dfbfa2c5e889bd9d35a617a94f10de025fc32cec4e0"
)
FINAL_ACCEPTANCE_SCHEMA_V3_SHA256 = "973869770105cd9e2d34e7a4011fdcd943c157737e4e96e25a26bec4f67f618e"
FINAL_ACCEPTANCE_SCHEMA_V3_LOCK_SHA256 = (
    "b75cbbb16b493979a646a2ac5785c53ce183e803461674b9408d967f505eb483"
)
MEMORY_SUCCESSOR_SHA256 = "c5f3334e4fecc81b38466046917d7aefdf1d6eaf7b0e8344458b05cf02455bc2"
MEMORY_SUCCESSOR_LOCK_SHA256 = "a60f8def2b6a8ed08e50b7469c5689ebedf8b23ffd74786605171e8866242bc0"
LAUNCH_ACTIVATION_SCHEMA_V4_SHA256 = "3f3d1e7b55f67ebbee736b6875a64868b8cf1db4c10f7e1eab25ab8d67bace2d"
LAUNCH_ACTIVATION_SCHEMA_V4_LOCK_SHA256 = (
    "a5297b0e62e81d42faae7c679c9cb079cc135d79cfe9428c3150e46a12ac7884"
)
FINAL_ACCEPTANCE_SCHEMA_V4_SHA256 = "072b71a45c4566dbb82de972dc46c0c9cfde27b110519ccfada70ad9596fc0b1"
FINAL_ACCEPTANCE_SCHEMA_V4_LOCK_SHA256 = (
    "4908978f1428187bfb0f705dd27dbc750555eb8421f9ff8263942bc230063e38"
)
IO_SUCCESSOR_SHA256 = "844f44c8e39178582f4a1cf7dcc5d16d510aad262e22104ff11eb93666d8fde2"
IO_SUCCESSOR_LOCK_SHA256 = "67dad1418e97fe30e751fdf720a40e99cf30c96b51f9e52b61945bb058d5cb52"
LAUNCH_ACTIVATION_SCHEMA_V5_SHA256 = "aef9326caf8056fc00e289a67e1fbc12148b8c5d9b6484ef0c356dbb26bc1c03"
LAUNCH_ACTIVATION_SCHEMA_V5_LOCK_SHA256 = (
    "5f4efb57d5e350cefab97d4c023b47d5aae5f93cd82c2e43a1daffac884ed094"
)
FINAL_ACCEPTANCE_SCHEMA_V5_SHA256 = "a77f9f4800ee697aeba532f76842e49023e014ff03c86ba61392837f6effb01f"
FINAL_ACCEPTANCE_SCHEMA_V5_LOCK_SHA256 = (
    "f4eb5f418297745e51c2a3122fc6e41d25d999b01971afd75ff85d61c283b1fa"
)
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
IO_DIAGNOSTIC_ENV = {
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "65536",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
    "TORCH_NCCL_DESYNC_DEBUG": "1",
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "COLL",
    "HUMAN2ROBOT_P2_SLOW_SAMPLE_SECONDS": "5",
}

FOUR_GPU_WORLD_SIZE = 4
FOUR_GPU_DP_WORLD_SIZE = 4
FOUR_GPU_FSDP_SHARD_SIZE = 4
FOUR_GPU_BATCH_PER_DP_RANK = 25
FOUR_GPU_GRADIENT_ACCUMULATION_STEPS = 2
FOUR_GPU_EFFECTIVE_GLOBAL_BATCH_SIZE = 200

FORMAL_SEEDS = (20260711, 20260712, 20260713)
FROZEN_COUNTS = {
    "learned_training_checkpoint": 48,
    "nonlearned_method_artifact": 3,
    "checkpoint_linked_evaluation": 147,
    "aggregate_report": 5,
}
ACTION_NEGATIVE_CONTROLS = {
    "same_frame_query_negative_control": "same_frame_query_detector",
    "swapped_role_negative_control": "swapped_role_detector",
    "scale_x2_negative_control": "scale_x2_detector",
}
RESOLUTION_VARIANTS = {
    "source_240x426_then_resize_224",
    "center_crop_240x424_then_resize_224",
    "center_crop_240x424_edge_pad_240x426_then_resize_224",
}


class MatrixContractError(RuntimeError):
    """Raised when frozen execution semantics cannot be bound exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatrixContractError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_four_gpu_successor(workspace: Path) -> dict[str, Any]:
    """Validate the runtime-only four-GPU successor and both v3 schemas."""

    exact_files = {
        FOUR_GPU_SUCCESSOR_RELATIVE_PATH: FOUR_GPU_SUCCESSOR_SHA256,
        FOUR_GPU_SUCCESSOR_LOCK_RELATIVE_PATH: FOUR_GPU_SUCCESSOR_LOCK_SHA256,
        LAUNCH_ACTIVATION_SCHEMA_V3_RELATIVE_PATH: LAUNCH_ACTIVATION_SCHEMA_V3_SHA256,
        LAUNCH_ACTIVATION_SCHEMA_V3_LOCK_RELATIVE_PATH: LAUNCH_ACTIVATION_SCHEMA_V3_LOCK_SHA256,
        FINAL_ACCEPTANCE_SCHEMA_V3_RELATIVE_PATH: FINAL_ACCEPTANCE_SCHEMA_V3_SHA256,
        FINAL_ACCEPTANCE_SCHEMA_V3_LOCK_RELATIVE_PATH: FINAL_ACCEPTANCE_SCHEMA_V3_LOCK_SHA256,
    }
    for relative_path, expected_sha256 in exact_files.items():
        _require(
            file_sha256(workspace / relative_path) == expected_sha256,
            f"Frozen four-GPU successor artifact changed: {relative_path}",
        )

    successor = _read_json(workspace / FOUR_GPU_SUCCESSOR_RELATIVE_PATH)
    _require(
        successor.get("schema_version") == "human2robot-m5b-p2-four-gpu-successor-v3",
        "Four-GPU successor schema changed",
    )
    _require(
        successor.get("status") == "frozen_approved_runtime_successor",
        "Four-GPU successor is not frozen and approved",
    )
    parent = successor.get("parent", {})
    _require(
        parent.get("formal_protocol_sha256") == PROTOCOL_SHA256,
        "Four-GPU successor parent protocol changed",
    )
    _require(
        parent.get("execution_supplement_sha256") == SUPPLEMENT_SHA256,
        "Four-GPU successor parent execution supplement changed",
    )
    superseded = successor.get("supersedes_runtime_only", {})
    _require(
        superseded.get("parent_visible_gpu_count") == 8
        and superseded.get("parent_gradient_accumulation_steps") == 1,
        "Four-GPU successor does not identify the superseded runtime",
    )
    runtime = successor.get("frozen_runtime", {})
    expected_runtime = {
        "world_size": FOUR_GPU_WORLD_SIZE,
        "data_parallel_world_size": FOUR_GPU_DP_WORLD_SIZE,
        "fsdp_shard_size": FOUR_GPU_FSDP_SHARD_SIZE,
        "batch_size_per_data_parallel_rank": FOUR_GPU_BATCH_PER_DP_RANK,
        "gradient_accumulation_steps": FOUR_GPU_GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch_size": FOUR_GPU_EFFECTIVE_GLOBAL_BATCH_SIZE,
        "visible_gpu_count": FOUR_GPU_WORLD_SIZE,
        "checkpoint_rank_count": FOUR_GPU_WORLD_SIZE,
        "max_optimizer_steps": 7000,
        "save_every_optimizer_steps": 1000,
        "precision": "bfloat16",
    }
    mismatches = {
        key: {"actual": runtime.get(key), "expected": value}
        for key, value in expected_runtime.items()
        if runtime.get(key) != value
    }
    _require(not mismatches, f"Four-GPU frozen runtime changed: {mismatches}")
    _require(
        FOUR_GPU_DP_WORLD_SIZE
        * FOUR_GPU_BATCH_PER_DP_RANK
        * FOUR_GPU_GRADIENT_ACCUMULATION_STEPS
        == FOUR_GPU_EFFECTIVE_GLOBAL_BATCH_SIZE,
        "Four-GPU effective global batch derivation changed",
    )
    lock = _read_json(workspace / FOUR_GPU_SUCCESSOR_LOCK_RELATIVE_PATH)
    _require(lock.get("status") == "locked_pending_execution", "Four-GPU successor lock is open")
    _require(
        lock.get("successor_file_sha256") == FOUR_GPU_SUCCESSOR_SHA256,
        "Four-GPU successor lock binding changed",
    )
    _require(lock.get("contains_experiment_results") is False, "Successor lock contains results")
    _require(lock.get("passes_p2") is False, "Successor lock may not pass P2")
    return {
        "path": FOUR_GPU_SUCCESSOR_RELATIVE_PATH.as_posix(),
        "file_sha256": FOUR_GPU_SUCCESSOR_SHA256,
        "lock_path": FOUR_GPU_SUCCESSOR_LOCK_RELATIVE_PATH.as_posix(),
        "lock_file_sha256": FOUR_GPU_SUCCESSOR_LOCK_SHA256,
        "frozen_runtime": dict(runtime),
        "contains_experiment_results": False,
        "passes_p2": False,
    }


def validate_memory_successor(workspace: Path) -> dict[str, Any]:
    """Validate the allocator-only v4 successor and its activation schemas."""

    exact_files = {
        MEMORY_SUCCESSOR_RELATIVE_PATH: MEMORY_SUCCESSOR_SHA256,
        MEMORY_SUCCESSOR_LOCK_RELATIVE_PATH: MEMORY_SUCCESSOR_LOCK_SHA256,
        LAUNCH_ACTIVATION_SCHEMA_V4_RELATIVE_PATH: LAUNCH_ACTIVATION_SCHEMA_V4_SHA256,
        LAUNCH_ACTIVATION_SCHEMA_V4_LOCK_RELATIVE_PATH: LAUNCH_ACTIVATION_SCHEMA_V4_LOCK_SHA256,
        FINAL_ACCEPTANCE_SCHEMA_V4_RELATIVE_PATH: FINAL_ACCEPTANCE_SCHEMA_V4_SHA256,
        FINAL_ACCEPTANCE_SCHEMA_V4_LOCK_RELATIVE_PATH: FINAL_ACCEPTANCE_SCHEMA_V4_LOCK_SHA256,
    }
    for relative_path, expected_sha256 in exact_files.items():
        _require(
            file_sha256(workspace / relative_path) == expected_sha256,
            f"Frozen memory-successor artifact changed: {relative_path}",
        )

    successor = _read_json(workspace / MEMORY_SUCCESSOR_RELATIVE_PATH)
    _require(
        successor.get("schema_version") == "human2robot-m5b-p2-memory-successor-v4",
        "Memory-successor schema changed",
    )
    _require(
        successor.get("status") == "frozen_approved_runtime_memory_successor",
        "Memory-successor is not frozen and approved",
    )
    parent = successor.get("parent", {})
    _require(
        parent.get("four_gpu_successor_sha256") == FOUR_GPU_SUCCESSOR_SHA256,
        "Memory-successor parent four-GPU hash changed",
    )
    _require(
        parent.get("formal_protocol_sha256") == PROTOCOL_SHA256,
        "Memory-successor parent protocol changed",
    )
    delta = successor.get("frozen_runtime_delta", {})
    _require(delta.get("only_authorized_runtime_delta") is True, "Memory delta is not exclusive")
    _require(
        delta.get("environment") == {"PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF},
        "Memory-successor allocator environment changed",
    )
    inherited = successor.get("inherited_exact_runtime", {})
    expected_runtime = {
        "world_size": FOUR_GPU_WORLD_SIZE,
        "data_parallel_world_size": FOUR_GPU_DP_WORLD_SIZE,
        "fsdp_shard_size": FOUR_GPU_FSDP_SHARD_SIZE,
        "batch_size_per_data_parallel_rank": FOUR_GPU_BATCH_PER_DP_RANK,
        "gradient_accumulation_steps": FOUR_GPU_GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch_size": FOUR_GPU_EFFECTIVE_GLOBAL_BATCH_SIZE,
        "visible_gpu_count": FOUR_GPU_WORLD_SIZE,
        "checkpoint_rank_count": FOUR_GPU_WORLD_SIZE,
        "max_optimizer_steps": 7000,
        "save_every_optimizer_steps": 1000,
        "precision": "bfloat16",
    }
    _require(inherited == expected_runtime, "Memory-successor changed inherited runtime values")
    failure = successor.get("observed_failure_basis", {})
    _require(failure.get("formal_result") is False, "OOM basis was upgraded to a formal result")
    _require(failure.get("failure_kind") == "torch.OutOfMemoryError", "OOM basis changed")
    _require(
        failure.get("completed_optimizer_iterations") == 2,
        "OOM completed-iteration evidence changed",
    )
    lock = _read_json(workspace / MEMORY_SUCCESSOR_LOCK_RELATIVE_PATH)
    _require(lock.get("status") == "locked_pending_execution", "Memory-successor lock is open")
    _require(
        lock.get("successor_file_sha256") == MEMORY_SUCCESSOR_SHA256,
        "Memory-successor lock binding changed",
    )
    _require(lock.get("contains_successful_cell_result") is False, "Memory lock contains success")
    _require(lock.get("passes_p2") is False, "Memory-successor lock may not pass P2")
    return {
        "path": MEMORY_SUCCESSOR_RELATIVE_PATH.as_posix(),
        "file_sha256": MEMORY_SUCCESSOR_SHA256,
        "lock_path": MEMORY_SUCCESSOR_LOCK_RELATIVE_PATH.as_posix(),
        "lock_file_sha256": MEMORY_SUCCESSOR_LOCK_SHA256,
        "pytorch_cuda_alloc_conf": PYTORCH_CUDA_ALLOC_CONF,
        "contains_successful_cell_result": False,
        "passes_p2": False,
    }


def validate_io_successor(workspace: Path) -> dict[str, Any]:
    """Validate the indexed-HDF5 v5 successor and its diagnostic bindings."""

    exact_files = {
        IO_SUCCESSOR_RELATIVE_PATH: IO_SUCCESSOR_SHA256,
        IO_SUCCESSOR_LOCK_RELATIVE_PATH: IO_SUCCESSOR_LOCK_SHA256,
        LAUNCH_ACTIVATION_SCHEMA_V5_RELATIVE_PATH: LAUNCH_ACTIVATION_SCHEMA_V5_SHA256,
        LAUNCH_ACTIVATION_SCHEMA_V5_LOCK_RELATIVE_PATH: LAUNCH_ACTIVATION_SCHEMA_V5_LOCK_SHA256,
        FINAL_ACCEPTANCE_SCHEMA_V5_RELATIVE_PATH: FINAL_ACCEPTANCE_SCHEMA_V5_SHA256,
        FINAL_ACCEPTANCE_SCHEMA_V5_LOCK_RELATIVE_PATH: FINAL_ACCEPTANCE_SCHEMA_V5_LOCK_SHA256,
    }
    for relative_path, expected_sha256 in exact_files.items():
        _require(
            file_sha256(workspace / relative_path) == expected_sha256,
            f"Frozen I/O-successor artifact changed: {relative_path}",
        )

    successor = _read_json(workspace / IO_SUCCESSOR_RELATIVE_PATH)
    _require(
        successor.get("schema_version") == "human2robot-m5b-p2-io-successor-v5",
        "I/O-successor schema changed",
    )
    _require(
        successor.get("status") == "frozen_approved_data_io_successor",
        "I/O-successor is not frozen and approved",
    )
    parent = successor.get("parent", {})
    _require(
        parent.get("memory_successor_sha256") == MEMORY_SUCCESSOR_SHA256,
        "I/O-successor parent memory hash changed",
    )
    _require(
        parent.get("formal_protocol_sha256") == PROTOCOL_SHA256,
        "I/O-successor parent protocol changed",
    )
    delta = successor.get("frozen_data_io_delta", {})
    _require(delta.get("no_full_episode_image_reads") is True, "Full image reads are not forbidden")
    _require(delta.get("model_input_semantics_changed") is False, "I/O successor changes model inputs")
    _require(
        delta.get("optimizer_or_batch_semantics_changed") is False,
        "I/O successor changes optimizer or batch semantics",
    )
    _require(
        successor.get("frozen_diagnostic_environment") == IO_DIAGNOSTIC_ENV,
        "I/O diagnostic environment changed",
    )
    inherited = successor.get("inherited_exact_runtime", {})
    expected_runtime = {
        "world_size": FOUR_GPU_WORLD_SIZE,
        "data_parallel_world_size": FOUR_GPU_DP_WORLD_SIZE,
        "fsdp_shard_size": FOUR_GPU_FSDP_SHARD_SIZE,
        "batch_size_per_data_parallel_rank": FOUR_GPU_BATCH_PER_DP_RANK,
        "gradient_accumulation_steps": FOUR_GPU_GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch_size": FOUR_GPU_EFFECTIVE_GLOBAL_BATCH_SIZE,
        "visible_gpu_count": FOUR_GPU_WORLD_SIZE,
        "checkpoint_rank_count": FOUR_GPU_WORLD_SIZE,
        "max_optimizer_steps": 7000,
        "save_every_optimizer_steps": 1000,
        "precision": "bfloat16",
        "pytorch_cuda_alloc_conf": PYTORCH_CUDA_ALLOC_CONF,
    }
    _require(inherited == expected_runtime, "I/O-successor changed inherited runtime values")
    implementation = successor.get("implementation_binding", {})
    for path_key, hash_key in (
        ("dataset_path", "dataset_sha256"),
        ("regression_test_path", "regression_test_sha256"),
    ):
        source_path = workspace / str(implementation.get(path_key, ""))
        _require(source_path.is_file(), f"I/O successor implementation is missing: {source_path}")
        _require(
            file_sha256(source_path) == implementation.get(hash_key),
            f"I/O successor implementation changed: {source_path}",
        )
    failure = successor.get("observed_failure_basis", {})
    _require(failure.get("formal_result") is False, "NCCL failure was upgraded to a formal result")
    _require(
        failure.get("failure_kind") == "ProcessGroupNCCLWatchdogTimeout",
        "I/O-successor failure basis changed",
    )
    _require(failure.get("completed_optimizer_iterations") == 100, "Failure iteration changed")
    lock = _read_json(workspace / IO_SUCCESSOR_LOCK_RELATIVE_PATH)
    _require(lock.get("status") == "locked_pending_execution", "I/O-successor lock is open")
    _require(
        lock.get("successor_file_sha256") == IO_SUCCESSOR_SHA256,
        "I/O-successor lock binding changed",
    )
    _require(lock.get("contains_successful_cell_result") is False, "I/O lock contains success")
    _require(lock.get("passes_p2") is False, "I/O-successor lock may not pass P2")
    return {
        "path": IO_SUCCESSOR_RELATIVE_PATH.as_posix(),
        "file_sha256": IO_SUCCESSOR_SHA256,
        "lock_path": IO_SUCCESSOR_LOCK_RELATIVE_PATH.as_posix(),
        "lock_file_sha256": IO_SUCCESSOR_LOCK_SHA256,
        "diagnostic_environment": dict(IO_DIAGNOSTIC_ENV),
        "indexed_hdf5_image_reads": True,
        "contains_successful_cell_result": False,
        "passes_p2": False,
    }


@dataclass(frozen=True)
class FrozenCell:
    cell_id: str
    artifact_kind: str
    experiment_id: str
    variant_id: str
    method_id: str | None
    seed: int | None
    parent_artifact_ids: tuple[str, ...]
    optimizer_steps: int | None
    formal_result: bool
    status: str

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "FrozenCell":
        expected = {
            "cell_id",
            "artifact_kind",
            "experiment_id",
            "variant_id",
            "method_id",
            "seed",
            "parent_artifact_ids",
            "optimizer_steps",
            "formal_result",
            "status",
        }
        _require(set(record) == expected, f"Unexpected registry cell fields: {set(record) ^ expected}")
        parents = record["parent_artifact_ids"]
        _require(isinstance(parents, list), f"parent_artifact_ids must be a list: {record['cell_id']}")
        return cls(
            cell_id=str(record["cell_id"]),
            artifact_kind=str(record["artifact_kind"]),
            experiment_id=str(record["experiment_id"]),
            variant_id=str(record["variant_id"]),
            method_id=None if record["method_id"] is None else str(record["method_id"]),
            seed=None if record["seed"] is None else int(record["seed"]),
            parent_artifact_ids=tuple(str(item) for item in parents),
            optimizer_steps=(
                None if record["optimizer_steps"] is None else int(record["optimizer_steps"])
            ),
            formal_result=bool(record["formal_result"]),
            status=str(record["status"]),
        )


@dataclass(frozen=True)
class EvaluationBinding:
    cell_id: str
    experiment_id: str
    variant_id: str
    method_id: str
    run_seed: int
    parent_artifact_id: str
    checkpoint_cell_id: str | None
    prepared_input_cell_id: str
    target_representation: str
    retrieval_modality: str
    time_view_id: str
    h_steps: int
    k_steps: int
    top_k: int
    pool_size: int
    query_offset_view_steps: int
    resolution_variant: str
    corruption_id: str | None
    corruption_severity: str | None
    negative_control_detector: str | None
    requires_model_inference: bool


@dataclass(frozen=True)
class CellBinding:
    cell: FrozenCell
    handler_kind: str
    prepared_entry: Mapping[str, Any] | None = None
    training_spec: P2TrainingSpec | None = None
    evaluation: EvaluationBinding | None = None


@dataclass(frozen=True)
class ExecutionMatrix:
    registry: Mapping[str, Any]
    prepared_manifest: Mapping[str, Any]
    cells_by_id: Mapping[str, FrozenCell]
    bindings_by_id: Mapping[str, CellBinding]
    topological_cell_ids: tuple[str, ...]
    formal_readiness_blockers: tuple[str, ...]
    report_covered_evaluation_ids: frozenset[str]

    def cells_of_kind(self, artifact_kind: str) -> tuple[CellBinding, ...]:
        return tuple(
            self.bindings_by_id[cell_id]
            for cell_id in self.topological_cell_ids
            if self.cells_by_id[cell_id].artifact_kind == artifact_kind
        )


def _expected_cell_id(cell: FrozenCell) -> str:
    parts = [cell.artifact_kind, cell.experiment_id, cell.variant_id]
    if cell.method_id is not None:
        parts.append(cell.method_id)
    if cell.seed is not None:
        parts.append(f"seed{cell.seed}")
    return "__".join(parts)


def _topological_order(cells_by_id: Mapping[str, FrozenCell]) -> tuple[str, ...]:
    children: dict[str, list[str]] = {cell_id: [] for cell_id in cells_by_id}
    indegree = {cell_id: 0 for cell_id in cells_by_id}
    for cell in cells_by_id.values():
        for parent_id in cell.parent_artifact_ids:
            _require(parent_id in cells_by_id, f"Unknown parent {parent_id} for {cell.cell_id}")
            _require(parent_id != cell.cell_id, f"Self dependency: {cell.cell_id}")
            children[parent_id].append(cell.cell_id)
            indegree[cell.cell_id] += 1
    queue = deque(cell_id for cell_id in cells_by_id if indegree[cell_id] == 0)
    ordered: list[str] = []
    while queue:
        cell_id = queue.popleft()
        ordered.append(cell_id)
        for child_id in children[cell_id]:
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                queue.append(child_id)
    _require(len(ordered) == len(cells_by_id), "Frozen registry contains a dependency cycle")
    return tuple(ordered)


def load_frozen_registry(workspace: Path) -> tuple[dict[str, Any], dict[str, FrozenCell], tuple[str, ...]]:
    path = workspace / REGISTRY_RELATIVE_PATH
    _require(file_sha256(path) == REGISTRY_SHA256, "Frozen registry file SHA256 changed")
    registry = _read_json(path)
    _require(registry.get("schema_version") == "human2robot-m5b-p2-cell-registry-v2", "Registry schema changed")
    _require(registry.get("registry_id") == "m5b_p2_claim_centered_203_cells_v2", "Registry id changed")
    _require(registry.get("status") == "frozen_pending_execution", "Registry is not frozen pending execution")
    _require(registry.get("formal_queue_allowed") is False, "Frozen registry unexpectedly opens formal queue")
    _require(registry.get("p2_acceptance_allowed") is False, "Frozen registry unexpectedly permits P2 acceptance")
    _require(tuple(registry.get("seeds", ())) == FORMAL_SEEDS, "Frozen seeds changed")
    _require(registry.get("cell_count") == 203, "Frozen registry cell count changed")
    _require(registry.get("counts") == FROZEN_COUNTS, "Frozen artifact-kind counts changed")
    records = registry.get("cells")
    _require(isinstance(records, list), "Registry cells must be a list")
    _require(canonical_json_sha256(records) == REGISTRY_CELLS_SHA256, "Frozen cells payload changed")
    _require(registry.get("cells_payload_sha256") == REGISTRY_CELLS_SHA256, "Registry cells hash binding changed")
    cells = [FrozenCell.from_record(record) for record in records]
    cells_by_id = {cell.cell_id: cell for cell in cells}
    _require(len(cells_by_id) == 203, "Registry cell ids are not unique")
    _require(Counter(cell.artifact_kind for cell in cells) == Counter(FROZEN_COUNTS), "Cell kinds do not match frozen counts")
    for cell in cells:
        _require(cell.cell_id == _expected_cell_id(cell), f"Noncanonical cell id: {cell.cell_id}")
        _require(cell.formal_result is False, f"Pending registry cell claims formal evidence: {cell.cell_id}")
        _require(cell.status == "pending", f"Frozen cell is not pending: {cell.cell_id}")
        if cell.artifact_kind == "learned_training_checkpoint":
            _require(cell.optimizer_steps == 7000, f"Training step contract changed: {cell.cell_id}")
            _require(not cell.parent_artifact_ids, f"Training cell unexpectedly has parents: {cell.cell_id}")
        else:
            _require(cell.optimizer_steps is None, f"Non-training cell has optimizer steps: {cell.cell_id}")
        if cell.artifact_kind == "checkpoint_linked_evaluation":
            _require(len(cell.parent_artifact_ids) == 1, f"Evaluation must have one artifact parent: {cell.cell_id}")
    order = _topological_order(cells_by_id)
    return registry, cells_by_id, order


def _validate_contract_against_spec(contract: Mapping[str, Any], spec: P2TrainingSpec, split: str) -> None:
    expected = {
        "experiment_id": spec.experiment_id,
        "variant_id": spec.variant_id,
        "method_id": spec.method_id,
        "seed": spec.seed,
        "target_representation": spec.target_representation,
        "retrieval_modality": spec.retrieval_modality,
        "time_view_id": spec.time_view_id,
        "H_steps": spec.h_steps,
        "K_steps": spec.k_steps,
        "top_k": spec.top_k,
        "pool_size": spec.pool_size,
        "query_offset_view_steps": spec.query_offset_view_steps,
        "split": split,
        "protocol_file_sha256": PROTOCOL_SHA256,
        "supplement_file_sha256": SUPPLEMENT_SHA256,
        "split_sha256": SPLIT_SHA256,
        "heldout_target_retrieval_feature_count": 0,
    }
    for key, value in expected.items():
        _require(contract.get(key) == value, f"Prepared {split} contract mismatch for {spec.cell_id}: {key}")


def load_prepared_manifest(
    workspace: Path,
    *,
    verify_artifact_hashes: bool = True,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    path = workspace / PREPARED_MANIFEST_RELATIVE_PATH
    _require(file_sha256(path) == PREPARED_MANIFEST_SHA256, "Prepared manifest SHA256 changed")
    manifest = _read_json(path)
    expected_top_level = {
        "schema_version": "human2robot-m5b-p2-prepared-artifacts-v2",
        "status": "complete",
        "formal_result": False,
        "learned_cell_count": 48,
        "heldout_target_retrieval_feature_count": 0,
        "protocol_file_sha256": PROTOCOL_SHA256,
        "supplement_file_sha256": SUPPLEMENT_SHA256,
        "registry_file_sha256": REGISTRY_SHA256,
        "split_sha256": SPLIT_SHA256,
        "pool_manifest_sha256": POOL_MANIFEST_SHA256,
        "tokenizer_checkpoint_sha256": TOKENIZER_SHA256,
    }
    for key, value in expected_top_level.items():
        _require(manifest.get(key) == value, f"Prepared manifest binding mismatch: {key}")
    entries = manifest.get("entries")
    _require(isinstance(entries, list) and len(entries) == 48, "Prepared manifest must contain 48 entries")
    entries_by_id = {str(entry.get("cell_id")): entry for entry in entries}
    _require(len(entries_by_id) == 48, "Prepared manifest cell ids are not unique")

    specs_by_id = {spec.cell_id: spec for spec in p2_training_specs()}
    _require(set(entries_by_id) == set(specs_by_id), "Prepared entries do not match the 48 learned specs")
    for cell_id, spec in specs_by_id.items():
        entry = entries_by_id[cell_id]
        _require(entry.get("config_name") == spec.config_name, f"Prepared config mismatch: {cell_id}")
        _require(entry.get("spec") == asdict(spec), f"Prepared spec mismatch: {cell_id}")
        train_contract = entry.get("train_contract")
        heldout_contract = entry.get("heldout_contract")
        _require(isinstance(train_contract, dict), f"Missing train contract: {cell_id}")
        _require(isinstance(heldout_contract, dict), f"Missing heldout contract: {cell_id}")
        _validate_contract_against_spec(train_contract, spec, "train")
        _validate_contract_against_spec(heldout_contract, spec, "heldout")
        _require(train_contract.get("query_count", 0) > 0, f"Empty train query set: {cell_id}")
        _require(heldout_contract.get("query_count", 0) > 0, f"Empty heldout query set: {cell_id}")
        if verify_artifact_hashes:
            for path_key, sha_key in (
                ("retrieval_index_path", "retrieval_index_sha256"),
                ("statistics_path", "statistics_sha256"),
            ):
                artifact_path = workspace / str(entry[path_key])
                _require(artifact_path.is_file(), f"Missing prepared artifact: {artifact_path}")
                _require(
                    file_sha256(artifact_path) == entry[sha_key],
                    f"Prepared artifact hash mismatch: {artifact_path}",
                )
    return manifest, entries_by_id


def _main_recap_cell_id(seed: int) -> str:
    return f"learned_training_checkpoint__M5B-MAIN-01__frozen_main__recap_hand_ret__seed{seed}"


def _split_corruption(variant_id: str) -> tuple[str | None, str | None]:
    patterns = (
        (r"^(frame_drop)_(5pct|10pct|20pct)$", None),
        (r"^(timestamp_jitter)_(5ms|10ms|20ms)$", None),
        (r"^(pause)_(0p2s|0p5s|1p0s)$", None),
        (r"^(step_jump)_(1|5|20)$", None),
    )
    for pattern, _ in patterns:
        match = re.match(pattern, variant_id)
        if match:
            return match.group(1), match.group(2)
    return None, None


def _evaluation_binding(
    cell: FrozenCell,
    cells_by_id: Mapping[str, FrozenCell],
    specs_by_id: Mapping[str, P2TrainingSpec],
) -> EvaluationBinding:
    _require(cell.seed is not None and cell.method_id is not None, f"Incomplete evaluation identity: {cell.cell_id}")
    parent_id = cell.parent_artifact_ids[0]
    parent = cells_by_id[parent_id]
    if parent.artifact_kind == "learned_training_checkpoint":
        _require(parent_id in specs_by_id, f"Evaluation parent has no training spec: {cell.cell_id}")
        source_spec = specs_by_id[parent_id]
        checkpoint_cell_id: str | None = parent_id
        prepared_input_cell_id = parent_id
        requires_model = True
    else:
        _require(
            parent.artifact_kind == "nonlearned_method_artifact" and cell.method_id == "retrieval_only",
            f"Unsupported non-checkpoint evaluation parent: {cell.cell_id}",
        )
        prepared_input_cell_id = _main_recap_cell_id(cell.seed)
        source_spec = specs_by_id[prepared_input_cell_id]
        checkpoint_cell_id = None
        requires_model = False

    values: dict[str, Any] = {
        "target_representation": source_spec.target_representation,
        "retrieval_modality": source_spec.retrieval_modality,
        "time_view_id": source_spec.time_view_id,
        "h_steps": source_spec.h_steps,
        "k_steps": source_spec.k_steps,
        "top_k": source_spec.top_k,
        "pool_size": source_spec.pool_size,
        "query_offset_view_steps": source_spec.query_offset_view_steps,
        "resolution_variant": "center_crop_240x424_then_resize_224",
    }
    corruption_id, corruption_severity = _split_corruption(cell.variant_id)
    detector = ACTION_NEGATIVE_CONTROLS.get(cell.variant_id)

    if cell.experiment_id == "M5B-MAIN-01":
        if cell.variant_id == "main_comparison_pool10":
            values["pool_size"] = 10
        elif cell.variant_id.startswith("pool_growth_pool"):
            values["pool_size"] = int(cell.variant_id.removeprefix("pool_growth_pool"))
        else:
            raise MatrixContractError(f"Unknown MAIN evaluation variant: {cell.cell_id}")
    elif cell.experiment_id == "M5B-REP-01":
        _require(cell.variant_id in {"residual", "absolute", "future_state"}, f"Unknown REP variant: {cell.cell_id}")
        values["target_representation"] = cell.variant_id
    elif cell.experiment_id == "M5B-ACTION-01":
        allowed = {
            "raw_human_plan_plus_tplus1_query_main",
            "phase_aligned_human_plan_plus_tplus1_query",
            "raw_human_plan_plus_lag_calibrated_query_diagnostic",
            *ACTION_NEGATIVE_CONTROLS,
        }
        _require(cell.variant_id in allowed, f"Unknown ACTION variant: {cell.cell_id}")
    elif cell.experiment_id == "M5B-RET-01":
        _require(cell.variant_id in {"random", "phase", "geometry", "visual", "geometry_plus_visual"}, f"Unknown RET variant: {cell.cell_id}")
        values["retrieval_modality"] = cell.variant_id
    elif cell.experiment_id == "M5B-SENS-01":
        match = re.fullmatch(r"topk(1|3|5|10)_h(4|8|16)_k(4|8)", cell.variant_id)
        _require(match is not None, f"Unknown SENS variant: {cell.cell_id}")
        values["top_k"], values["h_steps"], values["k_steps"] = map(int, match.groups())
    elif cell.experiment_id == "M5B-TIME-01":
        if cell.variant_id.startswith("time_view_"):
            values["time_view_id"] = cell.variant_id.removeprefix("time_view_")
        else:
            _require(corruption_id is not None, f"Unknown TIME variant: {cell.cell_id}")
    elif cell.experiment_id == "M5B-RES-01":
        _require(cell.variant_id in RESOLUTION_VARIANTS, f"Unknown RES variant: {cell.cell_id}")
        values["resolution_variant"] = cell.variant_id
    else:
        raise MatrixContractError(f"No evaluation handler family for {cell.cell_id}")

    if cell.method_id == "retrieval_only":
        values["target_representation"] = "retrieval_only"

    return EvaluationBinding(
        cell_id=cell.cell_id,
        experiment_id=cell.experiment_id,
        variant_id=cell.variant_id,
        method_id=cell.method_id,
        run_seed=cell.seed,
        parent_artifact_id=parent_id,
        checkpoint_cell_id=checkpoint_cell_id,
        prepared_input_cell_id=prepared_input_cell_id,
        corruption_id=corruption_id,
        corruption_severity=corruption_severity,
        negative_control_detector=detector,
        requires_model_inference=requires_model,
        **values,
    )


def _transitive_report_evaluations(
    report_ids: set[str],
    cells_by_id: Mapping[str, FrozenCell],
) -> frozenset[str]:
    seen: set[str] = set()
    evaluations: set[str] = set()
    stack = list(report_ids)
    while stack:
        cell_id = stack.pop()
        if cell_id in seen:
            continue
        seen.add(cell_id)
        cell = cells_by_id[cell_id]
        if cell.artifact_kind == "checkpoint_linked_evaluation":
            evaluations.add(cell_id)
        stack.extend(cell.parent_artifact_ids)
    return frozenset(evaluations)


def load_execution_matrix(
    workspace: Path | None = None,
    *,
    verify_prepared_artifact_hashes: bool = True,
) -> ExecutionMatrix:
    workspace = workspace or Path(__file__).resolve().parents[1]
    validate_four_gpu_successor(workspace)
    validate_memory_successor(workspace)
    validate_io_successor(workspace)
    _require(file_sha256(workspace / SUPPLEMENT_RELATIVE_PATH) == SUPPLEMENT_SHA256, "Frozen supplement SHA256 changed")
    registry, cells_by_id, order = load_frozen_registry(workspace)
    prepared, entries_by_id = load_prepared_manifest(
        workspace, verify_artifact_hashes=verify_prepared_artifact_hashes
    )
    specs_by_id = {spec.cell_id: spec for spec in p2_training_specs()}
    learned_registry_ids = {
        cell.cell_id
        for cell in cells_by_id.values()
        if cell.artifact_kind == "learned_training_checkpoint"
    }
    _require(learned_registry_ids == set(specs_by_id), "Training specs do not exactly cover frozen learned cells")

    bindings: dict[str, CellBinding] = {}
    for cell_id in order:
        cell = cells_by_id[cell_id]
        if cell.artifact_kind == "learned_training_checkpoint":
            bindings[cell_id] = CellBinding(
                cell=cell,
                handler_kind="train_step7000_checkpoint",
                prepared_entry=entries_by_id[cell_id],
                training_spec=specs_by_id[cell_id],
            )
        elif cell.artifact_kind == "nonlearned_method_artifact":
            _require(cell.seed is not None, f"Nonlearned artifact missing seed: {cell_id}")
            prepared_id = _main_recap_cell_id(cell.seed)
            bindings[cell_id] = CellBinding(
                cell=cell,
                handler_kind="retrieval_only_projection_artifact",
                prepared_entry=entries_by_id[prepared_id],
                training_spec=specs_by_id[prepared_id],
            )
        elif cell.artifact_kind == "checkpoint_linked_evaluation":
            evaluation = _evaluation_binding(cell, cells_by_id, specs_by_id)
            bindings[cell_id] = CellBinding(
                cell=cell,
                handler_kind="heldout_checkpoint_evaluation",
                prepared_entry=entries_by_id[evaluation.prepared_input_cell_id],
                training_spec=specs_by_id[evaluation.prepared_input_cell_id],
                evaluation=evaluation,
            )
        elif cell.artifact_kind == "aggregate_report":
            handler = (
                "full_matrix_completion_report_builder"
                if cell.variant_id == "full_matrix_completion_acceptance"
                else "qualitative_report_builder"
            )
            bindings[cell_id] = CellBinding(cell=cell, handler_kind=handler)
        else:
            raise MatrixContractError(f"No handler kind for {cell_id}")

    report_ids = {
        cell_id for cell_id, cell in cells_by_id.items() if cell.artifact_kind == "aggregate_report"
    }
    covered = _transitive_report_evaluations(report_ids, cells_by_id)
    all_evaluations = {
        cell_id
        for cell_id, cell in cells_by_id.items()
        if cell.artifact_kind == "checkpoint_linked_evaluation"
    }
    _require(covered == all_evaluations, "Terminal completion report does not cover all evaluations")
    bounds_path = workspace / WORKSPACE_BOUNDS_RELATIVE_PATH
    _require(file_sha256(bounds_path) == WORKSPACE_BOUNDS_SHA256, "Workspace bounds hash changed")
    bounds = _read_json(bounds_path)
    _require(bounds.get("status") == "frozen", "Workspace bounds are not frozen")
    _require(bounds.get("heldout_data_used") is False, "Workspace bounds use heldout data")
    lag_path = workspace / LAG_VIEW_MANIFEST_RELATIVE_PATH
    _require(file_sha256(lag_path) == LAG_VIEW_MANIFEST_SHA256, "Lag view manifest hash changed")
    lag = _read_json(lag_path)
    _require(lag.get("query_target_offset_view_steps") == 5, "Lag view offset changed")
    _require(lag.get("materialization", {}).get("gap_crossing_count") == 0, "Lag view crosses a segment")
    blockers: list[str] = []
    return ExecutionMatrix(
        registry=registry,
        prepared_manifest=prepared,
        cells_by_id=cells_by_id,
        bindings_by_id=bindings,
        topological_cell_ids=order,
        formal_readiness_blockers=tuple(blockers),
        report_covered_evaluation_ids=covered,
    )


if __name__ == "__main__":
    matrix = load_execution_matrix()
    print(
        json.dumps(
            {
                "cell_count": len(matrix.bindings_by_id),
                "counts": dict(Counter(item.cell.artifact_kind for item in matrix.bindings_by_id.values())),
                "formal_readiness_blockers": matrix.formal_readiness_blockers,
                "report_covered_evaluation_count": len(matrix.report_covered_evaluation_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )
