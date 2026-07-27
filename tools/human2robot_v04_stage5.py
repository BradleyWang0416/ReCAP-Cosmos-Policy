#!/usr/bin/env python3
"""Formal stage-5 preparation, launch, checkpoint retention, and audit."""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

from cosmos_policy.datasets.human2robot_dataset import (
    _matrix_to_rotation_6d,
    _rotation_6d_to_matrix,
)
from cosmos_policy.datasets.human2robot_v04_dataset import Human2RobotV04Dataset
from cosmos_policy.datasets.human2robot_v04_retrieval import poses_euler_to_10d, stable_sha256
from cosmos_policy.models.human2robot_adapter import validate_human2robot_batch
from tools import human2robot_v04_stage4 as stage4


SCHEMA = "human2robot-v04-stage5-v1"
METHODS = ("no_retrieval", "co_training", "recap_hand_ret")
TARGET_BY_METHOD = {
    "no_retrieval": "absolute",
    "co_training": "absolute",
    "recap_hand_ret": "residual",
}
CONFIG_BY_METHOD = {
    method: f"cosmos_predict2p5_2b_human2robot_v04_{method}_seed20260711" for method in METHODS
}
STAGE4_LOCK_SHA256 = "834fe271bfe964c0b4f6e6c3b8ba899191609643acd166df0fa8e2811fca7ade"
SOURCE_MANIFEST_SHA256 = "7869a078b19ba18aaa6a92c22bec26998a81d412165a02eb8cb6c6aec1c879ed"
INITIALIZATION_CHECKPOINT = Path(
    "/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/base/post-trained/"
    "81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
)
TOKENIZER_CHECKPOINT = Path("/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth")
MIN_HOME_BYTES = 150 * 1024**3
MIN_DATA1_BYTES = 100 * 1024**3
EXPECTED_RECORDS = 654
EXPECTED_WINDOWS = 244_372


class Stage5Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage5Error(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"Required stage-5 file is missing: {resolved}")
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": file_sha256(resolved)}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any], *, immutable: bool = True) -> None:
    if path.exists():
        if immutable:
            require(read_json(path) == dict(value), f"Refusing to replace different immutable JSON: {path}")
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        json.loads(partial.read_text(encoding="utf-8"))
        os.replace(partial, path)
        if immutable:
            path.chmod(0o444)
    finally:
        with contextlib.suppress(FileNotFoundError):
            partial.unlink()


def write_npz_atomic(path: Path, arrays: Mapping[str, Any]) -> None:
    require(not path.exists(), f"Refusing to replace immutable stage-5 index: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial.npz")
    try:
        np.savez(partial, **arrays)
        with np.load(partial, allow_pickle=False) as stored:
            require("starts" in stored.files and "offsets" in stored.files, "Stage-5 index round-trip failed")
            json.loads(str(stored["manifest_json"].item()))
        os.replace(partial, path)
        path.chmod(0o444)
    finally:
        with contextlib.suppress(FileNotFoundError):
            partial.unlink()


def _storage() -> dict[str, int]:
    home = shutil.disk_usage("/home").free
    data1 = shutil.disk_usage("/DATA1").free
    require(home >= MIN_HOME_BYTES, f"/home free space below 150 GiB: {home}")
    require(data1 >= MIN_DATA1_BYTES, f"/DATA1 free space below 100 GiB: {data1}")
    return {"home_free_bytes": home, "data1_free_bytes": data1}


def _frozen_inputs(workspace: Path, run_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_path = workspace / "data/Human2Robot/derived/v04/source_split_manifest.json"
    lock_path = workspace / "data/Human2Robot/derived/v04/stage4_protocol_lock.json"
    suite_path = run_root / "orchestrator_logs/stage5_full_suite_20260722/attempt_0003.receipt.json"
    source_binding = bind_file(source_path)
    lock_binding = bind_file(lock_path)
    suite_binding = bind_file(suite_path)
    require(source_binding["sha256"] == SOURCE_MANIFEST_SHA256, "Stage-1 source manifest SHA drift")
    require(lock_binding["sha256"] == STAGE4_LOCK_SHA256, "Stage-4 protocol-lock SHA drift")
    source = read_json(source_path)
    lock = read_json(lock_path)
    suite = read_json(suite_path)
    require(suite.get("status") == "PASSED" and suite.get("stage5_launch_allowed") is True, "Stage-5 suite does not authorize launch")
    require(suite.get("stage4_protocol_lock") == lock_binding, "Stage-5 suite protocol-lock binding drift")
    require(lock.get("status") == "VERIFIED_STAGE4", "Stage-4 lock is not verified")
    require(lock.get("training_allowed") is True and lock.get("stage5_allowed") is True, "Stage-4 lock blocks stage 5")
    require(lock.get("provenance_violation_count") == 0, "Stage-4 provenance violation blocks training")
    require(lock.get("future_target_independence_violation_count") == 0, "Stage-4 target leakage blocks training")
    require(lock.get("nonfinite_count") == 0, "Stage-4 nonfinite result blocks training")
    require(lock.get("missing_receipt_count") == 0, "Stage-4 missing receipt blocks training")
    require(lock.get("gap_crossing_count") == 0, "Stage-4 gap violation blocks training")
    training = lock.get("training_protocol", {})
    expected = {
        "methods": list(METHODS),
        "order": list(METHODS),
        "seed": 20260711,
        "gpu_count": 4,
        "batch_per_gpu": 25,
        "gradient_accumulation": 2,
        "effective_batch": 200,
        "optimizer_steps": 7000,
        "H_steps": 8,
        "K_steps": 8,
        "top_k": 3,
        "resolution": [224, 224],
        "learning_rate": 1e-4,
        "action_loss_multiplier": 16,
        "text_conditioning": "disabled_zero_embedding",
        "checkpoint_selection": "fixed_step_7000",
        "heldout_training_or_selection_allowed": False,
    }
    mismatches = {key: {"actual": training.get(key), "expected": value} for key, value in expected.items() if training.get(key) != value}
    require(not mismatches, f"Frozen stage-5 training protocol mismatch: {mismatches}")
    records = [dict(row) for row in source.get("records", []) if row.get("source_partition") == "seen_train"]
    records.sort(key=lambda row: (str(row["task"]), int(row["partition_rank"]), str(row["source_sha256"])))
    require(len(records) == EXPECTED_RECORDS, f"Expected {EXPECTED_RECORDS} seen-train records")
    require(sum(int(row["legal_window_count"]) for row in records) == EXPECTED_WINDOWS, "Seen-train window count drift")
    return {
        "source_manifest": source_binding,
        "stage4_protocol_lock": lock_binding,
        "stage5_suite": suite_binding,
    }, lock, {"records": records, "source": source}


def _states(file: h5py.File, role: str) -> np.ndarray:
    if role == "human":
        action = np.asarray(file["action"][:], dtype=np.float64)
        return poses_euler_to_10d(action[:, :6], action[:, 6])
    return poses_euler_to_10d(
        np.asarray(file["end_position"][:], dtype=np.float64),
        np.asarray(file["gripper_state"][:], dtype=np.float64),
    )


def _align_pool_batch(pool: np.ndarray, current: np.ndarray) -> np.ndarray:
    require(pool.ndim == 3 and pool.shape[1:] == (8, 10), f"Invalid pool batch shape: {pool.shape}")
    require(current.shape == (len(pool), 10), f"Invalid current batch shape: {current.shape}")
    aligned = pool.copy()
    aligned[:, :, :3] = current[:, None, :3] + pool[:, :, :3] - pool[:, :1, :3]
    pool_rot = _rotation_6d_to_matrix(pool[:, :, 3:9])
    current_rot = _rotation_6d_to_matrix(current[:, 3:9])
    anchor_inverse = np.swapaxes(pool_rot[:, 0], -1, -2)
    aligned_rot = current_rot[:, None] @ anchor_inverse[:, None] @ pool_rot
    aligned[:, :, 3:9] = _matrix_to_rotation_6d(aligned_rot)
    return aligned


def _update_range(minimum: np.ndarray, maximum: np.ndarray, values: np.ndarray) -> None:
    flat = np.asarray(values, dtype=np.float64).reshape(-1, 10)
    require(bool(np.all(np.isfinite(flat))), "Nonfinite stage-5 training statistic")
    minimum[:] = np.minimum(minimum, flat.min(axis=0))
    maximum[:] = np.maximum(maximum, flat.max(axis=0))


def _geometry_vectors(states: np.ndarray, starts: np.ndarray, geometry: Mapping[str, Any]) -> np.ndarray:
    histories = states[starts[:, None] + np.arange(8)[None, :]]
    relative = histories - histories[:, -1:, :]
    normalized = (relative - np.asarray(geometry["mean_10d"])[None, None]) / np.asarray(geometry["std_10d"])[None, None]
    flat = normalized.reshape(len(starts), -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    return (flat / np.maximum(norms, 1e-12)).astype(np.float32)


def _visual_vectors(feature_root: Path, record: Mapping[str, Any], role: str, starts: np.ndarray) -> np.ndarray:
    path = stage4._feature_shard_path(feature_root, record, role)
    stored_starts, features, manifest = stage4._read_feature_shard(path)
    require(np.array_equal(stored_starts, starts), f"WAN shard/start mismatch: {path}")
    require(manifest.get("role") == role, f"WAN shard role mismatch: {path}")
    flat = features.reshape(len(features), -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    return (flat / np.maximum(norms, 1e-12)).astype(np.float32)


def _combined_vectors(
    source_root: Path,
    feature_root: Path,
    record: Mapping[str, Any],
    starts: np.ndarray,
    role: str,
    geometry: Mapping[str, Any],
) -> np.ndarray:
    path = (source_root / str(record["source_relative_path"])).resolve()
    with h5py.File(path, "r") as file:
        states = _states(file, role)
    geometry_vectors = _geometry_vectors(states, starts, geometry)
    visual_vectors = _visual_vectors(feature_root, record, role, starts)
    scale = np.float32(1.0 / np.sqrt(2.0))
    return np.concatenate((geometry_vectors * scale, visual_vectors * scale), axis=1)


def _nearest_phase_indices(query_starts: np.ndarray, query_frames: int, candidate_starts: np.ndarray, candidate_frames: int) -> np.ndarray:
    query_phase = (query_starts.astype(np.float64) + 8.0) / float(query_frames)
    candidate_phase = (candidate_starts.astype(np.float64) + 8.0) / float(candidate_frames)
    right = np.searchsorted(candidate_phase, query_phase, side="left")
    right = np.clip(right, 0, len(candidate_phase) - 1)
    left = np.clip(right - 1, 0, len(candidate_phase) - 1)
    choose_left = np.abs(candidate_phase[left] - query_phase) <= np.abs(candidate_phase[right] - query_phase)
    return np.where(choose_left, left, right).astype(np.int64)


def _build_recap_retrieval(
    records: Sequence[Mapping[str, Any]],
    starts_by_record: Sequence[np.ndarray],
    *,
    source_root: Path,
    feature_root: Path,
    geometry: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = sum(len(starts) for starts in starts_by_record)
    candidate_record_indices = np.full((total, 3), -1, dtype=np.int32)
    candidate_starts = np.full((total, 3), -1, dtype=np.int64)
    distances_out = np.full((total, 3), np.inf, dtype=np.float32)
    task_records: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        task_records.setdefault(str(record["task"]), []).append(index)
    feature_cache: dict[tuple[int, str], np.ndarray] = {}
    offset = 0
    for query_record_index, (query_record, query_starts) in enumerate(zip(records, starts_by_record, strict=True)):
        task_indices = [index for index in task_records[str(query_record["task"])] if index != query_record_index]
        require(len(task_indices) >= 3, f"Fewer than three independent training sources for {query_record['task']}")
        qkey = (query_record_index, "robot")
        if qkey not in feature_cache:
            feature_cache[qkey] = _combined_vectors(
                source_root, feature_root, query_record, query_starts, "robot", geometry
            )
        query_vectors = feature_cache[qkey]
        distance_columns: list[np.ndarray] = []
        selected_start_columns: list[np.ndarray] = []
        for candidate_index in task_indices:
            candidate_record = records[candidate_index]
            candidate_episode_starts = starts_by_record[candidate_index]
            ckey = (candidate_index, "human")
            if ckey not in feature_cache:
                feature_cache[ckey] = _combined_vectors(
                    source_root, feature_root, candidate_record, candidate_episode_starts, "human", geometry
                )
            nearest = _nearest_phase_indices(
                query_starts,
                int(query_record["frame_count"]),
                candidate_episode_starts,
                int(candidate_record["frame_count"]),
            )
            selected_start_columns.append(candidate_episode_starts[nearest])
            distance_columns.append(np.linalg.norm(query_vectors - feature_cache[ckey][nearest], axis=1))
        distances = np.stack(distance_columns, axis=1)
        starts_matrix = np.stack(selected_start_columns, axis=1)
        for local_index, query_start in enumerate(query_starts):
            query_id = f"{query_record['source_sha256']}:{int(query_start)}:H8:K8"
            order = sorted(
                range(len(task_indices)),
                key=lambda column: (
                    float(distances[local_index, column]),
                    stable_sha256(
                        20260711,
                        query_id,
                        str(records[task_indices[column]]["human_content_sha256"]),
                    ),
                    str(records[task_indices[column]]["source_sha256"]),
                ),
            )[:3]
            global_index = offset + local_index
            candidate_record_indices[global_index] = np.asarray([task_indices[column] for column in order], dtype=np.int32)
            candidate_starts[global_index] = np.asarray([starts_matrix[local_index, column] for column in order], dtype=np.int64)
            distances_out[global_index] = np.asarray([distances[local_index, column] for column in order], dtype=np.float32)
        offset += len(query_starts)
        if (query_record_index + 1) % 10 == 0 or query_record_index + 1 == len(records):
            print(
                json.dumps(
                    {
                        "event": "stage5_retrieval_progress",
                        "episodes": query_record_index + 1,
                        "total_episodes": len(records),
                        "windows": offset,
                    }
                ),
                flush=True,
            )
    require(offset == total, "RECAP retrieval index cardinality drift")
    require(bool(np.all(candidate_record_indices >= 0)), "RECAP retrieval candidate is missing")
    require(bool(np.all(np.isfinite(distances_out))), "RECAP retrieval contains nonfinite distances")
    return candidate_record_indices, candidate_starts, distances_out


def _training_statistics(
    method: str,
    records: Sequence[Mapping[str, Any]],
    starts_by_record: Sequence[np.ndarray],
    *,
    source_root: Path,
    candidate_record_indices: np.ndarray | None,
    candidate_starts: np.ndarray | None,
) -> dict[str, list[float]]:
    query_min = np.full(10, np.inf)
    query_max = np.full(10, -np.inf)
    pool_min = np.full(10, np.inf)
    pool_max = np.full(10, -np.inf)
    residual_min = np.full(10, np.inf)
    residual_max = np.full(10, -np.inf)
    offset = 0
    human_cache: dict[int, np.ndarray] = {}
    for record_index, (record, starts) in enumerate(zip(records, starts_by_record, strict=True)):
        path = (source_root / str(record["source_relative_path"])).resolve()
        with h5py.File(path, "r") as file:
            robot = _states(file, "robot")
            human_same = _states(file, "human") if method == "co_training" else None
        current = robot[starts + 7]
        future_rows = starts[:, None] + np.arange(8, 16)[None, :]
        target = robot[future_rows]
        _update_range(query_min, query_max, target)
        if method == "no_retrieval":
            aligned = np.repeat(current[:, None], 8, axis=1)
            _update_range(pool_min, pool_max, aligned)
            _update_range(residual_min, residual_max, target - aligned)
        elif method == "co_training":
            assert human_same is not None
            aligned = _align_pool_batch(human_same[future_rows], current)
            _update_range(pool_min, pool_max, aligned)
            _update_range(residual_min, residual_max, target - aligned)
        else:
            assert candidate_record_indices is not None and candidate_starts is not None
            for rank in range(3):
                candidate_ids = candidate_record_indices[offset : offset + len(starts), rank]
                candidate_rows = candidate_starts[offset : offset + len(starts), rank]
                plans = np.empty((len(starts), 8, 10), dtype=np.float64)
                for candidate_index in np.unique(candidate_ids):
                    candidate_index_int = int(candidate_index)
                    if candidate_index_int not in human_cache:
                        candidate_path = (source_root / str(records[candidate_index_int]["source_relative_path"])).resolve()
                        with h5py.File(candidate_path, "r") as file:
                            human_cache[candidate_index_int] = _states(file, "human")
                    mask = candidate_ids == candidate_index
                    rows = candidate_rows[mask, None] + np.arange(8, 16)[None, :]
                    plans[mask] = human_cache[candidate_index_int][rows]
                aligned = _align_pool_batch(plans, current)
                _update_range(pool_min, pool_max, aligned)
                _update_range(residual_min, residual_max, target - aligned)
        offset += len(starts)
        if (record_index + 1) % 25 == 0 or record_index + 1 == len(records):
            print(
                json.dumps(
                    {
                        "event": "stage5_statistics_progress",
                        "method": method,
                        "episodes": record_index + 1,
                        "total_episodes": len(records),
                        "windows": offset,
                    }
                ),
                flush=True,
            )
    require(offset == EXPECTED_WINDOWS, "Stage-5 statistics window count drift")
    return {
        "query_bc_target_10d_min": query_min.tolist(),
        "query_bc_target_10d_max": query_max.tolist(),
        "pool_action_10d_min": pool_min.tolist(),
        "pool_action_10d_max": pool_max.tolist(),
        "residual_10d_min": residual_min.tolist(),
        "residual_10d_max": residual_max.tolist(),
    }


def prepare_training_inputs(
    method: str,
    *,
    workspace: Path,
    run_root: Path,
    source_root: Path,
) -> dict[str, Any]:
    require(method in METHODS, f"Unsupported stage-5 method: {method}")
    frozen_inputs, lock, source_context = _frozen_inputs(workspace, run_root)
    output_root = run_root / "stage5/training_inputs" / method
    index_path = output_root / "training_index.npz"
    statistics_path = output_root / "action_statistics.json"
    receipt_path = output_root / "preparation_receipt.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        require(receipt.get("status") == "FROZEN", f"Existing stage-5 preparation is invalid: {receipt_path}")
        require(receipt.get("method") == method, "Existing stage-5 preparation method mismatch")
        require(bind_file(index_path) == receipt.get("training_index"), "Existing stage-5 index binding drift")
        require(bind_file(statistics_path) == receipt.get("statistics"), "Existing stage-5 statistics binding drift")
        return receipt

    records = source_context["records"]
    starts_by_record: list[np.ndarray] = []
    offsets = [0]
    for record_index, record in enumerate(records, start=1):
        path = (source_root / str(record["source_relative_path"])).resolve()
        require(path.is_file(), f"Missing seen-train source: {path}")
        starts = stage4._legal_starts(record, path)
        starts_by_record.append(starts)
        offsets.append(offsets[-1] + len(starts))
        if record_index % 50 == 0 or record_index == len(records):
            print(json.dumps({"event": "stage5_index_progress", "method": method, "episodes": record_index, "windows": offsets[-1]}), flush=True)
    starts = np.concatenate(starts_by_record).astype(np.int64)
    offsets_array = np.asarray(offsets, dtype=np.int64)
    require(len(starts) == EXPECTED_WINDOWS, "Stage-5 training-index window count drift")

    candidate_record_indices = None
    candidate_starts = None
    retrieval_distances = None
    if method == "recap_hand_ret":
        geometry = read_json(run_root / "features/geometry_statistics.json")
        require(geometry.get("status") == "FROZEN", "Frozen geometry statistics are missing")
        candidate_record_indices, candidate_starts, retrieval_distances = _build_recap_retrieval(
            records,
            starts_by_record,
            source_root=source_root,
            feature_root=run_root / "features",
            geometry=geometry,
        )

    statistics_values = _training_statistics(
        method,
        records,
        starts_by_record,
        source_root=source_root,
        candidate_record_indices=candidate_record_indices,
        candidate_starts=candidate_starts,
    )
    training_sha = str(lock["training_protocol_sha256"])
    index_manifest = {
        "schema_version": f"{SCHEMA}-training-index",
        "status": "FROZEN",
        "created_at_utc": utc_now(),
        "method": method,
        "source_partition": "seen_train",
        "record_count": len(records),
        "legal_window_count": len(starts),
        "top_k_replication": 3,
        "expanded_sample_count": len(starts) * 3,
        "task_balanced_sampler_required": True,
        "training_protocol_sha256": training_sha,
        "source_split_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "future_rows_read_for_retrieval": 0,
        "target_datasets_read_for_retrieval": 0,
        "heldout_data_used": False,
        "records": records,
    }
    arrays: dict[str, Any] = {
        "starts": starts,
        "offsets": offsets_array,
        "manifest_json": np.asarray(json.dumps(index_manifest, ensure_ascii=False, sort_keys=True)),
    }
    if candidate_record_indices is not None:
        arrays.update(
            {
                "candidate_record_indices": candidate_record_indices,
                "candidate_starts": candidate_starts,
                "retrieval_distances": retrieval_distances,
            }
        )
    write_npz_atomic(index_path, arrays)
    statistics = {
        "schema_version": f"{SCHEMA}-action-statistics",
        "status": "FROZEN",
        "created_at_utc": utc_now(),
        "method": method,
        "source_partition": "seen_train",
        "record_count": len(records),
        "legal_window_count": len(starts),
        "top_k_replication": 3,
        "training_protocol_sha256": training_sha,
        "source_split_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "training_index": bind_file(index_path),
        "heldout_data_used": False,
        **statistics_values,
    }
    write_json_atomic(statistics_path, statistics)
    receipt = {
        "schema_version": f"{SCHEMA}-preparation-receipt",
        "status": "FROZEN",
        "created_at_utc": utc_now(),
        "method": method,
        "training_protocol_sha256": training_sha,
        "frozen_inputs": frozen_inputs,
        "training_index": bind_file(index_path),
        "statistics": bind_file(statistics_path),
        "record_count": len(records),
        "legal_window_count": len(starts),
        "expanded_sample_count": len(starts) * 3,
        "heldout_data_used": False,
        "future_rows_read_for_retrieval": 0,
        "target_datasets_read_for_retrieval": 0,
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def _config_output_root(run_root: Path, method: str) -> Path:
    return run_root / "stage5/training_runs/cosmos_policy/human2robot_v04_stage5" / CONFIG_BY_METHOD[method]


def _controlled_sources(workspace: Path) -> list[dict[str, Any]]:
    paths = [
        workspace / "tools/human2robot_v04_experiment.py",
        workspace / "tools/human2robot_v04_stage5.py",
        workspace / "cosmos_policy/datasets/human2robot_v04_dataset.py",
        workspace / "cosmos_policy/datasets/human2robot_v04_sampler.py",
        workspace / "cosmos_policy/config/experiment/human2robot_v04_experiment_configs.py",
        workspace / "cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py",
        workspace / "cosmos_policy/scripts/train_human2robot_v04.py",
        workspace / "cosmos_policy/trainer_human2robot_v04.py",
        workspace / "cosmos_policy/models/human2robot_adapter.py",
        workspace / "方案/v04/RECAP_Human2Robot_无泄漏单seed离线复现执行总计划.md",
        workspace / "方案/v04/CHANGELOG.md",
    ]
    return [bind_file(path) for path in paths]


def _checkpoint_step(path: Path) -> int | None:
    if not path.is_dir() or not path.name.startswith("iter_"):
        return None
    with contextlib.suppress(ValueError):
        return int(path.name.removeprefix("iter_"))
    return None


def prune_checkpoints(checkpoint_root: Path) -> list[str]:
    tracker = checkpoint_root / "latest_checkpoint.txt"
    if not tracker.is_file():
        return []
    with contextlib.suppress(ValueError):
        latest = int(tracker.read_text(encoding="utf-8").strip().removeprefix("iter_"))
        rolling_to_keep = {step for step in (latest, latest - 1000) if 0 < step < 7000}
        if latest >= 7000:
            rolling_to_keep = {5000, 6000}
        keep = rolling_to_keep | ({7000} if latest >= 7000 else set())
        removed: list[str] = []
        for path in sorted(checkpoint_root.glob("iter_*")):
            step = _checkpoint_step(path)
            if step is not None and step < latest and step not in keep:
                shutil.rmtree(path)
                removed.append(path.name)
        return removed
    return []


def _prune_loop(checkpoint_root: Path, stop: threading.Event) -> None:
    while not stop.wait(30.0):
        removed = prune_checkpoints(checkpoint_root)
        if removed:
            print(json.dumps({"event": "stage5_checkpoint_pruned", "removed": removed}), flush=True)


def _tree_binding(root: Path) -> dict[str, Any]:
    require(root.is_dir(), f"Final checkpoint directory is missing: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": file_sha256(path)})
    require(files, f"Final checkpoint has no files: {root}")
    return {
        "path": str(root.resolve()),
        "file_count": len(files),
        "size_bytes": sum(item["size_bytes"] for item in files),
        "bundle_sha256": canonical_sha256(files),
        "files": files,
    }


def _freeze_tree(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def run_stage5(
    method: str,
    *,
    workspace: Path,
    run_root: Path,
    source_root: Path = Path("/DATA1/wxs/DATASETS/Human2Robot/data/v1"),
) -> dict[str, Any]:
    require(method in METHODS, f"Unsupported stage-5 method: {method}")
    require(Path("/.dockerenv").is_file(), "Formal stage 5 must run inside Docker")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 4, "Formal stage 5 requires exactly four visible GPUs")
    gpu_devices = os.environ.get("HUMAN2ROBOT_V04_GPU_DEVICES", "")
    require(len(gpu_devices.split(",")) == 4, "Host GPU mapping is not bound to four devices")
    storage_before = _storage()
    frozen_inputs, lock, _ = _frozen_inputs(workspace, run_root)
    preparation = prepare_training_inputs(
        method,
        workspace=workspace,
        run_root=run_root,
        source_root=source_root,
    )
    dataset = Human2RobotV04Dataset(
        source_root=source_root,
        training_index_path=Path(str(preparation["training_index"]["path"])),
        statistics_path=Path(str(preparation["statistics"]["path"])),
        stage4_protocol_lock_path=workspace / "data/Human2Robot/derived/v04/stage4_protocol_lock.json",
        method_id=method,
        use_image_aug=True,
    )
    sample = dataset[0]
    require(all(validate_human2robot_batch(sample).values()), "Stage-5 dataset/model adapter probe failed")
    sampler = dataset.make_distributed_sampler(num_replicas=4, rank=0)
    first = list(zip(range(32), sampler, strict=False))
    require(len(first) == 32, "Task-balanced sampler startup probe is incomplete")
    require(len({item.global_sample_index for _, item in first}) == 32, "Task-balanced sampler duplicated startup indices")

    config_name = CONFIG_BY_METHOD[method]
    output_root = _config_output_root(run_root, method)
    checkpoint_root = output_root / "checkpoints"
    final_checkpoint = checkpoint_root / "iter_000007000"
    require(not final_checkpoint.exists(), f"Immutable final checkpoint already exists: {final_checkpoint}")
    active_path = run_root / "stage5/active" / f"{method}.json"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active = {
        "schema_version": f"{SCHEMA}-active-run",
        "status": "RUNNING",
        "created_at_utc": utc_now(),
        "method": method,
        "container_name": os.environ.get("HUMAN2ROBOT_V04_CONTAINER_NAME"),
        "host_gpu_devices": gpu_devices,
        "pid": os.getpid(),
        "output_root": str(output_root),
    }
    if active_path.exists():
        previous = read_json(active_path)
        require(previous.get("status") != "RUNNING", f"A stage-5 {method} run is already active")
    write_json_atomic(active_path, active, immutable=False)

    controlled = _controlled_sources(workspace)
    launch_manifest = {
        "schema_version": f"{SCHEMA}-launch-manifest",
        "status": "FROZEN",
        "created_at_utc": utc_now(),
        "method": method,
        "target_representation": TARGET_BY_METHOD[method],
        "config_name": config_name,
        "host_gpu_devices": gpu_devices,
        "container_logical_gpu_devices": [0, 1, 2, 3],
        "parallel_schedule": "no_retrieval_and_co_training_may_overlap_on_disjoint_four_gpu_groups",
        "scientific_method_order": list(METHODS),
        "training_protocol_sha256": lock["training_protocol_sha256"],
        "frozen_inputs": frozen_inputs,
        "preparation": preparation,
        "controlled_source": controlled,
        "controlled_source_bundle_sha256": canonical_sha256(controlled),
        "storage_before": storage_before,
        "resume_policy": "same-method latest valid DCP only; never cross-method or v03",
        "checkpoint_policy": "save every 1000; retain two rolling recovery points; freeze step7000 final",
    }
    launch_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    launch_manifest_path = run_root / "stage5/manifests" / f"{method}.{launch_id}.{uuid.uuid4().hex[:8]}.launch.json"
    write_json_atomic(launch_manifest_path, launch_manifest)

    env = os.environ.copy()
    env.pop("NCCL_DEBUG_SUBSYS", None)
    env.update(
        {
            "COSMOS_SKIP_HF_AUTO_DOWNLOAD": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "65536",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
            "TORCH_NCCL_DESYNC_DEBUG": "1",
            "NCCL_DEBUG": "WARN",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "IMAGINAIRE_OUTPUT_ROOT": str(run_root / "stage5/training_runs"),
            "RECAP_WORKSPACE": str(workspace),
            "HUMAN2ROBOT_ROOT": str(workspace / "data/Human2Robot"),
            "HUMAN2ROBOT_V04_STAGE5_INPUT_ROOT": str(run_root / "stage5/training_inputs"),
            "HUMAN2ROBOT_V04_STAGE5_METHOD": method,
            "HUMAN2ROBOT_V04_STAGE5_RUNTIME_BINDING_PATH": str(output_root / "stage5_runtime_binding.json"),
            "HUMAN2ROBOT_V04_STAGE5_PROGRESS_PATH": str(run_root / "stage5/progress" / f"{method}.json"),
            "HUMAN2ROBOT_V04_STAGE5_TRAINING_PROTOCOL_SHA256": str(lock["training_protocol_sha256"]),
            "COSMOS_PREDICT2P5_POSTTRAINED_CKPT": str(INITIALIZATION_CHECKPOINT),
            "COSMOS_PREDICT2P5_TOKENIZER_CKPT": str(TOKENIZER_CHECKPOINT),
        }
    )
    command = [
        str(workspace / ".venv/bin/torchrun"),
        "--nproc_per_node=4",
        "--master_addr=127.0.0.1",
        "--master_port=12450",
        "-m",
        "cosmos_policy.scripts.train_human2robot_v04",
        "--config=cosmos_policy/config/config.py",
        "--",
        f"experiment={config_name}",
    ]
    print(
        json.dumps(
            {
                "event": "stage5_training_start",
                "method": method,
                "host_gpu_devices": gpu_devices,
                "command": command,
                "output_root": str(output_root),
                "launch_manifest": bind_file(launch_manifest_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    stop = threading.Event()
    pruner = threading.Thread(target=_prune_loop, args=(checkpoint_root, stop), daemon=True)
    pruner.start()
    process = subprocess.run(command, cwd=workspace, env=env, check=False)
    stop.set()
    pruner.join(timeout=35.0)
    removed = prune_checkpoints(checkpoint_root)
    if removed:
        print(json.dumps({"event": "stage5_checkpoint_pruned", "removed": removed}), flush=True)
    if process.returncode != 0:
        failed = {**active, "status": "FAILED", "finished_at_utc": utc_now(), "returncode": process.returncode}
        write_json_atomic(active_path, failed, immutable=False)
        raise Stage5Error(f"Stage-5 training exited with code {process.returncode}: {method}")

    require(final_checkpoint.is_dir(), f"Training returned zero without step-7000 checkpoint: {final_checkpoint}")
    progress_binding = bind_file(run_root / "stage5/progress" / f"{method}.json")
    progress = read_json(Path(str(progress_binding["path"])))
    require(progress.get("optimizer_step") == 7000, "Stage-5 progress did not reach optimizer step 7000")
    require(progress.get("loss_finite_enforced") is True, "Stage-5 loss-finite guard was not enforced")
    final_binding = _tree_binding(final_checkpoint)
    require(final_binding["size_bytes"] > 0, "Final checkpoint is empty")
    _freeze_tree(final_checkpoint)
    completed = {
        "schema_version": f"{SCHEMA}-training-result",
        "status": "PASSED",
        "created_at_utc": utc_now(),
        "method": method,
        "seed": 20260711,
        "optimizer_steps": 7000,
        "loss_finite": True,
        "training_progress": progress_binding,
        "checkpoint_selection": "fixed_step_7000",
        "heldout_training_or_selection_used": False,
        "host_gpu_devices": gpu_devices,
        "command": command,
        "launch_manifest": bind_file(launch_manifest_path),
        "runtime_binding": bind_file(output_root / "stage5_runtime_binding.json"),
        "final_checkpoint": final_binding,
        "storage_after": _storage(),
        "formal_result": False,
        "performance_claim_allowed": False,
    }
    result_path = run_root / "stage5/results" / f"{method}.json"
    write_json_atomic(result_path, completed)
    write_json_atomic(active_path, {**active, "status": "COMPLETED", "finished_at_utc": utc_now()}, immutable=False)
    return {**completed, "result_receipt": bind_file(result_path), "training_started": True}
