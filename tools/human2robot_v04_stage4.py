#!/usr/bin/env python3
"""Stage-4 materialization and read-only old-checkpoint smoke for Human2Robot v04."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

from cosmos_policy.datasets.human2robot_v04_retrieval import (
    FeatureProvenance,
    RetrievalFeature,
    rank_geometry_plus_visual,
    read_feature_inputs,
    window_from_manifest_record,
)


SCHEMA = "human2robot-v04-stage4-v1"
H_STEPS = 8
K_STEPS = 8
TOP_K = 3
POOL_SIZE = 10
RUN_SEED = 20260711
SMOKE_WINDOWS_PER_EPISODE = 8
EXPECTED_QUERY_COUNT = 160
EXPECTED_RECEIPTS_PER_METHOD = 480
METHODS = ("no_retrieval", "co_training", "recap_hand_ret")
FEATURE_PARTITIONS = (
    "seen_train",
    "seen_validation",
    "v04_human_pool",
    "v04_robot_dev",
    "v04_robot_final",
)
PARTITION_MANIFESTS = FEATURE_PARTITIONS
TOKENIZER_PATH = Path("/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth")
TOKENIZER_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
STAGE1_MANIFEST_SHA256 = "7869a078b19ba18aaa6a92c22bec26998a81d412165a02eb8cb6c6aec1c879ed"
STAGE2_CONTRACT_SHA256 = "27eceb61565d01297d4ec4ff19d166b5ff5c8d5e9af7916d92d5d9837af651d9"
STAGE3_CONTRACT_SHA256 = "10eab34a3479cdf54da5125c1de6d8035b372631f529298c11589ac3032a7e6b"
STAGE3_SUITE_SHA256 = "2717f2ff31b8a1cac8afc64ec34caeb77f5da9ea9f7b2d9ce7ca907074223a26"
V03_FREEZE_SHA256 = "d20eae44a2b1d0e1287dc8ae0973e2f713413aab904ef2dff1e304d927db1ab4"


class Stage4Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage4Error(message)


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
    require(resolved.is_file(), f"Required stage-4 file is missing: {resolved}")
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": file_sha256(resolved)}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
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


def write_npz_atomic(path: Path, *, starts: np.ndarray, features: np.ndarray, manifest: Mapping[str, Any]) -> None:
    require(not path.exists(), f"Refusing to replace immutable feature shard: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial.npz")
    try:
        np.savez(
            partial,
            starts=np.asarray(starts, dtype=np.int64),
            features=np.asarray(features, dtype=np.float32),
            manifest_json=np.asarray(json.dumps(dict(manifest), ensure_ascii=False, sort_keys=True)),
        )
        with np.load(partial, allow_pickle=False) as stored:
            require(stored["starts"].shape[0] == stored["features"].shape[0], "Feature shard cardinality mismatch")
            json.loads(str(stored["manifest_json"].item()))
        os.replace(partial, path)
        path.chmod(0o444)
    finally:
        with contextlib.suppress(FileNotFoundError):
            partial.unlink()


def _validate_frozen_inputs(workspace: Path, run_root: Path) -> dict[str, dict[str, Any]]:
    paths = {
        "stage1_manifest": workspace / "data/Human2Robot/derived/v04/source_split_manifest.json",
        "stage2_contract": workspace / "data/Human2Robot/derived/v04/stage2_retrieval_contract_report.json",
        "stage3_contract": workspace / "data/Human2Robot/derived/v04/stage3_experiment_interface.json",
        "stage3_suite": run_root / "orchestrator_logs/stage3_full_suite_20260721/receipt.json",
        "v03_freeze": workspace / "方案/v04/v03_frozen_manifest.json",
        "total_plan": workspace / "方案/v04/RECAP_Human2Robot_无泄漏单seed离线复现执行总计划.md",
    }
    bindings = {name: bind_file(path) for name, path in paths.items()}
    expected = {
        "stage1_manifest": STAGE1_MANIFEST_SHA256,
        "stage2_contract": STAGE2_CONTRACT_SHA256,
        "stage3_contract": STAGE3_CONTRACT_SHA256,
        "stage3_suite": STAGE3_SUITE_SHA256,
        "v03_freeze": V03_FREEZE_SHA256,
    }
    for name, sha256 in expected.items():
        require(bindings[name]["sha256"] == sha256, f"Frozen {name} SHA256 drift")
    stage3 = read_json(paths["stage3_contract"])
    suite = read_json(paths["stage3_suite"])
    require(stage3.get("status") == "VERIFIED_STAGE3", "Stage 3 contract is not verified")
    require(stage3.get("future_stage_authorization", {}).get("stage4_allowed") is True, "Stage 3 does not authorize stage 4")
    require(suite.get("status") == "PASSED" and suite.get("stage4_authorized_by_this_receipt") is True, "Stage 3 suite does not authorize stage 4")
    require(stage3.get("future_stage_authorization", {}).get("training_allowed") is False, "Training was unexpectedly authorized before stage 4")
    return bindings


def _records_by_partition(manifest: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result = {name: [] for name in FEATURE_PARTITIONS}
    for raw in manifest.get("records", []):
        record = dict(raw)
        partition = str(record.get("source_partition"))
        if partition in result:
            result[partition].append(record)
    for partition in result:
        result[partition].sort(key=lambda row: (str(row["task"]), int(row["partition_rank"]), str(row["source_sha256"])))
    expected = {"seen_train": 654, "seen_validation": 82, "v04_human_pool": 40, "v04_robot_dev": 20, "v04_robot_final": 80}
    require({name: len(rows) for name, rows in result.items()} == expected, "Stage-1 partition cardinality drift")
    return result


def materialize_partition_manifests(
    manifest: Mapping[str, Any], *, derived_root: Path, source_binding: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    grouped = _records_by_partition(manifest)
    bindings: dict[str, dict[str, Any]] = {}
    for partition in PARTITION_MANIFESTS:
        rows = grouped[partition]
        payload = {
            "schema_version": f"{SCHEMA}-partition-manifest",
            "status": "FROZEN",
            "partition": partition,
            "source_split_manifest": dict(source_binding),
            "protocol_sha256": manifest["protocol_sha256"],
            "split_sha256": manifest["split_sha256"],
            "episode_count": len(rows),
            "legal_window_count": sum(int(row["legal_window_count"]) for row in rows),
            "records": rows,
        }
        path = derived_root / "stage4/manifests" / f"{partition}.json"
        write_json_atomic(path, payload)
        bindings[partition] = bind_file(path)
    return bindings


def _source_path(record: Mapping[str, Any], source_root: Path) -> Path:
    projection = record.get("projection")
    if isinstance(projection, Mapping):
        return Path(str(projection["path"])).resolve()
    return (source_root / str(record["source_relative_path"])).resolve()


def _record_roles(record: Mapping[str, Any]) -> tuple[str, ...]:
    role = str(record["role"])
    return ("human", "robot") if role == "paired" else (role,)


def _dataset_names(record: Mapping[str, Any], role: str) -> tuple[str, str, str | None]:
    projected = isinstance(record.get("projection"), Mapping)
    if role == "human":
        return (
            "data/demo_0/human/hand_action_7d" if projected else "action",
            "data/demo_0/human/images" if projected else "cam_data/human_camera",
            None,
        )
    return (
        "data/demo_0/robot/observed_eef_pose_6d" if projected else "end_position",
        "data/demo_0/robot/images" if projected else "cam_data/robot_camera",
        "data/demo_0/robot/gripper_state" if projected else "gripper_state",
    )


def _legal_starts(record: Mapping[str, Any], path: Path) -> np.ndarray:
    with h5py.File(path, "r") as file:
        if isinstance(record.get("projection"), Mapping):
            starts = np.asarray(file["data/demo_0/time/legal_window_start"][:], dtype=np.int64)
        else:
            from tools.human2robot_v04_data import _time_structure

            _, _, starts = _time_structure(
                np.asarray(file["step"][:], dtype=np.int64),
                np.asarray(file["timestamp"][:], dtype=np.int64),
            )
    require(len(starts) == int(record["legal_window_count"]), f"Legal-window count drift: {path}")
    return starts


def _states_10d(file: h5py.File, record: Mapping[str, Any], role: str) -> np.ndarray:
    from cosmos_policy.datasets.human2robot_v04_retrieval import poses_euler_to_10d

    state_name, _, gripper_name = _dataset_names(record, role)
    state = np.asarray(file[state_name][:], dtype=np.float64)
    if role == "human":
        return poses_euler_to_10d(state[:, :6], state[:, 6])
    require(gripper_name is not None, "Robot gripper dataset is missing")
    return poses_euler_to_10d(state, np.asarray(file[gripper_name][:], dtype=np.float64))


def fit_seen_train_geometry(
    records: Sequence[Mapping[str, Any]], *, source_root: Path, output_path: Path, split_sha256: str
) -> dict[str, Any]:
    if output_path.is_file():
        payload = read_json(output_path)
        require(payload.get("split_sha256") == split_sha256, "Geometry statistics split drift")
        return payload
    count = 0
    total = np.zeros(10, dtype=np.float64)
    square = np.zeros(10, dtype=np.float64)
    for episode_index, record in enumerate(records, start=1):
        path = _source_path(record, source_root)
        starts = _legal_starts(record, path)
        with h5py.File(path, "r") as file:
            for role in _record_roles(record):
                states = _states_10d(file, record, role)
                histories = states[starts[:, None] + np.arange(H_STEPS)[None, :]]
                relative = histories - histories[:, -1:, :]
                flat = relative.reshape(-1, 10)
                require(bool(np.all(np.isfinite(flat))), f"Nonfinite seen-train geometry: {path}:{role}")
                total += flat.sum(axis=0, dtype=np.float64)
                square += np.square(flat, dtype=np.float64).sum(axis=0, dtype=np.float64)
                count += int(flat.shape[0])
        if episode_index % 25 == 0 or episode_index == len(records):
            print(json.dumps({"event": "geometry_progress", "episodes": episode_index, "total": len(records), "rows": count}))
    require(count > 0, "No seen-train geometry rows")
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    require(bool(np.all(np.isfinite(mean))) and bool(np.all(np.isfinite(std))), "Nonfinite geometry statistics")
    require(bool(np.all(std > 1e-12)), "Degenerate seen-train geometry statistics")
    payload = {
        "schema_version": f"{SCHEMA}-geometry-statistics",
        "status": "FROZEN",
        "created_at_utc": utc_now(),
        "split_sha256": split_sha256,
        "source_partition": "seen_train",
        "roles": ["human", "robot"],
        "H_steps": H_STEPS,
        "K_steps": K_STEPS,
        "algorithm": "all_gap_safe_legal_windows_relative_to_current_population_mean_std_float64",
        "relative_row_count": count,
        "mean_10d": mean.tolist(),
        "std_10d": std.tolist(),
        "future_rows_read": 0,
        "target_datasets_read": 0,
    }
    write_json_atomic(output_path, payload)
    return payload


def select_evenly_spaced_starts(starts: Sequence[int] | np.ndarray, count: int = SMOKE_WINDOWS_PER_EPISODE) -> list[int]:
    values = np.asarray(starts, dtype=np.int64)
    require(len(values) >= count, f"Fewer than {count} legal windows")
    indices = np.rint(np.linspace(0, len(values) - 1, count)).astype(np.int64)
    selected = [int(values[index]) for index in indices]
    require(len(selected) == count and len(set(selected)) == count, "Even window selection is not unique")
    return selected


def _feature_shard_path(feature_root: Path, record: Mapping[str, Any], role: str) -> Path:
    task = str(record["task"])
    safe_task = task.replace("/", "__")
    return feature_root / "wan_shards" / str(record["source_partition"]) / safe_task / f"{int(record['partition_rank']):04d}_{role}.npz"


def _read_feature_shard(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as stored:
        starts = np.asarray(stored["starts"], dtype=np.int64)
        features = np.asarray(stored["features"], dtype=np.float32)
        manifest = json.loads(str(stored["manifest_json"].item()))
    require(features.ndim == 2 and len(starts) == len(features), f"Invalid feature shard: {path}")
    require(bool(np.all(np.isfinite(features))), f"Nonfinite feature shard: {path}")
    return starts, features, manifest


def _make_wan_encoder() -> Any:
    require(Path("/.dockerenv").is_file(), "WAN feature materialization requires Docker")
    require(torch.cuda.is_available(), "WAN feature materialization requires CUDA")
    require(TOKENIZER_PATH.is_file(), f"Frozen tokenizer is missing: {TOKENIZER_PATH}")
    require(file_sha256(TOKENIZER_PATH) == TOKENIZER_SHA256, "Frozen tokenizer SHA256 drift")
    from cosmos_policy.tokenizers.wan2pt1 import Wan2pt1VAEInterface

    return Wan2pt1VAEInterface(chunk_duration=4, vae_pth=str(TOKENIZER_PATH), load_mean_std=False)


def _encode_frame_batch(tokenizer: Any, frames: np.ndarray) -> np.ndarray:
    from cosmos_policy.datasets.human2robot_p2_contract import preprocess_resolution_frames

    videos = []
    for anchor in frames:
        warmup = np.zeros_like(anchor)
        sequence = np.concatenate((warmup[None], np.repeat(anchor[None], 4, axis=0)), axis=0)
        videos.append(preprocess_resolution_frames(sequence, "center_crop_240x424_then_resize_224"))
    batch = torch.stack(videos).cuda(non_blocking=False).float().div_(127.5).sub_(1.0)
    with torch.inference_mode():
        latent = tokenizer.encode(batch)
        require(latent.ndim == 5 and latent.shape[0] == len(frames) and latent.shape[2] == 2, "WAN feature shape mismatch")
        pooled = latent[:, :, 1].float().mean(dim=(-1, -2))
        pooled = pooled / torch.linalg.vector_norm(pooled, dim=1, keepdim=True).clamp_min(1e-12)
    result = pooled.cpu().numpy().astype(np.float32)
    require(bool(np.all(np.isfinite(result))), "WAN encoder produced nonfinite features")
    return result


def materialize_visual_cache(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source_root: Path,
    feature_root: Path,
    split_sha256: str,
    batch_size: int,
    encoder: Any | None = None,
    worker_rank: int = 0,
    worker_world_size: int = 1,
    write_index: bool = True,
) -> dict[str, Any]:
    require(batch_size > 0, "Visual batch size must be positive")
    require(worker_world_size > 0 and 0 <= worker_rank < worker_world_size, "Invalid visual worker rank")
    tokenizer = encoder
    shard_bindings: list[dict[str, Any]] = []
    feature_count = 0
    episodes = [(partition, record, role) for partition in FEATURE_PARTITIONS for record in grouped[partition] for role in _record_roles(record)]
    assigned = [row for index, row in enumerate(episodes) if index % worker_world_size == worker_rank]
    for episode_index, (partition, record, role) in enumerate(assigned, start=1):
        output = _feature_shard_path(feature_root, record, role)
        path = _source_path(record, source_root)
        starts = _legal_starts(record, path)
        expected = {
            "schema_version": f"{SCHEMA}-wan-shard",
            "split_sha256": split_sha256,
            "source_sha256": str(record["source_sha256"]),
            "source_relative_path": str(record["source_relative_path"]),
            "source_partition": partition,
            "task": str(record["task"]),
            "role": role,
            "H_steps": H_STEPS,
            "K_steps": K_STEPS,
            "tokenizer_checkpoint_sha256": TOKENIZER_SHA256,
            "visual_dataset": _dataset_names(record, role)[1],
            "visual_row_rule": "legal_window_start_plus_7_current_only",
            "future_frames_read": 0,
            "target_datasets_read": 0,
            "feature_count": len(starts),
        }
        if output.is_file():
            stored_starts, stored_features, stored_manifest = _read_feature_shard(output)
            require(stored_manifest == expected, f"WAN shard manifest drift: {output}")
            require(np.array_equal(stored_starts, starts), f"WAN shard starts drift: {output}")
            require(len(stored_features) == len(starts), f"WAN shard count drift: {output}")
        else:
            if tokenizer is None:
                tokenizer = _make_wan_encoder()
            batches: list[np.ndarray] = []
            _, image_name, _ = _dataset_names(record, role)
            with h5py.File(path, "r") as file:
                images = file[image_name]
                rows = starts + (H_STEPS - 1)
                for offset in range(0, len(rows), batch_size):
                    batch_rows = rows[offset : offset + batch_size]
                    anchors = np.stack([np.asarray(images[int(row)], dtype=np.uint8) for row in batch_rows])
                    batches.append(_encode_frame_batch(tokenizer, anchors) if encoder is None else np.asarray(tokenizer(anchors), dtype=np.float32))
            features = np.concatenate(batches, axis=0)
            require(len(features) == len(starts), f"WAN shard cardinality mismatch: {output}")
            write_npz_atomic(output, starts=starts, features=features, manifest=expected)
        binding = bind_file(output)
        binding.update({"partition": partition, "task": record["task"], "role": role, "feature_count": len(starts)})
        shard_bindings.append(binding)
        feature_count += len(starts)
        if episode_index % 10 == 0 or episode_index == len(assigned):
            print(json.dumps({"event": "wan_progress", "worker_rank": worker_rank, "episode_shards": episode_index, "total_shards": len(assigned), "features": feature_count}), flush=True)
    if not write_index:
        return {
            "schema_version": f"{SCHEMA}-wan-worker-result",
            "status": "PASSED",
            "worker_rank": worker_rank,
            "worker_world_size": worker_world_size,
            "shard_count": len(shard_bindings),
            "feature_count": feature_count,
            "future_frames_read": 0,
            "target_datasets_read": 0,
        }
    require(worker_rank == 0 and worker_world_size == 1, "Only the single-process verifier may write the WAN cache index")
    index_path = feature_root / "wan_cache_index.json"
    existing_index = read_json(index_path) if index_path.is_file() else None
    index = {
        "schema_version": f"{SCHEMA}-wan-cache-index",
        "status": "FROZEN",
        "created_at_utc": existing_index["created_at_utc"] if existing_index else utc_now(),
        "split_sha256": split_sha256,
        "tokenizer_checkpoint": bind_file(TOKENIZER_PATH),
        "shard_count": len(shard_bindings),
        "feature_count": feature_count,
        "future_frames_read": 0,
        "target_datasets_read": 0,
        "shards": shard_bindings,
    }
    index["shard_bundle_sha256"] = canonical_sha256(shard_bindings)
    write_json_atomic(index_path, index)
    return index


def run_visual_feature_workers(
    *, workspace: Path, feature_root: Path, source_root: Path, batch_size: int
) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--master_addr=127.0.0.1",
        "--master_port=29541",
        "--nproc_per_node=4",
        "tools/human2robot_v04_stage4_feature_worker.py",
        "--workspace",
        str(workspace),
        "--feature-root",
        str(feature_root),
        "--source-root",
        str(source_root),
        "--batch-size",
        str(batch_size),
    ]
    print(json.dumps({"event": "wan_workers_start", "command": command}), flush=True)
    process = subprocess.run(command, cwd=workspace, env=os.environ.copy(), check=False)
    require(process.returncode == 0, f"Parallel WAN feature workers failed: {process.returncode}")


def _feature_lookup(feature_root: Path, record: Mapping[str, Any], role: str) -> dict[int, np.ndarray]:
    starts, features, _ = _read_feature_shard(_feature_shard_path(feature_root, record, role))
    return {int(start): feature for start, feature in zip(starts, features, strict=True)}


def _closest_phase_start(record: Mapping[str, Any], starts: np.ndarray, phase: float) -> int:
    frame_count = int(record["frame_count"])
    phases = (starts + H_STEPS) / frame_count
    distance = np.abs(phases - phase)
    best = np.flatnonzero(distance == distance.min())
    return int(starts[int(best[0])])


def _retrieval_feature(window: Any, visual: np.ndarray, geometry: Mapping[str, Any]) -> RetrievalFeature:
    history, _, provenance = read_feature_inputs(window)
    from cosmos_policy.datasets.human2robot_v04_retrieval import geometry_feature, visual_feature_from_wan_latent

    frozen = FeatureProvenance(**{**asdict(provenance), "visual_feature_kind": "frozen_wan_latent"})
    return RetrievalFeature(
        geometry=geometry_feature(history, np.asarray(geometry["mean_10d"]), np.asarray(geometry["std_10d"])),
        visual=visual_feature_from_wan_latent(visual),
        provenance=frozen,
    )


def build_smoke_plan(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    feature_root: Path,
    geometry: Mapping[str, Any],
    output_path: Path,
    split_sha256: str,
) -> dict[str, Any]:
    if output_path.is_file():
        return read_json(output_path)
    pools: dict[str, list[Mapping[str, Any]]] = {}
    for record in grouped["v04_human_pool"]:
        pools.setdefault(str(record["task"]), []).append(record)
    require(all(len(rows) == POOL_SIZE for rows in pools.values()), "Human pool is not pool10 per task")
    visual_cache: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    queries: list[dict[str, Any]] = []
    for query_record in grouped["v04_robot_dev"]:
        query_path = Path(str(query_record["projection"]["path"]))
        query_starts = _legal_starts(query_record, query_path)
        selected_starts = select_evenly_spaced_starts(query_starts)
        qkey = (str(query_record["source_sha256"]), "robot")
        visual_cache[qkey] = _feature_lookup(feature_root, query_record, "robot")
        task_pool = pools[str(query_record["task"])]
        for query_start in selected_starts:
            query_window = window_from_manifest_record(query_record, query_start)
            features: dict[str, RetrievalFeature] = {
                query_window.window_id: _retrieval_feature(query_window, visual_cache[qkey][query_start], geometry)
            }
            candidates = []
            candidate_records: dict[str, Mapping[str, Any]] = {}
            for candidate_record in task_pool:
                cpath = Path(str(candidate_record["projection"]["path"]))
                starts = _legal_starts(candidate_record, cpath)
                candidate_start = _closest_phase_start(candidate_record, starts, query_window.phase)
                candidate = window_from_manifest_record(candidate_record, candidate_start)
                ckey = (str(candidate_record["source_sha256"]), "human")
                if ckey not in visual_cache:
                    visual_cache[ckey] = _feature_lookup(feature_root, candidate_record, "human")
                features[candidate.window_id] = _retrieval_feature(candidate, visual_cache[ckey][candidate_start], geometry)
                candidates.append(candidate)
                candidate_records[candidate.window_id] = candidate_record
            ranked = rank_geometry_plus_visual(query_window, candidates, features, pool_size=POOL_SIZE, top_k=TOP_K)
            require(len(ranked) == TOP_K, f"Top-k incomplete: {query_window.window_id}")
            rows = []
            for record in ranked:
                candidate = next(item for item in candidates if item.window_id == record.candidate_id)
                rows.append(
                    {
                        "candidate_record": dict(candidate_records[candidate.window_id]),
                        "candidate_start": int(candidate.history_rows[0]),
                        "retrieval": json.loads(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)),
                    }
                )
            queries.append(
                {
                    "query_record": dict(query_record),
                    "query_start": query_start,
                    "query_id": query_window.window_id,
                    "task": query_window.task,
                    "episode_id": query_window.episode_id,
                    "ranks": rows,
                }
            )
    queries.sort(key=lambda row: (row["task"], row["episode_id"], row["query_start"]))
    require(len(queries) == EXPECTED_QUERY_COUNT, f"Expected {EXPECTED_QUERY_COUNT} smoke queries, got {len(queries)}")
    payload = {
        "schema_version": f"{SCHEMA}-smoke-plan",
        "status": "FROZEN",
        "created_at_utc": utc_now(),
        "split_sha256": split_sha256,
        "query_partition": "v04_robot_dev",
        "candidate_partition": "v04_human_pool",
        "episode_count": len(grouped["v04_robot_dev"]),
        "query_count": len(queries),
        "rank_inference_count_per_method": len(queries) * TOP_K,
        "windows_per_episode": SMOKE_WINDOWS_PER_EPISODE,
        "window_selection": "round(linspace(0,N-1,8)) over sorted legal_window_start",
        "candidate_window_selection": "closest current-frame normalized phase per frozen pool episode; earliest legal start breaks ties",
        "retrieval_modality": "geometry_plus_visual",
        "pool_size": POOL_SIZE,
        "top_k": TOP_K,
        "seed": RUN_SEED,
        "future_rows_read_for_retrieval": 0,
        "target_datasets_read_for_retrieval": 0,
        "queries": queries,
    }
    write_json_atomic(output_path, payload)
    return payload


def _checkpoint_bindings(workspace: Path) -> list[dict[str, Any]]:
    frozen = read_json(workspace / "方案/v04/v03_frozen_manifest.json")
    rows = list(frozen.get("checkpoints", []))
    require([str(row["method"]) for row in rows] == list(METHODS), "Frozen v03 checkpoint order drift")
    for row in rows:
        path = Path(str(row["path"]))
        require(path.is_dir(), f"Frozen checkpoint is missing: {path}")
        require(int(row["optimizer_step"]) == 7000 and int(row["seed"]) == RUN_SEED, "Old checkpoint identity drift")
    return rows


def run_smoke_workers(
    *, workspace: Path, run_root: Path, feature_root: Path, smoke_plan_path: Path
) -> list[dict[str, Any]]:
    checkpoints = _checkpoint_bindings(workspace)
    summaries = []
    for method_index, row in enumerate(checkpoints):
        method = str(row["method"])
        output_root = run_root / "stage4/smoke" / method
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master_addr=127.0.0.1",
            f"--master_port={29542 + method_index}",
            "--nproc_per_node=4",
            "tools/human2robot_v04_stage4_worker.py",
            "--workspace",
            str(workspace),
            "--feature-root",
            str(feature_root),
            "--smoke-plan",
            str(smoke_plan_path),
            "--method",
            method,
            "--checkpoint",
            str(row["path"]),
            "--checkpoint-payload-sha256",
            str(row["payload_sha256"]),
            "--output-root",
            str(output_root),
        ]
        print(json.dumps({"event": "smoke_worker_start", "method": method, "command": command}))
        process = subprocess.run(command, cwd=workspace, env=os.environ.copy(), check=False)
        require(process.returncode == 0, f"Stage-4 smoke worker failed for {method}: {process.returncode}")
        summary_path = output_root / "summary.json"
        summary = read_json(summary_path)
        require(summary.get("status") == "PASSED", f"Stage-4 smoke did not pass for {method}")
        require(int(summary.get("receipt_count", 0)) == EXPECTED_RECEIPTS_PER_METHOD, f"Smoke receipt count mismatch for {method}")
        summaries.append(bind_file(summary_path))
    return summaries


def build_protocol_lock(
    *,
    workspace: Path,
    derived_root: Path,
    feature_root: Path,
    manifest: Mapping[str, Any],
    partition_bindings: Mapping[str, Any],
    geometry_binding: Mapping[str, Any],
    visual_index_binding: Mapping[str, Any],
    smoke_plan_binding: Mapping[str, Any],
    smoke_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    data_protocol = {
        "source_split_manifest_sha256": STAGE1_MANIFEST_SHA256,
        "protocol_sha256": manifest["protocol_sha256"],
        "split_sha256": manifest["split_sha256"],
        "partition_manifests": dict(partition_bindings),
    }
    retrieval_protocol = {
        "H_steps": H_STEPS,
        "K_steps": K_STEPS,
        "modality": "geometry_plus_visual",
        "geometry": dict(geometry_binding),
        "visual_cache": dict(visual_index_binding),
        "pool_size": POOL_SIZE,
        "top_k": TOP_K,
        "fusion": "equal_weight_concatenation_1_over_sqrt2",
        "tie_break": "SHA256(seed,query_id,human_content_sha256)",
        "seed": RUN_SEED,
    }
    training_protocol = {
        "methods": list(METHODS),
        "order": list(METHODS),
        "initialization": "/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
        "seed": RUN_SEED,
        "gpu_count": 4,
        "batch_per_gpu": 25,
        "gradient_accumulation": 2,
        "effective_batch": 200,
        "optimizer_steps": 7000,
        "H_steps": H_STEPS,
        "K_steps": K_STEPS,
        "top_k": TOP_K,
        "resolution": [224, 224],
        "learning_rate": 1e-4,
        "action_loss_multiplier": 16,
        "text_conditioning": "disabled_zero_embedding",
        "checkpoint_selection": "fixed_step_7000",
        "heldout_training_or_selection_allowed": False,
    }
    final_protocol = {
        "dev": {"episodes": 20, "queries_per_episode": 8, "queries": 160},
        "final": {"tasks": 4, "episodes_per_task": 20, "queries_per_episode": 8, "queries": 640},
        "top_k": TOP_K,
        "primary_pool_size": POOL_SIZE,
        "pool_curve": [1, 2, 4, 8, 10],
        "bootstrap_replicates": 10000,
        "bootstrap_seed": RUN_SEED,
        "oracle_phase_after_primary_only": True,
    }
    payload = {
        "schema_version": f"{SCHEMA}-protocol-lock",
        "status": "VERIFIED_STAGE4",
        "created_at_utc": utc_now(),
        "data_protocol": data_protocol,
        "data_protocol_sha256": canonical_sha256(data_protocol),
        "retrieval_protocol": retrieval_protocol,
        "retrieval_protocol_sha256": canonical_sha256(retrieval_protocol),
        "training_protocol": training_protocol,
        "training_protocol_sha256": canonical_sha256(training_protocol),
        "final_evaluation_protocol": final_protocol,
        "final_evaluation_protocol_sha256": canonical_sha256(final_protocol),
        "smoke_plan": dict(smoke_plan_binding),
        "smoke_summaries": [dict(row) for row in smoke_summaries],
        "provenance_violation_count": 0,
        "future_target_independence_violation_count": 0,
        "nonfinite_count": 0,
        "missing_receipt_count": 0,
        "gap_crossing_count": 0,
        "training_allowed": True,
        "stage5_allowed": True,
        "formal_performance_result": False,
        "performance_claim_allowed": False,
    }
    path = derived_root / "stage4_protocol_lock.json"
    write_json_atomic(path, payload)
    return payload


def run_stage4(
    *,
    workspace: Path,
    run_root: Path,
    derived_root: Path,
    feature_root: Path,
    source_root: Path = Path("/DATA1/wxs/DATASETS/Human2Robot/data/v1"),
    visual_batch_size: int = 32,
    run_smoke: bool = True,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    run_root = run_root.resolve()
    derived_root = derived_root.resolve()
    feature_root = feature_root.resolve()
    require(Path("/.dockerenv").is_file(), "Formal stage 4 must run inside Docker")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 4, "Formal stage 4 requires exactly four visible GPUs")
    frozen_inputs = _validate_frozen_inputs(workspace, run_root)
    manifest = read_json(Path(str(frozen_inputs["stage1_manifest"]["path"])))
    grouped = _records_by_partition(manifest)
    partition_bindings = materialize_partition_manifests(
        manifest, derived_root=derived_root, source_binding=frozen_inputs["stage1_manifest"]
    )
    geometry_path = feature_root / "geometry_statistics.json"
    geometry = fit_seen_train_geometry(
        grouped["seen_train"], source_root=source_root, output_path=geometry_path, split_sha256=str(manifest["split_sha256"])
    )
    run_visual_feature_workers(
        workspace=workspace,
        feature_root=feature_root,
        source_root=source_root,
        batch_size=visual_batch_size,
    )
    visual_index = materialize_visual_cache(
        grouped,
        source_root=source_root,
        feature_root=feature_root,
        split_sha256=str(manifest["split_sha256"]),
        batch_size=visual_batch_size,
    )
    smoke_plan_path = derived_root / "stage4_smoke_plan.json"
    smoke_plan = build_smoke_plan(
        grouped,
        feature_root=feature_root,
        geometry=geometry,
        output_path=smoke_plan_path,
        split_sha256=str(manifest["split_sha256"]),
    )
    require(smoke_plan["query_count"] == EXPECTED_QUERY_COUNT, "Smoke plan query count drift")
    if not run_smoke:
        return {
            "schema_version": SCHEMA,
            "status": "FEATURES_PREPARED_SMOKE_PENDING",
            "features_generated": True,
            "training_started": False,
            "evaluation_started": False,
            "training_allowed": False,
            "geometry_statistics": bind_file(geometry_path),
            "visual_cache_index": bind_file(feature_root / "wan_cache_index.json"),
            "smoke_plan": bind_file(smoke_plan_path),
        }
    smoke_summaries = run_smoke_workers(
        workspace=workspace, run_root=run_root, feature_root=feature_root, smoke_plan_path=smoke_plan_path
    )
    lock = build_protocol_lock(
        workspace=workspace,
        derived_root=derived_root,
        feature_root=feature_root,
        manifest=manifest,
        partition_bindings=partition_bindings,
        geometry_binding=bind_file(geometry_path),
        visual_index_binding=bind_file(feature_root / "wan_cache_index.json"),
        smoke_plan_binding=bind_file(smoke_plan_path),
        smoke_summaries=smoke_summaries,
    )
    return {
        "schema_version": SCHEMA,
        "status": "PASSED",
        "formal_result": False,
        "features_generated": True,
        "training_started": False,
        "evaluation_started": True,
        "old_checkpoint_read_only_smoke": True,
        "performance_claim_allowed": False,
        "training_allowed": True,
        "stage5_allowed": True,
        "frozen_inputs": frozen_inputs,
        "partition_manifests": partition_bindings,
        "geometry_statistics": bind_file(geometry_path),
        "visual_cache_index": bind_file(feature_root / "wan_cache_index.json"),
        "smoke_plan": bind_file(smoke_plan_path),
        "smoke_summaries": smoke_summaries,
        "protocol_lock": bind_file(derived_root / "stage4_protocol_lock.json"),
        "protocol_sha256s": {
            "data": lock["data_protocol_sha256"],
            "retrieval": lock["retrieval_protocol_sha256"],
            "training": lock["training_protocol_sha256"],
            "final_evaluation": lock["final_evaluation_protocol_sha256"],
        },
        "guardrails": {
            "provenance_violation_count": 0,
            "future_target_independence_violation_count": 0,
            "nonfinite_count": 0,
            "missing_receipt_count": 0,
            "gap_crossing_count": 0,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--run-root", type=Path, default=Path("/DATA1/wxs/ReCAP_M5B_V04_RUNS"))
    parser.add_argument("--derived-root", type=Path, default=Path("/workspace/data/Human2Robot/derived/v04"))
    parser.add_argument("--feature-root", type=Path, default=Path("/DATA1/wxs/ReCAP_M5B_V04_RUNS/features"))
    parser.add_argument("--source-root", type=Path, default=Path("/DATA1/wxs/DATASETS/Human2Robot/data/v1"))
    parser.add_argument("--visual-batch-size", type=int, default=32)
    parser.add_argument("--skip-smoke", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_stage4(
        workspace=args.workspace,
        run_root=args.run_root,
        derived_root=args.derived_root,
        feature_root=args.feature_root,
        source_root=args.source_root,
        visual_batch_size=args.visual_batch_size,
        run_smoke=not args.skip_smoke,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Stage4Error as error:
        print(f"stage-4 error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
