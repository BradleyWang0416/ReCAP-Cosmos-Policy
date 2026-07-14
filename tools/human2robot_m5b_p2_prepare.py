#!/usr/bin/env python3
"""Materialize and verify train-only M5B-P2 statistics and retrieval indices.

Formal artifacts are produced only inside the full Docker environment.  The
visual branch loads the frozen local WAN tokenizer and never permits a remote
weight URI or an unencoded substitute feature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from cosmos_policy.datasets.human2robot_dataset import _preprocess_video, align_pool_chunk
from cosmos_policy.datasets.human2robot_p2_contract import preprocess_resolution_frames
from cosmos_policy.datasets.human2robot_p2_dataset import Human2RobotP2Dataset, P2Window
from cosmos_policy.datasets.human2robot_p2_specs import P2TrainingSpec, p2_training_specs

SCHEMA_VERSION = "human2robot-m5b-p2-prepared-artifacts-v2"
INDEX_SCHEMA_VERSION = "human2robot-m5b-p2-retrieval-index-v1"
STATISTICS_SCHEMA_VERSION = "human2robot-m5b-p2-train-statistics-v1"
CELL_RECEIPT_SCHEMA_VERSION = "human2robot-m5b-p2-cell-receipt-v1"
VISUAL_CACHE_SCHEMA_VERSION = "human2robot-m5b-p2-visual-context-cache-v1"
PROGRESS_SCHEMA_VERSION = "human2robot-m5b-p2-materialize-progress-v1"
PROTOCOL_SHA256 = "7598dfa2ac2e129f5d21a295dad23b90f63c3c8e68811da73cbcc20eb95d5ce4"
SUPPLEMENT_SHA256 = "17d9fc308c50b9b7899793a4c8d3bca1eeba217053fbacb368e2f9a2e390d7ab"
REGISTRY_SHA256 = "502cc57d41c7e4829e872ac95a258d7dc1e8d0d8a27ddfc3cf0315d4d31ef2d6"
SPLIT_SHA256 = "1d3ef2377aa19938b06646f6d5fc31ec9f275fc9f37e253e1e9aa5eecdc5a968"
POOL_MANIFEST_SHA256 = "47e87be5800194de6e0ac99b47dbe23ef96a91298edbff3e9996b1484b489299"
TOKENIZER_PATH = Path("/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth")
TOKENIZER_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
VISUAL_MODALITIES = {"visual", "geometry_plus_visual"}


class PreparationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_npz_atomic(
    path: Path, ids: list[str], features: np.ndarray, manifest: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            ids=np.asarray(ids, dtype=np.str_),
            features=np.asarray(features, dtype=np.float32),
            manifest_json=np.asarray(json.dumps(manifest, sort_keys=True), dtype=np.str_),
        )
    os.replace(temporary, path)


def record_progress(
    output_root: Path,
    *,
    run_id: str,
    phase: str,
    completed_cells: int,
    total_cells: int,
    cell_id: str | None = None,
    detail: str | None = None,
) -> None:
    event = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "updated_at_utc": utc_now(),
        "run_id": run_id,
        "pid": os.getpid(),
        "phase": phase,
        "completed_cells": completed_cells,
        "total_cells": total_cells,
        "cell_id": cell_id,
        "detail": detail,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_root / "materialize_progress.json", event)
    with (output_root / "materialize.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    print(
        "[M5B-P2] "
        f"phase={phase} completed={completed_cells}/{total_cells} "
        f"cell={cell_id or '-'} detail={detail or '-'}",
        flush=True,
    )


def cell_artifact_paths(output_root: Path, spec: P2TrainingSpec) -> dict[str, Path]:
    return {
        "index": output_root / "indices" / f"{spec.cell_id}.npz",
        "statistics": output_root / "statistics" / f"{spec.cell_id}.json",
        "receipt": output_root / "receipts" / f"{spec.cell_id}.json",
    }


def load_completed_cell_receipt(
    workspace: Path, output_root: Path, spec: P2TrainingSpec
) -> dict[str, Any] | None:
    artifact_paths = cell_artifact_paths(output_root, spec)
    receipt_path = artifact_paths["receipt"]
    if not receipt_path.is_file():
        return None
    receipt = read_json(receipt_path)
    require(
        receipt.get("schema_version") == CELL_RECEIPT_SCHEMA_VERSION,
        f"Cell receipt schema drift: {spec.cell_id}",
    )
    require(receipt.get("status") == "complete", f"Cell receipt incomplete: {spec.cell_id}")
    require(receipt.get("formal_result") is False, f"Cell receipt claim drift: {spec.cell_id}")
    require(receipt.get("cell_id") == spec.cell_id, f"Cell receipt ID drift: {spec.cell_id}")
    frozen_bindings = {
        "protocol_file_sha256": PROTOCOL_SHA256,
        "supplement_file_sha256": SUPPLEMENT_SHA256,
        "registry_file_sha256": REGISTRY_SHA256,
        "split_sha256": SPLIT_SHA256,
        "pool_manifest_sha256": POOL_MANIFEST_SHA256,
    }
    for key, expected in frozen_bindings.items():
        require(receipt.get(key) == expected, f"Cell receipt {key} drift: {spec.cell_id}")
    entry = receipt.get("entry")
    require(isinstance(entry, dict), f"Cell receipt entry missing: {spec.cell_id}")
    require(entry.get("cell_id") == spec.cell_id, f"Cell receipt entry ID drift: {spec.cell_id}")
    require(entry.get("spec") == asdict(spec), f"Cell receipt spec drift: {spec.cell_id}")
    require(entry.get("config_name") == spec.config_name, f"Cell config drift: {spec.cell_id}")
    expected_statistics = str(artifact_paths["statistics"].relative_to(workspace))
    expected_index = str(artifact_paths["index"].relative_to(workspace))
    require(entry.get("statistics_path") == expected_statistics, f"Statistics path drift: {spec.cell_id}")
    require(entry.get("retrieval_index_path") == expected_index, f"Index path drift: {spec.cell_id}")
    require(receipt.get("statistics_path") == expected_statistics, f"Receipt statistics path drift: {spec.cell_id}")
    require(receipt.get("retrieval_index_path") == expected_index, f"Receipt index path drift: {spec.cell_id}")
    for kind, entry_hash_key, receipt_hash_key in (
        ("statistics", "statistics_sha256", "statistics_sha256"),
        ("index", "retrieval_index_sha256", "retrieval_index_sha256"),
    ):
        path = artifact_paths[kind]
        require(path.is_file(), f"Receipt-bound artifact missing: {path}")
        actual = file_sha256(path)
        require(entry.get(entry_hash_key) == actual, f"Entry hash drift: {path}")
        require(receipt.get(receipt_hash_key) == actual, f"Receipt hash drift: {path}")
    return entry


def write_completed_cell_receipt(
    workspace: Path,
    output_root: Path,
    spec: P2TrainingSpec,
    entry: dict[str, Any],
) -> None:
    artifact_paths = cell_artifact_paths(output_root, spec)
    payload = {
        "schema_version": CELL_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "formal_result": False,
        "created_at_utc": utc_now(),
        "cell_id": spec.cell_id,
        "protocol_file_sha256": PROTOCOL_SHA256,
        "supplement_file_sha256": SUPPLEMENT_SHA256,
        "registry_file_sha256": REGISTRY_SHA256,
        "split_sha256": SPLIT_SHA256,
        "pool_manifest_sha256": POOL_MANIFEST_SHA256,
        "statistics_path": str(artifact_paths["statistics"].relative_to(workspace)),
        "statistics_sha256": entry["statistics_sha256"],
        "retrieval_index_path": str(artifact_paths["index"].relative_to(workspace)),
        "retrieval_index_sha256": entry["retrieval_index_sha256"],
        "entry": entry,
    }
    write_json_atomic(artifact_paths["receipt"], payload)


def workspace_paths(workspace: Path) -> dict[str, Path]:
    human_root = workspace / "data/Human2Robot"
    return {
        "canonical_root": human_root / "canonical/v3",
        "main_view_path": human_root
        / "derived/views/nominal_camera_30hz_segmented"
        / "human_hand_robot_frame_raw/robot_ee_observed_t_plus_1_bc_proxy"
        / "train_only_tplus1_query_anchor_se3_identity_scale_v1",
        "m3_report_path": human_root / "derived/m3_v03/m3_validation_report.json",
        "m4_report_path": human_root / "derived/m4_v03/m4_launch_report.json",
        "protocol_path": workspace / "方案/v03/M5B_formal_acceptance_protocol_v1.json",
        "supplement_path": workspace / "方案/v03/M5B_P2_execution_supplement_v2.json",
        "registry_path": workspace / "方案/v03/M5B_P2_cell_registry_v2.json",
        "p1_pool_root": human_root / "derived/m5b_v03/p1_human_only_pool",
        "output_root": human_root / "derived/m5b_v03/p2_prepared_v2",
    }


def validate_frozen_inputs(workspace: Path) -> dict[str, Path]:
    paths = workspace_paths(workspace)
    require(file_sha256(paths["protocol_path"]) == PROTOCOL_SHA256, "Protocol hash drift")
    require(file_sha256(paths["supplement_path"]) == SUPPLEMENT_SHA256, "Supplement hash drift")
    require(file_sha256(paths["registry_path"]) == REGISTRY_SHA256, "Registry hash drift")
    split = read_json(paths["canonical_root"] / "task_split_manifest.json")
    require(split.get("split_sha256") == SPLIT_SHA256, "Split hash drift")
    require(
        file_sha256(paths["p1_pool_root"] / "pool_manifest.json") == POOL_MANIFEST_SHA256,
        "P1 pool hash drift",
    )
    return paths


def placeholder_statistics() -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "bootstrap-not-formal",
        "provenance": {"heldout_data_used": False},
    }
    for prefix in (
        "pool_action_10d",
        "query_bc_target_10d",
        "residual_10d",
        "future_state_transition_10d",
    ):
        result[f"{prefix}_min"] = [-10.0] * 10
        result[f"{prefix}_max"] = [10.0] * 10
    return result


def placeholder_index_manifest() -> dict[str, Any]:
    return {
        "schema_version": "bootstrap-not-formal",
        "heldout_target_used": False,
        "split_sha256": SPLIT_SHA256,
        "geometry_relative_10d_mean": [0.0] * 10,
        "geometry_relative_10d_std": [1.0] * 10,
    }


def dataset_kwargs(
    paths: dict[str, Path],
    spec: P2TrainingSpec,
    *,
    split: str,
    statistics_path: Path,
    index_path: Path,
    retrieval_modality: str | None = None,
    use_image_aug: bool = False,
) -> dict[str, Any]:
    return {
        "canonical_root": paths["canonical_root"],
        "main_view_path": paths["main_view_path"],
        "m3_report_path": paths["m3_report_path"],
        "m4_report_path": paths["m4_report_path"],
        "protocol_path": paths["protocol_path"],
        "supplement_path": paths["supplement_path"],
        "p1_pool_root": paths["p1_pool_root"],
        "split": split,
        "method_id": spec.method_id,
        "experiment_id": spec.experiment_id,
        "variant_id": spec.variant_id,
        "seed": spec.seed,
        "h_steps": spec.h_steps,
        "k_steps": spec.k_steps,
        "window_stride": 8,
        "top_k": spec.top_k,
        "pool_size": spec.pool_size,
        "retrieval_modality": retrieval_modality or spec.retrieval_modality,
        "time_view_id": spec.time_view_id,
        "query_offset_view_steps": spec.query_offset_view_steps,
        "target_representation": spec.target_representation,
        "statistics_path": statistics_path,
        "retrieval_index_path": index_path,
        "resolution_variant": "center_crop_240x424_then_resize_224",
        "use_image_aug": use_image_aug,
        "num_duplicates_per_image": 4,
        "text_conditioning": "disabled_zero_embedding",
    }


def context_key(spec: P2TrainingSpec) -> tuple[int, int, str, int]:
    return (spec.h_steps, spec.k_steps, spec.time_view_id, spec.query_offset_view_steps)


def build_window_context(
    paths: dict[str, Path], spec: P2TrainingSpec, staging_root: Path
) -> tuple[Human2RobotP2Dataset, Human2RobotP2Dataset]:
    key = "__".join(str(value) for value in context_key(spec))
    stats_path = staging_root / f"bootstrap_statistics__{key}.json"
    index_path = staging_root / f"bootstrap_index__{key}.npz"
    if not stats_path.exists():
        write_json_atomic(stats_path, placeholder_statistics())
    if not index_path.exists():
        write_npz_atomic(
            index_path,
            [],
            np.empty((0, 0), dtype=np.float32),
            placeholder_index_manifest(),
        )
    context_spec = replace(
        spec,
        method_id="recap_hand_ret",
        target_representation="residual",
        retrieval_modality="phase",
        seed=20260711,
    )
    train = Human2RobotP2Dataset(
        **dataset_kwargs(
            paths,
            context_spec,
            split="train",
            statistics_path=stats_path,
            index_path=index_path,
            retrieval_modality="phase",
        )
    )
    heldout = Human2RobotP2Dataset(
        **dataset_kwargs(
            paths,
            context_spec,
            split="heldout",
            statistics_path=stats_path,
            index_path=index_path,
            retrieval_modality="phase",
        )
    )
    return train, heldout


def _state_cache_get(
    cache: dict[tuple[str, str], np.ndarray],
    dataset: Human2RobotP2Dataset,
    window: P2Window,
    role: str,
) -> np.ndarray:
    key = (str(window.path), role)
    if key not in cache:
        cache[key] = dataset._states(window, role)
    return cache[key]


def geometry_statistics(
    train: Human2RobotP2Dataset,
) -> tuple[np.ndarray, np.ndarray, int]:
    cache: dict[tuple[str, str], np.ndarray] = {}
    relative_rows = []
    for role, windows in (("robot", train.queries), ("human", train.candidates)):
        for window in windows:
            states = _state_cache_get(cache, train, window, role)
            history = states[window.history_rows]
            relative_rows.append(history - history[-1])
    values = np.concatenate(relative_rows, axis=0).astype(np.float64)
    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0), 1e-6)
    require(np.all(np.isfinite(mean)) and np.all(np.isfinite(std)), "Nonfinite geometry statistics")
    return mean, std, len(values)


def feature_windows(
    train: Human2RobotP2Dataset, heldout: Human2RobotP2Dataset
) -> list[tuple[str, Human2RobotP2Dataset, P2Window, str]]:
    records: dict[str, tuple[str, Human2RobotP2Dataset, P2Window, str]] = {}
    for dataset in (train, heldout):
        for window in dataset.queries:
            feature_id = f"robot:{window.window_id}"
            records[feature_id] = (feature_id, dataset, window, "robot")
        for window in dataset.candidates:
            feature_id = f"human:{window.window_id}"
            records[feature_id] = (feature_id, dataset, window, "human")
    return [records[key] for key in sorted(records)]


def encode_visual_features(
    records: list[tuple[str, Human2RobotP2Dataset, P2Window, str]],
    *,
    batch_size: int,
    resolution_variant: str = "center_crop_240x424_then_resize_224",
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    require(Path("/.dockerenv").is_file(), "Visual features require the full Docker environment")
    require(torch.cuda.is_available(), "A CUDA GPU is required for the frozen WAN encoder")
    require(TOKENIZER_PATH.is_file(), f"Missing frozen tokenizer: {TOKENIZER_PATH}")
    require(file_sha256(TOKENIZER_PATH) == TOKENIZER_SHA256, "Tokenizer checkpoint hash drift")
    require(batch_size > 0, "Visual batch size must be positive")

    from cosmos_policy.tokenizers.wan2pt1 import Wan2pt1VAEInterface

    tokenizer = Wan2pt1VAEInterface(
        chunk_duration=4,
        vae_pth=str(TOKENIZER_PATH),
        load_mean_std=False,
    )
    result: dict[str, np.ndarray] = {}
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        videos = []
        for _, dataset, window, role in chunk:
            _, images, _ = dataset._states_and_images(window, role)
            anchor = images[window.current_row]
            # WAN's causal encoder accepts 1+4n frames.  The formal 37-frame
            # graph likewise starts with one blank warm-up frame, so encode one
            # blank followed by exactly four copies of the anchor and select
            # latent frame 1 (the anchor slot), never the warm-up latent.
            warmup = np.zeros_like(anchor)
            frames = np.concatenate((warmup[None], np.repeat(anchor[None], 4, axis=0)), axis=0)
            videos.append(preprocess_resolution_frames(frames, resolution_variant))
        batch = torch.stack(videos).cuda(non_blocking=False).float().div_(127.5).sub_(1.0)
        with torch.inference_mode():
            latent = tokenizer.encode(batch)
            require(
                latent.ndim == 5 and latent.shape[0] == len(chunk) and latent.shape[2] == 2,
                "WAN feature shape mismatch",
            )
            pooled = latent[:, :, 1].float().mean(dim=(-1, -2))
            pooled = pooled / torch.linalg.vector_norm(pooled, dim=1, keepdim=True).clamp_min(1e-12)
        values = pooled.cpu().numpy().astype(np.float32)
        require(np.all(np.isfinite(values)), "WAN encoder produced nonfinite visual features")
        for (feature_id, _, _, _), value in zip(chunk, values, strict=True):
            result[feature_id] = value
        del batch, latent, pooled
        processed = min(start + batch_size, len(records))
        batch_number = start // batch_size + 1
        if progress_callback is not None and (
            batch_number % 20 == 0 or processed == len(records)
        ):
            progress_callback(processed, len(records))
    require(len(result) == len(records), "Visual feature cardinality mismatch")
    return result


def read_feature_npz(path: Path) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    require(path.is_file(), f"Missing feature NPZ: {path}")
    with np.load(path, allow_pickle=False) as payload:
        require(
            {"ids", "features", "manifest_json"}.issubset(payload.files),
            f"Feature NPZ keys missing: {path}",
        )
        ids = [str(item) for item in payload["ids"].tolist()]
        features = np.asarray(payload["features"], dtype=np.float32)
        manifest = json.loads(str(payload["manifest_json"].item()))
    require(isinstance(manifest, dict), f"Feature manifest is not an object: {path}")
    require(features.ndim == 2, f"Feature matrix is not rank two: {path}")
    require(len(ids) == len(features), f"Feature cardinality mismatch: {path}")
    require(len(ids) == len(set(ids)), f"Duplicate feature IDs: {path}")
    require(np.all(np.isfinite(features)), f"Nonfinite feature values: {path}")
    return ids, features, manifest


def validate_existing_index(path: Path, spec: P2TrainingSpec) -> dict[str, Any]:
    ids, features, manifest = read_feature_npz(path)
    expected = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "cell_id": spec.cell_id,
        "experiment_id": spec.experiment_id,
        "variant_id": spec.variant_id,
        "method_id": spec.method_id,
        "seed": spec.seed,
        "retrieval_modality": spec.retrieval_modality,
        "H_steps": spec.h_steps,
        "K_steps": spec.k_steps,
        "time_view_id": spec.time_view_id,
        "split_sha256": SPLIT_SHA256,
        "pool_manifest_sha256": POOL_MANIFEST_SHA256,
        "heldout_target_used": False,
        "visual_encoder_used": spec.retrieval_modality in VISUAL_MODALITIES,
    }
    for key, value in expected.items():
        require(manifest.get(key) == value, f"Index {key} drift: {spec.cell_id}")
    require(
        manifest.get("visual_feature_count") == len(ids),
        f"Index visual count drift: {spec.cell_id}",
    )
    if spec.retrieval_modality in VISUAL_MODALITIES:
        require(len(features) > 0, f"Visual index is empty: {spec.cell_id}")
        require(
            manifest.get("visual_encoder_checkpoint_sha256") == TOKENIZER_SHA256,
            f"Tokenizer binding drift: {spec.cell_id}",
        )
    else:
        require(features.shape == (0, 0), f"Nonvisual index carries features: {spec.cell_id}")
    return manifest


def validate_existing_statistics(
    path: Path, spec: P2TrainingSpec, index_sha256: str
) -> dict[str, Any]:
    statistics = read_json(path)
    require(
        statistics.get("schema_version") == STATISTICS_SCHEMA_VERSION,
        f"Statistics schema drift: {spec.cell_id}",
    )
    provenance = statistics.get("provenance", {})
    expected = {
        "cell_id": spec.cell_id,
        "experiment_id": spec.experiment_id,
        "variant_id": spec.variant_id,
        "method_id": spec.method_id,
        "seed": spec.seed,
        "split": "train",
        "heldout_data_used": False,
        "split_sha256": SPLIT_SHA256,
        "retrieval_index_sha256": index_sha256,
        "time_view_id": spec.time_view_id,
        "retrieval_modality": spec.retrieval_modality,
        "target_representation": spec.target_representation,
        "H_steps": spec.h_steps,
        "K_steps": spec.k_steps,
    }
    for key, value in expected.items():
        require(provenance.get(key) == value, f"Statistics {key} drift: {spec.cell_id}")
    return statistics


def visual_cache_path(output_root: Path, key: tuple[int, int, str, int]) -> Path:
    suffix = "__".join(str(value) for value in key)
    return output_root / "_cache" / f"visual_context__{suffix}.npz"


def load_or_build_visual_context(
    output_root: Path,
    key: tuple[int, int, str, int],
    train: Human2RobotP2Dataset,
    heldout: Human2RobotP2Dataset,
    *,
    batch_size: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, np.ndarray], str]:
    path = visual_cache_path(output_root, key)
    if path.is_file():
        ids, features, manifest = read_feature_npz(path)
        expected = {
            "schema_version": VISUAL_CACHE_SCHEMA_VERSION,
            "context_key": list(key),
            "split_sha256": SPLIT_SHA256,
            "pool_manifest_sha256": POOL_MANIFEST_SHA256,
            "tokenizer_checkpoint_sha256": TOKENIZER_SHA256,
            "heldout_target_used": False,
            "visual_feature_count": len(ids),
        }
        for name, value in expected.items():
            require(manifest.get(name) == value, f"Visual cache {name} drift: {path}")
        return dict(zip(ids, features, strict=True)), "reused"
    records = feature_windows(train, heldout)
    visual = encode_visual_features(
        records,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
    ids = sorted(visual)
    features = np.stack([visual[item] for item in ids]).astype(np.float32)
    manifest = {
        "schema_version": VISUAL_CACHE_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "context_key": list(key),
        "split_sha256": SPLIT_SHA256,
        "pool_manifest_sha256": POOL_MANIFEST_SHA256,
        "tokenizer_checkpoint_sha256": TOKENIZER_SHA256,
        "heldout_target_used": False,
        "visual_feature_count": len(ids),
    }
    write_npz_atomic(path, ids, features, manifest)
    return visual, "encoded_and_cached"


def make_index_manifest(
    spec: P2TrainingSpec,
    geometry_mean: np.ndarray,
    geometry_std: np.ndarray,
    geometry_count: int,
    visual_feature_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "cell_id": spec.cell_id,
        "experiment_id": spec.experiment_id,
        "variant_id": spec.variant_id,
        "method_id": spec.method_id,
        "seed": spec.seed,
        "retrieval_modality": spec.retrieval_modality,
        "H_steps": spec.h_steps,
        "K_steps": spec.k_steps,
        "time_view_id": spec.time_view_id,
        "split_sha256": SPLIT_SHA256,
        "pool_manifest_sha256": POOL_MANIFEST_SHA256,
        "heldout_target_used": False,
        "heldout_robot_observation_history_allowed": True,
        "geometry_relative_10d_mean": geometry_mean.tolist(),
        "geometry_relative_10d_std": geometry_std.tolist(),
        "geometry_train_row_count": geometry_count,
        "visual_encoder_used": spec.retrieval_modality in VISUAL_MODALITIES,
        "visual_encoder_checkpoint_path": str(TOKENIZER_PATH)
        if spec.retrieval_modality in VISUAL_MODALITIES
        else None,
        "visual_encoder_checkpoint_sha256": TOKENIZER_SHA256
        if spec.retrieval_modality in VISUAL_MODALITIES
        else None,
        "visual_feature_definition": (
            "blank_causal_warmup_plus_four_identical_preprocessed_anchor_frames__"
            "anchor_latent_index1_spatial_mean__l2"
            if spec.retrieval_modality in VISUAL_MODALITIES
            else None
        ),
        "visual_feature_count": visual_feature_count,
        "tie_break": "sha256_run_seed_query_id_candidate_human_content_sha256",
    }


def _summary(prefix: str, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 10)
    require(len(values) > 0 and np.all(np.isfinite(values)), f"Invalid statistics values: {prefix}")
    return {
        f"{prefix}_count": int(len(values)),
        f"{prefix}_mean": values.mean(axis=0).tolist(),
        f"{prefix}_std": values.std(axis=0).tolist(),
        f"{prefix}_min": values.min(axis=0).tolist(),
        f"{prefix}_max": values.max(axis=0).tolist(),
    }


def compute_training_statistics(
    dataset: Human2RobotP2Dataset, spec: P2TrainingSpec, index_sha256: str
) -> dict[str, Any]:
    cache: dict[tuple[str, str], np.ndarray] = {}
    pools: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    transitions: list[np.ndarray] = []
    query_weight_sums: dict[int, float] = {}
    for example in dataset.examples:
        query = dataset.queries[example.query_index]
        robot = _state_cache_get(cache, dataset, query, "robot")
        current = robot[query.current_row]
        target = robot[query.future_rows]
        if example.candidate_index is None:
            raw_plan = np.repeat(current[None], dataset.h_steps, axis=0)
        else:
            candidate = dataset.candidates[example.candidate_index]
            human = _state_cache_get(cache, dataset, candidate, "human")
            raw_plan = human[candidate.future_rows]
        aligned = align_pool_chunk(raw_plan, current)
        future_transition = target - np.concatenate((current[None], target[:-1]), axis=0)
        pools.append(aligned)
        targets.append(target)
        residuals.append(target - aligned[: dataset.k_steps])
        transitions.append(future_transition)
        query_weight_sums[example.query_index] = query_weight_sums.get(example.query_index, 0.0) + (
            1.0 / example.effective_k
        )
    require(
        all(abs(value - 1.0) <= 1e-12 for value in query_weight_sums.values()),
        "Training query weights do not sum to one",
    )
    residual_array = np.concatenate(residuals, axis=0)
    payload: dict[str, Any] = {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "provenance": {
            "cell_id": spec.cell_id,
            "experiment_id": spec.experiment_id,
            "variant_id": spec.variant_id,
            "method_id": spec.method_id,
            "seed": spec.seed,
            "split": "train",
            "heldout_data_used": False,
            "split_sha256": SPLIT_SHA256,
            "retrieval_index_sha256": index_sha256,
            "time_view_id": spec.time_view_id,
            "retrieval_modality": spec.retrieval_modality,
            "target_representation": spec.target_representation,
            "H_steps": spec.h_steps,
            "K_steps": spec.k_steps,
            "algorithm": "exact_ranked_training_examples_train_only_minmax_v1",
            "query_weight_rule": "one_over_effective_k_sum_one",
        },
        "original_query_count": len(dataset.queries),
        "ranked_example_count": len(dataset.examples),
        "residual_norm_p99": float(np.quantile(np.linalg.norm(residual_array, axis=1), 0.99)),
    }
    payload.update(_summary("pool_action_10d", np.concatenate(pools, axis=0)))
    payload.update(_summary("query_bc_target_10d", np.concatenate(targets, axis=0)))
    payload.update(_summary("residual_10d", residual_array))
    payload.update(_summary("future_state_transition_10d", np.concatenate(transitions, axis=0)))
    return payload


def learned_registry_cell_ids(registry: dict[str, Any]) -> set[str]:
    return {
        str(cell["cell_id"])
        for cell in registry.get("cells", [])
        if cell.get("artifact_kind") == "learned_training_checkpoint"
    }


def materialize(workspace: Path, visual_batch_size: int) -> dict[str, Any]:
    require(Path("/.dockerenv").is_file(), "P2 prepared artifacts must be built in Docker")
    paths = validate_frozen_inputs(workspace)
    output_root = paths["output_root"]
    staging_root = output_root / "_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    specs = p2_training_specs()
    total_cells = len(specs)
    run_id = f"{utc_now()}__pid{os.getpid()}"
    registry_ids = learned_registry_cell_ids(read_json(paths["registry_path"]))
    require(registry_ids == {spec.cell_id for spec in specs}, "Spec/registry learned cells differ")
    entries_by_id: dict[str, dict[str, Any]] = {}
    record_progress(
        output_root,
        run_id=run_id,
        phase="resume_scan",
        completed_cells=0,
        total_cells=total_cells,
        detail="validating atomic cell receipts",
    )
    for spec in specs:
        entry = load_completed_cell_receipt(workspace, output_root, spec)
        if entry is not None:
            entries_by_id[spec.cell_id] = entry
            record_progress(
                output_root,
                run_id=run_id,
                phase="cell_skipped",
                completed_cells=len(entries_by_id),
                total_cells=total_cells,
                cell_id=spec.cell_id,
                detail="receipt and artifact hashes valid",
            )

    contexts: dict[
        tuple[int, int, str, int], tuple[Human2RobotP2Dataset, Human2RobotP2Dataset]
    ] = {}
    geometry_by_context: dict[tuple[int, int, str, int], tuple[np.ndarray, np.ndarray, int]] = {}
    visual_by_context: dict[tuple[int, int, str, int], dict[str, np.ndarray]] = {}

    def get_context(
        spec: P2TrainingSpec,
    ) -> tuple[Human2RobotP2Dataset, Human2RobotP2Dataset]:
        key = context_key(spec)
        if key not in contexts:
            record_progress(
                output_root,
                run_id=run_id,
                phase="context_build",
                completed_cells=len(entries_by_id),
                total_cells=total_cells,
                cell_id=spec.cell_id,
                detail="building canonical train/heldout windows",
            )
            contexts[key] = build_window_context(paths, spec, staging_root)
        return contexts[key]

    for spec in specs:
        if spec.cell_id in entries_by_id:
            continue
        record_progress(
            output_root,
            run_id=run_id,
            phase="cell_start",
            completed_cells=len(entries_by_id),
            total_cells=total_cells,
            cell_id=spec.cell_id,
            detail="resuming or materializing cell artifacts",
        )
        artifact_paths = cell_artifact_paths(output_root, spec)
        index_path = artifact_paths["index"]
        statistics_path = artifact_paths["statistics"]
        if index_path.is_file():
            validate_existing_index(index_path, spec)
            record_progress(
                output_root,
                run_id=run_id,
                phase="index_reused",
                completed_cells=len(entries_by_id),
                total_cells=total_cells,
                cell_id=spec.cell_id,
                detail="existing index contract valid",
            )
        else:
            key = context_key(spec)
            train_context, heldout_context = get_context(spec)
            if key not in geometry_by_context:
                geometry_by_context[key] = geometry_statistics(train_context)
            geometry_mean, geometry_std, geometry_count = geometry_by_context[key]
            visual: dict[str, np.ndarray] = {}
            if spec.retrieval_modality in VISUAL_MODALITIES:
                if key not in visual_by_context:
                    record_progress(
                        output_root,
                        run_id=run_id,
                        phase="visual_context_start",
                        completed_cells=len(entries_by_id),
                        total_cells=total_cells,
                        cell_id=spec.cell_id,
                        detail=f"context={key}",
                    )
                    visual_by_context[key], cache_status = load_or_build_visual_context(
                        output_root,
                        key,
                        train_context,
                        heldout_context,
                        batch_size=visual_batch_size,
                        progress_callback=lambda processed, total, current_spec=spec, current_key=key: record_progress(
                            output_root,
                            run_id=run_id,
                            phase="visual_encode",
                            completed_cells=len(entries_by_id),
                            total_cells=total_cells,
                            cell_id=current_spec.cell_id,
                            detail=f"context={current_key} features={processed}/{total}",
                        ),
                    )
                    record_progress(
                        output_root,
                        run_id=run_id,
                        phase="visual_context_complete",
                        completed_cells=len(entries_by_id),
                        total_cells=total_cells,
                        cell_id=spec.cell_id,
                        detail=f"context={key} status={cache_status}",
                    )
                visual = visual_by_context[key]
            ids = sorted(visual)
            features = (
                np.stack([visual[item] for item in ids]).astype(np.float32)
                if ids
                else np.empty((0, 0), dtype=np.float32)
            )
            index_manifest = make_index_manifest(
                spec, geometry_mean, geometry_std, geometry_count, len(ids)
            )
            write_npz_atomic(index_path, ids, features, index_manifest)
            validate_existing_index(index_path, spec)
            record_progress(
                output_root,
                run_id=run_id,
                phase="index_written",
                completed_cells=len(entries_by_id),
                total_cells=total_cells,
                cell_id=spec.cell_id,
                detail=f"features={len(ids)}",
            )
        index_sha = file_sha256(index_path)

        if statistics_path.is_file():
            validate_existing_statistics(statistics_path, spec, index_sha)
            record_progress(
                output_root,
                run_id=run_id,
                phase="statistics_reused",
                completed_cells=len(entries_by_id),
                total_cells=total_cells,
                cell_id=spec.cell_id,
                detail="existing train-only statistics contract valid",
            )
        else:
            bootstrap_stats = staging_root / f"statistics__{spec.cell_id}.json"
            write_json_atomic(bootstrap_stats, placeholder_statistics())
            ranked_dataset = Human2RobotP2Dataset(
                **dataset_kwargs(
                    paths,
                    spec,
                    split="train",
                    statistics_path=bootstrap_stats,
                    index_path=index_path,
                )
            )
            statistics = compute_training_statistics(ranked_dataset, spec, index_sha)
            write_json_atomic(statistics_path, statistics)
            validate_existing_statistics(statistics_path, spec, index_sha)
            record_progress(
                output_root,
                run_id=run_id,
                phase="statistics_written",
                completed_cells=len(entries_by_id),
                total_cells=total_cells,
                cell_id=spec.cell_id,
                detail=f"ranked_examples={len(ranked_dataset.examples)}",
            )

        exact_train = Human2RobotP2Dataset(
            **dataset_kwargs(
                paths,
                spec,
                split="train",
                statistics_path=statistics_path,
                index_path=index_path,
                use_image_aug=True,
            )
        )
        exact_heldout = Human2RobotP2Dataset(
            **dataset_kwargs(
                paths,
                spec,
                split="heldout",
                statistics_path=statistics_path,
                index_path=index_path,
            )
        )
        entry = {
            "cell_id": spec.cell_id,
            "spec": asdict(spec),
            "config_name": spec.config_name,
            "statistics_path": str(statistics_path.relative_to(workspace)),
            "statistics_sha256": file_sha256(statistics_path),
            "retrieval_index_path": str(index_path.relative_to(workspace)),
            "retrieval_index_sha256": index_sha,
            "train_contract": exact_train.contract_manifest(),
            "heldout_contract": exact_heldout.contract_manifest(),
        }
        write_completed_cell_receipt(workspace, output_root, spec, entry)
        validated_entry = load_completed_cell_receipt(workspace, output_root, spec)
        require(validated_entry is not None, f"Cell receipt was not committed: {spec.cell_id}")
        entries_by_id[spec.cell_id] = validated_entry
        record_progress(
            output_root,
            run_id=run_id,
            phase="cell_complete",
            completed_cells=len(entries_by_id),
            total_cells=total_cells,
            cell_id=spec.cell_id,
            detail="receipt committed atomically",
        )
    entries = [entries_by_id[spec.cell_id] for spec in specs]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "formal_result": False,
        "claim_boundary": "Prepared artifacts are launch inputs, not experiment results.",
        "created_at_utc": utc_now(),
        "protocol_file_sha256": PROTOCOL_SHA256,
        "supplement_file_sha256": SUPPLEMENT_SHA256,
        "registry_file_sha256": REGISTRY_SHA256,
        "split_sha256": SPLIT_SHA256,
        "pool_manifest_sha256": POOL_MANIFEST_SHA256,
        "tokenizer_checkpoint_sha256": TOKENIZER_SHA256,
        "learned_cell_count": len(entries),
        "heldout_target_retrieval_feature_count": 0,
        "entries": entries,
    }
    write_json_atomic(output_root / "prepared_manifest.json", manifest)
    record_progress(
        output_root,
        run_id=run_id,
        phase="complete",
        completed_cells=len(entries),
        total_cells=total_cells,
        detail="prepared_manifest.json committed",
    )
    return manifest


def verify(workspace: Path) -> dict[str, Any]:
    paths = validate_frozen_inputs(workspace)
    manifest_path = paths["output_root"] / "prepared_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == SCHEMA_VERSION, "Prepared manifest schema drift")
    require(manifest.get("status") == "complete", "Prepared manifest is incomplete")
    require(manifest.get("formal_result") is False, "Prepared inputs cannot be formal results")
    require(manifest.get("learned_cell_count") == 48, "Prepared cell count is not 48")
    require(manifest.get("heldout_target_retrieval_feature_count") == 0, "Retrieval leakage")
    entries = manifest.get("entries", [])
    require(len(entries) == 48, "Prepared entry count is not 48")
    expected_ids = {spec.cell_id for spec in p2_training_specs()}
    require({entry.get("cell_id") for entry in entries} == expected_ids, "Prepared cell IDs drift")
    for entry in entries:
        statistics_path = workspace / entry["statistics_path"]
        index_path = workspace / entry["retrieval_index_path"]
        require(file_sha256(statistics_path) == entry["statistics_sha256"], "Statistics hash drift")
        require(file_sha256(index_path) == entry["retrieval_index_sha256"], "Index hash drift")
        statistics = read_json(statistics_path)
        require(statistics.get("provenance", {}).get("heldout_data_used") is False, "Statistics leakage")
        with np.load(index_path, allow_pickle=False) as payload:
            index_manifest = json.loads(str(payload["manifest_json"].item()))
            require(index_manifest.get("heldout_target_used") is False, "Index leakage")
            require(len(payload["ids"]) == len(payload["features"]), "Index feature mismatch")
    return {
        "status": "passed",
        "formal_result": False,
        "prepared_manifest_path": str(manifest_path),
        "prepared_manifest_sha256": file_sha256(manifest_path),
        "learned_cell_count": 48,
        "heldout_target_retrieval_feature_count": 0,
    }


def status(workspace: Path) -> dict[str, Any]:
    paths = validate_frozen_inputs(workspace)
    output_root = paths["output_root"]
    specs = p2_training_specs()
    valid_indices = 0
    valid_statistics = 0
    valid_receipts = 0
    next_cell: dict[str, Any] | None = None
    for spec in specs:
        artifact_paths = cell_artifact_paths(output_root, spec)
        index_sha: str | None = None
        if artifact_paths["index"].is_file():
            validate_existing_index(artifact_paths["index"], spec)
            index_sha = file_sha256(artifact_paths["index"])
            valid_indices += 1
        if artifact_paths["statistics"].is_file():
            require(index_sha is not None, f"Statistics exist without index: {spec.cell_id}")
            validate_existing_statistics(artifact_paths["statistics"], spec, index_sha)
            valid_statistics += 1
        receipt = load_completed_cell_receipt(workspace, output_root, spec)
        if receipt is not None:
            valid_receipts += 1
        elif next_cell is None:
            if artifact_paths["statistics"].is_file():
                phase = "contract_receipt_pending"
            elif artifact_paths["index"].is_file():
                phase = "statistics_pending"
            else:
                phase = "index_pending"
            next_cell = {"cell_id": spec.cell_id, "phase": phase}
    progress_path = output_root / "materialize_progress.json"
    return {
        "status": "complete" if valid_receipts == len(specs) else "resumable_incomplete",
        "formal_result": False,
        "valid_index_count": valid_indices,
        "valid_statistics_count": valid_statistics,
        "valid_receipt_count": valid_receipts,
        "expected_cell_count": len(specs),
        "next_cell": next_cell,
        "progress": read_json(progress_path) if progress_path.is_file() else None,
        "materialize_log_path": str(output_root / "materialize.log"),
    }


def plan(workspace: Path) -> dict[str, Any]:
    paths = validate_frozen_inputs(workspace)
    specs = p2_training_specs()
    return {
        "status": "planned_not_executed",
        "formal_result": False,
        "output_root": str(paths["output_root"]),
        "learned_cell_count": len(specs),
        "visual_cell_count": sum(spec.retrieval_modality in VISUAL_MODALITIES for spec in specs),
        "context_count": len({context_key(spec) for spec in specs}),
        "requires_frozen_wan_encoder": True,
        "downloads_allowed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "status", "materialize", "verify"))
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--visual-batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "plan":
        result = plan(args.workspace)
    elif args.command == "status":
        result = status(args.workspace)
    elif args.command == "materialize":
        result = materialize(args.workspace, args.visual_batch_size)
    else:
        result = verify(args.workspace)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
