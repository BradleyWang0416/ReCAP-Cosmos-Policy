#!/usr/bin/env python3
"""Freeze and materialize the approved M5B-P2 successor contract.

This module never launches training or evaluation.  It preserves all v1 files,
creates the v2 semantic chain, reuses the 45 unaffected prepared entries, and
leaves the three offset=5 entries for the ordinary resumable materializer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


PROTOCOL_PATH = Path("方案/v03/M5B_formal_acceptance_protocol_v1.json")
V1_SUPPLEMENT_PATH = Path("方案/v03/M5B_P2_execution_supplement_v1.json")
V1_REGISTRY_PATH = Path("方案/v03/M5B_P2_cell_registry_v1.json")
V2_SUPPLEMENT_PATH = Path("方案/v03/M5B_P2_execution_supplement_v2.json")
V2_SUPPLEMENT_LOCK_PATH = Path("方案/v03/M5B_P2_execution_supplement_v2.lock.json")
V2_REGISTRY_PATH = Path("方案/v03/M5B_P2_cell_registry_v2.json")
V2_REGISTRY_LOCK_PATH = Path("方案/v03/M5B_P2_cell_registry_v2.lock.json")
WORKSPACE_BOUNDS_PATH = Path("方案/v03/M5B_P2_workspace_bounds_v1.json")
WORKSPACE_BOUNDS_LOCK_PATH = Path("方案/v03/M5B_P2_workspace_bounds_v1.lock.json")
LAUNCH_SCHEMA_PATH = Path("方案/v03/M5B_P2_launch_activation_schema_v2.json")
LAUNCH_SCHEMA_LOCK_PATH = Path("方案/v03/M5B_P2_launch_activation_schema_v2.lock.json")
FINAL_SCHEMA_PATH = Path("方案/v03/M5B_P2_final_acceptance_schema_v2.json")
FINAL_SCHEMA_LOCK_PATH = Path("方案/v03/M5B_P2_final_acceptance_schema_v2.lock.json")
PREPARED_V1_PATH = Path("data/Human2Robot/derived/m5b_v03/p2_prepared/prepared_manifest.json")
PREPARED_V2_PATH = Path("data/Human2Robot/derived/m5b_v03/p2_prepared_v2/prepared_manifest.json")
CANONICAL_ROOT = Path("data/Human2Robot/canonical/v3")
LAG_VIEW_RELATIVE_PATH = Path(
    "data/Human2Robot/derived/views/nominal_camera_30hz_segmented/"
    "human_hand_robot_frame_raw/robot_ee_observed_t_plus_5_lag_diagnostic/"
    "train_only_tplus5_query_anchor_se3_identity_scale_v1"
)
TERMINAL_CELL_ID = "aggregate_report__M5B-QUAL-01__full_matrix_completion_acceptance"
V1_REGISTRY_SHA256 = "4664d036bcf6bc41e8a44fac2afe04ff6de62c2a180a29d3433bd83e46604df5"
PROTOCOL_SHA256 = "7598dfa2ac2e129f5d21a295dad23b90f63c3c8e68811da73cbcc20eb95d5ce4"
GENERATOR_SHA256 = "8765d24606db00a8b875195c760092f2a1f7b4c28dda8db6564ad52b1ca6c0bd"
MATERIALIZER_SHA256 = "ac15c5b748e06771fee9b7247672c03c0b34ded5110c5c686bc55e11183ab313"
PROPOSAL_SHA256 = "edf692ea17242458e0e133d1dcc25685d4b02e7964845d2c2ee8fbb2a66ad733"
FORMAL_SEEDS = (20260711, 20260712, 20260713)
COUNTS = {
    "learned_training_checkpoint": 48,
    "nonlearned_method_artifact": 3,
    "checkpoint_linked_evaluation": 147,
    "aggregate_report": 5,
}


class SuccessorContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SuccessorContractError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _lock(path: Path, *, schema_version: str, status: str = "locked") -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "status": status,
        "artifact_path": path.as_posix(),
        "artifact_file_sha256": file_sha256(path),
        "contains_experiment_results": False,
        "passes_p2": False,
    }


def _contiguous_segments(segment_id: np.ndarray, gap_mask: np.ndarray) -> list[np.ndarray]:
    require(segment_id.ndim == gap_mask.ndim == 1 and len(segment_id) == len(gap_mask), "Bad segment metadata")
    boundaries = [0]
    for index in range(1, len(segment_id)):
        if bool(gap_mask[index]) or int(segment_id[index]) != int(segment_id[index - 1]):
            boundaries.append(index)
    boundaries.append(len(segment_id))
    return [np.arange(left, right, dtype=np.int64) for left, right in zip(boundaries[:-1], boundaries[1:]) if right > left]


def materialize_lag_view(workspace: Path) -> dict[str, Any]:
    source_path = workspace / (
        "data/Human2Robot/derived/views/nominal_camera_30hz_segmented/"
        "human_hand_robot_frame_raw/robot_ee_observed_t_plus_1_bc_proxy/"
        "train_only_tplus1_query_anchor_se3_identity_scale_v1/view_manifest.json"
    )
    source = read_json(source_path)
    preprocessing_path = workspace / CANONICAL_ROOT / "preprocessing_manifest.json"
    preprocessing = read_json(preprocessing_path)
    counts = {"train": 0, "heldout": 0}
    segment_count = 0
    for episode in preprocessing.get("episodes", []):
        split = str(episode["task_split"])
        path = workspace / str(episode["output_path"])
        with h5py.File(path, "r") as handle:
            demo = handle["data/demo_0"]
            segment_id = np.asarray(demo["metadata/segment_id"][:], dtype=np.int64)
            gap_mask = np.asarray(demo["metadata/gap_mask"][:], dtype=bool)
        for rows in _contiguous_segments(segment_id, gap_mask):
            segment_count += 1
            max_current = len(rows) - 5 - 8
            counts[split] += len(range(7, max_current + 1, 8)) if max_current >= 7 else 0
    manifest = dict(source)
    manifest.update(
        {
            "schema_version": "human2robot-m5b-p2-lag-derived-view-v1",
            "created_at_utc": utc_now(),
            "view_id": canonical_json_sha256(
                {
                    "parent_view_sha256": file_sha256(source_path),
                    "query_target_offset_view_steps": 5,
                    "H_steps": 8,
                    "K_steps": 8,
                    "window_stride": 8,
                }
            ),
            "parent_tplus1_view_path": str(source_path.relative_to(workspace)),
            "parent_tplus1_view_sha256": file_sha256(source_path),
            "query_action_view_id": "robot_ee_observed_t_plus_5_lag_diagnostic",
            "query_action_role": "diagnostic_strict_future_offset5_bc_proxy",
            "action_alignment_id": "train_only_tplus5_query_anchor_se3_identity_scale_v1",
            "query_target_offset_view_steps": 5,
            "strict_future_target": True,
            "gap_policy": "never_cross_segment",
            "terminal_policy": "drop_incomplete_future_chunks",
            "materialization": {
                "kind": "canonical_row_index_view",
                "query_offset_view_steps": 5,
                "H_steps": 8,
                "K_steps": 8,
                "window_stride": 8,
                "window_count_by_split": counts,
                "segment_count": segment_count,
                "gap_crossing_count": 0,
            },
            "claim_boundary": "diagnostic offset view; not a deployment command adapter",
        }
    )
    output = workspace / LAG_VIEW_RELATIVE_PATH / "view_manifest.json"
    write_json_atomic(output, manifest)
    lock = _lock(
        output,
        schema_version="human2robot-m5b-p2-lag-derived-view-lock-v1",
    )
    lock["query_target_offset_view_steps"] = 5
    lock["gap_crossing_count"] = 0
    write_json_atomic(output.with_name("view_manifest.lock.json"), lock)
    return {"path": str(output.relative_to(workspace)), "sha256": file_sha256(output), "window_count_by_split": counts}


def materialize_workspace_bounds(workspace: Path) -> dict[str, Any]:
    canonical = workspace / CANONICAL_ROOT
    split_path = canonical / "task_split_manifest.json"
    preprocessing_path = canonical / "preprocessing_manifest.json"
    split = read_json(split_path)
    preprocessing = read_json(preprocessing_path)
    train_episodes = [item for item in preprocessing.get("episodes", []) if item.get("task_split") == "train"]
    require(len(train_episodes) == 16, "Expected 16 train episodes")
    xyz_parts = []
    sources = []
    for episode in train_episodes:
        relative = Path(str(episode["output_path"]))
        with h5py.File(workspace / relative, "r") as handle:
            xyz = np.asarray(handle["data/demo_0/trajectories/robot_ee_observed_10d"][:, :3], dtype=np.float64)
        require(xyz.ndim == 2 and xyz.shape[1] == 3 and np.isfinite(xyz).all(), f"Invalid xyz: {relative}")
        xyz_parts.append(xyz)
        sources.append(
            {
                "canonical_path": relative.as_posix(),
                "source_relative_path": episode["source_relative_path"],
                "source_sha256": episode["source_sha256"],
                "row_count": len(xyz),
            }
        )
    values = np.concatenate(xyz_parts, axis=0)
    raw_min = values.min(axis=0)
    raw_max = values.max(axis=0)
    span = raw_max - raw_min
    margin = 0.05 * span
    payload = {
        "schema_version": "human2robot-m5b-p2-workspace-bounds-v1",
        "status": "frozen",
        "created_at_utc": utc_now(),
        "coordinate_system": "canonical robot_ee_observed_10d xyz",
        "coordinate_units": "canonical units; physical unit not asserted",
        "scope": "M5 offline prediction guardrail only; not M6 robot safety workspace",
        "source_split": "train_only",
        "split_manifest_path": str(split_path.relative_to(workspace)),
        "split_manifest_file_sha256": file_sha256(split_path),
        "split_sha256": split["split_sha256"],
        "preprocessing_manifest_path": str(preprocessing_path.relative_to(workspace)),
        "preprocessing_manifest_file_sha256": file_sha256(preprocessing_path),
        "source_episodes": sources,
        "source_episode_count": len(sources),
        "source_row_count": int(len(values)),
        "formula": "axiswise [train_min - 0.05*(train_max-train_min), train_max + 0.05*(train_max-train_min)]",
        "raw_train_xyz_min": raw_min.tolist(),
        "raw_train_xyz_max": raw_max.tolist(),
        "margin_fraction_of_axis_range": 0.05,
        "margin_xyz": margin.tolist(),
        "xyz_min": (raw_min - margin).tolist(),
        "xyz_max": (raw_max + margin).tolist(),
        "inference_policy": "do_not_clip; count every out-of-envelope predicted step; acceptance requires zero",
        "heldout_data_used": False,
    }
    output = workspace / WORKSPACE_BOUNDS_PATH
    write_json_atomic(output, payload)
    lock = _lock(output, schema_version="human2robot-m5b-p2-workspace-bounds-lock-v1")
    lock["split_sha256"] = split["split_sha256"]
    lock["heldout_data_used"] = False
    write_json_atomic(workspace / WORKSPACE_BOUNDS_LOCK_PATH, lock)
    return {"path": WORKSPACE_BOUNDS_PATH.as_posix(), "sha256": file_sha256(output), "xyz_min": payload["xyz_min"], "xyz_max": payload["xyz_max"]}


def freeze_core(workspace: Path) -> dict[str, Any]:
    require(file_sha256(workspace / PROTOCOL_PATH) == PROTOCOL_SHA256, "Protocol drift")
    require(file_sha256(workspace / V1_REGISTRY_PATH) == V1_REGISTRY_SHA256, "v1 registry drift")
    lag = materialize_lag_view(workspace)
    bounds = materialize_workspace_bounds(workspace)
    v1_supplement = read_json(workspace / V1_SUPPLEMENT_PATH)
    correction_path = workspace / "方案/v03/M5B_P2_execution_correction_v2.proposed.json"
    correction_sha = file_sha256(correction_path)
    supplement = dict(v1_supplement)
    supplement.update(
        {
            "schema_version": "human2robot-m5b-p2-execution-supplement-v2",
            "supplement_id": "m5b_p2_claim_centered_execution_v2",
            "status": "frozen_approved_execution_spec",
            "frozen_at_utc": utc_now(),
            "supersedes": {
                "path": V1_SUPPLEMENT_PATH.as_posix(),
                "file_sha256": file_sha256(workspace / V1_SUPPLEMENT_PATH),
                "mutation_allowed": False,
            },
            "approved_successor_decision": {
                "path": str(correction_path.relative_to(workspace)),
                "file_sha256": correction_sha,
                "approval_basis": "user directive: 按七项推荐冻结并实施",
                "approved_at_date": "2026-07-14",
            },
            "decision_freeze": {
                "1_sampler": {
                    "scheduler": "native_rectified_flow_scheduler",
                    "guidance": 1.5,
                    "num_steps": 35,
                    "ode_steps": 34,
                    "final_clean_x0_prediction": True,
                    "shift": 5.0,
                    "use_kerras_sigma_at_inference": True,
                    "variance_scale": False,
                    "legacy_2ab_forbidden": True,
                },
                "2_workspace": {**bounds, "clipping_allowed": False, "m6_safety_workspace": False},
                "3_completion": {
                    "terminal_cell_id": TERMINAL_CELL_ID,
                    "terminal_parent_count": 202,
                    "successor_cell_count": 203,
                },
                "4_lag_view": {**lag, "query_offset_view_steps": 5},
                "5_temporal": {
                    "materialization_point": "uint8 model video input before tokenizer.encode",
                    "mild_primary": ["frame_drop_5pct", "timestamp_jitter_5ms"],
                    "severe_pre_model_reject": ["frame_drop_20pct", "timestamp_jitter_20ms", "pause_1p0s", "step_jump_20"],
                    "intermediate_status": "diagnostic",
                    "masked_severe_allowed": False,
                },
                "6_resolution": {
                    "ranking": "same frozen WAN encoder, same candidates, same seeded ties, per-query top-k",
                    "mean_jaccard_threshold": 0.90,
                    "report_median_min_identical_ratio": True,
                    "primary_metric_relative_degradation_max": 0.05,
                },
                "7_activation": {
                    "launch_schema": LAUNCH_SCHEMA_PATH.as_posix(),
                    "final_acceptance_schema": FINAL_SCHEMA_PATH.as_posix(),
                    "launch_does_not_imply_p2_acceptance": True,
                },
            },
            "frozen_registry_contract": {
                "generator_code_sha256": GENERATOR_SHA256,
                "learned_training_checkpoint_count": 48,
                "nonlearned_method_artifact_count": 3,
                "checkpoint_linked_evaluation_count": 147,
                "aggregate_report_count": 5,
                "total_cell_count": 203,
            },
            "formal_launch_preconditions": [
                "successor hashes and locks validate",
                "full Docker test suite passes",
                "eight GPUs visible",
                "formal output mount is writable",
                "local weight hashes match",
                "source snapshot is materialized",
                "separate launch activation v2 is approved",
            ],
            "current_state": {
                "formal_queue_allowed": False,
                "p2_acceptance_allowed": False,
                "p2_status": "pending",
                "formal_results": "NEEDS_EXPERIMENT",
            },
        }
    )
    supplement_path = workspace / V2_SUPPLEMENT_PATH
    write_json_atomic(supplement_path, supplement)
    supplement_lock = {
        "schema_version": "human2robot-m5b-p2-execution-supplement-lock-v2",
        "status": "locked",
        "supplement_path": V2_SUPPLEMENT_PATH.as_posix(),
        "supplement_file_sha256": file_sha256(supplement_path),
        "parent_protocol_file_sha256": PROTOCOL_SHA256,
        "approved_proposal_file_sha256": PROPOSAL_SHA256,
        "approved_successor_decision_file_sha256": correction_sha,
        "candidate_registry_generator_file_sha256": GENERATOR_SHA256,
        "contains_experiment_results": False,
        "passes_p2": False,
    }
    write_json_atomic(workspace / V2_SUPPLEMENT_LOCK_PATH, supplement_lock)

    v1_registry = read_json(workspace / V1_REGISTRY_PATH)
    v1_cells = list(v1_registry["cells"])
    require(len(v1_cells) == 202, "v1 cell count drift")
    terminal = {
        "cell_id": TERMINAL_CELL_ID,
        "artifact_kind": "aggregate_report",
        "experiment_id": "M5B-QUAL-01",
        "variant_id": "full_matrix_completion_acceptance",
        "method_id": None,
        "seed": None,
        "parent_artifact_ids": [str(cell["cell_id"]) for cell in v1_cells],
        "optimizer_steps": None,
        "formal_result": False,
        "status": "pending",
    }
    cells = [*v1_cells, terminal]
    registry = dict(v1_registry)
    registry.update(
        {
            "schema_version": "human2robot-m5b-p2-cell-registry-v2",
            "registry_id": "m5b_p2_claim_centered_203_cells_v2",
            "status": "frozen_pending_execution",
            "formal_queue_allowed": False,
            "p2_acceptance_allowed": False,
            "supplement_id": supplement["supplement_id"],
            "supplement_path": V2_SUPPLEMENT_PATH.as_posix(),
            "supplement_file_sha256": file_sha256(supplement_path),
            "supplement_lock_path": V2_SUPPLEMENT_LOCK_PATH.as_posix(),
            "parent_v1_registry_path": V1_REGISTRY_PATH.as_posix(),
            "parent_v1_registry_file_sha256": V1_REGISTRY_SHA256,
            "counts": COUNTS,
            "cell_count": 203,
            "cells_payload_sha256": canonical_json_sha256(cells),
            "cells": cells,
            "current_blocker": "A separate launch activation v2 is required; formal results remain NEEDS_EXPERIMENT.",
        }
    )
    registry_path = workspace / V2_REGISTRY_PATH
    write_json_atomic(registry_path, registry)
    registry_lock = {
        "schema_version": "human2robot-m5b-p2-cell-registry-lock-v2",
        "status": "locked_pending_execution",
        "registry_path": V2_REGISTRY_PATH.as_posix(),
        "registry_file_sha256": file_sha256(registry_path),
        "registry_materializer_file_sha256": MATERIALIZER_SHA256,
        "candidate_registry_generator_file_sha256": GENERATOR_SHA256,
        "execution_supplement_file_sha256": file_sha256(supplement_path),
        "execution_supplement_lock_file_sha256": file_sha256(workspace / V2_SUPPLEMENT_LOCK_PATH),
        "counts": COUNTS,
        "cell_count": 203,
        "formal_queue_allowed": False,
        "contains_experiment_results": False,
        "passes_p2": False,
    }
    write_json_atomic(workspace / V2_REGISTRY_LOCK_PATH, registry_lock)
    return {
        "status": "frozen_pending_execution",
        "supplement_sha256": file_sha256(supplement_path),
        "supplement_lock_sha256": file_sha256(workspace / V2_SUPPLEMENT_LOCK_PATH),
        "registry_sha256": file_sha256(registry_path),
        "registry_lock_sha256": file_sha256(workspace / V2_REGISTRY_LOCK_PATH),
        "workspace_bounds": bounds,
        "lag_view": lag,
        "cell_count": 203,
        "formal_queue_allowed": False,
    }


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(file_sha256(destination) == file_sha256(source), f"Existing reusable artifact differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def seed_reusable_prepared(workspace: Path) -> dict[str, Any]:
    """Create v2 receipts for 45 semantically unchanged entries; never touch lag=5."""

    require(Path("/.dockerenv").is_file(), "Prepared successor seeding must run in Docker")
    from cosmos_policy.datasets.human2robot_p2_dataset import Human2RobotP2Dataset
    from cosmos_policy.datasets.human2robot_p2_specs import p2_training_specs
    from tools import human2robot_m5b_p2_prepare as prepare

    paths = prepare.validate_frozen_inputs(workspace)
    old = read_json(workspace / PREPARED_V1_PATH)
    old_entries = {str(item["cell_id"]): item for item in old["entries"]}
    reused = []
    skipped_lag = []
    for spec in p2_training_specs():
        if spec.query_offset_view_steps == 5:
            skipped_lag.append(spec.cell_id)
            continue
        destination = prepare.cell_artifact_paths(paths["output_root"], spec)
        old_entry = old_entries[spec.cell_id]
        _link_or_copy(workspace / str(old_entry["retrieval_index_path"]), destination["index"])
        _link_or_copy(workspace / str(old_entry["statistics_path"]), destination["statistics"])
        index_sha = file_sha256(destination["index"])
        prepare.validate_existing_index(destination["index"], spec)
        prepare.validate_existing_statistics(destination["statistics"], spec, index_sha)
        train = Human2RobotP2Dataset(
            **prepare.dataset_kwargs(
                paths,
                spec,
                split="train",
                statistics_path=destination["statistics"],
                index_path=destination["index"],
                use_image_aug=True,
            )
        )
        heldout = Human2RobotP2Dataset(
            **prepare.dataset_kwargs(
                paths,
                spec,
                split="heldout",
                statistics_path=destination["statistics"],
                index_path=destination["index"],
            )
        )
        entry = {
            "cell_id": spec.cell_id,
            "spec": asdict(spec),
            "config_name": spec.config_name,
            "statistics_path": str(destination["statistics"].relative_to(workspace)),
            "statistics_sha256": file_sha256(destination["statistics"]),
            "retrieval_index_path": str(destination["index"].relative_to(workspace)),
            "retrieval_index_sha256": index_sha,
            "train_contract": train.contract_manifest(),
            "heldout_contract": heldout.contract_manifest(),
            "successor_materialization": "hash-identical v1 statistics/index reused; v2 dataset contracts rebuilt",
        }
        prepare.write_completed_cell_receipt(workspace, paths["output_root"], spec, entry)
        require(
            prepare.load_completed_cell_receipt(workspace, paths["output_root"], spec) is not None,
            f"Reusable receipt did not validate: {spec.cell_id}",
        )
        reused.append(spec.cell_id)
    require(len(reused) == 45 and len(skipped_lag) == 3, "Expected 45 reused and three lag entries")
    return {"status": "seeded", "reused_count": len(reused), "lag_pending_count": len(skipped_lag), "lag_pending_cell_ids": skipped_lag}


def freeze_activation_schemas(workspace: Path) -> dict[str, Any]:
    prepared_path = workspace / PREPARED_V2_PATH
    require(prepared_path.is_file(), "Prepared v2 manifest is incomplete")
    exact = {
        "registry_sha256": file_sha256(workspace / V2_REGISTRY_PATH),
        "supplement_sha256": file_sha256(workspace / V2_SUPPLEMENT_PATH),
        "prepared_manifest_sha256": file_sha256(prepared_path),
        "workspace_bounds_sha256": file_sha256(workspace / WORKSPACE_BOUNDS_PATH),
        "lag_view_manifest_sha256": file_sha256(workspace / LAG_VIEW_RELATIVE_PATH / "view_manifest.json"),
    }
    launch = {
        "schema_version": "human2robot-m5b-p2-launch-activation-schema-v2",
        "status": "frozen_schema_not_activated",
        "artifact_schema_version": "human2robot-m5b-p2-launch-activation-v2",
        "required_exact_values": {
            "status": "approved",
            "launch_authorized": True,
            "formal_queue_allowed": True,
            "p2_acceptance_allowed": False,
            **exact,
            "native_rectified_flow_contract_resolved": True,
            "all_147_evaluations_bound_to_terminal_report": True,
            "docker_full_suite_passed": True,
            "source_snapshot_frozen": True,
            "gpu_count": 8,
            "storage_probe_passed": True,
            "formal_output_mount_writable": True,
            "local_weight_hashes_passed": True,
        },
        "claim_boundary": "This schema does not activate the queue. An independent artifact at the formal output root is required.",
    }
    write_json_atomic(workspace / LAUNCH_SCHEMA_PATH, launch)
    write_json_atomic(
        workspace / LAUNCH_SCHEMA_LOCK_PATH,
        _lock(workspace / LAUNCH_SCHEMA_PATH, schema_version="human2robot-m5b-p2-launch-activation-schema-lock-v2"),
    )
    final = {
        "schema_version": "human2robot-m5b-p2-final-acceptance-schema-v2",
        "status": "frozen_schema_not_accepted",
        "artifact_schema_version": "human2robot-m5b-p2-final-acceptance-v2",
        "required_exact_values": {
            "status": "passed",
            "formal_queue_allowed": True,
            "p2_acceptance_allowed": True,
            "terminal_cell_id": TERMINAL_CELL_ID,
            "terminal_report_status": "passed",
            "completed_cell_count": 203,
            **exact,
        },
        "claim_boundary": "May be issued only after the terminal report artifact passes; launch activation alone cannot pass P2.",
    }
    write_json_atomic(workspace / FINAL_SCHEMA_PATH, final)
    write_json_atomic(
        workspace / FINAL_SCHEMA_LOCK_PATH,
        _lock(workspace / FINAL_SCHEMA_PATH, schema_version="human2robot-m5b-p2-final-acceptance-schema-lock-v2"),
    )
    return {
        "status": "schemas_frozen_not_activated",
        "launch_schema_sha256": file_sha256(workspace / LAUNCH_SCHEMA_PATH),
        "final_schema_sha256": file_sha256(workspace / FINAL_SCHEMA_PATH),
        **exact,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-core", "seed-prepared", "freeze-activation-schemas"))
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.resolve()
    if args.command == "freeze-core":
        result = freeze_core(workspace)
    elif args.command == "seed-prepared":
        result = seed_reusable_prepared(workspace)
    else:
        result = freeze_activation_schemas(workspace)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SuccessorContractError as error:
        print(f"M5B-P2 successor error: {error}", file=__import__("sys").stderr)
        raise SystemExit(2) from error
