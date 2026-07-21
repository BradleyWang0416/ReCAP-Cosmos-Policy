#!/usr/bin/env python3
"""Stage-1 data/source identity protocol for Human2Robot v04.

This module is imported only by the audited v04 entry point.  Formal execution
is Docker-only and offline.  Raw Human2Robot HDF5 files are always opened read
only; held-out human and robot roles are materialized into physically separate
allowlisted projections under ``data/Human2Robot/derived/v04``.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import h5py
import numpy as np


SCHEMA = "human2robot-v04-stage1-data-v2"
SEED = 20260711
H_STEPS = 8
K_STEPS = 8
SEEN_CANDIDATE_COUNT = 737
HELDOUT_CANDIDATE_COUNT = 258
CONTENT_HASH_DEFINITION = "sha256(schema:role:complete_source_file_sha256)"

SEEN_TASKS = (
    "cloth/cloth12",
    "cloth/cloth13",
    "cloth/cloth21",
    "cloth/cloth31",
    "grab_both_cubes_v1",
    "grab_cup_v1",
    "grab_pencil2_v1",
    "grab_pencil_v1",
    "grab_to_plate1_and_back_v1",
    "grab_to_plate2_and_back_v1",
    "grab_to_plate2_and_pull_v1",
    "grab_to_plate2_v1",
    "grab_two_cubes2_v1",
    "pull_plate_grab_cube",
    "pull_plate_v1",
    "push_box_common_v1",
)
PRE_AMENDMENT_HELDOUT_TASKS = (
    "grab_cube2_v1",
    "grab_pencil1_v1",
    "grab_to_plate1_v1",
    "push_box_random_v1",
)
HELDOUT_TASKS = (
    "grab_cube2_v1",
    "push_plate_v1",
    "grab_to_plate1_v1",
    "push_box_random_v1",
)
LEGACY_QUARANTINE_COUNTS = {
    "grab_cube2_v1": 10,
    "push_plate_v1": 0,
    "grab_to_plate1_v1": 10,
    "push_box_random_v1": 10,
}
REPLACEMENT_CANDIDATES = (
    "push_plate_v1",
    "push_box_two_v1",
)
PARTITIONS = (
    "seen_train",
    "seen_validation",
    "legacy_quarantine",
    "v04_human_pool",
    "v04_robot_dev",
    "v04_robot_final",
    "reserve",
)

RAW_HUMAN_DATASETS = (
    "cam_data/human_camera",
    "action",
    "transformed_hand_coords",
    "transformed_hand_frames",
    "step",
    "timestamp",
)
RAW_ROBOT_DATASETS = (
    "cam_data/robot_camera",
    "end_position",
    "gripper_state",
    "step",
    "timestamp",
)
HUMAN_PROJECTION_DATASETS = (
    "data/demo_0/human/images",
    "data/demo_0/human/hand_action_7d",
    "data/demo_0/human/hand_coords",
    "data/demo_0/human/hand_frames",
    "data/demo_0/time/gap_mask",
    "data/demo_0/time/legal_window_start",
    "data/demo_0/time/segment_id",
    "data/demo_0/time/source_step",
    "data/demo_0/time/source_timestamp",
)
ROBOT_PROJECTION_DATASETS = (
    "data/demo_0/robot/gripper_state",
    "data/demo_0/robot/images",
    "data/demo_0/robot/observed_eef_pose_6d",
    "data/demo_0/time/gap_mask",
    "data/demo_0/time/legal_window_start",
    "data/demo_0/time/segment_id",
    "data/demo_0/time/source_step",
    "data/demo_0/time/source_timestamp",
)

Progress = Callable[[int, int, str], None]


class Stage1DataError(RuntimeError):
    """A fail-closed stage-1 data or provenance violation."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage1DataError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: Any, *, mode: int | None = None) -> None:
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
        if mode is not None:
            path.chmod(mode)
    finally:
        with contextlib.suppress(FileNotFoundError):
            partial.unlink()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    _require(resolved.is_file(), f"Required file is missing: {resolved}")
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": file_sha256(resolved)}


def source_rank(source_relative_path: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}:{source_relative_path}".encode("utf-8")).hexdigest()


def role_content_sha256(source_sha256: str, role: str) -> str:
    return hashlib.sha256(f"{SCHEMA}:{role}:{source_sha256}".encode("ascii")).hexdigest()


def _dataset(file: h5py.File | h5py.Group, name: str) -> h5py.Dataset:
    value = file.get(name)
    if not isinstance(value, h5py.Dataset):
        raise Stage1DataError(f"Required dataset is missing: {name} in {file.file.filename}")
    return value


def _time_structure(step: np.ndarray, timestamp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require(step.ndim == timestamp.ndim == 1, "step/timestamp must be one-dimensional")
    _require(len(step) == len(timestamp) and len(step) > 0, "step/timestamp length mismatch or empty")
    step64 = np.asarray(step, dtype=np.int64)
    timestamp64 = np.asarray(timestamp, dtype=np.int64)
    step_delta = np.diff(step64)
    timestamp_delta = np.diff(timestamp64)
    gap_mask = np.zeros(len(step64), dtype=np.bool_)
    if len(gap_mask) > 1:
        gap_mask[1:] = (step_delta != 1) | (timestamp_delta < 0) | (timestamp_delta > 1)
    segment_id = np.cumsum(gap_mask, dtype=np.int32)
    starts: list[int] = []
    for segment in np.unique(segment_id):
        rows = np.flatnonzero(segment_id == segment)
        if len(rows) < H_STEPS + K_STEPS:
            continue
        starts.extend(range(int(rows[0]), int(rows[-1]) - H_STEPS - K_STEPS + 2))
    return gap_mask, segment_id, np.asarray(starts, dtype=np.int64)


def _check_shape(dataset: h5py.Dataset, *, ndim: int, tail: Sequence[int] | None = None) -> None:
    _require(dataset.ndim == ndim, f"Bad rank for {dataset.name}: {dataset.shape}")
    if tail is not None:
        _require(tuple(dataset.shape[-len(tail) :]) == tuple(tail), f"Bad shape for {dataset.name}: {dataset.shape}")


def inspect_source(path: Path) -> dict[str, Any]:
    """Validate the full paired source contract without writing to raw data."""

    with h5py.File(path, "r") as source:
        human_image = _dataset(source, "cam_data/human_camera")
        robot_image = _dataset(source, "cam_data/robot_camera")
        action = _dataset(source, "action")
        coords = _dataset(source, "transformed_hand_coords")
        frames = _dataset(source, "transformed_hand_frames")
        eef = _dataset(source, "end_position")
        gripper = _dataset(source, "gripper_state")
        step_ds = _dataset(source, "step")
        timestamp_ds = _dataset(source, "timestamp")
        _check_shape(human_image, ndim=4, tail=(3,))
        _check_shape(robot_image, ndim=4, tail=(3,))
        _require(human_image.dtype in (np.dtype("uint8"), np.dtype("uint16")), f"Bad human image dtype: {path}")
        _require(robot_image.dtype in (np.dtype("uint8"), np.dtype("uint16")), f"Bad robot image dtype: {path}")
        _check_shape(action, ndim=2, tail=(7,))
        _check_shape(coords, ndim=3, tail=(24, 3))
        _check_shape(frames, ndim=3, tail=(4, 3))
        _check_shape(eef, ndim=2, tail=(6,))
        _check_shape(gripper, ndim=1)
        _check_shape(step_ds, ndim=1)
        _check_shape(timestamp_ds, ndim=1)
        lengths = {
            int(item.shape[0])
            for item in (human_image, robot_image, action, coords, frames, eef, gripper, step_ds, timestamp_ds)
        }
        _require(len(lengths) == 1, f"Time axes differ: {path}")
        frame_count = next(iter(lengths))
        _require(frame_count >= H_STEPS + K_STEPS, f"Fewer than H+K={H_STEPS + K_STEPS} frames: {path}")

        numeric = {
            "action": np.asarray(action[:], dtype=np.float64),
            "transformed_hand_coords": np.asarray(coords[:], dtype=np.float64),
            "transformed_hand_frames": np.asarray(frames[:], dtype=np.float64),
            "end_position": np.asarray(eef[:], dtype=np.float64),
            "gripper_state": np.asarray(gripper[:], dtype=np.float64),
        }
        for name, values in numeric.items():
            _require(np.isfinite(values).all(), f"{name} contains NaN/Inf: {path}")
        _require(np.all((numeric["action"][:, 6] >= 0) & (numeric["action"][:, 6] <= 1)), f"Bad human gripper: {path}")
        _require(np.all((numeric["gripper_state"] >= 0) & (numeric["gripper_state"] <= 1)), f"Bad robot gripper: {path}")
        step = np.asarray(step_ds[:], dtype=np.int64)
        timestamp = np.asarray(timestamp_ds[:], dtype=np.int64)
        gap_mask, segment_id, starts = _time_structure(step, timestamp)
        _require(len(starts) > 0, f"No legal gap-safe H={H_STEPS}/K={K_STEPS} window: {path}")
        counts = Counter(segment_id.tolist())
        return {
            "frame_count": frame_count,
            "legal_window_count": int(len(starts)),
            "gap_count": int(np.count_nonzero(gap_mask)),
            "segment_count": len(counts),
            "max_gap_safe_segment_frames": max(counts.values()),
            "human_image_shape": list(human_image.shape[1:]),
            "robot_image_shape": list(robot_image.shape[1:]),
        }


def _task_paths(source_root: Path, task: str) -> list[Path]:
    task_root = source_root / task
    _require(task_root.is_dir(), f"Missing task directory: {task_root}")
    paths = [path for path in task_root.glob("*.hdf5") if path.is_file()]
    _require(paths, f"No HDF5 episodes for task: {task}")
    return sorted(paths, key=lambda path: source_rank(path.relative_to(source_root).as_posix()))


def _legacy_source_inventory(
    workspace: Path,
    tasks: Sequence[str],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    canonical_path = workspace / "data/Human2Robot/canonical/v3/preprocessing_manifest.json"
    pool_path = workspace / "data/Human2Robot/derived/m5b_v03/p1_human_only_pool/selection_manifest.json"
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    quarantine = {task: set() for task in tasks}
    origins: dict[str, list[str]] = defaultdict(list)
    for origin, payload in (("v03_canonical", canonical), ("v03_p1_pool", pool)):
        for record in payload.get("episodes", []):
            task = str(record.get("task", ""))
            source_sha = str(record.get("source_sha256", ""))
            if task in quarantine and source_sha:
                quarantine[task].add(source_sha)
                origins[source_sha].append(origin)
    counts = {task: len(values) for task, values in quarantine.items()}
    return quarantine, {
        "canonical_manifest": bind_file(canonical_path),
        "p1_pool_selection_manifest": bind_file(pool_path),
        "per_task_source_sha256_count": counts,
        "source_origins": dict(sorted(origins.items())),
    }


def _legacy_quarantine(workspace: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    quarantine, provenance = _legacy_source_inventory(workspace, HELDOUT_TASKS)
    counts = provenance["per_task_source_sha256_count"]
    _require(counts == LEGACY_QUARANTINE_COUNTS, f"Legacy quarantine provenance changed: {counts}")
    return quarantine, provenance


def _protocol(legacy: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "protocol_revision": "v04.1-approved-heldout-amendment-20260721",
        "seed": SEED,
        "sort_key": 'SHA256("20260711:" + source_relative_path)',
        "seen_tasks": list(SEEN_TASKS),
        "heldout_tasks": list(HELDOUT_TASKS),
        "approved_heldout_amendment": {
            "removed": "grab_pencil1_v1",
            "added": "push_plate_v1",
            "reason": "removed task was byte-identical to seen grab_pencil_v1; replacement passed formal source audit",
            "replacement_assessment_receipt_sha256": "6e6072ff085bc831010d9adb7a0a13edd143e6beed873e1d355acdbf7892d20d",
        },
        "legacy_quarantine_expected_source_sha256_count": dict(LEGACY_QUARANTINE_COUNTS),
        "seen_split": {"train": "first n-ceil(0.1*n)", "validation": "last max(1,ceil(0.1*n))"},
        "heldout_assignment_order": [
            "legacy_quarantine:all legacy source SHA",
            "v04_human_pool:first 10 valid new sources",
            "v04_robot_dev:next 5 valid new sources",
            "v04_robot_final:next 20 valid new sources",
            "reserve:remaining valid new sources",
        ],
        "H_steps": H_STEPS,
        "K_steps": K_STEPS,
        "gap_rule": "step diff != 1 or timestamp diff < 0 or timestamp diff > 1",
        "raw_human_dataset_allowlist": list(RAW_HUMAN_DATASETS),
        "raw_robot_dataset_allowlist": list(RAW_ROBOT_DATASETS),
        "human_projection_dataset_allowlist": list(HUMAN_PROJECTION_DATASETS),
        "robot_projection_dataset_allowlist": list(ROBOT_PROJECTION_DATASETS),
        "content_hash_definition": CONTENT_HASH_DEFINITION,
        "legacy_manifest_sha256": {
            "canonical": legacy["canonical_manifest"]["sha256"],
            "p1_pool": legacy["p1_pool_selection_manifest"]["sha256"],
        },
        "validation_checkpoint_policy": "monitor_only_fixed_step_7000_no_checkpoint_selection",
    }


def _identity_record(
    *,
    source_root: Path,
    path: Path,
    source_sha: str,
    partition: str,
    task: str,
    role: str,
    summary: Mapping[str, Any],
    partition_rank: int,
) -> dict[str, Any]:
    relative = path.relative_to(source_root).as_posix()
    record: dict[str, Any] = {
        "source_relative_path": relative,
        "source_sha256": source_sha,
        "source_partition": partition,
        "task": task,
        "episode_id": f"{task}/{path.stem}",
        "role": role,
        "partition_rank": partition_rank,
        "source_sort_sha256": source_rank(relative),
        "frame_count": int(summary["frame_count"]),
        "legal_window_count": int(summary["legal_window_count"]),
        "gap_count": int(summary["gap_count"]),
        "segment_count": int(summary["segment_count"]),
        "max_gap_safe_segment_frames": int(summary["max_gap_safe_segment_frames"]),
        "window_identity": {
            "format": "<source_sha256>:<legal_window_start>:H8:K8",
            "inherits": ["source_relative_path", "source_sha256", "source_partition", "task", "episode_id", "role"],
        },
    }
    if role in ("human", "paired"):
        record["human_content_sha256"] = role_content_sha256(source_sha, "human")
    if role in ("robot", "paired"):
        record["robot_content_sha256"] = role_content_sha256(source_sha, "robot")
    return record


def build_source_split(
    source_root: Path,
    workspace: Path,
    *,
    progress: Progress | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Scan, validate, hash, rank, and assign every v04-relevant raw episode."""

    source_root = source_root.resolve()
    _require(source_root.is_dir(), f"Missing raw Human2Robot root: {source_root}")
    quarantine, legacy = _legacy_quarantine(workspace)
    task_paths = {task: _task_paths(source_root, task) for task in (*SEEN_TASKS, *HELDOUT_TASKS)}
    discovered = {task: len(paths) for task, paths in task_paths.items()}
    _require(sum(discovered[task] for task in SEEN_TASKS) == SEEN_CANDIDATE_COUNT, f"Seen inventory changed: {discovered}")
    _require(sum(discovered[task] for task in HELDOUT_TASKS) == HELDOUT_CANDIDATE_COUNT, f"Held-out inventory changed: {discovered}")
    total = sum(discovered.values())
    completed = 0
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def inspect(path: Path, task: str) -> tuple[str, dict[str, Any] | None]:
        nonlocal completed
        relative = path.relative_to(source_root).as_posix()
        source_sha = file_sha256(path)
        try:
            summary = inspect_source(path)
        except (Stage1DataError, OSError, ValueError) as error:
            rejected.append(
                {
                    "task": task,
                    "source_relative_path": relative,
                    "source_sha256": source_sha,
                    "source_sort_sha256": source_rank(relative),
                    "rejection_reason": f"{type(error).__name__}: {error}",
                }
            )
            summary = None
        completed += 1
        if progress is not None and (completed % 10 == 0 or completed == total):
            progress(completed, total, f"scan:{relative}")
        return source_sha, summary

    for task in SEEN_TASKS:
        valid: list[tuple[Path, str, dict[str, Any]]] = []
        for path in task_paths[task]:
            source_sha, summary = inspect(path, task)
            if summary is not None:
                valid.append((path, source_sha, summary))
        _require(len(valid) >= 5, f"Seen task {task} has only {len(valid)} valid episodes")
        validation_count = max(1, math.ceil(0.1 * len(valid)))
        train_count = len(valid) - validation_count
        for index, (path, source_sha, summary) in enumerate(valid):
            partition = "seen_train" if index < train_count else "seen_validation"
            rank = index + 1 if partition == "seen_train" else index - train_count + 1
            records.append(
                _identity_record(
                    source_root=source_root,
                    path=path,
                    source_sha=source_sha,
                    partition=partition,
                    task=task,
                    role="paired",
                    summary=summary,
                    partition_rank=rank,
                )
            )

    for task in HELDOUT_TASKS:
        quarantined: list[tuple[Path, str, dict[str, Any]]] = []
        new_valid: list[tuple[Path, str, dict[str, Any]]] = []
        for path in task_paths[task]:
            source_sha, summary = inspect(path, task)
            if source_sha in quarantine[task]:
                _require(summary is not None, f"Legacy quarantined source no longer satisfies contract: {path}")
                quarantined.append((path, source_sha, summary))
            elif summary is not None:
                new_valid.append((path, source_sha, summary))
        found = {source_sha for _, source_sha, _ in quarantined}
        _require(found == quarantine[task], f"Legacy quarantine sources missing for {task}")
        _require(len(new_valid) >= 35, f"Only {len(new_valid)} legal new held-out episodes for {task}; need 35")
        assignments = (
            ("legacy_quarantine", "paired", quarantined),
            ("v04_human_pool", "human", new_valid[:10]),
            ("v04_robot_dev", "robot", new_valid[10:15]),
            ("v04_robot_final", "robot", new_valid[15:35]),
            ("reserve", "paired", new_valid[35:]),
        )
        for partition, role, items in assignments:
            for rank, (path, source_sha, summary) in enumerate(items, start=1):
                records.append(
                    _identity_record(
                        source_root=source_root,
                        path=path,
                        source_sha=source_sha,
                        partition=partition,
                        task=task,
                        role=role,
                        summary=summary,
                        partition_rank=rank,
                    )
                )
    records.sort(key=lambda item: (PARTITIONS.index(item["source_partition"]), item["task"], item["partition_rank"]))
    rejected.sort(key=lambda item: (item["task"], item["source_sort_sha256"]))
    return records, rejected, discovered, legacy


def assess_heldout_replacements(
    *,
    workspace: Path,
    source_root: Path,
    candidates: Sequence[str] = REPLACEMENT_CANDIDATES,
    replaced_task: str = "grab_pencil1_v1",
    required_new_count: int = 35,
    execute: bool,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Read-only evidence for a possible held-out task protocol amendment.

    A successful diagnostic never authorizes the amendment or stage-1
    materialization.  It proves only candidate inventory, source-SHA
    independence, and the raw H=8/K=8 data contract.
    """

    workspace = workspace.resolve()
    source_root = source_root.resolve()
    candidate_tasks = tuple(dict.fromkeys(str(task) for task in candidates))
    _require(candidate_tasks, "At least one replacement candidate is required")
    _require(replaced_task in PRE_AMENDMENT_HELDOUT_TASKS, f"Not a pre-amendment held-out task: {replaced_task}")
    _require(required_new_count > 0, "required_new_count must be positive")
    frozen_tasks = (*SEEN_TASKS, *PRE_AMENDMENT_HELDOUT_TASKS)
    for task in candidate_tasks:
        relative = Path(task)
        _require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe candidate task path: {task}")
        _require(task not in frozen_tasks, f"Candidate is already a frozen v04 task: {task}")
    if not execute:
        return {
            "schema_version": f"{SCHEMA}-heldout-replacement-assessment-v1",
            "status": "DRY_RUN",
            "would_scan": str(source_root),
            "frozen_baseline_tasks": list(frozen_tasks),
            "candidate_tasks": list(candidate_tasks),
            "replaced_task": replaced_task,
            "required_new_episode_count": required_new_count,
            "protocol_change_authorized": False,
            "stage1_prepare_allowed": False,
        }

    _require(source_root.is_dir(), f"Missing raw Human2Robot root: {source_root}")
    task_paths = {task: _task_paths(source_root, task) for task in (*frozen_tasks, *candidate_tasks)}
    legacy_sources, legacy = _legacy_source_inventory(workspace, candidate_tasks)
    total = sum(len(paths) for paths in task_paths.values())
    completed = 0
    inventory_for_digest: list[dict[str, str]] = []
    baseline_sha_to_sources: dict[str, list[dict[str, str]]] = defaultdict(list)

    for task in frozen_tasks:
        for path in task_paths[task]:
            relative = path.relative_to(source_root).as_posix()
            source_sha = file_sha256(path)
            inventory_for_digest.append({"path": relative, "sha256": source_sha})
            baseline_sha_to_sources[source_sha].append({"task": task, "path": relative})
            completed += 1
            if progress is not None and (completed % 25 == 0 or completed == total):
                progress(completed, total, f"replacement_sha_baseline:{relative}")

    assessments: dict[str, dict[str, Any]] = {}
    candidate_sha_sets: dict[str, set[str]] = {}
    for task in candidate_tasks:
        accepted_by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
        all_paths_by_sha: dict[str, list[str]] = defaultdict(list)
        rejected: list[dict[str, str]] = []
        for path in task_paths[task]:
            relative = path.relative_to(source_root).as_posix()
            source_sha = file_sha256(path)
            all_paths_by_sha[source_sha].append(relative)
            inventory_for_digest.append({"path": relative, "sha256": source_sha})
            try:
                summary = inspect_source(path)
            except (Stage1DataError, OSError, ValueError) as error:
                rejected.append(
                    {
                        "source_relative_path": relative,
                        "source_sha256": source_sha,
                        "rejection_reason": f"{type(error).__name__}: {error}",
                    }
                )
            else:
                accepted_by_sha[source_sha].append({"source_relative_path": relative, **summary})
            completed += 1
            if progress is not None and (completed % 10 == 0 or completed == total):
                progress(completed, total, f"replacement_contract:{relative}")

        raw_baseline_overlap = sorted(set(all_paths_by_sha).intersection(baseline_sha_to_sources))
        valid_baseline_overlap = sorted(set(accepted_by_sha).intersection(baseline_sha_to_sources))
        legacy_overlap = sorted(set(accepted_by_sha).intersection(legacy_sources[task]))
        fresh_sha = sorted(set(accepted_by_sha).difference(baseline_sha_to_sources, legacy_sources[task]))
        accepted_records = [record for values in accepted_by_sha.values() for record in values]
        overlap_examples = [
            {
                "source_sha256": source_sha,
                "candidate_paths": all_paths_by_sha[source_sha],
                "baseline_sources": baseline_sha_to_sources[source_sha],
            }
            for source_sha in raw_baseline_overlap[:20]
        ]
        assessments[task] = {
            "candidate_file_count": len(task_paths[task]),
            "contract_valid_file_count": len(accepted_records),
            "contract_valid_unique_source_sha256_count": len(accepted_by_sha),
            "contract_rejected_file_count": len(rejected),
            "contract_rejections": rejected,
            "within_candidate_duplicate_source_sha256_count": sum(len(values) - 1 for values in accepted_by_sha.values()),
            "baseline_overlap_unique_source_sha256_count": len(raw_baseline_overlap),
            "contract_valid_baseline_overlap_unique_source_sha256_count": len(valid_baseline_overlap),
            "baseline_overlap_examples": overlap_examples,
            "legacy_source_sha256_count": len(legacy_sources[task]),
            "legacy_source_present_in_candidate_count": len(legacy_overlap),
            "fresh_valid_unique_source_sha256_count": len(fresh_sha),
            "required_new_episode_count": required_new_count,
            "reserve_count_after_required_assignment": max(0, len(fresh_sha) - required_new_count),
            "eligible_by_stage1_data_contract": len(fresh_sha) >= required_new_count and not raw_baseline_overlap,
            "legal_window_count_total": sum(int(record["legal_window_count"]) for record in accepted_records),
            "frame_count_min": min((int(record["frame_count"]) for record in accepted_records), default=0),
            "frame_count_max": max((int(record["frame_count"]) for record in accepted_records), default=0),
            "gap_count_total": sum(int(record["gap_count"]) for record in accepted_records),
        }
        candidate_sha_sets[task] = set(all_paths_by_sha)

    pairwise_candidate_overlap = {
        f"{left}__{right}": len(candidate_sha_sets[left].intersection(candidate_sha_sets[right]))
        for left_index, left in enumerate(candidate_tasks)
        for right in candidate_tasks[left_index + 1 :]
    }
    eligible = [task for task in candidate_tasks if assessments[task]["eligible_by_stage1_data_contract"]]
    data_quality_order = sorted(
        eligible,
        key=lambda task: (
            assessments[task]["contract_rejected_file_count"] == 0,
            assessments[task]["fresh_valid_unique_source_sha256_count"],
            assessments[task]["legal_window_count_total"],
            task,
        ),
        reverse=True,
    )
    inventory_for_digest.sort(key=lambda item: item["path"])
    return {
        "schema_version": f"{SCHEMA}-heldout-replacement-assessment-v1",
        "status": "PASSED",
        "stage1_status": "BLOCKED_PROTOCOL",
        "completed_at_utc": utc_now(),
        "source_root": str(source_root),
        "replaced_task": replaced_task,
        "frozen_baseline_task_count": len(frozen_tasks),
        "frozen_baseline_source_file_count": sum(len(task_paths[task]) for task in frozen_tasks),
        "scanned_source_file_count": total,
        "raw_inventory_sha256": canonical_sha256(inventory_for_digest),
        "candidate_assessments": assessments,
        "pairwise_candidate_source_sha256_overlap_count": pairwise_candidate_overlap,
        "eligible_candidates": eligible,
        "data_quality_order": data_quality_order,
        "data_quality_preference": data_quality_order[0] if data_quality_order else None,
        "legacy_manifest_bindings": {
            "canonical_manifest": legacy["canonical_manifest"],
            "p1_pool_selection_manifest": legacy["p1_pool_selection_manifest"],
        },
        "protocol_change_authorized": False,
        "stage1_prepare_allowed": False,
        "interpretation": "Diagnostic completion is not approval to change the frozen held-out task list.",
    }


def _numeric(group: h5py.Group, name: str, values: np.ndarray) -> h5py.Dataset:
    array = np.ascontiguousarray(values)
    chunks = (min(256, max(1, len(array))), *array.shape[1:])
    return group.create_dataset(name, data=array, compression="gzip", compression_opts=4, shuffle=True, chunks=chunks)


def _copy_images(source: h5py.Dataset, group: h5py.Group) -> str:
    indices = np.linspace(0, len(source) - 1, num=min(8, len(source)), dtype=np.int64)
    probe = np.asarray(source[indices])
    if source.dtype == np.dtype("uint8"):
        conversion = "identity_uint8"
    elif int(np.max(probe)) <= 255:
        conversion = "uint16_container_cast_to_uint8"
    else:
        conversion = "uint16_full_range_divide_by_257"
    destination = group.create_dataset(
        "images",
        shape=source.shape,
        dtype=np.uint8,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        chunks=(1, *source.shape[1:]),
    )
    for start in range(0, len(source), 64):
        stop = min(start + 64, len(source))
        batch = np.asarray(source[start:stop])
        if conversion == "identity_uint8":
            converted = batch
        elif conversion == "uint16_container_cast_to_uint8":
            _require(int(np.max(batch)) <= 255, f"Inconsistent uint16 image scale: {source.file.filename}")
            converted = batch.astype(np.uint8)
        else:
            converted = np.clip(batch / 257.0, 0, 255).astype(np.uint8)
        destination[start:stop] = converted
    return conversion


def _dataset_paths(path: Path) -> list[str]:
    paths: list[str] = []
    with h5py.File(path, "r") as file:
        file.visititems(lambda name, value: paths.append(name) if isinstance(value, h5py.Dataset) else None)
    return sorted(paths)


def _projection_allowlist(role: str) -> tuple[str, ...]:
    if role == "human":
        return HUMAN_PROJECTION_DATASETS
    if role == "robot":
        return ROBOT_PROJECTION_DATASETS
    raise Stage1DataError(f"Cannot materialize role: {role}")


def verify_projection(path: Path, record: Mapping[str, Any], protocol_sha256: str) -> dict[str, Any]:
    expected = list(_projection_allowlist(str(record["role"])))
    actual = _dataset_paths(path)
    _require(actual == sorted(expected), f"Projection dataset allowlist mismatch: {path}: {actual}")
    with h5py.File(path, "r") as file:
        demo = file["data/demo_0"]
        for field in ("source_relative_path", "source_sha256", "source_partition", "task", "episode_id", "role"):
            _require(str(demo.attrs.get(field)) == str(record[field]), f"Projection identity mismatch {field}: {path}")
        _require(str(demo.attrs.get("protocol_sha256")) == protocol_sha256, f"Projection protocol mismatch: {path}")
        content_field = "human_content_sha256" if record["role"] == "human" else "robot_content_sha256"
        _require(str(demo.attrs.get(content_field)) == str(record[content_field]), f"Projection content identity mismatch: {path}")
        step = np.asarray(demo["time/source_step"][:], dtype=np.int64)
        timestamp = np.asarray(demo["time/source_timestamp"][:], dtype=np.int64)
        gap, segment, starts = _time_structure(step, timestamp)
        _require(np.array_equal(gap, demo["time/gap_mask"][:]), f"Projection gap mask mismatch: {path}")
        _require(np.array_equal(segment, demo["time/segment_id"][:]), f"Projection segment id mismatch: {path}")
        _require(np.array_equal(starts, demo["time/legal_window_start"][:]), f"Projection legal windows mismatch: {path}")
        if record["role"] == "human":
            numeric_paths = ("human/hand_action_7d", "human/hand_coords", "human/hand_frames")
        else:
            numeric_paths = ("robot/observed_eef_pose_6d", "robot/gripper_state")
        for dataset_path in numeric_paths:
            _require(np.isfinite(np.asarray(demo[dataset_path][:])).all(), f"Nonfinite derived data: {path}:{dataset_path}")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": file_sha256(path), "dataset_paths": actual}


def materialize_projection(
    source_root: Path,
    derived_root: Path,
    record: Mapping[str, Any],
    protocol_sha256: str,
) -> dict[str, Any]:
    role = str(record["role"])
    _require(role in ("human", "robot"), f"Unsupported projection role: {role}")
    partition_dir = "human_pool" if role == "human" else ("robot_dev" if record["source_partition"] == "v04_robot_dev" else "robot_final")
    source_path = source_root / str(record["source_relative_path"])
    destination = derived_root / "episodes" / partition_dir / str(record["task"]) / f"{int(record['partition_rank']):02d}_{source_path.name}"
    if destination.exists():
        return verify_projection(destination, record, protocol_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        with h5py.File(source_path, "r") as source, h5py.File(partial, "w") as output:
            output.attrs["schema_version"] = SCHEMA
            output.attrs["role_only"] = True
            data = output.create_group("data")
            demo = data.create_group("demo_0")
            for field in ("source_relative_path", "source_sha256", "source_partition", "task", "episode_id", "role"):
                demo.attrs[field] = record[field]
            demo.attrs["schema_version"] = SCHEMA
            demo.attrs["protocol_sha256"] = protocol_sha256
            demo.attrs["H_steps"] = H_STEPS
            demo.attrs["K_steps"] = K_STEPS
            demo.attrs["contains_opposite_role_fields"] = False
            demo.attrs["content_hash_definition"] = CONTENT_HASH_DEFINITION
            if role == "human":
                demo.attrs["human_content_sha256"] = record["human_content_sha256"]
                human = demo.create_group("human")
                demo.attrs["image_conversion"] = _copy_images(_dataset(source, "cam_data/human_camera"), human)
                _numeric(human, "hand_action_7d", np.asarray(_dataset(source, "action")[:], dtype=np.float32))
                _numeric(human, "hand_coords", np.asarray(_dataset(source, "transformed_hand_coords")[:], dtype=np.float32))
                _numeric(human, "hand_frames", np.asarray(_dataset(source, "transformed_hand_frames")[:], dtype=np.float32))
            else:
                demo.attrs["robot_content_sha256"] = record["robot_content_sha256"]
                robot = demo.create_group("robot")
                demo.attrs["image_conversion"] = _copy_images(_dataset(source, "cam_data/robot_camera"), robot)
                _numeric(robot, "observed_eef_pose_6d", np.asarray(_dataset(source, "end_position")[:], dtype=np.float32))
                _numeric(robot, "gripper_state", np.asarray(_dataset(source, "gripper_state")[:], dtype=np.float32))
            step = np.asarray(_dataset(source, "step")[:], dtype=np.int64)
            timestamp = np.asarray(_dataset(source, "timestamp")[:], dtype=np.int64)
            gap, segment, starts = _time_structure(step, timestamp)
            time = demo.create_group("time")
            _numeric(time, "source_step", step)
            _numeric(time, "source_timestamp", timestamp)
            _numeric(time, "gap_mask", gap)
            _numeric(time, "segment_id", segment.astype(np.int32))
            _numeric(time, "legal_window_start", starts)
            demo.attrs["frame_count"] = len(step)
            demo.attrs["legal_window_count"] = len(starts)
            output.flush()
        os.replace(partial, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            partial.unlink()
    return verify_projection(destination, record, protocol_sha256)


def _partition_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(record["source_partition"]) for record in records)
    per_task: dict[str, dict[str, int]] = {}
    for task in (*SEEN_TASKS, *HELDOUT_TASKS):
        task_counts = Counter(str(record["source_partition"]) for record in records if record["task"] == task)
        per_task[task] = dict(sorted(task_counts.items()))
    return {"total": len(records), "partition_counts": {name: counts[name] for name in PARTITIONS}, "per_task": per_task}


def validate_invariants(records: Sequence[Mapping[str, Any]], rejected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = {
        "source_relative_path",
        "source_sha256",
        "source_partition",
        "task",
        "episode_id",
        "role",
        "legal_window_count",
        "window_identity",
    }
    for record in records:
        _require(required.issubset(record), f"Missing identity field(s): {required - set(record)}")
        _require(record["source_partition"] in PARTITIONS, f"Unknown partition: {record['source_partition']}")
        _require(int(record["legal_window_count"]) > 0, f"Episode has no legal windows: {record['episode_id']}")
        if record["role"] == "human":
            _require("human_content_sha256" in record and "robot_content_sha256" not in record, "Human role identity leak")
        if record["role"] == "robot":
            _require("robot_content_sha256" in record and "human_content_sha256" not in record, "Robot role identity leak")
    paths = [str(record["source_relative_path"]) for record in records]
    _require(len(set(paths)) == len(paths), "A source path appears in multiple partitions")
    rejected_paths = [str(record["source_relative_path"]) for record in rejected]
    _require(len(set(rejected_paths)) == len(rejected_paths), "A rejected source path appears more than once")
    _require(not set(paths).intersection(rejected_paths), "A source path is both accepted and rejected")
    for record in rejected:
        _require(
            {"source_relative_path", "source_sha256", "task", "rejection_reason"}.issubset(record),
            f"Rejected source lacks audit identity: {record}",
        )
    by_partition = {partition: {str(record["source_sha256"]) for record in records if record["source_partition"] == partition} for partition in PARTITIONS}
    within_partition_duplicate_sha256_count = {
        partition: sum(1 for count in Counter(str(record["source_sha256"]) for record in records if record["source_partition"] == partition).values() if count > 1)
        for partition in PARTITIONS
    }
    overlaps = {
        f"{left}__{right}": sorted(by_partition[left].intersection(by_partition[right]))
        for left_index, left in enumerate(PARTITIONS)
        for right in PARTITIONS[left_index + 1 :]
    }
    nonempty_overlaps = {name: values[:20] for name, values in overlaps.items() if values}
    _require(
        not nonempty_overlaps,
        f"Source SHA overlap between partitions: {json.dumps(nonempty_overlaps, sort_keys=True)}",
    )
    summary = _partition_summary(records)
    for task in SEEN_TASKS:
        counts = summary["per_task"][task]
        _require(counts.get("seen_train", 0) > 0 and counts.get("seen_validation", 0) >= 1, f"Incomplete seen split: {task}")
        _require(counts.get("seen_train", 0) + counts.get("seen_validation", 0) >= 5, f"Seen task below minimum: {task}")
    for task in HELDOUT_TASKS:
        counts = summary["per_task"][task]
        expected = {
            "legacy_quarantine": LEGACY_QUARANTINE_COUNTS[task],
            "v04_human_pool": 10,
            "v04_robot_dev": 5,
            "v04_robot_final": 20,
        }
        _require(all(counts.get(name, 0) == count for name, count in expected.items()), f"Held-out counts changed for {task}: {counts}")
    _require(summary["partition_counts"]["v04_robot_dev"] == 20, "Dev must contain 20 robot episodes")
    _require(summary["partition_counts"]["v04_robot_final"] == 80, "Final must contain 80 robot episodes")
    return {
        **summary,
        "pairwise_source_sha256_overlap": overlaps,
        "pairwise_source_sha256_overlap_count": 0,
        "within_partition_duplicate_sha256_count": within_partition_duplicate_sha256_count,
        "rejected_candidate_count": len(rejected),
        "raw_candidate_count": len(records) + len(rejected),
        "all_accepted_files_have_finite_gap_safe_H8_K8_windows": True,
    }


def _split_payload(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: record[key]
            for key in (
                "source_relative_path",
                "source_sha256",
                "source_partition",
                "task",
                "episode_id",
                "role",
                "partition_rank",
                "human_content_sha256",
                "robot_content_sha256",
            )
            if key in record
        }
        for record in records
    ]


def _raw_inventory_payload(
    records: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    return [
        {"path": str(item["source_relative_path"]), "sha256": str(item["source_sha256"])}
        for item in (*records, *rejected)
    ]


def prepare_data(
    *,
    workspace: Path,
    source_root: Path,
    derived_root: Path,
    execute: bool,
    progress: Progress | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    source_root = source_root.resolve()
    derived_root = derived_root.resolve()
    manifest_path = derived_root / "source_split_manifest.json"
    lock_path = derived_root / "source_split_manifest.lock.json"
    if not execute:
        return {
            "schema_version": SCHEMA,
            "status": "DRY_RUN",
            "would_scan": str(source_root),
            "would_write": str(derived_root),
            "commands_require_execute": True,
            "seen_candidate_count_expected": SEEN_CANDIDATE_COUNT,
            "heldout_candidate_count_expected": HELDOUT_CANDIDATE_COUNT,
            "heldout_legacy_quarantine_per_task": dict(LEGACY_QUARANTINE_COUNTS),
            "heldout_exact_new_targets_per_task": {"human_pool": 10, "robot_dev": 5, "robot_final": 20},
        }
    _require(not manifest_path.exists() and not lock_path.exists(), f"Frozen stage-1 manifest already exists: {manifest_path}")
    records, rejected, discovered, legacy = build_source_split(source_root, workspace, progress=progress)
    validation = validate_invariants(records, rejected)
    protocol = _protocol(legacy)
    protocol_sha256 = canonical_sha256(protocol)
    projections = [record for record in records if record["source_partition"] in ("v04_human_pool", "v04_robot_dev", "v04_robot_final")]
    total_projection = len(projections)
    for index, record in enumerate(projections, start=1):
        record["projection"] = materialize_projection(source_root, derived_root, record, protocol_sha256)
        if progress is not None and (index % 5 == 0 or index == total_projection):
            progress(index, total_projection, f"materialize:{record['source_partition']}:{record['episode_id']}")
    generation_code = [bind_file(Path(__file__)), bind_file(workspace / "tools/human2robot_v04.py")]
    split_payload = _split_payload(records)
    manifest = {
        "schema_version": SCHEMA,
        "status": "frozen",
        "created_at_utc": utc_now(),
        "source_root": str(source_root),
        "derived_root": str(derived_root),
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "split_sha256": canonical_sha256({"protocol_sha256": protocol_sha256, "records": split_payload}),
        "raw_inventory_sha256": canonical_sha256(_raw_inventory_payload(records, rejected)),
        "generation_code": generation_code,
        "legacy_quarantine_provenance": legacy,
        "discovered_source_episode_count": discovered,
        "records": records,
        "rejected_candidates": rejected,
        "validation": validation,
    }
    write_json_atomic(manifest_path, manifest, mode=0o444)
    lock = {
        "schema_version": f"{SCHEMA}-lock",
        "status": "locked",
        "created_at_utc": utc_now(),
        "manifest": bind_file(manifest_path),
        "protocol_sha256": protocol_sha256,
        "split_sha256": manifest["split_sha256"],
    }
    write_json_atomic(lock_path, lock, mode=0o444)
    return {
        "schema_version": SCHEMA,
        "status": "PASSED",
        "manifest": bind_file(manifest_path),
        "lock": bind_file(lock_path),
        "protocol_sha256": protocol_sha256,
        "split_sha256": manifest["split_sha256"],
        "validation": validation,
    }


def audit_data(
    *,
    workspace: Path,
    source_root: Path,
    derived_root: Path,
    execute: bool,
    progress: Progress | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    source_root = source_root.resolve()
    derived_root = derived_root.resolve()
    manifest_path = derived_root / "source_split_manifest.json"
    lock_path = derived_root / "source_split_manifest.lock.json"
    report_path = derived_root / "stage1_data_audit_report.json"
    if not execute:
        return {
            "schema_version": f"{SCHEMA}-audit",
            "status": "DRY_RUN",
            "would_verify_manifest": str(manifest_path),
            "would_rehash_raw_sources": True,
            "would_verify_all_projections": True,
        }
    _require(manifest_path.is_file() and lock_path.is_file(), "Stage-1 manifest/lock is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == SCHEMA, "Stage-1 manifest schema mismatch")
    _require(lock.get("schema_version") == f"{SCHEMA}-lock" and lock.get("status") == "locked", "Stage-1 lock schema/status mismatch")
    _require(manifest.get("source_root") == str(source_root), "Stage-1 manifest source root mismatch")
    _require(manifest.get("derived_root") == str(derived_root), "Stage-1 manifest derived root mismatch")
    _require(lock["manifest"]["sha256"] == file_sha256(manifest_path), "Frozen stage-1 manifest hash changed")
    _require(manifest["protocol_sha256"] == canonical_sha256(manifest["protocol"]), "Stage-1 protocol hash mismatch")
    _require(lock.get("protocol_sha256") == manifest["protocol_sha256"], "Stage-1 lock protocol hash mismatch")
    _require(lock.get("split_sha256") == manifest["split_sha256"], "Stage-1 lock split hash mismatch")
    for binding in manifest["generation_code"]:
        path = Path(binding["path"])
        _require(path.is_file() and file_sha256(path) == binding["sha256"], f"Generation code changed after prepare-data: {path}")
    records = manifest["records"]
    rejected = manifest.get("rejected_candidates", [])
    validation = validate_invariants(records, rejected)
    expected_split_sha256 = canonical_sha256(
        {"protocol_sha256": manifest["protocol_sha256"], "records": _split_payload(records)}
    )
    _require(manifest["split_sha256"] == expected_split_sha256, "Stage-1 split hash mismatch")
    expected_inventory_sha256 = canonical_sha256(_raw_inventory_payload(records, rejected))
    _require(manifest["raw_inventory_sha256"] == expected_inventory_sha256, "Stage-1 raw inventory hash mismatch")
    discovered_total = sum(int(count) for count in manifest["discovered_source_episode_count"].values())
    _require(discovered_total == len(records) + len(rejected), "Stage-1 discovered inventory count mismatch")
    total = len(records) + len(rejected)
    raw_hash_mismatches: list[str] = []
    projection_checks: list[dict[str, Any]] = []
    for index, record in enumerate((*records, *rejected), start=1):
        raw_path = source_root / record["source_relative_path"]
        if not raw_path.is_file() or file_sha256(raw_path) != record["source_sha256"]:
            raw_hash_mismatches.append(record["source_relative_path"])
        projection = record.get("projection")
        if projection is not None:
            output_path = Path(projection["path"])
            checked = verify_projection(output_path, record, manifest["protocol_sha256"])
            _require(checked["sha256"] == projection["sha256"], f"Projection hash changed: {output_path}")
            projection_checks.append(checked)
        if progress is not None and (index % 10 == 0 or index == total):
            progress(index, total, f"audit:{record['source_relative_path']}")
    _require(not raw_hash_mismatches, f"Raw source hashes changed: {raw_hash_mismatches[:10]}")
    partials = sorted(str(path) for path in derived_root.rglob("*.partial"))
    _require(not partials, f"Incomplete partial files remain: {partials[:10]}")
    _require(len(projection_checks) == 140, f"Expected 140 role-isolated projections, got {len(projection_checks)}")
    report = {
        "schema_version": f"{SCHEMA}-audit",
        "status": "PASSED",
        "completed_at_utc": utc_now(),
        "manifest": bind_file(manifest_path),
        "lock": bind_file(lock_path),
        "protocol_sha256": manifest["protocol_sha256"],
        "split_sha256": manifest["split_sha256"],
        "raw_source_hash_mismatch_count": 0,
        "raw_source_hash_verified_count": total,
        "accepted_source_hash_verified_count": len(records),
        "rejected_source_hash_verified_count": len(rejected),
        "raw_inventory_sha256": manifest["raw_inventory_sha256"],
        "projection_count": len(projection_checks),
        "human_projection_count": validation["partition_counts"]["v04_human_pool"],
        "robot_dev_projection_count": validation["partition_counts"]["v04_robot_dev"],
        "robot_final_projection_count": validation["partition_counts"]["v04_robot_final"],
        "projection_dataset_allowlists": {
            "human": list(HUMAN_PROJECTION_DATASETS),
            "robot": list(ROBOT_PROJECTION_DATASETS),
        },
        "validation": validation,
        "raw_data_open_mode": "read_only",
        "future_stage_authorization": {"stage2_allowed": True, "training_allowed": False},
    }
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        _require(previous.get("manifest", {}).get("sha256") == report["manifest"]["sha256"], "Existing audit report binds another manifest")
    else:
        write_json_atomic(report_path, report, mode=0o444)
    return {**report, "report": bind_file(report_path)}
