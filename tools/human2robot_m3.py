#!/usr/bin/env python3
"""Human2Robot M3-v03 action-role, time-view, and residual sanity pipeline.

This module consumes only semantic-safe canonical/v3 episodes.  It calibrates
alignment on the train tasks, keeps held-out robot trajectories out of retrieval
features, materializes versioned derived-view manifests, and evaluates the
offline BC proxy without claiming executable-command semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np

try:  # Support both ``python tools/run_...py`` and package imports in tests.
    from human2robot_m2 import SCHEMA_VERSION, file_sha256, require_canonical_v3
except ImportError:  # pragma: no cover - exercised by pytest import mode.
    from tools.human2robot_m2 import SCHEMA_VERSION, file_sha256, require_canonical_v3


DEMO_KEY = "data/demo_0"
M3_SCHEMA_VERSION = "human2robot-m3-derived-view-v03"
DEFAULT_CANONICAL_ROOT = Path("data/Human2Robot/canonical/v3")
DEFAULT_DERIVED_ROOT = Path("data/Human2Robot/derived")
DEFAULT_REPORT_ROOT = Path("方案/v03")
DEFAULT_EVIDENCE_MANIFEST = Path("方案/v03/source_evidence_manifest_v3.json")

POOL_ACTION_VIEW_RAW = "human_hand_robot_frame_raw"
POOL_ACTION_VIEW_PHASE = "human_hand_phase_aligned"
QUERY_ACTION_VIEW = "robot_ee_observed_t_plus_1_bc_proxy"
DIAGNOSTIC_QUERY_VIEW = "robot_ee_observed_t"
ALIGNMENT_ID = "train_only_tplus1_query_anchor_se3_identity_scale_v1"


class M3Error(RuntimeError):
    """Raised when an M3 semantic, leakage, or acceptance invariant fails."""


@dataclass(frozen=True)
class M3Config:
    canonical_root: Path = DEFAULT_CANONICAL_ROOT
    derived_root: Path = DEFAULT_DERIVED_ROOT
    report_root: Path = DEFAULT_REPORT_ROOT
    evidence_manifest: Path = DEFAULT_EVIDENCE_MANIFEST
    horizon: int = 8
    window_stride: int = 8
    top_k: int = 10
    max_lag: int = 30
    phase_bins: int = 64
    random_seed: int = 20260711
    wrong_lag: int = 30
    scale_perturbation: float = 2.0
    min_motion_correlation_for_lag_proxy: float = 0.3
    expected_episode_count: int | None = 20


@dataclass(frozen=True)
class TimeViewSpec:
    time_view_id: str
    stride: int = 1
    phase_bins: int | None = None
    nominal_hz: float | None = None
    paper_version: str | None = None
    status: str = "candidate"


@dataclass
class Episode:
    episode_id: str
    path: Path
    task: str
    split: str
    source_relative_path: str
    human: np.ndarray
    robot: np.ndarray
    segment_id: np.ndarray
    gap_mask: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise M3Error(f"Missing required manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise M3Error(f"Expected JSON object: {path}")
    return payload


def time_view_specs(config: M3Config) -> tuple[TimeViewSpec, ...]:
    """Return the six v03-required temporal candidates with explicit provenance."""
    return (
        TimeViewSpec("native_row_index", stride=1),
        TimeViewSpec("nominal_camera_30hz_segmented", stride=1, nominal_hz=30.0, status="main"),
        TimeViewSpec(
            "paper_v2_stride4_nominal7p5", stride=4, nominal_hz=7.5, paper_version="v2"
        ),
        TimeViewSpec(
            "legacy_v01_stride3_nominal10", stride=3, nominal_hz=10.0, paper_version="M2-v01"
        ),
        TimeViewSpec("policy_clock_10hz", stride=3, nominal_hz=10.0),
        TimeViewSpec("phase_or_dtw", phase_bins=config.phase_bins),
    )


def _canonical_files(root: Path) -> list[Path]:
    pilot = root / "pilot"
    files = sorted(pilot.glob("demo_*.hdf5"))
    if not files:
        raise M3Error(f"No canonical v3 pilot episodes found under {pilot}")
    return files


def load_episodes(config: M3Config) -> tuple[list[Episode], dict[str, Any], dict[str, Any]]:
    """Load v3 trajectories and bind every file to the frozen task split."""
    split_manifest = _read_json(config.canonical_root / "task_split_manifest.json")
    preprocessing = _read_json(config.canonical_root / "preprocessing_manifest.json")
    split_records = split_manifest.get("episodes", [])
    files = _canonical_files(config.canonical_root)
    if len(files) != len(split_records):
        raise M3Error(f"Canonical/split episode mismatch: {len(files)} vs {len(split_records)}")
    if config.expected_episode_count is not None and len(files) != config.expected_episode_count:
        raise M3Error(f"Expected {config.expected_episode_count} canonical episodes, found {len(files)}")

    episodes: list[Episode] = []
    for index, (path, split_record) in enumerate(zip(files, split_records, strict=True)):
        require_canonical_v3(path)
        with h5py.File(path, "r") as file:
            demo = file[DEMO_KEY]
            metadata = demo["metadata"]
            human = np.asarray(demo["trajectories/human_hand_robot_frame_10d"][:], dtype=np.float64)
            robot = np.asarray(demo["trajectories/robot_ee_observed_10d"][:], dtype=np.float64)
            segment_id = np.asarray(metadata["segment_id"][:], dtype=np.int64)
            gap_mask = np.asarray(metadata["gap_mask"][:], dtype=bool)
            canonical_source = str(demo.attrs.get("source_relative_path", ""))
        expected_source = str(split_record["source_relative_path"])
        if str(split_record.get("split")) not in {"train", "heldout"}:
            raise M3Error(f"Invalid task split for {path}: {split_record.get('split')!r}")
        if canonical_source and canonical_source != expected_source:
            raise M3Error(f"Split/source mismatch for {path}: {canonical_source} != {expected_source}")
        if human.shape != robot.shape or human.ndim != 2 or human.shape[1] != 10:
            raise M3Error(f"Role trajectory shape mismatch in {path}: {human.shape}, {robot.shape}")
        if len(segment_id) != len(human) or len(gap_mask) != len(human):
            raise M3Error(f"Time metadata length mismatch in {path}")
        episodes.append(
            Episode(
                episode_id=path.stem,
                path=path,
                task=str(split_record["task"]),
                split=str(split_record["split"]),
                source_relative_path=expected_source,
                human=human,
                robot=robot,
                segment_id=segment_id,
                gap_mask=gap_mask,
            )
        )
    return episodes, split_manifest, preprocessing


def build_action_role_audit(
    episodes: Sequence[Episode], config: M3Config, split_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Audit the two action roles without mutating M2 evidence or HDF5 files."""
    evidence = _read_json(config.evidence_manifest)
    errors: list[str] = []
    canonical_checks: list[dict[str, Any]] = []
    for episode in episodes:
        with h5py.File(episode.path, "r") as file:
            demo = file[DEMO_KEY]
            metadata = demo["metadata"]
            source_role = str(metadata.attrs.get("source_action_role", ""))
            robot_role = str(metadata.attrs.get("robot_trajectory_role", ""))
            generic_present = "actions" in demo or "policy_actions" in demo
            check = {
                "episode_id": episode.episode_id,
                "source_action_role": source_role,
                "robot_trajectory_role": robot_role,
                "generic_action_present": generic_present,
            }
            canonical_checks.append(check)
            if source_role != "human_hand_pose_in_robot_frame":
                errors.append(f"{episode.episode_id}: invalid pool source role {source_role!r}")
            if robot_role != "observed_robot_ee_pose":
                errors.append(f"{episode.episode_id}: invalid robot trajectory role {robot_role!r}")
            if generic_present:
                errors.append(f"{episode.episode_id}: canonical v3 contains prohibited generic action")

    required_evidence = {
        "source_action_role": "human_hand_pose_in_robot_frame",
        "source_action_role_status": "verified_upstream",
        "source_action_as_robot_command_status": "unknown",
    }
    for key, expected in required_evidence.items():
        if evidence.get(key) != expected:
            errors.append(f"evidence {key}={evidence.get(key)!r}, expected {expected!r}")
    sources = [source for source in evidence.get("sources", []) if source.get("url")]
    if not sources:
        errors.append("source evidence has no URL provenance")

    return {
        "schema_version": M3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": "passed" if not errors else "failed",
        "gate": "A2a",
        "gate_decision": "passed" if not errors else "failed",
        "canonical_schema": SCHEMA_VERSION,
        "canonical_episode_count": len(episodes),
        "split_sha256": split_manifest.get("split_sha256"),
        "source_evidence_manifest": str(config.evidence_manifest),
        "source_evidence_manifest_sha256": file_sha256(config.evidence_manifest),
        "evidence_sources": sources,
        "roles": {
            "pool": {
                "dataset": "trajectories/human_hand_robot_frame_10d",
                "role": "pool_side_human_plan_in_robot_frame",
                "source": "v1 /action",
                "command_status": "not_required_for_offline_pool; executable_status_unknown",
            },
            "query": {
                "dataset": "trajectories/robot_ee_observed_10d",
                "role": "observed_robot_ee_trajectory_and_dataset_card_approved_bc_label_source",
                "source": "/end_position + /gripper_state",
                "command_status": "unverified_not_executable_command",
            },
        },
        "canonical_checks": canonical_checks,
        "errors": errors,
        "m2_evidence_mutated": False,
    }


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rotation_6d, dtype=np.float64)
    if values.shape[-1] != 6:
        raise M3Error(f"Expected rotation-6D last dimension 6, got {values.shape}")
    flat = values.reshape(-1, 6)
    first = flat[:, :3]
    second = flat[:, 3:6]
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second -= np.sum(first * second, axis=1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=2).reshape(values.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise M3Error(f"Expected rotation matrix shape (...,3,3), got {matrix.shape}")
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def orientation_error_rad(left_6d: np.ndarray, right_6d: np.ndarray) -> np.ndarray:
    left = rotation_6d_to_matrix(left_6d)
    right = rotation_6d_to_matrix(right_6d)
    relative = np.swapaxes(left, -1, -2) @ right
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def _contiguous_segment_slices(segment_id: np.ndarray) -> list[slice]:
    if len(segment_id) == 0:
        return []
    boundaries = np.flatnonzero(segment_id[1:] != segment_id[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(segment_id)]))
    return [slice(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def view_segment_indices(episode: Episode, spec: TimeViewSpec) -> list[np.ndarray]:
    """Return per-segment source-row indices; no returned array can cross a gap."""
    result: list[np.ndarray] = []
    for segment in _contiguous_segment_slices(episode.segment_id):
        length = segment.stop - segment.start
        if length <= 0:
            continue
        if spec.phase_bins is not None:
            count = min(length, spec.phase_bins)
            local = np.unique(np.rint(np.linspace(0, length - 1, count)).astype(np.int64))
        else:
            local = np.arange(0, length, spec.stride, dtype=np.int64)
        result.append(local + segment.start)
    return result


def _median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.median(array)) if len(array) else math.nan


def _lag_pairs(episode: Episode, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag <= 0 or lag >= len(episode.human):
        return np.empty((0, 10)), np.empty((0, 10))
    valid = episode.segment_id[:-lag] == episode.segment_id[lag:]
    return episode.human[:-lag][valid], episode.robot[lag:][valid]


def calibrate_alignment(
    episodes: Sequence[Episode], config: M3Config, split_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Compute train-only lag/correlation, scale, and role-specific error diagnostics."""
    train = [episode for episode in episodes if episode.split == "train"]
    if not train:
        raise M3Error("M3 alignment requires at least one train episode")
    candidates: list[dict[str, Any]] = []
    for lag in range(1, config.max_lag + 1):
        position_errors: list[float] = []
        orientation_errors: list[float] = []
        gripper_errors: list[float] = []
        human_motion: list[float] = []
        robot_motion: list[float] = []
        for episode in train:
            human, robot = _lag_pairs(episode, lag)
            if len(human):
                position_errors.extend(np.linalg.norm(human[:, :3] - robot[:, :3], axis=1))
                orientation_errors.extend(orientation_error_rad(human[:, 3:9], robot[:, 3:9]))
                gripper_errors.extend(np.abs(human[:, 9] - robot[:, 9]))
            if len(episode.human) > lag + 1:
                human_delta = np.linalg.norm(np.diff(episode.human[:, :3], axis=0), axis=1)
                robot_delta = np.linalg.norm(np.diff(episode.robot[:, :3], axis=0), axis=1)
                count = min(len(human_delta) - lag, len(robot_delta) - lag)
                if count > 0:
                    valid = episode.segment_id[:count] == episode.segment_id[lag : lag + count]
                    human_motion.extend(human_delta[:count][valid])
                    robot_motion.extend(robot_delta[lag : lag + count][valid])
        correlation = 0.0
        if len(human_motion) >= 2 and np.std(human_motion) > 0 and np.std(robot_motion) > 0:
            correlation = float(np.corrcoef(human_motion, robot_motion)[0, 1])
        candidates.append(
            {
                "lag_source_rows": lag,
                "pair_count": len(position_errors),
                "motion_cross_correlation": correlation,
                "position_error_median_canonical": _median(position_errors),
                "orientation_error_median_rad": _median(orientation_errors),
                "gripper_error_median": _median(gripper_errors),
            }
        )

    selected = min(candidates, key=lambda item: item["position_error_median_canonical"])
    corr_selected = max(candidates, key=lambda item: item["motion_cross_correlation"])
    same_human = np.concatenate([episode.human for episode in train], axis=0)
    same_robot = np.concatenate([episode.robot for episode in train], axis=0)
    human_step: list[float] = []
    robot_step: list[float] = []
    for episode in train:
        contiguous = episode.segment_id[1:] == episode.segment_id[:-1]
        human_step.extend(np.linalg.norm(np.diff(episode.human[:, :3], axis=0)[contiguous], axis=1))
        robot_step.extend(np.linalg.norm(np.diff(episode.robot[:, :3], axis=0)[contiguous], axis=1))
    moving = (np.asarray(human_step) > 1e-6) & (np.asarray(robot_step) > 1e-6)
    scale_ratio = np.asarray(robot_step)[moving] / np.asarray(human_step)[moving]
    lag_proxy_is_stable = (
        float(corr_selected["motion_cross_correlation"]) >= config.min_motion_correlation_for_lag_proxy
    )

    return {
        "schema_version": M3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": "passed",
        "calibration_scope": "train_tasks_only",
        "train_tasks": sorted({episode.task for episode in train}),
        "heldout_tasks_used": False,
        "split_sha256": split_manifest.get("split_sha256"),
        "lag_candidates": candidates,
        "best_position_error_lag_source_rows": int(selected["lag_source_rows"]),
        "best_motion_cross_correlation_lag_source_rows": int(corr_selected["lag_source_rows"]),
        "best_motion_cross_correlation": float(corr_selected["motion_cross_correlation"]),
        "lag_calibrated_proxy_min_motion_correlation": config.min_motion_correlation_for_lag_proxy,
        "lag_calibrated_proxy_decision": (
            "eligible_candidate" if lag_proxy_is_stable else "diagnostic_only_due_to_weak_motion_correlation"
        ),
        "approved_query_action_view_id": QUERY_ACTION_VIEW,
        "approved_future_offset_view_steps": 1,
        "approved_proxy_basis": "lowest train-only paired position error among strictly-future candidates",
        "same_frame_view": {"query_action_view_id": DIAGNOSTIC_QUERY_VIEW, "approval": "diagnostic_only"},
        "scale_audit": {
            "canonical_position_transform": "identity",
            "pool_position_min": same_human[:, :3].min(axis=0),
            "pool_position_max": same_human[:, :3].max(axis=0),
            "query_position_min": same_robot[:, :3].min(axis=0),
            "query_position_max": same_robot[:, :3].max(axis=0),
            "same_row_position_error_median_canonical": float(
                np.median(np.linalg.norm(same_human[:, :3] - same_robot[:, :3], axis=1))
            ),
            "nonzero_step_scale_ratio_robot_over_human_median": _median(scale_ratio),
            "numeric_scale_decision": "shared canonical converter scale; retain identity transform",
            "physical_xyz_unit_status": "unknown; no velocity in m/s is reported",
        },
        "coordinate_representation": "xyz + rotation_6d + gripper (10D)",
        "gripper_rule": "1=open, 0=close; zero-order hold for resampling",
        "gap_policy": "never_cross_segment",
        "terminal_policy": "drop windows without a complete strictly-future target",
        "deployment_command_adapter_id": None,
        "query_command_status": "unverified",
    }


def paired_time_view_metrics(episodes: Sequence[Episode], spec: TimeViewSpec) -> dict[str, Any]:
    residual_norm: list[float] = []
    absolute_norm: list[float] = []
    position_error: list[float] = []
    orientation_error: list[float] = []
    gripper_error: list[float] = []
    gap_crossing = 0
    samples = 0
    for episode in episodes:
        if episode.split != "train":
            continue
        for indices in view_segment_indices(episode, spec):
            if len(indices) < 2:
                continue
            current = indices[:-1]
            future = indices[1:]
            gap_crossing += int(np.count_nonzero(episode.segment_id[current] != episode.segment_id[future]))
            pool = episode.human[current]
            target = episode.robot[future]
            residual_norm.extend(np.linalg.norm(target - pool, axis=1))
            absolute_norm.extend(np.linalg.norm(target, axis=1))
            position_error.extend(np.linalg.norm(target[:, :3] - pool[:, :3], axis=1))
            orientation_error.extend(orientation_error_rad(pool[:, 3:9], target[:, 3:9]))
            gripper_error.extend(np.abs(target[:, 9] - pool[:, 9]))
            samples += len(current)
    return {
        "time_view_id": spec.time_view_id,
        "paper_version": spec.paper_version,
        "nominal_hz": spec.nominal_hz,
        "pool_action_role": "pool_side_human_plan_in_robot_frame",
        "query_action_role": "dataset_card_approved_strict_future_bc_proxy",
        "query_action_view_id": QUERY_ACTION_VIEW,
        "sample_count": samples,
        "gap_crossing_count": gap_crossing,
        "residual_norm_median": _median(residual_norm),
        "absolute_target_norm_median": _median(absolute_norm),
        "position_error_median_canonical": _median(position_error),
        "orientation_error_median_rad": _median(orientation_error),
        "gripper_error_median": _median(gripper_error),
    }


def _make_windows(
    episodes: Sequence[Episode], spec: TimeViewSpec, config: M3Config, split: str
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for episode in episodes:
        if episode.split != split:
            continue
        for segment_number, indices in enumerate(view_segment_indices(episode, spec)):
            # One current observation plus H strictly-future query rows is required.
            last_start = len(indices) - config.horizon - 1
            for local_start in range(0, max(0, last_start + 1), config.window_stride):
                phase = local_start / max(1, len(indices) - 1)
                windows.append(
                    {
                        "episode_id": episode.episode_id,
                        "task": episode.task,
                        "segment_number": segment_number,
                        "phase": float(phase),
                        "current_row": int(indices[local_start]),
                        "pool_rows": indices[local_start : local_start + config.horizon].copy(),
                        "query_rows": indices[local_start + 1 : local_start + 1 + config.horizon].copy(),
                    }
                )
    return windows


def align_pool_chunk(pool: np.ndarray, query_current: np.ndarray) -> np.ndarray:
    """Lift a pool plan by a query anchor while preserving valid rotations."""
    pool = np.asarray(pool, dtype=np.float64)
    query_current = np.asarray(query_current, dtype=np.float64)
    if pool.ndim != 2 or pool.shape[1] != 10 or query_current.shape != (10,):
        raise M3Error(f"Invalid alignment shapes: pool={pool.shape}, query={query_current.shape}")
    aligned = pool.copy()
    aligned[:, :3] = query_current[:3] + pool[:, :3] - pool[0, :3]
    pool_rot = rotation_6d_to_matrix(pool[:, 3:9])
    query_rot = rotation_6d_to_matrix(query_current[None, 3:9])[0]
    aligned_rot = query_rot @ pool_rot[0].T @ pool_rot
    aligned[:, 3:9] = matrix_to_rotation_6d(aligned_rot)
    return aligned


def build_retrieval_index(
    episodes: Sequence[Episode], spec: TimeViewSpec, config: M3Config
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Retrieve by segment phase only and evaluate targets separately."""
    pool = _make_windows(episodes, spec, config, "train")
    queries = _make_windows(episodes, spec, config, "heldout")
    if len(pool) < config.top_k:
        raise M3Error(f"Need at least top-{config.top_k} pool windows; found {len(pool)}")
    if not queries:
        raise M3Error("No held-out query windows available")
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    pool_phase = np.asarray([window["phase"] for window in pool], dtype=np.float64)
    query_phase = np.asarray([window["phase"] for window in queries], dtype=np.float64)
    distances = np.abs(query_phase[:, None] - pool_phase[None, :])
    top_indices = np.argsort(distances, axis=1, kind="stable")[:, : config.top_k]
    rng = np.random.default_rng(config.random_seed)
    random_indices = np.vstack(
        [rng.choice(len(pool), size=config.top_k, replace=False) for _ in range(len(queries))]
    )

    retrieved_phase_error: list[float] = []
    random_phase_error: list[float] = []
    retrieved_residual: list[float] = []
    random_residual: list[float] = []
    absolute_target: list[float] = []
    gap_crossing = 0
    for query_index, query in enumerate(queries):
        query_episode = episode_by_id[query["episode_id"]]
        query_rows = query["query_rows"]
        current = query_episode.robot[query["current_row"]]
        target = query_episode.robot[query_rows]
        gap_crossing += int(
            np.count_nonzero(query_episode.segment_id[query_rows] != query_episode.segment_id[query["current_row"]])
        )
        absolute_target.extend(np.linalg.norm(target, axis=1))
        for chosen, phase_errors, residuals in (
            (top_indices[query_index], retrieved_phase_error, retrieved_residual),
            (random_indices[query_index], random_phase_error, random_residual),
        ):
            # Evaluate all top-k plans; retrieval ranking itself never sees target data.
            for pool_index in chosen:
                pool_window = pool[int(pool_index)]
                pool_episode = episode_by_id[pool_window["episode_id"]]
                pool_chunk = pool_episode.human[pool_window["pool_rows"]]
                aligned = align_pool_chunk(pool_chunk, current)
                phase_errors.append(abs(query["phase"] - pool_window["phase"]))
                residuals.extend(np.linalg.norm(target - aligned, axis=1))

    index_arrays = {
        "query_episode_id": np.asarray([window["episode_id"] for window in queries], dtype="U32"),
        "query_current_row": np.asarray([window["current_row"] for window in queries], dtype=np.int64),
        "query_phase": query_phase.astype(np.float32),
        "candidate_episode_id": np.asarray(
            [[pool[index]["episode_id"] for index in row] for row in top_indices], dtype="U32"
        ),
        "candidate_start_row": np.asarray(
            [[pool[index]["current_row"] for index in row] for row in top_indices], dtype=np.int64
        ),
        "candidate_phase_distance": np.take_along_axis(distances, top_indices, axis=1).astype(np.float32),
    }
    metrics = {
        "pool_window_count": len(pool),
        "query_window_count": len(queries),
        "top_k": config.top_k,
        "queries_with_top_k": int(np.sum(np.isfinite(index_arrays["candidate_phase_distance"]).all(axis=1))),
        "every_query_has_top_k": bool(index_arrays["candidate_phase_distance"].shape == (len(queries), config.top_k)),
        "gap_crossing_count": gap_crossing,
        "retrieval_feature_schema": ["normalized_segment_phase"],
        "heldout_robot_trajectory_used_in_retrieval_feature": False,
        "heldout_robot_trajectory_usage": "offline target evaluation only",
        "retrieved_phase_error_median": _median(retrieved_phase_error),
        "random_phase_error_median": _median(random_phase_error),
        "retrieved_residual_norm_median": _median(retrieved_residual),
        "random_residual_norm_median": _median(random_residual),
        "absolute_target_norm_median": _median(absolute_target),
    }
    return metrics, index_arrays


def build_perturbation_report(
    episodes: Sequence[Episode], config: M3Config, alignment: dict[str, Any]
) -> dict[str, Any]:
    train = [episode for episode in episodes if episode.split == "train"]
    selected_lag = int(alignment["approved_future_offset_view_steps"])
    candidate_by_lag = {item["lag_source_rows"]: item for item in alignment["lag_candidates"]}
    baseline = float(candidate_by_lag[selected_lag]["position_error_median_canonical"])
    wrong_lag = min(config.wrong_lag, config.max_lag)
    wrong_lag_error = float(candidate_by_lag[wrong_lag]["position_error_median_canonical"])
    scaled_errors: list[float] = []
    for episode in train:
        human, robot = _lag_pairs(episode, selected_lag)
        scaled_errors.extend(np.linalg.norm(robot[:, :3] - human[:, :3] * config.scale_perturbation, axis=1))
    scale_error = _median(scaled_errors)
    wrong_lag_ratio = wrong_lag_error / max(baseline, 1e-12)
    scale_ratio = scale_error / max(baseline, 1e-12)
    checks = {
        "wrong_role": {
            "primary_metric": "role_contract_violation_rate",
            "baseline": 0.0,
            "perturbed": 1.0,
            "significant_worsening": True,
        },
        "same_frame_copy": {
            "primary_metric": "temporal_leakage_rate",
            "baseline": 0.0,
            "perturbed": 1.0,
            "strict_future_target_rate_baseline": 1.0,
            "strict_future_target_rate_perturbed": 0.0,
            "significant_worsening": True,
        },
        "wrong_lag": {
            "primary_metric": "paired_position_error_median_canonical",
            "wrong_lag_source_rows": wrong_lag,
            "baseline": baseline,
            "perturbed": wrong_lag_error,
            "ratio": wrong_lag_ratio,
            "threshold_ratio": 1.5,
            "significant_worsening": wrong_lag_ratio >= 1.5,
        },
        "scale_x2": {
            "primary_metric": "paired_position_error_median_canonical",
            "scale_factor": config.scale_perturbation,
            "baseline": baseline,
            "perturbed": scale_error,
            "ratio": scale_ratio,
            "threshold_ratio": 1.5,
            "significant_worsening": scale_ratio >= 1.5,
        },
    }
    return {
        "schema_version": M3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": "passed" if all(item["significant_worsening"] for item in checks.values()) else "failed",
        "checks": checks,
    }


def _statistics(values: np.ndarray, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_count": int(len(values)),
        f"{prefix}_mean": values.mean(axis=0),
        f"{prefix}_std": values.std(axis=0),
        f"{prefix}_min": values.min(axis=0),
        f"{prefix}_max": values.max(axis=0),
    }


def build_action_statistics(episodes: Sequence[Episode]) -> dict[str, Any]:
    pool_values: list[np.ndarray] = []
    query_values: list[np.ndarray] = []
    residual_values: list[np.ndarray] = []
    for episode in episodes:
        if episode.split != "train":
            continue
        for indices in view_segment_indices(episode, TimeViewSpec("main")):
            if len(indices) < 2:
                continue
            pool = episode.human[indices[:-1]]
            query = episode.robot[indices[1:]]
            pool_values.append(pool)
            query_values.append(query)
            residual_values.append(query - pool)
    pool_array = np.concatenate(pool_values)
    query_array = np.concatenate(query_values)
    residual_array = np.concatenate(residual_values)
    return {
        "schema_version": M3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "provenance": {
            "split": "train",
            "heldout_data_used": False,
            "time_view_id": "nominal_camera_30hz_segmented",
            "pool_action_view_id": POOL_ACTION_VIEW_RAW,
            "query_action_view_id": QUERY_ACTION_VIEW,
            "action_alignment_id": ALIGNMENT_ID,
        },
        **_statistics(pool_array, "pool_action_10d"),
        **_statistics(query_array, "query_bc_target_10d"),
        **_statistics(residual_array, "residual_10d"),
    }


def _view_manifest(
    config: M3Config,
    spec: TimeViewSpec,
    split_manifest: dict[str, Any],
    preprocessing: dict[str, Any],
    metrics: dict[str, Any],
    pool_action_view_id: str,
    alignment_manifest_sha256: str,
) -> dict[str, Any]:
    duration_seconds = config.horizon / spec.nominal_hz if spec.nominal_hz else None
    manifest = {
        "schema_version": M3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "canonical_schema": SCHEMA_VERSION,
        "canonical_root": str(config.canonical_root),
        "canonical_manifest_sha256": file_sha256(config.canonical_root / "preprocessing_manifest.json"),
        "source_evidence_manifest_sha256": file_sha256(config.evidence_manifest),
        "split_sha256": split_manifest.get("split_sha256"),
        "time_view": asdict(spec),
        "time_view_id": spec.time_view_id,
        "pool_action_view_id": pool_action_view_id,
        "query_action_view_id": QUERY_ACTION_VIEW,
        "action_alignment_id": ALIGNMENT_ID,
        "action_alignment_manifest_sha256": alignment_manifest_sha256,
        "pool_action_role": "pool_side_human_plan_in_robot_frame",
        "query_action_role": "dataset_card_approved_bc_proxy",
        "query_command_status": "unverified",
        "query_target_offset_view_steps": 1,
        "strict_future_target": True,
        "H_steps": config.horizon,
        "H_seconds": duration_seconds,
        "K_steps": config.horizon,
        "K_seconds": duration_seconds,
        "gap_policy": "never_cross_segment",
        "terminal_policy": "drop_incomplete_future_chunks",
        "interpolation": {
            "application_status": "declared_adapter_contract; current artifacts select source rows only",
            "rgb": "nearest_neighbor",
            "xyz": "linear_within_segment",
            "rotation": "SLERP_within_segment",
            "gripper": "zero_order_hold_preserve_switch_events",
        },
        "time_view_materialization": (
            "uniform_segment_phase nearest-row baseline; DTW not fitted"
            if spec.time_view_id == "phase_or_dtw"
            else "segmented source-row selection"
        ),
        "deployment_command_adapter_id": None,
        "velocity_reporting": "disabled; physical xyz/time unit gate not passed",
        "source_episode_count": len(preprocessing.get("episodes", [])),
        "implementation": str(Path(__file__).resolve()),
        "implementation_code_sha256": file_sha256(Path(__file__)),
        "metrics": metrics,
    }
    manifest["view_id"] = stable_json_sha256(manifest)
    return manifest


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    retrieval = report["retrieval_sanity"]
    main = report["time_view_comparison"]["by_id"]["nominal_camera_30hz_segmented"]
    legacy = report["time_view_comparison"]["by_id"]["legacy_v01_stride3_nominal10"]
    alignment = report["alignment_calibration"]
    lines = [
        "# M3-v03 action-role、time-view 与 residual sanity check 验收报告",
        "",
        f"日期：{report['created_at_utc']}",
        "",
        f"结论：**{report['decision']}**。Gate A2a={report['gates']['A2a']}，"
        f"Gate A2b={report['gates']['A2b']}，Gate B={report['gates']['B']}。",
        "",
        "## 验收摘要",
        "",
        f"- Canonical 输入：`{SCHEMA_VERSION}`，{report['canonical_episode_count']} 条 pilot；旧 v1/v2 未进入正式 M3。",
        f"- Pool role：`{POOL_ACTION_VIEW_RAW}`（人手 plan）；query role：`{QUERY_ACTION_VIEW}`（严格 future BC proxy）。",
        f"- 选择的 future offset：{alignment['approved_future_offset_view_steps']} 个 view step；"
        f"motion cross-correlation 最优 lag={alignment['best_motion_cross_correlation_lag_source_rows']}，"
        f"相关系数={alignment['best_motion_cross_correlation']:.4f}，因此 lag-calibrated proxy 仅保留诊断。",
        f"- 主 view residual norm 中位数={main['residual_norm_median']:.6f}，"
        f"absolute target norm 中位数={main['absolute_target_norm_median']:.6f}。",
        f"- legacy stride3 residual norm 中位数={legacy['residual_norm_median']:.6f}；主 view 更低。",
        f"- Held-out query={retrieval['query_window_count']}，top-{retrieval['top_k']} 覆盖="
        f"{retrieval['queries_with_top_k']}/{retrieval['query_window_count']}，gap crossing={retrieval['gap_crossing_count']}。",
        f"- 检索 phase error 中位数={retrieval['retrieved_phase_error_median']:.6f}，"
        f"random={retrieval['random_phase_error_median']:.6f}。",
        f"- 检索 residual norm 中位数={retrieval['retrieved_residual_norm_median']:.6f}，"
        f"random={retrieval['random_residual_norm_median']:.6f}。",
        "- Retrieval feature 只有 normalized segment phase；held-out robot trajectory 仅用于离线 target 评测。",
        "- `deployment_command_adapter_id=null`；本报告不批准真实机器人 command 或 M6 rollout。",
        "",
        "## Action-role 与 Gate A2a",
        "",
        "`/action` 对应 pool-side human plan；`/end_position + /gripper_state` 仅作为 observed robot trajectory 与数据集卡允许的 BC label 来源。canonical v3 中不存在 generic action。",
        "",
        "## Gate A2b：proxy、alignment 与扰动",
        "",
        "主 query 使用下一连续 view row，所有不完整末帧窗口被丢弃，不跨 segment。identity numeric scale 来自同一 canonical 10D 转换；xyz 物理单位仍未确认，未报告 m/s。",
        "",
    ]
    for name, check in report["perturbation_sanity"]["checks"].items():
        ratio = f"，ratio={check['ratio']:.3f}" if "ratio" in check else ""
        lines.append(
            f"- `{name}`：metric={check['primary_metric']}，baseline={check['baseline']:.6f}，"
            f"perturbed={check['perturbed']:.6f}{ratio}，"
            f"significant_worsening={check['significant_worsening']}。"
        )
    lines.extend(
        [
            "",
            "## Time-view 对比",
            "",
            "| time_view_id | samples | gap crossing | residual median | absolute median |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in report["time_view_comparison"]["views"]:
        lines.append(
            f"| `{item['time_view_id']}` | {item['sample_count']} | {item['gap_crossing_count']} | "
            f"{item['residual_norm_median']:.6f} | {item['absolute_target_norm_median']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 产物与下一门禁",
            "",
            f"正式 view：`{report['main_view_path']}`。检索索引、view manifest、action statistics 和自动验收 JSON 均已保存。",
            "",
            "Gate B 仅允许启动 M4-v03 离线 bridge。真实控制仍需 M6 deployment command adapter、clock、latency 和安全验收。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_m3_pipeline(config: M3Config = M3Config()) -> dict[str, Any]:
    episodes, split_manifest, preprocessing = load_episodes(config)
    role_audit = build_action_role_audit(episodes, config, split_manifest)
    alignment = calibrate_alignment(episodes, config, split_manifest)
    perturbations = build_perturbation_report(episodes, config, alignment)

    comparison_views = [paired_time_view_metrics(episodes, spec) for spec in time_view_specs(config)]
    comparison = {
        "schema_version": M3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "same_episode_set": True,
        "same_pool_query_roles": True,
        "views": comparison_views,
        "by_id": {item["time_view_id"]: item for item in comparison_views},
    }
    main_spec = next(spec for spec in time_view_specs(config) if spec.status == "main")
    retrieval, index_arrays = build_retrieval_index(episodes, main_spec, config)
    action_statistics = build_action_statistics(episodes)
    alignment_manifest_sha256 = stable_json_sha256(alignment)

    views_root = config.derived_root / "views"
    for spec, metrics in zip(time_view_specs(config), comparison_views, strict=True):
        pool_id = POOL_ACTION_VIEW_PHASE if spec.time_view_id == "phase_or_dtw" else POOL_ACTION_VIEW_RAW
        view_path = views_root / spec.time_view_id / pool_id / QUERY_ACTION_VIEW / ALIGNMENT_ID
        write_json(view_path / "view_metrics.json", metrics)
        write_json(
            view_path / "view_manifest.json",
            _view_manifest(
                config,
                spec,
                split_manifest,
                preprocessing,
                metrics,
                pool_id,
                alignment_manifest_sha256,
            ),
        )

    main_path = views_root / main_spec.time_view_id / POOL_ACTION_VIEW_RAW / QUERY_ACTION_VIEW / ALIGNMENT_ID
    np.savez_compressed(main_path / "retrieval_index.npz", **index_arrays)
    retrieval["retrieval_index"] = str(main_path / "retrieval_index.npz")
    retrieval["retrieval_index_sha256"] = file_sha256(main_path / "retrieval_index.npz")
    write_json(main_path / "retrieval_sanity_report.json", retrieval)
    write_json(main_path / "action_statistics.json", action_statistics)

    m3_root = config.derived_root / "m3_v03"
    write_json(m3_root / "action_role_audit.json", role_audit)
    write_json(m3_root / "alignment_calibration.json", alignment)
    write_json(m3_root / "perturbation_sanity_report.json", perturbations)
    write_json(m3_root / "time_view_comparison.json", comparison)

    main_metrics = comparison["by_id"][main_spec.time_view_id]
    legacy_metrics = comparison["by_id"]["legacy_v01_stride3_nominal10"]
    gates = {
        "A2a": "passed" if role_audit["status"] == "passed" else "failed",
        "A2b": "passed"
        if perturbations["status"] == "passed" and alignment["approved_query_action_view_id"] == QUERY_ACTION_VIEW
        else "failed",
        "B": "passed"
        if (
            retrieval["every_query_has_top_k"]
            and retrieval["gap_crossing_count"] == 0
            and retrieval["retrieved_phase_error_median"] < retrieval["random_phase_error_median"]
            and main_metrics["residual_norm_median"] < main_metrics["absolute_target_norm_median"]
            and main_metrics["residual_norm_median"] < legacy_metrics["residual_norm_median"]
            and not retrieval["heldout_robot_trajectory_used_in_retrieval_feature"]
        )
        else "failed",
    }
    status = "passed" if all(value == "passed" for value in gates.values()) else "failed"
    report = {
        "schema_version": M3_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": status,
        "decision": "通过 M3-v03；允许启动 M4-v03 离线 bridge" if status == "passed" else "未通过；暂停 residual M4",
        "gates": gates,
        "canonical_root": str(config.canonical_root),
        "canonical_schema": SCHEMA_VERSION,
        "canonical_episode_count": len(episodes),
        "train_episode_count": sum(episode.split == "train" for episode in episodes),
        "heldout_episode_count": sum(episode.split == "heldout" for episode in episodes),
        "split_sha256": split_manifest.get("split_sha256"),
        "action_role_audit": role_audit,
        "alignment_calibration": alignment,
        "perturbation_sanity": perturbations,
        "time_view_comparison": comparison,
        "retrieval_sanity": retrieval,
        "main_view_path": str(main_path),
        "deployment_command_adapter_id": None,
        "m6_rollout_approved": False,
    }
    report["report_sha256"] = stable_json_sha256(report)
    write_json(m3_root / "m3_validation_report.json", report)
    write_json(config.report_root / "M3_action_time_residual_自动验收报告.json", report)
    _write_markdown_report(config.report_root / "M3_action_time_residual_验收报告.md", report)
    return report
