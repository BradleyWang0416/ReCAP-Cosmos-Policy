#!/usr/bin/env python3
"""Human2Robot M2-v03 semantic-safe native canonical conversion and validation.

The canonical layer preserves every source frame and every source time field.  It
does not invent a sampling rate.  Fixed-stride conversion remains available only
through the explicitly named legacy policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

SCHEMA_VERSION = "human2robot-canonical-hdf5-v3"
DEFAULT_SOURCE_ROOT = Path("/DATA1/wxs/DATASETS/Human2Robot/data/v1")
DEFAULT_OUTPUT_ROOT = Path("data/Human2Robot/canonical/v3")
DEFAULT_REPORT_ROOT = Path("方案/v03")
DEFAULT_SELECTION_MANIFEST = Path("data/Human2Robot/canonical/v1/preprocessing_manifest.json")
DEFAULT_EVIDENCE_MANIFEST = Path("方案/v03/source_evidence_manifest_v3.json")
DEFAULT_PARENT_V2_SPLIT = Path("data/Human2Robot/canonical/v2/task_split_manifest.json")
DEFAULT_V1_ROOT = Path("data/Human2Robot/canonical/v1")
DEFAULT_V1_REPORT = Path("方案/v01/M2_Human2Robot_canonical_HDF5_验收报告.md")
DEFAULT_V2_ROOT = Path("data/Human2Robot/canonical/v2")
DEFAULT_V2_REPORT = Path("方案/v02/M2_Human2Robot_native_time_验收报告.md")
DEMO_KEY = "data/demo_0"
STATE_DIM = 10
PRESERVE_NATIVE = "preserve_native"
LEGACY_FIXED_STRIDE3 = "legacy_fixed_stride3_assumed30"
TIMEBASE_STATUSES = {"trusted", "coarse", "discontinuous", "unknown"}

REQUIRED_SOURCE_DATASETS = {
    "action": (2, 7),
    "cam_data/human_camera": (4, 3),
    "cam_data/robot_camera": (4, 3),
    "end_position": (2, 6),
    "gripper_state": (1, None),
    "qpos": (2, 7),
    "qvel": (2, 7),
    "step": (1, None),
    "timestamp": (1, None),
    "transformed_hand_coords": (3, 3),
    "transformed_hand_frames": (3, 3),
}


class M2Error(RuntimeError):
    """Raised when conversion or validation cannot safely continue."""


@dataclass(frozen=True)
class ConversionConfig:
    source_root: Path = DEFAULT_SOURCE_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    report_root: Path = DEFAULT_REPORT_ROOT
    pilot_subdir: str = "pilot"
    episode_count: int = 20
    timebase_policy: str = PRESERVE_NATIVE
    selection_manifest: Path | None = None
    evidence_manifest: Path = DEFAULT_EVIDENCE_MANIFEST
    parent_v2_split_manifest: Path = DEFAULT_PARENT_V2_SPLIT
    heldout_task_count: int = 4
    split_seed: int = 20260711
    position_scale: float = 0.001
    euler_order: str = "xyz"
    step_gap_threshold: int = 1
    timestamp_gap_threshold: int = 1
    overwrite: bool = False
    image_compression: str = "gzip"
    image_compression_level: int = 4
    legacy_v1_root: Path = DEFAULT_V1_ROOT
    legacy_v1_report: Path = DEFAULT_V1_REPORT
    legacy_v2_root: Path = DEFAULT_V2_ROOT
    legacy_v2_report: Path = DEFAULT_V2_REPORT


@dataclass(frozen=True)
class ValidationLimits:
    workspace_min_m: tuple[float, float, float] = (-1.0, -1.0, -0.25)
    workspace_max_m: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gripper_min: float = 0.0
    gripper_max: float = 1.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
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
        json.dumps(value, indent=2, ensure_ascii=False, default=to_jsonable) + "\n",
        encoding="utf-8",
    )


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=to_jsonable).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_canonical_v3(path: Path) -> None:
    """Hard-reject v1/v2 and any non-v3 file on formal v3 read paths."""
    with h5py.File(path, "r") as file:
        schema = str(file.attrs.get("schema_version", ""))
        demo = file.get(DEMO_KEY)
        demo_schema = str(demo.attrs.get("schema_version", "")) if isinstance(demo, h5py.Group) else ""
    if schema != SCHEMA_VERSION or demo_schema != SCHEMA_VERSION:
        raise M2Error(
            f"Formal M2-v03 readers require schema {SCHEMA_VERSION}; "
            f"got root={schema!r}, demo={demo_schema!r}: {path}"
        )


def _load_evidence_manifest(config: ConversionConfig) -> tuple[dict[str, Any], str]:
    path = config.evidence_manifest.resolve()
    if not path.is_file():
        raise M2Error(f"Missing M2-v03 evidence manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "nominal_camera_fps": 30.0,
        "nominal_camera_fps_status": "verified_upstream",
        "euler_unit": "degree",
        "euler_order": "XYZ",
        "euler_evidence_status": "verified_upstream",
        "gripper_open_value": 1,
        "gripper_closed_value": 0,
        "gripper_evidence_status": "verified_upstream",
        "source_action_role": "human_hand_pose_in_robot_frame",
        "source_action_role_status": "verified_upstream",
        "source_action_as_robot_command_status": "unknown",
        "xyz_source_unit_status": "unknown",
    }
    for name, expected in required.items():
        if payload.get(name) != expected:
            raise M2Error(f"Evidence manifest {name} must be {expected!r}, got {payload.get(name)!r}")
    sources = payload.get("sources", [])
    if not sources or any(not item.get("url") or not item.get("version") or not item.get("accessed_at") for item in sources):
        raise M2Error("Evidence manifest sources require URL, version, and accessed_at provenance")
    return payload, file_sha256(path)


def _episode_number(path: Path) -> tuple[int, str]:
    suffix = path.stem.rsplit("_", 1)[-1]
    return (int(suffix), path.name) if suffix.isdigit() else (math.inf, path.name)


def discover_source_episodes(source_root: Path) -> list[Path]:
    """Return a deterministic task-round-robin ordering of source episodes."""
    if not source_root.is_dir():
        raise M2Error(f"Human2Robot source root does not exist: {source_root}")
    by_task: dict[str, list[Path]] = defaultdict(list)
    for path in source_root.rglob("episode_*.hdf5"):
        if path.is_file():
            by_task[str(path.parent.relative_to(source_root))].append(path)
    if not by_task:
        raise M2Error(f"No episode_*.hdf5 files found under {source_root}")
    for paths in by_task.values():
        paths.sort(key=_episode_number)

    ordered: list[Path] = []
    task_names = sorted(by_task)
    for episode_index in range(max(len(paths) for paths in by_task.values())):
        for task in task_names:
            if episode_index < len(by_task[task]):
                ordered.append(by_task[task][episode_index])
    return ordered


def legacy_resample_indices(frame_count: int, source_fps: float = 30.0, target_fps: float = 10.0) -> np.ndarray:
    """Reproduce the withdrawn v01 fixed-rate view; never use it implicitly."""
    if frame_count < 2:
        raise M2Error(f"At least two source frames are required, got {frame_count}")
    if not math.isfinite(source_fps) or not math.isfinite(target_fps):
        raise M2Error("Source and target FPS must be finite")
    if source_fps <= 0 or target_fps <= 0:
        raise M2Error("Source and target FPS must be positive")
    if target_fps > source_fps:
        raise M2Error(f"Upsampling is not allowed for canonical data: source_fps={source_fps}, target_fps={target_fps}")
    duration_s = (frame_count - 1) / source_fps
    target_count = int(math.floor(duration_s * target_fps + 1e-9)) + 1
    source_positions = np.arange(target_count, dtype=np.float64) * source_fps / target_fps
    indices = np.rint(source_positions).astype(np.int64)
    indices = np.clip(indices, 0, frame_count - 1)
    if len(indices) < 2 or not bool(np.all(np.diff(indices) > 0)):
        raise M2Error("Resampling did not produce at least two strictly increasing source indices")
    return indices


def frame_indices(frame_count: int, policy: str) -> np.ndarray:
    if frame_count < 2:
        raise M2Error(f"At least two source frames are required, got {frame_count}")
    if policy == PRESERVE_NATIVE:
        return np.arange(frame_count, dtype=np.int64)
    if policy == LEGACY_FIXED_STRIDE3:
        return legacy_resample_indices(frame_count)
    raise M2Error(f"Unsupported --timebase-policy: {policy}")


def audit_timebase(
    source_step: np.ndarray,
    source_timestamp: np.ndarray,
    *,
    step_gap_threshold: int = 1,
    timestamp_gap_threshold: int = 1,
) -> dict[str, Any]:
    """Classify serialized-record continuity independently of nominal camera FPS."""
    step = np.asarray(source_step, dtype=np.int64).reshape(-1)
    timestamp = np.asarray(source_timestamp, dtype=np.int64).reshape(-1)
    if len(step) != len(timestamp) or len(step) < 2:
        raise M2Error(f"Timebase arrays must have the same length >=2, got {len(step)} and {len(timestamp)}")
    step_diff = np.diff(step)
    timestamp_diff = np.diff(timestamp)
    gap_mask = np.zeros(len(step), dtype=np.bool_)
    gap_mask[1:] = (
        (step_diff < 0)
        | (step_diff > step_gap_threshold)
        | (timestamp_diff < 0)
        | (timestamp_diff > timestamp_gap_threshold)
    )
    segment_id = np.cumsum(gap_mask, dtype=np.int32)
    step_rollbacks = int(np.count_nonzero(step_diff < 0))
    timestamp_rollbacks = int(np.count_nonzero(timestamp_diff < 0))
    gap_count = int(np.count_nonzero(gap_mask))
    if gap_count:
        status = "discontinuous"
    elif np.count_nonzero(timestamp_diff) == 0:
        status = "unknown"
    else:
        status = "coarse"
    return {
        "timebase_status": status,
        "evidence_level": "verified_local source fields; timestamp semantics partially unknown",
        "timestamp_resolution": "source integer field; semantics partially unknown",
        "nominal_camera_fps": 30.0,
        "frame_count": len(step),
        "step_repeat_count": int(np.count_nonzero(step_diff == 0)),
        "step_jump_count": int(np.count_nonzero(step_diff > step_gap_threshold)),
        "step_rollback_count": step_rollbacks,
        "step_delta_min": int(np.min(step_diff)),
        "step_delta_max": int(np.max(step_diff)),
        "timestamp_repeat_count": int(np.count_nonzero(timestamp_diff == 0)),
        "timestamp_jump_count": int(np.count_nonzero(timestamp_diff > timestamp_gap_threshold)),
        "timestamp_rollback_count": timestamp_rollbacks,
        "timestamp_delta_min": int(np.min(timestamp_diff)),
        "timestamp_delta_max": int(np.max(timestamp_diff)),
        "gap_count": gap_count,
        "segment_count": int(segment_id[-1]) + 1,
        "gap_rule": {
            "step": f"diff < 0 or diff > {step_gap_threshold}",
            "timestamp": f"diff < 0 or diff > {timestamp_gap_threshold}",
            "boundary_convention": "gap_mask[i] marks a boundary between i-1 and i; gap_mask[0] is false",
        },
        "gap_mask": gap_mask,
        "segment_id": segment_id,
    }


def poses_euler_to_10d(
    poses: np.ndarray,
    gripper: np.ndarray,
    *,
    position_scale: float,
    euler_order: str,
) -> np.ndarray:
    """Convert ``xyz + Euler(deg) + gripper`` into ``xyz(m) + rot6d + gripper``."""
    poses = np.asarray(poses, dtype=np.float64)
    gripper = np.asarray(gripper, dtype=np.float64).reshape(-1)
    if poses.ndim != 2 or poses.shape[1] != 6:
        raise M2Error(f"Expected pose shape (T, 6), got {poses.shape}")
    if len(poses) != len(gripper):
        raise M2Error(f"Pose/gripper length mismatch: {len(poses)} vs {len(gripper)}")
    if not np.isfinite(poses).all() or not np.isfinite(gripper).all():
        raise M2Error("Pose or gripper contains NaN/Inf")
    matrices = Rotation.from_euler(euler_order, poses[:, 3:6], degrees=True).as_matrix()
    rotation_6d = np.concatenate((matrices[:, :, 0], matrices[:, :, 1]), axis=1)
    result = np.concatenate((poses[:, :3] * position_scale, rotation_6d, gripper[:, None]), axis=1)
    return result.astype(np.float32)


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    rotation_6d = np.asarray(rotation_6d, dtype=np.float64)
    if rotation_6d.ndim != 2 or rotation_6d.shape[1] != 6:
        raise M2Error(f"Expected rotation 6D shape (T, 6), got {rotation_6d.shape}")
    first = rotation_6d[:, :3]
    second = rotation_6d[:, 3:6]
    first_norm = np.linalg.norm(first, axis=1, keepdims=True)
    first_unit = first / np.maximum(first_norm, 1e-12)
    second_orthogonal = second - np.sum(first_unit * second, axis=1, keepdims=True) * first_unit
    second_norm = np.linalg.norm(second_orthogonal, axis=1, keepdims=True)
    second_unit = second_orthogonal / np.maximum(second_norm, 1e-12)
    third_unit = np.cross(first_unit, second_unit)
    return np.stack((first_unit, second_unit, third_unit), axis=2)


def inspect_source_episode(source_path: Path) -> dict[str, Any]:
    """Validate source representation and report time quality without rejecting gaps."""
    with h5py.File(source_path, "r") as source:
        lengths: dict[str, int] = {}
        for name, (ndim, final_dim) in REQUIRED_SOURCE_DATASETS.items():
            dataset = source.get(name)
            if not isinstance(dataset, h5py.Dataset):
                raise M2Error(f"{source_path}: missing source dataset {name}")
            if dataset.ndim != ndim:
                raise M2Error(f"{source_path}: {name} ndim={dataset.ndim}, expected {ndim}")
            if final_dim is not None and dataset.shape[-1] != final_dim:
                raise M2Error(f"{source_path}: {name} final dim={dataset.shape[-1]}, expected {final_dim}")
            lengths[name] = int(dataset.shape[0])
        if len(set(lengths.values())) != 1:
            raise M2Error(f"{source_path}: inconsistent source time axes: {lengths}")
        frame_count = next(iter(lengths.values()))
        if frame_count < 2:
            raise M2Error(f"{source_path}: only {frame_count} source frame(s)")
        human_shape = tuple(source["cam_data/human_camera"].shape)
        robot_shape = tuple(source["cam_data/robot_camera"].shape)
        if human_shape != robot_shape:
            raise M2Error(f"{source_path}: paired camera shape mismatch: {human_shape} vs {robot_shape}")
        supported_image_dtypes = {np.dtype("uint8"), np.dtype("uint16")}
        if source["cam_data/human_camera"].dtype not in supported_image_dtypes:
            raise M2Error(f"{source_path}: human RGB must be uint8 or uint16")
        if source["cam_data/robot_camera"].dtype not in supported_image_dtypes:
            raise M2Error(f"{source_path}: robot RGB must be uint8 or uint16")
        step = np.asarray(source["step"])
        timestamp = np.asarray(source["timestamp"])
        if not np.isfinite(step).all() or not np.isfinite(timestamp).all():
            raise M2Error(f"{source_path}: step/timestamp contains NaN/Inf")
        if not np.all(np.equal(step, np.asarray(step, dtype=np.int64))):
            raise M2Error(f"{source_path}: source step is not integer-valued")
        if not np.all(np.equal(timestamp, np.asarray(timestamp, dtype=np.int64))):
            raise M2Error(f"{source_path}: source timestamp is not integer-valued")
        time_audit = audit_timebase(np.asarray(step, dtype=np.int64), np.asarray(timestamp, dtype=np.int64))
    return {
        "frame_count": frame_count,
        "camera_shape": list(human_shape),
        "time_audit": {key: value for key, value in time_audit.items() if key not in {"gap_mask", "segment_id"}},
    }


def _create_numeric_dataset(group: h5py.Group, name: str, data: np.ndarray) -> h5py.Dataset:
    chunk_length = min(max(1, len(data)), 256)
    chunks = (chunk_length, *data.shape[1:])
    return group.create_dataset(name, data=data, compression="gzip", compression_opts=4, shuffle=True, chunks=chunks)


def _copy_selected_images(
    source: h5py.Dataset,
    destination_group: h5py.Group,
    name: str,
    indices: np.ndarray,
    config: ConversionConfig,
) -> str:
    probe_offsets = np.linspace(0, len(indices) - 1, num=min(8, len(indices)), dtype=np.int64)
    probe = np.asarray(source[indices[probe_offsets]])
    if source.dtype == np.dtype("uint8"):
        conversion = "identity_uint8"
    elif int(np.max(probe)) <= 255:
        conversion = "uint16_container_cast_to_uint8"
    else:
        conversion = "uint16_full_range_divide_by_257"
    shape = (len(indices), *source.shape[1:])
    destination = destination_group.create_dataset(
        name,
        shape=shape,
        dtype=np.uint8,
        compression=config.image_compression,
        compression_opts=config.image_compression_level,
        shuffle=True,
        chunks=(1, *source.shape[1:]),
    )
    for start in range(0, len(indices), 64):
        stop = min(start + 64, len(indices))
        batch = np.asarray(source[indices[start:stop]])
        if conversion == "identity_uint8":
            normalized = batch
        elif conversion == "uint16_container_cast_to_uint8":
            if int(np.max(batch)) > 255:
                raise M2Error(
                    f"{source.name}: uint16 RGB scale is inconsistent; probe <=255 but a later batch exceeds 255"
                )
            normalized = batch.astype(np.uint8)
        else:
            normalized = np.clip(batch / 257.0, 0, 255).astype(np.uint8)
        destination[start:stop] = normalized
    return conversion


def _read_selected(dataset: h5py.Dataset, indices: np.ndarray, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    array = np.asarray(dataset[indices])
    return array.astype(dtype, copy=False) if dtype is not None else array


def convert_episode(
    source_path: Path,
    destination_path: Path,
    config: ConversionConfig,
    *,
    demo_index: int,
    task_split: str,
    split_manifest_hash: str,
) -> dict[str, Any]:
    """Convert one episode atomically without assigning an unresolved policy-action role."""
    source_path = source_path.resolve()
    source_root = config.source_root.resolve()
    try:
        source_relative = source_path.relative_to(source_root)
    except ValueError as exc:
        raise M2Error(f"Source episode is outside --source-root: {source_path}") from exc

    source_summary = inspect_source_episode(source_path)
    evidence, evidence_hash = _load_evidence_manifest(config)
    indices = frame_indices(source_summary["frame_count"], config.timebase_policy)
    conversion_fingerprint = {
        "schema_version": SCHEMA_VERSION,
        "timebase_policy": config.timebase_policy,
        "position_scale": config.position_scale,
        "euler_order": config.euler_order,
        "step_gap_threshold": config.step_gap_threshold,
        "timestamp_gap_threshold": config.timestamp_gap_threshold,
        "image_compression": config.image_compression,
        "image_compression_level": config.image_compression_level,
        "source_evidence_manifest_sha256": evidence_hash,
    }
    config_hash = stable_json_sha256(conversion_fingerprint)
    code_hash = file_sha256(Path(__file__).resolve())
    source_hash = file_sha256(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() and not config.overwrite:
        with h5py.File(destination_path, "r") as existing:
            demo = existing.get(DEMO_KEY)
            if not isinstance(demo, h5py.Group):
                raise M2Error(f"Existing output is not canonical: {destination_path}")
            metadata = demo["metadata"]
            same_source = metadata.attrs.get("source_relative_path") == str(source_relative)
            same_schema = demo.attrs.get("schema_version") == SCHEMA_VERSION
            same_policy = metadata.attrs.get("timebase_policy") == config.timebase_policy
            same_config = metadata.attrs.get("conversion_config_sha256") == config_hash
            same_split = metadata.attrs.get("task_split") == task_split
            if not (same_source and same_schema and same_policy and same_config and same_split):
                raise M2Error(f"Existing output conflicts with requested conversion: {destination_path}")
        return {
            "status": "reused",
            "source_relative_path": str(source_relative),
            "output_path": str(destination_path),
            "source_frames": source_summary["frame_count"],
            "canonical_frames": len(indices),
            "task": str(source_relative.parent),
            "task_split": task_split,
            "source_sha256": source_hash,
            "time_audit": source_summary["time_audit"],
        }

    temporary_path = destination_path.with_suffix(destination_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        with h5py.File(source_path, "r") as source, h5py.File(temporary_path, "w") as destination:
            destination.attrs["schema_version"] = SCHEMA_VERSION
            destination.attrs["source_dataset"] = "Human2Robot v1"
            data_group = destination.create_group("data")
            demo = data_group.create_group("demo_0")
            demo.attrs["schema_version"] = SCHEMA_VERSION
            demo.attrs["demo_index"] = demo_index

            observations = demo.create_group("obs")
            robot_image_conversion = _copy_selected_images(
                source["cam_data/robot_camera"], observations, "robot_images", indices, config
            )
            robot_observed = poses_euler_to_10d(
                _read_selected(source["end_position"], indices),
                _read_selected(source["gripper_state"], indices),
                position_scale=config.position_scale,
                euler_order=config.euler_order,
            )
            actions_raw = _read_selected(source["action"], indices)
            human_hand_robot_frame = poses_euler_to_10d(
                actions_raw[:, :6],
                actions_raw[:, 6],
                position_scale=config.position_scale,
                euler_order=config.euler_order,
            )
            trajectories = demo.create_group("trajectories")
            robot_observed_dataset = _create_numeric_dataset(
                trajectories, "robot_ee_observed_10d", robot_observed
            )
            # An in-file hard link expresses two logical roles without duplicating values.
            observations["robot_state_10d"] = robot_observed_dataset
            _create_numeric_dataset(
                trajectories, "human_hand_robot_frame_10d", human_hand_robot_frame
            )

            metadata = demo.create_group("metadata")
            metadata.attrs["source_dataset"] = "Human2Robot v1"
            metadata.attrs["source_relative_path"] = str(source_relative)
            metadata.attrs["task"] = str(source_relative.parent)
            metadata.attrs["task_split"] = task_split
            metadata.attrs["split_manifest_sha256"] = split_manifest_hash
            metadata.attrs["timebase_policy"] = config.timebase_policy
            metadata.attrs["step_gap_threshold"] = config.step_gap_threshold
            metadata.attrs["timestamp_gap_threshold"] = config.timestamp_gap_threshold
            metadata.attrs["frame_selection"] = (
                "all_source_rows" if config.timebase_policy == PRESERVE_NATIVE else "fixed_stride3_legacy_only"
            )
            metadata.attrs["position_source_unit"] = "unknown; converter scale currently assumes millimetre"
            metadata.attrs["position_canonical_unit"] = "meter"
            metadata.attrs["position_unit_evidence_status"] = evidence["xyz_source_unit_status"]
            metadata.attrs["position_scale_to_canonical"] = config.position_scale
            metadata.attrs["euler_unit"] = evidence["euler_unit"]
            metadata.attrs["euler_order"] = evidence["euler_order"]
            metadata.attrs["euler_evidence_status"] = evidence["euler_evidence_status"]
            metadata.attrs["orientation_canonical"] = "rotation 6D: first two rotation-matrix columns"
            metadata.attrs["gripper_range"] = "[0, 1]"
            metadata.attrs["gripper_open_value"] = evidence["gripper_open_value"]
            metadata.attrs["gripper_closed_value"] = evidence["gripper_closed_value"]
            metadata.attrs["gripper_evidence_status"] = evidence["gripper_evidence_status"]
            metadata.attrs["source_action_role"] = evidence["source_action_role"]
            metadata.attrs["source_action_role_status"] = evidence["source_action_role_status"]
            metadata.attrs["source_action_as_robot_command_status"] = evidence[
                "source_action_as_robot_command_status"
            ]
            metadata.attrs["robot_trajectory_role"] = "observed_robot_ee_pose"
            metadata.attrs["generic_action_prohibited"] = True
            metadata.attrs["active_arm"] = "single"
            metadata.attrs["created_at_utc"] = utc_now()
            metadata.attrs["source_timestamp_note"] = "source values retained verbatim; not used to infer FPS"
            metadata.attrs["depth_note"] = "source paired depth is intentionally omitted from canonical pilot"
            metadata.attrs["robot_image_source_dtype"] = str(source["cam_data/robot_camera"].dtype)
            metadata.attrs["robot_image_conversion"] = robot_image_conversion
            metadata.attrs["source_sha256"] = source_hash
            metadata.attrs["conversion_code_sha256"] = code_hash
            metadata.attrs["conversion_config_sha256"] = config_hash
            metadata.attrs["source_evidence_manifest"] = str(config.evidence_manifest)
            metadata.attrs["source_evidence_manifest_sha256"] = evidence_hash
            metadata.attrs["evidence_accessed_at"] = evidence["accessed_at"]
            metadata.attrs["source_frame_count"] = source_summary["frame_count"]
            metadata.attrs["canonical_frame_count"] = len(indices)

            source_step = _read_selected(source["step"], indices, np.int64)
            source_timestamp = _read_selected(source["timestamp"], indices, np.int64)
            time_audit = audit_timebase(
                source_step,
                source_timestamp,
                step_gap_threshold=config.step_gap_threshold,
                timestamp_gap_threshold=config.timestamp_gap_threshold,
            )
            metadata.attrs["record_timebase_status"] = time_audit["timebase_status"]
            metadata.attrs["timestamp_resolution"] = time_audit["timestamp_resolution"]
            metadata.attrs["time_evidence_level"] = time_audit["evidence_level"]
            metadata.attrs["nominal_camera_fps"] = evidence["nominal_camera_fps"]
            metadata.attrs["nominal_camera_fps_status"] = evidence["nominal_camera_fps_status"]
            metadata.attrs["nominal_camera_fps_source"] = evidence["nominal_camera_fps_source"]
            metadata.attrs["nominal_camera_fps_source_url"] = evidence["nominal_camera_fps_source_url"]
            metadata.attrs["record_timebase_globally_trusted"] = False
            metadata.attrs["time_audit_json"] = json.dumps(
                {key: value for key, value in time_audit.items() if key not in {"gap_mask", "segment_id"}},
                sort_keys=True,
                default=to_jsonable,
            )
            if config.timebase_policy == LEGACY_FIXED_STRIDE3:
                metadata.attrs["legacy_warning"] = (
                    "withdrawn M2-v01 assumption; synthetic 10 Hz timeline is not native-time evidence"
                )
                _create_numeric_dataset(metadata, "timestamps", np.arange(len(indices), dtype=np.float64) / 10.0)
            _create_numeric_dataset(metadata, "source_indices", indices.astype(np.int64, copy=False))
            _create_numeric_dataset(metadata, "source_step", source_step)
            _create_numeric_dataset(metadata, "source_timestamp", source_timestamp)
            _create_numeric_dataset(metadata, "segment_id", time_audit["segment_id"].astype(np.int32, copy=False))
            _create_numeric_dataset(metadata, "gap_mask", time_audit["gap_mask"].astype(np.bool_, copy=False))
            _create_numeric_dataset(metadata, "qpos_raw", _read_selected(source["qpos"], indices, np.float32))
            _create_numeric_dataset(metadata, "qvel_raw", _read_selected(source["qvel"], indices, np.float32))
            _create_numeric_dataset(
                metadata,
                "end_position_raw",
                _read_selected(source["end_position"], indices, np.float32),
            )
            _create_numeric_dataset(metadata, "action_raw", actions_raw.astype(np.float32, copy=False))
            _create_numeric_dataset(
                metadata,
                "gripper_state_raw",
                _read_selected(source["gripper_state"], indices, np.float32),
            )

            human = metadata.create_group("human")
            human_image_conversion = _copy_selected_images(
                source["cam_data/human_camera"], human, "images", indices, config
            )
            metadata.attrs["human_image_source_dtype"] = str(source["cam_data/human_camera"].dtype)
            metadata.attrs["human_image_conversion"] = human_image_conversion
            _create_numeric_dataset(
                human,
                "hand_coords",
                _read_selected(source["transformed_hand_coords"], indices, np.float32),
            )
            _create_numeric_dataset(
                human,
                "hand_frames",
                _read_selected(source["transformed_hand_frames"], indices, np.float32),
            )
            destination.flush()
        os.replace(temporary_path, destination_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    return {
        "status": "converted",
        "source_relative_path": str(source_relative),
        "output_path": str(destination_path),
        "source_frames": source_summary["frame_count"],
        "canonical_frames": len(indices),
        "task": str(source_relative.parent),
        "task_split": task_split,
        "source_sha256": source_hash,
        "time_audit": {key: value for key, value in time_audit.items() if key not in {"gap_mask", "segment_id"}},
    }


def _load_selection_candidates(config: ConversionConfig) -> tuple[list[Path], dict[str, Any]]:
    if config.selection_manifest is None:
        return discover_source_episodes(config.source_root), {
            "policy": "deterministic task-round-robin, then numeric episode index",
            "manifest": None,
            "manifest_sha256": None,
        }
    manifest_path = config.selection_manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_paths = [record["source_relative_path"] for record in payload.get("episodes", [])]
    if len(relative_paths) < config.episode_count:
        raise M2Error(
            f"Selection manifest has {len(relative_paths)} episode(s), requested {config.episode_count}: {manifest_path}"
        )
    return [config.source_root / relative for relative in relative_paths], {
        "policy": "frozen source episode list from selection manifest",
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }


def _build_task_split_manifest(
    selected: Sequence[Path],
    config: ConversionConfig,
    selection: dict[str, Any],
) -> dict[str, Any]:
    tasks = sorted({str(path.parent.relative_to(config.source_root)) for path in selected})
    if config.heldout_task_count < 0 or config.heldout_task_count >= len(tasks):
        raise M2Error(
            f"--heldout-task-count must be in [0, task_count), got {config.heldout_task_count} for {len(tasks)} tasks"
        )
    ranked = sorted(tasks, key=lambda task: hashlib.sha256(f"{config.split_seed}:{task}".encode()).hexdigest())
    heldout_tasks = sorted(ranked[: config.heldout_task_count])
    heldout = set(heldout_tasks)
    assignments = {task: ("heldout" if task in heldout else "train") for task in tasks}
    parent_path = config.parent_v2_split_manifest.resolve()
    if not parent_path.is_file():
        raise M2Error(f"Missing frozen v2 parent split manifest: {parent_path}")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_assignments = {task: parent.get("tasks", {}).get(task) for task in tasks}
    if parent_assignments != assignments:
        raise M2Error("Computed v3 task split differs from frozen v2 parent split")
    frozen_payload = {
        "schema_version": SCHEMA_VERSION,
        "split_seed": config.split_seed,
        "heldout_task_count": config.heldout_task_count,
        "parent_v2_split_sha256": file_sha256(parent_path),
        "parent_v2_split_id": parent.get("split_sha256"),
        "tasks": assignments,
        "episodes": [
            {
                "source_relative_path": str(path.relative_to(config.source_root)),
                "task": str(path.parent.relative_to(config.source_root)),
                "split": assignments[str(path.parent.relative_to(config.source_root))],
            }
            for path in selected
        ],
    }
    return {
        "status": "frozen",
        "created_at_utc": utc_now(),
        "assignment_policy": "task-level SHA-256 ranking of '<seed>:<task>'; first N tasks held out",
        "selection_provenance": selection,
        "split_sha256": stable_json_sha256(frozen_payload),
        **frozen_payload,
        "train_tasks": sorted(task for task, split in assignments.items() if split == "train"),
        "heldout_tasks": heldout_tasks,
    }


def convert_dataset(config: ConversionConfig) -> dict[str, Any]:
    if config.episode_count < 1:
        raise M2Error("--episodes must be at least 1")
    if config.timebase_policy not in {PRESERVE_NATIVE, LEGACY_FIXED_STRIDE3}:
        raise M2Error(f"Unsupported --timebase-policy: {config.timebase_policy}")
    candidates, selection = _load_selection_candidates(config)
    selected: list[Path] = []
    rejected: list[dict[str, str]] = []
    for candidate in candidates:
        if len(selected) >= config.episode_count:
            break
        try:
            inspect_source_episode(candidate)
        except (M2Error, OSError, ValueError) as exc:
            rejected.append({"source_path": str(candidate), "error": str(exc)})
            continue
        selected.append(candidate)
    if len(selected) < config.episode_count:
        raise M2Error(
            f"Only {len(selected)} valid source episodes were selected; requested {config.episode_count}. "
            f"Rejected {len(rejected)} candidate(s)."
        )

    split_manifest = _build_task_split_manifest(selected, config, selection)
    split_manifest_path = config.output_root / "task_split_manifest.json"
    write_json(split_manifest_path, split_manifest)
    if config.timebase_policy == PRESERVE_NATIVE:
        write_json(config.report_root / "human2robot_task_split_manifest_v3.json", split_manifest)

    split_dir = config.output_root / config.pilot_subdir
    converted: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        task = str(candidate.parent.relative_to(config.source_root))
        destination = split_dir / f"demo_{index:05d}.hdf5"
        record = convert_episode(
            candidate,
            destination,
            config,
            demo_index=index,
            task_split=split_manifest["tasks"][task],
            split_manifest_hash=split_manifest["split_sha256"],
        )
        converted.append(record)

    manifest = {
        "status": "converted",
        "created_at_utc": utc_now(),
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "selection": selection,
        "split_manifest": str(split_manifest_path),
        "split_sha256": split_manifest["split_sha256"],
        "episodes": converted,
        "rejected_candidates": rejected,
        "evidence_status": {
            "nominal_camera_fps": "30.0 verified_upstream; not a globally trusted record clock",
            "position_unit": "unknown; numeric conversion retains the documented assumption",
            "euler_order": "degree XYZ verified_upstream",
            "source_action_role": "human_hand_pose_in_robot_frame verified_upstream",
            "source_action_as_robot_command": "unknown",
            "gripper_polarity": "1=open, 0=close verified_upstream",
        },
    }
    write_json(config.output_root / "preprocessing_manifest.json", manifest)
    timebase_report = {
        "status": "audited",
        "created_at_utc": utc_now(),
        "schema_version": SCHEMA_VERSION,
        "timebase_policy": config.timebase_policy,
        "episode_count": len(converted),
        "episodes": [
            {
                "source_relative_path": record["source_relative_path"],
                "task": record["task"],
                **record["time_audit"],
            }
            for record in converted
        ],
    }
    write_json(config.output_root / "timebase_audit_report.json", timebase_report)
    if config.timebase_policy == PRESERVE_NATIVE:
        write_json(config.report_root / "human2robot_timebase_audit_v3.json", timebase_report)
    return manifest


def _array_statistics(array: np.ndarray, prefix: str) -> dict[str, list[float]]:
    return {
        f"{prefix}_min": np.min(array, axis=0).tolist(),
        f"{prefix}_max": np.max(array, axis=0).tolist(),
        f"{prefix}_mean": np.mean(array, axis=0).tolist(),
        f"{prefix}_std": np.std(array, axis=0).tolist(),
        f"{prefix}_median": np.median(array, axis=0).tolist(),
    }


def _minmax_normalize(array: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    span = maximum - minimum
    normalized = np.zeros_like(array, dtype=np.float64)
    nonconstant = span > 1e-12
    normalized[:, nonconstant] = 2.0 * ((array[:, nonconstant] - minimum[nonconstant]) / span[nonconstant]) - 1.0
    return normalized


def compute_and_write_statistics(
    files: Sequence[Path],
    output_root: Path,
    *,
    required_split: str,
    split_manifest_hash: str,
) -> dict[str, Any]:
    if not files:
        raise M2Error("No canonical files supplied for statistics")
    robot_arrays: list[np.ndarray] = []
    human_arrays: list[np.ndarray] = []
    tasks: set[str] = set()
    quality_episodes: list[dict[str, Any]] = []
    contiguous_robot_displacements: list[np.ndarray] = []
    contiguous_human_displacements: list[np.ndarray] = []
    evidence_hashes: set[str] = set()
    for path in files:
        require_canonical_v3(path)
        with h5py.File(path, "r") as file:
            demo = file[DEMO_KEY]
            metadata = demo["metadata"]
            task_split = str(metadata.attrs.get("task_split", ""))
            if task_split != required_split:
                raise M2Error(f"Statistics received {task_split!r} episode, expected {required_split!r}: {path}")
            if metadata.attrs.get("split_manifest_sha256") != split_manifest_hash:
                raise M2Error(f"Statistics split hash mismatch: {path}")
            tasks.add(str(metadata.attrs["task"]))
            evidence_hashes.add(str(metadata.attrs["source_evidence_manifest_sha256"]))
            robot_for_episode = np.asarray(
                demo["trajectories/robot_ee_observed_10d"], dtype=np.float64
            )
            human_for_episode = np.asarray(
                demo["trajectories/human_hand_robot_frame_10d"], dtype=np.float64
            )
            robot_arrays.append(robot_for_episode)
            human_arrays.append(human_for_episode)
            gap_mask = np.asarray(metadata["gap_mask"], dtype=np.bool_)
            contiguous = ~gap_mask[1:]
            if np.any(contiguous):
                contiguous_robot_displacements.append(
                    np.linalg.norm(np.diff(robot_for_episode[:, :3], axis=0)[contiguous], axis=1)
                )
                contiguous_human_displacements.append(
                    np.linalg.norm(np.diff(human_for_episode[:, :3], axis=0)[contiguous], axis=1)
                )
            audit = json.loads(str(metadata.attrs["time_audit_json"]))
            quality_episodes.append(
                {
                    "path": str(path),
                    "task": str(metadata.attrs["task"]),
                    "frame_count": len(robot_for_episode),
                    **audit,
                }
            )
    robot = np.concatenate(robot_arrays, axis=0)
    human = np.concatenate(human_arrays, axis=0)
    if robot.shape[1] != STATE_DIM or human.shape[1] != STATE_DIM:
        raise M2Error(f"Statistics require role-specific 10D trajectories, got {robot.shape} and {human.shape}")
    if not np.isfinite(robot).all() or not np.isfinite(human).all():
        raise M2Error("Cannot compute statistics over NaN/Inf")
    if len(evidence_hashes) != 1:
        raise M2Error(f"Statistics require one evidence manifest hash, got {sorted(evidence_hashes)}")

    provenance = {
        "source_split": required_split,
        "split_manifest_sha256": split_manifest_hash,
        "episode_count": len(files),
        "task_count": len(tasks),
        "tasks": sorted(tasks),
        "frame_count": int(len(robot)),
        "heldout_data_used": False,
        "source_evidence_manifest_sha256": next(iter(evidence_hashes)),
        "generic_action_used": False,
    }
    robot_statistics = {
        **_array_statistics(robot, "robot_ee_observed_10d"),
        "_provenance": {**provenance, "role": "observed_robot_ee_pose"},
    }
    human_statistics = {
        **_array_statistics(human, "human_hand_robot_frame_10d"),
        "_provenance": {**provenance, "role": "human_hand_pose_in_robot_frame"},
    }
    write_json(output_root / "robot_observed_statistics.json", robot_statistics)
    write_json(output_root / "human_hand_robot_frame_statistics.json", human_statistics)

    normalized_robot = _minmax_normalize(
        robot,
        np.asarray(robot_statistics["robot_ee_observed_10d_min"]),
        np.asarray(robot_statistics["robot_ee_observed_10d_max"]),
    )
    normalized_human = _minmax_normalize(
        human,
        np.asarray(human_statistics["human_hand_robot_frame_10d_min"]),
        np.asarray(human_statistics["human_hand_robot_frame_10d_max"]),
    )
    post_norm = {
        **_array_statistics(normalized_robot, "robot_ee_observed_10d"),
        **_array_statistics(normalized_human, "human_hand_robot_frame_10d"),
        "_provenance": {
            **provenance,
            "roles": ["observed_robot_ee_pose", "human_hand_pose_in_robot_frame"],
        },
    }
    write_json(output_root / "dataset_statistics_post_norm_by_role.json", post_norm)

    robot_displacements = (
        np.concatenate(contiguous_robot_displacements) if contiguous_robot_displacements else np.asarray([])
    )
    human_displacements = (
        np.concatenate(contiguous_human_displacements) if contiguous_human_displacements else np.asarray([])
    )
    status_counts = {
        status: sum(record["timebase_status"] == status for record in quality_episodes)
        for status in sorted(TIMEBASE_STATUSES)
    }
    data_quality = {
        "status": "audited",
        "created_at_utc": utc_now(),
        "schema_version": SCHEMA_VERSION,
        "_provenance": provenance,
        "timebase_status_counts": status_counts,
        "gap_count": sum(int(record["gap_count"]) for record in quality_episodes),
        "segment_count": sum(int(record["segment_count"]) for record in quality_episodes),
        "step_repeat_count": sum(int(record["step_repeat_count"]) for record in quality_episodes),
        "step_jump_count": sum(int(record["step_jump_count"]) for record in quality_episodes),
        "step_rollback_count": sum(int(record["step_rollback_count"]) for record in quality_episodes),
        "timestamp_repeat_count": sum(int(record["timestamp_repeat_count"]) for record in quality_episodes),
        "timestamp_jump_count": sum(int(record["timestamp_jump_count"]) for record in quality_episodes),
        "timestamp_rollback_count": sum(int(record["timestamp_rollback_count"]) for record in quality_episodes),
        "contiguous_per_step_displacement": {
            "unit": "metres per retained source step (not m/s)",
            "human_hand_robot_frame_max": float(np.max(human_displacements)) if len(human_displacements) else None,
            "human_hand_robot_frame_p99": float(np.quantile(human_displacements, 0.99)) if len(human_displacements) else None,
            "robot_ee_observed_max": float(np.max(robot_displacements)) if len(robot_displacements) else None,
            "robot_ee_observed_p99": float(np.quantile(robot_displacements, 0.99)) if len(robot_displacements) else None,
        },
        "episodes": quality_episodes,
    }
    write_json(output_root / "data_quality_statistics.json", data_quality)
    return {
        "episode_count": len(files),
        "frame_count": int(len(robot)),
        "robot_observed_dim": int(robot.shape[1]),
        "human_hand_robot_frame_dim": int(human.shape[1]),
        "source_split": required_split,
        "split_manifest_sha256": split_manifest_hash,
        "normalization": "per-dimension min-max to [-1, 1]; constant dimensions map to 0",
        "files": [
            "robot_observed_statistics.json",
            "human_hand_robot_frame_statistics.json",
            "dataset_statistics_post_norm_by_role.json",
            "data_quality_statistics.json",
        ],
    }


def _check_rotation_6d(name: str, values: np.ndarray, errors: list[str]) -> None:
    first = values[:, 3:6]
    second = values[:, 6:9]
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    dot = np.sum(first * second, axis=1)
    if not np.allclose(first_norm, 1.0, atol=2e-4):
        errors.append(f"{name} rotation first column is not unit length")
    if not np.allclose(second_norm, 1.0, atol=2e-4):
        errors.append(f"{name} rotation second column is not unit length")
    if not np.allclose(dot, 0.0, atol=2e-4):
        errors.append(f"{name} rotation columns are not orthogonal")


def _per_step_motion_metrics(values: np.ndarray, gap_mask: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return 0.0, 0.0
    contiguous = ~np.asarray(gap_mask, dtype=np.bool_)[1:]
    if not np.any(contiguous):
        return 0.0, 0.0
    linear = np.linalg.norm(np.diff(values[:, :3], axis=0), axis=1)[contiguous]
    matrices = rotation_6d_to_matrix(values[:, 3:9])
    rotations = Rotation.from_matrix(matrices)
    angular = (rotations[:-1].inv() * rotations[1:]).magnitude()[contiguous]
    return float(np.max(linear)), float(np.max(angular))


def validate_canonical_episode(
    path: Path,
    limits: ValidationLimits,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    structure_errors: list[str] = []
    time_errors: list[str] = []
    evidence_errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    try:
        with h5py.File(path, "r") as file:
            demo = file.get(DEMO_KEY)
            if not isinstance(demo, h5py.Group):
                raise M2Error(f"Missing group {DEMO_KEY}")
            if demo.attrs.get("schema_version") != SCHEMA_VERSION:
                structure_errors.append(f"schema_version must be {SCHEMA_VERSION}")
            required_paths = [
                "obs/robot_images",
                "obs/robot_state_10d",
                "trajectories/robot_ee_observed_10d",
                "trajectories/human_hand_robot_frame_10d",
                "metadata/source_indices",
                "metadata/source_step",
                "metadata/source_timestamp",
                "metadata/segment_id",
                "metadata/gap_mask",
                "metadata/qpos_raw",
                "metadata/qvel_raw",
                "metadata/end_position_raw",
                "metadata/action_raw",
                "metadata/gripper_state_raw",
                "metadata/human/images",
                "metadata/human/hand_coords",
                "metadata/human/hand_frames",
            ]
            datasets: dict[str, h5py.Dataset] = {}
            for name in required_paths:
                item = demo.get(name)
                if not isinstance(item, h5py.Dataset):
                    structure_errors.append(f"Missing dataset {DEMO_KEY}/{name}")
                else:
                    datasets[name] = item
            if structure_errors:
                return {
                    "path": str(path),
                    "status": "failed",
                    "structure_validation": {"status": "failed", "errors": structure_errors},
                    "time_truth_validation": {"status": "not_run", "errors": []},
                    "evidence_role_validation": {"status": "not_run", "errors": []},
                    "errors": structure_errors,
                    "warnings": warnings,
                    "metrics": metrics,
                }

            lengths = {name: int(dataset.shape[0]) for name, dataset in datasets.items()}
            if len(set(lengths.values())) != 1:
                structure_errors.append(f"Time-axis mismatch: {lengths}")
            frame_count = lengths["trajectories/robot_ee_observed_10d"]
            if frame_count < 2:
                structure_errors.append(f"Episode must have at least 2 frames, got {frame_count}")
            metrics["frame_count"] = frame_count

            images = datasets["obs/robot_images"]
            human_images = datasets["metadata/human/images"]
            if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.dtype("uint8"):
                structure_errors.append(f"obs/robot_images must be uint8 (T,H,W,3), got {images.shape} {images.dtype}")
            if human_images.shape != images.shape or human_images.dtype != np.dtype("uint8"):
                structure_errors.append("metadata/human/images must match robot image shape and uint8 dtype")
            if images.compression is None or human_images.compression is None:
                structure_errors.append("Robot and human image datasets must be compressed")
            if frame_count >= 2:
                for name, dataset in (("obs/robot_images", images), ("metadata/human/images", human_images)):
                    sampled_frames = np.asarray(dataset[[0, frame_count - 1]])
                    if sampled_frames.shape != (2, *dataset.shape[1:]):
                        structure_errors.append(
                            f"{name} first/last frame read returned unexpected shape {sampled_frames.shape}"
                        )
                    metrics[f"{name}_sample_min"] = int(np.min(sampled_frames))
                    metrics[f"{name}_sample_max"] = int(np.max(sampled_frames))

            robot_observed = np.asarray(datasets["trajectories/robot_ee_observed_10d"], dtype=np.float64)
            human_hand = np.asarray(datasets["trajectories/human_hand_robot_frame_10d"], dtype=np.float64)
            for name, values in (
                ("obs/robot_state_10d", np.asarray(datasets["obs/robot_state_10d"], dtype=np.float64)),
                ("trajectories/robot_ee_observed_10d", robot_observed),
                ("trajectories/human_hand_robot_frame_10d", human_hand),
            ):
                if datasets[name].dtype != np.dtype("float32"):
                    structure_errors.append(f"{name} dtype must be float32, got {datasets[name].dtype}")
                if values.shape != (frame_count, STATE_DIM):
                    structure_errors.append(f"{name} must have shape (T,10), got {values.shape}")
                elif not np.isfinite(values).all():
                    structure_errors.append(f"{name} contains NaN/Inf")
                else:
                    _check_rotation_6d(name, values, structure_errors)
            if datasets["obs/robot_state_10d"].id != datasets["trajectories/robot_ee_observed_10d"].id:
                structure_errors.append("obs/robot_state_10d must be an in-file alias of robot_ee_observed_10d")
            if not np.array_equal(np.asarray(datasets["obs/robot_state_10d"]), robot_observed):
                structure_errors.append("robot observation alias values differ from observed trajectory")

            metadata = demo["metadata"]
            required_attrs = [
                "source_dataset",
                "source_relative_path",
                "task",
                "task_split",
                "split_manifest_sha256",
                "timebase_policy",
                "record_timebase_status",
                "timestamp_resolution",
                "nominal_camera_fps",
                "nominal_camera_fps_status",
                "nominal_camera_fps_source",
                "nominal_camera_fps_source_url",
                "record_timebase_globally_trusted",
                "frame_selection",
                "source_frame_count",
                "canonical_frame_count",
                "source_sha256",
                "conversion_code_sha256",
                "conversion_config_sha256",
                "time_audit_json",
                "position_canonical_unit",
                "position_unit_evidence_status",
                "position_scale_to_canonical",
                "euler_unit",
                "euler_order",
                "euler_evidence_status",
                "source_action_role",
                "source_action_role_status",
                "source_action_as_robot_command_status",
                "robot_trajectory_role",
                "generic_action_prohibited",
                "gripper_evidence_status",
                "gripper_open_value",
                "gripper_closed_value",
                "source_evidence_manifest",
                "source_evidence_manifest_sha256",
                "evidence_accessed_at",
                "robot_image_source_dtype",
                "robot_image_conversion",
                "human_image_source_dtype",
                "human_image_conversion",
            ]
            for name in required_attrs:
                if name not in metadata.attrs:
                    structure_errors.append(f"Missing metadata attribute {name}")
            if "actions" in demo or "policy_actions" in demo:
                evidence_errors.append("Canonical v3 must not contain generic actions or policy_actions")
            expected_evidence = {
                "nominal_camera_fps": 30.0,
                "nominal_camera_fps_status": "verified_upstream",
                "euler_unit": "degree",
                "euler_order": "XYZ",
                "euler_evidence_status": "verified_upstream",
                "gripper_open_value": 1,
                "gripper_closed_value": 0,
                "gripper_evidence_status": "verified_upstream",
                "source_action_role": "human_hand_pose_in_robot_frame",
                "source_action_role_status": "verified_upstream",
                "source_action_as_robot_command_status": "unknown",
                "robot_trajectory_role": "observed_robot_ee_pose",
                "position_unit_evidence_status": "unknown",
            }
            for name, expected in expected_evidence.items():
                if metadata.attrs.get(name) != expected:
                    evidence_errors.append(f"{name} must be {expected!r}, got {metadata.attrs.get(name)!r}")
            if not str(metadata.attrs.get("nominal_camera_fps_source", "")).startswith("arxiv:2502.16587v4"):
                evidence_errors.append("nominal_camera_fps_source must bind arXiv v4 Appendix A")
            if not str(metadata.attrs.get("nominal_camera_fps_source_url", "")).startswith("https://"):
                evidence_errors.append("nominal_camera_fps_source_url must contain upstream provenance")
            if len(str(metadata.attrs.get("source_evidence_manifest_sha256", ""))) != 64:
                evidence_errors.append("source_evidence_manifest_sha256 must be a SHA-256 digest")

            source_indices = np.asarray(datasets["metadata/source_indices"], dtype=np.int64)
            source_step = np.asarray(datasets["metadata/source_step"], dtype=np.int64)
            source_timestamp = np.asarray(datasets["metadata/source_timestamp"], dtype=np.int64)
            segment_id = np.asarray(datasets["metadata/segment_id"], dtype=np.int32)
            gap_mask = np.asarray(datasets["metadata/gap_mask"], dtype=np.bool_)
            expected_dtypes = {
                "metadata/source_indices": np.dtype("int64"),
                "metadata/source_step": np.dtype("int64"),
                "metadata/source_timestamp": np.dtype("int64"),
                "metadata/segment_id": np.dtype("int32"),
                "metadata/gap_mask": np.dtype("bool"),
            }
            for name, expected_dtype in expected_dtypes.items():
                if datasets[name].dtype != expected_dtype:
                    structure_errors.append(f"{name} dtype must be {expected_dtype}, got {datasets[name].dtype}")
            if not bool(np.all(np.diff(source_indices) > 0)):
                time_errors.append("metadata/source_indices is not strictly increasing")
            if str(metadata.attrs.get("timebase_policy", "")) != PRESERVE_NATIVE:
                time_errors.append("M2-v03 acceptance requires timebase_policy=preserve_native")
            if str(metadata.attrs.get("frame_selection", "")) != "all_source_rows":
                time_errors.append("M2-v03 acceptance requires frame_selection=all_source_rows")
            if not np.array_equal(source_indices, np.arange(frame_count, dtype=np.int64)):
                time_errors.append("source_indices must be exactly 0..T-1 for one-to-one native-frame mapping")
            if "timestamps" in metadata:
                time_errors.append("preserve_native output must not contain a synthetic metadata/timestamps dataset")
            if bool(metadata.attrs.get("record_timebase_globally_trusted", True)):
                time_errors.append("coarse/discontinuous canonical records must not be globally trusted")

            for name in [
                "metadata/qpos_raw",
                "metadata/qvel_raw",
                "metadata/end_position_raw",
                "metadata/action_raw",
                "metadata/gripper_state_raw",
                "metadata/human/hand_coords",
                "metadata/human/hand_frames",
            ]:
                values = np.asarray(datasets[name])
                if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
                    structure_errors.append(f"{name} contains non-numeric or non-finite values")
                if datasets[name].dtype != np.dtype("float32"):
                    structure_errors.append(f"{name} dtype must be float32, got {datasets[name].dtype}")

            time_audit = audit_timebase(
                source_step,
                source_timestamp,
                step_gap_threshold=int(metadata.attrs.get("step_gap_threshold", 1)),
                timestamp_gap_threshold=int(metadata.attrs.get("timestamp_gap_threshold", 1)),
            )
            if not np.array_equal(gap_mask, time_audit["gap_mask"]):
                time_errors.append("gap_mask does not match the declared step/timestamp gap rule")
            if not np.array_equal(segment_id, time_audit["segment_id"]):
                time_errors.append("segment_id does not match cumulative gap boundaries")
            if str(metadata.attrs.get("record_timebase_status", "")) not in TIMEBASE_STATUSES:
                time_errors.append(
                    f"Invalid record_timebase_status={metadata.attrs.get('record_timebase_status')!r}"
                )
            elif metadata.attrs.get("record_timebase_status") != time_audit["timebase_status"]:
                time_errors.append("record_timebase_status does not match the recomputed source time audit")
            stored_audit = json.loads(str(metadata.attrs.get("time_audit_json", "{}")))
            for name in (
                "step_repeat_count",
                "step_jump_count",
                "step_rollback_count",
                "timestamp_repeat_count",
                "timestamp_jump_count",
                "timestamp_rollback_count",
                "gap_count",
                "segment_count",
            ):
                if stored_audit.get(name) != time_audit[name]:
                    time_errors.append(f"Stored time audit {name} does not match recomputed value")

            source_frame_count = int(metadata.attrs.get("source_frame_count", -1))
            canonical_frame_count = int(metadata.attrs.get("canonical_frame_count", -1))
            if source_frame_count != frame_count or canonical_frame_count != frame_count:
                time_errors.append(
                    f"canonical/source frame count mismatch: source={source_frame_count}, "
                    f"canonical={canonical_frame_count}, datasets={frame_count}"
                )
            source_verified = False
            if source_root is not None:
                source_path = source_root / str(metadata.attrs.get("source_relative_path", ""))
                if not source_path.is_file():
                    time_errors.append(f"Source episode is unavailable for one-to-one verification: {source_path}")
                else:
                    with h5py.File(source_path, "r") as source:
                        source_t = int(source["step"].shape[0])
                        if source_t != frame_count:
                            time_errors.append(f"Source has {source_t} frames but canonical has {frame_count}")
                        if not np.array_equal(np.asarray(source["step"], dtype=np.int64), source_step):
                            time_errors.append("Canonical source_step is not an exact copy of the source")
                        if not np.array_equal(np.asarray(source["timestamp"], dtype=np.int64), source_timestamp):
                            time_errors.append("Canonical source_timestamp is not an exact copy of the source")
                        for stream in ("cam_data/human_camera", "cam_data/robot_camera", "action", "end_position"):
                            if int(source[stream].shape[0]) != frame_count:
                                time_errors.append(f"Source {stream} first dimension differs from canonical T")
                        source_action = np.asarray(source["action"], dtype=np.float32)
                        source_end = np.asarray(source["end_position"], dtype=np.float32)
                        source_gripper = np.asarray(source["gripper_state"], dtype=np.float32)
                        if not np.array_equal(
                            np.asarray(datasets["metadata/action_raw"], dtype=np.float32), source_action
                        ):
                            evidence_errors.append("metadata/action_raw is not an exact copy of source /action")
                        if not np.array_equal(
                            np.asarray(datasets["metadata/end_position_raw"], dtype=np.float32), source_end
                        ):
                            evidence_errors.append(
                                "metadata/end_position_raw is not an exact copy of source /end_position"
                            )
                        expected_human = poses_euler_to_10d(
                            source_action[:, :6], source_action[:, 6],
                            position_scale=float(metadata.attrs["position_scale_to_canonical"]), euler_order="xyz",
                        )
                        expected_robot = poses_euler_to_10d(
                            source_end, source_gripper,
                            position_scale=float(metadata.attrs["position_scale_to_canonical"]), euler_order="xyz",
                        )
                        if not np.array_equal(human_hand.astype(np.float32), expected_human):
                            evidence_errors.append("source /action is not mapped to human_hand_robot_frame_10d")
                        if not np.array_equal(robot_observed.astype(np.float32), expected_robot):
                            evidence_errors.append("source /end_position is not mapped to robot_ee_observed_10d")
                    if file_sha256(source_path) != metadata.attrs.get("source_sha256"):
                        time_errors.append("source_sha256 does not match the current source file")
                    source_verified = True

            if robot_observed.shape == (frame_count, STATE_DIM) and human_hand.shape == (frame_count, STATE_DIM):
                workspace_min = np.asarray(limits.workspace_min_m)
                workspace_max = np.asarray(limits.workspace_max_m)
                for name, values in (
                    ("trajectories/robot_ee_observed_10d", robot_observed),
                    ("trajectories/human_hand_robot_frame_10d", human_hand),
                ):
                    position = values[:, :3]
                    metrics[f"{name}_workspace_min_m"] = np.min(position, axis=0).tolist()
                    metrics[f"{name}_workspace_max_m"] = np.max(position, axis=0).tolist()
                    if bool(np.any(position < workspace_min) or np.any(position > workspace_max)):
                        structure_errors.append(
                            f"{name} position exceeds workspace {limits.workspace_min_m}..{limits.workspace_max_m} m"
                        )
                    gripper = values[:, -1]
                    metrics[f"{name}_gripper_min"] = float(np.min(gripper))
                    metrics[f"{name}_gripper_max"] = float(np.max(gripper))
                    if bool(np.any(gripper < limits.gripper_min - 1e-6) or np.any(gripper > limits.gripper_max + 1e-6)):
                        structure_errors.append(f"{name} gripper exceeds [{limits.gripper_min}, {limits.gripper_max}]")
                    linear_step, angular_step = _per_step_motion_metrics(values, gap_mask)
                    metrics[f"{name}_max_linear_displacement_m_per_source_step"] = linear_step
                    metrics[f"{name}_max_angular_displacement_rad_per_source_step"] = angular_step

            warnings.append(
                "30 Hz is nominal upstream camera evidence only; record timestamps remain unsuitable for global m/s"
            )
            warnings.append("Source xyz unit and /action-as-robot-command status remain unknown")
            metrics["task"] = str(metadata.attrs.get("task", ""))
            metrics["task_split"] = str(metadata.attrs.get("task_split", ""))
            metrics["split_manifest_sha256"] = str(metadata.attrs.get("split_manifest_sha256", ""))
            metrics["source_relative_path"] = str(metadata.attrs.get("source_relative_path", ""))
            metrics["source_frames"] = source_frame_count
            metrics["canonical_frames"] = canonical_frame_count
            metrics["source_one_to_one_verified"] = source_verified
            metrics["timebase_status"] = time_audit["timebase_status"]
            metrics["gap_count"] = time_audit["gap_count"]
            metrics["segment_count"] = time_audit["segment_count"]
    except (OSError, KeyError, ValueError, M2Error) as exc:
        structure_errors.append(str(exc))
    errors = structure_errors + time_errors + evidence_errors
    return {
        "path": str(path),
        "status": "passed" if not errors else "failed",
        "structure_validation": {
            "status": "passed" if not structure_errors else "failed",
            "errors": structure_errors,
        },
        "time_truth_validation": {
            "status": "passed" if not time_errors else "failed",
            "errors": time_errors,
        },
        "evidence_role_validation": {
            "status": "passed" if not evidence_errors else "failed",
            "errors": evidence_errors,
        },
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }


def validate_canonical_dataset(
    files: Sequence[Path],
    limits: ValidationLimits,
    *,
    minimum_episodes: int,
    minimum_tasks: int = 1,
    source_root: Path | None = None,
    split_manifest_path: Path | None = None,
) -> dict[str, Any]:
    episode_reports = [validate_canonical_episode(path, limits, source_root=source_root) for path in files]
    passed = [report for report in episode_reports if report["status"] == "passed"]
    failed = [report for report in episode_reports if report["status"] != "passed"]
    errors: list[str] = []
    if len(files) < minimum_episodes:
        errors.append(f"Found {len(files)} canonical episode(s), expected at least {minimum_episodes}")
    if failed:
        errors.append(f"{len(failed)} canonical episode(s) failed validation")

    task_names = {report.get("metrics", {}).get("task") for report in passed if report.get("metrics", {}).get("task")}
    if len(task_names) < minimum_tasks:
        errors.append(f"Found {len(task_names)} task(s), expected at least {minimum_tasks}")
    frame_count = sum(int(report.get("metrics", {}).get("frame_count", 0)) for report in passed)
    frame_mapping_passed = sum(
        report.get("metrics", {}).get("canonical_frames") == report.get("metrics", {}).get("source_frames")
        and report.get("metrics", {}).get("source_one_to_one_verified") is True
        for report in passed
    )
    split_validation: dict[str, Any] = {"status": "not_requested"}
    if split_manifest_path is not None:
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        expected_hash = split_manifest.get("split_sha256")
        expected_episodes = {
            record["source_relative_path"]: (record["task"], record["split"])
            for record in split_manifest.get("episodes", [])
        }
        split_errors: list[str] = []
        for report in passed:
            metrics_for_episode = report["metrics"]
            relative = metrics_for_episode["source_relative_path"]
            expected = expected_episodes.get(relative)
            if expected is None:
                split_errors.append(f"Episode absent from frozen split manifest: {relative}")
            elif expected != (metrics_for_episode["task"], metrics_for_episode["task_split"]):
                split_errors.append(f"Task/split assignment mismatch for {relative}")
            if metrics_for_episode["split_manifest_sha256"] != expected_hash:
                split_errors.append(f"Split hash mismatch for {relative}")
        if len(expected_episodes) != len(files):
            split_errors.append(
                f"Split manifest has {len(expected_episodes)} episode(s) but validator received {len(files)}"
            )
        split_validation = {
            "status": "passed" if not split_errors else "failed",
            "errors": split_errors,
            "split_sha256": expected_hash,
            "train_task_count": len(split_manifest.get("train_tasks", [])),
            "heldout_task_count": len(split_manifest.get("heldout_tasks", [])),
        }
        errors.extend(split_errors)
    aggregate: dict[str, Any] = {
        "episode_count": len(files),
        "passed_episode_count": len(passed),
        "failed_episode_count": len(failed),
        "task_count": len(task_names),
        "frame_count": frame_count,
        "source_one_to_one_verified_count": frame_mapping_passed,
        "timebase_status_counts": {
            status: sum(report.get("metrics", {}).get("timebase_status") == status for report in passed)
            for status in sorted(TIMEBASE_STATUSES)
        },
        "gap_count": sum(int(report.get("metrics", {}).get("gap_count", 0)) for report in passed),
        "segment_count": sum(int(report.get("metrics", {}).get("segment_count", 0)) for report in passed),
    }
    return {
        "status": "passed" if not errors else "failed",
        "checked_at_utc": utc_now(),
        "schema_version": SCHEMA_VERSION,
        "minimum_episodes": minimum_episodes,
        "minimum_tasks": minimum_tasks,
        "limits": asdict(limits),
        "errors": errors,
        "aggregate": aggregate,
        "split_validation": split_validation,
        "episodes": episode_reports,
    }


def _font() -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, 16)
    return ImageFont.load_default()


def _trajectory_points(xy: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    left, top, right, bottom = box
    minimum = np.min(xy, axis=0)
    maximum = np.max(xy, axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    scaled = (xy - minimum) / span
    return [(int(left + value[0] * (right - left)), int(bottom - value[1] * (bottom - top))) for value in scaled]


def compose_visualization_frame(
    human_frame: np.ndarray,
    robot_frame: np.ndarray,
    robot_observed: np.ndarray,
    human_hand_robot_frame: np.ndarray,
    frame_index: int,
    task: str,
    font: ImageFont.ImageFont,
    source_step: np.ndarray | None = None,
    source_timestamp: np.ndarray | None = None,
    segment_id: np.ndarray | None = None,
    gap_mask: np.ndarray | None = None,
) -> np.ndarray:
    paired = np.concatenate((human_frame, robot_frame), axis=1)
    height, width, _ = paired.shape
    canvas = Image.new("RGB", (width, height + 72), (20, 20, 20))
    canvas.paste(Image.fromarray(paired), (0, 0))
    draw = ImageDraw.Draw(canvas)
    half_width = width // 2
    draw.rectangle((0, 0, 105, 25), fill=(0, 0, 0))
    draw.rectangle((half_width, 0, half_width + 105, 25), fill=(0, 0, 0))
    draw.text((8, 4), "HUMAN", font=font, fill=(255, 255, 255))
    draw.text((half_width + 8, 4), "ROBOT", font=font, fill=(255, 255, 255))
    robot_pose = robot_observed[frame_index]
    human_pose = human_hand_robot_frame[frame_index]
    time_label = ""
    if source_step is not None and source_timestamp is not None and segment_id is not None:
        boundary = " GAP" if gap_mask is not None and bool(gap_mask[frame_index]) else ""
        time_label = (
            f"  source_step={int(source_step[frame_index])} source_ts={int(source_timestamp[frame_index])} "
            f"segment={int(segment_id[frame_index])}{boundary}"
        )
    line_1 = f"{task}  native_frame={frame_index}/{len(human_hand_robot_frame) - 1}{time_label}"
    line_2 = (
        f"ROBOT OBS xyz=[{robot_pose[0]:+.3f},{robot_pose[1]:+.3f},{robot_pose[2]:+.3f}] grip={robot_pose[-1]:.1f}  "
        f"HUMAN POSE xyz=[{human_pose[0]:+.3f},{human_pose[1]:+.3f},{human_pose[2]:+.3f}] grip={human_pose[-1]:.1f}"
    )
    draw.text((8, height + 6), line_1, font=font, fill=(240, 240, 240))
    draw.text((8, height + 34), line_2, font=font, fill=(240, 240, 240))

    trajectory_box = (width - 150, height + 6, width - 8, height + 64)
    human_points = _trajectory_points(human_hand_robot_frame[:, :2], trajectory_box)
    robot_points = _trajectory_points(robot_observed[:, :2], trajectory_box)
    if len(human_points) >= 2:
        draw.line(human_points, fill=(90, 170, 255), width=2)
        draw.line(robot_points, fill=(255, 170, 70), width=2)
    human_marker = human_points[frame_index]
    robot_marker = robot_points[frame_index]
    draw.ellipse(
        (human_marker[0] - 4, human_marker[1] - 4, human_marker[0] + 4, human_marker[1] + 4),
        fill=(90, 170, 255),
    )
    draw.ellipse(
        (robot_marker[0] - 4, robot_marker[1] - 4, robot_marker[0] + 4, robot_marker[1] + 4),
        fill=(255, 170, 70),
    )
    return np.asarray(canvas)


def _ffmpeg_encoder(ffmpeg: str) -> list[str]:
    listing = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    if " libx264 " in listing:
        return ["-vcodec", "libx264", "-crf", "20"]
    return ["-vcodec", "mpeg4", "-q:v", "4"]


def write_visualization(path: Path, output_path: Path, *, playback_fps: float) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise M2Error("ffmpeg was not found; cannot create M2 visualization videos")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as file:
        demo = file[DEMO_KEY]
        human = demo["metadata/human/images"]
        robot = demo["obs/robot_images"]
        robot_observed = np.asarray(demo["trajectories/robot_ee_observed_10d"])
        human_hand_robot_frame = np.asarray(demo["trajectories/human_hand_robot_frame_10d"])
        metadata = demo["metadata"]
        source_step = np.asarray(metadata["source_step"])
        source_timestamp = np.asarray(metadata["source_timestamp"])
        segment_id = np.asarray(metadata["segment_id"])
        gap_mask = np.asarray(metadata["gap_mask"])
        task = str(metadata.attrs["task"])
        first = compose_visualization_frame(
            human[0],
            robot[0],
            robot_observed,
            human_hand_robot_frame,
            0,
            task,
            _font(),
            source_step,
            source_timestamp,
            segment_id,
            gap_mask,
        )
        height, width, _ = first.shape
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{playback_fps:g}",
            "-i",
            "-",
            "-an",
            *_ffmpeg_encoder(ffmpeg),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdin is not None
        assert process.stderr is not None
        font = _font()
        try:
            for index in range(len(human_hand_robot_frame)):
                frame = compose_visualization_frame(
                    human[index],
                    robot[index],
                    robot_observed,
                    human_hand_robot_frame,
                    index,
                    task,
                    font,
                    source_step,
                    source_timestamp,
                    segment_id,
                    gap_mask,
                )
                process.stdin.write(frame.tobytes())
            process.stdin.close()
        except BrokenPipeError:
            pass
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise M2Error(f"ffmpeg failed for {path} with code {return_code}: {stderr}")
    return {
        "canonical_episode": str(path),
        "video": str(output_path),
        "task": task,
        "frame_count": len(human_hand_robot_frame),
        "overlay_roles": ["observed_robot_ee_pose", "human_hand_pose_in_robot_frame"],
        "playback_fps": playback_fps,
        "playback_rate_semantics": "MP4 encoding only; not evidence of source capture frequency",
    }


def create_visualizations(
    files: Sequence[Path],
    output_dir: Path,
    *,
    count: int,
    seed: int,
    playback_fps: float,
) -> dict[str, Any]:
    if count < 0:
        raise M2Error("Visualization count cannot be negative")
    if count > len(files):
        raise M2Error(f"Requested {count} visualizations from only {len(files)} episodes")
    selected = random.Random(seed).sample(list(files), count)
    records = [
        write_visualization(
            path,
            output_dir / f"sample_{index:02d}_{path.stem}.mp4",
            playback_fps=playback_fps,
        )
        for index, path in enumerate(selected)
    ]
    manifest = {
        "status": "passed",
        "created_at_utc": utc_now(),
        "sampling": "random.Random(seed).sample without replacement",
        "seed": seed,
        "count": count,
        "playback_fps": playback_fps,
        "playback_rate_semantics": "MP4 encoding only; not evidence of source capture frequency",
        "videos": records,
    }
    write_json(output_dir / "visualization_manifest.json", manifest)
    return manifest


def manifest_episode_files(manifest: dict[str, Any]) -> list[Path]:
    return [Path(record["output_path"]) for record in manifest["episodes"]]


def validate_statistics_artifacts(output_root: Path, split_manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    checked: list[str] = []
    heldout_tasks = set(split_manifest["heldout_tasks"])
    expected_roles = {
        "robot_observed_statistics.json": "observed_robot_ee_pose",
        "human_hand_robot_frame_statistics.json": "human_hand_pose_in_robot_frame",
        "dataset_statistics_post_norm_by_role.json": None,
        "data_quality_statistics.json": None,
    }
    for name, expected_role in expected_roles.items():
        path = output_root / name
        if not path.is_file():
            errors.append(f"Missing train-only statistics artifact: {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        provenance = payload.get("_provenance", {})
        if provenance.get("source_split") != "train":
            errors.append(f"{name}: source_split must be train")
        if provenance.get("split_manifest_sha256") != split_manifest["split_sha256"]:
            errors.append(f"{name}: split manifest hash mismatch")
        if provenance.get("heldout_data_used") is not False:
            errors.append(f"{name}: heldout_data_used must be false")
        if provenance.get("generic_action_used") is not False:
            errors.append(f"{name}: generic_action_used must be false")
        if expected_role is not None and provenance.get("role") != expected_role:
            errors.append(f"{name}: role must be {expected_role}")
        leaked = heldout_tasks.intersection(provenance.get("tasks", []))
        if leaked:
            errors.append(f"{name}: held-out task leakage: {sorted(leaked)}")
        checked.append(str(path))
    forbidden_paths = [
        output_root / "dataset_statistics.json",
        output_root / "dataset_statistics_post_norm.json",
        output_root / "delta_dataset_statistics.json",
    ]
    existing_forbidden = [str(path) for path in forbidden_paths if path.exists()]
    if existing_forbidden:
        errors.append(f"Forbidden generic/residual statistics artifacts exist: {existing_forbidden}")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "checked_files": checked,
        "forbidden_generic_statistics_absent": not existing_forbidden,
    }


def write_v1_deprecation_marker(config: ConversionConfig) -> Path | None:
    if not config.legacy_v1_root.is_dir():
        return None
    artifacts = [
        config.legacy_v1_root / "preprocessing_manifest.json",
        config.legacy_v1_root / "m2_pipeline_report.json",
        config.legacy_v1_root / "m2_validation_report.json",
        config.legacy_v1_root / "dataset_statistics.json",
        config.legacy_v1_root / "dataset_statistics_post_norm.json",
        config.legacy_v1_root / "delta_dataset_statistics.json",
        config.legacy_v1_root / "visualizations/visualization_manifest.json",
        config.legacy_v1_report,
    ]
    marker = {
        "status": "acceptance_withdrawn",
        "created_at_utc": utc_now(),
        "schema_version": "human2robot-canonical-hdf5-v1",
        "reason": "unverified 30 Hz assumption and synthetic 10 Hz timeline",
        "replacement": {
            "schema_version": SCHEMA_VERSION,
            "plan": str(config.report_root / "RECAP_人手示范用于上下文学习指导真实机器人复现实验方案.md"),
            "acceptance_report": str(config.report_root / "M2_Human2Robot_native_time_验收报告.md"),
        },
        "allowed_uses": ["legacy regression", "dtype/schema tests", "visualization", "time ablation"],
        "forbidden_uses": ["M2-v02 retrieval", "training statistics", "main experiments", "paper conclusions"],
        "frozen_artifact_sha256": {str(path): file_sha256(path) for path in artifacts if path.is_file()},
    }
    marker_path = config.legacy_v1_root / "DEPRECATED_M2_V01.json"
    write_json(marker_path, marker)
    return marker_path


def run_m2_pipeline(
    config: ConversionConfig,
    limits: ValidationLimits,
    *,
    visualization_count: int = 10,
    visualization_seed: int = 20260711,
    visualization_playback_fps: float = 10.0,
) -> dict[str, Any]:
    if config.timebase_policy != PRESERVE_NATIVE:
        raise M2Error("run_m2_pipeline is the M2-v03 acceptance path and requires preserve_native")
    superseded_v2_marker = config.legacy_v2_root / "SUPERSEDED_M2_V02.json"
    if not superseded_v2_marker.is_file():
        raise M2Error(f"Missing required frozen-v2 marker: {superseded_v2_marker}")
    manifest = convert_dataset(config)
    files = manifest_episode_files(manifest)
    split_manifest_path = config.output_root / "task_split_manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    train_files = [
        path for path, record in zip(files, manifest["episodes"], strict=True) if record["task_split"] == "train"
    ]
    statistics = compute_and_write_statistics(
        train_files,
        config.output_root,
        required_split="train",
        split_manifest_hash=split_manifest["split_sha256"],
    )
    statistics_validation = validate_statistics_artifacts(config.output_root, split_manifest)
    validation = validate_canonical_dataset(
        files,
        limits,
        minimum_episodes=config.episode_count,
        minimum_tasks=min(config.episode_count, 20),
        source_root=config.source_root,
        split_manifest_path=split_manifest_path,
    )
    validation["statistics_validation"] = statistics_validation
    if statistics_validation["status"] != "passed":
        validation["status"] = "failed"
        validation["errors"].extend(statistics_validation["errors"])
    write_json(config.output_root / "m2_validation_report.json", validation)
    if validation["status"] != "passed":
        raise M2Error(f"Canonical validation failed; see {config.output_root / 'm2_validation_report.json'}")
    visualizations = create_visualizations(
        files,
        config.output_root / "visualizations",
        count=visualization_count,
        seed=visualization_seed,
        playback_fps=visualization_playback_fps,
    )
    report = {
        "status": "passed",
        "completed_at_utc": utc_now(),
        "schema_version": SCHEMA_VERSION,
        "output_root": str(config.output_root),
        "conversion": {
            "episode_count": len(files),
            "task_count": len({record["task"] for record in manifest["episodes"]}),
            "canonical_frames": sum(record["canonical_frames"] for record in manifest["episodes"]),
            "source_frames": sum(record["source_frames"] for record in manifest["episodes"]),
            "timebase_policy": config.timebase_policy,
            "frame_selection": "all_source_rows",
            "nominal_camera_fps": 30.0,
            "nominal_camera_fps_status": "verified_upstream",
            "record_timebase_globally_trusted": False,
            "split_sha256": split_manifest["split_sha256"],
            "train_task_count": len(split_manifest["train_tasks"]),
            "heldout_task_count": len(split_manifest["heldout_tasks"]),
        },
        "statistics": statistics,
        "statistics_validation": statistics_validation,
        "validation": validation["aggregate"],
        "visualizations": {
            "count": visualizations["count"],
            "seed": visualizations["seed"],
            "playback_fps": visualizations["playback_fps"],
            "manual_review_status": "pending_manual_review",
        },
        "artifacts": {
            "preprocessing_manifest": str(config.output_root / "preprocessing_manifest.json"),
            "validation_report": str(config.output_root / "m2_validation_report.json"),
            "task_split_manifest": str(split_manifest_path),
            "timebase_audit_report": str(config.output_root / "timebase_audit_report.json"),
            "visualization_manifest": str(config.output_root / "visualizations/visualization_manifest.json"),
            "v2_superseded_marker": str(superseded_v2_marker),
            "source_evidence_manifest": str(config.evidence_manifest),
        },
    }
    write_json(config.output_root / "m2_pipeline_report.json", report)
    write_json(
        config.report_root / "M2_Human2Robot_semantic_safe_自动验收报告.json",
        {"pipeline": report, "validation": validation},
    )
    return report


def canonical_files(input_root: Path) -> list[Path]:
    if input_root.is_file():
        return [input_root]
    return sorted(path for path in input_root.glob("demo_*.hdf5") if path.is_file())


def ensure_paths_exist(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise M2Error(f"Missing expected artifact(s): {missing}")
