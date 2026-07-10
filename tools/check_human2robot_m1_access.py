#!/usr/bin/env python3
"""Check Human2Robot HDF5 data access for the v01 M1 milestone.

The check is intentionally read-only for the dataset root. It verifies that the
local Human2Robot v1 tree contains HDF5 episodes and that one paired episode can
be opened with synchronized human/robot visual streams, robot state/action
arrays, human hand coordinates, and monotonic timestamps.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_ROOT = Path("/DATA1/wxs/DATASETS/Human2Robot/data/v1")
DEFAULT_OUTPUT = Path("data/Human2Robot/m1_access_check_v1.json")
PREFERRED_EPISODE = Path("roll/episode_0.hdf5")

REQUIRED_DATASETS = {
    "action": {"ndim": 2, "last_dim": 7},
    "qpos": {"ndim": 2, "last_dim": 7},
    "qvel": {"ndim": 2, "last_dim": 7},
    "end_position": {"ndim": 2, "last_dim": 6},
    "gripper_state": {"ndim": 1},
    "step": {"ndim": 1},
    "timestamp": {"ndim": 1},
    "transformed_hand_coords": {"ndim": 3, "tail_shape": (24, 3)},
    "transformed_hand_frames": {"ndim": 3, "tail_shape": (4, 3)},
    "cam_data/human_camera": {"ndim": 4, "last_dim": 3},
    "cam_data/robot_camera": {"ndim": 4, "last_dim": 3},
}

OPTIONAL_DATASETS = {
    "depth_data/human_camera": {"ndim": 3},
    "depth_data/robot_camera": {"ndim": 3},
}


class CheckError(RuntimeError):
    """Raised when an M1 acceptance requirement cannot be satisfied."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Human2Robot v1 root. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--episode",
        type=Path,
        help=(
            "Episode path to check, relative to --root or absolute. "
            f"If omitted, the script prefers {PREFERRED_EPISODE}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for the JSON report. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{size_bytes}B"


def list_hdf5_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise CheckError(f"Human2Robot root directory is missing: {root}")
    files = sorted(root.rglob("*.hdf5"))
    if not files:
        raise CheckError(f"No HDF5 episodes found under {root}")
    return files


def task_counts(root: Path, files: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in files:
        task = str(path.parent.relative_to(root))
        counts[task] = counts.get(task, 0) + 1
    return dict(sorted(counts.items()))


def find_local_files(root: Path, patterns: tuple[str, ...]) -> list[str]:
    dataset_root = root.parent.parent if root.name == "v1" else root
    found: list[str] = []
    for pattern in patterns:
        found.extend(str(path) for path in dataset_root.rglob(pattern) if path.is_file())
    return sorted(set(found))


def get_dataset(file: h5py.File, name: str) -> h5py.Dataset:
    obj = file.get(name)
    if not isinstance(obj, h5py.Dataset):
        raise CheckError(f"Required dataset is missing: {name}")
    return obj


def check_shape(name: str, dataset: h5py.Dataset, expected: dict[str, Any]) -> None:
    shape = dataset.shape
    ndim = expected.get("ndim")
    if ndim is not None and len(shape) != ndim:
        raise CheckError(f"{name} has rank {len(shape)}, expected {ndim}: {shape}")
    last_dim = expected.get("last_dim")
    if last_dim is not None and (not shape or shape[-1] != last_dim):
        raise CheckError(f"{name} last dimension is {shape[-1:]}, expected {last_dim}")
    tail_shape = expected.get("tail_shape")
    if tail_shape is not None and tuple(shape[-len(tail_shape) :]) != tuple(tail_shape):
        raise CheckError(f"{name} tail shape is {shape}, expected *{tail_shape}")


def numeric_sample(dataset: h5py.Dataset) -> np.ndarray:
    if dataset.shape and dataset.shape[0] > 0 and dataset.ndim >= 3:
        index = (0,) + tuple(slice(None) for _ in dataset.shape[1:])
        return np.asarray(dataset[index])
    return np.asarray(dataset[()])


def dataset_summary(name: str, dataset: h5py.Dataset, expected: dict[str, Any]) -> dict[str, Any]:
    check_shape(name, dataset, expected)
    sample = numeric_sample(dataset)
    summary: dict[str, Any] = {
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
        "compression": dataset.compression,
    }
    if np.issubdtype(dataset.dtype, np.number):
        summary["sample_finite"] = bool(np.isfinite(sample).all())
        if sample.size:
            summary["sample_min"] = to_jsonable(np.nanmin(sample))
            summary["sample_max"] = to_jsonable(np.nanmax(sample))
            summary["sample_values"] = to_jsonable(sample.reshape(-1)[: min(6, sample.size)])
    return summary


def timestamp_summary(dataset: h5py.Dataset) -> dict[str, Any]:
    values = np.asarray(dataset[()])
    if values.ndim != 1:
        raise CheckError(f"timestamp is not 1-D: {values.shape}")
    if values.size == 0:
        raise CheckError("timestamp is empty")
    diffs = np.diff(values.astype(np.int64))
    monotonic = bool(np.all(diffs >= 0))
    if not monotonic:
        raise CheckError("timestamp is not monotonic non-decreasing")
    return {
        "count": int(values.size),
        "dtype": str(values.dtype),
        "first": to_jsonable(values[0]),
        "last": to_jsonable(values[-1]),
        "monotonic_non_decreasing": monotonic,
        "min_delta": to_jsonable(diffs.min()) if diffs.size else None,
        "max_delta": to_jsonable(diffs.max()) if diffs.size else None,
    }


def step_summary(dataset: h5py.Dataset) -> dict[str, Any]:
    values = np.asarray(dataset[()])
    if values.ndim != 1:
        raise CheckError(f"step is not 1-D: {values.shape}")
    if values.size == 0:
        raise CheckError("step is empty")
    diffs = np.diff(values.astype(np.int64))
    return {
        "count": int(values.size),
        "dtype": str(values.dtype),
        "first": to_jsonable(values[0]),
        "last": to_jsonable(values[-1]),
        "monotonic_non_decreasing": bool(np.all(diffs >= 0)),
        "min_delta": to_jsonable(diffs.min()) if diffs.size else None,
        "max_delta": to_jsonable(diffs.max()) if diffs.size else None,
    }


def validate_time_axis(summaries: dict[str, dict[str, Any]]) -> int:
    lengths = {
        name: int(summary["shape"][0])
        for name, summary in summaries.items()
        if summary.get("shape")
    }
    if not lengths:
        raise CheckError("No datasets expose a frame/time axis")
    unique_lengths = sorted(set(lengths.values()))
    if len(unique_lengths) != 1:
        raise CheckError(f"Dataset time-axis lengths do not match: {lengths}")
    frame_count = unique_lengths[0]
    if frame_count <= 0:
        raise CheckError("Selected episode has no frames")
    return frame_count


def select_episode(root: Path, files: list[Path], requested: Path | None) -> Path:
    if requested:
        candidate = requested if requested.is_absolute() else root / requested
        if not candidate.is_file():
            raise CheckError(f"Requested Human2Robot episode is missing: {candidate}")
        return candidate.resolve()

    preferred = root / PREFERRED_EPISODE
    if preferred.is_file():
        return preferred.resolve()
    return files[0].resolve()


def summarize_episode(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as file:
        required = {
            name: dataset_summary(name, get_dataset(file, name), expected)
            for name, expected in REQUIRED_DATASETS.items()
        }
        optional = {
            name: dataset_summary(name, dataset, expected)
            for name, expected in OPTIONAL_DATASETS.items()
            if isinstance((dataset := file.get(name)), h5py.Dataset)
        }
        frame_count = validate_time_axis({**required, **optional})
        timestamp = timestamp_summary(get_dataset(file, "timestamp"))
        step = step_summary(get_dataset(file, "step"))

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "size_human": human_size(path.stat().st_size),
        "frame_count": frame_count,
        "datasets": required,
        "optional_datasets": optional,
        "timestamp": timestamp,
        "step": step,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    files = list_hdf5_files(root)
    episode_path = select_episode(root, files, args.episode)
    counts = task_counts(root, files)
    total_size = sum(path.stat().st_size for path in files)
    readmes = find_local_files(root, ("README*", "readme*"))
    licenses = find_local_files(root, ("LICENSE*", "license*"))

    warnings = []
    if not licenses:
        warnings.append(
            "No local LICENSE file was found under the Human2Robot dataset directory; "
            "confirm upstream license before using the data beyond access validation."
        )
    if not readmes:
        warnings.append(
            "No local README file was found under the Human2Robot dataset directory; "
            "source/download metadata should be recorded separately."
        )
    episode_summary = summarize_episode(episode_path)
    optional_names = set(episode_summary["optional_datasets"])

    return {
        "status": "passed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "H&R / Human2Robot",
        "root": str(root),
        "layout": "task directories containing episode_*.hdf5 files",
        "inventory": {
            "hdf5_episode_count": len(files),
            "task_count": len(counts),
            "task_episode_counts": counts,
            "total_size_bytes": total_size,
            "total_size_human": human_size(total_size),
            "local_readme_files": readmes,
            "local_license_files": licenses,
        },
        "selected_episode": episode_summary,
        "acceptance_checks": {
            "hdf5_tree_found": True,
            "paired_human_robot_rgb_opened": True,
            "paired_human_robot_depth_opened": {
                "depth_data/human_camera",
                "depth_data/robot_camera",
            }.issubset(optional_names),
            "robot_state_action_opened": True,
            "human_hand_coordinates_opened": True,
            "timestamp_monotonic": True,
            "time_axis_consistent": True,
        },
        "warnings": warnings,
    }


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args)
    except CheckError as exc:
        report = {
            "status": "failed",
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=to_jsonable) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, indent=2, ensure_ascii=False, default=to_jsonable))
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=to_jsonable) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=to_jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
