#!/usr/bin/env python3
"""Check RH20T cfg data access for the v01 M1 milestone.

The check is intentionally read-only for the dataset root. It verifies that one
paired RH20T episode exposes robot low-dim arrays, a paired human RGB video, and
robot camera calibration files, then writes a compact JSON report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path("/DATA1/wxs/DATASETS/RH20T/320x180")
DEFAULT_OUTPUT = Path("data/RH20T/m1_access_check_cfg3.json")
DEFAULT_TASK_DESCRIPTION = Path("data/RH20T/task_description.json")
REQUIRED_LOWDIM_FILES = ("tcp_base.npy", "gripper.npy")
CALIBRATION_FILES = ("devices.npy", "intrinsics.npy", "extrinsics.npy", "tcp.npy")


class CheckError(RuntimeError):
    """Raised when an M1 acceptance requirement cannot be satisfied."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"RH20T 320x180 root. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--cfg",
        default="RH20T_cfg3",
        help="RH20T configuration directory name. Default: RH20T_cfg3",
    )
    parser.add_argument(
        "--episode",
        help=(
            "Robot episode id to check. If omitted, the script selects the first "
            "robot episode with low-dim data, paired human RGB video, and robot calibration."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path for the JSON report. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--task-description",
        type=Path,
        default=DEFAULT_TASK_DESCRIPTION,
        help=f"RH20T Task Description File path. Default: {DEFAULT_TASK_DESCRIPTION}",
    )
    parser.add_argument(
        "--skip-ffprobe",
        action="store_true",
        help="Only check video file presence and timestamps; do not open it with ffprobe.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_npy(path: Path) -> Any:
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        return value.item()
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def first_item(mapping_or_sequence: Any) -> tuple[Any, Any] | tuple[None, None]:
    if isinstance(mapping_or_sequence, dict):
        if not mapping_or_sequence:
            return None, None
        key = next(iter(mapping_or_sequence))
        return key, mapping_or_sequence[key]
    if isinstance(mapping_or_sequence, (list, tuple)):
        if not mapping_or_sequence:
            return None, None
        return 0, mapping_or_sequence[0]
    return None, None


def summarize_leaf(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        summary: dict[str, Any] = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if np.issubdtype(value.dtype, np.number):
            summary["finite"] = bool(np.isfinite(value).all())
            if value.size:
                summary["sample"] = to_jsonable(value.reshape(-1)[: min(4, value.size)])
        return summary
    if isinstance(value, dict):
        return {
            "type": "dict",
            "keys": [str(k) for k in list(value.keys())[:8]],
            "key_count": len(value),
        }
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    return {"type": type(value).__name__, "value": to_jsonable(value)}


def summarize_npy_object(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        summary: dict[str, Any] = {
            "type": "dict",
            "key_count": len(obj),
            "sample_keys": [str(k) for k in list(obj.keys())[:8]],
        }
        key, value = first_item(obj)
        summary["sample_key"] = str(key) if key is not None else None

        if isinstance(value, list):
            summary["sample_value_type"] = "list"
            summary["sample_value_length"] = len(value)
            _, nested = first_item(value)
            summary["sample_record"] = summarize_record(nested)
        elif isinstance(value, dict):
            summary["sample_value_type"] = "dict"
            summary["sample_value_length"] = len(value)
            nested_key, nested = first_item(value)
            summary["sample_nested_key"] = to_jsonable(nested_key)
            summary["sample_record"] = summarize_record(nested)
        else:
            summary["sample_record"] = summarize_leaf(value)
        return summary

    if isinstance(obj, np.ndarray):
        return summarize_leaf(obj)

    return summarize_leaf(obj)


def summarize_record(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        summary: dict[str, Any] = {
            "type": "dict",
            "keys": [str(k) for k in record.keys()],
        }
        for key, value in record.items():
            if key == "timestamp":
                summary["timestamp"] = to_jsonable(value)
            elif isinstance(value, np.ndarray):
                summary[f"{key}_shape"] = list(value.shape)
                summary[f"{key}_dtype"] = str(value.dtype)
                if np.issubdtype(value.dtype, np.number):
                    summary[f"{key}_finite"] = bool(np.isfinite(value).all())
                    summary[f"{key}_sample"] = to_jsonable(
                        value.reshape(-1)[: min(4, value.size)]
                    )
            elif isinstance(value, dict):
                summary[f"{key}_keys"] = [str(k) for k in list(value.keys())[:8]]
            else:
                summary[key] = to_jsonable(value)
        return summary
    return summarize_leaf(record)


def summarize_lowdim_file(path: Path) -> dict[str, Any]:
    obj = load_npy(path)
    summary = summarize_npy_object(obj)
    summary["path"] = str(path)
    summary["size_bytes"] = path.stat().st_size
    return summary


def timestamp_summary(path: Path) -> dict[str, Any]:
    timestamps = load_npy(path)
    if not isinstance(timestamps, np.ndarray):
        raise CheckError(f"timestamps file is not an ndarray: {path}")
    if timestamps.ndim != 1:
        raise CheckError(f"timestamps file is not 1-D: {path}")
    if timestamps.size == 0:
        raise CheckError(f"timestamps file is empty: {path}")
    monotonic = bool(np.all(np.diff(timestamps.astype(np.int64)) >= 0))
    return {
        "path": str(path),
        "count": int(timestamps.size),
        "dtype": str(timestamps.dtype),
        "first": to_jsonable(timestamps[0]),
        "last": to_jsonable(timestamps[-1]),
        "monotonic_non_decreasing": monotonic,
    }


def run_ffprobe(video_path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {"available": False, "opened": False, "reason": "ffprobe not found"}

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {
            "available": True,
            "opened": False,
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
        }

    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams", [])
    return {
        "available": True,
        "opened": bool(streams),
        "stream": streams[0] if streams else None,
    }


def summarize_video_episode(
    episode_rgb_dir: Path, *, require_video_decode: bool
) -> dict[str, Any]:
    if not episode_rgb_dir.is_dir():
        raise CheckError(f"RGB episode directory is missing: {episode_rgb_dir}")

    camera_dirs = sorted(
        p for p in episode_rgb_dir.iterdir() if p.is_dir() and p.name.startswith("cam_")
    )
    checked: list[dict[str, Any]] = []
    for camera_dir in camera_dirs:
        color_dir = camera_dir / "color"
        video_path = color_dir / "color.mp4"
        timestamps_path = color_dir / "timestamps.npy"
        if not video_path.is_file() or not timestamps_path.is_file():
            continue

        summary = {
            "camera": camera_dir.name,
            "color_mp4": str(video_path),
            "color_mp4_size_bytes": video_path.stat().st_size,
            "timestamps": timestamp_summary(timestamps_path),
        }
        if require_video_decode:
            summary["ffprobe"] = run_ffprobe(video_path)
        checked.append(summary)
        break

    if not checked:
        raise CheckError(f"No camera with color.mp4 + timestamps.npy in {episode_rgb_dir}")

    if require_video_decode and not checked[0].get("ffprobe", {}).get("opened"):
        raise CheckError(f"ffprobe could not open the checked video in {episode_rgb_dir}")

    return {
        "path": str(episode_rgb_dir),
        "camera_count": len(camera_dirs),
        "checked_camera": checked[0],
    }


def summarize_calibration(calib_root: Path, calib_id: str | int) -> dict[str, Any]:
    calib_dir = calib_root / str(calib_id)
    if not calib_dir.is_dir():
        raise CheckError(f"Calibration directory is missing: {calib_dir}")

    files: dict[str, Any] = {}
    for name in CALIBRATION_FILES:
        path = calib_dir / name
        if not path.is_file():
            raise CheckError(f"Calibration file is missing: {path}")
        files[name] = summarize_lowdim_file(path)

    image_files = (
        sorted((calib_dir / "imgs").glob("*")) if (calib_dir / "imgs").is_dir() else []
    )
    return {
        "path": str(calib_dir),
        "files": files,
        "image_file_count": len([p for p in image_files if p.is_file()]),
        "sample_image_files": [str(p) for p in image_files[:8]],
    }


def episode_has_required_files(
    episode: str, lowdim_cfg: Path, rgb_cfg: Path, calib_root: Path
) -> bool:
    robot_lowdim = lowdim_cfg / episode
    human_rgb = rgb_cfg / f"{episode}_human"
    metadata_path = robot_lowdim / "metadata.json"
    if not metadata_path.is_file():
        return False
    transformed = robot_lowdim / "transformed"
    if any(not (transformed / name).is_file() for name in REQUIRED_LOWDIM_FILES):
        return False
    if not human_rgb.is_dir():
        return False
    if not any(
        (camera / "color" / "color.mp4").is_file()
        and (camera / "color" / "timestamps.npy").is_file()
        for camera in human_rgb.glob("cam_*")
    ):
        return False
    try:
        metadata = load_json(metadata_path)
    except json.JSONDecodeError:
        return False
    calib_id = metadata.get("calib")
    return calib_id is not None and (calib_root / str(calib_id)).is_dir()


def select_episode(lowdim_cfg: Path, rgb_cfg: Path, calib_root: Path, requested: str | None) -> str:
    if requested:
        if not episode_has_required_files(requested, lowdim_cfg, rgb_cfg, calib_root):
            raise CheckError(f"Requested episode does not satisfy M1 access checks: {requested}")
        return requested

    for candidate in sorted(p.name for p in lowdim_cfg.iterdir() if p.is_dir()):
        if candidate.endswith("_human"):
            continue
        if episode_has_required_files(candidate, lowdim_cfg, rgb_cfg, calib_root):
            return candidate
    raise CheckError("No robot episode satisfies low-dim + paired human video + calibration checks")


def iter_paths_to_depth(root: Path, max_depth: int) -> list[Path]:
    pending: list[tuple[Path, int]] = [(root, 0)]
    paths: list[Path] = []
    while pending:
        path, depth = pending.pop()
        try:
            children = list(path.iterdir())
        except OSError:
            continue
        for child in children:
            paths.append(child)
            if child.is_dir() and depth < max_depth:
                pending.append((child, depth + 1))
    return paths


def find_task_description_files(root: Path) -> list[str]:
    names = []
    for path in iter_paths_to_depth(root.parent, max_depth=2):
        if path.is_file() and "task" in path.name.lower() and path.stat().st_size > 0:
            names.append(str(path))
            if len(names) >= 20:
                break
    return names


def task_id_from_episode(episode: str) -> str:
    parts = episode.split("_")
    if len(parts) < 2 or parts[0] != "task":
        raise CheckError(f"Cannot parse task id from episode name: {episode}")
    return f"{parts[0]}_{parts[1]}"


def resolve_task_description(path: Path, root: Path) -> Path:
    candidates = [
        path,
        Path("task_description.json"),
        root / "task_description.json",
        root.parent / "task_description.json",
    ]
    candidates.extend(Path(p) for p in find_task_description_files(root))

    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if candidate.is_file():
            return candidate.resolve()

    raise CheckError(
        "Task Description File was not found. Pass --task-description, "
        f"or place it at {DEFAULT_TASK_DESCRIPTION}."
    )


def summarize_task_description(path: Path, episode: str) -> dict[str, Any]:
    descriptions = load_json(path)
    if not isinstance(descriptions, dict) or not descriptions:
        raise CheckError(f"Task Description File is empty or not an object: {path}")

    task_id = task_id_from_episode(episode)
    task_entry = descriptions.get(task_id)
    if not isinstance(task_entry, dict):
        raise CheckError(f"Task Description File has no entry for {task_id}: {path}")

    english = task_entry.get("task_description_english")
    chinese = task_entry.get("task_description_chinese")
    if not english or not chinese:
        raise CheckError(f"Task Description entry is incomplete for {task_id}: {path}")

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "task_count": len(descriptions),
        "checked_task_id": task_id,
        "checked_description": {
            "task_description_english": english,
            "task_description_chinese": chinese,
        },
    }


def count_top_level_episodes(path: Path) -> dict[str, int]:
    robot_count = 0
    human_count = 0
    for child in path.iterdir():
        if not child.is_dir():
            continue
        if child.name.endswith("_human"):
            human_count += 1
        elif child.name.startswith("task_"):
            robot_count += 1
    return {"robot_episode_dirs": robot_count, "human_episode_dirs": human_count}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    lowdim_cfg = root / "LowDim" / args.cfg
    rgb_cfg = root / "RGB" / args.cfg
    calibration_cfg = root / "Calibration" / args.cfg
    calib_root = calibration_cfg / "calib"

    for path in (lowdim_cfg, rgb_cfg, calib_root):
        if not path.is_dir():
            raise CheckError(f"Required RH20T directory is missing: {path}")

    episode = select_episode(lowdim_cfg, rgb_cfg, calib_root, args.episode)
    robot_lowdim = lowdim_cfg / episode
    human_lowdim = lowdim_cfg / f"{episode}_human"
    robot_rgb = rgb_cfg / episode
    human_rgb = rgb_cfg / f"{episode}_human"

    robot_metadata = load_json(robot_lowdim / "metadata.json")
    human_metadata = load_json(human_lowdim / "metadata.json") if (human_lowdim / "metadata.json").is_file() else {}

    lowdim = {
        name: summarize_lowdim_file(robot_lowdim / "transformed" / name)
        for name in REQUIRED_LOWDIM_FILES
    }
    robot_video = summarize_video_episode(robot_rgb, require_video_decode=not args.skip_ffprobe)
    human_video = summarize_video_episode(human_rgb, require_video_decode=not args.skip_ffprobe)
    calibration = summarize_calibration(calib_root, robot_metadata["calib"])
    task_description_path = resolve_task_description(args.task_description, root)
    task_description = summarize_task_description(task_description_path, episode)

    warnings = []
    human_calib = human_metadata.get("calib")
    if human_calib is not None and not (calib_root / str(human_calib)).is_dir():
        warnings.append(
            "Paired human metadata references calibration "
            f"{human_calib}, but it is not present under {calib_root}."
        )

    return {
        "status": "passed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "RH20T",
        "root": str(root),
        "cfg": args.cfg,
        "episode": episode,
        "paired_human_episode": f"{episode}_human",
        "layout": {
            "lowdim": str(lowdim_cfg),
            "rgb": str(rgb_cfg),
            "calibration": str(calibration_cfg),
        },
        "top_level_counts": {
            "lowdim": count_top_level_episodes(lowdim_cfg),
            "rgb": count_top_level_episodes(rgb_cfg),
            "calibration_dirs": len([p for p in calib_root.iterdir() if p.is_dir()]),
        },
        "robot_metadata": robot_metadata,
        "human_metadata": human_metadata,
        "robot_lowdim": lowdim,
        "robot_video": robot_video,
        "human_video": human_video,
        "robot_calibration": calibration,
        "task_description": task_description,
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
