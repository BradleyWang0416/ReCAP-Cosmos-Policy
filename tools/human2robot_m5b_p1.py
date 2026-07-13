#!/usr/bin/env python3
"""Build and validate the M5B-P1 held-out human-only demonstration pool.

The source Human2Robot containers are paired, but this tool deliberately reads
only the six preregistered human/time datasets.  Robot cameras, robot EE
trajectories, qpos/qvel, normalization targets, and checkpoint-selection data
are neither opened nor copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch

from tools.human2robot_m2 import audit_timebase, poses_euler_to_10d
from tools.human2robot_m5b_protocol import file_sha256

SCHEMA_VERSION = "human2robot-m5b-p1-human-only-pool-v1"
REPORT_SCHEMA_VERSION = "human2robot-m5b-p1-data-acceptance-v1"
GATE_ID = "M5B-P1-DATA"
SELECTION_SEED = 20260711
REQUIRED_PER_TASK = 10
H_STEPS = 8
POOL_SIZES = [0, 1, 2, 4, 8, 10]
ALLOWED_SOURCE_DATASETS = (
    "cam_data/human_camera",
    "action",
    "transformed_hand_coords",
    "transformed_hand_frames",
    "step",
    "timestamp",
)
DERIVED_DATASETS = {
    "data/demo_0/human/images",
    "data/demo_0/human/hand_plan_10d",
    "data/demo_0/human/hand_coords",
    "data/demo_0/human/hand_frames",
    "data/demo_0/time/source_step",
    "data/demo_0/time/source_timestamp",
    "data/demo_0/time/segment_id",
    "data/demo_0/time/gap_mask",
}


class P1Error(RuntimeError):
    """Raised when P1 independence or leakage requirements are violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P1Error(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _episode_number(path: Path) -> tuple[int, str]:
    suffix = path.stem.rsplit("_", 1)[-1]
    return (int(suffix), path.name) if suffix.isdigit() else (2**63 - 1, path.name)


def rank_source_candidates(
    source_root: Path,
    task: str,
    anchor_relative_path: str,
    seed: int = SELECTION_SEED,
) -> list[Path]:
    """Keep the frozen pilot source first, then stable-hash-rank the rest."""
    task_root = source_root / task
    paths = sorted(task_root.glob("episode_*.hdf5"), key=_episode_number)
    anchor = source_root / anchor_relative_path
    _require(anchor in paths and anchor.is_file(), f"Missing frozen pilot anchor: {anchor_relative_path}")
    remainder = [path for path in paths if path != anchor]
    remainder.sort(
        key=lambda path: hashlib.sha256(
            f"{seed}:{path.relative_to(source_root)}".encode("utf-8")
        ).hexdigest()
    )
    return [anchor, *remainder]


def _dataset(source: h5py.File, name: str) -> h5py.Dataset:
    value = source.get(name)
    _require(isinstance(value, h5py.Dataset), f"Missing human-only source dataset: {name}")
    return value


def inspect_human_source(path: Path) -> dict[str, Any]:
    """Inspect only the allowlisted human/time source datasets."""
    with h5py.File(path, "r") as source:
        human_images = _dataset(source, "cam_data/human_camera")
        action = _dataset(source, "action")
        hand_coords = _dataset(source, "transformed_hand_coords")
        hand_frames = _dataset(source, "transformed_hand_frames")
        step_dataset = _dataset(source, "step")
        timestamp_dataset = _dataset(source, "timestamp")

        _require(human_images.ndim == 4 and human_images.shape[-1] == 3, f"Bad human RGB: {path}")
        _require(human_images.dtype in {np.dtype("uint8"), np.dtype("uint16")}, f"Bad RGB dtype: {path}")
        _require(action.ndim == 2 and action.shape[-1] == 7, f"Bad human action: {path}")
        _require(hand_coords.ndim == 3 and hand_coords.shape[-1] == 3, f"Bad hand coords: {path}")
        _require(hand_frames.ndim == 3 and hand_frames.shape[-1] == 3, f"Bad hand frames: {path}")
        _require(step_dataset.ndim == timestamp_dataset.ndim == 1, f"Bad time arrays: {path}")
        lengths = {
            int(human_images.shape[0]),
            int(action.shape[0]),
            int(hand_coords.shape[0]),
            int(hand_frames.shape[0]),
            int(step_dataset.shape[0]),
            int(timestamp_dataset.shape[0]),
        }
        _require(len(lengths) == 1, f"Human-only time axes differ: {path}")
        frame_count = next(iter(lengths))
        _require(frame_count >= H_STEPS, f"Fewer than H={H_STEPS} human frames: {path}")

        action_values = np.asarray(action[:], dtype=np.float64)
        step = np.asarray(step_dataset[:], dtype=np.int64)
        timestamp = np.asarray(timestamp_dataset[:], dtype=np.int64)
        _require(np.isfinite(action_values).all(), f"Human action contains NaN/Inf: {path}")
        _require(np.isfinite(step).all() and np.isfinite(timestamp).all(), f"Time contains NaN/Inf: {path}")
        _require(np.all((action_values[:, 6] >= 0) & (action_values[:, 6] <= 1)), f"Bad gripper: {path}")
        time_audit = audit_timebase(step, timestamp)
        counts = Counter(np.asarray(time_audit["segment_id"], dtype=np.int64).tolist())
        max_segment_frames = max(counts.values())
        _require(max_segment_frames >= H_STEPS, f"No gap-safe H={H_STEPS} segment: {path}")
    return {
        "frame_count": frame_count,
        "source_image_shape": list(human_images.shape[1:]),
        "source_image_dtype": str(human_images.dtype),
        "max_gap_safe_segment_frames": max_segment_frames,
        "gap_count": int(np.count_nonzero(time_audit["gap_mask"])),
        "source_datasets_read": list(ALLOWED_SOURCE_DATASETS),
        "forbidden_source_datasets_read": [],
    }


def select_sources(
    source_root: Path,
    heldout_tasks: Iterable[str],
    anchors: dict[str, str],
) -> tuple[dict[str, list[Path]], list[dict[str, str]], dict[str, int]]:
    selected: dict[str, list[Path]] = {}
    rejected: list[dict[str, str]] = []
    discovered: dict[str, int] = {}
    for task in sorted(heldout_tasks):
        ranked = rank_source_candidates(source_root, task, anchors[task])
        discovered[task] = len(ranked)
        accepted: list[Path] = []
        for candidate in ranked:
            try:
                inspect_human_source(candidate)
            except (P1Error, OSError, ValueError) as exc:
                rejected.append(
                    {"task": task, "source_relative_path": str(candidate.relative_to(source_root)), "error": str(exc)}
                )
                continue
            accepted.append(candidate)
            if len(accepted) == REQUIRED_PER_TASK:
                break
        _require(
            len(accepted) == REQUIRED_PER_TASK,
            f"Only {len(accepted)} valid independent sources for {task}; need {REQUIRED_PER_TASK}",
        )
        _require(accepted[0].relative_to(source_root).as_posix() == anchors[task], f"Anchor was rejected: {task}")
        selected[task] = accepted
    return selected, rejected, discovered


def _update_content_hash(digest: Any, name: str, array: np.ndarray) -> None:
    values = np.ascontiguousarray(array)
    digest.update(name.encode("utf-8"))
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(values.shape).encode("ascii"))
    digest.update(values.tobytes())


def _numeric(group: h5py.Group, name: str, values: np.ndarray) -> h5py.Dataset:
    array = np.ascontiguousarray(values)
    chunks = (min(max(1, len(array)), 256), *array.shape[1:])
    return group.create_dataset(
        name,
        data=array,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        chunks=chunks,
    )


def _copy_human_images(
    source: h5py.Dataset,
    group: h5py.Group,
    digest: Any,
) -> tuple[str, h5py.Dataset]:
    probe_indices = np.linspace(0, len(source) - 1, num=min(8, len(source)), dtype=np.int64)
    probe = np.asarray(source[probe_indices])
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
    digest.update(b"human/images")
    digest.update(b"uint8")
    digest.update(json.dumps(source.shape).encode("ascii"))
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
        converted = np.ascontiguousarray(converted)
        destination[start:stop] = converted
        digest.update(converted.tobytes())
    return conversion, destination


def _derived_dataset_paths(path: Path) -> set[str]:
    paths: set[str] = set()
    with h5py.File(path, "r") as file:
        file.visititems(lambda name, value: paths.add(name) if isinstance(value, h5py.Dataset) else None)
    return paths


def convert_human_episode(
    source_path: Path,
    destination_path: Path,
    *,
    source_root: Path,
    task: str,
    pool_rank: int,
    protocol_sha256: str,
    split_sha256: str,
    conversion_code_sha256: str,
    overwrite: bool,
) -> dict[str, Any]:
    summary = inspect_human_source(source_path)
    relative = source_path.relative_to(source_root).as_posix()
    source_sha256 = file_sha256(source_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() and not overwrite:
        with h5py.File(destination_path, "r") as file:
            demo = file["data/demo_0"]
            _require(demo.attrs.get("schema_version") == SCHEMA_VERSION, f"Wrong existing schema: {destination_path}")
            _require(demo.attrs.get("source_sha256") == source_sha256, f"Existing source changed: {destination_path}")
            _require(
                demo.attrs.get("conversion_code_sha256") == conversion_code_sha256,
                f"P1 conversion code changed: {destination_path}",
            )
            human_content_sha256 = str(demo.attrs["human_content_sha256"])
        _require(_derived_dataset_paths(destination_path) == DERIVED_DATASETS, f"Existing P1 datasets changed: {destination_path}")
        return {
            **summary,
            "status": "reused",
            "task": task,
            "pool_rank": pool_rank,
            "source_relative_path": relative,
            "source_sha256": source_sha256,
            "conversion_code_sha256": conversion_code_sha256,
            "human_content_sha256": human_content_sha256,
            "output_path": str(destination_path),
            "output_sha256": file_sha256(destination_path),
        }

    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    digest = hashlib.sha256()
    try:
        with h5py.File(source_path, "r") as source, h5py.File(temporary, "w") as destination:
            destination.attrs["schema_version"] = SCHEMA_VERSION
            destination.attrs["human_only"] = True
            data = destination.create_group("data")
            demo = data.create_group("demo_0")
            demo.attrs["schema_version"] = SCHEMA_VERSION
            demo.attrs["task"] = task
            demo.attrs["task_split"] = "heldout"
            demo.attrs["pool_rank"] = pool_rank
            demo.attrs["source_relative_path"] = relative
            demo.attrs["source_sha256"] = source_sha256
            demo.attrs["protocol_file_sha256"] = protocol_sha256
            demo.attrs["split_sha256"] = split_sha256
            demo.attrs["conversion_code_sha256"] = conversion_code_sha256
            demo.attrs["human_only"] = True
            demo.attrs["contains_robot_observation_or_target"] = False
            demo.attrs["source_datasets_read_json"] = json.dumps(ALLOWED_SOURCE_DATASETS)
            demo.attrs["forbidden_source_datasets_read_json"] = "[]"
            demo.attrs["source_action_role"] = "human_hand_pose_in_robot_frame"
            demo.attrs["source_action_role_status"] = "verified_upstream"
            demo.attrs["position_scale_to_canonical"] = 0.001
            demo.attrs["euler_order"] = "XYZ"

            human = demo.create_group("human")
            image_conversion, _ = _copy_human_images(_dataset(source, "cam_data/human_camera"), human, digest)
            action = np.asarray(_dataset(source, "action")[:], dtype=np.float64)
            plan = poses_euler_to_10d(action[:, :6], action[:, 6], position_scale=0.001, euler_order="XYZ").astype(np.float32)
            coords = np.asarray(_dataset(source, "transformed_hand_coords")[:], dtype=np.float32)
            frames = np.asarray(_dataset(source, "transformed_hand_frames")[:], dtype=np.float32)
            _numeric(human, "hand_plan_10d", plan)
            _numeric(human, "hand_coords", coords)
            _numeric(human, "hand_frames", frames)
            _update_content_hash(digest, "human/hand_plan_10d", plan)
            _update_content_hash(digest, "human/hand_coords", coords)
            _update_content_hash(digest, "human/hand_frames", frames)

            time = demo.create_group("time")
            step = np.asarray(_dataset(source, "step")[:], dtype=np.int64)
            timestamp = np.asarray(_dataset(source, "timestamp")[:], dtype=np.int64)
            time_audit = audit_timebase(step, timestamp)
            segment_id = np.asarray(time_audit["segment_id"], dtype=np.int32)
            gap_mask = np.asarray(time_audit["gap_mask"], dtype=np.bool_)
            _numeric(time, "source_step", step)
            _numeric(time, "source_timestamp", timestamp)
            _numeric(time, "segment_id", segment_id)
            _numeric(time, "gap_mask", gap_mask)
            _update_content_hash(digest, "time/source_step", step)
            _update_content_hash(digest, "time/source_timestamp", timestamp)
            _update_content_hash(digest, "time/segment_id", segment_id)
            _update_content_hash(digest, "time/gap_mask", gap_mask)
            demo.attrs["human_image_conversion"] = image_conversion
            demo.attrs["frame_count"] = len(step)
            demo.attrs["gap_count"] = int(np.count_nonzero(gap_mask))
            demo.attrs["human_content_sha256"] = digest.hexdigest()
            destination.flush()
        os.replace(temporary, destination_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    _require(_derived_dataset_paths(destination_path) == DERIVED_DATASETS, f"Derived file leaked datasets: {destination_path}")
    return {
        **summary,
        "status": "converted",
        "task": task,
        "pool_rank": pool_rank,
        "source_relative_path": relative,
        "source_sha256": source_sha256,
        "conversion_code_sha256": conversion_code_sha256,
        "human_content_sha256": digest.hexdigest(),
        "output_path": str(destination_path),
        "output_sha256": file_sha256(destination_path),
    }


def validate_records(
    records: list[dict[str, Any]],
    *,
    heldout_tasks: list[str],
    train_tasks: list[str],
    anchors: dict[str, str],
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    counts = {task: len(by_task[task]) for task in heldout_tasks}
    _require(set(by_task) == set(heldout_tasks), "Pool contains a non-heldout task")
    _require(all(count == REQUIRED_PER_TASK for count in counts.values()), f"Wrong per-task counts: {counts}")
    _require(not set(heldout_tasks).intersection(train_tasks), "Heldout/train task overlap")
    for task, task_records in by_task.items():
        task_records.sort(key=lambda item: item["pool_rank"])
        _require([item["pool_rank"] for item in task_records] == list(range(1, 11)), f"Bad pool ranks: {task}")
        _require(task_records[0]["source_relative_path"] == anchors[task], f"Pool rank 1 changed: {task}")
        _require(all(item["max_gap_safe_segment_frames"] >= H_STEPS for item in task_records), f"No H-safe demo: {task}")
        _require(all(not item["forbidden_source_datasets_read"] for item in task_records), f"Robot data read: {task}")

    source_paths = [item["source_relative_path"] for item in records]
    source_hashes = [item["source_sha256"] for item in records]
    human_hashes = [item["human_content_sha256"] for item in records]
    output_hashes = [item["output_sha256"] for item in records]
    _require(len(set(source_paths)) == len(records), "Duplicate source episode identity")
    _require(len(set(source_hashes)) == len(records), "Duplicate source file content")
    _require(len(set(human_hashes)) == len(records), "Duplicate human demonstration content")
    _require(len(set(output_hashes)) == len(records), "Duplicate derived artifact content")
    return {
        "per_task_independent_source_episode_count": counts,
        "required_per_task": REQUIRED_PER_TASK,
        "every_task_meets_requirement": True,
        "source_episode_identity_unique_count": len(set(source_paths)),
        "source_file_sha256_unique_count": len(set(source_hashes)),
        "human_content_sha256_unique_count": len(set(human_hashes)),
        "derived_file_sha256_unique_count": len(set(output_hashes)),
        "pool_sizes_are_nested_prefixes": POOL_SIZES,
        "heldout_train_task_overlap": [],
        "heldout_robot_dataset_read_count": 0,
        "forbidden_source_datasets_read": [],
        "derived_dataset_allowlist": sorted(DERIVED_DATASETS),
        "gap_safe_H_steps": H_STEPS,
    }


def _environment(repo_root: Path) -> dict[str, Any]:
    _require(Path("/.dockerenv").exists(), "P1 must run inside Docker")
    _require(repo_root.resolve() == Path("/workspace"), "P1 repo root must resolve to /workspace")
    return {
        "inside_docker": True,
        "workspace": str(repo_root.resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "downloads_performed": False,
        "environment_sync_performed": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    environment = _environment(repo_root)
    protocol_path = (repo_root / args.protocol).resolve() if not args.protocol.is_absolute() else args.protocol
    split_path = (repo_root / args.split_manifest).resolve() if not args.split_manifest.is_absolute() else args.split_manifest
    output_root = (repo_root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    frozen = protocol["frozen_data_contract"]
    protocol_sha256 = file_sha256(protocol_path)
    conversion_code_sha256 = file_sha256(Path(__file__).resolve())
    _require(frozen["required_heldout_independent_human_demos_per_task"] == REQUIRED_PER_TASK, "P1 target changed")
    _require(frozen["independence_unit"].startswith("source episode"), "Independence unit changed")
    _require(split["split_sha256"] == frozen["split_sha256"], "Frozen split hash changed")
    heldout_tasks = sorted(split["heldout_tasks"])
    train_tasks = sorted(split["train_tasks"])
    _require(len(heldout_tasks) == 4, "Expected four heldout tasks")
    anchors = {
        task: next(
            item["source_relative_path"]
            for item in split["episodes"]
            if item["task"] == task and item["split"] == "heldout"
        )
        for task in heldout_tasks
    }
    selected, rejected, discovered = select_sources(args.source_root.resolve(), heldout_tasks, anchors)
    records: list[dict[str, Any]] = []
    for task in heldout_tasks:
        for rank, source_path in enumerate(selected[task], start=1):
            destination = output_root / "episodes" / task / f"pool_{rank:02d}_{source_path.name}"
            records.append(
                convert_human_episode(
                    source_path,
                    destination,
                    source_root=args.source_root.resolve(),
                    task=task,
                    pool_rank=rank,
                    protocol_sha256=protocol_sha256,
                    split_sha256=split["split_sha256"],
                    conversion_code_sha256=conversion_code_sha256,
                    overwrite=args.overwrite,
                )
            )
    validation = validate_records(
        records,
        heldout_tasks=heldout_tasks,
        train_tasks=train_tasks,
        anchors=anchors,
    )
    source_selection_payload = [
        {
            "task": record["task"],
            "pool_rank": record["pool_rank"],
            "source_relative_path": record["source_relative_path"],
            "source_sha256": record["source_sha256"],
            "human_content_sha256": record["human_content_sha256"],
        }
        for record in sorted(records, key=lambda item: (item["task"], item["pool_rank"]))
    ]
    selection_id = stable_json_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "selection_seed": SELECTION_SEED,
            "anchors": anchors,
            "episodes": source_selection_payload,
        }
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen",
        "created_at_utc": utc_now(),
        "source_root": str(args.source_root.resolve()),
        "protocol_file_sha256": protocol_sha256,
        "conversion_code_sha256": conversion_code_sha256,
        "split_sha256": split["split_sha256"],
        "selection_seed": SELECTION_SEED,
        "selection_policy": "frozen pilot source first, then SHA-256 rank of '<seed>:<source_relative_path>'; first valid 10",
        "selection_id": selection_id,
        "heldout_tasks": heldout_tasks,
        "anchors": anchors,
        "discovered_source_episode_count": discovered,
        "rejected_candidates": rejected,
        "allowed_source_datasets": list(ALLOWED_SOURCE_DATASETS),
        "forbidden_source_datasets": [
            "cam_data/robot_camera",
            "end_position",
            "qpos",
            "qvel",
            "gripper_state",
        ],
        "episodes": source_selection_payload,
    }
    write_json(output_root / "selection_manifest.json", manifest)
    pool_manifest = {
        **manifest,
        "status": "passed",
        "records": records,
        "validation": validation,
    }
    write_json(output_root / "pool_manifest.json", pool_manifest)
    action_stats = repo_root / (
        "data/Human2Robot/derived/views/nominal_camera_30hz_segmented/"
        "human_hand_robot_frame_raw/robot_ee_observed_t_plus_1_bc_proxy/"
        "train_only_tplus1_query_anchor_se3_identity_scale_v1/action_statistics.json"
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "gate_id": GATE_ID,
        "status": "passed",
        "completed_at_utc": utc_now(),
        "environment": environment,
        "protocol_file_sha256": protocol_sha256,
        "conversion_code_sha256": conversion_code_sha256,
        "split_sha256": split["split_sha256"],
        "selection_id": selection_id,
        "pool_manifest_path": str(output_root / "pool_manifest.json"),
        "pool_manifest_sha256": file_sha256(output_root / "pool_manifest.json"),
        "validation": validation,
        "leakage_audit": {
            "human_only_source_dataset_allowlist_enforced": True,
            "heldout_robot_dataset_read_count": 0,
            "heldout_robot_target_used_for_retrieval_features": False,
            "heldout_robot_target_used_for_normalization": False,
            "heldout_robot_target_used_for_alignment_or_lag": False,
            "heldout_robot_target_used_for_checkpoint_selection": False,
            "derived_files_contain_robot_observation_or_target": False,
            "train_statistics_recomputed": False,
            "frozen_train_action_statistics_sha256": file_sha256(action_stats),
            "source_containers_are_paired_but_only_human_allowlist_was_read": True,
        },
        "independence_audit": {
            "unit": "source episode",
            "windows_or_chunks_count_as_independent": False,
            **validation,
        },
        "claim_boundary": {
            "m5b_p1_data": "passed",
            "m5b_p2_run_completeness": "pending",
            "m5_v03": "pending",
            "gate_c": "pending",
            "m6_rollout_approved": False,
            "query_command_status": "unverified",
            "deployment_command_adapter_id": None,
        },
    }
    write_json(output_root.parent / "p1_data_acceptance_report.json", report)
    return report


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    counts = report["validation"]["per_task_independent_source_episode_count"]
    rows = "\n".join(f"| `{task}` | {count} | 10 | passed |" for task, count in counts.items())
    text = f"""# M5B-P1-DATA 验收报告

日期：{report['completed_at_utc']}

结论：**M5B-P1-DATA 通过；M5B-P2、M5-v03、Gate C 与 M6 仍未通过。**

## 独立 human demonstrations

| held-out task | 独立 source episodes | 要求 | 状态 |
|---|---:|---:|---|
{rows}

- 独立性单位是不同 source episode；window/chunk 不计为独立重复。
- 40 个 source path、source file SHA256 与 human-content SHA256 均唯一。
- pool size `0/1/2/4/8/10` 使用每 task 同一冻结排序的嵌套前缀。

## 泄漏门禁

- 源容器虽然是 paired HDF5，但提取器只读取 human camera、human action、hand coords/frames、step 与 timestamp。
- `robot_camera/end_position/qpos/qvel/gripper_state` 读取数为 0；派生 HDF5 不包含 robot observation/target。
- held-out robot target 未用于 retrieval feature、normalization、alignment、lag 或 checkpoint selection。
- train-only action statistics 未重算，冻结 SHA256 为 `{report['leakage_audit']['frozen_train_action_statistics_sha256']}`。

## 当前边界

- P1 只证明正式 held-out human-only pool 已达到 10 条/任务并通过泄漏审计，不证明模型收益。
- P2 的全部 method×experiment×3-seed step-7000 checkpoint 尚未完成，Gate C 保持 pending。
- `query_command_status=unverified`，不得用于真实机器人 rollout。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=Path("/DATA1/wxs/DATASETS/Human2Robot/data/v1"))
    parser.add_argument("--protocol", type=Path, default=Path("方案/v03/M5B_formal_acceptance_protocol_v1.json"))
    parser.add_argument("--split-manifest", type=Path, default=Path("data/Human2Robot/canonical/v3/task_split_manifest.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/Human2Robot/derived/m5b_v03/p1_human_only_pool"))
    parser.add_argument("--acceptance-json", type=Path, default=Path("方案/v03/M5B_P1_DATA_自动验收报告.json"))
    parser.add_argument("--acceptance-markdown", type=Path, default=Path("方案/v03/M5B_P1_DATA_验收报告.md"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
        repo_root = args.repo_root.resolve()
        acceptance_json = args.acceptance_json if args.acceptance_json.is_absolute() else repo_root / args.acceptance_json
        acceptance_markdown = (
            args.acceptance_markdown if args.acceptance_markdown.is_absolute() else repo_root / args.acceptance_markdown
        )
        write_json(acceptance_json, report)
        write_markdown(acceptance_markdown, report)
    except (P1Error, OSError, ValueError, KeyError, StopIteration) as exc:
        print(json.dumps({"schema_version": REPORT_SCHEMA_VERSION, "gate_id": GATE_ID, "status": "failed", "error": str(exc)}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "gate_id": GATE_ID,
                "selection_id": report["selection_id"],
                "counts": report["validation"]["per_task_independent_source_episode_count"],
                "heldout_robot_dataset_read_count": report["leakage_audit"]["heldout_robot_dataset_read_count"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
