#!/usr/bin/env python3
"""Audited stage-0 entry point for the Human2Robot v04 experiment.

This module deliberately uses only the standard library until the CUDA probe.  It
can therefore emit a useful BLOCKED_ENVIRONMENT receipt even when the full ML
stack is broken.  Formal prepare/train/evaluate commands will be added behind
this gate in their respective stages; they must not bypass ``require_preflight``.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, TextIO

SCHEMA = "human2robot-v04-stage0-v1"
WORKSPACE = Path(os.environ.get("RECAP_WORKSPACE", "/workspace")).resolve()
RUN_ROOT = Path(os.environ.get("HUMAN2ROBOT_V04_RUN_ROOT", "/DATA1/wxs/ReCAP_M5B_V04_RUNS"))
LOG_ROOT = RUN_ROOT / "orchestrator_logs"
V03_ROOT = Path("/DATA1/wxs/ReCAP_M5B_P2_RUNS")
V03_DIAGNOSTIC_ROOT = Path("/DATA1/wxs/ReCAP_M5B_P2_DIAGNOSTICS")
FREEZE_MANIFEST = Path("方案/v04/v03_frozen_manifest.json")
FREEZE_LOCK = Path("方案/v04/v03_frozen_manifest.lock.json")
ASSET_REGISTRY = Path("方案/v04/v04_asset_registry.json")
PLAN_PATH = Path("方案/v04/RECAP_Human2Robot_无泄漏单seed离线复现执行总计划.md")
RUNBOOK_PATH = Path("docs/pusht_rag_docker_runbook.md")
V03_COMMIT = "9a6aacaffcacdd71a5ff6fd3fc92bf30eb711f2a"
V03_TREE = "daee165c61a2a94df11ede5467590cb5d3610164"
MIN_FREE_BYTES = 300 * 1024**3

OFFLINE_ENV = {
    "RECAP_WORKSPACE": "/workspace",
    "HUMAN2ROBOT_ROOT": "/workspace/data/Human2Robot",
    "HUMAN2ROBOT_V04_RUN_ROOT": "/DATA1/wxs/ReCAP_M5B_V04_RUNS",
    "COSMOS_HF_CHECKPOINT_ROOT": "/DATA1/wxs/_HUGGINGFACE",
    "COSMOS_SKIP_HF_AUTO_DOWNLOAD": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

V03_ARTIFACTS = {
    "no_retrieval": V03_ROOT / "cells/learned_training_checkpoint__M5B-MAIN-01__frozen_main__no_retrieval__seed20260711/artifact.json",
    "co_training": V03_ROOT / "cells/learned_training_checkpoint__M5B-MAIN-01__frozen_main__co_training__seed20260711/artifact.json",
    "recap_hand_ret": V03_ROOT / "cells/learned_training_checkpoint__M5B-MAIN-01__frozen_main__recap_hand_ret__seed20260711/artifact.json",
    "retrieval_only": V03_ROOT / "cells/nonlearned_method_artifact__M5B-MAIN-01__retrieval_only_projection__retrieval_only__seed20260711/artifact.json",
}
FULL149_ROOT = V03_DIAGNOSTIC_ROOT / "nonformal_full149_recap_hand_ret_seed20260711_iter7000_20260721"
FULL149_FILES = [
    FULL149_ROOT / "iter_000007000.json",
    FULL149_ROOT / "comparison_recap7000_vs_main_checkpoints.json",
    FULL149_ROOT / "COMPARISON_REPORT_RECAP7000_VS_MAIN_CHECKPOINTS.md",
]
V03_REPO_EVIDENCE = [
    Path("方案/v03/human2robot_task_split_manifest_v3.json"),
    Path("data/Human2Robot/derived/m5b_v03/p1_human_only_pool/pool_manifest.json"),
    Path("data/Human2Robot/derived/m5b_v03/p1_human_only_pool/selection_manifest.json"),
    Path("方案/v03/M5B_formal_acceptance_protocol_v1.md"),
    Path("方案/v03/M5B_formal_acceptance_protocol_v1.lock.json"),
    Path("方案/v03/M5B_P2_cell_registry_v2.json"),
    Path("方案/v03/M5B_P2_cell_registry_v2.lock.json"),
    Path("方案/v03/source_evidence_manifest_v3.json"),
    Path("方案/v03/m2_v02_frozen_code_manifest.json"),
]
V04_CONTROLLED_SOURCE = [
    Path("tools/human2robot_v04.py"),
    Path("tools/human2robot_v04_test.py"),
    Path("start_human2robot_v04_docker.sh"),
    ASSET_REGISTRY,
    PLAN_PATH,
    RUNBOOK_PATH,
]
FORBIDDEN_PROCESS_TOKENS = (
    "human2robot_m5b_p2_dag run-cell",
    "successor_watchdog",
    "human2robot_m5b_p2_inference",
    "torchrun",
)


class V04Error(RuntimeError):
    """A hard audit or environment failure."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V04Error(f"Expected a JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Any, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        json.loads(tmp.read_text(encoding="utf-8"))
        os.replace(tmp, path)
        if mode is not None:
            path.chmod(mode)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise V04Error(f"Required file is missing: {resolved}")
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": file_sha256(resolved)}


def tree_stats(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise V04Error(f"Required directory is missing: {path}")
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "path": str(path.resolve()),
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
    }


def run_capture(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": shlex.join(command),
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def active_v03_processes() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            command = (item / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command and any(token in command for token in FORBIDDEN_PROCESS_TOKENS):
            found.append({"pid": int(item.name), "command": command})
    return sorted(found, key=lambda row: row["pid"])


def _checkpoint_binding(method: str, artifact: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(str(artifact["checkpoint_manifest_path"]))
    manifest = read_json(manifest_path)
    checkpoint_path = Path(str(artifact["checkpoint_path"]))
    expected_payload = str(artifact["model_payload_sha256"])
    if manifest.get("primary_checkpoint_path") != str(checkpoint_path):
        raise V04Error(f"{method}: checkpoint path differs between artifact and manifest")
    if manifest.get("primary_checkpoint_payload_sha256") != expected_payload:
        raise V04Error(f"{method}: checkpoint payload SHA differs between artifact and manifest")
    return {
        "method": method,
        "seed": 20260711,
        "optimizer_step": 7000,
        "path": str(checkpoint_path),
        "payload_sha256": expected_payload,
        "tree": tree_stats(checkpoint_path),
        "manifest": bind_file(manifest_path),
        "artifact": bind_file(V03_ARTIFACTS[method]),
    }


def collect_v03_freeze(workspace: Path) -> dict[str, Any]:
    git_commit = run_capture(["git", "rev-parse", V03_COMMIT], cwd=workspace)
    git_tree = run_capture(["git", "show", "-s", "--format=%T", V03_COMMIT], cwd=workspace)
    if git_commit["exit_code"] or git_commit["stdout"] != V03_COMMIT:
        raise V04Error("Historical v03 commit is unavailable or changed")
    if git_tree["exit_code"] or git_tree["stdout"] != V03_TREE:
        raise V04Error("Historical v03 tree binding changed")

    artifacts = {name: read_json(path) for name, path in V03_ARTIFACTS.items()}
    completed = [value for value in artifacts.values() if value.get("status") == "completed"]
    learned = [value for value in completed if value.get("artifact_kind") == "learned_training_checkpoint"]
    if len(completed) != 4 or len(learned) != 3:
        raise V04Error("v03 completion state is not the frozen 4/203 with 3/48 learned cells")
    processes = active_v03_processes()
    if processes:
        raise V04Error(f"Active v03/torchrun processes prevent freeze: {processes}")

    checkpoints = [_checkpoint_binding(name, artifacts[name]) for name in ("no_retrieval", "co_training", "recap_hand_ret")]
    retrieval = artifacts["retrieval_only"]
    if retrieval.get("dataset_contract", {}).get("query_count") != 149:
        raise V04Error("Retrieval-only artifact no longer binds 149 queries")

    comparison = read_json(FULL149_FILES[1])
    if comparison.get("status") != "completed" or not comparison.get("comparability", {}).get("same_149_heldout_queries"):
        raise V04Error("Latest full-149 comparison is missing or incomplete")
    disk = shutil.disk_usage("/DATA1")
    return {
        "schema_version": f"{SCHEMA}-v03-freeze",
        "status": "LEGACY_ORACLE_PHASE_PILOT",
        "generated_at_utc": utc_now(),
        "mutation_allowed": False,
        "claim_boundary": "Historical phase-oracle pilot only; forbidden for new-task generalization claims.",
        "git": {"commit": V03_COMMIT, "tree": V03_TREE},
        "execution_state": {
            "completed_cells": 4,
            "total_cells": 203,
            "completed_training_cells": 3,
            "total_training_cells": 48,
            "completed_formal_evaluations": 0,
            "total_formal_evaluations": 147,
            "active_processes": processes,
            "automatic_resume_allowed": False,
        },
        "checkpoints": checkpoints,
        "retrieval_only": {
            "artifact": bind_file(V03_ARTIFACTS["retrieval_only"]),
            "artifact_payload_sha256": retrieval.get("artifact_payload_sha256"),
            "query_count": 149,
        },
        "latest_nonformal_full149": {
            "method": "recap_hand_ret",
            "seed": 20260711,
            "optimizer_step": 7000,
            "query_count": 149,
            "rank_inference_count": 447,
            "metrics_mean_lower_is_better": {
                "canonical": 0.02380439,
                "final_position": 0.01316358,
                "gripper": 0.01685420,
                "orientation": 0.00039520,
                "position": 0.00779624,
                "residual": 0.02380439,
            },
            "formal_result": False,
            "files": [bind_file(path) for path in FULL149_FILES],
        },
        "data_protocol_code": [bind_file(workspace / path) for path in V03_REPO_EVIDENCE],
        "storage_snapshot": {
            "captured_at_utc": utc_now(),
            "filesystem": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
            "v03_run_root": tree_stats(V03_ROOT),
            "v03_diagnostic_root": tree_stats(V03_DIAGNOSTIC_ROOT),
        },
    }


def freeze_v03(workspace: Path) -> dict[str, Any]:
    manifest_path = workspace / FREEZE_MANIFEST
    lock_path = workspace / FREEZE_LOCK
    if manifest_path.exists() or lock_path.exists():
        return verify_v03(workspace)
    manifest = collect_v03_freeze(workspace)
    write_json_atomic(manifest_path, manifest, mode=0o444)
    lock = {
        "schema_version": f"{SCHEMA}-v03-freeze-lock",
        "status": "locked_read_only",
        "manifest_path": str(FREEZE_MANIFEST),
        "manifest_sha256": file_sha256(manifest_path),
        "mutation_allowed": False,
    }
    write_json_atomic(lock_path, lock, mode=0o444)
    marker = {
        "schema_version": f"{SCHEMA}-v03-stop-marker",
        "status": "FROZEN_DO_NOT_RESUME",
        "created_at_utc": utc_now(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": lock["manifest_sha256"],
        "old_outputs_deleted": False,
    }
    write_json_atomic(RUN_ROOT / "stage0/V03_FROZEN_DO_NOT_RESUME.json", marker, mode=0o444)
    return verify_v03(workspace)


def verify_v03(workspace: Path) -> dict[str, Any]:
    manifest_path = workspace / FREEZE_MANIFEST
    lock_path = workspace / FREEZE_LOCK
    blockers: list[str] = []
    if not manifest_path.is_file() or not lock_path.is_file():
        return {"status": "blocked", "blockers": ["v03_freeze_manifest_or_lock_missing"]}
    manifest = read_json(manifest_path)
    lock = read_json(lock_path)
    if lock.get("manifest_sha256") != file_sha256(manifest_path):
        blockers.append("v03_freeze_manifest_hash_changed")
    for group in (manifest.get("data_protocol_code", []), manifest.get("latest_nonformal_full149", {}).get("files", [])):
        for binding in group:
            path = Path(str(binding["path"]))
            if not path.is_file() or path.stat().st_size != binding["size_bytes"] or file_sha256(path) != binding["sha256"]:
                blockers.append(f"frozen_file_changed:{path}")
    for checkpoint in manifest.get("checkpoints", []):
        for key in ("artifact", "manifest"):
            binding = checkpoint[key]
            path = Path(str(binding["path"]))
            if not path.is_file() or file_sha256(path) != binding["sha256"]:
                blockers.append(f"frozen_checkpoint_evidence_changed:{path}")
        current_tree = tree_stats(Path(str(checkpoint["path"])))
        if current_tree["file_count"] != checkpoint["tree"]["file_count"] or current_tree["total_bytes"] != checkpoint["tree"]["total_bytes"]:
            blockers.append(f"frozen_checkpoint_tree_changed:{checkpoint['method']}")
    retrieval_binding = manifest.get("retrieval_only", {}).get("artifact", {})
    retrieval_path = Path(str(retrieval_binding.get("path", "")))
    if not retrieval_path.is_file() or file_sha256(retrieval_path) != retrieval_binding.get("sha256"):
        blockers.append("frozen_retrieval_only_artifact_changed")
    marker_path = RUN_ROOT / "stage0/V03_FROZEN_DO_NOT_RESUME.json"
    if not marker_path.is_file():
        blockers.append("v03_stop_marker_missing")
    else:
        marker = read_json(marker_path)
        if marker.get("status") != "FROZEN_DO_NOT_RESUME" or marker.get("manifest_sha256") != lock.get("manifest_sha256"):
            blockers.append("v03_stop_marker_invalid")
    processes = active_v03_processes()
    if processes:
        blockers.append("active_v03_or_training_processes")
    return {
        "schema_version": f"{SCHEMA}-v03-verification",
        "checked_at_utc": utc_now(),
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "active_processes": processes,
    }


def mount_binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    best: dict[str, Any] | None = None
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return {"path": str(resolved), "status": "failed", "error": str(error)}
    for line in lines:
        left, _, right = line.partition(" - ")
        fields = left.split()
        if len(fields) < 6:
            continue
        mount_point = Path(fields[4].replace("\\040", " "))
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        candidate = {
            "path": str(resolved),
            "mount_point": str(mount_point),
            "mount_options": fields[5].split(","),
            "filesystem": right.split()[0] if right else None,
        }
        if best is None or len(str(mount_point)) > len(str(best["mount_point"])):
            best = candidate
    if best is None:
        return {"path": str(resolved), "status": "failed", "error": "mount_not_found"}
    best["mount_read_write"] = "rw" in best["mount_options"]
    best["path_writable_by_current_user"] = os.access(resolved, os.W_OK)
    # A bind such as /DATA1 can be mounted rw while its top-level directory is
    # intentionally not writable by the experiment user.  The dedicated run
    # root is checked with an actual create/read/delete probe separately.
    best["writable"] = best["mount_read_write"]
    best["status"] = "passed" if best["mount_read_write"] else "failed"
    return best


def _probe_write(path: Path) -> dict[str, Any]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".v04-write-probe-{os.getpid()}-{uuid.uuid4().hex}"
        probe.write_bytes(b"v04")
        with probe.open("rb") as stream:
            valid = stream.read() == b"v04"
        probe.unlink()
    except OSError as error:
        return {"path": str(path), "status": "failed", "error": str(error)}
    return {"path": str(path), "status": "passed" if valid else "failed"}


def _import_probe(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as error:  # compiled extensions can fail with non-ImportError exceptions
        return {"module": name, "status": "failed", "error": f"{type(error).__name__}: {error}"}
    return {"module": name, "status": "passed", "version": getattr(module, "__version__", None)}


def _gpu_probe() -> dict[str, Any]:
    result: dict[str, Any] = {"status": "failed", "logical_devices": []}
    try:
        import torch

        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        devices = []
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            devices.append({
                "logical_index": index,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "uuid": str(getattr(props, "uuid", "unavailable")),
            })
        result.update({
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_available": torch.distributed.is_nccl_available(),
            "nccl_version": list(torch.cuda.nccl.version()) if torch.cuda.is_available() else None,
            "logical_device_count": count,
            "logical_devices": devices,
        })
        result["status"] = "passed" if count == 4 and result["nccl_available"] else "failed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    host_ids = os.environ.get("HUMAN2ROBOT_V04_GPU_DEVICES", "").split(",")
    result["host_physical_indices"] = [value for value in host_ids if value]
    return result


def _nccl_worker() -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    value = torch.tensor([float(rank)], device=f"cuda:{local_rank}")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size - 1) / 2
    if value.item() != expected:
        raise V04Error(f"NCCL all-reduce mismatch on rank {rank}: {value.item()} != {expected}")
    dist.barrier()
    dist.destroy_process_group()
    print(json.dumps({"rank": rank, "local_rank": local_rank, "all_reduce": value.item()}))
    return 0


def _nccl_collective_probe(workspace: Path) -> dict[str, Any]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        master_port = listener.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc-per-node=4",
        "--master-addr=127.0.0.1",
        f"--master-port={master_port}",
        str(workspace / "tools/human2robot_v04.py"),
        "_nccl-worker",
    ]
    environment = {**os.environ, "NCCL_SOCKET_IFNAME": "lo", "GLOO_SOCKET_IFNAME": "lo"}
    try:
        result = subprocess.run(command, cwd=workspace, env=environment, text=True, capture_output=True, timeout=120, check=False)
    except Exception as error:
        return {"status": "failed", "command": shlex.join(command), "error": f"{type(error).__name__}: {error}"}
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "command": shlex.join(command),
        "exit_code": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def _asset_checks(workspace: Path) -> list[dict[str, Any]]:
    registry_path = workspace / ASSET_REGISTRY
    if not registry_path.is_file():
        return [{"status": "failed", "error": f"asset_registry_missing:{registry_path}"}]
    registry = read_json(registry_path)
    checks = []
    for asset in registry.get("assets", []):
        path = Path(str(asset["path"]))
        check = {"id": asset["id"], "path": str(path), "expected_size_bytes": asset["size_bytes"], "expected_sha256": asset["sha256"]}
        if not path.is_file() or not os.access(path, os.R_OK):
            check.update({"status": "failed", "error": "missing_or_unreadable"})
        elif path.stat().st_size != asset["size_bytes"]:
            check.update({"status": "failed", "error": "size_mismatch", "actual_size_bytes": path.stat().st_size})
        else:
            actual = file_sha256(path)
            check.update({"actual_sha256": actual, "status": "passed" if actual == asset["sha256"] else "failed"})
        checks.append(check)
    return checks


def build_session_receipt(workspace: Path) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA}-docker-session",
        "created_at_utc": utc_now(),
        "container": {
            "inside_docker": Path("/.dockerenv").is_file(),
            "hostname": platform.node(),
            "image": os.environ.get("HUMAN2ROBOT_V04_IMAGE", ""),
            "image_id": os.environ.get("HUMAN2ROBOT_V04_IMAGE_ID", ""),
            "container_name": os.environ.get("HUMAN2ROBOT_V04_CONTAINER_NAME", ""),
        },
        "user": {
            "container_uid": os.getuid(),
            "container_gid": os.getgid(),
            "host_uid": os.environ.get("HOST_USER_ID"),
            "host_gid": os.environ.get("HOST_GROUP_ID"),
        },
        "mounts": {"workspace": mount_binding(workspace), "data1": mount_binding(Path("/DATA1"))},
        "gpu_mapping": _gpu_probe(),
        "controlled_source": [bind_file(workspace / path) for path in V04_CONTROLLED_SOURCE],
    }


def build_preflight(workspace: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    checks["docker_session"] = build_session_receipt(workspace)
    container = checks["docker_session"]["container"]
    if not container["inside_docker"]:
        blockers.append("not_inside_docker")
    if container["image"] != "cosmos-policy:latest" or not str(container["image_id"]).startswith("sha256:"):
        blockers.append("docker_image_identity_not_bound")
    if str(workspace) != "/workspace":
        blockers.append("workspace_not_/workspace")
    if sys.version_info[:2] != (3, 10):
        blockers.append("python_is_not_3.10")
    expected_python = workspace / ".venv/bin/python"
    checks["python"] = {"executable": sys.executable, "version": platform.python_version(), "expected": str(expected_python)}
    if Path(sys.executable).resolve() != expected_python.resolve():
        blockers.append("not_using_/workspace/.venv")

    checks["offline_environment"] = {key: {"expected": value, "actual": os.environ.get(key)} for key, value in OFFLINE_ENV.items()}
    blockers.extend(f"offline_env_mismatch:{key}" for key, value in OFFLINE_ENV.items() if os.environ.get(key) != value)
    checks["mounts"] = checks["docker_session"]["mounts"]
    if not checks["mounts"]["workspace"].get("writable"):
        blockers.append("workspace_mount_not_writable")
    if not checks["mounts"]["data1"].get("writable"):
        blockers.append("data1_mount_not_writable")
    checks["run_root_write"] = _probe_write(RUN_ROOT)
    if checks["run_root_write"]["status"] != "passed":
        blockers.append("v04_run_root_not_writable")
    if not str(RUN_ROOT.resolve()).startswith("/DATA1/wxs/ReCAP_M5B_V04_RUNS"):
        blockers.append("v04_run_root_outside_allowed_boundary")

    disk = shutil.disk_usage("/DATA1")
    checks["storage"] = {"path": "/DATA1", "total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free, "minimum_free_bytes": MIN_FREE_BYTES}
    checks["storage"]["status"] = "passed" if disk.free >= MIN_FREE_BYTES else "failed"
    if checks["storage"]["status"] != "passed":
        blockers.append("data1_free_space_below_300_gib")

    human2robot = Path(os.environ.get("HUMAN2ROBOT_ROOT", ""))
    checks["data"] = {"root": str(human2robot), "exists": human2robot.is_dir(), "readable": os.access(human2robot, os.R_OK)}
    checks["data"]["status"] = "passed" if checks["data"]["exists"] and checks["data"]["readable"] else "failed"
    if checks["data"]["status"] != "passed":
        blockers.append("human2robot_data_missing_or_unreadable")

    checks["weights"] = _asset_checks(workspace)
    if not checks["weights"] or any(item["status"] != "passed" for item in checks["weights"]):
        blockers.append("weight_asset_check_failed")
    checks["compiled_extensions"] = [_import_probe(name) for name in ("flash_attn", "natten", "transformer_engine.pytorch")]
    if any(item["status"] != "passed" for item in checks["compiled_extensions"]):
        blockers.append("compiled_extension_import_failed")
    checks["gpu_cuda_nccl"] = checks["docker_session"]["gpu_mapping"]
    if checks["gpu_cuda_nccl"]["status"] != "passed":
        blockers.append("expected_four_cuda_gpus_with_nccl")
    physical = checks["gpu_cuda_nccl"].get("host_physical_indices", [])
    if len(physical) != 4 or len(set(physical)) != 4:
        blockers.append("host_gpu_mapping_not_four_unique_indices")
    checks["nccl_four_rank_collective"] = _nccl_collective_probe(workspace) if checks["gpu_cuda_nccl"]["status"] == "passed" else {"status": "skipped"}
    if checks["nccl_four_rank_collective"]["status"] != "passed":
        blockers.append("nccl_four_rank_all_reduce_failed")

    checks["v03_freeze"] = verify_v03(workspace)
    if checks["v03_freeze"]["status"] != "passed":
        blockers.append("v03_freeze_verification_failed")
    docs = [bind_file(workspace / path) for path in (PLAN_PATH, RUNBOOK_PATH)]
    checks["documents"] = docs
    blockers = list(dict.fromkeys(blockers))
    return {
        "schema_version": f"{SCHEMA}-preflight",
        "generated_at_utc": utc_now(),
        "status": "PASSED" if not blockers else "BLOCKED_ENVIRONMENT",
        "formal_v04_allowed": not blockers,
        "blockers": blockers,
        "checks": checks,
        "document_hashes": {Path(item["path"]).name: item["sha256"] for item in docs},
    }


class Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _attempt_paths(run_id: str) -> tuple[Path, int, dict[str, Path]]:
    run_dir = LOG_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts = [int(path.name.split("_")[1].split(".")[0]) for path in run_dir.glob("attempt_*.status.json")]
    attempt = max(attempts, default=0) + 1
    stem = run_dir / f"attempt_{attempt:04d}"
    return run_dir, attempt, {
        "log": Path(f"{stem}.log"),
        "command": Path(f"{stem}.command.txt"),
        "runtime": Path(f"{stem}.runtime.json"),
        "status": Path(f"{stem}.status.json"),
        "progress": Path(f"{stem}.progress.json"),
        "receipt": Path(f"{stem}.receipt.json"),
    }


def execute_with_audit(args: argparse.Namespace) -> int:
    run_id = args.run_id or f"{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}_{args.command}_{uuid.uuid4().hex[:8]}"
    run_dir, attempt, paths = _attempt_paths(run_id)
    started = utc_now()
    paths["command"].write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    runtime = {
        "schema_version": f"{SCHEMA}-runtime",
        "run_id": run_id,
        "attempt": attempt,
        "command": args.command,
        "started_at_utc": started,
        "pid": os.getpid(),
        "workspace": str(WORKSPACE),
        "run_root": str(RUN_ROOT),
        "python": sys.executable,
    }
    write_json_atomic(paths["runtime"], runtime)
    running = {"status": "RUNNING", "run_id": run_id, "attempt": attempt, "updated_at_utc": started, "log_path": str(paths["log"])}
    write_json_atomic(paths["status"], running)
    write_json_atomic(paths["progress"], {**running, "completed_units": 0, "total_units": 1})
    write_json_atomic(run_dir / "latest_log.json", {"run_id": run_id, "attempt": attempt, "log_path": str(paths["log"])})
    write_json_atomic(run_dir / "status.json", running)
    write_json_atomic(run_dir / "progress.json", {**running, "completed_units": 0, "total_units": 1})

    exit_code = 1
    receipt: dict[str, Any] = {}
    with paths["log"].open("w", encoding="utf-8", buffering=1) as log_stream:
        with contextlib.redirect_stdout(Tee(sys.__stdout__, log_stream)), contextlib.redirect_stderr(Tee(sys.__stderr__, log_stream)):
            print(json.dumps({"event": "started", "run_id": run_id, "attempt": attempt, "log_path": str(paths["log"]), "started_at_utc": started}, ensure_ascii=False))
            try:
                if args.command == "freeze-v03":
                    receipt = freeze_v03(WORKSPACE)
                elif args.command == "verify-v03":
                    receipt = verify_v03(WORKSPACE)
                elif args.command == "session-receipt":
                    receipt = build_session_receipt(WORKSPACE)
                    receipt["status"] = "passed" if receipt["container"]["inside_docker"] else "blocked"
                elif args.command == "preflight":
                    receipt = build_preflight(WORKSPACE)
                else:
                    raise V04Error(f"Unsupported command: {args.command}")
                exit_code = 0 if receipt.get("status") in ("passed", "PASSED") else 2
                print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
            except Exception as error:
                receipt = {"status": "FAILED", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
                print(receipt["traceback"], file=sys.stderr)
                exit_code = 1
    write_json_atomic(paths["receipt"], receipt)
    finished = utc_now()
    final_status = {
        "status": "COMPLETED" if exit_code == 0 else ("BLOCKED_ENVIRONMENT" if exit_code == 2 else "FAILED"),
        "run_id": run_id,
        "attempt": attempt,
        "exit_code": exit_code,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "log_path": str(paths["log"]),
        "receipt_path": str(paths["receipt"]),
        "receipt_sha256": file_sha256(paths["receipt"]),
    }
    progress = {**final_status, "completed_units": 1, "total_units": 1}
    write_json_atomic(paths["status"], final_status)
    write_json_atomic(paths["progress"], progress)
    write_json_atomic(run_dir / "status.json", final_status)
    write_json_atomic(run_dir / "progress.json", progress)
    print(json.dumps({"event": "finished", **final_status}, ensure_ascii=False))
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-v03", "verify-v03", "session-receipt", "preflight"))
    parser.add_argument("--run-id", help="Reuse a run id to create the next immutable attempt")
    return parser


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "_nccl-worker":
        return _nccl_worker()
    return execute_with_audit(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
