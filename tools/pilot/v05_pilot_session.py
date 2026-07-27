#!/usr/bin/env python3
"""v05 探索期容器内 session 回执与 preflight（E0.4 的容器侧部分）。

与 `tools/pilot/` 下 d01–d16 的区别：那些是类别 A 只读诊断，跑在宿主机、写 /tmp；
本文件是**探索期运行基建**，只在容器内运行，写探索期运行根。

为什么不复用 `tools/human2robot_v04.py session-receipt / preflight`：
它把回执写进 `/DATA1/wxs/ReCAP_M5B_V04_RUNS`，而该路径在探索期声明为只读
（总计划 §2.3）。本文件是其探索期对应物，只写 `V05_PILOT_RUN_ROOT`。

不纳入 `tools/human2robot_v04_experiment.py::_controlled_bindings()` 哈希绑定。
仅依赖标准库。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "human2robot-v05-pilot"
MIN_FREE_BYTES = 300 * 1024**3
PINNED_IMAGE_ID = "sha256:4fc8db9f70eeb96fee271ef282385163ec1da220dfed35da9c832fb6769891e8"

# 总计划 §2.2 的容器环境。与 v04 的 OFFLINE_ENV 有两处**有意不同**：
#   HUMAN2ROBOT_ROOT 指向原始只读数据（v04 指向仓库内派生目录）；
#   运行根为 V05_PILOT_RUN_ROOT（v04 为 HUMAN2ROBOT_V04_RUN_ROOT）。
PILOT_OFFLINE_ENV = {
    "RECAP_WORKSPACE": "/workspace",
    "HUMAN2ROBOT_ROOT": "/DATA1/wxs/DATASETS/Human2Robot/data/v1",
    "V05_PILOT_RUN_ROOT": "/DATA1/wxs/ReCAP_M5B_V05_PILOT",
    "COSMOS_HF_CHECKPOINT_ROOT": "/DATA1/wxs/_HUGGINGFACE",
    "COSMOS_SKIP_HF_AUTO_DOWNLOAD": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

# 探索期必须只读的路径（§2.3）。launcher 以 ro 覆盖挂载，这里验证约束真的生效。
READ_ONLY_PATHS = (
    "/DATA1/wxs/DATASETS/Human2Robot/data/v1",
    "/DATA1/wxs/ReCAP_M5B_P2_RUNS",
    "/DATA1/wxs/ReCAP_M5B_V04_RUNS",
    "/DATA1/wxs/_HUGGINGFACE",
)

# §2.3 模型资产登记。required=False 的缺失只记 warning，不 block——
# B5 只需要 SAM，E3 才需要 DINOv3。
MODEL_ASSETS = (
    ("cosmos_predict2p5_2b_init", "/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/base/post-trained", True),
    ("cosmos_tokenizer", "/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth", True),
    ("sam_vit_h", "/DATA1/wxs/_HUGGINGFACE/sam/sam_vit_h_4b8939/sam_vit_h_4b8939.pth", False),
    ("dinov3", "/DATA1/wxs/_HUGGINGFACE/facebook/dinov3", False),
    ("dinov2", "/DATA1/wxs/_HUGGINGFACE/facebook/dinov2", False),
    ("vjepa2_vitl", "/DATA1/wxs/_HUGGINGFACE/facebook/vjepa2-vitl-fpc64-256", False),
    ("depth_anything", "/DATA1/wxs/_HUGGINGFACE/depth-anything", False),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json_atomic(path: Path, value: Any) -> None:
    """先写同目录 .partial，flush + fsync 后原子 rename（§2.6）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class Tee:
    """同时写终端与日志文件，行缓冲（§2.5 禁止只在终端显示或延迟落盘）。"""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, payload: str) -> int:
        for stream in self._streams:
            stream.write(payload)
            stream.flush()
        return len(payload)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _path_is_writable(path: Path) -> bool:
    """实际尝试写入——`os.access` 在 ro bind mount 上会给出错误答案。"""
    if not path.is_dir():
        return False
    probe = path / f".v05_pilot_write_probe_{os.getpid()}_{uuid.uuid4().hex}"
    try:
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


def mount_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "writable": _path_is_writable(path),
    }


def _gpu_probe() -> dict[str, Any]:
    """容器内视角。宿主机编号 → UUID 的权威映射由 launcher 经环境变量传入。"""
    host_map: Any = []
    raw = os.environ.get("V05_PILOT_GPU_MAP_JSON", "").strip()
    if raw:
        try:
            host_map = json.loads(raw)
        except json.JSONDecodeError:
            host_map = {"parse_error": raw}

    container_visible: Any = None
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if out.returncode == 0:
            container_visible = [
                {
                    "container_index": parts[0].strip(),
                    "uuid": parts[1].strip(),
                    "name": parts[2].strip(),
                    "memory_total_mib": parts[3].strip(),
                }
                for parts in (line.split(",") for line in out.stdout.strip().splitlines() if line.strip())
                if len(parts) >= 4
            ]
    except (OSError, ValueError):
        container_visible = None

    return {
        "host_mapping": host_map,
        "container_visible": container_visible,
        "requested_devices": os.environ.get("V05_PILOT_GPU_DEVICES", ""),
    }


def build_session_receipt(workspace: Path) -> dict[str, Any]:
    run_root = Path(os.environ.get("V05_PILOT_RUN_ROOT", "/DATA1/wxs/ReCAP_M5B_V05_PILOT"))
    return {
        "schema_version": f"{SCHEMA}-docker-session",
        "created_at_utc": utc_now(),
        "run_id": os.environ.get("V05_PILOT_RUN_ID", ""),
        "attempt": os.environ.get("V05_PILOT_ATTEMPT", ""),
        "container": {
            "inside_docker": Path("/.dockerenv").is_file(),
            "hostname": platform.node(),
            "image": os.environ.get("V05_PILOT_IMAGE", ""),
            "image_id": os.environ.get("V05_PILOT_IMAGE_ID", ""),
            "pinned_image_id": PINNED_IMAGE_ID,
            "image_id_matches_pin": os.environ.get("V05_PILOT_IMAGE_ID", "") == PINNED_IMAGE_ID,
            "container_name": os.environ.get("V05_PILOT_CONTAINER_NAME", ""),
            "network_mode": os.environ.get("V05_PILOT_NETWORK_MODE", ""),
        },
        "user": {
            "container_uid": os.getuid(),
            "container_gid": os.getgid(),
            "host_uid": os.environ.get("HOST_USER_ID"),
            "host_gid": os.environ.get("HOST_GROUP_ID"),
        },
        "mounts": {
            "workspace": mount_binding(workspace),
            "data1": mount_binding(Path("/DATA1")),
            "pilot_run_root": mount_binding(run_root),
            "read_only_paths": {path: mount_binding(Path(path)) for path in READ_ONLY_PATHS},
        },
        "gpu_mapping": _gpu_probe(),
    }


def build_preflight(workspace: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    warnings: list[str] = []

    checks["docker_session"] = build_session_receipt(workspace)
    container = checks["docker_session"]["container"]

    if not container["inside_docker"]:
        blockers.append("not_inside_docker")
    if container["image"] != "cosmos-policy:latest" or not str(container["image_id"]).startswith("sha256:"):
        blockers.append("docker_image_identity_not_bound")
    if not container["image_id_matches_pin"]:
        # launcher 已要求显式放行；这里仍记为 warning，提醒必须登记进 PILOT_LOG。
        warnings.append("image_id_differs_from_pinned_value:must_be_registered_in_PILOT_LOG")
    if container["network_mode"] not in ("none", ""):
        warnings.append(f"network_enabled:{container['network_mode']}:must_be_registered_in_PILOT_LOG")

    if str(workspace) != "/workspace":
        blockers.append("workspace_not_/workspace")
    if sys.version_info[:2] != (3, 10):
        blockers.append("python_is_not_3.10")
    expected_python = workspace / ".venv/bin/python"
    checks["python"] = {
        "executable": sys.executable,
        "version": platform.python_version(),
        "expected": str(expected_python),
    }
    try:
        if Path(sys.executable).resolve() != expected_python.resolve():
            blockers.append("not_using_/workspace/.venv")
    except OSError:
        blockers.append("not_using_/workspace/.venv")

    checks["offline_environment"] = {
        key: {"expected": value, "actual": os.environ.get(key)} for key, value in PILOT_OFFLINE_ENV.items()
    }
    blockers.extend(
        f"offline_env_mismatch:{key}" for key, value in PILOT_OFFLINE_ENV.items() if os.environ.get(key) != value
    )

    mounts = checks["docker_session"]["mounts"]
    checks["mounts"] = mounts
    if not mounts["workspace"]["writable"]:
        blockers.append("workspace_mount_not_writable")
    if not mounts["pilot_run_root"]["writable"]:
        blockers.append("pilot_run_root_not_writable")
    # 只读约束必须真的生效——探索期唯一写入边界是 pilot run root（§2.3）。
    for path, binding in mounts["read_only_paths"].items():
        if not binding["exists"]:
            blockers.append(f"read_only_path_missing:{path}")
        elif binding["writable"]:
            blockers.append(f"read_only_path_is_writable:{path}")

    disk = shutil.disk_usage("/DATA1")
    checks["storage"] = {
        "path": "/DATA1",
        "total_bytes": disk.total,
        "used_bytes": disk.used,
        "free_bytes": disk.free,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "status": "passed" if disk.free >= MIN_FREE_BYTES else "failed",
    }
    if disk.free < MIN_FREE_BYTES:
        blockers.append("insufficient_free_space_on_/DATA1")
    elif disk.free < MIN_FREE_BYTES + 50 * 1024**3:
        warnings.append(
            f"free_space_margin_below_50GiB:{disk.free // 1024**3}GiB:"
            "E4 前必须按总计划 §四 E0.2 从非本课题目录追加清理"
        )

    assets: dict[str, Any] = {}
    for name, path_str, required in MODEL_ASSETS:
        path = Path(path_str)
        exists = path.exists()
        size = None
        if exists:
            try:
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else path.stat().st_size
            except OSError:
                size = None
        assets[name] = {"path": path_str, "exists": exists, "size_bytes": size, "required": required}
        if not exists:
            (blockers if required else warnings).append(f"model_asset_missing:{name}")
    checks["model_assets"] = assets

    return {
        "schema_version": f"{SCHEMA}-preflight",
        "created_at_utc": utc_now(),
        "status": "PASSED" if not blockers else "BLOCKED_ENVIRONMENT",
        "formal_result": False,
        "performance_claim_allowed": False,
        "blockers": blockers,
        "warnings": warnings,
        "checks": checks,
    }


def _attempt_paths(run_root: Path, run_id: str) -> dict[str, Any]:
    """按 §2.5 的固定布局建立 attempt 记录。

    每个命令使用**独立的 run_id**（launcher 传入 `<session>_session_receipt` /
    `<session>_preflight`），因此 attempt 文件名保持规范要求的 `attempt_000N.*`，
    且不会覆盖 launcher 在 session run 目录下的 `latest_log.json`。
    """
    run_dir = run_root / "orchestrator_logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # 重试必须新建整套 attempt_000N，不得覆盖或追加进旧 attempt（§2.5）。
    attempt = 1
    while (run_dir / f"attempt_{attempt:04d}.runtime.json").exists():
        attempt += 1
    tag = f"attempt_{attempt:04d}"
    return {
        "dir": run_dir,
        "attempt": attempt,
        "log": run_dir / f"{tag}.log",
        "command_txt": run_dir / f"{tag}.command.txt",
        "runtime": run_dir / f"{tag}.runtime.json",
        "status": run_dir / f"{tag}.status.json",
        "progress": run_dir / f"{tag}.progress.json",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v05 pilot container session receipt / preflight")
    parser.add_argument("command", choices=("session-receipt", "preflight"))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    workspace = Path(os.environ.get("RECAP_WORKSPACE", "/workspace"))
    run_root = Path(os.environ.get("V05_PILOT_RUN_ROOT", "/DATA1/wxs/ReCAP_M5B_V05_PILOT"))
    # 默认在 session run_id 上加命令后缀，避免与 launcher 的 session run 目录冲突。
    session_run_id = os.environ.get("V05_PILOT_RUN_ID", "")
    default_run_id = (
        f"{session_run_id}_{args.command.replace('-', '_')}"
        if session_run_id
        else f"pilot_{args.command}_{uuid.uuid4().hex[:12]}"
    )
    run_id = args.run_id or default_run_id

    try:
        paths = _attempt_paths(run_root, run_id)
    except OSError as error:
        print(f"[v05-pilot] 无法在 {run_root} 建立 attempt 记录：{error}", file=sys.stderr)
        return 2

    started = utc_now()
    receipt: dict[str, Any] = {}
    exit_code = 1

    paths["command_txt"].write_text(
        f"{sys.executable} {' '.join(sys.argv)}\nrun_id={run_id}\nattempt={paths['attempt']}\n",
        encoding="utf-8",
    )

    with paths["log"].open("w", encoding="utf-8", buffering=1) as log_stream:
        with contextlib.redirect_stdout(Tee(sys.__stdout__, log_stream)), contextlib.redirect_stderr(
            Tee(sys.__stderr__, log_stream)
        ):
            print(
                json.dumps(
                    {
                        "event": "started",
                        "run_id": run_id,
                        "attempt": paths["attempt"],
                        "command": args.command,
                        "log_path": str(paths["log"]),
                        "run_root": str(run_root),
                        "started_at_utc": started,
                    },
                    ensure_ascii=False,
                )
            )
            try:
                if args.command == "session-receipt":
                    receipt = build_session_receipt(workspace)
                    receipt["status"] = "PASSED" if receipt["container"]["inside_docker"] else "BLOCKED_ENVIRONMENT"
                else:
                    receipt = build_preflight(workspace)
                exit_code = 0 if receipt.get("status") == "PASSED" else 1
                print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
                if receipt.get("blockers"):
                    print(f"[v05-pilot] BLOCKERS: {receipt['blockers']}", file=sys.stderr)
                if receipt.get("warnings"):
                    print(f"[v05-pilot] WARNINGS: {receipt['warnings']}", file=sys.stderr)
            except Exception as error:  # noqa: BLE001 - 回执必须落盘，异常不能吞掉进度
                receipt = {"schema_version": f"{SCHEMA}-error", "status": "FAILED", "error": repr(error)}
                exit_code = 1
                print(f"[v05-pilot] FAILED: {error!r}", file=sys.stderr)

    status_payload = {
        "run_id": run_id,
        "attempt": paths["attempt"],
        "command": args.command,
        "status": receipt.get("status", "FAILED"),
        "exit_code": exit_code,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
    }
    progress_payload = {
        "run_id": run_id,
        "attempt": paths["attempt"],
        "completed": 1,
        "total": 1,
        "current_unit": args.command,
        "updated_at_utc": utc_now(),
    }

    write_json_atomic(paths["runtime"], receipt)
    write_json_atomic(paths["status"], status_payload)
    write_json_atomic(paths["progress"], progress_payload)
    # run 级汇总（§2.5 的目录布局要求），与 attempt 级同内容、原子更新。
    write_json_atomic(paths["dir"] / "status.json", status_payload)
    write_json_atomic(paths["dir"] / "progress.json", progress_payload)
    write_json_atomic(
        paths["dir"] / "latest_log.json",
        {
            "run_id": run_id,
            "attempt": paths["attempt"],
            "command": args.command,
            "log_path": str(paths["log"]),
            "started_at_utc": started,
        },
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
