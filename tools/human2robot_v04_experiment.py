#!/usr/bin/env python3
"""Single audited experiment interface for Human2Robot v04 stages 1--7.

The stage-1 generator entry point is hash-bound by the frozen split manifest and
must not be edited.  This front controller imports that frozen implementation
for data/preflight operations and owns the stage-3 command surface.  Commands
are dry-run by default; ``--execute`` is necessary, but never sufficient, for a
mutating or GPU operation because later-stage gates remain fail closed.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import shlex
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Mapping, TextIO

from tools import human2robot_v04 as frozen


SCHEMA = "human2robot-v04-experiment-interface-v2"
WORKSPACE = Path(os.environ.get("RECAP_WORKSPACE", "/workspace")).resolve()
RUN_ROOT = Path(os.environ.get("HUMAN2ROBOT_V04_RUN_ROOT", "/DATA1/wxs/ReCAP_M5B_V04_RUNS")).resolve()
LOG_ROOT = RUN_ROOT / "orchestrator_logs"
DERIVED_ROOT = WORKSPACE / "data/Human2Robot/derived/v04"
FEATURE_ROOT = RUN_ROOT / "features"

PUBLIC_COMMANDS = (
    "prepare-data",
    "audit-data",
    "prepare-features",
    "preflight",
    "train",
    "evaluate",
    "evaluate-oracle-phase",
    "report",
)
TRAIN_METHODS = ("no_retrieval", "co_training", "recap_hand_ret")
EVALUATION_METHODS = (*TRAIN_METHODS, "retrieval_only")
EVALUATION_SPLITS = ("dev", "final")
POOL_SIZES = (1, 2, 4, 8, 10)
SHA256_LENGTH = 64

STAGE_BY_COMMAND = {
    "prepare-data": "prepare",
    "audit-data": "prepare",
    "prepare-features": "prepare",
    "preflight": "preflight",
    "train": "training",
    "evaluate": "evaluation",
    "evaluate-oracle-phase": "oracle_diagnostic",
    "report": "report",
}
NEXT_STAGE_FOR_BLOCKED_COMMAND = {
    "prepare-features": 4,
    "train": 5,
    "evaluate": 6,
    "evaluate-oracle-phase": 7,
    "report": 7,
}


class Stage3InterfaceError(RuntimeError):
    """A command contract or immutable-audit failure."""


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


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise Stage3InterfaceError(f"Required file is missing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def write_json_atomic(path: Path, value: Any, *, immutable: bool = False) -> None:
    if immutable and path.exists():
        raise Stage3InterfaceError(f"Refusing to replace immutable JSON: {path}")
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
        if immutable:
            path.chmod(0o444)
    finally:
        with contextlib.suppress(FileNotFoundError):
            partial.unlink()


def _common_parser(command: str, subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(command)
    parser.add_argument("--run-id", help="Reuse a run id while creating a new immutable attempt")
    parser.add_argument("--execute", action="store_true", help="Request the real operation; dry-run is the default")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for name in ("prepare-data", "audit-data"):
        command = _common_parser(name, commands)
        command.add_argument(
            "--source-root",
            type=Path,
            default=Path("/DATA1/wxs/DATASETS/Human2Robot/data/v1"),
        )
        command.add_argument("--derived-root", type=Path, default=DERIVED_ROOT)

    command = _common_parser("prepare-features", commands)
    command.add_argument(
        "--source-root",
        type=Path,
        default=Path("/DATA1/wxs/DATASETS/Human2Robot/data/v1"),
    )
    command.add_argument("--derived-root", type=Path, default=DERIVED_ROOT)
    command.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    command.add_argument("--visual-batch-size", type=int, default=32)

    _common_parser("preflight", commands)

    command = _common_parser("train", commands)
    command.add_argument("--method", choices=TRAIN_METHODS, required=True)

    command = _common_parser("evaluate", commands)
    command.add_argument("--split", choices=EVALUATION_SPLITS, required=True)
    command.add_argument("--method", choices=EVALUATION_METHODS, required=True)
    command.add_argument("--pool-size", type=int, choices=POOL_SIZES, required=True)
    command.add_argument("--checkpoint", type=Path)

    command = _common_parser("evaluate-oracle-phase", commands)
    command.add_argument("--split", choices=EVALUATION_SPLITS, required=True)
    command.add_argument("--pool-size", type=int, choices=POOL_SIZES, default=10)
    command.add_argument("--checkpoint", type=Path)
    command.add_argument("--primary-receipt-sha256", required=True)

    command = _common_parser("report", commands)
    command.add_argument("--final-receipt", type=Path)
    return parser


def command_names() -> tuple[str, ...]:
    return PUBLIC_COMMANDS


def _is_sha256(value: str | None) -> bool:
    if value is None or len(value) != SHA256_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value)


def validate_args(args: argparse.Namespace) -> None:
    if args.command not in PUBLIC_COMMANDS:
        raise Stage3InterfaceError(f"Unsupported public command: {args.command}")
    if args.command == "prepare-features" and args.visual_batch_size <= 0:
        raise Stage3InterfaceError("prepare-features visual batch size must be positive")
    if args.command == "evaluate":
        if args.method in ("no_retrieval", "co_training") and args.pool_size != 10:
            raise Stage3InterfaceError("Learning baselines must record the canonical pool-size 10 evaluation condition")
        if args.execute and args.checkpoint is None:
            raise Stage3InterfaceError("Executed evaluation requires an explicitly bound checkpoint")
    if args.command == "evaluate-oracle-phase":
        if args.pool_size != 10:
            raise Stage3InterfaceError("oracle-phase diagnostic is frozen to pool-size 10")
        if not _is_sha256(args.primary_receipt_sha256):
            raise Stage3InterfaceError("oracle-phase requires a lowercase SHA256 of a completed primary receipt")
        if args.execute and args.checkpoint is None:
            raise Stage3InterfaceError("Executed oracle-phase evaluation requires an explicitly bound checkpoint")
    if args.command == "report" and args.execute and args.final_receipt is None:
        raise Stage3InterfaceError("Executed report requires an explicitly bound final receipt")


def normalized_parameters(args: argparse.Namespace) -> dict[str, Any]:
    omitted = {"command", "run_id", "execute"}
    result: dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if key in omitted:
            continue
        if isinstance(value, Path):
            result[key] = str(value.resolve())
        else:
            result[key] = value
    return result


def _controlled_bindings(workspace: Path) -> list[dict[str, Any]]:
    paths = [
        workspace / "tools/human2robot_v04_experiment.py",
        workspace / "tools/human2robot_v04.py",
        workspace / "tools/human2robot_v04_data.py",
        workspace / "tools/human2robot_v04_stage4.py",
        workspace / "tools/human2robot_v04_stage4_feature_worker.py",
        workspace / "tools/human2robot_v04_stage4_worker.py",
        workspace / "tools/human2robot_m5b_p2_inference.py",
        workspace / "tools/human2robot_m5b_p2_step_checkpoint_diagnostic.py",
        workspace / "方案/v04/RECAP_Human2Robot_无泄漏单seed离线复现执行总计划.md",
        workspace / "docs/pusht_rag_docker_runbook.md",
    ]
    return [bind_file(path) for path in paths]


def build_invocation_manifest(
    args: argparse.Namespace,
    *,
    workspace: Path,
    run_root: Path,
    run_id: str,
    attempt: int,
) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA}-invocation-manifest",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "attempt": attempt,
        "command": args.command,
        "state": STAGE_BY_COMMAND[args.command],
        "execute_requested": bool(args.execute),
        "dry_run": not bool(args.execute),
        "parameters": normalized_parameters(args),
        "workspace": str(workspace.resolve()),
        "run_root": str(run_root.resolve()),
        "v03_registry_or_artifact_reuse_allowed": False,
        "formal_output_boundary": str(run_root.resolve()),
        "controlled_source": _controlled_bindings(workspace),
    }


def planned_result(args: argparse.Namespace) -> dict[str, Any]:
    later_stage = NEXT_STAGE_FOR_BLOCKED_COMMAND.get(args.command)
    return {
        "schema_version": f"{SCHEMA}-dry-run",
        "status": "DRY_RUN",
        "command": args.command,
        "execute_requested": False,
        "stage": STAGE_BY_COMMAND[args.command],
        "parameters": normalized_parameters(args),
        "planned_mutation": args.command not in ("preflight", "audit-data"),
        "planned_gpu_operation": args.command in ("prepare-features", "train", "evaluate", "evaluate-oracle-phase"),
        "required_future_stage": later_stage,
        "training_started": False,
        "features_generated": False,
        "evaluation_started": False,
    }


def blocked_stage_result(args: argparse.Namespace) -> dict[str, Any]:
    stage = NEXT_STAGE_FOR_BLOCKED_COMMAND[args.command]
    return {
        "schema_version": f"{SCHEMA}-stage-gate",
        "status": "BLOCKED_STAGE_GATE",
        "command": args.command,
        "execute_requested": True,
        "required_stage": stage,
        "blockers": [f"stage{stage}_implementation_and_formal_authorization_missing"],
        "training_started": False,
        "features_generated": False,
        "evaluation_started": False,
        "formal_result": False,
    }


def dispatch(args: argparse.Namespace, *, workspace: Path) -> dict[str, Any]:
    if not args.execute:
        return planned_result(args)
    if args.command == "preflight":
        return frozen.build_preflight(workspace)
    if args.command == "prepare-features":
        preflight = frozen.build_preflight(workspace)
        if preflight.get("status") != "PASSED":
            return {
                "schema_version": f"{SCHEMA}-stage4-blocked",
                "status": "BLOCKED_ENVIRONMENT",
                "blockers": preflight.get("blockers", []),
                "preflight": preflight,
                "training_allowed": False,
            }
        from tools import human2robot_v04_stage4 as stage4

        result = stage4.run_stage4(
            workspace=workspace,
            run_root=RUN_ROOT,
            derived_root=args.derived_root,
            feature_root=args.feature_root,
            source_root=args.source_root,
            visual_batch_size=args.visual_batch_size,
            run_smoke=True,
        )
        result["preflight"] = preflight
        return result
    if args.command in ("prepare-data", "audit-data"):
        preflight = frozen.build_preflight(workspace)
        if preflight.get("status") != "PASSED":
            return {
                "schema_version": f"{SCHEMA}-stage1-blocked",
                "status": "BLOCKED_ENVIRONMENT",
                "blockers": preflight.get("blockers", []),
                "preflight": preflight,
            }
        from tools import human2robot_v04_data as stage1

        operation = stage1.prepare_data if args.command == "prepare-data" else stage1.audit_data
        result = operation(
            workspace=workspace,
            source_root=args.source_root,
            derived_root=args.derived_root,
            execute=True,
            progress=None,
        )
        result["preflight"] = preflight
        return result
    return blocked_stage_result(args)


def _attempt_paths(run_root: Path, run_id: str) -> tuple[Path, int, dict[str, Path]]:
    run_dir = run_root / "orchestrator_logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    attempts = [int(path.name.split("_")[1].split(".")[0]) for path in run_dir.glob("attempt_*.status.json")]
    attempt = max(attempts, default=0) + 1
    stem = run_dir / f"attempt_{attempt:04d}"
    return run_dir, attempt, {
        "log": Path(f"{stem}.log"),
        "command": Path(f"{stem}.command.txt"),
        "runtime": Path(f"{stem}.runtime.json"),
        "manifest": Path(f"{stem}.manifest.json"),
        "status": Path(f"{stem}.status.json"),
        "progress": Path(f"{stem}.progress.json"),
        "receipt": Path(f"{stem}.receipt.json"),
    }


def _exit_code(status: str) -> int:
    if status in {"PASSED", "passed", "DRY_RUN"}:
        return 0
    if status in {"BLOCKED_ENVIRONMENT", "BLOCKED_STAGE_GATE", "BLOCKED_PROTOCOL"}:
        return 2
    return 1


def execute_with_audit(
    args: argparse.Namespace,
    *,
    workspace: Path = WORKSPACE,
    run_root: Path = RUN_ROOT,
) -> int:
    validate_args(args)
    run_id = args.run_id or f"{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}_{args.command}_{uuid.uuid4().hex[:8]}"
    run_dir, attempt, paths = _attempt_paths(run_root, run_id)
    started = utc_now()
    paths["command"].write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    manifest = build_invocation_manifest(
        args,
        workspace=workspace,
        run_root=run_root,
        run_id=run_id,
        attempt=attempt,
    )
    write_json_atomic(paths["manifest"], manifest, immutable=True)
    manifest_binding = bind_file(paths["manifest"])
    runtime = {
        "schema_version": f"{SCHEMA}-runtime",
        "run_id": run_id,
        "attempt": attempt,
        "command": args.command,
        "started_at_utc": started,
        "pid": os.getpid(),
        "python": sys.executable,
        "manifest": manifest_binding,
    }
    write_json_atomic(paths["runtime"], runtime)
    running = {
        "status": "RUNNING",
        "run_id": run_id,
        "attempt": attempt,
        "updated_at_utc": started,
        "log_path": str(paths["log"]),
    }
    progress = {**running, "completed_units": 0, "total_units": 1, "current_unit": "dispatch"}
    write_json_atomic(paths["status"], running)
    write_json_atomic(paths["progress"], progress)
    write_json_atomic(run_dir / "latest_log.json", {"run_id": run_id, "attempt": attempt, "log_path": str(paths["log"])})
    write_json_atomic(run_dir / "status.json", running)
    write_json_atomic(run_dir / "progress.json", progress)

    result: dict[str, Any]
    with paths["log"].open("w", encoding="utf-8", buffering=1) as log_stream:
        with contextlib.redirect_stdout(Tee(sys.__stdout__, log_stream)), contextlib.redirect_stderr(Tee(sys.__stderr__, log_stream)):
            print(json.dumps({"event": "started", "run_id": run_id, "attempt": attempt, "manifest": manifest_binding}, ensure_ascii=False))
            try:
                result = dispatch(args, workspace=workspace)
            except Exception as error:
                result = {
                    "status": "FAILED",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                print(result["traceback"], file=sys.stderr)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))

    status = str(result.get("status", "FAILED"))
    exit_code = _exit_code(status)
    receipt = {
        "schema_version": f"{SCHEMA}-immutable-receipt",
        "status": status,
        "run_id": run_id,
        "attempt": attempt,
        "command": args.command,
        "execute_requested": bool(args.execute),
        "manifest": manifest_binding,
        "result": result,
    }
    write_json_atomic(paths["receipt"], receipt, immutable=True)
    finished = utc_now()
    final_status = {
        "status": "COMPLETED" if exit_code == 0 else status,
        "run_id": run_id,
        "attempt": attempt,
        "exit_code": exit_code,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "log_path": str(paths["log"]),
        "manifest_path": str(paths["manifest"]),
        "manifest_sha256": manifest_binding["sha256"],
        "receipt_path": str(paths["receipt"]),
        "receipt_sha256": file_sha256(paths["receipt"]),
    }
    final_progress = {**final_status, "completed_units": 1, "total_units": 1, "current_unit": "finished"}
    write_json_atomic(paths["status"], final_status)
    write_json_atomic(paths["progress"], final_progress)
    write_json_atomic(run_dir / "status.json", final_status)
    write_json_atomic(run_dir / "progress.json", final_progress)
    print(json.dumps({"event": "finished", **final_status}, ensure_ascii=False))
    return exit_code


def main() -> int:
    return execute_with_audit(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
