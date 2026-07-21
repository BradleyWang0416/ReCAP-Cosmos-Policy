#!/usr/bin/env python3
"""Audit the Human2Robot v04 stage-2 retrieval contract.

The command defaults to a dry run.  ``--execute`` first runs the complete v04
Docker preflight, then audits every stage-1 role-only projection without
creating visual caches, model outputs, or training artifacts.
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
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

import h5py

from cosmos_policy.datasets.human2robot_v04_retrieval import (
    CANDIDATE_PARTITION,
    PRIMARY_RETRIEVAL_MODALITY,
    QUERY_PARTITIONS,
    SCHEMA_VERSION,
    assert_pool_growth_nested,
    candidate_rejection_reason,
    filter_candidates,
    read_feature_inputs,
    validate_primary_config,
    window_from_manifest_record,
)


WORKSPACE = Path(os.environ.get("RECAP_WORKSPACE", "/workspace")).resolve()
RUN_ROOT = Path(os.environ.get("HUMAN2ROBOT_V04_RUN_ROOT", "/DATA1/wxs/ReCAP_M5B_V04_RUNS"))
LOG_ROOT = RUN_ROOT / "orchestrator_logs"
DEFAULT_DERIVED_ROOT = WORKSPACE / "data/Human2Robot/derived/v04"
REPORT_NAME = "stage2_retrieval_contract_report.json"


class Stage2AuditError(RuntimeError):
    """A fail-closed stage-2 audit violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage2AuditError(message)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"Required file is missing: {resolved}")
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": file_sha256(resolved)}


def write_json_atomic(path: Path, value: Mapping[str, Any], *, immutable: bool = False) -> None:
    if immutable:
        require(not path.exists(), f"Refusing to replace immutable artifact: {path}")
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


def _first_legal_start(record: Mapping[str, Any]) -> int:
    with h5py.File(Path(str(record["projection"]["path"])), "r") as file:
        starts = file["data/demo_0/time/legal_window_start"]
        require(len(starts) > 0, f"Projection has no legal window: {record['episode_id']}")
        return int(starts[0])


def audit_retrieval_contract(
    *,
    workspace: Path,
    derived_root: Path,
    execute: bool,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    report_path = derived_root / REPORT_NAME
    if not execute:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "DRY_RUN",
            "execute_required": True,
            "derived_root": str(derived_root.resolve()),
            "report_path": str(report_path.resolve()),
            "planned_checks": [
                "stage1 manifest/lock/audit and generation-code bindings",
                "all 140 role-only projection allowlists and source identities",
                "source SHA/path/partition candidate rejection",
                "history/current-only feature provenance",
                "geometry_plus_visual primary hard gate and top-k=3",
                "pool1/2/4/8/10 strict nesting",
            ],
            "training_allowed": False,
        }

    manifest_path = derived_root / "source_split_manifest.json"
    lock_path = derived_root / "source_split_manifest.lock.json"
    stage1_report_path = derived_root / "stage1_data_audit_report.json"
    manifest = read_json(manifest_path)
    lock = read_json(lock_path)
    stage1_report = read_json(stage1_report_path)
    require(lock.get("status") == "locked", "Stage-1 manifest is not locked")
    require(stage1_report.get("status") == "PASSED", "Stage-1 audit is not passed")
    require(stage1_report.get("future_stage_authorization", {}).get("stage2_allowed") is True, "Stage 2 is not authorized")
    require(stage1_report.get("future_stage_authorization", {}).get("training_allowed") is False, "Unexpected training authorization")
    require(file_sha256(manifest_path) == lock["manifest"]["sha256"], "Stage-1 manifest hash changed")
    require(file_sha256(manifest_path) == stage1_report["manifest"]["sha256"], "Stage-1 audit/manifest hash mismatch")
    require(file_sha256(lock_path) == stage1_report["lock"]["sha256"], "Stage-1 audit/lock hash mismatch")
    for binding in manifest.get("generation_code", []):
        bound_path = Path(str(binding["path"]))
        require(bound_path.is_file() and file_sha256(bound_path) == binding["sha256"], f"Stage-1 generation code changed: {bound_path}")

    records = [record for record in manifest.get("records", []) if "projection" in record]
    require(len(records) == 140, f"Expected 140 role-only projections, got {len(records)}")
    candidate_records = [record for record in records if record["source_partition"] == CANDIDATE_PARTITION]
    query_records = [record for record in records if record["source_partition"] in QUERY_PARTITIONS]
    require(len(candidate_records) == 40, f"Expected 40 human-pool projections, got {len(candidate_records)}")
    require(len(query_records) == 100, f"Expected 100 robot-query projections, got {len(query_records)}")

    windows = []
    provenance = []
    total = len(records)
    for index, record in enumerate(records, 1):
        window = window_from_manifest_record(record, _first_legal_start(record))
        _, _, feature_provenance = read_feature_inputs(window)
        windows.append(window)
        provenance.append(feature_provenance)
        if progress:
            progress(index, total, f"projection:{record['source_partition']}:{record['episode_id']}")
    candidates = [window for window in windows if window.source_partition == CANDIDATE_PARTITION]
    queries = [window for window in windows if window.source_partition in QUERY_PARTITIONS]
    nested = assert_pool_growth_nested(candidates)
    validate_primary_config({"retrieval_modality": PRIMARY_RETRIEVAL_MODALITY, "top_k": 3, "pool_size": 10})

    candidate_by_task: dict[str, list[Any]] = {}
    for candidate in candidates:
        candidate_by_task.setdefault(candidate.task, []).append(candidate)
    eligible_counts = []
    for query in queries:
        active, rejected = filter_candidates(query, candidate_by_task[query.task], pool_size=10)
        require(len(active) == 10 and not rejected, f"Active pool contract failed: {query.window_id}: {rejected}")
        eligible_counts.append(len(active))
        sample = active[0]
        require(candidate_rejection_reason(query, replace(sample, source_sha256=query.source_sha256), 10) == "same_source_sha256", "Same-SHA guard failed")
        require(
            candidate_rejection_reason(query, replace(sample, source_relative_path=query.source_relative_path), 10)
            == "same_source_relative_path",
            "Same-path guard failed",
        )

    candidate_sha = {window.source_sha256 for window in candidates}
    query_sha = {window.source_sha256 for window in queries}
    candidate_paths = {window.source_relative_path for window in candidates}
    query_paths = {window.source_relative_path for window in queries}
    require(not candidate_sha.intersection(query_sha), "Human-pool/query source SHA overlap")
    require(not candidate_paths.intersection(query_paths), "Human-pool/query source path overlap")
    require(all(not item.future_rows_read for item in provenance), "Feature provenance contains future rows")
    require(all(not item.target_datasets_read for item in provenance), "Feature provenance contains targets/actions")
    require(all(not item.opposite_role_datasets_read for item in provenance), "Feature provenance contains opposite-role fields")

    code_bindings = [
        bind_file(workspace / "cosmos_policy/datasets/human2robot_v04_retrieval.py"),
        bind_file(workspace / "cosmos_policy/datasets/human2robot_v04_retrieval_test.py"),
        bind_file(Path(__file__)),
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "VERIFIED_STAGE2",
        "completed_at_utc": utc_now(),
        "formal_result": False,
        "training_started": False,
        "features_generated": False,
        "primary_retrieval": {
            "modality": PRIMARY_RETRIEVAL_MODALITY,
            "top_k": 3,
            "pool_size": 10,
            "seed": 20260711,
            "geometry": "H8 relative-to-current 10D state; seen-train standardized; L2 normalized",
            "visual": "current frame frozen WAN latent; L2 normalized",
            "fusion": "equal-weight concatenation with 1/sqrt(2)",
            "tie_break": "SHA256(seed, query_id, human_content_sha256)",
        },
        "oracle_phase": {
            "primary_allowed": False,
            "name": "oracle_phase",
            "requires_primary_completion_receipt": True,
        },
        "projection_audit": {
            "projection_count": len(windows),
            "human_pool_episode_count": len(candidates),
            "robot_query_episode_count": len(queries),
            "robot_dev_episode_count": sum(window.source_partition == "v04_robot_dev" for window in queries),
            "robot_final_episode_count": sum(window.source_partition == "v04_robot_final" for window in queries),
            "representative_window_count": len(windows),
            "query_candidate_source_sha256_overlap_count": 0,
            "query_candidate_source_path_overlap_count": 0,
            "future_row_read_count": 0,
            "target_action_read_count": 0,
            "opposite_role_field_read_count": 0,
            "active_pool10_candidate_count_min": min(eligible_counts),
            "active_pool10_candidate_count_max": max(eligible_counts),
        },
        "pool_growth": {
            "sizes": [1, 2, 4, 8, 10],
            "strictly_nested": True,
            "source_sha256_by_size_and_task": nested,
            "same_checkpoint_required": True,
            "retraining_allowed": False,
        },
        "feature_provenance_contract": {
            "record_fields": [
                "query/candidate source_sha256",
                "query/candidate source_relative_path",
                "query/candidate source_partition",
                "retrieval rank/distance/tie_sha256",
                "geometry datasets/rows",
                "visual dataset/current row/feature kind",
            ],
            "geometry_future_rows_allowed": False,
            "visual_future_frames_allowed": False,
            "candidate_robot_fields_allowed": False,
        },
        "stage1_bindings": {
            "manifest": bind_file(manifest_path),
            "lock": bind_file(lock_path),
            "audit_report": bind_file(stage1_report_path),
            "protocol_sha256": manifest["protocol_sha256"],
            "split_sha256": manifest["split_sha256"],
            "raw_inventory_sha256": manifest["raw_inventory_sha256"],
        },
        "code_bindings": code_bindings,
        "future_stage_authorization": {"stage3_allowed": True, "training_allowed": False},
        "deferred_to_stage4": [
            "fit and freeze seen-train geometry statistics",
            "materialize frozen WAN visual feature cache",
            "run old-checkpoint read-only dev smoke",
        ],
    }
    write_json_atomic(report_path, report, immutable=True)
    return {
        "schema_version": f"{SCHEMA_VERSION}-receipt",
        "status": "PASSED",
        "completed_at_utc": utc_now(),
        "report": bind_file(report_path),
        "stage1_manifest": bind_file(manifest_path),
        "stage1_lock": bind_file(lock_path),
        "projection_count": len(windows),
        "training_started": False,
        "features_generated": False,
        "stage3_allowed": True,
        "training_allowed": False,
    }


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
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
    run_id = args.run_id or f"stage2_retrieval_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    run_dir, attempt, paths = _attempt_paths(run_id)
    started = utc_now()
    paths["command"].write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    write_json_atomic(
        paths["runtime"],
        {"schema_version": f"{SCHEMA_VERSION}-runtime", "run_id": run_id, "attempt": attempt, "started_at_utc": started, "python": sys.executable},
    )
    running = {"status": "RUNNING", "run_id": run_id, "attempt": attempt, "updated_at_utc": started, "log_path": str(paths["log"])}
    write_json_atomic(paths["status"], running)
    write_json_atomic(paths["progress"], {**running, "completed_units": 0, "total_units": 140, "current_unit": "startup"})
    write_json_atomic(run_dir / "status.json", running)

    def progress(completed: int, total: int, current: str) -> None:
        value = {**running, "completed_units": completed, "total_units": total, "current_unit": current, "updated_at_utc": utc_now()}
        write_json_atomic(paths["progress"], value)
        write_json_atomic(run_dir / "progress.json", value)

    exit_code = 1
    receipt: dict[str, Any]
    with paths["log"].open("w", encoding="utf-8", buffering=1) as log_stream:
        with contextlib.redirect_stdout(Tee(sys.__stdout__, log_stream)), contextlib.redirect_stderr(Tee(sys.__stderr__, log_stream)):
            print(json.dumps({"event": "started", **running}, ensure_ascii=False))
            try:
                preflight = None
                if args.execute:
                    from tools.human2robot_v04 import build_preflight

                    preflight = build_preflight(args.workspace)
                    if preflight.get("status") != "PASSED":
                        receipt = {
                            "schema_version": f"{SCHEMA_VERSION}-blocked",
                            "status": "BLOCKED_ENVIRONMENT",
                            "blockers": preflight.get("blockers", []),
                            "preflight": preflight,
                        }
                    else:
                        receipt = audit_retrieval_contract(
                            workspace=args.workspace,
                            derived_root=args.derived_root,
                            execute=True,
                            progress=progress,
                        )
                        receipt["preflight"] = preflight
                else:
                    receipt = audit_retrieval_contract(
                        workspace=args.workspace,
                        derived_root=args.derived_root,
                        execute=False,
                        progress=progress,
                    )
                exit_code = 0 if receipt.get("status") in {"PASSED", "DRY_RUN"} else 2
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
    write_json_atomic(paths["status"], final_status)
    write_json_atomic(paths["progress"], {**final_status, "completed_units": 140 if exit_code == 0 else 0, "total_units": 140, "current_unit": "finished"})
    write_json_atomic(run_dir / "status.json", final_status)
    write_json_atomic(run_dir / "progress.json", {**final_status, "current_unit": "finished"})
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true", help="Run full preflight and materialize the immutable stage-2 report")
    return parser


def main() -> int:
    return execute_with_audit(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
