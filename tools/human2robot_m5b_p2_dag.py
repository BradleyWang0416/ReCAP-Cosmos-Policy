#!/usr/bin/env python3
"""Fail-closed single-cell dispatcher for the frozen 203-cell M5B-P2 DAG.

The module exposes read-only inventory/plan commands and one explicit-cell
dispatcher.  It intentionally has no "run all" command: formal activation,
parent artifacts, and the cell's registered handler must all validate before a
single subprocess can start.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from tools.human2robot_m5b_p2 import source_manifest, source_paths, source_snapshot_matches_candidate
from tools.human2robot_m5b_p2_handlers import (
    DEFAULT_ARTIFACT_ROOT,
    HandlerContractError,
    build_handler_plans,
    require_formal_activation,
)
from tools.human2robot_m5b_p2_matrix import ExecutionMatrix, file_sha256, load_execution_matrix


class DagContractError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DagContractError(message)


def read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def artifact_path(root: Path, cell_id: str) -> Path:
    return root / "cells" / cell_id / "artifact.json"


def completed_artifact(root: Path, cell_id: str) -> dict[str, Any]:
    path = artifact_path(root, cell_id)
    artifact = read_json(path)
    _require(artifact.get("cell_id") == cell_id, f"Cell ID mismatch: {path}")
    _require(
        artifact.get("status") in {"completed", "completed_detector_triggered_excluded"},
        f"Cell is not completed: {cell_id}",
    )
    _require(artifact.get("formal_result") is True, f"Cell is not formal: {cell_id}")
    return artifact


def inventory(matrix: ExecutionMatrix, artifact_root: Path) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    counts = {"completed": 0, "missing": 0, "invalid": 0}
    for cell_id in matrix.topological_cell_ids:
        path = artifact_path(artifact_root, cell_id)
        if not path.is_file():
            status = "missing"
            detail: dict[str, Any] = {}
        else:
            try:
                completed_artifact(artifact_root, cell_id)
            except DagContractError as error:
                status = "invalid"
                detail = {"error": str(error), "file_sha256": file_sha256(path)}
            else:
                status = "completed"
                detail = {"file_sha256": file_sha256(path)}
        counts[status] += 1
        records[cell_id] = {
            "status": status,
            "artifact_kind": matrix.cells_by_id[cell_id].artifact_kind,
            "parent_artifact_ids": list(matrix.cells_by_id[cell_id].parent_artifact_ids),
            **detail,
        }
    ready = [
        cell_id
        for cell_id in matrix.topological_cell_ids
        if records[cell_id]["status"] == "missing"
        and all(records[parent_id]["status"] == "completed" for parent_id in records[cell_id]["parent_artifact_ids"])
    ]
    return {
        "schema_version": "human2robot-m5b-p2-dag-inventory-v1",
        "expected_cell_count": 203,
        "counts": counts,
        "ready_cell_ids": ready,
        "all_203_complete": counts["completed"] == 203,
        "formal_queue_allowed": False,
        "records": records,
    }


def require_current_activation(
    workspace: Path,
    artifact_root: Path,
    activation_path: Path,
    matrix: ExecutionMatrix,
) -> dict[str, Any]:
    """Validate queue authorization and bind it to the current source bytes."""

    activation = read_json(activation_path)
    try:
        require_formal_activation(activation, matrix)
    except HandlerContractError as error:
        raise DagContractError(str(error)) from error

    source = source_manifest(workspace, source_paths(workspace))
    code_sha256 = source["code_sha256"]
    expected_snapshot_manifest = artifact_root / "source_snapshots" / code_sha256 / "source_snapshot_manifest.json"
    _require(
        activation.get("candidate_code_sha256") == code_sha256,
        "Launch activation is bound to different candidate code",
    )
    _require(
        activation.get("source_snapshot_manifest_path") == str(expected_snapshot_manifest),
        "Launch activation is bound to a different source snapshot",
    )
    frozen_source = read_json(expected_snapshot_manifest)
    _require(
        source_snapshot_matches_candidate(frozen_source, source),
        "Frozen source snapshot differs from current candidate code",
    )

    receipt_value = activation.get("docker_suite_receipt_path")
    _require(isinstance(receipt_value, str), "Launch activation has no Docker-suite receipt path")
    receipt_path = Path(receipt_value)
    _require(receipt_path.is_file(), "Launch activation Docker-suite receipt is missing")
    _require(
        activation.get("docker_suite_receipt_sha256") == file_sha256(receipt_path),
        "Launch activation Docker-suite receipt hash mismatch",
    )
    return activation


def build_plan(
    workspace: Path,
    artifact_root: Path,
    activation_path: Path,
    matrix: ExecutionMatrix,
) -> dict[str, Any]:
    current = inventory(matrix, artifact_root)
    try:
        activation = require_current_activation(workspace, artifact_root, activation_path, matrix)
    except DagContractError as error:
        activation_status: dict[str, Any] = {"status": "invalid", "error": str(error)}
        formal_queue_allowed = False
    else:
        activation_status = {
            "status": "approved",
            "file_sha256": file_sha256(activation_path),
            "candidate_code_sha256": activation["candidate_code_sha256"],
        }
        formal_queue_allowed = True
    return {
        "schema_version": "human2robot-m5b-p2-dag-plan-v5",
        "formal_queue_allowed": formal_queue_allowed,
        "formal_queue_started": False,
        "ready_cell_ids": current["ready_cell_ids"],
        "counts": current["counts"],
        "matrix_blockers": list(matrix.formal_readiness_blockers),
        "launch_activation": activation_status,
    }


def run_registered_cell(
    workspace: Path,
    artifact_root: Path,
    activation_path: Path,
    cell_id: str,
) -> dict[str, Any]:
    _require(artifact_root == Path(DEFAULT_ARTIFACT_ROOT), "Formal artifact root differs from frozen root")
    matrix = load_execution_matrix(workspace)
    _require(cell_id in matrix.bindings_by_id, f"Unknown frozen cell: {cell_id}")
    require_current_activation(workspace, artifact_root, activation_path, matrix)
    current = inventory(matrix, artifact_root)
    record = current["records"][cell_id]
    _require(record["status"] != "invalid", f"Existing cell artifact is invalid: {cell_id}")
    if record["status"] == "completed":
        return completed_artifact(artifact_root, cell_id)
    _require(cell_id in current["ready_cell_ids"], f"Cell parents are not complete: {cell_id}")

    plan = build_handler_plans(matrix)[cell_id]
    env = os.environ.copy()
    env.update(dict(plan.environment))
    process = subprocess.run(
        list(plan.command),
        cwd=workspace,
        env=env,
        check=False,
    )
    _require(process.returncode == 0, f"Registered handler exited {process.returncode}: {cell_id}")
    return completed_artifact(artifact_root, cell_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--artifact-root", type=Path, default=Path(DEFAULT_ARTIFACT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory")
    plan = subparsers.add_parser("plan")
    plan.add_argument(
        "--activation-path",
        type=Path,
        default=Path(DEFAULT_ARTIFACT_ROOT) / "launch_activation_v5.json",
    )
    run = subparsers.add_parser("run-cell")
    run.add_argument("cell_id")
    run.add_argument(
        "--activation-path",
        type=Path,
        default=Path(DEFAULT_ARTIFACT_ROOT) / "launch_activation_v5.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.resolve()
    artifact_root = args.artifact_root.absolute()
    matrix = load_execution_matrix(workspace)
    if args.command == "inventory":
        result = inventory(matrix, artifact_root)
    elif args.command == "plan":
        result = build_plan(workspace, artifact_root, args.activation_path.absolute(), matrix)
    else:
        result = run_registered_cell(
            workspace,
            artifact_root,
            args.activation_path.absolute(),
            args.cell_id,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DagContractError as error:
        print(f"M5B-P2 DAG error: {error}", file=__import__("sys").stderr)
        raise SystemExit(2) from error
