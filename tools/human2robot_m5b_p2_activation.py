#!/usr/bin/env python3
"""Create the M5B-P2 launch activation after every non-result prerequisite passes.

This module can write a Docker-suite receipt and the launch-only activation.
It never starts a cell and it can never create final P2 acceptance.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from tools.human2robot_m5b_p2 import FORMAL_OUTPUT_ROOT, source_manifest, source_paths
from tools.human2robot_m5b_p2_handlers import FORMAL_OFFLINE_ENV, require_formal_activation
from tools.human2robot_m5b_p2_matrix import (
    LAG_VIEW_MANIFEST_SHA256,
    PREPARED_MANIFEST_SHA256,
    SUPPLEMENT_SHA256,
    WORKSPACE_BOUNDS_SHA256,
    file_sha256,
    load_execution_matrix,
)
from tools.human2robot_m5b_p2_preflight import run_preflight


MINIMUM_FROZEN_TEST_COUNT = 137


class ActivationContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ActivationContractError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_docker_suite(workspace: Path, receipt_path: Path) -> dict[str, Any]:
    require(Path("/.dockerenv").is_file(), "Docker suite must run inside the full container")
    source = source_manifest(workspace, source_paths(workspace))
    command = [
        ".venv/bin/pytest",
        "-q",
        "cosmos_policy/config/experiment/human2robot_experiment_configs_test.py",
        "cosmos_policy/datasets/human2robot_dataset_test.py",
        "cosmos_policy/datasets/human2robot_p2_contract_test.py",
        "cosmos_policy/datasets/human2robot_p2_dataset_test.py",
        "cosmos_policy/models/human2robot_adapter_test.py",
        *[str(path.relative_to(workspace)) for path in sorted((workspace / "tools").glob("human2robot*_test.py"))],
    ]
    environment = os.environ.copy()
    environment.update(FORMAL_OFFLINE_ENV)
    process = subprocess.run(
        command,
        cwd=workspace,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    matches = re.findall(r"(\d+) passed", process.stdout)
    passed_count = int(matches[-1]) if matches else 0
    passed = process.returncode == 0 and passed_count >= MINIMUM_FROZEN_TEST_COUNT
    receipt = {
        "schema_version": "human2robot-m5b-p2-docker-suite-receipt-v2",
        "status": "passed" if passed else "failed",
        "formal_result": False,
        "cell_execution_started": False,
        "created_at_utc": utc_now(),
        "candidate_code_sha256": source["code_sha256"],
        "candidate_source_file_count": len(source["files"]),
        "command": command,
        "offline_environment": FORMAL_OFFLINE_ENV,
        "returncode": process.returncode,
        "passed_test_count": passed_count,
        "minimum_frozen_test_count": MINIMUM_FROZEN_TEST_COUNT,
        "output_tail": process.stdout[-12000:],
    }
    write_json_atomic(receipt_path, receipt)
    require(passed, f"Docker suite failed; see {receipt_path}")
    return receipt


def issue_launch_activation(
    workspace: Path,
    artifact_root: Path,
    docker_suite_receipt_path: Path,
) -> dict[str, Any]:
    """Issue queue authorization only; final P2 acceptance remains false."""

    receipt = read_json(docker_suite_receipt_path)
    require(receipt.get("schema_version") == "human2robot-m5b-p2-docker-suite-receipt-v2", "Docker receipt schema drift")
    require(receipt.get("status") == "passed", "Docker suite is not passed")
    require(int(receipt.get("passed_test_count", 0)) >= MINIMUM_FROZEN_TEST_COUNT, "Docker test count is incomplete")
    preflight = run_preflight(
        workspace,
        artifact_root=artifact_root,
        verify_weight_hashes=True,
        require_launch_activation=False,
    )
    require(preflight["status"] == "passed", f"Pre-activation blockers remain: {preflight['blockers']}")
    require(
        receipt.get("candidate_code_sha256") == preflight["source_snapshot"]["candidate_code_sha256"],
        "Docker receipt and frozen source snapshot refer to different code",
    )
    matrix = load_execution_matrix(workspace)
    activation = {
        "schema_version": "human2robot-m5b-p2-launch-activation-v2",
        "status": "approved",
        "launch_authorized": True,
        "formal_queue_allowed": True,
        "p2_acceptance_allowed": False,
        "registry_sha256": matrix.prepared_manifest["registry_file_sha256"],
        "supplement_sha256": SUPPLEMENT_SHA256,
        "prepared_manifest_sha256": PREPARED_MANIFEST_SHA256,
        "workspace_bounds_sha256": WORKSPACE_BOUNDS_SHA256,
        "lag_view_manifest_sha256": LAG_VIEW_MANIFEST_SHA256,
        "native_rectified_flow_contract_resolved": True,
        "all_147_evaluations_bound_to_terminal_report": True,
        "docker_full_suite_passed": True,
        "source_snapshot_frozen": True,
        "gpu_count": 8,
        "storage_probe_passed": True,
        "formal_output_mount_writable": True,
        "local_weight_hashes_passed": True,
        "candidate_code_sha256": preflight["source_snapshot"]["candidate_code_sha256"],
        "source_snapshot_manifest_path": preflight["source_snapshot"]["manifest_path"],
        "docker_suite_receipt_path": str(docker_suite_receipt_path),
        "docker_suite_receipt_sha256": file_sha256(docker_suite_receipt_path),
        "pre_activation_probe": preflight,
        "issued_at_utc": utc_now(),
        "claim_boundary": "Launch authorization only; P2 acceptance and M6 rollout remain forbidden.",
    }
    require_formal_activation(activation, matrix)
    output = artifact_root / "launch_activation_v2.json"
    write_json_atomic(output, activation)
    return activation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--artifact-root", type=Path, default=FORMAL_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    suite = subparsers.add_parser("run-docker-suite")
    suite.add_argument("--receipt-path", type=Path)
    issue = subparsers.add_parser("issue-launch")
    issue.add_argument("--docker-suite-receipt-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.resolve()
    artifact_root = args.artifact_root.resolve()
    if args.command == "run-docker-suite":
        receipt = args.receipt_path or artifact_root / "docker_suite_receipt_v2.json"
        result = run_docker_suite(workspace, receipt.resolve())
    else:
        receipt = args.docker_suite_receipt_path or artifact_root / "docker_suite_receipt_v2.json"
        result = issue_launch_activation(workspace, artifact_root, receipt.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ActivationContractError as error:
        print(f"M5B-P2 activation error: {error}", file=__import__("sys").stderr)
        raise SystemExit(2) from error
