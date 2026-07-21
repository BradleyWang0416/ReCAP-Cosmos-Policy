#!/usr/bin/env python3
"""Run the frozen Human2Robot suite and emit a v04 stage-2 receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch


SCHEMA = "human2robot-v04-stage2-docker-suite-v1"
MINIMUM_TEST_COUNT = 194
EXPECTED_OFFLINE_ENV = {
    "COSMOS_SKIP_HF_AUTO_DOWNLOAD": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
}


class SuiteError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SuiteError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"Missing receipt input: {resolved}")
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": file_sha256(resolved)}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"Refusing to replace immutable suite receipt: {path}")
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
        path.chmod(0o444)
    finally:
        if partial.exists():
            partial.unlink()


def run_suite(
    *,
    workspace: Path,
    receipt_path: Path,
    prepare_receipt_path: Path,
    audit_receipt_path: Path,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    require(Path("/.dockerenv").is_file(), "Docker suite must run inside the frozen container")
    require(sys.executable == str(workspace / ".venv/bin/python"), f"Unexpected Python: {sys.executable}")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 4, "Docker suite requires exactly four visible GPUs")
    offline = {key: os.environ.get(key) for key in EXPECTED_OFFLINE_ENV}
    require(offline == EXPECTED_OFFLINE_ENV, f"Offline environment mismatch: {offline}")
    require(os.environ.get("HUMAN2ROBOT_V04_IMAGE_ID", "").startswith("sha256:"), "Docker image ID is not bound")

    explicit_tests = [
        "cosmos_policy/config/experiment/human2robot_experiment_configs_test.py",
        "cosmos_policy/datasets/human2robot_dataset_test.py",
        "cosmos_policy/datasets/human2robot_p2_contract_test.py",
        "cosmos_policy/datasets/human2robot_p2_dataset_test.py",
        "cosmos_policy/datasets/human2robot_v04_sampler_test.py",
        "cosmos_policy/datasets/human2robot_v04_retrieval_test.py",
        "cosmos_policy/models/human2robot_adapter_test.py",
    ]
    tool_tests = [str(path.relative_to(workspace)) for path in sorted((workspace / "tools").glob("human2robot*_test.py"))]
    command = [sys.executable, "-m", "pytest", "-q", *explicit_tests, *tool_tests]
    process = subprocess.run(
        command,
        cwd=workspace,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    matches = re.findall(r"(\d+) passed", process.stdout)
    passed_count = int(matches[-1]) if matches else 0

    controlled_paths = [
        workspace / "tools/human2robot_v04_docker_suite.py",
        workspace / "tools/human2robot_v04.py",
        workspace / "tools/human2robot_v04_test.py",
        workspace / "tools/human2robot_v04_data.py",
        workspace / "tools/human2robot_v04_data_test.py",
        workspace / "tools/human2robot_v04_retrieval.py",
        workspace / "tools/human2robot_v04_stage2_audit_test.py",
        workspace / "cosmos_policy/datasets/human2robot_v04_sampler.py",
        workspace / "cosmos_policy/datasets/human2robot_v04_sampler_test.py",
        workspace / "cosmos_policy/datasets/human2robot_v04_retrieval.py",
        workspace / "cosmos_policy/datasets/human2robot_v04_retrieval_test.py",
        workspace / "方案/v04/RECAP_Human2Robot_无泄漏单seed离线复现执行总计划.md",
        workspace / "docs/pusht_rag_docker_runbook.md",
    ]
    source_bindings = [bind_file(path) for path in controlled_paths]
    prepare_binding = bind_file(prepare_receipt_path)
    audit_binding = bind_file(audit_receipt_path)
    prepare = read_json(prepare_receipt_path)
    audit = read_json(audit_receipt_path)
    require(prepare.get("status") == "PASSED" and audit.get("status") == "PASSED", "Stage-1 prepare/audit is not passed")
    require(prepare.get("protocol_sha256") == audit.get("protocol_sha256"), "Prepare/audit protocol mismatch")
    require(prepare.get("split_sha256") == audit.get("split_sha256"), "Prepare/audit split mismatch")

    artifact_paths = [
        workspace / "data/Human2Robot/derived/v04/source_split_manifest.json",
        workspace / "data/Human2Robot/derived/v04/source_split_manifest.lock.json",
        workspace / "data/Human2Robot/derived/v04/stage1_data_audit_report.json",
        workspace / "data/Human2Robot/derived/v04/stage2_retrieval_contract_report.json",
    ]
    artifact_bindings = [bind_file(path) for path in artifact_paths]
    require(audit.get("manifest", {}).get("sha256") == artifact_bindings[0]["sha256"], "Audit manifest binding mismatch")
    require(audit.get("lock", {}).get("sha256") == artifact_bindings[1]["sha256"], "Audit lock binding mismatch")
    require(audit.get("report", {}).get("sha256") == artifact_bindings[2]["sha256"], "Audit report binding mismatch")
    stage2 = read_json(artifact_paths[3])
    require(stage2.get("status") == "VERIFIED_STAGE2", "Stage-2 retrieval contract is not verified")
    require(stage2.get("future_stage_authorization", {}).get("training_allowed") is False, "Stage-2 report unexpectedly authorizes training")

    passed = process.returncode == 0 and passed_count >= MINIMUM_TEST_COUNT
    receipt = {
        "schema_version": SCHEMA,
        "status": "PASSED" if passed else "FAILED",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "formal_result": False,
        "training_started": False,
        "stage3_authorized_by_this_receipt": passed,
        "training_allowed": False,
        "image": os.environ.get("HUMAN2ROBOT_V04_IMAGE"),
        "image_id": os.environ.get("HUMAN2ROBOT_V04_IMAGE_ID"),
        "host_gpu_devices": os.environ.get("HUMAN2ROBOT_V04_GPU_DEVICES"),
        "visible_gpu_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "offline_environment": offline,
        "command": command,
        "returncode": process.returncode,
        "passed_test_count": passed_count,
        "minimum_test_count": MINIMUM_TEST_COUNT,
        "source_bindings": source_bindings,
        "source_bundle_sha256": canonical_sha256(source_bindings),
        "stage1_artifact_bindings": artifact_bindings,
        "prepare_receipt": prepare_binding,
        "audit_receipt": audit_binding,
        "protocol_sha256": audit["protocol_sha256"],
        "split_sha256": audit["split_sha256"],
        "raw_inventory_sha256": audit["raw_inventory_sha256"],
        "output_tail": process.stdout[-16000:],
    }
    write_json_atomic(receipt_path.resolve(), receipt)
    require(passed, f"Full Human2Robot Docker suite failed; see {receipt_path}")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--prepare-receipt-path", type=Path, required=True)
    parser.add_argument("--audit-receipt-path", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_suite(
        workspace=args.workspace,
        receipt_path=args.receipt_path,
        prepare_receipt_path=args.prepare_receipt_path,
        audit_receipt_path=args.audit_receipt_path,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SuiteError as error:
        print(f"v04 Docker suite error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
