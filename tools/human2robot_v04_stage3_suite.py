#!/usr/bin/env python3
"""Run the frozen Human2Robot suite and emit a stage-3 verification receipt."""

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


SCHEMA = "human2robot-v04-stage3-docker-suite-v1"
MINIMUM_TEST_COUNT = 206
EXPECTED_OFFLINE_ENV = {
    "COSMOS_SKIP_HF_AUTO_DOWNLOAD": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
}


class Stage3SuiteError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage3SuiteError(message)


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
    require(resolved.is_file(), f"Missing suite input: {resolved}")
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


def _unwrap_preflight(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    result = receipt.get("result")
    return result if isinstance(result, Mapping) else receipt


def run_suite(
    *,
    workspace: Path,
    receipt_path: Path,
    stage3_contract_path: Path,
    preflight_receipt_path: Path,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    require(Path("/.dockerenv").is_file(), "Stage-3 suite must run inside the frozen container")
    require(sys.executable == str(workspace / ".venv/bin/python"), f"Unexpected Python: {sys.executable}")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 4, "Stage-3 suite requires exactly four visible GPUs")
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

    contract_binding = bind_file(stage3_contract_path)
    contract = read_json(stage3_contract_path)
    require(contract.get("status") == "VERIFIED_STAGE3", "Stage-3 interface contract is not verified")
    require(contract.get("future_stage_authorization", {}).get("stage4_allowed") is True, "Stage-3 contract does not authorize stage 4")
    require(contract.get("future_stage_authorization", {}).get("training_allowed") is False, "Stage-3 contract unexpectedly authorizes training")
    require(contract.get("state_machine", {}).get("state_count") == 8, "Stage-3 state machine is not the frozen eight states")
    require(contract.get("state_machine", {}).get("cell_registry") is None, "Stage-3 contract reuses a cell registry")
    preflight_binding = bind_file(preflight_receipt_path)
    preflight_receipt = read_json(preflight_receipt_path)
    preflight = _unwrap_preflight(preflight_receipt)
    require(preflight.get("status") == "PASSED" and preflight.get("blockers") == [], "Stage-3 formal preflight is not passed")

    controlled_paths = [
        workspace / "tools/human2robot_v04_experiment.py",
        workspace / "tools/human2robot_v04_stage3_audit.py",
        workspace / "tools/human2robot_v04_stage3_test.py",
        workspace / "tools/human2robot_v04_stage3_suite.py",
        workspace / "tools/human2robot_v04.py",
        workspace / "tools/human2robot_v04_data.py",
        workspace / "tools/human2robot_v04_retrieval.py",
        workspace / "cosmos_policy/datasets/human2robot_v04_sampler.py",
        workspace / "cosmos_policy/datasets/human2robot_v04_retrieval.py",
        workspace / "方案/v04/RECAP_Human2Robot_无泄漏单seed离线复现执行总计划.md",
        workspace / "docs/pusht_rag_docker_runbook.md",
    ]
    source_bindings = [bind_file(path) for path in controlled_paths]
    passed = process.returncode == 0 and passed_count >= MINIMUM_TEST_COUNT
    receipt = {
        "schema_version": SCHEMA,
        "status": "PASSED" if passed else "FAILED",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "formal_result": False,
        "features_generated": False,
        "training_started": False,
        "evaluation_started": False,
        "stage4_authorized_by_this_receipt": passed,
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
        "stage3_contract": contract_binding,
        "preflight_receipt": preflight_binding,
        "protocol_sha256": contract.get("protocol_sha256"),
        "split_sha256": contract.get("split_sha256"),
        "raw_inventory_sha256": contract.get("raw_inventory_sha256"),
        "source_bindings": source_bindings,
        "source_bundle_sha256": canonical_sha256(source_bindings),
        "output_tail": process.stdout[-16000:],
    }
    write_json_atomic(receipt_path.resolve(), receipt)
    require(passed, f"Full Human2Robot stage-3 Docker suite failed; see {receipt_path}")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--stage3-contract-path", type=Path, required=True)
    parser.add_argument("--preflight-receipt-path", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_suite(
        workspace=args.workspace,
        receipt_path=args.receipt_path,
        stage3_contract_path=args.stage3_contract_path,
        preflight_receipt_path=args.preflight_receipt_path,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Stage3SuiteError as error:
        print(f"stage-3 suite error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
