#!/usr/bin/env python3
"""Run the frozen pre-launch suite for Human2Robot v04 stage 5."""

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


SCHEMA = "human2robot-v04-stage5-suite-v1"
MINIMUM_TEST_COUNT = 215
STAGE4_LOCK_SHA256 = "834fe271bfe964c0b4f6e6c3b8ba899191609643acd166df0fa8e2811fca7ade"
EXPECTED_OFFLINE_ENV = {
    "COSMOS_SKIP_HF_AUTO_DOWNLOAD": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
}


class Stage5SuiteError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage5SuiteError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"Missing stage-5 suite input: {resolved}")
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": file_sha256(resolved)}


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"Refusing to replace immutable stage-5 suite receipt: {path}")
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


def run_suite(*, workspace: Path, receipt_path: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    require(Path("/.dockerenv").is_file(), "Stage-5 suite must run inside Docker")
    require(sys.executable == str(workspace / ".venv/bin/python"), f"Unexpected Python: {sys.executable}")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 4, "Stage-5 suite requires exactly four GPUs")
    offline = {key: os.environ.get(key) for key in EXPECTED_OFFLINE_ENV}
    require(offline == EXPECTED_OFFLINE_ENV, f"Offline environment mismatch: {offline}")
    stage4_lock = bind_file(workspace / "data/Human2Robot/derived/v04/stage4_protocol_lock.json")
    require(stage4_lock["sha256"] == STAGE4_LOCK_SHA256, "Stage-4 protocol lock SHA drift")

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
        workspace / "tools/human2robot_v04_experiment.py",
        workspace / "tools/human2robot_v04_stage5.py",
        workspace / "tools/human2robot_v04_stage5_suite.py",
        workspace / "tools/human2robot_v04_stage5_test.py",
        workspace / "cosmos_policy/datasets/human2robot_v04_dataset.py",
        workspace / "cosmos_policy/datasets/human2robot_v04_sampler.py",
        workspace / "cosmos_policy/config/experiment/human2robot_v04_experiment_configs.py",
        workspace / "cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py",
        workspace / "cosmos_policy/scripts/train.py",
        workspace / "cosmos_policy/trainer.py",
        workspace / "cosmos_policy/scripts/train_human2robot_v04.py",
        workspace / "cosmos_policy/trainer_human2robot_v04.py",
        workspace / "方案/v04/RECAP_Human2Robot_无泄漏单seed离线复现执行总计划.md",
        workspace / "方案/v04/CHANGELOG.md",
    ]
    source_bindings = [bind_file(path) for path in controlled_paths]
    passed = process.returncode == 0 and passed_count >= MINIMUM_TEST_COUNT
    receipt = {
        "schema_version": SCHEMA,
        "status": "PASSED" if passed else "FAILED",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage5_launch_allowed": passed,
        "training_started": False,
        "formal_result": False,
        "performance_claim_allowed": False,
        "image": os.environ.get("HUMAN2ROBOT_V04_IMAGE"),
        "image_id": os.environ.get("HUMAN2ROBOT_V04_IMAGE_ID"),
        "host_gpu_devices": os.environ.get("HUMAN2ROBOT_V04_GPU_DEVICES"),
        "visible_gpu_count": torch.cuda.device_count(),
        "offline_environment": offline,
        "stage4_protocol_lock": stage4_lock,
        "command": command,
        "returncode": process.returncode,
        "passed_test_count": passed_count,
        "minimum_test_count": MINIMUM_TEST_COUNT,
        "source_bindings": source_bindings,
        "source_bundle_sha256": canonical_sha256(source_bindings),
        "output_tail": process.stdout[-20000:],
    }
    write_json_atomic(receipt_path.resolve(), receipt)
    require(passed, f"Stage-5 suite failed; see {receipt_path}")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--receipt-path", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_suite(workspace=args.workspace, receipt_path=args.receipt_path)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Stage5SuiteError as error:
        print(f"stage-5 suite error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
