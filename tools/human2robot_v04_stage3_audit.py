#!/usr/bin/env python3
"""Audit and freeze the Human2Robot v04 stage-3 experiment interface."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

from tools import human2robot_v04_experiment as interface


SCHEMA = "human2robot-v04-stage3-contract-v1"
EXPECTED_STAGE1_INTERFACE_SHA256 = "8cbf7f5f3adbaa09c9bab4196c2b945289173064a0f50af7273bf15da9b26b00"
EXPECTED_STAGE2_CONTRACT_SHA256 = "27eceb61565d01297d4ec4ff19d166b5ff5c8d5e9af7916d92d5d9837af651d9"
EXPECTED_STAGE2_AUDIT_SHA256 = "80a1ba97679b404528110aa9917658dd6e5835fc4f5ce6d66dfb4dfd3215f919"
EXPECTED_STAGE2_SUITE_SHA256 = "ee36ac7348e1d9a8269c21ba137f18fe6142587175deadd1e02e16cfbfb2e375"


class Stage3AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage3AuditError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"Missing stage-3 input: {resolved}")
    return {"path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": file_sha256(resolved)}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: Mapping[str, Any], *, immutable: bool) -> None:
    if immutable:
        require(not path.exists(), f"Refusing to replace immutable stage-3 contract: {path}")
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
        if partial.exists():
            partial.unlink()


def _unwrap_preflight(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    result = receipt.get("result")
    return result if isinstance(result, Mapping) else receipt


def _validate_stage1_generation_binding(workspace: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    generation = manifest.get("generation_code", [])
    require(isinstance(generation, list), "Stage-1 generation_code binding is missing")
    rows = {Path(str(row["path"])).name: row for row in generation if isinstance(row, Mapping)}
    frozen_row = rows.get("human2robot_v04.py")
    require(frozen_row is not None, "Stage-1 manifest does not bind the frozen entry point")
    frozen_path = workspace / "tools/human2robot_v04.py"
    actual = bind_file(frozen_path)
    require(actual["sha256"] == EXPECTED_STAGE1_INTERFACE_SHA256, "Frozen stage-1 entry point changed")
    require(actual["sha256"] == frozen_row.get("sha256"), "Stage-1 generation-code hash no longer matches manifest")
    return actual


def command_contract() -> dict[str, Any]:
    return {
        "entry_point": "tools/human2robot_v04_experiment.py",
        "default_mode": "dry_run",
        "real_operation_requires_execute": True,
        "commands": {
            "prepare-data": {"execute_stage": 1, "gpu": False},
            "audit-data": {"execute_stage": 1, "gpu": False},
            "prepare-features": {"execute_stage": 4, "gpu": True},
            "preflight": {"execute_stage": 3, "gpu": True},
            "train": {"execute_stage": 5, "methods": list(interface.TRAIN_METHODS), "gpu": True},
            "evaluate": {
                "execute_stage": [6, 7],
                "splits": list(interface.EVALUATION_SPLITS),
                "methods": list(interface.EVALUATION_METHODS),
                "pool_sizes": list(interface.POOL_SIZES),
                "gpu": True,
            },
            "evaluate-oracle-phase": {
                "execute_stage": 7,
                "requires_completed_primary_receipt_sha256": True,
                "pool_size": 10,
                "diagnostic_only": True,
                "gpu": True,
            },
            "report": {"execute_stage": 7, "requires_final_receipt": True, "gpu": False},
        },
    }


def state_machine_contract() -> dict[str, Any]:
    states = [
        {"id": "prepare", "depends_on": [], "command": "prepare-features"},
        {"id": "preflight", "depends_on": ["prepare"], "command": "preflight"},
        {"id": "train_no_retrieval", "depends_on": ["preflight"], "command": "train --method no_retrieval"},
        {"id": "train_co_training", "depends_on": ["train_no_retrieval"], "command": "train --method co_training"},
        {"id": "train_recap_hand_ret", "depends_on": ["train_co_training"], "command": "train --method recap_hand_ret"},
        {
            "id": "dev",
            "depends_on": ["train_no_retrieval", "train_co_training", "train_recap_hand_ret"],
            "command": "evaluate --split dev",
        },
        {"id": "final", "depends_on": ["dev"], "command": "evaluate --split final"},
        {"id": "report", "depends_on": ["final"], "command": "report"},
    ]
    return {
        "states": states,
        "state_count": len(states),
        "cell_registry": None,
        "matrix_cell_count": None,
        "fixed_training_order": list(interface.TRAIN_METHODS),
        "oracle_phase_outside_primary_state_machine": True,
    }


def audit_interface(
    *,
    workspace: Path,
    output_path: Path,
    preflight_receipt_path: Path,
    stage2_audit_receipt_path: Path,
    stage2_suite_receipt_path: Path,
    execute: bool,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    if not execute:
        return {
            "schema_version": SCHEMA,
            "status": "DRY_RUN",
            "output_path": str(output_path.resolve()),
            "training_allowed": False,
            "features_generated": False,
            "training_started": False,
            "planned_checks": [
                "stage-1 generation-code hash remains frozen",
                "stage-2 audit and full-suite authorization",
                "exact dry-run command surface",
                "eight-state v04-only experiment state machine",
                "immutable manifest and receipt contract",
            ],
        }

    stage1_manifest_path = workspace / "data/Human2Robot/derived/v04/source_split_manifest.json"
    stage1_lock_path = workspace / "data/Human2Robot/derived/v04/source_split_manifest.lock.json"
    stage1_audit_path = workspace / "data/Human2Robot/derived/v04/stage1_data_audit_report.json"
    stage2_contract_path = workspace / "data/Human2Robot/derived/v04/stage2_retrieval_contract_report.json"
    bindings = {
        "stage1_manifest": bind_file(stage1_manifest_path),
        "stage1_lock": bind_file(stage1_lock_path),
        "stage1_audit": bind_file(stage1_audit_path),
        "stage2_contract": bind_file(stage2_contract_path),
        "stage2_audit_receipt": bind_file(stage2_audit_receipt_path),
        "stage2_suite_receipt": bind_file(stage2_suite_receipt_path),
        "preflight_receipt": bind_file(preflight_receipt_path),
    }
    require(bindings["stage2_contract"]["sha256"] == EXPECTED_STAGE2_CONTRACT_SHA256, "Stage-2 contract hash mismatch")
    require(bindings["stage2_audit_receipt"]["sha256"] == EXPECTED_STAGE2_AUDIT_SHA256, "Stage-2 audit hash mismatch")
    require(bindings["stage2_suite_receipt"]["sha256"] == EXPECTED_STAGE2_SUITE_SHA256, "Stage-2 suite hash mismatch")

    manifest = read_json(stage1_manifest_path)
    lock = read_json(stage1_lock_path)
    stage1_audit = read_json(stage1_audit_path)
    stage2_contract = read_json(stage2_contract_path)
    stage2_audit = read_json(stage2_audit_receipt_path)
    stage2_suite = read_json(stage2_suite_receipt_path)
    preflight_receipt = read_json(preflight_receipt_path)
    preflight = _unwrap_preflight(preflight_receipt)

    require(
        lock.get("manifest", {}).get("sha256") == bindings["stage1_manifest"]["sha256"],
        "Stage-1 lock mismatch",
    )
    require(stage1_audit.get("status") == "PASSED", "Stage-1 audit report is not passed")
    require(stage2_contract.get("status") == "VERIFIED_STAGE2", "Stage-2 contract is not verified")
    require(stage2_audit.get("status") == "PASSED", "Stage-2 audit receipt is not passed")
    require(stage2_suite.get("status") == "PASSED", "Stage-2 full-suite receipt is not passed")
    require(stage2_suite.get("stage3_authorized_by_this_receipt") is True, "Stage-2 suite does not authorize stage 3")
    require(stage2_suite.get("training_allowed") is False, "Stage-2 suite unexpectedly authorizes training")
    require(preflight.get("status") == "PASSED", "Formal stage-3 preflight is not passed")
    require(preflight.get("formal_v04_allowed") is True, "Formal stage-3 preflight does not allow v04")
    require(preflight.get("blockers") == [], "Formal stage-3 preflight has blockers")
    frozen_stage1_entry = _validate_stage1_generation_binding(workspace, manifest)

    commands = command_contract()
    require(tuple(commands["commands"].keys()) == interface.command_names(), "Public command set or order differs from frozen stage 3")
    state_machine = state_machine_contract()
    require(state_machine["state_count"] == 8, "Stage-3 state machine is not the frozen eight-state workflow")
    require(state_machine["cell_registry"] is None, "Stage-3 state machine must not use a cell registry")
    source_paths = [
        workspace / "tools/human2robot_v04_experiment.py",
        workspace / "tools/human2robot_v04_stage3_audit.py",
        workspace / "tools/human2robot_v04_stage3_test.py",
        workspace / "tools/human2robot_v04_stage3_suite.py",
        workspace / "方案/v04/RECAP_Human2Robot_无泄漏单seed离线复现执行总计划.md",
        workspace / "docs/pusht_rag_docker_runbook.md",
    ]
    source_bindings = [bind_file(path) for path in source_paths]
    contract = {
        "schema_version": SCHEMA,
        "status": "VERIFIED_STAGE3",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "formal_result": False,
        "scientific_results_generated": False,
        "features_generated": False,
        "training_started": False,
        "evaluation_started": False,
        "frozen_stage1_entry": frozen_stage1_entry,
        "inputs": bindings,
        "protocol_sha256": manifest.get("protocol_sha256"),
        "split_sha256": manifest.get("split_sha256"),
        "raw_inventory_sha256": manifest.get("raw_inventory_sha256"),
        "interface": commands,
        "state_machine": state_machine,
        "paths": {
            "formal_run_root": "/DATA1/wxs/ReCAP_M5B_V04_RUNS",
            "derived_root": "/workspace/data/Human2Robot/derived/v04",
            "v03_cell_registry_reuse_allowed": False,
            "v03_artifact_write_allowed": False,
        },
        "audit_guarantees": {
            "all_public_commands_default_dry_run": True,
            "execute_required_for_mutation_or_gpu": True,
            "each_invocation_has_independent_manifest": True,
            "each_invocation_has_immutable_receipt": True,
            "later_stage_execute_fail_closed_until_authorized": True,
            "primary_phase_configuration_allowed": False,
        },
        "source_bindings": source_bindings,
        "source_bundle_sha256": canonical_sha256(source_bindings),
        "future_stage_authorization": {"stage4_allowed": True, "training_allowed": False},
    }
    write_json_atomic(output_path.resolve(), contract, immutable=True)
    return contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--output-path", type=Path, default=Path("/workspace/data/Human2Robot/derived/v04/stage3_experiment_interface.json"))
    parser.add_argument("--preflight-receipt-path", type=Path, required=True)
    parser.add_argument("--stage2-audit-receipt-path", type=Path, required=True)
    parser.add_argument("--stage2-suite-receipt-path", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = audit_interface(
        workspace=args.workspace,
        output_path=args.output_path,
        preflight_receipt_path=args.preflight_receipt_path,
        stage2_audit_receipt_path=args.stage2_audit_receipt_path,
        stage2_suite_receipt_path=args.stage2_suite_receipt_path,
        execute=args.execute,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["status"] in {"DRY_RUN", "VERIFIED_STAGE3"} else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Stage3AuditError as error:
        print(f"stage-3 audit error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
