from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import human2robot_v04_experiment as interface
from tools import human2robot_v04_stage3_audit as audit


def test_public_command_surface_is_exact_and_dry_run_by_default() -> None:
    parser = interface.build_parser()
    argv_by_command = {
        "prepare-data": ["prepare-data"],
        "audit-data": ["audit-data"],
        "prepare-features": ["prepare-features"],
        "preflight": ["preflight"],
        "train": ["train", "--method", "no_retrieval"],
        "evaluate": ["evaluate", "--split", "dev", "--method", "recap_hand_ret", "--pool-size", "10"],
        "evaluate-oracle-phase": [
            "evaluate-oracle-phase",
            "--split",
            "final",
            "--primary-receipt-sha256",
            "a" * 64,
        ],
        "report": ["report"],
    }
    assert tuple(argv_by_command) == interface.PUBLIC_COMMANDS
    for command, argv in argv_by_command.items():
        args = parser.parse_args(argv)
        assert args.command == command
        assert args.execute is False


def test_train_method_is_required_and_restricted() -> None:
    parser = interface.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--method", "retrieval_only"])


def test_evaluate_contract_rejects_noncanonical_baseline_pool() -> None:
    args = interface.build_parser().parse_args(
        ["evaluate", "--split", "dev", "--method", "no_retrieval", "--pool-size", "1"]
    )
    with pytest.raises(interface.Stage3InterfaceError, match="canonical pool-size 10"):
        interface.validate_args(args)


def test_executed_evaluation_requires_explicit_checkpoint() -> None:
    args = interface.build_parser().parse_args(
        ["evaluate", "--split", "final", "--method", "recap_hand_ret", "--pool-size", "10", "--execute"]
    )
    with pytest.raises(interface.Stage3InterfaceError, match="explicitly bound checkpoint"):
        interface.validate_args(args)


def test_oracle_phase_requires_primary_receipt_sha_and_pool10() -> None:
    parser = interface.build_parser()
    args = parser.parse_args(
        ["evaluate-oracle-phase", "--split", "final", "--primary-receipt-sha256", "not-a-sha"]
    )
    with pytest.raises(interface.Stage3InterfaceError, match="completed primary receipt"):
        interface.validate_args(args)
    args = parser.parse_args(
        [
            "evaluate-oracle-phase",
            "--split",
            "final",
            "--pool-size",
            "8",
            "--primary-receipt-sha256",
            "b" * 64,
        ]
    )
    with pytest.raises(interface.Stage3InterfaceError, match="pool-size 10"):
        interface.validate_args(args)


def test_later_stage_execute_is_fail_closed_without_starting_work() -> None:
    args = interface.build_parser().parse_args(["train", "--method", "recap_hand_ret", "--execute"])
    result = interface.dispatch(args, workspace=Path("/workspace"))
    assert result["status"] == "BLOCKED_STAGE_GATE"
    assert result["required_stage"] == 5
    assert result["training_started"] is False
    assert result["features_generated"] is False
    assert result["evaluation_started"] is False


def test_dry_run_never_claims_feature_training_or_evaluation_output() -> None:
    args = interface.build_parser().parse_args(["prepare-features"])
    result = interface.dispatch(args, workspace=Path("/workspace"))
    assert result["status"] == "DRY_RUN"
    assert result["planned_gpu_operation"] is True
    assert result["features_generated"] is False
    assert result["training_started"] is False
    assert result["evaluation_started"] is False


def test_immutable_json_refuses_replacement_and_leaves_no_partial(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    interface.write_json_atomic(path, {"status": "DRY_RUN"}, immutable=True)
    assert path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(interface.Stage3InterfaceError, match="Refusing to replace"):
        interface.write_json_atomic(path, {"status": "FAILED"}, immutable=True)
    assert not list(tmp_path.rglob("*.partial"))


def test_each_invocation_gets_manifest_receipt_and_new_attempt(tmp_path: Path) -> None:
    workspace = Path("/workspace")
    args = interface.build_parser().parse_args(["report", "--run-id", "stage3_test"])
    assert interface.execute_with_audit(args, workspace=workspace, run_root=tmp_path) == 0
    assert interface.execute_with_audit(args, workspace=workspace, run_root=tmp_path) == 0
    run_dir = tmp_path / "orchestrator_logs/stage3_test"
    manifests = sorted(run_dir.glob("attempt_*.manifest.json"))
    receipts = sorted(run_dir.glob("attempt_*.receipt.json"))
    assert [path.name for path in manifests] == ["attempt_0001.manifest.json", "attempt_0002.manifest.json"]
    assert [path.name for path in receipts] == ["attempt_0001.receipt.json", "attempt_0002.receipt.json"]
    for manifest_path, receipt_path in zip(manifests, receipts, strict=True):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["manifest"]["sha256"] == interface.file_sha256(manifest_path)
        assert manifest["dry_run"] is True
        assert manifest["v03_registry_or_artifact_reuse_allowed"] is False
        assert manifest_path.stat().st_mode & 0o777 == 0o444
        assert receipt_path.stat().st_mode & 0o777 == 0o444
    assert not list(tmp_path.rglob("*.partial"))


def test_state_machine_is_eight_states_without_cell_registry() -> None:
    contract = audit.state_machine_contract()
    assert contract["state_count"] == 8
    assert contract["cell_registry"] is None
    assert contract["matrix_cell_count"] is None
    assert contract["fixed_training_order"] == list(interface.TRAIN_METHODS)
    assert [row["id"] for row in contract["states"]] == [
        "prepare",
        "preflight",
        "train_no_retrieval",
        "train_co_training",
        "train_recap_hand_ret",
        "dev",
        "final",
        "report",
    ]


def test_stage3_audit_defaults_to_nonmutating_dry_run(tmp_path: Path) -> None:
    result = audit.audit_interface(
        workspace=Path("/workspace"),
        output_path=tmp_path / "contract.json",
        preflight_receipt_path=tmp_path / "preflight.json",
        stage2_audit_receipt_path=tmp_path / "stage2-audit.json",
        stage2_suite_receipt_path=tmp_path / "stage2-suite.json",
        execute=False,
    )
    assert result["status"] == "DRY_RUN"
    assert result["training_allowed"] is False
    assert result["features_generated"] is False
    assert result["training_started"] is False
    assert not (tmp_path / "contract.json").exists()


def test_frozen_stage1_entry_hash_is_unchanged() -> None:
    path = Path("/workspace/tools/human2robot_v04.py")
    assert interface.file_sha256(path) == audit.EXPECTED_STAGE1_INTERFACE_SHA256
