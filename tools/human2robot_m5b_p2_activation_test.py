from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.human2robot_m5b_p2_activation import (
    ActivationContractError,
    issue_launch_activation,
)


WORKSPACE = Path(__file__).resolve().parents[1]


def test_launch_issuer_requires_a_real_passed_docker_receipt_before_preflight(tmp_path: Path) -> None:
    with pytest.raises(ActivationContractError, match="Missing JSON"):
        issue_launch_activation(
            WORKSPACE,
            tmp_path,
            tmp_path / "missing_docker_suite_receipt.json",
        )


def test_launch_issuer_rejects_receipt_from_non_four_gpu_container(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "human2robot-m5b-p2-docker-suite-receipt-v6",
                "status": "passed",
                "visible_gpu_count": 8,
                "passed_test_count": 999,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ActivationContractError, match="four-GPU environment"):
        issue_launch_activation(WORKSPACE, tmp_path, receipt)
