from __future__ import annotations

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
