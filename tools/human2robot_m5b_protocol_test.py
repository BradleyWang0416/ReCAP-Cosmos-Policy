from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

try:
    from tools.human2robot_m5b_protocol import (
        DEFAULT_PROTOCOL,
        FROZEN_SEEDS,
        ProtocolError,
        read_json,
        validate_protocol,
        validate_protocol_file,
    )
except ModuleNotFoundError:
    from human2robot_m5b_protocol import (
        DEFAULT_PROTOCOL,
        FROZEN_SEEDS,
        ProtocolError,
        read_json,
        validate_protocol,
        validate_protocol_file,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_protocol_passes_and_has_exactly_three_seeds() -> None:
    protocol = read_json(REPO_ROOT / DEFAULT_PROTOCOL)
    checks = validate_protocol(protocol)
    assert all(checks.values())
    assert protocol["frozen_training_protocol"]["optimization"]["seeds"] == FROZEN_SEEDS


def test_protocol_file_binds_current_parent_artifacts() -> None:
    result = validate_protocol_file(REPO_ROOT / DEFAULT_PROTOCOL, REPO_ROOT)
    assert result["validation_status"] == "passed"
    assert all(result["parent_artifact_checks"].values())


def test_seed_mutation_hard_fails() -> None:
    protocol = read_json(REPO_ROOT / DEFAULT_PROTOCOL)
    changed = copy.deepcopy(protocol)
    changed["frozen_training_protocol"]["optimization"]["seeds"] = [1, 2, 3]
    with pytest.raises(ProtocolError, match="exactly_three_frozen_seeds"):
        validate_protocol(changed)


def test_window_level_statistics_hard_fail() -> None:
    protocol = read_json(REPO_ROOT / DEFAULT_PROTOCOL)
    changed = json.loads(json.dumps(protocol))
    changed["statistical_protocol"]["statistical_unit"] = "window"
    with pytest.raises(ProtocolError, match="statistics_prevent_pseudoreplication"):
        validate_protocol(changed)
