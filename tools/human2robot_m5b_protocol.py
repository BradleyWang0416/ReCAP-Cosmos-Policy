#!/usr/bin/env python3
"""Validate and lock the preregistered Human2Robot M5-B protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "human2robot-m5b-formal-acceptance-protocol-v1"
DEFAULT_PROTOCOL = Path("方案/v03/M5B_formal_acceptance_protocol_v1.json")
DEFAULT_LOCK = Path("方案/v03/M5B_formal_acceptance_protocol_v1.lock.json")
FROZEN_SEEDS = [20260711, 20260712, 20260713]
REQUIRED_METHODS = {"no_retrieval", "retrieval_only", "co_training", "recap_hand_ret"}
REQUIRED_EXPERIMENTS = {
    "M5B-MAIN-01",
    "M5B-REP-01",
    "M5B-ACTION-01",
    "M5B-RET-01",
    "M5B-SENS-01",
    "M5B-TIME-01",
    "M5B-RES-01",
    "M5B-QUAL-01",
}
REQUIRED_GATES = {
    "M5B-P0-IMPLEMENTATION",
    "M5B-P1-DATA",
    "M5B-P2-RUN-COMPLETENESS",
    "M5B-G1-MAIN",
    "M5B-G2-POOL-GROWTH",
    "M5B-G3-MECHANISM",
    "M5B-G4-SENSITIVITY",
    "M5B-G5-TEMPORAL",
    "M5B-G6-RESOLUTION",
    "M5B-G7-GUARDRAILS",
    "M5B-G8-REPORTING",
}


class ProtocolError(RuntimeError):
    """Raised when a preregistered protocol invariant is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolError(f"Missing protocol or parent artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def validate_protocol(protocol: dict[str, Any]) -> dict[str, bool]:
    training = protocol.get("frozen_training_protocol", {})
    model = training.get("model", {})
    optimization = training.get("optimization", {})
    checkpoint = training.get("checkpoint", {})
    statistics = protocol.get("statistical_protocol", {})
    data = protocol.get("frozen_data_contract", {})
    metric = protocol.get("metric_registry", {})
    experiments = {item.get("experiment_id") for item in protocol.get("experiment_matrix", [])}
    methods = {item.get("method_id") for item in protocol.get("required_methods", [])}
    gates = {item.get("gate_id") for item in protocol.get("acceptance_gates", [])}
    seeds = optimization.get("seeds", [])

    checks = {
        "schema_frozen": protocol.get("schema_version") == SCHEMA_VERSION
        and protocol.get("status") == "frozen_pre_registration",
        "exactly_three_frozen_seeds": seeds == FROZEN_SEEDS
        and optimization.get("seed_count") == 3
        and len(set(seeds)) == 3,
        "formal_model_frozen": model.get("parameter_scale") == "2B"
        and model.get("action_dim") == 10
        and model.get("proprio_dim") == 10
        and model.get("H_steps") == 8
        and model.get("K_steps") == 8
        and model.get("state_t") == 10,
        "optimization_frozen": optimization.get("max_optimizer_steps") == 7000
        and optimization.get("batch_size_per_data_parallel_rank") == 25
        and optimization.get("learning_rate") == 0.0001
        and optimization.get("gradient_accumulation_steps") == 1
        and optimization.get("early_stopping") is False,
        "checkpoint_selection_frozen": checkpoint.get("save_every_steps") == 1000
        and checkpoint.get("primary_checkpoint_step") == 7000
        and checkpoint.get("saved_steps") == list(range(1000, 7001, 1000))
        and "no heldout" in checkpoint.get("selection_rule", ""),
        "required_methods_complete": methods == REQUIRED_METHODS,
        "required_experiments_complete": experiments == REQUIRED_EXPERIMENTS,
        "required_gates_complete": gates == REQUIRED_GATES,
        "primary_metric_frozen": metric.get("primary", {}).get("metric_id")
        == "position_error_median_canonical"
        and metric.get("primary", {}).get("direction") == "lower_is_better",
        "statistics_prevent_pseudoreplication": statistics.get("statistical_unit")
        == "heldout_task_x_seed"
        and statistics.get("forbidden_unit") == "individual windows or chunks"
        and statistics.get("experimental_seed_count") == 3
        and statistics.get("expected_primary_units_per_method") == 12,
        "bootstrap_and_multiplicity_frozen": statistics.get("bootstrap", {}).get("resamples")
        == 10000
        and statistics.get("hypothesis_test", {}).get("multiple_comparison_correction")
        == "Holm for the two primary baseline comparisons",
        "independent_pool_requirement_frozen": data.get(
            "required_heldout_independent_human_demos_per_task"
        )
        == 10
        and data.get("independence_unit")
        == "source episode, not a window or chunk from the same episode",
        "deployment_boundary_preserved": data.get("query_command_status") == "unverified"
        and data.get("deployment_command_adapter_id") is None,
        "negative_controls_not_main": next(
            item for item in protocol["experiment_matrix"] if item["experiment_id"] == "M5B-ACTION-01"
        ).get("negative_controls_forbidden_as_main")
        is True,
        "all_experiments_preregistered": all(
            item.get("status") == "PROPOSED" and item.get("required") is True
            for item in protocol.get("experiment_matrix", [])
        ),
        "m5_decision_requires_every_gate": protocol.get("full_m5_decision_rule", "").startswith(
            "M5-v03 is passed only if every"
        ),
        "execution_readiness_not_overclaimed": protocol.get("current_execution_readiness", {}).get(
            "status"
        )
        == "NOT_READY",
    }
    for name, passed in checks.items():
        _require(passed, f"Protocol validation failed: {name}")
    return checks


def verify_parent_artifacts(protocol: dict[str, Any], repo_root: Path) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for item in protocol.get("parent_artifacts", []):
        path = repo_root / item["path"]
        passed = path.is_file() and file_sha256(path) == item["file_sha256"]
        results[item["role"]] = passed
        _require(passed, f"Parent artifact hash mismatch: {item['path']}")
    return results


def validate_protocol_file(path: Path, repo_root: Path) -> dict[str, Any]:
    protocol = read_json(path)
    checks = validate_protocol(protocol)
    parent_checks = verify_parent_artifacts(protocol, repo_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "validation_status": "passed",
        "protocol_file_sha256": file_sha256(path),
        "validator_file_sha256": file_sha256(Path(__file__)),
        "checks": checks,
        "parent_artifact_checks": parent_checks,
        "seed_count": 3,
        "seeds": FROZEN_SEEDS,
        "execution_readiness": protocol["current_execution_readiness"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--write-lock", action="store_true")
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = validate_protocol_file(args.protocol, args.repo_root)
    except (ProtocolError, OSError, ValueError, KeyError, StopIteration) as exc:
        print(json.dumps({"validation_status": "failed", "error": str(exc)}, indent=2))
        return 1
    if args.write_lock:
        lock = {"locked_at_utc": utc_now(), **result}
        args.lock_path.parent.mkdir(parents=True, exist_ok=True)
        args.lock_path.write_text(
            json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
