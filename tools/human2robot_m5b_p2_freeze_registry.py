#!/usr/bin/env python3
"""Materialize the frozen, still-pending M5B-P2 registry from the approved scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.human2robot_m5b_p2_registry import build_candidate_registry

SCHEMA_VERSION = "human2robot-m5b-p2-cell-registry-v1"
STATUS = "frozen_pending_execution"
SUPPLEMENT_ID = "m5b_p2_claim_centered_execution_v1"
SUPPLEMENT_PATH = Path("方案/v03/M5B_P2_execution_supplement_v1.json")
SUPPLEMENT_SHA256 = "be6ca3cdeb7d725221cbefa4664a44f33531edea1b66a74ea2405bff54dfc4ba"
SUPPLEMENT_LOCK_PATH = Path("方案/v03/M5B_P2_execution_supplement_v1.lock.json")
PROPOSAL_SHA256 = "edf692ea17242458e0e133d1dcc25685d4b02e7964845d2c2ee8fbb2a66ad733"
CANDIDATE_GENERATOR_SHA256 = (
    "8765d24606db00a8b875195c760092f2a1f7b4c28dda8db6564ad52b1ca6c0bd"
)
EXPECTED_COUNTS = {
    "learned_training_checkpoint": 48,
    "nonlearned_method_artifact": 3,
    "checkpoint_linked_evaluation": 147,
    "aggregate_report": 4,
}


class RegistryFreezeError(RuntimeError):
    """Raised when an approved registry binding has drifted."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryFreezeError(message)


def build_frozen_registry(workspace: Path) -> dict[str, Any]:
    supplement_path = workspace / SUPPLEMENT_PATH
    supplement_lock_path = workspace / SUPPLEMENT_LOCK_PATH
    candidate_generator_path = workspace / "tools/human2robot_m5b_p2_registry.py"
    proposal_path = workspace / "方案/v03/M5B_P2_execution_supplement_v0.proposed.json"
    _require(file_sha256(supplement_path) == SUPPLEMENT_SHA256, "Supplement SHA256 drift")
    _require(file_sha256(proposal_path) == PROPOSAL_SHA256, "Approved proposal SHA256 drift")
    _require(
        file_sha256(candidate_generator_path) == CANDIDATE_GENERATOR_SHA256,
        "Candidate registry generator SHA256 drift",
    )
    lock = json.loads(supplement_lock_path.read_text(encoding="utf-8"))
    _require(lock.get("status") == "locked", "Execution supplement is not locked")
    _require(
        lock.get("supplement_file_sha256") == SUPPLEMENT_SHA256,
        "Execution supplement lock mismatch",
    )

    candidate = build_candidate_registry()
    _require(candidate.get("counts") == EXPECTED_COUNTS, "Candidate artifact counts changed")
    _require(candidate.get("cell_count") == 202, "Candidate cell count changed")
    cells = []
    for candidate_cell in candidate["cells"]:
        cell = dict(candidate_cell)
        cell["status"] = "pending"
        cell["formal_result"] = False
        cells.append(cell)
    _require(len({cell["cell_id"] for cell in cells}) == len(cells), "Duplicate cell IDs")
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_id": "m5b_p2_claim_centered_202_cells_v1",
        "status": STATUS,
        "formal_queue_allowed": False,
        "p2_acceptance_allowed": False,
        "supplement_id": SUPPLEMENT_ID,
        "supplement_path": SUPPLEMENT_PATH.as_posix(),
        "supplement_file_sha256": SUPPLEMENT_SHA256,
        "supplement_lock_path": SUPPLEMENT_LOCK_PATH.as_posix(),
        "approved_proposal_file_sha256": PROPOSAL_SHA256,
        "candidate_generator_file_sha256": CANDIDATE_GENERATOR_SHA256,
        "seeds": candidate["seeds"],
        "required_experiment_ids": candidate["required_experiment_ids"],
        "scope_interpretation": candidate["scope_interpretation"],
        "counts": EXPECTED_COUNTS,
        "cell_count": len(cells),
        "cells_payload_sha256": canonical_json_sha256(cells),
        "cells": cells,
        "current_blocker": "All cell handlers and Docker contract tests must pass before queue activation.",
    }


def main() -> int:
    workspace = Path("/workspace")
    print(json.dumps(build_frozen_registry(workspace), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
