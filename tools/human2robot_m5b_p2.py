#!/usr/bin/env python3
"""Formal Docker-only M5B-P2 training orchestrator and checkpoint auditor.

This module never treats a launch, a partial checkpoint, or a diagnostic run as
formal evidence.  It currently executes the nine P0-implemented learned-method
cells for ``M5B-MAIN-01`` and records the remaining frozen experiment families
as unsupported prerequisites.  Consequently the full P2 gate remains pending
until those families are implemented and their required cells complete.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from tools.human2robot_m5b_p2_registry import build_candidate_registry

SCHEMA_VERSION = "human2robot-m5b-p2-run-manifest-v1"
CELL_SCHEMA_VERSION = "human2robot-m5b-p2-cell-manifest-v1"
GATE_ID = "M5B-P2-RUN-COMPLETENESS"
PROTOCOL_ID = "m5b_v03_preregistered_3seed_formal_v1"
PROTOCOL_SHA256 = "7598dfa2ac2e129f5d21a295dad23b90f63c3c8e68811da73cbcc20eb95d5ce4"
SPLIT_SHA256 = "1d3ef2377aa19938b06646f6d5fc31ec9f275fc9f37e253e1e9aa5eecdc5a968"
INITIALIZATION_CHECKPOINT_SHA256 = (
    "565bbb2c9645737327983f4461e4d32627bba465b0a8dc26447edea144e1ff47"
)
INITIALIZATION_CHECKPOINT_PATH = Path(
    "/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/base/post-trained/"
    "81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
)
TOKENIZER_CHECKPOINT_PATH = Path(
    "/DATA1/wxs/_HUGGINGFACE/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth"
)
TOKENIZER_CHECKPOINT_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
P1_SELECTION_ID = "48e0c0f5c283a5a7b9f3de8eb6535f13f5f760cc325a81413053015fd6299afd"
PROPOSED_EXECUTION_SUPPLEMENT_PATH = Path(
    "方案/v03/M5B_P2_execution_supplement_v0.proposed.json"
)
PROPOSED_EXECUTION_SUPPLEMENT_SHA256 = (
    "edf692ea17242458e0e133d1dcc25685d4b02e7964845d2c2ee8fbb2a66ad733"
)
FROZEN_EXECUTION_SUPPLEMENT_PATH = Path("方案/v03/M5B_P2_execution_supplement_v1.json")
FROZEN_EXECUTION_SUPPLEMENT_SHA256 = (
    "be6ca3cdeb7d725221cbefa4664a44f33531edea1b66a74ea2405bff54dfc4ba"
)
FROZEN_EXECUTION_SUPPLEMENT_LOCK_PATH = Path(
    "方案/v03/M5B_P2_execution_supplement_v1.lock.json"
)
FROZEN_EXECUTION_SUPPLEMENT_LOCK_SHA256 = (
    "bdea172ada310f421c2398c57eb9536a8636ca3ba51302932de2e7b589fcff77"
)
FROZEN_CELL_REGISTRY_PATH = Path("方案/v03/M5B_P2_cell_registry_v1.json")
FROZEN_CELL_REGISTRY_SHA256 = (
    "4664d036bcf6bc41e8a44fac2afe04ff6de62c2a180a29d3433bd83e46604df5"
)
FROZEN_CELL_REGISTRY_LOCK_PATH = Path("方案/v03/M5B_P2_cell_registry_v1.lock.json")
FROZEN_CELL_REGISTRY_LOCK_SHA256 = (
    "a349b799e0364945a438d400cf74348ee9ba8c300a72e8f9cdb6b4202e9c3fba"
)
CANDIDATE_REGISTRY_GENERATOR_PATH = Path("tools/human2robot_m5b_p2_registry.py")
CANDIDATE_REGISTRY_GENERATOR_SHA256 = (
    "8765d24606db00a8b875195c760092f2a1f7b4c28dda8db6564ad52b1ca6c0bd"
)
FROZEN_REGISTRY_MATERIALIZER_PATH = Path("tools/human2robot_m5b_p2_freeze_registry.py")
FROZEN_REGISTRY_MATERIALIZER_SHA256 = (
    "ac15c5b748e06771fee9b7247672c03c0b34ded5110c5c686bc55e11183ab313"
)
FROZEN_CELL_COUNTS = {
    "learned_training_checkpoint": 48,
    "nonlearned_method_artifact": 3,
    "checkpoint_linked_evaluation": 147,
    "aggregate_report": 4,
}
FROZEN_CELL_COUNT = 202
FORMAL_OUTPUT_ROOT = Path("/DATA1/wxs/ReCAP_M5B_P2_RUNS")
FORMAL_SEEDS = (20260711, 20260712, 20260713)
LEARNED_METHODS = ("no_retrieval", "co_training", "recap_hand_ret")
FIXED_WORLD_SIZE = 8
FIXED_DP_WORLD_SIZE = 8
MAX_OPTIMIZER_STEPS = 7000
SAVE_EVERY_STEPS = 1000
BATCH_PER_DP_RANK = 25
H_STEPS = 8
K_STEPS = 8
SAVED_STEPS = tuple(range(1000, 7001, 1000))
MAIN_EXPERIMENT_ID = "M5B-MAIN-01"
REQUIRED_EXPERIMENT_IDS = (
    "M5B-MAIN-01",
    "M5B-REP-01",
    "M5B-ACTION-01",
    "M5B-RET-01",
    "M5B-SENS-01",
    "M5B-TIME-01",
    "M5B-RES-01",
    "M5B-QUAL-01",
)
CURRENTLY_IMPLEMENTED_CHECKPOINT_EXPERIMENT_IDS = (MAIN_EXPERIMENT_ID,)
UNRESOLVED_EXECUTION_DECISIONS = (
    {
        "decision_id": "P2-SCOPE-01",
        "question": (
            "Define the exact train-versus-evaluate cell scope and checkpoint-reuse rule "
            "for every frozen experiment variant."
        ),
        "why_blocking": (
            "The frozen protocol requires every method-experiment-seed cell to have a "
            "step-7000 checkpoint, while several robustness and qualitative variants are "
            "naturally evaluation-only."
        ),
    },
    {
        "decision_id": "P2-NONLEARNED-01",
        "question": (
            "Define the formal artifact replacing a step-7000 optimizer checkpoint for "
            "the nonlearned retrieval_only method."
        ),
        "why_blocking": "retrieval_only has no optimizer or learned checkpoint by definition.",
    },
    {
        "decision_id": "P2-REP-01",
        "question": (
            "Define the future_state target, loss, decoder, normalization, and evaluation "
            "mapping so it is distinct from the already-future absolute query target."
        ),
        "why_blocking": "The current adapter exposes residual and absolute targets only.",
    },
    {
        "decision_id": "P2-RET-01",
        "question": (
            "Freeze the random/phase/geometry/visual feature definitions, encoder checkpoint, "
            "index construction, top-k aggregation, and deterministic tie-breaking."
        ),
        "why_blocking": "The P1 human-only pool is not consumed by the current formal adapter.",
    },
    {
        "decision_id": "P2-VARIANTS-01",
        "question": (
            "Freeze executable materialization rules for action views, H/K alternatives, "
            "time perturbations, and the three resolution preprocessors."
        ),
        "why_blocking": (
            "The current adapter hard-requires the main action/time view, H/K=8/8, and 224 input."
        ),
    },
    {
        "decision_id": "P2-EVAL-01",
        "question": (
            "Freeze the held-out inference/evaluator contract, task-seed aggregation, guardrail "
            "counters, and qualitative case export schema."
        ),
        "why_blocking": "No formal Human2Robot held-out inference/evaluation runner exists yet.",
    },
)
REQUIRED_CHECKPOINT_BINDINGS = (
    "protocol_file_sha256",
    "code_sha256",
    "resolved_initialization_checkpoint_sha256",
    "canonical_schema",
    "split_sha256",
    "time_view_id",
    "pool_action_view_id",
    "query_action_view_id",
    "action_alignment_id",
    "view_id",
    "retrieval_index_sha256",
    "method_id",
    "experiment_id",
    "seed",
    "optimizer_steps",
    "batch_size_per_data_parallel_rank",
    "data_parallel_world_size",
    "H_steps",
    "K_steps",
)


class P2Error(RuntimeError):
    """Raised when formal P2 evidence is missing, inconsistent, or unsafe."""


@dataclass(frozen=True)
class MainTrainingCell:
    cell_id: str
    experiment_id: str
    method_id: str
    seed: int
    config_name: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise P2Error(message)


def require_full_docker_environment() -> None:
    require(
        Path("/.dockerenv").is_file(),
        "M5B-P2 commands must run inside the full project Docker environment",
    )


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Required JSON does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def exclusive_execution_lock(lock_path: Path, purpose: str) -> Iterator[None]:
    """Prevent concurrent formal jobs from sharing the fixed eight-GPU allocation."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_handle.seek(0)
            owner = lock_handle.read().strip() or "owner metadata unavailable"
            raise P2Error(
                f"Another M5B-P2 execution owns {lock_path}: {owner}"
            ) from error
        owner = {
            "pid": os.getpid(),
            "purpose": purpose,
            "acquired_at_utc": utc_now(),
        }
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(json.dumps(owner, sort_keys=True) + "\n")
        lock_handle.flush()
        os.fsync(lock_handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main_training_cells() -> list[MainTrainingCell]:
    return [
        MainTrainingCell(
            cell_id=f"{MAIN_EXPERIMENT_ID}__{method_id}__seed{seed}",
            experiment_id=MAIN_EXPERIMENT_ID,
            method_id=method_id,
            seed=seed,
            config_name=f"cosmos_predict2p5_2b_human2robot_{method_id}_seed{seed}",
        )
        for method_id in LEARNED_METHODS
        for seed in FORMAL_SEEDS
    ]


def protocol_experiment_coverage(
    protocol: dict[str, Any],
    *,
    execution_spec_frozen: bool,
    full_cell_registry_bound: bool,
) -> dict[str, Any]:
    matrix = protocol.get("experiment_matrix", [])
    protocol_ids = [item.get("experiment_id") for item in matrix if isinstance(item, dict)]
    require(tuple(protocol_ids) == REQUIRED_EXPERIMENT_IDS, "Frozen experiment matrix changed")
    unsupported = [
        experiment_id
        for experiment_id in REQUIRED_EXPERIMENT_IDS
        if experiment_id not in CURRENTLY_IMPLEMENTED_CHECKPOINT_EXPERIMENT_IDS
    ]
    return {
        "required_experiment_ids": list(REQUIRED_EXPERIMENT_IDS),
        "frozen_experiment_matrix": matrix,
        "checkpoint_execution_implemented": list(
            CURRENTLY_IMPLEMENTED_CHECKPOINT_EXPERIMENT_IDS
        ),
        "checkpoint_or_evaluation_execution_not_yet_implemented": unsupported,
        "unresolved_execution_decisions": [],
        "resolved_execution_decision_ids": [
            item["decision_id"] for item in UNRESOLVED_EXECUTION_DECISIONS
        ],
        "execution_supplement_status": "frozen_approved_execution_spec",
        "full_execution_spec_frozen": execution_spec_frozen,
        "full_cell_registry_bound": full_cell_registry_bound,
        "full_protocol_matrix_implemented": not unsupported,
        "claim_boundary": (
            "The execution semantics and 202-cell registry are frozen. The current runner "
            "still covers only nine learned cells in M5B-MAIN-01, so it cannot pass the "
            "full P2 gate while any registry handler is unsupported."
        ),
    }


def validate_protocol(workspace: Path) -> dict[str, Any]:
    protocol_path = workspace / "方案/v03/M5B_formal_acceptance_protocol_v1.json"
    require(file_sha256(protocol_path) == PROTOCOL_SHA256, "Frozen protocol SHA256 changed")
    protocol = read_json(protocol_path)
    require(protocol.get("protocol_id") == PROTOCOL_ID, "Frozen protocol ID changed")
    optimization = protocol["frozen_training_protocol"]["optimization"]
    checkpoint = protocol["frozen_training_protocol"]["checkpoint"]
    require(tuple(optimization["seeds"]) == FORMAL_SEEDS, "Frozen seeds changed")
    require(optimization["max_optimizer_steps"] == MAX_OPTIMIZER_STEPS, "Step budget changed")
    require(
        optimization["batch_size_per_data_parallel_rank"] == BATCH_PER_DP_RANK,
        "Per-rank batch changed",
    )
    require(checkpoint["save_every_steps"] == SAVE_EVERY_STEPS, "Save interval changed")
    require(tuple(checkpoint["saved_steps"]) == SAVED_STEPS, "Saved steps changed")
    require(
        tuple(checkpoint["required_manifest_bindings"]) == REQUIRED_CHECKPOINT_BINDINGS,
        "Required checkpoint bindings changed",
    )
    require(
        protocol["frozen_data_contract"]["split_sha256"] == SPLIT_SHA256,
        "Frozen split SHA256 changed",
    )
    return protocol


def validate_execution_supplement_proposal(workspace: Path) -> dict[str, Any]:
    path = workspace / PROPOSED_EXECUTION_SUPPLEMENT_PATH
    require(
        file_sha256(path) == PROPOSED_EXECUTION_SUPPLEMENT_SHA256,
        "Approved execution supplement proposal SHA256 changed",
    )
    proposal = read_json(path)
    require(
        proposal.get("status") == "PROPOSED_UNAPPROVED_NOT_FORMAL_EVIDENCE",
        "Execution supplement proposal must remain explicitly unapproved",
    )
    require(proposal.get("formal_queue_allowed") is False, "Proposal cannot allow a formal queue")
    require(proposal.get("p2_acceptance_allowed") is False, "Proposal cannot allow P2 acceptance")
    parent = proposal.get("parent_protocol", {})
    require(parent.get("file_sha256") == PROTOCOL_SHA256, "Proposal protocol SHA256 changed")
    require(parent.get("mutation_allowed") is False, "Proposal must not mutate the parent protocol")
    require(
        tuple(proposal.get("frozen_seed_candidates", [])) == FORMAL_SEEDS,
        "Proposal seed candidates changed",
    )
    scope = proposal.get("proposed_minimum_claim_centered_scope", {})
    rules = scope.get("learned_training_checkpoint_rules", [])
    require(isinstance(rules, list) and len(rules) == 8, "Proposal experiment rule count changed")
    candidate_count = sum(int(rule.get("candidate_new_checkpoint_count", 0)) for rule in rules)
    require(candidate_count == 48, "Proposal learned checkpoint count no longer sums to 48")
    require(
        scope.get("candidate_unique_learned_checkpoint_count") == candidate_count,
        "Proposal checkpoint total is inconsistent",
    )
    decision_ids = [
        item.get("decision_id") for item in proposal.get("blocking_open_decisions", [])
    ]
    require(
        decision_ids
        == [
            "P2-SCOPE-01",
            "P2-NONLEARNED-01",
            "P2-REP-01",
            "P2-RET-01",
            "P2-VARIANTS-01",
            "P2-EVAL-01",
        ],
        "Proposal blocking-decision registry changed",
    )
    require(
        proposal.get("next_state_transition", {}).get("required_user_approval") is True,
        "Proposal must require user approval before freezing",
    )
    registry = build_candidate_registry()
    require(registry.get("status") == proposal.get("status"), "Proposal/registry status mismatch")
    require(registry.get("formal_queue_allowed") is False, "Candidate registry cannot allow queueing")
    require(registry.get("p2_acceptance_allowed") is False, "Candidate registry cannot pass P2")
    require(
        registry.get("counts", {}).get("learned_training_checkpoint") == candidate_count,
        "Proposal/registry learned checkpoint count mismatch",
    )
    return {
        "path": PROPOSED_EXECUTION_SUPPLEMENT_PATH.as_posix(),
        "file_sha256": file_sha256(path),
        "status": proposal["status"],
        "formal_queue_allowed": False,
        "p2_acceptance_allowed": False,
        "candidate_unique_learned_checkpoint_count": candidate_count,
        "candidate_registry_cell_count": registry["cell_count"],
        "candidate_registry_counts": registry["counts"],
        "candidate_registry_sha256": registry["registry_sha256"],
        "blocking_decision_ids": decision_ids,
    }


def validate_frozen_execution_supplement(workspace: Path) -> dict[str, Any]:
    """Validate the approved P2 execution semantics and its immutable lock."""

    proposal = validate_execution_supplement_proposal(workspace)
    supplement_path = workspace / FROZEN_EXECUTION_SUPPLEMENT_PATH
    lock_path = workspace / FROZEN_EXECUTION_SUPPLEMENT_LOCK_PATH
    require(
        file_sha256(supplement_path) == FROZEN_EXECUTION_SUPPLEMENT_SHA256,
        "Frozen execution supplement SHA256 changed",
    )
    require(
        file_sha256(lock_path) == FROZEN_EXECUTION_SUPPLEMENT_LOCK_SHA256,
        "Frozen execution supplement lock SHA256 changed",
    )
    supplement = read_json(supplement_path)
    lock = read_json(lock_path)
    require(
        supplement.get("schema_version")
        == "human2robot-m5b-p2-execution-supplement-v1",
        "Frozen execution supplement schema changed",
    )
    require(
        supplement.get("supplement_id") == "m5b_p2_claim_centered_execution_v1",
        "Frozen execution supplement ID changed",
    )
    require(
        supplement.get("status") == "frozen_approved_execution_spec",
        "Execution supplement is not frozen and approved",
    )
    require(tuple(supplement.get("frozen_seeds", [])) == FORMAL_SEEDS, "Supplement seeds changed")
    parent = supplement.get("parent_protocol", {})
    require(parent.get("file_sha256") == PROTOCOL_SHA256, "Supplement parent hash changed")
    require(parent.get("mutation_allowed") is False, "Supplement may not mutate the parent")
    approved_proposal = supplement.get("approved_proposal", {})
    require(
        approved_proposal.get("file_sha256") == proposal["file_sha256"],
        "Frozen supplement no longer binds the approved proposal",
    )
    artifact_taxonomy = supplement.get("artifact_taxonomy", {})
    require(
        artifact_taxonomy.get("nonlearned_method_artifact", {}).get(
            "optimizer_checkpoint"
        )
        == "not_applicable_by_frozen_nonlearned_definition",
        "retrieval_only nonlearned artifact rule changed",
    )
    registry_contract = supplement.get("frozen_registry_contract", {})
    require(
        registry_contract.get("generator_code_sha256")
        == CANDIDATE_REGISTRY_GENERATOR_SHA256,
        "Supplement candidate generator hash changed",
    )
    require(
        {
            "learned_training_checkpoint": registry_contract.get(
                "learned_training_checkpoint_count"
            ),
            "nonlearned_method_artifact": registry_contract.get(
                "nonlearned_method_artifact_count"
            ),
            "checkpoint_linked_evaluation": registry_contract.get(
                "checkpoint_linked_evaluation_count"
            ),
            "aggregate_report": registry_contract.get("aggregate_report_count"),
        }
        == FROZEN_CELL_COUNTS,
        "Supplement artifact counts changed",
    )
    require(
        registry_contract.get("total_cell_count") == FROZEN_CELL_COUNT,
        "Supplement total cell count changed",
    )
    current = supplement.get("current_state", {})
    require(current.get("formal_queue_allowed") is False, "Frozen spec cannot itself open queue")
    require(current.get("p2_status") == "pending", "Frozen spec must not claim P2 passed")
    require(supplement.get("formal_launch_preconditions"), "Launch preconditions are missing")

    require(lock.get("status") == "locked", "Execution supplement lock is not locked")
    require(
        lock.get("supplement_file_sha256") == FROZEN_EXECUTION_SUPPLEMENT_SHA256,
        "Execution supplement lock binding changed",
    )
    require(
        lock.get("parent_protocol_file_sha256") == PROTOCOL_SHA256,
        "Execution supplement lock parent binding changed",
    )
    require(
        lock.get("approved_proposal_file_sha256")
        == PROPOSED_EXECUTION_SUPPLEMENT_SHA256,
        "Execution supplement lock proposal binding changed",
    )
    require(
        lock.get("candidate_registry_generator_file_sha256")
        == CANDIDATE_REGISTRY_GENERATOR_SHA256,
        "Execution supplement lock generator binding changed",
    )
    require(lock.get("contains_experiment_results") is False, "Supplement lock contains results")
    require(lock.get("passes_p2") is False, "Supplement lock may not pass P2")
    return {
        "path": FROZEN_EXECUTION_SUPPLEMENT_PATH.as_posix(),
        "file_sha256": FROZEN_EXECUTION_SUPPLEMENT_SHA256,
        "lock_path": FROZEN_EXECUTION_SUPPLEMENT_LOCK_PATH.as_posix(),
        "lock_file_sha256": FROZEN_EXECUTION_SUPPLEMENT_LOCK_SHA256,
        "status": supplement["status"],
        "supplement_id": supplement["supplement_id"],
        "formal_queue_allowed": False,
        "p2_acceptance_allowed": False,
        "artifact_counts": dict(FROZEN_CELL_COUNTS),
        "cell_count": FROZEN_CELL_COUNT,
        "approved_proposal": proposal,
    }


def validate_frozen_cell_registry(workspace: Path) -> dict[str, Any]:
    """Validate all 202 pending cells and their lock without treating them as results."""

    registry_path = workspace / FROZEN_CELL_REGISTRY_PATH
    lock_path = workspace / FROZEN_CELL_REGISTRY_LOCK_PATH
    generator_path = workspace / CANDIDATE_REGISTRY_GENERATOR_PATH
    materializer_path = workspace / FROZEN_REGISTRY_MATERIALIZER_PATH
    require(
        file_sha256(registry_path) == FROZEN_CELL_REGISTRY_SHA256,
        "Frozen cell registry SHA256 changed",
    )
    require(
        file_sha256(lock_path) == FROZEN_CELL_REGISTRY_LOCK_SHA256,
        "Frozen cell registry lock SHA256 changed",
    )
    require(
        file_sha256(generator_path) == CANDIDATE_REGISTRY_GENERATOR_SHA256,
        "Candidate registry generator SHA256 changed",
    )
    require(
        file_sha256(materializer_path) == FROZEN_REGISTRY_MATERIALIZER_SHA256,
        "Frozen registry materializer SHA256 changed",
    )
    registry = read_json(registry_path)
    lock = read_json(lock_path)
    require(
        registry.get("schema_version") == "human2robot-m5b-p2-cell-registry-v1",
        "Frozen cell registry schema changed",
    )
    require(
        registry.get("registry_id") == "m5b_p2_claim_centered_202_cells_v1",
        "Frozen cell registry ID changed",
    )
    require(registry.get("status") == "frozen_pending_execution", "Registry is not pending")
    require(registry.get("formal_queue_allowed") is False, "Pending registry cannot open queue")
    require(registry.get("p2_acceptance_allowed") is False, "Pending registry cannot pass P2")
    require(tuple(registry.get("seeds", [])) == FORMAL_SEEDS, "Registry seeds changed")
    require(
        tuple(registry.get("required_experiment_ids", [])) == REQUIRED_EXPERIMENT_IDS,
        "Registry experiment IDs changed",
    )
    require(registry.get("counts") == FROZEN_CELL_COUNTS, "Registry counts changed")
    require(registry.get("cell_count") == FROZEN_CELL_COUNT, "Registry cell count changed")
    require(
        registry.get("supplement_file_sha256") == FROZEN_EXECUTION_SUPPLEMENT_SHA256,
        "Registry supplement binding changed",
    )
    require(
        registry.get("candidate_generator_file_sha256")
        == CANDIDATE_REGISTRY_GENERATOR_SHA256,
        "Registry generator binding changed",
    )
    cells = registry.get("cells", [])
    require(isinstance(cells, list) and len(cells) == FROZEN_CELL_COUNT, "Registry cells missing")
    cell_ids = [cell.get("cell_id") for cell in cells if isinstance(cell, dict)]
    require(len(cell_ids) == FROZEN_CELL_COUNT, "Registry has non-object cells")
    require(len(set(cell_ids)) == FROZEN_CELL_COUNT, "Registry cell IDs are not unique")
    require(
        all(cell.get("status") == "pending" for cell in cells),
        "Frozen registry must contain only pending cells",
    )
    require(
        all(cell.get("formal_result") is False for cell in cells),
        "Frozen registry must contain no formal results",
    )
    actual_counts: dict[str, int] = {}
    for cell in cells:
        kind = cell.get("artifact_kind")
        actual_counts[kind] = actual_counts.get(kind, 0) + 1
    require(actual_counts == FROZEN_CELL_COUNTS, "Registry cell artifact counts changed")
    require(
        canonical_json_sha256(cells) == registry.get("cells_payload_sha256"),
        "Registry cells payload SHA256 changed",
    )
    known_ids = set(cell_ids)
    dangling = sorted(
        {
            parent
            for cell in cells
            for parent in cell.get("parent_artifact_ids", [])
            if parent not in known_ids
        }
    )
    require(not dangling, f"Registry contains dangling parent artifact IDs: {dangling[:3]}")

    require(lock.get("status") == "locked_pending_execution", "Registry lock status changed")
    require(
        lock.get("registry_file_sha256") == FROZEN_CELL_REGISTRY_SHA256,
        "Registry lock payload binding changed",
    )
    require(
        lock.get("registry_materializer_file_sha256")
        == FROZEN_REGISTRY_MATERIALIZER_SHA256,
        "Registry lock materializer binding changed",
    )
    require(
        lock.get("candidate_registry_generator_file_sha256")
        == CANDIDATE_REGISTRY_GENERATOR_SHA256,
        "Registry lock generator binding changed",
    )
    require(
        lock.get("execution_supplement_file_sha256")
        == FROZEN_EXECUTION_SUPPLEMENT_SHA256,
        "Registry lock supplement binding changed",
    )
    require(
        lock.get("execution_supplement_lock_file_sha256")
        == FROZEN_EXECUTION_SUPPLEMENT_LOCK_SHA256,
        "Registry lock supplement-lock binding changed",
    )
    require(lock.get("counts") == FROZEN_CELL_COUNTS, "Registry lock counts changed")
    require(lock.get("cell_count") == FROZEN_CELL_COUNT, "Registry lock cell count changed")
    require(lock.get("formal_queue_allowed") is False, "Registry lock cannot open queue")
    require(lock.get("contains_experiment_results") is False, "Registry lock contains results")
    require(lock.get("passes_p2") is False, "Registry lock may not pass P2")
    return {
        "path": FROZEN_CELL_REGISTRY_PATH.as_posix(),
        "file_sha256": FROZEN_CELL_REGISTRY_SHA256,
        "lock_path": FROZEN_CELL_REGISTRY_LOCK_PATH.as_posix(),
        "lock_file_sha256": FROZEN_CELL_REGISTRY_LOCK_SHA256,
        "status": registry["status"],
        "formal_queue_allowed": False,
        "p2_acceptance_allowed": False,
        "counts": dict(FROZEN_CELL_COUNTS),
        "cell_count": FROZEN_CELL_COUNT,
        "cells_payload_sha256": registry["cells_payload_sha256"],
    }


def validate_prerequisites(workspace: Path) -> dict[str, Any]:
    p0_path = workspace / "data/Human2Robot/derived/m5b_v03/p0_implementation_report.json"
    p1_path = workspace / "data/Human2Robot/derived/m5b_v03/p1_data_acceptance_report.json"
    p0 = read_json(p0_path)
    p1 = read_json(p1_path)
    require(p0.get("status") == "passed", "M5B-P0 is not passed")
    require(p1.get("status") == "passed", "M5B-P1 is not passed")
    require(
        p0.get("protocol_validation", {}).get("protocol_file_sha256") == PROTOCOL_SHA256,
        "P0 protocol binding changed",
    )
    require(p1.get("protocol_file_sha256") == PROTOCOL_SHA256, "P1 protocol binding changed")
    require(p0.get("formal_configs", {}).get("config_count") == 9, "P0 formal config count changed")
    require(p1.get("selection_id") == P1_SELECTION_ID, "P1 selection ID changed")
    counts = p1.get("validation", {}).get("per_task_independent_source_episode_count", {})
    require(len(counts) == 4 and set(counts.values()) == {10}, "P1 must retain 10 demos/task")
    require(
        p1.get("leakage_audit", {}).get("heldout_robot_dataset_read_count") == 0,
        "P1 leakage audit failed",
    )
    weight_bindings = p0.get("local_weight_bindings", {})
    initialization = weight_bindings.get("initialization_checkpoint", {})
    tokenizer = weight_bindings.get("tokenizer", {})
    require(
        initialization.get("path") == str(INITIALIZATION_CHECKPOINT_PATH)
        and initialization.get("file_sha256") == INITIALIZATION_CHECKPOINT_SHA256,
        "P0 initialization checkpoint binding changed",
    )
    require(
        tokenizer.get("path") == str(TOKENIZER_CHECKPOINT_PATH)
        and tokenizer.get("file_sha256") == TOKENIZER_CHECKPOINT_SHA256,
        "P0 tokenizer binding changed",
    )
    require(INITIALIZATION_CHECKPOINT_PATH.is_file(), "Local initialization checkpoint is missing")
    require(TOKENIZER_CHECKPOINT_PATH.is_file(), "Local tokenizer checkpoint is missing")
    require(
        file_sha256(INITIALIZATION_CHECKPOINT_PATH) == INITIALIZATION_CHECKPOINT_SHA256,
        "Current initialization checkpoint SHA256 mismatch",
    )
    require(
        file_sha256(TOKENIZER_CHECKPOINT_PATH) == TOKENIZER_CHECKPOINT_SHA256,
        "Current tokenizer checkpoint SHA256 mismatch",
    )
    return {
        "p0_report_path": str(p0_path.relative_to(workspace)),
        "p0_report_sha256": file_sha256(p0_path),
        "p1_report_path": str(p1_path.relative_to(workspace)),
        "p1_report_sha256": file_sha256(p1_path),
        "p1_selection_id": P1_SELECTION_ID,
        "initialization_checkpoint_path": str(INITIALIZATION_CHECKPOINT_PATH),
        "initialization_checkpoint_sha256": INITIALIZATION_CHECKPOINT_SHA256,
        "tokenizer_checkpoint_path": str(TOKENIZER_CHECKPOINT_PATH),
        "tokenizer_checkpoint_sha256": TOKENIZER_CHECKPOINT_SHA256,
        "downloads_performed": False,
    }


def source_paths(workspace: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "cosmos_policy", "tools", "pyproject.toml", "uv.lock"],
        cwd=workspace,
        check=True,
        stdout=subprocess.PIPE,
    )
    relative_paths = [Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]
    selected = [
        path
        for path in relative_paths
        if path.suffix in {".py", ".sh", ".toml", ".lock"}
        and "__pycache__" not in path.parts
    ]
    require(bool(selected), "No tracked source paths found for P2 snapshot")
    for required_path in (
        Path("tools/human2robot_m5b_p2.py"),
        CANDIDATE_REGISTRY_GENERATOR_PATH,
        FROZEN_REGISTRY_MATERIALIZER_PATH,
        Path("cosmos_policy/scripts/train.py"),
    ):
        require((workspace / required_path).is_file(), f"Required P2 source is missing: {required_path}")
        if required_path not in selected:
            selected.append(required_path)
    return sorted(selected, key=lambda path: path.as_posix())


def source_manifest(workspace: Path, paths: Iterable[Path]) -> dict[str, Any]:
    files = []
    for relative in paths:
        absolute = workspace / relative
        require(absolute.is_file(), f"Tracked source missing: {relative}")
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": file_sha256(absolute),
                "size_bytes": absolute.stat().st_size,
            }
        )
    return {
        "schema_version": "human2robot-m5b-p2-source-snapshot-v1",
        "files": files,
        "code_sha256": canonical_json_sha256(files),
    }


def materialize_source_snapshot(
    workspace: Path, snapshot_root: Path, manifest: dict[str, Any]
) -> Path:
    code_sha256 = manifest["code_sha256"]
    snapshot_path = snapshot_root / code_sha256
    manifest_path = snapshot_path / "source_snapshot_manifest.json"
    if snapshot_path.exists():
        existing = read_json(manifest_path)
        require(existing.get("code_sha256") == code_sha256, "Existing snapshot binding mismatch")
        return snapshot_path
    for item in manifest["files"]:
        relative = Path(item["path"])
        destination = snapshot_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(workspace / relative, destination)
    snapshot_manifest = dict(manifest)
    snapshot_manifest["created_at_utc"] = utc_now()
    write_json_atomic(manifest_path, snapshot_manifest)
    return snapshot_path


def expected_run_directory(output_root: Path, cell: MainTrainingCell) -> Path:
    return output_root / "cosmos_policy" / "human2robot_m5b_formal" / cell.config_name


def checkpoint_directory(run_directory: Path, step: int) -> Path:
    return run_directory / "checkpoints" / f"iter_{step:09d}"


def validate_dcp_checkpoint(path: Path, expected_world_size: int) -> dict[str, Any]:
    require(path.is_dir(), f"Checkpoint directory missing: {path}")
    components: dict[str, Any] = {}
    for component in ("model", "optim", "scheduler", "trainer"):
        component_path = path / component
        require(component_path.is_dir(), f"Checkpoint component missing: {component_path}")
        rank_files = sorted(component_path.glob("__*_0.distcp"))
        require(
            len(rank_files) == expected_world_size,
            f"{component_path} has {len(rank_files)} rank files, expected {expected_world_size}",
        )
        require(all(item.stat().st_size > 0 for item in rank_files), f"Empty DCP shard in {component_path}")
        metadata_path = component_path / ".metadata"
        require(metadata_path.is_file() and metadata_path.stat().st_size > 0, f"Missing DCP metadata: {metadata_path}")
        components[component] = {
            "rank_file_count": len(rank_files),
            "metadata_size_bytes": metadata_path.stat().st_size,
            "payload_size_bytes": sum(item.stat().st_size for item in rank_files),
        }
    return {"path": str(path), "components": components}


def checkpoint_payload_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file() and item.name != "m5b_p2_checkpoint_manifest.json"
        ),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    require(bool(files), f"Checkpoint payload is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def base_bindings(cell: MainTrainingCell, code_sha256: str) -> dict[str, Any]:
    return {
        "protocol_file_sha256": PROTOCOL_SHA256,
        "code_sha256": code_sha256,
        "resolved_initialization_checkpoint_sha256": INITIALIZATION_CHECKPOINT_SHA256,
        "canonical_schema": "human2robot-canonical-hdf5-v3",
        "split_sha256": SPLIT_SHA256,
        "time_view_id": "nominal_camera_30hz_segmented",
        "pool_action_view_id": "human_hand_robot_frame_raw",
        "query_action_view_id": "robot_ee_observed_t_plus_1_bc_proxy",
        "action_alignment_id": "train_only_tplus1_query_anchor_se3_identity_scale_v1",
        "view_id": "e73604700bb079ff2397fe951c9c60915060f8aafb80d170753b539c739ad8a5",
        "retrieval_index_sha256": "362bc28cfb96174208a37e63366b6c157e08e0de8e124066ca9d6057ec36c305",
        "method_id": cell.method_id,
        "experiment_id": cell.experiment_id,
        "seed": cell.seed,
        "optimizer_steps": MAX_OPTIMIZER_STEPS,
        "batch_size_per_data_parallel_rank": BATCH_PER_DP_RANK,
        "data_parallel_world_size": FIXED_DP_WORLD_SIZE,
        "H_steps": H_STEPS,
        "K_steps": K_STEPS,
    }


def validate_binding_keys(bindings: dict[str, Any]) -> None:
    require(tuple(bindings.keys()) == REQUIRED_CHECKPOINT_BINDINGS, "Checkpoint binding keys/order changed")


def initial_cell_record(cell: MainTrainingCell, output_root: Path, code_sha256: str) -> dict[str, Any]:
    bindings = base_bindings(cell, code_sha256)
    validate_binding_keys(bindings)
    run_directory = expected_run_directory(output_root, cell)
    return {
        **asdict(cell),
        "status": "pending",
        "formal_result": False,
        "attempt_count": 0,
        "run_directory": str(run_directory),
        "runtime_binding_path": str(run_directory / "m5b_p2_runtime_binding.json"),
        "cell_manifest_path": str(run_directory / "m5b_p2_cell_manifest.json"),
        "log_path": str(output_root / "orchestrator_logs" / f"{cell.cell_id}.log"),
        "bindings": bindings,
        "saved_steps_expected": list(SAVED_STEPS),
    }


def build_master_manifest(
    workspace: Path,
    output_root: Path,
    snapshot_path: Path,
    source: dict[str, Any],
    protocol: dict[str, Any],
    prerequisites: dict[str, Any],
    execution_supplement: dict[str, Any],
    frozen_cell_registry: dict[str, Any],
) -> dict[str, Any]:
    cells = [initial_cell_record(cell, output_root, source["code_sha256"]) for cell in main_training_cells()]
    coverage = protocol_experiment_coverage(
        protocol,
        execution_spec_frozen=True,
        full_cell_registry_bound=True,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_id": GATE_ID,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "status": "pending",
        "formal_result": False,
        "workspace": str(workspace),
        "output_root": str(output_root),
        "fixed_runtime": {
            "world_size": FIXED_WORLD_SIZE,
            "data_parallel_world_size": FIXED_DP_WORLD_SIZE,
            "batch_size_per_data_parallel_rank": BATCH_PER_DP_RANK,
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "save_every_steps": SAVE_EVERY_STEPS,
            "saved_steps": list(SAVED_STEPS),
            "offline_auto_download_disabled": True,
        },
        "source_snapshot": {
            "path": str(snapshot_path),
            "manifest_path": str(snapshot_path / "source_snapshot_manifest.json"),
            "code_sha256": source["code_sha256"],
            "file_count": len(source["files"]),
        },
        "prerequisites": prerequisites,
        "execution_supplement": execution_supplement,
        "frozen_cell_registry": frozen_cell_registry,
        "protocol_experiment_coverage": coverage,
        "implemented_main_training_cells": cells,
        "acceptance": {
            "implemented_cells_complete": False,
            "all_protocol_experiment_families_implemented": coverage[
                "full_protocol_matrix_implemented"
            ],
            "full_execution_spec_frozen": coverage["full_execution_spec_frozen"],
            "full_cell_registry_bound": coverage["full_cell_registry_bound"],
            "p2_gate_passed": False,
        },
        "claim_boundary": {
            "m5b_p2_run_completeness": "pending",
            "m5_v03": "pending",
            "gate_c": "pending",
            "m6_rollout_approved": False,
            "reason": (
                "The supplement and 202-cell registry are frozen, but required handlers "
                "beyond M5B-MAIN-01 are not yet executable."
            ),
        },
        "launch_safety": {
            "full_p2_queue_available": False,
            "implemented_main_subset_requires_explicit_acknowledgement": True,
            "concurrent_fixed_gpu_jobs_forbidden": True,
        },
    }


def update_master_acceptance(master: dict[str, Any]) -> None:
    cells = master["implemented_main_training_cells"]
    implemented_complete = bool(cells) and all(cell.get("status") == "completed" for cell in cells)
    all_families = bool(
        master["protocol_experiment_coverage"].get("full_protocol_matrix_implemented")
    )
    execution_spec_frozen = bool(
        master["protocol_experiment_coverage"].get("full_execution_spec_frozen")
    )
    full_cell_registry_bound = bool(
        master["protocol_experiment_coverage"].get("full_cell_registry_bound")
    )
    p2_passed = (
        implemented_complete
        and all_families
        and execution_spec_frozen
        and full_cell_registry_bound
    )
    master["acceptance"] = {
        "implemented_cells_complete": implemented_complete,
        "all_protocol_experiment_families_implemented": all_families,
        "full_execution_spec_frozen": execution_spec_frozen,
        "full_cell_registry_bound": full_cell_registry_bound,
        "p2_gate_passed": p2_passed,
    }
    if p2_passed:
        master["status"] = "passed"
    elif any(cell.get("status") == "running" for cell in cells):
        master["status"] = "running"
    elif any(cell.get("status") in {"failed", "invalid"} for cell in cells):
        master["status"] = "failed"
    else:
        master["status"] = "pending"
    master["formal_result"] = p2_passed
    master["claim_boundary"]["m5b_p2_run_completeness"] = "passed" if p2_passed else "pending"
    if implemented_complete and (
        not all_families or not execution_spec_frozen or not full_cell_registry_bound
    ):
        master["claim_boundary"]["reason"] = (
            "All nine M5B-MAIN-01 learned cells completed, but the remaining frozen "
            "registry handlers and experiment families are not complete; full P2 is still "
            "pending."
        )
    master["updated_at_utc"] = utc_now()


def prepare(workspace: Path, output_root: Path) -> dict[str, Any]:
    require_full_docker_environment()
    protocol = validate_protocol(workspace)
    execution_supplement = validate_frozen_execution_supplement(workspace)
    frozen_cell_registry = validate_frozen_cell_registry(workspace)
    prerequisites = validate_prerequisites(workspace)
    paths = source_paths(workspace)
    source = source_manifest(workspace, paths)
    snapshot_path = materialize_source_snapshot(workspace, output_root / "source_snapshots", source)
    manifest_path = workspace / "data/Human2Robot/derived/m5b_v03/run_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        require(
            existing.get("source_snapshot", {}).get("code_sha256") == source["code_sha256"],
            "An existing P2 manifest is bound to different code; do not mix formal cells",
        )
        return existing
    master = build_master_manifest(
        workspace,
        output_root,
        snapshot_path,
        source,
        protocol,
        prerequisites,
        execution_supplement,
        frozen_cell_registry,
    )
    write_json_atomic(manifest_path, master)
    return master


def verify_runtime_binding(path: Path, cell: MainTrainingCell, code_sha256: str) -> dict[str, Any]:
    binding = read_json(path)
    require(binding.get("cell_id") == cell.cell_id, "Runtime cell ID mismatch")
    require(binding.get("experiment_id") == cell.experiment_id, "Runtime experiment ID mismatch")
    require(binding.get("method_id") == cell.method_id, "Runtime method ID mismatch")
    require(binding.get("protocol_file_sha256") == PROTOCOL_SHA256, "Runtime protocol hash mismatch")
    require(binding.get("code_sha256") == code_sha256, "Runtime code hash mismatch")
    actual = binding.get("actual", {})
    require(actual.get("world_size") == FIXED_WORLD_SIZE, "Runtime global world size mismatch")
    require(actual.get("data_parallel_world_size") == FIXED_DP_WORLD_SIZE, "Runtime DP world size mismatch")
    require(actual.get("seed") == cell.seed, "Runtime seed mismatch")
    require(actual.get("max_optimizer_steps") == MAX_OPTIMIZER_STEPS, "Runtime step budget mismatch")
    require(
        actual.get("batch_size_per_data_parallel_rank") == BATCH_PER_DP_RANK,
        "Runtime batch mismatch",
    )
    require(actual.get("checkpoint_save_every_steps") == SAVE_EVERY_STEPS, "Runtime save interval mismatch")
    require(actual.get("gradient_accumulation_steps") == 1, "Runtime gradient accumulation mismatch")
    require(actual.get("sampler_seed") == cell.seed, "Runtime sampler seed mismatch")
    require(actual.get("H_steps") == H_STEPS and actual.get("K_steps") == K_STEPS, "Runtime H/K mismatch")
    require(
        actual.get("initialization_checkpoint_path") == str(INITIALIZATION_CHECKPOINT_PATH),
        "Runtime initialization checkpoint path mismatch",
    )
    require(
        actual.get("tokenizer_checkpoint_path") == str(TOKENIZER_CHECKPOINT_PATH),
        "Runtime tokenizer checkpoint path mismatch",
    )
    require(
        binding.get("optimization")
        == {
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "weight_decay": 0.1,
            "betas": [0.9, 0.999],
            "load_training_state": False,
            "load_ema_to_reg": True,
        },
        "Runtime optimizer/checkpoint loading contract mismatch",
    )
    require(
        binding.get("environment", {}).get("offline_auto_download_disabled") is True,
        "Offline/no-auto-download runtime binding is false",
    )
    require(
        binding.get("environment", {}).get("huggingface_offline") is True
        and binding.get("environment", {}).get("transformers_offline") is True,
        "Hugging Face/Transformers offline runtime binding is false",
    )
    require(
        binding.get("environment", {}).get("wandb_disabled") is True,
        "W&B runtime binding is not disabled",
    )
    return binding


def audit_completed_cell(
    record: dict[str, Any], cell: MainTrainingCell, code_sha256: str
) -> dict[str, Any]:
    run_directory = Path(record["run_directory"])
    runtime = verify_runtime_binding(Path(record["runtime_binding_path"]), cell, code_sha256)
    latest_path = run_directory / "checkpoints/latest_checkpoint.txt"
    require(latest_path.is_file(), f"Latest-checkpoint marker missing: {latest_path}")
    require(latest_path.read_text(encoding="utf-8").strip() == "iter_000007000", "Latest checkpoint is not step 7000")
    saved = [
        validate_dcp_checkpoint(checkpoint_directory(run_directory, step), FIXED_WORLD_SIZE)
        for step in SAVED_STEPS
    ]
    primary_path = checkpoint_directory(run_directory, MAX_OPTIMIZER_STEPS)
    payload_hash = checkpoint_payload_sha256(primary_path)
    resolved_config_path = run_directory / "config.yaml"
    require(resolved_config_path.is_file(), f"Resolved config missing: {resolved_config_path}")
    bindings = base_bindings(cell, code_sha256)
    validate_binding_keys(bindings)
    checkpoint_manifest = {
        "schema_version": CELL_SCHEMA_VERSION,
        "status": "completed",
        "formal_result": True,
        "completed_at_utc": utc_now(),
        "cell": asdict(cell),
        "bindings": bindings,
        "runtime_binding_path": record["runtime_binding_path"],
        "runtime_binding_sha256": file_sha256(Path(record["runtime_binding_path"])),
        "resolved_config_path": str(resolved_config_path),
        "resolved_config_sha256": file_sha256(resolved_config_path),
        "saved_checkpoints": saved,
        "primary_checkpoint_path": str(primary_path),
        "primary_checkpoint_payload_sha256": payload_hash,
        "no_imputation": True,
    }
    write_json_atomic(Path(record["cell_manifest_path"]), checkpoint_manifest)
    primary_sidecar = primary_path / "m5b_p2_checkpoint_manifest.json"
    write_json_atomic(primary_sidecar, checkpoint_manifest)
    return {
        "status": "completed",
        "formal_result": True,
        "completed_at_utc": checkpoint_manifest["completed_at_utc"],
        "runtime_binding_sha256": checkpoint_manifest["runtime_binding_sha256"],
        "resolved_config_sha256": checkpoint_manifest["resolved_config_sha256"],
        "primary_checkpoint_payload_sha256": payload_hash,
        "saved_steps_validated": list(SAVED_STEPS),
        "runtime": runtime["actual"],
    }


def training_command(
    workspace: Path,
    snapshot_path: Path,
    output_root: Path,
    record: dict[str, Any],
    cell: MainTrainingCell,
) -> tuple[list[str], dict[str, str]]:
    torchrun = workspace / ".venv/bin/torchrun"
    require(torchrun.is_file(), f"Container torchrun missing: {torchrun}")
    config_path = snapshot_path / "cosmos_policy/config/config.py"
    require(config_path.is_file(), f"Snapshot config missing: {config_path}")
    env = os.environ.copy()
    env.update(
        {
            "COSMOS_SKIP_HF_AUTO_DOWNLOAD": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "COSMOS_PREDICT2P5_POSTTRAINED_CKPT": str(
                INITIALIZATION_CHECKPOINT_PATH
            ),
            "COSMOS_PREDICT2P5_TOKENIZER_CKPT": str(TOKENIZER_CHECKPOINT_PATH),
            "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(FIXED_WORLD_SIZE)),
            "IMAGINAIRE_OUTPUT_ROOT": str(output_root),
            "RECAP_WORKSPACE": str(workspace),
            "HUMAN2ROBOT_ROOT": str(workspace / "data/Human2Robot"),
            "PYTHONPATH": str(snapshot_path),
            "HUMAN2ROBOT_P2_RUNTIME_BINDING_PATH": record["runtime_binding_path"],
            "HUMAN2ROBOT_P2_PROTOCOL_SHA256": PROTOCOL_SHA256,
            "HUMAN2ROBOT_P2_CODE_SHA256": record["bindings"]["code_sha256"],
            "HUMAN2ROBOT_P2_CELL_ID": cell.cell_id,
            "HUMAN2ROBOT_P2_EXPERIMENT_ID": cell.experiment_id,
            "HUMAN2ROBOT_P2_METHOD_ID": cell.method_id,
            "HUMAN2ROBOT_P2_EXPECTED_WORLD_SIZE": str(FIXED_WORLD_SIZE),
            "HUMAN2ROBOT_P2_EXPECTED_DP_WORLD_SIZE": str(FIXED_DP_WORLD_SIZE),
            "HUMAN2ROBOT_P2_EXPECTED_SEED": str(cell.seed),
            "HUMAN2ROBOT_P2_EXPECTED_MAX_ITER": str(MAX_OPTIMIZER_STEPS),
            "HUMAN2ROBOT_P2_EXPECTED_BATCH_PER_DP_RANK": str(BATCH_PER_DP_RANK),
            "HUMAN2ROBOT_P2_EXPECTED_SAVE_ITER": str(SAVE_EVERY_STEPS),
            "HUMAN2ROBOT_P2_EXPECTED_INIT_CKPT_PATH": str(
                INITIALIZATION_CHECKPOINT_PATH
            ),
            "HUMAN2ROBOT_P2_EXPECTED_TOKENIZER_PATH": str(
                TOKENIZER_CHECKPOINT_PATH
            ),
        }
    )
    command = [
        str(torchrun),
        f"--nproc_per_node={FIXED_WORLD_SIZE}",
        "--master_port=12430",
        "-m",
        "cosmos_policy.scripts.train",
        f"--config={config_path}",
        "--",
        f"experiment={cell.config_name}",
    ]
    return command, env


def load_master(workspace: Path) -> tuple[Path, dict[str, Any]]:
    path = workspace / "data/Human2Robot/derived/m5b_v03/run_manifest.json"
    master = read_json(path)
    require(master.get("schema_version") == SCHEMA_VERSION, "Wrong P2 manifest schema")
    return path, master


def find_cell_record(master: dict[str, Any], cell_id: str) -> dict[str, Any]:
    matches = [item for item in master["implemented_main_training_cells"] if item["cell_id"] == cell_id]
    require(len(matches) == 1, f"Unknown or duplicate cell: {cell_id}")
    return matches[0]


def _run_cell_unlocked(workspace: Path, cell_id: str) -> dict[str, Any]:
    manifest_path, master = load_master(workspace)
    cell_map = {cell.cell_id: cell for cell in main_training_cells()}
    require(cell_id in cell_map, f"Unsupported main training cell: {cell_id}")
    cell = cell_map[cell_id]
    record = find_cell_record(master, cell_id)
    code_sha256 = master["source_snapshot"]["code_sha256"]
    if record.get("status") == "completed":
        evidence = audit_completed_cell(record, cell, code_sha256)
        record.update(evidence)
        update_master_acceptance(master)
        write_json_atomic(manifest_path, master)
        return evidence

    record["status"] = "running"
    record["formal_result"] = False
    record["attempt_count"] = int(record.get("attempt_count", 0)) + 1
    record["attempt_started_at_utc"] = utc_now()
    record.pop("failure", None)
    update_master_acceptance(master)
    write_json_atomic(manifest_path, master)

    snapshot_path = Path(master["source_snapshot"]["path"])
    output_root = Path(master["output_root"])
    command, env = training_command(workspace, snapshot_path, output_root, record, cell)
    record["launch_command"] = command
    record["resume_policy"] = "resume only from this cell's latest valid DCP; never impute"
    write_json_atomic(manifest_path, master)
    log_path = Path(record["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.run(
                command,
                cwd=snapshot_path,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record["last_exit_code"] = process.returncode
        if process.returncode != 0:
            raise P2Error(f"Formal training process exited with code {process.returncode}")
        evidence = audit_completed_cell(record, cell, code_sha256)
        record.update(evidence)
    except BaseException as error:
        record["status"] = "failed"
        record["formal_result"] = False
        record["failure"] = {
            "recorded_at_utc": utc_now(),
            "type": type(error).__name__,
            "message": str(error),
            "no_imputation": True,
        }
        update_master_acceptance(master)
        write_json_atomic(manifest_path, master)
        raise
    update_master_acceptance(master)
    write_json_atomic(manifest_path, master)
    return record


def run_cell(
    workspace: Path,
    cell_id: str,
    acknowledge_partial_subset: bool = False,
) -> dict[str, Any]:
    require(
        acknowledge_partial_subset,
        "This is only an implemented M5B-MAIN-01 subset cell, not full P2; "
        "pass the explicit acknowledgement flag.",
    )
    require_full_docker_environment()
    lock_path = workspace / "data/Human2Robot/derived/m5b_v03/p2_execution.lock"
    with exclusive_execution_lock(lock_path, f"run-cell:{cell_id}"):
        return _run_cell_unlocked(workspace, cell_id)


def queue_implemented_main_subset(
    workspace: Path, acknowledge_partial_subset: bool = False
) -> dict[str, Any]:
    require(
        acknowledge_partial_subset,
        "The queue covers only nine learned M5B-MAIN-01 cells and cannot pass full P2; "
        "pass the explicit acknowledgement flag.",
    )
    require_full_docker_environment()
    lock_path = workspace / "data/Human2Robot/derived/m5b_v03/p2_execution.lock"
    with exclusive_execution_lock(lock_path, "queue-implemented-main-subset"):
        return _queue_implemented_main_subset_unlocked(workspace)


def _queue_implemented_main_subset_unlocked(workspace: Path) -> dict[str, Any]:
    manifest_path, master = load_master(workspace)
    master["queue"] = {
        "status": "running",
        "pid": os.getpid(),
        "started_at_utc": utc_now(),
        "policy": "sequential fixed-world-size execution; failed cells stop the queue",
    }
    write_json_atomic(manifest_path, master)
    try:
        for cell in main_training_cells():
            _run_cell_unlocked(workspace, cell.cell_id)
    except BaseException as error:
        _, master = load_master(workspace)
        master["queue"] = {
            **master.get("queue", {}),
            "status": "failed",
            "stopped_at_utc": utc_now(),
            "failure": {"type": type(error).__name__, "message": str(error)},
        }
        update_master_acceptance(master)
        write_json_atomic(manifest_path, master)
        raise
    _, master = load_master(workspace)
    master["queue"] = {
        **master.get("queue", {}),
        "status": "implemented_main_cells_completed",
        "stopped_at_utc": utc_now(),
    }
    update_master_acceptance(master)
    write_json_atomic(manifest_path, master)
    return master


def audit(workspace: Path) -> dict[str, Any]:
    require_full_docker_environment()
    manifest_path, master = load_master(workspace)
    code_sha256 = master["source_snapshot"]["code_sha256"]
    for cell in main_training_cells():
        record = find_cell_record(master, cell.cell_id)
        try:
            evidence = audit_completed_cell(record, cell, code_sha256)
        except P2Error as error:
            if record.get("status") == "completed":
                record["status"] = "invalid"
                record["formal_result"] = False
                record["audit_failure"] = str(error)
            continue
        record.update(evidence)
    update_master_acceptance(master)
    write_json_atomic(manifest_path, master)
    return master


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=FORMAL_OUTPUT_ROOT,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run = subparsers.add_parser("run-cell")
    run.add_argument("cell_id")
    run.add_argument("--acknowledge-partial-main-subset", action="store_true")
    queue = subparsers.add_parser("queue-implemented-main-subset")
    queue.add_argument("--acknowledge-partial-main-subset", action="store_true")
    subparsers.add_parser("audit")
    subparsers.add_parser("list-cells")
    subparsers.add_parser("show-frozen-scope")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.resolve()
    output_root = args.output_root.resolve()
    if args.command == "prepare":
        result = prepare(workspace, output_root)
    elif args.command == "run-cell":
        result = run_cell(
            workspace,
            args.cell_id,
            acknowledge_partial_subset=args.acknowledge_partial_main_subset,
        )
    elif args.command == "queue-implemented-main-subset":
        result = queue_implemented_main_subset(
            workspace,
            acknowledge_partial_subset=args.acknowledge_partial_main_subset,
        )
    elif args.command == "audit":
        result = audit(workspace)
    elif args.command == "show-frozen-scope":
        result = {
            "execution_supplement": validate_frozen_execution_supplement(workspace),
            "cell_registry": validate_frozen_cell_registry(workspace),
        }
    else:
        result = {"cells": [asdict(cell) for cell in main_training_cells()]}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except P2Error as error:
        print(f"M5B-P2 error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
