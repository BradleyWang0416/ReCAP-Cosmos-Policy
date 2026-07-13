#!/usr/bin/env python3
"""Generate the unapproved candidate M5B-P2 cell registry for review.

The generated registry is deliberately non-formal.  It cannot launch jobs or
pass P2.  Its only purpose is to make the proposed claim-centered execution
scope explicit before the execution supplement is approved and frozen.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

STATUS = "PROPOSED_UNAPPROVED_NOT_FORMAL_EVIDENCE"
FORMAL_SEEDS = (20260711, 20260712, 20260713)
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
LEARNED_METHODS = ("no_retrieval", "co_training", "recap_hand_ret")
POOL_SIZES = (0, 1, 2, 4, 8, 10)


@dataclass(frozen=True)
class CandidateCell:
    cell_id: str
    artifact_kind: str
    experiment_id: str
    variant_id: str
    method_id: str | None
    seed: int | None
    parent_artifact_ids: tuple[str, ...]
    optimizer_steps: int | None
    formal_result: bool = False
    status: str = STATUS


def _cell_id(
    artifact_kind: str,
    experiment_id: str,
    variant_id: str,
    method_id: str | None,
    seed: int | None,
) -> str:
    parts = [artifact_kind, experiment_id, variant_id]
    if method_id is not None:
        parts.append(method_id)
    if seed is not None:
        parts.append(f"seed{seed}")
    return "__".join(parts)


def _cell(
    artifact_kind: str,
    experiment_id: str,
    variant_id: str,
    method_id: str | None,
    seed: int | None,
    parents: tuple[str, ...] = (),
    optimizer_steps: int | None = None,
) -> CandidateCell:
    return CandidateCell(
        cell_id=_cell_id(artifact_kind, experiment_id, variant_id, method_id, seed),
        artifact_kind=artifact_kind,
        experiment_id=experiment_id,
        variant_id=variant_id,
        method_id=method_id,
        seed=seed,
        parent_artifact_ids=parents,
        optimizer_steps=optimizer_steps,
    )


def _train_id(experiment_id: str, variant_id: str, method_id: str, seed: int) -> str:
    return _cell_id(
        "learned_training_checkpoint",
        experiment_id,
        variant_id,
        method_id,
        seed,
    )


def _nonlearned_id(seed: int) -> str:
    return _cell_id(
        "nonlearned_method_artifact",
        "M5B-MAIN-01",
        "retrieval_only_projection",
        "retrieval_only",
        seed,
    )


def learned_training_cells() -> list[CandidateCell]:
    cells: list[CandidateCell] = []
    for method_id in LEARNED_METHODS:
        for seed in FORMAL_SEEDS:
            cells.append(
                _cell(
                    "learned_training_checkpoint",
                    "M5B-MAIN-01",
                    "frozen_main",
                    method_id,
                    seed,
                    optimizer_steps=7000,
                )
            )

    proposed_variants = {
        "M5B-REP-01": ("future_state",),
        "M5B-ACTION-01": (
            "phase_aligned_human_plan_plus_tplus1_query",
            "raw_human_plan_plus_lag_calibrated_query_diagnostic",
        ),
        "M5B-RET-01": ("random", "geometry", "visual", "geometry_plus_visual"),
        "M5B-SENS-01": ("topk3_h4_k4", "topk3_h16_k8"),
        "M5B-TIME-01": (
            "paper_v2_stride4_nominal7p5",
            "legacy_v01_stride3_nominal10",
            "policy_clock_10hz",
            "phase_or_dtw",
        ),
    }
    for experiment_id, variants in proposed_variants.items():
        for variant_id in variants:
            for seed in FORMAL_SEEDS:
                cells.append(
                    _cell(
                        "learned_training_checkpoint",
                        experiment_id,
                        variant_id,
                        "recap_hand_ret",
                        seed,
                        optimizer_steps=7000,
                    )
                )
    return cells


def nonlearned_method_cells() -> list[CandidateCell]:
    return [
        _cell(
            "nonlearned_method_artifact",
            "M5B-MAIN-01",
            "retrieval_only_projection",
            "retrieval_only",
            seed,
        )
        for seed in FORMAL_SEEDS
    ]


def _learned_parent(
    experiment_id: str,
    variant_id: str,
    seed: int,
    method_id: str = "recap_hand_ret",
) -> tuple[str, ...]:
    return (_train_id(experiment_id, variant_id, method_id, seed),)


def main_evaluation_cells() -> list[CandidateCell]:
    cells: list[CandidateCell] = []
    for seed in FORMAL_SEEDS:
        for method_id in (*LEARNED_METHODS, "retrieval_only"):
            if method_id == "retrieval_only":
                parents = (_nonlearned_id(seed),)
            else:
                parents = _learned_parent("M5B-MAIN-01", "frozen_main", seed, method_id)
            cells.append(
                _cell(
                    "checkpoint_linked_evaluation",
                    "M5B-MAIN-01",
                    "main_comparison_pool10",
                    method_id,
                    seed,
                    parents,
                )
            )
        for pool_size in POOL_SIZES[:-1]:
            cells.append(
                _cell(
                    "checkpoint_linked_evaluation",
                    "M5B-MAIN-01",
                    f"pool_growth_pool{pool_size}",
                    "recap_hand_ret",
                    seed,
                    _learned_parent("M5B-MAIN-01", "frozen_main", seed),
                )
            )
    return cells


def representation_evaluation_cells() -> list[CandidateCell]:
    parent_by_variant = {
        "residual": ("M5B-MAIN-01", "frozen_main", "recap_hand_ret"),
        "absolute": ("M5B-MAIN-01", "frozen_main", "co_training"),
        "future_state": ("M5B-REP-01", "future_state", "recap_hand_ret"),
    }
    cells = []
    for variant_id, (parent_experiment, parent_variant, parent_method) in parent_by_variant.items():
        for seed in FORMAL_SEEDS:
            cells.append(
                _cell(
                    "checkpoint_linked_evaluation",
                    "M5B-REP-01",
                    variant_id,
                    parent_method,
                    seed,
                    _learned_parent(parent_experiment, parent_variant, seed, parent_method),
                )
            )
    return cells


def action_evaluation_cells() -> list[CandidateCell]:
    variants = (
        "raw_human_plan_plus_tplus1_query_main",
        "phase_aligned_human_plan_plus_tplus1_query",
        "raw_human_plan_plus_lag_calibrated_query_diagnostic",
        "same_frame_query_negative_control",
        "swapped_role_negative_control",
        "scale_x2_negative_control",
    )
    trained_parent = {
        "phase_aligned_human_plan_plus_tplus1_query": "phase_aligned_human_plan_plus_tplus1_query",
        "raw_human_plan_plus_lag_calibrated_query_diagnostic": (
            "raw_human_plan_plus_lag_calibrated_query_diagnostic"
        ),
    }
    cells = []
    for variant_id in variants:
        parent_experiment = "M5B-ACTION-01" if variant_id in trained_parent else "M5B-MAIN-01"
        parent_variant = trained_parent.get(variant_id, "frozen_main")
        for seed in FORMAL_SEEDS:
            cells.append(
                _cell(
                    "checkpoint_linked_evaluation",
                    "M5B-ACTION-01",
                    variant_id,
                    "recap_hand_ret",
                    seed,
                    _learned_parent(parent_experiment, parent_variant, seed),
                )
            )
    return cells


def retrieval_evaluation_cells() -> list[CandidateCell]:
    cells = []
    for variant_id in ("random", "phase", "geometry", "visual", "geometry_plus_visual"):
        parent_experiment = "M5B-MAIN-01" if variant_id == "phase" else "M5B-RET-01"
        parent_variant = "frozen_main" if variant_id == "phase" else variant_id
        for seed in FORMAL_SEEDS:
            cells.append(
                _cell(
                    "checkpoint_linked_evaluation",
                    "M5B-RET-01",
                    variant_id,
                    "recap_hand_ret",
                    seed,
                    _learned_parent(parent_experiment, parent_variant, seed),
                )
            )
    return cells


def sensitivity_evaluation_cells() -> list[CandidateCell]:
    variants = ("topk1_h8_k8", "topk3_h8_k8", "topk5_h8_k8", "topk10_h8_k8")
    cells = []
    for variant_id in variants:
        for seed in FORMAL_SEEDS:
            cells.append(
                _cell(
                    "checkpoint_linked_evaluation",
                    "M5B-SENS-01",
                    variant_id,
                    "recap_hand_ret",
                    seed,
                    _learned_parent("M5B-MAIN-01", "frozen_main", seed),
                )
            )
    for variant_id in ("topk3_h4_k4", "topk3_h16_k8"):
        for seed in FORMAL_SEEDS:
            cells.append(
                _cell(
                    "checkpoint_linked_evaluation",
                    "M5B-SENS-01",
                    variant_id,
                    "recap_hand_ret",
                    seed,
                    _learned_parent("M5B-SENS-01", variant_id, seed),
                )
            )
    return cells


def time_evaluation_cells() -> list[CandidateCell]:
    time_views = (
        "nominal_camera_30hz_segmented",
        "paper_v2_stride4_nominal7p5",
        "legacy_v01_stride3_nominal10",
        "policy_clock_10hz",
        "phase_or_dtw",
    )
    cells = []
    for variant_id in time_views:
        parent_experiment = "M5B-MAIN-01" if variant_id == time_views[0] else "M5B-TIME-01"
        parent_variant = "frozen_main" if variant_id == time_views[0] else variant_id
        for seed in FORMAL_SEEDS:
            cells.append(
                _cell(
                    "checkpoint_linked_evaluation",
                    "M5B-TIME-01",
                    f"time_view_{variant_id}",
                    "recap_hand_ret",
                    seed,
                    _learned_parent(parent_experiment, parent_variant, seed),
                )
            )
    corruptions = {
        "frame_drop": ("5pct", "10pct", "20pct"),
        "timestamp_jitter": ("5ms", "10ms", "20ms"),
        "pause": ("0p2s", "0p5s", "1p0s"),
        "step_jump": ("1", "5", "20"),
    }
    for corruption_id, severities in corruptions.items():
        for severity in severities:
            for seed in FORMAL_SEEDS:
                cells.append(
                    _cell(
                        "checkpoint_linked_evaluation",
                        "M5B-TIME-01",
                        f"{corruption_id}_{severity}",
                        "recap_hand_ret",
                        seed,
                        _learned_parent("M5B-MAIN-01", "frozen_main", seed),
                    )
                )
    return cells


def resolution_evaluation_cells() -> list[CandidateCell]:
    variants = (
        "source_240x426_then_resize_224",
        "center_crop_240x424_then_resize_224",
        "center_crop_240x424_edge_pad_240x426_then_resize_224",
    )
    return [
        _cell(
            "checkpoint_linked_evaluation",
            "M5B-RES-01",
            variant_id,
            "recap_hand_ret",
            seed,
            _learned_parent("M5B-MAIN-01", "frozen_main", seed),
        )
        for variant_id in variants
        for seed in FORMAL_SEEDS
    ]


def report_cells() -> list[CandidateCell]:
    main_evaluations = main_evaluation_cells()
    seed_reports = [
        _cell(
            "aggregate_report",
            "M5B-QUAL-01",
            "seed_level_case_manifest",
            None,
            seed,
            tuple(cell.cell_id for cell in main_evaluations if cell.seed == seed),
        )
        for seed in FORMAL_SEEDS
    ]
    aggregate = _cell(
        "aggregate_report",
        "M5B-QUAL-01",
        "all_seed_best_worst_failure_report",
        None,
        None,
        tuple(cell.cell_id for cell in seed_reports),
    )
    return [*seed_reports, aggregate]


def candidate_cells() -> list[CandidateCell]:
    return [
        *learned_training_cells(),
        *nonlearned_method_cells(),
        *main_evaluation_cells(),
        *representation_evaluation_cells(),
        *action_evaluation_cells(),
        *retrieval_evaluation_cells(),
        *sensitivity_evaluation_cells(),
        *time_evaluation_cells(),
        *resolution_evaluation_cells(),
        *report_cells(),
    ]


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_candidate_registry() -> dict[str, Any]:
    cells = candidate_cells()
    records = [asdict(cell) for cell in cells]
    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell.artifact_kind] = counts.get(cell.artifact_kind, 0) + 1
    return {
        "schema_version": "human2robot-m5b-p2-candidate-cell-registry-v0",
        "status": STATUS,
        "formal_queue_allowed": False,
        "p2_acceptance_allowed": False,
        "seeds": list(FORMAL_SEEDS),
        "required_experiment_ids": list(REQUIRED_EXPERIMENT_IDS),
        "scope_interpretation": "minimum_claim_centered_candidate_not_literal_full_cross_product",
        "counts": counts,
        "cell_count": len(cells),
        "cells": records,
        "registry_sha256": canonical_json_sha256(records),
        "blocking_decisions": [
            "P2-SCOPE-01",
            "P2-NONLEARNED-01",
            "P2-REP-01",
            "P2-RET-01",
            "P2-VARIANTS-01",
            "P2-EVAL-01",
        ],
    }


def main() -> int:
    print(json.dumps(build_candidate_registry(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
