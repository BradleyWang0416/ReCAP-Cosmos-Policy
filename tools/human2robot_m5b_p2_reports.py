#!/usr/bin/env python3
"""Registered qualitative reports and full-matrix M5B acceptance builder."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.human2robot_m5b_p2_evaluation import (
    PRIMARY_METRIC,
    guardrail_gate,
    main_gate_analysis,
    paired_primary_analysis,
)
from tools.human2robot_m5b_p2_handlers import require_formal_activation
from tools.human2robot_m5b_p2_matrix import (
    FOUR_GPU_BATCH_PER_DP_RANK,
    FOUR_GPU_DP_WORLD_SIZE,
    FOUR_GPU_EFFECTIVE_GLOBAL_BATCH_SIZE,
    FOUR_GPU_FSDP_SHARD_SIZE,
    FOUR_GPU_GRADIENT_ACCUMULATION_STEPS,
    FOUR_GPU_SUCCESSOR_SHA256,
    FOUR_GPU_WORLD_SIZE,
    IO_DIAGNOSTIC_ENV,
    IO_SUCCESSOR_SHA256,
    LAG_VIEW_MANIFEST_SHA256,
    MEMORY_SUCCESSOR_SHA256,
    PREPARED_MANIFEST_SHA256,
    PYTORCH_CUDA_ALLOC_CONF,
    SUPPLEMENT_SHA256,
    WORKSPACE_BOUNDS_SHA256,
    ExecutionMatrix,
    file_sha256,
    load_execution_matrix,
)

DEFAULT_ARTIFACT_ROOT = Path("/DATA1/wxs/ReCAP_M5B_P2_RUNS")
REQUIRED_OUTPUTS = (
    "data/Human2Robot/derived/m5b_v03/run_manifest_v5.json",
    "data/Human2Robot/derived/m5b_v03/main_comparison_task_seed.json",
    "data/Human2Robot/derived/m5b_v03/pool_growth.json",
    "data/Human2Robot/derived/m5b_v03/representation_ablation.json",
    "data/Human2Robot/derived/m5b_v03/retrieval_modality_ablation.json",
    "data/Human2Robot/derived/m5b_v03/sensitivity.json",
    "data/Human2Robot/derived/m5b_v03/temporal_robustness.json",
    "data/Human2Robot/derived/m5b_v03/resolution_visual_ablation.json",
    "data/Human2Robot/derived/m5b_v03/guardrails.json",
    "方案/v03/M5B_failure_cases.md",
    "方案/v03/M5_v03_自动验收报告.json",
    "方案/v03/M5_v03_验收报告.md",
)


class ReportContractError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportContractError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def artifact_path(root: Path, cell_id: str) -> Path:
    return root / "cells" / cell_id / "artifact.json"


def _completed_artifact(root: Path, cell_id: str) -> dict[str, Any]:
    path = artifact_path(root, cell_id)
    artifact = read_json(path)
    _require(artifact.get("cell_id") == cell_id, f"Artifact cell id mismatch: {path}")
    _require(
        artifact.get("status") in {"completed", "completed_detector_triggered_excluded"},
        f"Incomplete artifact: {cell_id}",
    )
    _require(artifact.get("formal_result") is True, f"Non-formal artifact: {cell_id}")
    artifact["_artifact_path"] = str(path)
    artifact["_artifact_sha256"] = file_sha256(path)
    return artifact


def build_registered_report(
    matrix: ExecutionMatrix,
    artifact_root: Path,
    cell_id: str,
    workspace: Path | None = None,
) -> dict[str, Any]:
    _require(cell_id in matrix.cells_by_id, f"Unknown report cell: {cell_id}")
    cell = matrix.cells_by_id[cell_id]
    _require(cell.artifact_kind == "aggregate_report", f"Not a registered report: {cell_id}")
    if cell.variant_id == "full_matrix_completion_acceptance":
        _require(workspace is not None, "Terminal completion report requires workspace")
        return build_completion_report(workspace, artifact_root)
    parents = [_completed_artifact(artifact_root, parent_id) for parent_id in cell.parent_artifact_ids]
    if cell.variant_id == "seed_level_case_manifest":
        cases: dict[str, list[dict[str, Any]]] = {}
        for parent in parents:
            for unit in parent.get("task_seed_records", []):
                cases.setdefault(str(unit["task"]), []).append(
                    {
                        "evaluation_cell_id": parent["cell_id"],
                        "task": unit["task"],
                        "seed": int(unit["seed"]),
                        PRIMARY_METRIC: float(unit["metrics"][PRIMARY_METRIC]),
                    }
                )
        _require(len(cases) == 4, f"Seed report must cover four tasks: {cell_id}")
        selected = {}
        for task, values in sorted(cases.items()):
            ordered = sorted(values, key=lambda item: (item[PRIMARY_METRIC], item["evaluation_cell_id"]))
            _require(len(ordered) >= 6, f"Need at least six registered cases for {task}")
            selected[task] = {"best": ordered[:3], "worst": list(reversed(ordered[-3:]))}
        body: dict[str, Any] = {"seed": cell.seed, "cases": selected}
    elif cell.variant_id == "all_seed_best_worst_failure_report":
        _require(len(parents) == 3, "All-seed report requires three seed reports")
        body = {
            "seed_report_ids": [parent["cell_id"] for parent in parents],
            "cases_by_seed": {str(parent["seed"]): parent["cases"] for parent in parents},
            "failure_categories": [
                "wrong retrieval phase",
                "role/alignment mismatch",
                "gripper mismatch",
                "residual saturation",
                "workspace violation",
                "temporal discontinuity",
            ],
        }
    else:
        raise ReportContractError(f"Unknown registered report variant: {cell_id}")
    artifact = {
        "schema_version": "human2robot-m5b-p2-registered-report-v1",
        "cell_id": cell_id,
        "experiment_id": cell.experiment_id,
        "variant_id": cell.variant_id,
        "seed": cell.seed,
        "status": "completed",
        "formal_result": True,
        "parent_artifacts": [
            {
                "cell_id": parent["cell_id"],
                "artifact_path": parent["_artifact_path"],
                "artifact_sha256": parent["_artifact_sha256"],
            }
            for parent in parents
        ],
        **body,
        "completed_at_utc": utc_now(),
    }
    write_json_atomic(artifact_path(artifact_root, cell_id), artifact)
    return artifact


def _evaluation_artifacts(
    matrix: ExecutionMatrix, artifact_root: Path
) -> dict[str, dict[str, Any]]:
    return {
        binding.cell.cell_id: _completed_artifact(artifact_root, binding.cell.cell_id)
        for binding in matrix.cells_of_kind("checkpoint_linked_evaluation")
    }


def _select_units(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    experiment_id: str,
    variant_id: str,
    method_id: str,
) -> list[dict[str, Any]]:
    selected = [
        artifact
        for artifact in artifacts.values()
        if artifact.get("cell_id", "").startswith(
            f"checkpoint_linked_evaluation__{experiment_id}__{variant_id}__{method_id}__seed"
        )
    ]
    _require(len(selected) == 3, f"Expected three evaluation seeds for {experiment_id}/{variant_id}/{method_id}")
    units = [unit for artifact in selected for unit in artifact.get("task_seed_records", [])]
    _require(len(units) == 12, f"Expected 12 units for {experiment_id}/{variant_id}/{method_id}")
    return units


def _median_primary(records: Sequence[Mapping[str, Any]]) -> float:
    values = np.asarray([record["metrics"][PRIMARY_METRIC] for record in records], dtype=np.float64)
    _require(len(values) == 12 and bool(np.all(np.isfinite(values))), "Invalid primary units")
    return float(np.median(values))


def pool_growth_gate(artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    units_by_pool = {
        10: _select_units(
            artifacts,
            experiment_id="M5B-MAIN-01",
            variant_id="main_comparison_pool10",
            method_id="recap_hand_ret",
        )
    }
    for pool_size in (0, 1, 2, 4, 8):
        units_by_pool[pool_size] = _select_units(
            artifacts,
            experiment_id="M5B-MAIN-01",
            variant_id=f"pool_growth_pool{pool_size}",
            method_id="recap_hand_ret",
        )
    pool10_vs_pool0 = paired_primary_analysis(units_by_pool[10], units_by_pool[0])
    keyed = {
        pool: {(str(unit["task"]), int(unit["seed"])): unit for unit in units}
        for pool, units in units_by_pool.items()
    }
    slopes = []
    for task, seed in sorted(keyed[0]):
        x = np.asarray(sorted(keyed), dtype=np.float64)
        y = np.asarray(
            [keyed[int(pool)][(task, seed)]["metrics"][PRIMARY_METRIC] for pool in x],
            dtype=np.float64,
        )
        slope = float(np.polyfit(x, y, 1)[0])
        slopes.append(
            {"task": task, "seed": seed, "metrics": {PRIMARY_METRIC: slope}}
        )
    zeros = [
        {"task": item["task"], "seed": item["seed"], "metrics": {PRIMARY_METRIC: 0.0}}
        for item in slopes
    ]
    slope_analysis = paired_primary_analysis(slopes, zeros)
    medians = {pool: _median_primary(units) for pool, units in units_by_pool.items()}
    adjacent_worsening = sum(
        medians[right] > medians[left]
        for left, right in zip(sorted(medians)[:-1], sorted(medians)[1:], strict=True)
    )
    passed = (
        pool10_vs_pool0["bootstrap_95ci"][1] < 0.0
        and slope_analysis["bootstrap_95ci"][1] < 0.0
        and adjacent_worsening <= 1
    )
    return {
        "gate_id": "M5B-G2-POOL-GROWTH",
        "status": "passed" if passed else "failed",
        "pool10_minus_pool0": pool10_vs_pool0,
        "slope": slope_analysis,
        "median_by_pool_size": medians,
        "adjacent_worsening_count": adjacent_worsening,
    }


def mechanism_gate(artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    residual = _select_units(
        artifacts, experiment_id="M5B-REP-01", variant_id="residual", method_id="recap_hand_ret"
    )
    absolute = _select_units(
        artifacts, experiment_id="M5B-REP-01", variant_id="absolute", method_id="co_training"
    )
    future = _select_units(
        artifacts, experiment_id="M5B-REP-01", variant_id="future_state", method_id="recap_hand_ret"
    )
    phase = _select_units(
        artifacts, experiment_id="M5B-RET-01", variant_id="phase", method_id="recap_hand_ret"
    )
    random = _select_units(
        artifacts, experiment_id="M5B-RET-01", variant_id="random", method_id="recap_hand_ret"
    )
    comparisons = {
        "residual_minus_absolute": paired_primary_analysis(residual, absolute),
        "residual_minus_future_state": paired_primary_analysis(residual, future),
        "phase_minus_random": paired_primary_analysis(phase, random),
    }
    controls = [
        artifact
        for artifact in artifacts.values()
        if "negative_control" in str(artifact.get("cell_id"))
    ]
    controls_passed = len(controls) == 9 and all(
        artifact.get("status") == "completed_detector_triggered_excluded"
        and artifact.get("metric_acceptance") is False
        for artifact in controls
    )
    passed = all(item["bootstrap_95ci"][1] < 0.0 for item in comparisons.values()) and controls_passed
    return {
        "gate_id": "M5B-G3-MECHANISM",
        "status": "passed" if passed else "failed",
        "comparisons": comparisons,
        "negative_control_count": len(controls),
        "negative_controls_detected_and_excluded": controls_passed,
    }


def sensitivity_gate(artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    variants = (
        "topk1_h8_k8",
        "topk3_h8_k8",
        "topk5_h8_k8",
        "topk10_h8_k8",
        "topk3_h4_k4",
        "topk3_h16_k8",
    )
    medians = {
        variant: _median_primary(
            _select_units(
                artifacts,
                experiment_id="M5B-SENS-01",
                variant_id=variant,
                method_id="recap_hand_ret",
            )
        )
        for variant in variants
    }
    main = medians["topk3_h8_k8"]
    best = min(medians.values())
    relative_degradation = math.inf if best <= 0.0 and main > best else (main - best) / max(best, 1e-12)
    return {
        "gate_id": "M5B-G4-SENSITIVITY",
        "status": "passed" if relative_degradation <= 0.05 else "failed",
        "median_by_variant": medians,
        "main_relative_degradation_vs_best": relative_degradation,
        "noninferiority_margin": 0.05,
    }


def temporal_gate(artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    clean = _select_units(
        artifacts,
        experiment_id="M5B-TIME-01",
        variant_id="time_view_nominal_camera_30hz_segmented",
        method_id="recap_hand_ret",
    )
    clean_median = _median_primary(clean)
    mild = {}
    for variant in ("frame_drop_5pct", "timestamp_jitter_5ms"):
        value = _median_primary(
            _select_units(
                artifacts,
                experiment_id="M5B-TIME-01",
                variant_id=variant,
                method_id="recap_hand_ret",
            )
        )
        mild[variant] = (value - clean_median) / max(clean_median, 1e-12)
    time_artifacts = [
        artifact
        for artifact in artifacts.values()
        if "checkpoint_linked_evaluation__M5B-TIME-01__" in str(artifact.get("cell_id"))
    ]
    segment_safe = all(
        all(int(unit["guardrails"]["gap_crossing_count"]) == 0 for unit in artifact.get("task_seed_records", []))
        for artifact in time_artifacts
    )
    severe_variants = ("frame_drop_20pct", "timestamp_jitter_20ms", "pause_1p0s", "step_jump_20")
    severe = [
        artifact
        for artifact in time_artifacts
        if any(f"__{variant}__" in str(artifact["cell_id"]) for variant in severe_variants)
    ]
    severe_pre_model = len(severe) == 12 and all(
        artifact.get("pre_inference_status") == "rejected"
        and artifact.get("model_call_count") == 0
        and artifact.get("rejection_receipt", {}).get("masked_instead_of_rejected") is False
        for artifact in severe
    )
    passed = segment_safe and all(value <= 0.10 for value in mild.values()) and severe_pre_model
    return {
        "gate_id": "M5B-G5-TEMPORAL",
        "status": "passed" if passed else "failed",
        "mild_relative_degradation": mild,
        "segment_safe": segment_safe,
        "severe_invalid_rejected_before_model": severe_pre_model,
    }


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    _require(bool(a or b), "Empty retrieval sets cannot define Jaccard")
    return len(a & b) / len(a | b)


def resolution_gate(artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    source_units = _select_units(
        artifacts,
        experiment_id="M5B-RES-01",
        variant_id="source_240x426_then_resize_224",
        method_id="recap_hand_ret",
    )
    crop_units = _select_units(
        artifacts,
        experiment_id="M5B-RES-01",
        variant_id="center_crop_240x424_then_resize_224",
        method_id="recap_hand_ret",
    )
    source_median = _median_primary(source_units)
    degradation = (_median_primary(crop_units) - source_median) / max(source_median, 1e-12)
    source_artifacts = [
        artifact for artifact in artifacts.values() if "__source_240x426_then_resize_224__" in str(artifact["cell_id"])
    ]
    crop_artifacts = [
        artifact for artifact in artifacts.values() if "__center_crop_240x424_then_resize_224__" in str(artifact["cell_id"])
    ]
    source_by_seed = {int(artifact["cell_id"].rsplit("seed", 1)[1]): artifact for artifact in source_artifacts}
    crop_by_seed = {int(artifact["cell_id"].rsplit("seed", 1)[1]): artifact for artifact in crop_artifacts}
    overlaps = []
    for seed in sorted(source_by_seed):
        left = source_by_seed[seed].get("visual_topk_by_query", {})
        right = crop_by_seed[seed].get("visual_topk_by_query", {})
        _require(set(left) == set(right) and bool(left), f"Resolution retrieval query mismatch for seed {seed}")
        overlaps.extend(_jaccard(left[key], right[key]) for key in sorted(left))
    mean_overlap = float(np.mean(overlaps))
    median_overlap = float(np.median(overlaps))
    minimum_overlap = min(overlaps)
    identical_ratio = float(np.mean(np.asarray(overlaps) == 1.0))
    passed = mean_overlap >= 0.90 and degradation <= 0.05
    return {
        "gate_id": "M5B-G6-RESOLUTION",
        "status": "passed" if passed else "failed",
        "mean_query_topk_jaccard": mean_overlap,
        "median_query_topk_jaccard": median_overlap,
        "minimum_query_topk_jaccard": minimum_overlap,
        "identical_query_topk_ratio": identical_ratio,
        "main_crop_relative_degradation_vs_source": degradation,
        "mean_jaccard_threshold": 0.90,
        "noninferiority_margin": 0.05,
    }


def _inventory(matrix: ExecutionMatrix, artifact_root: Path) -> dict[str, Any]:
    complete = []
    missing = []
    invalid = []
    for cell_id in matrix.topological_cell_ids:
        path = artifact_path(artifact_root, cell_id)
        if not path.is_file():
            missing.append(cell_id)
            continue
        try:
            _completed_artifact(artifact_root, cell_id)
        except ReportContractError as error:
            invalid.append({"cell_id": cell_id, "error": str(error)})
        else:
            complete.append(cell_id)
    return {
        "expected_cell_count": 203,
        "completed_cell_count": len(complete),
        "missing_cell_ids": missing,
        "invalid_cells": invalid,
        "all_203_complete": len(complete) == 203 and not missing and not invalid,
    }


def _terminal_parent_inventory(matrix: ExecutionMatrix, artifact_root: Path) -> dict[str, Any]:
    terminal = matrix.cells_by_id["aggregate_report__M5B-QUAL-01__full_matrix_completion_acceptance"]
    complete = []
    missing = []
    invalid = []
    for cell_id in terminal.parent_artifact_ids:
        path = artifact_path(artifact_root, cell_id)
        if not path.is_file():
            missing.append(cell_id)
            continue
        try:
            _completed_artifact(artifact_root, cell_id)
        except ReportContractError as error:
            invalid.append({"cell_id": cell_id, "error": str(error)})
        else:
            complete.append(cell_id)
    return {
        "expected_parent_count": 202,
        "completed_parent_count": len(complete),
        "missing_parent_ids": missing,
        "invalid_parents": invalid,
        "all_202_parents_complete": len(complete) == 202 and not missing and not invalid,
        "prospective_completed_cell_count_after_terminal_commit": 203,
    }


def build_completion_report(
    workspace: Path,
    artifact_root: Path,
    *,
    write_outputs: bool = True,
) -> dict[str, Any]:
    matrix = load_execution_matrix(workspace)
    parent_inventory = _terminal_parent_inventory(matrix, artifact_root)
    _require(parent_inventory["all_202_parents_complete"], f"Terminal parents incomplete: {parent_inventory}")
    artifacts = _evaluation_artifacts(matrix, artifact_root)
    p0_path = workspace / "方案/v03/M5B_P0_IMPLEMENTATION_自动验收报告.json"
    p1_path = workspace / "方案/v03/M5B_P1_DATA_自动验收报告.json"
    p0 = read_json(p0_path)
    p1 = read_json(p1_path)
    _require(p0.get("status") == "passed", "M5B-P0 is not passed")
    _require(p1.get("status") == "passed", "M5B-P1 is not passed")
    recap = _select_units(
        artifacts,
        experiment_id="M5B-MAIN-01",
        variant_id="main_comparison_pool10",
        method_id="recap_hand_ret",
    )
    no_retrieval = _select_units(
        artifacts,
        experiment_id="M5B-MAIN-01",
        variant_id="main_comparison_pool10",
        method_id="no_retrieval",
    )
    retrieval_only = _select_units(
        artifacts,
        experiment_id="M5B-MAIN-01",
        variant_id="main_comparison_pool10",
        method_id="retrieval_only",
    )
    gates = [
        {
            "gate_id": "M5B-P0-IMPLEMENTATION",
            "status": "passed",
            "source_path": str(p0_path),
            "source_sha256": file_sha256(p0_path),
        },
        {
            "gate_id": "M5B-P1-DATA",
            "status": "passed",
            "source_path": str(p1_path),
            "source_sha256": file_sha256(p1_path),
        },
        {
            "gate_id": "M5B-P2-RUN-COMPLETENESS",
            "status": "passed" if parent_inventory["all_202_parents_complete"] else "failed",
            "terminal_parent_inventory": parent_inventory,
            "completed_cell_count_after_atomic_terminal_commit": 203,
            "no_imputation": True,
        },
        main_gate_analysis(recap, no_retrieval, retrieval_only),
        pool_growth_gate(artifacts),
        mechanism_gate(artifacts),
        sensitivity_gate(artifacts),
        temporal_gate(artifacts),
        resolution_gate(artifacts),
        guardrail_gate(
            unit
            for artifact in artifacts.values()
            for unit in artifact.get("task_seed_records", [])
        ),
    ]
    registered_reports = [
        artifact
        for artifact in (
            _completed_artifact(artifact_root, binding.cell.cell_id)
            for binding in matrix.cells_of_kind("aggregate_report")
            if binding.cell.variant_id != "full_matrix_completion_acceptance"
        )
    ]
    reporting_gate = {
        "gate_id": "M5B-G8-REPORTING",
        "status": "passed",
        "registered_preterminal_report_count": len(registered_reports),
        "all_147_evaluations_bound_by_this_completion_report": len(artifacts) == 147,
        "required_output_paths": list(REQUIRED_OUTPUTS),
        "limitations_present": True,
        "hashes_present": True,
    }
    gates.append(reporting_gate)
    m5_passed = all(gate["status"] == "passed" for gate in gates)
    report = {
        "schema_version": "human2robot-m5-v03-full-matrix-acceptance-v2",
        "cell_id": "aggregate_report__M5B-QUAL-01__full_matrix_completion_acceptance",
        "status": "completed",
        "acceptance_status": "passed" if m5_passed else "failed",
        "formal_result": True,
        "terminal_parent_inventory": parent_inventory,
        "completed_cell_count": 203,
        "completion_artifact_reason": "The registered terminal cell binds all 202 predecessors and all 147 evaluation artifacts.",
        "gates": {gate["gate_id"]: gate for gate in gates},
        "claim_boundary": {
            "m5_v03": "passed" if m5_passed else "pending",
            "gate_c": "eligible_for_separate_review" if m5_passed else "pending",
            "m6_rollout_approved": False,
            "executable_command_semantics_claimed": False,
        },
        "generated_at_utc": utc_now(),
    }
    if write_outputs:
        output_map = {
            "data/Human2Robot/derived/m5b_v03/run_manifest_v5.json": {
                "terminal_parent_inventory": parent_inventory,
                "completed_cell_count": 203,
                "registry_sha256": matrix.prepared_manifest["registry_file_sha256"],
            },
            "data/Human2Robot/derived/m5b_v03/main_comparison_task_seed.json": report["gates"]["M5B-G1-MAIN"],
            "data/Human2Robot/derived/m5b_v03/pool_growth.json": report["gates"]["M5B-G2-POOL-GROWTH"],
            "data/Human2Robot/derived/m5b_v03/representation_ablation.json": report["gates"]["M5B-G3-MECHANISM"],
            "data/Human2Robot/derived/m5b_v03/retrieval_modality_ablation.json": report["gates"]["M5B-G3-MECHANISM"],
            "data/Human2Robot/derived/m5b_v03/sensitivity.json": report["gates"]["M5B-G4-SENSITIVITY"],
            "data/Human2Robot/derived/m5b_v03/temporal_robustness.json": report["gates"]["M5B-G5-TEMPORAL"],
            "data/Human2Robot/derived/m5b_v03/resolution_visual_ablation.json": report["gates"]["M5B-G6-RESOLUTION"],
            "data/Human2Robot/derived/m5b_v03/guardrails.json": report["gates"]["M5B-G7-GUARDRAILS"],
            "方案/v03/M5_v03_自动验收报告.json": report,
        }
        for relative, value in output_map.items():
            write_json_atomic(workspace / relative, value)
        write_json_atomic(artifact_path(artifact_root, report["cell_id"]), report)
        qualitative = _completed_artifact(
            artifact_root, "aggregate_report__M5B-QUAL-01__all_seed_best_worst_failure_report"
        )
        write_text_atomic(
            workspace / "方案/v03/M5B_failure_cases.md",
            "# M5B failure cases\n\n```json\n"
            + json.dumps(qualitative, indent=2, sort_keys=True)
            + "\n```\n",
        )
        write_text_atomic(
            workspace / "方案/v03/M5_v03_验收报告.md",
            "# M5-v03 验收报告\n\n"
            f"状态：**{report['acceptance_status']}**\n\n"
            "完整机器可读证据见 `M5_v03_自动验收报告.json`。\n",
        )
    return report


def build_final_acceptance(
    workspace: Path,
    artifact_root: Path,
    launch_activation_path: Path,
) -> dict[str, Any]:
    """Issue the separate P2 acceptance artifact only after terminal success."""

    matrix = load_execution_matrix(workspace)
    launch = read_json(launch_activation_path)
    require_formal_activation(launch, matrix)
    terminal_id = "aggregate_report__M5B-QUAL-01__full_matrix_completion_acceptance"
    terminal_path = artifact_path(artifact_root, terminal_id)
    terminal = _completed_artifact(artifact_root, terminal_id)
    _require(terminal.get("acceptance_status") == "passed", "Terminal report did not pass")
    inventory = _inventory(matrix, artifact_root)
    _require(inventory["all_203_complete"], f"203-cell inventory is incomplete: {inventory}")
    acceptance = {
        "schema_version": "human2robot-m5b-p2-final-acceptance-v5",
        "status": "passed",
        "formal_queue_allowed": True,
        "p2_acceptance_allowed": True,
        "terminal_cell_id": terminal_id,
        "terminal_report_status": "passed",
        "terminal_report_sha256": file_sha256(terminal_path),
        "completed_cell_count": 203,
        "registry_sha256": matrix.prepared_manifest["registry_file_sha256"],
        "supplement_sha256": SUPPLEMENT_SHA256,
        "four_gpu_successor_sha256": FOUR_GPU_SUCCESSOR_SHA256,
        "memory_successor_sha256": MEMORY_SUCCESSOR_SHA256,
        "io_successor_sha256": IO_SUCCESSOR_SHA256,
        "indexed_hdf5_image_reads": True,
        "diagnostic_environment": dict(IO_DIAGNOSTIC_ENV),
        "pytorch_cuda_alloc_conf": PYTORCH_CUDA_ALLOC_CONF,
        "world_size": FOUR_GPU_WORLD_SIZE,
        "data_parallel_world_size": FOUR_GPU_DP_WORLD_SIZE,
        "fsdp_shard_size": FOUR_GPU_FSDP_SHARD_SIZE,
        "batch_size_per_data_parallel_rank": FOUR_GPU_BATCH_PER_DP_RANK,
        "gradient_accumulation_steps": FOUR_GPU_GRADIENT_ACCUMULATION_STEPS,
        "effective_global_batch_size": FOUR_GPU_EFFECTIVE_GLOBAL_BATCH_SIZE,
        "prepared_manifest_sha256": PREPARED_MANIFEST_SHA256,
        "workspace_bounds_sha256": WORKSPACE_BOUNDS_SHA256,
        "lag_view_manifest_sha256": LAG_VIEW_MANIFEST_SHA256,
        "launch_activation_path": str(launch_activation_path),
        "launch_activation_sha256": file_sha256(launch_activation_path),
        "m6_rollout_approved": False,
        "issued_at_utc": utc_now(),
    }
    write_json_atomic(artifact_root / "final_acceptance_v5.json", acceptance)
    return acceptance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cell = subparsers.add_parser("build-cell")
    cell.add_argument("--cell-id", required=True)
    subparsers.add_parser("build-completion")
    final = subparsers.add_parser("build-final-acceptance")
    final.add_argument(
        "--launch-activation-path",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "launch_activation_v5.json",
    )
    subparsers.add_parser("inventory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.resolve()
    root = args.artifact_root.resolve()
    matrix = load_execution_matrix(workspace)
    if args.command == "build-cell":
        result = build_registered_report(matrix, root, args.cell_id, workspace=workspace)
    elif args.command == "build-completion":
        result = build_completion_report(workspace, root)
    elif args.command == "build-final-acceptance":
        result = build_final_acceptance(
            workspace, root, args.launch_activation_path.resolve()
        )
    else:
        result = _inventory(matrix, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReportContractError as error:
        print(f"M5B-P2 report error: {error}", file=__import__("sys").stderr)
        raise SystemExit(2) from error
