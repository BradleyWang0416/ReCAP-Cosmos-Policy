#!/usr/bin/env python3
"""Held-out reconstruction, metrics, statistics, and report contracts for M5B-P2.

The functions are intentionally model-agnostic: a model runner supplies one
normalized action prediction per query/retrieval rank, while this module owns
the frozen inverse-normalization, canonical reconstruction, aggregation,
guardrails, task-seed aggregation, and paired statistical tests.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cosmos_policy.datasets.human2robot_p2_contract import (
    aggregate_canonical_predictions,
    canonical_window_metrics,
    project_canonical_trajectory,
    reconstruct_future_state,
)

PRIMARY_METRIC = "position_error_median_canonical"
SECONDARY_METRICS = (
    "orientation_error_median_rad",
    "gripper_error_median",
    "final_position_error_median_canonical",
    "canonical_error_median",
)
FORMAL_SEEDS = (20260711, 20260712, 20260713)
BOOTSTRAP_SEED = 20260711
BOOTSTRAP_RESAMPLES = 10_000


class EvaluationContractError(RuntimeError):
    """Raised instead of accepting incomplete, nonfinite, or unbound evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationContractError(message)


def _finite_array(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if shape is not None:
        _require(result.shape == shape, f"{name} shape is {result.shape}, expected {shape}")
    _require(bool(np.all(np.isfinite(result))), f"{name} contains nonfinite values")
    return result


def inverse_minmax(normalized: np.ndarray, minimum: Sequence[float], maximum: Sequence[float]) -> np.ndarray:
    """Invert the dataset's frozen [-1, 1] train-only min/max transform."""
    value = _finite_array(normalized, "normalized prediction")
    low = _finite_array(minimum, "normalization minimum", (10,))
    high = _finite_array(maximum, "normalization maximum", (10,))
    _require(bool(np.all(high >= low)), "Normalization maximum is below minimum")
    return (low + 0.5 * (value + 1.0) * (high - low + 1e-8)).astype(np.float32)


def reconstruct_rank_prediction(
    normalized_prediction: np.ndarray | None,
    *,
    target_representation: str,
    statistics: Mapping[str, Any],
    current_state_10d: np.ndarray,
    aligned_pool_10d: np.ndarray,
    k_steps: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return canonical Kx10 prediction and raw residual/transition if applicable."""
    current = _finite_array(current_state_10d, "current state", (10,))
    aligned = _finite_array(aligned_pool_10d, "aligned pool")
    _require(aligned.ndim == 2 and aligned.shape[1] == 10 and len(aligned) >= k_steps, "Bad aligned pool")
    _require(k_steps > 0, "K must be positive")
    if target_representation == "retrieval_only":
        _require(normalized_prediction is None, "Retrieval-only must not carry model output")
        return project_canonical_trajectory(aligned[:k_steps]), None

    normalized = _finite_array(normalized_prediction, "normalized prediction", (k_steps, 10))
    if target_representation == "residual":
        raw = inverse_minmax(
            normalized,
            statistics["residual_10d_min"],
            statistics["residual_10d_max"],
        )
        return project_canonical_trajectory(aligned[:k_steps] + raw), raw
    if target_representation == "absolute":
        raw = inverse_minmax(
            normalized,
            statistics["query_bc_target_10d_min"],
            statistics["query_bc_target_10d_max"],
        )
        return project_canonical_trajectory(raw), None
    if target_representation == "future_state":
        raw = inverse_minmax(
            normalized,
            statistics["future_state_transition_10d_min"],
            statistics["future_state_transition_10d_max"],
        )
        return reconstruct_future_state(current, raw), raw
    raise EvaluationContractError(f"Unknown target representation: {target_representation}")


def has_long_term_residual_saturation(raw_residual: np.ndarray | None, p99: float) -> bool:
    if raw_residual is None:
        return False
    value = _finite_array(raw_residual, "raw residual")
    _require(value.ndim == 2 and value.shape[1] == 10, "Bad residual shape")
    _require(math.isfinite(float(p99)) and float(p99) >= 0.0, "Bad residual P99")
    above = np.linalg.norm(value, axis=1) > float(p99)
    run = 0
    for item in above:
        run = run + 1 if bool(item) else 0
        if run >= 5:
            return True
    return False


def workspace_violation_count(
    canonical_prediction: np.ndarray,
    workspace_xyz_min: Sequence[float],
    workspace_xyz_max: Sequence[float],
) -> int:
    value = _finite_array(canonical_prediction, "canonical prediction")
    _require(value.ndim == 2 and value.shape[1] == 10, "Bad canonical prediction shape")
    low = _finite_array(workspace_xyz_min, "workspace xyz minimum", (3,))
    high = _finite_array(workspace_xyz_max, "workspace xyz maximum", (3,))
    _require(bool(np.all(high > low)), "Workspace bounds must be strictly increasing")
    outside = np.any((value[:, :3] < low) | (value[:, :3] > high), axis=1)
    return int(np.count_nonzero(outside))


@dataclass(frozen=True)
class RankPrediction:
    query_id: str
    task: str
    episode_id: str
    current_row: int
    retrieval_rank: int
    target_representation: str
    normalized_prediction: np.ndarray | None
    current_state_10d: np.ndarray
    aligned_pool_10d: np.ndarray
    query_target_10d: np.ndarray
    gap_crossing_count: int = 0
    heldout_target_retrieval_feature_count: int = 0


def evaluate_ranked_query(
    rank_predictions: Sequence[RankPrediction],
    *,
    statistics: Mapping[str, Any],
    workspace_xyz_min: Sequence[float] | None,
    workspace_xyz_max: Sequence[float] | None,
) -> dict[str, Any]:
    """Aggregate retrieval ranks and emit one window-level independent record."""
    _require(bool(rank_predictions), "No rank predictions supplied")
    ordered = sorted(rank_predictions, key=lambda item: item.retrieval_rank)
    first = ordered[0]
    _require([item.retrieval_rank for item in ordered] == list(range(len(ordered))), "Retrieval ranks are incomplete")
    identity = (
        first.query_id,
        first.task,
        first.episode_id,
        first.current_row,
        first.target_representation,
    )
    _require(
        all(
            (item.query_id, item.task, item.episode_id, item.current_row, item.target_representation)
            == identity
            for item in ordered
        ),
        "Rank predictions do not belong to the same query",
    )
    target = _finite_array(first.query_target_10d, "query target")
    _require(target.ndim == 2 and target.shape[1] == 10, "Bad query target")
    k_steps = len(target)
    canonical: list[np.ndarray] = []
    saturation_count = 0
    for item in ordered:
        _require(np.array_equal(np.asarray(item.query_target_10d), np.asarray(first.query_target_10d)), "Rank targets differ")
        prediction, raw = reconstruct_rank_prediction(
            item.normalized_prediction,
            target_representation=item.target_representation,
            statistics=statistics,
            current_state_10d=item.current_state_10d,
            aligned_pool_10d=item.aligned_pool_10d,
            k_steps=k_steps,
        )
        canonical.append(prediction)
        if item.target_representation == "residual":
            saturation_count += int(
                has_long_term_residual_saturation(raw, float(statistics["residual_norm_p99"]))
            )
    aggregate = aggregate_canonical_predictions(canonical)
    if workspace_xyz_min is None or workspace_xyz_max is None:
        violations: int | None = None
    else:
        violations = workspace_violation_count(aggregate, workspace_xyz_min, workspace_xyz_max)
    metrics = dict(canonical_window_metrics(aggregate, target))
    metrics["residual_norm_median"] = float(
        np.median(np.linalg.norm(target - aggregate, axis=1))
    )
    return {
        "query_id": first.query_id,
        "task": first.task,
        "episode_id": first.episode_id,
        "current_row": first.current_row,
        "rank_count": len(ordered),
        "metrics": metrics,
        "guardrails": {
            "nonfinite_prediction_count": 0,
            "gap_crossing_count": sum(int(item.gap_crossing_count) for item in ordered),
            "workspace_violation_count": violations,
            "workspace_clipping_applied_count": 0,
            "heldout_target_retrieval_feature_count": sum(
                int(item.heldout_target_retrieval_feature_count) for item in ordered
            ),
            "long_term_residual_saturation_count": saturation_count,
        },
        "canonical_prediction_10d": aggregate.tolist(),
    }


def aggregate_task_seed_windows(
    window_records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Collapse windows to the preregistered heldout-task x seed unit."""
    _require(seed in FORMAL_SEEDS, f"Unfrozen seed: {seed}")
    _require(bool(window_records), "No window records")
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in window_records:
        task = str(record.get("task", ""))
        _require(bool(task), "Window record has no task")
        by_task[task].append(record)
    _require(len(by_task) == 4, f"Expected four heldout tasks, got {len(by_task)}")
    result = []
    metric_ids = (PRIMARY_METRIC, *SECONDARY_METRICS, "residual_norm_median")
    for task in sorted(by_task):
        records = by_task[task]
        aggregated: dict[str, float] = {}
        for metric_id in metric_ids:
            values = np.asarray([item["metrics"][metric_id] for item in records], dtype=np.float64)
            _require(bool(np.all(np.isfinite(values))), f"Nonfinite {metric_id} for {task}")
            aggregated[metric_id] = float(np.median(values))
        guards: dict[str, int] = {}
        for guard_id in (
            "nonfinite_prediction_count",
            "gap_crossing_count",
            "workspace_violation_count",
            "workspace_clipping_applied_count",
            "heldout_target_retrieval_feature_count",
            "long_term_residual_saturation_count",
        ):
            values = [item["guardrails"].get(guard_id) for item in records]
            _require(all(value is not None for value in values), f"Unbound guardrail {guard_id} for {task}")
            guards[guard_id] = sum(int(value) for value in values)
        result.append(
            {
                "task": task,
                "seed": seed,
                "window_count": len(records),
                "metrics": aggregated,
                "guardrails": guards,
            }
        )
    return result


def _unit_values(
    records: Sequence[Mapping[str, Any]], metric_id: str = PRIMARY_METRIC
) -> dict[tuple[str, int], float]:
    result: dict[tuple[str, int], float] = {}
    for record in records:
        key = (str(record["task"]), int(record["seed"]))
        _require(key not in result, f"Duplicate task-seed unit: {key}")
        value = float(record["metrics"][metric_id])
        _require(math.isfinite(value), f"Nonfinite task-seed metric: {key}")
        result[key] = value
    return result


def _validate_twelve_units(values: Mapping[tuple[str, int], float]) -> tuple[str, ...]:
    tasks = tuple(sorted({task for task, _ in values}))
    _require(len(tasks) == 4, f"Expected four tasks, got {len(tasks)}")
    expected = {(task, seed) for task in tasks for seed in FORMAL_SEEDS}
    _require(set(values) == expected, "Task-seed units are missing or use unfrozen seeds")
    return tasks


def hierarchical_paired_bootstrap(
    differences: Mapping[tuple[str, int], float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    random_seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    tasks = _validate_twelve_units(differences)
    _require(resamples > 0, "Bootstrap resamples must be positive")
    rng = np.random.default_rng(random_seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        sample: list[float] = []
        for task in sampled_tasks:
            sampled_seeds = rng.choice(FORMAL_SEEDS, size=len(FORMAL_SEEDS), replace=True)
            sample.extend(differences[(str(task), int(seed))] for seed in sampled_seeds)
        estimates[index] = float(np.median(sample))
    low, high = np.percentile(estimates, (2.5, 97.5))
    return float(low), float(high)


def exact_one_sided_sign_flip_p(differences: Mapping[tuple[str, int], float]) -> float:
    _validate_twelve_units(differences)
    values = np.asarray([differences[key] for key in sorted(differences)], dtype=np.float64)
    observed = float(np.mean(values))
    favorable_or_equal = 0
    total = 1 << len(values)
    for bits in range(total):
        signs = np.fromiter(
            (1.0 if bits & (1 << index) else -1.0 for index in range(len(values))),
            dtype=np.float64,
        )
        favorable_or_equal += int(float(np.mean(values * signs)) <= observed + 1e-15)
    return favorable_or_equal / total


def paired_primary_analysis(
    treatment_records: Sequence[Mapping[str, Any]],
    baseline_records: Sequence[Mapping[str, Any]],
    *,
    metric_id: str = PRIMARY_METRIC,
) -> dict[str, Any]:
    treatment = _unit_values(treatment_records, metric_id)
    baseline = _unit_values(baseline_records, metric_id)
    _validate_twelve_units(treatment)
    _require(set(treatment) == set(baseline), "Paired comparison units differ")
    differences = {key: treatment[key] - baseline[key] for key in treatment}
    ci_low, ci_high = hierarchical_paired_bootstrap(differences)
    values = np.asarray(list(differences.values()), dtype=np.float64)
    task_medians = {
        task: float(np.median([differences[(task, seed)] for seed in FORMAL_SEEDS]))
        for task in sorted({key[0] for key in differences})
    }
    return {
        "metric_id": metric_id,
        "paired_difference": "treatment_minus_baseline",
        "unit_count": len(differences),
        "differences": [
            {"task": task, "seed": seed, "difference": differences[(task, seed)]}
            for task, seed in sorted(differences)
        ],
        "effect_median": float(np.median(values)),
        "effect_iqr": [float(item) for item in np.percentile(values, (25, 75))],
        "bootstrap_95ci": [ci_low, ci_high],
        "raw_p": exact_one_sided_sign_flip_p(differences),
        "task_median_differences": task_medians,
        "improved_task_count": sum(value < 0.0 for value in task_medians.values()),
    }


def holm_adjust_two(comparisons: Mapping[str, float]) -> dict[str, float]:
    _require(len(comparisons) == 2, "Frozen primary family contains exactly two comparisons")
    _require(all(0.0 <= float(value) <= 1.0 for value in comparisons.values()), "Invalid p value")
    ordered = sorted(comparisons.items(), key=lambda item: (float(item[1]), item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, raw) in enumerate(ordered):
        candidate = min(1.0, (len(ordered) - index) * float(raw))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def main_gate_analysis(
    recap_records: Sequence[Mapping[str, Any]],
    no_retrieval_records: Sequence[Mapping[str, Any]],
    retrieval_only_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    comparisons = {
        "recap_minus_no_retrieval": paired_primary_analysis(recap_records, no_retrieval_records),
        "recap_minus_retrieval_only": paired_primary_analysis(recap_records, retrieval_only_records),
    }
    adjusted = holm_adjust_two({name: item["raw_p"] for name, item in comparisons.items()})
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value
    passed = all(
        item["bootstrap_95ci"][1] < 0.0
        and item["holm_adjusted_p"] < 0.05
        and item["improved_task_count"] >= 3
        for item in comparisons.values()
    )
    return {
        "gate_id": "M5B-G1-MAIN",
        "status": "passed" if passed else "failed",
        "comparisons": comparisons,
    }


def guardrail_gate(task_seed_records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(task_seed_records)
    _require(bool(records), "No task-seed guardrail records")
    failures = []
    for record in records:
        for guard_id, value in record["guardrails"].items():
            if value is None or int(value) != 0:
                failures.append(
                    {
                        "task": record["task"],
                        "seed": int(record["seed"]),
                        "guardrail_id": guard_id,
                        "value": value,
                    }
                )
    return {
        "gate_id": "M5B-G7-GUARDRAILS",
        "status": "passed" if not failures else "failed",
        "checked_unit_count": len(records),
        "failures": failures,
    }


def qualitative_case_manifest(task_seed_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the frozen 3 best and 3 worst seed-level cases for every task."""
    values = _unit_values(task_seed_records)
    _validate_twelve_units(values)
    cases: dict[str, Any] = {}
    for task in sorted({key[0] for key in values}):
        ordered = sorted(
            (
                {"task": task, "seed": seed, PRIMARY_METRIC: values[(task, seed)]}
                for seed in FORMAL_SEEDS
            ),
            key=lambda item: (item[PRIMARY_METRIC], item["seed"]),
        )
        cases[task] = {"best": ordered[:3], "worst": list(reversed(ordered[-3:]))}
    return {
        "schema_version": "human2robot-m5b-p2-qualitative-case-manifest-v1",
        "selection_rule": "3 best and 3 worst task-seed cases per heldout task by primary metric",
        "cases": cases,
    }


def serialize_rank_prediction(value: RankPrediction) -> dict[str, Any]:
    result = asdict(value)
    for key in (
        "normalized_prediction",
        "current_state_10d",
        "aligned_pool_10d",
        "query_target_10d",
    ):
        if result[key] is not None:
            result[key] = np.asarray(result[key]).tolist()
    return result
