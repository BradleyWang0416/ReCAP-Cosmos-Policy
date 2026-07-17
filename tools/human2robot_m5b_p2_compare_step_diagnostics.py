#!/usr/bin/env python3
"""Paired comparison of two non-formal M5B-P2 checkpoint diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PRIMARY_METRIC = "canonical_error_median"
CORE_METRICS = (
    "canonical_error_median",
    "position_error_median_canonical",
    "final_position_error_median_canonical",
    "orientation_error_median_rad",
    "gripper_error_median",
    "residual_norm_median",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_map(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = {str(record["query_id"]): record for record in payload["window_records"]}
    _require(len(records) == len(payload["window_records"]), "Duplicate query_id")
    return records


def _selection_signature(payload: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    return sorted(
        (
            str(item["task"]),
            str(item["query_id"]),
            str(item["episode_id"]),
            int(item["current_row"]),
            int(item["query_index"]),
            int(item["selected_position"]),
            int(item["candidate_count"]),
            str(item["selection_rule"]),
        )
        for item in payload["selection"]["query_provenance"]
    )


def _metric_comparison(
    reference_records: Mapping[str, Mapping[str, Any]],
    candidate_records: Mapping[str, Mapping[str, Any]],
    metric: str,
) -> dict[str, Any]:
    query_ids = sorted(reference_records)
    reference_values = np.asarray(
        [float(reference_records[query_id]["metrics"][metric]) for query_id in query_ids]
    )
    candidate_values = np.asarray(
        [float(candidate_records[query_id]["metrics"][metric]) for query_id in query_ids]
    )
    deltas = candidate_values - reference_values
    reference_mean = float(reference_values.mean())
    candidate_mean = float(candidate_values.mean())
    return {
        "reference_mean": reference_mean,
        "candidate_mean": candidate_mean,
        "absolute_delta": candidate_mean - reference_mean,
        "relative_change_fraction": (
            (candidate_mean - reference_mean) / reference_mean
            if reference_mean != 0
            else None
        ),
        "lower_is_better": True,
        "improved_query_count": int(np.count_nonzero(deltas < 0)),
        "unchanged_query_count": int(np.count_nonzero(deltas == 0)),
        "worsened_query_count": int(np.count_nonzero(deltas > 0)),
        "query_count": len(query_ids),
    }


def compare(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    for name, payload in (("reference", reference), ("candidate", candidate)):
        _require(payload.get("status") == "completed", f"{name} diagnostic incomplete")
        _require(payload.get("formal_result") is False, f"{name} mislabeled formal")
        _require(
            payload.get("acceptance_eligible") is False,
            f"{name} mislabeled acceptance-eligible",
        )
    reference_iteration = int(reference["checkpoint"]["expected_iteration"])
    candidate_iteration = int(candidate["checkpoint"]["expected_iteration"])
    _require(candidate_iteration > reference_iteration, "Candidate must be a later checkpoint")
    _require(reference["method_id"] == candidate["method_id"], "Method mismatch")
    _require(reference["run_seed"] == candidate["run_seed"], "Run seed mismatch")
    _require(reference["binding"] == candidate["binding"], "Evaluation binding mismatch")
    _require(reference["sampler"] == candidate["sampler"], "Sampler mismatch")
    reference_selection = _selection_signature(reference)
    candidate_selection = _selection_signature(candidate)
    _require(reference_selection == candidate_selection, "Selected data/query mismatch")
    reference_records = _record_map(reference)
    candidate_records = _record_map(candidate)
    _require(reference_records.keys() == candidate_records.keys(), "Query set mismatch")
    for query_id in reference_records:
        reference_baseline = reference_records[query_id]["diagnostic_no_motion_baseline_metrics"]
        candidate_baseline = candidate_records[query_id]["diagnostic_no_motion_baseline_metrics"]
        _require(reference_baseline.keys() == candidate_baseline.keys(), "Baseline metric mismatch")
        for metric in reference_baseline:
            _require(
                np.isclose(
                    float(reference_baseline[metric]),
                    float(candidate_baseline[metric]),
                    rtol=0.0,
                    atol=1e-12,
                ),
                f"No-motion baseline changed for {query_id}: {metric}",
            )
    common_metrics = sorted(
        set.intersection(
            *(set(record["metrics"]) for record in reference_records.values()),
            *(set(record["metrics"]) for record in candidate_records.values()),
        )
    )
    metric_comparisons = {
        metric: _metric_comparison(reference_records, candidate_records, metric)
        for metric in common_metrics
    }
    primary_query_comparison = []
    for query_id in sorted(reference_records, key=lambda q: str(reference_records[q]["task"])):
        reference_value = float(reference_records[query_id]["metrics"][PRIMARY_METRIC])
        candidate_value = float(candidate_records[query_id]["metrics"][PRIMARY_METRIC])
        primary_query_comparison.append(
            {
                "task": str(reference_records[query_id]["task"]),
                "query_id": query_id,
                "reference": reference_value,
                "candidate": candidate_value,
                "absolute_delta": candidate_value - reference_value,
                "relative_change_fraction": (
                    (candidate_value - reference_value) / reference_value
                    if reference_value != 0
                    else None
                ),
                "improved": candidate_value < reference_value,
            }
        )
    guardrail_delta = {
        name: int(candidate["guardrail_totals"][name])
        - int(reference["guardrail_totals"][name])
        for name in sorted(reference["guardrail_totals"])
    }
    guards_no_regression = all(delta <= 0 for delta in guardrail_delta.values())
    comparable_core = [metric for metric in CORE_METRICS if metric in metric_comparisons]
    nonincreasing_core_count = sum(
        metric_comparisons[metric]["candidate_mean"]
        <= metric_comparisons[metric]["reference_mean"]
        for metric in comparable_core
    )
    primary = metric_comparisons[PRIMARY_METRIC]
    majority_threshold = len(reference_records) // 2 + 1
    positive = (
        primary["candidate_mean"] < primary["reference_mean"]
        and primary["improved_query_count"] >= majority_threshold
        and nonincreasing_core_count == len(comparable_core)
        and guards_no_regression
    )
    reference_no_motion = float(
        reference["diagnostic_no_motion_baseline_summary"][PRIMARY_METRIC]["mean"]
    )
    candidate_no_motion = float(
        candidate["diagnostic_no_motion_baseline_summary"][PRIMARY_METRIC]["mean"]
    )
    _require(
        np.isclose(reference_no_motion, candidate_no_motion, rtol=0.0, atol=1e-12),
        "Aggregate no-motion baseline changed",
    )
    return {
        "schema_version": "human2robot-m5b-p2-step-trend-diagnostic-v1",
        "status": "completed",
        "formal_result": False,
        "acceptance_eligible": False,
        "claim_boundary": (
            f"Paired trend diagnostic on {len(reference_records)} repeatedly observed fixed "
            "held-out queries; "
            "not formal evaluation, statistical evidence, generalization evidence, "
            "or M5B-P2 acceptance evidence."
        ),
        "reference": {
            "iteration": reference_iteration,
            "path": str(reference_path.resolve()),
            "sha256": _sha256(reference_path),
        },
        "candidate": {
            "iteration": candidate_iteration,
            "path": str(candidate_path.resolve()),
            "sha256": _sha256(candidate_path),
        },
        "comparability": {
            "same_method": True,
            "same_run_seed": True,
            "same_binding": True,
            "same_sampler": True,
            "same_selected_queries": True,
            "same_no_motion_targets": True,
            "query_count": len(reference_records),
            "rank_inference_count_per_checkpoint": int(
                candidate["selection"]["selected_rank_example_count"]
            ),
        },
        "metric_comparisons": metric_comparisons,
        "primary_query_comparison": primary_query_comparison,
        "guardrail_delta": guardrail_delta,
        "direction_assessment": {
            "label": (
                "positive_direction_on_fixed_diagnostic"
                if positive
                else "mixed_or_not_positive_on_fixed_diagnostic"
            ),
            "positive": positive,
            "rule": (
                "Primary mean lower; primary improves on a strict majority of paired queries; "
                "all six core metric means non-increasing; no guardrail regression."
            ),
            "primary_metric": PRIMARY_METRIC,
            "primary_improved_query_count": primary["improved_query_count"],
            "primary_query_count": primary["query_count"],
            "nonincreasing_core_metric_count": nonincreasing_core_count,
            "core_metric_count": len(comparable_core),
            "guards_no_regression": guards_no_regression,
            "no_motion_reference": reference_no_motion,
            "reference_better_than_no_motion": primary["reference_mean"] < reference_no_motion,
            "candidate_better_than_no_motion": primary["candidate_mean"] < candidate_no_motion,
        },
        "validity_risks": [
            f"{len(reference_records)} fixed held-out queries and one training seed were observed.",
            "Repeated inspection makes these queries a monitoring probe, not pristine acceptance evidence.",
            "No cross-seed uncertainty, retrieval comparison, or real-robot rollout.",
        ],
        "completed_at_unix": time.time(),
    }


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _report(payload: Mapping[str, Any]) -> str:
    reference_iteration = payload["reference"]["iteration"]
    candidate_iteration = payload["candidate"]["iteration"]
    assessment = payload["direction_assessment"]
    query_count = payload["comparability"]["query_count"]
    verdict = (
        "是，固定 held-out 查询集上呈正向趋势"
        if assessment["positive"]
        else "证据混合，不能判为正向"
    )
    lines = [
        f"# M5B-P2 iter-{reference_iteration} → iter-{candidate_iteration} 配对趋势诊断",
        "",
        f"> 非正式、不可用于验收；{query_count} 个重复观测的固定 held-out 查询，"
        "仅用于训练趋势监控。",
        "",
        "## 结论",
        "",
        f"**{verdict}。**",
        "",
        "## 总体指标（越低越好）",
        "",
        f"| 指标 | iter-{reference_iteration} | iter-{candidate_iteration} | 相对变化 | 改善查询 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in CORE_METRICS:
        comparison = payload["metric_comparisons"][metric]
        relative = comparison["relative_change_fraction"]
        relative_text = "n/a" if relative is None else f"{relative:+.2%}"
        lines.append(
            f"| `{metric}` | {_fmt(comparison['reference_mean'])} | "
            f"{_fmt(comparison['candidate_mean'])} | {relative_text} | "
            f"{comparison['improved_query_count']}/{comparison['query_count']} |"
        )
    lines.extend(
        [
            "",
            f"## 逐查询 `{PRIMARY_METRIC}`",
            "",
            f"| 任务 | iter-{reference_iteration} | iter-{candidate_iteration} | 相对变化 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["primary_query_comparison"]:
        relative = row["relative_change_fraction"]
        relative_text = "n/a" if relative is None else f"{relative:+.2%}"
        lines.append(
            f"| `{row['task']}` | {_fmt(row['reference'])} | "
            f"{_fmt(row['candidate'])} | {relative_text} |"
        )
    guardrails = ", ".join(
        f"`{name}`={delta:+d}" for name, delta in payload["guardrail_delta"].items()
    )
    lines.extend(
        [
            "",
            "## 判定依据",
            "",
            f"- 主指标改善查询：{assessment['primary_improved_query_count']}/"
            f"{assessment['primary_query_count']}",
            f"- 核心指标均值未恶化：{assessment['nonincreasing_core_metric_count']}/"
            f"{assessment['core_metric_count']}",
            f"- no-motion 主指标参考：{_fmt(assessment['no_motion_reference'])}；"
            f"iter-{reference_iteration} 是否更好：{assessment['reference_better_than_no_motion']}；"
            f"iter-{candidate_iteration} 是否更好：{assessment['candidate_better_than_no_motion']}",
            f"- 守卫变化：{guardrails}",
            "",
            "## 证据边界",
            "",
            f"该比较支持“训练在这 {query_count} 个固定 held-out 查询上朝更低误差移动”"
            "的窄结论。"
            "它不支持泛化、统计显著性、优于正式 baseline、真实机器人成功率或验收通过等主张。"
            "由于这些查询已被重复查看，应把它们视为 monitoring probe，不再作为独立的 pristine 验收证据。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compare(args.reference.resolve(), args.candidate.resolve())
    _write_text_atomic(
        args.output.resolve(),
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _write_text_atomic(args.report.resolve(), _report(payload))
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
