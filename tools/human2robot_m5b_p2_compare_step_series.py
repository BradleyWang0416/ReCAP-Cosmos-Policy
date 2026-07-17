#!/usr/bin/env python3
"""Compare a series of non-formal M5B-P2 checkpoint diagnostics."""

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


def _record_map(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = {str(record["query_id"]): record for record in payload["window_records"]}
    _require(len(records) == len(payload["window_records"]), "Duplicate query_id")
    return records


def _relative(current: float, previous: float) -> float | None:
    return (current - previous) / previous if previous != 0 else None


def compare_series(paths: Sequence[Path]) -> dict[str, Any]:
    _require(len(paths) >= 3, "At least three checkpoint summaries are required")
    entries = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _require(payload.get("status") == "completed", f"Incomplete summary: {path}")
        _require(payload.get("formal_result") is False, f"Formal summary not allowed: {path}")
        _require(payload.get("acceptance_eligible") is False, f"Acceptance summary not allowed: {path}")
        entries.append(
            {
                "iteration": int(payload["checkpoint"]["expected_iteration"]),
                "path": path,
                "sha256": _sha256(path),
                "payload": payload,
                "records": _record_map(payload),
            }
        )
    entries.sort(key=lambda entry: entry["iteration"])
    iterations = [entry["iteration"] for entry in entries]
    _require(len(set(iterations)) == len(iterations), "Duplicate iterations")
    anchor = entries[0]["payload"]
    anchor_selection = _selection_signature(anchor)
    anchor_queries = set(entries[0]["records"])
    for entry in entries[1:]:
        payload = entry["payload"]
        _require(payload["method_id"] == anchor["method_id"], "Method mismatch")
        _require(payload["run_seed"] == anchor["run_seed"], "Run seed mismatch")
        _require(payload["binding"] == anchor["binding"], "Binding mismatch")
        _require(payload["sampler"] == anchor["sampler"], "Sampler mismatch")
        _require(_selection_signature(payload) == anchor_selection, "Selection mismatch")
        _require(set(entry["records"]) == anchor_queries, "Query set mismatch")
    query_ids = sorted(anchor_queries, key=lambda q: str(entries[0]["records"][q]["task"]))
    no_motion = float(anchor["diagnostic_no_motion_baseline_summary"][PRIMARY_METRIC]["mean"])
    for entry in entries[1:]:
        candidate_no_motion = float(
            entry["payload"]["diagnostic_no_motion_baseline_summary"][PRIMARY_METRIC]["mean"]
        )
        _require(
            np.isclose(candidate_no_motion, no_motion, rtol=0.0, atol=1e-12),
            "No-motion aggregate changed",
        )
        for query_id in query_ids:
            reference_baseline = entries[0]["records"][query_id][
                "diagnostic_no_motion_baseline_metrics"
            ]
            candidate_baseline = entry["records"][query_id][
                "diagnostic_no_motion_baseline_metrics"
            ]
            _require(reference_baseline.keys() == candidate_baseline.keys(), "Baseline keys changed")
            for metric in reference_baseline:
                _require(
                    np.isclose(
                        float(reference_baseline[metric]),
                        float(candidate_baseline[metric]),
                        rtol=0.0,
                        atol=1e-12,
                    ),
                    f"No-motion target changed: {query_id} {metric}",
                )
    metric_series: dict[str, Any] = {}
    for metric in CORE_METRICS:
        values = [float(entry["payload"]["metric_summary"][metric]["mean"]) for entry in entries]
        metric_series[metric] = {
            "values": [
                {"iteration": iteration, "mean": value}
                for iteration, value in zip(iterations, values, strict=True)
            ],
            "segment_relative_changes": [
                {
                    "from_iteration": iterations[index - 1],
                    "to_iteration": iterations[index],
                    "relative_change_fraction": _relative(values[index], values[index - 1]),
                    "improved_or_equal": values[index] <= values[index - 1],
                }
                for index in range(1, len(values))
            ],
            "best_iteration": iterations[int(np.argmin(values))],
            "latest_vs_first_relative_change_fraction": _relative(values[-1], values[0]),
            "latest_vs_best_relative_change_fraction": _relative(min(values[-1:]), min(values)),
            "monotonic_nonincreasing": all(
                values[index] <= values[index - 1] for index in range(1, len(values))
            ),
        }
    primary_query_series = []
    for query_id in query_ids:
        values = [float(entry["records"][query_id]["metrics"][PRIMARY_METRIC]) for entry in entries]
        primary_query_series.append(
            {
                "task": str(entries[0]["records"][query_id]["task"]),
                "query_id": query_id,
                "values": [
                    {"iteration": iteration, "value": value}
                    for iteration, value in zip(iterations, values, strict=True)
                ],
                "best_iteration": iterations[int(np.argmin(values))],
                "latest_vs_previous_relative_change_fraction": _relative(values[-1], values[-2]),
                "latest_vs_first_relative_change_fraction": _relative(values[-1], values[0]),
                "latest_improved_vs_previous": values[-1] < values[-2],
                "latest_improved_vs_first": values[-1] < values[0],
            }
        )
    primary_values = [item["mean"] for item in metric_series[PRIMARY_METRIC]["values"]]
    best_index = int(np.argmin(primary_values))
    best_iteration = iterations[best_index]
    latest_better_than_first = primary_values[-1] < primary_values[0]
    latest_worse_than_best = primary_values[-1] > primary_values[best_index]
    latest_segment_primary_improved_count = sum(
        row["latest_improved_vs_previous"] for row in primary_query_series
    )
    latest_core_nonincreasing_count = sum(
        series["values"][-1]["mean"] <= series["values"][-2]["mean"]
        for series in metric_series.values()
    )
    all_guardrails_zero = all(
        all(int(value) == 0 for value in entry["payload"]["guardrail_totals"].values())
        for entry in entries
    )
    if all(series["monotonic_nonincreasing"] for series in metric_series.values()):
        label = "monotonic_positive_on_fixed_diagnostic"
    elif latest_better_than_first and latest_worse_than_best:
        label = "improved_vs_early_but_regressed_since_best"
    else:
        label = "mixed_or_not_positive_on_fixed_diagnostic"
    return {
        "schema_version": "human2robot-m5b-p2-step-series-diagnostic-v1",
        "status": "completed",
        "formal_result": False,
        "acceptance_eligible": False,
        "claim_boundary": (
            f"Multi-checkpoint trend diagnostic on {len(query_ids)} repeatedly observed fixed "
            "held-out queries; "
            "not formal evaluation, statistical evidence, generalization evidence, checkpoint "
            "selection evidence, or M5B-P2 acceptance evidence."
        ),
        "method_id": anchor["method_id"],
        "run_seed": anchor["run_seed"],
        "comparability": {
            "same_binding": True,
            "same_sampler": True,
            "same_selected_queries": True,
            "same_no_motion_targets": True,
            "query_count": len(query_ids),
            "rank_inference_count_per_checkpoint": int(
                anchor["selection"]["selected_rank_example_count"]
            ),
        },
        "checkpoints": [
            {
                "iteration": entry["iteration"],
                "path": str(entry["path"].resolve()),
                "sha256": entry["sha256"],
            }
            for entry in entries
        ],
        "no_motion_primary_reference": no_motion,
        "metric_series": metric_series,
        "primary_query_series": primary_query_series,
        "direction_assessment": {
            "label": label,
            "monotonic_positive": label == "monotonic_positive_on_fixed_diagnostic",
            "best_observed_iteration_on_probe": best_iteration,
            "latest_iteration": iterations[-1],
            "latest_better_than_first": latest_better_than_first,
            "latest_worse_than_best": latest_worse_than_best,
            "latest_better_than_no_motion": primary_values[-1] < no_motion,
            "best_better_than_no_motion": primary_values[best_index] < no_motion,
            "latest_segment_primary_improved_query_count": latest_segment_primary_improved_count,
            "primary_query_count": len(query_ids),
            "latest_segment_nonincreasing_core_metric_count": latest_core_nonincreasing_count,
            "core_metric_count": len(CORE_METRICS),
            "all_guardrails_zero": all_guardrails_zero,
        },
        "validity_risks": [
            f"{len(query_ids)} fixed held-out queries and one training seed were observed.",
            "Repeated inspection makes these queries a monitoring probe, not pristine acceptance evidence.",
            "Only checkpoint snapshots were observed; behavior between snapshots remains unobserved.",
            "No cross-seed uncertainty estimate, retrieval comparison, or real-robot rollout.",
        ],
        "completed_at_unix": time.time(),
    }


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _relative_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


def _report(payload: Mapping[str, Any]) -> str:
    iterations = [checkpoint["iteration"] for checkpoint in payload["checkpoints"]]
    assessment = payload["direction_assessment"]
    first_iteration = iterations[0]
    previous_iteration = iterations[-2]
    latest_iteration = iterations[-1]
    best_iteration = assessment["best_observed_iteration_on_probe"]
    iteration_path = " → ".join(f"iter-{iteration}" for iteration in iterations)
    headers = " | ".join(f"iter-{iteration}" for iteration in iterations)
    if assessment["monotonic_positive"]:
        verdict = "固定 probe 上呈单调正向趋势。"
    elif assessment["latest_better_than_first"] and assessment["latest_worse_than_best"]:
        verdict = (
            f"相对 iter-{first_iteration} 仍有净改善，但 iter-{latest_iteration} "
            f"未保持 iter-{best_iteration} 的最佳表现，因此不能判定训练持续向好。"
        )
    else:
        verdict = "固定 probe 上的证据混合，不能判定训练持续向好。"
    lines = [
        f"# M5B-P2 {iteration_path} 固定 probe 轨迹诊断",
        "",
        "> 非正式、不可用于验收或正式 checkpoint 选择。",
        "",
        "## 结论",
        "",
        f"**{verdict}**",
        "",
        "## 总体均值（越低越好）",
        "",
        f"| 指标 | {headers} | 最佳 checkpoint |",
        "|---|" + "---:|" * len(iterations) + "---:|",
    ]
    for metric in CORE_METRICS:
        series = payload["metric_series"][metric]
        values = " | ".join(_fmt(item["mean"]) for item in series["values"])
        lines.append(f"| `{metric}` | {values} | iter-{series['best_iteration']} |")
    lines.extend(
        [
            "",
            f"## 主指标 `{PRIMARY_METRIC}` 阶段变化",
            "",
            "| 区间 | 相对变化 | 是否改善 |",
            "|---|---:|---:|",
        ]
    )
    for segment in payload["metric_series"][PRIMARY_METRIC]["segment_relative_changes"]:
        lines.append(
            f"| iter-{segment['from_iteration']}→iter-{segment['to_iteration']} | "
            f"{_relative_text(segment['relative_change_fraction'])} | "
            f"{segment['improved_or_equal']} |"
        )
    lines.extend(
        [
            "",
            f"## 逐查询 `{PRIMARY_METRIC}`",
            "",
            f"| 任务 | {headers} | {previous_iteration}→{latest_iteration} |",
            "|---|" + "---:|" * len(iterations) + "---:|",
        ]
    )
    for row in payload["primary_query_series"]:
        values = " | ".join(_fmt(item["value"]) for item in row["values"])
        lines.append(
            f"| `{row['task']}` | {values} | "
            f"{_relative_text(row['latest_vs_previous_relative_change_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## 方向判定",
            "",
            f"- 当前标签：`{assessment['label']}`",
            f"- 固定 probe 上观察到的最佳点：iter-{assessment['best_observed_iteration_on_probe']}",
            f"- iter-{latest_iteration} 相对 iter-{previous_iteration} 改善的主指标查询："
            f"{assessment['latest_segment_primary_improved_query_count']}/"
            f"{assessment['primary_query_count']}",
            f"- iter-{latest_iteration} 相对 iter-{previous_iteration} 未恶化的核心指标均值："
            f"{assessment['latest_segment_nonincreasing_core_metric_count']}/"
            f"{assessment['core_metric_count']}",
            f"- no-motion 主指标参考：{_fmt(payload['no_motion_primary_reference'])}；"
            f"iter-{latest_iteration} 是否更好：{assessment['latest_better_than_no_motion']}",
            f"- {len(iterations)} 个 checkpoint 守卫是否全为 0：{assessment['all_guardrails_zero']}",
            "",
            "## 解释与边界",
            "",
            f"这组结果支持“模型较 iter-{first_iteration} 已学习到有效信号”，但不自动支持“后续训练持续改善”。"
            "局部回退可能来自 checkpoint 波动、学习率阶段、固定 probe 过拟合/欠拟合差异或小样本噪声；"
            f"仅凭这 {assessment['primary_query_count']} 个被重复观察的 held-out 查询和单一训练 seed，"
            f"不能判定整体训练失败，也不能据此正式选择 iter-{best_iteration}。"
            "应保持正式训练协议不变，并用额外 checkpoint probe 与最终完整 held-out 评估确认。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compare_series([path.resolve() for path in args.inputs])
    _write_text_atomic(
        args.output.resolve(),
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _write_text_atomic(args.report.resolve(), _report(payload))
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
