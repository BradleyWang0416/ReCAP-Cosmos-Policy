#!/usr/bin/env python3
"""Merge non-formal M5B-P2 step-checkpoint diagnostic shards.

The merged artifact remains explicitly outside the formal acceptance DAG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


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


def _summary(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    names = sorted({name for record in records for name in record[key]})
    return {
        name: {
            "mean": float(np.mean([float(record[key][name]) for record in records])),
            "median": float(np.median([float(record[key][name]) for record in records])),
            "min": float(np.min([float(record[key][name]) for record in records])),
            "max": float(np.max([float(record[key][name]) for record in records])),
        }
        for name in names
    }


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _report(payload: Mapping[str, Any]) -> str:
    iteration = int(payload["checkpoint"]["expected_iteration"])
    step_label = f"step-{iteration}"
    lines = [
        f"# M5B-P2 {step_label} 少量代表性非正式诊断报告",
        "",
        "> 本报告不是验收报告；`formal_result=false`，`acceptance_eligible=false`。",
        "",
        "## 范围",
        "",
        f"- checkpoint：`{payload['checkpoint']['path']}`",
        f"- 方法：`{payload['method_id']}`，seed `{payload['run_seed']}`",
        f"- 数据：held-out {payload['selection']['selected_query_count']} 个查询、"
        f"{payload['selection']['selected_rank_example_count']} 次 rank 推理，"
        f"覆盖 {payload['selection']['task_count']} 个任务",
        "- 选择：每个任务按 episode/current_row/window_id 排序后取中位代表查询，保留 top-3 ranks",
        "- 设备：主机 GPU 4/5，各自单卡、两个独立分片",
        "",
        "## 总体误差（越低越好）",
        "",
        f"| 指标 | {step_label} | no-motion | 相对变化 |",
        "|---|---:|---:|---:|",
    ]
    for name, comparison in payload["prediction_vs_no_motion"].items():
        relative = comparison["relative_change_fraction"]
        relative_text = "n/a" if relative is None else f"{relative:+.2%}"
        lines.append(
            f"| `{name}` | {_fmt(comparison['checkpoint_mean'])} | "
            f"{_fmt(comparison['no_motion_mean'])} | {relative_text} |"
        )
    lines.extend(
        [
            "",
            "## 分任务 canonical_error_median",
            "",
            f"| 任务 | {step_label} | no-motion | 相对变化 |",
            "|---|---:|---:|---:|",
        ]
    )
    for task, summary in payload["task_summary"].items():
        learned = summary["metrics"]["canonical_error_median"]["mean"]
        baseline = summary["diagnostic_no_motion_baseline_metrics"]["canonical_error_median"]["mean"]
        relative = (learned - baseline) / baseline if baseline else float("nan")
        lines.append(f"| `{task}` | {_fmt(learned)} | {_fmt(baseline)} | {relative:+.2%} |")
    guardrail_text = ", ".join(
        f"`{name}`={value}" for name, value in payload["guardrail_totals"].items()
    )
    runtime = payload["runtime"]
    lines.extend(
        [
            "",
            "## 运行与守卫",
            "",
            f"- 12 次推理总耗时：{runtime['inference_seconds']['sum']:.2f} 秒；"
            f"中位数：{runtime['inference_seconds']['median']:.2f} 秒/次；"
            f"首次调用含预热，最大：{runtime['inference_seconds']['max']:.2f} 秒",
            f"- 单分片峰值 allocated：{runtime['max_peak_allocated_bytes'] / 2**30:.2f} GiB；"
            f"reserved：{runtime['max_peak_reserved_bytes'] / 2**30:.2f} GiB",
            f"- 守卫合计：{guardrail_text}",
            "",
            "## 结论边界",
            "",
            f"该结果只说明 {step_label} 权重可以在完整 Docker/CUDA 环境中完成真实前向，"
            "并在这 4 个固定代表查询上产生有限、无 workspace 越界的动作预测。"
            "样本量太小，且只测试了一个训练 seed 和 no-retrieval 方法，因此不能据此判断正式泛化、"
            "方法优越性或 M5B-P2 是否通过验收。",
            "",
            "## 诊断期间发现的正式评估入口问题",
            "",
            "独立诊断适配器需要补齐 Hydra 参数形式、intermediate DCP iteration 识别、"
            "Human2Robot 严格 uint8 校验顺序，以及 BF16 autocast 上下文。"
            "本轮只在非正式脚本中做了隔离修正，未修改正在训练的冻结正式代码；"
            "正式评估前应另行冻结 successor。",
            "",
        ]
    )
    return "\n".join(lines)


def merge(inputs: Sequence[Path]) -> dict[str, Any]:
    _require(len(inputs) == 2, "Expected exactly two diagnostic shards")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    for index, shard in enumerate(shards):
        _require(shard.get("status") == "completed", f"Shard {index} is incomplete")
        _require(shard.get("formal_result") is False, f"Shard {index} is mislabeled formal")
        _require(
            shard.get("acceptance_eligible") is False,
            f"Shard {index} is mislabeled acceptance-eligible",
        )
    _require(shards[0]["checkpoint"] == shards[1]["checkpoint"], "Checkpoint mismatch")
    _require(shards[0]["binding"] == shards[1]["binding"], "Evaluation binding mismatch")
    _require(shards[0]["sampler"] == shards[1]["sampler"], "Sampler mismatch")
    indices = sorted(int(shard["selection"]["shard_index"]) for shard in shards)
    _require(indices == [0, 1], f"Unexpected shard indices: {indices}")
    records = sorted(
        [record for shard in shards for record in shard["window_records"]],
        key=lambda record: (str(record["task"]), str(record["query_id"])),
    )
    receipts = [receipt for shard in shards for receipt in shard["inference_receipts"]]
    _require(len(records) == 4, f"Expected four query records, found {len(records)}")
    _require(len(receipts) == 12, f"Expected twelve inference receipts, found {len(receipts)}")
    learned = _summary(records, "metrics")
    baseline = _summary(records, "diagnostic_no_motion_baseline_metrics")
    comparison = {}
    for name in sorted(set(learned) & set(baseline)):
        checkpoint_mean = learned[name]["mean"]
        no_motion_mean = baseline[name]["mean"]
        comparison[name] = {
            "checkpoint_mean": checkpoint_mean,
            "no_motion_mean": no_motion_mean,
            "absolute_delta": checkpoint_mean - no_motion_mean,
            "relative_change_fraction": (
                (checkpoint_mean - no_motion_mean) / no_motion_mean
                if no_motion_mean != 0
                else None
            ),
            "lower_is_better": True,
            "checkpoint_better_on_this_fixed_sample": checkpoint_mean < no_motion_mean,
        }
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[str(record["task"])].append(record)
    inference_seconds = np.asarray(
        [float(receipt["inference_seconds"]) for receipt in receipts], dtype=np.float64
    )
    binding = shards[0]["binding"]
    return {
        "schema_version": "human2robot-m5b-p2-intermediate-diagnostic-merged-v1",
        "status": "completed",
        "formal_result": False,
        "acceptance_eligible": False,
        "claim_boundary": shards[0]["claim_boundary"],
        "checkpoint": shards[0]["checkpoint"],
        "method_id": binding["method_id"],
        "run_seed": binding["run_seed"],
        "binding": binding,
        "sampler": shards[0]["sampler"],
        "selection": {
            "split": "heldout",
            "rule": "one sorted-median representative query per task; all top-3 ranks",
            "task_count": len(by_task),
            "tasks": sorted(by_task),
            "selected_query_count": len(records),
            "selected_rank_example_count": len(receipts),
            "query_provenance": [
                item
                for shard in shards
                for item in shard["selection"]["query_provenance"]
            ],
        },
        "metric_summary": learned,
        "diagnostic_no_motion_baseline_summary": baseline,
        "prediction_vs_no_motion": comparison,
        "task_summary": {
            task: {
                "query_count": len(task_records),
                "metrics": _summary(task_records, "metrics"),
                "diagnostic_no_motion_baseline_metrics": _summary(
                    task_records, "diagnostic_no_motion_baseline_metrics"
                ),
            }
            for task, task_records in sorted(by_task.items())
        },
        "window_records": records,
        "guardrail_totals": {
            name: int(sum(int(shard["guardrail_totals"][name]) for shard in shards))
            for name in sorted(shards[0]["guardrail_totals"])
        },
        "runtime": {
            "devices": [
                {
                    "cuda_visible_devices": shard["runtime"]["cuda_visible_devices"],
                    "gpu_name": shard["runtime"]["gpu_name"],
                    "gpu_total_memory_bytes": shard["runtime"]["gpu_total_memory_bytes"],
                }
                for shard in shards
            ],
            "backend_load_seconds": [
                float(shard["runtime"]["backend_load_seconds"]) for shard in shards
            ],
            "shard_total_wall_seconds": [
                float(shard["runtime"]["total_wall_seconds"]) for shard in shards
            ],
            "inference_seconds": {
                "count": int(len(inference_seconds)),
                "sum": float(inference_seconds.sum()),
                "mean": float(inference_seconds.mean()),
                "median": float(np.median(inference_seconds)),
                "min": float(inference_seconds.min()),
                "max": float(inference_seconds.max()),
                "p95": float(np.percentile(inference_seconds, 95)),
            },
            "max_peak_allocated_bytes": max(
                int(shard["runtime"]["peak_allocated_bytes"]) for shard in shards
            ),
            "max_peak_reserved_bytes": max(
                int(shard["runtime"]["peak_reserved_bytes"]) for shard in shards
            ),
        },
        "source_shards": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in inputs
        ],
        "completed_at_unix": time.time(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = merge([path.resolve() for path in args.inputs])
    _write_text_atomic(
        args.output.resolve(),
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _write_text_atomic(args.report.resolve(), _report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "formal_result": payload["formal_result"],
                "acceptance_eligible": payload["acceptance_eligible"],
                "output": str(args.output.resolve()),
                "report": str(args.report.resolve()),
                "metric_summary": payload["metric_summary"],
                "prediction_vs_no_motion": payload["prediction_vs_no_motion"],
                "guardrail_totals": payload["guardrail_totals"],
                "runtime": payload["runtime"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
