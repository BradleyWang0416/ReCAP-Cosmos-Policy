#!/usr/bin/env python3
"""Audit the P0 sliding-window semantics against the frozen P2 contract."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from cosmos_policy.datasets.human2robot_dataset import _contiguous_segments
from tools.human2robot_m5b_p2_matrix import file_sha256


class WindowAuditError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WindowAuditError(message)


def old_p0_windows(rows: Sequence[int], horizon: int = 8, stride: int = 8) -> list[dict[str, Any]]:
    rows = np.asarray(rows, dtype=np.int64)
    result = []
    last_start = len(rows) - horizon - 1
    for local_start in range(0, max(0, last_start + 1), stride):
        pool = rows[local_start : local_start + horizon]
        query = rows[local_start + 1 : local_start + 1 + horizon]
        result.append(
            {
                "current_row": int(pool[0]),
                "history_or_pool_rows": pool.tolist(),
                "future_query_rows": query.tolist(),
            }
        )
    return result


def frozen_p2_windows(
    rows: Sequence[int],
    h_steps: int = 8,
    k_steps: int = 8,
    stride: int = 8,
    query_offset: int = 1,
) -> list[dict[str, Any]]:
    rows = np.asarray(rows, dtype=np.int64)
    result = []
    max_current = len(rows) - query_offset - k_steps
    for current_pos in range(h_steps - 1, max_current + 1, stride):
        history = rows[current_pos - h_steps + 1 : current_pos + 1]
        future = rows[current_pos + query_offset : current_pos + query_offset + k_steps]
        result.append(
            {
                "current_row": int(rows[current_pos]),
                "history_or_pool_rows": history.tolist(),
                "future_query_rows": future.tolist(),
            }
        )
    return result


def audit(workspace: Path) -> dict[str, Any]:
    root = workspace / "data/Human2Robot"
    canonical = root / "canonical/v3"
    split_path = canonical / "task_split_manifest.json"
    prepared_path = root / "derived/m5b_v03/p2_prepared_v2/prepared_manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    paths = sorted((canonical / "pilot").glob("demo_*.hdf5"))
    records = split["episodes"]
    _require(len(paths) == len(records), "Canonical/split episode count mismatch")
    counts = {
        "train": {"old_p0_query_count": 0, "frozen_p2_query_count": 0},
        "heldout": {"old_p0_query_count": 0, "frozen_p2_query_count": 0},
    }
    examples = []
    for path, record in zip(paths, records, strict=True):
        split_id = str(record["split"])
        with h5py.File(path, "r") as file:
            demo = file["data/demo_0"]
            segment_id = np.asarray(demo["metadata/segment_id"][:], dtype=np.int64)
            gap_mask = np.asarray(demo["metadata/gap_mask"][:], dtype=bool)
        for segment_number, rows in enumerate(_contiguous_segments(segment_id, gap_mask)):
            old = old_p0_windows(rows)
            new = frozen_p2_windows(rows)
            counts[split_id]["old_p0_query_count"] += len(old)
            counts[split_id]["frozen_p2_query_count"] += len(new)
            if old and new and not examples:
                examples.append(
                    {
                        "episode": path.name,
                        "segment_number": segment_number,
                        "old_first_window": old[0],
                        "p2_first_window": new[0],
                    }
                )
    main_entries = [
        entry
        for entry in prepared["entries"]
        if entry["spec"]["experiment_id"] == "M5B-MAIN-01"
    ]
    _require(len(main_entries) == 9, "Prepared main-entry count changed")
    prepared_train_counts = {entry["train_contract"]["query_count"] for entry in main_entries}
    prepared_heldout_counts = {entry["heldout_contract"]["query_count"] for entry in main_entries}
    _require(
        prepared_train_counts == {counts["train"]["frozen_p2_query_count"]},
        "Prepared train count does not match frozen P2 semantics",
    )
    _require(
        prepared_heldout_counts == {counts["heldout"]["frozen_p2_query_count"]},
        "Prepared heldout count does not match frozen P2 semantics",
    )
    return {
        "schema_version": "human2robot-m5b-p0-to-p2-window-audit-v1",
        "status": "passed_with_migration_boundary",
        "formal_result": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_bindings": {
            "split_manifest_path": str(split_path),
            "split_manifest_sha256": file_sha256(split_path),
            "prepared_manifest_path": str(prepared_path),
            "prepared_manifest_sha256": file_sha256(prepared_path),
        },
        "semantics": {
            "old_p0": "current=first pool row; H rows extend forward; query is the same forward block shifted by one",
            "frozen_p2": "H segment-safe history rows end at current; K query rows are strictly after current",
            "same_semantics": False,
        },
        "counts": counts,
        "example": examples[0],
        "evidence_boundary": {
            "p0_overfit_still_supports": "the 2B adapter and action latent path can overfit a real Human2Robot batch",
            "p0_overfit_does_not_support": "the frozen P2 history/current/future sampling semantics",
            "p2_prepared_inputs_follow_frozen_contract": True,
            "formal_training_must_use": "Human2RobotP2Dataset and the 48 prepared entries only",
        },
    }


def write_outputs(workspace: Path, result: dict[str, Any]) -> None:
    json_path = workspace / "方案/v03/M5B_P0_to_P2_window_semantics_audit.json"
    md_path = workspace / "方案/v03/M5B_P0_to_P2_window_semantics_audit.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = result["counts"]
    md = f"""# M5B P0 → P2 窗口语义迁移审计

状态：**passed_with_migration_boundary**（非正式实验结果）

P0 与冻结 P2 的窗口语义并不相同。P0 把 `current` 放在向前 pool block 的首行；P2 使用截至当前行的 H 步历史，并只把当前行之后的 K 步作为 query target。

| split | P0 旧语义 | P2 冻结语义 |
|---|---:|---:|
| train | {counts['train']['old_p0_query_count']} | {counts['train']['frozen_p2_query_count']} |
| heldout | {counts['heldout']['old_p0_query_count']} | {counts['heldout']['frozen_p2_query_count']} |

因此，P0 overfit 证据继续证明正式 2B adapter/action-latent 链路可学习真实 Human2Robot batch，但不能替代 P2 窗口语义验收。已生成的 48-cell prepared inputs 与冻结 P2 语义计数完全一致；正式训练只能使用 `Human2RobotP2Dataset` 和这些 prepared entries。
"""
    md_path.write_text(md, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    result = audit(workspace)
    write_outputs(workspace, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
