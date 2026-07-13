#!/usr/bin/env python3
"""Validate one file or a split directory of Human2Robot canonical HDF5 episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from human2robot_m2 import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_ROOT,
    M2Error,
    ValidationLimits,
    canonical_files,
    to_jsonable,
    validate_canonical_dataset,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT_ROOT / "pilot")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-episodes", type=int, default=1)
    parser.add_argument("--minimum-tasks", type=int, default=1)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--workspace-min", type=float, nargs=3, default=(-1.0, -1.0, -0.25))
    parser.add_argument("--workspace-max", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        files = canonical_files(args.input)
        if not files:
            raise M2Error(f"No canonical demo_*.hdf5 files found at {args.input}")
        limits = ValidationLimits(
            workspace_min_m=tuple(args.workspace_min),
            workspace_max_m=tuple(args.workspace_max),
        )
        report = validate_canonical_dataset(
            files,
            limits,
            minimum_episodes=args.minimum_episodes,
            minimum_tasks=args.minimum_tasks,
            source_root=args.source_root,
            split_manifest_path=args.split_manifest,
        )
        if args.output:
            write_json(args.output, report)
    except (M2Error, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False, default=to_jsonable))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
