#!/usr/bin/env python3
"""Run Human2Robot M2-v03 semantic-safe native conversion and acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from human2robot_m2 import (
    DEFAULT_EVIDENCE_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PARENT_V2_SPLIT,
    DEFAULT_REPORT_ROOT,
    DEFAULT_SELECTION_MANIFEST,
    DEFAULT_SOURCE_ROOT,
    LEGACY_FIXED_STRIDE3,
    PRESERVE_NATIVE,
    ConversionConfig,
    M2Error,
    ValidationLimits,
    convert_dataset,
    run_m2_pipeline,
    to_jsonable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--pilot-subdir", default="pilot")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--timebase-policy",
        choices=(PRESERVE_NATIVE, LEGACY_FIXED_STRIDE3),
        default=PRESERVE_NATIVE,
        help="Default preserves all source frames; the fixed-stride option is withdrawn legacy behavior.",
    )
    parser.add_argument("--selection-manifest", type=Path, default=DEFAULT_SELECTION_MANIFEST)
    parser.add_argument("--evidence-manifest", type=Path, default=DEFAULT_EVIDENCE_MANIFEST)
    parser.add_argument("--parent-v2-split-manifest", type=Path, default=DEFAULT_PARENT_V2_SPLIT)
    parser.add_argument("--heldout-task-count", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=20260711)
    parser.add_argument("--position-scale", type=float, default=0.001)
    parser.add_argument("--euler-order", default="xyz")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--visualizations", type=int, default=10)
    parser.add_argument("--visualization-seed", type=int, default=20260711)
    parser.add_argument("--visualization-playback-fps", type=float, default=10.0)
    parser.add_argument("--workspace-min", type=float, nargs=3, default=(-1.0, -1.0, -0.25))
    parser.add_argument("--workspace-max", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ConversionConfig(
        source_root=args.source_root,
        output_root=args.output_root,
        report_root=args.report_root,
        pilot_subdir=args.pilot_subdir,
        episode_count=args.episodes,
        timebase_policy=args.timebase_policy,
        selection_manifest=args.selection_manifest,
        evidence_manifest=args.evidence_manifest,
        parent_v2_split_manifest=args.parent_v2_split_manifest,
        heldout_task_count=args.heldout_task_count,
        split_seed=args.split_seed,
        position_scale=args.position_scale,
        euler_order=args.euler_order,
        overwrite=args.overwrite,
    )
    limits = ValidationLimits(
        workspace_min_m=tuple(args.workspace_min),
        workspace_max_m=tuple(args.workspace_max),
    )
    try:
        if args.timebase_policy == LEGACY_FIXED_STRIDE3:
            manifest = convert_dataset(config)
            print(
                json.dumps(
                    {
                        "status": "legacy_generated_not_m2_v03_accepted",
                        "warning": "This explicit policy reproduces the withdrawn 30->10 fixed-stride view only.",
                        "manifest": manifest,
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=to_jsonable,
                )
            )
            return 0
        report = run_m2_pipeline(
            config,
            limits,
            visualization_count=args.visualizations,
            visualization_seed=args.visualization_seed,
            visualization_playback_fps=args.visualization_playback_fps,
        )
    except (M2Error, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False, default=to_jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
