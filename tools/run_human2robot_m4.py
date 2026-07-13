#!/usr/bin/env python3
"""Launch the M4-v03 offline Human2Robot paired bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from human2robot_m4 import (
    DEFAULT_CANONICAL_ROOT,
    DEFAULT_M3_REPORT,
    DEFAULT_MAIN_VIEW,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REPORT_ROOT,
    M4Config,
    M4Error,
    run_m4_launch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--main-view", type=Path, default=DEFAULT_MAIN_VIEW)
    parser.add_argument("--m3-report", type=Path, default=DEFAULT_M3_REPORT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=1e-2)
    parser.add_argument("--random-seed", type=int, default=20260711)
    parser.add_argument("--pool-growth", type=int, nargs="+", default=[1, 2, 4, 8, 0])
    parser.add_argument("--expected-episodes", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = M4Config(
        canonical_root=args.canonical_root,
        main_view_path=args.main_view,
        m3_report_path=args.m3_report,
        output_root=args.output_root,
        report_root=args.report_root,
        horizon=args.horizon,
        window_stride=args.window_stride,
        retrieval_top_k=args.retrieval_top_k,
        ridge_alpha=args.ridge_alpha,
        random_seed=args.random_seed,
        pool_growth_sizes=tuple(args.pool_growth),
        expected_episode_count=args.expected_episodes,
    )
    try:
        report = run_m4_launch(config)
    except (M4Error, OSError, ValueError, KeyError, np.linalg.LinAlgError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "launched" else 2


if __name__ == "__main__":
    raise SystemExit(main())
