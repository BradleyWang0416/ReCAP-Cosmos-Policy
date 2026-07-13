#!/usr/bin/env python3
"""Launch M5-A-v03 Human2Robot data/contract stress tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from human2robot_m5a import (
    DEFAULT_CANONICAL_ROOT,
    DEFAULT_M3_REPORT,
    DEFAULT_M4_CONFIG,
    DEFAULT_M4_REPORT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REPORT_ROOT,
    M5AConfig,
    M5AError,
    run_m5a_launch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--m3-report", type=Path, default=DEFAULT_M3_REPORT)
    parser.add_argument("--m4-report", type=Path, default=DEFAULT_M4_REPORT)
    parser.add_argument("--m4-config", type=Path, default=DEFAULT_M4_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--wrong-lag", type=int, default=30)
    parser.add_argument("--scale-perturbation", type=float, default=2.0)
    parser.add_argument("--frame-drop-every", type=int, default=10)
    parser.add_argument("--timestamp-jitter-std-seconds", type=float, default=0.008)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument("--step-jump", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260711)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = M5AConfig(
        canonical_root=args.canonical_root,
        m3_report_path=args.m3_report,
        m4_report_path=args.m4_report,
        m4_config_path=args.m4_config,
        output_root=args.output_root,
        report_root=args.report_root,
        expected_episode_count=args.expected_episodes,
        wrong_lag=args.wrong_lag,
        scale_perturbation=args.scale_perturbation,
        frame_drop_every=args.frame_drop_every,
        timestamp_jitter_std_seconds=args.timestamp_jitter_std_seconds,
        pause_seconds=args.pause_seconds,
        step_jump=args.step_jump,
        random_seed=args.random_seed,
    )
    try:
        report = run_m5a_launch(config)
    except (M5AError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "launched" else 2


if __name__ == "__main__":
    raise SystemExit(main())
