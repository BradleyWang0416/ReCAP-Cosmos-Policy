#!/usr/bin/env python3
"""Run Human2Robot M3-v03 action/time/residual acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from human2robot_m3 import (
    DEFAULT_CANONICAL_ROOT,
    DEFAULT_DERIVED_ROOT,
    DEFAULT_EVIDENCE_MANIFEST,
    DEFAULT_REPORT_ROOT,
    M3Config,
    M3Error,
    _json_default,
    run_m3_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--evidence-manifest", type=Path, default=DEFAULT_EVIDENCE_MANIFEST)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-lag", type=int, default=30)
    parser.add_argument("--phase-bins", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=20260711)
    parser.add_argument("--expected-episodes", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = M3Config(
        canonical_root=args.canonical_root,
        derived_root=args.derived_root,
        report_root=args.report_root,
        evidence_manifest=args.evidence_manifest,
        horizon=args.horizon,
        window_stride=args.window_stride,
        top_k=args.top_k,
        max_lag=args.max_lag,
        phase_bins=args.phase_bins,
        random_seed=args.random_seed,
        expected_episode_count=args.expected_episodes,
    )
    try:
        report = run_m3_pipeline(config)
    except (M3Error, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
