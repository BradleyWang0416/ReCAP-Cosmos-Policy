#!/usr/bin/env python3
"""One rank of the four-GPU stage-4 WAN feature materializer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from tools import human2robot_v04_stage4 as stage4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    stage4.require(world_size == 4 and 0 <= rank < 4 and 0 <= local_rank < 4, "WAN worker requires torchrun world size 4")
    torch.cuda.set_device(local_rank)
    manifest = stage4.read_json(args.workspace / "data/Human2Robot/derived/v04/source_split_manifest.json")
    grouped = stage4._records_by_partition(manifest)
    result = stage4.materialize_visual_cache(
        grouped,
        source_root=args.source_root,
        feature_root=args.feature_root,
        split_sha256=str(manifest["split_sha256"]),
        batch_size=args.batch_size,
        worker_rank=rank,
        worker_world_size=world_size,
        write_index=False,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except stage4.Stage4Error as error:
        print(f"stage-4 feature worker error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
