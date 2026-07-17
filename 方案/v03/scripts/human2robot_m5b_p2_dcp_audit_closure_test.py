from __future__ import annotations

import shutil
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemWriter, save

from human2robot_m5b_p2_dcp_audit_closure import (
    DcpAuditClosureError,
    validate_dcp_checkpoint,
)


def _write_single_owner_component(path: Path, *, empty_non_owner_ranks: bool) -> None:
    save(
        {"value": torch.arange(4)},
        storage_writer=FileSystemWriter(path),
    )
    owner = path / "__0_0.distcp"
    for rank in range(1, 4):
        target = path / f"__{rank}_0.distcp"
        if empty_non_owner_ranks:
            target.touch()
        else:
            shutil.copyfile(owner, target)


def test_valid_dcp_allows_unreferenced_empty_scheduler_and_trainer_rank_files(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "iter_000007000"
    _write_single_owner_component(checkpoint / "model", empty_non_owner_ranks=False)
    _write_single_owner_component(checkpoint / "optim", empty_non_owner_ranks=False)
    _write_single_owner_component(checkpoint / "scheduler", empty_non_owner_ranks=True)
    _write_single_owner_component(checkpoint / "trainer", empty_non_owner_ranks=True)

    evidence = validate_dcp_checkpoint(checkpoint, expected_world_size=4)

    assert evidence["components"]["scheduler"]["rank_file_count"] == 4
    assert evidence["components"]["scheduler"]["referenced_rank_files"] == [
        "__0_0.distcp"
    ]
    assert evidence["components"]["trainer"]["referenced_rank_files"] == [
        "__0_0.distcp"
    ]


def test_dcp_rejects_a_referenced_payload_truncated_below_metadata_range(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "iter_000007000"
    for component in ("model", "optim", "scheduler", "trainer"):
        _write_single_owner_component(
            checkpoint / component,
            empty_non_owner_ranks=component in {"scheduler", "trainer"},
        )
    (checkpoint / "scheduler/__0_0.distcp").write_bytes(b"")

    try:
        validate_dcp_checkpoint(checkpoint, expected_world_size=4)
    except DcpAuditClosureError as error:
        assert "range exceeds shard size" in str(error)
    else:
        raise AssertionError("truncated referenced DCP payload was accepted")
