#!/usr/bin/env python3
"""Close one completed M5B-P2 training cell with metadata-aware DCP audit.

PyTorch DCP may create empty rank files for replicated state that is owned by
one rank.  Integrity is therefore established from the DCP metadata storage
references and their byte ranges, not from a blanket non-empty-file rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from torch.distributed.checkpoint import FileSystemReader

import tools.human2robot_m5b_p2 as p2


COMPONENTS = ("model", "optim", "scheduler", "trainer")
DEFAULT_CELL_ID = (
    "learned_training_checkpoint__M5B-MAIN-01__frozen_main__"
    "no_retrieval__seed20260711"
)


class DcpAuditClosureError(p2.P2Error):
    """Raised when a checkpoint fails the metadata-aware closure contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DcpAuditClosureError(message)


def validate_dcp_checkpoint(path: Path, expected_world_size: int) -> dict[str, Any]:
    """Validate every metadata storage reference without rejecting empty placeholders."""

    _require(path.is_dir(), f"Checkpoint directory missing: {path}")
    components: dict[str, Any] = {}
    for component in COMPONENTS:
        component_path = path / component
        _require(component_path.is_dir(), f"Checkpoint component missing: {component_path}")
        rank_files = sorted(component_path.glob("__*_0.distcp"))
        _require(
            len(rank_files) == expected_world_size,
            f"{component_path} has {len(rank_files)} rank files, expected {expected_world_size}",
        )
        rank_files_by_name = {item.name: item for item in rank_files}
        metadata_path = component_path / ".metadata"
        _require(
            metadata_path.is_file() and metadata_path.stat().st_size > 0,
            f"Missing DCP metadata: {metadata_path}",
        )
        try:
            metadata = FileSystemReader(component_path).read_metadata()
        except Exception as error:
            raise DcpAuditClosureError(
                f"Unreadable DCP metadata in {component_path}: {type(error).__name__}: {error}"
            ) from error
        storage_data = metadata.storage_data
        _require(
            isinstance(storage_data, dict) and bool(storage_data),
            f"DCP metadata has no storage entries: {component_path}",
        )
        referenced_names: set[str] = set()
        referenced_bytes = 0
        for storage_info in storage_data.values():
            relative_path = Path(storage_info.relative_path)
            _require(
                len(relative_path.parts) == 1 and relative_path.name in rank_files_by_name,
                f"DCP metadata references an unexpected shard: {storage_info.relative_path}",
            )
            offset = int(storage_info.offset)
            length = int(storage_info.length)
            _require(offset >= 0 and length > 0, f"Invalid DCP byte range in {component_path}")
            shard = rank_files_by_name[relative_path.name]
            _require(
                shard.stat().st_size >= offset + length,
                f"DCP metadata range exceeds shard size: {shard}",
            )
            referenced_names.add(relative_path.name)
            referenced_bytes += length
        components[component] = {
            "rank_file_count": len(rank_files),
            "metadata_size_bytes": metadata_path.stat().st_size,
            "metadata_storage_entry_count": len(storage_data),
            "payload_size_bytes": sum(item.stat().st_size for item in rank_files),
            "referenced_payload_bytes": referenced_bytes,
            "referenced_rank_files": sorted(referenced_names),
            "empty_unreferenced_rank_files": sorted(
                item.name
                for item in rank_files
                if item.stat().st_size == 0 and item.name not in referenced_names
            ),
        }
    return {"path": str(path), "components": components}


def _sync_cell_mirror(master: dict[str, Any], record: dict[str, Any]) -> None:
    """Keep the compatibility training-cell mirror byte-equivalent after closure."""

    for mirror in master.get("learned_training_cells", []):
        if mirror.get("cell_id") == record["cell_id"]:
            mirror.clear()
            mirror.update(json.loads(json.dumps(record)))
            return
    raise DcpAuditClosureError(f"Learned-cell mirror missing: {record['cell_id']}")


def close_cell(workspace: Path, cell_id: str) -> dict[str, Any]:
    p2.require_full_docker_environment()
    manifest_path, master = p2.load_master(workspace)
    cells_by_id = {cell.cell_id: cell for cell in p2.main_training_cells()}
    _require(cell_id in cells_by_id, f"Unknown learned training cell: {cell_id}")
    cell = cells_by_id[cell_id]
    record = p2.find_cell_record(master, cell_id)
    code_sha256 = record["bindings"]["code_sha256"]

    original_validator = p2.validate_dcp_checkpoint
    p2.validate_dcp_checkpoint = validate_dcp_checkpoint
    try:
        evidence = p2.audit_completed_cell(record, cell, code_sha256)
    finally:
        p2.validate_dcp_checkpoint = original_validator

    record.pop("failure", None)
    record.pop("audit_failure", None)
    record.update(evidence)
    _sync_cell_mirror(master, record)
    p2.update_master_acceptance(master)
    p2.write_json_atomic(manifest_path, master)

    artifact_path = Path(record["registry_artifact_path"])
    receipt = {
        "schema_version": "human2robot-m5b-p2-dcp-audit-closure-receipt-v1",
        "status": "completed",
        "formal_result": False,
        "cell_id": cell_id,
        "cell_status": record["status"],
        "cell_formal_result": record["formal_result"],
        "source_code_sha256": code_sha256,
        "validator_path": str(Path(__file__).resolve()),
        "validator_sha256": p2.file_sha256(Path(__file__).resolve()),
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": p2.file_sha256(manifest_path),
        "registry_artifact_path": str(artifact_path),
        "registry_artifact_sha256": p2.file_sha256(artifact_path),
        "primary_checkpoint_path": record["run_directory"]
        + "/checkpoints/iter_000007000",
        "primary_checkpoint_payload_sha256": record[
            "primary_checkpoint_payload_sha256"
        ],
        "saved_steps_validated": record["saved_steps_validated"],
        "no_imputation": True,
    }
    receipt_path = Path(master["output_root"]) / "dcp_audit_closure_receipt_v1.json"
    p2.write_json_atomic(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--cell-id", default=DEFAULT_CELL_ID)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = close_cell(args.workspace.resolve(), args.cell_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
