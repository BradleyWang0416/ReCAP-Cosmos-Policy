from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from cosmos_policy.datasets.human2robot_p2_specs import p2_training_specs
from tools import human2robot_m5b_p2_prepare as prepare
from tools.human2robot_m5b_p2_prepare import (
    INDEX_SCHEMA_VERSION,
    compute_training_statistics,
    learned_registry_cell_ids,
    make_index_manifest,
    placeholder_statistics,
)

ROOT = Path(__file__).resolve().parents[1]


def test_placeholder_is_explicitly_nonformal_and_train_only() -> None:
    payload = placeholder_statistics()
    assert payload["schema_version"] == "bootstrap-not-formal"
    assert payload["provenance"]["heldout_data_used"] is False
    assert len(payload["future_state_transition_10d_min"]) == 10


def test_index_manifest_binds_no_heldout_target() -> None:
    spec = p2_training_specs()[0]
    manifest = make_index_manifest(spec, np.zeros(10), np.ones(10), 12, 0)
    assert manifest["schema_version"] == INDEX_SCHEMA_VERSION
    assert manifest["cell_id"] == spec.cell_id
    assert manifest["heldout_target_used"] is False
    assert manifest["visual_encoder_used"] is False


def test_specs_match_frozen_learned_registry() -> None:
    registry = json.loads(
        (ROOT / "方案/v03/M5B_P2_cell_registry_v2.json").read_text(encoding="utf-8")
    )
    assert learned_registry_cell_ids(registry) == {spec.cell_id for spec in p2_training_specs()}


def test_statistics_function_is_exposed_for_real_dataset_only() -> None:
    # Guard against silently replacing the real ranked-dataset computation with
    # a synthetic array-only shortcut in the formal materializer.
    assert compute_training_statistics.__annotations__["dataset"] == "Human2RobotP2Dataset"


def test_materialize_resumes_complete_cell_receipt_without_rebuilding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = p2_training_specs()[0]
    output_root = tmp_path / "data/Human2Robot/derived/m5b_v03/p2_prepared"
    index_path = output_root / "indices" / f"{spec.cell_id}.npz"
    statistics_path = output_root / "statistics" / f"{spec.cell_id}.json"
    receipt_path = output_root / "receipts" / f"{spec.cell_id}.json"
    registry_path = tmp_path / "registry.json"
    index_path.parent.mkdir(parents=True)
    statistics_path.parent.mkdir(parents=True)
    receipt_path.parent.mkdir(parents=True)
    index_path.write_bytes(b"already-complete-index")
    statistics_path.write_text(
        json.dumps({"provenance": {"heldout_data_used": False}}), encoding="utf-8"
    )
    registry_path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_id": spec.cell_id,
                        "artifact_kind": "learned_training_checkpoint",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    entry = {
        "cell_id": spec.cell_id,
        "spec": asdict(spec),
        "config_name": spec.config_name,
        "statistics_path": str(statistics_path.relative_to(tmp_path)),
        "statistics_sha256": prepare.file_sha256(statistics_path),
        "retrieval_index_path": str(index_path.relative_to(tmp_path)),
        "retrieval_index_sha256": prepare.file_sha256(index_path),
        "train_contract": {"split": "train"},
        "heldout_contract": {"split": "heldout"},
    }
    prepare.write_json_atomic(
        receipt_path,
        {
            "schema_version": "human2robot-m5b-p2-cell-receipt-v1",
            "status": "complete",
            "formal_result": False,
            "cell_id": spec.cell_id,
            "protocol_file_sha256": prepare.PROTOCOL_SHA256,
            "supplement_file_sha256": prepare.SUPPLEMENT_SHA256,
            "registry_file_sha256": prepare.REGISTRY_SHA256,
            "split_sha256": prepare.SPLIT_SHA256,
            "pool_manifest_sha256": prepare.POOL_MANIFEST_SHA256,
            "statistics_path": entry["statistics_path"],
            "statistics_sha256": entry["statistics_sha256"],
            "retrieval_index_path": entry["retrieval_index_path"],
            "retrieval_index_sha256": entry["retrieval_index_sha256"],
            "entry": entry,
        },
    )
    monkeypatch.setattr(
        prepare,
        "validate_frozen_inputs",
        lambda _workspace: {"output_root": output_root, "registry_path": registry_path},
    )
    monkeypatch.setattr(prepare, "p2_training_specs", lambda: [spec])

    def unexpected_rebuild(*_args, **_kwargs):
        raise AssertionError("a receipt-validated completed cell must be skipped")

    monkeypatch.setattr(prepare, "build_window_context", unexpected_rebuild)

    result = prepare.materialize(tmp_path, visual_batch_size=4)

    assert result["learned_cell_count"] == 1
    assert result["entries"] == [entry]
