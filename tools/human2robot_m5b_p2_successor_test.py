from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.human2robot_m5b_p2_matrix import file_sha256


WORKSPACE = Path(__file__).resolve().parents[1]


def read(relative: str) -> dict:
    return json.loads((WORKSPACE / relative).read_text(encoding="utf-8"))


def test_v1_registry_is_preserved_and_v2_adds_one_terminal_cell() -> None:
    v1_path = WORKSPACE / "方案/v03/M5B_P2_cell_registry_v1.json"
    assert file_sha256(v1_path) == "4664d036bcf6bc41e8a44fac2afe04ff6de62c2a180a29d3433bd83e46604df5"
    v1 = read("方案/v03/M5B_P2_cell_registry_v1.json")
    v2 = read("方案/v03/M5B_P2_cell_registry_v2.json")
    assert v2["cells"][:202] == v1["cells"]
    terminal = v2["cells"][-1]
    assert terminal["cell_id"] == "aggregate_report__M5B-QUAL-01__full_matrix_completion_acceptance"
    assert len(terminal["parent_artifact_ids"]) == 202
    assert set(terminal["parent_artifact_ids"]) == {cell["cell_id"] for cell in v1["cells"]}
    assert v2["formal_queue_allowed"] is False
    assert v2["p2_acceptance_allowed"] is False


def test_workspace_bounds_are_train_only_five_percent_envelope_without_clipping() -> None:
    bounds = read("方案/v03/M5B_P2_workspace_bounds_v1.json")
    raw_min = np.asarray(bounds["raw_train_xyz_min"])
    raw_max = np.asarray(bounds["raw_train_xyz_max"])
    margin = 0.05 * (raw_max - raw_min)
    np.testing.assert_allclose(bounds["margin_xyz"], margin)
    np.testing.assert_allclose(bounds["xyz_min"], raw_min - margin)
    np.testing.assert_allclose(bounds["xyz_max"], raw_max + margin)
    assert bounds["heldout_data_used"] is False
    assert bounds["source_episode_count"] == 16
    assert bounds["inference_policy"].startswith("do_not_clip")


def test_only_three_lag_entries_are_rematerialized_against_offset5_view() -> None:
    old = read("data/Human2Robot/derived/m5b_v03/p2_prepared/prepared_manifest.json")
    new = read("data/Human2Robot/derived/m5b_v03/p2_prepared_v2/prepared_manifest.json")
    old_by_id = {entry["cell_id"]: entry for entry in old["entries"]}
    lag = [entry for entry in new["entries"] if entry["spec"]["query_offset_view_steps"] == 5]
    reused = [entry for entry in new["entries"] if entry["spec"]["query_offset_view_steps"] == 1]
    assert len(lag) == 3
    assert len(reused) == 45
    assert all(entry.get("successor_materialization", "").startswith("hash-identical") for entry in reused)
    for entry in reused:
        old_entry = old_by_id[entry["cell_id"]]
        assert entry["statistics_sha256"] == old_entry["statistics_sha256"]
        assert entry["retrieval_index_sha256"] == old_entry["retrieval_index_sha256"]
    for entry in lag:
        assert entry["train_contract"]["query_count"] == 943
        assert entry["heldout_contract"]["query_count"] == 147
        assert entry["train_contract"]["time_view_manifest_sha256"] == (
            "53ab59227f865767f07fd4b8c6cea52689b7c22ec1359cedb975308644fe806d"
        )


def test_two_stage_schemas_are_frozen_but_neither_is_an_activation_or_result() -> None:
    launch = read("方案/v03/M5B_P2_launch_activation_schema_v2.json")
    final = read("方案/v03/M5B_P2_final_acceptance_schema_v2.json")
    assert launch["status"] == "frozen_schema_not_activated"
    assert launch["required_exact_values"]["p2_acceptance_allowed"] is False
    assert final["status"] == "frozen_schema_not_accepted"
    assert final["required_exact_values"]["p2_acceptance_allowed"] is True
    assert final["required_exact_values"]["terminal_report_status"] == "passed"
