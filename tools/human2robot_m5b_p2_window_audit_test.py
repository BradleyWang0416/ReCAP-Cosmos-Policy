from __future__ import annotations

from pathlib import Path

from tools.human2robot_m5b_p2_window_audit import audit, frozen_p2_windows, old_p0_windows


WORKSPACE = Path(__file__).resolve().parents[1]


def test_old_and_p2_window_semantics_have_different_current_anchor() -> None:
    rows = list(range(20))
    old = old_p0_windows(rows)
    new = frozen_p2_windows(rows)
    assert old[0]["current_row"] == 0
    assert old[0]["history_or_pool_rows"] == list(range(8))
    assert new[0]["current_row"] == 7
    assert new[0]["history_or_pool_rows"] == list(range(8))
    assert new[0]["future_query_rows"] == list(range(8, 16))


def test_real_prepared_counts_match_frozen_p2_not_old_p0() -> None:
    result = audit(WORKSPACE)
    assert result["status"] == "passed_with_migration_boundary"
    assert result["semantics"]["same_semantics"] is False
    assert result["counts"]["train"] == {
        "old_p0_query_count": 968,
        "frozen_p2_query_count": 954,
    }
    assert result["counts"]["heldout"] == {
        "old_p0_query_count": 153,
        "frozen_p2_query_count": 149,
    }
    assert result["evidence_boundary"]["p2_prepared_inputs_follow_frozen_contract"] is True
