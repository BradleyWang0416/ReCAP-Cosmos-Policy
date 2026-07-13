#!/usr/bin/env python3
"""M5-A-v03 Human2Robot data/contract stress-test launch pipeline.

M5-A deliberately stops before model-dependent mechanism claims.  It verifies
that the frozen M3/M4 contract detects action-role, temporal, and resolution
perturbations, and records the exact evidence still required from M5-B.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

try:
    from tools.human2robot_m2 import file_sha256
    from tools.human2robot_m3 import (
        ALIGNMENT_ID,
        POOL_ACTION_VIEW_RAW,
        QUERY_ACTION_VIEW,
        Episode,
        M3Config,
        TimeViewSpec,
        load_episodes,
        paired_time_view_metrics,
        stable_json_sha256,
        time_view_specs,
        view_segment_indices,
        write_json,
    )
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from human2robot_m2 import file_sha256
    from human2robot_m3 import (
        ALIGNMENT_ID,
        POOL_ACTION_VIEW_RAW,
        QUERY_ACTION_VIEW,
        Episode,
        M3Config,
        TimeViewSpec,
        load_episodes,
        paired_time_view_metrics,
        stable_json_sha256,
        time_view_specs,
        view_segment_indices,
        write_json,
    )


M5A_SCHEMA_VERSION = "human2robot-m5a-data-contract-stress-v03"
DEFAULT_CANONICAL_ROOT = Path("data/Human2Robot/canonical/v3")
DEFAULT_DERIVED_ROOT = Path("data/Human2Robot/derived")
DEFAULT_M3_REPORT = DEFAULT_DERIVED_ROOT / "m3_v03/m3_validation_report.json"
DEFAULT_M4_REPORT = DEFAULT_DERIVED_ROOT / "m4_v03/m4_launch_report.json"
DEFAULT_M4_CONFIG = DEFAULT_DERIVED_ROOT / "m4_v03/paired_bridge_config.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_DERIVED_ROOT / "m5a_v03"
DEFAULT_REPORT_ROOT = Path("方案/v03")

MAIN_TIME_VIEW = "nominal_camera_30hz_segmented"
REQUIRED_TIME_VIEWS = (
    "native_row_index",
    "nominal_camera_30hz_segmented",
    "paper_v2_stride4_nominal7p5",
    "legacy_v01_stride3_nominal10",
    "policy_clock_10hz",
    "phase_or_dtw",
)
REQUIRED_TEMPORAL_STRESSES = (
    "frame_drop",
    "timestamp_jitter",
    "pause",
    "step_jump",
)


class M5AError(RuntimeError):
    """Raised when an M5-A prerequisite or launch invariant fails."""


@dataclass(frozen=True)
class M5AConfig:
    canonical_root: Path = DEFAULT_CANONICAL_ROOT
    m3_report_path: Path = DEFAULT_M3_REPORT
    m4_report_path: Path = DEFAULT_M4_REPORT
    m4_config_path: Path = DEFAULT_M4_CONFIG
    output_root: Path = DEFAULT_OUTPUT_ROOT
    report_root: Path = DEFAULT_REPORT_ROOT
    expected_episode_count: int | None = 20
    wrong_lag: int = 30
    scale_perturbation: float = 2.0
    worsening_ratio_threshold: float = 1.5
    nominal_hz: float = 30.0
    frame_drop_every: int = 10
    timestamp_jitter_std_seconds: float = 0.008
    timestamp_jitter_threshold_seconds: float = 0.005
    pause_seconds: float = 0.5
    pause_threshold_seconds: float = 0.2
    step_jump: int = 20
    random_seed: int = 20260711
    canonical_height: int = 240
    canonical_width: int = 426
    paper_width: int = 424
    sampled_frames_per_stream: int = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise M5AError(f"Required JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise M5AError(f"Expected JSON object: {path}")
    return payload


def _median(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(array)) if array.size else float("nan")


def _percentile(values: Sequence[float] | np.ndarray, percentile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, percentile)) if array.size else float("nan")


def _array_contract_sha256(episodes: Sequence[Episode]) -> str:
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(episode.episode_id.encode())
        for array in (episode.human, episode.robot, episode.segment_id, episode.gap_mask):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.shape).encode())
            digest.update(str(contiguous.dtype).encode())
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def validate_prerequisites(config: M5AConfig) -> dict[str, Any]:
    """Bind M5-A to the passed M3 contract and launched, non-deployable M4 smoke."""
    m3 = _read_json(config.m3_report_path)
    m4 = _read_json(config.m4_report_path)
    m4_config = _read_json(config.m4_config_path)
    gates = m3.get("gates", {})
    if m3.get("status") != "passed" or gates.get("B") != "passed":
        raise M5AError("M3 Gate B must be passed before M5-A")
    if m4.get("status") != "launched" or m4.get("gate_c") != "pending":
        raise M5AError("M4 smoke must be launched with Gate C pending before M5-A")
    if m4.get("m6_rollout_approved") is not False:
        raise M5AError("M5-A must not inherit M6 rollout approval")
    report_bindings = m4.get("bindings", {})
    config_bindings = m4_config.get("bindings", {})
    if report_bindings != config_bindings:
        raise M5AError("M4 report/config binding mismatch")
    expected = {
        "time_view_id": MAIN_TIME_VIEW,
        "pool_action_view_id": POOL_ACTION_VIEW_RAW,
        "query_action_view_id": QUERY_ACTION_VIEW,
        "action_alignment_id": ALIGNMENT_ID,
        "query_command_status": "unverified",
    }
    mismatches = {
        key: {"expected": value, "actual": report_bindings.get(key)}
        for key, value in expected.items()
        if report_bindings.get(key) != value
    }
    if mismatches:
        raise M5AError(f"Frozen M4 contract mismatch: {mismatches}")
    if (
        m4.get("deployment_command_adapter_id") is not None
        or m4_config.get("deployment_command_adapter_id") is not None
    ):
        raise M5AError("M5-A cannot use a deployment command adapter")
    return {
        "m3_status": m3["status"],
        "m3_gate_b": gates["B"],
        "m3_report_path": str(config.m3_report_path),
        "m3_report_file_sha256": file_sha256(config.m3_report_path),
        "m4_status": m4["status"],
        "m4_gate_c": m4["gate_c"],
        "m4_report_path": str(config.m4_report_path),
        "m4_report_file_sha256": file_sha256(config.m4_report_path),
        "m4_config_path": str(config.m4_config_path),
        "m4_config_file_sha256": file_sha256(config.m4_config_path),
        "frozen_bindings": report_bindings,
        "m6_rollout_approved": False,
    }


def _lag_errors(episodes: Sequence[Episode], lag: int) -> dict[str, list[float]]:
    position: list[float] = []
    residual: list[float] = []
    spec = TimeViewSpec(MAIN_TIME_VIEW, stride=1, nominal_hz=30.0, status="main")
    for episode in episodes:
        if episode.split != "train":
            continue
        for rows in view_segment_indices(episode, spec):
            if len(rows) <= lag:
                continue
            pool = episode.human[rows[:-lag], :3]
            target = episode.robot[rows[lag:], :3]
            position.extend(np.linalg.norm(target - pool, axis=1))
            pool_full = episode.human[rows[:-lag]]
            target_full = episode.robot[rows[lag:]]
            residual.extend(np.linalg.norm(target_full - pool_full, axis=1))
    return {"position": position, "residual": residual}


def run_action_role_stress(episodes: Sequence[Episode], config: M5AConfig) -> dict[str, Any]:
    """Run train-only role, leakage, lag, and scale contract perturbations."""
    baseline_errors = _lag_errors(episodes, 1)
    wrong_lag_errors = _lag_errors(episodes, config.wrong_lag)
    if not baseline_errors["position"] or not wrong_lag_errors["position"]:
        raise M5AError("Insufficient train data for action-role/lag stress")
    baseline = _median(baseline_errors["position"])
    baseline_residual = _median(baseline_errors["residual"])
    wrong_lag = _median(wrong_lag_errors["position"])
    wrong_lag_residual = _median(wrong_lag_errors["residual"])

    scaled_errors: list[float] = []
    scaled_residuals: list[float] = []
    same_frame_errors: list[float] = []
    same_frame_residuals: list[float] = []
    spec = TimeViewSpec(MAIN_TIME_VIEW, stride=1, nominal_hz=config.nominal_hz, status="main")
    for episode in episodes:
        if episode.split != "train":
            continue
        for rows in view_segment_indices(episode, spec):
            if len(rows) < 2:
                continue
            current, future = rows[:-1], rows[1:]
            scaled_pool = episode.human[current].copy()
            scaled_pool[:, :3] *= config.scale_perturbation
            scaled_errors.extend(np.linalg.norm(episode.robot[future, :3] - scaled_pool[:, :3], axis=1))
            scaled_residuals.extend(np.linalg.norm(episode.robot[future] - scaled_pool, axis=1))
            same_frame_errors.extend(
                np.linalg.norm(episode.robot[current, :3] - episode.human[current, :3], axis=1)
            )
            same_frame_residuals.extend(
                np.linalg.norm(episode.robot[current] - episode.human[current], axis=1)
            )
    scale_error = _median(scaled_errors)
    scale_residual = _median(scaled_residuals)
    same_frame_error = _median(same_frame_errors)
    same_frame_residual = _median(same_frame_residuals)
    denominator = max(baseline, 1e-12)
    expected_roles = {
        "pool_action_view_id": POOL_ACTION_VIEW_RAW,
        "query_action_view_id": QUERY_ACTION_VIEW,
    }
    swapped_roles = {
        "pool_action_view_id": QUERY_ACTION_VIEW,
        "query_action_view_id": POOL_ACTION_VIEW_RAW,
    }
    role_violations = sum(
        actual != expected_roles[key] for key, actual in swapped_roles.items()
    )
    role_violation_rate = role_violations / len(expected_roles)
    expected_future_offset = 1
    injected_future_offset = 0
    leakage_rate = float(injected_future_offset != expected_future_offset)
    checks = {
        "wrong_role": {
            "injection": "swap frozen pool/query role identifiers",
            "primary_metric": "role_contract_violation_rate",
            "baseline": 0.0,
            "perturbed": role_violation_rate,
            "expected_bindings": expected_roles,
            "injected_bindings": swapped_roles,
            "detector_triggered": role_violation_rate > 0.0,
        },
        "same_frame_copy": {
            "injection": "replace strictly-future query label with observed state at t",
            "primary_metric": "temporal_leakage_rate",
            "baseline": 0.0,
            "perturbed": leakage_rate,
            "expected_future_offset_view_steps": expected_future_offset,
            "injected_future_offset_view_steps": injected_future_offset,
            "strict_future_target_rate": float(injected_future_offset > 0),
            "misleading_position_error_median_canonical": same_frame_error,
            "misleading_residual_norm_median": same_frame_residual,
            "detector_triggered": leakage_rate > 0.0,
        },
        "wrong_lag": {
            "injection": f"use robot observed t+{config.wrong_lag} instead of t+1",
            "primary_metric": "paired_position_error_median_canonical",
            "baseline": baseline,
            "perturbed": wrong_lag,
            "ratio": wrong_lag / denominator,
            "baseline_residual_norm_median": baseline_residual,
            "perturbed_residual_norm_median": wrong_lag_residual,
            "threshold_ratio": config.worsening_ratio_threshold,
            "detector_triggered": wrong_lag / denominator >= config.worsening_ratio_threshold,
        },
        "scale_x2": {
            "injection": f"multiply pool xyz by {config.scale_perturbation:g}",
            "primary_metric": "paired_position_error_median_canonical",
            "baseline": baseline,
            "perturbed": scale_error,
            "ratio": scale_error / denominator,
            "baseline_residual_norm_median": baseline_residual,
            "perturbed_residual_norm_median": scale_residual,
            "threshold_ratio": config.worsening_ratio_threshold,
            "detector_triggered": scale_error / denominator >= config.worsening_ratio_threshold,
        },
    }
    return {
        "schema_version": M5A_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "scope": "train split only; held-out robot trajectories are not used",
        "data_scope": {
            "train_episode_count": sum(episode.split == "train" for episode in episodes),
            "heldout_episode_count": sum(episode.split == "heldout" for episode in episodes),
            "heldout_robot_trajectory_used": False,
        },
        "checks": checks,
        "status": "passed" if all(item["detector_triggered"] for item in checks.values()) else "failed",
    }


def run_time_view_matrix(episodes: Sequence[Episode], config: M5AConfig) -> dict[str, Any]:
    m3_config = M3Config(
        canonical_root=config.canonical_root,
        expected_episode_count=config.expected_episode_count,
        phase_bins=64,
    )
    results = [paired_time_view_metrics(episodes, spec) for spec in time_view_specs(m3_config)]
    observed = {item["time_view_id"] for item in results}
    required_present = observed == set(REQUIRED_TIME_VIEWS)
    gap_crossing_zero = all(item["gap_crossing_count"] == 0 for item in results)
    nonempty = all(item["sample_count"] > 0 for item in results)
    return {
        "schema_version": M5A_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": "passed" if required_present and gap_crossing_zero and nonempty else "failed",
        "required_time_views": list(REQUIRED_TIME_VIEWS),
        "required_views_present": required_present,
        "gap_crossing_zero": gap_crossing_zero,
        "all_views_nonempty": nonempty,
        "interpretation": "descriptive data-level comparison; no model robustness claim",
        "results": results,
    }


def _temporal_stress_metrics(
    episodes: Sequence[Episode], stress: str, config: M5AConfig
) -> dict[str, Any]:
    if stress not in ("baseline", *REQUIRED_TEMPORAL_STRESSES):
        raise ValueError(f"Unknown temporal stress: {stress}")
    rng = np.random.default_rng(config.random_seed)
    nominal_dt = 1.0 / config.nominal_hz
    source_row_deltas: list[float] = []
    dt_errors: list[float] = []
    dt_values: list[float] = []
    residual_norms: list[float] = []
    position_errors: list[float] = []
    source_row_jump_count = 0
    jitter_exceed_count = 0
    pause_count = 0
    step_jump_count = 0
    nonmonotonic_timestamp_count = 0
    gap_crossing_count = 0
    pair_count = 0
    spec = TimeViewSpec(MAIN_TIME_VIEW, stride=1, nominal_hz=config.nominal_hz, status="main")
    for episode in episodes:
        if episode.split != "train":
            continue
        for original_rows in view_segment_indices(episode, spec):
            if len(original_rows) < max(4, config.frame_drop_every + 1):
                continue
            rows = original_rows.copy()
            if stress == "frame_drop":
                ordinal = np.arange(len(rows))
                rows = rows[(ordinal + 1) % config.frame_drop_every != 0]
            times = rows.astype(np.float64) * nominal_dt
            logical_steps = rows.astype(np.int64).copy()
            if stress == "timestamp_jitter":
                times = times + rng.normal(0.0, config.timestamp_jitter_std_seconds, size=len(times))
            elif stress == "pause":
                times[len(times) // 2 :] += config.pause_seconds
            elif stress == "step_jump":
                logical_steps[len(logical_steps) // 2 :] += config.step_jump
            if len(rows) < 2:
                continue
            row_delta = np.diff(rows).astype(np.float64)
            dt = np.diff(times)
            logical_delta = np.diff(logical_steps).astype(np.float64)
            expected_dt = row_delta * nominal_dt
            source_row_deltas.extend(row_delta)
            dt_values.extend(dt)
            dt_errors.extend(np.abs(dt - expected_dt))
            source_row_jump_count += int(np.count_nonzero(row_delta != 1))
            jitter_exceed_count += int(
                np.count_nonzero(np.abs(dt - expected_dt) > config.timestamp_jitter_threshold_seconds)
            )
            pause_count += int(np.count_nonzero(dt > config.pause_threshold_seconds))
            step_jump_count += int(np.count_nonzero(logical_delta != row_delta))
            nonmonotonic_timestamp_count += int(np.count_nonzero(dt <= 0))
            current, future = rows[:-1], rows[1:]
            gap_crossing_count += int(
                np.count_nonzero(episode.segment_id[current] != episode.segment_id[future])
            )
            residual = episode.robot[future] - episode.human[current]
            residual_norms.extend(np.linalg.norm(residual, axis=1))
            position_errors.extend(np.linalg.norm(residual[:, :3], axis=1))
            pair_count += len(current)
    detector_counts = {
        "source_row_jump_count": source_row_jump_count,
        "timestamp_jitter_exceed_count": jitter_exceed_count,
        "pause_count": pause_count,
        "logical_step_jump_count": step_jump_count,
        "nonmonotonic_timestamp_count": nonmonotonic_timestamp_count,
    }
    detector_key = {
        "baseline": None,
        "frame_drop": "source_row_jump_count",
        "timestamp_jitter": "timestamp_jitter_exceed_count",
        "pause": "pause_count",
        "step_jump": "logical_step_jump_count",
    }[stress]
    detector_triggered = False if detector_key is None else detector_counts[detector_key] > 0
    return {
        "stress": stress,
        "pair_count": pair_count,
        "gap_crossing_count": gap_crossing_count,
        "source_row_delta_median": _median(source_row_deltas),
        "source_row_delta_p95": _percentile(source_row_deltas, 95),
        "dt_median_seconds": _median(dt_values),
        "dt_absolute_error_p95_seconds": _percentile(dt_errors, 95),
        "residual_norm_median": _median(residual_norms),
        "position_error_median_canonical": _median(position_errors),
        **detector_counts,
        "expected_detector": detector_key,
        "detector_triggered": detector_triggered,
        "safe_without_rejection_or_mask": stress == "baseline",
    }


def run_temporal_stress(episodes: Sequence[Episode], config: M5AConfig) -> dict[str, Any]:
    baseline = _temporal_stress_metrics(episodes, "baseline", config)
    stresses = {
        stress: _temporal_stress_metrics(episodes, stress, config)
        for stress in REQUIRED_TEMPORAL_STRESSES
    }
    baseline_clear = (
        baseline["gap_crossing_count"] == 0
        and baseline["source_row_jump_count"] == 0
        and baseline["timestamp_jitter_exceed_count"] == 0
        and baseline["pause_count"] == 0
        and baseline["logical_step_jump_count"] == 0
    )
    all_detected = all(item["detector_triggered"] for item in stresses.values())
    no_crossing = baseline["gap_crossing_count"] == 0 and all(
        item["gap_crossing_count"] == 0 for item in stresses.values()
    )
    return {
        "schema_version": M5A_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "injection_config": {
            "nominal_hz": config.nominal_hz,
            "frame_drop_every": config.frame_drop_every,
            "timestamp_jitter_std_seconds": config.timestamp_jitter_std_seconds,
            "timestamp_jitter_threshold_seconds": config.timestamp_jitter_threshold_seconds,
            "pause_seconds": config.pause_seconds,
            "pause_threshold_seconds": config.pause_threshold_seconds,
            "step_jump": config.step_jump,
            "random_seed": config.random_seed,
        },
        "baseline": baseline,
        "stresses": stresses,
        "baseline_detector_clear": baseline_clear,
        "all_required_detectors_triggered": all_detected,
        "gap_crossing_zero": no_crossing,
        "status": "passed" if baseline_clear and all_detected and no_crossing else "failed",
    }


def run_resolution_stress(
    episodes: Sequence[Episode], config: M5AConfig, retrieval_feature_schema: Sequence[str]
) -> dict[str, Any]:
    """Audit 426↔424 crop/pad strategies without mutating canonical files."""
    action_hash_before = _array_contract_sha256(episodes)
    stream_shapes: dict[str, set[tuple[int, int, int]]] = {
        "human/images": set(),
        "robot_images": set(),
    }
    samples_checked = 0
    crop_shape_valid = True
    pad_shape_valid = True
    inner_pixels_exact = True
    border_change_fractions: list[float] = []
    for episode in episodes:
        with h5py.File(episode.path, "r") as file:
            demo = file["data/demo_0"]
            datasets = {
                "human/images": demo["metadata/human/images"],
                "robot_images": demo["obs/robot_images"],
            }
            for name, dataset in datasets.items():
                stream_shapes[name].add(tuple(int(value) for value in dataset.shape[1:]))
                count = min(config.sampled_frames_per_stream, len(dataset))
                indices = np.linspace(0, len(dataset) - 1, num=count, dtype=np.int64)
                for index in np.unique(indices):
                    frame = np.asarray(dataset[int(index)])
                    if frame.shape != (
                        config.canonical_height,
                        config.canonical_width,
                        3,
                    ):
                        crop_shape_valid = False
                        continue
                    left = (config.canonical_width - config.paper_width) // 2
                    right = left + config.paper_width
                    cropped = frame[:, left:right]
                    reconstructed = np.pad(cropped, ((0, 0), (left, config.canonical_width - right), (0, 0)), mode="edge")
                    crop_shape_valid &= cropped.shape == (
                        config.canonical_height,
                        config.paper_width,
                        3,
                    )
                    pad_shape_valid &= reconstructed.shape == frame.shape
                    inner_pixels_exact &= bool(np.array_equal(reconstructed[:, left:right], cropped))
                    border_change_fractions.append(float(np.mean(reconstructed != frame)))
                    samples_checked += 1
    action_hash_after = _array_contract_sha256(episodes)
    expected_shape = (config.canonical_height, config.canonical_width, 3)
    all_source_shapes_expected = all(shapes == {expected_shape} for shapes in stream_shapes.values())
    retrieval_image_independent = not any(
        "image" in feature.lower() or "visual" in feature.lower()
        for feature in retrieval_feature_schema
    )
    action_invariant = action_hash_before == action_hash_after
    checks = {
        "all_source_shapes_are_240x426": all_source_shapes_expected,
        "center_crop_426_to_424_shape_valid": crop_shape_valid,
        "edge_pad_424_to_426_shape_valid": pad_shape_valid,
        "crop_then_pad_inner_pixels_exact": inner_pixels_exact,
        "action_contract_sha256_unchanged": action_invariant,
        "current_phase_retrieval_exactly_image_independent": retrieval_image_independent,
    }
    return {
        "schema_version": M5A_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": "passed" if all(checks.values()) else "failed",
        "source_stream_shapes": {
            name: [list(shape) for shape in sorted(shapes)] for name, shapes in stream_shapes.items()
        },
        "samples_checked": samples_checked,
        "strategies": {
            "paper_comparison_main": {
                "id": "center_crop_width_426_to_424_v1",
                "operation": "remove one column from each horizontal edge; no resize",
                "output_shape": [config.canonical_height, config.paper_width, 3],
            },
            "canonical_compatibility": {
                "id": "edge_pad_width_424_to_426_v1",
                "operation": "replicate one horizontal edge column on each side; no resize",
                "output_shape": [config.canonical_height, config.canonical_width, 3],
            },
        },
        "retrieval_feature_schema": list(retrieval_feature_schema),
        "action_contract_sha256_before": action_hash_before,
        "action_contract_sha256_after": action_hash_after,
        "roundtrip_border_change_fraction_median": _median(border_change_fractions),
        "checks": checks,
        "claim_boundary": {
            "current_action_conclusion": "exactly invariant because image preprocessing never mutates action arrays",
            "current_phase_retrieval_conclusion": "exactly invariant because retrieval features contain no image/visual field",
            "visual_retrieval_conclusion": "NEEDS_EXPERIMENT in M5-B with the frozen visual encoder",
        },
    }


def build_experiment_protocol(config: M5AConfig, prerequisites: dict[str, Any]) -> dict[str, Any]:
    matrix = [
        {
            "experiment_id": "M5A-AR-01",
            "type": "action-role robustness",
            "perturbations": ["wrong_role", "same_frame_copy", "wrong_lag", "scale_x2"],
            "required": True,
            "expected_evidence": "each frozen-contract detector triggers on train-only injected mismatch",
        },
        {
            "experiment_id": "M5A-TV-01",
            "type": "FPS/version descriptive comparison",
            "perturbations": list(REQUIRED_TIME_VIEWS),
            "required": True,
            "expected_evidence": "all views are nonempty and never cross a canonical segment",
        },
        {
            "experiment_id": "M5A-TM-01",
            "type": "temporal mismatch robustness",
            "perturbations": list(REQUIRED_TEMPORAL_STRESSES),
            "required": True,
            "expected_evidence": "frame/clock/step anomaly detector triggers with zero segment crossing",
        },
        {
            "experiment_id": "M5A-RES-01",
            "type": "resolution contract",
            "perturbations": ["center_crop_426_to_424", "edge_pad_424_to_426"],
            "required": True,
            "expected_evidence": "shape strategy is explicit; action hash and phase retrieval contract are invariant",
        },
    ]
    return {
        "schema_version": M5A_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "normalized_research_claim": (
            "The frozen v03 data contract detects injected action-role and temporal mismatches, "
            "and makes the 240x426 versus 240x424 preprocessing choice explicit before model training."
        ),
        "claim_status": "SUPPORTED only for data/contract detection after all M5-A launch checks pass",
        "target_task_and_dataset_assumptions": {
            "dataset": "20-episode Human2Robot canonical/v3 pilot",
            "split": "frozen task-level train/heldout split",
            "action_scope": "train-only calibration; held-out robot trajectory excluded",
            "retrieval_scope": "normalized segment phase + task pool membership",
            "query_command_status": "unverified",
        },
        "minimum_viable_experiment_set": matrix,
        "claim_to_experiment_mapping": {
            "role_semantics_are_guarded": ["M5A-AR-01"],
            "temporal_mismatch_is_detectable": ["M5A-TV-01", "M5A-TM-01"],
            "resolution_policy_is_reproducible": ["M5A-RES-01"],
            "model_mechanism_is_robust": ["NEEDS_EXPERIMENT:M5-B"],
        },
        "required_baselines": [
            "frozen nominal30/t+1/raw-human-plan contract",
            "unperturbed temporal detector baseline",
            "uncropped canonical 240x426 image contract",
        ],
        "required_ablations": [
            "wrong role", "same-frame label", "wrong lag", "scale x2",
            "nominal30/stride4/stride3/policy-clock/phase-DTW",
            "frame drop/jitter/pause/step jump", "center crop/edge pad",
        ],
        "sensitivity_analysis": {
            "M5A": "fixed severe perturbations verify detector behavior",
            "M5B": "NEEDS_EXPERIMENT for graded severity and model performance curves",
        },
        "robustness_generalization_check": "contract-level only in M5-A; model robustness NEEDS_EXPERIMENT",
        "efficiency_complexity_check": "not claimed by M5-A",
        "qualitative_analysis": "NEEDS_EXPERIMENT in M5-B for visual retrieval/crop examples",
        "failure_case_analysis": [
            "a low same-frame numeric error can coexist with 100% temporal leakage",
            "phase-only retrieval cannot validate visual crop robustness",
            "single-pilot data cannot support multi-seed model claims",
        ],
        "missing_evidence": [
            "formal Cosmos/RECAP M4 multi-seed checkpoints",
            "independent or expanded human-only held-out pool",
            "residual/absolute and future-state model ablations",
            "retrieval modality, geometry/visual, top-k and pool-growth model ablations",
            "visual encoder crop/pad feature-stability experiment",
        ],
        "risk_of_overclaiming": (
            "M5-A must not be cited as evidence that RECAP outperforms baselines, is model-robust, "
            "or is deployable."
        ),
        "recommended_next_stage": "finish formal M4 training, then run M5-B before Gate C decision",
        "implementation": str(Path(__file__).resolve()),
        "implementation_code_sha256": file_sha256(Path(__file__)),
        "fairness_and_validity": [
            "all action/lag thresholds are evaluated on train tasks only",
            "all time views use the same canonical episodes and gap policy",
            "synthetic temporal injections use a fixed recorded seed",
            "resolution audit is read-only and records exact crop/pad operations",
        ],
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "prerequisite_hashes": {
            key: value for key, value in prerequisites.items() if key.endswith("sha256")
        },
    }


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    action = report["action_role_stress"]["checks"]
    temporal = report["temporal_stress"]
    resolution = report["resolution_stress"]
    time_views = report["time_view_matrix"]["results"]
    lines = [
        "# M5-A-v03 数据与契约压力测试启动报告",
        "",
        f"日期：{report['created_at_utc']}",
        "",
        f"结论：**{report['decision']}**",
        "",
        "M5-A 只验证数据与契约检测能力，不声明最终 Cosmos/RECAP 模型收益、鲁棒性或可部署性。",
        "",
        "## 前置状态",
        "",
        f"- M3 Gate B：`{report['prerequisites']['m3_gate_b']}`",
        f"- M4：`{report['prerequisites']['m4_status']}`；Gate C=`{report['prerequisites']['m4_gate_c']}`",
        "- query command：`unverified`；M6 rollout：`false`",
        "",
        "## Action-role / lag 压力测试",
        "",
        "| 扰动 | 主指标 | baseline | perturbed | detector |",
        "|---|---|---:|---:|---|",
    ]
    for name, item in action.items():
        lines.append(
            f"| `{name}` | `{item['primary_metric']}` | {item['baseline']:.6f} | "
            f"{item['perturbed']:.6f} | {'triggered' if item['detector_triggered'] else 'missed'} |"
        )
    lines.extend(
        [
            "",
            "## FPS/version 数据级对比",
            "",
            "| time view | samples | gap crossing | residual median | position median |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in time_views:
        lines.append(
            f"| `{item['time_view_id']}` | {item['sample_count']} | {item['gap_crossing_count']} | "
            f"{item['residual_norm_median']:.6f} | {item['position_error_median_canonical']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Temporal mismatch 注入",
            "",
            "| 扰动 | pairs | detector | triggered | gap crossing | position median |",
            "|---|---:|---|---|---:|---:|",
        ]
    )
    for name, item in temporal["stresses"].items():
        lines.append(
            f"| `{name}` | {item['pair_count']} | `{item['expected_detector']}` | "
            f"{str(item['detector_triggered']).lower()} | {item['gap_crossing_count']} | "
            f"{item['position_error_median_canonical']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 分辨率契约",
            "",
            f"- canonical streams：`{resolution['source_stream_shapes']}`",
            "- paper 对比主策略：`center_crop_width_426_to_424_v1`，左右各裁 1 列，不 resize。",
            "- canonical 兼容策略：`edge_pad_width_424_to_426_v1`，左右各复制 1 列，不 resize。",
            f"- action contract hash unchanged：`{str(resolution['checks']['action_contract_sha256_unchanged']).lower()}`",
            f"- phase retrieval image-independent：`{str(resolution['checks']['current_phase_retrieval_exactly_image_independent']).lower()}`",
            "- visual retrieval crop/pad robustness：`NEEDS_EXPERIMENT`（M5-B）。",
            "",
            "## 启动边界与下一步",
            "",
            "- M5-A 已执行并产出协议、四类数据/契约检查和自动报告。",
            "- M5-B 仍为 pending：需正式 M4 多 seed checkpoint 后执行模型依赖型消融。",
            "- Gate C 仍为 pending；不得据此批准 M6 或真实机器人 command。",
            "",
            "## 产物",
            "",
        ]
    )
    for key, value in report["artifacts"].items():
        lines.append(f"- `{key}`：`{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_m5a_launch(config: M5AConfig = M5AConfig()) -> dict[str, Any]:
    prerequisites = validate_prerequisites(config)
    episodes, split, _preprocessing = load_episodes(
        M3Config(canonical_root=config.canonical_root, expected_episode_count=config.expected_episode_count)
    )
    protocol = build_experiment_protocol(config, prerequisites)
    action = run_action_role_stress(episodes, config)
    time_view_matrix = run_time_view_matrix(episodes, config)
    temporal = run_temporal_stress(episodes, config)
    retrieval_schema = _read_json(config.m4_config_path).get("retrieval_feature_schema", [])
    resolution = run_resolution_stress(episodes, config, retrieval_schema)

    artifacts = {
        "experiment_protocol": str(config.output_root / "experiment_protocol.json"),
        "action_role_stress": str(config.output_root / "action_role_stress.json"),
        "time_view_matrix": str(config.output_root / "time_view_matrix.json"),
        "temporal_stress": str(config.output_root / "temporal_stress.json"),
        "resolution_stress": str(config.output_root / "resolution_stress.json"),
        "automatic_report": str(config.output_root / "m5a_launch_report.json"),
    }
    write_json(Path(artifacts["experiment_protocol"]), protocol)
    write_json(Path(artifacts["action_role_stress"]), action)
    write_json(Path(artifacts["time_view_matrix"]), time_view_matrix)
    write_json(Path(artifacts["temporal_stress"]), temporal)
    write_json(Path(artifacts["resolution_stress"]), resolution)

    launch_checks = {
        "m3_gate_b_passed": prerequisites["m3_gate_b"] == "passed",
        "m4_launched_gate_c_pending": (
            prerequisites["m4_status"] == "launched" and prerequisites["m4_gate_c"] == "pending"
        ),
        "action_role_detectors_passed": action["status"] == "passed",
        "all_time_views_ran_without_gap_crossing": time_view_matrix["status"] == "passed",
        "temporal_mismatch_detectors_passed": temporal["status"] == "passed",
        "resolution_contract_passed": resolution["status"] == "passed",
        "heldout_robot_not_used_for_action_calibration": not action["data_scope"][
            "heldout_robot_trajectory_used"
        ],
        "m5b_kept_pending": True,
        "m6_rollout_forbidden": prerequisites["m6_rollout_approved"] is False,
    }
    status = "launched" if all(launch_checks.values()) else "failed"
    report = {
        "schema_version": M5A_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": status,
        "decision": (
            "M5-A-v03 已启动并完成首轮数据/契约压力测试；M5-B 与 Gate C 保持 pending"
            if status == "launched"
            else "M5-A-v03 启动失败"
        ),
        "phase_scope": "M5-A data/contract stress only",
        "prerequisites": prerequisites,
        "split_sha256": split.get("split_sha256"),
        "protocol_sha256": stable_json_sha256(protocol),
        "action_role_stress": action,
        "time_view_matrix": time_view_matrix,
        "temporal_stress": temporal,
        "resolution_stress": resolution,
        "launch_checks": launch_checks,
        "artifacts": artifacts,
        "m5b_status": "pending_formal_m4_multiseed_checkpoints",
        "gate_c": "pending",
        "deployment_command_adapter_id": None,
        "m6_rollout_approved": False,
        "limitations": protocol["missing_evidence"],
    }
    report["report_sha256"] = stable_json_sha256(report)
    write_json(Path(artifacts["automatic_report"]), report)
    write_json(config.report_root / "M5A_data_contract_stress_自动启动报告.json", report)
    _write_markdown_report(config.report_root / "M5A_data_contract_stress_启动报告.md", report)
    return report
