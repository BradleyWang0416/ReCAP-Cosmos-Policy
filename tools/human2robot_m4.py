#!/usr/bin/env python3
"""M4-v03 offline Human2Robot paired-bridge launch pipeline.

This module intentionally implements a small, deterministic action-space bridge.
It is an offline launch artifact, not a deployment command adapter and not a
Gate-C acceptance claim.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from tools.human2robot_m2 import SCHEMA_VERSION, file_sha256
    from tools.human2robot_m3 import (
        ALIGNMENT_ID,
        POOL_ACTION_VIEW_RAW,
        QUERY_ACTION_VIEW,
        Episode,
        M3Config,
        TimeViewSpec,
        align_pool_chunk,
        load_episodes,
        matrix_to_rotation_6d,
        orientation_error_rad,
        rotation_6d_to_matrix,
        stable_json_sha256,
        view_segment_indices,
        write_json,
    )
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from human2robot_m2 import SCHEMA_VERSION, file_sha256
    from human2robot_m3 import (
        ALIGNMENT_ID,
        POOL_ACTION_VIEW_RAW,
        QUERY_ACTION_VIEW,
        Episode,
        M3Config,
        TimeViewSpec,
        align_pool_chunk,
        load_episodes,
        matrix_to_rotation_6d,
        orientation_error_rad,
        rotation_6d_to_matrix,
        stable_json_sha256,
        view_segment_indices,
        write_json,
    )


M4_SCHEMA_VERSION = "human2robot-m4-offline-bridge-v03"
DEFAULT_CANONICAL_ROOT = Path("data/Human2Robot/canonical/v3")
DEFAULT_DERIVED_ROOT = Path("data/Human2Robot/derived")
DEFAULT_M3_REPORT = DEFAULT_DERIVED_ROOT / "m3_v03/m3_validation_report.json"
DEFAULT_MAIN_VIEW = (
    DEFAULT_DERIVED_ROOT
    / "views/nominal_camera_30hz_segmented"
    / POOL_ACTION_VIEW_RAW
    / QUERY_ACTION_VIEW
    / ALIGNMENT_ID
)
DEFAULT_OUTPUT_ROOT = DEFAULT_DERIVED_ROOT / "m4_v03"
DEFAULT_REPORT_ROOT = Path("方案/v03")

BASELINES = ("no_retrieval", "retrieval_only", "co_training", "recap_hand_ret")
CHECKPOINT_BINDING_KEYS = (
    "canonical_schema",
    "canonical_manifest_sha256",
    "source_evidence_manifest_sha256",
    "split_sha256",
    "time_view_id",
    "pool_action_view_id",
    "query_action_view_id",
    "action_alignment_id",
    "pool_action_role",
    "query_action_role",
    "query_command_status",
    "policy_coordinate",
    "H_steps",
    "H_seconds",
    "K_steps",
    "K_seconds",
    "gap_policy",
    "alignment_version",
    "view_id",
    "retrieval_index_sha256",
)


class M4Error(RuntimeError):
    """Raised when the M4 contract, leakage, or checkpoint binding is invalid."""


@dataclass(frozen=True)
class M4Config:
    canonical_root: Path = DEFAULT_CANONICAL_ROOT
    main_view_path: Path = DEFAULT_MAIN_VIEW
    m3_report_path: Path = DEFAULT_M3_REPORT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    report_root: Path = DEFAULT_REPORT_ROOT
    horizon: int = 8
    window_stride: int = 8
    retrieval_top_k: int = 3
    ridge_alpha: float = 1e-2
    random_seed: int = 20260711
    pool_growth_sizes: tuple[int, ...] = (1, 2, 4, 8, 0)
    expected_episode_count: int | None = 20


@dataclass(frozen=True)
class BridgeSample:
    episode_id: str
    task: str
    split: str
    segment_number: int
    phase: float
    current_row: int
    current: np.ndarray
    human_plan: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        normalized = (values - self.mean) / self.scale
        design = np.concatenate((normalized, np.ones((len(normalized), 1))), axis=1)
        return design @ self.weights


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise M4Error(f"Required JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise M4Error(f"Expected JSON object: {path}")
    return payload


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path.cwd() / path


def validate_m4_contract(config: M4Config) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify Gate B and freeze all semantic/time/action bindings for M4."""
    view = _read_json(config.main_view_path / "view_manifest.json")
    m3_report = _read_json(config.m3_report_path)
    split = _read_json(config.canonical_root / "task_split_manifest.json")
    expected = {
        "canonical_schema": SCHEMA_VERSION,
        "time_view_id": "nominal_camera_30hz_segmented",
        "pool_action_view_id": POOL_ACTION_VIEW_RAW,
        "query_action_view_id": QUERY_ACTION_VIEW,
        "action_alignment_id": ALIGNMENT_ID,
        "pool_action_role": "pool_side_human_plan_in_robot_frame",
        "query_action_role": "dataset_card_approved_bc_proxy",
        "query_command_status": "unverified",
        "strict_future_target": True,
        "query_target_offset_view_steps": 1,
        "gap_policy": "never_cross_segment",
    }
    mismatches = {key: {"expected": value, "actual": view.get(key)} for key, value in expected.items() if view.get(key) != value}
    if mismatches:
        raise M4Error(f"M4 main-view contract mismatch: {mismatches}")
    if view.get("deployment_command_adapter_id") is not None:
        raise M4Error("Offline M4 requires deployment_command_adapter_id=null")
    if view.get("H_steps") != config.horizon or view.get("K_steps") != config.horizon:
        raise M4Error(
            f"H/K mismatch: view={view.get('H_steps')}/{view.get('K_steps')} config={config.horizon}"
        )
    if m3_report.get("status") != "passed" or m3_report.get("gates", {}).get("B") != "passed":
        raise M4Error("M3 Gate B is not passed")
    reported_view = _resolve_repo_path(m3_report.get("main_view_path", ""))
    if reported_view.resolve() != config.main_view_path.resolve():
        raise M4Error(f"M3 main view mismatch: {reported_view} != {config.main_view_path}")
    if split.get("split_sha256") != view.get("split_sha256"):
        raise M4Error("Task split hash does not match the selected M3 view")

    preprocessing = config.canonical_root / "preprocessing_manifest.json"
    if file_sha256(preprocessing) != view.get("canonical_manifest_sha256"):
        raise M4Error("Canonical preprocessing manifest hash mismatch")
    evidence = config.report_root / "source_evidence_manifest_v3.json"
    if file_sha256(evidence) != view.get("source_evidence_manifest_sha256"):
        raise M4Error("Source evidence manifest hash mismatch")
    alignment_path = config.m3_report_path.parent / "alignment_calibration.json"
    alignment = _read_json(alignment_path)
    if stable_json_sha256(alignment) != view.get("action_alignment_manifest_sha256"):
        raise M4Error("Action-alignment manifest hash mismatch")
    retrieval_index = config.main_view_path / "retrieval_index.npz"
    reported_index_hash = m3_report.get("retrieval_sanity", {}).get("retrieval_index_sha256")
    if file_sha256(retrieval_index) != reported_index_hash:
        raise M4Error("M3 retrieval index hash mismatch")
    return view, m3_report, split


def _time_view_spec(view: dict[str, Any]) -> TimeViewSpec:
    item = view["time_view"]
    return TimeViewSpec(
        time_view_id=str(item["time_view_id"]),
        stride=int(item["stride"]),
        phase_bins=item.get("phase_bins"),
        nominal_hz=item.get("nominal_hz"),
        paper_version=item.get("paper_version"),
        status=str(item.get("status", "main")),
    )


def build_bridge_samples(
    episodes: Sequence[Episode], spec: TimeViewSpec, config: M4Config
) -> tuple[list[BridgeSample], dict[str, Any]]:
    samples: list[BridgeSample] = []
    gap_crossing_count = 0
    for episode in episodes:
        for segment_number, rows in enumerate(view_segment_indices(episode, spec)):
            last_start = len(rows) - config.horizon - 1
            for local_start in range(0, max(0, last_start + 1), config.window_stride):
                current_row = int(rows[local_start])
                pool_rows = rows[local_start : local_start + config.horizon]
                target_rows = rows[local_start + 1 : local_start + 1 + config.horizon]
                if len(pool_rows) != config.horizon or len(target_rows) != config.horizon:
                    raise M4Error("Incomplete M4 window escaped terminal drop policy")
                segment = episode.segment_id[current_row]
                gap_crossing_count += int(np.count_nonzero(episode.segment_id[target_rows] != segment))
                samples.append(
                    BridgeSample(
                        episode_id=episode.episode_id,
                        task=episode.task,
                        split=episode.split,
                        segment_number=segment_number,
                        phase=float(local_start / max(1, len(rows) - 1)),
                        current_row=current_row,
                        current=episode.robot[current_row].copy(),
                        human_plan=episode.human[pool_rows].copy(),
                        target=episode.robot[target_rows].copy(),
                    )
                )
    if gap_crossing_count:
        raise M4Error(f"M4 windows cross {gap_crossing_count} segment boundaries")
    summary = {
        "sample_count": len(samples),
        "train_sample_count": sum(item.split == "train" for item in samples),
        "heldout_sample_count": sum(item.split == "heldout" for item in samples),
        "heldout_task_count": len({item.task for item in samples if item.split == "heldout"}),
        "gap_crossing_count": gap_crossing_count,
        "strict_future_offset_view_steps": 1,
        "retrieval_feature_schema": ["normalized_segment_phase", "task_pool_membership"],
        "heldout_robot_trajectory_used_in_retrieval_feature": False,
        "heldout_robot_trajectory_usage": "offline target evaluation only",
    }
    if not summary["train_sample_count"] or not summary["heldout_sample_count"]:
        raise M4Error("M4 requires non-empty train and held-out samples")
    return samples, summary


def _phase_features(sample: BridgeSample) -> np.ndarray:
    return np.concatenate((sample.current, np.asarray([sample.phase, sample.phase**2], dtype=np.float64)))


def _plan_features(sample: BridgeSample, aligned_plan: np.ndarray, phase_distance: float = 0.0) -> np.ndarray:
    return np.concatenate(
        (
            aligned_plan.reshape(-1),
            sample.current,
            np.asarray([sample.phase, sample.phase**2, phase_distance], dtype=np.float64),
        )
    )


def fit_ridge(features: np.ndarray, targets: np.ndarray, alpha: float) -> RidgeModel:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise M4Error(f"Invalid ridge shapes: X={x.shape}, Y={y.shape}")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (x - mean) / scale
    design = np.concatenate((normalized, np.ones((len(normalized), 1))), axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return RidgeModel(mean=mean, scale=scale, weights=weights)


def train_baselines(samples: Sequence[BridgeSample], config: M4Config) -> dict[str, RidgeModel]:
    train = [sample for sample in samples if sample.split == "train"]
    phase_x: list[np.ndarray] = []
    phase_y: list[np.ndarray] = []
    plan_x: list[np.ndarray] = []
    direct_y: list[np.ndarray] = []
    residual_y: list[np.ndarray] = []
    for sample in train:
        aligned = align_pool_chunk(sample.human_plan, sample.current)
        phase_x.append(_phase_features(sample))
        repeated_current = np.repeat(sample.current[None], config.horizon, axis=0)
        phase_y.append((sample.target - repeated_current).reshape(-1))
        plan_x.append(_plan_features(sample, aligned))
        direct_y.append(sample.target.reshape(-1))
        residual_y.append((sample.target - aligned).reshape(-1))
    return {
        "no_retrieval": fit_ridge(np.stack(phase_x), np.stack(phase_y), config.ridge_alpha),
        "co_training": fit_ridge(np.stack(plan_x), np.stack(direct_y), config.ridge_alpha),
        "recap_hand_ret": fit_ridge(np.stack(plan_x), np.stack(residual_y), config.ridge_alpha),
    }


def _uniform_pool(pool: Sequence[BridgeSample], size: int) -> list[BridgeSample]:
    ordered = sorted(pool, key=lambda item: (item.phase, item.episode_id, item.current_row))
    if size <= 0 or size >= len(ordered):
        return ordered
    indices = np.linspace(0, len(ordered) - 1, num=size, dtype=np.int64)
    return [ordered[int(index)] for index in np.unique(indices)]


def retrieve_human_plans(
    query: BridgeSample, heldout_pool: Sequence[BridgeSample], size: int, top_k: int
) -> list[tuple[np.ndarray, float]]:
    task_pool = [item for item in heldout_pool if item.task == query.task]
    selected_pool = _uniform_pool(task_pool, size)
    if not selected_pool:
        raise M4Error(f"No human pool entries for held-out task {query.task}")
    ranked = sorted(selected_pool, key=lambda item: abs(query.phase - item.phase))[:top_k]
    return [
        (align_pool_chunk(item.human_plan, query.current), abs(query.phase - item.phase))
        for item in ranked
    ]


def _project_actions(prediction: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, int]:
    actions = np.asarray(prediction, dtype=np.float64).copy()
    actions = actions.reshape(-1, 10)
    fallback_count = 0
    for index in range(len(actions)):
        first = actions[index, 3:6]
        second = actions[index, 6:9]
        cross_norm = np.linalg.norm(np.cross(first, second))
        if np.linalg.norm(first) < 1e-8 or np.linalg.norm(second) < 1e-8 or cross_norm < 1e-8:
            actions[index, 3:9] = current[3:9]
            fallback_count += 1
        else:
            matrix = rotation_6d_to_matrix(actions[index, 3:9])
            actions[index, 3:9] = matrix_to_rotation_6d(matrix)
    actions[:, 9] = np.clip(actions[:, 9], 0.0, 1.0)
    return actions, fallback_count


def _predict_methods(
    query: BridgeSample,
    plans: Sequence[tuple[np.ndarray, float]],
    models: dict[str, RidgeModel],
    config: M4Config,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    repeated_current = np.repeat(query.current[None], config.horizon, axis=0)
    no_ret_delta = models["no_retrieval"].predict(_phase_features(query)[None])[0].reshape(config.horizon, 10)
    raw_predictions: dict[str, np.ndarray] = {"no_retrieval": repeated_current + no_ret_delta}
    retrieval_predictions: list[np.ndarray] = []
    co_predictions: list[np.ndarray] = []
    recap_predictions: list[np.ndarray] = []
    for plan, phase_distance in plans:
        features = _plan_features(query, plan, phase_distance)[None]
        retrieval_predictions.append(plan)
        co_predictions.append(models["co_training"].predict(features)[0].reshape(config.horizon, 10))
        residual = models["recap_hand_ret"].predict(features)[0].reshape(config.horizon, 10)
        recap_predictions.append(plan + residual)
    raw_predictions.update(
        {
            "retrieval_only": np.mean(retrieval_predictions, axis=0),
            "co_training": np.mean(co_predictions, axis=0),
            "recap_hand_ret": np.mean(recap_predictions, axis=0),
        }
    )
    predictions: dict[str, np.ndarray] = {}
    fallbacks: dict[str, int] = {}
    for method, prediction in raw_predictions.items():
        predictions[method], fallbacks[method] = _project_actions(prediction, query.current)
    return predictions, fallbacks


def _median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.median(array)) if len(array) else float("nan")


def _metric_summary(
    predictions: Sequence[np.ndarray], targets: Sequence[np.ndarray], fallback_count: int
) -> dict[str, Any]:
    pred = np.stack(predictions)
    target = np.stack(targets)
    position = np.linalg.norm(pred[..., :3] - target[..., :3], axis=-1)
    orientation = orientation_error_rad(pred[..., 3:9], target[..., 3:9])
    gripper = np.abs(pred[..., 9] - target[..., 9])
    canonical = np.linalg.norm(pred - target, axis=-1)
    final_position = np.linalg.norm(pred[:, -1, :3] - target[:, -1, :3], axis=-1)
    return {
        "sample_count": len(pred),
        "canonical_error_median": _median(canonical.ravel()),
        "position_error_median_canonical": _median(position.ravel()),
        "orientation_error_median_rad": _median(orientation.ravel()),
        "gripper_error_median": _median(gripper.ravel()),
        "final_position_error_median_canonical": _median(final_position),
        "rotation_projection_fallback_count": int(fallback_count),
    }


def evaluate_pool_growth(
    samples: Sequence[BridgeSample], models: dict[str, RidgeModel], config: M4Config
) -> list[dict[str, Any]]:
    queries = [sample for sample in samples if sample.split == "heldout"]
    heldout_human_pool = list(queries)
    results: list[dict[str, Any]] = []
    for requested_size in config.pool_growth_sizes:
        predictions = {method: [] for method in BASELINES}
        targets: list[np.ndarray] = []
        phase_distances: list[float] = []
        fallback_counts = {method: 0 for method in BASELINES}
        effective_sizes: list[int] = []
        for query in queries:
            task_pool = [item for item in heldout_human_pool if item.task == query.task]
            effective_sizes.append(len(_uniform_pool(task_pool, requested_size)))
            plans = retrieve_human_plans(query, heldout_human_pool, requested_size, config.retrieval_top_k)
            phase_distances.extend(distance for _plan, distance in plans)
            sample_predictions, sample_fallbacks = _predict_methods(query, plans, models, config)
            for method in BASELINES:
                predictions[method].append(sample_predictions[method])
                fallback_counts[method] += sample_fallbacks[method]
            targets.append(query.target)
        label = "all" if requested_size <= 0 else str(requested_size)
        results.append(
            {
                "requested_pool_size_per_task": label,
                "effective_pool_size_per_task_median": _median(effective_sizes),
                "retrieval_phase_error_median": _median(phase_distances),
                "methods": {
                    method: _metric_summary(predictions[method], targets, fallback_counts[method])
                    for method in BASELINES
                },
            }
        )
    return results


def _model_arrays(models: dict[str, RidgeModel]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, model in models.items():
        arrays[f"{name}__mean"] = model.mean
        arrays[f"{name}__scale"] = model.scale
        arrays[f"{name}__weights"] = model.weights
    return arrays


def _checkpoint_bindings(
    view: dict[str, Any], m3_report: dict[str, Any], config: M4Config
) -> dict[str, Any]:
    nominal_hz = view["time_view"].get("nominal_hz")
    alignment_version = view.get("action_alignment_manifest_sha256")
    return {
        "canonical_schema": view["canonical_schema"],
        "canonical_manifest_sha256": view["canonical_manifest_sha256"],
        "source_evidence_manifest_sha256": view["source_evidence_manifest_sha256"],
        "split_sha256": view["split_sha256"],
        "time_view_id": view["time_view_id"],
        "pool_action_view_id": view["pool_action_view_id"],
        "query_action_view_id": view["query_action_view_id"],
        "action_alignment_id": view["action_alignment_id"],
        "pool_action_role": view["pool_action_role"],
        "query_action_role": view["query_action_role"],
        "query_command_status": view["query_command_status"],
        "policy_coordinate": {"policy_dt": None if nominal_hz is None else 1.0 / float(nominal_hz)},
        "H_steps": view["H_steps"],
        "H_seconds": view["H_seconds"],
        "K_steps": view["K_steps"],
        "K_seconds": view["K_seconds"],
        "gap_policy": view["gap_policy"],
        "alignment_version": alignment_version,
        "view_id": view["view_id"],
        "retrieval_index_sha256": m3_report["retrieval_sanity"]["retrieval_index_sha256"],
    }


def write_checkpoint(
    models: dict[str, RidgeModel], bindings: dict[str, Any], config: M4Config
) -> tuple[Path, Path, dict[str, Any]]:
    checkpoint_path = config.output_root / "checkpoints/offline_bridge_ridge_v0.npz"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(checkpoint_path, **_model_arrays(models))
    manifest = {
        "schema_version": M4_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "bindings": bindings,
        "binding_sha256": stable_json_sha256(bindings),
        "baselines": list(BASELINES),
        "training_scope": "seen paired tasks only",
        "implementation": str(Path(__file__).resolve()),
        "implementation_code_sha256": file_sha256(Path(__file__)),
        "deployment_command_adapter_id": None,
        "m6_rollout_approved": False,
    }
    manifest_path = checkpoint_path.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    return checkpoint_path, manifest_path, manifest


def load_checkpoint(
    checkpoint_path: Path, manifest_path: Path, expected_bindings: dict[str, Any]
) -> dict[str, RidgeModel]:
    manifest = _read_json(manifest_path)
    actual_bindings = manifest.get("bindings", {})
    mismatches = {
        key: {"expected": expected_bindings.get(key), "actual": actual_bindings.get(key)}
        for key in CHECKPOINT_BINDING_KEYS
        if actual_bindings.get(key) != expected_bindings.get(key)
    }
    if mismatches:
        raise M4Error(f"Checkpoint/view binding mismatch: {mismatches}")
    if file_sha256(checkpoint_path) != manifest.get("checkpoint_sha256"):
        raise M4Error("Checkpoint content hash mismatch")
    arrays = np.load(checkpoint_path, allow_pickle=False)
    models: dict[str, RidgeModel] = {}
    for name in ("no_retrieval", "co_training", "recap_hand_ret"):
        models[name] = RidgeModel(
            mean=arrays[f"{name}__mean"],
            scale=arrays[f"{name}__scale"],
            weights=arrays[f"{name}__weights"],
        )
    return models


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    full_pool = report["pool_growth"][-1]
    lines = [
        "# M4-v03 离线 paired bridge 启动报告",
        "",
        f"日期：{report['created_at_utc']}",
        "",
        "结论：**M4-v03 离线 bridge 已启动；Gate C 仍为 pending。**",
        "",
        "本报告只确认 action-space 离线闭环、四个固定 baseline、pool-growth smoke 与 checkpoint 契约已经可运行；不批准 M6 或真实机器人 command。",
        "",
        "## 冻结契约",
        "",
        f"- time view：`{report['bindings']['time_view_id']}`",
        f"- pool action：`{report['bindings']['pool_action_view_id']}`",
        f"- query action：`{report['bindings']['query_action_view_id']}`",
        f"- alignment：`{report['bindings']['action_alignment_id']}`",
        f"- H/K：{report['bindings']['H_steps']}/{report['bindings']['K_steps']}",
        "- deployment command adapter：`null`",
        "",
        "## 数据与泄漏门禁",
        "",
        f"- train windows：{report['dataset']['train_sample_count']}",
        f"- held-out windows：{report['dataset']['heldout_sample_count']}",
        f"- gap crossing：{report['dataset']['gap_crossing_count']}",
        "- retrieval feature：normalized segment phase + task pool membership",
        "- held-out robot trajectory：只用于离线 target 评测",
        "",
        "## 全池 smoke 指标",
        "",
        "| 方法 | position median | orientation median rad | gripper median |",
        "|---|---:|---:|---:|",
    ]
    for method in BASELINES:
        metrics = full_pool["methods"][method]
        lines.append(
            f"| `{method}` | {metrics['position_error_median_canonical']:.6f} | "
            f"{metrics['orientation_error_median_rad']:.6f} | {metrics['gripper_error_median']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 当前边界",
            "",
            "- 当前模型是确定性 ridge smoke bridge，用于打通数据、baseline、评测和 checkpoint 契约，不是最终 Cosmos/RECAP 主训练配置。",
            "- pilot 每个 held-out task 只有一个 paired episode；human-only pool 与 robot target 来自同一发布 pair，但 retrieval 代码不读取 held-out robot target。正式 Gate C 需要独立/扩充 human pool 与多 seed 训练。",
            "- pool-growth 是否总体改善、RECAP 是否稳定优于 No retrieval/Retrieval Only，必须在正式训练后验收；本报告不提前通过 Gate C。",
            "- `query_command_status=unverified`，不得用于真实机器人执行。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_m4_launch(config: M4Config = M4Config()) -> dict[str, Any]:
    view, m3_report, split = validate_m4_contract(config)
    m3_config = M3Config(
        canonical_root=config.canonical_root,
        expected_episode_count=config.expected_episode_count,
    )
    episodes, loaded_split, _preprocessing = load_episodes(m3_config)
    if loaded_split.get("split_sha256") != split.get("split_sha256"):
        raise M4Error("Loaded split changed after contract validation")
    samples, dataset_summary = build_bridge_samples(episodes, _time_view_spec(view), config)
    models = train_baselines(samples, config)
    pool_growth = evaluate_pool_growth(samples, models, config)
    bindings = _checkpoint_bindings(view, m3_report, config)
    checkpoint_path, manifest_path, checkpoint_manifest = write_checkpoint(models, bindings, config)
    load_checkpoint(checkpoint_path, manifest_path, bindings)

    paired_config = {
        "schema_version": M4_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "bindings": bindings,
        "baselines": list(BASELINES),
        "model_family": "deterministic_ridge_smoke_bridge",
        "implementation": str(Path(__file__).resolve()),
        "implementation_code_sha256": file_sha256(Path(__file__)),
        "train_scope": "seen paired tasks",
        "heldout_pool_scope": "heldout human plans only; robot trajectory evaluation only",
        "retrieval_feature_schema": dataset_summary["retrieval_feature_schema"],
        "pool_growth_sizes": ["all" if size <= 0 else size for size in config.pool_growth_sizes],
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "deployment_command_adapter_id": None,
    }
    paired_config["config_sha256"] = stable_json_sha256(paired_config)
    write_json(config.output_root / "paired_bridge_config.json", paired_config)
    write_json(config.output_root / "dataset_summary.json", dataset_summary)
    write_json(config.output_root / "pool_growth_smoke.json", {"results": pool_growth})
    safety_smoke = {
        "schema_version": M4_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "prediction_metrics_finite": all(
            np.isfinite(metric_value)
            for item in pool_growth
            for metrics in item["methods"].values()
            for metric_name, metric_value in metrics.items()
            if metric_name.endswith("median")
        ),
        "rotation_projection_fallback_count": sum(
            metrics["rotation_projection_fallback_count"]
            for item in pool_growth
            for metrics in item["methods"].values()
        ),
        "gap_crossing_count": dataset_summary["gap_crossing_count"],
        "gripper_postprocess": "clip_to_[0,1]",
        "workspace_clipping": "disabled_in_smoke; report prediction error in canonical coordinates",
        "query_command_status": bindings["query_command_status"],
        "deployment_command_adapter_id": None,
        "m6_rollout_approved": False,
    }
    write_json(config.output_root / "safety_smoke.json", safety_smoke)

    launch_checks = {
        "m3_gate_b_passed": True,
        "contract_bound": True,
        "four_fixed_baselines_ran": all(set(item["methods"]) == set(BASELINES) for item in pool_growth),
        "gap_crossing_zero": dataset_summary["gap_crossing_count"] == 0,
        "heldout_target_not_in_retrieval_feature": not dataset_summary[
            "heldout_robot_trajectory_used_in_retrieval_feature"
        ],
        "checkpoint_reload_hard_binding_passed": True,
        "deployment_command_adapter_absent": checkpoint_manifest["deployment_command_adapter_id"] is None,
    }
    status = "launched" if all(launch_checks.values()) else "failed"
    report = {
        "schema_version": M4_SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": status,
        "decision": "M4-v03 离线 bridge 已启动；Gate C 保持 pending" if status == "launched" else "M4 启动失败",
        "gate_c": "pending",
        "launch_checks": launch_checks,
        "bindings": bindings,
        "dataset": dataset_summary,
        "pool_growth": pool_growth,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_manifest_path": str(manifest_path),
        "safety_smoke": safety_smoke,
        "m3_report_sha256": m3_report.get("report_sha256"),
        "limitations": [
            "ridge smoke bridge is not the final Cosmos/RECAP training model",
            "one paired episode per heldout task; independent human-pool replication is pending",
            "multi-seed training and formal Gate C comparisons are pending",
            "query command status is unverified; M6 rollout is forbidden",
        ],
        "deployment_command_adapter_id": None,
        "m6_rollout_approved": False,
    }
    report["report_sha256"] = stable_json_sha256(report)
    write_json(config.output_root / "m4_launch_report.json", report)
    write_json(config.report_root / "M4_offline_bridge_自动启动报告.json", report)
    _write_markdown_report(config.report_root / "M4_offline_bridge_启动报告.md", report)
    return report
