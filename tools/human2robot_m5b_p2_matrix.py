#!/usr/bin/env python3
"""Fail-closed binding for the frozen 203-cell M5B-P2 successor matrix.

This module does not launch training.  It turns the frozen registry and the
materialized 48 learned-cell inputs into an executable DAG contract.  Every
runtime handler must consume these bindings rather than reconstructing cell
semantics from ad-hoc command-line defaults.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from cosmos_policy.datasets.human2robot_p2_specs import P2TrainingSpec, p2_training_specs

REGISTRY_RELATIVE_PATH = Path("方案/v03/M5B_P2_cell_registry_v2.json")
SUPPLEMENT_RELATIVE_PATH = Path("方案/v03/M5B_P2_execution_supplement_v2.json")
PREPARED_MANIFEST_RELATIVE_PATH = Path(
    "data/Human2Robot/derived/m5b_v03/p2_prepared_v2/prepared_manifest.json"
)
WORKSPACE_BOUNDS_RELATIVE_PATH = Path("方案/v03/M5B_P2_workspace_bounds_v1.json")
LAG_VIEW_MANIFEST_RELATIVE_PATH = Path(
    "data/Human2Robot/derived/views/nominal_camera_30hz_segmented/"
    "human_hand_robot_frame_raw/robot_ee_observed_t_plus_5_lag_diagnostic/"
    "train_only_tplus5_query_anchor_se3_identity_scale_v1/view_manifest.json"
)

REGISTRY_SHA256 = "502cc57d41c7e4829e872ac95a258d7dc1e8d0d8a27ddfc3cf0315d4d31ef2d6"
REGISTRY_CELLS_SHA256 = "cea1bbc669ff02e7c22f3511b84a136a255ea27dae60a4356876d8cd74b3be12"
SUPPLEMENT_SHA256 = "17d9fc308c50b9b7899793a4c8d3bca1eeba217053fbacb368e2f9a2e390d7ab"
PREPARED_MANIFEST_SHA256 = (
    "15a1bd6cc378079b04a821fe691fe293739acc827e183caa44633b76b6a629cd"
)
WORKSPACE_BOUNDS_SHA256 = "29e0fd8d4b58beabcf7cea7ba50488a0775a79b6f429596a3573a0bbb007eb6a"
LAG_VIEW_MANIFEST_SHA256 = "53ab59227f865767f07fd4b8c6cea52689b7c22ec1359cedb975308644fe806d"
PROTOCOL_SHA256 = "7598dfa2ac2e129f5d21a295dad23b90f63c3c8e68811da73cbcc20eb95d5ce4"
SPLIT_SHA256 = "1d3ef2377aa19938b06646f6d5fc31ec9f275fc9f37e253e1e9aa5eecdc5a968"
POOL_MANIFEST_SHA256 = "47e87be5800194de6e0ac99b47dbe23ef96a91298edbff3e9996b1484b489299"
TOKENIZER_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"

FORMAL_SEEDS = (20260711, 20260712, 20260713)
FROZEN_COUNTS = {
    "learned_training_checkpoint": 48,
    "nonlearned_method_artifact": 3,
    "checkpoint_linked_evaluation": 147,
    "aggregate_report": 5,
}
ACTION_NEGATIVE_CONTROLS = {
    "same_frame_query_negative_control": "same_frame_query_detector",
    "swapped_role_negative_control": "swapped_role_detector",
    "scale_x2_negative_control": "scale_x2_detector",
}
RESOLUTION_VARIANTS = {
    "source_240x426_then_resize_224",
    "center_crop_240x424_then_resize_224",
    "center_crop_240x424_edge_pad_240x426_then_resize_224",
}


class MatrixContractError(RuntimeError):
    """Raised when frozen execution semantics cannot be bound exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatrixContractError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FrozenCell:
    cell_id: str
    artifact_kind: str
    experiment_id: str
    variant_id: str
    method_id: str | None
    seed: int | None
    parent_artifact_ids: tuple[str, ...]
    optimizer_steps: int | None
    formal_result: bool
    status: str

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "FrozenCell":
        expected = {
            "cell_id",
            "artifact_kind",
            "experiment_id",
            "variant_id",
            "method_id",
            "seed",
            "parent_artifact_ids",
            "optimizer_steps",
            "formal_result",
            "status",
        }
        _require(set(record) == expected, f"Unexpected registry cell fields: {set(record) ^ expected}")
        parents = record["parent_artifact_ids"]
        _require(isinstance(parents, list), f"parent_artifact_ids must be a list: {record['cell_id']}")
        return cls(
            cell_id=str(record["cell_id"]),
            artifact_kind=str(record["artifact_kind"]),
            experiment_id=str(record["experiment_id"]),
            variant_id=str(record["variant_id"]),
            method_id=None if record["method_id"] is None else str(record["method_id"]),
            seed=None if record["seed"] is None else int(record["seed"]),
            parent_artifact_ids=tuple(str(item) for item in parents),
            optimizer_steps=(
                None if record["optimizer_steps"] is None else int(record["optimizer_steps"])
            ),
            formal_result=bool(record["formal_result"]),
            status=str(record["status"]),
        )


@dataclass(frozen=True)
class EvaluationBinding:
    cell_id: str
    experiment_id: str
    variant_id: str
    method_id: str
    run_seed: int
    parent_artifact_id: str
    checkpoint_cell_id: str | None
    prepared_input_cell_id: str
    target_representation: str
    retrieval_modality: str
    time_view_id: str
    h_steps: int
    k_steps: int
    top_k: int
    pool_size: int
    query_offset_view_steps: int
    resolution_variant: str
    corruption_id: str | None
    corruption_severity: str | None
    negative_control_detector: str | None
    requires_model_inference: bool


@dataclass(frozen=True)
class CellBinding:
    cell: FrozenCell
    handler_kind: str
    prepared_entry: Mapping[str, Any] | None = None
    training_spec: P2TrainingSpec | None = None
    evaluation: EvaluationBinding | None = None


@dataclass(frozen=True)
class ExecutionMatrix:
    registry: Mapping[str, Any]
    prepared_manifest: Mapping[str, Any]
    cells_by_id: Mapping[str, FrozenCell]
    bindings_by_id: Mapping[str, CellBinding]
    topological_cell_ids: tuple[str, ...]
    formal_readiness_blockers: tuple[str, ...]
    report_covered_evaluation_ids: frozenset[str]

    def cells_of_kind(self, artifact_kind: str) -> tuple[CellBinding, ...]:
        return tuple(
            self.bindings_by_id[cell_id]
            for cell_id in self.topological_cell_ids
            if self.cells_by_id[cell_id].artifact_kind == artifact_kind
        )


def _expected_cell_id(cell: FrozenCell) -> str:
    parts = [cell.artifact_kind, cell.experiment_id, cell.variant_id]
    if cell.method_id is not None:
        parts.append(cell.method_id)
    if cell.seed is not None:
        parts.append(f"seed{cell.seed}")
    return "__".join(parts)


def _topological_order(cells_by_id: Mapping[str, FrozenCell]) -> tuple[str, ...]:
    children: dict[str, list[str]] = {cell_id: [] for cell_id in cells_by_id}
    indegree = {cell_id: 0 for cell_id in cells_by_id}
    for cell in cells_by_id.values():
        for parent_id in cell.parent_artifact_ids:
            _require(parent_id in cells_by_id, f"Unknown parent {parent_id} for {cell.cell_id}")
            _require(parent_id != cell.cell_id, f"Self dependency: {cell.cell_id}")
            children[parent_id].append(cell.cell_id)
            indegree[cell.cell_id] += 1
    queue = deque(cell_id for cell_id in cells_by_id if indegree[cell_id] == 0)
    ordered: list[str] = []
    while queue:
        cell_id = queue.popleft()
        ordered.append(cell_id)
        for child_id in children[cell_id]:
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                queue.append(child_id)
    _require(len(ordered) == len(cells_by_id), "Frozen registry contains a dependency cycle")
    return tuple(ordered)


def load_frozen_registry(workspace: Path) -> tuple[dict[str, Any], dict[str, FrozenCell], tuple[str, ...]]:
    path = workspace / REGISTRY_RELATIVE_PATH
    _require(file_sha256(path) == REGISTRY_SHA256, "Frozen registry file SHA256 changed")
    registry = _read_json(path)
    _require(registry.get("schema_version") == "human2robot-m5b-p2-cell-registry-v2", "Registry schema changed")
    _require(registry.get("registry_id") == "m5b_p2_claim_centered_203_cells_v2", "Registry id changed")
    _require(registry.get("status") == "frozen_pending_execution", "Registry is not frozen pending execution")
    _require(registry.get("formal_queue_allowed") is False, "Frozen registry unexpectedly opens formal queue")
    _require(registry.get("p2_acceptance_allowed") is False, "Frozen registry unexpectedly permits P2 acceptance")
    _require(tuple(registry.get("seeds", ())) == FORMAL_SEEDS, "Frozen seeds changed")
    _require(registry.get("cell_count") == 203, "Frozen registry cell count changed")
    _require(registry.get("counts") == FROZEN_COUNTS, "Frozen artifact-kind counts changed")
    records = registry.get("cells")
    _require(isinstance(records, list), "Registry cells must be a list")
    _require(canonical_json_sha256(records) == REGISTRY_CELLS_SHA256, "Frozen cells payload changed")
    _require(registry.get("cells_payload_sha256") == REGISTRY_CELLS_SHA256, "Registry cells hash binding changed")
    cells = [FrozenCell.from_record(record) for record in records]
    cells_by_id = {cell.cell_id: cell for cell in cells}
    _require(len(cells_by_id) == 203, "Registry cell ids are not unique")
    _require(Counter(cell.artifact_kind for cell in cells) == Counter(FROZEN_COUNTS), "Cell kinds do not match frozen counts")
    for cell in cells:
        _require(cell.cell_id == _expected_cell_id(cell), f"Noncanonical cell id: {cell.cell_id}")
        _require(cell.formal_result is False, f"Pending registry cell claims formal evidence: {cell.cell_id}")
        _require(cell.status == "pending", f"Frozen cell is not pending: {cell.cell_id}")
        if cell.artifact_kind == "learned_training_checkpoint":
            _require(cell.optimizer_steps == 7000, f"Training step contract changed: {cell.cell_id}")
            _require(not cell.parent_artifact_ids, f"Training cell unexpectedly has parents: {cell.cell_id}")
        else:
            _require(cell.optimizer_steps is None, f"Non-training cell has optimizer steps: {cell.cell_id}")
        if cell.artifact_kind == "checkpoint_linked_evaluation":
            _require(len(cell.parent_artifact_ids) == 1, f"Evaluation must have one artifact parent: {cell.cell_id}")
    order = _topological_order(cells_by_id)
    return registry, cells_by_id, order


def _validate_contract_against_spec(contract: Mapping[str, Any], spec: P2TrainingSpec, split: str) -> None:
    expected = {
        "experiment_id": spec.experiment_id,
        "variant_id": spec.variant_id,
        "method_id": spec.method_id,
        "seed": spec.seed,
        "target_representation": spec.target_representation,
        "retrieval_modality": spec.retrieval_modality,
        "time_view_id": spec.time_view_id,
        "H_steps": spec.h_steps,
        "K_steps": spec.k_steps,
        "top_k": spec.top_k,
        "pool_size": spec.pool_size,
        "query_offset_view_steps": spec.query_offset_view_steps,
        "split": split,
        "protocol_file_sha256": PROTOCOL_SHA256,
        "supplement_file_sha256": SUPPLEMENT_SHA256,
        "split_sha256": SPLIT_SHA256,
        "heldout_target_retrieval_feature_count": 0,
    }
    for key, value in expected.items():
        _require(contract.get(key) == value, f"Prepared {split} contract mismatch for {spec.cell_id}: {key}")


def load_prepared_manifest(
    workspace: Path,
    *,
    verify_artifact_hashes: bool = True,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    path = workspace / PREPARED_MANIFEST_RELATIVE_PATH
    _require(file_sha256(path) == PREPARED_MANIFEST_SHA256, "Prepared manifest SHA256 changed")
    manifest = _read_json(path)
    expected_top_level = {
        "schema_version": "human2robot-m5b-p2-prepared-artifacts-v2",
        "status": "complete",
        "formal_result": False,
        "learned_cell_count": 48,
        "heldout_target_retrieval_feature_count": 0,
        "protocol_file_sha256": PROTOCOL_SHA256,
        "supplement_file_sha256": SUPPLEMENT_SHA256,
        "registry_file_sha256": REGISTRY_SHA256,
        "split_sha256": SPLIT_SHA256,
        "pool_manifest_sha256": POOL_MANIFEST_SHA256,
        "tokenizer_checkpoint_sha256": TOKENIZER_SHA256,
    }
    for key, value in expected_top_level.items():
        _require(manifest.get(key) == value, f"Prepared manifest binding mismatch: {key}")
    entries = manifest.get("entries")
    _require(isinstance(entries, list) and len(entries) == 48, "Prepared manifest must contain 48 entries")
    entries_by_id = {str(entry.get("cell_id")): entry for entry in entries}
    _require(len(entries_by_id) == 48, "Prepared manifest cell ids are not unique")

    specs_by_id = {spec.cell_id: spec for spec in p2_training_specs()}
    _require(set(entries_by_id) == set(specs_by_id), "Prepared entries do not match the 48 learned specs")
    for cell_id, spec in specs_by_id.items():
        entry = entries_by_id[cell_id]
        _require(entry.get("config_name") == spec.config_name, f"Prepared config mismatch: {cell_id}")
        _require(entry.get("spec") == asdict(spec), f"Prepared spec mismatch: {cell_id}")
        train_contract = entry.get("train_contract")
        heldout_contract = entry.get("heldout_contract")
        _require(isinstance(train_contract, dict), f"Missing train contract: {cell_id}")
        _require(isinstance(heldout_contract, dict), f"Missing heldout contract: {cell_id}")
        _validate_contract_against_spec(train_contract, spec, "train")
        _validate_contract_against_spec(heldout_contract, spec, "heldout")
        _require(train_contract.get("query_count", 0) > 0, f"Empty train query set: {cell_id}")
        _require(heldout_contract.get("query_count", 0) > 0, f"Empty heldout query set: {cell_id}")
        if verify_artifact_hashes:
            for path_key, sha_key in (
                ("retrieval_index_path", "retrieval_index_sha256"),
                ("statistics_path", "statistics_sha256"),
            ):
                artifact_path = workspace / str(entry[path_key])
                _require(artifact_path.is_file(), f"Missing prepared artifact: {artifact_path}")
                _require(
                    file_sha256(artifact_path) == entry[sha_key],
                    f"Prepared artifact hash mismatch: {artifact_path}",
                )
    return manifest, entries_by_id


def _main_recap_cell_id(seed: int) -> str:
    return f"learned_training_checkpoint__M5B-MAIN-01__frozen_main__recap_hand_ret__seed{seed}"


def _split_corruption(variant_id: str) -> tuple[str | None, str | None]:
    patterns = (
        (r"^(frame_drop)_(5pct|10pct|20pct)$", None),
        (r"^(timestamp_jitter)_(5ms|10ms|20ms)$", None),
        (r"^(pause)_(0p2s|0p5s|1p0s)$", None),
        (r"^(step_jump)_(1|5|20)$", None),
    )
    for pattern, _ in patterns:
        match = re.match(pattern, variant_id)
        if match:
            return match.group(1), match.group(2)
    return None, None


def _evaluation_binding(
    cell: FrozenCell,
    cells_by_id: Mapping[str, FrozenCell],
    specs_by_id: Mapping[str, P2TrainingSpec],
) -> EvaluationBinding:
    _require(cell.seed is not None and cell.method_id is not None, f"Incomplete evaluation identity: {cell.cell_id}")
    parent_id = cell.parent_artifact_ids[0]
    parent = cells_by_id[parent_id]
    if parent.artifact_kind == "learned_training_checkpoint":
        _require(parent_id in specs_by_id, f"Evaluation parent has no training spec: {cell.cell_id}")
        source_spec = specs_by_id[parent_id]
        checkpoint_cell_id: str | None = parent_id
        prepared_input_cell_id = parent_id
        requires_model = True
    else:
        _require(
            parent.artifact_kind == "nonlearned_method_artifact" and cell.method_id == "retrieval_only",
            f"Unsupported non-checkpoint evaluation parent: {cell.cell_id}",
        )
        prepared_input_cell_id = _main_recap_cell_id(cell.seed)
        source_spec = specs_by_id[prepared_input_cell_id]
        checkpoint_cell_id = None
        requires_model = False

    values: dict[str, Any] = {
        "target_representation": source_spec.target_representation,
        "retrieval_modality": source_spec.retrieval_modality,
        "time_view_id": source_spec.time_view_id,
        "h_steps": source_spec.h_steps,
        "k_steps": source_spec.k_steps,
        "top_k": source_spec.top_k,
        "pool_size": source_spec.pool_size,
        "query_offset_view_steps": source_spec.query_offset_view_steps,
        "resolution_variant": "center_crop_240x424_then_resize_224",
    }
    corruption_id, corruption_severity = _split_corruption(cell.variant_id)
    detector = ACTION_NEGATIVE_CONTROLS.get(cell.variant_id)

    if cell.experiment_id == "M5B-MAIN-01":
        if cell.variant_id == "main_comparison_pool10":
            values["pool_size"] = 10
        elif cell.variant_id.startswith("pool_growth_pool"):
            values["pool_size"] = int(cell.variant_id.removeprefix("pool_growth_pool"))
        else:
            raise MatrixContractError(f"Unknown MAIN evaluation variant: {cell.cell_id}")
    elif cell.experiment_id == "M5B-REP-01":
        _require(cell.variant_id in {"residual", "absolute", "future_state"}, f"Unknown REP variant: {cell.cell_id}")
        values["target_representation"] = cell.variant_id
    elif cell.experiment_id == "M5B-ACTION-01":
        allowed = {
            "raw_human_plan_plus_tplus1_query_main",
            "phase_aligned_human_plan_plus_tplus1_query",
            "raw_human_plan_plus_lag_calibrated_query_diagnostic",
            *ACTION_NEGATIVE_CONTROLS,
        }
        _require(cell.variant_id in allowed, f"Unknown ACTION variant: {cell.cell_id}")
    elif cell.experiment_id == "M5B-RET-01":
        _require(cell.variant_id in {"random", "phase", "geometry", "visual", "geometry_plus_visual"}, f"Unknown RET variant: {cell.cell_id}")
        values["retrieval_modality"] = cell.variant_id
    elif cell.experiment_id == "M5B-SENS-01":
        match = re.fullmatch(r"topk(1|3|5|10)_h(4|8|16)_k(4|8)", cell.variant_id)
        _require(match is not None, f"Unknown SENS variant: {cell.cell_id}")
        values["top_k"], values["h_steps"], values["k_steps"] = map(int, match.groups())
    elif cell.experiment_id == "M5B-TIME-01":
        if cell.variant_id.startswith("time_view_"):
            values["time_view_id"] = cell.variant_id.removeprefix("time_view_")
        else:
            _require(corruption_id is not None, f"Unknown TIME variant: {cell.cell_id}")
    elif cell.experiment_id == "M5B-RES-01":
        _require(cell.variant_id in RESOLUTION_VARIANTS, f"Unknown RES variant: {cell.cell_id}")
        values["resolution_variant"] = cell.variant_id
    else:
        raise MatrixContractError(f"No evaluation handler family for {cell.cell_id}")

    if cell.method_id == "retrieval_only":
        values["target_representation"] = "retrieval_only"

    return EvaluationBinding(
        cell_id=cell.cell_id,
        experiment_id=cell.experiment_id,
        variant_id=cell.variant_id,
        method_id=cell.method_id,
        run_seed=cell.seed,
        parent_artifact_id=parent_id,
        checkpoint_cell_id=checkpoint_cell_id,
        prepared_input_cell_id=prepared_input_cell_id,
        corruption_id=corruption_id,
        corruption_severity=corruption_severity,
        negative_control_detector=detector,
        requires_model_inference=requires_model,
        **values,
    )


def _transitive_report_evaluations(
    report_ids: set[str],
    cells_by_id: Mapping[str, FrozenCell],
) -> frozenset[str]:
    seen: set[str] = set()
    evaluations: set[str] = set()
    stack = list(report_ids)
    while stack:
        cell_id = stack.pop()
        if cell_id in seen:
            continue
        seen.add(cell_id)
        cell = cells_by_id[cell_id]
        if cell.artifact_kind == "checkpoint_linked_evaluation":
            evaluations.add(cell_id)
        stack.extend(cell.parent_artifact_ids)
    return frozenset(evaluations)


def load_execution_matrix(
    workspace: Path | None = None,
    *,
    verify_prepared_artifact_hashes: bool = True,
) -> ExecutionMatrix:
    workspace = workspace or Path(__file__).resolve().parents[1]
    _require(file_sha256(workspace / SUPPLEMENT_RELATIVE_PATH) == SUPPLEMENT_SHA256, "Frozen supplement SHA256 changed")
    registry, cells_by_id, order = load_frozen_registry(workspace)
    prepared, entries_by_id = load_prepared_manifest(
        workspace, verify_artifact_hashes=verify_prepared_artifact_hashes
    )
    specs_by_id = {spec.cell_id: spec for spec in p2_training_specs()}
    learned_registry_ids = {
        cell.cell_id
        for cell in cells_by_id.values()
        if cell.artifact_kind == "learned_training_checkpoint"
    }
    _require(learned_registry_ids == set(specs_by_id), "Training specs do not exactly cover frozen learned cells")

    bindings: dict[str, CellBinding] = {}
    for cell_id in order:
        cell = cells_by_id[cell_id]
        if cell.artifact_kind == "learned_training_checkpoint":
            bindings[cell_id] = CellBinding(
                cell=cell,
                handler_kind="train_step7000_checkpoint",
                prepared_entry=entries_by_id[cell_id],
                training_spec=specs_by_id[cell_id],
            )
        elif cell.artifact_kind == "nonlearned_method_artifact":
            _require(cell.seed is not None, f"Nonlearned artifact missing seed: {cell_id}")
            prepared_id = _main_recap_cell_id(cell.seed)
            bindings[cell_id] = CellBinding(
                cell=cell,
                handler_kind="retrieval_only_projection_artifact",
                prepared_entry=entries_by_id[prepared_id],
                training_spec=specs_by_id[prepared_id],
            )
        elif cell.artifact_kind == "checkpoint_linked_evaluation":
            evaluation = _evaluation_binding(cell, cells_by_id, specs_by_id)
            bindings[cell_id] = CellBinding(
                cell=cell,
                handler_kind="heldout_checkpoint_evaluation",
                prepared_entry=entries_by_id[evaluation.prepared_input_cell_id],
                training_spec=specs_by_id[evaluation.prepared_input_cell_id],
                evaluation=evaluation,
            )
        elif cell.artifact_kind == "aggregate_report":
            handler = (
                "full_matrix_completion_report_builder"
                if cell.variant_id == "full_matrix_completion_acceptance"
                else "qualitative_report_builder"
            )
            bindings[cell_id] = CellBinding(cell=cell, handler_kind=handler)
        else:
            raise MatrixContractError(f"No handler kind for {cell_id}")

    report_ids = {
        cell_id for cell_id, cell in cells_by_id.items() if cell.artifact_kind == "aggregate_report"
    }
    covered = _transitive_report_evaluations(report_ids, cells_by_id)
    all_evaluations = {
        cell_id
        for cell_id, cell in cells_by_id.items()
        if cell.artifact_kind == "checkpoint_linked_evaluation"
    }
    _require(covered == all_evaluations, "Terminal completion report does not cover all evaluations")
    bounds_path = workspace / WORKSPACE_BOUNDS_RELATIVE_PATH
    _require(file_sha256(bounds_path) == WORKSPACE_BOUNDS_SHA256, "Workspace bounds hash changed")
    bounds = _read_json(bounds_path)
    _require(bounds.get("status") == "frozen", "Workspace bounds are not frozen")
    _require(bounds.get("heldout_data_used") is False, "Workspace bounds use heldout data")
    lag_path = workspace / LAG_VIEW_MANIFEST_RELATIVE_PATH
    _require(file_sha256(lag_path) == LAG_VIEW_MANIFEST_SHA256, "Lag view manifest hash changed")
    lag = _read_json(lag_path)
    _require(lag.get("query_target_offset_view_steps") == 5, "Lag view offset changed")
    _require(lag.get("materialization", {}).get("gap_crossing_count") == 0, "Lag view crosses a segment")
    blockers: list[str] = []
    return ExecutionMatrix(
        registry=registry,
        prepared_manifest=prepared,
        cells_by_id=cells_by_id,
        bindings_by_id=bindings,
        topological_cell_ids=order,
        formal_readiness_blockers=tuple(blockers),
        report_covered_evaluation_ids=covered,
    )


if __name__ == "__main__":
    matrix = load_execution_matrix()
    print(
        json.dumps(
            {
                "cell_count": len(matrix.bindings_by_id),
                "counts": dict(Counter(item.cell.artifact_kind for item in matrix.bindings_by_id.values())),
                "formal_readiness_blockers": matrix.formal_readiness_blockers,
                "report_covered_evaluation_count": len(matrix.report_covered_evaluation_ids),
            },
            indent=2,
            sort_keys=True,
        )
    )
