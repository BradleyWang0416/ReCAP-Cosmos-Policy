"""Pure learned-cell specifications for the frozen 48-checkpoint P2 registry."""

from __future__ import annotations

from dataclasses import dataclass, replace

FORMAL_SEEDS = (20260711, 20260712, 20260713)


@dataclass(frozen=True)
class P2TrainingSpec:
    experiment_id: str
    variant_id: str
    method_id: str
    seed: int
    target_representation: str
    retrieval_modality: str = "phase"
    time_view_id: str = "nominal_camera_30hz_segmented"
    h_steps: int = 8
    k_steps: int = 8
    top_k: int = 3
    pool_size: int = 10
    query_offset_view_steps: int = 1

    @property
    def cell_id(self) -> str:
        return (
            f"learned_training_checkpoint__{self.experiment_id}__{self.variant_id}__"
            f"{self.method_id}__seed{self.seed}"
        )

    @property
    def config_name(self) -> str:
        if self.experiment_id == "M5B-MAIN-01" and self.variant_id == "frozen_main":
            return f"cosmos_predict2p5_2b_human2robot_{self.method_id}_seed{self.seed}"
        experiment = self.experiment_id.lower().replace("-", "_")
        return (
            f"cosmos_predict2p5_2b_human2robot_p2_{experiment}_{self.variant_id}_"
            f"{self.method_id}_seed{self.seed}"
        )

    @property
    def state_t(self) -> int:
        return 8 + self.h_steps // 4

    @property
    def action_latent_idx(self) -> int:
        return 5 + self.h_steps // 4

    @property
    def tokenizer_chunk_duration(self) -> int:
        return 29 + self.h_steps


def _seeded(template: P2TrainingSpec) -> list[P2TrainingSpec]:
    return [replace(template, seed=seed) for seed in FORMAL_SEEDS]


def p2_training_specs() -> list[P2TrainingSpec]:
    templates = [
        P2TrainingSpec("M5B-MAIN-01", "frozen_main", "no_retrieval", 0, "absolute"),
        P2TrainingSpec("M5B-MAIN-01", "frozen_main", "co_training", 0, "absolute"),
        P2TrainingSpec("M5B-MAIN-01", "frozen_main", "recap_hand_ret", 0, "residual"),
        P2TrainingSpec("M5B-REP-01", "future_state", "recap_hand_ret", 0, "future_state"),
        P2TrainingSpec(
            "M5B-ACTION-01",
            "phase_aligned_human_plan_plus_tplus1_query",
            "recap_hand_ret",
            0,
            "residual",
            time_view_id="phase_or_dtw",
        ),
        P2TrainingSpec(
            "M5B-ACTION-01",
            "raw_human_plan_plus_lag_calibrated_query_diagnostic",
            "recap_hand_ret",
            0,
            "residual",
            query_offset_view_steps=5,
        ),
        P2TrainingSpec("M5B-RET-01", "random", "recap_hand_ret", 0, "residual", "random"),
        P2TrainingSpec("M5B-RET-01", "geometry", "recap_hand_ret", 0, "residual", "geometry"),
        P2TrainingSpec("M5B-RET-01", "visual", "recap_hand_ret", 0, "residual", "visual"),
        P2TrainingSpec(
            "M5B-RET-01",
            "geometry_plus_visual",
            "recap_hand_ret",
            0,
            "residual",
            "geometry_plus_visual",
        ),
        P2TrainingSpec(
            "M5B-SENS-01", "topk3_h4_k4", "recap_hand_ret", 0, "residual", h_steps=4, k_steps=4
        ),
        P2TrainingSpec(
            "M5B-SENS-01", "topk3_h16_k8", "recap_hand_ret", 0, "residual", h_steps=16, k_steps=8
        ),
        P2TrainingSpec(
            "M5B-TIME-01",
            "paper_v2_stride4_nominal7p5",
            "recap_hand_ret",
            0,
            "residual",
            time_view_id="paper_v2_stride4_nominal7p5",
        ),
        P2TrainingSpec(
            "M5B-TIME-01",
            "legacy_v01_stride3_nominal10",
            "recap_hand_ret",
            0,
            "residual",
            time_view_id="legacy_v01_stride3_nominal10",
        ),
        P2TrainingSpec(
            "M5B-TIME-01",
            "policy_clock_10hz",
            "recap_hand_ret",
            0,
            "residual",
            time_view_id="policy_clock_10hz",
        ),
        P2TrainingSpec(
            "M5B-TIME-01",
            "phase_or_dtw",
            "recap_hand_ret",
            0,
            "residual",
            time_view_id="phase_or_dtw",
        ),
    ]
    result = [spec for template in templates for spec in _seeded(template)]
    if len(result) != 48 or len({item.cell_id for item in result}) != 48:
        raise RuntimeError("Frozen P2 learned-cell spec cardinality changed")
    return result


def training_spec_by_cell_id() -> dict[str, P2TrainingSpec]:
    return {spec.cell_id: spec for spec in p2_training_specs()}


def training_spec_by_config_name() -> dict[str, P2TrainingSpec]:
    result = {spec.config_name: spec for spec in p2_training_specs()}
    if len(result) != 48:
        raise RuntimeError("Frozen P2 config names are not unique")
    return result

