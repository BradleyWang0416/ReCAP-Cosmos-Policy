"""Batch contracts and a trainable one-batch probe for Human2Robot P0."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Human2RobotBatchError(RuntimeError):
    """Raised when a batch is incompatible with the formal 2B adapter."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Human2RobotBatchError(message)


def _shape(value: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def _batched_shape(value: torch.Tensor, unbatched: tuple[int, ...]) -> bool:
    return _shape(value) == unbatched or (_shape(value)[1:] == unbatched if value.ndim else False)


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def validate_human2robot_batch(batch: dict[str, Any]) -> dict[str, bool]:
    required = {
        "video",
        "actions",
        "proprio",
        "future_proprio",
        "retrieved_actions",
        "retrieved_proprio",
        "action_latent_idx",
        "retrieved_action_latent_idx",
        "retrieved_state_latent_idx",
        "current_image_latent_idx",
        "current_proprio_latent_idx",
        "future_image_latent_idx",
        "future_proprio_latent_idx",
        "method_id",
        "target_representation",
        "protocol_file_sha256",
        "query_command_status",
        "deployment_command_adapter_id",
        "strict_future_offset_view_steps",
        "gap_crossing_count",
    }
    missing = sorted(required - set(batch))
    _require(not missing, f"Missing formal batch keys: {missing}")
    video = batch["video"]
    actions = batch["actions"]
    retrieved = batch["retrieved_actions"]
    proprio = batch["proprio"]
    future_proprio = batch["future_proprio"]
    retrieved_proprio = batch["retrieved_proprio"]
    _require(isinstance(video, torch.Tensor), "video must be a torch.Tensor")
    _require(_batched_shape(video, (3, 37, 224, 224)), f"Invalid video shape: {_shape(video)}")
    _require(video.dtype == torch.uint8, f"Invalid video dtype: {video.dtype}; Predict2.5 requires uint8")
    _require(_batched_shape(actions, (8, 10)), f"Invalid action shape: {_shape(actions)}")
    _require(_batched_shape(retrieved, (8, 10)), f"Invalid retrieved action shape: {_shape(retrieved)}")
    _require(_batched_shape(proprio, (10,)), f"Invalid proprio shape: {_shape(proprio)}")
    _require(_batched_shape(future_proprio, (10,)), f"Invalid future proprio shape: {_shape(future_proprio)}")
    _require(_batched_shape(retrieved_proprio, (10,)), f"Invalid retrieved proprio shape: {_shape(retrieved_proprio)}")
    for name, value in (("actions", actions), ("retrieved_actions", retrieved), ("proprio", proprio)):
        _require(bool(torch.isfinite(value).all()), f"{name} contains non-finite values")
        _require(float(value.abs().max()) <= 1.0001, f"{name} escaped normalized range")

    expected_indices = {
        "retrieved_action_latent_idx": 4,
        "retrieved_state_latent_idx": 3,
        "current_image_latent_idx": 5,
        "current_proprio_latent_idx": 6,
        "action_latent_idx": 7,
        "future_image_latent_idx": 8,
        "future_proprio_latent_idx": 9,
    }
    for name, expected in expected_indices.items():
        tensor = torch.as_tensor(batch[name])
        _require(bool(torch.all(tensor == expected)), f"Wrong latent binding {name}: {tensor}")

    methods = _string_values(batch["method_id"])
    targets = _string_values(batch["target_representation"])
    expected_targets = {
        "recap_hand_ret": "residual",
        "co_training": "absolute",
        "no_retrieval": "absolute",
        "retrieval_only": "retrieval_only",
    }
    _require(all(method in expected_targets for method in methods), f"Unknown method IDs: {methods}")
    if len(targets) == 1 and len(methods) > 1:
        targets *= len(methods)
    _require(len(methods) == len(targets), "method/target batch length mismatch")
    _require(
        all(expected_targets[method] == target for method, target in zip(methods, targets, strict=True)),
        f"Method/target mismatch: {list(zip(methods, targets, strict=True))}",
    )
    _require(all(item == "unverified" for item in _string_values(batch["query_command_status"])), "Command status upgraded")
    _require(all(item in {"", "None"} for item in _string_values(batch["deployment_command_adapter_id"])), "Deployment adapter present")
    _require(bool(torch.all(torch.as_tensor(batch["strict_future_offset_view_steps"]) == 1)), "Target is not strict t+1")
    _require(bool(torch.all(torch.as_tensor(batch["gap_crossing_count"]) == 0)), "Batch crosses a segment")
    protocol_hashes = _string_values(batch["protocol_file_sha256"])
    _require(all(len(item) == 64 for item in protocol_hashes), "Protocol hash missing")
    checks = {
        "required_keys_present": True,
        "formal_shapes_valid": True,
        "normalized_values_finite": True,
        "latent_layout_valid": True,
        "method_target_semantics_valid": True,
        "strict_future_segment_safe": True,
        "deployment_boundary_preserved": True,
        "protocol_bound": True,
    }
    return checks


class Human2RobotAdapterOverfitProbe(nn.Module):
    """Small gradient probe over the exact formal adapter input/target tensors.

    This probe validates adapter learnability and target wiring.  It is not a
    substitute for a full 2B quality experiment; the formal config/import smoke
    separately verifies connection to the real 2B model class.
    """

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10 + 8 * 10 + 1, 256),
            nn.GELU(),
            nn.Linear(256, 8 * 10),
        )

    def forward(self, proprio: torch.Tensor, retrieved_actions: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if retrieved_actions.ndim == 2:
            retrieved_actions = retrieved_actions.unsqueeze(0)
        phase = phase.reshape(-1, 1).to(proprio.dtype)
        features = torch.cat((proprio, retrieved_actions.flatten(1), phase), dim=1)
        return self.network(features).reshape(-1, 8, 10)


def run_one_batch_overfit_probe(
    batch: dict[str, Any], steps: int = 500, learning_rate: float = 0.01, device: str = "cuda"
) -> dict[str, float | int | str | bool]:
    validate_human2robot_batch(batch)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise Human2RobotBatchError("CUDA is required for the Docker P0 overfit probe")
    torch.manual_seed(20260711)
    model = Human2RobotAdapterOverfitProbe().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    proprio = batch["proprio"].to(device=device, dtype=torch.float32)
    retrieved = batch["retrieved_actions"].to(device=device, dtype=torch.float32)
    phase = torch.as_tensor(batch["phase"], dtype=torch.float32, device=device)
    target = batch["actions"].to(device=device, dtype=torch.float32)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(proprio, retrieved, phase)
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    initial = losses[0]
    final = losses[-1]
    return {
        "scope": "formal_adapter_io_gradient_probe_not_full_2b_quality_claim",
        "device": device,
        "steps": steps,
        "initial_loss": initial,
        "final_loss": final,
        "loss_ratio": final / max(initial, 1e-12),
        "passed": final < 1e-4 and final < initial * 0.01,
    }
