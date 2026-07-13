"""Formal 2B retrieval-conditioned model adapter for Human2Robot."""

from __future__ import annotations

import torch

from cosmos_policy.models.human2robot_adapter import validate_human2robot_batch
from cosmos_policy.models.policy_video2world_model_pusht_ret import (
    CosmosPolicyPushTRetModelRectifiedFlow,
)


class CosmosPolicyHuman2RobotRetModelRectifiedFlow(CosmosPolicyPushTRetModelRectifiedFlow):
    """Real Cosmos-Predict2.5 2B model with strict Human2Robot batch checks."""

    def training_step(self, data_batch, iteration):
        # A one-batch overfit test is only interpretable when the rectified-flow
        # noise/time draw is held fixed.  Formal runs emit mode=0 and retain the
        # normal stochastic training path; only the explicit diagnostic dataset
        # override emits mode=1.
        diagnostic = torch.as_tensor(data_batch.get("diagnostic_overfit_mode", 0))
        if bool(torch.all(diagnostic == 1)):
            seed = int(torch.as_tensor(data_batch["diagnostic_overfit_seed"]).flatten()[0])
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        return super().training_step(data_batch, iteration)

    def get_data_and_condition(self, data_batch):
        validate_human2robot_batch(data_batch)
        return super().get_data_and_condition(data_batch)

    def get_x0_fn_from_batch(self, data_batch, guidance=1.5, **kwargs):
        validate_human2robot_batch(data_batch)
        return super().get_x0_fn_from_batch(data_batch, guidance=guidance, **kwargs)

    def get_velocity_fn_from_batch(self, data_batch, guidance=1.5, **kwargs):
        validate_human2robot_batch(data_batch)
        return super().get_velocity_fn_from_batch(data_batch, guidance=guidance, **kwargs)
