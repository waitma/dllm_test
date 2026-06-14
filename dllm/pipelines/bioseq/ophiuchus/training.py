from __future__ import annotations

from dataclasses import dataclass

import torch

from .loss import RDMCrossEntropyLoss
from .multichain import MultiChainOphiuchusAbModel


@dataclass
class OphiuchusAbTrainStepConfig:
    weighting: str = "reciprocal"
    softmin_snr: float | None = 20.0
    heavy_loss_weight: float = 1.0
    light_loss_weight: float = 1.0
    focal: bool = True
    gamma: float = 1.0
    ignore_index: int = 1


@dataclass
class OphiuchusAbTrainStepResult:
    loss: torch.Tensor
    heavy_loss: torch.Tensor
    light_loss: torch.Tensor


def compute_ophiuchus_ab_training_loss(
    model: MultiChainOphiuchusAbModel,
    batch: dict,
    config: OphiuchusAbTrainStepConfig | None = None,
    stage: str | None = None,
) -> OphiuchusAbTrainStepResult:
    config = config or OphiuchusAbTrainStepConfig()
    criterion = RDMCrossEntropyLoss(ignore_index=config.ignore_index, label_smoothing=0.0)

    logits, target, loss_mask, weights = model.compute_loss(
        batch,
        weighting=config.weighting,
        gamma=config.softmin_snr,
        stage=stage,
    )

    heavy_loss, _ = criterion(
        logits["heavy"],
        target["heavy"],
        loss_mask["heavy"],
        weights["heavy"],
        focal=config.focal,
        gamma=config.gamma,
    )
    light_loss, _ = criterion(
        logits["light"],
        target["light"],
        loss_mask["light"],
        weights["light"],
        focal=config.focal,
        gamma=config.gamma,
    )
    loss = config.heavy_loss_weight * heavy_loss + config.light_loss_weight * light_loss
    return OphiuchusAbTrainStepResult(loss=loss, heavy_loss=heavy_loss, light_loss=light_loss)
