from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class BioSeqDiffusionConfig:
    mask_token_id: int
    time_epsilon: float = 1e-3
    loss_norm: str = "token"


@dataclass
class BioSeqDiffusionLoss:
    loss: torch.Tensor
    logits: torch.Tensor
    masked_mask: torch.Tensor
    timesteps: torch.Tensor


def sample_masked_diffusion_inputs(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    mask_token_id: int,
    time_epsilon: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (0.0 < time_epsilon < 1.0):
        raise ValueError("time_epsilon must be in (0, 1)")

    batch_size, seq_len = input_ids.shape
    timesteps = torch.empty(batch_size, device=input_ids.device).uniform_(time_epsilon, 1.0)
    mask_probs = timesteps[:, None].expand(batch_size, seq_len)
    masked_mask = (torch.rand_like(mask_probs) < mask_probs) & loss_mask.bool()

    for row in range(batch_size):
        if loss_mask[row].any() and not masked_mask[row].any():
            valid_positions = torch.nonzero(loss_mask[row], as_tuple=False).flatten()
            choice = valid_positions[torch.randint(valid_positions.numel(), (1,), device=input_ids.device)]
            masked_mask[row, choice] = True

    noised_input_ids = input_ids.masked_fill(masked_mask, mask_token_id)
    labels = input_ids.masked_fill(~masked_mask, -100)
    return noised_input_ids, labels, timesteps


def compute_diffusion_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    diffusion_config: BioSeqDiffusionConfig,
) -> BioSeqDiffusionLoss:
    noised_input_ids, labels, timesteps = sample_masked_diffusion_inputs(
        input_ids=batch["input_ids"],
        loss_mask=batch["loss_mask"],
        mask_token_id=diffusion_config.mask_token_id,
        time_epsilon=diffusion_config.time_epsilon,
    )
    outputs = model(
        input_ids=noised_input_ids,
        chain_ids=batch.get("chain_ids"),
        attention_mask=batch.get("attention_mask"),
    )
    logits = getattr(outputs, "logits", outputs)
    token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)

    masked_mask = labels.ne(-100)
    if diffusion_config.loss_norm == "batch":
        loss = token_loss.sum() / input_batch_size(batch)
    elif diffusion_config.loss_norm == "sequence":
        per_sequence = token_loss.sum(dim=1) / masked_mask.sum(dim=1).clamp_min(1)
        loss = per_sequence.mean()
    elif diffusion_config.loss_norm == "token":
        loss = token_loss.sum() / masked_mask.sum().clamp_min(1)
    else:
        raise ValueError(f"Unsupported loss_norm: {diffusion_config.loss_norm}")

    return BioSeqDiffusionLoss(
        loss=loss,
        logits=logits,
        masked_mask=masked_mask,
        timesteps=timesteps,
    )


def input_batch_size(batch: dict[str, torch.Tensor]) -> int:
    return int(batch["input_ids"].shape[0])


class BioSeqDiffusionTrainer:
    def __init__(self, *args: Any, diffusion_config: BioSeqDiffusionConfig, **kwargs: Any) -> None:
        try:
            import transformers
        except ImportError as exc:
            raise ImportError("transformers is required for BioSeqDiffusionTrainer") from exc

        class _Trainer(transformers.Trainer):
            def __init__(self, *inner_args: Any, diffusion_config: BioSeqDiffusionConfig, **inner_kwargs: Any) -> None:
                super().__init__(*inner_args, **inner_kwargs)
                self.diffusion_config = diffusion_config

            def compute_loss(
                self,
                model: torch.nn.Module,
                inputs: dict[str, torch.Tensor],
                return_outputs: bool = False,
                **_: Any,
            ):
                result = compute_diffusion_loss(
                    model=model,
                    batch=inputs,
                    diffusion_config=self.diffusion_config,
                )
                if return_outputs:
                    return result.loss, {"logits": result.logits, "masked_mask": result.masked_mask}
                return result.loss

        self._trainer = _Trainer(*args, diffusion_config=diffusion_config, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._trainer, name)
