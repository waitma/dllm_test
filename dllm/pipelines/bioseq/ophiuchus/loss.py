from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def label_smoothed_nll_loss(
    lprobs: Tensor,
    target: Tensor,
    epsilon: float,
    ignore_index: int | None = None,
    focal: bool = False,
    gamma: float = 1.0,
    reduce: bool = True,
):
    flag = False
    if target.dim() == lprobs.dim() - 1:
        flag = True
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)

    if focal:
        p_true = torch.exp(-nll_loss)
        focal_term = (1 - p_true) ** gamma
        nll_loss *= focal_term
        smooth_loss *= focal_term

    if flag:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class RDMCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(
        self,
        scores: Tensor,
        target: Tensor,
        label_mask: Tensor | None = None,
        weights: Tensor | None = None,
        focal: bool = False,
        gamma: float = 1.0,
    ):
        n_tokens = target.numel()
        if self.ignore_index is not None:
            sample_size = target.ne(self.ignore_index).float().sum()
        else:
            sample_size = torch.tensor(n_tokens, device=target.device, dtype=torch.float32)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=F.log_softmax(scores, dim=-1),
            target=target,
            epsilon=self.label_smoothing,
            ignore_index=self.ignore_index,
            focal=focal,
            gamma=gamma,
            reduce=False,
        )

        if weights is not None:
            loss = loss * weights
            nll_loss = nll_loss * weights
        fullseq_loss = loss.sum() / sample_size
        fullseq_nll_loss = nll_loss.sum() / sample_size

        if label_mask is not None:
            label_mask = label_mask.float()
            loss = (loss * label_mask).sum() / sample_size
            nll_loss = (nll_loss * label_mask).sum() / sample_size
        else:
            loss, nll_loss = fullseq_loss, fullseq_nll_loss

        return loss, {
            "nll_loss": nll_loss.data,
            "sample_size": sample_size,
        }
