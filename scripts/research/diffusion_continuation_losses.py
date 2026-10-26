"""EOS focal loss with separately retained, unweighted CE diagnostics.

Activate pllm; python -B -m pytest
scripts/tests/immune_llada/test_diffusion_continuation_losses.py -q.
Corruption and distributed normalization are caller responsibilities: this
function returns sums so microbatch boundaries cannot change the objective.
"""
import math

import torch
from torch.nn import functional as F


def weighted_token_loss(logits, labels, weights, *, eos_id, eos_gamma=0., eos_alpha=1., per_view=False):
    """Apply focal only to true EOS labels; gamma=0 and alpha=1 recover CE.

    Noise-time weights must come from the actual corruption policy. They are
    separate from the focal factor, and must not be normalized again here.
    Ignored positions are removed before CE to avoid 0 * inf with masked logits.
    """
    if labels.shape != logits.shape[:-1] or weights.shape != labels.shape:
        raise ValueError("Loss shapes do not match")
    if not (math.isfinite(eos_gamma) and math.isfinite(eos_alpha)
            and eos_gamma >= 0 and eos_alpha > 0):
        raise ValueError("Invalid EOS focal parameters")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Token weights must be finite and nonnegative")
    selected = labels.ne(-100)
    if not selected.any():
        zero = logits[..., 0].nan_to_num(neginf=0., posinf=0.).sum() * 0
        return dict(loss_sum=zero, ce_sum=zero.detach(), eos_ce_sum=zero.detach(),
                    tokens=selected.sum(), eos_tokens=selected.sum(),
                    **({"view_loss_sums": logits.new_zeros(len(logits), dtype=torch.float32)} if per_view else {}))
    target = labels[selected]
    ce = F.cross_entropy(logits[selected].float(), target, reduction="none")
    eos = target.eq(eos_id)
    factor = torch.ones_like(ce)
    if eos_gamma:
        factor[eos] = (-torch.expm1(-ce[eos])).pow(eos_gamma)
    factor[eos] *= eos_alpha
    weighted = ce * factor * weights[selected]
    details = {}
    if per_view:
        values = torch.zeros(len(labels), device=logits.device, dtype=torch.float32)
        values.scatter_add_(0, selected.nonzero(as_tuple=True)[0], weighted.detach())
        details["view_loss_sums"] = values
    return dict(loss_sum=weighted.sum(),
                ce_sum=ce.detach().sum(), eos_ce_sum=ce[eos].detach().sum(),
                tokens=selected.sum(), eos_tokens=eos.sum(), **details)
