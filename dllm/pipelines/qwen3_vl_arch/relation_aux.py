"""In-batch multi-chain relation auxiliaries (cognate pairing + chain-drop).

Used by ``LLaDAEsmcFusion`` during training and by the pairing PLL scorer.
Depends only on torch so the math can be unit-tested without ESMC/LLaDA.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


RELATION_AUX_MODES = ("none", "cognate", "chain_drop", "both")


def generated_heavy_light_masks(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean ``[B, S]`` masks for the two generated receptor chains.

    Same role rule as ``sample_chain_conditioned_timesteps``: among generated
    residues (``diffusion_eligible_mask`` ∩ ``residue_mask``), the smallest
    ``chain_ids`` value is heavy and the next is light. Antigen / MHC / peptide
    context is ignored even when ``diffusion_all_chains`` widened eligibility
    for the reconstruction loss — pairing is a receptor-chain relation.
    """

    residue_mask = batch.get("residue_mask")
    if residue_mask is None:
        raise KeyError("generated_heavy_light_masks requires residue_mask")
    residue_mask = residue_mask.bool()
    eligible = batch.get("diffusion_eligible_mask")
    if eligible is None:
        eligible = residue_mask
    else:
        eligible = eligible.bool() & residue_mask
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        eligible = eligible & attention_mask.bool()

    batch_size, seq_len = residue_mask.shape
    device = residue_mask.device
    heavy_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    light_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    chain_ids = batch.get("chain_ids")
    if chain_ids is None:
        return heavy_mask, light_mask

    sentinel = torch.iinfo(torch.int64).max
    cid = chain_ids.to(dtype=torch.int64)
    cid = cid.masked_fill((~eligible) | cid.lt(0), sentinel)
    heavy_id = cid.min(dim=1).values
    has_heavy = heavy_id.lt(sentinel)
    cid_without_heavy = cid.masked_fill(cid.eq(heavy_id.unsqueeze(1)), sentinel)
    light_id = cid_without_heavy.min(dim=1).values
    has_light = light_id.lt(sentinel)
    heavy_mask = eligible & has_heavy.unsqueeze(1) & chain_ids.eq(heavy_id.unsqueeze(1))
    light_mask = eligible & has_light.unsqueeze(1) & chain_ids.eq(light_id.unsqueeze(1))
    return heavy_mask, light_mask


def pool_masked(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``hidden [B, S, H]`` over true positions of ``mask [B, S]``."""

    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denom


def visible_chain_mask(
    chain_mask: torch.Tensor,
    *,
    residue_mask: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    corruption_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Positions that may enter a pairing pool (generated, attended, uncorrupted)."""

    visible = chain_mask.bool()
    if residue_mask is not None:
        visible = visible & residue_mask.bool()
    if attention_mask is not None:
        visible = visible & attention_mask.bool()
    if corruption_mask is not None:
        visible = visible & ~corruption_mask.bool()
    return visible


def infonce_pair(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """In-batch InfoNCE: row ``i`` of ``anchor`` matches row ``i`` of ``positive``.

    Both arguments are ``[N, H]``. Returns a scalar. ``N < 2`` yields 0 so a
    one-pair micro-batch does not NaN the training step.
    """

    if anchor.shape[0] < 2:
        return anchor.new_zeros(())
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be > 0")
    anchor_n = F.normalize(anchor.float(), dim=-1)
    positive_n = F.normalize(positive.float(), dim=-1)
    logits = anchor_n @ positive_n.transpose(0, 1) / float(temperature)
    labels = torch.arange(anchor_n.size(0), device=anchor_n.device)
    return F.cross_entropy(logits, labels)


def cognate_infonce(
    hidden: torch.Tensor,
    heavy_mask: torch.Tensor,
    light_mask: torch.Tensor,
    *,
    residue_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    corruption_mask: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """ImmunoMatch-style: true H–L pairs vs in-batch random light chains."""

    heavy_vis = visible_chain_mask(
        heavy_mask,
        residue_mask=residue_mask,
        attention_mask=attention_mask,
        corruption_mask=corruption_mask,
    )
    light_vis = visible_chain_mask(
        light_mask,
        residue_mask=residue_mask,
        attention_mask=attention_mask,
        corruption_mask=corruption_mask,
    )
    valid = heavy_vis.any(dim=1) & light_vis.any(dim=1)
    if int(valid.sum()) < 2:
        return hidden.new_zeros(())
    heavy_pool = pool_masked(hidden[valid], heavy_vis[valid])
    light_pool = pool_masked(hidden[valid], light_vis[valid])
    return infonce_pair(heavy_pool, light_pool, temperature)


def chain_drop_infonce(
    hidden: torch.Tensor,
    heavy_mask: torch.Tensor,
    light_mask: torch.Tensor,
    *,
    residue_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    corruption_mask: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """SCEPTR-style: pair view vs single-chain view of the same receptor."""

    heavy_vis = visible_chain_mask(
        heavy_mask,
        residue_mask=residue_mask,
        attention_mask=attention_mask,
        corruption_mask=corruption_mask,
    )
    light_vis = visible_chain_mask(
        light_mask,
        residue_mask=residue_mask,
        attention_mask=attention_mask,
        corruption_mask=corruption_mask,
    )
    pair_vis = heavy_vis | light_vis
    valid = heavy_vis.any(dim=1) & light_vis.any(dim=1) & pair_vis.any(dim=1)
    if int(valid.sum()) < 2:
        return hidden.new_zeros(())
    pair_pool = pool_masked(hidden[valid], pair_vis[valid])
    heavy_pool = pool_masked(hidden[valid], heavy_vis[valid])
    light_pool = pool_masked(hidden[valid], light_vis[valid])
    return 0.5 * (
        infonce_pair(pair_pool, heavy_pool, temperature)
        + infonce_pair(pair_pool, light_pool, temperature)
    )


def compute_relation_aux(
    mode: str,
    hidden: torch.Tensor,
    heavy_mask: torch.Tensor,
    light_mask: torch.Tensor,
    *,
    residue_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    corruption_mask: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Dispatch ``none|cognate|chain_drop|both``. ``none`` returns 0."""

    mode = str(mode)
    if mode not in RELATION_AUX_MODES:
        raise ValueError(f"relation_aux must be one of {RELATION_AUX_MODES}, got {mode!r}")
    if mode == "none":
        return hidden.new_zeros(())
    terms: list[torch.Tensor] = []
    if mode in {"cognate", "both"}:
        terms.append(
            cognate_infonce(
                hidden,
                heavy_mask,
                light_mask,
                residue_mask=residue_mask,
                attention_mask=attention_mask,
                corruption_mask=corruption_mask,
                temperature=temperature,
            )
        )
    if mode in {"chain_drop", "both"}:
        terms.append(
            chain_drop_infonce(
                hidden,
                heavy_mask,
                light_mask,
                residue_mask=residue_mask,
                attention_mask=attention_mask,
                corruption_mask=corruption_mask,
                temperature=temperature,
            )
        )
    stacked = torch.stack([term.to(dtype=hidden.dtype) for term in terms])
    return stacked.mean()


def mean_token_logprob(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-row mean log p(target) over ``mask``. ``[B, S, V]`` → ``[B]``."""

    logp = F.log_softmax(logits.float(), dim=-1)
    token = logp.gather(-1, target_ids.long().unsqueeze(-1)).squeeze(-1)
    weights = mask.to(dtype=token.dtype)
    denom = weights.sum(dim=-1).clamp_min(1.0)
    return (token * weights).sum(dim=-1) / denom


def pairing_auroc(positive: list[float], negative: list[float]) -> float:
    """Mann–Whitney AUROC: P(pos > neg) + 0.5 P(tie). Empty side → NaN."""

    if not positive or not negative:
        return float("nan")
    pos = torch.tensor(positive, dtype=torch.float64)
    neg = torch.tensor(negative, dtype=torch.float64)
    greater = pos.unsqueeze(1) > neg.unsqueeze(0)
    tied = pos.unsqueeze(1) == neg.unsqueeze(0)
    return float((greater.double() + 0.5 * tied.double()).mean())
