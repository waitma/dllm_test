"""ESMC-conditioned LLaDA-8B fusion model (residue conditioning ablations)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import transformers

from dllm.pipelines.immune_llada.data import GRAMMAR_TOKENS, GrammarTokenizer
from dllm.pipelines.immune_llada.data.grammar import (
    FIXED_HEAVY_ENCODER_LENGTH, FIXED_LIGHT_ENCODER_LENGTH,
    FIXED_HEAVY_DECODER_SLOTS, FIXED_LIGHT_DECODER_SLOTS,
)
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (
    BioSeqDiffusionOutput,
    BioSeqEncoderDiffusionModel,
    apply_decoder_corruption_to_encoder,
    compute_masked_cross_entropy,
    sample_bioseq_diffusion_noise,
    sample_chain_conditioned_timesteps,
)
from dllm.pipelines.qwen3_vl_arch.relation_aux import (
    RELATION_AUX_MODES,
    compute_relation_aux,
    generated_heavy_light_masks,
)

logger = logging.getLogger(__name__)

RESIDUES = list("LAGVSERTIDPKQNFYMHWCXBUZO")


def decoder_mask_token_id(tok: transformers.PreTrainedTokenizer) -> int:
    """Resolve the shared training/evaluation mask ID without importing Trainer."""
    mask_id = tok.convert_tokens_to_ids("<|mdm_mask|>")
    unk = getattr(tok, "unk_token_id", None)
    if mask_id is None or (unk is not None and int(mask_id) == int(unk)):
        mask_id = tok.mask_token_id
    if mask_id is None:
        raise RuntimeError("LLaDA tokenizer is missing <|mdm_mask|> / mask_token_id")
    return int(mask_id)


class _FusionConfig(SimpleNamespace):
    """Attribute bag for the fusion model's config.

    Exposes ``to_dict`` so HF Trainer integrations (e.g. the W&B callback, which
    does ``model.config.to_dict()`` in ``on_train_begin``) can serialize the
    hyperparameters. Plain ``SimpleNamespace`` lacks it and raises AttributeError.
    """

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


FUSION_CONFIG_FILENAME = "fusion_config.json"


def _unwrap_fusion_module(model: nn.Module) -> nn.Module:
    """Return the underlying fusion module through DDP/FSDP-style wrappers."""

    seen: set[int] = set()
    current = model
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "config") and hasattr(current, "condition_proj"):
            return current
        next_module = getattr(current, "module", None)
        if next_module is None:
            next_module = getattr(current, "_fsdp_wrapped_module", None)
        if next_module is None or next_module is current:
            break
        current = next_module
    return current


def fusion_config_path(directory: str | Path) -> Path:
    """Return the sidecar configuration path for a fusion Trainer directory."""

    return Path(directory) / FUSION_CONFIG_FILENAME


def load_fusion_config(directory: str | Path) -> dict[str, Any]:
    """Read ``fusion_config.json``; old checkpoints intentionally default to ``{}``."""

    path = fusion_config_path(directory)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"fusion config must be a JSON object: {path}")
    return payload


def save_fusion_config(
    directory: str | Path,
    model: nn.Module,
    *,
    tokenizer: Any | None = None,
) -> Path:
    """Serialize fusion-only architecture/policy metadata beside checkpoint weights.

    HF's generic ``config.json`` is owned by the decoder.  This sidecar preserves
    the composite model's fixed-canvas and decoder-chain-EOS policy, including for
    Trainer checkpoints whose model class cannot be reconstructed by HF alone.
    """

    module = _unwrap_fusion_module(model)
    config = getattr(module, "config", None)
    payload = config.to_dict() if hasattr(config, "to_dict") else dict(vars(config))
    payload.setdefault("fusion_config_version", 2)
    payload.setdefault("fixed_receptor_lengths", False)
    payload.setdefault("fixed_heavy_encoder_length", FIXED_HEAVY_ENCODER_LENGTH)
    payload.setdefault("fixed_light_encoder_length", FIXED_LIGHT_ENCODER_LENGTH)
    payload.setdefault("fixed_heavy_decoder_slots", FIXED_HEAVY_DECODER_SLOTS)
    payload.setdefault("fixed_light_decoder_slots", FIXED_LIGHT_DECODER_SLOTS)
    payload.setdefault("fixed_heavy_residue_max", FIXED_HEAVY_DECODER_SLOTS - 1)
    payload.setdefault("fixed_light_residue_max", FIXED_LIGHT_DECODER_SLOTS - 1)
    payload.setdefault("predict_eos", False)
    if tokenizer is not None:
        for key, attr in (
            ("decoder_chain_eos_token_id", "_fusion_decoder_chain_eos_token_id"),
            ("decoder_chain_eos_policy", "_fusion_decoder_chain_eos_policy"),
        ):
            value = getattr(tokenizer, attr, None)
            if value is not None:
                payload[key] = value

    path = fusion_config_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def sample_bioseq_bert_noise(
    batch: dict[str, Any],
    mask_token_id: int,
    *,
    mask_ratio: float = 0.15,
    mask_prob: float = 0.8,
    random_prob: float = 0.1,
    residue_token_ids: torch.Tensor | None = None,
    all_chain_targets: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classic BERT MLM corruption (fixed rate + 80/10/10) on eligible tokens.

    Contrasts with :func:`sample_bioseq_diffusion_noise` (per-sequence rate
    ``t ~ U(eps, 1)`` and 100% ``<mask>`` replacement). Here a fixed
    ``mask_ratio`` fraction of eligible tokens is *selected*; among the selected,
    ``mask_prob`` become ``<mask>``, ``random_prob`` become a random residue
    token on *residue* positions (non-residue selected positions in this slice
    get ``<mask>`` instead — relation targets must not be rewritten to an amino
    acid), and the remainder keep their original id. Loss is computed on *all*
    selected positions (labels set there, ``-100`` elsewhere).

    ``all_chain_targets`` picks the eligible set. When True (the BERT default)
    every *real* residue in *all* chains is eligible (``residue_mask`` &
    attention, excluding synthetic completion placeholders), plus supervised
    relation targets (``relation_target_mask``: trainable ``<binding>`` /
    ``<nonbinding>`` only — not ``relation_token_mask``). Fixed grammar context
    such as MHC / peptide / antigen is trained too. When False the eligible set
    mirrors the diffusion sampler (``diffusion_eligible_mask``: generated chains
    plus those same relation targets, context excluded).

    Returns
    -------
    noised_input_ids : ``[B, S]`` — decoder input after 80/10/10.
    labels : ``[B, S]`` — clean ids on selected positions; ``-100`` elsewhere.
    selection_mask : ``[B, S]`` bool — all selected positions. Used to mirror
        masking onto the ESMC encoder (mask *every* selected residue there so the
        frozen encoder cannot leak a target through the condition path).
    """

    if not (0.0 < mask_ratio < 1.0):
        raise ValueError("mask_ratio must be in (0, 1)")
    if mask_prob < 0.0 or random_prob < 0.0 or (mask_prob + random_prob) > 1.0:
        raise ValueError("require mask_prob, random_prob >= 0 and their sum <= 1")

    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    residue_mask = batch.get("residue_mask")

    if all_chain_targets:
        # BERT objective: predict every *real* residue across *all* chains
        # (含固定上下文 MHC / peptide / antigen) 以及可训关系 token。
        # 合格集不是 diffusion 的 generated-only（diffusion_eligible_mask）。
        if residue_mask is None:
            raise KeyError("all_chain_targets=True requires residue_mask")
        eligible_mask = residue_mask.bool()
        eos_target_mask = batch.get("chain_eos_mask")
        if eos_target_mask is not None:
            eligible_mask = eligible_mask | eos_target_mask.bool()
        # synthetic X 是补全占位符，不是真实残基；renderer 的
        # diffusion_eligible_mask 已排除，all-chains 路径必须自己再排一次
        synthetic = batch.get("synthetic_residue_mask")
        if synthetic is not None:
            eligible_mask = eligible_mask & ~synthetic.bool()
        # 只并入 relation_target_mask（可训的 <binding>/<nonbinding>）。
        # 不能用 relation_token_mask：后者含 MHC→peptide 固定呈递 <binding>
        # 和 OAS/OTS null 前缀 <unknown>。缺 key 视为全 0，保持旧行为。
        relation_target = batch.get("relation_target_mask")
        if relation_target is not None:
            eligible_mask = eligible_mask | relation_target.bool()
    else:
        loss_mask = batch.get("diffusion_loss_mask", batch.get("diffusion_target_mask"))
        if loss_mask is None:
            raise KeyError("batch requires diffusion_loss_mask or diffusion_target_mask")
        explicit_eligible_mask = batch.get("diffusion_eligible_mask")
        eligible_mask = (
            explicit_eligible_mask.bool()
            if explicit_eligible_mask is not None
            else loss_mask.bool()
        )
        if explicit_eligible_mask is None and residue_mask is not None:
            eligible_mask = eligible_mask & residue_mask.bool()
    if attention_mask is not None:
        eligible_mask = eligible_mask & attention_mask.bool()
    if not eligible_mask.any():
        raise ValueError("batch has no eligible target tokens")

    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    selection_mask = (
        torch.rand(batch_size, seq_len, device=device) < mask_ratio
    ) & eligible_mask
    # Guarantee >=1 selected token per row with eligible positions (avoid zero loss).
    for row in range(batch_size):
        if eligible_mask[row].any() and not selection_mask[row].any():
            valid_positions = torch.nonzero(eligible_mask[row], as_tuple=False).flatten()
            choice = valid_positions[
                torch.randint(valid_positions.numel(), (1,), device=device)
            ]
            selection_mask[row, choice] = True

    labels = input_ids.masked_fill(~selection_mask, -100)
    noised_input_ids = input_ids.clone()
    roll = torch.rand(batch_size, seq_len, device=device)
    to_mask = selection_mask & (roll < mask_prob)
    to_random = selection_mask & (roll >= mask_prob) & (roll < mask_prob + random_prob)
    noised_input_ids[to_mask] = int(mask_token_id)
    # 80/10/10 的 random 切片只能对残基位写残基 id。关系 target 是
    # TOKEN_CLASS_RELATION，均匀抽残基会把 <binding>/<nonbinding> 改成氨基酸。
    # 对齐 diffusion sampler「Grammar and structure tokens are never uniformly
    # replaced」：非残基合格位落在 random 切片时改写为 <mask>。
    # 10% keep-original 仍按经典 MLM，所有位置保留原 id。
    if residue_mask is not None:
        residue_random = to_random & residue_mask.bool()
        other_random = to_random & ~residue_mask.bool()
    else:
        residue_random = to_random
        other_random = to_random.new_zeros(to_random.shape, dtype=torch.bool)
    if other_random.any():
        noised_input_ids[other_random] = int(mask_token_id)
    if residue_random.any():
        if residue_token_ids is not None and int(residue_token_ids.numel()) > 0:
            pool = residue_token_ids.to(device=device, dtype=noised_input_ids.dtype)
            picks = pool[torch.randint(int(pool.numel()), (int(residue_random.sum()),), device=device)]
            noised_input_ids[residue_random] = picks
        else:
            # No residue vocab provided -> fall back to <mask> for the random slice.
            noised_input_ids[residue_random] = int(mask_token_id)
    # Remaining selected positions keep their original id (the "10% unchanged").
    return noised_input_ids, labels, selection_mask


def all_residue_eligible_mask(batch: dict[str, Any]) -> torch.Tensor:
    """``[B, S]`` bool — every *real* residue of every chain, fixed context included.

    The diffusion counterpart of the BERT ``all_chain_targets=True`` eligible set.
    Grammar/structure tokens, padding, and synthetic completion placeholders stay
    excluded; what is added relative to ``diffusion_eligible_mask`` is precisely
    the fixed context (antigen / MHC / peptide) that the renderer marks as
    never-corrupted. Supervised relation targets (``relation_target_mask``) are
    carried over so that "all chains" is a true *superset* of generated-only:
    ``compute_loss`` **overwrites** ``diffusion_eligible_mask`` /
    ``diffusion_loss_mask`` with this mask, so anything omitted here is silently
    dropped from the objective.
    """

    residue_mask = batch.get("residue_mask")
    if residue_mask is None:
        raise KeyError("diffusion_all_chains=True requires residue_mask")
    # EOS canvas positions are valid decoder targets but are intentionally not
    # amino-acid residue slots, so they use a separate mask.
    eligible_mask = residue_mask.bool()
    eos_target_mask = batch.get("chain_eos_mask")
    if eos_target_mask is not None:
        eligible_mask = eligible_mask | eos_target_mask.bool()
    # synthetic X 是补全占位符，不是真实残基；renderer 的
    # diffusion_eligible_mask 已排除，all-chains 路径必须自己再排一次
    synthetic = batch.get("synthetic_residue_mask")
    if synthetic is not None:
        eligible_mask = eligible_mask & ~synthetic.bool()
    # 只并入 relation_target_mask（可训的 <binding>/<nonbinding>），与 BERT
    # all-chains 口径一致。不能用 relation_token_mask：后者含 MHC→peptide 固定
    # 呈递 <binding> 与 OAS/OTS null 前缀 <unknown>。
    # 必须并进来：compute_loss 会用本 mask **覆写** diffusion_eligible_mask /
    # diffusion_loss_mask，漏掉就等于打开 all-chains 反而丢掉了 generated-only
    # 本来就在训的 relation 监督 —— all-chains 必须是 generated-only 的超集。
    relation_target = batch.get("relation_target_mask")
    if relation_target is not None:
        eligible_mask = eligible_mask | relation_target.bool()
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        eligible_mask = eligible_mask & attention_mask.bool()
    if not eligible_mask.any():
        raise ValueError("batch has no eligible target tokens")
    return eligible_mask


def _base_id_to_token(base: Any) -> dict[int, str]:
    id_to_token = getattr(base, "id_to_token", None)
    if isinstance(id_to_token, dict) and id_to_token:
        return {int(k): str(v) for k, v in id_to_token.items()}

    backend = base
    for _ in range(3):
        nxt = getattr(backend, "tokenizer", None)
        if nxt is None or nxt is backend:
            break
        backend = nxt
        if hasattr(backend, "id_to_token") and hasattr(backend, "get_vocab_size"):
            return {
                i: str(backend.id_to_token(i))
                for i in range(int(backend.get_vocab_size()))
            }

    if hasattr(base, "convert_ids_to_tokens"):
        vocab_size = int(getattr(base, "vocab_size", 0) or len(base))
        return {i: str(base.convert_ids_to_tokens(i)) for i in range(vocab_size)}

    raise AttributeError(
        f"Cannot build id_to_token from base tokenizer type {type(base)!r}"
    )


def expand_llada_tokenizer_for_esmc_grammar(
    tok: transformers.PreTrainedTokenizer,
    gtok_esmc: GrammarTokenizer,
    *,
    allow_chain_eos_token: bool = False,
) -> tuple[dict[int, int], int]:
    """Build the ESMC-grammar -> LLaDA remap, including grammar ``<eos>``.

    Legacy callers keep the historical aliasing behavior and do not grow the
    vocabulary. Fixed-canvas v2 callers may opt into ``<chain_eos>`` only when
    the native decoder EOS is unsafe because it aliases padding or masking.
    When the native EOS is distinct, it is reused and no vocabulary growth is
    needed for the chain terminator.
    """

    residue_tokens = [f"<res_{aa}>" for aa in RESIDUES]
    new_tokens = residue_tokens + list(GRAMMAR_TOKENS) + ["<chainsep>"]
    n_added = tok.add_tokens(new_tokens)

    native_eos = getattr(tok, "eos_token_id", None)
    if native_eos is None:
        raise RuntimeError("LLaDA tokenizer must expose eos_token_id")
    native_eos = int(native_eos)
    pad_id = getattr(tok, "pad_token_id", None)
    mask_id = getattr(tok, "mask_token_id", None)
    eos_is_safe = native_eos != (None if pad_id is None else int(pad_id)) and native_eos != (
        None if mask_id is None else int(mask_id)
    )
    chain_eos_policy = "native_eos"
    chain_eos_id = native_eos
    if not eos_is_safe and allow_chain_eos_token:
        chain_eos_token = "<chain_eos>"
        added_chain_eos = tok.add_special_tokens(
            {"additional_special_tokens": [chain_eos_token]}
        )
        n_added += int(added_chain_eos)
        chain_eos_id = int(tok.convert_tokens_to_ids(chain_eos_token))
        if chain_eos_id < 0:
            raise RuntimeError("failed to add <chain_eos> to the LLaDA tokenizer")
        chain_eos_policy = "chain_eos"
    elif not eos_is_safe:
        # This is deliberately retained for old checkpoints. Their pad/eos alias
        # is ambiguous on inverse conversion, but changing the vocabulary would
        # make old weights unusable.
        chain_eos_policy = "legacy_eos_alias"

    setattr(tok, "_fusion_decoder_chain_eos_token_id", int(chain_eos_id))
    setattr(tok, "_fusion_decoder_chain_eos_policy", chain_eos_policy)
    logger.info(
        "Decoder chain EOS: policy=%s id=%d (native=%d, pad=%s, mask=%s, added=%d)",
        chain_eos_policy,
        chain_eos_id,
        native_eos,
        pad_id,
        mask_id,
        n_added,
    )

    remap: dict[int, int] = {}
    residue_set = set(RESIDUES)
    for base_id, token in _base_id_to_token(gtok_esmc.base_tokenizer).items():
        if token in residue_set:
            remap[int(base_id)] = int(tok.convert_tokens_to_ids(f"<res_{token}>"))

    for grammar_token in GRAMMAR_TOKENS:
        remap[int(gtok_esmc.special_id(grammar_token))] = int(
            tok.convert_tokens_to_ids(grammar_token)
        )

    remap[int(gtok_esmc.chain_separator_id())] = int(tok.convert_tokens_to_ids("<chainsep>"))
    grammar_pad_id = int(gtok_esmc.pad_token_id)
    remap[grammar_pad_id] = int(tok.pad_token_id if tok.pad_token_id is not None else native_eos)
    # ESM-family grammar uses id 2 for EOS. It must be legal as a supervised
    # decoder target in the fixed canvas, rather than being treated as padding.
    remap[int(gtok_esmc.eos_token_id)] = int(chain_eos_id)

    unk_id = getattr(tok, "unk_token_id", None)
    if unk_id is not None:
        for grammar_id, llada_id in remap.items():
            assert llada_id != unk_id, (
                f"remap[{grammar_id}] resolved to unk ({unk_id}); "
                "token was not added atomically"
            )

    return remap, n_added


def build_inverse_remap(
    remap: dict[int, int],
    *,
    preferred_source_ids: tuple[int, ...] = (),
    size: int | None = None,
) -> torch.Tensor:
    """Build LLaDA -> grammar ids with deterministic pad/EOS collision handling.

    ``preferred_source_ids`` selects which source wins an alias collision; it
    does not remove the underlying ambiguity. Callers must explicitly choose
    their policy when integrating this helper into checkpoint loading.
    """

    max_destination = max((int(dst) for dst in remap.values()), default=-1)
    inverse_size = max(int(size or 0), max_destination + 1)
    inverse = torch.full((inverse_size,), -1, dtype=torch.long)
    preferred = set(int(value) for value in preferred_source_ids)
    for source, destination in remap.items():
        destination = int(destination)
        if destination < 0 or destination >= inverse_size:
            continue
        if int(inverse[destination]) < 0 or int(source) in preferred:
            inverse[destination] = int(source)
    return inverse


def build_remap_lookup(remap: dict[int, int]) -> torch.Tensor:
    """LongTensor lookup[src_id] = llada_id; unmapped entries stay -1."""

    max_id = max(remap.keys()) if remap else 0
    lookup = torch.full((max_id + 1,), -1, dtype=torch.long)
    for src, dst in remap.items():
        lookup[int(src)] = int(dst)
    return lookup


class RemapCollator:
    """Remap decoder ids and labels from grammar space into LLaDA space."""

    def __init__(self, base_collator: Any, lookup_tensor: torch.Tensor) -> None:
        self.base_collator = base_collator
        self.lookup = lookup_tensor.long()

    def _remap(self, values: torch.Tensor, *, preserve_ignore_index: bool) -> torch.Tensor:
        if preserve_ignore_index:
            ignored = values.eq(-100)
            source = values.masked_fill(ignored, 0)
        else:
            ignored = torch.zeros_like(values, dtype=torch.bool)
            source = values
        source_cpu = source.cpu().long()
        ignored_cpu = ignored.cpu()
        if source_cpu.numel() and (
            int(source_cpu.min().item()) < 0
            or int(source_cpu.max().item()) >= int(self.lookup.numel())
        ):
            raise IndexError(
                f"decoder ids must be in [0, {int(self.lookup.numel())}); "
                f"got min={int(source_cpu.min())}, max={int(source_cpu.max())}"
            )
        remapped = self.lookup[source_cpu]
        if not bool((remapped[~ignored_cpu] >= 0).all()):
            bad = remapped.lt(0) & ~ignored_cpu
            bad_src = source_cpu[bad][:8].tolist()
            raise AssertionError(f"unmapped decoder grammar ids (sample): {bad_src}")
        remapped = remapped.to(device=values.device)
        return remapped.masked_fill(ignored.to(device=values.device), -100)

    def __call__(self, records: list[Any]) -> dict[str, Any]:
        batch = self.base_collator(records)
        batch["input_ids"] = self._remap(batch["input_ids"], preserve_ignore_index=False)
        if "labels" in batch and batch["labels"] is not None:
            batch["labels"] = self._remap(batch["labels"], preserve_ignore_index=True)
        return batch


class LLaDAEsmcFusion(BioSeqEncoderDiffusionModel):
    """Pretrained LLaDA decoder + ESMC encoder with residue_cond_mode ablations."""

    def __init__(
        self,
        decoder: nn.Module,
        encoder: nn.Module,
        encoder_hidden_size: int,
        decoder_mask_token_id: int,
        encoder_mask_token_id: int,
        residue_cond_mode: str = "add",
        condition_norm: bool = True,
        time_epsilon: float = 1e-3,
        loss_norm: str = "token",
        freeze_encoder: bool = True,
        train_objective: str = "diffusion",
        bert_mask_ratio: float = 0.15,
        bert_mask_prob: float = 0.8,
        bert_random_prob: float = 0.1,
        bert_all_chains: bool = True,
        diffusion_all_chains: bool = False,
        residue_token_ids: list[int] | None = None,
        gidd_uniform_ratio: float = 0.0,
        gidd_gamma: float = 1.0,
        independent_loss_ratio: float = 0.0,
        single_chain_ratio: float = 0.0,
        heavy2light_loss_ratio: float = 0.0,
        light2heavy_loss_ratio: float = 0.0,
        joint_loss_ratio: float = 1.0,
        focal: bool = False,
        focal_gamma: float = 1.0,
        loss_weight_type: str = "none",
        softmin_snr: float = 20.0,
        heavy_loss_weight: float = 1.0,
        light_loss_weight: float = 1.0,
        relation_aux: str = "none",
        relation_aux_weight: float = 0.1,
        relation_aux_temperature: float = 0.07,
        predict_eos: bool = False,
        fixed_receptor_lengths: bool = False,
        decoder_chain_eos_token_id: int | None = None,
        decoder_chain_eos_policy: str | None = None,
    ) -> None:
        nn.Module.__init__(self)
        if residue_cond_mode not in {"token", "feature", "add"}:
            raise ValueError(
                f"residue_cond_mode must be token|feature|add, got {residue_cond_mode!r}"
            )
        if train_objective not in {"diffusion", "bert"}:
            raise ValueError(
                f"train_objective must be diffusion|bert, got {train_objective!r}"
            )
        loss_weight_type = str(loss_weight_type)
        if loss_weight_type not in {"none", "uniform", "reciprocal"}:
            raise ValueError(
                f"loss_weight_type must be none|uniform|reciprocal, got {loss_weight_type!r}"
            )
        relation_aux = str(relation_aux)
        if relation_aux not in RELATION_AUX_MODES:
            raise ValueError(
                f"relation_aux must be one of {RELATION_AUX_MODES}, got {relation_aux!r}"
            )
        if float(relation_aux_weight) < 0.0:
            raise ValueError("relation_aux_weight must be >= 0")
        if float(relation_aux_temperature) <= 0.0:
            raise ValueError("relation_aux_temperature must be > 0")
        if not (0.0 <= float(gidd_uniform_ratio) < 1.0):
            raise ValueError("gidd_uniform_ratio must be in [0, 1)")
        if float(gidd_gamma) < 0.0:
            raise ValueError("gidd_gamma must be >= 0")
        chain_ratios = {
            "single_chain_ratio": float(single_chain_ratio),
            "heavy2light_loss_ratio": float(heavy2light_loss_ratio),
            "light2heavy_loss_ratio": float(light2heavy_loss_ratio),
            "independent_loss_ratio": float(independent_loss_ratio),
            "joint_loss_ratio": float(joint_loss_ratio),
        }
        ratio_sum = sum(chain_ratios.values())
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(f"Ophiuchus ratios must sum to 1.0, got {ratio_sum}")
        encoder_hidden_size = int(encoder_hidden_size)
        word_embeddings = decoder.get_input_embeddings()
        decoder_d_model = int(word_embeddings.embedding_dim)

        self.encoder = encoder
        self.decoder = decoder
        if fixed_receptor_lengths and (residue_cond_mode != "add" or not predict_eos):
            raise ValueError("fixed-canvas v2 requires additive fusion and predict_eos=True")
        if fixed_receptor_lengths and (
            decoder_chain_eos_token_id is None
            or decoder_chain_eos_token_id in {
                decoder_mask_token_id,
                getattr(getattr(decoder, "config", None), "pad_token_id", None),
            }
        ):
            raise ValueError("v2 requires a chain EOS distinct from decoder PAD/MASK")
        self.residue_cond_mode = str(residue_cond_mode)
        self._decoder_mask_token_id = int(decoder_mask_token_id)
        self._encoder_mask_token_id = int(encoder_mask_token_id)
        self._predict_eos = bool(predict_eos)
        self._time_epsilon = float(time_epsilon)
        self._loss_norm = str(loss_norm)
        self._train_objective = str(train_objective)
        self._bert_mask_ratio = float(bert_mask_ratio)
        self._bert_mask_prob = float(bert_mask_prob)
        self._bert_random_prob = float(bert_random_prob)
        self._bert_all_chains = bool(bert_all_chains)
        self._diffusion_all_chains = bool(diffusion_all_chains)
        self._gidd_uniform_ratio = float(gidd_uniform_ratio)
        self._gidd_gamma = float(gidd_gamma)
        self._chain_ratios = chain_ratios
        self._focal = bool(focal)
        self._focal_gamma = float(focal_gamma)
        self._loss_weight_type = loss_weight_type
        self._softmin_snr = float(softmin_snr)
        self._heavy_loss_weight = float(heavy_loss_weight)
        self._light_loss_weight = float(light_loss_weight)
        self._relation_aux = relation_aux
        self._relation_aux_weight = float(relation_aux_weight)
        self._relation_aux_temperature = float(relation_aux_temperature)
        self._last_relation_aux_loss: float | None = None
        # Non-persistent buffer: rides .to(device) with the model but is not saved
        # into checkpoints (integer, unaffected by .to(bfloat16)). Used to draw the
        # BERT "10% random" residue replacements in LLaDA id space.
        if residue_token_ids:
            self.register_buffer(
                "_residue_token_ids",
                torch.as_tensor(sorted(set(int(i) for i in residue_token_ids)), dtype=torch.long),
                persistent=False,
            )
        else:
            self._residue_token_ids = None
        self.condition_norm = (
            nn.LayerNorm(encoder_hidden_size) if condition_norm else None
        )
        self.condition_proj = nn.Linear(encoder_hidden_size, decoder_d_model, bias=False)
        # encode/gather/build_mask do not read self.config; stub satisfies Trainer
        # special-token alignment (eos/pad/bos) plus our training knobs.
        dec_cfg = getattr(decoder, "config", None)
        if fixed_receptor_lengths and decoder_chain_eos_policy is None:
            native_eos_id = getattr(dec_cfg, "eos_token_id", None)
            if native_eos_id is not None:
                # Eval reconstruction passes the resolved chain EOS ID without a tokenizer.
                decoder_chain_eos_policy = (
                    "native_eos" if decoder_chain_eos_token_id == native_eos_id else "chain_eos"
                )
        self.config = _FusionConfig(
            mask_token_id=self._decoder_mask_token_id,
            time_epsilon=self._time_epsilon,
            loss_norm=self._loss_norm,
            encoder_mask_token_id=self._encoder_mask_token_id,
            residue_cond_mode=self.residue_cond_mode,
            train_objective=self._train_objective,
            bert_all_chains=self._bert_all_chains,
            diffusion_all_chains=self._diffusion_all_chains,
            gidd_uniform_ratio=self._gidd_uniform_ratio,
            gidd_gamma=self._gidd_gamma,
            independent_loss_ratio=self._chain_ratios["independent_loss_ratio"],
            single_chain_ratio=self._chain_ratios["single_chain_ratio"],
            heavy2light_loss_ratio=self._chain_ratios["heavy2light_loss_ratio"],
            light2heavy_loss_ratio=self._chain_ratios["light2heavy_loss_ratio"],
            joint_loss_ratio=self._chain_ratios["joint_loss_ratio"],
            focal=self._focal,
            focal_gamma=self._focal_gamma,
            loss_weight_type=self._loss_weight_type,
            softmin_snr=self._softmin_snr,
            heavy_loss_weight=self._heavy_loss_weight,
            light_loss_weight=self._light_loss_weight,
            relation_aux=self._relation_aux,
            relation_aux_weight=self._relation_aux_weight,
            relation_aux_temperature=self._relation_aux_temperature,
            condition_norm=bool(condition_norm),
            hidden_size=decoder_d_model,
            condition_hidden_size=encoder_hidden_size,
            vocab_size=int(
                getattr(dec_cfg, "vocab_size", word_embeddings.num_embeddings)
            ),
            pad_token_id=int(getattr(dec_cfg, "pad_token_id", 1) or 1),
            eos_token_id=getattr(dec_cfg, "eos_token_id", None),
            bos_token_id=getattr(dec_cfg, "bos_token_id", None),
            predict_eos=self._predict_eos,
            forbidden_target_token_ids=None,
            fixed_receptor_lengths=bool(fixed_receptor_lengths),
            fixed_heavy_encoder_length=FIXED_HEAVY_ENCODER_LENGTH,
            fixed_light_encoder_length=FIXED_LIGHT_ENCODER_LENGTH,
            fixed_heavy_decoder_slots=FIXED_HEAVY_DECODER_SLOTS,
            fixed_light_decoder_slots=FIXED_LIGHT_DECODER_SLOTS,
            fixed_heavy_residue_max=FIXED_HEAVY_DECODER_SLOTS - 1,
            fixed_light_residue_max=FIXED_LIGHT_DECODER_SLOTS - 1,
            decoder_chain_eos_token_id=decoder_chain_eos_token_id,
            decoder_chain_eos_policy=decoder_chain_eos_policy,
        )
        if fixed_receptor_lengths:
            if self._residue_token_ids is None:
                raise ValueError("v2 requires decoder residue_token_ids")
            self.register_buffer(
                "_canvas_token_ids",
                torch.cat([self._residue_token_ids, torch.tensor([decoder_chain_eos_token_id])]),
                persistent=False,
            )
        else:
            self._canvas_token_ids = self._residue_token_ids
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def _denoise(
        self,
        input_ids: torch.Tensor | None = None,
        diffusion_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        chain_ids: torch.Tensor | None = None,
        position_ids_inner: torch.Tensor | None = None,
        position_ids_chain: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        residue_mask: torch.Tensor | None = None,
        encoder_input_ids: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        encoder_residue_mask: torch.Tensor | None = None,
        encoder_slot_mask: torch.Tensor | None = None,
        chain_slot_mask: torch.Tensor | None = None,
        encoder_chain_mask: torch.Tensor | None = None,
        encoder_position_ids: torch.Tensor | None = None,
        encoder_kwargs: dict[str, Any] | None = None,
        output_hidden_states: bool = False,
        **_: Any,
    ) -> BioSeqDiffusionOutput:
        _ = diffusion_state, position_ids_chain, timesteps
        if input_ids is None:
            raise ValueError("LLaDAEsmcFusion requires input_ids")

        if self.config.fixed_receptor_lengths:
            if encoder_slot_mask is None or chain_slot_mask is None:
                raise ValueError("v2 requires length-independent encoder_slot_mask and chain_slot_mask")
            # Use only layout-derived masks for features. Clean AA/EOS masks are
            # labels, not permissible conditioning information about hidden length.
            encoder_residue_mask = encoder_slot_mask
            residue_mask = chain_slot_mask

        word_embeddings = self.decoder.get_input_embeddings()
        inputs_embeds = word_embeddings(input_ids)
        skip_condition = (
            self.residue_cond_mode == "token" or encoder_input_ids is None
        )

        if not skip_condition:
            if encoder_position_ids is None and chain_ids is None:
                raise ValueError(
                    "chain_ids or encoder_position_ids are required for encoder conditions"
                )
            effective_encoder_residue_mask = encoder_residue_mask
            if encoder_position_ids is not None and residue_mask is not None:
                effective_encoder_residue_mask = residue_mask.unsqueeze(1)

            chain_token_condition = self.encode_chain_tokens(
                encoder_input_ids=encoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                encoder_residue_mask=effective_encoder_residue_mask,
                encoder_chain_mask=encoder_chain_mask,
                encoder_kwargs=encoder_kwargs,
            )
            if encoder_position_ids is not None:
                token_condition = self.gather_proxy_token_condition(
                    chain_token_condition,
                    encoder_position_ids=encoder_position_ids,
                    attention_mask=attention_mask,
                    residue_mask=residue_mask,
                )
            else:
                token_condition = self.gather_token_condition(
                    chain_token_condition,
                    chain_ids=chain_ids,
                    position_ids_inner=position_ids_inner,
                    attention_mask=attention_mask,
                    encoder_residue_mask=encoder_residue_mask,
                )
            condition_mask = self.build_encoder_condition_mask(
                chain_ids=chain_ids,
                position_ids_inner=position_ids_inner,
                attention_mask=attention_mask,
                encoder_position_ids=encoder_position_ids,
                residue_mask=residue_mask,
            )

            # Run norm/projection in the fusion params' own dtype (fp32 by default,
            # or bf16 under FSDP mixed precision), then cast back to the (possibly
            # bf16) decoder embedding dtype. The decoder may be loaded in bf16 while
            # these small heads stay fp32, so casting is required to avoid a
            # LayerNorm/Linear dtype mismatch.
            proj_dtype = self.condition_proj.weight.dtype
            cond = token_condition.to(dtype=proj_dtype)
            if self.condition_norm is not None:
                cond = self.condition_norm(cond)
            cond_h = self.condition_proj(cond).to(dtype=inputs_embeds.dtype)
            m = condition_mask.to(dtype=inputs_embeds.dtype).unsqueeze(-1)
            # v2 additive fusion preserves both LLaDA token identity and ESMC
            # condition. ``m`` limits the condition to mapped receptor residue
            # positions; it never changes the base wte at special/pad tokens.
            if self.residue_cond_mode == "feature":  # legacy checkpoints only
                inputs_embeds = inputs_embeds * (1.0 - m) + cond_h * m
            else:
                inputs_embeds = inputs_embeds + cond_h * m

        out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )
        logits = out.logits
        if self.config.fixed_receptor_lengths:
            # AA/EOS slots cannot emit text, grammar delimiters, PAD or MASK.
            allowed = torch.zeros(logits.size(-1), device=logits.device, dtype=torch.bool)
            allowed[self._canvas_token_ids.to(device=logits.device)] = True
            if chain_slot_mask is not None:
                logits = logits.masked_fill(
                    chain_slot_mask.bool().unsqueeze(-1) & ~allowed,
                    torch.finfo(logits.dtype).min,
                )
        hidden = None
        if output_hidden_states and getattr(out, "hidden_states", None):
            hidden = out.hidden_states[-1]
        return BioSeqDiffusionOutput(
            loss=None,
            logits=logits,
            hidden_states=hidden,
            encoder_condition=None,
        )

    @torch.no_grad()
    def last_hidden_state(self, **batch: Any) -> torch.Tensor:
        """Clean unmasked fusion hidden states for representation eval."""

        output = self._denoise(output_hidden_states=True, **batch)
        if output.hidden_states is None:
            raise RuntimeError("LLaDA decoder did not return hidden_states")
        return output.hidden_states

    def _diffusion_token_weights(
        self,
        token_t: torch.Tensor,
        zero_loss_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Per-token loss weights for reciprocal / single-chain zero-out.

        Continue-train defaults (``loss_weight_type='none'``, no single-chain
        bucket) return ``None`` so CE is unchanged.
        """

        if self._loss_weight_type == "reciprocal":
            if self._softmin_snr:
                weights = 1.0 / (token_t + 1.0 / float(self._softmin_snr))
            else:
                weights = 1.0 / token_t.clamp(min=self._time_epsilon)
        elif self._loss_weight_type in {"none", "uniform"}:
            if not zero_loss_mask.any():
                return None
            weights = torch.ones_like(token_t)
        else:
            raise ValueError(f"Unsupported loss_weight_type: {self._loss_weight_type!r}")
        return weights.masked_fill(zero_loss_mask, 0.0)

    def forward(self, **batch: Any) -> BioSeqDiffusionOutput:
        """DDP/FSDP-safe entry: HF Trainer calls ``model(**inputs)`` so the wrapper
        hooks fire. Delegates to the masked-diffusion training step."""
        return self.compute_loss(batch)

    def compute_loss(self, batch: dict[str, Any]) -> BioSeqDiffusionOutput:
        if (batch.get("encoder_slot_mask") is not None) != self.config.fixed_receptor_lengths:
            raise ValueError("Batch/model fixed-canvas policies differ")
        # Pair roles are defined on *generated* receptor residues. Snapshot
        # before ``diffusion_all_chains`` widens eligibility, otherwise antigen
        # / MHC / peptide would become "heavy" and the pairing aux would score
        # the wrong chains.
        heavy_pair_mask, light_pair_mask = generated_heavy_light_masks(batch)
        want_aux = (
            bool(self.training)
            and self._relation_aux != "none"
            and self._relation_aux_weight > 0.0
        )
        if self._train_objective == "bert":
            noised_input_ids, labels, corruption_mask = sample_bioseq_bert_noise(
                batch=batch,
                mask_token_id=self._decoder_mask_token_id,
                mask_ratio=self._bert_mask_ratio,
                mask_prob=self._bert_mask_prob,
                random_prob=self._bert_random_prob,
                residue_token_ids=self._residue_token_ids,
                all_chain_targets=self._bert_all_chains,
            )
            timesteps = None
        else:
            if self._diffusion_all_chains:
                # Both the timestep sampler and the noise sampler re-derive
                # eligibility from these two masks, so widening them here is what
                # pulls the fixed context into corruption, labels and the encoder
                # mirror. Nothing else about the objective changes: still one
                # t ~ U(eps, 1) per sequence and Bernoulli(t) absorbing masking.
                eligible_mask = all_residue_eligible_mask(batch)
                batch = {
                    **batch,
                    "diffusion_eligible_mask": eligible_mask,
                    "diffusion_loss_mask": eligible_mask,
                }
            timed = sample_chain_conditioned_timesteps(
                batch,
                ratios=self._chain_ratios,
                time_epsilon=self._time_epsilon,
                stage="val" if not self.training else "train",
            )
            noised_input_ids, labels, corruption_mask, timesteps = sample_bioseq_diffusion_noise(
                batch=batch,
                mask_token_id=self._decoder_mask_token_id,
                time_epsilon=self._time_epsilon,
                uniform_ratio=self._gidd_uniform_ratio,
                gidd_gamma=self._gidd_gamma,
                residue_token_ids=self._residue_token_ids,
                token_t=timed.token_t,
            )
        # For BERT, corruption_mask is the full selection (all 15%): mirror it onto
        # the encoder so the frozen ESMC never sees a target residue it must predict
        # (covers the 10% "kept" / 10% "random" decoder positions too).
        if self.residue_cond_mode != "token":
            noised_encoder_input_ids = apply_decoder_corruption_to_encoder(
                batch=batch,
                corruption_mask=corruption_mask,
                mask_token_id=self._encoder_mask_token_id,
            )
        else:
            noised_encoder_input_ids = None

        output = self._denoise(
            input_ids=noised_input_ids,
            attention_mask=batch.get("attention_mask"),
            chain_ids=batch.get("chain_ids"),
            position_ids_inner=batch.get("position_ids_inner"),
            position_ids_chain=batch.get("position_ids_chain"),
            timesteps=timesteps,
            residue_mask=batch.get("residue_mask"),
            encoder_input_ids=noised_encoder_input_ids,
            encoder_attention_mask=batch.get("encoder_attention_mask"),
            encoder_residue_mask=batch.get("encoder_residue_mask"),
            encoder_slot_mask=batch.get("encoder_slot_mask"),
            chain_slot_mask=batch.get("chain_slot_mask"),
            encoder_chain_mask=batch.get("encoder_chain_mask"),
            encoder_position_ids=batch.get("encoder_position_ids"),
            output_hidden_states=want_aux,
        )
        if self._train_objective == "bert":
            recon = compute_masked_cross_entropy(
                output.logits,
                labels,
                loss_norm=self._loss_norm,
                forbidden_token_ids=None,
            )
        else:
            token_weights = self._diffusion_token_weights(
                timed.token_t, timed.zero_loss_mask
            )
            # Continue-train defaults (both chain weights 1.0) keep the original
            # pooled CE. Passing the masks would switch to AirGen's sum of
            # per-chain means and change the loss scale.
            use_chain_loss = (
                self._heavy_loss_weight != 1.0 or self._light_loss_weight != 1.0
            )
            recon = compute_masked_cross_entropy(
                output.logits,
                labels,
                loss_norm=self._loss_norm,
                forbidden_token_ids=None,
                focal=self._focal,
                focal_gamma=self._focal_gamma,
                token_weights=token_weights,
                heavy_mask=timed.heavy_mask if use_chain_loss else None,
                light_mask=timed.light_mask if use_chain_loss else None,
                heavy_loss_weight=self._heavy_loss_weight,
                light_loss_weight=self._light_loss_weight,
            )
        aux = recon.new_zeros(())
        if want_aux:
            if output.hidden_states is None:
                raise RuntimeError(
                    "relation_aux requested hidden_states but the LLaDA decoder "
                    "did not return them"
                )
            aux = compute_relation_aux(
                self._relation_aux,
                output.hidden_states,
                heavy_pair_mask,
                light_pair_mask,
                residue_mask=batch.get("residue_mask"),
                attention_mask=batch.get("attention_mask"),
                corruption_mask=corruption_mask,
                temperature=self._relation_aux_temperature,
            )
            loss = recon + aux.to(dtype=recon.dtype) * self._relation_aux_weight
            self._last_relation_aux_loss = float(aux.detach().float().item())
        else:
            loss = recon
            # Eval stays reconstruction-only so top-k ranking is unchanged.
            self._last_relation_aux_loss = None
        return BioSeqDiffusionOutput(
            loss=loss,
            logits=output.logits,
            hidden_states=output.hidden_states,
            noised_input_ids=noised_input_ids,
            labels=labels,
            corruption_mask=corruption_mask,
            timesteps=timesteps,
            noised_encoder_input_ids=noised_encoder_input_ids,
            encoder_condition=output.encoder_condition,
            relation_aux_loss=aux.detach() if want_aux else None,
        )
