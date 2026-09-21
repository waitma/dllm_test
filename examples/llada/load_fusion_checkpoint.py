"""Load an ESMC×LLaDA fusion checkpoint for frozen downstream eval.

Training writes HuggingFace Trainer dirs (`model.safetensors` + tokenizer),
not the old grammar_v2 ``best.pt`` blobs. This loader rebuilds
``LLaDAEsmcFusion`` from tensor shapes, loads weights, and returns a
grammar-v2 collator that remaps decoder ids into LLaDA space.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DEFAULT_ESMC = PROJECT_ROOT / "model_weights" / "esmc" / "ESMC-300M"
DEFAULT_LLADA_ID = "GSAI-ML/LLaDA-8B-Base"
DEFAULT_LLADA_SNAP = Path(
    "/vepfs-mlp2/c20250601/251105016/conda/cache/huggingface/hub/"
    "models--GSAI-ML--LLaDA-8B-Base/snapshots/"
    "0f2787f2d87eac5eed8a087d5ecd24277e6255b2"
)


def is_fusion_checkpoint(path: str | Path) -> bool:
    """True when ``path`` is a fusion Trainer dir (or its ``model.safetensors``)."""

    p = Path(path)
    if p.is_file() and p.name == "model.safetensors":
        p = p.parent
    return p.is_dir() and (p / "model.safetensors").is_file()


def resolve_fusion_dir(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file() and p.name == "model.safetensors":
        p = p.parent
    if not (p.is_dir() and (p / "model.safetensors").is_file()):
        raise FileNotFoundError(f"not a fusion checkpoint dir: {path}")
    return p.resolve()


def fusion_weight_file(path: str | Path) -> Path:
    return resolve_fusion_dir(path) / "model.safetensors"


_FIXED_POLICY_CONSTANTS = {
    "fixed_heavy_encoder_length": 168,
    "fixed_light_encoder_length": 136,
    "fixed_heavy_decoder_slots": 167,
    "fixed_light_decoder_slots": 135,
    "fixed_heavy_residue_max": 166,
    "fixed_light_residue_max": 134,
}


def _read_fusion_policy(config: dict[str, Any], checkpoint_dir: Path) -> dict[str, Any]:
    """Normalize sidecar metadata, retaining sidecar-less legacy defaults."""

    fixed = bool(config.get("fixed_receptor_lengths", False))
    for key, expected in _FIXED_POLICY_CONSTANTS.items():
        if key in config and int(config[key]) != expected:
            raise ValueError(
                f"fusion checkpoint has incompatible {key}={config[key]!r}; "
                f"expected {expected}: {checkpoint_dir}"
            )
        if fixed and key not in config:
            raise ValueError(
                f"fixed-canvas checkpoint is missing {key}: {checkpoint_dir}"
            )

    predict_eos = bool(config.get("predict_eos", False))
    if fixed and not predict_eos:
        raise ValueError(
            f"fixed-canvas checkpoint must enable predict_eos: {checkpoint_dir}"
        )
    if not fixed and predict_eos:
        raise ValueError(
            f"legacy checkpoint cannot enable predict_eos without fixed_receptor_lengths: {checkpoint_dir}"
        )

    eos_id = config.get("decoder_chain_eos_token_id")
    eos_policy = config.get("decoder_chain_eos_policy")
    if fixed and (eos_id is None or eos_policy is None):
        raise ValueError(
            "fixed-canvas checkpoint is missing decoder_chain_eos_token_id/policy: "
            f"{checkpoint_dir}"
        )

    residue_cond_mode = str(config.get("residue_cond_mode", "add"))
    if fixed and residue_cond_mode != "add":
        raise ValueError(
            "fixed-canvas checkpoint must use residue_cond_mode='add': "
            f"{checkpoint_dir}"
        )
    return {
        "fixed_receptor_lengths": fixed,
        "predict_eos": predict_eos,
        "decoder_chain_eos_token_id": (
            None if eos_id is None else int(eos_id)
        ),
        "decoder_chain_eos_policy": (
            None if eos_policy is None else str(eos_policy)
        ),
        "condition_norm": bool(config.get("condition_norm", True)),
        "residue_cond_mode": residue_cond_mode,
    }


@dataclass
class FusionEvalBundle:
    """Everything downstream eval needs for a fusion checkpoint."""

    model: Any
    grammar_tokenizer: Any
    llada_tokenizer: Any
    collator: Any
    checkpoint_dir: Path
    d_model: int
    n_layers: int
    encoder_hidden: int
    residue_cond_mode: str


def _infer_decoder_shape(safetensors_path: Path) -> dict[str, int]:
    from safetensors import safe_open

    with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
        attn = [
            key
            for key in handle.keys()
            if key.startswith("decoder.") and key.endswith("attn_out.weight")
        ]
        if not attn:
            raise RuntimeError(f"no decoder attn_out weights in {safetensors_path}")
        d_model = int(handle.get_tensor(attn[0]).shape[0])
        n_layers = len(attn)
        ff = "decoder.model.transformer.blocks.0.ff_out.weight"
        mlp_hidden = int(handle.get_tensor(ff).shape[1]) if ff in handle.keys() else d_model * 4
        proj = "condition_proj.weight"
        encoder_hidden = (
            int(handle.get_tensor(proj).shape[1]) if proj in handle.keys() else 960
        )
    if d_model == 768:
        n_heads = 12
    elif d_model == 4096:
        n_heads = 32
    else:
        n_heads = max(1, d_model // 64)
    return {
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "mlp_hidden": mlp_hidden,
        "encoder_hidden": encoder_hidden,
    }


def _poison_parameters(module: torch.nn.Module) -> None:
    """Fill every float parameter with NaN so unloaded tensors are detectable.

    ``LLaDAModelLM(cfg, init_params=False)`` allocates without initialising, so a
    tensor the checkpoint never writes holds whatever the allocator had -- which
    scores as a random model with no error anywhere. Poisoning first turns that
    silent failure into a provable one (see :func:`_assert_fully_loaded`).
    """

    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.is_floating_point():
                parameter.fill_(float("nan"))


def _assert_fully_loaded(
    model: torch.nn.Module, checkpoint_keys: set[str], weights: Path
) -> None:
    """Require every float parameter to have been written by the checkpoint.

    A surviving NaN proves the tensor was never written (tied weights pass,
    because the tie partner fills the shared storage). Splitting on whether the
    name appears in the checkpoint separates "weight absent" from "weight
    present but already NaN on disk", which need different fixes.
    """

    unwritten: list[str] = []
    nan_on_disk: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        if not bool(torch.isnan(parameter).any()):
            continue
        (nan_on_disk if name in checkpoint_keys else unwritten).append(name)

    if unwritten:
        raise RuntimeError(
            f"{len(unwritten)} parameter(s) absent from {weights} were never "
            f"written and would hold uninitialised memory: {sorted(unwritten)[:12]}"
            + (" ..." if len(unwritten) > 12 else "")
            + ". Refusing to run eval on a partially-loaded model."
        )
    if nan_on_disk:
        raise RuntimeError(
            f"{len(nan_on_disk)} parameter(s) in {weights} contain NaN on disk: "
            f"{sorted(nan_on_disk)[:12]}"
            + (" ..." if len(nan_on_disk) > 12 else "")
            + ". The checkpoint itself is corrupt or training diverged."
        )


def _llada_config_path() -> str:
    snap = DEFAULT_LLADA_SNAP
    if (snap / "config.json").is_file():
        return str(snap)
    return DEFAULT_LLADA_ID


def _build_empty_decoder(
    shape: dict[str, int], dtype: torch.dtype, init_device: str = "cpu"
) -> torch.nn.Module:
    from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
    from dllm.pipelines.llada.models.modeling_llada import (
        LLaDAModel,
        LLaDAModelLM,
        create_model_config_from_pretrained_config,
    )

    cfg = LLaDAConfig.from_pretrained(_llada_config_path(), local_files_only=True)
    cfg.init_device = str(init_device)
    cfg.d_model = int(shape["d_model"])
    cfg.n_layers = int(shape["n_layers"])
    cfg.n_heads = int(shape["n_heads"])
    cfg.n_kv_heads = int(shape["n_heads"])
    cfg.mlp_hidden_size = int(shape["mlp_hidden"])
    # 270m scratch trained with untied ff_out; 8B pretrained keeps config tying
    # unless the checkpoint actually stores a separate transformer-level ff_out.
    if int(shape["d_model"]) == 768:
        cfg.weight_tying = False
    # LLaDAModelLM.__init__ hardcodes init_device="cuda" when it builds the inner
    # model itself, which OOMs an 8B decoder on a busy GPU and silently defeats
    # the caller's device choice. Build the inner model here instead so
    # ``init_device`` is honoured, then hand it over.
    inner_config = create_model_config_from_pretrained_config(cfg)
    inner_config.init_device = str(init_device)
    inner = LLaDAModel(inner_config, init_params=False)
    model = LLaDAModelLM(cfg, model=inner, init_params=False)
    return model.to(dtype=dtype)


def load_fusion_for_eval(
    checkpoint: str | Path,
    device: str | torch.device = "cuda",
    *,
    esmc_path: str | Path | None = None,
    max_length: int = 1024,
    dtype: str | torch.dtype = "bfloat16",
) -> FusionEvalBundle:
    """Rebuild ``LLaDAEsmcFusion`` and a remapping grammar collator."""

    import os
    import sys

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import transformers
    from safetensors.torch import load_file

    from dllm.pipelines.immune_llada.data import (
        GrammarBioSeqCollator,
        GrammarTokenizer,
        HuggingFaceEsmTokenizerAdapter,
    )
    from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import load_local_esmc_encoder
    from examples.llada.protein_fusion_model import (
        LLaDAEsmcFusion,
        RemapCollator,
        RESIDUES,
        build_remap_lookup,
        build_inverse_remap,
        decoder_mask_token_id,
        expand_llada_tokenizer_for_esmc_grammar,
        load_fusion_config,
    )

    ckpt_dir = resolve_fusion_dir(checkpoint)
    weights = fusion_weight_file(ckpt_dir)
    esmc_dir = Path(esmc_path) if esmc_path is not None else DEFAULT_ESMC
    if not (esmc_dir / "config.json").is_file():
        raise FileNotFoundError(f"ESMC config missing: {esmc_dir}")

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    if device == "cuda" or (isinstance(device, str) and device.startswith("cuda")):
        if not torch.cuda.is_available():
            device = "cpu"
    device = torch.device(device)

    config = load_fusion_config(ckpt_dir)
    policy = _read_fusion_policy(config, ckpt_dir)
    fixed_receptor_lengths = bool(policy["fixed_receptor_lengths"])
    saved_eos_id = policy["decoder_chain_eos_token_id"]
    saved_eos_policy = policy["decoder_chain_eos_policy"]
    condition_norm = bool(policy["condition_norm"])
    residue_cond_mode = str(policy["residue_cond_mode"])
    allow_chain_eos_token = fixed_receptor_lengths or saved_eos_policy == "chain_eos"

    shape = _infer_decoder_shape(weights)
    logger.info(
        "Fusion ckpt %s: d_model=%d n_layers=%d n_heads=%d mlp=%d enc_h=%d",
        ckpt_dir,
        shape["d_model"],
        shape["n_layers"],
        shape["n_heads"],
        shape["mlp_hidden"],
        shape["encoder_hidden"],
    )

    tok_src = ckpt_dir if (ckpt_dir / "tokenizer.json").is_file() else _llada_config_path()
    llada_tok = transformers.AutoTokenizer.from_pretrained(
        str(tok_src), padding_side="right", local_files_only=True
    )
    if llada_tok.pad_token is None:
        llada_tok.pad_token = llada_tok.eos_token
    if llada_tok.mask_token is None:
        llada_tok.add_special_tokens({"mask_token": "<|mdm_mask|>"})

    base_esmc = HuggingFaceEsmTokenizerAdapter.from_pretrained(
        esmc_dir, local_files_only=True
    )
    gtok = GrammarTokenizer(base_esmc)
    remap, n_added = expand_llada_tokenizer_for_esmc_grammar(
        llada_tok, gtok, allow_chain_eos_token=allow_chain_eos_token
    )
    lookup = build_remap_lookup(remap)
    actual_eos_id = int(getattr(llada_tok, "_fusion_decoder_chain_eos_token_id"))
    actual_eos_policy = str(getattr(llada_tok, "_fusion_decoder_chain_eos_policy"))
    if saved_eos_id is not None and actual_eos_id != int(saved_eos_id):
        raise ValueError(
            "decoder chain EOS id mismatch between checkpoint sidecar and tokenizer: "
            f"sidecar={saved_eos_id}, tokenizer={actual_eos_id}"
        )
    if saved_eos_policy is not None and actual_eos_policy != str(saved_eos_policy):
        raise ValueError(
            "decoder chain EOS policy mismatch between checkpoint sidecar and tokenizer: "
            f"sidecar={saved_eos_policy!r}, tokenizer={actual_eos_policy!r}"
        )
    if fixed_receptor_lengths and actual_eos_policy == "legacy_eos_alias":
        raise ValueError(
            "fixed-canvas checkpoint resolves chain EOS through the legacy pad/EOS "
            f"alias instead of a safe policy: {ckpt_dir}"
        )

    decoder = _build_empty_decoder(shape, torch_dtype, init_device=str(device))
    embed_rows = int(decoder.get_input_embeddings().weight.shape[0])
    if len(llada_tok) > embed_rows:
        decoder.resize_token_embeddings(len(llada_tok))
        embed_rows = int(decoder.get_input_embeddings().weight.shape[0])
    if getattr(decoder, "config", None) is not None:
        decoder.config.use_cache = False

    encoder = load_local_esmc_encoder(esmc_dir)
    with (esmc_dir / "config.json").open() as handle:
        encoder_hidden = int(json.load(handle)["d_model"])
    residue_token_ids = [
        int(llada_tok.convert_tokens_to_ids(f"<res_{aa}>")) for aa in RESIDUES
    ]
    model = LLaDAEsmcFusion(
        decoder=decoder,
        encoder=encoder,
        encoder_hidden_size=encoder_hidden,
        decoder_mask_token_id=decoder_mask_token_id(llada_tok),
        encoder_mask_token_id=int(gtok.mask_token_id),
        residue_cond_mode=residue_cond_mode,
        condition_norm=condition_norm,
        freeze_encoder=True,
        train_objective="diffusion",
        residue_token_ids=residue_token_ids,
        fixed_receptor_lengths=fixed_receptor_lengths,
        predict_eos=bool(policy["predict_eos"]),
        decoder_chain_eos_token_id=(
            actual_eos_id if fixed_receptor_lengths else None
        ),
    )
    model = model.to(dtype=torch_dtype)

    load_device = str(device) if device.type == "cuda" else "cpu"
    state = load_file(str(weights), device=load_device)
    _poison_parameters(model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(
            f"unexpected fusion keys in {weights}: {unexpected[:12]}"
        )
    checkpoint_keys = set(state)
    del state
    _assert_fully_loaded(model, checkpoint_keys, weights)
    if missing:
        # Reached only for tied weights and non-persistent buffers; anything that
        # would have stayed uninitialised already raised above.
        logger.info(
            "fusion keys absent from checkpoint but resolved (%d): %s",
            len(missing),
            missing[:12],
        )

    inverse = build_inverse_remap(
        remap,
        # In legacy checkpoints native pad and EOS may alias. Preserve the
        # historical inverse mapping by preferring grammar pad; v2 has a
        # dedicated chain-EOS id and therefore has no such collision.
        preferred_source_ids=(int(gtok.pad_token_id),)
        if not fixed_receptor_lengths
        else (),
        size=embed_rows,
    )
    model.register_buffer("llada_to_grammar_ids", inverse, persistent=False)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = model.to(device=device)

    base_collator = GrammarBioSeqCollator(
        tokenizer=gtok,
        max_sequence_length=int(max_length),
        max_protein_length=int(max_length),
        fixed_receptor_lengths=fixed_receptor_lengths,
    )
    collator = RemapCollator(base_collator, lookup)

    return FusionEvalBundle(
        model=model,
        grammar_tokenizer=gtok,
        llada_tokenizer=llada_tok,
        collator=collator,
        checkpoint_dir=ckpt_dir,
        d_model=int(shape["d_model"]),
        n_layers=int(shape["n_layers"]),
        encoder_hidden=int(encoder_hidden),
        residue_cond_mode=residue_cond_mode,
    )
