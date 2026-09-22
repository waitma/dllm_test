"""CPU checkpoint-loader regressions without pretrained model weights.

After activating ``pllm``, run from the repository root::

    python -m pytest scripts/tests/immune_llada/test_v2_token_checkpoint.py -q

The decoder is tiny; fusion construction uses a stub for loader edge cases and
the real LLaDAEsmcFusion for roundtrip/denoise integration. Safetensors, local
tokenizers, grammar, remapping, and integrity checks use production code.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from torch import nn
from transformers import PreTrainedTokenizerFast

from dllm.pipelines.immune_llada.data import (
    GrammarTokenizer,
    HuggingFaceEsmTokenizerAdapter,
)
from dllm.pipelines.immune_llada.data.esm_encoding import Esm2SequenceTokenizer
from dllm.pipelines.qwen3_vl_arch import modeling_bioseq
from downstream.grammar.common import antibody_pair_record
from examples.llada import load_fusion_checkpoint as loader
from examples.llada import protein_fusion_model as fusion
from examples.llada.protein_fusion_model import LLaDAEsmcFusion


class _TinyDecoder(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, 4)
        self.model = nn.Module()
        self.model.transformer = nn.Module()
        block = nn.Module()
        block.attn_out = nn.Linear(4, 4, bias=False)
        block.ff_out = nn.Linear(8, 4, bias=False)
        self.model.transformer.blocks = nn.ModuleList([block])
        self.config = SimpleNamespace(
            vocab_size=vocab_size, pad_token_id=1, eos_token_id=1, use_cache=True
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def resize_token_embeddings(self, vocab_size: int) -> nn.Embedding:
        self.embeddings = nn.Embedding(vocab_size, 4)
        self.config.vocab_size = vocab_size
        return self.embeddings

    def forward(
        self, *, inputs_embeds, attention_mask=None, output_hidden_states=False,
        use_cache=False,
    ):
        assert use_cache is False
        block = self.model.transformer.blocks[0]
        hidden = torch.tanh(block.attn_out(inputs_embeds))
        hidden = hidden + block.ff_out(torch.cat([hidden, hidden], dim=-1))
        return SimpleNamespace(
            logits=torch.nn.functional.linear(hidden, self.embeddings.weight),
            hidden_states=(hidden,) if output_hidden_states else None,
        )


class _TinyFusion(nn.Module):
    """Mirror the agreed optional-encoder state-dict contract, not model math."""

    def __init__(
        self, *, decoder, encoder, encoder_hidden_size, condition_norm, **policy
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        if encoder is None:
            assert policy["residue_cond_mode"] == "token"
            assert encoder_hidden_size == 0
            assert condition_norm is False
            self.condition_proj = None
            self.condition_norm = None
        else:
            self.condition_proj = nn.Linear(encoder_hidden_size, 4, bias=False)
            self.condition_norm = (
                nn.LayerNorm(encoder_hidden_size) if condition_norm else None
            )
        self.config = SimpleNamespace(
            condition_hidden_size=encoder_hidden_size,
            condition_norm=condition_norm,
            **policy,
        )


@pytest.fixture
def checkpoint_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    esmc_dir = tmp_path / "esmc-tokenizer-only"
    esmc_dir.mkdir()
    base = Esm2SequenceTokenizer()
    esmc_tokenizer = Tokenizer(models.WordLevel(base.token_to_id, unk_token="<unk>"))
    esmc_tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    esmc_tokenizer.post_processor = processors.TemplateProcessing(
        single="<cls> $A <eos>",
        special_tokens=[("<cls>", base.cls_token_id), ("<eos>", base.eos_token_id)],
    )
    esmc_tokenizer.save(str(esmc_dir / "tokenizer.json"))
    (esmc_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "ESMCTokenizer"}), encoding="utf-8"
    )
    encoder_loader = Mock(side_effect=AssertionError("ESMC weights must not be loaded"))
    monkeypatch.setattr(modeling_bioseq, "load_local_esmc_encoder", encoder_loader)
    monkeypatch.setattr(fusion, "LLaDAEsmcFusion", _TinyFusion)

    def build_decoder(shape, dtype, init_device):
        assert init_device == "cpu"
        assert shape["d_model"] == 4
        assert shape["mlp_hidden"] == 8
        # Exercise the loader's embedding resize before restoring saved weights.
        return _TinyDecoder(3).to(dtype=dtype)

    monkeypatch.setattr(loader, "_build_empty_decoder", build_decoder)

    def create(
        *, mode="token", fixed=True, with_encoder=False, condition_norm=False,
        sidecar=True,
    ):
        checkpoint = tmp_path / "checkpoint-10"
        checkpoint.mkdir()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                models.WordLevel(
                    {"<unk>": 0, "<eos>": 1, "<|mdm_mask|>": 2}, unk_token="<unk>"
                )
            ),
            unk_token="<unk>", eos_token="<eos>", pad_token="<eos>",
            mask_token="<|mdm_mask|>",
        )
        grammar = GrammarTokenizer(
            HuggingFaceEsmTokenizerAdapter.from_pretrained(esmc_dir)
        )
        fusion.expand_llada_tokenizer_for_esmc_grammar(
            tokenizer, grammar, allow_chain_eos_token=fixed
        )
        tokenizer.save_pretrained(checkpoint)
        encoder = None
        if with_encoder:
            (esmc_dir / "config.json").write_text(
                json.dumps({"d_model": 6}), encoding="utf-8"
            )
            encoder = nn.Linear(6, 6)
            encoder_loader.side_effect = lambda path: nn.Linear(6, 6)
        model = fusion.LLaDAEsmcFusion(
            decoder=_TinyDecoder(len(tokenizer)),
            encoder=encoder,
            encoder_hidden_size=6 if with_encoder else 0,
            condition_norm=condition_norm,
            decoder_mask_token_id=fusion.decoder_mask_token_id(tokenizer),
            encoder_mask_token_id=grammar.mask_token_id,
            residue_token_ids=[
                tokenizer.convert_tokens_to_ids(f"<res_{aa}>") for aa in fusion.RESIDUES
            ],
            residue_cond_mode=mode,
            fixed_receptor_lengths=fixed,
            predict_eos=fixed,
            decoder_chain_eos_token_id=(
                tokenizer._fusion_decoder_chain_eos_token_id if fixed else None
            ),
        )
        if sidecar:
            fusion.save_fusion_config(checkpoint, model, tokenizer=tokenizer)
        state = {key: value.clone() for key, value in model.state_dict().items()}
        weights = checkpoint / "model.safetensors"
        save_file(state, str(weights))
        return SimpleNamespace(
            checkpoint=checkpoint, weights=weights, esmc_dir=esmc_dir,
            encoder_loader=encoder_loader, state=state, tokenizer=tokenizer, model=model,
        )

    return create


def _load(checkpoint, *, dtype="float32"):
    return loader.load_fusion_for_eval(
        checkpoint.weights, device="cpu", esmc_path=checkpoint.esmc_dir, dtype=dtype
    )


def _assert_restored(bundle, checkpoint) -> None:
    assert bundle.checkpoint_dir == checkpoint.checkpoint.resolve()
    assert bundle.d_model == 4
    assert bundle.n_layers == 1
    assert bundle.model.training is False
    assert bundle.model.decoder.config.use_cache is False
    actual = bundle.model.state_dict()
    assert actual.keys() == checkpoint.state.keys()
    for key, expected in checkpoint.state.items():
        torch.testing.assert_close(actual[key], expected.to(dtype=actual[key].dtype))
    for parameter in bundle.model.parameters():
        assert parameter.device.type == "cpu"
        assert not parameter.requires_grad
        assert torch.isfinite(parameter).all()


@pytest.mark.parametrize("saved_norm", [False, True, None])
@pytest.mark.parametrize("dtype", ["float32", torch.bfloat16])
def test_v2_token_loads_without_encoder_assets(
    checkpoint_factory, saved_norm, dtype
) -> None:
    checkpoint = checkpoint_factory()
    config = fusion.load_fusion_config(checkpoint.checkpoint)
    if saved_norm is None:
        del config["condition_norm"]
    else:
        config["condition_norm"] = saved_norm
    (checkpoint.checkpoint / "fusion_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    assert not (checkpoint.esmc_dir / "config.json").exists()
    bundle = _load(checkpoint, dtype=dtype)

    checkpoint.encoder_loader.assert_not_called()
    _assert_restored(bundle, checkpoint)
    assert bundle.encoder_hidden == 0
    assert bundle.residue_cond_mode == "token"
    assert bundle.model.encoder is None
    assert bundle.model.condition_proj is None
    assert bundle.model.condition_norm is None
    assert bundle.model.config.condition_hidden_size == 0
    assert bundle.model.config.condition_norm is False
    assert bundle.model.config.predict_eos is True
    assert not any(
        key.startswith(("encoder.", "condition_proj.", "condition_norm."))
        for key in bundle.model.state_dict()
    )

    batch = bundle.collator([antibody_pair_record("ACDE", "GHIK")])
    assert bundle.collator.base_collator.fixed_receptor_lengths is True
    assert int(batch["chain_slot_mask"].sum()) == 167 + 135
    eos_id = bundle.llada_tokenizer._fusion_decoder_chain_eos_token_id
    assert eos_id != bundle.llada_tokenizer.pad_token_id
    assert batch["input_ids"][batch["chain_eos_mask"].bool()].eq(eos_id).all()
    assert int(bundle.model.llada_to_grammar_ids[eos_id]) == bundle.grammar_tokenizer.eos_token_id
    for aa in fusion.RESIDUES:
        decoder_id = bundle.llada_tokenizer.convert_tokens_to_ids(f"<res_{aa}>")
        assert int(bundle.model.llada_to_grammar_ids[decoder_id]) == (
            bundle.grammar_tokenizer.encode_residues(aa)[0]
        )


@pytest.mark.parametrize("with_encoder", [False, True], ids=["encoder-free", "legacy-encoder"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_token_fusion_load_roundtrip_and_denoise(
    checkpoint_factory, monkeypatch, with_encoder, dtype
):
    # Restore the real class before both checkpoint creation and eval loading.
    monkeypatch.setattr(fusion, "LLaDAEsmcFusion", LLaDAEsmcFusion)
    checkpoint = checkpoint_factory(with_encoder=with_encoder, condition_norm=with_encoder)
    bundle = _load(checkpoint, dtype=dtype)
    assert type(bundle.model) is LLaDAEsmcFusion
    _assert_restored(bundle, checkpoint)
    config = fusion.load_fusion_config(checkpoint.checkpoint)
    assert config["condition_hidden_size"] == bundle.encoder_hidden == (6 if with_encoder else 0)
    assert config["condition_norm"] is bundle.model.config.condition_norm is with_encoder
    assert bundle.model.config.fixed_receptor_lengths is True
    assert bundle.model.config.predict_eos is True
    if with_encoder:
        checkpoint.encoder_loader.assert_called_once_with(checkpoint.esmc_dir)
        assert bundle.model.encoder is not None
        assert bundle.model.condition_proj is not None
        assert bundle.model.condition_norm is not None
    else:
        checkpoint.encoder_loader.assert_not_called()
        assert not (checkpoint.esmc_dir / "config.json").exists()
        assert bundle.model.encoder is None
        assert bundle.model.condition_proj is None
        assert bundle.model.condition_norm is None
        assert all(key.startswith("decoder.") for key in bundle.model.state_dict())

    original = checkpoint.model.to(dtype=dtype).eval()
    encode = Mock(side_effect=AssertionError("token denoise must not use the encoder"))
    monkeypatch.setattr(original, "encode_chain_tokens", encode)
    monkeypatch.setattr(bundle.model, "encode_chain_tokens", encode)
    batch = bundle.collator([antibody_pair_record("ACDE", "GHIK")])
    slots = batch["chain_slot_mask"].bool()
    batch["input_ids"] = batch["input_ids"].masked_fill(
        slots, fusion.decoder_mask_token_id(bundle.llada_tokenizer)
    )
    with torch.no_grad():
        expected = original._denoise(**batch, output_hidden_states=True)
        actual = bundle.model._denoise(**batch, output_hidden_states=True)
        changed = bundle.model._denoise(
            **{**batch, "encoder_input_ids": torch.full_like(batch["encoder_input_ids"], 9999)},
            output_hidden_states=True,
        )
        decoder_output = bundle.model.decoder(
            inputs_embeds=bundle.model.decoder.get_input_embeddings()(batch["input_ids"]),
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
    encode.assert_not_called()
    assert actual.logits.shape == (*batch["input_ids"].shape, len(bundle.llada_tokenizer))
    assert actual.hidden_states.shape == (*batch["input_ids"].shape, 4)
    assert actual.encoder_condition is None
    assert torch.isfinite(actual.logits).all()
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.hidden_states, expected.hidden_states)
    torch.testing.assert_close(actual.logits, changed.logits)
    torch.testing.assert_close(actual.hidden_states, changed.hidden_states)
    torch.testing.assert_close(actual.hidden_states, decoder_output.hidden_states[-1])

    eos_id = bundle.llada_tokenizer._fusion_decoder_chain_eos_token_id
    assert bundle.model.config.decoder_chain_eos_token_id == eos_id
    assert bundle.model.config.decoder_chain_eos_policy == "chain_eos"
    assert int(bundle.model.llada_to_grammar_ids[eos_id]) == bundle.grammar_tokenizer.eos_token_id
    allowed = torch.zeros(len(bundle.llada_tokenizer), dtype=torch.bool)
    allowed[[bundle.llada_tokenizer.convert_tokens_to_ids(f"<res_{aa}>") for aa in fusion.RESIDUES]] = True
    allowed[eos_id] = True
    assert actual.logits[slots][:, ~allowed].eq(torch.finfo(dtype).min).all()
    torch.testing.assert_close(actual.logits[slots][:, allowed], decoder_output.logits[slots][:, allowed])
    torch.testing.assert_close(actual.logits[~slots], decoder_output.logits[~slots])


@pytest.mark.parametrize("with_encoder", [False, True])
def test_infer_shape_uses_zero_only_without_projection(checkpoint_factory, with_encoder):
    checkpoint = checkpoint_factory(with_encoder=with_encoder)
    assert loader._infer_decoder_shape(checkpoint.weights) == {
        "d_model": 4, "n_layers": 1, "n_heads": 1, "mlp_hidden": 8,
        "encoder_hidden": 6 if with_encoder else 0,
    }


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("condition_norm", [False, True])
def test_old_token_checkpoint_restores_all_conditioning_weights(
    checkpoint_factory, fixed, condition_norm
):
    checkpoint = checkpoint_factory(
        with_encoder=True, fixed=fixed, condition_norm=condition_norm
    )
    bundle = _load(checkpoint)
    checkpoint.encoder_loader.assert_called_once_with(checkpoint.esmc_dir)
    _assert_restored(bundle, checkpoint)
    assert bundle.residue_cond_mode == "token"
    assert bundle.encoder_hidden == 6
    assert bundle.model.encoder is not None
    assert bundle.model.condition_proj is not None
    assert bundle.model.config.condition_norm is condition_norm
    assert bundle.model.config.predict_eos is fixed
    if not fixed:
        # Retain legacy EOS/PAD inverse-map precedence instead of growing vocab.
        assert int(bundle.model.llada_to_grammar_ids[1]) == bundle.grammar_tokenizer.pad_token_id


@pytest.mark.parametrize(
    ("mode", "fixed", "sidecar"),
    [("add", False, False), ("feature", False, True), ("add", True, True)],
)
def test_existing_fusion_checkpoint_modes_still_load(
    checkpoint_factory, mode, fixed, sidecar
):
    checkpoint = checkpoint_factory(
        mode=mode, fixed=fixed, with_encoder=True, condition_norm=True, sidecar=sidecar
    )
    bundle = _load(checkpoint)
    checkpoint.encoder_loader.assert_called_once_with(checkpoint.esmc_dir)
    _assert_restored(bundle, checkpoint)
    assert bundle.residue_cond_mode == mode
    assert bundle.model.config.fixed_receptor_lengths is fixed


@pytest.mark.parametrize("with_encoder", [False, True])
@pytest.mark.parametrize("corruption", ["missing", "unexpected", "nan", "shape"])
def test_token_load_keeps_parameter_integrity_checks(
    checkpoint_factory, with_encoder, corruption
):
    checkpoint = checkpoint_factory(with_encoder=with_encoder, condition_norm=with_encoder)
    key = "condition_proj.weight" if with_encoder else "decoder.embeddings.weight"
    if corruption == "missing":
        del checkpoint.state[key]
        error = "absent.*never written"
    elif corruption == "unexpected":
        checkpoint.state["unknown.weight"] = torch.ones(1)
        error = "unexpected fusion keys"
    elif corruption == "nan":
        checkpoint.state[key].fill_(float("nan"))
        error = "contain NaN on disk"
    else:
        checkpoint.state[key] = checkpoint.state[key][:-1].clone()
        error = "size mismatch"
    save_file(checkpoint.state, str(checkpoint.weights))
    with pytest.raises(RuntimeError, match=error):
        _load(checkpoint)
    if not with_encoder:
        checkpoint.encoder_loader.assert_not_called()


def test_token_checkpoint_with_orphan_projection_is_not_silently_accepted(checkpoint_factory):
    checkpoint = checkpoint_factory()
    checkpoint.state["condition_proj.weight"] = torch.ones(4, 6)
    save_file(checkpoint.state, str(checkpoint.weights))
    with pytest.raises(RuntimeError, match="unexpected fusion keys.*condition_proj"):
        _load(checkpoint)
    checkpoint.encoder_loader.assert_not_called()


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("decoder_chain_eos_token_id", 1000, "EOS id mismatch"),
        ("decoder_chain_eos_policy", "native_eos", "EOS policy mismatch"),
        ("predict_eos", False, "must enable predict_eos"),
        ("residue_cond_mode", "feature", "residue_cond_mode"),
        ("fixed_heavy_decoder_slots", 166, "incompatible fixed_heavy_decoder_slots"),
        ("decoder_chain_eos_token_id", None, "missing decoder_chain_eos_token_id/policy"),
    ],
)
def test_v2_token_keeps_policy_and_tokenizer_validation(
    checkpoint_factory, key, value, error
):
    checkpoint = checkpoint_factory()
    config = fusion.load_fusion_config(checkpoint.checkpoint)
    config[key] = value
    (checkpoint.checkpoint / "fusion_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=error):
        _load(checkpoint)
    checkpoint.encoder_loader.assert_not_called()
