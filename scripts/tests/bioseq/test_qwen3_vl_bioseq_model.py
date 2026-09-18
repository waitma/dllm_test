"""Tests for the active BioSeq diffusion models with shared immune data.

Run in the ``pllm`` environment::

    python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py -q
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from dllm.pipelines.immune_llada.data import (
    BioSeqChain,
    BioSeqRecord,
    Esm2SequenceTokenizer,
    GrammarBioSeqCollator,
    GrammarTokenizer,
)
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (
    BioSeqDiffusionDecoder,
    BioSeqDiffusionTransformerConfig,
    BioSeqEncoderDiffusionModel,
    BioSeqLLaDAEncoderDiffusionModel,
    BioSeqLLaDA2EncoderDiffusionModel,
    BioSeqNoEncoderDiffusionModel,
    _convert_biohub_esmc_state_dict,
    apply_decoder_corruption_to_encoder,
    compute_masked_cross_entropy,
    forbidden_diffusion_target_token_ids,
    load_local_esm2_encoder,
    mask_forbidden_target_logits,
    sample_bioseq_diffusion_noise,
)

ESM2_650M_DIR = Path(
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esm2/esm2_t33_650M_UR50D"
)


class TinyEncoder(nn.Module):
    def __init__(self, vocab_size: int = 33, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> SimpleNamespace:
        hidden_states = self.embeddings(input_ids)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden_states)


_GRAMMAR_TOKENIZER = GrammarTokenizer(Esm2SequenceTokenizer())


def tiny_config(**overrides) -> BioSeqDiffusionTransformerConfig:
    values = {
        "vocab_size": _GRAMMAR_TOKENIZER.vocab_size,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "dropout": 0.0,
        "max_position_embeddings": 128,
        "max_chain_positions": 8,
        "max_chain_roles": 16,
        "max_task_types": 16,
        "time_epsilon": 0.75,
    }
    values.update(overrides)
    return BioSeqDiffusionTransformerConfig(**values)


def antibody_antigen_batch() -> dict[str, torch.Tensor]:
    records = [
        BioSeqRecord(
            chains=[
                BioSeqChain("QVQLVQSGAE", "antibody_heavy"),
                BioSeqChain("DIQMTQSPSS", "antibody_light"),
                BioSeqChain("MKTAYIAKQRQISFVKSHFS", "antigen"),
            ],
            task_type="antibody_antigen",
            source="unit",
            labels={"relation": "binding"},
        ),
        BioSeqRecord(
            chains=[
                BioSeqChain("EVQLVESGGG", "antibody_heavy"),
                BioSeqChain("EIVLTQSPAT", "antibody_light"),
                BioSeqChain("GILGFVFTLTVPSER", "antigen"),
            ],
            task_type="antibody_antigen",
            source="unit",
            labels={"relation": "binding"},
        ),
    ]
    return GrammarBioSeqCollator(_GRAMMAR_TOKENIZER)(records)


def test_no_encoder_model_compute_loss_runs_and_respects_fixed_context() -> None:
    torch.manual_seed(0)
    batch = antibody_antigen_batch()
    model = BioSeqNoEncoderDiffusionModel(tiny_config())

    output = model.compute_loss(batch)

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert output.logits.shape == (*batch["input_ids"].shape, tiny_config().vocab_size)
    assert output.corruption_mask is not None
    assert (output.corruption_mask & ~batch["diffusion_loss_mask"]).sum().item() == 0
    assert (output.corruption_mask & batch["fixed_context_mask"]).sum().item() == 0
    assert output.corruption_mask.sum().item() >= batch["input_ids"].shape[0]


def test_no_encoder_model_supports_gradient_checkpointing_backward() -> None:
    torch.manual_seed(0)
    batch = antibody_antigen_batch()
    model = BioSeqNoEncoderDiffusionModel(tiny_config(gradient_checkpointing=True))

    output = model.compute_loss(batch)
    assert output.loss is not None
    output.loss.backward()

    assert model.decoder.token_embeddings.weight.grad is not None
    assert torch.isfinite(model.decoder.token_embeddings.weight.grad).all()


def test_gradient_checkpointing_matches_regular_backward() -> None:
    torch.manual_seed(0)
    config = tiny_config(num_hidden_layers=3)
    regular = BioSeqDiffusionDecoder(config).train()
    checkpointed = BioSeqDiffusionDecoder(replace(config, gradient_checkpointing=True)).train()
    checkpointed.load_state_dict(regular.state_dict())

    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)
    timesteps = torch.tensor([0.2, 0.8])

    regular_logits = regular(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids_inner=position_ids,
        timesteps=timesteps,
    ).logits
    checkpointed_logits = checkpointed(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids_inner=position_ids,
        timesteps=timesteps,
    ).logits
    torch.testing.assert_close(checkpointed_logits, regular_logits)

    regular_logits.square().mean().backward()
    checkpointed_logits.square().mean().backward()
    regular_grads = dict(regular.named_parameters())
    checkpointed_grads = dict(checkpointed.named_parameters())
    for name, parameter in regular_grads.items():
        if parameter.grad is None:
            assert checkpointed_grads[name].grad is None
            continue
        assert checkpointed_grads[name].grad is not None
        torch.testing.assert_close(checkpointed_grads[name].grad, parameter.grad, rtol=1e-4, atol=1e-6)


def test_decoder_initialization_starts_near_uniform_cross_entropy() -> None:
    torch.manual_seed(0)
    config = tiny_config(vocab_size=64, hidden_size=128, num_hidden_layers=8, intermediate_size=512)
    decoder = BioSeqDiffusionDecoder(config).eval()
    input_ids = torch.randint(4, 32, (4, 32))
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)
    labels = torch.randint(0, config.vocab_size, input_ids.shape)

    logits = decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids_inner=position_ids,
        timesteps=torch.rand(input_ids.shape[0]),
    ).logits
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())

    assert torch.isfinite(loss)
    assert loss.item() < 6.0
    assert logits.std().item() < 1.0


def test_encoder_model_compute_loss_uses_diffusion_state_token_features() -> None:
    torch.manual_seed(0)
    batch = antibody_antigen_batch()
    encoder = TinyEncoder(vocab_size=_GRAMMAR_TOKENIZER.vocab_size, hidden_size=32)
    model = BioSeqEncoderDiffusionModel(tiny_config(), encoder=encoder, freeze_encoder=False)

    output = model.compute_loss(batch)

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert output.noised_encoder_input_ids is not None
    assert output.corruption_mask is not None
    expected_encoder_ids = apply_decoder_corruption_to_encoder(
        batch,
        corruption_mask=output.corruption_mask,
        mask_token_id=model.config.mask_token_id,
    )
    assert torch.equal(output.noised_encoder_input_ids, expected_encoder_ids)
    assert output.encoder_condition is not None
    assert output.encoder_condition.shape == (*batch["input_ids"].shape, 32)

    special_positions = batch["attention_mask"] & batch["position_ids_inner"].lt(0)
    assert output.encoder_condition[special_positions].abs().sum().item() == 0

    output.loss.backward()
    assert encoder.embeddings.weight.grad is not None
    assert torch.isfinite(encoder.embeddings.weight.grad).all()


def test_condition_norm_disabled_by_default() -> None:
    """Backward-compat: no condition LayerNorm unless explicitly enabled."""
    model = BioSeqEncoderDiffusionModel(
        tiny_config(), encoder=TinyEncoder(vocab_size=_GRAMMAR_TOKENIZER.vocab_size, hidden_size=32)
    )
    assert model.config.condition_norm is False
    assert model.decoder.condition_norm is None


def test_condition_norm_rescales_squashed_encoder_condition() -> None:
    """A tiny-scale encoder output (mimicking ESMC's gamma~0.04 final norm) must
    be renormalized to a usable scale before injection when condition_norm=True."""
    torch.manual_seed(0)
    batch = antibody_antigen_batch()

    class TinySquashedEncoder(nn.Module):
        def __init__(self, vocab_size: int, hidden_size: int) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=hidden_size)
            self.embeddings = nn.Embedding(vocab_size, hidden_size)

        def forward(self, input_ids, attention_mask=None):
            hidden = self.embeddings(input_ids) * 0.04  # squash like ESMC post-norm
            if attention_mask is not None:
                hidden = hidden * attention_mask.to(hidden.dtype).unsqueeze(-1)
            return SimpleNamespace(last_hidden_state=hidden)

    encoder = TinySquashedEncoder(_GRAMMAR_TOKENIZER.vocab_size, 32)
    model = BioSeqEncoderDiffusionModel(
        tiny_config(condition_norm=True), encoder=encoder, freeze_encoder=False
    )
    assert isinstance(model.decoder.condition_norm, nn.LayerNorm)

    output = model.compute_loss(batch)
    assert output.loss is not None and torch.isfinite(output.loss)

    residue = batch["attention_mask"] & batch["position_ids_inner"].ge(0)
    raw = output.encoder_condition[residue]
    normed = model.decoder.condition_norm(raw.to(model.decoder.condition_norm.weight.dtype))
    # Raw squashed condition has tiny std; LayerNorm restores ~unit scale.
    assert raw.std().item() < 0.1
    assert normed.std().item() > 0.5

    output.loss.backward()
    assert encoder.embeddings.weight.grad is not None
    assert torch.isfinite(encoder.embeddings.weight.grad).all()


def test_encoder_model_can_freeze_encoder() -> None:
    torch.manual_seed(0)
    batch = antibody_antigen_batch()
    encoder = TinyEncoder(vocab_size=_GRAMMAR_TOKENIZER.vocab_size, hidden_size=32)
    model = BioSeqEncoderDiffusionModel(tiny_config(), encoder=encoder, freeze_encoder=True)

    output = model.compute_loss(batch)
    assert output.loss is not None
    output.loss.backward()

    assert encoder.embeddings.weight.requires_grad is False
    assert encoder.embeddings.weight.grad is None


def test_encoder_forward_ignores_extra_collator_fields() -> None:
    torch.manual_seed(0)
    batch = antibody_antigen_batch()
    encoder = TinyEncoder(vocab_size=_GRAMMAR_TOKENIZER.vocab_size, hidden_size=32)
    model = BioSeqEncoderDiffusionModel(tiny_config(), encoder=encoder)

    model_inputs = dict(batch)
    input_ids = model_inputs.pop("input_ids")
    output = model(input_ids=input_ids, **model_inputs)

    assert output.logits.shape == (*batch["input_ids"].shape, tiny_config().vocab_size)


def _naive_gather_token_condition(
    chain_token_condition, chain_ids, position_ids_inner, attention_mask, encoder_residue_mask
):
    """Reference O(B*C) loop implementation (pre-vectorization) for equivalence checks."""
    batch_size, seq_len = chain_ids.shape
    _, max_chains, _, hidden_size = chain_token_condition.shape
    token_condition = chain_token_condition.new_zeros(batch_size, seq_len, hidden_size)
    valid_decoder = chain_ids.ge(0) & chain_ids.lt(max_chains) & position_ids_inner.ge(0)
    if attention_mask is not None:
        valid_decoder = valid_decoder & attention_mask.bool()
    for b in range(batch_size):
        for c in range(max_chains):
            decoder_positions = torch.nonzero(
                valid_decoder[b] & chain_ids[b].eq(c), as_tuple=False
            ).flatten()
            if decoder_positions.numel() == 0:
                continue
            if encoder_residue_mask is None:
                residue_token_positions = torch.arange(chain_token_condition.shape[2])
            else:
                residue_token_positions = torch.nonzero(
                    encoder_residue_mask[b, c], as_tuple=False
                ).flatten()
            residue_indices = position_ids_inner[b, decoder_positions]
            in_bounds = residue_indices.lt(residue_token_positions.numel())
            if not in_bounds.any():
                continue
            dp = decoder_positions[in_bounds]
            ep = residue_token_positions[residue_indices[in_bounds]]
            token_condition[b, dp] = chain_token_condition[b, c, ep]
    return token_condition


def _naive_apply_decoder_corruption_to_encoder(batch, corruption_mask, mask_token_id):
    """Reference per-token loop implementation (pre-vectorization)."""
    encoder_input_ids = batch["encoder_input_ids"]
    encoder_residue_mask = batch["encoder_residue_mask"]
    chain_ids = batch["chain_ids"]
    position_ids_inner = batch["position_ids_inner"]
    noised = encoder_input_ids.clone()
    batch_size, max_chains, _ = encoder_input_ids.shape
    for b in range(batch_size):
        for decoder_pos in torch.nonzero(corruption_mask[b], as_tuple=False).flatten().tolist():
            chain_index = int(chain_ids[b, decoder_pos].item())
            residue_index = int(position_ids_inner[b, decoder_pos].item())
            if chain_index < 0 or chain_index >= max_chains or residue_index < 0:
                continue
            residue_token_positions = torch.nonzero(
                encoder_residue_mask[b, chain_index], as_tuple=False
            ).flatten()
            if residue_index >= residue_token_positions.numel():
                continue
            noised[b, chain_index, residue_token_positions[residue_index]] = int(mask_token_id)
    return noised


def test_encoder_condition_replaces_only_token_emb_and_keeps_position_timestep() -> None:
    """Residue-site injection must replace the token embedding but ADD position /
    chain / timestep on top (not overwrite them)."""
    torch.manual_seed(0)
    batch = antibody_antigen_batch()
    encoder = TinyEncoder(vocab_size=_GRAMMAR_TOKENIZER.vocab_size, hidden_size=32)
    model = BioSeqEncoderDiffusionModel(tiny_config(), encoder=encoder)
    decoder = model.decoder

    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module, args):
        captured["x"] = args[0].detach().clone()

    handle = decoder.layers[0].register_forward_pre_hook(pre_hook)

    torch.manual_seed(1)
    noised_input_ids, _, corruption_mask, timesteps = sample_bioseq_diffusion_noise(
        batch=batch, mask_token_id=model.config.mask_token_id, time_epsilon=model.config.time_epsilon
    )
    noised_encoder_input_ids = apply_decoder_corruption_to_encoder(
        batch, corruption_mask=corruption_mask, mask_token_id=model.config.mask_token_id
    )
    output = model.forward(
        input_ids=noised_input_ids,
        attention_mask=batch["attention_mask"],
        chain_ids=batch["chain_ids"],
        position_ids_inner=batch["position_ids_inner"],
        position_ids_chain=batch["position_ids_chain"],
        timesteps=timesteps,
        residue_mask=batch["residue_mask"],
        encoder_input_ids=noised_encoder_input_ids,
        encoder_attention_mask=batch["encoder_attention_mask"],
        encoder_residue_mask=batch["encoder_residue_mask"],
        encoder_chain_mask=batch["encoder_chain_mask"],
    )
    handle.remove()

    token_condition = output.encoder_condition  # [B,S,H]: encoder feat at residue, 0 elsewhere
    condition_mask = model.build_encoder_condition_mask(
        chain_ids=batch["chain_ids"],
        position_ids_inner=batch["position_ids_inner"],
        attention_mask=batch["attention_mask"],
        encoder_position_ids=None,
        residue_mask=batch["residue_mask"],
    )

    # Recompute the expected injected hidden with the intended order.
    hidden = decoder.token_embeddings(noised_input_ids)
    replace_mask = condition_mask.to(hidden.dtype).unsqueeze(-1)
    hidden = hidden * (1.0 - replace_mask) + token_condition.to(hidden.dtype) * replace_mask
    safe_inner = batch["position_ids_inner"].clamp(min=0, max=model.config.max_position_embeddings - 1)
    inner_valid = batch["position_ids_inner"].ge(0).to(hidden.dtype).unsqueeze(-1)
    hidden = hidden + decoder.inner_position_embeddings(safe_inner) * inner_valid
    safe_chain = batch["position_ids_chain"].clamp(min=0, max=model.config.max_chain_positions - 1)
    chain_valid = batch["position_ids_chain"].ge(0).to(hidden.dtype).unsqueeze(-1)
    hidden = hidden + decoder.chain_position_embeddings(safe_chain) * chain_valid
    hidden = hidden + decoder.timestep_embeddings(timesteps).unsqueeze(1)
    hidden = hidden * batch["attention_mask"].to(hidden.dtype).unsqueeze(-1)

    assert torch.allclose(captured["x"], hidden, atol=1e-5)

    # Discriminator: at residue sites the injected hidden must NOT equal the pure
    # encoder feature (position + timestep survived on top of it).
    residue = batch["attention_mask"] & batch["position_ids_inner"].ge(0)
    assert not torch.allclose(captured["x"][residue], token_condition[residue], atol=1e-4)


def test_llada_backbone_inputs_embeds_path_and_backward() -> None:
    """ESMC-style encoder + LLaDA backbone: residue features replace wte at residue
    sites, grammar special tokens keep wte, masked-CE reaches encoder + LLaDA."""
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    batch = antibody_antigen_batch()
    # d_model must equal encoder hidden (pure replacement); n_heads divides d_model.
    encoder = TinyEncoder(vocab_size=_GRAMMAR_TOKENIZER.vocab_size, hidden_size=32)
    model = BioSeqLLaDAEncoderDiffusionModel(
        tiny_config(hidden_size=32, num_attention_heads=4, num_hidden_layers=2, intermediate_size=64),
        encoder=encoder,
        freeze_encoder=False,
    )

    # Backbone is a LLaDA LM stored under .decoder (so optimizer/DDP plumbing works).
    assert model.decoder.__class__.__name__ == "LLaDAModelLM"

    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module, args, kwargs):
        captured["inputs_embeds"] = kwargs["inputs_embeds"].detach().clone()

    handle = model.decoder.register_forward_pre_hook(pre_hook, with_kwargs=True)
    output = model.compute_loss(batch)
    handle.remove()

    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.logits.shape == (*batch["input_ids"].shape, _GRAMMAR_TOKENIZER.vocab_size)

    # Residue sites replaced by encoder feature; special/structure tokens keep wte.
    condition_mask = model.build_encoder_condition_mask(
        chain_ids=batch["chain_ids"],
        position_ids_inner=batch["position_ids_inner"],
        attention_mask=batch["attention_mask"],
        encoder_position_ids=None,
        residue_mask=batch["residue_mask"],
    )
    injected = captured["inputs_embeds"]
    residue = condition_mask
    special = batch["attention_mask"] & batch["position_ids_inner"].lt(0)
    assert torch.allclose(injected[residue], output.encoder_condition[residue], atol=1e-5)
    wte = model.decoder.get_input_embeddings()(output.noised_input_ids)
    assert torch.allclose(injected[special], wte[special], atol=1e-5)

    output.loss.backward()
    assert encoder.embeddings.weight.grad is not None
    assert torch.isfinite(encoder.embeddings.weight.grad).all()
    # LLaDA backbone received gradient too.
    wte_grad = model.decoder.get_input_embeddings().weight.grad
    assert wte_grad is not None and torch.isfinite(wte_grad).all()


def _tiny_llada2_config(**overrides) -> BioSeqDiffusionTransformerConfig:
    """Tiny but architecture-faithful LLaDA2-MoE config for unit tests.

    Keeps the MoE structure (routed + shared experts, group-limited routing, GQA,
    partial RoPE) but shrinks every dimension so a full forward/backward is cheap.
    head_dim=8 with partial_rotary_factor=0.5 => rope_dim=4 (even, required).
    n_group=2 divides num_experts=4; topk_group=2; experts_per_tok=2.
    """
    values = dict(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        moe_num_experts=4,
        moe_num_experts_per_tok=2,
        moe_num_shared_experts=1,
        moe_intermediate_size=16,
        moe_n_group=2,
        moe_topk_group=2,
        moe_first_k_dense_replace=1,
        moe_num_key_value_heads=2,
        moe_head_dim=8,
        moe_partial_rotary_factor=0.5,
    )
    values.update(overrides)
    return tiny_config(**values)


def test_llada2_backbone_inputs_embeds_path_and_backward() -> None:
    """ESMC-style encoder + LLaDA2-MoE backbone (from scratch): residue features
    replace word_embeddings at residue sites, grammar special tokens keep the
    embedding, and masked-CE reaches both the encoder and the LLaDA2 backbone."""
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    batch = antibody_antigen_batch()
    # hidden_size must equal encoder hidden (pure replacement).
    encoder = TinyEncoder(vocab_size=_GRAMMAR_TOKENIZER.vocab_size, hidden_size=32)
    model = BioSeqLLaDA2EncoderDiffusionModel(
        _tiny_llada2_config(),
        encoder=encoder,
        freeze_encoder=False,
    )

    # Backbone is a LLaDA2-MoE LM stored under .decoder (optimizer/DDP plumbing).
    assert model.decoder.__class__.__name__ == "LLaDA2MoeModelLM"
    # It really is an MoE: sparse blocks after first_k_dense_replace dense layers.
    moe_blocks = [
        m for m in model.decoder.modules() if m.__class__.__name__ == "LLaDA2MoeSparseMoeBlock"
    ]
    assert len(moe_blocks) >= 1

    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module, args, kwargs):
        captured["inputs_embeds"] = kwargs["inputs_embeds"].detach().clone()
        # A 4D bidirectional padding mask must be passed (bypasses causal mask).
        mask = kwargs.get("attention_mask")
        captured["mask_ndim"] = torch.tensor(mask.dim() if mask is not None else 0)

    handle = model.decoder.register_forward_pre_hook(pre_hook, with_kwargs=True)
    output = model.compute_loss(batch)
    handle.remove()

    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.logits.shape == (*batch["input_ids"].shape, _GRAMMAR_TOKENIZER.vocab_size)
    assert int(captured["mask_ndim"].item()) == 4  # bidirectional 4D mask

    # Residue sites replaced by encoder feature; special/structure tokens keep wte.
    condition_mask = model.build_encoder_condition_mask(
        chain_ids=batch["chain_ids"],
        position_ids_inner=batch["position_ids_inner"],
        attention_mask=batch["attention_mask"],
        encoder_position_ids=None,
        residue_mask=batch["residue_mask"],
    )
    injected = captured["inputs_embeds"]
    residue = condition_mask
    special = batch["attention_mask"] & batch["position_ids_inner"].lt(0)
    assert torch.allclose(injected[residue], output.encoder_condition[residue], atol=1e-5)
    wte = model.decoder.get_input_embeddings()(output.noised_input_ids)
    assert torch.allclose(injected[special], wte[special], atol=1e-5)

    output.loss.backward()
    assert encoder.embeddings.weight.grad is not None
    assert torch.isfinite(encoder.embeddings.weight.grad).all()
    # LLaDA2 backbone received gradient too (embedding + at least one expert).
    wte_grad = model.decoder.get_input_embeddings().weight.grad
    assert wte_grad is not None and torch.isfinite(wte_grad).all()
    expert_grads = [
        p.grad
        for block in moe_blocks
        for p in block.experts.parameters()
        if p.grad is not None
    ]
    assert expert_grads, "at least one routed expert must receive gradient"
    assert all(torch.isfinite(g).all() for g in expert_grads)


def test_vectorized_gather_and_corruption_match_naive_reference() -> None:
    """Vectorized gather/corruption-mirror must be bit-identical to the loop versions."""
    torch.manual_seed(0)
    batch = antibody_antigen_batch()
    encoder = TinyEncoder(vocab_size=_GRAMMAR_TOKENIZER.vocab_size, hidden_size=32)
    model = BioSeqEncoderDiffusionModel(tiny_config(), encoder=encoder)

    chain_condition = model.encode_chain_tokens(
        encoder_input_ids=batch["encoder_input_ids"],
        encoder_attention_mask=batch["encoder_attention_mask"],
        encoder_residue_mask=batch["encoder_residue_mask"],
        encoder_chain_mask=batch["encoder_chain_mask"],
    )

    # gather with residue mask (per-chain <cls>seq<eos> offset path)
    got = model.gather_token_condition(
        chain_condition,
        chain_ids=batch["chain_ids"],
        position_ids_inner=batch["position_ids_inner"],
        attention_mask=batch["attention_mask"],
        encoder_residue_mask=batch["encoder_residue_mask"],
    )
    expected = _naive_gather_token_condition(
        chain_condition,
        batch["chain_ids"],
        batch["position_ids_inner"],
        batch["attention_mask"],
        batch["encoder_residue_mask"],
    )
    assert torch.equal(got, expected)

    # gather without residue mask (identity offset path)
    got_no_mask = model.gather_token_condition(
        chain_condition,
        chain_ids=batch["chain_ids"],
        position_ids_inner=batch["position_ids_inner"],
        attention_mask=batch["attention_mask"],
        encoder_residue_mask=None,
    )
    expected_no_mask = _naive_gather_token_condition(
        chain_condition,
        batch["chain_ids"],
        batch["position_ids_inner"],
        batch["attention_mask"],
        None,
    )
    assert torch.equal(got_no_mask, expected_no_mask)

    # corruption mirror over several random noise draws
    for seed in range(5):
        torch.manual_seed(seed)
        _, _, corruption_mask, _ = sample_bioseq_diffusion_noise(
            batch=batch,
            mask_token_id=model.config.mask_token_id,
            time_epsilon=model.config.time_epsilon,
        )
        got_enc = apply_decoder_corruption_to_encoder(
            batch, corruption_mask=corruption_mask, mask_token_id=model.config.mask_token_id
        )
        expected_enc = _naive_apply_decoder_corruption_to_encoder(
            batch, corruption_mask, model.config.mask_token_id
        )
        assert torch.equal(got_enc, expected_enc)


def test_denoiser_accepts_soft_diffusion_state_without_internal_one_hot() -> None:
    torch.manual_seed(0)
    batch = antibody_antigen_batch()
    model = BioSeqNoEncoderDiffusionModel(tiny_config()).eval()
    soft_state = torch.zeros(*batch["input_ids"].shape, model.config.vocab_size)
    soft_state.scatter_(2, batch["input_ids"].unsqueeze(-1), 1.0)

    with torch.no_grad():
        from_ids = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids_inner=batch["position_ids_inner"],
            position_ids_chain=batch["position_ids_chain"],
        )
        from_soft_state = model(
            diffusion_state=soft_state,
            attention_mask=batch["attention_mask"],
            position_ids_inner=batch["position_ids_inner"],
            position_ids_chain=batch["position_ids_chain"],
        )

    assert torch.allclose(from_soft_state.logits, from_ids.logits, atol=1e-5)


def test_noise_sampler_masks_only_diffusion_loss_positions() -> None:
    torch.manual_seed(0)
    batch = antibody_antigen_batch()
    noised_input_ids, labels, corruption_mask, timesteps = sample_bioseq_diffusion_noise(
        batch,
        mask_token_id=32,
        time_epsilon=0.75,
    )

    assert noised_input_ids.shape == batch["input_ids"].shape
    assert labels.shape == batch["input_ids"].shape
    assert timesteps.shape == (batch["input_ids"].shape[0],)
    assert (corruption_mask & ~batch["diffusion_loss_mask"]).sum().item() == 0
    assert (labels.ne(-100) == corruption_mask).all()


def test_compute_masked_cross_entropy_forbids_mask_token_predictions() -> None:
    config = tiny_config(vocab_size=64, mask_token_id=32, pad_token_id=1)
    forbidden = forbidden_diffusion_target_token_ids(config)
    assert 32 in forbidden

    logits = torch.zeros(1, 2, config.vocab_size)
    logits[..., 32] = 100.0
    labels = torch.tensor([[4, 5]])

    masked_logits = mask_forbidden_target_logits(logits, forbidden)
    assert masked_logits.argmax(dim=-1).eq(32).sum().item() == 0

    loss = compute_masked_cross_entropy(logits, labels, forbidden_token_ids=forbidden)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_biohub_esmc_state_dict_key_conversion() -> None:
    raw = {
        "esmc.embed.weight": torch.ones(2, 3),
        "esmc.transformer.blocks.0.attn.layernorm_qkv.layer_norm_weight": torch.ones(3),
        "esmc.transformer.blocks.0.attn.layernorm_qkv.layer_norm_bias": torch.ones(3),
        "esmc.transformer.blocks.0.attn.layernorm_qkv.weight": torch.ones(9, 3),
        "esmc.transformer.blocks.0.ffn.layer_norm_weight": torch.ones(3),
        "esmc.transformer.blocks.0.ffn.layer_norm_bias": torch.ones(3),
        "esmc.transformer.blocks.0.ffn.fc1_weight": torch.ones(12, 3),
        "esmc.transformer.blocks.0.ffn.fc2_weight": torch.ones(3, 6),
        "esmc.transformer.blocks.0.ffn._extra_state": torch.empty(0),
        "lm_head.3.weight": torch.ones(64, 3),
    }

    converted = _convert_biohub_esmc_state_dict(raw)

    assert "embed.weight" in converted
    assert "transformer.blocks.0.attn.layernorm_qkv.0.weight" in converted
    assert "transformer.blocks.0.attn.layernorm_qkv.0.bias" in converted
    assert "transformer.blocks.0.attn.layernorm_qkv.1.weight" in converted
    assert "transformer.blocks.0.ffn.0.weight" in converted
    assert "transformer.blocks.0.ffn.0.bias" in converted
    assert "transformer.blocks.0.ffn.1.weight" in converted
    assert "transformer.blocks.0.ffn.3.weight" in converted
    assert "sequence_head.3.weight" in converted
    assert not any(key.endswith("._extra_state") for key in converted)


@pytest.mark.skipif(
    not (ESM2_650M_DIR / "config.json").is_file(),
    reason="Local ESM2-650M weights not available",
)
def test_local_esm2_encoder_disables_token_dropout_and_stays_finite() -> None:
    """ESM2 token_dropout rescales by 1/(1-mask_ratio); a fully-masked diffusion
    chain drives mask_ratio->1 -> NaN. The loader must disable token_dropout so a
    chain whose attended positions are all ``mask_token_id`` stays finite."""

    encoder = load_local_esm2_encoder(ESM2_650M_DIR).eval()
    assert encoder.config.token_dropout is False

    mask_id = encoder.config.mask_token_id
    fully_masked = torch.full((1, 16), mask_id, dtype=torch.long)
    attention_mask = torch.ones_like(fully_masked)
    with torch.no_grad():
        output = encoder(fully_masked, attention_mask=attention_mask)
    assert torch.isfinite(output.last_hidden_state).all()
