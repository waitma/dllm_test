from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch
import torch.nn as nn

from dllm.pipelines.qwen3_vl_arch.data import (
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
    BioSeqNoEncoderDiffusionModel,
    _convert_biohub_esmc_state_dict,
    apply_decoder_corruption_to_encoder,
    compute_masked_cross_entropy,
    forbidden_diffusion_target_token_ids,
    mask_forbidden_target_logits,
    sample_bioseq_diffusion_noise,
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
