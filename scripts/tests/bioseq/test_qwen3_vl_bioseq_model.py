from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from dllm.pipelines.qwen3_vl_arch.data import (
    BioSeqChain,
    BioSeqQwenDataCollator,
    BioSeqRecord,
    BioSeqViewSampler,
    Esm2SequenceTokenizer,
)
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (
    BioSeqDiffusionTransformerConfig,
    BioSeqEncoderDiffusionModel,
    BioSeqNoEncoderDiffusionModel,
    _convert_biohub_esmc_state_dict,
    apply_decoder_corruption_to_encoder,
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


def tiny_config(**overrides) -> BioSeqDiffusionTransformerConfig:
    values = {
        "vocab_size": 33,
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
        ),
        BioSeqRecord(
            chains=[
                BioSeqChain("EVQLVESGGG", "antibody_heavy"),
                BioSeqChain("EIVLTQSPAT", "antibody_light"),
                BioSeqChain("GILGFVFTLTVPSER", "antigen"),
            ],
            task_type="antibody_antigen",
            source="unit",
        ),
    ]
    collator = BioSeqQwenDataCollator(
        tokenizer=Esm2SequenceTokenizer(),
        view_sampler=BioSeqViewSampler(allowed_views=["full_denoise"], seed=0),
        require_homogeneous_task=True,
    )
    return collator(records)


def test_no_encoder_model_compute_loss_runs_and_respects_fixed_context() -> None:
    torch.manual_seed(0)
    batch = antibody_antigen_batch()
    model = BioSeqNoEncoderDiffusionModel(tiny_config())

    output = model.compute_loss(batch)

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert output.logits.shape == (*batch["input_ids"].shape, 33)
    assert output.corruption_mask is not None
    assert (output.corruption_mask & ~batch["diffusion_loss_mask"]).sum().item() == 0
    assert (output.corruption_mask & batch["fixed_context_mask"]).sum().item() == 0
    assert output.corruption_mask.sum().item() >= batch["input_ids"].shape[0]


def test_encoder_model_compute_loss_masks_encoder_targets_without_touching_antigen() -> None:
    torch.manual_seed(0)
    batch = antibody_antigen_batch()
    encoder = TinyEncoder(vocab_size=33, hidden_size=16)
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
    assert torch.equal(output.noised_encoder_input_ids[:, 2], batch["encoder_input_ids"][:, 2])

    output.loss.backward()
    assert encoder.embeddings.weight.grad is not None
    assert torch.isfinite(encoder.embeddings.weight.grad).all()


def test_encoder_model_can_freeze_encoder() -> None:
    torch.manual_seed(0)
    batch = antibody_antigen_batch()
    encoder = TinyEncoder(vocab_size=33, hidden_size=16)
    model = BioSeqEncoderDiffusionModel(tiny_config(), encoder=encoder, freeze_encoder=True)

    output = model.compute_loss(batch)
    assert output.loss is not None
    output.loss.backward()

    assert encoder.embeddings.weight.requires_grad is False
    assert encoder.embeddings.weight.grad is None


def test_encoder_forward_ignores_extra_collator_fields() -> None:
    torch.manual_seed(0)
    batch = antibody_antigen_batch()
    encoder = TinyEncoder(vocab_size=33, hidden_size=16)
    model = BioSeqEncoderDiffusionModel(tiny_config(), encoder=encoder)

    model_inputs = dict(batch)
    input_ids = model_inputs.pop("input_ids")
    output = model(input_ids=input_ids, **model_inputs)

    assert output.logits.shape == (*batch["input_ids"].shape, 33)


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
