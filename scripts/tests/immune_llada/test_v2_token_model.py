"""Decoder-only v2 CPU regressions.

Run: python -m pytest scripts/tests/immune_llada/test_v2_token_model.py -q
"""

from pathlib import Path
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml
from torch import nn

from dllm.pipelines.immune_llada.data import GrammarBioSeqCollator, GrammarTokenizer
from dllm.pipelines.qwen3_vl_arch import modeling_bioseq
from downstream.grammar.common import antibody_pair_record, run_grammar_generate, tcr_pair_record
from downstream.grammar.masks import light_chain_generation_partial_mask
from examples.llada import protein_fusion_model as fusion
from examples.llada import protein_pretrain_esmc as pretrain
from examples.llada.load_fusion_checkpoint import _read_fusion_policy


class TinyDecoder(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.embeddings = nn.Embedding(128, 12)
        self.head = nn.Linear(12, 128)
        self.config = SimpleNamespace(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            vocab_size=128,
        )
        self.last_embeds = None

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, inputs_embeds, **kwargs):
        self.last_embeds = inputs_embeds.detach().clone()
        return SimpleNamespace(logits=self.head(inputs_embeds))


def make_model(tokenizer, **kwargs):
    return fusion.LLaDAEsmcFusion(
        decoder=TinyDecoder(tokenizer), encoder=None, encoder_hidden_size=0,
        decoder_mask_token_id=tokenizer.mask_token_id,
        encoder_mask_token_id=tokenizer.mask_token_id,
        residue_cond_mode="token", fixed_receptor_lengths=True, predict_eos=True,
        decoder_chain_eos_token_id=tokenizer.eos_token_id,
        residue_token_ids=tokenizer.encode_residues("".join(fusion.RESIDUES)),
        **kwargs,
    )


def test_training_encoder_loader_token_needs_no_assets(tmp_path, monkeypatch):
    load = Mock(side_effect=AssertionError("ESMC must not be loaded"))
    monkeypatch.setattr(modeling_bioseq, "load_local_esmc_encoder", load)
    assert pretrain._load_training_encoder(tmp_path / "missing", "token") == (None, 0)
    load.assert_not_called()


def test_training_encoder_loader_add_is_unchanged(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"d_model": 6}')
    encoder = nn.Linear(6, 6)
    load = Mock(return_value=encoder)
    monkeypatch.setattr(modeling_bioseq, "load_local_esmc_encoder", load)
    assert pretrain._load_training_encoder(tmp_path, "add") == (encoder, 6)
    load.assert_called_once_with(tmp_path)


@pytest.mark.parametrize("record", [antibody_pair_record("ACDE", "FGH"), tcr_pair_record("CAV", "CASS")])
def test_token_v2_loss_backward_and_no_encoder_dependency(record, monkeypatch):
    tokenizer = GrammarTokenizer()
    batch = GrammarBioSeqCollator(tokenizer, fixed_receptor_lengths=True)([record])
    model = make_model(tokenizer)
    assert model.encoder is None
    assert model.condition_norm is None
    assert model.condition_proj is None
    assert all(key.startswith("decoder.") for key in model.state_dict())
    assert model.config.condition_hidden_size == 0
    assert model.config.condition_norm is False
    monkeypatch.setattr(
        fusion, "apply_decoder_corruption_to_encoder",
        Mock(side_effect=AssertionError("encoder mirror must not run during token training")),
    )
    torch.manual_seed(17)
    output = model(**batch)
    assert output.noised_encoder_input_ids is None
    assert torch.isfinite(output.loss)
    assert output.labels[batch["chain_eos_mask"]].ne(-100).any()
    torch.testing.assert_close(
        model.decoder.last_embeds,
        model.decoder.embeddings(output.noised_input_ids),
    )
    output.loss.backward()
    assert model.decoder.embeddings.weight.grad is not None
    eos_grad = model.decoder.head.weight.grad[tokenizer.eos_token_id]
    assert torch.isfinite(eos_grad).all() and eos_grad.abs().sum() > 0

    # No hidden reference can affect token training via the unused encoder stream.
    changed = {**batch, "encoder_input_ids": torch.full_like(batch["encoder_input_ids"], 9999)}
    torch.manual_seed(17)
    changed_output = model(**changed)
    torch.testing.assert_close(output.loss, changed_output.loss)
    torch.testing.assert_close(output.logits, changed_output.logits)


@pytest.mark.parametrize("cfg_scale", [0.0, 1.0])
def test_token_v2_pairing_sampler_preserves_conditions_and_generates_eos(cfg_scale):
    tokenizer = GrammarTokenizer()
    batch = GrammarBioSeqCollator(tokenizer, fixed_receptor_lengths=True)(
        [antibody_pair_record("ACDE", "FGH")]
    )
    model = make_model(tokenizer).eval()
    with torch.no_grad():
        model.decoder.head.weight.zero_()
        model.decoder.head.bias.zero_()
        model.decoder.head.bias[tokenizer.eos_token_id] = 20
    partial = light_chain_generation_partial_mask(batch, tokenizer)
    generated, _ = run_grammar_generate(
        model, batch, partial_mask=partial, max_iter=2,
        sampling_strategy="argmax", temperature=1.0, cfg_scale=cfg_scale,
    )
    generate = ~partial & batch["chain_slot_mask"]
    assert int(generate.sum()) == 135
    assert generated[generate].eq(tokenizer.eos_token_id).all()
    torch.testing.assert_close(generated[partial], batch["input_ids"][partial])
    assert model.encoder is None


def test_token_v2_metadata_roundtrip_and_resume(tmp_path):
    model = make_model(GrammarTokenizer())
    fusion.save_fusion_config(tmp_path, model)
    config = fusion.load_fusion_config(tmp_path)
    policy = _read_fusion_policy(config, tmp_path)
    assert policy["residue_cond_mode"] == "token"
    assert policy["condition_norm"] is False
    assert policy["predict_eos"] is True
    eos_kwargs = dict(
        decoder_chain_eos_token_id=model.config.decoder_chain_eos_token_id,
        decoder_chain_eos_policy=model.config.decoder_chain_eos_policy,
    )
    pretrain._validate_resume_policy(
        str(tmp_path), str(tmp_path), True, residue_cond_mode="token", **eos_kwargs,
    )
    with pytest.raises(ValueError, match="residue_cond_mode mismatch"):
        pretrain._validate_resume_policy(
            str(tmp_path), str(tmp_path), True, residue_cond_mode="add", **eos_kwargs,
        )
    model.config.condition_hidden_size = 6
    fusion.save_fusion_config(tmp_path, model)
    with pytest.raises(ValueError, match="encoder-bearing legacy token"):
        pretrain._validate_resume_policy(
            str(tmp_path), str(tmp_path), True, residue_cond_mode="token", **eos_kwargs,
        )


@pytest.mark.parametrize("mode", ["add", "feature"])
def test_missing_encoder_rejected_for_feature_modes(mode):
    tokenizer = GrammarTokenizer()
    with pytest.raises(ValueError, match="encoder=None requires"):
        fusion.LLaDAEsmcFusion(
            TinyDecoder(tokenizer), None, 0, tokenizer.mask_token_id,
            tokenizer.mask_token_id, residue_cond_mode=mode,
        )


def test_real_llada_cpu_encoder_free_eos_backward():
    from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
    from dllm.pipelines.llada.models.modeling_llada import (
        LLaDAModel, LLaDAModelLM, create_model_config_from_pretrained_config,
    )

    tokenizer = GrammarTokenizer()
    cfg = LLaDAConfig(
        d_model=32, n_layers=2, n_heads=4, n_kv_heads=4,
        mlp_hidden_size=128, vocab_size=128, embedding_size=128,
        max_sequence_length=512, block_type="llama", activation_type="silu",
        rope=True, weight_tying=False, init_device="cpu",
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    # Supply the real inner model to avoid the LM wrapper's hard-coded CUDA init.
    inner = LLaDAModel(create_model_config_from_pretrained_config(cfg), init_params=True)
    decoder = LLaDAModelLM(cfg, model=inner, init_params=False)
    model = fusion.LLaDAEsmcFusion(
        decoder, None, 0, tokenizer.mask_token_id, tokenizer.mask_token_id,
        residue_cond_mode="token", fixed_receptor_lengths=True, predict_eos=True,
        decoder_chain_eos_token_id=tokenizer.eos_token_id,
        residue_token_ids=tokenizer.encode_residues("".join(fusion.RESIDUES)),
    )
    assert isinstance(model.decoder, LLaDAModelLM)
    assert all(key.startswith("decoder.") for key in model.state_dict())
    assert model.encoder is model.condition_proj is model.condition_norm is None
    batch = GrammarBioSeqCollator(tokenizer, fixed_receptor_lengths=True)(
        [antibody_pair_record("ACDE", "FGH")]
    )
    torch.manual_seed(17)
    output = model(**batch)
    eos_targets = batch["chain_eos_mask"] & output.labels.ne(-100)
    assert eos_targets.any()
    assert output.noised_input_ids[eos_targets].eq(tokenizer.mask_token_id).all()
    assert output.labels[eos_targets].eq(tokenizer.eos_token_id).all()
    assert torch.isfinite(output.loss)
    # Isolate EOS supervision: its loss alone must update the real LLaDA backbone.
    eos_loss = torch.nn.functional.cross_entropy(output.logits[eos_targets], output.labels[eos_targets])
    eos_loss.backward()
    grads = [p.grad for p in model.decoder.model.transformer.blocks.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("job,mode,steps", [
    ("protein_esmc_llada270m_diffusion_v2_immune_v6_2node4gpu.yml", "add", "200000"),
    ("protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m.yml", "token", "1000000"),
])
def test_local_job_launch_keeps_all_arguments(job, mode, steps):
    path = Path(__file__).resolve().parents[3] / "train_jobs" / job
    if not path.exists():
        pytest.skip("Local launch YAMLs are excluded from version control")
    entry = yaml.safe_load(path.read_text())["Entrypoint"]
    launch = "accelerate launch" + entry.split("accelerate launch", 1)[1]
    # Execute only the launch stanza with a shell function instead of Accelerate.
    # bash -n alone misses a blank line that terminates a continued command.
    capture = 'set -eu\nRESUME_ARGS=()\naccelerate() { printf "%s\\0" "$@"; }\n'
    env = {**os.environ, "MLP_WORKER_NUM": "2", "MLP_WORKER_GPU": "4",
           "MLP_ROLE_INDEX": "1", "MLP_WORKER_0_HOST": "master.test",
           "MLP_WORKER_0_PORT": "29500", "ESMC_DIR": "/test/esmc",
           "TOKENIZER_DIR": "/test/tokenizer", "PREPARED_DATA_DIR": "/test/data",
           "OUTPUT_DIR": "/test/output"}
    result = subprocess.run(["bash", "-c", capture + launch], env=env, capture_output=True, check=True)
    argv = result.stdout.decode().split("\0")[:-1]
    for flag, value in {
        "--num_machines": "2", "--num_processes": "8", "--machine_rank": "1",
        "--fixed_receptor_lengths": "True", "--residue_cond_mode": mode,
        "--max_steps": steps, "--save_top_k": "3", "--output_dir": "/test/output",
    }.items():
        assert argv[argv.index(flag) + 1] == value
    assert argv[-2:] == ["--output_dir", "/test/output"]


def test_feature_replacement_remains_forbidden_in_v2():
    tokenizer = GrammarTokenizer()
    with pytest.raises(ValueError, match="add/token mode"):
        fusion.LLaDAEsmcFusion(
            TinyDecoder(tokenizer), nn.Linear(12, 12), 12,
            tokenizer.mask_token_id, tokenizer.mask_token_id,
            residue_cond_mode="feature", fixed_receptor_lengths=True, predict_eos=True,
        )
