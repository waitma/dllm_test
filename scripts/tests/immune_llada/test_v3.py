"""Run: python -m pytest scripts/tests/immune_llada/test_v3.py -q."""
import torch
import pytest

from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator, observe
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig, ImmuneModel


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def record(seq="CASSF", light="CAVW"):
    return BioSeqRecord([BioSeqChain(seq, "tcr_beta"), BioSeqChain(light, "tcr_alpha")], "tcr", "test")


def tiny(ablation="A4"):
    return ImmuneModel(ImmuneConfig(ablation=ablation, d_model=24, n_layers=2, n_heads=3,
                                   mlp_hidden_size=48, max_length=512, full_capacity=16, fragment_capacity=12))


def collator(ablation="A4", **kwargs):
    return ImmuneCollator(ablation, full_capacity=16, fragment_capacity=12, max_length=512, **kwargs)


def test_framework_anchors_survive_observed_projection():
    chain = BioSeqChain("XXXCASSWXX", "tcr_beta", metadata={
        "synthetic_residue_mask": [1,1,1,0,0,0,0,0,1,1],
        "observed_regions": ["CDR3"], "observed_anchor_regions": {"FR3": "C", "FR4": "W"}})
    obs = observe(chain)
    assert obs.sequence == "CASSW" and obs.scope == "junction"
    assert obs.original_indices == (3,4,5,6,7)


def test_core_letters_are_not_anchor_heuristics():
    chain = BioSeqChain("XXCASFXX", "tcr_beta", metadata={"synthetic_residue_mask": [1,1,0,0,0,0,1,1], "observed_regions": ["CDR3"]})
    assert observe(chain).sequence == "CASF"
    assert observe(chain).scope == "core"
    missing = BioSeqChain("XXX", "tcr_alpha", metadata={"synthetic_residue_mask": [1,1,1]})
    assert observe(missing).scope == "missing"


def test_discontinuous_real_fragments_fail_instead_of_joining():
    chain = BioSeqChain("ACXDF", "tcr_beta", metadata={"synthetic_residue_mask": [0,0,1,0,0]})
    with pytest.raises(ValueError, match="Discontinuous"):
        observe(chain)


def test_generation_inputs_are_reference_length_independent():
    c = collator(mode="generation")
    a, b = c([record("CAF")]), c([record("CASSGGGGSF")])
    for key in ImmuneModel.feature_args(a):
        assert torch.equal(a[key], b[key]), key
    assert a["labels"].eq(-100).all()


def test_generation_request_needs_no_reference_or_dummy_residues():
    from dllm.pipelines.immune_llada.v3.layout import GenerationRequest
    c=collator(mode="generation")
    request=GenerationRequest(conditions=(),targets=(("tcr_beta","full"),("tcr_alpha","full")))
    batch=c([request]); reference=c([record()])
    for key in ImmuneModel.feature_args(batch):
        assert torch.equal(batch[key],reference[key]),key
    assert not batch["residue_mask"].any()
    with pytest.raises(ValueError,match="generation mode"):
        collator()([request])


def test_single_terminal_and_attended_ignored_tail():
    c = collator()
    batch = c([record()])
    assert batch["terminal_mask"].sum() == 2
    assert (batch["labels"] == c.tokenizer.special_id("<protd>")).sum() == 2
    tails = batch["target_slot_mask"] & ~batch["eligible_mask"]
    assert tails.any() and batch["attention_mask"][tails].all()
    assert batch["input_ids"][tails].eq(c.tokenizer.mask_token_id).all()
    assert batch["labels"][tails].eq(-100).all()


@pytest.mark.parametrize("ablation", ["A1", "A2", "A3", "A4"])
def test_each_ablation_backward(ablation):
    c, model = collator(ablation), tiny(ablation)
    batch = c([record()])
    out = model(**batch, token_t=torch.ones_like(batch["input_ids"], dtype=torch.float))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.backbone.transformer.wte.weight.grad.abs().sum() > 0
    if model.role_embeddings is not None:
        assert model.role_embeddings.weight.grad.abs().sum() > 0
    if model.chain_bias is not None:
        assert model.chain_bias.grad.abs().sum() > 0


def test_representation_batch_padding_invariance():
    model = tiny().eval()
    c = collator(mode="representation")
    alone = model.represent(c([record("CAF", "CAW")]))["pooled"][0]
    padded = model.represent(c([record("CAF", "CAW"), record("CASSGGGGF", "CAVGGGGW")]))["pooled"][0]
    torch.testing.assert_close(alone, padded, atol=1e-6, rtol=1e-5)


def test_reference_label_masks_cannot_change_denoising():
    model = tiny().eval()
    batch = collator(mode="generation")([record()])
    a, _ = model.denoise(**model.feature_args(batch))
    batch["labels"].fill_(123456)
    batch["residue_mask"] = ~batch["residue_mask"]
    batch["terminal_mask"] = ~batch["terminal_mask"]
    b, _ = model.denoise(**model.feature_args(batch))
    torch.testing.assert_close(a, b)


def test_sampler_revises_end_and_preserves_context(monkeypatch):
    model = tiny().eval()
    rec = BioSeqRecord([BioSeqChain("ACDE", "peptide"), BioSeqChain("CASSF", "tcr_beta")], "tcr_epitope", "test")
    batch = collator(mode="generation")([rec])
    calls = []
    aa = model.tokenizer.encode_residues("A")[0]
    def fake(**features):
        calls.append(features["input_ids"].clone())
        logits = torch.full((*features["input_ids"].shape, model.tokenizer.vocab_size), -50.)
        logits[..., aa] = 50.
        pos = features["target_slot_mask"][0].nonzero().flatten()
        end = pos[1 if len(calls)==1 else 5]
        logits[0,end,aa] = -50.
        logits[0,end,model.terminal_id] = 50.
        return logits, None
    monkeypatch.setattr(model, "denoise", fake)
    out = model.generate(batch, steps=2)
    assert out["chains"][0][0]["sequence"] == "AAAAA"
    assert out["chains"][0][0]["status"] == "terminated"
    fixed = ~batch["target_slot_mask"]
    assert torch.equal(out["input_ids"][fixed], batch["input_ids"][fixed])


def test_save_load_exact_and_legacy_rejected(tmp_path):
    model = tiny().eval()
    model.save_pretrained(tmp_path)
    restored = ImmuneModel.from_pretrained(tmp_path).eval()
    batch = collator(mode="representation")([record()])
    torch.testing.assert_close(model.represent(batch)["pooled"], restored.represent(batch)["pooled"])
    with pytest.raises(ValueError):
        ImmuneConfig(schema_version="v2")


def test_overflow_is_explicit():
    with pytest.raises(ValueError, match="capacity_overflow"):
        collator()([record("A"*16)])


def test_optional_global_rope_matches_legacy():
    model = tiny("A1").backbone.eval()
    ids = torch.tensor([[4,5,6,7]])
    a = model(ids).logits
    b = model(ids, position_ids=torch.arange(4).unsqueeze(0)).logits
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_activation_checkpointing_threads_positions_and_gradients():
    from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
    model = tiny()
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    model(**collator()([record()]))["loss"].backward()
    assert model.chain_bias.grad.abs().sum() > 0


def test_training_entrypoint_cpu_save_resume_protocol(tmp_path):
    from transformers import TrainingArguments
    from examples.llada.immune_pretrain_v3 import ImmuneTrainer
    model = tiny()
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, max_steps=2, per_device_train_batch_size=2,
                             remove_unused_columns=False, label_names=["labels"], report_to="none",
                             save_steps=1, logging_steps=1, data_seed=42, disable_tqdm=True)
    trainer = ImmuneTrainer(model=model, args=args, train_dataset=[record(),record("CATF")], data_collator=collator())
    trainer.train()
    restored = ImmuneModel.from_pretrained(tmp_path/"checkpoint-2")
    torch.testing.assert_close(model.backbone.transformer.wte.weight, restored.backbone.transformer.wte.weight)
    resumed = ImmuneTrainer(model=tiny(), args=args, train_dataset=[record(),record("CATF")], data_collator=collator())
    resumed.train(resume_from_checkpoint=str(tmp_path/"checkpoint-1"))
    assert resumed.state.global_step == 2
    torch.testing.assert_close(model.backbone.transformer.wte.weight, resumed.model.backbone.transformer.wte.weight)


@pytest.mark.parametrize("ablation", ["A1", "A2", "A3", "A4"])
def test_chain_completion_preserves_condition(ablation):
    from dllm.pipelines.immune_llada.data.records import CHAIN_ROLE_TO_ID
    c = collator(ablation, mode="generation", target_roles=["tcr_alpha"])
    batch = c([record()])
    condition = batch["role_ids"] == CHAIN_ROLE_TO_ID["tcr_beta"]
    assert not batch["target_slot_mask"][condition].any()
    assert not batch["input_ids"][condition].eq(c.tokenizer.mask_token_id).any()
    assert batch["target_slot_mask"].any()


def test_resume_rejects_changed_data_and_schedule(tmp_path):
    import json
    from examples.llada.immune_pretrain_v3 import validate_resume
    provenance = {"data_manifest_sha256":"old", "scheduler_horizon":10000}
    (tmp_path/"run_provenance.json").write_text(json.dumps(provenance))
    validate_resume(tmp_path,provenance)
    with pytest.raises(ValueError,match="provenance"):
        validate_resume(tmp_path,{**provenance,"data_manifest_sha256":"new"})
    with pytest.raises(ValueError,match="provenance"):
        validate_resume(tmp_path,{**provenance,"scheduler_horizon":50000})


def test_training_rejects_incomplete_audit(tmp_path):
    import json
    from examples.llada.immune_pretrain_v3 import assert_ready
    (tmp_path/"dataset_manifest.json").write_text(json.dumps({"splits":{}}))
    (tmp_path/"audit_certificate.json").write_text(json.dumps({"status":"pending"}))
    with pytest.raises(ValueError, match="incomplete"):
        assert_ready(tmp_path)


def test_a0_legacy_ids_and_full_softmax():
    c=collator("A0")
    batch=c([record()])
    model=ImmuneModel(ImmuneConfig(ablation="A0",d_model=12,n_layers=1,n_heads=3,
                                 mlp_hidden_size=24,max_length=512)).eval()
    assert model.backbone.transformer.wte.weight.shape==(126464,12)
    assert c.tokenizer.mask_token_id==126336 and c.tokenizer.eos_token_id==126389
    assert c.tokenizer.encode_residues("AC")==[126347,126365]
    with torch.no_grad():
        logits,_=model.denoise(**model.feature_args(batch))
    assert torch.isfinite(logits).all()  # Unused text rows participate in A0 CE.
    out=model(**batch,token_t=torch.ones_like(batch["input_ids"],dtype=torch.float))
    out["loss"].backward()
    assert torch.isfinite(out["loss"])


def test_validation_reports_equal_source_macro_and_components(tmp_path):
    from transformers import TrainingArguments
    from examples.llada.immune_pretrain_v3 import ImmuneTrainer
    args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,per_device_eval_batch_size=2,
                           remove_unused_columns=False,label_names=["labels"],report_to="none",disable_tqdm=True,
                           prediction_loss_only=True)
    trainer=ImmuneTrainer(model=tiny(),args=args,eval_dataset={"one":[record()],"three":[record()]*3},data_collator=collator())
    metrics=trainer.evaluate()
    assert metrics["eval_loss"]==pytest.approx((metrics["eval_one_loss"]+metrics["eval_three_loss"])/2)
    assert metrics["eval_aa_loss"]==pytest.approx((metrics["eval_one_aa_loss"]+metrics["eval_three_aa_loss"])/2)
    assert "eval_three_terminal_count" in metrics


def test_memory_evidence_rejects_changed_code_or_unmeasured_batch(tmp_path,monkeypatch):
    import json,hashlib
    from scripts import profile_immune_v3 as profiling
    current={"model.py":"original"}
    monkeypatch.setattr(profiling,"code_hashes",lambda:current)
    summary={"A4":{"measurements":[{"batch_size":4,"status":"passed","precision":"fp32_parameters_adam_bf16_autocast"}]}}
    raw=json.dumps(summary)
    (tmp_path/"summary.json").write_text(raw)
    (tmp_path/"COMPLETE.json").write_text(json.dumps({"status":"passed","summary_sha256":hashlib.sha256(raw.encode()).hexdigest()}))
    (tmp_path/"source_sha256.json").write_text(json.dumps(current))
    assert profiling.verify_memory_profile(tmp_path,"A4",4)["batch_size"]==4
    with pytest.raises(ValueError,match="no passing profile"):
        profiling.verify_memory_profile(tmp_path,"A4",8)
    current["model.py"]="changed"
    with pytest.raises(ValueError,match="changed since"):
        profiling.verify_memory_profile(tmp_path,"A4",4)


def test_pilot_stop_preserves_scheduler_horizon():
    from transformers import TrainerControl,TrainerState
    from examples.llada.immune_pretrain_v3 import PilotStop
    state=TrainerState(global_step=1000,max_steps=10000)
    control=PilotStop(1000).on_step_end(None,state,TrainerControl())
    assert control.should_training_stop and control.should_save and control.should_evaluate
    assert state.max_steps==10000
