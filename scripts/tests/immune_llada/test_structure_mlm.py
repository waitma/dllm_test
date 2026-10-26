"""Run in pllm: USE_TF=0 python -B -m pytest THIS_FILE -q.

Small deterministic models test architecture and target contracts; actual-weight
verification is separate and never replaced by these random-model tests.
"""
from dataclasses import replace
import json

import pytest
import torch
from safetensors.torch import save_file

from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMCollator, FullTokenMLMModel
from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import StructureMLMModel, PairConfig, file_sha256
from dllm.pipelines.immune_llada.fixed_chain.structure_losses import chain_aware_kd, normalized_objective
from scripts.research.stage1_data import make_teacher_request, load_teacher_cache, batch_teacher_targets
from scripts.research.stage1_teacher import sample_pairs


@pytest.fixture(autouse=True)
def deterministic_small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(20)
    yield
    torch.set_num_threads(previous)


def config():
    return FixedChainConfig(d_model=32, n_layers=8, n_heads=4, mlp_hidden_size=64,
                            first_chain_capacity=16, second_chain_capacity=12, full_capacity=30)


def record(synthetic=False):
    return BioSeqRecord([
        BioSeqChain("ACDEFGHIK", "antibody_heavy", metadata={
            "synthetic_residue_mask": [int(synthetic)] * 3 + [0] * 6}),
        BioSeqChain("LMNPQR", "antibody_light")], "antibody", "unit_test")


def batch():
    clean = FullTokenMLMCollator(config())([record()])
    clean["noise_kind"] = torch.zeros_like(clean["input_ids"])
    return clean


def model(**kwargs):
    return StructureMLMModel(config(), PairConfig(width=8, query_chunk=8, **kwargs))


def test_parent_exact_zero_feedback_and_strict_roundtrip(tmp_path):
    parent = FullTokenMLMModel(config()).eval()
    parent.save_pretrained(tmp_path / "parent")
    student = StructureMLMModel.from_bert(tmp_path / "parent", PairConfig(width=8)).eval()
    assert student.parent_provenance["inherited_parameters"] == sum(p.numel() for p in parent.parameters())
    assert all(torch.equal(v, student.state_dict()[k]) for k, v in parent.state_dict().items())
    assert torch.equal(parent(**batch())["logits"], student(**batch())["logits"])
    student.save_pretrained(tmp_path / "student")
    restored = StructureMLMModel.from_pretrained(tmp_path / "student").eval()
    assert torch.equal(student(**batch())["logits"], restored(**batch())["logits"])
    with pytest.raises(ValueError):
        FullTokenMLMModel.from_pretrained(tmp_path / "student")


def test_all_eight_feedback_layers_get_gradients_and_teacher_is_detached():
    student = model().train()
    b = batch()
    positions = b["residue_mask"][0].nonzero().flatten()
    pairs = torch.tensor([[0, positions[0], positions[1]], [0, positions[0], positions[-1]]])
    output = student(pair_indices=pairs, **b)
    teacher = torch.randn_like(output["pair_logits"], requires_grad=True)
    kd = chain_aware_kd(output["pair_logits"], teacher, pairs, b["chain_ids"], b["noise_kind"])
    normalized_objective(output["loss_sum"], output["mlm_count"], kd).backward()
    assert teacher.grad is None
    assert all(p.weight.grad.isfinite().all() and p.weight.grad.abs().sum() > 0
               for p in student.pair_bias_outputs)
    assert student.pair_encoder.single_projection.weight.grad.abs().sum() > 0
    assert student.backbone.transformer.wte.weight.grad.abs().sum() > 0
    student.zero_grad(set_to_none=True)
    out = student(pair_indices=pairs, **b)
    kd = chain_aware_kd(out["pair_logits"], teacher, pairs, b["chain_ids"], b["noise_kind"])
    (kd["intra_sum"] + kd["inter_sum"]).backward()
    assert student.backbone.transformer.blocks[0].base.q_proj.weight.grad.abs().sum() > 0


def test_feedback_preserves_local_hard_mask_and_supervision_is_not_input():
    student = model().eval()
    for output in student.pair_bias_outputs:
        torch.nn.init.normal_(output.weight, std=.1)
    b = batch()
    captures = []
    handle = student.backbone.transformer.blocks[0].register_forward_hook(
        lambda module, args, out: captures.append(out.detach().clone()))
    reference = student(**b)["logits"]
    altered = {**b, "labels": torch.full_like(b["labels"], -100),
               "noise_kind": torch.ones_like(b["noise_kind"]),
               "residue_mask": ~b["residue_mask"], "teacher_logits": torch.randn(7, 64)}
    assert torch.equal(reference, student(**altered)["logits"])
    captures.clear()
    student(**b)
    changed = {**b, "input_ids": b["input_ids"].clone()}
    light = b["chain_ids"].eq(1) & b["residue_mask"]
    changed["input_ids"][light] = student.tokenizer.encode_residues("W")[0]
    student(**changed)
    handle.remove()
    heavy = b["chain_ids"].eq(0) & b["residue_mask"]
    assert torch.equal(captures[0][heavy], captures[1][heavy])


def test_chunk_checkpoint_matches_dense_gradients_and_pairs_are_symmetric():
    a = model().train()
    b = StructureMLMModel(config(), replace(a.pair_config, query_chunk=128,
                                            checkpoint_pair_chunks=False)).train()
    for output in a.pair_bias_outputs:
        torch.nn.init.normal_(output.weight, std=.03)
    b.load_state_dict(a.state_dict())
    x = batch()
    positions = x["residue_mask"][0].nonzero().flatten()
    pairs = torch.tensor([[0, positions[0], positions[-1]], [0, positions[-1], positions[0]]])
    oa, ob = a(pair_indices=pairs, **x), b(pair_indices=pairs, **x)
    assert torch.allclose(oa["pair_logits"][0], oa["pair_logits"][1], atol=1e-7)
    assert torch.allclose(oa["logits"], ob["logits"], atol=2e-6)
    for out in (oa, ob):
        (out["loss_sum"] + out["pair_logits"].square().sum()).backward()
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert (pa.grad is None) == (pb.grad is None), name
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, atol=3e-5, rtol=2e-4, msg=name)


def test_feedback_ablation_depths():
    assert [i for i in range(8) if model(feedback="all").uses_feedback(i)] == list(range(8))
    assert [i for i in range(8) if model(feedback="global").uses_feedback(i)] == [1, 3, 5, 7]
    assert not any(model(feedback="none").uses_feedback(i) for i in range(8))


def test_chain_normalization_missing_inter_and_mask_groups():
    pairs = torch.tensor([[0,0,1], [0,0,2], [0,1,2], [0,3,4], [0,0,3], [1,0,1]])
    chains = torch.tensor([[0,0,0,1,1], [0,0,-1,-1,-1]])
    noise = torch.tensor([[0,1,3,2,0], [0,0,0,0,0]])
    student, teacher = torch.randn(6,4,requires_grad=True), torch.randn(6,4)
    kd = chain_aware_kd(student, teacher, pairs, chains, noise)
    tl, sl = teacher.log_softmax(-1), student.log_softmax(-1)
    per = (tl.exp() * (tl-sl)).sum(-1)
    torch.testing.assert_close(kd["intra_sum"], (per[:3].mean()+per[3])/2+per[5])
    torch.testing.assert_close(kd["inter_sum"], per[4])
    assert kd["intra_count"] == 2 and kd["inter_count"] == 1
    assert (kd["vv_count"],kd["vm_count"],kd["mm_count"]) == (2,4,0)
    weighted = chain_aware_kd(student, teacher, pairs, chains, noise, mask_weights=(0,1,0))
    assert weighted["intra_count"] == 1 and weighted["inter_count"] == 1
    empty = chain_aware_kd(student[:0], teacher[:0], pairs[:0], chains, noise)
    loss = normalized_objective(student.sum()*0,0,empty)
    assert loss == 0
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))
    # A step split into uneven microbatches/ranks uses shared denominators.
    full = normalized_objective(per.sum(),6,kd,.2,.3)
    parts = []
    for row in (0,1):
        use = pairs[:,0].eq(row)
        sub = chain_aware_kd(student[use],teacher[use],pairs[use],chains,noise)
        parts.append(normalized_objective(per[use].sum(),use.sum(),sub,.2,.3,
                     denominators=(6,2,1), world_size=2))
    torch.testing.assert_close(full, sum(parts)/2)


def cache_fixture(tmp_path, request):
    pairs = sample_pairs(request, 8, 1)
    tensors = dict(teacher_pair_indices=pairs, teacher_logits=torch.randn(len(pairs),64),
                   student_positions=torch.tensor([p for c in request["chains"] for p in c["student_positions"]]))
    save_file(tensors, str(tmp_path / "targets.safetensors"))
    metadata = dict(schema="immune.structure_teacher.sparse_cache.v1",
                    request_sha256=request["request_sha256"], teacher_contract={"distogram_bins":64},
                    targets_sha256=file_sha256(tmp_path / "targets.safetensors"))
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))


def test_verified_mapping_keeps_masked_residues_and_rejects_changed_record(tmp_path):
    request = make_teacher_request(record(),config(),inter_evidence=[{"chains":[0,1],"evidence":"test pairing"}])
    cache_fixture(tmp_path, request)
    with pytest.raises(ValueError, match="quality unvalidated"):
        load_teacher_cache(tmp_path, request)
    cache = load_teacher_cache(tmp_path, request, allow_unvalidated=True)
    clean = batch()
    pairs, targets, _ = batch_teacher_targets([cache],clean,"cpu")
    assert len(pairs) == len(targets) == 24
    noisy = {**clean,"input_ids":clean["input_ids"].masked_fill(clean["residue_mask"],3),
             "noise_kind":clean["noise_kind"].masked_fill(clean["residue_mask"],1)}
    kd = chain_aware_kd(torch.randn_like(targets),targets,pairs,noisy["chain_ids"],noisy["noise_kind"])
    assert kd["mm_count"] == len(pairs)
    changed = {**clean,"input_ids":clean["input_ids"].clone()}
    changed["input_ids"][0,pairs[0,1]] = 0
    with pytest.raises(ValueError, match="different sequence"):
        batch_teacher_targets([cache],changed,"cpu")
    with (tmp_path / "targets.safetensors").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="file changed"):
        load_teacher_cache(tmp_path,request,allow_unvalidated=True)


def test_synthetic_fragment_and_inter_pair_eligibility():
    request = make_teacher_request(record(synthetic=True),config())
    assert len(request["chains"]) == 1 and request["chains"][0]["record_chain"] == 1
    with pytest.raises(ValueError, match="accepted chains"):
        make_teacher_request(record(synthetic=True),config(),inter_evidence=[{"chains":[0,1],"evidence":"x"}])
    partial = replace(record(),chains=[BioSeqChain("ACDE", "antibody_heavy",metadata={"sequence_scope":"cdr3"})])
    assert not make_teacher_request(partial,config())["chains"]
    request = make_teacher_request(record(),config())
    pairs = sample_pairs(request,100,1)
    cutoff = len(request["chains"][0]["sequence"])
    assert (pairs[:,0].lt(cutoff) == pairs[:,1].lt(cutoff)).all()


def test_optimizer_accumulation_and_exact_resume(tmp_path):
    from scripts.research.run_stage1_distillation import (
        make_optimizer, optimizer_step, save_checkpoint, restore_training)
    a,b=model(),model()
    b.load_state_dict(a.state_dict())
    oa,ob=[make_optimizer(m,1e-4,2e-4,.01) for m in (a,b)]
    one=batch()
    positions=one["residue_mask"][0].nonzero().flatten()
    p=torch.tensor([[0,positions[0],positions[1]],[0,positions[0],positions[-1]]])
    t=torch.randn(2,64)
    # Two supervised records and one MLM-only record, with unequal MLM counts.
    singles=[{k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in one.items()} for _ in range(3)]
    singles[1]["labels"][:,::2]=-100
    singles[2]["labels"][:,::3]=-100
    joined={k:torch.cat([x[k] for x in singles]) if isinstance(v,torch.Tensor) else v for k,v in one.items()}
    pp=torch.cat([p,p+torch.tensor([1,0,0])])
    optimizer_step(a,oa,[(joined,pp,torch.cat([t,t]),None)])
    micros=[(x,p,t,None) for x in singles[:2]]+[(singles[2],p[:0],t[:0],None)]
    optimizer_step(b,ob,micros)
    for name,value in a.state_dict().items():
        torch.testing.assert_close(value,b.state_dict()[name],atol=2e-6,rtol=2e-4,msg=name)
    generator=torch.Generator().manual_seed(117)
    save_checkpoint(tmp_path / "saved",b,ob,generator,1,3,{"test":"resume"})
    expected_random=torch.rand(11,generator=generator)
    optimizer_step(b,ob,micros)
    restored=StructureMLMModel.from_pretrained(tmp_path / "saved")
    opt=make_optimizer(restored,1e-4,2e-4,.01)
    restored_generator=torch.Generator()
    assert restore_training(tmp_path / "saved",opt,restored_generator,{"test":"resume"}) == (1,3)
    assert torch.equal(expected_random,torch.rand(11,generator=restored_generator))
    optimizer_step(restored,opt,micros)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in b.state_dict().items())
