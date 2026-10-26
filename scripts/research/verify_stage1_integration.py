"""Verify real-weight pilot results, exact resume and KD gradients to BERT.

In pllm: USE_TF=0 python -B -m scripts.research.verify_stage1_integration
  --bert ABS_PARENT --requests ABS_JSON --cache ABS_CACHE --run ABS_RUN
  --resume-run ABS_RESUMED_RUN --output ABS_JSON
Requires actual teacher receipts and joint optimization; does not measure
scientific generalization or replace same-budget ablation experiments.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from dllm.pipelines.immune_llada.data.records import BioSeqRecord
from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMModel
from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import StructureMLMModel, file_sha256
from dllm.pipelines.immune_llada.fixed_chain.structure_losses import chain_aware_kd
from scripts.research.stage1_data import load_teacher_cache
from scripts.research.run_stage1_distillation import prepare_microbatch, write_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def states_equal(left,right):
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(states_equal(v,right[k]) for k,v in left.items())
    if isinstance(left,(list,tuple)):
        return type(left) is type(right) and len(left)==len(right) and all(states_equal(a,b) for a,b in zip(left,right))
    return left==right


def run(args):
    torch.set_num_threads(8)
    torch.manual_seed(42)
    run_dir,resume_dir,cache_dir=map(Path,(args.run,args.resume_run,args.cache))
    result=json.loads((run_dir / "result.json").read_text())
    resumed=json.loads((resume_dir / "result.json").read_text())
    teacher=json.loads((cache_dir / "verification.json").read_text())
    require(result["status"] == resumed["status"] == teacher["status"] == "passed","A prerequisite run failed")
    contract=teacher["teacher_contract"]
    require(contract["all_parameters_frozen"] and bool(contract["weight_sha256"]),"Missing actual frozen teacher")
    require(not any(teacher["loading_info"].get(k) for k in
                    ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")),"Teacher loading incomplete")
    require(all(r["status"] == "cached" and r["repeat_max_abs"] == 0 for r in teacher["results"]),
            "Teacher deterministic replay not verified")
    final,resume_final=Path(result["final_checkpoint"]),Path(resumed["final_checkpoint"])
    state_a=load_file(str(final / "model.safetensors"))
    state_b=load_file(str(resume_final / "model.safetensors"))
    require(state_a.keys() == state_b.keys() and all(torch.equal(v,state_b[k]) for k,v in state_a.items()),
            "Resume did not reproduce uninterrupted optimizer updates exactly")
    require(result["history"][-1]["objective"] == resumed["history"][-1]["objective"],"Resumed metrics differ")
    # Locally produced checkpoint states; compare values rather than pickle bytes.
    optimizer_states=[torch.load(path / "training_state.pt",map_location="cpu",weights_only=False)
                      for path in (final,resume_final)]
    require(states_equal(*optimizer_states),"Resumed optimizer/RNG/data-cursor state differs")
    del optimizer_states
    model=StructureMLMModel.from_pretrained(final).to(args.device).eval()
    require(model.pair_config.feedback == "all" and model.config.n_layers == 8,"Expected all eight feedback layers")
    requests=json.loads(Path(args.requests).read_text())["requests"]
    records=[BioSeqRecord.from_prepared_dict(r["record"]) for r in requests]
    caches=[load_teacher_cache(cache_dir / r["request_sha256"],r,allow_unvalidated=True) for r in requests]
    require(model.teacher_contract == contract,"Student teacher contract differs")
    generator=torch.Generator().manual_seed(345)
    batch,pairs,targets,_=prepare_microbatch(model,records,caches,generator,args.device)
    parent=FullTokenMLMModel.from_pretrained(args.bert).to(args.device).eval()
    initialized=StructureMLMModel.from_bert(args.bert,model.pair_config).to(args.device).eval()
    with torch.no_grad():
        old=parent(**batch)["logits"]
        initial=initialized(**batch)["logits"]
        require(torch.equal(old,initial),"Zero-feedback student differs from actual BERT")
        logits=model(pair_indices=pairs,**batch)["logits"]
        reloaded=StructureMLMModel.from_pretrained(final).to(args.device).eval()
        require(torch.equal(logits,reloaded(pair_indices=pairs,**batch)["logits"]),"Saved forward replay differs")
    groups={"backbone":[],"pair_encoder":[],"pair_bias_outputs":[],"distogram_head":[]}
    initial_state=load_file(str(run_dir / "initialized" / "model.safetensors"))
    for name,value in model.state_dict().items():
        group=next((g for g in groups if name.startswith(g+".")),None)
        if group and not torch.equal(value,initial_state[name].to(value.device)):
            groups[group].append(name)
    require(all(groups.values()),"A required parameter group never updated")
    del reloaded,parent,initialized,initial_state,state_a,state_b
    model.train().zero_grad(set_to_none=True)
    out=model(pair_indices=pairs,**batch)
    kd=chain_aware_kd(out["pair_logits"],targets,pairs,batch["chain_ids"],batch["noise_kind"])
    (kd["intra_sum"]+kd["inter_sum"]).backward()
    kd_backbone_norms=[float(block.base.q_proj.weight.grad.norm()) for block in model.backbone.transformer.blocks]
    require(all(x>0 for x in kd_backbone_norms),"KD does not reach every backbone layer")
    require(all(all(x>0 for x in step["bias_gradient_norms"]) for step in result["history"]),
            "At least one feedback layer has no joint-training gradient")
    require(not targets.requires_grad,"Teacher targets carry gradients")
    observed_noise=batch["noise_kind"][pairs[:,0,None],pairs[:,1:]]
    require(observed_noise.eq(1).any(),"Masked real residues are missing from KD targets")
    metadata=[c["metadata"] for c in caches]
    report=dict(status="passed",scope="engineering_only",parent=model.parent_provenance,
        teacher_repository=contract["repository"],teacher_revision=contract["revision"],
        teacher_parameters=contract["parameters"],teacher_all_frozen=True,
        teacher_loading_complete=True,teacher_same_seed_replay_exact=True,
        teacher_native_distogram_bins=contract["distogram_bins"],teacher_num_loops=contract["num_loops"],
        parent_zero_feedback_logits_exact=True,resume_all_model_tensors_exact=True,
        resume_optimizer_rng_cursor_exact=True,
        saved_forward_replay_exact=True,updated_tensors_by_group={k:len(v) for k,v in groups.items()},
        all_eight_bias_layers_receive_gradients=True,kd_backbone_q_gradient_norms=kd_backbone_norms,
        masked_true_residues_included=True,teacher_target_pairs=sum(m["pair_count"] for m in metadata),
        final_checkpoint=str(final.resolve()),model_sha256=file_sha256(final / "model.safetensors"),
        peak_joint_training_cuda_gb=result["peak_cuda_allocated_gb"],joint_steps=result["steps"],
        teacher_quality_validated=False,scientific_gain_evaluated=False)
    write_json(args.output,report)
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("bert","requests","cache","run","resume-run","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--device",default="cuda")
    run(parser.parse_args())
