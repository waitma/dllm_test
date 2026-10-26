"""Single-device Stage 1 pilot with cached teacher targets and resumable updates.

Activate pllm, then:
  USE_TF=0 python -B -m scripts.research.run_stage1_distillation --bert ABS_PARENT
    --requests ABS_JSON --cache ABS_DIR --output ABS_DIR --steps 3
    --allow-unvalidated-smoke --checkpoint-every 1
This finite-corpus runner verifies local integration. Formal distributed runs
must be scheduled separately and qualify the complete training/validation data.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import torch

from dllm.pipelines.immune_llada.data.records import BioSeqRecord
from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMCollator, corrupt
from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import PairConfig, StructureMLMModel, file_sha256
from dllm.pipelines.immune_llada.fixed_chain.structure_losses import chain_aware_kd, normalized_objective
from scripts.research.stage1_data import verify_request, load_teacher_cache, batch_teacher_targets


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_optimizer(model, backbone_lr, pair_lr, weight_decay):
    inherited, added = [], []
    for name, parameter in model.named_parameters():
        (added if name.startswith(("pair_encoder.","layer_pair_encoders.","pair_bias_outputs.","distogram_head.")) else inherited).append(parameter)
    return torch.optim.AdamW([dict(params=inherited,lr=backbone_lr), dict(params=added,lr=pair_lr)],
                             weight_decay=weight_decay,foreach=False)


def prepare_microbatch(model, records, caches, generator, device):
    collator = FullTokenMLMCollator(model.config)
    clean = collator(records)
    indices, teacher, contract = batch_teacher_targets(caches,clean,device)
    noisy = corrupt(clean,collator.tokenizer,generator)
    noisy = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in noisy.items()}
    if teacher is None:
        teacher = torch.empty((0,model.pair_config.distogram_bins),device=device)
    return noisy, indices, teacher, contract


def autocast_context(device, precision):
    return torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def optimizer_step(model, optimizer, micros, *, intra_weight=.05, inter_weight=.05,
                   temperature=1., mask_weights=None, precision="fp32", clip_norm=1.):
    """Exact step-wide means for uneven microbatches, including MLM-only rows."""
    model.train()
    device = next(model.parameters()).device
    totals = torch.zeros(3,device=device)
    with torch.no_grad():
        for batch, pairs, teacher, _ in micros:
            groups = chain_aware_kd(torch.zeros_like(teacher),teacher,pairs,batch["chain_ids"],
                                    batch["noise_kind"],temperature,mask_weights)
            totals += torch.stack([batch["labels"].ne(-100).sum(),groups["intra_count"],groups["inter_count"]])
    if not totals.any():
        raise ValueError("Optimizer step contains no eligible supervision")
    optimizer.zero_grad(set_to_none=True)
    sums = torch.zeros(3,device=device)
    pair_count = 0
    for batch, pairs, teacher, _ in micros:
        with autocast_context(device,precision):
            out = model(pair_indices=pairs,**batch)
        kd = chain_aware_kd(out["pair_logits"],teacher,pairs,batch["chain_ids"],
                            batch["noise_kind"],temperature,mask_weights)
        loss = normalized_objective(out["loss_sum"],out["mlm_count"],kd,intra_weight,inter_weight,
                                    denominators=totals)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        sums += torch.stack([out["loss_sum"].detach(),kd["intra_sum"].detach(),kd["inter_sum"].detach()])
        pair_count += kd["pair_count"]
    bias_grad = [float(p.weight.grad.norm()) if p.weight.grad is not None else 0.
                 for p in model.pair_bias_outputs]
    diagnostics = dict(bias_gradient_norms=bias_grad,
        backbone_gradient_norm=float(model.backbone.transformer.wte.weight.grad.norm()),
        pair_gradient_norm=float(model.pair_encoder.single_projection.weight.grad.norm()),
        head_gradient_norm=float(model.distogram_head.weight.grad.norm()))
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),clip_norm,error_if_nonfinite=True)
    optimizer.step()
    means = (sums / totals.clamp_min(1)).tolist()
    return dict(mlm=means[0],intra_kl=means[1],inter_kl=means[2],
                objective=means[0]+intra_weight*means[1]+inter_weight*means[2],
                counts=totals.tolist(),pairs=pair_count,gradient_norm=float(grad_norm),**diagnostics)


def save_checkpoint(directory, model, optimizer, generator, step, cursor, contract):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(directory)
    torch.save(dict(schema="immune.structure_bert.pilot_state.v1",step=step,cursor=cursor,
                    optimizer=optimizer.state_dict(),noise_rng=generator.get_state(),
                    torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    contract=contract),directory / "training_state.pt")
    write_json(directory / "receipt.json",dict(model_sha256=file_sha256(directory / "model.safetensors"),
              state_sha256=file_sha256(directory / "training_state.pt"),step=step))


def restore_training(directory, optimizer, generator, contract):
    directory = Path(directory)
    receipt=json.loads((directory / "receipt.json").read_text())
    for filename,key in (("model.safetensors","model_sha256"),("training_state.pt","state_sha256")):
        if file_sha256(directory / filename) != receipt[key]:
            raise ValueError("Resume checkpoint receipt mismatch")
    # Only load locally produced, receipt-bound trainer states (pickle is not a data import format).
    state=torch.load(directory / "training_state.pt",map_location="cpu",weights_only=False)
    if state["schema"] != "immune.structure_bert.pilot_state.v1" or state["contract"] != contract:
        raise ValueError("Resume data/teacher/optimizer contract differs")
    optimizer.load_state_dict(state["optimizer"])
    generator.set_state(state["noise_rng"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"],state["cursor"]


def run(args):
    if int(os.environ.get("WORLD_SIZE","1")) != 1:
        raise ValueError("This local pilot runner is single-device; distributed execution is not qualified")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    generator=torch.Generator().manual_seed(args.seed+1)
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    started=time.time()
    request_file=Path(args.requests).resolve()
    requests=json.loads(request_file.read_text())["requests"]
    if not requests:
        raise ValueError("Empty pilot corpus")
    caches,records,cache_hashes=[],[],[]
    for request in requests:
        verify_request(request)
        records.append(BioSeqRecord.from_prepared_dict(request["record"]))
        directory=Path(args.cache).resolve() / request["request_sha256"]
        cache=load_teacher_cache(directory,request,allow_unvalidated=args.allow_unvalidated_smoke) if request["chains"] else None
        caches.append(cache)
        cache_hashes.append(file_sha256(directory / "metadata.json") if cache else None)
    teacher_contracts=[c["metadata"]["teacher_contract"] for c in caches if c is not None]
    teacher_contract=teacher_contracts[0] if teacher_contracts else None
    if any(c != teacher_contract for c in teacher_contracts):
        raise ValueError("Mixed teacher contracts")
    pair_config=PairConfig(width=args.pair_width,query_chunk=args.query_chunk,feedback=args.feedback,
                           distogram_bins=teacher_contract["distogram_bins"] if teacher_contract else 64)
    model=(StructureMLMModel.from_pretrained(args.resume) if args.resume
           else StructureMLMModel.from_bert(args.bert,pair_config)).to(args.device)
    if asdict(model.pair_config) != asdict(pair_config):
        raise ValueError("Resume architecture differs")
    if any(request["config"] != asdict(model.config) for request in requests):
        raise ValueError("Teacher request/student layout differs")
    if args.resume and model.teacher_contract != teacher_contract:
        raise ValueError("Resume teacher differs")
    model.teacher_contract=teacher_contract
    if not args.resume:
        model.save_pretrained(output / "initialized")
    optimizer=make_optimizer(model,args.backbone_lr,args.pair_lr,args.weight_decay)
    contract=dict(requests_sha256=file_sha256(request_file),cache_metadata_sha256=cache_hashes,
                  teacher=teacher_contract,pair=asdict(pair_config),parent=model.parent_provenance,
                  backbone_lr=args.backbone_lr,pair_lr=args.pair_lr,weight_decay=args.weight_decay,
                  intra_weight=args.intra_weight,inter_weight=args.inter_weight,temperature=args.temperature,
                  batch_size=args.batch_size,accumulation=args.accumulation,precision=args.precision,
                  seed=args.seed,clip_norm=args.clip_norm,allow_unvalidated_smoke=args.allow_unvalidated_smoke,
                  scheduler="constant_for_engineering_pilot",torch=torch.__version__,deterministic=True)
    step,cursor=restore_training(args.resume,optimizer,generator,contract) if args.resume else (0,0)
    if step >= args.steps:
        raise ValueError("Requested final step must exceed resume step")
    write_json(output / "run_config.json",dict(**contract,requested_steps=args.steps,device=args.device,
               scope="local_engineering_pilot",resume=str(Path(args.resume).resolve()) if args.resume else None))
    history=[]
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(step+1,args.steps+1):
        micros=[]
        for _ in range(args.accumulation):
            indices=[(cursor+i)%len(records) for i in range(args.batch_size)]
            cursor += args.batch_size
            micros.append(prepare_microbatch(model,[records[i] for i in indices],
                          [caches[i] for i in indices],generator,args.device))
        before=time.time()
        metrics=optimizer_step(model,optimizer,micros,intra_weight=args.intra_weight,
                  inter_weight=args.inter_weight,temperature=args.temperature,precision=args.precision,
                  clip_norm=args.clip_norm)
        metrics.update(step=step,seconds=time.time()-before)
        history.append(metrics)
        print(json.dumps(metrics),flush=True)
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics,allow_nan=False)+"\n")
        if step == args.steps or args.checkpoint_every and step % args.checkpoint_every == 0:
            save_checkpoint(output / f"checkpoint-{step}",model,optimizer,generator,step,cursor,contract)
    write_json(output / "result.json",dict(status="passed",scope="local_engineering_pilot",
               steps=args.steps,seconds=time.time()-started,history=history,parent=model.parent_provenance,
               peak_cuda_allocated_gb=torch.cuda.max_memory_allocated()/2**30 if torch.device(args.device).type == "cuda" else None,
               final_checkpoint=str(output / f"checkpoint-{args.steps}"),
               teacher_quality_validated=all(r["quality_status"]=="validated" for r in requests if r["chains"])))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bert",required=True)
    parser.add_argument("--requests",required=True)
    parser.add_argument("--cache",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--steps",type=int,default=3)
    parser.add_argument("--checkpoint-every",type=int,default=0)
    parser.add_argument("--batch-size",type=int,default=1)
    parser.add_argument("--accumulation",type=int,default=2)
    parser.add_argument("--pair-width",type=int,default=64)
    parser.add_argument("--query-chunk",type=int,default=16)
    parser.add_argument("--feedback",choices=("all","global","none"),default="all")
    parser.add_argument("--precision",choices=("fp32","bf16"),default="bf16")
    parser.add_argument("--backbone-lr",type=float,default=1e-5)
    parser.add_argument("--pair-lr",type=float,default=1e-4)
    parser.add_argument("--weight-decay",type=float,default=.01)
    parser.add_argument("--intra-weight",type=float,default=.05)
    parser.add_argument("--inter-weight",type=float,default=.05)
    parser.add_argument("--temperature",type=float,default=1.)
    parser.add_argument("--clip-norm",type=float,default=1.)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--cpu-threads",type=int,default=8)
    parser.add_argument("--allow-unvalidated-smoke",action="store_true")
    args=parser.parse_args()
    if min(args.steps,args.batch_size,args.accumulation,args.cpu_threads)<1 or args.checkpoint_every<0:
        parser.error("Invalid run sizes")
    if min(args.backbone_lr,args.pair_lr,args.clip_norm,args.temperature)<=0 or min(args.weight_decay,args.intra_weight,args.inter_weight)<0:
        parser.error("Invalid optimizer/loss settings")
    run(args)
