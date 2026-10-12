"""Train/profile no-ESMC ablations on audited data.

Run: python examples/llada/immune_pretrain_v3.py --ablation A4 --profile --output-dir /absolute/profile/path
Train: python examples/llada/immune_pretrain_v3.py --ablation A4 --data-dir /absolute/immune_v7_clean_r2 --profile-dir /absolute/profile --output-dir /absolute/run --batch-size MEASURED_BATCH --grad-accum 256_DIVIDED_BY_BATCH
GPU jobs are submitted through the repository's Volc YAML workflow.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import RandomSampler, Subset
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from dllm.pipelines.immune_llada.data.dataset import PreparedImmuneDataset
from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
from dllm.pipelines.immune_llada.v3.model import ImmuneConfig, ImmuneModel
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_ready(directory):
    """Fail closed: a directory named 'clean' is not an audit certificate."""
    manifest_path = directory / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    certificate = json.loads((directory / "audit_certificate.json").read_text())
    if certificate.get("status") != "passed" or certificate.get("pending"):
        raise ValueError("immune_v7 audit is incomplete")
    if certificate.get("manifest_sha256") != sha(manifest_path):
        raise ValueError("Audit certificate belongs to a different dataset manifest")
    for split in manifest["splits"].values():
        for shard in split["shards"]:
            if sha(directory / shard["path"]) != shard.get("sha256"):
                raise ValueError(f"Dataset shard changed: {shard['path']}")
    if certificate.get("forbidden_overlap_rows") != 0 or certificate.get("family_split_overlap") != 0:
        raise ValueError("Nonzero forbidden overlap in audit")
    from scripts.data.tcr_scope_gate import require_current_scope
    require_current_scope(directory)
    return manifest, certificate


def validate_resume(checkpoint, provenance):
    """Resume only against the same data, objective, seed, and model contract."""
    saved = json.loads((checkpoint / "run_provenance.json").read_text())
    if saved != provenance:
        raise ValueError("Resume data or training provenance differs")


class ImmuneTrainer(Trainer):
    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return RandomSampler(dataset, generator=torch.Generator().manual_seed(self.args.data_seed))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = model(**inputs)
        # Return only scalar loss to Trainer, never full vocabulary predictions.
        self._component_metrics = {k:float(v) for k,v in result.items() if k not in {"loss","sample_metrics"}}
        if getattr(self,"_eval_components",None) is not None:
            # gather_for_metrics discards distributed final-batch duplicates.
            samples=self.accelerator.gather_for_metrics(result["sample_metrics"].double())
            self._eval_components += samples.sum(0)
        return (result["loss"], {"loss":result["loss"]}) if return_outputs else result["loss"]

    def evaluation_loop(self, *args, metric_key_prefix="eval", **kwargs):
        self._eval_components=torch.zeros(4,device=self.args.device,dtype=torch.float64)
        try:
            output=super().evaluation_loop(*args,metric_key_prefix=metric_key_prefix,**kwargs)
            totals=self._eval_components.cpu().tolist()
            for index,kind in enumerate(("aa","terminal")):
                loss,count=totals[2*index:2*index+2]
                output.metrics[f"{metric_key_prefix}_{kind}_loss"]=loss/max(count,1)
                output.metrics[f"{metric_key_prefix}_{kind}_count"]=count
            return output
        finally:
            self._eval_components=None

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        dataset=self.eval_dataset if eval_dataset is None else eval_dataset
        if not isinstance(dataset,dict):
            return super().evaluate(dataset,ignore_keys=ignore_keys,metric_key_prefix=metric_key_prefix)
        metrics={}
        for source,data in dataset.items():
            metrics.update(self.evaluate(data,ignore_keys=ignore_keys,metric_key_prefix=f"{metric_key_prefix}_{source}"))
        for kind in ("loss","aa_loss","terminal_loss"):
            metrics[f"{metric_key_prefix}_{kind}"]=sum(metrics[f"{metric_key_prefix}_{source}_{kind}"] for source in dataset)/len(dataset)
        self.log({key:value for key,value in metrics.items() if key in {f"{metric_key_prefix}_{k}" for k in ("loss","aa_loss","terminal_loss")}})
        return metrics

    def log(self, logs, start_time=None):
        if "loss" in logs and hasattr(self, "_component_metrics"):
            logs = {**logs, **{"batch_"+k:v for k,v in self._component_metrics.items()}}
        return super().log(logs, start_time)

    def _save(self, output_dir=None, state_dict=None):
        from safetensors.torch import save_file
        directory = Path(output_dir or self.args.output_dir)
        directory.mkdir(parents=True,exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        state_dict = model.state_dict() if state_dict is None else state_dict
        save_file({k:v.detach().cpu().contiguous() for k,v in state_dict.items()},str(directory/"model.safetensors"),metadata={"format":"pt"})
        torch.save(self.args,directory/"training_args.bin")
        (directory / "immune_config.json").write_text(json.dumps(asdict(model.config), indent=2)+"\n")
        (directory / "vocabulary.json").write_text(json.dumps([model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)], indent=2)+"\n")
        if hasattr(self, "run_provenance"):
            (directory / "run_provenance.json").write_text(json.dumps(self.run_provenance,indent=2)+"\n")


class PilotStop(TrainerCallback):
    """Pause a pilot without shortening its declared cosine schedule."""
    def __init__(self,step):
        self.step=step

    def on_step_end(self,args,state,control,**kwargs):
        if self.step and state.global_step>=self.step:
            control.should_training_stop=True
            control.should_save=True
            control.should_evaluate=True
        return control


def profile(args, model, collator):
    """Synthetic memory/forward/backward smoke; no benchmark data or training claims."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU profiling requires CUDA")
    device = torch.device("cuda")
    # Match Trainer(bf16=True): FP32 parameters/Adam states, BF16 autocast.
    # Converting all parameters to BF16 would underestimate production memory.
    model.to(device=device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Fit a long context to the exact configured max_length/capacity. Full-size
    # target reservations are included; this is an intentionally adverse case.
    target_slots = 2*args.full_capacity if args.ablation == "A4" else 302
    context_length = args.max_length-target_slots-12
    if context_length < 1:
        raise ValueError("max_length too short for profile")
    rec = BioSeqRecord([BioSeqChain("A"*context_length,"antigen"), BioSeqChain("C"+"A"*120+"W","antibody_heavy"),
                        BioSeqChain("C"+"A"*105+"F","antibody_light")],"antibody_antigen","synthetic_profile")
    batch = {k:v.to(device) for k,v in collator([rec]*args.batch_size).items()}
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    losses=[]
    for _ in range(args.profile_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            out=model(**batch)
        if not torch.isfinite(out["loss"]):
            raise FloatingPointError("Nonfinite profile loss")
        out["loss"].backward(); optimizer.step()
        losses.append(float(out["loss"]))
    torch.cuda.synchronize()
    result=dict(status="passed", kind="synthetic_memory_profile", ablation=args.ablation,
                gpu=torch.cuda.get_device_name(), batch_size=args.batch_size, max_length=args.max_length,
                actual_length=batch["input_ids"].shape[1], max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                total_gib=torch.cuda.get_device_properties(device).total_memory/2**30,
                parameters=sum(p.numel() for p in model.parameters()), losses=losses,
                precision="fp32_parameters_adam_bf16_autocast",
                elapsed_seconds=time.monotonic()-start, config=asdict(model.config))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir / "profile.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ablation",choices=["A0","A1","A2","A3","A4"],required=True)
    p.add_argument("--data-dir",type=Path,default=ROOT/"data/prepared/immune_v7_clean_r2")
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--grad-accum",type=int,default=8)
    p.add_argument("--max-steps",type=int,default=10000)
    p.add_argument("--pilot-stop-step",type=int,default=0)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-length",type=int,default=2048)
    p.add_argument("--full-capacity",type=int,default=256)
    p.add_argument("--fragment-capacity",type=int,default=64)
    p.add_argument("--profile",action="store_true")
    p.add_argument("--profile-dir",type=Path)
    p.add_argument("--profile-steps",type=int,default=3)
    p.add_argument("--resume-from-checkpoint",type=Path)
    p.add_argument("--fsdp",default="")
    args=p.parse_args()
    set_seed(args.seed)
    config=ImmuneConfig(ablation=args.ablation,max_length=args.max_length,full_capacity=args.full_capacity,fragment_capacity=args.fragment_capacity)
    if args.resume_from_checkpoint:
        prior=json.loads((args.resume_from_checkpoint/"immune_config.json").read_text())
        if prior != asdict(config):
            raise ValueError("Resume architecture/vocabulary/layout configuration differs")
    model=ImmuneModel(config)
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    collator=ImmuneCollator(args.ablation,max_length=args.max_length,full_capacity=args.full_capacity,fragment_capacity=args.fragment_capacity)
    if args.profile:
        profile(args,model,collator)
        return
    manifest,certificate=assert_ready(args.data_dir)
    if args.profile_dir is None:
        raise ValueError("Formal training requires --profile-dir with measured memory evidence")
    from scripts.profile_immune_v3 import verify_memory_profile,code_hashes
    measured=verify_memory_profile(args.profile_dir,args.ablation,args.batch_size)
    if measured["config"]!=asdict(config):
        raise ValueError("Profile model configuration differs from training")
    import os
    world=int(os.environ.get("WORLD_SIZE","1"))
    if args.batch_size*args.grad_accum*world != 256:
        raise ValueError("Architecture suite requires global effective batch 256")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    provenance=dict(config=asdict(config),data_manifest_sha256=sha(args.data_dir/"dataset_manifest.json"),
                    certificate_sha256=sha(args.data_dir/"audit_certificate.json"),seed=args.seed,
                    downstream_scope_certificate_sha256=sha(args.data_dir/"downstream_scope_certificate.json"),
                    global_batch=256,source_sampling="natural_fixed_seed",model_parameters=sum(p.numel() for p in model.parameters()),
                    learning_rate=1e-4,warmup_steps=500,scheduler="cosine",scheduler_horizon=args.max_steps)
    provenance["source_sha256"]=code_hashes()
    if args.resume_from_checkpoint:
        validate_resume(args.resume_from_checkpoint,provenance)
    if int(os.environ.get("RANK","0"))==0:
        (args.output_dir/"run_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    train=PreparedImmuneDataset(args.data_dir,split="train")
    valid={}
    # Equal-source macro validation; freeze the same subset per source for all
    # architectures. Training sampling remains the separately declared policy.
    for source in sorted({s["source"] for s in manifest["splits"]["valid"]["shards"]}):
        data=PreparedImmuneDataset(args.data_dir,split="valid",source=source)
        indices=random.Random(42).sample(range(len(data)),min(len(data),2000))
        valid[source]=Subset(data,indices)
    training=TrainingArguments(output_dir=str(args.output_dir),per_device_train_batch_size=args.batch_size,
             per_device_eval_batch_size=args.batch_size,gradient_accumulation_steps=args.grad_accum,
             max_steps=args.max_steps,learning_rate=1e-4,warmup_steps=500,lr_scheduler_type="cosine",
             seed=args.seed,data_seed=42,bf16=True,remove_unused_columns=False,label_names=["labels"],
             eval_strategy="steps",eval_steps=1000,save_strategy="steps",save_steps=1000,
             save_total_limit=3,logging_steps=20,report_to="none",prediction_loss_only=True,
             dataloader_num_workers=4,fsdp=args.fsdp,
             fsdp_config={"transformer_layer_cls_to_wrap":["LLaDALlamaBlock"],"use_orig_params":True} if args.fsdp else None)
    trainer=ImmuneTrainer(model=model,args=training,train_dataset=train,eval_dataset=valid,data_collator=collator)
    trainer.run_provenance=provenance
    if args.pilot_stop_step:
        trainer.add_callback(PilotStop(args.pilot_stop_step))
    result=trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_metrics("train",result.metrics)
    trainer.save_model()


if __name__=="__main__":
    main()
