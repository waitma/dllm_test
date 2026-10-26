"""Eight-rank memory profiling before freezing the BERT training microbatch.

Run through launch_bert_mlm.py with a frozen profile plan on the existing pool.
Each GPU probes independently, then all ranks agree on the largest safe batch.
"""
import gc
import json
from pathlib import Path
import time

import torch
from dllm.pipelines.immune_llada.fixed_chain.mlm import BertMLMCollator, BertMLMModel, corrupt
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
from scripts.research.bert_mlm_data import write_json
from scripts.research.bert_mlm_objectives import components


def synthetic_batch(config, micro, length):
    """Extend context to the configured ceiling, keeping real receptor targets."""
    collator = BertMLMCollator(config)
    residues = "ACDEFGHIKLMNPQRSTVWY" * 20
    row = BioSeqRecord([BioSeqChain("ACDEFGHIKLMNPQRSTVWY"*10, "antigen"),
                        BioSeqChain(residues[:config.first_chain_capacity-1], "antibody_heavy"),
                        BioSeqChain(residues[:config.second_chain_capacity-1], "antibody_light")],
                       "antibody_antigen", "synthetic_mlm_profile")
    batch = collator([row])
    extra = length - batch["input_ids"].shape[1]
    if extra < 0:
        raise ValueError("Synthetic length too short")
    index = int((batch["role_ids"][0].eq(7) & batch["residue_mask"][0]).nonzero()[0])
    for key, value in batch.items():
        extension = value[:, index:index+1].expand(1, extra).clone()
        if key == "position_ids":
            extension[:] = torch.arange(201, 201+extra)
        batch[key] = torch.cat([value[:, :index], extension, value[:, index:]], 1).repeat(micro, 1)
    return batch, collator.tokenizer


def attempt(config, micro, length, device):
    torch.manual_seed(42)
    model = BertMLMModel(config).to(device).train()
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    raw, tokenizer = synthetic_batch(config, micro, length)
    batch = {k: v.to(device) for k, v in corrupt(raw, tokenizer, torch.Generator().manual_seed(42)).items()}
    torch.cuda.reset_peak_memory_stats(device)
    losses = []
    start = time.monotonic()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch)["loss_sum"] / batch["labels"].ne(-100).sum()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize(device)
    result = dict(status="passed", micro_batch=micro, length=length, steps=3, losses=losses,
                  grad_norm=float(norm), seconds_per_step=(time.monotonic()-start)/3,
                  allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                  reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                  total_gib=torch.cuda.get_device_properties(device).total_memory/2**30,
                  parameters=sum(p.numel() for p in model.parameters()))
    if result["allocated_gib"] > .9*result["total_gib"]:
        result["status"] = "insufficient_headroom"
    return result


def main(plan, device, rank, world):
    global BertMLMCollator, BertMLMModel, corrupt
    BertMLMCollator, BertMLMModel, corrupt, _ = components(plan.get("objective", "receptor_aa"))
    config = FixedChainConfig(**plan["model"])
    reports, selected = [], None
    for micro in (64, 32, 16, 8):
        try:
            report = attempt(config, micro, config.max_length, device)
        except torch.cuda.OutOfMemoryError:
            report = dict(status="oom", micro_batch=micro, length=config.max_length)
        gc.collect()
        torch.cuda.empty_cache()
        statuses = [None]*world
        torch.distributed.all_gather_object(statuses, report)
        reports.append(report)
        if all(x["status"] == "passed" for x in statuses):
            selected = micro
            break
    out = Path(plan["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"profile_rank{rank}.json", dict(rank=rank, gpu=torch.cuda.get_device_name(device),
                                                    selected_micro_batch=selected, attempts=reports))
    torch.distributed.barrier()
    if selected is None:
        raise RuntimeError("No safe microbatch")
    if rank == 0:
        all_reports = [json.loads((out / f"profile_rank{i}.json").read_text()) for i in range(world)]
        write_json(out / "profile.json", dict(status="passed", world_size=world, reports=all_reports,
                                              micro_batch=selected, grad_accum=512//world//selected,
                                              global_batch=512, sequence_length=config.max_length))
