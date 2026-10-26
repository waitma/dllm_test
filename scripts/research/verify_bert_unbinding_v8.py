"""Single-A100 profile and real-parent/real-data V8 loss/gradient check.

Activate pllm, then USE_TF=0 python -B -m scripts.research.verify_bert_unbinding_v8.
This diagnostic updates only an in-memory model; it never starts formal training.
"""
import gc
import argparse
import json
import os
from pathlib import Path

import torch

from dllm.pipelines.immune_llada.fixed_chain.unbinding_mlm import (
    UnbindingMLMCollator, UnbindingMLMModel, corrupt)
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from scripts.research.bert_mlm_data import sha, write_json, IndexedPreparedDataset
from scripts.research.bert_mlm_v8_data import INDEX_DIR

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'output/bert_unbinding_v8_from287000_20261026'
PARENT = ROOT / 'output/bert_full_token_v7_300k_20261017/train/checkpoint-287000'


def main(run=RUN, index_dir=INDEX_DIR, reuse_profile=None):
    os.environ['IMMUNE_SDPA_BACKEND'] = 'efficient'
    from scripts.research import profile_bert_mlm as profile
    profile.BertMLMCollator, profile.BertMLMModel, profile.corrupt = UnbindingMLMCollator, UnbindingMLMModel, corrupt
    config = FixedChainConfig(**json.loads((PARENT / 'immune_config.json').read_text()))
    device = torch.device('cuda', 0)
    reports = []
    reused = json.loads(reuse_profile.read_text()) if reuse_profile else None
    if reused and (reused['status'] != 'passed' or reused['sequence_length'] != config.max_length
                   or reused['reports'][-1]['parameters'] != 75601176
                   or reused['objective_sha256'] != sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/unbinding_mlm.py')):
        raise ValueError('Profile does not cover the same model/objective/length')
    for micro in (() if reused else (64, 32, 16, 8)):
        try:
            result = profile.attempt(config, micro, config.max_length, device)
        except torch.cuda.OutOfMemoryError:
            result = dict(status='oom', micro_batch=micro)
        reports.append(result)
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps(result), flush=True)
        if result['status'] == 'passed':
            break
    if reused:
        reports = reused['reports']
    if reports[-1]['status'] != 'passed':
        raise RuntimeError('No safe single-GPU profile')
    out = run / 'qualification'
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'single_gpu_profile.json', dict(status='passed', world_size=1,
        gpu=torch.cuda.get_device_name(), reports=reports,
        micro_batch=reports[-1]['micro_batch'], sequence_length=config.max_length,
        objective_sha256=sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/unbinding_mlm.py'),
        reused_from=str(reuse_profile) if reuse_profile else None,
        reused_sha256=sha(reuse_profile) if reuse_profile else None))
    index = json.loads((index_dir / 'index.json').read_text())
    data = IndexedPreparedDataset(index_dir / 'index.json')
    negative, positive, start = [], [], 0
    for shard in index['shards']:
        if shard['split'] != 'train':
            continue
        if Path(shard['path']).name.startswith('abexp_'):
            # Expansion shards can contain both labels; include each observed class.
            selected = set()
            for i in range(shard['records']):
                relation = data[start + i].labels.get('relation')
                if relation not in selected:
                    (negative if relation == 'nonbinding' else positive).append(start + i)
                    selected.add(relation)
                if selected == {'binding', 'nonbinding'}:
                    break
        elif Path(shard['path']).name.startswith('unbind_'):
            negative.append(start)
            for i in range(1, min(100, shard['records'])):
                if data[start + i].metadata.get('mhc_update'):
                    negative.append(start + i)
                    break
        elif len(positive) < 32:
            positive.extend(range(start, start + min(32 - len(positive), shard['records'])))
        start += shard['records']
    if len(positive) < len(negative):
        raise ValueError('Insufficient positive diagnostic records')
    records = [data[i] for pair in zip(negative, positive) for i in pair]
    model = UnbindingMLMModel.from_parent(PARENT).to(device).train()
    from safetensors.torch import load_file
    parent_state = load_file(str(PARENT / 'model.safetensors'))
    if any(not torch.equal(value.cpu(), parent_state[key]) for key, value in model.state_dict().items()):
        raise ValueError('Parent state not inherited exactly')
    del parent_state
    collator = UnbindingMLMCollator(config)
    clean = collator(records)
    noisy = corrupt(clean, model.tokenizer, torch.Generator().manual_seed(42))
    neg = clean['input_ids'].eq(model.tokenizer.special_id('<nonbinding>')).any(1)
    if int(neg.sum()) != len(negative):
        raise ValueError('Real-data diagnostic must include restored negatives')
    if not torch.equal(noisy['labels'][neg].ne(-100), clean['explicit_relation_mask'][neg]):
        raise ValueError('Sequence loss leaked into negatives')
    nonrelation = ~clean['explicit_relation_mask'][neg]
    if not torch.equal(noisy['input_ids'][neg][nonrelation], clean['input_ids'][neg][nonrelation]):
        raise ValueError('Negative context was corrupted')
    batch = {key: value.to(device) for key, value in noisy.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = model(**batch)
        loss = result['loss_sum'] / batch['labels'].ne(-100).sum()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite mixed-batch loss')
    report = dict(status='passed', parent_step=287000, exact_parent_state=True,
        parent_model_sha256=sha(PARENT / 'model.safetensors'), data_index_sha256=sha(index_dir / 'index.json'),
        records=len(records), negatives=int(neg.sum()), negative_sequence_targets=0,
        negative_context_unchanged=True, loss=float(loss.detach()), grad_norm=float(norm),
        selected_tokens=int(batch['labels'].ne(-100).sum()), world_size=1,
        completed_mhc_records=sum(bool(r.metadata.get('mhc_update')) for r in records),
        completed_negative_mhc_records=sum(bool(r.metadata.get('mhc_update'))
            for r, is_negative in zip(records, neg.tolist()) if is_negative),
        actual_sequence_length=int(batch['input_ids'].shape[1]))
    report['added_antibody_records'] = sum(r.source == 'igdesign_antibody' for r in records)
    report['added_antibody_negatives'] = sum(r.source == 'igdesign_antibody' and is_negative
        for r, is_negative in zip(records, neg.tolist()))
    manifest = json.loads(Path(index['manifest']).read_text())
    if manifest.get('antibody_expansion') and not report['added_antibody_negatives']:
        raise ValueError('No newly added antibody negative exercised')
    if manifest.get('mhc_completion', {}).get('counts', {}).get('changed', 0) and not report['completed_negative_mhc_records']:
        raise ValueError('No newly completed MHC record exercised')
    write_json(out / 'real_data_gpu_check.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--index-dir', type=Path, default=INDEX_DIR)
    parser.add_argument('--reuse-profile', type=Path)
    args = parser.parse_args()
    main(args.run_dir.resolve(), args.index_dir.resolve(), args.reuse_profile)
