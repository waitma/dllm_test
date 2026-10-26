"""Measure CE/representation gradient agreement on unlabeled OAS batches.

Activate pllm/protenix_abtcr and run locally:
python -B scripts/downstream/audit_final_preservation_gradients.py --out ABS_NEW_DIR
No optimizer updates. Uses the actual training losses, eight train batches and
eight existing screen batches. Raw gradients are not Adam parameter updates.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    prior = ROOT / 'output/diagnostics/final_layer_coordinate_pilot_20261018'
    spectral = ROOT / 'output/diagnostics/final_layer_spectral_pilot_20261018'
    data = ROOT / 'output/diagnostics/final_layer_preservation_pilot_20261018'
    arms = {'coordinate300': (prior / 'joint_coordinate', False),
            'uniform1000': (spectral / 'uniform_continue', False),
            'spectral1000': (spectral / 'spectral_continue', True)}
    sources = [Path(__file__), prior / 'calibration.npz', spectral / 'spectrum.npz',
               data / 'train.jsonl', data / 'screen.jsonl', data / 'schedule.json']
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                arms={k: str(v[0]) for k, v in arms.items()}, batches_per_split=8,
                selection='first eight fixed seed42 training-schedule batches; all128 screen in original order',
                gradients='raw Euclidean CE versus actual pooled loss +0.1 token relation + variance floor; no optimizer step',
                source_sha256={str(p): sha(p) for p in sources})
    write(out / 'plan.json', plan)
    status = dict(status='running', pid=os.getpid(), started_utc=plan['created_utc'])
    write(out / 'status.json', status)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from downstream.final_layer_preservation_eval import load
    from downstream.fixed_chain_eval import FixedChainBackbone
    from downstream.fixed_chain_layer_eval import block_hidden
    torch.set_num_threads(4)
    cal = np.load(prior / 'calibration.npz')
    spectrum = np.load(spectral / 'spectrum.npz')
    mu_s, mu_t, rms_s, rms_t, rotation = [torch.as_tensor(cal[k], device='cuda', dtype=torch.float32) for k in ['mu_s','mu_t','rms_s','rms_t','rotation']]
    axes = torch.tensor(spectrum['axes'].T, device='cuda', dtype=torch.float32)
    weights = torch.tensor(spectrum['weights'], device='cuda', dtype=torch.float32)
    denominator = float(np.mean(spectrum['eigenvalues'] * spectrum['weights']))
    rows = {split: [json.loads(line) for line in (data / f'{split}.jsonl').read_text().splitlines()] for split in ['train','screen']}
    indices = {'train': json.loads((data / 'schedule.json').read_text())[:8],
               'screen': [list(range(i, i+16)) for i in range(0,128,16)]}
    details, summary = [], {}
    def device(batch):
        return {k: v.to('cuda') if torch.is_tensor(v) else v for k,v in batch.items()}
    def pool(h, mask):
        return h.float().masked_fill(~mask[...,None],0).sum(1)/mask.sum(1,keepdim=True)
    def relation(s, t):
        s, t = s.float(), t.float()
        s, t = s-s.mean(-2,keepdim=True), t-t.mean(-2,keepdim=True)
        a, b = s@s.transpose(-1,-2), t@t.transpose(-1,-2)
        return (1-(a*b).sum((-1,-2))/(a.square().sum((-1,-2))*b.square().sum((-1,-2))).clamp_min(1e-20).sqrt()).mean()
    for name, (folder, use_spectral) in arms.items():
        model, identity = load(folder)
        from dllm.pipelines.immune_llada.fixed_chain.layout import FixedChainCollator
        from dllm.pipelines.immune_llada.data.records import BioSeqRecord
        backbone = FixedChainBackbone(model, identity)
        collator = FixedChainCollator.from_config(model.config, mode='train')
        names = set(json.loads((folder / 'checkpoint.json').read_text())['trainable_names'])
        params = [(n,p) for n,p in model.named_parameters() if n in names]
        for _, p in params: p.requires_grad_(True)
        model.train()
        for split, batches in indices.items():
            for batch_id, ids in enumerate(batches):
                torch.manual_seed(91700+batch_id)
                current = [rows[split][i] for i in ids]
                batch = device(collator(current))
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    ce = model(**batch)['loss']
                ce_grad = torch.autograd.grad(ce, [p for _,p in params])
                clean = device(backbone.collator([BioSeqRecord.from_prepared_dict(row) for row in current]))
                teacher = []
                hook = model.backbone.transformer.blocks[1].register_forward_hook(lambda mod, inp, result: teacher.append(block_hidden(result).detach()))
                try:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        _, hidden = model.denoise(**model.feature_args(clean), output_hidden_states=True)
                finally:
                    hook.remove()
                mask = clean['residue_mask'] & clean['attention_mask']
                s, t = pool(hidden, mask), pool(teacher[0], mask)
                error = (s-mu_s)/rms_s - ((t-mu_t)/rms_t)@rotation
                uniform = error.square().mean()
                weighted = ((error@rotation.T@axes).square()*weights).mean()/denominator
                student_tokens, teacher_tokens = [], []
                for i in range(len(current)):
                    available = mask[i].nonzero().flatten()
                    selected = available[torch.linspace(0,len(available)-1,32,device='cuda').long()]
                    student_tokens.append(hidden[i,selected]);teacher_tokens.append(teacher[0][i,selected])
                token = relation(torch.stack(student_tokens),torch.stack(teacher_tokens))
                sv, tv = (s-s.mean(0)).square().mean(), (t-t.mean(0)).square().mean().detach()
                rep = (weighted if use_spectral else uniform) + .1*token + F.relu(.25*tv-sv)/(tv+1e-12)
                rep_grad = torch.autograd.grad(rep, [p for _,p in params])
                stats = {}
                for (param_name,_), g, h in zip(params, ce_grad, rep_grad):
                    group = 'block'+str(int(param_name.split('.')[3])+1) if '.blocks.' in param_name else 'final_norm'
                    values = [float((g.float()*h.float()).sum()),float(g.float().square().sum()),float(h.float().square().sum())]
                    for key in [group,'all']:
                        stats.setdefault(key, np.zeros(3));stats[key] += values
                metrics = {k: dict(dot=float(v[0]),ce_norm=float(np.sqrt(v[1])),repr_norm=float(np.sqrt(v[2])),
                                   cosine=float(v[0]/np.sqrt(max(v[1]*v[2],1e-30))),
                                   repr_to_ce_norm=float(np.sqrt(v[2]/max(v[1],1e-30)))) for k,v in stats.items()}
                details.append(dict(arm=name,split=split,batch=batch_id,ce=float(ce.detach()),representation=float(rep.detach()),groups=metrics))
                del ce_grad, rep_grad, hidden, teacher, ce, rep
            selected = [x for x in details if x['arm']==name and x['split']==split]
            summary[name+'_'+split] = {group: dict(mean_cosine=float(np.mean([x['groups'][group]['cosine'] for x in selected])),
                negative_batches=sum(x['groups'][group]['cosine']<0 for x in selected),
                mean_repr_to_ce_norm=float(np.mean([x['groups'][group]['repr_to_ce_norm'] for x in selected]))) for group in selected[0]['groups']}
            status.update(arm=name,split=split)
            write(out / 'status.json', status)
            print(name,split,summary[name+'_'+split]['all'],flush=True)
        # Check the low-rank CE readout constraint separately from parameter-gradient agreement.
        heads = {n: list(p.shape) for n,p in model.named_parameters() if n not in names and ('ff_out' in n or 'lm_head' in n)}
        summary[name+'_head_parameter_shapes'] = heads
        del model,backbone,params
        torch.cuda.empty_cache()
    for path,digest in plan['source_sha256'].items(): assert sha(Path(path))==digest
    write(out / 'results.json',dict(summary=summary,batches=details))
    status.update(status='success',finished_utc=datetime.now(timezone.utc).isoformat())
    write(out / 'status.json',status)


if __name__ == '__main__':
    main()
