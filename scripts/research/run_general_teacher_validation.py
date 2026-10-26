"""Validate real frozen teachers on predeclared protein-only cases.

Activate pllm, then invoke the selected isolated Python with this absolute file
    and --model esmfold2|protenix|boltz2. --cpu-check validates imports and native
featurization without loading weights or running a GPU. Submit GPU execution
through the corresponding eval_jobs YAML. No training or full-corpus caching.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import random
import time
import traceback

import numpy as np
import torch

from general_teacher_adapters import ADAPTERS, ROOT


def write(path, obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def announce(**obj):
    print(json.dumps(obj),flush=True)


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def average_precision(y,score):
    """Threshold-group AP, with tied scores evaluated at the same threshold."""
    if not np.any(y):return None
    order=np.argsort(-score,kind='stable');y=y[order].astype(np.int64);score=score[order]
    end=np.r_[np.flatnonzero(np.diff(score)),len(score)-1]
    tp=np.cumsum(y)[end];prec=tp/(end+1)
    return float(np.sum(np.diff(np.r_[0,tp])*prec)/np.sum(y))


def quality(example, scores):
    if 'reference_coordinates' not in example:return None
    ref=np.load(example['reference_coordinates'])['distances']
    n=len(ref);chain=np.repeat(np.arange(len(example['chain_lengths'])),example['chain_lengths'])
    valid=np.isfinite(ref)&np.triu(np.ones((n,n),dtype=bool),k=1)
    separation=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    groups=dict(intra_long_range=valid&(chain[:,None]==chain[None,:])&(separation>=24),
                inter=valid&(chain[:,None]!=chain[None,:]))
    result={}
    for name,mask in groups.items():
        if not np.any(mask):continue
        y=(ref[mask]<8)
        result[name]=dict(pairs=int(mask.sum()),contacts=int(y.sum()),positive_fraction=float(y.mean()),
            ap={str(k):average_precision(y,score[mask]) for k,score in scores.items()})
    return result


def main(args):
    directory=ROOT/args.model
    directory.mkdir(parents=True,exist_ok=True)
    payload=json.loads((ROOT/'examples.json').read_text())
    examples=payload['examples']
    report=dict(model=args.model,started_utc=datetime.now(timezone.utc).isoformat(),
        scope='engineering sanity/performance; not held-out generalization',
        example_manifest_sha256=hashlib.sha256((ROOT/'examples.json').read_bytes()).hexdigest(),
        status='running',cpu_check=args.cpu_check,weights_cpu_check=args.weights_cpu_check,results=[])
    output=directory/('cpu_preflight.json' if args.cpu_check else 'weights_cpu_check.json' if args.weights_cpu_check else 'summary.json')
    write(output,report)
    try:
        torch.set_num_threads(4)
        device='cpu' if args.cpu_check or args.weights_cpu_check else 'cuda:0'
        if device!='cpu':
            if not torch.cuda.is_available():raise RuntimeError('CUDA required')
            report.update(gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,
                          cuda_version=torch.version.cuda)
        started=time.perf_counter()
        adapter=ADAPTERS[args.model](device,load_model=not args.cpu_check)
        report.update(model_load_seconds=time.perf_counter()-started,contract=adapter.contract)
        if not args.cpu_check:report['checkpoint_loading']=adapter.loading
        announce(event='adapter_ready',model=args.model)
        if args.weights_cpu_check:
            head=adapter.model.distogram_head if args.model=='protenix' else adapter.model.distogram_module
            values=[p.detach() for p in head.parameters()]
            if not all(torch.isfinite(p).all() for p in values) or not any(torch.count_nonzero(p)>0 for p in values):
                raise ValueError('Distance head is nonfinite or all-zero')
            report.update(status='passed',distance_head_parameters=sum(p.numel() for p in values))
            return
        for example in examples:
            name=example['id'];n=sum(example['chain_lengths'])
            local_seed=int(example['sequence_sha256'][:8],16)%(2**31)
            seed(local_seed)
            start=time.perf_counter();features=adapter.prepare(example)
            prep_seconds=time.perf_counter()-start
            row=dict(id=name,group=example['group'],length=n,chain_lengths=example['chain_lengths'],
                     seed=local_seed,preprocessing_seconds=prep_seconds)
            if args.cpu_check:
                row['status']='native_featurizer_passed'
                report['results'].append(row);write(output,report)
                announce(event='cpu_case_passed',model=args.model,id=name)
                continue
            def forward(full=False):
                seed(local_seed);torch.cuda.synchronize();t=time.perf_counter()
                with torch.inference_mode():x=adapter.predict(features,full=full)
                torch.cuda.synchronize();elapsed=time.perf_counter()-t
                return x.detach().float(),elapsed
            warm,warm_seconds=forward()
            if warm.shape!=(n,n,64) or not torch.isfinite(warm).all():
                raise ValueError('Invalid distance output '+str(warm.shape))
            row['warmup_seconds']=warm_seconds
            if args.model!='esmfold2' and name=='pdb_1jhl':
                full,seconds=forward(full=True)
                prob_diff=float((warm.softmax(-1)-full.softmax(-1)).abs().max())
                row['full_path_comparison']=dict(full_seconds=seconds,
                    logits_max_abs=float((warm-full).abs().max()),probability_max_abs=prob_diff,
                    probability_threshold=1e-4,status='passed' if prob_diff<=1e-4 else 'failed')
                write(directory/'full_path_comparison.json',row['full_path_comparison'])
                del full
                if prob_diff>1e-4:raise ValueError('Full/trunk distance distributions differ')
            torch.cuda.reset_peak_memory_stats()
            durations=[];replay=0.0
            for _ in range(payload['performance_repeats']):
                x,elapsed=forward();durations.append(elapsed)
                replay=max(replay,float((x-warm).abs().max()))
            symmetry=float((x-x.transpose(0,1)).abs().max())
            if symmetry>1e-4 or replay>1e-4:
                row.update(symmetry_max_abs=symmetry,replay_max_abs=replay,
                    probability_replay_max_abs=float((x.softmax(-1)-warm.softmax(-1)).abs().max()),
                    warm_forward_seconds=durations,status='failed')
                write(directory/'failed_case.json',row)
                raise ValueError(f'Symmetry/replay check failed: symmetry={symmetry}, replay={replay}')
            probabilities=x.softmax(-1)
            scores={k:probabilities[...,:k].sum(-1).cpu().numpy() for k in (18,19,20)}
            np.savez_compressed(directory/(name+'_short_distance_scores.npz'),
                                **{str(k):v for k,v in scores.items()})
            row.update(status='passed',warm_forward_seconds=durations,
                median_forward_seconds=float(np.median(durations)),
                median_preprocess_plus_forward_seconds=prep_seconds+float(np.median(durations)),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                symmetry_max_abs=symmetry,replay_max_abs=replay,
                scores_note='sums of first 18/19/20 native bins; boundary sensitivity, not calibrated P(d<8)',
                quality=quality(example,scores))
            report['results'].append(row);write(output,report)
            announce(event='case_complete',model=args.model,**row)
            del x,warm,probabilities,features
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        write(output,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=list(ADAPTERS),required=True)
    parser.add_argument('--cpu-check',action='store_true')
    parser.add_argument('--weights-cpu-check',action='store_true')
    main(parser.parse_args())
