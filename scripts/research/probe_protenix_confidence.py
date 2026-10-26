"""Inspect native PDE confidence with real weights on eight engineering cases.

Activate pllm, then use general_teacher_protenix_20261024/bin/python on this
absolute script with --output ABS_DIR. Set CUBLAS_WORKSPACE_CONFIG=:4096:8.
This is a bounded engineering diagnostic, not a calibration fit or corpus run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from general_teacher_adapters import ProtenixMini, ROOT
from run_general_teacher_validation import seed, write


def diagnostic(example, logits, confidence, coordinates):
    if 'reference_coordinates' not in example:
        return None
    reference=np.load(example['reference_coordinates'])['distances']
    n=len(reference)
    chain=np.repeat(np.arange(len(example['chain_lengths'])),example['chain_lengths'])
    valid=np.isfinite(reference)&np.triu(np.ones((n,n),dtype=bool),k=1)
    sep=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    probability=logits.float().log_softmax(-1).cpu().numpy()
    c=confidence.cpu().numpy()
    d=torch.cdist(coordinates.float(),coordinates.float()).cpu().numpy()
    # Native distogram target definition: number of boundaries strictly below d.
    target=(reference[...,None]>np.linspace(2.3125,21.6875,63)).sum(-1)
    nll=-np.take_along_axis(probability,target[...,None],axis=-1)[...,0]
    correct=np.abs(d-reference)<=4
    result={}
    for name,mask in dict(intra=valid&(chain[:,None]==chain[None,:])&(sep>=24),
                          inter=valid&(chain[:,None]!=chain[None,:])).items():
        if not mask.any():continue
        groups=[]
        for low,high in zip((0,.2,.4,.6,.8),(.2,.4,.6,.8,1.000001)):
            use=mask&(c>=low)&(c<high)
            if use.any():
                groups.append(dict(lower=low,upper=min(1.,high),pairs=int(use.sum()),
                    predicted_mean=float(c[use].mean()),actual_coordinate_error_le_4A=float(correct[use].mean()),
                    native_distogram_true_bin_nll=float(nll[use].mean())))
        result[name]=dict(pairs=int(mask.sum()),confidence_mean=float(c[mask].mean()),
            actual_coordinate_error_le_4A=float(correct[mask].mean()),
            native_distogram_true_bin_nll=float(nll[mask].mean()),bins=groups)
    return result


def main(output):
    if not output.is_absolute():raise ValueError('Use an absolute output directory')
    output.mkdir(parents=True,exist_ok=True)
    receipt=output/'summary.json'
    if receipt.exists():raise FileExistsError('Use a new directory; preserve prior diagnostic receipts')
    report=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),results=[],
        scope='engineering; native PDE is not calibrated for distogram correctness',
        reference_note='2HHB raw chain assignment only; identical-chain symmetry not resolved in calibration diagnostics',
        examples_sha256=hashlib.sha256((ROOT/'examples.json').read_bytes()).hexdigest())
    write(receipt,report)
    try:
        torch.set_num_threads(4)
        teacher=ProtenixMini('cuda:0')
        weight=teacher.model.confidence_head.linear_no_bias_pde.weight
        if not torch.isfinite(weight).all() or not torch.count_nonzero(weight):
            raise ValueError('Native PDE head weights are nonfinite or zero')
        report.update(contract=teacher.contract,pde_head_nonzero=True,
            pde_threshold_angstrom=4,gpu=torch.cuda.get_device_name(0))
        for example in json.loads((ROOT/'examples.json').read_text())['examples']:
            start=time.perf_counter();features=teacher.prepare(example)
            prep=time.perf_counter()-start
            local_seed=int(example['sequence_sha256'][:8],16)%(2**31)
            with torch.inference_mode():
                seed(local_seed);torch.cuda.synchronize();start=time.perf_counter()
                trunk=teacher.predict(features).float()
                torch.cuda.synchronize();trunk_seconds=time.perf_counter()-start
                durations=[]
                for _ in range(2):
                    seed(local_seed);torch.cuda.synchronize();start=time.perf_counter()
                    result=teacher.predict_with_confidence(features)
                    torch.cuda.synchronize();durations.append(time.perf_counter()-start)
                logits,confidence=result['logits'],result['confidence']
                difference=float((logits.softmax(-1)-trunk.softmax(-1)).abs().max())
                n=sum(example['chain_lengths'])
                if logits.shape!=(n,n,64) or confidence.shape!=(n,n):
                    raise ValueError('Output shape mismatch')
                if not all(torch.isfinite(result[k]).all() for k in ('logits','confidence','coordinates')):
                    raise ValueError('Nonfinite output')
                if difference>1e-4 or float((confidence-confidence.T).abs().max())>1e-4:
                    raise ValueError('Trunk parity or symmetric PDE check failed')
                i,j=torch.triu_indices(n,n,offset=1,device='cuda')
                packed=logits[i,j].half()
                quantization=float((packed.float().softmax(-1)-logits[i,j].softmax(-1)).abs().max())
                if quantization>1e-4:raise ValueError('FP16 probability error exceeds 1e-4')
                path=output/(example['id']+'_upper_triangle.npz')
                np.savez_compressed(path,logits=packed.cpu().numpy(),
                    confidence=confidence[i,j].half().cpu().numpy(),length=np.array(n),
                    confidence_cdf=result['confidence_cdf'][i,j].half().cpu().numpy(),
                    chain_lengths=np.array(example['chain_lengths']))
                row=dict(id=example['id'],length=n,preprocessing_seconds=prep,
                    trunk_seconds=trunk_seconds,full_confidence_seconds=durations,
                    warmed_confidence_seconds=durations[-1],probability_parity_max_abs=difference,
                    fp16_probability_max_abs=quantization,confidence_min=float(confidence[i,j].min()),
                    confidence_mean=float(confidence[i,j].mean()),confidence_max=float(confidence[i,j].max()),
                    confidence_means_by_threshold={str(t):float(result['confidence_cdf'][i,j,k].mean())
                        for k,t in enumerate((1,2,4))},
                    diagnostic=diagnostic(example,logits,confidence,result['coordinates']),
                    artifact=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            report['results'].append(row);write(receipt,report)
            print(json.dumps({k:v for k,v in row.items() if k!='diagnostic'}),flush=True)
            del trunk,result,logits,confidence,packed,features
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        report['finished_utc']=datetime.now(timezone.utc).isoformat();write(receipt,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
