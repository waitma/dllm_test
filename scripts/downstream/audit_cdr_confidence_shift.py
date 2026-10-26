"""Audit first-step raw-logit ordering using saved OAS CDR-span predictions.

Activate pllm/protenix_abtcr; from repository root run:
OPENBLAS_NUM_THREADS=4 python -B scripts/downstream/audit_cdr_confidence_shift.py --out ABS_NEW_DIRECTORY
CPU only. No sampling, training or new benchmark score. Teacher-partition
replacement is an oracle diagnostic, never a proposed deployable decoder.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    from scipy.special import logsumexp
    from scipy.stats import spearmanr
    root=ROOT/'output/diagnostics/final_layer_cdr_span_preservation_v2_20261018'
    assert json.loads((root/'status.json').read_text())['status']=='success'
    assert json.loads((root/'verification.json').read_text())['status']=='passed'
    vocab_path=ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint/vocabulary.json'
    vocab=json.loads(vocab_path.read_text())
    allowed=[vocab.index(x) for x in 'ACDEFGHIKLMNPQRSTVWY']+[vocab.index('<protd>'),vocab.index('.')]
    paths=[vocab_path,root/'status.json',root/'verification.json']
    arrays={}
    for model in ['parent','token1700','ce0_5000','ce1_5000']:
        p=root/f'{model}_predictions.npz';paths.append(p);arrays[model]=np.load(p)
    result={};checks=[]
    for region in ['CDR1','CDR2','CDR3']:
        prefix=region+'__span'
        raw={name:a[prefix+'_logits'].astype(np.float64)[:,allowed] for name,a in arrays.items()}
        ids=arrays['parent'][prefix+'_sequence_index']
        # The bridge's small protd penalty has no effect on these hard-masked logits.
        assert all(np.isneginf(x[:,-2]).all() for x in raw.values())
        teacher_max=raw['parent'].max(1);teacher_z=logsumexp(raw['parent'],axis=1)
        for name,x in raw.items():
            if name=='parent':continue
            np.testing.assert_array_equal(arrays[name][prefix+'_sequence_index'],ids)
            np.testing.assert_array_equal(arrays[name][prefix+'_gold'],arrays['parent'][prefix+'_gold'])
            maximum=x.max(1);z=logsumexp(x,axis=1);log_probability=maximum-z
            teacher_probability=teacher_max-teacher_z
            delta=maximum-teacher_max;partition=z-teacher_z;probability=log_probability-teacher_probability
            np.testing.assert_allclose(delta,partition+probability,atol=1e-12,rtol=1e-12)
            covariance=float(np.mean((partition-partition.mean())*(probability-probability.mean())))
            np.testing.assert_allclose(np.var(delta),np.var(partition)+np.var(probability)+2*covariance,atol=1e-12)
            # Arbitrary per-position offsets preserve all local probabilities and
            # argmax tokens, but raw position scores are free to move.
            offsets=np.linspace(-2.,2.,len(x))[:,None]
            np.testing.assert_allclose((x+offsets)-logsumexp(x+offsets,axis=1)[:,None],x-z[:,None],atol=1e-12)
            np.testing.assert_array_equal((x+offsets).argmax(1),x.argmax(1))
            scores=dict(raw=maximum,log_probability=log_probability,
                        teacher_partition_oracle=log_probability+teacher_z)
            metrics={}
            for score_name,score in scores.items():
                corr=[];jaccard=[];same=[];kept=[];per_sequence=[]
                for i in range(128):
                    mask=ids==i;parent=teacher_max[mask];student=score[mask]
                    cutoff=int(len(parent)*.75)
                    a=parent>=np.sort(parent)[cutoff]
                    b=student>=np.sort(student)[cutoff]
                    jac=float(np.sum(a&b)/np.sum(a|b))
                    correlation=None
                    if np.ptp(parent)>0 and np.ptp(student)>0:
                        correlation=float(spearmanr(parent,student).statistic)
                        assert np.isfinite(correlation);corr.append(correlation)
                    jaccard.append(jac);same.append(bool(np.array_equal(a,b)));kept.append(int(b.sum()))
                    per_sequence.append(dict(sequence=i,positions=len(parent),cutoff_index=cutoff,
                        parent_committed=np.flatnonzero(a).tolist(),candidate_committed=np.flatnonzero(b).tolist(),
                        raw_teacher_rank_spearman=correlation,jaccard=jac))
                metrics[score_name]=dict(mean_rank_spearman=float(np.mean(corr)),rank_valid_sequences=len(corr),
                    mean_commit_jaccard=float(np.mean(jaccard)),identical_commit_fraction=float(np.mean(same)),
                    mean_committed=float(np.mean(kept)),sequences=per_sequence)
            result[region+'/'+name]=dict(masked_positions=len(ids),raw_score_shift_mean=float(delta.mean()),
                raw_score_shift_variance=float(np.var(delta)),log_partition_shift_mean=float(partition.mean()),
                log_partition_shift_variance=float(np.var(partition)),log_probability_shift_variance=float(np.var(probability)),
                partition_probability_covariance=covariance,ordering=metrics)
            checks.append(dict(region=region,model=name,logit_shift_identity=True,
                               variance_decomposition=True,probability_and_argmax_offset_invariance=True))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'results.json',result)
    write(out/'plan.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
          source=str(root),scope='first forward of OAS screen whole-CDR masks only',
          confidence='AirGen argmax uses raw max logit; first linear4-step cutoff floor(.75*n) and strict-less tie rule',
          counterfactual='replace student per-position log partition by parent partition; oracle diagnostic, not benchmark performance',
          limitations='Small OAS screen, single forward, different scores may change tie counts; this cannot prove the source of Kong multi-step degradation.'))
    air=ROOT.parent/'airgen/AirGen-Dev/src/byprot/models/lm'
    paths += [Path(__file__),air/'dplm_multichain.py',air/'model_utils.py']
    write(out/'verification.json',dict(status='passed',checks=checks,source_sha256={str(p):sha(p) for p in paths}))
    write(out/'status.json',dict(status='success',finished_utc=datetime.now(timezone.utc).isoformat()))
    for name,row in result.items():
        print(name,{k:{a:round(v[a],6) for a in ['mean_rank_spearman','mean_commit_jaccard','identical_commit_fraction']} for k,v in row['ordering'].items()},flush=True)


if __name__=='__main__':main()
