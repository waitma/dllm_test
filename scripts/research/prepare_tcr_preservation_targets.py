"""Cache frozen original-final TCR replay targets and diagnose current drift.

Activate pllm/protenix_abtcr; from repository root run:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B \
  scripts/research/prepare_tcr_preservation_targets.py --out ABS_NEW_DIRECTORY
Use only the prepared4096/126 OTS replay pool. Cache native and no-MASK target
features, fit scalar calibration on train only, and compare the completed CE1
candidate on128 train plus126 screen rows. No labels, probe fit, or training.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    status=dict(status='running',stage='prepare',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json',status)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    torch.set_num_threads(4)
    data=ROOT/'output/diagnostics/tcr_preservation_replay_20261018'
    source_hashes={}

    def pin(path):
        source_hashes[str(path)]=sha(path)
        return path

    manifest=json.loads(pin(data/'manifest.json').read_text())
    assert json.loads(pin(data/'verification.json').read_text())['status']=='passed'
    groups,origins={},{}
    for split in ['train','screen']:
        path=pin(data/f'{split}.jsonl')
        assert sha(path)==manifest['files'][split]['sha256']
        groups[split]=[json.loads(line) for line in path.read_text().splitlines()]
        origins[split]=json.loads(pin(data/f'{split}_origins.json').read_text())
        assert sha(data/f'{split}_origins.json')==manifest['files'][split]['origins_sha256']
        assert len(groups[split])==len(origins[split])==manifest['files'][split]['rows']
    checkpoint=ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint'
    candidate=ROOT/'output/diagnostics/final_layer_joint_convergence_pilot_20261018/token_all_direction'
    assert json.loads(pin(candidate.parent/'status.json').read_text())['status']=='success'
    assert json.loads(pin(candidate/'verification.json').read_text())['status']=='passed'
    from downstream.fixed_chain_eval import load,FixedChainBackbone
    model,identity=load(checkpoint)
    assert identity==manifest['parent_model_identity']
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord
    from downstream.grammar.tcr_features import TCRRuntimeEmbedder

    class NoMaskBackbone(FixedChainBackbone):
        def _final_hidden(self,batch):
            modified={**batch,'attention_mask':batch['attention_mask'] & batch['input_ids'].ne(self.model.tokenizer.mask_token_id)}
            assert not (batch['residue_mask'] & batch['attention_mask'] & ~modified['attention_mask']).any()
            return super()._final_hidden(modified)

    classes=dict(native=FixedChainBackbone,no_mask_keys=NoMaskBackbone)
    records={split:[BioSeqRecord.from_prepared_dict(row) for row in rows] for split,rows in groups.items()}
    plan=dict(created_utc=status['started_utc'],data_manifest=str(data/'manifest.json'),teacher_identity=identity,
              teacher='frozen original fixed50000 actual final hidden, no adaptation or projection',
              modes=list(classes),calibration='per-mode train4096 mean and scalar centered RMS; screen not fitted',
              student_diagnostic=str(candidate),student_rows=dict(train=128,screen=len(records['screen'])),
              labels_used=False,training_started=False,
              intent='prepare a later TCR feature-preservation replay control; cache alone does not establish recovery')
    write(out/'plan.json',plan)
    features={split:{} for split in records}
    for mode,cls in classes.items():
        backbone=cls(model,identity,batch_size=8)
        runtime=TCRRuntimeEmbedder(backbone=backbone,batch_size=8)
        assert runtime.protocols['cdr3ab'].sha256==manifest['input_protocol']['sha256']
        first=origins['train'][:16]
        direct=backbone.vectors(records['train'][:16])
        runtime_values=runtime.embed_pairs([r['beta'] for r in first],[r['alpha'] for r in first])
        np.testing.assert_array_equal(direct,runtime_values)
        for split,rows in records.items():
            chunks=[]
            for start in range(0,len(rows),256):
                chunks.append(backbone.vectors(rows[start:start+256]))
                status.update(stage='teacher_features',mode=mode,split=split,completed=min(start+256,len(rows)),total=len(rows))
                write(out/'status.json',status)
                print(mode,split,status['completed'],len(rows),flush=True)
            features[split][mode]=np.concatenate(chunks)
            assert features[split][mode].shape==(len(rows),768)
        del backbone,runtime
    for split,arrays in features.items():np.savez_compressed(out/f'{split}_targets.npz',**arrays)
    calibration={}
    for mode in classes:
        x=features['train'][mode].astype(np.float64)
        mean=x.mean(0);rms=float(np.sqrt(np.mean((x-mean)**2)))
        assert rms>0 and np.isfinite(x).all()
        calibration[mode+'_mean']=mean;calibration[mode+'_rms']=rms
    np.savez_compressed(out/'calibration.npz',**calibration)
    del model
    torch.cuda.empty_cache()
    from downstream.final_layer_preservation_eval import load as load_student
    model,student_identity=load_student(candidate)
    write(out/'student_identity.json',student_identity)
    # Verify this model/readout against its already audited GDP cache before
    # diagnosing a new unlabeled pool. This is inference only, no refit.
    from downstream.grammar.ab_probes import prepare_task
    from dllm.pipelines.immune_llada.data.records import BioSeqChain
    _,pairs,_,_=prepare_task('gdp_a1')
    gdp=[BioSeqRecord([BioSeqChain(h,'antibody_heavy'),BioSeqChain(l,'antibody_light')],task_type='antibody',source='native_v5_probe') for h,l in pairs]
    diagnostics={};student={split:{} for split in records}
    for mode,cls in classes.items():
        backbone=cls(model,student_identity,batch_size=8)
        cache=candidate/('final_features.npy' if mode=='native' else 'final_no_mask_features.npy')
        np.testing.assert_array_equal(backbone.vectors(gdp),np.load(pin(cache)))
        for split,all_rows in records.items():
            rows=all_rows[:128] if split=='train' else all_rows
            x=backbone.vectors(rows).astype(np.float64);y=features[split][mode][:len(rows)].astype(np.float64)
            student[split][mode]=x
            rms=calibration[mode+'_rms']
            ux=x/np.linalg.norm(x,axis=1,keepdims=True);uy=y/np.linalg.norm(y,axis=1,keepdims=True)
            dx=1.-ux@ux.T;dy=1.-uy@uy.T
            upper=np.triu_indices(len(rows),1)
            xc=x-x.mean(0);yc=y-y.mean(0)
            kx=xc@xc.T;ky=yc@yc.T
            diagnostics[split+'/'+mode]=dict(rows=len(rows),teacher_train_centered_rms=rms,
                normalized_coordinate_mse=float(np.mean((x-y)**2)/rms**2),
                normalized_centered_mse=float(np.mean((xc-yc)**2)/rms**2),
                mean_pointwise_cosine=float(np.mean((ux*uy).sum(1))),
                pair_distance_spearman=float(spearmanr(dx[upper],dy[upper]).statistic),
                centered_cka=float(np.sum(kx*ky)/np.sqrt(np.sum(kx**2)*np.sum(ky**2))))
    for split,arrays in student.items():np.savez_compressed(out/f'{split}_student_diagnostic.npz',**arrays)
    write(out/'diagnostics.json',diagnostics)
    for path in [Path(__file__),ROOT/'downstream/fixed_chain_eval.py',ROOT/'downstream/grammar/tcr_features.py',ROOT/'downstream/final_layer_preservation_eval.py',candidate/'checkpoint.json']:
        pin(path)
    for path,digest in source_hashes.items():assert sha(Path(path))==digest,path
    write(out/'verification.json',dict(status='passed',prepared_vs_runtime_first16_bitwise=True,
          candidate_gdp_both_readouts_bitwise=True,calibration_train_only=True,
          teacher_target_rows={s:len(r) for s,r in records.items()},source_sha256=source_hashes,
          note='GDP cache used only for loader/readout verification; no labels or probe fitting in target preparation'))
    status.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json',status)
    print(json.dumps(diagnostics,indent=2),flush=True)


if __name__=='__main__':main()
