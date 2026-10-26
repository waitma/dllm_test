"""Test all-global representation attention in two completed trained models.

Activate pllm/protenix_abtcr and run locally:
python -B scripts/downstream/audit_trained_attention_topology.py --out ABS_NEW_DIR
All8 blocks and final RMSNorm execute; only attention topology changes in
memory during diagnostic forwards. Original options are restored each time.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    arms={'spectral1000':ROOT/'output/diagnostics/final_layer_spectral_pilot_20261018/spectral_continue',
          'aligned1700':ROOT/'output/diagnostics/final_layer_attention_alignment_20261018/nomask_continue'}
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms={k:str(v) for k,v in arms.items()},
              modes=['all_global','all_global_no_mask_keys'],readout='actual final RMSNorm hidden, global residue mean',
              intervention='segment_global=False gives global attention in all8 blocks; original True is alternating segment/global',
              scope='frozen representation-only diagnostic; all modes and properties, not a production default or generation claim',
              code_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'downstream/final_layer_preservation_eval.py',ROOT/'downstream/grammar/ab_probes.py']})
    write(out/'plan.json',plan);status=dict(status='running',pid=os.getpid(),started_utc=plan['created_utc']);write(out/'status.json',status)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from downstream.final_layer_preservation_eval import load
    from downstream.fixed_chain_eval import FixedChainBackbone
    torch.set_num_threads(4);results,checks,topology={},{},[]
    for name,folder in arms.items():
        model,identity=load(folder)
        from dllm.pipelines.immune_llada.data.records import BioSeqRecord,BioSeqChain
        from downstream.grammar.ab_probes import prepare_task,run_gdp
        from downstream.ophiuchus_eval.developability import FOLD_COL,ALL_COLS
        df,pairs,_,_=prepare_task('gdp_a1')
        records=[BioSeqRecord([BioSeqChain(h,'antibody_heavy'),BioSeqChain(l,'antibody_light')],task_type='antibody',source='native_v5_probe') for h,l in pairs]
        backbone=FixedChainBackbone(model,identity);options=dict(model.options)
        assert options['segment_global'] is True
        controls=np.load(folder/'final_features.npy');chunks={mode:[] for mode in plan['modes']}
        for start in range(0,len(records),8):
            batch=backbone.collator(records[start:start+8]);batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
            mask=batch['residue_mask'] & batch['attention_mask']
            def pool(h):return h.float().masked_fill(~mask[...,None],0).sum(1)/mask.sum(1,keepdim=True)
            original,_=backbone._final_hidden(batch)
            np.testing.assert_array_equal(pool(original).cpu().numpy(),controls[start:start+8])
            for mode in chunks:
                current=dict(batch)
                if mode=='all_global_no_mask_keys':current['attention_mask']=batch['attention_mask'] & batch['input_ids'].ne(model.tokenizer.mask_token_id)
                model.options['segment_global']=False
                try:
                    if start==0:
                        for i in range(8):
                            allowed=model.layer_allowed(i,current['chain_ids'],current['position_ids'],current['attention_mask'])
                            cross=mask[:,:,None] & mask[:,None,:] & current['chain_ids'][:,:,None].ne(current['chain_ids'][:,None,:])
                            count=int((allowed & cross).sum());assert count==int(cross.sum()) and count>0
                            topology.append(dict(arm=name,mode=mode,block=i+1,cross_residue_edges=count))
                    hidden,_=backbone._final_hidden(current)
                finally:
                    model.options.clear();model.options.update(options)
                chunks[mode].append(pool(hidden).cpu().numpy())
            assert model.options==options
        for mode in chunks:
            label=name+'_'+mode;destination=out/label;destination.mkdir()
            x=np.concatenate(chunks[mode]);np.save(destination/'features.npy',x)
            metrics=run_gdp(df,torch.from_numpy(x.astype(np.float64)),SimpleNamespace(out_dir=destination))
            write(destination/'metrics.json',metrics);results[label]=metrics;write(out/'metrics.json',results)
            checks[label]=[]
            for prop in ALL_COLS:
                saved=np.load(destination/prop/'legacy_oof.npz');valid=df[prop].notna().to_numpy()
                np.testing.assert_array_equal(saved['row_index'],np.flatnonzero(valid));np.testing.assert_array_equal(saved['fold'],df.loc[valid,FOLD_COL]);np.testing.assert_array_equal(saved['y_true'],df.loc[valid,prop])
                values=[float(spearmanr(saved['y_transformed'][saved['fold']==i],saved['prediction'][saved['fold']==i]).statistic) for i in range(5)]
                assert abs(np.mean(values)-metrics['properties'][prop]['legacy_reference']['cv_mean_spearman'])<1e-12
                checks[label].append(dict(property=prop,fold_scores=values))
            print(label,{p:v['legacy_reference']['cv_mean_spearman'] for p,v in metrics['properties'].items()},flush=True)
        del model,backbone,hidden,original
        torch.cuda.empty_cache();status.update(completed=name);write(out/'status.json',status)
    for path,digest in plan['code_sha256'].items():assert sha(Path(path))==digest
    write(out/'verification.json',dict(status='passed',native_features_bitwise_reproduced=True,options_restored=True,topology=topology,oof_checks=checks))
    status.update(status='success',finished_utc=datetime.now(timezone.utc).isoformat());write(out/'status.json',status)


if __name__=='__main__':main()
