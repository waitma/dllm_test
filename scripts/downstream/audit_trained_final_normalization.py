"""Compare actual block8 output with post-RMSNorm features after training.

Activate pllm/protenix_abtcr; run locally from the repository root:
python -B scripts/downstream/audit_trained_final_normalization.py --out ABS_NEW_DIR
All eight blocks execute normally. No projection, skipped block, or update.
Post-norm controls reproduce saved features; pre-norm scores are labeled apart.
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
          'aligned1700':ROOT/'output/diagnostics/final_layer_attention_alignment_20261018/nomask_continue',
          'chain1700':ROOT/'output/diagnostics/final_layer_chain_preservation_20261018/chain_contrast'}
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),arms={k:str(v) for k,v in arms.items()},
              modes=['native','no_mask_keys'],readouts=['block8_pre_norm','post_norm_control'],
              scope='Actual full block8 output, before vs after final RMSNorm; exploratory same-CV readout comparison, not new training',
              source_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'downstream/final_layer_preservation_eval.py',ROOT/'downstream/grammar/ab_probes.py']})
    write(out/'plan.json',plan)
    status=dict(status='running',stage='features',pid=os.getpid(),started_utc=plan['created_utc']);write(out/'status.json',status)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from downstream.final_layer_preservation_eval import load
    from downstream.fixed_chain_eval import FixedChainBackbone
    from downstream.fixed_chain_layer_eval import block_hidden
    torch.set_num_threads(4)
    results,checks,geometry={},{},{}
    for name,folder in arms.items():
        model,identity=load(folder)
        from dllm.pipelines.immune_llada.data.records import BioSeqRecord,BioSeqChain
        from downstream.grammar.ab_probes import prepare_task,run_gdp
        from downstream.ophiuchus_eval.developability import ALL_COLS,FOLD_COL
        df,pairs,_,_=prepare_task('gdp_a1')
        records=[BioSeqRecord([BioSeqChain(h,'antibody_heavy'),BioSeqChain(l,'antibody_light')],task_type='antibody',source='native_v5_probe') for h,l in pairs]
        backbone=FixedChainBackbone(model,identity);norm=model.backbone.transformer.ln_f
        if name=='spectral1000':
            controls={mode:np.load(ROOT/f'output/diagnostics/final_layer_mask_interaction_20261018/spectral1000_{mode}/features.npy') for mode in plan['modes']}
            old_metrics={mode:json.loads((ROOT/f'output/diagnostics/final_layer_mask_interaction_20261018/spectral1000_{mode}/metrics.json').read_text()) for mode in plan['modes']}
        else:
            controls={'native':np.load(folder/'final_features.npy'),'no_mask_keys':np.load(folder/'final_no_mask_features.npy')}
            old_metrics={'native':json.loads((folder/'gdp_a1/metrics.json').read_text()),'no_mask_keys':json.loads((folder/'gdp_no_mask_keys/metrics.json').read_text())}
        pre={mode:[] for mode in plan['modes']};traces={mode:[] for mode in plan['modes']}
        for start in range(0,len(records),8):
            batch=backbone.collator(records[start:start+8]);batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
            mask=batch['residue_mask'] & batch['attention_mask']
            def pool(h):return h.float().masked_fill(~mask[...,None],0).sum(1)/mask.sum(1,keepdim=True)
            for mode in plan['modes']:
                current=dict(batch)
                if mode=='no_mask_keys':current['attention_mask']=batch['attention_mask'] & batch['input_ids'].ne(model.tokenizer.mask_token_id)
                captured=[]
                hook=model.backbone.transformer.blocks[7].register_forward_hook(lambda mod,inp,result:captured.append(block_hidden(result).detach()))
                try:final,_=backbone._final_hidden(current)
                finally:hook.remove()
                assert len(captured)==1
                raw=captured[0]
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):reconstructed=norm(raw)
                torch.testing.assert_close(reconstructed,final,rtol=0,atol=0)
                np.testing.assert_array_equal(pool(final).cpu().numpy(),controls[mode][start:start+8])
                pre[mode].append(pool(raw).cpu().numpy())
                rms=raw.float().square().mean(-1).sqrt()
                for i in range(len(raw)):
                    values=rms[i][mask[i]]
                    traces[mode].append(dict(row=start+i,mean_token_rms=float(values.mean()),token_rms_cv=float(values.std()/values.mean()),pre_pool_norm=float(pool(raw)[i].norm()),post_pool_norm=float(pool(final)[i].norm())))
        for mode in plan['modes']:
            label=name+'_'+mode;destination=out/label;destination.mkdir()
            x=np.concatenate(pre[mode]);np.save(destination/'pre_norm_features.npy',x)
            result=run_gdp(df,torch.from_numpy(x.astype(np.float64)),SimpleNamespace(out_dir=destination))
            write(destination/'metrics.json',result);results[label]=dict(pre_norm=result,post_norm=old_metrics[mode]);write(out/'metrics.json',results)
            checks[label]=[]
            for prop in ALL_COLS:
                saved=np.load(destination/prop/'legacy_oof.npz');valid=df[prop].notna().to_numpy()
                np.testing.assert_array_equal(saved['row_index'],np.flatnonzero(valid));np.testing.assert_array_equal(saved['fold'],df.loc[valid,FOLD_COL]);np.testing.assert_array_equal(saved['y_true'],df.loc[valid,prop])
                scores=[float(spearmanr(saved['y_transformed'][saved['fold']==i],saved['prediction'][saved['fold']==i]).statistic) for i in range(5)]
                assert abs(np.mean(scores)-result['properties'][prop]['legacy_reference']['cv_mean_spearman'])<1e-12
                checks[label].append(dict(property=prop,fold_scores=scores))
            geometry[label]=traces[mode]
            print(label,{p:v['legacy_reference']['cv_mean_spearman'] for p,v in result['properties'].items()},flush=True)
        del model,backbone,norm,raw,final,reconstructed,captured
        torch.cuda.empty_cache();status.update(completed=name);write(out/'status.json',status)
    for path,digest in plan['source_sha256'].items():assert sha(Path(path))==digest
    write(out/'token_geometry.json',geometry)
    write(out/'verification.json',dict(status='passed',post_norm_features_bitwise_reproduced=True,final_norm_reconstruction_exact=True,oof_checks=checks))
    status.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat());write(out/'status.json',status)


if __name__=='__main__':main()
