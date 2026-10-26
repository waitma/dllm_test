"""Complete the frozen topology x MASK-key experiment and trace block eight.

Activate pllm then protenix_abtcr; from repository root:
python -B scripts/downstream/diagnose_fixed50000_block8.py --out ABS_NEW_DIRECTORY
Local CUDA features, float64 reference Ridge. No weights or source are modified.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.downstream.diagnose_fixed50000_depth import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from downstream.fixed_chain_eval import load, FixedChainBackbone, sha
    from downstream.fixed_chain_layer_eval import block_hidden
    torch.set_num_threads(4)
    parent=ROOT/'output/diagnostics/fixed50000_depth_20261018'
    checkpoint=ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint'
    modes=['native','no_mask_keys','all_global','global_no_mask_keys']
    names=[f'{mode}_{point}' for mode in modes for point in ('block8_post_attention','block8_post_attention_norm')]
    names += ['global_no_mask_keys_'+point for point in ('block2','block8','final_norm')]
    sources=[Path(__file__),ROOT/'scripts/downstream/diagnose_fixed50000_depth.py',
             parent/'features.npz', ROOT/'downstream/fixed_chain_eval.py',
             ROOT/'downstream/grammar/ab_probes.py', ROOT/'data/downstream/dev/GDPa1_v1.2_20250814.csv']
    write(out/'plan.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),readouts=names,
          parent=str(parent),checkpoint=str(checkpoint),
          design='complete 2x2: original vs full global; MASK keys attended vs excluded',
          sublayer='block8 post-attention captured at ff_norm input; norm uses original final RMSNorm',
          interpretation='frozen inference interventions, exploratory reference CV; post-attention skips FFN8 and is not a full final layer',
          source_sha256={str(p):sha(p) for p in sources}))
    write(out/'status.json',dict(status='running',stage='features'))
    model,identity=load(checkpoint,device='cuda')
    write(out/'identity.json',identity)
    backbone=FixedChainBackbone(model,identity)
    from dllm.pipelines.immune_llada.data.records import BioSeqChain,BioSeqRecord
    from downstream.grammar.ab_probes import prepare_task,run_gdp
    from downstream.ophiuchus_eval.developability import ALL_COLS,FOLD_COL
    df,pairs,_,_=prepare_task('gdp_a1')
    records=[BioSeqRecord([BioSeqChain(h,'antibody_heavy'),BioSeqChain(l,'antibody_light')],
                         task_type='antibody',source='native_v5_probe') for h,l in pairs]
    old=np.load(parent/'features.npz')
    chunks={n:[] for n in names}
    traces=[]
    original_options=dict(model.options)
    wrapped=model.backbone.transformer.blocks[7]
    norm=model.backbone.transformer.ln_f
    for start in range(0,len(records),8):
        batch=backbone.collator(records[start:start+8])
        batch={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in batch.items()}
        mask=batch['residue_mask'] & batch['attention_mask']
        def pool(h):
            return h.float().masked_fill(~mask[...,None],0).sum(1)/mask.sum(1,keepdim=True)
        for mode in modes:
            current=dict(batch)
            if mode in ('no_mask_keys','global_no_mask_keys'):
                current['attention_mask']=batch['attention_mask'] & batch['input_ids'].ne(model.tokenizer.mask_token_id)
            if mode in ('all_global','global_no_mask_keys'):
                model.options['segment_global']=False
            captured={}
            def capture(name):
                def hook(module,inputs,output):
                    captured[name]=block_hidden(output).detach().clone()
                return hook
            def before_ff(module,inputs):
                captured['post_attention']=inputs[0].detach().clone()
            hooks=[model.backbone.transformer.blocks[i-1].register_forward_hook(capture(f'block{i}')) for i in (2,7,8)]
            hooks.append(wrapped.base.ff_norm.register_forward_pre_hook(before_ff))
            try:
                final,_=backbone._final_hidden(current)
            finally:
                for h in hooks:h.remove()
                model.options.clear();model.options.update(original_options)
            captured['final_norm']=final
            if mode!='global_no_mask_keys':
                for point in ('block2','block8','final_norm'):
                    np.testing.assert_array_equal(pool(captured[point]).cpu().numpy(),old[f'{mode}_{point}'][start:start+8])
            else:
                for point in ('block2','block8','final_norm'):
                    chunks[f'{mode}_{point}'].append(pool(captured[point]).cpu())
            post=captured['post_attention']
            chunks[f'{mode}_block8_post_attention'].append(pool(post).cpu())
            with torch.inference_mode():
                post_norm=norm(post)
            chunks[f'{mode}_block8_post_attention_norm'].append(pool(post_norm).cpu())
            h7=captured['block7'].float()
            attn_delta=post.float()-h7
            ff_delta=captured['block8'].float()-post.float()
            # Per-antibody raw and pooled magnitudes, including cancellation.
            for i in range(len(h7)):
                row=dict(row=start+i,mode=mode)
                for name,h in [('block7',h7),('attention_delta',attn_delta),('post_attention',post),
                               ('ffn_delta',ff_delta),('block8',captured['block8'])]:
                    row[name+'_pooled_norm']=float(pool(h)[i].norm())
                    row[name+'_mean_token_norm']=float(h[i][mask[i]].float().norm(dim=-1).mean())
                traces.append(row)
        print('features',min(start+8,len(records)),len(records),flush=True)
    xs={n:torch.cat(v).numpy() for n,v in chunks.items()}
    assert all(np.isfinite(x).all() for x in xs.values())
    np.savez_compressed(out/'features.npz',**xs)
    write(out/'sublayer_trace.json',dict(rows=traces,norm_epsilon=norm.eps,
          parent_controls='all 9 existing feature matrices bitwise reproduced',
          options_restored=model.options==original_options))
    del model,backbone,wrapped,norm,captured,final,post,post_norm,h7,attn_delta,ff_delta,batch,current,h
    torch.cuda.empty_cache()
    results,checks={},[]
    write(out/'status.json',dict(status='running',stage='ridge'))
    for name in names:
        folder=out/name;folder.mkdir()
        result=run_gdp(df,torch.from_numpy(xs[name].astype(np.float64)),SimpleNamespace(out_dir=folder))
        write(folder/'metrics.json',result)
        results[name]=result;write(out/'metrics.json',results)
        for col in ALL_COLS:
            saved=np.load(folder/col/'legacy_oof.npz')
            valid=df[col].notna().to_numpy()
            np.testing.assert_array_equal(saved['row_index'],np.flatnonzero(valid))
            np.testing.assert_array_equal(saved['fold'],df.loc[valid,FOLD_COL].to_numpy())
            np.testing.assert_array_equal(saved['y_true'],df.loc[valid,col].to_numpy())
            scores=[float(spearmanr(saved['y_transformed'][saved['fold']==i],saved['prediction'][saved['fold']==i]).statistic) for i in range(5)]
            assert abs(np.mean(scores)-result['properties'][col]['legacy_reference']['cv_mean_spearman'])<1e-12
            checks.append(dict(readout=name,property=col,fold_scores=scores))
        print('finished',name,flush=True)
    plan=json.loads((out/'plan.json').read_text())
    for p,digest in plan['source_sha256'].items():assert sha(p)==digest,p
    write(out/'verification.json',dict(status='passed',oof_checks=checks,feature_sha256=sha(out/'features.npz')))
    write(out/'status.json',dict(status='success',finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':main()
