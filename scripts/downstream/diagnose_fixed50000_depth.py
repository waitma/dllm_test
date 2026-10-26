"""Local frozen depth/topology diagnostic; no checkpoint or production edits.

Activate pllm then protenix_abtcr, from repository root run:
python -B scripts/downstream/diagnose_fixed50000_depth.py --out ABS_NEW_DIRECTORY
Uses GPU 0 for features and four CPU threads for the existing reference Ridge.
All interventions and layers are reported; this is exploratory CV, not a test.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from downstream.fixed_chain_eval import load, FixedChainBackbone, sha
    from downstream.fixed_chain_layer_eval import block_hidden
    torch.set_num_threads(4)
    baseline = ROOT/'output/downstream_generation/fixed50000_gdpa1_pooling_20261018'
    checkpoint = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint'
    names = ['native_embedding']+[f'native_block{i}' for i in range(1, 9)]+['native_final_norm']
    names += [f'{mode}_{layer}' for mode in ('no_mask_keys', 'all_global')
              for layer in ('block2', 'block8', 'final_norm')]
    names += ['native_block2_unit_rms', 'native_block8_unit_rms', 'aa_composition', 'chain_lengths']
    sources = [Path(__file__), ROOT/'downstream/fixed_chain_eval.py',
               ROOT/'downstream/fixed_chain_layer_eval.py', ROOT/'downstream/grammar/ab_probes.py',
               ROOT/'data/downstream/dev/GDPa1_v1.2_20250814.csv', baseline/'features.npz']
    write(out/'plan.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), readouts=names,
          checkpoint=str(checkpoint), precision='native BF16 autocast, FP32 pool; float64 Ridge',
          pooling='observed H/L residue global mean',
          interventions=dict(no_mask_keys='Only attention_mask excludes MASK; IDs/positions unchanged',
                             all_global='Only segment_global=False, restores original after each forward'),
          protocol='original reference Ridge grid, all-label transform, predefined five folds',
          interpretation='exploratory; inference topology interventions are distribution shifts, not retrained models',
          source_sha256={str(p):sha(p) for p in sources}))
    write(out/'status.json', dict(status='running', stage='features'))
    model, identity = load(checkpoint, device='cuda')
    write(out/'identity.json', identity)
    backbone = FixedChainBackbone(model, identity)
    from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
    from downstream.grammar.ab_probes import prepare_task, run_gdp
    from downstream.ophiuchus_eval.developability import ALL_COLS, FOLD_COL
    df, pairs, _, _ = prepare_task('gdp_a1')
    records = [BioSeqRecord([BioSeqChain(h,'antibody_heavy'), BioSeqChain(l,'antibody_light')],
                           task_type='antibody', source='native_v5_probe') for h,l in pairs]
    chunks = {name:[] for name in names if name not in ('aa_composition', 'chain_lengths')}
    topology, token_stats = [], {f'block{i}':[] for i in range(1,9)}
    original_options = dict(model.options)
    norm = model.backbone.transformer.ln_f
    assert norm.__class__.__name__ == 'RMSLayerNorm' and norm.bias is None
    controls = np.load(baseline/'features.npz')
    for start in range(0, len(records), 8):
        batch = backbone.collator(records[start:start+8])
        batch = {k:v.to('cuda') if torch.is_tensor(v) else v for k,v in batch.items()}
        residue = batch['residue_mask'].bool() & batch['attention_mask'].bool()
        assert residue.sum(1).tolist() == [len(h)+len(l) for h,l in pairs[start:start+8]]
        def pool(hidden):
            return hidden.float().masked_fill(~residue[...,None], 0).sum(1)/residue.sum(1,keepdim=True)
        for mode in ('native', 'no_mask_keys', 'all_global'):
            current = dict(batch)
            if mode == 'no_mask_keys':
                current['attention_mask'] = batch['attention_mask'] & batch['input_ids'].ne(model.tokenizer.mask_token_id)
                assert not (residue & ~current['attention_mask']).any()
            if mode == 'all_global':
                model.options['segment_global'] = False
            if start == 0:
                for i in range(8):
                    allowed = model.layer_allowed(i, current['chain_ids'], current['position_ids'], current['attention_mask'])
                    different = current['chain_ids'][:,:,None].ne(current['chain_ids'][:,None,:])
                    rr = residue[:,:,None] & residue[:,None,:] & different
                    tail = batch['input_ids'].eq(model.tokenizer.mask_token_id)[:,None,:] & residue[:,:,None]
                    topology.append(dict(mode=mode, block=i+1, key=list(model.mask_key(i)),
                                         cross_chain_residue_edges=int((allowed & rr).sum()),
                                         residue_to_mask_edges=int((allowed & tail).sum())))
            captured = {}
            def capture(name):
                def hook(module, inputs, output):
                    hidden = block_hidden(output) if isinstance(output, tuple) else output
                    captured[name] = pool(hidden).cpu()
                    if mode == 'native' and name.startswith('block'):
                        h = hidden.float()
                        rms = h.pow(2).mean(-1).sqrt()
                        vals = rms[residue]
                        token_stats[name].append(dict(n=len(vals), mean=float(vals.mean()),
                                                     std=float(vals.std()), maximum=float(vals.max())))
                        if name in ('block2','block8'):
                            unit = (h*torch.rsqrt(h.pow(2).mean(-1,keepdim=True)+norm.eps)).to(hidden.dtype)
                            captured[name+'_unit_rms'] = pool(unit).cpu()
                            if name == 'block8':
                                reconstructed = unit*norm.weight
                                captured['norm_reconstructed'] = pool(reconstructed).cpu()
                return hook
            indices = range(8) if mode == 'native' else (1,7)
            hooks = [model.backbone.transformer.blocks[i].register_forward_hook(capture(f'block{i+1}')) for i in indices]
            if mode == 'native':
                hooks.append(model.backbone.transformer.emb_drop.register_forward_hook(capture('embedding')))
            try:
                hidden, _ = backbone._final_hidden(current)
            finally:
                for h in hooks:
                    h.remove()
                model.options.clear()
                model.options.update(original_options)
            captured['final_norm'] = pool(hidden).cpu()
            if mode == 'native':
                torch.testing.assert_close(captured.pop('norm_reconstructed'), captured['final_norm'], rtol=0, atol=0)
                for key, old in [('block2','block2_residue_global'), ('final_norm','final_norm_residue_global')]:
                    np.testing.assert_array_equal(captured[key].numpy(), controls[old][start:start+8])
            for key,value in captured.items():
                assert torch.isfinite(value).all()
                chunks[f'{mode}_{key}'].append(value)
        print('features', min(start+8,len(records)), len(records), flush=True)
    xs = {k:torch.cat(v).numpy() for k,v in chunks.items()}
    alphabet = 'LAGVSERTIDPKQNFYMHWCXBUZO'
    xs['aa_composition'] = np.array([[((h+l).count(a)/len(h+l)) for a in alphabet] for h,l in pairs], dtype=np.float64)
    xs['chain_lengths'] = np.array([[len(h),len(l)] for h,l in pairs], dtype=np.float64)
    assert set(xs) == set(names)
    np.savez_compressed(out/'features.npz', **xs)
    geometry = {}
    teacher = xs['native_block2'].astype(np.float64)
    teacher -= teacher.mean(0)
    kt = teacher@teacher.T
    for name,x in xs.items():
        x = x.astype(np.float64)
        centered = x-x.mean(0)
        energy = np.linalg.svd(centered, compute_uv=False)**2
        p = energy/energy.sum()
        unit = x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-30)
        k = centered@centered.T
        geometry[name] = dict(shape=list(x.shape), mean_norm=float(np.linalg.norm(x,axis=1).mean()),
              top_pc_fraction=float(p[0]), effective_rank=float(np.exp(-np.sum(p*np.log(np.maximum(p,1e-30))))),
              mean_pairwise_cosine=float(((unit@unit.T).sum()-len(x))/(len(x)*(len(x)-1))),
              centered_linear_cka_to_block2=float(np.sum(k*kt)/np.sqrt(np.sum(k*k)*np.sum(kt*kt))))
    write(out/'architecture.json',dict(options=original_options, topology_first_batch=topology,
          norm_type=norm.__class__.__name__,norm_epsilon=norm.eps,
          norm_weight_range=[float(norm.weight.min()),float(norm.weight.max())],
          token_rms_by_batch=token_stats,feature_geometry=geometry,
          controls='native block2 and final features bitwise identical to previous pooling experiment; RMS reconstruction exact'))
    del model, backbone, hidden, batch, current
    torch.cuda.empty_cache()
    results, checks = {}, []
    write(out/'status.json', dict(status='running',stage='ridge'))
    for name in names:
        folder = out/name
        folder.mkdir()
        result = run_gdp(df, torch.from_numpy(xs[name].astype(np.float64)), SimpleNamespace(out_dir=folder))
        write(folder/'metrics.json', result)
        results[name] = result
        write(out/'metrics.json', results)
        for col in ALL_COLS:
            saved = np.load(folder/col/'legacy_oof.npz')
            valid = df[col].notna().to_numpy()
            np.testing.assert_array_equal(saved['row_index'],np.flatnonzero(valid))
            np.testing.assert_array_equal(saved['fold'],df.loc[valid,FOLD_COL].to_numpy())
            np.testing.assert_array_equal(saved['y_true'],df.loc[valid,col].to_numpy())
            scores = [float(spearmanr(saved['y_transformed'][saved['fold']==i],saved['prediction'][saved['fold']==i]).statistic) for i in range(5)]
            assert abs(np.mean(scores)-result['properties'][col]['legacy_reference']['cv_mean_spearman'])<1e-12
            checks.append(dict(readout=name, property=col,fold_scores=scores))
        print('finished',name,flush=True)
    plan=json.loads((out/'plan.json').read_text())
    for path,digest in plan['source_sha256'].items():
        assert sha(path)==digest,path
    write(out/'verification.json',dict(status='passed',oof_checks=checks,feature_sha256=sha(out/'features.npz')))
    write(out/'status.json',dict(status='success',finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
