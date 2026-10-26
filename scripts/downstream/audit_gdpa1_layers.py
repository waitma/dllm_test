"""Frozen LoRA layer readout diagnostic, CPU only; no backbone updates.

Activate pllm then protenix_abtcr; from repository root run:
python -B scripts/downstream/audit_gdpa1_layers.py --out ABS_NEW_DIRECTORY
Report every predeclared readout and all five properties. This reuses inspected
CV folds and is diagnostic, not an independent test or a new default readout.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    run = ROOT/'output/downstream_generation/post_training_lora_s42_r1_20261014'
    control_path = ROOT/'output/diagnostics/gdpa1_airgen_20261017/lora3000/features.npz'
    readouts = ['embedding', 'block2', 'block4', 'block6', 'block8', 'final_norm']
    sources = [Path(__file__), ROOT/'downstream/grammar/ab_probes.py',
               ROOT/'downstream/fixed_chain_eval.py', ROOT/'downstream/post_training_eval.py',
               ROOT/'data/downstream/dev/GDPa1_v1.2_20250814.csv', control_path]
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(run/'checkpoint'),
                readouts=readouts, pooling='global observed-residue mean at every layer',
                protocol='unchanged AirGen reference Ridge grid and five folds; all five properties',
                device='cpu',precision='float32',batch_size=8,threads=8,
                selection='none; diagnostic only; do not replace headline or select a layer on test',
                sources={str(f):sha(f) for f in sources})
    write(out/'plan.json', plan)
    import numpy as np
    import torch
    from downstream.post_training_eval import load, PostTrainingBackbone
    torch.set_num_threads(8)
    model, identity = load(run/'checkpoint', device='cpu')
    write(out/'identity.json', identity)
    backbone = PostTrainingBackbone(model, identity)
    from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
    from downstream.grammar.ab_probes import prepare_task, run_gdp
    df, pairs, _, _ = prepare_task('gdp_a1')
    records = [BioSeqRecord([BioSeqChain(h,'antibody_heavy'),BioSeqChain(l,'antibody_light')],
                           task_type='antibody',source='native_v5_probe') for h,l in pairs]
    assert len(model.backbone.transformer.blocks) == 8
    chunks = {k:[] for k in readouts}
    pending = {}
    mask = None
    def capture(name):
        def hook(module, inputs, hidden):
            assert name not in pending and hidden.ndim == 3
            pooled = hidden.float().masked_fill(~mask[...,None],0).sum(1)/mask.sum(1,keepdim=True)
            assert torch.isfinite(pooled).all()
            pending[name] = pooled.cpu().clone()
        return hook
    handles = [model.backbone.transformer.emb_drop.register_forward_hook(capture('embedding'))]
    handles += [model.backbone.transformer.blocks[i-1].register_forward_hook(capture(f'block{i}')) for i in (2,4,6,8)]
    try:
        for start in range(0,len(records),8):
            batch = backbone.collator(records[start:start+8])
            mask = batch['residue_mask'] & batch['attention_mask']
            assert mask.sum(1).tolist() == [len(h)+len(l) for h,l in pairs[start:start+8]]
            hidden, _ = backbone._final_hidden(batch)
            pending['final_norm'] = (hidden.float().masked_fill(~mask[...,None],0).sum(1)/mask.sum(1,keepdim=True)).cpu()
            assert set(pending) == set(readouts)
            for name in readouts:
                chunks[name].append(pending[name])
            pending.clear()
            print('features',min(start+8,len(records)),len(records),flush=True)
    finally:
        for handle in handles:
            handle.remove()
    xs = {k:torch.cat(v) for k,v in chunks.items()}
    control = np.load(control_path)['global']
    torch.testing.assert_close(xs['final_norm'],torch.from_numpy(control),atol=0,rtol=0)
    np.savez_compressed(out/'features.npz', **{k:v.numpy() for k,v in xs.items()})
    stats, results = {}, {}
    for name, x in xs.items():
        values = x.numpy()
        singular = np.linalg.svd(values-values.mean(0),compute_uv=False)
        energy = singular**2
        stats[name] = dict(shape=list(x.shape),mean_norm=float(x.norm(dim=1).mean()),
                           top_pc_variance_fraction=float(energy[0]/energy.sum()),
                           effective_rank=float(np.exp(-np.sum((energy/energy.sum())*np.log(np.maximum(energy/energy.sum(),1e-30))))))
        folder = out/name
        folder.mkdir()
        metrics = run_gdp(df,x,SimpleNamespace(out_dir=folder))
        write(folder/'metrics.json',metrics)
        results[name] = metrics
        write(out/'metrics.json',results)
    # Independently rescore the saved out-of-fold predictions at each selected alpha.
    from scipy.stats import spearmanr
    from downstream.ophiuchus_eval.developability import ALL_COLS, FOLD_COL
    checks = []
    for name, result in results.items():
        for col in ALL_COLS:
            saved = np.load(out/name/col/'legacy_oof.npz')
            valid = df[col].notna().to_numpy()
            assert np.array_equal(saved['row_index'], np.flatnonzero(valid))
            assert np.array_equal(saved['fold'],df.loc[valid,FOLD_COL].to_numpy())
            assert np.array_equal(saved['y_true'],df.loc[valid,col].to_numpy())
            folds = [float(spearmanr(saved['y_transformed'][saved['fold']==i], saved['prediction'][saved['fold']==i]).statistic) for i in range(5)]
            score = float(np.mean(folds))
            assert abs(score-result['properties'][col]['legacy_reference']['cv_mean_spearman']) < 1e-12
            checks.append(dict(readout=name,property=col,fold_scores=folds,score=score))
    for path,digest in plan['sources'].items():
        assert sha(path)==digest,path
    write(out/'verification.json',dict(status='passed',final_norm_control_bitwise_equal=True,
                                      independent_oof_checks=checks,feature_statistics=stats))
    print('COMPLETE',out,flush=True)


if __name__ == '__main__':
    main()
