"""Measure all native layers of a verified adapted model under fixed GDP probes.

Activate pllm/protenix_abtcr; run from repository root:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B scripts/downstream/audit_final_candidate_depth.py --arm ABS_COMPLETED_ARM --out ABS_NEW_DIR
Reports all eight blocks and final RMSNorm without choosing a replacement layer.
Original blocks1/2 and the candidate's saved final features must match bitwise.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    arm, out = args.arm.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', stage='prepare', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json', state)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    torch.set_num_threads(4)
    original = ROOT/'output/diagnostics/fixed50000_depth_20261018'
    assert json.loads((arm.parent/'status.json').read_text())['status'] == 'success'
    verified = json.loads((arm/'verification.json').read_text())
    assert verified['status'] == 'passed' and verified['reloaded_features_bitwise_equal']
    assert json.loads((original/'verification.json').read_text())['status'] == 'passed'
    from downstream.final_layer_preservation_eval import load
    from downstream.fixed_chain_eval import FixedChainBackbone
    from downstream.fixed_chain_layer_eval import block_hidden
    model, identity = load(arm)
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
    from downstream.grammar.ab_probes import prepare_task, run_gdp
    from downstream.ophiuchus_eval.developability import ALL_COLS, FOLD_COL
    backbone = FixedChainBackbone(model, identity, batch_size=8)
    df, pairs, _, _ = prepare_task('gdp_a1')
    records = [BioSeqRecord([BioSeqChain(h,'antibody_heavy'),BioSeqChain(l,'antibody_light')],task_type='antibody',source='native_v5_probe') for h,l in pairs]
    names = [f'native_block{i}' for i in range(1,9)]+['native_final_norm']
    sources = [Path(__file__), ROOT/'downstream/fixed_chain_eval.py', ROOT/'downstream/fixed_chain_layer_eval.py',
        ROOT/'downstream/final_layer_preservation_eval.py', ROOT/'downstream/grammar/ab_probes.py',
        ROOT/'data/downstream/dev/GDPa1_v1.2_20250814.csv', original/'features.npz', original/'metrics.json',
        arm/'checkpoint.json', arm/'final_features.npy']
    hashes = {str(p):sha(p) for p in sources}
    write(out/'plan.json', dict(created_utc=state['started_utc'], arm=str(arm), model_identity=identity,
        layers=names, batch_size=8, attention='native', precision='BF16 forward, FP32 pool, float64 Ridge',
        pooling='observed H/L residue global mean', protocol='same fixed GDP reference folds and Ridge grid',
        limitations='Exploratory same-CV alpha selection, single adapted checkpoint. No layer replacement or causal attribution to an individual block.',
        source_sha256=hashes))
    chunks = {name:[] for name in names}
    for start in range(0,len(records),8):
        batch = backbone.collator(records[start:start+8])
        batch = {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        mask = batch['residue_mask'] & batch['attention_mask']
        assert mask.sum(1).tolist() == [len(h)+len(l) for h,l in pairs[start:start+8]]
        def pool(h):
            return (h.float().masked_fill(~mask[...,None],0).sum(1)/mask.sum(1,keepdim=True)).detach().cpu().numpy()
        def capture(name):
            def hook(module, inputs, output):
                chunks[name].append(pool(block_hidden(output)))
            return hook
        hooks = [block.register_forward_hook(capture(f'native_block{i+1}')) for i,block in enumerate(model.backbone.transformer.blocks)]
        try:
            hidden, _ = backbone._final_hidden(batch)
        finally:
            for hook in hooks:
                hook.remove()
        chunks['native_final_norm'].append(pool(hidden))
    features = {name:np.concatenate(values) for name,values in chunks.items()}
    baseline = np.load(original/'features.npz')
    for name in ['native_block1','native_block2']:
        np.testing.assert_array_equal(features[name],baseline[name])
    np.testing.assert_array_equal(features['native_final_norm'],np.load(arm/'final_features.npy'))
    np.savez_compressed(out/'features.npz',**features)
    teacher = baseline['native_block2'].astype(np.float64)
    teacher -= teacher.mean(0);kt = teacher@teacher.T
    geometry = {}
    for name,x in features.items():
        x = x.astype(np.float64);x -= x.mean(0)
        energy = np.linalg.svd(x,compute_uv=False)**2
        fractions = energy/energy.sum();kx = x@x.T
        geometry[name] = dict(effective_rank=float(np.exp(-np.sum(fractions*np.log(np.maximum(fractions,1e-30))))),
            top_pc_fraction=float(fractions[0]),centered_cka_to_original_block2=float(np.sum(kx*kt)/np.sqrt(np.sum(kx*kx)*np.sum(kt*kt))))
    write(out/'geometry.json',geometry)
    del model, backbone, hidden, batch
    torch.cuda.empty_cache()
    results, checks = {}, []
    for name in names:
        state.update(stage='ridge',layer=name);write(out/'status.json',state)
        folder = out/name;folder.mkdir()
        metrics = run_gdp(df,torch.from_numpy(features[name].astype(np.float64)),SimpleNamespace(out_dir=folder))
        write(folder/'metrics.json',metrics);results[name] = metrics
        for prop in ALL_COLS:
            saved = np.load(folder/prop/'legacy_oof.npz')
            old = np.load(original/name/prop/'legacy_oof.npz')
            valid = df[prop].notna().to_numpy()
            np.testing.assert_array_equal(saved['row_index'],np.flatnonzero(valid))
            np.testing.assert_array_equal(saved['fold'],df.loc[valid,FOLD_COL])
            for key in ['row_index','fold','y_true','y_transformed']:
                np.testing.assert_array_equal(saved[key],old[key])
            scores = [float(spearmanr(saved['y_transformed'][saved['fold']==k],saved['prediction'][saved['fold']==k]).statistic) for k in range(5)]
            assert abs(np.mean(scores)-metrics['properties'][prop]['legacy_reference']['cv_mean_spearman']) < 1e-12
            if name in ['native_block1','native_block2']:
                np.testing.assert_array_equal(saved['prediction'],old['prediction'])
            if name == 'native_final_norm':
                previous = np.load(arm/'gdp_a1'/prop/'legacy_oof.npz')
                np.testing.assert_array_equal(saved['prediction'],previous['prediction'])
            checks.append(dict(layer=name,property=prop,fold_scores=scores))
        write(out/'metrics.json',results)
        print(name,{p:v['legacy_reference']['cv_mean_spearman'] for p,v in metrics['properties'].items()},flush=True)
    for path,digest in hashes.items():
        assert sha(Path(path)) == digest,path
    write(out/'verification.json',dict(status='passed',source_sha256=hashes,oof_checks=checks,
        frozen_first_two_layers_bitwise_equal=True,final_features_and_oof_bitwise_equal=True))
    state.update(status='success',stage='complete',finished_utc=datetime.now(timezone.utc).isoformat());write(out/'status.json',state)


if __name__ == '__main__':
    main()
