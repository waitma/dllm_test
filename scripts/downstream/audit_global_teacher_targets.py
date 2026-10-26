"""Test OAS-only PCA teacher targets combining early and all-global features.

Activate pllm then protenix_abtcr; from the repository root:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B \
  scripts/downstream/audit_global_teacher_targets.py --out ABS_NEW_DIRECTORY
Extract frozen original-parent all-global features on the fixed4096 OAS train
rows. Fit PCA/scale/rotation on OAS only. Evaluate early512 control, global256,
and early256+global256 targets with the unchanged GDP reference probe. These
are teacher-design diagnostics, NOT adapted final-layer model results.
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
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', stage='prepare', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json', status)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from downstream.fixed_chain_eval import load, FixedChainBackbone
    torch.set_num_threads(4)
    root = ROOT/'output/diagnostics'
    sources = {}

    def pin(path):
        sources[str(path)] = sha(path)
        return path

    cal = np.load(pin(root/'final_layer_coordinate_pilot_20261018/calibration.npz'))
    prior_axes = np.load(pin(root/'multilayer_teacher_targets_20261018/oas_fitted_targets.npz'))
    depth = np.load(pin(root/'fixed50000_depth_20261018/features.npz'))
    data_root = root/'final_layer_preservation_pilot_20261018'
    manifest = json.loads(pin(data_root/'data.json').read_text())
    train_path = pin(data_root/'train.jsonl')
    assert sha(train_path) == manifest['files']['train']['sha256']
    rows = [json.loads(line) for line in train_path.read_text().splitlines()]
    assert len(rows) == len(cal['teacher']) == 4096
    checkpoint = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint'
    model, identity = load(checkpoint)
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
    from downstream.grammar.ab_probes import prepare_task, run_gdp
    from downstream.ophiuchus_eval.developability import ALL_COLS, FOLD_COL
    backbone = FixedChainBackbone(model, identity, batch_size=16)
    records = [BioSeqRecord.from_prepared_dict(row) for row in rows]
    options = dict(model.options)
    assert options['segment_global'] is True
    np.testing.assert_array_equal(backbone.vectors(records[:16]), cal['student'][:16])
    plan = dict(created_utc=status['started_utc'], parent_identity=identity,
                teacher='original fixed50000; segment_global=False for global attention in all eight blocks',
                variants={'early512': 'dimension control from original block2', 'global256': 'all-global final top256 raw PCs',
                          'early256_global256': 'native block2 top256 plus all-global final top256; equal branch energy'},
                calibration='PCA on OAS4096 only; no whitening; branch scalar normalization; thin OAS Procrustes into768 with original parent mean/RMS',
                selection='all three targets, all five properties; no GDP labels in PCA, scale or rotation',
                scope='teacher-design diagnostic, not an actual adapted backbone or deployable feature claim')
    write(out/'plan.json', plan)
    model.options['segment_global'] = False
    edges = []
    try:
        first = backbone.collator(records[:8])
        first = {k:v.cuda() if torch.is_tensor(v) else v for k,v in first.items()}
        mask = first['residue_mask'] & first['attention_mask']
        cross = mask[:, :, None] & mask[:, None, :] & first['chain_ids'][:, :, None].ne(first['chain_ids'][:, None, :])
        for i in range(8):
            allowed = model.layer_allowed(i, first['chain_ids'], first['position_ids'], first['attention_mask'])
            assert int((allowed & cross).sum()) == int(cross.sum()) > 0
            edges.append(int(cross.sum()))
        chunks = []
        for start in range(0, len(records), 256):
            chunks.append(backbone.vectors(records[start:start+256]))
            status.update(stage='oas_features', completed=start+len(chunks[-1]), total=len(records))
            write(out/'status.json', status)
            print('OAS', status['completed'], len(records), flush=True)
        global_train = np.concatenate(chunks).astype(np.float64)
        df, pairs, _, _ = prepare_task('gdp_a1')
        gdp_records = [BioSeqRecord([BioSeqChain(h,'antibody_heavy'), BioSeqChain(l,'antibody_light')],
                                    task_type='antibody', source='native_v5_probe') for h,l in pairs]
        backbone.batch_size = 8
        global_gdp = backbone.vectors(gdp_records).astype(np.float64)
        np.testing.assert_array_equal(global_gdp, depth['all_global_final_norm'])
    finally:
        model.options.clear()
        model.options.update(options)
    np.testing.assert_array_equal(backbone.vectors(gdp_records), depth['native_final_norm'])
    assert model.options == options
    np.save(out/'all_global_oas.npy', global_train)
    np.save(out/'all_global_gdp.npy', global_gdp)
    del model, backbone, first, allowed
    torch.cuda.empty_cache()
    global_mean = global_train.mean(0)
    _, singular, global_axes = np.linalg.svd(global_train-global_mean, full_matrices=False)
    np.savez_compressed(out/'global_pca.npz', mean=global_mean, axes=global_axes, eigenvalues=singular**2/len(global_train))
    branches = {
        'early': ((cal['teacher']-prior_axes['early_mean'])@prior_axes['early_axes'].T,
                  (depth['native_block2'].astype(np.float64)-prior_axes['early_mean'])@prior_axes['early_axes'].T),
        'global': ((global_train-global_mean)@global_axes.T, (global_gdp-global_mean)@global_axes.T)}
    variants = {'early512': [('early',512)], 'global256': [('global',256)],
                'early256_global256': [('early',256),('global',256)]}
    parent = (cal['student']-cal['mu_s'])/cal['rms_s']
    results, checks, geometry = {}, [], {}
    for name, parts in variants.items():
        train_parts, gdp_parts, scales = [], [], {}
        for branch, rank in parts:
            train, gdp = branches[branch]
            scale = float(np.sqrt(np.mean(train[:, :rank]**2)))
            scales[branch] = scale
            train_parts.append(train[:, :rank]/scale)
            gdp_parts.append(gdp[:, :rank]/scale)
        target_train, target_gdp = np.concatenate(train_parts, 1), np.concatenate(gdp_parts, 1)
        dim = target_train.shape[1]
        u, _, vh = np.linalg.svd(target_train.T@parent, full_matrices=False)
        rotation = u@vh
        scale = float(cal['rms_s']*np.sqrt(768/dim))
        np.testing.assert_allclose(rotation@rotation.T, np.eye(dim), atol=1e-12, rtol=1e-12)
        train = cal['mu_s']+target_train@rotation*scale
        gdp = cal['mu_s']+target_gdp@rotation*scale
        a, b = gdp-gdp.mean(0), (target_gdp-target_gdp.mean(0))*scale
        np.testing.assert_allclose(a@a.T, b@b.T, atol=1e-10, rtol=1e-10)
        assert abs(float(np.sqrt(np.mean((train-train.mean(0))**2)))-float(cal['rms_s'])) < 1e-10
        folder = out/name
        folder.mkdir()
        np.save(folder/'oas_targets.npy', train)
        np.save(folder/'gdp_targets.npy', gdp)
        np.savez_compressed(folder/'calibration.npz', rotation=rotation, scale=scale, mean=cal['mu_s'],
                            **{key+'_scale':value for key,value in scales.items()})
        if name == 'early512':
            old = root/'multilayer_teacher_raw_20261018/early512'
            np.testing.assert_allclose(train, np.load(pin(old/'oas_targets.npy')), atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(gdp, np.load(pin(old/'gdp_targets.npy')), atol=1e-9, rtol=1e-9)
        metrics = run_gdp(df, torch.from_numpy(gdp.copy()), SimpleNamespace(out_dir=folder))
        write(folder/'metrics.json', metrics)
        results[name] = metrics
        write(out/'results.json', results)
        geometry[name] = dict(input_dimension=dim, output_dimension=768, branch_scales=scales,
                              output_scale=scale, train_rms=float(np.sqrt(np.mean((train-train.mean(0))**2))))
        for prop in ALL_COLS:
            saved = np.load(folder/prop/'legacy_oof.npz')
            valid = df[prop].notna().to_numpy()
            np.testing.assert_array_equal(saved['row_index'], np.flatnonzero(valid))
            np.testing.assert_array_equal(saved['fold'], df.loc[valid,FOLD_COL])
            np.testing.assert_array_equal(saved['y_true'], df.loc[valid,prop])
            scores = [float(spearmanr(saved['y_transformed'][saved['fold']==k],saved['prediction'][saved['fold']==k]).statistic) for k in range(5)]
            assert abs(np.mean(scores)-metrics['properties'][prop]['legacy_reference']['cv_mean_spearman']) < 1e-12
            checks.append(dict(target=name, property=prop, fold_scores=scores))
        print(name, {p:v['legacy_reference']['cv_mean_spearman'] for p,v in metrics['properties'].items()}, flush=True)
    for path in [Path(__file__), ROOT/'downstream/fixed_chain_eval.py', ROOT/'downstream/grammar/ab_probes.py']:
        pin(path)
    for path, digest in sources.items():
        assert sha(Path(path)) == digest, path
    write(out/'verification.json', dict(status='passed', native_oas_control_bitwise=True,
          all_global_gdp_bitwise=True, restored_native_gdp_bitwise=True, all_eight_blocks_cross_edges=edges,
          early512_control_targets_reproduced=True, orthogonal_gram_preserved=True, geometry=geometry,
          oof_checks=checks, source_sha256=sources))
    status.update(status='success', stage='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    write(out/'status.json', status)


if __name__ == '__main__':
    main()
