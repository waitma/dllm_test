"""CPU-only GDPa1 protocol/readout diagnostic; no backbone updates or leaderboard changes.

Activate pllm, then protenix_abtcr; run python -B scripts/downstream/audit_gdpa1_airgen.py
--kind post --run ABS_EXISTING_RUN --out ABS_NEW_DIRECTORY. Use --kind fixed for parent.
The two predeclared readouts share exactly the same native final hidden states.
AirGen's three regression functions are executed directly from their AST, without
importing its model loader or changing its code. Original benchmark files are read-only.
"""
import argparse
import ast
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['post', 'fixed'], required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    air = ROOT.parent / 'airgen/AirGen-Dev/downstream/dev/finetune_dev.py'
    data = ROOT / 'data/downstream/dev/GDPa1_v1.2_20250814.csv'
    source_paths = [Path(__file__), air, data, ROOT/'downstream/grammar/ab_probes.py',
                    ROOT/'downstream/fixed_chain_eval.py', ROOT/'downstream/post_training_eval.py']
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), kind=a.kind,
                run=str(a.run), device='cpu', precision='float32', threads=8, batch_size=8,
                readouts=['global', 'chain_concat'], selection='none; report all five properties',
                scope='diagnostic only; native inputs unchanged; no default change',
                sources={str(f): sha(f) for f in source_paths})
    write(a.out/'plan.json', plan)
    import numpy as np
    import pandas as pd
    import scipy.stats
    import torch
    from sklearn.linear_model import Ridge
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import GridSearchCV, PredefinedSplit
    from sklearn.preprocessing import PowerTransformer
    torch.set_num_threads(8)
    if a.kind == 'post':
        from downstream.post_training_eval import load, PostTrainingBackbone as Backbone
    else:
        from downstream.fixed_chain_eval import load, FixedChainBackbone as Backbone
    model, identity = load(a.run/'checkpoint', device='cpu')
    backbone = Backbone(model, identity)
    from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord, CHAIN_ROLE_TO_ID
    from downstream.grammar.ab_probes import prepare_task, run_gdp
    df, pairs, _, _ = prepare_task('gdp_a1')
    records = [BioSeqRecord([BioSeqChain(h, 'antibody_heavy'), BioSeqChain(l, 'antibody_light')],
                           task_type='antibody', source='native_v5_probe') for h, l in pairs]
    write(a.out/'identity.json', identity)
    chunks = {'global': [], 'chain_concat': []}
    lengths = []
    for start in range(0, len(records), 8):
        batch = backbone.collator(records[start:start+8])
        features = model.feature_args(batch)
        assert 'labels' not in features
        hidden, _ = backbone._final_hidden(batch)
        mask = batch['residue_mask'].bool() & batch['attention_mask'].bool()
        hm = mask & batch['role_ids'].eq(CHAIN_ROLE_TO_ID['antibody_heavy'])
        lm = mask & batch['role_ids'].eq(CHAIN_ROLE_TO_ID['antibody_light'])
        assert torch.equal(hm | lm, mask) and not (hm & lm).any()
        expected = [[len(h), len(l)] for h, l in pairs[start:start+8]]
        counts = torch.stack([hm.sum(1), lm.sum(1)], dim=1).tolist()
        assert counts == expected, (counts, expected)
        def pool(m):
            return hidden.float().masked_fill(~m[..., None], 0).sum(1) / m.sum(1, keepdim=True)
        glob, heavy, light = pool(mask), pool(hm), pool(lm)
        assert torch.isfinite(glob).all() and torch.isfinite(heavy).all() and torch.isfinite(light).all()
        rebuilt = (heavy*hm.sum(1, keepdim=True)+light*lm.sum(1, keepdim=True))/mask.sum(1, keepdim=True)
        torch.testing.assert_close(glob, rebuilt, atol=2e-6, rtol=2e-5)
        if start == 0:
            control = torch.from_numpy(backbone.vectors(records[:8]))
            torch.testing.assert_close(glob.cpu(), control, atol=0, rtol=0)
        chunks['global'].append(glob.cpu())
        chunks['chain_concat'].append(torch.cat([heavy, light], dim=-1).cpu())
        lengths.extend(counts)
        print(f'features {min(start+8,len(records))}/{len(records)}', flush=True)
    xs = {k: torch.cat(v) for k, v in chunks.items()}
    np.savez_compressed(a.out/'features.npz', **{k: v.numpy() for k,v in xs.items()}, lengths=lengths)
    # Execute the actual AirGen regression functions, intercepting only the search
    # constructor to retrieve unrounded best_score_ and CV predictions.
    selected = {'gaussian_transform', 'spearman_corr', 'cross_validate'}
    tree = ast.parse(air.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    assert {n.name for n in nodes} == selected
    searches = []
    def capture_search(*args, **kwargs):
        search = GridSearchCV(*args, **kwargs)
        searches.append(search)
        return search
    env = dict(np=np, scipy=scipy, Ridge=Ridge, GridSearchCV=capture_search,
               PredefinedSplit=PredefinedSplit, make_scorer=make_scorer, PowerTransformer=PowerTransformer)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(air), 'exec'), env)
    historical = json.loads((a.run/'gdp_a1/metrics.json').read_text())
    report = dict(plan_sha256=sha(a.out/'plan.json'), identity=identity, n=len(df),
                  dimensions={k: list(v.shape) for k,v in xs.items()}, properties={},
                  global_matches_native_first_batch=True, all_residue_lengths_match=True,
                  caveat='CPU FP32 diagnostic; historical production used GPU BF16; no headline replacement')
    metrics = {}
    for name, x in xs.items():
        folder = a.out/name
        folder.mkdir()
        metrics[name] = run_gdp(df, x, SimpleNamespace(out_dir=folder))
        write(folder/'metrics.json', metrics[name])
    for prop, original in historical['properties'].items():
        valid = df[prop].notna().to_numpy()
        y = df.loc[valid, prop].to_numpy(dtype=float)
        folds = df.loc[valid, 'hierarchical_cluster_IgG_isotype_stratified_fold'].to_numpy(dtype=int)
        result = {'historical_gpu_global': original['legacy_reference']['cv_mean_spearman']}
        for name, x in xs.items():
            with redirect_stdout(io.StringIO()):
                env['cross_validate'](x.numpy()[valid], y, folds, scale_all=True)
            search = searches[-1]
            our = metrics[name]['properties'][prop]['legacy_reference']
            delta = float(search.best_score_) - our['cv_mean_spearman']
            assert abs(delta) < 1e-9, (name, prop, delta)
            assert float(search.best_params_['alpha']) == our['alpha']
            result[name] = dict(our_score=our['cv_mean_spearman'], airgen_score=float(search.best_score_),
                                alpha=our['alpha'], score_difference=delta)
        result['chain_minus_global'] = result['chain_concat']['our_score']-result['global']['our_score']
        result['cpu_minus_historical_global'] = result['global']['our_score']-result['historical_gpu_global']
        report['properties'][prop] = result
    assert all(sha(f) == digest for f, digest in plan['sources'].items())
    report['status'] = 'passed'
    write(a.out/'audit.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
