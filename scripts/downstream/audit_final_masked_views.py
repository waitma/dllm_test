"""Compare actual final-layer readouts under fixed partial input masking.

Activate pllm then protenix_abtcr, then run locally from repository root:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B \
  scripts/downstream/audit_final_masked_views.py --out ABS_NEW_DIRECTORY
Two frozen models, ratios .15/.5, four fixed views, all-residue/masked-residue
pooling. Original padding MASK keys are excluded; deliberately masked genuine
residue positions remain attention keys and queries. No training or defaults
change. Views cost four forwards per ratio; all variants are reported.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    diag = ROOT / 'output/diagnostics'
    arms = {
        'parent50000': ROOT / 'output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint',
        'spectral1000': diag / 'final_layer_spectral_pilot_20261018/spectral_continue',
    }
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                arms={k: str(v) for k, v in arms.items()}, ratios=[.15, .5], views=4,
                seed=20261018, pool=['all_residues', 'masked_residues'],
                attention='exclude original padding MASK keys; preserve intentionally masked real residue keys',
                mask_sampling='SHA256(seed, H, L, view) keyed per sequence; ceil ratio per chain; nested .15/.5 masks',
                readout='actual eighth block plus final RMSNorm; FP32 residue mean, then mean of four views',
                scope='fixed-model inference diagnostic; all variants/all GDP properties; exploratory same-CV, no generation claim',
                sources={str(p): sha(p) for p in [Path(__file__), ROOT / 'downstream/final_layer_preservation_eval.py', ROOT / 'downstream/fixed_chain_eval.py', ROOT / 'downstream/grammar/ab_probes.py']})
    write(out / 'plan.json', plan)
    status = dict(status='running', pid=os.getpid(), stage='features', started_utc=plan['created_utc'])
    write(out / 'status.json', status)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from downstream.fixed_chain_eval import load as load_parent, FixedChainBackbone
    from downstream.final_layer_preservation_eval import load as load_adapted
    torch.set_num_threads(4)
    results, checks, mask_reference = {}, [], {}
    identities = {}
    depth = np.load(diag / 'fixed50000_depth_20261018/features.npz')
    for name, folder in arms.items():
        model, identity = (load_parent(folder) if name == 'parent50000' else load_adapted(folder))
        identities[name] = identity
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
        from downstream.grammar.ab_probes import prepare_task, run_gdp
        from downstream.ophiuchus_eval.developability import FOLD_COL, ALL_COLS
        df, pairs, _, _ = prepare_task('gdp_a1')
        records = [BioSeqRecord([BioSeqChain(h, 'antibody_heavy'), BioSeqChain(l, 'antibody_light')], task_type='antibody', source='native_v5_probe') for h, l in pairs]
        backbone = FixedChainBackbone(model, identity)
        baseline = depth['no_mask_keys_final_norm'] if name == 'parent50000' else np.load(diag / 'final_layer_mask_interaction_20261018/spectral1000_no_mask_keys/features.npy')
        chunks = {f'mask{int(ratio*100)}_{pool}': [] for ratio in plan['ratios'] for pool in plan['pool']}
        reference_chunks = []
        for start in range(0, len(records), 8):
            batch = backbone.collator(records[start:start+8])
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            residue = batch['residue_mask'] & batch['attention_mask']
            attention = batch['attention_mask'] & batch['input_ids'].ne(model.tokenizer.mask_token_id)
            assert not (residue & ~attention).any()
            clean = {**batch, 'attention_mask': attention}
            def pool(hidden, selection):
                assert selection.any(-1).all()
                return (hidden.float().masked_fill(~selection[..., None], 0).sum(1)/selection.sum(1, keepdim=True)).cpu().numpy()
            hidden, _ = backbone._final_hidden(clean)
            control = pool(hidden, residue)
            np.testing.assert_array_equal(control, baseline[start:start+len(control)])
            reference_chunks.append(control)
            mask_rows = residue.cpu().numpy()
            chains = batch['chain_ids'].cpu().numpy()
            for ratio in plan['ratios']:
                views = {p: [] for p in plan['pool']}
                for view in range(plan['views']):
                    selection = np.zeros_like(mask_rows)
                    for i, (h, l) in enumerate(pairs[start:start+len(mask_rows)]):
                        key = f"{plan['seed']}\0{h}\0{l}\0{view}"
                        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
                        rng = np.random.default_rng(seed)
                        for chain in np.unique(chains[i, mask_rows[i]]):
                            positions = np.flatnonzero(mask_rows[i] & (chains[i] == chain))
                            positions = rng.permutation(positions)
                            count = max(1, int(np.ceil(ratio*len(positions))))
                            selection[i, positions[:count]] = True
                        spec = f'{start+i}:{ratio}:{view}'
                        selected = np.flatnonzero(selection[i]).tolist()
                        if name == 'parent50000':
                            mask_reference[spec] = selected
                        else:
                            assert mask_reference[spec] == selected
                        if ratio == .5:
                            small = mask_reference[f'{start+i}:0.15:{view}']
                            assert set(small).issubset(selected)
                    chosen = torch.as_tensor(selection, device='cuda')
                    assert not (chosen & ~residue).any()
                    inputs = batch['input_ids'].masked_fill(chosen, model.tokenizer.mask_token_id)
                    np.testing.assert_array_equal(inputs[~chosen].cpu().numpy(), batch['input_ids'][~chosen].cpu().numpy())
                    assert attention[chosen].all()
                    hidden, _ = backbone._final_hidden({**clean, 'input_ids': inputs})
                    views['all_residues'].append(pool(hidden, residue))
                    views['masked_residues'].append(pool(hidden, chosen))
                for pooling, values in views.items():
                    chunks[f'mask{int(ratio*100)}_{pooling}'].append(np.mean(np.stack(values), axis=0))
        np.save(out / f'{name}_clean_control_features.npy', np.concatenate(reference_chunks))
        for mode, arrays in chunks.items():
            label = f'{name}_{mode}'
            destination = out / label
            destination.mkdir()
            x = np.concatenate(arrays)
            assert x.shape == baseline.shape and np.isfinite(x).all()
            np.save(destination / 'features.npy', x)
            metrics = run_gdp(df, torch.from_numpy(x.astype(np.float64)), SimpleNamespace(out_dir=destination))
            write(destination / 'metrics.json', metrics)
            results[label] = metrics
            write(out / 'metrics.json', results)
            for prop in ALL_COLS:
                saved = np.load(destination / prop / 'legacy_oof.npz')
                valid = df[prop].notna().to_numpy()
                np.testing.assert_array_equal(saved['row_index'], np.flatnonzero(valid))
                np.testing.assert_array_equal(saved['fold'], df.loc[valid, FOLD_COL])
                np.testing.assert_array_equal(saved['y_true'], df.loc[valid, prop])
                values = [float(spearmanr(saved['y_transformed'][saved['fold'] == i], saved['prediction'][saved['fold'] == i]).statistic) for i in range(5)]
                assert abs(np.mean(values)-metrics['properties'][prop]['legacy_reference']['cv_mean_spearman']) < 1e-12
                checks.append(dict(readout=label, property=prop, fold_scores=values))
            print(label, {p: v['legacy_reference']['cv_mean_spearman'] for p, v in metrics['properties'].items()}, flush=True)
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        del model, backbone, hidden
        torch.cuda.empty_cache()
        status.update(completed=name)
        write(out / 'status.json', status)
    for path, digest in plan['sources'].items():
        assert sha(Path(path)) == digest
    write(out / 'mask_positions.json', mask_reference)
    write(out / 'identities.json', identities)
    write(out / 'verification.json', dict(status='passed', clean_control_features_bitwise_equal=True,
          same_masks_between_models=True, nested_masks=True, original_padding_only_excluded_from_attention=True,
          all_eight_blocks_and_final_norm=True, oof_checks=checks))
    status.update(status='success', stage='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    write(out / 'status.json', status)


if __name__ == '__main__':
    main()
