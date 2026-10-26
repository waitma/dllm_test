"""Test whether MASK-tail attention and representation training interact.

Activate pllm/protenix_abtcr; run locally from repository root:
python -B scripts/downstream/audit_preservation_mask_interaction.py --out ABS_NEW_DIR
All three prespecified trained models, native and no-MASK-key inference, all
five GDP properties. No training or generation-path changes.
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
from scripts.research.train_final_layer_preservation_pilot import write, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    spectral = ROOT / 'output/diagnostics/final_layer_spectral_pilot_20261018'
    assert json.loads((spectral / 'status.json').read_text())['status'] == 'success'
    out.mkdir(parents=True, exist_ok=False)
    arms = {'joint300': ROOT / 'output/diagnostics/final_layer_preservation_pilot_20261018/joint',
            'uniform1000': spectral / 'uniform_continue', 'spectral1000': spectral / 'spectral_continue'}
    plan = dict(arms={k: str(v) for k, v in arms.items()},
                interventions=['native', 'no_mask_keys'],
                method='Only attention_mask excludes MASK keys; same IDs/positions and original residue pooling',
                selection='all three checkpoints, both readouts, all five properties; exploratory same-CV comparison',
                scope='complete observed sequences only; no inference projection, no change to generation',
                created_utc=datetime.now(timezone.utc).isoformat(),
                code_sha256={str(p): sha(p) for p in [Path(__file__), ROOT / 'downstream/final_layer_preservation_eval.py', ROOT / 'downstream/grammar/ab_probes.py']})
    write(out / 'plan.json', plan)
    status = dict(status='running', pid=os.getpid(), stage='features', started_utc=plan['created_utc'])
    write(out / 'status.json', status)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from downstream.final_layer_preservation_eval import load
    from downstream.fixed_chain_eval import FixedChainBackbone
    torch.set_num_threads(4)
    results, checks = {}, []
    for name, folder in arms.items():
        model, identity = load(folder)
        from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
        from downstream.grammar.ab_probes import prepare_task, run_gdp
        from downstream.ophiuchus_eval.developability import FOLD_COL, ALL_COLS
        df, pairs, _, _ = prepare_task('gdp_a1')
        records = [BioSeqRecord([BioSeqChain(h, 'antibody_heavy'), BioSeqChain(l, 'antibody_light')], task_type='antibody', source='native_v5_probe') for h, l in pairs]
        backbone = FixedChainBackbone(model, identity)
        chunks = {mode: [] for mode in plan['interventions']}
        original = np.load(folder / 'final_features.npy')
        for start in range(0, len(records), 8):
            batch = backbone.collator(records[start:start+8])
            batch = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in batch.items()}
            residue = batch['residue_mask'] & batch['attention_mask']
            for mode in chunks:
                current = dict(batch)
                if mode == 'no_mask_keys':
                    current['attention_mask'] = batch['attention_mask'] & batch['input_ids'].ne(model.tokenizer.mask_token_id)
                    assert not (residue & ~current['attention_mask']).any()
                hidden, _ = backbone._final_hidden(current)
                pooled = (hidden.float().masked_fill(~residue[..., None], 0).sum(1)/residue.sum(1, keepdim=True)).cpu().numpy()
                if mode == 'native':
                    np.testing.assert_array_equal(pooled, original[start:start+8])
                chunks[mode].append(pooled)
        for mode in chunks:
            label = name + '_' + mode
            destination = out / label
            destination.mkdir()
            features = np.concatenate(chunks[mode])
            np.save(destination / 'features.npy', features)
            metrics = run_gdp(df, torch.from_numpy(features.astype(np.float64)), SimpleNamespace(out_dir=destination))
            write(destination / 'metrics.json', metrics)
            for col in ALL_COLS:
                saved = np.load(destination / col / 'legacy_oof.npz')
                valid = df[col].notna().to_numpy()
                np.testing.assert_array_equal(saved['row_index'], np.flatnonzero(valid))
                np.testing.assert_array_equal(saved['fold'], df.loc[valid, FOLD_COL])
                np.testing.assert_array_equal(saved['y_true'], df.loc[valid, col])
                scores = [float(spearmanr(saved['y_transformed'][saved['fold']==i], saved['prediction'][saved['fold']==i]).statistic) for i in range(5)]
                assert abs(np.mean(scores)-metrics['properties'][col]['legacy_reference']['cv_mean_spearman']) < 1e-12
                if mode == 'native':
                    old = json.loads((folder / 'gdp_a1/metrics.json').read_text())
                    assert abs(np.mean(scores)-old['properties'][col]['legacy_reference']['cv_mean_spearman']) < 1e-12
                checks.append(dict(readout=label, property=col, fold_scores=scores))
            results[label] = metrics
            write(out / 'metrics.json', results)
            print(label, {k: v['legacy_reference']['cv_mean_spearman'] for k,v in metrics['properties'].items()}, flush=True)
        del model, backbone, hidden
        torch.cuda.empty_cache()
        status.update(completed_arm=name)
        write(out / 'status.json', status)
    for path, digest in plan['code_sha256'].items():
        assert sha(Path(path)) == digest
    write(out / 'verification.json', dict(status='passed', checks=checks, native_features_bitwise_equal=True, native_scores_reproduced=True))
    status.update(status='success', stage='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    write(out / 'status.json', status)


if __name__ == '__main__':
    main()
