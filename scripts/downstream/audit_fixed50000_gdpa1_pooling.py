"""Compare six fixed50000 GDPa1 readouts, without changing weights or inputs.

Activate pllm then protenix_abtcr; run from the repository root:
python -B scripts/downstream/audit_fixed50000_gdpa1_pooling.py --out ABS_NEW_DIRECTORY
Uses the local GPU with existing BF16 autocast, then the original CPU Ridge CV.
"""
import argparse
from collections import Counter
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
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from downstream.fixed_chain_eval import load, FixedChainBackbone
    from downstream.fixed_chain_layer_eval import Layer2Backbone, block_hidden, sha
    torch.set_num_threads(4)
    baseline = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_20261013'
    layer_run = ROOT/'output/downstream_generation/fixed_chain_75m_step50000_layer2_20261018'
    readouts = [f'{layer}_{pool}' for layer in ('block2', 'final_norm')
                for pool in ('residue_global', 'residue_hl_concat', 'all_nonmask')]
    sources = [Path(__file__), ROOT/'downstream/fixed_chain_eval.py',
               ROOT/'downstream/fixed_chain_layer_eval.py', ROOT/'downstream/grammar/ab_probes.py',
               ROOT/'data/downstream/dev/GDPa1_v1.2_20250814.csv',
               baseline/'gdp_a1/metrics.json', layer_run/'gdp_a1/metrics.json']
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(baseline/'checkpoint'),
                device='cuda:0', precision='original_cuda_bfloat16_autocast_pool_fp32',
                batch_size=8, cpu_threads=4, readouts=readouts,
                masks=dict(residue_global='attention & residue',
                           residue_hl_concat='H and L observed residues averaged separately, concatenated H then L',
                           all_nonmask='attention & input_ids != MASK & input_ids != PAD; includes structural tokens and EOS'),
                changes='pooling only within each fixed depth; original forward input/attention unchanged',
                regression='unchanged AirGen reference Ridge grid, folds and label transform; all five properties',
                selection='report all six variants; no automatic replacement of default',
                source_sha256={str(p): sha(p) for p in sources})
    write(out/'plan.json', plan)
    write(out/'status.json', dict(status='running', stage='features'))
    model, identity = load(baseline/'checkpoint', device='cuda')
    write(out/'identity.json', identity)
    full = FixedChainBackbone(model, identity)
    early = Layer2Backbone(model, identity)
    from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord, CHAIN_ROLE_TO_ID
    from downstream.grammar.ab_probes import prepare_task, run_gdp
    from downstream.ophiuchus_eval.developability import ALL_COLS, FOLD_COL
    df, pairs, _, _ = prepare_task('gdp_a1')
    records = [BioSeqRecord([BioSeqChain(h, 'antibody_heavy'), BioSeqChain(l, 'antibody_light')],
                           task_type='antibody', source='native_v5_probe') for h, l in pairs]
    vocab = json.loads((baseline/'checkpoint/vocabulary.json').read_text())
    chunks = {key: [] for key in readouts}
    token_rows = []
    for start in range(0, len(records), 8):
        batch = full.collator(records[start:start+8])
        batch = {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in batch.items()}
        ids, attention = batch['input_ids'], batch['attention_mask'].bool()
        residues = attention & batch['residue_mask'].bool()
        hm = residues & batch['role_ids'].eq(CHAIN_ROLE_TO_ID['antibody_heavy'])
        lm = residues & batch['role_ids'].eq(CHAIN_ROLE_TO_ID['antibody_light'])
        nonmask = attention & ids.ne(model.tokenizer.mask_token_id) & ids.ne(model.tokenizer.pad_token_id)
        assert torch.equal(hm | lm, residues) and not (hm & lm).any()
        assert not (residues & ~nonmask).any()
        counts = torch.stack([hm.sum(1), lm.sum(1)], dim=1).tolist()
        assert counts == [[len(h), len(l)] for h, l in pairs[start:start+8]]
        ab_positions = ids.eq(model.tokenizer.special_id('<ab>'))
        assert (ab_positions.sum(1) == 1).all() and (nonmask & ab_positions).sum() == len(counts)
        assert not (nonmask & ids.eq(model.tokenizer.mask_token_id)).any()
        assert not (nonmask & ids.eq(model.tokenizer.pad_token_id)).any()
        observed = []
        handle = model.backbone.transformer.blocks[1].register_forward_hook(
            lambda module, inputs, output: observed.append(block_hidden(output).detach().clone()))
        try:
            final, _ = full._final_hidden(batch)
        finally:
            handle.remove()
        assert len(observed) == 1
        for i in range(len(counts)):
            special_ids = ids[i][nonmask[i] & ~residues[i]].tolist()
            token_rows.append(dict(row=start+i, heavy=counts[i][0], light=counts[i][1],
                                   all_nonmask=int(nonmask[i].sum()),
                                   excluded_mask=int((attention[i] & ids[i].eq(model.tokenizer.mask_token_id)).sum()),
                                   added_special_tokens=dict(Counter(vocab[x] for x in special_ids))))
        for layer, hidden in [('block2', observed[0]), ('final_norm', final)]:
            def pool(mask):
                assert mask.sum(1).gt(0).all()
                return hidden.float().masked_fill(~mask[..., None], 0).sum(1)/mask.sum(1, keepdim=True)
            global_mean, heavy, light = pool(residues), pool(hm), pool(lm)
            rebuilt = (heavy*hm.sum(1, keepdim=True)+light*lm.sum(1, keepdim=True))/residues.sum(1, keepdim=True)
            torch.testing.assert_close(global_mean, rebuilt, atol=2e-6, rtol=2e-5)
            pooled = dict(residue_global=global_mean, residue_hl_concat=torch.cat([heavy, light], dim=1),
                          all_nonmask=pool(nonmask))
            if start == 0:
                adapter = early if layer == 'block2' else full
                torch.testing.assert_close(global_mean.cpu(), torch.from_numpy(adapter.vectors(records[:8])), atol=0, rtol=0)
            for key, value in pooled.items():
                assert torch.isfinite(value).all()
                chunks[f'{layer}_{key}'].append(value.cpu())
        print('features', min(start+8, len(records)), len(records), flush=True)
    xs = {key: torch.cat(values) for key, values in chunks.items()}
    np.savez_compressed(out/'features.npz', **{key: value.numpy() for key, value in xs.items()})
    write(out/'token_audit.json', dict(n=len(records), rows=token_rows,
          dimensions={k: list(v.shape) for k, v in xs.items()},
          mask_and_padding_excluded=True, ab_token_included=True,
          native_global_first_batch_exact=True, chain_length_weighted_global_reconstruction=True))
    results, checks = {}, []
    write(out/'status.json', dict(status='running', stage='ridge'))
    for name, x in xs.items():
        folder = out/name
        folder.mkdir()
        result = run_gdp(df, x, SimpleNamespace(out_dir=folder))
        write(folder/'metrics.json', result)
        results[name] = result
        write(out/'metrics.json', results)
        for col in ALL_COLS:
            saved = np.load(folder/col/'legacy_oof.npz')
            valid = df[col].notna().to_numpy()
            np.testing.assert_array_equal(saved['row_index'], np.flatnonzero(valid))
            np.testing.assert_array_equal(saved['fold'], df.loc[valid, FOLD_COL].to_numpy())
            np.testing.assert_array_equal(saved['y_true'], df.loc[valid, col].to_numpy())
            scores = [float(spearmanr(saved['y_transformed'][saved['fold']==i], saved['prediction'][saved['fold']==i]).statistic) for i in range(5)]
            assert abs(np.mean(scores)-result['properties'][col]['legacy_reference']['cv_mean_spearman']) < 1e-12
            checks.append(dict(readout=name, property=col, fold_scores=scores))
    controls = {}
    for layer, source in [('block2', layer_run), ('final_norm', baseline)]:
        old = json.loads((source/'gdp_a1/metrics.json').read_text())
        current = results[layer+'_residue_global']
        for col in ALL_COLS:
            a, b = [r['properties'][col]['legacy_reference'] for r in (old, current)]
            assert abs(a['cv_mean_spearman']-b['cv_mean_spearman']) < 1e-12, (layer, col, a, b)
            assert a['alpha'] == b['alpha'], (layer, col)
        controls[layer] = 'all five existing GPU scores and selected alpha reproduced'
    for path, digest in plan['source_sha256'].items():
        assert sha(path) == digest
    assert not any(p.requires_grad for p in model.parameters()) and not model.training
    assert all(not b._forward_hooks for b in model.backbone.transformer.blocks)
    write(out/'verification.json', dict(status='passed', controls=controls, independent_oof_checks=checks,
                                       backbone_frozen=True, hooks_removed=True,
                                       feature_sha256=sha(out/'features.npz')))
    write(out/'status.json', dict(status='success', finished_utc=datetime.now(timezone.utc).isoformat(), readouts=readouts))
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
