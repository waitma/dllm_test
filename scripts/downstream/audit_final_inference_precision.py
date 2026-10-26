"""Test FP32 inference of one verified final-layer checkpoint against BF16.

Activate pllm then protenix_abtcr; from the repository root:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B \
  scripts/downstream/audit_final_inference_precision.py \
  --arm ABS_VERIFIED_CHECKPOINT --out ABS_NEW_DIRECTORY
Keep weights, inputs, pooling and float64 Ridge fixed. Reproduce both saved
BF16 readouts first; evaluate native and no-mask keys in FP32 with TF32 off.
This tests inference precision, not whether BF16 training restricted learning.
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
    parser.add_argument('--arm', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    arm, out = args.arm.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat())
    write(out / 'status.json', status)
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from downstream.final_layer_preservation_eval import load
    from downstream.fixed_chain_eval import FixedChainBackbone
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert json.loads((arm / 'verification.json').read_text())['status'] == 'passed'
    model, identity = load(arm)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
    from downstream.grammar.ab_probes import prepare_task, run_gdp
    from downstream.ophiuchus_eval.developability import ALL_COLS, FOLD_COL
    df, pairs, _, _ = prepare_task('gdp_a1')
    records = [BioSeqRecord([BioSeqChain(h, 'antibody_heavy'), BioSeqChain(l, 'antibody_light')],
                           task_type='antibody', source='native_v5_probe') for h, l in pairs]
    sources = {str(p): sha(p) for p in [Path(__file__), arm/'checkpoint.json', arm/'verification.json',
               arm/'final_features.npy', arm/'final_no_mask_features.npy',
               ROOT/'downstream/final_layer_preservation_eval.py', ROOT/'downstream/fixed_chain_eval.py',
               ROOT/'downstream/grammar/ab_probes.py']}
    write(out / 'plan.json', dict(checkpoint=str(arm), identity=identity, weights='unchanged FP32 stored weights',
          intervention='disable BF16 autocast and TF32 only; original batch8, FP32 residue pool, float64 Ridge',
          selection='both attention modes and all five properties; no checkpoint selection', source_sha256=sources))

    class PrecisionBackbone(FixedChainBackbone):
        fp32 = False
        no_mask = False

        def _final_hidden(self, batch):
            if self.no_mask:
                changed = batch['attention_mask'] & batch['input_ids'].ne(self.model.tokenizer.mask_token_id)
                assert not (batch['residue_mask'] & batch['attention_mask'] & ~changed).any()
                batch = {**batch, 'attention_mask': changed}
            if not self.fp32:
                return super()._final_hidden(batch)
            with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                logits, hidden = self.model.denoise(**self.model.feature_args(batch), output_hidden_states=True)
            assert hidden.dtype == torch.float32
            return hidden, logits

    backbone = PrecisionBackbone(model, identity)
    results, checks, changes = {}, [], {}
    for mode, filename in [('native', 'final_features.npy'), ('no_mask_keys', 'final_no_mask_features.npy')]:
        backbone.no_mask = mode == 'no_mask_keys'
        backbone.fp32 = False
        reference = np.load(arm / filename)
        np.testing.assert_array_equal(backbone.vectors(records), reference)
        backbone.fp32 = True
        values = backbone.vectors(records)
        assert np.isfinite(values).all()
        delta = values.astype(np.float64)-reference.astype(np.float64)
        centered = reference.astype(np.float64)-reference.astype(np.float64).mean(0)
        changes[mode] = dict(max_abs_difference=float(np.max(np.abs(delta))),
                            rms_difference=float(np.sqrt(np.mean(delta**2))),
                            rms_difference_over_centered_feature_rms=float(np.sqrt(np.mean(delta**2)/np.mean(centered**2))))
        folder = out / mode
        folder.mkdir()
        np.save(folder / 'features.npy', values)
        metrics = run_gdp(df, torch.from_numpy(values.astype(np.float64)), SimpleNamespace(out_dir=folder))
        write(folder / 'metrics.json', metrics)
        results[mode] = metrics
        for prop in ALL_COLS:
            saved = np.load(folder / prop / 'legacy_oof.npz')
            valid = df[prop].notna().to_numpy()
            np.testing.assert_array_equal(saved['row_index'], np.flatnonzero(valid))
            np.testing.assert_array_equal(saved['fold'], df.loc[valid, FOLD_COL])
            np.testing.assert_array_equal(saved['y_true'], df.loc[valid, prop])
            scores = [float(spearmanr(saved['y_transformed'][saved['fold']==k], saved['prediction'][saved['fold']==k]).statistic) for k in range(5)]
            assert abs(np.mean(scores)-metrics['properties'][prop]['legacy_reference']['cv_mean_spearman']) < 1e-12
            checks.append(dict(mode=mode, property=prop, fold_scores=scores))
    for path, digest in sources.items():
        assert sha(Path(path)) == digest, path
    write(out / 'results.json', results)
    write(out / 'feature_differences.json', changes)
    write(out / 'verification.json', dict(status='passed', bf16_features_bitwise_reproduced=True,
          tf32_disabled=True, fp32_hidden_verified=True, oof_checks=checks, source_sha256=sources))
    status.update(status='success', finished_utc=datetime.now(timezone.utc).isoformat())
    write(out / 'status.json', status)
    for name, metrics in results.items():
        print(name, {p: x['legacy_reference']['cv_mean_spearman'] for p, x in metrics['properties'].items()})


if __name__ == '__main__':
    main()
