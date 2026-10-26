"""Compare actual final-hidden alignment across train, screen and GDP sequences.

Activate pllm then protenix_abtcr; run locally from the repository root:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 USE_TF=0 python -B \
  scripts/downstream/audit_final_alignment_generalization.py \
  --arm ABS_VERIFIED_CHECKPOINT --out ABS_NEW_DIRECTORY
Use first128 fixed OAS train rows, all128 existing screen rows, and saved GDP
features. No GDP labels, PCA refitting, new probe, or intermediate checkpoint
selection. Both native and no-mask-key actual final-hidden readouts are kept.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

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
    from downstream.final_layer_preservation_eval import load
    from downstream.fixed_chain_eval import FixedChainBackbone
    from downstream.fixed_chain_layer_eval import block_hidden
    torch.set_num_threads(4)
    root = ROOT / 'output/diagnostics'
    sources = {}

    def pin(path):
        sources[str(path)] = sha(path)
        return path

    verified = json.loads(pin(arm / 'verification.json').read_text())
    assert verified['status'] == 'passed' and verified['reloaded_features_bitwise_equal']
    side = json.loads(pin(arm / 'checkpoint.json').read_text())
    assert sha(pin(Path(side['parameter_file']))) == side['sha256']
    cal = np.load(pin(root / 'final_layer_coordinate_pilot_20261018/calibration.npz'))
    spectrum = np.load(pin(root / 'final_layer_spectral_pilot_20261018/spectrum.npz'))
    axes, eigen = spectrum['axes'], spectrum['eigenvalues']
    data_root = root / 'final_layer_preservation_pilot_20261018'
    data_manifest = json.loads(pin(data_root / 'data.json').read_text())
    groups = {}
    for split in ('train', 'screen'):
        path = pin(data_root / f'{split}.jsonl')
        assert sources[str(path)] == data_manifest['files'][split]['sha256']
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        groups[split] = rows[:128]
        assert len(groups[split]) == 128
        if split == 'screen':
            assert len(rows) == 128
    write(out / 'plan.json', dict(checkpoint=str(arm), selection='first128 OAS train; all128 screen; all saved GDP sequences',
          labels_used=False, axes='existing OAS4096 PCA, means, scales and rotation; no refit',
          scope='alignment diagnostic only; neither new GDP performance nor a new readout'))
    model, identity = load(arm)
    from dllm.pipelines.immune_llada.data.records import BioSeqRecord
    backbone = FixedChainBackbone(model, identity)
    write(out / 'model_identity.json', identity)
    collections = {}

    def pool(hidden, mask):
        return (hidden.float().masked_fill(~mask[..., None], 0).sum(1)/mask.sum(1, keepdim=True)).cpu().numpy()

    for split, rows in groups.items():
        chunks = dict(teacher=[], native=[], no_mask_keys=[])
        for start in range(0, len(rows), 16):
            batch = backbone.collator([BioSeqRecord.from_prepared_dict(row) for row in rows[start:start+16]])
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            mask = batch['residue_mask'] & batch['attention_mask']
            teacher = []
            hook = model.backbone.transformer.blocks[1].register_forward_hook(
                lambda mod, inputs, result: teacher.append(block_hidden(result).detach()))
            try:
                hidden, _ = backbone._final_hidden(batch)
            finally:
                hook.remove()
            assert len(teacher) == 1
            chunks['teacher'].append(pool(teacher[0], mask))
            chunks['native'].append(pool(hidden, mask))
            modified = {**batch, 'attention_mask': batch['attention_mask'] & batch['input_ids'].ne(model.tokenizer.mask_token_id)}
            assert not (mask & ~modified['attention_mask']).any()
            hidden, _ = backbone._final_hidden(modified)
            chunks['no_mask_keys'].append(pool(hidden, mask))
        collections[split] = {name: np.concatenate(values).astype(np.float64) for name, values in chunks.items()}
        np.savez_compressed(out / f'{split}_features.npz', **collections[split])
    np.testing.assert_array_equal(collections['train']['teacher'], cal['teacher'][:128])
    collections['gdp'] = dict(teacher=np.load(pin(root / 'fixed50000_depth_20261018/features.npz'))['native_block2'].astype(np.float64),
                             native=np.load(pin(arm / 'final_features.npy')).astype(np.float64),
                             no_mask_keys=np.load(pin(arm / 'final_no_mask_features.npy')).astype(np.float64))
    results = {}
    for split, values in collections.items():
        target = ((values['teacher']-cal['mu_t'])/cal['rms_t']) @ axes.T
        for mode in ('native', 'no_mask_keys'):
            student = values[mode]
            assert student.shape == values['teacher'].shape and np.isfinite(student).all()
            prediction = ((student-cal['mu_s'])/cal['rms_s']) @ cal['rotation'].T @ axes.T
            error = prediction-target
            coordinate = float(np.mean(error**2))
            weights = spectrum['weights']
            spectral = float(np.mean(error**2*weights)/np.mean(eigen*weights))
            bands = {}
            for a, b in [(0, 16), (16, 64), (64, 256), (256, 768)]:
                pred, true = prediction[:, a:b], target[:, a:b]
                centered = (pred-pred.mean(0))-(true-true.mean(0))
                local_variance = float(np.mean((true-true.mean(0))**2))
                oas_variance = float(np.mean(eigen[a:b]))
                mse = float(np.mean((pred-true)**2))
                cmse = float(np.mean(centered**2))
                bias = float(np.mean((pred.mean(0)-true.mean(0))**2))
                assert abs(mse-cmse-bias) < 1e-10
                bands[f'PC{a+1}-{b}'] = dict(mse_over_oas_variance=mse/oas_variance,
                    centered_mse_over_oas_variance=cmse/oas_variance,
                    centered_mse_over_local_variance=cmse/local_variance,
                    mean_bias_over_oas_variance=bias/oas_variance,
                    local_variance_over_oas_variance=local_variance/oas_variance)
            results[split+'/'+mode] = dict(n=len(student), coordinate_mse=coordinate, spectral_mse=spectral, bands=bands)
    history = json.loads(pin(arm / 'validation.json').read_text())[-1]
    for key in ('coordinate_mse', 'spectral_mse'):
        assert abs(results['screen/no_mask_keys'][key]-history[key]) < 2e-6, key
    for path in [Path(__file__), ROOT/'downstream/final_layer_preservation_eval.py', ROOT/'downstream/fixed_chain_eval.py']:
        pin(path)
    for path, digest in sources.items():
        assert sha(Path(path)) == digest, path
    write(out / 'results.json', results)
    write(out / 'verification.json', dict(status='passed', original_train_teacher_bitwise_reproduced=True,
          screen_final_monitor_reproduced=True, bias_variance_decomposition=True, source_sha256=sources))
    status.update(status='success', finished_utc=datetime.now(timezone.utc).isoformat())
    write(out / 'status.json', status)
    for name, values in results.items():
        print(name, 'coordinate', round(values['coordinate_mse'], 6), 'spectral', round(values['spectral_mse'], 6),
              'centered_error_over_local_variance', [round(x['centered_mse_over_local_variance'], 6) for x in values['bands'].values()])


if __name__ == '__main__':
    main()
