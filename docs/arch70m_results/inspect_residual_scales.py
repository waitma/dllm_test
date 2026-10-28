"""Inspect frozen A4 screening checkpoints on CPU without running inference.

Activate pllm and set PYTHONPATH to the exploration root, then use
output/arch70m_env/bin/python /absolute/path/inspect_residual_scales.py
--output-dir /absolute/path/report. The confirmation trials remain untouched.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from safetensors import safe_open
import torch

from experiments.arch70m.common import digest, sha, source_binding

ROOT = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore')
RUN = ROOT / 'output/arch70m_v8_20261028'
SOURCE = '053297983e0f2ece2aaad807032695fb5aebd1e0f221138a9bf0d202cc9b9ac4'


def inspect(output):
    assert digest(source_binding()) == SOURCE
    torch.set_num_threads(1)
    trial = RUN / 'trials/screen_A4_s42'
    status = json.loads((trial / 'status.json').read_text())
    assert status['status'] == 'completed' and status['step'] == 1276
    rows, evidence = [], {}
    initial = float(torch.tensor(.1, dtype=torch.float32))
    for step in (500, 1250, 1276):
        checkpoint = trial / 'checkpoints' / f'step_{step:06d}'
        receipt = json.loads((checkpoint / 'receipt.json').read_text())
        assert receipt['status'] == 'complete'
        for name in ('model.safetensors', 'binding.json', 'state.json'):
            path = checkpoint / name
            actual = sha(path)
            assert actual == receipt['files'][name], path
            evidence[str(path)] = actual
        binding = json.loads((checkpoint / 'binding.json').read_text())
        state = json.loads((checkpoint / 'state.json').read_text())
        assert binding['source_sha256'] == SOURCE and binding['variant'] == 'A4' and binding['seed'] == 42
        assert state['step'] == step
        with safe_open(str(checkpoint / 'model.safetensors'), framework='pt', device='cpu') as tensors:
            keys = [k for k in tensors.keys() if k.endswith(('attention_scale', 'ffn_scale'))]
            assert len(keys) == 16
            for key in keys:
                values = tensors.get_tensor(key).double()
                assert values.shape == (768,) and torch.isfinite(values).all()
                layer = int(key.split('.')[3]) + 1
                mean = float(values.mean())
                rows.append(dict(step=step, layer=layer, component=key.rsplit('.', 1)[-1],
                    attention_visibility='chain' if layer % 2 else 'global', channels=768,
                    mean=mean, channel_std=float(values.std(unbiased=False)),
                    minimum=float(values.min()), maximum=float(values.max()),
                    relative_mean_change_percent=100 * (mean / initial - 1),
                    fraction_above_initial=float((values > initial).double().mean())))
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'layer_scales.csv').open('w') as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), variant='A4', seed=42,
        initial_scale=initial, extra_trainable_parameters=12288, source_sha256=SOURCE,
        checkpoint_file_sha256=evidence, layers=rows,
        definitions=dict(scale='Channelwise multiplier on the attention or FFN residual branch.',
            baseline='A0 keeps the same 16 vectors fixed at 0.1; they are nonpersistent buffers, not checkpoint parameters.',
            channel_std='Population SD across 768 channels within one layer, not a seed uncertainty interval.'),
        limitations=['Descriptive single-seed screening checkpoint analysis; no confirmation result was used.',
            'Scale changes alone do not measure branch contribution or attention probability; other weights adapt jointly.',
            'This does not establish that the learned scale pattern caused the observed probe AUROC difference.'])
    (output / 'residual_scale_analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'arch70m-residual-scales', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(left=.08, right=.98, top=.74, bottom=.25, wspace=.23)
    fig.text(.08, .93, 'A4 learned residual scales', fontsize=21, weight='bold', color='#243447')
    fig.text(.08, .86, 'Screening step 1276 | seed 42 | A0 fixed scale = 0.1', fontsize=12)
    for ax, component, title, color in zip(axes, ('attention_scale', 'ffn_scale'),
                                          ('Attention branch', 'FFN branch'), ('#0077BB', '#228833')):
        final = sorted((r for r in rows if r['step'] == 1276 and r['component'] == component), key=lambda r: r['layer'])
        x = [r['layer'] for r in final]
        mean = [r['mean'] for r in final]
        ax.fill_between(x, [r['mean']-r['channel_std'] for r in final],
                        [r['mean']+r['channel_std'] for r in final], color=color, alpha=.16)
        ax.plot(x, mean, color=color, marker='o', label='A4 mean across channels')
        ax.axhline(initial, color='#556677', ls='--', lw=1, label='A0 / initial scale')
        ax.set(title=title, ylabel='Residual multiplier', ylim=(.09, .116), xticks=x,
               xticklabels=[f'{i}\n'+('chain' if i % 2 else 'global') for i in x])
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.text(.08, .12, 'Layer number and its attention visibility. Shading: channel SD; this is not variation across training seeds.', fontsize=10)
    fig.text(.08, .06, 'Descriptive weight analysis only: other weights adapt jointly, so scale magnitudes do not establish causal feature contributions.', fontsize=9, color='#64748B')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'residual_scales.{ext}', dpi=180)
    svg = output / 'residual_scales.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print(json.dumps(dict(output=str(output), checkpoints_verified=3, scale_vectors=48,
                          final_layers=[r for r in rows if r['step'] == 1276]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    assert args.output_dir.is_absolute()
    inspect(args.output_dir)
