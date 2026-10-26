"""Plot the verified fixed5000 CE0/CE1 comparison without selecting checkpoints.

Activate pllm then protenix_abtcr; from repository root run:
python -B scripts/downstream/plot_final_joint_continuation.py --comparison ABS_AUDIT_DIRECTORY
Writes standalone PNG/SVG and source hashes into that completed audit directory.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.research.train_final_layer_preservation_pilot import write, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    args = parser.parse_args()
    folder = args.comparison.resolve()
    assert json.loads((folder/'verification.json').read_text())['status'] == 'passed'
    report = json.loads((folder/'results.json').read_text())
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'block2':'#555555', 'ce0':'#d45c35', 'ce1':'#1676a5'}
    names = {'block2':'Original block 2', 'ce0':'Final: CE weight 0', 'ce1':'Final: CE weight 1'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, mode in zip(axes[0], ['native', 'no_mask_keys']):
        properties = report['GDP'][mode]
        x = np.arange(len(properties))
        for i,name in enumerate(names):
            values = [v['scores'][name]['mean'] for v in properties.values()]
            bars = ax.bar(x+(i-1)*.25, values, width=.23, label=names[name], color=colors[name])
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+(.012 if value>=0 else -.012), f'{value:.3f}',
                        ha='center', va='bottom' if value>=0 else 'top', fontsize=7, rotation=90)
        ax.set_xticks(x, ['AC-SINS', 'HIC', 'PR_CHO', 'Titer', 'Tm2'])
        ax.set_ylim(-.14,.68)
        ax.set_ylabel('Mean five-fold Spearman')
        ax.axhline(0, lw=.7, color='#888888')
        ax.set_title('Native final readout' if mode=='native' else 'Final readout: tail MASK keys excluded')
    axes[0,0].legend(loc='upper right', fontsize=8)
    paths = [folder/'results.json', folder/'verification.json', Path(__file__)]
    for name, stem in [('ce0',''),('ce1','joint_')]:
        path = ROOT/f'output/diagnostics/final_layer_{stem}convergence_pilot_20261018/token_all_direction/validation.json'
        paths.append(path)
        history = json.loads(path.read_text())
        for ax,key in [(axes[1,0],'aa_accuracy_0.5'),(axes[1,1],'spectral_mse')]:
            ax.plot([r['step'] for r in history], [r[key] for r in history], label=names[name], color=colors[name])
    axes[1,0].set_title('Held-out 50% random-mask AA accuracy')
    axes[1,0].set_ylabel('Accuracy (not CDR infilling AAR)')
    axes[1,1].set_title('Held-out pooled spectral alignment error')
    axes[1,1].set_ylabel('MSE, lower is better')
    for ax in axes[1]:
        ax.set_xlabel('Adaptation step')
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Matched initialization, optimizer and batches; fixed 5000-step endpoints\nSingle-seed exploratory reference protocol', fontsize=13)
    for extension in ['png','svg']:
        fig.savefig(folder/f'comparison.{extension}', dpi=180)
    plt.close(fig)
    write(folder/'plot_sources.json', {str(p):sha(p) for p in paths})


if __name__=='__main__':
    main()
