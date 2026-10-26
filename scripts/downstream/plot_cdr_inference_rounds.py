"""Activate pllm; python -B FILE --out ABS to export completed-round CSV and H3 plots."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    assert json.loads((out/'completion_audit.json').read_text())['status'] == 'completed'
    tables = json.loads((out/'review_tables.json').read_text())
    with (out/'all_metrics.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['round', 'checkpoint', 'dataset', 'mode', 'policy', 'aar_percent',
                         'mean_forward_calls_all_tasks', 'sampler_seconds_all_tasks'])
        for n, cases in tables.items():
            for policy, values in cases.items():
                for key, score in values['scores'].items():
                    dataset, mode = key.split(':')
                    writer.writerow([n, 112000 if n == '3' else 50000, dataset, mode, policy, score,
                                     values['mean_actual_forwards'], values['seconds']])
    names = ['original4', 'original8', 'logprob4', 'entropy_0.5', 'repair8_0.25']
    labels = ['Original\n4 steps', 'Original\n8 steps', 'Log-prob\n4 steps', 'Entropy\nmax 8', 'Repair\n8 steps']
    colors = ['#667085', '#667085', '#2374ab', '#087e8b', '#8b5fbf']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for row, n in enumerate((2, 3)):
        audit = json.loads((out/f'round{n}/independent_audit.json').read_text())
        for col, dataset in enumerate(('kong', 'sab23')):
            ax = axes[row, col]
            for i, (name, color) in enumerate(zip(names, colors)):
                c = audit['comparisons'][name+':'+dataset+':cdrh3']
                delta = c['delta_vs_original1_pp']; low, high = c['paired_HL_group_bootstrap95ci_pp']
                ax.errorbar(i, delta, yerr=[[delta-low], [high-delta]], fmt='o', color=color,
                            markersize=6, capsize=4, linewidth=1.6)
            ax.axhline(0, color='#222222', linewidth=1, linestyle='--')
            ax.set_xticks(range(len(names)), labels, fontsize=8)
            count = 2960 if dataset == 'kong' else 60
            checkpoint = 50000 if n == 2 else 112000
            ax.set_title(f'{dataset.upper()} H3 | checkpoint {checkpoint} | n={count}', fontsize=10)
            ax.grid(axis='y', alpha=.2)
            if col == 0: ax.set_ylabel('AAR difference from 1 step (pp)')
    fig.suptitle('Frozen-policy confirmation: H3 recovery versus one-pass argmax', fontsize=13)
    fig.text(.5, .015, '95% paired H/L-group bootstrap intervals. Internal benchmark partition; exploratory, unadjusted comparisons.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(out/'h3_confirmation.png', dpi=180)
    fig.savefig(out/'h3_confirmation.svg')
    plt.close(fig)
    print(out/'h3_confirmation.png')


if __name__ == '__main__': main()
