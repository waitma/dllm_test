"""Activate pllm; python -B FILE --out ABS after ablation_delivery.json is complete.

Plot the six prespecified primary H3 component contrasts from the final delivery.
Writes primary_component_effects.svg and .png; never reads partial evaluations,
selects policies, or changes the frozen experiment protocol.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


FROZEN_PLAN_SHA256 = '60b4ea16fe684c03098b5bd8c283035dfd3835a4cc915dfbfb0a7197698cb22e'
CHECKPOINTS = ('112000', '50000')
CONTRASTS = (
    ('soft_distribution', 'Soft distribution\nsoft vs. hard context'),
    ('independent_feedback', 'Independent feedback\nfrozen vs. sequential'),
    ('single_pass_anchor', 'Single-pass anchor\nweight 0.5 vs. 0'),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_completed_delivery(out):
    path = out / 'ablation_delivery.json'
    if not path.exists():
        raise FileNotFoundError('Plot only after the final ablation_delivery.json is available')
    delivery = json.loads(path.read_text())
    assert delivery['status'] == 'complete' and delivery['winner_selected'] is False
    assert delivery['outcome'] == 'all_prespecified_ablations_reported'
    assert delivery['evidence_sha256'][str(out / 'plan.json')] == FROZEN_PLAN_SHA256
    assert sha(out / 'plan.json') == FROZEN_PLAN_SHA256
    assert delivery['bootstrap']['family_size'] == 6
    rows = delivery['primary_h3']
    assert len(rows) == 6
    expected = {(checkpoint, name) for checkpoint in CHECKPOINTS for name, _ in CONTRASTS}
    indexed = {}
    for row in rows:
        key = (row['checkpoint'], row['contrast'])
        assert key in expected and key not in indexed
        assert row['mode'] == 'cdrh3' and row['role'] == 'primary'
        assert row['bonferroni_family_tests'] == 6
        assert math.isclose(row['simultaneous_interval_confidence'], 1 - .05 / 6)
        low95, high95 = row['ci95_pp']
        low_adjusted, high_adjusted = row['ci_bonferroni_pp']
        assert all(math.isfinite(value) for value in (row['delta_pp'], low95, high95, low_adjusted, high_adjusted))
        assert low_adjusted <= low95 <= high95 <= high_adjusted
        indexed[key] = row
    assert set(indexed) == expected
    return path, delivery, indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    delivery_path, delivery, rows = load_completed_delivery(out)

    # Import plotting dependencies only after the final delivery has passed its
    # completion checks; this script does not inspect running-job artifacts.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {'112000': '#0072B2', '50000': '#D55E00'}
    offsets = {'112000': .14, '50000': -.14}
    labels = [label for _, label in CONTRASTS]
    centers = [2, 1, 0]
    limits = [0.]
    for row in rows.values():
        limits.extend(row['ci_bonferroni_pp'])
        limits.append(row['delta_pp'])
    low, high = min(limits), max(limits)
    padding = max((high - low) * .12, .025)
    title = 'CDR-H3 component ablation'
    cohort_size = delivery['counts']['counts_by_mode']['cdrh3']
    description = (
        'Full-minus-ablated AAR differences in percentage points. Thick lines: '
        'unadjusted 95% paired connected-component bootstrap intervals. Thin '
        'lines: 99.1667% Bonferroni-adjusted interval per contrast; six tests '
        'form a nominal 95% confidence family. Positive favors the full method. '
        f'Source delivery SHA256: {sha(delivery_path)}. '
        f'Plot source SHA256: {sha(Path(__file__))}.')

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.unicode_minus': False}):
        fig, ax = plt.subplots(figsize=(10, 5.8))
        fig.subplots_adjust(left=.30, right=.97, top=.73, bottom=.27)
        for center in (0, 2):
            ax.axhspan(center - .43, center + .43, color='#F4F6F8', zorder=0)
        ax.axvline(0, color='#444444', linewidth=1.1, linestyle='--', zorder=1)
        for center, (name, _) in zip(centers, CONTRASTS):
            for checkpoint in CHECKPOINTS:
                row = rows[(checkpoint, name)]
                y = center + offsets[checkpoint]
                color = colors[checkpoint]
                ax.hlines(y, *row['ci_bonferroni_pp'], color=color, linewidth=1.3, zorder=2)
                ax.hlines(y, *row['ci95_pp'], color=color, linewidth=5, zorder=3)
                ax.plot(row['delta_pp'], y, 'o', color=color, markersize=7,
                        markeredgecolor='white', markeredgewidth=.9, zorder=4)
        ax.set_xlim(low - padding, high + padding)
        ax.set_ylim(-.5, 2.5)
        ax.set_yticks(centers, labels)
        ax.tick_params(axis='y', length=0, pad=12)
        ax.tick_params(axis='x', labelsize=9)
        ax.set_xlabel('AAR difference: full - ablated (percentage points)', labelpad=10)
        ax.text(.99, 1.045, 'Positive favors full', transform=ax.transAxes,
                ha='right', va='bottom', color='#333333', fontsize=10)
        ax.grid(axis='x', color='#E2E5E9', linewidth=.65, zorder=0)
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#C2C7CE')
        fig.suptitle(title, x=.30, y=.975, ha='left', fontsize=17, fontweight='bold')
        fig.text(.30, .918, f'OAS supplementary cohort | {cohort_size:,} sequences per checkpoint',
                 ha='left', fontsize=10, color='#555555')
        handles = [Line2D([0], [0], color=colors[c], marker='o', linewidth=1.5,
                          markersize=6, label=f'Checkpoint {int(c):,}') for c in CHECKPOINTS]
        handles += [Line2D([0], [0], color='#555555', linewidth=5, label='95% interval'),
                    Line2D([0], [0], color='#555555', linewidth=1.3, label='99.1667% adjusted interval')]
        fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.29, .887),
                   ncol=2, frameon=False, fontsize=9, handlelength=2.6, columnspacing=2.)
        fig.text(.30, .105,
                 'Thin: 99.1667% Bonferroni-adjusted interval per contrast\n'
                 '(6 prespecified tests; nominal family confidence 95%).\n'
                 'Thick: unadjusted 95% paired connected-component bootstrap interval.',
                 ha='left', va='center', fontsize=9, color='#555555', linespacing=1.45)
        fig.savefig(out / 'primary_component_effects.svg',
                    metadata={'Title': title, 'Description': description,
                              'Creator': Path(__file__).name, 'Date': None})
        fig.savefig(out / 'primary_component_effects.png', dpi=240,
                    metadata={'Title': title, 'Description': description,
                              'Software': Path(__file__).name})
        plt.close(fig)
    print(json.dumps(dict(status='plotted', source_delivery_sha256=sha(delivery_path),
        outputs={name: sha(out / name) for name in ('primary_component_effects.svg', 'primary_component_effects.png')})))


if __name__ == '__main__':
    main()
