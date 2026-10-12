"""Plot existing R20-R27 artificial 500-step evidence without changing experiments.

Activate pllm, then run python scripts/plot_immune_refined_progress.py
  --pool-dir /absolute/pool --output-prefix /absolute/figure/prefix
Writes SVG/PNG and a JSON provenance sidecar. No model, GPU or checkpoint is used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.immune_research_supervision import inspect, grouped_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool-dir', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    state = inspect(args.pool_dir)
    records = [(job, value) for job, value in state['records'].items()
               if value.get('requested_steps') == 500 and value.get('variant', '').startswith(tuple(f'r{i}_' for i in range(20, 28)))]
    assert len(records) == 8 and len({value['variant'] for _, value in records}) == 8, 'Expected one current 500-step job per variant'
    series = []
    for job, record in sorted(records, key=lambda item: item[1]['variant']):
        assert record['source_status'] == 'matched' and record['kind'] == 'synthetic_generalization_diagnostic'
        points = [dict(step=0, **record['initial_metrics']), *record['trace']]
        rows = []
        for point in points:
            groups = grouped_metrics(point)
            row = dict(step=point['step'], mean_loss=point['mask_mean']['loss'],
                mean_aa=point['mask_mean']['aa_accuracy'],
                partial_aa=sum(point['mask_levels'][mask]['aa_accuracy'] for mask in ('mask25', 'mask50', 'mask75'))/3,
                conditional_full_aa=groups['conditional']['mask_levels']['mask100']['aa_accuracy'],
                unconditional_full_aa=groups['unconditional']['mask_levels']['mask100']['aa_accuracy'],
                mean_end=point['mask_mean']['terminal_accuracy'])
            assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in row.values())
            rows.append(row)
        source = args.pool_dir/'runs'/job/'summary.json'
        series.append(dict(job_id=job, variant=record['variant'], state=record['state'], points=rows,
            source_summary=str(source.resolve()), summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_root=record['source_root'], source_status=record['source_status']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['R20 Control', 'R21 Head gate', 'R22 Fixed gate', 'R23 Learnable scale',
              'R24 Fixed scale', 'R25 Late fusion', 'R26 Local/global', 'R27 Value residual']
    panels = [('mean_loss', 'Four-mask mean loss', False),
              ('mean_aa', 'Four-mask mean AA accuracy', True),
              ('conditional_full_aa', 'Conditional full-mask AA accuracy', True),
              ('unconditional_full_aa', 'Unconditional full-mask AA accuracy', True),
              ('partial_aa', 'Partial-mask AA accuracy (25/50/75%)', True),
              ('mean_end', 'End recall at true terminal positions (four-mask mean)', True)]
    figure, axes = plt.subplots(3, 2, figsize=(12, 10.5))
    for axis, (metric, title, accuracy) in zip(axes.flat, panels):
        for index, item in enumerate(series):
            axis.plot([point['step'] for point in item['points']],
                      [point[metric]*(100 if accuracy else 1) for point in item['points']],
                      color=plt.cm.tab10(index), marker='o', markersize=3, linewidth=1.5, label=labels[index])
        axis.set_title(title, fontsize=11)
        axis.set_xlabel('Optimizer step')
        axis.set_ylabel(('End recall (%)' if metric == 'mean_end' else 'Accuracy (%)') if accuracy else 'Cross-entropy')
        if accuracy:
            axis.set_ylim(-2, 102)
        else:
            axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
    minimum = min(item['points'][-1]['step'] for item in series)
    maximum = max(item['points'][-1]['step'] for item in series)
    steps = str(minimum) if minimum == maximum else f'{minimum}-{maximum}'
    figure.suptitle(f'Artificial heldout diagnostic: recorded step {steps}/500', fontsize=14, y=.98)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, .025), fontsize=9)
    figure.text(.5, .012, '576 artificial train / 72 heldout records; one seed; not biological training or a downstream quality claim.', ha='center', fontsize=8, color='#555555')
    figure.tight_layout(rect=(0, .1, 1, .95))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('svg', 'png'):
        figure.savefig(args.output_prefix.with_suffix('.'+extension), dpi=150)
    plt.close(figure)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), kind='synthetic_generalization_diagnostic',
                  biological_quality_claim=False, figure_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), series=series)
    args.output_prefix.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(recorded_steps=steps, svg=str(args.output_prefix.with_suffix('.svg').resolve()),
                         png=str(args.output_prefix.with_suffix('.png').resolve()))))


if __name__ == '__main__':
    main()
