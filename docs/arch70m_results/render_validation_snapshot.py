"""Render a read-only metrics snapshot with the exploration Python environment.

Run: python /absolute/path/render_validation_snapshot.py --output-dir /absolute/path/snapshot
Re-render: append --snapshot /absolute/path/snapshot/snapshot.json
This standalone report does not import or modify the frozen training implementation.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

DEFAULT_ROOT = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_v8_20261028')
NAMES = {
    'A0': 'Baseline', 'A1': 'QK norm', 'A2': 'Attention gate',
    'A3': 'QK norm + gate', 'A4': 'Learnable residual scale',
    'A5': 'All-global attention', 'A6': 'Tapered FFN', 'A7': 'Constant 0.5 gate',
}
COLORS = dict(zip(NAMES, ['#243447', '#4477AA', '#228833', '#AA3377', '#EE7733', '#CC3311', '#66AABB', '#999999']))
FIELDS = {
    'validation_ce': ('token', 'all', 'loss'),
    'cdr_ce': ('generation', 'cdr_ce'), 'cdr_aar': ('generation', 'cdr_aar'),
    'conditional_ce': ('generation', 'conditional_ce'),
    'conditional_aar': ('generation', 'conditional_aar'),
    'conditional_validity': ('generation', 'conditional_validity'),
    'conditional_length_mae': ('generation', 'conditional_length_mae'),
    'auroc': ('representation', 'macro_auroc'), 'auprc': ('representation', 'macro_auprc'),
}


def get(row, keys):
    for key in keys:
        if not isinstance(row, dict) or key not in row:
            return None
        row = row[key]
    return row


def capture(root):
    snapshot = {'captured_at_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
                'protocol': {'seed': 42, 'epochs': 1, 'train_records': 653288, 'global_batch': 512,
                             'aggregation': 'Generation and representation metrics are macro averages over sources.'},
                'trials': {}, 'source_file_sha256': {}}
    def read_bytes(path):
        data = path.read_bytes()
        snapshot['source_file_sha256'][str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
        return data
    def jsonl(path, step):
        rows = []
        for line in read_bytes(path).splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                break
            if value.get('step', 0) <= step:
                rows.append(value)
        return rows
    for variant in NAMES:
        folder = root / 'trials' / f'screen_{variant}_s42'
        if not (folder / 'status.json').exists():
            continue
        status = json.loads(read_bytes(folder / 'status.json'))
        binding = json.loads(read_bytes(folder / 'binding.json'))
        train = jsonl(folder / 'train.jsonl', status['step'])
        evals = jsonl(folder / 'eval.jsonl', status['step'])
        points = []
        for row in evals:
            if row['partition'] != 'screen':
                continue
            p = {'step': row['step']}
            p.update({field: get(row, path) for field, path in FIELDS.items() if get(row, path) is not None})
            if 'generation' in row:
                p['generation_sources'] = row['generation'].get('sources', {})
                p['representation_sources'] = row.get('representation', {}).get('sources', {})
            points.append(p)
        final = next((p for p in reversed(points) if p['step'] == status.get('total_steps') and 'auroc' in p), None)
        if status['status'] == 'completed':
            assert final is not None, f'{variant}: completed status without final task evaluation'
            assert len(train) == status['total_steps'], f'{variant}: missing training observations'
        snapshot['trials'][variant] = {
            'name': NAMES[variant], 'status': status,
            'binding': {k: binding.get(k) for k in ('source_sha256', 'data_sha256')},
            'train': [{k: r[k] for k in ('step', 'loss', 'seen_tokens', 'gpu_hours')} for r in train],
            'eval': points, 'final': final,
        }
    assert len({t['binding']['source_sha256'] for t in snapshot['trials'].values()}) == 1
    assert len({t['binding']['data_sha256'] for t in snapshot['trials'].values()}) == 1
    return snapshot


def save(fig, folder, name):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(folder / f'{name}.{ext}', dpi=180, facecolor='white')
    plt.close(fig)


def style(ax):
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#D1D8E0')
    ax.spines['bottom'].set_color('#D1D8E0')
    ax.tick_params(colors='#4B5563', labelsize=10)
    ax.grid(axis='x', color='#E8EDF2', linewidth=.7)
    ax.set_axisbelow(True)


def render(snapshot, folder):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titleweight': 'bold', 'axes.labelcolor': '#334155',
                         'text.color': '#243447', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'arch70m-validation-snapshot', 'pdf.fonttype': 42})
    trials = snapshot['trials']
    completed = [v for v, t in trials.items() if t['status']['status'] == 'completed' and t['final']]
    base = trials['A0']['final']
    at = datetime.fromisoformat(snapshot['captured_at_utc']).astimezone(timezone(timedelta(hours=8)))
    asof = at.strftime('%Y-%m-%d %H:%M UTC+8')
    running = ', '.join(f"{v}: {t['status']['step']}/{t['status']['total_steps']}" for v,t in trials.items() if t['status']['status'] != 'completed')
    completed_label = ', '.join(completed)
    metrics = [
        ('conditional_aar', 'Whole-chain generation: AAR', 'Higher is better | %', 100, '.2f'),
        ('conditional_length_mae', 'Whole-chain generation: length MAE', 'Lower is better | residues; valid outputs only', 1, '.2f'),
        ('auroc', 'Representation: macro AUROC', 'Higher is better', 1, '.4f'),
        ('validation_ce', 'Validation: selected-token CE', 'Lower is better', 1, '.4f'),
        ('cdr_aar', 'CDR infilling: AAR', 'Higher is better | %', 100, '.2f'),
        ('auprc', 'Representation: macro AUPRC', 'Higher is better', 1, '.4f'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(left=.06, right=.96, top=.79, bottom=.17, wspace=.40, hspace=.48)
    fig.text(.06, .958, 'Architecture screening | completed-run comparison', size=22, weight='bold')
    fig.text(.06, .92, f'{asof}  |  {completed_label}: final step 1276  |  same 10% V8 subset  |  seed 42', size=11)
    fig.text(.06, .883, 'A0 Baseline     A1 QK norm     A2 Attention gate     A3 QK norm + gate', size=11)
    fig.text(.06, .855, 'A4 Learnable residual scale     A5 All-global attention     A6 Tapered FFN' + ('     A7 Constant 0.5 gate' if 'A7' in completed else ''), size=11)
    for ax, (key, title, unit, scale, fmt) in zip(axes.flat, metrics):
        values = np.array([trials[v]['final'][key] * scale for v in completed])
        ref = base[key] * scale
        span = max(float(np.ptp(values)), .001)
        for y, (v, value) in enumerate(zip(completed, values)):
            ax.plot([ref, value], [y, y], color=COLORS[v], lw=2.5, alpha=.55)
            ax.scatter(value, y, s=65 if v != 'A0' else 85, c=COLORS[v], marker='D' if v == 'A0' else 'o', zorder=3)
            ax.text(value + span * .04, y, format(value, fmt), va='center', size=10, weight='bold' if v in ('A0', 'A5') else 'normal')
        ax.axvline(ref, color=COLORS['A0'], ls='--', lw=1, alpha=.5)
        ax.set_yticks(range(len(completed)), completed)
        ax.set_ylim(len(completed)-.5, -.6)
        ax.set_xlim(values.min()-span*.13, values.max()+span*.43)
        ax.set_title(title, loc='left', fontsize=12, pad=13)
        ax.set_xlabel(unit, fontsize=10, labelpad=9)
        style(ax)
    fig.text(.06, .075, 'Dashed line = A0 baseline. Each panel uses its own linear scale. Only completed trials appear in this comparison.', size=10)
    validity = ', '.join(f"{v} {trials[v]['final']['conditional_validity']*100:.1f}%" for v in completed)
    fig.text(.06, .049, 'Source-macro averages. Generation validity: ' + validity + '.', size=10)
    fig.text(.06, .023, 'Exploratory, single seed: no confidence intervals or confirmed winners. Protein sequence similarity is not experimental function.', size=10, color='#657386')
    save(fig, folder, 'final_comparison')

    curve_specs = [
        ('validation_ce', 'Validation selected-token CE', 'CE | lower is better', 1),
        ('conditional_ce', 'Whole-chain generation CE', 'CE | lower is better', 1),
        ('cdr_ce', 'CDR infilling CE', 'CE | lower is better', 1),
        ('conditional_aar', 'Whole-chain generation AAR', 'AAR (%) | higher is better', 100),
        ('cdr_aar', 'CDR infilling AAR', 'AAR (%) | higher is better', 100),
        ('auroc', 'Frozen representation probe', 'Macro AUROC | higher is better', 1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.subplots_adjust(left=.065, right=.97, top=.79, bottom=.15, wspace=.27, hspace=.40)
    fig.text(.065, .958, 'Validation trajectories | all recorded evaluation points', size=22, weight='bold')
    fig.text(.065, .92, f'{asof}  |  solid: completed trials  |  dashed: running {running or "none"}', size=12)
    handles = [Line2D([0],[0], color=COLORS[v], lw=2.5, ls='--' if t['status']['status']!='completed' else '-', label=f'{v} {NAMES[v]}') for v,t in trials.items()]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.06,.9), ncol=4, frameon=False, fontsize=10)
    for ax, (key, title, label, scale) in zip(axes.flat, curve_specs):
        for v,t in trials.items():
            pts = [p for p in t['eval'] if key in p]
            ax.plot([p['step'] for p in pts], [p[key]*scale for p in pts], color=COLORS[v],
                    ls='--' if t['status']['status']!='completed' else '-', lw=2 if v in ('A0','A5') else 1.3,
                    marker='o', markersize=3, alpha=.9)
        ax.set_title(title, loc='left', size=12)
        ax.set_xlabel('Optimizer step')
        ax.set_ylabel(label, size=10)
        ax.set_xlim(0,1276)
        ax.set_xticks([0,250,500,750,1000,1276])
        style(ax)
        ax.grid(axis='y', color='#E8EDF2', linewidth=.7)
    fig.text(.065, .054, 'Unsmoothed metrics, including step 0. Lines connect measurements only; unfinished runs stop at their latest evaluation.', size=10)
    fig.text(.065, .028, 'Compare architectures at matched steps. Unfinished trials are shown for progress only; final selection waits for all eight.', size=10, color='#657386')
    save(fig, folder, 'validation_curves')

    columns = ['variant','architecture','step', *FIELDS]
    with (folder/'final_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        for v in completed:
            final=trials[v]['final']
            writer.writerow(dict(variant=v, architecture=NAMES[v], step=final['step'], **{k:final[k] for k in FIELDS}))
    with (folder/'evaluation_curves.csv').open('w') as f:
        writer=csv.DictWriter(f,['variant','status','step',*FIELDS])
        writer.writeheader()
        for v,t in trials.items():
            for point in t['eval']:
                writer.writerow(dict(variant=v,status=t['status']['status'],step=point['step'],**{k:point.get(k) for k in FIELDS}))
    print(json.dumps({'captured_at':asof,'completed':completed,'running':running,'output':str(folder)}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path)
    args=parser.parse_args()
    assert args.output_dir.is_absolute() and args.root.is_absolute()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    snapshot=json.loads(args.snapshot.read_text()) if args.snapshot else capture(args.root)
    if not args.snapshot:
        (args.output_dir/'snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    render(snapshot,args.output_dir)
