"""Verify/reuse unchanged predictions and rescore restored raw rows on CPU.

Run: python SCRIPT --parent ABS [--watch]. Never starts GPU inference.
Watch imports the two already-running old assays after their successful exit.
"""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))
from scripts.downstream.run_ab_ag_corrected import report, sha, stamp, write
from downstream.benchmark.ab_antigen.abbibench_scope import ASSAYS, summarize_groups


def inventory(path):
    hashes = defaultdict(hashlib.sha256)
    counts = defaultdict(int)
    selected = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            key = row['assay'] if row['dataset'] == 'abbibench' else row['dataset']
            hashes[key].update((json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode())
            counts[key] += 1
            if key == '1mhp_LC_benchmarking_data':
                selected.append(row)
    return dict(sha256=sha(path), per_assay={k: dict(rows=counts[k], sha256=v.hexdigest())
                                          for k, v in hashes.items()}, mhp_rows=selected)


def prepare_audit(parent):
    root = ROOT / 'data/downstream'
    audit = {}
    for task in ('spr', 'abbibench'):
        a = root / 'ab_antigen_sequence_v1' / (task + '.jsonl')
        b = root / 'ab_antigen_sequence_v2_raw_rows' / (task + '.jsonl')
        audit[task] = dict(old=inventory(a), new=inventory(b), old_path=str(a), new_path=str(b))
    write(parent / 'reuse_data_audit.json', audit)
    return audit


def input_key(row):
    return tuple(row[k] for k in ('assay', 'antigen', 'heavy', 'light', 'wt_heavy', 'wt_light'))


def reconcile(run, audit):
    plan = json.loads((run / 'plan.json').read_text())
    old = Path(plan['old_run'])
    changed = []
    for task in ['ag-spr'] + ['ag-abbibench:' + a for a in ASSAYS]:
        if task in plan['gpu_tasks'] or (run / task / 'status.json').exists():
            continue
        source = old / task
        status = source / 'status.json'
        if not status.exists() or json.loads(status.read_text()).get('status') != 'success':
            continue
        metric_path = source / 'metrics.json'
        metrics = json.loads(metric_path.read_text())
        if metrics.get('status') != 'complete' or metrics['native_identity']['model_sha256'] != plan['model_sha256']:
            raise ValueError('Incomplete or wrong-checkpoint source: ' + str(metric_path))
        provenance = json.loads((source / 'provenance.json').read_text())
        name = 'spr' if task == 'ag-spr' else 'abbibench'
        key = 'spr' if name == 'spr' else task.partition(':')[2]
        data = audit[name]
        if provenance['input_sha256'] != data['old']['sha256']:
            raise ValueError('Original dataset differs from evaluated dataset')
        if plan['data_sha256'][data['new_path']] != data['new']['sha256']:
            raise ValueError('New dataset differs from planned dataset')
        a, b = data['old']['per_assay'][key], data['new']['per_assay'][key]
        if metrics['n'] != a['rows']:
            raise ValueError('Original result does not cover the complete assay')
        destination = run / task
        destination.mkdir(parents=True, exist_ok=True)
        proof = dict(model_sha256=plan['model_sha256'], original_metrics=str(metric_path),
                     original_metrics_sha256=sha(metric_path), original_data=a, current_data=b,
                     current_dataset_path=data['new_path'], current_dataset_sha256=data['new']['sha256'],
                     original_predictions=str(source / 'predictions.jsonl'), model_inference_run=False)
        result = copy.deepcopy(metrics)
        if a == b:
            proof['execution'] = 'reuse_verified_identical_rows'
        elif key == '1mhp_LC_benchmarking_data':
            original = source / 'predictions.jsonl'
            prediction_rows = [json.loads(l) for l in original.open()]
            old_refs = {r['id']: r for r in data['old']['mhp_rows']}
            if {r['id'] for r in prediction_rows} != set(old_refs) or len(prediction_rows) != len(old_refs):
                raise ValueError('Incomplete/duplicate source predictions')
            cache = {}
            for row in prediction_rows:
                if any(row.get(k) != v for k, v in old_refs[row['id']].items()):
                    raise ValueError('Prediction/reference data mismatch')
                cache_key = input_key(row)
                if cache_key in cache and cache[cache_key] != row['score']:
                    raise ValueError('Conflicting scores for identical model inputs')
                cache[cache_key] = row['score']
            scored = [{**r, 'score': cache[input_key(r)], 'features': None} for r in data['new']['mhp_rows']]
            from downstream.benchmark.ab_antigen.sequence_eval import group_metrics
            result['n'] = len(scored)
            result['per_assay'] = group_metrics(scored, group_key='assay')
            with (destination / 'predictions.jsonl').open('x') as f:
                for row in scored:
                    f.write(json.dumps(row) + '\n')
            proof.update(execution='rescore_all_raw_rows_from_verified_identical_model_inputs',
                         original_predictions_sha256=sha(original), restored_rows=b['rows'] - a['rows'])
        else:
            raise ValueError('Changed assay requires fresh inference: ' + key)
        result['data_version'] = 'ab_antigen_sequence_v2_raw_rows'
        result['reuse_provenance'] = proof
        if name == 'abbibench':
            result['benchmark16'] = summarize_groups(result['per_assay'])
        write(destination / 'provenance.json', proof)
        write(destination / 'metrics.json', result)
        write(destination / 'status.json', dict(status='success', task=task, execution=proof['execution'], finished_at=stamp()))
        changed.append(task)
    summary = report(run)
    return dict(run=str(run), changed=changed, counts=summary['counts'],
                abbibench_completed=summary['abbibench_benchmark16']['completed_assays'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    audit = prepare_audit(args.parent)
    models = json.loads((args.parent / 'launch_plan.json').read_text())['models']
    deadline = time.monotonic() + 604800
    while True:
        updates = [reconcile(Path(m['run']), audit) for m in models]
        write(args.parent / 'reconciliation_status.json', dict(updated_at=stamp(), models=updates))
        print(json.dumps(updates, ensure_ascii=False), flush=True)
        waiting = []
        for m in models:
            run = Path(m['run']);plan = json.loads((run / 'plan.json').read_text())
            waiting.extend(t for t in plan['waiting_existing'] if not (run / t / 'status.json').exists())
        if not args.watch or not waiting:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Existing assay jobs have not completed; see reconciliation_status.json')
        time.sleep(60)


if __name__ == '__main__':
    main()
