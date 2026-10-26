"""Score a completed V7 TCR prediction run with the existing sequence metrics.

Activate conda pllm and use the worker-compatible Python in a clean process:
python -I -B /project_c20250601/251105016/project/dllm_test/scripts/downstream/score_v7_tcr_generation.py --plan /absolute/plan.json --output-root /absolute/new_directory
This CPU-only entry reads saved predictions. It does not generate, annotate,
load weights, train, retry, submit jobs, or change benchmark membership.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import socket
import sys

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))
from downstream.v7_native import require, sha
from downstream.benchmark.common.reporting import object_sha256


def read(path):
    return json.loads(path.read_text())


def reference(path):
    require(path.is_absolute() and path == path.resolve() and path.is_file()
            and not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical file required')
    return dict(path=str(path), sha256=sha(path))


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def verify_predictions(plan):
    """Bind real runner exit, sources, raw candidates, and the scoring export."""
    for item in plan['sources'] + plan['data_files']:
        require(reference(Path(item['path'])) == item, 'Scoring dependency changed')
    runner = Path(plan['prediction_runner']['path'])
    require(reference(runner) == plan['prediction_runner'], 'Prediction entrypoint changed')
    logs = Path(plan['prediction_log_directory'])
    require((logs/'done').is_file() and not (logs/'failed').exists()
            and (logs/'exit_code').is_file() and (logs/'exit_code').read_text().strip() == '0',
            'Actual successful prediction exit is still required')
    require(socket.gethostname() == plan['prediction_host']
            and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == plan['prediction_boot_id'],
            'Prediction process identity must be checked on its original host and boot')
    proc = Path('/proc')/str(plan['prediction_pid'])/'stat'
    try:
        ticks = int(proc.read_text().rsplit(')', 1)[1].split()[19])
    except (FileNotFoundError, ProcessLookupError):
        ticks = None
    require(ticks != plan['prediction_start_ticks'], 'Original prediction process still active')
    api = runpy.run_path(str(runner), run_name='_saved_v7_tcr_artifact_reader')
    entries, binding = api['API']['bind_inputs']()
    output = Path(plan['prediction_directory'])
    api['API']['check_stop'](output)
    require(not (output/'FAILURE.json').exists(), 'Prediction failure exists')
    identity = read(output/'identity.json')
    require(identity['binding'] == binding and identity['code'] == api['API']['code_identity']()
            and identity['protocol'] == api['PROTOCOL']
            and identity['entrypoint'] == plan['prediction_runner']
            and identity['checkpoint'] == reference(api['API']['CHECKPOINT']/'formal_checkpoint.json'),
            'Prediction identity changed')
    complete = read(output/'PREDICTIONS_COMPLETE.json')
    require(complete['status'] == 'full_tcr_predictions_complete_scoring_pending'
            and complete['identity_sha256'] == object_sha256(identity)
            and complete['predictions'] == reference(output/'predictions.jsonl')
            and complete['designs'] == reference(output/'designs.jsonl'), 'Unsealed or changed predictions')
    rows = [json.loads(s) for s in (output/'predictions.jsonl').read_text().splitlines()]
    validation = api['validate_rows'](rows, identity, entries)
    require(all(complete[k] == v for k, v in validation.items() if k != 'status'),
            'Prediction completion counts differ')
    designs = [json.loads(s) for s in (output/'designs.jsonl').read_text().splitlines()]
    require([d['pmhc'] for d in designs] == list(entries), 'Scoring targets omitted or reordered')
    for i, (design, (key, entry)) in enumerate(zip(designs, entries.items())):
        group = rows[i*101:(i+1)*101]
        expected = dict(pmhc=key, target_index=i, category=entry['category'], epitope=entry['epitope'],
            mhc_pseudo=entry['mhc_pseudo'], sequences=[r['sequence'] for r in group[:100]],
            greedy=group[100]['sequence'], n_requested=100, n_raw=100,
            n_syntactically_valid=sum(r['syntactically_valid'] for r in group[:100]),
            n_terminated=sum(r['terminated_by_protd'] for r in group[:100]),
            protocol=api['PROTOCOL_ID'], identity_sha256=object_sha256(identity))
        require(design == expected, 'Scoring export changed candidates, controls, or metadata')
    return identity, validation, complete, entries, designs


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(); plan_ref = reference(args.plan); plan = read(args.plan)
    require(plan['schema'] == 'v7.tcr.saved_sequence_scorer.v1'
            and plan['entrypoint'] == reference(Path(__file__).resolve()), 'Unexpected scoring plan')
    for key, value in plan['environment'].items():
        require(os.environ.get(key) == value, 'Scoring environment differs: '+key)
    output = args.output_root
    require(output.is_absolute() and output == output.resolve() and not output.exists()
            and output.is_relative_to(ROOT/'output/downstream_generation')
            and not any(p.is_symlink() for p in (output, *output.parents)), 'New canonical output required')
    identity, validation, complete, entries, designs = verify_predictions(plan)
    output.mkdir(parents=True, exist_ok=False)
    started = dict(at_utc=datetime.now(timezone.utc).isoformat(), plan=plan_ref,
        prediction_completion=reference(Path(plan['prediction_directory'])/'PREDICTIONS_COMPLETE.json'),
        prediction_identity=identity, candidate_validation=validation,
        packages={n:importlib.metadata.version(n) for n in ('numpy', 'pandas', 'scikit-learn')})
    write(output/'started.json', started)
    try:
        from downstream.v7_tcr_scoring import score_saved_candidates
        metrics = score_saved_candidates({d['pmhc']:d['sequences'] for d in designs},
                                         {d['pmhc']:d['greedy'] for d in designs}, entries)
        metrics.update(candidate_validation=validation, prediction_identity_sha256=complete['identity_sha256'],
                       evaluation_protocol=identity['protocol'], scoring_plan=plan_ref,
                       packages=started['packages'], model_loaded=False, annotation_called=False)
        for item in plan['sources'] + plan['data_files']:
            require(reference(Path(item['path'])) == item, 'Scoring dependency changed during scoring')
        require(reference(args.plan) == plan_ref, 'Scoring plan changed during scoring')
        for item in (complete['predictions'], complete['designs']):
            require(reference(Path(item['path'])) == item, 'Predictions changed during scoring')
        write(output/'metrics.json', metrics)
        write(output/'COMPLETE.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
            status='all_saved_tcr_candidates_scored', metrics=reference(output/'metrics.json'),
            prediction_completion=started['prediction_completion'], plan=plan_ref,
            all_targets_retained=len(metrics['per_pmhc']) == 34,
            primary_coverage_complete_by_view=metrics['metric_availability']['primary_coverage_complete_by_view']))
        print(json.dumps(dict(status='all_saved_tcr_candidates_scored', output=str(output))), flush=True)
    except BaseException as error:
        write(output/'FAILURE.json', dict(at_utc=datetime.now(timezone.utc).isoformat(), error=repr(error)))
        raise


if __name__ == '__main__':
    main()
