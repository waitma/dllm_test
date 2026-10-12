"""Score a completed native V7 pairing run, without generating or training.

Activate conda pllm; use the worker-compatible Python with offline model paths:
python -I -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/score_v7_pairing.py --plan /absolute/plan.json --output-root /absolute/new_directory
The plan binds the existing scorer and local ImmunoMatch weights. The prediction
runner must have exited successfully. No retries, downloads, or weight copies.
"""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import sys

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))
from downstream.v7_native import sha, require


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
    """Verify actual exit and every raw candidate before importing the scorer."""
    for item in plan['sources']+plan['model_files']:
        require(reference(Path(item['path'])) == item, 'Scoring dependency changed')
    source = Path(plan['prediction_runner']['path'])
    require(reference(source) == plan['prediction_runner'], 'Prediction runner changed')
    logs = Path(plan['prediction_log_directory'])
    require((logs/'done').is_file() and not (logs/'failed').exists()
            and (logs/'exit_code').is_file() and (logs/'exit_code').read_text().strip() == '0',
            'Actual successful prediction exit is still required')
    proc = Path('/proc')/str(plan['prediction_pid'])
    if (proc/'stat').exists():
        try:
            ticks = int((proc/'stat').read_text().rsplit(')', 1)[1].split()[19])
        except (FileNotFoundError, ProcessLookupError):
            ticks = None
        require(ticks != plan['prediction_start_ticks'], 'Original prediction process still active')
    api = runpy.run_path(str(source), run_name='_v7_pairing_prediction_artifact_reader')
    records, binding = api['bind_inputs']()
    output = Path(plan['prediction_directory'])
    api['stopped'](output)
    identity = read(output/'identity.json')
    require(identity['binding'] == binding and identity['code'] == api['code_identity']()
            and identity['protocol'] == api['PROTOCOL'], 'Prediction identity changed')
    complete = read(output/'PREDICTIONS_COMPLETE.json')
    require(complete['status'] == 'full_predictions_complete_scoring_pending'
            and reference(output/'predictions.jsonl') == complete['predictions']
            and reference(output/'holdout500_n8.csv') == complete['scoring_csv'],
            'Unsealed or changed predictions')
    rows = [json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
    validation = api['validate_rows'](rows, identity, records)
    with (output/'holdout500_n8.csv').open(newline='') as stream:
        scored_rows = list(csv.DictReader(stream))
    require(len(scored_rows) == 4000 and all(
        (int(a['_input_row_idx']), int(a['variant_idx']), a['h_sequence'], a['raw_l_sequence'], a['gen_l_sequence'])
        == (b['row_index'], b['variant_idx'], b['h_sequence'], b['raw_l_sequence'], b['sequence'])
        for a, b in zip(scored_rows, rows)), 'Scoring CSV altered, missing, or reordered candidates')
    return identity, validation, complete


def validate_metrics(metrics, candidates):
    """Recount fixed-denominator rates from saved per-candidate observations."""
    require(metrics['total_pairs'] == candidates, 'Scorer dropped candidates')
    details = metrics['detailed_results']
    require(len(details) == candidates and [r['index'] for r in details] == list(range(candidates)),
            'Missing or reordered detailed scores')
    fields = {'gen_valid_rate':'gen_valid', 'ref_valid_rate':'ref_valid',
              'overall_chain_match_rate':'chain_match', 'overall_v_gene_match_rate':'v_gene_match',
              'overall_j_gene_match_rate':'j_gene_match',
              'overall_v_gene_family_match_rate':'v_gene_family_match',
              'overall_j_gene_family_match_rate':'j_gene_family_match'}
    for metric, field in fields.items():
        require(all(type(row[field]) is bool for row in details), 'Non-boolean score flags')
        expected = sum(row[field] for row in details)/candidates
        require(math.isclose(float(metrics[metric]), expected, rel_tol=0, abs_tol=1e-12),
                'Score denominator or aggregate differs: '+metric)
    for field in ('gen_immunomatch_mean', 'ref_immunomatch_mean', 'gen_better_ratio',
                  'diversity_mean', 'diversity_max'):
        require(math.isfinite(float(metrics[field])), 'Nonfinite score: '+field)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    plan_ref = reference(args.plan); plan = read(args.plan)
    require(plan['schema'] == 'v7.pairing.fixed_scorer.v1'
            and reference(Path(__file__).resolve()) == plan['entrypoint'], 'Unexpected scoring plan')
    output = args.output_root
    require(output.is_absolute() and output == output.resolve() and not output.exists()
            and output.is_relative_to(ROOT/'output/downstream_generation')
            and not any(p.is_symlink() for p in (output, *output.parents)), 'New canonical output required')
    for name, value in plan['environment'].items():
        require(os.environ.get(name) == value, 'Scoring environment differs: '+name)
    identity, validation, complete = verify_predictions(plan)
    output.mkdir(parents=True, exist_ok=False)
    started = dict(at_utc=datetime.now(timezone.utc).isoformat(), plan=plan_ref,
        prediction_completion=reference(Path(plan['prediction_directory'])/'PREDICTIONS_COMPLETE.json'),
        identity=identity, candidate_validation=validation,
        packages={name:importlib.metadata.version(name) for name in
                  ('torch','transformers','datasets','abnumber','numpy','pandas','joblib')})
    write(output/'started.json', started)
    try:
        # An exact small CSV copy isolates the scorer's per-chain CSV intermediates.
        scoring_csv = output/'holdout500_n8.csv'
        shutil.copyfile(complete['scoring_csv']['path'], scoring_csv)
        require(sha(scoring_csv) == complete['scoring_csv']['sha256'], 'Scoring input copy changed')
        os.chdir(output)
        sys.path.insert(0, plan['source_directory'])
        import generation_eval as evaluator
        observed_scores = {}
        original_score = evaluator.compute_immunomatch_score
        def observe_score(*args, **kwargs):
            result = original_score(*args, **kwargs)
            values = [float(value) for value in result]
            require(len(values) == 4000 and all(math.isfinite(v) and 0 <= v <= 1 for v in values),
                    'ImmunoMatch omitted candidates or returned invalid scores')
            observed_scores[kwargs['lseq_col']] = values
            return result
        evaluator.compute_immunomatch_score = observe_score
        try:
            metrics = evaluator.main(str(scoring_csv), 'gen_l_sequence', 'raw_l_sequence',
                            detailed_comparison=True, heavy_col_name='h_sequence', expected_count=8)
        finally:
            evaluator.compute_immunomatch_score = original_score
        validate_metrics(metrics, 4000)
        for column, metric in [('gen_l_sequence','gen_immunomatch_mean'),
                               ('raw_l_sequence','ref_immunomatch_mean')]:
            require(math.isclose(sum(observed_scores[column])/4000, float(metrics[metric]),
                                rel_tol=0, abs_tol=1e-12), 'ImmunoMatch mean differs from saved scores')
        write(output/'immunomatch_scores.json', observed_scores)
        details = metrics.pop('detailed_results')
        write(output/'candidate_scores.json', details)
        metrics.update(evaluation_protocol=identity['protocol'],
            prediction_identity_sha256=complete['identity_sha256'], candidate_validation=validation,
            scoring_plan=plan_ref, candidate_scores=reference(output/'candidate_scores.json'),
            immunomatch_scores=reference(output/'immunomatch_scores.json'),
            denominator_policy='IM and overall match/validity use all candidates; legacy diversity excludes empty sequences',
            environment=started['packages'])
        write(output/'metrics.json', metrics)
        cleanup = []
        for path in output.glob('holdout500_n8_*.csv'):
            cleanup.append(reference(path)); path.unlink()
        if (output/'tmp').is_dir() and not any((output/'tmp').iterdir()):
            (output/'tmp').rmdir()
        write(output/'COMPLETE.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
            status='full_predictions_scored', metrics=reference(output/'metrics.json'),
            candidate_scores=reference(output/'candidate_scores.json'), removed_intermediate_csvs=cleanup,
            prediction_completion=started['prediction_completion'], plan=plan_ref))
        print(json.dumps(dict(status='full_predictions_scored', output=str(output))), flush=True)
    except BaseException as error:
        write(output/'FAILURE.json', dict(at_utc=datetime.now(timezone.utc).isoformat(), error=repr(error)))
        raise


if __name__ == '__main__':
    main()
