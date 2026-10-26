"""Review a new where-DDP checkpoint using its native save and metric metadata.

Activate conda pllm, then run:
python -B /project_c20250601/251105016/project/dllm_test/scripts/research/review_v7_where_checkpoint.py --step 9000 --previous 8000
Only JSON and source hashes are read. No tensors, corpus, GPU, or job controls
are used. Previously reviewed points must not be repeated.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import runpy

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
BASE = ROOT/'output/v7_formal_eight_gpu_20261002/where_continuation_20261006'
RUN = BASE/'run'
HELPER = ROOT/'scripts/research/review_v7_formal_checkpoint.py'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--previous', type=int, required=True)
    args = parser.parse_args(argv)
    step, previous = args.step, args.previous
    require(8000 <= previous < step <= 25042, 'Expected a new point in the where run')
    output = BASE/f'checkpoint{step}_root_review.json'
    require(not output.exists(), 'Already reviewed; do not repeat')
    prior_path = BASE/f'checkpoint{previous}_root_review.json'
    prior = read(prior_path)
    require(prior['status'] == 'actual_native_world8_save_and_validation_metadata_passed'
            and prior['step'] == previous, 'Previous checkpoint was not accepted')
    helper = runpy.run_path(str(HELPER), run_name='_v7_metadata_helpers')
    equal, average = helper['equal'], helper['average']
    checkpoint = RUN/f'checkpoint-{step}'
    sidecar = read(checkpoint/'formal_checkpoint.json')
    binding = read(checkpoint/'run_provenance.json')
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(sidecar['schema'] == 'v7.formal.native_checkpoint.v1' and sidecar['step'] == step
            and sidecar['world_size'] == 8 and sidecar['writer_rank'] == 0
            and sidecar['rank_acknowledgments'] == list(range(8)), 'Native eight-rank save differs')
    require(digest == sidecar['binding_sha256'] == prior['binding_sha256']
            and binding == read(RUN/'run_provenance.json'), 'Same-run binding changed')
    require(binding['variant'] == 'f011' and binding['global_batch'] == 256
            and binding['schedule']['max_steps'] == 25042, 'Wrong formal experiment')
    manifest = BASE/'source_manifest_v1.json'
    require(sha(manifest) == binding['source_manifest_sha256'], 'Source manifest changed')
    files = read(manifest)['files']
    require(files == binding['source_files'], 'Bound source set changed')
    for name, expected in files.items():
        require(sha(Path(name)) == expected, 'Bound source changed: '+name)
    saved = sidecar['saved_parameter_digest']
    require(saved['numel'] == 75601176 and sidecar['rank_parameter_digests'] ==
            [dict(rank=rank, **saved) for rank in range(8)], 'Actual and saved parameters disagree')
    old_archive = Path(prior['metadata_archive'])
    require(sha(old_archive/'formal_checkpoint.json') == prior['native_save_receipt']['sha256'],
            'Previous native receipt archive changed')
    old_sidecar = read(old_archive/'formal_checkpoint.json')
    require(saved != old_sidecar['saved_parameter_digest'], 'Parameters did not change')
    require(sha(old_archive/'trainer_state.json') == old_sidecar['files']['trainer_state.json'],
            'Previous metric archive changed')
    names = {'model.safetensors', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json',
             'training_args.bin', 'immune_config.json', 'research_config.json',
             'vocabulary.json', 'run_provenance.json', *(f'rng_state_{rank}.pth' for rank in range(8))}
    require(set(sidecar['files']) == names, 'Native file inventory differs')
    archive = RUN/'checkpoint_retention_evidence'/f'checkpoint-{step}_{digest[:12]}'
    references = {}
    for name in names:
        path = checkpoint/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                'Missing native file: '+name)
        if name.endswith('.json'):
            require(sha(path) == sidecar['files'][name] == sha(archive/name),
                    'Native metadata/archive changed: '+name)
            references[name] = ref(archive/name)
    require(sha(checkpoint/'formal_checkpoint.json') == sha(archive/'formal_checkpoint.json'),
            'Native save receipt archive differs')
    _, values = helper['evaluation'](checkpoint/'trainer_state.json', step)
    _, old = helper['evaluation'](old_archive/'trainer_state.json', previous)
    require(all(not isinstance(v, float) or math.isfinite(v) for v in values.values()),
            'Nonfinite validation metric')
    sources = {name: len(indices) for name, indices in binding['validation_indices'].items()}
    require(sum(sources.values()) == values['eval_records_per_mask'] == 1330
            and values['eval_record_evaluations'] == 5320, 'Validation population differs')
    counts = core = 0
    for level in helper['LEVELS']:
        for source in sources:
            for count in ('aa_count', 'terminal_count', 'relation_count', 'selected_count'):
                key = f'eval_{level}_{source}_{count}'
                require(values[key] == old[key], 'Fixed selected population differs: '+key)
                counts += 1
        for metric in helper['CORE']:
            pairs = [(values[f'eval_{level}_{s}_{metric}'],
                      values[f"eval_{level}_{s}_{helper['DENOMINATOR'][metric]}"]) for s in sources]
            denominator = sum(n for _, n in pairs)
            micro = sum(v*n for v, n in pairs if v is not None)/denominator if denominator else None
            for weighting, expected in [('token_micro', micro), ('source_macro', average([v for v, _ in pairs]))]:
                equal(values[f'eval_{level}_{weighting}_{metric}'], expected, metric)
                core += 1
    for weighting in ('token_micro', 'source_macro'):
        for metric in helper['CORE']:
            equal(values[f'eval_mask_mean_{weighting}_{metric}'],
                  average([values[f'eval_{level}_{weighting}_{metric}'] for level in helper['LEVELS']]), metric)
            core += 1
    require((counts, core) == (112, 70), 'Metric review scope differs')
    end_metrics = ('first_terminal_accuracy', 'terminal_missing_rate', 'length_mae',
                   'terminal_precision', 'terminal_false_positive_rate')
    end_counts = ('chain_count', 'first_terminal_correct_count', 'terminal_missing_count',
                  'length_compared_count', 'terminal_true_positive_count', 'terminal_prediction_count',
                  'terminal_false_positive_count', 'aa_count')
    per_source = {s: {k: values[f'eval_mask100_{s}_{k}'] for k in (*end_metrics, *end_counts)} for s in sources}
    pooled = {k: values[f'eval_mask100_token_micro_{k}'] for k in (*end_metrics, *end_counts)}
    endpoint_checks = 0
    for weighting in ('token_micro', 'source_macro'):
        for key in end_counts:
            require(values[f'eval_mask100_{weighting}_{key}'] == sum(v[key] for v in per_source.values()),
                    'Endpoint count differs: '+key)
            endpoint_checks += 1
    require(pooled['length_compared_count'] == pooled['chain_count']-pooled['terminal_missing_count']
            and pooled['terminal_prediction_count'] == pooled['terminal_true_positive_count']+pooled['terminal_false_positive_count'],
            'Endpoint count partitions differ')
    for metric, numerator, denominator in (
            ('first_terminal_accuracy', 'first_terminal_correct_count', 'chain_count'),
            ('terminal_missing_rate', 'terminal_missing_count', 'chain_count'),
            ('terminal_precision', 'terminal_true_positive_count', 'terminal_prediction_count'),
            ('terminal_false_positive_rate', 'terminal_false_positive_count', 'aa_count')):
        equal(pooled[metric], pooled[numerator]/pooled[denominator] if pooled[denominator] else None, metric)
        endpoint_checks += 1
    length_sum = sum((v['length_mae'] or 0)*v['length_compared_count'] for v in per_source.values())
    equal(pooled['length_mae'], length_sum/pooled['length_compared_count'] if pooled['length_compared_count'] else None,
          'length_mae')
    endpoint_checks += 1
    for metric in end_metrics:
        equal(values[f'eval_mask100_source_macro_{metric}'], average([v[metric] for v in per_source.values()]), metric)
        endpoint_checks += 1
    require(endpoint_checks == 26, 'Endpoint review scope differs')
    retention_path = RUN/'checkpoint_retention_evidence'/f'checkpoint-{step}_decision.json'
    retention = read(retention_path)
    require(retention['status'] == 'applied' and retention['latest'] == str(checkpoint)
            and retention['metric'] == 'eval_loss' and retention['best_k'] == 3
            and retention['all_retained_states'] == 'full', 'Retention policy differs')
    best = sorted(retention['candidates'], key=lambda x: (x['eval_loss'], x['step']))[:3]
    expected_keep = {x['path'] for x in best} | {str(checkpoint)}
    require(set(retention['best']) == {x['path'] for x in best}
            and {x['path'] for x in retention['keep']} == expected_keep, 'Best3+latest selection differs')
    metric_keys = ['eval_loss'] + [f'eval_{level}_token_micro_{k}' for level in ('mask_mean', 'mask100')
        for k in ('aa_loss', 'aa_accuracy', 'terminal_loss', 'terminal_accuracy')]
    metric_keys += [f'eval_mask100_token_micro_{k}' for k in (*end_metrics, 'length_compared_count', 'chain_count')]
    report = dict(at_utc=datetime.now(timezone.utc).isoformat(),
        status='actual_native_world8_save_and_validation_metadata_passed', step=step, previous_step=previous,
        native_save_receipt=ref(checkpoint/'formal_checkpoint.json'), binding_sha256=digest,
        source_manifest=ref(manifest), prior_review=ref(prior_path), metadata_archive=str(archive),
        metadata_references=references, native_file_count=17, rank_saved_parameter_digests_equal=True,
        full_tensor_byte_validation='Eight native Trainer callbacks; root does not reload or rehash tensors',
        validation_count_checks=counts, core_aggregation_checks=core, endpoint_aggregation_checks=endpoint_checks,
        metrics={k: dict(previous=old[k], current=values[k]) for k in metric_keys},
        retention=ref(retention_path), retained_steps=[x['step'] for x in retention['keep']],
        removed_steps=[x['step'] for x in retention['remove']], metadata_reviewer=ref(Path(__file__).resolve()),
        legacy_metric_helper=ref(HELPER), model_tests_launched=0, segment_exit_acceptance=False)
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(output=ref(output), step=step, previous=previous,
        metrics=report['metrics'], retained_steps=report['retained_steps']), ensure_ascii=False))


if __name__ == '__main__':
    main()
