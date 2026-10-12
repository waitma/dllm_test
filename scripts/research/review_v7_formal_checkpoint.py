"""Review a NEW saved formal checkpoint using JSON metadata only.

After conda activate pllm:
  python -B /absolute/path/review_v7_formal_checkpoint.py --step 7000 --previous 6000
Reuses the previous accepted point. Never loads tensors, a dataset, or a model;
never controls jobs. An existing result is an error, not a request to rerun it.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
SUP = ROOT/'output/v7_formal_supervision_20261005'
RUN = ROOT/'output/v7_initial_training_20261005/run'
CORE = ('loss', 'aa_loss', 'terminal_loss', 'aa_accuracy', 'terminal_accuracy',
        'relation_loss', 'relation_accuracy')
DENOMINATOR = dict(loss='selected_count', aa_loss='aa_count', aa_accuracy='aa_count',
    terminal_loss='terminal_count', terminal_accuracy='terminal_count',
    relation_loss='relation_count', relation_accuracy='relation_count')
LEVELS = ('mask25', 'mask50', 'mask75', 'mask100')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def equal(actual, expected, name):
    require(actual is None and expected is None or actual is not None and expected is not None
        and math.isfinite(actual) and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), 'Metric differs: '+name)


def average(values):
    defined = [v for v in values if v is not None]
    return sum(defined)/len(defined) if defined else None


def evaluation(path, step):
    state = read(path)
    require(state['global_step'] == step, 'Checkpoint Trainer global_step differs')
    rows = [r for r in state['log_history'] if r.get('step') == step and 'eval_records_per_mask' in r]
    require(len(rows) == 1, 'Expected one complete validation event at the new point')
    return state, rows[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--previous', type=int, required=True)
    args = parser.parse_args(argv)
    step, previous = args.step, args.previous
    require(0 < previous < step <= 25042, 'Expected a new later point in the original horizon')
    target = SUP/f'checkpoint{step}_root_observation.json'
    endpoint_target = SUP/f'checkpoint{step}_endpoint_diagnostics.json'
    require(not target.exists() and not endpoint_target.exists(), 'Already reviewed; do not repeat acceptance')
    prior_path = SUP/f'checkpoint{previous}_root_observation.json'
    prior_endpoint_path = SUP/f'checkpoint{previous}_endpoint_diagnostics.json'
    prior, prior_endpoint = read(prior_path), read(prior_endpoint_path)
    checkpoint, archive = RUN/f'checkpoint-{step}', SUP/f'checkpoint_evidence/checkpoint-{step}'
    sidecar = read(checkpoint/'formal_checkpoint.json')
    require(sidecar['schema'] == 'v7.formal.native_checkpoint.v1' and sidecar['step'] == step
        and sidecar['world_size'] == 8 and sidecar['writer_rank'] == 0
        and sidecar['rank_acknowledgments'] == list(range(8)), 'Native eight-rank save receipt differs')
    saved = sidecar['saved_parameter_digest']
    require(saved['numel'] == 75601176 and sidecar['rank_parameter_digests'] ==
            [{'rank': rank, **saved} for rank in range(8)], 'Rank parameters or saved model disagree')
    require(saved != prior['parameter_digest'], 'Parameters did not change since the accepted point')
    require(sidecar['binding_sha256'] == prior['binding_sha256'], 'Formal training binding changed')
    names = {'model.safetensors', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json',
        'training_args.bin', 'immune_config.json', 'research_config.json', 'vocabulary.json',
        'run_provenance.json', *(f'rng_state_{rank}.pth' for rank in range(8))}
    require(set(sidecar['files']) == names, 'Native 17-file inventory differs')
    sizes = {}
    for name in names:
        path = checkpoint/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0, 'Missing native file: '+name)
        sizes[name] = path.stat().st_size
    refs = {}
    for name in ('formal_checkpoint.json', 'immune_config.json', 'research_config.json',
                 'run_provenance.json', 'trainer_state.json', 'vocabulary.json'):
        path = checkpoint/name
        digest = sha(path)
        require(sha(archive/name) == digest, 'Supervisor metadata archive differs: '+name)
        if name != 'formal_checkpoint.json':
            require(sidecar['files'][name] == digest, 'Native metadata hash differs: '+name)
        refs[str(path)] = {'sha256': digest, 'archived_path': str(archive/name)}
    binding = read(checkpoint/'run_provenance.json')
    require(hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()
            == sidecar['binding_sha256'], 'Actual provenance digest differs')
    require(binding['variant'] == 'f011' and binding['global_batch'] == 256
        and binding['schedule']['max_steps'] == 25042, 'Wrong formal experiment')
    for path, expected in prior_endpoint['source_bindings'].items():
        require(sha(Path(path)) == expected and binding['source_files'][path] == expected,
                'Frozen metric/driver source differs')
    state, values = evaluation(checkpoint/'trainer_state.json', step)
    prior_state = SUP/f'checkpoint_evidence/checkpoint-{previous}/trainer_state.json'
    _, old = evaluation(prior_state, previous)
    old_ref = next(v for k, v in prior['metadata_references'].items() if k.endswith('/trainer_state.json'))
    require(sha(prior_state) == old_ref['sha256'], 'Prior accepted metadata changed')
    sources = {name: len(indices) for name, indices in binding['validation_indices'].items()}
    require(sources == prior['validation_source_record_counts'] and sum(sources.values()) == 1330,
            'Fixed validation source selection changed')
    require(values['eval_records_per_mask'] == 1330 and values['eval_record_evaluations'] == 5320,
            'Validation population changed')
    core_checks, count_checks = 0, 0
    for level in LEVELS:
        for source in sources:
            for count in ('aa_count', 'terminal_count', 'relation_count', 'selected_count'):
                key = f'eval_{level}_{source}_{count}'
                require(values[key] == old[key], 'Fixed selected population changed: '+key)
                count_checks += 1
        for metric in CORE:
            pairs = [(values[f'eval_{level}_{s}_{metric}'], values[f'eval_{level}_{s}_{DENOMINATOR[metric]}'])
                     for s in sources]
            denominator = sum(n for v, n in pairs)
            weighted = sum(v*n for v, n in pairs if v is not None)/denominator if denominator else None
            macro = average([v for v, n in pairs])
            for weighting, expected in [('token_micro', weighted), ('source_macro', macro)]:
                key = f'eval_{level}_{weighting}_{metric}'
                equal(values[key], expected, key)
                core_checks += 1
    for weighting in ('token_micro', 'source_macro'):
        for metric in CORE:
            key = f'eval_mask_mean_{weighting}_{metric}'
            equal(values[key], average([values[f'eval_{level}_{weighting}_{metric}'] for level in LEVELS]), key)
            core_checks += 1
    require(core_checks == 70 and count_checks == 112, 'Review scope differs')
    summaries = {level: {weighting: {metric: values[f'eval_{level}_{weighting}_{metric}'] for metric in CORE}
        for weighting in ('token_micro', 'source_macro')} for level in ('mask_mean', 'mask100')}
    comparisons = {level: {weighting: {metric: dict(previous=before, current=summaries[level][weighting][metric],
        delta=summaries[level][weighting][metric]-before if before is not None else None)
        for metric, before in old_values.items()} for weighting, old_values in group.items()}
        for level, group in prior['metrics'].items()}
    per_source = {s: {m: dict(previous=old[f'eval_mask100_{s}_{m}'], current=values[f'eval_mask100_{s}_{m}'])
                     for m in CORE} for s in sources}
    metric_names = ('first_terminal_accuracy', 'terminal_missing_rate', 'length_mae',
                    'terminal_precision', 'terminal_false_positive_rate')
    counts = ('chain_count', 'first_terminal_correct_count', 'terminal_missing_count', 'length_compared_count',
              'terminal_true_positive_count', 'terminal_prediction_count', 'terminal_false_positive_count', 'aa_count')
    point = {'per_source': {s: {k: values[f'eval_mask100_{s}_{k}'] for k in (*metric_names, *counts)} for s in sources}}
    endpoint_checks = 0
    for weighting in ('token_micro', 'source_macro'):
        point[weighting] = {k: values[f'eval_mask100_{weighting}_{k}'] for k in (*metric_names, *counts)}
        for count in counts:
            require(point[weighting][count] == sum(p[count] for p in point['per_source'].values()),
                    'Endpoint count aggregation differs: '+weighting+'/'+count)
            endpoint_checks += 1
    pooled = point['token_micro']
    require(pooled['length_compared_count'] == pooled['chain_count']-pooled['terminal_missing_count']
        and pooled['terminal_prediction_count'] == pooled['terminal_true_positive_count']+pooled['terminal_false_positive_count'],
        'Endpoint count partition inconsistent')
    for metric, numerator, denominator in (
        ('first_terminal_accuracy', 'first_terminal_correct_count', 'chain_count'),
        ('terminal_missing_rate', 'terminal_missing_count', 'chain_count'),
        ('terminal_precision', 'terminal_true_positive_count', 'terminal_prediction_count'),
        ('terminal_false_positive_rate', 'terminal_false_positive_count', 'aa_count')):
        expected = pooled[numerator]/pooled[denominator] if pooled[denominator] else None
        equal(pooled[metric], expected, metric)
        endpoint_checks += 1
    length_sum = sum((p['length_mae'] or 0)*p['length_compared_count'] for p in point['per_source'].values())
    equal(pooled['length_mae'], length_sum/pooled['length_compared_count'], 'length_mae')
    endpoint_checks += 1
    for metric in metric_names:
        equal(point['source_macro'][metric], average([p[metric] for p in point['per_source'].values()]), metric)
        endpoint_checks += 1
    require(endpoint_checks == 26, 'Endpoint check scope differs')
    now = datetime.now(timezone.utc).isoformat()
    observation = dict(at_utc=now, schema='v7.formal.checkpoint_observation.v1', step=step,
        planned_steps=25042, complete_training=False, checkpoint_complete=True, native_inventory_count=17,
        native_inventory_verified_by=prior['native_inventory_verified_by'], root_rehashed_tensor_files=False,
        native_file_sizes=sizes, metadata_sha256_and_archive_match=True, rank_parameters_equal_and_match_saved=True,
        parameter_digest=saved, binding_sha256=sidecar['binding_sha256'], metadata_references=refs,
        validation_records_per_mask=1330, record_evaluations_across_four_masks=5320,
        validation_source_record_counts=sources, mask_levels=list(LEVELS), metrics=summaries,
        full_mask_per_source_comparison=per_source, unchanged_validation_count_fields=count_checks,
        reaggregated_metric_fields=core_checks, reaggregated_core_metrics=list(CORE),
        all_reported_core_metrics_reaggregated_from_per_source=True,
        previous_point_reused_from={'path': str(prior_path), 'sha256': sha(prior_path)},
        previous_step=previous, binding_unchanged_since_previous_checkpoint=True, parameters_changed_since_previous_checkpoint=True,
        comparison_to_previous=comparisons, evaluation_kind='regular_every1000' if step % 1000 == 0 else 'segment_boundary',
        updates_since_previous_evaluation=step-previous, new_jobs_or_restarts=0, limitations=[
            f'Checkpoint{step} is saved; total25042 and the current5016-to10016 resume/exit acceptance remain incomplete.',
            *prior['limitations'][1:]],
        reviewer={'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__))})
    endpoint = dict(schema='v7.formal.endpoint_diagnostics.v1', at_utc=now, steps=[previous, step],
        mask='mask100', metadata_only=True, new_model_runs=0, source_bindings=prior_endpoint['source_bindings'],
        trainer_state_bindings={str(prior_state): {'sha256': sha(prior_state)},
                               str(checkpoint/'trainer_state.json'): {'sha256': sha(checkpoint/'trainer_state.json')}},
        new_point_reaggregation_checks=endpoint_checks,
        previous_point_reused_from={'path': str(prior_endpoint_path), 'sha256': sha(prior_endpoint_path)},
        definitions=prior_endpoint['definitions'], points={str(previous): prior_endpoint['points'][str(previous)], str(step): point})
    for path, result in ((target, observation), (endpoint_target, endpoint)):
        with path.open('x') as stream:
            stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(observation=str(target), observation_sha256=sha(target),
        endpoint=str(endpoint_target), endpoint_sha256=sha(endpoint_target),
        core_checks=core_checks, count_checks=count_checks, endpoint_checks=endpoint_checks,
        metrics=summaries, endpoints=pooled), ensure_ascii=False))


if __name__ == '__main__':
    main()
