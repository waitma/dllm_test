"""Audit saved screening predictions and masked model inputs on CPU.

Activate pllm, set PYTHONPATH to the exploration root, and run with its
output/arch70m_env/bin/python:
python /absolute/path/audit_generation_context.py --output-dir /absolute/report
This reads only screening tasks; it performs no inference or model selection.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig, FixedChainModel
from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyCollator
from experiments.arch70m.evaluate import make_tasks

ROOT = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore')
RUN = ROOT / 'output/arch70m_v8_20261028'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def aar(prediction, reference):
    return sum(a == b for a, b in zip(prediction, reference)) / len(reference)


def audit(output):
    evidence = {}

    def read(path):
        raw = path.read_bytes()
        evidence[str(path)] = hashlib.sha256(raw).hexdigest()
        return raw

    settings = json.loads(read(ROOT / 'experiments/arch70m/base.json'))
    index = json.loads(read(RUN / 'data/index.json'))
    ref = index['files']['screen']['task_valid']
    raw = read(Path(ref['path']))
    assert hashlib.sha256(raw).hexdigest() == ref['sha256']
    model = SimpleNamespace(config=FixedChainConfig(**settings['model']),
                            family_spec=settings['family_prediction'])
    collator = VFamilyCollator(model.config, family_spec=model.family_spec, hide_relation=True)
    tokenizer = collator.tokenizer
    tasks, _ = make_tasks(model, [json.loads(line) for line in raw.splitlines() if line])
    task_lookup = {}
    for task in tasks:
        if task['kind'] != 'conditional':
            continue
        identity = task['source'], task['key'], task['role']
        assert identity not in task_lookup
        # Only the actual denoise arguments: no labels, residue mask or true length.
        features = {k: v.clone() for k, v in FixedChainModel.feature_args(task['clean']).items()}
        features['input_ids'][0, task['target']] = tokenizer.mask_token_id
        reference = ''.join(tokenizer.token(int(t)) for t in task['truth'][task['score_mask']])
        assert len(reference) == task['true_length']
        task_lookup[identity] = dict(reference=reference,
            input_sha256=digest({k: v.tolist() for k, v in features.items()}))

    task_rows, group_rows, source_rows, summary_rows, duplicate_inputs = [], [], [], [], []
    groups = defaultdict(list)
    for identity, task in task_lookup.items():
        groups[identity[0], identity[2]].append((identity, task))
    for (source, role), items in sorted(groups.items()):
        inputs = defaultdict(list)
        for identity, task in items:
            inputs[task['input_sha256']].append((identity, task))
        for signature, members in inputs.items():
            if len(members) > 1:
                duplicate_inputs.append(dict(source=source, role=role, input_sha256=signature,
                    tasks=len(members), unique_reference_sequences=len({t['reference'] for _, t in members}),
                    receptor_keys=[i[1] for i, _ in members]))

    for variant in [f'A{i}' for i in range(8)]:
        trial = RUN / 'trials' / f'screen_{variant}_s42'
        status = json.loads(read(trial / 'status.json'))
        assert status['status'] == 'completed' and status['step'] == 1276
        saved = json.loads(read(trial / 'metrics/screen_001276_observations.json'))['generation']
        observations = {(r['source'], r['key'], r['role']): r for r in saved if r['kind'] == 'conditional'}
        assert set(observations) == set(task_lookup)
        source_values = defaultdict(list)
        for (source, role), items in sorted(groups.items()):
            values, predictions = [], []
            by_input = defaultdict(set)
            for identity, task in items:
                observation = observations[identity]
                prediction = (observation['sequence'] or '') if observation['valid'] else ''
                own = aar(prediction, task['reference'])
                assert abs(own - observation['aar']) < 1e-12
                # Average over all other distinct target sequences in this source/role.
                # This is a score reassignment control, not a rerun with changed context.
                other_references = sorted({t['reference'] for _, t in items} - {task['reference']})
                assert other_references
                other = statistics.mean(aar(prediction, seq) for seq in other_references)
                value = dict(variant=variant, source=source, role=role, receptor_key=identity[1],
                    input_sha256=task['input_sha256'], reference_sha256=digest(task['reference']),
                    prediction_sha256=digest(prediction), valid=bool(observation['valid']),
                    own_target_aar=own, other_target_aar=other, own_minus_other=own-other,
                    other_unique_targets=len(other_references))
                task_rows.append(value)
                values.append(value)
                source_values[source].append(value)
                if observation['valid']:
                    predictions.append(prediction)
                    by_input[task['input_sha256']].add(prediction)
            counts = Counter(predictions)
            group_rows.append(dict(variant=variant, source=source, role=role, tasks=len(items),
                unique_inputs=len({t['input_sha256'] for _, t in items}),
                unique_references=len({t['reference'] for _, t in items}),
                valid_predictions=len(predictions), unique_predictions=len(counts),
                largest_identical_prediction_group=max(counts.values(), default=0),
                same_input_multiple_predictions=sum(len(v) > 1 for v in by_input.values()),
                **{k: statistics.mean(v[k] for v in values)
                   for k in ('own_target_aar', 'other_target_aar', 'own_minus_other')}))
        for source, values in sorted(source_values.items()):
            source_rows.append(dict(variant=variant, source=source, tasks=len(values),
                **{k: statistics.mean(v[k] for v in values)
                   for k in ('own_target_aar', 'other_target_aar', 'own_minus_other')}))
        summary_rows.append(dict(variant=variant, tasks=sum(map(len, source_values.values())),
            **{k: statistics.mean(statistics.mean(v[k] for v in values) for values in source_values.values())
               for k in ('own_target_aar', 'other_target_aar', 'own_minus_other')}))

    output.mkdir(parents=True, exist_ok=True)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), stage='screen_diagnostic_only',
        tasks=len(task_lookup), source_file_sha256=evidence, duplicate_inputs=duplicate_inputs,
        summary=summary_rows, by_source=source_rows,
        definitions=dict(own_target_aar='Original full-reference positional AAR; invalid predictions zero.',
            other_target_aar='Same saved prediction scored against each other distinct reference sequence within source/role, then averaged.',
            macro='Average tasks within each source, then equal average across four sources.',
            input_signature='All seven denoise input features after masking the entire target slot; labels and true length excluded.',
            unique_predictions='Valid decoded sequence strings, counted within source/role.'),
        limitations=['Post hoc single-seed screening diagnostic, not a selection or confirmation endpoint.',
            'Rescoring against other references does not intervene on the model input and cannot establish causal conditioning or binding specificity.',
            'High identity can reflect conserved framework; lower diversity is not inherently worse under greedy decoding.',
            'Repeated identical inputs can have different correct targets and are expected to produce the same greedy output.'])
    (output / 'context_analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    for name, values in [('per_task.csv', task_rows), ('by_source_role.csv', group_rows),
                         ('by_source.csv', source_rows), ('summary.csv', summary_rows)]:
        with (output / name).open('w') as f:
            writer = csv.DictWriter(f, sorted({k for row in values for k in row}), lineterminator='\n')
            writer.writeheader()
            writer.writerows(values)
    print(json.dumps(dict(summary=summary_rows, duplicate_inputs=duplicate_inputs,
                          observed_tasks=len(task_rows)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    assert args.output_dir.is_absolute()
    audit(args.output_dir)
