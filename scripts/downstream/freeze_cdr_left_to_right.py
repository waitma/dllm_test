"""Activate pllm; python -B FILE --out ABS to freeze the strict LTR follow-up.

Use the already scored CDR rows and both fixed checkpoints; do not tune or select.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path('/project_c20250601/251105016/project/dllm_test')
PARENT = ROOT / 'output/downstream_generation/cdr_block_budget_20261016'
PARENT_DELIVERY_SHA = '799c1ed01c32e26ec2af768f003cd08d34bf720a52857936da6355c99f645bf1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    assert not (out / 'plan.json').exists()
    assert sha(PARENT / 'budget_delivery.json') == PARENT_DELIVERY_SHA
    delivery = read(PARENT / 'budget_delivery.json')
    assert delivery['status'] == 'complete' and not delivery['winner_selected']
    previous = read(PARENT / 'plan.json')
    ablation = Path(previous['parent'])
    evidence = {**delivery['evidence_sha256'], str(PARENT / 'budget_delivery.json'): PARENT_DELIVERY_SHA}
    for checkpoint in previous['checkpoints']:
        for name in ('independent_audit.json', 'predictions.jsonl'):
            path = ablation / f'evaluation_{checkpoint}' / name
            evidence[str(path)] = sha(path)
    for path, digest in evidence.items():
        assert sha(path) == digest, path
    assert sha(PARENT / 'plan.json') == 'a671a2122df4efd4d9f0483fd08cd2446262edbfdc80ac81af8c1f7e9a143340'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'rows.json').exists()
    shutil.copy2(PARENT / 'rows.json', out / 'rows.json')
    rows = read(out / 'rows.json')
    counts = dict(Counter(row['mode'] for row in rows))
    assert counts == {'cdrh1': 526, 'cdrh2': 526, 'cdrh3': 4202}
    lengths = [len(row['mask_h']) for row in rows]
    mode_stats = {}
    for mode in sorted(counts):
        subset = [row for row in rows if row['mode'] == mode]
        values = [len(row['mask_h']) for row in subset]
        mode_stats[mode] = dict(n=len(subset), components=len({row['group'] for row in subset}),
            logical_sample_forwards=sum(values), mean_logical_nfe=sum(values) / len(values),
            min_length=min(values), max_length=max(values))
    batches = [lengths[start:start + 24] for start in range(0, len(rows), 24)]
    cost = dict(logical_sample_forwards=sum(lengths), physical_sample_forwards=sum(max(v) * len(v) for v in batches),
        physical_batch_forwards=sum(max(v) for v in batches), batches=len(batches))
    assert cost == dict(logical_sample_forwards=74087, physical_sample_forwards=121948, physical_batch_forwards=5083, batches=219)
    sample_use = 'Same previously scored ablation and budget rows, order and batches. Fixed follow-up, not independent unseen confirmation; no unused reserve consumed.'
    data = dict(parent=str(PARENT), rows_sha256=sha(out / 'rows.json'), rows_identical_to_parent=True,
        counts_by_mode=counts, mode_stats=mode_stats, cost_per_checkpoint=cost, sample_use=sample_use,
        parent_data_plan_sha256=sha(PARENT / 'data_plan.json'))
    write(out / 'data_plan.json', data)
    sources = {}
    (out / 'source').mkdir(exist_ok=False)
    for relative in ('downstream/run_cdr_left_to_right.py', 'downstream/cdr_left_to_right.py',
                     'downstream/audit_cdr_left_to_right.py', 'downstream/freeze_cdr_left_to_right.py',
                     'tests/test_cdr_left_to_right.py'):
        original = ROOT / 'scripts' / relative
        destination = out / 'source' / original.name
        shutil.copy2(original, destination)
        sources[str(destination)] = sha(destination)
    plan = dict(schema='cdr.left.to.right.protocol.v1', created_utc=datetime.now(timezone.utc).isoformat(),
        parent=str(PARENT), ablation_parent=str(ablation), block_parent=previous['block_parent'],
        marginal_parent=previous['marginal_parent'], checkpoints=previous['checkpoints'], batch_size=24,
        precision='fp32', sampling='argmax', policy='left_to_right', policies=['left_to_right'],
        reused_controls=['single_pass', 'original4', 'original8', 'grouped4', 'grouped8'],
        counts_by_mode=counts, source_sha256=sources,
        external_source_sha256={**previous['source_sha256'], **previous['external_source_sha256']},
        parent_evidence_sha256=evidence, rows_sha256=sha(out / 'rows.json'),
        parent_rows_sha256=sha(PARENT / 'rows.json'), parent_plan_sha256=sha(PARENT / 'plan.json'),
        parent_audit_sha256={cp: sha(PARENT / f'evaluation_{cp}/independent_audit.json') for cp in previous['checkpoints']},
        ablation_audit_sha256={cp: sha(ablation / f'evaluation_{cp}/independent_audit.json') for cp in previous['checkpoints']},
        data_plan_sha256=sha(out / 'data_plan.json'),
        bootstrap=dict(repeats=50000, seed=20261016, unit='connected_component', family_size=2),
        primary_contrast='left_to_right_vs_single_pass', primary_mode='cdrh3',
        inferential_rule='Two prespecified H3 contrasts, one per checkpoint: LTR minus single pass. Paired component percentile bootstrap: 95% plus Bonferroni 97.5% intervals (1.25/98.75 percentiles). Cross-checkpoint improvement requires both adjusted lower bounds above zero; zero crossing is inconclusive, not equivalence.',
        secondary_rule='H1/H2 and LTR versus original4/original8/grouped4/grouped8 are descriptive 95% intervals. No winner selection or tuning. Local family correction does not cover all historical adaptive exploration.',
        decoding='Ascending target positions, one hard argmax token per active row per query. Earlier predicted tokens remain visible; current and later targets stay MASK. The first all-MASK forward is reused for the first position. Observed framework/light and all other positions stay unchanged. Bidirectional backbone attention unchanged.',
        token_rule='Keep the same frozen predictor vocabulary: 20 amino acids plus EOS and chain separator, original EOS penalty. No extra token restriction or early stopping; fill exactly the known target length. Non-AA outputs score incorrect and are reported.',
        budget_definition='Per-row logical NFE equals target length L. Batch executes max(L) calls including first prior, retaining inactive rows. Report padded physical sample-forwards separately; gates are separate.',
        sample_use=sample_use,
        limitations=['This is a new complete decoding policy, not a computation-matched isolation of left-to-right order.',
            'Right-hand observed framework and light chain remain available; this is not a causal autoregressive model.',
            'No training or hyperparameter selection, and no fresh held-out reserve is consumed.',
            'Inherited exact/cluster exclusions do not establish universal pretraining or homology decontamination.'])
    write(out / 'plan.json', plan)
    print(json.dumps(dict(status='frozen', plan_sha256=sha(out / 'plan.json'), rows=len(rows), cost=cost)))


if __name__ == '__main__':
    main()
