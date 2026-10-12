"""Collect the actual nine-task V7 CDR batch and fixed historical V5 references.

Activate conda pllm after actual two-epoch training and nine evaluations finish:
python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/collect_v7_cdr_results.py --output /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_cdr/report.json
This stdlib collector never loads a model, generates sequences, reruns scoring,
edits RESULTS.md, starts jobs or fabricates missing results. It creates a new
JSON only after all existing native completion and audit gates pass. The root
reviewer must inspect the artifact before updating the sole results document.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import statistics


ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
EPOCH = ROOT/'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006'
RECOVERY = EPOCH/'recovery_v1'
GENERATION = RECOVERY/'generation_pool_v1'
GATE = ROOT/'output/v7_local_ablation_20261005/post_generation_pool_v2/source_v1/v7_post_generation_acceptance.py'
GATE_SHA = '9a05fb9cee32aa6e60790fb2b1afb07eb3b63aa8dafe127e05d14cb070338f0c'
OUTPUT_ROOT = ROOT/'downstream/benchmark/outputs/v7_f011_step50082_cdr'
MODES = ('cdrh1', 'cdrh2', 'cdrh3', 'cdrl1', 'cdrl2', 'cdrl3')
AGGREGATION = 'sequence_macro_then_fold_macro; fold_sd_ddof0; float32_per_sequence'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    require(path.is_absolute() and path == path.resolve()
            and not any(p.is_symlink() for p in (path, *path.parents)) and path.is_file(),
            'Canonical existing evidence required: '+str(path))
    return dict(path=str(path), sha256=sha(path))


def read(path):
    ref(path)
    return json.loads(path.read_text())


def historical_v5():
    """Read fixed prior runs; do not select best iterations or recompute AAR."""
    base = ROOT/'output/downstream_generation'
    values = {}
    for step in (92000, 159000):
        date = '20260915' if step == 92000 else '20260917'
        for dataset in ('kong', 'sab23'):
            if step == 92000 and dataset == 'sab23':
                continue
            for mode in (MODES[:3] if dataset == 'kong' else MODES):
                path = base/f'ab_v5_{step}_full_{date}'/f'cdr-{dataset}'/(mode+'.log')
                reference = ref(path)
                matches = re.findall(r'^Average AAR all folds: ([0-9]+(?:\.[0-9]+)?)$',
                                     path.read_text(), flags=re.M)
                require(len(matches) == 1, 'One historical final aggregate required: '+str(path))
                value = float(matches[0])
                require(math.isfinite(value) and 0 <= value <= 100, 'Historical AAR outside percent range')
                values[(step, dataset, mode)] = dict(aar_percent=value, source=reference,
                    evidence_level='historical_aggregate_log_not_independently_rescored',
                    max_iter=1 if step == 92000 else 2, sampling_strategy='argmax')
    path = base/'ab_v5_92000_cdr_iter_20260916/summary.json'
    source = ref(path); summary = read(path)
    require(summary.get('complete') is True
            and summary.get('protocol') == 'cdr_iter_sweep_encoder_feedback_v1'
            and summary.get('checkpoint_sha256') ==
                '2f003a5b42a15498a7d6f2fc51414cad717f96a9cb405dca2388fcef0de3039c',
            'Fixed historical 92000 reference differs')
    for mode in MODES:
        item = summary['metrics'][mode+'/iter1']
        value = item['average_aar_all_folds']
        require(item['mode'] == mode and item['max_iter'] == 1 and item['n_sequences'] == 60
                and item['checkpoint_sha256'] == summary['checkpoint_sha256']
                and math.isfinite(value) and 0 <= value <= 100, 'Wrong historical single-step reference')
        values[(92000, 'sab23', mode)] = dict(aar_percent=value, source=source,
            evidence_level='previously_accepted_saved_prediction_summary_not_rescored_this_collection',
            max_iter=1, sampling_strategy='argmax')
    return values


def collect():
    require(sha(GATE) == GATE_SHA, 'Frozen completion gate changed')
    gate = runpy.run_path(str(GATE), run_name='_v7_cdr_collection_gate')
    fields = dict(formal_completion=ref(RECOVERY/'COMPLETE.json'),
        formal_stage=str(RECOVERY/'launches_v1/segment_50082'),
        generation_completion=ref(GENERATION/'COMPLETE.json'),
        generation_release=ref(GENERATION/'release.json'),
        generation_package=ref(GENERATION/'package_v1.json'))
    acceptance = gate['check'](fields)
    done = read(GENERATION/'COMPLETE.json')
    catalog = read(GENERATION/'catalog.json')
    historical = historical_v5()
    rows = []
    for task in sorted(done['tasks'], key=lambda t: (t['cell'][0], MODES.index(t['cell'][1]))):
        dataset, mode = task['cell']
        row = next(r for r in catalog['jobs'] if r['cell'] == task['cell'])
        directory = Path(row['evaluation_output'])/mode
        metric = gate['bound'](task['metrics'], directory/'metrics.json')
        manifest_path = directory/'manifest.json'; manifest = read(manifest_path)
        summary_path = directory.parent/'summary.json'; summary = read(summary_path)
        identity = manifest['identity']; parameters = identity['parameters']
        folds = list(range(10 if dataset == 'kong' else 1))
        require(metric['folds'] == folds and len(metric['fold_aar_percent']) == len(folds)
                and metric['aggregation'] == AGGREGATION, 'Fold aggregation protocol differs')
        require(all(math.isfinite(x) and 0 <= x <= 100 for x in metric['fold_aar_percent'])
                and math.isclose(statistics.mean(metric['fold_aar_percent']),
                    metric['average_aar_all_folds'], rel_tol=0, abs_tol=1e-10)
                and math.isclose(statistics.pstdev(metric['fold_aar_percent']),
                    metric['fold_sd_ddof0'], rel_tol=0, abs_tol=1e-10), 'Stored fold summary differs')
        require(manifest['status'] == 'complete'
                and manifest['identity_sha256'] == metric['identity_sha256']
                and manifest['predictions_sha256'] == metric['predictions_sha256']
                and identity['checkpoint'] == summary['checkpoint'], 'Artifact identity differs')
        require(parameters['mode'] == mode and parameters['max_iter'] == 1
                and parameters['sampling_strategy'] == 'argmax' and parameters['max_samples'] is None
                and parameters['resolved_folds'] == len(folds)
                and parameters['protocol'] == summary['protocol'] == 'v7_native_a4_cdr_iter1_argmax_v1'
                and summary['step'] == 50082 and summary['dataset'] == dataset,
                'Fixed native generation protocol differs')
        current = metric['average_aar_all_folds']
        references = {str(step): historical[(step, dataset, mode)] for step in (92000, 159000)}
        rows.append(dict(dataset=dataset, mode=mode, n_sequences=metric['n_sequences'],
            aar_percent=current, fold_aar_percent=metric['fold_aar_percent'],
            fold_sd_ddof0=metric['fold_sd_ddof0'], metrics=task['metrics'],
            manifest=ref(manifest_path), summary=ref(summary_path), input_audit=task['input_audit'],
            historical_v5=references,
            delta_pp_vs_v5_92000_iter1=current-references['92000']['aar_percent'],
            delta_pp_vs_v5_159000_iter2=current-references['159000']['aar_percent']))
    require(len(rows) == 9 and sum(r['n_sequences'] for r in rows) == 9741,
            'Nine-task batch coverage differs')
    return dict(schema='v7.cdr.actual_result_collection.v1', at_utc=datetime.now(timezone.utc).isoformat(),
        status='nine_actual_cdr_tasks_collected_pending_root_publication', checkpoint_step=50082,
        checkpoint=done['checkpoint'], generation_acceptance=acceptance, predecessor_evidence=fields,
        collector=ref(Path(__file__).resolve()), completion_gate=dict(path=str(GATE), sha256=GATE_SHA),
        aggregation=AGGREGATION, sampling_strategy='argmax', max_iter=1, rows=rows,
        all_nine_tasks_present=True, cdr_region_cases=9741, headline_document_edited=False,
        model_forward_calls=0, new_prediction_rescoring=False,
        limitations=['Known-length CDR infilling; not autonomous-length or antigen-conditioned design.',
            'V5 comparison changes data, model scale, training budget and input layout; not a controlled architecture effect.',
            'V5 159000 uses iter2, whereas V7 and V5 92000 references use iter1.',
            'Legacy aggregate logs do not establish independently reproduced per-sequence AAR.',
            '9741 counts CDR-region cases, not 9741 distinct antibodies; no combined cross-CDR score.',
            'No test-based checkpoint/iteration selection; no claim of all-scope pretraining decontamination.',
            'Light-chain pairing and T4 are outside this nine-task batch and remain separate.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    path = args.output
    require(path.is_absolute() and path == path.resolve() and path.is_relative_to(OUTPUT_ROOT)
            and not any(p.is_symlink() for p in (path, *path.parents)) and path.suffix == '.json'
            and not path.exists(), 'Require a fresh canonical result JSON in the assigned directory')
    result = collect()  # No output creation precedes all real completion gates.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], output=ref(path), rows=len(result['rows']))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
