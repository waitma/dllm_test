"""Run the independent 120k large-batch architecture protocol, without submitting.

CPU-only entry check (artificial rows, tiny model, two updates):
  source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
  conda activate pllm
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -B \
    /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_large_batch_ablation.py \
    --variant f111 --seed 42 --micro-batch 16 --grad-accum 2 --cpu-smoke \
    --steps 2 --output-dir /absolute/new_cpu_check
Parameter-only checks: run this absolute script with --self-test.
Actual runs additionally require --data-dir, --batch-plan and its
--batch-plan-sha256. All 22 conditions share that selected profile plan.
Existing p02 checkpoints cannot migrate into this independent protocol.
New tests save metrics only unless --save-checkpoints is explicitly requested.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'v7_context_factorial_120k_large_batch_v1'
PLAN_SCHEMA = 'v7.large_batch_selection.v1'
SAMPLE_SHA = '1efa27ea19d10bc94143c3977c80dba8c3a29bad921c093ca7f24447635b9478'
HORIZON = 3750
EVAL_MICRO = 4
VARIANT_NAMES = ('f000', 'f001', 'f010', 'f011', 'f100', 'f101', 'f110', 'f111',
                 'cv_segment_global', 'cv_late_fusion', 'fixed_gate_control')
BATCH_PAIRS = ((16, 2), (32, 1))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_parameters(args):
    if args.variant not in VARIANT_NAMES or args.seed not in (42, 43):
        raise ValueError('Require a registered architecture and seed 42 or 43')
    if ((args.micro_batch, args.grad_accum) not in BATCH_PAIRS
            or args.eval_every != 250 or not 1 <= args.steps <= HORIZON):
        raise ValueError('Require micro16/GA2 or micro32/GA1, global32, eval250, and 1..3750 updates')
    for path in (args.output_dir, args.data_dir, args.batch_plan):
        if path is not None and not path.is_absolute():
            raise ValueError('Use absolute directory and plan paths')
    if args.cpu_smoke:
        if (args.steps != 2 or args.data_dir is not None or args.resume
                or args.batch_plan is not None or args.batch_plan_sha256 is not None):
            raise ValueError('CPU check requires two fresh artificial updates without data or profile plan')
    elif args.data_dir is None or args.batch_plan is None or args.batch_plan_sha256 is None:
        raise ValueError('Actual runs require the registered subset and common selected batch plan')


def checked_batch_plan(args):
    """Read real profile references; this entry never manufactures a selection."""
    path = args.batch_plan
    if (path.is_symlink() or path != path.resolve() or not path.is_file()
            or re.fullmatch('[0-9a-f]{64}', args.batch_plan_sha256) is None
            or sha(path) != args.batch_plan_sha256):
        raise ValueError('Batch selection path/SHA differs')
    plan = json.loads(path.read_text())
    expected = {'schema': PLAN_SCHEMA, 'study': STUDY, 'status': 'selected',
                'micro_batch': args.micro_batch, 'grad_accum': args.grad_accum,
                'global_batch': 32, 'evaluation_micro_batch': EVAL_MICRO,
                'sample_manifest_sha256': SAMPLE_SHA, 'variants': list(VARIANT_NAMES),
                'seeds': [42, 43]}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError('Selected batch plan does not cover this complete study/configuration')
    evidence = plan.get('profile_evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('Selected batch plan requires actual profile evidence references')
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {'path', 'sha256'}:
            raise ValueError('Profile evidence must be a file/SHA reference')
        source = Path(item['path'])
        if (not source.is_absolute() or source.is_symlink() or source != source.resolve()
                or not source.is_file() or re.fullmatch('[0-9a-f]{64}', item['sha256']) is None
                or sha(source) != item['sha256']):
            raise ValueError('Actual profile evidence path/SHA differs')
    return {'path': str(path), 'sha256': args.batch_plan_sha256, 'profile_evidence': evidence}


def check_resume_provenance(previous, args):
    """Reject protocol migration before reading a binary checkpoint."""
    if (previous.get('study', {}).get('name') != STUDY
            or previous.get('study', {}).get('evaluation_micro_batch') != EVAL_MICRO
            or previous.get('micro_batch') != args.micro_batch
            or previous.get('grad_accum') != args.grad_accum
            or previous.get('global_batch') != 32):
        raise ValueError('Resume requires this same large-batch protocol; old p02 cannot migrate')


def fixed_evaluator(evaluate):
    def evaluation(model, collate, rows, sources, device, training_micro, cpu):
        return evaluate(model, collate, rows, sources, device, EVAL_MICRO, cpu)
    return evaluation


def prepare(args):
    validate_parameters(args)
    if args.resume:
        if not args.output_dir.is_dir() or (args.output_dir / 'COMPLETE.json').exists():
            raise ValueError('Resume requires an existing incomplete run')
        check_resume_provenance(json.loads((args.output_dir / 'run_provenance.json').read_text()), args)
    study = {'name': STUDY, 'factor_order': ['G', 'S', 'L'],
             'relation_objective': 'existing_binding_positives_only',
             'context_sequence_loss': False, 'negative_pair_construction': False,
             'expected_train_records': 120000, 'complete_epoch_updates': HORIZON,
             'evaluation': 'source_and_context_strata_with_relation_token_counts_v1',
             'evaluation_micro_batch': EVAL_MICRO, 'loss': 'existing_micro_mean',
             'training_batch_protocol': f'micro{args.micro_batch}_ga{args.grad_accum}_global32',
             'mask_stream': 'seed*1000003+(step-1)*grad_accum+physical_micro_index',
             'p02_mask_or_gradient_equivalence_claim': False,
             'profile_status': 'not_applicable_artificial_cpu' if args.cpu_smoke else 'selected'}
    if not args.cpu_smoke:
        study['batch_plan'] = checked_batch_plan(args)
        if sha(args.data_dir / 'sample_manifest.json') != SAMPLE_SHA:
            raise ValueError('Require the registered 120000-row context subset')
        evidence_path = args.data_dir.parent / 'context_augmentation_preflight.json'
        evidence = json.loads(evidence_path.read_text())
        if (evidence.get('status') != 'passed' or evidence.get('sample_manifest_sha256') != SAMPLE_SHA
                or evidence.get('layout_sha256') != sha(ROOT / 'dllm/pipelines/immune_llada/v3/layout.py')):
            raise ValueError('Context subset input check differs from this source/data version')
        study['input_preflight_sha256'] = sha(evidence_path)
    if not args.resume:
        if args.output_dir.exists():
            logs = {'train.log', 'worker.log', 'stdout.log', 'stderr.log'}
            if (args.output_dir.is_symlink() or not args.output_dir.is_dir()
                    or any(path.name not in logs or not path.is_file() or path.is_symlink()
                           for path in args.output_dir.iterdir())):
                raise ValueError('Fresh output must be new or contain only worker logs')
        else:
            args.output_dir.mkdir(parents=True, exist_ok=False)
    return study


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--variant', required=True, choices=VARIANT_NAMES)
    parser.add_argument('--seed', required=True, type=int, choices=(42, 43))
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=HORIZON)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--micro-batch', required=True, type=int, choices=(16, 32))
    parser.add_argument('--grad-accum', required=True, type=int, choices=(1, 2))
    parser.add_argument('--batch-plan', type=Path)
    parser.add_argument('--batch-plan-sha256')
    parser.add_argument('--cpu-smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-checkpoints', action='store_true')
    return parser.parse_args(argv)


def self_test():
    """Four pure parameter/wiring controls; no imports of ML, data, or models."""
    import unittest
    from types import SimpleNamespace
    def args(**changes):
        values = dict(variant='f111', seed=42, micro_batch=16, grad_accum=2, eval_every=250,
                      steps=2, output_dir=Path('/tmp/large_batch_contract_uncreated'), data_dir=None,
                      batch_plan=None, batch_plan_sha256=None, cpu_smoke=True, resume=False)
        return SimpleNamespace(**{**values, **changes})
    class Contracts(unittest.TestCase):
        def test_allowed_batch_pairs(self):
            for micro, accum in BATCH_PAIRS:
                validate_parameters(args(micro_batch=micro, grad_accum=accum))
            for changes in ({'micro_batch': 4, 'grad_accum': 8}, {'grad_accum': 1},
                            {'micro_batch': 32}, {'eval_every': 25}, {'seed': 44}):
                with self.assertRaises(ValueError):
                    validate_parameters(args(**changes))
        def test_cpu_boundary(self):
            for changes in ({'data_dir': Path('/tmp/unread_real_data')}, {'resume': True},
                            {'steps': 3}, {'batch_plan': Path('/tmp/unread_plan')}, {'cpu_smoke': False}):
                with self.assertRaises(ValueError):
                    validate_parameters(args(**changes))
        def test_validation_stays_micro4(self):
            observed = []
            evaluator = fixed_evaluator(lambda *values: observed.append(values[-2]))
            for micro, _ in BATCH_PAIRS:
                evaluator(None, None, [], [], 'cpu', micro, True)
            self.assertEqual(observed, [4, 4])
        def test_resume_protocol_identity(self):
            previous = {'study': {'name': STUDY, 'evaluation_micro_batch': 4},
                        'micro_batch': 16, 'grad_accum': 2, 'global_batch': 32}
            check_resume_provenance(previous, args())
            with self.assertRaisesRegex(ValueError, 'old p02'):
                check_resume_provenance({**previous, 'study': {'name': 'v7_context_factorial_120k_v1'}}, args())
            with self.assertRaises(ValueError):
                check_resume_provenance(previous, args(micro_batch=32, grad_accum=1))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps({'status': 'passed', 'cases': result.testsRun, 'scope': 'stdlib_parameter_contract_only',
                      'GPU_profile_executed': False, 'training_executed': False}))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ['--self-test']:
        return self_test()
    args = parse_args(argv)
    study = prepare(args)
    sys.path.insert(0, str(ROOT))
    from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel, VARIANTS
    from scripts.research import run_v7_architecture_pilot as core
    from scripts.research.context_pilot_metrics import evaluate
    if tuple(VARIANTS) != VARIANT_NAMES:
        raise ValueError('Registered architecture matrix differs')
    try:
        core.run(args, horizon=HORIZON, model_factory=FactorialModel,
                 evaluator=fixed_evaluator(evaluate), study=study, expected_train_records=120000)
    except BaseException:
        core.write_json(args.output_dir / 'FAILURE.json',
                        {'status': 'failed', 'study': STUDY, 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
