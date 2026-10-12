"""Run one registered context-enriched architecture ablation, without submitting.

Activate conda pllm; run this absolute script with --variant f000 --seed 42
--data-dir /absolute/sample_context_plus20k --output-dir /absolute/new_run.
Defaults: 120000 rows, global batch 32, one epoch / 3750 updates. CPU entry check:
--variant f111 --seed 42 --cpu-smoke --steps 2 --output-dir /absolute/new_cpu_run.
The original p01 jobs keep using their frozen source and 3125-update schedule.
New tests save metrics only unless --save-checkpoints is explicitly requested.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dllm.pipelines.immune_llada.research_refined.factorial import FactorialModel, VARIANTS
from scripts.research import run_v7_architecture_pilot as core
from scripts.research.context_pilot_metrics import evaluate

SAMPLE_SHA = '1efa27ea19d10bc94143c3977c80dba8c3a29bad921c093ca7f24447635b9478'
STUDY = 'v7_context_factorial_120k_v1'
HORIZON = 3750


def prepare(args):
    if not args.output_dir.is_absolute() or (args.data_dir is not None and not args.data_dir.is_absolute()):
        raise ValueError('Use absolute directory paths')
    if (not 0 <= args.seed < 2**31 or not 1 <= args.steps <= HORIZON
            or args.eval_every != 250 or args.micro_batch != 4 or args.grad_accum != 8):
        raise ValueError('Require valid seed, 1..3750 updates, eval250, micro4 and accumulation8')
    study = {'name': STUDY, 'factor_order': ['G', 'S', 'L'],
             'relation_objective': 'existing_binding_positives_only',
             'context_sequence_loss': False, 'negative_pair_construction': False,
             'expected_train_records': 120000, 'complete_epoch_updates': HORIZON,
             'evaluation': 'source_and_context_strata_with_relation_token_counts_v1'}
    if args.cpu_smoke:
        if args.steps != 2 or args.data_dir is not None:
            raise ValueError('CPU functionality check requires exactly two updates and no real data directory')
    else:
        if args.data_dir is None or core.sha(args.data_dir / 'sample_manifest.json') != SAMPLE_SHA:
            raise ValueError('Require the registered 120000-row context subset')
        evidence_path = args.data_dir.parent / 'context_augmentation_preflight.json'
        evidence = json.loads(evidence_path.read_text())
        layout = ROOT / 'dllm/pipelines/immune_llada/v3/layout.py'
        if (evidence.get('status') != 'passed' or evidence.get('sample_manifest_sha256') != SAMPLE_SHA
                or evidence.get('layout_sha256') != core.sha(layout)):
            raise ValueError('Context subset input check differs from this source/data version')
        study['input_preflight_sha256'] = core.sha(evidence_path)
    if args.resume:
        if not args.output_dir.is_dir() or (args.output_dir / 'COMPLETE.json').exists():
            raise ValueError('Resume requires an existing incomplete run')
    elif args.output_dir.exists():
        worker_logs = {'train.log', 'worker.log', 'stdout.log', 'stderr.log'}
        if (args.output_dir.is_symlink() or not args.output_dir.is_dir()
                or any(path.name not in worker_logs or not path.is_file() or path.is_symlink()
                       for path in args.output_dir.iterdir())):
            raise ValueError('Fresh output must be new or contain only worker logs')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--variant', required=True, choices=tuple(VARIANTS))
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=HORIZON)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--micro-batch', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--cpu-smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-checkpoints', action='store_true')
    args = parser.parse_args()
    study = prepare(args)
    try:
        core.run(args, horizon=HORIZON, model_factory=FactorialModel,
                 evaluator=evaluate, study=study, expected_train_records=120000)
    except BaseException:
        core.write_json(args.output_dir / 'FAILURE.json',
                        {'status': 'failed', 'study': STUDY, 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
