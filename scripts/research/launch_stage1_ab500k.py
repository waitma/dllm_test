"""Validate and run one independently submitted AB500K job.

Activate pllm; frozen THIS_FILE --plan ABS --task teacher_cache_0 --verify-only.
The generated shell calls this coordinator once per node, before torchrun.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.bert_mlm_data import sha, write_json
from scripts.research.launch_stage1_training import bound_json, verify_data


def verify(plan_path, full_data=True, task=None):
    """Check immutable code/data/weights; this does not imply GPU qualification."""
    plan = json.loads(Path(plan_path).read_text())
    if plan.get('schema') != 'immune.stage1.ab500k_pilot.v1':
        raise ValueError('Unsupported AB500K launch schema')
    source = Path(plan['source_root'])
    if Path(__file__).resolve() != source / 'scripts/research/launch_stage1_ab500k.py':
        raise ValueError('Run the frozen launcher through the generated shell')
    for path, digest in plan['sources'].items():
        if sha(path) != digest:
            raise ValueError('Frozen source changed: ' + path)
    subset = bound_json(plan['subset_manifest'], plan['subset_manifest_sha256'])
    from scripts.research.stage1_ab500k_data import validate_subset_files
    validate_subset_files(plan['subset_manifest'])
    qualification = bound_json(plan['subset_validation'], plan['subset_validation_sha256'])
    if qualification.get('status') != 'passed' or qualification.get('manifest_sha256') != plan['subset_manifest_sha256']:
        raise ValueError('Subset has no matching full alignment validation')
    if subset['counts'] != {'train': 500000, 'valid': 4096}:
        raise ValueError('Wrong frozen experiment size')
    if subset['data_index_sha256'] != plan['data_index_sha256']:
        raise ValueError('Wrong parent data index')
    if plan.get('initialization', {}).get('mode') == 'scratch':
        from scripts.research.stage1_ab500k_initialization import verify_scratch_initialization
        verify_scratch_initialization(plan)
    else:
        parent = Path(plan['parent_checkpoint'])
        receipt = bound_json(parent / 'checkpoint_receipt.json', plan['parent_receipt_sha256'])
        if receipt['status'] != 'complete' or receipt['step'] != 287000:
            raise ValueError('Parent checkpoint is not complete BERT287000')
        for name in ('model.safetensors', 'research_config.json', 'immune_config.json', 'vocabulary.json', 'state.json', 'run_provenance.json'):
            if sha(parent / name) != receipt['files'][name]:
                raise ValueError('Parent checkpoint changed: ' + name)
        if receipt['files']['model.safetensors'] != plan['parent_model_sha256']:
            raise ValueError('Parent model hash differs')
    data_report = verify_data(plan) if full_data else {'status': 'not_requested'}
    if plan['teacher_quality_policy'] != 'exploratory_unvalidated':
        raise ValueError('Exploratory teacher policy must remain explicit')
    teacher = plan['teacher_contract']
    weights = Path(plan['model_dir'])
    def check_weight(item):
        name, digest = item
        if sha(weights / name) != digest:
            raise ValueError('Teacher weight/config changed: ' + name)
    if task != 'train_mlm':
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(check_weight, {'config.json': teacher['config_sha256'], **teacher['weight_sha256']}.items()))
    auxiliary = sum(p.stat().st_size for p in Path(plan['subset_manifest']).parent.rglob('*') if p.is_file())
    if auxiliary > plan['auxiliary_budget_bytes']:
        raise ValueError('Subset data exceed auxiliary storage budget')
    report = dict(status='passed', qualification='cpu_inputs_only', data=data_report,
        subset_counts=subset['counts'], subset_bytes=auxiliary, structural_accuracy='unvalidated',
        cache_ready=Path(plan['cache_manifest']).is_file(),
        gpu_checks='pending_user_submission', submitted=False)
    return plan, report


def command(plan_path, plan, task, extra):
    """Return argv only; never invoke a cloud submission command."""
    source = Path(plan['source_root']) / 'scripts/research'
    if task.startswith('teacher_cache_'):
        index = int(task.rsplit('_', 1)[1])
        if index not in range(4):
            raise ValueError('Teacher job index must be 0..3')
        return [plan['teacher_python'], '-I', '-B', '-m', 'torch.distributed.run', '--standalone',
            '--nproc_per_node=4', '--max_restarts=0', str(source / 'stage1_ab500k_teacher.py'),
            '--plan', str(plan_path), '--job-index', str(index), *extra]
    if task in ('train_mlm', 'train_pair_kd'):
        arm = task.removeprefix('train_')
        resume = [] if any(x == '--resume' or x.startswith('--resume=') for x in extra) else ['--resume', 'latest']
        return [plan['student_python'], '-I', '-B', '-m', 'torch.distributed.run', '--standalone',
            '--nproc_per_node=8', '--max_restarts=0', str(source / 'run_stage1_ab500k.py'),
            '--plan', str(plan_path), '--arm', arm, *resume, *extra]
    if task == 'check_cache':
        action = 'check' if Path(plan['cache_manifest']).is_file() else 'index'
        return [plan['student_python'], '-I', '-B', str(source / 'stage1_ab500k_cache.py'), action,
            '--plan', str(plan_path), *extra]
    if task == 'report':
        return [plan['student_python'], '-I', '-B', str(source / 'report_stage1_ab500k.py'),
            '--plan', str(plan_path), '--output', plan['report_output'], *extra]
    raise ValueError('Unknown task: ' + task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--task', required=True, choices=tuple(f'teacher_cache_{i}' for i in range(4))+
        ('train_mlm', 'train_pair_kd', 'check_cache', 'report'))
    parser.add_argument('--verify-only', action='store_true')
    args, extra = parser.parse_known_args()
    plan, report = verify(args.plan, task=args.task)
    source = Path(plan['source_root']) / 'scripts/research'
    if args.verify_only:
        if args.task.startswith('teacher_cache_'):
            result = subprocess.run([plan['teacher_python'], '-I', '-B', str(source / 'stage1_ab500k_teacher.py'),
                '--plan', str(args.plan), '--verify-only'], check=True, capture_output=True, text=True)
            report['teacher_runtime'] = json.loads(result.stdout.strip().splitlines()[-1])
        if args.task == 'train_pair_kd' and not report['cache_ready']:
            report.update(status='not_ready', reason='Generate all 16 teacher shards and run check_cache.sh first')
        elif args.task.startswith('train_'):
            result = subprocess.run([plan['student_python'], '-I', '-B', str(source / 'run_stage1_ab500k.py'),
                '--plan', str(args.plan), '--arm', args.task.removeprefix('train_'), '--verify-only'],
                capture_output=True, text=True)
            if result.returncode:
                report.update(status='not_ready' if result.returncode == 2 else 'failed',
                    reason=result.stderr[-3000:] or result.stdout[-3000:])
            else:
                report['student_admission'] = 'passed_cpu_only'
        write_json(args.plan.parent / (args.task + '.preflight.json'), report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if report['status'] == 'passed' else 2 if report['status'] == 'not_ready' else 1
    # Hold a separate per-task lock for the entire child lifecycle, including checks.
    with (args.plan.parent / (args.task + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(args.plan.parent / (args.task + '.preflight.json'), report)
        argv = command(args.plan, plan, args.task, extra)
        result = subprocess.run(argv, cwd=plan['source_root'])
        return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
