"""Cold dispatcher for one separately authorized capacity protocol revision.

Activate conda pllm. The future pool entry calls main(argv) before ML imports:
--variant p05_v7_capacity_v2 --mode profile-synthetic --steps 36
--profile-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/
immune_research_pool_c20250601_20260930/profiles/p05_v7_capacity_v2
--output-dir <the exact canonical pool run in capacity_job_v2.json>.
prepare(argv) is read-only: no model, queue, subprocess, plan or GPU execution.
The root-created ledger is required; this module never creates or retries it.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import sys

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
OWN = ROOT/'output/v7_large_batch_ablation_20261002'
POOL = ROOT/'output/immune_research_pool_c20250601_20260930'
LEDGER = OWN/'capacity_job_v2.json'
LEDGER_SHA = '3429ab434375501e56860e52dd83ebe6702f436a6b18b54d26752b7ff2efc152'
PREVIOUS = ROOT/'scripts/research/pool_large_batch_dispatch.py'
PREVIOUS_SHA = 'f259fb0f0946fb7434c1f78b06f64e5d186ece7b781fd29dac7f3f0fe0e5d866'
FREEZE = OWN/'source_snapshot_v1.json'
FREEZE_SHA = '2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783'
PROFILE_SHA = 'e44e10f134965299f1c288ed7d606b62eb832436230395112a3e6ddf4c542233'
OLD_JOB = '1790981433321562451_p04_v7_capacity_febb28e3'
OLD_RUN = POOL/'runs'/OLD_JOB
ALIAS = 'p05_v7_capacity_v2'
JOB = {'variant': ALIAS, 'mode': 'profile-synthetic', 'steps': 36}
PROTOCOL = {'variants': ['f000', 'f111', 'cv_segment_global'], 'micro_batches': [4, 16, 32],
    'global_batch': 32, 'warmup_steps': 2, 'measure_steps': 2, 'candidate_seconds': 240,
    'max_seconds': 1800, 'optimizer_update_hard_cap': 100, 'seed': 42}
PRIOR = {
    'ledger': {'path': str(OWN/'capacity_job.json'),
               'sha256': 'ae027e88a841abed0fc0b0e2beb764a42535c111fcb02b3451dc1d248896ae1f'},
    'result': {'path': str(POOL/'results'/f'{OLD_JOB}.json'),
               'sha256': 'd3244877fd1584c4face5066bc00f336b6b9a5fde7da90e3c5b31f49179a69c3'},
    'summary': {'path': str(OLD_RUN/'summary.json'),
                'sha256': 'ff8551e5117f14f7219a4e35875ae41791929bc8ee72bc3bc9d6ae9d91c74693'},
    'measurement': {'path': str(OLD_RUN/'f000_micro4/measurement.json'),
                    'sha256': 'f986ebf3fe9b607ebc46574d26e51bdba116bc6aca3dc17abcb1c20fd4ad9e33'}}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve() and
            not any(part.is_symlink() for part in (path, *path.parents)), 'Canonical nonsymlink path required')
    return path


def sha(path):
    return hashlib.sha256(canonical(path).read_bytes()).hexdigest()


def read_bound(reference):
    path = canonical(reference['path'])
    require(sha(path) == reference['sha256'], 'Bound metadata SHA changed: '+str(path))
    value = json.loads(path.read_text())
    require(sha(path) == reference['sha256'], 'Metadata changed while reading')
    return value


def prior_failure(references):
    require(references == PRIOR, 'Require the fixed original failure references')
    values = {name: read_bound(reference) for name, reference in references.items()}
    ledger, result, summary, measurement = (values[key] for key in ('ledger', 'result', 'summary', 'measurement'))
    old_job = {'variant': 'p04_v7_capacity', 'mode': 'profile-synthetic', 'steps': 72}
    old_argv = ['--variant', 'p04_v7_capacity', '--mode', 'profile-synthetic', '--steps', '72',
                '--profile-dir', str(POOL/'profiles/p04_v7_capacity'), '--output-dir', str(OLD_RUN)]
    require(ledger['job_id'] == OLD_JOB and ledger['job'] == old_job and ledger['run_dir'] == str(OLD_RUN),
            'Prior ledger identity differs')
    require(result['job_id'] == OLD_JOB and result['job'] == old_job and result['state'] == 'failed'
            and type(result['returncode']) is int and result['returncode'] == 1, 'Prior job was not the actual failed attempt')
    require(result['argv'][1:3] == ['-u', str(ROOT/'scripts/run_immune_research.py')]
            and result['argv'][3:] == old_argv, 'Prior pool envelope differs')
    cells = summary['measurements']
    require(summary['status'] == 'failed' and summary['planned_optimizer_updates'] == 72
            and summary['candidate_seconds'] == 180 and summary['max_seconds'] == 1800
            and summary['optimizer_update_hard_cap'] == 100 and len(cells) == 1
            and (cells[0]['variant'], cells[0]['micro_batch'], cells[0]['status']) == ('f000', 4, 'timeout')
            and 'TimeoutExpired' in summary['error'] and summary['profile_script_sha256'] == PROFILE_SHA
            and summary['source_root'] == str(OWN/'source_snapshot_v1'), 'Prior summary is not the first-cell timeout')
    losses = measurement['measured_losses']
    require((measurement['variant'], measurement['micro_batch'], measurement['status']) == ('f000', 4, 'failed')
            and measurement['warmup_steps'] == 2 and measurement['measure_steps'] == 6
            and measurement['completed_optimizer_steps'] == 4 and len(losses) == 2
            and all(type(value) in (float, int) and math.isfinite(value) for value in losses)
            and 'InterruptedError' in measurement['error'] and 'OutOfMemory' not in measurement['error'],
            'Prior measurement does not establish two warmup and two measured updates before timeout')
    return old_argv


def previous_module():
    require(sha(PREVIOUS) == PREVIOUS_SHA, 'Previous capacity dispatcher changed')
    module = runpy.run_path(str(PREVIOUS))
    require(sha(PREVIOUS) == PREVIOUS_SHA, 'Previous dispatcher changed while loading')
    return module


def prepare(argv):
    flags = {'--variant', '--mode', '--steps', '--profile-dir', '--output-dir'}
    require(len(argv) == 10 and set(argv[::2]) == flags, 'Exactly five unique pool options are required')
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--variant', required=True, choices=(ALIAS,))
    parser.add_argument('--mode', required=True, choices=('profile-synthetic',))
    parser.add_argument('--steps', required=True, type=int, choices=(36,))
    parser.add_argument('--profile-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    ledger_sha = sha(LEDGER)
    require(ledger_sha == LEDGER_SHA, 'Require the independently allocated single-attempt v2 ledger')
    ledger = read_bound({'path': str(LEDGER), 'sha256': ledger_sha})
    require(ledger['schema'] == 'v7.capacity.enqueue_intent.v2' and ledger['pool_dir'] == str(POOL)
            and ledger['job'] == JOB and ledger['protocol'] == PROTOCOL
            and ledger['automatic_retry'] is False and type(ledger['max_attempts']) is int
            and ledger['max_attempts'] == 1, 'Capacity v2 ledger protocol differs')
    job_id = ledger['job_id']
    require(isinstance(job_id, str) and re.fullmatch(r'[0-9]+_'+ALIAS+r'_[0-9a-f]{8}', job_id),
            'Require the separately allocated v2 job ID')
    run = canonical(POOL/'runs'/job_id)
    require(canonical(args.profile_dir) == POOL/'profiles'/ALIAS and canonical(args.output_dir) == run
            and ledger['run_dir'] == str(run) and run != OLD_RUN, 'Output must match the unique v2 ledger')
    require(not (POOL/'results'/f'{job_id}.json').exists() and not (POOL/'finished'/f'{job_id}.json').exists(),
            'This single attempt already has a terminal record; retries are forbidden')
    require(not run.exists() or (run.is_dir() and all(path.is_file() and not path.is_symlink()
            and path.name in ('train.log', 'worker.log', 'stdout.log', 'stderr.log') for path in run.iterdir())),
            'Refusing to reuse a prior profile output')
    require(ledger['source_freeze'] == {'path': str(FREEZE), 'sha256': FREEZE_SHA}, 'Frozen source binding differs')
    old_argv = prior_failure(ledger['previous_failure'])
    legacy = previous_module()
    # Reuse only the SHA-bound cold verifier; never invoke the old job/main.
    _, command = legacy['prepare'](old_argv)
    require(command[command.index('--output-dir')+1] == str(OLD_RUN)
            and command[command.index('--measure-steps')+1] == '6'
            and command[command.index('--candidate-seconds')+1] == '180', 'Legacy profile command differs')
    command = list(command)
    for flag, value in (('--output-dir', str(run)), ('--measure-steps', '2'), ('--candidate-seconds', '240')):
        command[command.index(flag)+1] = value
    require(sha(LEDGER) == ledger_sha, 'V2 ledger changed during preparation')
    return args, command


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    selected = [argv[index+1] for index, value in enumerate(argv[:-1]) if value == '--variant']
    selected += [value.split('=', 1)[1] for value in argv if value.startswith('--variant=')]
    if ALIAS not in selected:
        return previous_module()['main'](argv)
    args, command = prepare(argv)
    require(re.fullmatch('[0-7]', os.environ.get('CUDA_VISIBLE_DEVICES', '')), 'Require one pool-assigned GPU mask')
    os.execv(sys.executable, command)


if __name__ == '__main__':
    raise SystemExit(main())
