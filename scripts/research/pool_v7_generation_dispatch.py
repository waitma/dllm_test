"""Run one explicitly published V7 CDR task on an existing pool GPU.

Activate conda pllm. The future post-training pool entry calls main(argv).
prepare(argv) only validates metadata and sources. No resource submission,
annotation, optimizer, training, automatic retry, or checkpoint copy occurs.
The live pool accepts only train/profile envelopes; mode=train,steps=1 is a
compatibility envelope, explicitly recorded as zero-update generation evaluation.
The original audited V7 parent remains the envelope's data_dir. Actual evaluation
inputs are separately pinned by the CDR benchmark inventory and task protocol.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
POOL = ROOT/'output/immune_research_pool_c20250601_20261005'
EPOCH = ROOT/'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006'
BASE = EPOCH/'generation_pool_v1'
RELEASE = BASE/'release.json'
DATA = ROOT/'data/prepared/immune_v7_test_scope_final_20261002'
COORDINATOR = ROOT/'output/generic_eight_rank_integration_20261002/eight_rank_coordinator_v1.py'
CELLS = tuple((dataset, mode) for dataset, modes in (
    ('kong', ('cdrh1', 'cdrh2', 'cdrh3')),
    ('sab23', ('cdrl1', 'cdrl2', 'cdrl3', 'cdrh1', 'cdrh2', 'cdrh3'))) for mode in modes)
ALIASES = {f'p07_v7_cdr_{dataset}_{mode}': (dataset, mode) for dataset, mode in CELLS}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve() and path.is_file() and not path.is_symlink(),
            'Canonical regular metadata required: '+str(path))
    return json.loads(path.read_text())


def reference(path):
    return dict(path=str(path), sha256=sha(path))


def checked(value):
    require(reference(Path(value['path'])) == value, 'Bound evidence changed: '+value['path'])
    return read(value['path'])


def write_new(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def completion(release):
    actual = checked(release['training_acceptance'])
    require(release['training_acceptance']['path'] == str(EPOCH/'COMPLETE.json')
            and actual['status'] == 'actual_world8_epoch2_continuation_passed'
            and actual['step'] == 50082 and actual['epoch'] == 2
            and actual['eight_native_restores_and_owned_exits'], 'Real complete two-epoch training is required')
    require(actual['checkpoint'] == release['checkpoint'], 'Fixed endpoint changed')
    side = checked(release['checkpoint'])
    require(side['step'] == 50082 and side['world_size'] == 8 and side['rank_acknowledgments'] == list(range(8)),
            'Native checkpoint is incomplete')
    for item in actual['actual_pool_results']:
        result = checked(item)
        require(result['state'] == 'succeeded' and result['returncode'] == 0, 'Final training exit failed')
    audit = checked(release['data_audit'])
    require(audit['status'] == 'passed' and audit['data_dir'] == str(DATA)
            and all(sha(DATA/name) == value for name, value in audit['metadata'].items()), 'Original audited data metadata changed')
    require(read(DATA/'dataset_manifest.json')['training_ready'] is True, 'Original data readiness changed')


def prepare(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--variant', choices=ALIASES, required=True)
    parser.add_argument('--mode', choices=('train',), required=True)
    parser.add_argument('--steps', type=int, choices=(1,), required=True)
    for name in ('data-dir', 'profile-dir', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    require(args.data_dir == DATA and args.profile_dir == POOL/'profiles'/args.variant
            and args.output_dir.parent == POOL/'runs' and args.output_dir == args.output_dir.resolve()
            and re.fullmatch(r'[0-9]+_'+re.escape(args.variant)+r'_[0-9a-f]{8}', args.output_dir.name),
            'Evaluation ticket envelope differs')
    release = read(RELEASE)
    require(release['schema'] == 'v7.cdr.pool_release.v1' and release['work_type'] == 'generation_evaluation'
            and release['optimizer_updates'] == 0 and release['automatic_retry'] is False,
            'Explicit evaluation release required')
    for name, expected in release['files'].items():
        require(sha(name) == expected, 'Published evaluation dependency changed: '+name)
    completion(release)
    catalog = checked(release['catalog'])
    require(catalog['schema'] == 'v7.cdr.pool_catalog.v1' and len(catalog['jobs']) == len(ALIASES), 'Full nine-task catalog required')
    require({tuple(row['cell']) for row in catalog['jobs']} == set(CELLS)
            and len({row['job_id'] for row in catalog['jobs']}) == len(ALIASES), 'Duplicate or omitted task')
    rows = [row for row in catalog['jobs'] if row['job_id'] == args.output_dir.name]
    require(len(rows) == 1, 'Task is not in the published catalog')
    row = rows[0]
    expected = dict(variant=args.variant, mode='train', steps=1, data_dir=str(DATA))
    require(row['job'] == expected and tuple(row['cell']) == ALIASES[args.variant], 'Task identity differs')
    require(not args.output_dir.exists() and not Path(row['evaluation_output']).exists(), 'No repeated task or result overwrite')
    for folder in ('results', 'finished'):
        require(not (POOL/folder/(args.output_dir.name+'.json')).exists(), 'Evaluation attempt already terminal')
    require(not (BASE/'STOP').exists() and not (POOL/'STOP').exists() and not (POOL/'STOP_NOW').exists(), 'Explicit stop present')
    checkpoint = EPOCH/'run/checkpoint-50082'
    command = [sys.executable, '-I', '-B', str(EPOCH/'generation_source_v3/scripts/downstream/run_v7_cdr_eval.py'),
        '--checkpoint', str(checkpoint), '--training-acceptance', str(EPOCH/'COMPLETE.json'),
        '--dataset', row['cell'][0], '--mode', row['cell'][1], '--device', 'cuda:0',
        '--inventory', str(EPOCH/'generation_benchmark_inventory_v1.json'), '--output-root', row['evaluation_output']]
    return args, row, release, command


def main(argv=None):
    args, row, release, command = prepare(list(sys.argv[1:] if argv is None else argv))
    require(re.fullmatch('[0-7]', os.environ.get('CUDA_VISIBLE_DEVICES', '')), 'One actual pool-assigned GPU required')
    require(read(POOL/'running'/(row['job_id']+'.json')) == row['job'], 'Worker does not own this exact ticket')
    api = runpy.run_path(str(COORDINATOR), run_name='_cdr_actual_process_identity')
    parent = api['process_identity'](os.getppid())
    controller = api['process_identity'](os.getpid())
    require(parent == release['pool_process'] and controller['parent_pid'] == parent['pid']
            and controller['parent_start_ticks'] == parent['start_ticks']
            and controller['boot_id'] == parent['boot_id'] and controller['hostname'] == parent['hostname'],
            'Evaluation is outside the original live pool')
    args.output_dir.mkdir(exist_ok=False)
    write_new(args.output_dir/'EVALUATION_START.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
        work_type='generation_evaluation', optimizer_updates=0, release=reference(RELEASE),
        assigned_gpu=os.environ['CUDA_VISIBLE_DEVICES'], pool_process=parent, controller=controller, command=command))
    environment = dict(os.environ, USE_TF='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'PYTHONPATH'):
        environment.pop(key, None)
    # Keep the evaluator in the same owned process group so the existing pool
    # can reap the entire group on its original timeout or explicit stop.
    result = subprocess.run(command, env=environment, cwd=ROOT, check=False)
    write_new(args.output_dir/'EVALUATION_EXIT.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
        returncode=result.returncode, command=command, work_type='generation_evaluation', optimizer_updates=0))
    require(result.returncode == 0, 'Generation evaluator failed; do not retry automatically')
    output = Path(row['evaluation_output'])
    summary = read(output/'summary.json')
    mode = row['cell'][1]
    require(summary['status'] == 'complete' and summary['coverage'] == 'assigned_mode'
            and summary['assigned_mode'] == mode and set(summary['modes']) == {mode}
            and summary['step'] == 50082, 'Assigned generation task is incomplete')
    sys.path.insert(0, str(EPOCH/'generation_source_v3'))
    io = runpy.run_path(str(EPOCH/'generation_source_v3/downstream/grammar/cdr_result_io.py'), run_name='_cdr_score_only')
    scores = io['score_artifacts'](output/mode)
    require(scores == read(output/mode/'metrics.json') and scores['status'] == 'complete', 'Saved predictions do not reproduce scores')
    write_new(args.output_dir/'COMPLETE.json', dict(status='generation_evaluation_complete',
        work_type='generation_evaluation', optimizer_updates=0, checkpoint=release['checkpoint'],
        summary=reference(output/'summary.json'), metrics=reference(output/mode/'metrics.json'),
        actual_evaluator_exit=reference(args.output_dir/'EVALUATION_EXIT.json')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
