"""Seal the requested second-stage transition after the real parent has finished.

Activate conda pllm; run this file only from its frozen epoch-two package.
The second-stage supervisor calls main() once after the existing 25042-step
supervisor exits successfully. This cannot fabricate a future checkpoint,
relax the original audit, submit a resource job, or restart a parent process.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
PARENT = ROOT/'output/v7_formal_eight_gpu_20261002/where_continuation_20261006'
BASE = ROOT/'output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006'
SOURCE = BASE/'source_snapshot_v1'
MANIFEST = BASE/'source_manifest_v1.json'
MANIFEST_SHA = 'e58e0a5331c55b3ec9a0e8bfc7268455adbc5e852e96c0512078ab8e264d379a'
DATA = ROOT/'data/prepared/immune_v7_test_scope_final_20261002'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def write_new(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    require(not (BASE/'transition_v1.json').exists() and not (BASE/'seal_receipt.json').exists(),
            'Refusing to repeat transition sealing')
    require(not (BASE/'STOP').exists() and not (PARENT/'STOP').exists(), 'Explicit stop is present')
    require(sha(MANIFEST) == MANIFEST_SHA, 'Epoch-two source manifest changed')
    for filename, expected in read(MANIFEST)['files'].items():
        require(sha(Path(filename)) == expected, 'Epoch-two bound source changed')
    parent = read(PARENT/'COMPLETE.json')
    require(parent['step'] == 25042 and parent['status'] == 'actual_world8_continuation_passed'
            and parent['eight_native_restores_and_owned_exits'], 'Real complete parent acceptance is required')
    parent_launch = read(PARENT/'supervisor_launch.json')
    require(not (Path('/proc')/str(parent_launch['pid'])).exists(), 'Existing parent supervisor has not exited')
    parent_api = runpy.run_path(str(PARENT/'launcher_v2/launch_v7_where_segment.py'), run_name='_parent_cleanup_only')
    parent_api['cleanup_checked'](PARENT/'launches_v2/segment_25042', success=True)
    checkpoint = PARENT/'run/checkpoint-25042'
    require(parent['checkpoint'] == ref(checkpoint/'formal_checkpoint.json'), 'Parent checkpoint receipt changed')
    sidecar = read(checkpoint/'formal_checkpoint.json')
    require(sidecar['step'] == 25042 and sidecar['world_size'] == 8
            and sidecar['rank_acknowledgments'] == list(range(8)) and len(sidecar['files']) == 17,
            'Incomplete native parent checkpoint')
    for name, expected in sidecar['files'].items():
        path = checkpoint/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0, 'Missing parent checkpoint member')
        if name.endswith('.json'):
            require(sha(path) == expected, 'Parent checkpoint metadata changed')
    state = read(checkpoint/'trainer_state.json')
    require(state['global_step'] == 25042 and abs(state['epoch']-25042/25041) < 1e-9,
            'Actual parent epoch differs from the checked loader geometry')
    marker = read(PARENT/'run/COMPLETE.json')
    require(marker['step'] == 25042 and marker['binding_sha256'] == sidecar['binding_sha256'],
            'Parent native completion marker differs')
    cpu_log = BASE/'cpu_schedule_resume_offset1_fixed.log'
    cpu = json.loads(cpu_log.read_text().splitlines()[-1])
    require(cpu['status'] == 'passed' and cpu['one_already_consumed_second_epoch_batch_skipped']
            and cpu['resumed_weights_optimizer_scheduler_equal_continuous'], 'CPU continuation check did not pass')
    driver = runpy.run_path(str(SOURCE/'scripts/research/run_v7_formal.py'), run_name='_epoch2_metadata_driver')
    args = SimpleNamespace(source_manifest=MANIFEST, source_manifest_sha256=MANIFEST_SHA,
                           source_root=SOURCE, repo_root=ROOT, data_dir=DATA)
    audit = driver['check_parent'](args)
    require(audit['status'] == 'passed', 'Original read-only data gate rejected epoch two')
    write_new(BASE/'parent_gate.stdout.json', audit)
    for name, original in [('anchor_sidecar.json', checkpoint/'formal_checkpoint.json'),
                           ('parent_provenance.json', PARENT/'run/run_provenance.json'),
                           ('ancestor_provenance.json', PARENT/'parent_provenance.json')]:
        destination = BASE/name
        require(not destination.exists(), 'Existing transition evidence must be reviewed, not overwritten')
        destination.write_bytes(original.read_bytes())
    api = runpy.run_path(str(SOURCE/'scripts/research/v7_epoch_transition.py'), run_name='_epoch2_transition_metadata')
    old = read(BASE/'parent_provenance.json')
    ancestor = read(BASE/'ancestor_provenance.json')
    spec = dict(schema='v7.epoch2.ddp_transition.v1', user_request=ref(BASE/'request_v1.json'),
        old_source_manifest=ref(PARENT/'source_manifest_v1.json'), new_source_manifest=ref(MANIFEST),
        old_provenance=ref(BASE/'parent_provenance.json'), ancestor_provenance=ref(BASE/'ancestor_provenance.json'),
        old_output_dir=str(PARENT/'run'), new_source_root=str(SOURCE), new_output_dir=str(BASE/'run'),
        anchor_checkpoint=str(checkpoint), anchor_sidecar=ref(BASE/'anchor_sidecar.json'),
        lr_extension=api['LR_EXTENSION'],
        retention_owners={str(PARENT/'run'): driver['digest'](old),
            str(ROOT/'output/v7_initial_training_20261005/run'): driver['digest'](ancestor)},
        actual_parent_acceptance=ref(PARENT/'COMPLETE.json'))
    transition = BASE/'transition_v1.json'
    write_new(transition, spec)
    argv = read(PARENT/'initial_driver_argv.json')
    replacements = {'--source-root': str(SOURCE), '--output-dir': str(BASE/'run'),
        '--source-manifest': str(MANIFEST), '--source-manifest-sha256': MANIFEST_SHA,
        '--max-steps': '50082', '--stop-after': '25050', '--resume-from-checkpoint': str(checkpoint),
        '--transition': str(transition), '--transition-sha256': sha(transition)}
    for flag, value in replacements.items():
        require(argv.count(flag) == 1, 'Ambiguous parent argument template: '+flag)
        argv[argv.index(flag)+1] = value
    parsed = driver['parse_args'](argv)
    api['transition'](parsed)
    write_new(BASE/'initial_driver_argv.json', argv)
    receipt = dict(at_utc=datetime.now(timezone.utc).isoformat(), status='actual_parent_and_original_data_gate_passed',
        transition=ref(transition), manifest=ref(MANIFEST), parent=ref(PARENT/'COMPLETE.json'),
        original_parent_data_audit=ref(BASE/'parent_gate.stdout.json'),
        initial_driver_argv=ref(BASE/'initial_driver_argv.json'), sealer=ref(Path(__file__).resolve()))
    write_new(BASE/'seal_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
