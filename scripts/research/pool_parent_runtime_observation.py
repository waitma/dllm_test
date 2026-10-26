"""Capture the unique capacity controller's real parent pool using stdlib only.

Activate conda pllm. A future early entry calls observe(original_argv) before
dispatching the unique capacity job. For cold controls, run:
  CUDA_VISIBLE_DEVICES='' python -B /project_c20250601/251105016/project/dllm_test/scripts/research/test_pool_parent_runtime_observation.py

Nonmatching jobs return False without filesystem/environment access. Matching
jobs write one exclusive metadata receipt under pool/logs, never in the profile
run directory. No process, listener, GPU query, data loader or training is run.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys
import uuid


ROOT = Path('/project_c20250601/251105016/project/dllm_test')
POOL = ROOT / 'output/immune_research_pool_c20250601_20260930'
POOL_SOURCE = ROOT / 'scripts/immune_research_pool.py'
ALIAS = 'p04_v7_capacity'
JOB_ID = '1790981433321562451_p04_v7_capacity_febb28e3'
COORDINATOR = ROOT / 'output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/eight_rank_coordinator_v2.py'
COORDINATOR_SHA = 'dbaf3e8536370309a42fd2c33d23b020224b62d981451b2038387a21009e4001'
PROCESS_KEYS = {'pid', 'parent_pid', 'parent_start_ticks', 'process_group_id', 'start_ticks', 'hostname', 'boot_id'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(path):
    require(path.is_absolute() and path == path.resolve()
            and not any(item.is_symlink() for item in (path, *path.parents)), 'Canonical nonsymlink path required')
    return path


def values(argv, option):
    return ([argv[index + 1] for index, value in enumerate(argv[:-1]) if value == option]
            + [value.split('=', 1)[1] for value in argv if value.startswith(option + '=')])


def options(argv, allowed, switches=()):
    result, index = {}, 0
    while index < len(argv):
        token = argv[index]
        name, separator, attached = token.partition('=')
        require(name in allowed or name in switches, 'Unexpected command option')
        require(name not in result, 'Duplicate command option')
        if name in switches:
            require(not separator, 'Boolean work switch cannot have a value')
            result[name] = True
        else:
            if separator:
                value = attached
            else:
                index += 1
                require(index < len(argv), 'Missing command option value')
                value = argv[index]
            require(value and not value.startswith('--'), 'Missing command option value')
            result[name] = value
        index += 1
    return result


def coordinator_api():
    require(sha(canonical(COORDINATOR)) == COORDINATOR_SHA, 'Bound V4 process observer source differs')
    return runpy.run_path(str(COORDINATOR), run_name='_read_only_pool_parent_identity')


def read_parent_cmdline(pid):
    raw = Path(f'/proc/{pid}/cmdline').read_bytes()
    require(0 < len(raw) <= 16384 and raw.endswith(b'\0'), 'Invalid bounded parent command metadata')
    argv = [part.decode('utf-8', errors='strict') for part in raw[:-1].split(b'\0')]
    require(all(argv), 'Empty parent command argument')
    return argv


def read_parent_cwd(pid):
    return str(canonical(Path(os.readlink(f'/proc/{pid}/cwd'))))


def validate_parent_argv(argv, cwd):
    require(isinstance(argv, list) and len(argv) >= 4 and
            re.fullmatch(r'python(?:[0-9]+(?:\.[0-9]+)*)?', Path(argv[0]).name), 'Expected a Python pool work invocation')
    index, seen = 1, set()
    while index < len(argv) and argv[index] in ('-u', '-B', '-I'):
        require(argv[index] not in seen, 'Duplicate Python invocation flag')
        seen.add(argv[index]); index += 1
    require(index + 1 < len(argv) and argv[index + 1] == 'work', 'Expected the original pool work action')
    script = Path(argv[index])
    if not script.is_absolute():
        script = Path(cwd) / script
    require(canonical(script) == POOL_SOURCE, 'Parent command resolves to another pool source')
    parsed = options(argv[index + 2:], ('--pool-dir', '--expected-gpus', '--idle-seconds', '--max-seconds', '--job-seconds'),
                     ('--standby-load',))
    require('--pool-dir' in parsed and canonical(Path(parsed['--pool-dir'])) == POOL, 'Parent work points to another pool')
    require(parsed.get('--expected-gpus', '8') == '8', 'Parent pool must declare eight slots')
    for name, default, minimum in (('--idle-seconds', '0', 0), ('--max-seconds', '259200', 1), ('--job-seconds', '7200', 1)):
        value = parsed.get(name, default)
        require(re.fullmatch('[0-9]+', value) and int(value) >= minimum, 'Invalid bounded pool work option')
    return str(script)


def validate_identities(controller, parent):
    for identity in (controller, parent):
        require(isinstance(identity, dict) and set(identity) == PROCESS_KEYS, 'Complete seven-field process identity required')
        require(all(type(identity[key]) is int and identity[key] > 0 for key in PROCESS_KEYS - {'hostname', 'boot_id'})
                and isinstance(identity['hostname'], str) and identity['hostname']
                and isinstance(identity['boot_id'], str) and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', identity['boot_id']),
                'Malformed process identity')
    require(controller['parent_pid'] == parent['pid'] and controller['parent_start_ticks'] == parent['start_ticks']
            and controller['hostname'] == parent['hostname'] and controller['boot_id'] == parent['boot_id'],
            'Controller is not a child of the observed pool identity')


def publish(path, payload, api):
    """Link a fully fsynced temporary file exclusively; never replace a receipt."""
    canonical(path)
    require(path.parent.is_dir(), 'Existing pool logs directory required')
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    created = False
    try:
        api['_exclusive_json'](temporary, payload)
        created = True
        os.link(temporary, path, follow_symlinks=False)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if created:
            temporary.unlink()


def observe(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # This guard is intentionally pure: even malformed unrelated jobs cause no reads.
    if ALIAS not in values(argv, '--variant') or not any(Path(value).name == JOB_ID for value in values(argv, '--output-dir')):
        return False
    flags = ('--variant', '--mode', '--steps', '--profile-dir', '--output-dir')
    parsed = options(argv, flags)
    require(set(parsed) == set(flags) and parsed['--variant'] == ALIAS
            and parsed['--mode'] == 'profile-synthetic' and parsed['--steps'] == '72', 'Unique capacity command envelope differs')
    run_dir = POOL / 'runs' / JOB_ID
    require(canonical(Path(parsed['--output-dir'])) == run_dir
            and canonical(Path(parsed['--profile-dir'])) == POOL / 'profiles' / ALIAS, 'Unique capacity job paths differ')
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(re.fullmatch('[0-7]', mask), 'Require exactly one pool-assigned physical CUDA mask')
    api = coordinator_api()
    controller_pid, parent_pid = os.getpid(), os.getppid()
    controller, parent = api['process_identity'](controller_pid), api['process_identity'](parent_pid)
    require(controller['pid'] == controller_pid and parent['pid'] == parent_pid, 'Observed PID differs')
    validate_identities(controller, parent)
    cwd, pool_argv = read_parent_cwd(parent_pid), read_parent_cmdline(parent_pid)
    resolved = validate_parent_argv(pool_argv, cwd)
    sources = {str(path): sha(canonical(path)) for path in (Path(__file__).resolve(), COORDINATOR, POOL_SOURCE)}
    require(sources[str(COORDINATOR)] == COORDINATOR_SHA, 'Observer source changed during capture')
    require(api['process_identity'](controller_pid) == controller and api['process_identity'](parent_pid) == parent
            and os.getppid() == parent_pid and read_parent_cwd(parent_pid) == cwd
            and read_parent_cmdline(parent_pid) == pool_argv, 'Process identity/cwd/argv changed during capture')
    payload = dict(schema='v7.pool_parent.runtime_observation.v1', observed_at=datetime.now(timezone.utc).isoformat(),
        job_id=JOB_ID, alias=ALIAS, run_dir=str(run_dir), pool_dir=str(POOL), controller=controller,
        pool_process=parent, pool_argv=pool_argv, pool_cwd=cwd, resolved_pool_source=resolved,
        visible_device=mask, source_sha256=sources, source_scope='observed disk source bytes; not proof of parent loaded modules',
        metadata_only=True, training_admitted=False, training_executed=False, GPU_profile_executed=False,
        gpu_hardware_observed=False, actual_data_read=False, credentials_read=False,
        scope='process identity and allowlisted command metadata only; no resource reservation or training approval')
    publish(POOL / 'logs' / (JOB_ID + '.pool_parent_runtime_observation.json'), payload, api)
    return True


if __name__ == '__main__':
    observe()
