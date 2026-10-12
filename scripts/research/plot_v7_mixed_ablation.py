"""Report explicitly selected V7 attempts from pools and independent local runs.

Activate conda pllm, then run:
  python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_mixed_ablation.py --selection /absolute/selection.json --output-prefix /absolute/figures/mixed --endpoint-step 3750

Selection schema v7.mixed_ablation_selection.v1 has batch_plan={path,sha256},
source_cohorts={revision:{path,sha256}} and exactly 22 cells. make_selection()
fills unspecified cells with pending. A pool cell adds attempt_id, source_cohort,
catalog={path,sha256}, pool_dir and job_id. A local cell adds attempt_id,
source_cohort, run_dir, launch_receipt={path,sha256}, and optional exit_evidence.
Source cohorts describe the source dictionary recorded in run_provenance, not
proof that every recorded module executed. No sequences/models/GPUs are read.

Existing direct-Popen launch receipts are accepted but never imply exit zero.
For future tmux runs use launch schema v7.local.tmux_launch.v1: training_argv,
command (the full bash/runner/task/env envelope), runner={path,sha256}, task_name,
log_dir, started={path,sha256}, source_manifest and source_manifest_sha256.
Exit evidence is a file/SHA reference to schema v7.local.tmux_exit.v1 containing
the same launch_receipt reference, runner, task_name, command, started, and
references exit_code, done/failed (the unused sentinel is null). The actual
runner header, sentinel contents and training exit code are checked. Launcher
or pane return codes are never used. This program does not create receipts.

Guarded future launches use v7.local.tmux_launch.v2 and tmux_exit.v2. The launch
also binds admission, ledger, guard_source, guard_argv and guard_execution.
The admission is immutable pre-start configuration; guard_execution records
actual pre-exec checks, not successful training. Exit v2 repeats these five
bindings. Reading a report never calls a guard or performs admission again.
"""
import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shlex


LARGE_PATH = Path(__file__).with_name('plot_v7_large_batch_ablation.py').resolve()
spec = importlib.util.spec_from_file_location('mixed_large_batch_helpers', LARGE_PATH)
large = importlib.util.module_from_spec(spec)
spec.loader.exec_module(large)
base, require = large.base, large.require
SCHEMA = 'v7.mixed_ablation_selection.v1'
LOCAL_ENV = dict(CUDA_VISIBLE_DEVICES='0', USE_TF='0', OMP_NUM_THREADS='8',
    MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1',
    TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
CLEAR_ENV = {'PYTHONPATH', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
             'MASTER_ADDR', 'MASTER_PORT'}


def make_selection(batch_plan, source_cohorts, selected_cells=()):
    """Construct metadata only; explicit selections replace pending design slots."""
    selected = {}
    for item in selected_cells:
        key = (item['variant'], item['seed'])
        require(key not in selected, 'Duplicate selected cell')
        selected[key] = copy.deepcopy(item)
    expected = {(v, s) for v in base.VARIANTS for s in base.SEEDS}
    require(set(selected) <= expected, 'Unexpected selected cell')
    return dict(schema=SCHEMA, batch_plan=batch_plan, source_cohorts=source_cohorts,
        cells=[selected.get((v, s), dict(variant=v, seed=s, transport='pending'))
               for v in base.VARIANTS for s in base.SEEDS])


def canonical(path, directory=False):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve() and not path.is_symlink(),
            f'Use a canonical absolute path: {path}')
    if directory:
        require(path.is_dir(), f'Expected metadata/source directory: {path}')
    return path


def reference(ref, evidence, parse=True):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Require a file/SHA reference')
    return large.reference(canonical(ref['path']), ref['sha256'], evidence, parse=parse)


def recorded_sources(revision, ref, provenance, evidence, cohorts):
    sources = provenance['source_sha256']
    if revision in cohorts:
        require(cohorts[revision]['provenance_source_sha256'] == sources,
                'One provenance source revision contains different source dictionaries')
        return
    manifest = reference(ref, evidence)
    root = canonical(manifest['snapshot_root'], directory=True)
    declared = manifest['source_sha256']
    require(isinstance(declared, dict), 'Missing frozen source dictionary')
    for name, digest in sources.items():
        relative = Path(name)
        require(not relative.is_absolute() and relative.suffix == '.py'
                and '..' not in relative.parts and declared.get(name) == digest,
                f'Provenance source differs from its frozen manifest: {name}')
        path = canonical(root / relative)
        require(path.is_relative_to(root), 'Source path escapes its frozen root')
        large.reference(path, digest, evidence, parse=False)
    cohorts[revision] = dict(manifest_reference=ref, snapshot_root=str(root),
        provenance_source_sha256=sources,
        provenance_source_dictionary_sha256=base.sha(json.dumps(sources, sort_keys=True).encode()),
        scope='Recorded provenance sources; individual module execution is not established')


def validate_training_argv(argv, item, plan, plan_ref, source_root, provenance=None):
    require(isinstance(argv, list) and len(argv) == 25 and all(isinstance(x, str) for x in argv),
            'Expected the exact fresh large-batch training argv')
    require(Path(argv[0]).is_absolute() and argv[1:4] == ['-I', '-B', '-u']
            and argv[4] == str(source_root / 'scripts/research/run_v7_large_batch_ablation.py'),
            'Local launch does not bind the frozen entry')
    options = dict(zip(argv[5::2], argv[6::2]))
    require(len(options) == 10, 'Duplicate training option')
    expected = {'--variant': item['variant'], '--seed': str(item['seed']),
        '--data-dir': options.get('--data-dir'), '--output-dir': item['run_dir'],
        '--steps': '3750', '--eval-every': '250', '--micro-batch': str(plan['micro_batch']),
        '--grad-accum': str(plan['grad_accum']), '--batch-plan': plan_ref['path'],
        '--batch-plan-sha256': plan_ref['sha256']}
    require(options == expected and isinstance(options['--data-dir'], str)
            and Path(options['--data-dir']).is_absolute(), 'Local training protocol/attempt differs')
    if provenance is not None:
        require(str(Path(options['--data-dir']) / 'sample_manifest.json') == provenance['data']['path'],
                'Local launch/provenance data path differs')
    return argv


def runner_header(log_dir, command, task, evidence):
    path = canonical(log_dir / 'run.log')
    require(path.is_file(), 'Missing actual runner log header')
    with path.open('rb') as stream:
        lines = [stream.readline(16384) for _ in range(3)]
    require(all(line.endswith(b'\n') for line in lines), 'Incomplete runner header')
    text = [line.decode() for line in lines]
    require(text[0].rstrip().endswith(' START ' + task) and text[1].startswith('  cmd:    ')
            and shlex.split(text[1][10:]) == command[3:] and text[2].startswith('  TMPDIR: '),
            'Runner header does not bind the actual managed command')
    raw = b''.join(lines)
    evidence[str(path)] = dict(sha256=base.sha(raw), bytes=len(raw), scope='first_three_lines_only')


def guarded_local_launch(launch, evidence):
    """Validate archived v2 metadata without executing any admission helper."""
    admission = reference(launch['admission'], evidence)
    reference(launch['guard_source'], evidence, parse=False)
    require(admission.get('schema') == 'v7.local.metrics_only_admission.v1'
            and admission.get('study') == large.STUDY
            and admission.get('automatic_retry') is False,
            'Guarded admission schema/study/retry differs')
    require(admission.get('guard_source') == launch['guard_source']
            and admission.get('ledger') == launch['ledger']
            and admission.get('runner') == launch['runner']
            and admission.get('task_name') == launch['task_name']
            and admission.get('log_dir') == launch['log_dir']
            and admission.get('source_freeze') == dict(path=launch['source_manifest'],
                sha256=launch['source_manifest_sha256']), 'Guarded admission/launch binding differs')
    managed = reference(admission['managed_local_plan'], evidence)
    reference(admission['original_catalog'], evidence)
    reference(admission['pool_adapter_source'], evidence, parse=False)
    cell = admission.get('variant'), admission.get('seed')
    candidates = [row for row in managed['commands'] if (row['variant'], row['seed']) == cell]
    require(len(candidates) == 1 and managed['runner'] == launch['runner'], 'Guarded managed candidate differs')
    candidate = candidates[0]
    argv = launch['training_argv']
    require(candidate['training_argv'] == argv == admission.get('training_argv')
            and all(candidate[k] == admission[k] for k in ('task_name', 'output_dir', 'log_dir')),
            'Guarded underlying training command differs')
    options = dict(zip(argv[5::2], argv[6::2]))
    require(options.get('--variant') == cell[0] and options.get('--seed') == str(cell[1])
            and options.get('--output-dir') == admission['output_dir']
            and admission['batch_plan'] == dict(path=options.get('--batch-plan'),
                sha256=options.get('--batch-plan-sha256')), 'Guarded command/cell/batch differs')
    ledger = reference(launch['ledger'], evidence)
    cells = ledger.get('cells')
    require(ledger.get('schema') == 'v7.metrics_only.ablation_claims.v1'
            and ledger.get('study') == large.STUDY and isinstance(cells, list) and len(cells) == 22,
            'Guarded full claim ledger differs')
    expected = {(v, s) for v in base.VARIANTS for s in base.SEEDS}
    require(all(type(row.get('seed')) is int for row in cells)
            and {(row['variant'], row['seed']) for row in cells} == expected,
            'Guarded ledger contains duplicate/unknown cells')
    claim = next(row for row in cells if (row['variant'], row['seed']) == cell)
    require(claim.get('state') == 'local_owned' and len(claim.get('attempts', [])) == 1,
            'Guarded local cell is not exclusively claimed')
    attempt = claim['attempts'][0]
    require(attempt.get('transport') == 'local'
            and (attempt.get('variant'), attempt.get('seed')) == cell
            and all(attempt.get(k) == admission[k] for k in ('task_name', 'output_dir', 'log_dir')),
            'Guarded attempt identity differs')
    lock = attempt['lock_file']
    reference(lock, evidence, parse=False)
    lock_path = Path(lock['path'])
    require(lock_path.parent == Path(launch['ledger']['path']).parent / 'cell_locks'
            and lock_path.name == f'{cell[0]}_seed{cell[1]}.lock' and lock_path.stat().st_size == 0,
            'Guarded shared lock binding differs')
    device_lock = admission['local_device_lock']
    reference(device_lock, evidence, parse=False)
    require(device_lock['path'] == str(Path(launch['ledger']['path']).parent / 'local_gpu0.lock')
            and Path(device_lock['path']).stat().st_size == 0, 'Guarded local device lock differs')
    guard_argv = [argv[0], '-I', '-B', '-u', launch['guard_source']['path'],
        '--admission-path', launch['admission']['path'], '--admission-sha256', launch['admission']['sha256'],
        '--ledger-path', launch['ledger']['path'], '--ledger-sha256', launch['ledger']['sha256'],
        '--variant', cell[0], '--seed', str(cell[1])]
    require(launch.get('guard_argv') == guard_argv, 'Guarded invocation differs')
    require(launch['guard_execution']['path'] == str(Path(launch['log_dir']) / 'guard_execution.json'),
            'Guard execution path differs')
    execution = reference(launch['guard_execution'], evidence)
    require(execution.get('schema') == 'v7.local.metrics_only_guard_execution.v1'
            and execution.get('scope') == 'pre_exec_admission_only'
            and execution.get('CUDA_VISIBLE_DEVICES') == LOCAL_ENV['CUDA_VISIBLE_DEVICES']
            and all(execution.get(k) == launch[k] for k in ('admission', 'ledger', 'guard_source',
                'task_name', 'training_argv'))
            and (execution.get('variant'), execution.get('seed')) == cell,
            'Actual guard pre-exec receipt differs')
    identity, locked = execution['process_identity'], execution['lock_identity']
    require(type(identity.get('pid')) is int and identity['pid'] > 0
            and type(identity.get('start_ticks')) is int and identity['start_ticks'] > 0
            and isinstance(identity.get('boot_id'), str) and bool(identity['boot_id'])
            and identity.get('argv') == guard_argv
            and all(type(locked.get(k)) is int and locked[k] >= 0 for k in ('st_dev', 'st_ino'))
            and locked.get('inheritable') is True, 'Actual guard identity/lock receipt differs')
    device_locked = execution['device_lock_identity']
    require(all(type(device_locked.get(k)) is int and device_locked[k] >= 0 for k in ('st_dev', 'st_ino'))
            and device_locked.get('inheritable') is True, 'Actual guard device lock receipt differs')
    return guard_argv


def tmux_launch(launch, evidence):
    runner = launch['runner']
    reference(runner, evidence, parse=False)
    require(Path(runner['path']).name == 'run_in_tmux.sh', 'Require the recorded existing tmux runner')
    task, command = launch['task_name'], launch['command']
    require(isinstance(task, str) and re.fullmatch('[A-Za-z0-9_][A-Za-z0-9_.-]*', task), 'Invalid task identity')
    require(isinstance(command, list) and command[:4] == ['bash', runner['path'], task, 'env'],
            'Managed launch envelope differs')
    tail, cleared, environment = command[4:], [], {}
    while len(tail) >= 2 and tail[0] == '-u':
        cleared.append(tail[1]); tail = tail[2:]
    while tail and '=' in tail[0]:
        name, value = tail[0].split('=', 1)
        require(name not in environment, 'Duplicate environment assignment')
        environment[name] = value; tail = tail[1:]
    underlying = (guarded_local_launch(launch, evidence)
        if launch.get('schema') == 'v7.local.tmux_launch.v2' else launch['training_argv'])
    require(set(cleared) == CLEAR_ENV and len(cleared) == len(CLEAR_ENV)
            and environment == LOCAL_ENV and tail == underlying,
            'Managed launch environment/underlying argv differs')
    log_dir = canonical(launch['log_dir'], directory=True)
    require(log_dir == canonical(runner['path']).parents[1] / 'logs' / task,
            'Managed log directory does not bind runner/task')
    require(launch['started']['path'] == str(log_dir / 'started'), 'Started sentinel path differs')
    reference(launch['started'], evidence, parse=False)
    runner_header(log_dir, command, task, evidence)


def local_launch(item, plan, plan_ref, source_ref, evidence, provenance=None):
    launch = reference(item['launch_receipt'], evidence)
    require(launch.get('source_manifest') == source_ref['path']
            and launch.get('source_manifest_sha256') == source_ref['sha256']
            and launch.get('checkpoint_saving') is False, 'Launch/source/retention identity differs')
    manifest = reference(source_ref, evidence)
    argv = launch.get('training_argv', launch.get('command'))
    validate_training_argv(argv, item, plan, plan_ref, canonical(manifest['snapshot_root'], directory=True), provenance)
    if provenance is not None:
        require(provenance.get('checkpoint_saving') is False, 'Local attempt is not metrics-only')
        if 'versions' in launch:
            require(launch['versions']['torch'] == provenance['torch_version'], 'Launch/runtime torch identity differs')
    if launch.get('schema') in ('v7.local.tmux_launch.v1', 'v7.local.tmux_launch.v2'):
        tmux_launch(launch, evidence)
    else:
        require(launch.get('kind') == 'independent_fresh_attempt'
                and type(launch.get('pid')) is int and launch['pid'] > 0,
                'Require a real direct-Popen or managed tmux launch record')
    return launch


def local_exit(item, launch, evidence):
    if item.get('exit_evidence') is None:
        return None, None
    require(launch.get('schema') in ('v7.local.tmux_launch.v1', 'v7.local.tmux_launch.v2'),
            'Direct-Popen launch has no managed training exit receipt')
    receipt = reference(item['exit_evidence'], evidence)
    version = launch['schema'].rsplit('.', 1)[-1]
    require(receipt.get('schema') == 'v7.local.tmux_exit.' + version
            and receipt.get('launch_receipt') == item['launch_receipt'], 'Exit/launch receipt binding differs')
    keys = ('runner', 'task_name', 'command', 'started')
    if version == 'v2':
        keys += ('admission', 'ledger', 'guard_source', 'guard_argv', 'guard_execution')
    for key in keys:
        require(receipt.get(key) == launch[key], f'Exit/start identity differs: {key}')
    log_dir = Path(launch['log_dir'])
    code_ref = receipt['exit_code']
    require(code_ref['path'] == str(log_dir / 'exit_code'), 'Training exit code path differs')
    reference(code_ref, evidence, parse=False)
    raw = (log_dir / 'exit_code').read_text()
    require(re.fullmatch(r'(0|[1-9][0-9]{0,2})\n', raw) is not None and int(raw) <= 255,
            'Invalid actual training exit code')
    code = int(raw)
    wanted, unwanted = ('done', 'failed') if code == 0 else ('failed', 'done')
    require(receipt.get(unwanted) is None and not (log_dir / unwanted).exists()
            and not (log_dir / unwanted).is_symlink(), 'Contradictory completion/failure sentinel')
    ref = receipt[wanted]
    require(ref['path'] == str(log_dir / wanted), 'Terminal sentinel path differs')
    reference(ref, evidence, parse=False)
    start = datetime.fromisoformat(Path(launch['started']['path']).read_text().strip().replace('Z', '+00:00'))
    end = datetime.fromisoformat((log_dir / wanted).read_text().strip().replace('Z', '+00:00'))
    require(start.tzinfo is not None and end.tzinfo is not None and end >= start, 'Terminal timestamp precedes start')
    return code, receipt


def completion(run, evidence):
    complete, provenance, summary = run['completion'], run['provenance'], run['summary']
    if complete is None:
        return False
    require(provenance is not None and complete.get('status') == 'completed'
            and complete.get('global_step') == 3750 and complete.get('kind') == provenance['kind']
            and complete.get('formal_training') is False and complete.get('biological_quality_claim') is False,
            'Completion is not a native complete large-batch cycle')
    require(run['points'] and run['points'][-1] == summary and summary['step'] == 3750
            and summary['records'] == summary['subset_train_records'] == 120000
            and summary['epoch'] == 1 and summary['position'] == 0, 'Native completion/epoch endpoint differs')
    for key, filename in [('summary', 'summary.json'), ('provenance', 'run_provenance.json')]:
        require(complete[key + '_sha256'] == evidence[str(Path(run['run_dir']) / filename)]['sha256'],
                'Completion artifact SHA differs')
    return True


def collect(selection_path, endpoint_step=3750):
    require(type(endpoint_step) is int and endpoint_step in (500, 3750), 'Choose explicit step 500 or 3750')
    evidence, warnings, cohorts, pool_cache, runs, identities, initial = {}, [], {}, {}, [], {}, {}
    selection = base.read_json(canonical(selection_path), evidence)
    require(selection.get('schema') == SCHEMA, 'Unexpected mixed selection schema')
    plan_ref = selection['batch_plan']
    plan, bound_plan = large.selected_plan(canonical(plan_ref['path']), plan_ref['sha256'], evidence)
    refs = selection['source_cohorts']
    require(isinstance(refs, dict) and all(re.fullmatch('[A-Za-z0-9_.-]+', x) for x in refs),
            'Invalid provenance source revision names')
    expected = {(v, s) for v in base.VARIANTS for s in base.SEEDS}
    cells, seen, attempts, directories = selection['cells'], set(), set(), set()
    require(isinstance(cells, list) and len(cells) == 22, 'Selection must explicitly contain all 22 slots')
    protocol, endpoints, budgets = None, {}, {}
    for item in cells:
        key = (item['variant'], item['seed'])
        require(type(key[1]) is int and key in expected and key not in seen, 'Unexpected/duplicate design slot')
        seen.add(key)
        run = dict(variant=key[0], seed=key[1], transport=item['transport'], attempt_id=None, job_id=None,
            provenance_source_revision=None, status='pending', run_dir=None, points=[], provenance=None,
            progress=None, completion=None, summary=None, failure=None, pool_result=None,
            launch_receipt=None, exit_receipt=None, actual_returncode=None,
            artifact_complete=False, exit_verified=False, terminal_verified=False)
        if item['transport'] == 'pending':
            require(set(item) == {'variant', 'seed', 'transport'}, 'Pending slot must not masquerade as an observed attempt')
            runs.append(run); continue
        require(item['transport'] in ('pool', 'local'), 'Unsupported transport')
        attempt, revision = item['attempt_id'], item['source_cohort']
        require(isinstance(attempt, str) and re.fullmatch('[A-Za-z0-9_.-]+', attempt)
                and (item['transport'], attempt) not in attempts and revision in refs,
                'Invalid/duplicate attempt or missing provenance source revision')
        attempts.add((item['transport'], attempt))
        if item['transport'] == 'pool':
            catalog = item['catalog']; reference(catalog, evidence)
            pool = canonical(item['pool_dir'], directory=True)
            cache_key = (catalog['path'], catalog['sha256'], str(pool))
            if cache_key not in pool_cache:
                old = large.collect(Path(catalog['path']), pool, Path(plan_ref['path']), plan_ref['sha256'])
                evidence.update(old['input_snapshots']); warnings.extend(old['warnings'])
                pool_cache[cache_key] = {x['job_id']: x for x in old['runs']}
            require(item['job_id'] in pool_cache[cache_key], 'Selected pool job is absent from its original catalog')
            old = pool_cache[cache_key][item['job_id']]
            require((old['variant'], old['seed']) == key, 'Pool catalog/selected slot differs')
            run.update(copy.deepcopy(old)); run['run_dir'] = old['directory']
            result = run['pool_result']
            if result is not None and type(result.get('returncode')) is int:
                run['actual_returncode'] = result['returncode']; run['exit_verified'] = True
        else:
            directory = canonical(item['run_dir'], directory=True)
            run['run_dir'] = str(directory)
            run.update({name: base.read_json(directory / file, evidence, optional=True) for name, file in
                [('provenance', 'run_provenance.json'), ('progress', 'progress.json'),
                 ('completion', 'COMPLETE.json'), ('summary', 'summary.json')]})
            run['points'] = base.read_curve(directory / 'curve.jsonl', evidence, warnings)
            launch = local_launch(item, plan, bound_plan, refs[revision], evidence, run['provenance'])
            code, receipt = local_exit(item, launch, evidence)
            run.update(launch_receipt=launch, exit_receipt=receipt, actual_returncode=code, exit_verified=code is not None)
        run.update(transport=item['transport'], attempt_id=attempt, provenance_source_revision=revision)
        require(run['run_dir'] not in directories, 'Two selected attempts reference the same run directory')
        directories.add(run['run_dir'])
        run['failure'] = base.read_json(Path(run['run_dir']) / 'FAILURE.json', evidence, optional=True)
        provenance = run['provenance']
        if provenance is not None:
            require((provenance['variant'], provenance['seed']) == key, 'Selected attempt/provenance differs')
            current = large.identity(provenance, plan, bound_plan)
            recorded_sources(revision, refs[revision], provenance, evidence, cohorts)
            common = {k: v for k, v in current.items() if k != 'source_sha256'}
            require(protocol is None or common == protocol, 'Cross-revision data/config/training/evaluation protocol differs')
            protocol = common
            seed_sha = provenance['common_initialization_sha256']
            require(isinstance(seed_sha, str) and re.fullmatch('[0-9a-f]{64}', seed_sha)
                    and (key[1] not in initial or initial[key[1]] == seed_sha), 'Same-seed initialization differs')
            initial[key[1]] = seed_sha
            base.validate_points(run)
            identities[base.cell(*key)] = dict(provenance_source_revision=revision,
                source_sha256=provenance['source_sha256'], checkpoint_saving=provenance.get('checkpoint_saving', True))
        else:
            require(not run['points'] and run['completion'] is None, 'Curve/completion without provenance')
        native_complete = completion(run, evidence)
        failed = run['failure'] is not None or (run['actual_returncode'] is not None and run['actual_returncode'] != 0)
        if item['transport'] == 'pool':
            failed = failed or (run['pool_result'] is not None and run['pool_result'].get('state') != 'succeeded')
        run['artifact_complete'] = native_complete and not failed
        run['terminal_verified'] = run['artifact_complete'] and run['actual_returncode'] == 0 and run['exit_verified']
        if failed:
            run['status'] = 'failed'
        elif run['terminal_verified']:
            run['status'] = 'completed_exit_verified'
        elif native_complete:
            run['status'] = 'completed_exit_unconfirmed'
        elif run['exit_verified']:
            run['status'] = 'exited_without_native_completion'
        elif item['transport'] == 'local':
            run['status'] = 'observed_incomplete_exit_unconfirmed'
        run['latest_evaluation_step'] = run['points'][-1]['step'] if run['points'] else None
        point = next((x for x in run['points'] if x['step'] == endpoint_step), None)
        if point is not None:
            exposure = tuple(point[x] for x in ('records', 'input_tokens', 'eligible_tokens', 'selected_aa_terminal_tokens'))
            require(key[1] not in budgets or budgets[key[1]] == exposure, 'Same-seed endpoint exposure/mask budget differs')
            budgets[key[1]] = exposure
            endpoints[base.cell(*key)] = point
        runs.append(run)
    require(seen == expected, 'Missing explicit design slot')
    source_ids = {x['provenance_source_dictionary_sha256'] for x in cohorts.values()}
    coverage = all(run['artifact_complete'] for run in runs)
    report = dict(schema='v7.mixed_ablation_report.v1', created_at=datetime.now(timezone.utc).isoformat(),
        study=large.STUDY, endpoint_step=endpoint_step, endpoint_policy='fixed_requested_step_no_fallback_or_best_step',
        status='complete_artifacts' if coverage else 'partial' if endpoints else 'awaiting_requested_endpoint',
        final=False, legacy_final=False, coverage_complete=coverage,
        all_exit_verified=all(x['exit_verified'] for x in runs),
        all_successful_terminal_verified=all(x['terminal_verified'] for x in runs),
        source_homogeneous=bool(source_ids) and len(source_ids) == 1,
        source_homogeneous_scope='observed_provenance_dictionaries_only',
        endpoint_cells=sorted(endpoints), missing_endpoint_cells=[base.cell(v, s) for v, s in sorted(expected)
            if base.cell(v, s) not in endpoints], runs=runs, endpoints=endpoints,
        batch_plan=bound_plan, selected_batch_plan=plan, common_protocol=protocol,
        common_initialization_by_seed=initial, source_cohorts=cohorts, run_source_identities=identities,
        selection_file=str(selection_path), selection=selection, input_snapshots=evidence, warnings=warnings,
        source_sha256={str(p): base.sha(p.read_bytes()) for p in (Path(__file__).resolve(), LARGE_PATH, large.BASE_PATH)},
        figures=[], confidence_intervals=False, biological_quality_claim=False, formal_training=False,
        limitations=['Mixed source revisions remain explicit; static math checks do not establish bitwise outputs.',
            'Same-seed differences across revisions are observations, not isolated causal architecture effects.',
            'mask_mean averages four mask levels; mask100 is full masking. They are not interchangeable.',
            'Existing binding positives do not evaluate binary binding discrimination.',
            'Only two seeds; no confidence intervals or significance claims. No cross-environment speed comparison.',
            'Native COMPLETE is not an observed process exit; launcher/pane success is not training success.',
            'Missing/pending/shorter runs stay absent at the fixed requested endpoint.'])
    report.update(base.derive(endpoints, endpoint_step))
    by_cell = {base.cell(x['variant'], x['seed']): x for x in runs}
    for delta in report['paired_deltas']:
        a, b = by_cell[base.cell('f000', delta['seed'])], by_cell[base.cell(delta['variant'], delta['seed'])]
        delta.update(mask='mask_mean', baseline_revision=a['provenance_source_revision'],
            observed_revision=b['provenance_source_revision'],
            source_homogeneous_pair=identities[base.cell('f000', delta['seed'])]['source_sha256'] ==
                                    identities[base.cell(delta['variant'], delta['seed'])]['source_sha256'])
    for contrast in report['factorial_contrasts']:
        selected = [identities[base.cell(v, contrast['seed'])] for v in base.FACTORIAL]
        contrast.update(mask='mask_mean', source_revisions=sorted({x['provenance_source_revision'] for x in selected}),
            source_homogeneous_contrast=all(x['source_sha256'] == selected[0]['source_sha256'] for x in selected))
    for delta in report['component_removal_deltas']:
        a, b = identities[base.cell('f111', delta['seed'])], identities[base.cell(delta['variant'], delta['seed'])]
        delta.update(mask='mask_mean', baseline_revision=a['provenance_source_revision'],
            observed_revision=b['provenance_source_revision'], source_homogeneous_pair=a['source_sha256'] == b['source_sha256'])
    return report


def export(prefix, report):
    prefix = canonical(prefix)
    protected = [Path(x['run_dir']) for x in report['runs'] if x['run_dir'] is not None]
    protected += [Path(x['snapshot_root']) for x in report['source_cohorts'].values()]
    require(not any(prefix.is_relative_to(path) for path in protected), 'Report output cannot overwrite a run/source directory')
    targets = [prefix.with_suffix('.json'), prefix.with_suffix('.csv')]
    targets += [prefix.with_name(prefix.name + name).with_suffix('.csv') for name in ('_curves', '_paired')]
    targets += [prefix.with_name(prefix.name + '_architecture').with_suffix('.' + ext) for ext in ('svg', 'pdf', 'png')]
    require(all(str(path) not in report['input_snapshots'] and not path.is_symlink() for path in targets),
            'Report output collides with source/input evidence or a symlink')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []
    for run in report['runs']:
        common = {k: run[k] for k in ('variant', 'seed', 'transport', 'attempt_id', 'job_id',
            'provenance_source_revision', 'status', 'artifact_complete', 'exit_verified', 'terminal_verified')}
        point = report['endpoints'].get(base.cell(run['variant'], run['seed']))
        for mask in ('mask_mean', 'mask100'):
            metrics = {} if point is None else (point['validation']['token_pooled']['mask_mean'] if mask == 'mask_mean'
                else point['validation']['token_pooled']['mask_levels'][mask])
            rows.append(dict(**common, step=report['endpoint_step'], mask=mask, observed=point is not None,
                **{k: metrics.get(k) for k in base.COUNT_FOR}))
        for point in run['points']:
            validation = point['validation']
            strata = [('pooled', 'ALL', validation['token_pooled'], validation['records'])]
            strata += [(kind, name, value, value['records']) for kind, key in
                [('source', 'sources'), ('context', 'context_strata')] for name, value in validation[key].items()]
            for kind, name, aggregate, records in strata:
                for mask in (*base.MASKS, 'mask_mean'):
                    metrics = aggregate['mask_mean'] if mask == 'mask_mean' else aggregate['mask_levels'][mask]
                    curves.append(dict(**common, step=point['step'], records=point['records'],
                        input_tokens=point['input_tokens'], eligible_tokens=point['eligible_tokens'],
                        selected_aa_terminal_tokens=point['selected_aa_terminal_tokens'], stratum_kind=kind,
                        stratum=name, validation_records=records, mask=mask,
                        **{k: metrics.get(k) for k in (*base.COUNT_FOR, *sorted(set(base.COUNT_FOR.values())),
                            *(x + '_sum' for x in base.COUNT_FOR))}))
    base.csv_table(prefix.with_suffix('.csv'), rows, list(rows[0]))
    curve_columns = list(curves[0]) if curves else ['variant', 'seed', 'transport', 'attempt_id', 'step', 'mask']
    base.csv_table(prefix.with_name(prefix.name + '_curves').with_suffix('.csv'), curves, curve_columns)
    pairs = report['paired_deltas']
    base.csv_table(prefix.with_name(prefix.name + '_paired').with_suffix('.csv'), pairs,
        list(pairs[0]) if pairs else ['variant', 'seed', 'step', 'metric', 'mask', 'difference', 'baseline_revision', 'observed_revision'])
    draw(report, prefix)
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def draw(report, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    revisions = sorted(report['source_cohorts'])
    colors = {name: plt.get_cmap('tab10')(i % 10) for i, name in enumerate(revisions)}
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
    by_cell = {base.cell(x['variant'], x['seed']): x for x in report['runs']}
    for axis, mask, title in zip(axes, ('mask_mean', 'mask100'),
                                ('mask_mean: equal average of four masks', 'mask100: full masking')):
        for index, variant in enumerate(base.VARIANTS):
            for seed, offset, marker in ((42, -.14, 'o'), (43, .14, 's')):
                key = base.cell(variant, seed); run = by_cell[key]
                point = report['endpoints'].get(key)
                if point is None:
                    axis.text(1.01, index + offset, f's{seed} missing', fontsize=6, color='#64748b',
                              va='center', transform=axis.get_yaxis_transform())
                    continue
                pooled = point['validation']['token_pooled']
                value = pooled['mask_mean']['aa_loss'] if mask == 'mask_mean' else pooled['mask_levels'][mask]['aa_loss']
                axis.scatter(value, index + offset, marker=marker,
                    color=colors[run['provenance_source_revision']],
                    alpha=1. if run['terminal_verified'] else .55, s=42)
        axis.set_yticks(range(len(base.VARIANTS)), base.VARIANTS)
        axis.set_ylim(len(base.VARIANTS) - .5, -.5)
        axis.set_title(title); axis.set_xlabel('AA cross entropy (lower is better)')
        axis.grid(axis='x', alpha=.2)
        if not report['endpoints']:
            axis.text(.5, .5, 'No observed cells at the requested step', ha='center', transform=axis.transAxes)
    handles = [Line2D([], [], marker='o', linestyle='', color=colors[name], label=name) for name in revisions]
    handles += [Line2D([], [], marker=marker, linestyle='', color='#334155', label=f'seed {seed}')
                for seed, marker in ((42, 'o'), (43, 's'))]
    fig.legend(handles=handles, title='Recorded provenance source revision / seed', loc='lower center',
               bbox_to_anchor=(.5, .04), ncol=max(2, min(4, len(handles))), fontsize=8)
    stage = 'COMPLETE ARTIFACTS (not legacy final)' if report['coverage_complete'] else 'PARTIAL'
    fig.suptitle(f'{stage}: V7 architecture comparison — fixed step {report["endpoint_step"]}; '
                 f'{len(report["endpoint_cells"])}/22 observed cells', fontsize=13)
    fig.text(.5, .012, 'Faded points lack verified successful completion. Missing cells stay blank. '
        'Mixed sources are explicit; two seeds, no significance or biological-quality claim.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .15, .95, .95))
    for extension in ('svg', 'pdf', 'png'):
        path = prefix.with_name(prefix.name + '_architecture').with_suffix('.' + extension)
        fig.savefig(path, dpi=180, metadata={'Creator': 'plot_v7_mixed_ablation.py'})
        report['figures'].append(str(path))
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--endpoint-step', type=int, choices=(500, 3750), default=3750)
    args = parser.parse_args(argv)
    report = collect(args.selection, args.endpoint_step)
    export(args.output_prefix, report)
    print(json.dumps({key: report[key] for key in ('status', 'endpoint_step', 'coverage_complete',
        'all_exit_verified', 'all_successful_terminal_verified', 'source_homogeneous', 'missing_endpoint_cells', 'figures')}))


if __name__ == '__main__':
    main()
