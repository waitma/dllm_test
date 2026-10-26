"""Fixed pool-member dispatcher for one original-model synthetic world8 group.

After activating pllm, source preparation has no execution side effects:
  python -I -B /project_c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/gpu_pool_dispatcher_v1.py --prepare-stage
This prepares reviewable registry/runner bytes and a group mapping only after
the actual source closure is frozen. It never publishes, submits, enqueues,
cancels or restarts anything. Existing workers eventually supply the exact
five runner arguments for one of eight fixed aliases. Actual member execution
also requires root's separately verified launch_gate.json and all original
tasks to have finished; it has no prepared-data or real-training interface.
Every old identity delegates to the unchanged, SHA-bound prior dispatcher.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import sys
import time
import traceback
import uuid


ROOT = Path('/project_c20250601/251105016/project/dllm_test')
OLD = ROOT/'output/generic_eight_worker_group_integration_20261002'
OWN = OLD/'actual_gpu_original_Trainer_v1'
SELF = OWN/'gpu_pool_dispatcher_v1.py'
BOOTSTRAP = OWN/'gpu_worker_bootstrap_v1.py'
COLD = OWN/'gpu_worker_cold_rank_v1.py'
BUDGET = OWN/'execution_budget.json'
BUDGET_SHA256 = '4397c4fec564113d9871523760c8e4df5531426c2d715d9ba7254b30d8637d5d'
FREEZE = OWN/'source_freeze_before_CPU.json'
PLAN = OWN/'group_launch_plan.json'
GATE = OWN/'launch_gate.json'
POOL = ROOT/'output/immune_research_pool_c20250601_20260930'
RUNNER = ROOT/'scripts/run_immune_research.py'
REGISTRY = ROOT/'dllm/pipelines/immune_llada/research/model.py'
PUBLISHED = ROOT/'scripts/architecture_prototypes/gpu_pool_dispatcher_v1.py'
LEGACY = ROOT/'output/generic_remaining_slot_learning_integration_20261002/pool_slot_learning_dispatcher_v1.py'
LEGACY_SHA256 = '49f9c7e45bca01289cbf87e638e2b0c2f38a301565227a965934ae3fa9ba2df9'
RUNTIME_MANIFEST_SHA256 = '2ef8f6777120434c0371083838d3ad03d259e807153d7798a745197e7e9cdefd'
MEMBERS = tuple('g01_r21_world8_member_'+str(index) for index in range(8))
FALSE_FLAGS = ('execution_ready', 'GPU_admitted', 'data_admitted', 'training_admitted', 'production_admitted',
               'real_training_started', 'biological_quality_claim')
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    path = Path(value)
    require(path.is_absolute() and path == path.resolve()
        and not any(item.is_symlink() for item in (path, *path.parents)),
        'An absolute canonical nonsymlink path is required')
    return path


def sha(value):
    path = canonical(value)
    require(path.is_file(), 'A regular source/evidence file is required')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'Duplicate JSON key')
            value[key] = item
        return value
    return json.loads(canonical(path).read_text(), object_pairs_hook=pairs,
        parse_constant=lambda item: require(False, 'Nonfinite JSON value'))


def write_once(path, value):
    path = canonical(path)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def load_fixed(path, expected):
    require(sha(path) == expected, 'Frozen helper source differs: '+str(path))
    module = runpy.run_path(str(path), run_name='_fixed_original_GPU_pool_dependency')
    require(sha(path) == expected, 'Helper changed during import')
    return module


def legacy_module():
    return load_fixed(LEGACY, LEGACY_SHA256)


def selected_variant(argv):
    values = []
    for index, value in enumerate(argv):
        if value == '--variant':
            require(index+1 < len(argv), 'Missing variant value')
            values.append(argv[index+1])
        elif value.startswith('--variant='):
            values.append(value.partition('=')[2])
    require(len(values) == 1 and values[0], 'Exactly one variant is required')
    return values[0]


def source_binding():
    require(sha(BUDGET) == BUDGET_SHA256, 'Bounded synthetic phase budget changed')
    freeze = read_json(FREEZE)
    sources = freeze.get('source_sha256')
    require(type(sources) is dict and sources and all(
        type(name) is str and type(value) is str and re.fullmatch('[0-9a-f]{64}', value)
        and sha(name) == value for name, value in sources.items()),
        'Actual complete current source closure is required')
    require(freeze.get('budget_sha256') == BUDGET_SHA256,
        'Source freeze belongs to a different finite budget')
    for path in (SELF, BOOTSTRAP, COLD):
        require(sources.get(str(path)) == sha(path), 'Required fixed source is absent from freeze')
    current = canonical(__file__)
    require(current in (SELF, PUBLISHED) and sha(current) == sha(SELF),
        'Dispatcher must be the exact frozen private source or its published copy')
    bootstrap = load_fixed(BOOTSTRAP, sources[str(BOOTSTRAP)])
    observed = bootstrap['source_metadata']()
    require(observed['source_sha256'] == sources
        and observed['budget_sha256'] == BUDGET_SHA256
        and observed['runtime_manifest_sha256'] == RUNTIME_MANIFEST_SHA256,
        'Complete measured bootstrap/runtime closure differs')
    return bootstrap, observed


def validate_plan(plan, observed, work_binding):
    fields = {'schema', 'group_id', 'group_directory', 'members', 'jobs', 'work_binding',
              'source_freeze_sha256', 'source_sha256', 'runtime_manifest_sha256',
              'GPU_group_attempts_max', *FALSE_FLAGS}
    require(type(plan) is dict and set(plan) == fields
        and plan.get('schema') == 'generic.original_Trainer_world8_pool_launch_plan.v1',
        'Fixed group plan schema differs')
    require(plan.get('source_freeze_sha256') == sha(FREEZE)
        and plan.get('source_sha256') == observed['source_sha256']
        and plan.get('runtime_manifest_sha256') == RUNTIME_MANIFEST_SHA256
        and plan.get('work_binding') == work_binding,
        'Group plan source, original runtime or fixed work differs')
    members = plan.get('members')
    require(type(members) is list and len(members) == 8
        and [row.get('member_id') for row in members] == list(MEMBERS)
        and all(set(row) == {'member_id', 'job_id'} for row in members),
        'Exactly eight ordered immutable member identities are required')
    jobs = plan.get('jobs')
    require(type(jobs) is dict and len(jobs) == 8
        and len({row['job_id'] for row in members}) == 8,
        'Exactly eight distinct immutable job identities are required')
    for row in members:
        require(re.fullmatch(r'[1-9][0-9]*_'+re.escape(row['member_id'])+r'_[0-9a-f]{8}', row['job_id']),
            'Member job ID does not match its actual worker envelope')
        require(jobs.get(row['job_id']) == dict(variant=row['member_id'],
            mode='profile-synthetic', steps=6), 'Only honest six-update synthetic envelopes are allowed')
    directory = canonical(plan['group_directory'])
    require(directory.parent == OWN/'groups' and directory.name == plan.get('group_id')
        and re.fullmatch(r'original_r21_world8_[0-9]+_[0-9a-f]{8}', directory.name),
        'Group directory must be fresh and privately owned')
    require(plan.get('GPU_group_attempts_max') == 1
        and all(plan.get(field) is False for field in FALSE_FLAGS),
        'Group plan cannot admit formal data/training or retries')
    require(not any(name in plan for name in ('hostname', 'boot_id', 'hardware',
        'data_dir', 'selection_path', 'certificate_sha256')),
        'Root metadata must not invent GPU-host or real-data identities')
    return directory


def launch_gate(observed):
    gate = read_json(GATE)
    require(gate.get('schema') == 'generic.original_Trainer_world8_synthetic_launch_gate.v1'
        and gate.get('status') == 'passed_ready_for_one_synthetic_group'
        and gate.get('original7_terminal_and_accepted') is True
        and gate.get('source_freeze_sha256') == sha(FREEZE)
        and gate.get('source_sha256') == observed['source_sha256']
        and gate.get('runtime_manifest_sha256') == RUNTIME_MANIFEST_SHA256,
        'Separately verified current synthetic launch prerequisites are missing')
    evidence = gate.get('prerequisite_files_sha256')
    require(type(evidence) is dict and evidence and all(
        type(name) is str and type(value) is str and sha(name) == value
        for name, value in evidence.items()), 'Current prerequisite evidence hashes differ')
    require(all(gate.get(field) is False for field in FALSE_FLAGS),
        'Synthetic engineering gate must never promote real-data admission')
    require(not (POOL/'STOP').exists() and not (POOL/'STOP_NOW').exists(), 'Existing pool is stopped')
    return gate


def validate_argv(argv, plan):
    names = {'--variant', '--mode', '--steps', '--profile-dir', '--output-dir'}
    options = [value.partition('=')[0] for value in argv if value.startswith('--')]
    require(set(options) == names and len(options) == len(names),
        'Only five exact worker options are accepted; no data/resume/plan/seed override')
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in sorted(names):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)
    require(args.variant in MEMBERS and args.mode == 'profile-synthetic' and args.steps == '6',
        'Only fixed eight-member/six-update transport is accepted')
    args.output_dir, args.profile_dir = canonical(args.output_dir), canonical(args.profile_dir)
    require(args.output_dir.parent == POOL/'runs' and not args.output_dir.exists()
        and args.profile_dir == POOL/'profiles'/args.variant,
        'Member output/profile ownership differs')
    rows = [row for row in plan['members'] if row['member_id'] == args.variant]
    require(len(rows) == 1 and args.output_dir.name == rows[0]['job_id'],
        'Actual worker job does not match the immutable group mapping')
    running = read_json(POOL/'running'/(rows[0]['job_id']+'.json'))
    require(running == plan['jobs'][rows[0]['job_id']], 'Existing worker has not claimed this exact job')
    require(re.fullmatch('[0-7]', os.environ.get('CUDA_VISIBLE_DEVICES', '')),
        'Exactly one actual existing-worker physical GPU mask is required')
    require('torch' not in sys.modules, 'Member supervisor must enter without ML import')
    return args


def _complete_group(summary):
    counts = ('joined_member_count', 'registered_rank_count',
              'recorded_rank_exit_count', 'member_cleanup_count')
    return (all(summary.get(field) == 8 for field in counts)
        and summary.get('abort') is None
        and summary.get('terminal') == 'succeeded'
        and summary.get('coordinator_terminal') == 'succeeded')


def run_member(argv):
    bootstrap, observed = source_binding()
    plan = read_json(PLAN)
    directory = validate_plan(plan, observed, bootstrap['WORK_BINDING'])
    launch_gate(observed)
    args = validate_argv(argv, plan)
    started, deadline = time.monotonic(), time.monotonic()+1800
    args.output_dir.mkdir()
    try:
        if os.environ['CUDA_VISIBLE_DEVICES'] == '0':
            require(not directory.exists(), 'Never reuse or retry a previous GPU group')
            bootstrap['create_intent'](directory, members=plan['members'],
                work_binding=bootstrap['WORK_BINDING'], join_timeout_seconds=180.,
                heartbeat_timeout_seconds=30., startup_timeout_seconds=180.)
        else:
            until = min(deadline, time.monotonic()+180)
            while not (directory/'intent.json').is_file():
                require(time.monotonic() < until, 'Actual mask0 parent did not create the fixed intent')
                time.sleep(.1)
        intent = bootstrap['read_intent'](directory)
        require(intent['members'] == plan['members'], 'Actual-host intent differs from group mapping')
        deadline = min(deadline, intent['group_deadline_monotonic'])
        cold = load_fixed(COLD, observed['source_sha256'][str(COLD)])
        receipt = cold['member_parent'](plan_path=directory/'intent.json',
            member_id=args.variant, output_dir=directory/args.variant)
        write_once(args.output_dir/'member_process.json', receipt)
        require(receipt.get('status') == 'completed', 'Actual rank exit/descendant cleanup failed')
        while True:
            summary = bootstrap['group_summary'](directory, intent['intent_sha256'])
            if _complete_group(summary):
                break
            require(summary.get('abort') is None
                and summary.get('terminal') not in ('failed', 'aborted')
                and summary.get('coordinator_terminal') not in ('failed', 'aborted'),
                'Actual group failed; no automatic member replacement is permitted')
            require(time.monotonic() < deadline, 'Eight actual member cleanup receipts did not complete')
            time.sleep(.1)
        require(source_binding()[1] == observed, 'Software changed during member execution')
        launch_gate(observed)
        require(time.monotonic() <= deadline, 'Actual group exceeded its complete1800-second budget')
        outer = dict(kind='original_Trainer_world8_synthetic_member_transport',
            job_id=args.output_dir.name, member_id=args.variant,
            group_directory=str(directory), intent_sha256=intent['intent_sha256'],
            source_sha256=observed['source_sha256'], group_summary=summary,
            actual_member_receipt_sha256=sha(args.output_dir/'member_process.json'),
            declared_actual_updates_per_rank=6, completed_pilot_stage=False,
            elapsed_seconds=time.monotonic()-started,
            **{field: False for field in FALSE_FLAGS})
        write_once(args.output_dir/'summary.json', outer)
        write_once(args.output_dir/'COMPLETE.json', dict(status='completed', kind=outer['kind'],
            summary_sha256=sha(args.output_dir/'summary.json'),
            group_directory=str(directory), intent_sha256=intent['intent_sha256'],
            **{field: False for field in FALSE_FLAGS}))
        return 0
    except BaseException:
        error = traceback.format_exc()
        if (directory/'intent.json').is_file():
            try:
                actual = bootstrap['read_intent'](directory)
                bootstrap['abort_group'](directory, actual['intent_sha256'], error[-2000:])
            except BaseException:
                error += '\nGroup-abort publication: '+traceback.format_exc()
        write_once(args.output_dir/'FAILED.json', dict(status='failed', error=error,
            automatic_retry=False, **{field: False for field in FALSE_FLAGS}))
        raise


def _registry_node(source):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'VARIANTS' for target in node.targets)]
    require(len(nodes) == 1 and isinstance(nodes[0].value, ast.Dict),
        'Original registry must remain one literal dictionary')
    values = ast.literal_eval(nodes[0].value)
    require(type(values) is dict and all(type(name) is str for name in values),
        'Original registry literal differs')
    return nodes[0], values


def cleanup_owned_validation_checkpoints(group_directory, review_path):
    """Delete only reviewed temporary synthetic checkpoints after all real ranks exit.

    This is an explicit root maintenance API, never a worker-startup or retry
    path. The actual root review must bind every remaining checkpoint byte.
    Preserve native JSON metadata and deletion hashes alongside the review.
    """
    bootstrap, observed = source_binding()
    directory, review_path = canonical(group_directory), canonical(review_path)
    require(directory.parent == OWN/'groups' and directory.is_dir()
        and review_path.is_relative_to(OWN) and review_path.suffix == '.json',
        'Cleanup is limited to this phase\'s reviewed owned synthetic group')
    review = read_json(review_path)
    require(review.get('schema') == 'generic.original_Trainer_world8_root_review.v1'
        and review.get('status') == 'passed_original_model_synthetic_conformance'
        and review.get('group_directory') == str(directory)
        and review.get('source_sha256') == observed['source_sha256']
        and review.get('real_training_started') is False,
        'Actual independent root review is required before checkpoint deletion')
    intent = bootstrap['read_intent'](directory)
    require(_complete_group(bootstrap['group_summary'](directory, intent['intent_sha256'])),
        'Every actual rank and member must finish cleanup before deleting state')
    bound_files = review.get('input_files_sha256')
    require(type(bound_files) is dict and bound_files, 'Complete actual reviewed input hashes are required')
    checkpoints = [directory/'trainer'/name/('checkpoint-'+str(step))
        for name, step in (('continuous', 3), ('stage2', 2), ('resumed', 3))]
    inventories = []
    for checkpoint in checkpoints:
        checkpoint = canonical(checkpoint)
        require(checkpoint.is_dir(), 'Reviewed temporary checkpoint is missing')
        provenance = read_json(checkpoint/'run_provenance.json')
        binding = provenance.get('eight_rank_binding', {})
        require(provenance.get('fixture_only') is True
            and provenance.get('real_training_started') is False
            and binding.get('training_ready') is False
            and binding.get('certificate_sha256') is None
            and binding.get('scope_certificate_sha256') is None,
            'Formal training or recovery checkpoints must never be deleted here')
        inventory = {}
        for file in sorted(checkpoint.rglob('*')):
            canonical(file)
            require(not file.is_symlink(), 'Checkpoint cleanup refuses symlink entries')
            if file.is_dir():
                continue
            require(file.is_file() and bound_files.get(str(file)) == sha(file),
                'Every actual checkpoint file must match the root review')
            inventory[str(file)] = sha(file)
        inventories.append(dict(checkpoint=str(checkpoint), files_sha256=inventory))
    evidence = directory/'reviewed_checkpoint_cleanup'
    require(not evidence.exists(), 'Temporary checkpoint cleanup cannot be retried')
    evidence.mkdir()
    for checkpoint, inventory in zip(checkpoints, inventories):
        metadata = evidence/checkpoint.parent.name
        metadata.mkdir()
        for file in sorted(checkpoint.rglob('*.json')):
            target = metadata/file.relative_to(checkpoint)
            target.parent.mkdir(parents=True, exist_ok=True)
            with file.open('rb') as reader, target.open('xb') as writer:
                shutil.copyfileobj(reader, writer)
            require(sha(target) == inventory['files_sha256'][str(file)],
                'Preserved native checkpoint metadata changed during copy')
    write_once(evidence/'before_deletion.json', dict(review_path=str(review_path),
        review_sha256=sha(review_path), checkpoint_inventories=inventories,
        group_directory=str(directory), real_training_started=False))
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)
    result = dict(schema='generic.original_Trainer_world8_checkpoint_cleanup.v1',
        removed_checkpoint_directories=[str(path) for path in checkpoints],
        all_owned_validation_checkpoints_removed=not any(path.exists() for path in checkpoints),
        actual_training_recovery_state_touched=False, native_JSON_metadata_preserved=True,
        review_path=str(review_path), review_sha256=sha(review_path),
        **{field: False for field in FALSE_FLAGS})
    require(result['all_owned_validation_checkpoints_removed'], 'Temporary checkpoint deletion is incomplete')
    write_once(evidence/'cleanup_receipt.json', result)
    return result


def prepare_stage_metadata_without_publication():
    """Generate exact review bytes and eight fixed IDs; no actual-host intent or queue write."""
    bootstrap, observed = source_binding()
    stage = OWN/'stage_v1'
    require(not stage.exists() and not PLAN.exists(), 'Never overwrite an existing stage/group mapping')
    protected = read_json(OWN/'protected_before.json')
    for path in (REGISTRY, RUNNER):
        require(protected.get(str(path)) == sha(path), 'Original publication target changed before staging')
    registry_source = REGISTRY.read_text()
    node, previous = _registry_node(registry_source)
    require(not any(name in previous for name in MEMBERS), 'Member aliases must be new')
    lines = registry_source.splitlines(keepends=True)
    line = lines[node.end_lineno-1]
    require(line.strip() == '}', 'Registry closing line cannot be staged exactly')
    additions = ''.join('    '+repr(name)+': '+repr(dict(launcher_only=True,
        architecture_refined=True, model_variant='r21_refined_head_gate',
        synthetic_world8_member_only=True, source_profile_seed=43,
        synthetic_learning_seed=42)) + ',\n' for name in MEMBERS)
    staged_registry = ''.join(lines[:node.end_lineno-1])+additions+''.join(lines[node.end_lineno-1:])
    _, after = _registry_node(staged_registry)
    require({name: after[name] for name in previous} == previous
        and set(after)-set(previous) == set(MEMBERS), 'Only eight literal member additions are allowed')
    old_functions = [ast.dump(item, include_attributes=False) for item in ast.parse(registry_source).body
        if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    new_functions = [ast.dump(item, include_attributes=False) for item in ast.parse(staged_registry).body
        if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    require(old_functions == new_functions, 'Model/registry functions must remain exactly unchanged')
    runner_source = RUNNER.read_text()
    before = "'scripts/architecture_prototypes/pool_slot_learning_dispatcher_v1.py'"
    after_literal = "'scripts/architecture_prototypes/gpu_pool_dispatcher_v1.py'"
    require(runner_source.count(before) == 1, 'Original runner dispatcher literal differs')
    staged_runner = runner_source.replace(before, after_literal, 1)
    ast.parse(staged_runner)
    stamp, suffix = str(time.time_ns()), uuid.uuid4().hex[:8]
    group_id = 'original_r21_world8_'+stamp+'_'+suffix
    members = [dict(member_id=name, job_id=str(time.time_ns())+'_'+name+'_'+uuid.uuid4().hex[:8])
        for name in MEMBERS]
    plan = dict(schema='generic.original_Trainer_world8_pool_launch_plan.v1',
        group_id=group_id, group_directory=str(OWN/'groups'/group_id), members=members,
        jobs={row['job_id']:dict(variant=row['member_id'], mode='profile-synthetic', steps=6)
              for row in members}, work_binding=bootstrap['WORK_BINDING'],
        source_freeze_sha256=sha(FREEZE), source_sha256=observed['source_sha256'],
        runtime_manifest_sha256=RUNTIME_MANIFEST_SHA256, GPU_group_attempts_max=1,
        **{field: False for field in FALSE_FLAGS})
    validate_plan(plan, observed, bootstrap['WORK_BINDING'])
    stage.mkdir()
    (OWN/'groups').mkdir(exist_ok=True)
    changes = []
    for target, filename, contents in ((REGISTRY, 'research_model.py', staged_registry),
            (RUNNER, 'run_immune_research.py', staged_runner),
            (PUBLISHED, 'gpu_pool_dispatcher_v1.py', SELF.read_text())):
        staged = stage/filename
        with staged.open('x') as handle:
            handle.write(contents)
        changes.append(dict(target=str(target), staged_path=str(staged),
            before_sha256=sha(target) if target.is_file() else None,
            after_sha256=sha(staged), published=False))
    write_once(PLAN, plan)
    result = dict(schema='generic.original_Trainer_world8_publication_stage.v1',
        source_freeze_sha256=sha(FREEZE), changes=changes,
        group_plan_path=str(PLAN), group_plan_sha256=sha(PLAN),
        old_registry_values_preserved=True, model_function_AST_preserved=True,
        runner_change='one fixed dispatcher path literal',
        published=False, enqueued=False, actual_host_intent_created=False,
        **{field: False for field in FALSE_FLAGS})
    write_once(stage/'stage_metadata.json', result)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['--prepare-stage']:
        print(json.dumps(prepare_stage_metadata_without_publication(), indent=2, sort_keys=True))
        return 0
    if argv == ['--help']:
        return legacy_module()['main'](argv)
    if selected_variant(argv) not in MEMBERS:
        return legacy_module()['main'](argv)
    return run_member(argv)


if __name__ == '__main__':
    raise SystemExit(main())
