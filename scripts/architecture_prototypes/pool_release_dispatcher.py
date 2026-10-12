"""Standard-library-only staged dispatcher; no live entry uses it yet.

Activate pllm; CPU tests:
python -m pytest -q scripts/tests/immune_llada/test_pool_release_dispatcher.py
Future stable entry calls main(); --help prints routing information without
importing Torch, a model, or the diagnostic adapter. No submit/queue operations.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC = 'd00_cuda_attention_probe_v1'
TRAINER_DIAGNOSTIC = 'd01_pilot_runtime_gpu_r20_v1'
TRAINER_ROUTE = 'trainer_runtime_v1'
RUNTIME_NAME = 'r20_r27_pilot_v3_20261001'
RUNTIME_ENTRY = 'scripts/architecture_prototypes/pilot_runtime_entry_v1.py'
RUNTIME_MANIFEST = 'pilot_runtime_manifest.json'
RUNTIME_SCHEMA = 'immune.pilot_runtime_source_package.v1'
RUNTIME_PARENT_SHA256 = '46399204c7018d00726e3ae08142bd4eb28475c8a9a2991b3a5c90725c8824f7'
RUNTIME_ADDITIONS = frozenset({RUNTIME_ENTRY,
    'scripts/architecture_prototypes/pilot_gpu_release_producer_v1.py'})
RUNTIME_ALIASES = ('downstream/benchmark/data',
    'output/research/uptcr_20260930/released_tables/data/finetune/abpm')
CANDIDATE_PACKAGE = 'scripts/architecture_prototypes/pilot_candidate_runtime_package_v1.py'
CANDIDATE_INSPECTOR = 'scripts/architecture_prototypes/pilot_candidate_runtime_inspect_v1.py'
CANDIDATE_POOL = 'output/immune_research_pool_c20250601_20260930'
CANDIDATE_RUNTIMES = {
    'r21_refined_head_gate': dict(package_name='r21_pilot_v1_20261001',
        entry='scripts/architecture_prototypes/pilot_runtime_entry_r21_v1.py',
        producer='scripts/architecture_prototypes/pilot_gpu_release_producer_r21_v1.py', route='candidate_trainer_r21_v1'),
    'r24_refined_fixed_scale': dict(package_name='r24_pilot_v1_20261001',
        entry='scripts/architecture_prototypes/pilot_runtime_entry_r24_v1.py',
        producer='scripts/architecture_prototypes/pilot_gpu_release_producer_r24_v1.py', route='candidate_trainer_r24_v1'),
    'r26_refined_local_global': dict(package_name='r26_pilot_v1_20261001',
        entry='scripts/architecture_prototypes/pilot_runtime_entry_r26_v1.py',
        producer='scripts/architecture_prototypes/pilot_gpu_release_producer_r26_v1.py', route='candidate_trainer_r26_v1'),
}
MODEL_FAMILIES = {
    'r00_r11_20260930': (
        'r00_a4_control', 'r01_span_mask', 'r02_chain_completion', 'r03_chain_balanced_loss',
        'r04_qk_norm', 'r05_register4', 'r06_local_conv7', 'r07_mask_ratio_film', 'r08_vicreg',
        'r09_equal_source', 'r10_role_pair_bias', 'r11_span_qknorm_register'),
    'r12_r19_20260930': (
        'r12_architecture_control', 'r13_head_output_gate', 'r14_fixed_half_attention', 'r15_layerscale',
        'r16_fixed_residual_scale', 'r17_chain_late_fusion', 'r18_chain_local_global', 'r19_value_residual'),
    'r20_r27_20261001': (
        'r20_refined_control', 'r21_refined_head_gate', 'r22_refined_fixed_gate', 'r23_refined_layerscale',
        'r24_refined_fixed_scale', 'r25_refined_late_fusion', 'r26_refined_local_global', 'r27_refined_value_residual'),
}
ENTRIES = dict(zip(MODEL_FAMILIES, ('scripts/run_immune_research.py',
    'scripts/run_immune_architecture.py', 'scripts/run_immune_refined.py')))
MODEL_VARIANTS = {variant: family for family, variants in MODEL_FAMILIES.items() for variant in variants}
DIAGNOSTIC_SOURCES = {
    'scripts/architecture_prototypes/pool_attention_diagnostic.py': '7482ca23b8644a8b065ad1a405b4822e7979fd9da972a32e746a03531f8cb656',
    'scripts/architecture_prototypes/cuda_attention_probe.py': '193ef514bc29683d6579e9c94ec72215df77a5b0aa51ba81e153ba6a881183a0',
    'scripts/architecture_prototypes/attention_backends.py': '9b4225b300020cf3329410de9567038f2d968d2c00516329b87c0eb275ba103a',
}
STARTUP_GUARD = 'scripts/architecture_prototypes/release_startup/sitecustomize.py'
STARTUP_GUARD_SHA256 = 'ab4c1dee12a637c79e93d23a4b413ea3ec982e465b08ca895c04e53d5a7b0396'
GUARDED_LAUNCH = '--_guarded-release-launch'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate manifest key')
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=pairs)
    json.dumps(value, allow_nan=False)
    return value


def read_registry(path):
    assignments = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'VARIANTS' for target in node.targets)]
    require(len(assignments) == 1 and isinstance(assignments[0].value, ast.Dict), 'Exactly one literal registry required')
    keys = [ast.literal_eval(key) for key in assignments[0].value.keys]
    require(all(type(key) is str for key in keys) and len(keys) == len(set(keys)), 'Duplicate/invalid variant registry key')
    result = ast.literal_eval(assignments[0].value)
    require(all(type(value) is dict for value in result.values()), 'Invalid variant options')
    return result


def selected_variant(argv):
    values = []
    for index, value in enumerate(argv):
        if value == '--variant':
            require(index+1 < len(argv) and not argv[index+1].startswith('-'), 'Missing variant value')
            values.append(argv[index+1])
        elif value.startswith('--variant='):
            values.append(value.partition('=')[2])
    require(len(values) == 1, 'Exactly one --variant is required; duplicates rejected')
    require(values[0] in MODEL_VARIANTS or values[0] in (DIAGNOSTIC, TRAINER_DIAGNOSTIC), 'Unknown variant')
    return values[0]


def validate_diagnostic_argv(argv):
    # This repeats the fixed envelope before importing Torch; the typed adapter
    # remains the final path/evidence validator even with an old in-memory worker.
    allowed = {'--variant', '--mode', '--steps', '--profile-dir', '--output-dir'}
    options = [value.partition('=')[0] for value in argv if value.startswith('--')]
    require(set(options) == allowed and len(options) == len(allowed), 'Diagnostic requires exactly five unique options')
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in sorted(allowed):
        parser.add_argument(name, required=True, type=int if name == '--steps' else str)
    args = parser.parse_args(argv)
    require(args.variant == DIAGNOSTIC and args.mode == 'profile-synthetic' and args.steps == 1,
            'Only the fixed d00 profile-synthetic/1 envelope is supported')
    output, profile = Path(args.output_dir), Path(args.profile_dir)
    require(output.is_absolute() and profile.is_absolute() and output == output.resolve()
            and profile == profile.resolve(), 'Diagnostic paths must be canonical absolute paths')
    require(output.parent.name == 'runs' and output.parent.is_dir() and not output.exists(), 'Fresh pool/runs output required')
    require(profile == output.parent.parent/'profiles'/DIAGNOSTIC, 'Diagnostic profile path belongs to another pool/variant')


def validate_trainer_argv(argv):
    """The existing pool command is the entire D01 transport, never a runner API."""
    allowed = {'--variant', '--mode', '--steps', '--profile-dir', '--output-dir'}
    options = [value.partition('=')[0] for value in argv if value.startswith('--')]
    require(set(options) == allowed and len(options) == len(allowed),
            'Trainer diagnostic requires exactly five unique options')
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in sorted(allowed):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)
    require(args.variant == TRAINER_DIAGNOSTIC and args.mode == 'profile-synthetic' and args.steps == '1',
            'Only the fixed d01 profile-synthetic/1 envelope is supported')
    output, profile = Path(args.output_dir), Path(args.profile_dir)
    require(output.is_absolute() and profile.is_absolute() and output == output.resolve()
            and profile == profile.resolve(), 'Trainer diagnostic paths must be canonical absolute paths')
    require(output.parent.name == 'runs' and output.parent.is_dir() and not output.exists(),
            'Fresh pool/runs output required')
    require(profile == output.parent.parent/'profiles'/TRAINER_DIAGNOSTIC,
            'Trainer diagnostic profile path belongs to another pool/variant')


def verify_files(root, hashes):
    require(type(hashes) is dict and hashes, 'Missing source file hashes')
    for relative, expected in hashes.items():
        require(type(relative) is str and not Path(relative).is_absolute(), 'Manifest path must be relative')
        path = root/relative
        require(path == path.resolve() and path.is_relative_to(root) and path.is_file(), 'Missing/noncanonical source file')
        require(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected) and sha(path) == expected,
                f'Source digest differs: {relative}')


def candidate_envelope(argv):
    """Reserve only these model variants' exact profile-synthetic/1 intent."""
    if selected_variant(argv) not in CANDIDATE_RUNTIMES:
        return False
    def values(name):
        return [argv[index+1] if value == name and index+1 < len(argv) else value.partition('=')[2]
                for index, value in enumerate(argv) if value == name or value.startswith(name+'=')]
    # Duplicate/extra options on this reserved intent are rejected by the fixed
    # validator, instead of falling into a historical model runner.
    return 'profile-synthetic' in values('--mode') and '1' in values('--steps')


def validate_candidate_argv(argv):
    allowed = {'--variant', '--mode', '--steps', '--profile-dir', '--output-dir'}
    options = [value.partition('=')[0] for value in argv if value.startswith('--')]
    require(set(options) == allowed and len(options) == len(allowed),
            'Candidate Trainer requires exactly five unique options')
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in sorted(allowed):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)
    require(args.variant in CANDIDATE_RUNTIMES and args.mode == 'profile-synthetic' and args.steps == '1',
            'Only the three fixed candidate profile-synthetic/1 envelopes are supported')
    output, profile = Path(args.output_dir), Path(args.profile_dir)
    pool = ROOT/CANDIDATE_POOL
    require(output.is_absolute() and profile.is_absolute() and output == output.resolve()
            and profile == profile.resolve() and not output.is_symlink() and not profile.is_symlink(),
            'Candidate paths must be canonical absolute nonsymlink paths')
    require(output.parent == pool/'runs' and output.parent.is_dir() and not output.exists(),
            'Fresh candidate output in the fixed pool/runs required')
    require(profile == pool/'profiles'/args.variant and profile.is_dir(),
            'Candidate requires its own existing original model profile')


def candidate_runtime_entry(variant):
    """Validate the fixed 524+4 source package, without importing project/ML code."""
    require(type(variant) is str and variant in CANDIDATE_RUNTIMES, 'Unknown fixed candidate')
    identity = CANDIDATE_RUNTIMES[variant]
    parent, _ = frozen_entry('r20_r27_20261001')
    require(sha(parent/'release_manifest.json') == RUNTIME_PARENT_SHA256, 'Candidate parent manifest SHA differs')
    original = strict_json(parent/'release_manifest.json')
    base_files, measured = original['copied_files_sha256'], original['measured_source_sha256']
    require(len(base_files) == 524 and len(measured) == 39
            and ENTRIES['r20_r27_20261001'] in measured, 'Candidate parent must retain original 524/39 sources')
    family = MODEL_FAMILIES['r20_r27_20261001']
    terminals = original.get('terminal_job_ids')
    require(original.get('family') == list(family) and type(terminals) is dict and set(terminals) == set(family)
            and all(type(value) is str and value for value in terminals.values()), 'Candidate parent terminal identities differ')
    evidence = original.get('archive_evidence_sha256')
    require(type(evidence) is dict and evidence, 'Candidate parent archive evidence missing')
    for name, expected in evidence.items():
        require(type(name) is str and Path(name).is_absolute(), 'Candidate parent evidence must be absolute')
        path = Path(name)
        require(path == path.resolve() and path.is_file() and not path.is_symlink()
                and type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected) and sha(path) == expected,
                'Candidate parent archive evidence differs')
    root = ROOT/'output/immune_pilot_runtimes'/identity['package_name']
    require(root == root.resolve() and root.is_dir() and not root.is_symlink(), 'Fixed candidate runtime missing/noncanonical')
    path = root/RUNTIME_MANIFEST
    require(path.is_file() and not path.is_symlink(), 'Missing candidate runtime manifest')
    manifest = strict_json(path)
    require(manifest.get('schema') == RUNTIME_SCHEMA and manifest.get('runtime_root') == str(root)
            and manifest.get('source_root') == str(ROOT) and manifest.get('base_snapshot') == str(parent)
            and manifest.get('candidate_variant') == variant and manifest.get('package_name') == root.name
            and manifest.get('base_manifest_sha256') == RUNTIME_PARENT_SHA256
            and manifest.get('status') == 'source_package_only_not_gpu_or_data_acceptance'
            and all(manifest.get(key) is False for key in ('gpu_accepted', 'data_accepted', 'training_started')),
            'Candidate runtime identity or source-only status differs')
    require(manifest.get('base_copied_files_sha256') == base_files
            and manifest.get('measured_source_sha256') == measured
            and manifest.get('inherited_terminal_job_ids') == terminals, 'Candidate replaced original parent/source proof')
    required = {identity['entry'], identity['producer'], CANDIDATE_PACKAGE, CANDIDATE_INSPECTOR}
    additions = manifest.get('additions_sha256')
    require(type(additions) is dict and set(additions) == required and not set(additions)&set(base_files),
            'Candidate requires only its four fixed NEW source paths')
    verify_files(ROOT, additions)
    files = {**base_files, **additions}
    require(len(files) == 528 and manifest.get('runtime_files_sha256') == files, 'Candidate full 528 source map differs')
    verify_files(root, files)
    actual = {str(file.relative_to(root)) for package in ('dllm', 'examples', 'scripts')
              for file in (root/package).rglob('*') if file.is_file() and file.suffix in {'.py', '.json', '.yaml', '.yml'}}
    require(actual == set(files), 'Candidate contains unrecorded or missing source/config files')
    aliases = {name:str(ROOT/name) for name in RUNTIME_ALIASES}
    require(manifest.get('external_resource_aliases') == aliases, 'Candidate resource aliases differ')
    links = {str((Path(folder)/name).relative_to(root))
             for folder, directories, names in os.walk(root, followlinks=False)
             for name in directories+names if (Path(folder)/name).is_symlink()}
    require(links == set(aliases), 'Candidate contains extra or missing aliases')
    for name, target in aliases.items():
        alias = root/name
        require(alias.is_symlink() and os.readlink(alias) == target
                and alias.resolve() == Path(target).resolve() and alias.is_dir(), 'Candidate alias redirected or unavailable')
    verify_files(ROOT, {STARTUP_GUARD: STARTUP_GUARD_SHA256})
    return root, root/identity['entry']


def frozen_entry(family):
    require(family in ENTRIES, 'Unknown fixed release family')
    root = ROOT/'output/immune_research_releases'/family
    require(root == root.resolve() and root.is_dir(), 'Required fixed release snapshot is missing or symlinked')
    manifest_path = root/'release_manifest.json'
    require(manifest_path.is_file() and not manifest_path.is_symlink(), 'Missing release manifest')
    manifest = strict_json(manifest_path)
    require(manifest.get('source_root') == str(ROOT), 'Release source root differs')
    measured = manifest.get('measured_source_sha256')
    if family == 'r00_r11_20260930':
        require(manifest.get('kind') == 'immutable_legacy_source' and type(manifest.get('copied_files')) is int
                and manifest['copied_files'] > 0, 'Unknown legacy release format')
        # Historical R00 has only 32 measured hashes, not a full copy manifest.
        hashes = measured
    else:
        require(manifest.get('schema_version') == 1 and manifest.get('frozen_root') == str(root), 'Release manifest identity differs')
        hashes = manifest.get('copied_files_sha256')
        require(type(measured) is dict and measured and type(hashes) is dict
                and all(hashes.get(name) == digest for name, digest in measured.items()), 'Measured/copy source maps differ')
    verify_files(root, hashes)
    require(ENTRIES[family] in hashes, 'Entrypoint is outside the frozen evidence')
    return root, root/ENTRIES[family]


def trainer_runtime_entry():
    """Verify the fixed source package using only stdlib before executing its entry."""
    parent, _ = frozen_entry('r20_r27_20261001')
    require(sha(parent/'release_manifest.json') == RUNTIME_PARENT_SHA256,
            'Trainer runtime parent manifest differs')
    original = strict_json(parent/'release_manifest.json')
    base_files = original['copied_files_sha256']
    require(original.get('family') == list(MODEL_FAMILIES['r20_r27_20261001'])
            and type(original.get('terminal_job_ids')) is dict
            and set(original['terminal_job_ids']) == set(MODEL_FAMILIES['r20_r27_20261001']),
            'Trainer runtime parent lacks eight terminal identities')
    evidence = original.get('archive_evidence_sha256')
    require(type(evidence) is dict and evidence, 'Trainer runtime parent lacks evidence bindings')
    for name, expected in evidence.items():
        require(type(name) is str and Path(name).is_absolute(), 'Parent evidence path must be absolute')
        path = Path(name)
        require(path == path.resolve() and path.is_file() and sha(path) == expected,
                'Trainer runtime parent evidence differs')
    root = ROOT/'output/immune_pilot_runtimes'/RUNTIME_NAME
    require(root == root.resolve() and root.is_dir(), 'Required fixed Trainer runtime is missing or symlinked')
    path = root/RUNTIME_MANIFEST
    require(path.is_file() and not path.is_symlink(), 'Missing Trainer runtime manifest')
    manifest = strict_json(path)
    require(manifest.get('schema') == RUNTIME_SCHEMA and manifest.get('runtime_root') == str(root)
            and manifest.get('source_root') == str(ROOT) and manifest.get('base_snapshot') == str(parent)
            and manifest.get('base_manifest_sha256') == RUNTIME_PARENT_SHA256
            and manifest.get('status') == 'source_package_only_not_gpu_or_data_acceptance'
            and all(manifest.get(key) is False for key in ('gpu_accepted', 'data_accepted', 'training_started')),
            'Trainer runtime identity or source-only status differs')
    require(manifest.get('base_copied_files_sha256') == base_files
            and manifest.get('measured_source_sha256') == original['measured_source_sha256']
            and manifest.get('inherited_terminal_job_ids') == original['terminal_job_ids'],
            'Trainer runtime replaced parent source/proof bindings')
    additions = manifest.get('additions_sha256')
    require(type(additions) is dict and set(additions) == RUNTIME_ADDITIONS
            and not set(additions)&set(base_files), 'Trainer runtime requires only the two fixed NEW additions')
    # The undeployed entry and producer are mutable while being reviewed. A
    # package is eligible only after their canonical source bytes are frozen.
    verify_files(ROOT, additions)
    files = {**base_files, **additions}
    require(manifest.get('runtime_files_sha256') == files, 'Trainer runtime full source map differs')
    verify_files(root, files)
    actual = {str(file.relative_to(root)) for package in ('dllm', 'examples', 'scripts')
              for file in (root/package).rglob('*') if file.is_file() and file.suffix in {'.py', '.json', '.yaml', '.yml'}}
    require(actual == set(files), 'Trainer runtime contains unrecorded or missing source/config files')
    aliases = {name: str(ROOT/name) for name in RUNTIME_ALIASES}
    require(manifest.get('external_resource_aliases') == aliases, 'Trainer runtime resource aliases differ')
    links = {str((Path(folder)/name).relative_to(root))
             for folder, directories, names in os.walk(root, followlinks=False)
             for name in directories+names if (Path(folder)/name).is_symlink()}
    require(links == set(aliases), 'Trainer runtime contains extra or missing aliases')
    for name, target in aliases.items():
        alias = root/name
        require(alias.is_symlink() and os.readlink(alias) == target
                and alias.resolve() == Path(target).resolve() and alias.is_dir(),
                'Trainer runtime alias is redirected or unavailable')
    return root, root/RUNTIME_ENTRY


def route_name(variant, argv=None):
    if argv is not None and candidate_envelope(argv):
        return CANDIDATE_RUNTIMES[variant]['route']
    if variant == TRAINER_DIAGNOSTIC:
        return TRAINER_ROUTE
    return 'diagnostic' if variant == DIAGNOSTIC else MODEL_VARIANTS[variant]


def resolve_route(argv):
    variant = selected_variant(argv)
    if candidate_envelope(argv):
        validate_candidate_argv(argv)
        return candidate_runtime_entry(variant)
    if variant == TRAINER_DIAGNOSTIC:
        validate_trainer_argv(argv)
        return trainer_runtime_entry()
    if variant == DIAGNOSTIC:
        validate_diagnostic_argv(argv)
        verify_files(ROOT, DIAGNOSTIC_SOURCES)
        return ROOT, ROOT/'scripts/architecture_prototypes/pool_attention_diagnostic.py'
    return frozen_entry(MODEL_VARIANTS[variant])


def guarded_environment(root, argv):
    """Only audited ordinary Python children inherit this fixed startup guard."""
    verify_files(ROOT, {STARTUP_GUARD: STARTUP_GUARD_SHA256})
    variant = selected_variant(argv)
    route = route_name(variant, argv)
    return dict(os.environ, PYTHONPATH=os.pathsep.join((str((ROOT/STARTUP_GUARD).parent), str(root))),
                PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                IMMUNE_RELEASE_ROUTE=route, IMMUNE_RELEASE_GUARD_SHA256=STARTUP_GUARD_SHA256)


def run_guarded(argv):
    """Fresh-process entry: verify startup before executing any project code."""
    root, entry = resolve_route(argv)
    verify_files(ROOT, {STARTUP_GUARD: STARTUP_GUARD_SHA256})
    variant = selected_variant(argv)
    route = route_name(variant, argv)
    require(getattr(sys, '_immune_release_isolation', None) == dict(
        root=str(root), route=route, guard_sha256=STARTUP_GUARD_SHA256, pid=os.getpid()),
        'Fresh Python did not complete the fixed startup guard')
    # CPython inserts the script directory after sitecustomize. Here the script
    # is the live stdlib dispatcher, so remove that bootstrap path before runpy.
    sys.path[:] = [str(root), *(value for value in sys.path if value
        and Path(value).resolve() != root and (not Path(value).resolve().is_relative_to(ROOT)
        or Path(value).resolve().is_relative_to(root)))]
    require(all(list(getattr(sys.modules.get(name), '__path__', [])) == [str(root/name)]
                for name in ('scripts', 'examples')), 'Project namespace escaped selected release')
    sys.argv = [str(entry), *argv]
    runpy.run_path(str(entry), run_name='__main__')
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['--help']:
        print('Fixed release dispatcher: --variant NAME [existing runner arguments]')
        print('28 frozen model variants; d01 requires the fixed Trainer runtime, profile-synthetic/steps=1 and pool paths.')
        print('R21/R24/R26 reserve exactly profile-synthetic/steps=1 for their own fixed Trainer source packages.')
        return 0
    try:
        if argv and argv[0] == GUARDED_LAUNCH:
            require(GUARDED_LAUNCH not in argv[1:], 'Duplicate internal bootstrap flag')
            return run_guarded(argv[1:])
        require(GUARDED_LAUNCH not in argv, 'Internal bootstrap flag is not a runner option')
        root, entry = resolve_route(argv)
        environment = guarded_environment(root, argv)
        os.chdir(root)
        # exec preserves the pool's PID/process-group ownership, but replaces all
        # Python modules; no live model is imported before snapshot selection.
        os.execve(sys.executable, [sys.executable, '-u', str(Path(__file__).resolve()),
                                  GUARDED_LAUNCH, *argv], environment)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'Dispatch rejected: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
