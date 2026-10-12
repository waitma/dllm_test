"""Dispatch eight explicit artificial seed diagnostics; never submit a task.

Activate pllm; root alone may run the focused CPU checks:
  python -B -m pytest -p no:cacheprovider /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/private_sources/scripts/tests/immune_llada/test_pool_seed_dispatcher_v2.py -q
Deployment is separate. Real-built runtime manifest SHA values must be bound
once below before this new branch can execute. No passing fixture is supplied.
The strict profile-synthetic/1 envelope means one artificial acceptance
invocation; its measured optimizer windows are reported by the fixed producer.
Other existing identities delegate to the unchanged SHA-pinned dispatcher.
D00 remains paused. This extension has no real-data or training entry point.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys


ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
OWN_ROOT = ROOT/'output/immune_multiseed_pilot_engineering_20261001'
POOL = ROOT/'output/immune_research_pool_c20250601_20260930'
SELF = 'scripts/architecture_prototypes/pool_seed_release_dispatcher_v2.py'
GUARD = 'scripts/architecture_prototypes/seed_release_startup/sitecustomize.py'
GUARD_SHA256 = '4f13aa6aa0cdc522cbd937099f73c596ffbf2566a472a5261b478159f5f8118e'
OLD_DISPATCHER = 'scripts/architecture_prototypes/pool_release_dispatcher.py'
OLD_DISPATCHER_SHA256 = 'cd44da25396852f315c039f07df1ea476e310229852b6c2bab5b8b6e8e2db07a'
RUNTIME_MANIFEST = 'seed_runtime_manifest_v2.json'
RUNTIME_SCHEMA = 'immune.seed_pilot_runtime_source_package.v2'
RUNTIME_PACKAGE = 'scripts/architecture_prototypes/pilot_seed_runtime_package_v2.py'
RUNTIME_ENTRY = 'scripts/architecture_prototypes/pilot_seed_runtime_entry_v2.py'
RUNTIME_MANIFEST_SHA256 = {'r20': '022332af75f06f7a76047d048cfe57b10a224e2d1b450b973ea43d6ec11f99e3', 'r21': 'b6772cc422afe7753046730dc097f1a056a9b314c92ce7027b8b369dd1905c62', 'r24': '2bef3d7e31415a7acb84b94243805b2c7999f58c8522905859991df706633b41', 'r26': '9c9242cdcef252f94c4d0a7354c5a6cde5eb7f2288e2b085ed4f9108ac417594'}
MODELS = {'r20': 'r20_refined_control', 'r21': 'r21_refined_head_gate',
          'r24': 'r24_refined_fixed_scale', 'r26': 'r26_refined_local_global'}
IDENTITIES = {
    'd02_pilot_seed_gpu_r20_seed42_v2': dict(method='r20', seed=42),
    'd02_pilot_seed_gpu_r20_seed43_v2': dict(method='r20', seed=43),
    'd02_pilot_seed_gpu_r21_seed42_v2': dict(method='r21', seed=42),
    'd02_pilot_seed_gpu_r21_seed43_v2': dict(method='r21', seed=43),
    'd02_pilot_seed_gpu_r24_seed42_v2': dict(method='r24', seed=42),
    'd02_pilot_seed_gpu_r24_seed43_v2': dict(method='r24', seed=43),
    'd02_pilot_seed_gpu_r26_seed42_v2': dict(method='r26', seed=42),
    'd02_pilot_seed_gpu_r26_seed43_v2': dict(method='r26', seed=43),
}
GUARDED = '--_seed-guarded-release-launch'
PAUSED = 'd00_cuda_attention_probe_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Missing actual frozen SHA256 binding')
    return value


def canonical(value):
    path = Path(value)
    require(path.is_absolute() and path == path.resolve() and not path.is_symlink(),
            'Expected canonical absolute nonsymlink path')
    return path


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate metadata key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite metadata')
    value = json.loads(canonical(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)
    require(type(value) is dict, 'Expected metadata object')
    return value


def selected_variant(argv):
    values = []
    for index, item in enumerate(argv):
        if item == '--variant':
            require(index+1 < len(argv) and not argv[index+1].startswith('-'), 'Missing variant')
            values.append(argv[index+1])
        elif item.startswith('--variant='):
            values.append(item.partition('=')[2])
    require(len(values) == 1 and values[0], 'Exactly one nonempty variant required')
    require(values[0] != PAUSED, 'Independent CUDA operator branch remains paused')
    return values[0]


def identity(alias):
    require(type(alias) is str and alias in IDENTITIES, 'Unknown explicit seed diagnostic')
    selected = IDENTITIES[alias]
    require(type(selected['seed']) is int and selected['seed'] in (42, 43), 'Invalid exact seed identity')
    method, seed = selected['method'], selected['seed']
    return dict(selected, diagnostic_variant=alias, variant=MODELS[method],
        model_variant=MODELS[method], data_seed=seed,
        transport_variant=f't{method[1:]}_{method}_seed{seed}_certified_pilot_v2')


def validate_argv(argv):
    allowed = {'--variant', '--mode', '--steps', '--profile-dir', '--output-dir'}
    options = [item.partition('=')[0] for item in argv if item.startswith('--')]
    require(set(options) == allowed and len(options) == len(allowed), 'Exactly five unique diagnostic options required')
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for name in sorted(allowed):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)
    identity(args.variant)
    require(args.mode == 'profile-synthetic' and args.steps == '1', 'Only fixed profile-synthetic/1 invocation allowed')
    output, profile = canonical(args.output_dir), canonical(args.profile_dir)
    require(output.parent == POOL/'runs' and output.parent.is_dir() and not output.exists(), 'Fresh output in the fixed pool/runs required')
    require(profile == POOL/'profiles'/args.variant and profile.parent.is_dir(), 'Profile belongs to a different pool/identity')
    args.output_dir, args.profile_dir = output, profile
    return args


def verify_files(root, files):
    require(type(files) is dict and files, 'Missing full runtime source map')
    for name, expected in files.items():
        require(type(name) is str and name and str(Path(name)) == name and not Path(name).is_absolute()
                and '..' not in Path(name).parts, 'Unsafe runtime source path')
        path = canonical(root/name)
        require(path.is_relative_to(root) and path.is_file() and sha(path) == digest(expected), 'Runtime source SHA differs: '+name)


def runtime_for_alias(alias):
    selected = identity(alias)
    method = selected['method']
    expected = digest(RUNTIME_MANIFEST_SHA256[method])
    root = canonical(OWN_ROOT/'runtime_packages'/(method+'_seed_runtime_v2'))
    path = canonical(root/RUNTIME_MANIFEST)
    require(path.is_file() and sha(path) == expected, 'Actual runtime manifest SHA differs or package is missing')
    manifest = strict_json(path)
    bound = manifest.get('diagnostic_identities', {}).get(alias)
    require(type(bound) is dict and type(bound.get('seed')) is int and type(bound.get('data_seed')) is int,
            'Runtime diagnostic seed fields must be exact integers')
    require(manifest.get('schema') == RUNTIME_SCHEMA and manifest.get('method') == method
        and manifest.get('source_root') == str(ROOT) and manifest.get('runtime_root') == str(root)
        and manifest.get('model_variant') == MODELS[method]
        and bound == selected,
        'Runtime manifest identity differs')
    files = manifest.get('runtime_files_sha256')
    require(type(files) is dict and RUNTIME_PACKAGE in files and RUNTIME_ENTRY in files, 'Runtime lacks verifier/entry binding')
    verify_files(root, files)  # No package code is loaded before these actual hashes.
    sys.dont_write_bytecode = True
    namespace = runpy.run_path(str(root/RUNTIME_PACKAGE), run_name='_seed_dispatch_runtime_verifier')
    require(namespace['ROOT'] == root and namespace['SCHEMA'] == RUNTIME_SCHEMA, 'Runtime verifier root/schema differs')
    require(namespace['verify_runtime'](root) == manifest and sha(path) == expected, 'Runtime changed during bootstrap')
    return root, root/RUNTIME_ENTRY, manifest


def guard_marker(alias, root):
    selected = identity(alias)
    return dict(alias=alias, method=selected['method'], seed=selected['seed'], root=str(root),
                guard_sha256=digest(GUARD_SHA256), pid=os.getpid())


def guarded_environment(alias, root):
    source = canonical(ROOT/GUARD)
    require(source.is_file() and sha(source) == digest(GUARD_SHA256), 'New startup guard SHA differs')
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('IMMUNE_RELEASE_', 'IMMUNE_SEED_RELEASE_'))}
    env.update(PYTHONPATH=os.pathsep.join((str(source.parent), str(root))),
        PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
        IMMUNE_SEED_RELEASE_ALIAS=alias, IMMUNE_SEED_RELEASE_GUARD_SHA256=GUARD_SHA256)
    return env


def run_guarded(argv):
    args = validate_argv(argv)
    root, entry, _ = runtime_for_alias(args.variant)
    require(getattr(sys, '_immune_seed_release_isolation', None) == guard_marker(args.variant, root),
            'Fresh Python did not complete the exact seed startup guard')
    require(sha(canonical(ROOT/GUARD)) == digest(GUARD_SHA256), 'Startup guard changed after launch')
    sys.path[:] = [str(root), *(value for value in sys.path if value and Path(value).resolve() != root
        and (not Path(value).resolve().is_relative_to(ROOT) or Path(value).resolve().is_relative_to(root)))]
    require(all(list(getattr(sys.modules.get(name), '__path__', [])) == [str(root/name)]
                for name in ('scripts', 'examples')), 'Project namespace escaped seed runtime')
    sys.dont_write_bytecode = True
    sys.argv = [str(entry), *argv]
    runpy.run_path(str(entry), run_name='__main__')
    return 0


def delegate_legacy(argv):
    selected_variant(argv)  # D00 must never reach the unchanged legacy dispatcher.
    source = canonical(ROOT/OLD_DISPATCHER)
    require(source.is_file() and sha(source) == OLD_DISPATCHER_SHA256, 'Legacy dispatcher SHA differs')
    return runpy.run_path(str(source), run_name='_seed_dispatch_legacy')['main'](argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['--help']:
        print('Eight fixed artificial seed diagnostics: profile-synthetic/1; legacy dispatcher is SHA-pinned; D00 paused.')
        return 0
    try:
        if argv and argv[0] == GUARDED:
            require(GUARDED not in argv[1:], 'Duplicate internal seed bootstrap flag')
            return run_guarded(argv[1:])
        require(GUARDED not in argv, 'Internal seed bootstrap flag is not a runner option')
        alias = selected_variant(argv)
        if alias not in IDENTITIES:
            return delegate_legacy(argv)
        args = validate_argv(argv)
        root, _, _ = runtime_for_alias(alias)
        env = guarded_environment(alias, root)
        os.chdir(root)
        os.execve(sys.executable, [sys.executable, '-B', '-u', str(Path(__file__).resolve()), GUARDED, *argv], env)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print('Seed dispatch rejected: '+str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
