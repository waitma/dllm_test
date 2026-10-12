"""Dispatch the fixed R21 seed43 CE repair through its new full runtime.

Activate pllm; root alone runs the new binding controls:
  python -B -m pytest -p no:cacheprovider /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/test_generic_CE_full_binding.py -q
Only d02_pilot_seed_gpu_r21_seed43_v2 uses this new source. Other identities
delegate to the byte-pinned previous CE tiny dispatcher; D00 remains paused.
The actual source-only runtime manifest and private startup guard are pinned.
Software binding is not profile, GPU, data or training acceptance. The original
entry measures a new profile and retains the full release, normal exit and
descendant cleanup protocol under its total 3600-second deadline. This parent
does not create output, receipts, data certificates or real-training requests.
Publication and execution remain root-owned actions after source review.
"""
import hashlib
import os
from pathlib import Path
import re
import runpy
import sys
from types import SimpleNamespace


ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
OWN = ROOT/'output/generic_CE_full_release_20261002'
PRIVATE_ROOT = OWN/'private_sources'
POOL = ROOT/'output/immune_research_pool_c20250601_20260930'
SELF = 'pool_generic_CE_full_dispatcher_v1.py'
BASE = ROOT/'scripts/architecture_prototypes/pool_seed_release_dispatcher_v2.py'
BASE_SHA256 = 'd5ba656d12965436494d00ef9df662a6d1956e23404c9343cc42517c21109a8d'
LEGACY = ROOT/'output/generic_ce_layout_repair_20261002/pool_generic_CE_repair_dispatcher_v1.py'
LEGACY_SHA256 = 'a38a9837708947a2a7700fc83f29c1155429e53cb6f23645623b1eadead7d283'
ALIAS = 'd02_pilot_seed_gpu_r21_seed43_v2'
RUNTIME = OWN/'runtime_packages/r21_seed_runtime_v2'
RUNTIME_MANIFEST = 'seed_runtime_manifest_v2.json'
RUNTIME_SCHEMA = 'immune.seed_pilot_runtime_source_package.ce_row_major.full.v1'
RUNTIME_MANIFEST_SHA256 = '2ef8f6777120434c0371083838d3ad03d259e807153d7798a745197e7e9cdefd'
RUNTIME_PACKAGE = 'scripts/architecture_prototypes/pilot_seed_runtime_package_v2.py'
RUNTIME_ENTRY = 'scripts/architecture_prototypes/pilot_seed_runtime_entry_v2.py'
RUNTIME_PRODUCER = 'scripts/architecture_prototypes/pilot_seed_gpu_release_producer_v2.py'
MODEL = 'dllm/pipelines/immune_llada/research/model.py'
FIXED_SOURCES = {
    RUNTIME_PACKAGE: '9518ca18812b62016aa666a606392244949edd43990fc8f1ca2a214ea6858358',
    RUNTIME_ENTRY: '25f74614c627f7f7fa12dcbe09031d27add738296e98218b902bcad03c9f1213',
    RUNTIME_PRODUCER: '7c00563ad697977c70267894c6a764e248fa844d589041febce6988882d39e12',
    MODEL: '7e79bfb7d2327c07aa3c39e0bed89997fa627b3fa6e20cb084630b347e886592',
}
GUARD = 'scripts/architecture_prototypes/seed_release_startup/sitecustomize.py'
GUARD_SHA256 = '2f8a2700542bd03fca7198a4efe91c6cc03c1bc84d35945b6febbb10d39e44e0'
GUARDED = '--_seed-guarded-release-launch'
PAUSED = 'd00_cuda_attention_probe_v1'
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    path = Path(value)
    require(path.is_absolute() and path == path.resolve() and not path.is_symlink(),
            'Expected canonical absolute nonsymlink path')
    return path


def source_byte_binding():
    source = canonical(OWN/SELF)
    current = canonical(__file__)
    require(current in (source, ROOT/'scripts/architecture_prototypes'/SELF)
            and source.is_file() and sha(current) == sha(source),
            'Full dispatcher differs from its frozen private source')
    return sha(source)


def base_helpers():
    source_byte_binding()
    source = canonical(BASE)
    require(source.is_file() and sha(source) == BASE_SHA256, 'Original full dispatcher helper changed')
    return SimpleNamespace(**runpy.run_path(str(source), run_name='_CE_full_fixed_helpers'))


def selected_variant(argv):
    return base_helpers().selected_variant(argv)


def identity(alias):
    require(type(alias) is str and alias == ALIAS, 'Only the fixed R21 seed43 full identity is intercepted')
    selected = base_helpers().identity(alias)
    require(selected['method'] == 'r21' and type(selected['seed']) is int and selected['seed'] == 43
            and type(selected['data_seed']) is int and selected['data_seed'] == 43,
            'Fixed full diagnostic seed differs')
    return selected


def validate_argv(argv):
    base = base_helpers()
    args = base.validate_argv(argv)
    identity(args.variant)
    require(re.fullmatch(r'[1-9][0-9]*_'+re.escape(ALIAS)+r'_[0-9a-f]{8}', args.output_dir.name),
            'Fresh output must carry this explicit diagnostic task identity')
    require(base.strict_json(POOL/'pool.json').get('source_root') == str(ROOT), 'Fixed pool source root differs')
    return args


def runtime_for_alias(alias):
    selected = identity(alias)
    before = source_byte_binding()
    base = base_helpers()
    expected = base.digest(RUNTIME_MANIFEST_SHA256)
    root = canonical(RUNTIME)
    path = canonical(root/RUNTIME_MANIFEST)
    require(path.is_file() and sha(path) == expected, 'Actual new full runtime manifest SHA differs')
    manifest = base.strict_json(path)
    bound = manifest.get('diagnostic_identities', {}).get(alias)
    require(type(bound) is dict and type(bound.get('seed')) is int and type(bound.get('data_seed')) is int,
            'Runtime diagnostic seed fields must be exact integers')
    require(manifest.get('schema') == RUNTIME_SCHEMA and manifest.get('method') == 'r21'
            and manifest.get('source_root') == str(ROOT) and manifest.get('runtime_root') == str(root)
            and manifest.get('model_variant') == selected['model_variant']
            and manifest.get('diagnostic_identities') == {ALIAS: selected}, 'New full runtime identity differs')
    files = manifest.get('runtime_files_sha256')
    require(type(files) is dict and all(files.get(name) == expected_sha for name, expected_sha in FIXED_SOURCES.items()),
            'New full bootstrap/model sources are not exactly bound')
    base.verify_files(root, files)  # All actual source hashes precede package code.
    for name, expected_sha in FIXED_SOURCES.items():
        require(sha(canonical(PRIVATE_ROOT/name)) == expected_sha, 'Frozen private full source differs: '+name)
    package = SimpleNamespace(**runpy.run_path(str(root/RUNTIME_PACKAGE), run_name='_CE_full_runtime_verifier'))
    require(package.ROOT == root and package.SCHEMA == RUNTIME_SCHEMA
            and package.OWN_ROOT == OWN and package.PRIVATE_ROOT == PRIVATE_ROOT,
            'New full verifier root/schema differs')
    require(package.verify_runtime(root) == manifest and sha(path) == expected,
            'New full runtime changed during bootstrap')
    require(source_byte_binding() == before, 'Full dispatcher changed during bootstrap')
    return root, root/RUNTIME_ENTRY, manifest


def guard_marker(alias, root):
    selected = identity(alias)
    require(canonical(root) == RUNTIME, 'Startup marker belongs to another runtime')
    return dict(alias=alias, method=selected['method'], seed=selected['seed'], root=str(root),
                guard_sha256=base_helpers().digest(GUARD_SHA256), pid=os.getpid())


def guarded_environment(alias, root):
    guard_marker(alias, root)
    source = canonical(PRIVATE_ROOT/GUARD)
    require(source.is_file() and sha(source) == base_helpers().digest(GUARD_SHA256), 'Private full startup guard SHA differs')
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('IMMUNE_RELEASE_', 'IMMUNE_SEED_RELEASE_'))}
    env.update(PYTHONPATH=os.pathsep.join((str(source.parent), str(root))),
        PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
        IMMUNE_SEED_RELEASE_ALIAS=alias, IMMUNE_SEED_RELEASE_GUARD_SHA256=GUARD_SHA256)
    return env


def binding(alias=ALIAS):
    """Read actual software bindings only; do not initialize or execute a model."""
    root, entry, manifest = runtime_for_alias(alias)
    guarded_environment(alias, root)
    legacy = canonical(LEGACY)
    require(legacy.is_file() and sha(legacy) == LEGACY_SHA256, 'Previous CE tiny dispatcher changed')
    return dict(schema=RUNTIME_SCHEMA, identity=identity(alias), runtime_root=str(root), entry=str(entry),
        runtime_manifest_sha256=RUNTIME_MANIFEST_SHA256, runtime_files_sha256=manifest['runtime_files_sha256'],
        guard_sha256=GUARD_SHA256, dispatcher_sha256=source_byte_binding(),
        base_sha256=BASE_SHA256, legacy_sha256=LEGACY_SHA256)


def run_guarded(argv):
    args = validate_argv(argv)
    root, entry, manifest = runtime_for_alias(args.variant)
    require(getattr(sys, '_immune_seed_release_isolation', None) == guard_marker(args.variant, root),
            'Fresh Python did not complete the exact private full startup guard')
    require(sha(canonical(PRIVATE_ROOT/GUARD)) == base_helpers().digest(GUARD_SHA256), 'Private full guard changed after launch')
    sys.path[:] = [str(root), *(value for value in sys.path if value and Path(value).resolve() != root
        and (not Path(value).resolve().is_relative_to(ROOT) or Path(value).resolve().is_relative_to(root)))]
    require(all(list(getattr(sys.modules.get(name), '__path__', [])) == [str(root/name)]
                for name in ('scripts', 'examples')), 'Project namespace escaped the new full runtime')
    package = SimpleNamespace(**runpy.run_path(str(root/RUNTIME_PACKAGE), run_name='_CE_full_loaded_runtime_verifier'))
    require(package.verify_loaded_runtime(root, manifest) == manifest, 'Loaded full runtime source/namespace differs')
    sys.dont_write_bytecode = True
    sys.argv = [str(entry), *argv]
    runpy.run_path(str(entry), run_name='__main__')
    return 0


def delegate_legacy(argv):
    selected_variant(argv)  # D00 cannot reach any legacy dispatcher.
    source = canonical(LEGACY)
    require(source.is_file() and sha(source) == LEGACY_SHA256, 'Previous CE tiny dispatcher changed')
    return runpy.run_path(str(source), run_name='_CE_full_legacy_dispatcher')['main'](argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['--help']:
        print('Fixed R21 seed43 CE full source; profile-synthetic/1; legacy byte-pinned; D00 paused.')
        return 0
    try:
        if argv and argv[0] == GUARDED:
            require(GUARDED not in argv[1:], 'Duplicate internal full bootstrap flag')
            return run_guarded(argv[1:])
        require(GUARDED not in argv, 'Internal full bootstrap flag is not a runner option')
        alias = selected_variant(argv)
        if alias != ALIAS:
            return delegate_legacy(argv)
        args = validate_argv(argv)
        root, _, _ = runtime_for_alias(args.variant)
        env = guarded_environment(args.variant, root)
        os.chdir(root)
        os.execve(sys.executable, [sys.executable, '-B', '-u', str(canonical(__file__)), GUARDED, *argv], env)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print('CE full dispatch rejected: '+str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
