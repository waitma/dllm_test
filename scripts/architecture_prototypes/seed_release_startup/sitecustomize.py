"""Fixed startup isolation for eight artificial seed diagnostic identities.

Activate pllm; root may run the new focused CPU checks:
  python -B -m pytest -p no:cacheprovider /project_c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/private_sources/scripts/tests/immune_llada/test_pool_seed_dispatcher_v2.py -q
Only the new dispatcher supplies this directory and fixed IMMUNE_SEED_RELEASE_*
values. Ordinary audited Python children inherit this guard; -I/-S or altered
environments are outside its contract. No model, CUDA, data or task code runs.
"""
import os as _os

try:
    import hashlib as _hashlib
    from importlib.machinery import ModuleSpec as _ModuleSpec
    from importlib.util import module_from_spec as _module_from_spec
    from pathlib import Path as _Path
    import sys as _sys

    _original = _Path('/project_c20250601/251105016/project/dllm_test')
    _source = _Path(__file__)
    if _source != _source.resolve() or _source.is_symlink():
        raise ValueError('Seed guard source must be canonical')
    _expected = _os.environ['IMMUNE_SEED_RELEASE_GUARD_SHA256']
    if _hashlib.sha256(_source.read_bytes()).hexdigest() != _expected:
        raise ValueError('Seed guard source SHA differs')
    _identities = {
        'd02_pilot_seed_gpu_r20_seed42_v2': ('r20', 42),
        'd02_pilot_seed_gpu_r20_seed43_v2': ('r20', 43),
        'd02_pilot_seed_gpu_r21_seed42_v2': ('r21', 42),
        'd02_pilot_seed_gpu_r21_seed43_v2': ('r21', 43),
        'd02_pilot_seed_gpu_r24_seed42_v2': ('r24', 42),
        'd02_pilot_seed_gpu_r24_seed43_v2': ('r24', 43),
        'd02_pilot_seed_gpu_r26_seed42_v2': ('r26', 42),
        'd02_pilot_seed_gpu_r26_seed43_v2': ('r26', 43),
    }
    _alias = _os.environ['IMMUNE_SEED_RELEASE_ALIAS']
    if _alias not in _identities:
        raise ValueError('Unknown exact seed diagnostic identity')
    _method, _seed = _identities[_alias]
    _root = _original/'output/immune_multiseed_pilot_engineering_20261001/runtime_packages'/(_method+'_seed_runtime_v2')
    if _root != _root.resolve() or not _root.is_dir() or _root.is_symlink():
        raise ValueError('Fixed seed runtime is missing or noncanonical')
    if any(_name.split('.')[0] in ('scripts', 'examples', 'dllm') for _name in _sys.modules):
        raise ValueError('Project module preceded the seed startup guard')
    _sys.dont_write_bytecode = True
    _kept = []
    for _value in _sys.path:
        if not _value:
            continue
        _path = _Path(_value).resolve()
        if _path == _root or (_path.is_relative_to(_original) and not _path.is_relative_to(_root)):
            continue
        _kept.append(_value)
    _sys.path[:] = [str(_root), *_kept]
    for _name in ('scripts', 'examples'):
        _directory = _root/_name
        if not _directory.is_dir() or (_directory/'__init__.py').exists():
            raise ValueError('Expected fixed seed runtime namespace')
        _spec = _ModuleSpec(_name, loader=None, is_package=True)
        _spec.submodule_search_locations = [str(_directory)]
        _sys.modules[_name] = _module_from_spec(_spec)
    _sys._immune_seed_release_isolation = dict(alias=_alias, method=_method, seed=_seed,
        root=str(_root), guard_sha256=_expected, pid=_os.getpid())
except BaseException as _error:
    try:
        _os.write(2, ('Seed startup rejected: '+str(_error)+'\n').encode())
    finally:
        _os._exit(78)
