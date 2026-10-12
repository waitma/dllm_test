"""Fixed job-local Python startup guard; not a general Python sandbox.

Activate pllm and run scripts/tests/immune_llada/test_pool_release_dispatcher.py.
The dispatcher alone supplies this directory on PYTHONPATH and the two fixed
IMMUNE_RELEASE_* values. Audited ordinary Python children inherit the guard;
-I/-S, changed environments, and arbitrary code are outside this contract.
"""
# Python normally ignores exceptions from sitecustomize. Keep all operational
# imports/checks inside this boundary and terminate before the target can run.
import os as _os

try:
    import hashlib as _hashlib
    from importlib.machinery import ModuleSpec as _ModuleSpec
    from importlib.util import module_from_spec as _module_from_spec
    from pathlib import Path as _Path
    import sys as _sys

    _source = _Path(__file__).resolve()
    _original = _source.parents[3]
    _expected = _os.environ['IMMUNE_RELEASE_GUARD_SHA256']
    if _hashlib.sha256(_source.read_bytes()).hexdigest() != _expected:
        raise ValueError('Startup guard source SHA differs')
    _route = _os.environ['IMMUNE_RELEASE_ROUTE']
    if _route == 'diagnostic':
        _root = _original
    elif _route == 'trainer_runtime_v1':
        _root = _original/'output/immune_pilot_runtimes/r20_r27_pilot_v3_20261001'
    elif _route in ('candidate_trainer_r21_v1', 'candidate_trainer_r24_v1', 'candidate_trainer_r26_v1'):
        _candidates = {'candidate_trainer_r21_v1': 'r21_pilot_v1_20261001',
                      'candidate_trainer_r24_v1': 'r24_pilot_v1_20261001',
                      'candidate_trainer_r26_v1': 'r26_pilot_v1_20261001'}
        _root = _original/'output/immune_pilot_runtimes'/_candidates[_route]
    elif _route in ('r00_r11_20260930', 'r12_r19_20260930', 'r20_r27_20261001'):
        _root = _original/'output/immune_research_releases'/_route
    else:
        raise ValueError('Unknown fixed release route')
    if _root != _root.resolve() or not _root.is_dir():
        raise ValueError('Selected release is missing or noncanonical')
    if any(name.split('.')[0] in ('scripts', 'examples', 'dllm') for name in _sys.modules):
        raise ValueError('Project modules were loaded before the startup guard')

    _kept = []
    for _value in _sys.path:
        if not _value:
            continue
        _path = _Path(_value).resolve()
        if _path == _root:
            continue
        if _path.is_relative_to(_original) and not _path.is_relative_to(_root):
            continue
        _kept.append(_value)
    _sys.path[:] = [str(_root), *_kept]
    for _name in ('scripts', 'examples'):
        _directory = _root/_name
        if not _directory.is_dir() or (_directory/'__init__.py').exists():
            raise ValueError('Expected fixed project namespace: '+_name)
        _spec = _ModuleSpec(_name, loader=None, is_package=True)
        _spec.submodule_search_locations = [str(_directory)]
        _sys.modules[_name] = _module_from_spec(_spec)
    # This per-process marker is checked by the fresh top-level bootstrap and by
    # real CPU child sentinels; it is never a model or CUDA acceptance marker.
    _sys._immune_release_isolation = dict(
        root=str(_root), route=_route, guard_sha256=_expected, pid=_os.getpid())
except BaseException as _error:
    try:
        _os.write(2, ('Release startup rejected: '+str(_error)+'\n').encode())
    finally:
        _os._exit(78)
