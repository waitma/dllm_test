"""Stage the fixed D01 pool route independently of the historical D00 patch.

Activate pllm, then:
python scripts/architecture_prototypes/stage_pool_trainer_runtime_routing.py --output-dir /absolute/fresh/stage_trainer_v1
Only review artifacts are written. The three measured sources remain unchanged;
CPU tests apply the diff inside a temporary repository. No runtime is created,
no data is read, and no job is submitted or enqueued by this helper.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

if __package__:
    from .stage_pool_diagnostic_routing import BASE_SHA256, replace_once
    from . import pool_release_dispatcher as dispatch
else:
    from stage_pool_diagnostic_routing import BASE_SHA256, replace_once
    import pool_release_dispatcher as dispatch

ROOT = Path(__file__).resolve().parents[2]
DISPATCHER = 'scripts/architecture_prototypes/pool_release_dispatcher.py'
STARTUP_GUARD = dispatch.STARTUP_GUARD
DIAGNOSTIC = dispatch.TRAINER_DIAGNOSTIC
TYPED_REPORTING = ('scripts/immune_diagnostic_reporting.py',
    'scripts/immune_research_supervision.py', 'scripts/monitor_immune_research_pool.py')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def proposed_sources(root=ROOT):
    original = {name: (root/name).read_text() for name in BASE_SHA256}
    if any(sha(text.encode()) != BASE_SHA256[name] for name, text in original.items()):
        raise ValueError('Active source SHA differs; review before regenerating the Trainer stage')
    proposed = dict(original)
    name = 'scripts/run_immune_research.py'
    early = '''if __name__ == '__main__':
    # Route before any ML import; the selected release starts in a fresh process.
    import runpy as _route_runpy
    from pathlib import Path as _RoutePath
    _route_dispatcher = _RoutePath(__file__).resolve().parents[1]/'scripts/architecture_prototypes/pool_release_dispatcher.py'
    raise SystemExit(_route_runpy.run_path(str(_route_dispatcher))['main']())

'''
    proposed[name] = replace_once(original[name], 'import argparse\n', early+'import argparse\n')
    tail = original[name][original[name].index("\nif __name__ == '__main__':"):]
    proposed[name] = replace_once(proposed[name], tail, '\n# CLI dispatch occurs above, before Torch/model imports.\n')
    name = 'dllm/pipelines/immune_llada/research/model.py'
    proposed[name] = replace_once(proposed[name],
        "    'r27_refined_value_residual': {'architecture_refined': True, 'value_residual': True},\n}",
        "    'r27_refined_value_residual': {'architecture_refined': True, 'value_residual': True},\n"
        "    'd01_pilot_runtime_gpu_r20_v1': {'diagnostic_only': True},\n}")
    proposed[name] = replace_once(proposed[name],
        "    def __new__(cls, config=None, variant='r00_a4_control'):\n",
        "    def __new__(cls, config=None, variant='r00_a4_control'):\n"
        "        if VARIANTS.get(variant, {}).get('diagnostic_only'):\n"
        "            raise ValueError('Diagnostic variants cannot construct a model')\n")
    name = 'scripts/immune_research_pool.py'
    proposed[name] = replace_once(proposed[name], "            return ast.literal_eval(node.value)\n",
        "            if not isinstance(node.value, ast.Dict):\n"
        "                raise ValueError('Research registry must be a literal dictionary')\n"
        "            keys = [ast.literal_eval(key) for key in node.value.keys]\n"
        "            if any(type(key) is not str for key in keys) or len(keys) != len(set(keys)):\n"
        "                raise ValueError('Duplicate or invalid variant registry key')\n"
        "            return ast.literal_eval(node.value)\n")
    proposed[name] = replace_once(proposed[name], "    limit = 5000 if job['mode'] == 'train' else 100\n",
        "    if variants()[job['variant']].get('diagnostic_only'):\n"
        "        if (job != dict(variant='d01_pilot_runtime_gpu_r20_v1', mode='profile-synthetic', steps=1)\n"
        "                or type(job.get('steps')) is not int):\n"
        "            raise ValueError('Diagnostic requires the exact d01 profile-synthetic/1 envelope')\n"
        "        return\n"
        "    limit = 5000 if job['mode'] == 'train' else 100\n")
    proposed[name] = replace_once(proposed[name],
        "if not options.get('architecture_v2') and not options.get('architecture_refined')]",
        "if not options.get('architecture_v2') and not options.get('architecture_refined')\n"
        "                            and not options.get('diagnostic_only')]")
    for name, text in proposed.items():
        ast.parse(text, filename=name)
    return original, proposed


def stage(output_dir):
    output = Path(output_dir)
    if not output.is_absolute() or output.exists() or output != output.resolve():
        raise ValueError('Use a fresh canonical absolute staging directory')
    original, proposed = proposed_sources()
    required = [DISPATCHER, STARTUP_GUARD, *sorted(dispatch.RUNTIME_ADDITIONS), *TYPED_REPORTING]
    hashes = {name: sha((ROOT/name).read_bytes()) for name in required}
    patch = ''.join(''.join(difflib.unified_diff(original[name].splitlines(keepends=True),
        text.splitlines(keepends=True), fromfile='a/'+name, tofile='b/'+name))
        for name, text in proposed.items())
    output.mkdir(parents=True)
    (output/'routing.patch').write_text(patch)
    for name, text in proposed.items():
        path = output/'proposed'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    manifest = dict(schema='immune.staged_trainer_runtime_routing.v1', deployed=False,
        source_root=str(ROOT), before_sha256=BASE_SHA256,
        after_sha256={name: sha(text.encode()) for name, text in proposed.items()},
        patch_sha256=sha(patch.encode()), required_dispatcher={DISPATCHER: hashes[DISPATCHER]},
        required_startup_guard={STARTUP_GUARD: hashes[STARTUP_GUARD]},
        required_runtime_additions={name: hashes[name] for name in sorted(dispatch.RUNTIME_ADDITIONS)},
        required_typed_reporting={name: hashes[name] for name in TYPED_REPORTING},
        runtime_root=str(ROOT/'output/immune_pilot_runtimes'/dispatch.RUNTIME_NAME),
        parent_manifest_sha256=dispatch.RUNTIME_PARENT_SHA256,
        fixed_model_variant_count=28, diagnostic_variant=DIAGNOSTIC, diagnostic_variant_count=1,
        fixed_job=dict(variant=DIAGNOSTIC, mode='profile-synthetic', steps=1),
        independent_of_historical_d00_stage=True,
        prerequisites=['All eight 500-step jobs are terminal with verified original evidence.',
            'The immutable R20 parent archive and the eight rebound release_map entries are verified.',
            'Entry and producer bytes are frozen; the fixed runtime package is verified before execution.',
            'CPU temporary-repository patch, strict envelope, runtime-source and inherited-child guard checks pass.',
            'The local monitor reloads typed D01 support before interpreting an approved D01 result.'],
        limits=['This helper never applies measured-source edits, creates the runtime, reads datasets, or enqueues work.',
            'The existing in-memory worker still uses its old validator; the dispatcher and runtime entry enforce the fixed envelope.',
            'The transport step 1 is one invocation; reported progress is the three observed artificial optimizer windows.',
            'No attention/operator GPU proof is a Trainer route prerequisite.',
            'Inherited isolation covers audited ordinary Python children, not -I/-S or arbitrary changed environments.'])
    (output/'deployment_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output-dir', type=Path, required=True)
    stage(parser.parse_args().output_dir)


if __name__ == '__main__':
    main()
