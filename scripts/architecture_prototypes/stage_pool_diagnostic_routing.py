"""Generate a reviewable patch, never apply it to the active repository.

Activate pllm, then:
python scripts/architecture_prototypes/stage_pool_diagnostic_routing.py --output-dir /absolute/new/stage
CPU tests apply the generated patch only inside a temporary repository copy.
Actual deployment requires all running jobs terminal and the R20 archive verified.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_SHA256 = {
    'scripts/run_immune_research.py': '711e7201250be7cc554022a18a555a95c0e10655440b5563499282c142afb6fa',
    'dllm/pipelines/immune_llada/research/model.py': '2cad8a5ffb787cb108f7ab22702e7476011b42a1c837f0dc21b0803ed6c66092',
    'scripts/immune_research_pool.py': 'f1997a0e4d15a24637fb7030500ee92ff2641107dd595872429d9bfd0875e5b2',
}
DISPATCHER = 'scripts/architecture_prototypes/pool_release_dispatcher.py'
STARTUP_GUARD = 'scripts/architecture_prototypes/release_startup/sitecustomize.py'
DIAGNOSTIC = 'd00_cuda_attention_probe_v1'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Patch anchor is missing or ambiguous')
    return text.replace(old, new, 1)


def proposed_sources(root=ROOT):
    original = {name: (root/name).read_text() for name in BASE_SHA256}
    if any(sha(text.encode()) != BASE_SHA256[name] for name, text in original.items()):
        raise ValueError('Active source SHA differs; regenerate only after explicit review')
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
    old_tail = original[name][original[name].index("\nif __name__ == '__main__':"):]
    proposed[name] = replace_once(proposed[name], old_tail, '\n# CLI dispatch occurs above, before Torch/model imports.\n')
    name = 'dllm/pipelines/immune_llada/research/model.py'
    proposed[name] = replace_once(proposed[name],
        "    'r27_refined_value_residual': {'architecture_refined': True, 'value_residual': True},\n}",
        "    'r27_refined_value_residual': {'architecture_refined': True, 'value_residual': True},\n"
        "    'd00_cuda_attention_probe_v1': {'diagnostic_only': True},\n}")
    proposed[name] = replace_once(proposed[name],
        "    def __new__(cls, config=None, variant='r00_a4_control'):\n",
        "    def __new__(cls, config=None, variant='r00_a4_control'):\n"
        "        if VARIANTS.get(variant, {}).get('diagnostic_only'):\n"
        "            raise ValueError('Diagnostic variants cannot construct a model')\n")
    name = 'scripts/immune_research_pool.py'
    proposed[name] = replace_once(proposed[name],
        "            return ast.literal_eval(node.value)\n",
        "            if not isinstance(node.value, ast.Dict):\n"
        "                raise ValueError('Research registry must be a literal dictionary')\n"
        "            keys = [ast.literal_eval(key) for key in node.value.keys]\n"
        "            if any(type(key) is not str for key in keys) or len(keys) != len(set(keys)):\n"
        "                raise ValueError('Duplicate or invalid variant registry key')\n"
        "            return ast.literal_eval(node.value)\n")
    proposed[name] = replace_once(proposed[name],
        "    limit = 5000 if job['mode'] == 'train' else 100\n",
        "    if variants()[job['variant']].get('diagnostic_only'):\n"
        "        if (job != dict(variant='d00_cuda_attention_probe_v1', mode='profile-synthetic', steps=1)\n"
        "                or type(job.get('steps')) is not int):\n"
        "            raise ValueError('Diagnostic requires the exact d00 profile-synthetic/1 envelope')\n"
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
    output.mkdir(parents=True)
    patch = ''.join(''.join(difflib.unified_diff(original[name].splitlines(keepends=True),
        text.splitlines(keepends=True), fromfile='a/'+name, tofile='b/'+name)) for name, text in proposed.items())
    (output/'routing.patch').write_text(patch)
    for name, text in proposed.items():
        path = output/'proposed'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    manifest = dict(schema_version='immune.staged_diagnostic_routing.v1', deployed=False,
        source_root=str(ROOT), before_sha256=BASE_SHA256,
        after_sha256={name: sha(text.encode()) for name, text in proposed.items()},
        patch_sha256=sha(patch.encode()), required_dispatcher={DISPATCHER: sha((ROOT/DISPATCHER).read_bytes())},
        required_startup_guard={STARTUP_GUARD: sha((ROOT/STARTUP_GUARD).read_bytes())},
        prerequisites=['All eight 500-step jobs are terminal with verified original evidence.',
            'R20 snapshot r20_r27_20261001 is archived and independently verified.',
            'Existing release_map R20 entries are rebound to the frozen snapshot before changing sources.',
            'Typed monitor/supervisor support is deployed separately.',
            'CPU temporary-repository patch/route and inherited-startup-guard tests passed; this patch is not yet applied.'],
        fixed_model_variant_count=28, diagnostic_variant_count=1,
        limits=['Current in-memory worker keeps old validate; dispatcher and typed adapter still reject invalid diagnostic arguments.',
            'Old R00 manifest binds only its 32 measured files, not every copied supporting file.',
            'Startup isolation covers audited ordinary Python children inheriting the job environment, not -I/-S or arbitrary code.'])
    (output/'deployment_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    stage(args.output_dir)


if __name__ == '__main__':
    main()
