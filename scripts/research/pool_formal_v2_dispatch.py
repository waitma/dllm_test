"""Cold formal entry preserving the frozen gates and the capacity-v2 chain.

Activate conda pllm. The future pool entry calls main(original_argv); use
prepare(original_argv) for a read-only formal-ticket check. No catalog, model
choice, queue, process group or training is created by this integration.
Repository: /vepfs-mlp2/c20250601/251105016/project/dllm_test.
"""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test')
BASE = ROOT/'output/v7_formal_eight_gpu_20261002/pool_launcher_source_v1/pool_formal_dispatch.py'
BASE_SHA = '21f2ffe71d23950207eb4c067b0cb26cf5dbb875a13e2ba15bdec3a7d3f5ce59'
PREVIOUS = ROOT/'output/v7_large_batch_ablation_20261002/launcher_source_v2/scripts/research/pool_large_batch_ablation_v2_dispatch.py'
PREVIOUS_SHA = 'b088ba3c19f58060a05393411cdf57a3abe8ecd7ba91508add62142896476cc3'
MANIFEST = ROOT/'output/v7_large_batch_ablation_20261002/launcher_source_v2.json'
MANIFEST_SHA = '4cd5f793e74f56730a0ca4468cebb2783fe2de22b33c7325f664f54634db43aa'


def bound(path, expected):
    if not path.is_absolute() or path != path.resolve() or path.is_symlink() or not path.is_file():
        raise ValueError('Expected a canonical regular source/manifest file: '+str(path))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Frozen entry source/manifest SHA differs: '+str(path))
    return raw


manifest = json.loads(bound(MANIFEST, MANIFEST_SHA))
if (manifest['schema'] != 'v7.large_batch_launcher_source.v2'
        or Path(manifest['snapshot_root'])/'scripts/research/pool_large_batch_ablation_v2_dispatch.py' != PREVIOUS
        or manifest['source_sha256']['scripts/research/pool_large_batch_ablation_v2_dispatch.py'] != PREVIOUS_SHA):
    raise ValueError('Frozen large-batch-v2 manifest identity differs')
bound(PREVIOUS, PREVIOUS_SHA)
bound(BASE, BASE_SHA)
_base = runpy.run_path(str(BASE))
bound(BASE, BASE_SHA)
_runtime = _base['prepare'].__globals__
# The two substitutions affect only this private namespace; no files change.
_runtime.update(PREVIOUS=PREVIOUS, PREVIOUS_SHA=PREVIOUS_SHA)
prepare = _runtime['prepare']
main = _runtime['main']

if __name__ == '__main__':
    raise SystemExit(main())
