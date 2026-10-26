"""Activate pllm; python -B SCRIPT --checkpoint ABS --baseline-run ABS --run NEW_ABS.

Prepare a fresh AirGen-aligned reference-pairing plan without running a model.
baseline-run must contain the same checkpoint's native-EOS pairing outputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from downstream.fixed_chain_pairing import protocol


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--baseline-run', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    checkpoint, baseline, run = (x.resolve() for x in (args.checkpoint, args.baseline_run, args.run))
    side = json.loads((checkpoint/'formal_checkpoint.json').read_text())
    binding = json.loads((checkpoint/'run_provenance.json').read_text())
    if binding['variant'] != 'fixed_chain_full_global' or sha(checkpoint/'model.safetensors') != side['files']['model.safetensors']:
        raise ValueError('Expected an unchanged fixed-chain checkpoint')
    baseline_metrics = json.loads((baseline/'metrics.json').read_text())
    identity = baseline_metrics['native_identity']
    if identity['model_sha256'] != side['files']['model.safetensors'] or identity['step'] != side['step']:
        raise ValueError('Native-EOS baseline must belong to the requested checkpoint')
    if not (baseline/'predictions.jsonl').is_file():
        raise ValueError('Native-EOS predictions required for historical GPU check')
    benchmark = ROOT/'data/oas_previous_clean/splits/holdout_select_500_eval_OAS.csv'
    sources = [Path(__file__).resolve(), ROOT/'scripts/downstream/run_fixed_chain_pairing_reference.py',
               ROOT/'downstream/fixed_chain_pairing.py', ROOT/'downstream/fixed_chain_eval.py',
               ROOT/'downstream/grammar/light_chain_pairing.py']
    sources += sorted((ROOT/'downstream/comp_chain/eval_scripts').rglob('*.py'))
    source_root = Path(binding['source_root'])
    for relative in ['dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py',
                     'dllm/pipelines/bioseq/ophiuchus/sampling.py']:
        path = source_root/relative
        if binding['source_files'].get(str(path)) != sha(path):
            raise ValueError('Sampler source is not checkpoint-bound: '+str(path))
        sources.append(path)
    plan = dict(checkpoint=str(checkpoint), checkpoint_step=side['step'],
                model_sha256=side['files']['model.safetensors'], protocol=protocol(),
                benchmark=dict(path=str(benchmark), sha256=sha(benchmark)),
                source_sha256={str(p): sha(p) for p in sources},
                baseline_metrics=str(baseline/'metrics.json'),
                baseline_predictions=str(baseline/'predictions.jsonl'),
                created_utc=datetime.now(timezone.utc).isoformat())
    run.mkdir(parents=True, exist_ok=False)
    (run/'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False)+'\n')
    print(str(run/'plan.json'))


if __name__ == '__main__':
    main()
