"""Evaluate frozen candidates once on reserved confirmation records.

Activate pllm/protenix_abtcr; python -m scripts.research.run_fixed_chain_confirmation
    --plan /absolute/frozen_plan.json --recipe task_mix_seed42 --device 0
No optimization, checkpoint choice or threshold tuning occurs here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
import torch
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainModel
from dllm.pipelines.immune_llada.fixed_chain.post_training import PostTrainingModel
from scripts.research.run_fixed_chain_post_training import (
    sha, write, assign_source_groups, evaluate, generation_probe,
)


def validate_plan(path, name):
    """Validate the model and protocol binding before opening confirmation data."""
    plan = json.loads(path.read_text())
    if plan.get('schema') != 'fixed_chain.confirmation.v1' or plan.get('status') != 'frozen':
        raise ValueError('Confirmation plan must be frozen')
    if name not in plan['models'] or set(plan['models']) != set(plan['recipes']):
        raise ValueError('Candidate binding differs')
    for file, expected in plan['source_hashes'].items():
        if sha(file) != expected:
            raise ValueError('Frozen evaluation source changed: ' + file)
    if sha(plan['protocol']) != plan['protocol_sha256']:
        raise ValueError('Selection protocol changed')
    spec = plan['models'][name]
    checkpoint = Path(spec['checkpoint'])
    for filename, expected in spec['files'].items():
        if sha(checkpoint / filename) != expected:
            raise ValueError('Candidate checkpoint changed: ' + filename)
    if sha(plan['data_manifest']) != plan['data_manifest_sha256']:
        raise ValueError('Confirmation manifest changed')
    return plan, spec


def run(path, name, device_index):
    plan, spec = validate_plan(path, name)
    out = Path(plan['output']) / name
    out.mkdir(exist_ok=False)
    write(out / 'status.json', {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat()})
    try:
        # This receipt records the first confirmation-data access for this model.
        write(out / 'data_access.json', {'at_utc': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': sha(path), 'protocol_sha256': plan['protocol_sha256'],
              'model_sha256': spec['files']['model.safetensors'], 'split': 'confirm'})
        manifest = json.loads(Path(plan['data_manifest']).read_text())
        ref = manifest['files']['confirm']
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('Confirmation rows changed')
        rows = assign_source_groups([json.loads(line) for line in Path(ref['path']).read_text().splitlines()],
                                    manifest['source_counts'], 'confirm')
        torch.set_num_threads(2)
        torch.cuda.set_device(device_index)
        device = torch.device('cuda', device_index)
        cls = FixedChainModel if spec.get('parent') else PostTrainingModel
        model = cls.from_pretrained(spec['checkpoint']).to(device).eval().requires_grad_(False)
        write(out / 'metrics.json', evaluate(model, rows, device, semantic=True, collect_records=True))
        write(out / 'generation.json', generation_probe(model, rows, device,
              per_source_limit=128, steps=16))
        write(out / 'status.json', {'status': 'success', 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'model_sha256': spec['files']['model.safetensors'], 'protocol_sha256': plan['protocol_sha256'],
              'optimizer_updates': 0, 'rows': len(rows)})
    except Exception:
        write(out / 'status.json', {'status': 'failed', 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    run(args.plan, args.recipe, args.device)
