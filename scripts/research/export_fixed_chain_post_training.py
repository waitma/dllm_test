"""Verify and export completed pilot checkpoints for independent evaluation.

Activate pllm/protenix_abtcr; python -m scripts.research.export_fixed_chain_post_training --run /absolute/screen_r3
Only successful trials are exported. Existing exports are verified, never overwritten.
"""
import argparse
import json
from pathlib import Path
import sys
import torch
from dllm.pipelines.immune_llada.fixed_chain import FixedChainCollator
from dllm.pipelines.immune_llada.fixed_chain.post_training import PostTrainingModel, RECIPES
from scripts.research.run_fixed_chain_post_training import load_model, sha, write


def export_trial(trial, rows, plan_path=None):
    status = json.loads((trial / 'status.json').read_text())
    if status['status'] != 'success':
        return {'trial': str(trial), 'status': 'not_complete'}
    checkpoint = trial / 'checkpoint.pt'
    digest = sha(checkpoint)
    if digest != status['checkpoint_sha256']:
        raise ValueError('Saved checkpoint changed: ' + str(trial))
    out = trial / 'export'
    if out.exists():
        receipt = json.loads((out / 'export_verification.json').read_text())
        if receipt['checkpoint_sha256'] != digest or any(sha(out / k) != v for k, v in receipt['files'].items()):
            raise ValueError('Export changed: ' + str(trial))
        if plan_path is not None:
            if not (out / 'training_binding.json').exists():
                raise ValueError('Existing export predates source binding; do not silently rewrite it')
            if sha(plan_path) != receipt['training_plan_sha256']:
                raise ValueError('Existing export belongs to another training plan')
        return receipt
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    recipe = state['recipe']
    if state['options'] != RECIPES[recipe] or state['step'] != status['step']:
        raise ValueError('Recipe/step differs from completion receipt')
    model = load_model(Path(state['parent']), recipe).eval()
    if plan_path is not None:
        source = Path(plan_path).parent / 'source'
        for name, module in tuple(sys.modules.items()):
            filename = getattr(module, '__file__', None)
            if filename and name.split('.')[0] in ('dllm', 'examples', 'scripts'):
                if not Path(filename).resolve().is_relative_to(source):
                    raise ValueError('Source-bound export must import the frozen training source first: ' + name)
    model.load_state_dict(state['state_dict'], strict=True)
    parent, plan_hash = state['parent'], state['plan_sha256']
    del state
    batch = FixedChainCollator(model.config)(rows)
    features = model.feature_args(batch)
    features['input_ids'] = batch['input_ids'].masked_fill(batch['eligible_mask'], model.tokenizer.mask_token_id)
    with torch.no_grad():
        before = model.denoise(**features)[0]
    temporary = trial / 'export.partial'
    temporary.mkdir(exist_ok=False)
    model.save_pretrained(temporary)
    if plan_path is not None:
        plan_path = Path(plan_path)
        if sha(plan_path) != plan_hash:
            raise ValueError('Export plan differs from trained checkpoint')
        plan = json.loads(plan_path.read_text())
        if any(sha(file) != expected for file, expected in plan['source_hashes'].items()):
            raise ValueError('Frozen training source changed')
        # Loader uses the exact training implementation, including adaptations.
        write(temporary / 'training_binding.json', {
            'schema': 'fixed_chain.post_training.eval_binding.v1',
            'source_root': str(plan_path.parent / 'source'), 'source_files': plan['source_hashes'],
            'training_plan': str(plan_path), 'training_plan_sha256': plan_hash,
            'recipe': recipe, 'seed': plan['seed'], 'step': status['step'], 'parent': parent,
            'pilot_checkpoint_sha256': digest})
    restored = PostTrainingModel.from_pretrained(temporary).eval()
    with torch.no_grad():
        after = restored.denoise(**features)[0]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        if not torch.equal(value, restored.state_dict()[name]):
            raise ValueError('Exported parameter differs: ' + name)
    receipt = {'status': 'passed', 'trial': str(trial), 'recipe': recipe, 'step': status['step'],
               'checkpoint_sha256': digest, 'parent': parent, 'training_plan_sha256': plan_hash,
               'roundtrip_logits_exact': True, 'all_state_tensors_exact': True,
               'feature_shape': list(batch['input_ids'].shape),
               'files': {p.name: sha(p) for p in temporary.iterdir() if p.is_file()}}
    write(temporary / 'export_verification.json', receipt)
    temporary.rename(out)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--plan', type=Path, help='Bind exports to their immutable training source')
    parser.add_argument('--screen-data', type=Path, help='Screen rows for nested multi-seed run directories')
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows, sources = [], set()
    for line in (args.screen_data or args.run.parent / 'data/screen.jsonl').open():
        row = json.loads(line)
        if row['source'] not in sources:
            rows.append(row)
            sources.add(row['source'])
        if len(rows) == 3:
            break
    reports = []
    for status in sorted(args.run.glob('*/status.json')):
        result = export_trial(status.parent, rows, args.plan)
        reports.append(result)
        print(status.parent.name, result['status'], flush=True)
    write(args.run / 'exports.json', reports)


if __name__ == '__main__':
    main()
