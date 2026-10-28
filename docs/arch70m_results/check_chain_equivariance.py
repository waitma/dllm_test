"""Check protein-block permutation equivariance of frozen screen models on CPU.

Activate pllm, set CUDA_VISIBLE_DEVICES='' and PYTHONPATH to the exploration
root, then run its output/arch70m_env/bin/python with this absolute script path
and --output-dir /absolute/report. No training data or checkpoint is modified.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import load_file
import torch

from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig, FixedChainModel
from experiments.arch70m.common import digest, sha, source_binding, write
from experiments.arch70m.evaluate import make_tasks, rows
from experiments.arch70m.models import build

ROOT = Path('/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore')
RUN = ROOT / 'output/arch70m_v8_20261028'
SOURCE = '053297983e0f2ece2aaad807032695fb5aebd1e0f221138a9bf0d202cc9b9ac4'


def errors(actual, expected):
    same_finite = torch.equal(torch.isfinite(actual), torch.isfinite(expected))
    assert same_finite
    finite = torch.isfinite(expected)
    assert not torch.isnan(actual).any() and not torch.isnan(expected).any()
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    difference = (actual[finite] - expected[finite]).abs()
    # FP32 CPU engineering tolerance; unrelated to training replay thresholds.
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=1e-5)
    return dict(max_absolute=float(difference.max()), mean_absolute=float(difference.mean()),
                finite_mask_equal=same_finite)


def permute_blocks(features):
    chain_ids = features['chain_ids'][0]
    # Reverse contiguous blocks, preserving every block's internal residue order.
    boundaries = [0] + (chain_ids[1:] != chain_ids[:-1]).nonzero().flatten().add(1).tolist() + [len(chain_ids)]
    blocks = [torch.arange(a, b) for a, b in zip(boundaries[:-1], boundaries[1:])]
    permutation = torch.cat(blocks[::-1])
    assert torch.equal(permutation.sort().values, torch.arange(len(chain_ids)))
    assert not torch.equal(permutation, torch.arange(len(chain_ids)))
    return {k: v[:, permutation].clone() for k, v in features.items()}, permutation


def rename_chains(features):
    result = {k: v.clone() for k, v in features.items()}
    original = features['chain_ids']
    ids = sorted(set(original[original >= 0].tolist()))
    mapping = {old: 100 + i for i, old in enumerate(reversed(ids))}
    for old, new in mapping.items():
        result['chain_ids'][original == old] = new
    return result, mapping


@torch.inference_mode()
def check(output):
    assert not torch.cuda.is_available(), 'Run this diagnostic with CUDA_VISIBLE_DEVICES empty'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert digest(source_binding()) == SOURCE
    settings = json.loads((ROOT / 'experiments/arch70m/base.json').read_text())
    index = json.loads((RUN / 'data/index.json').read_text())
    ref = index['files']['screen']['task_valid']
    config = SimpleNamespace(config=FixedChainConfig(**settings['model']), family_spec=settings['family_prediction'])
    tasks, _ = make_tasks(config, rows(ref))
    selected = []
    # Shortest eligible task in each source, independent of model predictions.
    for source in ('asd_antibody', 'oas', 'tcr_native'):
        candidates = [t for t in tasks if t['kind'] == 'conditional' and t['source'] == source
                      and len(set(t['clean']['chain_ids'][t['clean']['chain_ids'] >= 0].tolist())) >= 2]
        assert candidates, source
        selected.append(min(candidates, key=lambda t: (t['truth'].numel(), t['key'], t['role'])))
    output.mkdir(parents=True, exist_ok=True)
    evidence, observations, sample_receipts = {}, [], []
    for variant in ('A0', 'A4', 'A5'):
        trial = RUN / 'trials' / f'screen_{variant}_s42'
        status = json.loads((trial / 'status.json').read_text())
        assert status['status'] == 'completed' and status['step'] == 1276
        checkpoint = trial / 'checkpoints/step_001276'
        receipt = json.loads((checkpoint / 'receipt.json').read_text())
        assert receipt['status'] == 'complete'
        for name in ('model.safetensors', 'binding.json', 'state.json'):
            path = checkpoint / name
            evidence[str(path)] = sha(path)
            assert evidence[str(path)] == receipt['files'][name]
        binding = json.loads((checkpoint / 'binding.json').read_text())
        assert binding['source_sha256'] == SOURCE and binding['variant'] == variant
        model = build(settings, variant, seed=42).eval()
        model.load_state_dict(load_file(str(checkpoint / 'model.safetensors'), device='cpu'), strict=True)
        for task in selected:
            features = {k: v.clone() for k, v in FixedChainModel.feature_args(task['clean']).items()}
            features['input_ids'][0, task['target']] = model.tokenizer.mask_token_id
            base_logits, base_hidden = model.denoise(**features, output_hidden_states=True)
            permuted, permutation = permute_blocks(features)
            renamed, mapping = rename_chains(features)
            if variant == 'A0':
                sample_receipts.append(dict(source=task['source'], receptor_key=task['key'], role=task['role'],
                    length=features['input_ids'].shape[1], input_sha256=digest({k: v.tolist() for k, v in features.items()}),
                    block_permutation=permutation.tolist(), chain_renaming=mapping))
            for name, changed, order in [('block_order', permuted, permutation),
                                         ('chain_id_bijection', renamed, torch.arange(permutation.numel()))]:
                inverse = order.argsort()
                original_biases = model._build_attention_biases(features['chain_ids'], features['position_ids'], features['attention_mask'])
                changed_biases = model._build_attention_biases(changed['chain_ids'], changed['position_ids'], changed['attention_mask'])
                assert set(original_biases) == set(changed_biases)
                for key in original_biases:
                    assert torch.equal(original_biases[key], changed_biases[key][:, :, inverse][:, :, :, inverse])
                logits, hidden = model.denoise(**changed, output_hidden_states=True)
                row = dict(variant=variant, source=task['source'], transformation=name,
                    logits=errors(logits[:, inverse], base_logits), hidden=errors(hidden[:, inverse], base_hidden),
                    attention_bias_equal=True)
                observations.append(row)
                print(json.dumps(row), flush=True)
        del model
    assert digest(source_binding()) == SOURCE
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='passed', device='cpu', dtype='float32',
        source_sha256=SOURCE, task_file=ref, checkpoint_file_sha256=evidence, examples=sample_receipts,
        observations=observations, atol=5e-5, rtol=1e-5,
        scope='Single denoise forward with trained screening weights; all seven features move together, including role and within-chain position.',
        limitations=['Three representative screening inputs per architecture, not a proof for all inputs or GPU kernels.',
            'Permutation is applied to in-memory model features only; the original V8 records, tokenizer and collator remain unchanged.',
            'This does not exchange biological role identities or reverse residues inside a protein.',
            'No decoder, positional top-k tie-breaking, data augmentation or downstream performance claim is tested.',
            'No confirmation evaluation or candidate reselection was performed.'])
    write(output / 'chain_equivariance.json', report)
    flat = [dict(variant=r['variant'], source=r['source'], transformation=r['transformation'],
                 logit_max_absolute=r['logits']['max_absolute'], hidden_max_absolute=r['hidden']['max_absolute'],
                 attention_bias_equal=r['attention_bias_equal']) for r in observations]
    with (output / 'checks.csv').open('w') as handle:
        writer = csv.DictWriter(handle, list(flat[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(flat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    assert args.output_dir.is_absolute()
    check(args.output_dir)
