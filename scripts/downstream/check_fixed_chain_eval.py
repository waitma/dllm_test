"""Activate pllm/protenix_abtcr; python -B SCRIPT --run ABS --device cpu|cuda.

Check actual checkpoint features, label blindness and generation adapters.
GPU checks must run through the associated eval_jobs YAML.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    from downstream.fixed_chain_eval import load, FixedChainBackbone, generate
    model, identity = load(args.run/'checkpoint', args.device)
    import torch
    import numpy as np
    torch.set_num_threads(4)
    from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord, CHAIN_ROLE_TO_ID
    from downstream.benchmark.tcr_binding.query_protocol import TCRBindingQueryProtocol, TCRBindingEmbedder
    from downstream.grammar.tcr_features import TCRRepresentationProtocol
    backbone = FixedChainBackbone(model, identity)
    checks = {}
    for mode in ('cdr3b', 'cdr3ab', 'longab'):
        protocol = TCRBindingQueryProtocol(input_mode=mode)
        record = protocol.record('GILGFVFTL', 'CASSIRSSYEQYF', None if mode == 'cdr3b' else 'CAVRDSNYQLIW')
        batch = protocol.collate([record], backbone.collator, decoder_mask_token_id=model.tokenizer.mask_token_id)
        relation = batch['relation_token_mask']
        assert relation.sum() == 1 and batch['input_ids'][relation].eq(model.tokenizer.mask_token_id).all()
        assert 'labels' not in batch and not record.labels
        batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        hidden, _ = backbone._final_hidden(batch)
        assert torch.isfinite(hidden).all()
        vector = TCRBindingEmbedder(backbone, protocol).embed_pairs(
            ['CASSIRSSYEQYF'], None if mode == 'cdr3b' else ['CAVRDSNYQLIW'], ['GILGFVFTL'])
        assert vector.shape == (1, 768) and np.isfinite(vector).all()
        checks['t1_'+mode] = dict(masked_relation=True, finite_features=True, feature_dim=768)
    protocol = TCRRepresentationProtocol()
    record = protocol.record('CASSIRSSYEQYF')
    batch = protocol.collate([record], backbone.collator)
    assert not batch['relation_target_mask'].any()
    assert np.isfinite(backbone.vectors([record])).all()
    checks['t2_t3'] = dict(null_context=True, no_relation_target=True)
    record = BioSeqRecord([BioSeqChain('ACDEFGHIK', 'antigen'),
                           BioSeqChain('CASSIRSSYEQYF', 'antibody_heavy'),
                           BioSeqChain('CAVRDSNYQLIW', 'antibody_light')], 'antibody_antigen', 'adapter_check')
    batch = backbone.collator([record])
    receptor = batch['residue_mask'] & batch['target_slot_mask']
    assert int(receptor.sum()) == len(record.chains[1].sequence)+len(record.chains[2].sequence)
    assert int((batch['residue_mask'] & ~batch['target_slot_mask']).sum()) == 9
    assert np.isfinite(backbone.vectors([record])).all()
    if args.device == 'cuda':
        from downstream.v5_native_generation_eval import score_pair
        score, features = score_pair(backbone, *[c.sequence for c in record.chains])
        assert np.isfinite(score) and len(features) == 3*768 and np.isfinite(features).all()
    checks['ab_ag'] = dict(exact_receptor_positions=True, pooling_excludes_tail=True,
                           pll_forward_tested=args.device == 'cuda')
    for name, conditions, targets in [
        ('pairing', [BioSeqChain('CASSIRSSYEQYF', 'antibody_heavy')], ['antibody_light']),
        ('t4', [BioSeqChain('GILGFVFTL', 'peptide')], ['tcr_beta'])]:
        result = generate(model, conditions, targets, 42, steps=2, greedy=True, single_chain=name == 't4')
        assert result['protocol'] == 'fixed_chain_native_eos_187_154'
        assert len(result['chains']) == (2 if name == 'pairing' else 1)
        if name == 'pairing':
            assert result['chains'][0] == conditions[0].sequence
        # Random short decoding may fail quality; the contract preserves every failure.
        checks[name] = dict(api='passed', reference_target_supplied=False, smoke_output_status=result['status'])
    result = dict(status='passed', device=args.device, model=identity, checks=checks)
    (args.run/f'{args.device}_adapter_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
