"""Check the released dataset through the actual fixed-chain data reader.

Activate pllm/protenix_abtcr, then from repository root run:
  USE_TF=0 python -B scripts/data/check_v7_no_negative_design_reader.py

No weights or GPU are loaded. This verifies real saved records, role positions,
synthetic-mask loss exclusion and per-chain EOS in the current reader.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.export_v7_no_negative_design import OUTPUT, LAYOUT
from scripts.data.rebuild_v7_from_v5 import sha, write
from dllm.pipelines.immune_llada.data.dataset import PreparedImmuneDataset
from dllm.pipelines.immune_llada.data.records import CHAIN_ROLE_TO_ID
from dllm.pipelines.immune_llada.fixed_chain import FixedChainCollator, FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.layout import capacity, ROLES


def main(output=OUTPUT):
    global OUTPUT
    OUTPUT = Path(output)
    manifest = json.loads((OUTPUT / 'dataset_manifest.json').read_text())
    verification = json.loads((OUTPUT / 'verification.json').read_text())
    assert verification['status'] == 'passed'
    assert verification['dataset_manifest_sha256'] == sha(OUTPUT / 'dataset_manifest.json')
    assert manifest['reader_contract']['layout'] == LAYOUT
    assert manifest['reader_contract']['a4_allowed'] is False
    config_path = OUTPUT / manifest['reader_contract'].get('config_path', 'immune_config.json')
    if 'config_path' not in manifest['reader_contract']:
        config_path = ROOT / 'configs/models/v5_fixed_chain_f011_75m_20261012/immune_config.json'
    config = FixedChainConfig(**json.loads(config_path.read_text()))
    assert config.layout_contract == LAYOUT
    assert config.first_chain_capacity == manifest['reader_contract']['first_chain_capacity']
    assert config.second_chain_capacity == manifest['reader_contract']['second_chain_capacity']
    if 'config_sha256' in manifest['reader_contract']:
        assert sha(config_path) == manifest['reader_contract']['config_sha256']
    collator = FixedChainCollator(config)
    tokenizer = collator.tokenizer
    checks, sizes = [], {}
    for split, spec in manifest['splits'].items():
        dataset = PreparedImmuneDataset(OUTPUT, split=split)
        assert len(dataset) == spec['records']
        sizes[split] = len(dataset)
        indices, offset = set(), 0
        for shard in spec['shards']:
            assert sha(OUTPUT / shard['path']) == shard['sha256']
            indices.update((offset, offset + shard['records'] - 1))
            offset += shard['records']
        for i in sorted(indices):
            record = dataset[i]
            relation = record.labels.get('relation', 'absent')
            assert relation in {'binding', 'unknown', 'absent'}, relation
            batch = collator([record])
            details = []
            chain_ids = []
            for chain in record.chains:
                if chain.role not in ROLES:
                    continue
                positions = (batch['role_ids'][0] == CHAIN_ROLE_TO_ID[chain.role]).nonzero().flatten()
                cap = capacity(config, chain.role)
                assert len(positions) == cap
                ids = batch['chain_ids'][0, positions].unique().tolist()
                assert len(ids) == 1
                chain_ids += ids
                assert batch['position_ids'][0, positions].tolist() == list(range(1, cap + 1))
                synthetic = chain.metadata.get('synthetic_residue_mask', [0] * len(chain.sequence))
                tokens = batch['input_ids'][0, positions].tolist()
                labels = batch['labels'][0, positions].tolist()
                observed = any(not x for x in synthetic)
                for j, flag in enumerate(synthetic):
                    if flag:
                        assert tokens[j] == tokenizer.mask_token_id and labels[j] == -100
                end = len(chain.sequence)
                if observed:
                    assert tokens[end] == labels[end] == tokenizer.eos_token_id
                else:
                    assert all(t == tokenizer.mask_token_id for t in tokens)
                    assert all(t == -100 for t in labels)
                assert all(t == tokenizer.mask_token_id for t in tokens[end + int(observed):])
                assert all(t == -100 for t in labels[end + int(observed):])
                details.append(dict(role=chain.role, stored_length=end, synthetic=sum(synthetic),
                                    slots=cap, has_observed_residues=observed,
                                    eos_position=end + 1 if observed else None,
                                    eos_after_synthetic_tail=bool(synthetic and synthetic[-1]) if observed else False))
            assert len(chain_ids) == len(set(chain_ids)), 'Receptor chains must have distinct IDs'
            checks.append(dict(split=split, dataset_index=i, source_group=record.source_group, relation=relation, chains=details))
        print(split, len(dataset), 'real-reader samples', len(indices), flush=True)
    write(OUTPUT / 'reader_acceptance.json', dict(status='passed', required_layout=LAYOUT, splits=sizes,
          actual_reader='PreparedImmuneDataset + FixedChainCollator', samples=checks,
          dataset_manifest_sha256=sha(OUTPUT / 'dataset_manifest.json'), code_sha256=sha(Path(__file__)),
          collator_sha256=sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/layout.py'),
          config_sha256=sha(config_path)))
    print(json.dumps(dict(status='passed', samples=len(checks), splits=sizes)), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=OUTPUT)
    main(parser.parse_args().data_dir)
