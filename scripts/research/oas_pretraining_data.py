"""Select OAS train/validation shards for scratch pretraining.

Activate pllm/protenix_abtcr; run:
python -B -m pytest scripts/tests/immune_llada/test_oas_pretraining_data.py -q
"""
import hashlib
import json
from pathlib import Path
import random


def describe_selection(data_dir):
    """Bind manifest source groups, not the provenance string inside each row."""
    path = Path(data_dir) / 'dataset_manifest.json'
    content = path.read_bytes()
    manifest = json.loads(content)
    splits = {}
    for split in ('train', 'valid'):
        shards = [item for item in manifest['splits'][split]['shards']
                  if item['source'] == 'oas']
        if not shards or any(item['records'] <= 0 for item in shards):
            raise ValueError(f'No nonempty OAS shards for {split}')
        splits[split] = {'records': sum(item['records'] for item in shards),
                         'shards': shards}
    return {'schema': 'immune.oas_pretraining_selection.v1',
            'data_dir': str(Path(data_dir).resolve()),
            'manifest_sha256': hashlib.sha256(content).hexdigest(),
            'train_sources': ['oas'], 'validation_sources': ['oas'],
            'splits': splits}


def load_datasets(data_dir, valid_per_source, expected_selection):
    """Index only OAS files; retain the baseline's seeded validation sampling."""
    from torch.utils.data import Subset
    from dllm.pipelines.immune_llada.data.dataset import PreparedImmuneDataset

    if valid_per_source <= 0:
        raise ValueError('Validation sample budget must be positive')
    selection = describe_selection(data_dir)
    if selection != expected_selection:
        raise ValueError('OAS data selection differs from the launch binding')
    train = PreparedImmuneDataset(data_dir, split='train', source='oas')
    valid = PreparedImmuneDataset(data_dir, split='valid', source='oas')
    for split, dataset in (('train', train), ('valid', valid)):
        if len(dataset) != selection['splits'][split]['records']:
            raise ValueError(f'OAS {split} count differs from the manifest')
    indices = random.Random(42).sample(range(len(valid)), min(len(valid), valid_per_source))
    return train, {'oas': Subset(valid, indices)}, {'oas': indices}
