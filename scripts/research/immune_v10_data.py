"""Read the V10 data release without changing historical model/reader contracts.

Activate pllm. Verify an immutable release with:
  USE_TF=0 python -m scripts.research.immune_v10_data --index ABS_INDEX
Use V10DataCollator with IndexedPreparedDataset for this data version.
V10 here names a dataset, independently of the existing V10 model architecture.
"""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path

import numpy as np

from dllm.pipelines.immune_llada.data.records import BioSeqRecord
from dllm.pipelines.immune_llada.fixed_chain.v9_bert import V9Collator, corrupt
from dllm.pipelines.immune_llada.fixed_chain.v_family import family_targets as old_family_targets
from scripts.research.bert_mlm_data import IndexedPreparedDataset, sha

BINDING_KIND = 'immune.bert_mlm.nanobody_mix.v1'
READER = 'scripts.research.immune_v10_data.V10DataCollator'
REGIONS = ('FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'FR4')


def reader_view(item):
    """Keep VNAR identity in storage; reuse the existing single-domain tensor slot."""
    record = BioSeqRecord.from_prepared_dict(item) if isinstance(item, dict) else item
    if 'nanobody_vnar' not in record.chain_roles:
        return record
    if (record.chain_roles != ['nanobody_vnar']
            or record.metadata.get('receptor_type') != 'VNAR'
            or record.metadata.get('relation_supervised') is not False
            or record.labels):
        raise ValueError('VNAR repertoire requires one observed domain and no binding label')
    chain = record.chains[0]
    if (set(chain.regions) != set(REGIONS)
            or ''.join(chain.regions[r] for r in REGIONS) != chain.sequence):
        raise ValueError('VNAR regions must exactly reconstruct the observed domain')
    return replace(record, chains=[replace(chain, role='nanobody_vhh')])


def family_targets(record):
    return old_family_targets(reader_view(record))


class V10DataCollator(V9Collator):
    """Same tensors and parameters for legacy rows; explicit VNAR storage adapter."""

    def row(self, item):
        return super().row(reader_view(item))

    def __call__(self, records):
        return super().__call__([reader_view(record) for record in records])


def verify_release(index_path, *, full_hash=True):
    index_path = Path(index_path)
    index = json.loads(index_path.read_text())
    if index.get('binding_kind') != BINDING_KIND:
        raise ValueError('Not a V10 nanobody mix binding')
    manifest_path = Path(index['manifest'])
    manifest = json.loads(manifest_path.read_text())
    policy = json.loads(Path(index['mix_policy']).read_text())
    verification = json.loads(Path(index['verification']).read_text())
    binding = json.loads(Path(index['binding']).read_text())
    for path, expected in [(manifest_path, index['manifest_sha256']),
                           (index['binding'], index['binding_sha256']),
                           (index['mix_policy'], index['mix_policy_sha256']),
                           (index['verification'], index['verification_sha256'])]:
        if sha(path) != expected:
            raise ValueError('Changed release component: ' + str(path))
    if (binding['schema'] != BINDING_KIND or binding['records'] != index['records']
            or binding['manifest_sha256'] != index['manifest_sha256']
            or verification['status'] != 'passed'
            or verification['dataset_manifest_sha256'] != index['manifest_sha256']
            or manifest['reader_contract']['adapter'] != READER
            or manifest['reader_contract']['adapter_sha256'] != sha(__file__)):
        raise ValueError('Release/reader qualification differs')
    family = json.loads(Path(index['family_spec']).read_text())
    if (sha(index['family_spec']) != index['family_spec_sha256']
            or family['fit_split'] != 'train'
            or family['fit_manifest_sha256'] != index['manifest_sha256']):
        raise ValueError('Family vocabulary is not fitted to this training split')
    specs = {str(Path(index['data_dir']) / s['path']): s
             for block in manifest['splits'].values() for s in block['shards']}
    if set(specs) != {s['path'] for s in index['shards']}:
        raise ValueError('Incomplete manifest/index coverage')

    def check(spec):
        ref = specs[spec['path']]
        if any(spec[k] != ref[k] for k in ('records', 'sha256', 'source', 'split')):
            raise ValueError('Shard disagrees with manifest')
        path = Path(spec['path'])
        if str(path.resolve(strict=True)) != spec['resolved_path']:
            raise ValueError('Shard target changed')
        offsets = np.load(spec['index'], mmap_mode='r')
        if (len(offsets) != spec['records'] or offsets.dtype != np.uint64
                or (len(offsets) and (offsets[0] != 0 or offsets[-1] >= path.stat().st_size
                                     or np.any(offsets[1:] <= offsets[:-1])))):
            raise ValueError('Invalid byte offsets')
        if full_hash and (sha(path) != spec['sha256'] or sha(spec['index']) != spec['index_sha256']):
            raise ValueError('Shard/offset bytes changed')
        return spec['records']

    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(check, index['shards']))
        if full_hash:
            for path, expected in pool.map(lambda pair: (pair[0], sha(pair[0])), policy['inputs'].items()):
                if expected != policy['inputs'][path]:
                    raise ValueError('Source binding changed: ' + path)
    for spec in index['files'].values():
        if sha(spec['path']) != spec['sha256']:
            raise ValueError('Development artifact changed')
    totals = {split: sum(s['records'] for s in index['shards'] if s['split'] == split)
              for split in ('train', 'valid')}
    if totals != index['records'] or totals != {k: v['records'] for k, v in manifest['splits'].items()}:
        raise ValueError('Split totals disagree')
    return dict(status='passed', records=totals, rows=sum(counts), shards=len(counts),
                native_VNAR_storage=True, required_collator=READER,
                dataset_manifest_sha256=index['manifest_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_release(args.index)), flush=True)
