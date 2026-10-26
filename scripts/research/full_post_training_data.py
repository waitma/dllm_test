"""Load full-data row-index views and the existing screening validation set.

Activate pllm/protenix_abtcr; exercised by test_fixed_chain_formal_post_training.py.
The source JSONL format and bytes remain unchanged; indices bind explicit exclusions.
"""
import json
from pathlib import Path
import numpy as np
from torch.utils.data import ConcatDataset, Dataset
from dllm.pipelines.immune_llada.data.dataset import PreparedImmuneDataset
from dllm.pipelines.immune_llada.data.records import BioSeqRecord
from scripts.research.run_fixed_chain_post_training import sha, assign_source_groups, source_of


class IndexedRows(Dataset):
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # uint64 NumPy indices must not enter the dataset's integer offset math.
        return self.dataset[int(self.indices[index])]


def load_full_data(index_path, expected_sha, pilot_path, pilot_sha):
    if sha(index_path) != expected_sha or sha(pilot_path) != pilot_sha:
        raise ValueError('Full-data index or pilot manifest changed')
    index = json.loads(Path(index_path).read_text())
    pilot = json.loads(Path(pilot_path).read_text())
    if (index['schema'] != 'fixed_chain.full_post_training_index.v1' or index['status'] != 'ready'
            or index['pilot_manifest_sha256'] != pilot_sha
            or index['parent_manifest_sha256'] != pilot['parent_manifest_sha256']):
        raise ValueError('Full-data and pilot provenance differ')
    if sha(index['parent_manifest']) != index['parent_manifest_sha256']:
        raise ValueError('Original manifest changed')
    datasets = []
    for source, file in sorted(index['files'].items()):
        if sha(file['path']) != file['sha256']:
            raise ValueError('Full-data row index changed')
        indices = np.load(file['path'], mmap_mode='r', allow_pickle=False)
        if (indices.ndim != 1 or indices.dtype != np.uint64 or len(indices) != file['retained_rows']
                or not len(indices) or int(indices[-1]) >= file['original_rows']
                or not np.all(indices[1:] > indices[:-1])):
            raise ValueError('Invalid ordered row index')
        dataset = PreparedImmuneDataset(Path(index['parent_manifest']).parent, split='train', source=source)
        if len(dataset) != file['original_rows']:
            raise ValueError('Dataset order/count changed')
        datasets.append(IndexedRows(dataset, indices))
    train = ConcatDataset(datasets)
    if len(train) != index['retained_train_rows']:
        raise ValueError('Indexed training coverage differs')
    ref = pilot['files']['screen']
    if sha(ref['path']) != ref['sha256']:
        raise ValueError('Screening validation rows changed')
    rows = assign_source_groups([json.loads(x) for x in Path(ref['path']).read_text().splitlines()],
                                pilot['source_counts'], 'screen')
    valid = {source: [BioSeqRecord.from_prepared_dict(row) for row in rows if source_of(row) == source]
             for source in sorted(index['files'])}
    selection = {source: {'file': ref['path'], 'sha256': ref['sha256'],
                          'source_rows': [i for i, row in enumerate(rows) if source_of(row) == source]}
                 for source in valid}
    return train, valid, selection
