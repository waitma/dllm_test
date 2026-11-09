"""Current immune model/data identities; activate pllm and run this module.

USE_TF=0 python -B -m dllm.pipelines.immune_llada.release_versions --verify-local
Dataset V10 and model V12 are independently versioned. This is source metadata,
not a claim that the model has been trained or that the old launch was migrated.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

MODEL_VERSION = 'V12'
MODEL_ARCHITECTURE = 'airgen_mint_v12'
MODEL_CHECKPOINT_SCHEMA = 'immune.fixed_chain.v12_airgen.v1'
DATA_VERSION = 'V10'
DATA_REVISION = 'nbheldout_20261106'
ROOT = Path('/project_c20250601/251105016/project/dllm_test')
DATA_POINTER = ROOT / 'data/prepared/LATEST_V10.json'
DATA_INDEX = ROOT / 'output/immune_v10_nbheldout_20261106/data/index.json'
DATA_INDEX_SHA256 = 'd9e905187f5c1661a89cb41bd2e8ec1f87ad09e072713ba1fa7ba1785d135d5f'
DATA_RECORDS = {'train': 10583642, 'valid': 87315}

CURRENT_RELEASE = dict(
    model=dict(version=MODEL_VERSION, architecture=MODEL_ARCHITECTURE,
               checkpoint_schema=MODEL_CHECKPOINT_SCHEMA,
               module='dllm.pipelines.immune_llada.fixed_chain.v12_airgen',
               parameters=77476073,
               description='MINT3x1280; QK LayerNorm; SwiGLU3584; per-layer/head inter/Pair/Recognition bias; '
                           'dynamic type feedback; mixed CDR corruption; no scope or family prediction'),
    data=dict(version=DATA_VERSION, revision=DATA_REVISION, pointer=str(DATA_POINTER),
              index=str(DATA_INDEX), index_sha256=DATA_INDEX_SHA256, records=DATA_RECORDS),
    training=dict(started=False, prepared_for_this_model_data_pair=False,
                  status='architecture_versioned; V10_dataset_verified; training_integration_pending',
                  pending=['Native VNAR reader integration', 'Span eligibility for VNAR with empty CDR2',
                           'Fresh V12/V10 plan and eight-rank profile/replay']),
)


def current_release():
    """Return a separate copy so callers cannot mutate the version registry."""
    return deepcopy(CURRENT_RELEASE)


def verify_local_data():
    """Check the latest pointer and immutable index identity without training."""
    pointer = json.loads(DATA_POINTER.read_text())
    if (pointer.get('status') != 'passed' or pointer.get('release_label') != DATA_VERSION
            or pointer.get('revision') != DATA_REVISION or pointer.get('index') != str(DATA_INDEX)
            or pointer.get('index_sha256') != DATA_INDEX_SHA256 or pointer.get('records') != DATA_RECORDS
            or hashlib.sha256(DATA_INDEX.read_bytes()).hexdigest() != DATA_INDEX_SHA256):
        raise ValueError('Latest V10 data differs from the recorded model/data release identities')
    index = json.loads(DATA_INDEX.read_text())
    if (index.get('release_label') != DATA_VERSION or index.get('revision') != DATA_REVISION
            or index.get('records') != DATA_RECORDS):
        raise ValueError('V10 index metadata differs from the recorded release')
    return dict(status='passed', scope='pointer_and_index_identity; no_full_shard_rehash',
                records=deepcopy(DATA_RECORDS), index_sha256=DATA_INDEX_SHA256)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-local', action='store_true')
    args = parser.parse_args()
    result = current_release()
    if args.verify_local:
        result['local_data_check'] = verify_local_data()
    print(json.dumps(result, indent=2))
