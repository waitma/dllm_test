"""Bind V8 to the existing indexed BERT trainer without weakening V7 gates.

Activate pllm; run USE_TF=0 python -B -m scripts.research.bert_mlm_v8_data.
V7 offsets/dev probes are reused only after hash and coverage verification.
Restored negatives receive new byte offsets and additional token validation.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path

from scripts.data.prepare_immune_v8 import OUTPUT, PARENT, PROBE_INDEX, WORK, parent_gate, held_probe_keys
from scripts.research.bert_mlm_data import sha, write_json, scan_shard

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / 'output/bert_unbinding_v8_from287000_20261026/data'
BINDING_SCHEMA = 'immune.bert_mlm.v8_binding.v1'


def release_binding(data_dir):
    data_dir = Path(data_dir)
    manifest_path = data_dir / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if 'antibody_expansion' in manifest:
        return antibody_release_binding(data_dir, manifest)
    if 'mhc_completion' in manifest:
        return completed_release_binding(data_dir, manifest)
    digest = sha(manifest_path)
    restoration = manifest['v8_restoration']
    parent = Path(restoration['parent'])
    if sha(parent / 'dataset_manifest.json') != restoration['parent_manifest_sha256']:
        raise ValueError('V8 immutable V7 parent changed')
    parent_manifest = parent_gate(parent)
    contract = manifest['reader_contract']
    config_path = data_dir / contract['config_path']
    model = json.loads(config_path.read_text())
    if (contract['layout'] != 'immune.fixed_chain.f011.v1' or contract['a4_allowed']
            or sha(config_path) != contract['config_sha256']
            or (model['first_chain_capacity'], model['second_chain_capacity'],
                model['full_capacity'], model['max_length']) != (167, 135, 304, 2048)):
        raise ValueError('V8 layout/config changed')
    evidence = {}
    for name in ('verification.json', 'reader_acceptance.json'):
        path = data_dir / name
        check = json.loads(path.read_text())
        if check['status'] != 'passed' or check['dataset_manifest_sha256'] != digest:
            raise ValueError('V8 acceptance changed: ' + name)
        if (check['config_sha256'] != sha(config_path)
                or check['collator_sha256'] != sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/layout.py')
                or check['objective_sha256'] != sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/unbinding_mlm.py')):
            raise ValueError('V8 reader/objective changed after acceptance')
        evidence[str(path)] = sha(path)
    base = {s['path']: s for block in parent_manifest['splits'].values() for s in block['shards']}
    shards, inherited = [], set()
    for split, block in manifest['splits'].items():
        if sum(s['records'] for s in block['shards']) != block['records']:
            raise ValueError('V8 manifest totals differ')
        for entry in block['shards']:
            path = data_dir / entry['path']
            resolved = path.resolve(strict=True)
            if entry['path'] in base:
                original = base[entry['path']]
                if (resolved != (parent / entry['path']).resolve(strict=True)
                        or any(entry[k] != original[k] for k in ('records', 'sha256', 'source'))):
                    raise ValueError('V8 inherited shard differs from V7')
                inherited.add(entry['path'])
            elif resolved != path or not path.name.startswith('unbind_'):
                raise ValueError('Unexpected restored V8 shard')
            shards.append(dict(path=str(path), resolved_path=str(resolved), split=split,
                               source=entry['source'], records=entry['records'], sha256=entry['sha256']))
    if inherited != set(base) or len({s['path'] for s in shards}) != len(shards):
        raise ValueError('V8 must preserve every V7 shard exactly once')
    return dict(schema=BINDING_SCHEMA, data_dir=str(data_dir), manifest=str(manifest_path),
                manifest_sha256=digest, config_path=str(config_path), config_sha256=sha(config_path),
                model=model, evidence=evidence, shards=shards,
                records={s: b['records'] for s, b in manifest['splits'].items()})


def antibody_release_binding(data_dir, manifest):
    """Bind a separately verified antibody addition without changing its parent."""
    expansion = manifest['antibody_expansion']
    parent_dir = Path(expansion['parent'])
    if parent_dir.resolve() == data_dir.resolve():
        raise ValueError('Cyclic antibody parent')
    parent = release_binding(parent_dir)
    if parent['manifest_sha256'] != expansion['parent_manifest_sha256']:
        raise ValueError('Antibody expansion parent changed')
    digest = sha(data_dir / 'dataset_manifest.json')
    config_path = data_dir / manifest['reader_contract']['config_path']
    if (sha(config_path) != parent['config_sha256']
            or manifest['reader_contract'] != json.loads(Path(parent['manifest']).read_text())['reader_contract']):
        raise ValueError('Antibody expansion model/layout changed')
    evidence = {}
    for name in ('verification.json', 'reader_acceptance.json', 'independent_acceptance.json'):
        path = data_dir / name
        check = json.loads(path.read_text())
        if (check['status'] != 'passed' or check['dataset_manifest_sha256'] != digest
                or check['config_sha256'] != parent['config_sha256']
                or check['collator_sha256'] != sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/layout.py')
                or check['objective_sha256'] != sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/unbinding_mlm.py')):
            raise ValueError('Antibody expansion evidence changed')
        evidence[str(path)] = sha(path)
    previous = {str(Path(s['path']).relative_to(parent_dir)): s for s in parent['shards']}
    additions = {s['path']: s for s in expansion['added_shards']}
    if set(previous) & set(additions) or len(additions) != len(expansion['added_shards']):
        raise ValueError('Antibody addition paths overlap')
    shards, inherited, added = [], set(), set()
    for split, block in manifest['splits'].items():
        extra = sum(s['records'] for s in additions.values() if s['split'] == split)
        if (block['records'] != parent['records'][split] + extra
                or sum(s['records'] for s in block['shards']) != block['records']):
            raise ValueError('Antibody expansion counts differ')
        for spec in block['shards']:
            path = data_dir / spec['path']
            resolved = path.resolve(strict=True)
            if spec['path'] in previous:
                old = previous[spec['path']]
                if (resolved != Path(old['resolved_path'])
                        or any(spec[k] != old[k] for k in ('records', 'sha256', 'source', 'split'))):
                    raise ValueError('Antibody child modified a parent shard')
                inherited.add(spec['path'])
            elif spec['path'] in additions:
                if (spec != additions[spec['path']] or resolved != path
                        or not path.name.startswith('abexp_') or spec['source'] != 'igdesign_antibody'
                        or spec['records'] <= 0):
                    raise ValueError('Unexpected antibody addition')
                added.add(spec['path'])
            else:
                raise ValueError('Unbound antibody shard')
            shards.append(dict(path=str(path), resolved_path=str(resolved), split=split,
                source=spec['source'], records=spec['records'], sha256=spec['sha256']))
    if (inherited != set(previous) or added != set(additions)
            or len(shards) != len(previous) + len(additions)):
        raise ValueError('Antibody child shard coverage differs')
    return dict(schema=BINDING_SCHEMA, data_dir=str(data_dir), manifest=str(data_dir / 'dataset_manifest.json'),
        manifest_sha256=digest, config_path=str(config_path), config_sha256=parent['config_sha256'],
        model=parent['model'], evidence=evidence, shards=shards,
        records={s: b['records'] for s, b in manifest['splits'].items()},
        antibody_parent_binding_sha256=hashlib.sha256(json.dumps(parent, sort_keys=True).encode()).hexdigest())


def completed_release_binding(data_dir, manifest):
    """Bind a verified MHC-only child release to its unchanged V8 parent."""
    completion = manifest['mhc_completion']
    parent_dir = Path(completion['parent'])
    if parent_dir.resolve() == data_dir.resolve():
        raise ValueError('Cyclic MHC parent')
    parent = release_binding(parent_dir)
    if parent['manifest_sha256'] != completion['parent_manifest_sha256']:
        raise ValueError('MHC completion parent manifest changed')
    digest = sha(data_dir / 'dataset_manifest.json')
    config_path = data_dir / manifest['reader_contract']['config_path']
    if (sha(config_path) != parent['config_sha256']
            or manifest['reader_contract'] != json.loads(Path(parent['manifest']).read_text())['reader_contract']):
        raise ValueError('MHC completion model/layout changed')
    evidence = {}
    for name in ('verification.json', 'reader_acceptance.json', 'independent_acceptance.json'):
        path = data_dir / name
        check = json.loads(path.read_text())
        if check['status'] != 'passed' or check['dataset_manifest_sha256'] != digest:
            raise ValueError('MHC completion acceptance is missing/stale')
        if name != 'independent_acceptance.json' and (
                check['config_sha256'] != parent['config_sha256']
                or check['collator_sha256'] != sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/layout.py')
                or check['objective_sha256'] != sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/unbinding_mlm.py')):
            raise ValueError('MHC completion reader/objective changed')
        evidence[str(path)] = sha(path)
    previous = {str(Path(s['path']).relative_to(parent_dir)): s for s in parent['shards']}
    shards = []
    for split, block in manifest['splits'].items():
        if block['records'] != parent['records'][split] or sum(s['records'] for s in block['shards']) != block['records']:
            raise ValueError('MHC completion split totals changed')
        for spec in block['shards']:
            old = previous.pop(spec['path'])
            path = data_dir / spec['path']
            resolved = path.resolve(strict=True)
            if (any(spec[k] != old[k] for k in ('source', 'split', 'records'))
                    or spec['parent_sha256'] != old['sha256']):
                raise ValueError('MHC completion shard identity changed')
            if spec['sha256'] == old['sha256']:
                if spec['mhc_completed'] != 0 or resolved != Path(old['resolved_path']):
                    raise ValueError('Unchanged MHC child shard target changed')
            elif (not 0 < spec['mhc_completed'] <= spec['records'] or resolved != path
                  or not path.name.startswith('unbind_') or spec['source'] not in ('trait', 'tcr_native', 'tcr_papers')):
                raise ValueError('Unexpected changed MHC shard')
            shards.append(dict(path=str(path), resolved_path=str(resolved), split=split,
                               source=spec['source'], records=spec['records'], sha256=spec['sha256']))
    if previous:
        raise ValueError('MHC child does not cover its entire parent')
    return dict(schema=BINDING_SCHEMA, data_dir=str(data_dir), manifest=str(data_dir / 'dataset_manifest.json'),
        manifest_sha256=digest, config_path=str(config_path), config_sha256=parent['config_sha256'],
        model=parent['model'], evidence=evidence, shards=shards, records=parent['records'],
        mhc_parent_binding_sha256=hashlib.sha256(json.dumps(parent, sort_keys=True).encode()).hexdigest())


def verify_binding(path, expected_sha, model):
    if sha(path) != expected_sha:
        raise ValueError('V8 data binding changed')
    bound = json.loads(Path(path).read_text())
    if (bound['schema'] != BINDING_SCHEMA or bound['model'] != model
            or bound != release_binding(Path(bound['data_dir']))):
        raise ValueError('V8 release/model/evidence changed')
    return bound


def prepare(workers=8, data_dir=OUTPUT, output_dir=INDEX_DIR):
    bound = release_binding(data_dir)
    old = json.loads(PROBE_INDEX.read_text())
    if old['manifest_sha256'] != sha(PARENT / 'dataset_manifest.json'):
        raise ValueError('V7 offset index belongs to another release')
    held = held_probe_keys()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / 'binding.json', bound)
    expected = {s['path']: s for s in bound['shards']}
    def reuse(entry):
        path = data_dir / Path(entry['path']).relative_to(PARENT)
        reference = expected[str(path)]
        if any(entry[k] != reference[k] for k in ('resolved_path', 'records', 'sha256', 'source', 'split')):
            raise ValueError('V7 offset record contract differs')
        if sha(path) != entry['sha256'] or sha(entry['index']) != entry['index_sha256']:
            raise ValueError('V7 shard or offsets changed')
        return dict(entry, path=str(path))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(reuse, old['shards']))
    files = {}
    for name, reference in old['files'].items():
        if sha(reference['path']) != reference['sha256']:
            raise ValueError('V7 development file changed')
        files[name] = deepcopy(reference)
    manifest = json.loads((data_dir / 'dataset_manifest.json').read_text())
    jobs = [(str(data_dir), dict(s, split=split, receptor_capacities=dict(
                antibody_heavy=167, antibody_light=135, tcr_beta=167, tcr_alpha=135)), str(output_dir), held)
            for split, block in manifest['splits'].items() for s in block['shards']
            if Path(s['path']).name.startswith(('unbind_', 'abexp_'))]
    negatives = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(scan_shard, jobs):
            reference = expected[result['path']]
            if (any(result[k] != reference[k] for k in ('resolved_path', 'records', 'sha256', 'source', 'split'))
                    or result['excluded'] or (result['split'] == 'train' and result['overlap'])):
                raise ValueError('Restored negative index/filtering differs')
            negatives.extend(json.loads(text) for _, text in result.pop('validation'))
            result.pop('probe')
            result.pop('overlap')
            results.append(result)
    if set(expected) != {s['path'] for s in results}:
        raise ValueError('V8 offset coverage differs')
    token_rows = [json.loads(line) for line in Path(files['token_valid']['path']).read_text().splitlines()]
    token_rows.extend(negatives)
    from dllm.pipelines.immune_llada.fixed_chain.unbinding_mlm import UnbindingMLMCollator
    from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
    collator = UnbindingMLMCollator(FixedChainConfig(**bound['model']))
    for row in token_rows:
        collator([row['record']])
    token_path = output_dir / 'token_valid.jsonl'
    token_path.write_text(''.join(json.dumps(row) + '\n' for row in token_rows))
    files['token_valid'] = dict(path=str(token_path), sha256=sha(token_path), records=len(token_rows))
    index = dict(schema='immune.bert_mlm.index.v1', binding_kind=BINDING_SCHEMA,
        data_dir=str(data_dir), manifest=bound['manifest'], manifest_sha256=bound['manifest_sha256'],
        binding=str(output_dir / 'binding.json'), binding_sha256=sha(output_dir / 'binding.json'),
        shards=sorted(results, key=lambda s: (s['split'], s['path'])), files=files, records=bound['records'],
        probe_sources=old['probe_sources'], parent_index=str(PROBE_INDEX), parent_index_sha256=sha(PROBE_INDEX),
        probe_valid_candidates_exposed_to_train=0,
        additional_negative_token_valid=sum(r['record'].get('labels', {}).get('relation') == 'nonbinding' for r in negatives),
        additional_token_valid=len(negatives),
        probe_protocol=old['probe_protocol'])
    for split, count in bound['records'].items():
        if sum(s['records'] for s in results if s['split'] == split) != count:
            raise ValueError('V8 indexed totals differ')
    write_json(output_dir / 'index.json', index)
    print(json.dumps(dict(status='passed', index=str(output_dir / 'index.json'), records=bound['records'],
                          negative_validation=len(negatives))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=OUTPUT)
    parser.add_argument('--output', type=Path, default=INDEX_DIR)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    prepare(args.workers, args.data_dir.resolve(), args.output.resolve())
