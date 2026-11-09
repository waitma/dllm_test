"""Validate an immutable, verified subset of a previously accepted data release.

Activate pllm; exercised by scripts.data.clean_binding_unseen_v9 and the normal
scripts.research.run_bert_mlm.data_gate. No source data are changed here.
"""
import json
from pathlib import Path

from scripts.research.bert_mlm_data import sha


def filtered_release_binding(data_dir, manifest):
    from scripts.research.bert_mlm_v8_data import release_binding, BINDING_SCHEMA
    filtering = manifest['binding_unseen_filter']
    parent_dir = Path(filtering['parent'])
    if parent_dir.resolve() == data_dir.resolve():
        raise ValueError('Cyclic filtered parent')
    if sha(parent_dir / 'dataset_manifest.json') != filtering['parent_manifest_sha256']:
        raise ValueError('Filtered parent manifest changed')
    # Archival ancestry is readable; direct new training on a superseded root
    # remains forbidden by release_binding's default entrypoint.
    parent = release_binding(parent_dir, _allow_superseded_ancestor=True)
    config = data_dir / manifest['reader_contract']['config_path']
    if (sha(config) != parent['config_sha256'] or manifest['reader_contract'] !=
            json.loads(Path(parent['manifest']).read_text())['reader_contract']):
        raise ValueError('Filtered release changed reader/config')
    policy = Path(filtering['policy_path'])
    receipt_path = Path(filtering['shard_receipts'])
    if sha(policy) != filtering['policy_sha256'] or sha(receipt_path) != filtering['shard_receipts_sha256']:
        raise ValueError('Filter policy/receipts changed')
    receipts = json.loads(receipt_path.read_text())['shards']
    previous = {str(Path(s['path']).relative_to(parent_dir)): s for s in parent['shards']}
    if set(previous) != set(receipts):
        raise ValueError('Filtered receipts do not cover every parent shard')
    digest = sha(data_dir / 'dataset_manifest.json')
    evidence = {}
    for name in ('verification.json', 'reader_acceptance.json', 'independent_acceptance.json'):
        path = data_dir / name
        check = json.loads(path.read_text())
        if (check['status'] != 'passed' or check['dataset_manifest_sha256'] != digest
                or check['config_sha256'] != parent['config_sha256']
                or check['policy_sha256'] != sha(policy)
                or check['shard_receipts_sha256'] != sha(receipt_path)):
            raise ValueError('Filtered acceptance missing/stale')
        for key, filename in (('collator_sha256', 'layout.py'), ('objective_sha256', 'unbinding_mlm.py'),
                              ('vhh_objective_sha256', 'unbinding_vhh.py')):
            if check[key] != sha(Path(__file__).resolve().parents[2] /
                                  'dllm/pipelines/immune_llada/fixed_chain' / filename):
                raise ValueError('Filtered reader changed')
        evidence[str(path)] = sha(path)
    seen, shards = set(), []
    for split, block in manifest['splits'].items():
        removed = sum(r['removed'] for rel,r in receipts.items() if previous[rel]['split'] == split)
        if (sum(s['records'] for s in block['shards']) != block['records']
                or block['records'] != parent['records'][split] - removed):
            raise ValueError('Filtered split totals differ')
        for spec in block['shards']:
            rel = spec['path']
            if rel in seen or rel not in previous:
                raise ValueError('Unexpected/repeated filtered shard')
            seen.add(rel)
            old, receipt = previous[rel], receipts[rel]
            path = data_dir / rel
            resolved = path.resolve(strict=True)
            if (receipt['parent_sha256'] != old['sha256']
                    or receipt['parent_records'] != old['records']
                    or spec['records'] != old['records'] - receipt['removed']
                    or spec['sha256'] != receipt['sha256']
                    or spec['source'] != old['source'] or spec['split'] != split
                    or old['split'] != split or spec['records'] < 0):
                raise ValueError('Filtered shard does not match its receipt')
            if receipt['removed']:
                if resolved != path:
                    raise ValueError('Changed filtered shard is not independently owned')
            elif resolved != Path(old['resolved_path']) or spec['sha256'] != old['sha256']:
                raise ValueError('Unchanged filtered shard differs from parent')
            shards.append(dict(path=str(path), resolved_path=str(resolved), split=split,
                source=spec['source'], records=spec['records'], sha256=spec['sha256']))
    if seen != set(previous):
        raise ValueError('Incomplete filtered shard coverage')
    return dict(schema=BINDING_SCHEMA, data_dir=str(data_dir), manifest=str(data_dir / 'dataset_manifest.json'),
        manifest_sha256=digest, config_path=str(config), config_sha256=parent['config_sha256'],
        model=parent['model'], evidence=evidence, shards=shards,
        records={s: b['records'] for s,b in manifest['splits'].items()},
        required_objective=parent['required_objective'], filter_policy_sha256=sha(policy))
