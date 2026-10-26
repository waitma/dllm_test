"""Read-only exact exposure audit for the fixed112000 -> LoRA3000 lineage.

Activate pllm, then run from repository root:
python -B scripts/downstream/audit_ab_pretraining_exposure.py --out ABS_NEW_DIRECTORY
Scans every V5 train/valid shard and the actual LoRA pilot training file. This
does not alter data, retrain models, or certify absence of homologous sequences.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dllm.pipelines.immune_llada.data.sources import _h3_core_from_chain


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def norm(value):
    return str(value or '').strip().upper().replace('-', '').replace('J', 'L')


def keys(row):
    chains = defaultdict(list)
    for role, seq in zip(row['chain_roles'], row['chains']):
        chains[role].append(norm(seq))
    h, l = chains['antibody_heavy'], chains['antibody_light']
    cores = {norm(row.get('identifiers', {}).get('heavy_h3_core', ''))}
    for i, role in enumerate(row['chain_roles']):
        if role == 'antibody_heavy':
            cores.add(norm(row.get('regions', {}).get(str(i), {}).get('CDR3', '')))
    cores.update(_h3_core_from_chain(s) for s in h)
    cores.discard('')
    return dict(heavy=set(h), light=set(l), pair={a+'|'+b for a in h for b in l}, h3=cores)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    ckpt = ROOT/'output/downstream_generation/post_training_lora_s42_r1_20261014/checkpoint'
    binding_path = ckpt/'training_binding.json'
    binding = json.loads(binding_path.read_text())
    plan_path = Path(binding['training_plan'])
    assert sha(plan_path) == binding['training_plan_sha256']
    plan = json.loads(plan_path.read_text())
    pilot_path = Path(plan['data_manifest'])
    assert sha(pilot_path) == plan['data_manifest_sha256']
    pilot = json.loads(pilot_path.read_text())
    parent = Path(binding['parent'])
    for path, digest in plan['parent_hashes'].items():
        assert sha(path) == digest, path
    provenance = json.loads((parent/'run_provenance.json').read_text())
    data = Path(provenance['data_dir'])
    manifest_path = data/'dataset_manifest.json'
    assert sha(manifest_path) == provenance['parent_metadata']['dataset_manifest.json']
    assert sha(manifest_path) == pilot['parent_manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    scope = ROOT/'output/v7_from_v5_scaffold_20261015/test_scope_v2'
    registry = json.loads((scope/'registry.json').read_text())
    assert sha(scope/'banks.json') == registry['banks_sha256']
    inputs_checked = {}
    for path, digest in registry['inputs'].items():
        if any(s in path for s in ('/cdr_infilling/', '/specificity/', '/dev/', 'ab_fixed_v2_fv_manifest')):
            assert sha(path) == digest, path
            inputs_checked[path] = digest
    banks = {k:v for k,v in json.loads((scope/'banks.json').read_text()).items()
             if k.startswith('ab_cdr/') or k in ('ab_repr/specificity', 'ab_repr/gdp_a1')}
    inverse = {role:defaultdict(set) for role in ('heavy', 'light', 'pair', 'h3')}
    for name, bank in banks.items():
        for role in inverse:
            for key in bank[role]:
                inverse[role][key].add(name)
    frozen = [Path(__file__), binding_path, plan_path, pilot_path, manifest_path,
              parent/'run_provenance.json', scope/'banks.json', scope/'registry.json']
    audit_plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                      lineage=dict(checkpoint=str(ckpt), parent=str(parent),
                                   pretraining_data=str(data), recipe=binding['recipe'], step=binding['step']),
                      artifacts={str(p):sha(p) for p in frozen},
                      evaluation_inputs_checked=inputs_checked,
                      cohorts={k:registry['cohorts'][k] for k in banks},
                      matching='exact normalized H, L, same-record H|L; explicit or motif H3 core',
                      limitations=['Exact overlap audit, not homology certification.',
                                   'Bank includes raw and audited Fv inputs; unique keys are not sample counts.',
                                   'Corpus membership is not per-optimizer-step sample replay.',
                                   'H3 is a shared local sequence, not full-antibody identity.'])
    write(out/'plan.json', audit_plan)
    entries = []
    for split, info in manifest['splits'].items():
        for entry in info['shards']:
            path = data/entry['path']
            entries.append((split, entry['source'], path, entry['records'], pilot['parent_shard_hashes'][str(path)]))
    spec = pilot['files']['train']
    entries.append(('lora_pilot_train', 'pilot_mixed', Path(spec['path']), spec['records'], spec['sha256']))
    counts = defaultdict(Counter)
    matched = defaultdict(lambda: defaultdict(set))
    witnesses = defaultdict(list)
    file_results = []
    for split, source, path, expected_n, digest in entries:
        h = hashlib.sha256()
        ab_n = 0
        before = path.stat()
        with path.open('rb') as f:
            n = 0
            for n, raw in enumerate(f, 1):
                h.update(raw)
                if b'"antibody_heavy"' not in raw and b'"antibody_light"' not in raw:
                    continue
                row = json.loads(raw)
                if not any(r in ('antibody_heavy','antibody_light') for r in row['chain_roles']):
                    continue
                ab_n += 1
                hit = defaultdict(dict)
                for role, values in keys(row).items():
                    for value in values:
                        for cohort in inverse[role].get(value, ()):
                            hit[cohort].setdefault(role, []).append(value)
                            matched[(split, cohort)][role].add(value)
                for cohort, roles in hit.items():
                    counts[(split, source, cohort)]['any_exact_key'] += 1
                    counts[(split, source, cohort)].update(roles.keys())
                    group = (split, cohort)
                    # Keep a few witnesses for each matching type, including paired matches.
                    for role, values in roles.items():
                        wg = (split, cohort, role)
                        if len(witnesses[wg]) < 5:
                            witnesses[wg].append(dict(path=str(path), line=n, source=source,
                                                      raw_sha256=hashlib.sha256(raw).hexdigest(), values=values))
        after = path.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns), path
        assert n == expected_n and h.hexdigest() == digest, (path,n,h.hexdigest())
        result = dict(split=split,source=source,path=str(path),rows=n,antibody_rows=ab_n,sha256=h.hexdigest())
        file_results.append(result)
        print(json.dumps(result), flush=True)
    for path, digest in audit_plan['artifacts'].items():
        assert sha(path) == digest, path
    output = dict(status='passed', meaning='scan integrity passed; see overlap counts, not a zero-overlap verdict',
                  files=file_results, counts=[dict(split=k[0],source=k[1],cohort=k[2],**v) for k,v in sorted(counts.items())],
                  matched_keys=[dict(split=k[0],cohort=k[1],keys={r:sorted(v) for r,v in d.items()})
                                for k,d in sorted(matched.items())],
                  witnesses=[dict(split=k[0],cohort=k[1],role=k[2],rows=v) for k,v in sorted(witnesses.items())])
    write(out/'exposure.json', output)
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
