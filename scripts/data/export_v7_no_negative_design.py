"""Filter V5 into a separately verified, negative-free, fixed-chain data release.

Activate pllm and protenix_abtcr. Run from the repository root:
  USE_TF=0 python -B scripts/data/export_v7_no_negative_design.py candidates --workers 8
  USE_TF=0 python -B scripts/data/export_v7_no_negative_design.py match --workers 16
  USE_TF=0 python -B scripts/data/export_v7_no_negative_design.py export --workers 8

The source corpus, downstream splits, reader and model are never modified.
Every retained JSONL record is copied byte-for-byte. Audit-key repairs do not
change stored sequences, masks, identifiers, chain positions or EOS supervision.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.rebuild_v7_from_v5 import SOURCE, WORK, sha, write
from scripts.data.v7_rebuild_scope import SCOPE_DIR, SCOPE, freeze, verify as verify_scope
from scripts.data.export_v7_binding_design import (
    save_pickle, read_pickle, upstream_lookup, audit_keys, unresolved_tcr, bank_index, reasons,
)

RUN = WORK / 'no_negative_r1'
OUTPUT = ROOT / 'data/prepared/immune_v7_from_v5_no_negative_design_20261015'
MMSEQS = Path('/project_c20250601/251105016/conda/envs/flow/bin/mmseqs')
LAYOUT = 'immune.fixed_chain.f011.v1'
LABEL_POLICY = 'exclude explicit nonbinding; retain binding and unlabeled/unknown unchanged'


def excluded_by_label(row):
    relation = row.get('labels', {}).get('relation', 'absent')
    assert relation in {'binding', 'nonbinding', 'unknown', 'absent'}, relation
    return relation == 'nonbinding'


def check_bindings():
    verify_scope()
    binding = json.loads((RUN / 'binding.json').read_text())
    for path, digest in binding['files'].items():
        assert sha(path) == digest, path
    return binding


def candidate_shard(shard):
    import numpy as np
    name = shard['path'].replace('/', '__')
    folder = RUN / 'candidates'
    receipt = folder / (name + '.json')
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        assert previous['source_sha256'] == sha(SOURCE / shard['path'])
        for path, digest in previous['files'].items():
            assert sha(path) == digest
        return previous
    folder.mkdir(parents=True, exist_ok=True)
    candidates, counts, values, near = [], Counter(), defaultdict(set), defaultdict(set)
    states = np.zeros(shard['records'], dtype=np.uint8)
    lookup = upstream_lookup()
    digest = hashlib.sha256()
    with (SOURCE / shard['path']).open('rb') as handle:
        n = 0
        for raw in handle:
            digest.update(raw)
            assert raw.strip() and raw.endswith(b'\n'), shard['path']
            row = json.loads(raw)
            relation = row.get('labels', {}).get('relation', 'absent')
            counts['label/' + relation] += 1
            assert relation in {'binding', 'nonbinding', 'unknown', 'absent'}, relation
            if excluded_by_label(row):
                states[n] = 1
            else:
                keys, issues = audit_keys(row, lookup)
                counts.update('key/' + issue for issue in issues)
                if unresolved_tcr(keys):
                    states[n] = 3  # unresolved observed core is quarantined
                    counts['unresolved_tcr_core'] += 1
                else:
                    candidates.append((n, keys))
                    for role, value in keys.items():
                        if isinstance(value, str) and value:
                            values[role].add(value)
                            if not keys['paired_tcr'] and role in {'cdr3a', 'cdr3b', 'peptide'}:
                                near[role].add(value)
            n += 1
    assert n == shard['records'] and digest.hexdigest() == shard['source_sha256']
    key_path, state_path = folder / (name + '.pkl'), folder / (name + '.npy')
    save_pickle(key_path, dict(candidates=candidates, values=dict(values), near=dict(near)))
    np.save(state_path, states, allow_pickle=False)
    result = dict(path=shard['path'], source=shard['source'], split=shard['split'], records=n,
                  source_sha256=digest.hexdigest(), candidates=len(candidates), counts=dict(counts),
                  key_path=str(key_path), state_path=str(state_path),
                  files={str(p): sha(p) for p in (key_path, state_path)})
    write(receipt, result)
    return result


def candidates(workers):
    freeze()
    RUN.mkdir(parents=True, exist_ok=True)
    inv = json.loads((WORK / 'inventory.json').read_text())
    assert inv['status'] == 'complete'
    assert inv['manifest_sha256'] == sha(SOURCE / 'dataset_manifest.json')
    files = [Path(__file__), ROOT / 'scripts/data/export_v7_binding_design.py',
             WORK / 'inventory.json', WORK / 'inventory_binding.json',
             SOURCE / 'dataset_manifest.json', SCOPE_DIR / 'registry.json', SCOPE_DIR / 'banks.json',
             ROOT / 'dllm/pipelines/immune_llada/fixed_chain/layout.py',
             ROOT / 'configs/models/v5_fixed_chain_f011_75m_20261012/immune_config.json']
    binding = dict(scope=SCOPE, relation_policy=LABEL_POLICY,
                   layout=LAYOUT, files={str(p): sha(p) for p in files})
    if (RUN / 'binding.json').exists():
        assert json.loads((RUN / 'binding.json').read_text()) == binding
    else:
        write(RUN / 'binding.json', binding)
    receipts = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(candidate_shard, shard) for shard in inv['shards']]
        for future in as_completed(futures):
            item = future.result()
            receipts.append(item)
            print(json.dumps(dict(stage='candidates', completed=len(receipts), total=len(inv['shards']),
                                  shard=item['path'], eligible_candidates=item['candidates'])), flush=True)
    receipts.sort(key=lambda x: x['path'])
    values, near, counts = defaultdict(set), defaultdict(set), Counter()
    for r in receipts:
        counts.update(r['counts'])
        data = read_pickle(Path(r['key_path']))
        for role, seqs in data['values'].items():
            values[role].update(seqs)
        for role, seqs in data['near'].items():
            near[role].update(seqs)
    path = RUN / 'candidate_keys.pkl'
    save_pickle(path, dict(values=dict(values), near=dict(near)))
    write(RUN / 'candidates.json', dict(status='complete', counts=dict(counts), shards=receipts,
          records=sum(r['records'] for r in receipts), candidates=sum(r['candidates'] for r in receipts),
          unique={r: len(s) for r, s in values.items()}, near_unique={r: len(s) for r, s in near.items()},
          keys_sha256=sha(path), binding_sha256=sha(RUN / 'binding.json')))


def near_matches(role, sequences, refs, workers):
    import numpy as np
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
    path = RUN / ('near_' + role + '.pkl')
    receipt = path.with_suffix('.json')
    identity = dict(candidates_sha256=sha(RUN / 'candidates.json'), registry_sha256=sha(SCOPE_DIR / 'registry.json'),
                    metric='global normalized Levenshtein', cutoff=0.8)
    if receipt.exists():
        report = json.loads(receipt.read_text())
        assert report['identity'] == identity and sha(path) == report['sha256']
        return read_pickle(path)
    start = time.monotonic()
    hits, witnesses = {}, {}
    sequences, references = sorted(sequences), sorted(refs)
    assert references or not sequences, role
    for start_index in range(0, len(sequences), 512):
        batch = sequences[start_index:start_index + 512]
        scores = process.cdist(batch, references, scorer=Levenshtein.normalized_similarity,
                               score_cutoff=0.8, workers=workers, dtype=np.float32)
        for i in np.flatnonzero(scores.max(axis=1) > 0):
            mask = 0
            for j in np.flatnonzero(scores[i] > 0):
                mask |= refs[references[j]]
            hits[batch[i]] = mask
            j = int(scores[i].argmax())
            witnesses[batch[i]] = (references[j], float(scores[i, j]))
        if start_index % (512 * 50) == 0:
            print(json.dumps(dict(stage='near', role=role, scanned=min(start_index+512, len(sequences)),
                                  total=len(sequences), matches=len(hits))), flush=True)
    save_pickle(path, hits)
    save_pickle(RUN / ('near_' + role + '_witnesses.pkl'), witnesses)
    write(receipt, dict(identity=identity, scanned=len(sequences), refs=len(references), matches=len(hits),
                        elapsed_seconds=time.monotonic()-start, sha256=sha(path)))
    return hits


def clustered_matches(role, sequences, refs, threshold, workers):
    folder = RUN / 'clusters'
    folder.mkdir(exist_ok=True)
    path, receipt = folder / (role + '.pkl'), folder / (role + '.json')
    allseqs = sorted(set(sequences) | set(refs))
    identity = dict(candidates_sha256=sha(RUN / 'candidates.json'), registry_sha256=sha(SCOPE_DIR / 'registry.json'),
                    threshold=threshold, coverage=0.8, binary_sha256=sha(MMSEQS),
                    sequences_sha256=hashlib.sha256('\n'.join(allseqs).encode()).hexdigest())
    if receipt.exists():
        report = json.loads(receipt.read_text())
        assert report['identity'] == identity and sha(path) == report['sha256']
        return read_pickle(path)
    fasta = folder / (role + '.fasta')
    with fasta.open('w') as handle:
        for i, seq in enumerate(allseqs):
            handle.write(f'>s{i}\n{seq}\n')
    prefix = folder / (role + '_mmseqs')
    command = [str(MMSEQS), 'easy-linclust', str(fasta), str(prefix), str(folder / ('tmp_' + role)),
               '--min-seq-id', str(threshold), '-c', '0.8', '--cov-mode', '0', '--cluster-mode', '1',
               '--threads', str(workers), '--remove-tmp-files', '1']
    print(json.dumps(dict(stage='cluster', role=role, sequences=len(allseqs))), flush=True)
    with (folder / (role + '.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    membership = {}
    with Path(str(prefix) + '_cluster.tsv').open() as handle:
        for line in handle:
            representative, member = line.strip().split('\t')
            index = int(member[1:])
            assert index not in membership
            membership[index] = representative
    assert set(membership) == set(range(len(allseqs)))
    cohort_masks = defaultdict(int)
    for i, seq in enumerate(allseqs):
        cohort_masks[membership[i]] |= refs.get(seq, 0)
    matches = {seq: cohort_masks[membership[i]] for i, seq in enumerate(allseqs)
               if seq in sequences and cohort_masks[membership[i]]}
    save_pickle(path, matches)
    write(receipt, dict(identity=identity, sequences=len(allseqs), candidate_sequences=len(sequences),
                        matches=len(matches), command=command, sha256=sha(path),
                        membership_sha256=sha(Path(str(prefix) + '_cluster.tsv'))))
    return matches


def match(workers):
    check_bindings()
    report = json.loads((RUN / 'candidates.json').read_text())
    assert report['keys_sha256'] == sha(RUN / 'candidate_keys.pkl')
    data = read_pickle(RUN / 'candidate_keys.pkl')
    names, exact, t4 = bank_index()
    near, clusters, contained = {}, {}, {}
    for role in ('peptide', 'cdr3a', 'cdr3b'):
        near[role] = near_matches(role, data['near'].get(role, set()), exact.get(role, {}), workers)
    for role, threshold in (('heavy', .95), ('h3', .8), ('antigen', .7)):
        clusters[role] = clustered_matches(role, data['values'].get(role, set()), exact.get(role, {}), threshold, workers)
    for antigen in data['values'].get('antigen', []):
        mask = 0
        for peptide, value in exact.get('peptide', {}).items():
            if peptide in antigen:
                mask |= value
        if mask:
            contained[antigen] = mask
    path = RUN / 'match_index.pkl'
    save_pickle(path, dict(names=names, exact=exact, t4=t4, near=near, clusters=clusters, contained=contained))
    write(RUN / 'matching.json', dict(status='complete', index_sha256=sha(path),
          registry_sha256=sha(SCOPE_DIR / 'registry.json'), candidates_sha256=sha(RUN / 'candidates.json'),
          near={r: len(d) for r, d in near.items()}, clusters={r: len(d) for r, d in clusters.items()},
          antigen_containment=len(contained)))


def export(workers):
    import numpy as np
    check_bindings()
    inventory = json.loads((RUN / 'candidates.json').read_text())
    matching = json.loads((RUN / 'matching.json').read_text())
    assert matching['status'] == 'complete' and matching['candidates_sha256'] == sha(RUN / 'candidates.json')
    assert matching['index_sha256'] == sha(RUN / 'match_index.pkl')
    index = read_pickle(RUN / 'match_index.pkl')
    assert not OUTPUT.exists(), 'Use a new release path; never overwrite a dataset'
    stage = OUTPUT.with_name('.' + OUTPUT.name + '.staging')
    stage.mkdir(exist_ok=False)
    (stage / 'selection').mkdir()
    total, by_source, shard_results = Counter(), defaultdict(Counter), []
    with (stage / 'evaluation_exclusions.jsonl').open('w') as ledger:
        for shard in inventory['shards']:
            for path, digest in shard['files'].items():
                assert sha(path) == digest
            candidates = read_pickle(Path(shard['key_path']))['candidates']
            state = np.load(shard['state_path'], allow_pickle=False)
            c = by_source[shard['split'] + '/' + shard['source']]
            for row_index, keys in candidates:
                assert state[row_index] == 0
                hits = reasons(keys, index)
                if hits:
                    state[row_index] = 4
                    ledger.write(json.dumps(dict(shard=shard['path'], row=row_index, reasons=hits,
                                                 paired_tcr=keys['paired_tcr'])) + '\n')
                    c.update('reason/' + name for name in hits)
            kept = np.flatnonzero(state == 0)
            assert len(kept) <= shard['candidates']
            target = stage / shard['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            h, oh, rows, written = hashlib.sha256(), hashlib.sha256(), 0, 0
            retained_labels = Counter()
            with (SOURCE / shard['path']).open('rb') as source, target.open('wb') as out:
                for raw in source:
                    h.update(raw)
                    if state[rows] == 0:
                        out.write(raw)
                        oh.update(raw)
                        written += 1
                        relation = json.loads(raw).get('labels', {}).get('relation', 'absent')
                        retained_labels['retained_' + relation] += 1
                    rows += 1
            assert h.hexdigest() == shard['source_sha256'] and rows == shard['records']
            assert written == len(kept)
            codes = np.bincount(state, minlength=5)
            counts = dict(input=rows, retained=int(codes[0]), nonbinding=int(codes[1]),
                          unlabeled_removed_by_policy=int(codes[2]), unresolved_tcr_core=int(codes[3]),
                          test_overlap=int(codes[4]), **dict(retained_labels))
            assert codes[2] == 0 and not retained_labels['retained_nonbinding']
            assert sum(retained_labels.values()) == written
            c.update(counts)
            total.update(counts)
            state_file = stage / 'selection' / (shard['path'].replace('/', '__') + '.npy')
            np.save(state_file, state, allow_pickle=False)
            shard_results.append(dict(path=shard['path'], source=shard['source'], split=shard['split'],
                 records=written, sha256=oh.hexdigest(), input_records=rows, input_sha256=h.hexdigest(),
                 selection_path=str(state_file.relative_to(stage)), selection_sha256=sha(state_file), counts=counts))
            print(json.dumps(dict(stage='export', shard=shard['path'], **counts)), flush=True)
    # A second source-to-output pass reconstructs keys independently of the
    # candidate cache and verifies each emitted byte and each selection decision.
    verified = verify_output(stage, shard_results, workers)
    assert verified['forbidden_overlap'] == verified['nonbinding'] == 0
    assert verified['binding'] == total['retained_binding']
    assert verified['unknown'] == total['retained_unknown']
    assert verified['absent'] == total['retained_absent']
    manifest = json.loads((SOURCE / 'dataset_manifest.json').read_text())
    manifest['created_at'] = datetime.now(timezone.utc).isoformat()
    manifest['dataset_version'] = OUTPUT.name
    manifest['splits'] = {}
    for split in ('train', 'valid'):
        items = [s for s in shard_results if s['split'] == split and s['records']]
        sources = Counter()
        for item in items:
            sources[item['source']] += item['records']
        manifest['splits'][split] = dict(records=sum(s['records'] for s in items), shards=items,
            sources={name: dict(records=n, input_records=by_source[split + '/' + name]['input'])
                     for name, n in sources.items()})
    manifest['sources'] = sorted({s['source'] for s in shard_results if s['records']})
    manifest['reader_contract'] = dict(layout=LAYOUT,
          reader='dllm.pipelines.immune_llada.fixed_chain.layout.FixedChainCollator',
          first_chain_capacity=187, second_chain_capacity=154, a4_allowed=False,
          source_records='byte-identical retained V5 JSONL', synthetic_scaffold='preserved')
    manifest['rebuild'] = dict(scope=SCOPE, source=str(SOURCE), source_manifest_sha256=sha(SOURCE / 'dataset_manifest.json'),
          label_policy=LABEL_POLICY, counts=dict(total), registry_path=str(SCOPE_DIR / 'registry.json'),
          registry_sha256=sha(SCOPE_DIR / 'registry.json'), match_index_sha256=matching['index_sha256'],
          foundation_splits='original V5 split assignments and within-split multiplicities retained')
    write(stage / 'dataset_manifest.json', manifest)
    write(stage / 'cleaning_report.json', dict(status='verified', totals=dict(total), by_source=dict(by_source),
          source_shards=shard_results, cohort_bit_order=index['names'],
          evaluation_exclusions_sha256=sha(stage / 'evaluation_exclusions.jsonl'),
          binding=check_bindings(), matching=matching,
          limitations=json.loads((SCOPE_DIR / 'registry.json').read_text())['limitations']))
    write(stage / 'verification.json', dict(**verified, dataset_manifest_sha256=sha(stage / 'dataset_manifest.json'),
          cleaning_report_sha256=sha(stage / 'cleaning_report.json')))
    (stage / 'README.md').write_text(
        '# V5 → V7 no-negative design data\n\n'
        'Only explicitly nonbinding records are removed by label. Binding and unlabeled/unknown repertoire are retained for downstream-test filtering, without relabeling.\n\n'
        'Use the existing FixedChainCollator, 187/154 slots. A4 is not supported. '
        'Retained JSONL records, CDR3β scaffolds and synthetic masks are byte-identical to V5.\n\n'
        f'[Build and scope]({ROOT / "docs/V7_FROM_V5_REBUILD_20261015.md"})\n')
    stage.rename(OUTPUT)
    write(RUN / 'release.json', dict(status='complete', output=str(OUTPUT), counts=dict(total),
          manifest_sha256=sha(OUTPUT / 'dataset_manifest.json'), verification_sha256=sha(OUTPUT / 'verification.json')))
    print(json.dumps(dict(stage='release', output=str(OUTPUT), **dict(total))), flush=True)


def init_verifier(index_path):
    global _VERIFY_INDEX, _VERIFY_LOOKUP, _VERIFY_COLLATOR
    from dllm.pipelines.immune_llada.fixed_chain import FixedChainCollator, FixedChainConfig
    config = FixedChainConfig(**json.loads((ROOT / 'configs/models/v5_fixed_chain_f011_75m_20261012/immune_config.json').read_text()))
    _VERIFY_INDEX = read_pickle(index_path)
    _VERIFY_LOOKUP = upstream_lookup()
    _VERIFY_COLLATOR = FixedChainCollator(config)


def verify_shard(stage, shard):
    import numpy as np
    from dllm.pipelines.immune_llada.data.preprocessing.validators import validate_prepared_row
    index, lookup, collator = _VERIFY_INDEX, _VERIFY_LOOKUP, _VERIFY_COLLATOR
    counts, samples = Counter(), []
    states = np.load(stage / shard['selection_path'], allow_pickle=False)
    assert sha(stage / shard['selection_path']) == shard['selection_sha256']
    assert len(states) == shard['input_records']
    indices = np.flatnonzero(states == 0)
    parity = set(indices[:2]) | set(indices[-2:])
    before_hash, after_hash = hashlib.sha256(), hashlib.sha256()
    with (SOURCE / shard['path']).open('rb') as source, (stage / shard['path']).open('rb') as out:
        n, emitted = 0, 0
        for i, raw in enumerate(source):
            before_hash.update(raw)
            n += 1
            row = json.loads(raw)
            relation = row.get('labels', {}).get('relation', 'absent')
            assert relation in {'binding', 'nonbinding', 'unknown', 'absent'}, relation
            # Deliberately recompute the decision from the original row rather
            # than trusting candidate-cache states or keys.
            if relation == 'nonbinding':
                expected = 1
            else:
                keys, _ = audit_keys(row, lookup)
                expected = 3 if unresolved_tcr(keys) else 4 if reasons(keys, index) else 0
            assert states[i] == expected, (shard['path'], i, states[i], expected)
            if expected:
                continue
            actual = out.readline()
            assert actual == raw, (shard['path'], i, 'retained bytes differ')
            after_hash.update(actual)
            emitted += 1
            validate_prepared_row(row)
            counts['retained'] += 1
            counts[relation] += 1
            counts['forbidden_overlap'] += int(bool(reasons(keys, index)))
            for j, seq in enumerate(row['chains']):
                mask = row.get('chain_metadata', {}).get(str(j), {}).get('synthetic_residue_mask', [])
                assert not mask or len(mask) == len(seq)
                counts['synthetic_positions'] += sum(mask)
                counts['synthetic_chains'] += int(any(mask))
            if i in parity:
                before, after = collator([row]), collator([json.loads(actual)])
                assert before.keys() == after.keys()
                assert all((before[k] == after[k]).all().item() for k in before)
                samples.append(dict(shard=shard['path'], source_row=i, relation=relation,
                                    tokens=int(after['attention_mask'].sum()), all_features_equal=True))
        assert not out.read(1), 'Unexpected extra output'
    assert n == shard['input_records'] and emitted == shard['records']
    assert before_hash.hexdigest() == shard['input_sha256'] and after_hash.hexdigest() == shard['sha256']
    return dict(path=shard['path'], counts=dict(counts), samples=samples)


def verify_output(stage, shards, workers):
    counts = Counter({name: 0 for name in ('retained', 'binding', 'unknown', 'absent',
                                          'nonbinding', 'forbidden_overlap')})
    samples = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_verifier,
                             initargs=(RUN / 'match_index.pkl',)) as pool:
        futures = [pool.submit(verify_shard, stage, shard) for shard in shards]
        for future in as_completed(futures):
            result = future.result()
            counts.update(result['counts'])
            samples.extend(result['samples'])
            print(json.dumps(dict(stage='verify', shard=result['path'],
                                  verified=result['counts'].get('retained', 0))), flush=True)
    samples.sort(key=lambda x: (x['shard'], x['source_row']))
    assert counts['retained'] == sum(s['records'] for s in shards)
    assert counts['retained'] == counts['binding'] + counts['unknown'] + counts['absent']
    return dict(status='passed', scope=SCOPE, label_policy=LABEL_POLICY, **dict(counts),
                source_to_output_bytes='all retained records identical',
                selection_recomputed=True, collator_parity=samples, required_layout=LAYOUT,
                current_reader_sha256=sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/layout.py'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('candidates', 'match', 'export'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.stage == 'candidates':
        candidates(args.workers)
    elif args.stage == 'match':
        match(args.workers)
    else:
        export(args.workers)
