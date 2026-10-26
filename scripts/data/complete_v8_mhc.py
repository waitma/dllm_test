"""Complete source-proven MHC contexts on V8 negatives without changing labels.

Activate pllm; run USE_TF=0 python -B -m scripts.data.complete_v8_mhc --workers 4.
Then run verify_v7_mhc_completion with this staging directory and V8 parent;
use --publish only after the independent acceptance passes.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from scripts.data.complete_v7_mhc import (
    FASTA, B2M, PSEUDO, POLICY, b2m_reference, plan_candidate, checked_replacement, encoded)
from scripts.data.mhc_reference_resolution import HLAReference
from scripts.data.update_mhc_inputs import non_mhc
from scripts.data.rebuild_v7_from_v5 import sha, write
from scripts.data.prepare_immune_v8 import OUTPUT as PARENT
from dllm.pipelines.immune_llada.data.mhc import PseudoTable

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'output/mhc_completion_v8_20261026'
OUTPUT = ROOT / 'data/prepared/immune_v8_unbinding_slots167_135_mhc_completed_20261026'
STAGE = OUTPUT.with_name(OUTPUT.name + '.building')
CANDIDATES = WORK / 'allele_candidates.jsonl'


def check_negative_batch(before, after, tokenizer):
    """Protect receptor features and enforce relation-only targets after expansion."""
    import torch
    from dllm.pipelines.immune_llada.data.records import CHAIN_ROLE_TO_ID
    for role in ('tcr_beta', 'tcr_alpha', 'peptide'):
        old = before['role_ids'].eq(CHAIN_ROLE_TO_ID[role])
        new = after['role_ids'].eq(CHAIN_ROLE_TO_ID[role])
        for name in ('input_ids', 'labels', 'position_ids', 'target_slot_mask',
                     'eligible_mask', 'residue_mask', 'terminal_mask'):
            if not torch.equal(before[name][old], after[name][new]):
                raise ValueError('Non-MHC feature changed: ' + role + '/' + name)
    selected = after['labels'].ne(-100)
    if not torch.equal(selected, after['explicit_relation_mask']):
        raise ValueError('Negative sequence or grammar loss introduced')
    if not after['input_ids'][selected].eq(tokenizer.special_id('<nonbinding>')).any():
        raise ValueError('Lost nonbinding target')
    mhc = after['role_ids'].eq(CHAIN_ROLE_TO_ID['mhc'])
    if len(after['chain_ids'][mhc].unique()) != 2 or selected[mhc].any():
        raise ValueError('MHC components/loss differ')


def process_shard(job):
    shard, candidates, manifest_sha, candidate_sha = job
    import torch
    torch.set_num_threads(1)
    from dllm.pipelines.immune_llada.fixed_chain.unbinding_mlm import UnbindingMLMCollator, corrupt
    from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
    source, target = PARENT / shard['path'], STAGE / shard['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    if not candidates:
        if sha(source) != shard['sha256']:
            raise ValueError('Parent shard changed')
        target.symlink_to(source)
        return dict(path=shard['path'], records=shard['records'], changed=0, max_tokens=0,
                    sha256=shard['sha256'], counts=dict(records=shard['records'], changed=0), statuses={})
    collator = UnbindingMLMCollator(FixedChainConfig(**json.loads((PARENT / 'immune_config.json').read_text())))
    digest, counts, statuses, seen, max_tokens = hashlib.sha256(), Counter(), Counter(), set(), 0
    ledger_path = STAGE / 'audit' / (shard['path'].replace('/', '__') + '.jsonl')
    with source.open('rb') as src, target.open('wb') as out, ledger_path.open('wb') as ledger:
        for index, raw in enumerate(src):
            digest.update(raw)
            counts['records'] += 1
            row = json.loads(raw)
            if row['labels']['relation'] != 'nonbinding' or row['split'] != shard['split']:
                raise ValueError('Expected unchanged negative split/label')
            candidate, plan = candidates[index]
            seen.add(index)
            if (candidate['old_row_sha256'] != hashlib.sha256(raw).hexdigest()
                    or candidate['current_shard'] != shard['path'] or candidate['source'] != shard['source']):
                raise ValueError('Candidate does not match parent bytes/source')
            statuses[plan['status']] += 1
            audit = dict(current_row=index, status=plan['status'], allele_names=candidate['allele_names'],
                         parent_row_sha256=candidate['old_row_sha256'])
            if plan['accepted']:
                updated = checked_replacement(row, candidate, plan, manifest_sha, candidate_sha, output_dir=OUTPUT)
                before, after = collator([row]), collator([updated])
                check_negative_batch(before, after, collator.tokenizer)
                length = int(after['attention_mask'].sum())
                if length > collator.config.max_length:
                    raise ValueError('MHC completion exceeds model sequence budget')
                max_tokens = max(max_tokens, length)
                # Exercise actual corruption on each distinct context/role layout.
                if counts['changed'] == 0:
                    noisy = corrupt(after, collator.tokenizer, torch.Generator().manual_seed(42))
                    relation = after['explicit_relation_mask']
                    if (not torch.equal(noisy['labels'].ne(-100), relation)
                            or not torch.equal(noisy['input_ids'][~relation], after['input_ids'][~relation])):
                        raise ValueError('Negative corruption changed sequence context')
                counts['changed'] += 1
                counts['class_' + plan['mhc_class']] += 1
                counts['added_to_missing_mhc'] += not candidate['original_mhc']
                payload = encoded(updated)
                audit.update(output_row_sha256=hashlib.sha256(payload).hexdigest(), token_length=length,
                             mhc_class=plan['mhc_class'], mhc_lengths=[len(e['sequence']) for e in plan['entries']])
                out.write(payload)
            else:
                out.write(raw)
            ledger.write(encoded(audit))
    if seen != set(candidates) or counts['records'] != shard['records'] or digest.hexdigest() != shard['sha256']:
        raise ValueError('Parent/candidate coverage changed')
    return dict(path=shard['path'], records=counts['records'], changed=counts['changed'], max_tokens=max_tokens,
                sha256=sha(target), counts=dict(counts), statuses=dict(statuses),
                audit_path=str(ledger_path.relative_to(STAGE)), audit_sha256=sha(ledger_path))


def build(workers):
    if OUTPUT.exists() or STAGE.exists():
        raise FileExistsError('Refusing to overwrite existing MHC release/build')
    manifest_path = PARENT / 'dataset_manifest.json'
    manifest_sha = sha(manifest_path)
    parent = json.loads(manifest_path.read_text())
    for name in ('verification.json', 'reader_acceptance.json'):
        check = json.loads((PARENT / name).read_text())
        if check['status'] != 'passed' or check['dataset_manifest_sha256'] != manifest_sha:
            raise ValueError('V8 parent acceptance changed')
    audit_path = WORK / 'allele_candidates_audit.json'
    audit = json.loads(audit_path.read_text())
    if audit['parent_manifest_sha256'] != manifest_sha or audit['output_sha256'] != sha(CANDIDATES):
        raise ValueError('Allele candidate provenance changed')
    pins = {**audit['source_files'], str(audit_path): sha(audit_path), str(CANDIDATES): sha(CANDIDATES),
            str(Path(audit['source_report'])): audit['source_report_sha256']}
    overrides = {}
    override_path = WORK / 'trait_classII_candidates.jsonl'
    override_audit_path = WORK / 'trait_classII_candidates_audit.json'
    if override_path.exists():
        evidence = json.loads(override_audit_path.read_text())
        if evidence['status'] != 'complete' or evidence['output_sha256'] != sha(override_path):
            raise ValueError('Class II evidence changed')
        pins.update(evidence['inputs'])
        pins.update({str(p): sha(p) for p in (override_path, override_audit_path)})
        for line in override_path.open():
            item = json.loads(line)
            overrides[(item['current_shard'], item['current_row'])] = item
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Bound input changed: ' + path)
    (STAGE / 'provenance').mkdir(parents=True)
    (STAGE / 'audit').mkdir()
    shutil.copyfile(PARENT / 'immune_config.json', STAGE / 'immune_config.json')
    for path in (FASTA, B2M, PSEUDO):
        shutil.copyfile(path, STAGE / 'provenance' / path.name)
        pins[str(path)] = sha(path)
    ledger_path = STAGE / 'provenance/allele_candidates.jsonl'
    reference, b2m, pseudo = HLAReference.from_fasta(FASTA), b2m_reference(), PseudoTable(PSEUDO)
    by_shard, cache, statuses = defaultdict(dict), {}, Counter()
    with ledger_path.open('wb') as out:
        for line in CANDIDATES.open():
            item = json.loads(line)
            override = overrides.pop((item['current_shard'], item['current_row']), None)
            if override:
                if any(override[k] != item[k] for k in ('identity', 'split', 'old_row_sha256')):
                    raise ValueError('Class II evidence identity differs')
                item['base_allele_names'] = item['allele_names']
                item['allele_names'], item['evidence'] = override['allele_names'], override['evidence']
            plan = plan_candidate(item, reference, b2m, pseudo, cache)
            by_shard[item['current_shard']][item['current_row']] = (item, plan)
            statuses[plan['status']] += 1
            out.write(encoded(item))
    if overrides:
        raise ValueError('Unused Class II evidence')
    candidate_sha = sha(ledger_path)
    write(STAGE / 'provenance/allele_resolutions.json', cache)
    print(json.dumps(dict(stage='planned', candidates=sum(statuses.values()), statuses=dict(statuses))), flush=True)
    shards = [s for block in parent['splits'].values() for s in block['shards']]
    if not set(by_shard) <= {s['path'] for s in shards}:
        raise ValueError('Unknown candidate shards')
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(process_shard, [(s, by_shard.get(s['path'], {}), manifest_sha, candidate_sha) for s in shards]):
            results.append(result)
            if result['statuses']:
                print(json.dumps(result), flush=True)
    by_path, totals, splits = {r['path']: r for r in results}, Counter(), {}
    manifest = deepcopy(parent)
    manifest.update(created_at=datetime.now(timezone.utc).isoformat(), dataset_version=OUTPUT.name)
    for split, block in manifest['splits'].items():
        counts = Counter()
        for shard in block['shards']:
            result = by_path[shard['path']]
            shard.update(parent_sha256=shard['sha256'], sha256=result['sha256'], mhc_completed=result['changed'])
            counts.update(result['counts'])
        splits[split] = dict(counts)
        totals.update(counts)
    for path in (Path(__file__), ROOT / 'scripts/data/complete_v7_mhc.py',
                 ROOT / 'scripts/data/mhc_reference_resolution.py', ROOT / 'scripts/data/collect_v7_mhc_alleles.py',
                 ROOT / 'scripts/data/recover_trait_mhcII_evidence.py'):
        pins[str(path)] = sha(path)
    manifest['mhc_completion'] = dict(policy=POLICY, parent=str(PARENT), parent_manifest_sha256=manifest_sha,
        counts=dict(totals), splits=splits, candidate_statuses=dict(statuses), sources=pins,
        candidate_ledger_sha256=candidate_sha,
        scope='Only restored V8 TCR nonbinding rows; source-proven reference MHC, never reverse-infer from pseudo',
        invariants='same records/order/splits/receptor/peptide/labels/weights/synthetic masks')
    write(STAGE / 'dataset_manifest.json', manifest)
    evidence = dict(status='passed', dataset_manifest_sha256=sha(STAGE / 'dataset_manifest.json'),
        parent_manifest_sha256=manifest_sha, records={s: b['records'] for s, b in manifest['splits'].items()},
        counts=dict(totals), candidate_statuses=dict(statuses),
        all_changed_rows_actual_unbinding_collator=True, negative_sequence_targets=0,
        max_changed_input_tokens=max(r['max_tokens'] for r in results),
        config_sha256=sha(STAGE / 'immune_config.json'),
        collator_sha256=sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/layout.py'),
        objective_sha256=sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/unbinding_mlm.py'))
    write(STAGE / 'verification.json', evidence)
    write(STAGE / 'reader_acceptance.json', evidence)
    write(STAGE / 'cleaning_report.json', dict(status='passed', counts=dict(totals), shards=results))
    print(json.dumps(dict(stage='built', **evidence)), flush=True)


def publish():
    if OUTPUT.exists() or not STAGE.exists():
        raise FileExistsError('Expected new staging release')
    digest = sha(STAGE / 'dataset_manifest.json')
    for name in ('verification.json', 'reader_acceptance.json', 'independent_acceptance.json'):
        check = json.loads((STAGE / name).read_text())
        if check['status'] != 'passed' or check['dataset_manifest_sha256'] != digest:
            raise ValueError('Missing/stale acceptance: ' + name)
    STAGE.rename(OUTPUT)
    write(WORK / 'release.json', dict(status='published', output=str(OUTPUT), manifest_sha256=digest,
                                    independent_acceptance_sha256=sha(OUTPUT / 'independent_acceptance.json')))
    print(json.dumps(dict(status='published', output=str(OUTPUT), manifest_sha256=digest)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    publish() if args.publish else build(args.workers)
