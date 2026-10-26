"""Complete traceable MHC contexts without changing receptor records or splits.

Activate pllm, then run USE_TF=0 python -B scripts/data/complete_v7_mhc.py --workers 4.
Validate the .building directory with check_v7_no_negative_design_reader.py and an
independent audit before running this script with --publish. No model is loaded.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.rebuild_v7_from_v5 import sha, write
from scripts.data.mhc_reference_resolution import HLAReference, parse_allele
from scripts.data.update_mhc_inputs import identity, non_mhc, replace_mhc
from dllm.pipelines.immune_llada.data.mhc import PseudoTable

PARENT = ROOT / 'data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_20261015'
OUTPUT = ROOT / 'data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_mhc_completed_20261015'
STAGE = OUTPUT.with_name(OUTPUT.name + '.building')
WORK = ROOT / 'output/mhc_completion_20261015'
CANDIDATES = WORK / 'allele_candidates.jsonl'
CLASS_II = WORK / 'trait_classII_candidates.jsonl'
FASTA = ROOT / 'data/tcr_pmhc_fulllength/imgt_hla/hla_prot.fasta'
B2M = ROOT / 'data/reference/mhc_completion_20261015/b2m_P61769.json'
PSEUDO = ROOT / 'data/mhc_sequence_v1/reference/MHC_pseudo.dat'
POLICY = 'v7_mhc_reference_completion_coverage_v1'
GROUPS = {'trait', 'tcr_native', 'tcr_papers'}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()


def b2m_reference():
    """Use the annotated mature human chain and cross-check the existing reference."""
    entry = json.loads(B2M.read_text())
    assert entry['primaryAccession'] == 'P61769'
    assert entry['organism']['taxonId'] == 9606
    chains = [f for f in entry['features'] if f['type'] == 'Chain'
              and f.get('description') == 'Beta-2-microglobulin']
    assert len(chains) == 1
    start = chains[0]['location']['start']['value']
    end = chains[0]['location']['end']['value']
    assert (start, end) == (21, 119)
    sequence = entry['sequence']['value'][start - 1:end]
    source = ROOT / 'scripts/data/build_fulllength_tcr_pmhc.py'
    assignments = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'B2M_MATURE' for t in n.targets)]
    assert len(assignments) == 1 and ast.literal_eval(assignments[0].value) == sequence
    return dict(sequence=sequence, metadata=dict(mhc_component='b2m',
        mhc_representation='reference_sequence', mhc_origin='UniProt:P61769',
        sequence_scope='mature', reference_range_1based=[start, end],
        reference_sha256=sha(B2M), reference_derived=True, mhc_policy=POLICY))


def resolve_context(allele_name, reference, b2m):
    """Require an explicit class-I allele or both members of a class-II pair."""
    parts = [x.strip() for x in re.split(r'__|[+/;,|]', allele_name) if x.strip()]
    if not parts:
        return dict(accepted=False, status='missing_allele')
    results = [reference.resolve(part) for part in parts]
    try:
        genes = [parse_allele(part).gene for part in parts]
    except ValueError:
        return dict(accepted=False, status='invalid_allele', resolutions=results)
    if len(parts) == 1 and genes[0] in {'A', 'B', 'C', 'E'}:
        components, mhc_class, bounds = ['mhc_heavy'], 'I', (330, 400)
    elif len(parts) == 1 and genes[0].startswith('D'):
        return dict(accepted=False, status='missing_explicit_classII_partner', resolutions=results)
    elif len(parts) == 2:
        pairs = {('DPA1', 'DPB1'), ('DQA1', 'DQB1')}
        pairs.update(('DRA', beta) for beta in ('DRB1', 'DRB3', 'DRB4', 'DRB5'))
        if tuple(genes) not in pairs:
            if tuple(reversed(genes)) in pairs:
                results.reverse()
            else:
                return dict(accepted=False, status='invalid_classII_gene_pair', resolutions=results)
        components, mhc_class, bounds = ['mhc_alpha', 'mhc_beta'], 'II', (230, 310)
    else:
        return dict(accepted=False, status='unsupported_allele_components', resolutions=results)
    failures = sorted({r['status'] for r in results if not r['accepted']})
    if failures:
        return dict(accepted=False, status='+'.join(failures), resolutions=results)
    if any(r['expression_suffix'] in {'N', 'Q'} for r in results):
        return dict(accepted=False, status='null_or_questionable_expression', resolutions=results)
    if any(not r['starts_with_methionine'] or not bounds[0] <= r['sequence_length'] <= bounds[1]
           for r in results):
        return dict(accepted=False, status='reference_scope_not_accepted', resolutions=results)
    entries = []
    for component, result in zip(components, results):
        entries.append(dict(sequence=result['sequence'], metadata=dict(
            mhc_component=component, mhc_class=mhc_class,
            mhc_representation='reference_sequence', mhc_origin='IPD-IMGT/HLA',
            mhc_allele=result['normalized_allele'], sequence_scope='reference_longest',
            scope_note='Unique longest reference covers every matching entry; not a measured sample sequence.',
            reference_derived=True, reference_sha256=reference.reference_sha256,
            resolution_status=result['status'], mhc_policy=POLICY)))
    if mhc_class == 'I':
        entries.append(b2m)
    return dict(accepted=True, status='resolved', entries=entries, mhc_class=mhc_class,
                resolutions=results, normalized_alleles=[r['normalized_allele'] for r in results])


def plan_candidate(candidate, reference, b2m, pseudo, cache):
    names = candidate['allele_names']
    if not names or not candidate['evidence']:
        return dict(accepted=False, status='no_source_allele_evidence')
    if any(not n.strip() for n in names):
        return dict(accepted=False, status='missing_allele')
    resolved = []
    for name in names:
        if name not in cache:
            cache[name] = resolve_context(name, reference, b2m)
        resolved.append(cache[name])
    failures = sorted({r['status'] for r in resolved if not r['accepted']})
    if failures:
        return dict(accepted=False, status='+'.join(failures))
    signatures = {tuple(e['sequence'] for e in r['entries']) for r in resolved}
    if len(signatures) != 1:
        return dict(accepted=False, status='conflicting_source_allele_sequences')
    # A supplied pseudo sequence must agree with the independently pinned pseudo
    # table when that allele is present there. It is never reverse-mapped to an allele.
    if resolved[0]['mhc_class'] == 'I' and candidate['original_mhc']:
        for result in resolved:
            expected, pseudo_status = pseudo.lookup(result['normalized_alleles'][0])
            if pseudo_status == 'conflicting_reference':
                return dict(accepted=False, status='source_pseudo_reference_conflict')
            if expected and candidate['original_mhc'] != [expected]:
                return dict(accepted=False, status='source_pseudo_allele_mismatch')
    return dict(accepted=True, status='accepted', entries=resolved[0]['entries'],
        mhc_class=resolved[0]['mhc_class'], normalized_alleles=sorted({a for r in resolved
            for a in r['normalized_alleles']}))


def checked_replacement(row, candidate, plan, manifest_sha, candidate_sha, *, output_dir=OUTPUT):
    """Re-anchor evidence to the original record; protect all non-MHC content."""
    assert identity(row, candidate['source']) == candidate['identity']
    assert non_mhc(row)[1] == candidate['split']
    assert [s for s, r in zip(row['chains'], row['chain_roles']) if r == 'mhc'] == candidate['original_mhc']
    info = dict(policy=POLICY, status='reference_context_completed',
        allele_candidates=candidate['allele_names'], normalized_alleles=plan['normalized_alleles'],
        mhc_class=plan['mhc_class'], original_mhc=candidate['original_mhc'],
        parent_manifest_sha256=manifest_sha, parent_shard=candidate['current_shard'],
        parent_row=candidate['current_row'], parent_row_sha256=candidate['old_row_sha256'],
        evidence_sha256=hashlib.sha256(encoded(candidate['evidence'])).hexdigest(),
        candidate_ledger_sha256=candidate_sha,
        evidence_ledger=str(output_dir / 'provenance/allele_candidates.jsonl'))
    result = replace_mhc(row, plan['entries'], info)
    assert non_mhc(result) == non_mhc(row)
    assert set(result) - set(row) <= {'chain_metadata', 'regions'}
    for key in set(row) - {'chains', 'chain_roles', 'chain_metadata', 'regions', 'metadata', 'task_type'}:
        assert result[key] == row[key], key
    return result


def process_shard(args):
    """Validate every altered record through the actual reader and collator."""
    shard, candidates, manifest_sha, candidate_sha = args
    target = STAGE / shard['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    source = PARENT / shard['path']
    if not candidates:
        assert sha(source) == shard['sha256']
        target.symlink_to(os.path.relpath(source, target.parent))
        return dict(path=shard['path'], records=shard['records'], changed=0,
                    sha256=shard['sha256'], max_tokens=0, labels={}, statuses={})
    import torch
    torch.set_num_threads(1)
    from dllm.pipelines.immune_llada.fixed_chain import FixedChainConfig, FixedChainCollator
    from dllm.pipelines.immune_llada.data.records import CHAIN_ROLE_TO_ID
    config = FixedChainConfig(**json.loads((STAGE / 'immune_config.json').read_text()))
    collator = FixedChainCollator(config)
    counts, labels, statuses, length_counts = Counter(), Counter(), Counter(), Counter()
    digest = hashlib.sha256()
    ledger_path = STAGE / 'audit' / (shard['path'].replace('/', '__') + '.jsonl')
    seen = set()
    with source.open('rb') as src, target.open('wb') as out, ledger_path.open('wb') as ledger:
        for index, raw in enumerate(src):
            digest.update(raw)
            row = json.loads(raw)
            counts['records'] += 1
            labels[row.get('labels', {}).get('relation', 'absent')] += 1
            assert row.get('split') == shard['split']
            if index not in candidates:
                out.write(raw)
                continue
            seen.add(index)
            candidate, plan = candidates[index]
            assert candidate['current_shard'] == shard['path'] and candidate['source'] == shard['source']
            assert candidate['old_row_sha256'] == hashlib.sha256(raw).hexdigest()
            assert candidate['split'] == shard['split']
            statuses[plan['status']] += 1
            audit = dict(current_row=index, parent_row_sha256=candidate['old_row_sha256'],
                         status=plan['status'], allele_names=candidate['allele_names'])
            if not plan['accepted']:
                out.write(raw)
            else:
                updated = checked_replacement(row, candidate, plan, manifest_sha, candidate_sha)
                before, after = collator([row]), collator([updated])
                for role in ('tcr_beta', 'tcr_alpha'):
                    old_mask = before['role_ids'] == CHAIN_ROLE_TO_ID[role]
                    new_mask = after['role_ids'] == CHAIN_ROLE_TO_ID[role]
                    for feature in ('input_ids', 'labels', 'position_ids', 'target_slot_mask',
                                    'eligible_mask', 'residue_mask', 'terminal_mask'):
                        assert torch.equal(before[feature][old_mask], after[feature][new_mask]), feature
                mhc = after['role_ids'] == CHAIN_ROLE_TO_ID['mhc']
                assert len(after['chain_ids'][mhc].unique()) == 2
                assert not after['eligible_mask'][mhc].any()
                assert not after['target_slot_mask'][mhc].any()
                assert after['labels'][mhc].eq(-100).all()
                length = int(after['attention_mask'].sum())
                assert length <= config.max_length
                length_counts[length] += 1
                counts['changed'] += 1
                counts['class_' + plan['mhc_class']] += 1
                counts['added_to_missing_mhc'] += not candidate['original_mhc']
                payload = encoded(updated)
                audit.update(output_row_sha256=hashlib.sha256(payload).hexdigest(),
                             mhc_class=plan['mhc_class'], token_length=length,
                             mhc_lengths=[len(e['sequence']) for e in plan['entries']])
                out.write(payload)
            ledger.write(encoded(audit))
    assert seen == set(candidates)
    assert counts['records'] == shard['records'] and digest.hexdigest() == shard['sha256']
    # Second read: unchanged records must be byte-identical, changed records must
    # preserve all target/label/split content, and there must be no row addition.
    with source.open('rb') as src, target.open('rb') as out:
        for index, raw in enumerate(src):
            new_raw = out.readline()
            assert new_raw
            if index in candidates and candidates[index][1]['accepted']:
                assert non_mhc(json.loads(raw)) == non_mhc(json.loads(new_raw))
            else:
                assert raw == new_raw
        assert not out.read(1)
    return dict(path=shard['path'], records=counts['records'], changed=counts['changed'],
        counts=dict(counts), sha256=sha(target), parent_sha256=digest.hexdigest(),
        max_tokens=max(length_counts, default=0), labels=dict(labels), statuses=dict(statuses),
        audit_path=str(ledger_path.relative_to(STAGE)), audit_sha256=sha(ledger_path))


def build(workers):
    assert not OUTPUT.exists() and not STAGE.exists(), 'Refusing to overwrite a release/build'
    parent_path = PARENT / 'dataset_manifest.json'
    manifest_sha = sha(parent_path)
    parent = json.loads(parent_path.read_text())
    for name in ('verification.json', 'reader_acceptance.json'):
        check = json.loads((PARENT / name).read_text())
        assert check['status'] == 'passed' and check['dataset_manifest_sha256'] == manifest_sha
    reference, b2m, pseudo = HLAReference.from_fasta(FASTA), b2m_reference(), PseudoTable(PSEUDO)
    source_audit = json.loads((WORK / 'allele_candidates_audit.json').read_text())
    assert source_audit['output_sha256'] == sha(CANDIDATES)
    assert source_audit['parent_manifest_sha256'] == manifest_sha
    assert source_audit['source_report_sha256'] == sha(source_audit['source_report'])
    for path, digest in source_audit['source_files'].items():
        assert sha(path) == digest, path
    overrides = {}
    if CLASS_II.exists():
        override_audit = json.loads((WORK / 'trait_classII_candidates_audit.json').read_text())
        assert override_audit['status'] == 'complete'
        assert override_audit['output_sha256'] == sha(CLASS_II)
        for path, digest in override_audit['inputs'].items():
            assert sha(path) == digest, path
        for line in CLASS_II.open():
            item = json.loads(line)
            key = (item['current_shard'], item['current_row'])
            assert key not in overrides
            overrides[key] = item
    merged = []
    for line in CANDIDATES.open():
        item = json.loads(line)
        key = (item['current_shard'], item['current_row'])
        override = overrides.pop(key, None)
        if override is not None and override.get('accepted', True):
            assert override['old_row_sha256'] == item['old_row_sha256']
            assert override['split'] == item['split']
            assert override['identity'] == item['identity']
            item['base_allele_names'] = item['allele_names']
            item['base_evidence'] = item['evidence']
            item['allele_names'] = override['allele_names']
            item['evidence'] = override['evidence']
        merged.append(item)
    assert not overrides
    (STAGE / 'provenance').mkdir(parents=True)
    (STAGE / 'audit').mkdir()
    shutil.copyfile(PARENT / 'immune_config.json', STAGE / 'immune_config.json')
    # Parent capacity-selection artifacts remain traceable under their original paths.
    (STAGE / 'selection').symlink_to(os.path.relpath(PARENT / 'selection', STAGE))
    shutil.copyfile(PARENT / 'capacity_exclusions.jsonl', STAGE / 'capacity_exclusions.jsonl')
    for path in (FASTA, B2M, PSEUDO, CANDIDATES, WORK / 'allele_candidates_audit.json', CLASS_II,
                 WORK / 'trait_classII_candidates_audit.json'):
        if path.exists():
            shutil.copyfile(path, STAGE / 'provenance' / ('base_' + path.name if path == CANDIDATES else path.name))
    merged_path = STAGE / 'provenance/allele_candidates.jsonl'
    with merged_path.open('wb') as out:
        for item in merged:
            out.write(encoded(item))
    candidate_sha = sha(merged_path)
    by_shard, cache, statuses = defaultdict(dict), {}, Counter()
    for item in merged:
        plan = plan_candidate(item, reference, b2m, pseudo, cache)
        key = item['current_row']
        assert key not in by_shard[item['current_shard']]
        by_shard[item['current_shard']][key] = (item, plan)
        statuses[plan['status']] += 1
    write(STAGE / 'provenance/allele_resolutions.json', cache)
    print(json.dumps(dict(phase='planned', candidates=len(merged), statuses=dict(statuses))), flush=True)
    shards = [s for block in parent['splits'].values() for s in block['shards']]
    assert set(by_shard) <= {s['path'] for s in shards}
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(process_shard, [(s, by_shard.get(s['path'], {}), manifest_sha,
                                               candidate_sha) for s in shards]):
            results.append(result)
            if result['changed'] or result['statuses']:
                print(json.dumps(result), flush=True)
    assert sha(parent_path) == manifest_sha
    by_path = {r['path']: r for r in results}
    manifest = deepcopy(parent)
    manifest.update(created_at=datetime.now(timezone.utc).isoformat(), dataset_version=OUTPUT.name)
    totals, splits = Counter(), {}
    for split, block in manifest['splits'].items():
        split_counts = Counter()
        for shard in block['shards']:
            result = by_path[shard['path']]
            shard['parent_sha256'] = shard['sha256']
            shard['sha256'] = result['sha256']
            shard['mhc_completed'] = result['changed']
            split_counts.update(result.get('counts', dict(records=result['records'], changed=0)))
        assert split_counts['records'] == block['records']
        splits[split] = dict(split_counts)
        totals.update(split_counts)
    sources = {str(path): sha(path) for path in (FASTA, B2M, PSEUDO, CANDIDATES,
        WORK / 'allele_candidates_audit.json', Path(__file__),
        ROOT / 'scripts/data/mhc_reference_resolution.py', ROOT / 'scripts/data/collect_v7_mhc_alleles.py')}
    if CLASS_II.exists():
        sources[str(CLASS_II)] = sha(CLASS_II)
        for path in (WORK / 'trait_classII_candidates_audit.json',
                     ROOT / 'scripts/data/recover_trait_mhcII_evidence.py'):
            sources[str(path)] = sha(path)
    # Include the raw source pins rather than relying only on a path to their audit.
    sources.update(source_audit['source_files'])
    if CLASS_II.exists():
        sources.update(override_audit['inputs'])
    manifest['mhc_completion'] = dict(policy=POLICY, parent=str(PARENT),
        parent_manifest_sha256=manifest_sha, counts=dict(totals), splits=splits,
        candidate_statuses=dict(statuses), sources=sources,
        candidate_ledger_sha256=candidate_sha,
        invariants='same rows, order, splits, receptor/peptide, labels, identifiers, weights and synthetic masks',
        unchanged_shards='relative symlinks to hash-verified immutable parent',
        sequence_note='Reference-derived MHC; longest HLA with exact coverage consistency and explicit allele resolution; mature B2M.')
    write(STAGE / 'dataset_manifest.json', manifest)
    write(STAGE / 'cleaning_report.json', dict(status='passed', policy=POLICY,
        parent=str(PARENT), counts=dict(totals), splits=splits, statuses=dict(statuses), shards=results))
    write(STAGE / 'verification.json', dict(status='passed', retained=totals['records'],
        changed=totals['changed'], nonbinding=0, capacity_overflow=0,
        all_changed_rows_actual_collator=True, unchanged_rows_byte_identical=True,
        all_shards_parent_hash_verified=True, max_changed_input_tokens=max(r['max_tokens'] for r in results),
        test_isolation='Inherited exact same receptor/peptide records and splits; MHC excluded from protected matching keys.',
        dataset_manifest_sha256=sha(STAGE / 'dataset_manifest.json'),
        cleaning_report_sha256=sha(STAGE / 'cleaning_report.json')))
    (STAGE / 'README.md').write_text(
        '# V7 with reference-derived MHC completion\n\n'
        f'Parent: {PARENT}\n\n'
        'Use the local immune_config.json: H/beta=167, L/alpha=135 (including EOS), max_length=2048.\n'
        'Rows, splits, receptor/peptide sequences, labels and synthetic masks are preserved. '
        'Unresolved contexts remain byte-identical. HLA chains are explicitly reference-derived; '
        'B2M is the annotated mature 99-aa human sequence. See provenance and audit ledgers.\n\n'
        'Unchanged shards link to the immutable parent; retain both directories. '
        'Existing training jobs are not migrated.\n')
    print(json.dumps(dict(stage=str(STAGE), counts=dict(totals), splits=splits)), flush=True)


def publish():
    assert not OUTPUT.exists() and STAGE.exists()
    manifest_sha = sha(STAGE / 'dataset_manifest.json')
    for name in ('verification.json', 'reader_acceptance.json', 'independent_acceptance.json'):
        check = json.loads((STAGE / name).read_text())
        assert check['status'] == 'passed' and check['dataset_manifest_sha256'] == manifest_sha, name
    STAGE.rename(OUTPUT)
    print(json.dumps(dict(status='published', output=str(OUTPUT), dataset_manifest_sha256=manifest_sha)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    publish() if args.publish else build(args.workers)
