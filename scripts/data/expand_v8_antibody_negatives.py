"""Add traceable antibody SPR negatives to an immutable child of V8.

Activate conda pllm; from /project_c20250601/251105016/project/dllm_test run:
  USE_TF=0 python -B -m scripts.data.expand_v8_antibody_negatives --stage all
Source CSVs and RCSB reference FASTAs are pinned by download_manifest.json.
The released HCDRs are restored on the study's fixed VH/VL framework, checked
against every corresponding full-Fv record in the existing FLAb snapshot.
No sequence generation or affinity-based relabelling is performed. VHH data
are inventoried separately because the current fixed-chain reader excludes VHH.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

from scripts.data import export_v7_no_negative_design as filters
from scripts.data.export_v7_binding_design import bank_index, reasons, save_pickle, read_pickle
from scripts.data.rebuild_v7_from_v5 import keys_for_row, sha, write
from scripts.data.v7_rebuild_scope import SCOPE_DIR, verify as verify_scope
from scripts.data.prepare_immune_v8 import held_probe_keys
from scripts.research.bert_mlm_data import receptor_key

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/antibody_negatives/igdesign_20261026'
FLAB = ROOT / 'data/ppi_task_raw/raw/flab/FLAb/data/binding'
PARENT = ROOT / 'data/prepared/immune_v8_unbinding_slots167_135_mhc_completed_20261026'
OUTPUT = ROOT / 'data/prepared/immune_v8_unbinding_slots167_135_mhc_ab_expanded_20261026'
WORK = ROOT / 'output/antibody_negative_expansion_20261026'
CDRS = ('HCDR1', 'HCDR2', 'HCDR3')
AA = set('ACDEFGHIKLMNPQRSTVWY')
TARGETS = {
    # 6PPG is IL17F, despite being associated with this cohort elsewhere.
    # Map the explicitly stated assay target IL17A to its UniProt reference.
    'Afasevikumab-IL17A': ('Q16552', 'interleukin-17a'),
    'Bimagrumab-ACVR2B': ('5NGV', 'activin receptor type-2b'),
    'Eculizumab-C5': ('5I5K', 'complement c5'),
    'Osocimab-FXI': ('6HHC', 'coagulation factor xi'),
    'Spesolimab-IL36R': ('6U6U', 'interleukin-1 receptor-like 2'),
    'Tezepelumab-TSLP': ('5J13', 'thymic stromal lymphopoietin'),
    'Utomilumab-TNFRSF9': ('6A3W', 'tumor necrosis factor receptor superfamily member 9'),
}


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def strict_relation(row):
    """Separate consistently observed outcomes from mixed/absent measurements."""
    try:
        total, bound = float(row['Replicates']), float(row['Binding Replicates'])
        ratio = float(row['Ratio Binding Replicates'])
    except (KeyError, ValueError, TypeError):
        return None, 'invalid_replicates'
    if (not all(math.isfinite(v) for v in (total, bound, ratio)) or total < 2
            or total != int(total) or bound != int(bound) or not 0 <= bound <= total
            or not math.isclose(ratio, bound / total, rel_tol=1e-7, abs_tol=1e-9)):
        return None, 'invalid_replicates'
    binder = str(row.get('Binder', '')).strip().lower()
    if binder not in ('true', 'false'):
        return None, 'invalid_binder'
    if bound == 0 and binder == 'false' and not str(row.get('KD (nM)', '')).strip():
        return 'nonbinding', 'zero_binding_replicates'
    if bound == total and binder == 'true':
        try:
            kd = float(row['KD (nM)'])
        except (KeyError, ValueError, TypeError):
            return None, 'invalid_positive_affinity'
        if math.isfinite(kd) and kd > 0:
            return 'binding', 'all_replicates_binding'
    if 0 < bound < total:
        return None, 'discordant_replicates'
    return None, 'inconsistent_source_label'


def framework_template(rows):
    """Require one fixed framework and paired light chain across the study."""
    templates, lights = set(), set()
    for row in rows:
        heavy = row['heavy']
        if not heavy or not row['light'] or not set(heavy + row['light']) <= AA:
            raise ValueError('Invalid FLAb full-Fv reference')
        for name in CDRS:
            cdr = row[name]
            if not cdr or not set(cdr) <= AA or heavy.count(cdr) != 1:
                raise ValueError('Ambiguous CDR/framework mapping')
            heavy = heavy.replace(cdr, '{' + name + '}')
        templates.add(heavy)
        lights.add(row['light'])
    if len(templates) != 1 or len(lights) != 1:
        raise ValueError('Study does not have a unique fixed VH/VL framework')
    return next(iter(templates)), next(iter(lights))


def antigen_reference(path, description):
    """Use the named RCSB entity; record that this is a reference construct."""
    matches = []
    for block in Path(path).read_text().split('>')[1:]:
        header, *lines = block.splitlines()
        if description in header.split('|')[2].strip().lower():
            sequence = ''.join(''.join(lines).split()).upper()
            if not sequence or not set(sequence) <= AA:
                raise ValueError('Noncanonical reference antigen')
            matches.append((header, sequence))
    if len(matches) != 1:
        raise ValueError(f'Ambiguous reference antigen in {path}: {description}')
    return matches[0]


def candidate_rows():
    """Reconstruct only published measured constructs and retain row lineage."""
    from dllm.pipelines.immune_llada.data.records import BioSeqChain, BioSeqRecord
    from dllm.pipelines.immune_llada.data.sources import _h3_core_from_chain
    from dllm.pipelines.immune_llada.data.preprocessing.validators import add_schema_version
    manifest_path = RAW / 'download_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    files = {item['path']: item['sha256'] for item in manifest['files']}
    files[str(manifest_path)] = sha(manifest_path)
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError('Pinned raw input changed: ' + path)
    rows, audit, counts = [], [], defaultdict(Counter)
    for target, (pdb, description) in TARGETS.items():
        source = RAW / (target + '.csv')
        reference = FLAB / ('shanehsazzadeh2024igdesign_' + target + '_kd.csv')
        files[str(reference)] = sha(reference)
        originals, full = read_csv(source), read_csv(reference)
        template, light = framework_template(full)
        original_keys = {tuple(r[k] for k in CDRS) for r in originals}
        for row in full:
            if (tuple(row[k] for k in CDRS) not in original_keys
                    or template.format(**row) != row['heavy'] or light != row['light']):
                raise ValueError('Full-Fv reference does not match author-released CDRs')
        header, antigen = antigen_reference(RAW / (pdb + '.fasta'), description)
        for number, raw in enumerate(originals, 2):
            counts[target]['input'] += 1
            relation, decision = strict_relation(raw)
            evidence = dict(source=str(source), source_sha256=files[str(source)],
                source_row=number, target=raw['Target'], reference_antibody=raw['Reference Ab'],
                source_binder=raw['Binder'], binding_replicates=raw['Binding Replicates'],
                replicates=raw['Replicates'], kd_nm=raw['KD (nM)'], decision=decision)
            audit.append(evidence)
            counts[target][decision] += 1
            if relation is None:
                continue
            if any(not raw[k] or not set(raw[k]) <= AA for k in CDRS):
                raise ValueError('Unreleased/noncanonical CDR sequence')
            heavy = template.format(**raw)
            h3 = _h3_core_from_chain(heavy)
            if not h3:
                evidence['sequence_decision'] = 'quarantined_unresolved_h3_audit_key'
                counts[target]['quarantined_unresolved_h3_audit_key'] += 1
                continue
            record = BioSeqRecord(
                [BioSeqChain(antigen, 'antigen'), BioSeqChain(heavy, 'antibody_heavy'),
                 BioSeqChain(light, 'antibody_light')],
                'antibody_antigen', 'igdesign_antibody', 'train',
                metadata=dict(relation_supervised=True, evidence_kind='SPR_replicate_consistent',
                    source_dataset='IgDesign', source_revision=manifest['revision'],
                    measurement=evidence, framework_reference=str(reference),
                    framework_reference_sha256=files[str(reference)],
                    antibody_sequence_provenance='released_HCDRs_on_study_fixed_VH_VL_framework',
                    framework_verified_full_fv_rows=len(full),
                    antigen_reference_id=pdb, antigen_reference_entity=header,
                    antigen_reference_database='UniProt' if pdb == 'Q16552' else 'RCSB',
                    antigen_sequence_provenance='reference_sequence_not_assay_construct_confirmation'),
                labels=dict(relation=relation),
                identifiers=dict(heavy_sequence=heavy, heavy_h3_core=h3,
                    benchmark_heavy_key='H:' + heavy, benchmark_h3_key='h3:' + h3))
            rows.append(add_schema_version(record.to_dict()))
    return rows, audit, dict(files), {k: dict(v) for k, v in counts.items()}


def audit_exported_source_rows(rows):
    """Independently check exported chains against raw measured CDRs and references.

    Use coordinate splicing in one published full-Fv reference instead of the
    template replacement used by the exporter. Re-read labels and raw FASTA.
    """
    checked = 0
    cache = {}
    for row in rows:
        meta = row['metadata']
        measurement = meta['measurement']
        source = measurement['source']
        if source not in cache:
            cache[source] = read_csv(source)
        raw = cache[source][measurement['source_row'] - 2]
        n, k = float(raw['Replicates']), float(raw['Binding Replicates'])
        negative = row['labels']['relation'] == 'nonbinding'
        if not all(math.isfinite(v) and v.is_integer() for v in (n, k)) or n < 2:
            raise ValueError('Independent replicate count audit failed')
        if negative and (k != 0 or raw['Binder'].lower() != 'false' or raw['KD (nM)'].strip()):
            raise ValueError('Independent negative source audit failed')
        if not negative and (k != n or raw['Binder'].lower() != 'true'
                             or not float(raw['KD (nM)']) > 0):
            raise ValueError('Independent positive source audit failed')
        ref = read_csv(meta['framework_reference'])[0]
        intervals = sorted((ref['heavy'].index(ref[name]),
                            ref['heavy'].index(ref[name]) + len(ref[name]), raw[name])
                           for name in CDRS)
        reconstructed = ref['heavy']
        for start, end, replacement in reversed(intervals):
            reconstructed = reconstructed[:start] + replacement + reconstructed[end:]
        sequences, header = {}, None
        for line in (RAW / (meta['antigen_reference_id'] + '.fasta')).read_text().splitlines():
            if line.startswith('>'):
                header = line[1:]
                sequences[header] = ''
            elif header is not None:
                sequences[header] += line.strip()
        chains = dict(zip(row['chain_roles'], row['chains']))
        if (chains['antibody_heavy'] != reconstructed or chains['antibody_light'] != ref['light']
                or chains['antigen'] != sequences[meta['antigen_reference_entity']]):
            raise ValueError('Independent source sequence audit failed')
        checked += 1
    return dict(status='passed', records=checked,
                method='raw_source_label_checks_and_independent_coordinate_splicing')


def vhh_inventory():
    """Inventory labelled VHH without silently changing the current reader scope."""
    items = []
    for target in ('hIL6', 'hTNFa'):
        folder = ROOT / 'data/nanobody_raw/nbbench/hf_data' / target
        for split in ('train', 'val', 'test'):
            path = folder / (split + '.csv')
            counts = Counter(r['label'] for r in read_csv(path))
            items.append(dict(target=target, split=split, path=str(path), sha256=sha(path),
                              records=sum(counts.values()), labels=dict(counts)))
    return dict(status='inventoried_not_in_current_training', files=items,
                reason='Current fixed-chain reader explicitly excludes nanobody_vhh; preserve VHH identity and original splits')


def candidates():
    verify_scope()
    rows, audit, files, counts = candidate_rows()
    WORK.mkdir(parents=True, exist_ok=True)
    if (WORK / 'candidates.json').exists():
        raise FileExistsError('Candidate receipt exists; use --stage match/export')
    path = WORK / 'candidate_records.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (WORK / 'source_label_audit.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in audit))
    write(WORK / 'vhh_inventory.json', vhh_inventory())
    write(WORK / 'candidates.json', dict(status='complete', records=len(rows), counts=counts,
        labels=dict(Counter(r['labels']['relation'] for r in rows)), files=files,
        records_sha256=sha(path), source_audit_sha256=sha(WORK / 'source_label_audit.jsonl'),
        parent_manifest_sha256=sha(PARENT / 'dataset_manifest.json'),
        registry_sha256=sha(SCOPE_DIR / 'registry.json')))
    print(json.dumps(dict(stage='candidates', records=len(rows), counts=counts)), flush=True)


def verify_candidates():
    report = json.loads((WORK / 'candidates.json').read_text())
    for path, digest in report['files'].items():
        if sha(path) != digest:
            raise ValueError('Candidate source changed: ' + path)
    if (sha(WORK / 'candidate_records.jsonl') != report['records_sha256']
            or sha(PARENT / 'dataset_manifest.json') != report['parent_manifest_sha256']
            or sha(SCOPE_DIR / 'registry.json') != report['registry_sha256']):
        raise ValueError('Candidate/parent/scope identity changed')
    verify_scope()
    return report


def match(workers):
    verify_candidates()
    rows = [json.loads(line) for line in (WORK / 'candidate_records.jsonl').read_text().splitlines()]
    values = defaultdict(set)
    for row in rows:
        keys, _ = keys_for_row(row)
        for role in ('heavy', 'light', 'h3', 'antigen'):
            if keys.get(role):
                values[role].add(keys[role])
    names, exact, t4 = bank_index()
    filters.RUN = WORK
    clusters = {role: filters.clustered_matches(role, values[role], exact.get(role, {}), cutoff, workers)
                for role, cutoff in (('heavy', .95), ('h3', .8), ('antigen', .7))}
    contained = {}
    for antigen in values['antigen']:
        bits = 0
        for peptide, mask in exact.get('peptide', {}).items():
            if peptide in antigen:
                bits |= mask
        if bits:
            contained[antigen] = bits
    save_pickle(WORK / 'match_index.pkl', dict(names=names, exact=exact, t4=t4,
        near={}, clusters=clusters, contained=contained))
    write(WORK / 'matching.json', dict(status='complete', index_sha256=sha(WORK / 'match_index.pkl'),
        candidates_sha256=sha(WORK / 'candidates.json'), registry_sha256=sha(SCOPE_DIR / 'registry.json')))
    print(json.dumps(dict(stage='matched', candidates=len(rows))), flush=True)


def scan_parent(job):
    """Verify all parent bytes and locate candidate heavy chains in either split."""
    spec, heavies = job
    path = PARENT / spec['path']
    seen, matched, digest = 0, [], hashlib.sha256()
    # Antibody source names are explicit in this frozen parent manifest.
    inspect = spec['source'] in ('oas', 'oas_paired', 'asd_antibody', 'nanobody', 'asd_nanobody')
    with path.open('rb', buffering=4 * 1024 * 1024) as stream:
        for line in stream:
            digest.update(line)
            seen += 1
            if inspect:
                row = json.loads(line)
                for role, seq in zip(row['chain_roles'], row['chains']):
                    if role == 'antibody_heavy' and seq in heavies:
                        matched.append(row)
                        break
    if seen != spec['records'] or digest.hexdigest() != spec['sha256']:
        raise ValueError('Parent shard changed: ' + str(path))
    return dict(path=spec['path'], records=seen, sha256=digest.hexdigest(), matched=matched)


def export(workers):
    from scripts.research.bert_mlm_v8_data import release_binding
    from dllm.pipelines.immune_llada.fixed_chain.unbinding_mlm import UnbindingMLMCollator, corrupt
    from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
    from dllm.pipelines.immune_llada.data.preprocessing.validators import validate_prepared_row
    import torch
    verify_candidates()
    matching = json.loads((WORK / 'matching.json').read_text())
    if (matching['index_sha256'] != sha(WORK / 'match_index.pkl')
            or matching['candidates_sha256'] != sha(WORK / 'candidates.json')):
        raise ValueError('Stale candidate matching')
    bound = release_binding(PARENT)
    parent = json.loads((PARENT / 'dataset_manifest.json').read_text())
    rows = [json.loads(line) for line in (WORK / 'candidate_records.jsonl').read_text().splitlines()]
    rebuilt, _, _, _ = candidate_rows()
    if rebuilt != rows:
        raise ValueError('Repeated source reconstruction differs')
    index, held = read_pickle(WORK / 'match_index.pkl'), held_probe_keys()
    candidate_heavies = {r['chains'][1] for r in rows}
    original_shards = [dict(s, split=split) for split, b in parent['splits'].items() for s in b['shards']]
    parent_hits, parent_receipts = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(scan_parent, [(s, candidate_heavies) for s in original_shards]):
            parent_hits.extend(result.pop('matched'))
            parent_receipts.append(result)
    exposure = defaultdict(set)
    prior_pairs = defaultdict(set)
    for row in parent_hits:
        chain = dict(zip(row['chain_roles'], row['chains']))
        exposure[chain['antibody_heavy']].add(row['split'])
        if chain.get('antigen'):
            prior_pairs[(chain['antibody_heavy'], chain.get('antibody_light', ''), chain['antigen'])].add(
                row.get('labels', {}).get('relation'))
    collator = UnbindingMLMCollator(FixedChainConfig(**bound['model']))
    generator = torch.Generator().manual_seed(42)
    counts, targets, audit, keep = Counter(), defaultdict(Counter), [], []
    pair_labels = defaultdict(set)
    for row in rows:
        pair_labels[tuple(row['chains'])].add(row['labels']['relation'])
    emitted = set()
    for row in rows:
        key, _ = keys_for_row(row)
        hits = reasons(key, index)
        heavy, light, antigen = key['heavy'], key['light'], key['antigen']
        pair = (heavy, light, antigen)
        seen = exposure[heavy]
        split = ('valid' if seen == {'valid'} else 'train' if seen == {'train'} else
                 'valid' if int(hashlib.sha256(heavy.encode()).hexdigest()[:8], 16) % 10 == 0 else 'train')
        row['split'] = split
        reason = None
        prior = prior_pairs.get(pair, set()) | prior_pairs.get((heavy, '', antigen), set())
        relation = row['labels']['relation']
        if hits:
            reason = 'protected_test_overlap'
        elif len(seen) > 1:
            reason = 'parent_train_valid_receptor_overlap'
        elif relation in prior:
            reason = 'already_in_parent'
        elif prior & ({'binding', 'nonbinding'} - {relation}) or len(pair_labels[tuple(row['chains'])]) > 1:
            reason = 'conflicting_relation'
        elif pair in emitted:
            reason = 'duplicate_pair'
        elif split == 'train' and receptor_key(row) in held:
            reason = 'probe_valid_overlap'
        if reason is None:
            validate_prepared_row(row)
            try:
                batch = collator([row])
            except ValueError as error:
                if 'overflow' not in str(error):
                    raise
                reason = 'capacity_overflow'
            if reason is None:
                noised = corrupt(batch, collator.tokenizer, generator)
                if relation == 'nonbinding':
                    if (batch['eligible_mask'] & ~batch['explicit_relation_mask']).any():
                        raise ValueError('Negative has sequence loss')
                    other = ~batch['explicit_relation_mask']
                    if (noised['input_ids'][other] != batch['input_ids'][other]).any():
                        raise ValueError('Negative context was corrupted')
                elif not (batch['eligible_mask'] & batch['residue_mask']).any():
                    raise ValueError('Positive lost MLM eligibility')
                emitted.add(pair)
                keep.append(row)
        decision = reason or 'retained_' + relation
        target = row['metadata']['measurement']['target']
        counts[decision] += 1
        targets[target][decision] += 1
        audit.append(dict(source=row['metadata']['measurement']['source'],
            source_row=row['metadata']['measurement']['source_row'], decision=decision, split=split,
            reasons={k: [name for i, name in enumerate(index['names']) if mask & (1 << i)]
                     for k, mask in hits.items()}))
    write(WORK / 'selection_report.json', dict(status='complete', counts=dict(counts),
        by_target={k: dict(v) for k, v in targets.items()}, parent_hits=len(parent_hits)))
    (WORK / 'selection_audit.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in audit))
    if not any(r['labels']['relation'] == 'nonbinding' for r in keep):
        raise ValueError('No admissible new antibody negatives; selection report preserved')
    stage = OUTPUT.with_name(OUTPUT.name + '.building')
    if OUTPUT.exists() or stage.exists():
        raise FileExistsError('Refusing to overwrite published/building dataset')
    stage.mkdir()
    shutil.copyfile(PARENT / 'immune_config.json', stage / 'immune_config.json')
    shards, added = [], []
    for spec in original_shards:
        dest = stage / spec['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(PARENT / spec['path'])
        shards.append({k: spec[k] for k in ('path', 'source', 'records', 'sha256', 'split')})
    for split in parent['splits']:
        selected = [r for r in keep if r['split'] == split]
        if not selected:
            continue
        path = Path(split) / 'abexp_igdesign_antibody-00000.jsonl'
        (stage / path).write_text(''.join(json.dumps(r) + '\n' for r in selected))
        spec = dict(path=str(path), source='igdesign_antibody', split=split, records=len(selected),
                    sha256=sha(stage / path), labels=dict(Counter(r['labels']['relation'] for r in selected)))
        shards.append(spec)
        added.append(spec)
    manifest = {k: deepcopy(parent[k]) for k in ('format', 'schema_version', 'reader_contract')}
    manifest.update(dataset_version=OUTPUT.name, created_at=datetime.now(timezone.utc).isoformat(), splits={})
    for split in parent['splits']:
        specs = [s for s in shards if s['split'] == split]
        sources = Counter()
        for spec in specs:
            sources[spec['source']] += spec['records']
        manifest['splits'][split] = dict(records=sum(s['records'] for s in specs), shards=specs,
            sources={k: dict(records=v) for k, v in sources.items()})
    manifest['sources'] = sorted({s['source'] for s in shards})
    manifest['antibody_expansion'] = dict(parent=str(PARENT), parent_manifest_sha256=bound['manifest_sha256'],
        added_shards=added, work=str(WORK), candidates_sha256=sha(WORK / 'candidates.json'),
        matching_sha256=sha(WORK / 'matching.json'), selection_sha256=sha(WORK / 'selection_audit.jsonl'),
        policy='Consistent SPR negatives plus same-source positives; published sequences only; fixed study VH/VL reconstruction; '
               'reference antigen context; V7 protected tests unchanged; original V8 bytes unchanged; no VHH admitted')
    write(stage / 'dataset_manifest.json', manifest)
    independent = audit_exported_source_rows([
        json.loads(line) for spec in added
        for line in (stage / spec['path']).read_text().splitlines()])
    evidence = dict(status='passed', dataset_manifest_sha256=sha(stage / 'dataset_manifest.json'),
        records={s: b['records'] for s, b in manifest['splits'].items()},
        added_records=len(keep), added_labels=dict(Counter(r['labels']['relation'] for r in keep)),
        all_parent_shards_sha256_verified=True, all_new_rows_actual_collator=True,
        all_new_negatives_sequence_loss_zero=True, all_new_negatives_context_unchanged=True,
        independent_source_reconstruction=independent, new_valid_heavy_in_parent_train=0,
        config_sha256=sha(stage / 'immune_config.json'),
        collator_sha256=sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/layout.py'),
        objective_sha256=sha(ROOT / 'dllm/pipelines/immune_llada/fixed_chain/unbinding_mlm.py'))
    for name in ('verification.json', 'reader_acceptance.json', 'independent_acceptance.json'):
        write(stage / name, evidence)
    write(stage / 'cleaning_report.json', json.loads((WORK / 'selection_report.json').read_text()))
    write(WORK / 'parent_verification.json', parent_receipts)
    stage.rename(OUTPUT)
    write(WORK / 'release.json', dict(**evidence, output=str(OUTPUT)))
    print(json.dumps(dict(stage='published', output=str(OUTPUT), **evidence)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('candidates', 'match', 'export', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    for name, operation in (('candidates', candidates), ('match', match), ('export', export)):
        if args.stage in ('all', name):
            operation() if name == 'candidates' else operation(args.workers)
