"""Freeze ten-percent V8 offsets; run.sh --stage prepare --output-root ABS."""
import collections
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import numpy as np
try:
    import orjson
except ImportError:
    orjson = json
from .common import ROOT, read, write, sha, digest, stamp

RECEPTORS = ('antibody_heavy', 'antibody_light', 'tcr_alpha', 'tcr_beta', 'nanobody_vhh')
REGIONS = ('FR1', 'CDR1', 'FR2', 'CDR2', 'FR3', 'CDR3', 'FR4')

def identity(row):
    parts = []
    for i, (role, seq) in enumerate(zip(row['chain_roles'], row['chains'])):
        if role not in RECEPTORS: continue
        flags = row.get('chain_metadata', {}).get(str(i), {}).get('synthetic_residue_mask', [])
        parts.append((role, ''.join(c for j, c in enumerate(seq) if not flags or not flags[j])))
    return hashlib.sha256(json.dumps(sorted(parts)).encode()).hexdigest()

def score(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big')

def partition(key):
    return 'screen' if score('arch70m-split-41016:' + key) % 2 == 0 else 'confirm'

def sample_rows(rows, cap, by_label=False):
    groups = collections.defaultdict(dict)
    for row in rows:
        group = (row['source_group'], row['record'].get('labels', {}).get('relation') if by_label else '')
        groups[group].setdefault(row['receptor_key'], row)
    return [r for group in sorted(groups, key=str) for r in
            sorted(groups[group].values(), key=lambda x: score('samples:' + x['receptor_key']))[:cap]]

def write_rows(path, rows):
    with path.open('w') as f:
        for row in rows: f.write(json.dumps(row) + '\n')
    return dict(path=str(path), sha256=sha(path), records=len(rows))

def scan_train(job):
    shard, held, out = job
    path = Path(shard['path']); before = path.stat()
    if sha(shard['index']) != shard['index_sha256']: raise ValueError('Parent offsets changed')
    admitted = set(int(x) for x in np.load(shard['index']))
    groups = collections.defaultdict(list); excluded = 0; found = 0; offset = 0; h = hashlib.sha256()
    with path.open('rb', buffering=8 << 20) as f:
        for line in f:
            here = offset; offset += len(line); h.update(line)
            if here not in admitted: continue
            found += 1; row = orjson.loads(line); key = identity(row)
            if key in held: excluded += 1; continue
            label = row.get('labels', {}).get('relation', 'unlabelled')
            signature = ','.join(sorted(r for r in row['chain_roles'] if r in RECEPTORS))
            groups[label, signature].append((score('arch70m-data-42:' + path.name + ':' + str(here)), here))
    if found != len(admitted) or h.hexdigest() != shard['sha256']: raise ValueError('Parent shard binding differs: ' + str(path))
    if (before.st_size, before.st_mtime_ns) != (path.stat().st_size, path.stat().st_mtime_ns): raise ValueError('Shard changed during scan')
    selected = []; strata = []
    for key, values in sorted(groups.items()):
        count = max(1, round(len(values) / 10))
        chosen = sorted(values)[:count]; selected.extend(x[1] for x in chosen)
        strata.append(dict(relation=key[0], signature=key[1], population=len(values), selected=count))
    idx = Path(out) / ('train_' + path.stem + '.npy')
    np.save(idx, np.asarray(sorted(selected), dtype=np.uint64))
    return {**shard, 'index': str(idx), 'index_sha256': sha(idx), 'records': len(selected),
            'parent_records': shard['records'], 'excluded_held_receptor': excluded, 'strata': strata}

def prepare(output, data_root):
    out = Path(output) / 'data'; out.mkdir(parents=True, exist_ok=True)
    base = read(ROOT / 'experiments/arch70m/base.json')
    if sha(base['data_index']) != base['data_index_sha256']: raise ValueError('Parent index changed')
    parent = read(base['data_index'])
    from .ab_specificity import inputs, clean
    ab_frame, ab_pairs, _, ab_audit = inputs()
    binding = dict(parent_index_sha256=base['data_index_sha256'], data_root=str(data_root), fraction=.1, seed=42, version=2,
                   protected_ab_binding=ab_audit)
    if (out / 'index.json').exists():
        index = read(out / 'index.json')
        if index['preparation'] != binding: raise ValueError('Prepared data parameters changed')
        return index
    valid = []; held = set(); provenance = []
    for item in parent['shards']:
        if item['split'] != 'valid': continue
        path = Path(data_root) / item['split'] / Path(item['path']).name
        if sha(path) != item['sha256']: raise ValueError('Validation shard changed: ' + str(path))
        if sha(item['index']) != item['index_sha256']: raise ValueError('Validation offsets changed')
        with path.open('rb') as f:
            for offset in np.load(item['index']):
                f.seek(int(offset)); row = orjson.loads(f.readline()); key = identity(row); held.add(key)
                valid.append(dict(record=row, source_group=item['source'], receptor_key=key,
                                  origin=dict(path=str(path), offset=int(offset))))
        provenance.append(dict(path=str(path), sha256=item['sha256']))
    probe = {}
    for name in ('probe_train', 'probe_valid'):
        ref = parent['files'][name]
        if sha(ref['path']) != ref['sha256']: raise ValueError('Protected probe changed')
        probe[name] = [json.loads(line) for line in Path(ref['path']).read_text().splitlines() if line]
        for r in probe[name]: r['receptor_key'] = identity(r['record'])
        provenance.append(ref)
    held.update(r['receptor_key'] for r in probe['probe_valid'])
    # Protect the entire downstream CV benchmark, including the audited Fv form.
    ab_full_pairs=list(zip(map(clean,ab_frame.h_sequence),map(clean,ab_frame.l_sequence)))
    ab_keys={identity(dict(chains=list(p),chain_roles=['antibody_heavy','antibody_light'])) for p in ab_pairs+ab_full_pairs}
    held.update(ab_keys)
    files = {}
    for part in ('screen', 'confirm'):
        rows = [x for x in valid if partition(x['receptor_key']) == part and x['receptor_key'] not in ab_keys]
        files[part] = dict(token_valid=write_rows(out / f'{part}_token.jsonl', sample_rows(rows, 64 if part == 'screen' else 128)),
                           task_valid=write_rows(out / f'{part}_tasks.jsonl', sample_rows(rows, 16 if part == 'screen' else 64)),
                           probe_train=write_rows(out / f'{part}_probe_train.jsonl', probe['probe_train']),
                           probe_valid=write_rows(out / f'{part}_probe_valid.jsonl',
                                [x for x in probe['probe_valid'] if partition(x['receptor_key']) == part]))
    jobs = []
    for item in parent['shards']:
        if item['split'] == 'train':
            item = {**item, 'path': str(Path(data_root) / 'train' / Path(item['path']).name)}
            jobs.append((item, held, str(out)))
    result = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(scan_train, job) for job in jobs]):
            shard = f.result(); result.append(shard)
            print(json.dumps(dict(stage='data_shard', completed=len(result), total=len(jobs), selected=shard['records'])), flush=True)
    result.sort(key=lambda x: x['path'])
    index = dict(schema='arch70m.data.v1', preparation=binding, data_dir=str(data_root),
        shards=result, files=files, records=sum(x['records'] for x in result),
        excluded_held_receptor=sum(x['excluded_held_receptor'] for x in result),
        held_receptor_keys=len(held), validation_rows=len(valid),
        protected_ab_keys=len(ab_keys), protected_ab_binding=ab_audit,
        validation_binding=provenance, created=stamp(),
        scope='Fixed stratified 10% per source shard/relation/receptor kind; exact receptor isolation, not a homology guarantee. Existing protected probe head-training split is evaluation-only.')
    write(out / 'index.json', index)
    print(json.dumps(dict(stage='data_prepared', records=index['records'], index=str(out/'index.json'))), flush=True)
    return index
