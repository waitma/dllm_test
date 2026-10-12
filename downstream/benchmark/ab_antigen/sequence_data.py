"""Build the four audited sequence subsets (author data; project protocol).

Run: python -m downstream.benchmark.ab_antigen.sequence_data
Requires acquire_sequence --features first. Never modifies upstream assets.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data/downstream/ab_antigen_sequence_v1'
AA = set('ACDEFGHIKLMNPQRSTVWY')
MAX_LENGTH = 2048


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as out:
        for row in rows:
            out.write(json.dumps(row, allow_nan=False) + '\n')
    tmp.replace(path)


def load_feature(path):
    import torch
    allow = [(np._core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
             (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.ndarray, np.dtype]
    allow += [type(np.dtype(t)) for t in ['float32', 'float64', 'int64', 'int32', 'bool']]
    with torch.serialization.safe_globals(allow):
        return torch.load(path, map_location='cpu', weights_only=True)


def check_sequences(ag, h, l):
    for seq in (ag, h, l):
        if not isinstance(seq, str) or not seq:
            return 'missing_chain'
        if set(seq) - AA:
            return 'noncanonical_residue'
    if len(h) > 166 or len(l) > 134:
        return 'receptor_canvas_overflow'
    if len(ag) + 302 + 32 > MAX_LENGTH:
        return 'context_overflow'
    return None


def record(dataset, source_id, ag, h, l, **kw):
    return dict(id=digest([dataset, source_id])[:24], source_id=str(source_id), dataset=dataset,
                antigen=ag, heavy=h, light=l, antigen_key=digest(ag), antibody_key=digest([h,l]), **kw)


def covunibind():
    base = ROOT / 'data/downstream/ab_antigen_official/covunibind/1a3d3a135c21e2a07bf75c9eb7015585b168bf65/data'
    outputs, audits = {}, {}
    for name, filename in [('binding', 'covabdab_binding.parquet'), ('spr', 'spr_ab_affinity.parquet')]:
        df = pd.read_parquet(base / filename)
        rows, excluded = [], []
        for idx, row in df.iterrows():
            ag, h, l = [row[c] for c in ['antigen_sequence','antibody_vh_sequence','antibody_vl_sequence']]
            reason = check_sequences(ag,h,l)
            if reason:
                excluded.append(dict(source_id=int(idx), reason=reason)); continue
            expected = 'binding' if name == 'binding' else 'log10_kd'
            if row.target_type != expected:
                excluded.append(dict(source_id=int(idx), reason='unexpected_endpoint')); continue
            value = float(row.target_value)
            if not math.isfinite(value) or (name == 'binding' and value not in (0,1)):
                excluded.append(dict(source_id=int(idx), reason='invalid_label')); continue
            rows.append(record(name, int(idx), ag,h,l, label=value if name=='binding' else -value,
                raw_label=value, endpoint=expected, source_doi=str(row.source_doi),
                assay=str(row.assay_name), lineage=str(row.antigen_lineage),
                antibody_name=str(row.antibody_name), pkd=None,
                pkd_status='not_applicable' if name=='binding' else 'unit_and_censoring_unverified'))
        # Retain independent assays but exclude contradictory measurements within an assay.
        groups=defaultdict(list)
        for r in rows:
            groups[(r['antigen_key'],r['antibody_key'],r['source_doi'],r['assay'])].append(r)
        keep=[]
        for group in groups.values():
            if len({r['raw_label'] for r in group})>1:
                excluded.extend(dict(source_id=r['source_id'],reason='conflicting_replicates') for r in group)
            else:
                keep.append(group[0]); excluded.extend(dict(source_id=r['source_id'],reason='exact_duplicate') for r in group[1:])
        outputs[name]=keep
        audits[name]=dict(raw=len(df), retained=len(keep), exclusions=excluded,
            source_sha256=hashlib.sha256((base/filename).read_bytes()).hexdigest(),
            score_direction='higher_is_stronger', pkd_status='blocked_unit_and_censoring' if name=='spr' else None)
    return outputs,audits


def official_path(base,value):
    path=base/Path(value).relative_to('data')
    if path.exists():return path
    matches=[p for p in path.parent.glob('*') if p.name.lower()==path.name.lower()]
    return matches[0] if len(matches)==1 else path


def abbibench():
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1
    base=DATA/'raw/abbibench'
    metadata=json.loads((base/'metadata.json').read_text())
    rows, exclusions, assays = [], [], {}
    from downstream.benchmark.ab_antigen.sequence_metrics import numbering
    wt_sequences=[]
    for meta in metadata.values():
        pdb=official_path(base,meta['pdb_path'])
        if pdb.exists():
            struct=PDBParser(QUIET=True).get_structure('wt',pdb)[0]
            for c in [meta['heavy_chain'],meta['light_chain']]:
                wt_sequences.append(''.join(seq1(r.resname) for r in struct[c] if r.id[0]==' '))
    number_dir=DATA/'audit/abbibench_wt_numbering_case_resolved';number_dir.mkdir(parents=True,exist_ok=True)
    wt_numbering=numbering(wt_sequences,number_dir)
    seen_files=set()
    for key, meta in metadata.items():
        pdb=official_path(base,meta['pdb_path'])
        for filename in meta['affinity_data']:
            path=official_path(base,filename)
            if str(path) in seen_files: continue
            seen_files.add(str(path))
            assay=path.stem
            if not path.exists() or not pdb.exists():
                exclusions.append(dict(assay=assay,reason='official_metadata_missing_file'));continue
            structure=PDBParser(QUIET=True).get_structure(key,pdb)[0]
            def chain(c):
                return ''.join(seq1(r.resname) for r in structure[c] if r.id[0]==' ')
            raw_h,raw_l=chain(meta['heavy_chain']),chain(meta['light_chain'])
            nh,nl=wt_numbering.get(digest(raw_h)),wt_numbering.get(digest(raw_l))
            if not nh or not nl or nh['chain_type']!='H' or nl['chain_type'] not in ['K','L']:
                exclusions.append(dict(assay=assay,reason='wt_chain_numbering_failed'));continue
            wt_h,wt_l=nh['fv'],nl['fv']
            ag=''.join(chain(c) for c in meta['antigen_chains'])
            wt_reason=check_sequences(ag,wt_h,wt_l)
            df=pd.read_csv(path)
            assays[assay]=dict(raw=len(df), antigen_chains=meta['antigen_chains'],
                antigen_policy='official_chain_order_concatenation', wt_heavy=wt_h,wt_light=wt_l,
                direction_status='higher_stronger_paper_v2_table1', wt_status=wt_reason,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for idx, row in df.iterrows():
                h,l=row.heavy_chain_seq,row.light_chain_seq
                trimmed=[];boundary_error=False
                for seq,raw,num in [(h,raw_h,nh),(l,raw_l,nl)]:
                    if not isinstance(seq,str):boundary_error=True;break
                    if len(seq)==len(raw):
                        a,b=num['start'],num['end']
                        if seq[:a]!=raw[:a] or seq[b:]!=raw[b:]:boundary_error=True;break
                        trimmed.append(seq[a:b])
                    elif len(seq)==len(num['fv']):trimmed.append(seq)
                    else:boundary_error=True;break
                if boundary_error:
                    exclusions.append(dict(assay=assay,source_id=int(idx),reason='unverified_variant_fv_boundary'));continue
                h,l=trimmed
                reason=check_sequences(ag,h,l) or wt_reason
                if reason:
                    exclusions.append(dict(assay=assay,source_id=int(idx),reason=reason));continue
                value=float(row.binding_score)
                if not math.isfinite(value):
                    exclusions.append(dict(assay=assay,source_id=int(idx),reason='invalid_label'));continue
                rows.append(record('abbibench',f'{assay}:{idx}',ag,h,l,assay=assay,
                    label=value,raw_label=value,endpoint='binding_score',
                    direction_status='higher_stronger_paper_v2_table1',wt_heavy=wt_h,wt_light=wt_l))
    # Remove exact duplicates without averaging experimental labels.
    unique=[];seen=set()
    for row in rows:
        key=(row['assay'],row['antigen_key'],row['antibody_key'],row['label'])
        if key in seen:exclusions.append(dict(source_id=row['source_id'],reason='exact_duplicate'))
        else:unique.append(row);seen.add(key)
    rows=unique
    return rows,dict(retained=len(rows),assays=assays,exclusions=exclusions,
        gate='official_asset_completeness_checked',direction_evidence='https://arxiv.org/html/2506.04235v2#S3.SS1',unmapped_csv=sorted(str(p.name) for p in (base/'binding_affinity').glob('*.csv') if str(p) not in seen_files))


def chimera():
    base=DATA/'raw/chimera'; split=json.loads((base/'splits/antigen_fold.json').read_text())
    membership={cid:s for s,ids in split.items() for cid in ids}
    summary=pd.read_csv(base/'metadata/final_summary.csv')
    rows,exclusions=[],[]
    for cid in summary.complex_id:
        path=base/'complex_features'/f'{cid}.pt'
        if not path.exists():
            raise FileNotFoundError(f'Incomplete acquisition: {path}')
        x=load_feature(path); ag,h,l=x['antigen_sequence'],x['heavy_sequence'],x['light_sequence']
        # Author IMGT numbering locates VH/VL without arbitrary length clipping.
        bounds={}; mapping_error=False
        for chain,full in [('heavy',h),('light',l)]:
            numbered=[entry for entry in x['numbering']['imgt'].get(chain,[]) if entry[2]!='-']
            fv=''.join(entry[2] for entry in numbered)
            start=full.find(fv) if fv else -1
            if start<0 or full.find(fv,start+1)>=0:
                mapping_error=True;break
            bounds[chain]=(start,start+len(fv))
        if mapping_error:
            exclusions.append(dict(source_id=cid,reason='unverified_imgt_fv_mapping'));continue
        h=h[slice(*bounds['heavy'])];l=l[slice(*bounds['light'])]
        reason=check_sequences(ag,h,l)
        if reason or cid not in membership:
            exclusions.append(dict(source_id=cid,reason=reason or 'no_official_split'));continue
        # Reference-only CA contacts use the original 8 A CA definition, not generated structures.
        contacts={}
        for chain in ('heavy','light'):
            a=np.asarray(x[f'{chain}_ca_coords']); b=np.asarray(x['antigen_ca_coords'])
            contact=(np.linalg.norm(a[:,None,:]-b[None,:,:],axis=-1)<8).any(axis=1)
            start,end=bounds[chain]
            contacts[chain]=(np.where(contact[start:end])[0]).tolist()
        rows.append(record('chimera',cid,ag,h,l,split=membership[cid],
            numbering={c:[e for e in x['numbering']['imgt'][c] if e[2]!='-'] for c in bounds},
            cdr_masks={c:x['cdr_masks']['imgt'][c][slice(*bounds[c])] for c in bounds},fv_bounds=bounds,
            contact_indices=contacts,contact_definition='reference_CA_CA_lt_8A'))
    ag_splits=defaultdict(set)
    for r in rows:ag_splits[r['antigen_key']].add(r['split'])
    keep=[]
    for r in rows:
        if len(ag_splits[r['antigen_key']])>1:
            exclusions.append(dict(source_id=r['source_id'],reason='antigen_exact_cross_split'))
        else:keep.append(r)
    return keep,dict(raw=len(summary),retained=len(keep),exclusions=exclusions,
        split='official_antigen_fold_filtered_exact_antigen_conflicts',protocol='chimera_sequence_only_v1')


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--without-chimera',action='store_true');args=p.parse_args()
    outputs,audits=covunibind()
    outputs['abbibench'],audits['abbibench']=abbibench()
    if not args.without_chimera:
        outputs['chimera'],audits['chimera']=chimera()
    for name,rows in outputs.items():write_rows(DATA/f'{name}.jsonl',rows)
    write_json(DATA/'build_audit.json',dict(protocol='ab_ag_sequence_v1',subsets=audits,
        max_length=MAX_LENGTH,provenance='author_data_project_protocol_not_official_rerun'))
    print(json.dumps({k:len(v) for k,v in outputs.items()}))


if __name__=='__main__':main()
