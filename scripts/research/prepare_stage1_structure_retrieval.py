"""Prepare experimental Fab geometry for frozen representation evaluation.

Activate pllm, then use protenix_abtcr Python (Biopython dependency):
python -B /project_c20250601/251105016/project/dllm_test/scripts/research/prepare_stage1_structure_retrieval.py --output ABS_NEW_DIRECTORY
Reads the existing SAbDab2 archive without extracting files. Selects one valid
representative per released antibody cluster before any model inference. Stores
only H/L sequences, IMGT residue keys, CA coordinates and provenance; no antigen
features or predicted structures are used.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

import numpy as np
from Bio.PDB import MMCIFParser
from Bio.Data.PDBData import protein_letters_3to1
from evaluate_stage1_frozen_gdp import ROOT, sha, write


def parse_member(raw,record):
    parser=MMCIFParser(QUIET=True)
    structure=parser.get_structure(record['instance'],io.StringIO(raw.decode()))
    meta=parser._mmcif_dict
    method=';'.join(meta.get('_exptl.method',[]))
    if not any(x in method for x in ['X-RAY DIFFRACTION','ELECTRON MICROSCOPY']):return None,'method'
    resolution=[]
    for key in ['_refine.ls_d_res_high','_em_3d_reconstruction.resolution']:
        for value in meta.get(key,[]):
            try:resolution.append(float(value))
            except ValueError:pass
    if not resolution or not np.isfinite(resolution).all() or min(resolution)>3.5:return None,'resolution'
    result=dict(record,method=method,resolution_angstrom=min(resolution),cif_sha256=hashlib.sha256(raw).hexdigest())
    first=next(structure.get_models())
    for role in ['h','l']:
        if record[role+'_chain'] not in first:return None,'chain_missing'
        residues=[r for r in first[record[role+'_chain']] if r.id[0]==' ']
        if any('CA' not in r for r in residues):return None,'CA_missing'
        if any(r.resname not in protein_letters_3to1 for r in residues):return None,'nonstandard_residue'
        seq=''.join(protein_letters_3to1[r.resname] for r in residues)
        if seq!=record[role]:return None,'sequence_mismatch'
        keys=[f'{role}:{r.id[1]}:{r.id[2].strip()}' for r in residues]
        if len(set(keys))!=len(keys):return None,'duplicate_residue_key'
        for number,(low,high) in enumerate([(27,38),(56,65),(105,117)],1):
            observed=''.join(protein_letters_3to1[r.resname] for r in residues if low<=r.id[1]<=high)
            if observed!=record[role+'_regions']['CDR'+str(number)]:return None,'IMGT_region_mismatch'
        xyz=np.array([r['CA'].coord for r in residues],dtype=np.float64)
        if not np.isfinite(xyz).all():return None,'nonfinite_coordinates'
        result[role+'_keys']=keys;result[role+'_ca']=xyz.tolist()
        result[role+'_framework']=[not any(low<=r.id[1]<=high for low,high in [(27,38),(56,65),(105,117)]) for r in residues]
        result[role+'_cdr3_indices']=[i for i,r in enumerate(residues) if 105<=r.id[1]<=117]
    return result,None


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    assert out.is_absolute()
    if out.exists():raise FileExistsError(out)
    canonical=ROOT/'data/immune_receptor_v2/canonical/antibody/sabdab2_ml.jsonl'
    archive=ROOT/'data/sabdab2_ml/raw/splits.tar.gz'
    candidates={};counts=Counter()
    with canonical.open() as f:
        for line in f:
            row=json.loads(line);context=row['context']
            if context.get('antibody_type')!='FAB':counts['not_FAB']+=1;continue
            entities={e['role']:e for e in row['entities']}
            if not {'antibody_heavy','antibody_light'}<=entities.keys():counts['missing_chain']+=1;continue
            h,l=entities['antibody_heavy'],entities['antibody_light']
            if set(h['sequence']+l['sequence'])-set('ACDEFGHIKLMNPQRSTVWY'):counts['noncanonical_sequence']+=1;continue
            if len(h['sequence'])>166 or len(l['sequence'])>134:counts['slot_capacity']+=1;continue
            if not 5<=len(h['regions']['CDR3'])<=25:counts['H3_length']+=1;continue
            instance=context['instance']
            record=dict(instance=instance,pdb=context['pdb_id'],cluster=context['released_antibody_cluster'],
                h=h['sequence'],l=l['sequence'],h_chain=h['metadata']['chain_id'],l_chain=l['metadata']['chain_id'],
                h_regions=h['regions'],l_regions=l['regions'],original_split=row['source_records'][0]['original_split'])
            if instance in candidates:
                assert candidates[instance]==record
                counts['duplicate_antigen_component']+=1
            candidates[instance]=record
    out.mkdir(parents=True)
    plan=dict(schema='stage1.experimental_structure.inventory.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        canonical=str(canonical),canonical_sha256=sha(canonical),archive=str(archive),archive_sha256=sha(archive),
        script_sha256=sha(__file__),candidate_instances=len(candidates),pre_filters=dict(counts),
        selection='FAB; canonical H/L within slots; H3 length 5..25; experimental X-ray/cryo-EM resolution <=3.5A; '
        'all variable-domain CA present; exact sequence and IMGT CDR agreement. Choose lexicographically '
        'smallest SHA256(instance) among valid instances within each released antibody cluster.',
        intended_evaluation='Frozen global residue-mean embeddings; deterministic query/gallery partition within H3-length groups; '
        'experimental Fv/H3 RMSD after common-framework Kabsch alignment; same-PDB, same-cluster, '
        'identical H3 and >=90% IMGT-aligned Fv identity pairs excluded. Sequence identity retrieval baseline.',
        status='preparing',model_inference_executed=False)
    write(out/'inventory_plan.json',plan)
    valid={};failures=[];found=set();start=time.monotonic();next_log=0
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            if not member.name.endswith('.cif'):continue
            instance=Path(member.name).stem
            if instance not in candidates:continue
            if instance in found:raise ValueError('Duplicate archive instance '+instance)
            found.add(instance);raw=tar.extractfile(member).read()
            try:result,reason=parse_member(raw,candidates[instance])
            except (ValueError,KeyError,IndexError) as error:
                result=None;reason='parse_error:'+str(error)
            if result is None:failures.append(dict(instance=instance,reason=reason))
            else:
                result['archive_member']=member.name
                cluster=result['cluster'];key=hashlib.sha256(instance.encode()).hexdigest()
                if cluster not in valid or key<valid[cluster][0]:valid[cluster]=(key,result)
            if time.monotonic()>=next_log:
                print(json.dumps(dict(phase='parse_geometry',processed=len(found),candidates=len(candidates),
                                     valid_clusters=len(valid),elapsed_seconds=time.monotonic()-start)),flush=True)
                next_log=time.monotonic()+60
    for instance in sorted(candidates.keys()-found):failures.append(dict(instance=instance,reason='archive_member_missing'))
    selected=[value[1] for _,value in sorted(valid.items())]
    for i,r in enumerate(selected):r['row']=i
    write(out/'structures.json',selected);write(out/'exclusions.json',failures)
    plan.update(status='complete',selected_clusters=len(selected),members_found=len(found),
                geometry_exclusions=dict(Counter(x['reason'] for x in failures)),
                h3_length_counts=dict(Counter(len(x['h_regions']['CDR3']) for x in selected)),
                structures_sha256=sha(out/'structures.json'),exclusions_sha256=sha(out/'exclusions.json'),
                seconds=time.monotonic()-start,
                limitations=['Canonical source split is not a student train/test guarantee',
                             'Training exposure audit and query/gallery geometry scoring remain pending'])
    write(out/'inventory_plan.json',plan)
    print(json.dumps({k:plan[k] for k in ['status','selected_clusters','geometry_exclusions','h3_length_counts','seconds']}),flush=True)


if __name__=='__main__':main()
