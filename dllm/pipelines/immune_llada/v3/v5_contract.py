"""Pure-Python V5 record geometry; used by the full audit and its parity tests.

Run python -B -m pytest scripts/tests/immune_llada/test_v5_data_binding.py.
"""
import math
import re

AA = re.compile(r'[LAGVSERTIDPKQNFYMHWCXBUZOJ]+', re.IGNORECASE)
RECEPTORS = {'antibody_heavy','antibody_light','tcr_beta','tcr_alpha'}
ORDER = ['mhc','peptide','antigen','antibody_heavy','antibody_light','tcr_beta','tcr_alpha']


def geometry(row, capacity=302, max_length=2048):
    """Return exact canvas length and reasons; never crop or silently drop a row."""
    reasons=[]
    seqs,roles=row['chains'],row['chain_roles']
    if 'nanobody_vhh' in roles or row['task_type'] in ('nanobody','nanobody_antigen'):
        return {'excluded':'vhh','reasons':[]}
    if any(role not in ORDER for role in roles):
        return {'reasons':['unsupported_role']}
    if any(roles.count(role)>1 for role in set(roles)-{'mhc'}):
        return {'reasons':['duplicate_role']}
    if any(not seq or AA.fullmatch(seq) is None for seq in seqs):
        reasons.append('invalid_residue_sequence')
    if not math.isfinite(row.get('weight',1)):
        reasons.append('nonfinite_weight')
    receptors=[(role,seq,index) for index,(role,seq) in enumerate(zip(roles,seqs)) if role in RECEPTORS]
    receptors.sort(key=lambda x:ORDER.index(x[0]))
    signature=tuple(x[0] for x in receptors)
    if signature not in (('antibody_heavy',),('antibody_light',),('antibody_heavy','antibody_light'),
                          ('tcr_beta',),('tcr_alpha',),('tcr_beta','tcr_alpha')):
        return {'reasons':reasons+['unsupported_receptor_object']}
    synthetic=0;missing=0;observed=0
    for index,seq in enumerate(seqs):
        meta=row.get('chain_metadata',{}).get(str(index),{})
        flags=meta.get('synthetic_residue_mask')
        if flags is not None and (not isinstance(flags,list) or len(flags)!=len(seq) or not set(flags)<={0,1}):
            reasons.append('invalid_synthetic_provenance')
            continue
        if roles[index] in RECEPTORS:
            count=sum(flags) if flags is not None else 0
            synthetic+=count;observed+=len(seq)-count;missing+=int(count==len(seq))
    lookup={role:seq for role,seq in zip(roles,seqs)}
    if signature[0].startswith('tcr_'):
        prefix=0
        mhc=[seq for role,seq in zip(roles,seqs) if role=='mhc']
        peptide=lookup.get('peptide') or lookup.get('antigen')
        if mhc:prefix+=sum(map(len,mhc))+len(mhc)+2
        if peptide:prefix+=len(peptide)+4
        if not mhc and not peptide:prefix=4
        if 'peptide' in roles and 'antigen' in roles:reasons.append('ambiguous_tcr_context')
    else:
        if row['task_type']=='antibody' and len(seqs)>=2 and roles[:2]!=['antibody_heavy','antibody_light']:
            reasons.append('antibody_positional_role_mismatch')
        if 'mhc' in roles or 'peptide' in roles:reasons.append('unsupported_antibody_context')
        if 'antigen' in roles:
            if row['task_type'] not in ('antibody_antigen','antibody_neutralization'):
                reasons.append('unsupported_antigen_task')
            prefix=len(lookup['antigen'])+3
        else:prefix=4
    object_length=sum(len(x[1]) for x in receptors)+len(receptors)
    total=prefix+2+capacity
    if object_length>capacity:reasons.append('object_capacity_overflow')
    if total>max_length:reasons.append('sequence_budget_overflow')
    if max(map(len,seqs))>max_length:reasons.append('chain_budget_overflow')
    kind=('single' if len(receptors)==1 else 'paired')+('_scaffold' if synthetic else '_full')
    return dict(reasons=sorted(set(reasons)),object_length=object_length,total_length=total,
                category=kind,synthetic_positions=synthetic,observed_residues=observed,
                missing_chains=missing,receptor_roles=signature)
