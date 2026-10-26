"""Freeze small protein-only teacher-validation inputs before inference.

Activate pllm and run this absolute file. Reuses three existing Fab examples,
one OAS validation pair, and downloads four named public PDB structures.
These are engineering cases, not a held-out generalization benchmark.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB import PDBParser

ROOT = Path('/project_c20250601/251105016/project/dllm_test/output/general_teacher_validation_20261024')
OLD = Path('/project_c20250601/251105016/project/dllm_test/output/stage1_cross_chain_diagnostic_20261024')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    structures = ROOT / 'structures'
    structures.mkdir(parents=True, exist_ok=True)
    examples = []
    previous = json.loads((OLD / 'examples.json').read_text())
    for e in previous['examples']:
        if e['id'] != 'valid_01' and e['group'] != 'structure_reference':
            continue
        row = dict(id=e['id'], group=e['group'],
                   sequences=[c['sequence'] for c in e['request']['chains']])
        if 'reference' in e:
            p = OLD / 'structures' / (e['reference']['pdb_id'] + '_coordinates.npz')
            row['reference_coordinates'] = str(p)
            row['reference_sha256'] = sha(p)
        examples.append(row)
    for pdb_id, chains in [('1UBQ', ['A']), ('1MBN', ['A']), ('1AKE', ['A']), ('2HHB', ['A','B','C','D'])]:
        path = structures / (pdb_id + '.pdb')
        url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as r:
                path.write_bytes(r.read())
        model = PDBParser(QUIET=True).get_structure(pdb_id, path)[0]
        sequences, coords, mapping = [], [], []
        for chain_id in chains:
            residues = [r for r in model[chain_id] if r.id[0]==' ' and r.resname in protein_letters_3to1]
            sequence = ''.join(protein_letters_3to1[r.resname] for r in residues)
            if not sequence or any(x not in 'ACDEFGHIKLMNPQRSTVWY' for x in sequence):
                raise ValueError('Invalid canonical sequence ' + pdb_id)
            sequences.append(sequence)
            for r, aa in zip(residues, sequence):
                atom = 'CA' if aa == 'G' else 'CB'
                coords.append(r[atom].coord if atom in r else [np.nan]*3)
                mapping.append([chain_id, int(r.id[1]), r.id[2], aa])
        coords = np.asarray(coords, dtype=np.float32)
        target = structures / (pdb_id + '_coordinates.npz')
        np.savez_compressed(target, coordinates=coords,
                            distances=np.linalg.norm(coords[:,None]-coords[None,:], axis=-1))
        examples.append(dict(id='pdb_'+pdb_id.lower(), group='general_structure', sequences=sequences,
            reference_coordinates=str(target), reference_sha256=sha(target), pdb_url=url,
            pdb_sha256=sha(path), chains=chains, residue_mapping=mapping,
            input_scope='observed canonical protein residues; cofactors/ligands excluded'))
    for e in examples:
        e['chain_lengths'] = list(map(len, e['sequences']))
        e['sequence_sha256'] = hashlib.sha256(json.dumps(e['sequences']).encode()).hexdigest()
    payload = dict(scope='predeclared engineering/label sanity cases; not held-out model selection',
        representative_atom='CB; CA for glycine; missing coordinates excluded',
        seed=20261024, performance_repeats=3, examples=examples)
    (ROOT / 'examples.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps([dict(id=e['id'], lengths=e['chain_lengths']) for e in examples]))


if __name__ == '__main__':
    main()
