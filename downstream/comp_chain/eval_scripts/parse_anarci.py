from anarci import anarci
import re
from typing import Union

def locus_from_gene(gene: Union[str, None]) -> Union[str, None]:
    if not gene:
        return None
    m = re.match(r'^(IGH|IGK|IGL)', gene.upper())
    return m.group(1) if m else None

def parse_anarci_results(results):
    """
    Parse the ANARCI return structure:
    results = (numbering, alignment_details, hit_tables)

    Returns: A dict for each sequence containing:
      - chain: 'H'/'K'/'L'
      - v_call: e.g. 'IGHV3-23*04'
      - j_call: e.g. 'IGHJ2*01'
      - locus: 'IGH'/'IGK'/'IGL'
    """
    numbering, alignment_details, hit_tables = results

    out = []
    # alignment_details is a list (one element per sequence)
    # In the structure provided, each element is itself a list containing several dicts (usually take the first one)
    for i, per_seq_hits in enumerate(alignment_details):
        # per_seq_hits looks like:
        # [{'id': 'human_H', 'chain_type': 'H', 'germlines': {'v_gene': [('human','IGHV3-23*04'), score], 'j_gene': [('human','IGHJ2*01'), score]}}]
        record = {'index': i, 'chain': None, 'v_call': None, 'j_call': None, 'locus': None}

        if isinstance(per_seq_hits, list) and per_seq_hits:
            best = per_seq_hits[0]  # Take the first hit
            chain = best.get('chain_type')
            record['chain'] = chain

            gl = best.get('germlines', {})
            # Structure of v_gene/j_gene is [('species','GENE*allele'), score]
            v_call = None
            j_call = None
            if isinstance(gl.get('v_gene'), list) and gl['v_gene']:
                v_call = gl['v_gene'][0][1]  # Take the second item from ('human', 'IGHV...')
            if isinstance(gl.get('j_gene'), list) and gl['j_gene']:
                j_call = gl['j_gene'][0][1]

            record['v_call'] = v_call
            record['j_call'] = j_call

            # Parse locus from V/J gene prefix
            record['locus'] = locus_from_gene(v_call) or locus_from_gene(j_call)

        out.append(record)

    return out

if __name__ == "__main__":
    # Example: Antibody variable region sequences
    sequences = [
        ("seq1", "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDTAVYYCARDTRQLAPYTFDYWGQGTLVTVSS"),
        # ("seq2", "DIQMTQSPASLSASVGETVTITCRASQDVNTAVAWYQQKPGKAPKLLIYDASTRATGIPDRFSGSGSGTDFTLTISSLQPEDFATYYCQQ"),
        # ("seq3", "QSVLTQPPSVSAAPGQKVTISCSGSSSNIGNNYVSWYQQLPGTAPKLLIYENNKRPSGIPDRFSGSKSGTSATLGITGLQTGDEADYYCGTWDSSLSVLFGGGTKLTVL")
    ]

    # Call ANARCI with IMGT numbering scheme
    results = anarci(sequences, scheme='imgt', assign_germline=True)
    print(results)

    # results 是一个 tuple (numbering, alignment_details)
    numbering, alignment_details, _ = results

    parsed = parse_anarci_results(results)
    for r in parsed:
        print(r)