import pandas as pd
from polyleven import levenshtein
from ._large_molecule_descriptors import LargeMoleculeDescriptors


def get_descriptors_as_dict(sequence: str) -> dict:
    return {k: v for k, v in LargeMoleculeDescriptors.from_sequence(sequence).asdict().items() if k in set(LargeMoleculeDescriptors.descriptor_names())}


def rename_df(df: pd.DataFrame, prefix: str):
    df.rename({c: f"{prefix}_{c}" for c in df.columns}, inplace=True, axis=1)
    

def edit_dist(seq1, seq2):
    return levenshtein(seq1, seq2) / 1