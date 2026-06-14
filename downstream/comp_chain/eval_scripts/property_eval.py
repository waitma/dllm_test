from metrics._utils import get_descriptors_as_dict
from metrics._large_molecule_descriptors import LargeMoleculeDescriptors
from metrics._get_batch_descriptors import get_batch_descriptors
import pandas as pd
import argparse
from tqdm import tqdm

TOKEN_GAP = "-"

def get_property_eval(generated_samples: pd.DataFrame, ref_dataset: pd.DataFrame, chain: str):
    # Determine sequence column names based on chain type
    if chain == 'fv_heavy':
        gen_seq_col = 'gen_h_sequence'
        raw_seq_col = 'raw_h_sequence'
        prefix = 'fv_heavy_'
    else:  # fv_light
        gen_seq_col = 'gen_l_sequence'
        raw_seq_col = 'raw_l_sequence'
        prefix = 'fv_light_'
    
    sample_df = []
    for sample_dict in tqdm(generated_samples.to_dict('records'), desc='Processing generated samples'): 
        all_labels = {}
            
        seq = sample_dict[gen_seq_col].replace(TOKEN_GAP, "")
        all_labels.update(LargeMoleculeDescriptors(seq).asdict())
        all_labels = {prefix + key: value for key, value in all_labels.items()}
        # Add sequence column for validation
        all_labels[f'{chain}_seq'] = seq
        
        sample_df.append({**sample_dict, **all_labels})
    sample_df = pd.DataFrame(sample_df)


    ref_df = []
    # duplicate
    ref_dataset = ref_dataset.copy()
    ref_dataset = ref_dataset.drop_duplicates(subset=[raw_seq_col])
    for sample_dict in tqdm(ref_dataset.to_dict('records'), desc='Processing reference samples'): 
        all_labels = {}
            
        seq = sample_dict[raw_seq_col].replace(TOKEN_GAP, "")
        all_labels.update(LargeMoleculeDescriptors(seq).asdict())
        all_labels = {prefix + key: value for key, value in all_labels.items()}
        # Add sequence column for validation
        all_labels[f'{chain}_seq'] = seq
        
        ref_df.append({**sample_dict, **all_labels})
    ref_df = pd.DataFrame(ref_df)

    wasserstein_distances, avg_wd, total_wd, prop_valid = get_batch_descriptors(sample_df, ref_df, chain)
    return wasserstein_distances, avg_wd, total_wd, prop_valid

def argparse_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_fpath', type=str, required=True)
    parser.add_argument('--ref_fpath', type=str, required=True)
    parser.add_argument('--chain', type=str, required=True)
    return parser.parse_args()  

def main(
    generated_fpath: str,
    ref_fpath: str,
    chain: str,
):
    generated_samples = pd.read_csv(generated_fpath)
    ref_dataset = pd.read_csv(ref_fpath)
    wasserstein_distances, avg_wd, total_wd, prop_valid = get_property_eval(generated_samples, ref_dataset, chain)

    print("\n" + "="*60)
    print(f"{'Property Evaluation Results':^60}")
    print("="*60)
    
    print(f"\n{'Wasserstein Distances by Feature:':^60}")
    print("-"*60)
    for feature, distance in sorted(wasserstein_distances.items()):
        feature_name = feature.replace('fv_light_', '').replace('fv_heavy_', '').replace('_wd', '')
        print(f"  {feature_name:<40} {distance:>15.6f}")
    
    print("\n" + "-"*60)
    print(f"{'Summary Statistics:':^60}")
    print("-"*60)
    print(f"  {'Average Wasserstein Distance:':<40} {avg_wd:>15.6f}")
    print(f"  {'Total Wasserstein Distance:':<40} {total_wd:>15.6f}")
    print(f"  {'Proportion of Valid Samples:':<40} {prop_valid:>15.2%}")
    print("="*60 + "\n")
    
    return wasserstein_distances, avg_wd, total_wd, prop_valid


if __name__ == "__main__":
    args = argparse_main()
    main(
        generated_fpath=args.generated_fpath,
        ref_fpath=args.ref_fpath,
        chain=args.chain,
    )