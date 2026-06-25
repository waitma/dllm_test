from typing import List, Tuple
from scipy.stats import pearsonr
from anarci import anarci
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp

# Add the current directory to path to import gene_number
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parse_anarci import parse_anarci_results, locus_from_gene
from immunomatch_score import main as compute_immunomatch_score
from abnumber import Chain

def find_germline_sequence(Seq: str) -> str:
    """
    Find the germline sequence for a given sequence.
    """
    chain = Chain(Seq, scheme='imgt')
    alignment = chain.find_merged_human_germline()
    alignment_lines = str(alignment).strip().split('\n')
    target_seq = alignment_lines[2]  # germline sequence (alignment)
    return target_seq


def count_mutations(seq: str, germline: str) -> int:
    """
    Count the number of mutations (AA mismatches) between a sequence and its germline.
    Assumes seq and germline are aligned and of equal length.
    """
    if len(seq) != len(germline):
        # Handle sequences of different lengths by taking the minimum length
        min_len = min(len(seq), len(germline))
        seq = seq[:min_len]
        germline = germline[:min_len]
    return sum(aa1 != aa2 for aa1, aa2 in zip(seq, germline))

def compute_vh_vl_mutation_correlation(
    vh_sequences: List[str],
    vh_germlines: List[str],
    vl_sequences: List[str],
    vl_germlines: List[str]
):
    """
    Compute the Pearson correlation between VH and VL mutation counts.
    All lists must be the same length (paired VH and VL for each antibody).
    """
    vh_mut_counts = [count_mutations(seq, gl) for seq, gl in zip(vh_sequences, vh_germlines)]
    vl_mut_counts = [count_mutations(seq, gl) for seq, gl in zip(vl_sequences, vl_germlines)]

    r, p_value = pearsonr(vh_mut_counts, vl_mut_counts)
    return r, p_value, vh_mut_counts, vl_mut_counts


def _anarci_worker(args_tuple):
    """
    Worker function to process a single sequence with ANARCI.
    Used for parallel processing with ProcessPoolExecutor.
    
    Args:
        args_tuple: Tuple of (index, sequence)
        
    Returns:
        Tuple of (index, result_dict, success_flag)
    """
    idx, sequence = args_tuple
    try:
        # Run ANARCI on single sequence
        anarci_input = [(f"seq_{idx}", sequence)]
        results = anarci(anarci_input, scheme='imgt', assign_germline=True)
        
        # Parse results
        parsed_results = parse_anarci_results(results)
        
        if parsed_results and len(parsed_results) > 0:
            result = parsed_results[0]
            result['sequence'] = sequence
            result['index'] = idx
            result['valid'] = result.get('chain') is not None
            return idx, result, True
        else:
            result = {
                'index': idx, 'chain': None, 'v_call': None, 'j_call': None,
                'locus': None, 'valid': False, 'sequence': sequence
            }
            return idx, result, False
    except Exception as e:
        result = {
            'index': idx, 'chain': None, 'v_call': None, 'j_call': None,
            'locus': None, 'valid': False, 'sequence': sequence, 'error': str(e)
        }
        return idx, result, False


def process_sequences_with_anarci_batch(sequences: List[str], n_jobs: int = -1, chunk_size: int = 1000) -> List[dict]:
    """
    Main function to process a list of sequences with ANARCI using parallel processing.
    
    Args:
        sequences: List of antibody sequences to analyze
        n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential)
        chunk_size: Size of chunks for processing
        
    Returns:
        List of dictionaries containing parsed ANARCI results for each sequence
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    from tqdm import tqdm
    
    if not sequences:
        return []
    
    # Create indexed sequence tuples
    indexed_sequences = list(enumerate(sequences))
    
    # Decide whether to use parallel processing
    use_parallel = n_jobs != 1 and len(sequences) > 50
    
    if use_parallel:
        # Use ProcessPoolExecutor for better control
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        results = [None] * len(sequences)
        failed_count = 0
        
        # Process in chunks to control memory
        for chunk_start in range(0, len(sequences), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(sequences))
            chunk_sequences = indexed_sequences[chunk_start:chunk_end]
            
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                mp_context=mp.get_context("spawn")
            ) as executor:
                futures = [executor.submit(_anarci_worker, seq_tuple) for seq_tuple in chunk_sequences]
                
                for future in tqdm(as_completed(futures), 
                                 total=len(futures), 
                                 desc=f"Parsing sequences [{chunk_start}:{chunk_end}]"):
                    idx, result, success = future.result()
                    results[idx] = result
                    if not success:
                        failed_count += 1
        
        if failed_count > 0:
            print(f"Warning: {failed_count} sequences failed ANARCI processing")
            
    else:
        # Sequential processing with progress bar
        results = []
        for seq_tuple in tqdm(indexed_sequences, desc="Parsing sequences"):
            idx, result, success = _anarci_worker(seq_tuple)
            results.append(result)
    
    return results


def compute_validity(df: pd.DataFrame, col_name: str) -> Tuple[pd.DataFrame, float]:
    """
    Compute the validity of the generation results using batch processing with gene_number functions.
    """
    # Get all sequences
    sequences = df[col_name].tolist()
    
    # Process all sequences in batch with ANARCI using gene_number functions
    anarci_results = process_sequences_with_anarci_batch(sequences)
    
    # Extract validity information
    valid_flags = [result['valid'] for result in anarci_results]
    
    # Add validity column to dataframe
    df["valid"] = valid_flags
    valid_rate = sum(valid_flags) / len(valid_flags) if valid_flags else 0.0
    
    return df, valid_rate


def encode_sequences_for_similarity(sequences: List[str]) -> np.ndarray:
    """
    Encode sequences using k-mer representation for cosine similarity calculation.
    Uses 3-mer (triplet) amino acid k-mers as features.
    """
    def get_kmers(sequence: str, k: int = 3) -> List[str]:
        """Extract k-mers from a sequence."""
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    
    # Convert sequences to k-mer strings
    kmer_sequences = [' '.join(get_kmers(seq)) for seq in sequences]
    
    # Use CountVectorizer to create feature vectors
    vectorizer = CountVectorizer()
    feature_matrix = vectorizer.fit_transform(kmer_sequences)
    
    return feature_matrix.toarray()


def compute_diversity(
    generated_sequences: List[str]
) -> Tuple[float, float]:
    """
    Compute the pairwise cosine diversity of 3-mer subsequences.
    
    This matches the method: "we calculated the pairwise cosine diversity of 3-mer 
    subsequences of the paired sequences within each set"
    
    Args:
        generated_sequences: List of generated sequences (should be non-empty sequences)
        
    Returns:
        Tuple of (mean_diversity, max_diversity) - higher values indicate higher diversity
        where diversity = 1 - cosine_similarity
    """
    # Filter out any remaining empty sequences
    valid_sequences = [seq for seq in generated_sequences if seq.strip() != '']
    
    if len(valid_sequences) < 2:
        return 0.0, 0.0
    
    # Encode sequences for similarity calculation using 3-mer representation
    encoded_sequences = encode_sequences_for_similarity(valid_sequences)
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(encoded_sequences)
    
    n_sequences = len(valid_sequences)
    
    # Calculate pairwise diversity for all unique sequence pairs
    pairwise_diversities = []
    
    for i in range(n_sequences):
        for j in range(i + 1, n_sequences):  # Only upper triangle to avoid duplicates
            similarity = similarity_matrix[i, j]
            diversity = 1 - similarity  # Convert similarity to diversity (cosine distance)
            pairwise_diversities.append(diversity)
    
    # Calculate final statistics
    mean_diversity = np.mean(pairwise_diversities) if pairwise_diversities else 0.0
    max_diversity = np.max(pairwise_diversities) if pairwise_diversities else 0.0
    
    return mean_diversity, max_diversity


def _compute_group_diversity(heavy_seq: str, light_seqs: List[str]) -> Tuple[str, float, dict]:
    """
    Helper function to compute diversity for a single heavy chain group.
    Used for parallel processing.
    
    Args:
        heavy_seq: Heavy chain sequence (group identifier)
        light_seqs: List of light chain sequences in this group
        
    Returns:
        Tuple of (heavy_seq, mean_diversity, group_stats_dict)
    """
    if len(light_seqs) < 2:
        return heavy_seq, 0.0, {
            'n_light_chains': len(light_seqs),
            'mean_diversity': 0.0,
            'max_diversity': 0.0
        }
    else:
        mean_div, max_div = compute_diversity(light_seqs)
        return heavy_seq, mean_div, {
            'n_light_chains': len(light_seqs),
            'mean_diversity': mean_div,
            'max_diversity': max_div
        }


def compute_diversity_by_heavy_chain(
    df: pd.DataFrame, 
    heavy_col: str, 
    light_col: str,
    n_jobs: int = -1
) -> Tuple[float, float, dict]:
    """
    Compute diversity by grouping light chains by their corresponding heavy chains.
    For each heavy chain group, compute the diversity of light chains within that group.
    Then compute the overall diversity across all groups.
    Uses parallel processing for improved performance on large datasets.
    
    Args:
        df: DataFrame containing heavy and light chain sequences
        heavy_col: Column name for heavy chain sequences
        light_col: Column name for light chain sequences
        n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential)
        
    Returns:
        Tuple of (overall_mean_diversity, overall_max_diversity, group_stats)
        where group_stats contains per-group diversity statistics
    """
    # Clean sequences
    heavy_sequences = clean_sequences(df[heavy_col].tolist())
    light_sequences = clean_sequences(df[light_col].tolist())
    
    # Group by heavy chain
    groups = {}
    for i, (h_seq, l_seq) in enumerate(zip(heavy_sequences, light_sequences)):
        if h_seq.strip() != '' and l_seq.strip() != '':  # Only include valid pairs
            if h_seq not in groups:
                groups[h_seq] = []
            groups[h_seq].append(l_seq)
    
    if not groups:
        return 0.0, 0.0, {}
    
    # Parallel computation of diversity for each group
    # Use sequential processing if n_jobs=1 or only a few groups
    use_parallel = n_jobs != 1 and len(groups) > 10
    
    if use_parallel:
        # Use multiprocessing backend for CPU-intensive diversity calculations
        backend = 'multiprocessing'
        
        # Parallel processing with progress bar
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_compute_group_diversity)(heavy_seq, light_seqs)
            for heavy_seq, light_seqs in tqdm(groups.items(), desc="Computing diversity per group")
        )
    else:
        # Sequential processing
        results = []
        for heavy_seq, light_seqs in groups.items():
            results.append(_compute_group_diversity(heavy_seq, light_seqs))
    
    # Unpack results
    group_diversities = []
    group_stats = {}
    for heavy_seq, mean_div, stats in results:
        group_diversities.append(mean_div)
        group_stats[heavy_seq] = stats
    
    # Calculate overall statistics
    overall_mean_diversity = np.mean(group_diversities) if group_diversities else 0.0
    overall_max_diversity = np.max(group_diversities) if group_diversities else 0.0
    
    return overall_mean_diversity, overall_max_diversity, group_stats


def compare_sequences_detailed(gen_sequences: List[str], ref_sequences: List[str]) -> dict:
    """
    Compare generated and reference sequences using ANARCI parsing.
    
    Args:
        gen_sequences: List of generated sequences
        ref_sequences: List of reference sequences
        
    Returns:
        Dictionary containing various comparison metrics
    """
    if len(gen_sequences) != len(ref_sequences):
        raise ValueError("Generated and reference sequences must have the same length")
    
    # Clean sequences first
    gen_sequences = clean_sequences(gen_sequences)
    ref_sequences = clean_sequences(ref_sequences)
    
    # Parse both generated and reference sequences
    print("Parsing generated sequences...")
    gen_results = process_sequences_with_anarci_batch(gen_sequences)
    print("Parsing reference sequences...")
    ref_results = process_sequences_with_anarci_batch(ref_sequences)
    
    # Initialize counters
    total_pairs = len(gen_sequences)
    both_valid = 0
    chain_match = 0
    v_gene_match = 0
    j_gene_match = 0
    v_gene_family_match = 0
    j_gene_family_match = 0
    
    detailed_results = []
    
    for i, (gen_result, ref_result) in enumerate(zip(gen_results, ref_results)):
        pair_result = {
            'index': i,
            'gen_valid': gen_result['valid'],
            'ref_valid': ref_result['valid'],
            'gen_chain': gen_result.get('chain'),
            'ref_chain': ref_result.get('chain'),
            'gen_locus': gen_result.get('locus'),
            'ref_locus': ref_result.get('locus'),
            'gen_v_call': gen_result.get('v_call'),
            'ref_v_call': ref_result.get('v_call'),
            'gen_j_call': gen_result.get('j_call'),
            'ref_j_call': ref_result.get('j_call'),
            'gen_length': len(gen_sequences[i]) if i < len(gen_sequences) else 0,
            'ref_length': len(ref_sequences[i]) if i < len(ref_sequences) else 0,
            'chain_match': False,
            'v_gene_match': False,
            'j_gene_match': False,
            'v_gene_family_match': False,
            'j_gene_family_match': False
        }
        
        # Only compare if both sequences are valid
        if gen_result['valid'] and ref_result['valid']:
            both_valid += 1
            
            # Chain type match (H/K/L)
            if gen_result.get('chain') and ref_result.get('chain'):
                if gen_result['chain'] == ref_result['chain']:
                    chain_match += 1
                    pair_result['chain_match'] = True
            
            # Chain type and locus are the same thing, so we don't need separate locus matching
            
            # V gene comparison - check family first, then exact match
            if gen_result.get('v_call') and ref_result.get('v_call'):
                # V gene family match (e.g., IGHV3 vs IGHV3-23)
                gen_v_family = extract_gene_family(gen_result['v_call'])
                ref_v_family = extract_gene_family(ref_result['v_call'])
                if gen_v_family and ref_v_family and gen_v_family == ref_v_family:
                    v_gene_family_match += 1
                    pair_result['v_gene_family_match'] = True
                    
                    # V gene exact match (only check if family matches)
                    if gen_result['v_call'] == ref_result['v_call']:
                        v_gene_match += 1
                        pair_result['v_gene_match'] = True
            
            # J gene comparison - check family first, then exact match
            if gen_result.get('j_call') and ref_result.get('j_call'):
                # J gene family match
                gen_j_family = extract_gene_family(gen_result['j_call'])
                ref_j_family = extract_gene_family(ref_result['j_call'])
                if gen_j_family and ref_j_family and gen_j_family == ref_j_family:
                    j_gene_family_match += 1
                    pair_result['j_gene_family_match'] = True
                    
                    # J gene exact match (only check if family matches)
                    if gen_result['j_call'] == ref_result['j_call']:
                        j_gene_match += 1
                        pair_result['j_gene_match'] = True
        
        detailed_results.append(pair_result)
    
    # Calculate rates
    gen_valid_rate = sum(1 for r in gen_results if r['valid']) / total_pairs
    ref_valid_rate = sum(1 for r in ref_results if r['valid']) / total_pairs
    both_valid_rate = both_valid / total_pairs
    
    # Calculate sequence length statistics
    gen_lengths = [len(seq) for seq in gen_sequences]
    ref_lengths = [len(seq) for seq in ref_sequences]
    
    gen_length_mean = np.mean(gen_lengths)
    gen_length_std = np.std(gen_lengths)
    ref_length_mean = np.mean(ref_lengths)
    ref_length_std = np.std(ref_lengths)
    
    # Calculate match rates in two ways:
    # 1. Among all sequence pairs (including invalid ones as non-matches)
    # 2. Among only valid pairs (for reference)
    
    # Overall match rates (invalid sequences count as non-matches)
    overall_chain_match_rate = chain_match / total_pairs
    overall_v_gene_match_rate = v_gene_match / total_pairs
    overall_j_gene_match_rate = j_gene_match / total_pairs
    overall_v_gene_family_match_rate = v_gene_family_match / total_pairs
    overall_j_gene_family_match_rate = j_gene_family_match / total_pairs
    
    # Match rates among valid pairs only (for comparison)
    if both_valid > 0:
        valid_chain_match_rate = chain_match / both_valid
        valid_v_gene_match_rate = v_gene_match / both_valid
        valid_j_gene_match_rate = j_gene_match / both_valid
        valid_v_gene_family_match_rate = v_gene_family_match / both_valid
        valid_j_gene_family_match_rate = j_gene_family_match / both_valid
    else:
        valid_chain_match_rate = 0.0
        valid_v_gene_match_rate = 0.0
        valid_j_gene_match_rate = 0.0
        valid_v_gene_family_match_rate = 0.0
        valid_j_gene_family_match_rate = 0.0
    
    return {
        'total_pairs': total_pairs,
        'gen_valid_rate': gen_valid_rate,
        'ref_valid_rate': ref_valid_rate,
        'both_valid_rate': both_valid_rate,
        'both_valid_count': both_valid,
        'gen_length_mean': gen_length_mean,
        'gen_length_std': gen_length_std,
        'ref_length_mean': ref_length_mean,
        'ref_length_std': ref_length_std,
        # Overall match rates (including invalid sequences as non-matches)
        'overall_chain_match_rate': overall_chain_match_rate,
        'overall_v_gene_match_rate': overall_v_gene_match_rate,
        'overall_j_gene_match_rate': overall_j_gene_match_rate,
        'overall_v_gene_family_match_rate': overall_v_gene_family_match_rate,
        'overall_j_gene_family_match_rate': overall_j_gene_family_match_rate,
        # Match rates among valid pairs only (for reference)
        'valid_chain_match_rate': valid_chain_match_rate,
        'valid_v_gene_match_rate': valid_v_gene_match_rate,
        'valid_j_gene_match_rate': valid_j_gene_match_rate,
        'valid_v_gene_family_match_rate': valid_v_gene_family_match_rate,
        'valid_j_gene_family_match_rate': valid_j_gene_family_match_rate,
        'detailed_results': detailed_results
    }


def clean_sequences(sequences: List[str]) -> List[str]:
    """
    Clean sequence list by handling missing/invalid sequences.
    
    Args:
        sequences: Raw sequence list from DataFrame
        
    Returns:
        Cleaned sequence list with empty/invalid sequences filtered
    """
    cleaned = []
    for seq in sequences:
        # Handle NaN, None, empty string, or whitespace-only strings
        if pd.isna(seq) or seq is None or str(seq).strip() == '':
            cleaned.append('')  # Keep as empty string for consistency
        else:
            cleaned.append(str(seq).strip())  # Convert to string and strip whitespace
    return cleaned


def extract_gene_family(gene_call: str) -> str:
    """
    Extract gene family from gene call (e.g., IGHV3-23*01 -> IGHV3)
    """
    if not gene_call:
        return None
    
    import re
    # Match pattern like IGHV3, IGKV1, IGLV2, IGHJ4, etc.
    match = re.match(r'(IG[HKL][VDJ]\d+)', gene_call)
    return match.group(1) if match else None


def main(generate_results_file: str, gen_col_name: str, ref_col_name: str = None, 
         detailed_comparison: bool = True, heavy_col_name: str = 'h_sequence',
         expected_count: int = None):
    """
    Compute the evaluation metrics for the generation results.
    
    Args:
        generate_results_file: Path to the CSV file containing generation results
        gen_col_name: Column name for generated sequences
        ref_col_name: Column name for reference sequences (optional)
        detailed_comparison: Whether to perform detailed sequence comparison
        heavy_col_name: Column name for heavy chain sequences
        expected_count: Expected number of light chains per heavy chain (for sampling statistics)
    """
    
    # Read the generation results
    df = pd.read_csv(generate_results_file)

    # Calculate the diversity using pairwise cosine distance, grouped by heavy chain
    generated_sequences = clean_sequences(df[gen_col_name].tolist())
    
    # Filter out empty sequences for diversity calculation
    valid_sequences = [seq for seq in generated_sequences if seq.strip() != '']
    
    print(f"Total sequences: {len(generated_sequences)}, Valid sequences: {len(valid_sequences)}")
    
    # Use the new diversity calculation method that groups by heavy chain
    diversity_mean, diversity_max, group_stats = compute_diversity_by_heavy_chain(
        df, heavy_col_name, gen_col_name
    )
    
    # Print group statistics
    print(f"Number of heavy chain groups: {len(group_stats)}")
    for heavy_seq, stats in list(group_stats.items())[:5]:  # Show first 5 groups
        print(f"  Heavy chain group: {heavy_seq[:20]}... -> {stats['n_light_chains']} light chains, diversity: {stats['mean_diversity']:.4f}")
    if len(group_stats) > 5:
        print(f"  ... and {len(group_stats) - 5} more groups")
    
    results = {
        'diversity_mean': diversity_mean,
        'diversity_max': diversity_max,
        'n_heavy_chain_groups': len(group_stats)
    }

    # Here we evaluate the immunomatch score
    gen_scores = compute_immunomatch_score(
        df_dir=generate_results_file, 
        hseq_col=heavy_col_name,
        lseq_col=gen_col_name,
        eval_split=True,
    )
    gen_immunomatch_mean = np.mean(gen_scores)
    
    # Initialize ref scores and comparison metrics
    ref_immunomatch_mean = None
    gen_better_ratio = None
    
    # If reference sequences are provided, compute their scores and comparison
    if ref_col_name:
        ref_scores = compute_immunomatch_score(
            df_dir=generate_results_file,
            hseq_col=heavy_col_name,
            lseq_col=ref_col_name,
            eval_split=True,
        )
        ref_immunomatch_mean = np.mean(ref_scores)
        
        # Calculate the ratio where gen_score > ref_score
        gen_better_count = sum(1 for gen_score, ref_score in zip(gen_scores, ref_scores) 
                              if gen_score > ref_score)
        gen_better_ratio = gen_better_count / len(gen_scores)
    
    results['gen_immunomatch_mean'] = gen_immunomatch_mean
    if ref_immunomatch_mean is not None:
        results['ref_immunomatch_mean'] = ref_immunomatch_mean
        results['gen_better_ratio'] = gen_better_ratio
    
    # If reference sequences are provided, perform detailed comparison
    if ref_col_name and detailed_comparison:
        print("\n" + "="*50)
        print("Performing detailed sequence comparison...")
        print("="*50)
        
        reference_sequences = clean_sequences(df[ref_col_name].tolist())
        comparison_results = compare_sequences_detailed(generated_sequences, reference_sequences)
        
        # Print detailed results
        print(f"Total sequence pairs: {comparison_results['total_pairs']}")
        print(f"Generated sequences valid rate: {comparison_results['gen_valid_rate']:.4f}")
        print(f"Reference sequences valid rate: {comparison_results['ref_valid_rate']:.4f}")
        print(f"Both sequences valid rate: {comparison_results['both_valid_rate']:.4f}")
        print(f"Both sequences valid count: {comparison_results['both_valid_count']}")
        
        print(f"\nSequence length statistics:")
        print(f"Generated sequences length: {comparison_results['gen_length_mean']:.1f} ± {comparison_results['gen_length_std']:.1f}")
        print(f"Reference sequences length: {comparison_results['ref_length_mean']:.1f} ± {comparison_results['ref_length_std']:.1f}")
        
        print(f"\nOverall match rates (among all {comparison_results['total_pairs']} pairs, invalid = non-match):")
        print(f"Chain type match rate: {comparison_results['overall_chain_match_rate']:.4f}")
        print(f"V gene exact match rate: {comparison_results['overall_v_gene_match_rate']:.4f}")
        print(f"J gene exact match rate: {comparison_results['overall_j_gene_match_rate']:.4f}")
        print(f"V gene family match rate: {comparison_results['overall_v_gene_family_match_rate']:.4f}")
        print(f"J gene family match rate: {comparison_results['overall_j_gene_family_match_rate']:.4f}")
        
        # Add comparison results to return dictionary
        results.update(comparison_results)
    
    print(f"\nSummary - Diversity Mean: {results['diversity_mean']:.4f}, Diversity Max: {results['diversity_max']:.4f}")
    print(f"Gen Immunomatch Mean: {results['gen_immunomatch_mean']:.4f}")
    if ref_col_name:
        if 'ref_immunomatch_mean' in results:
            print(f"Ref Immunomatch Mean: {results['ref_immunomatch_mean']:.4f}")
        if 'gen_better_ratio' in results:
            print(f"Gen > Ref Ratio: {results['gen_better_ratio']:.4f}")
    if ref_col_name and detailed_comparison:
        print(f"Overall Chain Match: {results['overall_chain_match_rate']:.4f}")
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate antibody sequence generation results')
    parser.add_argument('--input_file', '-i', required=True, 
                       help='Path to the CSV file containing generation results')
    parser.add_argument('--gen_col', '-g', required=True,
                       help='Column name for generated sequences')
    parser.add_argument('--ref_col', '-r', default=None,
                       help='Column name for reference sequences (optional)')
    parser.add_argument('--output_file', '-o', default=None,
                       help='Path to save evaluation results (optional)')
    parser.add_argument('--no_detailed_comparison', action='store_true',
                       help='Skip detailed sequence comparison')
    parser.add_argument('--heavy_col', default='h_sequence',
                       help='Column name for heavy chain sequences (for immunomatch score, default: h_sequence)')
    parser.add_argument('--expected_count', type=int, default=None,
                       help='Expected number of light chains per heavy chain (for sampling statistics)')
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output_file is None:
        input_name = args.input_file.replace('.csv', '')
        args.output_file = f"{input_name}_eval.csv"
    
    print(f"Input file: {args.input_file}")
    print(f"Generated sequences column: {args.gen_col}")
    print(f"Reference sequences column: {args.ref_col}")
    print(f"Output file: {args.output_file}")
    print(f"Detailed comparison: {not args.no_detailed_comparison}")
    print(f"Expected count per heavy chain: {args.expected_count}")
    print("-" * 50)
    
    # Run evaluation
    compare_results = main(
        generate_results_file=args.input_file,
        gen_col_name=args.gen_col,
        ref_col_name=args.ref_col,
        detailed_comparison=not args.no_detailed_comparison,
        heavy_col_name=args.heavy_col,
        expected_count=args.expected_count
    )
    
    # Save results (excluding detailed_results)
    save_results = {k: v for k, v in compare_results.items() if k != 'detailed_results'}
    pd.DataFrame([save_results]).to_csv(args.output_file, index=False)
    print(f"Results saved to: {args.output_file}")