#!/usr/bin/env python3
"""
Compare abnumber chain type detection with CSV l_locus field.
Author: [your name]
Date: 2025-01-25
"""

import argparse
import pandas as pd
from pathlib import Path
from abnumber import Chain


def detect_chain_type_with_abnumber(sequence):
    """
    Use abnumber to detect chain type (K or L) from sequence.
    
    Parameters
    ----------
    sequence : str
        The light chain sequence
        
    Returns
    -------
    str or None
        'K' for kappa, 'L' for lambda, None if cannot determine
    """
    try:
        # Try to parse as light chain
        chain = Chain(sequence, scheme='imgt')
        
        # Check if it's kappa or lambda
        if hasattr(chain, 'chain_type'):
            chain_type = chain.chain_type
            if chain_type == 'K':
                return 'K'
            elif chain_type == 'L':
                return 'L'
        

    except Exception as e:
        print(f"Error parsing sequence with abnumber: {e}")
        return None


def compare_chain_types(csv_path):
    """
    Compare abnumber detected chain types with CSV l_locus field.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing antibody sequences
        
    Returns
    -------
    pd.DataFrame
        DataFrame with comparison results
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Check required columns
    if 'l_sequence' not in df.columns:
        raise ValueError("CSV file must contain 'l_sequence' column")
    if 'l_locus' not in df.columns:
        raise ValueError("CSV file must contain 'l_locus' column")
    
    print(f"Reading CSV file: {csv_path}")
    print(f"Total sequences: {len(df)}")
    
    # Initialize results
    results = []
    
    for idx, row in df.iterrows():
        l_sequence = row['l_sequence']
        l_locus_csv = row['l_locus']
        
        # Detect chain type with abnumber
        l_locus_detected = detect_chain_type_with_abnumber(l_sequence)
        
        # Determine if they match
        match = (l_locus_detected == l_locus_csv)
        
        results.append({
            'index': idx,
            'l_sequence': l_sequence,
            'l_locus_csv': l_locus_csv,
            'l_locus_detected': l_locus_detected,
            'match': match
        })
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} sequences...")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    total = len(results_df)
    matches = results_df['match'].sum()
    accuracy = matches / total if total > 0 else 0
    
    print("\n" + "="*60)
    print("Chain Type Comparison Results:")
    print("="*60)
    print(f"Total sequences: {total}")
    print(f"Matched: {matches}")
    print(f"Accuracy: {accuracy:.2%}")
    print("="*60)
    
    # Show mismatch cases
    mismatches = results_df[~results_df['match']]
    if len(mismatches) > 0:
        print(f"\nMismatches ({len(mismatches)}):")
        print("-"*60)
        for idx, row in mismatches.iterrows():
            print(f"Row {row['index']}:")
            print(f"  CSV l_locus: {row['l_locus_csv']}")
            print(f"  Detected: {row['l_locus_detected']}")
            print(f"  Sequence: {row['l_sequence'][:50]}...")
            print()
    else:
        print("\nAll sequences matched!")
    
    return results_df


def main():
    """
    Main function to run chain type comparison.
    """
    parser = argparse.ArgumentParser(
        description='Compare abnumber chain type detection with CSV l_locus field'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        help='Path to the CSV file containing antibody sequences'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: input_file_chain_type_comparison.csv)'
    )
    
    args = parser.parse_args()
    
    csv_path = args.csv_path
    output_path = args.output
    
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    if output_path is None:
        output_path = csv_path.replace('.csv', '_chain_type_comparison.csv')
    
    try:
        results_df = compare_chain_types(csv_path)
        
        # Save results to CSV
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
