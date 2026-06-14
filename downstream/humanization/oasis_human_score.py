#!/usr/bin/env python3
"""
Convert CSV file with generated sequences to FASTA format and run BioPhi OASis scoring.
Author: [your name]
Date: 2025-10-25
"""

import subprocess
import os
import sys
import pandas as pd
from pathlib import Path


def csv_to_fasta(csv_path, fasta_path):
    """
    Convert CSV file with generated sequences to FASTA format.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing generated sequences
    fasta_path : str
        Path for the output FASTA file
        
    Returns
    -------
    int
        Number of sequences written to FASTA file
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = ['pdb_id', 'variant_idx', 'generated_heavy', 'generated_light']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create FASTA file
        with open(fasta_path, 'w') as f:
            for idx, row in df.iterrows():
                # Write heavy chain sequence with _VH suffix
                heavy_seq_id = f">{row['pdb_id']}_{row['variant_idx']}_VH"
                heavy_seq = row['generated_heavy']
                f.write(f"{heavy_seq_id}\n{heavy_seq}\n")
                
                # Write light chain sequence with _VL suffix
                light_seq_id = f">{row['pdb_id']}_{row['variant_idx']}_VL"
                light_seq = row['generated_light']
                f.write(f"{light_seq_id}\n{light_seq}\n")
        
        print(f"Converted {len(df)} sequences to FASTA format: {fasta_path}")
        return len(df) * 2  # Heavy and light chains
        
    except Exception as e:
        print(f"Error converting CSV to FASTA: {e}")
        return 0


def run_oasis(fasta_path, oasis_db_path, output_path="oasis.xlsx", biophi_cmd="biophi"):
    """
    Run BioPhi OASis humanness scoring on a FASTA file.

    Parameters
    ----------
    fasta_path : str
        Path to the FASTA file (e.g., "mabs.fa")
    oasis_db_path : str
        Path to the OASis database file (e.g., "OASis_9mers_v1.db")
    output_path : str, optional
        Path for the output Excel file (default: "oasis.xlsx")
    biophi_cmd : str, optional
        The base BioPhi command (default: "biophi")

    Returns
    -------
    int
        The return code from the subprocess call
    """

    fasta_path = Path(fasta_path).resolve()
    oasis_db_path = Path(oasis_db_path).resolve()
    output_path = Path(output_path).resolve()

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    if not oasis_db_path.exists():
        raise FileNotFoundError(f"OASis DB not found: {oasis_db_path}")

    cmd = [
        biophi_cmd, "oasis", str(fasta_path),
        "--oasis-db", str(oasis_db_path),
        "--output", str(output_path)
    ]

    print("Running BioPhi OASis...")
    print("Command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"Done! Results saved to: {output_path}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"OASis run failed with return code {e.returncode}")
        return e.returncode


def main():
    """
    Main function to convert CSV to FASTA and run OASis scoring.
    """
    if len(sys.argv) < 3:
        print("Usage: python oasis_human_score.py <csv_path> <oasis_db_path> [output.xlsx]")
        print("Example: python oasis_human_score.py task_output.csv OASis_9mers_v1.db oasis_scores.xlsx")
        sys.exit(1)

    csv_file = sys.argv[1]
    oasis_db = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else "oasis_scores.xlsx"
    
    # Generate FASTA file name from CSV file name
    fasta_file = csv_file.replace('.csv', '_sequences.fasta')
    
    print(f"Converting CSV to FASTA: {csv_file} -> {fasta_file}")
    num_sequences = csv_to_fasta(csv_file, fasta_file)
    
    if num_sequences == 0:
        print("Failed to convert CSV to FASTA. Exiting.")
        sys.exit(1)
    
    print(f"Running OASis scoring on {num_sequences} sequences...")
    return_code = run_oasis(fasta_file, oasis_db, output)
    
    if return_code == 0:
        print(f"OASis scoring completed successfully! Results saved to: {output}")
        
        # Read and display OASis Identity scores
        try:
            df = pd.read_excel(output)
            print("\n" + "="*60)
            print("OASis Scores Summary:")
            print("="*60)
            
            if 'OASis Identity' in df.columns:
                mean_identity = df['OASis Identity'].mean()
                std_identity = df['OASis Identity'].std()
                print(f"\nOASis Identity Score:")
                print(f"  Mean: {mean_identity:.4f}")
                print(f"  Std: {std_identity:.4f}")
                print(f"  Range: [{df['OASis Identity'].min():.4f}, {df['OASis Identity'].max():.4f}]")
                print(f"  Total sequences: {len(df)}")
            else:
                print(f"Available columns: {df.columns.tolist()}")
                print(f"Total sequences: {len(df)}")
            
            print("="*60)
            
        except Exception as e:
            print(f"Warning: Could not read OASis results: {e}")
    else:
        print(f"OASis scoring failed with return code: {return_code}")
        sys.exit(return_code)


if __name__ == "__main__":
    main()