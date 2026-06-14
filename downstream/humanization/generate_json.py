#!/usr/bin/env python3
"""
Generate JSON file for structure prediction from CSV file with generated sequences.
Author: [your name]
Date: 2025-10-25
"""

import json
import pandas as pd
import sys
from pathlib import Path


def parse_pdb_chain_pairs(csv_path):
    """
    Read CSV file with PDB ID and chain pairs.
    Format: pdb_id,chain_pair (e.g., "8tq8,H-L")
    Returns: {pdb_id: [chain1, chain2]} dictionary
    """
    df = pd.read_csv(csv_path, header=None, names=["pdb_id", "chain_pair"])
    pdb_to_chains = {}
    
    for _, row in df.iterrows():
        pdb_id = str(row["pdb_id"]).strip()
        chain_pair = str(row["chain_pair"]).strip()
        # Split "H-L" -> ["H", "L"]
        chains = [c.strip() for c in chain_pair.split("-") if c.strip()]
        pdb_to_chains[pdb_id] = chains
    
    return pdb_to_chains


def csv_to_json(csv_path, json_path, chain_csv_path, model_seed=42):
    """
    Convert CSV file with generated sequences to JSON format for structure prediction.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing generated sequences
    json_path : str
        Path for the output JSON file
    chain_csv_path : str
        Path to CSV file with PDB ID and chain pairs
    model_seed : int, optional
        Model seed for structure prediction (default: 42)
        
    Returns
    -------
    int
        Number of entries written to JSON file
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = ['pdb_id', 'variant_idx', 'generated_heavy', 'generated_light']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Read chain pairs
        pdb_to_chains = parse_pdb_chain_pairs(chain_csv_path)
        
        # Group by original_idx to create entries
        json_data = []
        
        for original_idx in df['pdb_id'].unique():
            # Get all variants for this original sequence
            variants = df[df['pdb_id'] == original_idx]
            
            # Get chain IDs for this PDB
            if original_idx not in pdb_to_chains:
                print(f"Warning: No chain information found for {original_idx}, using default H-L")
                heavy_chain_id = "H"
                light_chain_id = "L"
            else:
                chains = pdb_to_chains[original_idx]
                if len(chains) >= 2:
                    heavy_chain_id = chains[0]
                    light_chain_id = chains[1]
                else:
                    print(f"Warning: Insufficient chain information for {original_idx}, using default H-L")
                    heavy_chain_id = "H"
                    light_chain_id = "L"
            
            # Create entry for each variant
            for _, row in variants.iterrows():
                entry = {
                    "name": f"{original_idx}_{heavy_chain_id}_{light_chain_id}_{row['variant_idx']}",
                    "modelSeeds": [model_seed],
                    "sequences": [
                        {
                            "proteinChain": {
                                "id": heavy_chain_id,
                                "sequence": row['generated_heavy'],
                                "count": 1
                            }
                        },
                        {
                            "proteinChain": {
                                "id": light_chain_id, 
                                "sequence": row['generated_light'],
                                "count": 1
                            }
                        }
                    ],
                    "dialect": "alphafold3",
                    "version": 1
                }
                json_data.append(entry)
        
        # Write JSON file
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        print(f"Generated {len(json_data)} entries for structure prediction: {json_path}")
        return len(json_data)
        
    except Exception as e:
        print(f"Error converting CSV to JSON: {e}")
        return 0


def main():
    """
    Main function to convert CSV to JSON for structure prediction.
    """
    if len(sys.argv) < 3:
        print("Usage: python generate_json.py <csv_path> <chain_csv_path> [json_path] [model_seed]")
        print("Example: python generate_json.py task_output.csv test_chains.csv structure_prediction.json 42")
        sys.exit(1)

    csv_file = sys.argv[1]
    chain_csv_file = sys.argv[2]
    json_file = sys.argv[3] if len(sys.argv) > 3 else "structure_prediction.json"
    model_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    
    print(f"Converting CSV to JSON: {csv_file} -> {json_file}")
    print(f"Using chain information from: {chain_csv_file}")
    print(f"Using model seed: {model_seed}")
    
    num_entries = csv_to_json(csv_file, json_file, chain_csv_file, model_seed)
    
    if num_entries == 0:
        print("Failed to convert CSV to JSON. Exiting.")
        sys.exit(1)
    
    print(f"Successfully generated {num_entries} entries for structure prediction!")


if __name__ == "__main__":
    main()
