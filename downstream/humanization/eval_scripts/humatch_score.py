import pandas as pd
import subprocess
import tempfile
import os
from typing import List, Tuple


def prepare_humatch_input(
    heavy_sequences: List[str],
    light_sequences: List[str],
    species_list: List[str] = None
) -> pd.DataFrame:
    """
    Prepare input DataFrame for Humatch-classify.
    
    Args:
        heavy_sequences: List of heavy chain sequences
        light_sequences: List of light chain sequences
        species_list: List of species labels (optional)
        
    Returns:
        DataFrame with columns: is_human, heavy, light
    """
    if len(heavy_sequences) != len(light_sequences):
        raise ValueError("Heavy and light sequences must have the same length")
    
    # Determine is_human based on species
    if species_list is None:
        is_human = [1] * len(heavy_sequences)
    else:
        is_human = [1 if str(sp).lower() == 'human' else 0 
                   for sp in species_list]
    
    df = pd.DataFrame({
        'is_human': is_human,
        'heavy': heavy_sequences,
        'light': light_sequences
    })
    
    return df


def run_humatch_classify(
    input_df: pd.DataFrame,
    output_file: str = None
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Run Humatch-classify on the input DataFrame.
    
    Args:
        input_df: DataFrame with columns: is_human, heavy, light
        output_file: Optional output file path for Humatch results
        
    Returns:
        Tuple of (humatch_results_df, cnn_p_scores)
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_input:
        input_df.to_csv(tmp_input.name, index=False)
        tmp_input_path = tmp_input.name
    
    try:
        if output_file is None:
            tmp_output = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            output_file = tmp_output.name
            tmp_output.close()
        
        # Run Humatch-classify command
        cmd = [
            'Humatch-classify',
            '-i', tmp_input_path,
            '--vh_col', 'heavy',
            '--vl_col', 'light',
            '-o', output_file
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Read results
        result_df = pd.read_csv(output_file)
        
        # Extract CNN_P scores
        if 'CNN_P' in result_df.columns:
            cnn_p_scores = result_df['CNN_P'].tolist()
        else:
            raise ValueError("CNN_P column not found in Humatch output")
        
        return result_df, cnn_p_scores
        
    finally:
        # Clean up temporary input file
        if os.path.exists(tmp_input_path):
            os.remove(tmp_input_path)


def compute_humatch_score(
    df_dir: str,
    hseq_col: str = 'h_sequence',
    lseq_col: str = 'l_sequence',
    species_col: str = 'species',
    eval_split: bool = False
) -> List[float]:
    """
    Compute Humatch CNN_P scores for antibody pairs.
    
    Args:
        df_dir: Path to CSV file containing sequences
        hseq_col: Column name for heavy chain sequences
        lseq_col: Column name for light chain sequences
        species_col: Column name for species labels
        eval_split: Whether this is for evaluation (not used, kept for consistency)
        
    Returns:
        List of CNN_P scores
    """
    # Read input data
    df = pd.read_csv(df_dir)
    
    # Extract sequences
    heavy_sequences = df[hseq_col].tolist()
    light_sequences = df[lseq_col].tolist()
    
    # Extract species if available
    if species_col in df.columns:
        species_list = df[species_col].tolist()
    else:
        species_list = None
    
    # Prepare Humatch input
    humatch_input = prepare_humatch_input(
        heavy_sequences,
        light_sequences,
        species_list
    )
    
    # Run Humatch-classify
    _, cnn_p_scores = run_humatch_classify(humatch_input)
    
    return cnn_p_scores


def main(
    df_dir: str,
    hseq_col: str = 'h_sequence',
    lseq_col: str = 'l_sequence',
    species_col: str = 'species',
    output_file: str = None
) -> List[float]:
    """
    Main function to compute Humatch scores.
    
    Args:
        df_dir: Path to CSV file
        hseq_col: Heavy chain column name
        lseq_col: Light chain column name
        species_col: Species column name
        output_file: Optional output file for full Humatch results
        
    Returns:
        List of CNN_P scores
    """
    return compute_humatch_score(
        df_dir=df_dir,
        hseq_col=hseq_col,
        lseq_col=lseq_col,
        species_col=species_col
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute Humatch CNN_P scores')
    parser.add_argument('--input_file', '-i', required=True,
                       help='Path to CSV file containing sequences')
    parser.add_argument('--heavy_col', default='h_sequence',
                       help='Column name for heavy chain sequences')
    parser.add_argument('--light_col', default='l_sequence',
                       help='Column name for light chain sequences')
    parser.add_argument('--species_col', default='species',
                       help='Column name for species labels')
    parser.add_argument('--output_file', '-o', default=None,
                       help='Output file for full Humatch results')
    
    args = parser.parse_args()
    
    scores = main(
        df_dir=args.input_file,
        hseq_col=args.heavy_col,
        lseq_col=args.light_col,
        species_col=args.species_col,
        output_file=args.output_file
    )
    
    print(f"Computed {len(scores)} Humatch CNN_P scores")
    print(f"Mean CNN_P: {sum(scores)/len(scores):.4f}")
    print(f"Min CNN_P: {min(scores):.4f}, Max CNN_P: {max(scores):.4f}")

