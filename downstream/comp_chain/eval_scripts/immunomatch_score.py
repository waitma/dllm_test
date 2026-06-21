from transformers import (
        RoFormerTokenizer,
        pipeline,
        RoFormerForSequenceClassification,
        Trainer,TrainingArguments
    )

import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_curve,roc_auc_score
from matplotlib import pyplot as plt
import seaborn as sns

from functools import partial
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
from abnumber import Chain
import os

# Functions for the calculation of the pairing score for single VH-VL
def preprocess_single_seq(seq):
    """
    Add the gap between adjacent amino acids, so that the model treat each amino acid as an individual token.
    args: input: seq: a string of amino acids
    """
    return " ".join(list(seq))

def pairing_score_single_pair (h_seq,l_seq,model_checkpoint):
    """
    Output the pairing score of a single pair of VH and VL sequences
    args:input:
    h_seq: a gapped sequence outputted from the function preprocess_seq
    l_seq: a gapped sequence outputted from the function preprocess_seq
    model_checkpoint: the checkpoint of the version of the immunoMatch of your interest
    """
    tokenizer = RoFormerTokenizer.from_pretrained(model_checkpoint)
    model=RoFormerForSequenceClassification.from_pretrained(model_checkpoint)
    model_output=model(**tokenizer(h_seq,l_seq,return_tensors="pt",padding="max_length",max_length=256))
    pairing_score=torch.nn.functional.softmax(model_output.logits, dim=1)[0][1].item()
    return pairing_score


# Functions for the calculation of the pairing score for batches of VH-VL

def preprocess_seq(example,hseqcol="input_Hseq",lseqcol="input_Lseq"):
    return {"input_Hseq":" ".join(list(example[hseqcol])), "input_Lseq":" ".join(list(example[lseqcol]))}

def tokenize_function(examples, tokenizer, hseqcol="input_Hseq",lseqcol="input_Lseq",max_length=256,return_tensors="pt"):
    return tokenizer(examples[hseqcol], examples[lseqcol], padding="max_length", truncation=True, max_length=max_length, return_tensors=return_tensors)

def tokenize_the_datasets(df_dir,hseq_col,lseq_col,tokenizer):
    """
    Tokenize the datasets
    args:input:
    df_dir: str, the directory of the dataset
    hseq_col: str, the column name of the heavy chain sequence
    lseq_col: str, the column name of the light chain sequence
    """
    df=pd.read_csv(df_dir)
    datasets=load_dataset("csv", data_files={"test":df_dir})
    tokenized_datasets=datasets.map(partial(preprocess_seq,hseqcol=hseq_col,lseqcol=lseq_col))
    tokenized_datasets=tokenized_datasets.map(partial(tokenize_function,tokenizer=tokenizer),batched=True)
    return df, tokenized_datasets

def pairing_scores_batches(df_dir,hseq_col,lseq_col,model_checkpoint):
    """
    Load the model and make the pairing prediction on batches of sequences
    args:input:
    df_dir: the directory of the csv files holding the sequences of pairs of VH and VL sequences
    hseq_col: the column name of the column of VH sequences
    lseq_col: the column name of the column of VL sequences
    model_checkpoint: the chesck point of the version of ImmunoMatch of your interest
    """

    tokenizer = RoFormerTokenizer.from_pretrained(model_checkpoint,local_files_only=True)
    model=RoFormerForSequenceClassification.from_pretrained(model_checkpoint,local_files_only=True)

    df, tokenized_datasets = tokenize_the_datasets(df_dir, hseq_col, lseq_col, tokenizer)

    device="cuda" if torch.cuda.is_available() else "cpu"
    batch_size=48
    args = TrainingArguments(
    f"tmp",
    save_strategy = "epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    report_to="none"
    )

    trainer = Trainer(
            model,
            args,
            tokenizer=tokenizer,
            )
    pred_result=trainer.predict(tokenized_datasets["test"])

    pairing_scores=torch.nn.functional.softmax(torch.tensor(pred_result.predictions),dim=1)[:,1].tolist()
    df["pairing_scores"]=pairing_scores

    return df

def _get_chain_type_single(seq: str) -> str:
    """
    Helper function to get chain type for a single sequence.
    Used for parallel processing.
    """
    try:
        chain = Chain(seq, scheme="imgt")
        if chain.chain_type == "K":
            return "K"
        elif chain.chain_type == "L":
            return "L"
        else:
            return None
    except Exception:
        return None


def alloc_light_chain_type(df: pd.DataFrame, l_col: str, type_col: str, n_jobs: int = -1):
    """
    Allocate light chain types (K/L) for sequences using parallel processing.
    First checks if chain types are already available in the dataframe.
    
    Args:
        df: DataFrame containing sequences
        l_col: Column name containing light chain sequences
        type_col: Column name to store chain types
        n_jobs: Number of parallel jobs (-1 for all CPUs, 1 for sequential)
        
    Returns:
        Tuple of (updated_df, none_count)
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm
    
    df = df.copy()
    
    # Check if chain types are already available in the dataframe
    # Map l_col to corresponding chain type column
    col_mapping = {
        'raw_l_sequence': 'raw_light_type',
        'gen_l_sequence': 'gen_light_type',
    }
    
    # Get the corresponding chain type column based on l_col
    existing_col = col_mapping.get(l_col)
    
    
    if existing_col is not None and existing_col in df.columns:
        print(f"Found existing chain type column: {existing_col} (mapped from {l_col})")
        # Use existing chain type data
        df[type_col] = df[existing_col]
        
        df[type_col] = (
            df[type_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(
                {
                    "k": "K",
                    "kappa": "K",
                    "igk": "K",
                    "l": "L",
                    "lambda": "L",
                    "igl": "L",
                    "nan": None,
                    "none": None,
                    "": None,
                }
            )
        )

        none_count = df[type_col].isna().sum()
        
        print(f"Using existing chain types. None count: {none_count}")
        return df, none_count
    
    # If no existing chain type column, perform analysis
    print("No existing chain type column found. Performing chain type analysis...")
    sequences = df[l_col].tolist()
    
    # Decide whether to use parallel processing
    use_parallel = n_jobs != 1 and len(sequences) > 50
    
    if use_parallel:
        # Parallel processing with progress bar
        types = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(_get_chain_type_single)(seq)
            for seq in tqdm(sequences, desc="Determining chain types")
        )
    else:
        # Sequential processing with progress bar
        types = [
            _get_chain_type_single(seq)
            for seq in tqdm(sequences, desc="Determining chain types")
        ]
    
    # Count None values
    none_count = sum(1 for t in types if t is None)
    
    # Create updated dataframe
    df[type_col] = types
    
    return df, none_count 
    
    
def debug_main(
    eval_split=True,
    df_dir="/mnt/nas-new/home/yangnianzu/jm/bjzgc_zyh/data/baseline_generate_results/lichen_results.csv",
    hseq_col='h_sequence',
    lseq_col='raw_l_sequence',  # "gen_l_sequence" / "raw_l_sequence":
    type_col='type',
    draw=False,
    
):  
    if not eval_split:
        immunoMatch_version = 'fraternalilab/immunomatch' 
        # @param ["fraternalilab/immunomatch", "fraternalilab/immunomatch-kappa", "fraternalilab/immunomatch-lambda"]

        result_df = pairing_scores_batches(
            df_dir, 
            hseq_col,
            lseq_col,
            model_checkpoint=immunoMatch_version
        )


    else:
        # Use local paths for models
        immuno_kappa = '/mnt/nas-new/home/yangnianzu/jm/bjzgc_zyh/AirGen-Dev/ckpt/immun/immunomatch-kappa'
        immuno_lambda = '/mnt/nas-new/home/yangnianzu/jm/bjzgc_zyh/AirGen-Dev/ckpt/immun/immunomatch-lambda'
        
        data_df = pd.read_csv(df_dir)
        data_df["_immunomatch_row_id"] = range(len(data_df))
        data_df, none_count = alloc_light_chain_type(data_df, lseq_col, type_col, n_jobs=-1)
        if none_count > 0:
            print(f"Warning: {none_count} light chain sequences cannot be identified by abnumber.")
        
        # Only process data with identifiable types, preserve original indices
        valid_data = data_df[data_df[type_col].notna()].copy()
        kappa_data = valid_data[valid_data[type_col] == "K"].copy()
        lambda_data = valid_data[valid_data[type_col] == "L"].copy()
        
        print(f'Kappa dataset: {len(kappa_data)}')
        print(f'Lambda dataset: {len(lambda_data)}')
        print(f'Skipped (None type): {none_count}')
        
        # Initialize result DataFrame, maintaining same structure as original data
        result_df = data_df.copy()
        result_df['pairing_scores'] = None  # Initialize as None
        
        # Process kappa data
        if len(kappa_data) > 0:
            base, ext = os.path.splitext(df_dir)
            kappa_csv = f"{base}_{lseq_col}_kappa{ext or '.csv'}"
            kappa_data.to_csv(kappa_csv, index=False)
            
            k_pairing_batch_result = pairing_scores_batches(
                kappa_csv,
                hseq_col,
                lseq_col, 
                immuno_kappa,
            )
            kappa_mapped = 0
            for _, row in k_pairing_batch_result.iterrows():
                score = row['pairing_scores']
                mask = result_df["_immunomatch_row_id"] == row["_immunomatch_row_id"]
                if mask.any():
                    result_df.loc[mask, 'pairing_scores'] = score
                    kappa_mapped += 1
                else:
                    print(f"Warning: Could not find matching kappa row_id: {row['_immunomatch_row_id']}")
            print(f"Successfully mapped {kappa_mapped}/{len(k_pairing_batch_result)} kappa scores")
        
        # Process lambda data
        if len(lambda_data) > 0:
            base, ext = os.path.splitext(df_dir)
            lambda_csv = f"{base}_{lseq_col}_lambda{ext or '.csv'}"
            lambda_data.to_csv(lambda_csv, index=False)
            
            l_pairing_batch_result = pairing_scores_batches(
                lambda_csv,
                hseq_col,
                lseq_col, 
                immuno_lambda,
            )
            lambda_mapped = 0
            for _, row in l_pairing_batch_result.iterrows():
                score = row['pairing_scores']
                mask = result_df["_immunomatch_row_id"] == row["_immunomatch_row_id"]
                if mask.any():
                    result_df.loc[mask, 'pairing_scores'] = score
                    lambda_mapped += 1
                else:
                    print(f"Warning: Could not find matching lambda row_id: {row['_immunomatch_row_id']}")
            print(f"Successfully mapped {lambda_mapped}/{len(l_pairing_batch_result)} lambda scores")
    


    
    # Print statistics
    result_df['pairing_scores'] = result_df['pairing_scores'].apply(lambda x: x if x is not None else 0)
    valid_scores = result_df['pairing_scores']
    return valid_scores


def main(
    df_dir,
    hseq_col='h_sequence',
    lseq_col='raw_l_sequence',  # "gen_l_sequence" / "raw_l_sequence":
    eval_split=True,
    type_col='type',
    draw=False,
    
):  
    if not eval_split:
        immunoMatch_version = 'fraternalilab/immunomatch' 
        # @param ["fraternalilab/immunomatch", "fraternalilab/immunomatch-kappa", "fraternalilab/immunomatch-lambda"]

        result_df = pairing_scores_batches(
            df_dir, 
            hseq_col,
            lseq_col,
            model_checkpoint=immunoMatch_version
        )


    else:
        # Use local paths for models
        immuno_kappa = '/vepfs-mlp2/mlp-public/zhuyiheng/hub/checkpoints/immunomatch-kappa'
        immuno_lambda = '/vepfs-mlp2/mlp-public/zhuyiheng/hub/checkpoints/immunomatch-lambda'
        
        data_df = pd.read_csv(df_dir)
        data_df["_immunomatch_row_id"] = range(len(data_df))
        data_df, none_count = alloc_light_chain_type(data_df, lseq_col, type_col, n_jobs=-1)
        if none_count > 0:
            print(f"Warning: {none_count} light chain sequences cannot be identified by abnumber.")
        
        # Only process data with identifiable types, preserve original indices
        valid_data = data_df[data_df[type_col].notna()].copy()
        kappa_data = valid_data[valid_data[type_col] == "K"].copy()
        lambda_data = valid_data[valid_data[type_col] == "L"].copy()
        
        print(f'Kappa dataset: {len(kappa_data)}')
        print(f'Lambda dataset: {len(lambda_data)}')
        print(f'Skipped (None type): {none_count}')
        
        # Initialize result DataFrame, maintaining same structure as original data
        result_df = data_df.copy()
        result_df['pairing_scores'] = None  # Initialize as None
        
        # Process kappa data
        if len(kappa_data) > 0:
            base, ext = os.path.splitext(df_dir)
            kappa_csv = f"{base}_{lseq_col}_kappa{ext or '.csv'}"
            kappa_data.to_csv(kappa_csv, index=False)
            
            k_pairing_batch_result = pairing_scores_batches(
                kappa_csv,
                hseq_col,
                lseq_col, 
                immuno_kappa,
            )
            kappa_mapped = 0
            for _, row in k_pairing_batch_result.iterrows():
                score = row['pairing_scores']
                mask = result_df["_immunomatch_row_id"] == row["_immunomatch_row_id"]
                if mask.any():
                    result_df.loc[mask, 'pairing_scores'] = score
                    kappa_mapped += 1
                else:
                    print(f"Warning: Could not find matching kappa row_id: {row['_immunomatch_row_id']}")
            print(f"Successfully mapped {kappa_mapped}/{len(k_pairing_batch_result)} kappa scores")
        
        # Process lambda data
        if len(lambda_data) > 0:
            base, ext = os.path.splitext(df_dir)
            lambda_csv = f"{base}_{lseq_col}_lambda{ext or '.csv'}"
            lambda_data.to_csv(lambda_csv, index=False)
            
            l_pairing_batch_result = pairing_scores_batches(
                lambda_csv,
                hseq_col,
                lseq_col, 
                immuno_lambda,
            )
            lambda_mapped = 0
            for _, row in l_pairing_batch_result.iterrows():
                score = row['pairing_scores']
                mask = result_df["_immunomatch_row_id"] == row["_immunomatch_row_id"]
                if mask.any():
                    result_df.loc[mask, 'pairing_scores'] = score
                    lambda_mapped += 1
                else:
                    print(f"Warning: Could not find matching lambda row_id: {row['_immunomatch_row_id']}")
            print(f"Successfully mapped {lambda_mapped}/{len(l_pairing_batch_result)} lambda scores")
    


    
    # Print statistics
    result_df['pairing_scores'] = result_df['pairing_scores'].apply(lambda x: x if x is not None else 0)
    valid_scores = result_df['pairing_scores']
    return valid_scores
    # if len(valid_scores) > 0:
    #     print(f"Valid pairing scores: {len(valid_scores)}/{len(result_df)}")
    #     print(f"Mean pairing score: {valid_scores.mean():.4f}")
    #     print(f"Std pairing score: {valid_scores.std():.4f}")
    #     print(f"Min pairing score: {valid_scores.min():.4f}")
    #     print(f"Max pairing score: {valid_scores.max():.4f}")
    # else:
    #     print("No valid pairing scores computed!")
    
   
    # sns.kdeplot(result_df, x="pairing_scores")

if __name__ == "__main__":
    main()
