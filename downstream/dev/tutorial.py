from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/vepfs-mlp2/mlp-public/zhuyiheng/hub/checkpoints/p-IgGen"

for target in ["Titer", "HIC", "PR_CHO", "Tm2", 'AC-SINS_pH7.4']:
    print(f"Evaluating for {target}")
    
    df = pd.read_csv("downstream/dev/GDPa1_v1.2_20250814.csv")
    df = df.dropna(subset=[target])

    # Tokenize the sequences
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Paired sequence handling: Concatenate heavy and light chains and add beginning ("1") and end ("2") tokens
    # (e.g. ["EVQLV...", "DIQMT..."] -> "1E V Q L V ... D I Q M T ... 2")
    sequences = [
        "1" + " ".join(heavy) + " ".join(light) + "2"
        for heavy, light in zip(
            df["vh_protein_sequence"],
            df["vl_protein_sequence"],
        )
    ]

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Takes about 60 seconds for 242 sequences on my CPU, and 1.1s on GPU
    batch_size = 16
    mean_pooled_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch = tokenizer(sequences[i:i+batch_size], return_tensors="pt", padding=True, truncation=True)
        outputs = model(batch["input_ids"].to(device), output_hidden_states=True)
        embeddings = outputs["hidden_states"][-1].detach().cpu().numpy()
        mean_pooled_embeddings.append(embeddings.mean(axis=1))
    mean_pooled_embeddings = np.concatenate(mean_pooled_embeddings)

    # Train a linear regression on these
    fold_col = "hierarchical_cluster_IgG_isotype_stratified_fold"
    X = mean_pooled_embeddings
    y = df[target].to_numpy(dtype=float)

    # sanity check
    assert len(X) == len(df) == len(y)

    fold_values = df[fold_col].to_numpy()
    unique_folds = [f for f in np.unique(fold_values) if f == f]  # drop NaN

    per_fold_stats = []
    y_pred_all = np.full(len(df), np.nan)   # align with df rows
    y_true_all = np.full(len(df), np.nan)   # optional, for plotting/metrics

    for f in unique_folds:
        test_idx = np.where(fold_values == f)[0]
        train_idx = np.where(fold_values != f)[0]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        lm = Ridge()
        lm.fit(X_train, y_train)
        y_pred = lm.predict(X_test)

        # write back into the positions of df
        y_pred_all[test_idx] = y_pred
        y_true_all[test_idx] = y_test

        rho = spearmanr(y_test, y_pred).statistic
        per_fold_stats.append((int(f), rho, len(y_test)))

    # Overall metric across all rows that participated in CV
    mask = ~np.isnan(y_true_all)
    overall_rho = spearmanr(y_true_all[mask], y_pred_all[mask]).statistic

    print("Fold\tN\tSpearman_rho")
    for f, rho, n in per_fold_stats:
        print(f"{f}\t{n}\t{rho:.4f}")
    print(f"Overall (all folds)\t{mask.sum()}\t{overall_rho:.4f}")

