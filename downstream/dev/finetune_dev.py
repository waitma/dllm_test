from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, PredefinedSplit
from sklearn.preprocessing import PowerTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import ChainPaddingCollator
from downstream.embeddings import OphiuchusEmbeddingModel
from dllm.pipelines.bioseq import Esm2ProteinTokenizer, ophiuchus_ab_checkpoint_path


class DevDataset(Dataset):
    def __init__(self, csv_path: str, target_col: str):
        df = pd.read_csv(csv_path)
        df = df[df[target_col].notna()]
        self.heavy = df["vh_protein_sequence"].astype(str).str.replace("J", "L").tolist()
        self.light = df["vl_protein_sequence"].astype(str).str.replace("J", "L").tolist()
        self.target = df[target_col].tolist()
        self.fold = df["hierarchical_cluster_IgG_isotype_stratified_fold"].tolist()

    def __len__(self) -> int:
        return len(self.heavy)

    def __getitem__(self, index: int):
        return self.heavy[index], self.light[index], self.target[index], self.fold[index]


def extract_embeddings(model, loader, device):
    features = []
    labels = []
    folds = []
    collator = ChainPaddingCollator(tokenizer=Esm2ProteinTokenizer())
    model.eval()
    with torch.no_grad():
        for heavy, light, target, fold in tqdm(loader, desc="embeddings"):
            input_ids, chain_ids, _ = collator.stack_chains(list(heavy), list(light))
            embedding = model(input_ids.to(device), chain_ids.to(device))
            features.append(embedding.cpu().numpy())
            labels.append(np.asarray(target, dtype=np.float32))
            folds.append(np.asarray(fold, dtype=np.int64))
    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(folds, axis=0),
    )


def run_cv_regression(x, y, folds):
    transformer = PowerTransformer()
    x_t = transformer.fit_transform(x)
    y = SimpleImputer(strategy="mean").fit_transform(y.reshape(-1, 1)).ravel()
    cv = PredefinedSplit(folds)
    search = GridSearchCV(
        Ridge(),
        param_grid={"alpha": np.logspace(-3, 3, 13)},
        cv=cv,
        scoring="r2",
    )
    search.fit(x_t, y)
    pred = search.predict(x_t)
    return {
        "cv_r2": float(r2_score(y, pred)),
        "spearman": float(scipy.stats.spearmanr(y, pred).statistic),
        "alpha": float(search.best_params_["alpha"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Developability regression with Ophiuchus embeddings.")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--target-col", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=str(ophiuchus_ab_checkpoint_path()))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    model = OphiuchusEmbeddingModel(checkpoint_path=args.checkpoint_path, device=device)
    loader = DataLoader(DevDataset(args.csv_path, args.target_col), batch_size=args.batch_size, shuffle=False)
    x, y, folds = extract_embeddings(model, loader, device)
    print(run_cv_regression(x, y, folds))


if __name__ == "__main__":
    main()
