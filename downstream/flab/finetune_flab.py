from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import PowerTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import ChainPaddingCollator
from downstream.embeddings import OphiuchusEmbeddingModel
from dllm.pipelines.bioseq import Esm2ProteinTokenizer, ophiuchus_ab_checkpoint_path


class PropertyDataset(Dataset):
    def __init__(self, csv_path: str, heavy_col: str, light_col: str, target_col: str):
        data = pd.read_csv(csv_path)
        self.heavy = data[heavy_col].astype(str).str.replace("J", "L").tolist()
        self.light = data[light_col].astype(str).str.replace("J", "L").tolist()
        self.target = data[target_col].tolist()

    def __len__(self) -> int:
        return len(self.heavy)

    def __getitem__(self, index: int):
        return self.heavy[index], self.light[index], self.target[index]


def extract_embeddings(model, loader, device):
    features = []
    labels = []
    collator = ChainPaddingCollator(tokenizer=Esm2ProteinTokenizer())
    model.eval()
    with torch.no_grad():
        for heavy, light, target in tqdm(loader, desc="embeddings"):
            input_ids, chain_ids, _ = collator.stack_chains(list(heavy), list(light))
            embedding = model(input_ids.to(device), chain_ids.to(device))
            features.append(embedding.cpu().numpy())
            labels.append(np.asarray(target, dtype=np.float32))
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def run_regression(x_train, y_train, x_test, y_test):
    transformer = PowerTransformer()
    x_train_t = transformer.fit_transform(x_train)
    x_test_t = transformer.transform(x_test)
    search = GridSearchCV(
        Ridge(),
        param_grid={"alpha": np.logspace(-3, 3, 13)},
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="r2",
    )
    search.fit(x_train_t, y_train)
    pred = search.predict(x_test_t)
    return {
        "r2": float(r2_score(y_test, pred)),
        "spearman": float(scipy.stats.spearmanr(y_test, pred).statistic),
        "alpha": float(search.best_params_["alpha"]),
    }


def main():
    parser = argparse.ArgumentParser(description="FLAb-style property regression with Ophiuchus embeddings.")
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--target-col", type=str, required=True)
    parser.add_argument("--heavy-col", type=str, default="heavy")
    parser.add_argument("--light-col", type=str, default="light")
    parser.add_argument("--checkpoint-path", type=str, default=str(ophiuchus_ab_checkpoint_path()))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    model = OphiuchusEmbeddingModel(checkpoint_path=args.checkpoint_path, device=device)

    train_loader = DataLoader(
        PropertyDataset(args.train_csv, args.heavy_col, args.light_col, args.target_col),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        PropertyDataset(args.test_csv, args.heavy_col, args.light_col, args.target_col),
        batch_size=args.batch_size,
        shuffle=False,
    )

    x_train, y_train = extract_embeddings(model, train_loader, device)
    x_test, y_test = extract_embeddings(model, test_loader, device)
    metrics = run_regression(x_train, y_train, x_test, y_test)
    print(metrics)


if __name__ == "__main__":
    main()
