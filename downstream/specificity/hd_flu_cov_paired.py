from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import ChainPaddingCollator
from downstream.embeddings import OphiuchusEmbeddingModel
from dllm.pipelines.bioseq import Esm2ProteinTokenizer, ophiuchus_ab_checkpoint_path


class PairClsDataset(Dataset):
    def __init__(self, csv_path: str, h_col: str = "h_sequence", l_col: str = "l_sequence", y_col: str = "label"):
        df = pd.read_csv(csv_path)
        self.h = df[h_col].astype(str).tolist()
        self.l = df[l_col].astype(str).tolist()
        self.y = df[y_col].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.h[idx], self.l[idx], self.y[idx]


class PairClassifier(nn.Module):
    def __init__(self, in_dim: int = 1280, hidden_dim: int = 512, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def extract_embeddings(model, loader, device):
    features = []
    labels = []
    collator = ChainPaddingCollator(tokenizer=Esm2ProteinTokenizer())
    model.eval()
    with torch.no_grad():
        for heavy, light, label in tqdm(loader, desc="embeddings"):
            input_ids, chain_ids, _ = collator.stack_chains(list(heavy), list(light))
            embedding = model(input_ids.to(device), chain_ids.to(device))
            features.append(embedding.cpu())
            labels.append(torch.tensor(label, dtype=torch.long))
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Specificity classification with Ophiuchus embeddings.")
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=str(ophiuchus_ab_checkpoint_path()))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    embed_model = OphiuchusEmbeddingModel(checkpoint_path=args.checkpoint_path, device=device)
    train_loader = DataLoader(PairClsDataset(args.train_csv), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(PairClsDataset(args.test_csv), batch_size=args.batch_size, shuffle=False)

    x_train, y_train = extract_embeddings(embed_model, train_loader, device)
    x_test, y_test = extract_embeddings(embed_model, test_loader, device)

    classifier = PairClassifier(in_dim=x_train.shape[-1], num_classes=int(y_train.max().item()) + 1).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    classifier.train()
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(classifier(x_train), y_train)
        loss.backward()
        optimizer.step()

    classifier.eval()
    with torch.no_grad():
        pred = classifier(x_test.to(device)).argmax(dim=-1).cpu().numpy()
    y_true = y_test.numpy()
    print(
        {
            "accuracy": float(accuracy_score(y_true, pred)),
            "f1_macro": float(f1_score(y_true, pred, average="macro")),
            "mcc": float(matthews_corrcoef(y_true, pred)),
        }
    )


if __name__ == "__main__":
    main()
