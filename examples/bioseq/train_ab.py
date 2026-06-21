from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.bioseq import (
    Esm2ProteinTokenizer,
    MultiChainOphiuchusAbModel,
    OphiuchusAbTrainStepConfig,
    OphiuchusAbTrainingCollator,
    compute_ophiuchus_ab_training_loss,
    load_ophiuchus_checkpoint,
)
from dllm.pipelines.bioseq.ophiuchus.model import OphiuchusAbBackbone


class AntibodyCsvDataset(Dataset):
    def __init__(self, csv_path: Path) -> None:
        with csv_path.open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        return {
            "chains": self._extract_heavy_light(row),
            "task_type": "antibody",
        }

    @staticmethod
    def _extract_heavy_light(row: dict[str, str]) -> list[str]:
        if row.get("vh_protein_sequence") and row.get("vl_protein_sequence"):
            return [row["vh_protein_sequence"], row["vl_protein_sequence"]]
        if row.get("heavy") and row.get("light"):
            return [row["heavy"], row["light"]]

        chain1 = row.get("cleaned_chain1_seq", "")
        chain2 = row.get("cleaned_chain2_seq", "")
        if not chain1 or not chain2:
            raise KeyError(
                "AntibodyCsvDataset requires vh_protein_sequence/vl_protein_sequence, "
                "heavy/light, or cleaned_chain1_seq/cleaned_chain2_seq columns"
            )

        chain1_type = (row.get("chain1_anarci_type") or row.get("chain1_type") or "").upper()
        chain2_type = (row.get("chain2_anarci_type") or row.get("chain2_type") or "").upper()
        if chain2_type.startswith("H") and not chain1_type.startswith("H"):
            return [chain2, chain1]
        return [chain1, chain2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Ophiuchus-Ab diffusion antibody model.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--init-multimer", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/vepfs-mlp2/c20250601/251105016/project/dllm_test/.models/bioseq/ophiuchus-ab"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10)
    return parser.parse_args()


def run_torch_training(
    model: torch.nn.Module,
    dataset: Dataset,
    collator,
    train_step,
    args: argparse.Namespace,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    iterator = iter(loader)

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        result = train_step(model=model, batch=batch)
        loss = result.loss if hasattr(result, "loss") else result
        loss.backward()
        optimizer.step()
        value = loss.item()
        if hasattr(result, "heavy_loss"):
            print(
                f"step={step} loss={value:.4f} "
                f"heavy={result.heavy_loss.item():.4f} light={result.light_loss.item():.4f}"
            )
        else:
            print(f"step={step} loss={value:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        args.output_dir / "pytorch_model.pt",
    )


def _move_batch_to_device(batch, device):
    if isinstance(batch, dict):
        moved = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                moved[key] = _move_batch_to_device(value, device)
            elif torch.is_tensor(value):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved
    return batch


def build_ophiuchus_training_stack(args: argparse.Namespace):
    collator = OphiuchusAbTrainingCollator(tokenizer=Esm2ProteinTokenizer())
    backbone = OphiuchusAbBackbone()
    if args.init_multimer:
        backbone.init_multimer_attention()
    model = MultiChainOphiuchusAbModel(net=backbone)
    if args.checkpoint_path is not None:
        load_ophiuchus_checkpoint(model, args.checkpoint_path)
    train_config = OphiuchusAbTrainStepConfig()
    train_step = lambda model, batch: compute_ophiuchus_ab_training_loss(model, batch, train_config)
    return model, collator, train_step


def main() -> None:
    args = parse_args()
    dataset = AntibodyCsvDataset(args.train_csv)
    model, collator, train_step = build_ophiuchus_training_stack(args)
    run_torch_training(
        model=model,
        dataset=dataset,
        collator=collator,
        train_step=train_step,
        args=args,
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
