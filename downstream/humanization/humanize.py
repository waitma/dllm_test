from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downstream.common import ChainPaddingCollator, load_model, run_generate
from dllm.pipelines.bioseq import Esm2ProteinTokenizer, ophiuchus_ab_checkpoint_path

try:
    import abnumber

    ABNUMBER_AVAILABLE = True
except ImportError:
    ABNUMBER_AVAILABLE = False


def create_fr_mask_with_abnumber(sequence: str, scheme: str = "imgt"):
    if not ABNUMBER_AVAILABLE:
        return sequence, sequence
    chain = abnumber.Chain(sequence, scheme=scheme, assign_germline=False)
    fv_seq = chain.seq
    masked_seq = ""
    for region_name in ["fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"]:
        region_seq = getattr(chain, f"{region_name}_seq", "")
        masked_seq += ("X" * len(region_seq)) if region_name.startswith("fr") else region_seq
    return masked_seq, fv_seq


def parse_pdb_chain_pairs(csv_path: str) -> list[dict[str, str | list[str]]]:
    df = pd.read_csv(csv_path, header=None, names=["pdb_id", "chain_pair"])
    pairs: list[dict[str, str | list[str]]] = []
    for row_idx, row in df.iterrows():
        pdb_id = str(row["pdb_id"]).strip().lower()
        chain_pair = str(row["chain_pair"]).strip()
        chains = [part.strip() for part in chain_pair.split("-") if part.strip()]
        pairs.append(
            {
                "pdb_id": pdb_id,
                "chain_pair": chain_pair,
                "chains": chains,
                "sample_id": f"{pdb_id}_{chain_pair.replace('-', '')}",
                "row_idx": int(row_idx),
            }
        )
    return pairs


def resolve_structure_file(pdb_dir: str, pdb_id: str) -> str:
    for ext in (".cif", ".pdb"):
        candidate = os.path.join(pdb_dir, f"{pdb_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Structure file for {pdb_id} not found under {pdb_dir}")


def extract_seq_from_structure(structure_file: str):
    from Bio.PDB import MMCIFParser, PDBParser, PPBuilder

    suffix = Path(structure_file).suffix.lower()
    parser = MMCIFParser(QUIET=True) if suffix == ".cif" else PDBParser(QUIET=True)
    structure = parser.get_structure("structure", structure_file)
    ppb = PPBuilder()
    chain_seqs = {}
    for model in structure:
        for chain in model:
            seqs = [str(pp.get_sequence()) for pp in ppb.build_peptides(chain)]
            if seqs:
                chain_seqs[chain.id] = "".join(seqs)
    return chain_seqs, Path(structure_file).stem


class HumanizationDataset(Dataset):
    def __init__(
        self,
        pdb_dir: str,
        info_csv_fpath: str,
        start_index: int = 0,
        end_index: int | None = None,
    ):
        self.pdb_dir = pdb_dir
        self.pairs = parse_pdb_chain_pairs(info_csv_fpath)[start_index:end_index]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        pdb_id = str(pair["pdb_id"])
        chain_ids = pair["chains"]
        structure_file = resolve_structure_file(self.pdb_dir, pdb_id)
        chain_seqs, _ = extract_seq_from_structure(structure_file)
        heavy_seq = chain_seqs[chain_ids[0]]
        light_seq = chain_seqs[chain_ids[1]]
        heavy_masked, fv_heavy_seq = create_fr_mask_with_abnumber(heavy_seq, "imgt")
        light_masked, fv_light_seq = create_fr_mask_with_abnumber(light_seq, "imgt")
        light_masked = light_masked[:-3] + fv_light_seq[-3:]
        return heavy_masked, light_masked, fv_heavy_seq, fv_light_seq, pdb_id, str(pair["sample_id"]), str(pair["chain_pair"])


class HumanizationCollator:
    def __init__(self, tokenizer: Esm2ProteinTokenizer | None = None):
        self.tokenizer = tokenizer or Esm2ProteinTokenizer()
        self.base = ChainPaddingCollator(tokenizer=self.tokenizer)

    def __call__(self, batches):
        heavy_chains, light_chains, heavy_labels, light_labels = zip(*batches)
        input_ids, chain_ids, meta = self.base.stack_chains(list(heavy_chains), list(light_chains))
        labels, label_chain_ids, _ = self.base.stack_chains(list(heavy_labels), list(light_labels))
        x_id = self.tokenizer.token_to_id["X"]
        input_ids = input_ids.masked_fill(input_ids.eq(x_id), self.tokenizer.mask_token_id)
        return input_ids, chain_ids, labels, meta["heavy_max_len"]


def generate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint_path, device=device)
    tokenizer = model.tokenizer
    dataset = HumanizationDataset(
        args.pdb_dir,
        args.info_csv_fpath,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    collator = HumanizationCollator(tokenizer=tokenizer)

    all_heavy_aars = []
    all_light_aars = []
    rows = []

    for idx, sample in enumerate(dataset):
        heavy_masked, light_masked, heavy_seq, light_seq, pdb_id, sample_id, chain_pair = sample
        batch_data = [sample[:4] for _ in range(args.n_sequences)]
        chains, chain_ids, labels, heavy_len = collator(batch_data)
        chains = chains.to(device)
        chain_ids = chain_ids.to(device)
        labels = labels.to(device)

        output_tokens, _ = run_generate(
            model,
            chains,
            chain_ids,
            tokenizer,
            max_iter=args.max_iter,
            sampling_strategy=args.sampling_strategy,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
        )

        correct = (output_tokens == labels) * chains.eq(tokenizer.mask_token_id)
        heavy_mask = chains[:, :heavy_len].eq(tokenizer.mask_token_id)
        light_mask = chains[:, heavy_len:].eq(tokenizer.mask_token_id)
        heavy_aar = (correct[:, :heavy_len] & heavy_mask).sum(1) / heavy_mask.sum(1).clamp_min(1)
        light_aar = (correct[:, heavy_len:] & light_mask).sum(1) / light_mask.sum(1).clamp_min(1)
        all_heavy_aars.extend(heavy_aar.detach().cpu().tolist())
        all_light_aars.extend(light_aar.detach().cpu().tolist())

        for i in range(args.n_sequences):
            heavy_out = tokenizer.decode(output_tokens[i, 1:heavy_len - 1].tolist(), skip_special_tokens=True)
            light_out = tokenizer.decode(output_tokens[i, heavy_len + 1 : -1].tolist(), skip_special_tokens=True)
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "sample_id": sample_id,
                    "chain_pair": chain_pair,
                    "variant_idx": i,
                    "heavy_native": heavy_seq,
                    "light_native": light_seq,
                    "generated_heavy": heavy_out,
                    "generated_light": light_out,
                }
            )
        print(f"[{idx + 1}/{len(dataset)}] {sample_id} heavy_aar={heavy_aar.mean().item()*100:.2f}%")

    print(f"Overall heavy AAR: {np.mean(all_heavy_aars) * 100:.2f}%")
    print(f"Overall light AAR: {np.mean(all_light_aars) * 100:.2f}%")
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Saved {args.output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Humanize antibody FR regions with Ophiuchus-Ab.")
    parser.add_argument("--pdb-dir", type=str, required=True)
    parser.add_argument("--info-csv-fpath", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, default=str(ophiuchus_ab_checkpoint_path()))
    parser.add_argument("--output-csv", type=str, default="humanization_results.csv")
    parser.add_argument("--n-sequences", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling-strategy", type=str, default="gumbel_argmax")
    parser.add_argument("--max-iter", type=int, default=32)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
