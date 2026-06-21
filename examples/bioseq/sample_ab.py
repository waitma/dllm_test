from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.bioseq import (
    Esm2ProteinTokenizer,
    MultiChainOphiuchusAbModel,
    OphiuchusAbInferenceCollator,
    ophiuchus_ab_checkpoint_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample antibody sequences with Ophiuchus-Ab.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=ophiuchus_ab_checkpoint_path(),
    )
    parser.add_argument("--heavy", type=str, required=True)
    parser.add_argument("--light", type=str, default="")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--sampling-strategy", choices=("gumbel_argmax", "argmax", "vanilla"), default="gumbel_argmax")
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--fix-heavy", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def decode_chain(tokenizer: Esm2ProteinTokenizer, token_ids: torch.Tensor, start: int, end: int) -> str:
    return tokenizer.decode(token_ids[start:end].tolist(), skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MultiChainOphiuchusAbModel.from_checkpoint(args.checkpoint_path, device="cpu")
    model.to(device)
    model.eval()

    example = {
        "chains": [args.heavy, args.light or "A"],
        "task_type": "antibody",
    }
    if args.fix_heavy:
        example["fix_chain_indices"] = [0]

    collator = OphiuchusAbInferenceCollator()
    batch = collator([example])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    partial_masks = batch.pop("partial_masks", None)

    with torch.no_grad():
        output_tokens, _ = model.generate(
            batch,
            max_iter=args.max_iter,
            partial_masks=partial_masks,
            sampling_strategy=args.sampling_strategy,
            cfg_scale=args.cfg_scale,
        )

    tokenizer = Esm2ProteinTokenizer()
    heavy = decode_chain(tokenizer, output_tokens[0], 0, 150)
    light = decode_chain(tokenizer, output_tokens[0], 150, 278)
    print(f"heavy={heavy}")
    print(f"light={light}")


if __name__ == "__main__":
    main()
