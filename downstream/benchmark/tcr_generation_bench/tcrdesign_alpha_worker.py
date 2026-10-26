#!/usr/bin/env python
"""Run released TcrDesign alpha decoding with one checkpoint load per job.

Direct smoke invocation (normally called by ``run_public_beta.py``)::

    /project_c20250601/251105016/conda/envs/tcrdesign/bin/python \
      /project_c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/tcrdesign_alpha_worker.py \
      --input /absolute/generated_betas.csv --output /absolute/generated_alphas.csv \
      --tcrdesign-repo /project_c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TcrDesign \
      --alpha-per-beta 10 --device cuda

The model architecture, token construction, checkpoint, temperature, official
beam-search function, and the released terminal-``GF`` cleanup are identical to
``TcrDesign/utils/generate_tcr.py::generate_alphaTCRs``.  Batching removes only
the upstream implementation's repeated checkpoint reload; it does not repair,
filter, deduplicate, or rescore decoded sequences.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch


ANTIGEN_MAX_LEN = 23
CDR3_MAX_LEN = 22
HIDDEN = 256
LAYERS = 6
ATTN_HEADS = 4
D_FF_FOLD = 4
ENCODER_HIDDEN = 256
DECODER_HIDDEN = 256
ENCODER_LAYERS = 1
DROPOUT = 0.3
TEACHER_FORCING = 1


def _decode_official(tokens: torch.Tensor, vocab) -> str:
    decoded = [vocab.itos[int(token)] for token in tokens]
    del decoded[0]
    try:
        del decoded[decoded.index("<eos>"):]
    except ValueError:
        pass
    sequence = "".join(decoded)
    if sequence.endswith("GF"):
        sequence = sequence[:-2]
    return sequence


def _encoder_row(epitope: str, cdr3b: str, vocab) -> list[int]:
    if len(epitope) > ANTIGEN_MAX_LEN - 2:
        raise ValueError(f"epitope exceeds official 21-aa limit: {epitope}")
    if len(cdr3b) > CDR3_MAX_LEN - 2:
        raise ValueError(f"generated beta exceeds official 20-aa limit: {cdr3b}")
    try:
        return (
            [vocab.sos_index]
            + [vocab.stoi[aa] for aa in epitope]
            + [vocab.eos_index]
            + [vocab.pad_index] * (ANTIGEN_MAX_LEN - len(epitope) - 2)
            + [vocab.sos_index]
            + [vocab.stoi[aa] for aa in cdr3b]
            + [vocab.eos_index]
            + [vocab.pad_index] * (CDR3_MAX_LEN - len(cdr3b) - 2)
        )
    except KeyError as exc:
        raise ValueError(f"sequence contains a token absent from official vocabulary: {exc}") from exc


def run(args) -> None:
    repo = Path(args.tcrdesign_repo).resolve()
    baselines = repo.parent
    if str(baselines) not in sys.path:
        sys.path.insert(0, str(baselines))

    from TcrDesign.dataset import WordVocab
    from TcrDesign.gru_generate import Atten, AttenSeq2Seq, Decoder, Encoder_both
    from TcrDesign.model import BERT
    from TcrDesign.utils import beam_search

    # The released pickle was created with ``WordVocab`` attached to the
    # training script's ``__main__`` module.  Official ``tcrdesign_G.py`` also
    # imports it into ``__main__`` before unpickling; mirror that requirement.
    globals()["WordVocab"] = WordVocab

    inputs = pd.read_csv(args.input, keep_default_na=False)
    required = {
        "condition_index", "target_id", "epitope", "mhc_allele",
        "generated_cdr3b", "beta_rank", "raw_beta_score",
    }
    missing = sorted(required - set(inputs.columns))
    if missing:
        raise ValueError(f"input is missing columns: {missing}")
    if args.alpha_per_beta <= 0 or args.batch_size <= 0:
        raise ValueError("alpha-per-beta and batch-size must be positive")

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    vocab = WordVocab.load_vocab(str(repo / "data" / "vocab_1mer.pkl"))
    peptide_bert = BERT(
        len(vocab), hidden=HIDDEN, n_layers=LAYERS, attn_heads=ATTN_HEADS,
        d_ff_fold=D_FF_FOLD, has_next=False,
    )
    beta_bert = BERT(
        len(vocab), hidden=HIDDEN, n_layers=LAYERS, attn_heads=ATTN_HEADS,
        d_ff_fold=D_FF_FOLD, has_next=False,
    )
    encoder = Encoder_both(
        peptide_bert, beta_bert, ENCODER_HIDDEN, ENCODER_LAYERS,
        DECODER_HIDDEN, ANTIGEN_MAX_LEN,
    )
    attention = Atten(ENCODER_HIDDEN, DECODER_HIDDEN)
    decoder = Decoder(
        len(vocab), ENCODER_HIDDEN, DECODER_HIDDEN, DROPOUT,
        attention, args.temperature,
    )
    model = AttenSeq2Seq(encoder, decoder, device, TEACHER_FORCING)
    checkpoint = repo / "weights" / "rnn" / "beta_alpha" / "RNN_tcr.ep"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    output_rows: list[dict[str, object]] = []
    total_batches = (len(inputs) + args.batch_size - 1) // args.batch_size
    for start in range(0, len(inputs), args.batch_size):
        batch = inputs.iloc[start:start + args.batch_size]
        batch_number = start // args.batch_size + 1
        print(
            f"[{batch_number}/{total_batches}] alpha decode conditions "
            f"{start + 1}-{start + len(batch)}",
            flush=True,
        )
        encoder_input = torch.tensor([
            _encoder_row(
                str(row.epitope).upper(), str(row.generated_cdr3b).upper(), vocab,
            )
            for row in batch.itertuples(index=False)
        ])
        decoded = beam_search(
            num_beams=args.alpha_per_beta,
            max_length=args.max_len,
            vocab=vocab,
            decoder=model,
            has_encoder=True,
            encoder_input=encoder_input,
            decoder_input_prefix=None,
            with_cuda=use_cuda,
        )
        for local_index, row in enumerate(batch.itertuples(index=False)):
            begin = local_index * args.alpha_per_beta
            end = begin + args.alpha_per_beta
            for alpha_rank, tokens in enumerate(decoded[begin:end], start=1):
                output_rows.append({
                    "target_id": row.target_id,
                    "epitope": str(row.epitope).upper(),
                    "generated_cdr3b": str(row.generated_cdr3b).upper(),
                    "generated_cdr3a": _decode_official(tokens, vocab),
                    "beta_rank": int(row.beta_rank),
                    "alpha_rank": alpha_rank,
                    "raw_beta_score": row.raw_beta_score,
                    # The released beam_search computes scores internally but
                    # does not return them.  Keep the interchange field empty.
                    "raw_alpha_score": "",
                    "mhc_allele": row.mhc_allele,
                })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(output, index=False)
    print(f"wrote {len(output_rows)} generated alpha rows -> {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tcrdesign-repo", required=True)
    parser.add_argument("--alpha-per-beta", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
