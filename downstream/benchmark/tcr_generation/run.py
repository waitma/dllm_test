#!/usr/bin/env python
"""T4 -- TCR CDR3-beta generation.

Generate CDR3-beta sequences and evaluate them against a held-out repertoire
(distribution match) and the training set (novelty). This harness is model-
agnostic: the trained foundation model can be plugged in by writing its samples
to a file and scoring them with ``--method file --samples-file path``.

Built-in training-free reference generators:
  pwm     : per-length position weight matrix (length ~ empirical, each position
            sampled from that length's AA frequencies). OLGA-lite positional model.
  markov  : order-k amino-acid Markov chain with start/end, length from samples.

Metrics: novelty (vs train), mean NN edit distance (vs train sample), k-mer JSD
(vs holdout), unique fraction. Lower JSD + high novelty + plausible NN distance
indicate diverse, realistic, non-memorized sequences.

Example:
    python tcr_generation/run.py --method markov --order 2 --n 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common import metrics  # noqa: E402

DATA = BENCH / "data" / "tcr_generation"
OUT = BENCH / "outputs" / "tcr_generation"
AA = "ACDEFGHIKLMNPQRSTVWY"


def load_seqs(path):
    return [s for s in Path(path).read_text().splitlines() if s]


def sample_pwm(train, n, rng):
    by_len = defaultdict(list)
    for s in train:
        by_len[len(s)].append(s)
    lengths = list(by_len.keys())
    weights = np.array([len(by_len[L]) for L in lengths], dtype=float)
    weights /= weights.sum()
    # per-length position frequency tables
    pwm = {}
    for L in lengths:
        mat = np.zeros((L, len(AA)))
        idx = {a: i for i, a in enumerate(AA)}
        for s in by_len[L]:
            for p, ch in enumerate(s):
                if ch in idx:
                    mat[p, idx[ch]] += 1
        mat += 1e-3
        mat /= mat.sum(1, keepdims=True)
        pwm[L] = mat
    out = []
    for _ in range(n):
        L = rng.choice(lengths, p=weights)
        mat = pwm[L]
        seq = "".join(AA[rng.choice(len(AA), p=mat[p])] for p in range(L))
        out.append(seq)
    return out


def sample_markov(train, n, order, rng):
    trans = defaultdict(Counter)
    lengths = []
    start = "^" * order
    for s in train:
        lengths.append(len(s))
        ctx = start
        for ch in s:
            trans[ctx][ch] += 1
            ctx = (ctx + ch)[-order:]
        trans[ctx]["$"] += 1
    len_arr = np.array(lengths)
    out = []
    for _ in range(n):
        target_len = int(rng.choice(len_arr))
        ctx = start
        seq = []
        for _ in range(target_len):
            counter = trans.get(ctx)
            if not counter:
                break
            choices, cnts = zip(*counter.items())
            probs = np.array(cnts, dtype=float)
            probs /= probs.sum()
            nxt = choices[rng.choice(len(choices), p=probs)]
            if nxt == "$":
                break
            seq.append(nxt)
            ctx = (ctx + nxt)[-order:]
        if seq:
            out.append("".join(seq))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["pwm", "markov", "file"], default="markov")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--samples-file", default=None, help="for --method file")
    ap.add_argument("--train-ref-cap", type=int, default=20000,
                    help="cap train seqs used for NN-distance reference")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = load_seqs(DATA / "train_cdr3b.txt")
    holdout = load_seqs(DATA / "holdout_cdr3b.txt")
    rng = np.random.default_rng(args.seed)
    print(f"train={len(train)} holdout={len(holdout)}")

    if args.method == "pwm":
        gen = sample_pwm(train, args.n, rng)
        model_name = "PWM(per-length)"
    elif args.method == "markov":
        gen = sample_markov(train, args.n, args.order, rng)
        model_name = f"Markov(order={args.order})"
    else:
        gen = load_seqs(args.samples_file)
        model_name = f"file:{Path(args.samples_file).name}"

    # NN-distance reference: capped random train sample (edit distance is O(n)).
    ref_train = list(rng.choice(train, size=min(args.train_ref_cap, len(train)),
                                replace=False))
    m = metrics.generation_metrics(gen, dist_reference=holdout, train_set=ref_train, k=3)

    tag = (args.method if args.method != "markov" else f"markov{args.order}")
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples.txt").write_text("\n".join(gen) + "\n")
    result = {"method": args.method, "model": model_name, "n_requested": args.n, **m}
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n=== {model_name} ===")
    print(f"  n={m['n_generated']} unique={m['n_unique']} "
          f"novelty={m.get('novelty'):.3f} NN_dist={m.get('mean_nn_distance'):.2f} "
          f"kmer_JSD={m.get('kmer_jsd'):.4f}")
    print(f"  -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
