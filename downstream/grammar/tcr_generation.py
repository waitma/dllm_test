"""Immune LLaDA masked-diffusion TCR sampling adapters -- 3 modes.

Run with::

    python -m downstream.grammar.tcr_generation --mode fulllength --checkpoint CHECKPOINT --out OUTPUT.jsonl

Legacy CDR3-only modes are retained for legacy checkpoints. Fixed-canvas v2
checkpoints use ``fulllength`` for full-chain generation instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.immune_llada.data import BioSeqChain, BioSeqRecord
from downstream.grammar.common import (
    build_eval_collator,
    collator_fixed_receptor_lengths,
    load_grammar_checkpoint,
    run_grammar_generate,
)
from downstream.grammar.masks import cdr3b_span_partial_mask, tcr_generation_partial_mask
from downstream.grammar.metrics import extract_chain_sequence

DEFAULT_CKPT = PROJECT_ROOT / "output" / "protein_esmc_llada270m_diffusion_immune" / "checkpoint-42000"

# Empirical full CDR3b length distribution (C..F) from the OTS repertoire train
# split; used only by the legacy CDR3-only samplers.
_CDR3B_LEN_DIST = {
    9: 0.0004, 10: 0.0032, 11: 0.0291, 12: 0.0727, 13: 0.1603, 14: 0.2255,
    15: 0.2453, 16: 0.1463, 17: 0.0671, 18: 0.0304, 19: 0.0121, 20: 0.0048,
    21: 0.0016, 22: 0.0007, 23: 0.0002,
}
_LEN_VALUES = np.array(sorted(_CDR3B_LEN_DIST))
_LEN_PROBS = np.array([_CDR3B_LEN_DIST[L] for L in _LEN_VALUES], dtype=float)
_LEN_PROBS /= _LEN_PROBS.sum()

# Typical full-length TCR chain lengths from OTS for legacy diagnostics.
_ALPHA_LEN, _BETA_LEN = 113, 114
# Fixed-v2 decoder canvases include one EOS slot: alpha/light=135 and beta/heavy=167.
_FIXED_ALPHA_AA, _FIXED_BETA_AA = 134, 166


def _sample_lengths(n: int, rng: np.random.Generator) -> list[int]:
    return [int(x) for x in rng.choice(_LEN_VALUES, size=n, p=_LEN_PROBS)]


def _placeholder(length: int) -> str:
    """A syntactically valid legacy CDR3b placeholder ``C A...A F``."""

    length = max(3, int(length))
    return "C" + "A" * (length - 2) + "F"


def _fixed_canvas_placeholder(length: int) -> str:
    """Fill a full-chain v2 canvas without importing CDR3 semantics."""

    return "A" * int(length)


def _model_is_fixed_v2(model) -> bool:
    config = getattr(model, "config", None)
    marker = getattr(config, "fixed_receptor_lengths", None)
    if marker is not None:
        return bool(marker)
    marker = getattr(model, "fixed_receptor_lengths", None)
    if marker is not None:
        return bool(marker)
    collator = getattr(model, "_fusion_eval_collator", None)
    marker = collator_fixed_receptor_lengths(collator)
    if marker is not None:
        return marker
    return bool(getattr(model, "_predict_eos", False))


def _decode_chain(token_ids: torch.Tensor, batch: dict, row: int, chain_index: int, tokenizer) -> str:
    """Decode one chain using the generated EOS token at its own decoder slots."""

    return extract_chain_sequence(
        token_ids,
        batch["attention_mask"][row],
        batch["residue_mask"][row],
        tokenizer,
        chain=chain_index,
        position_ids_chain=(
            batch["position_ids_chain"][row]
            if batch.get("position_ids_chain") is not None
            else None
        ),
        chain_slot_mask=(
            batch["chain_slot_mask"][row]
            if batch.get("chain_slot_mask") is not None
            else None
        ),
    )


class BioSeqTcrSampler:
    def __init__(self, checkpoint: str | Path, device: str = "cuda", seed: int = 42):
        device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.device = torch.device(device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model, self.tokenizer = load_grammar_checkpoint(checkpoint, device=self.device)
        self.fixed_v2 = _model_is_fixed_v2(self.model)
        self.collator = build_eval_collator(self.model, self.tokenizer)
        self.rng = np.random.default_rng(seed)

    def _to_device(self, batch):
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _require_legacy_cdr3_semantics(self, operation: str) -> None:
        if self.fixed_v2:
            raise ValueError(
                f"{operation} is a legacy CDR3-only sampler and is incompatible with "
                "fixed_receptor_lengths v2; use fulllength_pairs for fixed-canvas generation"
            )

    # -- Setting B: epitope-conditioned CDR3b -------------------------------
    def conditional_cdr3b(
        self,
        epitope: str,
        k: int,
        batch_size: int = 64,
        max_iter: int = 32,
        temperature: float = 1.0,
        sampling_strategy: str = "gumbel_argmax",
        mhc_pseudo: str | None = None,
    ) -> list[str]:
        """Sample legacy variable-length CDR3b designs for ``epitope``."""

        self._require_legacy_cdr3_semantics("conditional_cdr3b")
        out: list[str] = []
        lengths = _sample_lengths(k, self.rng)
        # Chain indices follow grammar emission order (MHC block, then peptide,
        # then receptor), not the order of BioSeqRecord.chains.
        target_chain = 2 if mhc_pseudo else 1
        for start in range(0, k, batch_size):
            chunk = lengths[start:start + batch_size]
            records = [
                BioSeqRecord(
                    chains=(
                        ([BioSeqChain(mhc_pseudo, "mhc")] if mhc_pseudo else [])
                        + [BioSeqChain(epitope, "antigen"), BioSeqChain(_placeholder(L), "tcr_beta")]
                    ),
                    task_type="tcr_epitope", source="bioseq_sample",
                )
                for L in chunk
            ]
            batch = self._to_device(self.collator(records))
            partial = tcr_generation_partial_mask(batch, target_chain_indices=target_chain)
            tokens, _ = run_grammar_generate(
                self.model,
                batch,
                partial_mask=partial,
                max_iter=max_iter,
                sampling_strategy=sampling_strategy,
                temperature=temperature,
            )
            for row in range(len(records)):
                seq = _decode_chain(tokens[row], batch, row, target_chain, self.tokenizer)
                if seq:
                    out.append(seq)
        return out

    # -- Setting A: unconditional CDR3b ------------------------------------
    def unconditional_cdr3b(
        self,
        n: int,
        batch_size: int = 128,
        max_iter: int = 32,
        temperature: float = 1.0,
        sampling_strategy: str = "gumbel_argmax",
    ) -> list[str]:
        """Sample an unconditional legacy variable-length CDR3b repertoire."""

        self._require_legacy_cdr3_semantics("unconditional_cdr3b")
        out: list[str] = []
        lengths = _sample_lengths(n, self.rng)
        for start in range(0, n, batch_size):
            chunk = lengths[start:start + batch_size]
            records = [
                BioSeqRecord(
                    chains=[BioSeqChain(_placeholder(L), "tcr_beta")],
                    task_type="tcr", source="bioseq_sample",
                )
                for L in chunk
            ]
            batch = self._to_device(self.collator(records))
            partial = tcr_generation_partial_mask(batch, target_chain_indices=0)
            tokens, _ = run_grammar_generate(
                self.model,
                batch,
                partial_mask=partial,
                max_iter=max_iter,
                sampling_strategy=sampling_strategy,
                temperature=temperature,
            )
            for row in range(len(records)):
                seq = _decode_chain(tokens[row], batch, row, 0, self.tokenizer)
                if seq:
                    out.append(seq)
        return out

    # -- Diagnostic: CDR3b infilling, no epitope ---------------------------
    def infill_cdr3b(
        self,
        sequences: list[str],
        mask_width: int,
        batch_size: int = 128,
        max_iter: int = 32,
        temperature: float = 1.0,
        sampling_strategy: str = "gumbel_argmax",
    ) -> list[dict]:
        """Re-fill a legacy variable-length CDR3b window, with no epitope."""

        self._require_legacy_cdr3_semantics("infill_cdr3b")
        out: list[dict] = []
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start:start + batch_size]
            spans: list[tuple[int, int]] = []
            for seq in chunk:
                length = len(seq)
                width = (length - 2) if mask_width <= 0 else min(int(mask_width), length - 2)
                lo = (length - width) // 2
                spans.append((lo, lo + width))
            records = [
                BioSeqRecord(
                    chains=[BioSeqChain(seq, "tcr_beta")],
                    task_type="tcr", source="bioseq_infill",
                )
                for seq in chunk
            ]
            batch = self._to_device(self.collator(records))
            partial = cdr3b_span_partial_mask(batch, chain_index=0, span=spans)
            tokens, _ = run_grammar_generate(
                self.model,
                batch,
                partial_mask=partial,
                max_iter=max_iter,
                sampling_strategy=sampling_strategy,
                temperature=temperature,
            )
            for row, seq in enumerate(chunk):
                pred = _decode_chain(tokens[row], batch, row, 0, self.tokenizer)
                lo, hi = spans[row]
                out.append({
                    "index": start + row,
                    "truth": seq,
                    "pred": pred,
                    "span": [lo, hi],
                    "truth_window": seq[lo:hi],
                    "pred_window": pred[lo:hi] if len(pred) == len(seq) else "",
                    "length_ok": len(pred) == len(seq),
                })
        return out

    def infill_cdr3b_in_pair(
        self,
        pairs: list[tuple[str, str, str]],
        mask_width: int,
        batch_size: int = 32,
        max_iter: int = 32,
        temperature: float = 1.0,
        sampling_strategy: str = "gumbel_argmax",
    ) -> list[dict]:
        """Re-fill a legacy CDR3b window inside a full-length pair."""

        self._require_legacy_cdr3_semantics("infill_cdr3b_in_pair")
        out: list[dict] = []
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            spans: list[tuple[int, int]] = []
            windows: list[str] = []
            for _, beta_fv, core in chunk:
                anchored_len = len(core) + 2
                width = (anchored_len - 2) if mask_width <= 0 else min(int(mask_width), anchored_len - 2)
                lo_anchored = (anchored_len - width) // 2
                offset = beta_fv.find(core)
                if offset < 1:
                    raise ValueError("could not map CDR3b core into the beta full-length sequence")
                lo = offset + (lo_anchored - 1)
                spans.append((lo, lo + width))
                windows.append(beta_fv[lo:lo + width])
            records = [
                BioSeqRecord(
                    chains=[BioSeqChain(alpha, "tcr_alpha"), BioSeqChain(beta, "tcr_beta")],
                    task_type="tcr", source="bioseq_infill_pair",
                )
                for alpha, beta, _ in chunk
            ]
            batch = self._to_device(self.collator(records))
            # The renderer emits beta before alpha for tcr_pair: beta=chain 0.
            partial = cdr3b_span_partial_mask(batch, chain_index=0, span=spans)
            tokens, _ = run_grammar_generate(
                self.model,
                batch,
                partial_mask=partial,
                max_iter=max_iter,
                sampling_strategy=sampling_strategy,
                temperature=temperature,
            )
            for row, (_, beta_fv, core) in enumerate(chunk):
                pred = _decode_chain(tokens[row], batch, row, 0, self.tokenizer)
                lo, hi = spans[row]
                out.append({
                    "index": start + row,
                    "truth": beta_fv,
                    "pred": pred,
                    "span": [lo, hi],
                    "cdr3b_core": core,
                    "truth_window": windows[row],
                    "pred_window": pred[lo:hi] if len(pred) == len(beta_fv) else "",
                    "length_ok": len(pred) == len(beta_fv),
                })
        return out

    # -- Setting C: unconditional full-length alpha/beta -------------------
    def fulllength_pairs(
        self,
        n: int,
        batch_size: int = 16,
        max_iter: int = 64,
        temperature: float = 1.0,
        sampling_strategy: str = "gumbel_argmax",
    ) -> list[dict]:
        """Generate paired TCRs, using fixed alpha/beta canvases for v2."""

        out: list[dict] = []
        alpha_length = _FIXED_ALPHA_AA if self.fixed_v2 else _ALPHA_LEN
        beta_length = _FIXED_BETA_AA if self.fixed_v2 else _BETA_LEN
        for start in range(0, n, batch_size):
            take = min(batch_size, n - start)
            records = [
                BioSeqRecord(
                    chains=[
                        BioSeqChain(
                            _fixed_canvas_placeholder(alpha_length)
                            if self.fixed_v2
                            else _placeholder(alpha_length),
                            "tcr_alpha",
                        ),
                        BioSeqChain(
                            _fixed_canvas_placeholder(beta_length)
                            if self.fixed_v2
                            else _placeholder(beta_length),
                            "tcr_beta",
                        ),
                    ],
                    task_type="tcr", source="bioseq_sample",
                )
                for _ in range(take)
            ]
            batch = self._to_device(self.collator(records))
            # Renderer order is beta=chain 0, alpha=chain 1. Both full canvases,
            # including EOS slots, are generated by the v2 mask.
            partial = tcr_generation_partial_mask(batch, target_chain_indices={0, 1})
            tokens, _ = run_grammar_generate(
                self.model,
                batch,
                partial_mask=partial,
                max_iter=max_iter,
                sampling_strategy=sampling_strategy,
                temperature=temperature,
            )
            for row in range(take):
                beta = _decode_chain(tokens[row], batch, row, 0, self.tokenizer)
                alpha = _decode_chain(tokens[row], batch, row, 1, self.tokenizer)
                out.append({"index": start + row, "tcr_alpha": alpha, "tcr_beta": beta})
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["uncond", "conditional", "fulllength", "infill"], required=True)
    ap.add_argument("--infill-input", default=None)
    ap.add_argument("--mask-width", type=int, default=3)
    ap.add_argument("--infill-layout", choices=["single", "pair"], default="single")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--eval-json", default=None, help="eval_conditional.json (conditional mode)")
    ap.add_argument("--eval-set", choices=["held20", "benchmark14", "unseen", "all"], default="all")
    ap.add_argument("-k", "--k", type=int, default=100, help="designs per epitope (conditional)")
    ap.add_argument("-n", "--n", type=int, default=5000, help="samples (uncond / fulllength)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sampling-strategy", default="gumbel_argmax")
    ap.add_argument("--with-mhc", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampler = BioSeqTcrSampler(args.checkpoint, device=args.device, seed=args.seed)

    if args.mode == "uncond":
        bs = args.batch_size or 128
        mi = args.max_iter or 32
        n = 64 if args.smoke else args.n
        seqs = sampler.unconditional_cdr3b(
            n, batch_size=bs, max_iter=mi,
            temperature=args.temperature, sampling_strategy=args.sampling_strategy,
        )
        out_path.write_text("\n".join(seqs) + "\n")
        print(f"[uncond] wrote {len(seqs)} CDR3b -> {out_path}")

    elif args.mode == "infill":
        mi = args.max_iter or 32
        if args.infill_layout == "pair":
            import pandas as pd
            src = Path(args.infill_input) if args.infill_input else (
                PROJECT_ROOT / "data" / "ots_paired_clean" / "splits" / "holdout.csv"
            )
            df = pd.read_csv(src, low_memory=False)
            pairs = []
            for row in df.itertuples():
                beta, alpha = str(row.cleaned_chain1_seq), str(row.cleaned_chain2_seq)
                core = str(row.chain1_CDR3)
                if core and core != "nan" and beta.find(core) >= 1:
                    pairs.append((alpha, beta, core))
            if args.smoke:
                pairs = pairs[:32]
            elif args.n and args.n < len(pairs):
                pairs = pairs[:args.n]
            rows = sampler.infill_cdr3b_in_pair(
                pairs, mask_width=args.mask_width, batch_size=args.batch_size or 32,
                max_iter=mi, temperature=args.temperature,
                sampling_strategy=args.sampling_strategy,
            )
            src_name, n_src = f"{src.name} (tcr_pair)", len(pairs)
        else:
            src = Path(args.infill_input) if args.infill_input else (
                PROJECT_ROOT / "downstream" / "benchmark" / "data"
                / "tcr_generation_fullref" / "holdout_cdr3b.txt"
            )
            cores = [line.strip() for line in src.read_text().splitlines() if line.strip()]
            seqs = [
                ("C" if not core.startswith("C") else "") + core
                + ("F" if not core.endswith(("F", "W")) else "")
                for core in cores
            ]
            if args.smoke:
                seqs = seqs[:64]
            elif args.n and args.n < len(seqs):
                seqs = seqs[:args.n]
            rows = sampler.infill_cdr3b(
                seqs, mask_width=args.mask_width, batch_size=args.batch_size or 128,
                max_iter=mi, temperature=args.temperature,
                sampling_strategy=args.sampling_strategy,
            )
            src_name, n_src = f"{src.name} (tcr_single)", len(seqs)
        with out_path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        ok = [row for row in rows if row["length_ok"]]
        hit = sum(sum(a == b for a, b in zip(row["truth_window"], row["pred_window"])) for row in ok)
        total = sum(len(row["truth_window"]) for row in ok)
        print(f"[infill] source {src_name}, n={len(rows)}/{n_src}, mask_width={args.mask_width}")
        print(f"  length preserved {len(ok)}/{len(rows)}")
        print(f"  window AAR = {hit}/{total} = {hit / max(1, total) * 100:.2f}%")

    elif args.mode == "conditional":
        if not args.eval_json:
            raise SystemExit("--eval-json required for conditional mode")
        eval_entries = json.loads(Path(args.eval_json).read_text())
        if args.eval_set == "unseen":
            unseen_path = PROJECT_ROOT / "downstream" / "benchmark" / "data" / "tcr_generation_bench" / "bioseq_unseen_pmhc.json"
            unseen = set(json.loads(unseen_path.read_text()))
            eval_entries = {p: entry for p, entry in eval_entries.items() if p in unseen}
        elif args.eval_set != "all":
            eval_entries = {p: entry for p, entry in eval_entries.items() if entry["category"] == args.eval_set}
        items = list(eval_entries.items())
        if args.smoke:
            items = items[:2]
        k = 16 if args.smoke else args.k
        if args.with_mhc:
            missing = [p for p, entry in items if not entry.get("mhc_pseudo")]
            if missing:
                raise SystemExit(f"--with-mhc needs mhc_pseudo on every entry; missing for {missing}")
        with out_path.open("w") as handle:
            for index, (pmhc, entry) in enumerate(items):
                sequences = sampler.conditional_cdr3b(
                    entry["epitope"], k, batch_size=args.batch_size or 64,
                    max_iter=args.max_iter or 32, temperature=args.temperature,
                    sampling_strategy=args.sampling_strategy,
                    mhc_pseudo=entry["mhc_pseudo"] if args.with_mhc else None,
                )
                handle.write(json.dumps({
                    "pmhc": pmhc,
                    "epitope": entry["epitope"],
                    "layout": "tcr_pmhc" if args.with_mhc else "tcr_peptide",
                    "sequences": sequences,
                }) + "\n")
                handle.flush()
                print(f"  [{index + 1}/{len(items)}] {pmhc}: {len(sequences)} CDR3b")

    else:
        pairs = sampler.fulllength_pairs(
            8 if args.smoke else args.n,
            batch_size=args.batch_size or 16,
            max_iter=args.max_iter or 64,
            temperature=args.temperature,
            sampling_strategy=args.sampling_strategy,
        )
        with out_path.open("w") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair) + "\n")
        print(f"[fulllength] wrote {len(pairs)} alpha/beta pairs -> {out_path}")


if __name__ == "__main__":
    main()
