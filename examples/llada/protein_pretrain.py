"""
LLaDA-8B-Base pretraining on OAS/OTS immune sequences (grammar-v2).

Two paradigms:
  - diffusion: MDLMTrainer + LinearAlphaScheduler + loss_weight_type="scheduler"
  - bert:     BERTMLMTrainer with classic 80/10/10 MLM (mask_prob≈15%)

Vocabulary strategy: keep the full LLaDA BPE vocab, add residue/grammar tokens,
and remap grammar-v2 ids -> LLaDA ids so 8B embeddings/lm_head are inherited.

Local smoke (no 8B weights):
    python examples/llada/protein_pretrain.py --dry_run True --max_rows_per_source 8

Training (example, LoRA, no 4bit):
    accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
        examples/llada/protein_pretrain.py --lora True --train_mode diffusion
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset

import dllm
from dllm.core.schedulers import LinearAlphaScheduler
from dllm.core.trainers import MDLMTrainer
from dllm.pipelines.bioseq.datasets import (
    ImmuneSourceSpec,
    build_mixed_immune_dataset,
    oas_paired_row_to_record,
    ots_paired_row_to_record,
)
from dllm.pipelines.qwen3_vl_arch.data import (
    GRAMMAR_TOKENS,
    BioSeqChain,
    BioSeqRecord,
    Esm2SequenceTokenizer,
    GrammarRenderer,
    GrammarTokenizer,
)

logger = dllm.utils.get_default_logger(__name__)

OAS_DEFAULT_DIR = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/oas_previous_clean/splits"
)
OTS_DEFAULT_DIR = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean/final"
)

RESIDUES = list("LAGVSERTIDPKQNFYMHWCXBUZO")


class BERTMLMTrainer(MDLMTrainer):
    """Classic BERT MLM: select ``mask_prob`` positions, then 80/10/10 corrupt.

    Of the selected positions:
      - 80% replaced with ``mask_token_id``
      - 10% replaced with a random vocabulary token
      - 10% left unchanged
    Loss is computed on all selected positions (including the kept 10%).
    """

    def __init__(self, *args, mask_prob: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        if not (0.0 < mask_prob <= 1.0):
            raise ValueError(f"mask_prob must be in (0, 1], got {mask_prob}")
        self.mask_prob = float(mask_prob)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape
        device = input_ids.device
        maskable_mask = labels != -100

        # 1) Select ~mask_prob of maskable positions (BERT's 15%).
        selected = (
            torch.rand((b, l), device=device) < self.mask_prob
        ) & maskable_mask

        # 2) Among selected: 80% [MASK], 10% random, 10% keep.
        # Use a second uniform draw; apply only on selected positions.
        corruption = torch.rand((b, l), device=device)
        mask_replace = selected & (corruption < 0.8)
        random_replace = selected & (corruption >= 0.8) & (corruption < 0.9)
        # keep_replace = selected & (corruption >= 0.9)  # left as original

        noised_input_ids = input_ids.clone()
        noised_input_ids = torch.where(
            mask_replace,
            torch.as_tensor(
                self.processing_class.mask_token_id, device=device, dtype=input_ids.dtype
            ),
            noised_input_ids,
        )
        if random_replace.any():
            vocab_size = int(getattr(model.config, "vocab_size", 0) or 0)
            if vocab_size <= 0:
                vocab_size = int(len(self.processing_class))
            random_ids = torch.randint(
                low=0, high=vocab_size, size=(b, l), device=device, dtype=input_ids.dtype
            )
            # Avoid putting pad/mask as the "random" token when possible.
            pad_id = getattr(self.processing_class, "pad_token_id", None)
            mask_id = getattr(self.processing_class, "mask_token_id", None)
            if pad_id is not None:
                random_ids = torch.where(
                    random_ids == pad_id, (random_ids + 1) % vocab_size, random_ids
                )
            if mask_id is not None:
                random_ids = torch.where(
                    random_ids == mask_id, (random_ids + 1) % vocab_size, random_ids
                )
            noised_input_ids = torch.where(random_replace, random_ids, noised_input_ids)

        # 3) Forward + CE on all selected positions.
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        assert (
            input_ids[maskable_mask] == labels[maskable_mask]
        ).all(), "Mismatch between input_ids and labels at valid positions"

        token_nll = F.cross_entropy(
            logits.transpose(1, 2),
            input_ids,
            reduction="none",
        )
        token_nll = token_nll * selected.to(token_nll.dtype)

        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll.detach(),
            weight=selected.to(dtype=logits.dtype).detach(),
        )

        # Normalize by number of selected (BERT-masked) tokens, not all maskable.
        if self.loss_norm_type == "token":
            token_nll = token_nll / selected.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll = token_nll / selected.sum(-1, keepdim=True).clamp_min(1) / b
        elif self.loss_norm_type == "batch":
            token_nll = token_nll / b
        else:
            raise ValueError("Invalid loss_norm_type.")
        loss = token_nll.sum()
        return (loss, outputs) if return_outputs else loss


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "oas+ots"  # unused placeholder for base DataArguments
    oas_dir: str = OAS_DEFAULT_DIR
    ots_dir: str = OTS_DEFAULT_DIR
    train_split: str = "train"
    eval_split: str = "valid"
    max_rows_per_source: int | None = None
    # Kept in step with protein_pretrain_esmc.py and the base DataArguments so a
    # flagless run never silently trains on a shorter-truncated corpus.
    max_length: int = 1024


@dataclass
class TrainingArguments(dllm.core.trainers.MDLMConfig):
    output_dir: str = ".models/LLaDA-8B-Base/oas-ots"
    train_mode: str = field(
        default="diffusion",
        metadata={"help": 'Pretraining paradigm: "diffusion" or "bert"'},
    )
    mask_prob: float = field(
        default=0.15,
        metadata={
            "help": (
                "BERT mode: fraction of maskable tokens selected for MLM; "
                "among them apply classic 80% [MASK] / 10% random / 10% keep"
            )
        },
    )
    dry_run: bool = field(
        default=False,
        metadata={
            "help": "Build tokenizer/remap/dataset only; print samples and exit (no 8B load)"
        },
    )
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    max_steps: int = 1000
    num_train_epochs: float = 1.0
    eval_strategy: str = "steps"
    eval_steps: float = 0.2
    save_steps: float = 0.2
    logging_steps: float = 10
    report_to: str = "none"


def build_immune_specs(data_args: DataArguments) -> list[ImmuneSourceSpec]:
    return [
        ImmuneSourceSpec(
            "oas", Path(data_args.oas_dir), oas_paired_row_to_record, "antibody"
        ),
        ImmuneSourceSpec(
            "ots", Path(data_args.ots_dir), ots_paired_row_to_record, "tcr"
        ),
    ]


def expand_tokenizer_and_build_remap(
    tok: transformers.PreTrainedTokenizer,
) -> tuple[dict[int, int], int]:
    """Add residue/grammar tokens to LLaDA tok and build grammar_id -> llada_id remap."""

    residue_tokens = [f"<res_{aa}>" for aa in RESIDUES]
    new_tokens = residue_tokens + list(GRAMMAR_TOKENS) + ["<chainsep>"]
    n_added = tok.add_tokens(new_tokens)
    logger.info("Added %d new tokens to LLaDA tokenizer (requested %d)", n_added, len(new_tokens))

    gtok = GrammarTokenizer()
    esm = Esm2SequenceTokenizer()
    remap: dict[int, int] = {}

    residue_set = set(RESIDUES)
    for base_id, token in esm.id_to_token.items():
        if token in residue_set:
            remap[int(base_id)] = int(tok.convert_tokens_to_ids(f"<res_{token}>"))

    for grammar_token in GRAMMAR_TOKENS:
        remap[int(gtok.special_id(grammar_token))] = int(
            tok.convert_tokens_to_ids(grammar_token)
        )

    remap[int(gtok.chain_separator_id())] = int(tok.convert_tokens_to_ids("<chainsep>"))

    unk_id = getattr(tok, "unk_token_id", None)
    if unk_id is not None:
        for grammar_id, llada_id in remap.items():
            assert llada_id != unk_id, (
                f"remap[{grammar_id}] resolved to unk ({unk_id}); "
                "token was not added atomically"
            )

    return remap, n_added


def make_remap_ids(remap: dict[int, int]) -> Callable[[list[int]], list[int]]:
    def remap_ids(ids: list[int]) -> list[int]:
        out: list[int] = []
        for i in ids:
            try:
                out.append(remap[int(i)])
            except KeyError as exc:
                raise KeyError(f"unmapped grammar id {i}") from exc
        return out

    return remap_ids


class GrammarRemapDataset(Dataset):
    """Wrap immune CSV rows -> LLaDA-space input_ids/labels via grammar-v2 renderer."""

    def __init__(
        self,
        base_ds,
        renderer: GrammarRenderer,
        remap_ids: Callable[[list[int]], list[int]],
        max_length: int,
    ) -> None:
        self.base_ds = base_ds
        self.renderer = renderer
        self.remap_ids = remap_ids
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.base_ds)

    def _roles_for(self, rec: dict) -> list[BioSeqChain]:
        task_type = rec["task_type"]
        chains = rec["chains"]
        if task_type == "antibody":
            return [
                BioSeqChain(chains[0], "antibody_heavy"),
                BioSeqChain(chains[1], "antibody_light"),
            ]
        if task_type == "tcr":
            return [
                BioSeqChain(chains[0], "tcr_beta"),
                BioSeqChain(chains[1], "tcr_alpha"),
            ]
        raise ValueError(f"Unsupported task_type for protein_pretrain: {task_type}")

    def __getitem__(self, i: int, _retries: int = 0) -> dict:
        try:
            rec = self.base_ds[i]
            record = BioSeqRecord(
                chains=self._roles_for(rec),
                task_type=rec["task_type"],
                source=rec.get("source", ""),
            )
            row = self.renderer.encode(record)
            ids = self.remap_ids(row["input_ids"])
            mask = list(row["diffusion_loss_mask"])
            if len(ids) > self.max_length:
                ids = ids[: self.max_length]
                mask = mask[: self.max_length]
            labels = [token_id if m else -100 for token_id, m in zip(ids, mask)]
            return {"input_ids": ids, "labels": labels}
        except Exception as exc:
            if _retries >= max(len(self) - 1, 0):
                raise RuntimeError(
                    f"Failed to encode sample {i} after exhausting retries"
                ) from exc
            logger.warning("Skipping bad sample %d (%s); trying next", i, exc)
            return self.__getitem__((i + 1) % len(self), _retries=_retries + 1)


def _find_sample_by_task(
    base_ds, wrapped: GrammarRemapDataset, task_type: str
) -> tuple[dict, dict]:
    for idx in range(len(base_ds)):
        rec = base_ds[idx]
        if rec.get("task_type") == task_type:
            return rec, wrapped[idx]
    raise RuntimeError(f"No sample with task_type={task_type!r} in dry_run dataset")


def dry_run_check(
    tok: transformers.PreTrainedTokenizer,
    base_ds,
    wrapped: GrammarRemapDataset,
) -> None:
    for task_type in ("antibody", "tcr"):
        rec, sample = _find_sample_by_task(base_ds, wrapped, task_type)
        ids = sample["input_ids"]
        labels = sample["labels"]
        tokens = tok.convert_ids_to_tokens(ids)
        mask_sum = sum(1 for lab in labels if lab != -100)
        print(f"\n=== dry_run sample: {task_type} (source={rec.get('source')}) ===")
        print(f"chains: {rec['chains']}")
        print(f"tokens ({len(tokens)}): {tokens}")
        print(f"sum(maskable)==len(ids): {mask_sum == len(ids)} ({mask_sum}/{len(ids)})")
        for token_id, lab in zip(ids, labels):
            if lab != -100:
                assert lab == token_id, "labels must equal input_ids at maskable positions"
        assert mask_sum == len(ids), "OAS/OTS should have all-ones diffusion_loss_mask"
    print("\nDRY RUN OK")


def train() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    if training_args.train_mode not in {"diffusion", "bert"}:
        raise ValueError(
            f"train_mode must be 'diffusion' or 'bert', got {training_args.train_mode!r}"
        )
    if model_args.load_in_4bit:
        raise ValueError(
            "protein_pretrain 不支持 load_in_4bit，会破坏 resize"
        )

    # ----- Tokenizer + remap ------------------------------------------------------
    tok = dllm.utils.get_tokenizer(model_args=model_args)
    old_vocab = len(tok)
    remap, n_added = expand_tokenizer_and_build_remap(tok)
    remap_ids = make_remap_ids(remap)
    logger.info(
        "Tokenizer vocab: %d -> %d (added %d; remap covers %d grammar ids)",
        old_vocab,
        len(tok),
        n_added,
        len(remap),
    )

    gtok = GrammarTokenizer()
    renderer = GrammarRenderer(gtok)

    # ----- Dataset ----------------------------------------------------------------
    specs = build_immune_specs(data_args)
    max_rows = data_args.max_rows_per_source
    if training_args.dry_run and max_rows is None:
        max_rows = 8

    base_train, train_counts, _ = build_mixed_immune_dataset(
        specs, split=data_args.train_split, max_rows_per_source=max_rows
    )
    logger.info("Train source counts: %s", train_counts)
    train_ds = GrammarRemapDataset(
        base_train, renderer, remap_ids, max_length=data_args.max_length
    )

    if training_args.dry_run:
        dry_run_check(tok, base_train, train_ds)
        return

    # ----- Model (after dry_run so smoke never loads 8B) ---------------------------
    # IMPORTANT: resize the embedding/lm_head on the *base* model BEFORE any LoRA
    # wrapping. If we resized after get_peft_model, the new token rows would land
    # outside the ModulesToSaveWrapper copy and stay untrained. So load the base
    # unwrapped (lora=False), resize, then apply PEFT ourselves.
    want_lora = bool(model_args.lora)
    base_model_args = dataclasses.replace(model_args, lora=False)
    model = dllm.utils.get_model(model_args=base_model_args)
    model.resize_token_embeddings(len(tok))
    logger.info(
        "Resized embeddings to vocab_size=%d (was ~%d + %d new tokens)",
        len(tok),
        old_vocab,
        n_added,
    )

    if want_lora:
        # Keep the new-token embedding rows trainable by saving wte/ff_out fully.
        if model_args.modules_to_save is None:
            model_args.modules_to_save = "wte,ff_out"
        else:
            existing = {
                m.strip()
                for m in str(model_args.modules_to_save).split(",")
                if m.strip()
            }
            existing.update({"wte", "ff_out"})
            model_args.modules_to_save = ",".join(sorted(existing))
        logger.info("Applying LoRA after resize; modules_to_save=%s", model_args.modules_to_save)
        model = dllm.utils.load_peft(model=model, model_args=model_args)

    # ----- Eval dataset -----------------------------------------------------------
    base_eval, eval_counts, _ = build_mixed_immune_dataset(
        specs, split=data_args.eval_split, max_rows_per_source=max_rows
    )
    logger.info("Eval source counts: %s", eval_counts)
    eval_ds = GrammarRemapDataset(
        base_eval, renderer, remap_ids, max_length=data_args.max_length
    )

    # ----- Trainer / masking paradigm ---------------------------------------------
    data_collator = dllm.utils.NoAttentionMaskWrapper(
        transformers.DataCollatorForSeq2Seq(
            tok,
            return_tensors="pt",
            padding=True,
            label_pad_token_id=-100,
        )
    )

    if training_args.train_mode == "bert":
        logger.info(
            "bert mode: classic MLM select=%.2f then 80%% MASK / 10%% random / 10%% keep",
            float(training_args.mask_prob),
        )
        trainer = BERTMLMTrainer(
            model=model,
            tokenizer=tok,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
            mask_prob=float(training_args.mask_prob),
            data_collator=data_collator,
        )
    else:
        # MDLMConfig default is already "scheduler"; keep explicit for clarity.
        if training_args.loss_weight_type is None:
            training_args.loss_weight_type = "scheduler"
        logger.info(
            "diffusion mode: LinearAlphaScheduler, loss_weight_type=%s",
            training_args.loss_weight_type,
        )
        trainer = MDLMTrainer(
            model=model,
            tokenizer=tok,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
            scheduler=LinearAlphaScheduler(),
            data_collator=data_collator,
        )
    logger.info("Start training (mode=%s)...", training_args.train_mode)
    trainer.train()

    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    logger.info("Saved final checkpoint + tokenizer to %s", final_dir)


if __name__ == "__main__":
    train()
