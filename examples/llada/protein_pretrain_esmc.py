"""
ESMC-conditioned LLaDA-8B fusion training on OAS/OTS (grammar-v2).

Ablation switch ``residue_cond_mode ∈ {token, feature, add}``:
  - token:   no ESMC; residue positions use learned <res_X> embeddings only
  - feature: emb = wte*(1-m) + proj(cond)*m
  - add:     emb = wte + proj(cond)*m

Dry run (no 8B load):
    PYTHONPATH=. python examples/llada/protein_pretrain_esmc.py \\
        --dry_run True --max_rows_per_source 8
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Must precede huggingface_hub / transformers imports (constants are snapshotted at import).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import transformers
from torch.utils.data import Dataset

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import dllm
from dllm.pipelines.bioseq.datasets import (
    ImmuneSourceSpec,
    build_mixed_immune_dataset,
    oas_paired_row_to_record,
    ots_paired_row_to_record,
)
from dllm.pipelines.qwen3_vl_arch.data import (
    BioSeqChain,
    BioSeqRecord,
    GrammarBioSeqCollator,
    GrammarTokenizer,
    HuggingFaceEsmTokenizerAdapter,
)
from examples.llada.protein_fusion_model import (
    RESIDUES,
    LLaDAEsmcFusion,
    RemapCollator,
    build_remap_lookup,
    expand_llada_tokenizer_for_esmc_grammar,
)

logger = dllm.utils.get_default_logger(__name__)

OAS_DEFAULT_DIR = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/oas_previous_clean/splits"
)
OTS_DEFAULT_DIR = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean/final"
)


# Defined locally (not imported from protein_pretrain) so this entry does not pull
# in dllm.core.schedulers -> lm_eval, which is absent in the ESMC training env.
@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "oas+ots"
    oas_dir: str = OAS_DEFAULT_DIR
    ots_dir: str = OTS_DEFAULT_DIR
    train_split: str = "train"
    eval_split: str = "valid"
    max_rows_per_source: int | None = None
    # Separate (usually smaller) cap for the eval split so periodic validation
    # during a long run stays cheap; None means "use the full valid split".
    max_eval_rows_per_source: int | None = 2000
    max_length: int = 512
    esmc_path: str = "model_weights/esmc/ESMC-300M"
    max_protein_length: int = 512


def build_immune_specs(data_args: "DataArguments") -> list[ImmuneSourceSpec]:
    return [
        ImmuneSourceSpec(
            "oas", Path(data_args.oas_dir), oas_paired_row_to_record, "antibody"
        ),
        ImmuneSourceSpec(
            "ots", Path(data_args.ots_dir), ots_paired_row_to_record, "tcr"
        ),
    ]


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = ".models/LLaDA-8B-Base/oas-ots-esmc-fusion"
    residue_cond_mode: str = field(
        default="add",
        metadata={"help": 'Residue conditioning: "token" | "feature" | "add"'},
    )
    train_objective: str = field(
        default="diffusion",
        metadata={
            "help": (
                'Masking objective: "diffusion" (per-seq rate t~U(eps,1), 100% '
                '<mask>) or "bert" (fixed rate + 80/10/10 MLM).'
            )
        },
    )
    bert_mask_ratio: float = 0.15
    bert_mask_prob: float = 0.8
    bert_random_prob: float = 0.1
    freeze_encoder: bool = True
    condition_norm: bool = True
    # Enable LLaDA's own activation checkpointing on the decoder. We call it
    # directly (not HF's args.gradient_checkpointing, which would try to call
    # gradient_checkpointing_enable on the composite fusion model that lacks it).
    decoder_grad_ckpt: bool = True
    # Retain only the K checkpoints with the lowest eval loss (mirrors bioseq
    # ValLossTopKCheckpointManager). Requires eval_strategy != "no". 0 disables.
    save_top_k: int = 5
    dry_run: bool = field(
        default=False,
        metadata={"help": "Tokenizer/remap/collator smoke only; do not load 8B"},
    )
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    max_steps: int = 1000
    num_train_epochs: float = 1.0
    eval_strategy: str = "no"
    save_steps: float = 500
    logging_steps: float = 10
    report_to: str = "none"
    remove_unused_columns: bool = False
    label_names: list[str] = field(default_factory=list)
    dataloader_num_workers: int = 0


class ImmuneBioSeqDataset(Dataset):
    """CSV immune rows -> BioSeqRecord with role mapping for GrammarBioSeqCollator."""

    def __init__(self, base_ds) -> None:
        self.base_ds = base_ds

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
        raise ValueError(f"Unsupported task_type: {task_type}")

    def __getitem__(self, i: int, _retries: int = 0) -> BioSeqRecord:
        try:
            rec = self.base_ds[i]
            return BioSeqRecord(
                chains=self._roles_for(rec),
                task_type=rec["task_type"],
                source=rec.get("source", ""),
            )
        except Exception as exc:
            if _retries >= max(len(self) - 1, 0):
                raise RuntimeError(
                    f"Failed to encode sample {i} after exhausting retries"
                ) from exc
            logger.warning("Skipping bad sample %d (%s); trying next", i, exc)
            return self.__getitem__((i + 1) % len(self), _retries=_retries + 1)


class FusionTrainer(transformers.Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, **kwargs
    ):  # noqa: ANN001
        # Call through forward (model(**inputs)) so DDP/FSDP wrappers hook the
        # step and gradients sync / params gather correctly. Calling a custom
        # method like model.compute_loss would bypass the wrapper.
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys=None
    ):  # noqa: ANN001
        # BioSeqDiffusionOutput is a plain dataclass, not a dict-like ModelOutput,
        # so HF's default prediction_step (which does outputs["loss"]) would fail.
        # We only need the scalar eval loss to rank checkpoints, so return
        # loss-only. Loss is a stochastic masked-diffusion estimate (noise sampled
        # in forward); averaged over the eval set it is a stable relative metric.
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = model(**inputs).loss.detach()
        return (loss, None, None)


class TopKValLossCheckpointCallback(transformers.TrainerCallback):
    """Keep only the ``save_top_k`` checkpoints with the lowest eval loss.

    HF's ``save_total_limit`` keeps the *most recent* K (protecting just the
    single best); this replicates the bioseq ``ValLossTopKCheckpointManager``
    semantics (lower ``eval_loss`` is better) on the HF Trainer path. Runs on the
    main process only, after each checkpoint is written (eval precedes save at the
    same step, so the metric for this step is already in ``log_history``).
    """

    def __init__(self, save_top_k: int = 5, metric_name: str = "eval_loss") -> None:
        self.save_top_k = int(save_top_k)
        self.metric_name = metric_name
        self.entries: list[tuple[float, int, str]] = []  # (val_loss, step, path)

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        if self.save_top_k <= 0 or not state.is_world_process_zero:
            return
        val_loss = None
        for record in reversed(state.log_history):
            if self.metric_name in record:
                val_loss = float(record[self.metric_name])
                break
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if val_loss is None or not os.path.isdir(ckpt_dir):
            return

        self.entries = [e for e in self.entries if e[2] != ckpt_dir]
        self.entries.append((val_loss, int(state.global_step), ckpt_dir))
        self.entries.sort(key=lambda e: e[0])
        while len(self.entries) > self.save_top_k:
            _, _, worst = self.entries.pop()
            if os.path.isdir(worst):
                shutil.rmtree(worst, ignore_errors=True)
                logger.info(
                    "Pruned checkpoint outside top-%d eval_loss: %s",
                    self.save_top_k,
                    worst,
                )

        manifest = os.path.join(args.output_dir, "topk_val_manifest.json")
        payload = {
            "metric": self.metric_name,
            "save_top_k": self.save_top_k,
            "checkpoints": [
                {"val_loss": m, "step": s, "path": p} for m, s, p in self.entries
            ],
        }
        with open(manifest, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def _load_llada_tokenizer(model_name_or_path: str) -> transformers.PreTrainedTokenizer:
    """Load LLaDA tokenizer directly (mirrors dllm.utils.get_tokenizer LLaDA branch)
    without importing dllm.pipelines.a2d, which needs transformers >= 4.5x."""

    tok = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, padding_side="right", local_files_only=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"mask_token": "<|mdm_mask|>"})
    return tok


def _load_llada_decoder(model_name_or_path: str, dtype: str = "bfloat16") -> torch.nn.Module:
    """Load pretrained LLaDA-8B backbone directly via LLaDAModelLM.from_pretrained
    (avoids AutoModel registration / a2d import on older transformers)."""

    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    return LLaDAModelLM.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype, local_files_only=True
    )


def _resolve_esmc_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _ROOT / p
    return p.resolve()


def _decoder_mask_token_id(tok: transformers.PreTrainedTokenizer) -> int:
    mask_id = tok.convert_tokens_to_ids("<|mdm_mask|>")
    unk = getattr(tok, "unk_token_id", None)
    if mask_id is None or (unk is not None and int(mask_id) == int(unk)):
        mask_id = tok.mask_token_id
    if mask_id is None:
        raise RuntimeError("LLaDA tokenizer is missing <|mdm_mask|> / mask_token_id")
    return int(mask_id)


def dry_run_check(
    tok: transformers.PreTrainedTokenizer,
    dataset: ImmuneBioSeqDataset,
    collator: RemapCollator,
) -> None:
    seen: set[str] = set()
    for idx in range(len(dataset)):
        rec = dataset[idx]
        if rec.task_type in seen:
            continue
        batch = collator([rec])
        ids = batch["input_ids"][0].tolist()
        tokens = tok.convert_ids_to_tokens(ids)
        enc = batch["encoder_input_ids"]
        print(f"\n=== dry_run sample: {rec.task_type} (source={rec.source}) ===")
        print(f"chains: {[c.sequence[:20] + '...' for c in rec.chains]}")
        print(f"tokens ({len(tokens)}): {tokens}")
        print(f"encoder_input_ids shape: {tuple(enc.shape)}")
        assert enc.ndim == 3, "encoder_input_ids must be [B,C,L]"
        assert bool((batch["input_ids"] >= 0).all()), "remapped decoder ids must be >= 0"
        assert "<prots>" in tokens and "<protd>" in tokens
        assert ("<ab>" in tokens) or ("<tcr>" in tokens)
        assert any(t.startswith("<res_") for t in tokens)
        assert "<chainsep>" in tokens
        seen.add(rec.task_type)
        if seen >= {"antibody", "tcr"}:
            break
    if seen < {"antibody", "tcr"}:
        raise RuntimeError(f"dry_run missing task types; seen={seen}")
    print("\nDRY RUN OK")


def train() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    if training_args.residue_cond_mode not in {"token", "feature", "add"}:
        raise ValueError(
            f"residue_cond_mode must be token|feature|add, got "
            f"{training_args.residue_cond_mode!r}"
        )
    if training_args.train_objective not in {"diffusion", "bert"}:
        raise ValueError(
            f"train_objective must be diffusion|bert, got "
            f"{training_args.train_objective!r}"
        )
    if model_args.load_in_4bit:
        raise ValueError("protein_pretrain_esmc does not support load_in_4bit")

    esmc_path = _resolve_esmc_path(data_args.esmc_path)
    if not (esmc_path / "config.json").is_file():
        raise FileNotFoundError(f"ESMC config not found under {esmc_path}")

    # ----- Tokenizers + remap -----------------------------------------------------
    tok = _load_llada_tokenizer(model_args.model_name_or_path)
    old_vocab = len(tok)
    base_esmc = HuggingFaceEsmTokenizerAdapter.from_pretrained(
        esmc_path, local_files_only=True
    )
    gtok_esmc = GrammarTokenizer(base_esmc)
    remap, n_added = expand_llada_tokenizer_for_esmc_grammar(tok, gtok_esmc)
    lookup = build_remap_lookup(remap)
    logger.info(
        "Tokenizer vocab: %d -> %d (added %d; remap covers %d ids)",
        old_vocab,
        len(tok),
        n_added,
        len(remap),
    )

    # ----- Dataset / collator -----------------------------------------------------
    specs = build_immune_specs(data_args)
    max_rows = data_args.max_rows_per_source
    if training_args.dry_run and max_rows is None:
        max_rows = 8

    base_train, train_counts = build_mixed_immune_dataset(
        specs, split=data_args.train_split, max_rows_per_source=max_rows
    )
    logger.info("Train source counts: %s", train_counts)
    train_ds = ImmuneBioSeqDataset(base_train)
    base_collator = GrammarBioSeqCollator(
        tokenizer=gtok_esmc,
        max_sequence_length=data_args.max_length,
        max_protein_length=data_args.max_protein_length,
    )
    collator = RemapCollator(base_collator, lookup)

    if training_args.dry_run:
        dry_run_check(tok, train_ds, collator)
        return

    # ----- Model ------------------------------------------------------------------
    with (esmc_path / "config.json").open() as handle:
        esmc_cfg = json.load(handle)
    encoder_hidden = int(esmc_cfg["d_model"])

    from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import load_local_esmc_encoder

    decoder = _load_llada_decoder(
        model_args.model_name_or_path, dtype=getattr(model_args, "dtype", "bfloat16")
    )
    # LLaDA-8B pads its vocab embedding to a multiple of 128 (126464 rows) while the
    # base tokenizer is 126346; our +42 residue/grammar tokens (ids 126346..126387)
    # land inside those unused padding rows. Only grow the embedding if the tokenizer
    # actually exceeds the padded size -- shrinking would desync wte vs the LM head.
    embed_rows = int(decoder.get_input_embeddings().weight.shape[0])
    if len(tok) > embed_rows:
        decoder.resize_token_embeddings(len(tok))
        logger.info("Resized token embeddings %d -> %d", embed_rows, len(tok))
    else:
        logger.info(
            "Reusing base embedding padding rows (tok=%d <= embed=%d); no resize",
            len(tok),
            embed_rows,
        )
    if bool(getattr(training_args, "decoder_grad_ckpt", True)):
        decoder.gradient_checkpointing_enable()
        if getattr(decoder, "config", None) is not None:
            decoder.config.use_cache = False
        logger.info("Enabled LLaDA decoder gradient checkpointing")
    encoder = load_local_esmc_encoder(esmc_path)
    # LLaDA-space ids of the <res_*> tokens, for the BERT "10% random" replacement.
    residue_token_ids = [int(tok.convert_tokens_to_ids(f"<res_{aa}>")) for aa in RESIDUES]
    model = LLaDAEsmcFusion(
        decoder=decoder,
        encoder=encoder,
        encoder_hidden_size=encoder_hidden,
        decoder_mask_token_id=_decoder_mask_token_id(tok),
        encoder_mask_token_id=int(gtok_esmc.mask_token_id),
        residue_cond_mode=training_args.residue_cond_mode,
        condition_norm=bool(training_args.condition_norm),
        freeze_encoder=bool(training_args.freeze_encoder),
        train_objective=training_args.train_objective,
        bert_mask_ratio=float(training_args.bert_mask_ratio),
        bert_mask_prob=float(training_args.bert_mask_prob),
        bert_random_prob=float(training_args.bert_random_prob),
        residue_token_ids=residue_token_ids,
    )
    # Decoder is bf16 (from_pretrained) while ESMC/condition heads default to fp32;
    # unify to bf16 so the FSDP root flat-param group has a single dtype.
    model = model.to(torch.bfloat16)

    eval_rows = data_args.max_eval_rows_per_source
    if training_args.dry_run and eval_rows is None:
        eval_rows = 8
    base_eval, eval_counts = build_mixed_immune_dataset(
        specs, split=data_args.eval_split, max_rows_per_source=eval_rows
    )
    logger.info("Eval source counts: %s", eval_counts)
    eval_ds = ImmuneBioSeqDataset(base_eval)

    training_args.remove_unused_columns = False
    training_args.label_names = []

    trainer = FusionTrainer(
        model=model,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        data_collator=collator,
    )
    # Retain the K lowest eval_loss checkpoints (bioseq top-k semantics). Needs a
    # metric, so only active when eval runs; otherwise HF keeps everything.
    if int(getattr(training_args, "save_top_k", 0)) > 0:
        if str(training_args.eval_strategy) == "no":
            logger.warning(
                "save_top_k=%d requested but eval_strategy=no; top-k pruning is "
                "disabled (no eval_loss to rank by).",
                training_args.save_top_k,
            )
        else:
            trainer.add_callback(
                TopKValLossCheckpointCallback(save_top_k=int(training_args.save_top_k))
            )
    logger.info(
        "Start fusion training (objective=%s, residue_cond_mode=%s)...",
        training_args.train_objective,
        training_args.residue_cond_mode,
    )
    trainer.train()

    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    logger.info("Saved final checkpoint + tokenizer to %s", final_dir)


if __name__ == "__main__":
    train()
