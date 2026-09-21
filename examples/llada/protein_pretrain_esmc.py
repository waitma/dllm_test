"""
ESMC-conditioned LLaDA-8B fusion training on OAS/OTS (grammar-v2).

Ablation switch ``residue_cond_mode ∈ {token, feature, add}``:
  - token:   no ESMC; residue positions use learned <res_X> embeddings only
  - feature: emb = wte*(1-m) + proj(cond)*m
  - add:     emb = wte + proj(cond)*m

Dry run (no 8B load):
    PYTHONPATH=. python examples/llada/protein_pretrain_esmc.py \\
        --dry_run True --prepared_data_dir /absolute/path/to/prepared/immune_v3_heterotypic
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Must precede huggingface_hub / transformers imports (constants are snapshotted at import).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import transformers
from torch.utils.data import ConcatDataset

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import dllm
from dllm.pipelines.immune_llada.data import (
    GrammarBioSeqCollator,
    GrammarTokenizer,
    HuggingFaceEsmTokenizerAdapter,
    load_prepared_dataset,
)
from dllm.pipelines.immune_llada.data.registry import parse_sources

from examples.llada.protein_fusion_model import (
    RESIDUES,
    LLaDAEsmcFusion,
    RemapCollator,
    build_remap_lookup,
    decoder_mask_token_id as _decoder_mask_token_id,
    expand_llada_tokenizer_for_esmc_grammar,
    load_fusion_config,
    save_fusion_config,
)

logger = dllm.utils.get_default_logger(__name__)


FIXED_HEAVY_ENCODER_LENGTH = 168
FIXED_LIGHT_ENCODER_LENGTH = 136
FIXED_HEAVY_DECODER_SLOTS = 167
FIXED_LIGHT_DECODER_SLOTS = 135
FIXED_HEAVY_RESIDUE_MAX = 166
FIXED_LIGHT_RESIDUE_MAX = 134


# Defined locally (not imported from protein_pretrain) so this entry does not pull
# in dllm.core.schedulers -> lm_eval, which is absent in the ESMC training env.
@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"
    # Decoder init: "pretrained" loads the real LLaDA-8B weights; "scratch" builds a
    # randomly-initialized LLaDA (optionally shrunk via the size overrides below),
    # borrowing only the tokenizer/config from model_name_or_path. The ~270M small
    # track uses scratch with d=768/L=8/h=12.
    decoder_init: str = "pretrained"
    decoder_d_model: int | None = None
    decoder_n_layers: int | None = None
    decoder_n_heads: int | None = None
    decoder_mlp_hidden: int | None = None
    decoder_weight_tying: bool = False


@dataclass
class DataArguments(dllm.utils.DataArguments):
    # This is a semantic prepared dataset, not a raw source directory. Raw CSV/
    # JSONL parsing and all rejecting filters happen in scripts/data preprocessing.
    prepared_data_dir: str = "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v3_heterotypic"
    dataset_args: str = "oas+ots"
    # Deprecated compatibility flags accepted by older job YAMLs. They are never
    # read: all source paths, blocklists, row caps, and sampling now belong to the
    # offline preparation command and its manifest.
    oas_dir: str | None = None
    ots_dir: str | None = None
    asd_antibody_dir: str | None = None
    asd_nanobody_dir: str | None = None
    trait_dir: str | None = None
    tcr_native_dir: str | None = None
    tcr_papers_dir: str | None = None
    tcr_repertoire_dir: str | None = None
    replaces_trait_blocklist: str | None = None
    t4_refbinder_blocklist: str | None = None
    t2t3_eval_blocklist: str | None = None
    ots_benchmark_blocklist: str | None = None
    oas_benchmark_blocklist: str | None = None
    asd_antibody_benchmark_blocklist: str | None = None
    asd_nanobody_benchmark_blocklist: str | None = None
    trait_benchmark_blocklist: str | None = None
    max_rows_per_source: int | None = None
    max_eval_rows_per_source: int | None = None
    subsample_seed: int | None = None
    train_split: str = "train"
    eval_split: str = "valid"
    # Select source subsets from the prepared manifest for mixed training or
    # per-source evaluation. No runtime row cap or dynamic filtering is applied.
    eval_per_source: bool = True
    max_length: int = 1024
    esmc_path: str = "model_weights/esmc/ESMC-300M"
    max_protein_length: int = 1024





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
    # BERT objective only: when True, compute MLM loss over every *real* residue
    # in all chains (including fixed context: MHC/peptide/antigen) plus
    # supervised relation targets (relation_target_mask: trainable
    # <binding>/<nonbinding> only -- never relation_token_mask, which also
    # lights up the MHC->peptide presentation <binding> and OAS/OTS <unknown>
    # prefix). When False, restrict to generated chains
    # (diffusion_eligible_mask, which already contains those relation targets).
    # No effect on the diffusion objective.
    # Synthetic completion placeholders (the literal `X` that v4/v5 receptor
    # completion writes for missing chains/regions) are excluded either way --
    # see sample_bioseq_bert_noise, which ANDs in ~synthetic_residue_mask.
    bert_all_chains: bool = True
    # Diffusion objective only: the mirror image of bert_all_chains. When True the
    # eligible set becomes every *real* residue of every chain, so fixed grammar
    # context (antigen / MHC / peptide) is corrupted and scored like a generated
    # chain -- i.e. no chain is held fixed. Synthetic completion placeholders stay
    # excluded (all_residue_eligible_mask ANDs in ~synthetic_residue_mask).
    # False (the default) keeps the historical generated-only behaviour so
    # existing diffusion checkpoints stay continuable.
    diffusion_all_chains: bool = False
    # GIDD hybrid noise + Ophiuchus ratio / focal / reciprocal loss. Defaults
    # keep the original absorbing-mask + unweighted CE so existing diffusion
    # checkpoints continue-train unchanged.
    gidd_uniform_ratio: float = 0.0
    gidd_gamma: float = 1.0
    independent_loss_ratio: float = 0.0
    single_chain_ratio: float = 0.0
    heavy2light_loss_ratio: float = 0.0
    light2heavy_loss_ratio: float = 0.0
    joint_loss_ratio: float = 1.0
    focal: bool = False
    focal_gamma: float = 1.0
    loss_weight_type: str = field(
        default="none",
        metadata={"help": 'Diffusion token weights: "none" | "uniform" | "reciprocal"'},
    )
    softmin_snr: float = 20.0
    heavy_loss_weight: float = 1.0
    light_loss_weight: float = 1.0
    # Opt-in pairing auxiliary. ``none`` keeps the reconstruction-only loss so
    # existing jobs and eval_loss ranking are unchanged. See MULTI_CHAIN_RELATION.md.
    relation_aux: str = field(
        default="none",
        metadata={"help": 'Pairing aux: "none" | "cognate" | "chain_drop" | "both"'},
    )
    relation_aux_weight: float = 0.1
    relation_aux_temperature: float = 0.07
    # One switch controls the complete fixed-canvas v2 policy. In particular,
    # v2 automatically enables EOS prediction and the separate chain-EOS token
    # when the native decoder EOS aliases padding.
    fixed_receptor_lengths: bool = False
    freeze_encoder: bool = True
    condition_norm: bool = True
    # Enable LLaDA's own activation checkpointing on the decoder. We call it
    # directly (not HF's args.gradient_checkpointing, which would try to call
    # gradient_checkpointing_enable on the composite fusion model that lacks it).
    decoder_grad_ckpt: bool = True
    # Retain only the K checkpoints with the lowest eval loss (mirrors bioseq
    # ValLossTopKCheckpointManager). Requires eval_strategy != "no". 0 disables.
    save_top_k: int = 3
    # Must be False so the Trainer actually writes optimizer / FSDP / RNG for the
    # *latest* retained ckpt (see slim_checkpoints). Older top-k dirs are slimmed
    # to weights-only after each save (~32GB); only the newest kept dir stays full
    # (~125GB) for mid-run resume.
    save_only_model: bool = False
    # After each save + top-k prune: delete resume-only artefacts from every
    # retained checkpoint *except the latest (highest step)*. No-op on the
    # latest dir. Requires save_only_model=False to have anything to keep there.
    slim_checkpoints: bool = True
    dry_run: bool = field(
        default=False,
        metadata={"help": "Tokenizer/remap/collator smoke only; do not load 8B"},
    )
    # Load fusion ``model.safetensors`` into a freshly built model *before*
    # FSDP wraps it. Used when the source pack has a different FSDP world size
    # (e.g. 4-gpu -> 8-gpu): optimizer / pytorch_model_fsdp.bin cannot transfer,
    # but the consolidated weights can. Ignored when resume_from_checkpoint is set.
    init_fusion_weights: str | None = None
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



class FusionTrainer(transformers.Trainer):
    # Source row counts retained for composition reporting; eval_loss gives each
    # source equal weight regardless of its row count.
    eval_source_rows: dict[str, int] | None = None

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"
    ):  # noqa: ANN001
        """Add a combined ``eval_loss`` when evaluating a dict of sources.

        With a dict ``eval_dataset``, ``Trainer.evaluate`` recurses once per key
        and emits only ``eval_<source>_loss`` -- no plain ``eval_loss``, which is
        what ``TopKValLossCheckpointCallback`` and ``metric_for_best_model`` rank
        by. Average the per-source losses with equal weight and ``log`` the
        result so it reaches both ``state.log_history`` and wandb. This is a
        macro-average across sources; each source's own loss is unchanged.
        """

        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        resolved = eval_dataset if eval_dataset is not None else self.eval_dataset
        loss_key = f"{metric_key_prefix}_loss"
        if (
            not isinstance(resolved, dict)
            or loss_key in metrics
            or not resolved
        ):
            return metrics

        source_losses = []
        for name in resolved:
            value = metrics.get(f"{metric_key_prefix}_{name}_loss")
            if value is None:
                continue
            source_losses.append(float(value))
        # Publish only when every source reported: a mean over a shifting subset
        # would rank checkpoints against a moving target.
        if len(source_losses) == len(resolved):
            metrics[loss_key] = sum(source_losses) / len(source_losses)
            self.log({loss_key: metrics[loss_key]})
        return metrics

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


def _unwrap_fusion_module(model):  # noqa: ANN001
    """Walk DDP/FSDP wrappers to the fusion module that holds aux state."""

    seen: set[int] = set()
    current = model
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_last_relation_aux_loss") and hasattr(
            current, "_relation_aux"
        ):
            return current
        nxt = getattr(current, "module", None)
        if nxt is None:
            nxt = getattr(current, "_fsdp_wrapped_module", None)
        current = nxt
    return model


class RelationAuxLogCallback(transformers.TrainerCallback):
    """Copy ``relation_aux_loss`` onto the Trainer log dict so wandb sees it.

    Inserted at the front of the callback list: HF's WandbCallback reads ``logs``
    in ``on_log``, so a later callback would miss the extra key.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return
        fusion = _unwrap_fusion_module(kwargs.get("model"))
        value = getattr(fusion, "_last_relation_aux_loss", None)
        if value is None:
            return
        logs["relation_aux_loss"] = float(value)


class CanvasDropLogCallback(transformers.TrainerCallback):
    """Surface the fixed-canvas collator's runtime drop ledger in the logs.

    Counts are per-process and only cover the shards this rank actually read,
    so they are a monitoring signal, not a corpus-wide filter report. A sudden
    rise means the canvas no longer fits the data being fed in.
    """

    def __init__(self, collator: Any) -> None:
        self.collator = collator

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return
        counts = getattr(self.collator, "drop_counts", None)
        if not counts:
            return
        for reason, value in counts.items():
            logs[f"canvas_drop/{reason}"] = int(value)
        logs["canvas_drop/total"] = int(sum(counts.values()))


# Resume-only artefacts written by HF Trainer + FSDP when save_only_model=False.
# Dropping them keeps model.safetensors (~32GB) and cuts a ckpt from ~125GB to ~32GB.
_BULKY_CKPT_FILES = (
    "optimizer.bin",
    "pytorch_model_fsdp.bin",
    "pytorch_model.bin",
    "scheduler.pt",
    "training_args.bin",
    "trainer_state.json",
)


def slim_checkpoint_dir(ckpt_dir: str) -> list[str]:
    """Remove resume-only files; leave model weights + tokenizer files.

    Returns the list of deleted basenames (empty if nothing removed).
    """

    removed: list[str] = []
    if not os.path.isdir(ckpt_dir):
        return removed
    for name in _BULKY_CKPT_FILES:
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            os.remove(path)
            removed.append(name)
    # Per-rank RNG dumps are tiny individually but useless without optimizer.
    for name in list(os.listdir(ckpt_dir)):
        if name.startswith("rng_state") and name.endswith(".pth"):
            os.remove(os.path.join(ckpt_dir, name))
            removed.append(name)
    return removed


class FusionConfigCheckpointCallback(transformers.TrainerCallback):
    """Write fusion policy metadata for every Trainer checkpoint on rank zero."""

    def __init__(self, tokenizer: Any | None = None) -> None:
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        if not state.is_world_process_zero:
            return
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
        if not checkpoint_dir.is_dir():
            return
        model = kwargs.get("model")
        if model is None:
            raise RuntimeError("Trainer save callback did not receive the fusion model")
        tokenizer = (
            kwargs.get("processing_class")
            or kwargs.get("tokenizer")
            or self.tokenizer
        )
        save_fusion_config(checkpoint_dir, model, tokenizer=tokenizer)


class TopKValLossCheckpointCallback(transformers.TrainerCallback):
    """Keep the best-``save_top_k`` by eval_loss **plus** the latest (resumable) ckpt.

    HF's ``save_total_limit`` keeps the *most recent* K (protecting just the
    single best); this replicates bioseq ``ValLossTopKCheckpointManager``
    semantics (lower ``eval_loss`` is better) on the HF Trainer path, and adds an
    explicit guarantee that the *latest* checkpoint is always retained so training
    can resume from the newest point.

    Two retention guarantees are decoupled (a checkpoint can satisfy both):

    - **Best-K by eval_loss**: the ``save_top_k`` checkpoints with the lowest
      ``eval_loss`` are kept for model quality.
    - **Latest (resume)**: the highest-step checkpoint is *always* kept, even if
      its ``eval_loss`` is outside the best-K, so mid-run resume never regresses to
      an older step.

    After each save (eval precedes save at the same step, so the metric is already
    in ``log_history``):

    1. Prune every tracked checkpoint that is neither in the best-K nor the latest.
    2. If ``slim_checkpoints``, strip optimizer / FSDP / RNG / scheduler from every
       retained dir *except the latest* (highest step). Only that latest dir keeps
       the full resume payload written by the Trainer (requires
       ``save_only_model=False``); every other kept dir is weights-only.
    """

    def __init__(
        self,
        save_top_k: int = 3,
        metric_name: str = "eval_loss",
        slim_checkpoints: bool = True,
    ) -> None:
        self.save_top_k = int(save_top_k)
        self.metric_name = metric_name
        self.slim_checkpoints = bool(slim_checkpoints)
        self.entries: list[tuple[float, int, str]] = []  # (val_loss, step, path)

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        if not state.is_world_process_zero:
            return
        step = int(state.global_step)
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        if not os.path.isdir(ckpt_dir):
            return

        # eval_loss for the just-written checkpoint (eval precedes save at the same
        # step, so it is already in log_history). Missing metric -> +inf so it never
        # wins a best-K slot, but it is still protected below as the latest ckpt.
        val_loss = None
        for record in reversed(state.log_history):
            if self.metric_name in record:
                val_loss = float(record[self.metric_name])
                break
        rank_loss = val_loss if val_loss is not None else float("inf")

        # Register/refresh this checkpoint, then drop any dirs that vanished.
        self.entries = [e for e in self.entries if e[2] != ckpt_dir]
        self.entries.append((rank_loss, step, ckpt_dir))
        self.entries = [e for e in self.entries if os.path.isdir(e[2])]

        # Latest (highest step) is always kept with full resume state; the best-K by
        # eval_loss are kept too (slimmed to weights-only below).
        latest_step = max(s for _, s, _ in self.entries)
        latest_path = next(p for _, s, p in self.entries if s == latest_step)
        keep: set[str] = {latest_path}
        if self.save_top_k > 0:
            for _, _, path in sorted(self.entries, key=lambda e: e[0])[: self.save_top_k]:
                keep.add(path)
        else:
            keep.update(p for _, _, p in self.entries)

        # Prune checkpoints that are neither in the best-K nor the latest.
        for _, _, path in list(self.entries):
            if path not in keep and os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                logger.info(
                    "Pruned checkpoint outside best-%d eval_loss and not latest: %s",
                    self.save_top_k,
                    path,
                )
        self.entries = [e for e in self.entries if os.path.isdir(e[2])]

        # Slim every retained checkpoint except the latest (resume) one.
        resume_path = latest_path
        if self.slim_checkpoints:
            for _, _, path in self.entries:
                if path == resume_path:
                    continue
                removed = slim_checkpoint_dir(path)
                if removed:
                    logger.info(
                        "Slimmed non-latest %s (removed %s)", path, ", ".join(removed)
                    )
            logger.info("Kept full resume state at latest checkpoint: %s", resume_path)

        manifest = os.path.join(args.output_dir, "topk_val_manifest.json")
        payload = {
            "metric": self.metric_name,
            "save_top_k": self.save_top_k,
            "resume_checkpoint": resume_path,
            "checkpoints": [
                {
                    "val_loss": (None if m == float("inf") else m),
                    "step": s,
                    "path": p,
                    "resumable": p == resume_path,
                }
                for m, s, p in sorted(self.entries, key=lambda e: e[0])
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


def _load_llada_decoder(
    model_name_or_path: str,
    dtype: str = "bfloat16",
    *,
    decoder_init: str = "pretrained",
    d_model: int | None = None,
    n_layers: int | None = None,
    n_heads: int | None = None,
    mlp_hidden: int | None = None,
    weight_tying: bool = False,
) -> torch.nn.Module:
    """Load the LLaDA decoder backbone.

    ``decoder_init="pretrained"`` loads the real LLaDA-8B weights via
    ``LLaDAModelLM.from_pretrained`` (avoids AutoModel registration / a2d import
    on older transformers). ``decoder_init="scratch"`` builds a
    randomly-initialized LLaDA from the (optionally shrunk) config, borrowing only
    the tokenizer/config from ``model_name_or_path`` — used for the ~270M
    small-model track (d=768/L=8/h=12).
    """

    from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    if decoder_init == "pretrained":
        return LLaDAModelLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, local_files_only=True
        )
    if decoder_init != "scratch":
        raise ValueError(f"decoder_init must be pretrained|scratch, got {decoder_init!r}")

    from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig

    # Take config (dims, vocab/embedding padding, special-token ids) from the 8B
    # checkpoint, then optionally shrink and disable weight tying (untied ff_out
    # avoids safetensors shared-tensor save issues). init_params=True random-inits.
    cfg = LLaDAConfig.from_pretrained(model_name_or_path, local_files_only=True)
    if d_model is not None:
        cfg.d_model = int(d_model)
    if n_layers is not None:
        cfg.n_layers = int(n_layers)
    if n_heads is not None:
        cfg.n_heads = int(n_heads)
        # The 8B config pins n_kv_heads=n_heads(=32); after shrinking n_heads it
        # would no longer divide the (now smaller) query heads. Realign to plain
        # multi-head attention so num_q_heads % num_kv_heads == 0.
        cfg.n_kv_heads = int(n_heads)
    if mlp_hidden is not None:
        cfg.mlp_hidden_size = int(mlp_hidden)
    cfg.weight_tying = bool(weight_tying)
    if int(cfg.d_model) % int(cfg.n_heads) != 0:
        raise ValueError(
            f"decoder_d_model={cfg.d_model} must be divisible by decoder_n_heads={cfg.n_heads}"
        )
    model = LLaDAModelLM(cfg, init_params=True)
    return model.to(torch_dtype)


def _resolve_esmc_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _ROOT / p
    return p.resolve()


def dry_run_check(
    tok: transformers.PreTrainedTokenizer,
    dataset: Any,
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

def _resume_checkpoint_dir(resume_ckpt: bool | str | None, output_dir: str) -> Path | None:
    """Resolve the checkpoint whose policy must match this training invocation."""

    if resume_ckpt is True:
        candidates = [
            path
            for path in Path(output_dir).glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
        ]
        return (
            max(candidates, key=lambda path: int(path.name.split("-")[-1]))
            if candidates
            else None
        )
    if not resume_ckpt:
        return None
    path = Path(str(resume_ckpt))
    if path.name == "model.safetensors":
        path = path.parent
    return path


def _validate_resume_policy(
    resume_ckpt: bool | str | None,
    output_dir: str,
    fixed_receptor_lengths: bool,
    *,
    decoder_chain_eos_token_id: int | None = None,
    decoder_chain_eos_policy: str | None = None,
) -> None:
    """Reject legacy/v2 or fixed-canvas tokenizer policy changes before resume."""

    checkpoint_dir = _resume_checkpoint_dir(resume_ckpt, output_dir)
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        return
    config = load_fusion_config(checkpoint_dir)
    saved_fixed = bool(config.get("fixed_receptor_lengths", False))
    if saved_fixed != bool(fixed_receptor_lengths):
        raise ValueError(
            "resume checkpoint policy mismatch: checkpoint has "
            f"fixed_receptor_lengths={saved_fixed}, but this run requests "
            f"{bool(fixed_receptor_lengths)} ({checkpoint_dir}). Refusing to "
            "resume across legacy and fixed-canvas v2 policies."
        )
    if saved_fixed:
        expected = {
            "fixed_heavy_encoder_length": FIXED_HEAVY_ENCODER_LENGTH,
            "fixed_light_encoder_length": FIXED_LIGHT_ENCODER_LENGTH,
            "fixed_heavy_decoder_slots": FIXED_HEAVY_DECODER_SLOTS,
            "fixed_light_decoder_slots": FIXED_LIGHT_DECODER_SLOTS,
            "fixed_heavy_residue_max": FIXED_HEAVY_RESIDUE_MAX,
            "fixed_light_residue_max": FIXED_LIGHT_RESIDUE_MAX,
        }
        for key, value in expected.items():
            if int(config.get(key, -1)) != value:
                raise ValueError(
                    f"resume checkpoint has incompatible {key}={config.get(key)!r}; "
                    f"expected {value}: {checkpoint_dir}"
                )
        if not bool(config.get("predict_eos", False)):
            raise ValueError(
                f"fixed-canvas resume checkpoint does not enable predict_eos: {checkpoint_dir}"
            )
        if config.get("decoder_chain_eos_token_id") is None or config.get(
            "decoder_chain_eos_policy"
        ) in (None, "legacy_eos_alias"):
            raise ValueError(
                "fixed-canvas resume checkpoint is missing a safe decoder chain "
                f"EOS policy: {checkpoint_dir}"
            )
        if decoder_chain_eos_token_id is None or decoder_chain_eos_policy in (
            None, "legacy_eos_alias"
        ):
            raise ValueError(
                "fixed-canvas resume requires a safe decoder chain EOS ID/policy "
                f"from the current tokenizer: {checkpoint_dir}"
            )
        saved_eos_id = int(config["decoder_chain_eos_token_id"])
        if saved_eos_id != int(decoder_chain_eos_token_id):
            raise ValueError(
                "decoder chain EOS id mismatch between resume checkpoint and current tokenizer: "
                f"sidecar={saved_eos_id}, tokenizer={decoder_chain_eos_token_id} "
                f"({checkpoint_dir})"
            )
        saved_eos_policy = str(config["decoder_chain_eos_policy"])
        if saved_eos_policy != str(decoder_chain_eos_policy):
            raise ValueError(
                "decoder chain EOS policy mismatch between resume checkpoint and current tokenizer: "
                f"sidecar={saved_eos_policy!r}, tokenizer={decoder_chain_eos_policy!r} "
                f"({checkpoint_dir})"
            )


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
    if training_args.fixed_receptor_lengths and training_args.residue_cond_mode != "add":
        raise ValueError(
            "fixed_receptor_lengths=True requires residue_cond_mode='add'; "
            "v2 does not support replacement/token fusion"
        )
    if training_args.train_objective not in {"diffusion", "bert"}:
        raise ValueError(
            f"train_objective must be diffusion|bert, got "
            f"{training_args.train_objective!r}"
        )
    if training_args.relation_aux not in {"none", "cognate", "chain_drop", "both"}:
        raise ValueError(
            "relation_aux must be none|cognate|chain_drop|both, got "
            f"{training_args.relation_aux!r}"
        )
    if model_args.decoder_init not in {"pretrained", "scratch"}:
        raise ValueError(
            f"decoder_init must be pretrained|scratch, got {model_args.decoder_init!r}"
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
    remap, n_added = expand_llada_tokenizer_for_esmc_grammar(
        tok, gtok_esmc, allow_chain_eos_token=bool(training_args.fixed_receptor_lengths)
    )
    lookup = build_remap_lookup(remap)
    logger.info(
        "Tokenizer vocab: %d -> %d (added %d; remap covers %d ids)",
        old_vocab,
        len(tok),
        n_added,
        len(remap),
    )

    # ----- Prepared semantic dataset / collator ----------------------------------
    selected_sources = parse_sources(data_args.dataset_args)
    train_parts = [
        load_prepared_dataset(
            data_args.prepared_data_dir,
            split=data_args.train_split,
            source=source,
        )
        for source in selected_sources
    ]
    train_counts = {source: len(part) for source, part in zip(selected_sources, train_parts)}
    logger.info("Prepared train source counts: %s", train_counts)
    train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    base_collator = GrammarBioSeqCollator(
        tokenizer=gtok_esmc,
        max_sequence_length=data_args.max_length,
        max_protein_length=data_args.max_protein_length,
        fixed_receptor_lengths=bool(training_args.fixed_receptor_lengths),
        # Prepared shards were budgeted under the old variable-length grammar,
        # so the fixed canvas makes a small tail unrenderable. Auditing
        # immune_v6_binding_only found 4,580 / 7,381,499 such rows (0.062%);
        # drop and count them rather than aborting the run.
        drop_overflow_records=bool(training_args.fixed_receptor_lengths),
    )
    collator = RemapCollator(base_collator, lookup)

    if training_args.dry_run:
        dry_run_check(tok, train_ds, collator)
        return

    resume_ckpt = getattr(training_args, "resume_from_checkpoint", None)
    if resume_ckpt in (None, "", "False", "false", "0"):
        resume_ckpt = None
    elif resume_ckpt in (True, "True", "true", "1"):
        resume_ckpt = True
    else:
        resume_ckpt = str(resume_ckpt)
    _validate_resume_policy(
        resume_ckpt,
        training_args.output_dir,
        bool(training_args.fixed_receptor_lengths),
        decoder_chain_eos_token_id=int(getattr(tok, "_fusion_decoder_chain_eos_token_id")),
        decoder_chain_eos_policy=str(getattr(tok, "_fusion_decoder_chain_eos_policy")),
    )

    # ----- Model ------------------------------------------------------------------
    with (esmc_path / "config.json").open() as handle:
        esmc_cfg = json.load(handle)
    encoder_hidden = int(esmc_cfg["d_model"])

    from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import load_local_esmc_encoder

    decoder = _load_llada_decoder(
        model_args.model_name_or_path,
        dtype=getattr(model_args, "dtype", "bfloat16"),
        decoder_init=model_args.decoder_init,
        d_model=model_args.decoder_d_model,
        n_layers=model_args.decoder_n_layers,
        n_heads=model_args.decoder_n_heads,
        mlp_hidden=model_args.decoder_mlp_hidden,
        weight_tying=model_args.decoder_weight_tying,
    )
    logger.info(
        "Decoder init=%s (d_model=%s n_layers=%s n_heads=%s mlp_hidden=%s weight_tying=%s)",
        model_args.decoder_init,
        model_args.decoder_d_model,
        model_args.decoder_n_layers,
        model_args.decoder_n_heads,
        model_args.decoder_mlp_hidden,
        model_args.decoder_weight_tying,
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
        bert_all_chains=bool(training_args.bert_all_chains),
        diffusion_all_chains=bool(training_args.diffusion_all_chains),
        residue_token_ids=residue_token_ids,
        gidd_uniform_ratio=float(training_args.gidd_uniform_ratio),
        gidd_gamma=float(training_args.gidd_gamma),
        independent_loss_ratio=float(training_args.independent_loss_ratio),
        single_chain_ratio=float(training_args.single_chain_ratio),
        heavy2light_loss_ratio=float(training_args.heavy2light_loss_ratio),
        light2heavy_loss_ratio=float(training_args.light2heavy_loss_ratio),
        joint_loss_ratio=float(training_args.joint_loss_ratio),
        focal=bool(training_args.focal),
        focal_gamma=float(training_args.focal_gamma),
        loss_weight_type=str(training_args.loss_weight_type),
        softmin_snr=float(training_args.softmin_snr),
        heavy_loss_weight=float(training_args.heavy_loss_weight),
        light_loss_weight=float(training_args.light_loss_weight),
        relation_aux=str(training_args.relation_aux),
        relation_aux_weight=float(training_args.relation_aux_weight),
        relation_aux_temperature=float(training_args.relation_aux_temperature),
        fixed_receptor_lengths=bool(training_args.fixed_receptor_lengths),
        predict_eos=bool(training_args.fixed_receptor_lengths),
        decoder_chain_eos_token_id=(
            int(getattr(tok, "_fusion_decoder_chain_eos_token_id"))
            if bool(training_args.fixed_receptor_lengths)
            else None
        ),
    )
    # Decoder is bf16 (from_pretrained) while ESMC/condition heads default to fp32;
    # unify to bf16 so the FSDP root flat-param group has a single dtype.
    fixed_receptor_lengths = bool(training_args.fixed_receptor_lengths)
    model = model.to(torch.bfloat16)
    model._fixed_receptor_lengths = fixed_receptor_lengths
    model.config.fixed_receptor_lengths = fixed_receptor_lengths
    model.config.fixed_heavy_encoder_length = FIXED_HEAVY_ENCODER_LENGTH
    model.config.fixed_light_encoder_length = FIXED_LIGHT_ENCODER_LENGTH
    model.config.fixed_heavy_decoder_slots = FIXED_HEAVY_DECODER_SLOTS
    model.config.fixed_light_decoder_slots = FIXED_LIGHT_DECODER_SLOTS
    model.config.fixed_heavy_residue_max = FIXED_HEAVY_RESIDUE_MAX
    model.config.fixed_light_residue_max = FIXED_LIGHT_RESIDUE_MAX
    model.config.decoder_chain_eos_token_id = int(
        getattr(tok, "_fusion_decoder_chain_eos_token_id")
    )
    model.config.decoder_chain_eos_policy = str(
        getattr(tok, "_fusion_decoder_chain_eos_policy")
    )
    model.config.predict_eos = fixed_receptor_lengths

    init_weights = getattr(training_args, "init_fusion_weights", None)
    if init_weights in (None, "", "False", "false", "0"):
        init_weights = None
    else:
        init_weights = str(init_weights)
    if resume_ckpt and init_weights:
        logger.warning(
            "resume_from_checkpoint=%s is set; ignoring init_fusion_weights=%s",
            resume_ckpt,
            init_weights,
        )
        init_weights = None
    if init_weights:
        from pathlib import Path as _Path

        from safetensors.torch import load_file as _load_safetensors

        weight_path = _Path(init_weights)
        if weight_path.is_dir():
            weight_path = weight_path / "model.safetensors"
        if not weight_path.is_file():
            raise FileNotFoundError(f"init_fusion_weights not found: {weight_path}")
        state = _load_safetensors(str(weight_path), device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"init_fusion_weights missing {len(missing)} keys, sample={missing[:8]}"
            )
        logger.info(
            "Loaded fusion weights from %s (tensors=%d, unexpected=%d)",
            weight_path,
            len(state),
            len(unexpected),
        )
        del state

    # Prepared validation shards are fixed during preprocessing. Evaluation is
    # intentionally full-split: no runtime reservoir sampling or row filtering.
    eval_parts = [
        load_prepared_dataset(
            data_args.prepared_data_dir,
            split=data_args.eval_split,
            source=source,
        )
        for source in selected_sources
    ]
    eval_counts = {source: len(part) for source, part in zip(selected_sources, eval_parts)}
    logger.info("Prepared eval source counts: %s", eval_counts)

    # Per-source eval datasets make it visible which data regime drives checkpoint
    # selection. Every dataset is already prepared; this branch only selects a
    # manifest shard group and does not parse/filter/retry samples.
    eval_source_rows: dict[str, int] | None = None
    if data_args.eval_per_source and len(eval_parts) > 1:
        eval_ds = {source: part for source, part in zip(selected_sources, eval_parts)}
        eval_source_rows = {source: len(part) for source, part in zip(selected_sources, eval_parts)}
        total_eval = sum(eval_source_rows.values())
        logger.info(
            "Eval set composition (%d rows): %s",
            total_eval,
            ", ".join(
                f"{name}={n} ({n / total_eval:.1%})"
                for name, n in sorted(eval_source_rows.items(), key=lambda kv: -kv[1])
            ),
        )
    else:
        eval_ds = eval_parts[0] if len(eval_parts) == 1 else ConcatDataset(eval_parts)

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
    trainer.eval_source_rows = eval_source_rows
    # Aux scalar must land in ``logs`` before WandbCallback.on_log reads it.
    trainer.callback_handler.callbacks.insert(0, RelationAuxLogCallback())
    if bool(training_args.fixed_receptor_lengths):
        trainer.callback_handler.callbacks.insert(0, CanvasDropLogCallback(collator))
    # The sidecar is independent of top-k retention: every Trainer checkpoint
    # gets its policy metadata before any later retention callback runs.
    trainer.add_callback(FusionConfigCheckpointCallback(tokenizer=tok))
    # Retain the K lowest eval_loss checkpoints (bioseq top-k semantics). Needs a
    # metric, so only active when eval runs; otherwise HF keeps everything.
    save_top_k = int(getattr(training_args, "save_top_k", 0))
    slim_ckpts = bool(getattr(training_args, "slim_checkpoints", True))
    if save_top_k > 0 and str(training_args.eval_strategy) == "no":
        logger.warning(
            "save_top_k=%d requested but eval_strategy=no; top-k pruning is "
            "disabled (no eval_loss to rank by).",
            save_top_k,
        )
        save_top_k = 0
    if save_top_k > 0 or slim_ckpts:
        trainer.add_callback(
            TopKValLossCheckpointCallback(
                save_top_k=save_top_k,
                slim_checkpoints=slim_ckpts,
            )
        )
    save_only_model = bool(getattr(training_args, "save_only_model", False))
    if slim_ckpts and save_only_model:
        logger.warning(
            "slim_checkpoints=True keeps only the latest top-k dir resumable, but "
            "save_only_model=True never writes optimizer/FSDP state — set "
            "save_only_model=False for mid-run resume."
        )
    logger.info(
        "Checkpoint policy: save_only_model=%s slim_checkpoints=%s "
        "(slim all but latest) save_top_k=%d",
        save_only_model,
        slim_ckpts,
        save_top_k,
    )
    logger.info(
        "Start fusion training (objective=%s, residue_cond_mode=%s, "
        "bert_all_chains=%s, diffusion_all_chains=%s)...",
        training_args.train_objective,
        training_args.residue_cond_mode,
        training_args.bert_all_chains,
        training_args.diffusion_all_chains,
    )
    logger.info(
        "Chain-t ratios: joint=%.3f heavy2light=%.3f light2heavy=%.3f "
        "independent=%.3f single_chain=%.3f",
        float(training_args.joint_loss_ratio),
        float(training_args.heavy2light_loss_ratio),
        float(training_args.light2heavy_loss_ratio),
        float(training_args.independent_loss_ratio),
        float(training_args.single_chain_ratio),
    )
    logger.info(
        "Relation aux: mode=%s weight=%.4f temperature=%.4f "
        "(train-only; eval_loss stays reconstruction)",
        training_args.relation_aux,
        float(training_args.relation_aux_weight),
        float(training_args.relation_aux_temperature),
    )
    if resume_ckpt:
        logger.info("Resuming from checkpoint: %s", resume_ckpt)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_model(final_dir)
    if trainer.is_world_process_zero():
        tok.save_pretrained(final_dir)
        save_fusion_config(final_dir, model, tokenizer=tok)
    logger.info("Saved final checkpoint + tokenizer to %s", final_dir)


if __name__ == "__main__":
    train()
