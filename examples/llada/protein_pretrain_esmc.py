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
    asd_antibody_benchmark_keys,
    asd_antibody_row_to_record,
    asd_nanobody_benchmark_keys,
    asd_nanobody_row_to_record,
    build_mixed_immune_dataset,
    load_exclusion_keys,
    oas_benchmark_keys,
    oas_paired_row_to_record,
    ots_exclusion_key,
    ots_paired_row_to_record,
    tcr_native_exclusion_key,
    tcr_native_row_to_record,
    tcr_repertoire_exclusion_key,
    tcr_repertoire_row_to_record,
    trait_benchmark_key,
    trait_exclusion_key,
    trait_row_to_record,
    with_exclusion_filter,
    with_exclusion_filter_multi,
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
ASD_ANTIBODY_DEFAULT_DIR = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/asd/step6_final/antibody"
)
ASD_NANOBODY_DEFAULT_DIR = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/asd/step6_final/nanobody"
)
TRAIT_DEFAULT_DIR = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/trait/step4_final"
)
TCR_NATIVE_DEFAULT_DIR = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_native/dataset"
)
# Training corpora of published TCR design/generation papers (TCRT5, TCRDiff,
# GRATCR, TCR-epiDiff) plus the three TcrDesign-2026 epitope layers, normalized
# to the unified tier schema by scripts/data/tcr_native/{ingest,finalize}_papers.py.
# Separate from TCR_NATIVE_DEFAULT_DIR so the corpus the current checkpoints
# trained on stays byte-identical.
#
# Points at v2. The v1 corpus (``data/tcr_papers/dataset``, 407,112 kept rows)
# is still on disk and the two pre-v3 job configs pin it explicitly so their
# checkpoints stay reproducible. The default moved because everything that does
# *not* pass ``--tcr_papers_dir`` -- ad-hoc statistics, layout counts, leakage
# audits -- was silently measuring v1 while v3 trains on v2, which made every
# such number quietly wrong by 274k rows.
TCR_PAPERS_DEFAULT_DIR = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_papers_v2/dataset"
)
# Unlabeled single-chain CDR3b repertoire (TcrDesign-2026 pretrain/bCDR3), built
# by scripts/data/tcr_native/build_repertoire.py. This is the only source that
# renders as the ``tcr_single`` layout, which the T4 Setting-A unconditional
# benchmark decodes with and which had zero training coverage before.
TCR_REPERTOIRE_DEFAULT_DIR = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_repertoire/dataset"
)
# replaces_trait exclusion-key blocklist: TRAIT/PISTE CDR3 rows superseded by
# native full-length Fv. Applied statelessly to the TRAIT loader.
REPLACES_TRAIT_BLOCKLIST = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_native/dataset/replaces_trait_blocklist.txt"
)
# OTS benchmark decontamination: CDR3b cores that leak into the downstream
# TCR-binding benchmark (exact + 0.80 cluster). Filtered statelessly from the
# OTS loader. NOTE: this diverges the oas+ots base from the two already-running
# BERT ablation jobs (which trained on leaked OTS); those jobs are left as-is.
OTS_BENCHMARK_BLOCKLIST = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_native/dataset/ots_benchmark_blocklist.txt"
)
# Antibody + TRAIT benchmark decontamination blocklists (stateless exclusion
# keys) for the remaining fusion-mix sources. Antibody filters always apply.
_DS_DIR = "/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_native/dataset"
OAS_BENCHMARK_BLOCKLIST = f"{_DS_DIR}/oas_benchmark_blocklist.txt"
ASD_ANTIBODY_BENCHMARK_BLOCKLIST = f"{_DS_DIR}/asd_antibody_benchmark_blocklist.txt"
ASD_NANOBODY_BENCHMARK_BLOCKLIST = f"{_DS_DIR}/asd_nanobody_benchmark_blocklist.txt"
TRAIT_BENCHMARK_BLOCKLIST = f"{_DS_DIR}/trait_benchmark_blocklist.txt"
# T4 epitope-conditioned-generation decontamination: (CDR3b-core|epitope) pairs
# that are reference binders in the generation benchmark's answer key. The older
# blocklists only protect the *binding* benchmarks, so before this filter existed
# 29,431 of the 67,013 answer-key pairs (43.9%) sat inside the training mix.
# Applied to every epitope-conditioned source (tcr_native, TRAIT, tcr_papers).
T4_REFBINDER_BLOCKLIST = f"{_DS_DIR}/t4_refbinder_blocklist.txt"
# T2-clustering + T3-representation evaluation-set decontamination:
# (CDR3b-core|epitope) pairs from tcr_clustering/tcrs.csv and
# tcr_representation_paper6/target_binders.csv, widened to edit distance <=1.
# Neither set was ever in any decontamination bank, which is why the 2026-08-27
# audit measured 89.3% / 67.7% effective pair leakage against them. Built by
# scripts/data/tcr_native/build_t2t3_eval_blocklist.py --lev 1.
# Applied to every epitope-conditioned source (tcr_native, TRAIT, tcr_papers).
T2T3_EVAL_BLOCKLIST = f"{_DS_DIR}/t2t3_eval_blocklist.txt"


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
    dataset_args: str = "oas+ots"
    oas_dir: str = OAS_DEFAULT_DIR
    ots_dir: str = OTS_DEFAULT_DIR
    asd_antibody_dir: str = ASD_ANTIBODY_DEFAULT_DIR
    asd_nanobody_dir: str = ASD_NANOBODY_DEFAULT_DIR
    trait_dir: str = TRAIT_DEFAULT_DIR
    tcr_native_dir: str = TCR_NATIVE_DEFAULT_DIR
    tcr_papers_dir: str = TCR_PAPERS_DEFAULT_DIR
    tcr_repertoire_dir: str = TCR_REPERTOIRE_DEFAULT_DIR
    replaces_trait_blocklist: str = REPLACES_TRAIT_BLOCKLIST
    t4_refbinder_blocklist: str = T4_REFBINDER_BLOCKLIST
    t2t3_eval_blocklist: str = T2T3_EVAL_BLOCKLIST
    ots_benchmark_blocklist: str = OTS_BENCHMARK_BLOCKLIST
    oas_benchmark_blocklist: str = OAS_BENCHMARK_BLOCKLIST
    asd_antibody_benchmark_blocklist: str = ASD_ANTIBODY_BENCHMARK_BLOCKLIST
    asd_nanobody_benchmark_blocklist: str = ASD_NANOBODY_BENCHMARK_BLOCKLIST
    trait_benchmark_blocklist: str = TRAIT_BENCHMARK_BLOCKLIST
    train_split: str = "train"
    eval_split: str = "valid"
    max_rows_per_source: int | None = None
    # Separate (usually smaller) cap for the eval split so periodic validation
    # during a long run stays cheap; None means "use the full valid split".
    max_eval_rows_per_source: int | None = 2000
    # Makes the row caps above a uniform random sample rather than a prefix.
    # Required for correctness, not just variety: tcr_native/valid.csv is sorted
    # by its `source` column, so the old prefix cap silently excluded
    # minervina / tenx / covidvac from every eval_loss ever computed (see
    # downstream/benchmark/audit_2026_08_27/RETRAIN_PLAN.md §3a). Set to None to
    # restore the legacy prefix behaviour.
    subsample_seed: int | None = 0
    # Evaluate each source separately so it is visible which data regime
    # checkpoint selection is actually driven by. The overall `eval_loss` used for
    # top-k selection is then the row-weighted mean of the per-source losses.
    eval_per_source: bool = True
    # 1024 to match every train_jobs/*.yml. These previously defaulted to 512,
    # so any helper script that did not pass the flags explicitly silently
    # measured a *different* corpus: at 512 the antigen budget per ASD record is
    # only ~268 aa, which rejects the 607 aa antigen shared by >50% of the rows.
    # That is how RETRAIN_PLAN §7.8 came to report asd_antibody as 159,331 rows
    # instead of the 276,412 the jobs actually train on.
    max_length: int = 1024
    esmc_path: str = "model_weights/esmc/ESMC-300M"
    max_protein_length: int = 1024


def _record_chain_lengths_ok(
    record: dict, max_protein_length: int, max_total_length: int
) -> bool:
    """True if every chain fits ``max_protein_length`` and the estimated rendered
    length fits ``max_total_length``.

    The grammar renderer *raises* (not skips) on any chain longer than
    ``max_protein_length``, and the collator *raises* when the flat token stream
    exceeds ``max_sequence_length`` — both would crash a training step. ASD
    antigens run 200-1024 aa, so we drop over-long rows at CSV-read time instead.
    The total estimate over-counts grammar tokens (``3*n_chains + 8``) so it never
    lets through a row the collator would then reject.
    """

    chains = record.get("chains") or []
    if max_protein_length and any(len(seq) > max_protein_length for seq in chains):
        return False
    if max_total_length:
        approx = sum(len(seq) for seq in chains) + 3 * len(chains) + 8
        if approx > max_total_length:
            return False
    return True


def with_length_filter(row_to_record, max_protein_length: int, max_total_length: int):
    """Wrap ``row_to_record`` to drop records whose chains exceed the length caps."""

    if not max_protein_length and not max_total_length:
        return row_to_record

    def _wrapped(row: dict):
        record = row_to_record(row)
        if record is None:
            return None
        if not _record_chain_lengths_ok(record, max_protein_length, max_total_length):
            return None
        return record

    return _wrapped


def build_immune_specs(data_args: "DataArguments") -> list[ImmuneSourceSpec]:
    """Build source specs from the ``+``-joined ``dataset_args`` token list.

    Tokens: ``oas``, ``ots``, ``asd`` (== asd_antibody + asd_nanobody),
    ``asd_antibody``, ``asd_nanobody``, ``trait``. Unknown tokens raise so
    typos surface. ASD is antibody-antigen recognition (long antigen context);
    TRAIT is TCR-pMHC recognition (peptide + optional HLA pseudo-sequence).
    """

    tokens = [t.strip() for t in str(data_args.dataset_args).split("+") if t.strip()]
    if "asd" in tokens:
        tokens = [t for t in tokens if t != "asd"] + ["asd_antibody", "asd_nanobody"]

    # replaces_trait: when native full-length Fv is mixed in, drop the TRAIT/PISTE
    # CDR3 rows it supersedes (stateless (CDR3b-core|epitope) exclusion keys).
    trait_exclusions: set[str] = set()
    if "trait" in tokens and "tcr_native" in tokens:
        trait_exclusions = load_exclusion_keys(data_args.replaces_trait_blocklist)

    # Benchmark decontamination (always applied, independent of other tokens):
    # drop training rows whose sequences leak into a downstream TEST set.
    #   TCR binding benchmark (NM2025 seen/unseen + public): OTS, TRAIT (+ native
    #     already decontaminated at build time).
    #   Antibody benchmarks (CDR-H3 0.70 / heavy+light 0.95): OAS, ASD ab/nb.
    ots_exclusions: set[str] = load_exclusion_keys(data_args.ots_benchmark_blocklist)
    oas_exclusions: set[str] = load_exclusion_keys(data_args.oas_benchmark_blocklist)
    asd_ab_exclusions: set[str] = load_exclusion_keys(data_args.asd_antibody_benchmark_blocklist)
    asd_nb_exclusions: set[str] = load_exclusion_keys(data_args.asd_nanobody_benchmark_blocklist)
    trait_benchmark_exclusions: set[str] = load_exclusion_keys(data_args.trait_benchmark_blocklist)
    # T4 generation answer key -- (CDR3b-core|epitope). TRAIT ships the full
    # junction so it reuses trait_exclusion_key (which strips anchors); the
    # unified tcr_native/tcr_papers schema already stores the core.
    t4_exclusions: set[str] = load_exclusion_keys(data_args.t4_refbinder_blocklist)
    # T2 clustering + T3 representation eval sets. Same (CDR3b-core|epitope) key
    # space as t4, so it reuses the same two key functions.
    t2t3_exclusions: set[str] = load_exclusion_keys(data_args.t2t3_eval_blocklist)

    logger.info(
        "Blocklist key counts: replaces_trait=%d trait_benchmark=%d ots=%d oas=%d "
        "asd_ab=%d asd_nb=%d t4_refbinder=%d t2t3_eval=%d",
        len(trait_exclusions),
        len(trait_benchmark_exclusions),
        len(ots_exclusions),
        len(oas_exclusions),
        len(asd_ab_exclusions),
        len(asd_nb_exclusions),
        len(t4_exclusions),
        len(t2t3_exclusions),
    )

    # stack replaces_trait (superseded-by-native) + TCR binding decontam + T4
    # generation decontam + T2/T3 eval decontam (all built once)
    _trait_row_to_record = with_exclusion_filter(
        with_exclusion_filter(
            with_exclusion_filter(
                with_exclusion_filter(
                    trait_row_to_record, trait_exclusions, trait_exclusion_key
                ),
                trait_benchmark_exclusions,
                trait_benchmark_key,
            ),
            t4_exclusions,
            trait_exclusion_key,
        ),
        t2t3_exclusions,
        trait_exclusion_key,
    )
    _tcr_native_row_to_record = with_exclusion_filter(
        with_exclusion_filter(
            tcr_native_row_to_record, t4_exclusions, tcr_native_exclusion_key
        ),
        t2t3_exclusions,
        tcr_native_exclusion_key,
    )
    # Unlabeled repertoire has no epitope, so the (cdr3b_core|epitope) keys above
    # can never match. Project them down to bare cores and add the already
    # core-keyed OTS blocklist. Over-blocks by design: a benchmark CDR3b is
    # dropped whichever epitope it was an answer for.
    repertoire_core_exclusions: set[str] = (
        {k.split("|", 1)[0] for k in t4_exclusions}
        | {k.split("|", 1)[0] for k in t2t3_exclusions}
        | ots_exclusions
    )

    builders = {
        "oas": lambda: ImmuneSourceSpec(
            "oas",
            Path(data_args.oas_dir),
            with_exclusion_filter_multi(oas_paired_row_to_record, oas_exclusions, oas_benchmark_keys),
            "antibody",
        ),
        "ots": lambda: ImmuneSourceSpec(
            "ots",
            Path(data_args.ots_dir),
            with_exclusion_filter(ots_paired_row_to_record, ots_exclusions, ots_exclusion_key),
            "tcr",
        ),
        "asd_antibody": lambda: ImmuneSourceSpec(
            "asd_antibody",
            Path(data_args.asd_antibody_dir),
            with_exclusion_filter_multi(
                asd_antibody_row_to_record, asd_ab_exclusions, asd_antibody_benchmark_keys
            ),
            "antibody_antigen",
        ),
        "asd_nanobody": lambda: ImmuneSourceSpec(
            "asd_nanobody",
            Path(data_args.asd_nanobody_dir),
            with_exclusion_filter_multi(
                asd_nanobody_row_to_record, asd_nb_exclusions, asd_nanobody_benchmark_keys
            ),
            "nanobody_antigen",
        ),
        "trait": lambda: ImmuneSourceSpec(
            "trait",
            Path(data_args.trait_dir),
            _trait_row_to_record,
            "tcr_pmhc",
        ),
        "tcr_native": lambda: ImmuneSourceSpec(
            "tcr_native",
            Path(data_args.tcr_native_dir),
            _tcr_native_row_to_record,
            "tcr_pmhc",
        ),
        # Published TCR design/generation training corpora (TCRT5 / TCRDiff /
        # GRATCR / TCR-epiDiff), already deduped against the live corpus and
        # decontaminated against both the binding and the T4 generation
        # benchmarks by finalize_papers.py. Same unified schema as tcr_native,
        # so it reuses the same row_to_record.
        "tcr_papers": lambda: ImmuneSourceSpec(
            "tcr_papers",
            Path(data_args.tcr_papers_dir),
            _tcr_native_row_to_record,
            "tcr_pmhc",
        ),
        # Unlabeled single-chain CDR3b repertoire -> the only source that renders
        # as ``tcr_single``. Already decontaminated at build time against the OTS
        # holdout/valid, T4, T2/T3 and binding-benchmark core sets; the load-time
        # filter here is defence in depth for when those blocklists grow after
        # the corpus was built.
        "tcr_repertoire": lambda: ImmuneSourceSpec(
            "tcr_repertoire",
            Path(data_args.tcr_repertoire_dir),
            with_exclusion_filter(
                tcr_repertoire_row_to_record,
                repertoire_core_exclusions,
                tcr_repertoire_exclusion_key,
            ),
            "tcr",
        ),
    }
    specs: list[ImmuneSourceSpec] = []
    seen: set[str] = set()
    for token in tokens:
        if token not in builders:
            raise ValueError(f"Unknown dataset token {token!r}; known: {sorted(builders)}")
        if token in seen:
            continue
        seen.add(token)
        specs.append(builders[token]())
    if not specs:
        raise ValueError(f"No dataset sources parsed from {data_args.dataset_args!r}")

    # Drop over-long rows at load time so the renderer/collator never crash on a
    # chain (e.g. long ASD antigens) that exceeds the caps. Stacks on top of the
    # per-source benchmark/decontam filters already applied above.
    max_protein_length = int(getattr(data_args, "max_protein_length", 0) or 0)
    max_total_length = int(getattr(data_args, "max_length", 0) or 0)
    for spec in specs:
        spec.row_to_record = with_length_filter(
            spec.row_to_record, max_protein_length, max_total_length
        )
    return specs


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
    # BERT objective only: when True, compute MLM loss over every residue in all
    # chains (including fixed context: MHC/peptide/antigen). When False, restrict
    # to generated chains (diffusion_eligible_mask). No effect on the diffusion
    # objective, which is always generated-only.
    bert_all_chains: bool = True
    # Diffusion objective only: the mirror image of bert_all_chains. When True the
    # eligible set becomes every residue of every chain, so fixed grammar context
    # (antigen / MHC / peptide) is corrupted and scored like a generated chain --
    # i.e. no chain is held fixed. False (the default) keeps the historical
    # generated-only behaviour so existing diffusion checkpoints stay continuable.
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


class ImmuneBioSeqDataset(Dataset):
    """CSV immune rows -> BioSeqRecord with role mapping for GrammarBioSeqCollator."""

    def __init__(self, base_ds) -> None:
        self.base_ds = base_ds

    def __len__(self) -> int:
        return len(self.base_ds)

    def _roles_for(self, rec: dict) -> list[BioSeqChain]:
        # Records may carry explicit per-chain roles (ASD recognition plane:
        # antigen + antibody/nanobody). Honor them verbatim.
        roles = rec.get("roles")
        if roles is not None:
            return [BioSeqChain(seq, role) for seq, role in zip(rec["chains"], roles)]
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
            labels = {"relation": rec["relation"]} if rec.get("relation") else {}
            return BioSeqRecord(
                chains=self._roles_for(rec),
                task_type=rec["task_type"],
                source=rec.get("source", ""),
                labels=labels,
            )
        except Exception as exc:
            if _retries >= max(len(self) - 1, 0):
                raise RuntimeError(
                    f"Failed to encode sample {i} after exhausting retries"
                ) from exc
            logger.warning("Skipping bad sample %d (%s); trying next", i, exc)
            return self.__getitem__((i + 1) % len(self), _retries=_retries + 1)


class FusionTrainer(transformers.Trainer):
    # Row counts per eval source, used to recombine per-source losses. Set by the
    # caller when eval_dataset is a dict.
    eval_source_rows: dict[str, int] | None = None

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"
    ):  # noqa: ANN001
        """Add a combined ``eval_loss`` when evaluating a dict of sources.

        With a dict ``eval_dataset``, ``Trainer.evaluate`` recurses once per key
        and emits only ``eval_<source>_loss`` -- no plain ``eval_loss``, which is
        what ``TopKValLossCheckpointCallback`` and ``metric_for_best_model`` rank
        by. Rather than paying for a second pass over a blended copy, reconstruct
        the global figure as the row-weighted mean of the per-source losses (equal
        to the blended loss up to batch-padding effects) and ``log`` it so it
        reaches both ``state.log_history`` and wandb.
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
            or not self.eval_source_rows
        ):
            return metrics

        weighted = 0.0
        covered = 0
        for name, rows in self.eval_source_rows.items():
            value = metrics.get(f"{metric_key_prefix}_{name}_loss")
            if value is None:
                continue
            weighted += float(value) * rows
            covered += rows
        # Publish only when every source reported: a mean over a shifting subset
        # would rank checkpoints against a moving target.
        if covered and covered == sum(self.eval_source_rows.values()):
            metrics[loss_key] = weighted / covered
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

    base_train, train_counts, _ = build_mixed_immune_dataset(
        specs,
        split=data_args.train_split,
        max_rows_per_source=max_rows,
        sample_seed=data_args.subsample_seed if max_rows is not None else None,
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
    )
    # Decoder is bf16 (from_pretrained) while ESMC/condition heads default to fp32;
    # unify to bf16 so the FSDP root flat-param group has a single dtype.
    model = model.to(torch.bfloat16)

    resume_ckpt = getattr(training_args, "resume_from_checkpoint", None)
    if resume_ckpt in (None, "", "False", "false", "0"):
        resume_ckpt = None
    elif resume_ckpt in (True, "True", "true", "1"):
        resume_ckpt = True
    else:
        resume_ckpt = str(resume_ckpt)

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

    eval_rows = data_args.max_eval_rows_per_source
    if training_args.dry_run and eval_rows is None:
        eval_rows = 8
    base_eval, eval_counts, eval_parts = build_mixed_immune_dataset(
        specs,
        split=data_args.eval_split,
        max_rows_per_source=eval_rows,
        sample_seed=data_args.subsample_seed if eval_rows is not None else None,
    )
    logger.info("Eval source counts: %s", eval_counts)

    # Per-source eval datasets make it visible which data regime drives checkpoint
    # selection. Previously a single blended eval_loss hid that it was measured
    # mostly on full-length paired data (69.9% of the eval rows) while that regime
    # was only 33% of training -- and will be ~8% once tcr_papers is mixed in.
    # HF logs one `eval_<source>_loss` per dict key; PerSourceEvalLossCallback then
    # recombines them into the plain `eval_loss` the top-k selector ranks by, so
    # nothing is evaluated twice.
    eval_source_rows: dict[str, int] | None = None
    if data_args.eval_per_source and len(eval_parts) > 1:
        eval_ds = {part.name: ImmuneBioSeqDataset(part) for part in eval_parts}
        eval_source_rows = {part.name: len(part) for part in eval_parts}
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
    trainer.eval_source_rows = eval_source_rows
    # Aux scalar must land in ``logs`` before WandbCallback.on_log reads it.
    trainer.callback_handler.callbacks.insert(0, RelationAuxLogCallback())
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
    tok.save_pretrained(final_dir)
    logger.info("Saved final checkpoint + tokenizer to %s", final_dir)


if __name__ == "__main__":
    train()
