# Project Process

## Active Volc Training Tasks

> Only non-terminal jobs (`Initialized` / `Queue` / `Staging` / `Running` / `Killing`). Remove a row when the job reaches `Success`, `Failed`, or `Killed`. Update after every `volc ml_task submit` or `cancel`. Rule: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/.cursor/rules/volc-train-task-log.mdc`.

Last updated: 2026-06-25 (UTC+8, resubmitted grammar-v2 test jobs after chain-separator fix)

| Status | Task ID | Job Name | YAML |
|--------|---------|----------|------|
| Queue | t-20260625140029-66gnq | qwen3_vl_bioseq_grammar_v2_no_encoder_qwen0_6b | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v2_no_encoder_qwen0_6b.yml` |
| Queue | t-20260625140032-t2vzk | qwen3_vl_bioseq_grammar_v2_esmc300m | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m.yml` |
| Queue | t-20260625140036-sz7wd | qwen3_vl_bioseq_grammar_v2_esmc600m | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v2_esmc600m.yml` |
| Queue | t-20260625140039-dxt5t | qwen3_vl_bioseq_grammar_v2_esm2_650m | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v2_esm2_650m.yml` |

## 2026-06-07

- Confirmed `/vepfs-mlp2/c20250601/251105016/project/dllm_test` is the active project root.
- Confirmed `/c20250601/mj/model_weights` is the required model weight root.
- Chose `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq` instead of a top-level `airgen_bioseq` package because `/vepfs-mlp2/c20250601/251105016/project/dllm_test/pyproject.toml` currently packages only `dllm`.
- Started the BioSeq pipeline implementation with independent data, model, diffusion training, weight download, and weight verification modules.
- Added the rule that all later task, plan, and process changes must be written to Markdown files with absolute paths.
- Changed `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/__init__.py` and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/__init__.py` to lazy imports so importing BioSeq does not force-load old `dllm.core` dependencies.

## Current Status

- BioSeq scaffold + docs: in place. Active training path is grammar-v2 under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch`.
- Encoder loading validated: ESMC via Biohub `esm==3.2.3` (`load_local_esmc_encoder` / `BioSeqEncoderDiffusionModel.from_esmc`); ESM2 via HF `EsmModel`.

## Model Weights

Full path inventory is the single source of truth in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJ_GUIDE.md`. Shared root `/c20250601/mj/model_weights`; Volc-mounted copies under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights` for cluster jobs.

- ESMC (native loader, vocab 64, id 31 = `|`): `ESMC-300M` d_model 960 (L30/H15), `ESMC-600M` d_model 1152 (L36/H18), `ESMC-6B`.
- ESM2 (HF `EsmModel`, vocab 33, id 31 = `<null_1>`): `esm2_t6_8M` hidden 320, `esm2_t12_35M` 480, `esm2_t30_150M` 640, `esm2_t33_650M` 1280 (Ophiuchus-Ab base), `esm2_t36_3B` 2560. 15B not downloaded.
- Ophiuchus-Ab: `ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt`, `md5:9baa0d3fbe908930d9a7d4f8d8b6144c`, from `https://zenodo.org/records/18478480`.

## 2026-06-11

- Investigated `https://arxiv.org/pdf/2604.24506`, titled `MIMIC: A Generative Multimodal Foundation Model for Biomolecules`.
- Key architecture reference: MIMIC uses a split-track encoder-decoder architecture, sums aligned modalities within nucleic-acid/protein coordinate tracks, keeps semantic/context tokens as separate groups, uses register tokens, local group-reset RoPE, and cross-attention decoding.
- Key data reference: LORE aligns approximately 13 million RNA transcripts and 15.5 million proteins across sequence, structure, conservation, regulatory, surface, abundance, functional text, taxonomy, and experimental context modalities.
- Key training reference: MIMIC uses pathway-based sampling for partially observed modality combinations, asymmetric encoder/decoder token budgets, target packing, staged context curriculum, length-bucketed dynamic batching, and register-token reconstruction under random token dropout.
- BioSeq implication: keep the current `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq` sequence-diffusion path, but plan the multimodal extension as split-track residue-aligned conditioning rather than a simple concatenation of ESMC, structure, and task metadata.
- Release check: `https://github.com/PolymathicAI/MIMIC` currently contains only `LICENSE`, `README.md`, and `assets`; GitHub releases API returned an empty list. Public Hugging Face API checks for `polymathic-ai/mimic` and `polymathic-ai/lore` did not expose public model/data artifacts. The official text says code, weights, and LORE assets are still being prepared for public release.
- Investigated recent task/benchmark directions useful for BioSeq. Recommended immediate task roadmap: AbBiBench-style antibody-antigen affinity/design, FLAb-style antibody developability, IMMREP-style TCR-pMHC specificity, DecoderTCR-style TCR-pMHC sequence modeling, TCR-pMHC structure-oracle evaluation, PPLM-PPI-style paired PPI modeling, ProteinGym-style mutation fitness scoring, and PFMBench/ProteinBench-style broad external evaluation.
- Release/data availability notes: AbBiBench has public code at `https://github.com/MSBMI-SAFE/AbBiBench` and a public Hugging Face dataset at `https://huggingface.co/datasets/AbBibench/Antibody_Binding_Benchmark_Dataset`; FLAb has public code/data entry points at `https://github.com/Graylab/FLAb` and `https://registry.opendata.aws/flab/`; DecoderTCR has public model resources listed at `https://virtualcellmodels.cziscience.com/model/decoder-tcr`; IMMREP25 is a Kaggle challenge announced by IEDB; PFMBench, ProteinBench, and ProteinGym have public code or benchmark resources.
- Implementation decision: before downloading new datasets, add task adapters and a normalized JSONL schema under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq` so antibody, TCR-pMHC, PPI, mutation-fitness, and structure-oracle tasks share one BioSeq data contract.

## 2026-06-13

- Audited `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data`. Total size is approximately 92G. No active data pipeline process was found.
- Existing processed paired antibody summary in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/data_processing_summary.md`: OAS paired raw 3,087,576, final 2,251,632 with train 2,229,115, valid 11,258, holdout 11,259.
- Current OAS paired antibody split uses `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/oas_previous_clean/splits/cleaned_merged_data_step_clustered_{train,valid,holdout}_oas_label.csv`; it has 2,486,442 train rows, 12,553 valid rows, and 12,653 holdout rows excluding CSV headers.
- Existing OTS paired TCR final split under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean/final` has 2,102,715 train rows, 10,619 valid rows, and 10,621 holdout rows excluding CSV headers.
- Existing nanobody final split under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed/step6_final` has 11,649,792 train rows, 58,862 valid rows, and 58,981 holdout rows excluding CSV headers. Pipeline log reports final total 11,767,635 sequences.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/processed` is current-model safer than `processed_v2`: it has 805,095 train JSONL rows and 14,405 val rows, with max chain length 512. Sources are PPI, VDJdb, MIRA, and McPAS.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/processed_v2` has 803,591 train rows and 14,377 val rows, but preserves chains up to length 32,000; 396,482 training chains exceed length 512 and 106,416 exceed length 1024, so it needs filtering/cropping/bucketing before BioSeq training.
- Current unified JSONL outputs do not yet include the large OAS paired antibody or nanobody final splits. Those datasets remain in CSV form and need BioSeq task adapters.
- Downstream SAbDab CDR infilling data exists under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/cdr_infilling/sabdab`; each CDR has 10 folds with roughly 3.3k total examples. SAb23H2 converted CDR files have 60 examples per CDR.
- Several downstream files under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/flab`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/in_silico`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/comp_chain` are symlinks to `/vepfs-mlp2/c20250506/251105017/mj/AirGen-Dev/...`; those targets are currently missing from this environment, so the symlinks are broken.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data` contains 1,010 AppleDouble `._*` metadata files. They should be ignored by loaders and cleanup scripts.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/adapters.py` with canonical `bioseq.v1` JSONL schema and adapters for OAS paired antibody CSV, OTS paired TCR CSV, nanobody CSV, and existing processed JSONL.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/tools/convert_data.py`, a CLI that converts local immune sequence sources to BioSeq JSONL and requires absolute paths.
- Verified unit tests with `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq -q`: 6 passed.
- Verified adapter smoke conversions with real local files and `--limit 2`, writing temporary outputs under `/tmp/bioseq_adapter_smoke`: OAS, OTS, nanobody, and existing processed JSONL each produced valid `bioseq.v1` rows.
- No full OAS/OTS/nanobody conversion has been run yet; the next decision is where to store sharded canonical JSONL outputs and what mixture weights to use for antibody, nanobody, TCR, PPI, and TCR-pMHC training.

## 2026-06-13 Ophiuchus-Ab migration from AirGen-Dev

- User request: migrate only Ophiuchus-Ab training and model code from `/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev` into `/vepfs-mlp2/c20250601/251105016/project/dllm_test`, keep sampling logic aligned with AirGen, and do not migrate unrelated AirGen tasks.
- Copied AirGen mint ESM2 implementation to `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/mint` so checkpoint keys (`embed_tokens`, `layers.*.self_attn.*`, `layers.*.multimer_attn.*`, `emb_layer_norm_after`, `lm_head.dense/layer_norm`) map directly without an adapter.
- Added Ophiuchus-Ab exact stack under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus`:
  - `model.py`: `OphiuchusAbBackbone` (33-layer ESM2 + multimer attention + token dropout, no position embedding)
  - `multichain.py`: `MultiChainOphiuchusAbModel` with AirGen-compatible `construct_x_t`, `compute_loss`, `forward_decoder`, `_decoding`, and `generate`
  - `sampling.py`: `gumbel_argmax`, `topk_masking`, and related helpers copied from AirGen `model_utils.py`
  - `loss.py`: `RDMCrossEntropyLoss` with reciprocal weighting and focal loss
  - `collator.py`: `OphiuchusAbTrainingCollator` (`heavy_tokens` / `light_tokens` batch layout) and `OphiuchusAbInferenceCollator`
  - `training.py`: `compute_ophiuchus_ab_training_loss`
  - `multichain.load_ophiuchus_checkpoint`: loads `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt` with zero missing/unexpected keys
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_ab.py`:
  - `--preset ophiuchus-ab` initially defaulted to `--backend ophiuchus` (exact AirGen training path)
  - The lightweight `bioseq` backend was later removed from this antibody entry point because it is not the official Ophiuchus-Ab architecture.
  - `--init-multimer` initializes multimer attention from self-attention before training
  - `--checkpoint-path` can resume from the official Ophiuchus-Ab checkpoint
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/sample_ab.py` for AirGen-aligned iterative decoding (`max_iter=500`, `gumbel_argmax`, optional `cfg_scale`, optional fixed heavy chain via `--fix-heavy`).
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_ophiuchus_migration.py`.
- Verified `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq -q`: 10 passed, including exact checkpoint load and a 4-step generation check.
- Checkpoint logic now enforced in code: Ophiuchus-Ab is not treated as a generic BioSeq decoder checkpoint; only the mint ESM2 backbone inside `MultiChainOphiuchusAbModel` is loadable from `Ophiuchus-Ab.ckpt`.
- Remaining Ophiuchus-Ab scope (not started): POAS CSV training dataloader wiring, DDP/multi-GPU trainer, and downstream heavy→light / CDR infill scripts.

## 2026-06-13 downstream migration

- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream` as the BioSeq downstream task root (separate from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream` data assets).
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/common.py` with `load_model`, AirGen-aligned collators, and shared `run_generate`.
- Migrated Ophiuchus-Ab downstream scripts aligned to `MultiChainOphiuchusAbModel`:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/comp_chain/generate_light_from_csv.py`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/infill/zeroshot_cdr.py`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/infill/zeroshot_sab23h2.py`
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/README.md` documenting layout and usage.
- Extended `Esm2ProteinTokenizer.encode()` in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/data.py` for AirGen-style masked sequence encoding.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_downstream_common.py`.
- Expanded `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream` with the remaining AirGen downstream tree:
  - Aligned scripts: `embeddings.py`, `humanization/humanize.py`, `flab/finetune_flab.py`, `dev/finetune_dev.py`, `specificity/hd_flu_cov_paired.py`
  - Copied legacy/reference scripts and eval utilities: `comp_chain/eval_scripts/`, `humanization/eval_scripts/`, `flab/finetune_flab_align.py`, `flab/finetune_flab_esm_ppi.py`, `dev/finetune_dev_pplm.py`, `in_silico/finetune_in_silico.py`, and related helper files
  - Copied `infill/IgGM_Test_set/` benchmark assets
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/README.md` to distinguish aligned Ophiuchus entry points vs legacy AirGen copies.
- Remaining Ophiuchus-Ab scope: POAS CSV training dataloader wiring and DDP/multi-GPU trainer.

## 2026-06-13 remove antibody lightweight backend

- Removed the lightweight `bioseq` backend from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_ab.py`.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_ab.py` now always builds `OphiuchusAbBackbone` plus `MultiChainOphiuchusAbModel` from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus`.
- Removed `--preset`, `--backend`, `--trainer`, lightweight model-size flags, the synthetic default dataset, and the generic `BioSeqDiffusionTrainer` path from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_ab.py`.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_ab.py` now requires `--train-csv` and supports `vh_protein_sequence`/`vl_protein_sequence`, `heavy`/`light`, or local `cleaned_chain1_seq`/`cleaned_chain2_seq` columns.
- Deleted the stale temporary checkpoint directory `/vepfs-mlp2/c20250601/251105016/project/dllm_test/.models/bioseq/antibody-smoke-test`.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` so antibody training is documented as exact Ophiuchus-Ab only.
- Verified `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_ab.py` and `python /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_ab.py --help`; the help output no longer exposes `--preset`, `--backend`, `--trainer`, or lightweight model-size flags.

## 2026-06-13 antibody inference audit

- Confirmed `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/sample_ab.py` already uses `MultiChainOphiuchusAbModel.from_checkpoint` and `OphiuchusAbInferenceCollator`; it does not expose a lightweight `bioseq` backend.
- Confirmed aligned downstream inference uses `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/common.py`, which loads `MultiChainOphiuchusAbModel` plus `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt`.
- Fixed `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/common.py` so heavy-to-light generation masks the light-chain suffix after any provided prompt and also masks an empty light-chain input. This prevents an empty light chain from being treated as fixed context during `model.generate`.
- Added coverage in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_downstream_common.py` for prompt-preserving light-chain masking and empty-light-chain generation masking.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` with the exact Ophiuchus-Ab inference surface and the rule that antibody inference must not use a lightweight generic BioSeq backend.
- Verified `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_downstream_common.py -q`: 4 passed.
- Verified `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/sample_ab.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/common.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/comp_chain/generate_light_from_csv.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/infill/zeroshot_cdr.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/infill/zeroshot_sab23h2.py`.
- Verified `python /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/sample_ab.py --help` and `python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/comp_chain/generate_light_from_csv.py --help`; both expose Ophiuchus-Ab inference parameters only.

## 2026-06-13 ESM2 base-init scope decision

- User clarified that a Hugging Face ESM2 base-weight initialization path is not needed for the current antibody path because ESM2 weights are already present under `/c20250601/mj/model_weights/esm2`.
- Confirmed local ESM2 snapshots include `config.json`, tokenizer files, and PyTorch-compatible weight files under `/c20250601/mj/model_weights/esm2/esm2_t6_8M_UR50D`, `/c20250601/mj/model_weights/esm2/esm2_t12_35M_UR50D`, `/c20250601/mj/model_weights/esm2/esm2_t30_150M_UR50D`, `/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D`, and `/c20250601/mj/model_weights/esm2/esm2_t36_3B_UR50D`.
- Removed generic Hugging Face ESM2 pretrain initialization from the current remaining Ophiuchus-Ab scope. If generic ESM2 initialization is needed later, it should use the local `/c20250601/mj/model_weights/esm2` snapshots instead of adding a network download dependency.

## 2026-06-13 Ophiuchus-Ab MINT architecture audit

- User raised concern that the Ophiuchus-Ab migration might be missing MINT cross-attention behavior from `/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev`.
- Checked `/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/configs/experiment/ab/mint_650m_stage1.yaml`: Ophiuchus-Ab uses `model._target_: dplm_multichain` and `model.net.arch_type: mint`.
- Checked `/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/src/byprot/models/lm/model_utils.py`: `arch_type: mint` instantiates `MintForDPLM`, not `EsmForDPLM` or the DPLM adapter cross-attention path.
- Checked `/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/src/byprot/models/lm/mint_dplm.py`: `MintForDPLM.forward` calls the MINT ESM2 model with `chain_ids`; `forward_encoder` returns `{}`. There is no encoder-decoder cross-attention call in this Ophiuchus-Ab path.
- Checked `/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/src/byprot/models/lm/modules/mint/modules.py`: the cross-chain mechanism is `multimer_attn`, which mixes attention logits based on whether token pairs come from different `chain_ids`.
- Verified `diff -qr /vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/src/byprot/models/lm/modules/mint /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/mint`: only generated `__pycache__` directories differ.
- Inspected `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt`: 768 state-dict keys, `cross` keys 0, `adapter` keys 0, `encoder_hidden` keys 0, `multimer_attn` keys 198, `self_attn` keys 363. This confirms the released Ophiuchus-Ab checkpoint is a MINT multimer-attention checkpoint, not a cross-attention adapter checkpoint.
- Current conclusion: the migration is not missing a HuggingFace-style cross-attention module for Ophiuchus-Ab. The real remaining migration risk is that `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/multichain.py` is a dependency-light reimplementation of AirGen `dplm_multichain.py`, not a verbatim copy; add parity tests for noising, loss masks, and generation mask transitions before trusting long training runs.

## 2026-06-13 Ophiuchus-Ab checkpoint import test

- Tested current migrated model import with `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt`.
- Script instantiated `MultiChainOphiuchusAbModel` from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus` and called `load_ophiuchus_checkpoint(..., strict=True)`.
- Result: checkpoint exists, size `3253751875` bytes, checkpoint state keys `768`, migrated backbone state keys `768`, strict-load missing keys `0`, strict-load unexpected keys `0`, shape mismatches `0`.
- Checkpoint key summary from the same run: `cross` keys `0`, `adapter` keys `0`, `multimer_attn` keys `198`.
- Tensor equality spot checks after load all had max absolute difference `0.0`: `embed_tokens.weight`, `layers.0.self_attn.q_proj.weight`, `layers.0.multimer_attn.q_proj.weight`, and `lm_head.dense.weight`.
- Verified `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_ophiuchus_migration.py -q`: 4 passed in 55.69s.

## 2026-06-13 multi-dataset DDP training + variable length

- User request: make the Ophiuchus-Ab path a trainable multi-node/multi-GPU version, load all three immune datasets (OAS paired antibody, OTS paired TCR, nanobody VHH) together, run a training smoke check, and stop using fixed chain lengths.
- Confirmed the exact Ophiuchus-Ab model already supports variable length: `MultiChainOphiuchusAbModel.compute_loss` splits logits by `targets.size(1)` and the mint ESM2 `forward` builds a real `padding_mask = tokens.eq(padding_idx)` that feeds attention, token dropout, and the diffusion loss. Only `OphiuchusAbTrainingCollator` hard-coded the `(150, 128)` lengths.
- Verified token-id parity between the mint `Alphabet` ("ESM-1b") and `Esm2ProteinTokenizer`: `<cls>`=0, `<pad>`=1, `<eos>`=2, `<unk>`=3, `<mask>`=32. So padding with the real `<pad>` id is correctly masked by the backbone.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/collator.py::MultiChainDynamicCollator`: variable-length two-slot collator that pads each batch to the longest chain-1 / chain-2 sequence with `<pad>` (not `<eos>`), caps per-chain length via `max_length` (default 512), supports single-chain examples (nanobody) with a minimal `[<cls>, <eos>]` chain-2 placeholder, and emits per-example `weights`. It reuses the existing `compute_loss` two-slot contract unchanged.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/datasets.py`: streams the three local CSV corpora into `{chains, task_type, source, weight}` records (OAS heavy oriented to slot 0 via anarci type; OTS beta oriented to slot 0; nanobody single chain), skips invalid/empty sequences, caps rows with `max_rows`, and exposes `default_immune_specs` + `build_mixed_immune_dataset` returning a `ConcatDataset` plus per-source counts.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_bioseq_ddp.py`: torchrun-compatible trainer. It wraps only the parameter-holding backbone (`model.net`) in `DistributedDataParallel` so `compute_loss` drives a single synced forward; auto-selects `nccl`+`cuda:LOCAL_RANK` when CUDA is available and `gloo`+CPU otherwise; uses `DistributedSampler`, AdamW, optional warmup, grad accumulation, grad clipping, optional bf16 autocast, rank-0 logging and checkpoint saving.
- Exported `MultiChainDynamicCollator`, `ImmuneCsvDataset`, `ImmuneSourceSpec`, `build_mixed_immune_dataset`, and `default_immune_specs` from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/__init__.py`.
- Environment note: this node has torch 2.11.0+cu130 but the NVIDIA driver (12020) is too old, so `torch.cuda.is_available()` is False; smoke checks ran on CPU. The DDP code is GPU-ready and selects nccl automatically on a working CUDA cluster.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_dynamic_training.py` (5 tests: variable-length paired collation, single-chain nanobody placeholder, max-length cap, synthetic-CSV source extraction with heavy/beta orientation, invalid-row skipping).
- Verified `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq -q`: 19 passed (includes the existing migration tests).
- Smoke check 1 (single process, CPU): `python examples/bioseq/train_bioseq_ddp.py --limit-per-source 64 --batch-size 2 --max-steps 3 --max-length 128 --num-workers 0 --init-multimer` loaded oas=64/ots=64/nanobody=64, ran 3 forward/backward/optimizer steps with non-trivial heavy/light losses, and saved `.models/bioseq/ophiuchus-ab-mixed/final.pt`.
- Smoke check 2 (DDP, 2 processes, gloo/CPU): `torchrun --standalone --nproc_per_node=2 examples/bioseq/train_bioseq_ddp.py --limit-per-source 64 --batch-size 2 --max-steps 3 --max-length 128 --num-workers 0 --init-multimer` ran with `world_size=2 distributed=True`, synced gradients, and saved the final checkpoint from rank 0.
- Known simplification: single-chain nanobody examples use a `[<cls>, <eos>]` chain-2 placeholder (chain id 1) so the two-slot loss stays unchanged; this contributes a negligible "empty second chain" eos signal. A future cleaner option is a per-slot loss weight that zeros the placeholder slot for single-chain samples.

### Volc ML Platform 2-node 16-GPU test submission

- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/bioseq_mixed_16gpu_test.yml` (modeled on the Protenix-v2 16-GPU templates).
- Resource: `TaskRoleSpecs` worker `RoleReplicas: 2`, `Flavor: ml.pni2.28xlarge` => 2 nodes x 8 GPU = 16 GPU; queue `c20250601`; image `cr-mlp-cn-beijing.cr.volces.com/public/airgen:v1`; `Preemptible: true`, `Priority: 6`, `ActiveDeadlineSeconds: 7200`.
- Entrypoint activates `/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr` (torch 2.8.0, verified to import the bioseq package), runs a preflight import + `scripts/tests/bioseq/test_dynamic_training.py`, then `torchrun --nnodes=${MLP_WORKER_NUM} --nproc_per_node=${MLP_WORKER_GPU} --node_rank=${MLP_ROLE_INDEX} --master_addr=${MLP_WORKER_0_HOST} --master_port=${MLP_WORKER_0_PORT} examples/bioseq/train_bioseq_ddp.py` with mixed OAS+OTS+nanobody, variable length, `--limit-per-source 50000 --max-length 320 --batch-size 8 --max-steps 150 --bf16 --init-multimer`.
- The test job intentionally trains from `--init-multimer` (no `--checkpoint-path`) because the Ophiuchus checkpoint lives under `/c20250601/mj/model_weights`, which is outside the declared Vepfs mount and may not be available on cluster nodes. The three datasets live under the mounted `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data`.
- Submitted via `/root/.codex/skills/volc-no-proxy/scripts/volc-no-proxy.sh ml_task submit --conf .../train_jobs/bioseq_mixed_16gpu_test.yml`: `task_id=t-20260614010923-n5xmn`, initial state `Staging` (2026-06-13 17:09).

### wandb integration + resubmission

- User request: training should use wandb, and monitor the job (poll every 2 minutes) once it reaches Running.
- Added wandb support to `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_bioseq_ddp.py`: rank-0-only `setup_wandb()` with `--wandb-mode {online,offline,disabled}`, `--wandb-project`, `--wandb-entity`, `--wandb-run-name`, `--wandb-dir`. It tries the requested mode, falls back online -> offline -> disabled (so a cluster node that cannot reach api.wandb.ai never blocks or crashes the 16-GPU job), and logs `train/loss`, `train/heavy_loss`, `train/light_loss`, `train/lr`, `perf/samples_per_sec`, `train/epoch` plus run config.
- Staged the wandb API key from this node's `~/.netrc` to `/vepfs-mlp2/c20250601/251105016/.secrets/wandb_api_key` (chmod 600, outside the git repo). The job entrypoint exports `WANDB_API_KEY` from that file and sets `WANDB_DIR` under the run output directory.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/bioseq_mixed_16gpu_test.yml` to export the wandb key/dir and pass `--wandb-mode online --wandb-project bioseq-ophiuchus --wandb-run-name bioseq_mixed_16gpu_test`.
- Environment finding: under `/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr` (torch 2.8.0) `torch.cuda.is_available()` is True on this node, so that env is GPU-capable (the base miniforge `cu130` torch reported False only because of the old driver). A CPU-free GPU smoke (`device=cuda:0`) trained 2 steps and logged a wandb offline run successfully.
- Cancelled the no-wandb task `t-20260614010923-n5xmn` (`ml_task cancel -i ...` -> "cancel success") to avoid two concurrent 16-GPU jobs.
- Resubmitted the wandb-enabled job: `task_id=t-20260614011617-wv8bb` (2026-06-13 17:16). volc `ml_task` verbs in this CLI: `submit/cancel/get/logs/list/instance` (no `kill`); `ml_task logs` requires BOTH `-t <task_id>` and `-i <instance>`; instance ids are `worker_0`/`worker_1` (from `ml_task instance list -i <task_id>`).

### 16-GPU test run result (SUCCESS)

- Task `t-20260614011617-wv8bb` reached `Running` with both instances `worker_0` and `worker_1` Running, i.e. 2 nodes x 8 GPU = 16 GPU.
- Preflight (torch/cuda print, bioseq import, `test_dynamic_training.py`) passed, then `torchrun` trained the full 150 steps over mixed OAS+OTS+nanobody at variable length.
- Loss decreased steadily: step0 `loss=146.23` (heavy 72.87 / light 73.36) -> step140 `loss=12.25` (heavy 6.18 / light 6.07). Throughput ~85-90 samples/s at global batch 128, bf16.
- wandb online sync worked from the cluster: run `https://wandb.ai/codema/bioseq-ophiuchus/runs/flpo3unq` (project `https://wandb.ai/codema/bioseq-ophiuchus`). So the cluster nodes can reach api.wandb.ai; the offline fallback was not needed.
- Final checkpoint saved: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/bioseq_mixed_16gpu_test/final.pt` (~3.25 GB backbone state dict).
- Conclusion: the multi-node/multi-GPU variable-length mixed-data training path is verified on 16 GPU with working wandb logging. This run was a short 150-step test; a real run should raise `--max-steps`, lift `--limit-per-source` (or switch to a streaming dataset for full corpora), and optionally tune `--batch-size`/`--max-length`.

### Why the task ended + robust checkpoint/resume logic

- "Why did it stop" investigation: `ml_task get -i t-20260614011617-wv8bb` shows `State: Success`, `ExitCode: 0`, empty `DiagInfo`. The job was NOT preempted or errored; it simply finished the configured `--max-steps 150` and exited cleanly. The short duration is the 150-step test setting, not a failure.
- Audited the original checkpoint logic in `examples/bioseq/train_bioseq_ddp.py`: it only saved `{"backbone_state_dict": ...}` and the 16-GPU test used `--save-interval 0`, so there was no periodic checkpoint, no optimizer/step state, and no resume path — unsafe for a preemptible long run.
- Hardened `examples/bioseq/train_bioseq_ddp.py`:
  - `save_checkpoint` now stores `backbone_state_dict` + `optimizer_state_dict` + `step` + `epoch` + `args`, writes atomically (temp file + `os.replace`), and always refreshes `output_dir/latest.pt` (periodic saves overwrite `latest.pt` only, so disk stays bounded; `final.pt` is also written at the end).
  - Added `maybe_resume()` and a `--resume` flag (`auto` resumes `<output-dir>/latest.pt` if present — the safe default for preemptible jobs; a path resumes that file; `none` disables). Optimizer tensors are moved back to the training device after load, and `start_step` continues the loop/lr schedule.
  - Default `--save-interval` changed from 0 to 200; saves are rank-0 only with a following `barrier()`.
- Verified on GPU (env `protenix_abtcr`, `device=cuda:0`): a fresh run saved `latest.pt`/`final.pt` (~9.76 GB each, now including optimizer state), then a second run with `--resume auto` logged `[resume] loaded output/.../latest.pt -> resuming at step 2` and continued steps 2->3, confirming weights+optimizer+step resume works.
- Updated `train_jobs/bioseq_mixed_16gpu_test.yml` to use `--save-interval 50 --resume auto` so resubmissions checkpoint periodically and auto-resume after preemption.

### Formal long training submission (16 GPU, from pretrained)

- Copied the Ophiuchus-Ab pretrained checkpoint onto the mounted Vepfs so cluster nodes can read it: `/vepfs-mlp2/c20250601/251105016/model_weights/ophiuchus_ab/Ophiuchus-Ab.ckpt` (3253751875 bytes, size-matched to the `/c20250601/mj/...` source).
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/bioseq_mixed_16gpu_train.yml`: 2 nodes x 8 GPU, fine-tunes from the pretrained checkpoint (`--checkpoint-path .../Ophiuchus-Ab.ckpt`, no `--init-multimer` since the checkpoint already has multimer weights) on a balanced 3-source mixture (`--limit-per-source 300000` => ~900k samples, which also keeps antibody/TCR from being drowned by the 11.6M nanobody rows), `--max-length 320 --batch-size 8 --max-steps 20000 --warmup-steps 200 --bf16`, periodic `--save-interval 500 --resume auto`, online wandb (project `bioseq-ophiuchus`, run `bioseq_mixed_16gpu_train`), `ActiveDeadlineSeconds: 86400`.
- Submitted: `task_id=t-20260614015013-lz2b8` (2026-06-13 17:50). Monitoring state every ~2 minutes and checking logs once Running.

## 2026-06-13 Qwen3-VL architecture audit

- Reviewed local Qwen3-VL implementation under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/base_model/Qwen3-VL/qwen-vl-finetune/qwenvl/model/qwen3_vl`.
- Confirmed dense Qwen3-VL is composed of a vision tower (`Qwen3VLVisionModel`), a connector/merger (`Qwen3VLVisionPatchMerger`), and a Qwen3-style causal text decoder (`Qwen3VLTextModel`) wrapped by `Qwen3VLForConditionalGeneration`.
- Confirmed processor behavior: image/video placeholders are expanded into the exact number of visual tokens, and `mm_token_type_ids` mark text/image/video regions for multimodal RoPE.
- Confirmed forward path: image/video tensors are encoded by the vision tower, merged visual embeddings are `masked_scatter`-inserted into the special-token positions of `inputs_embeds`, and visual deepstack features are injected back into early decoder hidden states at visual token positions.
- Confirmed position handling: Qwen3-VL computes 3D T/H/W visual positions plus text positions through multimodal RoPE and caches `rope_deltas` for generation.
- Confirmed local Qwen3-VL also contains a MoE variant under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/base_model/Qwen3-VL/qwen-vl-finetune/qwenvl/model/qwen3_vl_moe`; MoE changes the text decoder MLP/router path, while the visual tower/merger pattern remains aligned.
- Confirmed finetuning knobs split trainability and learning rates for LLM, merger/projector, and vision tower through `tune_mm_llm`, `tune_mm_mlp`, `tune_mm_vision`, `mm_projector_lr`, and `vision_tower_lr`.

## 2026-06-13 BioSeq diffusion-loss design clarification

- User clarified that the foundation-model objective should remain diffusion-only even when an encoder is used.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` to make ESMC/ESM2 trainable by default in the feature-conditioned path.
- Added the training-mask rule that fixed context chains, such as antigen in antibody-antigen generation, stay clean and visible, do not participate in remasking, and do not receive direct diffusion loss.
- Added the gradient-flow rule that fixed context encoders/connectors still receive gradients through the target-chain denoising loss when their features condition the decoder.

## 2026-06-13 multi-chain tokenization research

- User proposed adding special tokens to distinguish antibody, antigen, TCR, TCR-pMHC, and PPI chains.
- Surveyed multi-chain handling patterns in AlphaFold-Multimer, ESMFold, ProteinMPNN, RFdiffusion, Protenix, and recent PLM-based multi-chain PPI work.
- Main takeaway: multi-chain inputs should not rely on one generic separator only. Mature systems preserve chain/entity/copy identity and fixed/design status through explicit chain metadata and masks.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` with a `Multi-Chain Tokenization Notes` section.
- Proposed BioSeq design: use special tokens for chain/task boundaries and biological role hints, while using per-residue embeddings and masks for `chain_id`, `entity_id`, copy id, chain role, residue position, target/context status, and diffusion/fixed control.

## 2026-06-13 AirGen parity tests

- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/airgen_reference.py` with a minimal copy of AirGen `q_sample_comp` / `construct_x_t` logic for offline parity checks (full AirGen import is blocked by pytorch_lightning version drift).
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_airgen_parity.py`: verifies migrated `MultiChainOphiuchusAbModel.construct_x_t` matches the AirGen reference on fixed seeds (train + val stage), plus a `_decoding` mask-transition smoke test.
- Verified `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq -q`: 25 passed.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/bioseq_mixed_16gpu_stage1.yml` for the next real mixed-data run: 16 GPU, 5000 steps, full corpora (no `--limit-per-source`), init from Ophiuchus-Ab checkpoint staged under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt`, checkpoint save every 1000 steps.
- **Current step**: 16-GPU smoke test passed; AirGen parity gate passed; **ready to submit stage-1 training** (`bioseq_mixed_16gpu_stage1.yml`). After stage-1: full JSONL conversion + IRBench `common/` metrics/schema.
- Staged Ophiuchus-Ab checkpoint to `/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt` (3.1G) so cluster nodes can load it from the Vepfs mount.
- Submitted stage-1 mixed training job: `task_id=t-20260614013528-lkxcn` (2026-06-13 17:35), config `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/bioseq_mixed_16gpu_stage1.yml`, 16 GPU, 5000 steps, full corpora, init from staged Ophiuchus-Ab checkpoint, save every 1000 steps.

## 2026-06-13 tokenization simplification

- User clarified that the first foundation pretraining format should be simpler: the core objective is masked generation with different mask probabilities, not a heavily annotated biological prompt language.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` to reduce the token vocabulary to amino acids, basic special tokens, `<chain_sep>`, and a small set of complex-type header tokens such as `<type_ab>`, `<type_tcr_pmhc>`, and `<type_ppi>`.
- Replaced detailed role-token defaults with chain ordering conventions under each header token.
- Added the two-level position rule: chain-internal residue position plus outer chain index; biological relationship representations should be learned later through encoder/connector features rather than through many special tokens.

## 2026-06-13 immune receptor foundation model considerations

- Researched current antibody and TCR-pMHC modeling/data resources to identify what BioSeq must preserve beyond minimal sequence tokens.
- Key findings: paired heavy-light antibody data, paired alpha-beta TCR data, peptide/MHC context, V/D/J gene annotations, CDR/framework spans, assay metadata, and explicit negative sampling are all important for immune receptor modeling and should be preserved in schema.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` with an `Immune Receptor Modeling Considerations` section.
- Design decision: keep first-version tokenizer minimal, but preserve immune metadata in dataset records and expose it later through mask policies, evaluation splits, or encoder/connector features.

## 2026-06-13 additional immune data type survey

- User asked what other data types should be considered for the immune receptor foundation model.
- Surveyed repertoire-scale AIRR-seq resources, paired receptor databases, antigen/specificity-labeled data, structural immune-complex resources, MHC/epitope data, single-cell immune multi-omics, germline/numbering references, functional/developability labels, synthetic screening data, and negative/decoy protocols.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` with an `Additional Data Types` section.
- Design decision: only repertoire-scale and paired-chain data should be first-class pretraining pools at the beginning; specificity, structure, function, and clinical/multi-omics data should initially be treated as conditioning, fine-tuning, evaluation, or metadata sources.

## 2026-06-13 TCR sequence data expansion audit

- User asked whether current TCR sequence data is enough and whether more TCR data should be downloaded.
- Audited local data state: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean/final` has about 2.12M cleaned paired alpha/beta records; `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr` already includes VDJdb, McPAS-TCR, ImmuneCODE/MIRA, IEDB archive, and PIRD-related files; `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/processed_v2/stats.json` includes processed VDJdb/McPAS/MIRA/TCR/PPI counts.
- Conclusion: current local TCR data is enough to start first masked-generation pretraining. The more urgent code/data work is unified adapters, manifests, schema normalization, and mixture weights, not blind download volume.
- Recommended future downloads are split into two pools: bulk/unpaired TCR repertoire data under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw`, and TCR-pMHC specificity/context data under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_specificity_raw`.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` with `TCR Data Expansion Notes`.

## 2026-06-13 TCRdb2.0 raw download

- User requested a TCRdb2.0 download for expanding bulk TCR repertoire coverage.
- Target root: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0`.
- Resolved the TCRdb2.0 frontend/API through `/TCRdb2/` and saved site/API provenance under `site_assets/` and `manifests/`; the API returned 267 project rows and 263 unique downloadable project IDs.
- Downloaded all available raw files by type:
  - 263 project zip files to `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/raw/project_zips`.
  - 263 metadata CSV files to `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/raw/metadata`.
  - 1 healthy reference zip to `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/raw/healthy`.
- Wrote URL and HEAD manifests under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/manifests`, including `tcrdb2_project_zip_urls.txt`, `tcrdb2_metadata_urls.txt`, `tcrdb2_special_urls.txt`, and `tcrdb2_head_inventory.tsv`.
- Size validation passed against remote `Content-Length`: project zips `17954067855` bytes, metadata `6351148` bytes, healthy reference `1289897700` bytes, with `missing_count=0` and `mismatch_count=0`; details are in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/manifests/tcrdb2_download_validation.tsv`.
- Lightweight zip structure validation passed for 264 zip files with `invalid_count=0` and `empty_zip_count=0`; summary is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/manifests/tcrdb2_zip_structure_check.json`.

## 2026-06-14 BioSeq pipeline and fast validation plan

- User asked to reorganize the overall foundation-model pipeline beyond data download and define how to quickly validate model quality on downstream tasks.
- Audited local integration points: training entry `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_bioseq_ddp.py`, source adapters under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq`, and IRBench under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark`.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` with an `End-to-End Pipeline` section covering raw data, manifests, adapters, canonical JSONL, mixture weights, masking/collation, no-encoder Ophiuchus/MINT training, future ESMC/ESM feature-conditioned training, checkpointing, and evaluation.
- Updated the same plan with a `Fast Downstream Validation Plan`: Tier 0 held-out diffusion sanity, Tier 1 frozen-embedding IRBench, Tier 2 generation/infill, and Tier 3 task-specific fine-tuning.
- Fixed `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/model_api.py` so `BioSeqEmbedder` can load current DDP checkpoints containing `backbone_state_dict`, generic `state_dict`, or a plain state dict. This makes `--embedder bioseq:/abs/path/final.pt` compatible with checkpoints produced by `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_bioseq_ddp.py`.
- Verified the edited benchmark model API with `python -m py_compile` and a lightweight factory import check.

## 2026-06-14 data format audit for BioSeq foundation

- User asked to first quantify the current `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data` formats before adapting Qwen into an immune receptor diffusion foundation model.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md` with top-level data sizes, file-format inventory, current JSONL schema, OAS/OTS/nanobody clean CSV schemas and row counts, TCR specificity resources, TCRdb2.0 raw schema, PPI Arrow schema, downstream benchmark formats, and model-input implications.
- Key conclusion: current `processed`/`processed_v2` JSONL is only a partial unified format for PPI + TCR-epitope; the large clean OAS, OTS, nanobody, and TCRdb2.0 pools are not fully merged into one canonical BioSeq training corpus yet.
- Qwen adaptation should target a stable `bioseq.v1` boundary with `chains`, `task_type`, `complex_type`, `chain_roles`, `targets`, `regions`, `metadata`, plus tensor fields for `chain_ids`, `chain_role_ids`, inner/outer position ids, `diffusion_loss_mask`, and `fixed_context_mask`.
- Refined the schema after user pointed out chain-level `targets` is too simple. `targets` is now treated as a coarse default only; real training/inference must use `generation_spec` or a sampled task view that resolves to token-level `visible_mask`, `fixed_context_mask`, `diffusion_target_mask`, and `diffusion_loss_mask`. Required supported views include heavy-to-light, beta+epitope-to-alpha, FR-conditioned CDR infilling, single-CDR infilling, and CDR-conditioned FR generation.

## 2026-06-14 Qwen3-VL architecture migration

- User asked to migrate only the latest Qwen architecture code from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/base_model` into `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines` before BioSeq-specific modifications.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch` with only dense and MoE Qwen3-VL architecture files: `configuration`, `modeling`, and `modular` for `qwen3_vl` and `qwen3_vl_moe`.
- Intentionally did not migrate Qwen2/Qwen2.5, processors, video processors, finetuning scripts, data loaders, demo assets, cookbooks, evaluation scripts, or Docker files.
- Adjusted the migrated architecture imports so shared Hugging Face utilities resolve through `transformers.*`; local dense/MoE Qwen3-VL imports remain relative inside the migrated package.
- Verified the migrated files with `python -m py_compile` and package-level imports for `dllm.pipelines.qwen3_vl_arch`, `qwen3_vl`, and `qwen3_vl_moe`.
- Current runtime `transformers==4.48.1` is older than the Qwen3-VL snapshot and lacks newer internal APIs such as `transformers.masking_utils`, `transformers.modeling_layers`, `transformers.vision_utils`, `transformers.utils.output_capturing`, `transformers.initialization`, `RopeParameters`, and `auto_docstring`, so concrete model/config imports will need a matching newer Transformers version or local compatibility shims before execution.

## 2026-06-14 BioSeq foundation data loader

- User chose to prioritize a training-time loader instead of full offline conversion to JSON/JSONL.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data` with canonical records, source loaders, mixture datasets, view sampler, ESM-family tokenizers, and BioSeq foundation diffusion collator.
- First-version sources cover OAS paired antibody CSV, OTS paired TCR CSV, nanobody CSV, existing processed JSONL for PPI/TCR-epitope, and an optional PPI Arrow source.
- Loader output is `BioSeqRecord`; masking is deferred to `BioSeqViewSampler` and `BioSeqQwenDataCollator`, which emit `visible_mask`, `fixed_context_mask`, `diffusion_target_mask`, and `diffusion_loss_mask`.
- Encoding defaults to ESM2/MINT-compatible token ids through `Esm2SequenceTokenizer`; the collator also emits per-chain `encoder_input_ids`, `encoder_attention_mask`, `encoder_residue_mask`, `encoder_chain_mask`, and `encoder_chain_role_ids` for ESM2/ESMC feature-conditioned training.
- Verified with `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq -q`: 27 passed.

## 2026-06-14 ESM tokenizer and multi-chain paper verification

- Verified local ESM2 tokenizer files under `/c20250601/mj/model_weights/esm2/esm2_t30_150M_UR50D`, `/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D`, and `/c20250601/mj/model_weights/esm2/esm2_t36_3B_UR50D`: ids 0-32 are `<cls>`, `<pad>`, `<eos>`, `<unk>`, `L`, `A`, `G`, `V`, `S`, `E`, `R`, `T`, `I`, `D`, `P`, `K`, `Q`, `N`, `F`, `Y`, `M`, `H`, `W`, `C`, `X`, `B`, `U`, `Z`, `O`, `.`, `-`, `<null_1>`, `<mask>`.
- Verified local ESMC tokenizer files under `/c20250601/mj/model_weights/esmc/ESMC-300M`, `/c20250601/mj/model_weights/esmc/ESMC-600M`, and `/c20250601/mj/model_weights/esmc/ESMC-6B`: ids 0-30 and 32 match the ESM2 amino-acid/special-token ids above, but id 31 is `|` rather than `<null_1>`, and `special_tokens_map.json` marks `|` as an additional special token.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/README.md` with the tokenizer rule: use `Esm2SequenceTokenizer` for Ophiuchus-Ab/MINT/ESM2 paths; use the local Hugging Face tokenizer adapter or an ESMC-specific tokenizer for ESMC encoder paths.
- Re-extracted `/tmp/esm_protein.pdf` to `/tmp/esm_protein.txt` with Ghostscript and checked the ESMC/ESMFold2 preprint. The paper's ESMC section describes a masked language model over protein sequences and single-chain contact evaluation; the explicit multi-chain treatment appears in ESMFold2.
- Paper-specific conclusion: ESMFold2 uses frozen ESMC 6B representations. For multiple protein chains, each chain is encoded independently by ESMC, then ESMFold2 crops/concatenates chain representations into a complex-level folding trunk with pair representations and atom-level diffusion. Therefore BioSeq should keep multi-chain interaction learning in the decoder/collator/attention or an ESMFold2-style pair module, not assume ESMC alone models cross-chain interactions.

## 2026-06-14 BioSeq foundation view-mask correction

- User clarified that `full_denoise` must not include antigen or pMHC context chains in diffusion loss.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/view_sampler.py`: `full_denoise` now targets only eligible chains. It honors explicit `metadata["targets"]` when present, then filters out antigen, peptide, MHC, HLA-like, and epitope roles as fixed context.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/sources.py` so `processed_json_to_record` only writes `metadata["targets"]` when the source row explicitly provides `targets`; rows without targets now let the view sampler infer eligible chains from roles.
- Added coverage in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py` to verify that `full_denoise` keeps peptide, MHC, and antigen fixed even when metadata accidentally lists all chains as targets.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/README.md` with the corrected `full_denoise` semantics.

## 2026-06-14 BioSeq foundation data reading diagnostic

- Added temporary diagnostic script `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/inspect_qwen3_vl_data_reading.py`.
- The script samples local sources through the current BioSeq foundation loader, then checks source parsing, record roles/tasks, `full_denoise` masks, default random views, empty-loss examples, collator errors, and whether fixed context roles accidentally enter `diffusion_loss_mask`.
- Verified syntax with `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/inspect_qwen3_vl_data_reading.py`.
- Ran `python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/inspect_qwen3_vl_data_reading.py --limit-per-source 512 --batch-size 16 --max-chain-length 512`. OAS, OTS, nanobody, and `processed_v2` all yielded 512 records with no unknown roles/tasks, no collator errors, no empty-loss examples, and `full_denoise context_loss_tokens=0`.
- Observed and fixed a data-shape issue: first 512 `processed_v2` records included PPI chains mostly as role `other` rather than `protein_a`/`protein_b`. Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/sources.py` so `source=ppi` records normalize the first two chains to `protein_a` and `protein_b`.
- Added coverage in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py` for PPI partner role normalization; `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py -q` now reports `8 passed`.
- Re-ran processed_v2 diagnostics after the fix: first 512 records now report roles `protein_a=411`, `protein_b=411`, `tcr_alpha=42`, `tcr_beta=101`, and `antigen=86`; issues remain `none`.
- Optional PPI Arrow source check failed with `ImportError: PpiArrowSource requires the datasets package`; the default CSV/JSONL loader path does not depend on this package.

## 2026-06-14 MHC-conditioned peptide/TCR view

- Clarified terminology: `full_denoise` is a mask/view policy for denoising eligible target residues; `chain` means a biological sequence entity such as antibody heavy/light, TCR alpha/beta, peptide, MHC, antigen, or PPI partner, while `residue` means one amino-acid token inside a chain.
- User requested the TCR-pMHC case where MHC is fixed while peptide and TCR chains both participate in diffusion.
- Added `mhc_to_peptide_tcr` to `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/view_sampler.py`. It requires an MHC/HLA-like chain as context and targets available `tcr_alpha`, `tcr_beta`, `peptide`, and `epitope` chains. For `tcr_epitope` or `tcr_pmhc` records, an `antigen` role is treated as peptide/epitope target when this view is selected.
- Added `tcr_mhc_to_peptide` for fixed TCR alpha/beta plus MHC/HLA designing peptide or epitope.
- Added `pmhc_to_tcr` for fixed peptide or epitope plus MHC/HLA designing TCR alpha/beta.
- Confirmed current FR/CDR views are not antibody-only. They are region-driven and apply to TCR full-chain data whenever the source adapter preserves `FR*` and `CDR*` region annotations.
- Added test coverage in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py` verifying that `mhc_to_peptide_tcr` puts alpha/beta/peptide in `diffusion_loss_mask` and keeps MHC in `fixed_context_mask`.
- Added tests for `tcr_mhc_to_peptide` and `pmhc_to_tcr`, verifying peptide-only target masks and alpha/beta-only target masks respectively.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/README.md` with this view.
- Verified `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py -q`: `11 passed`.
- Re-ran the temporary data diagnostic with `python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/inspect_qwen3_vl_data_reading.py --limit-per-source 128 --batch-size 16 --max-chain-length 512`; OAS, OTS, nanobody, and `processed_v2` all reported `issues: none`.

## 2026-06-14 Antibody-antigen and nanobody-antigen views

- User clarified that the same fixed/target logic used for TCR alpha/beta should also apply to antibody-antigen and nanobody-antigen cases.
- Added antibody/nanobody antigen-conditioned views to `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/view_sampler.py`:
  - `antigen_to_antibody`: fixed antigen, target available antibody heavy/light chains.
  - `antigen_to_nanobody`: fixed antigen, target nanobody VHH.
  - `heavy_antigen_to_light`: fixed antibody heavy plus antigen, target antibody light.
  - `light_antigen_to_heavy`: fixed antibody light plus antigen, target antibody heavy.
- Added task ids for `antibody_antigen` and `nanobody_antigen` in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/records.py`.
- Added tests in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py` for antigen-to-antibody, heavy+antigen-to-light, and antigen-to-nanobody masks.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/README.md` with antibody/nanobody antigen view semantics.
- Verified `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py -q`: `15 passed`.
- Re-ran the temporary data diagnostic with `python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/inspect_qwen3_vl_data_reading.py --limit-per-source 128 --batch-size 16 --max-chain-length 512`; OAS, OTS, nanobody, and `processed_v2` all reported `issues: none`.

## 2026-06-14 Remove antibody inverse-antigen views

- User clarified that antibody and nanobody inverse antigen-generation views are not needed for the current antibody design setting.
- Removed `antibody_to_antigen` and `nanobody_to_antigen` from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/view_sampler.py` default and build path.
- Removed the nanobody-to-antigen mask test from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py`.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/README.md`: antigen is fixed conditioning context for antibody/nanobody receptor generation and does not receive diffusion loss by default.
- Verified `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py -q`: `14 passed`.
- Re-ran `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/inspect_qwen3_vl_data_reading.py --limit-per-source 128 --batch-size 16 --max-chain-length 512`; OAS, OTS, nanobody, and `processed_v2` all reported `issues: none`.

## 2026-06-14 Antibody/TCR generation task survey

- Surveyed current antibody generation tasks: DiffAb, RFdiffusion antibody design, AbX, IgLM, Ophiuchus-Ab notes, and paired antibody language models all support prioritizing receptor-side generation or infilling: CDR generation, full heavy/light generation, heavy-light pairing/completion, antigen-conditioned antibody design, and optional framework/humanization-style infilling.
- Conclusion for antibody/nanobody views: keep antigen as fixed clean context; train diffusion loss on antibody heavy/light, VHH, or CDR/FR spans. Do not include default inverse antigen generation.
- Surveyed current TCR generation tasks: TCR-TRANSLATE, TCRdesign, TCR-pMHC binding task definitions, and paired alpha/beta TCR structure analyses support pMHC/epitope-conditioned TCR generation and paired alpha/beta completion as primary training/evaluation directions.
- Conclusion for TCR views: prioritize `pmhc_to_tcr`, alpha/beta paired-chain completion, and CDR/FR infilling when annotations exist. Keep peptide/epitope generation views as lower-weight or downstream-specific because they are useful but less central than receptor generation.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` with the view priority and source list.

## 2026-06-14 Task-specific BioSeq view profiles

- User clarified that antibody-antigen generation also includes fixed antigen plus antibody FR regions generating CDR regions, so views must be divided by data/task type rather than treated as one flat global list.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/view_sampler.py` so default sampling uses task-specific profiles for antibody, antibody-antigen, nanobody-antigen, TCR, TCR-epitope, TCR-pMHC, PPI, and generic records.
- Added `antigen_fr_to_cdr` and `antigen_single_cdr`: fixed antigen plus receptor non-target residues generate antibody/nanobody CDR spans.
- Added `pmhc_fr_to_cdr` and `pmhc_single_cdr`: fixed peptide/epitope plus MHC/HLA and non-target TCR residues generate TCR CDR spans.
- Kept `allowed_views` override behavior for ablations and downstream-specific training, including lower-priority peptide-generation/co-design views such as `tcr_mhc_to_peptide` and `mhc_to_peptide_tcr`.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/README.md` with the task-specific profile semantics.
- Verified syntax with `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/view_sampler.py`.
- Verified `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py -q`: `17 passed`.
- Re-ran `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/inspect_qwen3_vl_data_reading.py --limit-per-source 128 --batch-size 16 --max-chain-length 512`; OAS, OTS, nanobody, and `processed_v2` all reported `issues: none` under the new default profiles.

## 2026-06-14 Task-homogeneous training batches

- User raised a training-stability requirement: a batch should not mix different task/view objectives.
- This section is now superseded by the later `2026-06-14 Mixed-task BioSeq foundation batches` decision. `TaskHomogeneousBatchDataset` remains available for ablations/debugging, but it is not the current default foundation-training path.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/mixture.py::TaskHomogeneousBatchDataset`. It wraps any BioSeq record stream and groups records by BioSeq task group.
- Added `bioseq_task_group` and `bioseq_record_fingerprint` helpers. `bioseq_task_group` separates nanobody from paired antibody, and separates antibody-antigen/nanobody-antigen from generic antibody records based on chain roles.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/collator.py`: `BioSeqQwenDataCollator` now samples one shared generation view per batch by default through `BioSeqViewSampler.sample_batch`, and can enforce homogeneous task groups with `require_homogeneous_task=True`.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/view_sampler.py` with `sample_batch`, `compatible_views`, and public `build` helpers.
- At this point the documented training path was source stream -> `TaskHomogeneousBatchDataset` -> `DataLoader(batch_size=None)` -> `BioSeqQwenDataCollator(require_homogeneous_task=True)`. That path was later replaced by mixed physical batches using `WeightedMixtureDataset` -> `DataLoader(batch_size=N)` -> `BioSeqQwenDataCollator(single_view_per_batch=False, require_homogeneous_task=False)`.
- The old training-stability note favored homogeneous physical microbatches plus gradient accumulation; the current default instead uses mixed physical microbatches and controls imbalance through source mixture weights.
- Verified syntax with `python -m py_compile` for `mixture.py`, `collator.py`, and `view_sampler.py`.
- Verified `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py -q`: `20 passed`.
- Re-ran `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/inspect_qwen3_vl_data_reading.py --limit-per-source 128 --batch-size 16 --max-chain-length 512`; OAS, OTS, nanobody, and `processed_v2` all reported `issues: none`.

## 2026-06-14 Batch de-duplication boundary correction

- User clarified that sample de-duplication is already handled during data processing, so batch-level de-duplication should not be part of the default training path.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/mixture.py::TaskHomogeneousBatchDataset`: `deduplicate_within_batch` now defaults to `False`; setting it to `True` is only a defensive option for debugging untrusted/overlapping streams.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/README.md` and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` to clarify the boundary at that time: data processing owns de-duplication, while the training batcher owned task/view homogeneity. The current mixed-task path no longer uses task/view homogeneity as the default batcher responsibility.
- Verified syntax with `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/mixture.py`.
- Verified `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py -q`: `21 passed`.

## 2026-06-14 Simple view sampling probability

- User clarified that view sampling should stay simple: keep `full_denoise` high, and randomly sample other condition views from the remaining probability mass.
- Historical note from that iteration: `BioSeqViewSampler` used `full_denoise_probability=0.5` by default and split the remaining probability across compatible condition views.
- This historical 0.5 setting was superseded later on 2026-06-16; the current foundation default is `full_denoise_probability=1.0`.
- The same rule is used for both single-record sampling and batch-level shared-view sampling.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/README.md` and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` with the default probability rule.
- Verified syntax with `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/view_sampler.py`.
- Verified `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py -q`: `24 passed`.
- Re-ran `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/inspect_qwen3_vl_data_reading.py --limit-per-source 128 --batch-size 16 --max-chain-length 512`; OAS, OTS, nanobody, and `processed_v2` all reported `issues: none`. OAS/OTS/nanobody each sampled `full_denoise` for `80/128` examples in the deterministic diagnostic run, reflecting the new high full-denoise rate under batch-level shared-view sampling.

## 2026-06-14 PPI and interaction task data ingestion

- User requested downloading and consolidating PPI/interaction task datasets covering STRING/MINT, Figshare gold-standard PPI, HumanPPI, YeastPPI, SKEMPI, PDBbind, SWING MutInt, FLAb, SARS-CoV-2 antibody binding, TDC TCR-epitope, PISTE, TEIM, oncoPPI, and CoV-AbDab.
- Created the raw data root `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw` with per-source raw directories, manifests, logs, and processed outputs.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/build_ppi_interaction_csv.py` to rebuild:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_sources_manifest.csv`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_records_summary.csv`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_records_unified.csv`
- The unified CSV currently contains `6271559` row-level records plus header. Source counts: FLAb `4606793`, PISTE `1051227`, Figshare gold-standard PPI `274500`, oncoPPI `106536`, HumanPPI `68945`, TDC TCR-epitope `47182`, TEIM `45603`, YeastPPI `38158`, CoV-AbDab `12918`, SWING MutInt `12612`, and SKEMPI `7085`.
- HumanPPI and YeastPPI were downloaded as LMDB split zip files, extracted under their source raw directories, and parsed through a pickle/LMDB reader mapping `primary_1`, `primary_2`, and `interaction` into the unified schema.
- Figshare files and SKEMPI files initially hit transient `502` or unsupported range-download errors; they were successfully re-downloaded with single-connection `curl` retries.
- CoV-AbDab CSV, numbering JSON, and bibliography were downloaded. The PDB structures tarball was stopped as a partial optional structure attachment because it was not needed for the current CSV and was still a slow long-tail download.
- STRING-DB v12.0 physical links/sequences were identified as multi-GB files (`protein.physical.links.v12.0.txt.gz`, `protein.physical.links.full.v12.0.txt.gz`, `protein.sequences.v12.0.fa.gz`) and remain partial rather than blocking this task.
- PDBbind+ remains blocked by login/subscription through its site API. The SARS-CoV-2 binding bioRxiv supplement remains blocked by HTTP 403 from this environment.
- Validation: no active download/conversion processes remained; `wc -l` on the unified CSV returned `6271560`, and the summary record sum returned `6271559`.

## 2026-06-14 BioSeq foundation training model layer

- User request: start building the trainable model architecture under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch`, with both a no-encoder version and an ESMC/ESM feature-conditioned version. Feature-conditioned training should follow the BioSeq diffusion loss/noise rules.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`.
- Implemented `BioSeqDiffusionTransformerConfig`, `BioSeqDiffusionDecoder`, `BioSeqNoEncoderDiffusionModel`, and `BioSeqEncoderDiffusionModel`.
- Implemented BioSeq diffusion utilities in the same module:
  - `sample_bioseq_diffusion_noise`: samples timestep corruption only from `diffusion_loss_mask` / `diffusion_target_mask`, keeps fixed context residues clean, guarantees at least one corrupted target residue per eligible row, and returns noised decoder ids plus labels.
  - `apply_decoder_corruption_to_encoder`: maps corrupted decoder residues back to per-chain encoder residues and masks those encoder tokens before the encoder forward pass.
  - `compute_masked_cross_entropy`: computes denoising cross-entropy only on corrupted target positions.
- The no-encoder model computes diffusion loss directly over the qwen3_vl_arch collator output.
- The initial feature-conditioned implementation ran a per-chain encoder and projected encoder states into the decoder. This was later corrected so ESMC/ESM features stay token-aligned from the diffusion state `x_t` before the multi-chain denoiser runs.
- Important leakage rule implemented: when a target residue is corrupted for decoder diffusion, the corresponding residue token in `encoder_input_ids` is also replaced with `<mask>` before encoder forward. Fixed context chains such as antigen remain clean and can condition target denoising.
- Exported the new model/loss utilities from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/__init__.py`.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py`.
- Test coverage verifies no-encoder masked-diffusion loss, ESMC/ESM feature-conditioned loss, encoder target masking, fixed antigen preservation, trainable encoder gradients, frozen encoder behavior, extra collator-field tolerance, and diffusion sampler mask boundaries.
- Local ESMC loading check: `/c20250601/mj/model_weights/esmc/ESMC-300M/config.json` declares `model_type="esmc"` and `transformers_version="4.57.6"`. Current environment has `transformers==4.48.1`, so `transformers.AutoModel.from_pretrained("/c20250601/mj/model_weights/esmc/ESMC-300M")` fails because this Transformers version does not recognize ESMC.
- Resulting constraint at this point was that unit tests used a tiny differentiable encoder to validate the training path. This was later resolved by adding the local Biohub `esm==3.2.3` ESMC loader documented in the 2026-06-14 compatibility fix section below.
- Verified syntax:
  - `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py`
- Verified tests:
  - `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py -q`: `5 passed`
  - `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py -q`: `29 passed`

## 2026-06-14 BioSeq foundation DDP training path

- User clarified that training needs to support multi-node/multi-GPU execution.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/mixture.py` so iterable record streams are DDP-aware. `SequentialMultiSourceDataset` and `WeightedMixtureDataset` now shard source rows with `global_shard_index = rank * num_workers + worker_id` and `num_shards = world_size * num_workers`.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py`.
- The new trainer supports:
  - `torchrun` single-node and multi-node launch through environment variables `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`.
  - no-encoder `--model-type no_encoder` training today.
  - ESMC/ESM feature-conditioned `--model-type encoder` through `BioSeqEncoderDiffusionModel.from_esmc(...)`, now backed by the local Biohub ESMC fallback loader when Hugging Face `AutoModel` cannot recognize `model_type="esmc"`.
  - initially supported task-homogeneous BioSeq batches from `TaskHomogeneousBatchDataset`; the default was later changed to mixed-task physical batches.
  - initially supported shared-view collation through `BioSeqQwenDataCollator(require_homogeneous_task=True)`; the default was later changed to per-record views with `require_homogeneous_task=False`.
  - `--device auto|cuda|cpu`, so CPU DDP smoke tests can run even on nodes with fewer visible GPUs than requested local ranks.
  - bf16 autocast, gradient accumulation, grad clipping, warmup, rank-0 logging, optional wandb, checkpoint save/resume, and separate encoder LR when an encoder model is active.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_ddp_training.py`, which runs a one-step single-process no-encoder training smoke on real local OAS data and checks that `final.pt` is saved.
- Added cluster template `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_16gpu_smoke.yml` for a 2-node x 8-GPU no-encoder qwen3_vl_arch smoke run.
- Local first `torchrun --standalone --nproc_per_node=2` attempt failed because the current node reports CUDA available but has fewer visible CUDA devices than local ranks; rank 1 could not bind `cuda:1`. This is an environment/resource issue, not a DDP code issue.
- Added `--device cpu` and re-ran local CPU DDP:
  - `torchrun --standalone --nproc_per_node=2 /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py --device cpu --model-type no_encoder --sources oas --limit-per-source 16 --batch-size 2 --max-steps 2 --max-chain-length 64 --max-sequence-length 256 --hidden-size 32 --num-hidden-layers 1 --num-attention-heads 4 --intermediate-size 64 --dropout 0.0 --num-workers 0 --save-interval 1 --resume none --wandb-mode disabled --output-dir /tmp/qwen3_vl_bioseq_ddp_smoke`
  - Result: `world_size=2`, DDP initialized with gloo, two optimizer steps completed, `latest.pt` and `final.pt` saved.
- Verified syntax:
  - `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/mixture.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_ddp_training.py`
- Verified tests:
  - `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_ddp_training.py -q`: `30 passed`

## 2026-06-14 Local ESMC loader compatibility fix

- User reported that `transformers==4.48.1` does not recognize `model_type="esmc"` for local ESMC checkpoints.
- Confirmed that installing a newer `transformers` alone does not make `AutoModel.from_pretrained("/c20250601/mj/model_weights/esmc/ESMC-300M")` work for these local snapshots. The working local path is Biohub `esm==3.2.3` with native `esm.models.esmc.ESMC`.
- Installed/confirmed Biohub `esm==3.2.3` in the active development environment and the task environment `/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr`, keeping `transformers==4.48.1`.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py::LocalESMCEncoder`, `_convert_biohub_esmc_state_dict`, and `load_local_esmc_encoder`.
- The loader reads `/c20250601/mj/model_weights/esmc/<model>/config.json` and `model.safetensors`, builds native Biohub ESMC with `d_model`, `n_heads`, and `n_layers`, maps local Hugging Face-style keys into native `esm` keys, and loads with `strict=True`.
- Updated `BioSeqEncoderDiffusionModel.from_esmc(...)` so it still tries Hugging Face `AutoModel` first, but falls back to `load_local_esmc_encoder(...)` when `AutoModel` cannot recognize ESMC.
- Added `--encoder-use-flash-attn` to `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py` and exported the local ESMC loader from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/__init__.py`.
- Verified syntax:
  - `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/__init__.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py`
- Verified tests:
  - `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_ddp_training.py -q`: `31 passed`
- Verified real ESMC-300M local loading in both current Python and `/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python`: `LocalESMCEncoder`, hidden size `960`, output shape `(1, 5, 960)`.
- Added a stricter ESMC-300M state-dict validation: raw local safetensors have `398` keys including wrapper metadata/extra-state entries; converted native ESMC state dict has `308` keys; native Biohub `ESMC(d_model=960, n_heads=15, n_layers=30)` also has `308` keys. Missing keys, unexpected keys, and shape mismatches are all `0`, and parameter numel matches exactly at `332997184`. Forward output with padding mask is finite with shape `(2, 7, 960)`.
- Verified ESMC-600M with the same strict check. Config is `d_model=1152`, `n_heads=18`, `n_layers=36`, `vocab_size=64`, `mask_token_id=32`, and `pad_token_id=1`. Raw local safetensors have `476` keys; converted native ESMC state dict has `368` keys; native Biohub `ESMC(d_model=1152, n_heads=18, n_layers=36)` also has `368` keys. Missing keys, unexpected keys, and shape mismatches are all `0`, and parameter numel matches exactly at `575036992`. `strict=True` load passed, and wrapper forward output is finite with shape `(2, 7, 1152)`.

## 2026-06-14 Mixed-task BioSeq foundation batches

- User clarified that physical training batches should contain different tasks, rather than only one task group per batch.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py`: `build_loader` now feeds `WeightedMixtureDataset` directly into `DataLoader(batch_size=args.batch_size, drop_last=True)` and no longer wraps the stream with `TaskHomogeneousBatchDataset`.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/collator.py`: `BioSeqQwenDataCollator.single_view_per_batch` now defaults to `False`, so each record samples its own compatible view inside a mixed batch.
- The DDP trainer now constructs `BioSeqQwenDataCollator(single_view_per_batch=False, require_homogeneous_task=False)`. The legacy `--deduplicate-within-batch` flag is retained only as a deprecated compatibility flag and is ignored by the mixed-task path.
- Training logs now report per-batch view counts such as `views=antigen_to_antibody:1,full_denoise:3` instead of assuming one shared view for the whole batch.
- `TaskHomogeneousBatchDataset` remains available as an ablation/debugging wrapper, but it is no longer the default foundation-training path.
- Added a data-loader test that collates antibody and antibody-antigen records in the same batch with per-record views: `["full_denoise", "antigen_to_antibody"]`.
- Verified syntax with `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/collator.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py`.
- Verified tests:
  - `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_ddp_training.py -q`: `32 passed`

## 2026-06-14 Formal ESMC offline training verification

- Confirmed the earlier encoder model ran ESMC per chain/sequence. This was later corrected so `BioSeqEncoderDiffusionModel` keeps token-level ESMC features from `x_t` and gathers them back to decoder residue positions instead of using a chain summary.
- Confirmed ESMC encoder parameters are trainable by default. `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py::optimizer_for_model` creates an AdamW group from `model.encoder.parameters()` when `requires_grad=True`; the formal ESMC-300M and ESMC-600M YAMLs do not pass `--freeze-encoder`.
- Fixed local ESMC tokenizer loading for the BioSeq foundation data path. `transformers==4.48.1` cannot import the `ESMCTokenizer` class declared by the local ESMC snapshots, so `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/esm_encoding.py` now falls back to loading `/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esmc/ESMC-300M/tokenizer.json` or `/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esmc/ESMC-600M/tokenizer.json` through the `tokenizers` library.
- Fixed an edge case in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/collator.py`: if a sampled generation view has no token-level `diffusion_loss_mask` after `max_chain_length` or `max_sequence_length` truncation, the collator falls back to `full_denoise` for that record when `full_denoise` still has eligible target tokens.
- Added a regression test in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py` for the truncated-CDR case where `single_cdr` would otherwise produce zero loss tokens.
- Verified syntax:
  - `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/collator.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/esm_encoding.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py`
- Verified tests:
  - `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_ddp_training.py -q`: `34 passed`
- Verified local ESMC-300M encoder training entrypoint with offline wandb and project-disk output:
  - Command used `/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esmc/ESMC-300M` for both `--encoder-path` and `--tokenizer-path`, `--model-type encoder`, `--wandb-mode offline`, `--max-steps 1`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/local_checks/qwen3_vl_encoder_offline_local_check` as output.
  - Result: `step=0 loss=19.8149 tasks=antibody:1 views=full_denoise:1 corrupted=25`, and `final.pt` was saved under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/local_checks/qwen3_vl_encoder_offline_local_check/final.pt`.
- Verified local ESMC-600M encoder forward/loss without checkpoint saving:
  - Command used `/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esmc/ESMC-600M` for both tokenizer and encoder loading, with a tiny decoder and `freeze_encoder=True` for a no-grad compatibility check.
  - Result: `esmc600m_forward_loss_ok loss 16.4906 logits (1, 22, 64) corrupted 5`.
- Added the formal no-encoder baseline config `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_stage1.yml`.
  - `TaskName`: `qwen3_vl_bioseq_no_encoder_stage1`
  - Model path: `--model-type no_encoder`, with no `--encoder-path`, no `--encoder-lr`, and no `--freeze-encoder`
  - Tokenizer path: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esmc/ESMC-300M`, using only the ESMC tokenizer/vocab 64 for comparability with ESMC encoder runs
  - Batch: `--batch-size 8 --grad-accum 1`, 16 GPU effective batch `128`
  - Wandb: `WANDB_MODE=offline`, `--wandb-mode offline`, project `bioseq-qwen3-vl`, run `qwen3_vl_bioseq_no_encoder_stage1`
- Verified local no-encoder training entrypoint with ESMC tokenizer and offline wandb:
  - Result: `step=0 loss=13.9534 tasks=antibody:1 views=full_denoise:1 corrupted=22`, and `final.pt` was saved under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/local_checks/qwen3_vl_no_encoder_offline_local_check/final.pt`.
- The first local attempt used `/tmp/qwen3_vl_encoder_offline_local_check` and failed only at checkpoint write because the root filesystem `/` was full. Formal training outputs must stay on `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/...`.
- Verified a CPU training smoke with `--sources oas,ots,nanobody --batch-size 4 --max-steps 1`: loss was finite and logs showed mixed per-record views, e.g. `views=fr_to_cdr:1,light_to_heavy:1,single_cdr:2`.
- Verified the mixed source stream directly: the first 12 records from `WeightedMixtureDataset` with OAS, OTS, and nanobody included `nanobody`, `tcr`, and `antibody` task groups before batching.

## 2026-06-14 Formal BioSeq foundation ESMC encoder training configs

- User clarified that the next runs should be formal encoder training, not smoke jobs, and should train two versions: ESMC-300M and ESMC-600M. User also clarified wandb should be offline, not online.
- Staged ESMC weights into the Volc-mounted project tree so training jobs do not depend on `/c20250601/mj/model_weights` being visible inside worker containers:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esmc/ESMC-300M`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esmc/ESMC-600M`
- Added formal job config `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc300m_stage1.yml`.
  - `TaskName`: `qwen3_vl_bioseq_esmc300m_stage1`
  - Model path: `--model-type encoder --encoder-path /vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esmc/ESMC-300M`
  - Tokenizer path: same ESMC-300M snapshot, `--vocab-size 64`
  - Data: `--sources oas,ots,nanobody,processed_v2`, no per-source limit
  - Batch: `--batch-size 2 --grad-accum 4`, 16 GPU effective batch `128`
  - Training: `--max-steps 50000 --lr 1e-4 --encoder-lr 2e-5 --warmup-steps 1000 --bf16`
  - Output: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_esmc300m_stage1`
  - Wandb: `WANDB_MODE=offline`, `--wandb-mode offline`, project `bioseq-qwen3-vl`, run `qwen3_vl_bioseq_esmc300m_stage1`
- Added formal job config `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc600m_stage1.yml`.
  - `TaskName`: `qwen3_vl_bioseq_esmc600m_stage1`
  - Model path: `--model-type encoder --encoder-path /vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esmc/ESMC-600M`
  - Tokenizer path: same ESMC-600M snapshot, `--vocab-size 64`
  - Data: `--sources oas,ots,nanobody,processed_v2`, no per-source limit
  - Batch: `--batch-size 1 --grad-accum 8`, 16 GPU effective batch `128`
  - Training: `--max-steps 50000 --lr 1e-4 --encoder-lr 1e-5 --warmup-steps 1000 --bf16`
  - Output: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_esmc600m_stage1`
  - Wandb: `WANDB_MODE=offline`, `--wandb-mode offline`, project `bioseq-qwen3-vl`, run `qwen3_vl_bioseq_esmc600m_stage1`
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py` so wandb receives per-batch view, task-group, and source-count metrics: `batch_views/*`, `batch_tasks/*`, and `batch_sources/*`.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/collator.py` to emit `task_groups` and `task_types` string lists in the batch.
- Verified staged ESMC files:
  - ESMC-300M `model.safetensors`: `1332036392` bytes
  - ESMC-600M `model.safetensors`: `2300205696` bytes
- Verified YAML parsing for both new configs.
- Verified syntax:
  - `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/collator.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py`
- Verified tests:
  - `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_ddp_training.py -q`: `32 passed`

## 2026-06-14 Formal BioSeq foundation stage-1 task submissions

- Submitted `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc300m_stage1.yml` with the no-proxy Volc wrapper.
  - Task id: `t-20260614215611-mkchk`
  - Initial status from `volc ml_task get`: `Queue`
- Submitted `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc600m_stage1.yml` with the no-proxy Volc wrapper.
  - Task id: `t-20260614215620-kbwb8`
  - Initial status from `volc ml_task get`: `Queue`
- Submitted `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_stage1.yml` with the no-proxy Volc wrapper.
  - Task id: `t-20260614215631-tf8ql`
  - Initial status from `volc ml_task get`: `Queue`
- All three submissions use offline wandb and output under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/...`.
- These initial 2-node submissions failed:
  - `t-20260614215611-mkchk`: worker allocation failed because one worker requested 8 GPUs when available GPUs were `0`.
  - `t-20260614215620-kbwb8` and `t-20260614215631-tf8ql`: preflight pytest failed before training. The worker container did not mount `/c20250601/mj/model_weights/esm2/...`, and importing `transformers.AutoTokenizer` / `AutoModel` triggered a scikit-learn binary import requiring `GLIBC_2.32`.
- Fixed `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/esm_encoding.py` so local ESMC tokenizer snapshots load directly from `tokenizer.json` when `config.json` declares `model_type="esmc"` or `tokenizer_config.json` declares `ESMCTokenizer`.
- Fixed `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py::BioSeqEncoderDiffusionModel.from_esmc` so local ESMC snapshots go directly through `load_local_esmc_encoder(...)` without importing `transformers.AutoModel`.
- Updated the three formal YAMLs to single-node full-8-GPU jobs:
  - `RoleReplicas: 1`
  - `Flavor: ml.pni2.28xlarge`
  - Tags changed from `16gpu` to `8gpu`
  - Effective batch remains `128` by increasing grad accumulation: ESMC-300M `--batch-size 2 --grad-accum 8`, ESMC-600M `--batch-size 1 --grad-accum 16`, no-encoder `--batch-size 8 --grad-accum 2`.
  - Removed full pytest preflight from the formal training entrypoints and kept file/import/tokenizer checks.
- Verified locally after fixes:
  - `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/esm_encoding.py`
  - `BioSeqEncoderDiffusionModel.from_esmc(...)` returned `LocalESMCEncoder 960` for `/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esmc/ESMC-300M`.
  - `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py -q`: `33 passed`
- Submitted corrected 8-GPU stage-1 tasks:
  - ESMC-300M: `t-20260614223522-m5l87`, initial status `Queue`
  - ESMC-600M: `t-20260614223523-8shmt`, initial status `Queue`
  - no-encoder: `t-20260614223319-hzqdr`, status `Running`
- Confirmed no-encoder training is running on 8 GPUs:
  - `world_size=8 device=cuda:0 model_type=no_encoder effective_batch=128`
  - Offline wandb directory: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_no_encoder_stage1/wandb`
  - Logged loss samples: step `0` loss `191.8261`, step `20` loss `175.5349`, step `40` loss `126.2785`, step `60` loss `56.4232`.
- Checked formal task status again:
  - ESMC-300M `t-20260614223522-m5l87`: `Queue`
  - ESMC-600M `t-20260614223523-8shmt`: `Queue`
  - no-encoder `t-20260614223319-hzqdr`: `Running`
- `volc ml_task top --task t-20260614223319-hzqdr --instance worker_0` currently fails with `websocket: bad handshake`, so platform top cannot be relied on for GPU memory.
- Added CUDA memory metrics to `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py`.
  - stdout now appends `mem_peak=<peak_allocated_gb>/<total_gb>` at each log interval on CUDA.
  - offline wandb now receives `perf/gpu_mem_allocated_gb`, `perf/gpu_mem_reserved_gb`, `perf/gpu_mem_peak_allocated_gb`, `perf/gpu_mem_peak_reserved_gb`, and `perf/gpu_mem_total_gb`.
  - In DDP, metrics are max-reduced across ranks so rank 0 logs the largest observed GPU memory usage.
- Current no-encoder task was already running before the memory logging patch, so it will not emit the new memory fields. The queued ESMC-300M and ESMC-600M tasks will pick up the updated script when they launch from the shared Vepfs project path.
- Current batch settings remain conservative until the first memory readings are available:
  - no-encoder: per-GPU `--batch-size 8 --grad-accum 2`
  - ESMC-300M: per-GPU `--batch-size 2 --grad-accum 8`
  - ESMC-600M: per-GPU `--batch-size 1 --grad-accum 16`
- Batch-size adjustment rule for next resubmission: keep effective batch near `128`; if peak allocated memory is below about 70% of device memory for several log windows, increase per-GPU batch and reduce grad accumulation. If peak is above about 90%, keep or reduce per-GPU batch.
- Continued polling after the corrected submissions:
  - ESMC-300M `t-20260614223522-m5l87`: still `Queue`, no container logs yet.
  - ESMC-600M `t-20260614223523-8shmt`: still `Queue`, no container logs yet.
  - no-encoder `t-20260614223319-hzqdr`: still `Running`.
- Latest no-encoder logs showed training continued normally through step `6520`.
  - Recent loss range was roughly `1.15` to `2.87`, with step `6520` loss `2.3875`.
  - Throughput stayed around `1590` to `1625` samples/s after checkpoint pauses.
  - Checkpoints were saved at step `4000`, `5000`, and `6000` to `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_no_encoder_stage1/latest.pt`.
- A later no-encoder log check reached step `7900`; recent losses ranged roughly from `0.63` to `2.54`, and throughput remained around `1590` to `1622` samples/s.
- No new failure was observed in the running no-encoder task. The two encoder tasks have not started yet, so there is no new encoder-side runtime error to fix or resubmit.
- Did not submit duplicate same-name jobs while the corrected ESMC jobs are already queued, because duplicate submissions would compete for the same output and offline wandb directories under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/...`.

## 2026-06-14 Encoder DDP unused-parameter failure and resubmission

- The corrected encoder tasks later launched and failed quickly:
  - ESMC-300M `t-20260614223522-m5l87`: `Failed`, worker exit code `1`.
  - ESMC-600M `t-20260614223523-8shmt`: `Failed`, worker exit code `1`.
- Both failures had the same root cause from PyTorch DDP:
  - `RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.`
  - DDP reported parameters that did not receive gradients on each rank.
  - For ESMC-300M the reported unused parameter indices included `302 303 304 305 306 307`; for ESMC-600M they included `362 363 364 365 366 367`.
- Interpretation: this was not an ESMC weight/tokenizer loading issue. It was a DDP graph issue caused by some encoder-mode parameters not participating in the loss for some mixed-task / mixed-view batches.
- Fixed `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py`:
  - Added `should_find_unused_parameters(args)`.
  - Encoder mode now enables DDP `find_unused_parameters=True` automatically.
  - The CLI flag `--find-unused-parameters` remains available and explicit.
- Updated encoder formal YAMLs to pass `--find-unused-parameters` explicitly:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc300m_stage1.yml`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc600m_stage1.yml`
- Added tests in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_ddp_training.py`:
  - encoder mode enables DDP unused-parameter detection by default.
  - no-encoder mode keeps it disabled by default.
- Verified:
  - `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_ddp_training.py`
  - `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_ddp_training.py -q`: `3 passed`
  - Both encoder YAML files parse and contain `--find-unused-parameters`.
- Resubmitted the two formal encoder jobs with the no-proxy Volc wrapper:
  - ESMC-300M: `t-20260615015007-kw5jx`
  - ESMC-600M: `t-20260615015007-g99bk`
- Initial status after resubmission:
  - ESMC-300M `t-20260615015007-kw5jx`: `Queue`
  - ESMC-600M `t-20260615015007-g99bk`: `Queue`
  - No worker instances exist yet for these new queued tasks, so there are no new logs yet.

## 2026-06-14 Current formal training status

- Latest Volc status check:
  - no-encoder `t-20260614223319-hzqdr`: `Success`.
  - ESMC-300M encoder `t-20260615015007-kw5jx`: `Queue`.
  - ESMC-600M encoder `t-20260615015007-g99bk`: `Queue`.
- no-encoder completed the planned `50000` stage-1 steps.
  - Tail logs reached step `49980` and then saved final checkpoint.
  - Recent loss examples: step `49820` loss `2.5908`, step `49900` loss `0.5640`, step `49980` loss `2.0503`.
  - Recent throughput stayed around `1589` to `1622` samples/s.
  - Final checkpoint: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_no_encoder_stage1/final.pt`
  - Latest checkpoint: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_no_encoder_stage1/latest.pt`
  - Both checkpoint files are about `434M`.
  - Offline wandb run: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_no_encoder_stage1/wandb/wandb/offline-run-20260614_143508-qt02xjok`
- The resubmitted ESMC encoder jobs still have no worker instances, so there are no new logs after the DDP `find_unused_parameters` fix yet.

## 2026-06-14 Fixed-resource resubmission

- User requested switching training jobs from idle/preemptible resources to fixed resources.
- Updated the formal job YAMLs from `Preemptible: true` to `Preemptible: false`:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_stage1.yml`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc300m_stage1.yml`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc600m_stage1.yml`
- Verified all three YAML files parse and have `Preemptible=False`.
- Canceled the queued idle-resource encoder tasks:
  - ESMC-300M `t-20260615015007-kw5jx`: `Killed`.
  - ESMC-600M `t-20260615015007-g99bk`: `Killed`.
- Submitted fixed-resource encoder tasks with the no-proxy Volc wrapper:
  - ESMC-300M: `t-20260615015635-rm62r`.
  - ESMC-600M: `t-20260615015634-crkld`.
- Initial fixed-resource status:
  - ESMC-300M `t-20260615015635-rm62r`: `Queue`.
  - ESMC-600M `t-20260615015634-crkld`: `Queue`.

## 2026-06-14 Reverted encoder submissions to idle/preemptible resources

- User requested changing the two encoder submissions back to idle/preemptible resources.
- Canceled the queued fixed-resource encoder tasks:
  - ESMC-300M `t-20260615015635-rm62r`: `Killed`.
  - ESMC-600M `t-20260615015634-crkld`: `Killed`.
- Updated only the two encoder formal YAMLs from `Preemptible: false` back to `Preemptible: true`:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc300m_stage1.yml`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc600m_stage1.yml`
- Left `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_stage1.yml` as `Preemptible: false`; the no-encoder stage-1 training has already completed successfully and was not resubmitted.
- Verified both encoder YAMLs parse, have `Preemptible=True`, and still contain `--find-unused-parameters`.
- Submitted the two idle/preemptible encoder jobs with the no-proxy Volc wrapper:
  - ESMC-300M: `t-20260615020032-5gg96`.
  - ESMC-600M: `t-20260615020032-s787s`.
- Initial status after idle-resource resubmission:
  - ESMC-300M `t-20260615020032-5gg96`: `Queue`.
  - ESMC-600M `t-20260615020032-s787s`: `Queue`.

## 2026-06-14 BioSeq foundation naming cleanup

- Standardized public documentation wording from the old Qwen-derived labels to `BioSeq foundation` or `BioSeq foundation-model`.
- Kept compatibility identifiers unchanged for paths, task names, output directories, wandb project names, and test/script filenames: `qwen3_vl_arch`, `qwen3_vl_bioseq_*`, and `bioseq-qwen3-vl`.
- Updated visible descriptions in the formal training YAMLs, README files, project guide, model plan, data-format audit, training script docstring, and data-inspection CLI help text.
- Current no-encoder stage-1 config uses per-GPU microbatch `--batch-size 8`, `--grad-accum 2`, `world_size=8`, so the optimizer effective batch is `8 * 2 * 8 = 128`.

## 2026-06-14 No-encoder architecture wording correction

- Corrected inaccurate autoregressive wording for `BioSeqNoEncoderDiffusionModel`.
- The `no_encoder` model has no ESM/ESMC encoder, but its internal BioSeq diffusion stack uses bidirectional self-attention over the noised token stream. It should be described as `no-encoder` or `encoder-free bidirectional diffusion transformer`, not as a causal/autoregressive architecture.

## 2026-06-14 Encoder feature-conditioning correction

- Corrected the ESMC/ESM encoder path semantics after review: ESMC is used to extract token-level features from the current diffusion state `x_t`, not to pool clean sequence features into a chain-level condition.
- The BioSeq denoiser remains responsible for multi-chain denoising over the concatenated token stream, matching the no-encoder path's chain-aware input layout.
- Removed the old pooled conditioning path from `BioSeqEncoderDiffusionModel`; the encoder output is now gathered back to decoder residue positions as per-token features before the multi-chain denoiser runs.
- Added denoiser support for direct `diffusion_state` input. Discrete states use embedding lookup; floating states use weighted embedding projection with `state @ embedding_weight`, avoiding internal one-hot expansion.
- The already-running ESMC encoder jobs were launched before this correction and should be treated as old-implementation runs unless they are explicitly restarted from the updated code.

## 2026-06-14 Restarted encoder training with token-feature implementation

- User requested stopping the old encoder runs and retraining with the corrected feature-conditioning implementation.
- Canceled the old running encoder tasks:
  - ESMC-300M old task `t-20260615020032-5gg96`: `Killed`.
  - ESMC-600M old task `t-20260615020032-s787s`: `Killed`.
- Updated the two encoder YAMLs to avoid resuming old flawed checkpoints:
  - `TaskName` changed to `qwen3_vl_bioseq_esmc300m_feat_v2_stage1` and `qwen3_vl_bioseq_esmc600m_feat_v2_stage1`.
  - `--resume none` is now explicit.
  - Output and offline wandb directories now use `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_esmc300m_feat_v2_stage1` and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_esmc600m_feat_v2_stage1`.
- Submitted the corrected v2 feature-conditioned training tasks with the no-proxy Volc wrapper:
  - ESMC-300M v2: `t-20260615042843-kkxcm`, state `Running`, launched `2026-06-14 20:29:04 UTC`.
  - ESMC-600M v2: `t-20260615042843-cmkpb`, state `Running`, launched `2026-06-14 20:29:01 UTC`.
- Platform task details confirm both new tasks include `--resume none`, new v2 output directories, and the corrected shared project code path.
- Both v2 jobs reached step `0` and started logging:
  - ESMC-300M v2: `loss=230.2500`, `effective_batch=128`, `samples/s=1049.8`, `mem_peak=8.4GB/79.2GB`.
  - ESMC-600M v2: `loss=200.8462`, `effective_batch=128`, `samples/s=698.5`, `mem_peak=13.9GB/79.2GB`.
- The preflight tokenizer log prints `tokenizer ok 33 32` because local ESMC `tokenizer.json` exposes 33 emitted tokens with `<mask>` id `32`, while ESMC model configs still declare `vocab_size=64`; the training command keeps `--vocab-size 64` so the denoiser logits stay aligned with the ESMC head size.

## 2026-06-15 Submitted BioSeq OAS+OTS 8GPU comparison run

- User requested direct training of the `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq` version with OAS + OTS mixed data only, using the native BioSeq dataloader format, on one 8-GPU node, and with wandb project aligned to the `qwen3_vl_arch` comparison runs.
- Added source selection to `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_bioseq_ddp.py` via `--sources`, preserving the old default `oas,ots,nanobody` while allowing this run to pass `--sources oas,ots`.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/bioseq_oas_ots_8gpu_stage1.yml`.
  - Resources: `RoleReplicas: 1`, `Flavor: ml.pni2.28xlarge`, i.e. 1 node x 8 GPU.
  - Data: `--sources oas,ots`, `--limit-per-source 300000`, default BioSeq OAS/OTS processed CSV dirs under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data`.
  - Model: BioSeq/Ophiuchus-Ab trainer initialized from `/vepfs-mlp2/c20250601/251105016/model_weights/ophiuchus_ab/Ophiuchus-Ab.ckpt`.
  - Wandb: `--wandb-project bioseq-qwen3-vl`, `--wandb-run-name bioseq_oas_ots_8gpu_stage1`.
- Verification before submit:
  - `python -m py_compile examples/bioseq/train_bioseq_ddp.py`
  - `python -m pytest scripts/tests/bioseq/test_dynamic_training.py -q`: `6 passed`
  - YAML parsed successfully with `TaskName=bioseq_oas_ots_8gpu_stage1`, `RoleReplicas=1`, `Flavor=ml.pni2.28xlarge`.
- Submitted with `/root/.codex/skills/volc-no-proxy/scripts/volc-no-proxy.sh ml_task submit --conf train_jobs/bioseq_oas_ots_8gpu_stage1.yml`.
  - Task id: `t-20260615102316-jhtdq`
  - Status at `2026-06-15 02:24:52 UTC`: `Queue`

## 2026-06-18 Ophiuchus-Ab generation downstream smoke and CDR metric check

- User requested checking `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream` Ophiuchus-Ab generation tasks against the Ophiuchus-Ab paper, prioritizing CDR infilling, humanization, and heavy-to-light generation.
- Paper mapping:
  - CDR infilling metric: amino-acid recovery (AAR), including SAb23H2 and SAbDab.
  - Light-chain pairing metric: ImmunoMatch score and generated-vs-reference comparison.
  - Humanization metric: OASis score and AbNatiV score.
- Fixed downstream generation compatibility:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/common.py`: `CdrInfillCollator` now masks light-chain CDRs for `cdrl*` modes instead of always masking heavy-chain positions.
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/infill/zeroshot_cdr.py`: dataset now passes the CDR mode to the collator; defaults restored to AirGen/paper-style `--sampling-strategy argmax --max-iter 4`.
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/infill/zeroshot_sab23h2.py`: maps `cdrh*`/`cdrl*` mode names to local SAb23H2 directories `h_cdr*`/`l_cdr*`; defaults restored to `argmax`, `max_iter=4`.
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/humanization/humanize.py`: supports `.pdb` as well as `.cif` inputs and writes `variant_idx`, `generated_heavy`, `generated_light` columns compatible with `oasis_human_score.py`.
- Verification:
  - `python -m py_compile downstream/common.py downstream/infill/zeroshot_cdr.py downstream/infill/zeroshot_sab23h2.py downstream/humanization/humanize.py downstream/comp_chain/generate_light_from_csv.py`
  - `python -m pytest scripts/tests/bioseq/test_downstream_common.py -q`: `5 passed`
- Ran Ophiuchus-Ab SAb23H2 CDR infilling on A100 using `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt`, `argmax`, `max_iter=4`, `temperature=1.0`, `cfg_scale=0.0`.
  - Output log: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/sab23h2_cdr_infill.log`
  - Metrics JSON: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/sab23h2_cdr_infill_metrics.json`
  - AAR (%): `cdrh3=34.50`, `cdrh2=67.88`, `cdrh1=74.52`, `cdrl3=73.08`, `cdrl2=79.41`, `cdrl1=81.09`.
- Ran heavy-to-light generation smoke on 3 OAS holdout pairs constructed from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/oas_previous_clean/splits/compat_for_current_loader_oasrule/holdout.csv`.
  - Input: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/comp_chain_oas_holdout_smoke_input.csv`
  - Output: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/comp_chain_oas_holdout_smoke_n1.csv`
  - Generated light chains are non-empty, lengths `106-111`, simple reference AAR around `87.0%`.
- Ran humanization generation smoke on two SAb23H2 native PDBs using generated chain-pair CSV.
  - Output: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/humanization_smoke.csv`
  - Smoke AAR against original FR/CDR labels: heavy `96.15%`, light `97.06%`.
- Current blockers for exact paper-score reproduction beyond CDR AAR:
  - Local `data/downstream/comp_chain` is missing the real `test_data_oas_holdout.csv` and only contains `._test_data_oas_holdout.csv`.
  - ImmunoMatch local checkpoints expected by `downstream/comp_chain/eval_scripts/immunomatch_score.py` are absent: `/vepfs-mlp2/mlp-public/zhuyiheng/hub/checkpoints/immunomatch-kappa` and `immunomatch-lambda`.
  - `biophi`, the OASis DB, and `abnativ` are absent, so OASis/AbNatiV humanization scoring cannot run locally.
  - The paper's 27-murine-antibody humanization benchmark data is not present under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/humanization`.

## 2026-06-15 Resumed v2 ESMC encoder training after resource reclaim

- Checked the corrected token-feature ESMC v2 jobs and confirmed there were no active `qwen3_vl_bioseq_esmc*` tasks in the non-terminal task list.
- The prior v2 tasks had been stopped by platform resource reclaim:
  - ESMC-300M v2 `t-20260615042843-kkxcm`: `Killed`, ended `2026-06-15 07:08:02 UTC`.
  - ESMC-600M v2 `t-20260615042843-cmkpb`: `Killed`, ended `2026-06-15 06:58:43 UTC`.
- Local logs/checkpoints before resubmission:
  - ESMC-300M v2 latest log reached `step=32200`, `loss=0.2862`; latest resumable checkpoint is `latest.pt` at saved step `32000`.
  - ESMC-600M v2 latest log reached `step=15920`, `loss=0.7797`; latest resumable checkpoint is `latest.pt` at saved step `15000`.
- Added separate resume configs rather than changing the scratch configs:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc300m_feat_v2_resume.yml`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc600m_feat_v2_resume.yml`
- Both resume configs keep the same v2 output directories and use `--resume auto`, so the trainer loads `<output-dir>/latest.pt` with model, optimizer, and step state before continuing to `--max-steps 50000`.
- Submitted the resume jobs with the no-proxy Volc wrapper:
  - ESMC-300M resume: `t-20260616015357-4m7bs`, initial status `Queue`, start timestamp `2026-06-15 17:53:57 UTC`.
  - ESMC-600M resume: `t-20260616015357-g96bf`, initial status `Queue`, start timestamp `2026-06-15 17:53:57 UTC`.

## 2026-06-16 Fixed corrupted-token logging normalization

- User noticed that encoder runs reported much smaller `corrupted=` values than the no-encoder run.
- Audited the diffusion path and confirmed no-encoder and encoder training both call the same `sample_bioseq_diffusion_noise(...)`; the actual corruption sampling probability is not lower in the encoder model.
- Root cause was the logging denominator: `corrupted=` previously used only `output.corruption_mask.sum()` from rank 0's final local micro-batch at the current optimizer step. Because the memory-tuned configs use different local micro-batch sizes, the raw logged value was not comparable:
  - no-encoder: `batch-size 8`, `grad-accum 2`, `world_size 8`
  - ESMC-300M: `batch-size 2`, `grad-accum 8`, `world_size 8`
  - ESMC-600M: `batch-size 1`, `grad-accum 16`, `world_size 8`
- On the existing logs, the old per-rank micro-batch averages scale back to similar effective-step counts once multiplied by `grad_accum * world_size`: no-encoder about `12.7k`, ESMC-300M about `12.7k`, and ESMC-600M about `13.9k` corrupted tokens per optimizer step.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py` so future logs aggregate corrupted and eligible token counts across local gradient accumulation and all DDP ranks before reporting:
  - `corrupted=` is now the average global corrupted-token count per optimizer step over the logging window.
  - Logs now also include `eligible=` and `corrupt_rate=`.
  - Wandb now receives `train/corrupted_tokens`, `train/eligible_tokens`, `train/corruption_rate`, and raw window totals.
  - `perf/samples_per_sec` now uses the actual number of optimizer steps in the elapsed window instead of always multiplying by `log_interval`, fixing the first-log throughput overestimate.
- Verification:
  - `python -m py_compile examples/bioseq/train_qwen3_vl_bioseq_ddp.py`
  - `python -m pytest scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py scripts/tests/bioseq/test_qwen3_vl_ddp_training.py -q`: `10 passed`
- Current task/checkpoint status after the earlier resume jobs:
  - ESMC-300M resume `t-20260616015357-4m7bs`: `Success`; `latest.pt` and `final.pt` both report `step=50000`.
  - ESMC-600M resume `t-20260616015357-g96bf`: `Killed` by resource reclaim; `latest.pt` reports `step=27000`.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc600m_feat_v2_resume2.yml` and submitted it with the no-proxy Volc wrapper.
  - ESMC-600M resume2: `t-20260616193220-7sq2k`, initial status `Queue`, start timestamp `2026-06-16 11:32:21 UTC`.

## 2026-06-16 Restarted comparable runs with per-GPU batch size 8

- User clarified that comparability should require each GPU's local micro-batch to be at least `8`, not only the optimizer effective batch being equal.
- Confirmed the previous completed/resume encoder configs did not satisfy that local-batch rule:
  - no-encoder: `--batch-size 8 --grad-accum 2`
  - ESMC-300M v2: `--batch-size 2 --grad-accum 8`
  - ESMC-600M v2: `--batch-size 1 --grad-accum 16`
- Canceled the queued old 600M resume2 task because it still used the old `--batch-size 1` local-batch setting:
  - `t-20260616193220-7sq2k`: `cancel success`; platform state later showed `Killed`.
- Added clean from-scratch batch8 configs with new output/wandb directories and `--resume none`:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_batch8_stage1.yml`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc300m_feat_v2_batch8_stage1.yml`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc600m_feat_v2_batch8_stage1.yml`
- All three configs now use `--batch-size 8 --grad-accum 2` on `1 x ml.pni2.28xlarge` (`8` GPUs), so the per-GPU local batch is `8` and the optimizer effective batch is `8 * 2 * 8 = 128`.
- Submitted the three batch8-per-GPU jobs with the no-proxy Volc wrapper:
  - no-encoder batch8: `t-20260616222740-mgkmd`, initial status `Queue`, timestamp `2026-06-16 14:27:40 UTC`.
  - ESMC-300M batch8: `t-20260616222739-crf8b`, initial status `Queue`, timestamp `2026-06-16 14:27:40 UTC`.
  - ESMC-600M batch8: `t-20260616222740-gpgks`, initial status `Queue`, timestamp `2026-06-16 14:27:40 UTC`.
- Risk note: ESMC-600M with trainable encoder and local batch `8` may exceed 80GB GPU memory. The first running logs will be used to confirm whether it fits; if it OOMs, this confirms the requested local-batch constraint is above the current 600M memory envelope.

## 2026-06-16 Added validation loss logging

- User requested validation loss in addition to training loss.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py`:
  - Added `--val-interval` default `1000`, `--val-batches` default `20`, and `--val-split` default `valid`.
  - Added a validation loader using the same source mixture and collator on the validation split. CSV sources use `valid.csv`; `processed_v2` maps `valid` to `val.jsonl`.
  - Validation runs in `eval()` + `torch.no_grad()` and uses bf16 autocast when training uses `--bf16`.
  - Logs now print `val_loss`, `val_corrupted`, `val_eligible`, and `val_corrupt_rate`.
  - Wandb receives `val/loss`, `val/corrupted_tokens`, `val/eligible_tokens`, `val/corruption_rate`, and `val/batches`.
- Updated the three batch8-per-GPU YAMLs to explicitly include `--val-interval 1000 --val-batches 20 --val-split valid`.
- The three batch8 jobs were still `Queue` after this script update, so they will pick up validation-loss logging when they start:
  - no-encoder batch8 `t-20260616222740-mgkmd`: `Queue`
  - ESMC-300M batch8 `t-20260616222739-crf8b`: `Queue`
  - ESMC-600M batch8 `t-20260616222740-gpgks`: `Queue`
- Verification:
  - `python -m py_compile examples/bioseq/train_qwen3_vl_bioseq_ddp.py`
  - `python -m pytest scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py scripts/tests/bioseq/test_qwen3_vl_ddp_training.py -q`: `10 passed`
  - CPU smoke with `--val-interval 1 --val-batches 1` printed `step=1 val_loss=13.5535 ...`.

## 2026-06-16 Audited train/validation loss semantics

- User asked to check the training loss from first principles after questioning whether the validation loader really existed.
- Confirmed validation source files exist:
  - OAS/OTS/nanobody CSV sources have `valid.csv`.
  - `processed_v2` has `val.jsonl`, and `default_source_configs(split="valid")` maps that source to `val`.
- Direct loader diagnostic:
  - train loader sample: shape `[2, 766]`, sources `ppi, ppi`, loss-mask tokens `1468`.
  - validation loader sample: shape `[2, 125]`, sources `nanobody, vdjdb`, loss-mask tokens `22`.
  - OAS-only validation diagnostic produced `sources=['oas_paired', 'oas_paired']` with nonzero `diffusion_loss_mask`.
- Loss definition after audit:
  - `sample_bioseq_diffusion_noise(...)` samples timesteps per sequence and masks only `diffusion_loss_mask` residues; fixed context chains stay visible.
  - labels are original `input_ids` only on corrupted positions and `-100` elsewhere.
  - `compute_masked_cross_entropy(..., loss_norm="token")` computes cross-entropy only on corrupted target tokens and divides by the number of corrupted tokens.
  - Encoder and no-encoder both use this same diffusion loss; encoder additionally applies the same corruption state to the per-chain ESMC input so the encoder sees `x_t`, not clean targets.
- Found and fixed another logging issue: `train/loss` had still been the rank-0 final micro-batch loss, not a global DDP/grad-accum loss. This was analogous to the earlier corrupted-token logging issue.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py`:
  - Added `loss_logging_denominator(...)`.
  - `train/loss` now aggregates loss numerator and denominator across all local gradient-accumulation micro-batches and all DDP ranks before logging.
  - With the default `loss_norm="token"`, train and validation losses are now both global corrupted-token-weighted cross entropy values.
  - Added `train/loss_denominator` and `val/loss_denominator` to wandb so the effective denominator is visible.
- Added a regression test in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_ddp_training.py` proving `build_validation_loader(...)` reads the valid split and produces a nonempty `diffusion_loss_mask`.
- Verification:
  - `python -m py_compile examples/bioseq/train_qwen3_vl_bioseq_ddp.py`
  - `python -m pytest scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py scripts/tests/bioseq/test_qwen3_vl_ddp_training.py -q`: `11 passed`
  - Direct train/val loss diagnostic on CPU:
    - train: `loss=13.2893`, denominator `3`, corrupted `3`, eligible `16`
    - val: `loss=14.4457`, denominator `33`, corrupted `33`, eligible `86`
- The three batch8 jobs were still queued after this fix, so they should use the corrected train/val loss logging when launched:
  - no-encoder batch8 `t-20260616222740-mgkmd`: `Queue`
  - ESMC-300M batch8 `t-20260616222739-crf8b`: `Queue`
  - ESMC-600M batch8 `t-20260616222740-gpgks`: `Queue`

## 2026-06-16 Simplified foundation objective to full denoise

- User clarified that mixed conditional views are too complex for the foundation model. The default foundation objective should be simple: all eligible generated residues use diffusion loss.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/view_sampler.py`: `BioSeqViewSampler` now defaults to `full_denoise_probability=1.0`.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py`: the foundation DDP trainer now constructs `BioSeqViewSampler(allowed_views=["full_denoise"])` directly and logs `training_views=full_denoise`.
- Kept `--full-denoise-probability` accepted as a hidden compatibility argument, but it is ignored by the DDP foundation training loader so the ablation view path is not reachable from training.
- Removed `--full-denoise-probability 1.0` from the three batch8-per-GPU YAMLs and updated their descriptions to say mixed-task full-denoise batches.
- Semantics after this change:
  - Foundation train/validation batches default to `view_names=full_denoise`.
  - `full_denoise` targets all eligible non-context chains/residues.
  - Antigen, peptide, MHC, HLA-like, and epitope context chains remain visible and do not receive diffusion loss unless a future explicit target policy changes that.
  - Conditional views remain available in `BioSeqViewSampler` for separate experiments, but the DDP foundation trainer does not route through them.
- Verified a local loader diagnostic after the change:
  - `training_views=full_denoise`
  - train batch `view_names=['full_denoise', 'full_denoise']`, loss-mask tokens `462`
  - validation batch `view_names=['full_denoise', 'full_denoise']`, loss-mask tokens `456`
- Verified the hidden compatibility argument is ignored by the foundation trainer:
  - command intentionally passed `--full-denoise-probability 0.0`
  - parsed value was `0.0`, but train and validation loader batches still had `view_names=['full_denoise', 'full_denoise']`
- Checked the three submitted batch8 tasks with the no-proxy Volc wrapper after the code change. All still have `State=Queue` and empty `LaunchTime`, so they have not started with the old objective:
  - no-encoder batch8 `t-20260616222740-mgkmd`: `Queue`
  - ESMC-300M batch8 `t-20260616222739-crf8b`: `Queue`
  - ESMC-600M batch8 `t-20260616222740-gpgks`: `Queue`

## 2026-06-17 Submitted encoder batch8 retry jobs

- User asked to submit resume/retry jobs after the encoder batch8 jobs were killed.
- Checked killed task runtimes from platform fields:
  - ESMC-300M `t-20260616222739-crf8b`: `2026-06-16 20:48:15 UTC` to `2026-06-16 21:01:29 UTC`, 13m14s.
  - ESMC-600M `t-20260616222740-gpgks`: `2026-06-16 21:05:52 UTC` to `2026-06-16 21:07:32 UTC`, 1m40s.
- Checkpoint audit:
  - 300M output had `latest.pt.tmp` but no `latest.pt`.
  - `torch.load(output/qwen3_vl_bioseq_esmc300m_feat_v2_batch8_stage1/latest.pt.tmp)` failed with `PytorchStreamReader failed reading zip archive: failed finding central directory`, so it is not a valid resume checkpoint.
  - 600M output had no checkpoint.
- Added retry YAMLs:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc300m_feat_v2_batch8_retry1.yml`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc600m_feat_v2_batch8_retry1.yml`
- Both retry YAMLs keep per-GPU local `--batch-size 8`, `--grad-accum 2`, full-denoise training through the current DDP trainer, validation loss logging, and `--resume auto` against the same batch8 output dirs. Because no valid `latest.pt` exists right now, they will start from step 0 unless a valid checkpoint appears before launch.
- Submitted with the no-proxy Volc wrapper:
  - ESMC-300M batch8 retry1: `t-20260617122359-shsjg`, submitted `2026-06-17 04:23:59 UTC`, status after submit `Running`, launch `2026-06-17 04:24:17 UTC`.
  - ESMC-600M batch8 retry1: `t-20260617122410-2v6ms`, submitted `2026-06-17 04:24:10 UTC`, status after submit `Queue`.
- The no-encoder batch8 task `t-20260616222740-mgkmd` remained `Queue` with empty `LaunchTime`.

## 2026-06-17 Resubmitted encoder batch8 retry jobs

- User asked to resubmit the two killed encoder retry jobs.
- Before resubmission, confirmed both output dirs now have valid-looking formal checkpoints:
  - `output/qwen3_vl_bioseq_esmc300m_feat_v2_batch8_stage1/latest.pt`, mtime `2026-06-17 05:40:05 UTC`, size `4448777981` bytes.
  - `output/qwen3_vl_bioseq_esmc600m_feat_v2_batch8_stage1/latest.pt`, mtime `2026-06-17 05:00:04 UTC`, size `7351174209` bytes.
- Reused the existing batch8 retry YAMLs with `--resume auto`, so the new jobs should resume from those `latest.pt` checkpoints:
  - 300M YAML: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc300m_feat_v2_batch8_retry1.yml`
  - 600M YAML: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc600m_feat_v2_batch8_retry1.yml`
- Submitted with the no-proxy Volc wrapper:
  - ESMC-300M resubmission: `t-20260617161809-44ml9`, created `2026-06-17 08:18:10 UTC`, status after submit `Queue`.
  - ESMC-600M resubmission: `t-20260617161817-4g7cv`, created `2026-06-17 08:18:18 UTC`, status after submit `Queue`.
- The no-encoder batch8 task `t-20260616222740-mgkmd` remained `Queue` with empty `LaunchTime`.

## 2026-06-18 Resubmitted killed encoder batch8 jobs and synced wandb

- User asked to continue the killed encoder jobs and upload their wandb logs.
- Latest killed retry1 resubmissions were resource-reclaimed:
  - ESMC-300M `t-20260617161809-44ml9`: killed after running from `2026-06-17 10:15:48 UTC` to `2026-06-17 13:00:35 UTC`; last observed train loss `1.0501`, last observed val loss `0.9832`.
  - ESMC-600M `t-20260617161817-4g7cv`: killed after running from `2026-06-17 10:17:06 UTC` to `2026-06-17 12:48:03 UTC`; last observed train loss `1.0564`, last observed val loss `0.9457`.
- Before resubmission, confirmed formal checkpoints exist:
  - `output/qwen3_vl_bioseq_esmc300m_feat_v2_batch8_stage1/latest.pt`, mtime `2026-06-17 12:48 UTC`, size `4448782269` bytes.
  - `output/qwen3_vl_bioseq_esmc600m_feat_v2_batch8_stage1/latest.pt`, mtime `2026-06-17 12:39 UTC`, size `7351179137` bytes.
- Added retry2 YAMLs with the same output dirs, per-GPU local `--batch-size 8`, `--grad-accum 2`, validation loss logging, and `--resume auto`:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc300m_feat_v2_batch8_retry2.yml`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc600m_feat_v2_batch8_retry2.yml`
- Submitted with the no-proxy Volc wrapper:
  - ESMC-300M retry2: `t-20260618164444-xthm7`, created `2026-06-18 08:44:45 UTC`, current status after submit `Queue`.
  - ESMC-600M retry2: `t-20260618164444-vvpz7`, created `2026-06-18 08:44:45 UTC`, current status after submit `Queue`.
- Ran `wandb sync` for the existing offline encoder runs under both batch8 output dirs:
  - 300M runs: `https://wandb.ai/codema/bioseq-qwen3-vl/runs/xn7pqut2`, `https://wandb.ai/codema/bioseq-qwen3-vl/runs/ihvc9u81`, `https://wandb.ai/codema/bioseq-qwen3-vl/runs/pjcrfxbf`
  - 600M runs: `https://wandb.ai/codema/bioseq-qwen3-vl/runs/prglmdi5`, `https://wandb.ai/codema/bioseq-qwen3-vl/runs/1ghvcopr`, `https://wandb.ai/codema/bioseq-qwen3-vl/runs/1n5561jl`
- `wandb sync` exited successfully and uploaded config/summary/output logs; populated runs reported W&B/GCS HTTP 403 only for `wandb-metadata.json`. The earliest 600M offline run `prglmdi5` had a 7-byte run file and effectively contains no training history.

## 2026-06-18 Added flat validation metric aliases for wandb

- User pointed out that encoder runs appeared to have no `val_loss` in wandb.
- Confirmed the trainer already writes validation to wandb under slash-style keys such as `val/loss`, while stdout prints `val_loss=...`; early killed runs before step 1000 had no validation because `--val-interval 1000`.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py` so validation metrics now include both styles:
  - existing keys: `val/loss`, `val/loss_denominator`, `val/corrupted_tokens`, `val/eligible_tokens`, `val/corruption_rate`, `val/batches`
  - flat aliases: `val_loss`, `val_loss_denominator`, `val_corrupted_tokens`, `val_eligible_tokens`, `val_corruption_rate`, `val_batches`
- Added a unit test in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_ddp_training.py` to assert the slash-style keys and flat aliases are both present and equal.
- Verification:
  - `python -m py_compile examples/bioseq/train_qwen3_vl_bioseq_ddp.py scripts/tests/bioseq/test_qwen3_vl_ddp_training.py`
  - `PYTHONPATH=$PWD pytest scripts/tests/bioseq/test_qwen3_vl_ddp_training.py -q` -> `7 passed`
- The retry2 encoder tasks were still `Queue` after the code change, so they should pick up the alias fix when they launch:
  - ESMC-300M retry2 `t-20260618164444-xthm7`: `Queue`
  - ESMC-600M retry2 `t-20260618164444-vvpz7`: `Queue`

## 2026-06-18 Submitted no-encoder transformer size sweep from scratch

- User requested deleting previous training logs and restarting three no-encoder transformer trainings from scratch to test whether larger model capacity improves loss descent.
- Deleted old no-encoder wandb log directories while preserving old checkpoints for traceability:
  - `output/qwen3_vl_bioseq_no_encoder_stage1/wandb`
  - `output/qwen3_vl_bioseq_no_encoder_batch8_stage1/wandb`
- Added three from-scratch no-encoder batch8 YAMLs. All use the same data, objective, batch, optimizer, validation, and checkpoint schedule:
  - data: `--sources oas,ots,nanobody,processed_v2`
  - objective: current foundation full-denoise path
  - per-GPU local batch: `--batch-size 8`
  - effective batch: `8 GPUs * batch 8 * grad_accum 2 = 128`
  - training: `--max-steps 50000 --lr 1e-4 --warmup-steps 1000 --bf16`
  - validation: `--val-interval 1000 --val-batches 20 --val-split valid`
  - clean start: `--resume none`
  - offline wandb under each output directory
- Size sweep:
  - small: hidden `384`, layers `6`, heads `6`, FFN `1536`, about `16.99M` parameters
  - base: hidden `512`, layers `8`, heads `8`, FFN `2048`, about `37.86M` parameters
  - large: hidden `768`, layers `12`, heads `12`, FFN `3072`, about `121.28M` parameters
- YAMLs:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_small_batch8_fromscratch.yml`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_base_batch8_fromscratch.yml`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_large_batch8_fromscratch.yml`
- Submitted with the no-proxy Volc wrapper:
  - small: `t-20260618171357-mzjxx`, created `2026-06-18 09:13:58 UTC`, status after submit `Queue`
  - base: `t-20260618171357-hvdv2`, created `2026-06-18 09:13:58 UTC`, status after submit `Queue`
  - large: `t-20260618171358-lcn59`, created `2026-06-18 09:13:58 UTC`, status after submit `Queue`
- Verification:
  - YAML parsing succeeded for all three configs.
  - `python -m py_compile examples/bioseq/train_qwen3_vl_bioseq_ddp.py`
  - Parameter counts were computed by instantiating `BioSeqNoEncoderDiffusionModel` for each size locally.
- Note: repository `.gitignore` ignores `*.yml`, so these job YAML files are present locally but do not appear in `git status`.

## 2026-06-18 Prepared no-encoder 300M/600M/1B size sweep

- User clarified that the previous no-encoder size sweep was too small and should instead target a 300M / 600M / 1B style no-encoder transformer comparison.
- Confirmed previous sweep sizes were only about `16.99M`, `37.86M`, and `121.28M` parameters, so they are not comparable to ESMC-300M/600M/1B scale.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`:
  - `BioSeqSelfAttention` now uses `torch.nn.functional.scaled_dot_product_attention`.
  - `BioSeqDiffusionTransformerConfig` now has `gradient_checkpointing`.
  - `BioSeqDiffusionDecoder` checkpoints transformer blocks during training when `gradient_checkpointing=True`.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py` to expose `--gradient-checkpointing`.
- Added a model test covering no-encoder gradient-checkpointed backward in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py`.
- Prepared three from-scratch no-encoder YAMLs. All keep per-GPU local `--batch-size 8`, `--grad-accum 2`, full-denoise objective, validation loss logging, offline wandb, `--resume none`, SDPA, and `--gradient-checkpointing`:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_300m_batch8_fromscratch.yml`
    - hidden `1024`, layers `18`, heads `16`, FFN `4096`, about `314.88M` parameters
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_600m_batch8_fromscratch.yml`
    - hidden `1536`, layers `16`, heads `24`, FFN `6144`, about `629.60M` parameters
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_1b_batch8_fromscratch.yml`
    - hidden `1792`, layers `20`, heads `28`, FFN `7168`, about `1.06B` parameters
- Verification:
  - YAML parsing succeeded for all three configs.
  - `python -m py_compile dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py examples/bioseq/train_qwen3_vl_bioseq_ddp.py scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py scripts/tests/bioseq/test_qwen3_vl_ddp_training.py`
  - `PYTHONPATH=$PWD pytest scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py scripts/tests/bioseq/test_qwen3_vl_ddp_training.py -q` -> `15 passed`
- Did not submit the new 300M/600M/1B YAMLs yet because the previous 17M/38M/121M no-encoder sweep tasks are still queued. Replacing them requires explicit user confirmation to cancel:
  - `t-20260618171357-mzjxx`
  - `t-20260618171357-hvdv2`
  - `t-20260618171358-lcn59`

## 2026-06-18 Cancelled small sweep and submitted no-encoder 300M/600M/1B

- User confirmed cancelling the previous 17M/38M/121M no-encoder size sweep and submitting the new 300M/600M/1B no-encoder runs.
- Cancelled previous queued tasks with the no-proxy Volc wrapper:
  - small `t-20260618171357-mzjxx`: final status `Killed`, finish `2026-06-18 11:32:41 UTC`
  - base `t-20260618171357-hvdv2`: final status `Killed`, finish `2026-06-18 11:32:42 UTC`
  - large `t-20260618171358-lcn59`: final status `Killed`, finish `2026-06-18 11:32:42 UTC`
- Submitted the new from-scratch no-encoder size sweep:
  - 300M `t-20260618193252-cdj8b`: `qwen3_vl_bioseq_no_encoder_300m_batch8_fromscratch`, status `Queue`
  - 600M `t-20260618193252-fwj49`: `qwen3_vl_bioseq_no_encoder_600m_batch8_fromscratch`, status `Queue`
  - 1B `t-20260618193252-qc7xw`: `qwen3_vl_bioseq_no_encoder_1b_batch8_fromscratch`, status `Queue`
- All three new runs use `--resume none`, per-GPU local `--batch-size 8`, `--grad-accum 2`, `--gradient-checkpointing`, SDPA attention, validation loss logging, and clean per-run output/wandb directories.

## 2026-06-18 Ophiuchus-Ab generation downstream metrics

- User asked to validate Ophiuchus-Ab generation-stage downstream metrics for CDR infill, humanization, and generate-light/light-chain pairing.
- Fixed downstream generation/eval plumbing:
  - `downstream/common.py`: CDR infill collator now masks light-chain CDRs when dataset rows carry `cdrl*` mode metadata.
  - `downstream/infill/zeroshot_cdr.py` and `downstream/infill/zeroshot_sab23h2.py`: return/pass mode metadata and use paper-style deterministic CDR defaults (`argmax`, `max_iter=4`).
  - `downstream/humanization/humanize.py`: supports `.pdb` and `.cif` inputs and writes `variant_idx`, `generated_heavy`, `generated_light`.
  - `downstream/comp_chain/generate_light_from_csv.py`: preserves input metadata (`raw_light_type`, source row), adds `variant_idx`, and supports `--heavy-batch-size`.
  - `downstream/comp_chain/eval_scripts/immunomatch_score.py`: correctly normalizes K/L chain-type columns, falls back to abnumber when `gen_light_type` is absent, and maps ImmunoMatch scores by original row id instead of repeated heavy-chain sequence.
- Installed/downloaded metric dependencies:
  - `datasets`, `rjieba`, `accelerate` for ImmunoMatch/Transformers evaluation.
  - `polyleven` for the light-chain property/Wasserstein evaluator.
  - ImmunoMatch kappa/lambda checkpoints under `/vepfs-mlp2/mlp-public/zhuyiheng/hub/checkpoints/`.
  - `promb` for OASis Identity without the 22GB BioPhi OASis DB.
  - AbNatiV VH/VKappa/VLambda checkpoints under `/vepfs-mlp2/c20250601/251105016/model_weights/abnativ/pretrained_models/`, scored through `.venv_abnativ` on CPU because the package CUDA path produced a device mismatch.
- CDR infill outputs:
  - SAb23H2: `output/downstream_generation/ophiuchus_ab/sab23h2_cdr_infill_metrics.json`
    - L1 `81.09`, L2 `79.41`, L3 `73.08`, H1 `74.52`, H2 `67.88`, H3 `34.50` AAR percent.
  - SAbDab external indices: `output/downstream_generation/ophiuchus_ab/sabdab_cdr_infill_metrics.json`
    - H1 `74.64`, H2 `68.56`, H3 `39.25`, L1 `73.99`, L2 `82.61`, L3 `73.01` AAR percent.
- Light-chain pairing outputs:
  - Exact paper `data/downstream/comp_chain/test_data_oas_holdout.csv` was not present; only `._test_data_oas_holdout.csv` existed.
  - Built a reconstructed OAS paired input from local raw OAS data: `output/downstream_generation/ophiuchus_ab/comp_chain_oas_holdout_500_input.csv`.
  - Confirmed against `/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/downstream/comp_chain/generate_light_from_csv.py` that the original downstream script uses `argmax`, `max_iter=4`, `cfg_scale=0`, and keeps CLS plus the first three light-chain amino acids fixed (`start_idx = 4`).
  - Updated migrated `downstream/comp_chain/generate_light_from_csv.py` default `--light-prompt-tokens` from `4` to `3` to match the original downstream script and the paper table's "initial 3-residue prompting" setting.
  - Generated CSV: `output/downstream_generation/ophiuchus_ab/comp_chain_oas_holdout_500_prompt3_fast_n8.csv`.
  - Eval CSV: `output/downstream_generation/ophiuchus_ab/comp_chain_oas_holdout_500_prompt3_fast_n8_eval.csv`.
  - Property eval JSON: `output/downstream_generation/ophiuchus_ab/comp_chain_oas_holdout_500_prompt3_fast_n8_property_eval.json`.
  - Key metrics on 4000 generated sequences: validity `1.0`, Gen ImmunoMatch `0.3879`, Ref ImmunoMatch `0.5686`, Gen>Ref `0.356`, chain match `1.0`, V gene `0.384`, J gene `0.380`, V family `0.948`, J family `0.380`, diversity near `0.0`, W property average Wasserstein distance `0.0727`.
  - Added the matching no-prompt run on the same reconstructed 500-row input with `--light-prompt-tokens 0`, `argmax`, `max_iter=4`, `cfg_scale=0.0`, `num_seqs=8`, `heavy_batch_size=16`.
  - No-prompt generated CSV: `output/downstream_generation/ophiuchus_ab/comp_chain_oas_holdout_500_noprompt_fast_n8.csv`.
  - No-prompt eval CSV: `output/downstream_generation/ophiuchus_ab/comp_chain_oas_holdout_500_noprompt_fast_n8_eval.csv`.
  - No-prompt property eval JSON: `output/downstream_generation/ophiuchus_ab/comp_chain_oas_holdout_500_noprompt_fast_n8_property_eval.json`.
  - No-prompt key metrics on 4000 generated sequences: validity `1.0`, Gen ImmunoMatch `0.4857`, Ref ImmunoMatch `0.5686`, Gen>Ref `0.492`, chain match `0.620`, V gene `0.038`, J gene `0.166`, V family `0.272`, J family `0.166`, diversity near `0.0`, W property average Wasserstein distance `0.1398`.
  - Earlier prompt4/50-row smoke outputs are kept under `output/downstream_generation/ophiuchus_ab/comp_chain_oas_holdout_500_fast_n8*` and `output/downstream_generation/ophiuchus_ab/comp_chain_oas_holdout_50_fast_n8*`.
- Humanization outputs:
  - Exact paper 27 murine SAbDab benchmark could not be fetched from SAbDab during this run; local SAb23H2 native PDBs yielded 26 paired structures parseable by the existing loader.
  - Generated with `argmax`, `max_iter=4`, `n_sequences=8`, `cfg_scale=0.0`.
  - Generated CSV: `output/downstream_generation/ophiuchus_ab/humanization_sab23h2_26_fast.csv`.
  - OASis CSV: `output/downstream_generation/ophiuchus_ab/humanization_sab23h2_26_fast_oasis.csv`.
  - AbNatiV outputs: `output/downstream_generation/ophiuchus_ab/abnativ_humanization_sab23h2_26_fast/`.
  - Key metrics: heavy FR AAR `87.63%`, light FR AAR `88.50%`, OASis Identity overall `0.8613` (VH `0.8454`, VL `0.8771`), AbNatiV VH `0.8889`, combined light AbNatiV `0.9373` over 200/208 light sequences scored.
- Added a reproducible humanization benchmark reconstruction script:
  - Script: `scripts/downstream/rebuild_igcraft_humanization_split.py`.
  - It filters local SAbDab summary rows to paired murine antibodies in `2024-02-01` through `2025-03-25`, resolves RCSB FASTA chain IDs with author-chain priority, deduplicates by VH/VL variable sequences, and writes pair-specific PDB copies so multiple H/L pairs from one PDB are not collapsed by the existing loader.
  - Rebuild output: `output/downstream_generation/ophiuchus_ab/igcraft_rebuild/`.
  - Parsed `305/305` candidate rows, deduped to `131` unique pairs, and selected `27` known-example-augmented pairs containing `8TFH`, `8TXU`, and `8TVH`.
  - Selected set has `21` antigen-bound and `6` unbound pairs; IgCraft reports `20/7`, so this is a paper-aligned reconstruction, not a confirmed exact split.
- Ran humanization fast evaluation on the reconstructed 27-pair set:
  - Generated CSV: `output/downstream_generation/ophiuchus_ab/humanization_igcraft_rebuild_27_fast.csv`.
  - OASis CSV: `output/downstream_generation/ophiuchus_ab/humanization_igcraft_rebuild_27_fast_oasis.csv`.
  - AbNatiV outputs: `output/downstream_generation/ophiuchus_ab/abnativ_humanization_igcraft_rebuild_27_fast/`.
  - Key metrics on 216 generated paired sequences: heavy FR AAR `88.37%`; light FR AAR `92.89%` from generation stdout and `91.45%` when recomputed from decoded CSV; OASis Identity overall `0.6319` (VH `0.6114`, VL `0.6523`); AbNatiV VH `0.7477`; combined light AbNatiV `0.7169`.
- Added shard controls for slower paper-style generation:
  - `downstream/comp_chain/generate_light_from_csv.py` now supports `--start-index` and `--end-index`.
  - `downstream/humanization/humanize.py` now supports `--start-index` and `--end-index`.
- Unified machine-readable summary written to `output/downstream_generation/ophiuchus_ab/generation_metrics_summary.json`.
- Human-readable report written to `output/downstream_generation/ophiuchus_ab/generation_metrics_report.md`.
- Added paper consistency validation:
  - Script: `scripts/downstream/validate_ophiuchus_generation_against_paper.py`.
  - JSON: `output/downstream_generation/ophiuchus_ab/paper_consistency_check.json`.
  - Markdown: `output/downstream_generation/ophiuchus_ab/paper_consistency_check.md`.
  - Verdicts: CDR `mostly_consistent` (`8/9` within tolerance; SAbDab H3 is low by `4.30` percentage points), prompt3 light-chain pairing `not_reproduced` (`4/10`), no-prompt light-chain pairing `not_reproduced` (`5/10`), humanization `not_reproduced` (`2/3`, with OASis/Identity `63.19` vs paper `83.4`).
- Verification:
  - `LD_LIBRARY_PATH=/vepfs-mlp2/c20250601/251105016/conda/envs/flow/lib:${LD_LIBRARY_PATH:-} python -m py_compile downstream/common.py downstream/infill/zeroshot_cdr.py downstream/infill/zeroshot_sab23h2.py downstream/humanization/humanize.py downstream/comp_chain/generate_light_from_csv.py downstream/comp_chain/eval_scripts/immunomatch_score.py downstream/comp_chain/eval_scripts/generation_eval.py scripts/downstream/rebuild_igcraft_humanization_split.py`
  - `LD_LIBRARY_PATH=/vepfs-mlp2/c20250601/251105016/conda/envs/flow/lib:${LD_LIBRARY_PATH:-} python -m pytest scripts/tests/bioseq/test_downstream_common.py -q` -> `5 passed`

## 2026-06-19 Exact downstream split correction and humanization metric validation

- User clarified the exact light-chain holdout is local:
  - `data/oas_previous_clean/splits/holdout_select_500_eval_OAS.csv`
  - Symlinked as `data/downstream/comp_chain/test_data_oas_holdout.csv`.
  - Verified 500 holdout rows, K=316 and L=184.
- User supplied the exact 27-pair humanisation split:
  - `/vepfs-mlp2/c20250601/251105016/project/.whalent_tmp/2026-06-19/exz26lrcqa6xebsf4s58inavc-humanisation.zip`
  - Extracted to `data/downstream/humanization/humanisation/`.
  - `test_chains.csv` has 28 rows because PDB `8onk` has two chain-pair rows; the current loader keys by PDB id, so one `8onk` pair is evaluated and the parsed dataset has 27 structures.
- Re-ran exact OAS holdout light-chain pairing with the same fast deterministic downstream config (`argmax`, `max_iter=4`, `num_seqs=8`, `cfg_scale=0.0`):
  - Prompt3 outputs: `output/downstream_generation/ophiuchus_ab/comp_chain_test_oas_holdout_500_prompt3_fast_n8*`.
  - Prompt3 metrics: ImmunoMatch `0.3235`, Better `23.6%`, chain match `99.8%`, V exact `0.356`, J exact `0.380`, V family `0.898`, J family `0.380`, diversity near `0`, valid `100%`, W property `0.0967`.
  - No-prompt outputs: `output/downstream_generation/ophiuchus_ab/comp_chain_test_oas_holdout_500_noprompt_fast_n8*`.
  - No-prompt metrics: ImmunoMatch `0.4536`, Better `30.4%`, chain match `62.0%`, V exact `0.038`, J exact `0.182`, V family `0.236`, J family `0.182`, diversity near `0`, valid `100%`, W property `0.1181`.
- Re-ran exact uploaded humanisation split:
  - Generated CSV: `output/downstream_generation/ophiuchus_ab/humanization_exact27_fast.csv`.
  - Direct FR native recovery: heavy `86.72%`, light `90.50%`.
  - AbNatiV outputs: `output/downstream_generation/ophiuchus_ab/abnativ_humanization_exact27_fast/`.
  - AbNatiV means: VH `0.7217`, combined light `0.6967`.
- Validated the humanization OASis metric issue:
  - The previous local `*_oasis.csv` artifact was not official BioPhi OASis; it was a local sequence-identity approximation.
  - Official BioPhi humanness scoring was run through the public BioPhi endpoint with IMGT numbering/CDRs and the default relaxed OASis threshold.
  - Combined BioPhi summary: `output/downstream_generation/ophiuchus_ab/biophi_public_exact27_relaxed/humanization_exact27_fast_biophi_oasis_relaxed_summary.csv`.
  - Official exact-split BioPhi metrics: OASis Identity `55.32%`, Heavy OASis Identity `52.82%`, Light OASis Identity `58.00%`, Heavy Germline Content `68.51%`, Light Germline Content `73.26%`.
- Updated final reports:
  - `output/downstream_generation/ophiuchus_ab/generation_metrics_summary.json`
  - `output/downstream_generation/ophiuchus_ab/humanization_exact27_fast_metrics.json`
  - `output/downstream_generation/ophiuchus_ab/paper_consistency_check.json`
  - `output/downstream_generation/ophiuchus_ab/paper_consistency_check.md`
  - `output/downstream_generation/ophiuchus_ab/generation_metrics_report.md`
- Final exact-split paper consistency verdict:
  - CDR infill remains mostly consistent: `8/9`.
  - Light-chain pairing is not reproduced on the exact OAS holdout: prompt3 `4/10`, no-prompt `4/10`.
  - Humanization is not reproduced on the uploaded exact split: `2/3`; OASis is `55.32` vs paper `83.4`, while BioPhi germline-content values are close to the paper VH/VL sequence-identity values.

## 2026-06-21 BioSeq no-encoder training code reading guide and 300M/600M/1B loss debug

- User asked for a reading guide for `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py` and to debug why the no-encoder 300M/600M/1B runs had loss problems.
- Recommended code reading order:
  1. `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py` (`main()`, `build_loader()`, `compute_training_output()`)
  2. `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/README.md`
  3. `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/sources.py` -> `records.py` -> `mixture.py` -> `view_sampler.py` -> `collator.py`
  4. `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py` (`sample_bioseq_diffusion_noise`, decoder forward, loss)
- Foundation training pipeline summary:
  - `source loader -> BioSeqRecord -> WeightedMixtureDataset -> BioSeqViewSampler(full_denoise) -> BioSeqQwenDataCollator -> sample_bioseq_diffusion_noise -> BioSeqNoEncoderDiffusionModel / BioSeqEncoderDiffusionModel -> masked CE loss`
  - DDP does not use `DistributedSampler`; iterable sources shard by `rank * num_workers + worker_id`.
  - Foundation trainer hard-codes `allowed_views=["full_denoise"]`; antigen/MHC/peptide remain fixed context and do not receive diffusion loss.
- Audited completed no-encoder size-sweep logs under:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_no_encoder_300m_batch8_fromscratch`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_no_encoder_600m_batch8_fromscratch`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_no_encoder_1b_batch8_fromscratch`
- Observed outcomes:
  - 300M: trained normally until about step `18000`, then collapsed at step `18760` from train loss `~7.96` to `~95`; val loss jumped from `~3.29` at step `18000` to `~90.84` at step `19000`; final train loss stayed around `~83`.
  - 600M: stable final train loss `~5.7`, best val loss `~2.74` at step `6000`, final val loss `~3.12`; checkpoint is usable.
  - 1B: late degradation from train loss `~4` to `~20` by step `49600`, then `NaN` from step `49780`; `latest.pt` contains `191` non-finite tensors and must not be used.
- Root cause for the broken 300M checkpoint:
  - Loaded `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_no_encoder_300m_batch8_fromscratch/latest.pt` and evaluated on a real OAS-style batch.
  - The diverged model predicts token id `32` (`<mask>`) on essentially all corrupted positions (`661/661` in the local check), with near-zero entropy and eval loss `~90.48`.
  - The healthy 600M checkpoint on the same batch predicts amino-acid tokens with eval loss `~2.94`.
  - Conclusion: this is a masked-diffusion mode-collapse failure, not a data-loader bug. Corrupted decoder inputs already contain `<mask>`, and tied input/output embeddings (`lm_head.weight = token_embeddings.weight`) make `<mask>` an easy attractor unless its logit is excluded from the denoising objective.
- Additional training-config issue found in the actual submitted runs:
  - WandB config for the 300M job shows `--lr 1e-4` and no `--lr-scheduler cosine`, even though the current YAMLs on disk specify scaled LRs (`5e-5` / `3e-5` / `2e-5`) plus cosine decay.
  - Constant `1e-4` likely contributed to the 1B late-stage instability, but it does not explain the sudden 300M `<mask>` collapse by itself because 600M survived with the same submitted LR.
- Code fix applied in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`:
  - Added `forbidden_diffusion_target_token_ids()` and `mask_forbidden_target_logits()`.
  - `compute_masked_cross_entropy()` now masks logits for `<cls>`, `<pad>`, `<eos>`, `<unk>`, and `<mask>` before CE.
  - Trainer and model `compute_loss()` paths now pass the forbidden-id set from config.
  - Added regression test `test_compute_masked_cross_entropy_forbids_mask_token_predictions` in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py`.
- Recommended rerun plan:
  - Do not resume from the broken 300M or 1B checkpoints.
  - Keep the 600M checkpoint as the only usable no-encoder large run from this sweep.
  - Resubmit 300M/1B from scratch with the forbidden-logit fix, explicit `--lr-scheduler cosine`, and the size-scaled LRs already written in the YAMLs.
  - Optionally save `best.pt` by validation loss instead of only `latest.pt`.

## 2026-06-21 Cancelled non-preemptible grammar_v1 tasks and resubmitted on idle (preemptible) resources

- Context: the six `grammar_v1` jobs submitted 2026-06-20 22:15-22:16 (UTC+8) were `Preemptible: false` and had been stuck in `Queue` for ~12.5h without getting fixed resources. User asked to cancel them and resubmit on idle/preemptible resources, and to keep this log updated after every submission.
- Scope (confirmed with user): only the six queued `grammar_v1` jobs. The running job `qwen3_vl_bioseq_esmc300m_feat_v2_batch8_fromscratch` (`t-20260620015806-wrj9n`, Running ~40.8h) was left untouched.
- Edited `Preemptible: false` -> `Preemptible: true` in six YAMLs under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/`:
  - `qwen3_vl_bioseq_grammar_v1_esmc300m.yml`, `qwen3_vl_bioseq_grammar_v1_esmc600m.yml`, `qwen3_vl_bioseq_grammar_v1_no_encoder_38m.yml`, `qwen3_vl_bioseq_grammar_v1_no_encoder_300m.yml`, `qwen3_vl_bioseq_grammar_v1_no_encoder_600m.yml`, `qwen3_vl_bioseq_grammar_v1_no_encoder_1b.yml`
- Cancelled the six previous non-preemptible queued tasks (`volc ml_task cancel -i <id>` -> `cancel success`):
  - esmc300m `t-20260621061542-dcgmp`
  - esmc600m `t-20260621061542-8fkq2`
  - no_encoder_38m `t-20260621061542-gswsj`
  - no_encoder_300m `t-20260621061602-scbwj`
  - no_encoder_600m `t-20260621061542-q4997`
  - no_encoder_1b `t-20260621061543-2qvml`
- Resubmitted on idle/preemptible resources (`volc ml_task submit --conf <yaml>` -> `创建任务成功`), initial status `Queue` (verified via `volc ml_task list ... -o json`):
  - esmc300m `t-20260621185805-78pr8`
  - esmc600m `t-20260621185808-qsrfm`
  - no_encoder_38m `t-20260621185811-2b477`
  - no_encoder_300m `t-20260621185814-gdxf8`
  - no_encoder_600m `t-20260621185818-dsv9p`
  - no_encoder_1b `t-20260621185821-kftpd`
- Volc CLI notes for this environment (no no-proxy/SparkMCP wrapper used this time):
  - `volc ml_task list` and `volc ml_task get` open an interactive TUI and hang in non-interactive shells even when redirected to a file; always pass `-o json`.
  - `--limit` must be `<= 300` (`--limit 500` -> `A parameter specified in the request is not valid: Limit`).
  - This CLI has no `kill` verb; cancel with `volc ml_task cancel -i <id>`.
  - `-n grammar_v1` returned 0 matches while `-n bioseq` matched the same jobs; use `-n bioseq` plus a local `grammar_v1` JobName filter, and include `--status Initialized,Queue,Staging,Running,Killing` to see freshly submitted jobs.

## 2026-06-21 Cancelled esmc300m_feat_v2_batch8_fromscratch

- User requested cancelling the long-running non-preemptible job `qwen3_vl_bioseq_esmc300m_feat_v2_batch8_fromscratch`.
- Cancelled `t-20260620015806-wrj9n` via `volc ml_task cancel -i t-20260620015806-wrj9n` -> `cancel success` (was Running ~40.8h).
- Config: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_esmc300m_feat_v2_batch8_fromscratch.yml` (`Preemptible: false`).
- Removed from `## Active Volc Training Tasks` (terminal state); only the six preemptible `grammar_v1` jobs remain active.
- Added Cursor rule `/vepfs-mlp2/c20250601/251105016/project/dllm_test/.cursor/rules/volc-train-task-log.mdc`: after every submit/cancel, update this file; delete task IDs from the Active table when training reaches a terminal state.

## 2026-06-21 Full no-encoder stability debug + fix + resubmit (38M/300M/600M/1B, no-grammar)

- User asked for a complete debug of the no-encoder loss failure (look at gradients / numerics, not just a guess), then resubmit the three no-encoder no-grammar runs plus a default-parameter ~38M run. Confirmed the 38M default config (`hidden 512 / 8 layers / 8 heads / FFN 2048`) is exactly the downstream Transformer size used inside the ESMC encoder runs (`37.9M` params).
- Gradient / numerical evidence gathered:
  - Per-tensor weight stats on `latest.pt`: 300M and 600M have finite weights with `max_abs ~4.85`; 1B `latest.pt` has `191` non-finite tensors (already known NaN at step ~49780).
  - Per-layer residual-stream RMS via forward hooks on a real OAS batch: broken 300M residual RMS rises smoothly `711 -> 1223` and `final_layernorm` renormalizes it (out RMS `~0.93`); healthy 600M residual RMS is actually far larger (`18k -> 306k`) yet still trains fine. So the failure is NOT exploding activations/weights.
  - Prediction analysis: broken 300M predicts `<mask>` (id 32) on `661/661` corrupted positions, entropy `~0.06`, eval loss `~90.48`; healthy 600M predicts amino acids, eval loss `~2.94`.
- Local reproduction with `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/diagnose_no_encoder_stability.py` (new):
  - `mid` (159M) at constant `lr=5e-4` for 1000 steps and even extreme `lr=2e-3` for 500 steps stayed healthy (loss `~2.8-2.9`, grad_norm `~1-2`, `raw_pred_mask_frac=0`). The per-step update is stable; the real-run failure is a mid/late-training stochastic spike that a constant LR cannot recover from, after which the model falls into the `<mask>` attractor.
  - Root cause (confirmed, not guessed): masked-diffusion mode collapse. Corrupted decoder inputs already contain `<mask>`, input/output embeddings are tied (`lm_head.weight = token_embeddings.weight`), so `<mask>` is a self-reinforcing fixed point unless its logit is removed from the objective. At step 0 the fresh model already predicts `<mask>` on `~33%` of corrupted positions.
- Fixes implemented:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`:
    - `forbidden_diffusion_target_token_ids()` + `mask_forbidden_target_logits()`; `compute_masked_cross_entropy()` masks `<cls>/<pad>/<eos>/<unk>/<mask>` logits before CE (already wired into trainer and both `compute_loss` paths). This removes the `<mask>` attractor at the source.
    - Added optional per-head query/key RMSNorm (`qk_norm` config flag + `BioSeqSelfAttention.q_norm/k_norm`) for attention-logit stability against the stochastic spike. Default off, so existing tests/behavior are unchanged.
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py`: added `--qk-norm` and passes `qk_norm` into the config.
- Fix verification:
  - `forbidden-mask=1 + qk-norm=1 + cosine` (mid, lr 5e-4, 400 steps): loss `4.3 -> 2.4`, grad_norm stable `~1-2`, `raw_pred_mask_frac=0` throughout, cosine LR decays correctly.
  - End-to-end trainer smoke (`protenix_abtcr`, single GPU, `--qk-norm`): step 0->2 loss `4.24 -> 3.61`, grad_norm `9.8 -> 5.3`, checkpoint saved.
  - `python -m pytest scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py` -> `11 passed` (includes the new forbidden-mask regression test).
- YAML updates (added `--qk-norm`; cosine + size-scaled LRs already present). These are the NO-grammar no-encoder configs the user asked for:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_38m_batch8_fromscratch.yml` (hidden 512 / 8 / 8 / 2048, lr 1e-4)
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_300m_batch8_fromscratch.yml` (hidden 1024 / 18 / 16 / 4096, lr 5e-5)
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_600m_batch8_fromscratch.yml` (hidden 1536 / 16 / 24 / 6144, lr 3e-5)
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_1b_batch8_fromscratch.yml` (hidden 1792 / 20 / 28 / 7168, lr 2e-5)
- Submitted from scratch (`--resume none`) with the no-proxy Volc wrapper (`创建任务成功`, verified via `ml_task get -o json`):
  - 38M `t-20260621195439-d2fz4`: `Running`
  - 300M `t-20260621195443-lckq8`: `Queue`
  - 600M `t-20260621195454-btl6t`: `Initialized`
  - 1B `t-20260621195457-jhlwx`: `Initialized`
- Did not resume from the broken 300M/1B checkpoints; all four start fresh with the forbidden-logit fix, qk-norm, cosine schedule, and size-scaled LRs.

## 2026-06-21 Scaling fix: switch no-encoder 300M/600M/1B from wide-shallow to ESMC-style depth-first

- User flagged that the way we were growing the no-encoder models looked wrong and asked how ESM scales. Verified official configs:
  - ESM-2: 8M(6L,320) -> 35M(12L,480) -> 150M(30L,640) -> 650M(33L,1280) -> 3B(36L,2560) -> 15B(48L,5120). Depth-first; #heads held at 20 with head_dim growing 16->128; FFN strictly 4x (GELU MLP).
  - ESMC (this project's encoder family; Pre-LN, RoPE, SwiGLU, no biases): 300M(30L,960,15h) -> 600M(36L,1152,18h) -> 6B(80L,2560,40h). head_dim fixed 64. SwiGLU expansion ~8/3 (~2.67x) so FFN param count matches a standard 4x GELU MLP.
- Audited our real param counts by instantiating `BioSeqNoEncoderDiffusionModel` (`/tmp/param_audit.py`, env `protenix_abtcr`):
  - 38M current: L8 H512 h8 FFN2048(4.0x) -> 37.9M
  - 300M current: L18 H1024 h16 FFN4096(4.0x) -> 314.8M
  - 600M current: L16 H1536 h24 FFN6144(4.0x) -> 629.5M
  - 1B current: L20 H1792 h28 FFN7168(4.0x) -> 1061.1M
- Problems identified (with data):
  - Layer count was non-monotonic / regressed: 300M=18L but 600M=**16L** (fewer layers despite 2x params); 1B only 20L.
  - Far too wide-and-shallow vs ESM/ESMC: at ~300M ESMC uses 30L (we used 18L); at ~600M ESMC uses 36L (we used 16L). Depth ~half of ESMC.
  - FFN at full 4x with SwiGLU (3 matrices) makes per-layer FFN = 12 H^2 = 3x the attention (4 H^2), so the param budget was spent on width instead of depth. ESMC's ~2.67x SwiGLU frees budget for more layers.
  - head_dim=64 was already correct; kept.
  - Note: our model uses learned absolute position embeddings (`max_position_embeddings=4096`), whereas ESM-2/ESMC use RoPE. User chose to defer the RoPE change to a separate task.
- Fix applied (same param budget, depth-first, head_dim=64, FFN ~2.67x; verified via `/tmp/param_audit.py`):
  - 300M: L18 H1024 h16 FFN4096 -> **L28 H960 h15 FFN2560** (321.2M)
  - 600M: L16 H1536 h24 FFN6144 -> **L38 H1152 h18 FFN3072** (620.8M)
  - 1B: L20 H1792 h28 FFN7168 -> **L52 H1280 h20 FFN3456** (1049.6M)
  - 38M control left unchanged (it intentionally mirrors the downstream Transformer inside the ESMC encoder).
  - Edited `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_{300m,600m,1b}_batch8_fromscratch.yml` (hidden/layers/heads/intermediate); kept qk-norm, cosine, gradient-checkpointing, and the existing LRs (5e-5 / 3e-5 / 2e-5).
- Cancelled the previous wide-shallow submissions (`volc ml_task cancel` -> `cancel success`): 300M `t-20260621195443-lckq8`, 600M `t-20260621195454-btl6t`, 1B `t-20260621195457-jhlwx`. The 38M job `t-20260621195439-d2fz4` (Running) was left untouched.
- Resubmitted the depth-first configs (`创建任务成功`, all `Initialized` via `ml_task get -o json`):
- Follow-up (not done this round, per user): swap learned absolute position embeddings for RoPE to fully match the ESM/ESMC architecture.

## 2026-06-21 Synced no-encoder 38M offline wandb run

- User requested uploading the completed 38M training log to W&B.
- Volc job: `qwen3_vl_bioseq_no_encoder_38m_batch8_fromscratch`, `task_id=t-20260621195439-d2fz4`, final status `Success`.
- Offline run dir: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_no_encoder_38m_batch8_fromscratch/wandb/wandb/offline-run-20260621_115539-1nxe3is3` (`run-1nxe3is3.wandb`, ~6.5 MB).
- Sync command (env `protenix_abtcr`, `WANDB_API_KEY` from `/vepfs-mlp2/c20250601/251105016/.secrets/wandb_api_key`):
  - `wandb sync .../offline-run-20260621_115539-1nxe3is3` -> `done`
- W&B URL: `https://wandb.ai/codema/bioseq-qwen3-vl/runs/1nxe3is3`
- Removed `t-20260621195439-d2fz4` from `## Active Volc Training Tasks` (terminal state).

## 2026-06-21 Cancelled legacy no-encoder batch8 jobs (grammar-v2 migration)

- Cancelled three legacy wide-shallow / no-grammar training jobs (`volc ml_task cancel` -> `cancel success`):
  - `t-20260621201405-zq8wz` — `qwen3_vl_bioseq_no_encoder_300m_batch8_fromscratch` (was `Running`)
  - `t-20260621201409-bzvg6` — `qwen3_vl_bioseq_no_encoder_600m_batch8_fromscratch` (was `Queue`)
  - `t-20260621201412-d6fnq` — `qwen3_vl_bioseq_no_encoder_1b_batch8_fromscratch` (was `Queue`)
- YAML paths (removed from repo in same cleanup): `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_no_encoder_{300m,600m,1b}_batch8_fromscratch.yml`
- Removed the three task IDs from `## Active Volc Training Tasks`; six grammar_v1 jobs remain queued.
## 2026-06-21 Cancelled queued grammar_v1 jobs (pre-resubmit old yaml)

- Context: grammar-v2 code and YAML updates (sqrt weights + cosine for 600m/1b) landed; pytest 26 passed. Cancelled six queued preemptible `grammar_v1` jobs that still pointed at the previous submission.
- Cancelled (`volc ml_task cancel` -> `cancel success`, prior status `Queue`):
  - `t-20260621185805-78pr8` — `qwen3_vl_bioseq_grammar_v1_esmc300m`
  - `t-20260621185808-qsrfm` — `qwen3_vl_bioseq_grammar_v1_esmc600m`
  - `t-20260621185811-2b477` — `qwen3_vl_bioseq_grammar_v1_no_encoder_38m`
  - `t-20260621185814-gdxf8` — `qwen3_vl_bioseq_grammar_v1_no_encoder_300m`
  - `t-20260621185818-dsv9p` — `qwen3_vl_bioseq_grammar_v1_no_encoder_600m`
  - `t-20260621185821-kftpd` — `qwen3_vl_bioseq_grammar_v1_no_encoder_1b`

## 2026-06-21 Resubmitted grammar_v1 with updated YAML

- Resubmitted all six `grammar_v1` configs via `volc ml_task submit --conf` (wrapper: `volc-no-proxy.sh`). `Preemptible: true` on all YAMLs. All six `创建任务成功`; initial status `Initialized` (`ml_task get -o json`).
  - `t-20260621223241-z55tl` — `qwen3_vl_bioseq_grammar_v1_esmc300m` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_esmc300m.yml`
  - `t-20260621223245-ggbzf` — `qwen3_vl_bioseq_grammar_v1_esmc600m` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_esmc600m.yml`
  - `t-20260621223248-wwpcb` — `qwen3_vl_bioseq_grammar_v1_no_encoder_38m` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_no_encoder_38m.yml`
  - `t-20260621223252-rlj8x` — `qwen3_vl_bioseq_grammar_v1_no_encoder_300m` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_no_encoder_300m.yml`
  - `t-20260621223256-w7sgz` — `qwen3_vl_bioseq_grammar_v1_no_encoder_600m` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_no_encoder_600m.yml`
  - `t-20260621223259-2cbkd` — `qwen3_vl_bioseq_grammar_v1_no_encoder_1b` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_no_encoder_1b.yml`
- Updated `## Active Volc Training Tasks` with the six new task IDs.

## 2026-06-21 Resubmitted grammar_v1_esmc300m as non-preemptible

- Cancelled preemptible encoder job: `t-20260621223241-z55tl` (`qwen3_vl_bioseq_grammar_v1_esmc300m`, was `Initialized`) via `volc ml_task cancel` -> `cancel success`.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_esmc300m.yml`: `Preemptible: false`.
- Resubmitted via `volc ml_task submit --conf` (wrapper: `volc-no-proxy.sh`).
  - `task_id=t-20260621225143-xjbn4`, initial status `Initialized`, `Preemptible: false`.
- Replaced `t-20260621223241-z55tl` with `t-20260621225143-xjbn4` in `## Active Volc Training Tasks`.

## 2026-06-21 Resubmitted grammar_v1 with polynomial warmup + cosine decay LR

- Added `--lr-scheduler polynomial` to `examples/bioseq/train_qwen3_vl_bioseq_ddp.py`: AirGen-style linear warmup from `--warmup-init-lr` (default `1e-7`) to peak LR, then cosine decay to `min_lr_ratio × base_lr` (default `0.1`). Encoder and decoder param groups share the same schedule shape on their respective `initial_lr` values.
- Updated all six `train_jobs/qwen3_vl_bioseq_grammar_v1_*.yml` with `--warmup-steps 2000 --warmup-init-lr 1e-7 --lr-scheduler polynomial --min-lr-ratio 0.1`. Decoder peak `1e-4`; encoder jobs also keep `--encoder-lr 2e-5` (decays to `2e-6`).
- Preemptible policy: `esmc300m` `Preemptible: false`; other five jobs `Preemptible: true`.
- Cancelled prior six grammar_v1 tasks (`cancel success`):
  - `t-20260621225143-xjbn4` (esmc300m, was Running)
  - `t-20260621223245-ggbzf` (esmc600m)
  - `t-20260621223248-wwpcb` (no_encoder_38m)
  - `t-20260621223252-rlj8x` (no_encoder_300m)
  - `t-20260621223256-w7sgz` (no_encoder_600m)
  - `t-20260621223259-2cbkd` (no_encoder_1b)
- Resubmitted all six via `volc-no-proxy.sh ml_task submit --conf`; initial status `Initialized`:
  - esmc300m `t-20260621231245-hs2g6` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_esmc300m.yml` (`Preemptible: false`)
  - esmc600m `t-20260621231248-vx6wn` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_esmc600m.yml` (`Preemptible: true`)
  - no_encoder_38m `t-20260621231251-wb4k6` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_no_encoder_38m.yml` (`Preemptible: true`)
  - no_encoder_300m `t-20260621231254-qn54x` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_no_encoder_300m.yml` (`Preemptible: true`)
  - no_encoder_600m `t-20260621231258-pgzcl` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_no_encoder_600m.yml` (`Preemptible: true`)
  - no_encoder_1b `t-20260621231301-j4qxb` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_no_encoder_1b.yml` (`Preemptible: true`)

## 2026-06-21 Single-GPU batch probe + grammar_v1 batch128 tuned resubmit

- Ran `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/probe_grammar_v1_batch_size.py` on local A100-80GB with real grammar data, `max_sequence_length=2112`, stress batches `seq_len≈2054`, bf16 + gradient checkpointing.
- Peak GPU memory (stress batch): no_encoder 38M/300M/600M/1B all OK through `batch_size=16` (max peak 20.1 GB); encoder ESMC-300M `batch_size=8` peak 29.6 GB, `batch_size=16` peak 57.2 GB; encoder ESMC-600M `batch_size=8` peak 42.7 GB, `batch_size=16` OOM.
- Final YAML batch settings (all `effective_batch=128` on 8 GPU):
  - esmc300m / esmc600m: `batch_size=8`, `grad_accum=2`
  - no_encoder 38m / 300m / 600m / 1b: `batch_size=16`, `grad_accum=1`
- Preemptible: esmc300m `false`; other five `true`.
- Cancelled prior six tasks (`t-20260621232429-59dd4` through `t-20260621232446-6nltq`) and resubmitted:
  - esmc300m `t-20260621233326-7tpvg`
  - esmc600m `t-20260621233329-8ddcg`
  - no_encoder_38m `t-20260621233332-9djzn`
  - no_encoder_300m `t-20260621233335-26vnf`
  - no_encoder_600m `t-20260621233339-8dtmr`
  - no_encoder_1b `t-20260621233342-4xsr4`
- Initial status: `Initialized`. Monitor script task id updated to `t-20260621233326-7tpvg`.

## 2026-06-22 esmc300m Failed: NCCL watchdog + resubmit with safer batch

- Failed task `t-20260621233326-7tpvg` (`qwen3_vl_bioseq_grammar_v1_esmc300m`): ran ~25 min, **0 training steps**, exit code 1.
- Root cause (Volc logs `ml_task logs -t ... -i worker_0`): **NOT OOM**. PyTorch **ProcessGroupNCCL watchdog** killed rank 4 after **480s** with no collective progress during the first optimizer step (`batch_size=8`, `grad_accum=2`). `NCCL_TIMEOUT_SECOND=1800` does not cover PyTorch's separate `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` (default 480).
- Fix in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_esmc300m.yml`:
  - `batch_size=4`, `grad_accum=4` (still effective_batch=128; single-GPU probe peak ~16GB on stress batch)
  - `export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`
- Resubmitted: `task_id=t-20260622083259-7kzgf`, initial status `Initialized`, `Preemptible: false`.
- Monitor script updated to pull Volc failure snippets on `Failed`/`Killed`.

## 2026-06-22 2-GPU DDP debug job for multi-card hang diagnosis

- **Not a gradient NaN issue**: failed task `t-20260621233326-7tpvg` died from NCCL watchdog (480s) before any logged step; resubmit `t-20260622083259-7kzgf` still Running ~7+ min with no `step=` yet (first encoder+DDP step may be very slow or still hung).
- Added `--debug-ddp-timing` to `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py` (per-rank logs: batch wait / forward / backward / clip_grad / optimizer_step).
- Submitted 2-GPU debug: `task_id=t-20260622084005-wx6fh` via `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/debug_qwen3_vl_bioseq_grammar_v1_esmc300m_2gpu.yml`
  - `Flavor: ml.pni2.7xlarge` (2 GPU), `Preemptible: false`, `Priority: 6`
  - Phase A: `num_workers=0`, 4 steps; Phase B: `num_workers=2` (production-like), 8 steps
  - Same encoder config as production, `log_interval=1`, wandb disabled
- Added `export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800` to the five queued grammar_v1 YAMLs (600m + four no_encoder) so resubmits inherit the fix.

## 2026-06-22 esmc300m production stopped; 8-GPU debug on 非闲时

- Local dataloader probe (`scripts/debug/probe_grammar_v1_dataloader_ddp.py`): 8-rank CPU gloo + `num_workers=0/2` — **no hang** (max batch wait ~5s). Single-GPU encoder stress batch forward ~3.8s, backward ~0.5s — **not a pure data-loader stall**.
- Cancelled stuck production esmc300m `t-20260622083259-7kzgf` and 2-GPU debug `t-20260622084005-wx6fh`; esmc300m slot repurposed for debug only (no full training until DDP root cause fixed).
- Resubmitted 8-GPU debug: `task_id=t-20260622084553-n9qsk` via `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/debug_qwen3_vl_bioseq_grammar_v1_esmc300m_8gpu.yml`
  - `Flavor: ml.pni2.28xlarge` (8 GPU), **`Preemptible: false`（非闲时）**, `Priority: 6`
  - Phase A: `num_workers=0`, 4 steps; Phase B: `num_workers=2`, 8 steps; `--debug-ddp-timing`, wandb disabled
- Monitor script retargeted to debug task/output: `scripts/monitor_grammar_v1_esmc300m.py` → `t-20260622084553-n9qsk`, output `output/debug_grammar_v1_esmc300m_8gpu`.

## 2026-06-22 esmc300m 8-GPU debug Success — root cause confirmed

- Debug task `t-20260622084553-n9qsk` finished **`Success`** in ~195s (`Preemptible: false`, `ml.pni2.28xlarge`).
- **Phase A** (`num_workers=0`, 4 steps) and **Phase B** (`num_workers=2`, 8 steps) both completed; saved `output/debug_grammar_v1_esmc300m_8gpu/phase_nw2/final.pt`.
- Per-rank timing (typical): `batch_ready` <0.2s, `forward` ~0.05–0.13s, `backward` ~0.1–0.36s, `clip_grad` ~0.01s, `optimizer_step` ~0.01s; **8 ranks stay in sync**; peak GPU mem ~21GB.
- **Root cause of original failure `t-20260621233326-7tpvg`**: NOT OOM, NOT NaN, NOT dataloader hang. First DDP collective (`clip_grad_norm_` allreduce with `find_unused_parameters=True`) under `batch_size=8 grad_accum=2` + ESMC encoder exceeded PyTorch **`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` default 480s** → NCCL watchdog SIGABRT. `NCCL_TIMEOUT_SECOND=1800` alone does not help.
- **Verified fix**: `batch_size=4 grad_accum=4` + `export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800` (already in production YAML). Production esmc300m can be resubmitted with this config.

## 2026-06-22 Resubmit five preemptible grammar_v1 jobs with NCCL fix

- **Risk assessment**: All 8-GPU DDP jobs need `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`. **Encoder** jobs (`esmc600m`) with old `batch_size=8 grad_accum=2` had the **same high risk** as failed esmc300m; **no_encoder** jobs (`bs=16 ga=1`) are lighter and unlikely to hit 480s, but old queued submits still lacked the heartbeat env var.
- Cancelled old Queue tasks (old YAML, no heartbeat / esmc600m still bs=8):
  - `t-20260621233329-8ddcg` esmc600m
  - `t-20260621233332-9djzn` no_encoder_38m
  - `t-20260621233335-26vnf` no_encoder_300m
  - `t-20260621233339-8dtmr` no_encoder_600m
  - `t-20260621233342-4xsr4` no_encoder_1b
- YAML updates before resubmit: all five now `export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`; esmc600m changed to `batch_size=4 grad_accum=4`.
- Resubmitted (`Preemptible: true`, initial `Initialized`):
  - esmc600m `t-20260622085359-qcxhs`
  - no_encoder_38m `t-20260622085402-cdlwt`
  - no_encoder_300m `t-20260622085405-gm62n`
  - no_encoder_600m `t-20260622085409-kcfsq`
  - no_encoder_1b `t-20260622085413-pqgkk`

## 2026-06-22 Submit esmc300m production (非闲时) + long-run monitor

- Submitted production esmc300m: `task_id=t-20260622085513-t4225` via `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_esmc300m.yml`
  - `Preemptible: false`, `batch_size=4 grad_accum=4`, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800`, `log_interval=20`
- Enhanced monitor: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/monitor_grammar_v1_esmc300m.py`
  - Volc log step grep, WandB step tracking, stale-step alert (default 30 min), state file `output/grammar_v1_esmc300m/monitor_state.json`
  - Log: `output/grammar_v1_esmc300m/monitor.log`; background stdout: `monitor_stdout.log`

## 2026-06-22 esmc300m cancel/resubmit: num_workers=0 after 20min stall

- Real-time poll (20×30s, ~20min): `t-20260622085513-t4225` stayed `Running` with **no `step=`**, WandB 0 points, no NCCL error yet — likely stuck on first batch with `num_workers=2` + full Arrow load.
- Cancelled `t-20260622085513-t4225`; YAML changed `--num-workers 2` → `--num-workers 0` (matches successful debug Phase A).
- Resubmitted: `task_id=t-20260622091611-vkq5v`, `Preemptible: false`. Monitor retargeted to new task id.

## 2026-06-22 esmc300m root cause: uncached load_from_disk + resubmit with cache

- **Why debug passed but production hung**: debug used `--limit-per-source 2000`; production streams full OAS/OTS shards. `GrammarArrowSource.iter_records` called `load_from_disk` on **every iterator restart** (~minutes per call on multi-million-row shards). 8 ranks × 4 sources desync on first batch → DDP allreduce stuck → NCCL watchdog (480s or 1800s).
- **Fix**: process-local Arrow cache in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/grammar.py` (`_cached_grammar_arrow_dataset`); warm restart 0.0s vs cold ~0.8s locally (was >300s without cache).
- **Also**: `WeightedMixtureDataset` skips empty shards on restart instead of crashing (`mixture.py`).
- Cancelled hung `t-20260622091611-vkq5v` (Running ~98min, 0 steps, NCCL 1800s watchdog).
- Resubmitted: `task_id=t-20260622110228-pj2hb`, `num_workers=0`, `--debug-ddp-timing`, `Preemptible: false`.

## 2026-06-22 Submit MINT MMseqs cluster on ml.c1ie.21xlarge

- Local MMseqs cluster (13 threads, step2 prefilter ~21h) stopped to avoid conflict with Volc job.
- Submitted `mint_string_mmseqs_cluster_c1ie` via `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/mint_string_mmseqs_cluster_c1ie.yml`
- `task_id=t-20260622090726-lgpsv`, initial status `Initialized`, `Preemptible: false`, flavor `ml.c1ie.21xlarge`
- Reuses existing `DB100` on vepfs; cleans `mmseqs_tmp` and reruns `mmseqs cluster --min-seq-id 0.50` with `MMSEQS_THREADS=$(nproc)`
- Output target: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/stringdb_mint/clu50.tsv`
- Volc log: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/pipeline_logs/mmseqs_cluster_volc.log`
- Local `run_mint_pipeline_after_cluster.sh` orchestrator continues polling for `clu50.tsv` on shared vepfs.

## 2026-06-22 Resubmit MINT MMseqs cluster on preemptible resources

- Cancelled non-preemptible task `t-20260622090726-lgpsv` (mint_string_mmseqs_cluster_c1ie), final status `Killed`.
- Resubmitted via same YAML with `Preemptible: true`: `task_id=t-20260622092811-r9bdb`, initial status `Initialized`, flavor `ml.c1ie.21xlarge`.

## 2026-06-22 Fix MMseqs cluster preemptible (YAML alone did not apply)

- Verified via `volc ml_task get --format Preemptible`: `t-20260622092811-r9bdb` was **`Preemptible: false`** on platform despite YAML `Preemptible: true` — task used native queue quota (组内固定资源), not idle/borrowed quota.
- Cancelled `t-20260622092811-r9bdb` (`cancel success`, final `Killed`).
- Resubmitted with explicit CLI flag: `volc ml_task submit --conf .../mint_string_mmseqs_cluster_c1ie.yml --preemptible`
  - `task_id=t-20260622094943-n9bnv`, **`Preemptible: true`** confirmed, initial status `Initialized`.
- Note: future submits should pass `--preemptible` (or verify field after submit); YAML-only `Preemptible: true` was not honored for this job.

## 2026-06-22 Resubmit MMseqs cluster after preemptible kill

- Previous preemptible run `t-20260622094943-n9bnv` reached `Killed` (~73 min, cascaded step1 align in progress; no `clu50.tsv`).
- Resubmitted: `volc ml_task submit --conf .../mint_string_mmseqs_cluster_c1ie.yml --preemptible`
  - `task_id=t-20260622112332-njxvr`, `Preemptible: true`, initial status `Initialized`.

## 2026-06-22 Repair MINT physical links + resume splits/shards pipeline

- `build_mint_string_splits.py` failed on corrupt `protein.physical.links.full.v12.0.txt.gz` (`zlib.error: invalid stored block lengths`; `gzip -t` failed).
- Moved corrupt file to `.../stringdb_mint/protein.physical.links.full.v12.0.txt.gz.corrupt_20260622`.
- Re-downloading via `aria2c` from STRING-DB (log: `data/ppi_task_raw/processed/pipeline_logs/mint_links_redownload.log`).
- Started `scripts/data/run_mint_pipeline_after_links.sh` (waits for valid gzip → `build_mint_string_splits.py` → `build_mint_grammar_shards.py` train+valid).
- `clu50.tsv` already present (59309604 lines); MMseqs Volc task `t-20260622112332-njxvr` finished `Success`.

## 2026-06-23 Fix MINT splits OOM + restart pipeline

- Re-download completed: `protein.physical.links.full.v12.0.txt.gz` size **15528028374** bytes, `gzip -t` OK.
- Second splits attempt (`2026-06-22T18:21:16Z`) was **OOM-killed** while loading all links into RAM (`build_mint_string_splits.py:387246 Killed` in `mint_pipeline_after_links_stdout.log`). Root cause: in-memory `read_links()` list, not corrupt gzip.
- Refactored `scripts/data/build_mint_string_splits.py` to **disk-backed** link materialization + index shuffle/filter (matches MINT `stringdb.py` semantics, avoids OOM).
- Hardened `scripts/data/download_stringdb_assets.sh`: verify `gzip -t` + expected byte size before skipping re-download.
- Restarted `run_mint_pipeline_after_links.sh --skip-wait` (log: `data/ppi_task_raw/processed/pipeline_logs/mint_pipeline_after_links.log`).

## 2026-06-24 Submitted MINT native stringdb splits on ml.g3a.48xlarge (preemptible)

- Cancelled local disk-backed `build_mint_string_splits.py` run (stuck at ~10M/2.36B filter after ~13h; random seek on 144GB flat file).
- Added `scripts/data/run_mint_stringdb_native.py` — in-memory port of MINT `stringdb.py` (seeds 137/731, 250k valid, cluster dedup + train/valid disjoint filter).
- Submitted `mint_stringdb_splits_g3a48xlarge` via `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/mint_stringdb_splits_g3a48xlarge.yml`
- `task_id=t-20260624025343-pbrwh`, initial status `Initialized`, `Preemptible: true`, flavor `ml.g3a.48xlarge` (192 vCPU, 768 GiB per Volc ML Platform spec docs).
- Log: `data/ppi_task_raw/processed/pipeline_logs/mint_stringdb_native_volc.log`
- Outputs target: `data/ppi_task_raw/processed/mint_string_pretrain_v1/{validation,training_filtered}.{links,seqs}.txt.gz`

## 2026-06-24 Cancelled g3a mint splits; run native MINT on dhlpw WebShell

- Cancelled `t-20260624025343-pbrwh` (`mint_stringdb_splits_g3a48xlarge`, was `Queue`) via `volc ml_task cancel` -> `cancel success`.
- Plan: run `scripts/data/run_mint_stringdb_on_node.sh` inside **WebShell** of running task `t-20260623161337-dhlpw` (`ml.pni2.28xlarge`, ~2TB RAM, vepfs mounted).
- Background log: `data/ppi_task_raw/processed/pipeline_logs/mint_stringdb_native_bg.log`

## 2026-06-22 no_encoder_38m grammar_v1 Failed (t-20260622085402-cdlwt)

- Task `qwen3_vl_bioseq_grammar_v1_no_encoder_38m` via `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_no_encoder_38m.yml`, final status **`Failed`**, `Preemptible: true`.
- Timeline: queued ~10.8h; actual train ~11min (WandB init 11:31 UTC → NCCL abort 11:42 UTC). **0 optimizer steps**, no checkpoint.
- **Root trigger**: `[rank1] FloatingPointError: non-finite training loss detected at step=0` on first forward pass.
- **Terminal failure mode**: other ranks stuck on DDP `ALLREDUCE` (SeqNum=4, 10min PyTorch default op timeout) → NCCL watchdog SIGABRT.
- **Likely contributors vs healthy esmc300m job**: YAML still uses `--num-workers 2` (esmc300m uses `0`); no `--debug-ddp-timing`; 8-rank first-batch desync amplifies NaN detection collective hang.
- Recommended resubmit: align with esmc300m fixes (`num_workers=0`, optional `bs=4 ga=4`, `--debug-ddp-timing`); investigate rank1 step-0 NaN locally if reproduces.

## 2026-06-22 no_encoder_300m / no_encoder_600m Failed (same pattern as 38m)

- **no_encoder_300m** `t-20260622085405-gm62n`: **`Failed`**, `Preemptible: true`, queued ~11h, train ~11min, **0 steps**, no checkpoint.
- **no_encoder_600m** `t-20260622085409-kcfsq`: **`Failed`**, same timeline and symptoms.
- **Root trigger (both)**: `[rank1] FloatingPointError: non-finite training loss detected at step=0`.
- **Terminal failure**: DDP `ALLREDUCE` SeqNum=8 timeout **600s** → NCCL SIGABRT (same as 38m).
- **Not data/OOM**: WandB shows model init only (`parameters=312M/626M`, `effective_batch=128`); no logged training step.
- **YAML gap vs healthy esmc300m**: `--num-workers 2`, `batch_size=16 grad_accum=1`, no `--debug-ddp-timing`.

## 2026-06-22 Resubmit no_encoder (38m/300m/600m/1b) + esmc600m with DDP fixes

- Updated all four `train_jobs/qwen3_vl_bioseq_grammar_v1_no_encoder_*.yml` and `esmc600m.yml`: `--num-workers 0`, `--batch-size 4 --grad-accum 4`, `--debug-ddp-timing` (effective batch still 128).
- Cancelled queued old submits: `t-20260622085413-pqgkk` (1b), `t-20260622085359-qcxhs` (esmc600m).
- Resubmitted (`Preemptible: true`, initial status `Initialized`):
  - no_encoder_38m `t-20260622202245-zsn5d`
  - no_encoder_300m `t-20260622202249-prgdr`
  - no_encoder_600m `t-20260622202252-jcmfd`
  - no_encoder_1b `t-20260622202256-z4p5c`
  - esmc600m `t-20260622202259-t2kdc`

## 2026-06-22 Pipeline cleanup: grammar-only data path + DDP footgun guardrails

- **双轨 data 收尾**: legacy collator/view_sampler 在上一轮已删除；本轮清掉残留引用——`scripts/debug/probe_no_encoder_step0_nan.py`（去掉 `view_seed`，按 rank 改用 `source_seed`），`scripts/tests/bioseq/test_qwen3_vl_ddp_training.py`（去掉 `full_denoise_probability` / `--full-denoise-probability` / `--max-chain-length`，`view_names` 断言改为 `grammar_v2`）。全仓 `.py` 已无 `BioSeqQwenDataCollator` / `BioSeqViewSampler` / `view_seed` / `allowed_views` 引用。
- **DDP 数据层 footgun（代码级修复）**: `examples/bioseq/train_qwen3_vl_bioseq_ddp.py`
  - `--num-workers` 默认 `2` → `0`（memory-mapped Arrow 源的安全路径；>0 时每个 worker 进程独立分片无限加权流，会让首 batch 在各 rank 间错位、触发 NCCL collective 超时——正是近期 `no_encoder` step-0 挂死的诱因之一）。
  - 分布式且 `num_workers>0` 时打印告警。
  - loader 构建后、训练循环前加 `torch.distributed.barrier()` 对齐各 rank 起步。
- **文档/code 一致性（grammar-only）**:
  - `dllm/pipelines/qwen3_vl_arch/data/README.md` 重写为 `GrammarArrowSource -> WeightedMixtureDataset -> TaskHomogeneousBatchDataset -> GrammarBioSeqCollator`，新增 `num_workers` footgun 说明，掩码键改为实际 emit 的 `diffusion_loss_mask` / `diffusion_eligible_mask` / `fixed_context_mask` 等。
  - `PROJ_GUIDE.md`、`BIOSEQ_MODEL_PLAN.md`、`dllm/pipelines/qwen3_vl_arch/README.md` 的训练路径/掩码/`--num-workers 0`/py_compile+pytest 命令同步更新；条件 view 的设计意图保留，但说明改由 grammar 的 `<fixs>...<fixd>` + 推理期 partial-mask prompt 实现，不再有运行时 view sampler。
- **验证**: `python -m pytest scripts/tests/bioseq/ -q` → 75 passed；相关脚本 `py_compile` 全过。
- **未处理（明确留作后续）**: "单体 trainer"（`train_qwen3_vl_bioseq_ddp.py` ~900 行）拆分为 data/optim/loop/checkpoint 模块属于较大重构，风险较高，未在本轮动；建议作为独立后续项，避免与本次数据层/文档修复混在一起引入回归。

## 2026-06-23 BioSeq 专用 Trainer + DataModule 抽取（行为等价重构）

- **动机**: 上一节遗留的"单体 trainer"。`examples/bioseq` 下 `train_qwen3_vl_bioseq_ddp.py` 与 `train_bioseq_ddp.py` 各自重复实现 `setup_distributed`/`is_main`/`log`/`move_batch`/`setup_wandb`/`lr_at`/`save_checkpoint`/`maybe_resume`/主循环——这是"配置漂移"的结构性根因。
- **新增 `dllm/pipelines/qwen3_vl_arch/data/datamodule.py::GrammarDataModule`**（169 行）：封装 tokenizer + `GrammarArrowSource → WeightedMixtureDataset → TaskHomogeneousBatchDataset → GrammarBioSeqCollator`，把 `num_workers=0` 安全默认与 rank×worker 分片固化在数据层。`from_args` 用 `getattr` 容忍 debug/probe 脚本的最小 namespace。已加入 `data/__init__.py` 导出。
- **新增 `dllm/pipelines/qwen3_vl_arch/training/`**（`trainer.py` 491 行）：`BioSeqTrainer` 持有与模型无关的 infra——distributed context、LR schedule、grad-accum 主循环、非有限值守卫、window 日志 + wandb、validation 编排、checkpoint/resume、启动 barrier。模型/数据语义通过 `TrainStepFns` 适配层注入（`compute_output`/`loss_denominator`/`eligible_token_count`/`token_class_metrics`/`evaluate_validation`），循环不 import 任何入口脚本。
- **`train_qwen3_vl_bioseq_ddp.py` 收薄**：927 → 474 行。保留模型语义 glue（`compute_training_output`/`evaluate_validation`/`loss_logging_denominator`/`diffusion_eligible_token_count`/`token_class_loss_metrics`）于入口模块，从而 `test_qwen3_vl_ddp_training.py` 对 `train_ddp.*` 的 monkeypatch 表面与 `build_loader`/`build_tokenizer`/`lr_at` 等导入面完全不变。`main()` 变为：parse_args → setup_distributed → GrammarDataModule → model/optimizer → BioSeqTrainer.fit。
- **边界原则**: 共享的只有 infra；ophiuchus（`train_bioseq_ddp.py`）本轮**未动**，后续可通过提供自己的 `TrainStepFns` + DataModule 复用同一 `BioSeqTrainer`。
- **验证（行为等价）**:
  - `pytest scripts/tests/bioseq/ -q` → **75 passed**（含跑通重构后 `main()` 的单进程 subprocess smoke）。
  - 2-rank CPU DDP smoke（torchrun）：init/barrier/loop/all-reduce 日志/validation/checkpoint(latest+final)/clean shutdown 全部正常，`loss/views=grammar_v2/tasks` 输出与重构前一致。
  - `--resume auto` smoke：正确加载 `latest.pt` 从 step 3 续跑。
  - probe/downstream 依赖脚本 `py_compile` 全过；无 lint 错误。

## 2026-06-23 Val-loss top-k checkpoint retention (save_top_k=10)

- Added `dllm/pipelines/qwen3_vl_arch/training/checkpointing.py::ValLossTopKCheckpointManager`.
- CLI: `--save-top-k` default **10** (`0` disables). On each validation pass, if `val/loss` enters the top-K lowest seen so far, rank0 writes a full checkpoint under `output_dir/checkpoints/step_{step}_val_{loss}.pt`, maintains `checkpoints/topk_manifest.json`, and refreshes `best.pt` as the current best alias.
- Unchanged: periodic `latest.pt` (`--save-interval`) for resume, plus `final.pt` at normal training end. Resume still defaults to `latest.pt`, not `best.pt`.
- Tests: `scripts/tests/bioseq/test_bioseq_checkpointing.py` (3 passed) + existing DDP tests green; CPU smoke with `--val-interval 2 --save-top-k 10` wrote `best.pt` + one ranked checkpoint.

## 2026-06-23 Submitted grammar_v1 downstream eval (esmc600m + no_encoder_38m)

- Submitted single-GPU (`ml.pni2.3xlarge`, preemptible) downstream eval for two trained variants (both `step=50000`):
  - **ESMC-600M encoder**: `eval_grammar_v1_esmc600m_downstream` via `train_jobs/eval_grammar_v1_esmc600m_downstream.yml`, `task_id=t-20260623152944-t757c`, initial status `Queue`, checkpoint `output/grammar_v1_esmc600m/latest.pt`.
  - **No-encoder 38M**: `eval_grammar_v1_no_encoder_38m_downstream` via `train_jobs/eval_grammar_v1_no_encoder_38m_downstream.yml`, `task_id=t-20260623152946-tfh4t`, initial status `Queue`, checkpoint `output/grammar_v1_no_encoder_38m/latest.pt`.
- Each job runs (same protocol as esmc300m baseline): SAbDab CDR-H1/H2/H3 10-fold (`argmax`, `max_iter=4`) then OAS holdout500 light pairing (`gumbel_argmax`, `max_iter=32`, `num_seqs=8`, `light_prompt_tokens=3`).
- Shared runner: `scripts/downstream/run_grammar_variant_downstream_eval.sh`; outputs under `output/downstream_generation/grammar_v1_{esmc600m,no_encoder_38m}_*`.

## 2026-06-23 Failed downstream eval jobs (esmc600m + no_encoder_38m)

- Both eval tasks reached terminal **`Failed`** (exit code 1). Volc logs via `ml_task logs -t <id> -i worker_0`.
- **esmc600m** `t-20260623152944-t757c`: CDR-H1 completed (**AAR 73.64%**, 10-fold). Failed at CDR-H2 start because **`output/downstream_generation/` was deleted** mid-run → bash redirect error `No such file or directory` (not OOM / not model error).
- **no_encoder_38m** `t-20260623152946-tfh4t`: CDR-H1/H2/H3 all completed (**67.84% / 59.61% / 35.20%**). Failed ~2.5 min into light pairing 500; stdout has no Python traceback — most likely **preemptible job killed** (`Preemptible: true`) or process SIGKILL before logs flushed.
- Current vepfs state: entire `output/downstream_generation/` directory is missing (including prior esmc300m / ophiuchus_ab artifacts); need `mkdir -p` + resubmit evals.

## 2026-06-23 Resubmitted all three grammar_v1 downstream eval jobs

- Confirmed `output/downstream_generation/` missing (no esmc300m CDR/pairing artifacts on vepfs); all three variants resubmitted on single-GPU `ml.pni2.3xlarge`, **`Preemptible: false`**.
- **esmc300m**: `eval_grammar_v1_esmc300m_downstream`, `task_id=t-20260623161354-8ssl2`, yaml `train_jobs/eval_grammar_v1_esmc300m_downstream.yml`, ckpt `output/grammar_v1_esmc300m/latest.pt`.
- **esmc600m**: `eval_grammar_v1_esmc600m_downstream`, `task_id=t-20260623161357-l9w7f`, yaml `train_jobs/eval_grammar_v1_esmc600m_downstream.yml`, ckpt `output/grammar_v1_esmc600m/latest.pt`.
- **no_encoder_38m**: `eval_grammar_v1_no_encoder_38m_downstream`, `task_id=t-20260623161400-xckdb`, yaml `train_jobs/eval_grammar_v1_no_encoder_38m_downstream.yml`, ckpt `output/grammar_v1_no_encoder_38m/latest.pt`.
- Script hardening: `run_grammar_variant_downstream_eval.sh` now supports `esmc300m` and `mkdir -p` before each CDR/pairing log write; yml entrypoints also `mkdir -p output/downstream_generation` before tee.

## 2026-06-23 Submitted grammar_v1 Qwen0.6B ablation (supervised)

- User requested two supervised ablation runs for scaling comparison:
  1. **No-encoder Qwen3-0.6B trunk** (`L28/H1024/16h/FFN3072`, `--qk-norm`, cosine LR).
  2. **ESMC-300M encoder + same ~0.6B decoder** (vs existing 38M decoder baseline).
- Decoder ~394M params in our stack (ESMC vocab=33 vs Qwen vocab=151k); architecture matches official Qwen3-0.6B trunk dims. GQA not implemented — full MHA with head_dim=64.
- Submitted (`Preemptible: true`, initial status `Initialized`):
  - `qwen3_vl_bioseq_grammar_v1_no_encoder_qwen0_6b` → `task_id=t-20260623161338-k979t`, yaml `train_jobs/qwen3_vl_bioseq_grammar_v1_no_encoder_qwen0_6b.yml`, output `output/grammar_v1_no_encoder_qwen0_6b`, batch 4 × grad_accum 4.
  - `qwen3_vl_bioseq_grammar_v1_esmc300m_decoder0_6b` → `task_id=t-20260623161337-dhlpw`, yaml `train_jobs/qwen3_vl_bioseq_grammar_v1_esmc300m_decoder0_6b.yml`, output `output/grammar_v1_esmc300m_decoder0_6b`, batch 2 × grad_accum 8 (memory headroom for encoder+large decoder).

## 2026-06-24 Resumed no_encoder_qwen0_6b from checkpoint (preemptible kill)

- Prior run `t-20260623161338-k979t` reached terminal **`Killed`** (preemptible). Checkpoints on vepfs: `best.pt` @ step 2000 (val=1.476), `latest.pt` @ step 2500.
- Updated `train_jobs/qwen3_vl_bioseq_grammar_v1_no_encoder_qwen0_6b.yml`: `--resume none` → **`--resume auto`**, same `OUTPUT_DIR=output/grammar_v1_no_encoder_qwen0_6b`.
- Resubmitted `qwen3_vl_bioseq_grammar_v1_no_encoder_qwen0_6b` → `task_id=t-20260624024912-7mv8m`, initial status `Initialized`, **`Preemptible: true`**. Training will resume from **`latest.pt` step 2500** toward `max_steps=50000`.

## 2026-06-24 Resubmitted no_encoder_qwen0_6b with explicit checkpoint resume

- Cancelled queued auto-resume job `t-20260624174108-4fsjv` (was `--resume auto`).
- Prior run `t-20260624024912-7mv8m` terminal **`Killed`** (preemptible). On-disk checkpoint: `output/grammar_v1_no_encoder_qwen0_6b/latest.pt` @ **step 14500**.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_no_encoder_qwen0_6b.yml`: set `RESUME_CKPT="${OUTPUT_DIR}/latest.pt"`, `test -f`, and **`--resume "${RESUME_CKPT}"`** (explicit path, not auto).
- Resubmitted → `task_id=t-20260624174241-cbbs2`, initial status **`Queue`**, **`Preemptible: true`**. Continues from step 14500 toward `max_steps=50000`.

## 2026-06-24 no_encoder_qwen0_6b resume Failed (t-20260624174241-cbbs2)

- Task `qwen3_vl_bioseq_grammar_v1_no_encoder_qwen0_6b` via `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v1_no_encoder_qwen0_6b.yml`, final status **`Failed`**, exit code 1, ran ~13 min.
- Volc logs (`ml_task logs -t t-20260624174241-cbbs2 -i worker_0`): **`trainer.resume()` → `load_state_dict` vocab mismatch**.
  - Checkpoint `latest.pt` @ step **14500**: `decoder.token_embeddings.weight` / `lm_head.weight` shape **`[51, 1024]`**
  - Current model built from updated grammar tokenizer: shape **`[53, 1024]`** (base ESMC 33 + **20** grammar tokens; checkpoint used 33 + **18**)
- Root cause: **`grammar.py` grammar-v2 token table grew** (e.g. added `<protbs>`, `<protbd>`) after the checkpoint was saved; explicit `--resume latest.pt` cannot load incompatible embedding rows.
- Fix options: (1) implement partial resume with embedding resize for new grammar ids; (2) revert tokenizer/grammar token list to the 51-token layout used at step 14500; (3) restart with `--resume none` (lose optimizer state / step counter).

## 2026-06-24 Fixed vocab-expand resume + resubmitted no_encoder_qwen0_6b

- Implemented vocab-row expansion in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/training/checkpointing.py` (`adapt_model_state_dict_for_resume`, `load_resume_payload`): copies overlapping `token_embeddings` / tied `lm_head` rows when checkpoint vocab (51) < current model (53); skips optimizer state when vocab grows (new token rows get fresh Adam moments).
- Wired into `BioSeqTrainer.resume()`; added unit tests in `scripts/tests/bioseq/test_bioseq_checkpointing.py` (5 passed). Local smoke: `latest.pt` @ step 14500 loads into 53-token model successfully.
- Resubmitted `qwen3_vl_bioseq_grammar_v1_no_encoder_qwen0_6b` → `task_id=t-20260624181740-5hhhl`, initial status **`Initialized`**, **`Preemptible: true`**, still `--resume "${OUTPUT_DIR}/latest.pt"`.

## 2026-06-24 Docs cleanup + ESM2 weights confirmed

- Confirmed all 5 ESM2 snapshots present under `/c20250601/mj/model_weights/esm2`: `esm2_t6_8M` (hidden 320), `esm2_t12_35M` (480), `esm2_t30_150M` (640), `esm2_t33_650M` (1280, Ophiuchus-Ab base), `esm2_t36_3B` (2560). ESMC: `ESMC-300M` (960), `ESMC-600M` (1152), `ESMC-6B`.
- Consolidated the bloated `## Current Status` block into a concise status + `## Model Weights` summary (added ESM2 hidden dims); full path inventory remains the single source of truth in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJ_GUIDE.md`. Removed duplicated pending/downloaded bullets and research prose already preserved in dated sections.
- Refreshed the Active table from `volc ml_task list` (non-terminal only): removed 9 terminal rows (3 eval jobs, `pj2hb`, and the 06-22 38m/300m/600m/1b/esmc600m); kept `t-20260624174241-cbbs2` (Queue) and `t-20260623161337-dhlpw` (Running).

## 2026-06-24 Synced docs to implemented grammar-v2 (Round 1 baseline)

- Updated the four docs to match grammar-v2 Round 1 (superseded by Round 2 on 2026-06-25; see below).
- Round-1 snapshot: structure tokens `<ab>`, `<tcr>`, `<nb>`, `<pep>`, `<prots>`, `<protd>` plus 10 relation tokens; leading type markers outside `<prots>`; repeated `<prots>` per chain; PPI dual modes; condition projection. **No longer current.**

## 2026-06-25 Grammar v2 Round 2 docs + ab-ag type markers

- Implemented grammar-v2 Round 2 in code (`grammar.py`, `modeling_bioseq.py`, train entry, tests): type markers inside `<prots>`, `.` chain separator, PPI conditional-only, encoder embedding replacement (no projection), `hidden_size` aligned to encoder latent dim.
- Antigen-conditioned forms now include receptor type markers: **AB-Ag** `<prots> ANTIGEN <protd> <binding> <prots> <ab> HEAVY . LIGHT <protd>`; **NB-Ag** `<prots> ANTIGEN <protd> <binding> <prots> <nb> VHH <protd>`.
- Synced docs: `GRAMMAR_V1.md`, `BIOSEQ_MODEL_PLAN.md`, `PROJ_GUIDE.md`, `PPI_DATA.md`, `scripts/data/README.md`. Removed stale Round-1 references (`ppi_joint`, leading type markers outside `<prots>`, repeated `<prots>` per chain, condition projection).
- Bioseq pytest: **81 passed** after Round 2 + ab-ag/nb-ag marker change.

## 2026-06-25 Submitted grammar-v2 test training (batch 128)

- Created and submitted four preemptible 8×A100 grammar-v2 test jobs (`--batch-size 16`, `--grad-accum 1` → global **128**/step), `--resume none`, `--max-steps 10000`:
  - **no_encoder ~0.6B** (`L28/H1024`): `t-20260625123349-rzdgz` via `train_jobs/qwen3_vl_bioseq_grammar_v2_no_encoder_qwen0_6b.yml` → `output/grammar_v2_no_encoder_qwen0_6b`
  - **ESMC-300M** (decoder hidden auto **960**): `t-20260625123353-hln2f` via `train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m.yml` → `output/grammar_v2_esmc300m`
  - **ESMC-600M** (decoder hidden auto **1152**): `t-20260625123356-4vfhd` via `train_jobs/qwen3_vl_bioseq_grammar_v2_esmc600m.yml` → `output/grammar_v2_esmc600m`
  - **ESM2-650M** (decoder hidden auto **1280**): `t-20260625123359-dzst8` via `train_jobs/qwen3_vl_bioseq_grammar_v2_esm2_650m.yml` → `output/grammar_v2_esm2_650m`
- All four initial status **`Queue`**, **`Preemptible: true`**. Previous grammar-v1 jobs no longer in active table (terminal or finished).

## 2026-06-25 Resubmitted grammar-v2 jobs (chain separator + ESM2 path fix)

- **Root cause (first batch, all Failed ~10 min)**:
  - no_encoder / ESMC-300M / ESMC-600M: `GrammarRenderer.encode` → `chain_separator_id()` raised `AttributeError: Base tokenizer must expose token_to_id for chain separator '.'` because `HuggingFaceEsmTokenizerAdapter` / `TokenizersEsmTokenizer` did not expose `.` lookup (ESMC `tokenizer.json` has id 29, but adapter lacked API).
  - ESM2-650M: entrypoint `test -f /c20250601/mj/model_weights/.../config.json` failed on cluster (path not on Vepfs mount); Volc logs nearly empty.
- **Fixes**:
  - Added `token_id()` to `TokenizersEsmTokenizer` and `HuggingFaceEsmTokenizerAdapter`; relaxed `GrammarTokenizer.chain_separator_id()` to use it.
  - Symlinked ESM2 weights to `/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/esm2/`; updated esm2 YAML to Vepfs path.
  - Added `test_esmc_hf_tokenizer_supports_chain_separator`; grammar tests 11/11 pass.
- **Resubmitted** (same YAML batch 128 config, `--resume none`):
  - `t-20260625140029-66gnq` no_encoder_qwen0_6b
  - `t-20260625140032-t2vzk` esmc300m
  - `t-20260625140036-sz7wd` esmc600m
  - `t-20260625140039-dxt5t` esm2_650m
- Initial status: all **`Queue`**, **`Preemptible: true`**. Failed predecessors: `t-20260625123349-rzdgz`, `t-20260625123353-hln2f`, `t-20260625123356-4vfhd`, `t-20260625123359-dzst8`.

## 2026-06-25 Deleted v12 detailed; started v11 MINT-minimal download

- Confirmed `protein.links.detailed.v*.txt.gz` is **per-channel evidence subscores only** (neighborhood/fusion/cooccurence/coexpression/experimental/database/textmining + combined_score); it has **NO mode/action** (binding/activation/inhibition/catalysis/reaction/ptmod/expression). mode lives in `protein.actions.v11.0.txt.gz` (v12 replaced it with the API-only `regulatory` network). Refs: STRING help/database (`network.actions` table), help/faq (score columns), STRING-2025 NAR (PMC11701646, regulatory directionality).
- Detailed is **not needed** for MINT-style processing. **Deleted** corrupt `protein.links.detailed.v12.0.txt.gz` (203,534,412,387 bytes, `gzip -t` failed) + `.aria2` control file → freed ~190 GB.
- MINT minimal input set (per `run_mint_stringdb_native.py`): `protein.physical.links.full.*.txt.gz` + `protein.sequences.*.fa` (+ locally-generated `clu50.tsv` from MMseqs2 50% clustering — NOT a download).
- Refactored `scripts/data/download_stringdb_assets.sh` to be version-aware: `--version v11.0`, `--with-actions` (v11 only), `--with-detailed` (opt-in). Default = MINT minimal (full physical links + sequences) with `gzip -t` + byte-size verification (v11.0 full=33,237,086,717; sequences.fa.gz=5,526,372,370; actions=12,858,366,570) and `aria2c -c` resume.
- Started **v11.0 MINT-minimal** background download (`pid=1859694`, log: `data/ppi_task_raw/processed/pipeline_logs/string_v11_minimal_download.log`): `protein.sequences.v11.0.fa.gz` (~5.1 GB) then `protein.physical.links.full.v11.0.txt.gz` (~31 GB).
- Pending after download: gunzip sequences → MMseqs2 `clu50.v11.0.tsv` → `run_mint_stringdb_native.py --links-gz ... --sequences-fa ... --cluster-tsv ... --output-dir data/ppi_task_raw/processed/mint_string_pretrain_v11.0`.

## 2026-06-25 Prepared v11 MMseqs2 clustering job (not yet submitted)

- Parametrized `scripts/data/run_mint_mmseqs_cluster.sh` with `STRING_VERSION` env (default `v12.0` keeps `DB100`/`clu50.tsv` for backward compat; other versions → `DB100_<ver>` + `clu50.<ver>.tsv` + per-version tmp dir + log, so v11 never clobbers the existing v12 `clu50.tsv`).
- Created `train_jobs/mint_string_mmseqs_cluster_v11.yml` (clone of `mint_string_mmseqs_cluster_c1ie.yml`, `ml.c1ie.21xlarge`, CPU-only, preemptible, `STRING_VERSION=v11.0`, output `clu50.v11.0.tsv`).
- **Submit gated on download**: clustering needs the complete `protein.sequences.v11.0.fa`; the v11 minimal download (`pid=1859694`) is still in progress (STRING source ~230 KiB/s; sequences ~10% as of 05:46, then 31 GB physical links). Submit `mint_string_mmseqs_cluster_v11` only after `gzip -t protein.sequences.v11.0.fa.gz` passes.
