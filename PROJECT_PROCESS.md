# Project Process

## 2026-06-07

- Confirmed `/vepfs-mlp2/c20250601/251105016/project/dllm_test` is the active project root.
- Confirmed `/c20250601/mj/model_weights` is the required model weight root.
- Chose `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq` instead of a top-level `airgen_bioseq` package because `/vepfs-mlp2/c20250601/251105016/project/dllm_test/pyproject.toml` currently packages only `dllm`.
- Started the BioSeq pipeline implementation with independent data, model, diffusion training, weight download, and weight verification modules.
- Added the rule that all later task, plan, and process changes must be written to Markdown files with absolute paths.
- Changed `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/__init__.py` and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/__init__.py` to lazy imports so importing BioSeq does not force-load old `dllm.core` dependencies.

## Current Status

- BioSeq code scaffold: in progress.
- Documentation scaffold: in progress.
- ESMC/ESM2 weights under `/c20250601/mj/model_weights`: pending.
- Ophiuchus-Ab checkpoint under `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab`: pending.
- Weight downloads should fetch Hugging Face/PyTorch-compatible model files and skip TensorFlow `.h5` duplicates.
- ESMC weights downloaded and file-verified:
  - `/c20250601/mj/model_weights/esmc/ESMC-300M`
  - `/c20250601/mj/model_weights/esmc/ESMC-600M`
  - `/c20250601/mj/model_weights/esmc/ESMC-6B`
- Download tool changed to use known model manifests and size-checked direct curl for large weight files because Hugging Face `snapshot_download` and raw curl retry both produced unreliable large-file writes on `/c20250601/mj/model_weights`.
- Large weight files now stage partial downloads under `/vepfs-mlp2/c20250601/251105016/project/.download_tmp/bioseq_weights` and copy only complete, size-checked files into `/c20250601/mj/model_weights`.
- User changed the current weight scope: `/c20250601/mj/model_weights/esm2/esm2_t48_15B_UR50D` is no longer required now.
- Stopped the active 15B download and removed its partial staging directory and partial target directory.
- ESM2 weights downloaded and file-verified:
  - `/c20250601/mj/model_weights/esm2/esm2_t6_8M_UR50D`
  - `/c20250601/mj/model_weights/esm2/esm2_t12_35M_UR50D`
  - `/c20250601/mj/model_weights/esm2/esm2_t30_150M_UR50D`
  - `/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D`
  - `/c20250601/mj/model_weights/esm2/esm2_t36_3B_UR50D`
- ESM2 smoke-load passed for `/c20250601/mj/model_weights/esm2/esm2_t30_150M_UR50D`, `/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D`, and `/c20250601/mj/model_weights/esm2/esm2_t36_3B_UR50D`.
- Ophiuchus-Ab checkpoint downloaded from `https://zenodo.org/records/18478480` to `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt`.
- Ophiuchus-Ab checkpoint checksum verified: `md5:9baa0d3fbe908930d9a7d4f8d8b6144c`.
- Investigated ESMC/ESMFold2 multi-chain handling. Current conclusion: ESMC is used as a per-sequence language-model representation source, while ESMFold2 handles complexes through structure-prediction inputs and downstream folding/diffusion modules that build pairwise complex representations.
- Checked Biohub/EvolutionaryScale ESM documentation and the ESMC/ESMFold2 preprint. The implementation constraint for `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq` is now: encode protein chains with ESMC as per-chain/per-sequence features, then perform multi-chain interaction modeling in BioSeq chain-aware attention or an ESMFold2-style pair/folding/diffusion module.
- Cloned `https://github.com/Ophiuchus-Team/Ophiuchus-Ab` to `/tmp/Ophiuchus-Ab-src` for reference only. The official inference path uses `facebook/esm2_t33_650M_UR50D`, `--use_multimer`, `--sep_chains`, heavy length 150, and light length 128.
- Added Ophiuchus-compatible ESM2 tokenization, Ophiuchus-Ab fixed-length antibody collation, Ophiuchus-Ab model preset helpers, ESM2 token dropout support, and optional no-position embedding support under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq`.
- Added an initial temporary `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_ab.py` helper for early pipeline checks; this was later superseded by the exact Ophiuchus-Ab training entry.
- Verified `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq` with `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq -q`: 5 passed.
- Verified the temporary helper before the exact migration. Its generated checkpoint was not an Ophiuchus-Ab checkpoint and should not be used for Ophiuchus-Ab work.
- Inspected `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt`: 768 state dict keys, includes `layers.*.self_attn.*`, `layers.*.multimer_attn.*`, `layers.*.self_attn.rot_emb.inv_freq`, `emb_layer_norm_after.*`, and `lm_head.dense/layer_norm.*`; no learned position embedding keys.

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
- Current OAS compatible split under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/oas_previous_clean/splits/compat_for_current_loader_oasrule` has 2,486,414 train rows, 12,553 valid rows, and 12,653 holdout rows excluding CSV headers.
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
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` to make ESMC/ESM2 trainable by default in the encoder-conditioned path.
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
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` with an `End-to-End Pipeline` section covering raw data, manifests, adapters, canonical JSONL, mixture weights, masking/collation, no-encoder Ophiuchus/MINT training, future encoder-conditioned training, checkpointing, and evaluation.
- Updated the same plan with a `Fast Downstream Validation Plan`: Tier 0 held-out diffusion sanity, Tier 1 frozen-embedding IRBench, Tier 2 generation/infill, and Tier 3 task-specific fine-tuning.
- Fixed `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/model_api.py` so `BioSeqEmbedder` can load current DDP checkpoints containing `backbone_state_dict`, generic `state_dict`, or a plain state dict. This makes `--embedder bioseq:/abs/path/final.pt` compatible with checkpoints produced by `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_bioseq_ddp.py`.
- Verified the edited benchmark model API with `python -m py_compile` and a lightweight factory import check.
