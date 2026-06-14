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

## 2026-06-14 data format audit for Qwen-style BioSeq

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

## 2026-06-14 Qwen3-VL BioSeq data loader

- User chose to prioritize a training-time loader instead of full offline conversion to JSON/JSONL.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data` with canonical records, source loaders, mixture datasets, view sampler, ESM-family tokenizers, and Qwen-style diffusion collator.
- First-version sources cover OAS paired antibody CSV, OTS paired TCR CSV, nanobody CSV, existing processed JSONL for PPI/TCR-epitope, and an optional PPI Arrow source.
- Loader output is `BioSeqRecord`; masking is deferred to `BioSeqViewSampler` and `BioSeqQwenDataCollator`, which emit `visible_mask`, `fixed_context_mask`, `diffusion_target_mask`, and `diffusion_loss_mask`.
- Encoding defaults to ESM2/MINT-compatible token ids through `Esm2SequenceTokenizer`; the collator also emits per-chain `encoder_input_ids`, `encoder_attention_mask`, `encoder_residue_mask`, `encoder_chain_mask`, and `encoder_chain_role_ids` for future ESM2/ESMC encoder-conditioned training.
- Verified with `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq -q`: 27 passed.

## 2026-06-14 ESM tokenizer and multi-chain paper verification

- Verified local ESM2 tokenizer files under `/c20250601/mj/model_weights/esm2/esm2_t30_150M_UR50D`, `/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D`, and `/c20250601/mj/model_weights/esm2/esm2_t36_3B_UR50D`: ids 0-32 are `<cls>`, `<pad>`, `<eos>`, `<unk>`, `L`, `A`, `G`, `V`, `S`, `E`, `R`, `T`, `I`, `D`, `P`, `K`, `Q`, `N`, `F`, `Y`, `M`, `H`, `W`, `C`, `X`, `B`, `U`, `Z`, `O`, `.`, `-`, `<null_1>`, `<mask>`.
- Verified local ESMC tokenizer files under `/c20250601/mj/model_weights/esmc/ESMC-300M`, `/c20250601/mj/model_weights/esmc/ESMC-600M`, and `/c20250601/mj/model_weights/esmc/ESMC-6B`: ids 0-30 and 32 match the ESM2 amino-acid/special-token ids above, but id 31 is `|` rather than `<null_1>`, and `special_tokens_map.json` marks `|` as an additional special token.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/README.md` with the tokenizer rule: use `Esm2SequenceTokenizer` for Ophiuchus-Ab/MINT/ESM2 paths; use the local Hugging Face tokenizer adapter or an ESMC-specific tokenizer for ESMC encoder paths.
- Re-extracted `/tmp/esm_protein.pdf` to `/tmp/esm_protein.txt` with Ghostscript and checked the ESMC/ESMFold2 preprint. The paper's ESMC section describes a masked language model over protein sequences and single-chain contact evaluation; the explicit multi-chain treatment appears in ESMFold2.
- Paper-specific conclusion: ESMFold2 uses frozen ESMC 6B representations. For multiple protein chains, each chain is encoded independently by ESMC, then ESMFold2 crops/concatenates chain representations into a complex-level folding trunk with pair representations and atom-level diffusion. Therefore BioSeq should keep multi-chain interaction learning in the decoder/collator/attention or an ESMFold2-style pair module, not assume ESMC alone models cross-chain interactions.

## 2026-06-14 Qwen3-VL BioSeq view-mask correction

- User clarified that `full_denoise` must not include antigen or pMHC context chains in diffusion loss.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/view_sampler.py`: `full_denoise` now targets only eligible chains. It honors explicit `metadata["targets"]` when present, then filters out antigen, peptide, MHC, HLA-like, and epitope roles as fixed context.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/sources.py` so `processed_json_to_record` only writes `metadata["targets"]` when the source row explicitly provides `targets`; rows without targets now let the view sampler infer eligible chains from roles.
- Added coverage in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py` to verify that `full_denoise` keeps peptide, MHC, and antigen fixed even when metadata accidentally lists all chains as targets.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/README.md` with the corrected `full_denoise` semantics.

## 2026-06-14 Qwen3-VL BioSeq data reading diagnostic

- Added temporary diagnostic script `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/inspect_qwen3_vl_data_reading.py`.
- The script samples local sources through the current Qwen3-VL BioSeq loader, then checks source parsing, record roles/tasks, `full_denoise` masks, default random views, empty-loss examples, collator errors, and whether fixed context roles accidentally enter `diffusion_loss_mask`.
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
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/mixture.py::TaskHomogeneousBatchDataset`. It wraps any BioSeq record stream and groups records by BioSeq task group.
- Added `bioseq_task_group` and `bioseq_record_fingerprint` helpers. `bioseq_task_group` separates nanobody from paired antibody, and separates antibody-antigen/nanobody-antigen from generic antibody records based on chain roles.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/collator.py`: `BioSeqQwenDataCollator` now samples one shared generation view per batch by default through `BioSeqViewSampler.sample_batch`, and can enforce homogeneous task groups with `require_homogeneous_task=True`.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/view_sampler.py` with `sample_batch`, `compatible_views`, and public `build` helpers.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/README.md` and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` with the required training path: source stream -> `TaskHomogeneousBatchDataset` -> `DataLoader(batch_size=None)` -> `BioSeqQwenDataCollator(require_homogeneous_task=True)`.
- Training-stability note: physical microbatches should be homogeneous, but optimizer steps should accumulate gradients across several task-homogeneous microbatches or use a later weighted task scheduler to avoid high-variance single-task updates.
- Verified syntax with `python -m py_compile` for `mixture.py`, `collator.py`, and `view_sampler.py`.
- Verified `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py -q`: `20 passed`.
- Re-ran `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/inspect_qwen3_vl_data_reading.py --limit-per-source 128 --batch-size 16 --max-chain-length 512`; OAS, OTS, nanobody, and `processed_v2` all reported `issues: none`.

## 2026-06-14 Batch de-duplication boundary correction

- User clarified that sample de-duplication is already handled during data processing, so batch-level de-duplication should not be part of the default training path.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/mixture.py::TaskHomogeneousBatchDataset`: `deduplicate_within_batch` now defaults to `False`; setting it to `True` is only a defensive option for debugging untrusted/overlapping streams.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/README.md` and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md` to clarify the boundary: data processing owns de-duplication, while the training batcher owns task/view homogeneity.
- Verified syntax with `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/mixture.py`.
- Verified `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py -q`: `21 passed`.

## 2026-06-14 Simple view sampling probability

- User clarified that view sampling should stay simple: keep `full_denoise` high, and randomly sample other condition views from the remaining probability mass.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/view_sampler.py`: `BioSeqViewSampler` now has `full_denoise_probability=0.5` by default.
- Sampling rule: if `full_denoise` and at least one condition view are compatible, sample `full_denoise` with probability `0.5`, otherwise sample uniformly from compatible condition views with probability `0.5`. If no condition view is compatible, fall back to `full_denoise`.
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

## 2026-06-14 Qwen3-VL BioSeq training model layer

- User request: start building the trainable model architecture under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch`, with both a no-encoder version and an ESMC/ESM encoder-conditioned version. Encoder-conditioned training should follow the BioSeq diffusion loss/noise rules.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`.
- Implemented `BioSeqDiffusionTransformerConfig`, `BioSeqDiffusionDecoder`, `BioSeqNoEncoderDiffusionModel`, and `BioSeqEncoderDiffusionModel`.
- Implemented BioSeq diffusion utilities in the same module:
  - `sample_bioseq_diffusion_noise`: samples timestep corruption only from `diffusion_loss_mask` / `diffusion_target_mask`, keeps fixed context residues clean, guarantees at least one corrupted target residue per eligible row, and returns noised decoder ids plus labels.
  - `apply_decoder_corruption_to_encoder`: maps corrupted decoder residues back to per-chain encoder residues and masks those encoder tokens before the encoder forward pass.
  - `compute_masked_cross_entropy`: computes denoising cross-entropy only on corrupted target positions.
- The no-encoder model computes diffusion loss directly over the qwen3_vl_arch collator output.
- The encoder-conditioned model runs a per-chain encoder, pools residue states into chain features, gathers those features back to decoder token positions by `chain_ids`, and conditions the decoder through a projection. Encoder parameters are trainable by default; `freeze_encoder=True` is available for ablations.
- Important leakage rule implemented: when a target residue is corrupted for decoder diffusion, the corresponding residue token in `encoder_input_ids` is also replaced with `<mask>` before encoder forward. Fixed context chains such as antigen remain clean and can condition target denoising.
- Exported the new model/loss utilities from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/__init__.py`.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py`.
- Test coverage verifies decoder-only loss, encoder-conditioned loss, encoder target masking, fixed antigen preservation, trainable encoder gradients, frozen encoder behavior, extra collator-field tolerance, and diffusion sampler mask boundaries.
- Local ESMC loading check: `/c20250601/mj/model_weights/esmc/ESMC-300M/config.json` declares `model_type="esmc"` and `transformers_version="4.57.6"`. Current environment has `transformers==4.48.1`, so `transformers.AutoModel.from_pretrained("/c20250601/mj/model_weights/esmc/ESMC-300M")` fails because this Transformers version does not recognize ESMC.
- Resulting constraint at this point was that unit tests used a tiny differentiable encoder to validate the training path. This was later resolved by adding the local Biohub `esm==3.2.3` ESMC loader documented in the 2026-06-14 compatibility fix section below.
- Verified syntax:
  - `python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py`
- Verified tests:
  - `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py -q`: `5 passed`
  - `python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py -q`: `29 passed`

## 2026-06-14 Qwen3-VL BioSeq DDP training path

- User clarified that training needs to support multi-node/multi-GPU execution.
- Updated `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/mixture.py` so iterable record streams are DDP-aware. `SequentialMultiSourceDataset` and `WeightedMixtureDataset` now shard source rows with `global_shard_index = rank * num_workers + worker_id` and `num_shards = world_size * num_workers`.
- Added `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py`.
- The new trainer supports:
  - `torchrun` single-node and multi-node launch through environment variables `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`.
  - decoder-only `--model-type no_encoder` training today.
  - encoder-conditioned `--model-type encoder` through `BioSeqEncoderDiffusionModel.from_esmc(...)`, now backed by the local Biohub ESMC fallback loader when Hugging Face `AutoModel` cannot recognize `model_type="esmc"`.
  - task-homogeneous BioSeq batches from `TaskHomogeneousBatchDataset`.
  - shared-view collation through `BioSeqQwenDataCollator(require_homogeneous_task=True)`.
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
