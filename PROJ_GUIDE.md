# dllm_test Project Guide

## Long-Term Rules

- The project root is `/vepfs-mlp2/c20250601/251105016/project/dllm_test`.
- The model weight root is `/c20250601/mj/model_weights`.
- All project docs, configs, examples, and scripts must use absolute paths.
- Plan changes must be recorded in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md`.
- Process changes must be recorded in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md`.
- Long-term project rules must be recorded in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJ_GUIDE.md`.

## Active foundation-training data boundary

- The only current immune training implementation is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada`; the formal entry is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`.
- The active data contract is raw source → adapter → `BioSeqRecord` → offline deterministic filters → prepared semantic JSONL/manifest → training-time grammar, padding, per-chain encoder reconstruction, and diffusion/MLM masking. No model-ready token cache is produced.
- Current prepared roots: **v4** `data/prepared/immune_v4_beta_relation` (published) and **v5** `data/prepared/immune_v5_receptor_completion` (published). v3 / `immune_v3_heterotypic` remain on disk as live artifacts some checkpoints still point at; they are not the current training default. v4 headline counts live in `docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md` §4.1; v5 counts / v4 Δ live in §4.2. Operational how-to: `dllm/pipelines/immune_llada/README.md`. Layout: `DATA_FORMAT_AUDIT.md`.
- Receptor completion is gated by `tcr_region_profile.completion_sources` (default `("tcr_repertoire",)`). Withholding a source from that list is what disables completion. Do not branch completion on the `sequence_scope` label — use actual `alpha_fv` / `beta_fv` / `cdr3a` / `cdr3b` contents. Do not infer junction vs core from the first/last character; use provenance. `add_tcr_region_lengths` must skip `synthetic_regions` (circular-profile guard). `sample_region_lengths` must sort support lengths numerically (`atomic_json_dump` uses `sort_keys=True`). Drops that need audit attribution must be named filters (e.g. `quality.blank_epitope`), not silent adapter `None`. Manifest `filter_names` contract (union of constructed filters, not `BLOCKLIST_NAMES` keys): `/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md`. Item-13 counter numbers (`downgraded_all_x_mhc` / partner completion): plan §4.2.
- The active recipe remains immune-receptor-focused: antibody H/L, TCR α/β, antibody-antigen, and TCR-epitope/pMHC. Do not make deleted PPI/STRING/MINT training builders, nanobody/VHH, antigen-free neutralization, or specificity-free bulk TCR into current foundation-training inputs.
- The old `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data` alias tree, old `training` tree, old raw-CSV dataset implementation, and retired training entry points are deleted. Do not restore them or document them as current dependencies. The remaining `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`, `sampling_bioseq.py`, and `relation_aux.py` are model-layer compatibility code used by the fusion model and selected diagnostics.
- The former AB/TCR v2 canonical work under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/immune_receptor_v2` and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/IMMUNE_RECEPTOR_DATA_V2.md` remains a historical/candidate data artifact. It is not the current runtime training input while its export/training gates are false.
- Historical audit snapshots under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/refactor_baseline` and historical evidence under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive` must remain readable and must be labeled historical, not silently deleted or presented as executable current pipelines.
- Data layout and schema facts must stay synchronized with `/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md` and the independent pipeline description at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/README.md`.

## MINT benchmark rules

- MINT GeneralPPI 的 canonical 范围只有五项：`HumanPPI`、`YeastPPI`、
  `Bernett`（显示名 Gold-standard PPI）、`MutationalPPI`、`SKEMPI`。
  `Pdb-bind` 和 `MutationalPPI_cs` 不得加入默认注册表或全量下游编排。
- Canonical 数据根必须是
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/mint_official`；
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/mint` 仅为历史诊断数据。
- embedding cache 名必须包含 `mint_official_06694b7` 数据协议标识；数据版本切换后不得
  复用无该标识的历史 `.pt`。
- 每项数据必须由固定 MINT commit 中该目录的 `prepare_data.ipynb` 原样逐 code cell
  生成，并通过 notebook/input/output SHA256、schema、行数和 split 校验；不得用本地
  “等价实现”覆盖 canonical 输出。
- MutationalPPI 和 SKEMPI 的模型输入必须同时保留 WT 与 mutant 序列，并使用两者的
  PPI embedding 差；不能沿用公开 `MutationalPPI` 路由中只读 WT pair 的行为。
- 论文、notebook、公开评测代码不一致时不得猜造未发布 split。当前混合主表固定为：
  HumanPPI、YeastPPI、Gold-standard PPI 使用论文 Source Data `[P]`；MutationalPPI、
  SKEMPI 使用提交的本地固定 split 重跑全部 8 个 Figure-2 模型并标 `[L]`。后二者不得
  用论文值静默补空，也不得称为 paper-exact。
- `[L]` cache 必须同时带 `mint_official_06694b7_localfixed-v1-l2048`；MutationalPPI
  protocol 固定为 `mutppi_pair_group10_sgkf_seed0_v1`，SKEMPI 固定为
  `skempi_notebook_complex3_mt19937_v1`。选表入口只认这两个 protocol、完整数据、
  separate-chain、WT/mutant 对齐窗口和 3-repetition 640-hidden MLP；run cap 为 2048，
  但 ESM-1b 与 ProGen2-Large 必须按各自原生限制使用 1024，且逐模型记录实际长度。
- MINT 混合结果唯一机器可读入口为
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/mint_tasks/_selected_baseline_results.csv`；
  `_baseline_audit.json` 中存在 `missing_local_fixed_models` 时不得宣称五任务 baseline 已完成。
- Canonical 说明与已知差异位于
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/mint_tasks/OFFICIAL_REBUILD_AUDIT.md`，
  machine-readable provenance 位于
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/mint_official/manifest.json`。

## Volc job layout

- **Training YAMLs** live under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/`; only current immune-fusion training/resume jobs belong here. Old `grammar_v2` YAMLs are historical and must not be used to infer a runnable pipeline.
- **Eval YAMLs** live under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/`; downstream, pairing, and metrics jobs belong here. The old grammar-v2 generators/wrappers/sweeps/retry jobs are retired and must not be regenerated.
- Current local evaluation examples are the retained immune-fusion scripts under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream`: `run_immune_fusion_gen.sh`, `run_immune_fusion_pairing.sh`, `run_immune_fusion_repr.sh`, and `run_pairing_pll.sh`. They use the active fusion checkpoint line and the retained public implementations under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar`.
- Submit commands, when a current job is explicitly approved, use absolute YAML paths:
  - Training: `volc ml_task submit --conf /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/<job>.yml`
  - Eval: `volc ml_task submit --conf /vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/<job>.yml`
- Historical grammar-v2 checkpoints, wrappers, sweeps, and retry jobs remain only as provenance where archived; do not run them, regenerate them, or append their numbers to new headline tables. Do not alter external baseline protocols.

## Training batch sizing (Volc)

- **Same experiment, different encoders** (e.g. LLaDA ESMC-300M vs 600M): use the **same global effective batch** (`batch_size × grad_accum × world_size`) for fair comparison.
- **Maximize GPU memory**: on the target Volc flavor, increase per-GPU `--batch-size` until near OOM (~75–85% `mem_peak`), then set `--grad-accum` to hit the agreed global batch. Do not keep a legacy micro-batch just because an old run used it.
- **Before every submit**: profile short run on matching GPU; record `mem_peak` in YAML `Description` and/or `PROJECT_PROCESS.md`. Rule detail: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/.cursor/rules/volc-batch-sizing.mdc`.

## BioSeq Pipeline Boundary

- The active immune data pipeline is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada`; its formal training entry is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`.
- The active prepared-data path is `raw → adapter → BioSeqRecord → offline filter → prepared semantic JSONL → training-time grammar/padding/per-chain encoder reconstruction/masking`. The prepared loader still builds offsets and decodes semantic rows; it is not a zero-cost token cache.
- The current model-layer compatibility code retained under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch` is limited to `modeling_bioseq.py`, `sampling_bioseq.py`, and `relation_aux.py`. The deleted `data` alias tree, deleted `training` tree, deleted grammar document, and deleted legacy entry points must not be documented as active dependencies.
- `examples/llada/protein_fusion_model.py` reuses the retained model-layer functions for diffusion/MLM corruption, encoder-state mirroring, and relation auxiliary loss. The current immune data grammar/collator lives under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`.
- Public downstream grammar remains active where the fusion scripts use it: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar` retains the shared CDR, light-chain pairing, and TCR-generation implementations. Do not describe that directory as entirely deleted.
- Retained typical downstream entry points are `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_immune_fusion_gen.sh`, `run_immune_fusion_pairing.sh`, `run_immune_fusion_repr.sh`, and `run_pairing_pll.sh`; external baseline protocols and artifacts remain unchanged.
- Historical identifiers such as `qwen3_vl_arch`, `qwen3_vl_bioseq_*`, and `bioseq-qwen3-vl` may remain in old paths, task IDs, output directories, or archived records, but they are not current training instructions.

## Downstream Feature Extraction (post-LLaDA, mandatory)

- **All downstream tasks must extract features by passing sequences through the final LLaDA decoding architecture (post-LLaDA), not only the ESMC encoder's last layer.** This applies uniformly across every task family and both task modes:
  - Task families: MINT PPI (protein–protein interaction), antibody / nanobody (AB), and TCR.
  - Task modes: generative (e.g. CDR infilling, light-chain pairing, TCR generation) and predictive / embedding (e.g. binding classification, clustering, few-shot representation, PPI heads).
- Rationale: ESMC-encoder-only features capture only the frozen backbone and do not reflect the LLaDA decoder training that defines our model. Encoder-only extraction understates our contribution.
- Practical guidance:
  - Generative tasks are post-LLaDA by construction (decoding runs the full fusion model = ESMC encoder + LLaDA decoder), e.g. `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/cdr_infill.py`, `light_chain_pairing.py`, and `tcr_generation.py`.
  - Predictive / embedding tasks must use the post-LLaDA decoder path. Current fusion runners use the explicit `grammar:decoder:global:/absolute/path/to/checkpoint` specification; do **not** use encoder-only controls for headline numbers.
  - Encoder-only runs may be kept only as explicitly labeled controls (distinct tag), never as the primary reported result.

## External Baseline Protocol (original-only)

- For Nature Methods 2025 T1 binding, an external model baseline must use the author's original checkpoint, the author's inference code, and test rows derived from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ntmethod_binding/original.zip`. It must not load `train.csv`, fit a head, or retrain the baseline.
- If the original checkpoint/code/environment is unavailable or the official inference fails, use the paper's original-model value and label it exactly **“论文值，未本地复现”**. Never replace it with a locally retrained approximation, random score, or another checkpoint variant.
- The checkpoint/output tag mapping is part of the protocol. Multi-checkpoint wrappers (TEINet, ERGO, TPBTE) must bind the canonical variant tag before importing the wrapper; an output directory name alone is not evidence of which checkpoint ran.
- Canonical T1 source decisions live in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025/summary_original_baselines.csv`. Local-only diagnostics live in `summary_local_runs.csv`; paper transcriptions live in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/external/paper_reported_baselines.csv`.
- Nature Methods 2025 retrained artifacts are audited separately under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained`. Never merge them into the canonical original-model row. Verify both official archive MD5 values before inference, pair fold N only with fold-N checkpoint/data, use `precrec`-compatible AUPRC, and retain local plus paper values side by side. If both five-fold mean AUROC/AUPRC do not match the paper at four decimals, the selected retrained reference must be the paper value labelled “论文值，发布 artifact 未精确复现”. TCR-H may read fold train only to rebuild its released descriptor feature mask; `training_performed` must remain false. Officially dropped rows (currently ERGO-AE tail batches) must be counted explicitly rather than hidden.
- When placing Ours into the retrained-fold comparison, freeze every BioSeq backbone parameter and cache only `GrammarEmbedder(feature_source="decoder", pool_mode="global")` features from a joint peptide–CDR3β record. Train one independent MLP on each official fold train; validation must be carved only from that train. Do not reuse the old normalized `downstream/benchmark/data/tcr_binding_nm2025/train.csv` for this comparison.
- Rank Ours first against the local official-checkpoint reruns on the same `retrain.zip` members. Keep paper-reported retrained means in a second, explicitly mixed-source reference ranking. Preserve both AUROC and `precrec` AUPRC ranks and never silently substitute the paper value into the same-release primary table.

## Weight Layout

```text
/c20250601/mj/model_weights/esmc/ESMC-300M
/c20250601/mj/model_weights/esmc/ESMC-600M
/c20250601/mj/model_weights/esmc/ESMC-6B
/c20250601/mj/model_weights/esm2/esm2_t6_8M_UR50D
/c20250601/mj/model_weights/esm2/esm2_t12_35M_UR50D
/c20250601/mj/model_weights/esm2/esm2_t30_150M_UR50D
/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D
/c20250601/mj/model_weights/esm2/esm2_t36_3B_UR50D
/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab
```

The environment variable `BIOSEQ_MODEL_WEIGHTS_ROOT` may override the root, but its default must remain `/c20250601/mj/model_weights`.

`/c20250601/mj/model_weights/esm2/esm2_t48_15B_UR50D` is optional and is not part of the current default download set.

Current ESMC environment note: local ESMC checkpoints declare `model_type="esmc"`, but `transformers==4.48.1` does not recognize that model type. The retained model-layer loader is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py::BioSeqEncoderDiffusionModel.from_esmc` / `load_local_esmc_encoder`; those paths fall back to Biohub `esm==3.2.3` and load local safetensors without relying on `AutoModel` alone. Active ESMC tokenization is under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/esm_encoding.py`.

Historical BioSeq foundation stage-1 runs used offline wandb and output roots such as `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_esmc300m_stage1`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_esmc600m_stage1`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_no_encoder_stage1`. Preserve those paths as provenance only; current training must use the approved immune-fusion entry and output configuration. Do not use `/tmp` for ESMC training outputs or checkpoint tests because the root filesystem can be full and ESMC checkpoints are multi-GB.

## Historical PPI Data Layout（audit only; not current foundation training）

- PPI and interaction task raw data root: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw`.
- PPI and interaction task processed outputs:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_sources_manifest.csv`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_records_summary.csv`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_records_unified.csv`
- Historical rebuild command, retained for audit provenance only:

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/build_ppi_interaction_csv.py
```

- `interaction_records_unified.csv` is an audit/consolidation table. It is not a current training input; current foundation training uses prepared semantic records under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`.

## Public epitope-conditioned CDR3β Track-A rules

- The authoritative protocol is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/TCR_BETA_PUBLIC_TRACK_A.md`; do not mix its table with the older T4 three-setting metrics.
- Generate and retain exactly 1,000 raw candidates per model-target before filtering. A valid CDR3β contains only 20 standard amino acids, is 7–24 aa, starts with C, and ends with F or W. Never add anchors or otherwise repair a raw model output.
- References are target-specific values from `benchmark_data_w_preds.csv::reference_translations`; prediction columns are never references. Exact and recovery hits must remain within the same pMHC target.
- Run TcrDesign only in beta mode for the Track-A ranking. A separate qualitative artifact may run the official generated-beta-conditioned alpha module, but its beta inputs must come from the canonical generated `generations.csv`, never from reference/oracle β. TCR-epiDiff remains `reconstruction_only` unless the authors release a genuine pure-random de-novo sampler; never seed it with a test reference for a de-novo comparison.
- All comparable models use the same external exact/recovery/GIANA/Jaccard evaluator. GIANA is run per model-target with generated and reference sequences in the same pool and without V genes.
- TCRT5 `rank` and `raw_score` must be based on cumulative transition log-likelihood, not the length-normalized Hugging Face beam score. Every TCRT5 rerun must refresh `tcrt5_paper_consistency.{csv,md}` and verify candidate-set, exact-hit, and RVR GIANA agreement with the author artifact/paper.
- Do not compare the canonical all-14 TCRT5 exact-hit total directly with the paper's sparse-13 aggregate: the paper treats `RVRAYTYSK_HLA-A*03:01` as a separate deep-simulation experiment. Likewise, label the paper-native same-length identity recovery separately from Track A normalized-Levenshtein recovery.
- **RETIRED (2026-09-08): the `grammar_v2` BioSeq row is no longer reproducible.** All `grammar_v2_*` checkpoints were deleted during storage cleanup, so the 7-layer `step117000` pin (`output/grammar_v2_esmc300m_integrated_llada_7l_step117000/best.pt`, SHA256 `e3d4c98ea84b4aa2280fe76ec5bf8a806de376699610e9a96960e66ceddb5c01`) and the `step189000` RVR diagnostic now reference absent files. Its training corpus `data/bioseq_grammar_v1` was already gone, so the run cannot be retrained. Treat the recorded numbers as frozen historical artifacts: existing reports under `outputs/tcr_beta_public_benchmark/` and `checkpoint_comparison/bioseq_step*/` stay readable, but no `grammar_v2` row may be regenerated, re-scored, or added to a new table. Any future local BioSeq row must come from the `examples/llada` immune-fusion line. Retired provenance (top-k manifests, wandb summaries, deleted-file inventory) is archived at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/grammar_v2_retired_20260908`.
- Checkpoint diagnostics for the active line must use the same fixed protocol, derive the model label from the validated checkpoint manifest, and write under `outputs/tcr_beta_public_benchmark/checkpoint_comparison/<run_name>`. Never append a single-target or exploratory checkpoint run to the canonical all-14 tables.
- Canonical outputs live at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results`; every rerun must refresh `reproduction_status.md` and retain runtime provenance.
- TcrDesign qualitative α output is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/tcrdesign_alpha_generations.csv`; the official-release audit is `tcrdesign_alpha_evaluation_audit.json` in the same directory. Until an official runnable evaluator is released, keep `Official alpha evaluation: unavailable` and `Included in quantitative benchmark: no`; do not invent alpha exact/recovery/pairing metrics or use TcrDesign-B as a common αβ evaluator.

## TCRT5 full-eval canonical path and rules（2026-07-23）

- Full runner: `/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_tcrt5_full_eval.py all --with-pgen --reference-pgen --pgen-workers 16`.
- Canonical root: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval`; the human-readable authority is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/TCRT5_FULL_EVAL_REPORT.md`, and the machine-readable paper comparison is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/paper_correspondence.csv`.
- Never merge the paper's top-20 `K=100`, sparse13 `K=1000`, and separate RVR `K=1000` into one paper aggregate. The current all-14 comparison is an engineering view and must retain its own protocol label.
- For native recovery, use Levenshtein only to choose the closest same-length reference, then score positional Hamming identity; only use reference-length-normalized Levenshtein when no same-length reference exists. Do not substitute the common Track-A `1-Levenshtein/max(lengths)` formula.
- Do not report a paper-compatible mAP unless cumulative model-log-likelihood scores determine rank. Main-release ordered lists and models with scoreless emission order may expose AP diagnostics, but their evidence field must say proxy/diagnostic.
- Char-BLEU requires the greedy hypothesis and 20 nearest references. The main release contains stored numeric Char-BLEU but not the greedy strings, so it is a numeric companion check, not an independent recomputation from released hypotheses.
- OLGA reports must retain positive fraction plus positive `log10 Pgen` moments. The Fig.4 moments are not strictly reconstructable from the published candidate lists and stated protocol; keep the all-positive and `[-16,-5]` sensitivity rows, and never tune an undocumented cutoff to force agreement.
- After evaluator changes, run `/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python -m pytest -q /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/test_tcrt5_full_eval.py` and refresh all manifests under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval`.
