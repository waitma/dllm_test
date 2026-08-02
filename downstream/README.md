# BioSeq downstream tasks

Downstream scripts migrated from `/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/downstream`.

Data defaults: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream`

**进展总览**：[`downstream.md`](downstream.md) · [`benchmark/RESULTS.md`](benchmark/RESULTS.md) · [`benchmark/PROGRESS.md`](benchmark/PROGRESS.md)

---

## 目录结构（AB 抗体任务 vs TCR benchmark）

| 目录 | 内容 |
|------|------|
| `benchmark/` | TCR / PPI / NbBench 统一评测（IRBench） |
| `grammar/` | **我们的** grammar_v2 生成适配器（CDR infill、light pairing） |
| `flab/` | FLAb 属性回归 + baseline sweep |
| `infill/` | CDR infilling（Ophiuchus-Ab + PLM baseline） |
| `comp_chain/` | Light-chain pairing |
| `humanization/` | 抗体人源化 |
| `dev/` | Developability（GDPa1，待数据） |
| `specificity/` | 特异性分类（待数据） |
| `in_silico/` | Desautels m396 亲和力（待数据） |
| `mint_tasks/` | MINT GeneralPPI |
| `common.py` / `embeddings.py` | Ophiuchus-Ab 共享加载与嵌入 |

---

## AB 任务（Ophiuchus-Ab 论文口径 · 2026-07-08 baseline 扩充）

锚点模型 = **grammar_v2 post-LLaDA**（`output/grammar_v2_esmc{300,600}m_cmp500k_llada/best.pt`）。
对照 baseline 使用本地已有 ckpt：**ESM-2 650M**、**Ophiuchus-Ab 官方 ckpt**、AntiBERTy / IgBERT / ProtBERT 等。

| 任务 | 我们的脚本 | Baseline 脚本 | 结果目录 |
|------|-----------|---------------|----------|
| CDR infilling (SabDab) | `grammar/cdr_infill.py` | `infill/run_cdr_baselines.py` (AntiBERTy / AbLang2) | `output/downstream_generation/cdr_baselines/` |
| Light pairing (OAS500) | `grammar/light_chain_pairing.py` | `pairing_baselines/run_pairing_baselines.py` (p-IgGen / LICHEN) | `output/downstream_generation/{grammar_v2_*_light_pairing_*, pairing_baselines/}` |
| FLAb 属性回归 | `flab/run_flab_baselines.py` | 同脚本 `--embedder esm2_650m\|ophiuchus\|…` | `output/downstream_generation/flab_baselines/` |
| CDR (Ophiuchus 生成) | `infill/zeroshot_cdr.py` | — | — |
| Humanization | `humanization/humanize.py` | Ophiuchus-Ab 已通；HuDiff/IgCraft 未接 | `output/downstream_generation/*_humanization*` |
| Humanization (Ours) | `grammar/humanization.py` | — | `output/downstream_generation/grammar_v2_*_humanization*` |

详见各子目录 README：`flab/README.md`、`infill/README.md`、`humanization/README.md`。

### 一键复现（AB baseline sweep）

```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test

# FLAb：4 数据集 × 8 embedder（含 ESM-2 650M + Ophiuchus-Ab + grammar_v2）
bash downstream/flab/run_flab_all.sh
python downstream/flab/summarize_ab_baselines.py

# CDR PLM baseline（AntiBERTy / AbLang2 zero-shot AAR）
bash downstream/infill/run_cdr_all.sh antiberty
bash downstream/infill/run_cdr_all.sh ablang2

# Pairing 生成 baseline（p-IgGen / LICHEN，heavy→light，n=8）
export PATH=$ENV/bin:$PATH HF_DATASETS_CACHE=/tmp/hf_datasets_cache  # PATH 需含 hmmscan
export https_proxy=100.68.162.212:3128 http_proxy=100.68.162.212:3128  # p-IgGen 拉 HF 权重
python downstream/pairing_baselines/run_pairing_baselines.py --baseline piggen --device cuda \
  --output output/downstream_generation/pairing_baselines/piggen_holdout500_n8.csv --resume
python downstream/comp_chain/eval_scripts/generation_eval.py \
  -i output/downstream_generation/pairing_baselines/piggen_holdout500_n8.csv \
  -g gen_l_sequence -r raw_l_sequence --heavy_col h_sequence --expected_count 8

# grammar_v2 CDR + pairing + humanization
bash scripts/downstream/run_grammar_v2_variant_downstream_eval.sh esmc600m_cmp500k_llada

# humanization only (Ours)
python -m downstream.grammar.humanization \
  --checkpoint-path output/grammar_v2_esmc300m_integrated_llada/best.pt \
  --output-csv output/downstream_generation/grammar_v2_esmc300m_integrated_llada_humanization.csv \
  --device cuda --num-seqs 8
```

### 本地 checkpoint 路径

| 模型 | 路径 |
|------|------|
| ESM-2 650M | `/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D` |
| Ophiuchus-Ab | `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt` |
| grammar_v2 300M | `output/grammar_v2_esmc300m_cmp500k_llada/best.pt` |
| grammar_v2 600M | `output/grammar_v2_esmc600m_cmp500k_llada/best.pt` |
| AbLang2-paired | `ablang2/model-weights-ablang2-paired/` (env 内) + 备份 `/c20250601/mj/model_weights/ablang2/` |
| p-IgGen | HF `ollieturnbull/p-IgGen`（缓存 `/c20250601/mj/model_weights/hf_cache`） |
| LICHEN | `downstream/pairing_baselines/weights/lichen_model/Model/model_weights.pt` |

### 已知 blocker

| 任务 | Blocker |
|------|---------|
| in_silico Desautels m396（论文 binding 口径） | 数据 dead symlink，全盘无副本 |
| dev GDPa1（论文 Table 4 developability）/ specificity | 本地无数据 |
| CDR dyMEAN/IgGM | 需结构预测 + 独立 repo |
| humanization HuDiff/IgCraft | 外部生成器未接（Ours grammar + Ophiuchus-Ab 已通） |
| ~~AbLang2 (CDR)~~ ✅ 已跑通 | — |
| ~~pairing p-IgGen/LiChen~~ ✅ 已跑通 | — |

---

## Aligned to `MultiChainOphiuchusAbModel`

These entry points use `common.py` and the Ophiuchus-Ab checkpoint:

| Task | Script |
|------|--------|
| Heavy → light | `comp_chain/generate_light_from_csv.py` |
| CDR infill (SAbDab) | `infill/zeroshot_cdr.py` |
| CDR infill (SAb23H2) | `infill/zeroshot_sab23h2.py` |
| Humanization | `humanization/humanize.py` |
| FLAb property regression | `flab/finetune_flab.py`（legacy）/ `flab/run_flab_baselines.py`（baseline 对比） |
| Developability regression | `dev/finetune_dev.py` |
| Specificity classification | `specificity/hd_flu_cov_paired.py` |
| Shared embeddings | `embeddings.py` |

## Legacy AirGen copies (optional third-party deps)

The following files were copied for reference and evaluation utilities. They may still import `byprot` or require tools such as `anarci`, `abnumber`, `abnativ`, or `wandb`:

- `comp_chain/eval_scripts/`
- `humanization/eval_scripts/`, `humanization/structure_rmsd.py`, `humanization/oasis_human_score.py`, ...
- `flab/finetune_flab_align.py`, `flab/finetune_flab_esm_ppi.py`
- `dev/finetune_dev_pplm.py`, `dev/tutorial.py`
- `in_silico/finetune_in_silico.py`
- `specificity/HD_Flu_Cov-paired.py`
- `infill/zeroshot_SAb23H2.py` (AirGen filename; prefer `zeroshot_sab23h2.py`)

Prefer the aligned scripts in the first table for Ophiuchus-Ab inference and embedding extraction.

## Recommended decoding configs (Ophiuchus-Ab checkpoint)

| Task | sampling_strategy | max_iter | cfg_scale | Notes |
|------|-------------------|----------|-----------|-------|
| CDR infilling | `argmax` | 4 | 0.0 | Mask one CDR region at a time |
| Light-chain pairing | `gumbel_argmax` | 32 | 0.0 | Optional `light_prompt_tokens=3` (first 3 light residues fixed) |
| Humanization | `gumbel_argmax` | 32 | 0.0 | FR regions masked; light C-terminal 3 residues kept native |

CDR infilling uses short deterministic decoding; full-length generation tasks (pairing, humanization) match AirGen defaults with longer iterative refinement.

## Examples

```bash
python downstream/comp_chain/generate_light_from_csv.py \
  --df-path data/downstream/comp_chain/test_data_oas_holdout.csv --output-file /path/to/output.csv

python downstream/infill/zeroshot_cdr.py \
  --test-set data/downstream/cdr_infilling/sabdab/cdrh3 --mode cdrh3

python downstream/humanization/humanize.py \
  --pdb-dir /path/to/cif --info-csv-fpath /path/to/chain_pairs.csv --output-csv humanized.csv

python downstream/flab/run_flab_baselines.py \
  --dataset data/downstream/flab/flab_raw/koenig2017mutational_kd_g6.csv \
  --embedder ophiuchus --device cuda \
  --output output/downstream_generation/flab_baselines/g6_Kd__ophiuchus.json
```
