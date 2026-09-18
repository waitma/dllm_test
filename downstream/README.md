# BioSeq downstream tasks

> **Ophiuchus-Ab baseline 表征探针专项（2026-09-13）**：纠错台账和后续 agent 验收入口见 [AB baseline 测评指南](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/AB_BASELINE_EVALUATION_AUDIT.md)。本次仅登记文档，未修改实现或解冻 headline。

> **Agent 阅读顺序（唯一）**：[`benchmark/PROJGUIDE.md`](benchmark/PROJGUIDE.md)（怎么做）→ [`tasks/<TASK>.md`](tasks/)（做什么、怎么填）→ [`benchmark/RESULTS.md`](benchmark/RESULTS.md) §0（数字）。
> 在范围内六份：[`tasks/TCR_T1_BINDING.md`](tasks/TCR_T1_BINDING.md) · [`TCR_T2_CLUSTERING.md`](tasks/TCR_T2_CLUSTERING.md) · [`TCR_T3_REPRESENTATION.md`](tasks/TCR_T3_REPRESENTATION.md) · [`TCR_T4_GENERATION.md`](tasks/TCR_T4_GENERATION.md) · [`AB_CDR_INFILLING.md`](tasks/AB_CDR_INFILLING.md) · [`AB_LIGHT_CHAIN_PAIRING.md`](tasks/AB_LIGHT_CHAIN_PAIRING.md)。
> 文档规则：[`../.cursor/rules/downstream-doc-sync.mdc`](../.cursor/rules/downstream-doc-sync.mdc)。排行榜 **论文值优先**。

Downstream scripts migrated from `/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/downstream`.

Data defaults: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream`

**归档（非现行入口）**：[`AB_TCR_EVAL_SUMMARY.md`](AB_TCR_EVAL_SUMMARY.md) · [`downstream.md`](downstream.md)（旧 Ours 数字 VOID）。现行数字只看 [`benchmark/RESULTS.md`](benchmark/RESULTS.md) §0。

---

## 评测范围（2026-08-29 收窄，强制）

2026-09-13 TCR 当前子任务与双链生成延后决定见 [benchmark README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md)；本轮新的 Ours 汇总只收指定 49000，不将旧 checkpoint 数字补入空项。

> **当前只做 AB（抗体）与 TCR 两个维度的任务，其余一律不考虑。**

- **在范围内**：TCR T1 Binding / T2 Clustering / T3 Representation / T4 Generation；AB CDR infilling（SAbDab Kong + SAb23H2）、AB Light-chain pairing。AB 的 humanization / GDPa1 / specificity / m396 属 AB 维度、本轮按计划排除。后三项数据已落地，见 [`../data/downstream/PROBE_DATA.md`](../data/downstream/PROBE_DATA.md)。
- **⛔ 已冻结，不在范围内**：`mint_tasks/`（MINT GeneralPPI）、`benchmark/ppi/`（STRING 90/90）、`benchmark/nbbench/`（纳米抗体）、`flab/`（抗体属性回归，锚点是 MINT 而非 AB 主线）。
- 冻结含义：**代码、数据与已落盘产物全部保留**，但不再更新数字、不进 headline、不写入论文表、不作为结论依据。

范围定义与逐任务锚点见 [`benchmark/README.md`](benchmark/README.md) §0 / §3。

**Ophiuchus-Ab 官方 ckpt 复跑**（不含 humanization）：[`ophiuchus_eval/`](ophiuchus_eval/) · `bash downstream/ophiuchus_eval/run_eval.sh`

---

## ASD 预处理（抗体↔抗原 recognition 语料）

把 `data/asd`（抗原特异性抗体库，1,227,083 行）清洗成 **抗体 Fv + 抗原结构域** 配对语料，
纳米/常规抗体分库，产出 CSV，并接入 `examples/llada/protein_pretrain_esmc.py` 的
`antibody_antigen` / `nanobody_antigen` recognition 平面。完整逐步记录（含每步丢弃计数、
亲和力测法、接线与训练命令）见 [`asd/README.md`](asd/README.md)。

| 项 | 内容 |
|------|------|
| 脚本 | `asd/scripts/step0_load.py … step6_split.py`（+ `affinity_summary.py`、`validate_wiring.py`） |
| 产物 | `asd/step0…step6_final/`（全部在 `downstream/asd/` 下） |
| 最终 CSV | `asd/step6_final/{antibody,nanobody}/{train,valid,holdout}.csv`（90/5/5，mmseqs cluster-disjoint） |
| 规模 | 抗体 994,354（train 894,918）· 纳米 144,700（train 130,230），去重后 |
| Fv 提取 | RIOT IMGT `sequence_alignment_aa` 优先，abnumber(ANARCI) 兜底 + scFv 拆 VH/VL |
| 抗原 | ≤1024aa 直通；>1024aa 用 pyhmmer/Pfam 按结构域切片 |
| 亲和力 | [`asd/affinity_summary.md`](asd/affinity_summary.md)；`bool==0 → <nonbinding>`，其余 `<binding>` |
| 参考库 | `data/reference/pfam/Pfam-A.hmm`、`data/reference/mmseqs/`（v17） |
| 接线 | `dllm/pipelines/bioseq/datasets.py`（`asd_*_row_to_record`）+ 该 entry 的 `_roles_for`/`build_immune_specs`；grammar/ESMC 无需改动 |

训练（ASD 抗原上下文最长 1024aa，须放开长度上限）：
```bash
PYTHONPATH=. python examples/llada/protein_pretrain_esmc.py \
  --dataset_args asd --max_protein_length 1024 --max_length 1408
```

---

## TRAIT 预处理（TCR↔pMHC recognition 语料）

把 `data/trait`（TRAIT TCR–抗原库）洗成 **peptide + HLA 伪序列 + CDR3α/β** 语料，
接入同一入口的 `tcr_pmhc` / `tcr_epitope` 平面。逐步记录见 [`trait/README.md`](trait/README.md)。

| 项 | 内容 |
|------|------|
| 脚本 | `trait/scripts/step0_load.py … step4_split.py` + `validate_wiring.py` |
| 最终 CSV | `trait/step4_final/{train,valid,holdout}.csv` |
| 规模 | 75,380（train 67,842 / valid 3,769 / holdout 3,769）；binding 50,057 / nonbinding 25,323 |
| 序列 | 无全长 TCR；CDR3 + 表位；HLA 用 PISTE 34aa 伪序列 |
| 负样本 | 主表全是 binder；Omics 负样本按抗原抽样后与文献对去重 |
| 接线 | `trait_row_to_record` + `--dataset_args trait` |

```bash
PYTHONPATH=. python examples/llada/protein_pretrain_esmc.py --dataset_args trait
```

---

## 目录结构（AB 抗体任务 vs TCR benchmark）

| 目录 | 内容 | 范围 |
|------|------|:--:|
| `benchmark/` | TCR 四类统一评测（IRBench）；内含已冻结的 `ppi/` 与 `nbbench/` | ✅ |
| `grammar/` | **我们的** grammar_v2 生成适配器（CDR infill、light pairing） | ✅ |
| `infill/` | CDR infilling（Ophiuchus-Ab + PLM baseline） | ✅ |
| `comp_chain/` | Light-chain pairing | ✅ |
| `ophiuchus_eval/` | Ophiuchus-Ab 官方 ckpt 复跑（CDR + pairing） | ✅ |
| `humanization/` | 抗体人源化（AB 维度，本轮排除） | ◐ |
| `dev/` | Developability（GDPa1；数据已落地，本轮排除） | ◐ |
| `specificity/` | 特异性分类（CurrAb；数据已落地，本轮排除） | ◐ |
| `in_silico/` | Desautels m396 亲和力（数据已落地，本轮排除） | ◐ |
| `flab/` | FLAb 属性回归 + baseline sweep | ⛔ 冻结 |
| `mint_tasks/` | MINT GeneralPPI | ⛔ 冻结 |
| `common.py` / `embeddings.py` | Ophiuchus-Ab 共享加载与嵌入 | ✅ |

✅ 在范围内 · ◐ 属 AB 维度、本轮按计划排除（探针数据见 [`../data/downstream/PROBE_DATA.md`](../data/downstream/PROBE_DATA.md)） · ⛔ 已冻结（见上方「评测范围」）

---

## AB 任务（Ophiuchus-Ab 论文口径 · 2026-07-08 baseline 扩充）

锚点模型 = **immune fusion**（`output/protein_esmc_llada{270m,8b}_{bert,diffusion}_immune/checkpoint-*`）。旧 grammar_v2 ckpt **VOID**。现行任务文档：[`tasks/`](tasks/)。
对照 baseline 使用本地已有 ckpt：**ESM-2 650M**、**Ophiuchus-Ab 官方 ckpt**、AntiBERTy / IgBERT / ProtBERT 等。

| 任务 | 我们的脚本 | Baseline 脚本 | 结果目录 | 范围 |
|------|-----------|---------------|----------|:--:|
| CDR infilling (SabDab) | `grammar/cdr_infill.py` | `infill/run_cdr_baselines.py` (AntiBERTy / AbLang2) | `output/downstream_generation/cdr_baselines/` | ✅ |
| Light pairing (OAS500) | `grammar/light_chain_pairing.py` | `pairing_baselines/run_pairing_baselines.py` (p-IgGen / LICHEN) | `output/downstream_generation/{grammar_v2_*_light_pairing_*, pairing_baselines/}` | ✅ |
| CDR (Ophiuchus 官方 ckpt) | [`ophiuchus_eval/`](ophiuchus_eval/) | — | `output/downstream_generation/ophiuchus_ab/` | ✅ |
| Humanization | `humanization/humanize.py` | Ophiuchus-Ab 已通；HuDiff/IgCraft 未接 | `output/downstream_generation/*_humanization*` | ◐ 本轮排除 |
| Humanization (Ours) | `grammar/humanization.py` | — | `output/downstream_generation/grammar_v2_*_humanization*` | ◐ 本轮排除 |
| ⛔ FLAb 属性回归 | `flab/run_flab_baselines.py` | 同脚本 `--embedder esm2_650m\|ophiuchus\|…` | `output/downstream_generation/flab_baselines/` | ⛔ 冻结 |

详见各子目录 README：`infill/README.md`、`humanization/README.md`。`flab/README.md` 仅作存档。

### 一键复现（AB baseline sweep）

```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test

# ⛔ FLAb 已冻结（见「评测范围」），以下两行仅作存档，不要用于出数：
#   bash downstream/flab/run_flab_all.sh
#   python downstream/flab/summarize_ab_baselines.py

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

# ⚠️ 以下 grammar_v2 命令是历史 VOID 模板，不作 headline。现行入口见 tasks/ 与
# scripts/downstream/run_immune_fusion_{repr,gen,pairing}.sh
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
| in_silico / GDPa1 / specificity | 数据已落地，本轮按计划不跑；见 [`../data/downstream/PROBE_DATA.md`](../data/downstream/PROBE_DATA.md) |
| CDR dyMEAN/IgGM | 需结构预测 + 独立 repo |
| humanization HuDiff/IgCraft | 外部生成器未接（Ours grammar + Ophiuchus-Ab 已通） |
| ~~AbLang2 (CDR)~~ ✅ 已跑通 | — |
| ~~pairing p-IgGen/LiChen~~ ✅ 已跑通 | — |

---

## Aligned to `MultiChainOphiuchusAbModel`

Official-ckpt local rerun (no humanization) lives in [`ophiuchus_eval/`](ophiuchus_eval/). Shared helpers stay in `common.py`.

| Task | Script |
|------|--------|
| Heavy → light | `ophiuchus_eval/light_pairing.py` |
| CDR infill (SAbDab) | `ophiuchus_eval/cdr_sabdab.py` |
| CDR infill (SAb23H2) | `ophiuchus_eval/cdr_sab23h2.py` |
| Humanization | `humanization/humanize.py`（本轮不跑） |
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
- `infill/zeroshot_cdr.py`, `infill/zeroshot_sab23h2.py`, `comp_chain/generate_light_from_csv.py`（历史端口；评测入口改走 `ophiuchus_eval/`）
- `infill/zeroshot_SAb23H2.py` (AirGen filename)

Prefer [`ophiuchus_eval/`](ophiuchus_eval/) for official-ckpt reruns.

## Recommended decoding configs (Ophiuchus-Ab checkpoint)

| Task | sampling_strategy | max_iter | cfg_scale | Notes |
|------|-------------------|----------|-----------|-------|
| CDR infilling | `argmax` | **1**（报告口径；sweep 仅作波动带） | 0.0 | 论文未公布 max_iter。SAbDab 用 Kong `sabdab_kong`，禁止默认旧 `sabdab/` |
| Light-chain pairing | `gumbel_argmax` | **124** | 0.0–1.5 | 官方 AirGen `heavy2light.sh` 为 prompt3、124 步、固定窗口自吐 EOS；cfg 在测试集上扫过，不得把 cfg=1.5 标成超过论文。Ours 的现行适配协议另见 [AB pairing §4.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)，不能与官方长度设置混称 |
| Humanization | `gumbel_argmax` | 32 | 0.0 | FR regions masked; light C-terminal 3 residues kept native |

CDR infilling 报告口径固定 `iter=1`。pairing 对齐官方 `max_iter=124` + gumbel；不要用 argparse 默认 32。

## Examples

```bash
bash downstream/ophiuchus_eval/run_eval.sh

python downstream/ophiuchus_eval/light_pairing.py \
  --df-path data/downstream/comp_chain/test_data_oas_holdout.csv --output-file /path/to/output.csv

python downstream/ophiuchus_eval/cdr_sabdab.py \
  --test-set data/downstream/cdr_infilling/sabdab_kong/cdrh3 --mode cdrh3

python downstream/humanization/humanize.py \
  --pdb-dir /path/to/cif --info-csv-fpath /path/to/chain_pairs.csv --output-csv humanized.csv

python downstream/flab/run_flab_baselines.py \
  --dataset data/downstream/flab/flab_raw/koenig2017mutational_kd_g6.csv \
  --embedder ophiuchus --device cuda \
  --output output/downstream_generation/flab_baselines/g6_Kd__ophiuchus.json
```
