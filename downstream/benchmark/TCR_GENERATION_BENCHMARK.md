> ⚠️ **ARCHIVE / 非现行入口（2026-09-02）**：T4 现行入口 [`../tasks/TCR_T4_GENERATION.md`](../tasks/TCR_T4_GENERATION.md)。**Track-A 14×1000 与 Setting-B K=100 禁止合表。** 数字只写 [`RESULTS.md`](RESULTS.md) §0.4。

# TCR Generation Benchmark（TCRT5 锚定 · 三 setting）

> **2026-07-21 Track-A override:** the strict public epitope-conditioned
> CDR3β comparison requested for TCRT5, GRATCR, beta-only TcrDesign, and
> TCR-epiDiff is defined in
> `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/TCR_BETA_PUBLIC_TRACK_A.md`.
> For that track, its 7–24 aa C/F-or-W validity rule, 1,000-raw-candidate
> denominator, exact/reference-recall/recovery/GIANA/Jaccard metrics, and
> `reconstruction_only` exclusion override older T4 rules below. The older
> three-setting protocol remains historical context and must not be mixed into
> the new Track-A result table. The authoritative document also records a
> separate 140,000-row TcrDesign generated-β→generated-α qualitative artifact;
> it has no released official alpha evaluator and does not enter either the
> Track-A ranking or the historical Setting-C metric table below.

> IRBench 的 **T4 Generation** 方向，以 **TCRT5 (Nature Machine Intelligence 2025)** 为主评测口径重构。
> 一个方向、三种 conditioning setting；所有生成器走同一 harness（`tcr_generation_bench/run.py`）、
> 同一指标代码（`common/metrics.py` 的 `tcrt5_*` 套件），数字均可由 `outputs/tcr_generation_bench/` 复现。
>
> 相关文件：设计口径见本文件 + `TCR_BENCHMARK_DESIGN.md §T4`；排行榜见 `RESULTS.md`；过程见 `PROGRESS.md`；总览见 `../downstream.md`。

---

## 0. 文献调研与主构建基础选型（2024–2026，含 2026 新方法）

> 本节回答"为什么这样搭 T4"。结论：**以 TCRT5 (Nat Mach Intell 2025) 为主构建基础/锚定评测协议**，**补入最对口的近期扩散 baseline TCRDiff（2026，已复现）**与**全长 SOTA baseline（TcrDesign 2026）**；**TCR-epiDiff（2025 扩散）虽有官方权重但生成代码非 turnkey-reproducible，降级为仅引用**（同 LSMTCR/TCRGen 处理）；对无公开代码/权重的工作（LSMTCR、TCRGen）只引用不接入。

### 0.1 候选方法调研表（均核实原文 + 官方代码/权重可得性，2026-07-05）

| 方法 | 出处/年份 | 任务口径 | 架构 | 官方代码 | 官方权重 | 采纳 |
|------|-----------|----------|------|----------|----------|------|
| **TCRT5** (TCR-TRANSLATE) | Nat Mach Intell **2025** | Setting B: pMHC→CDR3β | seq2seq/T5 | `pirl-unc/tcr_translate` `97aeab10` | HF `dkarthikeyan1/tcrt5_ft_tcrdb` + Zenodo 15068617 | ✅ **主构建基础/锚** |
| **TcrDesign** | bioRxiv **2026**.01 | Setting C: 全长 epitope→α/β | BERT + Seq2Seq | `XSLiuLab/TcrDesign` | Zenodo 14545852 (2.48GB) | ✅ 全长 SOTA baseline |
| **TCRDiff** | bioRxiv **2026**.06 | Setting B: pMHC+V基因→CDR3αβ | **条件离散扩散(DPLM+cross-attn)** | `zhangyumeng1sjtu/TCRDiff` `e5d2ecba` | Zenodo 20586708 (195MB+536MB) | ✅ **新增 2026 扩散 baseline** |
| **TCR-epiDiff** | Bioinformatics **2025** | Setting B: epitope→CDR3β | **DDPM(U-Net+ProtT5-XL 条件)** | `seoseyeon/TCR-epiDiff` | 仓库内 `Model/*.pth`（本地已有 12MB） | ⛔ **仅引用**（官方生成代码非 turnkey，见 §0.3 + §7；不 reimplement） |
| GRATCR / ER-Transformer | 2024 / 2023 | Setting B | BERT+GPT / enc-dec | 官方预存预测 | `benchmark_data_w_preds.csv` | ✅ 经 TCRT5 benchmark 文件 |
| OLGA / SONIA / soNNia | 2019 / 2021 | Setting A: 无条件 CDR3β + pgen | VDJ 重组 / 后选择 | `statbiophys/*` | bundled `human_T_beta` | ✅ Setting A 参照 |
| LSMTCR | arXiv 2509.07627 (2025) | Setting C: 全长 α/β | Epitope-BERT+GPT+基因 Transformer | ❌ 无公开仓库 | ❌ | ⛔ 仅引用（不可复现） |
| TCRGen | bioRxiv 2025 (Lee-CBG) | Setting B: ICL 新表位 | in-context LM | `Lee-CBG/TCRGen` | ❌ "soon"(未发布) | ⛔ 仅引用（权重未发布） |

### 0.2 为什么"主构建基础"选 TCRT5（而非替换为更新的 2026 方法）

早期 T4 曾用多篇文献拼装的口径（recovery/exact/JSD），主指标非任一权威协议，baseline 只有 training-free 的 Markov/PWM/CondPool（recovery≈0）。改以 TCRT5 为锚：

- **它是候选里唯一的 turnkey 可复现基准**：同时提供①官方 held-out 数据划分（held20 target-rich + 14 sparse benchmark pMHC）②其他方法（GRATCR/ER-Transformer）**官方预存预测**③官方评测代码 `src/evaluation.py`。其余 2026 方法给的是"方法+权重"（可作 baseline），但没有打包好的"基准协议/划分/竞品预测"。
- baseline 与我们模型可在**同一 epitope 集 + 同一指标实现**下比；能直接 import 官方预存预测做**口径交叉验证**（§6，42 block 0 mismatch）。
- TCRT5 已完整 clone（`baselines/TCRT5_tcr_translate/`，commit `97aeab1008`），评测实现可逐函数照搬。

### 0.3 为什么"补" TCRDiff（而 TCR-epiDiff 降级为仅引用）

我们的 BioSeq 是 **masked/离散扩散**语言模型。为给它**最公平、最当代**的对照，Setting B 主榜接入**同为扩散**的近期工作 **TCRDiff（2026，离散扩散 DPLM，条件 pMHC+V 基因，MAGE-A3 体外验证）**（✅ 已复现 benchmark14）。

**TCR-epiDiff（2025，DDPM，U-Net，ProtT5-XL 表位条件）降级为"仅引用（cite-only）"**：本地虽有官方权重 `Model/TCR-epiDiff_Best_model.pth`，但**官方生成代码并非 turnkey-reproducible**（端到端核对结论，见 §7 blocker）——

1. `Tutorial_Code/TCR-epiDiff/TCR-epiDiff_main.py` 顶部 `from TCR-epiDiff_model import *` 是**非法模块名**（带连字符不能 import），脚本本身无法直接运行；
2. 依赖**未随仓库发布**的硬编码绝对路径 pkl 输入 `/tcr_epitope_peptide.pkl`、`/CDR3_epi_onehot_emb_covid.pkl`，作者自注"数据太大需自建"；
3. 采样是 **denoise-from-a-real-TCR**（对已存在 TCR 先 `add_noise` 再去噪），需要真实 TCR one-hot 种子 batch，**并非 clean de-novo 表位→CDR3β 生成**；
4. `UNet1D(in_channels=4)` 使用**非显然的 4 通道（核苷酸/密码子）编码**，其构造仅在 notebook 隐式给出；且 `Data_Processing.py` 的 one-hot 构造函数引用了**从未定义**的 `nucleotide_to_index`（另一处 broken reference）。

跑通它需要**重写模型 I/O + 预处理管线**，违反项目铁律"仅用官方代码、不 reimplement 模型/预处理"。因此按与 **LSMTCR/TCRGen** 一致的方式**只在文献表引用、标注不可复现，绝不编造其数字**。**Setting C** 由 **TcrDesign（2026）**承担全长 epitope-specific SOTA。

---

## 1. 三种 setting

| Setting | 任务 | 数据 / eval | 指标 | baseline | Ours-BioSeq |
|---------|------|-------------|------|----------|-------------|
| **A 无条件** | CDR3β repertoire 分布匹配 | OTS paired-clean：`data/tcr_generation/{train_cdr3b,holdout_cdr3b}.txt`（200k train + 9837 holdout） | k-mer JSD↓ · novelty · NN 距离 · **OLGA pgen 校准** | OLGA(human_T_beta) / soNNia / TCRT5-uncond | uncond CDR3β 采样 |
| **B 表位条件** ★主榜 | 给定 epitope 生成 binder CDR3β | TCRT5 held-out **20 pMHC**（`topk_val_df.csv`，target-rich）+ **14 sparse benchmark pMHC**（`benchmark_data_w_preds.csv`） | TCRT5 套件：Precision@K / Recall@K / F1@K / native seq-recovery / Char-BLEU / mean edit dist / diversity | TCRT5(HF beam / 官方预存) / **TCRDiff(2026 扩散)** / TcrDesign-G(beta) / GRATCR·ER(官方预存) / OLGA·soNNia(uncond 下限)（TCR-epiDiff 2025 扩散仅引用，官方生成代码非 turnkey） | 表位条件 CDR3β 设计 |
| **C 全长 α/β** | 全长 TCR α/β 组装 | 同 Setting B 的表位集（条件生成）；无条件全长看 validity | 全长 validity（合法 AA + CDR3 可提取率）→ 提取 CDR3β 后套 Setting B 指标 | **TcrDesign 全长 pipeline**（`tcrdesign.py`）作 SOTA（已跑通：F1 0.061 / seq-rec 0.772） | 无条件全长 α/β（OTS 分布）+ CDR3β 提取 |

---

## 2. 主评测集（对齐 TCRT5）

`scripts/prepare_tcr_generation_bench.py` → `data/tcr_generation_bench/`：

- **held20**（`topk_val_df.csv`）：TCRT5 的 held-out top-20 target-rich pMHC，共 92674 对；每个 pMHC 的 ref binder = 该 pMHC 下全部去重 CDR3β（数百~上万条）。这是"多样本"评测（recall@k = unique_correct / min(k, n_ref)）。
- **benchmark14**（`benchmark_data_w_preds.csv`）：TCRT5 的 13 sparse benchmark epitope（文件含 14 pMHC，含 RTATKAYNV / RTATKQYNV 两个近似表位），每 pMHC 11~895 条 ref binder，**并存有官方 TCRT5 / GRATCR / ER-Transformer 预测**（用于交叉验证 + 官方 baseline 行）。
- **conditioning/train pool**（`train_binders.csv`）：TCRT5 topk train 的 (epitope, cdr3b)，324692 对 / 1583 表位，用于 novelty 参照。
- `eval_conditional.json`：34 pMHC 的 {epitope, allele, mhc_pseudo, category, ref_binders, n_ref}。

HLA pseudo-seq 取自 TCRT5 `refs/hla_pseudo_seqs.csv`（34aa NetMHCpan 伪序列），TCRT5 条件输入格式 `"[PMHC]{peptide}[SEP]{mhc_pseudo}[EOS]"`。

---

## 3. 防泄露口径（核心卖点 · 必读）

`leakage_report.json`（`scripts/prepare_tcr_generation_bench.py` 生成）核查 eval 表位/binder 是否出现在 **BioSeq 训练语料**（`grammar_v2_esmc300m_cmp500k_llada`，sources `oas,ots,tcr,ppi`；只有 `tcr` 源携带表位-CDR3β 配对）与 TCRT5 conditioning pool。结论：

| 视图 | 表位在 BioSeq 训练集 | ref binder 作为 (ep,cdr3b) pair 泄露 | CDR3β 任意处出现 |
|------|----------------------|--------------------------------------|------------------|
| **held20**（20 pMHC，68482 ref） | 18/20 | **47.9%** | ~51% |
| **benchmark14**（14 pMHC，1312 ref） | 7/14 | 19.7%（集中在 7 个"脏"表位，各 100%） | 24.3% |
| **benchmark14 干净子集**（epitope 未见，7 pMHC） | 0/7 | 0% | 0~18%（个位数为主） |
| ref binder 在 TCRT5 conditioning pool | — | **0**（与训练完全隔离） | — |

**处理规则（诚实标注，写进 RESULTS / downstream.md）：**

1. **BioSeq 主榜 = 干净集**：BioSeq 的 zero-shot 能力以 **`bioseq_unseen` 视图**（epitope 不在 BioSeq 训练集的 9 个 pMHC = 7 benchmark14 + 2 held20）为准；`benchmark14` 整体次之。
2. **held20 对 BioSeq 是"记忆/seen"对照**：BioSeq 在 held20 的 precision/recovery **受 ~48% 训练泄露污染、偏乐观**，必须显式标注，不作为 BioSeq 的能力主张。
3. **TCRT5 / GRATCR / ER / TcrDesign 不是我们训练的**：held20 / benchmark14 对它们仍是各自论文口径的 held-out，可照常报（held20 尤其是 TCRT5 论文的官方 held-out split）。
4. harness 自动输出 5 个聚合视图：`overall / held20 / benchmark14 / bioseq_seen / bioseq_unseen`，便于同时看"论文口径"与"防泄露口径"。

---

## 4. 统一 harness

`tcr_generation_bench/run.py --setting {A,B,C} --method {tcrt5,olga,sonnia,tcrdesign,file}`：

- `tcrt5`：HF 权重（`dkarthikeyan1/tcrt5_ft_tcrdb`，经 `HF_ENDPOINT=https://hf-mirror.com` 下载），Setting B beam search / Setting A 采样。
- `olga` / `sonnia`：`baselines/OLGA` human_T_beta 无条件采样 + pgen；Setting B 下作 epitope-agnostic 下限（同一 pool 打到每个表位）。
- `file`：读统一 samples 文件（BioSeq adapter 或任意外部生成器）。
- `tcrdesign`：读 TcrDesign wrapper（独立 env）产出的 samples 文件。

统一 samples 格式：
- 条件（B/C）：JSONL，每行 `{"pmhc": "...", "epitope": "...", "sequences": [...]}`（可选 `"greedy"`）。
- 无条件（A）：`.txt`（每行一条 CDR3β）或 JSONL 带 `"sequences"`。

指标：`common/metrics.py` 的 `tcrt5_precision_at_k / tcrt5_recall_at_k / tcrt5_f1_at_k / tcrt5_mean_edit_distance / tcrt5_mean_sequence_recovery / tcrt5_sequence_bleu / tcrt5_dataset_metrics`，与官方 `src/evaluation.py` 逐函数对齐。

---

## 5. BioSeq 采样 adapter

`downstream/grammar/tcr_generation.py`（复用 `run_grammar_generate` + `grammar_v2_esmc300m_cmp500k_llada/best.pt`），扩展 `downstream/grammar/masks.py` 新增 `tcr_generation_partial_mask` / `cdr3b_span_partial_mask`（按 `position_ids_chain` 定位 tcr_beta / tcr_alpha 链）。三模式：

- `conditional`（Setting B）：record `[antigen(fixed epitope), tcr_beta(placeholder)]`，task `tcr_epitope` → grammar 把 peptide 作 fixed context、生成 β 链 = CDR3β。
- `uncond`（Setting A）：record `[tcr_beta(placeholder)]`，task `tcr` → 无条件采 CDR3β。placeholder 长度按 OTS CDR3β 经验长度分布采样，让模型自控长度。
- `fulllength`（Setting C）：record `[tcr_alpha(~113), tcr_beta(~114)]`，task `tcr` → 无条件全长 α/β（**BioSeq 的 tcr 源只有 CDR3、ots 源才是全长且无表位**，故 BioSeq 只能做无条件全长；epitope 条件全长由 TcrDesign 承担）。

**能力边界（诚实）**：BioSeq 的表位条件生成是 CDR3β 级（其强项）；全长仅无条件（OTS 分布），不做 epitope 条件全长。CDR3β 提取用 ANARCI（IMGT 104–118），失败退化为 `C…FGxG` 锚定正则。

---

## 6. 口径交叉验证

`scripts/test_tcrt5_metrics.py`：
1. **gold 单测**：从官方 `tests/test_evaluation.py` 逐条搬来的精确期望值（precision=2/3、recall、f1、edit=mean[0,2,1]、seq_recovery=mean[1,5/7,1/2]、find_n_closest、char-bleu 单调性）——全 PASS。
2. **对官方实现逐函数比对**：把 `benchmark_data_w_preds.csv` 的官方 TCRT5/GRATCR/ER 预测同时喂给我们 ported 指标与官方 `ModelEvaluator` 静态方法，**42 个 (epitope×method) block 全部 0 mismatch**（tol 1e-9）。

结论：我们的指标实现与 TCRT5 官方逐位一致，可放心用于全体 baseline + Ours 的排行榜。

---

## 7. baseline 清单（官方代码/权重）

| baseline | 仓库 / 权重 | commit | setting | 状态 |
|----------|-------------|--------|---------|------|
| TCRT5 | `dkarthikeyan1/tcrt5_ft_tcrdb`（HF）+ `baselines/TCRT5_tcr_translate` | 97aeab1008 | A/B | ✅ benchmark14 官方预测；held20 HF beam |
| GRATCR / ER-Transformer | `benchmark_data_w_preds.csv` 官方预存预测 | — | B | ✅ benchmark14 |
| **TCRDiff** (2026 扩散) | `zhangyumeng1sjtu/TCRDiff` + Zenodo 20586708（ckpt 195MB + data 536MB） | e5d2ecba | B | ✅ benchmark14（`wrappers/tcrdiff_generate.py` + `scripts/run_tcrdiff_baseline.sh`） |
| **TCR-epiDiff** (2025 扩散) | `seoseyeon/TCR-epiDiff`（本地 `Model/TCR-epiDiff_Best_model.pth` 12MB + 源码） | d7b726ff | B | ⛔ **仅引用**：官方生成代码非 turnkey-reproducible（`from TCR-epiDiff_model import *` 非法模块名；硬编码未发布 pkl `/tcr_epitope_peptide.pkl`·`/CDR3_epi_onehot_emb_covid.pkl`；采样为 denoise-from-real-TCR 非 de-novo；4 通道核苷酸编码 + 未定义 `nucleotide_to_index`）。跑通需 reimplement 预处理/模型 I/O，违反"官方代码 only"铁律，故不接入、不编造数字。wrapper `wrappers/tcr_epidiff_generate.py` 保持 cite-only 骨架（`--check-only` 返回 BLOCKED）。 |
| OLGA | `baselines/OLGA` human_T_beta | 1104df2dec | A/B下限 | ✅ |
| soNNia/SONIA | `baselines/SONIA` | acfb902 | A/B下限 | ✅ 官方 `generate_sequences_post` + bundled `human_T_beta` 权重 |
| TcrDesign | `XSLiuLab/TcrDesign` + Zenodo `tcrdesign_weights.tar.gz`（2.48GB, rec 14545852） | 668416a02e | B/C SOTA | ✅ Setting B beta（`wrappers/tcrdesign_generate.py`）；Setting C full ✅（34 pMHC k=100：F1 0.061 / seq-rec 0.772 / d_edit 2.58） |
| BioSeq (Ours) | `output/grammar_v2_esmc300m_cmp500k_llada/best.pt` | — | A/B/C | ✅ |

---

## 8. 环境与网络

- 主 env `protenix_abtcr`（torch2.8 + sklearn + 单 GPU），GPU 任务前置 `LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH`。补装 `Levenshtein / nltk / sentencepiece / mhcgnomes`（TCRT5 eval 依赖）。
- HF：直连不通，走 `HF_ENDPOINT=https://hf-mirror.com`（已验证可下 TCRT5 权重）。Zenodo / PyPI 直连可用（无需 socks5 代理）；Zenodo 单连接很慢（~87KB/s 甚至停滞），大权重统一用 `aria2c -x16 -s16` 多连接。
- TcrDesign 独立 env（`environment.yaml`：py3.8.16 / torch1.12 / sklearn1.2.2）+ Zenodo 权重解压到 `TcrDesign/weights/`。一键：`bash scripts/download_tcr_generation_baselines.sh` → `bash scripts/run_tcrdesign_baseline.sh beta all 100`.
- **TCRDiff** 复用主 env（torch2.8）+ `baselines/TCRDiff/_extra_deps`（rotary_embedding_torch / warmup_scheduler）。一键：`bash scripts/run_tcrdiff_baseline.sh benchmark14 100`.
- **TCR-epiDiff** ⛔ **仅引用**（官方生成代码非 turnkey，见 §0.3 / §7）：不建独立 env、不下 ProtT5-XL、不接 wrapper 生成路径；wrapper `wrappers/tcr_epidiff_generate.py` 为 cite-only 骨架（`--check-only` 返回 `BLOCKED: cite-only ...`）。本地权重 `Model/TCR-epiDiff_Best_model.pth` 仅作 provenance 保留。
- **GPU 重活 / 长跑 / 大下载优先提交 Volc**（`volc ml_task submit`，走 `/root/.codex/skills/volc-no-proxy/scripts/volc-no-proxy.sh`）：BioSeq Setting C 全长生成已提交（task `t-20260705205817-8mvdh`，`eval_jobs/eval_tcr_generation_setC_bioseq.yml`）。
- 复现命令见 `RESULTS.md` T4 节。

---

## 9. TCRT5 完整评测重建（2026-07-23）

完整实现与命令：

```bash
/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_tcrt5_full_eval.py \
  all --with-pgen --reference-pgen --pgen-workers 16
```

评测不再只停留在早期的 P/R/F1/edit/recovery port。当前完整覆盖 Char-BLEU、paper-native recovery、exact P/R/F1、Hit@K/rank distribution、论文 AP 定义、global unique、pairwise Jaccard、positional entropy、`k=2..12` natural-log JSD、OLGA Pgen、长度分布、≥90% identity、GIANA 和 polyspecificity。

作者 `20×100` release 的 240 个原子块在 P/R/F1/edit/recovery/Char-BLEU 上全部精确回归。Supplementary Table 2 非 mAP 为 `68/72` 格在显示精度内一致；四格差异来自作者发布 CSV 与论文表之间，而不是 evaluator。主表 mAP 缺 cumulative likelihood，不能精确复现；当前 Track-A TCRT5 保留了该 score，所以当前 TCRT5 的 AP 可按论文 rank 定义计算。

Fig.3–5 可对应项包括：length、181 binders、polyspecificity `r=-0.957638`、entropy-loss position、k-mer curve、sparse13 exact=2、RVR exact=8/ranks 和 GIANA `t=231,c=23`。OLGA non-zero=`1996/2000` 对应，但公开候选的 Pgen moments 不能同时还原论文四个分布统计，已在 sensitivity 表中保持为明确复现边界。

当前 all-14 结果的关键行：

| Model | Exact | paper-native ≥90% | GIANA hit | Global unique | mean Jaccard | Pgen+ | mean log10 Pgen+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| BioSeq-7L-step117000 | 0 | 0 | 1 | 13,988 | 0.000044 | 0.9798 | -17.9253 |
| GRATCR | 0 | 4 | 4 | 629 | 0.067642 | 1.0000 | -6.8664 |
| TCRT5 | 10 | 154 | 335 | 7,788 | 0.107176 | 1.0000 | -6.9504 |
| TcrDesign | 4 | 142 | 402 | 13,083 | 0.007130 | 0.9928 | -8.8010 |

权威报告：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/TCRT5_FULL_EVAL_REPORT.md`。机器对应表：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/paper_correspondence.csv`。回归测试：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/test_tcrt5_full_eval.py`，当前 `8/8 passed`。
