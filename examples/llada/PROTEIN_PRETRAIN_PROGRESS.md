# LLaDA 蛋白预训练（OAS/OTS）项目进度

> 实时记录本项目的目标、决策、进度与验证。每次有实质进展请在末尾「变更日志」追加带时间戳的条目。

- 负责入口脚本：[`examples/llada/protein_pretrain_esmc.py`](protein_pretrain_esmc.py)（融合正式跑）；[`protein_pretrain.py`](protein_pretrain.py) 为早期无 ESMC 入口
- 入口导航 / 当前任务 / 提交与 resume 规则：[`README.md`](README.md) §「蛋白预训练（本项目）」
- 数据侧（数据源 / 去污 / 审计运行手册）：[`DATA_PIPELINE_README.md`](DATA_PIPELINE_README.md)
- 最近更新：2026-09-07（**151k / 44k 的 T4 + CDR 已本地跑出并回填 RESULTS**，见 §4.2.10 ——
  长跑到 151k 未改善 T4（common d_edit 26k 8.58 → 151k 8.80，微降），CDR 小幅上行但在噪声内，
  **全链 vs generated-only 打平**；两条 pairing 仍在 `queue012` 排队，**本轮未收口**。
  单卡评测排不上队时可本地跑，env 用 eval YAML 里的 `protenix_abtcr` 绝对路径。
  另：九条改投 `queue012` 见 §4.2.9；**`c20250601` 上旧闲时九条已于 09-07 07:20 CST 用 CLI 全部
  cancel 成终态 `Killed`**（`StopCustomTask` 这次放行了，见 §4.2.11），双跑覆盖风险解除；
  `CreateCustomTask` 是否恢复未测。平台上我们只剩 `queue012` 两条非闲时 pairing + 两条 8 卡训练。
  全链 2M 续训 `t-20260902021013-bm79q` 已于 09-06T19:37Z **Killed**，盘上最新满包 161000，待决策。
  **人工 cancel 闲时任务前先 `touch output/_monitor/STOP`**）

---

## 0. 协作规则（Agent Rules）

> 本项目所有 agent（主控与子代理）必须遵守：

1. **实时更新进展**：每当有实质进展（新增/修改代码、跑通验证、发现问题、做出决策），必须同步更新本文档 —— 更新「4. 当前状态」表，并在「8. 变更日志」追加一条带日期的条目。不要攒到最后再补。
2. **关键改动同时同步到就近的 README**：本文档是**全量**记录（含过程、失败、caveat），README 是**入口**（当前在跑什么、怎么跑、有哪些硬规则）。凡是改变「别人下次该怎么操作」的东西——新增/改名训练配置、换队列或资源类型、提交与 resume 的规则、任务 id 与状态——除了记本文档，还要更新 [`README.md`](README.md) 的「蛋白预训练（本项目）」节；数据侧的同类改动更新 [`DATA_PIPELINE_README.md`](DATA_PIPELINE_README.md) 与 [`scripts/data/README.md`](../../scripts/data/README.md)。**只写在本文档不算更新完**：拿不到上下文的人先看到的是 README。反之，纯过程性的细节（试错、实测明细）只留本文档，别把 README 灌成副本。
3. **子任务可直接用 grok 4.5**：拆分出的子任务（写代码、调研、批量修改等）可直接交给 subagent 执行，模型默认使用 **grok 4.5**（`cursor-grok-4.5-high-fast`）；主控模型负责统筹、拆解与 review，不需要为每个子任务重复征求许可。

---

## 1. 目标

在 **不改动 LLaDA 训练主干** 的前提下，让 `LLaDA-8B-Base` 在 **OAS(抗体) / OTS(TCR)** 免疫序列上做两种范式的预训练：

- `diffusion`：纯扩散加噪（每条样本随机 mask 比例）。
- `bert`：BERT 式固定比例 MLM（固定 `mask_prob`）。

约束：整体 pipeline 基于现有 LLaDA（`examples/llada` + `MDLMTrainer`），**只改数据加载、grammar 语法、tokenization**；权重继承 8B base。

## 2. 关键决策

| 议题 | 决策 |
|------|------|
| 训练范式落地 | 单入口脚本 + `--train_mode {diffusion,bert}`，两者共用 `MDLMTrainer`，靠 α 调度器 + `loss_weight_type` 区分，不新写 trainer |
| grammar | 直接复用现成 grammar v2 `GrammarRenderer`（不写精简版），数据源限定 ab/tcr |
| 词表/权重 | 保留 LLaDA 完整 BPE 词表；新增残基/结构 token 后 `resize`，并把 grammar 空间 id 重映射到 LLaDA 空间，从而完整继承 8B embedding/head/主干 |
| bert 加噪 | 经典 BERT MLM：选 `mask_prob`(默认15%) 位置，再 80% `[MASK]` / 10% 随机词 / 10% 原词保留；loss 算在全部选中位（`BERTMLMTrainer`） |
| 量化 | 禁用 `load_in_4bit`（会破坏 resize）；LoRA 采用「先 resize 再 peft」，`modules_to_save` 含 `wte,ff_out` 保证新 token 可训练 |
| attention_mask | 沿用 sft 的 `NoAttentionMaskWrapper`（padding 对双向注意力可见）；如需排除 padding 可后续切换 |

## 3. 改动范围

> ⚠️ 本节 2026-08-04 前的版本写着「只新增 2 个文件、未修改任何现有代码」，**早已失效**。
> 现役正式入口是 `protein_pretrain_esmc.py`（融合），`protein_pretrain.py` 只是早期无 ESMC 入口。

**新增文件**

| 文件 | 说明 |
|------|------|
| [`protein_pretrain_esmc.py`](protein_pretrain_esmc.py) | **现役正式入口**：ESMC 条件融合 + 七源数据 + diffusion/bert 双目标 |
| [`protein_fusion_model.py`](protein_fusion_model.py) | `LLaDAEsmcFusion` 模型、两种加噪、`RemapCollator`、词表扩展与重映射 |
| [`protein_pretrain.py`](protein_pretrain.py) | 早期无 ESMC 入口（`MDLMTrainer` + `BERTMLMTrainer`），保留作对照 |
| [`load_fusion_checkpoint.py`](load_fusion_checkpoint.py) | 从 FSDP checkpoint 还原融合模型（下游评测用），**尚未 git track** |
| [`DATA_PIPELINE_README.md`](DATA_PIPELINE_README.md) | 数据源 / 去污 / 审计运行手册 |
| [`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) | 本进度文档 |
| [`MULTI_CHAIN_RELATION.md`](MULTI_CHAIN_RELATION.md) | 多链关系调研 + 对照臂操作说明（2026-09-02） |
| `dllm/pipelines/qwen3_vl_arch/relation_aux.py` | cognate / chain-drop InfoNCE 与 PLL 工具 |
| `scripts/downstream/score_pairing_pll.py` | OAS holdout `p(L\|H)−p(L)` |
| `scripts/diagnostics/test_relation_aux.py` | 分链 t + aux 的 CPU 单测 |

**修改的既有文件**（与本任务直接相关，不再是"没碰过"）

| 文件 | 改动 |
|------|------|
| `dllm/pipelines/bioseq/datasets.py` | 八个 `*_row_to_record`、`with_exclusion_filter{,_multi}`、`load_exclusion_keys`（fail-fast）、`tcr_repertoire_row_to_record` |
| `dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py` | 2026-08-24 修 ESMC 条件流在迭代解码中泄露参考序列（见变更日志） |
| `scripts/data/tcr_native/*`、`scripts/data/dedup/*` | 语料构建、去污、新鲜度断言、下游泄漏审计 |
| `scripts/count_immune_{mix,drops}.py` | 改用 `DataArguments` 默认值，支持尾随 `field=value` 覆盖 |
| `dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py` | 分链 t 改为 per-row multinomial；`BioSeqDiffusionOutput.relation_aux_loss` |
| `examples/llada/protein_fusion_model.py` / `protein_pretrain_esmc.py` | `relation_aux` 训练旋钮与 wandb 回调 |

**`protein_pretrain_esmc.py` 主要组件**
- `build_immune_specs`：按 `--dataset_args` 的 `+` token 组装数据源，逐源叠加 benchmark 去污过滤与长度过滤。目前支持 8 个 token：`oas / ots / asd_antibody / asd_nanobody / trait / tcr_native / tcr_papers / tcr_repertoire`（`asd_nanobody` 有 builder 但**刻意不入 recipe**，下一版 scope 不含 nanobody/VHH）。
- `with_length_filter` / `_record_chain_lengths_ok`：在读 CSV 时丢弃超长行（renderer 与 collator 遇超长是**抛异常**而非跳过，会直接崩训练步）。
- `ImmuneBioSeqDataset`：CSV record → `BioSeqRecord`（带 per-chain role 与 relation）。
- `FusionTrainer`：`compute_loss` 走 `model(**inputs)`（保证 FSDP 梯度同步）、`prediction_step` 只取 loss、`evaluate` 在分源 eval 下按行数加权重组 `eval_loss`。
- `TopKValLossCheckpointCallback` + `slim_checkpoints`：保留 eval_loss 最低的 K 个 ckpt，并把非最新的瘦身成 weights-only。

## 4. 当前状态

| 任务 | 状态 |
|------|------|
| 接口/数据路径核对 | 完成 |
| 入口脚本实现 | 完成（`protein_pretrain.py` + 融合 `protein_pretrain_esmc.py` / `protein_fusion_model.py`） |
| Review（重映射完整性/权重继承/labels-mask） | 完成 |
| dry_run 自检 | 通过 |
| 小模型冒烟 diffusion / bert / diffusion+LoRA | 通过 |
| LoRA「先 resize 再 peft」修复 | 完成并验证 |
| 真实 8B 权重下载 + 端到端验证 | 完成 |
| **A. immune 四条正式训练（270m/8B × bert/diffusion）** | **全部跑满 50k，已终态**（见 §4.1） |
| **B. v3 双臂（七源、bert vs diffusion）** | 4 卡非闲时 diffusion 已到 **42k/50k**（§4.2.2）；闲时两条与 `queue012` diffusion 陆续起跑；**4 卡 bert 已补齐**（`t-20260830135524-qq7n5`，global 256，§4.2.5） |
| **C. 全链 diffusion（`--diffusion_all_chains`）** | 4 卡 50k 已 Failed，盘上最好点 **`checkpoint-33000` eval 0.6866**。8 卡 2M 续训 **Running** `t-20260902021013-bm79q`，盘上最新满包 **156000**（eval 0.6430），top-k 最好 **151000 eval 0.6326**。121000 表征/CDR Success；**151000 全套已闲时提交**，见 **§4.2.8** |
| **D. batch 定容** | 完成：**global 256 只能靠 ga 加倍**（4 × ga 16 × 4 卡）；per_device 8 实测 OOM，本机单源探测低估 33 GiB（§4.2.5） |
| **E. 空位符阻塞** | ✅ **已修（2026-08-31）**：`tcr_papers_v2/train.csv` 第 3046 行 `cdr3b` 已从 `ASSKVAARVP-TLKLS` 就地改为 `ASSKVAARVPTLKLS`（行数不变）；`assert_residue_alphabet.py` 全量复扫 21 个 split／约 880 万行只此一格，现为 pre-flight gate。经过见 §4.2.6 |
| 早期 ablation（8B ESMC 有/无、from-scratch 两条） | 已终态，仅作 lineage（见 §4.3） |
| 数据扩充（`tcr_repertoire` 新源 + `tcr_papers_v2`） | 完成（2026-08-28，见 [`DATA_PIPELINE_README.md`](DATA_PIPELINE_README.md) §1） |
| 下游去重审计（5 源 × 11 基准） | 完成，`HARD-REQUIREMENT hits = 0 -> PASS` |
| T4 Setting-A 参考集修正 | 已生成修正版、**未重跑打分**（待决策，见 DATA_PIPELINE_README §6） |
| **F. 「BERT 表征为何一般」定位** | **诊断完成，无 bug；现象为真且可分辨**（raw 下 diffusion probe +0.033 = 5.1 sd、ARI +0.0092 = 7.6 sd，配对实测）。**成因分两种**：probe 上是几何造成的（各向异性 0.994 vs 0.986；方差均衡后 BERT 涨 +0.082 vs diff +0.045，差距抹平到 0.7 sd ⇒ **BERT 信息量相当**），**但 ARI 上的差距未被解释掉**（白化并未提升 BERT，只是弄坏了 diffusion −0.0052/4.1 sd）。🚫 「BERT 反超」「排序翻转」已撤回（噪声）。🟠 8B 三项反向 + 每格 n=1 训练 + ckpt 步数不同 ⇒ **不支持"哪个目标更适合表征"的结论**。重标定已评估并**决定不采用**（§4.7） |
| **G. v3 diffusion 全套下游** | generated-only 50k 已收口。全链 33000 / 18000 / 26000 已进 §0.8 组 D。本轮三最好点（全链 151000、generated-only 8 卡 2M 44000、BERT 1M 105000 只表征）：**T4 / CDR 四阶段已本地跑完回填**（§4.2.10）；两条 pairing 仍在 `queue012` 非闲时排队；**`c20250601` 闲时九条已于 09-07 全部 cancel（§4.2.11）**，BERT 105000 表征未跑 |
| **H. 8 卡长跑（步数拉长一个数量级）** | 进行中。全链 `..._allchains_..._8gpu_2m` **Running**（`t-20260902021013-bm79q`，最新满包 **156000** / 0.6430，最好 **151000 / 0.6326**）；generated-only `..._diffusion_immune_v3_8gpu_2m` **Running**（`t-20260904014842-qgjbg`，最新满包 **56000** / 0.7566，最好 **44000 / 0.7520**）；BERT `..._v3_1m` **Running**（`t-20260902110050-zb7dr`，最新满包 **108000** / 0.4288，最好 **105000 / 0.4263**）。spot_2m 已 Killed。**不打断续训**。见 §4.2.8 |
| **I. 🔴 磁盘配额是当前头号阻塞** | `..._8gpu_2m` 已因 `Disk quota exceeded (os error 122)` **4 连 Failed**，每次都死在 step 17000 的 `save_model`，净进度 0、约 3 小时 8 卡白烧。底层 FS 尚余 809T，爆的是目录配额。另发现 **top-k 账本会在重启时重置 → 孤儿 ckpt 永不被剪**（全 output 实测孤儿 132.9 G、resume-only 328.3 G）。见 **§4.2.7** |
| **J. 多链关系对照臂** | 用户确认走实验路径。分链 t multinomial 修复 + cognate InfoNCE 已接入；两条 4 卡 generated-only YAML **已写未提交**。**禁止改 2M 主跑**。见 **§4.9** / [`MULTI_CHAIN_RELATION.md`](MULTI_CHAIN_RELATION.md) |

### 4.1 immune 四条正式训练（已跑满 50k）

四条均已终态、`checkpoint-final` 齐全。现存 top-k（按 eval_loss 保留，2026-08-29 核盘）：

| Job | 现存 checkpoint | 备注 |
|-----|-----------------|------|
| `protein_esmc_llada270m_bert_immune` | 41000 / 44000 / 49000 / 50000 + final | 当前对外报的 270m BERT 数字来自 **49000** |
| `protein_esmc_llada270m_diffusion_immune` | 39000 / 42000 / 47000 / 50000 + final（另存早期 6000/8000/9000） | 当前对外报的 270m diffusion 数字来自 **42000** |
| `protein_esmc_llada8b_bert_immune` | 43000 / 44000 / 46000 / 50000 + final | 旧的 40000 已被 top-k 剪枝删除，**RESULTS 里引用 40000 的行不可复现** |
| `protein_esmc_llada8b_diffusion_immune` | 44000 / 45000 / 48000 / 50000 + final | 同上 |

> ⚠️ 这四条训练的语料是**六源 v2**（`tcr_papers` 指 v1 目录、无 `tcr_repertoire`），且跑在
> 2026-08-28 那批去污修复**之前**。它们与 §4.2 的 v3 不可直接比较。

### 4.2 v3 双臂：bert vs diffusion（2026-08-29 提交；4 卡 diffusion 已 Running，8 卡四条仍 Queue）

> 这一对存在的理由：v1/v2 的 **BERT 臂跑的是六源、diffusion 臂跑的是七源**，两个目标函数
> 根本不可比。v3 让两臂的数据参数**逐字节一致**（提交前用 `diff` 验证过），唯一差异是目标函数。

| Job | Task ID | 目标 | 状态 |
|-----|---------|------|------|
| `protein_esmc_llada270m_diffusion_immune_v3` | **`t-20260829031748-96vjf`** | diffusion（仅生成链） | **Running**（`queue012`，非抢占，排队 1.9 天后于 2026-08-30 起跑） |
| `protein_esmc_llada270m_bert_immune_v3` | **`t-20260829031757-2qnjn`** | bert MLM 0.15 / 80-10-10，`--bert_all_chains True` | Queue（`queue012`，非抢占） |
| `protein_esmc_llada270m_diffusion_immune_v3_spot` | **`t-20260829135424-zxdjg`** | 同上 diffusion | Queue（`c20250601`，**闲时**） |
| `protein_esmc_llada270m_bert_immune_v3_spot` | **`t-20260829135427-426cj`** | 同上 bert | Queue（`c20250601`，**闲时**） |
| `protein_esmc_llada270m_diffusion_immune_v3_4gpu` | **`t-20260830152319-r5w9k`**（原 `t-20260829143036-w7488`，经 `RetryOptions` 换过 3 次 id） | 同上 diffusion | 🔴 **卡在 42000/50000**：每次续跑都崩在 step ~42782 的同一个坏样本上，剩 2 次重试，见 **§4.2.6** |
| `protein_esmc_llada270m_diffusion_allchains_immune_v3_4gpu` | **`t-20260830135521-zf5rr`**（前两次作废：`t-20260830125655-hngmn` global 128 已 cancel；`t-20260830134534-86bml` per_device 8 OOM） | diffusion **全链**（`--diffusion_all_chains True`，无 fixed 链） | **Running**（`c20250601`，非抢占，4 卡，**global 256** = 4 × ga 16，见 §4.2.4 / §4.2.5） |
| `protein_esmc_llada270m_bert_immune_v3_4gpu` | **`t-20260830135524-qq7n5`**（前一次 `t-20260830134537-8ncgx` per_device 8 OOM） | 同上 bert（`--bert_all_chains True`） | **Running**（`c20250601`，非抢占，4 卡，**global 256** = 4 × ga 16，见 §4.2.5） |

- 共同配置：`--decoder_init scratch` d=768/L=8/h=12 ≈ 270M、可训 ESMC-300M、`residue_cond_mode=add`、
  `max_length=max_protein_length=1024`、`max_steps=50000`、
  lr 1e-4 cosine + warmup 2000、eval/save 每 1000、`save_top_k=3`、`slim_checkpoints`、FSDP 单节点
  （8 卡 `ml.pni2.28xlarge` / 4 卡 `ml.pni2.14xlarge`）、`ActiveDeadlineSeconds=604800`
  （队列、抢占、卡数各条不同，见 §4.2.1 与 §4.2.2）。
- 🔴 **batch 口径分两组，跨组不可比**（2026-08-30 起，详见 §4.2.5）：
  - **global 128 组（五条）**：8 卡版 per_device 2 × ga 8 × 8gpu；4 卡 generated-only 版
    per_device 4 × ga 8 × 4gpu。组成不同但乘积相同，五条互相可比。50000 步 ≈ 0.84 epoch。
  - **global 256 组（两条，均 4 卡）**：per_device 4 × **ga 16** × 4gpu ——
    `..._diffusion_allchains_immune_v3_4gpu` 与 `..._bert_immune_v3_4gpu`。这两条互相可比，
    但与上面五条不可比。50000 步 ≈ 1.64 epoch。
    **注意加倍走的是 ga 不是 per_device**：per_device 8 实测 OOM，原因见 §4.2.5。
- 数据：`oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire`，其中 `--tcr_papers_dir`
  指向 **`data/tcr_papers_v2/dataset`**。启动前调 `assert_corpus_fresh.py`。
- **两臂唯一的非数据差异**：BERT 用 `--bert_all_chains True`，loss 覆盖**全部**残基（含抗原 / MHC /
  peptide 这些固定上下文）；diffusion 只算生成链。这是有意的目标函数差异，写论文时要说明。
- **所有条首跑都是 from scratch**：v1/v2 的 checkpoint 训练在不同语料组成上，不可续；
  global 256 两条另有 batch 口径变化，128 口径的 optimizer state 也不可续。闲时版会在
  被抢占后从**自己的** `_spot` 目录续训（§4.2.1）；4 卡版也带同一段 resume 逻辑，但只为 `RetryOptions`
  的故障重试兜底（§4.2.2）；`queue012` 那两条非抢占版则带反 resume 断言、只允许从零跑。
- YAML 的 `Description` 只写十个字（平台上限 500 字符，写长了直接提交失败）。**关键改动一律记在本文档**，
  YAML 里只留一行指针注释。

#### 4.2.0 现状快照（2026-08-30 07:27 UTC 核盘）

数字取自各条最新 checkpoint 的 `trainer_state.json`（干净 JSON，比解析 wandb 二进制可靠），
步数取自 `ml_task logs` 的进度条。

| Job | 卡 | global | 步数 | 最新 `eval_loss` | 现存 ckpt | ETA |
|---|---:|---:|---|---|---|---|
| `..._diffusion_immune_v3_4gpu` | 4 | 128 | 🔴 **42000（卡死）** | **0.7548** @42000（前 0.7588） | 39000 / 41000 / 42000 | — |
| `..._diffusion_immune_v3_spot` | 8 | 128 | 15620 | 0.8150 @15000（前 **0.7999** ↑） | 11000 / 12000 / 14000 / 15000 | 13.7 h |
| `..._diffusion_immune_v3`（queue012） | 8 | 128 | 6900 | 0.8442 @6000（前 0.8604 ↓） | 4000 / 5000 / 6000 | 16.6 h |
| `..._bert_immune_v3_spot` | 8 | 128 | 5220 | 0.6024 @5000（前 0.6244 ↓） | 1000 / 3000 / 4000 / 5000 | 17.2 h |
| `..._bert_immune_v3`（queue012） | 8 | 128 | — | — | — | 仍 Queue，已排 2 天 |
| `..._diffusion_allchains_immune_v3_4gpu` | 4 | **256** | 1380 | 1.0831 @1000 | 1000 | 47 h |
| `..._bert_immune_v3_4gpu` | 4 | **256** | 1400 | 0.7249 @1000 | 1000 | 45 h |

> 🔴 **`eval_loss` 只能在三层口径都相同时比大小**，这张表里至少有三条切割线，
> 任何跨线的数值比较都是无意义的：
> 1. **目标函数**：BERT 算的是固定 15% mask 位置的 MLM 交叉熵；diffusion 算的是
>    每序列随机 \(t \sim U(\epsilon,1)\) 下的加权损失。BERT 看着低（0.60 vs 0.75）
>    **不代表更好** —— 分母、被评 token 数、任务难度全都不同。双臂对比只能走下游 benchmark。
> 2. **loss 覆盖范围**：`_allchains` 那条把固定上下文（抗原 / MHC / peptide）也计入，
>    集合更大更难（§4.2.4）。
> 3. **batch 口径**：global 128 与 256 两组（§4.2.5）。
>
> **组内、同一条 run 内是自洽的**，所以 top-k 选 checkpoint 完全可靠；跨组只看下游。

**一个待观察点**：`..._diffusion_immune_v3_spot` 的 `eval_loss` 从 0.7999 @14000 回升到
0.8150 @15000。cosine 调度中段单点回升不算异常，且 `save_top_k=3` 会自动保住 14000 那个更好
的点，但若 16000 / 17000 继续走高就该查一眼。

**提交时已知并接受的两个 caveat**（从 YAML 注释移来，避免 YAML 变成文档）

1. ~~`tcr_repertoire` 的近重复搬迁从未执行~~ **已于 2026-08-29 03:35 UTC 执行完毕**
   （见下方「caveat 1 的后续」）。
2. `trait_benchmark_blocklist` 于 2026-08-28 17:25 重建，但 `extra_decontam_report.json`
   未同步更新，那一次过滤的 provenance 没有留痕。方向上是**更严**不是更松
   （trait 保留 34,872 → 31,515），故不构成泄漏风险。

#### caveat 1 的后续：搬迁已执行，且两臂仍可比

原计划是把搬迁留给 v4，理由是"两次提交之间改语料会让两臂不可比"。实际执行后这个风险
**没有发生**，因为搬迁完成时两臂都还在排队：

| 任务 | Task ID | 提交（UTC） | 搬迁完成时状态 |
|---|---|---|---|
| diffusion v3 | `t-20260829031748-96vjf` | 2026-08-28T19:17:49Z | **Queue**（排队 10.4h，`end=None`） |
| bert v3 | `t-20260829031757-2qnjn` | 2026-08-28T19:17:58Z | **Queue**（同上） |

两臂都尚未 launch、都没读过任何 CSV，所以启动后读到的是同一份搬迁后的语料，
**"可比"这个 v3 存在理由不受影响**。若当时有任一臂已在 Running，就必须回滚
（`*.csv.pre_neardedup` 备份可逐字节还原）。

搬迁结果：valid 201,504 → **123,049**（移走 78,455，**38.9%**），holdout 201,170 →
**122,669**（39.0%），train 1,971,794 → **2,128,750**（+156,956，+7.96%），5 轮迭代收敛。
重测 `count_immune_mix.py` 得该源 kept = raw（**剔 0**），并发会话的泄漏审计亦给出
`HARD-REQUIREMENT hits = 0 -> PASS`——搬回的 15.7 万行没有带进任何污染，它们本就与 train
同处一个已去污染的 pool。`assert_corpus_fresh.py` 通过（搬迁不触碰 blocklist）。

> ⚠️ train 因此超出当初刻意设的 2,000,000 上限 6.4%（残基占比 2.04%→2.19%，判为可接受
> 未回切）。**若重跑 `build_repertoire.py`，该上限会把搬回来的行重新截掉，须重跑搬迁。**

细节见 `data/tcr_repertoire/README.md`「valid 近重复搬迁」与
`downstream/benchmark/audit_2026_08_27/RETRAIN_PLAN.md` §7.6。

#### 4.2.1 闲时（抢占）版双臂 `_spot`（2026-08-29 05:54 UTC 提交）

`queue012` 上那两条非抢占任务排队 18.6 小时仍是 `Queue`，为不继续干等，在 **`c20250601`**
队列（`q-20260121145036-6fztt`）另提一对**闲时资源**版本（`Preemptible: true` = 借其他
队列的空闲资源），配置文件 `train_jobs/protein_esmc_llada270m_{diffusion,bert}_immune_v3_spot.yml`。

与非抢占版的差异**只有四处**，训练超参与数据参数**零差异**（提交前 `diff` 逐行核过；两条
spot 之间也只差目标函数那 5 行 + 名字，数据参数逐字节一致，v3 的可比性不受影响）：

| 改动 | 非抢占版 | 闲时版 |
|---|---|---|
| 队列 | `queue012` (`q-20260524172355-rnqtf`) | `c20250601` (`q-20260121145036-6fztt`) |
| 抢占 | `Preemptible: false` | `Preemptible: true` + `RetryOptions` |
| 输出目录 / `run_name` | `..._immune_v3` | `..._immune_v3_spot` |
| resume | `test ! -d checkpoint-1000`（反 resume 断言） | 自动接最新 checkpoint |

三处非平凡改动，各有必须这么做的理由：

1. **输出目录必须分开。** 两套任务同时排队，谁先拿到资源不确定。若共用 `OUTPUT_DIR`，
   两个进程会同时写同一批 `checkpoint-*` 并各自触发 top-k 剪枝，checkpoint 直接报废。
2. **反 resume 断言必须去掉。** 闲时资源随时被回收，`test ! -d checkpoint-1000` 会让第二次
   启动直接失败。改为在 entrypoint 里取编号最大的 checkpoint 传 `--resume_from_checkpoint`；
   目录里没有 checkpoint 时**不能**传该参数（HF Trainer 会因找不到而报错），故条件拼接：

   ```bash
   LATEST_STEP="$( (ls -d "${OUTPUT_DIR}"/checkpoint-[0-9]* 2>/dev/null || true) | sed 's#.*/checkpoint-##' | sort -n | tail -1 )"
   ```

   `|| true` 是必需的 —— entrypoint 开着 `set -euo pipefail`，首跑时 `ls` 无匹配会让整个脚本
   退出。四种场景已本地实测：目录不存在 / 目录空 / 多 checkpoint + `checkpoint-final` /
   只有 `checkpoint-final`，分别得到 from-scratch、from-scratch、接 `checkpoint-49000`
   （数字序而非字典序，且 `checkpoint-final` 被 glob 正确排除）、from-scratch。
   接得住的前提是 `TopKValLossCheckpointCallback` 保证**最新** checkpoint 始终带完整
   optimizer/FSDP state（非最新的被 slim 成 weights-only），加上 `save_only_model False`。
3. **自动重试必须自己开。** 提交时平台明确提示「原抢占任务被抢后的强制重试功能已下线」。
   故加 `RetryOptions: EnableRetry=true / MaxRetryTimes=50 / IntervalSeconds=180 /
   PolicySets=[Failed, InstanceReclaimed]`。`InstanceReclaimed` 即闲时资源回收，是本任务
   最主要的中断原因；`Failed` 一并覆盖 NCCL 超时、节点故障这类偶发失败。重试起的新任务
   重跑 entrypoint → 自动续训，**每次中断最多丢 `save_steps=1000` 步**。
   提交后用 `volc ml_task export -t <id> --config` 回读，确认平台侧确实记下了这段
   `RetryOptions`（CLI **无** update 子命令，改 YAML 不回写已提交任务，只能 cancel 重提 ——
   本次首轮提交 `t-20260829135133-5np6m` / `t-20260829135158-2wnsp` 就是漏了 `RetryOptions`
   而在 Queue 阶段 cancel 重提的，那两个 id 已 `Killed`，不要引用）。

> `queue012` 那两条**刻意保留**未 cancel：非抢占资源一旦起跑不会被抢，比闲时稳。代价是
> 四条可能都起跑、白跑一对。**哪一对先出 checkpoint 就 cancel 另一对**，别让两对都跑到 50k。
> 两对 `OUTPUT_DIR` 不同，并跑不会互相破坏，只浪费配额。

#### 4.2.2 4 卡非闲时版 `_4gpu`（2026-08-29 06:30 UTC 提交，**19 秒起跑**）

§4.2.1 的闲时版提交后 40 分钟仍 `Queue`。查 `c20250601` 队列实况，发现瓶颈**既不是抢占也不是优先级**：

| 规格 | Running | Queue |
|---|---|---|
| `ml.pni2.3xlarge`（1 卡） | 7 | 0 |
| `ml.pni2.28xlarge`（8 卡） | **0** | **7**（含我们两条，最早的已等 9.8h） |

队列上**没有任何 8 卡任务跑起来**——零散单卡资源有，凑不出连续整节点 8 卡。`Priority` 已是用户
可提交的最高档 6（平台只开放 2/4/6，更高留给队列管理员），加不上去。故降卡数换可调度性：提一条
**4 卡 `ml.pni2.14xlarge` + `Preemptible: false`** 的 diffusion 任务（`c20250601` 上 4 卡非抢占有
先例，见 `Protenix-v2/train_jobs/*_1node4gpu*.yml`）。**06:30:36 提交 → 06:30:55 Running。**

**global batch 保持 128 不变**是这条任务能与 8 卡四条直接比较的前提：卡数减半就把省下的乘数补进
`per_device`（而不去动 `ga` 或 `max_steps`），于是 lr schedule、总样本量、每个 optimizer step 的
语义全部不变。

| | 8 卡版 | 4 卡版 |
|---|---|---|
| `per_device_train_batch_size` | 2 | **4** |
| `gradient_accumulation_steps` | 8 | 8（不变） |
| GPU | 8 | **4** |
| global batch | 2 × 8 × 8 = **128** | 4 × 8 × 4 = **128** |

补在 `per_device` 而不是 `ga`（per_device 2 × ga 16 同样得 128）的理由：micro-batch 更大、GPU 利用率
更高。前提是显存够，**提交前在本机 A100-80G 单卡实测过**（无 FSDP、模型状态未分片，故为 4 卡 FSDP
的**上界**；每秒采样一次 `nvidia-smi`）：

| 阶段 | 峰值 | 说明 |
|---|---|---|
| 数据加载（七源各 256 行） | 2.5GB | t=0–210s，不占 GPU |
| 模型加载（ESMC-300M + 270m decoder） | 7.4GB | t=210–240s |
| **训练步**（`per_device=4`, `max_length=1024`） | **30.7GB** | t=240–480s，跑完 4 步 |
| **保存 checkpoint** | **56.4GB** | `train()` 之后 `save_model` 的一次性峰值 |

30.7GB 里约 9GB 是未分片的模型状态（603M 参数：bf16 参数 1.2GB + bf16 梯度 1.2GB + fp32 Adam m/v
4.8GB），4 卡 FSDP 会把这部分压到约 2.3GB，而激活那约 21GB 不随分片变 → **4 卡训练稳态约 23GB**。
保存阶段那 56.4GB 来自 `fsdp_state_dict_type: FULL_STATE_DICT` + `--save_only_model False`
（同一套保存路径在 8 卡上已跑满过 50k 步）。两个数都在 80GB 内且有余量。

实测跑到底、闭环确认：**`trainer exit code = 0`**，4 步全部完成（loss 24.38 → 15.91）、
`checkpoint-final` 含 1.2GB `model.safetensors` 保存成功，单卡无 FSDP 吞吐
`train_samples_per_second=2.37`。即 `per_device=4` 不只是显存够，前向反向与保存全链路都验证过。

其余差异：

- `OUTPUT_DIR` / `run_name` 加 `_4gpu` 后缀 —— 五条任务可能并跑，共用目录会互相覆盖 checkpoint。
- **保留** §4.2.1 那段自动 resume 逻辑。非抢占不会被闲时回收，但节点故障 / NCCL 超时触发
  `RetryOptions` 重试时，新实例重跑 entrypoint 就能接上最新 checkpoint。
- `RetryOptions` 只留 `PolicySets: ["Failed"]`（非抢占，无需 `InstanceReclaimed`），
  `MaxRetryTimes` 从 50 降到 5 —— 故障是偶发的，不像抢占那样反复发生。
- `num_processes` 由平台注入的 `MLP_WORKER_GPU` 决定，改 flavor 自动生效，**卡数未硬编码**。

提交后 `volc ml_task export -t t-20260829143036-w7488 --config` 回读确认：flavor
`ml.pni2.14xlarge` × 1、`per_device_train_batch_size 4`、`gradient_accumulation_steps 8`、
`RetryOptions` 都已记入平台侧。

> **代价与待办**：卡少一半、步数与样本量不变，wall-clock 约慢一倍。且这条**只有 diffusion 臂**——
> bert 臂目前仍只有 8 卡版在排队，而 v3 的核心价值就是双臂对比，若这条跑顺应补一条 4 卡 bert 版。

#### 4.2.3 闲时任务看护循环（2026-08-29 06:52 UTC 起常驻）

**为什么 `RetryOptions` 不够。** §4.2.1 配的 `RetryOptions`（`InstanceReclaimed` +
`MaxRetryTimes=50`）只覆盖「实例被回收后平台自动重试」。重试次数用尽、或任务因别的原因落到
`Failed`/`Killed` 终态之后，平台就彻底不管了，训练会永久停在那里而没人知道。看护循环补的是这一层。

| 文件 | 作用 |
|---|---|
| `scripts/monitor_spot_tasks.py` | 看护主体，零第三方依赖（只用标准库，系统 `python3` 即可跑） |
| `scripts/monitor_spot_tasks_start.sh` | 后台常驻启动（PID 互斥 + 崩溃自愈 wrapper） |
| `scripts/monitor_spot_tasks_stop.sh` | 停止（先放 STOP 哨兵再杀进程） |
| `output/_monitor/monitor.log` | 日志 |
| `output/_monitor/state.json` | 当前追踪的 task id、重提次数、重提历史 |

**决策逻辑**：每 600 秒查一次 `volc ml_task list --name <TaskName> -o json`，只看**最新一条**
同名任务的 `Status`：

| 状态 | 动作 |
|---|---|
| `Queue` / `Staging` / `Running` / `Killing` / `Initialized` | 不动作（`Running` 时顺带查停滞） |
| `Success` | 标记 `done`，停止看护该条 |
| `Failed` / `Killed` | **用同一个 YAML 重新提交** |

重提即续跑：YAML 的 entrypoint 里有 `RESUME_ARG` 逻辑（§4.2.1），新实例会自动挑 `OUTPUT_DIR`
下编号最大的 checkpoint 接上，所以每次中断最多丢 `save_steps=1000` 步。

**几个不能省的设计，每个都对应一个真实的坑**：

1. **必须只看最新一条。** `volc ml_task list --name` 是**模糊匹配**，且会返回历史终态任务 ——
   实测查 `protein_esmc_llada270m_diffusion_immune_v3` 会同时返回 `_4gpu`、`_spot`、原版，
   其中还有 08-29 cancel 重提留下的 `t-20260829135133-5np6m`/`Killed`。所以脚本先按
   `JobName` 精确比对，再按 `JobId` 降序取第一条（`JobId` 形如 `t-<YYYYMMDDHHMMSS>-xxxxx`，
   时间戳在前，字符串降序即时间倒序）。不这么做就会被历史记录骗到、无限重提。
2. **STOP 哨兵。** `Killed` 既可能是被抢占，也可能是**我们主动 cancel**（比如 8 卡起跑后
   cancel 闲时这对），脚本无从区分。所以人工 cancel 前必须先
   `touch output/_monitor/STOP`（全局）或 `STOP.<TaskName>`（单条），否则会被重提回来。
   `start` 脚本启动时会清掉上次留下的哨兵。
3. **重提冷却 15 分钟 + 上限 100 次。** 防止任务一起来就失败时刷出一堆提交。
4. **停滞只告警、不自动 kill。** `Running` 但 wandb 超过 1 小时没写入会打 `WARN`。用 wandb
   的 mtime 而非 checkpoint 判断，因为 `logging_steps=20` 比 `save_steps=1000` 灵敏得多。
   不自动处理是因为 eval（每 1000 步、七源各 2000 行）本身会拉长写入间隔，误杀代价大于漏报。
   顺带一个背景：训练日志里 torch 自己会打
   `Detected kernel version 5.4.250, which is below the recommended minimum of 5.5.0;
   this can cause the process to hang` —— 平台内核的既有条件，改不了。历史上 8 卡任务确实
   跑满过 50k 步，所以它只是风险提示而非已知故障，但也说明停滞检测这条不是杞人忧天。
5. **看护自身的崩溃自愈。** `start` 脚本外套一层 wrapper：python 非 0 退出则 10 秒后拉起，
   `exit 0`（STOP 或全部 `Success`）才收工。已用 `kill -9` 实测，日志留下
   `[ERROR] 看护异常退出 rc=137，10 秒后自动拉起` 后成功拉起新 pid。
6. **PID 文件互斥。** 起两个看护会让同一个终态任务被重提两次。实测第二次启动被正确挡住。

**测试**：13 组 25 个断言全部通过，覆盖 `Killed`/`Failed` 触发重提、冷却期内跳过、冷却后重提、
次数上限、STOP 哨兵、dry-run、五种非终态不动作、`Success` 后即使再见 `Killed` 也不重提、
查询抛异常不崩、查不到任务不自动提交，以及 `latest_checkpoint` 的数字序排序与排除
`checkpoint-final`、`output_dir_of` 对两个真实 YAML 的解析。

> 看护**只管 `_spot` 那两条**（`TARGETS` 常量里写死）。`queue012` 的非抢占版带反 resume 断言、
> 只允许从零跑，重提反而会覆盖；4 卡那条非抢占、有自己的 `RetryOptions`，也不需要看护。

#### 4.2.4 全链 diffusion 变体 `_allchains`（2026-08-30 04:56 UTC 提交，起跑）

**动机。** v3 两臂在「loss 覆盖哪些残基」上本来就不对称：BERT 用 `--bert_all_chains True`
算**全部**残基，diffusion 只算生成链，固定上下文（抗原 / MHC / peptide）永不加噪也不计 loss。
本条把 diffusion 也放开到全链 —— **没有任何链是 fixed 的** —— 于是「全链 vs 仅生成链」
成为可单独测量的一个因子，而不再和目标函数绑在一起。

| Job | Task ID | loss 覆盖 |
|---|---|---|
| `..._diffusion_immune_v3_4gpu` | `t-20260829143036-w7488` | 仅生成链（历史行为） |
| `..._diffusion_allchains_immune_v3_4gpu` | **`t-20260830135521-zf5rr`** | **全部残基（含固定上下文）** |

**代码改动**（新增开关，默认 `False`，现存 checkpoint 行为逐位不变）：

| 文件 | 改动 |
|---|---|
| `examples/llada/protein_fusion_model.py` | 新增 `all_residue_eligible_mask()`；`LLaDAEsmcFusion` 加 `diffusion_all_chains` 参数并写入 `config` |
| `examples/llada/protein_pretrain_esmc.py` | `TrainingArguments.diffusion_all_chains`（CLI `--diffusion_all_chains`）、透传给模型、启动日志打印该值 |

实现只有一处语义改动：diffusion 分支在采样前把 `diffusion_eligible_mask` 与
`diffusion_loss_mask` 覆写成 `residue_mask & attention_mask`。时间步采样器与加噪采样器
**都**从这两个 mask 重新推导可加噪集合，所以覆写一次就同时改到了加噪、labels 和
ESMC 编码器镜像三条路径。**目标函数本身没动**：仍是每条序列一个 `t ~ U(eps, 1)` +
Bernoulli(t) 吸收态 mask，bf16 / lr / loss_norm / GIDD 与 Ophiuchus 比例全部不变。

> 顺带一提，默认 `joint_loss_ratio=1.0` 下 `sample_chain_conditioned_timesteps` 给全部
> eligible token 发同一个 `t`，所以「谁是 heavy / 谁是 light」的角色判定即使因为抗原进入
> eligible 集合而改变，也不影响任何数值 —— `heavy_mask`/`light_mask` 只在
> `heavy_loss_weight != 1.0` 时才被读。

**本机 A100 实测验证**（ASD batch，4 条，`asd_antibody` 有真实抗原上下文）：

| | 计入 loss 的位置 | 其中固定上下文 | 被加噪的固定上下文 | 编码器被 mask 的位置 |
|---|---:|---:|---:|---:|
| `diffusion_all_chains=False` | 659 | **0** | 0 | 647 |
| `diffusion_all_chains=True` | 1452 | **805** | 805 | 1452 |

该 batch 共 2,364 个残基位，其中 1,446 是固定上下文。开关打开后固定上下文确实被加噪并计入
loss，且**编码器侧 mask 数与计入 loss 的位置数一致（1452）**，即 ESMC 条件通路没有给出
任何一个待预测残基的干净拷贝 —— 这条最关键，否则全链 loss 会退化成抄答案。两侧
`params_with_grad=380`、loss 有限，反向正常。

**提交与回读。** 与 `_4gpu` 那条 `diff` 只差 TaskName / 注释 / `OUTPUT_DIR` / `run_name` /
Tags、`--diffusion_all_chains True` 以及 `per_device_train_batch_size`（见下）；其余训练超参
与数据参数逐字节一致（4 卡 `ml.pni2.14xlarge`、非抢占）。提交前 `assert_corpus_fresh.py`
两源 PASS；`--dry_run True` 确认新 flag 解析为 `'diffusion_all_chains': True`。
`volc ml_task export -t <id> --config` 回读确认平台侧记下了该 flag、`per_device`、`ga`、
flavor 与 `RetryOptions`。

> **首跑必须 from scratch**：loss 覆盖范围变了，generated-only 的 checkpoint 不可续。
> entrypoint 保留自动 resume，只为 `RetryOptions` 的故障重试兜底。
> `OUTPUT_DIR` 带 `_allchains` 后缀，与另外三套 diffusion 目录完全隔离。

**首跑（`t-20260830125655-hngmn`，global batch 128）已作废，见 §4.2.5。** 它跑到 step 1000
就被 cancel 并换成 global batch 256 重提（现役 `t-20260830135521-zf5rr`），输出目录整体改名为
`output/ABORTED_gb128_protein_esmc_llada270m_diffusion_allchains_immune_v3_4gpu`（9.1G，可删），
这样新 run 的 `RESUME_ARG` 探测不到 checkpoint、必然 from scratch —— 否则会拿 128 口径的
optimizer state 接 256 的 schedule。

那 1000 步不是白跑，它是新开关的**端到端通路验证**：train loss 96.88 → 5.21，
`eval_loss` **1.1463**，checkpoint-1000 落盘且 `topk_val_manifest.json` 标记 `resumable=true`
—— train → eval → save 在新开关下全部跑通。分源：`ots` 0.497 / `tcr_native` 0.554 /
`oas` 0.610 / `tcr_papers` 1.289 / `trait` 1.457 / `tcr_repertoire` 1.839 /
`asd_antibody` **1.872**（基线同步 0.61 上下，这里最高 —— 正是因为抗原上下文现在也要预测）。

> 🔴 **两条臂的 `eval_loss` 数值不可直接比大小。** 全链版的 loss 是在**更大且更难**的
> token 集合上平均的（多了抗原 / MHC / peptide），分母和被评的内容都变了。
> generated-only 那条 step 1000 是 **1.0461**，本条是 **1.1463**，这个差不代表「更差」。
> 要比就得走下游 benchmark，或另写一路只在生成链 token 上算的 eval 指标。
> **top-k 选 checkpoint 仍是自洽的**（同一条 run 内同一口径），只是跨臂不可比。
> 重提到 global 256 后又多了一层不可比（batch 口径也不同了），见 §4.2.5。

#### 4.2.5 global batch 256 两条：per_device 加倍失败、改走 ga（2026-08-30 05:55 UTC 提交）

**结论先行。** **global batch 256 是可行的，但只能靠 `ga` 加倍，不能靠 `per_device` 加倍。**
最终配置 `per_device 4 × ga 16 × 4 卡 = 256`。`per_device 8` 在真实七源混合上会 OOM
（下面有 task id 和日志）；`per_device 256` 更是差约 20 倍，从来不可能。

| Job | Task ID | per_device × ga × 卡 | global | 状态 |
|---|---|---:|---:|---|
| `..._diffusion_allchains_immune_v3_4gpu` | **`t-20260830135521-zf5rr`** | 4 × 16 × 4 | **256** | Running（05:58 起跑） |
| `..._bert_immune_v3_4gpu`（新建） | **`t-20260830135524-qq7n5`** | 4 × 16 × 4 | **256** | Running（05:58 起跑） |

**🔴 教训：单源探测会严重低估显存，必须用真实混合语料探测。**
第一版把 `per_device` 从 4 提到 8 并提交（`t-20260830134534-86bml` / `t-20260830134537-8ncgx`），
两条**都在 step 2 OOM**：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.29 GiB.
GPU 3 has a total capacity of 79.15 GiB of which 965.25 MiB is free.
Of the allocated memory 76.29 GiB is allocated by PyTorch
```

而本机探测预测只有约 43 GiB —— **差了 33 GiB**。差在探测脚本只喂了 `asd_antibody`，
编码器输入形状是 `(8, 2, 870)`，即**每样本只有 2 条链**；真实七源混合里 TRAIT / TCR 样本带
peptide + MHC + TCRα + TCRβ，**链数可达 4-5**。ESMC 编码器的激活量是按
`per_device × 每样本链数 × 链长` 走的，链数翻倍多显存就翻倍多，所以按 ASD 定容必然乐观。

放大这个误差的两个结构性原因（`scripts/accelerate_configs/fsdp_llada.yaml`）：
`fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP` + `fsdp_transformer_layer_cls_to_wrap:
LLaDALlamaBlock` 意味着 **只有 decoder 的 block 被 FSDP 分片，ESMC 编码器整个不分片**，
每张卡都持有完整的 ESMC-300M（且 `mixed_precision: bf16` 下还有 fp32 master 权重）；
同时 `fsdp_activation_checkpointing: false`，decoder 靠 `--decoder_grad_ckpt True` 自己开了
梯度检查点，**编码器则完全没有** —— 编码器激活是全额驻留的。

> **下次定容的正确做法**：探测脚本必须用完整的
> `oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire` 混合采样，
> 并按 `encoder_input_ids` 的 `[B, C, L]` 里 **C 的最大值**取最坏批次，而不是只挑最长的行。
> 序列长度只是一个维度，链数是另一个，两个都要取最坏。

**为什么 ga 路线是零风险的。** 梯度累积不改变任何一次 forward/backward 的张量形状，
显存画像与 `per_device 4` 完全相同，而 `per_device 4` 已被 generated-only 那条**跑满 42k 步**
实证过。`diff` 核对：这两条与 `..._diffusion_immune_v3_4gpu` 相比，
`per_device_train_batch_size 4` **逐字相同**，只有 `gradient_accumulation_steps` 从 8 变 16
（外加各自的目标函数 flag / `run_name` / `OUTPUT_DIR`）。

**起跑实测（step 280，2026-08-30 06:15 UTC，两条均零报错）**：

| Job | loss 轨迹 | ETA |
|---|---|---|
| diffusion 全链 | 24.68 → 22.44 → **21.50** | ~50 h |
| bert | 14.59 → 13.10 → **12.41** | ~48 h |

**BERT 那条是新文件** `train_jobs/protein_esmc_llada270m_bert_immune_v3_4gpu.yml`，从 8 卡
`_spot` 版派生：`diff` 目标函数与数据 flag 后只差 batch 与 `run_name`，
`--bert_all_chains True` / mask 比例 / 七源目录全部一致。它补齐了 bert 臂在 4 卡非闲时的缺位
（此前 bert 只有 8 卡版在排队）。

> 🔴 **代价（明知故犯，用户拍板）。** `max_steps=50000` 没动，于是样本量从 640 万翻到
> **1280 万 ≈ 1.64 epoch**（128 那档是 0.84 epoch），墙钟也约翻倍（~50h）。这两条
> **与 global 128 的五条不可比** —— 包括已经跑到 42k 步的 generated-only 那条。
> 这两条之间 batch 口径相同，可比。
> 曾评估过 `per_device 8 × ga 4 = global 128`（同口径、纯提速、可比性不破），用户选择了 256；
> 而那个方案事后看也会 OOM，因为它同样要 `per_device 8`。

#### 4.2.6 🔴 未修复阻塞：一个空位符 `-` 让 42k 那条陷入重试死循环（2026-08-30 06:16 UTC 发现）

**症状。** `..._diffusion_immune_v3_4gpu` 反复失败并被 `RetryOptions` 换 id 重提，
每次都从 `checkpoint-42000` 续上、走约 780 步、崩在同一处：
`t-20260829143036-w7488` → `t-20260830131621-7dgwx`（崩于 42782）
→ `t-20260830141756-hhpm5`（崩） → **`t-20260830152319-r5w9k`**（07:23 UTC 起，第 3 次重试）。
`MaxRetryTimes: 5` 已用 3 次，**剩 2 次、约 2 小时**（单个循环 ≈ 加载 5min + 780 步 20min
+ NCCL 等 1800s 超时 + 重试间隔 180s）。`7dgwx` 的日志给出确切死因：

```
86%|████████▌ | 42782/50000 [23:06<3:04:17, 1.53s/it][rank2]: Traceback
AssertionError: Caught AssertionError in DataLoader worker process 3.
  raise AssertionError(f"unmapped decoder grammar ids (sample): {bad_src}")
AssertionError: unmapped decoder grammar ids (sample): [30]
[rank1] Watchdog caught collective operation timeout: WorkNCCL(..., OpType=_ALLGATHER_BASE,
        Timeout(ms)=1800000) ran for 1800032 milliseconds before timing out
```

rank2 在数据加载时抛断言，其余 rank 卡在 all-gather 直到 NCCL 1800s 超时，整个任务 Failed。

**根因（已定位到单个字符）。** `RemapCollator` 的映射表只覆盖 `RESIDUES`
（`LAGVSERTIDPKQNFYMHWCXBUZO`，25 个字符，含 XBUZO 模糊码）加 grammar / chainsep / pad。
ESMC 词表里的单字符 token 中有三个落在映射表外：**id 29 `.`、id 30 `-`、id 31 `|`**。
崩溃报的 `[30]` 就是**比对空位符 `-`**。

全语料扫描只找到 **1 行**（脚本按"该列大多数非空值都是纯残基"判定哪些列是序列列，
避免 `chain_id` 之类的元数据列误报）：

| 文件 | 行 | 列 | 值 |
|---|---:|---|---|
| `data/tcr_papers_v2/dataset/train.csv` | **3046** | `cdr3b` | **`ASSKVAARVP-TLKLS`** |

train split 其余六源（`oas` 473 万 / `ots` 210 万 / `asd_antibody` 85 万 / `trait` 6.9 万 /
`tcr_repertoire` 213 万，合计 850 万行）全部 OK；**holdout split 五源 19 万行也全部 OK**，
所以 eval 通路不会踩到这个 bug（只有 train 会）。

**扫描器已落仓库，结论可复现**：`scripts/data/assert_residue_alphabet.py`。
发现坏行时 **exit 1**，可直接用来 gate 提交（和 `assert_corpus_fresh.py` 并列放进 entrypoint）：

```bash
python scripts/data/assert_residue_alphabet.py                 # 七源 train split
python scripts/data/assert_residue_alphabet.py --split holdout
python scripts/data/assert_residue_alphabet.py --paths a.csv    # 单文件
```

脚本里的 `RESIDUES` 常量必须与 `examples/llada/protein_pretrain_esmc.py` 保持同步。

**为什么是死循环。** HF Trainer 默认 `ignore_data_skip=False`，resume 时用同一 seed 重建
dataloader 并跳到原位置，所以采样顺序可复现 —— 每次从 `checkpoint-42000` 续跑都会在
**约 42782 步**再次撞上同一行。`MaxRetryTimes: 5` 已用掉 2 次，**剩 3 次、约 3 小时后这条
84% 完成度的任务将永久卡在 42000 步**。

**修法（✅ 已于 2026-08-31 执行）。** 把那一个字符删掉：
`ASSKVAARVP-TLKLS` → `ASSKVAARVPTLKLS`（15 aa，CDR3β 的正常长度；单个 `-` 明显是录入/
格式化残留，去空位后就是真实序列）。这样做的好处是**行数不变**，采样顺序与
`checkpoint-42000` 的 resume 语义完全不受影响；删行则会移位。
**去污已预先核过**：`ASSKVAARVPTLKLS` 在全部 10 个 blocklist 里命中 **0 次**，
带 `-` 的原形也不在任何 blocklist 中，所以改这一格不会引入基准污染。

> ✅ **2026-09-01 复核**：磁盘上带 `-` 的原形已 **0 次命中**，修正形 `ASSKVAARVPTLKLS`
> 恰好 **1 次**，修改确已落地、行数未变。
>
> ⚠️ 以下是当时"为什么没有直接改"的顾虑，保留作决策记录 —— 后来是连同
> `finalize_report.json` 一起更新并重跑 `assert_corpus_fresh.py` 才执行的：
>
> ⚠️ **为什么没有直接改**：改 `train.csv` 属于绕过 `finalize_report.json` 的带外修改，
> 会让 build report 描述的语料与磁盘上的不一致 —— 这正是 2026-08-28 踩过的坑
> （report 写着 PASS，train.csv 里却留着 3 个 T4 答案）。要改就该同时更新
> `finalize_report.json` 并重跑 `assert_corpus_fresh.py`，这个决定留给语料的 owner。
>
> **另一条路**（不碰数据）：在 `expand_llada_tokenizer_for_esmc_grammar` 里把
> `.` / `-` / `|` 映射到已有的 `<res_X>`（"未知残基"）。不改词表大小、不动语料、
> checkpoint 完全兼容，但会把数据质量问题静默吞掉，且失去这条断言作为守卫的价值。
> 断言本身是对的 —— 它成功抓到了这个 bug，不该被削弱。

### 4.2.7 8 卡长跑（1M / 2M 步）与磁盘配额事故（2026-09-01）

50k 短跑已收口出数（§4.2.5、RESULTS §0.8），本轮把步数拉长一个数量级。四条 2M/1M 配置：

| Job | YAML | 队列/资源 | global batch | 起点 | 状态（2026-09-01 19:20Z） |
|---|---|---|---|---|---|
| `..._diffusion_allchains_immune_v3_8gpu_2m` | `train_jobs/protein_esmc_llada270m_diffusion_allchains_immune_v3_8gpu_2m.yml` | `c20250601` 8 卡 `ml.pni2.28xlarge`，非抢占 | **256** = per_device 2 × ga 16 × 8 | 4 卡 `checkpoint-33000` 的 **weights-only** | **Running** `t-20260902021013-bm79q`，盘上最新 **18000**（eval 0.7044；前 4 条 Failed，见下） |
| `..._bert_immune_v3_1m` | `train_jobs/protein_esmc_llada270m_bert_immune_v3_1m.yml` | `queue012` 8 卡，非抢占 | **256** = per_device 2 × ga 16 × 8 | **从头**（不加载 50k；崩溃才从本目录满包续） | Queue **`t-20260902110050-zb7dr`** |
| `..._diffusion_immune_v3_spot_2m` | `train_jobs/protein_esmc_llada270m_diffusion_immune_v3_spot_2m.yml` | `c20250601` **闲时** | 256 | — | Queue `t-20260901032620-vvngv`（看护循环在追） |
| `..._diffusion_allchains_immune_v3_4gpu_2m` | `train_jobs/protein_esmc_llada270m_diffusion_allchains_immune_v3_4gpu_2m.yml` | 4 卡 | 256 | — | `t-20260901032611-npf27` **Killed**（换成上面的 8 卡版） |

LR 调度统一改为**与 AirGen DPLM 相同的 polynomial power=1**：BERT 1M 那条峰值 **4e-5**、
`lr_end=1e-5`、`warmup=0`（续训不需要重新 warmup）；`..._8gpu_2m` 因为 optimizer 是新建的，
峰值仍是 **1e-4**、`lr_end=1e-5`。

⚠️ **这批与 50k 那批不可混排**：目标函数、batch 口径、LR 调度三样都变了。

**为什么 `..._8gpu_2m` 首跑只能加载权重、不能满包续。** 源 checkpoint 是 4 卡的 FSDP pack，
`optimizer.bin` / `pytorch_model_fsdp.bin` **不能跨 world size 续**，所以 entrypoint 用
`--init_fusion_weights` 只吃 `model.safetensors`，optimizer 与 polynomial 从 step 0 新建。
entrypoint 因此有两个挑选函数，语义不同、**不要合并**：

- `pick_latest_full`：要求 `optimizer.bin` + `pytorch_model_fsdp.bin` + `scheduler.pt` **三件齐全**
  才认，用于本目录已写出 8 卡 pack 后的普通 `--resume_from_checkpoint`。
- `pick_latest_weights`：只要有 `model.safetensors` 就认，用于跨 world size 的首次 init。

#### 🔴 事故：`Disk quota exceeded`，同一处 4 连 Failed，净进度 0

```
File ".../transformers/trainer.py", line 3926, in _save
    safetensors.torch.save_file(
safetensors_rust.SafetensorError: Error while serializing:
    I/O error: Disk quota exceeded (os error 122)
```

崩点**不在训练步，在 `_maybe_log_save_evaluate → _save_checkpoint → save_model`**：

| Task ID | 生命周期（UTC） | 结果 |
|---|---|---|
| `t-20260901033935-h55f4` | 00:55 → 15:02（14h07m，step 0 → 17000） | 存盘配额爆，留下 **132 MiB 半截** `checkpoint-17000/model.safetensors` |
| `t-20260901230256-ns4q4` | 15:02 → 16:06（58 min） | 从 16000 续，跑满 1000 步，同一处爆 |
| `t-20260902000627-j4mqm` | 16:10 → 17:08（58 min） | 同上 |
| `t-20260902010820-t6hq6` | 17:12 → 18:09（58 min） | 同上 |
| `t-20260902021013-bm79q` | 18:13 → Running | 19:10:42 **存盘成功**，`checkpoint-17000` 完整 9.0 G |

- **爆的是目录/租户配额，不是文件系统**：`df` 显示 `fs_vepfs-cnbj2c98dea54433` 3.1P 已用 2.3P、
  **尚余 809T**。只看 `df` 会完全误判方向。
- **一次 save 需要约 9.0 GB 一次性余量**（`model.safetensors` 2.3G + `pytorch_model_fsdp.bin` 2.3G
  + `optimizer.bin` 4.5G），而 **top-k 剪枝发生在写完之后** —— 峰值需求是「现有占用 + 一整个新 ckpt」。
- **`RetryOptions` 把它变成了循环**：`MaxRetryTimes: 50` / `PolicySets: [Failed]`，每轮
  「从 16000 续 → 跑 1000 步 → 同一处爆」耗 58 分钟。3 小时 8 卡非抢占资源换到 **0 步净进度**，
  按剩余重试次数最多还能这样烧约 46 小时。
- **第 5 次成功不代表修好了**：期间只是别的任务腾出了空间，配额仍贴在临界点，下一次 save 随时再爆。

✅ **`pick_latest_full` 这层防御是有效的，本次没有重演 §4.2.6 的「从坏 ckpt 反复续跑」**：
它要求三件套齐全，因此正确跳过了只有半截 `model.safetensors` 的 `checkpoint-17000`、回退到
完整的 16000。**新增 checkpoint 挑选逻辑时必须保留这个三件套判据**，只看目录名或只看
`model.safetensors` 会直接掉进死循环。

#### 新发现：top-k 账本在重启时重置 → 孤儿 checkpoint 永不被剪

`TopKValLossCheckpointCallback` 从**本进程**的 `log_history` 取 `eval_loss` 重建 top-k，
**看不到上一个进程存过什么**。于是每次崩溃-重启都会把上一轮的 checkpoint 甩出账本，
剪枝逻辑再也不会碰它们 —— 崩溃循环本身在单调抬高占用，反过来更容易撞配额，形成正反馈。

实测本目录：`topk_val_manifest.json` 只剩 `checkpoint-17000` 一条，而盘上还躺着
`checkpoint-7000/13000/15000`（各 2.3G，已 slim）+ **`checkpoint-16000`（9.0G，未 slim）**。

全 `output/` 只读盘点（`scripts` 未落仓库，一次性脚本）：

| 口径 | 量 |
|---|---:|
| checkpoint 数据合计 | **790.8 G** |
| 其中 resume-only（`optimizer.bin`/`pytorch_model_fsdp.bin`/`rng_state_*`/`scheduler.pt`，剪掉不丢权重） | **328.3 G** |
| 账本外孤儿 checkpoint 整体 | **132.9 G** |
| 两条 8B run 的 `checkpoint-50000`（fat，各 124.5 G） | 249.0 G |
| `checkpoint-final` 与 `checkpoint-50000/model.safetensors` **逐字节相同**（md5 实测）却是独立副本 | 69.1 G |

#### ✅ 已执行的回收：`output/` 1.2 T → **767 G**（释放 389 GB，2026-09-01 19:30Z）

| 组 | 动作 | 释放 |
|---|---|---:|
| 1a | 删 `ABORTED_gb128_..._4gpu/` 整目录（`t-20260830125655-hngmn` 已 cancel） | 9.1 G |
| 1b | 两条**已终态** 8B run 的 `checkpoint-50000` 剪 resume-only（8B 不会再续训） | 94 G × 2 |
| 1c | 5 处 `checkpoint-final/model.safetensors` **改硬链接**指向同 run 的 `checkpoint-50000` | 69 G |
| 2a–2c | 已 Killed 的两条 spot run + 两条 4 卡 run + 早期 `diffusion_immune` 的账本外孤儿 | 117 G |
| 2d | 正在跑的 `..._8gpu_2m` 的 slim 孤儿 `checkpoint-7000/13000/15000` | 6.9 G |

⚠️ **`checkpoint-final` 是改硬链接不是删** —— `eval_jobs/eval_v3_bert_final_*.yml` 有 8 处引用
`checkpoint-final`、`train_jobs` 有 2 处引用 `checkpoint-50000`，**两边路径都必须留着**。
改之前逐个自查了大小 + 首尾 64 MB md5，改之后 `nlink=2`／inode 相同，且
`safetensors.safe_open` 能正常解析（8B 602 个张量、270m 386 个）。

**刻意保留、不能碰的两个**：`..._diffusion_allchains_immune_v3_4gpu/checkpoint-33000`
（`..._8gpu_2m` 的权重来源）、`..._bert_immune_v3/checkpoint-50000` 完整满包
（`..._v3_1m` 的 resume 来源）。

**未回收**：5 条 grammar_v2 run 合计 339 G。它们内部没有 optimizer（查过 zip 目录），
没有"只剪 optimizer"的省法，只能整份删；而 `..._7l` 被 `PROJ_GUIDE.md` 钉住（见下），
其余四条要不要留权重是科研决策，**留待拍板**。

> 仍待做：① 让 callback 启动时**从磁盘现存 checkpoint 重建账本**而不是只看本进程
> `log_history`（否则每次重启继续产孤儿）；② save 前做配额预检，余量不足时降级为
> weights-only 而不是让整个任务 `exit 1`。**这两项没做之前，回收只是把配额往后推，
> 不是修好了。**

#### 核实过、不是 bug 的两件事

- **resume 是健康的。** 续跑后 step 16300–17000 的 train loss 7.2–7.7、`learning_rate 9.925e-05`，
  与首跑同 step 的 7.2–7.8／同 lr 逐条吻合 —— 权重与 polynomial 调度器都正确恢复了。
- **train loss ~7.3 与 `eval_loss` 0.7028 差一个数量级，原因未查明。** 但两条路径的 CE 计算相同
  （`loss_weight_type='none'`、`_diffusion_token_weights` 返回 `None`、无 chain 加权），且首跑与
  resume 后完全一致，所以不是本次故障的征兆。4 卡那条 train loss 从 97 起步也是同一现象。
  **登记为未解项**，不要拿 train loss 与 eval_loss 跨口径比。

#### 量级：2M 步 ≈ 68 天

2.95 s/it × 2,000,000 = **1,640 小时 ≈ 68 天**（进度条自印 `1644:30:32`）。
`ActiveDeadlineSeconds: 7776000`（90 天）放得下，但意味着独占 8 卡两个多月，
且每 1000 步都要过一次 9.0 GB 的配额窄门。**这个步数目标是否按 68 天规划的，待确认。**

### 4.2.8 全链两个 ckpt 全套下游（2026-09-01 20:41Z 提交，**8/8 Success，数字已回填**）

用户要求同时评：

1. 4 卡 50k 配方停住的最好点 `..._allchains_..._4gpu/checkpoint-33000`（eval **0.6866**）
2. 从该点 weights-only 续出的 8 卡 2M 当前快照 `..._allchains_..._8gpu_2m/checkpoint-18000`（eval **0.7044**）

覆盖与 generated-only 相同：T1/T2/T3 + AB CDR + pairing + T4。BERT 仍只评表征，本轮不碰。
**不要 cancel / 打断** 续训 `t-20260902021013-bm79q`。

| Tag | Run | Ckpt | 配方 | eval_loss |
|---|---|---|---|---:|
| `ours_fusion_v3_allchains_33000` | `..._allchains_immune_v3_4gpu` | `checkpoint-33000` | 50k cosine，global 256，满包 | 0.6866 |
| `ours_fusion_v3_allchains_8gpu2m_18000` | `..._allchains_immune_v3_8gpu_2m` | `checkpoint-18000` | 2M polynomial 重开，global 256，weights-only 自 33000 | 0.7044 |

**对照纪律（写进任何结论前先看）**

- 两条都是 `--diffusion_all_chains True`、global 256，**下游数字可以并排看方向**（源点 vs 续训早期快照）。
- **`eval_loss` 不可比**：optimizer / FSDP 跨 4→8 卡已重置，polynomial 从 step 0 重开；续训 18k 的 0.7044 **高于** 源点 0.6866 是预期，不是退化判据。
- 与 RESULTS 组 C（generated-only / global 128 / 50k cosine）**不可混排**。组 C 的 0.75 与全链 0.69 分母不同（固定上下文是否计入 loss）。
- 18k / 2M ≈ 0.9% 进度。这是用户点名的快照，**不是** 2M 配方的 headline；续训继续跑，本评测不追后续 step。

| Job | Task ID | 覆盖 | 状态 |
|---|---|---|---|
| `eval-v3-allchains-33000-repr` | `t-20260902044100-wnkdb` | T1 / T2 / T3 | **Success** 29.5 min |
| `eval-v3-allchains-33000-cdr` | `t-20260902044104-29sqz` | AB CDR | **Success** 15.7 min |
| `eval-v3-allchains-33000-pairing` | `t-20260902044108-ddnrw` | pairing | **Success** 2.75 h |
| `eval-v3-allchains-33000-t4` | `t-20260902044113-xxnp7` | T4 | **Success** 11.4 min |
| `eval-v3-allchains-8gpu2m-18000-repr` | `t-20260902044117-vh48s` | T1 / T2 / T3 | **Success** 28.0 min |
| `eval-v3-allchains-8gpu2m-18000-cdr` | `t-20260902044121-q2rt5` | AB CDR | **Success** 15.8 min |
| `eval-v3-allchains-8gpu2m-18000-pairing` | `t-20260902044126-mw2s6` | pairing | **Success** 2.74 h |
| `eval-v3-allchains-8gpu2m-18000-t4` | `t-20260902044130-zx5vk` | T4 | **Success** 11.4 min |

**Headline 数字（组 D；勿与组 C 混排）**

| | allch@33000 | allch@18k | 对照 |
|---|---:|---:|---|
| T1 seen AUPRC | 0.766 | 0.761 | v3 generated-only 0.763 / 0.767 |
| T1 unseen AUROC | 0.514 | 0.501 | 全体≈随机 |
| T2 ARI mean | 0.022 | 0.022 | 旧 270m diff **0.028**；v3 0.017 / 0.021 |
| T3 deep k=200 | 0.703 | 0.710 | v3 0.705 / 0.709 |
| T4 common d_edit↓ | 8.62 | 8.73 | v3 8.40 / 8.58；OLGA 6.34 |
| SAb23H2 H3 iter=2 | 35.4 | 34.8 | v3 34.1 / 34.8；Ophiuchus ~36.7 |
| Pairing ImmunoMatch | **0.436** | **0.416** | v3 generated-only **0.353**（贴地板）；旧 270m 0.417 |

读法：表征与 generated-only 同档。**配对是全链唯一明显抬起来的任务**（离开 0.353 地板，回到旧 270m）。T4 没有变好。18k 相对 33000 没有系统性变好——这是 2M polynomial 重开后 0.9% 进度，不是续训失败。

队列 `c20250601`，单卡 `ml.pni2.3xlarge`，`Preemptible: false`（pairing 长，闲时易被杀）。
YAML：`eval_jobs/eval_v3_allchains_{33000,8gpu2m_18000,8gpu2m_26000}_{repr,cdr,pairing,t4}.yml`。
ID 账本：`output/downstream_generation/eval_v3_allchains_20260902_task_ids.tsv`、
`output/downstream_generation/eval_v3_allchains_26000_20260902_task_ids.tsv`。
数字回填：[`RESULTS.md`](../../downstream/benchmark/RESULTS.md) §0.1–§0.6 按 tag、**§0.8 组 D**。

**续训最新 `checkpoint-26000`（2026-09-02 02:43Z 提交，数字待回填）**

续训 `t-20260902021013-bm79q` 仍 Running。盘上最新满包是 **26000**（eval **0.6902**）；
top-k 里 25000 的 val **0.6861** 更好，本轮按用户「最新」评 26000，不评 25000。
18000 已被 top-k 剪掉，下游数字仍在 RESULTS。不打断续训。

| Job | Task ID | 覆盖 | 状态 |
|---|---|---|---|
| `eval-v3-allchains-8gpu2m-26000-repr` | `t-20260902104333-87ccm` | T1 / T2 / T3 | 已提交 |
| `eval-v3-allchains-8gpu2m-26000-cdr` | `t-20260902104338-7ptgf` | AB CDR | 已提交 |
| `eval-v3-allchains-8gpu2m-26000-pairing` | `t-20260902104342-68qz8` | pairing | 已提交 |
| `eval-v3-allchains-8gpu2m-26000-t4` | `t-20260902104346-882r9` | T4 | 已提交 |

tag：`ours_fusion_v3_allchains_8gpu2m_26000`。四条均已 Success，数字仍待回填 RESULTS §0.8。

**续训 top-k 最好 `checkpoint-69000`（2026-09-04 01:37 CST 提交，闲时单卡）**

用户要求测当前 8 卡 2M 下游。盘上最新满包 72000 eval 0.6609；top-k 最好 **69000 eval 0.6577**。按最好 val 评 69000，权重硬链接到 `eval_snapshot_69000/`。不打断续训。首提非抢占四条已 cancel，按用户改闲时重提。

| Job | Task ID | 覆盖 | 状态 |
|---|---|---|---|
| `eval-v3-allchains-8gpu2m-69000-repr` | `t-20260904013714-4w8nk` | T1 / T2 / T3 | **Success** |
| `eval-v3-allchains-8gpu2m-69000-cdr` | `t-20260904013717-rhmzz` | AB CDR | **Success** |
| `eval-v3-allchains-8gpu2m-69000-pairing` | `t-20260904013720-nprjc` | pairing | **Success** |
| `eval-v3-allchains-8gpu2m-69000-t4` | `t-20260904104404-t6j7v` | T4 | 闲时补提（前次 Killed） |

tag：`ours_fusion_v3_allchains_8gpu2m_69000`。队列 `c20250601`，`Preemptible: true`，单卡 `ml.pni2.3xlarge`。
YAML：`eval_jobs/eval_v3_allchains_8gpu2m_69000_{repr,cdr,pairing,t4}.yml`。
账本：`output/downstream_generation/eval_v3_allchains_69000_20260904_task_ids.tsv`。
表征 / CDR / pairing 数字待回填 RESULTS §0.8；T4 已用 `eval_snapshot_69000/` 闲时补提 `t-20260904104404-t6j7v`。

**续训 top-k 最好 `checkpoint-73000`（2026-09-04 10:41 CST 提交，闲时单卡）**

用户要求测 73000。盘上最新满包 83000 eval 0.6567；top-k 最好 **73000 eval 0.6535**。权重硬链接到 `eval_snapshot_73000/`。不打断续训。

| Job | Task ID | 覆盖 | 状态 |
|---|---|---|---|
| `eval-v3-allchains-8gpu2m-73000-repr` | `t-20260904104145-77cfr` | T1 / T2 / T3 | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-73000-cdr` | `t-20260904104148-6x5m9` | AB CDR | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-73000-pairing` | `t-20260904104152-pxjd4` | pairing | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-73000-t4` | `t-20260904104155-tctxm` | T4 | 闲时已提交 |

tag：`ours_fusion_v3_allchains_8gpu2m_73000`。队列 `c20250601`，`Preemptible: true`，单卡 `ml.pni2.3xlarge`。
YAML：`eval_jobs/eval_v3_allchains_8gpu2m_73000_{repr,cdr,pairing,t4}.yml`。
账本：`output/downstream_generation/eval_v3_allchains_73000_20260904_task_ids.tsv`。

**BERT 1M 当前 `checkpoint-34000` 表征（2026-09-04 10:49 CST 提交，闲时单卡）**

用户要求测 BERT 1M 当前点。盘上最新满包 **34000** eval **0.4647**（同 run 内 33000 val **0.4608** 更好，本轮按「当前 34000」评）。只跑 T1/T2/T3，不跑 CDR / pairing / T4。权重硬链接到 `eval_snapshot_34000/`。不打断 `t-20260902110050-zb7dr`。

| Job | Task ID | 覆盖 | 状态 |
|---|---|---|---|
| `eval-v3-bert-1m-34000-repr` | `t-20260904104944-mdr7x` | T1 / T2 / T3 | 闲时已提交 |

tag：`ours_fusion_v3_bert_1m_34000`。队列 `c20250601`，`Preemptible: true`，单卡 `ml.pni2.3xlarge`。
YAML：`eval_jobs/eval_v3_bert_1m_34000_repr.yml`。
账本：`output/downstream_generation/eval_v3_bert_1m_34000_20260904_task_ids.tsv`。
与 50k `ours_fusion_v3_bert_final` **不可混排 `eval_loss`**（1M 是 global 256 + polynomial 从头）。
首提 `t-20260904104944-mdr7x` 约 3 分钟后被闲时抢走，无产物。

**续训 `checkpoint-121000` 表征 + 生成（2026-09-05 / 09-06）**

用户要求测全链 diffusion 下游。121000 当时是 top-k 最好（eval **0.6416**）。表征只跑 T1/T2/T3，**已 Success**。用户确认后补交 CDR / pairing / T4。原计划补评最新 122000，但该包已被 top-k 剪掉；当前最新满包 **143000** eval **0.6536**，当前最好改为 **137000 eval 0.6332**，故表征改评 137000。权重硬链接 `eval_snapshot_121000/`、`eval_snapshot_137000/`。不打断 `t-20260902021013-bm79q`。

121000 表征数字：T1 seen AUPRC **0.766** / unseen AUROC **0.527**；T2 ARI **0.025**；T3 deep k=200 **0.711** / broad k=100 **0.645**；probe-AUROC **0.807**。

| Job | Task ID | 覆盖 | 状态 |
|---|---|---|---|
| `eval-v3-allchains-8gpu2m-121000-repr` | `t-20260905191346-rbt8z` | T1 / T2 / T3 | **Success** |
| `eval-v3-allchains-8gpu2m-121000-cdr` | `t-20260906125811-fdrvj` | AB CDR | **Success** |
| `eval-v3-allchains-8gpu2m-121000-pairing` | `t-20260906132314-lqbrp`（前次 `t-20260906125815-q8h8b` Killed） | pairing | 闲时补提 |
| `eval-v3-allchains-8gpu2m-121000-t4` | `t-20260906132318-d5t9d`（前次 `t-20260906125820-db5bd` Killed） | T4 | 闲时补提 |
| `eval-v3-allchains-8gpu2m-137000-repr` | `t-20260906125806-xkg7q` | T1 / T2 / T3 | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-144000-repr` | `t-20260906132412-tklfb` | T1 / T2 / T3 | 闲时已提交 |

tag：`ours_fusion_v3_allchains_8gpu2m_121000` / `ours_fusion_v3_allchains_8gpu2m_137000`。队列 `c20250601`，`Preemptible: true`，单卡 `ml.pni2.3xlarge`。
YAML：`eval_jobs/eval_v3_allchains_8gpu2m_{121000_{repr,cdr,pairing,t4},137000_repr}.yml`。
账本：`output/downstream_generation/eval_v3_allchains_121000_137000_20260906_task_ids.tsv`。

**三条长跑各自最好 val 点（2026-09-06 23:35 CST 提交，闲时单卡）**

用户要求评当前三条训练的最好 `eval_loss` 点。BERT **只跑 T1/T2/T3**，不交 CDR / pairing / T4。
权重硬链接到各 run 的 `eval_snapshot_*`，避免 top-k 剪掉评测输入。三条训练不打断。

| Run | 最好点 | eval_loss | 最新满包 | tag | 覆盖 |
|---|---|---:|---|---|---|
| 全链 `..._allchains_..._8gpu_2m` | `checkpoint-151000` | **0.6326** | 156000 / 0.6430 | `ours_fusion_v3_allchains_8gpu2m_151000` | T1–T3 + CDR + pairing + T4 |
| generated-only `..._immune_v3_8gpu_2m` | `checkpoint-44000` | **0.7520** | 56000 / 0.7566 | `ours_fusion_v3_genonly_8gpu2m_44000` | T1–T3 + CDR + pairing + T4 |
| BERT `..._v3_1m` | `checkpoint-105000` | **0.4263** | 108000 / 0.4288 | `ours_fusion_v3_bert_1m_105000` | T1/T2/T3 **only** |

| Job | Task ID | 覆盖 | 状态 |
|---|---|---|---|
| `eval-v3-allchains-8gpu2m-151000-repr` | `t-20260906233442-d2gzt` | T1 / T2 / T3 | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-151000-cdr` | `t-20260907001930-qtf2s`（前次 `t-20260906233447-rh2j9` Killed） | AB CDR | 闲时补提 |
| `eval-v3-allchains-8gpu2m-151000-pairing` | `t-20260906233451-fjzmm` | pairing | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-151000-t4` | `t-20260906233455-djlmk` | T4 | 闲时已提交 |
| `eval-v3-genonly-8gpu2m-44000-repr` | `t-20260906233500-9f7vp` | T1 / T2 / T3 | 闲时已提交 |
| `eval-v3-genonly-8gpu2m-44000-cdr` | `t-20260906233504-x2n8b` | AB CDR | 闲时已提交 |
| `eval-v3-genonly-8gpu2m-44000-pairing` | `t-20260906233508-svn2t` | pairing | 闲时已提交 |
| `eval-v3-genonly-8gpu2m-44000-t4` | `t-20260906233513-mpfv5` | T4 | 闲时已提交 |
| `eval-v3-bert-1m-105000-repr` | `t-20260906233517-9x8kq` | T1 / T2 / T3 | 闲时已提交 |

队列 `c20250601`，`Preemptible: true`，单卡 `ml.pni2.3xlarge`。
YAML：`eval_jobs/eval_v3_{allchains_8gpu2m_151000,genonly_8gpu2m_44000}_{repr,cdr,pairing,t4}.yml`、`eval_jobs/eval_v3_bert_1m_105000_repr.yml`。
账本：`output/downstream_generation/eval_v3_bestval_151000_44000_105000_20260906_task_ids.tsv`。
三条 `eval_loss` 不能互相比；generated-only 8 卡 2M 与组 C（50k / global 128）也不可混排。

> ⚠️ **上表九条 ID 已被 §4.2.9 的 `queue012` 非闲时重提取代**，它们在 `c20250601` 排了 4.9 小时零起跑。
> 09-07 07:20 CST 已全部 cancel 成 `Killed`（§4.2.11）。引用任务 ID 时看 §4.2.9。

#### 4.2.9 九条下游改投 `queue012` 非闲时（2026-09-07 12:43 CST，`c20250601` 权限已失）

用户要求：仍未起跑的单卡任务改成**非闲时**重提。§4.2.8 那九条在 `c20250601` 闲时从
2026-09-06T15:34Z 排到 20:31Z（**4.9 小时**）全是 `Queue`，一条没起跑 —— 期间该队列上
Running 的 11 条单卡全是别人的任务。

**为什么连队列一起换（不只是关抢占）。** 先只把 `Preemptible` 改 `false` 重提，九条全被 IAM 拒：

```
User is not authorized to perform: ml_platform:CreateCustomTask
  on resource: trn:ml_platform:cn-beijing:2106112826:resourcequeue/q-20260121145036-6fztt
```

即本账号（`zhuyiheng`）对 `c20250601` **已无提交权限**。用一条探针任务确认 `queue012` 可用：
`probe-perm-check-donotrun` → `t-20260907044147-jvzpn`，提交成功、`cancel success`、终态 `Killed`，
未占用资源。故九条 YAML 统一改 `ResourceQueueName: "queue012"` + `Preemptible: false`，
单卡 `ml.pni2.3xlarge` / `Priority: 6` / entrypoint / ckpt 路径 / tag **零改动**。

| Job | 新 Task ID（`queue012` 非闲时） | 被取代的闲时 ID（仍 `Queue`） |
|---|---|---|
| `eval-v3-allchains-8gpu2m-151000-repr` | **`t-20260907044259-qd52v`** | `t-20260906233442-d2gzt` |
| `eval-v3-allchains-8gpu2m-151000-cdr` | **`t-20260907044303-ks4x6`** | `t-20260907001930-qtf2s` |
| `eval-v3-allchains-8gpu2m-151000-pairing` | **`t-20260907044307-btjxr`** | `t-20260906233451-fjzmm` |
| `eval-v3-allchains-8gpu2m-151000-t4` | **`t-20260907044310-wlm9h`** | `t-20260906233455-djlmk` |
| `eval-v3-genonly-8gpu2m-44000-repr` | **`t-20260907044314-mtzn9`** | `t-20260906233500-9f7vp` |
| `eval-v3-genonly-8gpu2m-44000-cdr` | **`t-20260907044317-kfhk2`** | `t-20260906233504-x2n8b` |
| `eval-v3-genonly-8gpu2m-44000-pairing` | **`t-20260907044321-ht8b6`** | `t-20260906233508-svn2t` |
| `eval-v3-genonly-8gpu2m-44000-t4` | **`t-20260907044324-tcmhs`** | `t-20260906233513-mpfv5` |
| `eval-v3-bert-1m-105000-repr` | **`t-20260907044328-wwfjn`** | `t-20260906233517-9x8kq` |

**回读确认。** 九条逐条 `ml_task get -i <id> --format JobId,JobName,ResourceQueueId,Preemptible,Priority,Status -o json`：
`Preemptible=False`、`ResourceQueueId=q-20260524172355-rnqtf`（`queue012`）、`Priority=6`、初始 `Queue`。
注意 `ml_task export --config` **不导出 `Preemptible` 字段**（模板里没有这一项），核对抢占只能用 `get --format`。

~~🔴 **旧九条 cancel 不掉，双跑会互相覆盖产物。**~~ **已解除（09-07 07:20 CST，§4.2.11）。**
提交当时它们的 `Creator` 是 `251105016`，本账号 cancel 报 `ml_platform:StopCustomTask` 未授权。
新旧同名、`run_immune_fusion_{repr,gen}.sh` 的产物前缀 `output/downstream_generation/<tag>` 也相同，
闲时那批一旦抢到资源就会与新批写同一批文件。次日早上同一账号再 cancel 九条全部 `cancel success`，
`get` 回读九条终态 `Killed`，无需再走控制台。

账本：`output/downstream_generation/eval_v3_bestval_queue012_nonpreempt_20260907_task_ids.tsv`。

**（同日 13:24 CST 追加）只留两条 pairing，其余七条 cancel。** 上表九条在 `queue012` 又排了
41 分钟仍 9/9 `Queue`。查队列构成发现 `queue012` 的拥堵**全在单卡**：非终态 758 条里
单卡 607 排队 / 106 Running，而多卡排队总共只有 27 条（4 卡 11、8 卡 15、24 卡 1）。
我们九条正好都在最挤的单卡档，跑得快慢不决定谁先出数字 —— **谁先排上才决定**。

按 36 条历史 Success 实测的各阶段耗时中位数排优先级（下表），pairing 比其余阶段慢一个数量级，
是本轮的关键路径，故保留两条 pairing、cancel 其余七条让位。cancel 时七条**都还在 `Queue`**，
无中断损失；ckpt / tag 未动，要补做直接用原 YAML 重提。

| 阶段 | 中位 | 范围 | 次数 | 解码步数 |
|---|---:|---|---:|---|
| T4 生成 | 12 min | 11–28 | 9 | 脚本写死 32 |
| AB CDR | 16 min | 16–43 | 9 | `CDR_MAX_ITER=2` |
| 表征 T1/T2/T3 | 28 min | 27–55 | 11 | — |
| **AB light pairing** | **165 min** | 164–180 | 6 | `PAIR_MAX_ITER=124` |
| T4 max_iter sweep | 72 min | — | 1 | 8/32/64/128 |

差异来自解码步数（pairing 124 步 vs CDR 2 步 / T4 32 步），不是数据量；偏慢的离群值
（repr 55min、CDR 43min）都出自 `4gpu-42000` 那批，与阶段无关。

保留：`t-20260907044307-btjxr`（151000-pairing）、`t-20260907044321-ht8b6`（44000-pairing）。
已 cancel（均 `cancel success`）：`t-20260907044259-qd52v` / `t-20260907044303-ks4x6` /
`t-20260907044310-wlm9h` / `t-20260907044314-mtzn9` / `t-20260907044317-kfhk2` /
`t-20260907044324-tcmhs` / `t-20260907044328-wwfjn`。

~~🔴 控制台优先停 `c20250601` 那两条闲时 pairing~~（`t-20260906233451-fjzmm` /
`t-20260906233508-svn2t`）—— **已于 09-07 07:20 CST 随其余七条一起 cancel 成 `Killed`（§4.2.11）**。
当时的顾虑：与保留的两条同名同产物前缀，且 pairing 要跑满 2.7 小时，闲时资源中途被抢就是白跑，
还会把新任务的产物覆盖成半成品。

#### 4.2.10 四个 T4 / CDR 阶段本地单卡跑完，数字已回填 RESULTS（2026-09-07 21:28–22:25Z）

§4.2.9 那批在 `queue012` 也排不动，四个 T4/CDR 阶段改本机跑。**这条通路以后可复用**：
单卡评测在两个队列都排不上时，本地跑与平台跑同口径，不必干等。

**怎么跑的。** 本机一张空闲 A100-80G（起跑前 `nvidia-smi` 0% / 4MiB、无其他进程）。
env **直接激活 eval YAML 里写的绝对路径**
`/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr`（不要用 `base` / `pllm`），
其余 `PYTHONPATH` / `CUDA_VISIBLE_DEVICES=0` / `HF_HUB_OFFLINE=1` / `PYTORCH_CUDA_ALLOC_CONF`
与 entrypoint 逐行照抄，`CDR_MAX_ITER=2`、batch `4 8` 不变，故产物与平台跑同口径。
一张卡所以**四阶段串行**。Runner `output/_local_runs/run_local_t4_cdr.sh`，
日志 `local_run.log`，逐阶段耗时与退出码 `status.tsv`。

| 顺序 | 阶段 | 耗时 | exit | 平台历史中位 |
|---:|---|---:|---:|---:|
| 1 | allchains 151000 T4 | 11 min | 0 | 12 |
| 2 | genonly 44000 T4 | 11 min | 0 | 12 |
| 3 | allchains 151000 CDR | 17 min | 0 | 16 |
| 4 | genonly 44000 CDR | 16 min | 0 | 16 |

合计 56 分钟，与平台耗时一致 —— 说明本机单卡与 `ml.pni2.3xlarge` 算力等效，
**排不上队时本地跑没有速度代价**。

**结果（数字权威在 [RESULTS.md](../../downstream/benchmark/RESULTS.md) §0.4 / §0.5 / §0.8，此处只记结论）**

1. **长跑到 151k 没有改善 T4，反而微降。** common-6 d_edit：26k **8.58** → 121k **8.78** →
   151k **8.80**；seq-rec 0.212 → 0.210 → 0.218。仍远差于旧 270m（6.99）和**不看表位的
   OLGA（6.34）**。§0.4 那条主结论「unseen 表位上条件信号接近不存在」**长跑没能撼动**。
2. **CDR 小幅上行，但在噪声内。** SAbDab 旧快照 H3 26k 41.68 → 151k **41.92**（+0.24），
   H2 +1.34；SAb23H2 六 CDR 均值 67.70 → 68.29。同臂 18k→26k 的 H2 波动就有 +0.41，
   所以这不足以当"长跑有效"的证据。**这是 2M 计划的 7.6% 进度。**
3. **全链 vs generated-only 打平，`--diffusion_all_chains` 至今无可见收益。**
   151k − 44k：SAbDab H1/H2/H3 +0.28 / +0.45 / +0.13，SAb23H2 互有胜负
   （六 CDR 均值 68.29 vs 68.28，差 0.01），T4 则是 151k **落后 0.19**。
   全部落在已登记的 `max_iter` 协议不确定性带宽内。
4. **CDR 走的是旧快照，不是 Kong 3,127。** wrapper 默认 `data/downstream/cdr_infilling/sabdab`，
   该缺陷 2026-09-02 已登记。故数字只进 §0.5 的「旧快照」表，**Kong 主表不填**。

⚠️ **本轮未收口**：两条 pairing（`t-20260907044307-btjxr` / `t-20260907044321-ht8b6`）仍在
`queue012` 排队，而 **pairing 是 headline 指标**。上面四条只覆盖 T4 与 CDR。
BERT 1M 105000 的表征作业已 cancel、未跑（要补可同样本地跑，`run_immune_fusion_repr.sh`，约 28min）。

~~🔴 产物覆盖风险已升级为现实损失风险~~ —— **已解除（09-07 07:20 CST，§4.2.11）**。
当时 `c20250601` 上四条同名闲时任务
（`t-20260906233455-djlmk` 151000-t4、`t-20260906233513-mpfv5` 44000-t4、
`t-20260907001930-qtf2s` 151000-cdr、`t-20260906233504-x2n8b` 44000-cdr）一旦抢到资源，
会覆盖刚跑出来的 `output/downstream_generation/<tag>_*`。四条均已 `Killed`，本地产物安全。

#### 4.2.11 `c20250601` 旧闲时九条全部 cancel（2026-09-07 07:20 CST）

用户要求把用来测评模型效果的闲时任务停掉。cancel 前核查：

- `ml_task list --name eval-v3`：我们的非终态任务共 11 条 —— `c20250601` 上 9 条闲时
  （`Preemptible: true`，Creator `251105016`，**全部仍 `Queue`，一条没起跑过**），
  加 `queue012` 上两条非闲时 pairing（`t-20260907044307-btjxr` / `t-20260907044321-ht8b6`）。
- 看护循环 `scripts/monitor_spot_tasks.py` 仍在跑（PID 3777252），但 `TARGETS` 只写了训练任务
  `protein_esmc_llada270m_diffusion_immune_v3_spot_2m`（且已放 STOP 哨兵），**不会重提评测任务**，
  cancel 不需要先动它。

**结果：九条 `cancel success`，`get` 回读全部终态 `Killed`。** 09-07 中午同一账号 cancel 报
`StopCustomTask` 未授权，这次没再报 —— 权限侧发生了什么未知，`CreateCustomTask` 是否也恢复
**未测**（测它要真提任务，不值得）。以后对 `c20250601` 存量任务先直接 `ml_task cancel` 试，不必默认走控制台。

| Job | 闲时 Task ID | 终态 |
|---|---|---|
| `eval-v3-allchains-8gpu2m-151000-repr` | `t-20260906233442-d2gzt` | Killed |
| `eval-v3-allchains-8gpu2m-151000-cdr` | `t-20260907001930-qtf2s` | Killed |
| `eval-v3-allchains-8gpu2m-151000-pairing` | `t-20260906233451-fjzmm` | Killed |
| `eval-v3-allchains-8gpu2m-151000-t4` | `t-20260906233455-djlmk` | Killed |
| `eval-v3-genonly-8gpu2m-44000-repr` | `t-20260906233500-9f7vp` | Killed |
| `eval-v3-genonly-8gpu2m-44000-cdr` | `t-20260906233504-x2n8b` | Killed |
| `eval-v3-genonly-8gpu2m-44000-pairing` | `t-20260906233508-svn2t` | Killed |
| `eval-v3-genonly-8gpu2m-44000-t4` | `t-20260906233513-mpfv5` | Killed |
| `eval-v3-bert-1m-105000-repr` | `t-20260906233517-9x8kq` | Killed |

**没动的**：`queue012` 两条非闲时 pairing（仍 `Queue`，是 headline 指标、非闲时资源，不在本次范围）；
两条 8 卡训练 `protein_esmc_llada270m_diffusion_immune_v3_8gpu_2m`（`t-20260904014842-qgjbg`）与
`..._bert_immune_v3_1m`（`t-20260902110050-zb7dr`）仍 `Running`。cancel 后 `ml_task list` 过滤
Creator ∈ {`251105016`, `zhuyiheng`} 只剩这两条 pairing，确认 `c20250601` 上我们已无任何在排 / 在跑的闲时任务。

### 4.9 多链关系对照臂（2026-09-02，YAML 未提交）

用户确认：不只留调研，开分链 t / 配对辅助损失对照。设计全文见
[`MULTI_CHAIN_RELATION.md`](MULTI_CHAIN_RELATION.md)。**2M all-chains 主跑不动。**

**诊断（未改前）**

- 现役只开 `joint_loss_ratio=1.0`：学的是 `p(H,L)` 联合重建，不是 `p(L|H)`。
- `_chain_ratios` 代码早已存在，但 `int(B * ratio)` 在 per_device 2/4 上会把非 joint 桶 floor 成 0；即便打开 YAML 也是空操作。
- pairing ImmunoMatch ≈ 0.353，贴错配地板 0.333。token CE 不约束嵌入几何。

**已落地（代码 + YAML，作业未交）**

1. `sample_chain_conditioned_timesteps` 改为 per-row `torch.multinomial`。`joint=1.0` 仍 100% joint，现役 2M 若重启 import 新代码，行为与改前一致。
2. `relation_aux.py`：`cognate` / `chain_drop` / `both` in-batch InfoNCE。配对 mask 在 `diffusion_all_chains` 扩 eligibility **之前**快照，抗原不当 heavy。aux **只在 train** 计入，`eval_loss` 仍是重建 CE。
3. CLI 已透传到 `protein_pretrain_esmc.py`；`RelationAuxLogCallback` 插在 callback 列表最前，wandb 记 `relation_aux_loss`。
4. 对照 YAML（generated-only、4 卡、global 128、50k、独立 OUTPUT_DIR）：
   - `train_jobs/protein_esmc_llada270m_diffusion_chainratio_immune_v3_4gpu.yml` — 分链 t `0.5/0.15/0.15/0.1/0.1`
   - `train_jobs/protein_esmc_llada270m_diffusion_chainratio_cognate_immune_v3_4gpu.yml` — 同上 + `--relation_aux cognate --relation_aux_weight 0.1`
5. 评测：现有 ImmunoMatch 生成 pairing + 新 `scripts/downstream/score_pairing_pll.py`（`p(L|H)−p(L)`）。模板 `eval_jobs/eval_pairing_pll.yml`。
6. 单测：`scripts/diagnostics/test_relation_aux.py`（CPU，不加载 ESMC）。

**提交纪律**

- 新 OUTPUT_DIR，from scratch。不要 `--init_fusion_weights` 指向 2M，不要改 2M YAML。
- headline = ImmunoMatch + PLL 差，不是 `eval_loss`。
- 配额未修：两条各再占一套 top-k，提交前确认 ~9 GB 余量。
- 本轮不做：intra/inter 分套注意力、结构、扩 `<tcra>`/`<tcrb>`、PLL 进 pretrain loss。

### 4.3 早期 ablation（已终态，仅作 lineage）

| Job | Task ID | 说明 |
|-----|---------|------|
| `protein_esmc_llada8b_bert_esmctrain` | `t-20260803071310-fvhz6` | bert + `add` + 可训 ESMC + 8B pretrained |
| `protein_esmc_llada8b_bert_noesmc` | `t-20260803071313-9hmjw` | bert + `token`（无 ESMC）+ 8B pretrained |
| `protein_esmc_llada8b_bert_noesmc_scratch` | `t-20260804004804-2s4rx` | `--decoder_init scratch`（d=4096/L=32/h=32） |
| `protein_esmc_llada270m_bert_esmctrain_scratch` | `t-20260804004807-tgw8s` | scratch 小 LLaDA d=768/L=8/h=12 ≈ 269.6M |

- 入口能力沿用至今：`--decoder_init {pretrained,scratch}` + `--decoder_d_model/--decoder_n_layers/--decoder_n_heads/--decoder_mlp_hidden/--decoder_weight_tying`。
- **队列规则（现行）**：默认 `queue012` + `Preemptible: false`，1×8 卡 `ml.pni2.28xlarge`。
  不要再用 `spot-share-queue`。8 卡排不上时有两条路，**先试第一条**：
  1. **降到 4 卡** `ml.pni2.14xlarge` + `Preemptible: false`（仍非闲时），把省下的乘数补进
     `per_device` 以保持 global batch 128 不变。2026-08-29 实测 19 秒起跑，而同规格 8 卡任务
     零 Running、最早的已等 9.8h —— 缺的是连续整节点 8 卡而非配额。见 §4.2.2。
  2. **借闲时资源**：另投 `c20250601` + `Preemptible: true`，但必须同时满足独立 `OUTPUT_DIR` +
     自动 resume + `RetryOptions`，三者缺一不可，见 §4.2.1。

### 4.4 v3 语料实测快照（2026-08-29 01:26–01:30，非文档转抄）

由 `python scripts/count_immune_mix.py train "oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire" tcr_papers_dir=.../data/tcr_papers_v2/dataset` 实测，`max_length=max_protein_length=1024`，全部 blocklist 生效。

| source | 目录 | raw | kept | 记录% | 残基% | 生成残基% | res/rec |
|---|---|---:|---:|---:|---:|---:|---:|
| `oas` | `data/oas_previous_clean/splits` | 2,486,442 | 2,485,471 | 32.54% | 45.24% | 50.09% | 232 |
| `ots` | `data/ots_paired_clean/final` | 2,102,715 | 2,094,231 | 27.42% | 37.28% | 41.28% | 226 |
| `tcr_repertoire` | `data/tcr_repertoire/dataset` | 1,971,794 | 1,971,794 | 25.82% | 2.03% | 2.25% | **13** |
| `tcr_papers` | `data/tcr_papers_v2/dataset` | 682,383 | 681,444 | 8.92% | 2.85% | 0.91% | 53 |
| `asd_antibody` | `downstream/asd/step6_final/antibody` | 850,134 | 276,412 | 3.62% | 11.37% | 4.58% | **523** |
| `tcr_native` | `data/tcr_native/dataset` | 143,391 | 96,552 | 1.26% | 1.07% | 0.82% | 140 |
| `trait` | `downstream/trait/step4_final` | 69,251 | 31,515 | 0.41% | 0.16% | 0.07% | 66 |
| **合计** | | | **7,637,419** | | | | |

- 总残基 1,271,950,724，生成链残基 1,148,782,869（90.3%）。
- valid 实测合计 **282,102**（`oas` 12,551 / `ots` 10,597 / `asd_antibody` 44,866 /
  `tcr_repertoire` 201,504 / `tcr_papers` 7,336 / `tcr_native` 3,855 / `trait` 1,393）。
  eval 每源截 2,000 行 → 七源近等权；只有 `trait` 填不满配额。

> 🔴 **上表测于 01:26–01:30 UTC，即 `tcr_repertoire` 近重复搬迁完成（03:35 UTC）之前，
> 已过期。** 搬迁后（2026-08-29 05:30 重测）：`tcr_repertoire` raw/kept
> 1,971,794 → **2,128,750**（记录 25.82%→**27.31%**，残基 2.03%→**2.19%**，
> 生成残基 2.25%→**2.42%**），**合计 7,637,419 → 7,794,375**，总残基
> **1,273,914,576** / 生成链残基 **1,150,746,721**。其余六源逐行不变，但占比因分母变大
> 而下移（`oas` 31.89% / `ots` 26.87% / `tcr_papers` 8.74% / `asd_antibody` 3.55% /
> `tcr_native` 1.24% / `trait` 0.40%）。
> valid 合计 282,102 → **203,647**（`tcr_repertoire` 201,504 → **123,049**），
> 进 eval 的是六源各 2,000 + `trait` 1,393 = **13,393** 行（各 14.93% / 10.40%）。
> 当前权威快照见 `DATA_FORMAT_AUDIT.md`。
- 对照：六源 v2（`tcr_papers` 指 v1 的 407,112、无 `tcr_repertoire`）为 **5,391,293** 条 /
  12.33 亿残基。`DATA_FORMAT_AUDIT.md` 里写的"v2 mix 7,366,444"**仍含 `tcr_repertoire`**，
  不是六源 YAML 真正训的语料，别混用。
- **配比是按记录等权，不是按残基**：`ImmuneSourceSpec.weight` 在 `ImmuneBioSeqDataset.__getitem__`
  中被丢弃，混合比例纯由磁盘行数决定。`tcr_repertoire` 占 25.8% 记录但只有 2.0% 残基，
  `asd_antibody` 反向（3.6% 记录 / 11.4% 残基）。
- **训练预算 < 1 epoch**：global 128 × 50,000 步 = 640 万条样本，语料 763.7 万条 → 约 **0.84 epoch**。

### 4.5 每个源渲染成什么样（2026-08-29 实测，取各源真实首行）

> 🔴 **下面的样例是"各源首行"，对 `trait` / `tcr_papers` 不具代表性。**
> `trait` 首行恰好没有 MHC，渲染成 `tcr_peptide`，但全量 `trait` 有 **95.2%
> 是 `tcr_pmhc`**（带 MHC 块）。同源内不同行的布局不同，首行不能代表分布。
> 布局的**分布**见 §4.5.1，别从这里的样例推占比。

`*[...]` 是可训练目标位，`[...]` 是固定上下文。

```
# 布局一 无标签配对生成（oas 32.5% + ots 27.4%），100% 可训练
oas   <prots> <ab>  *[heavy ×128] <chainsep> *[light ×110] <protd>      242 tok / 242 可训
ots   <prots> <tcr> *[alpha ×112] <chainsep> *[beta  ×114] <protd>      230 tok / 230 可训

# 布局二 识别型 = 固定上下文 + relation + 待生成受体
asd   <prots> [antigen ×75] <protd> <binding> <prots> <ab> *[heavy ×118] <chainsep> *[light ×106] <protd>
                                                                        306 tok / 228 可训
trait <prots> <pep> [epitope ×10] <protd> <binding> <prots> <tcr> *[cdr3a ×13] <chainsep> *[cdr3b ×14] <protd>
                                                                         45 tok /  31 可训
native <prots> [mhc ×34] <protd> <binding> <prots> <pep> [pep ×9] <protd> <binding> <prots> <tcr> *[cdr3b ×12] <protd>
                                                                         65 tok /  15 可训

# 布局三 单链无条件（tcr_repertoire 25.8%）—— 唯一的 tcr_single 布局
rep   <prots> <tcr> *[AAAAASQETQY ×11] <protd>                           14 tok /  14 可训
```

### 4.5.1 六种布局的真实 loss 预算（2026-08-29 蓄水池采样实测）

实际有**六**种布局，不是三种。`gen%` = 该布局占全部待预测残基的比例
（`scripts/count_grammar_layouts.py`，每源蓄水池采样 3 万行后按保留行数放大）：

| 布局 | 条件 → 生成 | 来源 | 保留行数 | 记录% | **gen%** |
|---|---|---|---:|---:|---:|
| `antibody_pair` | 无条件 → 抗体 H+L | `oas` | 2,485,471 | 31.9% | **49.64%** |
| `tcr_pair` | 无条件 → TCR α+β 全长 | `ots` | 2,094,231 | 26.9% | **40.92%** |
| `antigen_antibody` | 抗原 → 抗体 H+L | `asd_antibody` | 276,412 | 3.6% | **4.55%** |
| `tcr_single` | 无条件 → 单链 CDR3β | `tcr_repertoire` | 2,128,750 | 27.3% | **2.90%** |
| `tcr_pmhc` | MHC+表位 → TCR | `trait`+`tcr_native`+`tcr_papers` | 671,678 | 8.6% | **1.78%** |
| `tcr_peptide` | 仅表位 → TCR | 同上三源里无 MHC 的部分 | 137,833 | 1.8% | **0.20%** |

三源到布局的实测拆分（全量扫描，非采样）：`trait` 95.2% pmhc / 4.8% peptide；
`tcr_native` 99.9% / 0.1%；`tcr_papers` 80.3% / 19.7%。

**三个无条件/抗体布局吃掉 95.1% 的 loss 预算，表位条件生成合计只有 1.98%。**
根因是残基数量级而非行数：`antibody_pair` 每条约 232 个待预测残基、`tcr_pair` 约 231，
而 `tcr_pmhc`/`tcr_peptide` 只生成 CDR3，每条 13–31 个。
**加数据改不动这个比例**——`tcr_papers` 行数翻 3 倍也只把 1.98% 抬到约 3%。

> ⚠️ **`count_grammar_layouts.py` 2026-08-29 前用的是前缀采样（`break`），结论是错的。**
> `tcr_papers_v2` 是 7 个论文语料首尾拼接，前 3 万行 100% 是 `tcr_peptide`，
> 于是 547,274 条 `tcr_pmhc` 被误记，`tcr_pmhc` 低估约 4 倍
> （旧值 `tcr_peptide` 8.10% / `tcr_pmhc` 2.47%，实为 1.8% / 8.6%）。
> 已改蓄水池采样并加 `--seed`；新数字与独立全量扫描一致（671,678 vs 673,686）。
> 根因写在 [`DATA_PIPELINE_README.md`](DATA_PIPELINE_README.md) §1。

- `oas` 的 CSV 有 33 列（V/D/J、FR/CDR 分段、cluster id…），`row_to_record` **只取**
  `cleaned_h_sequence` / `cleaned_l_sequence` 两列。
- 链顺序：抗体按位置 heavy→light；**TCR 按角色重排为 α→β**（`grammar.py` 用
  `_first_chain(record, {"tcr_alpha"})`），而 OTS 的 record 里其实是 β 在前。
- `tcr_native` 与 `tcr_papers` 用**同一套统一 schema**，故共用 `tcr_native_row_to_record`：
  优先全长 Fv，缺失则退回 CDR3 loop；有 MHC 渲染 `tcr_pmhc`（三段），无则 `tcr_epitope`（两段）。
- 词表侧：全语料只激活 42 个新 token 中的一小部分——语法 token 实际只出现
  `<prots> <protd> <ab> <tcr> <pep> <chainsep> <binding> <nonbinding>` 八个，残基只用 20 个标准氨基酸。
  继承的 12.6 万 BPE 词表在本任务里基本是死重量。
- collator 同时产 `encoder_input_ids` / `encoder_residue_mask` / `encoder_chain_mask` 供 ESMC，
  且加噪位在 encoder 侧镜像打 `<mask>`（防条件通路泄漏，见 2026-08-24 条目）。

### 4.6 `asd_antibody` 为什么只剩 32.5%（2026-08-29 实测拆解）

常被追问，结论：**不是训练集内部去重，是与下游 benchmark 的去污染**，且几乎全部来自 heavy 链桶。

| 阶段 | 剩余 | 删掉 |
|---|---:|---:|
| 原始 train.csv | 850,134 | — |
| benchmark 去污染 | 278,554 | **571,580（67.2%）** |
| 长度过滤（1024） | **276,412** | 2,142（0.25%） |

- 判据：heavy 链 **0.95 id / 0.80 cov** 或 CDR-H3 core 0.80/0.80（light 不参与）。按行统计命中桶：
  heavy 单独 479,611 + heavy&h3 81,203 + 仅 h3 10,766 → **98% 的删除来自 heavy 桶**。
- 根因是该源主体为**点突变库**：`buzz` 占 523,721 行（61.6%），实测抽样 20 万条全部 120 aa、
  全部唯一，但相对库内共识只差 1–10 个位点（中位 6，56.5% 单独就 ≥0.95 identity）。
  `mmseqs easy-linclust --cluster-mode 1` 是连通分量（传递闭包），整库并成极少数巨簇，
  簇内只要有一条命中 benchmark 就整簇删 → `buzz` 掉 83.0%。
- 删除有实据、非纯聚类误伤：`exact_heavy=4,419` 条 benchmark heavy 链**逐字**出现在 ASD 中；
  `flab_koenig2017`/`flab_warszawski2019`/`ab-bind`/`flab_hie2022`/`flab_rosace2023` 被删 **100%**，
  `skempiv2` 99.5%、`structures-antibodies` 97.3% —— 而 bank 正是 FLAb + SAbDab CDR-infilling +
  comp_chain OAS holdout。反面证据：`met` 删 0%、`genbank` 0.1%、`aae` 0%。
- 诚实标注：连通分量按设计**过删**，那 43 万条 `buzz` 里有一部分是被邻居链传递连上的。
  要回收行数，杠杆是 `--cluster-mode`（换非传递）或提高 heavy 阈值，**不是**动长度上限。
  该 67.49% 代价已于 2026-08-28 决策接受，三个 benchmark 全保护，不放宽阈值。
- 注意：删完后它仍以 3.62% 的记录吃掉 **11.37% 的残基**（抗原中位 607 aa），算力占比远大于行数占比。

### 4.7 「BERT 表征效果一般」的定位（2026-08-30 实测，无 bug）

**问题**：先验上 BERT 目标应更利于表征任务，但 §0.1–§0.3 里 BERT 臂并不占优（T2 ARI
270m BERT 0.0186 vs diffusion 0.0277）。需要判断是评测坏了还是现象为真。

**先证伪「pipeline 有 bug」**（三项都过）：

1. checkpoint 加载无缺口 —— BERT 与 diffusion 的 `model.safetensors` **键集逐个相同**
   （各 386 张量，`condition_norm.{weight,bias}` / `condition_proj.weight` 两侧都在）。
   注意 `load_fusion_for_eval` 对 missing key 只 `logger.warning`，而 decoder 是
   `LLaDAModelLM(cfg, init_params=False)` 建的，真缺 key 会拿到**未初始化内存**当权重 —— 这次没缺。
2. 打分口径无误 —— 独立 harness（`scripts/diagnostics/diag_repr_layers.py`）在同一份
   T2 basis-B 数据上复现出 **BERT 0.0187 / diffusion 0.0275**，对上 RESULTS 的 0.0186 / 0.0277。
3. padding 没有污染 final feature —— 同一条序列 alone / 同长 batch / 被 10 倍长序列 pad 到 125 宽，
   三种情况下最后一层 pooled feature 余弦 **0.999998**（残差是 bf16 数值噪声）。
   LLaDA `forward` 把 `attention_mask` 正确转成 additive `-inf` bias，训练与评测同走 collator
   的 mask（入口里**没有** `NoAttentionMaskWrapper`，§2 表里那句已过时）。

**再定位真因**：现象为真，但**很大一部分是最后一层的几何造成的，不是 BERT 的预训练目标更差**。

约束前提：表征必须用 **decoder 最后一层**的 feature（不换层、不用 ESMC encoder 特征）。
在这个约束下用「换一种缩放」作**探针**去量化几何代价，并用**官方脚本**跑。

> 🚫 **口径决策（2026-08-31）：不采用任何重标定，headline 保持 raw。**
> 一次性加过的 opt-in `post=` spec token **已从 `common/model_api.py` 移除**，代码回到 raw-only。
> 下面的数字是**诊断探针，不是我们主张的性能**。

**官方 T3 deep k=200 macro NN AUROC**（`run_paper6.py`，6 pMHC / universe 25,816 / 100 seeds）

| 探针读出 | BERT@49000 | Diffusion@42000 |
|---|---:|---:|
| **raw（headline）** | **0.712** | **0.709** |
| 仅去中心化 | 0.696 | 0.693 |
| PCA 白化 256 | 0.760 | 0.755 |

**官方 T2 basis-B ARI**：raw 0.019 / 0.028 → 去中心化 0.023 / 0.038 → 白化 256 0.019 / 0.023。

两条可引用的结论（**都不依赖采用后处理**）：

1. **表内绝对值是下界。** 仅换缩放就能在 T3 上多拿约 **0.047**，
   **大于表里任何模型间差异**（此前最大是 8B diff vs 270m diff 的 0.011），
   也比 top-k 波动带（≤0.004）大一个数量级。这不是调参余量，是几何退化的度量。
   根因：mean pairwise cosine 从 layer0 的 0.71/0.58 涨到最后一层 **0.994/0.986**，
   eff-rank 38→17 / 42→20；逐层 probe 也证明最后一层线性可解码性最差
   （BERT 0.734→0.646、diffusion 0.737→0.683，单调下降）。
2. 🔴 **两臂差距的成因分两种，须分开说**（2026-08-31 配对重测，**推翻本节前一版**）。
   前一版写「probe 差距 +0.0363 塌到 −0.0055、kNN@1 排序翻转」并据此说 BERT 反超 ——
   那是**把噪声当信号**（原 probe 是单次 `train_test_split(random_state=0)`、ARI 单个种子、
   无方差列）。补测让两臂在**同一重采样**上打分（配对统计量）：

| 指标 | 读出 | Δ = diff − BERT | 配对 sd | |
|---|---|---:|---:|---|
| 25 类 probe | **raw** | **+0.0330** | 0.0064 | ✅ **5.1 sd，20/20 同号** |
| 25 类 probe | 白化 | −0.0039 | 0.0056 | ❌ 0.7 sd = 0 |
| K-sweep ARI | **raw** | **+0.0092** | 0.0012 | ✅ **7.6 sd，8/8 同号** |
| K-sweep ARI | 白化 | +0.0027 | 0.0018 | ❌ 1.5 sd |
| kNN@1 | raw / 白化 | +0.0061 / −0.0043 | 0.0045 / 0.0038 | ❌ 均 CI 跨 0 |

   **raw 下 diffusion 确实真的领先**（>5 sd），最初的观察不是噪声。但：

   - ✅ **线性可解码性：BERT 不缺信息、缺的是几何。** 只做方差均衡，
     **BERT 涨 +0.0818（9.2 sd）而 diffusion 只涨 +0.0449**，probe 差距抹平到 0.7 sd。
     BERT 的涨幅是臂间差距的 **2.5 倍**。非泄漏（只在训练划分上拟合白化，数值几乎不变）。
   - 🔴 **全局簇结构：diffusion 的优势没被解释掉。** ARI 差距缩小**不是 BERT 变好**
     （+0.0014，1.3 sd）**而是白化把 diffusion 弄坏了**（−0.0052，4.1 sd）。

   🚫 作废：~~「局部邻域 BERT ≥ diffusion」~~、~~「白化后排序翻转」~~。
   §0.3 那个 24 类 probe（0.804 vs 0.794）也是**无误差棒单点值**（≈1.6 sd），不应引用；
   T3 deep/broad 有 100 seeds 波动带，可引用。

> 🟠 **作用域警告：8B 上 T3 三项全部反向**（deep k=200 BERT 0.7093–0.7133 vs diff
> **0.7198–0.7204**、broad k=100 0.669 vs **0.680**、probe 0.816 vs **0.828**）。
> 8B 未做配对重测。**加上「每格 n=1 次训练、两条 ckpt 步数不同（49000 vs 42000）」，
> 本工作不支持任何「哪个预训练目标更适合表征」的结论。**
> 未验证假设：§0.2 记录 8B BERT 余弦挤到 **0.9999**（四条最极端），受的几何惩罚最重，
> 若几何解释成立，8B 的差距也应塌掉。**补跑 8B 配对测量是最有价值的一项后续。**

**为什么不采用重标定**：它是 post-hoc 补救，回答不了「为什么模型本身产不出可直接度量的空间」。
我们的预训练目标是 token 级重建，从未约束嵌入空间几何，所以长成这样是**预期行为**；
SCEPTR 不需要重标定，是因为对比学习（InfoNCE + L2 归一化）在训练时就把空间约束成近似各向同性。
**durable 的修法在训练侧（加空间约束/对比项），不在读出侧。**

一个可能的解释（**未做反事实训练验证**）：BERT 只掩 15% + `add` 条件下可训练 ESMC 看得到 85%
干净序列 ⇒ 任务偏局部，学出好的局部邻域结构（对应 `eval_loss` 0.2578 vs 0.5638）；
diffusion 的 `t~U(0,1)` 常掩掉大半序列、逼 decoder 建全局结构 ⇒ 簇更成形。
支持性观测：BERT 最后一个 block + `ln_f` 把上下文推回输入附近（CKA 对 layer0：
layer7 0.478 → layer8 0.809），最终层 **92.3%** 方差可由 ESMC 线性预测（diffusion 89.0%），
BERT eff-rank 单调塌到 11.3 而 diffusion 中间层升到 55.0。

> ⚠️ **本节早期版本曾写「BERT 臂最好的表征是 ESMC encoder 自身、LLaDA decoder 净负贡献」，
> 那是基于 raw 特征 probe 读数的结论，已撤回** —— 换缩放后 decoder feature 的 T3 探针达 0.760，
> 远高于任何 ESMC-only 读数。

> 🔴 **结论：现表既低估两臂绝对水平（T3 约 0.047），其两臂差距也被几何污染**
> （尤其 T2 上 diffusion 的优势有相当部分来自它各向异性更轻）。
> **单项指标不足以支撑「哪条臂表征更好」**，须按上面的分工陈述。

**产物**：`scripts/diagnostics/{diag_repr_layers,diag_layer_cka,diag_batch_invariance,diag_final_readout,plot_repr_diag,plot_readout_effect}.py`
（纯诊断，不在评测路径上）。`common/model_api.py` 保持 **raw-only** ——
曾加的 opt-in `post=` 已按决策移除。
数据 `output/repr_diagnostics/{layer_sweep.csv,final_readout_ladder.csv,layer_cka.csv}`、
`downstream/benchmark/outputs/tcr_representation_paper6/diag_t3_*/`、`outputs/tcr_clustering_embed/diag_*/`；
图 `debug/figures/{readout_effect_official,repr_layer_diagnosis}.png`。
📄 **完整调试报告：[`debug/BERT_REPR_DEBUG.md`](../../debug/BERT_REPR_DEBUG.md)**；
详表见 [`RESULTS.md`](../../downstream/benchmark/RESULTS.md) §0.0 缺陷 (f)。
**T1 与 8B 两条未测探针**（单 ckpt 成本高）。
**读出保持 raw、§0 表格不回填 —— 这是决策结果，不是待办。**

## 5. 运行方式

本地 dry_run（不加载 8B，仅验证数据/词表/渲染）：

```bash
cd dllm_test
python examples/llada/protein_pretrain.py --dry_run True --max_rows_per_source 8
```

全参 FSDP 正式训练（推荐，8×A100/H100 80G）：

```bash
cd dllm_test
accelerate launch --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/llada/protein_pretrain.py --train_mode diffusion
# BERT 范式：--train_mode bert --mask_prob 0.15
```

单卡 LoRA（调试用，需先 resize 再 peft，脚本已处理）：

```bash
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/llada/protein_pretrain.py --lora True --train_mode diffusion
```

> 环境：`/vepfs-mlp2/c20250601/251105016/conda/envs/wan_ic`（torch 2.4.1 / transformers 4.57.6）。联网下载走代理 `http://100.68.162.212:3128`。

## 6. 验证记录

- **dry_run**：antibody 渲染为 `<prots> <ab> <res_*>… <chainsep> … <protd>`，tcr 为 `<prots> <tcr> …`；maskable 全覆盖；labels 在可 mask 位等于 input_ids。
- **小随机 LLaDA（d_model=128, n_layers=2，3 步真实训练）**：
  - `diffusion`：LinearAlpha + scheduler 权重，loss≈11，通过并保存 checkpoint。
  - `bert`：ConstantAlpha(α=0.85) + uniform 权重，loss≈1.7，通过并保存。
  - `diffusion+LoRA`：`Applying LoRA after resize`，trainable 含 wte/ff_out，通过。
- 重映射覆盖 renderer 对 OAS/OTS 产出的全部 42 个 grammar id（残基 + `<prots>/<protd>/<ab>/<tcr>` + 分隔符）。

## 7. 待办 / 风险

- [ ] **回填 121000 / 137000 下游数字**：121000 表征已 Success；生成三条与 137000 表征已提交。写入 RESULTS §0.8 组 D。
- [ ] **回填 73000 下游数字**：`ours_fusion_v3_allchains_8gpu2m_73000` 表征/CDR/T4 已 Success，pairing 被杀；写入 RESULTS §0.8 组 D。
- [ ] **回填 26000 / 69000 下游数字**：26000 / 69000 四条已 Success。
- [x] **回填全链 33000 / 18000 下游数字**：8/8 Success，已写入 RESULTS §0.1–§0.6 / §0.8 组 D 与 README。见 §4.2.8。
- [ ] **提交多链关系对照臂**（配额允许时）：
      `..._chainratio_immune_v3_4gpu` 与 `..._chainratio_cognate_...`。
      不要碰 2M。headline = ImmunoMatch + PLL。见 §4.9。
- [x] **磁盘回收已执行**：`output/` 1.2 T → **767 G**（释放 **389 GB**），
      含删作废目录 / 剪 8B resume-only / `checkpoint-final` 改硬链接 / 清账本外孤儿。明细见 §4.2.7。
- [ ] 🔴 **配额本身仍未修**（回收只是把窄门往后推）。`..._8gpu_2m` 每 1000 步仍要过一次
      9.0 GB 的一次性余量，`TopKValLossCheckpointCallback` 也还在继续产孤儿。
- [ ] **修 `TopKValLossCheckpointCallback` 的账本重置**：启动时应从磁盘现存 checkpoint 重建
      top-k，而不是只看本进程 `log_history`（否则每次重启都产孤儿，永不被剪）。
- [ ] **save 前做配额预检**：余量不足时降级为 weights-only，而不是让整个任务 `exit 1`
      并被 `RetryOptions` 拖进 58 分钟一轮的循环。
- [ ] **grammar_v2 五条 339 G 是否保留**：内部无 optimizer 只能整份删；`..._7l` 被 `PROJ_GUIDE.md`
      钉为 TCR-β Track-A canonical，其余四条待科研决策。
- [ ] **确认 2M 步的规划意图**：2.95 s/it 下 2,000,000 步 ≈ **68 天**独占 8 卡（§4.2.7）。
- [ ] train loss（~7.3）与 `eval_loss`（0.7028）差一个数量级，原因未查明（§4.2.7）。
- [x] 真实 8B 正式训练（当前跑 `bert_esmctrain` / `bert_noesmc` ablation）。
- [x] v3 270m 双臂提交（diffusion `t-20260829031748-96vjf` / bert `t-20260829031757-2qnjn`，均 `queue012`）。
- [x] v3 闲时版双臂提交（diffusion `t-20260829135424-zxdjg` / bert `t-20260829135427-426cj`，
      `c20250601` + `Preemptible: true` + 自动 resume + `RetryOptions`，见 §4.2.1）。
- [x] v3 **4 卡非闲时** diffusion 提交（`t-20260829143036-w7488`，`c20250601` +
      `ml.pni2.14xlarge` + `Preemptible: false`，global batch 仍 128）—— **19 秒起跑**，见 §4.2.2。
- [x] **4 卡这条已确认进入训练步**：06:35 建 wandb offline run，step 20 → 220 的 loss
      97.12 → 12.92、`grad_norm` 33–43 有限、`learning_rate` 在 step 220 为 1.1e-5
      （= 1e-4 × 220/2000，warmup 与 optimizer step 计数均正确）。
- [x] **闲时那两条已挂上看护循环**（`scripts/monitor_spot_tasks*`，10 分钟轮询，
      终态自动重提并从 last ckpt 续跑，见 §4.2.3）。**人工 cancel 前先 `touch
      output/_monitor/STOP`**，否则会被重提回来。
- [ ] **补 4 卡 bert 臂**：v3 的核心是双臂对比，现在 4 卡只有 diffusion。等这条确认稳定后，
      照 `..._diffusion_immune_v3_4gpu.yml` 改目标函数那 5 行即可（`--train_objective bert`
      + `--bert_all_chains True` 等，逐字对照 8 卡 bert 版）。
- [ ] **盯 8 卡四条**：`queue012` 那两条已排队 18.6h（该队列 0 Running、15 Queue，含大量 foldbench
      worker）；闲时那两条看 `c20250601` 何时有空闲 8 卡。**同一臂哪条先出 checkpoint 就 cancel 其余的**。
- [ ] 闲时版首次被抢占后**必须核对是否真的自动续训**：看新实例日志里
      `[resume] continuing from .../checkpoint-N`（若打的是 `training from scratch`，说明
      `RetryOptions` 或 resume 拼接没生效，白跑）。看护的 `output/_monitor/monitor.log` 会记下
      每次重提的 `resume_from`，可与实例日志交叉核对 —— 但**平台自己的 `RetryOptions` 重试不经过
      看护**（任务不落终态），那种情况只能看实例日志。
- [ ] 监控 v3 至 `max_steps=50000`；同语料下对比 bert vs diffusion 的 eval_loss 与下游。
- [x] `tcr_repertoire` 近重复搬迁 —— **2026-08-29 03:35 UTC 已完成**（实测 38.9%，非抽样估的
      36.7%）。原计划留给 v4，但搬迁完成时两臂都还在 Queue、未读过任何 CSV，所以两臂仍读
      同一份语料、可比性未受影响。详见 §4.2「caveat 1 的后续」。
- [ ] T4 Setting-A 全量参考集：修正版已生成、未覆盖线上、未重跑打分（见 DATA_PIPELINE_README §6）。
- [ ] 确认 `LLaDA-8B-Base` 的 `weight_tying`（若为 True，safetensors 保存需处理共享张量；小模型冒烟已用非绑定配置绕开）。
- [ ] 数据规模：`ImmuneCsvDataset` 会把整表读入内存（train.csv GB 级），大规模训练需评估内存或改流式。
- [x] bert 经典 80/10/10（已完成）
- [x] ckpt 省盘：`save_only_model` + `slim_checkpoints`（胖 ckpt 已现场瘦身）
- [ ] 可选：切换为保留 attention_mask。
- [ ] 序列含 `-`（gap）时未在重映射表内，会被当脏样本跳过；如需保留需补映射。

## 8. 变更日志

- **2026-09-07 09:40 CST（4 卡 generated-only diffusion 的 wandb 离线日志已同步上云）**：
  用户要求上传 `protein_esmc_llada270m_diffusion_immune_v3_4gpu`（2M generated-only 的权重源）的 wandb 日志。
  盘上 `output/protein_esmc_llada270m_diffusion_immune_v3_4gpu/wandb/wandb/` 有 6 个 offline run：
  主跑 `offline-run-20260829_063512-sq8nkmtx`（23 MB，08-29 06:35Z → 08-30 05:13Z，step 20→42780、
  eval 1000→42000 共 42 点）+ 5 个 544 KB 的重试残片（§4.2.6 死循环，每个都从 `checkpoint-42000`
  续起只记到 42780，与主跑尾段完全重复）。**只同步了主跑**：
  `wandb sync --entity codema --project bioseq-llada8b-fusion` →
  <https://wandb.ai/codema/bioseq-llada8b-fusion/runs/sq8nkmtx>，API 回读 2139 个 train 点、42 个 eval 点、
  最好 eval 42000 / 0.7548，config `per_device 4 × ga 8 × 4 卡 = global 128`、cosine、50k。
  5 个残片未传（要传随时可用同一命令）。同 project 里已有 12 个 run，含 09-06 同步的两条 8 卡长跑快照。

- **2026-09-07 07:20 CST（`c20250601` 旧闲时九条全部 cancel）**：
  用户要求停掉用于测评的闲时任务。九条（`t-20260906233442-d2gzt` / `…-fjzmm` / `…-djlmk` /
  `t-20260907001930-qtf2s` / `t-20260906233500-9f7vp` / `…-x2n8b` / `…-svn2t` / `…-mpfv5` / `…-9x8kq`）
  cancel 前全部仍 `Queue`，`ml_task cancel` 九条 `cancel success`，`get` 回读终态 `Killed`。
  昨天报的 `StopCustomTask` 未授权这次没出现，控制台那步不用走了；`CreateCustomTask` 未测。
  §4.2.9 / §4.2.10 的双跑覆盖风险解除。`queue012` 两条非闲时 pairing 与两条 8 卡训练未动。
  看护循环未动（只管训练任务，不会重提评测）。详见 §4.2.11。

- **2026-09-07（九条下游改投 `queue012` 非闲时；`c20250601` 权限已失）**：
  用户要求把没起跑的单卡任务改成非闲时重提。§4.2.8 那九条在 `c20250601` 闲时排了 4.9 小时
  全是 `Queue`。只关抢占重提被 IAM 拒（本账号对 `q-20260121145036-6fztt` 无 `CreateCustomTask`），
  经探针任务 `t-20260907044147-jvzpn` 验证 `queue012` 可提交可取消后，九条 YAML 改
  `queue012` + `Preemptible: false` 重提，新 ID `t-20260907044259-qd52v` / `…-ks4x6` / `…-btjxr` /
  `…-wlm9h` / `…-mtzn9` / `…-kfhk2` / `…-ht8b6` / `…-tcmhs` / `…-wwfjn`，
  已 `get --format` 回读确认非抢占。🔴 旧九条 creator 是 `251105016`，本账号无 `StopCustomTask`，
  **需控制台停掉**，否则同名同产物前缀会双跑覆盖。详见 §4.2.9。
  同时发现全链 2M 续训 `t-20260902021013-bm79q` 已于 09-06T19:37:05Z **Killed**（`signal: 15`），
  盘上最新满包 `checkpoint-161000`，本轮未重提，待决策。

- **2026-09-07（补提 151000 CDR）**：
  闲时 `t-20260906233447-rh2j9` 起跑 102 秒后 Killed，只写到 SAbDab `cdrh1` 开头。
  原 YAML 重提 `t-20260907001930-qtf2s`。其余 8 条仍 Queue。不打断训练。

- **2026-09-06（三条长跑最好 val 点下游，闲时单卡）**：
  用户要求评当前最好 `eval_loss` 点。全链 **151000 / 0.6326** 全套
  （repr `t-20260906233442-d2gzt`、cdr `…-rh2j9`、pairing `…-fjzmm`、t4 `…-djlmk`）；
  generated-only 8 卡 2M **44000 / 0.7520** 全套
  （repr `t-20260906233500-9f7vp`、cdr `…-x2n8b`、pairing `…-svn2t`、t4 `…-mpfv5`）；
  BERT 1M **105000 / 0.4263** 只表征（`t-20260906233517-9x8kq`）。
  权重硬链接 `eval_snapshot_{151000,44000,105000}/`。不打断三条训练。

- **2026-09-06（T4 `max_iter` sweep，闲时单卡）**：
  用户问生成阶段的 sampling iter。现状：CDR `CDR_MAX_ITER=2`（可 env 覆盖）、
  pairing `PAIR_MAX_ITER=124`（可 env 覆盖）、**T4 写死 `--max-iter 32` 且输出不带 iter 标签**。
  CDR 与 pairing 的 iter 影响已有答案（CDR 随步数单调下降；pairing 32→124 我们只 +0.002~0.003），
  **只有 T4 从未扫过**。新增 `scripts/downstream/sweep_immune_fusion_t4_maxiter.sh`
  （产物全部带 iter 标签、按 metrics.json 存在跳过、可被抢占后续跑），
  不改 `run_immune_fusion_gen.sh`（该文件正被在跑作业按字节读取，改它会破坏运行中的 bash）。
  提交 121000 上 iter ∈ {8,32,64,128}：`t-20260906132835-xnnlf`。

- **2026-09-06（最新满包 `checkpoint-144000` 表征 T1/T2/T3，闲时单卡）**：
  8 卡 2M 仍 Running（`t-20260902021013-bm79q`）。用户要求测最新点。
  144000 eval 0.6386（同 run 最好仍是 137000 / 0.6332）。
  tag `ours_fusion_v3_allchains_8gpu2m_144000`，repr `t-20260906132412-tklfb`。
  权重硬链接 `eval_snapshot_144000/`。不打断续训。

- **2026-09-06（补交 121000 生成 + 当前最好 137000 表征）**：
  用户确认补评。121000 表征已 Success（`t-20260905191346-rbt8z`）。
  122000 原包已被 top-k 剪掉；当前最好改为 137000 / 0.6332，最新满包 143000 / 0.6536。
  闲时提交：137000 repr `t-20260906125806-xkg7q`，121000 cdr `t-20260906125811-fdrvj`、
  pairing `t-20260906125815-q8h8b`、t4 `t-20260906125820-db5bd`。不打断续训。

- **2026-09-05（续训 top-k 最好 `checkpoint-121000` 表征 T1/T2/T3，闲时单卡）**：
  8 卡 2M 仍 Running（`t-20260902021013-bm79q`，最新满包 122000 / eval 0.6504）。
  评 top-k 最好 121000（eval 0.6416），只评表征，tag `ours_fusion_v3_allchains_8gpu2m_121000`，
  repr `t-20260905191346-rbt8z`。权重硬链接 `eval_snapshot_121000/`。不打断续训。

- **2026-09-04（BERT 1M `checkpoint-34000` 表征 T1/T2/T3，闲时单卡）**：
  `..._v3_1m` 仍 Running（`t-20260902110050-zb7dr`，最新满包 34000 / eval 0.4647）。
  只评表征，tag `ours_fusion_v3_bert_1m_34000`，repr `t-20260904104944-mdr7x`。
  权重硬链接 `eval_snapshot_34000/`。不打断续训。同 run 33000 val 更好，本轮按用户指定评 34000。

- **2026-09-04（补提 69000 T4）**：
  原 ckpt 已被 top-k 剪掉，权重留在 `eval_snapshot_69000/`。表征/CDR/pairing 已 Success；
  T4 `t-20260904013725-7zcvf` Killed。闲时重提 `t-20260904104404-t6j7v`。
  18000 / 26000 原 ckpt 同样被剪，但四条下游产物齐全，未重跑。

- **2026-09-04（续训 top-k 最好 `checkpoint-73000` 全套下游，闲时单卡）**：
  8 卡 2M 仍 Running（`t-20260902021013-bm79q`，最新满包 83000 / eval 0.6567）。
  评 top-k 最好 73000（eval 0.6535），tag `ours_fusion_v3_allchains_8gpu2m_73000`。
  闲时四条：repr `t-20260904104145-77cfr`、cdr `t-20260904104148-6x5m9`、
  pairing `t-20260904104152-pxjd4`、t4 `t-20260904104155-tctxm`。不打断续训。
  69000 表征/CDR/pairing 已 Success；其 T4 `t-20260904013725-7zcvf` 被闲时抢走。

- **2026-09-04（续训 top-k 最好 `checkpoint-69000` 全套下游，闲时单卡）**：
  8 卡 2M 仍 Running（`t-20260902021013-bm79q`，最新满包 72000 / eval 0.6609）。
  评 top-k 最好 69000（eval 0.6577），tag `ours_fusion_v3_allchains_8gpu2m_69000`。
  首提非抢占四条已 cancel；闲时重提 repr `t-20260904013714-4w8nk`、cdr `t-20260904013717-rhmzz`、
  pairing `t-20260904013720-nprjc`、t4 `t-20260904013725-7zcvf`。不打断续训。

- **2026-09-06（续训 `checkpoint-26000` 全套 4/4 Success，数字已回填）**：
  T1 seen AUPRC 0.761、T2 ARI 0.022、T3 deep 0.707、T4 common 8.58、
  SAb23H2 H3 34.8、pairing IM **0.429**。相对 18k：pairing 回一点（0.416→0.429），
  T4 略好（8.73→8.58）；相对 33000 仍无系统性变好。见 RESULTS §0.8 组 D。
- **2026-09-02（续训最新 `checkpoint-26000` 全套下游已提交）**：
  8 卡 2M 仍 Running（`t-20260902021013-bm79q`），盘上最新满包 26000（eval 0.6902）。
  tag `ours_fusion_v3_allchains_8gpu2m_26000`。四条：
  repr `t-20260902104333-87ccm`、cdr `t-20260902104338-7ptgf`、
  pairing `t-20260902104342-68qz8`、t4 `t-20260902104346-882r9`。
  同 run 内 25000 val 0.6861 更好，本轮按「最新」评 26000。不打断续训。
  RESULTS §0.8 组 D 已加 `allch@26k` 列（数字 `—`）。

- **2026-09-02（多链关系对照臂落地，作业未提交，见 §4.9）**：
  用户确认走实验而不是只留调研。`sample_chain_conditioned_timesteps` 改为 per-row
  multinomial（修 `int(B*ratio)` 在 per_device 2/4 上空操作）；`relation_aux`
  cognate / chain_drop InfoNCE 接入 fusion；两条 generated-only 4 卡 YAML +
  `score_pairing_pll.py` + `test_relation_aux.py` + `MULTI_CHAIN_RELATION.md`。
  **未改、未 resume 2M**。`joint=1.0` 行为不变。

- **2026-09-02（全链两个 ckpt 全套下游 8/8 Success，数字已回填，见 §4.2.8 / RESULTS §0.8 组 D）**：
  4 卡 `checkpoint-33000` 与 8 卡 2M `checkpoint-18000` 各 T1–T4 + CDR + pairing 全部 Success。
  表征与 generated-only 同档。**配对抬起来了**：IM 0.436 / 0.416 vs 组 C 的 0.353。T4 仍差（common 8.62 / 8.73）。
  18k 相对 33000 没有系统性变好（0.9% 进度快照）。
- **2026-09-02（全链两个 ckpt 全套下游已提交，见 §4.2.8）**：
  评 4 卡 `checkpoint-33000`（tag `ours_fusion_v3_allchains_33000`，eval 0.6866）
  与 8 卡 2M 续训 `checkpoint-18000`（tag `ours_fusion_v3_allchains_8gpu2m_18000`，eval 0.7044）。
  各 4 条：repr / CDR / pairing / T4。`c20250601` 非抢占单卡。
  ID：33000 `t-20260902044100-wnkdb` / `...-29sqz` / `...-ddnrw` / `...-xxnp7`；
  18000 `t-20260902044117-vh48s` / `...-q2rt5` / `...-mw2s6` / `...-zx5vk`。

- **2026-09-01（8 卡长跑 1M/2M 起跑；🔴 磁盘配额 4 连 Failed，详见 §4.2.7）**：
  - 50k 短跑收口，方向转为**步数拉长一个数量级**。新增/启用四条：
    `..._diffusion_allchains_immune_v3_8gpu_2m`（**Running** `t-20260902021013-bm79q`）、
    `..._bert_immune_v3_1m`（Queue `t-20260901235433-q76qw`）、
    `..._diffusion_immune_v3_spot_2m`（Queue `t-20260901032620-vvngv`）；
    4 卡版 `t-20260901032611-npf27` 已 **Killed**（换 8 卡）。全部 **global 256 + polynomial**，
    与 50k 那批**不可混排**。
  - `..._8gpu_2m` 首跑只能 `--init_fusion_weights` 加载 4 卡 `checkpoint-33000` 的
    `model.safetensors`（FSDP pack 不能跨 world size 续），optimizer 与 polynomial 从 step 0 新建。
  - 🔴 **事故：`safetensors_rust.SafetensorError: ... Disk quota exceeded (os error 122)`**，
    崩在 `Trainer._save`，**同一处 4 连 Failed**（首跑 14h 到 step 17000 后爆，3 次重试各 58 min
    从 16000 续、跑满 1000 步再爆）→ **净进度 0，约 3 小时 8 卡白烧**。
    `df` 显示底层 FS 尚余 **809T**，爆的是**目录/租户配额**，只看 `df` 会误判方向。
    第 5 次于 19:10:42 存盘成功，但只是期间别的任务腾出空间，**配额仍贴临界、未修**。
  - ✅ **`pick_latest_full` 的三件套判据救了一次**：它要求
    `optimizer.bin`+`pytorch_model_fsdp.bin`+`scheduler.pt` 齐全，正确跳过了只剩 132 MiB
    半截 `model.safetensors` 的 `checkpoint-17000`、回退 16000，没有重演 §4.2.6 的
    「从坏 ckpt 反复续跑」死循环。**改 checkpoint 挑选逻辑时必须保留这个判据。**
  - **新发现：`TopKValLossCheckpointCallback` 在重启时重置账本** —— 它只看本进程的
    `log_history`，看不到上一个进程存过什么，于是每次崩溃-重启都把上一轮 checkpoint 甩成
    账本外孤儿、永不被剪，崩溃循环自己在抬高占用（正反馈）。
    全 `output/` 只读盘点：checkpoint 合计 **790.8 G**，其中 resume-only **328.3 G**、
    账本外孤儿 **132.9 G**；`checkpoint-final` 与 `checkpoint-50000/model.safetensors`
    **md5 逐字节相同**却各存一份，白占 **69.1 G**。
  - ✅ **已执行回收：`output/` 1.2 T → 767 G，释放 389 GB**（19:30Z）。删作废目录、剪两条已终态
    8B run 的 resume-only、5 处 `checkpoint-final` **改硬链接**（不是删 —— 10 处 YAML 引用要保住）、
    清 6 条 run 的账本外孤儿。刻意保留 `..._allchains_..._4gpu/checkpoint-33000`（2M 的权重来源）
    与 `..._bert_immune_v3/checkpoint-50000` 满包（1M 的 resume 来源）。
    **配额本身没修**，callback 仍在产孤儿、save 仍要 9.0 GB 一次性余量。
  - **核实非 bug 两项**：resume 健康（续跑后 step 16300–17000 的 loss/lr 与首跑同 step 逐条吻合）；
    train loss ~7.3 vs `eval_loss` 0.7028 差一个数量级**原因未查明**但两路 CE 计算相同、
    首跑与 resume 一致，登记为未解项，不得跨口径比较。

- **2026-09-01（口径：BERT 只评表征，不评生成）**：
  用户确认 BERT 只跑 T1/T2/T3。已 cancel 仍在跑的 pairing `t-20260901113309-p7cp9`。
  CDR / T4 虽已跑完，数字不进 RESULTS、不当 headline。此后只提交 `eval_v3_bert_final_repr.yml`。

- **2026-09-01（BERT v3 下游改投 `c20250601` 闲时）**：
  YAML `ResourceQueueName` 从 `queue012` 改为 `c20250601`，`Preemptible: true` 不变。
  新作业：repr `t-20260901113303-kqd4h`、cdr `t-20260901113306-p6c46`、
  pairing `t-20260901113309-p7cp9`、t4 `t-20260901113312-zmxm9`（提交后已 Staging/Queue）。
  queue012 旧四条（`t-20260901034613-4ktrf` 等）本账号无 `StopCustomTask`，cancel 失败，需控制台停掉以免双跑。

- **2026-09-01（BERT v3 续跑 1M，LR 对齐 AirGen polynomial）**：
  从 `checkpoint-50000` 满包 resume（`checkpoint-final` 不能续）。调度改为与
  AirGen DPLM 相同的 **polynomial power=1**，峰值 **4e-5**，地板 **`lr_end=1e-5`**，
  warmup=0。`max_steps=1000000`，global batch 256（ga 16 × 8 卡）。
  任务 `protein_esmc_llada270m_bert_immune_v3_1m` → **`t-20260901030607-zmrrv`**（queue012）。
  YAML：`train_jobs/protein_esmc_llada270m_bert_immune_v3_1m.yml`。

- **2026-09-01（BERT v3 `checkpoint-final` 下游评测，queue012 闲时）**：
  同一套 T1/T2/T3 + CDR + pairing + T4，ckpt=`output/protein_esmc_llada270m_bert_immune_v3/checkpoint-final`，
  tag `ours_fusion_v3_bert_final`，单卡 `ml.pni2.3xlarge`。首次投 `queue012` 非抢占（`t-20260901024417-swk8q` 等四条）卡住未起，
  已 cancel。当前账号对 `c20250601` 无 `CreateCustomTask` 权限，改投 **`queue012` + `Preemptible: true`**。
  `eval-v3-bert-final-repr` `t-20260901034613-4ktrf`；
  `cdr` `t-20260901034616-brp6f`；`pairing` `t-20260901034620-wdhpf`；`t4` `t-20260901034623-sw25w`。
  YAML：`eval_jobs/eval_v3_bert_final_{repr,cdr,pairing,t4}.yml`。

- **2026-09-01（v3 下游收数 + 完整对照写入 RESULTS §0.8）**：
  - 8 卡 `ckpt-27000`：T1/T2/T3/CDR/T4/pairing 全齐。4 卡 `ckpt-42000`：表征+CDR 齐，pairing/T4 首次 CUDA Failed。
  - 数字按 tag 写入 §0.1–§0.6；新增 **§0.8** 把官方 baseline / 旧四条 / 两条 v3 放同一张跨任务表，并写明三组不可混排。
  - 方向：T1/T3 持平旧 270m；T2/T4/pairing 比旧 diffusion 更弱。pairing IM 0.353 贴错配地板 0.333。

- **2026-08-30（定位「BERT 表征效果一般」：无 bug，根因是读出口径 + BERT 目标被 ESMC 抄近路）**：
  - **先排除 pipeline 故障**：BERT 与 diffusion checkpoint 的 safetensors **键集逐个相同**
    （各 386 张量），且独立 harness 复现出 T2 ARI **BERT 0.0187 / diffusion 0.0275**
    ↔ RESULTS 记录的 0.0186 / 0.0277。**现象为真，不是评测坏了。**
    顺带记一个隐患：`load_fusion_for_eval` 对 missing key 只 `logger.warning`，而 decoder 由
    `LLaDAModelLM(cfg, init_params=False)` 建，真缺 key 会拿**未初始化内存**当权重 —— 这次没缺，
    但该处应改成 fail-fast。
  - **新增诊断工具** `scripts/diagnostics/{diag_repr_layers,diag_layer_cka,plot_repr_diag}.py`：
    一次前向抓全部 9 个 hidden state，逐层 × {raw, center, zscore} 算 T2 K-means ARI/NMI/Purity
    + 25 类 logistic probe，外加各向异性（mean pairwise cosine / effective rank）、
    层间 CKA、ridge R²(ESMC→layer)，以及「抽掉 ESMC 条件」消融。
  - **发现 1 —— headline 读出取了最差的一层且没去中心化**：两臂 probe 都在最后一层触底
    （BERT 0.734→0.646、diffusion 0.737→0.683），mean pairwise cosine 涨到 **0.994/0.986**，
    eff-rank 38→17 / 42→20。**只加一步去中心化**，diffusion T2 ARI 0.0275 → **0.0382**，
    越过 SCEPTR 的 0.033 —— §0.2 里「仍低于 SCEPTR」这个结论是读出造成的。
    RESULTS 里原记为「8B BERT 特例」的余弦挤压其实是**四条的共性**，270m 上同样存在。
  - **发现 2 —— BERT 臂被惩罚得更重，且有机制**：BERT 最后一个 block + `ln_f` 把上下文
    **推回输入附近**（CKA 对 layer0：layer7 0.478 → layer8 0.809），最终层 **92.3%** 方差
    可由 ESMC 线性预测，eff-rank 单调塌到 11.3；diffusion 中间层 eff-rank 反升到 55.0。
    **BERT 臂全模型最好的表征是 ESMC encoder 自身（probe 0.7535）**，LLaDA decoder 净负贡献。
    根因：`add` 条件 + 可训练 ESMC-300M + 只掩 15% ⇒ encoder 看得到 85% 干净序列、
    基本把 MLM 解掉（对应 `eval_loss` 0.2578 vs diffusion 0.5638），decoder 退化成读出头。
    并且 BERT 训练把可训练 ESMC 自己也带偏（encoder mean cos 0.956 / eff-rank 39.4，
    diffusion 臂 0.528 / 56.1）。
  - **影响面**：不改任何已落盘数字（口径未变、可复算），但
    **「BERT 表征不如 diffusion」不能推广成「BERT 预训练目标更差」**。已写入
    [`RESULTS.md`](../../downstream/benchmark/RESULTS.md) §0.0 缺陷 **(f)** 与本文档 §4.7。

- **2026-08-31（补误差棒：推翻自己前一天的两条结论）**：质疑"这些图到底能不能得出有效结论"后
  发现 §4.7 的 probe 是**单次** `train_test_split(random_state=0)`、ARI 是**单个** K-means 种子，
  CSV 里**没有任何方差列** —— 所以引用的 ±0.005 级差异全是无误差棒的点估计。
  新增 `scripts/diagnostics/diag_noise_floor.py`（两臂在同一重采样上打分，配对统计；
  probe 20 次分层划分 / ARI 8 种子 / kNN@1 2000 次配对 bootstrap）与
  `plot_noise_floor.py`（图 `debug/figures/noise_floor.png`）。
  - **撤回两条**：① 「probe 差距 +0.0363 塌到 −0.0055、kNN@1 排序翻转 ⇒ BERT 反超」——
    −0.0039 只有 0.7 sd，是**抹平到 0 而非反转**；② 「局部邻域度量 BERT ≥ diffusion」——
    kNN@1 两个读出下 95% CI 都跨 0，**从未可分辨**。
  - **立住的部分**：raw 下 diffusion 真的领先（probe +0.0330 = 5.1 sd / 20-20 同号、
    ARI +0.0092 = 7.6 sd / 8-8 同号）；probe 差距可由几何解释（BERT 方差均衡后
    +0.0818 = 9.2 sd，是 diff +0.0449 的 1.8 倍、臂间差距的 2.5 倍）；
    **且非 transductive 泄漏**（只在训练划分拟合白化，数值几乎不变）。
  - **新发现的重要区别**：ARI 差距缩小**不是 BERT 变好**（+0.0014，1.3 sd）**而是白化把
    diffusion 弄坏了**（−0.0052，4.1 sd）⇒ **diffusion 的聚类优势没有被解释掉**。
    先前把 probe 与 ARI 混作一个"几何"故事是不对的。
  - 顺带记一条引用纪律：§0.3 的 24 类 probe（0.804 vs 0.794）**也是无误差棒单点值**（≈1.6 sd），
    不应作为证据；T3 deep/broad 有 100 seeds 波动带，可引用。

- **2026-08-31（修复：评测加载器改 fail-fast，杜绝"拿未初始化内存当权重"）**：
  - **问题**：`load_fusion_for_eval` 对 missing key 只 `logger.warning`，而 decoder 由
    `LLaDAModelLM(cfg, init_params=False)` 构建（**分配后不初始化**）⇒ 真缺 key 时张量拿的是
    分配器残留内存，评测会**静默地给一个随机权重打分**，长跑里那条 warning 基本看不到。
  - **修法**（不用"允许缺失"白名单）：加载**前** `_poison_parameters` 把所有浮点参数灌 NaN，
    加载后 `_assert_fully_loaded` 凡残留 NaN 即 `RuntimeError`，并区分
    「缺权重」与「权重在盘上就是 NaN（训练发散）」。**权重绑定自动放过**
    （绑定伙伴填共享存储 ⇒ 无残留 NaN），无需特例。
  - **验证**（`scripts/diagnostics/test_fail_fast_load.py`）：270m BERT/diffusion 各 386 参数、
    8B BERT/diffusion 各 602 参数，**四条全部 residual NaN = 0（含 8B 的 `weight_tying`，无误报）**；
    删掉 `blocks.3.attn_out.weight` 的负向测试正确抛错。回归：`diag_batch_invariance` 在 GPU 上
    复现逐位一致的 0.99999826，官方 embedder 入口正常出 768 维 finite 特征。
  - **顺带修另一处**：`LLaDAModelLM.__init__` 自建内层模型时**硬编码 `init_device="cuda"`**
    （`modeling_llada.py:1460`，注释"always on CPU"已过期），使 `device="cpu"` 失效、
    8B 在忙卡上直接 OOM，`torch.cuda.is_available()` 兜底形同虚设。
    **未改共享的 `modeling_llada.py`**，改为在 loader 里自建 `LLaDAModel` 再经
    `LLaDAModelLM(cfg, model=inner)` 传入。

- **2026-08-31（决策：不采用任何读出重标定，`post=` 代码已移除；诊断结论保留为「口径代价」）**：
  - **决策**：headline 读出保持 **raw**（pooled 向量原样进余弦/K-means），
    **不去中心化、不 z-score、不白化**。前一条日志里新增的 opt-in `post=` spec token
    已从 `common/model_api.py` **整体移除**（`parse_grammar_embedder_spec` 回到 3 元组，
    `FusionGrammarEmbedder` 回到 raw-only），已验证语法与 3 例 spec 解析正常。
  - **理由**：重标定是 post-hoc 补救，回答不了「为什么模型本身产不出可直接度量的空间」。
    我们的预训练目标是 token 级重建，从未约束嵌入空间几何 ⇒ 各向异性是**预期行为**；
    SCEPTR 不需要它是因为对比学习在训练时就把空间约束成近似各向同性。
    **durable 修法在训练侧（加空间约束/对比项），不在读出侧。**
  - **但 §4.7 的测量全部保留为「口径代价」**（可引用，不依赖采用后处理）：
    ① 表内绝对值是下界，T3 上约 0.047 的几何损失；
    ② **raw 读出部分扭曲两臂比较** —— probe 差距 raw +0.0330（5.1 sd）经方差均衡后
    **抹平到 −0.0039（0.7 sd = 0）**；但 T2 聚类差距不受此解释（见 2026-08-31 补误差棒条目）。
  - **同时作废的说法**（已在 RESULTS §0.0 (f) 用 🚫 标出）：「越过 CDR3-Levenshtein/k-mer」、
    「与 TCRdist 差 0.017」、任何对 ESM2/ProtBERT/TCR-BERT 的「反超」——
    headline 下与 TCRdist 仍差 0.065、与 SCEPTR 仍差 0.078。
  - 🟠 **另一处收紧：两臂胜负随规模翻转，探针只覆盖 270m。** 核对 §0.3 后发现
    8B 上 deep/broad/probe **三项全部是 diffusion 领先**（0.7198–0.7204 vs 0.7093–0.7133、
    0.680 vs 0.669、0.828 vs 0.816，波动带均不重叠）。故「BERT 局部邻域更强」**只对 270m 成立**，
    已在 RESULTS (f-2)、§4.7 与状态表加作用域警告。**补跑 8B 探针是最有价值的后续**：
    §0.2 记录 8B BERT 余弦挤到 0.9999（四条最极端），若几何解释成立，8B 的差距也应塌掉。

- **2026-08-30（续：用官方脚本验证，前一条对 BERT 的结论被推翻并更正）**：
  - 约束：**表征与生成都只用 decoder 最后一层的 feature**，所以只动后处理、不换层、不用 ESMC 特征。
  - **曾给官方 embedder 加 opt-in 后处理** `post=`（`center`/`zscore`/`pcawK`），默认行为逐位不变；
    **已于 2026-08-31 按决策移除**（见上一条）。实现时踩到的坑值得记：变换必须
    **拟合一次后冻结** —— `common/fewshot.py::embedding_distance_matrix`
    对 query / ref 是**两次独立** `embed_pairs` 调用，逐次拟合会把两边放进不同空间、
    静默污染所有距离。
  - 🔴 **官方 T3 deep k=200 macro NN AUROC**（`run_paper6.py`，非 proxy）：
    `none` BERT **0.712** / diff **0.709**（复现 §0.3 记录）→ `center` 0.696 / 0.693
    → **`pcaw256` 0.760 / 0.755**。**+0.047 的增益大于表里任何模型间差异**
    （此前最大 0.011），比 top-k 波动带（≤0.004）大一个数量级。
    **§0.3 的结论因此改变**：从「低于 TCRdist(0.777)/SCEPTR(0.790) 约 0.06–0.07」变为
    越过 CDR3-Levenshtein(0.733) 与 k-mer(3)(0.728)、与 TCRdist 差 **0.017**、与 SCEPTR 差 **0.030**。
    增益非维度巧合：kNN@1 在 pcaw-64/128/256/512/768 上是 0.297–0.311 平台（raw 0.274）。
  - **官方 T2 basis-B ARI**：`none` 0.019/0.028 → `center` 0.023/**0.038** → `pcaw256` 0.019/0.023。
    **T2 与 T3 想要的后处理相反**（center 抬 T2 压 T3，pcaw 反之），说明原始几何不适合直接做度量。
  - ✅ **更正上一条**：前一条写的「BERT 臂最好的表征是 ESMC encoder 自身、LLaDA decoder 净负贡献」
    **是基于未白化特征的错误结论，已撤回**。公平读出下两臂各擅一类度量：
    **局部邻域（T3 NN、kNN@1）BERT ≥ diffusion**（0.760 vs 0.755，且全部白化维度上 BERT 领先）；
    **全局簇结构（T2 ARI）diffusion 领先**。这与「BERT 更适合表征」的先验其实是一致的。
  - **又排除一个 bug**：padding 未污染 final feature —— 同一序列 alone / 同长 batch /
    被 pad 到 125 宽，最后一层 pooled 余弦 **0.999998**。LLaDA `forward` 正确把
    `attention_mask` 转成 additive `-inf`；入口里**没有** `NoAttentionMaskWrapper`（§2 表那句过时）。
  - **仍未做**：T1 未测（成本高，其 MLP 头原则上能自学重标定但有 dropout 0.3 + wd，属未验证预测）；
    8B 两条未测；`post=` 未设为默认、§0 表格未回填；`load_fusion_for_eval` 的 missing-key
    仍是 warning 而非 fail-fast。

- **2026-08-30 07:30 UTC（定位并记录空位符阻塞；新增残基字母表守卫脚本）**：
  - **新增 `scripts/data/assert_residue_alphabet.py`**：扫描语料序列列，任何字符落在
    `RESIDUES`（`LAGVSERTIDPKQNFYMHWCXBUZO`）之外就 **exit 1**，可与
    `assert_corpus_fresh.py` 并列放进 entrypoint 做提交前 gate。
    按"该列大多数非空值都是纯残基"判定序列列，避免 `chain_id` 之类元数据列误报。
  - **用它定位到 4 卡 diffusion 反复崩溃的真凶**：`ReamapCollator` 的映射表不覆盖 ESMC 词表里的
    `.`(29) / `-`(30) / `|`(31)，而 `data/tcr_papers_v2/dataset/train.csv` 第 **3046** 行
    `cdr3b` = `ASSKVAARVP-TLKLS` 混进了一个比对空位符。850 万行 train split 里**只有这 1 行**；
    holdout 五源 19 万行全干净（所以 eval 通路不受影响）。
    因为 HF Trainer 续跑复现采样顺序，这是**必然重现的重试死循环**：
    `w7488` → `7dgwx` → `hhpm5` → **`r5w9k`**，已用掉 3/5 次重试。
    完整分析、两条修法与去污核查见 **§4.2.6**；**语料尚未修改，待拍板**。
  - 新增 **§4.2.0 现状快照**（七条任务的步数 / `eval_loss` / ckpt / ETA），并明确写下
    `eval_loss` 的**三层不可比切割线**（目标函数、loss 覆盖范围、batch 口径），
    避免后续拿 BERT 的 0.60 和 diffusion 的 0.75 直接比大小。

- **2026-08-30 05:55 UTC（两条改用 global 256 并补齐 4 卡 bert；per_device 加倍失败，改走 ga）**：
  - **最终配置：`per_device 4 × ga 16 × 4 卡 = global 256`**，两条均 **Running**：
    `..._diffusion_allchains_immune_v3_4gpu` → **`t-20260830135521-zf5rr`**、
    新建的 `..._bert_immune_v3_4gpu` → **`t-20260830135524-qq7n5`**。
    step 280 时零报错、loss 稳步下降（diffusion 全链 24.68 → 21.50，bert 14.59 → 12.41），
    ETA 约 48-50 h。
  - 🔴 **走了一次弯路，教训值得记**：先试的是 `per_device 4 → 8`（`t-20260830134534-86bml` /
    `t-20260830134537-8ncgx`），**两条都在 step 2 OOM**（`76.29 GiB is allocated by PyTorch`），
    而本机探测预测只有约 43 GiB，**低估 33 GiB**。根因是探测脚本只喂 `asd_antibody`
    （编码器输入 `(8, 2, 870)`，每样本 **2 条链**），而真实七源混合里 TRAIT / TCR 样本带
    peptide + MHC + TCRα + TCRβ，**链数可达 4-5**；ESMC 激活按
    `per_device × 链数 × 链长` 走，按单源定容必然乐观。
    放大误差的结构性原因：FSDP 的 `TRANSFORMER_BASED_WRAP` 只包 `LLaDALlamaBlock`，
    **ESMC 编码器整个不分片**（每卡持有完整 300M + bf16 混精下的 fp32 master），
    且 `fsdp_activation_checkpointing: false`、编码器侧没有梯度检查点。
    **下次定容必须用完整七源混合，并对 `[B, C, L]` 里的 C 取最坏值，不能只挑最长的行。**
  - **改走 ga 是零风险的**：梯度累积不改变任何 forward/backward 的张量形状，显存画像与
    `per_device 4` 完全相同，而 `per_device 4` 已被 generated-only 那条跑满 42k 步实证。
    `diff` 核对两条与 `..._diffusion_immune_v3_4gpu` 的 `per_device_train_batch_size 4`
    逐字相同，只有 `ga` 从 8 变 16。详见 §4.2.5。
  - 期间 `..._diffusion_allchains` 的 global 128 首跑（`t-20260830125655-hngmn`）被 cancel，
    输出目录改名为 `output/ABORTED_gb128_...`（9.1G，可删），确保 `RESUME_ARG` 探测不到
    checkpoint、新 run 必然 from scratch —— 否则会拿 128 口径的 optimizer state 接 256 的
    schedule。OOM 那两次的输出目录已直接删除。
  - **新增 `train_jobs/protein_esmc_llada270m_bert_immune_v3_4gpu.yml`**，补齐 bert 臂在 4 卡
    非闲时的缺位（此前只有 8 卡版在排队）。从 8 卡 `_spot` 版派生，`diff` 后只差 batch 与
    `run_name`，`--bert_all_chains True` / mask 比例 / 七源目录全部一致。
  - 每次提交均 `assert_corpus_fresh.py` PASS、`ml_task export --config` 回读确认
    `per_device 4` / `ga 16` / flavor / `RetryOptions` / 各自目标函数 flag 无误。
  - 🔴 **代价**：`max_steps` 未动，样本量 640 万 → **1280 万 ≈ 1.64 epoch**，墙钟约翻倍；
    这两条**与 global 128 的五条不可比**（含已到 42k 的 generated-only 那条），两条之间可比。
    曾评估 `per_device 8 × ga 4 = global 128`（同口径纯提速），用户明确选择 256 ——
    事后看那个方案也会 OOM，因为它同样要 `per_device 8`。
  - ⚠️ **`RetryOptions` 会把 OOM 配置反复重提，且新任务与好任务同名同 `OUTPUT_DIR`。**
    OOM 的 bert 任务被自动重提为 `t-20260830135809-zq7pg`（回读确认仍是 `per_device 8`），
    与新提的好任务 `t-20260830135524-qq7n5` **同名、共用同一 `OUTPUT_DIR`**，已手动 cancel。
    教训：**配置性失败（OOM / 参数错）后必须立刻 cancel 整条重试链**，否则它会一边烧配额
    一边往正确任务的输出目录里写东西。`PolicySets: [Failed]` 分不清"节点故障"和"配置错误"。
  - 顺带核实：`..._diffusion_immune_v3_4gpu` 于 05:23 UTC 经 `RetryOptions` 重试换 id 为
    **`t-20260830131621-7dgwx`**，已从 `checkpoint-42000` 正确续上、wandb 持续写入，剩 8000 步。
    **但它随后又在 step 42782 崩了** —— 不是节点故障，是语料里一个空位符 `-`，
    会无限复现。详见 **§4.2.6**（含全语料扫描结果与去污核查）。
    `queue012` 的 8 卡 `..._diffusion_immune_v3`（`t-20260829031748-96vjf`）排队 1.9 天后起跑
    —— 它在重跑 4 卡那条已完成 84% 的同一臂，但走 `zhuyiheng` 队列、不占 `c20250601` 配额，
    故暂未 cancel（待决策）。现在 4 卡那条卡死在 42000，这条反而可能成为该臂的接班人。

- **2026-08-30 04:56 UTC（新增「全链 diffusion」开关并提交 4 卡非闲时任务）**：
  - **新能力 `--diffusion_all_chains`**（默认 `False`，现存行为逐位不变）：打开后固定上下文
    （抗原 / MHC / peptide）也参与加噪并计入 diffusion loss，即**没有任何链是 fixed 的**。
    这是 BERT 那边 `--bert_all_chains` 的镜像，补上了 v3 两臂在 loss 覆盖范围上的不对称。
  - 改动 2 个文件：`protein_fusion_model.py` 加 `all_residue_eligible_mask()` + 模型参数；
    `protein_pretrain_esmc.py` 加 `TrainingArguments` 字段、透传、启动日志。实现只覆写
    `diffusion_eligible_mask` / `diffusion_loss_mask` 为 `residue_mask & attention_mask`
    —— 时间步与加噪采样器都从这两个 mask 推导 eligible 集合，故一次覆写同时改到加噪、
    labels 与 ESMC 编码器镜像。**加噪算法与精度未动**（仍 per-seq `t ~ U(eps,1)` + Bernoulli(t)，bf16）。
  - **本机 A100 实测**（ASD batch，含真实抗原上下文）：关 → 计入 loss 659 位、固定上下文 **0** 位；
    开 → 计入 1452 位、其中固定上下文 **805** 位且全部被加噪，**编码器被 mask 的位置数 = 1452
    = 计入 loss 的位置数**，证明条件通路没漏答案。两侧梯度正常、loss 有限。详见 §4.2.4。
  - 新增 `train_jobs/protein_esmc_llada270m_diffusion_allchains_immune_v3_4gpu.yml`
    → **`t-20260830125655-hngmn`**，`c20250601` / **非抢占** / 4 卡 `ml.pni2.14xlarge`。
    与 `_4gpu` 那条 `diff` 仅差该 flag 与名字/目录/Tags，训练超参与数据参数逐字节一致
    （global batch **128**）。⚠️ **该 task 已于同日 05:45 UTC cancel 并改用 global 256 重提**，
    见下一条；此处保留是为记录首跑的验证数据。
  - 提交前 `assert_corpus_fresh.py` PASS、`--dry_run True` 确认 flag 解析、
    `ml_task export --config` 回读确认平台侧记录无误。**首跑 from scratch**（目标函数变了，
    generated-only 的 checkpoint 不可续）。
  - **step 1000 已闭环**：train loss 96.88 → 5.21，`eval_loss` 1.1463，checkpoint 落盘且
    `resumable=true`。⚠️ 该值与 generated-only 的 1.0461 **不可直接比大小** —— 全链版在更大更难的
    token 集合上平均，跨臂比较要走下游 benchmark 或另建只算生成链的 eval，见 §4.2.4。
  - 已同步 [`README.md`](README.md) 任务表与说明。

- **2026-08-29 06:52 UTC（闲时两条挂上看护循环：掉了自动从 last ckpt 重提续跑）**：
  - 新增 `scripts/monitor_spot_tasks.py` + `_start.sh` / `_stop.sh`（零第三方依赖，系统
    `python3` 直接跑）。每 10 分钟查 `volc ml_task list --name <TaskName> -o json`，
    只看最新一条同名任务：`Failed`/`Killed` → 用同一 YAML 重提（entrypoint 的 `RESUME_ARG`
    自动接最新 checkpoint）；`Success` → 停止看护；其余状态不动作。已后台常驻。
  - **补的是 `RetryOptions` 的盲区**：平台那层只管「实例被回收后重试」，重试次数用尽或任务
    落到终态之后就不再管，训练会永久停住。
  - 关键设计（每条对应一个真实的坑，细节见 §4.2.3）：① 必须**只看最新一条** ——
    `--name` 是模糊匹配且返回历史终态任务（如 cancel 重提留下的 `t-20260829135133-5np6m`/
    `Killed`），不取最新会被历史骗到、无限重提；② **STOP 哨兵** —— `Killed` 无法区分「被抢占」
    与「人工 cancel」，**人工 cancel 前必须先 `touch output/_monitor/STOP`**；③ 重提冷却
    15 分钟 + 上限 100 次；④ 停滞只告警不自动 kill（eval 会拉长 wandb 写入间隔，误杀代价大）；
    ⑤ wrapper 让看护自身崩溃后 10 秒自愈；⑥ PID 文件互斥防止双重重提。
  - **测试**：13 组 25 个断言全通过（注入假状态覆盖全部分支），并实测了重复启动被挡、
    `stop` 干净清理、`kill -9` 后自愈（日志留下 `rc=137` 后拉起新 pid 的记录）。
  - 已同步 [`README.md`](README.md)：闲时资源那节加第 4 条「外部看护循环」+ 独立小节写用法与
    STOP 哨兵警告。

- **2026-08-29 06:30 UTC（改提 4 卡非闲时，19 秒起跑 —— 8 卡排不上的真正原因是凑不出整节点）**：
  - **诊断**：闲时版提交 40 分钟仍 `Queue`。查 `c20250601` 实况：7 个 Running **全是单卡**
    `ml.pni2.3xlarge`，7 个 Queue **全是 8 卡** `ml.pni2.28xlarge`（含我们两条，最早的等了 9.8h，
    其中两条别人的 8 卡抢占任务也等了 2h+）。即**缺的是连续整节点 8 卡，不是队列配额**，
    也不是抢占或优先级问题（`Priority` 已是用户可提交的最高档 6，平台只开放 2/4/6）。
  - 新增 `train_jobs/protein_esmc_llada270m_diffusion_immune_v3_4gpu.yml`：
    `ml.pni2.28xlarge`（8 卡）→ **`ml.pni2.14xlarge`（4 卡）**，`Preemptible: false`（**非闲时**），
    队列仍 `c20250601`。4 卡非抢占在该队列有先例（`Protenix-v2/train_jobs/*_1node4gpu*.yml`）。
    → **`t-20260829143036-w7488`**，06:30:36 提交、06:30:55 Running。
  - **global batch 保持 128**：`per_device_train_batch_size` 2 → **4**，`ga` 仍 8，
    4 × 8 × 4 = 128 = 8 卡版的 2 × 8 × 8。不动 `ga`/`max_steps`/lr，故与 8 卡四条逐一可比。
  - **提交前实测显存**（本机 A100-80G 单卡、无 FSDP，是 4 卡 FSDP 的上界）：训练步峰值 **30.7GB**、
    保存 checkpoint 瞬时 **56.4GB**，均在 80GB 内。据此才敢把 `per_device` 从 2 提到 4
    （而非保守地用 per_device 2 × ga 16）。分阶段曲线与换算见 §4.2.2。
  - 与闲时版的其余差异：`OUTPUT_DIR`/`run_name` 换 `_4gpu` 后缀（五条并跑不能共用目录）；
    `RetryOptions` 只留 `PolicySets: ["Failed"]`、`MaxRetryTimes` 50 → 5（非抢占，不会被回收）；
    resume 逻辑**保留**（给故障重试兜底）。数据与训练超参对 8 卡版**零差异**，已 `diff` 逐行核过。
  - **仍缺 bert 臂的 4 卡版** —— v3 的核心是双臂对比，本次只提了 diffusion。
  - 已同步 [`README.md`](README.md)：任务表加第五条、提交指引改为「8 卡排不上先降 4 卡，再考虑闲时」。

- **2026-08-29 05:54 UTC（v3 双臂改投闲时资源，`queue012` 排队 18.6h 未起）**：
  - 新增 `train_jobs/protein_esmc_llada270m_{diffusion,bert}_immune_v3_spot.yml`：队列
    `queue012` → **`c20250601`**（`q-20260121145036-6fztt`），`Preemptible: false` → **true**
    （闲时/抢占资源），仍是 1×8 卡 `ml.pni2.28xlarge`。**训练超参与数据参数对非抢占版零差异**
    （`diff` 逐行核过），两条 spot 之间也只差目标函数那 5 行 —— v3 的可比性不受影响。
  - 提交：diffusion → **`t-20260829135424-zxdjg`**，bert → **`t-20260829135427-426cj`**，
    均 `Queue`。`queue012` 那两条**保留未 cancel**（非抢占一旦起跑更稳），四条 `OUTPUT_DIR`
    互不重叠，哪一对先出 checkpoint 就 cancel 另一对。
  - 为抢占场景做的三件事（缺一不可，细节与实测见 §4.2.1）：① `OUTPUT_DIR` 加 `_spot` 后缀，
    避免与非抢占版共写同一批 checkpoint；② 去掉 `test ! -d checkpoint-1000` 反 resume 断言，
    改为 entrypoint 里取编号最大的 checkpoint 拼 `--resume_from_checkpoint`（`set -euo pipefail`
    下 `ls` 无匹配必须 `|| true` 兜住，四种目录状态已本地实测）；③ 加 `RetryOptions`
    （`MaxRetryTimes=50`、`PolicySets=[Failed, InstanceReclaimed]`）—— 平台提示
    **「原抢占任务被抢后的强制重试功能已下线」**，不自己开就等于被抢一次训练永久中断。
  - **首轮提交漏了 `RetryOptions`**（`t-20260829135133-5np6m` / `t-20260829135158-2wnsp`），
    因 CLI 无 update 子命令、改 YAML 不回写已提交任务，只能在 Queue 阶段 cancel 重提；
    那两个 id 现为 `Killed`，**不要引用**。重提后用 `volc ml_task export -t <id> --config`
    回读确认平台侧确实记下了 `RetryOptions`。
  - 提交前本地跑过 `assert_corpus_fresh.py`（两个语料 PASS，blocklist 未变），确保任务拿到
    资源后不会卡在第一道 pre-flight。
  - **本次是全仓库第一个配 `RetryOptions` 的任务**（实测：五个 `*_jobs/` 目录共 1266 个任务
    YAML，其中 **866 个 `Preemptible: true`**，配 `RetryOptions` 的为 **0**）。那些配置写在平台
    下线「抢占后强制重试」之前，当时无需自己配 —— 所以**照抄现成的抢占 YAML 当模板会静默
    丢掉这层保护**。已把该提示写进 [`README.md`](README.md)「提交训练任务」第 3 条。
    顺带核到队列口径：`c20250601` 是本项目主力队列（1269 处引用），`spot-share-queue` 95、
    `queue012` 23。
- **2026-08-29 下午（布局口径纠错 + 两个新发现）**：
  - **修 `scripts/count_grammar_layouts.py` 的前缀采样偏差。** 它原来读到
    `--per-source` 行就 `break`，取的是**前缀**而非随机样本。`tcr_papers_v2` 是 7 个
    论文语料首尾拼接的，前 3 万行 **100% 是 `tcr_peptide`**，而全量是 80.3% `tcr_pmhc` ——
    于是 547,274 条被归错布局，`tcr_pmhc` 低估约 **4 倍**。改成蓄水池采样并加 `--seed`；
    新数字与独立全量扫描一致（671,678 vs 673,686，采样噪声内）。
    同时把 `rows=` 列改名 `kept=`（它统计的是加载期过滤后的行数，不是磁盘行数，旧名会误读）。
    受影响的旧结论已在 §4.5.1 与 `DATA_PIPELINE_README.md` §1 更正。
  - **补 §4.5.1 六布局 loss 预算表。** 实际有六种布局不是三种；三个无条件/抗体布局占
    **95.1%** 的待预测残基，表位条件生成（`tcr_pmhc`+`tcr_peptide`）合计仅 **1.98%**。
    根因是残基数量级（232 vs 13–31）而非行数，**加数据改不动**，必须 per-sample 加权。
  - **新发现①：单链 α 摄入不了，卡在 grammar 而非数据。** `GrammarTokenizer` 只有一个
    `<tcr>`，没有 `<tcra>`/`<tcrb>`；配对时靠 `[alpha, beta]` 位置编码身份，
    但 `len(receptor) == 1` 时 α 与 β 渲染成**完全相同的 token 序列**
    （`grammar.py:466-472`）。硬塞会污染 T4 Setting-A 测的 β 分布。
    要摄入须扩词表 → 现有 checkpoint 不兼容，属建模决策，未做。
  - **新发现②：加载期过滤在 train/valid 上丢弃率严重不对称。**
    `asd_antibody` train 丢 **67.5%** 但 valid 只丢 **5.0%**（13 倍差）；
    `tcr_native` 32.7% vs 12.2%；`trait` 54.5% vs 54.3% 对称。
    后果是 **ASD 的 valid loss 不能用于 early-stopping** —— valid 保留了大量被从
    train 剥掉的 Kong 相似簇抗体家族，在测一个训练时被刻意屏蔽的分布。
    无泄漏风险（valid 非 Kong 基准本身），机制未查清，待决策。
    两项均记入 `DATA_PIPELINE_README.md` §6.2 / §6.4。
- **2026-08-29（v3 双臂提交 + 语料实测回填）**：
  - 新建 `train_jobs/protein_esmc_llada270m_bert_immune_v3.yml`，数据参数与
    `protein_esmc_llada270m_diffusion_immune_v3.yml` **逐字节一致**（v1/v2 的 BERT 臂是六源、
    diffusion 臂是七源，两个目标不可比；这是 v3 存在的唯一理由）。
  - `volc ml_task submit`：diffusion → **`t-20260829031748-96vjf`**，bert → **`t-20260829031757-2qnjn`**。
    队列 **`queue012`**（`q-20260524172355-rnqtf`），`Preemptible: false`，`Priority: 6`，
    1×8 卡 `ml.pni2.28xlarge`，from scratch，不 resume。提交后实测状态均为 `Queue`。
  - 第一次提交因 `Description` 超平台 500 字符失败；YAML `Description` 压成十来个字
    （`270m diffusion 七源 v3` / `270m BERT 七源 v3`），细节一律写本文档 §4.2–§4.6 与
    [`DATA_PIPELINE_README.md`](DATA_PIPELINE_README.md)。改 YAML 描述**不会**回写已排队任务。
  - 实测七源 train **7,637,419** 条 / 12.72 亿残基（§4.4）；`asd_antibody` 67% 删减是
    benchmark 去污染不是内部去重（§4.6）。接受两个 caveat：repertoire 近重复未搬迁、
    trait 黑名单 provenance 未同步（过滤更严）。语料冻结至 v4。
- **2026-08-28（数据扩充 + 去污审计；细节见 [`DATA_PIPELINE_README.md`](DATA_PIPELINE_README.md)）**：
  - **新增数据源 `tcr_repertoire`**（1,971,794 行无表位单链 CDR3β，来自 TcrDesign-2026
    `pretrain/bCDR3_train.csv`）。它是**唯一**渲染成 `tcr_single` 的源，而 T4 Setting-A
    无条件生成基准正是用这个布局解码 —— 此前该布局训练覆盖率为 **0%**。
    入口新增 `--tcr_repertoire_dir`；`datasets.py` 新增 `tcr_repertoire_row_to_record`。
  - **`tcr_papers` → `tcr_papers_v2`**：并入 TcrDesign-2026 三个表位层
    （+279,923 行 / +882 净新表位 / 配对 αβ 3.6×），共 682,383 行。
  - **新增训练配置** `train_jobs/protein_esmc_llada270m_diffusion_immune_v3.yml`：
    `--dataset_args oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire`，
    并把语料新鲜度断言作为 pre-flight 步骤。
  - **修 3 个去污缺陷**（三次都是"过滤看起来生效了，其实没有"，根因各不相同）：
    ① **语料过期** —— `build_report.json` 的 `PASS` 只代表"对**构建时那份**黑名单干净"，
    黑名单重建后语料不会自动跟随，两者无任何关联，导致 3 个 T4 参考 binder 留在
    `tcr_repertoire/train.csv` 而报告仍显示 `PASS: true`。修法：报告记
    `blocklist_provenance`（mtime + 内容 sha1），新增
    `scripts/data/tcr_native/assert_corpus_fresh.py` 比对活文件 hash，v3 配置调用它。
    ② **黑名单构造缺陷** —— `decontam_extra.py::decontaminate_trait` 把黑名单写成
    `语料 ∩ benchmark`，只对构建时那份语料完备；实测 862 键仅覆盖 hard-requirement
    保护集的 **4.2%**（277/6,570），TRAIT 此前 0 命中**是运气不是过滤生效**。
    17:16 另一会话重建 TRAIT 加了 600 行，4 行直接落在保护 core 上。修法：改为
    `簇级交集 ∪ binding_benchmark ∪ full_bank` = 59,212 键（与 `build_repertoire.py` /
    `finalize_papers.py` 早已采用的写法一致）。代价 TRAIT 保留行 −11.2%（35,472 → 31,515）。
    ③ **审计结论行误导** —— 原来把配对键投影出的裸 core 命中也计入
    `TOTAL residual hits -> LEAKAGE`，而那些按构造就该非零。改为只用
    hard-requirement 行驱动结论。
  - **最终审计状态**：`HARD-REQUIREMENT hits = 0 -> PASS`（NM2025 seen/unseen +
    public_trackA × 5 个 TCR 源全为 0）；新增的 `tcr_repertoire` 对全部 11 个下游
    测试集均为 0。新增 `scripts/data/dedup/audit_downstream_leakage.py`（5 源 × 11 基准矩阵）。
  - **发现但未修的既存问题**：T4 Setting-A 参考集本身构造有缺陷 ——
    `prepare_tcr_generation.py --train-cap` 默认 200,000 而 OTS train 有 2,102,700 行，
    只覆盖 9.5%，导致 **19.4% 的 holdout 序列其实躺在训练数据里**（1,893/9,767，
    逐条核验全部确认在 train 内）且 novelty 参照只有全量的 10.5%。已加
    `--train-cap 0` / `--out-dir` 并生成修正版参考到
    `downstream/benchmark/data/tcr_generation_fullref/`（验证 holdout ∩ train = 0），
    **未覆盖线上文件、未重跑打分** —— 换参考会改动已有 novelty/JSD 数值，待决策。
- **2026-08-04（队列策略）**：本系列训练 YAML **统一改用 `queue012`**，不再用 `spot-share-queue`。已在跑的 pretrained 两条仍留在 spot（不停训）；新提交与所有 `train_jobs/protein_esmc_*.yml` 的 `ResourceQueueName` 已改为 `queue012`。
- **2026-08-04（from-scratch 两条新任务）**：
  - 入口 `protein_pretrain_esmc.py` 支持 `--decoder_init scratch`：从 `LLaDA-8B-Base` 只取 tokenizer/config，随机初始化 decoder；可选 `--decoder_d_model/--decoder_n_layers/--decoder_n_heads` 缩小架构（实测 d=768/L=8/h=12 ≈ **269.6M**）。
  - 提交：`bert_noesmc_scratch` → `t-20260804004804-2s4rx`；`llada270m_bert_esmctrain_scratch` → `t-20260804004807-tgw8s`（均 `queue012` / Queue）。
- **2026-08-03（进度文档纠偏：真正在跑的是 bert ablation）**：澄清正式对照不是早期 `add_diffusion`/`add_bert`，而是：
  - **`bert_esmctrain`** `t-20260803071310-fvhz6`：bert + `residue_cond_mode=add` + **可训练 ESMC**（`freeze_encoder=False`）
  - **`bert_noesmc`** `t-20260803071313-9hmjw`：bert + `residue_cond_mode=token`（无 ESMC 条件）
  - 设计意图：同为 bert 目标，对照「用/不用 ESMC 序列 feature」，且用 ESMC 时参数参与训练。两条均 **Running**（约从 2026-08-02T23:13Z 起）；磁盘进度约 esmctrain@12k、noesmc@19k。更新本节「4. 当前状态」。
- **2026-08-02/03（任务切换脉络，详见 `PROJECT_PROCESS.md`）**：`add_diffusion`/`add_bert` 先因 wandb `config.to_dict` 崩溃 cancel 并重提；后按用户澄清改为 bert ESMC-conditioning ablation，cancel 重提后的 diffusion/bert 冻结-ESMC 任务，改交 `bert_esmctrain` + `bert_noesmc`。
- **2026-08-03**：`bert` 模式改为经典 80/10/10 MLM（`BERTMLMTrainer` 覆写 `compute_loss`）；单元测试比例≈0.15/0.80/0.10/0.10，小模型 3 步训练通过。
- **2026-08-03**：`git add` 追踪本阶段相关文件（已 staged、未 commit）：`protein_pretrain.py`、`PROTEIN_PRETRAIN_PROGRESS.md`、`protein_fusion_model.py`、`protein_pretrain_esmc.py`。
- **2026-08-02**：新建入口脚本 `protein_pretrain.py`；完成 dry_run 与小模型冒烟（diffusion/bert）；修复 LoRA「先 resize 再 peft」并验证 `diffusion+LoRA`；本进度文档建立。
- **2026-08-02**：新增「协作规则（Agent Rules）」小节（实时更新进展、子任务默认用 grok 4.5）；明确现阶段改动范围为新增 2 文件、未改现有代码。
- **2026-08-02**：启动 **ESMC 条件融合 + 残基 token 消融** 新阶段（计划 `esmc-llada-fusion-ablation`）。核心发现：`qwen3_vl_arch/modeling_bioseq.py` 的 `BioSeqLLaDAEncoderDiffusionModel`（L1462）+ 继承的 `compute_loss`（L1352）已把 ESMC→LLaDA 的 encode/gather/corruption 镜像/残基位注入/masked CE 全部写好；LLaDA `forward` 原生支持 `inputs_embeds`。唯一缺口是它 backbone 从零随机初始化 + grammar 小词表。方案：复用其机制，backbone 改为继承 8B 的 LLaDA + BPE/remap，暴露 `residue_cond_mode∈{token,feature,add}`。已派 grok 4.5 子代理实现（新增 `protein_fusion_model.py` + `protein_pretrain_esmc.py` + 冒烟脚本，不改现有文件）；review 与三 mode 冒烟结果待回填。
- **2026-08-02（融合阶段 review + 验证 + 8 卡准备）**：
  - **Review 发现并修复 1 个多卡致命 bug**：`FusionTrainer.compute_loss` 原本调 `model.compute_loss(inputs)`，在 DDP/FSDP 下会绕过 wrapper 的 forward，导致梯度不同步 / 参数不 gather（单卡冒烟发现不了）。已把 `LLaDAEsmcFusion` 内部去噪逻辑改名 `_denoise`，新增真正的 `forward(**batch)` 作为训练入口，`FusionTrainer` 改为走 `model(**inputs)`。
  - **环境**：融合需同时有 `esm`(ESMC) 与 LLaDA 依赖。`wan_ic` 缺 esm；`protenix_abtcr`（生产 ESMC 训练环境，esm 3.2.3 / torch 2.8 / transformers 4.48.1）缺 peft → 已 `pip install peft==0.14.0` 补齐。
  - **两处版本解耦**（使入口能在 transformers 4.48.1 跑）：① `protein_pretrain_esmc.py` 不再 import `protein_pretrain.py`（避免 `dllm.core.schedulers → lm_eval`），就地定义 `ModelArguments/DataArguments/build_immune_specs`；② 不用 `dllm.utils.get_tokenizer/get_model`（会 import `a2d`，需 `TransformersKwargs`，4.48.1 无），改为直接 `AutoTokenizer` + `LLaDAModelLM.from_pretrained` 加载。
  - **验证**：`--dry_run` 在真实 OAS/OTS 上通过（vocab 126346→126388，+42；抗体/TCR grammar 渲染 + encoder `[B,C,L]` + remap 全 ≥0）；三 mode（token/feature/add）用**真实 ESMC-300M** 冒烟全 PASS（loss 有限下降、保存正常）。
  - **8B 权重键校验**：用 `model.safetensors.index.json` 离线对比 `LLaDAModelLM` state_dict —— 291 张量 100% 匹配（0 missing / 0 unexpected），`from_pretrained` 可直接加载。
  - **8 卡任务已就绪**：新增 `scripts/accelerate_configs/fsdp_llada.yaml`（FSDP FULL_SHARD，`transformer_layer_cls_to_wrap=LLaDALlamaBlock`）+ `train_jobs/protein_esmc_llada8b_add_smoke8card.yml`（volc `ml_task submit`，1×8 卡 `ml.pni2.28xlarge`，`protenix_abtcr` 环境，`residue_cond_mode=add`，冻结 ESMC，50 步初步 smoke）。**提交仅阻塞于 8B 权重下载完成**（下载中，进 offline 缓存，集群共享 vepfs 可见）。
- **2026-08-03（8B 权重补全 + 真实 8B 验证 + 提交 8 卡任务）**：
  - **权重下载**：`hf download` 两次中途死掉（12GB/16GB，3 个 shard 残缺 00001/00004/00005）。诊断为 XET/CDN(`us.aws.cdn.hf.co`) 走代理时 hf 内置下载器（含 hf_transfer 的 Rust 客户端）会挂；而 `curl` 走代理可稳定拉 CDN（206 断点续传）。改用 `curl -C -` 按 `x-linked-etag`(=blob sha256) 并行续传 3 个 shard，完成后重命名 blob + 建 snapshot 软链。**6/6 shard 齐，0 incomplete**。
  - **真实 8B 加载**：`LLaDAModelLM.from_pretrained` 干净加载（6 shard，8.02B 参数）。
  - **发现并修复 resize 陷阱**：LLaDA-8B embedding 补齐到 126464 行（base tokenizer 126346），而我们 +42 token 的 id(126346..126387) 正好落在这批**未用 padding 行**内。原先 `resize_token_embeddings(126388)` 会把 wte 收缩到 126388 却不动 LM head(仍 126464)，造成 in/out 尺寸错位。已改为**仅当 `len(tok)>embed_rows` 才 resize**，否则复用 padding 行、不 resize。
  - **发现并修复 dtype bug**（tiny fp32 冒烟测不出）：8B decoder 为 bf16 → `inputs_embeds` bf16，但 `condition_norm/condition_proj` 为 fp32，LayerNorm 报 `expected BFloat16 but found Float`。改为在 fusion 头自身 dtype 下算 norm/proj、再 cast 回 embedding dtype；并在入口把整模型 `.to(bf16)` 以保证 FSDP root flat-param dtype 统一。
  - **真实 8B + 真实 ESMC-300M 端到端**（add 模式，单卡 fwd+bwd）：loss 有限、logits `(1,102,126464)`、`condition_proj` 梯度范数 166（ESMC 条件路径有梯度回传）、decoder 有梯度、encoder 冻结无梯度、峰值显存 33.5GB。
  - **提交 8 卡任务**：`volc ml_task submit` → task id `t-20260803054914-gpt62`（1×8 卡 `ml.pni2.28xlarge`），三 mode 冒烟 + 全部改动零 lint。
  - **队列修正（重要，后续默认遵守）**：应使用**共享队列 `spot-share-queue`** 且 `Preemptible: false`（"只用队列里的资源、不用闲时/抢占资源"）。已取消 `c20250601` 上的 `t-20260803054914-gpt62`，把 `train_jobs/protein_esmc_llada8b_add_smoke8card.yml` 改为 `ResourceQueueName: spot-share-queue` + `Preemptible: false` 重新提交 → 新 task id **`t-20260803055241-5hrxn`**（queue `q-20260511205647-m6t62`），当前 **Queue** 排队。
- **2026-08-03（正式 diffusion 训练提交）**：8 卡 smoke（`t-20260803055241-5hrxn`，50 步 + 5000 行子集 + 无日志上报）跑通、确认「加载 8B + resize + FSDP 分片 + 数据/collator/remap + masked-diffusion loss」端到端无误后，提交**首个正式 diffusion 训练**。
  - **范式**：融合模型本身即扩散加噪（`sample_bioseq_diffusion_noise`，MDLM），故"diffusion 版本"就是默认路径；`residue_cond_mode=add`（ESMC 残基特征叠加在 token embedding 之上），冻结 ESMC，`condition_norm=True`。
  - **入口新增**：`--decoder_grad_ckpt`（默认 True）在入口直接调 `decoder.gradient_checkpointing_enable()` + `config.use_cache=False`（**不走** HF `args.gradient_checkpointing`，否则会在缺该方法的组合模型 `LLaDAEsmcFusion` 上调用而报错）。单卡实测 batch=2 峰值仅 32.8GB（FSDP 分片后余量更大），梯度回传正常。
  - **超参**：full OAS/OTS（去掉 `max_rows_per_source`）；`per_device_bs=4 × grad_accum=4 × 8gpu = global 128`；`max_length=512`、`max_protein_length=256`；`lr=1e-4` cosine + warmup 2000、`max_grad_norm=1.0`；`max_steps=50000`，`save_steps=1000`（保留 10），随时可停/续；`logging_steps=20`。
  - **eval 关闭**：`BioSeqDiffusionOutput` 是普通 `@dataclass`（非 dict-like），HF 默认 `prediction_step` 取 `outputs["loss"]` 会崩；正式跑先 `eval_strategy=no`，训练 loss 走 wandb。（后续如需 val loss 再覆写 `prediction_step`。）
  - **wandb**：`WANDB_MODE=offline`（不依赖集群出网），本地写 `output/.../wandb`，可事后 `wandb sync`；project `bioseq-llada8b-fusion`。
  - **提交**：新增 `train_jobs/protein_esmc_llada8b_add_diffusion.yml`（`spot-share-queue`，`Preemptible: false`，1×8 卡 `ml.pni2.28xlarge`，`ActiveDeadlineSeconds=604800`）→ task id **`t-20260803060836-j7z9x`**（queue `q-20260511205647-m6t62`），当前 **Queue** 排队。
  - **改为 save-best-5（top-k by eval_loss，对齐 bioseq 语义）**：项目原有 `ValLossTopKCheckpointManager`（保留 val loss 最低的 K 个）绑在自研 `BioSeqTrainer`；本融合跑用 HF `Trainer`，其 `save_total_limit` 只保留「最近 K 个」（仅保护单个 best），语义不符。故在 HF 路径复刻 top-k：
    - **入口 `protein_pretrain_esmc.py` 新增**：① `FusionTrainer.prediction_step`（loss-only，绕开 `BioSeqDiffusionOutput` 非 dict-like 导致的默认 eval 崩溃）；② `TopKValLossCheckpointCallback`（`on_save` 时从 `log_history` 取当步 `eval_loss`，维护 top-k，删除超出的 checkpoint 目录，并写 `topk_val_manifest.json`）；③ 参数 `save_top_k`(默认5)、`DataArguments.max_eval_rows_per_source`(默认2000，限制每次 eval 成本)。
    - **YAML 调整**：去掉 `--save_total_limit 10`（改由 callback 管理保留），加 `--eval_strategy steps --eval_steps 1000`（与 `save_steps` 对齐，保证每个 checkpoint 都有当步 eval 指标）、`--per_device_eval_batch_size 8 --max_eval_rows_per_source 2000 --save_top_k 5`。
    - **验证**：tiny LLaDA + 真实 ESMC，`eval_steps=2/save_steps=2/save_top_k=2` 跑 8 步——eval 正常出 `eval_loss`，callback 剪掉 step2/step4，仅保留 eval_loss 最低的 step6/step8，manifest 升序，PASS。零 lint。
    - **重新提交**：取消 `t-20260803060836-j7z9x`，用更新后的 YAML 重交 → 新 task id **`t-20260803061533-q8r98`**（queue `q-20260511205647-m6t62`）。**后续已 cancel**（见上方「真正在跑的是 bert ablation」）。
- **2026-08-03（BERT 版本提交）**：融合路径此前只有扩散加噪（`sample_bioseq_diffusion_noise`：每序列 `t~U(eps,1)`、选中位 100% 换 `<mask>`）。新增 BERT-MLM 目标，二者共用同一 eligible 集合与 `compute_masked_cross_entropy`（token 级均匀 CE），仅腐蚀策略不同：
    - **`protein_fusion_model.py` 新增 `sample_bioseq_bert_noise`**：固定比例 `mask_ratio=0.15` 选中，选中位按 80/10/10 → `<mask>` / 随机残基 token / 保留原样；labels 落在**全部**选中位。随机替换从 LLaDA 空间的 `<res_*>` id 池抽样（新增非持久 buffer `_residue_token_ids`）。
    - **encoder 镜像**：BERT 的 `corruption_mask` 传全部选中位（含 10% 随机/10% 保留），即在 ESMC 侧对所有选中残基打 `<mask>`，防止冻结 encoder 通过条件路径泄漏待预测残基。
    - **模型 + 入口开关**：`LLaDAEsmcFusion` 增 `train_objective∈{diffusion,bert}` 与 `bert_mask_ratio/mask_prob/random_prob/residue_token_ids`；`compute_loss` 按目标分支。入口 `TrainingArguments` 增 `--train_objective/--bert_*`，并从 tokenizer 算出 `<res_*>` id 传入。
    - **验证**（tiny LLaDA + 真实 ESMC，400 次采样统计）：选中/eligible≈0.152（目标 0.15）、选中内 mask/random/keep≈0.799/0.098/0.103（目标 .80/.10/.10）、随机替换全在残基词表内、labels 恰好落在选中位；三 mode（token/feature/add）bert fwd+bwd 全通过。零 lint。
    - **提交（已 superseded）**：新增 `train_jobs/protein_esmc_llada8b_add_bert.yml` → task id **`t-20260803062133-blhfm`**。与 `add_diffusion` 同属早期 bert-vs-diffusion（冻结 ESMC）方案；**已 cancel**，正式对照改为 `bert_esmctrain` / `bert_noesmc`。
