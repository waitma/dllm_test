# LLaDA 蛋白预训练（OAS/OTS）项目进度

> 蛋白训练线的决策与结论笔记本。Volc 任务 / 实验进展的**权威账本**是
> [`PROJECT_PROCESS.md`](../../PROJECT_PROCESS.md)；下游数字权威在
> [`RESULTS.md`](../../downstream/benchmark/RESULTS.md)。本文不复述 prepared 布局。

- 现役入口：[`protein_pretrain_esmc.py`](protein_pretrain_esmc.py)（ESMC 条件融合，正式跑）
- 操作手册（怎么跑、怎么提交、resume 规则）：[`README.md`](README.md) §「蛋白预训练（本项目）」
- 数据操作入口：[`dllm/pipelines/immune_llada/README.md`](../../dllm/pipelines/immune_llada/README.md)
- raw 去污事故：[`DATA_PIPELINE_README.md`](DATA_PIPELINE_README.md)
- 下游数字权威表：[`RESULTS.md`](../../downstream/benchmark/RESULTS.md)
- 多链关系对照臂设计：[`MULTI_CHAIN_RELATION.md`](MULTI_CHAIN_RELATION.md)

**最近更新 2026-09-12**：v5 开跑前核算——`--save_top_k 3` / `--eval_steps 1000` /
`ActiveDeadlineSeconds 950400` 三者不动。步时/eval/配额数字见 §6.3 / §6.4 / §6.7 与
[`SPEED_ANALYSIS.md`](../../SPEED_ANALYSIS.md)。离线 binding 评估脚本见 §5.6。
v5 8-GPU diffusion 仍 Queue。账本：[`PROJECT_PROCESS.md`](../../PROJECT_PROCESS.md) 同日条。

---

## 0. 协作规则（Agent Rules）

1. **实时更新进展**：有实质进展（改代码、跑通验证、发现问题、做决策）就同步本文档 —— 更新 §3 当前状态，
   并在 §11 追加一行。不要攒到最后补。
2. **关键改动按 owner 同步**：任务/进展 → [`PROJECT_PROCESS.md`](../../PROJECT_PROCESS.md)；
   训练怎么提交 → [`README.md`](README.md)；prepared 数据怎么跑 →
   [`dllm/pipelines/immune_llada/README.md`](../../dllm/pipelines/immune_llada/README.md)；
   raw 去污事故 → [`DATA_PIPELINE_README.md`](DATA_PIPELINE_README.md)。
   本文只记训练线决策与结论，不要把 README 灌成副本。
3. **子任务可直接用 grok 4.5**（`cursor-grok-4.5-high-fast`）：主控负责统筹、拆解与 review，
   不必为每个子任务重复征求许可。

---

## 1. 目标与关键决策

在**不改动 LLaDA 训练主干**的前提下，让 LLaDA 在 **OAS(抗体) / OTS(TCR)** 免疫序列上预训练，
对照两种目标函数：`diffusion`（每样本随机 mask 比例）与 `bert`（固定比例 MLM）。
只改数据加载、grammar 语法、tokenization。

| 议题 | 决策 |
|------|------|
| 训练范式落地 | 单入口 + `--train_objective {diffusion,bert}`，共用一套 eligible 集合与 masked CE，只有腐蚀策略不同，不新写 trainer |
| grammar | 复用现成 grammar v2 `GrammarRenderer`，数据源限定 ab/tcr |
| 词表/权重 | 保留 LLaDA 完整 BPE 词表；新增残基/结构 token 后 resize，并把 grammar 空间 id 重映射到 LLaDA 空间，完整继承 embedding/head/主干 |
| bert 加噪 | 经典 MLM：选 15% 位置，再 80% `[MASK]` / 10% 随机词 / 10% 原词；loss 算在全部选中位 |
| diffusion 加噪 | 每序列一个 `t ~ U(eps,1)` + Bernoulli(t) 吸收态 mask |
| ESMC 条件 | `residue_cond_mode=add`（残基特征叠加在 token embedding 上），加噪位在 encoder 侧**镜像打 `<mask>`**，防条件通路泄漏答案 |
| 量化 | 禁用 `load_in_4bit`（破坏 resize）；LoRA 必须「先 resize 再 peft」，`modules_to_save` 含 `wte,ff_out` |
| 表征读出口径 | headline 保持 **raw**（不去中心化 / 不 z-score / 不白化）。理由见 §8 |

---

## 2. 代码与入口

**现役文件**

| 文件 | 说明 |
|------|------|
| [`protein_pretrain_esmc.py`](protein_pretrain_esmc.py) | **正式入口**：ESMC 条件融合 + 七源数据 + diffusion/bert 双目标 |
| [`protein_fusion_model.py`](protein_fusion_model.py) | `LLaDAEsmcFusion` 模型、两种加噪、`RemapCollator`、词表扩展与重映射 |
| [`load_fusion_checkpoint.py`](load_fusion_checkpoint.py) | 从 FSDP checkpoint 还原融合模型（下游评测入口） |
| [`protein_pretrain.py`](protein_pretrain.py) | 早期无 ESMC 入口，仅作对照，**不再用于正式跑** |
| `dllm/pipelines/qwen3_vl_arch/relation_aux.py` | cognate / chain-drop InfoNCE 与 PLL 工具（§4.4） |
| `scripts/data/assert_residue_alphabet.py` | 语料残基字母表 gate（§6.8） |
| `scripts/monitor_spot_tasks*` | 闲时任务看护循环（§6.6） |
| `scripts/downstream/score_pairing_pll.py` | OAS holdout `p(L\|H) − p(L)` |
| `scripts/downstream/score_binding_relation.py` | 离线 binding 关系 token 评估（§5.6）；不碰训练路径 |

**改过的既有文件**：`dllm/pipelines/bioseq/datasets.py`（八个 `*_row_to_record`、`with_exclusion_filter{,_multi}`、
`load_exclusion_keys` fail-fast）、`dllm/pipelines/qwen3_vl_arch/{modeling,sampling}_bioseq.py`（分链 t 改 per-row
multinomial；`relation_aux_loss`；修 ESMC 条件流在迭代解码中泄漏参考序列）、`scripts/data/{tcr_native,dedup}/*`。

**入口主要组件**

- `build_immune_specs`：按 `--dataset_args` 的 `+` token 组装数据源，逐源叠加去污过滤与长度过滤。
  支持 8 个 token：`oas / ots / asd_antibody / asd_nanobody / trait / tcr_native / tcr_papers / tcr_repertoire`
  （`asd_nanobody` 有 builder 但**刻意不入 recipe**）。
- `with_length_filter`：读 CSV 时就丢弃超长行 —— renderer 与 collator 遇超长是**抛异常**而非跳过，会直接崩训练步。
- `FusionTrainer`：`compute_loss` 走 `model(**inputs)`（**必须**，否则 FSDP 下梯度不同步）、
  `prediction_step` 只取 loss（绕开 `BioSeqDiffusionOutput` 非 dict-like）、`evaluate` 按行数加权重组 `eval_loss`。
- `TopKValLossCheckpointCallback` + `slim_checkpoints`：保留 eval_loss 最低的 K 个 ckpt，非最新的瘦身成 weights-only。
  ⚠️ 已知缺陷见 §6.4。

---

## 3. 当前状态（2026-09-12）

### v4 / v5 数据线（2026-09-12）

布局与补全细节不要写在这里。入口：
[`dllm/pipelines/immune_llada/README.md`](../../dllm/pipelines/immune_llada/README.md)；
设计：[`docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md`](../../docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md)。

- ✅ v4 `data/prepared/immune_v4_beta_relation` 已发布（plan §4.1）。
- ✅ §2.5 / §2.6 已在全量语料落地并核验；v5 `immune_v5_receptor_completion` 已发布（plan §4.2）。
- ✅ v4 `tcr_repertoire` 可从自身 manifest 重生（3,000 行 0 mismatch）；v5 同 shard byte-identical。
- ✅ v5 8-GPU diffusion 已提交（Queue）。task id / YAML / 取消 / 可复现 commit：
  [`PROJECT_PROCESS.md`](../../PROJECT_PROCESS.md) 2026-09-12 条。v4 4 卡臂仍未提交。
  YAML 头部写「见本文 §4.2」是从 v4 YAML 抄来的；§4.2 仍是 v3 50k，不是本 run。
  **2026-09-12 开跑前核算：三个 flag 不动**（`--save_top_k 3` / `--eval_steps 1000` /
  `ActiveDeadlineSeconds 950400`），未 cancel、未重提交。步时 / eval / 配额 / deadline
  见 §6.3 / §6.4 / §6.7；底稿 `_jobmon/DECISION_NUMBERS.md`（仓库外）。
- `immune_receptor_v2` 仍未通过 runtime/training gate。

## 3.1 历史 v3 状态（2026-09-07 平台 + 盘上核实）

**平台在跑**（数字取自各 run 的 `topk_val_manifest.json`，非日志转抄）

| 任务 | Task ID | 盘上最新满包 | 最好点 | 进度 |
|---|---|---|---|---|
| `..._diffusion_immune_v3_8gpu_2m`（generated-only） | `t-20260904014842-qgjbg` | **101000 / 0.7456** | 同为最新 | 5.1% / 2M |
| `..._bert_immune_v3_1m` | `t-20260902110050-zb7dr` | 153000 / 0.4257 | **152000 / 0.4107** | 15.3% / 1M |

**已终态、待决策**

| 任务 | Task ID | 终态 | 盘上最新满包 = 最好点 |
|---|---|---|---|
| `..._diffusion_allchains_immune_v3_8gpu_2m`（全链） | `t-20260902021013-bm79q` | **Killed** 09-06T19:37Z（`signal: 15`） | **161000 / 0.6277** |

- 161000 满包完整、`resume_checkpoint` 已指向它，**要续可直接接上**（同 world size，三件套齐全）。
- 2M 目标按 2.95 s/it 算约 **68 天**独占 8 卡，只跑到 8.1%。是否续跑属科研决策，未定。
- ⚠️ 两条在跑的 run 都已越过 §5.1 评过的那个点（genonly 44000 → 101000、BERT 105000 → 152000），
  且 `eval_loss` 仍在缓慢下降。**§5.1 的下游数字对应的是当时的最好点，不是这两条的当前水平。**

**其他**

- 下游评测：**无在排 / 在跑任务**。三条最好点已全部收口（§5.1）。
- 看护循环在跑（PID 3777252，已连续 5.8 天），`TARGETS` 只含 `..._diffusion_immune_v3_spot_2m`
  且已放 STOP 哨兵，**不会重提任何评测任务**。
- `output/` 09-01 回收后 767 G，中途涨回约 **820 G**；2026-09-12 快照约 **466 GB**。
  配额缺陷未修，见 §6.4。
- 多链关系对照臂：代码与 YAML 就绪，**作业未提交**（§4.4）。

---

## 4. 训练任务台账

### 4.1 三条长跑（当前主线）

50k 短跑已收口出数后，方向转为**步数拉长一个数量级**。

| Run | 目标 / loss 覆盖 | 步数目标 | 起点 | LR 调度 |
|---|---|---|---|---|
| `..._diffusion_allchains_immune_v3_8gpu_2m` | diffusion，**全链**（含抗原/MHC/peptide） | 2M | 4 卡 `checkpoint-33000` 的 **weights-only** | polynomial power=1，峰值 1e-4 → `lr_end=1e-5` |
| `..._diffusion_immune_v3_8gpu_2m` | diffusion，仅生成链 | 2M | 4 卡 generated-only `checkpoint-42000` 的 **weights-only** | 同上 |
| `..._bert_immune_v3_1m` | bert MLM 0.15，`--bert_all_chains True`（全残基） | 1M | 50k 的 `checkpoint-50000` 满包 resume | polynomial power=1，峰值 **4e-5** → 1e-5，warmup=0 |

共同配置：`--decoder_init scratch` d=768/L=8/h=12 ≈ 270M、可训 ESMC-300M、`residue_cond_mode=add`、
`max_length=max_protein_length=1024`、**global batch 256**（per_device 2 × ga 16 × 8 卡）、
eval/save 每 1000、`save_top_k=3`、FSDP 单节点 8 卡 `ml.pni2.28xlarge`。

> 🔴 **这批与 50k 那批不可混排**：目标函数、batch 口径、LR 调度三样都变了。
> 全链 2M 的 optimizer 跨 4→8 卡已重置、polynomial 从 step 0 重开，所以它早期 `eval_loss`
> **高于**源点 0.6866 是预期，不是退化判据。

**为什么全链 2M 首跑只能加载权重、不能满包续**：源 checkpoint 是 4 卡的 FSDP pack，
`optimizer.bin` / `pytorch_model_fsdp.bin` **不能跨 world size 续**，故用 `--init_fusion_weights`
只吃 `model.safetensors`。entrypoint 因此有两个挑选函数，语义不同、**不要合并**：

- `pick_latest_full`：要求 `optimizer.bin` + `pytorch_model_fsdp.bin` + `scheduler.pt` **三件齐全**才认，
  用于同 world size 的普通 `--resume_from_checkpoint`。
- `pick_latest_weights`：只要有 `model.safetensors` 就认，用于跨 world size 的首次 init。

generated-only `..._8gpu_2m` **同样不是 from scratch**：YAML 用 `--init_fusion_weights` 吃 4 卡 `checkpoint-42000` 的 `model.safetensors`，optimizer / polynomial 从 step 0 新建。`warmup_steps: 0` 只证明对已训 fusion 热启动安全，不是从零训可以不要 warmup 的证据；真从零的 v5 用 2000（PROJECT_PROCESS 2026-09-12 条）。该 run 的 train loss ~7.2（4 卡源点约 3.0）是新 optimizer / reduction 口径，不是发散；eval 一直在 0.75 一带（最好点 §3.1），不要和 4 卡 train loss 比。

### 4.2 v3 50k 双臂（已终态）

存在理由：v1/v2 的 **BERT 臂跑六源、diffusion 臂跑七源**，两个目标函数根本不可比。
v3 让两臂数据参数**逐字节一致**（提交前 `diff` 验证），唯一差异是目标函数。

| Run | 卡 / global | 终态 | 最好点 | 下游 |
|---|---|---|---|---|
| `..._diffusion_allchains_immune_v3_4gpu` | 4 / **256** | 50k 未跑满，Failed | **33000 / 0.6866** | ✅ 全套，组 D |
| `..._bert_immune_v3_4gpu` | 4 / **256** | 终态 | — | 未评 |
| `..._diffusion_immune_v3_4gpu` | 4 / 128 | 🔴 卡死 42000（§6.8 单字符 bug） | 42000 / 0.7548 | 表征 + CDR，组 C |
| `..._diffusion_immune_v3`（8 卡） | 8 / 128 | 终态 | 27000 | ✅ 全套，组 C |
| `..._bert_immune_v3`（8 卡） | 8 / 128 | 跑满 50k | `checkpoint-final` | 表征（BERT 只评表征，§5.4） |
| `..._{diffusion,bert}_immune_v3_spot` | 8 / 128 | Killed | — | 未评 |

数据：`oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire`，`--tcr_papers_dir` 指
`data/tcr_papers_v2/dataset`。**两臂唯一的非数据差异**：BERT 用 `--bert_all_chains True`
（loss 覆盖全部残基，含固定上下文），diffusion 只算生成链 —— 这是有意的目标函数差异，写论文要说明。

> 🔴 **batch 口径分两组，跨组不可比**：global **128** 组（五条）50000 步 ≈ 0.84 epoch；
> global **256** 组（两条 4 卡）≈ 1.64 epoch。加倍走的是 `ga` 不是 `per_device`，原因见 §6.2。

**全链变体 `_allchains` 的动机与验证**：v3 两臂在「loss 覆盖哪些残基」上本来不对称。
本条把 diffusion 也放开到全链（**没有任何链是 fixed 的**），于是「全链 vs 仅生成链」成为可单独测量的因子，
不再与目标函数绑在一起。实现只覆写 `diffusion_eligible_mask` / `diffusion_loss_mask` 为
`residue_mask & attention_mask` —— 时间步与加噪采样器都从这两个 mask 推导，一次覆写同时改到加噪、
labels 与 ESMC 镜像三条路径，**目标函数本身没动**。

本机 A100 实测（ASD batch，含真实抗原上下文）：

| | 计入 loss | 其中固定上下文 | 被加噪的固定上下文 | 编码器被 mask |
|---|---:|---:|---:|---:|
| `diffusion_all_chains=False` | 659 | **0** | 0 | 647 |
| `diffusion_all_chains=True` | 1452 | **805** | 805 | **1452** |

**编码器被 mask 数 = 计入 loss 数（1452）** 是最关键的一条 —— 证明 ESMC 条件通路没给出任何一个
待预测残基的干净拷贝，否则全链 loss 会退化成抄答案。

### 4.3 早期跑（lineage，仅供追溯）

| 时间 | Run / 批次 | 一句话 |
|---|---|---|
| 2026-08-02/03 | 融合通路打通 | 8B + 真实 ESMC-300M 端到端 fwd/bwd 通过；修 resize 陷阱（LLaDA embedding 已补齐到 126464，+42 token 落在未用 padding 行内，不该 resize）、bf16/fp32 dtype bug、FSDP 梯度不同步 bug |
| 2026-08-03 | `add_diffusion` / `add_bert` | 早期 8B 冻结-ESMC 双臂，**已 cancel**，被 ablation 四条取代 |
| 2026-08-03/04 | ablation 四条：`8b_bert_esmctrain` / `8b_bert_noesmc` / `8b_bert_noesmc_scratch` / `270m_bert_esmctrain_scratch` | 对照「用/不用 ESMC 条件」与「pretrained/scratch decoder」。已终态，仅作 lineage |
| 2026-08 中 | **immune 四条 50k**（270m/8B × bert/diffusion） | 跑满 50k、`checkpoint-final` 齐全，有下游数字（RESULTS 组 A/B）。⚠️ 语料是**六源 v2**且在 08-28 去污修复**之前**，**与 v3 不可比** |
| — | grammar_v2 五条 | 合计 339 G，内部无 optimizer。`..._7l` 被 `PROJ_GUIDE.md` 钉为 TCR-β Track-A canonical，其余四条留否待决策 |

> ⚠️ immune 四条的 top-k 剪枝删过一些 ckpt：RESULTS 里引用 `8b_bert` **40000** 的行**不可复现**
> （现存 43000/44000/46000/50000）。当前对外报的 270m 数字来自 bert **49000** / diffusion **42000**。

### 4.4 多链关系对照臂（代码就绪，作业未提交）

用户确认走实验路径。设计全文见 [`MULTI_CHAIN_RELATION.md`](MULTI_CHAIN_RELATION.md)。**2M 主跑不动。**

诊断：现役只开 `joint_loss_ratio=1.0`，学的是 `p(H,L)` 联合重建而非 `p(L|H)`；`_chain_ratios` 代码早已存在，
但 `int(B * ratio)` 在 per_device 2/4 上把非 joint 桶 floor 成 0，即便打开 YAML 也是空操作；
pairing ImmunoMatch ≈ 0.353 贴错配地板 0.333 —— token CE 不约束嵌入几何。

已落地：① `sample_chain_conditioned_timesteps` 改 per-row `torch.multinomial`（`joint=1.0` 行为不变）；
② `relation_aux.py` 的 `cognate` / `chain_drop` / `both` in-batch InfoNCE，配对 mask 在 `diffusion_all_chains`
扩 eligibility **之前**快照（抗原不当 heavy），aux **只在 train** 计入，`eval_loss` 仍是纯重建 CE；
③ CLI 透传 + `RelationAuxLogCallback`；④ 两条对照 YAML（generated-only、4 卡、global 128、50k、独立 OUTPUT_DIR）：
`..._diffusion_chainratio_immune_v3_4gpu.yml` 与 `..._chainratio_cognate_...yml`；
⑤ 单测 `scripts/diagnostics/test_relation_aux.py`（CPU，不加载 ESMC）。

提交纪律：新 OUTPUT_DIR、from scratch，不要 `--init_fusion_weights` 指向 2M，不要改 2M YAML。
**headline = ImmunoMatch + PLL 差，不是 `eval_loss`。** 提交前确认约 9 GB 配额余量。
本轮不做：intra/inter 分套注意力、结构、扩 `<tcra>`/`<tcrb>`、PLL 进 pretrain loss。

---

## 5. 下游评测台账

覆盖口径：**表征** = T1 binding / T2 clustering / T3 representation；**生成** = AB CDR infilling、
AB light pairing、T4 TCR generation。数字权威在 [`RESULTS.md`](../../downstream/benchmark/RESULTS.md)，
本节只记口径、状态与结论。

### 5.1 三条长跑最好点 —— ✅ 已收口（2026-09-06/07）

三条训练各自当前最好 val 点，权重硬链接到各 run 的 `eval_snapshot_*/`（避免 top-k 剪掉评测输入），
训练全程未打断。

| Run | ckpt | eval_loss | tag | 覆盖 |
|---|---|---:|---|---|
| 全链 2M | `checkpoint-151000` | 0.6326 | `ours_fusion_v3_allchains_8gpu2m_151000` | 表征 + CDR + pairing + T4 |
| generated-only 2M | `checkpoint-44000` | 0.7520 | `ours_fusion_v3_genonly_8gpu2m_44000` | 表征 + CDR + pairing + T4 |
| BERT 1M | `checkpoint-105000` | 0.4263 | `ours_fusion_v3_bert_1m_105000` | **表征 only** |

**headline 数字**

| | allch@151k | genonly@44k | bert1m@105k | 参照 |
|---|---:|---:|---:|---|
| T2 ARI mean | **0.0239** | 0.0172 | 0.0151 | 旧 270m diff 0.028；SCEPTR 0.033 |
| T3 deep k=200 | 0.707 | 0.706 | **0.712** | TCRdist 0.777；SCEPTR 0.790 |
| Pairing ImmunoMatch | **0.438** | 0.399 | —（不评生成） | ref 0.699；组 C 地板 0.353；全链@33000 0.436 |
| T4 common d_edit ↓ | 8.80 | 8.61 | — | 旧 270m 6.99；**OLGA（不看表位）6.34** |

**四条结论**

1. **长跑到 151k 没有改善 T4，反而微降。** common-6 d_edit：26k 8.58 → 121k 8.78 → 151k **8.80**；
   seq-rec 0.212 → 0.210 → 0.218。仍远差于旧 270m 和不看表位的 OLGA。
   §0.4 的主结论「unseen 表位上条件信号接近不存在」**长跑没能撼动**。
2. **CDR 小幅上行，但在噪声内。** SAbDab 旧快照 H3 26k 41.68 → 151k **41.92**（+0.24），H2 +1.34；
   SAb23H2 六 CDR 均值 67.70 → 68.29。同臂 18k→26k 的 H2 波动就有 +0.41，**不足以当"长跑有效"的证据**。
   这是 2M 计划的 7.6% 进度。
3. **配对是全链唯一稳定抬起来的任务。** 全链 0.438 vs generated-only 0.399，且全链在 33000（0.436）→
   151000（0.438）上稳定；两者都明显离开了组 C 的 0.353 地板 —— 说明**长跑本身也抬 pairing**
   （generated-only 从 0.353 到 0.399），但**全链额外再抬一档**。这是目前 `--diffusion_all_chains` 唯一可见的收益。
4. **表征三项上三臂基本同档**（T3 差 0.006、T2 差 0.009）。BERT 的 T3 略高但它是**单点无误差棒**，
   按 §8 的引用纪律**不能**据此说"哪个目标更适合表征"。

> ⚠️ **CDR 走的是旧快照，不是 Kong 3,127。** wrapper 默认 `data/downstream/cdr_infilling/sabdab`，
> 该缺陷 2026-09-02 已登记。故 CDR 数字只进 RESULTS §0.5 的「旧快照」表，**Kong 主表不填**。
>
> ⚠️ 三条的 `eval_loss` **不能互相比**（目标函数 / loss 覆盖 / 调度都不同，§6.1）；
> generated-only 2M 与组 C（50k / global 128）也不可混排。

**这一轮怎么跑完的**：九条平台作业先在 `c20250601` 闲时排 4.9 小时零起跑，改投 `queue012` 非闲时后
又排 41 分钟仍 9/9 Queue —— 查队列构成发现 `queue012` 的拥堵**全在单卡**（非终态 758 条里单卡 607 排队，
多卡排队总共只有 27 条），我们九条正好都在最挤的档。故只保留两条 pairing（关键路径，比其余阶段慢一个数量级）
让位，其余**改本机单卡跑**。两条 pairing 最终于 09-07T07:10Z / 07:12Z **Success**。
本地跑通路见 §5.5。

### 5.2 全链 2M 的中间点序列（组 D）

同一条 run 的连续快照，用来看「长跑是否在改善下游」。全部 tag 前缀 `ours_fusion_v3_allchains_8gpu2m_`。

| ckpt | eval_loss | 覆盖与状态 | RESULTS 回填 |
|---|---:|---|---|
| 4 卡源点 33000 | 0.6866 | 全套 4/4 Success | ✅ 已回填 |
| 18000 | 0.7044 | 全套 4/4 Success | ✅ 已回填 |
| 26000 | 0.6902 | 全套 4/4 Success | ✅ 已回填 |
| 69000 | 0.6577 | 表征/CDR/pairing/T4 全 Success | ⬜ 待回填 |
| 73000 | 0.6535 | 表征/CDR/T4 Success；**pairing Killed** | ⬜ 待回填 |
| 121000 | 0.6416 | 表征/CDR/T4 Success；**pairing Killed**。另有 `max_iter` sweep Success | ⬜ 部分待回填 |
| 137000 | 0.6332 | 表征 Success | ⬜ 待回填 |
| 144000 | 0.6386 | 表征 Success | ⬜ 待回填 |
| 151000 | 0.6326 | 全套收口，见 §5.1 | ⬜ 待回填 |

121000 表征数字（已读出）：T1 seen AUPRC **0.766** / unseen AUROC **0.527**；T2 ARI **0.025**；
T3 deep k=200 **0.711** / broad k=100 **0.645**；probe-AUROC **0.807**。

已记录的方向：**18k → 151k 没有系统性变好**；表征与 generated-only 同档；pairing 与 T4 见 §5.1。
18000 / 26000 / 69000 / 121000 的原 ckpt 已被 top-k 剪掉，但下游产物齐全，未重跑。

### 5.3 组 C 与早期（generated-only / 六源）

| Run / ckpt | tag | 覆盖与状态 |
|---|---|---|
| 8 卡 v3 `ckpt-27000` | `ours_fusion_v3_diffusion_*` | 表征 + CDR + T4 + pairing 全齐 |
| 4 卡 v3 `ckpt-42000` | 同组 | 表征 + CDR 齐；pairing / T4 首次 CUDA Failed |
| BERT v3 `checkpoint-final`（50k） | `ours_fusion_v3_bert_final` | **表征 only**（决策见 §5.4）。CDR/T4 虽跑过，数字不进 RESULTS |
| BERT 1M `ckpt-34000` | `ours_fusion_v3_bert_1m_34000` | 表征作业被闲时抢走，**无产物** |
| immune 四条 50k（旧六源） | RESULTS 组 A/B | 有数字，**与 v3 不可比**（§4.3） |

组 C 方向：T1/T3 持平旧 270m；T2/T4/pairing 比旧 diffusion 更弱；pairing IM 0.353 贴错配地板 0.333。

### 5.4 口径决策

- **BERT 只评表征，不评生成**（2026-09-01 用户确认）。已 cancel 在跑的 BERT pairing；
  此后 BERT 只提 `*_repr` 作业。
- **表征读出保持 raw**，不做任何重标定。§8 的白化/去中心化数字是**诊断探针，不是我们主张的性能**。
- **T4 Setting-A 参考集**：修正版已生成（`downstream/benchmark/data/tcr_generation_fullref/`，
  验证 holdout ∩ train = 0），但**未覆盖线上、未重跑打分** —— 换参考会改动已有 novelty/JSD 数值，待决策。
  原缺陷：`prepare_tcr_generation.py --train-cap` 默认 200,000 而 OTS train 有 210 万行，只覆盖 9.5%，
  导致 **19.4% 的 holdout 序列其实躺在训练数据里**（1,893/9,767，逐条核验）。

### 5.5 评测阶段耗时基线与本地跑通路

按 36 条历史 Success 实测的中位耗时。**差异来自解码步数，不是数据量。**

| 阶段 | 中位 | 范围 | 解码步数 |
|---|---:|---|---|
| T4 生成 | 12 min | 11–28 | 写死 32 |
| AB CDR | 16 min | 16–43 | `CDR_MAX_ITER=2` |
| 表征 T1/T2/T3 | 28 min | 27–55 | — |
| **AB light pairing** | **165 min** | 164–180 | `PAIR_MAX_ITER=124` |
| T4 `max_iter` sweep | 72 min | — | 8/32/64/128 |

**单卡评测在两个队列都排不上时，改本机跑，没有速度代价。** 09-06/07 实测本机一张空闲 A100-80G，
四个 T4/CDR 阶段合计 56 分钟，与平台中位逐条一致。做法：

- env **直接激活 eval YAML 里写的绝对路径** `/vepfs-mlp2/.../conda/envs/protenix_abtcr`（不要用 `base` / `pllm`）。
- `PYTHONPATH` / `CUDA_VISIBLE_DEVICES=0` / `HF_HUB_OFFLINE=1` / `PYTORCH_CUDA_ALLOC_CONF`
  与 entrypoint 逐行照抄，`CDR_MAX_ITER=2`、batch `4 8` 不变 —— 产物才与平台跑同口径。
- Runner `output/_local_runs/run_local_t4_cdr.sh`，日志 `local_run.log`，逐阶段耗时与退出码 `status.tsv`。
- 一张卡所以**串行**。

⚠️ **本地跑与平台同名作业会写同一个产物前缀** `output/downstream_generation/<tag>`。
本地跑之前必须确认平台上同名作业已终态，否则闲时任务一旦抢到资源就会覆盖本地产物。

### 5.6 离线 binding 关系 token 评估（2026-09-12）

脚本：[`scripts/downstream/score_binding_relation.py`](../../scripts/downstream/score_binding_relation.py)
（commit `0b31896`）。从任意 fusion checkpoint 打分，**不碰训练路径**。

- 只对 `relation_target_mask==1` 打分。**不能用 `relation_token_mask`**：后者包含
  MHC→peptide 呈递 `<binding>`（固定上下文）和 OAS/OTS null 前缀 `<unknown>`。
- teacher-forced：只把目标位置换成 decoder mask token，其余保持干净。金标留在输入里
  会被 MDM 直接抄，指标虚高。
- 示例：

  ```bash
  python scripts/downstream/score_binding_relation.py \
    --checkpoint <ckpt> \
    --prepared-data-dir data/prepared/immune_v5_receptor_completion \
    --max-records 400 --device cuda
  ```

- 管线自检：v3 `checkpoint-42000`（未学过 relation target）400 条得 acc=0 /
  AUROC≈0.50 / NLL≈28，且 400 条 argmax 全落到同一非金标 token（id `126375`）——预期。
- 全量 valid 正负极不均衡：asd **39866/5000**、tcr_papers **4602/2323**、
  tcr_native **3848/7**、trait **540/853**。accuracy 基本无意义；`tcr_native` 仅 7 条
  nonbinding，该源 AUROC 永远是噪声。
- `tcr_papers` 记录里 `record.source` 仍写作 `tcr_native`，脚本靠 **shard 名**分层；
  **底层字段错误仍未修**。

---

## 6. 工程规则（踩坑换来的硬约束）

### 6.1 🔴 `eval_loss` 只能在三层口径全同时比大小

任何跨线比较都是无意义的：

1. **目标函数**：BERT 是固定 15% mask 位的 MLM CE；diffusion 是每序列随机 `t` 下的加权损失。
   BERT 看着低（0.42 vs 0.63）**不代表更好** —— 分母、被评 token 数、任务难度全不同。
2. **loss 覆盖范围**：`_allchains` 把固定上下文（抗原/MHC/peptide）也计入，集合更大更难。
3. **batch 口径 / LR 调度**：global 128 vs 256；cosine vs polynomial 重开。

**组内、同一条 run 内是自洽的**，所以 top-k 选 checkpoint 完全可靠；跨组只看下游 benchmark。

### 6.2 🔴 显存定容必须用真实混合语料，单源探测会严重低估

把 `per_device` 从 4 提到 8 曾导致两条任务**都在 step 2 OOM**（`76.29 GiB is allocated by PyTorch`），
而本机探测预测只有约 43 GiB —— **差了 33 GiB**。

根因：探测脚本只喂 `asd_antibody`，编码器输入 `(8, 2, 870)` 即**每样本 2 条链**；真实七源混合里
TRAIT / TCR 样本带 peptide + MHC + TCRα + TCRβ，**链数可达 4–5**。ESMC 激活按
`per_device × 每样本链数 × 链长` 走，按单源定容必然乐观。

放大误差的两个结构性原因（`scripts/accelerate_configs/fsdp_llada.yaml`）：
`TRANSFORMER_BASED_WRAP` + `fsdp_transformer_layer_cls_to_wrap: LLaDALlamaBlock` 意味着
**只有 decoder block 被分片，ESMC 编码器整个不分片**，每卡持有完整 ESMC-300M（bf16 下还有 fp32 master）；
且 `fsdp_activation_checkpointing: false`，decoder 靠 `--decoder_grad_ckpt` 自己开梯度检查点，
**编码器则完全没有**，激活全额驻留。

> **正确做法**：探测必须用完整七源混合采样，并按 `encoder_input_ids` 的 `[B, C, L]` 里 **C 的最大值**
> 取最坏批次，而不是只挑最长的行。序列长度只是一个维度，链数是另一个，两个都要取最坏。

### 6.3 global batch 加倍走 `ga`，不走 `per_device`

梯度累积不改变任何一次 forward/backward 的张量形状，显存画像与 `per_device 4` 完全相同，
而 `per_device 4` 已被跑满 42k 步实证过。**零风险（指 OOM，不是墙钟）。**

**不能拿一次运行的 s/it 直接外推到另一个 `per_device` / `ga` 配置。** 必须先核对
`training_args.bin` 里的三个因子（`per_device × ga × world_size`）。microbatch 放大是
**次线性**的：`t(batch4)/t(batch2) ≈ 1.29`，不是 2.0。

| 配置 | JobId / 出处 | bin 实配 | tqdm |
|---|---|---|---|
| 历史 v3 8 卡 | `t-20260901033935-h55f4`、`t-20260904014842-qgjbg` | `per_device=2`、`ga=16`、`world_size=8`、`max_length=1024`，全局 **256** | **2.94–2.95 s/it** |
| v3 4 卡（与 v5 同单卡负载） | `..._v3_4gpu` / `checkpoint-42000/training_args.bin` | `per_device=4`、`ga=8`、`world_size=4` | **1.51–1.91 s/it** |
| 本次 v5 8 卡（尚未起跑） | `t-20260912021346-5xq4s` | `per_device=4`、`ga=8`、8 卡，全局同为 **256** | 外推 **1.6–2.5 s/it**（中枢约 2.2）→ 训练 **4.0–5.8 天** |

v5 相对上述 v3 8 卡约为 **0.65×** 步时（`0.5 × 1.29`：累积步数减半 × microbatch 次线性变慢）。
吞吐数字的权威展开：[`SPEED_ANALYSIS.md`](../../SPEED_ANALYSIS.md)。

**训练期 eval 也必须按源分别算，不能用单源速率外推全量。**
`..._v3_8gpu_2m/checkpoint-105000` 的 `eval_oas_runtime=4.4792` 对应每源 2000 行 → OAS 约
**446.5 samples/sec**，本身就是 8 卡，不用再做卡数归一。同一次 run 里 `asd_antibody` 只有约
**114 samples/sec**（约 3.9× 慢），且占 v5 valid 的 44%。按源外推：v5 全量 valid **102,308**
行，单次全量 eval 约 **8.7 分钟**；`--eval_steps 1000` × 200 次合计约 **29 小时 ≈ 1.21 天**。

### 6.4 🔴 磁盘配额：仍是未修的结构性风险

曾因 `Disk quota exceeded (os error 122)` **同一处 4 连 Failed**，每次都死在 step 17000 的
`save_model`，**净进度 0、约 3 小时 8 卡白烧**。

- **爆的是目录/租户配额，不是文件系统，也不是 inode**：`t-20260901033935-h55f4` 在 step 17000
  写 `model.safetensors` 时报 `Disk quota exceeded (os error 122)`，无 inode 字样；
  `df -i` inode 仅 5%。09-01 撞墙当日 `df` 尚余 809T；2026-09-12 `df` 仍有 702T。
  **`df` 不反映租户配额。** 配额上限查不到（`quota` / `lfs` / `mmlsquota` 本机都不存在）。
  **判断磁盘风险要用「上次撞墙水位」**：当时 `output/` 约 **1.2 TB**；2026-09-12
  `output/` 约 **466 GB**，距该水位约 734 GB。
- **一次 save 需要约 9.0 GB 一次性余量**（`model.safetensors` 2.3G + `pytorch_model_fsdp.bin` 2.3G
  + `optimizer.bin` 4.5G），而 **top-k 剪枝发生在写完之后** —— 峰值需求是「现有占用 + 一整个新 ckpt」。
- **不是 `RetryOptions` 把它变成循环**：这 4 连是不同 JobId 的独立提交
  （`t-20260901033935-h55f4` / `t-20260901230256-ns4q4` / `t-20260902000627-j4mqm` /
  `t-20260902010820-t6hq6`）。每个 JobId 用 `ml_task instance list` 都只有 1 个实例；
  相邻提交间隔 5～30 秒，与 `IntervalSeconds: 180` 不符。对照 `t-20260901235433-q76qw`
  同样 `EnableRetry: true` 却停在 Failed、只有 1 个实例。每轮「从上个 ckpt 续 → 跑 1000 步
  → 同一处爆」仍耗 58 分钟——**配额不足就会连续 Failed、净进度 0**，只是机制不能算到平台重试头上。
- **后来存盘成功不代表修好了**：只是别的任务腾出了空间，配额仍贴在临界点。

**新发现（未修）：top-k 账本在重启时重置 → 孤儿 checkpoint 永不被剪。**
`TopKValLossCheckpointCallback` 从**本进程**的 `log_history` 重建 top-k，看不到上一个进程存过什么。
于是每次崩溃-重启都把上一轮 ckpt 甩出账本，剪枝再也不会碰它们 —— 崩溃循环本身在单调抬高占用，
形成正反馈。实测全 `output/` 曾有账本外孤儿 **132.9 G**。

**09-01 已执行的回收：`output/` 1.2 T → 767 G（释放 389 GB）** —— 删作废目录、剪两条已终态 8B run 的
resume-only（各 94 G）、5 处 `checkpoint-final/model.safetensors` **改硬链接**（69 G）、清 6 条 run 的孤儿。

> ⚠️ **`checkpoint-final` 是改硬链接不是删** —— `eval_jobs` 有 8 处引用 `checkpoint-final`、
> `train_jobs` 有 2 处引用 `checkpoint-50000`，**两边路径都必须留着**。
> 刻意保留不能碰：`..._allchains_..._4gpu/checkpoint-33000`（2M 权重来源）、
> `..._bert_immune_v3/checkpoint-50000` 满包（1M resume 来源）。
>
> 🔴 **回收只是把窄门往后推，不是修好了。** 中途曾涨回约 820 G；2026-09-12 快照 `output/`
> 约 **466 GB**。两项待做见 §10。
>
> **slim 语义（2026-09-12 核实）**：`--slim_checkpoints True` 只作用于「被保留下来、
> 但不是 latest」的目录；**latest 满包永不 slim**。实测 slim 后 **2.3 GB**
> （2,423,849,254 B）、latest 满包 **9.1 GB**（9,667,566,831 B）。三种策略总占用：
> `save_top_k=3` 稳态 **13.5–16 GB**（峰值约 25 GB）；`save_top_k=10` 约 **32 GB**；
> `save_top_k=0`（全留 200 个）约 **467 GB** → 会把 `output/` 推到约 933 GB，贴着
> 1.2 TB 死亡水位。v5 因此保持 `save_top_k=3`。

### 6.5 checkpoint 挑选必须验「三件套」

`pick_latest_full` 要求 `optimizer.bin` + `pytorch_model_fsdp.bin` + `scheduler.pt` 齐全才认。
配额事故中它**正确跳过了只有半截 132 MiB `model.safetensors` 的坏 ckpt**、回退到完整的上一个，
没有重演「从坏 ckpt 反复续跑」的死循环。**新增 checkpoint 挑选逻辑时必须保留这个判据** ——
只看目录名或只看 `model.safetensors` 会直接掉进死循环。

### 6.6 闲时（抢占）任务：三件套缺一不可 + STOP 哨兵

1. **独立 `OUTPUT_DIR`。** 两套任务同时排队，谁先拿到资源不确定。共用目录会让两个进程同时写同一批
   `checkpoint-*` 并各自触发 top-k 剪枝，checkpoint 直接报废。
2. **自动 resume。** 闲时资源随时被回收，反 resume 断言会让第二次启动直接失败。entrypoint 取编号最大的
   checkpoint 传 `--resume_from_checkpoint`；**目录里没有 checkpoint 时不能传该参数**（HF Trainer 会报错），
   故条件拼接，且 `set -euo pipefail` 下 `ls` 无匹配必须 `|| true` 兜住。
3. **`RetryOptions` 必须自己开**（平台已下线「抢占后强制重试」）：
   `MaxRetryTimes` / `IntervalSeconds` / `PolicySets=[Failed, InstanceReclaimed]`。
   ⚠️ **照抄现成的抢占 YAML 当模板会静默丢掉这层保护** —— 全仓库 866 个 `Preemptible: true` 的 YAML 里，
   配了 `RetryOptions` 的原本是 0，那些配置都写在平台下线该功能之前。

**外部看护循环**（`scripts/monitor_spot_tasks*`）补的是 `RetryOptions` 的盲区：重试次数用尽或任务落到
终态之后平台就彻底不管了，训练会永久停住而没人知道。每 600 秒查一次同名任务，`Failed`/`Killed` → 用同一 YAML 重提。

> 🔴 **人工 cancel 闲时任务前必须先 `touch output/_monitor/STOP`**（或 `STOP.<TaskName>`），
> 否则会被看护重提回来 —— `Killed` 既可能是被抢占也可能是我们主动 cancel，脚本无从区分。
>
> 另两个不能省的设计：① **只看最新一条** —— `ml_task list --name` 是模糊匹配且会返回历史终态任务，
> 不取最新会被历史记录骗到、无限重提；② 停滞**只告警不自动 kill**（eval 本身会拉长 wandb 写入间隔，
> 误杀代价大于漏报）。

### 6.7 队列策略

默认 `queue012` + `Preemptible: false`。8 卡排不上时**先试第一条**：

1. **降到 4 卡** `ml.pni2.14xlarge` + 非闲时，把省下的乘数补进 `per_device` 以保持 global batch 不变
   （lr schedule、总样本量、每个 optimizer step 的语义全部不变）。实测 **19 秒起跑**，
   而同规格 8 卡任务零 Running、最早的已等 9.8h —— **缺的是连续整节点 8 卡，不是队列配额**，
   也不是优先级问题（`Priority` 已是用户可提交的最高档 6，平台只开放 2/4/6）。
2. **借闲时资源**：另投 `c20250601` + `Preemptible: true`，但必须满足 §6.6 的三件套。

已知的权限与工具坑：

- 本账号（`zhuyiheng`）对 `c20250601` **已无 `CreateCustomTask` 权限**（`q-20260121145036-6fztt`）。
  探针任务法：提一条 `probe-perm-check-donotrun` 立刻 cancel，用来验证某队列能否提交。
- `StopCustomTask` 权限曾对 Creator 为 `251105016` 的任务报未授权，次日同一账号又放行了，原因未知。
  **对存量任务先直接 `ml_task cancel` 试，不必默认走控制台。**
- `ml_task export --config` **不导出 `Preemptible` 字段**，核对抢占只能用 `get --format`。
- **`ActiveDeadlineSeconds` 是 MaxRuntime，从 Launch 起算，不含排队。**
  证据：`t-20260904014842-qgjbg` 排队 22.8 小时，tqdm 累计墙钟 88 小时对齐 Launch→End，
  不是 Create→End。`ml_task` 的 `Start` / `Elapsed` 含排队，不要当 deadline 时钟。
  v5 `950400`（11 天）对照预期总墙钟 5.2–7.0 天，余量充足；超时线约 **4.15 s/it**。
- **`RoleRestartPolicy`（角色级）与 `RetryOptions`（作业级）是两层。** `ml_task get` 不回显
  `RetryOptions` 属正常（`--helpformat` 可选字段列表里没有该字段）；判断是否被接受要用
  `ml_task export --config`。v5 `t-20260912021346-5xq4s` 已实证平台接受了 YAML 里的
  `RetryOptions`，并额外补了默认 `EnableReserveResourceOnRetry: false`。
- CLI **无 update 子命令**，改 YAML 不回写已提交任务，只能 cancel 重提。
- `Description` 上限 500 字符，写长了直接提交失败 —— YAML 里只留一行指针注释，细节写本文档。
- ⚠️ **配置性失败（OOM / 参数错）后必须立刻 cancel 整条重试链**：`PolicySets: [Failed]` 分不清
  「节点故障」与「配置错误」，会把坏配置反复重提，而新任务与好任务**同名、共用同一 `OUTPUT_DIR`**。

### 6.8 语料字母表必须做 pre-flight gate

一个比对空位符 `-` 曾让一条 84% 完成度的任务陷入重试死循环。`RemapCollator` 的映射表只覆盖
`RESIDUES`（`LAGVSERTIDPKQNFYMHWCXBUZO`）加 grammar/chainsep/pad，而 ESMC 词表里
**id 29 `.`、id 30 `-`、id 31 `|`** 落在映射表外。全语料 850 万行只有 1 行命中
（`data/tcr_papers_v2/dataset/train.csv` 第 3046 行 `cdr3b` = `ASSKVAARVP-TLKLS`），
但因 HF Trainer 默认 `ignore_data_skip=False`、resume 时用同一 seed 复现采样顺序，
每次从同一 ckpt 续跑都会在**约 42782 步**再次撞上它。

✅ **已修（2026-08-31）**：就地改成 `ASSKVAARVPTLKLS`（**行数不变**，采样顺序与 resume 语义不受影响；
删行则会移位）。去污预先核过：修正形在 10 个 blocklist 里命中 0 次。同时更新了 `finalize_report.json`
并重跑 `assert_corpus_fresh.py`。09-01 复核：带 `-` 原形 0 次命中，修正形恰好 1 次。

`scripts/data/assert_residue_alphabet.py` 已落仓库，发现坏行 **exit 1**，与 `assert_corpus_fresh.py`
并列放进 entrypoint 做 gate。脚本里的 `RESIDUES` 常量必须与入口保持同步。

> 曾考虑的另一条路（把 `.` `-` `|` 映射到 `<res_X>`）**没有采用**：不改词表、checkpoint 兼容，
> 但会把数据质量问题静默吞掉，且失去这条断言作为守卫的价值。断言本身是对的 —— 它成功抓到了这个 bug。

### 6.9 语料新鲜度：`PASS` 不等于干净

`build_report.json` 的 `PASS` 只代表「对**构建时那份**黑名单干净」，黑名单重建后语料不会自动跟随，
两者无任何关联 —— 曾导致 3 个 T4 参考 binder 留在 `train.csv` 而报告仍显示 `PASS: true`。
修法：报告记 `blocklist_provenance`（mtime + 内容 sha1），`assert_corpus_fresh.py` 比对活文件 hash，
训练配置在 entrypoint 里调用它。

---

## 7. 数据与语料事实（当前口径）

### 7.1 七源 train 实测快照

以下是 `tcr_repertoire` 近重复搬迁**之后**（2026-08-29 05:30 重测）的权威数字。
`max_length=max_protein_length=1024`，全部 blocklist 生效。当前权威快照另见 `DATA_FORMAT_AUDIT.md`。

| source | 目录 | kept | 记录% | 残基% | res/rec |
|---|---|---:|---:|---:|---:|
| `oas` | `data/oas_previous_clean/splits` | 2,485,471 | 31.89% | 45.24% | 232 |
| `ots` | `data/ots_paired_clean/final` | 2,094,231 | 26.87% | 37.28% | 226 |
| `tcr_repertoire` | `data/tcr_repertoire/dataset` | 2,128,750 | 27.31% | **2.19%** | **13** |
| `tcr_papers` | `data/tcr_papers_v2/dataset` | 681,444 | 8.74% | 2.85% | 53 |
| `asd_antibody` | `downstream/asd/step6_final/antibody` | 276,412 | 3.55% | **11.37%** | **523** |
| `tcr_native` | `data/tcr_native/dataset` | 96,552 | 1.24% | 1.07% | 140 |
| `trait` | `downstream/trait/step4_final` | 31,515 | 0.40% | 0.16% | 66 |
| **合计** | | **7,794,375** | | | |

- 总残基 **1,273,914,576**，生成链残基 **1,150,746,721**（90.3%）。
- valid 合计 **203,647**；进 eval 的是六源各 2,000 + `trait` 1,393 = **13,393** 行。
- **配比按记录等权，不是按残基**：source weight 已写入 prepared `BioSeqRecord`，但当前 collator/loss 尚未使用它，混合比例仍由 prepared manifest 中的行数决定。
- **训练预算**：global 256 × 50,000 步 = 1280 万条 ≈ **1.64 epoch**（global 128 那档 ≈ 0.84 epoch）。
- ⚠️ 搬迁让 train 超出当初刻意设的 2,000,000 上限 6.4%（判为可接受）。
  **若重跑 `build_repertoire.py`，该上限会把搬回来的行重新截掉，须重跑搬迁。**
- ⚠️ `DATA_FORMAT_AUDIT.md` 里写的「v2 mix 7,366,444」**仍含 `tcr_repertoire`**，
  不是六源 YAML 真正训的语料（六源 v2 是 5,391,293 条），别混用。

### 7.2 六种布局的真实 loss 预算

`gen%` = 该布局占全部待预测残基的比例（蓄水池采样实测）：

> ⚠️ **下表行数/gen% 是 2026-08-29 口径，v4/v5 后未按布局重算**；布局语义已变（详见 §7.3）：
> 三个无条件布局现在都带固定前缀 `<prots> <null> <protd> <unknown>`（全 `is_fixed`，不计 loss）；
> `tcr_single` 已废弃，`tcr_repertoire` 统一渲染为 β→α 双链（约 92% token 是不算 loss 的 `X`）；
> relation token 在 v4 是可训 target。v4 逐源行数见 plan §4.1；v5 行数与
> `diffusion_loss_mask` 监督份额见 plan §4.2（口径不同；不要把历史 `gen%` 或误记的
> 5.4% / 0.13% 当成 v4 监督份额）。

| 布局 | 条件 → 生成 | 来源 | 记录% | **gen%** |
|---|---|---|---:|---:|
| `antibody_pair` | 固定 null 前缀 → 抗体 H+L | `oas` | 31.9% | **49.64%** |
| `tcr_pair` | 固定 null 前缀 → TCR β+α 全长 | `ots` | 26.9% | **40.92%** |
| `antigen_antibody` | 抗原 → 抗体 H+L | `asd_antibody` | 3.6% | **4.55%** |
| ~~`tcr_single`~~ → `tcr_pair` | 固定 null 前缀 → 真实 CDR3β + `X` 补全双链 | `tcr_repertoire` | 27.3% | **2.90%** |
| `tcr_pmhc` | MHC+表位 → TCR | `trait`+`tcr_native`+`tcr_papers` | 8.6% | **1.78%** |
| `tcr_peptide` | 仅表位 → TCR | 同上三源里无 MHC 的部分 | 1.8% | **0.20%** |

三源到布局的实测拆分（全量扫描）：`trait` 95.2% pmhc / 4.8% peptide；`tcr_native` 99.9% / 0.1%；
`tcr_papers` 80.3% / 19.7%。

🔴 **三个无条件/抗体布局吃掉 95.1% 的 loss 预算，表位条件生成合计只有 1.98%。**
根因是残基数量级而非行数：`antibody_pair` 每条约 232 个待预测残基，而 `tcr_pmhc`/`tcr_peptide`
只生成 CDR3、每条 13–31 个。**加数据改不动这个比例** —— `tcr_papers` 行数翻 3 倍也只把 1.98% 抬到约 3%，
必须 per-sample 加权。这大概是 §5.1 结论 1（T4 长跑不改善）的结构性原因。

> ⚠️ `count_grammar_layouts.py` 2026-08-29 前用的是**前缀采样**，结论是错的：`tcr_papers_v2` 是
> 7 个论文语料首尾拼接，前 3 万行 100% 是 `tcr_peptide`，导致 `tcr_pmhc` 低估约 4 倍。
> 已改蓄水池采样并加 `--seed`，新数字与独立全量扫描一致（671,678 vs 673,686）。

### 7.3 每个源渲染成什么样

```
# 前缀约定（v4）：无条件布局在前面跟一段固定 null context，与条件布局形状对齐
#   <prots> <null> <protd> <unknown>      四个 token 全固定，不算 loss、不加噪、不消耗 chain index

# 布局一 无标签配对生成（oas + ots），残基位 100% 可训
 oas  [<prots> <null> <protd> <unknown>] <prots> <ab>  *[heavy ×128] <chainsep> *[light ×110] <protd>
 ots  [<prots> <null> <protd> <unknown>] <prots> <tcr> *[beta  ×114] <chainsep> *[alpha ×112] <protd>

# 布局二 识别型 = 固定上下文 + relation（v4 起为可训 target）+ 待生成受体
 asd   <prots> [antigen ×75] <protd> *<binding> <prots> <ab> *[heavy ×118] <chainsep> *[light ×106] <protd>
 trait <prots> <pep> [epitope ×10] <protd> *<binding> <prots> <tcr> *[cdr3b ×14] <chainsep> *[cdr3a ×13] <protd>

# 布局三 beta-only（tcr_repertoire）—— v4 起统一渲染成 alpha/beta 双链，不再是 tcr_single
 rep  [<prots> <null> <protd> <unknown>] <prots> <tcr> *[CASSQETQYF ×13 真实 CDR3β + X 补全] <chainsep> [alpha 全 X] <protd>
      → 约 8% token 算 loss，约 92% 是 synthetic X（不算 loss、不加噪，但吃满 attention）
```

> `*[...]` / `*<...>` 是可训练目标位，`[...]` 是固定上下文。generated block 内的骨架 token
> （`<prots>` / `<ab>`·`<tcr>` / `<protd>`）同样加噪并计 loss，图里为可读性未逐个标星号。

> ⚠️ 上面是「各源首行」，对 `trait` / `tcr_papers` **不具代表性**（`trait` 首行恰好没有 MHC，
> 但全量 95.2% 带 MHC 块）。布局的**分布**看 §7.2，别从样例推占比。

其他口径事实：

- `oas` 的 CSV 有 33 列，`row_to_record` **只取** `cleaned_h_sequence` / `cleaned_l_sequence` 两列。
- 链顺序：抗体按位置 heavy→light；**TCR 按角色重排为 β→α**（`grammar.py` 的 `receptor = [beta, alpha]`），
  与 OTS record 里 β 在前一致。这一点不只是美观：`sample_chain_conditioned_timesteps` 和
  `generated_heavy_light_masks` 都把**最小 chain id 当 heavy**，α 在前会让模型把 α 误当 heavy 链。
- `tcr_native` 与 `tcr_papers` 用同一套统一 schema，故共用 `tcr_native_row_to_record`：
  优先全长 Fv，缺失则退回 CDR3 loop。
- 词表侧：全语料只激活新 token 中的一小部分 —— 语法 token 实际出现
  `<prots> <protd> <ab> <tcr> <pep> <chainsep> <binding> <nonbinding> <null> <unknown>` 十个
  （`<null>`/`<unknown>` 自 v4 的无条件前缀起启用），残基除 20 个标准氨基酸外多一个 `X`
  （beta-only 补全位，不算 loss）。
  **继承的 12.6 万 BPE 词表在本任务里基本是死重量。**
- **单链 α 摄入不了，卡在 grammar 而非数据**：`GrammarTokenizer` 只有一个 `<tcr>`，没有 `<tcra>`/`<tcrb>`，
  配对时靠 `[beta, alpha]` 位置编码身份，但 `len(receptor) == 1` 时 α 与 β 渲染成**完全相同的 token 序列**。
  硬塞会污染 T4 Setting-A 测的 β 分布。要摄入须扩词表 → 现有 checkpoint 不兼容，属建模决策，未做。

### 7.4 `asd_antibody` 为什么只剩 32.5%

常被追问，结论：**不是训练集内部去重，是与下游 benchmark 的去污染**，且 98% 来自 heavy 链桶。

| 阶段 | 剩余 | 删掉 |
|---|---:|---:|
| 原始 train.csv | 850,134 | — |
| benchmark 去污染 | 278,554 | **571,580（67.2%）** |
| 长度过滤（1024） | **276,412** | 2,142（0.25%） |

- 判据：heavy 链 0.95 id / 0.80 cov 或 CDR-H3 core 0.80/0.80（light 不参与）。
- 根因是该源主体为**点突变库**：`buzz` 占 61.6% 行，抽样 20 万条全部 120 aa、全部唯一，
  但相对库内共识只差 1–10 个位点。`mmseqs easy-linclust --cluster-mode 1` 是连通分量（传递闭包），
  整库并成极少数巨簇，簇内只要有一条命中 benchmark 就整簇删 → `buzz` 掉 83.0%。
- 删除有实据：4,419 条 benchmark heavy 链**逐字**出现在 ASD 中；`flab_*` / `ab-bind` 被删 100%。
  反面证据：`met` 删 0%、`genbank` 0.1%。
- 诚实标注：连通分量按设计**过删**。要回收行数，杠杆是 `--cluster-mode` 或提高 heavy 阈值，
  **不是**动长度上限。该代价已于 2026-08-28 决策接受，三个 benchmark 全保护，不放宽阈值。
- 注意：删完后它仍以 3.6% 的记录吃掉 **11.4% 的残基**（抗原中位 607 aa），算力占比远大于行数占比。

### 7.5 加载期过滤在 train/valid 上严重不对称

`asd_antibody` train 丢 **67.5%** 但 valid 只丢 **5.0%**（13 倍差）；`tcr_native` 32.7% vs 12.2%；
`trait` 54.5% vs 54.3% 对称。

后果：**ASD 的 valid loss 不能用于 early-stopping** —— valid 保留了大量被从 train 剥掉的
Kong 相似簇抗体家族，在测一个训练时被刻意屏蔽的分布。无泄漏风险（valid 非 Kong 基准本身），
机制未查清，待决策。

### 7.6 下游去重审计

5 源 × 11 基准矩阵（`scripts/data/dedup/audit_downstream_leakage.py`）：
**`HARD-REQUIREMENT hits = 0 -> PASS`**（NM2025 seen/unseen + public_trackA × 5 个 TCR 源全为 0）。

审计本身修过三个缺陷，三次都是「过滤看起来生效了，其实没有」：① 语料过期（§6.9）；
② **黑名单构造缺陷** —— 曾把黑名单写成 `语料 ∩ benchmark`，只对构建时那份语料完备，实测 862 键仅覆盖
hard-requirement 保护集的 4.2%，**TRAIT 此前 0 命中是运气不是过滤生效**；已改为
`簇级交集 ∪ binding_benchmark ∪ full_bank` = 59,212 键，代价 TRAIT 保留行 −11.2%；
③ 审计结论行误导 —— 曾把按构造就该非零的裸 core 命中计入总数，已改为只用 hard-requirement 行驱动结论。

---

## 8. 表征分析：「BERT 表征为何一般」的定位

**问题**：先验上 BERT 目标应更利于表征，但 RESULTS §0.1–§0.3 里 BERT 臂并不占优。
需要判断是评测坏了还是现象为真。**结论：无 bug，现象为真且可分辨，但成因分两种、须分开说。**

**先证伪「pipeline 有 bug」**（三项都过）：

1. checkpoint 加载无缺口 —— BERT 与 diffusion 的 `model.safetensors` 键集逐个相同（各 386 张量）。
2. 打分口径无误 —— 独立 harness 在同一份 T2 数据上复现出 BERT 0.0187 / diffusion 0.0275，
   对上 RESULTS 的 0.0186 / 0.0277。
3. padding 没污染 final feature —— 同一序列 alone / 同长 batch / 被 pad 到 125 宽，
   最后一层 pooled 余弦 **0.999998**。

**成因（2026-08-31 配对重测，推翻了此前两版结论）**

在同一重采样上给两臂打分（配对统计量，probe 20 次分层划分 / ARI 8 种子 / kNN@1 2000 次 bootstrap）：

| 指标 | 读出 | Δ = diff − BERT | 配对 sd | 判定 |
|---|---|---:|---:|---|
| 25 类 probe | **raw** | **+0.0330** | 0.0064 | ✅ 5.1 sd，20/20 同号 |
| 25 类 probe | 白化 | −0.0039 | 0.0056 | ❌ 0.7 sd = 0 |
| K-sweep ARI | **raw** | **+0.0092** | 0.0012 | ✅ 7.6 sd，8/8 同号 |
| K-sweep ARI | 白化 | +0.0027 | 0.0018 | ❌ 1.5 sd |
| kNN@1 | raw / 白化 | +0.0061 / −0.0043 | 0.0045 / 0.0038 | ❌ 均 CI 跨 0 |

- ✅ **线性可解码性：BERT 不缺信息、缺的是几何。** 只做方差均衡，**BERT 涨 +0.0818（9.2 sd）
  而 diffusion 只涨 +0.0449**，probe 差距抹平到 0.7 sd。BERT 的涨幅是臂间差距的 2.5 倍。
  非 transductive 泄漏（只在训练划分上拟合白化，数值几乎不变）。
- 🔴 **全局簇结构：diffusion 的优势没被解释掉。** ARI 差距缩小**不是 BERT 变好**（+0.0014，1.3 sd）
  **而是白化把 diffusion 弄坏了**（−0.0052，4.1 sd）。

**几何退化的度量（可引用，不依赖采用后处理）**

官方 T3 deep k=200：raw **0.712 / 0.709** → 仅去中心化 0.696 / 0.693 → PCA 白化 256 **0.760 / 0.755**。
**仅换缩放就能多拿约 0.047，大于表里任何模型间差异**（此前最大 0.011），也比 top-k 波动带（≤0.004）
大一个数量级。这不是调参余量，是几何退化的度量 —— 即**表内绝对值是下界**。
根因：mean pairwise cosine 从 layer0 的 0.71/0.58 涨到最后一层 **0.994/0.986**，eff-rank 38→17 / 42→20。

一个可能的机制解释（**未做反事实训练验证**）：BERT 只掩 15% + `add` 条件下可训 ESMC 看得到 85% 干净序列
⇒ 任务偏局部；diffusion 的 `t~U(0,1)` 常掩掉大半序列、逼 decoder 建全局结构 ⇒ 簇更成形。
支持性观测：BERT 最后一个 block + `ln_f` 把上下文推回输入附近（CKA 对 layer0：layer7 0.478 → layer8 0.809），
最终层 92.3% 方差可由 ESMC 线性预测（diffusion 89.0%）。

**为什么不采用重标定**（2026-08-31 决策，`post=` 代码已从 `common/model_api.py` 移除）：
它是 post-hoc 补救，回答不了「为什么模型本身产不出可直接度量的空间」。我们的预训练目标是 token 级重建，
从未约束嵌入空间几何，所以长成这样是**预期行为**；SCEPTR 不需要重标定，是因为对比学习
（InfoNCE + L2 归一化）在训练时就把空间约束成近似各向同性。
**durable 的修法在训练侧（加空间约束/对比项），不在读出侧** —— 这正是 §4.4 对照臂要测的东西。

### 🚫 已撤回，不要再引用

- ~~「BERT 反超 diffusion」「白化后排序翻转」~~ —— 白化下是**抹平到 0（0.7 sd）而非反转**。
- ~~「局部邻域 BERT ≥ diffusion」~~ —— kNN@1 两个读出下 95% CI 都跨 0，**从未可分辨**。
- ~~「BERT 臂最好的表征是 ESMC encoder 自身、LLaDA decoder 净负贡献」~~ —— 基于 raw 特征的错误结论。
- ~~「越过 CDR3-Levenshtein / k-mer」「与 TCRdist 差 0.017」~~ —— headline（raw）下与 TCRdist 仍差 0.065、
  与 SCEPTR 仍差 0.078。
- RESULTS §0.3 那个 24 类 probe（0.804 vs 0.794）是**无误差棒单点值**（≈1.6 sd），不应引用；
  T3 deep/broad 有 100 seeds 波动带，可引用。

### 🟠 作用域警告

**8B 上 T3 三项全部反向**（deep k=200 BERT 0.7093–0.7133 vs diff 0.7198–0.7204、broad 0.669 vs 0.680、
probe 0.816 vs 0.828，波动带均不重叠），且 8B 未做配对重测。加上**每格 n=1 次训练、两条 ckpt 步数不同
（49000 vs 42000）**，**本工作不支持任何「哪个预训练目标更适合表征」的结论。**

未验证假设：8B BERT 余弦挤到 **0.9999**（四条最极端），受的几何惩罚最重，若几何解释成立，
8B 的差距也应塌掉。**补跑 8B 配对测量是最有价值的一项后续。**

**产物**：`scripts/diagnostics/{diag_repr_layers,diag_layer_cka,diag_batch_invariance,diag_final_readout,
diag_noise_floor,plot_*}.py`（纯诊断，不在评测路径上）。完整报告
[`debug/BERT_REPR_DEBUG.md`](../../debug/BERT_REPR_DEBUG.md)；详表见 RESULTS §0.0 缺陷 (f)。

### 顺带修的一个真隐患：评测加载器改 fail-fast

`load_fusion_for_eval` 原本对 missing key 只 `logger.warning`，而 decoder 由
`LLaDAModelLM(cfg, init_params=False)` 构建（**分配后不初始化**）⇒ 真缺 key 时张量拿的是分配器残留内存，
评测会**静默地给一个随机权重打分**。修法：加载**前** `_poison_parameters` 把所有浮点参数灌 NaN，
加载后 `_assert_fully_loaded` 凡残留 NaN 即 `RuntimeError`，并区分「缺权重」与「权重在盘上就是 NaN」。
权重绑定自动放过（绑定伙伴填共享存储 ⇒ 无残留 NaN）。验证：270m/8B 四条全部 residual NaN = 0，
删单个张量的负向测试正确抛错。

另修：`LLaDAModelLM.__init__` 自建内层模型时**硬编码 `init_device="cuda"`**（注释「always on CPU」已过期），
使 `device="cpu"` 失效、8B 在忙卡上直接 OOM。**未改共享的 `modeling_llada.py`**，改为在 loader 里自建
`LLaDAModel` 再经 `LLaDAModelLM(cfg, model=inner)` 传入。

---

## 9. 运行方式

环境：`/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr`（ESMC + LLaDA 依赖齐全，
esm 3.2.3 / torch 2.8 / transformers 4.48.1 + peft 0.14.0）。下游评测也用这个 env。
联网走代理 `http://100.68.162.212:3128`。

本地 dry_run（不加载 8B，仅验证数据/词表/渲染）：

```bash
cd dllm_test
python examples/llada/protein_pretrain_esmc.py --dry_run True --max_rows_per_source 8
```

正式训练走 volc 作业提交，配置在 `train_jobs/protein_esmc_*.yml`，提交步骤与硬规则见
[`README.md`](README.md)「提交训练任务」。提交前必过两道 pre-flight：
`assert_corpus_fresh.py`（§6.9）与 `assert_residue_alphabet.py`（§6.8）。

下游评测走 `eval_jobs/*.yml`，runner 是 `scripts/downstream/run_immune_fusion_{repr,gen,pairing}.sh`。
排不上队时可本地跑，口径要求见 §5.5。

---

## 10. 待办 / 未解

**决策类**

- [ ] **全链 2M 是否续跑**：已 Killed 于 161000（0.6277，同时是最好点），满包完整可直接 resume。
      2M 目标 ≈ 68 天独占 8 卡，现在只跑到 8.1%，而 §5.1 显示长跑对 T4 无改善、对 CDR 在噪声内。
- [ ] **提交多链关系对照臂**（§4.4）：`..._chainratio_immune_v3_4gpu` 与 `..._chainratio_cognate_...`。
      不要碰 2M。headline = ImmunoMatch + PLL。
- [ ] **grammar_v2 五条 339 G 是否保留**：内部无 optimizer 只能整份删；`..._7l` 被 `PROJ_GUIDE.md` 钉住，
      其余四条待科研决策。
- [ ] **T4 Setting-A 参考集是否换成修正版**（§5.4）：换了会改动已有 novelty/JSD 数值。

**工程缺陷（未修）**

- [ ] 🔴 **磁盘配额本身未修**（§6.4）。09-01 回收后 767 G，中途涨回约 820 G；
      2026-09-12 快照 `output/` 约 **466 GB**（距上次 1.2 TB 撞墙水位约 734 GB）。
      8 卡长跑每 1000 步仍要过一次约 9.0 GB 的一次性余量窄门。
- [ ] **修 `TopKValLossCheckpointCallback` 的账本重置**：启动时应从磁盘现存 checkpoint 重建 top-k，
      而不是只看本进程 `log_history`，否则每次重启都产孤儿、永不被剪。
- [ ] **save 前做配额预检**：余量不足时降级为 weights-only，而不是让整个任务 `exit 1`
      后再被独立重提、再烧 58 分钟一轮（历史上那 4 连 Failed 是 4 个不同 JobId 的独立提交，
      不是同一 JobId 被平台 `RetryOptions` 自动重试；见 §6.4）。
- [x] 旧 `ImmuneCsvDataset` 动态 raw CSV loader 已删除；当前训练只消费 prepared dataset。若重建语料，使用 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/preprocess_immune_dataset.py`，不要恢复训练期整表加载。

**回填 RESULTS**

- [ ] 组 D 中间点 69000 / 73000 / 121000 / 137000 / 144000 / 151000 与 generated-only 44000、
      BERT 1M 105000 的数字写入 RESULTS §0.1–§0.6 与 §0.8 组 D（§5.1、§5.2 已有读出值）。

**未解 / 待验证**

- [ ] **train loss（~7.3）与 `eval_loss`（0.70）差一个数量级，原因未查明。** 两条路径的 CE 计算相同
      （`loss_weight_type='none'`、无 chain 加权），且首跑与 resume 后完全一致，所以不是故障征兆。
      **登记为未解项，不要拿 train loss 与 eval_loss 跨口径比。**
- [ ] **补跑 8B 配对测量**（§8 作用域警告）—— 目前最有价值的一项后续。
- [ ] 单链 α 摄入需扩 `<tcra>`/`<tcrb>` 词表，与现有 checkpoint 不兼容（§7.3）。
- [ ] ASD valid loss 不可用于 early-stopping，机制未查清（§7.5）。
- [ ] 可选：从 `NoAttentionMaskWrapper` 切换为保留 attention_mask。

---

## 11. 变更日志

> 一行一条，细节在对应小节。不要在这里重复正文内容。

- **2026-09-12** — v5 开跑前核算：三个 flag 不动。步时/eval/配额/deadline 见 §6.3 / §6.4 / §6.7；离线 binding 脚本见 §5.6。
- **2026-09-12** — 更正 §6.4 / §10：4 连 Failed 是 4 个不同 JobId 的独立提交，不是平台 `RetryOptions` 自动重试。
- **2026-09-12** — 更正 §4.1：generated-only 8gpu_2m 是 4 卡 `checkpoint-42000` weights-only 热启动，不是 from scratch。
- **2026-09-12** — 提交 v5 8-GPU diffusion（Queue）。账本 PROJECT_PROCESS 同日条。
- **2026-09-12** — 表位源补全 + all-X 已在全量语料核验；v5 已发布。见 PROJECT_PROCESS 同日条与 plan §2.5/§2.6/§4.2。
- **2026-09-11** — 无条件布局加固定 `<null>` 前缀（renderer 改动，预处理无需重跑）；wandb 切 `online`。细节见 plan §2.3.2。
- **2026-09-11** — v4 数据线完成流式 region profile、beta-only 双链补全、synthetic X loss exclusion、relation target diffusion；链顺序在渲染层统一为 **β→α**（`grammar.py` 的 `receptor = [beta, alpha]`），anchor 判定改 provenance 驱动，corpus 切到 `tcr_repertoire_junc80`。全量 preprocessing 跑完（raw 8,441,615 → kept 7,768,293，train 7,665,574 / valid 102,719），全量审计零不变量违规，`nonbinding` 负样本路径在真实数据上验通。正式训练尚未提交。见 `docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md`。
- **2026-09-07** — 三条长跑最好点下游**全部收口**：表征三条 + T4/CDR 四条本机单卡跑完，
  两条 pairing 平台 Success（全链 151k IM **0.438** / generated-only 44k **0.399**）。见 §5.1。
- **2026-09-07** — 九条下游作业在 `c20250601` 闲时排 4.9h 零起跑，改投 `queue012` 非闲时后仍 9/9 Queue；
  按阶段耗时只留两条 pairing、其余改本地跑。`c20250601` 旧闲时九条全部 cancel 成 `Killed`，双跑覆盖风险解除。见 §5.1 / §6.7。
- **2026-09-07** — 全链 2M `t-20260902021013-bm79q` 于 09-06T19:37Z **Killed**（`signal: 15`），
  盘上 161000 满包完整，待决策。见 §3。
- **2026-09-07** — 4 卡 generated-only（2M 权重源）的 wandb 离线主跑已同步上云：
  2139 train 点 / 42 eval 点，最好 42000 / 0.7548。5 个重试残片未传（与主跑尾段完全重复）。
- **2026-09-04 → 09-06** — 全链 2M 中间点 69000 / 73000 / 121000 / 137000 / 144000 陆续评测；
  新增 T4 `max_iter` sweep 脚本（此前 T4 写死 32 且输出不带 iter 标签，是三个阶段里唯一没扫过的）。见 §5.2。
- **2026-09-02** — 多链关系对照臂落地（分链 t per-row multinomial 修复 + cognate InfoNCE + 两条 YAML），
  **作业未提交，2M 未改未 resume**。见 §4.4。
- **2026-09-02** — 全链 33000 与 8 卡 2M 18000 全套下游 8/8 Success：表征与 generated-only 同档，
  **配对抬起来了**（IM 0.436 / 0.416 vs 组 C 的 0.353），T4 仍差。见 §5.2。
- **2026-09-01** — 8 卡长跑 1M/2M 起跑（global 256 + polynomial，与 50k 那批不可混排）；
  🔴 磁盘配额同一处 4 连 Failed、净进度 0；执行回收 `output/` 1.2 T → 767 G。见 §4.1 / §6.4。
- **2026-09-01** — 口径决策：**BERT 只评表征不评生成**；表征读出保持 raw，`post=` 重标定代码移除。见 §5.4 / §8。
- **2026-08-31** — 补误差棒，**推翻自己前两天的两条结论**（「BERT 反超」「局部邻域 BERT 更强」均撤回）；
  评测加载器改 fail-fast，杜绝「拿未初始化内存当权重」。见 §8。
- **2026-08-31** — 修 `tcr_papers_v2` 第 3046 行的空位符 `-`，解除 4 卡任务的重试死循环。见 §6.8。
- **2026-08-30** — 定位「BERT 表征一般」：无 bug，根因是最后一层几何退化；新增一批诊断脚本。见 §8。
- **2026-08-30** — 新增 `--diffusion_all_chains` 开关并提交 4 卡任务；
  global batch 改 256（`per_device 8` 实测 OOM，改走 `ga`）；补齐 4 卡 bert 臂。见 §4.2 / §6.2 / §6.3。
- **2026-08-29** — v3 双臂提交（数据参数逐字节一致，唯一差异是目标函数）；
  改投 4 卡非闲时**19 秒起跑**（8 卡排不上是凑不出整节点）；闲时版三件套 + 看护循环上线；
  布局口径纠错（前缀采样 → 蓄水池采样，`tcr_pmhc` 曾低估 4 倍）。见 §4.2 / §6.6 / §6.7 / §7.2。
- **2026-08-28** — 数据扩充（新增 `tcr_repertoire`、`tcr_papers` → v2）+ 修三个去污缺陷，
  审计达 `HARD-REQUIREMENT hits = 0 -> PASS`。见 §7.1 / §7.6 / §6.9。
- **2026-08-04** — 队列统一改 `queue012`，不再用 `spot-share-queue`；入口支持 `--decoder_init scratch`。
- **2026-08-02/03** — 融合通路从零打通到 8B 端到端跑通（resize 陷阱、dtype bug、FSDP 梯度不同步 bug
  各修一处），bert 改经典 80/10/10 MLM，top-k by eval_loss 落地。见 §4.3。
