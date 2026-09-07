# LLaDA

> 📄 Paper: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992) | 💻 Code: [github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)

Resources and examples for training (finetuning & pretraining) and evaluating diffusion language models **LLaDA**.

> [!IMPORTANT]
> **本目录同时承载本项目的免疫受体（抗体 / TCR）蛋白预训练工作线**，现役入口是
> [`protein_pretrain_esmc.py`](protein_pretrain_esmc.py)，与下面的上游 LLaDA 通用示例互不相干。
> 见 [蛋白预训练（本项目）](#蛋白预训练本项目)。

## Table of Contents
- [蛋白预训练（本项目）](#蛋白预训练本项目) — 本项目工作线；以下各节为上游 dllm 原文
- [Files](#files)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)

<!-- ## Setup
> [!IMPORTANT]  
> **Slurm users:** Update `scripts/train.slurm.sh` and `mkdir .logs`: see [(optional) Slurm setup](/README.md#optional-slurm-setup) for details.
>
> **MoE checkpoints:** For models like [`LLaDA-MoE-7B-A1B-Base`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base), set `"model_type"` to `"lladamoe"` in the checkpoint’s `config.json`:
> ```diff
> - "model_type": "llada",
> + "model_type": "lladamoe",
> ```
> -->


## 蛋白预训练（本项目）

在**不改动 LLaDA 训练主干**的前提下，让 LLaDA 在 OAS（抗体）/ OTS（TCR）等七源免疫序列上做
ESMC 条件融合预训练，支持 `diffusion`（扩散加噪）与 `bert`（固定比例 MLM）两种目标。

| 文件 | 作用 |
|---|---|
| [`protein_pretrain_esmc.py`](protein_pretrain_esmc.py) | **现役正式入口**：ESMC 条件融合 + 七源语料 + diffusion/bert 双目标 |
| [`protein_fusion_model.py`](protein_fusion_model.py) | `LLaDAEsmcFusion` 模型、两种加噪、`RemapCollator`、词表扩展与重映射 |
| [`protein_pretrain.py`](protein_pretrain.py) | 早期无 ESMC 入口，保留作对照 |
| [`load_fusion_checkpoint.py`](load_fusion_checkpoint.py) | 从 FSDP checkpoint 还原融合模型（下游评测用） |
| [`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) | **进度 / 决策 / 验证的权威记录**，任何实质改动先记这里 |
| [`MULTI_CHAIN_RELATION.md`](MULTI_CHAIN_RELATION.md) | 多链关系调研 + 分链 t / cognate 对照臂（**不要和 2M 混跑**） |
| [`DATA_PIPELINE_README.md`](DATA_PIPELINE_README.md) | 数据源、去污、审计运行手册 |

### 当前训练任务（2026-09-01 19:20Z）

🔴 **头号阻塞：目录磁盘配额。** 现役 `..._8gpu_2m` 已因
`safetensors_rust.SafetensorError: ... Disk quota exceeded (os error 122)`
在 **step 17000 的 `save_model` 上 4 连 Failed** —— 每轮「从 16000 续 → 跑满 1000 步 → 同一处爆」
耗 58 分钟，**净进度 0**，约 3 小时 8 卡非抢占资源白烧。第 5 次（`t-20260902021013-bm79q`）
于 19:10:42 存盘成功，但只是期间别的任务腾出了空间，**配额仍贴临界、问题未修**。

- ⚠️ **别看 `df` 下结论**：底层 `fs_vepfs-cnbj2c98dea54433` 3.1P 已用 2.3P、**尚余 809T**。
  爆的是**目录/租户配额**，不是文件系统满。
- 一次 save 需要约 **9.0 GB 一次性余量**（`model.safetensors` 2.3G + `pytorch_model_fsdp.bin` 2.3G
  + `optimizer.bin` 4.5G），而 **top-k 剪枝在写完之后**，峰值 = 现有占用 + 一整个新 ckpt。
- `RetryOptions: MaxRetryTimes: 50` 会把它变成循环，按剩余次数最多还能烧约 46 小时。
- ✅ **已回收 389 GB：`output/` 1.2 T → 767 G**（2026-09-01 19:30Z）。删作废目录、剪两条已终态
  8B run 的 resume-only、5 处 `checkpoint-final` **改硬链接**、清 6 条 run 的账本外孤儿。
- ⚠️ **但配额本身没修**：`TopKValLossCheckpointCallback` 仍在每次重启时重置账本、继续产孤儿，
  save 仍需 9.0 GB 一次性余量。回收只是把窄门往后推。细节见
  [`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) §4.2.7。
- ⚠️ **两个刻意保留、清盘时不要碰**：
  `..._diffusion_allchains_immune_v3_4gpu/checkpoint-33000`（`..._8gpu_2m` 的权重来源）、
  `..._bert_immune_v3/checkpoint-50000` 完整满包（`..._v3_1m` 的 resume 来源）。
- ⚠️ **`checkpoint-final` 现在是 `checkpoint-50000/model.safetensors` 的硬链接**（`nlink=2`）。
  两条路径都仍然有效（`eval_jobs` 引用前者 8 处、`train_jobs` 引用后者 2 处），但
  **改其中一个就是改另一个**，不要原地覆写。

**新一轮：8 卡长跑（步数拉长一个数量级）。** 50k 短跑已收口出数，现役配置全部
**global 256 + polynomial power=1**，与 50k 那批**不可混排**（目标函数、batch、LR 调度三样都变了）：

| Job | Task ID | 队列 | 卡数 | max_steps | 状态 |
|---|---|---|---|---:|---|
| `protein_esmc_llada270m_diffusion_allchains_immune_v3_8gpu_2m` | **`t-20260902021013-bm79q`** | `c20250601` | 8 | **2,000,000** | **Running** @26k（weights-only 自 4 卡 33000；前 4 条配额 Failed） |
| `protein_esmc_llada270m_bert_immune_v3_1m` | **`t-20260902110050-zb7dr`** | `queue012` | 8 | **1,000,000** | Queue，**从头训**（global **256** = 2 × ga 16 × 8；polynomial `4e-5→1e-5`，warmup 2000。不加载 50k。前几条已 cancel/Failed） |
| `protein_esmc_llada270m_diffusion_immune_v3_spot_2m` | `t-20260901032620-vvngv` | `c20250601` 闲时 | 8 | 2,000,000 | Queue（看护循环在追） |
| `protein_esmc_llada270m_diffusion_allchains_immune_v3_4gpu_2m` | `t-20260901032611-npf27` | — | 4 | — | **Killed**（换成 8 卡版） |

⚠️ **2M 步在 2.95 s/it 下是 1,640 小时 ≈ 68 天**独占 8 卡（进度条自印 `1644:30:32`）。
`ActiveDeadlineSeconds: 7776000`（90 天）放得下，但这个步数目标是否按 68 天规划的**待确认**。

⚠️ **`..._8gpu_2m` 首跑只能加载权重、不能满包续**：源是 4 卡 FSDP pack，
`optimizer.bin` / `pytorch_model_fsdp.bin` 不能跨 world size，所以走
`--init_fusion_weights <4卡 ckpt>/model.safetensors`，optimizer 与 polynomial 从 step 0 新建。
entrypoint 因此有两个挑选函数、**语义不同不要合并**：`pick_latest_full` 要求
`optimizer.bin`+`pytorch_model_fsdp.bin`+`scheduler.pt` **三件齐全**（本目录已有 8 卡 pack 时的
普通 resume），`pick_latest_weights` 只认 `model.safetensors`（跨 world size 的首次 init）。
✅ 三件套判据这次救了一命 —— 它正确跳过了只剩 132 MiB 半截 `model.safetensors` 的
`checkpoint-17000`、回退到完整的 16000，**没有重演 §4.2.6 的「从坏 ckpt 反复续跑」死循环**。
**改 checkpoint 挑选逻辑时必须保留这个判据。**

### 多链关系对照臂（YAML 已就位，未提交）

pairing ImmunoMatch 贴地板（≈0.353 vs 错配 0.333）是因为现役只学 joint 重建。
对照实验**新开 OUTPUT_DIR**，不要焊进 `..._8gpu_2m`。详见
[`MULTI_CHAIN_RELATION.md`](MULTI_CHAIN_RELATION.md) 与
[`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) §4.9。

| Job | 旋钮 | 状态 |
|---|---|---|
| `protein_esmc_llada270m_diffusion_chainratio_immune_v3_4gpu` | generated-only + 分链 t `0.5/0.15/0.15/0.1/0.1` | YAML 已写，**未提交** |
| `protein_esmc_llada270m_diffusion_chainratio_cognate_immune_v3_4gpu` | 同上 + `--relation_aux cognate` | YAML 已写，**未提交** |

- 配方对齐 v3 **4 卡 generated-only 50k / global 128**，from scratch。
- Headline：生成 pairing **ImmunoMatch** + `scripts/downstream/score_pairing_pll.py` 的 `p(L\|H)−p(L)`。
- **不要**和 2M / all-chains / BERT 并排 `eval_loss`。
- 提交前先看配额（一次 save ≈ 9 GB）。命令见 `MULTI_CHAIN_RELATION.md` §4.3。

---

以下是已收口的 v3 双臂 50k 短跑（七源语料，bert vs diffusion，两臂数据参数逐字节一致，唯一差异是目标函数）：

| Job | Task ID | 队列 | 卡数 | 抢占 | global batch | 状态 |
|---|---|---|---|---|---|---|
| `protein_esmc_llada270m_diffusion_immune_v3` | `t-20260829031748-96vjf` | `queue012` | 8 | 否 | 128 | **Failed**；last 42000，无 final |
| `protein_esmc_llada270m_bert_immune_v3` | `t-20260829031757-2qnjn` | `queue012` | 8 | 否 | 128 | **Success** @50000；best val 48000 |
| `protein_esmc_llada270m_bert_immune_v3_1m` | **`t-20260902110050-zb7dr`** | `queue012` | 8 | 否 | **256** | Queue；**从头**训到 1M，polynomial `4e-5→1e-5`，warmup 2000 |
| `protein_esmc_llada270m_diffusion_immune_v3_spot` | `t-20260829135424-zxdjg` | `c20250601` | 8 | **是（闲时）** | 128 | Running |
| `protein_esmc_llada270m_bert_immune_v3_spot` | `t-20260830095121-lx6kp` | `c20250601` | 8 | **是（闲时）** | 128 | Running（看护循环重提过） |
| `protein_esmc_llada270m_diffusion_immune_v3_4gpu` | **`t-20260830152319-r5w9k`** | `c20250601` | **4** | 否 | 128 | 🔴 **卡在 42000/50000**（headline eval **0.7548 @42000**；top-k：39000 / 41000 / 42000）— 续跑崩在 step ~42782，见下 |
| `protein_esmc_llada270m_diffusion_allchains_immune_v3_4gpu` | **`t-20260830135521-zf5rr`** | `c20250601` | **4** | 否 | **256** | **Running**（见 §4.2.4 / §4.2.5） |
| `protein_esmc_llada270m_bert_immune_v3_4gpu` | **`t-20260830135524-qq7n5`** | `c20250601` | **4** | 否 | **256** | **Running**（2026-08-30 新建，见 §4.2.5） |

### 下游评测（v3 diffusion）

已收口的是两条 **generated-only / global 128** diffusion 的 headline ckpt（全套 T1/T2/T3 + AB CDR + pairing + T4）。
**BERT 只评表征（T1/T2/T3），不跑 CDR / pairing / T4。**

**现役（2026-09-07 13:25 CST）：三条长跑各自最好 val 点在评，T4 + CDR 已出数。** 全链
**`checkpoint-151000`**（eval **0.6326**）全套；generated-only 8 卡 2M
**`checkpoint-44000`**（eval **0.7520**）全套；BERT 1M **`checkpoint-105000`**
（eval **0.4263**）**只评表征**。三条训练不打断。组 D / 组 C / BERT 1M
**`eval_loss` 不可比**，下游也不要混排。

**关键结果（2026-09-07，T4 + CDR，本地单卡跑；数字权威在
[RESULTS.md](../../downstream/benchmark/RESULTS.md) §0.4 / §0.5 / §0.8 组 D）**

| 结论 | 依据 |
|---|---|
| **长跑到 151k 没改善 T4，反而微降** | common-6 d_edit：26k **8.58** → 121k **8.78** → 151k **8.80**；仍远差于旧 270m 6.99 与不看表位的 OLGA 6.34 |
| **CDR 小幅上行，但在噪声内** | SAbDab 旧快照 H3：26k 41.68 → 151k **41.92**（+0.24）；同臂 18k→26k 的 H2 波动就有 +0.41 |
| **全链 vs generated-only 打平，`--diffusion_all_chains` 无可见收益** | 151k − 44k：SAbDab H1/H2/H3 +0.28/+0.45/+0.13，SAb23H2 互有胜负，T4 则**落后 0.19** |

⚠️ **两条 pairing 还没跑**（`t-20260907044307-btjxr` / `t-20260907044321-ht8b6` 仍在
`queue012` 排队），而 pairing 是 headline 指标 —— **上面三条只覆盖 T4 与 CDR，本轮尚未收口。**
BERT 1M 105000 的表征作业已 cancel、未跑。

> **为什么这四个阶段是本地跑的。** `queue012` 单卡是最挤的档位（非终态 758 条里单卡
> 607 排队 / 106 Running，多卡排队仅 27 条），九条排 41 分钟零起跑。按实测中位耗时
> **T4 12min < CDR 16min < repr 28min ≪ pairing 165min**，保留 pairing 在队列、
> 其余七条 cancel，四个 T4/CDR 阶段拿本机 A100-80G 串行跑完（56 分钟，4/4 exit 0）。
> env 直接激活 eval YAML 里指定的
> `/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr`，`CDR_MAX_ITER=2`、
> batch `4 8` 与平台 entrypoint 逐行一致，**与平台跑同口径**。
> Runner：`output/_local_runs/run_local_t4_cdr.sh`（日志 `local_run.log`，耗时 `status.tsv`）。
> **本地跑单卡评测是排不上队时的正当退路**，产物路径与平台跑完全相同。

| Run | Headline ckpt | eval_loss | tag |
|---|---|---:|---|
| 8 卡 `..._immune_v3` | `output/protein_esmc_llada270m_diffusion_immune_v3/checkpoint-27000` | 0.7676 | `ours_fusion_v3_diff_27000` |
| 4 卡 `..._immune_v3_4gpu` | `output/protein_esmc_llada270m_diffusion_immune_v3_4gpu/checkpoint-42000` | 0.7548 | `ours_fusion_v3_4gpu_diff_42000` |
| 4 卡 `..._allchains_..._4gpu` | `output/protein_esmc_llada270m_diffusion_allchains_immune_v3_4gpu/checkpoint-33000` | **0.6866** | `ours_fusion_v3_allchains_33000` |
| 8 卡 `..._allchains_..._8gpu_2m` | `output/protein_esmc_llada270m_diffusion_allchains_immune_v3_8gpu_2m/checkpoint-18000` | **0.7044** | `ours_fusion_v3_allchains_8gpu2m_18000` |
| 8 卡 `..._allchains_..._8gpu_2m` | `output/protein_esmc_llada270m_diffusion_allchains_immune_v3_8gpu_2m/checkpoint-26000` | **0.6902** | `ours_fusion_v3_allchains_8gpu2m_26000` |
| 8 卡 `..._allchains_..._8gpu_2m` | `output/protein_esmc_llada270m_diffusion_allchains_immune_v3_8gpu_2m/checkpoint-69000` | **0.6577** | `ours_fusion_v3_allchains_8gpu2m_69000` |
| 8 卡 `..._allchains_..._8gpu_2m` | `output/protein_esmc_llada270m_diffusion_allchains_immune_v3_8gpu_2m/checkpoint-73000` | **0.6535** | `ours_fusion_v3_allchains_8gpu2m_73000` |
| 8 卡 `..._allchains_..._8gpu_2m` | `output/protein_esmc_llada270m_diffusion_allchains_immune_v3_8gpu_2m/eval_snapshot_121000` | **0.6416** | `ours_fusion_v3_allchains_8gpu2m_121000` |
| 8 卡 `..._allchains_..._8gpu_2m` | `output/protein_esmc_llada270m_diffusion_allchains_immune_v3_8gpu_2m/eval_snapshot_137000` | **0.6332** | `ours_fusion_v3_allchains_8gpu2m_137000` |
| 8 卡 `..._allchains_..._8gpu_2m` | `output/protein_esmc_llada270m_diffusion_allchains_immune_v3_8gpu_2m/eval_snapshot_151000` | **0.6326** | `ours_fusion_v3_allchains_8gpu2m_151000` |
| 8 卡 `..._immune_v3_8gpu_2m`（generated-only） | `output/protein_esmc_llada270m_diffusion_immune_v3_8gpu_2m/eval_snapshot_44000` | **0.7520** | `ours_fusion_v3_genonly_8gpu2m_44000` |
| 8 卡 `..._bert_immune_v3_1m` | `output/protein_esmc_llada270m_bert_immune_v3_1m/eval_snapshot_105000` | **0.4263** | `ours_fusion_v3_bert_1m_105000` |

- 表征入口：`scripts/downstream/run_immune_fusion_repr.sh <ckpt> <tag> 16`（读出 `grammar:decoder:global:<ckpt>`）。
- 生成入口：`scripts/downstream/run_immune_fusion_gen.sh <ckpt> <tag> 4 8`（`GEN_STAGES=cdr|pairing|t4`）。
  解码步数：CDR `CDR_MAX_ITER=2`、pairing `PAIR_MAX_ITER=124` 可用 env 覆盖；**T4 在该脚本里写死 32**。
  要扫 T4 步数用 `scripts/downstream/sweep_immune_fusion_t4_maxiter.sh <ckpt> <tag> "8 32 64 128" 8`
  （产物带 iter 标签，不与上面的默认 32 产物冲突）。
- 作业 YAML：generated-only 用 `eval_jobs/eval_v3_{8gpu_27000,4gpu_42000}_{repr,cdr,pairing,t4}.yml`；
  全链用 `eval_jobs/eval_v3_allchains_{33000,8gpu2m_18000,8gpu2m_26000,8gpu2m_69000,8gpu2m_73000,8gpu2m_121000,8gpu2m_137000,8gpu2m_151000}_{repr,cdr,pairing,t4}.yml`；
  generated-only 8 卡 2M 用 `eval_jobs/eval_v3_genonly_8gpu2m_44000_{repr,cdr,pairing,t4}.yml`；
  BERT 50k 只提交 `eval_jobs/eval_v3_bert_final_repr.yml`；1M 用 `eval_jobs/eval_v3_bert_1m_{34000,105000}_repr.yml`。
- 提交：`bash scripts/volc-no-proxy.sh ml_task submit --conf eval_jobs/<yml>`。
  🔴 **2026-09-07 起下游单卡一律投 `queue012` + `Preemptible: false`**：本账号对 `c20250601`
  （`q-20260121145036-6fztt`）**已无 `CreateCustomTask` 权限**（09-07 中午实测），往那个队列提交直接被 IAM 拒。
  `StopCustomTask` 当时也被拒，但 **09-07 07:20 CST 同一账号再试，`c20250601` 上 creator 为 `251105016`
  的九条闲时任务全部 `cancel success`** —— 存量任务要停先直接 `ml_task cancel` 试，不必默认走控制台；
  `CreateCustomTask` 是否也恢复未测。本轮三最好点九条已按此改投（见下表新 ID）。
  69000 / 73000 / 121000 全套 / 137000 表征那几批是**历史**的 `c20250601` 闲时提交，更早的全链 8 条是非抢占。
  BERT 表征**不交 CDR / pairing / T4**。
  BERT 50k tag `ours_fusion_v3_bert_final`，ckpt 为 `output/protein_esmc_llada270m_bert_immune_v3/checkpoint-final`。
  BERT 1M 当前最好点 tag `ours_fusion_v3_bert_1m_105000`，快照 `output/protein_esmc_llada270m_bert_immune_v3_1m/eval_snapshot_105000/`。
  已产出的 BERT CDR/T4 数字**不进 RESULTS、不当 headline**。
- 任务 ID 账本：闲时首提 `output/downstream_generation/eval_v3_bestval_151000_44000_105000_20260906_task_ids.tsv`；
  `queue012` 非闲时重提 `output/downstream_generation/eval_v3_bestval_queue012_nonpreempt_20260907_task_ids.tsv`。
- 数字进 [`downstream/benchmark/RESULTS.md`](../../downstream/benchmark/RESULTS.md) §0.1–§0.6 与 **§0.8**，按 tag 检索。本表不抄排行榜。作业结束后本表只补 Success + RESULTS 锚点。

| Job | Task ID | 覆盖 | 状态 |
|---|---|---|---|
| `eval-v3-8gpu-27000-repr` | `t-20260831005742-fbc4r` | T1 / T2 / T3 | Success → [RESULTS §0.1–§0.3](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-8gpu-27000-cdr` | `t-20260831005745-6tmwn` | AB CDR infill | Success → [§0.5](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-8gpu-27000-pairing` | `t-20260831005749-qpgjw` | AB light pairing | Success → [§0.6](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-8gpu-27000-t4` | `t-20260831005752-27hv7` | T4 generation | Success → [§0.4](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-4gpu-42000-repr` | `t-20260831005755-59vjd` | T1 / T2 / T3 | Success → [§0.1–§0.3](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-4gpu-42000-cdr` | `t-20260831005759-qfwq4` | AB CDR infill | Success → [§0.5](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-4gpu-42000-pairing` | `t-20260901011610-tb4hh`（首次 `t-20260831005803-xvl6m` Failed） | AB light pairing | 重提 |
| `eval-v3-4gpu-42000-t4` | `t-20260901011614-5wzjc`（首次 `t-20260831005807-spgtc` Failed） | T4 generation | Success → [§0.4](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-bert-final-repr` | `t-20260901113303-kqd4h` | T1 / T2 / T3 **only** | Success |
| `eval-v3-bert-1m-34000-repr` | `t-20260904104944-mdr7x` | T1 / T2 / T3 **only** | 闲时已提交 |
| `eval-v3-bert-final-cdr` | `t-20260901113306-p6c46` | 误提，已 Success | **不作数** |
| `eval-v3-bert-final-pairing` | `t-20260901113309-p7cp9` | 误提 | **已 cancel** |
| `eval-v3-bert-final-t4` | `t-20260901113312-zmxm9` | 误提，已 Success | **不作数** |
| `eval-v3-allchains-33000-repr` | `t-20260902044100-wnkdb` | T1 / T2 / T3 | Success → [§0.1–§0.3](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-allchains-33000-cdr` | `t-20260902044104-29sqz` | AB CDR infill | Success → [§0.5](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-allchains-33000-pairing` | `t-20260902044108-ddnrw` | AB light pairing | Success → [§0.6](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-allchains-33000-t4` | `t-20260902044113-xxnp7` | T4 generation | Success → [§0.4](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-allchains-8gpu2m-18000-repr` | `t-20260902044117-vh48s` | T1 / T2 / T3 | Success → [§0.1–§0.3](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-allchains-8gpu2m-18000-cdr` | `t-20260902044121-q2rt5` | AB CDR infill | Success → [§0.5](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-allchains-8gpu2m-18000-pairing` | `t-20260902044126-mw2s6` | AB light pairing | Success → [§0.6](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-allchains-8gpu2m-18000-t4` | `t-20260902044130-zx5vk` | T4 generation | Success → [§0.4](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-allchains-8gpu2m-26000-repr` | `t-20260902104333-87ccm` | T1 / T2 / T3 | Success → [§0.1–§0.3](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-allchains-8gpu2m-26000-cdr` | `t-20260902104338-7ptgf` | AB CDR infill | Success → [§0.5](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-allchains-8gpu2m-26000-pairing` | `t-20260902104342-68qz8` | AB light pairing | Success → [§0.6](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-allchains-8gpu2m-26000-t4` | `t-20260902104346-882r9` | T4 generation | Success → [§0.4](../../downstream/benchmark/RESULTS.md) |
| `eval-v3-allchains-8gpu2m-69000-repr` | `t-20260904013714-4w8nk` | T1 / T2 / T3 | Success（数字待回填） |
| `eval-v3-allchains-8gpu2m-69000-cdr` | `t-20260904013717-rhmzz` | AB CDR infill | Success（数字待回填） |
| `eval-v3-allchains-8gpu2m-69000-pairing` | `t-20260904013720-nprjc` | AB light pairing | Success（数字待回填） |
| `eval-v3-allchains-8gpu2m-69000-t4` | `t-20260904104404-t6j7v`（前次 `t-20260904013725-7zcvf` Killed） | T4 generation | 闲时补提 |
| `eval-v3-allchains-8gpu2m-73000-repr` | `t-20260904104145-77cfr` | T1 / T2 / T3 | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-73000-cdr` | `t-20260904104148-6x5m9` | AB CDR infill | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-73000-pairing` | `t-20260904104152-pxjd4` | AB light pairing | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-73000-t4` | `t-20260904104155-tctxm` | T4 generation | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-121000-repr` | `t-20260905191346-rbt8z` | T1 / T2 / T3 | **Success** |
| `eval-v3-allchains-8gpu2m-121000-cdr` | `t-20260906125811-fdrvj` | AB CDR infill | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-121000-pairing` | `t-20260906125815-q8h8b` | AB light pairing | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-121000-t4` | `t-20260906125820-db5bd` | T4 generation | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-137000-repr` | `t-20260906125806-xkg7q` | T1 / T2 / T3 | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-144000-repr` | `t-20260906132412-tklfb` | T1 / T2 / T3 | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-121000-t4-itersweep` | `t-20260906132835-xnnlf` | T4 `max_iter` 8/32/64/128 | 闲时已提交 |
| `eval-v3-allchains-8gpu2m-151000-pairing` | **`t-20260907044307-btjxr`**（旧闲时 `t-20260906233451-fjzmm` 09-07 已 cancel → Killed） | AB light pairing | **`queue012` 非闲时，Queue（保留）** |
| `eval-v3-genonly-8gpu2m-44000-pairing` | **`t-20260907044321-ht8b6`**（旧闲时 `t-20260906233508-svn2t` 09-07 已 cancel → Killed） | AB light pairing | **`queue012` 非闲时，Queue（保留）** |
| `eval-v3-allchains-8gpu2m-151000-repr` | `t-20260907044259-qd52v` | T1 / T2 / T3 | 09-07 已 cancel（让位 pairing） |
| `eval-v3-allchains-8gpu2m-151000-cdr` | `t-20260907044303-ks4x6` | AB CDR infill | 09-07 已 cancel（让位 pairing） |
| `eval-v3-allchains-8gpu2m-151000-t4` | `t-20260907044310-wlm9h` | T4 generation | 09-07 已 cancel（让位 pairing） |
| `eval-v3-genonly-8gpu2m-44000-repr` | `t-20260907044314-mtzn9` | T1 / T2 / T3 | 09-07 已 cancel（让位 pairing） |
| `eval-v3-genonly-8gpu2m-44000-cdr` | `t-20260907044317-kfhk2` | AB CDR infill | 09-07 已 cancel（让位 pairing） |
| `eval-v3-genonly-8gpu2m-44000-t4` | `t-20260907044324-tcmhs` | T4 generation | 09-07 已 cancel（让位 pairing） |
| `eval-v3-bert-1m-105000-repr` | `t-20260907044328-wwfjn` | T1 / T2 / T3 **only** | 09-07 已 cancel（让位 pairing） |

> **2026-09-07 只留 pairing**：`queue012` 单卡是最挤的档位（607 条同规格排队 / 106 Running），
> 九条排 41 分钟零起跑。按实测中位耗时 **T4 12min < CDR 16min < repr 28min ≪ pairing 165min**，
> pairing 是关键路径，其余七条 cancel 让位（cancel 时均仍 `Queue`，无中断损失）。
> 七条 ckpt / tag 未变，要补做直接用原 YAML 重提。
>
> **2026-09-07 07:20 CST：`c20250601` 上旧闲时九条（`t-20260906233442-d2gzt` … `t-20260906233517-9x8kq`
> 及补提的 `t-20260907001930-qtf2s`）已全部 `ml_task cancel` 成 `Killed`**，cancel 时全部仍 `Queue`、
> 从未起跑。双跑覆盖 `output/downstream_generation/<tag>_*` 的风险解除。平台上我们只剩上面两条
> `queue012` pairing 与两条 8 卡训练在非终态。详见 PROGRESS §4.2.11。

跨任务总表：[RESULTS §0.8](../../downstream/benchmark/RESULTS.md) **组 D**。33000 / 18000 / 26000 已收口。
本轮三最好点数字待回填。**不与组 C（generated-only 50k）混排**；generated-only 8 卡 2M 也是另一组。
账本：`output/downstream_generation/eval_v3_bestval_151000_44000_105000_20260906_task_ids.tsv`。

对照规则（写进任何结论前先看）：

1. 两条 v3 同配方、同 global 128，**可以互比**；8 卡评的是 27k、4 卡是 42k，差距首先是进度，不是卡数。
2. 与 RESULTS 里旧 270m diffusion@42000 **不可并排**——旧条是六源、去污前。
3. T4 跨模型只认 `bioseq_unseen_common`（6 个 pMHC）；Char-BLEU 不跨解码器比。

🔴 **batch 口径分两组，跨组比 loss 无意义**：前五条是 global **128**（8 卡 per_device 2 × ga 8；
4 卡 per_device 4 × ga 8），后两条是 global **256**（per_device 4 × **ga 16** × 4 卡）。
128 那档 50000 步 ≈ 0.84 epoch，256 那档 ≈ 1.64 epoch（墙钟约 50 h）。组内可比、跨组不可比。

⚠️ **加倍只能走 `ga`，不能走 `per_device`。** `per_device 8` 实测在真实七源混合上 step 2 就
OOM（`76.29 GiB is allocated`），而只喂 `asd_antibody` 的本机探测预测才 43 GiB —— 低估 33 GiB，
因为 ASD 样本只有 2 条链，而 TRAIT / TCR 样本带 peptide + MHC + TCRα + TCRβ 共 4-5 条，
ESMC 编码器的激活是按 `per_device × 链数 × 链长` 走的，且它**不被 FSDP 分片、也没有
activation checkpointing**。`ga` 不改变张量形状，显存画像与已跑满 42k 步的 `per_device 4`
完全相同，所以是零风险路径。定容细节与复现方法见
[`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) §4.2.5。

✅ **空位符阻塞已修（2026-08-31）**：七源 × train/valid/holdout 共 21 个文件、约 880 万行，
非法残基字符（`.` / `-` / `|`）只出现 **1 格**——`tcr_papers_v2/train.csv` 第 3046 行
`cdr3b` 已从 `ASSKVAARVP-TLKLS` 改成 `ASSKVAARVPTLKLS`（行数不变，4 卡从
`checkpoint-42000` 续跑不会移位）。`assert_residue_alphabet.py` 现默认扫全部 21 个文件，
exit 0。4 卡 diffusion 可以 resume 了，见
[`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) §4.2.6。

提交前的语料 gate 现在有两个，建议都跑：

```bash
python scripts/data/tcr_native/assert_corpus_fresh.py \
    data/tcr_papers_v2/dataset/finalize_report.json \
    data/tcr_repertoire/dataset/build_report.json   # 去污新鲜度
python scripts/data/assert_residue_alphabet.py       # 残基字母表，发现坏行 exit 1
```

⚠️ **`eval_loss` 不可跨口径比大小**（三层切割线：目标函数、loss 覆盖范围、batch），
BERT 的 0.60 不代表比 diffusion 的 0.75 好。各条实时数字与解释见
[`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) §4.2.0。

⚠️ **配置性失败后要立刻 cancel 重试链**：`RetryOptions: PolicySets: [Failed]` 分不清节点故障
和参数错，OOM 那条 bert 被自动重提成同名任务 `t-20260830135809-zq7pg`（仍是 per_device 8），
与正确任务共用同一 `OUTPUT_DIR`，已手动 cancel。

`OUTPUT_DIR` 互不重叠，并跑不会互相破坏，只浪费配额 ——
**同一臂哪条先出 checkpoint 就 cancel 其余的**，别让多条都跑到 50k。
当前 `queue012` 的 8 卡 diffusion 正在重跑 4 卡那条已完成 84% 的同一臂（只是不占
`c20250601` 配额），是最该先 cancel 的一条。
超参、语料实测与设计意图见 [`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) §4.2。

`..._diffusion_allchains_...` 那条是 **`--diffusion_all_chains True`** 的变体：**没有任何链是
fixed 的**，抗原 / MHC / peptide 也被加噪并计入 diffusion loss（generated-only 的镜像，对应
BERT 那边的 `--bert_all_chains`）。它与 `..._diffusion_immune_v3_4gpu` 除该开关、batch、
`OUTPUT_DIR`、`run_name` 外逐字一致（`diff` 核过）。
**首跑必须 from scratch** —— 目标函数和 batch 口径都变了，旧 checkpoint 不可续。

### 提交训练任务

配置在 `train_jobs/*.yml`（volc `ml_task submit` 格式），提交前先跑语料新鲜度断言：

```bash
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
python scripts/data/tcr_native/assert_corpus_fresh.py \
    data/tcr_papers_v2/dataset/finalize_report.json \
    data/tcr_repertoire/dataset/build_report.json   # 不过就别提交，任务拿到资源也会挂在这一步
volc ml_task submit --conf train_jobs/<job>.yml
```

默认 `queue012` + `Preemptible: false`，1×8 卡 `ml.pni2.28xlarge`。**8 卡排不上时有两条路，先试第一条。**

> 🔴 **2026-09-07 权限变更：`c20250601` 已提交不了。** 本账号对该队列
> （`q-20260121145036-6fztt`）无 `CreateCustomTask`，`ml_task submit` 返回
> `User is not authorized to perform: ml_platform:CreateCustomTask`。当时对该队列上别人 creator 的
> 存量任务也报无 `StopCustomTask`，但 **09-07 07:20 CST 再试 cancel 已放行**（九条闲时评测全部 Killed，
> 见 PROGRESS §4.2.11），所以停任务先试 CLI，不必默认走控制台。提交侧是否恢复未测，
> 「② 借闲时资源」那条路目前仍按**走不通**处理，新任务（含下游单卡）一律投 `queue012`。
> `queue012` 上 submit / cancel 均已实测可用。

#### ① 降到 4 卡 + 非闲时（优先）

2026-08-29 实测：`c20250601` 上 7 条 8 卡任务全在 Queue（最早的等了 9.8 小时），而 Running 的
**全是单卡**——缺的是**连续整节点 8 卡**，不是队列配额。同一时刻提一条 4 卡非抢占任务
（`ml.pni2.14xlarge` + `Preemptible: false`，先例见 `Protenix-v2/train_jobs/*_1node4gpu*.yml`），
**19 秒就 Running**。卡数减半但把省下的乘数补进 `per_device`，global batch 不变、跨卡数可比：

| | 8 卡 | 4 卡 |
|---|---|---|
| `per_device_train_batch_size` | 2 | **4** |
| `gradient_accumulation_steps` | 8 | 8 |
| global batch | 2 × 8 × 8 = **128** | 4 × 8 × 4 = **128** |

`per_device` 翻倍前先测过显存。单卡 A100-80G、无 FSDP（模型状态未分片，故是 4 卡 FSDP 的**上界**）、
`per_device=4` + `max_length=1024`：训练稳态峰值 **30.7GB**，`save_model` 保存 checkpoint 时
瞬时冲到 **55GB**（`fsdp_state_dict_type: FULL_STATE_DICT` + `--save_only_model False`），均在 80GB 内。
`num_processes` 由平台注入的 `MLP_WORKER_GPU` 决定，改 flavor 即可，**不要硬编码卡数**。
代价只是 wall-clock 变慢（同样 50k 步、同样样本量，卡少一半）。
模板见 `train_jobs/protein_esmc_llada270m_diffusion_immune_v3_4gpu.yml`。

#### ② 借闲时（抢占）资源

改投 `c20250601` + `Preemptible: true`（**闲时 / 抢占资源**），但闲时资源随时被回收，
**必须同时满足下面三条，缺一不可**（模板见 `train_jobs/*_immune_v3_spot.yml`）：

1. **独立 `OUTPUT_DIR`**（如加 `_spot` 后缀）。抢占版与非抢占版会同时排队、谁先拿到资源不确定；
   共用目录会让两个进程同时写同一批 `checkpoint-*` 并各自触发 top-k 剪枝，checkpoint 直接报废。
2. **自动 resume**。去掉 `test ! -d checkpoint-1000` 这类反 resume 断言（否则第二次启动直接失败），
   改为取编号最大的 checkpoint 拼 `--resume_from_checkpoint`；目录里没有 checkpoint 时**不能**传
   该参数，HF Trainer 会因找不到而报错。entrypoint 开着 `set -euo pipefail`，首跑时 `ls` 无匹配
   会让整个脚本退出，所以 `|| true` 不能省：

   ```bash
   LATEST_STEP="$( (ls -d "${OUTPUT_DIR}"/checkpoint-[0-9]* 2>/dev/null || true) | sed 's#.*/checkpoint-##' | sort -n | tail -1 )"
   ```

   能接得住靠 `TopKValLossCheckpointCallback` 保证**最新** checkpoint 始终带完整 optimizer/FSDP
   state（非最新的被 slim 成 weights-only），且 `--save_only_model False`。
3. **`RetryOptions`**。平台的「抢占后强制重试」已下线，不自己开就等于被抢一次训练永久中断：

   ```yaml
   RetryOptions:
     EnableRetry: true
     MaxRetryTimes: 50
     IntervalSeconds: 180
     PolicySets: ["Failed", "InstanceReclaimed"]   # InstanceReclaimed = 闲时资源回收
   ```

   配合第 2 条，每次被抢占最多丢 `save_steps`（现为 1000）步。

   > ⚠️ **不要照抄仓库里现成的抢占 YAML 当模板。** 五个 `*_jobs/` 目录共 1266 个任务配置，
   > 其中 866 个是 `Preemptible: true`，但**没有一个**配了 `RetryOptions`（2026-08-29 实测）——
   > 它们写在平台下线「抢占后强制重试」之前，当时不需要自己配。现在照抄就会静默失去这层保护。

4. **外部看护循环**（见下）。`RetryOptions` 只管「实例被回收后重试」，一旦重试次数用尽或任务
   因别的原因落到 `Failed`/`Killed` 终态，平台就不再管，训练会永久停在那里。

#### 闲时任务看护循环（掉了自动从 last ckpt 续跑）

```bash
bash scripts/monitor_spot_tasks_start.sh   # 后台常驻，脱离终端
bash scripts/monitor_spot_tasks_stop.sh    # 停止
tail -f output/_monitor/monitor.log        # 看日志
cat output/_monitor/state.json             # 看当前追踪的 task id 与重提历史
```

每 10 分钟轮询一次 `volc ml_task list --name <TaskName>`，只看**最新一条**同名任务：非终态
（`Queue`/`Staging`/`Running`/`Killing`/`Initialized`）不动作；`Success` 标记完成并停止看护；
落到 `Failed`/`Killed` 就用同一个 YAML 重新提交 —— entrypoint 里的 `RESUME_ARG` 会自动挑
`OUTPUT_DIR` 下编号最大的 checkpoint，所以**重提即从 last ckpt 续跑**。

> 🚨 **人工 cancel 任务前，务必先放 STOP 哨兵**，否则看护会把你 cancel 掉的任务重提回来：
>
> ```bash
> touch output/_monitor/STOP                    # 停掉整个循环
> touch output/_monitor/STOP.<TaskName>         # 只停某一条
> ```
>
> 比如 8 卡那对起跑后要 cancel 闲时这对，就得先 `touch STOP`。

几个刻意的设计（细节与实测见 [`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) §4.2.3）：

- **只认最新一条同名任务。** `--name` 是模糊匹配且会返回历史终态记录（例如 08-29 cancel 重提
  留下的 `t-20260829135133-5np6m`/`Killed`），不取最新就会被历史骗到、无限重提。
- **重提有冷却（15 分钟）和上限（100 次）**，防止任务秒失败时刷出一堆提交。
- **停滞只告警、不自动 kill。** `Running` 但 wandb 超过 1 小时没更新会打 `WARN`，但不自动处理 ——
  eval（每 1000 步、七源各 2000 行）本身就会拉长写入间隔，误杀比漏报代价大。
- **看护自己崩了会被拉起。** 启动脚本外面套了一层 wrapper，python 非 0 退出 10 秒后自动重启
  （已用 `kill -9` 实测，日志留下 `rc=137` 后自愈的记录）；`exit 0`（STOP 或全部 `Success`）才收工。
- **重复启动被 PID 文件挡住** —— 起两个看护会让同一个终态任务被重提两次。

> `volc ml_task` **没有 update 子命令**，改 YAML 不会回写已提交的任务 —— 配置写错只能 cancel 重提。
> 提交后用 `volc ml_task export -t <id> --config` 回读，确认平台侧真的记下了 `RetryOptions`。
> 另外 volc 命令**不要走代理**（`env -u http_proxy -u https_proxy ...`）。

---

## Files
```
# Pipeline modules relevant to LLaDA
dllm/pipelines/llada
├── __init__.py                     # Package initialization
├── models/
│   ├── configuration_lladamoe.py   # LLaDA-MoE model configuration
│   ├── configuration_llada.py      # LLaDA model configuration
│   ├── modeling_lladamoe.py        # LLaDA-MoE model architecture
│   └── modeling_llada.py           # LLaDA model architecture
├── eval.py                         # Evaluation module
├── sampler.py                      # Inference module
└── trainer.py                      # Training module (pretraining, SFT, and GRPO/RL)

# Example entry points for training / inference / evaluation
examples/llada
├── chat.py                         # Interactive inference example
├── eval.sh                         # Automatic evaluation example
├── grpo.py                         # GRPO/RL training entry point
├── sample.py                       # Inference example
├── pt.py                           # Pretraining example
├── README.md                       # Documentation (you are here)
└── sft.py                          # Supervised finetuning example
```

> 上面是上游的文件树。本项目在同一目录另加了 `protein_pretrain_esmc.py`、
> `protein_fusion_model.py`、`protein_pretrain.py`、`load_fusion_checkpoint.py`
> 与两份中文文档，见 [蛋白预训练（本项目）](#蛋白预训练本项目)。
<!-- > [!NOTE] -->
<!-- >  - We fixed attention mask bugs in [`modeling_lladamoe.py`](/dllm/pipelines/llada/models/modeling_lladamoe.py) and [`modeling_llada.py`](/dllm/pipelines/llada/models/modeling_llada.py). We recommend loading models with `dllm.utils.get_tokenizer`; otherwise `import dllm` before calling `AutoModel.from_pretrained` to ensure the correct models from `dllm` are used. 
> 
>  - We fixed bugs in `chat_template` and assign `mask_token` through `dllm.utils.get_tokenizer`. If you use `AutoTokenizer`, keep in mind to set `chat_template` and `mask_token` appropriately yourselves. -->

<!-- > [!WARNING]  
> Before loading MoE checkpoints (e.g., [inclusionAI/LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base)), first overwrite the `model_type` field from `inclusionAI/LLaDA-MoE-7B-A1B-Base/config.json`:  
> ```diff
> - "model_type": "llada",
> + "model_type": "lladamoe",
> ``` -->

## Training

> Read [Useful tips for training](/README.md#useful-tips-for-training) and [(optional) Slurm setup](/README.md#optional-slurm-setup) before training.
>
> **MoE checkpoints:** For models like [`LLaDA-MoE-7B-A1B-Base`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base), set `"model_type"` to `"lladamoe"` in the checkpoint’s `config.json`:
<!-- > ```diff
> - "model_type": "llada",
> + "model_type": "lladamoe",
> ```
> -->

### SFT

For example, to SFT [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) on the [`alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset for instruction following on 8 GPUs, run:
```shell
accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/llada/sft.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 1024 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir ".models/LLaDA-8B-Base/alpaca"
```
If you are using slurm and want to train across, for example, 2 nodes (16 GPUs total), run:
```shell
sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llada/sft.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 1024 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir ".models/LLaDA-8B-Base/alpaca"
```

<!-- **Reproducing [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)**. Though LLaDA is trained on proprietary data, we tried our best to reproduce LLaDA-8B-Instruct by finetuning LLaDA-8B-Base using our training pipeline on public instruction-following dataset [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture): -->

#### Reproducing [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) with SFT
Though LLaDA is trained on proprietary data, we tried our best to reproduce [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) by finetuning [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) with SFT on the [`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) dataset:

```shell
# Preprocessing SFT data (optional, but can avoid redundant preprocessing for multi-node training)
python dllm/tools/preprocess_sft_dataset.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --sft_map_fn_path "dllm.utils.default_sft_map_fn" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir ".data/sft/llada/tulu-3-sft-mixture" \
    --num_proc 64

# Train on 24*8=192 A100s with FSDP, take about 8 hours
sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llada/sft.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args ".data/sft/llada/tulu-3-sft-mixture" \
    --load_preprocessed_data True \
    --max_length 1024 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir ".models/LLaDA-8B-Base/tulu-3-sft-mixture"
```
<!-- [TODO] Training curves are on Wandb; checkpoints with evaluation results are available on Hugging Face. See the [Evaluation](#evaluation) section below for evaluation instructions. -->


### RL (GRPO)

We adapt [GRPO](https://arxiv.org/abs/2402.03300) (Group Relative Policy Optimization) for masked diffusion language models via `DiffuGRPOTrainer`, which replaces autoregressive generation with iterative denoising. The implementation follows the [d1/diffu-grpo](https://github.com/dllm-reasoning/d1/tree/main/diffu-grpo) reference.

For example, to run GRPO on [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) with `gsm8k` on 1 GPU:
```shell
accelerate launch \
    --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/llada/grpo.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --dataset gsm8k \  # supported: gsm8k, countdown, sudoku, math, code
    --num_train_epochs 1 \
    --output_dir ".models/LLaDA-8B-Instruct/gsm8k-grpo"
```

To train with LoRA on 8 GPUs using DeepSpeed ZeRO-2:
```shell
accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/llada/grpo.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --lora_r 128 --lora_alpha 64 \
    --dataset gsm8k \
    --num_train_epochs 10 --learning_rate 3e-6 \
    --num_generations 6 --per_device_train_batch_size 6 \
    --beta 0.04 --epsilon 0.5 \
    --output_dir ".models/LLaDA-8B-Instruct/gsm8k-grpo"
```

Key diffusion-specific arguments: `--block_size`, `--steps`, `--remasking`, `--p_mask_prompt`.
Key GRPO arguments: `--beta`, `--epsilon`, `--num_generations`, `--num_iterations`.

### Pretraining

Pretrain on [`mlfoundations/dclm-baseline-1.0`](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) from scratch using 192 GPUs (24x8) and FSDP:
```shell
sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llada/pt.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "mlfoundations/dclm-baseline-1.0" \
    --max_length 1024 \
    --max_steps 2000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --output_dir ".models/LLaDA-8B-Base/dclm-baseline-1.0"
```

## Inference
We support batch inference for standard sampling and infilling:
```shell
python examples/llada/sample.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"
```
We also support interactive multi-turn dialogue with visualization:
```shell
python examples/llada/chat.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"
```

## Evaluation
> Read [(optional) Evaluation setup](/README.md#optional-evaluation-setup) before running evaluation. 

For example, to evaluate [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) on [gsm8k](https://huggingface.co/datasets/openai/gsm8k) using 4 GPUs, run:
```shell
# Use model_args to adjust the sampling arguments for evaluation.
accelerate launch --num_processes 4 \
    dllm/pipelines/llada/eval.py \
    --tasks "gsm8k_cot" \
    --model "llada" \
    --apply_chat_template \
    --num_fewshot 5 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,max_new_tokens=512,steps=512,block_size=512,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"
```

To automatically evaluate [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) and [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) on all benchmarks, run:
```shell
bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Instruct --instruct True
bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Base --instruct False
```

For **Fast-dLLM** sampling and evaluation with LLaDA, see the [Fast-dLLM README](../fastdllm/README.md).

### Evaluation results

> Results (Reproduced) are evaluated using our framework, while results (Official) come from the original [paper](https://arxiv.org/abs/2502.09992). All evaluation settings follow the configurations in the [LLaDA](https://github.com/ML-GSAI/LLaDA) repository, with minor adjustments. 

|               | MMLU | BBH | ARC&#8209;C | Hellaswag | TruthfulQA | WinoGrande | PIQA | GSM8K | Math | GPQA | HumanEval | MBPP | CEval | CMMLU |
|:----------------|:----:|:-----:|:-----------:|:-----------:|:------------:|:----:|:-----:|:----:|:-----:|:----:|:-----------:|:----:|:------:|:------:|
| [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) (Official) | 65.9 | 49.7 | 45.9 | 70.5 | 46.1 | 74.8 | 73.6 | 70.3 | 31.4 | 25.2 | 35.4 | 40.0 | 70.5 | 69.9 |
| [`LLaDA-8B-Base`](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) (Reproduced) | 65.9 | 47.2 | 44.1 | 69.2 | 45.6 | 70.4 | 70.7 | 70.7 | 32.4 | 31.9 | 32.9 | 38.8 | 70.4 | 69.8 |


<p align="center" style="color: #808080; font-size: 0.9em;">
Table 1. Evaluation results of 
<a href="https://huggingface.co/GSAI-ML/LLaDA-8B-Base" style="color: #808080; text-decoration: none;">
<code>LLaDA-8B-Base</code>
</a>.
</p>

|                 | MMLU | MMLU&#8209;Pro | ARC&#8209;C | Hellaswag | GSM8K | Math | GPQA | HumanEval | MBPP | 
|:----------------|:----:|:---------:|:-----:|:-----------:|:-----:|:----:|:----:|:-----------:|:----:|
| [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) (Official) | 65.5 | 37.0 | 88.5 | 74.6 | 69.4 | 31.9 | 33.3 | 49.4 | 41.0 |
| [`LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) (Reproduced) | 69.8 | 36.2 | 86.4 | 76.7 | 74.7 | 31.9 | 30.6 | 47.0 | 40.0 |

<p align="center" style="color: #808080; font-size: 0.9em;">
Table 2. Evaluation results of 
<a href="https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct" style="color: #808080; text-decoration: none;">
<code>LLaDA-8B-Instruct</code>
</a>.
</p>
