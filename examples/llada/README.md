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
> **Slurm users:** Update `scripts/train.slurm.sh` and `mkdir .logs`: see [(optional) Slurm setup](../../README.md#optional-slurm-setup) for details.
>
> **MoE checkpoints:** For models like [`LLaDA-MoE-7B-A1B-Base`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base), set `"model_type"` to `"lladamoe"` in the checkpoint’s `config.json`:
> ```diff
> - "model_type": "llada",
> + "model_type": "lladamoe",
> ```
> -->


## 蛋白预训练（本项目）

在**不改动 LLaDA 训练主干**的前提下，让 LLaDA 在 OAS（抗体）/ OTS（TCR）等七源免疫序列上做
ESMC 条件融合预训练。历史工作线支持 `diffusion` 与 `bert` 两种目标；**本轮 v4 只提交一个正式
`diffusion` 版本**，不把 BERT、all-chain 或 relation auxiliary 对照臂列为正式目标。

| 文件 | 作用 |
|---|---|
| [`protein_pretrain_esmc.py`](protein_pretrain_esmc.py) | **现役正式入口**：ESMC 条件融合 + 七源语料 + diffusion/bert 双目标 |
| [`protein_fusion_model.py`](protein_fusion_model.py) | `LLaDAEsmcFusion` 模型、两种加噪、`RemapCollator`、词表扩展与重映射 |
| `protein_pretrain.py` | **已删除**（2026-09-10）；不要恢复。现役入口是上一行 |
| [`load_fusion_checkpoint.py`](load_fusion_checkpoint.py) | 从 FSDP checkpoint 还原融合模型（下游评测用） |
| [`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) | 蛋白训练线决策笔记本（Volc 账本权威在 [`PROJECT_PROCESS.md`](../../PROJECT_PROCESS.md)） |
| [`MULTI_CHAIN_RELATION.md`](MULTI_CHAIN_RELATION.md) | 多链关系调研 + 分链 t / cognate 对照臂（**不要和 2M 混跑**） |
| [`DATA_PIPELINE_README.md`](DATA_PIPELINE_README.md) | raw 语料去污事故（prepared 入口见 pipeline README） |
| [`dllm/pipelines/immune_llada/README.md`](../../dllm/pipelines/immune_llada/README.md) | **当前 prepared 数据操作入口**（v4 / v5 均已发布） |

### 当前训练任务与数据版本（2026-09-12）

**v4 正式训练入口（仅一个 diffusion 版本，尚未提交）**：

- 配置：[`train_jobs/protein_esmc_llada270m_diffusion_immune_v4_4gpu.yml`](../../train_jobs/protein_esmc_llada270m_diffusion_immune_v4_4gpu.yml)
- prepared data：`data/prepared/immune_v4_beta_relation`
- output：`output/protein_esmc_llada270m_diffusion_immune_v4_4gpu`
- 数据语义（补全、all-X、null 前缀、relation target）不要在此复述：
  [`dllm/pipelines/immune_llada/README.md`](../../dllm/pipelines/immune_llada/README.md)
  与 [`docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md`](../../docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md)。
- **v5 已发布**：`configs/data/immune_v5_receptor_completion.yaml` →
  `data/prepared/immune_v5_receptor_completion/`。行数 / v4 对照：plan §4.2。
- **v5 8-GPU 已提交**（Queue）：账本 [`PROJECT_PROCESS.md`](../../PROJECT_PROCESS.md) 2026-09-12 条。配置 [`train_jobs/protein_esmc_llada270m_diffusion_immune_v5_8gpu.yml`](../../train_jobs/protein_esmc_llada270m_diffusion_immune_v5_8gpu.yml)。
- **wandb `online`**，节点不可达则自动回退 offline（egress 未在计算节点验证）。
- 提交前：smoke/full preprocess、manifest/profile 审计、residue alphabet、corpus freshness。

v3 任务 / eval 数字 / task id 账本在 [`PROJECT_PROCESS.md`](../../PROJECT_PROCESS.md)
与 [`RESULTS.md`](../../downstream/benchmark/RESULTS.md)，下面只留仍会踩坑的操作规则。

### 仍有效的操作陷阱（v3 任务账本见 PROJECT_PROCESS）

🔴 **目录磁盘配额（v3 长跑踩过，v4 提交前仍有效）。** 当时的 `..._8gpu_2m` 已因
`safetensors_rust.SafetensorError: ... Disk quota exceeded (os error 122)`
在 **step 17000 的 `save_model` 上 4 连 Failed** —— 每轮「从 16000 续 → 跑满 1000 步 → 同一处爆」
耗 58 分钟，**净进度 0**，约 3 小时 8 卡非抢占资源白烧。第 5 次（`t-20260902021013-bm79q`）
于 19:10:42 存盘成功，但只是期间别的任务腾出了空间，**配额仍贴临界、问题未修**。

- ⚠️ **别看 `df` 下结论**：底层 `fs_vepfs-cnbj2c98dea54433` 3.1P 已用 2.3P、**尚余 809T**。
  爆的是**目录/租户配额**，不是文件系统满。
- 一次 save 需要约 **9.0 GB 一次性余量**（`model.safetensors` 2.3G + `pytorch_model_fsdp.bin` 2.3G
  + `optimizer.bin` 4.5G），而 **top-k 剪枝在写完之后**，峰值 = 现有占用 + 一整个新 ckpt。
- 当时的 4 连 Failed 是 **4 个不同 JobId 的独立提交**（`t-20260901033935-h55f4` /
  `t-20260901230256-ns4q4` / `t-20260902000627-j4mqm` / `t-20260902010820-t6hq6`），
  不是同一 JobId 被平台 `RetryOptions` 自动重试：每个 JobId 只有 1 个实例，相邻提交间隔
  5～30 秒，与 `IntervalSeconds: 180` 不符。对照 `t-20260901235433-q76qw` 同样
  `EnableRetry: true` 却停在 Failed。**配额不足仍会导致连续 Failed、净进度 0**。
  证据见 [`PROJECT_PROCESS.md`](../../PROJECT_PROCESS.md) Active 表旁注与
  [`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) §6.4。
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

v3 长跑 / 50k 短跑 / 下游 task id 与 headline 数字的权威位置：
[`PROJECT_PROCESS.md`](../../PROJECT_PROCESS.md)、
[`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) §3.1、
[`RESULTS.md`](../../downstream/benchmark/RESULTS.md)。
不要把过期 Running/Queue 表抄回本 README。

⚠️ **跨 world size 只能 weights-only init**：4 卡 FSDP pack 的 `optimizer.bin` /
`pytorch_model_fsdp.bin` 不能直接给 8 卡续。entrypoint 两个挑选函数语义不同、不要合并：
`pick_latest_full` 要求三件套齐全，`pick_latest_weights` 只认 `model.safetensors`。
三件套判据必须保留（曾正确跳过半截 `checkpoint-17000`）。

### 多链关系对照臂（YAML 已就位，未提交）

不要焊进现役 2M `OUTPUT_DIR`。设计见 [`MULTI_CHAIN_RELATION.md`](MULTI_CHAIN_RELATION.md)。
**不要**和 2M / all-chains / BERT 并排 `eval_loss`。提交前先看配额（一次 save ≈ 9 GB）。

⚠️ **`eval_loss` 不可跨口径比**（目标函数 / loss 覆盖 / batch）。BERT 的 0.60 不代表
比 diffusion 的 0.75 好。加倍 batch 只能走 `ga`，不能走 `per_device`（`per_device 8`
在真实七源上 step 2 OOM）。空位符 `.`/`-`/`|` 会杀死 DataLoader rank；提交前跑
`assert_residue_alphabet.py`。细节见 PROTEIN_PRETRAIN_PROGRESS。

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
> `ml_task get` 不回显属正常：`RoleRestartPolicy`（角色级）与 `RetryOptions`（作业级）是两层。
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

> Read [Useful tips for training](../../README.md#useful-tips-for-training) and [(optional) Slurm setup](../../README.md#optional-slurm-setup) before training.
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
> Read [(optional) Evaluation setup](../../README.md#optional-evaluation-setup) before running evaluation. 

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
