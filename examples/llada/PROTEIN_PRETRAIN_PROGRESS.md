# LLaDA 蛋白预训练（OAS/OTS）项目进度

> 实时记录本项目的目标、决策、进度与验证。每次有实质进展请在末尾「变更日志」追加带时间戳的条目。

- 负责入口脚本：[`examples/llada/protein_pretrain.py`](protein_pretrain.py)
- 最近更新：2026-08-02

---

## 0. 协作规则（Agent Rules）

> 本项目所有 agent（主控与子代理）必须遵守：

1. **实时更新进展**：每当有实质进展（新增/修改代码、跑通验证、发现问题、做出决策），必须同步更新本文档 —— 更新「4. 当前状态」表，并在「8. 变更日志」追加一条带日期的条目。不要攒到最后再补。
2. **子任务可直接用 grok 4.5**：拆分出的子任务（写代码、调研、批量修改等）可直接交给 subagent 执行，模型默认使用 **grok 4.5**（`cursor-grok-4.5-high-fast`）；主控模型负责统筹、拆解与 review，不需要为每个子任务重复征求许可。

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

## 3. 改动范围（现阶段）

本任务到目前为止只**新增了 2 个文件**，**未修改任何现有代码文件**：

| 文件 | 类型 | 说明 |
|------|------|------|
| [`examples/llada/protein_pretrain.py`](protein_pretrain.py) | 新增（约 409 行） | 训练入口脚本 |
| [`examples/llada/PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) | 新增 | 本进度文档 |

> 注意：仓库 `git status` 中其它 `M`（如 `dllm/pipelines/bioseq/datasets.py` 的 nanobody 目录名改动）是**先前已存在的未提交改动，与本任务无关**，我们没有触碰。

脚本主要组件：
- `expand_tokenizer_and_build_remap`：加 25 残基 `<res_X>` + 16 grammar token + `<chainsep>`（共 42），建 `grammar_id → llada_id` 重映射表。
- `GrammarRemapDataset`：`build_mixed_immune_dataset`(仅 oas+ots) → `BioSeqRecord`(带角色) → `GrammarRenderer.encode` → 重映射 → `{input_ids, labels}`。
- `ConstantAlphaScheduler`：bert 模式的常数 α 调度器。
- `train()`：参数解析 → tokenizer+remap → (dry_run 提前退出) → get_model+resize(+LoRA) → 选调度器 → `MDLMTrainer` 训练与保存。

## 4. 当前状态

| 任务 | 状态 |
|------|------|
| 接口/数据路径核对 | 完成 |
| 入口脚本实现 | 完成 |
| Review（重映射完整性/权重继承/labels-mask） | 完成 |
| dry_run 自检 | 通过 |
| 小模型冒烟 diffusion / bert / diffusion+LoRA | 通过 |
| LoRA「先 resize 再 peft」修复 | 完成并验证 |
| 真实 8B 权重正式训练 | 未开始（需下载权重 + 多卡 FSDP） |

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

- [ ] 真实 8B 正式训练（下载权重、确认 FSDP 配置、监控 loss）。
- [ ] 确认 `LLaDA-8B-Base` 的 `weight_tying`（若为 True，safetensors 保存需处理共享张量；小模型冒烟已用非绑定配置绕开）。
- [ ] 数据规模：`ImmuneCsvDataset` 会把整表读入内存（train.csv GB 级），大规模训练需评估内存或改流式。
- [x] bert 经典 80/10/10（已完成）
- [ ] 可选：切换为保留 attention_mask。
- [ ] 序列含 `-`（gap）时未在重映射表内，会被当脏样本跳过；如需保留需补映射。

## 8. 变更日志

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
    - **重新提交**：取消 `t-20260803060836-j7z9x`，用更新后的 YAML 重交 → 新 task id **`t-20260803061533-q8r98`**（queue `q-20260511205647-m6t62`），已进入 **Running**。
- **2026-08-03（BERT 版本提交）**：融合路径此前只有扩散加噪（`sample_bioseq_diffusion_noise`：每序列 `t~U(eps,1)`、选中位 100% 换 `<mask>`）。新增 BERT-MLM 目标，二者共用同一 eligible 集合与 `compute_masked_cross_entropy`（token 级均匀 CE），仅腐蚀策略不同：
    - **`protein_fusion_model.py` 新增 `sample_bioseq_bert_noise`**：固定比例 `mask_ratio=0.15` 选中，选中位按 80/10/10 → `<mask>` / 随机残基 token / 保留原样；labels 落在**全部**选中位。随机替换从 LLaDA 空间的 `<res_*>` id 池抽样（新增非持久 buffer `_residue_token_ids`）。
    - **encoder 镜像**：BERT 的 `corruption_mask` 传全部选中位（含 10% 随机/10% 保留），即在 ESMC 侧对所有选中残基打 `<mask>`，防止冻结 encoder 通过条件路径泄漏待预测残基。
    - **模型 + 入口开关**：`LLaDAEsmcFusion` 增 `train_objective∈{diffusion,bert}` 与 `bert_mask_ratio/mask_prob/random_prob/residue_token_ids`；`compute_loss` 按目标分支。入口 `TrainingArguments` 增 `--train_objective/--bert_*`，并从 tokenizer 算出 `<res_*>` id 传入。
    - **验证**（tiny LLaDA + 真实 ESMC，400 次采样统计）：选中/eligible≈0.152（目标 0.15）、选中内 mask/random/keep≈0.799/0.098/0.103（目标 .80/.10/.10）、随机替换全在残基词表内、labels 恰好落在选中位；三 mode（token/feature/add）bert fwd+bwd 全通过。零 lint。
    - **提交**：新增 `train_jobs/protein_esmc_llada8b_add_bert.yml`（其余超参与 diffusion 版一致，`spot-share-queue`/`Preemptible:false`/1×8 卡）→ task id **`t-20260803062133-blhfm`**（queue `q-20260511205647-m6t62`），当前 **Queue**。diffusion 版 `t-20260803061533-q8r98` 已 **Running**。
