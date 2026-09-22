# Project Process

> Last updated: 2026-09-22
>
> 本页保留历史任务账本；顶部最新条目描述当前代码清理和文档同步状态。除明确标注为“本轮已验证”的项目外，历史测试、吞吐和任务数字不能被解释为本轮验证通过。
>
> **训练（2026-09-17 cancel）**：`t-20260912150714-lwf28` → **Killed**（End=`2026-09-17T08:13:43Z`）；BERT 闲时 `t-20260912035657-dd8v9` → **Killed**（End=`2026-09-17T09:37:24Z`）。未 cancel 评测 / FABind。

> **157000 矩阵（未启动，2026-09-17）**：用户选定 v5 **157000**（pairing 拟 p0 / CFG=0 / **iter=16**）后，源目录 `checkpoint-157000` 已被 top-k 删除（2026-09-17 07:12:46，训练写入 158000 时 prune）。无评测快照、无 SHA、无 submit/cancel、未改用 158000。训练 `t-20260912150714-lwf28` 未动。140000 RESULTS 主行保持。
>
> **当前 140000 矩阵（已完成，2026-09-17）**：独立评测快照已锁定；数字已写入 [RESULTS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。49000/92000 主行未覆盖。pairing 仅 p0/CFG0，CDR 为 iter=2，Specificity 为 100-epoch。本轮文档回填，无 submit/cancel。
>
> **92000 矩阵（历史对照）**：非 pairing 产物仍保留在 RESULTS。AB pairing 49000 修复后 p3 行仍在主表；92000 pairing 不入主表。
>
> **49000 矩阵终态（对照）**：指定 v5 49000 的 AB 11/11 与 TCR 10/10 已完成并通过各自结果核对。结果见 [RESULTS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。

## 2026-09-22 v2 无 ESMC 对照配置（代码配置已纳入提交，训练未提交）

- 用户要求增加与 v2 相同的无 ESMC encoder 版本并延长训练；新增 [decoder-only YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m.yml)。沿用 2 节点 × 4 GPU、c20250601 非抢占、immune_v6_binding_only、decoder d768/L8/h12 scratch、generated-only diffusion、global batch 256、LR 1e-4、warmup 2,000；建议预算设为 1,000,000 steps（5 倍），cosine horizon 同步延长。55 天 runtime cap 为上限，不是耗时预测；步数建议尚未提交执行。
- `token` 新训练不加载 ESMC 权重，模型和 checkpoint 均无 encoder/projection/norm 参数；只保留同一词表 tokenizer 资产。仍使用固定 decoder 167/135 槽和 EOS 监督，旧融合模型不受影响。代码同时放开 v2 token policy 并补齐 checkpoint 恢复；拒绝跨 token/add 或 encoder-bearing→encoder-free 直接 resume。
- 独立任务、W&B run 和输出目录，不与融合版本共用。旧 token checkpoint 仍可评估加载；旧 token 训练恢复到新结构需显式处理，不能直接续跑。
- 首轮 pllm 验证：5 个 v2 测试文件 **128 passed**（含新增 44 项），覆盖 EOS loss/backward、CFG 采样、无 encoder 特征依赖、真实 fusion 类 FP32/BF16 checkpoint roundtrip。YAML 解析、bash 语法、参数名及 2×4/1M/token 开关检查通过。
- 用户追问后的复核：专项 **131 passed**。新增真实 `LLaDAModelLM`（CPU、d32/L2）前向及仅 EOS 位置 loss 的反向传播测试，确认 backbone 获得有限非零梯度，state_dict 全为 decoder 参数。新测试最初的 swiglu 与 llama block 不兼容，已将测试配置改为 silu；未修改生产 LLaDA 实现。这里不是 d768/L8 或分布式训练实测。
- 复核发现 no-ESMC YAML 的 `--eval_steps` 后有空行截断续行命令（`bash -n` 无法检测）；已删除空行，并新增两份 YAML 的实际 Shell 参数捕获测试（用函数代替 accelerate，不执行训练），确认 output_dir/save_top_k 等末尾参数与 2×4、mode、steps 一并传入。
- 全量 `scripts/tests/immune_llada/` 为 **453 passed / 5 failed**：3 条已知 parity（KeyError prepared / StopIteration），2 条 profiler 的 Accelerate 全局状态冲突；后两条单独运行 **2 passed**。未修改无关测试，未运行真实大模型/GPU/FSDP 训练，未 submit/cancel，无新任务 ID。
- 更正配置说明：`max_eval_rows_per_source` 是当前入口不读取的兼容字段，实际评估使用完整 prepared valid split；新 YAML 不再写这个无效旗标，但评估行为与融合版相同。
- `train_jobs/` 受仓库既有 `.gitignore` 规则忽略；两份新 YAML 将通过强制添加纳入本轮代码提交，未改忽略规则。训练任务本身仍未 submit。

## 2026-09-22 v2 两节点四卡训练配置（代码配置已纳入提交，训练未提交）

- 按用户要求新增 [训练 YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_esmc_llada270m_diffusion_v2_immune_v6_2node4gpu.yml)：2 × `ml.pni2.14xlarge`，每节点 4 GPU，总计 8 GPU；沿用最近主训练的 `c20250601` 非抢占队列。多节点 Accelerate/FSDP 使用平台注入的 rank、master 地址与端口。
- 启用 `fixed_receptor_lengths=True`、`residue_cond_mode=add`，使用 `immune_v6_binding_only`；ESMC 可训练，decoder d768/L8/h12 从零初始化。每卡 batch 4 × 累积 8 × 8 GPU，丢弃超长行前 global batch 256；200,000 steps，LR 1e-4，cosine，warmup 2,000。
- 独立任务名和输出目录 `protein_esmc_llada270m_diffusion_v2_immune_v6_2node4gpu`，只恢复该目录内完整 checkpoint（含 fusion_config.json），不续跑旧 v5。当前长度定义仍为包含 CLS 的 encoder 168/136，未修改模型协议。
- 验证：pllm 环境 YAML 解析、Entrypoint 的 `bash -n`、2×4 资源配置和 v2 开关检查通过；prepared 数据 train/valid 目录存在。未运行跨节点/GPU 训练，未 submit/cancel，无新 task ID。下文 9 月 21 日“尚无 YAML”为历史状态。

## 2026-09-21 变长生成 v2：固定 encoder 画布 + 变长 decoder

**代码状态：已实现、已提交并 push 到 `origin/main`、已跑通 CPU 端到端 smoke 训练；尚未做真实规模训练。** 合同细节见
[VARIABLE_LENGTH_GENERATION_V2.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/VARIABLE_LENGTH_GENERATION_V2.md)。
改动前的基线快照是 `c234b07`（"Pre-variable-length-generation snapshot"），**已 push**，可随时整体回退。
v2 两笔提交也已在远端：`cf36477`（变长生成合同）→ `955a1ec`（超长行丢弃并计数，当前 `origin/main` HEAD）。
全部 v2 行为由单一开关 `--fixed_receptor_lengths` 控制，默认 `False`，legacy checkpoint 路径逐字不变。

**测试环境（2026-09-21 复测）**：`/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr`
（Python 3.12.12 / torch 2.8.0 / pytest 7.4.4 / nltk 3.9.4）。第一次跑测试用的不是这个环境，缺 `nltk`。

已落地的七条（与用户 2026-09-21 需求一一对应）：

1. 基线 `c234b07` 已 push（`git ls-remote origin main` 核对一致）。走的是本机已有代理
   `127.0.0.1:18089`，没用到 10808。
2. 下游 light-chain pairing 不再需要 ref 长度或采样先验：`--light-length-mode` 默认改为
   `auto`，v2 checkpoint 解析成 `fixed_v2`，legacy 仍走 `reference`；显式选错组合直接报错。
3. 固定长度：重链 / β 的 ESMC 画布 `168`（含 `<cls>`），轻链 / α `136`。**TCR 复用抗体的这套长度**，
   不再用 TCR 自己统计的长度。decoder 槽位 = 画布 − 1，即 `167` / `135`。
4. 每条链按"真实残基 + 重复 EOS 补到定长"渲染，EOS 是注意力可见 token 而非 padding，
   由 `chain_eos_mask` 单独标记，并在 `predict_eos=True` 时作为正常的 diffusion/MLM
   监督目标——模型要自己学"什么时候结束"。留一个 EOS 槽，所以实际最长残基数是 `166` / `134`。
   **LLaDA decoder 确实新加了 token**：本地 LLaDA tokenizer 的原生 `<|endoftext|>`(126081)
   同时被当成 EOS 和 PAD（实测 `eos_token_id == pad_token_id`），直接复用会和 padding 混淆，
   所以 v2 新增 `<chain_eos>` 并 `resize_token_embeddings`；policy 与 id 写进
   `fusion_config.json`，legacy checkpoint 保留旧的 alias 行为不动。
5. ESMC feature **不再替换** `wte`，改为相加：`inputs_embeds = wte(x_t) + condition_proj(ESMC) * mask`。
   旧的替换语义降级为 `residue_cond_mode="feature"`，只给 legacy checkpoint 用；
   `fixed_receptor_lengths=True` 时强制 `add`，构造期就 assert。
   重链 / 轻链之间的 `<chainsep>` split token 保持不变（实测抗体对布局：
   6 个前缀 special + 重链 167 槽 + `<chainsep>` + 轻链 135 槽 + `<protd>` = 310 token）。
6. 泄漏审查：训练期被 corrupt 的 decoder 槽在送进 ESMC 前一律镜像成 encoder MASK；
   推理期未 commit 的槽保持 MASK，只有已 commit 的生成残基才写回 ESMC 流。
   forward 里 `fixed_receptor_lengths` 分支把 `residue_mask` / `encoder_residue_mask`
   换成纯布局推导的 `chain_slot_mask` / `encoder_slot_mask`——干净的 AA/EOS mask 是 label，
   不能当成"隐藏长度"的条件信息。encoder attention mask 在定长链上恒为全 1（实测 `[168, 136]`），
   不会从 attention 形状里泄长度。ref 后缀在任何模式下都不渲染。
7. 定长画布装不下的行：**用户 2026-09-21 拍板「丢弃并计数」，已实现**，见下一节。

### 定长画布装不下的行：丢弃并计数

只读全量审计脚本
[audit_v2_canvas_data.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/debug/audit_v2_canvas_data.py)
对 `immune_v6_binding_only`（7,381,499 行）与 `immune_v3_heterotypic`（7,997,971 行）
逐行跑完，报告在 `logs/v2_canvas_data_audit/audit.json`（`status=complete`，
`code_unchanged_during_scan=true`，`renderer_mapping_issue_rows=0`，即角色映射本身没问题）。
两类行渲染不出来：

- **超长受体链**：v6 里 5 行（oas 1 / ots 2 / tcr_repertoire 1 / trait 1）残基数超过定长画布，
  最极端的是一条 186 残基的 `antibody_heavy` 和一条 148 残基的 `tcr_alpha`。
- **grammar 总长超 1024**：v6 里 train 4,289 行 + valid 286 行超过 `--max_length 1024`，
  **全部在 `asd_antibody`**（长抗原 + 现在被补到 167/135 的受体画布）；v3_heterotypic 对应
  5,008 + 2,033 行。根因是 prepared shards 的 `budget.max_length` 过滤是按**旧的变长**
  grammar 估的，定长画布把长度抬上去了，旧过滤不再成立。
- 合计 v6 4,580 / 7,381,499（0.062%），v3_heterotypic 7,044 / 7,997,971（0.088%）。

renderer 是在 **collator 里**跑的、原先没有跳过机制，这两类行一旦被采到就会中断训练。
现在 `GrammarBioSeqCollator(drop_overflow_records=True)` 丢弃并计数，
`protein_pretrain_esmc.py` 只在 `--fixed_receptor_lengths` 打开时启用它，legacy 路径仍然报错。

- 计数落在 `collator.drop_counts` / `drop_counts_by_source`，键为
  `canvas.receptor_overflow` 和 `budget.max_length`——后者**特意复用**预处理的过滤名，
  方便和 `filter_report.json` 对照。`CanvasDropLogCallback` 把它们写进 Trainer 日志的
  `canvas_drop/*`，collator 自己按指数退避打 warning。计数是**每进程**的、只覆盖该 rank
  实际读到的 shard，是监控信号而不是全语料过滤报告。
- 整批全被丢弃时**直接报错**，不让空 batch 静默变成 no-op（按实测比例，batch=4 时约 1e-13/批）。
- 超长受体链单独定义了 `ReceptorCanvasOverflow(ValueError)`，避免靠字符串匹配区分错误类型。
- ⚠️ 被丢的 4,289 条 over-budget 行占 `asd_antibody` train 的 **1.9%**（不是全语料的 0.06%），
  丢的是最长抗原那一尾。抗原条件生成在超长抗原上的指标值得盯一下。

本轮实测（2026-09-21）：

- **`protenix_abtcr` 复测（用户指定环境）**：`scripts/tests/immune_llada/` 全量
  **411 passed / 3 failed**（`logs/v2_abtcr_pytest/`，25s，CPU）。v2 专项三文件
  **84 passed**。剩下 3 条全部是 `test_full_parity.py`（`KeyError: 'prepared'` /
  `StopIteration`），是预处理审计工具的既有问题，不是 v2 回归。
- 第一次跑测试不在这个环境里，当时是 **394 passed / 20 failed**：多出来的 17 条是缺
  `nltk`（`test_tcr_generation_result_io` / `test_tcr_scoring_audit`）。`protenix_abtcr`
  自带 nltk 3.9.4，这 17 条已过。用 `git stash` 退回基线后，parity 那 3 条同样失败。
- 画布布局冒烟：抗体 / TCR 成对样本都得到 `encoder_input_ids=[B,2,168]`、
  每行槽位 `{heavy:167, light:135}`、`<chainsep>` 唯一、encoder attention 在定长链上全 1。
- 丢弃逻辑在真实语料行上验证：把审计点名的三行（`oas-00017:8523` 186 残基重链、
  `ots-00008:69012` 148 残基 α 链、`asd_antibody-00000:1695` grammar 1029）拼进一个
  223 行的 smoke 语料，开丢弃则保留 220 行、账本为
  `{canvas.receptor_overflow: 2, budget.max_length: 1}` 且来源归属正确；关丢弃则恰好 3 个 batch 报错。
- **端到端 CPU smoke 训练跑通**（`logs/v2_fixed_canvas_smoke/`，`rc=0`）：缩小 decoder
  （d256/L2/h4）在七个源上跑 12 步，`--fixed_receptor_lengths True --residue_cond_mode add
  --freeze_encoder False`。loss 4.22 → 约 3.1，grad_norm 有限，七个源的 per-source eval 都出数，
  6/12/final 三个 checkpoint 都写出。`fusion_config.json` 里
  `fixed_receptor_lengths=true`、`predict_eos=true`、`residue_cond_mode="add"`、
  `decoder_chain_eos_policy="chain_eos"`、`decoder_chain_eos_token_id=126389`
  （与 pad/eos `126081`、mask `126336` 都不同）、六个画布常数齐全。
- 该 checkpoint 经 `load_fusion_for_eval` 重新加载后策略无损，
  `resolve_light_length_mode(model, "auto")` 返回 `fixed_v2`，collator 给出
  `[1,2,168]` + `encoder_slot_mask`，对 135 个轻链槽位全量去噪后**输出变长链**
  （抗体 fixture 18 残基、TCR fixture 2 残基），由模型自己生成的 EOS 截断。
  12 步的序列内容当然是噪声，这里要证的是**长度由模型决定，不再来自 ref 或先验**。
- 本地 A100 用不了：torch `2.11.0+cu130` 要求的驱动比机器上的 `535.129.03`（CUDA 12.2）新，
  所以 smoke 只能走 CPU；vendored 的 `LLaDAModelLM.__init__` 还硬编码了
  `init_device = "cuda"`，在临时 harness 里 monkey-patch 掉，**没有改仓库代码**。

其他未做项：

- 尚未创建 v2 的 train job yml；v2 没有真实规模（d768/L8）的 GPU 训练和 loss 曲线。
- `tcr_generation_v5`（CDR3-only）与 `tcr_generation` 的 CDR3-only 模式对 v2 checkpoint
  **主动报错拒绝**——这两条下游评测在 v2 下暂不可用，需要改成 full-length 协议。
- ESMC 原生 prediction-head 辅助 loss 仍是 follow-up，未实现。
- `downstream/grammar/tcr_generation.py` 原先被 `.gitignore` 的 `/downstream/grammar/*`
  规则挡在版本控制外，但它带着 v2 改动、又被 v2 测试 import，本轮已加进 allowlist 并提交。
- ⚠️ 根盘（overlay 20G）本轮一度写到 98%：第一次 smoke 把 `--output_dir` 放在 `/tmp`，
  最后一次 save 报 `No space left on device`。已清掉并把 smoke 产物挪到
  `conda/cache/tmp/v2_smoke`。这正是 2026-09-19 记过的 TMPDIR 坑，再记一次。

## 2026-09-19 ASD 去污染对称化与 immune_v6 语料重建

- 问题：v5 run 的 `eval_asd_antibody_loss` 从 0.6359@43k 涨到 0.7413@159k，其余六源只动 +0.001~+0.02。定位到两条原因：切分两侧去污染不对称（去污染名单读的是切分后的 `step6_final/train.csv`，构成循环依赖，见 [decontam_extra.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/tcr_native/decontam_extra.py) 第 52 行），以及非结合样本被当成条件生成的目标。**ASD 没有做序列去重**——它是深度突变扫描库，去重等于删监督信号；按 0.90 对重链聚类会把 buzz 的 89k 行压成 31 簇。
- 四项改动：抗原长度下限 20 残基（abbd 里那条 14 残基肽 `PDVDLGDISGINAS` 带来的 17,585 行被划出，该短肽是上游原生数据，不是本项目切碎的）；标签不可判定的行直接丢弃（`bool` 解析失败、`fuzzy` 未知分箱、未识别 `affinity_type`、`ic_50` 一律不再猜成 binding）；标签冲突的三元组整组丢弃（原先保留 binding）；切分重建改为**先去污染 → 再删非结合 → 再整簇装箱 90/5/5**，打破循环依赖，脚本 [step6_symmetric.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/asd/scripts/step6_symmetric.py)。
- 行数流水（抗体库）：1,147,962 →（长度下限）1,122,776 →（不可判定）1,056,039 →（冲突整组）926,084 →（去污染）343,620 →（删非结合 88,954）254,666 → train 229,008 / valid 12,842 / holdout 12,816。纳米抗体 128,395 →（去污染 6 行）128,389 → train 115,593 / valid 6,417 / holdout 6,379，本来全是结合数据，binding-only 删 0 行。
- 删掉的非结合行按来源：buzz 60,250 / covid-19 12,656 / met 12,420 / hiv 3,628。TCR 三源（trait / tcr_native / tcr_papers）不重建语料，由配置里 `recipe.drop_nonbinding_sources` 在加载时删，train 侧合计约 29.6 万行；`tcr_repertoire` 的 210 万行标签是 unknown，不等于非结合，不动。
- 验证（走真实 loader 与 `build_filters`，未手工复刻键）：去污染名单在 train/valid/holdout 三侧命中率均 0.00%，旧语料对照为 67.23% / 2.99% / 3.12%；三侧 relation 全为 binding，只剩 `budget.max_length` 的正常丢弃。冗余顺带改善：最大那条抗原占比 32.33% → 20.91%，比之前的配额脚本（23.27%）更干净，[step7_cap.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/asd/scripts/step7_cap.py) 大概率不再需要。
- 产物：[step6_symmetric](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/asd/step6_symmetric)（抗体 / 纳米抗体各三侧 csv + stats.json + 聚类缓存）、两份对称去污染名单 `asd_{antibody,nanobody}_symmetric_blocklist.txt`（519,555 / 3 键）、新数据配置 [immune_v6_binding_only.yaml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/data/immune_v6_binding_only.yaml)。与 v5 配置只差三处：ASD 路径、两个名单路径、新增 `recipe` 段。**`step6_final/` 和 v5 配置一行未动**，v5 训练语料可原样回退；备份在 `downstream/asd/backup_20260919/`（613M）。
- prepared shards 已生成：[immune_v6_binding_only](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v6_binding_only)，13G / 78 个 shard / 7,381,499 行，`validation_report.json` 为 `passed`，与 v5 结构一致。manifest 里 `recipe.drop_nonbinding_sources` 字段已写入。加载时删掉的非结合行：tcr_papers 200,457 / tcr_native 43,742 / trait 20,681，合计 264,880；asd_antibody 删 0（语料重建时已删）。四个目标源的 `labels.relation` 在 train/valid 两侧**全为 binding**（逐行读 shard 核对）；tcr_repertoire 保持 210 万行 `unknown` 未动。与 v5 的 `filter_report.json` 逐源对照：`filter_reasons` 除新增 `recipe.binding_only` 一项外**逐位相同**，oas / ots / tcr_repertoire 保留行数 delta 为 **+0**，证明过滤器没有泄到不应作用的源上。ASD 三侧去污染命中为 0，只剩 `budget.max_length` 的正常丢弃（train 3,095 / valid 291）。
- 交接时留的 `tcr_papers` 疑点已消：小量 dry run 里该源 `dropped_filters=0`，原因是文件是七个语料拼接的、前缀 100% 是 `tcr_peptide`（已知偏置）；全量跑出现 198,134 行被删，符合预期，不是 bug。
- 已告知并接受的代价：`relation` 参与 loss，删负样本后 `<nonbinding>` 学不到，[relation 判别基准](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_relation_token.py) 失效；过滤同时作用于 valid 且切分重建，**`eval_loss` 与历史 run 不可横向比**（四件事一起做，只断一次）；patents 占比 26.7% → 37.2%，它是全语料唯一 `confidence=medium`（来自专利声称）且零负样本，等于用冗余下降换标签质量稀释，用户确认不再分层。建议评测按 dataset 拆开报。
- ASD valid 从 44,866 降到 12,551（语料重建所致，非过滤器），再确认一次 `eval_asd_antibody_loss` 与历史 run 不可横向比。
- ⚠️ **现阶段只用常规抗体库，纳米抗体不进训练**：上面报的纳米抗体行数（train 115,593 / valid 6,417 / holdout 6,379）是**已构建但未被消费**的产物——`immune_v6_binding_only.yaml`（以及前身 v5）的源列表里只有 `asd_antibody`，没有 `asd_nanobody`，prepared shards 的七个源里也没有它。配置里 `asd_nanobody_benchmark` 那行 blocklist 路径会被读取但永不生效（`build_filters` 只在 `source == asd_nanobody` 时查它）。本轮已在 [asd/README.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/asd/README.md) 顶部加了同样的提示，因为那份文档每张表都把两库并列报，容易误认为两边都在训练里。
- 本地长任务约定（用户要求，2026-09-19）：一律进 tmux，通用 runner [run_in_tmux.sh](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_in_tmux.sh)。用法 `bash scripts/run_in_tmux.sh <任务名> <命令...>`，日志落 `logs/<任务名>/run.log`，带 `started/done/failed` + `exit_code` 哨兵、`flock` 单实例互斥、`remain-on-exit` 保留 pane、`stdbuf -oL` 解开块缓冲。已冒烟验过正常/失败/同名拒绝/flock 四条路径。本轮 shards 是 `nohup` 跑的（已完成且核对通过，不重跑），后续重跑与本地评测走 runner。
- ⚠️ TMPDIR 坑：真实值是 `conda/cache/tmp`，**`/vepfs-mlp2/c20250601/251105016/tmp` 并不存在**（本轮启动预处理时误传了后者）。`TMPDIR` 指向不存在的目录时 Python 的 `tempfile` **静默回退到 `/tmp`、不报错**，而 `/` 是 overlay（20G，长期 95% 满，仅剩 1.2G），写量大的任务会把根盘写爆。本次未出事（产物 13G 完整、`validation_report` 为 `passed`、行数与 `filter_report` 对得上），所以不重跑；runner 里已改为 `mkdir -p` 后再断言存在。
- 未做：新 train job yml 尚未创建（需从 [v5 8gpu spot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_esmc_llada270m_diffusion_immune_v5_8gpu_spot.yml) 复制并改 `PREPARED_DATA_DIR`，不改 v5 的）；重建后的训练与评测未跑，曲线是否走平尚未验证。

## 2026-09-17 v5 159000 TCR 评测

- 最新（15:56Z）：159000 全部 AB/TCR 最终汇总已写入 [RESULTS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)，并完成产物身份、数量、折完整性、部分聚合与文档核对；不等于重新运行所有逐预测评分。范围与SHA记录：[文档核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_159000_20260917/results_document_audit.json)。历史“待整理”状态已由本条取代，无新任务操作。

> 最新：159000 AB 六项、TCR 十项均已执行完成；完整独立结果审计待做。详见本文末“159000 AB/TCR 执行完成核对”，下面提交与运行状态为历史快照。

- 12:56:19Z 复核：T1 四轨 Running，T4 benchmark14 / held20 Queue；均 c20250601 非闲时单卡。四项 T2/T3 未提交云端，避免重复输出。本地 tmux=`tcr-159000-local`，pane PID=3326686，12:55:57Z 启动 T2A，后续顺序 T2B → T3-broad → T3-deep；状态看 [local_repr.log](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_159000_20260917/local_repr.log) 与各任务 run_manifest.json，done 仅表示执行完成待结果验收。
- 81 项 CPU 测试和真实 GPU 三类 gate 已通过：T1 四轨 batch/order consistency，T2/T3 四轨 batch/order consistency，T4 隐藏 target 扰动/固定 context/正式32步采样路径/评分JSON写读。数据、代码及权重身份锁定在 gate 中，运行时核对；不是质量分数，也不是去污染证明。[机器可读总索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_159000_20260917/all_159000_tcr_submissions.json)。

- submit：`t-20260917205616-jwmqq`，`tcr-v5-159000-20260917-t4-held20`；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_159000_20260917_t4-held20.yml)，c20250601、Preemptible=false、单卡；已提交，待查询状态。

- submit：`t-20260917205613-lr9cf`，`tcr-v5-159000-20260917-t4-benchmark14`；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_159000_20260917_t4-benchmark14.yml)，c20250601、Preemptible=false、单卡；已提交，待查询状态。

- submit：`t-20260917205541-gnj99`，`tcr-v5-159000-20260917-t1-others-longab`；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_159000_20260917_t1-others_longab.yml)，c20250601、Preemptible=false、单卡；已提交，待查询状态。

- submit：`t-20260917205538-r2ldn`，`tcr-v5-159000-20260917-t1-others-cdr3b`；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_159000_20260917_t1-others_cdr3b.yml)，c20250601、Preemptible=false、单卡；已提交，待查询状态。

- submit：`t-20260917205534-878fc`，`tcr-v5-159000-20260917-t1-cdr3ab`；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_159000_20260917_t1-cdr3ab.yml)，c20250601、Preemptible=false、单卡；已提交，待查询状态。

- submit：`t-20260917205531-55dxp`，`tcr-v5-159000-20260917-t1-cdr3b`；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_159000_20260917_t1-cdr3b.yml)，c20250601、Preemptible=false、单卡；已提交，待查询状态。

- 用户授权补齐 TCR 十项，可优先本地。复用 AB 留存的 159000 快照及 SHA，不新选权重、不重复提交 AB。T1 四轨与 T4 两数据集准备非闲时单卡云作业；T2A/T2B/T3-broad/T3-deep 在本机 A100 串行，避免平台排队；不停止 occupy.py 或看护程序。
- 81 项 CPU 测试通过。正在执行真实 GPU 输入预检（与 140000 的 CPU identity-only 记录区分），通过后才执行正式任务；不修改既有输入/评分协议或主结果。
- 输出根：[159000 TCR](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_159000_20260917)；本地入口：[run_local_repr.sh](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_159000_20260917/run_local_repr.sh)。任务 ID 与本地状态在本节更新。

## 2026-09-17 v5 159000 AB 评测

- 12:42Z 六项均提交成功：pairing 16步 / CDR Kong / Specificity 为 Running，CDR SAb23 / GDPa1 / m396 为 Staging。38 项 CPU 测试、真实 native GPU 预检、4步真实 ESMC 反馈与隐藏参考替换 gate 均通过；预检不是正式成绩。本机预检已结束，正式评测均在云端。完整任务索引：[job_plan.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917/job_plan.json)。

- submit：`t-20260917204150-2kzlj`，m396；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_159000_full_20260917_m396.yml)，Preemptible=false，单卡；提交成功、待查询状态。

- submit：`t-20260917204147-ztdwf`，gdp_a1；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_159000_full_20260917_gdp_a1.yml)，Preemptible=false，单卡；提交成功、待查询状态。

- submit：`t-20260917204143-6b7vr`，specificity；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_159000_full_20260917_specificity.yml)，Preemptible=false，单卡；提交成功、待查询状态。

- submit：`t-20260917204140-gqlsz`，cdr-sab23；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_159000_full_20260917_cdr-sab23.yml)，Preemptible=false，单卡；提交成功、待查询状态。

- submit：`t-20260917204115-pj2rk`，pair-p0-cfg0；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_159000_full_20260917_pair-p0-cfg0.yml)，Preemptible=false，单卡；提交成功、待查询状态。

- 用户要求测试 159000，并明确 pairing 为 **16 步**；本轮限 AB：pairing p0/CFG0/reference-length/iter16/500×8，CDR Kong 与 SAb23H2 argmax/iter2，Specificity 五折固定 100 epoch，GDPa1 参考 CV，m396 六档主协议。pairing 与 140000 的 124 步不同，不能仅归因于权重；不提交 TCR、p3 或 CFG 网格。124 步 pairing 从未提交。
- submit：`t-20260917203938-l9f72`，cdr-kong，2026-09-17T12:39:38Z；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_159000_full_20260917_cdr-kong.yml)，Preemptible=false，单卡；现已 Running。
- 已留存独立快照 [159000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_159000_full_20260917)，源与副本 SHA256 一致：`01b3d1446b21f56b3a3658528a405cb34dee7dc889eb1f162e423d64141faf3e`；源 trainer_state.global_step=159000。未复制优化器，源权重及历史评测保留。
- 准备六项 c20250601 非闲时单卡任务；38 项 CPU 测试通过（protenix_abtcr 实际运行环境）。本地执行小规模真实 GPU 预检，非质量结果；正式提交状态在下方更新。不修改训练、BERT 看护或 occupy.py；RESULTS 未填未完成分数。
- 产物：[159000 AB](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917)。通用 prepare 产生的额外 p3/CFG YAML 不在本轮提交范围。

## 2026-09-17 Cancelled v5 8-GPU training

- 操作：cancel。用户要求停掉 Running 的 8 卡 diffusion，不 cancel 评测 / FABind / 无关队列。
- `task_id=t-20260912150714-lwf28`，任务名 `protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601`
- YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601.yml`
- 最终状态：**Killed**，Start=`2026-09-12T07:07:14Z`，End=`2026-09-17T08:13:43Z`，`Preemptible: false`
- BERT 闲时 `t-20260912035657-dd8v9` 查询仍为 **Queue**（未 Running），未 cancel。
- 盘上 checkpoint：37000 / 43000 / 92000 / **159000**（完整 resumable，`global_step=159000`）。**158000 已不在**。未启动评测或 158000 下游矩阵。

## 2026-09-17 Stopped: v5 157000 weights already pruned

- 操作：无 submit / 无 cancel。用户明确要求对 **checkpoint-157000** 跑完整 AB+TCR 矩阵（pairing 仅 p0 / CFG=0 / **`--max-iter 16`**，CDR iter=2）。到达时源目录已不存在。
- 源路径 `dllm_test/output/protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601/checkpoint-157000`：No such file。`ab_eval_checkpoints/` 无 `ab_v5_157000_*`。未创建 `ab_v5_157000_full_20260917` / `tcr_v5_157000_20260917`。
- 训练日志：2026-09-17 06:27:11 仍 keep 157000（`eval_loss=0.790879`）；2026-09-17 07:12:46 prune 157000，keep latest `checkpoint-158000`（`eval_loss=0.790139`，`trainer_state.global_step=158000`）。topk 现为 37000 / 43000 / 92000 + resume 158000。
- 未替换 158000 或其他 step；未动训练 `t-20260912150714-lwf28`；未改 RESULTS 主表数字；无本机 pid、无新 Volc task id。

## 2026-09-17 RESULTS 写入 accepted v5 140000

- 操作：只读回填文档。从 AB/TCR 产物 JSON 与聚合日志写入 [RESULTS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)；49000/92000 历史行未改。无 submit/cancel，未动训练 `t-20260912150714-lwf28`。
- 快照 [ab_v5_140000_llada_20260917](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_140000_llada_20260917)，SHA256=`a482afa425ef4f058d88e38d316aca096341af471b504efe948434d3ebc2dfe2`。

## 2026-09-17 Submitted Ours v5 140000 full AB/TCR matrix

- 操作：submit + 本机串行。用户要求对 v5 **140000** 按 AB/TCR 审计清单重跑，协议改为 pairing **仅 p0/CFG0**、CDR **iter=2**。未 cancel 训练 `t-20260912150714-lwf28`，未覆盖 49000/92000 产物或 RESULTS 主行。
- 锁定独立快照 [ab_v5_140000_llada_20260917](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_140000_llada_20260917)，权重 SHA256=`a482afa425ef4f058d88e38d316aca096341af471b504efe948434d3ebc2dfe2`，与源 `checkpoint-140000` 一致；`trainer_state.global_step=140000`。未复制优化器。
- 提交前 CPU 113 passed（pairing / sampling / AB probes / T1–T4）。TCR 侧为 CPU identity gate（非 GPU batch-consistency）。机器可读索引：[all_140000_matrix_submissions.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/all_140000_matrix_submissions.json)。
- 17:58Z 复核：pairing / CDR Kong / CDR SAb23 / T1 cdr3b / T1 cdr3ab / T1 others_cdr3b 为 **Running**；T1 others_longab / T4×2 为 **Queue**。本机 Specificity 已启动。完整 ID 与 YAML 见文末同名章节。

## 2026-09-17 无前缀 pairing 六档步数扫描完成验收

- 用户授权新增六档步数对照；准确协议、输入和验收见 [AB pairing §4.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。固定49000修复后p0，六档完整500×8，124步旧产物只作参照；不换模型、不动其他任务。
- 六项非闲时单卡现均Success，最晚于2026-09-16T21:40:00Z结束。2026-09-17重新只读核验全部24,000候选、六份真实反馈gate、原124步参照、各率分母和summary记录的指标／CSV／manifest哈希均通过；冻结源码／数据／原参考产物无漂移。提交前45项CPU回归为历史已通过，本轮未重新采样、评分或跑CPU suite。
- [实验目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_pairing_iters_p0_20260916)已保存完整产物，[汇总](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_pairing_iters_p0_20260916/summary.json)为complete=true、pending=[]、errors={}。全部档作为独立诊断写入[RESULTS §0.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-pairing)，不按扫描最优值更新p3主行。六项已移出两张Active表；没有操作或重查140000及其他任务，其状态保持原快照。

## 2026-09-16 本地 SAb23H2 多步 CDR 对照完成

- 用户明确授权本地测试多个iter。准确协议、验收及入口见 [AB CDR §4.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md)。先跑完整SAb23H2六CDR、iter1/2/4/8，固定92000留存快照与原batch1/argmax，Kong不在本轮启动范围；不按测试集最优步数替换主行。
- 37项CPU回归通过；未修改共享生成器、训练、checkpoint、旧预测或指标。新增独立runner保存逐条预测、SHA、实时状态、真实encoder输入审计，iter1须复现旧结果。
- 本机A100-80GB，原`occupy.py`与其他tmux不动；tmux=`ab-cdr-92000-iters-20260916`，pane PID=`2535199`、Python PID=`2535233`，Python启动=`2026-09-16T07:58:12Z`，完成=`2026-09-16T08:02:12Z`，进程已退出。含加载与审计共240秒，PyTorch峰值分配显存约2.77GiB。本地任务不进入Volc Active表，无submit/cancel。[status.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_cdr_iter_20260916/status.json)为complete，[日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_cdr_iter_20260916.log)及逐条产物已保留。
- 1,440条预测与24格指标独立验收通过，六模式iter1精确复现旧日志；真实模型反馈／隐藏参考扰动检查通过，细节只维护于[任务 §4.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md)。完整数值作为诊断写入 [RESULTS §0.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr)，原主表与其他任务产物未改。

## 2026-09-16 ESMC 反馈修复与 AB pairing 重评

- 实施与协议 owner：[AB pairing §7 k](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。生成／采样两份 CPU suite 28 passed；已添加真实权重逐轮输入门禁及独立单卡非闲时 YAML，准备 p0/p3 两组 500×8 重跑。固定既有 49000 评测快照，未改权重、训练或历史生成／指标；未重跑其他下游任务。
- 17:25Z 查询 p0/p3 均 Success，完整生成与评分产物已验收；终态耗时见本页末尾 dated 记录，已从两张 Active 表移除。只将有前缀修复后结果写入 RESULTS，p0 保留诊断产物；本轮只读核验与文档同步，无新推理、submit/cancel，未改其他任务。

## 2026-09-16 RESULTS 写入 92000 非 pairing 主行

- 按用户决定：pairing 先不管；其余已完成 92000 数字写入 [RESULTS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。49000 同行保留对照。AB pairing 六组与 §0.6b 消融仍用 49000；未写入不完整 92000 pairing。
- 数字来自各任务产物 JSON/聚合日志，不是 49000 comparison 的重算；本轮没有训练、推理、submit 或 cancel。

## 2026-09-15 本地 T1 recognition token 直接评分

- 用户授权本地测试；复用独立留存 v5 49000，未修改其他本地/云端任务。进程 PID `2002254` 已 exit 0，本地 A100 单卡、batch 8；完整 others β-only AS 测试，10,312 个独立输入，评分耗时 258.66 秒（不含加载）。入口及协议见 [TCR T1 §4.1a](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md)。状态：完成；55 项输入协议 CPU 测试通过，逐行数量/有限分数/重新计算 AUROC/脚本哈希/独立集跨折 SHA 均验证通过。按用户决定不写入 `RESULTS.md` 正式结果；产物仅保留于 [relation-token](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_relation_token/ours_v5_49000_20260915_others_cdr3b) 供诊断。未跑其余三条输入轨。

## 2026-09-15 AB / TCR 结果文档重排与 TCR 最终收口同步

- 按用户要求将 [RESULTS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md) 改为上半部分 AB、下半部分 TCR；逐任务集中 baseline、明确 checkpoint 的 Ours 结果及必要协议限制。AB prompt/CFG、历史长度/sampler 和 Specificity 训练预算诊断分层呈现，未把失效多步产物当成 v5 消融。
- TCR [最终 comparison](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/comparison_v1/final_49000_vs_baselines_and_29000.json) 已确认 complete=true、missing_tasks=[]；补齐遗漏的大数据 T1 β-only，移除顶部过期运行状态。当前矩阵仍固定 49000；29k 仅作已有 others 历史对照，AB 92000 仅用于已完成的 Specificity。
- 整理前 RESULTS 全文保存在 [RESULTS_ARCHIVE_20260915.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_ARCHIVE_20260915.md)，原正文完整保留并加归档警示。已冻结任务、旧排队过程和重复/失效表格仅移出当前主文档，不删除预测、指标、日志、权重或历史审计。
- 本次为文档整理及只读复核：最终比较所登记 139 个来源文件 SHA 全部一致；AB 当前采样/Specificity 指标从既有 metrics 读取。没有新训练/模型推理、没有 submit/cancel，Active 表及其他运行任务不动。数字仅维护在 RESULTS，关键决定记入 TCR 审计 §9.28。
- 文档验收：主文档 459 行、60 个唯一链接均可解析且为绝对路径，表格列数及页内锚点检查通过；归档去除新增警示头后与整理前完整正文一致。另逐项检查 T1/T3/GDPa1/m396 共 56 个当前显示值与结果 JSON 一致，9 个 AB CDR 格与各自聚合日志一致；git diff --check 通过。未把这些文档检查称为新一次模型评测。

## 2026-09-15 Foundation eval macro mean 与 antigen→AB 调研

- 已将多来源 `eval_loss` 从样本数加权改为来源等权均值；没有排除来源。实现及兼容边界见 [模型方案](/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md#2026-09-15-foundation-eval-改为来源等权均值)。
- 本轮 CPU 回归 7 passed；未加载模型权重、未跑真实数据评估、未修改 conda 依赖、未提交/取消/重启训练任务，Active 表不变。已有 checkpoint 和历史 loss 未重算。
- Antigen 条件生成调研已留档：[研究记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/ANTIGEN_CONDITIONED_AB_RESEARCH.md)。未实施新的下游任务或改排行榜。

## 2026-09-15 Cancel queue012 v5 8-GPU；提交 FABind 2×4

- **操作**：用户授权停掉 queue012 上的 v5 8 卡 diffusion，把调度位给 FABind 两节点四卡复现。
- **cancel**：`t-20260912021346-5xq4s`（`protein_esmc_llada270m_diffusion_immune_v5_8gpu`）→ **Killed**。不在闲时看护 TARGETS，未放 STOP。c20250601 的 `t-20260912150714-lwf28` 与 BERT 闲时 `t-20260912035657-dd8v9` 未动。
- **submit**：`t-20260915103955-twdv5` / `fabind-paper-2node4gpu`，queue012 非闲时，2 × `ml.pni2.14xlarge`，初始 Queue，Creator `zhuyiheng`。YAML：`/vepfs-mlp2/c20250601/251105016/project/FABind/repro/train_jobs/fabind_paper_2node4gpu.yml`。账本：[`FABind/REPRODUCTION_PROCESS.md`](/vepfs-mlp2/c20250601/251105016/project/FABind/REPRODUCTION_PROCESS.md)。
- FABind 不是 LLaDA 线，不列入上方 Active 训练表；本条只记录对 `5xq4s` 的 cancel。

## 2026-09-15 AB 分类头核查与论文baseline展示

- Specificity原始头类/实际调用、启动脚本与本轮保存头已核对，结构/宽度/选模证据唯一见 [AB Native Probes §4.4](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md#44-specificity-分类头对齐核查2026-09-15)。CPU读取原类AST，在768/2560维共享同一state_dict时前向逐位一致；读取fold0头确认768维与fixed_last=100。先激活pllm后使用既有protenix_abtcr，无基础模型加载或重训。
- 用户要求的baseline展示规则登记于 [PROJGUIDE §0.2.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md)。从本地论文PDF第17–18页核对Table 1–5，当前AB结果各节补论文参考，Figure 4缺少精确点值保持明确缺失；Ours及历史复跑原值不改。
- 本轮仅核查及文档同步，没有生产代码、权重、数据或评测产物变更，无submit/cancel，Active不变。

## 2026-09-15 AB GDPa1 reference-only 协议同步

- 按用户决定更新 GDPa1 入口及协议 owner，详见 [AB Native Probes §4.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md#42-预测头与协议切割线)；[RESULTS §0.6a](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md) 调整当前/历史展示，不改既有数值或原始产物。
- 按要求先激活 pllm，再切既有 protenix_abtcr：`python -m pytest scripts/tests/bioseq/test_ab_native_probes.py -q -p no:cacheprovider`，8 passed（8.28s）。覆盖参考 GridSearchCV 的 alpha/折均值/OOF一致性、仅一次选参搜索、缺失标签行映射、不生成 nested 产物及历史文件不变，并回归 Specificity、m396、原生输入与单卡矩阵。
- 未加载模型权重、重提特征、启动全量评测或 submit/cancel；Active 表及其他任务不变，无新增模型分数。

## 2026-09-13 T4 现成配对测试数据核查

- 已用pllm进行只读CSV全行解析及官方solutions字段对照，核对作者生成/评分代码、现有benchmark14交集和本地数据SHA。来源、方法与结果边界见 [T4 §7.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)，日志见 [审计 §9.20](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。仅研究与文档/身份清单更新，无生产代码、数据副本、权重、模型分数或submit/cancel变化，Active表不变。

## 2026-09-13 T4 双链 baseline 研究与分组登记

- 论文/官方代码、相关本地权重存在性和配对数据候选表头核对完成；方法清单、尚未修复的适配问题和执行门禁见 [T4 §7.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)，用户决定见 [审计 §9.19](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。本轮只有研究与文档同步，无生产/数据变更、权重下载或重新SHA核验、GPU运行及submit/cancel；没有新模型成绩，Active表不变。

## 2026-09-13 benchmark14 格式与 TCRT5 输出范围核对

- 对照论文公开正文、官方代码及本地CSV/JSON解释用户问题；结论与字段实例见 [T4 §7.4.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)，关键理解见 [审计 §9.18](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。只读核查与文档补充，无生产改动、数据写入、GPU运行或submit/cancel，Active不变，无新性能指标。

## 2026-09-13 T4 区域语义确认与源数据字段核查

- 用户已确认此前的生成区域选择；具体协议与上游/评测输入的字段区分见 [T4 §7.4](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)，关键日志见 [审计 §9.17](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。该选择不再是待用户澄清项，但生成实现及任务专属gate仍未完成。
- 激活pllm后进行只读CSV全行扫描，检查全部评测JSON字段并复核官方源文件SHA；身份清单和文档已同步。未修改生产代码、数据、权重或旧指标，没有GPU运行及submit/cancel，Active表不变。

## 2026-09-13 TCR 49000 提交前复核（未提交）

- 用户本轮范围、仅最新 checkpoint 汇总要求与逐任务未关闭 gate 见 [TCR 审计 §9.16](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。当前不满足“所有任务确认无问题”的前置条件；没有启动正式 TCR 评测，Active 表不变，也未操作既有 AB/训练作业。
- 可用模型为既有独立 [49000 快照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913)，本轮重新读取全部 `model.safetensors` 得 SHA256=`d5f892bce7d18106a7c417a3e4605f2bf1feac2bfa80709ca74fb149e5c07d92`，与快照 manifest 和原复制记录相符；trainer_state.global_step=49000。该目录虽用 AB 命名，保存的是同一完整 fusion checkpoint，不是 AB 微调权重；此轮仅只读核验，未覆盖或复制快照。
- 快照 manifest 所指原训练路径 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601/checkpoint-49000` 在 21:34Z 检查时已不存在；不推断删除原因，不改用其他 checkpoint。现有快照权重、tokenizer、trainer state 仍在，后续执行应锁定已验证快照而非失效原路径。
- CPU 复跑：激活 pllm，执行 `python -m pytest -q`，测试文件为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_tcr_binding_query.py`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_tcr_binding_others_beta.py`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_tcr_binding_long.py`，**55 passed in 5.49s**。这是输入/缓存等 CPU 回归，不是 49000 TCR 真实 GPU 验收或性能结果。
- 本轮只有只读核验、既有 CPU 测试与文档登记；没有 submit/cancel、训练数据/生产代码/权重变更、新模型指标或旧结果覆盖。

## 2026-09-13 T4 联合 αβ 生成提议核对

- 用户方向及待确认的X/MASK、目标区域边界见 [T4 §7.4](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) / [审计 §9.15](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。本轮仅静态核对/文档，无生产改动、GPU运行或submit/cancel，Active不变。

## 2026-09-13 T4 对 v5 train / valid 的全量重叠核查

- 已按实际启动参数和prepared manifest完成只读全量核查，结果、复跑入口与代码/数据身份见 [T4 §7.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)，关键日志见 [审计 §9.14](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。最终v2完整性门禁和独立产物检查均通过，初版未通过报告不作为结论。
- 新增CPU诊断脚本、报告和文档；没有训练数据、blocklist、生产生成代码、权重修改，没有submit/cancel或模型新分数，Active表不变。

## 2026-09-13 T4 baseline 生成方式审计

- 研究结果及条件问题见 [T4 §7.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) / [审计 §9.13](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)，RESULTS 同步引用限制；仅文档更新，没有生产代码/数据改动、权重加载、submit/cancel或新增指标，Active不变。

## 2026-09-13 T4 当前实现与结果状态初查

- 当前 T4 输入/目标/采样路径、数据条目及部分旧 metrics 已只读核对；详细发现见 [T4 §7.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)，关键日志见 [TCR 审计 §9.12](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。RESULTS §0.4 同步版本与指标解释边界，原数字保留。
- 仅文档/静态诊断，未改生产代码或数据、未加载权重、提交或取消任务；Active 表不变，无新模型性能。不将此次初查称为 v5 或完整生成审计验收。

## 2026-09-13 Ours AB 单卡矩阵（09-14复核：11项全部Success）

最新状态与完整结果已收口，见末尾2026-09-14 AB终态章节及RESULTS对应三节；下方“已提交/尚无完整CSV”保留为09-13历史提交快照。

- 用户授权在我们的模型上评测 AB，并增加 pairing prompt0/prompt3 × CFG；非闲时、非抢占、独立单卡。实现/CPU 验收和方法边界见 [原生 AB 任务](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md) 与 [pairing §4.4](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。
- 配置生成器 [prepare_ab_native_jobs.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/prepare_ab_native_jobs.py) 规划6 pairing +2 CDR数据集 +3原生探针，均 c20250601 / ml.pni2.3xlarge / 1 replica / Preemptible:false。
- 用户已指定使用最新 v5；权重选择、独立快照 SHA、GPU 预检和正式任务记录见本文件末尾 `Submitted Ours AB v5 49000` 系列条目。真实模型 GPU gate 已通过，11项正式单卡任务已全部提交；不混用旧42000或后续训练步数。正式 pairing 已产生独立 manifest / progress 文件，尚无完整 CSV 与评分。
- 本轮仅改评测适配、测试及文档；不修改训练或已有 baseline/生成结果。相关30项CPU测试通过，不能据此声称完整评测已完成。
- 三份真实数据的原生输入预检已通过（无模型加载），均在1024 token上限内；pllm 的生成相关测试23 passed、原生探针文件因缺依赖明确skip，protenix_abtcr 中全部30项通过。仅对3个新增 downstream 文件加入精确gitignore例外，不放开数据/模型/产物。

## 2026-09-13 v5 训练 checkpoint 状态复核（19:26–19:28 UTC）

- 操作：只读查询当前三条 v5 任务的 `ml_task get`，读取两条 diffusion 的 worker 日志、checkpoint 文件列表、`trainer_state.json` 与 `topk_val_manifest.json`；没有 submit/cancel、改参、复制权重或启动评测。
- `t-20260912150714-lwf28`（c20250601，8 卡非闲时）：**Running**，日志已到约 **47,900 / 200,000**；最新已保存 [checkpoint-47000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601/checkpoint-47000)，trainer state=47000，最新 eval_loss=0.764063。当前保留 checkpoint 中验证 loss 最低为 **43000 / 0.747695**，依据 [top-k manifest](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601/topk_val_manifest.json)。
- `t-20260912021346-5xq4s`（queue012，8 卡非闲时）：**Running**，日志已到约 **38,400 / 200,000**；最新已保存 [checkpoint-38000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune_v5_8gpu/checkpoint-38000)，trainer state=38000，最新 eval_loss=0.757379。当前保留 checkpoint 中验证 loss 最低为 **37000 / 0.750480**，依据 [top-k manifest](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune_v5_8gpu/topk_val_manifest.json)。
- `t-20260912035657-dd8v9`（BERT，c20250601，8 卡闲时）：仍为 **Queue**；`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_bert_immune_v5_8gpu_spot/` 当前仅见 wandb，没有 checkpoint 目录。
- 最新两个 diffusion checkpoint 都有 `model.safetensors`、tokenizer、trainer state 和续训状态文件；本轮未加载权重做推理。“日志训练步数”“最新保存”“验证 loss 最优”分别报告，不相互替代，验证 loss 最优也不等于 pairing 最优。训练仍持续写入，以上为查询时快照。
- 刚完成的 AB pairing 对应旧 `protein_esmc_llada270m_diffusion_immune/checkpoint-42000`，**不是上述 v5**；不能把旧模型下游分数当作 v5 当前水平。下游权威数字仍见 RESULTS §0.6。

## 2026-09-13 T3 论文 baseline 实验复核

- 后续文档整理：按用户要求将关键理解及使用准则直接加入 [TCR 总审计 §9.11.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)，详细证据仍由任务 §7.2 维护。仅文档更新，未执行新实验。

- 完成原文/作者 notebook/模型卡核对、本地源数据来源匹配与 artifact 字段只读检查；详细协议及失配见 [T3 §7.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md)，关键记录见 [总审计 §9.11](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。引用状态已同步至 RESULTS / README / BASELINE_VERIFICATION，旧模型分数未改。
- 本轮未提交/取消任务，未加载权重、改生产代码/原始数据或执行模型重评；Active 表不变。论文对齐和纠正后 v5 T3 均仍待实施/验收，不能以此次研究作为新实验完成凭据。

## 2026-09-13 T3 representation / few-shot 专项初查

- 详细发现、CPU 验证与数据边界见 [T3 任务 §7.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md)，关键记录与身份见 [TCR 审计 §9.10](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。完成诊断与文档同步；未加载模型、修改生产代码/数据、提交或取消任务，Active 表不变，无新性能指标。T3 v5 适配、真实模型与其余 gate 仍待做，T4 尚未专项验证。

## 2026-09-13 T2 clustering 输入/关系审计

- 按用户要求开始审计；完成静态接线与真实 tokenizer/原生 collator 的 CPU 标签边界检查（decoder 用替身），详细发现见 [T2 任务 §7.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T2_CLUSTERING.md)，关键身份与后续事项见 [TCR 审计 §9.9](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。
- 仅诊断与文档同步；未加载模型权重、修改生产 Python/训练 renderer、提交或取消作业；没有新性能结果，Active 表不变。T2 v5 运行时适配及真实模型重评仍待做，T3/T4 专项未完成。

## 2026-09-13 TCR T1–T4 测试准则确认

- 用户希望测试全部已构建 TCR 任务，且后续必须遵守指定审计；已登记任务文档清单和持续执行要求，见 [TCR 审计 §9.8](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)，并在项目指南/模型计划加入必读引用。
- 本轮仅只读核对文档入口与提交前 gate、固化用户要求；未跑模型测试、未提交或取消任务、未产生新指标，Active 表不变。T2–T4 专项输入/协议验收尚未完成，不把此前 T1 Success 当作全套完成。

## 2026-09-13 TCR T1 运行时输入纠错

- LongAB 同数据对照 **Success**：`t-20260914005618-vr2nb`，单卡非闲时，五折×三测试集及产物复核完成；实施/验证见 [TCR 审计 §9.7](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)，指标见 [RESULTS §0.1c](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)，提交/终态账本见末尾 `Submitted TCR v5 LongAB control on others`。

- v5 29k 同 others 数据的 β-only 对照 **Success**：`t-20260913202402-c9hf5`，五折×三测试集及逐行核验完成；协议/验收见 [TCR 审计 §9.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)，指标见 [RESULTS §0.1b](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。不影响旧 αβ 结果或其他任务。
- 用户最新授权在本地资源繁忙时使用单卡非闲时；新任务已 **Success**，独立重跑及产物复核完成，详见末尾 `Submitted TCR others nonspot rerun` 及 [TCR 审计 §9.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。旧闲时任务仍 Killed，本地占卡/FABind 进程未动；这是新授权下的独立提交，不是自动重提取消任务。
- 按用户要求再次同步关键改动登记与历史/当前状态区分，见 [TCR 审计 §9.2 文档复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。本次仅文档复核，未重新测试、提交或取消作业，Active 表不变。
- 实现、验证、数据轨道和验收边界统一见 [TCR baseline 审计 §9](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。初次 beta-only 修正仅做 CPU 验证；后续 others/cdr3ab GPU 五折已 **Success**，见本文件末尾 `Submitted TCR others CDR3ab v5` 账本。新指标只写 [RESULTS §0.1a](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)，不替换历史 β-only 榜。

## 2026-09-13 AB pairing / AirGen 对照诊断

- 全量评测已 **Success**，完整产物已核验；结果见 [RESULTS §0.6 全量](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)，协议验收及因果归因限制见 [任务 §7 h](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。本文件末尾保留该作业的提交、运行和终态时间线，下方为历史阶段记录。
- 用户已授权开始评测；新作业见本文件末尾 `Submitted reference-length pairing v3` 账本。实现/测试状态统一见 [任务 §7 h](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)；下方“仅文档”条目为本次实施前的历史记录。
- 按用户要求在 [AB 测评审计 §9](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/AB_BASELINE_EVALUATION_AUDIT.md) 登记新的长度协议决定，并同步任务手册、结果引用提示及入口文档；具体协议只维护在任务 §4.3。只修改 Markdown，未更改 CLI/wrapper/YAML、修复 sampler、启动测试或提交评测，既有分数保留。
- 追加 PAD attention 追问的静态核查，跟踪至 LLaDA attention bias，并核对 AirGen poas 训练 collator/噪声目标；方案限制见 [任务文档 §7 i/j 第 5–7 点](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。本次未运行新的 CPU 模型测试、训练或 GPU 评测，仅更新方法说明。
- 追加用户关于 padding、终止符与 CFG 的复核：[任务文档 §7 i/j](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。本轮实际 collator/mask/parser 的双样本 CPU 检查全部通过（退出码 0）；仅更新审计文档，未改训练/推理源码、运行模型评测或提交作业。
- 按用户要求核查官方 baseline 与 Ours 的生成、评分和历史产物；新发现的问题及唯一证据记录见 [AB pairing §7 h](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。
- 已在 pllm 环境用 AST 提取真实函数做四 token、两步解码的 CPU 最小诊断，退出码 0，约 4.9 秒；未加载模型、未用 GPU。结果证明状态差异，不证明模型分数变化。
- 按下游规则同步任务缺陷、相关旧结果引用限制与方法记录；未修改采样代码、覆盖结果、提交或取消作业。修复和真实模型受控重评尚未执行。

## 2026-09-13 Ophiuchus-Ab baseline 探针测评文档

- 按用户要求创建 [AB baseline 测评审计与完善指南](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/AB_BASELINE_EVALUATION_AUDIT.md)，作为三个冻结表征探针的纠错台账与后续验收入口；细节只在该文档维护。
- 已同步下游入口、范围说明旁的审计链接、结果引用警示、方法计划与项目指南；现有 headline 范围与旧数值不变。
- 本轮只核查源码/既有产物并修改文档，未修改测评实现、提交或取消作业，也未产生新模型指标；后续修复与运行尚待执行。
- 文档检查：本地引用目标全部存在；只读 jq 命令已验证末轮与历史记录两种聚合，确认原文件含五折、每折 100 轮；已通过所涉已跟踪文件的 diff 空白检查。

## 2026-09-13 TCR baseline 评测纠错文档

- 按用户要求创建 [TCR baseline 纠错记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)，集中登记代码证据、问题状态、修正方案与验收要求。
- 已同步下游任务入口、结果引用警示、方法计划和下游进展；本轮只做静态代码/官方来源审计，未修改训练或评测实现、未提交或取消作业、未产生新指标。

## 2026-09-12 Submitted v5 8-GPU immune diffusion

- **操作**：submit（from scratch；独立 `OUTPUT_DIR`，不接 v3 / v4 / 2M checkpoint）
- **task_id**：`t-20260912021346-5xq4s`
- **任务名**：`protein_esmc_llada270m_diffusion_immune_v5_8gpu`
- **YAML**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_esmc_llada270m_diffusion_immune_v5_8gpu.yml`
- **提交时刻**：`2026-09-11T18:13:46Z`（2026-09-12 02:13:46 UTC+8）
- **初始状态**：`Queue`，队列 `queue012`（`q-20260524172355-rnqtf`），1×8 卡 `ml.pni2.28xlarge`
- **Preemptible**：false
- **取消**：`bash scripts/volc-no-proxy.sh ml_task cancel -i t-20260912021346-5xq4s`
- **可复现 commit**：`b356b28`（新增 YAML）→ `8006151`（固定 200k step / 11 天 deadline）→ `6c753c8`（作业实际执行的 `examples/llada/protein_pretrain_esmc.py` 与 `protein_fusion_model.py`；此 commit 之前会跑到未提交工作区代码）
- **配置**：权威在 YAML 头部注释与 Entrypoint。固定 `--max_steps 200000`（不传 `--num_train_epochs`）、global batch 256 = `per_device 4 × ga 8 × 8 GPU`、cosine `1e-4` / warmup 2000 / `max_grad_norm 1.0`、`ActiveDeadlineSeconds 950400`、eval/save 每 1000、`save_top_k 3`、`slim_checkpoints True`。
- **数据**：`data/prepared/immune_v5_receptor_completion`。行数 / 监督份额 / 不变量：plan §4.2。受体补全设计：plan §2.5 / §2.6。
- **墙钟估计（2026-09-12 开跑前核算，v5 作业本身尚未起跑）**：权威数字与推导见
  `SPEED_ANALYSIS.md` 与 [`PROTEIN_PRETRAIN_PROGRESS.md`](examples/llada/PROTEIN_PRETRAIN_PROGRESS.md) §6.3 / §6.4 / §6.7。
  核算底稿（仓库外）：`/vepfs-mlp2/c20250601/251105016/project/_jobmon/DECISION_NUMBERS.md`。
  **最终决定：`--save_top_k 3`、`--eval_steps 1000`、`ActiveDeadlineSeconds 950400` 三者均保持不动，未 cancel、未重提交。**
- **开跑前核算（可复用教训）**：
  - **步时**：不能拿一次运行的 s/it 直接外推到另一个 `per_device` / `ga` 配置，必须先核对 `training_args.bin` 的三个因子（`per_device × ga × world_size`）。历史 v3 8 卡 `t-20260901033935-h55f4` / `t-20260904014842-qgjbg` 实测 **2.94–2.95 s/it**，bin 里是 `per_device=2`、`ga=16`、`world_size=8`、`max_length=1024`，全局 batch **256**。v5 同为全局 256，切法是 `per_device=4 × ga=8 × 8`。microbatch 放大次线性：`t(batch4)/t(batch2) ≈ 1.29`，故 v5 每步墙钟约为该 v3 的 **0.65×**。主证据：`..._v3_4gpu`（`per_device 4 × ga 8 × 4` 卡，单卡负载与 v5 相同）tqdm **1.51–1.91 s/it**。外推 v5 **1.6–2.5 s/it**（中枢约 2.2），训练 **4.0–5.8 天**。
  - **eval**：`eval_oas_runtime=4.4792` → OAS 约 **446.5 samples/sec**，来自 `..._v3_8gpu_2m/checkpoint-105000`，本身就是 8 卡，不用再做卡数归一。**不能用单源速率外推全量**：同一次 8 卡 run 里 `asd_antibody` 只有约 **114 samples/sec**（约 3.9× 慢），且占 v5 valid 的 44%。按源分别外推：v5 全量 valid **102,308** 行，单次全量 eval 约 **8.7 分钟**；`--eval_steps 1000` × 200 次合计约 **29 小时 ≈ 1.21 天**。
  - **checkpoint / slim**：`--slim_checkpoints True` 只作用于「被保留下来、但不是 latest」的目录；**latest 满包永不 slim**。实测 slim 后 **2.3 GB**（2,423,849,254 B）、latest 满包 **9.1 GB**（9,667,566,831 B）。`save_top_k=3` 稳态 **13.5–16 GB**（峰值约 25 GB）；`save_top_k=10` 约 **32 GB**；`save_top_k=0`（全留 200 个）约 **467 GB**。
  - **配额**：上次死因是容量配额，不是 inode（`t-20260901033935-h55f4` 在 step 17000 写 `model.safetensors` 时报 `Disk quota exceeded (os error 122)`，无 inode 字样；`df -i` inode 仅 5%）。配额上限查不到（`quota` / `lfs` / `mmlsquota` 本机都不存在）；`df` 今日仍有 702T，**`df` 不反映租户配额**。判断磁盘风险要用「上次撞墙水位」：当时 `output/` 约 **1.2 TB**；2026-09-12 `output/` 约 **466 GB**，距该水位约 734 GB。因此 `save_top_k=0` 再吃 467 GB → 约 933 GB、贴着死亡水位；`save_top_k=3` 只加 16–25 GB。
  - **`ActiveDeadlineSeconds`**：平台字段语义是 **MaxRuntime**，**从 Launch 起算，不含排队**。证据：`t-20260904014842-qgjbg` 排队 22.8 小时，其 tqdm 累计墙钟 88 小时对齐的是 Launch→End，不是 Create→End。本次 `950400`（11 天）对照预期总墙钟 5.2–7.0 天，余量充足；超时线在约 **4.15 s/it**。
- **读结果时注意**（本条新登记）：
  - `--max_eval_rows_per_source 2000` 在当前代码只是未消费的 CLI 字段，每次 eval 跑完整 valid（行数：plan §4.2；本 run 约 200 次满 eval）。
  - wandb 回退机制：`examples/llada/README.md`。历史 44 个 run 目录全是 `offline-run-*`、0 个 online、只有 5 个曾 `wandb sync`；计算节点探测 `api.wandb.ai` 失败则会 offline，盘上 `${OUTPUT_DIR}/wandb` 仍在 VePFS。
  - YAML 写了 `RetryOptions`（`EnableRetry` / `MaxRetryTimes: 5` / `IntervalSeconds: 180` / `PolicySets: [Failed]`）。已用 `ml_task export --task t-20260912021346-5xq4s --config` 实证平台接受了这段配置：导出文件第 64-70 行逐字包含上述字段，并额外补了默认 `EnableReserveResourceOnRetry: false`（证据：`/vepfs-mlp2/c20250601/251105016/project/_jobmon/t-20260912021346-5xq4s.yaml` 与 `_jobmon/retry_evidence/`）。手动重提时 entrypoint 的 `RESUME_ARG` 仍会从最新 checkpoint 续。语义见 `examples/llada/README.md`。
  - **`RoleRestartPolicy`（角色级）与 `RetryOptions`（作业级）是两层。** `ml_task get` 不回显 `RetryOptions` 属正常（`--helpformat` 可选字段列表里没有该字段）；`TaskRoleSpecs` 下的 `RoleRestartPolicy: "Never"` / `RoleRestartMaxRetryCount: 0` 是角色/Pod 级重启，和作业级 `RetryOptions` 不构成矛盾。判断平台是否接受了 `RetryOptions`，要用 `ml_task export --config`，不要用 `ml_task get`。
  - `save_top_k` 看合计 `eval_loss`。valid 里 `asd_antibody` 占比远高于 train（plan §4.2 逐源表），抗体–抗原会主导选模，TCR–epitope 不是主信号。checkpoint 保留另有调查；这里只登记构成错配与担心，不给修法。

## 2026-09-12 filter_report item 13 计数 + manifest `filter_names`（代码已落地；v5 产物未重写）

- 实现已提交 `95c04a96`。`filter_report.json` 现对每个 source 与 `totals` 写入
  `downgraded_all_x_mhc` / `beta_only_completed` / `alpha_only_completed`
  （kept-record transformation，与六个 first-failure drop 计数分开；
  `dropped_all_x_epitope` 不重复，仍走 `quality.blank_epitope`）。
  manifest `filter_names` 改为实际构造的 `RecordFilter` 之并集（不再是
  `BLOCKLIST_NAMES` 配置键）。旧缺陷与新语义的权威登记：`DATA_FORMAT_AUDIT.md`。
- **数字权威**：`docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md` §4.2
  （post-hoc，从已发布 JSONL 推导；含朴素 MHC-strip 启发式会错约 10 倍的陷阱）。
  v5 13G 未重跑，已发布 `filter_report` / manifest 仍是旧字段。不要在本页复述表。
- 验证：套件 **198 passed / 5 既有失败**（`test_full_parity` ×3、
  `test_profiling` autocast ×2）；v4 `tcr_repertoire` byte-equivalence 仍是
  **0 mismatches**。
- **仍未修、不要标完成**：(1) ingest `_drop_placeholder` 仍漏用在 `epitope` /
  `mhc_pseudo`（重建 `tcr_papers_v2/dataset/` 会重新放进 all-X；**当前最重要
  未修项**；权威：`examples/llada/DATA_PIPELINE_README.md` §6.0.1）；
  (2) `tcr_papers` 记录 identity 的 `source` 仍是 `"tcr_native"`
  （`metadata.dataset_source` 与 shard 名是对的）；(3) 已发布 v5 manifest /
  `filter_report` 仍是旧字段，只有下次 preprocess 才会写出新键。
- 未提交新的 Volc 训练任务。

## 2026-09-12 表位源受体补全 + all-X 处置（v5 已全量核验）

- 代码在 `dllm/pipelines/immune_llada/data/`（该目录从未纳入 git，不要用 commit 叙述）。
  操作入口：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/README.md`。
  布局：`DATA_FORMAT_AUDIT.md`。设计/风险：`docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md`。
- §2.5 / §2.6 **已在全量语料落地并核验**。产物
  `data/prepared/immune_v5_receptor_completion/`（13G）。逐源行数 / v4 对照 /
  监督 token 份额 / 不变量 / item 13 计数的权威表：plan §4.2。headline train
  **7,626,885**；唯一 drop 是 `quality.blank_epitope`（38,689 train + 411 valid，
  全部 `tcr_papers`）。
- 故意偏离计划：(a) 按实际 fv/CDR3 列分流，不按 `sequence_scope`（351 条
  `tcr_native` 标 `fv` 却只有 `beta_fv`）；(b) all-X epitope 用命名 filter 而非
  adapter `None`。§3.1 item 13 计数与 `filter_names` 硬编码缺陷：**同日稍后已修**
  （见上方；数字 plan §4.2；已发布 v5 产物未重写）。
- 验证：用 v4 冻结 profile 从 raw CSV 重生 3,000 条 `tcr_repertoire`，对盘上 shard
  **0 mismatch**；当时套件 194 passed / 5 既有无关失败（随后补测至 **198**，见上方）；
  v5 全量不变量见 plan §4.2。
- 仍未修、不要标完成：ingest `_drop_placeholder` 漏用（重建 `tcr_papers_v2` 会重新放进
  all-X）；`tcr_papers` 记录 identity 的 `source` 仍是 `"tcr_native"`。
  （`filter_names` 代码已修，不要再列在未修里。）
- 未提交新的 Volc 训练任务。v3 / `immune_v3_heterotypic` 仍在盘上，部分 checkpoint
  与 eval YAML 仍指向它们。

## 2026-09-11 旧 BioSeq 训练线清理后的文档同步（代码完成，验证待主 agent）

- 已继续完成本轮允许范围内的实际文档清理：旧 7L/Arrow/直读 CSV 叙述已降级为历史证据，当前训练/数据/速度边界统一指向 immune LLaDA prepared semantic JSONL 线；未改 `downstream/grammar` 核心实现。
- 已按当前工作树同步本轮允许范围内的八份文档：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJ_GUIDE.md`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/FUTURE_EXPERIMENTS.md`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/SPEED_ANALYSIS.md`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/README.md`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/README.md`。
- 当前代码事实按已完成清理记录：旧 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data` alias 整层和旧 `training` 已删除；`GRAMMAR_V1.md`、旧 BioSeq/llada 训练入口、`bioseq/datasets.py`、PPI/STRING/MINT builders、专属 tests/jobs 均已删除。`refactor_baseline` 审计快照及 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive` 历史证据保留。
- 当前免疫数据实现唯一为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada`，正式训练入口为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`；数据流为 raw → source adapter → `BioSeqRecord` → offline filter → prepared semantic JSONL → training-time grammar/padding/per-chain encoder reconstruction/masking。无 model-ready token cache。
- 既有验收数字是 **v3_heterotypic** 口径（raw **8,671,293**，kept **7,997,971**），不是 v4/v5。v4 行数见 plan §4.1；v5 见 plan §4.2。
- 本轮文档清理未重新运行全量 preprocessing、strict parity、真实 GPU/FSDP overlap、DataLoader 吞吐或模型回归；主 agent 待补验证位置：**[TODO: 主 agent 在本轮完成验证后补命令、退出码、耗时和 artifact 绝对路径]**。不得把“文档同步完成”写成“验证通过”。
- 下游保留典型当前入口：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_immune_fusion_gen.sh`、`run_immune_fusion_pairing.sh`、`run_immune_fusion_repr.sh`、`run_pairing_pll.sh`/`score_pairing_pll.py`，以及对应 benchmark。`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar` 仍被 fusion 使用，CDR、light pairing、TCR generation 的公共实现保留；本轮不改外部 baseline，也不在本文件重复另一 agent 负责的 downstream/eval README。

## 2026-09-10 Immune LLaDA 旧训练 loader 清理完成

- 已迁移 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/trait/scripts/validate_wiring.py`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/asd/scripts/validate_wiring.py` 和 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/ab_cdr_leakage.py` 到 `immune_llada.data` 的 source adapter/record 工具；这些脚本仍可读取各自下游原始数据做离线验证，但不再依赖旧 immune dataset。
- 已将 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_dynamic_training.py` 中的旧 `ImmuneCsvDataset` 文件测试替换为 OAS/OTS/nanobody source adapter 的 canonical `BioSeqRecord` 测试。
- 已删除退役入口 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain.py` 和旧实现 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/datasets.py`；`bioseq` 顶层不再导出 `ImmuneCsvDataset`、`ImmuneSourceSpec`、`build_mixed_immune_dataset`、`default_immune_specs`。
- 本条旧记录中的 qwen data 兼容 alias、PPI/STRING/MINT 训练共享模块及其专属入口随后已清理删除；仅保留 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/refactor_baseline` 审计快照、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive` 历史证据和明确列出的当前 model-layer/eval 公共实现。此处保留作为历史账本，不是当前依赖。

## 2026-09-09 用户确认离线丢弃 51 条同类型 paired 数据

- 已明确采用离线丢弃，不修正配对、不保留旧位置角色。`immune_llada/data/preprocessing/filters.py` 在 OAS/OTS 原有规则后增加 `quality.homotypic_pair`，仅按两条链角色相同判断，不按序列相同判断；不影响其他 source 的单链样本，不向训练时增加操作。新 manifest 记录该规则。
- 新增/调整 offline filter、prepared loader 和 strict audit 回归；合并 **104 passed in 24.70s**。首次新增全过滤 fixture 的测试因报告省略零值 `prepared` 键而失败，已修正测试读取零值方式，未弱化实际差异检查。
- 新数据已完成重建并通过策略验收：`data/prepared/immune_v3_heterotypic/`，7,997,971 kept / 673,322 filter drops；train 7,794,324、valid 203,647。旧 `immune_v3` 保留作回滚/对照；`protein_pretrain_esmc.py` 和 profiler 默认路径已切换到新目录。
- 复验报告：`refactor_baseline/homotypic_drop_20260909/policy_acceptance.json`，仅 OAS train 38、OTS train 13 条按批准规则丢弃；strict audit 退出码 2 是预期历史差异，不是策略验收失败。
- 其余必要收尾仍为 manifest freshness/version、旧 YAML/兼容参数清理、本地真实 GPU DataLoader 验收；详细说明与计划已同步。

## 2026-09-09 Immune 数据重构验收：全量差异已定位，尚未全部通过

- 新增独立历史实现的全量审计 `scripts/data/run_immune_full_parity.py`，固定 commit `7f6f2351702dc307f710bce7228a53eae391b26f`；不是兼容 alias 自比较，也不是仅比数量/集合。
- 后台完整检查已结束（1,368.99 秒）：8,671,293 raw / 7,998,022 kept；过滤决策与计数无差异，新 canonical vs prepared 完整行多重集无差异。**严格旧/新语义 parity failed，退出码 2**：OAS 38 + OTS 13 条 role 差异，对应 102 个不同指纹。
- 全部 51 条差异完成定向归因：原始标注为 H/H 或 β/β，旧 wrapper 按位置赋 H/L 或 β/α。38 条 OAS rendering 相同；**13 条 OTS 从 `tcr_pair` 变成 `tcr_single`，改变模型输入和 encoder masks**。不能用 reservoir 抽样的 0 mismatch 宣称全量 grammar 等价。未修改 prepared 数据，需确认离线处理或历史兼容策略。
- 证据：`refactor_baseline/full_parity_20260909/report.json`、`refactor_baseline/full_parity_diagnosis_20260909/run/`。诊断的 `passed` 只表示已复现差异，不覆盖全量 parity 的失败结论。
- 本地模型 profiler `scripts/debug/profile_immune_llada.py` 已实现，修正 BF16 autocast、核验 GPU UUID、全 tensor digest 和 artifact 命名。实际 GPU profiling 尚未运行：外部 compute PID 3074807 持续占用 A100，未干扰该进程；也没有启动正式 Volc 任务。
- 合并相关回归 **94 passed in 35.59s**，含 41 项 profiler CPU 测试及 H/H、β/β 审计检测回归；六个新增/修改 Python 验收脚本与测试 AST 检查通过。测试通过不等于数据语义差异已解决。
- 仍需收尾：完整 manifest freshness/version 契约、15 份 v3 YAML 和旧 DataArguments 兼容字段同步清理、本地 GPU 空闲后的真实 loader overlap 验证。prepared loader 仍需启动建索引、逐行 JSON decode，不能声称零数据成本。
- 验收说明：`docs/IMMUNE_LLADA_DATA_ACCEPTANCE.md`；计划和 pipeline README 已同步。本轮审计/诊断后台进程均已结束，无模型 profiling 子进程运行。

## 2026-09-09 Immune LLaDA 数据 pipeline 重构（全量验证）

- 新建独立数据模块：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/`，新的免疫数据逻辑不放在 `qwen3_vl_arch` 目录下；使用说明见该目录 `README.md`。
- 已完成 canonical `BioSeqRecord`、source registry、七个 source adapter、显式 filter API，以及流式 offline preprocessing CLI：`scripts/data/preprocess_immune_dataset.py`。
- 全量 prepared 数据已生成到 `data/prepared/immune_v3/`：8,671,293 raw rows，7,998,022 kept，673,271 filter drops，0 schema drops，0 errors；train 7,794,375、valid 203,647，共 88 个 JSONL shards。
- preprocessing 输出 semantic JSONL shards、`dataset_manifest.json`、`filter_report.json`、`validation_report.json` 和 `schema.json`；shard 和报告采用 atomic publish，`schema.json.token_cache=false`，未生成 model-ready token cache。
- `examples/llada/protein_pretrain_esmc.py` 已切换为只读取 prepared manifest；训练期仍保留 grammar rendering、batch padding、per-chain encoder input 重建和 diffusion/MLM 随机逻辑。旧 raw CSV/filter/坏样本重试逻辑不再进入训练热路径。
- 相关回归测试为 `40 passed in 4.84s`；compileall 通过；正式训练环境下全量 manifest dry-run 和 A100 1-step forward/backward/checkpoint save 均通过，loss=12.4375、grad_norm=30.625。
- 已完成 prepared loader + required collator 的本地吞吐基线：workers=0 为 1367.76 samples/s，workers=1 为 721.62，workers=2 为 1337.44；正式 GPU/FSDP worker overlap 仍需单独测量，不能据此武断固定 worker 数。
- parity baseline 已生成到 `refactor_baseline/immune_v3_baseline.json`，记录每个 split/source 的数量、首尾记录哈希和聚合 canonical record 哈希；新增 sampled adapter parity 报告 `refactor_baseline/immune_adapter_parity_train_5000.json` 与 `immune_adapter_parity_valid_5000.json`：train 35,000 行、valid 32,440 行全部匹配，无单边 drop、error 或 mismatch。该结果不替代全量 fingerprint/multiplicity parity；checkpoint restore smoke 已通过：full checkpoint `checkpoint-1` 成功恢复并继续到 `checkpoint-2`。没有启动正式 Volc 训练任务。
- 详细计划：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_LLADA_DATA_REFACTOR_PLAN.md`。

## 2026-09-08 存储清理：退役 `grammar_v2` 旧训练线，回收 379G

**背景**：`dllm_test/` 占 1.34TB，其中 `output/` 822G。审计发现 `output/` 里同时存着两条
训练线的 checkpoint，而 run 名字里的 `llada` 有两种含义，容易误判为同一条线：

| | `grammar_v2_*`（旧线，已删） | `protein_esmc_llada*`（当前线，保留） |
|---|---|---|
| 训练入口 | `examples/bioseq/train_qwen3_vl_bioseq_ddp.py` | `examples/llada/protein_pretrain_esmc.py` |
| 保存代码 | `dllm/pipelines/qwen3_vl_arch/training/checkpointing.py` | `FusionTrainer(transformers.Trainer)` |
| 落盘形态 | `step_NNNNNNN_val_X.XXXX.pt` / `best.pt` / `latest.pt` | `checkpoint-NNNNN/model.safetensors` |
| 模型 | ESMC-300M/600M + 从零训的小 LLaDA decoder（hidden 512、7 或 28 层） | ESMC + LLaDA-8B / 270m fusion |
| wandb project | `bioseq-qwen3-vl` | — |

`grammar_v2_*_llada` 里的 `llada` 只是 `args.model_type="llada"`（decoder 架构），
**与 `examples/llada` 训练脚本无关**。判据取自 `.pt` 内嵌 `args`
（`grammar_data_dir=data/bioseq_grammar_v1`、`save_top_k=5`、`hidden_size=512`）
与文件命名格式，二者精确对应 `qwen3_vl_arch/training/checkpointing.py`。

**决策依据**：`downstream/benchmark/RESULTS.md` 的 VOID 横幅（2026-08-15 立、08-20 复核）
已判定所有 `grammar_v2 / cmp500k / mint / integrated 7L` 数字作废，当前架构是
`examples/llada` immune fusion；9-07 的 PPT 也只提 270M/8B。用户确认与主训练线无关者即可删。

**已删除（合计 379G）**：

| 路径 | 回收 |
|---|---|
| `output/grammar_v2_esmc600m_cmp500k_llada` | 86G |
| `output/grammar_v2_esmc300m_mint_llada` | 67G |
| `output/grammar_v2_esmc300m_cmp500k_llada` | 59G |
| `output/grammar_v2_esmc300m_integrated_llada` | 50G |
| `output/grammar_v2_esmc300m_integrated_llada_8src_pre_b1` | 42G |
| `output/grammar_v2_esmc300m_integrated_llada_7l` + 10 个 `_stepNNNNNN` 快照目录 | 55G |
| `tmp/smoke_llada_esmc300m_cfg`（同旧线的 smoke 产物，`latest.pt`+`final.pt`） | 17G |
| `.models/bioseq/ophiuchus-ab-mixed/final.pt`（旧轻量 bioseq 后端自训权重，仅 `backbone_state_dict`，0 下游引用） | 3.1G |
| `_feature_cache/ours_bioseq7l_step389500_frozen_globalmean_mlp` | 1.7G |
| `output/grammar_v1_esmc300m`、`grammar_v2_*_killed_step988_*`（仅日志/wandb） | ~3M |

结果：`output/` 822G → **465G**，`tmp/` → 空，`.models/` → 空，`dllm_test/` 1.34TB → **约 955G**。

**留档**：`docs/archive/grammar_v2_retired_20260908/` —— 6 份 `topk_manifest.json`
（含每个 retained step 的 `val_loss`）、2 份 `wandb-summary.json`、
`deleted_file_inventory.tsv`（295 个文件的路径/字节数/mtime）。共 49K。

**不可复现声明**：训练语料 `data/bioseq_grammar_v1` 早已不在盘上，故这批 run
**删除前就已无法重训**。已记录的历史数字（`RESULTS.md` 历史节、
`outputs/tcr_beta_public_benchmark/`、`checkpoint_comparison/bioseq_step*/`）
保留可读，但不得再生成、再打分或写入新表。

**遗留待办（未处理，需决策）**：
- `paper/math_commands.tex` 与 `paper/sections/4_experiments.tex`（8-30 改过）仍把 headline
  `\OursCkpt` 定义为 `grammar_v2_esmc300m_integrated_llada_7l` step 16,000，该 ckpt 现已不存在。
  这与 `RESULTS.md` 的 VOID 横幅本就矛盾。论文若继续投，须整体改锚到 `examples/llada` 线
  （8B BERT `checkpoint-43000` / 8B diffusion `checkpoint-45000` / 270m 系列）。
- `PROJ_GUIDE.md` 中 `step117000` SHA256 钉子与 `gen_eval_grammar_v2_*` 生成器已就地标注
  RETIRED；约 40 个 smoke / `run_paper6_*` 脚本的默认 `--embedder grammar:*` 路径仍指向已删文件，
  未逐个清理（均为模块级 Path 常量，不触发 import 错误）。

## 2026-09-07 T4 / CDR 四个阶段改本地单卡跑（不再等 `queue012`）

**操作**：用户决定被 cancel 的 T4 与 CDR 四个阶段本地跑。本机有**一张空闲 A100-80G**
（`nvidia-smi` 0% / 4MiB，无其他进程），这四个阶段本来就是单卡 `ml.pni2.3xlarge` 作业，
本地跑与平台跑的计算完全等价 —— **pairing 仍留在 `queue012` 排队，不本地跑**
（要 2.7 小时，且两条 pairing 是本轮关键路径，占住本地卡会把 T4/CDR 也拖慢）。

Runner：`output/_local_runs/run_local_t4_cdr.sh`（`nohup` 后台，日志
`output/_local_runs/local_run.log`，每阶段耗时与退出码写 `output/_local_runs/status.tsv`）。
**四阶段串行**，因为只有一张卡；env 与 eval YAML 的 entrypoint 逐行对齐
（`protenix_abtcr` env、`CUDA_VISIBLE_DEVICES=0`、`HF_HUB_OFFLINE=1`、`CDR_MAX_ITER=2`、
batch `4 8`），故产物与平台跑出的完全同口径。

| 顺序 | 阶段 | ckpt | tag | 对应已 cancel 的平台任务 |
|---:|---|---|---|---|
| 1 | T4 | `eval_snapshot_151000` | `ours_fusion_v3_allchains_8gpu2m_151000` | `t-20260907044310-wlm9h` |
| 2 | T4 | `eval_snapshot_44000` | `ours_fusion_v3_genonly_8gpu2m_44000` | `t-20260907044324-tcmhs` |
| 3 | CDR | `eval_snapshot_151000` | `ours_fusion_v3_allchains_8gpu2m_151000` | `t-20260907044303-ks4x6` |
| 4 | CDR | `eval_snapshot_44000` | `ours_fusion_v3_genonly_8gpu2m_44000` | `t-20260907044317-kfhk2` |

预计合计约 1 小时（T4 12min×2 + CDR 16min×2 的历史中位数）。**21:28:38Z 起跑**，
第一条 T4 held20 已在出序列（7/20 epitope，GPU 60%）。产物仍写
`output/downstream_generation/<tag>_*`，与平台任务同路径。

⚠️ **本地跑与闲时旧九条有同路径写冲突风险**：`c20250601` 上未停的
`t-20260906233455-djlmk`（151000-t4）、`t-20260906233513-mpfv5`（44000-t4）、
`t-20260907001930-qtf2s`（151000-cdr）、`t-20260906233504-x2n8b`（44000-cdr）
一旦抢到闲时资源，会和本地跑写同一批文件。**控制台停旧九条这件事现在更紧迫了。**

BERT 105000 表征（`t-20260907044328-wwfjn`）与两条 repr 未纳入本次本地跑，用户未要求；
要补做同样可本地跑（`run_immune_fusion_repr.sh`，历史中位 28min）。

---

## 2026-09-07 只留两条 pairing，其余七条 cancel（`queue012` 单卡排不动）

**操作**：用户决定只保留 pairing。`queue012` 上那批非闲时九条排了 41 分钟仍 **9/9 `Queue`**，
按历史耗时中位数排序（T4 12min / CDR 16min / repr 28min / **pairing 165min**），pairing 是关键路径，
其余七条先让位。cancel 时七条**都还在 `Queue`**，未中断任何已起跑的计算。

| 保留（`queue012` 非闲时） | Task ID |
|---|---|
| `eval-v3-allchains-8gpu2m-151000-pairing` | `t-20260907044307-btjxr` |
| `eval-v3-genonly-8gpu2m-44000-pairing` | `t-20260907044321-ht8b6` |

已 cancel 七条（均 `cancel success`）：`t-20260907044259-qd52v`（151000-repr）、
`t-20260907044303-ks4x6`（151000-cdr）、`t-20260907044310-wlm9h`（151000-t4）、
`t-20260907044314-mtzn9`（44000-repr）、`t-20260907044317-kfhk2`（44000-cdr）、
`t-20260907044324-tcmhs`（44000-t4）、`t-20260907044328-wwfjn`（bert-105000-repr）。
这七条的 ckpt / tag 未变，要补做直接用原 YAML 重提即可。

**下游各阶段实测耗时（36 条历史 Success 的中位数，供以后排优先级用）**

| 阶段 | 中位 | 范围 | 解码步数 |
|---|---:|---|---|
| T4 生成 | 12 min | 11–28 | 脚本写死 32 |
| AB CDR | 16 min | 16–43 | `CDR_MAX_ITER=2` |
| 表征 T1/T2/T3 | 28 min | 27–55 | — |
| **AB light pairing** | **165 min** | 164–180 | `PAIR_MAX_ITER=124` |

pairing 慢一个数量级是解码步数差异所致，不是数据量。偏慢的离群值（repr 55min / CDR 43min）都来自
`4gpu-42000` 那批，与阶段无关。

🔴 **`c20250601` 闲时旧九条仍全在 `Queue`，其中两条 pairing 与保留的两条同名同产物前缀**
（`t-20260906233451-fjzmm` / `t-20260906233508-svn2t`）。本账号无权 cancel，**需控制台停掉**，
优先停这两条 pairing：跑满要 2.7 小时且是闲时资源，中途被抢即白跑，还会把新任务产物覆盖成半成品。

---

## 2026-09-07 九条单卡下游改投 `queue012` 非闲时重提（`c20250601` 权限已失）

**操作**：用户要求把仍未起跑的单卡采样/评测任务改成非闲时重提。九条 `eval-v3-*`（全链 151000 全套、
generated-only 44000 全套、BERT 1M 105000 表征）在 `c20250601` 闲时排了 **4.9 小时仍全是 `Queue`**，
零起跑。改 `Preemptible: false`，队列由 `c20250601` 改为 **`queue012`**，单卡 `ml.pni2.3xlarge` 不变。

**为什么连队列一起换**：本账号对 `c20250601`（`q-20260121145036-6fztt`）**已无 `CreateCustomTask` 权限**，
非闲时重提直接被 IAM 拒（`RequestID=202609070441150F35291B10C95F36E668` 等九条）。用一条探针任务
（`probe-perm-check-donotrun` → `t-20260907044147-jvzpn`，提交后立即 cancel 成功、已 `Killed`）确认
`queue012` 可提交可取消，故九条统一改投 `queue012`。

| Job | 新 Task ID（queue012 非闲时） | 旧 Task ID（c20250601 闲时，仍 Queue） |
|---|---|---|
| `eval-v3-allchains-8gpu2m-151000-repr` | `t-20260907044259-qd52v` | `t-20260906233442-d2gzt` |
| `eval-v3-allchains-8gpu2m-151000-cdr` | `t-20260907044303-ks4x6` | `t-20260907001930-qtf2s` |
| `eval-v3-allchains-8gpu2m-151000-pairing` | `t-20260907044307-btjxr` | `t-20260906233451-fjzmm` |
| `eval-v3-allchains-8gpu2m-151000-t4` | `t-20260907044310-wlm9h` | `t-20260906233455-djlmk` |
| `eval-v3-genonly-8gpu2m-44000-repr` | `t-20260907044314-mtzn9` | `t-20260906233500-9f7vp` |
| `eval-v3-genonly-8gpu2m-44000-cdr` | `t-20260907044317-kfhk2` | `t-20260906233504-x2n8b` |
| `eval-v3-genonly-8gpu2m-44000-pairing` | `t-20260907044321-ht8b6` | `t-20260906233508-svn2t` |
| `eval-v3-genonly-8gpu2m-44000-t4` | `t-20260907044324-tcmhs` | `t-20260906233513-mpfv5` |
| `eval-v3-bert-1m-105000-repr` | `t-20260907044328-wwfjn` | `t-20260906233517-9x8kq` |

九条新任务已 `ml_task get` 回读确认 `Preemptible=False` + `ResourceQueueId=q-20260524172355-rnqtf`
（`queue012`），初始状态均 `Queue`。账本
`output/downstream_generation/eval_v3_bestval_queue012_nonpreempt_20260907_task_ids.tsv`。

🔴 **旧九条 cancel 不掉，有双跑风险**：它们的 `Creator` 是 `251105016`，本账号（`zhuyiheng`）
`ml_task cancel` 报 `User is not authorized to perform: ml_platform:StopCustomTask`。同名任务与新九条
**共用同一 `output/downstream_generation/` 产物前缀**，若闲时那批之后抢到资源会与新批互相覆盖。
**需要到控制台手动停掉旧九条**（或用 `251105016` 账号 cancel）。

🔴 **同时发现主训练已停**：`protein_esmc_llada270m_diffusion_allchains_immune_v3_8gpu_2m`
`t-20260902021013-bm79q` 已于 **2026-09-06T19:37:05Z `Killed`**（日志末尾 `signal: 15`，非代码报错）。
盘上最新满包 `checkpoint-161000`。该条不在本次改动范围内，未重提，待决策。

⚠️ **20:54 重复提交九条，已全部 cancel，不要引用**：另一路会话未先查平台就按同样九个 YAML
重提了一遍 —— `t-20260907045438-fxzf8` / `t-20260907045450-qv244` / `t-20260907045454-wxclf` /
`t-20260907045457-hhtrg` / `t-20260907045500-mmlhg` / `t-20260907045504-jqlzl` /
`t-20260907045507-tgxm9` / `t-20260907045511-978hn` / `t-20260907045514-jx2rs`，
9/9 `cancel success`。**教训**：`eval_jobs/*.yml` 落盘或被改写**不等于**没提交过；
重提前必须先 `bash scripts/volc-no-proxy.sh ml_task list -n eval-v3 --limit 40 -o json`
查平台实况，按 `Creator` + `Start` 认清哪批是自己的。

---

## 2026-09-06 三条长跑最好 val 点下游（闲时单卡）

**操作**：评当前三条训练各自最低 `eval_loss` 点。全链 **151000 / 0.6326** 全套；generated-only 8 卡 2M **44000 / 0.7520** 全套；BERT 1M **105000 / 0.4263** 只 T1/T2/T3。**不打断**三条训练。权重硬链接 `eval_snapshot_{151000,44000,105000}/`（nlink=2）。队列 `c20250601`，单卡 `ml.pni2.3xlarge`，`Preemptible: true`。

| Job | Task ID | YAML | 覆盖 |
|---|---|---|---|
| `eval-v3-allchains-8gpu2m-151000-repr` | `t-20260906233442-d2gzt` | `eval_jobs/eval_v3_allchains_8gpu2m_151000_repr.yml` | T1/T2/T3 |
| `eval-v3-allchains-8gpu2m-151000-cdr` | `t-20260907001930-qtf2s`（前次 `t-20260906233447-rh2j9` Killed） | `eval_jobs/eval_v3_allchains_8gpu2m_151000_cdr.yml` | CDR |
| `eval-v3-allchains-8gpu2m-151000-pairing` | `t-20260906233451-fjzmm` | `eval_jobs/eval_v3_allchains_8gpu2m_151000_pairing.yml` | pairing |
| `eval-v3-allchains-8gpu2m-151000-t4` | `t-20260906233455-djlmk` | `eval_jobs/eval_v3_allchains_8gpu2m_151000_t4.yml` | T4 |
| `eval-v3-genonly-8gpu2m-44000-repr` | `t-20260906233500-9f7vp` | `eval_jobs/eval_v3_genonly_8gpu2m_44000_repr.yml` | T1/T2/T3 |
| `eval-v3-genonly-8gpu2m-44000-cdr` | `t-20260906233504-x2n8b` | `eval_jobs/eval_v3_genonly_8gpu2m_44000_cdr.yml` | CDR |
| `eval-v3-genonly-8gpu2m-44000-pairing` | `t-20260906233508-svn2t` | `eval_jobs/eval_v3_genonly_8gpu2m_44000_pairing.yml` | pairing |
| `eval-v3-genonly-8gpu2m-44000-t4` | `t-20260906233513-mpfv5` | `eval_jobs/eval_v3_genonly_8gpu2m_44000_t4.yml` | T4 |
| `eval-v3-bert-1m-105000-repr` | `t-20260906233517-9x8kq` | `eval_jobs/eval_v3_bert_1m_105000_repr.yml` | T1/T2/T3 only |

账本 `output/downstream_generation/eval_v3_bestval_151000_44000_105000_20260906_task_ids.tsv`。数字进 RESULTS §0.8，跑完再回填。三条 `eval_loss` 不可互比。

---

## 2026-09-04 补提 69000 T4（原 ckpt 已剪、闲时 Killed）

**操作**：盘点先前全链评测。`checkpoint-18000` / `checkpoint-26000` / `checkpoint-69000` 均已被 top-k 剪掉。18000 / 26000 四条下游产物齐全，无法也无需重跑。69000 表征/CDR/pairing 已 Success；**T4 被闲时抢走**（held20 只写了 2/20），权重仍在 `eval_snapshot_69000/`。用原 YAML 重提 T4，不打断续训。

| Job | Task ID | YAML | 初始状态 | Preemptible |
|---|---|---|---|---|
| `eval-v3-allchains-8gpu2m-69000-t4` | `t-20260904104404-t6j7v`（前次 `t-20260904013725-7zcvf` Killed） | `eval_jobs/eval_v3_allchains_8gpu2m_69000_t4.yml` | Queue | **true** |

账本 `output/downstream_generation/eval_v3_allchains_69000_t4_resubmit_20260904_task_ids.tsv`。

---

## 2026-09-04 全链 8 卡 2M `checkpoint-73000` 全套下游（抢占单卡）

**操作**：评当前 8 卡 2M top-k 最好 val 点 `checkpoint-73000`（eval **0.6535**；最新满包 83000 为 0.6567，未评）。覆盖与 69000 相同：T1/T2/T3 + AB CDR + pairing + T4。**不打断**续训 `t-20260902021013-bm79q`。权重硬链接到 `eval_snapshot_73000/`（nlink=2），避免 top-k 剪枝弄丢评测输入。队列 `c20250601`，单卡 `ml.pni2.3xlarge`，`Preemptible: true`。

| Job | Task ID | YAML | 初始状态 | Preemptible |
|---|---|---|---|---|
| `eval-v3-allchains-8gpu2m-73000-repr` | `t-20260904104145-77cfr` | `eval_jobs/eval_v3_allchains_8gpu2m_73000_repr.yml` | Queue | **true** |
| `eval-v3-allchains-8gpu2m-73000-cdr` | `t-20260904104148-6x5m9` | `eval_jobs/eval_v3_allchains_8gpu2m_73000_cdr.yml` | Queue | **true** |
| `eval-v3-allchains-8gpu2m-73000-pairing` | `t-20260904104152-pxjd4` | `eval_jobs/eval_v3_allchains_8gpu2m_73000_pairing.yml` | Queue | **true** |
| `eval-v3-allchains-8gpu2m-73000-t4` | `t-20260904104155-tctxm` | `eval_jobs/eval_v3_allchains_8gpu2m_73000_t4.yml` | Queue | **true** |

tag `ours_fusion_v3_allchains_8gpu2m_73000`。账本 `output/downstream_generation/eval_v3_allchains_73000_20260904_task_ids.tsv`。数字进 RESULTS §0.8 组 D，跑完再回填。

69000 四条已终态，不列入 Active：repr/cdr/pairing Success，t4 `t-20260904013725-7zcvf` **Killed**（闲时抢走，held20 只写了 2/20）。

---

## 2026-09-04 全链 8 卡 2M `checkpoint-69000` 全套下游（抢占单卡）

**操作**：评当前 8 卡 2M 最好 val 点 `checkpoint-69000`（eval **0.6577**；最新满包 72000 为 0.6609，未评）。覆盖与 18000/26000 相同：T1/T2/T3 + AB CDR + pairing + T4。**不打断**续训 `t-20260902021013-bm79q`。权重硬链接到 `eval_snapshot_69000/`，避免 top-k 剪枝弄丢评测输入。

首提 4 条非抢占（`t-20260904013629-skjzb` / `…-fbkqp` / `…-sjc44` / `…-kzls5`）后按用户要求改为抢占单卡，已 `cancel`。重提：

| Job | Task ID | YAML | 初始状态 | Preemptible |
|---|---|---|---|---|
| `eval-v3-allchains-8gpu2m-69000-repr` | `t-20260904013714-4w8nk` | `eval_jobs/eval_v3_allchains_8gpu2m_69000_repr.yml` | Staging | **true** |
| `eval-v3-allchains-8gpu2m-69000-cdr` | `t-20260904013717-rhmzz` | `eval_jobs/eval_v3_allchains_8gpu2m_69000_cdr.yml` | Staging | **true** |
| `eval-v3-allchains-8gpu2m-69000-pairing` | `t-20260904013720-nprjc` | `eval_jobs/eval_v3_allchains_8gpu2m_69000_pairing.yml` | Queue | **true** |
| `eval-v3-allchains-8gpu2m-69000-t4` | `t-20260904013725-7zcvf` | `eval_jobs/eval_v3_allchains_8gpu2m_69000_t4.yml` | 已提交 | **true** |

队列 `c20250601`，单卡 `ml.pni2.3xlarge`。tag `ours_fusion_v3_allchains_8gpu2m_69000`。账本 `output/downstream_generation/eval_v3_allchains_69000_20260904_task_ids.tsv`。数字进 RESULTS §0.8 组 D，跑完再回填。

---

## 2026-09-01 8 卡长跑（1M/2M 步）起跑；磁盘配额把一条 8 卡任务卡成 4 连 Failed

**触发**：核对"那条 Running 的 8 卡 2M 任务前面 4 连 Failed"到底是什么问题。

**方向变化**：50k 短跑已收口出数（RESULTS §0.8），本轮把步数拉长一个数量级。现役三条
全部 **global 256 + polynomial power=1**，与 50k 那批**不可混排**（目标函数、batch 口径、
LR 调度三样都变了）：`..._diffusion_allchains_immune_v3_8gpu_2m`（Running）、
`..._bert_immune_v3_1m`（Queue）、`..._diffusion_immune_v3_spot_2m`（Queue）；
4 卡版 `t-20260901032611-npf27` 已 Killed，换成 8 卡。

**根因**：`safetensors_rust.SafetensorError: Error while serializing: I/O error:
Disk quota exceeded (os error 122)`，崩在 `Trainer._save → safetensors.torch.save_file`，
**不在训练步**。首跑 14h07m 从 step 0 跑到 17000 后在存盘时爆，留下 132 MiB 半截
`model.safetensors`；随后又有 3 条同名新任务各 58 分钟，「从 16000 续 → 跑满 1000 步 → 同一处再爆」，
**约 3 小时 8 卡非抢占资源换到 0 步净进度**。这 4 条是不同 JobId 的独立提交，不是同一 JobId 被平台 `RetryOptions` 自动重试（每个 JobId 只有 1 个实例，提交间隔 5～30 秒，与 `IntervalSeconds: 180` 不符；JobId 与实例证据见下方 Active 表旁注）。

⚠️ **别看 `df` 下结论。** 底层 `fs_vepfs-cnbj2c98dea54433` 3.1P 已用 2.3P、**尚余 809T** ——
爆的是**目录/租户配额**。一次 save 需要约 **9.0 GB 一次性余量**（`model.safetensors` 2.3G +
`pytorch_model_fsdp.bin` 2.3G + `optimizer.bin` 4.5G），而 **top-k 剪枝发生在写完之后**，
峰值 = 现有占用 + 一整个新 checkpoint。第 5 次（`t-20260902021013-bm79q`）于 19:10:42
存盘成功，但只是期间别的任务腾出了空间，**配额仍贴临界、问题未修**。

✅ **`pick_latest_full` 的三件套判据救了一次。** 它要求 `optimizer.bin` +
`pytorch_model_fsdp.bin` + `scheduler.pt` 齐全才认 checkpoint，因此正确跳过了半截的
`checkpoint-17000`、回退到完整的 16000，**没有重演 §4.2.6 那种「从坏 ckpt 反复续跑」死循环**。
新增 checkpoint 挑选逻辑时必须保留这个判据。

**新发现：`TopKValLossCheckpointCallback` 在重启时重置账本。** 它只从**本进程**的
`log_history` 取 `eval_loss` 重建 top-k，看不到上一个进程存过什么，于是每次崩溃-重启都把
上一轮 checkpoint 甩成账本外孤儿、永不被剪 —— **崩溃循环自己在抬高占用，形成正反馈**。
实测该目录 `topk_val_manifest.json` 只剩 `checkpoint-17000` 一条，盘上却还有
7000/13000/15000（各 2.3G 已 slim）+ 16000（9.0G 未 slim）。

**全 `output/` 只读盘点**（1.2T，其中 checkpoint 数据 790.8 G）：

| 口径 | 量 |
|---|---:|
| resume-only（`optimizer.bin`/`pytorch_model_fsdp.bin`/`rng_state_*`/`scheduler.pt`，剪掉不丢权重） | **328.3 G** |
| 账本外孤儿 checkpoint 整体 | **132.9 G** |
| `checkpoint-final` 与 `checkpoint-50000/model.safetensors` **md5 逐字节相同**却各存一份 | **69.1 G** |
| 两条已终态 8B run 的 `checkpoint-50000`（fat，各 124.5 G） | 249.0 G |

⚠️ `output/grammar_v2_esmc300m_integrated_llada_7l_step{117000,121000,164000,189000,359000,389500}/best.pt`
是**硬链接**到 `..._7l/checkpoints/*` 与 `..._7l/latest.pt`（共享 29.2 G），**删单边不释放空间**；
且 `PROJ_GUIDE.md` 把 `..._7l_step117000/best.pt` 及其 SHA256 钉为 TCR-β public Track-A 的
canonical 行，**不能动**。这些 `.pt` 内部也没有 optimizer，没有"只剪 optimizer"的省法。

**核实过、不是 bug 的两项**：① resume 健康 —— 续跑后 step 16300–17000 的 train loss 7.2–7.7、
`learning_rate 9.925e-05` 与首跑同 step 逐条吻合；② train loss ~7.3 与 `eval_loss` 0.7028
差一个数量级**原因未查明**，但两条路径 CE 计算相同（`loss_weight_type='none'`、
`token_weights=None`）且首跑与 resume 一致，登记为未解项，**不得跨口径比较**。

**✅ 已执行磁盘回收：`output/` 1.2 T → 767 G，释放 389 GB**（19:30Z，任务全程 Running 未受影响）：

| 组 | 动作 | 释放 |
|---|---|---:|
| 1a | 删 `ABORTED_gb128_..._4gpu/` 整目录 | 9.1 G |
| 1b | 两条已终态 8B run 的 `checkpoint-50000` 剪 resume-only | 94 G × 2 |
| 1c | 5 处 `checkpoint-final/model.safetensors` **改硬链接**指向同 run 的 `checkpoint-50000` | 69 G |
| 2a–2c | 已 Killed 的 spot ×2 / 4 卡 ×2 / 早期 `diffusion_immune` 的账本外孤儿 | 117 G |
| 2d | 正在跑的 `..._8gpu_2m` 的 slim 孤儿 `checkpoint-7000/13000/15000` | 6.9 G |

⚠️ `checkpoint-final` 是**改硬链接不是删** —— `eval_jobs` 引用它 8 处、`train_jobs` 引用
`checkpoint-50000` 2 处，两边路径都得留。改前逐个自查大小 + 首尾 64 MB md5，改后 `nlink=2`、
inode 相同、`safetensors.safe_open` 正常解析（8B 602 张量／270m 386 张量）。
**今后不要原地覆写这两个路径中的任何一个 —— 改一个就是改另一个。**

⚠️ **刻意保留**：`..._diffusion_allchains_immune_v3_4gpu/checkpoint-33000`（2M 的权重来源）、
`..._bert_immune_v3/checkpoint-50000` 完整满包（1M 的 resume 来源）。

**未做 / 待拍板**：**配额本身没修** —— callback 账本重建与 save 前配额预检两项均未实现，
回收只是把 9.0 GB 的窄门往后推；grammar_v2 五条 339 G 是否保留待科研决策；
2M 步 ≈ **68 天**独占 8 卡（2.95 s/it）的规划意图待确认。

**文档同步**：`examples/llada/PROTEIN_PRETRAIN_PROGRESS.md`（新增 §4.2.7、§4 状态表新增 H/I 行并
修正已过期的 E 行、§7 待办、§8 变更日志）、`examples/llada/README.md`（当前训练任务节重写）、
本文件（Active 表回填三条 + 本节）。

---

## 2026-08-29 布局口径纠错：前缀采样把 `tcr_pmhc` 低估 4 倍；两个新发现

**触发**：回答"数据处理是否完整"时要按 grammar 布局核对任务覆盖，跑
`scripts/count_grammar_layouts.py` 得到 `tcr_peptide` 8.10% / `tcr_pmhc` 2.47%，
与"三个表位源大多带 MHC"的直觉相反，于是做了一次独立全量扫描交叉核对。

**根因**：该脚本读满 `--per-source` 就 `break`，取的是**前缀而非随机样本**。
布局是「行填了哪些字段」的函数，而 `tcr_papers_v2` 是 7 个论文语料首尾拼接的
（`tcrt5` / `tcrdesign26_pmhc` / `tcrdiff` / `tcrdesign26_beta` / …），
**前 3 万行 100% 是 `tcr_peptide`**，全量却是 80.3% `tcr_pmhc`。
547,274 条被归错布局，`tcr_pmhc` 低估约 **4 倍**。

同一陷阱也坑过 `PROTEIN_PRETRAIN_PROGRESS.md` §4.5 "取各源真实首行"举例的做法：
`trait` 首行恰好无 MHC，被画成 `tcr_peptide`，而全量 `trait` **95.2% 是 `tcr_pmhc`**。

**修法**：改蓄水池采样 + `--seed`；`rows=` 列改名 `kept=`（它统计的是加载期过滤后的
行数，不是磁盘行数）。新数字与独立全量扫描一致：`tcr_pmhc` 671,678 vs 673,686、
`tcr_peptide` 137,833 vs 135,825（采样噪声内）。

**修正后的六布局 loss 预算**（`gen%` = 占全部待预测残基）：

| 布局 | 保留行数 | 记录% | gen% |
|---|---:|---:|---:|
| `antibody_pair` | 2,485,471 | 31.9% | **49.64%** |
| `tcr_pair` | 2,094,231 | 26.9% | **40.92%** |
| `antigen_antibody` | 276,412 | 3.6% | **4.55%** |
| `tcr_single` | 2,128,750 | 27.3% | **2.90%** |
| `tcr_pmhc` | 671,678 | 8.6% | **1.78%** |
| `tcr_peptide` | 137,833 | 1.8% | **0.20%** |

三个无条件/抗体布局吃掉 **95.1%**，表位条件生成合计仅 **1.98%**。根因是残基数量级
（232 vs 13–31）而非行数，**加数据改不动**——`tcr_papers` 翻 3 倍也只到约 3%。

**新发现①：单链 α 摄入不了，卡在 grammar 而非数据。** 实测同一序列分别标
`tcr_alpha` / `tcr_beta`，两者 `grammar_name` 都是 `tcr_single` 且**全部 11 个输出
字段逐字节相同**。`GrammarTokenizer` 只有一个 `<tcr>`；配对布局靠 `[alpha, beta]`
位置编码身份，单链时无位置可用（`grammar.py:466-472`）。
顺带纠正 `GRAMMAR_V1.md` 的错误陈述——它写"链身份由 `position_ids_chain` /
`chain_ids` 编码"，该说法**只对多链块成立**，且渲染输出里根本没有 `chain_ids` 键。
要摄入 α 须扩词表 → 现有 checkpoint 不兼容，属建模决策，未做。

**新发现②：加载期过滤在 train/valid 上丢弃率严重不对称。**

| 源 | train 丢弃率 | valid 丢弃率 |
|---|---:|---:|
| `asd_antibody` | **67.5%**（850,134→276,412） | **5.0%**（47,230→44,866） |
| `tcr_native` | 32.7%（143,391→96,552） | 12.2%（4,390→3,855） |
| `trait` | 54.5%（69,251→31,515） | 54.3%（3,050→1,393） |

`trait` 对称说明这不是通病。后果：**ASD 的 valid loss 不能做 early-stopping** ——
valid 保留了大量被从 train 剥掉的 Kong 相似簇抗体家族，在测一个训练时被刻意屏蔽的
分布。无泄漏风险（valid 非 Kong 基准本身）。机制未查清，猜测是 step6 整簇装箱让
Kong 相似簇集中落进 train，未验证。**待决策**：重建语料使两 split 去污一致，
或在监控里标注该指标不可比。

**文档同步**：`examples/llada/DATA_PIPELINE_README.md`（§1 采样陷阱 / §1.1 预算表 /
§6.2 单链 α / §6.4 split 不对称）、`examples/llada/PROTEIN_PRETRAIN_PROGRESS.md`
（§4.5 首行警告 / 新增 §4.5.1 / §8 变更日志）、`scripts/data/README.md`（第 5 坑 +
采样规则）、`downstream/asd/README.md`、`data/tcr_repertoire/README.md`、
`dllm/pipelines/qwen3_vl_arch/GRAMMAR_V1.md`。

---

## 2026-08-28 v3 三源补做 valid 近重复搬迁；§3 构成错配收尾

**触发**：审查「模型选择发生在与训练分布不同的集合上」（RETRAIN_PLAN §3）时发现，
2026-08-28 早先做过的 valid 近重复搬迁只覆盖 `tcr_native` 和 `tcr_papers` **v1**，
而 v3 训练用的是 `tcr_papers_v2` —— 后者由 `finalize_papers.py --out-root` 从零重建，
**不继承 v1 已清洗的 split**。「这个源处理过了」不能推广到它的重建版本，与 §7.3 的
blocklist 新鲜度事故是同一类错误。

**实测的近重复率**（valid 行的 CDR3β core 与 train 某行 Lev≤1）：

| source | valid 近重复率 | 处理 |
|---|---:|---|
| `tcr_papers_v2` | 49.2% | 已搬迁，valid 14,449→7,336，train 668,331→682,383（+2.10%） |
| `tcr_repertoire` | **38.9%** | 已搬迁，valid 201,504→123,049，train 1,971,794→**2,128,750**（+7.96%） |
| `trait` | 19.1% | 已搬迁，valid 3,769→3,050，train 67,842→69,251（+2.08%） |

搬迁而非丢弃，迭代到不动点（v2 用 3 轮、trait 用 5 轮——搬入 train 的行会成为新参考）。
所有源被搬走的 eval 行里，`(CDR3β, epitope)` 在 train 中精确出现的都是 **0 行**：
这修的是选模信号的可信度，不是补答案键泄漏。原文件备份为 `*.csv.pre_neardedup`。

**另外三个源补测后判定无需处理**。此前只查了带 `cdr3b` 列的源，而 `oas`+`ots`+
`asd_antibody` 占训练残基的 94%、从未查过。按各源真实生成目标测：`asd_antibody`
（`heavy_fv`+`light_fv`）exact 0.00% / Lev≤1 4.20%，`ots`（全长配对 Fv）0.00% / 1.85%，
`oas` 0.00% / 0.45%。

**度量单位必须匹配生成目标，否则结论会反过来**——这是本轮最容易做错的一点。`ots` 若按
CDR3 量是 exact 27.25% / Lev≤1 69.90%，看着像重大泄漏；按全长配对量则是 0.00% / 1.85%。
天然 repertoire 中同一条 CDR3β 本就会与不同 α 链配对，而模型生成整条链，CDR3 复现不构成
答案键。反之四个 TCR 源的 epitope/MHC 是固定上下文、CDR3 才是去噪目标，那边必须按 CDR3
量。判据是 `ImmuneSourceSpec.roles`（实测 `oas`/`ots` 的 `gen_res == res`，
`asd_antibody` 为 36%）。

**§3 的构成错配是过期记录，已改判**。那条「valid 73% / train 8%」描述的是「单一混合
eval + 前缀截断」时代。现在 `subsample_seed=0` 走 reservoir 抽样、eval 分源各截 2,000 行，
实测 eval 构成是七源近等权（各 14.8%，`trait` 11.0%）。错配仍在但方向相反且是有意选择——
按 train 比例加权会让占 86% 的 `oas+ots+tcr_repertoire` 主导选模。**要做的是在论文里写清
`eval_loss` 是源等权口径，不是继续改代码。** 详见 RETRAIN_PLAN §3c。

**顺带修的三个坑**：
- `move_near_dup_eval_rows.py` 的 CDR3β 锚点约定改为按源配置：`trait` 存带锚点全 junction，
  与 `tcr_native`/`tcr_papers` 的 anchor-free core 相反，写死一个值会静默算错。
- `cdist` 分块由固定 2000 改为按参考集大小自适应（`tcr_repertoire` 的 1.97M 参考集下，
  固定 2000 会申请 ~16 GB 矩阵）。
- 报告 JSON 改为按数据集合并写入：此前跑子集会抹掉上一次其他数据集的记录。

**搬迁后重测的 v3 语料口径**（2026-08-29 05:30，`count_immune_mix.py`）：train 合计
**7,794,375** 条（原 7,626,737），总残基 1,273,914,576 / 生成链残基 1,150,746,721。
`tcr_repertoire` raw 2,128,750 / kept 2,128,750（**剔 0**）——搬回 train 的 156,956 行无一
命中 blocklist，它们本就与 train 同处一个已去污染的 pool。启动断言
`assert_corpus_fresh.py` 通过（搬迁不触碰 blocklist）。两个 v3 训练任务
（`t-20260829031748-96vjf` / `t-20260829031757-2qnjn`）当时仍在 **Queue**，排队 10 小时未
启动，所以重写期间没有任何作业在读这些 CSV，启动后读到的是修好的语料。

**一个必须记住的副作用**：`tcr_repertoire` 的 train 超出了当初刻意设的 2,000,000 token
预算上限 6.4%（残基占比 2.04%→2.19%，判为可接受未回切）。**若重跑 `build_repertoire.py`，
那个上限会把搬回来的行重新截掉，届时须重跑搬迁。**

**顺带发现一处并发改动（非本次工作）**：`trait_benchmark` blocklist 于 2026-08-28
17:25:35Z 被另一会话从 862 键重建到 **59,212** 键，`trait` 的 train 保留数由 34,872 降到
31,515（−9.6%）、valid 由 1,485 降到 1,393。`trait` 是**加载期**过滤，所以不需要重建语料
就自动跟上；这与 `tcr_repertoire`（构建期去污染，blocklist 变了必须重跑构建）正好相反。

**未做**：`trait` 只有 1,393 行 valid（低于 2,000 配额，主因是 `replaces_trait` 顶替过滤
与上述 blocklist 重建，而非本次搬迁），七源未严格等权（各 14.93% vs `trait` 10.40%），
已登记不修。

## 2026-08-04 AB/TCR v2 cluster 去污染与 technical export 完成

- 本轮继续只处理 AB/TCR 数据层，没有修改模型、renderer、checkpoint 或 sampling
  weights，也没有启动训练。修复 benchmark role parser 对 `tcra`/`tcrb` 和抗体
  `CDRH*`/`CDRL*` 的识别；重跑后 bank 为 antibody heavy 14,883、light 10,596、
  TCR alpha 25,629、beta 92,811、peptide 449。2026-08-03 记录中的旧 overlap 数值由
  本节及机器报告取代。
- core build `ir2exp_f7a60484c7e3a20db6a2` 已应用 exact blocklist 与 MMseqs2
  connected-component near-neighbor quarantine。TCR primary 从 41,489 隔离 37,060，
  保留 4,429（3,986/222/221）；另产生 3,970 条 train-only negatives，known-positive
  collision=0。antibody recognition 从 308,267 隔离 144,936，保留 163,331
  （146,998/8,167/8,166）；property 从 19,981 隔离 6,395，保留 13,586
  （12,227/722/637）。全部 protocol 的 split audit 通过且 residual benchmark cluster
  match=0；80-artifact manifest 共 2,578,231,408 bytes。
- strict pairing build `ir2pair_6a1a5b62752caccd2cd5` 不复用旧 valid/holdout，只从
  source train 按 upstream pair group 重分 98/1/1。OAS 输入 2,486,442，隔离
  987,593，保留 1,498,849（1,468,754/15,295/14,800）；OTS 输入 2,102,715，隔离
  627,438，保留 1,475,277（1,445,804/14,732/14,741）。OTS 的 261,753 条缺 required
  bucket 均为 B-D/B-B/A-A/D-D scope exclusion，已整组 block，不是 parser 丢链。
  28-artifact manifest 共 5,409,708,251 bytes。
- 候选 recipe `ir2recipe_1e3eac44551e88aedc6e` 固定 OAS H/L、OTS alpha/beta、
  antibody recognition、TCR recognition 和 antibody property 五个 plane/pack，合计
  real train 3,077,769、valid 39,138、test 38,565，另保留 3,970 条独立 train-only
  synthetic negatives。manifest SHA256 为
  `2119da4bc0f36cefd805aea4a409508399734b2d19694bd55423db6662e166be`。
- 当前 gate 为 `technical_data_export_ready=true`，但
  `rights_review_complete=false`、`runtime_views_ready=false`、
  `sampling_weights_frozen=false`，因此 `export_ready=false`、
  `training_ready=false`、`training_started=false`。剩余 blocker 是 SAbDab2/CATNAP
  权利审查、per-chain/per-region mask 与 relation-target renderer、四平面 sampling/
  token-budget 冻结。
- 权威报告：
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/IMMUNE_RECEPTOR_DATA_V2.md`；
  machine summary：
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/immune_receptor_v2/reports/summary.json`。

## 2026-08-03 AB/TCR canonical v2 构建完成

- 本轮只执行 antibody/TCR 数据层，不修改模型、renderer、训练 recipe 或 checkpoint，
  也没有启动训练。新增可重建 `bioseq.v2` schema、source adapters、稳定 group/component
  split、benchmark quarantine/leakage audit 与 artifact manifest。
- 官方源已完整下载并校验：IEDB receptor/T-cell exports、CATNAP 2026-08-01 六文件、
  SAbDab2 0.1.0 `splits.tar.gz`。此前“SAbDab2 无 `abag_split.csv`”来自损坏旧下载，
  完整归档已确认同时含 antibody-only 与 antigen-aware split。
- TCR 六源 adapter 输入 1,021,452 条，exact-measurement merge 后 union 为 922,479，
  strict core 41,489；antibody 六源输入 472,922，union 为 470,949，interaction core
  308,267，含 intrinsic property 后总 core 328,248。
- 九套 deterministic split 全部通过 disjoint audit。antibody-antigen joint-hard 因
  74.32% 巨型连通分量只能得到约 98.11/0.94/0.94，保留严格 disjointness，不拆分
  component 伪造 90/5/5。
- 修复 benchmark quarantine 只扫描 TCR 根导致 antibody overlap=0 的审计错误；新 bank
  覆盖 16 个根、142 个文件。候选 union 命中 TCR exact receptor-peptide 187,002、
  paired-alpha/beta-peptide 15,885、receptor-pMHC 16,632；antibody exact receptor
  102,837。以上是重叠审计命中，尚不是最终删除行数。
- 当前状态固定为 `canonicalized=true`、`split_ready=true`、
  `export_ready=false`、`training_started=false`。下一步仍须应用 quarantine、完成
  cluster-level near-neighbor decontamination、仅在 train 内构造 negatives，并补齐
  SAbDab2/CATNAP 权利记录后才能物化训练 export。
- 权威说明：
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/IMMUNE_RECEPTOR_DATA_V2.md`；
  机器可读 summary：
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/immune_receptor_v2/reports/summary.json`。

## 2026-07-23 免疫受体训练数据范围审计

- 下一版训练范围固定为 antibody H/L、TCR α/β、antibody-antigen 和
  TCR-epitope/pMHC；nanobody/VHH、MINT/STRING/general-PPI、当前无 antigen
  sequence 的 neutralization 和无 specificity 的 bulk TCR 不进入新 recipe。
- 已只读审计 SAbDab2、FLAb/AbRank/Kothiwal、PISTE、TDC、TEIM 和 legacy
  VDJdb/MIRA/McPAS。严格 SAbDab2 得到 3,980 rows / 2,489 unique Ab–Ag triples；
  AbRank 当前长度上限内得到 76,515 sequence-complete rows；四个 TCR specificity
  正例源从名义 224,488 降为 142,456 exact union。
- 当前没有重建 Arrow、提交训练或改变任何 checkpoint。下一步门槛是 canonical
  schema、双侧 group split、benchmark decontamination、per-chain view mask 和
  relation-target renderer。
- 完整事实与配方：
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/TRAINING_DATA_CATALOG.md`。

## 2026-07-21 MINT 五任务官方数据重建

- 本轮没有修改或训练 BioSeq 模型；工作范围是 MINT benchmark 数据与 baseline 协议接线。
- 固定 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/mint_official/mint_repository`
  到 MINT commit `06694b7606e2d00b76ec58daf5c7aecdaf7cd283`，并对五个
  `prepare_data.ipynb` 逐一固定 SHA256。
- 新增 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/mint_tasks/prepare_official_data.py`
  和 `execute_notebook_cells.py`；在独立 `mint-prep-official` 环境中原样、按顺序执行每个
  code cell，再校验 schema、行数、标签、mutation 变化和 SKEMPI complex fold 泄漏。
- 完成 HumanPPI 26,319/234/180、YeastPPI 4,945/95/394、Gold-standard
  163,192/59,260/52,048、MutationalPPI 3,406、SKEMPI 6,706 行的重建；总 manifest
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/mint_official/manifest.json`
  为 `validated`。
- MutationalPPI 使用 SWING 历史 commit `6cde98ae8ca5b7d420bd3474d190fc96bc5189da`
  的 3,422 行输入，并固定 1,299 个实际访问 accession 到 Swiss-Prot release `2025_01`；
  notebook 跳过 16 行后得到与论文一致的 3,406 行。
- `tasks.py` 默认根切换到 `mint_official`，注册表缩为五项；MutationalPPI 改为保留
  WT/mutant 四序列，两个 embedding extractor 均走 WT-mutant 差分。全量下游编排也移除
  PDB-Bind。新 embedding cache 统一追加 `mint_official_06694b7`，避免静默复用旧数据缓存。
- 论文冲突已写入机器可读 audit：Gold-standard notebook train 比论文多 173 行；
  SKEMPI notebook shuffle 未设 seed且当前 fold 大小不等于论文；公开 MutationalPPI 路由
  与论文 complex-wise CV 不一致。最终混合选表规定前三项用 Source Data `[P]`，后两项
  用可审计本地固定 split 的 8-baseline 重跑 `[L]`，不以论文值补空。
- SKEMPI 本地 diagnostic fold 的完整 NumPy 状态已提交为
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/mint_tasks/skempi_numpy_state.json`；
  回灌复跑后 CSV SHA256 仍为 `af584bf528a71b4d064943385834fd60b134b0aab7e7f97011747ec6777b5af5`。
- 新增测试 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_mint_official_rebuild.py`；
  `pllm` 激活后在实际评测环境运行结果为 12 passed。
- 详细审计：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/mint_tasks/OFFICIAL_REBUILD_AUDIT.md`。

## Active focus (2026-07-03 onward): LLaDA backbone ONLY

> Manage **only the LLaDA-backbone line** (`--model-type llada` / `llada_esm2`, `BioSeqLLaDAEncoderDiffusionModel`). Update this file in real time on every LLaDA submit/cancel/status change. Rule: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/.cursor/rules/volc-train-task-log.mdc`.

## Active Volc Training Tasks

> Only non-terminal jobs (`Initialized` / `Queue` / `Staging` / `Running` / `Killing`). Remove a row when the job reaches `Success`, `Failed`, or `Killed`.

Last updated: 2026-09-17T12:21:56Z（只读核对；BERT 看护自动重提，当前 Queue）

| Task ID | Job / TaskName | 队列 | 卡数 | max_steps | 状态 |
|---|---|---|---:|---:|---|
| `t-20260917174134-kxmkk` | `protein_esmc_llada270m_bert_immune_v5_8gpu_spot` | `c20250601` 闲时 | 8 | — | Queue（看护于 09:41Z 自动重提） |

已从本表移除（2026-09-17 用户授权 cancel）：`t-20260912150714-lwf28`（`protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601`，**Killed**，End=`2026-09-17T08:13:43Z`）；`t-20260912035657-dd8v9`（`protein_esmc_llada270m_bert_immune_v5_8gpu_spot`，**Killed**，End=`2026-09-17T09:37:24Z`）。评测 / FABind 未动。

已从本表移除（2026-09-12 用户授权 cancel）：`t-20260912111652-sqfcw`（`protein_esmc_llada270m_diffusion_immune_v5_8gpu_spot`，**Killed**）。正式臂与 BERT 闲时未动。

已从本表移除（2026-09-11 查询已终态）：`t-20260911085733-5b727`（`eval-ophiuchus-ab-esm-head-epochs`，**Success**）。

已从本表移除（2026-09-11 探针评测终态）：`t-20260911054553-jb7v5`（`eval_ophiuchus_ab_official_probe`，**Success**）。
已从本表移除（2026-09-11 查询已终态）：`t-20260902021013-bm79q`（Killed）、`t-20260901235433-q76qw`（Failed）、`t-20260901032620-vvngv`（Killed）。

🔴 **`..._8gpu_2m` 前 4 条同名任务全部 `Failed`，根因是目录磁盘配额**（不是文件系统满，底层尚余 809T）：
`t-20260901033935-h55f4` / `t-20260901230256-ns4q4` / `t-20260902000627-j4mqm` / `t-20260902010820-t6hq6`。
这是 **4 次独立提交（4 个不同 JobId）**，不是平台按 `RetryOptions` 自动重试：每个 JobId 用 `ml_task instance list` 都只有 1 个实例（`…-worker-0`），没有带重试序号的第二实例，也没有第二轮 `LaunchTime`；相邻 JobId「上一条 End → 下一条提交」间隔 5～30 秒（h55f4 End `15:02:42Z` → ns4q4 提交 `15:02:56Z` 为 +14s；ns4q4 → j4mqm 约 +29s；j4mqm → t6hq6 约 +5s），与 YAML 的 `IntervalSeconds: 180` 不符。若是平台按配置重试，应表现为「同一 JobId、新实例、间隔 ≥180s」。同配 `EnableRetry: true, MaxRetryTimes: 5` 的 `t-20260901235433-q76qw` 也停在 Failed、instance 只有 1 条。看护脚本不是这 4 连的来源（`scripts/llada_train_watchdog_jobs.json` 只有已 paused 的 grammar 任务；`scripts/monitor_spot_tasks.py` 只盯 `…_v3_spot_2m`）。配额不足仍会导致连续 Failed——教训保留，只是这 4 个新 JobId 不能算平台重试。
每轮「从 `checkpoint-16000` 续 → 跑满 1000 步 → 在 step 17000 的 `save_model` 上
`Disk quota exceeded (os error 122)`」耗 58 分钟，**净进度 0**。已终态故不列入本表。
经过、可回收量盘点与分级方案见
[`examples/llada/PROTEIN_PRETRAIN_PROGRESS.md`](examples/llada/PROTEIN_PRETRAIN_PROGRESS.md) §4.2.7。

已终态、已从本表移除：`t-20260901032611-npf27`（`..._4gpu_2m`，Killed，换成 8 卡版）、
`t-20260901030607-zmrrv`（`..._v3_1m` 首条，Failed）。

**历史（2026-08-28 记录，仍有效）**：2026-08-28 23:46 逐条查询确认，此前挂在本表的 8 条**全部 `Success`**，按
`volc-train-task-log.mdc` 移除：`t-20260816051821-fj6p7`（8b_bert_immune）、
`t-20260816051844-zm4cl`（8b_diffusion_immune）、`t-20260820053314-4gmfb`、
`t-20260820053318-blk66`、`t-20260820053506-l62kt`、`t-20260820053323-29dln`、
`t-20260820053327-m5c6n`、`t-20260820053511-p8p2r`（后六条为 `eval-immune-*`
repr/gen，本不该记在训练表里）。

四条 immune fusion 训练均已跑满 50k 并留有 `checkpoint-final`：
`270m_bert` / `270m_diffusion` / `8b_bert` / `8b_diffusion`。

**v3 尚未提交**：`train_jobs/protein_esmc_llada270m_diffusion_immune_v3.yml`
（2026-08-28 11:47 落盘）为七源 from-scratch 配置，启动前调用
`assert_corpus_fresh.py`。提交是算力决策，等确认。

**当前训练**：
1. **Immune fusion（queue012）**：① `8b_bert_immune` / ② `8b_diffusion_immune`（Running，`Preemptible: false`）；③ `270m_bert_immune` 已 **Success**（50k）；④ `270m_diffusion_immune` 从闲时改为非闲时续训（`t-20260817153216-qvtwx`，resume `checkpoint-9000`）。数据 `oas+ots+asd_antibody+trait+tcr_native`，无 nanobody，ASD 不过 benchmark filter。
2. **旧 ablation 已终态，已从 Active 表移除**：`bert_esmctrain`/`bert_noesmc`/`bert_noesmc_scratch` = Failed；`270m_bert_esmctrain_scratch` = Success。

**队列策略（2026-08-04 起）**：本系列新提交一律用 **`queue012`**，不要用 `spot-share-queue`；相关 `train_jobs/protein_esmc_*.yml` 已全部改为 `ResourceQueueName: queue012`。

### 2026-08-02 ESMC×LLaDA-8B fusion wandb `to_dict` 崩溃修复 + 重提

- **22:15/22:22Z 旧任务崩溃**：`t-20260803061533-q8r98`（add_diffusion）与 `t-20260803062133-blhfm`（add_bert）在 `on_train_begin` 崩溃——W&B 回调（`--report_to wandb`）执行 `model.config.to_dict()`，而旧版 `LLaDAEsmcFusion.config` 是裸 `types.SimpleNamespace`，无 `to_dict`，rank0 抛 `AttributeError`。两个 `output/.../wandb/` 目录为空、无 checkpoint，进度为零；rank0 死后其余 rank 卡在 NCCL，volc 仍误报 **Running**（僵尸态）。
- **22:47Z 源码修复**：`examples/llada/protein_fusion_model.py` 已把 `_FusionConfig` 定义为 `SimpleNamespace` 子类并实现 `to_dict()`（返回 `dict(vars(self))`），供 HF W&B 集成序列化超参。修复落盘时间晚于两个任务的启动时间，故旧进程仍跑旧代码。
- **22:53Z cancel**：`t-20260803061533-q8r98`、`t-20260803062133-blhfm` 均 `ml_task cancel` 成功。
- **22:54Z 重提**：用相同 YAML 重新 `ml_task submit`（走 no-proxy 封装），新任务将导入已修复代码。add_diffusion → `t-20260803065402-dqmbg`；add_bert → `t-20260803065406-5rsk8`。
- **22:57Z 确认恢复**：两个新任务日志显示 wandb offline run 正常初始化（`to_dict` 修复生效）、loss 稳步下降（`~78/71 → ~34 → ~22 → ~17`），无 Traceback。取日志命令：`ml_task logs -t <id> -i worker-0 -l 40`。

### 2026-08-02 训练设计调整：改为 bert ESMC-conditioning ablation

- **背景（用户澄清本意）**：本质对照应为"**都用 bert 目标**，一条用 ESMC 提序列 feature、一条不用 ESMC"，且 **ESMC 参数要参与训练**。原在跑的 `add_bert`(bert,add,frozen) + `add_diffusion`(diffusion,add,frozen) 与此不符（是 bert-vs-diffusion，且 ESMC 均冻结）。
- **单链编码确认**：ESMC 走单链提取——`encoder_input_ids`=`[B,C,L]`，`encode_chain_tokens` reshape 成 `[B*C,L]` 单次前向，每链独立编码互不 attend，符合预期。
- **23:12Z cancel**：`t-20260803065402-dqmbg`、`t-20260803065406-5rsk8` 均 `ml_task cancel` 成功。
- **23:13Z 新建 + 提交**：新增两个 job（bert 目标）并提交：
  - `train_jobs/protein_esmc_llada8b_bert_esmctrain.yml`（add + **可训练 ESMC** `freeze_encoder=False`，per_device 2/ga 8 保住全局 128 并给 ESMC 反传留显存）→ `t-20260803071310-fvhz6`
  - `train_jobs/protein_esmc_llada8b_bert_noesmc.yml`（token + 无 ESMC，per_device 4/ga 4）→ `t-20260803071313-9hmjw`
- **待观察**：`bert_esmctrain` 解冻 ESMC 后显存上升，需确认加载 8B 后不 OOM；`bert_noesmc` 的 token 路径此前只在 add 模式跑过 smoke，需确认首个 step 正常。
- **01:14Z 省盘 save 策略**：根因是 `save_only_model` 未开 → FSDP 同时写 `model.safetensors`(32G) + `pytorch_model_fsdp.bin`(32G 重复) + `optimizer.bin`(60G+)，单 ckpt ~125G。已改 `protein_pretrain_esmc.py` 默认 `save_only_model=True` + `slim_checkpoints=True`（存后删 resume-only 大文件），job yml 同步显式打开；已有 ckpt 现场瘦身 **~130G→33G**。用户要求**不停训**；在跑任务内存仍是旧代码，下次 save 可能再写胖文件，需事后瘦身。
- **01:17Z top-k 改为 3**：`save_top_k` 默认与 yml 从 5 → **3**（仍按最低 eval_loss）。不停现有任务；磁盘侧已按 top-3 对齐 manifest。
- **02:26Z 再次瘦身（不停训）**：旧进程写出的胖 ckpt 已现场处理——删除 `optimizer.bin`/`pytorch_model_fsdp.bin`/`rng_state_*`，并按 eval_loss 裁到 top-3。`bert_esmctrain` 保留 2000/1000（~33G×2）；`bert_noesmc` 保留 4000/3000/2000（~33G×3），删掉最差的 step1000。

## Active Volc Evaluation Tasks

Last updated: 2026-09-17T14:51:05Z（159000 AB/TCR十二项云端均Success，移出Active；四项本地亦完成）

已从本表移除（2026-09-17 查询已终态）：`t-20260917015632-hx44v` pairing p0/CFG0、`t-20260917015637-5k6vp` CDR Kong、`t-20260917015642-9t4rd` CDR SAb23、`t-20260917015648-qmhgs` T1 cdr3b、`t-20260917015653-492rz` T1 cdr3ab、`t-20260917015658-blclc` T1 others_cdr3b、`t-20260917015703-xnnpp` T1 others_longab、`t-20260917015708-rsccg` T4 benchmark14、`t-20260917015713-rk77d` T4 held20（均为 **Success**）。另移出已 Killed 的 92000 闲时 pairing 残留四项：`t-20260915235443-7sb5d` / `t-20260915235447-qcz5f` / `t-20260915235450-8dvfl` / `t-20260915235454-6dmgm`。

| Task ID | Job / TaskName | 队列 | 卡数 | 覆盖 | 状态 |
|---|---|---|---|---|---|

159000 AB/TCR 云作业均已 Success，当前本表无活跃评测。历史 v3 队列警告及提交状态仅保留为当时记录。

> **2026-08-25 23:47 收口**：SAb23H2 sweep 与 pairing `max_iter=124` 均已 Success，数字写入 `downstream/benchmark/RESULTS.md` §0.5 / §0.6。活跃 `eval-immune-*` 列表为空。SAbDab Kong + 官方 ckpt 复现已在 §0.5。仍待做：T4-held20 immune 语料泄露报告、T4 sparse-13 重评分。

> **2026-08-25 01:41 那批八条已全部终态**（7 Success + 1 Failed）。`eval-immune-8b-diff-pairing` 的 Failed 是最后写 metrics JSON 时 `OSError: [Errno 122] Disk quota exceeded`，**指标已算完并留在日志里**（ImmunoMatch 0.4376 / diversity 0.5288）。`output/` 已占 1.1T，其中两条 8B immune 训练各 250G，需要清理决策。
> 那批的 CDR / pairing 数字均为 `max_iter=8 / 32`，**已被本轮 `max_iter=2 / 124` 的对齐重跑取代**。

### 2026-08-25 生成评测改为阶段并行

- **问题**：`run_immune_fusion_gen.sh` 是串行的，实测（上一轮 8B）CDR 79 分钟 → pairing 106 分钟 → T4 28 分钟，共约 **3h40m**，pairing 白等前面 80 分钟才开始。
- **改动**：runner 新增 `GEN_STAGES`（`cdr,pairing,t4`，默认全跑），三阶段输出互不重叠；日志按阶段命名，避免并行互相覆盖。
- **效果**：拆成 2 ckpt × 3 阶段共 6 条单卡作业并行，墙钟从 ~3h40m 降到 ~1h45m（由最长的 pairing 决定），pairing 结果提前约 80 分钟。
- **已取消**：串行的 `t-20260825010632-krz7q` / `t-20260825010636-jvpsn`（各跑了约 32 分钟，仅完成 CDR-h1）。
- **8B pairing 首次提交被拒**：Volc `Description` 上限 500 可见字符，原文 528。缩短后重提成功。
- **resume 隐患已修**：`light_chain_pairing.py` 的 `_generation_signature` 只比对数据/ckpt/参数，不含代码版本，导致被取消的旧作业留下的 `.progress.pt`（泄露版代码生成的半成品）会被新作业当成可续跑。已删除残留文件，并新增 `GENERATION_PROTOCOL_VERSION=2` 进签名，协议或解码路径变更后旧 partial 自动失效。
- **旧产物隔离**：只修长度泄露、未修 encoder 泄露的那批 pairing 产物移到 `output/downstream_generation/_stale_pre_encoder_fix/`（含 README 说明为混合口径、不可用）。

> 2026-08-20 提交的六条 `eval-immune-*` repr/gen 作业已全部 Success，结果见 `downstream/benchmark/RESULTS.md` §0，已从本表移除。

### 2026-08-15 Ophiuchus-Ab official-ckpt eval Success

- `t-20260815220631-lmjr9` **Success**（14:06→14:47Z，elapsed 2465s），已从 Active 表移除。
- pairing 完成：OAS holdout500 prompt3 argmax，gen ImmunoMatch=`0.352`（ref `0.699`），chain-match=`0.998`，v-gene-family=`0.904`。产物 `output/downstream_generation/ophiuchus_ab/light_pairing_holdout500_prompt3_metrics.json`。
- SAbDab/SAb23H2 沿用先前 JSON（skip 未重跑）。本轮不含 humanization。

### 2026-07-21 MINT baseline local-fixed reruns

- **17:04Z submit**：MutationalPPI 全 8 个 Figure-2 模型的本地固定划分重跑已提交，task ID=`t-20260722010436-tsfqc`，初始状态 **Initialized**，`Preemptible: false`。使用完整 3,406 行、提交的 WT pair-group 10-fold、separate-chain embedding、WT/mutant 对齐 2048-token 窗口、3 次 640-hidden MLP；结果标 `[L]`，不声称为论文未公开 fold。
- **17:05Z submit**：SKEMPI 全 8 个 Figure-2 模型的本地固定划分重跑已提交，task ID=`t-20260722010520-6ph8f`，初始状态 **Initialized**，`Preemptible: false`。使用完整 6,706 行、提交的 notebook complex-held-out 三折、separate-chain embedding、WT/mutant 对齐 2048-token 窗口、3 次 640-hidden MLP；结果标 `[L]`，不声称为论文未公开 fold。
- **17:06Z status**：两个任务均已进入 **Queue**；`ml_task get` 显示资源队列 `q-20260121145036-6fztt`、单卡 `ml.pni2.3xlarge`。
- **17:31Z start / 17:43Z check**：MutationalPPI 已进入 **Running**，没有被挤占。MINT 首格已 Success 并写出 31 条记录（30 个 fold×seed evaluation + aggregate）：AUPRC=`0.676924±0.068255`、AUROC=`0.834633±0.041971`；当前正在抽取 ESM2-150M embedding。SKEMPI 仍为 **Queue**，尚无输出，因此当前不重提。
- **17:49Z progress**：MutationalPPI 已完成 **2/8**；ESM2-150M aggregate 结构完整（31 records，30 evaluations），AUPRC=`0.734887±0.081104`、AUROC=`0.868111`，当前进入 ESM2-650M。平台状态仍 Running，无挤占/失败；SKEMPI 仍 Queue。
- **17:57Z progress**：MutationalPPI 已完成 **3/8**；ESM2-650M aggregate 结构完整（31 records，30 evaluations），AUPRC=`0.729359±0.082382`、AUROC=`0.860677`，当前进入 ESM-1b。平台仍 Running；SKEMPI 仍 Queue。
- **17:58Z failure diagnosis**：MutationalPPI 原任务 `t-20260722010436-tsfqc` 在 ESM-1b 首次编码时 **Failed**，不是挤占。根因为 ESM-1b 使用 absolute positional embedding（本地 HF config `max_position_embeddings=1026`），2048 tokens 触发 CUDA gather index 越界；前三格 metrics/cache 完整保留。
- **18:02Z SKEMPI cancel**：SKEMPI 原任务刚进入 Running，但已在进程内载入相同旧配置，后续必然在 ESM-1b 失败；为避免浪费整轮，执行 cancel 成功，状态进入 **Killing**。runner 已修为全局上限 2048、ESM-1b 原生 1024，并将实际长度写入 provenance；12 tests passed。待分别重提 MutationalPPI 剩余 5 格和 SKEMPI 全 8 格。
- **18:01Z old SKEMPI terminal**：原任务 `t-20260722010520-6ph8f` 已为 **Killed**，从 Active 表移除。
- **18:03Z MutationalPPI retry submit**：仅剩余 5 格的 native-cap retry 已提交，task ID=`t-20260722020255-p92dd`，初始 **Initialized**、`Preemptible: false`；runner 会保留并识别前三格 success manifest，不重新训练它们。
- **18:03Z SKEMPI retry submit**：修正版全 8 格任务已提交，task ID=`t-20260722020320-l9grz`，初始 **Initialized**、`Preemptible: false`；全局 token cap=2048，仅 ESM-1b 按原生上限使用 1024，并逐模型写实际长度。
- **18:05Z retry status**：MutationalPPI retry 已 **Running**，SKEMPI retry 已进入 **Queue**；没有再次挤占或失败。
- **18:11Z retry progress**：MutationalPPI 已完成 **4/8**；ESM-1b 在原生 1024-token 上限下完成全部 10 folds × 3 repeats（31 records），AUPRC=`0.733634±0.073799`、AUROC=`0.863292±0.029926`，当前进入 ESM2-3B。平台任务保持 **Running**；SKEMPI 仍为 **Queue**，没有 Killed/Failed，暂不重提。
- **18:13Z SKEMPI start**：SKEMPI native-cap retry 已由 **Queue → Running**；fresh manifest `resumed_at=2026-07-21T18:10:44Z`，8 个目标模型完整，当前 MINT 为 running 且逐模型 provenance 已写 `max_sequence_tokens=2048`。没有挤占/失败。
- **18:24Z SKEMPI progress**：SKEMPI 已完成 **1/8**；MINT metrics 结构为 4 records（3 个 predefined complex-held-out folds + weighted aggregate），Pearson=`0.349086±0.190785`，总测试行数 6,706，当前进入 ESM2-150M。平台仍 **Running**，没有挤占/失败。
- **18:29Z SKEMPI progress**：SKEMPI 已完成 **2/8**；ESM2-150M 的 4-record aggregate Pearson=`0.349330±0.132447`，当前进入 ESM2-650M。MutationalPPI/ESM2-3B 已写出 23/30 个 fold×repeat evaluations；两个平台任务均保持 **Running**。
- **18:31Z MutationalPPI progress**：MutationalPPI 已完成 **5/8**；ESM2-3B aggregate 结构完整（31 records，30 evaluations），AUPRC=`0.751966±0.073092`、AUROC=`0.862089±0.036812`，当前进入 ProGen2-Large。SKEMPI 保持 2/8、ESM2-650M running；平台无挤占/失败。
- **18:38Z SKEMPI progress**：SKEMPI 已完成 **3/8**；ESM2-650M 的 4-record aggregate Pearson=`0.303452±0.165793`，当前进入 ESM-1b，并在命令/provenance 中使用原生上限 1024。MutationalPPI 正在持续下载 ProGen2-Large 5.56 GB 权重分片；两个平台任务均 **Running**。
- **18:50Z SKEMPI native-cap verified**：SKEMPI 已完成 **4/8**；ESM-1b 在 1024-token 原生上限下完成全部三折，4-record aggregate Pearson=`0.347601±0.139727`、总测试行数 6,706，未复发 CUDA gather 越界，当前进入 ESM2-3B。MutationalPPI/ProGen2-Large 权重仍持续下载；平台无挤占/失败。
- **18:53Z ProGen2 shared-cache prefetch**：ProGen2-Large 总权重约 5.56 GB、两分片；MutationalPPI 作业持续下载第一分片，同时从已激活的 `pllm` 环境启动第二分片的独立 Xet 预取。两分片使用不同 cache lock，不覆盖当前写入；完成后 MutationalPPI 与 SKEMPI 共用同一缓存。Xet 日志已确认持续收到 206 range responses，非卡死。
- **18:56Z MutationalPPI network failure**：retry `t-20260722020255-p92dd` 于 `18:56:28Z` **Failed**，不是资源挤占。ProGen2-Large 第一分片下载在约 2.27 GB 时发生 `requests.exceptions.ChunkedEncodingError / IncompleteRead`；manifest 完整保留前 **5/8** success，仅 ProGen2-Large 标 failed。先在 CPU/Xet 共享 cache 中断点补齐权重，再只重提 ProGen2-Large、ProtT5-UniRef、ProtT5-BFD，避免重复前五格及再次浪费 A100 下载时间。
- **19:11Z MutationalPPI remaining-3 submit**：ProGen2-Large 两个分片已完整落盘并核对总字节数 `5,558,740,304`；ProtT5-UniRef 已完整缓存，ProtT5-BFD PyTorch 权重正在共享 cache 高速预取且会在执行到该模型前完成。仅 ProGen2-Large、ProtT5-UniRef、ProtT5-BFD 的离线/非抢占 retry 已提交，task ID=`t-20260722031109-rkn88`，YAML=`eval_mint_mutationalppi_remaining3_preloaded_retry.yml`，`Preemptible: false`。
- **19:12Z remaining-3 status**：`t-20260722031109-rkn88` 已直接进入 **Running**；SKEMPI `t-20260722020320-l9grz` 同时保持 **Running**，当前没有资源挤占。
- **19:12Z remaining-3 tokenizer failure**：`t-20260722031109-rkn88` 于 `19:12:08Z` **Failed**，不是挤占。ProGen2 两个权重分片已在离线模式下成功载入，但随后 `AutoTokenizer` 发现共享 cache 缺少仓库内仅 1.6 KB 的 `tokenizer.json`；manifest 仍完整保留前 5/8 success。补齐 tokenizer 并用实际 eval 环境做 offline load 预检后，再重提相同 remaining-3。
- **19:15Z remaining-3 offline resubmit**：已补齐 ProGen2 `tokenizer.json` / `generation_config.json`，并用实际 `protenix_abtcr` 环境在 `HF_HUB_OFFLINE=1` 下验证 `GPT2TokenizerFast` 可加载和编码。相同 remaining-3 非抢占任务重新提交，task ID=`t-20260722031535-dvn5c`。
- **19:16Z offline retry status**：`t-20260722031535-dvn5c` 已直接进入 **Running**；SKEMPI 同时保持 **Running**，当前无资源挤占。ProtT5-BFD PyTorch 权重预取已到约 11.13 GB，接近完成。
- **19:16Z dual ProGen2 failure**：MutationalPPI `t-20260722031535-dvn5c` 于 `19:16:47Z`、SKEMPI `t-20260722020320-l9grz` 于 `19:16:30Z` 均 **Failed**，不是资源挤占。两边均成功离线加载 ProGen2 权重/tokenizer 并开始编码，随后在长度超过 1024 时由模型固定 causal mask 报维度不匹配（MutationalPPI `1024 vs 1084`；SKEMPI `1024 vs 2048`）。ProGen2 config 明确 `n_positions=1024`，因此应像 ESM-1b 一样使用原生 1024-token 上限。SKEMPI 在失败前已完成 **5/8**；ESM2-3B 4-record aggregate Pearson=`0.296480±0.137196`。两任务均保留前 5 格 success，待修正 ProGen2 cap 后仅重提各自 remaining-3。
- **19:23Z MutationalPPI dual-native-cap submit**：runner/汇总约束已把 ESM-1b 与 ProGen2-Large 都设为原生 1024，其余六模型保持 run cap 2048；12/12 tests passed。ProGen2 与两套 ProtT5 权重/tokenizer 均已离线缓存并按真实 tokenizer 路径预检。MutationalPPI remaining-3 非抢占任务已提交，task ID=`t-20260722032251-pd4qq`。
- **19:23Z SKEMPI dual-native-cap submit**：同一修正版的 SKEMPI remaining-3 非抢占任务已提交，task ID=`t-20260722032319-9wfwt`；会保留前 5/8 success，仅运行 ProGen2-Large、ProtT5-UniRef、ProtT5-BFD。
- **19:24Z dual-native-cap status**：MutationalPPI `t-20260722032251-pd4qq` 与 SKEMPI `t-20260722032319-9wfwt` 均已直接进入 **Running**，当前无资源挤占。
- **19:43Z MutationalPPI progress**：MutationalPPI 已完成 **6/8**；ProGen2-Large 在原生 1024-token 上限下完成全部 10 folds × 3 repeats（31 records），AUPRC=`0.714032±0.087962`、AUROC=`0.842711±0.041239`，当前进入 ProtT5-UniRef。SKEMPI/ProGen2-Large 仍 **Running**；没有挤占/失败。
- **19:50Z SKEMPI progress**：SKEMPI 已完成 **6/8**；ProGen2-Large 在原生 1024-token 上限下完成三折（4 records），Pearson=`0.273263±0.063060`、总测试行数 6,706，当前进入 ProtT5-UniRef。两任务现在都只剩 ProtT5-UniRef 与 ProtT5-BFD，平台均 **Running**。
- **19:53Z MutationalPPI progress**：MutationalPPI 已完成 **7/8**；ProtT5-UniRef aggregate 结构完整（31 records，30 evaluations），AUPRC=`0.726431±0.077810`、AUROC=`0.857915±0.036205`，最终项 ProtT5-BFD 已从完整离线 cache 加载并开始编码。SKEMPI/ProtT5-UniRef 仍 **Running**。
- **20:02Z SKEMPI progress**：SKEMPI 已完成 **7/8**；ProtT5-UniRef 4-record aggregate Pearson=`0.273646±0.168868`，最终项 ProtT5-BFD 已开始。MutationalPPI/ProtT5-BFD 已写出 22/30 个 fold×repeat evaluations；两个平台任务均 **Running**。
- **20:03Z MutationalPPI final**：remaining-3 retry `t-20260722032251-pd4qq` 已 **Success** 并从 Active 表移除；最终 manifest 为 `status=success`、**8/8** baseline success。ProtT5-BFD 完成 31 records（10 folds × 3 repeats + aggregate），AUPRC=`0.716531±0.081206`、AUROC=`0.849961±0.041317`，使用 2048-token cap。该任务没有发生资源挤占，也无需再提交；SKEMPI 最后一项继续 Running。
- **20:13Z SKEMPI final**：remaining-3 retry `t-20260722032319-9wfwt` 已 **Success** 并从 Active 表移除；最终 manifest 为 `status=success`、**8/8** baseline success。ProtT5-BFD 完成 4 records（三折 + aggregate），Pearson=`0.306710±0.088267`、总测试行数 6,706，使用 2048-token cap。汇总 audit 已达到 expected/observed=`16/16`、missing=`[]`；两个 MINT 本地重跑最终都没有因资源挤占终止，不需再提交。
- **20:19Z MINT final validation**：重新生成 `[P]/[L]` 混合表并逐文件验收：2 个 manifest 均 success，16/16 baseline metrics 齐全，共 280 records；MutationalPPI 每模型 31 records（10 folds × 3 repeats + aggregate），SKEMPI 每模型 4 records（三折 + aggregate）。data/evaluation protocol、cache tag、raw metric space 与模型 token cap 全部匹配；selected table=40 rows，其中 `[L]`=16、无重复/缺失；回归测试 **12/12 passed**。`downstream/benchmark/RESULTS.md` 与审计文档已回填最终数值。
- **协议复核**：论文 Figure 2 caption 要求所有任务三次 experimental repeats，而公开 MutationalPPI `cv` 分支忽略 `--rep`；本地 `[L]` 协议显式执行固定 10 folds × 3 MLP seeds，并记录该差异，不声称复刻未公开论文实现。
- 提交前一次平台 active-list query 未返回 pairing 作业；随后在 `17:05Z` 逐 task ID 直接查询确认两个 iter96 作业仍为 **Running**，并与本轮新增 iter32/64 作业一并恢复到 Active 表。以逐 task ID 的状态为准。

> 原补测范围的 40/40 个平台作业均已 Success；step359000 的 3 个 FLAb 内部缺失项也已由独立补测全部补齐。因此 step117000、step121000、step164000、step359000、step384800 均已形成完整汇总。按用户最新要求，新增 pairing ablation 只保留 canonical pairing 最好的两个早期 checkpoint：step121000 与 step189000；step384800/step389500 的新增对照均已停止。

**当前最终快照**：`output/grammar_v2_esmc300m_integrated_llada_7l_step389500/best.pt`；checkpoint 内 `step=389500`，size=`5227942507`，sha256=`5d1231992b65aa75a883e86662f34da136fa87e8ba031b804ff4c942ab1326c9`，manifest：`output/grammar_v2_esmc300m_integrated_llada_7l_step389500/checkpoint_manifest.json`。

**7L 旧 best.pt 下游 eval**：首轮 8 作业均已 **Success**（ids 见 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_7l_task_ids.tsv`）；Pdb-bind 补提 `t-20260712192011-z5r98` 也已 **Success**。

**7L step101300 latest.pt 下游 eval**：8/8 作业均 **Success**，无活跃 eval。最终汇总：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step101300.json`；latest-only 报告：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/STEP101300_DOWNSTREAM_REPORT.md`。

**7L step194100 latest.pt 下游 eval（已完成）**：

无活跃 eval。7 个原始任务作业 Success；FLAb 原 parent 被抢占后，已由保存的 `g6_Kd` + 3 个 Success 补提组成完整 4/4 结果。

## 2026-07-21 T1 external baseline 改为 original-only 来源选择

- **协议**：external baseline 有可运行 original checkpoint 时，只用官方 checkpoint + 官方推理代码 + `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ntmethod_binding/original.zip`；不加载 `train.csv`、不 retrain。不可运行时直接采用论文 original-model 数值，并标 **“论文值，未本地复现”**。
- **test-only 数据链**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/prepare_nm2025_binding.py --tests-only` 只展开 `original.zip`，生成 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_binding_nm2025/original_test_manifest.json`；每个 normalized test CSV 有独立 SHA256。实测运行前后 cdr3b/others 两个 train CSV 哈希不变。
- **代码**：新增 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/original_protocol.py`（15 个 paper method 的 checkpoint/wrapper catalog）；`run_official_baseline.py` 只读 test、校验 original.zip sha256、在 wrapper import 前绑定 TEINet/ERGO/TPBTE variant，并强制 wrapper 报告推理时真正打开的全部 model artifact 与 catalog 逐路径一致，再把 `retrained=false` / provenance 写入 metrics；`summarize_nm2025.py` 生成 canonical `summary_original_baselines.csv` 与 `summary_main.csv`，另留 `summary_local_runs.csv` diagnostic。
- **发现并修复**：ERGO 官方 featurizer 会原地把 peptide list 改为 token ids，旧 wrapper 随后把 token ids 写进 `group`；已在 featurize 前保存原始行标识，4 个 ERGO 变体的 AS seen/unseen 已用 original checkpoint 重跑并通过结构校验。此前 `--tag` 只改输出目录、不保证选择对应 checkpoint 的时序问题也已修复。
- **artifact 对账修复**：旧 metadata 对 TEIM 只写 bundle `TEIM.ckpt`，没有反映运行时还打开 `epi_ae.ckpt`；PanPep 也没有记录 zero-shot memory state。现在 TEIM 记录实际 `teim_seq.ckpt + epi_ae.ckpt`，PanPep 记录 `model.pt + Content_memory.pkl + Query.pkl`，ERGO-AE 记录 classifier + TCR autoencoder。TEIM 的 `teim_seq.ckpt` 与 bundle `TEIM.ckpt`、PanPep 的 `model.pt` 与 bundle `PanPep.pt` 分别 sha256 完全一致；受影响模型已重跑。
- **当前选择**：15 个论文 original-model baseline 中 13 个已用新 runner 完成 AS seen/unseen 官方 rerun，全部 metrics 均内嵌 checkpoint path、实际 variant、original.zip checksum 与 `retrained=false`；SETE 官方 pickle 在 seen/AS 报输入 433 features、模型期待 1 feature，TEPCAM 尚无可运行 wrapper，二者使用论文 fallback。SETE 论文没有 unseen 值，保持空缺，不补造。
- **验证**：`protenix_abtcr` 环境两份测试 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/test_baseline_protocols.py` + `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/test_nm2025_metrics.py` = **15 passed**；strict provenance audit 为 0 error / 0 warning。`pllm` 环境 py_compile 通过，但该环境缺 `scikit-learn`，完整协议测试无法在其中 import 既有 metrics 模块。

## 2026-07-21 T1 retrained checkpoint 五折复现审计（不进入 original-only 主表）

- **官方 artifact**：Figshare DOI `10.6084/m9.figshare.27020455` 的 `retrain.zip`（20,005,661 bytes，MD5 `5cf77befd7a07e0cb359540f050fb7b4`）和 `Retraining_model.zip`（13,912,594,261 bytes，MD5 `aa6ea6c175738675692848d036c2244c`）已完整下载并严格校验。数据协议固定为 CDR3β-only / AS、5 folds；每 fold 评 `seen_test`、共享 `seen_independent`、共享 `unseen_independent`，headline AUPRC 使用 R `precrec::evalmod` 默认兼容实现而非 sklearn AP。
- **已完成 8 个模型、120 个 fold×eval 推理**：ATM-TCR、NetTCR、epiTCR、TEIM、TCR-H、TEINet、ERGO-AE、ERGO-lstm。入口为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_retrained_baseline.py`；每个 `metrics.json` 记录 archive/checkpoint/test SHA、官方代码路径、环境、兼容修复和是否读取 train。没有模型参数拟合。
- **unseen 对论文（AUROC/AUPRC）**：ATM-TCR `0.519152/0.513599`、NetTCR `0.520259/0.515647`、epiTCR `0.507540/0.510554`、TCR-H `0.526935/0.526691`、ERGO-AE `0.509877/0.510554`、ERGO-lstm `0.500926/0.504867` 均同时 round 到论文四位小数；TEIM 为 `0.533745/0.523310`（论文 `0.5326/0.5218`），TEINet 为 `0.496934/0.499520`（论文 `0.5025/0.5057`）。
- **seen artifact 系统偏移**：8/8 模型的 seen 格都没有同时复现论文四位小数。除 TEINet/近随机 ERGO-lstm 外，`seen_test` 通常较论文低约 `0.008–0.013`，`seen_independent` 通常较论文高约 `0.003–0.012`；同一批 checkpoint 在 unseen 却有 6/8 精确复现，证据更支持发布 seen CSV 与论文内部 seen 数据版本不一致，而不是 checkpoint/metric 配对错误。
- **两项官方协议特例**：TCR-H 推理必须读取对应 fold train CSV 重建正相关 `>0.8` 的描述符过滤列，204 维恰好降为 checkpoint 要求的 130 维；这只做特征选择，不调用 `fit`。ERGO-AE 官方 `ae_utils.get_batches(batch=50)` 丢弃尾部不足 50 行，因此 unseen 每 fold 明确只评 `3150/3162` 行；ERGO-lstm 不丢尾批。两者均在 provenance 中显式记录。
- **结果来源选择**：统一表 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/comparison_AS.csv`（及 JSON）同时保留 local/paper/delta；只有 AUROC 与 AUPRC 均 round 到论文四位小数时选择本地值，否则选择论文值并标 **“论文值，发布 artifact 未精确复现”**。这张表是 retrained artifact 诊断，不替换 `summary_original_baselines.csv` 的 canonical original-only 决策。
- **验证与下一批**：并行 TCR-H 分块推理与单批 3,162 个概率逐值完全相同（max abs delta `0`）；协议/metrics 测试现为 **22 passed**。`Retraining_model.zip` 共 24 个模型目录，剩余 16 个按各自 artifact 形态专项接入；SETE 已确认是 fold×epitope 多 pickle 且论文无 unseen 行，不能错误套用单-checkpoint runner。

## 2026-07-20 Submitted full downstream eval for 7L step338600 latest checkpoint

- **Checkpoint cutoff**：固定 `output/grammar_v2_esmc300m_integrated_llada_7l/latest.pt` 为不可变 `output/grammar_v2_esmc300m_integrated_llada_7l_step338600/best.pt`；checkpoint 内 `step=338600`，sha256=`6c3f47862ad9543f8e0af39e623fb6ad674f6234eb44f48cad911e82f1c95f13`。
- **Submit**：T1–T4、MINT、AB、FLAb、NbBench 共 8 个单卡 `ml.pni2.3xlarge`、`Preemptible: true` 作业；提交后核对 8/8 均为 **Running**。
- **Task records**：`output/downstream_generation/eval_step338600_task_ids.tsv`；提交日志 `output/downstream_generation/eval_step338600_submit.log`。
- **评测修复已生效**：T4 使用 run-specific tag，并在生成后显式重算 Setting A/B/C metrics；MINT collector 覆盖六种任务 schema，不再复用旧 metrics 或漏掉回归任务。
- **08:41Z progress**：T1/T2/T3 已 **Success**；step338600 指标分别为 T1 seen AUPRC/AUROC=`0.5378/0.5093`、unseen=`0.5128/0.4985`；T2 ARI/NMI/Purity=`0.0154/0.1069/0.2857`；T3 k5/20/100/200 AUROC=`0.6350/0.6631/0.6922/0.7026`。T4/MINT/AB/FLAb/NbBench 仍 Running。
- **08:42Z progress**：T4 **Success**（elapsed 468s）；Setting-A JSD/novelty/PGen-positive=`0.3514/1.0000/0.8917`；Setting-B overall F1/d_edit/seq-recovery/diversity=`0.0000/7.6803/0.3807/1.0000`；Setting-C valid-AA/CDR3-extracted=`1.0000/0.0000`。本轮确认使用 step338600 run-specific 新生成样本及新 metrics，不再读取旧 tag。
- **汇总修复**：并发单族 collector 改为各写 `summary_<run>_<task>.json`，最终再生成无后缀全量 summary，避免并发覆盖；FLAb aggregate 主指标改为 nested 10×5-fold outer mean R²（Spearman 仅作 artifact 诊断）。
- **报告修复**：通用 fill report 修正 `Pdb-bind` 大小写，并覆盖 MINT 分类/回归全部 canonical metrics；FLAb 标题与 aggregate 同步为 nested-CV mean R²。
- **FLAb 调度修复**：发现全量编排仍直连单进程 `run_flab_baselines.py`，而 step194100 补跑所用 8-worker wrapper 未回接；已将主编排改接 `run_flab_parallel.py`。协议不变，仅并行 inner-CV fits。step338600 慢 parent 将在无结果产出时停止，并由 4 个数据集独立并行补跑。
- **08:58Z FLAb cancel**：慢 parent `t-20260720163454-hqd7p` 运行约 23 分钟仍未产出首个 JSON；平台已返回 `cancel success`（状态仍在 Running→Killing 过渡）。取消发生在任何 step338600 FLAb 结果落盘之前，不会混合半成品。
- **08:58Z FLAb fast split submit**：4 个独立、8-worker、同协议作业提交成功：g6_Kd=`t-20260720165824-dnx28`，g6_er=`t-20260720165823-h8cf9`，trastuzumab_kd=`t-20260720165823-cj7nh`，d44_Kd=`t-20260720165823-mk5ph`。记录：`output/downstream_generation/eval_step338600_flab_parallel_task_ids.tsv`。
- **08:59Z FLAb split status**：慢 parent 已 **Killed**（无结果文件）；trastuzumab_kd/d44_Kd 已 **Running**，g6_Kd/g6_er 为 **Queue**。
- **09:00Z FLAb split status**：g6_er 已进入 **Running**；仅 g6_Kd 仍 Queue。
- **09:02Z MINT final**：`t-20260720163446-np8gz` **Success**（elapsed 1604s），六任务/22 个 canonical 指标齐全且有限；HumanPPI AUROC/AUPRC=`0.7197/0.7339`，Bernett=`0.5614/0.5661`，YeastPPI=`0.5689/0.5869`，MutationalPPI=`0.5932/0.1670`，SKEMPI Pearson/Spearman/RMSE=`0.3958/0.2887/1.8729`，Pdb-bind=`0.6372/0.6338/1.4901`。两项回归 artifact 均标记 `metric_space=raw_target_units`，日志无失败模式。
- **09:04Z FLAb split status**：g6_Kd 也已进入 **Running**，4/4 fast splits 全部在算。
- **09:09Z NbBench final**：`t-20260720163457-7plqr` **Success**（elapsed 1992s）；9 scalar + 3 residue + generative CDR infilling 共 15 个 canonical 指标齐全且有限。nanobody-type acc=`0.9935`，hIL6 AUROC=`0.8806`，Paratope AUPRC=`0.4681`，生成式 masked EM/BR/exact-seq=`0.2847/0.3730/0.0000`；日志无失败模式（存在 sklearn 收敛/病态矩阵 warning，未导致步骤失败）。
- **09:11Z FLAb preemption**：4 个 preemptible split 均被平台置为 **Killed**（g6_Kd/g6_er/d44 elapsed 632s，trastuzumab elapsed 711s）；日志无 Traceback/OOM，且 4 个目标 JSON 均不存在。YAML 已切换 `Preemptible: false`，保留 8-worker 同协议配置，准备稳定重提。
- **09:12Z FLAb stable submit**：4 个非抢占 stable split 提交成功：g6_Kd=`t-20260720171141-bm9wj`，g6_er=`t-20260720171141-mx2kl`，trastuzumab_kd=`t-20260720171141-qxhjv`，d44_Kd=`t-20260720171141-r24pr`。记录：`output/downstream_generation/eval_step338600_flab_stable_task_ids.tsv`。
- **10:22Z FLAb runtime cache**：`run_flab_baselines.py` 为每个 outer fold 的 sklearn Pipeline 启用临时 joblib cache，13 个 Ridge α 复用同一 fold-local `PowerTransformer` 结果；outer/inner folds、预处理拟合边界、α 网格与 R² 计算均不变。该改动会在排队中的 step338600/step189000 FLAb 作业启动时生效。
- **10:28Z AB resumable generation**：`downstream/grammar/light_chain_pairing.py` 改为每个 heavy batch 原子保存生成行和 Python/NumPy/Torch/CUDA RNG 状态；同配置重启可从精确 batch/RNG 状态继续。最终 `_n8.csv` 仍只在 500×8 全部完成后原子发布，避免上轮在 116/125 batches 被抢占时所有生成进度丢失，也不会让 collector 误读半成品。
- **09:13Z AB preemption**：AB parent `t-20260720163450-dhqgc` 同轮被平台置为 **Killed**（elapsed 2125s）。CDR-H1/H2/H3 均已完整 exit=0 并落盘；light pairing 停在 116/125，尚无 metrics JSON。已准备仅 pairing 的 non-preemptible retry；同时 YAML 生成器将未来 AB/FLAb 默认设为 non-preemptible。
- **09:14Z AB pairing stable submit**：仅补 light pairing 的 non-preemptible 作业已提交，ID=`t-20260720171326-g4vp2`；不重复已完成的三个 CDR。记录：`output/downstream_generation/eval_step338600_ab_pairing_stable_task_id.tsv`。
- **10:07Z AB pairing batch8 fast submit**：stable 作业持续 Queue，另提交 protocol-equivalent、独立输出的 preemptible 快速重试 `t-20260720180726-2766f`；只把 `heavy_batch_size=4` 调为 `8`，目标是在典型抢占窗口内完成 500-heavy / n8 / prompt3 / max_iter32 全量生成与评分。
- **09:14Z stable queue status**：AB pairing 与 4 个 FLAb non-preemptible 作业均已进入 **Queue**。
- **10:53Z partial aggregate**：step338600 已完整汇总 T1/T2/T3/T4/MINT/NbBench 6 个任务族，AB 三个 CDR 完成、pairing 仍 Queue，FLAb 4/4 stable splits 仍 Queue。机器可读结果为 `output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step338600.json`，阶段报告为 `output/downstream_generation/STEP338600_DOWNSTREAM_REPORT.md`；在 pairing/FLAb 终态前明确标记 partial，不作 8/8 完成声明。协议自检：baseline/NM2025 pytest 7 passed；TCRT5 gold tests 全通过且 42 个上游 prediction blocks mismatch=0。
- **2026-07-21 04:23Z final reconciliation**：4 个 FLAb stable 作业与 AB pairing stable 作业均为 Success；AB batch8 fast 备份作业为 Killed。已重新 collect 为 8/8 完整 summary。AB H1/H2/H3 AAR=`38.3563/37.7398/33.6738`，pairing ImmunoMatch=`0.000403`；FLAb g6_Kd/g6_er/trastuzumab_kd/d44_Kd nested-CV mean R²=`0.2119/0.5106/0.1959/0.1849`。

## 2026-07-17 Submitted full downstream eval for 7L step194100 latest checkpoint

- **Checkpoint cutoff**：在训练继续写 checkpoint 时，将当时最新 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l/latest.pt` 固定为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step194100/best.pt`；checkpoint 内 `step=194100`，size=`5227942507`，sha256=`5207d1750aae7408fb9fcafec7ace1bc20d46267407db154b937f9f05cdbc2f3`。源 `latest.pt` 随后已原子替换为新 inode，快照保持稳定。manifest：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step194100/checkpoint_manifest.json`。
- **Submit**：T1–T4、MINT、AB、FLAb、NbBench 共 8 个单卡 `ml.pni2.3xlarge`、`Preemptible: true` 作业；2026-07-17T08:06Z 核对均为 **Running**。
- **Task IDs**：t1 `h5gtx`；t2 `j7t6j`；t3 `6m9kj`；t4 `x865k`；mint `mrp9v`；ab `tt8s4`；flab `chvq6`；nbbench `fc499`（完整 ID 见 `eval_step194100_task_ids.tsv`）。
- **YAML / records**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_integrated_step194100_{t1,t2,t3,t4,mint,ab,flab,nbbench}.yml`；`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_step194100_task_ids.tsv`；`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_step194100_submit.log`。
- **08:08Z progress**：t2 `t-20260717160455-j7t6j` **Success**（elapsed 171s）；step194100 Basis-B `ARI=0.0213 / NMI=0.1247 / Purity=0.3012`。
- **08:09Z progress**：t1 `t-20260717160448-h5gtx` **Success**（elapsed 238s）；seen AUPRC/AUROC=`0.5447/0.5024`，unseen=`0.5232/0.5266`。
- **08:10Z progress**：t3 `t-20260717160500-6m9kj` **Success**（elapsed 252s）；few-shot AUROC k5/20/100/200=`0.6388/0.6606/0.6860/0.6970`。
- **08:13Z progress**：t4 `t-20260717160504-x865k` **Success**（elapsed 475s）；Setting-A JSD/novelty/PGen-positive=`0.3333/1.0000/0.9957`；Setting-B overall F1/d_edit/seq-recovery/diversity=`0.0000/6.8538/0.4465/1.0000`；Setting-C valid-AA/CDR3-extracted/extracted-JSD=`1.0000/0.0020/0.9981`。
- **08:32Z progress**：mint `t-20260717160509-mrp9v` **Success**（elapsed 1607s），六个数据集齐全；HumanPPI AUROC/AUPRC=`0.7353/0.7225`；Bernett=`0.5764/0.5851`；YeastPPI=`0.5846/0.5990`；MutationalPPI=`0.5901/0.1668`；SKEMPI Pearson/Spearman/RMSE=`0.5093/0.4152/1.7542`；Pdb-bind=`0.6390/0.6350/1.4824`。
- **08:33Z progress**：nbbench `t-20260717160524-fc499` **Success**（elapsed 1675s）；9 个 scalar + 2 个 residue + generative CDR infilling 全部落盘。nanobody-type acc=`0.9938`，hIL6 AUROC=`0.9337`，Paratope AUPRC=`0.6184`，生成式 masked EM/BR/exact-seq=`0.5671/2.7401/0.0007`。
- **09:10Z progress**：ab `t-20260717160514-tt8s4` **Success**（elapsed 3888s）；CDR-H1/H2/H3 all-fold AAR=`78.5418/72.8338/56.0415`；light-pairing generated ImmunoMatch=`0.592719`；全部子步骤 exit=0。
- **11:18Z FLAb interruption**：原 flab `t-20260717160519-chvq6` 在 elapsed 11567s 后被平台置为 **Killed**；日志无 Traceback/ERROR，作业原配置 `Preemptible: true`。终止前已完整保存 `g6_Kd`：nested-CV mean R²=`0.251790`，pooled Spearman=`0.539861`；`g6_er` 尚未完成，另外两项未启动。
- **16:50Z FLAb retries**：将缺失 `g6_er / trastuzumab_kd / d44_Kd` 拆成 3 个单数据集、`Preemptible: false` 作业并行补提，仍只加载 step194100 固定快照，协议不变；IDs=`t-20260718005041-45thj / t-20260718005041-2w5pp / t-20260718005041-l42j6`，初查均为 **Initialized**。记录：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_step194100_flab_retry_task_ids.tsv`。
- **16:52Z retry status**：3 个补提任务均已从 Initialized 进入 **Queue**。
- **17:52Z retry reschedule**：3 个非抢占补提在 Queue 等待约 1h 后仍未获资源，均在开始计算前 cancel 成功；切换为相同 10×5 nested-CV、8 个 joblib worker 的执行加速版后重新提交。数据、split、PowerTransformer、13-alpha Ridge 网格与指标定义不变，仅并行执行 inner-CV fits。
- **17:53Z fast retry submit**：加速补提 IDs=`t-20260718015253-css55 / t-20260718015253-j7zpm / t-20260718015253-smshp`，均为 `Preemptible: true`，初查 **Initialized**；入口已在实际 `protenix_abtcr` 环境验证 joblib worker 生效。
- **17:54Z fast retry status**：`trastuzumab_kd` / `d44_Kd` 已为 **Running**；`g6_er` 为 **Queue**。
- **18:08Z fast retry status**：`g6_er` 也已进入 **Running**，3 个缺失数据集全部在算。
- **18:16Z fast retry progress**：`trastuzumab_kd` `t-20260718015253-j7zpm` **Success**（elapsed 1376s，含排队）；nested-CV mean R²=`0.131734`，pooled Spearman/Pearson=`0.461915/0.447383`，日志无错误。
- **18:17Z fast retry progress**：`d44_Kd` `t-20260718015253-smshp` **Success**（elapsed 1439s，含排队）；nested-CV mean R²=`0.223824`，pooled Spearman/Pearson=`0.482818/0.484176`，日志无错误。
- **18:33Z FLAb final**：`g6_er` `t-20260718015253-css55` **Success**（elapsed 2346s，含排队）；nested-CV mean R²=`0.512299`，pooled Spearman/Pearson=`0.713532/0.719987`。FLAb 4/4 齐全；step194100 全部 8 个任务族均已有结果。
- **18:35Z final verification**：全量 collector 已重写 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step194100.json`，包含 8 组 / 87 个有限数值字段；checkpoint SHA 与 manifest 一致；成功任务及补提日志无 Traceback/ERROR/OOM。latest-only 报告：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/STEP194100_DOWNSTREAM_REPORT.md`。


## 2026-07-10 取消整合版 resume 排队

- **操作**：cancel `t-20260710122856-55nmb`（Status→Killed）；watchdog loop 已停，`llada_train_watchdog_jobs.json` 设 `paused=true`。
- **保留**：`output/grammar_v2_esmc300m_integrated_llada/{latest,best}.pt`（~step 4000，val≈0.75）。
- **Active 表**：无活跃训练任务。

## 2026-07-10 Submitted integrated LLaDA 7L / 3-node on queue012

- **操作**：用户自行 submit 成功（本机 AK 此前无 CreateCustomTask；`queue006` 曾 Stopped）。
- **task_id**：`t-20260710203518-skfm7`
- **JobName**：`bioseq_esmc300m_integrated_llada_7l_3node_q012`
- **YAML**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_integrated_llada_queue006.yml`
- **队列**：`queue012`（`q-20260524172355-rnqtf`），creator=`zhuyiheng`
- **初始状态**：`Queue`；`Preemptible: false`
- **配置**：LLaDA `--num-hidden-layers 7`；3×`ml.pni2.28xlarge`=24 GPU；`bs=6 ga=1` → global batch **144**；from scratch；output=`output/grammar_v2_esmc300m_integrated_llada_7l/`（与 28L ckpt 隔离）

## 2026-07-10 启动 7L 训练状态轮询（每 10 分钟）

- **脚本**：`bash scripts/watch_llada_train_jobs.sh --daemon 10m`（pid 写入 `scripts/logs/llada_train_watchdog.loop.pid`）。
- **配置**：`scripts/llada_train_watchdog_jobs.json` 跟踪 `integrated_7l_3node_q012`；**`monitor_only=false`**（Killed/Failed → resume/scratch）。
- **Resume YAML**：`train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_integrated_llada_7l_resume_queue012.yml`（`--resume auto`）。
- **Scratch YAML**：`train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_integrated_llada_queue006.yml`。
- **日志**：`scripts/logs/llada_train_watchdog.log`。
- **volc wrapper**：`scripts/volc-no-proxy.sh`（清代理后调 volc）。

## 2026-07-14 Resumed 7L training after Failed

- **旧任务**：`t-20260710203518-skfm7` **Failed**（elapsed ~75.3h）；`latest.pt` step **101300**。
- **watchdog**：曾 `RESUBMIT_FAIL`（当时 AK 无提交权限）。
- **操作**：重新 submit resume YAML 成功。
- **新 task_id**：`t-20260714101935-v68kf`，JobName=`bioseq_esmc300m_integrated_llada_7l_3node_q012_resume`，初始 **Initialized**，`queue012`，`Preemptible: false`，creator=`zhuyiheng`。
- **YAML**：`train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_integrated_llada_7l_resume_queue012.yml`（`--resume auto` from `latest.pt`）。
- **watchdog**：`llada_train_watchdog_jobs.json` 已更新为新 task_id；daemon 已重启。

## 2026-07-15 Cancel duplicate 7L resume Queues; keep earliest

- **保留**：`t-20260713235356-8t6c9`（最早 resume，仍 Queue）。
- **cancel**：`t-20260714000400-6mzgl`、`t-20260714001404-bw2c4`、`t-20260714101935-v68kf` → 均 **Killed**。
- **watchdog**：改跟踪 `8t6c9`。

## 2026-07-09 整合版抢占轮询（每 10 分钟）

- **脚本**：`scripts/watch_llada_train_jobs.sh --daemon 10m`（后台；pid `scripts/logs/llada_train_watchdog.loop.pid`）。
- **配置**：`scripts/llada_train_watchdog_jobs.json` 跟踪 `integrated_nomint_bs8` → 当前 `t-20260710010623-q9rb2`。
- **Resume YAML**：`train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_integrated_llada_resume.yml`（`--resume auto`，要求 `latest.pt`；与 scratch 同 7 源 / bs8 / 16GPU）。
- **策略**：`Killed`/`Failed` + 有 `latest.pt` → resume；无 ckpt → 重提 scratch；`Success` 只记 DONE 不重提。

## 2026-07-09 整合版去 MINT + bs8 重提（16 卡 global batch 128）

- **改动**：去掉预训练源 `mint_ppi`、`mint_actions`（MINT 仅作下游评测）；manifest 重写为 **7 源**（weight sum 12.50；share：oas/ots 24%、nanobody 16%、tcr_piste/tcr_pmhc_fulllength 12%、ppi 8%、neutralization 4%）。
- **Batch**：`--batch-size 8 --grad-accum 1` × 16 GPU → **global batch 128**（试打满 A100-80G；此前 bs4 仅 46GB/79GB）。若 OOM 再退回 bs4 ga2。
- **操作**：cancel Queue 中的 `t-20260710010211-84d94`（仍含 MINT、bs4）→ 重提 `t-20260710010623-q9rb2`（Queue，Preemptible，save100）。

## 2026-07-09 整合版训练被抢占 → 重提（save-interval 1000→100）

- **终态**：`t-20260709190944-2z4xv` **Killed**（闲时抢占），跑到 ~step 988、`samples/s≈35`、`mem_peak=46GB`；`save-interval=1000` 故 **无 latest.pt/best.pt**。
- **速度**：相对 cmp500k 基线（~34–40 samples/s）**未提升**；YAML 未开 `--encoder-use-flash-attn`（镜像无 `flash_attn`，开了也是 no-op）；`wait_s` mean≈1.83s（数据侧仍偏慢）。
- **操作**：归档 `output/..._killed_step988_*`；YAML `--save-interval 100`；from-scratch 重提 `t-20260710010211-84d94`（后被 q9rb2 取代）。

## 2026-07-09 C1 eval 重提（修正 deadline + 7d 等 ckpt）

- **原因**：首版 eval YAML 的 `ActiveDeadlineSeconds`（flab 12h、t1–t3 24h）短于 48h 等 ckpt 轮询，训练仍在 Queue 时 eval 会先被平台杀掉。
- **改动**：`gen_eval_integrated_ymls.sh` — 等 ckpt 10080×60s（7 天）；deadline 8–10 天（按任务族）。
- **操作**：cancel 旧 8 作业（`8n66j`…`k2fzh`）→ 重提新 8 作业（`wf4cx`…`fl44v`），Preemptible，1×GPU。

`t-20260709024845-fh2p9`（8 源、pre-B1）已终态（`ml_task list -n bioseq` 无活动任务），断点 step 3000（val 1.4338）；其 output 已归档到 `output/grammar_v2_esmc300m_integrated_llada_8src_pre_b1/`，被 2z4xv 从头取代。
Cancelled 2026-07-09T02:48Z（用户要求改 16 卡两节点）: `t-20260709010716-krqjz` (8 GPU 单节点 Queue)。
Cancelled earlier: `t-20260707102700-cq4r9`（300M 非闲时 Running → 用户要求全改闲时）。
Cancelled/superseded earlier: `t-20260705173643-cjtdl` / `t-20260705175135-ghb4z` (prior round Failed/Killed), `dbvwc` (old 300M resume), `t-20260705140936-vsrff` (old 600M), `t-20260704173749-jgxw9` (old MINT). All restarted from latest.pt with nw1.

## Live Progress — LLaDA backbone

| Item | Status | Notes |
|------|--------|-------|
| LLaDA ESMC-300M train (OAS mix) | **Resubmitted (Queue, 闲时)** | cancel 非闲时 `cq4r9` → 闲时 `gzswj`，save100 |
| LLaDA ESMC-600M train (OAS mix) | **Queue (闲时)** | `d4hqx`，save100 |
| LLaDA ESMC-300M MINT-only | **Queue (闲时)** | `9x4hq`，save100 |
| Downstream AB eval | **Success** | 300M/600M CDR + pairing metrics ✅（7-05 best.pt） |

## 2026-07-08 论文下游对比表补全（对齐参考文献口径，仅补表不跑实验）

- **动机**：`paper/sections/4_experiments.tex` 的下游对比表覆盖度明显窄于 `downstream/benchmark/RESULTS.md`（已跑但没进论文）与锚定文献原文。用户要求对照参考文献完善下游比较（tables-only，全 10 任务）。
- **改动文件**：`paper/sections/4_experiments.tex`、`paper/sections/A_appendix.tex`、`paper/references.bib`、`paper/math_commands.tex`（新增 `\OursCkptB`/`\OursCkptM` 宏）、`paper/main.tex`（加 `placeins` + `\FloatBarrier` 修 Fig1 浮动到末页）。数字全部取自 `outputs/` 与 RESULTS.md，无新实验。
- **正文补全**：T1 主表 Ours 扩 3 ckpt；T2 新增 NAR-GAB 官方 9 方法 Purity/Retention 表（Basis A）+ TCREmbedding 表 Ours 扩 3 ckpt；T3 deep 表加 k=1 列 + Ours 3 ckpt；T4 补 ER-Transformer/bioseq_unseen 行 + 新增 held20、Setting C 全长表；MINT 分类表 Ours 扩 3 ckpt、回归表 Ours 扩 3 ckpt；抗体 pairing 改 prompt3-对-prompt3 公平对照、FLAb 补 One-hot/ProtBERT；NbBench 正文扩为 官方ESM2-150M/官方最佳/Ours 对照。
- **附录新增**（A.4–A.8）：T1 Tier-1 八官方变体 + AS/PS/HS 负样本 + 近似去重；T3 broad 24-epitope + linear-probe；MINT 4 PLM baseline（ProtT5-XL/ESM-1b/ESM2-3B/ProGen2）+ Acc/F1；抗体 de-novo pairing；NbBench 11 模型×任务全榜（`outputs/external/nbbench_official.csv` 转录 + 本地 pipeline 行）。
- **编译**：本机无 texlive；新建 conda env `tex`（`conda/envs/tex`，tectonic 0.16.9，需代理拉 bundle）。`tectonic -X compile main.tex` 成功，BibTeX error 0（顺手修 references.bib 中 nathan2025tcrbench/feng2025tcrembedding 作者字段的空名/尾逗号），仅剩 1 处 2.3pt overfull（可忽略）。
- **2026-07-09 全表逐格核对 outputs（用户要求"确定对应得上"）**：写脚本把论文每格数字与 `outputs/` 原始文件比对（T1 主表+Tier-1+neg-source+near-dedup、T2 Basis A/B、T3 deep/broad/probe、T4 A/B/C+held20、MINT cls/reg/PLM/AccF1、抗体 CDR/pairing(prompt3+denovo)/FLAb、NbBench 主表+全榜）。**除 1 处外全部精确吻合**。修正项：T3 linear-probe 辅表（tab:t3-probe）的 Ours-esmc300m 行原为 `0.825/0.451/0.438`（沿用 RESULTS.md 该行，疑似串了 Ophiuchus/旧 ckpt 值），改回原始 `outputs/tcr_representation/_summary.json` 的 post-LLaDA globalfeat esmc300m 真值 `0.793/0.398/0.412`，并把 probe-AUROC/Acc 的加粗改到 Ophiuchus（0.825/0.452）。注：RESULTS.md 该辅表行同样偏差（stale），但 probe 表为"向后兼容辅报"，本次只保证论文↔原始文件一致。文献校准（SCEPTR Table SI 逐 pMHC Δ≈0.006–0.024、NAR-GAB 6/9 Purity/Retention、NbBench 官方 CSV 逐字转录、TCRT5 42-block 0 mismatch）均由结果文件佐证。
- **2026-07-09 T2-B 补三条手工/理化嵌入基线（自跑，非转录）**：TCREmbedding 原文 19 方法里缺的对手中，只有"训练-free 手工嵌入"能在本地按同协议诚实自跑（其深度方法仓库无代码、catELMo 依赖冲突见下）。`common/featurizers.py` 新增 `AtchleyFeaturizer`（Atchley 2005 PNAS Table 2 五因子逐字转录，mean+std→10 维）/`Blosum62Featurizer`（Biopython BLOSUM62 行向量 mean→20 维）/`OneHotFeaturizer`（AA 组成 mean→20 维）；`tcr_clustering/run_embed_bench.py::_build_backend` 注册 `atchley/blosum62/onehot`。9,033 CDR3β/25 表位、K∈{10..100}、kmeans+hierarchical 跑通（各 ~秒级，CPU）。**K-means mean-over-K（ARI/NMI/Purity）**：onehot 0.007/0.064/0.254、blosum62 0.004/0.056/0.240、atchley 0.002/0.047/0.233。结论**部分**印证 Feng et al. "手工嵌入≥数据驱动 PLM"：组成类（kmer 0.010/0.079/0.268、onehot）追平/略胜通用 PLM（ESM2 0.007/0.068/0.258、ProtBERT 0.006/0.061/0.248），描述子类（BLOSUM/Atchley）反而偏弱；只有专用 TCR 模型（SCEPTR 0.033/0.159/0.339、TCR-BERT 0.016/0.095/0.287）明显领先——即决定因素是"是否 TCR 专用"而非"手工 vs 学习"。论文 `tab:t2` 加这 3 行 + 正文段说明；`outputs/tcr_clustering_embed/_summary.csv` 已重生成；tectonic 重编译 **13 页**、BibTeX 0 错、预览 `paper/preview/main_p01–13.png`（页数由 12→13：正文 T2 段增内容）。
- **2026-07-09 缺口可行性核查 + 硬阻塞归档（用户"其余缺口我来补"）**：逐项核查后确认以下缺口**本地无法诚实补齐**，均保留为 future work，不编数字：(1) **Ophiuchus 原文 Table 2/3/4 逐格数值**——本地缓存全文（agent-tools md）表格是图片、无数字；biorxiv 全文 HTML/PDF 与 Zenodo 页 WebFetch/curl 均超时（本机无代理直连不通，GitHub 可通但 README 无数值、Zenodo API 只挂 800M `.ckpt`），无处诚实转录；论文相应表已注明"协议不同/未转录"，维持不动。(2) **GDPa1 developability**——本地无 GDPa1 assay 标签（Ophiuchus repo `examples/GDPa1_v1.2_20250814.csv` 只有 247 行 VH/VL 序列、无 developability 测量列），无法跑 ridge 回归。(3) **Desautels m396 结合亲和**——`data/downstream/in_silico/Desautels_insilico_data.csv` 软链接目标已失效（文件丢失），`finetune_in_silico.py` 无数据可跑。(4) **dyMEAN/IgGM（CDR infilling 结构基线）**——需各自 repo/权重 + 预测结构，无代理无法拉取。(5) **catELMo + TCREmbedding 另 ~14 个深度嵌入**——TCREmbedding 仓库只含数据集（`TCRantigenData_unique_test.csv`）不含方法实现；catELMo GitHub clone 超时且 allennlp 2.10↔torch 2.8 冲突（沿用既有 blocker）。(6) **T1 NM2025 剩余 ~44 模型**——用户已明确跳过。
- **2026-07-08 收敛到单 ckpt（用户要求）**：每张表的"我们的模型"只保留 esmc300m（`grammar_v2_esmc300m_cmp500k_llada`，当前在训的稳定 headline），删除全部 esmc600m / esmc300m-mint 行/列（含 MINT 分类表收回单列、回归表单行、抗体 CDR/pairing/FLAb 去 600m、NbBench 全榜去 600m/mint、附录 T1-neg/T3-broad/T3-probe/mint-accf1/ab-pair-denovo）。因删行重算加粗（CDR-H3→esmc300m 44.68、FLAb g6Kd→Ophiuchus 0.616、T3-probe probe-Acc→Ophiuchus 0.452、MINT 回归 4 格→esmc300m）。删 `\OursCkptB/\OursCkptM` 宏；附录 A.1 改为"600m/mint 正在重训、本版从所有表移除"。重编译后 **12 页**，预览图 `paper/preview/main_p01–12.png`。

## 2026-07-09 整合版 9 源重训：并入全长 TCR-pMHC + B1 grammar 修复 + flash-attn guard

**原因**：审查（`ARCH_AUDIT.md`）发现 3 处设计-实现缺口，用户要求本轮全修并重训整合版取代 fh2p9：
- **B1**：`grammar.py` TCR 分支把 pMHC↔TCR 识别 token 写死 `<binding>`，吞掉 PISTE 负样本标签 → 改为读 `record.labels["relation"]`，未标注默认 `<binding>`（向后兼容）；单测 `scripts/tests/bioseq/test_grammar_tcr_relation.py`（4 例）+ `test_qwen3_vl_grammar.py` 全过（protenix_abtcr env 24 passed）。
- **B2**：Stitchr/thimble（HUMAN IMGT）+ IMGT/HLA 重建**全长五实体 TCR-pMHC**（`scripts/data/build_fulllength_tcr_pmhc.py`）：VDJdb+McPAS human MHCI → 配对全长 α/β + 全长 MHC-I 重链 + 成熟 B2M + peptide。134,526 解析 → 78,740 stitched → **下游 CDR3β 去重去掉 15,324** → 唯一 59,093（train 57,913 / valid 590 / holdout 590）。shard `data/bioseq_grammar_v1/tcr_pmhc_fulllength/`；renderer 验证五实体布局 + 2×`<binding>` + valid/train 0 重叠。数据详见 `DATA_FORMAT_AUDIT.md`。
- **B3**：`load_local_esmc_encoder` 给 `--encoder-use-flash-attn` 加**导入守卫**（缺 `flash_attn` 时告警回退 SDPA，不崩训练）；parity harness `scripts/tests/bioseq/check_esmc_flash_parity.py`（本地无 flash_attn，SKIP；训练镜像装包后跑到 PARITY OK 再开旗标）。见 `SPEED_ANALYSIS.md` lever 1。

**改动**（`train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_integrated_llada.yml`）：`--sources` 加 `tcr_pmhc_fulllength`（8→9 源），`--tcr-pmhc-fulllength-weight 1.5`；datamodule + train CLI 加对应 weight 参数。manifest 由 `inventory_integrated.py --write-manifest` 重写为 9 源（weight sum 15.50；share：oas/ots 19.4%、nanobody/mint_ppi 12.9%、tcr_piste/tcr_pmhc_fulllength 9.7%、ppi/mint_actions 6.5%、neutralization 3.2%）。

**操作**：fh2p9（8 源 pre-B1）已终态、output 归档 `..._8src_pre_b1/`；submit `t-20260709190944-2z4xv`（Initialized, Preemptible, 2 节点×8 GPU, bs4 ga2 gb128, `--resume none` from scratch）。

## 2026-07-09 整合版改 2 节点 16 卡重提

**原因**：用户要求从 8 卡单节点改为 2 节点 × 8 GPU = 16 卡。

**改动**（`train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_integrated_llada.yml`）：
- `RoleReplicas: 1 → 2`（2 节点，`ml.pni2.28xlarge` 每节点 8 GPU）
- `grad-accum: 4 → 2`（保持 global batch **128**：bs4 × ga2 × 16 GPU）
- torchrun 仍用 `${MLP_WORKER_NUM}` / `${MLP_ROLE_INDEX}` 等多节点变量

**操作**：cancel `t-20260709010716-krqjz`（Queue）→ submit `t-20260709024845-fh2p9`（Initialized, Preemptible）。

## 2026-07-09 整合版 ESMC-300M 8 源训练提交（valid↔train 去重 + downstream 去重）

**原因**：用户确认整合版取代旧三路 cmp500k/MINT；要求所有数据用上、训练前验证 valid 与下游 test 均已去重。

**valid↔train 检查**（`scripts/data/dedup/check_valid_in_train.py`，whole-record SHA1 sorted-chain key）：

| source | valid keys | train rows | train hits | ref leaked |
|--------|-----------:|-----------:|-----------:|-----------:|
| oas | 12,553 | 2,484,758 | 0 | 0 |
| ots | 10,619 | 2,085,414 | 0 | 0 |
| nanobody | 58,311 | 10,922,486 | 0 | 0 |
| mint_ppi | 207,893 | 81,717,793 | 0 | 0 |
| tcr_piste | 71,036 | 221,027 | **7** | **7** |
| ppi | 2,915 | 261,901 | 0 | 0 |
| mint_actions | 249,651 | 9,103,961 | 0 | 0 |
| neutralization | — | 11,474 | — | (无 valid shard) |

**修复**：`tcr_piste` 7 行 `apply_validleak.py --promote` → train **221,020**（`train_dsonly` 备份下游去重版）。其余 7 源零重叠。

**downstream test 去重**（此前已完成）：8 源 `train` 均已 promote（`train_prededup` 保留原始行）；manifest `mix=integrated_v1_deduped`。

**Submit**：`qwen3_vl_bioseq_grammar_v2_esmc300m_integrated_llada` via `train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_integrated_llada.yml`
- `task_id=t-20260709010716-krqjz`, initial status `Initialized`, `Preemptible: true`
- 8 源 weight 3/3/2/2/1.5/1/1/0.5；ESMC-300M + LLaDA 28L/16H/3840；bs4 ga4 nw1 gb128；`--resume none`；output `output/grammar_v2_esmc300m_integrated_llada/`

## 2026-07-08 grammar_v2 light pairing de-novo 对照 eval 提交

- **动机**：外部 baseline p-IgGen/LICHEN 已补 de-novo + prompt3 两口径；用户要求补跑**我们自己**的 de-novo（`light-prompt-tokens=0`）以隔离"chain match=1.000 是否纯 prompt3 artifact"。
- **脚本**：新增 `scripts/downstream/run_grammar_v2_pairing_prompt_sweep.sh`——对同一 best.pt 依次跑 de-novo(prompt0) + prompt3(sameckpt)，各自 `generation_eval.py` 打分。产物：`grammar_v2_{variant}_light_pairing_holdout500_{denovo,prompt3_sameckpt}_{n8.csv,_metrics.json}`。
- **提交**（`eval_jobs/eval_grammar_v2_esmc{300,600}m_cmp500k_llada_pairing_denovo.yml`，非训练、单 GPU、`ml.pni2.3xlarge`、Preemptible）：
  - 300M：`t-20260708162214-kzhpm`
  - 600M：`t-20260708162226-dwzlr`
- **预估**：本地冒烟 ~52s/4heavy(x8)，500 heavy ≈ 108min/prompt → 每作业 ~3.5–4h(含 eval)。产出后并入 downstream.md / RESULTS.md 对照表。

## 2026-07-08 修 mint_ppi 关系 token bug（`<unknown>`→`<binding>`）+ 停三路重建 shard

**问题**：审计 grammar-v2 数据 I/O 时发现，`mint_ppi`（82.4M 物理结合对，训练最大源）在 Arrow shard 里 `relation` 全是 `"unknown"`，渲染成 `<unknown>` 而非设计的 `<binding>`（对比：STRING `ppi` 源正确为 `<binding>`，`mint_actions` 正确为各 mode）。关系 token 是固定上下文（不加噪、不算 loss），但作为条件 embedding 影响生成——等于最大 PPI 源的 binding 语义被污染。

**根因**：`scripts/data/build_mint_grammar_shards.py::iter_mint_rows` 假设"links 第 3 列=关系 mode"。该假设仅对 `mint_actions`（3 列 `target actor mode`）成立；`mint_ppi` 来自 STRING `protein.physical.links.full.v12.0`（10 列：两 ID + 8 通道分数），`parts[2]="0"`（neighborhood 分数）被 `normalize_relation` 兜底成 `"unknown"`，覆盖了 `default_relation="binding"`。

**修复**：改为"仅当 `default_relation is None`（即该源确有 mode 列，=mint_actions）才读 `parts[2]`；否则用 `default_relation`（mint_ppi=binding），忽略分数列"。验证：小样本重建 2000 行 mint_ppi valid → relation 全 `binding`、渲染 `<binding>`（fixed=1）；mint_actions 无回归（catalysis/reaction/activation/... 分布正常）。

**操作**：
- 停 watchdog loop（PID 2023819 kill），cancel 三路训练（`867kq`/`pxrgx`/`79xbx`），确认 0 个非终态 bioseq 任务。
- 删 `data/bioseq_grammar_v1/.mint_shards_filter1024_v12v11` marker，后台跑 `scripts/data/rebuild_mint_training_shards.sh`（仅重建 mint_ppi train+valid，mint_actions 不动；随后刷新 grammar manifest）。日志 `data/ppi_task_raw/processed/pipeline_logs/rebuild_mint_training_shards.log`。
- **重建完成后需重新提交三路训练**（从各自 `latest.pt` `--resume auto`；注意已训 ckpt 是在 `<unknown>` 上学的，`<binding>`/`<unknown>` embedding 会有分布迁移）。

## 2026-07-08 统一多链模型：整合数据版执行启动（step: future-readme）

**原因**：用户要求整合当前全部数据训一个统一"生成+理解"多链版本。计划已收敛为"近期只做数据（按类去重 + 接入干净数据 + headline），骨干不动"，把 L1-L5 关系/理解/对齐增强、3.5 混合腐蚀、3.6 采样逻辑全部后置。需先固化后置清单，避免这些经调研得到的方向在执行期丢失。

**目的**：建 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/FUTURE_EXPERIMENTS.md`，记录每项后置实验的 原因 / 调研依据 / 落地要点 / 预期验证口径，作为整合数据版执行的第一步。

**效果**：已创建 `FUTURE_EXPERIMENTS.md`（绝对路径、遵循 `AGENTS.md`），含 L1 relpos / L2 理解头 / L3 REPA 对齐 Protenix / L4 pairformer trunk+contact / L5 encoder 跨链注意力 / 3.5 BERT-GIDD 腐蚀 / 3.6 Flexibility-Trap 采样；3.6 明确定为可控开关、**默认保持现有 `confidence-deterministic-linear` 采样**。方法决策同步见 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md`。

**整合数据版实测盘点（决定配方）**：已建 shard train ≈ 97.05M；加待建 nanobody（去 nbbench 后 11,540,148）≈ 108.6M 唯一行。采样按 `WeightedMixtureDataset` 的 weight 驱动（占比=weight/Σweight，与行数无关）。默认 weight 向量：oas 3.0 / ots 3.0 / nanobody 2.0 / mint_ppi 2.0（全量保留） / tcr_piste 1.5 / ppi 1.0 / mint_actions 1.0（capped） / neutralization 0.5 / sabdab2_abag 0.5。

**后续步骤**：nb 去 `nbbench_*` + step2 去污染 + 长度窗 ~90-160 → 按类去重（ab/nb CDRH3 70%+全长95-98%；mint 全局40%；tcr CDR3β 单连接；抗原 70-90%）→ 建 nanobody / SAbDab2-abag shard + tcr 用 tcr_piste 替换 processed_v2 → 整合 manifest+weight → 提交整合版训练 + headline（干净 vs 现状）。

## 2026-07-08 整合数据版：nanobody 去泄漏清洗（step: nb-clean）

**原因**：`step6_final/train.csv`（11,649,792 行）里混入了 109,644 行 `nbbench_*` 源——这些是 NbBench 下游 nanobody 基准的样本（step0 解析器把它们并入了训练语料），留着即对下游评测直接泄漏；另有长度离群（p100=180、p0=50）需要切。

**目的**：产出干净的 nanobody 训练 CSV 供 grammar shard 使用；只去 nbbench + 长度窗，序列相似度去重留给独立工具。

**效果**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/clean_nanobody_training.py`（env `pllm`）产出 `data/nanobody_processed/step7_clean/{train,valid,holdout}.csv`。train 11,649,792 → **11,525,884**（去 nbbench 109,644 + 去长度窗外[90,160] 14,264）；valid 58,862 → 58,311；holdout 58,981 → 58,336。已验证 train 残留 nbbench = 0。schema 与 step6_final 一致（`cleaned_seq` + FR/CDR 区段列）。数据布局同步见 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md`。

**基础设施结论（explore-infra）**：
- grammar shard 目录约定 `data/bioseq_grammar_v1/<source>/<split>`，每行 schema = `chains/roles/task_type/source/split/relation/weight`（`grammar_record_from_arrow`）。
- nanobody 读取器已存在：`sources.py::nanobody_row_to_record`（role=`nanobody_vhh`, task_type=`antibody`, 保留 FR/CDR），且 `default_source_configs` 已含 `CsvSourceConfig("nanobody", DEFAULT_NANOBODY_DIR,...)` → 只需把 `DEFAULT_NANOBODY_DIR` 指到 step7_clean 并在 `build_bioseq_grammar_v1.py` 加 `nanobody` 分支即可建 shard。
- 训练权重入口：`train_qwen3_vl_bioseq_ddp.py` 有 `--{oas,ots,nanobody,processed_v2,tcr,ppi}-weight`；`datamodule.from_args` 仅映射这 6 个，其余源（mint_ppi/mint_actions/tcr_piste/neutralization/sabdab2_abag）`source_weight` 默认回落 1.0 → 整合版需新增这些 weight 参数并在 `from_args` 补映射。
- 去重基础设施：`downstream/benchmark/common/leakage.py` 有 `dedup_against/leakage_report/min_edit_distance`（rapidfuzz Levenshtein）；`data/pipeline/step2_decontaminate.py` 已封装 MMseqs2 search（createdb/search/convertalis，`--min-seq-id/-c/-s`）可复用作全长相似度去重。
- SAbDab2 ML 数据集下载中：Zenodo record 20083995 `splits.tar.gz`（876MB），落 `data/sabdab2_ml/raw/`（Zenodo 限速 ~400KB/s，后台 aria2c 续传）。

## 2026-07-08 整合数据版：按类去重工具 + 下游 bank（step: dedup-tool）

**原因**：整合训练前必须保证 9 源训练语料不泄漏进任一下游 test（否则 headline 全线虚高）。通用蛋白 50% 阈值不适用于 Ab/TCR 设计——CDR3 是克隆型主键、全长变化小，需按类定阈值（见计划文献结论）。

**目的**：建可复用的"训练↔下游测试"去重工具，产 overlap 报告 + 逐行 blocklist，供 shard 过滤消费。

**落地**：
- `scripts/data/dedup/build_downstream_banks.py`：把所有下游 **test/eval**（OAS pairing holdout / SAbDab CDR-infill H1-3 / FLAb / OTS holdout / NbBench 12 任务 / IRBench-PPI / MINT 6 任务 / NM2025 / TCR clustering·representation·generation）读成按生物类型的 bank，落 `data/dedup/banks/`。用 `records.normalize_sequence`（J→L+大写+去空格）与 shard 侧一致。**只收 test 侧**（跳过各 benchmark 自带 train/background split，避免过删）。bank 规模：ab_cdrh3=9,619；ab_heavy=106,806；ab_light=6,329；tcr_cdr3b=32,887；tcr_cdr3a=16,969；antigen=221；ppi_proteins=15,051。
- `scripts/data/dedup/dedup_train_vs_downstream.py`：按域匹配——**tcr=CDR3β 精确单连接**（set 交）；**ab/nb=CDRH3 精确 + CDRH3 70% 聚类(mmseqs) + 全长 95%(mmseqs, `flow` env mmseqs 18)**；**ppi=全局 40% id(mmseqs)**。产 `data/dedup/reports/<src>.json` + `data/dedup/blocklists/<src>.jsonl`。
- `scripts/data/dedup/apply_blocklists.py`：按 blocklist 的 row index（与 extractor 迭代序一致：CSV=DictReader 序、Arrow=load_from_disk 序）过滤，CSV→`*.dedup.csv`、Arrow→`<src>/train_dedup`，原件不动。

**效果（全 8 源已跑完并 promote，第 9 源 mint_ppi 进行中）**：每源 `train` 现为去重后行、`train_prededup` 保留原始行、`train_dedup` 为中间产物；训练读 `train` 即读干净数据。逐源泄漏（leaked/raw）：

| 源 | 规则 | raw | 泄漏 | 去重后 train | leak% |
|----|------|-----|------|------|-------|
| oas | CDRH3 精确+70%聚类 + 全长95% | 2,486,442 | 1,684 | 2,484,758 | 0.07% |
| ots | CDR3β 精确单连接 | 2,102,715 | 17,301 | 2,085,414 | 0.82% |
| nanobody | CDRH3 精确+70%聚类 + 全长95% | 11,525,884 | 603,398 | 10,922,486 | 5.24% |
| tcr (processed_v2) | CDR3β 精确单连接 | 163,725 | 18,215 | 145,510 | 11.13% |
| tcr_piste | CDR3β 精确单连接 | 284,144 | 63,117 | 221,027 | 22.21% |
| ppi | 全局40% linclust | 319,429 | 57,528 | 261,901 | 18.01% |
| mint_actions | 全局40% linclust | 9,237,455 | 133,494 | 9,103,961 | 1.45% |
| neutralization | 全长95%（grammar shard 无 CDR，仅全长） | 12,346 | 872 | 11,474 | 7.06% |

→ tcr/tcr_piste 与 TCR 基准重叠最重（同源公库随机划分），ppi 次之；抗体侧全长/CDRH3 泄漏低。所有 promote 行数已核验 = raw − leaked（`apply_blocklists.py --promote` 逐行索引与 extractor 迭代序一致，且已断言 shard 行数==report n_rows 防错位）。

**mint_ppi（最大源，82.4M 对/16.4M 蛋白）专用去重**：generic 逐行 Arrow extractor 过慢，改用 `scripts/data/dedup/dedup_mint_ppi.py`——(1) 对"PPI test bank ∪ mint_ppi 蛋白全集"跑一次 `easy-linclust`@40%，co-cluster 到 test 的 mint 蛋白记为泄漏；(2) 按 shard builder 同序同过滤（`ppi_record`：双链 valid 且 ≤1024aa）流式 links，任一链泄漏则该 arrow-row 泄漏。产同格式 report+blocklist 供 `apply_blocklists.py --promote` 消费。**进行中**（linclust ~16.4M 蛋白，RSS ~27GB）。

**SAbDab2 blocker（build-shards，当时结论，已由 2026-08-03 校正）**：当时的
本地 `splits.tar.gz` 解包后只看到 `ab_split.csv` + 427 CIF，因而暂时把 SAbDab2
从主路线解耦。后续 MD5 校验确认该本地件损坏/不完整；完整官方归档实际包含
`abag_split.csv`。该段只保留为历史过程，不能再作为当前数据事实。

**mint_ppi 重建已完成**：`rebuild_mint_training_shards.sh` 跑约 2h25min 完成，`mint_ppi/train`=82,441,955 行、relation 全 `binding`（bug 已修）、136 shard；marker `.mint_shards_filter1024_v12v11` 已写。其脚本重写的 `manifest.json` 为 6 源（含已去重的 oas/ots/tcr/ppi/mint_actions 计数 + 原始 mint_ppi），最终整合 manifest 由 `inventory_integrated.py --write-manifest` 覆盖为 8 源。

**整合训练配方已固化**：`train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_integrated_llada.yml`——8 源 `oas,ots,nanobody,mint_ppi,tcr_piste,ppi,mint_actions,neutralization`，weight 3/3/2/2/1.5/1/1/0.5（share 21.4/21.4/14.3/14.3/10.7/7.1/7.1/3.6%，`WeightedMixtureDataset` 按 weight/Σweight 采样与行数无关）。骨干与 cmp500k 版一致（ESMC-300M + LLaDA，28L/16H/3840，bs8 ga2 gb128，num_workers1）。**待 mint_ppi 去重 promote 后**再写整合 manifest 并提交。

## 2026-07-08 LLaDA 训练提速（num_workers=2 + no_sync）

**动机**：LLaDA line ~30-35 samples/s（8 GPU），远低于旧 AirGen ~50k step/day。逐 microstep 计时定位瓶颈：`wait_s≈1.03s`（数据加载/grammar-encode 串行阻塞 GPU），约占单 microstep 的一半；`backward_s≈0.84s` 含每 microstep 的 DDP all-reduce。

**根因（本地单卡 A/B profiling，`output/prof_R*/run.log`）**：
- **数据加载是主瓶颈**：`num_workers` 0→2 使 grammar-encode（`GrammarBioSeqCollator`，CPU 密集）在 side process 与 GPU 前向/反向重叠，`wait_s` 明显下降。
- **GC 必须保留**：关梯度检查点后即 OOM（`prof_R1_nogc` bs8 step3 OOM，`prof_R1b` bs4 也在长序列 step8 OOM；开 GC 时长序列峰值 ~72GB）。故**不动 batch、不关 GC**。
- 冻结 ESMC `sequence_head` / 关 `find_unused_parameters`：profiling 未测出收益，且关 find_unused 对双塔异构 batch 有 DDP 崩溃风险 → **放弃**。

**两项改动**：
1. `trainer.py` `fit()`：grad-accum 非边界 microstep 用 `DistributedDataParallel.no_sync()`，梯度本地累积、仅在 optimizer.step 前 all-reduce 一次（ga=4 时省 3/4 梯度通信，**数学等价**，仅 DDP 包装下生效）。
2. `datamodule.py`：`num_workers>0` 时启用 `persistent_workers` + `prefetch_factor=4`；五个 LLaDA resume YAML `--num-workers 1→2`。

**验证**：
- 正确性闸门：`scripts/tests/bioseq/test_qwen3_vl_ddp_training.py` 11 passed；2 进程 CPU DDP 跑 no_sync 路径两 rank `grad_norm` 完全一致。
- 双卡 smoke `t-20260708003825-9pwx8`（`ml.pni2.7xlarge`×1=2 GPU，bs4 ga4 nw2，镜像生产配置，`--find-unused-parameters`）：**400/400 step 跑完、无 NCCL 超时、无 OOM**；两 rank `grad_norm` 每步一致（如 step399=2.4687）→ 证明 `num_workers=2` 分片对 DDP+find_unused 安全、no_sync 梯度归约正确。`mem_peak≈46GB`（与生产 bs4 相同）。
- 提速证据：smoke（nw2）`mean wait_s≈0.49/microstep` vs 生产（nw1）`≈1.03` → 数据加载 stall 约减半。
- **踩坑**：首个 smoke 用 bs8 ga2 复现历史"填满 80G"配置，step8 OOM（77GB，num_workers 改变 batch 组成使长序列批次跨过 79GB）→ 生产实际用 bs4 ga4（peak ~46GB），smoke 已改为镜像生产的 bs4 ga4。

**部署**：五个 resume YAML（300m/600m cmp500k 各闲时+非闲时、300m mint）均已 `--num-workers 2`；no_sync 为 `trainer.py` 代码改动。watchdog 监控的三路闲时 resume（`scripts/llada_train_watchdog_jobs.json`）在任务 **Killed/Failed** 时会用上述 YAML 自动续跑 → **下次被打断即自然带上 nw2 + no_sync，无需手动 cancel 当前 Running 任务**。非闲时 YAML 仅在手提非闲时续跑时生效。

## 2026-07-07 三路全部改闲时资源

- **300M cmp500k**: cancel 非闲时 `cq4r9`（Running）→ 闲时重提 `t-20260707105343-gzswj`（save100）。
- **600M / MINT**: 已是闲时 `d4hqx` / `9x4hq`（Queue），watchdog 配置已统一为闲时 YAML。
- **策略**: 三路均 `Preemptible: true`，`--save-interval 100`，watchdog 10min 自动续跑。

## 2026-07-07 save-interval=100 + 闲时策略 + watchdog 10min

- **save-interval**: 三路 resume YAML 均改为 `--save-interval 100`（原 1000），减少抢占后丢步（例：MINT `ggwm8` 跑到 step 14380 但 latest.pt 仍 step 14000，丢 ~380 step）。
- **资源策略**: 仅 **300M cmp500k** 非闲时（`cq4r9` Running）；**600M / MINT** 改回闲时续跑。
- **Submit**:
  - MINT: `t-20260707105105-9x4hq`（闲时, save100）
  - 600M: 取消非闲时 `w22xx` → `t-20260707105108-d4hqx`（闲时, save100）
- **Watchdog**: 周期 **30min → 10min**；配置 `scripts/llada_train_watchdog_jobs.json` 已同步新 task_id 与 YAML。

## 2026-07-07 LLaDA 三路训练 watchdog（30min 自动续跑）

- **脚本**: `scripts/watch_llada_train_jobs.sh` + 配置 `scripts/llada_train_watchdog_jobs.json`
- **监控对象**:
  - 300M cmp500k 非闲时 `cq4r9` → YAML `..._resume_nonpreempt.yml`
  - 600M cmp500k 非闲时 `w22xx` → YAML `..._resume_nonpreempt.yml`
  - 300M MINT 闲时 `ggwm8` → YAML `..._mint_llada_resume.yml`
- **逻辑**: 每 **30min** 查 Volc 状态；`Killed`/`Failed` 且 `latest.pt` 存在则自动 `ml_task submit` 并更新 JSON 里 `task_id`；`Success` 仅记录不续跑。
- **日志**: `scripts/logs/llada_train_watchdog.log`；后台 loop PID: `scripts/logs/llada_train_watchdog.loop.pid`
- **首次 tick** (2026-07-07T02:34Z): 300M cmp500k **Running**，600M **Queue**，MINT **Running** — 无需续跑。

## 2026-07-07 600M cmp500k 改非闲时资源续跑

- **600M cmp500k `t-20260707094319-sz8wt`**: 闲时被抢占 **Killed**（~42min，log step ~7400）；`latest.pt` 完好（Jul-7 01:55 UTC）。
- **Submit（非闲时）**: `qwen3_vl_bioseq_grammar_v2_esmc600m_cmp500k_llada_resume_nonpreempt` via `train_jobs/qwen3_vl_bioseq_grammar_v2_esmc600m_cmp500k_llada_resume_nonpreempt.yml`
- `task_id=t-20260707103221-w22xx`, initial status `Initialized`, **`Preemptible: false`**, 超参同闲时 resume（bs4/ga4/nw1, `--resume auto`）。

## 2026-07-07 300M cmp500k 改非闲时资源续跑

- **背景**: 闲时任务 `jbn2v` / `5zwgv` 均在排卡后数分钟内被抢占 **Killed**（log step ~8007）。
- **Submit（非闲时）**: `qwen3_vl_bioseq_grammar_v2_esmc300m_cmp500k_llada_resume_nonpreempt` via `train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_cmp500k_llada_resume_nonpreempt.yml`
- `task_id=t-20260707102700-cq4r9`, initial status `Queue`, **`Preemptible: false`**, 超参同闲时 resume（bs4/ga4/nw1, `--resume auto`）。

## 2026-07-07 300M cmp500k 再次抢占 Killed → resubmitted

- **300M cmp500k `t-20260707094315-jbn2v`**: 排卡后仅 ~6min 即被抢占 **Killed**（log step ~8007）；`latest.pt` 已更新（Jul-7 01:47 UTC）。
- **Resubmit**: `t-20260707100614-5zwgv` via `train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_cmp500k_llada_resume.yml`，initial status `Queue`，`Preemptible: true`。
- **300M MINT `ggwm8`**: 仍 **Running**（~step 14064），未停。

## 2026-07-07 三路抢占 Killed → resubmitted from latest.pt

- **300M cmp500k `t-20260706104738-5p9ht`**: **Killed**（抢占），断点 log step ~8940；`latest.pt` 完好。
- **600M cmp500k `t-20260706104738-5st84`**: **Killed**，断点 log step ~7920；`latest.pt` 完好。
- **MINT `t-20260706105506-xf8gd`**: **Killed**，断点 log step ~14600；`latest.pt` 完好（Jul-6 07:27 UTC）。
- **Resubmit（`--resume auto`；超参未改 bs4/ga4/nw1 + stdout 落盘）**:
  - `t-20260707094315-jbn2v` (300M cmp500k), initial status `Queue`, `Preemptible: true`
  - `t-20260707094319-sz8wt` (600M cmp500k), initial status `Queue`, `Preemptible: true`
  - `t-20260707094323-ggwm8` (MINT 300M), initial status `Queue`, `Preemptible: true`

## 2026-07-06 MINT-only 300M 抢占 Killed → resubmitted from latest.pt

- **MINT `t-20260705210842-k7wnf`**: 可抢占资源被抢占 **Killed**（非代码错误）。`latest.pt` 完好（step **13000**，Jul-5 17:33 UTC）。
- **训练数据**（仅 MINT 两源）: `mint_ppi` train **82,441,955** rows + `mint_actions` train **9,237,455** rows；valid 250k/207k。YAML `--sources mint_ppi,mint_actions`。
- **Resubmit**: `t-20260706105506-xf8gd` via `train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_mint_llada_resume.yml`，`--resume auto`，initial status `Queue`，`Preemptible: true`。

## 2026-07-06 cmp500k 300M/600M 抢占 Killed → resubmitted from latest.pt

- **cmp500k 300M `t-20260705220246-9p9j4`**: 可抢占资源被抢占 **Killed**（非代码错误）。`latest.pt` 完好（Jul-5 18:24 UTC）。
- **cmp500k 600M `t-20260705220235-d4zzs`**: 同轮 **Killed**。`latest.pt` 完好（Jul-5 15:38 UTC）。
- **MINT `t-20260705210842-k7wnf`**: 同轮 **Killed**（本轮未重提）。
- **Resubmit（`--resume auto` 从各自 `latest.pt`；超参未改 bs4/ga4/nw1 + stdout 落盘）**:
  - `t-20260706104738-5p9ht` (cmp500k 300M), initial status `Queue`, `Preemptible: true`
  - `t-20260706104738-5st84` (cmp500k 600M), initial status `Queue`, `Preemptible: true`

## 2026-07-05 cmp500k-300M bs=8 CUDA OOM 坐实回退 bs=4 & 600M 被抢占重提

- **cmp500k 300M `t-20260705210852-r95m8` (bs=8/ga=2/nw1)**: **Failed**。这轮 stdout 已落盘，`output/grammar_v2_esmc300m_cmp500k_llada/logs/train_20260705_135123_rank0.log` 确认在 `loss.backward()` 峰值 **CUDA OOM**（80G 被 bs=8 吃爆）。结论：bs=8 在该模型/序列长度下不可行。
- **Fix — 回退 batch**: 300M cmp500k resume YAML 把 `--batch-size 8 --grad-accum 2` 改回 `--batch-size 4 --grad-accum 4`（global batch 仍 4×4×8=**128** 不变）；保留 nw1、`--debug-ddp-timing`、stdout 落盘。其它超参一律未动。Description 已注明 OOM 回退原因。
- **cmp500k 600M `t-20260705175122-p82qx` (bs=4/ga=4/nw1)**: 可抢占资源被抢占 **Killed**@13:42Z（非代码错误）。本轮**未改任何超参**，仅补 stdout 落盘（`mkdir -p logs` + `STAMP` + `torchrun ... > logs/train_${STAMP}_rank${MLP_ROLE_INDEX}.log 2>&1`，直接重定向不用 `| tee`）便于将来失败回溯。
- **Resubmit（`--resume auto` 从各自 `latest.pt`）**:
  - `t-20260705220246-9p9j4` (cmp500k 300M, bs4/ga4/nw1), initial status `Initialized`, `Preemptible: true`
  - `t-20260705220235-d4zzs` (cmp500k 600M, bs4/ga4/nw1), initial status `Queue`, `Preemptible: true`
- **MINT `t-20260705210842-k7wnf`**: 仍 **Running**（Elapsed ~0.9h），本轮未改动。
- **验证**: `ml_task list ... -o json` 确认 `9p9j4`=Initialized、`d4zzs`=Queue、`k7wnf`=Running，均非终态。

## 2026-07-05 cmp500k-300M Failed & MINT preempted → resubmitted with stdout 落盘

- **cmp500k 300M `t-20260705173643-cjtdl`**: **Failed**@10:29Z (sped-up bs8/ga2/nw1 run). 主假设为后期 OOM（bs8 填显存），但**因该轮 stdout 未落盘、任务终态后 `volc ml_task logs` 返回 exit 255，无法坐实失败原因**。
- **MINT `t-20260705175135-ghb4z`**: 可抢占资源被抢占 **Killed**@11:55Z（非代码错误）。
- **Fix — stdout 落盘**: 两个 resume YAML 的 `torchrun` 追加 `> "${OUTPUT_DIR}/logs/train_${STAMP}_rank${MLP_ROLE_INDEX}.log" 2>&1`（直接重定向，退出码天然是 torchrun 的，不掩盖失败；`set -euo pipefail` 下不用 `| tee`）；并在 torchrun 前 `mkdir -p "${OUTPUT_DIR}/logs"` + `STAMP=$(date +%Y%m%d_%H%M%S)`。超参未改（cmp500k 仍 bs8/ga2/nw1 以复现并验证 OOM 假设；MINT 仍 bs4/ga4/nw1），保留 `--debug-ddp-timing`。
- **Resubmit（`--resume auto` 从各自 `latest.pt`）**:
  - `t-20260705210852-r95m8` (cmp500k 300M, from step 5000), initial status `Queue`, `Preemptible: true`
  - `t-20260705210842-k7wnf` (MINT 300M, from step 11000), initial status `Queue`, `Preemptible: true`
- **600M `t-20260705175122-p82qx`**: 仍 **Running**（Elapsed ~3.3h），本轮未改动。
- 日志将在任务真正 Running 后写入 `output/<run>/logs/train_<STAMP>_rank<N>.log`（提交后短时间内尚未产生，属正常）。

## 2026-07-05 Speed optimization P0+P1 (LLaDA training throughput)

- **Diagnosis** (from run logs): 300M ~4s/step, mem_peak 46/79GB (headroom), data-load `wait_s` ~50% of step time (`num_workers=0`); MINT ~6.7s/step, `wait_s`~1.69s.
- **P0 (fill memory)**: 300M micro-batch `bs=4 ga=4 -> bs=8 ga=2` (global batch 128 unchanged; ~2x data per wall-second at same optimizer-step count). 600M/MINT keep `bs=4` (mem ~65GB / long PPI seqs).
- **P1 (overlap data load)**: `num_workers=0 -> 1` + `persistent_workers` + `prefetch_factor=4` in [datamodule.py](dllm_test/dllm/pipelines/qwen3_vl_arch/data/datamodule.py). Key safety fact: `num_workers=1` keeps the exact shard identity of 0 (`distributed_worker_shard -> (rank, world_size)`), so DDP data order is unchanged and the multi-worker re-shard desync (why nw was pinned to 0) is avoided. Verified: nw=0 vs nw=1 first-5-batch shapes/tasks/token-sums byte-identical.
- New arg `--prefetch-factor` in [train_qwen3_vl_bioseq_ddp.py](dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py); YAMLs updated: both cmp500k 300M (base+resume, bs8/ga2/nw1), both 600M (base+resume, nw1), MINT (nw1).
- P2 (token-based dynamic batching, plan-preferred) NOT done this round.

**Monitor (run anytime)**

```bash
# 一次性查看三路训练状态
bash /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/monitor_llada_train_eval.sh

# 自动监控 + 断线续跑（每 10min，已在后台运行时可查日志）
bash /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/watch_llada_train_jobs.sh          # 单次
bash /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/watch_llada_train_jobs.sh --loop 10m  # 循环
tail -f /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/logs/llada_train_watchdog.log
# 停止循环: kill $(cat .../scripts/logs/llada_train_watchdog.loop.pid)

# 任务 ID 配置（watchdog 续跑后自动更新）:
# scripts/llada_train_watchdog_jobs.json

tail -f /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/grammar_v2_esmc300m_cmp500k_llada_downstream_summary.txt
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY volc ml_task list -n bioseq --status Initialized,Queue,Staging,Running,Killing -o json --limit 100
```

## 2026-07-05 Submitted AB-generation downstream re-eval on current best.pt (300M+600M cmp500k)

- **Why**: the 7-04 AB-generation numbers (CDR-H1/H2/H3 AAR + light-chain pairing ImmunoMatch) in `output/downstream_generation/grammar_v2_esmc{300,600}m_cmp500k_llada_downstream_summary.txt` were produced on ckpts that were **overwritten by Jul-5 resume training** (300M `best.pt` mtime `2026-07-05T13:52:50Z`, 600M `best.pt` mtime `2026-07-05T13:07:44Z`). Re-run both AB-generation tasks (CDR infilling + light-chain pairing) on the **current best.pt** and report deltas.
- **Snapshot ckpt evaluated** (from `checkpoints/topk_manifest.json`, since 300M training resume is still writing):
  - 300M cmp500k: `best.pt` = step 5000, val/loss 0.8095
  - 600M cmp500k: `best.pt` = step 4000, val/loss 0.8493
- **Submit** (no-proxy wrapper; existing YAMLs unchanged; queue `c20250601`, `Preemptible: true`, single GPU `ml.pni2.3xlarge`):
  - 300M: `eval_grammar_v2_esmc300m_cmp500k_llada_downstream` via `eval_jobs/eval_grammar_v2_esmc300m_cmp500k_llada_downstream.yml` → `task_id=t-20260705220152-768fw`, initial status `Running`.
  - 600M: `eval_grammar_v2_esmc600m_cmp500k_llada_downstream` via `eval_jobs/eval_grammar_v2_esmc600m_cmp500k_llada_downstream.yml` → `task_id=t-20260705220153-zwgdk`, initial status `Running`.
- **What it runs**: `scripts/downstream/run_grammar_v2_variant_downstream_eval.sh` = CDR-H1/H2/H3 SabDab 10-fold (Average AAR) + OAS holdout500 light-chain pairing (light-prompt-tokens=3, num-seqs=8) with ImmunoMatch/ANARCI metrics; results land in `output/downstream_generation/`.
- Note: downstream **eval** jobs, out of the LLaDA-training active-table scope; recorded here as history only. (Results + old/new delta appended below once the jobs finish.)

## 2026-07-05 AB downstream eval completed (partial)

**Tasks**（out of active table）：
- `t-20260705220152-768fw` → **Failed** @ 2026-07-05T14:58:48Z — CDR ✅；light pairing 生成 ✅；ImmunoMatch **Failed**（`transformers` + Keras 3 → 需 `pip install tf-keras`）
- `t-20260705220153-zwgdk` → **Success** @ 2026-07-05T15:33:44Z — 全流程 ✅

**CDR Average AAR all folds（7-05 best.pt vs 7-04 旧 ckpt）**

| ckpt | H1 | H2 | H3 | Δ (pp) |
|------|:--:|:--:|:--:|:------|
| 300m | **68.76** | **63.64** | **44.68** | +4.0 / +5.7 / +4.1（旧 64.72/57.93/40.54） |
| 600m | **71.88** | **67.27** | **45.25** | +2.9 / +4.9 / +3.5（旧 68.95/62.34/41.80） |

**Light pairing（300M + 600M 新 metrics）**

| ckpt | gen ImmunoMatch | V gene match | gen>ref ratio | chain match |
|------|:---------------:|:------------:|:-------------:|:-----------:|
| 300m | **0.633**（旧 0.430） | **0.835**（旧 0.577） | **0.393**（旧 0.263） | **1.000** |
| 600m | **0.607**（旧 0.455） | **0.835**（旧 0.577） | **0.368**（旧 0.263） | **1.000**（旧 0.995） |

源：`output/downstream_generation/grammar_v2_esmc{300,600}m_cmp500k_llada_downstream_summary.txt`、`*_light_pairing_holdout500_prompt3_metrics.json`（mtime 2026-07-05）。

## 2026-07-05 300M pairing metrics-only 补跑（Success）

- **Why**：主作业 `768fw` light pairing 生成完成后 ImmunoMatch 因缺 `tf-keras` 崩溃；生成 CSV（4000 行）已落盘。
- **Submit**：`eval_jobs/eval_grammar_v2_esmc300m_cmp500k_llada_pairing_metrics.yml`（entrypoint 含 `pip install -q tf-keras` + `run_grammar_v2_llada_pairing_metrics_only.sh`）→ `task_id=t-20260705233630-mrmzv`
- **Result**：**Success** @ 2026-07-05T16:07:55Z — gen ImmunoMatch **0.633** / ref **0.699** / gen>ref **0.393** / chain **1.000** / V gene **0.835** / diversity **0.163**
- **Log**：`output/downstream_generation/eval_grammar_v2_esmc300m_cmp500k_llada_pairing_volc.log`
- **YAML 永久修复**：`eval_grammar_v2_esmc300m_cmp500k_llada_downstream.yml` 亦加入 `pip install -q tf-keras`，避免全流程重跑再 Fail

## 2026-07-05 Submitted T1/T2/T3 post-LLaDA re-eval (methodology fix: Ours-BioSeq features MUST be post-LLaDA)

- **Why**: the TCR downstream tasks were extracting Ours-BioSeq features from the **ESMC encoder only** (`grammar:encoder:` / `bioseq:` → `EsmcBioSeqEmbedder.last_hidden_state`), which evaluates the frozen ESMC backbone, NOT our trained LLaDA diffusion model. Fixed to **post-LLaDA** = features that have passed through the LLaDA decoder (final decoder hidden states).
- **Code**: `downstream/benchmark/common/model_api.py::GrammarEmbedder.embed()` now honours `feature_source`: `decoder` builds a single-chain `tcr` grammar record (`<prots><tcr> SEQ <protd>`), runs the clean/unmasked LLaDA decoder (`_final_hidden`), mean-pools `hidden_states[-1]` over residues; `encoder` keeps the old ESMC path (comparison only). T3 fewshot already routes decoder→`embed_pairs` (joint β+α, placeholder peptide). T1 runner default flipped to `grammar:decoder:`. Smoke: decoder vs encoder per-seq cosine ~0.09, both finite, deterministic.
- **Submit** (all `eval_jobs/*.yml`, queue `c20250601`, `Preemptible: false`, single GPU `ml.pni2.3xlarge`; NEW output tags so encoder-only results are preserved for comparison):
  - T1 TCR-binding: `eval_tcr_binding_nm2025_postllada` → `task_id=t-20260705215728-n4xqj`, initial status `Running`. Tags `ours_postllada_{esmc300m_cmp500k,esmc600m_cmp500k,esmc300m_mint}`.
  - T2 TCR-clustering (basis-A embed-threshold + basis-B TCREmbedding): `eval_t2_clustering_postllada` → `task_id=t-20260705215733-f7k8q`, initial status `Running`. Tags `ours_postllada_*`.
  - T3 TCR-representation (deep 6-pMHC + broad 24-epitope): `eval_t3_representation_postllada` → `task_id=t-20260705215736-lxpzz` **Failed**@14:02Z (deep track OK, broad track crashed `KeyError: 'alpha'` in `embed_pairs` on a β-only row within an α-containing batch). **Fixed** `model_api.py::embed_pairs` to zero-fill a missing chain segment (smoke re-verified). Resubmitted → `task_id=t-20260705220618-qdw4j` `Running` (a double-submit `t-20260705220605-2975b` was `cancel`led). Auto tags `grammar_dec_*_llada`.
- Covers T1/T2/T3 × 3 checkpoints (esmc300m-cmp500k / esmc600m-cmp500k / esmc300m-mint). MINT PPI (P0) already post-LLaDA (mint_tasks decoder path) — NOT re-run. Baselines unchanged.
- Note: downstream **eval** jobs, out of the LLaDA-training active-table scope; recorded here as history only.

## 2026-07-05 Submitted T1/T2/T3/PPI globalfeat re-eval (whole-feature protocol)

- **Why**: post-LLaDA re-eval still used **segmented** features (per-chain mean-pool then concat `[β‖α‖pep]` or `sep_chains`), which is not the our-model headline口径. Headline must be **one global mean-pool** over the full grammar record after LLaDA.
- **Code**: `GrammarEmbedder(pool_mode="global")` default; MINT grammar `sep_chains=False`; T1 joint `tcr_peptide` record; smoke `scripts/smoke_globalfeat_pool.py` OK (global `[N,H]` ≠ segment concat).
- **Submit** (NEW tags; legacy `ours_postllada_*` / `*_sep_*` preserved):
  - T1: `eval_tcr_binding_nm2025_globalfeat.yml` → `t-20260705230306-mk5jr`
  - T2: `eval_t2_clustering_globalfeat.yml` → `t-20260705230310-p97r7`
  - T3: `eval_t3_representation_globalfeat.yml` → `t-20260705230313-s7xsw`
  - PPI: `eval_mint_ppi_globalfeat.yml` → `t-20260705230316-wwphr`
- Note: downstream **eval** jobs, out of the LLaDA-training active-table scope; recorded here as history only.

## 2026-07-05 Submitted MINT GeneralPPI (P0) downstream eval — MINT official model

- **Submit**: `eval_mint_ppi_benchmark` via `eval_jobs/eval_mint_ppi_benchmark.yml`
  - `task_id=t-20260705210235-5k5zz`, initial status `Running`, queue `c20250601`, `Preemptible: false`, single GPU (`ml.pni2.3xlarge`).
- **What it runs**: OFFICIAL MINT multimer-ESM2 model (vendored `mint_tasks/_mint_weights/mint.ckpt`, 3.25GB, obtained 07-05 via hf-mirror) on **HumanPPI + Bernett**, MINT two-stage protocol (cache chain embeddings → train MLP head), CAP train=3000 / Bernett val=1500 / test full / rep=3. Fills the `mint-official` column of the 2×3 PPI grid.
- **Split of work** (see `downstream/mint_tasks/EXECUTION_PLAN.md`): esm2-650M baseline + our grammar model cells run locally (prior `_run_bench.sh`, 09:40); MINT-official cells moved to Volc submission (disjoint cache dir `mint_sep_t3000/`, no race with local).
- **2026-07-05 22:50 follow-up**: Volc job **completed** — metrics.json on vepfs: HumanPPI AUROC 0.9378 / Bernett 0.7386. Local grid **6/6 complete** (Bernett × grammar AUROC 0.6182, `_run_bench.sh` done @15:08 UTC+8).
- Note: this is a downstream **eval** job, out of the LLaDA-training active-table scope; recorded here as history only.

## 2026-07-03 Focus switch → LLaDA-only

- **Scope**: manage only LLaDA-backbone training/eval.
- **Active YAMLs**: `train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_cmp500k_llada.yml`, `train_jobs/qwen3_vl_bioseq_grammar_v2_esmc600m_cmp500k_llada.yml`.
- **First Volc A/B (07-02)**: `t-20260703011643-zw6wz` / `t-20260703011655-qzvv8` Failed exit 1 ~55s; no output dir.

## 2026-07-03 LLaDA local end-to-end verification

- Entrypoint gates + short train/val/resume/DDP all pass locally.

## 2026-07-04 Resubmitted LLaDA A/B training

- `t-20260704132723-f8ncd` (ESMC-300M), `t-20260704132723-rqblz` (ESMC-600M); 8×A100 preemptible.

## 2026-07-04 Submitted LLaDA downstream eval

- `t-20260704142956-8tlwl` / `t-20260704142956-tlgsc`; CDR H1/H2/H3 + light pairing + ImmunoMatch.
- Fixed `downstream/grammar/common.py::load_grammar_checkpoint` (merge ckpt args with `parse_args()` defaults).

## 2026-07-04 Downstream eval Failed (ImmunoMatch path) + fix

- **Failed tasks**: `t-20260704142956-8tlwl` / `t-20260704142956-tlgsc` — CDR + light generation **completed**; crashed at ImmunoMatch scoring.
- **Root cause**: Volc workers only mount `c20250601/251105016`; ImmunoMatch ckpts were under `/vepfs-mlp2/mlp-public/...` (inaccessible).
- **Fix**: copied ckpts to `data/downstream/immunomatch/`; updated `immunomatch_score.py::_immunomatch_ckpt()`.
- **Resubmit (pairing metrics only)**: `t-20260704164723-rdl82` / `t-20260704164726-vnck9` — ImmunoMatch path fixed; failed again @ ANARCI `BrokenProcessPool` (bash heredoc `<stdin>` spawn).
- **Fix #2**: proper entry script `scripts/downstream/run_grammar_v2_pairing_metrics_only.py`; resubmit `t-20260704165143-427lb` (300M), `t-20260704165147-7xhc8` (600M).

## 2026-07-04 Monitor + downstream interim (step-1000 ckpt, CDR complete)

- **300M eval** (ckpt @ step 1000): CDR-H1 **64.72%**, H2 **57.93%**, H3 **40.54%**; light gen CSV saved (4000 rows).
- **600M eval** (ckpt @ step 1000): CDR-H1 **68.95%**, H2 **62.34%**, H3 **41.80%**; light gen CSV saved.
- **600M train** `rqblz`: **Killed** (preempt) @ step ~2440; resume submitted `t-20260704164748-5h7kr`.

## 2026-07-04 Training batch sizing policy (volc-batch-sizing.mdc)

- Removed YAML 中「对齐 in-house decoder / cmp500k A/B」表述；新规则见 `.cursor/rules/volc-batch-sizing.mdc` + `PROJ_GUIDE.md` §Training batch sizing。
- **同实验跨 encoder**：global batch **128** 保持一致（300M/600M LLaDA 均为 bs=4×ga=4×8 GPU）。
- **显存 profile（当前 run，Volc 8×A100-80G）**：300M **~46GB/80G**（偏低，下次 submit/resume 前应 profile 提高 micro-batch，如 bs=8 ga=2）；600M **~65GB/80G**（已接近上限，bs=4 合理）。
- **当前 run 不改 batch**（已在跑）；resume/重提 300M 时按规则先 profile 再改 YAML。

## 2026-07-04 Submitted MINT-only ESMC-300M LLaDA pretrain

- Task `t-20260704173749-jgxw9`; YAML `train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_mint_llada.yml`
- Sources: `mint_ppi` + `mint_actions` only; splits follow `mint_string_pretrain_v1` / `mint_string_actions_v11` (MINT reference seeds 137/731, 250k valid, MMseqs2 50% cluster disjoint).
- Output: `output/grammar_v2_esmc300m_mint_llada/`

## 2026-07-04 Volc YAML 目录整理

- `train_jobs/` 仅保留当前 LLaDA 训练 YAML（300M / 600M / 600M resume）。
- downstream / pairing eval YAML 迁至 `eval_jobs/`；生成脚本：`scripts/downstream/gen_eval_grammar_v2_llada_ymls.sh`、`gen_eval_grammar_v2_cmp500k_ymls.sh`。
- 规则写入 `PROJ_GUIDE.md` §Volc job layout。

## 2026-07-05 Resumed cmp500k ESMC-300M/600M LLaDA (OAS mix)

- Previous runs `f8ncd` / `5h7kr` both **Killed** (preempt) @ step ~5119 / ~3740; checkpoints intact.
- Created `train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_cmp500k_llada_resume.yml` (300M had no resume YAML before).
- Resubmitted both with `--resume auto` from `latest.pt`:
  - `t-20260705140932-xzltj` (ESMC-300M), initial status `Initialized`, `Preemptible: true`
  - `t-20260705140936-vsrff` (ESMC-600M), initial status `Initialized`, `Preemptible: true`

## 2026-07-05 300M cmp500k resume 被抢占，重新提交

- `t-20260705140932-xzltj` (ESMC-300M resume) 查询状态为 **Killed**（LaunchTime 06:14 后很快被抢占，`latest.pt` 完好）。
- `t-20260705140936-vsrff` (ESMC-600M resume) 仍在 **Queue** 排队。
- 用同一 YAML（`--resume auto` 从 `latest.pt`）重新提交 300M：新任务 `t-20260705162050-dbvwc`，状态 `Queue`。
- 恢复点：`output/grammar_v2_esmc300m_cmp500k_llada/latest.pt`（step ~5717，best val 0.8428 @ step 5000）。

## 2026-07-07 Submitted T3 representation globalfeat re-eval (current best.pt)

- **Why**: 3× `best.pt` mtime newer than existing T3 fewshot.json (2026-07-05); resume 后权重未重跑 T3；统一 headline 为 `grammar:decoder:global:` + tag `ours_globalfeat_*`；新增 `checkpoint` provenance in fewshot.json.
- **Code**: `common/provenance.py`; `tcr_representation/run.py` + `run_paper6.py` record `checkpoint.{path,mtime,size_bytes,sha256}`.
- **YAML**: `eval_jobs/eval_t3_representation_globalfeat.yml` — explicit `--tag ours_globalfeat_<run>` per ckpt; 3 ckpt × (paper6 + broad).
- **Submitted**: `task_id=t-20260707172056-fwktl`, job `eval_t3_representation_globalfeat`, initial status `Queue`, `Preemptible: false`, 1× `ml.pni2.3xlarge`.
- **Final**: **Success** (ExitCode 0, elapsed ~907s, end 2026-07-07T09:36:03Z). Outputs: `outputs/tcr_representation{,_paper6}/ours_globalfeat_*` + `_summary.json`.
- **Log**: `output/downstream_generation/eval_t3_representation_globalfeat_volc.log`

## 2026-07-12 7L 下游状态核对 + mint Pdb-bind 补提

- **核对**：先前 8 作业均已 **Success**（不在跑）：t1–t4 / ab / flab / nbbench 产物齐全；mint 中 HumanPPI/Bernett/YeastPPI/MutationalPPI/SKEMPI ✅，**Pdb-bind 失败**。
- **根因**：`run_all_downstream.py` 用任务名 `PDB-Bind`，`tasks.TASK_CONFIGS` 键为 `Pdb-bind` → `get_task_datasets` 返回 `0` → TypeError。
- **修复**：orchestrator 改为 `Pdb-bind`；`tasks.py` 增加别名并在未知任务时 raise（不再静默返回 0）。
- **补提**：`eval_jobs/eval_integrated_mint.yml` → `task_id=t-20260712192011-z5r98`，初始 `Initialized`，单卡 `ml.pni2.3xlarge`，`Preemptible: true`。

## 2026-07-14 Submitted full downstream eval for 7L step101300 latest checkpoint

- **Checkpoint**：从 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l/latest.pt` 固定硬链接快照到 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step101300/best.pt`；checkpoint 内 `step=101300`，size=`5227938475`，sha256=`e076419966276235a103e9e76fc7e9a07a72ef7f0b4005f2f859661bbf531ec1`。manifest：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step101300/checkpoint_manifest.json`。
- **隔离**：run/tag 均带 `step101300`，不会覆盖旧 best.pt 的 T1/T2/T3/MINT/AB/FLAb/NbBench 产物。生成器和提交脚本新增可选 job/record prefix，默认行为保持不变。
- **T4 口径修复**：旧 orchestrator 只重新生成 `ours_bioseq` 样本、未重新评分，collector 因此读取 2026-07-05 的 stale metrics（旧报告的 JSD=0.2952 不具 checkpoint provenance）。现改为 checkpoint-specific tag，并依次跑 Setting A/B/C 评分后收集 JSD、PGen、条件生成与全长有效性指标。
- **Submit**：8 个单卡 `ml.pni2.3xlarge`、`Preemptible: true` 作业，初始状态为 `Initialized/Staging`：
  - t1 `t-20260714224022-5w957`；t2 `t-20260714224025-5dmt6`；t3 `t-20260714224028-7jg6p`；t4 `t-20260714224032-wpt5h`
  - mint `t-20260714224035-jml98`；ab `t-20260714224038-xdwdj`；flab `t-20260714224042-w2hcq`；nbbench `t-20260714224045-5nf62`
- **YAML**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_integrated_step101300_{t1,t2,t3,t4,mint,ab,flab,nbbench}.yml`。
- **Task IDs / submit log**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_step101300_task_ids.tsv`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_step101300_submit.log`。
- **14:43Z progress**：t2 `t-20260714224025-5dmt6` **Success**（elapsed 175s）；step101300 Basis-B `ARI=0.0188 / NMI=0.1169 / Purity=0.2947`。
- **14:45Z progress**：t1 `t-20260714224022-5w957` **Success**（elapsed 236s），seen AUPRC/AUROC=`0.5502/0.5050`、unseen=`0.4871/0.5005`；t3 `t-20260714224028-7jg6p` **Success**（elapsed 253s），few-shot AUROC k5/20/100/200=`0.6343/0.6614/0.6886/0.6986`。
- **14:47Z progress**：flab `t-20260714224042-w2hcq` **Success**（elapsed 354s）；Spearman g6_Kd/g6_er/trastuzumab_kd/d44_Kd=`0.5557/0.7207/0.4392/0.4441`。
- **14:49Z progress**：t4 `t-20260714224032-wpt5h` **Success**（elapsed 491s）；Setting-A JSD/novelty/PGen-positive=`0.2911/1.0000/0.9747`；Setting-B overall F1=`0.0000`、d_edit=`6.7244`、seq-recovery=`0.4405`；Setting-C valid-AA=`1.0000`、CDR3-extracted=`0.0020`（1/500）。
- **T4 fair retro-score**：对 2026-07-12 保留的旧 best（step≈41000）样本离线同口径重评分（不重新生成）：Setting-A JSD/PGen-positive=`0.2726/0.6620`；Setting-B F1/d_edit/seq-recovery=`0.0000/6.5647/0.4419`；Setting-C CDR3-extracted=`0.0260`（13/500）。因此 step101300 的 PGen 合法性明显改善，但 repertoire JSD、条件最近邻距离和全长规范性均退化。产物 tag=`ours_bioseq_esmc300m_integrated_llada_7l_step41000_retro`。
- **15:09Z progress**：mint `t-20260714224035-jml98` 与 nbbench `t-20260714224045-5nf62` 均 **Success**（elapsed 1672s / 1682s）。MINT：Human/Bernett/Yeast AUROC=`0.7865/0.6027/0.5986`，MutationalPPI AUROC/AUPRC=`0.6030/0.1688`，SKEMPI Spearman/RMSE=`0.3620/0.8496`，Pdb-bind Spearman/RMSE=`0.6322/0.8117`。NbBench：VR acc=`0.9568`、Paratope AUPRC=`0.6179`、probe CDR BR=`1.5033`；generative CDR EM/BR=`0.5615/2.6864`。
- **MINT/NbBench 汇总修复**：collector 过去只保留 MINT 三个分类任务且 NbBench 只写 `present=true`；现统一收集 MINT 六任务实际 schema（含 Accuracy/F1/回归指标）以及 NbBench 9 scalar + 3 residue + generative CDR 指标。最终在全部作业终态后重跑一次全量 collect，避免并行作业写出的 task-only summary 互相覆盖。
- **15:46Z final**：ab `t-20260714224038-xdwdj` **Success**（elapsed 3921s），因此 8/8 作业全部 Success。CDR-H1/H2/H3 all-fold AAR=`77.1121/71.9674/54.7963`；light-pairing ImmunoMatch=`0.5935`。已用修复后的 collector 重写全量 summary，包含 t1/t2/t3/t4/mint/ab/flab/nbbench 全 8 组、87 个数值字段、0 errors。
- **15:51Z latest-only clarification**：用户明确只看最新 checkpoint。逐项复核 8 份正式评测日志，其启动行全部为 `/output/grammar_v2_esmc300m_integrated_llada_7l_step101300/best.pt`；没有任何正式 task 使用旧 step41000 checkpoint。`STEP101300_DOWNSTREAM_REPORT.md` 已重写为只包含 step101300 绝对结果，不再展示旧 checkpoint 对照。

## 2026-07-14 Baseline-only paper alignment（覆盖旧 baseline 排名结论）

- **范围**：只修 downstream baseline、评测协议与文档；没有修改模型、训练配置或 checkpoint，也没有提交新 Volc/GPU 作业。
- **统一证据层**：`[P]` 论文原值、`[A]` 官方 prediction artifact 重评分、`[R]` 官方代码重跑、`[L]` 本地重实现、`[C]` 本地控制。协议不一致时 `[P]` 为主，其他来源不得混表排名。
- **主要修复**：T2 LD indel lookup；T4 sparse-13 target list；MINT 回归 inverse-transform；FLAb nested 10×5-fold R²；Ophiuchus CDR fold 路径；NbBench paper/local leaderboard 拆分。
- **撤回**：旧 MINT SKEMPI/PDB-Bind transformed-space Pearson/RMSE 及领先结论；旧 FLAb 5-fold Spearman 论文对比；本地 CDR masked-fill 与 Ophiuchus Table 2 混排；NbBench sklearn probe 与 Table 5 混排；T4 benchmark14 与 sparse-13 混排。
- **论文替代值**：T1 Supplementary Table 4、T2 NAR-GAB 主表、MINT Source Data、Ophiuchus-Ab Table 2、MINT Figure 3b、NbBench Table 5 已导入 `downstream/benchmark/outputs/external/paper_reported_baselines.csv`（282 rows）并在表格标 `[P]`。
- **验证**：baseline protocol tests 4 passed；strict provenance audit 0 error / 0 warning；Python compile 通过。机器没有 LaTeX engine，改做静态 brace/table/tabular/列数检查，0 error。
- **pending**：MINT raw-unit regression、FLAb nested-R²、Ophiuchus exact local CDR 为 baseline-only 重跑；NbBench exact MLP/3-seed 可选。旧历史记录若与本条冲突，以本条和 `BASELINE_VERIFICATION.md` 为准。

## 2026-07-20 Submitted full downstream eval for retained 7L step189000 checkpoint

- **Checkpoint**：将 Top-K 文件 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l/checkpoints/step_0189000_val_0.4913.pt` 以硬链接固定为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step189000/best.pt`；内部 `step=189000`，validation loss=`0.49125400149843346`，size=`5227967829`，SHA256=`b5388deccff040641e04c90b9e769cf89cd5160fe932148bd465de4ca8f957ca`。Manifest：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step189000/checkpoint_manifest.json`。
- **范围**：T1、T2、T3、T4、MINT、AB、FLAb、NbBench 全 8 个任务族；run/tag 均带 `step189000`，不会覆盖其他 checkpoint 产物。
- **提交记录**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_step189000_task_ids.tsv`；提交日志 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_step189000_submit.log`。
- **10:17Z canceled**：这组对照与用户要求的 step338600 AB/FLAb 补测共用默认单卡池，step189000 T4 已占用 GPU、而必需补测仍在 Queue。为优先闭环最新 checkpoint，已 cancel 8/8 step189000 作业；8 个作业均已 Killed。未将任何 step189000 部分产物纳入正式结果。
- **10:18Z retry1 submit**：用户再次明确要求测试 step189000，因此重新提交完整 8 个任务族；T1=`t-20260720181809-ppm5l`、T2=`t-20260720181813-76wjp`、T3=`t-20260720181817-tpw2c`、T4=`t-20260720181820-cr6fq`、MINT=`t-20260720181823-4p9wl`、AB=`t-20260720181826-rjqct`、FLAb=`t-20260720181830-jw2wm`、NbBench=`t-20260720181833-nflxb`。提交后 8/8 均为 Queue；记录 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_step189000_retry1_task_ids.tsv`，日志 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_step189000_retry1_submit.log`。首轮 T4 的部分生成产物不计为正式结果，retry1 成功后统一重汇总。
- **2026-07-21 04:23Z terminal reconciliation**：AB `t-20260720181826-rjqct` 与 FLAb `t-20260720181830-jw2wm` 均 Success；T1/T2/T3/T4/MINT/NbBench 在 `2026-07-20T11:42:39Z`–`11:42:40Z` 统一进入 Killed，且没有对应的完整 step189000 产物，因此当前完成度为 **2/8**，不能标记为全量完成。
- **AB final**：CDR-H1/H2/H3 all-fold AAR=`77.5696/72.5663/55.8712`；light-pairing generated ImmunoMatch=`0.606259`。其中 H3 仅比 step194100 低 `0.1703` 个百分点、比 step338600 高 `22.1974` 个百分点；pairing 高于 step194100 `0.013540`，且远高于 step338600 的 `0.000403`。
- **FLAb final**：g6_Kd/g6_er/trastuzumab_kd/d44_Kd nested 10×5 CV mean R²=`0.2651/0.5270/0.1878/0.2418`，四项均值=`0.3054`；高于 step194100 均值 `0.2799` 和 step338600 均值 `0.2758`。对论文 MINT 同协议值，只有 g6_Kd 略高（`0.2651` vs `0.2530`），其余三项仍低。
- **Partial artifacts**：机器可读汇总 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step189000_partial.json`；阶段报告 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/STEP189000_DOWNSTREAM_REPORT.md`。
- **初始状态**：T4=`Staging`；其余 7 项=`Queue`。T1/T2/T3/T4/MINT/NbBench 为 `Preemptible: true`；长任务 AB/FLAb 为 `Preemptible: false`。
- **Task IDs / YAMLs**：
  - T1 `t-20260720181132-4k7c9` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_integrated_step189000_t1.yml`
  - T2 `t-20260720181136-cfnfv` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_integrated_step189000_t2.yml`
  - T3 `t-20260720181139-6jtqg` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_integrated_step189000_t3.yml`
  - T4 `t-20260720181143-ss7t8` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_integrated_step189000_t4.yml`
  - MINT `t-20260720181146-pd4nk` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_integrated_step189000_mint.yml`
  - AB `t-20260720181150-qc658` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_integrated_step189000_ab.yml`
  - FLAb `t-20260720181154-97lnw` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_integrated_step189000_flab.yml`
  - NbBench `t-20260720181157-5zzfm` — `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_integrated_step189000_nbbench.yml`

## 2026-07-21 Resubmitted missing step189000 downstream families on stable resources

- **Operation**：只补交 retry1 缺失的 T1、T2、T3、T4、MINT、NbBench；已 Success 的 AB/FLAb 不重复运行。6 个任务统一 `Preemptible: false`、Priority `6`、单卡 `ml.pni2.3xlarge`。
- **Checkpoint validation**：所有入口均指向绝对路径 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step189000/best.pt`；提交前复核 SHA256=`b5388deccff040641e04c90b9e769cf89cd5160fe932148bd465de4ca8f957ca`，与 manifest 一致；6 份 YAML 均通过解析和 Entrypoint `bash -n`。
- **Task IDs**：T1=`t-20260721125137-4swz5`，T2=`t-20260721125140-bdd92`，T3=`t-20260721125144-6zzg8`，T4=`t-20260721125147-527rg`，MINT=`t-20260721125150-9cxk6`，NbBench=`t-20260721125153-cqsbq`。
- **YAMLs**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_integrated_step189000_retry2_stable_{t1,t2,t3,t4,mint,nbbench}.yml`。
- **Submission record**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_step189000_retry2_stable_task_ids.tsv`；日志 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_step189000_retry2_stable_submit.log`。
- **Initial status @04:52Z**：T1=`Running`；T2/T3/T4/MINT/NbBench=`Queue`。
- **04:56Z T1 final**：T1 `t-20260721125137-4swz5` Success；seen AUPRC/AUROC=`0.5284/0.4884`，unseen=`0.5149/0.5247`。正式 task summary 为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step189000_t1.json`；日志有 done marker、无失败。T2 已进入 Running。
- **04:58Z T2 final**：T2 `t-20260721125140-bdd92` Success；ARI/NMI/Purity=`0.0202/0.1223/0.3007`。正式 task summary 为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step189000_t2.json`；日志有 done marker、无失败。T3 已进入 Running。
- **05:02Z T3 final**：T3 `t-20260721125144-6zzg8` Success；few-shot AUROC k5/20/100/200=`0.6357/0.6591/0.6860/0.6970`。正式 task summary 为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step189000_t3.json`；日志有 done marker、无失败。T4 已进入 Running。
- **05:10Z T4 final**：T4 `t-20260721125147-527rg` Success；Setting-A JSD/novelty/PGen-positive=`0.3588/1.0000/0.9970`；Setting-B overall F1/d_edit/seq-recovery/diversity=`0.0000/6.5774/0.4656/1.0000`；Setting-C valid-AA/CDR3-extracted=`1.0000/0.0000`。retry2 重新生成了 conditional/unconditional/full-length 全部样本并重算指标，首轮半成品已排除。MINT 已进入 Running。
- **05:38Z MINT final**：MINT `t-20260721125150-9cxk6` Success；六任务/22 个 canonical 指标齐全。HumanPPI AUROC/AUPRC=`0.7468/0.7515`，Bernett=`0.5822/0.5909`，YeastPPI=`0.5761/0.5838`，MutationalPPI=`0.5898/0.1643`，SKEMPI Pearson/Spearman/RMSE=`0.4205/0.3655/1.8385`，Pdb-bind=`0.6382/0.6313/1.4825`。日志有 done marker、无失败；NbBench 已进入 Running。

## 2026-07-21 Submitted step189000 NbBench stall diagnostic

- **Reason**：NbBench retry2-stable 主任务 `t-20260721125153-cqsbq` 实际 Running 满 60 分钟仍未写出首个 `nanobody_type/metrics.json`；主任务状态、实例状态与系统 stderr 均无错误，因此暂不取消，先用独立输出做最小化诊断。
- **Submit**：`t-20260721143849-2h7rr`，YAML `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_integrated_step189000_nbbench_nanobody_diag.yml`，初始状态 Staging，`Preemptible: true`，单卡 `ml.pni2.3xlarge`，deadline 7200s。
- **Isolation**：仅运行 `nanobody_type`；tag=`ours_esmc300m_integrated_llada_7l_step189000_diag`，日志 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_grammar_v2_esmc300m_integrated_llada_7l_step189000_nbbench_nanobody_diag_volc.log`；不会覆盖正式 NbBench 产物。入口设置 `PYTHONUNBUFFERED=1` 便于观测首项进度。
- **06:40Z diagnostic final**：诊断任务 `t-20260721143849-2h7rr` **Success**；同一 checkpoint 的 `nanobody_type` 在 77.2 秒完成，`probe_acc=0.9952`，说明 checkpoint、数据和 NbBench 单项路径均正常。
- **06:41Z recovery**：retry2 主任务已实际运行约 64 分钟，仍未写出首项指标；结合隔离诊断成功，确认该实例异常卡住。已 cancel `t-20260721125153-cqsbq`，当前状态 **Killing**；待进入 Killed 后以非抢占、unbuffered 的 retry3 配置重提完整 NbBench。
- **06:42Z cancel final**：retry2 主任务 `t-20260721125153-cqsbq` 已进入终态 **Killed**，已从活跃表移除。retry3 YAML 已通过解析与 Entrypoint `bash -n`；checkpoint SHA256 复核仍为 `b5388deccff040641e04c90b9e769cf89cd5160fe932148bd465de4ca8f957ca`。
- **06:43Z retry3 submit**：已提交完整 NbBench 恢复任务 `t-20260721144334-g9n2c`，初始状态 **Initialized**；YAML `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_integrated_step189000_retry3_stable_nbbench.yml`，`Preemptible: false`，正式 canonical tag 不变，并启用 `PYTHONUNBUFFERED=1`。提交记录 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_step189000_retry3_stable_task_ids.tsv`。
- **06:44Z retry3 running**：任务已进入 **Running**，实际入口于 `06:43:55Z` 启动，正式输出首项 `nanobody_type` 已开始。
- **07:11Z retry3 final**：完整 NbBench 恢复任务 `t-20260721144334-g9n2c` **Success**，已从活跃表移除；12/12 主任务及生成式 CDR 指标齐全。关键结果：nanobody-type acc=`0.9952`，polyreaction AUROC=`0.8342`，thermo-tm/seq Spearman=`0.7081/0.6366`，VR acc=`0.9430`，Paratope AUPRC=`0.6232`，CDR probe BR=`1.4444`，生成式 BR/EM/exact=`2.7672/0.5712/0.0007`。task summary：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step189000_nbbench.json`。至此 step189000 下游 **8/8** 完成。
- **07:14Z final aggregation**：已生成全量汇总 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step189000.json` 和最终报告 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/STEP189000_DOWNSTREAM_REPORT.md`。独立校验结果为 8 个 task family、86 个有限数值字段，与 8 份 task summary 逐项一致；全部正式日志有 done marker，未发现 Traceback/OOM/失败子步骤，checkpoint SHA256 与 manifest 一致。T4 因 `0/500` CDR3 extracted，条件 JSD 无定义而未生成，这不是缺失任务。

## 2026-07-21 Froze all currently retained untested Top-K checkpoints for downstream evaluation

- **Inventory cutoff**：`2026-07-21T07:17:38Z` 读取训练目录 `checkpoints/topk_manifest.json`；当前 `save_top_k=5`，按 val loss 排序为 step117000 (`0.4430236166`)、step164000 (`0.4786416662`)、step359000 (`0.4832039739`)、step121000 (`0.4903497546`)、step189000 (`0.4912540015`)。
- **Deduplication**：step189000 已完成 8/8 下游测试，因此不重复提交；step117000、step121000、step164000、step359000 尚无完整下游汇总，本轮范围为 **4 checkpoints × 8 families = 32 jobs**。
- **Identity validation**：五个 retained 文件的 manifest step 均与 checkpoint 内部 `step` 一致；四个未测文件 size 均为 `5227967829` bytes。
- **Immutable snapshots**：已将四个 retained Top-K 文件以硬链接固定为 `output/grammar_v2_esmc300m_integrated_llada_7l_step{117000,121000,164000,359000}/best.pt`，并分别写入 `checkpoint_manifest.json`。SHA256：step117000=`e3d4c98ea84b4aa2280fe76ec5bf8a806de376699610e9a96960e66ceddb5c01`；step121000=`51f0eee5fa7b127a0487a2ea8fdbc322bda6f6908ff2662087bd87d4980d4613`；step164000=`929d9d2f6c57ce15a407b147de3b39daabd5d5bb934011ed074c22791f845d14`；step359000=`9b4ea96b9969ca795f3c149d75ed34c789d062e38a2a8fa2aea1b3b7613d70c6`。
- **Config validation**：生成 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_integrated_step{117000,121000,164000,359000}_topk_stable_{t1,t2,t3,t4,mint,ab,flab,nbbench}.yml` 共 32 份；TaskName 全部唯一、YAML/Entrypoint 语法通过、checkpoint 路径与 family 参数正确，均为非抢占资源并设置 `PYTHONUNBUFFERED=1`。
- **07:20Z–07:22Z submit**：32/32 作业提交成功；初始查询 step117000 T1=`Running`，其余 31 个=`Queue`。task ID、初始状态和绝对 YAML 路径详见 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_topk_remaining_20260721_task_ids.tsv`。
- **07:24Z step117000 T1 final**：`t-20260721152058-8dwx9` **Success**，已从活跃表移除；seen AUPRC/AUROC=`0.5555/0.5159`，unseen=`0.5224/0.5058`。summary=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step117000_t1.json`；日志有 done marker 且无错误。step117000 T2/T3 已进入 Running；总体完成度 `1/32`。
- **07:26Z step117000 T2 final**：`t-20260721152102-c7wvq` **Success**，已从活跃表移除；ARI/NMI/Purity=`0.0200/0.1187/0.2984`。summary=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step117000_t2.json`；日志有 done marker 且无错误。step117000 T3=`Running`、T4=`Staging`；总体完成度 `2/32`。
- **07:27Z status**：step117000 T4 `t-20260721152110-g5g7r` 已由 Staging 转为 **Running**；T3 同时保持 Running。
- **07:28Z step117000 T3 final**：`t-20260721152106-bl79m` **Success**，已从活跃表移除；few-shot AUROC k5/20/100/200=`0.6347/0.6597/0.6885/0.6995`。summary=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step117000_t3.json`；日志有 done marker 且无错误。MINT 已进入 Running；总体完成度 `3/32`。
- **07:34Z step117000 T4 final**：`t-20260721152110-g5g7r` **Success**，已从活跃表移除；Setting-A JSD/novelty/PGen-positive=`0.5106/1.0000/0.9917`，Setting-B overall d_edit/recovery=`7.0491/0.4346`，Setting-C valid-AA/CDR3-extracted=`1.0000/0.0000`。summary=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step117000_t4.json`；日志有 done marker 且无错误。AB 已进入 Staging；总体完成度 `4/32`。
- **07:34Z status**：step117000 AB `t-20260721152118-l99rm` 已由 Staging 转为 **Running**；MINT 同时保持 Running。
- **07:55Z step117000 MINT final**：`t-20260721152114-rtgnj` **Success**，已从活跃表移除；HumanPPI AUROC/AUPRC=`0.7261/0.7222`，Bernett=`0.5877/0.5956`，YeastPPI=`0.5867/0.5951`，MutationalPPI=`0.5945/0.1684`，SKEMPI Pearson/Spearman/RMSE=`0.4244/0.3932/1.8521`，Pdb-bind=`0.6538/0.6496/1.4631`。summary=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step117000_mint.json`；22 个 canonical 指标齐全、日志无错误。
- **07:58Z–08:05Z step121000 T1/T2/T3 final**：T1 `t-20260721152130-lrmfb`、T2 `t-20260721152134-xgc4t`、T3 `t-20260721152138-r8mhm` 均 **Success** 并从活跃表移除。T1 seen AUPRC/AUROC=`0.5318/0.4908`、unseen=`0.4781/0.4794`；T2 ARI/NMI/Purity=`0.0213/0.1262/0.3059`；T3 k5/20/100/200=`0.6332/0.6609/0.6899/0.6995`。三份 summary 与 done marker 齐全，日志无错误。
- **08:08Z status**：step117000 AB/FLAb/NbBench 与 step121000 T4 为 **Running**；其余 28 个为 Queue。总体 `8/40` Success、`4/40` Running、无 Failed/Killed。

## 2026-07-21 Froze current training checkpoint step384800 for downstream evaluation

- **Training state**：训练任务 `t-20260713235356-8t6c9` 保持 Running。用户要求同时测试目前训练到的 checkpoint，因此在 `2026-07-21T07:40:52Z` 将当时的 mutable `latest.pt` inode 固定为独立快照。
- **Snapshot**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step384800/best.pt`；checkpoint 内部 `step=384800`，size=`5227942507` bytes，source inode=`25871519244`，source mtime=`2026-07-21T07:39:42.615506595Z`，SHA256=`a837764b939294bbc6f6e0f48b1220cec76e94ec0afc726b5c94358853d5ba07`。manifest=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step384800/checkpoint_manifest.json`。
- **Deduplication**：step384800 不在已完成集合 `{101300,189000,194100,338600}`，也不在已提交的 retained Top-K 集合 `{117000,121000,164000,359000}`，因此需新增完整 T1/T2/T3/T4/MINT/AB/FLAb/NbBench 8-family 测试。
- **Config validation**：8 份 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_integrated_step384800_latest_stable_<family>.yml` 均通过 YAML、Entrypoint `bash -n`、唯一 TaskName、绝对 checkpoint 路径和 family 参数校验；统一 `Preemptible: false`、`PYTHONUNBUFFERED=1`。
- **07:41Z–07:42Z submit**：step384800 的 8/8 作业均提交成功，初始查询均为 **Queue**。T1=`t-20260721154148-hxw5h`、T2=`t-20260721154151-xzltk`、T3=`t-20260721154154-5jrgr`、T4=`t-20260721154158-6vsmh`、MINT=`t-20260721154202-q5nsx`、AB=`t-20260721154206-gc2rz`、FLAb=`t-20260721154209-f8q9r`、NbBench=`t-20260721154213-jlhs6`。记录：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_step384800_latest_task_ids.tsv`。
- **Snapshot immutability check**：提交后训练已将 mutable `latest.pt` 原子替换为新 inode `25871519240`（mtime `07:42:45Z`），而 step384800 snapshot 保持原 inode `25871519244`、size/SHA 不变，证明本轮评测不会追随继续训练的权重变化。

## 2026-07-21 Cancelled 7L training `8t6c9` (user request)

- **操作**：cancel `t-20260713235356-8t6c9`（`bioseq_esmc300m_integrated_llada_7l_3node_q012_resume`）
- **YAML**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_grammar_v2_esmc300m_integrated_llada_7l_resume_queue012.yml`
- **状态**：Running → Killing → **Killed**；`Preemptible: false`；queue012；非终态 7L 训练任务数 = 0
- **停前进度**：`latest.pt` step ≈ **389500**（mtime 2026-07-21 10:01）
- **watchdog**：loop PID 已 kill；`scripts/llada_train_watchdog_jobs.json` 中 `integrated_7l_3node_q012` 设 `paused=true`、`monitor_only=true`，防止 Failed/Killed 自动重提

## 2026-07-21 Completed retained Top-K/step384800 sweep and submitted final step389500

- **12:46Z platform reconciliation**：retained Top-K step117000/121000/164000/359000 与 cutoff step384800 的原始 **40/40** 个 Volc 作业全部进入 `Success`；这些终态作业均已从 Active 表移除。已生成五份全量机器可读汇总：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step{117000,121000,164000,359000,384800}.json`。
- **Completeness audit**：step117000、step121000、step164000、step384800 的 8 个任务族及预期子指标齐全。step359000 的 FLAb parent 尽管平台为 Success，但 `g6_Kd`、`g6_er`、`d44_Kd` 三个内部命令分别遇到 `torch.AcceleratorError: CUDA error: unspecified launch failure`；只有 `trastuzumab_kd:R2=0.1741` 已完成，因此该 checkpoint 暂不能标记为完整 8/8 指标闭环。
- **Zero-value collector fix**：step359000 与 step384800 light-pairing 的原始 metrics 均为 `gen_immunomatch_mean=0`、`gen_valid_rate=0`、`both_valid_count=0/4000`，属于真实模型结果。修复 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_all_downstream.py` 中用 truthiness 选择兼容字段的问题，汇总现正确保留 `0.0`，不再错误写成 `null`；`pllm` 环境 `py_compile` 与重新 collect 通过。
- **12:51Z FLAb recovery submit**：缺失三项已拆成独立、非抢占、fail-fast 作业并均进入 Running：g6_Kd=`t-20260721205134-vlwc8`、g6_er=`t-20260721205141-ttvjj`、d44_Kd=`t-20260721205148-tzqm5`。
- **Final checkpoint snapshot**：训练停止后读取 mutable `latest.pt` 内部 `step=389500`，固定为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step389500/best.pt`；size=`5227942507`，mtime=`2026-07-21T10:01:20.054835410Z`，SHA256=`5d1231992b65aa75a883e86662f34da136fa87e8ba031b804ff4c942ab1326c9`；manifest=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step389500/checkpoint_manifest.json`。
- **12:51Z–12:52Z final eval submit**：step389500 完整 T1/T2/T3/T4/MINT/AB/FLAb/NbBench 8 个非抢占作业全部提交成功。初始状态 T1=`Running`，其余 7 个=`Queue`；task IDs 与绝对 YAML 路径记录在 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_step389500_final_and_step359000_flab_retry_task_ids.tsv`。
- **12:54Z–12:57Z step389500 T1/T2 final**：T1 `t-20260721205153-wh9ds` 与 T2 `t-20260721205202-wwfmt` 均 `Success` 并从 Active 表移除。T1 seen AUPRC/AUROC=`0.5015/0.4606`、unseen=`0.5274/0.5137`；T2 ARI/NMI/Purity=`0.0159/0.1043/0.2840`。两份 task summary 与 done marker 齐全，未发现 Traceback/CUDA/OOM/非零子步骤；T3 已进入 Running。
- **13:01Z step359000 FLAb recovery final**：g6_Kd `t-20260721205134-vlwc8`、g6_er `t-20260721205141-ttvjj`、d44_Kd `t-20260721205148-tzqm5` 均 `Success`，日志无 Traceback/CUDA/OOM；nested-CV mean R² 分别为 `0.1880/0.4459/0.1912`。结合原 parent 已完成的 trastuzumab_kd=`0.1741`，FLAb four-task mean=`0.2498`。重新 collect 后 step359000 为 8 families / 87 个有限指标，8 份 task summary 逐项一致；最终 summary=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada_7l_step359000.json`。
- **13:01Z step389500 T3 final**：`t-20260721205210-xrn5x` `Success` 并从 Active 表移除；few-shot AUROC k5/20/100/200=`0.6202/0.6526/0.6845/0.6942`，summary 与 done marker 齐全。T4/MINT/AB/FLAb 已进入 Running，NbBench 保持 Queue。
- **Comparison report**：当前跨 checkpoint 决策表写入 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/CURRENT_CKPT_SWEEP_REPORT.md`。
- **13:07Z step389500 T4 final**：`t-20260721205216-fbk7g` `Success` 并从 Active 表移除；Setting-A JSD/novelty/mean-NN-distance/PGen-positive=`0.5303/1.0000/5.8542/0.9513`，Setting-B overall d_edit/seq-recovery/diversity/F1=`7.6665/0.3795/1.0000/0.0000`，Setting-C valid-AA/CDR3-extracted=`1.0000/0.0000`。summary 与 done marker 齐全，日志无错误。
- **13:12Z light-pairing prompt/iteration ablation submit**：按用户要求将 light 前 3 个 residue 由固定 prompt 改为全部生成（`light_prompt_tokens=0`），并把解码 `max_iter` 提升到 `128`。为区分两个因素，在 step384800 上同时提交 prompt0/iter32、prompt3/iter128、prompt0/iter128 三臂；另对 pairing 既有最优 step121000 与最终 step389500 提交 prompt0/iter128。五个作业均保持 canonical holdout500、每条 heavy 生成 8 条 light、heavy batch size 4、seed 42、`gumbel_argmax`，统一非抢占且使用独立输出前缀。
- **13:12Z ablation task IDs**：step384800 prompt0/iter32=`t-20260721211212-294pw`、prompt3/iter128=`t-20260721211217-qfld4`、prompt0/iter128=`t-20260721211221-g7ngf`；step121000 prompt0/iter128=`t-20260721211230-ss4cm`；step389500 prompt0/iter128=`t-20260721211237-dksg6`。记录：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_pairing_prompt_iter_ablation_20260721_task_ids.tsv`。
- **13:14Z status**：五个 pairing ablation 作业均为 **Queue**；step389500 原完整评测为 4/8 Success、MINT/AB/FLAb/NbBench 4/8 Running。
- **13:26Z–14:04Z step389500 remaining families final**：MINT `t-20260721205222-lrb8h`、FLAb `t-20260721205230-2g5z8`、NbBench `t-20260721205235-dh7dk`、AB `t-20260721205226-ccc99` 依次进入 **Success** 并从 Active 表移除；至此最终 step389500 原完整评测为 **8/8 Success**。AB canonical prompt3/iter32 的 light-pairing ImmunoMatch=`0`、generated valid rate=`0/4000`。
- **14:44Z ablation first final**：step384800 prompt3/iter128 `t-20260721211217-qfld4` **Success**；ImmunoMatch=`0`、generated valid rate=`0/4000`、diversity mean=`0.05144`。相较同 checkpoint canonical prompt3/iter32（ImmunoMatch=`0`、valid=`0/4000`），仅把 iteration 从 32 提到 128 没有恢复可识别 light-chain，当前看不到收益。
- **14:46Z ablation failure audit**：step384800 prompt0/iter32 `t-20260721211212-294pw`、step384800 prompt0/iter128 `t-20260721211221-g7ngf`、step121000 prompt0/iter128 `t-20260721211230-ss4cm` 均在第一个 heavy batch 遇到 `torch.AcceleratorError: CUDA error: unspecified launch failure`，无 generation CSV/metrics，属于运行失败而不是模型得分；已从 Active 表移除，准备保持协议不变重提。step389500 prompt0/iter128 已完成 125/125 generation batches 和 4000 条 CSV，正在 ImmunoMatch scoring，任务保持 Running。
- **14:50Z ablation retry1 submit**：三个 transient CUDA 失败项保持 checkpoint、holdout500、prompt/iteration、num-seqs=8、heavy-batch-size=4、seed42、gumbel_argmax 全部不变，以非抢占资源重提。step384800 prompt0/iter32=`t-20260721225024-pjft2`、step384800 prompt0/iter128=`t-20260721225028-hz72m`，初始均 Staging；step121000 prompt0/iter128=`t-20260721225032-w8m7q`，初始 Queue。记录：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_pairing_prompt_iter_ablation_retry1_20260721_task_ids.tsv`。
- **14:51Z retry1 running**：三个 retry1 作业均已进入 **Running**；step389500 prompt0/iter128 同时保持 Running、metrics 尚未落盘。
- **14:56Z stop later-checkpoint ablation**：用户明确只继续测试前期效果较好的 checkpoint、不再测试 389xxx。已 cancel step389500 prompt0/iter128 `t-20260721211237-dksg6`，平台终态 **Killed**，并从 Active 表移除。取消前 generation 已完成 125/125、留下 4000-row CSV，但 ImmunoMatch metrics 尚未产生，因此该 partial artifact 不作为正式结果。现仅保留 step121000 与 step384800 的三个 retry1 Running 作业。
- **15:00Z narrow ablation to early best checkpoints**：重新按 canonical prompt3/iter32 ImmunoMatch 排名，step121000=`0.608464`、step189000=`0.606259` 为前两名；step384800=`0` 不属于“前面效果好的 ckpt”。因此 cancel step384800 prompt0/iter32 retry1 `t-20260721225024-pjft2` 与 prompt0/iter128 retry1 `t-20260721225028-hz72m`，二者均已 **Killed** 并从 Active 表移除。
- **15:01Z early-best step189000 submit**：新增 step189000 prompt0/iter128 非抢占全量 holdout500 作业 `t-20260721230134-wdqd9`，初始 **Staging**；checkpoint SHA256=`b5388deccff040641e04c90b9e769cf89cd5160fe932148bd465de4ca8f957ca` 与 manifest 一致。当前只保留 step121000 prompt0/iter128 Running 与 step189000 prompt0/iter128 Staging。记录：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_pairing_early_best_prompt0_iter128_20260721_task_ids.tsv`。
- **15:02Z early-best running**：step189000 prompt0/iter128 已由 Staging 进入 **Running**；当前两个早期最佳 checkpoint 对照均为 Running。
- **16:03Z step121000 ablation final**：prompt0/iter128 retry1 `t-20260721225032-w8m7q` 平台 **Success**，4000/4000 generated pairs 均有效；ImmunoMatch=`0.572159`、Gen>Ref ratio=`0.3270`、diversity mean=`0.077497`。相较 canonical prompt3/iter32 的 `0.608464`，ImmunoMatch 下降 `0.036305`（相对 `-5.97%`）。metrics=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/grammar_v2_esmc300m_integrated_llada_7l_step121000_light_pairing_holdout500_prompt0_iter128_metrics.json`。
- **16:14Z step189000 ablation final**：prompt0/iter128 `t-20260721230134-wdqd9` 平台 **Success**；3913/4000 generated pairs 有效（valid rate=`0.97825`），ImmunoMatch=`0.574066`、Gen>Ref ratio=`0.35175`、diversity mean=`0.122550`。相较 canonical prompt3/iter32 的 `0.606259`，ImmunoMatch 下降 `0.032193`（相对 `-5.31%`），valid rate 由 `0.9995` 降至 `0.97825`。metrics=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/grammar_v2_esmc300m_integrated_llada_7l_step189000_light_pairing_holdout500_prompt0_iter128_metrics.json`。
- **16:23Z ablation conclusion**：两个早期最佳 checkpoint 均已完成且日志有 done marker、无 Traceback/CUDA/OOM；本次联合改变 prompt3→prompt0 与 iter32→128 后，ImmunoMatch 均下降，因此不能据此认为更多 iteration 有提升。该两因素联合实验不能单独归因 prompt 或 iteration；当前正式推荐仍为 canonical prompt3/iter32。

## 2026-07-21 Public epitope-conditioned CDR3β Track A completed locally

- **Scope**：按新严格口径先完成 TCRT5、GRATCR、TcrDesign beta-only，再追加本地 BioSeq-7L-step117000；四个可比较模型均为每 target 1,000 条、14 targets，canonical 表共 56,000 raw candidates。未提交 Volc 作业。
- **TCR-epiDiff 判定**：发布推理路径为 `original_TCR[0] -> add_noise -> denoise`，无纯随机 `x_T` reverse sampler，故标记 `reconstruction_only`，未使用真实测试 TCR 作种子，也未进入 de-novo 排名。
- **数据审计**：14 targets、1,312 条去重 reference；`RVRAYTYSK_HLA-A*03:01` 有 895 条；预测列没有被当作 reference。审计文件为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_beta_public_benchmark/benchmark_audit.json`。
- **全量结果**：TCRT5 Valid/Unique/Exact1000/Recovery90/GIANA=`1.0000/1.0000/10/235/335`；TcrDesign=`0.999643/1.0000/4/197/402`；GRATCR=`1.0000/0.074214/0/7/4`；BioSeq-7L-step117000=`0.998786/0.999714/0/0/1`。四者 GIANA 56/56 block 均 `status=ok`。
- **验证**：56 个 block 全部精确 1,000 行、rank 1–1000 连续；加入 α 条件来源、官方评价可用性、TCRT5 paper-rank 与 BioSeq step117000 metadata guard 后，协议回归 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/test_public_beta_benchmark.py` 为 21/21 passed；GIANA 小样本和全量均成功。
- **产物**：最终目录 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results`；权威协议 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/TCR_BETA_PUBLIC_TRACK_A.md`。

## 2026-07-21 TCRT5 paper-consistency audit and ranking correction

- **候选一致性**：本地官方权重解码与作者 `benchmark_data_w_preds.csv` 在 14 个 target 上逐集合完全一致，14,000/14,000 候选相交、每个 block Jaccard=1.0；不存在用作者预测行替换本地生成的问题。
- **排序根因与修复**：旧适配器直接保存 Hugging Face length-normalized `sequences_scores`，论文 rank-cutoff 使用累计生成 token log-likelihood。现改为 `compute_transition_scores(...).sum()` 后降序，`raw_score` 也保存累计值。13,990/14,000 rank 与作者 artifact 完全相同，剩余 10 条为 top-100 之外的相邻浮点近似并列；14/14 个 top-100 集合完全一致。Transformers 4.40.2 与 4.48.1 的 RVR 输出逐序列、顺序、分数相同，已排除依赖版本因素。
- **论文数字复核**：稀疏 13-pMHC 部分 exact hit=2，命中 target 为 `FTDALGIDEY_HLA-A*01:01`、`HPNGYKSLSTL_HLA-B*07:02`；单独 RVR 实验 exact hit=8，rank=`11,17,83,258,259,430,535,603`；GIANA `t=231,c=23`，均与论文/Figure 5 一致。当前 all-14 summary 的 exact=10 是 2+8，不是论文的单一 aggregate。
- **恢复率边界**：RVR 按论文同长度 identity/Hamming 定义 Recovery≥90%=109；按本 Track-A 统一 `1-Levenshtein/max(lengths)` 定义为 166。两者均保留但禁止混称同一指标。Figure 5 的 `c=23` 可复现为 reference-containing cluster 数；作者公开 GIANA 输出在这些 cluster 中实际覆盖 33 条不同 reference row，论文正文将 23 写成 unique references 存在歧义。
- **物化与运行边界**：排序修复后已完成官方 checkpoint 的 14-target 清洁重跑；canonical `generations.csv` 中 TCRT5 `raw_score` 是直接的累计 transition score，不是迁移近似。此前用经 runtime 验证的恒等式 `cumulative_score = old_score × (len(CDR3β)+2)` 完成过可逆协议检查；旧 length-normalized 表备份为同目录 `generations.pre_tcrt5_paper_rank.csv`，但不再是 canonical 结果。统一指标已在清洁重跑结果上重新物化；GIANA 输入只依赖候选集合，而清洁重跑与旧运行逐 target 集合完全一致，因此复用的 42 个已成功 GIANA block 不受排序修复影响。
- **审计产物**：逐 target `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/tcrt5_paper_consistency.csv`；人读报告为同目录 `tcrt5_paper_consistency.md`。

## 2026-07-21 TcrDesign generated-β → generated-α qualitative addendum completed locally

- **官方能力审计**：`tcrdesign_G.py -mode alpha`、`utils/generate_tcr.py::generate_alphaTCRs`、完整流程 `utils/generate_whole.py` 与 `weights/rnn/beta_alpha/RNN_tcr.ep` 均存在；论文/补充材料报告 α BLOSUM62、最小编辑距离和 TcrDesign-B 辅助 binding score，但仓库与 Zenodo 未发布可直接运行的 α-reference 作图/评价 CLI。`clip/` 只有 `CDR3a,CDR3b` 对比训练类，无 pairing checkpoint、检索指标或评价入口。因此结论固定为 `Alpha generation: available / Official alpha evaluation: unavailable / Included in quantitative benchmark: no`。
- **一致性 smoke**：1 条 generated β ×10 α 与官方单条 CLI 逐序列、逐顺序完全一致；2 条 generated β 的 batch 解码也分别与两次官方 CLI 完全一致。
- **正式生成**：使用 canonical `generations.csv` 中全部 14×1,000 条 TcrDesign `de_novo` generated β，每条生成 10 个 α beam；共 140,000 个有序 `(generated CDR3α, generated CDR3β)` pair。没有 oracle/reference β 条件路径，没有 V/J、TcrDesign-B 或自定义 α 指标；未提交 Volc 作业。
- **完整性**：14 targets 均 10,000 行；14,000 个 `(target,beta_rank)` 均 10 行且 `alpha_rank=1..10`；回连生成 β 表 0 missing / 0 sequence mismatch；`raw_alpha_score` 因官方 API 不返回 score 保持空值。
- **产物**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/tcrdesign_alpha_generations.csv`（26,345,623 bytes；SHA256 `7f334f1f2a9d4d750af1ebfd00b1af4f3b8c993c27c9fc88155ac78111e52131`）与同目录 `tcrdesign_alpha_evaluation_audit.json`。Track-A β 排名结果未改变。

## 2026-07-21 BioSeq 7L step117000 public CDR3β Track-A completed locally

- **Checkpoint choice**：用户要求测试 7L 的 `11xxxx` 优质 checkpoint；唯一匹配项为 retained Top-K step117000，也是当前 Top-K validation loss 最低者（`0.4430236165889635`）。快照 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step117000/best.pt`，SHA256=`e3d4c98ea84b4aa2280fe76ec5bf8a806de376699610e9a96960e66ceddb5c01`，manifest step/path/size 均校验。
- **Protocol**：新增 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/bioseq_generator.py` 与 `generate-bioseq` CLI；条件仅为 epitope，不消费 MHC pseudo-sequence；seed42、batch100、32 iterations、temperature1.0、`gumbel_argmax`。所有 target-chain residue 均被 mask/generate，placeholder 只决定长度；无固定或事后补写 C/F/W。模型不返回可比较 likelihood，故 `rank=seeded emission order`、`raw_score=empty`。
- **Smoke**：RVR 10 条成功，10/10 valid、10/10 unique、exact=0；原始输出保存在 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/smoke/bioseq_7l_step117000.csv`。正式 all-target run 独立从 seed42 开始，不混用 smoke 候选。
- **Full generation**：14 targets ×1,000=14,000 条全部生成并合入 canonical `generations.csv`；每 block rank 1–1000 连续、0 missing/empty。13,983/14,000 valid，17 条无效均为末位非 F/W；13,979 个 within-target unique valid。长度 9–23 aa，均值 14.552 aa。
- **Unified result**：ValidRate=`0.998786`、UniqueRate=`0.999714`、ExactHit@100/1000=`0/0`、ReferenceRecall=`0`、Recovery90Hit=`0`、GIANAHit=`1`、reference cluster/sequence coverage=`1/1`、mean cross-epitope Jaccard=`0.000044`（91 pair 中仅 8 pair 各共享 1 条，最大 `0.000501`）。唯一 GIANA hit 为 HPNGYKSLSTL 的 `CALSESGGGELDF`，与 reference `CASSESGAGELFF` 同 cluster。
- **RVR**：Valid/Unique=`1/1`，Exact1000/Recovery90/GIANA=`0/0/0`，median best recovery=`0.588235`；最佳 recovery=`0.785714`，包括 `CASSDGGAGELTF`↔`CASSRLGGAGELFF` 与 `CASSVGGQGGELTF`↔`CASSGGQGGKLFF`。
- **Interpretation**：step117000 已学到很强的 CDR3β 语法有效性和低跨表位复用，但当前 epitope-only sampling 对 target-specific reference neighborhood 的恢复明显弱于 TCRT5/TcrDesign；不能用高 Valid/Unique 单独得出生成质量更好的结论。
- **Artifacts**：聚焦报告 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/bioseq_step117000_report.md`；另有同目录 `bioseq_step117000_{generations,per_target_metrics,top_recoveries,giana_hits}.csv`。最终 `summary_metrics.csv`、`per_target_metrics.csv`、GIANA 56 blocks 与 `reproduction_status.md` 已统一刷新。

## 2026-07-21 Submitted early-best light-pairing prompt0/iter96 ablation

- **Scope**：仅测试此前 canonical pairing 最好的两个早期 checkpoint：step121000 与 step189000；不测试用户已排除的 389xxx checkpoint。沿用 holdout500、每条 heavy 生成 8 条 light、heavy batch size 4、light prompt tokens 0、`gumbel_argmax`、seed 42，只把上一轮 `max_iter=128` 改为 `96`，使用独立输出前缀，不覆盖既有结果。
- **Checkpoint identity**：step121000 SHA256=`51f0eee5fa7b127a0487a2ea8fdbc322bda6f6908ff2662087bd87d4980d4613`；step189000 SHA256=`b5388deccff040641e04c90b9e769cf89cd5160fe932148bd465de4ca8f957ca`；均与各自 manifest 一致。
- **Submit @16:28Z**：step121000=`t-20260722002822-p9s6k`，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_pairing_ablation_step121000_prompt0_iter96.yml`；step189000=`t-20260722002826-n29z8`，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_pairing_ablation_step189000_prompt0_iter96.yml`。两项均为 `Preemptible: false`、单卡 `ml.pni2.3xlarge`、初查 **Staging**。
- **Record**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_pairing_early_best_prompt0_iter96_20260721_task_ids.tsv`。
- **16:29Z status**：两个作业均已由 Staging 进入 **Running**，各自日志已创建并进入模型/评分依赖初始化阶段，暂未发现错误。
- **17:07Z progress**：step121000 与 step189000 的 iter96 均已完成 125/125 generation batches，正在 ImmunoMatch/ANARCI scoring；metrics 尚未落盘，任务仍为 **Running**。

## 2026-07-21 Expanded early-best light-pairing iteration grid

- **Design**：在 step121000 与 step189000 上新增 prompt0/iter32 和 prompt0/iter64；与既有 prompt0/iter96、prompt0/iter128 组成 `32/64/96/128` iteration 梯度。holdout500、num-seqs=8、heavy-batch-size=4、`gumbel_argmax`、seed42、checkpoint 与输出隔离策略均保持不变。prompt0/iter32 同时提供相对 canonical prompt3/iter32 的 prompt-only 对照。
- **Submit @17:04Z–17:05Z**：step121000 iter32=`t-20260722010426-k26hp`、iter64=`t-20260722010433-gmjj5`；step189000 iter32=`t-20260722010436-jsxkx`、iter64=`t-20260722010501-f26cb`。四项均为单卡 `ml.pni2.3xlarge`、`Preemptible: false`、Priority 6。
- **Status @17:05Z**：step121000 iter32/64 均 **Running**；step189000 iter32/64 均 **Queue**。原有 step121000/189000 iter96 两项保持 **Running**。
- **Progress @17:07Z**：step121000 iter32=`23/125`、iter64=`12/125` generation batches；两者正常运行且未发现错误。step189000 iter32/64 尚在 Queue。
- **Final @2026-07-22 02:11Z audit**：六个新增平台作业均为 **Success**，所有 generation CSV/metrics 均完整，日志有 done marker 且无 Traceback/CUDA/OOM。step121000 prompt0 iter32/64/96/128 ImmunoMatch=`0.610792/0.590018/0.584413/0.572159`，valid rate 均为 `1.0000`；step189000=`0.601284/0.587584/0.586982/0.574066`，valid rate=`0.9770/0.9780/0.9825/0.97825`。
- **Conclusion**：两个 checkpoint 在 prompt0 下均以 iter32 最优，更多 iteration 没有收益。全网格最优为 step121000 prompt0/iter32=`0.610792`，较该 checkpoint canonical prompt3/iter32=`0.608464` 高 `0.002329`（相对 `+0.38%`）；增益很小，单 seed42 下不作统计显著性声明。step189000 prompt0/iter32=`0.601284`，较 canonical `0.606259` 低 `0.004975`。
- **Duplicate-submit audit**：step189000 prompt0/iter64 首次 CLI 返回延迟但实际创建了 `t-20260722010440-jp5zk`，随后又创建记录任务 `t-20260722010501-f26cb`；两项均 Success、同 checkpoint/config/seed 并写同一前缀。最终 CSV 为 4000 rows、500 个 heavy 各恰好 8 条，metrics detailed_results=4000、JSON 完整，未发现并发写入污染；最终指标按完整落盘文件记为 `0.587584`。

## 2026-07-22 BioSeq step121000 MINT three-PPI formal evaluation

- **Scope lock @03:39Z**：按用户要求固定 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step121000/best.pt`（step `121000`，SHA256 `51f0eee5fa7b127a0487a2ea8fdbc322bda6f6908ff2662087bd87d4980d4613`），只评测 HumanPPI、YeastPPI、Gold-standard PPI（registry=`Bernett`）；不使用旧 `max_train=3000/max_length=512` 产物，不包含 MutationalPPI/SKEMPI。
- **Protocol repair**：三项均使用完整 MINT public-notebook 数据、joint two-chain grammar record + global residue mean、BioSeq 原生每链 1024 截断、640-hidden two-layer MLP、batch 16、100 epochs、best validation selection、3 repetitions。Human/Yeast public split与论文计数一致；Gold public notebook train=`163,192`，比论文 `163,019` 多 173，结果必须标 non-paper-exact。回归测试 `15/15 passed`。
- **Profile submit @03:39Z**：提交 matching-A100 最坏长度（1024+1024 residues）embedding batch profile，task ID=`t-20260722113922-4wl5x`，初始 **Initialized**，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_three_ppi_profile.yml`，`Preemptible: false`。候选 batch=`1/2/4/8`；`03:40Z` 已进入 **Running**，正式三项任务待 profile 实测后提交。
- **Profile-1 final @03:41Z**：`t-20260722113922-4wl5x` **Success** 并从 Active 表移除。A100 80GB 上 worst-case batch `1/2/4/8` 的 peak allocated=`2.332/3.022/4.403/7.163 GiB`，均成功；batch 8 远低于目标显存利用率，因此继续 profile `16/32/64/96`，不直接沿用过小 batch。
- **Profile-2 submit @03:46Z**：large-batch follow-up 已提交，task ID=`t-20260722114602-5xrg6`，初始 **Initialized**，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_three_ppi_profile_large.yml`，`Preemptible: false`；候选 batch=`16/32/64/96`。
- **Profile-2 final @03:48Z**：`t-20260722114602-5xrg6` **Success** 并从 Active 表移除。Worst-case batch `16/32/64/96` peak allocated=`12.685/23.729/45.816/67.903 GiB`（batch96 peak reserved=`70.387 GiB`，A100 total=`79.151 GiB`），全部成功。正式 embedding batch 固定为 **96**；这是每条 pair 两链均恰为 1024 residues 的上界压力测试。
- **HumanPPI submit @03:51Z**：正式 full-split 作业已提交，task ID=`t-20260722115108-8nwbl`，初始 **Initialized**，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_humanppi_official_full.yml`，`Preemptible: false`。
- **YeastPPI submit @03:51Z**：正式 full-split 作业已提交，task ID=`t-20260722115138-w2xgp`，初始 **Initialized**，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_yeastppi_official_full.yml`，`Preemptible: false`。
- **Gold-standard submit @03:52Z**：正式 full-split 作业已提交，task ID=`t-20260722115221-jmj42`，初始 **Initialized**，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_goldppi_official_full.yml`，`Preemptible: false`；该结果 provenance 固定标注 public notebook train 比论文多 173 行，`paper_comparable=false`。
- **Initial status @03:53Z**：HumanPPI、YeastPPI、Gold-standard 三项均已直接进入 **Running**，当前没有 Queue、抢占或失败。
- **YeastPPI final @04:02Z**：`t-20260722115138-w2xgp` **Success** 并从 Active 表移除；formal manifest/status、三份 embedding shape（`4945/95/394 × 960`）、4 条 metrics records（3 reps + summary）、checkpoint SHA/protocol provenance 均通过自动验收。Primary Accuracy=`0.602369±0.007846`；论文 `[P]` 最佳 MINT=`0.686971±0.010634`，Ours-best gap=`-0.084602`。
- **Gold val/test worker submit @04:05Z**：为避免 Gold parent 在 train 后再串行耗时抽取 val/test，提交只写 `val.pt` 与 `test.pt` 的 non-overlapping worker，task ID=`t-20260722120535-sdwdp`，初始 **Initialized**，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_goldppi_valtest_extract.yml`，`Preemptible: false`。parent 当前只写 `train.pt`，文件不重叠；worker 完成后 parent 会按 cache resume 逻辑跳过 val/test。新增 split-worker 路径后回归测试 `17/17 passed`。
- **Gold val/test worker status @04:06Z**：`t-20260722120535-sdwdp` 已进入 **Running**，没有 Queue 或挤占。
- **HumanPPI final @04:38Z**：`t-20260722115108-8nwbl` **Success** 并从 Active 表移除；formal manifest/status、embedding shapes（`26319/234/180 × 960`）、4 条 metrics records、checkpoint SHA 与 exact-public-split provenance 全部通过自动验收。三次 Accuracy=`0.700000/0.677778/0.655556`，summary=`0.677778±0.018144`；论文 `[P]` 最佳 MINT=`0.879630±0.006929`，Ours-best gap=`-0.201852`。
- **Gold train shard 0/4 submit @04:45Z**：提交 original rows `[0,40798)` 的连续 train shard，task ID=`t-20260722124532-pf9rl`，初始 **Initialized**，`Preemptible: false`，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_goldppi_train_shard0of4.yml`。
- **Gold train shard 1/4 submit @04:46Z**：提交 original rows `[40798,81596)`，task ID=`t-20260722124559-lqdg9`，初始 **Initialized**，`Preemptible: false`，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_goldppi_train_shard1of4.yml`。
- **Gold train shard 2/4 submit @04:47Z**：提交 original rows `[81596,122394)`，task ID=`t-20260722124657-p8vm5`，初始 **Initialized**，`Preemptible: false`，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_goldppi_train_shard2of4.yml`。
- **Gold train shard 3/4 submit @04:48Z**：提交 original rows `[122394,163192)`，task ID=`t-20260722124737-bqcjd`，初始 **Initialized**，`Preemptible: false`，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_goldppi_train_shard3of4.yml`。四个 shard 边界合计精确覆盖 163,192 行、无重叠；回归测试 `18/18 passed`。
- **Serial parent cancel @04:49Z**：shard0/1 已 Running 且首批正常，shard2/3 因当前并发槽为 Queue；按预案 cancel 尚未落盘 `train.pt` 的串行 parent `t-20260722115221-jmj42`，CLI 返回 `cancel success`，状态先记 **Killing**。此操作只释放资源，不保留或混用 parent 的内存中 partial embedding。
- **Serial parent terminal @04:49Z**：`t-20260722115221-jmj42` 已为 **Killed** 并从 Active 表移除；释放的资源使 shard2 进入 **Running**。当前 shard0/1/2 Running，shard3 Queue，val/test worker Running。
- **Gold test-only submit @04:53Z**：预先提交 test-only worker `t-20260722125316-879t6`，初始 **Initialized**，`Preemptible: false`，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_goldppi_test_extract.yml`。它当前等待 train shard3 之后的资源槽；一旦现有 val/test worker 写完 `val.pt`，将取消其尚未落盘的 test 阶段并由该 test-only worker完成，避免阻塞 shard3。
- **Gold test-only status @04:53Z**：`t-20260722125316-879t6` 已进入 **Queue**，顺序位于更早提交的 shard3 之后。
- **Gold full-test worker cancel @04:57Z**：为在 train shards 依次释放资源后并行四个连续 test shards，取消尚未启动、未产生任何文件的 full-test Queue 作业 `t-20260722125316-879t6`；CLI 返回 `cancel success`，状态先记 **Killing**。
- **Gold full-test worker terminal @04:57Z**：`t-20260722125316-879t6` 已为 **Killed** 并从 Active 表移除；没有启动容器或生成 `test.pt`。
- **Gold test shard 0/4 submit @05:00Z**：提交 original test rows `[0,13012)`，task ID=`t-20260722125938-2hvlz`，初始 **Initialized**，`Preemptible: false`。
- **Gold test shard 1/4 submit @05:02Z**：提交 original test rows `[13012,26024)`，task ID=`t-20260722130225-bvvqp`，初始 **Initialized**，`Preemptible: false`。
- **Gold test shard 2/4 submit @05:03Z**：提交 original test rows `[26024,39036)`，task ID=`t-20260722130251-84cd9`，初始 **Initialized**，`Preemptible: false`。
- **Gold test shard 3/4 submit @05:04Z**：提交 original test rows `[39036,52048)`，task ID=`t-20260722130345-fh242`，初始 **Initialized**，`Preemptible: false`。四个 test shard 连续覆盖 52,048 行、无重叠；YAML shell 语法与边界回归均通过。
- **Gold shard status @05:06Z**：test shard 0/1/2/3 均已进入 **Queue**；当前四个 GPU 槽由 val worker 与 train shard 0/1/2 占用，train shard 3 排在更早的 Queue。各 Running worker 日志持续前进，无失败或被挤占；这是预期的资源排队，不需要重提。
- **Gold validation verified / worker cancel @05:30Z**：`val.pt` 已完整落盘并通过硬校验，shape=`(59260, 960)`、dtype=`float32`。随后立即取消 `t-20260722120535-sdwdp`，CLI 返回 `cancel success`，状态先记 **Killing**；该 worker 的 test 阶段尚未生成 `test.pt`，后续只使用四个无重叠 test shards，避免重复写同一 cache。
- **Gold val worker terminal @05:31Z**：`t-20260722120535-sdwdp` 已为 **Killed** 并从 Active 表移除；释放的 GPU 槽使 train shard 3/4 `t-20260722124737-bqcjd` 进入 **Running**。当前四个 train shards 全部 Running，四个 test shards 继续 Queue。
- **Gold train shard 0 final @05:47Z**：`t-20260722124532-pf9rl` **Success** 并从 Active 表移除；`train.part-00-of-04.pt` shape=`(40798, 960)`、dtype=`float32`，sidecar manifest 同步存在。释放的槽位已让 test shard 0/4 `t-20260722125938-2hvlz` 进入 **Running**。
- **Gold train shard 1 final @05:48Z**：`t-20260722124559-lqdg9` **Success** 并从 Active 表移除；`train.part-01-of-04.pt` shape=`(40798, 960)`、dtype=`float32`。释放的槽位已让 test shard 1/4 `t-20260722130225-bvvqp` 进入 **Running**。
- **Gold train shard 2 final @05:50Z**：`t-20260722124657-p8vm5` **Success** 并从 Active 表移除；`train.part-02-of-04.pt` shape=`(40798, 960)`、dtype=`float32`。释放的槽位已让 test shard 2/4 `t-20260722130251-84cd9` 进入 **Running**。
- **Gold test shard 0 final @06:07Z**：`t-20260722125938-2hvlz` **Success** 并从 Active 表移除；`test.part-00-of-04.pt` shape=`(13012, 960)`、dtype=`float32`。释放的槽位已让 test shard 3/4 `t-20260722130345-fh242` 进入 **Running**；当前四个 MINT 活跃任务全部 Running、无 Queue。
- **Gold test shard 1 final @06:09Z**：`t-20260722130225-bvvqp` **Success** 并从 Active 表移除；`test.part-01-of-04.pt` shape=`(13012, 960)`、dtype=`float32`。剩余 test shards 2/3 与 train shard 3 均继续 Running。
- **Gold test shard 2 final @06:10Z**：`t-20260722130251-84cd9` **Success** 并从 Active 表移除；`test.part-02-of-04.pt` shape=`(13012, 960)`、dtype=`float32`。当前只剩 train shard 3 与 test shard 3 两个 MINT extraction 任务。
- **Gold test shard 3 final @06:27Z**：`t-20260722130345-fh242` **Success** 并从 Active 表移除；`test.part-03-of-04.pt` shape=`(13012, 960)`、dtype=`float32`。四个 test shards 已完整覆盖 52,048 行，当前只剩 train shard 3 extraction。
- **Gold test merge @06:28Z**：四个 test part 按 original contiguous row order 合并为 `test.pt`，硬校验 shape=`(52048, 960)`、dtype=`float32`，`test.merge.json` 同步确认 `row_order=original_contiguous_order`。首次命令仅因本地未导出 `PYTHONPATH` 在 import 阶段退出，补齐环境后成功，未修改任何 part 文件。
- **Gold train shard 3 final @06:32Z**：`t-20260722124737-bqcjd` **Success** 并从 Active 表移除；`train.part-03-of-04.pt` shape=`(40798, 960)`、dtype=`float32`。至此四个 train shards 与四个 test shards 全部成功，无剩余 MINT extraction 任务。
- **Gold train merge @06:33Z**：四个 train part 按 original contiguous row order 合并为 `train.pt`；train/val/test 三份 full embedding 统一硬校验为 `(163192/59260/52048, 960)`、`float32`，train/test merge manifest 均确认原始行序。
- **Gold head rep0 submit @06:33Z**：提交 repetition index/seed 0，task ID=`t-20260722143320-k6xls`，初始 **Initialized**，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_goldppi_head_rep0.yml`；协议为 960→640→1、100 epochs、batch 16、best validation metric，`Preemptible: false`。
- **Gold head rep1 submit @06:34Z**：提交 repetition index/seed 1，task ID=`t-20260722143347-g6hdz`，初始 **Initialized**，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_goldppi_head_rep1.yml`，`Preemptible: false`。
- **Gold head rep2 submit @06:34Z**：提交 repetition index/seed 2，task ID=`t-20260722143413-9zdct`，初始 **Initialized**，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_goldppi_head_rep2.yml`，`Preemptible: false`。三个 repetition 使用独立 metrics suffix，避免并发写同一结果文件。
- **Gold head status @06:35Z**：rep0/1/2 三个 task 均已直接进入 **Running**，没有 Queue 或资源挤占；三者只读同一组冻结 embedding，分别写 `rep0/rep1/rep2` metrics 文件。
- **Gold head rep0 final @06:57Z**：`t-20260722143320-k6xls` **Success** 并从 Active 表移除；单次 best-validation-selected test AUPRC=`0.5940368643`（Accuracy=`0.5575814633`、AUROC=`0.5835156595`）。结果 metadata 固定 checkpoint step/SHA、100 epochs、hidden 640、joint global pair mode，并正确标注 Gold `paper_comparable=false`（public notebook train +173）。
- **Gold head reps1/2 final @06:58Z**：`t-20260722143347-g6hdz` 与 `t-20260722143413-9zdct` 均为 **Success** 并从 Active 表移除。rep1 AUPRC=`0.5914499184`（Accuracy=`0.5550837688`、AUROC=`0.5822591187`）；rep2 AUPRC=`0.5907788355`（Accuracy=`0.5523170919`、AUROC=`0.5798333897`）。三个 repetition 文件现已齐全。
- **Gold formal merge/validation @06:59Z**：三个 seed 文件合并为 canonical 4 records（3 reps + summary），formal manifest **success**。Gold AUPRC=`0.5920885394±0.0014046507`；embedding shapes=`(163192/59260/52048, 960)`；checkpoint、input mode、hidden 640、100 epochs、三次 repetition 与 `paper_comparable=false` 全部通过硬校验。
- **Three-PPI final summary @07:06Z**：HumanPPI Accuracy=`0.6777777778±0.0181443685`、YeastPPI Accuracy=`0.6023688663±0.0078457009`、Gold-standard PPI AUPRC=`0.5920885394±0.0014046507`。对应最佳论文 `[P]` 均为 MINT，Ours−best gaps=`-0.2018518522/-0.0846023687/-0.0950679686`；Human/Yeast `paper_comparable=true`，Gold 因 notebook train +173 为 false。完整 8×`[P]` 后追加 BioSeq `[C]` 的表、差值、JSON 与报告已写入 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/mint_tasks/_ours_step121000_official3ppi/`。汇总器同时修复 baseline 表缺少可选 `paper_comparable` 列时的兼容处理；MINT 回归测试最终 `19 passed`。
- **Record**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_pairing_early_best_prompt0_iter32_64_20260721_task_ids.tsv`。

## 2026-07-22 BioSeq step189000 RVR epitope-only CDR3β diagnostic

- **Scope**：固定公开 Track-A 的重点 target `RVRAYTYSK_HLA-A*03:01`，使用 BioSeq 7L `step189000` checkpoint 做独立单-target诊断；输入模型的条件仅为 epitope `RVRAYTYSK`，MHC allele/pseudo-sequence 只保留为 target metadata。协议与 step117000 一致：seed `42`、batch `100`、`32` iterations、temperature `1.0`、`gumbel_argmax`、原始输出不补 C/F/W。
- **Checkpoint**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step189000/best.pt`；manifest step=`189000`、validation loss=`0.49125400149843346`、SHA256=`b5388deccff040641e04c90b9e769cf89cd5160fe932148bd465de4ca8f957ca`。
- **Smoke test**：10/10 valid、10/10 unique，ExactHit=`0`，median best recovery=`0.585714`；成功验证 checkpoint、动态 model label 与独立输出路径。
- **Formal 1,000**：998/1,000 valid (`0.998`)，998/998 unique (`1.000`)，ExactHit@100/1000=`0/0`，ReferenceRecall=`0`，Recovery>=90% unique hits=`0`，median best recovery=`0.600000`，maximum recovery=`0.800000`；GIANA generated hit/reference-cluster/reference-sequence coverage=`0/0/0`。两个 invalid raw outputs 均缺少末端 F/W。
- **Checkpoint comparison**：step117000 在相同 RVR 协议下 median recovery=`0.588235`、maximum=`0.785714`，step189000 分别小幅变为 `0.600000/0.800000`；两者仍均为 ExactHit=`0`、Recovery90Hit=`0`、GIANAHit=`0`。两个 checkpoint 的合法去重候选集合交集为 `0`，Jaccard=`0`。
- **Baseline context**：同一 RVR target 上，TCRT5/TcrDesign/GRATCR 的 Track-A Recovery90Hit 分别为 `166/108/2`，GIANAHit 分别为 `231/103/4`；因此 step189000 的语法有效性和候选多样性强，但没有缩小可检测的 reference-neighborhood specificity 差距。
- **Isolation**：本次结果只写入 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/checkpoint_comparison/bioseq_step189000_rvr_epitope_only`，没有追加或改写 canonical 56,000-row all-14 `generations.csv`。单 target 不定义 cross-epitope Jaccard；只报告 step117000-vs-step189000 的同-target候选集合 Jaccard。
- **Artifacts**：聚焦报告为 `report.md`，并保存 `generations.csv`、`per_target_metrics.csv`、`summary_metrics.csv`、`top_recoveries.csv`、`invalid_generations.csv`、`rvr_comparison.csv` 与完整 GIANA 输入/输出。BioSeq adapter 已改为从 manifest 读取 retained checkpoint step 并生成动态 label，默认 step117000 行为保持不变；相关 benchmark tests=`22/22 passed`。

## 2026-07-22 T1 official retrain-fold Ours frozen-head evaluation

- **Scope**：按用户要求使用当前最终 BioSeq-7L `step389500` checkpoint（SHA256 `5d1231992b65aa75a883e86662f34da136fa87e8ba031b804ff4c942ab1326c9`），冻结整个 ESMC→LLaDA backbone；每条输入为 role-explicit joint peptide–CDR3β binding grammar record，取最后 post-LLaDA residue hidden states 的一次 global mean pool，再训练 `960→256→128→1` MLP。每 fold 仅从对应 official train 内划 10% stratified validation，test 不参与 early stopping。
- **Data**：直接读取并校验 Figshare `retrain.zip`（20,005,661 bytes；MD5 `5cf77befd7a07e0cb359540f050fb7b4`）的五套 AS fold train/test，不使用旧本地 `downstream/benchmark/data/tcr_binding_nm2025/train.csv`。五 fold train 对各自三套 test 均为 `0` shared clonotype / `0` exact pair；unseen-independent 另为 `0` shared epitope。全部 split 的 unique `(Epitope,CDR3B)` 并集 `462,883`，pair-universe SHA256=`24f7de5f97346de0962ced427f7cbd8d0e293211b14219f00462f48dae22283c`。
- **Implementation**：新增 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_retrained_ours.py`，包含可恢复的 `float32` mmap frozen-feature cache、逐 fold head/provenance、precrec AUPRC、五折 sample-SD 汇总和两张来源分离排名表。主表 `ranking_local_release_AS.csv` 与相同 release 上的官方 checkpoint 本地 rerun 比；`ranking_paper_reference_AS.csv` 单独标 `mixed_source=true`，不替换 canonical original-only 表。
- **Validation**：retrained Ours + baseline protocol tests `21/21 passed`；TCR beta-only role/relation grammar tests `6/6 passed`；合计 `27/27 passed`，MLP CPU synthetic smoke 成功。YAML parse 与 Python compile 均通过。
- **Submit @04:35Z**：正式单卡、non-preemptible 作业已提交，task ID=`t-20260722123557-94sll`，YAML=`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_binding_retrained_ours_step389500.yml`；`04:35:58Z` 进入 **Running**。embedding batch 从 `256` 起，CUDA OOM 时自动减半；每 4,096 pairs flush `features.partial.npy` 与 resume state。
- **Feature/head completion @04:47Z**：`462,883 × 960 float32` frozen feature mmap 完整生成，实际 batch 始终为 `256`、平均约 `962 pairs/s`，无 OOM/NaN；partial/state 已原子清除。五个 head 的 early-stop epochs=`79/79/89/84/99`，每个 `head.pt` 都绑定 checkpoint SHA、对应 train member SHA、pair-universe SHA、train-only mean/std 和 `backbone_trainable_parameters=0`。
- **Five-fold result（mean±sample SD）**：seen-test AUROC/AUPRC=`0.653164±0.002554 / 0.669947±0.002600`；seen-independent=`0.546188±0.002596 / 0.555760±0.002036`；unseen-independent=`0.510345±0.006680 / 0.514325±0.009919`。unseen 仍处于近随机带，不能因相对 rank 较好而声称已获得强 unseen binding 泛化。
- **Primary same-release rank（Ours + 8 local official-checkpoint reruns）**：seen-test AUROC/AUPRC 均 `7/9`；seen-independent 均 `7/9`；unseen AUROC=`5/9`、AUPRC=`4/9`。unseen AUPRC 仅比 ATM-TCR 高 `0.000726`、比第三名 NetTCR 低 `0.001322`，差距很小。mixed-source paper-reference 排名序号恰好相同，但只作辅报。
- **Final validation/platform**：feature shape/dtype/finite spot checks、5 heads、15 prediction blocks、总 `197,484` prediction rows、逐文件 AUROC/precrec AUPRC 重算、两张各 `54` rows 的排名表全部通过；primary `mixed_source=false`、paper-reference `mixed_source=true`。另生成 baseline+Ours 各 9 行的合并宽表，并按用户展示要求再拆成 `comparison_with_ours_{seen_test,seen_independent}_local_release_AS.{csv,md}`：每张 baseline 按 AUPRC 降序、Ours 固定最后、无 unseen 列。ERGO-AE 在对应 official inference 的 batch-50 tail-drop 已写入 `n_note`。平台 task `t-20260722123557-94sll` 于 `04:47:25Z` **Success**，已从 Active 表移除。

## 2026-07-23 Complete TCRT5 generation evaluator reconstruction

- **Scope**：完成独立的 TCRT5 full-eval 路径，严格分开作者主评测 `20 pMHC × 100`、稀疏评测 `13 pMHC × 1,000`、独立 RVR `1 × 1,000`，并在当前四模型 Track-A `14 target × 1,000` 上重算。实现位于 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/tcrt5_full_eval.py`，CLI 位于 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_tcrt5_full_eval.py`。
- **Metric coverage**：已物化 Char-BLEU、paper-native recovery、Precision/Recall/F1@K、exact Hit@K/rank、论文描述的 AP=`mean(P@1…P@K)`、global unique、全 pair Jaccard、positional `H(gen)-H(ref)`、natural-log k-mer JSD (`k=2..12`)、长度分布、OLGA Pgen、≥90% identity、GIANA reference-cluster hit 和 915-sequence polyspecificity。
- **Native recovery subtlety fixed**：作者实现先按 Levenshtein 选最近同长 reference，再对选中的 reference 算逐位置 Hamming identity；只有不存在同长 reference 时才使用 `1-Levenshtein/len(reference)`。直接使用同长 `1-Levenshtein/length` 会对移位 motif 产生错误高分，已加入专门单测。
- **Author-release regression**：240 个 model×pMHC block 的 P/R/F1、edit distance、native recovery、原子 Char-BLEU 全部与作者 CSV 在 `1e-12` 内一致。另一个 12-model dataset companion 的 P/R/F1/edit 聚合一致，但 native recovery 与同发布包 240-block 聚合最大差 `0.001311`；因 companion 未发布其候选，记录为作者产物内部差异，不判 evaluator 失败。
- **Paper correspondence**：Supplementary Table 2 非 mAP 的 `68/72` 单元格在论文显示精度内一致。四个超出正常舍入范围的格为 TCRBART-0 recovery median，以及 TCRBART-0(B)、TCRT5-0(M)、TCRBART-FT Char-BLEU；重建值仍与作者发布 CSV 一致。论文 mAP 的 cumulative likelihood 未随主表 2,000-generation artifact 发布，故 12 个主表 mAP 只保留 ordered-list proxy，禁止冒充复现值；当前 Track-A TCRT5 有 cumulative transition scores，可按论文定义精确计算。
- **Figure checks**：长度 `14.583±1.211` 对论文 `14.6±1.2`，reference 长度 `14.531±2.022` 对 `14.5±2.0`；known binders=`181`、polyspecificity Pearson `r=-0.957638`、最强 entropy loss position=`6`（论文约 position 5）、k-mer JSD `k2=0.158250,k12=0.684829`、sparse13 exact=`2`、RVR exact=`8` 且 ranks=`11,17,83,258,259,430,535,603`、GIANA `t=231,c=23` 均对应。
- **OLGA boundary**：对作者公开候选运行完整 human_T_beta Pgen，non-zero=`1996/2000` 与论文精确一致。公开候选的 occurrence-weighted生成 `log10 Pgen=-6.876846±0.890584`，论文为 `-7.04±0.85`；target-deduplicated reference all-positive 为 `-10.854881±3.716057`，论文为 `-9.83±2.356`。按 Fig.4 可见窗口 `[-16,-5]` 后 reference 为 `-9.840730±2.369588`，仍不反向调 cutoff。OLGA 1.2.4 wheel 与本地 human_T_beta 四个模型文件 SHA256 相同，版本不能解释差异；分布矩标记为缺少最终 Figure 聚合/过滤快照。
- **Current rebuilt result**：BioSeq/TCRT5/TcrDesign/GRATCR 的 all-14 exact=`0/10/4/0`，paper-native ≥90%=`0/154/142/4`，GIANA hit=`1/335/402/4`，global unique=`13988/7788/13083/629`。Pgen-positive fraction=`0.9798/1.0000/0.9928/1.0000`，但 mean positive log10 Pgen=`-17.9253/-6.9504/-8.8010/-6.8664`；BioSeq 的非零率不能替代概率分布校准或 target-specific 功能邻域。
- **Validation and outputs**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/test_tcrt5_full_eval.py` 为 `8/8 passed`，Python compile 通过。权威人读报告为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/TCRT5_FULL_EVAL_REPORT.md`，总对应表为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/paper_correspondence.csv`。

## 2026-08-15 Submitted Ophiuchus-Ab official-ckpt eval on Volc

- **操作**：submit（停掉开发机本地 pairing 后改走集群）
- **task_id**：`t-20260815220631-lmjr9`
- **任务名**：`eval_ophiuchus_ab_official`
- **YAML**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ophiuchus_ab_official.yml`
- **初始状态**：`Queue`，队列 `q-20260121145036-6fztt`，单卡 `ml.pni2.3xlarge`
- **Preemptible**：true
- **口径**：官方 ckpt、不含 humanization；已完成的 SAbDab/SAb23H2 JSON 会 skip，实际跑 OAS holdout500 pairing（prompt3 argmax）
- **权重**：仓库内已有副本 `dllm_test/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt`（与 `/c20250601/mj/model_weights/ophiuchus_ab/...` 同源，不是新下载）

## 2026-08-16 Submitted three ESMC×LLaDA immune fusion jobs

- **操作**：submit（前三个 immune 任务；第四个 270m diffusion 未交）
- **数据**：`oas+ots+asd_antibody+trait+tcr_native`；**不用 ASD nanobody**；`--asd_antibody_benchmark_blocklist ""`（ASD 不过 benchmark filter）
- **ASD relation**：重标后写入 CSV 的只有 `binding`/`nonbinding`。train 850,134 行，binding 471,676（55.5%），nonbinding 378,458（44.5%）。`bool==0` 与 BUZZ `fuzzy` l/m → nonbinding；弱定量（alphaseq>3 等）已删除，不标成 binding
- **队列**：`queue012`（`q-20260524172355-rnqtf`），1×8 卡 `ml.pni2.28xlarge`，`Preemptible: false`
- **任务**：
  - `protein_esmc_llada8b_bert_immune` → `t-20260816051821-fj6p7`，YAML `train_jobs/protein_esmc_llada8b_bert_immune.yml`，初始 `Staging`
  - `protein_esmc_llada8b_diffusion_immune` → `t-20260816051844-zm4cl`，YAML `train_jobs/protein_esmc_llada8b_diffusion_immune.yml`，初始 `Staging`
  - `protein_esmc_llada270m_bert_immune` → `t-20260816051844-dnj4s`，YAML `train_jobs/protein_esmc_llada270m_bert_immune.yml`，初始 `Staging`

## 2026-08-16 Submitted fourth immune fusion job on preemptible

- **操作**：submit（第四个 immune 任务；前三个未 cancel / 未重提）
- **task_id**：`t-20260816052434-76qtw`
- **任务名**：`protein_esmc_llada270m_diffusion_immune`
- **YAML**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_esmc_llada270m_diffusion_immune.yml`
- **初始状态**：`Queue`，队列 `queue012`（`q-20260524172355-rnqtf`），1×8 卡 `ml.pni2.28xlarge`
- **Preemptible**：true（闲时；queue012 接受 preemptible，未切 spot-share-queue）
- **数据**：`oas+ots+asd_antibody+trait+tcr_native`；不用 nanobody；`--asd_antibody_benchmark_blocklist ""`

## 2026-08-16 Immune job monitor + stale Active cleanup

- **操作**：monitor（未 submit / 未 cancel / 未重提）
- **平台核对**：旧 ablation 均为终态，从 Active 表删除——`t-20260803071310-fvhz6` Failed、`t-20260803071313-9hmjw` Failed、`t-20260804004804-2s4rx` Failed、`t-20260804004807-tgw8s` Success
- **四个 immune 均为 Running**：`fj6p7` / `zm4cl` / `dnj4s` / `76qtw`；前三个已出有限 loss（无 NaN/OOM/traceback），第四个刚进 Running、仍在 tokenizer/加载
- **旧 ckpt**：中间 `checkpoint-*` 此前已裁完，本轮未再删文件（释放 0）

## 2026-08-16 Resubmitted 270m diffusion immune after preempt kill

- **操作**：resubmit（闲时抢占 / 平台 SIGTERM；无代码/数据 bug）
- **旧 task_id**：`t-20260816052434-76qtw`，终态 `Killed`（Elapsed 25276s ≈ 7h，End `2026-08-16T04:25:50Z`）
- **杀因**：worker-0 在 step ~11980、loss ~2.2–2.4 正常训练中收到 `Received 15 death signal` / `got signal: 15`（SIGTERM）；无 traceback/OOM/NaN。YAML `Preemptible: true`，判定为闲时抢占。
- **新 task_id**：`t-20260816123736-gwff2`
- **任务名**：`protein_esmc_llada270m_diffusion_immune`
- **YAML**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_esmc_llada270m_diffusion_immune.yml`（未改；未手写 resume）
- **初始状态**：`Queue`，队列 `queue012`（`q-20260524172355-rnqtf`），1×8 卡 `ml.pni2.28xlarge`
- **Preemptible**：true（闲时，按用户要求保持）
- **ckpt**：`output/protein_esmc_llada270m_diffusion_immune/checkpoint-11000` 为 latest full resume；YAML/脚本无自动 resume，未发明 `--resume_from_checkpoint`
- **未动**：`t-20260816051821-fj6p7` / `t-20260816051844-zm4cl` / `t-20260816051844-dnj4s`

## 2026-08-17 270m diffusion immune 改为非闲时并 resume

- **操作**：闲时 `t-20260816123736-gwff2` 已 Killed（抢占）；YAML 改为 `Preemptible: false`，从 `checkpoint-9000`（本轮带 optimizer 的最新完整 ckpt）续训后提交
- **新 task_id**：`t-20260817153216-qvtwx`
- **任务名**：`protein_esmc_llada270m_diffusion_immune`
- **YAML**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_esmc_llada270m_diffusion_immune.yml`
- **初始状态**：`Queue`，队列 `queue012`（`q-20260524172355-rnqtf`），1×8 卡 `ml.pni2.28xlarge`
- **Preemptible**：false（与另外三个非闲时一致）
- **未动**：`t-20260816051821-fj6p7` / `t-20260816051844-zm4cl`；`dnj4s` 已 Success，从 Active 表移除

## 2026-08-20 Submitted data-ready immune fusion evals (BERT vs diffusion)

- **操作**：submit 六条评测（四条表征 + 两条 diffusion 生成）；随后 cancel 同名重复提交（并行 agent 早约 5s）
- **口径**：只评本地已有数据。BERT=表征-only（T1 冻骨干+五折 MLP，T2/T3 表征）。Diffusion=表征+生成（SAbDab/SAb23H2 CDR、OAS holdout500 prompt3、T4 held20/unseen）。Humanization / GDPa1 / HD-Flu-CoV / m396 不跑。
- **锁定 ckpt**：8B BERT/diffusion `checkpoint-40000`；270m BERT `checkpoint-49000`；270m diffusion `checkpoint-42000`
- **队列**：`c20250601`（`q-20260121145036-6fztt`），单卡 `ml.pni2.3xlarge`，`Preemptible: false`
- **保留（本 executor 提交）**：
  - `eval-immune-270m-bert-repr` → `t-20260820053043-hwbv7`，YAML `eval_jobs/eval_immune_270m_bert_repr.yml`，初始 `Running`
  - `eval-immune-270m-diff-repr` → `t-20260820053046-pkljp`，YAML `eval_jobs/eval_immune_270m_diff_repr.yml`，初始 `Running`
  - `eval-immune-270m-diff-gen` → `t-20260820053049-pcpf8`，YAML `eval_jobs/eval_immune_270m_diff_gen.yml`，初始 `Running`
  - `eval-immune-8b-bert-repr` → `t-20260820053052-5svsz`，YAML `eval_jobs/eval_immune_8b_bert_repr.yml`，初始 `Running`
  - `eval-immune-8b-diff-repr` → `t-20260820053056-2k6dz`，YAML `eval_jobs/eval_immune_8b_diff_repr.yml`，初始 `Running`
  - `eval-immune-8b-diff-gen` → `t-20260820053059-ddtbc`，YAML `eval_jobs/eval_immune_8b_diff_gen.yml`，初始 `Running`
- **cancel 同名重复**：`t-20260820053038-zdffw`、`t-20260820053041-c9hhp`、`t-20260820053044-2t8mk`、`t-20260820053047-r77pn`、`t-20260820053050-rrs4s`、`t-20260820053053-44d9s`（均 `cancel success`）
- **进度**：`dllm_test/downstream/DATA_READY_EVAL_PROGRESS.md`

## 2026-08-20 Data-ready eval ID correction after duplicate race

- **操作**：cancel extras; **do not resubmit again** unless a keeper fails
- **原因**：supervisor + executor both submitted; first keepers (`hwbv7` etc.) were cancelled in the race. Second-wave **canonical live IDs**:
  - `eval-immune-270m-bert-repr` → `t-20260820053314-4gmfb`
  - `eval-immune-270m-diff-repr` → `t-20260820053318-blk66`
  - `eval-immune-270m-diff-gen` → `t-20260820053320-gszxv`
  - `eval-immune-8b-bert-repr` → `t-20260820053323-29dln`
  - `eval-immune-8b-diff-repr` → `t-20260820053327-m5c6n`
  - `eval-immune-8b-diff-gen` → `t-20260820053330-9sj65`
- **队列**：`c20250601`，`ml.pni2.3xlarge`，`Preemptible: false`
- **政策**：GPU eval/train 只走 `volc ml_task`；禁止在登录机跑 T1 retrain / embed / generation

## 2026-08-20 Resubmitted immune fusion evals after cancel race

- **操作**：第一波六条（`…hwbv7` 等）与并行 agent 互 cancel，随后 Failed/Killing。立即 `ml_task submit` 重提；保留每个 JobName 最早仍 Running 的一条，cancel 同名 extras
- **保留**：
  - `eval-immune-270m-bert-repr` → `t-20260820053314-4gmfb` Running
  - `eval-immune-270m-diff-repr` → `t-20260820053318-blk66` Running
  - `eval-immune-270m-diff-gen` → `t-20260820053320-gszxv` Running
  - `eval-immune-8b-bert-repr` → `t-20260820053323-29dln` Running
  - `eval-immune-8b-diff-repr` → `t-20260820053327-m5c6n` Running
  - `eval-immune-8b-diff-gen` → `t-20260820053330-9sj65` Running
- **cancel extras**：`t-20260820053325-5spmk`、`t-20260820053337-s2s4r`、`t-20260820053331-pzpnf`、`t-20260820053328-vqld7`、`t-20260820053335-rpnwl`、`t-20260820053339-ltb6g`
- **第一波终态**：`…hwbv7`/`…pkljp`/`…5svsz`/`…2k6dz` Killing→Killed；`…pcpf8`/`…ddtbc` Failed（cancel，非 harness bug）
- **Preemptible**：false；队列 `c20250601`；单卡 `ml.pni2.3xlarge`
- **请勿再 cancel 上述六条 keepers**（gen keepers 已于 05:35 因 harness 失败换成 `l62kt` / `p8p2r`，见下一节）

## 2026-08-25 top-k ckpt 全量 sweep（24 条）

**动机**：此前每条 run 只评了 `eval_loss` 最低的那一步，但 `save_top_k=3` 还留着另外几步。评一遍能回答「下游指标对 ckpt 选择有多敏感」——如果波动大于模型间差距，那么「8B 优于 270m」「diffusion 优于 BERT」这类结论就不稳。

**范围与成本权衡**（实测单 ckpt 耗时）：

| 阶段 | 耗时 | 是否铺 |
|---|---|---|
| T1（463k 对冻结特征抽取 + 五折 MLP） | **~113 分钟**（8B） | **不铺**，16 ckpt 要 30 小时 |
| T2 clustering | ~5 分钟 | 铺 |
| T3 few-shot | ~8 分钟 | 铺 |
| CDR（SAbDab 3 + SAb23H2 6，max_iter=2） | ~15 分钟（270m）/ ~30 分钟（8B） | 铺（仅 diffusion） |

**代码改动**：`run_immune_fusion_repr.sh` 新增 `REPR_STAGES`（默认 `t1,t2,t3`），日志按阶段命名。T1 太贵，sweep 只跑 `t2,t3`。

**评的 step**（各 run 的 top-k 备份，含 best；**排除** `checkpoint-final` 与 270m diffusion 的 6k/8k/9k resume 点）：

| Run | steps | best |
|---|---|---|
| 8B BERT | 43000 / 44000 / 46000 / 50000 | **43000** |
| 8B diffusion | 44000 / 45000 / 48000 / 50000 | **45000** |
| 270m BERT | 41000 / 44000 / 49000 / 50000 | **49000** |
| 270m diffusion | 39000 / 42000 / 47000 / 50000 | **42000** |

**提交**：24 条，全部成功（前缀 `eval-topk-`，YAML `eval_jobs/topk_*.yml`）
- **T2+T3 × 16**：四条 run 各 4 个 step（表征 BERT 与 diffusion 都测）
- **CDR × 8**：仅两条 diffusion run（生成只测 diffusion），`max_iter=2`

代表性 ID：`topk_8b_diff_45000_t23` → `t-20260825172608-429kg`；`topk_8b_diff_45000_cdr` → `t-20260825172604-986nr`；`topk_270m_bert_49000_t23` → `t-20260825172513-dpf9b`。完整 24 个 ID 见提交日志。

**读法**：best step 那几条应与已有 §0.2/§0.3/§0.5 数字一致（可作回归校验）；其余 step 用来给每个指标画出「ckpt 选择带来的波动带」。若波动带宽于模型间差距，相关结论必须降级为「无显著差异」。

## 2026-08-25 解码参数全线对齐官方（`max_iter` 是唯一有官方依据的设置）

**sweep 结论（官方 ckpt，`t-20260825131918-nxsqg` Success，22 分钟）**

SAb23H2 在 `max_iter=2` 上**几乎逐格复现论文 Table 1**，证明模型权重 / 解码路径 / CDR 定义 / AAR 算法全部正确：

| | L1 | L2 | L3 | H1 | H2 | H3 |
|---|---|---|---|---|---|---|
| max_iter=1 | 80.88 | 80.68 | 72.36 | 73.81 | 68.32 | 35.40 |
| **max_iter=2** | 80.92 | 80.21 | 72.81 | 74.52 | **68.59** | **36.70** |
| max_iter=4（旧默认） | 81.19 | 79.64 | 72.48 | 74.05 | 67.32 | 34.65 |
| max_iter=8 | 80.80 | 79.88 | 73.28 | 74.05 | 67.60 | 34.83 |
| 论文 `[P]` | 81.1 | 80.1 | 73.7 | 74.8 | **68.6** | **36.8** |

`max_iter=2` 时 H2 差 **0.01**、H3 差 **0.10**。之前 SAb23H2 那 2.3 pp 缺口**纯粹来自 argparse 默认值 4**。

SAbDab 单调随步数下降（AAR 是逐位指标，错误提交会向后传播）：

| max_iter | H1（论文 75.50） | H2（论文 70.18） | H3（论文 43.55） |
|---|---|---|---|
| 1 | 75.47 (−0.03) | 70.73 (+0.55) | **42.00 (−1.55)** |
| 2 | 75.44 (−0.06) | 70.71 (+0.53) | 41.43 (−2.12) |
| 4 | 75.54 (+0.04) | 70.63 (+0.45) | 40.96 (−2.59) |
| 8 | 75.62 (+0.12) | 70.64 (+0.46) | 40.70 (−2.85) |

H1 任意步数精确命中、H2 稳定高 0.5、H3 最好仍差 1.55 pp。**同 ckpt 同代码在 SAb23H2 上 H3 差 0.10、在 SAbDab 上差 1.55 → 剩余缺口不是解码，只能是样本口径**（3320 行 vs 3131 unique pdb_id vs 论文 3,127 complexes）。逐样本诊断 `eval-ophiuchus-cdrh3-persample` → `t-20260825135453-lh966`（dump 逐行 AAR + pdb_id + CDR 长度）。

**参数对照（`run/zero_shot_test.sh` 是论文真实设置，argparse 默认不是）**

| 任务 | 官方 | 我们（旧） | 现已改为 |
|---|---|---|---|
| CDR SAb23H2 | `max_iter 2` + argmax | 4（基线）/ **8**（我们模型） | **2** |
| CDR SAbDab | 脚本未列 | 4 / **8** | **2** |
| Light pairing | **`max_iter 124`** + gumbel + cfg **0/1/1.5** | 32 + gumbel + cfg 0 | **124**，cfg 扫 0/1/1.5 |
| probe 类（flab/dev/in_silico） | `--use_multimer --sep_chains` | 未接（无数据） | 将来接线时须对齐这两个开关 |

采样策略两边本来就一致（CDR=argmax 重建任务、pairing=gumbel_argmax 设计任务），`num_seqs 8` / `light_prompt_tokens 3` 一致，`cleaned_*` 列与原列在本地 CSV 完全相同（500/500）。

**代码改动**：`run_immune_fusion_gen.sh` / `run_immune_fusion_pairing.sh` / `ophiuchus_eval/run_eval.sh` 新增 `CDR_MAX_ITER`（默认 **2**）、`PAIR_MAX_ITER`（默认 **124**）、`PAIR_CFG_SCALE`（默认 0.0），**产物文件名带上 `iter{N}` / `cfg{X}`**，不同预算的结果互不覆盖、旧结果保留。T4 仍用 32（TCRT5 是另一套协议，无官方对应值）。

**注意：我们模型此前那批 CDR 数字（270m H3 41.91 / 8B H3 42.03）是在 `max_iter=8` 下测的，即对自己最不利的设置**；encoder 泄露修复的结论不受影响（那是 10+ pp 量级，`max_iter` 只值 1–2 pp）。

**已提交（对齐后重跑）**：
- `eval-immune-270m-diff-cdr` → `t-20260825150935-2kq2h`（max_iter=2）
- `eval-immune-8b-diff-cdr` → `t-20260825150938-h59fp`（max_iter=2）
- `eval-immune-270m-diff-pairing` → `t-20260825150942-whmkl`（max_iter=124）
- `eval-immune-8b-diff-pairing` → `t-20260825150946-sxd64`（max_iter=124）
- `eval-ophiuchus-pairing-official` → `t-20260825150949-7f74h`（max_iter=124，cfg 0/1/1.5 三档）

## 2026-08-25 Ophiuchus 复现缺口定位：argparse 默认值 ≠ 论文实际参数

- **现象**：官方 ckpt 复跑 SAbDab H1/H2 与论文吻合（75.58/70.61 vs 75.50/70.18），但 **H3 低 2.7 pp**（40.89 vs 43.55）；SAb23H2 H3 低 2.3 pp（34.50 vs 36.8）。
- **已排除**：
  - **长度截断**：heavy 最长 168 残基（170 token）超出 150 上限的只有 3 行 / 3320；CDR-H3 掩码上界被截断的仅 **2 行**（CDR 长 62/63 的牛源超长 loop），无「掩码完全落在 150 之外」的行。0.06% 的样本解释不了 2.7 pp。
  - **CDR 定义 / 位置**：`heavy[start:end+1]` 与 `cdrh3_seq` 3320/3320 完全匹配。
  - **`cleaned_*` 列**：pairing holdout CSV 里 `cleaned_h_sequence` 与 `h_sequence` 完全一致（500/500），非变量。
- **样本口径差异（已量化，非主因）**：release 的 fold 共 **3320 行**，但 **unique pdb_id = 3131**，论文 Table 2 声明 **3,127 complexes**。172 个 PDB 条目贡献 2–3 个不同 Fv 对（其中仅 2 个 H/CDR3 完全相同）。论文按复合物计数、我们按行计数。算术上：若论文那 3127 条为 43.55、额外 193 行为 0，合并均值 = 41.02，接近我们的 40.89——但要求额外行 AAR≈0 不合理，故仅记为口径差异，不作为解释。
- **🔑 根因方向：`run/zero_shot_test.sh` 显示论文没有用 argparse 默认值。**

  | 参数 | argparse 默认（我们用的） | 官方 run 脚本（论文实际） |
  |---|---|---|
  | CDR SAb23H2 `max_iter` | 4 | **2** |
  | pairing `max_iter` | 32 | **124** |
  | pairing `sampling_strategy` | argmax | **gumbel_argmax** |
  | pairing `cfg_scale` | 0.0 | **0.0 / 1.0 / 1.5 三档都跑** |
  | pairing 输入列 | `h_sequence` | `cleaned_h_sequence`（本地等价） |

- **理论支持**：AAR 是逐位准确率。迭代解码在已提交（可能错误）token 上继续条件化，误差会传播并**拉低** AAR；`max_iter=1` 是纯一次性 marginal argmax，最大化期望逐位准确率。所以「步数越少 AAR 越高」，方向与「论文 43.55 > 我们 40.89（max_iter=4）」一致，且官方 SAb23H2 那行正是 `max_iter 2`。
- **已提交 sweep**：`eval-ophiuchus-cdr-maxiter-sweep` → **`t-20260825131918-nxsqg`**，脚本 `scripts/downstream/sweep_ophiuchus_cdr_maxiter.sh`，扫 `max_iter ∈ {1,2,4,8}` × {cdrh3, cdrh1, cdrh2}（SAbDab 10 折）+ SAb23H2 六 CDR，产物 `output/downstream_generation/ophiuchus_ab/maxiter_sweep/`，末尾自动打印与论文的 delta 表。
- **pairing 待办**：我们的 Ophiuchus pairing 基线（gumbel，ImmunoMatch 0.641 vs 论文 0.695）用的是 `max_iter=32`，官方是 **124**，且论文扫过 `cfg_scale ∈ {0,1,1.5}`。这 0.054 的残差很可能同源，待 CDR sweep 结论出来后一并按官方参数复跑。

## 2026-08-24 【核心 bug】ESMC encoder 条件流在迭代解码中泄露参考序列

- **发现路径**：修完 pairing 长度泄露后，270m 重跑结果崩到 ImmunoMatch 0.0 / valid 0.0。诊断显示**长度恰好相同**的那 15.5% 行 identity 仍只有 0.150——这些行 decoder 输入与旧 reference 模式**完全一致**，却给出截然不同的结果，说明泄露不在 decoder 侧。
- **根因**：`dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py::_model_logits` 原来把 `corruption_mask = output_tokens.eq(mask) & generation_mask` 传给 `apply_decoder_corruption_to_encoder`。该函数只在**掩码置位处**写 `<mask>`，其余位置保留 `batch["encoder_input_ids"]`——而这个张量是 collator 从**干净 record** 建的。迭代解码每提交一个位置，`corruption_mask` 就缩小一格，于是 ESMC 条件流**重新暴露该位置的真实参考残基**，decoder 下一步直接抄。
- **实测（单条 light-pairing record，104 个待生成残基）**：提交 0% → 暴露 0/104；**提交 50% → 暴露 52/104**；**提交 90% → 暴露 93/104**。
- **为什么训练没这个问题**：训练是单次前向，`corruption_mask` 就等于全部 target 集合，encoder 恰好把所有 target 位置都掩掉。迭代推理才会出现「掩码随步数收缩」这一情形。
- **修复**：推理时改为镜像**整个 `generation_mask`**（全轨迹恒定），保证 ESMC 在任何一步都看不到 target 残基；cfg 分支同步改为 `generation_mask | partial_mask`。修复后实测全轨迹暴露 0/104。
- **回归测试**：`scripts/tests/bioseq/test_sampling_bioseq.py::test_encoder_never_sees_target_residues_during_decoding`，6 个测试全通过。
- **影响面（全部旧生成数字作废）**：所有走 `generate_bioseq` 的 ESMC-fusion 生成任务——**AB CDR infilling、light pairing、T4 TCR generation**。这解释了 CDR AAR 为何异常高（SAbDab H3 比论文高 11–14 pp）：不是（只是）OAS 语料重叠，而是解码过程被喂了答案。**RESULTS §0.4/§0.5/§0.6 的旧数字全部作废。**
- **重跑（修复后代码 + 8B 换新 ckpt）**：`eval-immune-270m-diff-gen` → `t-20260825010632-krz7q`；`eval-immune-8b-diff-gen` → `t-20260825010636-jvpsn`；`eval-immune-8b-bert-repr` → `t-20260825010639-qpshc`；`eval-immune-8b-diff-repr` → `t-20260825010642-sgm7d`。均 Running。
- **表征任务不受此 bug 影响**（单次前向 mean-pool，无迭代解码）；8B 两条 repr 重跑只是因为 ckpt-40000 被剪枝删除、必须换 43000/45000。270m 两条 repr 数字仍有效。
- **已取消**：`t-20260824235242-glhcr`（8B pairing，跑的是修复前代码，结果无意义）。

## 2026-08-24 Pairing 协议修复 + 重提（长度泄露 / argmax 塌缩）

- **诊断依据**：对照官方 `airgen/AirGen-Dev/downstream/comp_chain/generate_light_from_csv.py`。官方 `LightMaskingCollate` 用**固定 128-token light 缓冲区**（`chain_lengths={'fv_heavy':150,'fv_light':128}`），`light_tokens[i, 4:]` 全部置 mask，解码后 `split('<eos>')[0]` 截断——**长度由模型自己吐 EOS 决定，不泄露**。
- **我们的缺陷 (a) 长度泄露**：`downstream/grammar/light_chain_pairing.py` 用 `antibody_pair_record(heavy, light)` 把真实 light 装进 grammar record，`masks.py::light_chain_generation_partial_mask` 只把这些位置改 mask，故槽位数 = 参考长度。grammar-v2 的 `<prots>…<protd>` 块**没有链内终止符**，槽位数就是生成长度。实测 4000 对：长度 100% 一致、平均相同度 0.946、逐字复制 4.7%（8B）/9.7%（270m）。
- **修复 (a)**：新增 `--light-length-mode {prior,reference}`，默认 **prior**——从 OAS **train** split（300 万行）统计的 light 长度直方图采样，与本行参考无关。先验落盘 `data/downstream/comp_chain/oas_train_light_length_prior.json`（mean 108.86 / median 108 / 范围 91–141，与 holdout 边际 108.798 基本一致）。placeholder = `ref[:prompt] + 'A'*(L-prompt)`，填充位全被 mask，只有槽位数进模型。输出新增 `light_length_mode` / `target_light_length` / `ref_light_length` 三列可审计。`reference` 模式保留但打印警告。
- **基线缺陷 (b) argmax 塌缩**：`downstream/ophiuchus_eval/light_pairing.py` 的移植是忠实的（同样固定 128 槽位 + `<eos>` 截断，**不泄露长度**，实测长度一致率仅 40.6%），唯一问题是 `argmax` 确定性解码使 n=8 完全相同（diversity 4.9e-16、每 heavy 唯一数 1.00），所以 ImmunoMatch 0.352 是**单次采样重复 8 遍**。论文报 diversity 0.335，显然不是纯 argmax。
- **修复 (b)**：`run_eval.sh` 新增 `PAIRING_SAMPLING`（默认 `gumbel_argmax`），产物按策略命名，旧 argmax 结果保留作 provenance。`dplm_multichain.py` 支持 `vanilla/argmax/gumbel_argmax`，与我们同口径。
- **新增自动诊断**：`scripts/downstream/pairing_leakage_diagnostic.py`，每次 pairing 跑完输出 same_length_frac / mean_identity / exact_copy_frac / unique_per_heavy 与 verdict。已用旧产物回归验证：旧 Ours 判定 `length_leak_suspected=true`，旧 Ophiuchus 判定 `sampler_collapse_suspected=true`。
- **新增只跑 pairing 的 runner**：`scripts/downstream/run_immune_fusion_pairing.sh`（不重跑 CDR/T4）。
- **⚠️ 8B checkpoint-40000 已被删除**：8B 两条训练已跑完 50k，`TopKValLossCheckpointCallback` 按 top-3 剪枝把 40000 删了。现存最优：**8B diffusion `checkpoint-45000`（0.5348，优于旧 40000 的 0.5411）**、**8B BERT `checkpoint-43000`（0.2566，优于旧 40000 的 0.2624）**。270m 两条（BERT 49000 / diffusion 42000）仍在。**RESULTS §0 里 8B 那几行对应的权重已不存在，不可复现，需在新 ckpt 上重跑。**
- **提交（队列 `c20250601`，`ml.pni2.3xlarge`，非抢占）**：
  - `eval-immune-8b-diff-pairing-fix` → **`t-20260824235242-glhcr`** Running（ckpt-45000，tag `ours_fusion_8b_diff_45000`）。首次提交 `t-20260824234939-88m2t` **Failed**：`test -f ${CKPT}/model.safetensors` 因 40000 被剪枝而失败，已换 45000 重提。
  - `eval-immune-270m-diff-pairing-fix` → **`t-20260824234942-zp2vg`** Running（ckpt-42000）
  - `eval-ophiuchus-pairing-sampling` → **`t-20260824234946-r9bx2`** Running（`PAIRING_SAMPLING=gumbel_argmax`；CDR 阶段因 JSON 已存在而 skip）

## 2026-08-20 FusionConfig gen crash + cluster-only policy

- **政策（用户纠偏）**：训练与 GPU eval **立刻** `volc ml_task` 提交。禁止等本地卡，禁止在登录机跑 T1 retrain / embed / generation。登录机无本轮 eval Python；A100 上 `occupy.py`（Jul20 起）与本评测无关。
- **失败**：`t-20260820053320-gszxv` / `t-20260820053330-9sj65` Failed：`_FusionConfig` 缺 `forbidden_target_token_ids`（`generate_bioseq`）。
- **修复**：`examples/llada/protein_fusion_model.py` 给 `_FusionConfig` 补 `forbidden_target_token_ids=None` 与数值 `pad_token_id`。
- **只重提 gen**：`eval-immune-270m-diff-gen` → `t-20260820053506-l62kt`；`eval-immune-8b-diff-gen` → `t-20260820053511-p8p2r`（Staging）。四条 repr keepers 未动。
- **监督日志**：`dllm_test/downstream/DATA_READY_EVAL_SUPERVISOR.md`

## 2026-08-28 训练语料口径修正 + 补齐缺失 TCR 源（无任务提交）

本轮只动数据层与审计脚本，**未提交/取消任何 volc 任务，未改模型**，Active 表不变。
数据侧权威记录已同步到
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md`
「当前训练语料实测快照（2026-08-28）」。

- **`max_length` 默认值统一为 1024**（此前是量错语料的根因）。
  `examples/llada/protein_pretrain_esmc.py` 与 `examples/llada/protein_pretrain.py` 的
  `DataArguments` 把基类 `dllm/utils/configs.py` 的 1024 往下覆盖成 512，而四个
  `train_jobs/protein_esmc_*immune*.yml` 都显式传 1024。任何不显式传参的辅助脚本因此
  静默量到另一个语料。已删除覆盖。
- **`asd_antibody` 行数修正 159,331 → 276,412**。该源抗原中位数 607 aa，卡在 512 与
  1024 的总长预算之间（512 下抗原只剩 ~268 aa），是唯一对该参数敏感的源。
  `RETRAIN_PLAN.md` §7.2b/§7.8 初版数字作废，已在文内标注修正。
- **ASD 去污染代价修正并已决策**。此前记为「砍掉 81%」，实测真实去除率
  **67.49%**（长度过滤在 1024 下只剔 0.3%）。阈值为 `mmseqs easy-linclust
  --cluster-mode 1`（connected component）：CDR-H3 core 0.80 identity / 0.80 coverage，
  heavy 0.95/0.80，light 不参与。**用户 2026-08-28 确认接受该代价，三个 benchmark 全
  保护**，不放宽阈值。
- **审计脚本修正**：`scripts/count_immune_mix.py` / `scripts/count_immune_drops.py`
  改为一律使用 `DataArguments` 默认值，不再硬编码 512 与空 blocklist 路径；关闭
  blocklist 只能传 `none`（`""` 会被 `load_exclusion_keys` fail-fast 抛错，见
  RETRAIN_PLAN §7.3）。
- **七源实测（v3 mix，1024，全 blocklist 生效，共 7,626,737 条）**：`oas` 2,485,471 /
  `ots` 2,094,231 / `tcr_repertoire` 1,971,794 / `tcr_papers`(v2) 667,405 /
  `asd_antibody` 276,412 / `tcr_native` 96,552 / `trait` 34,872。总残基
  1,271,440,536。v2 mix（`tcr_papers` 指 v1，407,112）为 **7,366,444**。
- ~~**`tcr_papers` 的默认目录仍是 v1**~~，只有 v3 的 yml 用 `--tcr_papers_dir` 覆盖到 v2。
  不传该参数的统计量到的是 v2 mix。已给 `count_immune_mix.py` 加尾随 `field=value`
  覆盖参数，便于复现 v3。
  **→ 2026-08-29 已改默认为 v2**（见下节）；本条描述的坑已消除。
- **⚠️ `tcr_repertoire` 于 11:50 被并行会话重建**：`t4_refbinder` blocklist 从 61,136
  刷到 62,893 core（并集 123,135 → 124,176），train 1,971,136 → **1,971,794**。
  本轮所有表格已是重建后的口径；引用前先读 `build_report.json`。
- **语料新鲜度机制（并行会话同日修复，已记录）**：构建报告的 `PASS` 只对「构建时那份
  blocklist」有效，blocklist 重建后语料不会自动跟随，过期语料与干净语料报告同形。
  实测 `tcr_repertoire` 曾有 3 个 T4 参考 binder 残留在 train 中。修复为报告新增
  `blocklist_provenance`（path+mtime+sha1）、新增
  `scripts/data/tcr_native/assert_corpus_fresh.py`（校验 `PASS` 且比对活文件 hash，
  无 provenance 报 UNVERIFIED），v3 yml 启动前改调该脚本。
  **规则：blocklist 重建后依赖它的语料必须重跑并重校验。**
- **`tcr_papers` 决策：加，且已在跑 v2**。`TCR_PAPERS_DIR` 已指向
  `data/tcr_papers_v2/dataset`（train 668,331、唯一 epitope 3,218、净新 epitope
  +1,763，v1 为 +881），比 v1 四源多 TcrDesign-2026 三层；未新增 dataset token，
  `--dataset_args` 里仍写 `tcr_papers`。
- **`tcr_repertoire` 已接入 v3**（1,971,136 条无标签单链 CDR3β，唯一渲染为
  `tcr_single` 布局的源，此前该布局训练覆盖为 0）。
- **补齐两个盘上缺失的源**（脚本
  `scripts/data/download_missing_tcr_sources.sh`，走实验室代理 3128，逐文件核对官方
  MD5，可重跑）：VDJdb 2026-06-03（`vdjdb_full.txt` 139,745 → 192,754 行，+37.9%）；
  Zenodo 11208211 OTS/TCRLang（三个 tar.gz，MD5 全部与官方一致）。**两者均未接线训练。**
- **⚠️ TCRLang 官方 test/eval 不可用作本项目 benchmark**：按全长 β+α 精确配对比对
  `data/ots_paired_clean/final/train.csv`，test 命中 98.2%、eval 命中 98.3%。同源于 OTS
  但本项目自行重切 split，把对方测试集切进了训练集；反向也不能与该论文数字直接比较。
- **遗留未解**：`ImmuneSourceSpec.weight` 在 `ImmuneBioSeqDataset.__getitem__` 中被丢弃，
  混合比例纯由磁盘行数决定。`tcr_repertoire` 占 25.85% 记录但只占 2.04% 残基，
  `asd_antibody` 反向（3.62% 记录 / 11.37% 残基）。要真正控制配比需先让 `weight` 生效。

### 2026-08-28 23:46 状态核对 + Active 表清空

- 逐条查询确认 Active 训练表里残留的 8 条**全部 `Success`**（含六条本不该记在训练表里的
  `eval-immune-*`），评测表的两条 sab23h2 亦已 Success。两张表按
  `volc-train-task-log.mdc` 清空并刷新时间戳。
- `volc ml_task list --status Queue,Staging,Running,Killing` 按 `immune`/`esmc`/
  `ophiuchus` 过滤均返回「没有匹配条件的任务」：**本项目当前零非终态任务**。
- 四条 immune fusion 训练均已跑满 50k，`checkpoint-final` 齐全。现存 top-k：
  `270m_bert` 41000/44000/49000/50000；`270m_diffusion` 39000/42000/47000/50000
  （另存早期 6000/8000/9000）；`8b_bert` 43000/44000/46000/50000；
  `8b_diffusion` 44000/45000/48000/50000。
- `output/` 仍占 **1.1T**，但**当前不构成风险，无需清理**（2026-08-29 00:54 复核）：
  `/vepfs-mlp2/c20250601/251105016` 是 3.1P 共享 vepfs，**可用 841T / 已用 74%**，
  500MB 写入测试瞬时通过（3.9 GB/s），且未配置用户级 quota。2026-08-25 那次
  `Errno 122 Disk quota exceeded` 是**暂时性**的（共享盘或目录配额当时触顶），
  并非我们自己的用量所致——1.1T 相对 841T 可用量无关紧要。
  如果该错误再现，先看 `df -h` 与共享盘水位，不要先删自己的 checkpoint。
- 语料新鲜度复检通过：`assert_corpus_fresh.py` 对 `tcr_papers_v2` 与
  `tcr_repertoire` 均 OK。`tcr_papers_v2/finalize_report.json` 于 11:56 重写，
  但仅新增 `blocklist_provenance`，行数口径未变（`final_rows=696,920`、
  train 668,331）。
- **v3 未提交**：`train_jobs/protein_esmc_llada270m_diffusion_immune_v3.yml`
  已就绪（七源 from-scratch，启动前调 `assert_corpus_fresh.py`），等算力决策。

### 2026-08-29 01:45 下游泄漏审计：TRAIT 黑名单构造缺陷 + T4 参考集缺陷

新增 `scripts/data/dedup/audit_downstream_leakage.py`（5 个 TCR 源 × 11 个下游测试集
的残留泄漏矩阵，把每行推过该源**真实的** `row_to_record` 后再比对，即训练真正看到的数据）。
最终状态 **`HARD-REQUIREMENT hits = 0 -> PASS`**（`NM2025_seen` / `NM2025_unseen` /
`public_trackA` 三个零容忍基准全零）；`tcr_repertoire` 对全部 11 个测试集均为 0。

- **⚠️ 上一节的七源表已过期。** 我改了 trait 黑名单，另一并行会话在 17:16 UTC
  重建了 TRAIT 与 `tcr_papers_v2` 语料。新口径：

  | 源 | 旧 | 新 | 差 |
  |---|---:|---:|---:|
  | `tcr_papers`(v2) | 667,405 | **681,444** | +14,039 |
  | `trait` | 34,872 | **31,515** | −3,357 |
  | 合计 | 7,626,737 | **7,637,419** | +10,682 |

  其余五源不变（`oas` 2,485,471 / `ots` 2,094,231 / `tcr_repertoire` 1,971,794 /
  `asd_antibody` 276,412 / `tcr_native` 96,552）。
- **TRAIT 黑名单是 `语料 ∩ benchmark` 的交集写法，结构上挡不住语料重建**（已修）。
  `decontam_extra.py::decontaminate_trait` 原本先扫语料收集 core 再与 benchmark 求交集，
  所以黑名单只对构建时那份语料完备；新增行的 core 从没进过候选集。实测那 862 个键
  只覆盖 hard-requirement 保护集的 **4.2%**（277/6,570，分基准 0.8% / 6.4% / 13.2%）——
  **本源此前 0 命中是运气，不是过滤起了作用**。17:16 那次重建加了 600 行，
  4 行直接落在保护 core 上（`ASSVGGISPLH`、`ASSYGGPEQF` → NM2025_unseen；
  `ASSVGTGYEQY`、`ASSVGRNTEAF` → public_trackA），等 mtime 稳定 40s 后复核确认为真泄漏。
  改为 `簇级交集 ∪ binding_benchmark ∪ full_bank` = **59,212 键**（原 862），
  与 `build_repertoire.py` / `finalize_papers.py` 早已采用的写法一致。
  代价 trait −11.2%；附带 T2/T3 四列从 162/2,332/287/208 全部归零。
  原文件备份 `trait_benchmark_blocklist.txt.pre_union_bak`。
  **规则：精确黑名单必须 benchmark 派生，不能写成 `语料 ∩ benchmark`。**
- **§前节「语料新鲜度机制」补一半**：断言只能*发现*脱节，*挡住*后果的是加载期过滤。
  `tcr_repertoire` 原先没有加载期过滤（注释写"构建期已去污、无表位可作键"），
  故那 3 条落盘即等于进训练。已补 `repertoire_core_exclusions`
  （`t4` ∪ `t2t3` 投影成裸 core ∪ `ots_benchmark`），实测 3 条全部被挡下。
- **⚠️ T4 Setting-A 参考集自身有构造缺陷（既存问题，未修正打分）**：
  `prepare_tcr_generation.py --train-cap` 默认 200,000 而 OTS train 有 2,102,700 行，
  只覆盖 **9.5%**；该参数同时决定 novelty 参照集和 holdout 去重对象。后果是
  **19.4% 的 holdout 序列其实躺在训练数据里**（多出的 1,893 条逐条核验全部确认在
  train 内），且 novelty 参照只有全量的 10.5%（180,918 vs 1,718,935）。
  两个方向都虚高分数。已加 `--train-cap 0` / `--out-dir` 并生成修正版到
  `downstream/benchmark/data/tcr_generation_fullref/`（验证 holdout ∩ train 全量 = 0），
  **未覆盖线上文件、未重跑打分** —— 换参考会改动已有 novelty/JSD 数值，等决策。
  引用 T4 Setting A 数字前先确认口径。
- **审计脚本自身的结论行也修了**：原来把配对键投影出的裸 core 命中一并加总报
  `TOTAL residual hits = 75929 -> LEAKAGE`，而那些按构造就该非零
  （实测 `tcr_papers_v2` 约 7.7 万、`tcr_native` 约 7.4 万裸 core 命中 T4，
  但 `(core|epitope)` 配对层命中为 **0**；T4 黑名单 68,846 键 100% 含 `|`）。
  改为只用 hard-requirement 行驱动结论，其余标为 info 并注明为何非零合理。
- **`TCR_PAPERS_DEFAULT_DIR` 改为 v2**（`data/tcr_papers/dataset` →
  `data/tcr_papers_v2/dataset`）。原来的默认是个静默坑：所有不传
  `--tcr_papers_dir` 的东西（临时统计、`count_grammar_layouts.py`、泄漏审计）
  都在量 v1，而 v3 训的是 v2，两者差 274k 行。三个 job 配置都显式传目录，
  不受影响——`bert_immune` / `diffusion_immune` 钉 v1（保已跑 checkpoint 可复现），
  `diffusion_immune_v3` 钉 v2。v1 语料仍在盘上。
- **训练就绪性实测（2026-08-29 02:10）**：照 v3 yml 跑完整 pre-flight——
  9 个数据目录 + ESMC/LLaDA 权重齐、4 个 blocklist 非空、
  `assert_corpus_fresh.py` 通过；再用 v3 全参数做 `--dry_run`，
  七源各 64 行全部加载成功（`DRY RUN OK`），词表 remap 覆盖 43 id，
  抗体/TCR 两种布局渲染正常，日志确认 `trait_benchmark=59212` 新黑名单生效。
  **数据侧可以开训。**
- **遗留（不阻塞开训）**：`ImmuneSourceSpec.weight` 仍被丢弃——
  `ImmuneCsvDataset` 把 `spec.weight` 写进每条 record，但下游无人消费
  （`protein_fusion_model.py` 里的 `weights` 是 `_diffusion_token_weights`，
  按扩散时间的 per-token 权重，与 per-sample 源权重无关）。
  实测七源 weight 全为默认 1.0，**故这是缺失功能而非静默 bug**；
  但配比只能靠行数控制，也因此表位条件的 loss 预算提不上去。
- **⚠️ 追记（2026-08-29 13:30 复核）**：`tcr_repertoire` 于 **03:35 UTC**
  切分比例变了：train 1,971,794 → **2,128,750**（+156,956），
  valid 201,504 → **123,049**，holdout 201,170 → **122,669**。
  **成因更正（2026-08-29）：这不是"语料被重建"，而是本文件顶部那条记录的
  valid 近重复搬迁**（`move_near_dup_eval_rows.py --datasets tcr_repertoire --apply`，
  把 valid/holdout 中与 train Lev≤1 的行移入 train，5 轮迭代约 10 小时）。
  `build_repertoire.py` 未重跑，blocklist 也未变动——这也是
  `assert_corpus_fresh.py` 仍然通过的原因。
  复核结论：`assert_corpus_fresh.py` 通过；泄漏审计在新语料上
  **`HARD-REQUIREMENT hits = 0 -> PASS`**，本源对 11 个下游测试集全 0，
  新增的 15.7 万行未带进泄漏（与 `count_immune_mix.py` 测得的 kept=raw、剔 0 一致）。
  新七源合计 **7,794,375**（原 7,637,419）：
  `oas` 31.89% / `tcr_repertoire` 27.31% / `ots` 26.87% / `tcr_papers` 8.74% /
  `asd_antibody` 3.55% / `tcr_native` 1.24% / `trait` 0.40%。
  无标签合计 86.07%，有标签 TCR 识别 10.39%，抗体-抗原 3.55%。
  **本源近期反复在变，引用规模前先读 `build_report.json`。**
- **文档**：新增 `examples/llada/DATA_PIPELINE_README.md`（数据源 / 黑名单清单 /
  两个键空间 / 运行手册 / 三次去污事故复盘）；同步更新
  `examples/llada/PROTEIN_PRETRAIN_PROGRESS.md`、`scripts/data/README.md`、
  `downstream/trait/README.md`、`downstream/benchmark/README.md`、
  `data/trait/README.md`（原写着 "not yet wired"，早已接线）、
  `data/tcr_repertoire/README.md`、`data/tcr_papers/EXPANSION_AUDIT_2026_08_28.md`。

## 2026-09-11 提交 Ophiuchus-Ab 官方 ckpt 探针复现

- submit `eval_ophiuchus_ab_official_probe`
- YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ophiuchus_ab_official_probe.yml`
- `task_id=t-20260911054553-jb7v5`，初始 `Queue` → 查询时已 `Staging`
- 队列 `c20250601`，`Preemptible: true`，1×`ml.pni2.3xlarge`
- 权重用本地官方副本 `dllm_test/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt`（3.1G，与 `/c20250601/mj/model_weights/ophiuchus_ab/...` 同源）
- 任务：Table 5 CurrAb、Fig 4 m396、Table 4 GDPa1；不进 headline

## 2026-09-11 Ophiuchus-Ab 官方 ckpt 探针评测 Success

- `task_id=t-20260911054553-jb7v5`（`eval_ophiuchus_ab_official_probe`）终态 **Success**（Start `2026-09-10T21:45:53Z`，End `2026-09-10T22:03:58Z`，elapsed 1085s）
- YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ophiuchus_ab_official_probe.yml`
- 队列 `c20250601`，`Preemptible: true`，1×`ml.pni2.3xlarge`
- 产物：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/{specificity_hd_flu_cov,affinity_m396,developability_gdp_a1}_metrics.json`
- Table 4 GDPa1 变换空间 Spearman 对齐论文四列：AC-SINS 0.511 / HIC 0.550 / PR_CHO 0.460 / Titer 0.359
- Table 5 CurrAb 低于论文：Acc 0.6085 / F1 0.6068 / MCC 0.4139 vs `[P]` 0.6796 / 0.6790 / 0.5203
- Fig 4 m396 0.5% train Spearman mean 0.930（论文是趋势图，本轮未跑 MINT/AbMAP）
- **不进 headline**；Active 表已删该行

## 2026-09-11 提交 Table 5 ESM head 50/100 epoch 闲时复训

- submit `eval-ophiuchus-ab-esm-head-epochs`
- YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ophiuchus_ab_esm_head_epochs.yml`
- `task_id=t-20260911085733-5b727`，初始 `Queue`
- 队列 `c20250601`，`Preemptible: true`，1×`ml.pni2.3xlarge`
- 复用 `output/downstream_generation/ophiuchus_ab/specificity_embeddings.pt`；先 50 epoch 再 100 epoch
- 产物目录：`output/downstream_generation/ophiuchus_ab/specificity_esm_ep{50,100}/`
- 不覆盖 5-epoch `specificity_hd_flu_cov_metrics.json`；不进 headline

## 2026-09-11 Table 5 ESM head 50/100 epoch 闲时复训 Success

- `task_id=t-20260911085733-5b727`（`eval-ophiuchus-ab-esm-head-epochs`）终态 **Success**（Start `2026-09-11T00:57:34Z`，End `2026-09-11T01:06:17Z`，elapsed 523s）
- YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ophiuchus_ab_esm_head_epochs.yml`
- 队列 `c20250601`，`Preemptible: true`，1×`ml.pni2.3xlarge`
- 末轮五折均值：50 epoch Acc 0.6651 / F1 0.6640 / MCC 0.4982；100 epoch Acc 0.6778 / F1 0.6768 / MCC 0.5171（论文 `[P]` 0.6796 / 0.6790 / 0.5203）
- 100 epoch 曲线在 ~60 轮后 Acc 平台约 0.67–0.68，test CE 平台约 0.78；不进 headline
- Active 表已删该行

## 2026-09-11 提交 Table 5 ESM head 200/300 epoch 闲时加长

- submit `eval-ophiuchus-ab-esm-head-epochs-long`
- YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ophiuchus_ab_esm_head_epochs_long.yml`
- `task_id=t-20260911190334-kpfsz`，初始 `Queue`
- 队列 `c20250601`，`Preemptible: true`，1×`ml.pni2.3xlarge`
- 复用 `specificity_embeddings.pt`；独立训 200 再训 300，不覆盖 ep50/ep100
- 产物：`output/downstream_generation/ophiuchus_ab/specificity_esm_ep{200,300}/`；不进 headline

## 2026-09-12 提交 v5 8 卡 BERT 臂（闲时队列）

- submit `protein_esmc_llada270m_bert_immune_v5_8gpu_spot`
- YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_esmc_llada270m_bert_immune_v5_8gpu_spot.yml`
- `task_id=t-20260912035657-dd8v9`，初始 `Queue`
- 队列 **`c20250601`**（闲时，不是 `queue012`），`Preemptible: true`，`Priority: 6`，1×`ml.pni2.28xlarge`（8 卡）
- 数据 = v5 `data/prepared/immune_v5_receptor_completion`（train 7,626,885，七源计数与 plan §4.2 逐源一致）；
  `tcr_repertoire` 走 `data/tcr_repertoire_junc80/dataset`，`tcr_papers` 走 `data/tcr_papers_v2/dataset`
- 目标函数：`--train_objective bert --bert_all_chains True --bert_mask_ratio 0.15 --bert_mask_prob 0.8 --bert_random_prob 0.1`
- 与已提交的 diffusion 臂 `t-20260912021346-5xq4s` **只差目标函数**：
  from scratch 270m（d768/L8/h12）、可训 ESMC-300M、`residue_cond_mode add`、`max_length 1024`、
  global batch **256**、`max_steps 200000`、cosine `1e-4`、`warmup_steps 2000`、
  `save/eval_steps 1000`、`save_top_k 3`、`slim_checkpoints True` 全同
- global 256 走 `per_device 2 × ga 16 × 8`（diffusion 臂是 `4 × 8 × 8`）：优化器语义相同（§6.3），
  但 `per_device 2` 是本机型已实证配置，`per_device 4` 在更长的 v5 记录上尚无起跑实证；
  闲时任务 `PolicySets` 含 `Failed`，坏配置会被反复重提，故刻意走已实证档
- 闲时三件套齐（§6.6）：独立 `OUTPUT_DIR=output/protein_esmc_llada270m_bert_immune_v5_8gpu_spot`（提交前为空）；
  `pick_latest_full` 满包自动 resume（三件套 + `model.safetensors`，目录为空则不传 `--resume_from_checkpoint`）；
  `RetryOptions` `EnableRetry/MaxRetryTimes 50/IntervalSeconds 180/PolicySets [Failed, InstanceReclaimed]`
- 平台已接受确认：`ml_task get --format` 回显 `Preemptible: true`；`ml_task export -t ... --config`
  回显完整 `RetryOptions`（`export` 不回显 `Preemptible` / `ResourceQueueName`，是已知 CLI 缺陷）
- 看护循环：`scripts/monitor_spot_tasks.py` `TARGETS` 已加本任务，已重启（wrapper pid 3251489 /
  python pid 3251491），日志确认「追踪到任务 t-20260912035657-dd8v9（状态 Queue）」。
  🔴 **人工 cancel 前必须先 `touch output/_monitor/STOP.protein_esmc_llada270m_bert_immune_v5_8gpu_spot`**，否则会被重提回来
- 前置代码修复（必须在同一 checkout）：全链合格集排除 synthetic `X`，见
  `examples/llada/PROTEIN_PRETRAIN_PROGRESS.md` §6.10。v5 上旧口径 36.8% 的 BERT 目标是占位符
- 未决策：BERT 臂不训 relation token（`<binding>`/`<nonbinding>` 不在 `residue_mask` 里），
  与 diffusion 臂不对称，也不能用 `score_binding_relation.py` 评。见 PROGRESS §10

## 2026-09-12 v5 BERT 闲时臂：改为也训 relation target；核实「从未起跑」

- `t-20260912035657-dd8v9`（`protein_esmc_llada270m_bert_immune_v5_8gpu_spot`）
  **`LaunchTime` 为空、State 仍 `Queue`**，创建后 7.1 小时一步未跑，`OUTPUT_DIR` 下无任何 checkpoint。
  entrypoint 在 Launch 时才从 vepfs 读代码，故**改代码即生效，未 cancel、未重提**。
- 代码改动（用户决策「bert 需要学 relation」）：两个全链入口都并入 `relation_target_mask` ——
  `sample_bioseq_bert_noise(all_chain_targets=True)` 与 `all_residue_eligible_mask`。
  用的是 `relation_target_mask` **不是** `relation_token_mask`（后者含固定呈递 `<binding>`
  与 null 前缀 `<unknown>`，v5 实测大 2.15×）。顺带修镜像缺陷：`diffusion_all_chains=True`
  会用 `all_residue_eligible_mask` 覆写 eligible/loss mask，不并入 relation 就等于打开全链
  反而丢掉 generated-only 本来在训的 relation 监督。细节见 PROGRESS §6.10.1。
- 80/10/10 的 random 切片改为只对残基位写残基 id，非残基合格位改写 `<mask>`
  （对齐 diffusion sampler 既有约定），否则会把 `<binding>` 变成氨基酸。
- 验证：`scripts/tests/immune_llada/` **213 passed / 3 failed**，3 条全是 `test_full_parity.py`
  的既存失败（该文件 0 处引用 `protein_fusion_model`，且 `dllm/pipelines/immune_llada/` 对 HEAD 无改动）。
- 队列权限澄清（**推翻本日早先那条「已过期」的记法**）：权限按 `ml_task submit` 时**当前机器的账号**算。
  同一 vepfs 挂多台机器、各机凭证是不同账号。`zhuyiheng` 对 `c20250601` 确实无 `CreateCustomTask`
  （v5 diffusion `t-20260912021346-5xq4s` 的 Creator 就是它，投 `queue012`）；`251105016` 有
  （`t-20260911054553-jb7v5`、`t-20260912035657-dd8v9`）。见 PROGRESS §6.7。
- 平台 8 卡供给核实（2026-09-12 02:5x UTC，299 条非终态）：8 卡 **9 Queue / 6 Running**，
  而 1 卡 190 Running。我们用的两个队列里 8 卡**全部 Queue、零 Running**，4 卡却都在跑
  （`c20250601` 4 卡 2 Running；`queue012` 4 卡 3 Running）—— 再次印证 §6.7「缺的是连续整节点 8 卡」。
- 闲时历史起跑等待 13–19 小时，且有多条从未排上（`LaunchTime` 为空即从未起跑）。判据见 PROGRESS §6.7

## 2026-09-12 提交 v5 8 卡 diffusion 闲时臂（queue012 抢占，赛跑第二条线）

- submit `protein_esmc_llada270m_diffusion_immune_v5_8gpu_spot`
- YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_esmc_llada270m_diffusion_immune_v5_8gpu_spot.yml`
- `task_id=t-20260912111652-sqfcw`，初始 `Queue`，Creator `zhuyiheng`
- 队列 **`queue012`**（`q-20260524172355-rnqtf`）+ `Preemptible: true`，`Priority: 6`，1×`ml.pni2.28xlarge`
- 与正式非抢占臂 `t-20260912021346-5xq4s` **同配置 generated-only diffusion**（`--train_objective diffusion --diffusion_all_chains False --relation_aux none`），只在队列/抢占/OUTPUT_DIR/resume 校验/重试上分叉。独立 `OUTPUT_DIR=output/protein_esmc_llada270m_diffusion_immune_v5_8gpu_spot`（提交前目录不存在）
- 闲时三件套：`pick_latest_full` 验满包（`optimizer.bin` + `pytorch_model_fsdp.bin` + `scheduler.pt` + `model.safetensors`）；`RetryOptions` `Failed` + `InstanceReclaimed`，`MaxRetryTimes: 50`；`ActiveDeadlineSeconds: 7776000`（90 天）
- 平台交叉确认：`ml_task get --format` → `Preemptible: true`、队列 `q-20260524172355-rnqtf`、规格 `ml.pni2.28xlarge`；`export --config` → `RetryOptions` 与 `ActiveDeadlineSeconds: 7776000` 已被接受（`export` 不回显 `Preemptible`，已知 CLI 缺陷）
- **可复用结论（配额）**：CLI 没有队列配额接口。官方 OpenAPI `GetResourceQueue` / `ListResourceQueues`（`ml_platform`，`2024-07-01` / `2021-10-01`）能查。queue012 **有** SharedResource 共享池（当时占用 **78** 卡）；c20250601 共享占用 **24** 卡。验证方法：两边都精确满足「Running 抢占 GPU 数 == `SharedResource`/`SharedQuotaAllocated`」。抢占吃共享池，不吃专用配额账面剩余（queue012 剩 15 卡碎片、c20250601 剩 19，两个 8 卡抢占任务仍在 Queue）。因此本任务与正式非抢占臂不是零和。
- **当前供给**：queue012 共享池被 78 个 1 卡抢占占满，**0 个** 8 卡抢占在跑；唯一先例 `zyw-new_sampler-find_seal` 已排 17+ 小时。提交是占排队位，不是马上能起跑。
- 看护：`scripts/monitor_spot_tasks.py` `TARGETS` 含 BERT 闲时臂与本任务；wrapper pid 1819497 / python pid 1819499。状态监控独立实例 pid 1819646，日志 `_jobmon/watch_t-20260912111652-sqfcw.log`（不干扰正式臂 watcher pid 1675184）。
- 🔴 **人工 cancel 本任务前必须先 `touch output/_monitor/STOP.protein_esmc_llada270m_diffusion_immune_v5_8gpu_spot`**，否则看护会重提回来。BERT 闲时臂对应 `STOP.protein_esmc_llada270m_bert_immune_v5_8gpu_spot`。

## 2026-09-12 新提 v5 diffusion 闲时臂到 c20250601（不取消 queue012）

- submit `protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601`
- YAML：`train_jobs/protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601.yml`
- `task_id=t-20260912150055-44gvc`，初始 `Queue`，`LaunchTime` 空
- Creator `251105016`，队列 **`c20250601`**（`q-20260121145036-6fztt`），`Preemptible: true`
- 训练旗标与 queue012 正式臂 `t-20260912021346-5xq4s` 一致：v5、diffusion、generated-only、
  `per_device 4 × ga 8 × 8`、200k、cosine 1e-4、warmup 2000
- **独立** `OUTPUT_DIR=output/protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601`
  （不与正式臂、也不与 `zhuyiheng` 的 `t-20260912111652-sqfcw` / `..._8gpu_spot` 共用）
- **未取消** queue012 上两条：`t-20260912021346-5xq4s`（正式非抢占）与
  `t-20260912111652-sqfcw`（zhuyiheng 误投的同配置抢占副本）。本账号无 `StopCustomTask`
- 看护已接管；旧名 `..._8gpu_spot` 留 STOP 哨兵，避免误重提 sqfcw 那条

## 2026-09-12 cancel queue012 闲时 diffusion `t-20260912111652-sqfcw`

- 用户明确授权**只 cancel 这一条**。正式臂 `t-20260912021346-5xq4s` 与 BERT 闲时 `t-20260912035657-dd8v9` **未动**，cancel 后仍为 `Queue`。
- **顺序**：先 `touch output/_monitor/STOP.protein_esmc_llada270m_diffusion_immune_v5_8gpu_spot`（看护 `check_one` 见哨兵即跳过；未 touch 全局 `output/_monitor/STOP`，未重启看护 wrapper 1819497 / python 1819499），再 `./scripts/volc-no-proxy.sh ml_task cancel --id t-20260912111652-sqfcw`。
- 平台终态：`JobId=t-20260912111652-sqfcw`，`Status=Killed`，`End=2026-09-12T07:04:18Z`。未重提。YAML 文件保留、未改。
- `scripts/monitor_spot_tasks.py` `TARGETS` 已去掉 `..._diffusion_immune_v5_8gpu_spot`；保留 v3 spot 与 BERT 闲时。看护**不会**重提本任务（STOP 哨兵 + TARGETS 已删）。
- 状态监控：`touch _jobmon/STOP.watch_t-20260912111652-sqfcw`，pid 1831910 已退出（日志 `STOP file present; exiting`）。未碰 `_jobmon/STOP`。正式臂 watcher 1675184 与 BERT watcher 1831911 仍活。

## 2026-09-12 纠正：c20250601 上提非抢占 diffusion；Tags 改短

- 用户要的是 queue012 正式臂的**非闲时**副本投到 `c20250601`，不是抢占。
- 误提闲时版 `t-20260912150055-44gvc`（`Preemptible: true`）已 cancel（本账号创建，有权限）。
- 新提非抢占：`t-20260912150714-lwf28` / `protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601`
  - 队列 `c20250601`（`q-20260121145036-6fztt`），Creator `251105016`
  - **`Preemptible: false`**，`RetryOptions` 仅 `Failed`，`ActiveDeadlineSeconds 950400`
  - 训练旗标对齐 `t-20260912021346-5xq4s`；独立 `OUTPUT_DIR=..._c20250601`
  - 不进看护 `TARGETS`（非抢占）
- **Tags 纪律**：YAML Tags 最多 3 个短标签。本文件与
  `examples/llada/PROTEIN_PRETRAIN_PROGRESS.md` 写口径。
  已提交的 `lwf28` 平台侧 Tags 仍是长列表（改 YAML 不回写）；之后新提按短标签。

## 2026-09-13 Submitted reference-length pairing v3

- 操作：submit；用户要求“开始测试进行评价”。
- task_id：`t-20260913180327-vszq6`；任务名 `eval-immune-270m-42000-pairing-reference-v3`。
- **终态复核（19:01Z）**：平台 **Success**，End=`2026-09-13T12:59:20Z`，Start=`2026-09-13T10:03:28Z`，总墙钟 2:55:52（含 pilot 与 full）。全量生成约 2:21:33；完整 CSV / metrics 已落盘并通过核验，数值唯一记录在 [RESULTS §0.6 全量](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。已从两个 Active 表移除本任务；未修改或取消其他作业。本轮只读核验产物、复跑 CPU 回归并同步文档，没有新 GPU 作业。
- 12:23Z 状态复核：平台仍为 **Running**；全量 generation log 为 220/250 batch，即 440/500 条重链、3,520/4,000 个候选（88%）。按当前日志 ETA，生成约剩 17 分钟；之后才开始全量 ImmunoMatch / ANARCI 评分，全量 metrics 尚未落盘。本轮没有重提、取消或更改参数。
- YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_immune_270m_42000_pairing_reference_v3.yml`。
- 提交时刻：2026-09-13T10:03:27Z；首次查询 10:04Z 为 **Running**，平台 Start=10:03:28Z。
- 10:19Z 阶段更新：pilot32 的 CSV / metrics 已完整落盘并通过逐行复核；新数值见 [RESULTS §0.6 预检](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。同一作业 full500 已启动并写出 manifest、首批进度；仍为 Running，未完成全量评价。pilot 生成约 9 分钟；按 full 首批约 33 秒 / 2 条重链估计，完整生成约需 2.3 小时，另加评分，属估计而非完成承诺。
- 资源：`c20250601` / `q-20260121145036-6fztt`，1×`ml.pni2.3xlarge`，**Preemptible: false**，最长 12 小时。
- checkpoint：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune/checkpoint-42000`；沿用历史 270M 快照作本次评测，不改训练任务。
- 顺序：固定前 32 条重链预检评分，再完整 500 条；每条 n=8，prompt3，reference 长度，gumbel_argmax，iter124，cfg0，seed42，heavy batch=2。32 条只作 pilot，不代替全量结论。
- 产物根：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_pairing_reference_v3_20260913_270m42000/`；每阶段单独 CSV、manifest、generation/scoring log、metrics JSON。
- 环境：先激活 pllm 跑 CPU regression；真实生成与评分显式用既有 `protenix_abtcr` Python（ESMC/ANARCI/ImmunoMatch 依赖齐全）。未向共享环境安装包；已通过实际 checkpoint 的 CPU load + tokenizer/collator 检查。
- 本次只更新评测代码与必要的共享 mask-ID helper 提取（行为不变），未修改训练 recipe、提交训练或覆盖旧结果。具体修复和测试边界见任务文档，不把运行中当成已出分。

## 2026-09-13 Submitted TCR others CDR3ab v5

- 操作：submit；用户授权实现并测试 others 数据的 CDR3α 输入版本。
- task_id：`t-20260913180748-44599`；任务名 `eval-tcr-others-cdr3ab-v5-29000`；提交时刻约 2026-09-13T10:07:48Z，首次查询 10:08Z 为 **Running**，平台 Start=10:07:48Z。
- 终态：**Success**；平台 End=2026-09-13T10:10:40Z，Elapsed=172 秒；10:11Z 核验后从 Active 表移除。五折×三个测试集均完成；15 份预测逐行、字段、标签、hash、重算指标与汇总检查通过，详见审计 §9；未提交/取消其他任务。
- YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_cdr3ab_v5_29000.yml`。
- 资源：`c20250601`，1×`ml.pni2.3xlarge`，**Preemptible: false**，最长 4 小时；没有修改/取消任何预训练任务。
- 实验配置、checkpoint 快照、数据/代码身份和 CPU 验证见 [TCR 审计 §9：others 第一版](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)；不是全参数微调，也不是全部 others 字段 baseline 复现。
- 日志：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_tcr_others_cdr3ab_v5_29000.log`；结果根：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_v5_29000_others_cdr3ab_masked_runtime_v2/cdr3ab/AS/`。

## 2026-09-13 TCR others spot submit：CLI 配置校验失败（未创建任务）

- 操作：submit；用户要求使用闲时资源开始测试。任务名 `eval-tcr-others-cdr3ab-v5-29000-spot`；YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_cdr3ab_v5_29000_spot.yml`；配置 **Preemptible: true**。
- 10:27Z 首次尝试在本地 CLI 解析时退出：`cannot unmarshal !!seq into common.RetryOptions`。没有创建远端任务、没有 task_id 或平台初始状态；Active 表不增加虚假任务行，其他任务未动。
- 已将 RetryOptions 从旧模板的列表形式修为当前 CLI 所需的映射形式；仅修新 eval YAML，不修改原实验或历史训练 YAML。校验后再次 submit，成功任务另记下方。

## 2026-09-13 Submitted TCR others CDR3ab v5 spot

- 操作：submit（修复本地配置解析后再次提交，非重复运行中的任务）；task_id=`t-20260913182826-c4pw2`；任务名 `eval-tcr-others-cdr3ab-v5-29000-spot`。平台 CreateTime=2026-09-13 10:28:26，首次查询状态 **Queue**，`LaunchTime` 为空，尚未真正启动。平台 `Start` 在排队时也有值，不能以该字段宣称已经开始计算。
- 10:30Z 再次查询仍为 **Queue**、LaunchTime 为空、Preemptible=true；本次交接为已提交等待调度，非测试完成。
- YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_cdr3ab_v5_29000_spot.yml`。
- 资源：`c20250601` / `q-20260121145036-6fztt`，单卡 `ml.pni2.3xlarge`；`ml_task get --format` 已确认 **Preemptible: true**。YAML ActiveDeadlineSeconds=14400；仅 InstanceReclaimed 自动重试，最多 3 次、间隔 180 秒，不启用无限重提或额外看护进程。
- 协议/恢复边界和独立输出路径见 [TCR 审计 §9.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)；保留非闲时已完成实验，不修改/取消任何其他任务。此次新实验的分数不能沿用前一个 tag。

## 2026-09-13 Cancel TCR others spot：用户改用本地测试

- 操作：cancel；用户明确要求取消排队任务，不再提交任务，改用本地测试。
- task_id=`t-20260913182826-c4pw2`；任务名 `eval-tcr-others-cdr3ab-v5-29000-spot`；YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_cdr3ab_v5_29000_spot.yml`；队列 c20250601，**Preemptible: true**。
- 取消前状态 Queue、LaunchTime 为空；11:16Z CLI 返回 `cancel success`，随后确认终态 **Killed**、End=2026-09-13T11:16:17Z、LaunchTime 仍为空，已从 Active 表移除。没有启动过模型计算；不取消其他任务、不自动重提本任务、不删除已有快照/代码/结果。
- 本地只读资源检查发现 A100-SXM4-80GB 上已有两个计算进程，15,220 MiB 显存占用、GPU 利用率 88%；没有终止这些进程或立即叠加新的本地模型评测。具体执行边界登记于 [TCR 审计 §9.4](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。

## 2026-09-13 Submitted TCR others nonspot rerun

- 操作：submit；用户最新明确授权“如果本地不行，可以提交一个非闲时资源的卡”。本地 GPU 仍有占卡脚本和 FABind 训练，不终止或叠加这些进程；提交前查询无同范围在途 TCR others 任务。
- task_id=`t-20260913192406-hxhfb`；任务名 `eval-tcr-others-cdr3ab-v5-29000-nonspot`；平台 CreateTime=2026-09-13 11:24:07，LaunchTime=11:24:33，首次查询为 **Running**。
- 终态 **Success**，End=2026-09-13T11:27:24Z，平台 Elapsed=197 秒（包含提交到启动的等待）；11:28Z 核验后从 Active 表移除。五折×三测试集全部完成，产物复核见审计 §9.5；不是读取旧缓存后直接退出。
- YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_cdr3ab_v5_29000_nonspot.yml`。
- 资源：c20250601 / q-20260121145036-6fztt，1×ml.pni2.3xlarge，平台 get 已确认 **Preemptible: false**，YAML ActiveDeadlineSeconds=14400；没有重新启用已取消的闲时任务，没有提交预训练或修改其他作业。
- 独立重评的协议、tag/缓存/日志及验证状态见 [TCR 审计 §9.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。本次不是 full finetuning，也不是更换模型后的新实验；不覆盖此前已完成的同协议五折结果。

## 2026-09-13 Submitted TCR v5 beta-only control on others

- 操作：submit；用户要求测试 v5 checkpoint，承接同 others 数据的 β-only 对照。沿用此前允许的单卡非闲时方式；本地 GPU 利用率 95%，未动已有进程；提交前无同范围在途 TCR 任务。
- task_id=`t-20260913202402-c9hf5`；任务名 `eval-tcr-others-cdr3b-v5-29000-nonspot`；Start=2026-09-13T12:24:02Z，首次查询 **Staging**。
- 12:25Z 确认 **Running**，新日志显示加载同一 29k 权重并开始抽取 18,868 个 peptide/beta 输入。
- 终态 **Success**，LaunchTime=2026-09-13T12:24:32Z，End=2026-09-13T12:27:40Z；平台 Elapsed=218 秒，Launch→End=188 秒；平台确认 Preemptible=false。已从 Active 表移除，15 份预测与原始官方行集及重算指标核验通过，见审计 §9.6；指标唯一归档于 RESULTS §0.1b。
- YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_cdr3b_v5_29000_nonspot.yml`；c20250601 / q-20260121145036-6fztt，1×ml.pni2.3xlarge；配置 **Preemptible: false**，ActiveDeadlineSeconds=14400。
- 数据/代码/模型身份、43 项 CPU 回归及独立产物目录见 [TCR 审计 §9.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。提交时尚无新指标；不是大 β-only 数据集重跑、全参数微调或官方 baseline 重训。

## 2026-09-13 Submitted TCR v5 LongAB control on others

- 操作：submit；用户批准 LongA/LongB 版本及相同 v5 29k、相同 others/AS 五折测试。提交前无在途 TCR 作业，本地 A100 利用率 87%，未干扰现有进程。
- task_id=`t-20260914005618-vr2nb`；TaskName=`eval-tcr-others-longab-v5-29000-nonspot`；首次查询 **Running**，CreateTime=2026-09-13T16:56:19Z，LaunchTime=16:56:37Z，平台确认 Preemptible=false。任务 ID 使用平台命名日期，本记录使用 UTC。
- 终态 **Success**，End=2026-09-13T16:59:36Z，提交至结束 197 秒，Launch→End=179 秒；17:00Z 从 Active 表移除。全部 15 份预测、原始行/标签及 hash、指标重算、五折汇总、训练内标准化核验通过。新分数仅归档于 RESULTS §0.1c；未重新提交或取消其他作业。
- YAML：[eval_tcr_others_longab_v5_29000_nonspot.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_longab_v5_29000_nonspot.yml)；c20250601，1×ml.pni2.3xlarge，配置 **Preemptible=false**，ActiveDeadlineSeconds=14400。
- 协议、独立输出/cache/tag 与 55 项 CPU 回归见 [TCR 审计 §9.7](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。提交时没有 LongAB 新指标；旧 β-only/CDR3αβ 结果不覆盖。

## 2026-09-13 Submitted Ours AB v5 49000 preflight

- 操作：submit；task_id=`t-20260914045427-6lsdw`；TaskName=`ab-v5-49000-llada-20260913-preflight`；首次查询 Staging，CreateTime=2026-09-13T20:54:28Z。
- YAML：[eval_ab_v5_49000_llada_20260913_preflight.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_llada_20260913_preflight.yml)；c20250601 / ml.pni2.3xlarge / 1 replica / 1 GPU；**Preemptible:false，非闲时、非抢占**。仅 GPU 执行门禁，不产出正式成绩。
- 用户指定用最新 v5；20:50Z 核对两条 diffusion 保存点，c20250601 最新为 49000，queue012 为 40000；选前者。trainer_state.global_step=49000，最新 eval_loss=0.760879465386504；不是按最优 loss 选模。
- 源：[checkpoint-49000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601/checkpoint-49000)；独立评测快照：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913)。复制前后权重 SHA256 一致：`d5f892bce7d18106a7c417a3e4605f2bf1feac2bfa80709ca74fb149e5c07d92`；不复制优化器，不改变训练任务。
- 11项正式作业配置已生成，见 [job_plan.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913/job_plan.json)；提交以 GPU gate 通过为前提。30项 CPU 回归通过。
- 终态 **Success**，End=2026-09-13T20:56:02Z，20:56Z 从两张 Active 表移除；A100-80GB 真实模型加载、3任务16样本表征、6种prompt/CFG、9种CDR模式全部通过，服务器端30项测试通过。原始门禁报告：[passed.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913/preflight/passed.json)。预检只验证执行不变量，不作为正式质量结果。

## 2026-09-13 Submitted Ours AB v5 49000 pair-p0-cfg0

- 操作：submit；task_id=`t-20260914045749-mv7sv`；TaskName=`ab-v5-49000-llada-20260913-pair-p0-cfg0`；首次查询 Queue，查询时间 2026-09-13T20:57:52Z。
- YAML：[eval_ab_v5_49000_llada_20260913_pair-p0-cfg0.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_llada_20260913_pair-p0-cfg0.yml)；c20250601 / ml.pni2.3xlarge / 1 replica / 1 GPU；**Preemptible:false，非闲时、非抢占**。
- 统一独立评测快照：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913)；实验计划与产物根：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913)。本条是提交记录，不代表评测完成。

## 2026-09-13 Submitted Ours AB v5 49000 pair-p0-cfg1

- 操作：submit；task_id=`t-20260914045802-jxn8p`；TaskName=`ab-v5-49000-llada-20260913-pair-p0-cfg1`；首次查询 Queue，查询时间 2026-09-13T20:58:05Z。
- YAML：[eval_ab_v5_49000_llada_20260913_pair-p0-cfg1.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_llada_20260913_pair-p0-cfg1.yml)；c20250601 / ml.pni2.3xlarge / 1 replica / 1 GPU；**Preemptible:false，非闲时、非抢占**。
- 统一独立评测快照：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913)；实验计划与产物根：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913)。本条是提交记录，不代表评测完成。

## 2026-09-13 Submitted Ours AB v5 49000 pair-p0-cfg1p5

- 操作：submit；task_id=`t-20260914045806-sgrlb`；TaskName=`ab-v5-49000-llada-20260913-pair-p0-cfg1p5`；首次查询 Waiting，查询时间 2026-09-13T20:58:08Z。
- YAML：[eval_ab_v5_49000_llada_20260913_pair-p0-cfg1p5.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_llada_20260913_pair-p0-cfg1p5.yml)；c20250601 / ml.pni2.3xlarge / 1 replica / 1 GPU；**Preemptible:false，非闲时、非抢占**。
- 统一独立评测快照：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913)；实验计划与产物根：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913)。本条是提交记录，不代表评测完成。

## 2026-09-13 Submitted Ours AB v5 49000 pair-p3-cfg0

- 操作：submit；task_id=`t-20260914045810-ss6ml`；TaskName=`ab-v5-49000-llada-20260913-pair-p3-cfg0`；首次查询 Waiting，查询时间 2026-09-13T20:58:13Z。
- YAML：[eval_ab_v5_49000_llada_20260913_pair-p3-cfg0.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_llada_20260913_pair-p3-cfg0.yml)；c20250601 / ml.pni2.3xlarge / 1 replica / 1 GPU；**Preemptible:false，非闲时、非抢占**。
- 统一独立评测快照：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913)；实验计划与产物根：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913)。本条是提交记录，不代表评测完成。

## 2026-09-13 Submitted Ours AB v5 49000 pair-p3-cfg1

- 操作：submit；task_id=`t-20260914045814-csgm4`；TaskName=`ab-v5-49000-llada-20260913-pair-p3-cfg1`；首次查询 Waiting，查询时间 2026-09-13T20:58:16Z。
- YAML：[eval_ab_v5_49000_llada_20260913_pair-p3-cfg1.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_llada_20260913_pair-p3-cfg1.yml)；c20250601 / ml.pni2.3xlarge / 1 replica / 1 GPU；**Preemptible:false，非闲时、非抢占**。
- 统一独立评测快照：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913)；实验计划与产物根：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913)。本条是提交记录，不代表评测完成。

## 2026-09-13 Submitted Ours AB v5 49000 pair-p3-cfg1p5

- 操作：submit；task_id=`t-20260914045817-trh7j`；TaskName=`ab-v5-49000-llada-20260913-pair-p3-cfg1p5`；首次查询 Waiting，查询时间 2026-09-13T20:58:20Z。
- YAML：[eval_ab_v5_49000_llada_20260913_pair-p3-cfg1p5.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_llada_20260913_pair-p3-cfg1p5.yml)；c20250601 / ml.pni2.3xlarge / 1 replica / 1 GPU；**Preemptible:false，非闲时、非抢占**。
- 统一独立评测快照：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913)；实验计划与产物根：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913)。本条是提交记录，不代表评测完成。

## 2026-09-13 Submitted Ours AB v5 49000 specificity

- 操作：submit；task_id=`t-20260914045834-cfk6n`；TaskName=`ab-v5-49000-llada-20260913-specificity`；首次查询 Waiting，查询时间 2026-09-13T20:58:36Z。
- YAML：[eval_ab_v5_49000_llada_20260913_specificity.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_llada_20260913_specificity.yml)；c20250601 / ml.pni2.3xlarge / 1 replica / 1 GPU；**Preemptible:false，非闲时、非抢占**。
- 统一独立评测快照：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913)；实验计划与产物根：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913)。本条是提交记录，不代表评测完成。

## 2026-09-13 Submitted Ours AB v5 49000 gdp_a1

- 操作：submit；task_id=`t-20260914045837-2lp8v`；TaskName=`ab-v5-49000-llada-20260913-gdp-a1`；首次查询 Waiting，查询时间 2026-09-13T20:58:40Z。
- YAML：[eval_ab_v5_49000_llada_20260913_gdp_a1.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_llada_20260913_gdp_a1.yml)；c20250601 / ml.pni2.3xlarge / 1 replica / 1 GPU；**Preemptible:false，非闲时、非抢占**。
- 统一独立评测快照：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913)；实验计划与产物根：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913)。本条是提交记录，不代表评测完成。

## 2026-09-13 Submitted Ours AB v5 49000 m396

- 操作：submit；task_id=`t-20260914045841-v242p`；TaskName=`ab-v5-49000-llada-20260913-m396`；首次查询 Waiting，查询时间 2026-09-13T20:58:44Z。
- YAML：[eval_ab_v5_49000_llada_20260913_m396.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_llada_20260913_m396.yml)；c20250601 / ml.pni2.3xlarge / 1 replica / 1 GPU；**Preemptible:false，非闲时、非抢占**。
- 统一独立评测快照：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913)；实验计划与产物根：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913)。本条是提交记录，不代表评测完成。

## 2026-09-13 Submitted Ours AB v5 49000 cdr-kong

- 操作：submit；task_id=`t-20260914045845-pgvzj`；TaskName=`ab-v5-49000-llada-20260913-cdr-kong`；首次查询 Waiting，查询时间 2026-09-13T20:58:48Z。
- YAML：[eval_ab_v5_49000_llada_20260913_cdr-kong.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_llada_20260913_cdr-kong.yml)；c20250601 / ml.pni2.3xlarge / 1 replica / 1 GPU；**Preemptible:false，非闲时、非抢占**。
- 统一独立评测快照：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913)；实验计划与产物根：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913)。本条是提交记录，不代表评测完成。

## 2026-09-13 Submitted Ours AB v5 49000 cdr-sab23

- 操作：submit；task_id=`t-20260914045849-zgkfs`；TaskName=`ab-v5-49000-llada-20260913-cdr-sab23`；首次查询 Waiting，查询时间 2026-09-13T20:58:52Z。
- YAML：[eval_ab_v5_49000_llada_20260913_cdr-sab23.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_llada_20260913_cdr-sab23.yml)；c20250601 / ml.pni2.3xlarge / 1 replica / 1 GPU；**Preemptible:false，非闲时、非抢占**。
- 统一独立评测快照：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913)；实验计划与产物根：[ab_v5_49000_llada_20260913](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913)。本条是提交记录，不代表评测完成。

## 2026-09-13 Ours AB v5 49000 matrix submission complete

- 11项正式任务全部提交；2026-09-13T20:59:14Z 逐项查询为5项 Running、6项 Queue，另独立 GPU preflight 已 Success。Active 表仅保留正式在途任务；后续状态以重新查询为准。
- 全部使用同一快照、独立单卡非闲时（Preemptible:false）。机器可读任务索引：[submissions.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913/submissions.json)；准确 YAML 与首次状态见上方逐项 dated 条目。
- pairing 全量六组各500×8候选、iter124；三个原生探针及两套 CDR 数据分别运行。门禁短迭代序列不混入正式 CSV；尚无完整测评成绩，也未覆盖旧 baseline/旧42000结果。

## 2026-09-13 Submitted TCR 49000 t1-cdr3b

- submit：`t-20260914062343-ppw4j`，TaskName=`tcr-v5-49000-20260913-t1-cdr3b`；首次查询 Queue（2026-09-13T22:24Z）。c20250601 / ml.pni2.3xlarge / 单卡；Preemptible:false（非闲时）。
- YAML：[eval_tcr_v5_49000_20260913_t1-cdr3b.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t1-cdr3b.yml)。固定同一 v5 49000 快照；全 5 folds × 3 test sets，冻结骨干 + MLP，不是全参微调。
- Binding GPU gate 已通过；提交不代表质量评测完成。

## 2026-09-13 Submitted TCR 49000 t1-cdr3ab

- submit：`t-20260914062429-xhj5d`，TaskName=`tcr-v5-49000-20260913-t1-cdr3ab`；首次查询 Queue（2026-09-13T22:24:57Z）。c20250601 / ml.pni2.3xlarge / 单卡；Preemptible:false（非闲时）。
- YAML：[eval_tcr_v5_49000_20260913_t1-cdr3ab.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t1-cdr3ab.yml)。同一 v5 49000 快照；全 5 folds × 3 test sets，冻结骨干 + MLP。
- 提交记录结束：`t-20260914062429-xhj5d`；通过 binding GPU gate，不等于质量评测完成。

## 2026-09-13 Submitted TCR 49000 t1-others_cdr3b

- submit：`t-20260914062458-rzd6q`，TaskName=`tcr-v5-49000-20260913-t1-others-cdr3b`；首次查询 Queue（2026-09-13T22:25:01Z）。c20250601 / ml.pni2.3xlarge / 单卡；Preemptible:false（非闲时）。
- YAML：[eval_tcr_v5_49000_20260913_t1-others_cdr3b.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t1-others_cdr3b.yml)。同一 v5 49000 快照；全 5 folds × 3 test sets，冻结骨干 + MLP。
- 提交记录结束：`t-20260914062458-rzd6q`；通过 binding GPU gate，不等于质量评测完成。

## 2026-09-13 Submitted TCR 49000 t1-others_longab

- submit：`t-20260914062502-nj89p`，TaskName=`tcr-v5-49000-20260913-t1-others-longab`；首次查询 Queue（2026-09-13T22:25:04Z）。c20250601 / ml.pni2.3xlarge / 单卡；Preemptible:false（非闲时）。
- YAML：[eval_tcr_v5_49000_20260913_t1-others_longab.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t1-others_longab.yml)。同一 v5 49000 快照；全 5 folds × 3 test sets，冻结骨干 + MLP。
- 提交记录结束：`t-20260914062502-nj89p`；通过 binding GPU gate，不等于质量评测完成。

## 2026-09-13 Submitted TCR 49000 t2a

- submit：`t-20260914063213-6twth`，TaskName=`tcr-v5-49000-20260913-t2a`；首次查询 Waiting（2026-09-13T22:32:16Z）。c20250601 / ml.pni2.3xlarge / 单卡；Preemptible:false（非闲时）。
- YAML：[eval_tcr_v5_49000_20260913_t2a.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t2a.yml)。同一 v5 49000 快照；运行时 v5 补全、无表位/结合标签输入、observed-only 池化。T3 标为本地对照，不冒称论文同版复现。
- 提交记录结束：`t-20260914063213-6twth`；T2/T3 GPU gate 和 5 项运行环境回归已通过，不等于质量评测完成。

## 2026-09-13 Submitted TCR 49000 t2b

- submit：`t-20260914063217-fmff2`，TaskName=`tcr-v5-49000-20260913-t2b`；首次查询 Waiting（2026-09-13T22:32:20Z）。c20250601 / ml.pni2.3xlarge / 单卡；Preemptible:false（非闲时）。
- YAML：[eval_tcr_v5_49000_20260913_t2b.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t2b.yml)。同一 v5 49000 快照；运行时 v5 补全、无表位/结合标签输入、observed-only 池化。T3 标为本地对照，不冒称论文同版复现。
- 提交记录结束：`t-20260914063217-fmff2`；T2/T3 GPU gate 和 5 项运行环境回归已通过，不等于质量评测完成。

## 2026-09-13 Submitted TCR 49000 t3deep

- submit：`t-20260914063220-gnnsk`，TaskName=`tcr-v5-49000-20260913-t3deep`；首次查询 Waiting（2026-09-13T22:32:23Z）。c20250601 / ml.pni2.3xlarge / 单卡；Preemptible:false（非闲时）。
- YAML：[eval_tcr_v5_49000_20260913_t3deep.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t3deep.yml)。同一 v5 49000 快照；运行时 v5 补全、无表位/结合标签输入、observed-only 池化。T3 标为本地对照，不冒称论文同版复现。
- 提交记录结束：`t-20260914063220-gnnsk`；T2/T3 GPU gate 和 5 项运行环境回归已通过，不等于质量评测完成。

## 2026-09-13 Submitted TCR 49000 t3broad

- submit：`t-20260914063224-dghz2`，TaskName=`tcr-v5-49000-20260913-t3broad`；首次查询 Waiting（2026-09-13T22:32:27Z）。c20250601 / ml.pni2.3xlarge / 单卡；Preemptible:false（非闲时）。
- YAML：[eval_tcr_v5_49000_20260913_t3broad.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t3broad.yml)。同一 v5 49000 快照；运行时 v5 补全、无表位/结合标签输入、observed-only 池化。T3 标为本地对照，不冒称论文同版复现。
- 提交记录结束：`t-20260914063224-dghz2`；T2/T3 GPU gate 和 5 项运行环境回归已通过，不等于质量评测完成。

## 2026-09-13 Submitted TCR 49000 t4-benchmark14

- submit：`t-20260914064033-ktgjc`，TaskName=`tcr-v5-49000-20260913-t4-benchmark14`；首次查询 Waiting（2026-09-13T22:40:35Z）。c20250601 / ml.pni2.3xlarge / 单卡；Preemptible:false（非闲时）。
- YAML：[eval_tcr_v5_49000_20260913_t4-benchmark14.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t4-benchmark14.yml)。同一 v5 49000 快照，任务定义与门禁见 [T4 §8.0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。双链生成未提交。
- 提交记录结束：`t-20260914064033-ktgjc`；T4 专属 GPU gate 与 5 项 CPU 回归通过，不等于生成评分完成。

## 2026-09-13 Submitted TCR 49000 t4-held20

- submit：`t-20260914064037-74plf`，TaskName=`tcr-v5-49000-20260913-t4-held20`；首次查询 Waiting（2026-09-13T22:40:39Z）。c20250601 / ml.pni2.3xlarge / 单卡；Preemptible:false（非闲时）。
- YAML：[eval_tcr_v5_49000_20260913_t4-held20.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t4-held20.yml)。同一 v5 49000 快照，任务定义与门禁见 [T4 §8.0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。双链生成未提交。
- 提交记录结束：`t-20260914064037-74plf`；T4 专属 GPU gate 与 5 项 CPU 回归通过，不等于生成评分完成。

## 2026-09-13 TCR 49000 当前矩阵提交完成

- 2026-09-13T22:43:13Z 查询十项新 TCR 作业，全部 Queue；排队不等于开始运行或完成。准确 YAML/ID/状态索引：[submissions.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/submissions.json)。
- 三类 GPU gate 已通过；实际运行环境联合回归 71 passed，新增脚本 py_compile 通过。关键改动、范围及可引用性限制见 [TCR 审计 §9.21](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。
- 正式结果未完成；只收指定 49000，不拼接旧 checkpoint。双链生成延后、未提交；未取消/修改既有 AB 或训练作业。

## 2026-09-13 Cancel stale TCR 49000 t2a for scoring re-audit

- cancel：`t-20260914063213-6twth`，TaskName=`tcr-v5-49000-20260913-t2a`；2026-09-13T22:56:52Z 取消前 Queue、CLI cancel success，随即查询终态 Killed。
- YAML：[eval_tcr_v5_49000_20260913_t2a.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t2a.yml)；Preemptible:false，c20250601 / ml.pni2.3xlarge / 单卡。
- 原因：T2A retention 锚点仍是旧 universe、阈值有损取整；修复会改变 T2/T3 共用验收指纹，四项均先取消旧提交，重新验收后按新版本重提。T1/T4 保留。
- 取消记录结束：`t-20260914063213-6twth`；不删除历史 gate/YAML/结果，不操作 AB 或训练作业。

## 2026-09-13 Cancel stale TCR 49000 t2b for scoring re-audit

- cancel：`t-20260914063217-fmff2`，TaskName=`tcr-v5-49000-20260913-t2b`；2026-09-13T22:56:53Z 取消前 Queue、CLI cancel success，随即查询终态 Killed。
- YAML：[eval_tcr_v5_49000_20260913_t2b.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t2b.yml)；Preemptible:false，c20250601 / ml.pni2.3xlarge / 单卡。
- 原因：T2A retention 锚点仍是旧 universe、阈值有损取整；修复会改变 T2/T3 共用验收指纹，四项均先取消旧提交，重新验收后按新版本重提。T1/T4 保留。
- 取消记录结束：`t-20260914063217-fmff2`；不删除历史 gate/YAML/结果，不操作 AB 或训练作业。

## 2026-09-13 Cancel stale TCR 49000 t3deep for scoring re-audit

- cancel：`t-20260914063220-gnnsk`，TaskName=`tcr-v5-49000-20260913-t3deep`；2026-09-13T22:56:54Z 取消前 Queue、CLI cancel success，随即查询终态 Killed。
- YAML：[eval_tcr_v5_49000_20260913_t3deep.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t3deep.yml)；Preemptible:false，c20250601 / ml.pni2.3xlarge / 单卡。
- 原因：T2A retention 锚点仍是旧 universe、阈值有损取整；修复会改变 T2/T3 共用验收指纹，四项均先取消旧提交，重新验收后按新版本重提。T1/T4 保留。
- 取消记录结束：`t-20260914063220-gnnsk`；不删除历史 gate/YAML/结果，不操作 AB 或训练作业。

## 2026-09-13 Cancel stale TCR 49000 t3broad for scoring re-audit

- cancel：`t-20260914063224-dghz2`，TaskName=`tcr-v5-49000-20260913-t3broad`；2026-09-13T22:56:55Z 取消前 Queue、CLI cancel success，随即查询终态 Killed。
- YAML：[eval_tcr_v5_49000_20260913_t3broad.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t3broad.yml)；Preemptible:false，c20250601 / ml.pni2.3xlarge / 单卡。
- 原因：T2A retention 锚点仍是旧 universe、阈值有损取整；修复会改变 T2/T3 共用验收指纹，四项均先取消旧提交，重新验收后按新版本重提。T1/T4 保留。
- 取消记录结束：`t-20260914063224-dghz2`；不删除历史 gate/YAML/结果，不操作 AB 或训练作业。

## 2026-09-13 Submitted TCR 49000 suite audit

- submit：`t-20260914070810-2k5j8`，TaskName=`tcr-v5-49000-20260913-suite-audit`；2026-09-13T23:08:46Z 首次查询 Queue。
- YAML：[eval_tcr_v5_49000_20260913_suite-audit.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_suite-audit.yml)；与 AB 预检同资源：c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false，3600 秒。
- 本地 A100-80GB 已通过 T1 四轨、T2 A/B、T3 deep/broad、T4 两数据集的小样本执行评分验收及 81 项回归；云端额外重新验证实际运行环境。gate：[suite_audit_local_v2/passed.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/suite_audit_local_v2/passed.json)。
- 提交记录结束：`t-20260914070810-2k5j8`；仅执行/评分云端复验，不是正式质量结果。

## 2026-09-13 Resubmitted TCR 49000 t2a-v2 after scoring audit

- resubmit：`t-20260914070922-l6fz2`，TaskName=`tcr-v5-49000-20260913-t2a-v2`；2026-09-13T23:09:25Z 首次查询 Queue。替换已取消的同范围 v1 提交，非重复跑。
- YAML：[eval_tcr_v5_49000_20260913_t2a-v2.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t2a-v2.yml)；c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false，43200 秒。
- 指定 49000 权重不变；v2 真实模型输入 gate、T1–T4 完整小样本执行评分 gate 和 81 项回归已通过，启动再次核验 v2 源码/输入指纹。独立 runtime_v2 输出不覆盖旧 gate/结果。云端额外复验另为独立任务。
- 提交记录结束：`t-20260914070922-l6fz2`；只放行注明限制的本地对照，不证明论文同版复现或预训练无污染。

## 2026-09-13 Resubmitted TCR 49000 t2b-v2 after scoring audit

- resubmit：`t-20260914070926-86nrn`，TaskName=`tcr-v5-49000-20260913-t2b-v2`；2026-09-13T23:09:30Z 首次查询 Queue。替换已取消的同范围 v1 提交，非重复跑。
- YAML：[eval_tcr_v5_49000_20260913_t2b-v2.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t2b-v2.yml)；c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false，43200 秒。
- 指定 49000 权重不变；v2 真实模型输入 gate、T1–T4 完整小样本执行评分 gate 和 81 项回归已通过，启动再次核验 v2 源码/输入指纹。独立 runtime_v2 输出不覆盖旧 gate/结果。云端额外复验另为独立任务。
- 提交记录结束：`t-20260914070926-86nrn`；只放行注明限制的本地对照，不证明论文同版复现或预训练无污染。

## 2026-09-13 Resubmitted TCR 49000 t3deep-v2 after scoring audit

- resubmit：`t-20260914070931-shsrz`，TaskName=`tcr-v5-49000-20260913-t3deep-v2`；2026-09-13T23:09:34Z 首次查询 Queue。替换已取消的同范围 v1 提交，非重复跑。
- YAML：[eval_tcr_v5_49000_20260913_t3deep-v2.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t3deep-v2.yml)；c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false，43200 秒。
- 指定 49000 权重不变；v2 真实模型输入 gate、T1–T4 完整小样本执行评分 gate 和 81 项回归已通过，启动再次核验 v2 源码/输入指纹。独立 runtime_v2 输出不覆盖旧 gate/结果。云端额外复验另为独立任务。
- 提交记录结束：`t-20260914070931-shsrz`；只放行注明限制的本地对照，不证明论文同版复现或预训练无污染。

## 2026-09-13 Resubmitted TCR 49000 t3broad-v2 after scoring audit

- resubmit：`t-20260914070935-lzmgz`，TaskName=`tcr-v5-49000-20260913-t3broad-v2`；2026-09-13T23:09:38Z 首次查询 Queue。替换已取消的同范围 v1 提交，非重复跑。
- YAML：[eval_tcr_v5_49000_20260913_t3broad-v2.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t3broad-v2.yml)；c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false，43200 秒。
- 指定 49000 权重不变；v2 真实模型输入 gate、T1–T4 完整小样本执行评分 gate 和 81 项回归已通过，启动再次核验 v2 源码/输入指纹。独立 runtime_v2 输出不覆盖旧 gate/结果。云端额外复验另为独立任务。
- 提交记录结束：`t-20260914070935-lzmgz`；只放行注明限制的本地对照，不证明论文同版复现或预训练无污染。

## 2026-09-13 TCR re-audit final status snapshot

- 2026-09-13T23:13:11Z 查询当前矩阵：10 项正式＋1 项独立云端 suite 复验全部 **Queue**。旧 T2/T3 四项 Killed，不在 Active 表。新五项平台 get 再核实 Preemptible:false、队列 ID q-20260121145036-6fztt、worker=1、ml.pni2.3xlarge；T1/T4 六项保留，没有重复提交。
- 当前机器可读索引：[submissions_v2.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/submissions_v2.json)。原 submissions.json 保留为 22:43 的历史快照，不能把其中已取消四项的 Queue 当最新状态。
- 本地套件 81 passed，产物：[cpu_tests.xml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/suite_audit_local_v2/cpu_tests.xml)；真实 GPU 完整小样本 suite passed；最终 gate/当前源码数据身份再次一致，新增代码 py_compile 与相关 git diff --check 通过。云端 suite 尚未运行，不标成功；无新正式性能数字。
- 关键修正、已验范围和剩余科学边界见 [TCR 审计 §9.22](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。本轮没有修改/取消 AB 或预训练作业、权重、原始数据及历史性能结果。

## 2026-09-13 Cancel queued t2a-v2 for local short-job policy

- cancel：`t-20260914070922-l6fz2`，TaskName=`tcr-v5-49000-20260913-t2a-v2`；2026-09-13T23:17:44Z 取消前 Queue，CLI cancel success，随后确认 Killed（End=2026-09-13T23:17:43Z）。
- YAML：[eval_tcr_v5_49000_20260913_t2a-v2.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t2a-v2.yml)；原资源 c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false。
- 用户授权短任务本地运行；已确认当前正式输出目录不存在，取消云端后按相同 YAML 参数本地串行执行。
- 转本地取消记录结束：`t-20260914070922-l6fz2`；原 YAML、gate 与历史产物保留，未修改其他任务。

## 2026-09-13 Cancel queued t2b-v2 for local short-job policy

- cancel：`t-20260914070926-86nrn`，TaskName=`tcr-v5-49000-20260913-t2b-v2`；2026-09-13T23:17:45Z 取消前 Queue，CLI cancel success，随后确认 Killed（End=2026-09-13T23:17:45Z）。
- YAML：[eval_tcr_v5_49000_20260913_t2b-v2.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t2b-v2.yml)；原资源 c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false。
- 用户授权短任务本地运行；已确认当前正式输出目录不存在，取消云端后按相同 YAML 参数本地串行执行。
- 转本地取消记录结束：`t-20260914070926-86nrn`；原 YAML、gate 与历史产物保留，未修改其他任务。

## 2026-09-13 Cancel queued t3broad-v2 for local short-job policy

- cancel：`t-20260914070935-lzmgz`，TaskName=`tcr-v5-49000-20260913-t3broad-v2`；2026-09-13T23:17:46Z 取消前 Queue，CLI cancel success，随后确认 Killed（End=2026-09-13T23:17:46Z）。
- YAML：[eval_tcr_v5_49000_20260913_t3broad-v2.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t3broad-v2.yml)；原资源 c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false。
- 用户授权短任务本地运行；已确认当前正式输出目录不存在，取消云端后按相同 YAML 参数本地串行执行。
- 转本地取消记录结束：`t-20260914070935-lzmgz`；原 YAML、gate 与历史产物保留，未修改其他任务。

## 2026-09-13 Cancel queued t1-others_cdr3b for local short-job policy

- cancel：`t-20260914062458-rzd6q`，TaskName=`tcr-v5-49000-20260913-t1-others-cdr3b`；2026-09-13T23:17:48Z 取消前 Queue，CLI cancel success，随后确认 Killed（End=2026-09-13T23:17:47Z）。
- YAML：[eval_tcr_v5_49000_20260913_t1-others_cdr3b.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t1-others_cdr3b.yml)；原资源 c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false。
- 用户授权短任务本地运行；已确认当前正式输出目录不存在，取消云端后按相同 YAML 参数本地串行执行。
- 转本地取消记录结束：`t-20260914062458-rzd6q`；原 YAML、gate 与历史产物保留，未修改其他任务。

## 2026-09-13 Cancel queued t1-cdr3ab for local short-job policy

- cancel：`t-20260914062429-xhj5d`，TaskName=`tcr-v5-49000-20260913-t1-cdr3ab`；2026-09-13T23:17:49Z 取消前 Queue，CLI cancel success，随后确认 Killed（End=2026-09-13T23:17:48Z）。
- YAML：[eval_tcr_v5_49000_20260913_t1-cdr3ab.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t1-cdr3ab.yml)；原资源 c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false。
- 用户授权短任务本地运行；已确认当前正式输出目录不存在，取消云端后按相同 YAML 参数本地串行执行。
- 转本地取消记录结束：`t-20260914062429-xhj5d`；原 YAML、gate 与历史产物保留，未修改其他任务。

## 2026-09-13 Cancel queued t1-others_longab for local short-job policy

- cancel：`t-20260914062502-nj89p`，TaskName=`tcr-v5-49000-20260913-t1-others-longab`；2026-09-13T23:17:50Z 取消前 Queue，CLI cancel success，随后确认 Killed（End=2026-09-13T23:17:50Z）。
- YAML：[eval_tcr_v5_49000_20260913_t1-others_longab.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t1-others_longab.yml)；原资源 c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false。
- 用户授权短任务本地运行；已确认当前正式输出目录不存在，取消云端后按相同 YAML 参数本地串行执行。
- 转本地取消记录结束：`t-20260914062502-nj89p`；原 YAML、gate 与历史产物保留，未修改其他任务。

## 2026-09-13 Cancel queued suite-audit for local short-job policy

- cancel：`t-20260914070810-2k5j8`，TaskName=`tcr-v5-49000-20260913-suite-audit`；2026-09-13T23:17:52Z 取消前 Queue，CLI cancel success，随后确认 Killed（End=2026-09-13T23:17:51Z）。
- YAML：[eval_tcr_v5_49000_20260913_suite-audit.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_suite-audit.yml)；原资源 c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false。
- 独立云端复验不再重复排队，沿用已完整通过的本地 suite gate；不声称云端复验成功。
- 转本地取消记录结束：`t-20260914070810-2k5j8`；原 YAML、gate 与历史产物保留，未修改其他任务。

## 2026-09-13 Started short TCR evaluations locally

- 用户允许简单、短耗时任务在本机运行；六个正式任务及一个冗余云端复验均已在 Queue 时取消、确认 Killed。保留四个云端正式任务：大数据 T1 cdr3b、T3 deep、T4 benchmark14/held20，最后查询仍 Queue。未操作 AB/预训练/常驻占卡进程。
- 本机 host=`di-20260121220916-9rvmj`，A100-SXM4-80GB；启动前已有约 1.7 GiB GPU 常驻占用，并非完全空闲。tmux session=`tcr49000-local-short`，runner PID=719232，启动于 2026-09-13T23:19:23Z；本轮每次仅一个正式子任务占用 GPU，OMP/OpenBLAS 仍为 8 线程。
- 顺序：t2a-v2 → t2b-v2 → t3broad-v2 → t1-others_cdr3b → t1-cdr3ab → t1-others_longab。目前 T2A 已启动，实际读取 4,779 行，子任务 10 项评分回归通过，其余等待本地串行队列。
- 入口：[run_tcr_local_short.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_tcr_local_short.py)；冻结计划：[local_plan_v1.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/local_plan_v1.json)；实时状态：[local_short_v1/status.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/local_short_v1/status.json)。任务日志保留原 OUT/key/run.log，另在 local_short_v1 保存逐任务完整日志和耗时。
- 启动前 `--check-only`、py_compile 通过；逐项核对云端任务 ID/名称/Killed、YAML SHA/bash 语法、结果目录不存在及原工程 gate 当前身份。原 YAML Entrypoint 原样执行，不降低完整数据、epoch、fold、候选预算或改变 batch/seed。运行互斥锁与结果目录保护防止误重跑；GPU 空闲显存不足 24 GiB 拒绝启动，失败即停止后续，单项实测超过 10 分钟则完成该项后暂停后续排程。
- 本地启动登记结束：原数据、权重、采样/训练参数与实验输出身份不变；完整结果仍需独立验收，不以进程退出码替代指标复核。

## 2026-09-13 TCR local serial execution status

- 2026-09-13T23:24:03Z 本地进度快照：t2a-v2=executed_pending_result_audit（78.68s）；t2b-v2=executed_pending_result_audit（146.29s）；t3broad-v2=running；t1-others_cdr3b=pending；t1-cdr3ab=pending；t1-others_longab=pending。进程/状态文件仍为 running；`executed_pending_result_audit` 表示执行和预期 JSON 落盘，不表示最终指标独立复算验收。
- 运行地点索引：[execution_plan_v3.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/execution_plan_v3.json)，实时状态唯一取 local_short_v1/status.json。四项较大任务仍云端 Queue（本轮最近查询）；没有重提已取消的七项或干预其他作业。
- 本地迁移代码 py_compile/check-only 和文档 diff 检查通过；checkpoint/输入 gate/YAML 指纹保持一致。后续按完整产物验收再汇总，只收 49000，不把本轮落盘结果与旧 checkpoint 拼接。


## 2026-09-13 Cancel queued t3deep-v2 after others completed

- cancel：`t-20260914070931-shsrz`，TaskName=`tcr-v5-49000-20260913-t3deep-v2`；2026-09-13T23:38:43Z 复核终态：取消前 Queue，CLI cancel success，随后确认 Killed（End=2026-09-13T23:38:00Z）。
- YAML：[eval_tcr_v5_49000_20260913_t3deep-v2.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t3deep-v2.yml)；原资源 c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false。
- 用户要求 others 完成后本地接续并持续监控。六项前置队列已全部执行完成（最后一项 LongAB 于 23:36:37Z 正常退出），原参数/完整数据/49000 不变；云端终态确认后才允许本地启动，原 YAML/历史产物保留。
- 接续本地取消记录结束：`t-20260914070931-shsrz`；只操作本轮 TCR 排队作业，未干预其他任务。


## 2026-09-13 Cancel queued t4-benchmark14 after others completed

- cancel：`t-20260914064033-ktgjc`，TaskName=`tcr-v5-49000-20260913-t4-benchmark14`；2026-09-13T23:38:44Z 复核终态：取消前 Queue，CLI cancel success，随后确认 Killed（End=2026-09-13T23:38:44Z）。
- YAML：[eval_tcr_v5_49000_20260913_t4-benchmark14.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t4-benchmark14.yml)；原资源 c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false。
- 用户要求 others 完成后本地接续并持续监控。六项前置队列已全部执行完成（最后一项 LongAB 于 23:36:37Z 正常退出），原参数/完整数据/49000 不变；云端终态确认后才允许本地启动，原 YAML/历史产物保留。
- 接续本地取消记录结束：`t-20260914064033-ktgjc`；只操作本轮 TCR 排队作业，未干预其他任务。


## 2026-09-13 Cancel queued t4-held20 after others completed

- cancel：`t-20260914064037-74plf`，TaskName=`tcr-v5-49000-20260913-t4-held20`；2026-09-13T23:38:45Z 复核终态：取消前 Queue，CLI cancel success，随后确认 Killed（End=2026-09-13T23:38:45Z）。
- YAML：[eval_tcr_v5_49000_20260913_t4-held20.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t4-held20.yml)；原资源 c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false。
- 用户要求 others 完成后本地接续并持续监控。六项前置队列已全部执行完成（最后一项 LongAB 于 23:36:37Z 正常退出），原参数/完整数据/49000 不变；云端终态确认后才允许本地启动，原 YAML/历史产物保留。
- 接续本地取消记录结束：`t-20260914064037-74plf`；只操作本轮 TCR 排队作业，未干预其他任务。


## 2026-09-13 Cancel queued t1-cdr3b after others completed

- cancel：`t-20260914062343-ppw4j`，TaskName=`tcr-v5-49000-20260913-t1-cdr3b`；2026-09-13T23:38:47Z 复核终态：取消前 Queue，CLI cancel success，随后确认 Killed（End=2026-09-13T23:38:46Z）。
- YAML：[eval_tcr_v5_49000_20260913_t1-cdr3b.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t1-cdr3b.yml)；原资源 c20250601 / ml.pni2.3xlarge / 1 GPU / Preemptible:false。
- 用户要求 others 完成后本地接续并持续监控。六项前置队列已全部执行完成（最后一项 LongAB 于 23:36:37Z 正常退出），原参数/完整数据/49000 不变；云端终态确认后才允许本地启动，原 YAML/历史产物保留。
- 接续本地取消记录结束：`t-20260914062343-ppw4j`；只操作本轮 TCR 排队作业，未干预其他任务。


## 2026-09-13 Remaining TCR jobs started locally after others finished

- 2026-09-13T23:41:24Z 状态：六项前置本地任务全部执行完成；others 三轨的 45 个逐样本预测文件已独立验收通过，覆盖五折×三个测试集，模型/特征/head/数据成员身份、训练内标准化均值、逐预测评分和折间 mean/std 一致。验收证据：[others.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v1/others.json)；新数字仅入 RESULTS，不复制本过程日志。
- 四个剩余云端作业均 Queue 时取消至 Killed；最新按本轮 TaskName 查询返回空列表。准确 ID/YAML/Preemptible:false 与终态已在上一组逐项记录；没有重提或操作其他 AB/训练作业。
- 本地接续启动：host=di-20260121220916-9rvmj，A100-80GB，tmux=tcr49000-local-remaining，PID=732591，start=2026-09-13T23:39:40Z。顺序 T3 deep → T4 benchmark14 → T4 held20 → T1 大数据 cdr3b；T3 deep 已进入正式完整 25,816 行任务。每次仅一项 GPU 评测，原常驻进程保留。
- 新入口 py_compile、6 项前置/范围拒绝检查、完整 --check-only（前置六项/原输出空/云端 Killed/YAML/gate）通过；启动时重新验证。指纹：plan a3c022ad2b768d566ae532c9eb4febbd22fe0108198d274c25c794ad6845226c，runner 0664ec3c1dc61a42d05b4a27952f7b6e44e48daca96251b9e177c4157d87c222。
- 冻结计划：[local_remaining_plan_v1.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/local_remaining_plan_v1.json)；实时状态：[local_remaining_v1/status.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/local_remaining_v1/status.json)；最新运行地点索引：[execution_plan_v4.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/execution_plan_v4.json)。v3 及此前提交索引均保留为历史。
- 接续启动记录结束：所有取消与运行地点可追溯，持续监控状态/日志/显存；执行完成不自动等于论文可比或预训练无重叠。


## 2026-09-13 Persistent local TCR monitoring and audited partial results

- 2026-09-13T23:56:41Z：已完整执行 7/10 正式任务（others×3、T2 A/B、T3 broad/deep），对应结果审计通过；T1 为逐预测重算，T2/T3 为保存曲线/episode/聚合一致性，不把两类审计说成相同强度。新数字只收指定 49000，见 RESULTS §0.1–§0.3。T3 deep 23:45:17Z exit 0、耗时 336.68 秒。
- 当前运行 T4 benchmark14（随后 held20，再大数据 T1 cdr3b）；原本地 runner PID 732591 未中断。每个 pMHC 100 stochastic + 1 independent greedy；32 为 max_iter 上限，不强求实际 history 长度必须等于 32。原配置未调小、未知上下文 X 未改。
- 新持久监控：[monitor_tcr_49000_local.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/monitor_tcr_49000_local.py)，tmux=tcr49000-monitor-v2，PID=741215，start=2026-09-13T23:54:20+00:00；每 30 秒核对两队列、producer PID/命令/host、计划/runner/YAML 指纹、GPU 和日志心跳。日志 900 秒未更新记警示，不凭此杀任务；生产子任务失败由原 runner 停止后续，监控/结果审计异常标 attention_required，不取消/杀死或重试任何其他进程。
- 已完成任务自动核对既有审计报告及产物 SHA；以后完成的 T4/T1 会以 CPU 启动已固定脚本复核，成功后才加入本轮 JSON 汇总。全部 10 项执行且审计通过才记 all_tasks_executed_and_audits_passed；不把待跑项写成完成。实时 [monitor_v2/status.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/monitor_v2/status.json)、[results_49000.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/monitor_v2/results_49000.json)；冻结 [monitor_plan_v2.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/monitor_plan_v2.json)。已观察至少两轮心跳、7 项验收、alerts=[]。
- 监控准备中保留一处检查失败：新增生成结果审计曾错误要求 actual steps==32，实际已落盘完整候选为 30/31 步。原 sampler 在所有目标解掩码后 break，这是正常规则，不是降预算或生产 bug。仅停止自建监控 PID 739836，保留 [v1 diagnostic](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/monitor_v1/auditor_check_failure.json) 与快照；修结果审计为 1≤steps≤32 并保持完整序列/上下文/预算约束，CPU 检查重跑通过，以新计划 v2 启动。生产 runner/原 sampler/其他任务未动。
- 新脚本 py_compile、拒绝缺失队列/指标不一致的检查、已生成三个 pMHC 的实际步数/固定上下文检查及 monitor --check-only 通过；现有七项结果报告的 source/artifact SHA 一致。文档登记关键区别，未覆盖旧结果或原始数据。
- 持续监控启动登记结束：后台会继续运行，JSON 心跳/逐任务审计为实时事实来源；未承诺未经实现的消息推送或自动修改论文表。

## 2026-09-14 TCR status query: generation output serialization failed

- 2026-09-14T04:18:55Z 查询确认：7/10 项已完整执行并通过各自层级产物核对；T4 benchmark14 于 00:15:30Z exit 1，local_remaining_v1 停止，held20 和大数据 T1 cdr3b 未启动。监控于 00:15:52Z 转 attention_required 并退出；tmux 已无本轮 runner/monitor，不能继续把此前运行快照说成当前仍在跑。
- benchmark14 的 14 个 pMHC、每个 100 条有效候选＋独立 greedy 已全部落盘。失败发生在保存 metrics.json 时，ValueError: Out of range float values are not JSON compliant: nan；不是 CUDA/OOM/非法序列失败。只读 CPU 重评分复现：非有限值只来自辅助 novelty_vs_conditioning 及其聚合；当前模型/数据/评分代码身份与原 manifest 一致。具体缺陷 owner 见 [T4 §7.7](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)，本次不新增正式 T4 数字。
- 证据：[原运行日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/local_remaining_v1/t4-benchmark14.log)、[队列终态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/local_remaining_v1/status.json)、[monitor 终态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/monitor_v2/status.json)。monitor.jobs 中 benchmark14=running/13 是抛错前的上一轮快照，不应覆盖根 status/error 与正式队列终态。
- 本轮 TaskName 云端查询为空；已取消任务没有恢复，也未操作 AB/其他常驻进程。本次仅核查、只读 CPU 诊断及同步状态文档，没有修改生产/监控代码、覆盖产物、重跑、重提或恢复队列。后续可复用现有完整设计序列修复写出/重评分，再安排未启动两项。

## 2026-09-14 TCR serialization repaired and local queue resumed

- 用户授权修复后，完成 T4 专属 JSON 边界/CPU 恢复、100 项回归、新版真实 GPU gate、benchmark14 全量独立重评分；具体协议和证据唯一见 [T4 §7.7](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) / [总审计 §9.26](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。新增正式数字只进入 RESULTS §0.4，旧 failed/raw/日志完整保留。
- 真实恢复计划 check-only 通过，原 7 个完成结果与新 benchmark14 恢复结果均核对 artifact SHA；只启动两项原本 pending 的任务。原配置 YAML 未改，Preemptible:false；没有新云端 submit/cancel，云端 Active 表不新增本地 PID。启动前只读确认 `t-20260914064037-74plf`（held20）与 `t-20260914062343-ppw4j`（大数据 β-only）的原 TaskName 匹配且仍为 Killed，避免双跑。
- 本地 producer：host=`di-20260121220916-9rvmj`，tmux=`tcr49000-resume-v2`，PID=889428，start=`2026-09-14T04:37:22+00:00`。固定 [local_resume_plan_v2.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/local_resume_plan_v2.json)，状态 [local_resume_v2/status.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/local_resume_v2/status.json)。held20 模型进程 PID=889676，原 [held20 YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t4-held20.yml) 与后续 [T1 cdr3b YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_49000_20260913_t1-cdr3b.yml) 不变；仅 T4 入口默认新 gate/新输出版本，候选/数据/权重参数不变。
- 持久 monitor：tmux=`tcr49000-monitor-v3`，PID=889780，start=`2026-09-14T04:37:35+00:00`。复用未修改 monitor 代码、冻结 [monitor_plan_v3.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/monitor_plan_v3.json)，[实时状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/monitor_v3/status.json) / [49000-only 结果](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/monitor_v3/results_49000.json)。截至 04:39:36Z 已连续 5 次心跳，executed=8、audited=8，held20 running、T1 cdr3b pending、alerts=[]；首个 pMHC 已 100/100 valid。完成后自动只读验收并收数，不自行重试/改预算。
- 本轮只操作该 TCR 恢复与监控，未改 checkpoint、原始数据、采样器、AB 代码/作业或其他常驻 GPU 进程。新执行位置索引见 [execution_plan_v5.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/execution_plan_v5.json)，旧 v4 与失败队列保留历史。

## 2026-09-14 Ours AB v5 49000 全部完成（只读复核）

- 04:15 UTC 逐项查询11项正式任务全部 **Success**；从两张 Active 表移除，未 submit/cancel、未重跑模型、未修改训练或原始指标。独立 preflight 仍为此前已记录的 Success。
- 本次11项终态如下；每项对应的绝对 YAML、非闲时/Preemptible:false 和统一快照 SHA 见此前 `Submitted Ours AB v5 49000` 条目。

| 子任务 | task ID | 状态 | End（UTC） |
|---|---|---|---|
| pair-p0-cfg0 | `t-20260914045749-mv7sv` | Success | 2026-09-13T23:39:07Z |
| pair-p0-cfg1 | `t-20260914045802-jxn8p` | Success | 2026-09-14T00:15:58Z |
| pair-p0-cfg1p5 | `t-20260914045806-sgrlb` | Success | 2026-09-14T00:19:50Z |
| pair-p3-cfg0 | `t-20260914045810-ss6ml` | Success | 2026-09-13T23:42:33Z |
| pair-p3-cfg1 | `t-20260914045814-csgm4` | Success | 2026-09-14T00:16:26Z |
| pair-p3-cfg1p5 | `t-20260914045817-trh7j` | Success | 2026-09-14T02:30:17Z |
| specificity | `t-20260914045834-cfk6n` | Success | 2026-09-13T23:43:22Z |
| gdp_a1 | `t-20260914045837-2lp8v` | Success | 2026-09-13T23:43:54Z |
| m396 | `t-20260914045841-v242p` | Success | 2026-09-13T23:58:03Z |
| cdr-kong | `t-20260914045845-pgvzj` | Success | 2026-09-13T23:53:18Z |
| cdr-sab23 | `t-20260914045849-zgkfs` | Success | 2026-09-13T23:55:29Z |

- 六组 pairing 均完整500×8；原始行/重链/参考轻链/前缀/声明长度及逐条有效率、V/J/类型匹配汇总检查通过；六组记录的权重、输入数据及生成源码 SHA 全部一致。本轮未重跑 ImmunoMatch 神经评分器。
- 三探针 metrics 与保存预测核对结果及数值统一见 [RESULTS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md) 的2026-09-14 AB条目；CDR为已有日志聚合值，缺逐序列预测，不能声称独立全量重评分。

## 2026-09-14 TCR comparison collection and remaining execution

- 05:08:03Z 只读检查：原 producer PID 889428、held20 PID 889676、monitor PID 889780 正常；8/10 已审计，held20 已落盘 14/20 个目标（每个 100/100 valid），大数据 T1 cdr3b 仍待前一项结束。未修改 YAML、模型输入/gate/预算，未 submit/cancel、重启或操作 AB/其他 GPU 进程。
- 新增官方补充表原件和收集/比较代码；来源身份唯一见下载清单，协议/比较局限见任务 owner，关键决策见 [总审计 §9.27](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。激活 pllm 后切到 protenix_abtcr，10 项 pytest 在 0.38 秒内通过、py_compile 和 monitor --check-only 通过；已生成明确 incomplete 的分层比较 JSON，不误标当前全量结束。
- 新 CPU-only 等待器：host=`di-20260121220916-9rvmj`，tmux=`tcr49000-comparison-v1`，PID=907034，start=`2026-09-14T05:07:38+00:00`。入口 `collect_tcr_49000_comparison.py --wait-for-complete`，输出目录 [comparison_v1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/comparison_v1)，[实时收口状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/comparison_v1/final_49000_vs_baselines_and_29000.status.json) 和 waiter.log 每 30 秒更新。只等既有 monitor 完成；全部 10 项审计通过后才创建 `final_49000_vs_baselines_and_29000.json`，否则明确异常，不重试模型、不自动修改 Markdown 数字。新旧部分汇总保留，不覆盖原始评测/历史结果。
- 补充回归：05:12Z，CPU 执行 binding query/others-beta/long、generation-v5、native-features、scoring-audit、generation-result-io、comparison-collection 八个测试文件，**104 passed in 5.06s**；仅两条已有 `np.trapz` 弃用警告，不改运行中的评分实现。数字转录另核对 41 个已报告 others 论文单元格、9 个新旧 checkpoint 对照行、6 个生成 baseline 行，均与来源匹配；`git diff --check` 通过。
- 后续状态（05:23:54Z）：held20 于 **05:20:39Z exit 0**，完整生成约 43 分钟；monitor 于 **05:21:10Z** 已完成其独立审计，audited=9，未出现写出故障。T1 大数据 cdr3b 于 **05:20:42Z** 自动接续，日志已至 **14,336/462,883（3.1%）** 唯一输入、约 90.6 pairs/s；五折分类头尚未开始，不能用 feature 百分比当训练 epoch 或完整评测进度。此时预计特征阶段还需约 83 分钟，之后另有完整头训练与验收，估算不作完成承诺。队列/monitor/collector 健康，云端作业不恢复。
- 最新已完成部分见 [audited9_20260914.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/comparison_v1/audited9_20260914.json)，`complete=false, missing=[t1-cdr3b]`；held20 数字与 baseline 对照已入 RESULTS §0.4。自动 final JSON 仍须等待最后一项审计，Markdown 不自动覆写。

## 2026-09-15 Submitted AB v5 49000 Specificity 200 epoch diagnostic

- 操作：submit（独立新实验，非100轮状态续训）。task_id=`t-20260915103954-2sw2c`，TaskName=`ab-v5-49000-specificity-ep200-20260915`；YAML：[eval_ab_v5_49000_specificity_ep200_20260915.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_specificity_ep200_20260915.yml)。`Preemptible:false`，`c20250601`，单worker/单卡`ml.pni2.3xlarge`，deadline7200秒。
- API Start=`2026-09-15T02:39:55Z`，首次get状态`Running`；02:40:43Z已读到真实worker输出fold0第23/200轮，非仅调度状态。随后fold0训练完成并进入fold1，尚无完整五折200轮结果。原10项CPU门禁及精确缓存provenance检查通过，特征没有重提。
- 具体LR、fixed-last、诊断曲线/中间头和禁止选优契约见 [AB Native Probes §4.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md#45-specificity-200-epoch-学习曲线诊断2026-09-15)。同v5 49000快照，SHA=`d5f892bce7d18106a7c417a3e4605f2bf1feac2bfa80709ca74fb149e5c07d92`；从旧缓存独立复制4398×768特征并严格核对数据/顺序/源码/权重/精度，不修改原目录。
- 新产物：[ep200目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_specificity_ep200_20260915)，运行日志`run.log`，每折实时`epoch_metrics.json`，每20轮/末轮diagnostic head，全部完成后`epoch_mean.json`、5个selected heads、OOF、metrics。不得从第一个fold中间指标外推完整结果。
- 本轮10项CPU回归通过（5.23s），日志开关不改变同seed训练loss/最终state_dict；真实缓存CPU门禁和YAML/Bash/资源/新目录检查通过。未修改基础模型、pooling、其他AB/TCR任务或GDPa1，不重跑Ophiuchus；其50/100/200/300轮历史记录仅用于诊断对照，数字见RESULTS §0.6a。

### 2026-09-15T02:47:04Z Success 与结果验收

- Volc终态：`Success`，Start=`2026-09-15T02:39:55Z`，End=`2026-09-15T02:47:04Z`，Elapsed=429秒；已从两张Active表移除。
- 固定第200轮正式五折等权均值：Accuracy=`0.6273252146 ± 0.0128182319`、macro-F1=`0.6257822727 ± 0.0127047818`、MCC=`0.4416933132 ± 0.0193624453`（SD ddof=1）；末轮train loss=`0.7427279986`、留出折loss=`0.8384949803`。论文Ophiuchus-Ab baseline为0.6796/0.6790/0.5203，不以本地Ophiuchus重跑替换。
- 完整性与复算：5折各200条逐轮记录、每折10个诊断头、5个selected head、4398条OOF齐全；所有折末轮`lr_next_step=0`。逐轮五折均值、正式折指标、正式均值与第200轮逐字段复算最大误差均为0。OOF全样本聚合复算为0.627331/0.625866/0.441598，与正式折等权均值的约1e-4微差来自聚合方式。
- 收敛诊断：同一次200轮运行第100→200轮留出折loss `0.847112→0.838495`、Accuracy `0.620958→0.627325`；平均loss在第200轮为全程最低，各折最低loss在141/184/193/178/187轮，末轮均低于各折第100轮。判断为后期缓慢改善伴随波动、没有明显过拟合反弹；第191轮Accuracy瞬时最高0.630281仅作诊断，未用于选头或正式报分。

## 2026-09-15 Submitted AB v5 step92000 Specificity 200 epoch comparison

- 用户要求用“最新89000”测试；只读盘点确认v5训练的step89000已被top-k清理，当前实际最新完整保存点为step92000（训练任务`t-20260912150714-lwf28`仍Running）。为遵循“最新”意图，按step92000执行并明确标号，不将其写成89000。
- 提交：task_id=`t-20260915132856-tlxtc`，TaskName=`ab-v5-92000-specificity-ep200-20260915`；首次查询`Staging`，05:30Z已进入`Running`，API Start=`2026-09-15T05:28:56Z`，worker Launch=`05:29:19Z`。YAML：[eval_ab_v5_92000_specificity_ep200_20260915.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_specificity_ep200_20260915.yml)；`c20250601`非闲时、单卡`ml.pni2.3xlarge`、`Preemptible:false`、deadline7200秒。
- 在训练top-k继续清理前冻结只读评测快照：[ab_v5_92000_llada_20260915](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_92000_llada_20260915)，权重SHA256=`2f003a5b42a15498a7d6f2fc51414cad717f96a9cb405dca2388fcef0de3039c`，与源checkpoint逐位一致；`trainer_state.global_step=92000`。快照/YAML/Bash/新输出目录门禁通过。
- 与step49000正式比较保持同一4398条TTE五折、post-LLaDA完整H/L有效残基global mean、768维、batch8、lr5e-5、seed42+fold、200轮fresh head和fixed-last；逐轮留出折曲线仅诊断。新checkpoint必须重新提取表征，不复用49000的`features.pt`。输出：[step92000结果目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_ep200_20260915)。

### 2026-09-15T05:36:53Z Success 与结果验收

- Volc终态`Success`，Start=`2026-09-15T05:28:56Z`，End=`05:36:53Z`，Elapsed=477秒；已从两张Active表移除。4398条step92000表征重新提取完成，没有复用step49000缓存。
- 固定第200轮五折等权均值：Accuracy=`0.6434677836 ± 0.0157851642`、macro-F1=`0.6417304340 ± 0.0152889822`、MCC=`0.4661190972 ± 0.0240161409`；train loss=`0.7071762518`、留出折loss=`0.8211460233`。相对同协议step49000/200轮分别提升`+0.016143/+0.015948/+0.024426`，相对论文Ophiuchus-Ab仍差`0.036132/0.037270/0.054181`。
- 完整性与复算：五折各连续1–200轮、各10个诊断头、5个selected head、4398条OOF齐；末轮LR全为0。逐轮均值、折指标、折等权均值与第200轮一致性最大误差均为0。OOF全样本聚合为Accuracy/F1/MCC=`0.643474/0.641885/0.465870`，与折等权均值的微差来自聚合方式。
- 收敛：同次运行第100→200轮平均留出折loss=`0.828617→0.821146`、Accuracy=`0.639832→0.643468`；五折末轮loss均不高于各自第100轮。平均loss第200轮全程最低；各折最低loss轮次`102/184/173/119/139`，属于后期平台波动。Accuracy瞬时最高为第189轮`0.646197`，仅诊断，不用于选头。

## 2026-09-15 Submitted v5 92000 Specificity 300 / 400 / 500 epoch

- 用户要求比较三个更长预算；全部是同92000的独立fresh heads，不从200轮续训。缓存、scheduler和fixed-last契约见 [AB Native Probes §4.7](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md)。只增加本次三项，既有训练/AB/TCR作业不动。
- submit：`t-20260915143055-j547p`，TaskName=`ab-v5-92000-specificity-ep300-20260915`，YAML：[300轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_specificity_ep300_20260915.yml)。提交时间06:30:55Z，06:31:16Z首次复核状态 **Queue**；单worker/单卡 `ml.pni2.3xlarge`，`c20250601`，`Preemptible:false`，deadline7200秒。
- submit：`t-20260915143055-pgzq6`，TaskName=`ab-v5-92000-specificity-ep400-20260915`，YAML：[400轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_specificity_ep400_20260915.yml)。提交时间06:30:55Z，06:31:16Z首次复核状态 **Running**；单worker/单卡 `ml.pni2.3xlarge`，`c20250601`，`Preemptible:false`，deadline7200秒。
- submit：`t-20260915143055-xzftx`，TaskName=`ab-v5-92000-specificity-ep500-20260915`，YAML：[500轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_specificity_ep500_20260915.yml)。提交时间06:30:55Z，06:31:16Z首次复核状态 **Queue**；单worker/单卡 `ml.pni2.3xlarge`，`c20250601`，`Preemptible:false`，deadline7200秒。
- 提交前10项CPU回归通过（8.32秒）；三份YAML的单卡/非闲时/预算/缓存来源及bash -n通过；真实缓存门禁核对4398×768、五折、checkpoint step/SHA以及特征provenance通过，临时副本未触发骨干加载。源features.pt SHA=`276981b93993375f1e6f270a9199bd487a410b515354aa16069d66b17619bb0e`，检查前后不变。
- 每项输出独立目录 `output/downstream_generation/ab_v5_92000_specificity_ep{300,400,500}_20260915`，含run.log、run_manifest、features副本、每折逐轮曲线/诊断头/最终头/预测，全部完成后有epoch_mean与metrics。不得由某一折或最高曲线点提前填正式结果。当前结果页仅标待完成，历史200轮不覆盖。

## 2026-09-15 v5 92000 长预算评测自动验收

后台会话`ab92000-specificity-budget-monitor-v2`接替仅只读的旧监控（只停止监控会话，未取消训练任务）。发布器4项CPU测试通过；每30秒核验，限定维护这三条结果及对应终态。日志：[monitor_v2.log](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_budget_audit_20260915/monitor_v2.log)。

<!-- ab92000-budget-terminal:start -->
- `t-20260915193857-zmtc6` / 100轮：**Success**；End=2026-09-15T11:43:19Z；[完整验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_budget_audit_20260915/ep100_audit.json)。
- `t-20260915143055-j547p` / 300轮：**Success**；End=2026-09-15T06:54:19Z；[完整验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_budget_audit_20260915/ep300_audit.json)。
- `t-20260915143055-pgzq6` / 400轮：**Success**；End=2026-09-15T06:44:11Z；[完整验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_budget_audit_20260915/ep400_audit.json)。
- `t-20260915143055-xzftx` / 500轮：**Success**；End=2026-09-15T07:10:37Z；[完整验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_budget_audit_20260915/ep500_audit.json)。
- 数字只进入 RESULTS §0.6a 的训练预算诊断表；未按曲线最优轮选模。其余任务与原始产物未修改。
<!-- ab92000-budget-terminal:end -->

## 2026-09-15 Submitted Ours v5 92000 full AB/TCR matrix

- 操作：submit。用户要求对当前最新完整保存点 92000 按 AB/TCR 审计清单逐项测试；49000 结果仅作对照，不覆盖。锁定既有独立快照 [ab_v5_92000_llada_20260915](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_92000_llada_20260915)，权重 SHA256=`2f003a5b42a15498a7d6f2fc51414cad717f96a9cb405dca2388fcef0de3039c`，与训练目录 `checkpoint-92000` 重算一致；`trainer_state.global_step=92000`。未复制优化器，未改训练任务。训练侧现已出现 95000，本轮仍固定 92000。
- Specificity：200/300/400/500 轮已 Success 并写入 RESULTS 诊断表，故不重复提交 100 轮作业。YAML 已生成但未 submit：[eval_ab_v5_92000_full_20260915_specificity.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_specificity.yml)。
- 其余 20 项均为 `c20250601` / `ml.pni2.3xlarge` / 1 replica / 1 GPU / **Preemptible:false**。首次查询：`pair-p0-cfg0` 为 Staging，其余 19 项为 Queue。机器可读索引：[all_92000_matrix_submissions.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/all_92000_matrix_submissions.json)。
- AB 10 项（独立输出根 [ab_v5_92000_full_20260915](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915)）：
  - `t-20260915153001-lpxs5` `ab-v5-92000-full-20260915-pair-p0-cfg0` → [pair-p0-cfg0.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p0-cfg0.yml)
  - `t-20260915153015-g4999` `ab-v5-92000-full-20260915-pair-p0-cfg1` → [pair-p0-cfg1.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p0-cfg1.yml)
  - `t-20260915153020-qhkn9` `ab-v5-92000-full-20260915-pair-p0-cfg1p5` → [pair-p0-cfg1p5.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p0-cfg1p5.yml)
  - `t-20260915153025-7dc7n` `ab-v5-92000-full-20260915-pair-p3-cfg0` → [pair-p3-cfg0.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p3-cfg0.yml)
  - `t-20260915153030-j87kk` `ab-v5-92000-full-20260915-pair-p3-cfg1` → [pair-p3-cfg1.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p3-cfg1.yml)
  - `t-20260915153034-9vsmd` `ab-v5-92000-full-20260915-pair-p3-cfg1p5` → [pair-p3-cfg1p5.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p3-cfg1p5.yml)
  - `t-20260915153040-km76g` `ab-v5-92000-full-20260915-gdp-a1` → [gdp_a1.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_gdp_a1.yml)
  - `t-20260915153045-jj6fj` `ab-v5-92000-full-20260915-m396` → [m396.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_m396.yml)
  - `t-20260915153050-stwdn` `ab-v5-92000-full-20260915-cdr-kong` → [cdr-kong.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_cdr-kong.yml)
  - `t-20260915153055-tfnfz` `ab-v5-92000-full-20260915-cdr-sab23` → [cdr-sab23.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_cdr-sab23.yml)
- TCR 10 项（独立输出根 [tcr_v5_92000_20260915](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_92000_20260915)）：
  - `t-20260915153101-dn5qm` `tcr-v5-92000-20260915-t1-cdr3b` → [t1-cdr3b.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t1-cdr3b.yml)
  - `t-20260915153106-d4ccf` `tcr-v5-92000-20260915-t1-cdr3ab` → [t1-cdr3ab.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t1-cdr3ab.yml)
  - `t-20260915153111-4q55g` `tcr-v5-92000-20260915-t1-others-cdr3b` → [t1-others_cdr3b.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t1-others_cdr3b.yml)
  - `t-20260915153116-t29vf` `tcr-v5-92000-20260915-t1-others-longab` → [t1-others_longab.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t1-others_longab.yml)
  - `t-20260915153121-hp4j9` `tcr-v5-92000-20260915-t2a-v2` → [t2a-v2.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t2a-v2.yml)
  - `t-20260915153125-hzz2q` `tcr-v5-92000-20260915-t2b-v2` → [t2b-v2.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t2b-v2.yml)
  - `t-20260915153130-kkwcf` `tcr-v5-92000-20260915-t3broad-v2` → [t3broad-v2.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t3broad-v2.yml)
  - `t-20260915153135-dcfgj` `tcr-v5-92000-20260915-t3deep-v2` → [t3deep-v2.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t3deep-v2.yml)
  - `t-20260915153143-t6f4z` `tcr-v5-92000-20260915-t4-benchmark14` → [t4-benchmark14.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t4-benchmark14.yml)
  - `t-20260915153147-n87nx` `tcr-v5-92000-20260915-t4-held20` → [t4-held20.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t4-held20.yml)
- 提交前 CPU 回归 108 passed（pairing / sampling / AB probes / T1–T4 协议测试）。TCR 侧为复用 49000 生成器，增加 `TCR_EVAL_*` pin；本轮门禁是 CPU 协议指纹 + 权重 SHA/step，不是新的 GPU batch-consistency gate。未把本次提交写成成绩，RESULTS 数字待作业 Success 且产物验收后再填。

## 2026-09-15 Split 92000 matrix: local A100 + preemptible cluster

- 操作：cancel + submit。用户决定：本机可跑的探针/表征走本地 A100；light-chain pairing 及同类生成/绑定作业提交集群，并且改成闲时。训练 `t-20260912150714-lwf28` 与 BERT 闲时 `t-20260912035657-dd8v9` 未动。49000 对照数字未覆盖。
- **保留未 cancel**：`t-20260915153001-lpxs5` pairing p0/CFG0 已 **Success**（End=`2026-09-15T10:10:55Z`，产物 [holdout500_metrics.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/pair-p0-cfg0/holdout500_metrics.json)）；`t-20260915153015-g4999` pairing p0/CFG1 仍 **Running**（`Preemptible: false`）。
- **cancel** 18 条排队中的原非闲时作业，终态均为 **Killed**（`t-20260915153111-4q55g` / `t-20260915153116-t29vf` 首次 cancel 后仍短暂 Queue，二次 cancel 后 Killed）。机器可读：[92000_cancel_report.json](/tmp/92000_cancel_report.json)。
  - pairing：`t-20260915153020-qhkn9`、`t-20260915153025-7dc7n`、`t-20260915153030-j87kk`、`t-20260915153034-9vsmd`
  - 改本机：`t-20260915153040-km76g`（gdp_a1）、`t-20260915153045-jj6fj`（m396）、`t-20260915153121-hp4j9`（t2a）、`t-20260915153125-hzz2q`（t2b）、`t-20260915153130-kkwcf`（t3broad）、`t-20260915153135-dcfgj`（t3deep）
  - 改闲时：`t-20260915153050-stwdn`、`t-20260915153055-tfnfz`、`t-20260915153101-dn5qm`、`t-20260915153106-d4ccf`、`t-20260915153111-4q55g`、`t-20260915153116-t29vf`、`t-20260915153143-t6f4z`、`t-20260915153147-n87nx`
- **submit 闲时** 12 条：同一 YAML 已改为 `Preemptible: true`，队列仍 `c20250601` / 单卡 `ml.pni2.3xlarge`。索引：[spot_resubmissions.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/spot_resubmissions.json)。10:25:24Z 复核：
  - `t-20260915182201-75kdf` pair-p0-cfg1p5 → [pair-p0-cfg1p5.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p0-cfg1p5.yml) **Running**
  - `t-20260915182207-5n2mq` pair-p3-cfg0 → [pair-p3-cfg0.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p3-cfg0.yml) **Running**
  - `t-20260915182211-btc4k` pair-p3-cfg1 → [pair-p3-cfg1.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p3-cfg1.yml) **Running**
  - `t-20260915182216-6tdqs` pair-p3-cfg1p5 → [pair-p3-cfg1p5.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p3-cfg1p5.yml) Queue
  - `t-20260915182220-ts2bg` cdr-kong → [cdr-kong.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_cdr-kong.yml) Queue
  - `t-20260915182225-xlj8j` cdr-sab23 → [cdr-sab23.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_cdr-sab23.yml) Queue
  - `t-20260915182229-4xs47` t1-cdr3b → [t1-cdr3b.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t1-cdr3b.yml) Queue
  - `t-20260915182235-xtcbg` t1-cdr3ab → [t1-cdr3ab.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t1-cdr3ab.yml) Queue
  - `t-20260915182240-fbf9c` t1-others_cdr3b → [t1-others_cdr3b.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t1-others_cdr3b.yml) Queue
  - `t-20260915182246-dg74f` t1-others_longab → [t1-others_longab.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t1-others_longab.yml) Queue
  - `t-20260915182250-457bk` t4-benchmark14 → [t4-benchmark14.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t4-benchmark14.yml) Queue
  - `t-20260915182255-ptmzj` t4-held20 → [t4-held20.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t4-held20.yml) Queue
- **本机串行**（不进 Active 表）：脚本 [run_local_serial.sh](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/run_local_serial.sh)，日志 [local_serial.log](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/local_serial.log)，宿主 `di-20260121220916-9rvmj` / NVIDIA A100-SXM4-80GB，pid `1873356`。启动前 CPU 协议测试 43 passed。10:25Z：gdp_a1 已写出 [metrics.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/gdp_a1/metrics.json)；t2a 已写出曲线；t2b 正在跑（pid `1875713`）；其后 t3broad → t3deep → m396。未把未验收 92000 数字写入 RESULTS 主行。

## 2026-09-15 Submitted 92000 Specificity 100 epoch

- submit：`t-20260915193857-zmtc6`，TaskName=`ab-v5-92000-specificity-ep100-20260915`；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_specificity_ep100_20260915.yml)。11:38:58Z提交，11:39:15Z为Staging；c20250601单卡ml.pni2.3xlarge，Preemptible:false。14项CPU测试及bash语法检查通过。本地GPU已有占用，未改动其他任务。
- 冻结92000骨干，独立100轮五折分类头；复制并严格验证200轮特征缓存，不从旧头续训。后台会话`ab92000-specificity-ep100-monitor`复用完整验收发布器，并接入100轮ID；完成后更新RESULTS及本文件终态。产物：[100轮目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_ep100_20260915)。

## 2026-09-15 Resubmitted preempted 92000 missing jobs as spot

- 操作：submit。用户要求重跑缺少项。未 cancel 仍 Running 的 pairing p0/CFG1 `t-20260915153015-g4999`，未动训练。已完成项不重提。
- pairing 保留各目录 `holdout500_n8.progress.pt`，重提后从中断 batch 续跑。T1 cdr3b 保留 `_feature_cache/ours_tcr_v5_92000_20260915_cdr3b/cdr3b/AS` 部分特征（约 45%）。T4 不能续写：已将不完整 `setting_B` 目录挪开（benchmark14 8 条、held20 1 条），新鲜生成。
- 全部 `c20250601` / 单卡 `ml.pni2.3xlarge` / **Preemptible: true**。索引：[spot_resubmissions_missing_20260915.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/spot_resubmissions_missing_20260915.json)。
  - `t-20260915202050-8z5v8` pair-p0-cfg1p5 → [pair-p0-cfg1p5.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p0-cfg1p5.yml) Staging
  - `t-20260915202053-q6shs` pair-p3-cfg0 → [pair-p3-cfg0.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p3-cfg0.yml) Staging
  - `t-20260915202057-qnsf2` pair-p3-cfg1 → [pair-p3-cfg1.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p3-cfg1.yml) Queue
  - `t-20260915202101-2j6j5` pair-p3-cfg1p5 → [pair-p3-cfg1p5.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p3-cfg1p5.yml) Queue
  - `t-20260915202105-qr58s` t1-cdr3b → [t1-cdr3b.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t1-cdr3b.yml) Queue
  - `t-20260915202109-hkqvx` t4-benchmark14 → [t4-benchmark14.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t4-benchmark14.yml) Queue
  - `t-20260915202112-qlvtx` t4-held20 → [t4-held20.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_92000_20260915_t4-held20.yml) Queue
- 未把本次提交写成 RESULTS 成绩。

## 2026-09-15 Move queued 92000 non-pairing jobs to local A100

- 操作：用户要求 pairing 以外的 92000 queue 改本机，不必再排队。未改 YAML / 协议 / 权重 / 数据。训练 `t-20260912150714-lwf28` 与 BERT 闲时 `t-20260912035657-dd8v9` 未动。未把未验收 92000 数字写入 RESULTS。
- **现场**：补提的 7 条闲时作业均已是 **Killed**（约 12:34–12:47Z 抢占），无需再 cancel。pairing p0/CFG1 `t-20260915153015-g4999` 已 **Success**（End=`2026-09-15T13:28:29Z`）。CDR 两项与 T1 三轨仍为既有 Success。本机第一批 `run_local_serial.sh` 已完成，GPU 空余约 79 GiB（`occupy.py` 占位未动）。
- **pairing 留集群**：同一冻结 YAML 重提闲时，续 `holdout500_n8.progress.pt`。索引：[pairing_resubmit_local_move_20260915.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/pairing_resubmit_local_move_20260915.json)。
  - `t-20260915213936-6rr6g` pair-p0-cfg1p5 → [pair-p0-cfg1p5.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p0-cfg1p5.yml) Queue
  - `t-20260915213939-g5jf2` pair-p3-cfg0 → [pair-p3-cfg0.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p3-cfg0.yml) Queue
  - `t-20260915213943-gnb8k` pair-p3-cfg1 → [pair-p3-cfg1.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p3-cfg1.yml) Queue
  - `t-20260915213947-s9jc7` pair-p3-cfg1p5 → [pair-p3-cfg1p5.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p3-cfg1p5.yml) Queue
- **本机串行**（不进 Active 表）：脚本 [run_local_remaining.sh](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_92000_20260915/run_local_remaining.sh)，宿主 `di-20260121220916-9rvmj` / A100-80GB，pid `2001280`。顺序：冻结 YAML 的 T1 cdr3b → T4 benchmark14 → T4 held20。T1 从 `217600/462883` 续特征缓存。T4 不能续写：再次挪开不完整 `setting_B`（空 `designs.jsonl`，后缀 `.preempted_20260915T133907Z`）。启动前 CPU 协议测试已过。未覆盖 RESULTS。

## 2026-09-15 Resubmitted remaining 92000 pairing after second spot kill

- 操作：submit。用户要求继续提交被抢占的 pairing。未改 YAML / 协议 / 权重。训练未动。未把未验收 92000 数字写入 RESULTS。
- 上一轮 `6rr6g` / `g5jf2` / `gnb8k` / `s9jc7` 均已 **Killed**，无在途重复。本机 remaining 已于 `2026-09-15T15:43:41Z` 写完 `local_remaining.done`（T1 cdr3b、T4 两项均已出产物，待验收）。
- 同一冻结 YAML 重提闲时，续各目录 `holdout500_n8.progress.pt`。索引：[pairing_resubmit_20260915T1554Z.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/pairing_resubmit_20260915T1554Z.json)。
  - `t-20260915235443-7sb5d` pair-p0-cfg1p5 → [pair-p0-cfg1p5.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p0-cfg1p5.yml) Staging
  - `t-20260915235447-qcz5f` pair-p3-cfg0 → [pair-p3-cfg0.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p3-cfg0.yml) Staging
  - `t-20260915235450-8dvfl` pair-p3-cfg1 → [pair-p3-cfg1.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p3-cfg1.yml) Queue
  - `t-20260915235454-6dmgm` pair-p3-cfg1p5 → [pair-p3-cfg1p5.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_full_20260915_pair-p3-cfg1p5.yml) Queue

## 2026-09-16 Submitted pairing ESMC-feedback v5 p0

- 操作：submit；`task_id=t-20260916131024-kdcz5`，TaskName=`ab-v5-49000-encoder-sync-20260916-p0`；YAML：[p0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_encoder_sync_20260916_p0.yml)。`c20250601` 单卡，`Preemptible:false`；05:10:51Z 首次查询 **Running**，Start=`2026-09-16T05:10:24Z`。
- 固定独立 49000 权重、holdout500×8、p0、CFG0；先真实模型逐步输入门禁，再 iter124 全量推理及评分。新目录 `output/downstream_generation/ab_v5_49000_pairing_encoder_sync_20260916/pair-p0-cfg0`（仓库绝对根见 YAML）；旧产物不覆盖，不取消其他任务。
- CPU：pllm 生成 28 passed + 训练 corruption 3 passed；实际执行环境 protenix_abtcr 生成／AB probe 三套共 38 passed。真实门禁和完整分数以作业实际产物为准。

## 2026-09-16 Submitted pairing ESMC-feedback v5 p3

- 操作：submit；`task_id=t-20260916131128-ht8s4`，TaskName=`ab-v5-49000-encoder-sync-20260916-p3`；YAML：[p3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_encoder_sync_20260916_p3.yml)。`c20250601` 单卡，`Preemptible:false`；05:12:10Z 首次查询 **Running**，Start=`2026-09-16T05:11:29Z`。
- 与上述 p0 固定相同权重／数据／采样预算，仅允许前三轻链残基；流程为逐步输入门禁 → full500×8 → 原有评分。独立输出：[p3 run](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_pairing_encoder_sync_20260916/pair-p3-cfg0)。不取消其他任务，旧产物保留。
- p0 门禁已通过：[gate](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_pairing_encoder_sync_20260916/pair-p0-cfg0/encoder_feedback_gate.json)；实际 A100、49000 权重及输入 CSV SHA 与修复前一致；4 步 ESMC 可见生成残基递增，隐藏参考扰动不影响两路输入或生成结果。p0 全量已完成首批，约 33 秒／2 heavies，估计生成约 2.3 小时（非完整评分完成时间）。

### 2026-09-16T05:14:53Z 两组真实门禁与启动验收

- p0 和 [p3 gate](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_pairing_encoder_sync_20260916/pair-p3-cfg0/encoder_feedback_gate.json) 均通过；各4步／4候选，两路状态一致、隐藏参考扰动不影响输入或生成。与各自旧实验相比，权重 SHA、数据 SHA 及所有采样配置一致；gate／full manifest／当前磁盘九份源码 SHA 全部一致。
- 实际 full500 已分别生成 12 条 heavy／96 候选（p0）和 8 条 heavy／64 候选（p3）；平台二者均 Running，约34秒／batch。尚无完整指标，不将4步门禁输出当作性能结果。后续自动按既有脚本完成生成和评分；原始日志／progress／manifest 在各自目录，终态再验收 metrics 并更新主表，未启动其他新评测。
- `git diff --check` 通过；本轮未取消任何已有任务，未修改评测快照或历史 CSV／metrics。

## 2026-09-16 Pairing ESMC-feedback p0/p3 终态验收

- 17:25Z 平台只读查询均 **Success**：p0 `t-20260916131024-kdcz5`，Start=`2026-09-16T05:10:24Z`，End=`2026-09-16T07:51:37Z`，Elapsed=9,673秒；p3 `t-20260916131128-ht8s4`，Start=`2026-09-16T05:11:29Z`，End=`2026-09-16T07:53:11Z`，Elapsed=9,702秒。两项均约2小时42分（含生成与评分），从Active表移除，其他任务未操作或重查。
- 两组各500×8预测与全部评分记录完整，协议／身份／分母验收见 [AB pairing §7 k](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。RESULTS仅更新p3与论文主对照，保留修复前源产物，不恢复CFG消融表。
- 本轮没有重新采样或评分，没有模型／数据／训练代码变更，也没有submit/cancel；只读验收后同步结果、过程及协议状态。

## 2026-09-16 Submitted no-prefix pairing iteration sweep

- 操作submit：iter8，`t-20260917014444-tgctn`，TaskName=`ab-v5-49000-pair-iters-p0-20260916-s8`；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_pairing_iters_p0_20260916_iter8.yml)。c20250601／单卡／Preemptible:false；17:44:56Z查询Staging，Start=`2026-09-16T17:44:44Z`。完整500×8，新独立目录，不取消其他任务。
- 操作submit：iter16，`t-20260917014624-xrz96`，TaskName=`ab-v5-49000-pair-iters-p0-20260916-s16`；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_pairing_iters_p0_20260916_iter16.yml)。c20250601／单卡／Preemptible:false；2026-09-16T17:46:26Z查询Queue，Start=`2026-09-16T17:46:24Z`。完整500×8，新独立目录，不取消其他任务。
- 操作submit：iter32，`t-20260917014628-p7tqj`，TaskName=`ab-v5-49000-pair-iters-p0-20260916-s32`；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_pairing_iters_p0_20260916_iter32.yml)。c20250601／单卡／Preemptible:false；2026-09-16T17:46:30Z查询Waiting，Start=`2026-09-16T17:46:28Z`。完整500×8，新独立目录，不取消其他任务。
- 操作submit：iter64，`t-20260917014632-cv9z4`，TaskName=`ab-v5-49000-pair-iters-p0-20260916-s64`；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_pairing_iters_p0_20260916_iter64.yml)。c20250601／单卡／Preemptible:false；2026-09-16T17:46:35Z查询Queue，Start=`2026-09-16T17:46:32Z`。完整500×8，新独立目录，不取消其他任务。
- 操作submit：iter96，`t-20260917014636-s2shg`，TaskName=`ab-v5-49000-pair-iters-p0-20260916-s96`；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_pairing_iters_p0_20260916_iter96.yml)。c20250601／单卡／Preemptible:false；2026-09-16T17:46:38Z查询Queue，Start=`2026-09-16T17:46:36Z`。完整500×8，新独立目录，不取消其他任务。
- 操作submit：iter128，`t-20260917014639-bpp64`，TaskName=`ab-v5-49000-pair-iters-p0-20260916-s128`；[YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_pairing_iters_p0_20260916_iter128.yml)。c20250601／单卡／Preemptible:false；2026-09-16T17:46:42Z查询Queue，Start=`2026-09-16T17:46:40Z`。完整500×8，新独立目录，不取消其他任务。
<!-- pairing-step-submit-log:end -->

### 2026-09-16T17:47:11Z 步数扫描启动核对

- 六项全部查询确认，8/16/32 Running，64/96/128 Queue。iter8已通过真实反馈gate并生成30/250 batches，其余Running项处于启动／gate阶段。完整IM尚未产生；所有档均完整500×8，没有减样本或跳过评分。
- 冻结manifest、旧124步参照验收和45项CPU检查通过；各档将自动生成与评分并更新汇总JSON，未配置测试集best-step替换主行。没有取消、抢占或修改原有训练／评测任务。

## 2026-09-17 Submitted Ours v5 140000 full AB/TCR matrix

- 操作：submit。用户要求对当前完整保存点 140000 按 AB/TCR 审计清单逐项测试；49000/92000 结果仅作对照，不覆盖。锁定独立快照 [ab_v5_140000_llada_20260917](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_140000_llada_20260917)，权重 SHA256=`a482afa425ef4f058d88e38d316aca096341af471b504efe948434d3ebc2dfe2`，与训练目录 `checkpoint-140000` 重算一致；`trainer_state.global_step=140000`。未复制优化器，未改训练任务 `t-20260912150714-lwf28`。
- 协议变更：pairing **仅 p0 / CFG=0**（reference length, iter=124, gumbel_argmax, seed=42, 500×8）；**不跑 p3，不跑 p0/p3×CFG 六格**。CDR Kong + SAb23H2 为 **argmax / max-iter=2**（只改 140000 YAML，未改 49000/92000 历史产物或 prepare 默认值）。其余任务同 92000/49000 审计矩阵。Specificity 无既有 140000 完整产物，按标准 100-epoch native 协议在本机跑，不跳过。
- 提交前 CPU 113 passed（pairing / sampling / AB probes / T1–T4）。TCR 侧复用 49000 生成器 + `TCR_EVAL_*` pin；本轮门禁是 CPU 协议指纹 + 权重 SHA/step（[write_tcr_cpu_identity_gates.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/write_tcr_cpu_identity_gates.py)），不是新的 GPU batch-consistency gate。
- **集群闲时** 9 项，`c20250601` / `ml.pni2.3xlarge` / 1 replica / 1 GPU / **Preemptible: true**。17:58Z 复核见下。机器可读索引：[all_140000_matrix_submissions.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/all_140000_matrix_submissions.json)。
  - `t-20260917015632-hx44v` `ab-v5-140000-full-20260917-pair-p0-cfg0` → [pair-p0-cfg0.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_140000_full_20260917_pair-p0-cfg0.yml) **Running**
  - `t-20260917015637-5k6vp` `ab-v5-140000-full-20260917-cdr-kong` → [cdr-kong.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_140000_full_20260917_cdr-kong.yml) **Running**
  - `t-20260917015642-9t4rd` `ab-v5-140000-full-20260917-cdr-sab23` → [cdr-sab23.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_140000_full_20260917_cdr-sab23.yml) **Running**
  - `t-20260917015648-qmhgs` `tcr-v5-140000-20260917-t1-cdr3b` → [t1-cdr3b.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_140000_20260917_t1-cdr3b.yml) **Running**
  - `t-20260917015653-492rz` `tcr-v5-140000-20260917-t1-cdr3ab` → [t1-cdr3ab.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_140000_20260917_t1-cdr3ab.yml) **Running**
  - `t-20260917015658-blclc` `tcr-v5-140000-20260917-t1-others-cdr3b` → [t1-others_cdr3b.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_140000_20260917_t1-others_cdr3b.yml) **Running**
  - `t-20260917015703-xnnpp` `tcr-v5-140000-20260917-t1-others-longab` → [t1-others_longab.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_140000_20260917_t1-others_longab.yml) **Queue**
  - `t-20260917015708-rsccg` `tcr-v5-140000-20260917-t4-benchmark14` → [t4-benchmark14.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_140000_20260917_t4-benchmark14.yml) **Queue**
  - `t-20260917015713-rk77d` `tcr-v5-140000-20260917-t4-held20` → [t4-held20.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_v5_140000_20260917_t4-held20.yml) **Queue**
- **本机串行**（不进 Active 表）：脚本 [run_local_serial.sh](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/run_local_serial.sh)，日志 [local_serial.log](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/local_serial.log)，宿主 `di-20260121220916-9rvmj` / NVIDIA A100-SXM4-80GB，pid `2780601`。顺序：Specificity 100-epoch → GDPa1 → T2A → T2B → T3-broad → T3-deep → m396。17:57:37Z 已 start specificity。未把未验收 140000 数字写入 RESULTS 主行。
- 产物根：AB [ab_v5_140000_full_20260917](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917)；TCR [tcr_v5_140000_20260917](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_140000_20260917)。prepare 生成的 p0-cfg1/p3 网格 YAML 已标 DO NOT SUBMIT，未提交。

## 2026-09-17 Pairing p0 iteration sweep 终态核对

- 本轮只读查询六项均Success（队列c20250601、独立单卡、Preemptible:false），逐项结束UTC：
  - iter8 `t-20260917014444-tgctn`，End=`2026-09-16T18:15:35Z`。
  - iter16 `t-20260917014624-xrz96`，End=`2026-09-16T18:26:52Z`。
  - iter32 `t-20260917014628-p7tqj`，End=`2026-09-16T18:54:17Z`。
  - iter64 `t-20260917014632-cv9z4`，End=`2026-09-16T19:49:52Z`。
  - iter96 `t-20260917014636-s2shg`，End=`2026-09-16T20:37:52Z`。
  - iter128 `t-20260917014639-bpp64`，End=`2026-09-16T21:40:00Z`。
- 全部24,000条新增候选及原124步参照验收通过，完整输入／统计边界见[AB pairing §4.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)，数值只维护于RESULTS独立诊断表。未替换p3主结果或140000矩阵；六项从Active表移除，旧提交快照保留，新增[终态快照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_pairing_iters_p0_20260916/terminal_status_20260917.json)。
- 没有submit/cancel、重新推理／评分或权重／生产代码修改。本轮检查不等于重新运行ImmunoMatch／ANARCI，也未查其他平台任务。

## 2026-09-17 Stopped: v5 157000 weights already pruned

- 操作：无 submit / 无 cancel。用户选定 **157000** 全矩阵（pairing p0 / CFG=0 / **max-iter=16**；CDR iter=2）。源 `checkpoint-157000` 已不在；无独立评测快照；未用 158000 顶替。
- 训练 `t-20260912150714-lwf28` 未 cancel。2026-09-17 07:12:46 日志：`Pruned checkpoint .../checkpoint-157000`；latest 为 `checkpoint-158000`（`global_step=158000`）。
- 无本机 pid、无新 Volc task id、无 submissions JSON、RESULTS 主行未改。

## 2026-09-17 Cancelled v5 8-GPU training

- 操作：cancel
- `task_id=t-20260912150714-lwf28`，任务名 `protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601`
- YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_esmc_llada270m_diffusion_immune_v5_8gpu_c20250601.yml`
- 最终状态：**Killed**，Start=`2026-09-12T07:07:14Z`，End=`2026-09-17T08:13:43Z`
- `Preemptible: false`（c20250601 非闲时，1× `ml.pni2.28xlarge`）
- 未 cancel 评测任务、FABind、无关队列。BERT 闲时 `t-20260912035657-dd8v9` 仍为 **Queue**，未 cancel。
- 盘上仍保留 `checkpoint-37000` / `43000` / `92000` / **`159000`**（完整 resumable，`trainer_state.global_step=159000`）。**`checkpoint-158000` 已不在**（top-k 在写入 159000 时 prune）。未做 SHA 快照，未启动评测或 158000 下游矩阵。

## 2026-09-17T12:21:56Z 平台与本机只读核对

- 未 submit / cancel / kill，也未启动评测。按本项目任务名前缀查询非终态列表，并核对历史 Active 表 ID；未发现 AB/TCR v5 或 159000 活跃评测。本机没有训练、特征抽取、生成或评分进程。查询范围是当前 CLI 可见任务和当前开发机，不代表所有其他主机。
- BERT 闲时 `t-20260917174134-kxmkk` 为 Queue，09:41:35Z 创建。监控日志记录旧 `t-20260912035657-dd8v9` Killed 后，09:41:37Z 自动重提；不是本轮操作，不是 159000 评测。YAML：[BERT v5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_esmc_llada270m_bert_immune_v5_8gpu_spot.yml)。
- 本机 `monitor_spot_tasks.py` PID 3616013、父包装循环 PID 3616011 仍存活；BERT 没有 STOP 哨兵，取消云任务后仍可能再次自动重提。未修改看护或 STOP 文件。证据：[监控日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/_monitor/monitor.log)、[状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/_monitor/state.json)。
- 本机 `python occupy.py` PID 13069 仍运行，cwd 为 flow_antibody_tcr；代码为 GPU 循环矩阵计算而非训练/评测。A100 80GB 查询时显存约 1693 MiB，GPU 利用率约 61%；驱动返回的 GPU PID 不在当前进程命名空间，未据此作精确 PID 映射。tmux `tcrvalid-adc` 当前 pane 为 bash，没有对应评测子进程。
- 从 Active 表移出以下已核实的历史终态（不是本轮终止任务）：
  - `t-20260911190334-kpfsz`：Success，End=`2026-09-11T14:53:27Z`。
  - `t-20260907044259-qd52v`：Killed，End=`2026-09-06T21:24:58Z`。
  - `t-20260907044303-ks4x6`：Killed，End=`2026-09-06T21:24:59Z`。
  - `t-20260907044307-btjxr`：Success，End=`2026-09-07T07:10:02Z`。
  - `t-20260907044310-wlm9h`：Killed，End=`2026-09-06T21:24:59Z`。
  - `t-20260907044314-mtzn9`：Killed，End=`2026-09-06T21:25:00Z`。
  - `t-20260907044317-kfhk2`：Killed，End=`2026-09-06T21:25:00Z`。
  - `t-20260907044321-ht8b6`：Success，End=`2026-09-07T07:12:21Z`。
  - `t-20260907044324-tcmhs`：Killed，End=`2026-09-06T21:25:01Z`。
  - `t-20260907044328-wwfjn`：Killed，End=`2026-09-06T21:25:01Z`。
  - `t-20260906233442-d2gzt`：Killed，End=`2026-09-06T23:20:08Z`。
  - `t-20260907001930-qtf2s`：Killed，End=`2026-09-06T23:20:19Z`。
  - `t-20260906233451-fjzmm`：Killed，End=`2026-09-06T23:20:18Z`。
  - `t-20260906233455-djlmk`：Killed，End=`2026-09-06T23:20:19Z`。
  - `t-20260906233500-9f7vp`：Killed，End=`2026-09-06T23:20:20Z`。
  - `t-20260906233504-x2n8b`：Killed，End=`2026-09-06T23:20:22Z`。
  - `t-20260906233508-svn2t`：Killed，End=`2026-09-06T23:20:20Z`。
  - `t-20260906233513-mpfv5`：Killed，End=`2026-09-06T23:20:22Z`。
  - `t-20260906233517-9x8kq`：Killed，End=`2026-09-06T23:20:22Z`。

## 2026-09-17 159000 AB/TCR 执行完成核对

- 2026-09-17T14:51:05Z 只读检查：AB 六项云端与 TCR 六项云端均 Success；本机 T2A/T2B/T3-broad/T3-deep 四项均 run_manifest.status=success、global_step=159000，local_repr.done 存在，无残留本地评测进程。总计 16/16 执行完成，零 Failed；不等于完整独立结果审计完成。
- 本地四项 12:55:57Z 至 13:06:46Z，约 10分49秒；全部任务最晚于 13:46:38Z 完成（T1 cdr3b）。AB 四份指标 JSON 与九份 CDR 聚合日志、TCR 四份 Binding summary、T2/T3 指标与两份 T4 metrics 均已落盘。本轮未重跑模型或评分、未 submit/cancel。
- Pairing 仍为用户指定16步；4,000候选／500重链的聚合指标可读。具体数字先以原始 metrics 为准，未完成逐预测／折／split核对前不自动更新 RESULTS。历史 submission JSON 与之前状态记录是当时快照，最新平台终态见 [terminal_status_20260917.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_159000_20260917/terminal_status_20260917.json)。
- 已将以下12项终态从 Active Evaluation 表移出（不是本轮停止任务）：
  - `t-20260917204150-2kzlj` ab-v5-159000-full-20260917-m396：Success，End=`2026-09-17T12:57:06Z`。
  - `t-20260917204147-ztdwf` ab-v5-159000-full-20260917-gdp-a1：Success，End=`2026-09-17T12:43:17Z`。
  - `t-20260917204143-6b7vr` ab-v5-159000-full-20260917-specificity：Success，End=`2026-09-17T12:47:24Z`。
  - `t-20260917204140-gqlsz` ab-v5-159000-full-20260917-cdr-sab23：Success，End=`2026-09-17T12:44:49Z`。
  - `t-20260917204115-pj2rk` ab-v5-159000-full-20260917-pair-p0-cfg0：Success，End=`2026-09-17T13:21:05Z`。
  - `t-20260917203938-l9f72` ab-v5-159000-full-20260917-cdr-kong：Success，End=`2026-09-17T12:53:31Z`。
  - `t-20260917205616-jwmqq` tcr-v5-159000-20260917-t4-held20：Success，End=`2026-09-17T13:20:57Z`。
  - `t-20260917205613-lr9cf` tcr-v5-159000-20260917-t4-benchmark14：Success，End=`2026-09-17T13:13:33Z`。
  - `t-20260917205541-gnj99` tcr-v5-159000-20260917-t1-others-longab：Success，End=`2026-09-17T12:58:49Z`。
  - `t-20260917205538-r2ldn` tcr-v5-159000-20260917-t1-others-cdr3b：Success，End=`2026-09-17T12:58:51Z`。
  - `t-20260917205534-878fc` tcr-v5-159000-20260917-t1-cdr3ab：Success，End=`2026-09-17T12:58:47Z`。
  - `t-20260917205531-55dxp` tcr-v5-159000-20260917-t1-cdr3b：Success，End=`2026-09-17T13:46:38Z`。
