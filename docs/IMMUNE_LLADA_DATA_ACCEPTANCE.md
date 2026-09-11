# Immune LLaDA 数据重构验收记录

> ⚠️ **v3-era acceptance archive（2026-09-09）。** 这不是当前 v4/v5 合同。
> 当前 prepared 入口：[`dllm/pipelines/immune_llada/README.md`](../dllm/pipelines/immune_llada/README.md)。
> 本文保留 51 条 homotypic-pair 证据、`quality.homotypic_pair` 决策、以及
> `immune_v3` / `immune_v3_heterotypic` 的复跑命令——那些目录仍在盘上。

更新时间：2026-09-12（仅加 banner）。设计计划：`docs/IMMUNE_LLADA_DATA_REFACTOR_PLAN.md`。

## 当前结论

**主要迁移已完成；尚未通过全部验收。** 训练入口已经不再解析原始 CSV、读取 blocklist、执行 filter 或重试替换样本；grammar、动态 padding、per-chain encoder 重建、diffusion/MLM mask 按要求保留。未生成 model-ready token cache。

尚不能声称训练数据语义完全不变，也不能声称数据加载不会阻塞 GPU。

## 1. 全量过滤与指纹多重集检查

维护入口：`scripts/data/run_immune_full_parity.py`。

- 历史实现固定为 commit `7f6f2351702dc307f710bce7228a53eae391b26f`，从 Git blob 隔离加载，避免旧模块 alias 指向新实现而发生自比较。
- 比较逐行 keep/drop、旧训练语义与新语义、旧语义与 prepared 多重集、新完整 canonical 行与 prepared 多重集。外部排序保留每个指纹的重复次数，不仅比较集合或总数。
- 固定审计时的配置、八个 blocklist、当前 pipeline 代码身份；扫描记录 raw/prepared SHA256，并检查文件在扫描时是否变化。

| 项目 | 结果 |
|---|---:|
| 七源 train/valid raw | 8,671,293 |
| 旧/新/prepared 保留 | 7,998,022 |
| filter drops | 673,271 |
| 逐行过滤决策差异 | 0 |
| 数量差异 | 0 |
| 新完整 canonical vs prepared 多重集差异 | 0 |
| 旧/新语义不同的行 | **51** |
| 旧语义 vs prepared 不同指纹 | **102** |
| 全量耗时 | 1,368.99 秒（约 22 分 49 秒） |
| 最终状态/退出码 | **failed / 2** |

证据：`refactor_baseline/full_parity_20260909/report.json`、`status.json`、`exit.json`。14 个 source/split 组合中，train OAS、train OTS 失败，其余 12 个通过。

前缀 smoke：`refactor_baseline/full_parity_smoke_20260909/report.json`，14,000 raw / 10,313 kept，通过。但前缀与随机抽样均不能替代全量检查。

### 差异全部归因

`refactor_baseline/full_parity_diagnosis_20260909/run/` 保存全部 51 条记录和定向 grammar/collator 对比。诊断额外扫描 train OAS/OTS，读取 4,589,157 raw 行，耗时 706.62 秒；raw SHA256 与全量报告相同。

| 来源 | 行数 | 原始标注 | 旧训练角色 | 新角色 | 定向 grammar 结果 |
|---|---:|---|---|---|---|
| OAS | 38 | `l_locus=H` | H/L | H/H | 38 条均相同，仍为 `antibody_pair` |
| OTS | 13 | 两条链均为 B | β/α | β/β | **13 条均从 `tcr_pair` 变成 `tcr_single`** |

全部差异仅为 `chain_roles`；序列、task、source、labels、weight 一致。OAS 中 37 条两条序列相同、1 条不同；OTS 中 7 条相同、6 条不同。因此不能把全部差异简单归因于完全重复的两条序列。

- 全量报告的 1,792 条 reservoir grammar 样本未命中这些差异，抽样 mismatch=0 不代表所有 grammar 相同。
- 定向检查 51 条，38 条 OAS rendering 相同，13 条 OTS rendering 不同；按每批 4 条组成的 13 个 batch 中，4 个 collator batch 不同，涉及 encoder 输入、token、mask 等。
- 诊断报告中的 `status=passed` **仅表示已复现预期差异数量，不是 parity 通过**。
- 定向对比使用内建 ESM2 residue vocab，不是全量实际 LLaDA tokenizer/remap/模型等价性证明。

**用户已确认：直接离线丢弃这 51 条，不修正配对、不兼容旧位置角色。** 已在 OAS/OTS 的原有过滤之后追加 `quality.homotypic_pair`，拒绝两条链角色相同的 paired 记录；不按序列是否相同判断，不影响其他 source 的合法单链数据，也不向训练阶段增加操作。新 manifest 会记录该规则。

新数据已完成重建并通过策略验收：`data/prepared/immune_v3_heterotypic/`，总保留 **7,997,971** 条、过滤 **673,322** 条；train **7,794,324**，valid **203,647**。旧 `immune_v3` 保留作回滚/对照。训练入口和 profiler 默认目录已切换到新版本。

后台重建/复验已通过：`refactor_baseline/homotypic_drop_20260909/policy_acceptance.json`。流程确认只发生批准的 OAS train 38 / OTS train 13 条排除；strict historical parity 的退出码 2 是预期的旧数据差异，不代表新策略验收失败。

### 可复跑命令

从仓库根目录运行，使用具备项目依赖的 Python（本地使用 `protenix_abtcr` 环境）。工作目录必须全新，防止覆盖验收证据：

```sh
python -B -u scripts/data/run_immune_full_parity.py \
  --max-rows 1000 --sample-limit 32 \
  --work-dir refactor_baseline/full_parity_smoke_rerun \
  --background --max-seconds 600

python -B -u scripts/data/run_immune_full_parity.py \
  --work-dir refactor_baseline/full_parity_rerun \
  --background --max-seconds 21600
```

检查输出目录的 `launch.json`、`worker.json`、`run.log`、`status.json`、`exit.json` 和 `report.json`。后台启动成功不是验收通过。

## 2. 本地真实模型 DataLoader profiling

维护入口：`scripts/debug/profile_immune_llada.py`。详细交接：`refactor_baseline/local_profile_20260909/HANDOFF.md`。

已实现：

- 复用真实训练入口的 tokenizer、grammar/remap/collator、ESMC+LLaDA 初始化和 `FusionTrainer.compute_loss()`。
- scratch decoder d768/L8/h12/**MLP3072**，冻结 ESMC-300M，BF16、decoder gradient checkpointing。
- 相同 semantic 样本、batch plan、初始模型与 optimizer 状态，对比 workers=0/1/2/4。
- 正式诊断每组 24 warmup + 96 measured steps；计时包含真实 next/H2D/forward/backward/AdamW/CUDA 同步。记录初始化、next wait、吞吐及资源遥测。
- 修正显式 Accelerate autocast、按核验的 GPU UUID 选择设备、全 batch tensor digest、context-only GPU artifact 命名。
- 有 GPU/IO 门禁、deadline、PID/status/log；不提交 Volc、不保存 checkpoint、不干扰外部进程。

**实际 GPU profiler 还未运行。** GPU 检查发现外部 compute PID `3074807`（约 1,680 MiB，设备有持续计算），因此没有启动模型子进程，也没有可靠的 worker 最优配置结论。此前 A100 单步 forward/backward 与 checkpoint restore smoke 已通过，但不是本次 DataLoader profiling。

运行前必须同时满足：无全量/诊断扫描、GPU 空闲、处理策略明确。然后才创建 IO clearance 并顺序运行：

```sh
python -B scripts/debug/profile_immune_llada.py --check-gpu
# 仅确认无其他扫描后执行：
printf 'main-agent-cleared\n' > refactor_baseline/local_profile_20260909/IO_CLEAR
python -B scripts/debug/profile_immune_llada.py \
  --run --background --smoke --wait-seconds 300 --max-seconds 1800
# smoke 成功且资源仍空闲后，单独执行：
python -B scripts/debug/profile_immune_llada.py \
  --run --background --workers 0 1 2 4 \
  --warmup 24 --steps 96 --wait-seconds 300 --max-seconds 3600
```

这是固定学习率、单卡、均衡 source/length 的诊断循环，**不是完整 Trainer/FSDP 等价测试或自然语料比例吞吐**。单次固定顺序对比也不足以宣称稳健最优 worker 数。

## 3. 回归验证

加入离线同类型配对过滤后，本轮合并执行：**104 passed in 24.70s**。上一轮结果为 94 passed；本轮覆盖同类型/异类型、相同/不同序列、不影响单链 source、离线落盘和 strict audit 对已批准排除的报告。

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
  -B -m pytest -p no:cacheprovider \
  scripts/tests/immune_llada \
  scripts/tests/bioseq/test_qwen3_vl_grammar.py \
  scripts/tests/bioseq/test_dynamic_training.py \
  scripts/tests/bioseq/test_grammar_tcr_role_resolution.py \
  scripts/tests/bioseq/test_grammar_tcr_relation.py \
  scripts/tests/bioseq/test_grammar_unknown_relation.py -q
```

包含离线 preprocess/filter/prepared loader、指纹重复次数损坏检测、真实发现的 H/H 与 β/β 差异检测、grammar 兼容，以及 profiler 的 41 项 CPU 测试。数据策略已确认，测试通过表示新过滤与差异报告符合预期，**不表示新全量 prepared 数据已完成重建/验收**。

## 4. 仍未实现/未证明的边界

1. 历史 prepared manifest 没有原始输入/blocklist hash，本次审计无法证明预处理时的历史文件未变；完整 freshness/version 启动契约仍未完成。
2. prepared loader 启动仍全扫 JSONL 建 byte-offset 索引；取样仍有 JSON decode 和轻量 record validation。不再执行 filter 不等于零数据开销。
3. 15 份 v3 YAML 与 `DataArguments` 的旧 raw/blocklist/row-cap 兼容字段仍需同步收尾。
4. `bioseq/datasets.py`、`qwen3_vl_arch/data/` 仍有有效共享调用者，不应为减少文件数量直接删除。
5. 没有正式 Volc 训练、没有 FSDP profiling、没有新的模型性能数字。
