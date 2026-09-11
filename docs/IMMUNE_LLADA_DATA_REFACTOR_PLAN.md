# Immune LLaDA 数据处理重构计划

> ⚠️ **COMPLETED / historical（主实现已于 2026-09-09 落地）。** 当前操作入口是
> [`dllm/pipelines/immune_llada/README.md`](../dllm/pipelines/immune_llada/README.md)。
> 本文保留 filter / manifest / mix-dependent / freshness 的设计取舍，不要当现役 v4/v5
> 行数或「尚未实现」清单读。v3 验收数字见
> [`IMMUNE_LLADA_DATA_ACCEPTANCE.md`](IMMUNE_LLADA_DATA_ACCEPTANCE.md)。

## 1. 文档目的

本文档规划免疫蛋白数据处理代码的重构工作。目标是将当前分散在多个目录、训练入口和脚本中的数据逻辑，整理为独立的 `immune_llada` pipeline，并将数据过滤从训练时动态执行改为训练前一次性离线处理。

本次重构服务于当前的免疫蛋白训练主线：

```text
ESMC-300M encoder + LLaDA decoder
```

当前涉及的主要数据源包括：

```text
oas
ots
asd_antibody
trait
tcr_native
tcr_papers
tcr_repertoire
```

> `examples/bioseq/train_bioseq_ddp.py` 已确认退役，不再作为兼容目标，也不会继续使用。

### 1.1 实施状态（2026-09-09 核对）

重构已完成主要实现，当前进入真实数据、模型和吞吐验证阶段。当前状态：

| 计划产物/步骤 | 现状 |
|---|---|
| `dllm/pipelines/immune_llada/` | 已创建，canonical records、grammar、collator、ESM tokenizer、source registry 已实现 |
| `scripts/data/preprocess_immune_dataset.py` | 已实现，支持 CSV/JSONL、offline filter、semantic JSONL shard、manifest、报告和 dry-run |
| `configs/data/immune_v3.yaml` | 已创建并验证七个真实 source 路径和八个非空 blocklist |
| `refactor_baseline/immune_v3_baseline.json` | 已生成；包含 manifest 计数、每 source/split 首尾哈希和聚合 canonical record 哈希 |
| offline prepared dataset | 已全量生成：8,671,293 raw / 7,998,022 kept / 673,271 filter drops / 0 schema drops / 0 errors |
| `protein_pretrain_esmc.py` 训练入口切换 | 已切换为 prepared loader；全量 manifest dry-run 和 A100 1-step forward/backward/save 已通过 |
| checkpoint restore smoke | 已通过；临时 full checkpoint-1 成功恢复并继续到 checkpoint-2 |
| 全量旧/新过滤与 multiplicity parity | 已执行，22m49s；过滤/计数一致，新 canonical vs prepared 全量一致；严格旧/新语义 parity 未通过：OAS 38 + OTS 13 条链角色差异 |
| 差异定向 grammar 检查 | 51 条全部定位；用户已确认离线丢弃。新增 `quality.homotypic_pair`，104 项合并回归通过；新版本数据重建/复验单独追踪 |
| worker/GPU overlap profiling | 本地真实模型 profiler 已实现并修正 BF16 autocast/设备 UUID；41 项 CPU 测试通过。实际 GPU profiling 因外部进程占用尚未运行，不再等待正式训练资源；FSDP 不在本地单卡验收范围 |
| `examples/llada/protein_pretrain.py` | 已确认退役并删除；正式入口为 `examples/llada/protein_pretrain_esmc.py` |
| `dllm/pipelines/bioseq/datasets.py` | 已确认仅为旧免疫 raw CSV 动态 loader，active callers 已迁移并删除；共享 BioSeq/PPI/STRING/MINT 模块保留 |

新增的 sampled adapter parity 验证：

```text
refactor_baseline/immune_adapter_parity_train_5000.json
refactor_baseline/immune_adapter_parity_valid_5000.json
```

结果为：train 7 个 active source 各抽取 5,000 行，共 35,000 行；valid 共读取
32,440 行（各 source 最多 5,000 行）。两套 adapter 均保留的记录全部匹配，
`chains`、`task_type`、`source` 以及旧链路显式提供的 `roles`/`relation` 均无差异；
未出现单边 drop、adapter error 或 mismatch。该结果是 sampled parity，不替代全量
fingerprint/multiplicity parity。后续已使用历史 commit 的独立实现执行全量检查，详见下文；前缀检查没有覆盖到同类型链的边界样本。

当前已完成的 smoke 范围：

```text
source adapter + registry + filters + preprocessing: 14 passed
canonical grammar/collator compatibility + dynamic compatibility: 26 passed
合计: 40 passed
真实七源限量 dry-run: train/valid 各 source 1000 raw rows，成功
真实七源全量 preprocessing: 8,671,293 raw / 7,998,022 kept，成功
正式训练入口 full-manifest dry-run + A100 1-step forward/backward/save: 成功
full checkpoint restore + continuation: checkpoint-1 → checkpoint-2，成功
```

相关测试已覆盖真实 prepared loader、grammar/collator、per-chain encoder input reconstruction
和训练入口 dry-run；A100 1-step 及 full checkpoint restore continuation 已完成。旧的
`bioseq/datasets.py` 动态链路及旧 `protein_pretrain.py` 入口已删除；当前免疫主线使用
`immune_llada` prepared loader，PPI/STRING/MINT 共享模块与兼容 alias 保留。

**本轮新增验收证据**：

- `scripts/data/run_immune_full_parity.py` 固定旧 commit `7f6f2351702dc307f710bce7228a53eae391b26f`，隔离加载历史 adapter/filter/wrapper/grammar，避免与当前兼容 alias 自比较；按多重集比较指纹，保留重复次数。
- `refactor_baseline/full_parity_smoke_20260909/report.json`：14,000 raw / 10,313 kept 前缀 smoke 通过。
- `refactor_baseline/full_parity_20260909/report.json`：全量 8,671,293 raw / 7,998,022 kept；filter decision/count mismatch=0，canonical-vs-prepared mismatch=0；旧/新 semantic mismatch=51，旧/落盘多重集差异为 102 个指纹。退出码 **2**，不能标为全量通过。
- `refactor_baseline/full_parity_diagnosis_20260909/run/`：全查 51 条差异，均仅 `chain_roles` 不同；OAS 原始 `l_locus=H`，OTS 原始两条链均标 B。13 条 OTS 实际改变 grammar、encoder 输入和 masks。原先 1,792 条 reservoir grammar 抽样未覆盖这些差异；诊断报告的 `passed` 仅表示差异已重现，不代表 parity 通过。
- 合并回归结果：**104 passed in 24.70s**，包含 H/H、β/β 离线丢弃策略、strict audit 差异检测及 profiler 的 41 项 CPU 测试；新增/修改验收脚本的 AST 语法检查通过。
- 历史 prepared manifest 没有原始输入/blacklist hash。本轮仅钉住审计时的文件，不能反推预处理时的历史文件未变。

完整验收记录、复跑命令和阻塞项见 `docs/IMMUNE_LLADA_DATA_ACCEPTANCE.md`。本文档其余内容为设计目标和后续计划，**不代表所有条款已实现**。

### 1.2 验收前仍需完成/确认的事项

1. **已完成同类型 paired 数据离线丢弃和新数据切换**：规则 `quality.homotypic_pair` 已实现，104 项合并回归通过。新数据 `data/prepared/immune_v3_heterotypic/` 已重建并通过策略验收：7,997,971 kept；旧数据保留作回滚/对照。训练和 profiler 默认路径已切换；strict historical parity 的退出码 2 是批准排除导致的预期结果。
2. **黑名单漂移的强制机制**（§8.1）：仍未实现完整 freshness/version 启动契约。当前 manifest validation 主要检查 schema/format/splits，不证明输入和 blacklist 未漂移。不得把本次离线审计代称为启动期强校验。
3. **本地模型 profiling**：GPU 空闲且没有审计扫描时运行 smoke 和 workers=0/1/2/4 对比。现有 CPU 数字不能证明 GPU 无 data starvation，也不支持 FSDP 性能结论。
4. **旧 YAML 参数收尾**：`DataArguments` 仍保留 raw/blocklist/row-cap 兼容字段；需要与 15 份 v3 YAML 同步清理，不能只删 dataclass 字段。

`records.py` / `grammar.py` 归属已落实：免疫实现归 `immune_llada`，旧目录保留共享调用者需要的兼容 alias，不反向依赖 `qwen3_vl_arch` 的免疫实现。

`replaces_trait` 的 mix-dependent 归属已经确定：按 §6.6 由预处理阶段按 mix 组合生成
prepared dataset，训练期不保留该过滤逻辑。

---

## 2. 当前问题

### 2.1 数据逻辑分散

当前数据处理逻辑的**历史来源**曾分布在：

```text
/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/datasets.py（已删除）
/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/（共享模块/兼容 alias，保留）
/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain.py（已删除）
/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py
/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/
/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/
```

当前正式入口和免疫数据实现已收敛到 `immune_llada`；下游 PPI/STRING/MINT 所需共享代码不属于本轮删除范围。

正式的 ESMC + LLaDA 训练入口已不再依赖旧 `bioseq/datasets.py`；当前使用 prepared loader 和 `immune_llada` 的 canonical records、grammar 与 collator。`qwen3_vl_arch` 中仍保留的模块是 PPI/STRING/MINT 共享实现或有意的兼容 alias，不是旧 immune loader。

### 2.2 存在重复实现

多个位置分别实现了 OAS、OTS、nanobody 等数据适配、序列规范化和 row-to-record 转换逻辑。部分下游脚本还自行实现 source adapter，容易造成不同入口之间的行为不一致。

### 2.3 Filter 逻辑复杂且在训练时执行

当前过滤逻辑包含多层 wrapper、source-specific blacklist 和动态长度过滤。训练时读取原始数据并执行过滤会带来以下问题：

- 每次训练都重复处理相同原始数据；
- 训练启动慢且占用较多内存；
- filter 规则和 blacklist 变化不容易追踪；
- 过滤原因缺少统一统计；
- 不同训练入口可能使用不同的过滤逻辑；
- 难以保证训练数据与审计结果一致。

离线化已完成：旧 `ImmuneCsvDataset` 在训练入口中删除，raw CSV 解析、filter、blacklist 和异常样本替换由 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/preprocess_immune_dataset.py` 一次性完成。prepared loader 仅对已准备的 semantic JSONL 做轻量反序列化，不重新解析 raw source 字段或推断角色。

本次重构**不要求也不计划离线生成 model-ready token cache**。以下操作属于模型训练必需的数据组装路径，应继续保留在训练阶段：

- `GrammarRenderer` 将 semantic record 渲染为当前 grammar token stream；
- batch 内动态 padding 和 tensor 组装；
- 根据当前 batch 重建 per-chain encoder 输入；
- 训练目标所需的 diffusion/MLM 随机 mask 或噪声。

重构的性能目标是：训练阶段不再执行任何原始数据处理、去污染、过滤或异常样本替换；而不是消除 grammar 和 encoder batch 构造本身的必要成本。

### 2.4 旧入口退役

以下入口确认不再使用：

```text
examples/bioseq/train_bioseq_ddp.py
```

因此不需要为该入口保留新数据结构兼容层。与其专属的代码在确认没有其他调用者后，可以一并删除。

---

## 3. 重构目标

### 3.1 建立独立 pipeline

新 pipeline 不放在 `qwen3_vl_arch` 或其他与模型无关的目录下，建议使用：

```text
dllm/pipelines/immune_llada/
```

它与现有 pipeline 并列，专门负责免疫蛋白数据训练相关的数据读取、标准化、过滤、grammar 和 batch 构造。

### 3.2 建立统一数据流

目标数据流：

```text
Raw CSV / JSONL / Arrow
        │
        ▼
Dataset Reader
        │
        ▼
Source Adapter
        │
        ▼
BioSeqRecord
        │
        ▼
Offline Filter Pipeline
        │
        ▼
Prepared Semantic Dataset
        │
        ▼
Grammar Renderer
        │
        ▼
Grammar Collator + per-chain encoder input construction
        │
        ▼
Model Training Batch
```

Prepared dataset 保存经过标准化和过滤的 semantic records；不要求把 `input_ids`、grammar masks 或 encoder streams 预先固化为唯一训练数据。训练阶段可以根据当前 tokenizer、grammar 配置和 batch 重新生成这些模型输入。

### 3.3 训练前完成过滤

所有确定性的过滤操作在训练前执行一次，生成经过验证的 prepared dataset。训练阶段只加载 prepared dataset，不再读取 blacklist，也不再拼接复杂的动态 filter。

### 3.4 删除无效和重复代码

在完成调用关系确认和新旧链路验证后，删除重复 adapter、旧数据构建逻辑、退役训练入口及其无调用者的专属模块。

### 3.5 保证可复现、可审计、可回滚

每个 prepared dataset 都应保存输入数据、filter 配置、blacklist 和 schema 的 hash，以及逐 source 的输入量、保留量和 drop reason。

---

## 4. 目标目录结构

```text
dllm/pipelines/
├── immune_llada/
│   ├── __init__.py
│   ├── README.md
│   └── data/
│       ├── __init__.py
│       ├── records.py
│       ├── sources.py
│       ├── registry.py
│       ├── grammar.py
│       ├── collator.py
│       ├── dataset.py
│       ├── manifests.py
│       └── preprocessing/
│           ├── __init__.py
│           ├── pipeline.py
│           ├── filters.py
│           ├── writers.py
│           ├── reports.py
│           └── validators.py
```

预处理命令行入口：

```text
scripts/data/preprocess_immune_dataset.py
```

建议增加数据配置：

```text
configs/data/immune_v3.yaml
```

如果项目现有配置目录尚未建立，可以先根据仓库现有配置组织方式决定实际位置，但配置内容应与训练入口解耦。

---

## 5. 模块职责

### 5.1 `data/records.py`

定义 canonical record schema，所有 source adapter 的输出统一为：

```python
BioSeqRecord | None
```

建议核心对象包括：

```python
BioSeqRecord
BioSeqChain
```

`BioSeqRecord` 至少需要保存：

```text
chains
source
split
task_type
grammar_name
identifiers
metadata
```

每条 chain 至少保存：

```text
role
sequence
```

建议增加显式的 `identifiers: dict[str, str]`，用于保存 filter 和审计所需的稳定标识，例如：

```text
cdr3b_core
epitope
cdr3b_core|epitope
heavy_sequence
heavy_h3_core
benchmark_key
source_row_id
```

不再让正式训练主线依赖未经约束的 `dict[str, Any]` record。

这是一次**破坏性 schema 变更，不是新增字段**。现有定义
（`qwen3_vl_arch/data/records.py:99`）是 `frozen=True` dataclass，字段为：

```text
chains / task_type / source / split / metadata / labels / weight
```

既没有 `identifiers`，也没有 `grammar_name`。所有构造点都要改，需在 §11 单列一步
枚举并迁移：训练入口、`scripts/data/build_bioseq_grammar_v1.py`、`downstream/grammar/`、
`debug/probe_*.py`、`debug/verify_*.py`。

`grammar_name` 目前是**派生值**，由 `GrammarRenderer.encode` 依据 chain role 和
task_type 计算（`ppi_conditional` / `antigen_antibody` / `tcr_pmhc` / `single_entity`
等）。落盘存一份会产生两个真相源，离线值与运行时重算值可能漂移。二选一并写死：

- **保持纯派生，不落盘**（推荐）；
- 落盘但加载时重算并断言一致，不一致 fail-fast。

不允许"存了就直接信任"。

### 5.2 `data/sources.py`

负责各 source 的读取和字段适配。每个 adapter 只处理 source-specific 逻辑：

- 读取输入行；
- 解析字段；
- 序列规范化；
- 检查必要字段；
- 构造 chain 和 role；
- 设置 task type 和 grammar；
- 填充 identifiers 和 metadata。

source adapter 不负责 blacklist、全局过滤、写文件或训练采样。

### 5.3 `data/registry.py`

集中注册 source 名称、adapter、输入路径解析和 split 配置，避免在训练脚本中维护大量 source-specific 分支。

目标形式：

```python
SOURCE_REGISTRY = {
    "oas": ...,
    "ots": ...,
    "asd_antibody": ...,
    "trait": ...,
    "tcr_native": ...,
    "tcr_papers": ...,
    "tcr_repertoire": ...,
}
```

registry 实际需覆盖**八个** token，而非七个：额外的 `asd_nanobody` 有 builder 和
2,086 键 blocklist，现有文档标注"刻意不入 recipe"。registry 必须显式表态（保留
adapter、不进默认 recipe），不得静默省略 —— 静默省略会让后人以为是漏实现而
"补"回默认配方。

### 5.4 `data/grammar.py`

负责将 `BioSeqRecord` 渲染为模型使用的 grammar 表示。应保持当前 grammar 行为，包括：

```text
input_ids
fixed_context_mask
diffusion_loss_mask
diffusion_eligible_mask
token_class_ids
position_ids_chain
position_ids_inner
```

重构初期不进行大范围 grammar layout 或命名修改，优先保证行为一致。

### 5.5 `data/collator.py`

负责将 grammar 表示组装为训练 batch，包括：

```text
residue_mask
structure_token_mask
relation_token_mask
chain_ids
encoder_input_ids
encoder_attention_mask
encoder_residue_mask
encoder_chain_mask
```

ESMC encoder 的输入仍然按每条 chain 组织为：

```text
[B, C, L]
```

ESMC 条件只注入 residue 位置，grammar token 和 relation token 不接收 ESMC 条件。

### 5.6 `data/dataset.py`

提供 prepared semantic dataset loader。训练阶段只从 prepared dataset 加载标准化后的 record，不读取原始 CSV，也不重新执行去污染 filter。

loader 可以返回 `BioSeqRecord`，由训练阶段的 grammar renderer 和 collator 继续完成必要的 token stream、mask、padding 和 per-chain encoder 输入构造；这些操作不属于需要移除的过滤逻辑。

### 5.7 `data/preprocessing/`

只负责训练前离线处理：

```text
读取 source
生成 record
执行 filter
记录统计
写出 prepared dataset
生成 manifest 和报告
执行验证
```

### 5.8 `data/manifests.py`

定义 manifest 的生成、读取和校验逻辑，确保训练使用的数据版本可追踪。

---

## 6. Filter 设计

### 6.1 从嵌套 wrapper 改为显式 pipeline

不再使用类似下面的嵌套形式：

```python
with_exclusion_filter(
    with_exclusion_filter(
        with_exclusion_filter(...)
    )
)
```

改为显式的 filter 列表：

```python
filters = [
    InvalidSequenceFilter(...),
    MissingFieldFilter(...),
    MaxLengthFilter(...),
    BenchmarkExclusionFilter(...),
    T4ReferenceBinderFilter(...),
]
```

每个 filter 提供明确的名称和判断逻辑：

```python
class RecordFilter(Protocol):
    name: str

    def evaluate(self, record: BioSeqRecord) -> FilterDecision:
        ...
```

### 6.2 建议的 filter 类型

根据当前数据语义，至少整理以下过滤器：

```text
InvalidSequenceFilter
MissingFieldFilter
MaxResidueLengthFilter        # 毋基级，tokenizer 无关，离线安全
OasBenchmarkFilter
AsdAntibodyBenchmarkFilter
AsdNanobodyBenchmarkFilter
OtsBenchmarkFilter
TraitBenchmarkFilter
T4ReferenceBinderFilter
T2T3EvaluationFilter
RepertoireCoreProjectionFilter
ReplacesTraitFilter           # mix-dependent，非去污，见 §6.6
```

长度规则分为每链 residue 上限和 grammar 总长度预算；两者都在预处理阶段处理，训练阶段只保留不丢样本的安全断言，见 §6.7。

具体是否保留为独立 class，应以实际复杂度为准。简单规则可以组合成参数化 filter，但不能隐藏关键语义。

### 6.3 过滤统计

每条被过滤的 record 至少记录一个主 drop reason；预处理报告中按 source、split 和 reason 统计：

```json
{
  "schema":   { "invalid_sequence": 1200, "missing_field": 34 },
  "budget":   { "max_residue_length": 5021 },
  "dedup":    { "replaces_trait": 3357 },
  "decontam": { "oas_benchmark": 14000, "t4_reference_binder": 43000 }
}
```

`dedup` 与 `decontam` **必须分组**：前者是“被更好的同义数据顶替”，后者是“防止评测集泄漏”。
合并计数会掩盖去污覆盖率，使审计无法回答“评测集是否真的全部被挡住”。

建议同时记录：

- 输入总数；
- 成功转换 record 数；
- 保留数；
- 过滤数；
- 每个 filter 的命中数；
- 少量可复现的样本 fingerprint；
- 每个 source 的 train/valid 分布。

### 6.4 Key 空间必须明确

以下两种 key 不能混用：

```text
CDR3b_core
CDR3b_core|epitope
```

`tcr_repertoire` 没有 epitope，需要将配对 blacklist 投影为裸 core 时，必须使用显式的 repertoire projection 逻辑，不能在通用 filter 中隐式猜测。

### 6.5 保留去污染审计

以下脚本不能因为简化 filter 而删除：

```text
scripts/data/tcr_native/assert_corpus_fresh.py
scripts/data/dedup/audit_downstream_leakage.py
scripts/data/assert_residue_alphabet.py
```

它们应被纳入预处理流程的验证阶段，或由预处理后单独调用并把结果写入 validation report。

### 6.6 mix-dependent filter

`replaces_trait` 是否启用取决于 source mix。现有实现先根据 mix 装配排除集合
（`protein_pretrain_esmc.py:271`）：

```python
if "trait" in tokens and "tcr_native" in tokens:
    trait_exclusions = load_exclusion_keys(data_args.replaces_trait_blocklist)
```

过滤结果同时取决于**本次训练选了哪些 source**和当前 record 的 key。在预处理初始化时
确定 mix 和排除集合后，仍可复用 §6.1 的逐 record 判断，无需额外的动态过滤框架。
它语义上是“被全长 Fv 顶替”的**去重**，不是去污，统计应归入 `dedup`。

**本计划采用 A：prepared dataset 按 mix 组合生成。**

manifest 必须记录 source token 列表，`immune_v3` 这种命名不足以标识数据版本，需带
mix fingerprint。训练阶段不保留 `replaces_trait` 过滤；如果训练 recipe 改变，需要生成
对应的新 prepared dataset，避免训练期根据当前 source 组合临时改变语料。

B（顶替过滤留在训练期）不采用，因为它会违反“训练前完成确定性过滤”的目标。

### 6.7 长度过滤的分层

两个上限需要在预处理阶段完成过滤，但训练阶段仍保留不改变数据的安全校验：

- `max_protein_length`（每链 residue 数）：与 tokenizer、grammar 无关 → **预处理阶段执行**；
- `max_length`（grammar token 总数）：依赖当前 grammar 布局 → **预处理阶段使用同一套
  renderer 或等价的严格长度计算执行**。

预处理可以调用 renderer 只计算并校验长度，但不保存 token stream，也不生成
model-ready cache。这样所有确定性超长样本在训练前被移除，同时不改变训练阶段必须保留的
grammar encode 和 encoder batch 重建。

第一阶段保留现有 `_record_chain_lengths_ok` 的保守估算规则（序列总长加
`3*n_chains + 8`），不因改用精确长度而自动放宽入选范围；renderer 长度检查用于验证。
若以后放宽预算规则，应作为独立的数据版本变更，而不是夹带在代码迁移中。

训练阶段仍保留一道与 renderer 一致的廉价 fail-fast guard。renderer 或 collator 遇超长时
应直接报错而不是跳过；如果 prepared dataset 中出现超限记录，说明 manifest、grammar
版本或预处理实现不一致，必须修复数据集后重跑训练，而不能在 batch 内过滤。

manifest 必须记录 grammar 与 tokenizer 版本，否则 prepared dataset 被绑死在特定
(grammar, tokenizer) 组合上而无从察觉。

---

## 7. Offline Preprocessing 设计

### 7.1 入口

建议命令：

```bash
python scripts/data/preprocess_immune_dataset.py \
    --config configs/data/immune_v3.yaml \
    --output-dir data/prepared/immune_v3
```

建议支持：

```text
--dry-run
--max-rows
--source
--split
--overwrite
--resume
```

含义：

- `--dry-run`：只读取和统计，不写出最终数据；
- `--max-rows`：调试时限制每个 source 的输入行数；
- `--source`：只处理指定 source；
- `--split`：只处理指定 split；
- `--overwrite`：明确允许覆盖已有输出；
- `--resume`：按**已提交分片**恢复。跳过已在 shard manifest 中登记完成的分片，
  而不是复用残留临时目录（详见 §7.4）。

### 7.2 核心执行步骤

```text
1. 加载配置
2. 计算配置和输入文件 hash
3. 初始化 source registry
4. 流式读取 source
5. 转换为 BioSeqRecord
6. 按顺序执行 filter
7. 记录 drop reason 和统计
8. 流式写出保留 record
9. 运行 residue alphabet、freshness、leakage 等验证
10. 生成 manifest、filter report、schema 和 validation report
11. 使用临时目录完成原子提交
```

### 7.3 输出格式

建议保存 semantic record。**本计划不要求离线生成 model-ready cache，也不把离线 token 化作为重构目标。**

```text
data/prepared/immune_v3/
├── oas/
│   ├── train/
│   └── valid/
├── ots/
│   ├── train/
│   └── valid/
├── asd_antibody/
├── trait/
├── tcr_native/
├── tcr_papers/
├── tcr_repertoire/
├── dataset_manifest.json
├── filter_report.json
├── schema.json
└── validation_report.json
```

每条 prepared record 保存：

```text
chains
roles
sequences
regions
task_type
source
split
identifiers
metadata
labels
weight
```

其中 `regions`（链级）、`labels`、`weight` 是现有 schema 已有字段，不可在序列化时丢弃。
`grammar_name` 是否落盘取决于 §5.1 的决策，默认不落盘。

prepared dataset 不需要保存以下训练期模型输入：

```text
input_ids
fixed_context_mask
diffusion_loss_mask
encoder_input_ids
```

这些内容继续由训练阶段根据当前 tokenizer、grammar 和 batch 生成。本次重构不增加 model-ready token cache；性能验证也不以增加该缓存为前提。

### 7.4 大数据量和磁盘安全

当前数据规模较大，预处理必须采用流式读取和分片写入，避免把完整 CSV 读入内存，也不能无控制地复制原始数据。

建议：

- 按 source、split 写 shard；
- **按 shard 粒度提交**：单个 shard 写临时文件 → fsync → 原子 rename → 记入
  shard manifest；
- 全部 shard 完成后再提交 dataset 级 manifest；
- 输出目录存在时默认拒绍覆盖；
- 支持单 source 重跑；
- 记录每个 shard 的行数和 hash。

不采用"全局临时目录 + 末尾一次 rename"：该方案与 `--resume` 相干扰（跳进程残留的
临时目录无法区分"写完未提交"与"写到一半被杀"）。shard 级提交 + shard manifest
同时满足原子性和可恢复。

尚未提交 dataset 级 manifest 的输出目录**不得被训练读取**；训练启动时把
"manifest 缺失"视为数据集未完成，直接 fail-fast。

### 7.5 训练期性能边界

本次重构的目标不是删除所有 CPU 数据组装，而是保证训练期只执行模型 batch 所必需的工作。
训练期允许存在以下操作：

```text
Prepared semantic record -> GrammarRenderer
Grammar token stream -> batch padding / tensor creation
当前 batch -> per-chain encoder input reconstruction
当前训练目标 -> diffusion/MLM 随机 mask 或噪声
```

这些操作不能被误判为冗余 filter，也不应为了追求离线化而改变当前模型输入语义。

启动时的 schema、manifest 和 blacklist hash 一次性校验属于必要的安全检查（§8.1），不属于逐样本 filter。不得每个 batch 或每个 epoch 重复校验；也不应在每个 rank 启动时重新扫描整套原始语料来计算 hash。

训练期必须不存在以下操作：

```text
Raw CSV/JSONL 解析
source-specific row-to-record 转换
为逐样本过滤加载 blacklist、构造 exclusion set
去污染和 benchmark exclusion
非法序列/缺字段/超长样本的动态过滤
坏样本重试、跳过或用其他样本替换
```

长度规则需要区分两类：

- 每链 residue 长度和可由 semantic record 确定的预算，在预处理阶段过滤；
- grammar token 总长度依赖当前 renderer，训练期可以保留一个廉价的 fail-fast assertion，发现不一致时直接报错，**不得在 batch 内静默删除或替换样本**。

当前 `GrammarBioSeqCollator` 中的 renderer、per-chain encoder stream 构造、padding 和
`torch.tensor` 组装均属于必需路径，应迁移到新 pipeline，但不能删除。它们是否成为吞吐瓶颈，
必须通过 profiling 决定，而不是在设计阶段假设可以消除。

对于 `DataLoader`，应以实际训练吞吐和 DDP 一致性为依据选择配置：

- `num_workers=0`：作为正确性基线，便于复现并避免 worker shard 顺序问题；
- `num_workers>0`：在确认 iterable/map-style dataset 的分片、随机种子和 batch 顺序一致后，用于将 CPU collator 与 GPU forward/backward 重叠；
- `pin_memory=True`：CUDA 训练启用，并确认训练侧使用非阻塞 host-to-device copy；
- `persistent_workers` 和 `prefetch_factor`：只在启用 worker 后配置，并通过实测确定，不默认盲目增大。

应记录至少以下指标：

```text
samples/sec
residues/sec
step time
data wait time / idle time
CPU utilization
GPU utilization
host memory
worker memory
```

性能验收采用同一 prepared dataset、同一 batch size、同一模型配置，对比 `num_workers=0`
与候选 worker 配置。若增加 worker 没有降低 data wait time，保留更简单的配置。

---

## 8. Manifest 和可复现性

`dataset_manifest.json` 至少包含：

```json
{
  "dataset_name": "immune_v3",
  "schema_version": "1",
  "created_at": "...",
  "mix": {
    "source_tokens": ["oas", "ots", "asd_antibody", "trait", "tcr_native", "tcr_papers", "tcr_repertoire"],
    "mix_fingerprint": "..."
  },
  "sources": {
    "oas": {
      "input_path": "...",
      "input_hash": "...",
      "input_rows": 0,
      "converted_rows": 0,
      "kept_rows": 0
    }
  },
  "filters": {
    "config_hash": "...",
    "blacklists": {
      "replaces_trait": "...",
      "trait_benchmark": "...",
      "ots_benchmark": "...",
      "oas_benchmark": "...",
      "asd_antibody_benchmark": "...",
      "asd_nanobody_benchmark": "...",
      "t4_refbinder": "...",
      "t2t3_eval": "..."
    }
  },
  "grammar": {
    "name": "...",
    "version": "..."
  },
  "tokenizer": {
    "name": "...",
    "revision": "..."
  }
}
```

`blacklists` 必须列齐全部八个黑名单（对应 `DATA_PIPELINE_README.md` §2 的清单），
缺一个就等于那一层去污无从校验。`mix_fingerprint` 的必要性见 §6.6 决策 A。

### 8.1 黑名单漂移必须硬校验（待定项 1）

原计划把启动检查写为"blacklist hash 是否已记录"，只校验字段存在，**不校验与当前
黑名单文件一致**。这是本次重构最大的安全回退，必须补上。

背景：`DATA_FORMAT_AUDIT.md` 明确写过

> 断言只能*发现*脘节，*挡住*后果的是加载期过滤 ……双层不是冗余，它是唯一能兜住
> "黑名单重建了但语料没跟着重建"的机制。

双层去污是有意设计，历史上正是靠加载期那层挡住过真实泄漏（2026-08-28 `tcr_repertoire`
那 3 条）。§3.3 移除加载期过滤后，若只用"黑名单变了应重新生成数据集"这条**流程
约定**替代，等于把原本由代码强制的东西降级成人的纪律。

**强制机制（推荐方案）**：训练启动时重算当前八个黑名单文件的 hash，与 manifest
逐项比对，**不一致则拒绍训练**（非 warning）。错误信息必须指出是哪个黑名单变了、
并提示重跑预处理命令。

输入 hash 与全量数据审计在预处理阶段完成。训练启动的 blacklist hash 检查只读取规定的黑名单文件，不重新过滤 prepared records；它的耗时需实测，不能仅凭文件个数断言便宜。分布式启动应协调一次校验并向所有 rank 传递成功或失败结果，避免重复 I/O 或其他 rank 挂起。校验不放在 `__getitem__`、collator 或每个 epoch 中执行。

这样漂移会变成**启动时硬失败**，而不是静默训练在污染数据上。

若选择不实现硬校验，必须在本节显式记录"接受黑名单漂移风险"及理由，不得静默省略。

### 8.2 训练启动检查清单

```text
prepared dataset 是否存在
dataset 级 manifest 是否已提交（缺失 = 未完成，fail-fast）
schema version 是否兼容
source 是否齐全
split 是否齐全
blacklist hash 是否与当前文件逐项一致（§8.1，不一致则拒绍训练）
grammar / tokenizer 版本是否与 manifest 匹配（§6.7）
mix token 列表是否与本次训练请求一致（§6.6 决策 A 下）
```

blacklist 发生变化时必须重新生成 prepared dataset，而不是继续复用旧 artifact。

---

## 9. 训练入口重构

重点修改：

```text
examples/llada/protein_pretrain_esmc.py
```

如果 `examples/llada/protein_pretrain.py` 仍然属于现役训练入口，也一并迁移；如果已经退役，则先确认调用关系后删除，不为其保留不必要的兼容设计。

### 9.1 训练入口最终职责

训练入口只保留：

```text
解析模型和训练参数
加载 prepared dataset
构建 tokenizer
构建 grammar renderer
构建 collator
构建 ESMC + LLaDA 模型
构建 trainer
启动训练
```

### 9.2 应迁出的逻辑

以下逻辑应从训练入口迁出，并标注去处：

| 逻辑 | 去处 |
|---|---|
| `build_immune_specs` | `data/registry.py` + `data/preprocessing/pipeline.py` |
| source path 解析 | `data/registry.py` |
| source-specific row parsing | `data/sources.py` |
| blacklist 加载 | `data/preprocessing/filters.py` |
| 动态 exclusion filter | `data/preprocessing/filters.py`（§6.1） |
| 毋基级长度过滤 | `data/preprocessing/filters.py`（§6.7） |
| 原始数据读取 | `data/preprocessing/pipeline.py` |
| 复杂 eval 前缀截断 | **不迁移，消除**（见下） |

**关于 eval 前缀截断。** 现有实现是 `max_eval_rows_per_source` + `subsample_seed`，
其复杂度来自一个真实的正确性约束（`protein_pretrain_esmc.py:189` 注释）：行数上限
必须是**均匀随机采样而非前缀**，因为 `tcr_native/valid.csv` 是排过序的，取前缀会得到
有偏的验证集。

离线化后应在**写出阶段按 source/split 打乱顺序**（固定 seed，记入 manifest）。那么
前缀本身就已经是一个合法随机样本，训练期的 eval 上限退化成廉价的 `take(n)`，
`subsample_seed` 路径可以删除。这是**消除复杂度**，不是把它换个地方放。

注意：eval 行数上限本身是训练期的算力预算旋钮，**不应**离线写死 —— 离线写死等于
把一个可调参数烧进数据集。prepared dataset 存完整 valid split。

训练阶段允许并且需要保留：

- `BioSeqRecord` 到 grammar token stream 的编码；
- batch 内动态 padding；
- `fixed_context_mask`、`diffusion_loss_mask` 等模型输入的构造；
- per-chain encoder 输入重建；
- diffusion/MLM 所需的随机 mask 或噪声；
- 已准备好的 source 之间的采样；保持现有 recipe，不因重构把按行数混合改成新的加权混合策略；
- prepared valid split 上的轻量索引选取和 eval 数量上限（不扫描全量做 reservoir sampling）；
- 启动时一次性的 manifest/hash 校验及训练期必要的 shape、长度 fail-fast 检查。

训练阶段禁止保留：

- 原始 CSV/JSONL 解析；
- 为逐样本过滤加载 blacklist、构造 exclusion set 和执行 decontamination（不包括 §8.1 的启动 hash 校验）；
- source-specific row-to-record 转换；
- 非法序列、缺字段、每链超长等动态过滤；
- 遇到坏样本后跳过并替换为其他样本；
- 训练 batch 内 dedup 或其他会改变语料口径的隐式过滤。

grammar token 总长度可以保留一个与 renderer 一致的廉价 fail-fast 校验，但发现超限时必须直接报错，不能在训练期动态丢弃或替换样本。

---

## 10. 旧代码和冗余代码处理策略

### 10.1 退役入口

确认不再使用后删除：

```text
examples/bioseq/train_bioseq_ddp.py
```

同目录下还有一个**名字相近但不同的活跃入口**，原计划未提及：

```text
examples/bioseq/train_qwen3_vl_bioseq_ddp.py
```

它是 `GrammarDataModule` 的主要消费者（`import ... data import GrammarDataModule,
DEFAULT_GRAMMAR_DATA_DIR, TOKEN_CLASS_NAMES`），直接关系到 §10.5 的共享模块决策。
**不得因名字相近而误删**；若意图是整体退役 bioseq DDP 链路，需对它单独做退役评估。
同目录的 `train_ab.py` / `sample_ab.py` 也需在 Phase 0 确认调用关系。

同步清理相关启动说明、当前使用文档和无效配置引用。历史实验记录如仍有价值，应改为明确的 archived/legacy 说明，而不是继续作为推荐入口。

### 10.2 `bioseq/datasets.py`

目前不能立即删除，因为 LLaDA 训练入口和辅助脚本仍可能依赖它。处理顺序：

```text
1. 创建 immune_llada/data
2. 迁移七个 source adapter
3. 统一为 BioSeqRecord
4. 迁移 protein_pretrain_esmc.py
5. 迁移 protein_pretrain.py 或确认其退役
6. 迁移统计、审计和下游脚本
7. 全仓库搜索旧 import 和命令调用
8. 删除 bioseq/datasets.py
9. 清理 bioseq/__init__.py 中对应导出
```

### 10.3 `bioseq/ophiuchus`

不能仅因为旧 DDP 入口退役就直接删除整个目录。需要先确认其他训练、推理或 checkpoint 工具是否使用：

```text
bioseq/ophiuchus/model.py
bioseq/ophiuchus/collator.py
bioseq/ophiuchus/checkpoint.py
```

如果确认只服务于退役入口，则可以一并删除；否则只删除无调用者的模块。

### 10.4 `qwen3_vl_arch/data`

免疫 LLaDA 数据处理逻辑迁移到 `immune_llada` 后，`qwen3_vl_arch` 不应继续作为**免疫训练主线**的正式依赖。

但该目录**不只服务免疫**。它同时承载 PPI / STRING / MINT 链路：

```text
ppi_relations.py     ppi_splits.py      string_channels.py
grammar_builders.py  mixture.py         datamodule.py
esm_encoding.py
```

因此不能整目录搬走。迁移期间可以暂时保留兼容导出，但不保留重复实现。最终目标：

- 只有一份 `BioSeqRecord`；
- 只有一套 source adapter；
- 只有一套 grammar renderer；
- 只有一套 collator；
- 新 pipeline 内部不导入 `qwen3_vl_arch` 下的**免疫专用**逻辑。

注意最后一条与 §10.5 的决策相关：如果共享模块留在原处，"不导入 `qwen3_vl_arch`"就
无法字面成立，需改为"不导入免疫专用逻辑"。

### 10.5 共享模块的归属（待定项 2）

`records.py` 和 `grammar.py` 是**免疫与 PPI 共用**的，现有导入方至少包括：

```text
examples/llada/protein_pretrain_esmc.py        免疫主线
examples/llada/protein_pretrain.py             免疫
examples/llada/protein_fusion_model.py         免疫
examples/llada/load_fusion_checkpoint.py       免疫
examples/bioseq/train_qwen3_vl_bioseq_ddp.py   PPI / grammar shard
scripts/data/build_bioseq_grammar_v1.py        PPI + 免疫
scripts/data/audit_ppi_sources.py              PPI
scripts/count_grammar_layouts.py               统计
downstream/grammar/                            下游评测
debug/probe_*.py 、 debug/verify_*.py           调试（6 个文件）
```

**三选一，未定案不得开始 Phase 1**：

- **A. immune 拿走，PPI 反向依赖 `immune_llada`**。代价：依赖方向难看（PPI 为何要
  依赖免疫 pipeline），但改动量最小。
- **B. 复制两份**。直接违反 §10.4 "只有一份 `BioSeqRecord`"，**不推荐**；列在此处仅为
  明确排除。
- **C. 抽第三个共享层**（如 `dllm/pipelines/bioseq_common/`），免疫和 PPI 均依赖它。
  代价：多一层目录和一轮全仓库 import 改写；语义上最干净。

建议在 Phase 0 的调用关系清单产出后立即定案：如果 PPI 侧导入点少，选 C；如果 PPI 侧
导入点多且短期不动，选 A 并在此记录依赖方向是有意为之。

---

## 11. 分阶段实施计划

### Phase 0：基线和调用关系

不修改业务逻辑，先完成：

- 七个在役 source 的 raw count（`asd_nanobody` 单列，见 §5.3）；
- 当前 filter 后 count；
- 每个 source 的 train/valid count；
- task/grammar 分布；
- 固定样本 record fingerprint；
- grammar token stream；
- fixed/diffusion masks；
- encoder 输入；
- 当前训练配置和数据路径；
- 全仓库 import、脚本和命令调用关系；
- **`records.py` / `grammar.py` 的完整导入方清单**（§10.5 定案的输入）；
- **全部 `BioSeqRecord` 构造点清单**（§5.1 schema 迁移的输入）。

输出建议：

```text
refactor_baseline/immune_v3_baseline.json
```

`examples/bioseq/train_bioseq_ddp.py` 不纳入新旧链路 parity 目标，只做退役依赖确认。

**Phase 0 结束时必须给出 §1.2 两项决策的定案，否则不进 Phase 1。**

### Phase 1：创建独立 pipeline

**前置：§1.2 两项决策已定案。**

新建：

```text
dllm/pipelines/immune_llada/
```

按 §10.5 的定案处理共享模块，将免疫专用的数据模型和 grammar 逻辑整理到新路径，
先不删除旧实现。同时完成 §5.1 的 schema 迁移（`identifiers` 字段 + 全部构造点）。

### Phase 2：统一七个 source

将以下 source 全部转换为 canonical record：

```text
oas
ots
asd_antibody
trait
tcr_native
tcr_papers
tcr_repertoire
```

验证：

- chain role 正确；
- chain 顺序一致；
- task type 一致；
- grammar name 一致；
- identifiers 完整；
- metadata 和 source/split 保留；
- `regions` / `labels` / `weight` 保留（§7.3）。

`asd_nanobody` 按 §5.3 的定案处理：保留 adapter、不进默认 recipe。

### Phase 3：实现 offline filter pipeline

**前置：§6.6 的 mix-dependence 决策已定案。**

新增并实现：

```text
data/preprocessing/pipeline.py
data/preprocessing/filters.py
data/preprocessing/writers.py
data/preprocessing/reports.py
data/preprocessing/validators.py
data/manifests.py
scripts/data/preprocess_immune_dataset.py
```

先支持小规模 `--dry-run` 和 `--max-rows`，确认统计和 drop reason 正确后再运行全量数据。
drop reason 必须按 §6.3 的四组分类输出。

### Phase 4：生成 prepared dataset

按 source 和 split 流式生成 prepared artifact，并输出：

```text
dataset_manifest.json
filter_report.json
schema.json
validation_report.json
```

验证所有审计脚本均能通过。

### Phase 5：新旧链路 parity

当前进度：已完成 raw source adapter 的 sampled parity（报告见
`refactor_baseline/immune_adapter_parity_train_5000.json` 和
`refactor_baseline/immune_adapter_parity_valid_5000.json`）。数据集层的全量
fingerprint -> multiplicity map，以及 rendering 层逐字节 parity 仍按下面的完整
验收标准待执行。

parity 分数据集层与 rendering 层。**只改变过滤执行位置不应改变样本数量或内容**，写出顺序可以变化。

**第一层：数据集层（数量和多重集合相等，不要求存储顺序相等）**

```text
per (source, split) 的保留 record count 相等
per (source, split) 的 record fingerprint -> 出现次数映射相等
```

锁定输入、blacklist、mix、长度规则和采样范围后比较。不能只比较普通集合，否则重复样本
被删除或增加也可能误判为通过。fingerprint 应覆盖链顺序、角色、序列、labels、regions
等语义字段；重排后的物理行号不作为样本内容变化。

任何差异都需要归因。mix 或长度规则的有意变化应单独发布数据版本，不作为本次重构
默认允许的差异；不能以“过滤换了位置”或“总数接近”为由放行。

**第二层：rendering 层（逐字节相等）**

在**固定 record 列表 + 锁定 seed** 下比（绕开采样和顺序变量）：

```text
chain order
chain role
task_type
grammar_name
grammar token stream
fixed_context_mask
diffusion_loss_mask
diffusion_eligible_mask
encoder input
```

这一层必须**完全一致**，任何差异都是 grammar 或 collator 回归。注意 `GrammarRenderer`
带 `rng`，比较时必须注入相同 seed，否则差异无意义。

出现差异时必须区分：

- 存储顺序变化（允许，但固定样本对照时需对齐）；
- 真实的 record 内容、数量或重复次数变化（不是过滤迁移的预期结果）；
- grammar 或 collator 回归（不允许）。

### Phase 6：切换训练入口

修改：

```text
examples/llada/protein_pretrain_esmc.py
```

新增类似参数：

```text
--prepared_data_dir data/prepared/immune_v3
```

同时必须实现：

- §8.1 的黑名单 hash 硬校验（不一致拒绍训练）；
- §8.2 的完整启动检查清单；
- §6.7 的 token 级长度 guard。

训练配置切换到 prepared dataset。正式训练使用新的输出目录和数据版本，避免覆盖既有 checkpoint 或混淆历史实验。

### Phase 7：迁移辅助和下游脚本

原计划只列了三个脚本，实际消费方远不止此。完整清单（以 Phase 0 产出为准）：

**统计与审计**

```text
scripts/count_grammar_layouts.py
scripts/count_immune_drops.py
scripts/count_immune_mix.py
scripts/data/audit_training_data_scale.py
```

**下游评测**

```text
scripts/downstream/score_pairing_pll.py
downstream/grammar/common.py
```

**调试与验证**

```text
debug/grid_sampling.py
debug/probe_batch_isolation.py
debug/probe_jmotif.py
debug/probe_vocab_mass.py
debug/verify_jmotif.py
debug/verify_jmotif2.py
debug/verify_sampling.py
```

**数据构建**

```text
scripts/data/build_bioseq_grammar_v1.py
```

删除脚本中重复的 row-to-record 实现，统一使用 `immune_llada` 的 canonical adapter 或 prepared artifact。

注意 `count_immune_drops.py` 和 `count_immune_mix.py` 现在**故意复用训练入口的
`build_immune_specs`**，目的是保证"数出来的就是训练看到的"。迁移后必须保留这个
属性：改为直接读 prepared dataset + filter report，而不是重实现一遍统计逻辑。

### Phase 8：删除旧实现

确认全仓库没有有效调用后，删除：

```text
examples/bioseq/train_bioseq_ddp.py
dllm/pipelines/bioseq/datasets.py
```

以及已经无调用者的旧 source adapter、旧 filter wrapper 和旧训练数据逻辑。

删除前必须完成：

```text
全仓库 import 搜索
全仓库命令调用搜索
训练配置搜索
测试和 smoke test
新旧链路 parity
```

---

## 12. 验证方案

### 12.1 单元测试

新测试放入现有测试目录（仓库已有 `scripts/tests/bioseq/`，包含
`test_qwen3_vl_grammar.py`、`test_grammar_tcr_relation.py` 等），建议新建：

```text
scripts/tests/immune_llada/
```

不另开顶层 `tests/` 目录。为以下部分增加测试：

- 每个 source adapter 的最小有效输入；
- 缺字段、非法序列和超长数据；
- 每个 blacklist filter；
- 裸 core 与配对 key 的区分；
- repertoire core projection；
- record schema 序列化和反序列化（含 `regions` / `labels` / `weight` 往返）；
- grammar rendering；
- collator 输出 shape 和 mask；
- **黑名单 hash 不一致时确实拒绍训练**（§8.1，回归测试）；
- **`replaces_trait` 在不同 mix 下的行为**（§6.6）。

### 12.2 预处理 smoke test

使用少量样本运行：

```bash
python scripts/data/preprocess_immune_dataset.py \
    --config configs/data/immune_v3.yaml \
    --output-dir /tmp/immune_llada_smoke \
    --dry-run \
    --max-rows 1000
```

然后运行实际小规模写出，验证：

- 输出目录结构；
- manifest；
- filter report；
- schema；
- validation report；
- prepared loader。

### 12.3 训练 smoke test

先做 dataset/collator dry run，再用少量 prepared 样本实际运行 forward/backward 和 checkpoint 保存恢复。当前训练入口的 `--dry_run` 会在模型构建前返回，不能把它当作训练 smoke test 已通过的依据。验证：

- 数据能够加载；
- grammar 能够渲染；
- collator 能够组 batch；
- ESMC encoder 输入 shape 正确；
- LLaDA forward 和 loss 正常；
- checkpoint 能够保存和恢复；
- 记录固定步数下的 `step time`、`data wait time` 和 GPU utilization，确认没有因 prepared loader、record 反序列化或 collator 配置引入明显的数据等待。

### 12.4 性能 profiling

在不改变模型、batch size、数据版本和训练目标的条件下，对同一批 prepared dataset 做短程对比：

```text
num_workers=0
候选 num_workers > 0
pin_memory 开启/关闭（仅 CUDA）
prefetch_factor / persistent_workers 的候选值
```

比较：

```text
samples/sec
residues/sec
平均 step time
data wait time
GPU utilization
CPU/host memory
```

先以 `num_workers=0` 作为正确性基线，再选择能够降低 data wait time 且不改变 DDP 样本顺序、seed 和 batch 数量的配置。若 collator 并未造成可观的数据等待，则保留简单配置，不为理论上的并行预取增加复杂性。

性能对照分两步：先固定模型、样本批次、source 配比、长度分布和 worker 配置，比较旧数据链路与新链路；再使用同一 prepared dataset 调整 worker 配置。不要引入 task-homogeneous batching、长度分桶或新的混合策略来掩盖 loader 的回归。

测量分开记录启动到首 batch 的耗时和 warm-up 后的稳定吞吐；区分 microbatch 与 optimizer step，保持梯度累积配置一致。短程 profiling 不含保存 checkpoint 和周期 eval，另行记录这两项开销；记录 data wait 的分位数，不能仅凭 GPU utilization 判断数据瓶颈。

该 profiling 只用于确认训练阶段的必要 grammar/collator 工作没有成为不可接受的吞吐瓶颈；不以生成 model-ready token cache 为解决方案。未完成实测前，不承诺“零等待”或具体提速比例。

### 12.5 全量数据验证

全量预处理完成后检查：

- 七个在役 source 均有预期 split；
- 保留量与基线差异可解释（差异归因见 Phase 5 第一层）；
- drop reason 总量闭合，且 `dedup` 与 `decontam` 分组正确（§6.3）；
- residue alphabet 审计通过；
- corpus freshness 审计通过；
- downstream leakage 审计通过；
- manifest 中八个黑名单 hash 均完整（§8）；
- 训练读取的 prepared dataset 与 manifest 一致；
- **故意改动一个黑名单后，训练确实启动失败**（§8.1 的端到端验证）。

---

## 13. 回滚策略

重构期间不覆盖原始数据、旧 checkpoint 和旧训练输出：

- 新 prepared dataset 使用新的版本目录；
- 新训练使用新的 `OUTPUT_DIR`；
- 每个 shard 写临时文件后原子提交（§7.4）；
- 旧代码在 parity 和 smoke test 通过前不删除；
- 删除操作集中在最后一个独立阶段；
- manifest 保留输入、配置和 blacklist hash，便于重建相同数据版本。

如果新 pipeline 出现问题，可以回滚训练配置到旧入口和旧数据版本，而不会影响原始数据和已有 checkpoint。

---

## 14. 完成标准

本次重构完成需要满足：

1. 新数据代码位于 `dllm/pipelines/immune_llada/`；
2. 新 pipeline 有完整的 `README.md`；
3. 正式训练不再依赖 `qwen3_vl_arch` 下的**免疫专用**数据模块（共享模块按 §10.5 定案）；
4. 七个在役 source 统一输出 `BioSeqRecord`，`asd_nanobody` 的定位已显式记录（§5.3）；
5. 确定性过滤只在训练前离线执行；训练期只保留不改变数据的 token 级长度/shape
   fail-fast guard，mix-dependent 顶替逻辑按 §6.6 生成对应 prepared dataset；
6. 训练阶段只加载 prepared dataset，不读原始 CSV、不读黑名单做过滤；
7. prepared dataset 有 manifest、filter report、schema 和 validation report；
8. blacklist key 语义明确且经过验证；
9. **训练启动时校验黑名单 hash 与 manifest 逐项一致，不一致则拒绍训练**（§8.1）；
10. `protein_pretrain_esmc.py` 不再包含 source-specific 数据处理和动态 blacklist 逻辑；
11. `examples/bioseq/train_bioseq_ddp.py` 已退役并删除；`train_qwen3_vl_bioseq_ddp.py`
    的去向已单独结论（§10.1）；
12. `bioseq/datasets.py` 在所有有效调用迁移后删除；
13. 重复 adapter、重复 filter 和无效脚本已删除；
14. 单元测试、smoke test、数据审计和训练 dry run 通过；
15. 完成训练数据路径 profiling，记录并确认 data wait time、GPU utilization 和 step throughput 在可接受范围内；
16. Phase 5 两层 parity 均达成，数据集层差集已逐条归因；
17. 既有数据、checkpoint 和训练结果未被覆盖；
18. 迁移过程中的关键注释（尤其去污语义）未丢失，见 §16。

---

## 15. 最终职责边界

```text
dllm/pipelines/immune_llada/
    当前免疫蛋白 LLaDA 训练的数据 pipeline

dllm/pipelines/immune_llada/data/
    record、source、grammar、collator、prepared dataset loader

dllm/pipelines/immune_llada/data/preprocessing/
    训练前离线读取、标准化、过滤、写出、审计和 manifest

examples/llada/protein_pretrain_esmc.py
    ESMC + LLaDA 模型训练入口

examples/bioseq/train_bioseq_ddp.py
    已退役，删除

dllm/pipelines/bioseq/datasets.py
    迁移完成后删除

```text
dllm/pipelines/qwen3_vl_arch/data/
    不再承载免疫 LLaDA 专用数据处理逻辑；
    仍承载 PPI / STRING / MINT 链路（§10.4）；
    共享模块归属见 §10.5
```

核心原则：

> 原始数据只在训练前处理一次；所有 source 统一为 `BioSeqRecord`；过滤结果可审计、可复现；训练阶段只读取已经过滤和验证过的 prepared dataset；免疫数据逻辑归属于独立的 `immune_llada` pipeline。

但离线化不得以丢失运行期防御为代价：黑名单漂移必须在启动时硬失败（§8.1）。

---

## 16. 注释与文档规约

现有代码的注释里有大量**"为什么"而非"是什么"**的信息，重构最容易丢的恰好是这部分。
例如 `datasets.py:382` 解释为何 repertoire 必须把配对键投影成裸 core、以及为何这是
"deliberately over-aggressive" 的 defence in depth；`protein_pretrain_esmc.py:189`
解释为何 eval 行数上限必须是随机采样而非前缀。这类信息一旦丢掉，后人会把它当成
bug 然后"修好"。

因此本次重构遵守：

1. **逐段搬运原注释，不重写。** 原注释里的历史事例、日期、实测数字一律保留。
2. **每个 filter 的 docstring 必须说明**：键空间（裸 core 还是 `core|epitope`）、
   为何选这个键、过挡/漏挡的取舍、属于 decontam 还是 dedup。
3. **跳源依赖必须注明依赖条件。** 如 `replaces_trait` 仅当 `tcr_native` 同时在 mix 中
   才生效（§6.6）。
4. **故意的保守行为要标注为故意。** 例如长度估算的过估、repertoire 的过挡，
   否则会被当成精度问题优化掉。
5. **fail-fast 路径要写明为何不能降级为 warning。** 尤其是 §8.1 的黑名单校验和
   `load_exclusion_keys` 的空集报错。
6. **删除旧代码时，先确认其注释里的知识已在新位置有落点。** Phase 8 的删除清单
   逐个文件执行这项检查。
