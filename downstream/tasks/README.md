# 在范围内的下游任务文档

历史作业 YAML 位置见 [归档索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/README.md)；现行配置与生成器模板见 [评测目录说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/README.md)。

2026-09-30 新增 [UpTCR binding benchmark](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md#uptcr-binding)：原发布划分及标签原样使用，不清洗去重；baseline用论文 [P]，Ours实测单列。已纳入范围；执行状态统一见 [UpTCR 当前结果](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#uptcr-binding-status)。

2026-09-27 AB-AG 范围更新以 [benchmark README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md) 为准；纯序列派生协议和本模型入口见[任务手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。

**2026-09-24 当前决定：FLAb2 已排除；MAGE 保留。** FLAb2 已从 active source lock 和方法清单移除，不再处理其缺失代码或依赖。范围以 [README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md) 为准。

2026-09-24 新授权任务：[AB-AG 官方协议手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)（抗原条件配对生成、binding、KD；只使用作者发布的比较代码，禁止 VHH）。当前状态见 [AB-AG 执行状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-antigen)。下方为历史六项任务索引。

2026-09-13 TCR 当前执行子任务与双链生成延后决定唯一见 [benchmark README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md)；具体新入口/gate 见各 TCR task §8.0，关键日志见 [审计 §9.21](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。

阅读顺序：[`../benchmark/PROJGUIDE.md`](../benchmark/PROJGUIDE.md) → **本目录对应任务文档** → [`../benchmark/RESULTS.md`](../benchmark/RESULTS.md) §0。

模板：[`_TEMPLATE.md`](_TEMPLATE.md)（8 节结构不许改）。规则：[`../../.cursor/rules/downstream-doc-sync.mdc`](../../.cursor/rules/downstream-doc-sync.mdc)。

| 任务 | 文档 | RESULTS | BERT | Diffusion |
|---|---|---|:--:|:--:|
| T1 Binding | [TCR_T1_BINDING.md](TCR_T1_BINDING.md) | §0.1 | yes | yes |
| T2 Clustering | [TCR_T2_CLUSTERING.md](TCR_T2_CLUSTERING.md) | §0.2 | yes | yes |
| T3 Representation | [TCR_T3_REPRESENTATION.md](TCR_T3_REPRESENTATION.md) | §0.3 | yes | yes |
| T4 Generation | [TCR_T4_GENERATION.md](TCR_T4_GENERATION.md) | §0.4 | **no** | yes |
| AB CDR infilling | [AB_CDR_INFILLING.md](AB_CDR_INFILLING.md) | §0.5 | **no** | yes |
| AB Light pairing | [AB_LIGHT_CHAIN_PAIRING.md](AB_LIGHT_CHAIN_PAIRING.md) | §0.6 | **no** | yes |

排行榜：**论文值 `[P]` 优先**；`[R]` 复跑只作附行。本文档不复述数字。

当前范围以 [README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md) 为准；本表是六项任务索引，附加任务见该节和原生探针手册。

2026-09-24 范围同步：用户授权的 BioMap ΔG 官方 baseline 首轮执行以 [benchmark README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md#0-评测范围2026-08-29-收窄强制) 为准；运行状态见 [BioMap 官方运行状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#biomap-official-run)。
