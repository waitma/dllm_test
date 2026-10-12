# 训练数据与待测 benchmark 去重范围核对

## 后续最终版本已通过确认范围审核 — 2026-10-02T18:53:05+00:00

本文早先待办为历史状态；字段回填、T4 覆盖与入口规则已在新最终版本收尾。数据路径、证书及限定范围以[最终验收记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_SCOPE_FINAL_20261002.md)为准，旧版本文件不变。

## 2026-10-02T18:15:56+00:00 当前审核范围已按用户要求缩减

原 V7 264 条 H3 未确认项与 GDPa1/Specificity/m396 专门保护证据不再要求确认，不再阻断本轮审核。历史缺证据结论保留；最新必需待办见[r2 记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_R2_20261002.md)。

## 后续 r2 修复已验收 — 2026-10-02T17:40:43+00:00

合成占位链误判配对的问题已在新 r2 版本修复，旧版历史结论保持；整体剩余审核项未自动关闭。最新数据路径、逐行差异与复审结果见[r2 完成记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_R2_20261002.md)。

## 后续语义审核未通过 — 2026-10-02T17:05:42+00:00

此前工程验收通过不等于清理条件正确；现已发现合成占位链误判为真实配对的问题。当前版本仍未通过整体审核。以[本轮审核及全量影响统计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_AUDIT_20261002.md)为准，原构建数字保留为历史事实。

## 2026-10-02T16:56:15+00:00 后续配对完全重复重构已完成

旧章节中的“尚未导出”是历史状态。现已导出新版本并完成独立选择/字节复核；规则、数量、产物与未解决审核项以[本轮记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_REBUILD_20261002.md)为准。

## 2026-10-01T23:36:04.060217+00:00 “后来 test/query”术语解释（仅说明既有范围记录）

“后来”指本文件 2026-10-01T11:15:42+00:00 的范围记录；本回合没有作出新的保护范围决定。test 是最终测试划分里实际预测并计分的样本；query 是 few-shot 评测中待预测、实际参与评分的样本；support/reference 是给评测方法的少量已标注参考样本。该范围记录不自动排除 benchmark train/valid/仅作为 support 的记录；只要一条记录在其他评测轮次承担 query 角色，就应纳入实际评分范围。

既有锁定范围清单包括 NM2025 的 8 个 test.csv cohort、tcr_binding/test、tcr_representation/test、UpTCR few-shot/majority/zero-shot 的 test，以及两项 clustering 全集和 paper6 background_pool/target_binders。paper6 background 是测试负例；target_binders 会在不同轮次或目标中进入 query，不能把整个池视为仅 support。依据：[旧命中表范围核算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/dedup_test_only_review_20261001/scope_counts.json)、[few-shot 实际评分代码](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/fewshot.py:447)。这里的清单属于该次 TCR 范围核算，不代表全部新增抗体任务已完成覆盖。

这说明需要保护哪些评测样本；匹配/排除判据是另外的协议项。范围记录尚未变成完整新版训练数据与有效审核绑定；本回合未扫描序列/训练分片、执行去重、修改 gate、生成证书或启动训练，原 Goal 状态未变。

## 2026-10-01T23:31:00.205418+00:00 两份审核文件的含义（代码核对，未执行审核）

此前“证书/两份有效审核”的表述容易误解。这里指项目自身定义的两份 JSON 检查报告；现有代码没有要求外部机构签章或两位人工审核人，也不是权限审批。

| 文件 | 当前代码检查的内容 |
|---|---|
| audit_certificate.json | 记录指定数据版本的清理审核状态、未解决项、规定的禁止重叠及 train/valid 分组交叉；绑定 dataset_manifest.json。训练入口另核对实际分片 SHA。 |
| downstream_scope_certificate.json | 记录代码定义的下游保护范围审核；绑定同一 dataset_manifest.json、范围版本、benchmark 输入与证据文件 SHA，避免旧范围的通过记录被用于新增或已变更任务。 |

“有效”要求真实核查证据支持 passed、无 pending、禁止重叠符合门禁且文件/范围/版本绑定一致；仅新建 JSON、填写 passed 或复制旧报告不能满足要求。明确 training_ready 及原有其他准入条件仍保留。

代码依据：[训练入口 assert_ready](/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/immune_pretrain_v3.py:39)、[下游范围检查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/tcr_scope_gate.py:13)。目前下游检查器仍引用旧 all-support/query 范围；用户后来确定的实际 test/query 范围尚未形成完整一致的新审核绑定。本轮只解释现有实现，没有修改范围或门禁、执行清理/注释/扫描、生成证书或启动训练。

## 2026-10-01T16:26:25.076053+00:00 当前审核未完成的原因澄清

已有清理和部分审计；最终数据审核尚未收尾。旧clean_r2 audit_certificate状态pending，标签unresolved_benchmark_h3；MHC输入更新没有对应新manifest的双证书和明确ready。expanded导出报告仍有role coverage/near-neighbor review、独立parent/expanded scope复扫、双证书及upstream parent-antigen溯源pending。此前补覆盖的序列注释分支被安全检查中止，按用户要求没有恢复；后续模型工程检查并未解决这些数据项。最新test/query保护决定与旧gate SCOPE/input清单也未形成一致的新审核绑定。缺证书反映上述审核工作未完成，不是机器权限申请；这些记录不证明新范围已有残留泄漏。 [原始状态与来源](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/resumed_external_review_1790871604_1.json)。本次仅metadata与既有记录解释，未重新搜索序列/注释/扫描训练分片、未修改范围gate、未出具证书或更新prepared。

## 只保护测试集：范围更正与 OTS 删除原因 — 2026-10-01T11:15:42+00:00

用户明确只保护测试集；此前全量 support/adaptation/validation/query 保护不再作为后续默认方案。官方 benchmark 行保持原样，从 foundation 语料排除测试暴露；不得把 benchmark train/valid/support 自动加入保护。以下是对既有命中记录的范围重算，并非新数据导出或认证。

已校验锁定 row_hits SHA256，并检查各 shard 行号递增、范围有效，以行级 union 统计，未重新执行序列注释或相似性搜索。[可复核统计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/dedup_test_only_review_20261001/scope_counts.json)记录完整 cohort 清单与掩码：排除 7 个明确 train/valid cohort，保留 17 个实际评测 cohort。paper6 的 background 是测试负例；target_binders 在跨 target 的评测中参与 query，不能作为纯 support 池排除。代码依据为[评分循环](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/fewshot.py)中的 keep 排除当次 reference，其他 universe 行均参与评分。两项 clustering 全集用于评测，也保留。

| OTS 统计（train + valid） | 行数 |
|---|---:|
| V7 基数 | 1,705,047 |
| 旧全范围删除 | 1,627,265（95.44%） |
| 仅测试范围、沿用旧匹配规则时删除 | 1,622,364（95.15%） |
| 仅因排除 train/valid 恢复 | 4,901 |
| 测试范围下保留 | 82,683 |
| 测试范围删除中，有至少一个 exact 原因 | 116,026 |
| 测试范围删除中，仅 near80 原因 | 1,506,338 |
| 测试范围 exact_ab：两条 CDR3 同时完全匹配 | 244 |

旧规则是任一单链 CDR3 的全局 Levenshtein 相似度 `1-distance/max_length >= 0.80` 即删除整条配对记录，并不要求两条链匹配同一个测试配对。exact 原因也包括单链完全匹配，不等于配对重复。near-only 中仅 alpha 命中 500,865、仅 beta 命中 141,628、两条链各有命中 863,845；最后一项不证明同一个 benchmark 配对同时相似。near-only 数量较旧全范围上升，是部分行的非测试 exact 原因被排除后重新归类，并非新增删除。

全部来源：仅测试范围沿用旧规则将保留 4,808,266 条，相比旧 expanded candidate 恢复 26,423 条。这证明大幅损失主要由匹配粒度造成，缩小保护范围本身不能解决。

处理状态：范围决定已修正文档；旧 expanded candidate 保留为历史产物，不作为满足新要求的最终版。本轮未更改数据行、冻结命中表或旧导出器，未生成新 candidate。配对重复与单链相似应分开报告；是否排除单链近似属于独立评测口径，不能擅自等同于配对去重。后续导出必须从原 MHC 版本恢复筛选，而不能继续在已过筛的 candidate 上过滤。

## 用户确认旧 OAS 与其他任务复核 — 2026-10-01T04:09:15+00:00

用户明确 Kong/SAb 使用已去重的旧 OAS，因此按既有处理保留，不再将这 36 个当前路径文件当作新增未处理数据集。这是用户提供的来源确认，不冒称本轮独立完成全量序列认证。

| 其他任务 | 对 V7 去重执行记录的结论 |
|---|---|
| GDPa1 / Specificity / m396 | 未找到专门纳入保护库并对 V7 执行排除的证据。checked builder 没有这三项专门读取分支，两份 registry 无对应来源；评测内部 fold 检查不能替代训练语料去重。可能有其他来源造成的间接保护，未据此认定全部泄漏。 |
| AB light pairing | 旧 OAS holdout 已登记进 V7 保护库，并用于既有过滤。 |
| 原有 TCR T1 / T2 / T3 已登记范围 | 有保护集来源和 V7 过滤记录；不扩张成所有版本/支持池均已完成。 |
| UpTCR + 扩展支持/查询池 | 后续已按 24 个 cohort 的锁定命中表生成 expanded candidate；原 V7 和后补候选必须区分，候选仍未通过训练认证。 |
| AB–抗原 binding / SPR / AbBiBench / CHIMERA | V7 registry 有显式保护统计并用于过滤；保护的 split/抗原范围按各任务既有规则，不能外推至其他补充 baseline 数据。 |
| T4 当前 benchmark14 / held20 | 有旧 holdout/公共参考库保护，但本轮未建立其对当前 conditional 入口的完整等价覆盖，仍待核对；不能只凭文件名就判定未做。 |

结论优先级：剩余三项抗体探针的专门保护证据缺口最明确；T4 需要来源映射复核。机器可读依据及源码/registry SHA 见 [其他任务来源核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/dedup_scope_review_20261001/other_benchmark_provenance.json)。本轮没有执行新去重、重划分或修改任何 benchmark。

## 核对更正：V7 已执行过去重

用户指出 V7 已处理评测保护集后，重新核对原清理报告、证书和哈希绑定的 effective bank。V7 clean_r2 的 [cleaning_report](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_clean_r2/cleaning_report.json) 确有数据内精确语义去重（802条）及各类下游重叠过滤；[旧范围证书](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_clean_r2/audit_certificate.json) 记录 forbidden_overlap_rows=0、family_split_overlap=0，但状态仍 pending（unresolved_benchmark_h3）。这些是已执行工作的证据，不能说 V7 没做过去重。

前次 40 文件路径/哈希未登记的结论仅是溯源绑定缺口，不能等同 40 文件的内容全部未被保护，更不能据此认定 V7 存在实际残留泄漏。对当前 CDR 文件已存显式字段做只读 exact-membership 检查，确认与旧保护库有重合；例如 Kong 的 1,819 个不同显式 H3 值有 1,625 个在库中，SAb23H2 的 60 条不同重链有 40 条在库中。[检查记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/dedup_scope_review_20261001/existing_v7_cdr_bank_membership.json)绑定原 effective-bank SHA，未推断/重新注释序列或做相似度搜索。

该直接字段比较未消除所有表示差异，缺失 exact 值也不等于没有经过相似性排除；没有重新扫描 V7 分片，因此不能把上述计数当作残留泄漏数量或完整覆盖证书。后续应复用 V7 已有去重，补齐当前 benchmark 的溯源绑定和剩余覆盖验证，避免把缺少文件登记误当作全部未处理而从头重跑。数据与官方 benchmark 仍未修改。

更新：2026-10-01T03:54:36.632452+00:00。状态：已完成初步入口与旧清单校验，尚未生成新清洗数据，benchmark 内部是否改动等待用户明确。此前用户明确保留官方划分，本次“都去重”可能指训练↔benchmark，也可能包含 benchmark 内部清洗，需要区分。

- 旧两份 registry 合并为 79 个唯一输入文件，当前 SHA256 全部匹配，无缺失。这只证明旧输入未变，不证明当前 benchmark 全覆盖。见 [绑定核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/dedup_scope_review_20261001/bound_input_check.json)。
- 当前 CDR Kong / SAb23H2 入口和 GDPa1、Specificity、m396 输入/参考文件共检查 40 个，其中 40 个路径不在旧清单、40 个内容哈希不在旧清单。应补范围核对，不能直接复用旧“覆盖完成”结论。见 [当前入口差异](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/dedup_scope_review_20261001/current_entry_scope_gaps.json)。路径/哈希缺口不等于已经证明存在序列重叠。
- 已存在候选 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_expanded_candidate_20260930`，train 4,735,948、valid 45,895；仍 training_ready=false，缺独立重扫和双证书。旧严格并集过滤后 native/TRAIT 的 valid 各仅 4 条；不能把所有去除理由都称为精确重复，也不能通过复制/逐行移动凑验证数。
- 接下来需明确官方 benchmark 是否保持原划分；两种方案均保留原始数据、建立新的版本目录及逐条删除原因。尚未执行新的去重、重划分、训练或 benchmark 测试。

建议将训练语料对全部待测 benchmark 的排除，与 benchmark 内部重复/标签冲突诊断分开。若需要清洗 benchmark，生成独立派生版本，单列结果，不把新测试集得分直接冒充原论文协议得分。
