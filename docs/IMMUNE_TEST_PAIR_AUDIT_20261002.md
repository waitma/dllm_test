# 新版数据审核：暂不通过

## 后续 r2 修复已验收 — 2026-10-02T17:40:43+00:00

合成占位链误判配对的问题已在新 r2 版本修复，旧版历史结论保持；整体剩余审核项未自动关闭。最新数据路径、逐行差异与复审结果见[r2 完成记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_R2_20261002.md)。

审核完成：2026-10-02T17:05:42+00:00。对象：/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_pair_exact_20261002。绑定清单 SHA256：`8b7818b3ceec2863f2bf3634de512e0f8fdb681743fd7c80b189a173d42b9c1c`。

## 必须修正：合成占位链被当成真实配对

上一轮导出器只检查 chain_roles 同时有 tcr_alpha/tcr_beta；独立选择验证器也重复了此假设。它们证明代码按既定条件执行，未证明条件符合数据语义。现有 synthetic_residue_mask 全 1 的链没有任何观测残基，应视为缺失链，不能参与配对重复判定。既有 sources.py 的 is_fully_synthesized_chain / partner_completion_flags 和 layout.observe 均支持这一语义。

当前 4,191,874 条具有双角色字段的 TCR 记录中，真正有双链观测 1,735,257 条，只有单链等非配对观测 2,456,617 条。后一组中 **398,331 条**仍有测试范围内的既有命中，按本版所声明的“非配对沿用测试范围内旧规则”本应被排除。这是确定的规则实现错误；其中 115,557 条含至少一个 exact 原因，其余为近似原因。命中还可能是 peptide 等旧规则原因，不能把这些数字都称为受体完全重复或配对重复。

| 来源 | 有合成占位链的非真实配对 | 其中仍有测试命中 |
|---|---:|---:|
| ots | 0 | 0 |
| trait | 8,926 | 7,981 |
| tcr_native | 9,356 | 8,694 |
| tcr_papers | 417,076 | 319,149 |
| tcr_repertoire | 2,021,259 | 62,507 |

OTS 全部 1,704,803 条保留记录都有双链观测，不受此分类错误影响；对其现有两个 CDR3 字段核查冻结的测试配对库，完全匹配重叠为 0。先前“4,200,307 条都是配对 TCR”的说法应修正为“4,200,307 条具有 α/β 两角色字段的输入记录”。

## 已通过与未覆盖项

- 全部 6,858,614 行、81 个当前分片的 SHA256、记录数和 split 字段检查通过；138 个上游绑定文件当前内容校验通过。
- train/valid 的 v7_family_group 交集为 0，缺失 family 行数为 0。这是既有 family ID 的隔离检查，不是新做的序列聚类。
- 1,726,279 条真实双链记录可直接读取两个现有 CDR3 字段，对测试配对库的完全匹配重叠为 0。
- 8,978 条 tcr_native 真实双链记录缺少完整的显式 CDR3 字段组合；本轮不推断、不做新注释，不能将未核查项当作零重叠。
- 测试源中仍有已登记的非标准 junction 等字段问题；按字段用途评估，不将 MHC 等位基因标签或 background 标签误报成需要修复的蛋白序列。
- 原 V7 的 264 条 benchmark H3 未确认项仍在。GDPa1/Specificity/m396 的专门保护执行证据与当前 T4 完整等价覆盖仍未建立；沿用历史来源审核的缺口结论，不据此断言这些数据全部泄漏。Kong/SAb 继续采用用户确认的旧 OAS 已处理来源。
- scripts/data/tcr_scope_gate.py 仍硬编码旧 all-support/query 范围，与用户现在的 test-only 口径不一致。不能为了通过而伪造旧范围证书，也不能只改门禁绕过数据问题。

## 后续修复边界

应以真实观测链区分双链/单链：真正双链保留同一测试配对 exact_ab 规则；只有单链观测的记录按测试范围内非配对规则处理。修复后从原 MHC 父版本导出新版本，保留当前有问题版本与审计证据，再按观测语义独立复核。缺失 CDR3、剩余 benchmark 覆盖和 H3 项需有对应证据才能完成整体审核。

本轮审核未改变数据或训练门禁，未生成通过证书、未更改 training_ready。结果为 **not_passed**，而非审核正在后台运行。没有恢复此前停止的注释分支。

[机器报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_pair_release_audit_20261002/report.json)、[全量日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_pair_release_audit_20261002/audit.log)、[审核脚本](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/audit_test_pair_release.py)。四项合成掩码边界自检通过；没有输出原始序列。
