# 真实观测链修复与 r2 重构

## 后续最终版本已通过确认范围审核 — 2026-10-02T18:53:05+00:00

本文早先待办为历史状态；字段回填、T4 覆盖与入口规则已在新最终版本收尾。数据路径、证书及限定范围以[最终验收记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_SCOPE_FINAL_20261002.md)为准，旧版本文件不变。

## 8,765 条 alpha 核心字段的来源已追溯 — 2026-10-02T18:19:18+00:00

全部来自 tcr_native 中的 tcr_pmhc_fulllength 子来源，上游 provenance 全部为 fulllength:vdjdb（不是 McPAS 或 OTS）；当前 train 8,735 条、valid 30 条。对当前记录与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_native/unified/fulllength.csv 的既有 alpha_fv/beta_fv/epitope 三字段做精确来源连接：8,765/8,765 均匹配，上游 cdr3a 全部非空，且每条对应值唯一。[机器证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_pair_release_rebuild_r2_20261002/missing_alpha_origin_review.json)绑定当前清单与上游 CSV 哈希，仅输出计数/行号/来源 ID，不输出序列。

来源链：VDJdb → build_fulllength_tcr_pmhc.py 的历史 Stitchr/thimble 衍生记录 → data/tcr_pmhc_fulllength/records_train.jsonl → ingest_repo.py 已有导入结果 data/tcr_native/unified/fulllength.csv → tcr_native prepared → V7/MHC/r2。此为现有产物追溯，没有执行这些历史构建/注释步骤。

直接原因见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/sources.py：_receptor_chains 对已有完整 Fv 直接创建不含 regions 的链；_recognition_identifiers 仅保存 cdr3b_core，没有 cdr3a_core。所以是 prepared 序列化时未保留已有字段，不是原始来源没有 alpha CDR3。先前“缺少 alpha 核心字段”的描述限定于当前 prepared，不能推导成上游无数据或需要重做注释。

当前状态已由“来源未确认”推进为“可从既有字段恢复；待实际回填和配对审核”。本轮只回答来源并保存证据，未改数据/冻结清单、未宣称配对审核已通过。

## 用户缩减必需审核范围 — 2026-10-02T18:15:56+00:00

用户明确不再要求确认：原 V7 的 264 条抗体 H3 未确认项，以及 GDPa1、Specificity、m396 专门保护执行证据。本轮后续审核中将这四项作为用户明确排除的范围，不再作为阻断项；排除不等于已完成确认或通过。原始审计报告和冻结清单保持历史原貌，当前有效待办以本条及[机器可读范围修订](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_pair_release_rebuild_r2_20261002/user_scope_amendment.json)为准。

仍需完成：8,765 条 TCR 的 alpha CDR3 核心字段证据/处理方式；当前 T4 测试范围覆盖；训练入口 test-only 规则对齐与绑定新版清单的最终审核。用户没有排除这些项目。本次仅修订必需审核范围，不改变数据、评测任务或 training_ready。

## 完成结果 — 2026-10-02T17:40:43+00:00

[新版数据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_pair_exact_20261002_r2)保留 **6,460,283** 条：train **6,410,755**、valid **49,528**。清单 SHA256：`77b91a4b3f2a67f2bc1491589a1850392141f33c68ebb30ec21c94ff6c9a3108`。

| 来源 | 保留 | 相比 r1 追加删除 |
|---|---:|---:|
| oas | 2,497,359 | 0 |
| ots | 1,704,803 | 0 |
| asd_antibody | 169,381 | 0 |
| trait | 2,307 | 7,981 |
| tcr_native | 12,311 | 8,694 |
| tcr_papers | 115,370 | 319,149 |
| tcr_repertoire | 1,958,752 | 62,507 |

相比 r1 的追加删除总计 **398,331** 条，逐行删除台账集合差已核验，按来源与上次语义审核的漏过滤记录数一致；新增删除全部属于非真实配对。r1 原有删除全部保留，真实配对删除记录 ID 集合完全不变，OTS 删除记录 ID 集合完全不变（仍删除 244、保留 1,704,803）。

14 项不同边界测试通过（10 项 v2 与 4 项原范围测试）。独立逐行字节/选择复核、81 个分片读回及 138 个原绑定校验通过；新清单额外绑定观测策略源码哈希。全量语义复审确认非配对记录保留的测试命中为 **0**，family 跨 train/valid 的组数为 **0**。1,726,492 条可用既有核心字段核查的真实配对，对冻结测试配对库的完全匹配重叠为 **0**。没有新做注释、相似性搜索、family 聚类或修改原划分。

原 8,978 条字段不足记录中，213 条可用已有 cdr3b_core 标识加上现有 alpha CDR3 字段核查，现已纳入复审；其余 **8,765** 条仍缺 alpha 核心字段。原 V7 H3 与当前 benchmark 覆盖证据、训练入口旧 all-support/query 范围仍待处理。`fix_acceptance.status=observed_partner_bug_fix_passed` 仅代表本次缺陷修复通过；`audit_report.status=not_passed` 与 `training_ready=false` 如实保留，不能解释为整体数据已认证。

证据：[修复验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_pair_release_rebuild_r2_20261002/fix_acceptance.json)、[全量语义复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_pair_release_rebuild_r2_20261002/audit_report.json)、[独立逐行复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_pair_exact_20261002_r2/selection_verification.json)、[清理报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_pair_exact_20261002_r2/cleaning_report.json)、[既有字段补查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_pair_release_rebuild_r2_20261002/missing_explicit_fields_review.json)。日志位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_pair_release_rebuild_r2_20261002。


开始：2026-10-02T17:24:16+00:00。当前状态：r2 已发布；配对误判修复及其独立复审通过，整体训练审核仍未通过。

## 规则与实现

用户授权继续上一轮审核后的修复。仅保护实际测试/query；真正有 α、β 两条观测链的记录采用同一测试记录的 CDR3 配对完全重复规则。任一链 synthetic_residue_mask 全 1 或序列为空，均不能作为观测伙伴；因此单链记录恢复采用本版已声明的非配对测试范围匹配规则。部分合成框架但仍有观测残基的链不被当作全合成占位链。掩码长度/取值异常直接停止，不能默默按双链保留。

本次复用已有观测掩码定义，不新建序列注释或相似性搜索。从原 MHC 版本重导，不在旧候选上继续删改。旧脚本、旧数据及旧报告保留。

新规则 ID：test_only_observed_cdr3_pair_nonpaired_legacy_v2。目标：/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_pair_exact_20261002_r2。

导出器：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/export_immune_test_pair_clean_v2.py

独立逐行验证器：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/verify_immune_test_pair_clean_v2.py

观测语义审核器：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/audit_test_pair_release_v2.py

13 项测试通过（9 项新观测边界与 4 项原测试范围/配对边界）。导出后将分别执行独立字节/选择验证与全量观测语义审核，核对旧版 398,331 条漏过滤影响得到消除，并确认 OTS 保留数量不被改变。

日志与机器报告目录：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_pair_release_rebuild_r2_20261002。

## 审核边界

此次修复不能自动关闭 H3、当前 benchmark 完整覆盖、8,978 条缺少完整显式 CDR3 字段组合及训练门禁旧范围问题。缺少来源证据时不生成通过证书，不修改 ready，不启动真实训练。
