# 测试集保护与 CDR3 配对完全重复重构

## 后续语义审核未通过 — 2026-10-02T17:05:42+00:00

此前工程验收通过不等于清理条件正确；现已发现合成占位链误判为真实配对的问题。当前版本仍未通过整体审核。以[本轮审核及全量影响统计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_AUDIT_20261002.md)为准，原构建数字保留为历史事实。

## 完成结果 — 2026-10-02T16:56:15+00:00

[新版本](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_pair_exact_20261002)：共 **6,858,614** 条，train **6,806,404**、valid **52,210**。从源版本 6,901,815 条删除 43,201 条；相比旧 expanded candidate 恢复 **2,076,771** 条。OTS 删除 **244** 条、保留 **1,704,803** 条；全部 4,200,307 条配对 TCR 中删除 exact_ab 命中 8,433 条，非配对 ASD 删除 34,768 条。

| 来源 | 输入 | 删除 | 保留 |
|---|---:|---:|---:|
| oas | 2,497,359 | 0 | 2,497,359 |
| ots | 1,705,047 | 244 | 1,704,803 |
| asd_antibody | 204,149 | 34,768 | 169,381 |
| trait | 10,639 | 351 | 10,288 |
| tcr_native | 22,614 | 1,609 | 21,005 |
| tcr_papers | 440,748 | 6,229 | 434,519 |
| tcr_repertoire | 2,021,259 | 0 | 2,021,259 |

四项判定边界测试通过，81 个来源分片全部处理完成；138 个唯一绑定路径逐项校验通过，所有输出分片均已重新读取核对 SHA256 与行数。独立验证器另按源行序与原命中表逐条重算选择，逐字节比较全部保留行并核对删除台账，已通过。

[清理报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_pair_exact_20261002/cleaning_report.json)、[独立选择/字节复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_pair_exact_20261002/selection_verification.json)、[删除台账](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_pair_exact_20261002/removed_rows.jsonl)、[清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_pair_exact_20261002/dataset_manifest.json)。清单 SHA256：`8b7818b3ceec2863f2bf3634de512e0f8fdb681743fd7c80b189a173d42b9c1c`。

此次完成的是明确规则下的数据重构及工程验收。清单仍为 audit_pending / training_ready=false；既有 H3 与当前 benchmark 覆盖审核未在此轮解决，也没有进行双链近似搜索。旧冻结版本、官方 benchmark、训练配置和训练状态均未改。后续若改用双链近似，同一配对的双链共同近似必须另有记录级证据，不能再用 cohort 交集代替。


开始时间：2026-10-02T16:46:51+00:00。状态：导出及独立逐行选择/字节复核已完成，数据目录已发布；仍未取得整体训练审核认证。

来源：/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1。目标：/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_pair_exact_20261002。不修改旧 V7、旧 expanded candidate 或官方 benchmark。

## 本轮采用的规则

只保护实际测试/query；从既有 24 个 cohort 排除 7 个明确 train/valid cohort。paper6 background/target 与 clustering 集参与实际 query，保留保护。用户尚未回复完全重复与双链近似的选择，本轮按已告知的默认值构建 CDR3 配对完全重复版本；不冒称用户明确选择了该阈值。

具有 tcr_alpha 与 tcr_beta 的记录只按既有 exact_ab 证据删除：同一测试记录的两个 CDR3 分别完全相同。单链完全相同、单链近似、两链分别命中不同测试记录、仅共享 peptide 均不能独立触发配对记录删除。不是完整链完全重复，也不是双链 80% 近似筛选。

非配对记录沿用旧匹配方法，但将命中 cohort 限于实际测试范围。此项与配对规则分开记录。原 V7 已有排除继承不变，本轮恢复的是后补 expanded 过滤造成的过度排除，不声称追溯重建 V7 之前全部过滤。

## 工程验收与审核边界

四项判定边界测试通过。导出器逐文件校验旧证据与父版本 SHA256、行数、MHC 前后非 MHC 内容/行序、数据结构及 split；保留行字节不变。写完后再次读取全部输出文件核对 SHA256 与行数，再发布版本。原 train/valid 不移动，因此不引入新的跨划分 family 重叠。

新增 removed_rows.jsonl 使用原分片非空行的零起始索引，记录删除原因与 cohort bitmask；保留记录的映射可由原行序减去删除索引重建。清单提供规则、源清单/旧证据/删除台账/导出器哈希。

这些是既有命中证据下的筛选与文件验收，不是新的序列扫描或全 benchmark 覆盖认证。新版本仍保留 audit_pending；原 V7 H3、字段语义、当前 benchmark 覆盖与来源限制不能因重新导出自动消除。不恢复此前停止的序列注释分支、不修改 ready、不启动训练。

实现：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/export_immune_test_pair_clean.py

测试：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_test_pair_clean_export.py

运行日志：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/dedup_test_only_review_20261001/rebuild_test_pair_20261002.log
