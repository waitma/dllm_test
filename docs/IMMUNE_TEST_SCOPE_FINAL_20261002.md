# 测试集范围最终收尾

## 最终结果 — 2026-10-02T18:53:05+00:00

[最终数据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002)共 **6,460,277** 条：train **6,410,749**、valid **49,528**。状态 `training_ready=true`、`decontamination.status=passed`、`pending=[]`。最终清单 SHA256：`28e3cbf8fb860c13b514a11d600579257a233fefe9f70ea070b042d04f8a424c`。旧 r2 及旧证据保持原样，旧 pending 不应与此新版本混用。

8,765 条 alpha 核心字段按已有上游来源回填并复核，发现并删除 6 条 exact_test_cdr3_pair 命中；其余 8,759 条回填记录保留。新增剔除全部来自 train/tcr_native；OTS 保持 1,704,803 条。

| 来源 | 最终保留 |
|---|---:|
| oas | 2,497,359 |
| ots | 1,704,803 |
| asd_antibody | 169,381 |
| trait | 2,307 |
| tcr_native | 12,305 |
| tcr_papers | 115,370 |
| tcr_repertoire | 1,958,752 |

全量逐行复核确认：1,735,251 条保留的真实双链记录均可取得已有核心字段，未解决字段为 0，规则内完全配对/T4 条件键重叠为 0；train/valid family 交集为 0。原非配对过滤和旧 V7 检查基于已校验的父版本证据继承，输出仅删除行或回填已有标识，不改链序列、标签或划分。T4 参考范围与键覆盖的权威说明见[T4 任务记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。

训练入口实际调用 `examples.llada.immune_pretrain_v3.assert_ready` 已通过，包含当前分片、证书、测试输入及所有依赖证据的哈希复核；各来源 train/valid 均非空。没有启动训练。valid 的 trait 仍为 8 条、tcr_native 为 42 条，低样本限制保留，不通过移动或复制填充。

测试：15 项新规则/门禁/回填检查通过；44 项受影响回归中，pllm 通过 43 项，剩余 1 项仅因缺 sklearn 在现有 protenix_abtcr 环境补跑通过。合计 59 项通过，无共享环境依赖修改。

## 当前审核证明与使用

- [数据审核证明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002/audit_certificate.json)
- [下游范围审核证明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002/downstream_scope_certificate.json)
- [全量最终复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002/final_verification.json)
- [真实训练入口验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_scope_finalization_20261002/readiness_check.json)
- [回填与删除台账](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002/repair_ledger.jsonl)

后续显式使用 `--data-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002`。旧作业的旧数据路径没有被批量重写，历史证书不能替代此版本证书。`construction_report.json` 的 manifest_sha256 记录的是构建阶段待审核清单；最终通过状态的哈希以两份证书和 final_verification.json 为准。

范围严格限定：只保护已声明的实际测试/query；真实双链采用同一测试记录 CDR3α/CDR3β 完全匹配，非配对沿用父版测试范围规则；T4 按既有 beta-core＋epitope reference 条件键审计。未声称完成新的双链近似搜索或生物学新颖性认证。原 V7 264 条 H3、GDPa1/Specificity/m396 专门保护确认由用户明确排除，已写入清单和证书，不作为已确认通过的任务。


开始：2026-10-02T18:40:43+00:00。状态：已完成；按用户确认的 operational test-only 范围通过最终验收。目标：/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002。

从 r2 修复 8,765 条既有上游 alpha CDR3 标识，按同一测试记录的双链 CDR3 完全匹配重新过滤；不运行新注释/近似搜索。导入器 _recognition_identifiers 同步保留 cdr3a_core，避免后续再次丢失。旧版数据及证据保留。

T4 当前 ref_binders 均在历史 blocklist 中；实际目标定义与覆盖记录见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md。本轮还将扫描输出中 beta-core＋epitope 条件键残留；仅保护测试 reference，不保护 support-only train_binders。对无条件来源沿用已验证的父版本保护规则，不冒称全局单链近似隔离。

范围 ID：immune_test_only_observed_pair_exact_v3。用户排除的原 V7 264 条 H3、GDPa1/Specificity/m396 专门保护确认记录在清单与最终证明中；排除不代表已审核通过。

方法：继承哈希绑定父版本的非配对过滤与旧 V7 检查结果（其唯一 pending H3 由用户排除），对实际新文件重新检查回填对应关系、保留行内容、全量双链明确字段/完全重复、T4 条件键、family 划分及文件哈希。只有实际扫描通过后才写入对应范围的审核状态；不将工程通过当作生物学新颖性或双链近似隔离证明。

实现：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/finalize_immune_test_scope.py

训练入口：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/tcr_scope_gate.py（旧冻结版本仍走旧范围；新版本要求新范围、实际测试输入哈希、最终核查及明确排除项）。

日志：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_scope_finalization_20261002。当前新规则/门禁/回填 15 项测试通过；受影响导入器回归正在执行。
