# Expanded strict-union 候选导出器

更新：2026-09-30。此工具只负责既有逐行命中清单的版本化应用和软件完整性检查；不会解释新的生物序列注释、重新计算近邻、签发证书或将数据标为可训练。当前 annotation 审查仍有 80 项 pending，不能把软件测试通过写成该缺口已解决。

实现：[export_immune_expanded_clean.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/export_immune_expanded_clean.py)。测试：[test_expanded_clean_export.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_expanded_clean_export.py)。格式兼容：[PreparedImmuneDataset](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/dataset.py)。前置状态见 [IMMUNE_TRAINING_DATA_READINESS.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TRAINING_DATA_READINESS.md)。

## 三个独立模式

| 模式 | 实际读取与验证 | 写入内容 | 不包含的结论 |
|---|---|---|---|
| `plan`，默认 | SHA 绑定的小型 manifest、plan、audit report、registry、MHC source report；按已有审计 totals 估算 | 指定的新 report | 不验证逐行命中、分片字节或 family 成员 |
| `verify` | 所有直接依赖绑定 SHA；逐行扫描 MHC 和父版本；流式合并 row_hits；SQLite 聚合 family | 指定的新 report 和自动清理的临时 SQLite | 不运行新相似度审计，不生成新 prepared 数据 |
| `export` | 与 verify 相同 | 全新候选目录：筛选后分片、逐行 mapping、manifest、cleaning report | 不搬动 split，不复制旧证书，不写新证书 |

默认输入固定为当前 MHC 与 clean_r2 expanded 审计计划，并 pin 两个入口 SHA；换输入必须显式传入新的 SHA。已存在的 report 或候选目录会拒绝覆盖，候选与 report 不得落在原 prepared 目录中。API `load_evidence(...)`、`plan_report(...)`、`verify_or_export(...)` 可供后续受控流程调用。

在仓库根目录先激活 `pllm`，元数据预览：

```bash
USE_TF=0 python scripts/data/export_immune_expanded_clean.py --mode plan --report /new/path/expanded_plan.json
```

全量只读核验：

```bash
USE_TF=0 python -u scripts/data/export_immune_expanded_clean.py --mode verify --min-valid-rows 100 --scratch-dir output/immune_data_readiness_20260930 --report output/immune_data_readiness_20260930/expanded_export_verified_plan.json
```

`--min-valid-rows 100` 仅为可行性分析阈值，不是有效验证集的充分条件，也不是训练门禁。`verify` 无 `--output-dir`；在主线程确认物化时，才使用 `--mode export --output-dir /new/candidate`。

当前输入 138 个唯一绑定文件，按文件 stat 合计 14,969,695,272 bytes，包括去掉 symlink 重复的 prepared 分片、230,657,069-byte row_hits、benchmark 和 MHC provenance。单进程 CPU 流式处理；family 累计使用约 10k key 缓冲及磁盘 SQLite，临时库应预留数 GB，优先使用共享 scratch。每完成一个分片输出进度。准确运行耗时和 family 数量等待真实 verify；不能用 tiny 测试耗时估算全量性能。

## 筛选与可追溯性

唯一支持策略为 `strict_union_all_declared_exact_and_edit80_v1`：删除任一已声明 exact 或 edit80 理由命中的行。逐行去除理由保持原 cohort bitmask，输出保存 cohort bit 顺序；`exact_reason_rows` 与 `near_only_rows` 单独计数，80% 编辑相似度不称为精确重复。禁止未知理由、空理由、越界 bitmask、重复/乱序命中和超出分片范围的行号。索引始终为分片内从零开始的非空记录编号。

MHC 当前/父 manifest 必须具有完全相同的分片成员、顺序、来源、split 和行数；旧审计逐分片 SHA 与父 manifest 一致。流式扫描对每行比较所有非 MHC 链序列、按链索引的 regions/metadata、来源、标识、权重、标签、split、非 MHC metadata 及未知顶层扩展字段。MHC chain 数量变化导致的索引位移按语义链重映射检查；保留的记录沿用原 JSON 行内容，序列和元数据不经重新序列化或截断。只有文件缺少结尾换行时补足行分隔符。

候选分片仍使用原相对分片名，但根目录全新、字节独立写入；空分片不写入新 manifest。`row_mapping.jsonl` 覆盖所有原记录，保留 `old`/`new` 分片与零基行号、`source_group`、`split`、`family_group`、原 `reasons`。被移除行的 `new=null`，保留行的理由为空对象。记录本身完整保留 MHC、metadata、`v7_family_group` 及原 source；manifest source group 不替换行级来源。

新 manifest 的 `mhc_update` 保留为历史来源信息，候选的删选关系另由 `expanded_cleanup` 描述；候选已不具有原版本完整行成员集合。`expanded_cleanup` 绑定父/MHC manifest、旧 audit/hits、row mapping 和 exporter 源码 SHA。manifest 顶层与 `decontamination` 的 `training_ready=false`，`decontamination.status=audit_pending`。不能对这个候选直接套用假设完整父行成员的旧 MHC verifier，更不能继承旧证书。

## 验证集的整 family 可行性

SQLite 记录删选前和删选后的 family/source/split 行数。若原 family 跨 train/valid，立即拒绝；即使发生交叉的行会被删除，也不会掩盖原始划分问题。

报告给出每个来源剩余 train/valid 行数、family 数和所选阈值下的缺口。建议移动时整组包含该 family 在所有来源中的剩余行，并要求每个来源至少保留 1 条 train；程序只构造一个确定性的可行示例，永不执行移动。没找到该示例时报告 `no_constructive_solution_not_a_proof_of_infeasibility`，不会误称严格数学不可行。数量满足不代表独立性、多样性或评测效力已验证。

已有 [元数据预览](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/expanded_export_metadata_plan.json) 明确标为 `estimated_not_row_verified`：总保留 4,781,843 条，TRAIT 与 TCR native 的 valid 均只剩 4 条，family 可行性仍待全量行扫描。原来的单行随机补 valid 不在工具中提供。

## CPU 验收与当前边界

58 tests passed，3.07 秒。覆盖小型 plan 不读取大文件、绑定篡改拒绝、MHC 变化/索引位移、非 MHC 字段及行序保护、非空行编号、命中理由与 bitmask、family 跨 split、整 family 建议不执行、原目录不变、候选目录冲突、空分片及 PreparedImmuneDataset 真实加载。所有测试数据在临时目录人工构造；没有生成真实可训练证书或 checkpoint。

证据：[测试日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/expanded_export_pytest.log)、[JUnit XML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/expanded_export_pytest.xml)、[验收 JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/expanded_export_validation.json)。以上为工具的 CPU 验收；随后主线程完成真实 `verify`，并授权本分支将锁定行号筛选物化到新的不可训练 candidate。导出与独立 SHA/行号映射/selection 检查的实测结果另见 [IMMUNE_EXPANDED_CANDIDATE_EXPORT_20260930.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EXPANDED_CANDIDATE_EXPORT_20260930.md)。没有改 split、签证、改变 GPU 指纹文件或启动训练；CPU 测试与产物字节完整性均不能替代去污染认证。
