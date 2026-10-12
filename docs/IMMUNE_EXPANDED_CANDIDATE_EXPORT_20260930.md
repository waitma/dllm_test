# Expanded 候选语料物化记录

更新：2026-09-30 22:08 UTC。状态：锁定行号清单的候选物化及独立软件完整性检查已完成；候选仍明确不可训练。此记录只涉及既有行 ID 的筛选、字节完整性、metadata 和行号映射；不执行新序列注释、近邻搜索、序列角色推断或训练认证。

## 已核验的输入

[全量 verify 报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/expanded_export_verified_plan.json) 的 SHA256 为 `1b74fbbe97bcac06b63dfb0ddbd92858ea8b0a9ff6110a052314460d66b404dc`。138 个唯一绑定文件全部 verified，pending binding 为空；81 个原分片的进度记录与报告聚合一致。独立 [preflight acceptance](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/expanded_export_preflight_acceptance.json) 将绑定集与当前锁定输入重新生成的绑定集逐项比较，并复核行数、来源、split、family 报告与假设迁移算术。

总输入 6,901,815 条，严格并集移除 2,119,972 条，保留 4,781,843 条。其中有 exact 理由的移除为 343,422 条，仅有 edit80 理由为 1,776,550 条；两者不可混称精确重复。候选实际保持以下原划分：

| Source group | 保留 train | 保留 valid | 保留 valid family |
|---|---:|---:|---:|
| OAS | 2,472,502 | 24,857 | 20,683 |
| ASD antibody | 167,391 | 595 | 66 |
| OTS | 77,252 | 530 | 511 |
| TCR native | 646 | 4 | 4 |
| TCR papers | 81,293 | 732 | 305 |
| TCR repertoire | 1,936,013 | 19,173 | 19,114 |
| TRAIT | 851 | 4 | 4 |
| 合计 | 4,735,948 | 45,895 | 不跨来源相加为全局 family 数 |

`family_split_overlap=0`。报告生成了一个含 175 个整 family 的可行性示例，只用于数量分析，**未执行任何迁移**。TRAIT/native 的低样本与低 family 数保留为明确限制，不通过单行移动或复制补 valid。以上不是清理认证；原有 annotation、独立重扫和双证书 pending 均保留。

## 候选导出

目标目录：[immune_v7_expanded_candidate_20260930](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_expanded_candidate_20260930)。启动前已确认目录及同名 symlink 均不存在。使用已验收的 [exporter](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/export_immune_expanded_clean.py)，没有改变策略、阈值、输入清单或脚本：

```bash
USE_TF=0 python -u scripts/data/export_immune_expanded_clean.py --mode export --min-valid-rows 100 --scratch-dir output/immune_data_readiness_20260930 --output-dir data/prepared/immune_v7_expanded_candidate_20260930 --report output/immune_data_readiness_20260930/expanded_candidate_export_report.json
```

启动时间约 21:52 UTC，exec session `24614`，Python PID `1959044`。日志：[expanded_candidate_export.log](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/expanded_candidate_export.log)。原数据只读；输出先写新 staging 目录，核验完成才发布候选目录。每条保留记录沿用当前 MHC 行的 JSON 内容；所有原记录进入新旧索引映射，删除理由保留原 cohort bitmask。

候选 manifest 顶层与 decontamination 均为 `training_ready=false`、`audit_pending`。不会复制或生成 `audit_certificate.json` / `downstream_scope_certificate.json`。没有启动真实训练。

## 导出后的独立检查

已执行 [check_expanded_candidate_artifacts.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/check_expanded_candidate_artifacts.py)。独立检查 exit0，用时 80.815 秒，完成以下检查：

1. 独立重算所有候选分片 SHA 和非空行数，核对来源/split 聚合及父/MHC 来源 SHA。
2. 顺序检查完整 row mapping，验证每个原索引恰好出现一次、每个保留新索引连续，并逐条比较删除行与锁定 `row_hits` 的行号及理由。
3. 创建并重新验证固定 seed42、每来源最多256记录的 [validation selection 工具](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/validation_selection.py) 产物，实测 PreparedImmuneDataset 索引对应。
4. 调用既有 `assert_ready`，确认候选仍因缺少认证而拒绝；该操作不启动训练、不改门禁。

导出 exec `24614` 和独立检查 exec `34848` 均 exit0。候选共 81 个独立分片、9,136,609,607 bytes；row mapping 为 6,901,815 条、1,354,009,330 bytes。每个原行号只出现一次，保留新索引连续，全部 2,119,972 个删除映射与锁定命中清单逐条一致。原 prepared 和 benchmark 未修改；没有新注释、相似度重算或 family/split 迁移。临时 SQLite 和 staging 已自动清理，没有 checkpoint。

- 新 manifest SHA256：`601db781ebf81ee857dde98ddcfb2d1da8e121a4c57ff2d80c864d420a4ebfce`。
- row mapping SHA256：`23ce1f78987860e98243f3a418f2c15544c84f823198ef7d0cfec9a3517a3fe0`。
- selection 内容 SHA256：`359f8b702f692a0ecc6890d9fe44aded47dfef699deb6375e87ccf00ee054a41`。

最终证据：[export report](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/expanded_candidate_export_report.json)、[artifact validation](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/expanded_candidate_artifact_validation.json)、[独立检查日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/expanded_candidate_artifact_validation.log)。

## 固定验证子集与训练门禁

[validation selection](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/expanded_candidate_validation_selection.json) 共 1,288 条，585,691 bytes，已重新生成并逐字段验证；所有来源的 PreparedImmuneDataset 总数、所选首尾索引和既有 family 元数据检查通过。

| Source group | 选中记录数 | 选中已知 family 数 |
|---|---:|---:|
| ASD antibody | 256 | 50 |
| OAS | 256 | 254 |
| OTS | 256 | 250 |
| TCR native | 4 | 4 |
| TCR papers | 256 | 186 |
| TCR repertoire | 256 | 256 |
| TRAIT | 4 | 4 |

TRAIT/native 的 available/selected records 与 family 数不足分别保留明确警告。选中行无需补造或改 split；这些数字也不构成验证效力证明。

实际调用现有 `assert_ready(candidate)` 的结果为 `FileNotFoundError: audit_certificate.json`，符合未认证候选的预期。候选中两种证书均不存在，`training_ready=false` 与所有原有 pending 保持不变。**字节与索引验收通过不能代替新 manifest 的独立去重审计、尚未解决的角色/注释问题或训练认证。没有启动真实训练。**
