# 训练接入支持版本更新及必要CPU检查

## 2026-10-01T12:17:53.844266+00:00 四候选私有接入新增检查完成

四候选正式训练接入的私有源码与新增CPU拒绝/身份检查完成，pllm/worker各24/24通过；1项静态来源缺口已修正。原R20 12函数AST/字节、6live源码、4runtime pins保持。清理30个自有控制文件/1118字节和10别名，验证checkpoint=0。未部署/入队/实际stage或准入执行，不代表架构效果；MHC无ready字段且双证书缺，真实小batch仍未启动。 [本次状态、SHA与限制](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CANDIDATE_TRAINING_BRIDGES_20261001.md)、[实际CPU原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/validation.json)、[清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/cleanup.json)。 本页下方R20九项与更早48/23项均是历史版本，不能替代本次四身份覆盖。

## 2026-10-01T12:01:52.464326+00:00 四候选正式训练接入准备

R21/R24/R26已具备各自正式API，当前补齐独立外部执行身份、来源/验收/续训绑定；4项私有源码准备进行中，CPU尚未执行，未部署/入队。原R20门禁和已验runtime保持，指定MHC缺双证书及ready绑定，真实小batch未运行。[本轮状态与边界](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CANDIDATE_TRAINING_BRIDGES_20261001.md)。

更新时间：2026-10-01T08:38:15.022797+00:00。私有源码准备及两个Python环境的聚焦接入检查已完成，正式接入尚未部署，真实语料训练尚未启动。继续原Goal `01a0f245-4559-7b62-9dc2-0917a8ed2267`，没有新建Goal、重启现有8卡容器/监听/worker或提交GPU任务。

## 问题与本次更新

此前的R20训练接入准备使用旧入口选择代码和旧启动隔离支持。三候选工程验收后，这些支持已有新版本，直接复用旧准备会丢失现有候选处理或遇到版本不一致。

本次在[新私有源码目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_candidate_support_rebase_cpu_20261001/isolated_worktree)将R20的固定训练入口选择叠加到当前支持源码；保留R21/R24/R26处理及D01、28个历史模型入口。R20训练接入的所有函数AST与上一已检查版本一致，包括原数据审核、双证书、preflight、训练及完整续训校验；只更新未来外部manifest位置。启动隔离源码与当前已验版本逐字节一致。没有改变模型、训练目标、attention后端或数据。

- [源码差异](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_candidate_support_rebase_cpu_20261001/source_rebase.patch)与[准备时的源码复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_candidate_support_rebase_cpu_20261001/source_review.json)。后者status为当时CPU待检查，最终结论以[验收记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_candidate_support_rebase_cpu_20261001/validation.json)为准，SHA `54557558bf890f7adef310b056e20d57d36773c766e5bbb9437afed96c07aa94`。
- 保持的R20源码包：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v3_20261001`；manifest SHA `cc13036fc492109afbf4c9cf9fd11a4c87684fc9702b83ce86c961b2a08294fe`。
- 新的未来manifest路径：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pool_launchers/t20_r20_pilot_support_v2_20261001/launcher_manifest.json`。该文件没有创建，当前不是训练准入凭证。

## 实际检查与适用范围

| 环境 | 本次案例 | 结果 | 原始证据 |
|---|---:|---|---|
| pllm / Python 3.10.19 | 9 | 全通过，2.80秒 | [日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_candidate_support_rebase_cpu_20261001/pllm_focused_attempt1.log)、[JUnit](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_candidate_support_rebase_cpu_20261001/pllm_focused_attempt1.junit.xml) |
| worker / Python 3.12.12 | 9 | 全通过，2.79秒 | [日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_candidate_support_rebase_cpu_20261001/worker_focused_attempt1.log)、[JUnit](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_candidate_support_rebase_cpu_20261001/worker_focused_attempt1.junit.xml) |

这9项检查覆盖三候选实际固定528源包选择（只读，不运行模型）、三个候选的普通train意图仍保持历史处理、缺少新训练准入绑定时在任何执行/输出之前拒绝、临时review stage绑定当前支持版本，以及普通CPU子进程继承当前源码隔离。运行中没有Torch模型、GPU训练、数据分片读取或审核证书生成。

此前48项/环境的R20训练接入与生命周期检查保留原证据，未无依据重复；本次全部launcher函数及原候选/runtime校验函数AST不变的比较，与新接入检查一起支撑本次结论。该结论只覆盖通用工程兼容性，不能证明真实语料准入或方法效果。R20/R21/R24/R26已通过的GPU工程验收也没有重跑。

## 清理

[清理清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_candidate_support_rebase_cpu_20261001/cleanup.json)：两个本次自有临时目录已删除，共14文件/137742字节；checkpoint为0，实际训练恢复状态未修改。保留私有源码、源码差异、JUnit/日志及版本核对证据。

## 尚未完成与接续

本次私有入口准备仍仅适用于R20控制，三候选各自的训练API不等于已经有已部署的外部池训练接入。后续训练整合仍需逐候选明确版本绑定、接入及原审核条件；不能修改已验固定运行包或用R20通过记录代替其他候选。

当前数据仅核实metadata：manifest SHA `601db781ebf81ee857dde98ddcfb2d1da8e121a4c57ff2d80c864d420a4ebfce`，training_ready=false、audit_certificate.json与downstream_scope_certificate.json均缺。真实1000→累计5000与最终预训练未启动；有效审核/version binding未具备之前不能发布准入pointer或运行真实语料。此前安全检查停止的序列注释分支不恢复，独立CUDA算子测试保持暂停。

2026-10-01T09:37:59+00:00接续更新：[新CV联合能力原型](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_CV_JOINT_PROTOTYPE_20261001.md)的8种实现和CPU功能验收已完成，GPU入口/500步曲线待完成。它与本页每环境9项接入检查分开计数，不改变本页私有训练支持未部署及真实数据未准入状态。

当前交接入口：[IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)；总账：[PROJECT_PROCESS.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。
