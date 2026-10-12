# Evaluation job configurations

本目录保留当前 v2 / immune_v6、AB–抗原评测配置，以及生成器和历史结果校验仍需读取的文件。任务状态见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)，评测入口与协议见 [任务索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/README.md)。文件存在不表示已经提交或正在运行。

以下旧文件有代码依赖，继续保留原路径：

- [eval_immune_270m_42000_pairing_reference_v3.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_immune_270m_42000_pairing_reference_v3.yml)：AB / v2 评测生成器读取的模板，也是源码发布检查的必需文件。
- [eval_tcr_others_longab_v5_29000_nonspot.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_longab_v5_29000_nonspot.yml)：TCR 原生评测生成器读取的模板。
- `eval_ab_v5_49000_pairing_iters_p0_20260916_iter{8,16,32,64,96,128}.yml`：六份配置用于 [pairing_iter_sweep.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/pairing_iter_sweep.py) 的历史身份和哈希核验。

早期 v5 矩阵、重试/参数扫描及旧通用评测配置已移至 [2026-09-30 归档索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/README.md)。按原文件名查清单即可定位。旧配置保留原始协议与路径，不代表可直接作为当前任务模板；当前实验应使用相应生成器和任务手册。
