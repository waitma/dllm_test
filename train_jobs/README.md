# Training job configurations

本目录保存近期训练配置、当前研究实验目录和仍被看护脚本读取的旧配置。任务是否已提交或完成，以 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md) 为准，文件存在不表示正在运行。

2026-10-01 03:26 UTC：当前目标是预训练整合并启动最终版本训练。原八卡池已经部署固定runtime/D01；实际v1因oracle阶段预算失败，正在准备新不可变v2和T20实际语料入口。数据双证书仍缺，正式训练未开始。[实际整合记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PRETRAIN_INTEGRATION_20261001.md)。下一版本protocol.yml仍是不可提交规划文件，不能把execution_enabled=false改成已有执行器。

- v2 / immune_v6 配置及 v3 profile 模板保留在本目录。
- 当前实验入口：[queue012 research](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_queue012/README.md)、[queue012 explore](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_v3_explore_queue012/README.md)、[research pool](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_pool_c20250601/README.md)。
- `protein_esmc_llada270m_diffusion_immune_v3_spot_2m.yml` 与 `protein_esmc_llada270m_bert_immune_v5_8gpu_spot.yml` 被 [monitor_spot_tasks.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/monitor_spot_tasks.py) 引用，继续保留。

早期训练配置已移至 [2026-09-30 归档索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/README.md)。按文件名查询该处清单，可取得原路径、归档路径及 SHA256；旧文档中的原提交命令是历史记录。归档配置需重新核对数据、权重和运行环境后才适合复用。
