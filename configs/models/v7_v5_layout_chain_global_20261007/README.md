# 当前新训练版本

**数据改用V5**：/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v5_receptor_completion（train 7,626,885；valid 102,308）。技术目录名保留不变，数据身份以version.json为准。新训练已在持续八卡容器内从头开始。

| 项目 | 配置 |
|---|---|
| 模型 | 8层／768维／12头／FFN3072，紧凑50词表，无ESMC，随机初始化 |
| 注意力与残差 | 蛋白对象内／全局交替，残差系数1.0 |
| 语法 | context→关系→受体对象；双链用`.`连接 |
| 生成容量 | 对象容量302＝166+134残基＋split＋protd；学习`.`及`<protd>`，无需真实生成长度 |
| CDR-only | 保留V5已有全长scaffold，synthetic位转mask、不计损失 |
| 范围 | 抗体H/L、TCR；不纳入VHH |

完整当前规则见[BIOSEQ_MODEL_PLAN.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md)和[DATA_FORMAT_AUDIT.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md)。不再在本文件重复全部格式和实验日志。

模型配置：/vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/research_config.json。
数据及状态：/vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/version.json。
输入／生成接口：[downstream/v5_layout.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v5_layout.py)，支持整体生成或已知前链→后链；反向补链需另定协议。

97项代码测试通过；V5全部7,729,193条容量及格式检查通过，最大对象299、最大总输入1218，未发现VHH或需要裁剪的记录。数据及启动源码已绑定，原生门禁再次验证85分片内容哈希。

[启动包](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v1/README.md)包含冻结源码、配置、数据绑定、参数数组及预检回执。实际训练仍需八rank launcher及batch／预算／资源调度，已提交新八卡任务，但首次启动在NCCL后失败；正式0步。[全量报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/full_audit.json)。

2026-10-07运行状态：/vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/training_config.json 的超参数已与实际提交清单逐项核对一致；training_submitted表示有过提交，formal_training_started=true表示已经产生正式更新步。状态及容器记录以 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 为准。
