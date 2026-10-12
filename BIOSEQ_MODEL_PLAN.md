# 当前模型方案

更新：2026-10-10T08:12:07.932612+00:00

本轮采用**V5数据集＋约270M新模型**（实际268,590,112参数）；前一轮75.6M已完成。选择V5数据不等于恢复旧V5的ESMC、大词表或旧模型权重。

| 项目 | 当前决定 |
|---|---|
| 数据 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v5_receptor_completion |
| 范围 | 常规抗体H/L、TCR；不纳入VHH |
| 训练预算 | 300,000更新步；global batch512=8卡×micro64×累积1；LR1e-4、warmup2000、cosine；八卡synthetic显存检查已通过 |
| 初始化 | 随机初始化，无ESMC |
| 主干 | 16层、1024维、16头、FFN 4096，紧凑50词表；268,590,112参数 |
| 残差 | 系数1.0，取消原0.1缩放 |
| 注意力 | 偶数层蛋白对象内全注意力，奇数层全局；对象由prots/protd划分 |
| 输入顺序 | V5对象布局：context→关系→受体，双链用`.`分隔 |
| 生成 | 对象容量302（原固定decoder总槽位167+135），模型预测`.`及`<protd>` |
| 长度特征 | 对象内位置连续，`.`只分隔输出序列，不切换attention段或重置RoPE |
| CDR-only | 保留已构建全长scaffold；synthetic位置输入mask、不计损失 |

当前配置：/vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v5_noesmc_270m_20261010/research_config.json；参数预算与数据身份见同目录training_config.json及version.json。下方带2026-10-07日期及launch_v1/v2/v3的说明属于75.6M历史记录。

基础实现原有97项CPU测试、V5全部7,729,193条布局审核保持；270M新增五项门禁CPU检查及八卡synthetic通过，启动时85分片完整哈希通过，现已从头开始正式训练；270M已提交验证最佳32000步的37项下游评测，尚无完成成绩；状态见PROJECT_PROCESS.md。报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/full_audit.json，启动入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v1/README.md。

支持整体生成及已知前链→生成后链；已知后链反向补链需另定条件协议。实验记录与更正只维护在[实验记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/EXPERIMENT_LOG.md)，不在本文件重复堆叠过程日志。

详细接口：[configs/models/v7_v5_layout_chain_global_20261007/README.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/README.md)。旧方案与完整推导：[完整历史](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/project_docs_20261007T061603Z/BIOSEQ_MODEL_PLAN.md)。

运行配置：/vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/training_config.json；完整启动参数：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v1/launch_training_args.json。步数以历史下游150,000步为参照翻倍，样本处理量为原150,000×256的四倍；首次直接提交启动失败，后已在持续八卡容器内从头开始正式训练。

2026-10-07启动整合：当前入口为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v5_object302_scratch_8gpu_20261007.yml，源码包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v2。初始化明确为随机从头训练，所有模型参数可训练；新增split_loss/split_accuracy用于单独统计`.`，总验证CE仍覆盖全部合法监督token，不改训练loss。

运行超参数以 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/training_config.json 为准；当前参数清单逐项匹配。launch_v3为当前容器内启动版本，保留训练目标、模型主体、数据及300000/global512配置，已通过八卡synthetic和通信启动检查并开始正式训练；已完成多轮验证/保存；最新步数与验收见 [PROJECT_PROCESS.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。

2026-10-07历史loss对照口径：当前V5新no-ESMC与已定位旧V6 binding-only no-ESMC的数据、目标token、遮蔽协议、loss聚合及样本预算不同，暂不据原始曲线更改架构或宣称收益。训练loss更低和全遮蔽验证loss更高均不足以直接排名；同口径对照要求与证据见 [V5_CURRENT_VS_LEGACY_NOESMC_LOSS_20261007.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V5_CURRENT_VS_LEGACY_NOESMC_LOSS_20261007.md)。本次只读对照，不新增评测任务。


### 2026-10-09 评测修复

下游单β生成采用显式单链词表约束（禁止dot）的独立v2评测协议；这是推理协议修复，不是训练架构/布局改动。训练中的蛋白对象内attention、可学习split与protd保持原样。

## 2026-10-10 用户要求扩大到270M

新实验使用16层、1024维、16头、FFN4096，CPU实例化核实268,590,112个可训练参数（约270M）；旧模型实际75,601,176参数。保留50词表、no-ESMC、V5数据、对象容量302、对象内/全局交替attention、残差1.0、原mask目标及验证口径，从头初始化。配置 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v5_noesmc_270m_20261010/research_config.json。

训练预算保持300000步/global512/LR1e-4/warmup2000/cosine，候选micro64×GA1×8卡（同global预算下最大micro）；先在原持续八卡容器验收整模synthetic前后向/显存与通信，再用独立票据正式启动。新数据绑定门禁仅将架构固定值切换为新尺寸，85分片内容哈希、原布局审核与training_ready停线保持；未重写数据审核或改旧冻结包。原75.6M产物保留，用于同口径对照。

270M实际已开训；以 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v5_noesmc_270m_20261010/training_config.json 和运行包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_270m_train_bs64_20261010 为当前入口。micro64/GA1保持global512，GPU验收和原生数据门禁均通过；首个训练日志loss有限。模型维度以外的数据、目标、注意力模式、验证采样及总训练预算继承75.6M；微批次/累积变化需在比较训练loss时注明。
