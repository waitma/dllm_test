# 当前V5新no-ESMC与历史no-ESMC的loss核对

更新：2026-10-07T23:49:46.004508+00:00。只读历史日志、配置和源码；未加载模型或序列、未重跑评测、未改当前训练。

## 对照身份与结论

本次找到的旧no-ESMC长训练为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m，实际argv数据为immune_v6_binding_only；当前为V5数据＋新no-ESMC模型。找到的旧V5长训练是ESMC融合版本，没有把它冒称V5 no-ESMC基线。

两边均前期下降快、后期趋缓，但总loss不是同口径。当前训练日志数值更低、主验证loss更高，都不能单独证明新架构优劣。当前主验证在45k取得1.992721最佳，到72k仍未刷新；10k→72k相对下降约1.41%。旧版自身的随机遮蔽验证在同区间从0.801961降至0.724127，约9.71%；这是各自曲线变化，不是匹配数据/任务的架构收益对照。

## 原始日志数字

训练列是该step原生记录：旧每20步、新每25步记录，未擅自重标尺度。相同步数下当前名义样本呈现量是旧版两倍（global512对256）；旧72k epoch2.519859，当前72k epoch4.833230。

| 步数 | 旧no-ESMC训练loss | 当前训练loss | 旧随机遮蔽验证loss | 当前全遮蔽验证loss |
|---|---:|---:|---:|---:|
| 1000 | 5.7488 | 0.9462 | 1.190758 | 2.179086 |
| 10000 | 2.8673 | 0.4477 | 0.801961 | 2.028133 |
| 40000 | 2.6051 | 0.4204 | 0.731683 | 2.000729 |
| 72000 | 2.5553 | 0.3885 | 0.724127 | 1.999634 |

## 为什么不能直接相除或排名

- 数据和验证集：旧V6 binding-only，旧入口按来源遍历prepared valid；当前V5七来源，每源固定256条、合计1792条。
- 遮蔽：旧prediction_step前向随机采样遮蔽；当前固定25/50/75/100四等级，checkpoint选择使用100%目标遮蔽的source-macro loss。当前72k四档分别0.645770、0.722784、0.884275、1.999634；四档source-macro均值1.063116，与旧连续随机遮蔽也不等价。
- 监督token：旧保存fusion_config明确固定H/L槽位167/135、重复chain EOS；当前目标包含真实目标残基、已有关系标签、`.`和单个对象结束符，尾部空槽位不计loss。两边都保留context条件，不能把当前较大full-mask loss直接解释成模型更差。
- 聚合：两版总验证loss均按来源等权；旧来源内部是Trainer汇总batch loss，当前来源内部按被选token计数汇总，数据长度及EOS占比也不同。
- 训练日志尺度：旧178k loss2.4464、179k loss0.3218发生明显跳变，不能说一步区间获得约8倍质量提升。旧记录git提交的FusionTrainer没有显式model_accepts_loss_kwargs=False，当前冻结Trainer明确设置False并验证micro-mean累积契约；历史未提交修改/实际进程trace不完整，未武断将全部旧loss除8或声称找到了全部跳变原因。
- 日程：global256/100万步余弦与global512/30万步余弦，按相同步数并非同样训练样本量和学习率阶段。

## 对后续判断的含义

当前训练稳定下降，当前完整验证主loss在2.0附近波动；与旧版的架构优劣仍未由这些曲线证明。公平结论需要相同数据记录、目标token、mask随机种子/比例、归一化及样本预算的独立评测协议；这仅是对照要求，本轮未启动新GPU任务或切换正在运行的模型。当前新版本仍无下游成绩。

## 图与证据

![分别展示两种原生协议的loss曲线](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_vs_legacy_noesmc_loss_20261007/loss_trajectories.png)

图中各面板分别使用原生协议和坐标范围；训练图为对数轴，平滑线覆盖1000更新步，不作跨面板高度排名。

- [结构化原始曲线与源SHA256](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_vs_legacy_noesmc_loss_20261007/comparison.json)
- [选定步数CSV](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_vs_legacy_noesmc_loss_20261007/selected_steps.csv)
- [可缩放SVG](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_vs_legacy_noesmc_loss_20261007/loss_trajectories.svg)
- 旧Trainer状态：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/checkpoint-182000/trainer_state.json
- 当前72k归档：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512/checkpoint_retention_evidence/checkpoint-72000_ce6f98b98cda/trainer_state.json
- 旧实际argv：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/wandb/wandb/run-20260923_135822-bjnhqrdh/files/wandb-metadata.json
- 旧保存配置：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m/checkpoint-182000/fusion_config.json
- 当前冻结训练/验证：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v3/source/scripts/architecture_prototypes/validation_metrics.py
- 当前冻结八rank累积约束：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v3/source/output/generic_eight_rank_trainer_integration_20261002/eight_rank_trainer_adapter_v1.py
