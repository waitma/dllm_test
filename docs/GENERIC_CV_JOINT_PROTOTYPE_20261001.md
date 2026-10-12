# 通用整数对象：理解与生成联合原型验收

## 2026-10-01T11:29:06.399847+00:00 最新接续覆盖下方CPU阶段状态

八原型CPU35项语义证据保持。新增GPU引擎已完成CPU44个唯一案例/环境的三阶段覆盖，固定包15文件与8配置核对完成；尚未接入现有8卡执行端，GPU/500步结果未运行。[当前GPU工程协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_JOINT_GPU_SEGMENT_PROTOCOL_20261001.md)、[CPU验收原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/validation_profile_count_fix.json)、[固定源码包及核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/source_package_review.json)、[指定数据门禁](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/requested_training_data_gate.json)、[当前配置SHA](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/updated_plan_manifest.json)。下方09:37记录和旧配置清单是当时原件，当前配置以本节链接为准。指定正式数据仍缺审核，人工成绩不作正式模型排名。

更新：2026-10-01T09:37:59+00:00。维护agent：`/root`。原Goal `01a0f245-4559-7b62-9dc2-0917a8ed2267`保持active。本页是G00–G07的实现与CPU验收记录；**八个方案已实现，CPU功能验收通过，GPU入口与计划500步比较尚未完成**。现有R20/R21/R24/R26的Trainer GPU工程验收属于另一套模型和证据，不与本页合并排名。

## 接手时先看

| 节点 | 真实状态 | 证据 |
|---|---|---|
| 文献与实验设计 | 已完成到研究页所列阅读范围；7份作者源码快照、8份YAML | [研究记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CV_UNIFIED_RELATIONAL_EXPLORATION_20261001.md) |
| 8种原型实现 | 已完成，独立d192/L4/H6，从头初始化 | [模型源码](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/generic_joint_model.py) |
| CPU语义检查 | 两环境各35个唯一用例完成覆盖；worker首轮1项失败，经定向修正后通过 | [验收原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_prototype_cpu_20261001/validation.json) |
| 实际优化器执行 | worker环境8配置各3窗口/9微批/24人工记录；pllm另有G00 CLI执行 | 下方逐配置原始结果 |
| 原始评价重算 | 9份功能报告的原始logits/labels独立重算通过 | 验收原件的`functional_runs` |
| 只读监督 | 已结束，无阻断代码问题；两项解释边界已登记 | [监督记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_prototype_cpu_20261001/supervisor_static_review.json) |
| GPU方法比较 | 新入口待实现；未入队，未测显存/GPU秒，未运行500步 | [配置状态与SHA](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_prototype_cpu_20261001/updated_plan_manifest.json) |
| 真实语料 | 未启动；training_ready=false，双证书缺失 | [统一接手状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md) |

验收原件SHA256：`00261018a90d12e8600679085d0c093c41317be55f6f332d8b957e027f0eb233`。源码、功能报告及日志的具体SHA均在原件内；本页不能替代原件。

## 人工任务、输入与共同模型

只使用程序生成的16类整数符号。随机段A与关系标签delta（0–3）决定`B=(A+delta)%16`，目标段`C=(A+B)%16`；可另带一个独立随机干扰段。训练长度8/16，固定验证含8/16/24，干扰段概率0.5。[人工数据源码](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/generic_joint_data.py)没有语料文件读取接口。

模型输入只有tokens、roles、positions、valid四项。角色1/2为条件，3为目标，4为可选干扰；使用角色embedding与段内sin/cos位置编码。理解端目标段完全遮蔽，关系标签不进入输入；生成端遮蔽目标段25/50/75/100%，仅在选中位置计算预测损失。PAD不作为注意力key或有效隐藏特征。长度由人工协议提供，**本次没有验证学习EOS、未知长度或停止生成**。

同一个student主干提供关系分类和逐位置生成两种输出；任务输入视图不同，参数共享。G06/G07额外EMA teacher仅用于训练，冻结且无优化器梯度，两端推理都使用student。主干为4层pre-LN Transformer、d192/H6、普通PyTorch SDPA，无dropout；它与现有免疫模型的d768/L8、约75.60M参数不同。

联合目标为逐记录平均的masked-token CE、逐记录关系CE及可选表示对齐项之和，按实际完整窗口记录数归一化；复用[record_loss.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/record_loss.py)，不额外除以微批数量。共同模块在可选模块之前构造，同seed的共同参数初始化一致。

整段置换时内容、角色、位置和valid一起移动，验证存储顺序变化下的输出关系；这不等同于交换语义角色。移除条件内容后的生成差异用于条件依赖诊断，保留原标签的该项不作为新反事实任务准确率。训练/验证有独立随机流和记录ID，**不据此认证内容层面严格去重**。

## 八个实际配置与参数量

| 配置 | 实际变化 | 可训练参数 | 总参数含teacher | CPU功能原件 |
|---|---|---:|---:|---|
| G00 | 共享主干与两种输出头 | 1,862,036 | 1,862,036 | [G00](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_cpu_20261001_g00_shared_joint_worker/CPU_FUNCTIONAL.json) |
| G01 | 理解沿用final_norm，生成增加独立norm | 1,862,420 | 1,862,420 | [G01](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_cpu_20261001_g01_task_norm_worker/CPU_FUNCTIONAL.json) |
| G02 | 最后两层按任务增加rank32适配层，up零初始化 | 1,912,084 | 1,912,084 | [G02](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_cpu_20261001_g02_late_adapters_worker/CPU_FUNCTIONAL.json) |
| G03 | 4个全局register，主干全局注意力 | 1,862,804 | 1,862,804 | [G03](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_cpu_20261001_g03_register4_worker/CPU_FUNCTIONAL.json) |
| G04 | 4个register，限制主干直接跨角色注意力 | 1,862,804 | 1,862,804 | [G04](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_cpu_20261001_g04_bottleneck4_worker/CPU_FUNCTIONAL.json) |
| G05 | 与G04相同，register增为16个 | 1,865,108 | 1,865,108 | [G05](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_cpu_20261001_g05_bottleneck16_worker/CPU_FUNCTIONAL.json) |
| G06 | EMA0.99；student第2层对齐低噪声teacher第4层，权重0.1 | 1,899,284 | 3,761,320 | [G06](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_cpu_20261001_g06_self_alignment_worker/CPU_FUNCTIONAL.json) |
| G07 | 与G06相同teacher/投影/前向/EMA，辅助权重0 | 1,899,284 | 3,761,320 | [G07](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_cpu_20261001_g07_alignment_compute_control_worker/CPU_FUNCTIONAL.json) |

G06/G07 teacher各1,862,036参数，计算与存储成本均需计入。G07没有删除teacher前向。G04/G05限制的是**主干跨角色注意力**；关系分类头仍直接拼接角色1/2的池化特征，不能称所有理解信息必须通过register。dense attention加mask不代表稀疏计算或已测加速。

## CPU验收与首轮修正

| 环境 | 首轮 | 改动后定向复查 | 最终覆盖口径 |
|---|---|---|---|
| pllm / Python3.10.19 / Torch2.8.0+cu126 | 35通过 | 受影响2项通过、33未重跑 | 35个唯一案例；不声称最终版本一次完整35项重跑 |
| worker / Python3.12.12 / Torch2.8.0 | 34通过、1失败 | 受影响2项通过、33未重跑 | 35个唯一案例；同上 |

检查包含任务数学关系、标签与输入分离、teacher遮蔽嵌套、双输出头及共同主干梯度、teacher无梯度、目标内容不泄露理解输入、段置换/padding、瓶颈通信、共同参数初始化、记录归一化与不等长尾微批梯度一致性、EMA/零权重对照、已知数学oracle的4次前向生成、真实数据参数拒绝和原始指标重算。[测试源码](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_generic_joint_prototype.py)。

首轮worker失败为评价CE汇总：FP32两种求和顺序得到146.34414672851562与146.3441619873047，差1.5259e-5，超过原1e-5阈值。保存实际FP32 logits后诊断，FP64的两种汇总差约2.8e-14。修正仅把**评价报告CE**改为从实际FP32 logits以FP64累计；训练CE仍FP32，原阈值未降低。两个受影响案例定向复查通过。[失败诊断](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_prototype_cpu_20261001/recount_failure_diagnosis.json)、[首轮状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_prototype_cpu_20261001/attempt1_review.json)、[修正后源码清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_prototype_cpu_20261001/source_after_reporting_fix.json)。旧字节、首轮日志/JUnit和定向复查日志/JUnit均保留。

8个worker功能执行均使用真实d192/L4/H6、AdamW、global8、micro3（3/3/2）、固定学习率3e-4，各完成3次优化器更新/9微批/24条人工训练记录。另有[G00的pllm CLI执行](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_cpu_20261001_g00_shared_joint_pllm_cli/CPU_FUNCTIONAL.json)。记录了有限梯度、两头及首层QKV参数变化、实际AdamW状态；没有checkpoint。该工作不采用规划的global128/warmup25/500步日程，**只验功能与统计正确性，不能按三步指标排名**。

初始/最终固定验证保留实际关系logits/labels、四档mask token logits/labels、4次前向生成预测/目标、分长度与干扰段分组、条件移除差异、置换误差及表示方差。根端以标准库双精度logsumexp从保存数据独立重算9份报告，并复核分组计数及参数公式。CPU训练耗时仅覆盖优化器循环，不含全程评价，不能转为GPU速度结论；GPU秒和显存均未测。

## 清理与失败证据

[清理原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_prototype_cpu_20261001/cleanup.json)记录四个自有临时路径最终不存在，验证checkpoint=0。首次根端包装脚本已删除两个实际fixture目录，随后误要求定向检查必须创建另外两个目录而中断；后两路径实际未创建。文件/字节计数未持久化，保留null，不补造数字。[首次包装中断](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_prototype_cpu_20261001/root_review_attempt1.json)保留。此问题不改模型结果，后续独立重算完成；实际训练续训状态未修改。

保留原始输出、日志/JUnit、首轮失败、修正前字节与配置快照以及最终来源SHA。已完成的模型和语义检查不无依据重复。

## 下一步及持续边界

1. 实现并复审固定的通用整数GPU入口及现有池接入；限制变体/配置/来源、资源预算与人工数据范围，G方案使用独立身份，不借用R候选身份。现有已验运行包不热改。
2. 接入完成后，在原8卡池实测各方案microbatch、显存和耗时，再执行规划的500步、每50步评价；global128/warmup25保持设计值，不能用本次3步验收代替学习曲线。是否延长1000看validation，多seed确认候选。
3. 分开判断关系理解与生成、未见长度和干扰段表现及条件依赖；记录参数/teacher成本，不只按总loss选优。本页没有真实生物序列效果结论。

现有8卡平台任务、worker和监听PID2213164保持，不新增平台提交、不重启、不重复已完成作业。8份YAML均`execution_enabled=false`，是方法设计配置，不是Volc资源配置；[目录说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_cv_joint_exploration_20261001/README.md)。独立CUDA算子测试继续暂停。

真实数据training_ready=false且audit_certificate.json/downstream_scope_certificate.json均缺，真实语料训练不启动，不伪造审核或降低准入。此前安全检查停止的序列注释分支不恢复、重试、转交或绕过。原Goal最终训练目标未达成；本轮有通用工程进展，保持active。关键节点同步总账、当前接手入口、方案和长期约定。
