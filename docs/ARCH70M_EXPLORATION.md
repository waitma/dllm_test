# V8 约 70M 架构探索实验记录

2026-10-28，执行用户批准的计划。工作树 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore`，分支 `explore/arch70m-v8-20261028`。当前 1B 训练使用的工作树、冻结源码和输出未改动。

## 2026-10-29 蛋白块换序的模型层核查

在本地确认训练继续运行时，仅用CPU FP32读取A0/A4/A5筛选终点step1276权重，模型/binding/state文件SHA与完整checkpoint回执一致。每组选择ASD、OAS、TCR native各一条最短的合格多链整链MASK输入（按长度、受体key、角色排序，不按模型结果选择），分别检查连续chain_id块反序，以及非负chain_id的一一重编号，共18项检查。原始V8记录、collator和训练输入流程保持不变；变换只作用于诊断内存中的七项denoise特征。

将换序后的输出逆置换回原顺序，三模型的logits和最终hidden states均在预设CPU FP32工程容差`atol=5e-5, rtol=1e-5`内一致；所有attention bias经逆置换后逐位一致。全体最大logit绝对差为1.90735e-5、hidden绝对差为7.12276e-6；单独重编号chain_id时两类输出均逐位相同。这里的容差只用于浮点换序检查，与训练重放门槛无关。

这与源码结构一致：token/role/scope嵌入逐位置计算，RoPE使用随蛋白块移动的链内position_ids，attention的同链/异链偏置只依赖chain_id是否相等，FFN与残差尺度均逐位置作用。因此在完整特征随置换P移动时，attention bias按`B -> P B P^T`变换，输出应按`f(PX)=P f(X)`变换。A5把每层改成全局可见，仍保留这些结构；A4的通道尺度也不引入蛋白序号。

这项实测支持：在这些输入和CPU前向上，A5保留蛋白块换序等变性；因此A5与A0的筛选差异不能简单解释成“只有一个模型具备蛋白顺序等变性”。它不证明全部输入或GPU内核逐位相同，不把交换H/L生物学角色或倒置链内残基当作对称操作，也没有测试16轮生成及top-k平分时的选位规则。三种子确认仍按原队列与阈值运行。

产物：[脚本](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/check_chain_equivariance.py)、[18项检查CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/chain_equivariance_20261029/checks.csv)、[输入签名、置换、checkpoint SHA和结果](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/chain_equivariance_20261029/chain_equivariance.json)。

## 2026-10-29 A4残差尺度检查

本地确认运行期间，只在CPU读取A4已完成筛选的step500/1250/1276权重；模型、binding和state的SHA均与checkpoint回执一致。检查48个尺度向量，每个768维，无非有限值。冻结源码未改，没有额外GPU推理或使用confirm结果。脚本：[inspect_residual_scales.py](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/inspect_residual_scales.py)。

A4新增的12,288个参数是8层×2条残差分支×768通道的尺度系数，初始均为0.1。筛选终点的FFN系数通道均值按层依次为0.095983、0.097109、0.099781、0.102425、0.104981、0.106650、0.108567、0.109889；首层相对初值-4.02%，末层+9.89%。attention尺度也不相同，例如第5层链内attention均值0.107428，第6层全局attention均值0.099855。A0保持固定0.1；其尺度是非持久buffer，不应因checkpoint中没有这些键而误认为尺度不存在。

这证明本轮A4确实学到了层间和通道间不同的尺度，不是所有层始终维持初值。它只提供后续检验分层残差设计的线索：系数不是attention概率，也不直接代表分支输出贡献，其他投影权重可以共同适应，不能将单种子的AUROC改善因果归于该曲线。图中阴影为同一层768通道的标准差，不是三种子置信区间。九组三种子确认和原判定规则保持，尚未据此新增或替换候选。

产物：[尺度图](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/residual_scale_audit_20261029/residual_scales.png)、[三个checkpoint逐层CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/residual_scale_audit_20261029/layer_scales.csv)、[定义与权重SHA](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/residual_scale_audit_20261029/residual_scale_analysis.json)。同目录包含PDF/SVG。

## 2026-10-29 生成输入与目标区分诊断

在本地确认队列运行期间，进一步检查八组筛选输出是否只体现通用序列恢复。CPU复用冻结collator与make_tasks，重建117条整链任务的七项实际denoise输入（整条目标槽位已MASK），并核验936条已保存预测的原AAR。没有读取confirm评测或重新推理；脚本位于[audit_generation_context.py](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/audit_generation_context.py)。

对每个已保存预测，同时按原口径计算正确目标AAR，及同来源、同链类型中其他不同参考序列的平均AAR；排除与本目标完全相同的参考序列，其他参考按序列去重。后者是分数重配对诊断，不是改变条件后重新生成，也不是结合特异性的测量。两列先来源内按任务平均，再四来源宏平均。

| 筛选终点 | 正确目标AAR | 其他目标平均AAR | 正确减其他（百分点） |
|---|---:|---:|---:|
| A0 | 12.18% | 12.21% | -0.03 |
| A4 | 12.04% | 11.49% | +0.55 |
| A5 | 37.02% | 35.42% | +1.60 |

A5按来源的正确减其他差值为ASD -0.56、OAS -0.80、OTS +0.18、TCR native +7.58个百分点。由此推断，当前高整链AAR含有较强的保守序列贡献，目标区分迹象主要在本轮TCR native样本；不能仅凭37.02%宣称条件特异性或抗体设计能力已提升。该诊断也不能反推模型没有使用条件：要证明因果条件依赖，需要后续以固定模型和解码进行条件替换干预，并与生物学任务结果分开解释。

重复输出也已核对实际输入：TCR native beta的6个不同参考目标，在整链MASK后具有完全相同的模型输入；八个架构各自均对这6条任务生成相同序列。A5在每个来源/链类型内的不同预测数等于不同输入数（总计112个输入对应117条任务），因此这处六次重复不能直接判为模式坍塌。greedy输出不同也不能单独证明生成多样性充分。

原A5/A4候选选择、九组确认队列、独立留出集及判定阈值保持。上述后验诊断补充解释生成质量，不作为事后改选规则，不改变原V8数据形式。产物：[汇总及定义](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/context_audit_20261029/context_analysis.json)、[来源对比](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/context_audit_20261029/by_source.csv)、[输入与输出数量](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/context_audit_20261029/by_source_role.csv)、[逐任务重评分](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/context_audit_20261029/per_task.csv)。

## 2026-10-29 生成覆盖率与相同位置核查

围绕持续目标“得到能够提升表现的架构”，对八组筛选终点的936条整链任务输出（每组117条）逐条重新计分，原发布AAR全部复现。只读取screen原始任务与已保存预测，不做新GPU推理、不读取confirm结果、不改变原选择或确认阈值。核查脚本位于[analyze_generation_coverage.py](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/analyze_generation_coverage.py)，不属于冻结训练源码。

| A0 vs A5，筛选终点 | A0 | A5 |
|---|---:|---:|
| 参考序列位置覆盖率（来源宏平均） | 33.60% | 95.82% |
| 两者共同覆盖位置的准确率（来源宏平均） | 37.87% | 53.64% |
| 输出不足真实长度一半的任务 | 107 / 117 | 0 / 117 |

共同覆盖位置按每条配对任务的 `min(A0预测长度,A5预测长度,参考长度)` 确定，因此两模型比较同一组位置；它是描述性筛选诊断，不替代原全参考长度AAR。四来源共同位置准确率分别为 ASD 32.82%→57.26%、OAS 43.51%→55.87%、OTS 37.42%→41.79%、TCR native 37.71%→59.65%。改善同时伴随提前EOS减少与共同位置预测更准确，不能仅归因于“生成长度更长”，也未由此证实跨链注意力的因果机制。

整链生成中的CDR位置与“给定其余残基的CDR补全”是不同任务。基于精确FR/CDR注释的后验重算显示，OAS整链CDR AAR为5.82%→26.38%、OTS为8.60%→20.54%；这不改变此前CDR补全AAR 80.08%→79.36%的记录。所有指标均只作当前单种子筛选解释，确认阶段仍使用原来的独立留出集与三种子规则。

产物：[诊断图](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/coverage_audit_20261029/coverage_diagnostic.png)、[来源汇总](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/coverage_audit_20261029/by_source.csv)、[相同位置比较](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/coverage_audit_20261029/shared_positions.csv)、[定义及日志SHA](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/coverage_audit_20261029/coverage_analysis.json)。

## 2026-10-29 本地三种子确认队列

用户指示“继续探索，所有的都在本地这个机器上面跑”。八组筛选已全部完成，A7 于 2026-10-29 00:16:52 +08:00 结束，整链 AAR 13.89%、条件 CE 3.751843、CDR CE 0.815607、宏 AUROC 0.835241。依既定筛选规则冻结 A5 生成候选、A4 表征候选，以及 A0 对照；[selection.json](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_v8_20261028/selection.json)保存八组终点评测SHA和选择分数。筛选选择没有使用confirm留出集。

2026-10-29 00:33:49 +08:00 已在当前开发机唯一 GPU0 启动持久后台队列，控制器 PID122888（flock）/122889（suite），首个worker PID122910。顺序为 A0 seed42/43/44 → A5 seed42/43/44 → A4 seed42/43/44，共9组；每组从头初始化3epochs、3828步，global512，micro沿用既定显存profile策略。仍是653,288条固定10%子集、原V8数据形式和筛选源码 SHA `053297983e0f2ece2aaad807032695fb5aebd1e0f221138a9bf0d202cc9b9ac4`。旧执行分工已归档，后续探索全部本机运行。

2026-10-29 01:05 +08:00，首组confirm_A0_s42已完成step250的screen完整评测并保存checkpoint；token CE=1.050245、整链AAR=18.57%、宏AUROC=0.796904。六文件SHA独立复核通过；CPU严格加载75,741,903参数及80组Adam状态，优化器/scheduler步数250、LR、next_batch250和seen_records128000一致，权重/动量无非有限值，Python/NumPy/CPU/noise RNG可加载且CUDA RNG已保存。此为文件完整性与CPU加载检查，未做CUDA数值重放，未中断训练；独立confirm终点评测尚未开始。验证回执：[step250检查](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_checkpoints/confirm_A0_s42_step250.json)。

2026-10-29 01:35 +08:00，confirm_A0_s42完成step500完整screen评测和checkpoint。相较step250：token CE从1.050245降至0.816243，CDR AAR从68.97%升至76.41%，宏AUROC从0.796904升至0.851734；整链AAR从18.57%降至16.28%，长度MAE从46.74升至65.77，条件CE从3.724049升至3.749587。生成和表征指标方向并不一致，应继续原完整确认协议，不能单用token loss或中间AUROC宣称胜出；确认阶段的LR日程与1epoch筛选不同，不将此点直接与筛选终点排名比较。六文件SHA、step/next_batch500、seen_records256000与冻结源码/数据绑定已核验，181条生成和106条表征观测完整；截至核查的逐步训练记录连续且loss/梯度均有限。未进行本点GPU数值重放，独立confirm终点评测仍未使用。曲线已纳入新点；[step500回执](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_checkpoints/confirm_A0_s42_step500.json)。

2026-10-29 02:06 +08:00，confirm_A0_s42完成step750完整screen评测与checkpoint。相较step500，token CE从0.816243降至0.752463，CDR AAR从76.41%升至78.45%；整链AAR从16.28%降至10.61%，长度MAE从65.77升至80.36，条件CE从3.749587升至4.290635；宏AUROC从0.851734变为0.849416，宏AUPRC从0.631775变为0.635469。整链生成退化与token/CDR改善再次同时出现，继续分别评估各目标，不据此提前改选或缩短确认。六文件SHA、step/next_batch750、seen_records384000与冻结源码/数据绑定均核验；181条生成、106条表征观测完整。生成和表征best checkpoint记录均仍为step500，未被较晚的step750覆盖。更新完整曲线/CSV；[step750回执](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_checkpoints/confirm_A0_s42_step750.json)。这是本组中间screen结果，独立confirm终点评测尚未开始，未做本点GPU重放。

2026-10-29 02:36 +08:00，confirm_A0_s42完成step1000完整screen评测与checkpoint。相较step750，token CE从0.752463降至0.725498，CDR AAR从78.45%升至79.50%；整链AAR基本不变（10.61%→10.65%），长度MAE略降（80.36→79.58），条件CE继续升高（4.290635→4.675561）。宏AUROC为0.851295、宏AUPRC为0.624515，尚未超过step500的最佳AUROC。生成和表征best checkpoint均仍保留step500。六文件SHA、step/next_batch1000、seen_records512000与冻结源码/数据绑定通过核验；181条生成、106条表征观测完整，逐步训练记录连续且loss/梯度均有限。已更新完整曲线/CSV与[step1000回执](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_checkpoints/confirm_A0_s42_step1000.json)。这是确认训练中的中间screen结果；独立confirm终点评测仍未使用，未进行本点GPU数值重放，九组多种子协议继续执行。

2026-10-29 03:10 +08:00，confirm_A0_s42完成第一轮训练及step1250/1276完整screen评测，已进入第二轮。step1250：token CE0.716113、CDR AAR79.63%、整链AAR9.05%、条件CE4.610025、长度MAE84.62、宏AUROC0.873140、宏AUPRC0.652609；step1276：token CE0.717845、CDR AAR80.00%、整链AAR9.25%、条件CE4.919931、长度MAE83.63、宏AUROC0.861855、宏AUPRC0.643873。表征AUROC在1250步达到本次运行新最佳后回落；整链AAR仍低于1000步的10.65%，不能由token/CDR改善推断整链生成改善。best生成仍为step500，best表征更新为step1250，两者均保留。两个checkpoint各六份恢复文件SHA、冻结源码/数据绑定及181条生成/106条表征观测通过核验；step1276的数据游标为epoch=1、next_batch=0、seen_records=653288，逐步记录连续、loss/梯度有限，累计记录数与训练日志一致。已更新曲线/CSV；回执：[step1250](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_checkpoints/confirm_A0_s42_step1250.json)、[step1276](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_checkpoints/confirm_A0_s42_step1276.json)。这些是三轮确认训练中的中间screen结果；独立confirm留出集仍未使用，未进行这两个点的GPU数值重放，既定九组确认继续。

2026-10-29 03:37 +08:00，confirm_A0_s42完成step1500完整screen评测与checkpoint。token CE为0.710401、CDR CE为0.778400、CDR AAR为80.42%；相较第一轮末step1276，整链AAR从9.25%恢复至10.67%，条件CE从4.919931降至4.376183，长度MAE从83.63降至81.40；宏AUROC从0.861855降至0.836977、宏AUPRC为0.628518。生成best仍为step500，表征best仍为step1250，均保留。六文件SHA、冻结源码/数据绑定及181条生成/106条表征观测核验通过；游标epoch=1、next_batch=224、seen_records=767976、seen_tokens=252133174与逐步日志一致，训练记录连续且loss/梯度有限。已更新完整曲线/CSV与[step1500回执](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_checkpoints/confirm_A0_s42_step1500.json)。这些是本组中间screen结果，独立confirm留出集仍未使用；未进行本点GPU数值重放，九组本地确认继续，尚不能判断架构是否确认胜出。

2026-10-29 04:07 +08:00，confirm_A0_s42完成step1750完整screen评测与checkpoint。相较step1500，token CE从0.710401降至0.700540，CDR CE从0.778400降至0.771982，但CDR AAR从80.42%回落至79.92%；整链AAR从10.67%降至9.29%，条件CE从4.376183升至4.656743，长度MAE从81.40升至85.80。宏AUROC回升至0.850284、宏AUPRC为0.653195，尚未超过step1250的最佳AUROC；best生成仍为step500，best表征仍为step1250。六文件SHA、冻结源码/数据绑定与181条生成/106条表征观测通过核验；游标epoch=1、next_batch=474、seen_records=895976、seen_tokens=294223586与逐步日志一致，loss/梯度有限。曲线和CSV已更新，回执：[step1750](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_checkpoints/confirm_A0_s42_step1750.json)。token CE下降仍未转化为整链生成改善，继续原九组三种子确认协议，不据中间screen指标改选；独立confirm留出集仍未使用，未进行本点GPU数值重放。

每100步token验证，每250步及epoch末完整生成/表征评测和checkpoint；每组结束后评估confirm留出集及AB specificity，刷新曲线并自动commit/push探索分支。全部9组结束后输出配对三种子统计，当前不宣称确认成功。后台进程独立于本次终端连接；机器重启后可用下列原指令恢复，控制器锁和trial锁防止重复运行，已完成试验经绑定校验后跳过。

```bash
flock -n /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_v8_20261028/local_confirmation_controller.lock bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/experiments/arch70m/run.sh --stage confirm --gpus 0
```

运行回执：[local_confirmation_controller.json](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_v8_20261028/local_confirmation_controller.json)；控制台：[local_confirm_20261029.console.log](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_v8_20261028/local_confirm_20261029.console.log)。下方四卡命令保留为历史交付入口，当前队列不使用它们。

## 2026-10-28 22:54 +08:00 评测快照与可视化

最新核对：本地 A0/A1/A2 和四卡侧 A3/A4/A5/A6 均完成 seed42、1 epoch、1276 步，含最终 token/生成/冻结表征评测；A7 快照时为 602/1276 步，最近一次完整评测为 step500。以下历史首轮记录与执行分工均按其当时状态理解。当前未做八组最终候选选择或三种子确认。

新增[已完成实验终点对比图](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/snapshot_20261028T1454Z/final_comparison.png)、[完整验证曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/snapshot_20261028T1454Z/validation_curves.png)；同步提供同名 SVG/PDF、[指标 CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/snapshot_20261028T1454Z/final_metrics.csv)及[可复现数据快照](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/snapshot_20261028T1454Z/snapshot.json)。A7 只以虚线展示已观测点，不参与终点对比；所有验证曲线不平滑。

| 终点指标 | A0 原版 | A1 QK Norm | A2 门控 | A3 QK+门控 | A4 可学习残差 | A5 全局注意力 | A6 FFN 重分配 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 验证 CE ↓ | 0.719104 | 0.723178 | 0.719791 | 0.726911 | 0.718150 | 0.746810 | 0.716032 |
| 条件整链 AAR ↑ | 12.18% | 9.40% | 9.85% | 15.41% | 12.04% | 37.02% | 12.89% |
| 条件整链长度 MAE ↓ | 76.50 | 78.72 | 85.31 | 50.07 | 80.29 | 5.61 | 77.65 |
| CDR AAR ↑ | 80.08% | 79.81% | 80.71% | 80.25% | 80.89% | 79.36% | 80.65% |
| 表征宏 AUROC ↑ | 0.825244 | 0.826418 | 0.832627 | 0.828297 | 0.847814 | 0.822220 | 0.835813 |
| 表征宏 AUPRC ↑ | 0.630206 | 0.643171 | 0.675151 | 0.671228 | 0.636231 | 0.686560 | 0.644566 |

初步观测：A5 整链 AAR 相对 A0 增加 24.84 个百分点，长度 MAE 从 76.50 降至 5.61，条件 CE 从 4.404985 降至 3.426782；同时 token 验证 CE、CDR AAR 和宏 AUROC 略差，因此改善集中于本轮整链生成指标。A4 宏 AUROC 最高，A6 token 验证 CE 最低。A2 的生成有效率宏平均为 91.30%（其他已完成组均 100%），长度 MAE 仅统计有效输出，须一起解读。

生成和表征指标按来源宏平均；整链生成有 117 条链任务、CDR 有 64 条任务，任务可能来自同一记录，不能当成独立生物样本数；表征验证仅 tcr_papers 33 条、trait 73 条。当前单种子、小样本筛选，没有统计显著性结论，也不能将 AAR 等同于实验功能验证。原始记录/数据格式和冻结训练源码保持；这次只读取日志、生成可视化并更新文档，没有启动新训练或评测。

复现图表（激活 pllm 后使用已有探索环境）：

```bash
/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_env/bin/python /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/render_validation_snapshot.py --output-dir /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/snapshot_20261028T1454Z --snapshot /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/snapshot_20261028T1454Z/snapshot.json
```

## 问题与固定预算

比较架构是否改善固定解码的生成质量、冻结特征的表征质量。两项目标同等重要；更低 MLM loss 不能单独支持“生成更好”的结论。

基础模型从当前正式 V8 冻结源码导入，来源与逐文件 SHA 见 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/experiments/arch70m/baseline_sources.json`。原来的 RMSNorm、SwiGLU、RoPE、双向全头注意力、交替链局部/全局可见性、0.1 残差缩放、V-family 辅助目标均保留作为 A0。尺寸缩至 8 层、768 维、12 头、FFN 3072，实际 **75,741,903** 参数。所有模型从头训练，同种子下形状相同的原有参数使用相同初值；A6 因 FFN 尺寸改变而单独初始化这些层。

固定：10% 数据，global batch 512，AdamW LR 1e-4、betas (0.9,0.999)、epsilon 1e-8、weight decay .01、梯度裁剪 1，前 5% 步 warmup 后 cosine，BF16、关闭 TF32、PyTorch 确定性算法、整层 activation checkpointing。原 V8 token/family 目标、167/135 链槽位、最大序列 2048、183 类 family 词表均不变。micro 先倍增再按 8 条粒度细化，reserved+外部占用不超过总显存 85%；不整除 512 时最后一个 micro 缩短，仍恰好 512 条。损失按整个有效 batch 的有效 token/chain 数归一化。profile 后重新初始化模型和优化器。

## 架构矩阵与来源

| ID | 相对 A0 的唯一改动 | 参数数 | 目的 |
|---|---|---:|---|
| A0 | 原版 V8，缩至上述尺寸 | 75,741,903 | 对照 |
| A1 | RoPE 前 Q/K 每头 RMSNorm | 75,742,927 | 检验 attention logit 稳定性 |
| A2 | SDPA 后、输出投影前，可学习 sigmoid 每头门控 | 75,815,727 | 检验输入相关门控 |
| A3 | A1 + A2 | 75,816,751 | 检验组合效果 |
| A4 | 固定残差缩放改为可学习向量，初值仍 .1 | 75,754,191 | 检验分层更新尺度 |
| A5 | 各层均全局可见 | 75,741,903 | 检验局部/全局交替设计 |
| A6 | 逐层 FFN 宽度 4608/4464/4032/3408/2736/2112/1680/1536 | 75,741,903 | 固定 FFN 总参数量下的层间分配 |
| A7 | SDPA 输出固定乘 .5 | 75,741,903 | 排除 A2 仅因平均幅度减小而受益 |

QK norm 参考 [Qwen3 技术报告](https://arxiv.org/html/2505.09388v1)；门控位置参考 [Gated Attention](https://arxiv.org/abs/2505.06708)；A6 参考 [Tapered Language Models](https://arxiv.org/abs/2606.23670) 的宽度重分配思想，按本模型 FFN 预算离散化。这些是实验动机，未预设它们对免疫序列有效。[recAMPLIFY](https://arxiv.org/abs/2610.26747) 的循环计算与 [mHC](https://arxiv.org/abs/2512.24880) 留待后续，因为本轮优先保持计算/参数比较清楚。

## 数据与防泄漏边界

**2026-10-28 用户确认：所有架构探索保持当前数据形式。** 沿用 V8 prepared JSONL 的字段、链序列与顺序、chain roles/metadata、relation/family 标签，以及现有 tokenizer、词表、特殊 token 和固定链输入布局；复用冻结的 reader/collator。10% 抽样只选择原记录 offsets。独立实验索引与评估记账信息不改变训练记录 schema，不为架构变体新增或转换输入格式。已核对 109 个训练分片的路径与内容 SHA 均与父索引相同；本次仅补充约束，正在运行的实验无需变动。

原数据只读：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/data/prepared/immune_v8_four_relations_20261026_r2`。使用当前验证过的父 index，每个训练分片内按 relation、receptor signature 分层，以确定性 hash 选约十分之一，最小分层至少 1 条。不是“只读开头 10%”。训练索引、验证样本和全部输入 SHA 固定在探索 output 的 `data/index.json`。

先排除所有原 valid 和 protected probe_valid 的 observed receptor identity，再排除 AB specificity 全部 4398 个样本的完整链及已审计 Fv 对。identity 包含角色和真实观测序列，去掉 synthetic residues，涵盖 VHH。保护的是精确受体组合，**不声称同源性、单链或结构簇完全隔离**。原 probe_train 只用于拟合冻结特征头，不加入额外预训练步骤。

valid/probe_valid 依据同一个受体 hash 固定分为 screen / confirm，避免相同受体跨两边。screen 每来源最多 64 条 token 验证、16 条 task 候选；confirm 分别 128/64 条。task 再排除非结合、合成链、空链；CDR 只接受 FR/CDR 注释精确拼回完整序列的样本。当前生成覆盖 asd_antibody、oas、ots、tcr_native，CDR 覆盖 oas/ots；不把这些结果泛化到 VHH 或合成模板链。排除数逐次写入评估结果。

## 阶段与选择规则

1. 本地先跑 A0、A1，单卡大 micro batch，seed 42、1 epoch；这两组直接计入筛选。
2. 八组各 1 epoch、seed 42。四卡是四个独立单卡进程，不启动 DDP/NCCL。相同有效 batch 和记录次序，尾 batch 保留，不重复填充样本。
3. 全部八组完成后才允许生成 `selection.json`。仅用 screen 最后一个 epoch 的指标：生成方向按 conditional/CDR CE 的等权相对改善排序，要求有效率不低于 A0 超过 2 个百分点、有效样本长度 MAE 不超过 A0 的 1.1 倍；表征方向按来源宏平均 AUROC 排序。两个候选相同时，以两方向平均名次补第二名。如果没有生成候选通过 guard，明确标记 fallback，不宣称其已改善生成。
4. A0 + 两个候选重新随机初始化，各跑 3 epochs × seeds 42/43/44；不延续筛选 checkpoint。确认候选/数据/源码绑定后才允许该阶段运行。每个确认试验结束后评估 confirm，并跑 AB specificity 二级验证。

所有运行记录逐步训练 CE、family CE、学习率、梯度范数、实际 token/record 数、吞吐、显存；每 100 步 token 验证，每 250 步及 epoch 末做完整 task/probe 评估。初始 step 0 也评估。生成使用相同 16 步 greedy confidence-unmask、AA/EOS 合法输出、原生 EOS，不提供目标长度。conditional CE 只计真实残基；所有缺 EOS、空链、不完整链保留，有效率按全部生成任务分母，失败 AAR 为 0，AAR 只计 EOS 前真正生成的序列，长度 MAE 明确仅对有效链统计。

冻结表征探针隐藏 relation token，不把 family/relation 标签作为特征。头为 train-only StandardScaler + 固定 C=.1 LogisticRegression；按来源报告 AUROC/AUPRC，并记录打乱训练标签的控制。当前仅 tcr_papers/trait 两来源，样本较小，是开发诊断。

AB specificity 使用官方固定五折、global observed-residue mean 特征、固定 100 epochs 的 768→768 tanh→3 分类头，AdamW 5e-5、batch 8、10% warmup、fold seeds 42+k，不在外层 test 上选 epoch。只有 2/4398 对需要现有 ANARCI_IMGT Fv 边界转换；输入和折完全审计，转换后也检查跨折精确重复。标为 **fixed-v2-fv**，不与 full-chain 口径混称。模型参数冻结，保存 head 并加载后重算预测。此任务不参与候选筛选，也不新增主榜数字。

确认报告以同种子 A0 为配对参照，5000 次 seed→来源内受体 cluster bootstrap，两个候选×两方向四项比较使用 Bonferroni（98.75% 双侧区间），并要求至少 2/3 seeds 同向。生成还必须检查 AAR/有效率/长度 guard。三种子和小样本局限保留；未完成的实验不参加推断。原始曲线全部保存，平滑只用于展示。不同宽度组同时报告训练 GPU 小时；它不含 profile/评估/保存，不能当作总成本。

## 2026-10-28 本地首轮完成记录

09:56 UTC 核对：A0、A1 均完成 seed 42 的 1 epoch（1276 步、653,288 条），分别于 05:05:02 UTC、07:59:16 UTC 结束。两组源码 SHA 和数据索引 SHA 一致，沿用原数据形式。各有 1276 条逐步训练记录、17 个验证记录点和 7 次完整生成/表征评估（含 step 0）；最终 checkpoint 均为 step_001276。

| 最终 screen 指标 | A0 原版 | A1 QK Norm |
|---|---:|---:|
| token 验证 CE ↓ | 0.719104 | 0.723178 |
| CDR CE ↓ | 0.820305 | 0.846391 |
| CDR AAR ↑ | 80.0755% | 79.8113% |
| 条件整链 CE ↓ | 4.404985 | 4.126281 |
| 条件整链 AAR ↑ | 12.1802% | 9.4014% |
| 条件整链长度 MAE ↓（残基，仅有效链） | 76.4994 | 78.7209 |
| 表征宏平均 AUROC ↑ | 0.825244 | 0.826418 |
| 表征宏平均 AUPRC ↑ | 0.630206 | 0.643171 |

QK Norm 的条件 CE 更低，但实际整链 AAR 更低、长度误差更大，CDR 指标略差；表征宏平均 AUROC 差异很小且两来源方向不一致。当前整链生成效果仍弱，不能用 loss 降低宣称生成质量改善。结果仅单种子、小样本 screen；A2–A7 和三种子 confirm 尚未执行，未选择胜出架构。控制器已退出，本地 GPU 空闲。两组训练成功；终态非零由远程发布缺少 GitHub 凭据造成，本地结果提交为 10f2a3a / 097d88f。

## 保存、恢复与隔离

当前发布状态（2026-10-28 10:28 UTC）：开发机已由 `waitma` 完成设备登录并验证仓库写权限，探索分支首次推送 4ecc061 成功且远端 SHA 已核对。此前 GitHub 凭据缺失的记录为历史状态；当前发布回执见 [publication_status.json](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_v8_20261028/publication_status.json)。[GitHub 探索分支](https://github.com/waitma/dllm_test/tree/explore/arch70m-v8-20261028)。

每 250 步及 epoch 末保存，初始 step 0 也保存。包含 safetensors 权重、Adam 状态、scheduler、Python/NumPy/CPU/CUDA/noise RNG、epoch 与 batch cursor、源码/数据/环境绑定。临时目录写完并 SHA 验证后原子重命名。保留最近两份 + screen 最佳生成/表征 + 各 epoch 末。相同指令自动恢复，已完成匹配试验自动跳过；变更源码、数据、micro batch 或环境不冒充续训。

每个 worker 从 SHA 命名的独立源码快照导入，工作树之后的编辑不会改变正在运行的模型。单试验 flock 防重入；机器间通过显式分片避免重复调度。默认以本地 JSONL 为记录依据；`--wandb` 可选，首次创建唯一 ID，只有恢复同一 checkpoint 才复用，W&B 不可用时保留本地日志。每个试验终态生成文档/图，再 commit 并 push **探索分支**；失败也记录，不删证据、不 force-push、不操作 main。发布失败写 `publication_error.json` 并返回失败状态，不能冒称推送成功。

## 执行命令

**当前分工（2026-10-28）**：A0/A1 已完成；开发机单卡继续 A2（注意力门控），用户在四卡机器提交 A3–A7。两边读取同一冻结 10% 索引、使用原 V8 数据形式，统一写入当前探索输出；分配不重叠。四卡任务尚待用户提交，不能把准备好的入口记作已启动。

独立四卡入口：[launch_arch70m_4gpu.sh](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_4gpu.sh)。默认筛选列表已改为 **A3,A4,A5,A6,A7**，避开本地 A2；每张卡独立训练并验证一个变体，空闲后接下一组。入口复用现有冻结训练器，未改数据形式或既有试验源码绑定。`--trials` 可显式改列表；下面的历史完整矩阵命令仅在同名试验未运行时使用。

```bash
# 开发机单卡 A2（已由助手启动后无需重复运行）。
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/experiments/arch70m/run.sh --stage screen --trials A2 --gpus 0
# 用户四卡任务：A3-A7；与开发机的 A2 不重叠。
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_4gpu.sh --gpus 0,1,2,3 --trials A3,A4,A5,A6,A7
# 全部八组筛选结束后，运行三种子确认。
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_4gpu.sh --stage confirm
```

指定卡号用 `--gpus 0,1,2,3`；预览命令用 `--dry-run`。默认沿用原输出根及恢复/跳过规则，数据分片保持只读。脚本启动前检查四张指定卡是否可见；本地仅一卡时拒绝启动。两台机器共享同一路径和输出时，分别执行：

```bash
# 两台四卡机器进一步分配远端五组：机器 0 跑 A3/A5/A7。
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_4gpu.sh --machine-index 0 --machine-count 2
# 机器 1 跑 A4/A6；本地开发机仍只跑 A2。
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_4gpu.sh --machine-index 1 --machine-count 2
```

默认每组结束仍自动本地 commit 并尝试 push；缺少 GitHub 写入凭据时，结果保留并记录发布失败，最终进程可因发布失败返回非零。筛选/训练状态以每组 `status.json` 为准。

同一挂载路径的开发机和 GPU 节点使用以下命令。`run.sh` 激活 pllm，并首次创建探索专用 `output/arch70m_env`（system-site-packages venv）安装 matplotlib 3.10.7；不修改原 pllm。可用 `ARCH_CONDA_PREFIX` 指定同依赖的环境。输入和输出都写绝对路径。控制台输出在各 trial 的 `console.log`；逐步状态在 `status.json`。

本地两组（1 GPU）：

以下为历史/通用命令；**当前 A2 在开发机运行，四卡任务使用上方 A3–A7 命令**。完整矩阵或旧分片命令只能在同名试验均未运行时使用。

```bash
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/experiments/arch70m/run.sh --stage screen --trials A0,A1 --gpus 0
```

本地 A0/A1 正在执行时，其他机器的四卡先跑剩余六组，避免重复调度正在持锁的试验：

```bash
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/experiments/arch70m/run.sh --stage screen --trials A2,A3,A4,A5,A6,A7 --gpus 0,1,2,3
```

本地 A0/A1 结束后，单机四卡完整筛选（匹配的已完成 A0/A1 自动跳过）：

```bash
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/experiments/arch70m/run.sh --stage screen --gpus 0,1,2,3
```

两台各四卡机器，共用同一个 output（分片参数必须成对设置）：

```bash
# 机器 0：A0/A2/A4/A6
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/experiments/arch70m/run.sh --stage screen --gpus 0,1,2,3 --machine-index 0 --machine-count 2
# 机器 1：A1/A3/A5/A7
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/experiments/arch70m/run.sh --stage screen --gpus 0,1,2,3 --machine-index 1 --machine-count 2
```

全部筛选结束后，在一台四卡机器做确认，或用同样 machine 参数拆分确认的九个 trial：

```bash
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/experiments/arch70m/run.sh --stage confirm --gpus 0,1,2,3
```

单机依次完成全部阶段可用 `--stage all --gpus 0,1,2,3`；多机器建议分阶段执行，避免某台筛选未结束就尝试确认。其他输出位置使用 `--output-root ABS`，只读数据挂载使用 `--data-root ABS`；迁移机器需保留父 index 及其绑定绝对路径，不能只改字符串绕过 SHA。无共享存储的节点应先按原绝对路径复制源码及已冻结的完整 data/index/offset/validation 目录和原始 shards，再分别运行分片；将不重叠 trials 目录合并回同一根，再选择候选。

实时刷新图表：

```bash
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/experiments/arch70m/run.sh --stage report
```

## 实施与检查记录

- 独立四卡 shell：语法、screen/confirm 与多机参数转发、可见设备继承、缺卡/重复卡拦截通过；驱动可能忽略无效编号并返回成功，入口额外验证实际四个不同 GPU UUID。当前开发机只有一卡，未做四卡实跑。入口初次检查造成的 A1 状态误标已依据最终 checkpoint 全部文件 SHA 和原始训练/评估记录纠正，无新增训练步；原始异常、未开始的 A2/A3 检查输出及恢复回执均保留在 [检查证据](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_v8_20261028/validation/four_gpu_launcher_20261028T100322Z/reconciliation.json)。
- 从正式 V8 源码建立隔离工作树；没有启动/取消/改写现役 1B 作业。
- CPU：原版 A0 初始权重、前向、梯度逐位一致；八变体有限梯度；累积归一化；checkpoint/Adam/RNG 保存恢复；标签隔离；目标长度隐藏；常量门控控制；新参数序列化；机器分片和配对统计检查。
- 真实数据、小尺寸模型、GPU：完整 screen token/generation/probe 评估流程通过。此检查只验证管线，不作为 75M 架构实验结果。
- AB 输入：4398 记录、官方五折各记录恰好一次 OOF、Fv 边界及跨折重复审计通过。AB 头训练结果待确认阶段。
- 最终子集 **653,288 条**，109 个训练分片，共排除 170 条 held receptor 重叠；1 epoch 为 **1276 步**，3 epochs 为 **3828 步**。
- 原始 10% 准备产物保留在探索输出 `data_v1_before_ab_protection`；正式探索使用补齐 AB 保护后的 `data/index.json`。

本节以下由报告程序根据真实产物更新；pending/failed 不代表效果已验证。

2026-10-28 首次显存 profile：A0/A1 均选择 micro56，九个56条 micro + 一个8条 micro，global512。最大长度2048、含 Adam 状态、3次实际优化器更新；A0 allocated62.89/reserved77.91 GiB，A1 allocated64.87/reserved79.41 GiB，总94.97 GiB。micro64 超过 reserved+外部占用85%门槛，选择56。8项CPU回归全部通过。

<!-- ARCH70M_RESULTS_BEGIN -->
自动记录更新时间：2026-10-28T20:08:12Z。原始日志：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_v8_20261028`。

| Trial | Status | Step | Micro × accumulation | Last train CE | Peak allocated GiB |
|---|---|---:|---|---:|---:|
| screen_A0_s42 | completed | 1276 / 1276 | 56 × 9 + 8 | 0.3116 | 21.5761 |
| screen_A1_s42 | completed | 1276 / 1276 | 56 × 9 + 8 | 0.3126 | 22.5498 |
| screen_A2_s42 | completed | 1276 / 1276 | 56 × 9 + 8 | 0.3088 | 21.6665 |
| screen_A3_s42 | completed | 1276 / 1276 | 56 × 9 + 8 | 0.3164 | 22.7299 |
| screen_A4_s42 | completed | 1276 / 1276 | 56 × 9 + 8 | 0.3105 | 21.5765 |
| screen_A5_s42 | completed | 1276 / 1276 | 96 × 5 + 32 | 0.3173 | 25.9867 |
| screen_A6_s42 | completed | 1276 / 1276 | 56 × 9 + 8 | 0.3120 | 21.5255 |
| screen_A7_s42 | completed | 1276 / 1276 | 56 × 9 + 8 | 0.3157 | 21.5772 |
| confirm_A0_s42 | training | 1757 / 3828 | 56 × 9 + 8 | 0.2597 | 21.5778 |
| confirm_A0_s43 | pending | 0 / — | — | — | — |
| confirm_A0_s44 | pending | 0 / — | — | — | — |
| confirm_A5_s42 | pending | 0 / — | — | — | — |
| confirm_A5_s43 | pending | 0 / — | — | — | — |
| confirm_A5_s44 | pending | 0 / — | — | — | — |
| confirm_A4_s42 | pending | 0 / — | — | — | — |
| confirm_A4_s43 | pending | 0 / — | — | — | — |
| confirm_A4_s44 | pending | 0 / — | — | — | — |

![完整实验曲线](arch70m_results/curves.png)

原始逐步曲线与分来源指标：[CSV](arch70m_results/all_curves.csv)；汇总：[JSON](arch70m_results/summary.json)。
图中训练 CE 同时展示原始值和 25 步移动平均；验证指标不平滑。训练 GPU 小时不含评估、保存和 profile，不能当作端到端成本。
未完成试验不参与候选选择或显著性判断。三种子确认必须从头训练；筛选 checkpoint 不作为确认初始化。
<!-- ARCH70M_RESULTS_END -->

2026-10-28T02:32:25.344359+00:00：本地控制器 PID 69214 已启动 A0→A1 顺序队列，A0 已完成初始完整评估与 checkpoint-0，观察到真实训练 step 17/1276；A1 显存 profile 完成，等待 A0 释放唯一 GPU。实现提交 `e3c6692`。GitHub 推送未成功：当前机器没有 credential helper、GitHub token 或 SSH identity；已向用户请求在机器上登录，绝不将本地 commit 记作已推送。每个试验终态仍会保留本地结果 commit 并尝试 push。
