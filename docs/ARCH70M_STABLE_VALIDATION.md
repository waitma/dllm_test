> 当前执行核验：2026-10-31T05:29:55Z UTC。九组GPU资格全部通过，A0 seed42 step14/3828；其余8组排队。源/数据/三epoch绑定和实际命令通过检查。当前尚无稳定性结论，目标仍未完成。

# 稳定架构完整确认：2026-10-31

用户最新指示：继续完整验证，直到获得稳定提升的架构。本轮从既有筛选转入冻结候选的三种子确认；目标尚未完成，不能保证任何候选会通过。后台依次训练、验证、保存、汇总并发布，不持续人工盯进度。完成全部试验后按下述冻结标准判断；未通过保持负结果，后续继续设计架构，不降低门槛或挑选最好seed。

## 上轮结果与候选

共同micro32、seed43、1epoch的四组已全部完成，最终checkpoint共24文件SHA核验通过：

| 架构 | 整链AAR | CDR AAR | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| B0=A5，全局/固定残差 | 36.03% | 80.56% | 0.8253 | 0.6730 |
| B1，全局/可学习残差 | 37.41% | 80.05% | 0.8391 | 0.6905 |
| B2，每层链内→全局 | 28.21% | 79.06% | 0.8295 | 0.6671 |
| B3，B2＋可学习残差 | 25.63% | 79.73% | 0.8148 | 0.6464 |

B1相比A5整链AAR+1.38个百分点、AUROC+0.0139、AUPRC+0.0175，但CDR AAR-0.50个百分点；单种子不能宣称稳定。B2/B3的长度MAE约26.98/28.83，明显大于全局组7.57/6.30，不因其条件CE较低而晋级。选择A5(B0)和B1，新增原始A0同预算对照；选择证据的metrics SHA已写入不可变protocol/selection。未查看新候选的confirm终点作选择。

## 执行矩阵与数据约束

A0、B0(A5)、B1 × seeds42/43/44，共9组；每组随机初始化，3epochs/3828步，micro32×16=global512，最大长度2048，参数约75.75M。本轮完整验证明确扩展此前仅seed43的限制，不启动旧A0/A5/A4九组派发器。

GPU0在本开发机串行运行，顺序为每个seed的A0→B0→B1，再到下一seed。九个seed/架构组合都在2048最大长度做3次真实更新资格检查，loss/梯度有限且reserved＋外部显存≤总量85%；全部通过才正式训练。资格权重丢弃，每组新进程重新初始化。

保留原V8 prepared JSONL、链序列/顺序/角色、元数据、relation/family、词表/特殊token、固定链布局、reader/collator。原固定10%训练index逐字节相同，SHA `4beaac6b54a1f46492533aeb77c2de2c6ac2a0bd8266fd57a6bb312679435c51`；每epoch653,288条，三轮1,959,864条消费记录。未创建新训练数据或修改记录格式。LR1e-4、原AdamW/5%warmup日程、bf16、整层梯度重计算、原MLM/family与噪声/解码规则保持；日程按3epoch确认预算计算。

每步记录loss/梯度/显存/耗时，每100步token评估，250步及epoch末完整screen任务评估和六文件原子checkpoint；保存最新、最佳生成/表征及epoch末完整恢复状态。同绑定可从latest恢复，完成项必须通过最终checkpoint、数据游标和confirm/抗体评估产物检查才跳过。

## 完整验证与稳定门槛

所有模型采用固定最终step3828权重，在原confirm留出划分上评估；不按confirm表现挑checkpoint。confirm输入包括token_valid898条、task_valid450条、probe_valid105条；训练过程的screen曲线与这些最终指标分别保存。原保护抗体Fv五折固定末轮评估一并执行，只作辅助验证、不用于候选选择。

冻结三个配对：B0 vs A0、B1 vs A0、B1 vs B0；生成与表征两个方向，共六项比较。完整9组结束后才计算主稳定结论：

1. 对训练seed、再对来源内精确受体cluster做5000次配对bootstrap；六项比较Bonferroni校正，99.1667%双侧区间下界>0，并且3个seed的增益全部同向。
2. 生成主指标沿用原各任务/来源等权的相对CE改善，平均≥2%；还须至少2/3配对seed整链AAR与CDR AAR不退步、有效率下降≤2个百分点、长度MAE≤对照1.1倍。
3. 表征主指标为平均macro AUROC增加≥0.005，同时平均AUPRC不回退。AUPRC作为护栏，未单独宣称其显著性。
4. 只宣告通过的方向。B1相对原架构提升不等同于其超过A5；直接B1 vs B0也单独校正检验。未通过不宣告成功、不更换主指标或降低阈值。

三种子与小样本仍有限，精确受体聚类不等于同源独立验证；AAR不证明抗原/结合特异性。这里验证的是固定协议上的可重复提升，外部任务普适性需另行验证。全部最终文件还需要根任务复核；后台统计文件不会自行将总目标标为完成。

## 代码与结果

模型实现不改，新增独立`architecture_confirm`适配、统计和控制器。原A0-A7、第二轮与共同micro32探索代码/输出均保留。原共同micro32源码SHA `eb2fda14bc98a3386637d7b590233538464f6b6e78125d002ea2e14cc4e8f443`；本轮源码SHA `3c9c424d816b9fda7adfba5047d9e075ac15dc036342b59461edd35031e50971`。

入口：

```bash
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_confirm.sh --gpu 0
```

输出：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_confirm_20261031`。冻结源码：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_confirm_20261031/sources/3c9c424d816b9fda7adfba5047d9e075ac15dc036342b59461edd35031e50971`。

6项CPU检查通过：原初始化保持、种子与九组资格绑定、未完成时禁止稳定判定、效应量/方向/解码护栏、已知效应的配对bootstrap及错配拒绝、历史保护和快照导入。九个seed/架构组合的真实GPU资格检查均已通过；首组A0 seed42已正式训练，详见启动回执。

每组结束自动commit/push到`explore/arch70m-v8-20261028`，更新完整曲线PNG/PDF/SVG、原始逐步CSV、配对终点与统计；推送失败记pending，不覆盖训练退出码。三轮九组是多天计算预算，按实际记录报告进展，不提前承诺结果。

[CPU/快照验证](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_20261031/validation.json)
[实时进度与完整曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_20261031/REPORT.md)
[冻结判定](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_20261031/decision.json)

完整结束后`decision.json`和`continuation.json`给出通过的方向或仍未稳定的原因。若未稳定，目标继续未完成，需据负结果设计下一轮，不能把当前验证结束当成目标达成。

## 实际启动核验（2026-10-31T05:29:55Z UTC）

控制器PID307655，首组worker PID307866；实际--train-one/--variant A0/--seed42命令及快照工作目录已从/proc核验，绑定epochs3/3828步、micro32/global512和源码/数据SHA匹配。九组全部完成各3次maxlen2048真实更新，loss/梯度有限、显存低于原85%门槛。首组正式运行首次核验step2，本次记录step14；本次未进行checkpoint GPU数值重放。

| GPU资格组 | allocated GiB | reserved GiB | safe |
|---|---:|---:|---|
| A0_s42 | 36.37 | 44.01 | True |
| B0_s42 | 24.40 | 31.82 | True |
| B1_s42 | 24.40 | 32.01 | True |
| A0_s43 | 36.37 | 44.01 | True |
| B0_s43 | 24.40 | 31.82 | True |
| B1_s43 | 24.40 | 32.01 | True |
| A0_s44 | 36.37 | 44.01 | True |
| B0_s44 | 24.40 | 31.82 | True |
| B1_s44 | 24.40 | 32.01 | True |

[启动证据](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_20261031/startup_verified.json)。后续由后台控制器执行，不持续人工轮询；统计门槛未通过前不宣告稳定架构目标完成。
