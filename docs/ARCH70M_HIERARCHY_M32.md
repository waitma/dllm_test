# 本地层次注意力探索：micro32共同对照

2026-10-29 UTC，用户指示“开始探索”。本轮补齐尚未正式训练的B2/B3，并用共同micro重跑A5/B1作主要对照。截至2026-10-29T14:13:34Z，四组GPU资格全部通过；B2已正式训练至核验时step7/1276，B3→A5(B0)→B1排队。启动后交后台执行，不持续人工轮询。

| 运行顺序 | ID | 架构 | 参数量 |
|---:|---|---|---:|
| 1 | B2 | 每层链内注意力→全局注意力→FFN2048，固定残差尺度0.1 | 75,748,047 |
| 2 | B3 | B2＋三个子层均可学习的通道残差尺度，初值0.1 | 75,766,479 |
| 3 | B0（A5对照） | 全局注意力＋固定残差尺度，FFN3072，与A5实现相同 | 75,741,903 |
| 4 | B1（同micro对照） | 全局注意力＋可学习残差尺度，FFN3072 | 75,754,191 |

全部在本机GPU0独立单GPU串行执行，只有seed43，不增加种子；各1epoch/1276步、global512=16×micro32。末个数据批488条由原采样器完整保留。LR1e-4、AdamW、warmup、bf16、确定性设置、整层梯度重计算、MLM/family目标、噪声/解码/评估代码均复用原冻结版本。

## 数据与可比性

原V8 prepared JSONL记录、链序列/顺序/角色、元数据、relation/family标签、词表/特殊token、固定链输入、reader和collator保持。字节相同的固定10% index SHA `4beaac6b54a1f46492533aeb77c2de2c6ac2a0bd8266fd57a6bb312679435c51`；653,288条训练记录，评估划分保持。

主要比较：B2对B0（参数近似匹配的层次架构组合）、B3对B2（层次架构下的残差缩放）、B3对B1（有可学习缩放时的架构组合）、B1对B0（全局架构下缩放作用）。B2/B3通过FFN3072→2048支付新增attention预算，因此不能把差异单独归因于注意力顺序。此为原模型序列结构的消融，不复现外部模型完整配方。

先前micro56的B2/B3在三次真实更新后达到reserved＋外部约91%，超过85%门槛，正式训练0步。原日志保持；micro56的A5/B1结果仅历史参考，不混入本轮同micro主比较。本轮在新目录从随机权重初始化，不沿用旧profile的权重或旧微批配置。

## 显存资格、恢复和发布

四组都先在micro32/maxlen2048上各做3次真实更新，要求loss/梯度有限，reserved＋外部显存不超过总显存85%。资格过程每组使用独立进程；四组全部通过后才启动正式训练。每个正式worker重新启动并初始化，资格记录以源码/数据/micro和文件SHA绑定；缺失、不匹配或被改动均拒绝启动。

每步本地JSONL记录loss、梯度、显存与GPU用时；每100步token评估，250步及epoch末完整screen任务评估和checkpoint；保留原六文件原子checkpoint、优化器/scheduler/RNG/数据游标和SHA回执。重启同命令可从latest恢复，已完成项只在绑定和最终checkpoint核验通过后跳过。

每组结束自动生成完整曲线PNG/PDF/SVG、逐步CSV及同micro终点差值，commit/push到`explore/arch70m-v8-20261028`。Git推送失败保存本地提交并记pending，不覆盖训练错误。共享发布锁保护报告/提交。仅做启动核验，此后不持续人工轮询。

## 路径与命令

新入口：

```bash
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_hierarchy.sh --gpu 0
```

代码目录：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/architecture_hierarchy`。输出：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_hierarchy_m32_20261029`。只适配冻结trainer的build/source_binding；原A0–A7与原B1–B3文件保持，模型定义不改。

原第二轮源码SHA：`93518fa2a14bc029e808cefae7aef8c38d79804683bb5274af27ff5f4e67d65e`。
本轮源码SHA：`eb2fda14bc98a3386637d7b590233538464f6b6e78125d002ea2e14cc4e8f443`。
快照：`/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_hierarchy_m32_20261029/sources/eb2fda14bc98a3386637d7b590233538464f6b6e78125d002ea2e14cc4e8f443`。

启动/恢复检查4项通过：错误micro/数据/源码或不安全profile拒绝；四组资格缺失/被篡改拒绝；完成项checkpoint损坏拒绝；原输出保护和独立快照适配导入。旧模型数学检查保持先前8项通过状态，本轮未重复运行。四组真实GPU资格检查均已通过；具体峰值和实际进程见下方启动回执。

[CPU及快照验证](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/hierarchy_m32/validation.json)
[状态、完整曲线与CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/hierarchy_m32/REPORT.md)

终点同时看验证CE、整链AAR/条件CE/长度MAE、CDR和表征AUROC/AUPRC，不只按训练loss排名。一个种子不作显著性或结合特异性结论；本轮尚无效果结果。

## 正式启动回执（2026-10-29T14:13:34Z UTC）

控制器PID217039，B2 worker PID217143；从/proc核验了实际命令及worker工作目录为本轮源码快照。四组都通过micro32/maxlen2048三次真实更新，loss和梯度有限，profile源码/数据/资格文件SHA均匹配；B2随后由全新进程开始正式训练，首次核验step1，本次核验step7。原85%门槛保持。

| 变体 | 最大长度allocated GiB | reserved GiB | reserved＋外部/总量 |
|---|---:|---:|---:|
| B2 | 42.22 | 49.11 | 52.44% |
| B3 | 42.31 | 49.20 | 52.54% |
| B0 | 24.40 | 31.82 | 34.25% |
| B1 | 24.40 | 32.01 | 34.44% |

[启动证据](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/hierarchy_m32/startup_verified.json)。本次只是启动成功检查，四组终点尚未完成；由后台控制器逐组保存、评估并发布完整曲线。
