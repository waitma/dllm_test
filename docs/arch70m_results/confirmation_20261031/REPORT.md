# 三种子完整架构确认

更新时间：2026-10-31T05:23:46Z UTC；完成0/9；decision=not_assessed。

A0原架构、B0=A5全局注意力、B1=全局＋可学习残差。配对seed42/43/44，各随机初始化3epoch/3828步；共同micro32/global512，原V8形式与固定10%数据。
候选来自既有screen筛选；所有九组完成后才对confirm留出终点作配对统计。训练过程中只用screen曲线观察，不按confirm结果选checkpoint。原抗体特异性五折评估一并保留，属于辅助验证。
主要门槛：三种子增益全同向，5000次seed→来源内受体cluster配对bootstrap，三对模型×两终点共六项比较使用Bonferroni(99.1667%双侧区间)，区间下界>0；生成平均相对CE增益≥2%并通过原解码质量guard≥2/3；表征平均AUROC增益≥0.005且平均AUPRC不回退。只宣告通过的方向，不声称全面提升。

| 变体 | seed | 状态 | step | confirm整链AAR | confirm CDR AAR | confirm AUROC | confirm AUPRC |
|---|---:|---|---:|---:|---:|---:|---:|
| A0 | 42 | pending | 0 | — | — | — | — |
| B0 | 42 | pending | 0 | — | — | — | — |
| B1 | 42 | pending | 0 | — | — | — | — |
| A0 | 43 | pending | 0 | — | — | — | — |
| B0 | 43 | pending | 0 | — | — | — | — |
| B1 | 43 | pending | 0 | — | — | — | — |
| A0 | 44 | pending | 0 | — | — | — | — |
| B0 | 44 | pending | 0 | — | — | — | — |
| B1 | 44 | pending | 0 | — | — | — | — |

![完整screen训练/验证曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_20261031/curves.png)

[逐步原始CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_20261031/all_curves.csv) · [终点汇总](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_20261031/summary.csv) · [冻结统计判定](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/confirmation_20261031/decision.json)

尚未完成全部配对不作稳定性结论；门槛不通过则目标仍未完成，保留负结果并继续设计下一轮，不降低阈值、不挑最好seed。三种子与小样本受体聚类限制保留；不是同源独立或普适下游能力证明。
训练GPU小时不含评估和checkpoint保存；模型按实验最新/最佳生成/最佳表征及epoch末节点保存完整恢复状态。
