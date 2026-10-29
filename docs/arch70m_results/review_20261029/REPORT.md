# 架构探索结果复核（2026-10-29）

核查时间：2026-10-29T13:24:16Z UTC。
A0–A7：seed42与seed43各8组均完成1epoch/1276步；本地第二轮B1完成，B2/B3止于训练前显存门槛。GPU当前空闲，本次只审计、作图和整理结果，没有启动新任务。

## 同预算 seed43 终点

| 架构 | 验证token CE↓ | 整链AAR↑ | 长度MAE↓ | CDR AAR↑ | AUROC↑ | AUPRC↑ |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0.7161 | 3.42% | 99.03 | 79.54% | 0.8354 | 0.6714 |
| A1 | 0.7242 | 5.62% | 91.13 | 79.95% | 0.8242 | 0.6882 |
| A2 | 0.7190 | 4.77% | 99.91 | 79.79% | 0.8138 | 0.6807 |
| A3 | 0.7244 | 3.37% | 100.16 | 79.75% | 0.7953 | 0.6600 |
| A4 | 0.7154 | 3.32% | 100.20 | 79.23% | 0.8338 | 0.6835 |
| A5 | 0.7523 | 36.51% | 6.57 | 80.28% | 0.8279 | 0.6719 |
| A6 | 0.7143 | 7.40% | 95.67 | 80.03% | 0.8186 | 0.6609 |
| A7 | 0.7176 | 4.11% | 101.25 | 80.28% | 0.8124 | 0.6398 |
| B1 | 0.7520 | 35.35% | 8.36 | 80.50% | 0.8321 | 0.6895 |

## 目前能得出的结论

1. A5是目前整链生成最有重复性的候选：seed42/43整链AAR为37.02%/36.51%，A0为12.18%/3.42%；长度MAE分别5.61/6.57，对照76.50/99.03。A5在seed42用了micro96，seed43已经统一micro56，优势依然存在。
2. B1相对A5（同seed43）整链AAR下降1.16个百分点，CDR AAR上升0.22个百分点；AUROC增加0.0042、AUPRC增加0.0176。它提供一个可能更平衡的候选，尚无全面胜出的证据。B1的AUPRC在本次seed43九组中最高，AUROC仍低于A0。
3. A4在seed42的AUROC优势（0.8478对0.8252）未在seed43重现（0.8338对0.8354），不能延续“表征稳定最好”的说法。表征暂无稳定赢家。
4. A6在两个种子上的验证token CE最低（0.7160/0.7143），但整链AAR远低于A5。较低MLM loss不能直接推断更好的生成。
5. 两个种子只提供重复性线索；B1仅一个种子。当前是原screen划分上的探索，完整留出确认尚未针对这些候选重新完成。整链生成117条、CDR填补64条，表征两个验证来源共106条（33+73）；小幅差异不能视为统计显著或泛化改善。AAR也不证明抗原/结合特异性。

## B2/B3为何没有结果

| 变体 | 最大长度profile allocated | reserved | 外部 | reserved＋外部/总量 | 正式步数 |
|---|---:|---:|---:|---:|---:|
| B2 | 73.14GiB | 85.70GiB | 0.70GiB | 90.97% | 0 |
| B3 | 73.30GiB | 85.86GiB | 0.70GiB | 91.14% | 0 |

两组均完成3次micro56/maxlen2048试运行更新，随后因reserved＋外部显存超过总量85%主动停止。总量94.97GiB，门槛80.73GiB；日志没有OOM或NCCL失败。这是显存预算未通过，不能据此评价架构效果。若继续，应另行记录统一较小micro的协议并先实测，global512和原数据仍可保持；本次没有修改配方或重新启动。

## 曲线与审计

![各架构终点](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/review_20261029/endpoints.png)

![完整同seed曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/review_20261029/curves.png)

[终点CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/review_20261029/endpoints.csv) · [审计回执](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/review_20261029/audit.json)

17份终点指标已与eval日志匹配；每组1276条训练记录连续，loss/总loss/梯度均有限，消费653,288条、214,482,921个nonpad tokens。原数据index、settings、source、seed、epoch和micro绑定检查通过；额外逐一重算seed43九组最终checkpoint共54文件SHA并核验epoch/数据游标。未重跑推理或GPU数值重放。
原八卡[报告](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/seed43_validation/REPORT.md)与[逐步原始CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/seed43_validation/all_curves.csv)。本地第二轮[报告](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/round2/REPORT.md)与[逐步原始CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/round2/all_curves.csv)。
