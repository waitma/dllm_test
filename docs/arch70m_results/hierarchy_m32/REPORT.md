# 层次注意力探索：共同micro32

更新时间：2026-10-30T04:08:13Z UTC；完成4/4。

统一seed43、1epoch/1276步、global512=16×micro32，原V8形式及653,288条固定10%子集。所有变体先通过micro32/maxlen2048的3次真实更新与原85%显存门槛，再由全新进程正式训练。
B0是与A5架构相同的共同micro32对照；B1是全局＋可学习缩放；B2每层链内→全局→FFN2048；B3为B2＋可学习缩放。四组复用之前冻结的模型实现，不修改原目标/数据/评估。
主要比较仅使用本轮micro32终点。旧micro56结果独立保留为历史参考，不与本轮混作严格配对；单种子不作显著性/结合特异性结论。B2/B3同时重分配attention和FFN预算。

| 变体 | 状态 | step | 整链AAR↑ | CDR AAR↑ | AUROC↑ | AUPRC↑ | profile allocated/reserved GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| B2 | completed | 1276 | 0.2821 | 0.7906 | 0.8295 | 0.6671 | 42.22/49.11 |
| B3 | completed | 1276 | 0.2563 | 0.7973 | 0.8148 | 0.6464 | 42.31/49.20 |
| B0 (A5 control) | completed | 1276 | 0.3603 | 0.8056 | 0.8253 | 0.6730 | 24.40/31.82 |
| B1 | completed | 1276 | 0.3741 | 0.8005 | 0.8391 | 0.6905 | 24.40/32.01 |

![完整曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/hierarchy_m32/curves.png)

[逐步原始CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/hierarchy_m32/all_curves.csv) · [同micro终点差值](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/hierarchy_m32/comparisons.csv) · [历史micro56参考](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/hierarchy_m32/historical_micro56.csv)

原micro56失败及B1完成证据保持在 /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_round2_20261029。本轮输出 /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_hierarchy_m32_20261029。
