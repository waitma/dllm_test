# A5全局注意力：QK归一化×单位初始门控，加原始A0对照

更新时间：2026-11-02T16:05:54Z UTC；完成0/5。

五组均随机初始化，seed43/1epoch/1276步、micro32/global512；原V8形式与固定653,288条10%子集。
C0为全新A5对照；C1仅Q/K每头RMS归一化；C2仅SDPA输出2sigmoid门控，零初始化使初始输出与A5相同；C3组合。
C0-C3的8层全局注意力、链内RoPE、chain bias、FFN3072、目标/数据/解码均保持；参数差异最大约0.1%，不改变FFN预算。另从头训练原始A0以作同micro/同epoch配方对照。
只用screen评估，固定epoch末终点；五组齐全才计算因子差值/交互。一种子不能证明稳定提升；已查看的旧confirm不是新候选的全新留出。

| 模型 | 状态 | step | 整链AAR | CDR AAR | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|
| C0 | training | 21 | — | — | — | — |
| C1 | qualified | 0 | — | — | — | — |
| C2 | qualified | 0 | — | — | — | — |
| C3 | qualified | 0 | — | — | — | — |
| A0 | qualified | 0 | — | — | — | — |

![完整loss/生成/表征曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/attention_20261102/curves.png)
![重点任务曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/attention_20261102/target_curves.png)

[逐步原始CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/attention_20261102/all_curves.csv) · [因子差值与交互](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/attention_20261102/factorial_effects.csv)

方法与参考文献见 docs/ARCH70M_ATTENTION_FACTORIAL.md；曲线训练CE同时含原始值和25步尾均值，任务验证不平滑。
