# A0–A7：仅新增 seed 43

更新时间：2026-10-29T01:29:49Z；完成 0/8。

固定原V8形式、653,288条10%子集；每组从头初始化1epoch/1276步，global512。
八张卡各自执行一个单GPU进程，没有DDP/NCCL；统一micro batch由manifest记录。
与seed42同epoch终点比较；seed42原先A5的micro96、其余micro56，须保留该数值累积差异。
本轮只作第二种子复核，不能沿用原三种子显著性判定。

| 架构 | seed43状态 | step | 整链AAR | CDR AAR | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|
| A0 | failed | 0 | — | — | — | — |
| A1 | failed | 0 | — | — | — | — |
| A2 | failed | 0 | — | — | — | — |
| A3 | failed | 0 | — | — | — | — |
| A4 | failed | 0 | — | — | — | — |
| A5 | failed | 0 | — | — | — | — |
| A6 | failed | 0 | — | — | — | — |
| A7 | failed | 0 | — | — | — | — |

![完整曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/seed43_validation/curves.png)

[原始逐步曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/seed43_validation/all_curves.csv) · [同seed相对A0差值](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/seed43_validation/paired_vs_A0.csv)
