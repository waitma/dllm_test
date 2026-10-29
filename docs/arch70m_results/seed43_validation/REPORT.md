# A0–A7：仅新增 seed 43

更新时间：2026-10-29T04:44:21Z；完成 4/8。

固定原V8形式、653,288条10%子集；每组从头初始化1epoch/1276步，global512。
八张卡各自执行一个单GPU进程，没有DDP/NCCL；统一micro batch由manifest记录。
与seed42同epoch终点比较；seed42原先A5的micro96、其余micro56，须保留该数值累积差异。
本轮只作第二种子复核，不能沿用原三种子显著性判定。

| 架构 | seed43状态 | step | 整链AAR | CDR AAR | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|
| A0 | completed | 1276 | 0.0342 | 0.7954 | 0.8354 | 0.6714 |
| A1 | training | 1142 | — | — | — | — |
| A2 | training | 1255 | — | — | — | — |
| A3 | training | 1123 | — | — | — | — |
| A4 | training | 1268 | — | — | — | — |
| A5 | completed | 1276 | 0.3651 | 0.8028 | 0.8279 | 0.6719 |
| A6 | completed | 1276 | 0.0740 | 0.8003 | 0.8186 | 0.6609 |
| A7 | completed | 1276 | 0.0411 | 0.8028 | 0.8124 | 0.6398 |

![完整曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/seed43_validation/curves.png)

[原始逐步曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/seed43_validation/all_curves.csv) · [同seed相对A0差值](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/seed43_validation/paired_vs_A0.csv)
