# 本地第二轮架构探索

更新时间：2026-10-29T04:54:26Z；新试验完成 0/3。

仅seed43；原V8形式与固定10%子集；global512、micro56、1epoch/1276步，冻结原MLM配方和评估代码。
A0/A4/A5复用八卡侧同seed、同预算结果；B1全局＋可学习残差，B2每层链内→全局→FFN2048，B3为B2＋可学习残差。
B2/B3重新分配attention/FFN参数预算，不能将差异只归于attention顺序；不复现PoET-2全套训练。
单种子筛选结果只用于继续探索；整链AAR不是结合特异性证据。

| 变体 | 类型 | 状态 | step | 整链AAR | CDR AAR | AUROC | AUPRC |
|---|---|---|---:|---:|---:|---:|---:|
| A0 | reference | completed | 1276 | 0.0342 | 0.7954 | 0.8354 | 0.6714 |
| A4 | reference | completed | 1276 | 0.0332 | 0.7923 | 0.8338 | 0.6835 |
| A5 | reference | completed | 1276 | 0.3651 | 0.8028 | 0.8279 | 0.6719 |
| B1 | new | pending | 0 | — | — | — | — |
| B2 | new | pending | 0 | — | — | — | — |
| B3 | new | pending | 0 | — | — | — | — |

![完整曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/round2/curves.png)

[原始CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/round2/all_curves.csv) · [同seed对照差值](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/round2/comparisons.csv)
