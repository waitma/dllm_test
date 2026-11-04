# 相同五架构：仅新增seed42配对复核

更新时间：2026-11-04T00:44:48Z UTC；seed42完成5/5，seed43完成5/5；队列状态finished。

同一冻结模型/原V8形式/653,288条固定10%数据，均随机初始化1epoch/1276步、micro32/global512。只新增seed42；先等待原seed43队列完整结束，再做五组新GPU资格与训练。
只用screen固定最终checkpoint；两种子用于复核方向是否重复，不等同于三种子稳定性确认或新的未使用留出。全部10组齐全后才生成配对差值和两种子均值/样本标准差，不挑最好seed。

| 模型 | seed | 状态 | step | 整链AAR | CDR AAR | AUROC | AUPRC |
|---|---:|---|---:|---:|---:|---:|---:|
| C0 | 42 | completed | 1276 | 0.3659 | 0.7894 | 0.8247 | 0.6874 |
| C1 | 42 | completed | 1276 | 0.3601 | 0.7845 | 0.8066 | 0.6077 |
| C2 | 42 | completed | 1276 | 0.3769 | 0.8004 | 0.8475 | 0.6843 |
| C3 | 42 | completed | 1276 | 0.3392 | 0.7823 | 0.8146 | 0.6576 |
| A0 | 42 | completed | 1276 | 0.1229 | 0.7980 | 0.8416 | 0.6002 |
| C0 | 43 | completed | 1276 | 0.3603 | 0.8056 | 0.8253 | 0.6730 |
| C1 | 43 | completed | 1276 | 0.3805 | 0.7867 | 0.8228 | 0.7077 |
| C2 | 43 | completed | 1276 | 0.3462 | 0.8019 | 0.8459 | 0.6660 |
| C3 | 43 | completed | 1276 | 0.3534 | 0.7885 | 0.8569 | 0.7356 |
| A0 | 43 | completed | 1276 | 0.0354 | 0.7962 | 0.8452 | 0.6767 |

![完整loss/生成/表征曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/attention_seed42_20261103/curves.png)
![重点任务双种子曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/attention_seed42_20261103/target_curves.png)

[原始逐步CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/attention_seed42_20261103/all_curves.csv) · [同seed配对差值](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/attention_seed42_20261103/paired_comparisons.csv) · [两seed汇总](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/attention_seed42_20261103/two_seed_summary.csv)
