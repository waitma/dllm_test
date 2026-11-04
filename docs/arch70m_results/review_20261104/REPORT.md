# 架构探索整体复核：两个种子全部完成

复核时间：2026-11-04T00:44:49Z UTC。

最新C0/C1/C2/C3/A0×seed42/43全部10/10完成，均step1276；原V8形式和固定10%数据、micro32/global512一致。两个后台队列均finished且无失败；60份最终checkpoint文件SHA、绑定/完整训练步数与数据游标在本次独立核验通过。
下表仅为本轮相同one-epoch screen协议上的双种子均值，不能与历史三epoch confirm端点混为同一比较。

| 模型 | 整链AAR | CDR AAR | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| A0 | 7.92% | 79.71% | 0.8434 | 0.6385 |
| C0 | 36.31% | 79.75% | 0.8250 | 0.6802 |
| C1 | 37.03% | 78.56% | 0.8147 | 0.6577 |
| C2 | 36.16% | 80.12% | 0.8467 | 0.6752 |
| C3 | 34.63% | 78.54% | 0.8357 | 0.6966 |

C0=A5全局注意力；C1=额外Q/K归一化；C2=额外单位初始输出门控；C3=两者；A0=原始架构。

C2相对A5的AUROC在seed42/43分别增加0.02286/0.02068，AUPRC分别下降0.00311/0.00696；是表征方向较一致的候选，存在PR与生成取舍。相对原A0平均AUROC仅增加0.00333，不能据此宣称通过原稳定性标准。
C3在seed43的AUROC/AUPRC优势未复现：seed42相对A5分别下降0.01013/0.02977；CDR在两个种子均下降。C1也未形成一致的整体改善，其CDR和AUROC在两seed均低于A5。
A5整链AAR优势在本轮两seed继续复现（均值36.31%，原A0为7.92%），继续作为生成对照。
历史A0/A5/B1三seed三epoch完整确认与本地逐样本复核均已结束，冻结判定仍not_yet_stable；B1可学习残差与B2/B3层次方案没有形成已确认胜出方案。当前无通过预设稳定提升判据的架构。
本轮只有两训练种子，属于方向重复性检查；不追加未授权seed、不自动启动确认。旧confirm已查看，后续稳定性确认需另行预注册独立证据。

![两seed终点](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/review_20261104/endpoints.png)

[完整双seed曲线](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/attention_seed42_20261103/REPORT.md) · [终点CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/review_20261104/endpoints.csv) · [均值与样本标准差](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/review_20261104/two_seed_means.csv)
