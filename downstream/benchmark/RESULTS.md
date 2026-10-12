# AB / TCR 下游评测结果

<!-- v5_native_270m_best32000_20261012:start -->
## 270M 验证最佳 checkpoint 32000 下游评测

固定选点：2026-10-12T09:26:24.333933+00:00；268.59M，无ESMC，V5数据；epoch 2.14823，eval_loss 1.993467。按训练验证集最低eval_loss选取，与已完成的75.6M三组权重独立对照；本次为训练中期选点。

当前入口的Ophiuchus-Ab官方权重端到端复现尚未完成，不宣称论文复现已验收。同样37项；CDR复用AirGen对齐评测（Kong4步/SAb2步）；TCR采用单链禁止dot v2，CDR3提取/语法有效不代表全长β域或binding有效。原划分、探针、评分器保持；预训练重叠未完全排除。与45000对应修复协议比较，不能与其原无约束失败协议混用。
更新：2026-10-12T09:36:03.206669+00:00；状态：`{'success': 8, 'running': 4, 'pending': 25}`。选点及SHA：[selection](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/selection.json)。

| 任务 | 状态 | 结果／说明 | 证据 |
|---|---|---|---|
| cdr-kong | success | modes.cdrh1.airgen_aar_percent=75.0954；modes.cdrh1.mean_sequence_aar_percent=76.1766；modes.cdrh2.airgen_aar_percent=69.0671；modes.cdrh2.mean_sequence_aar_percent=69.3679；modes.cdrh3.airgen_aar_percent=41.9162；modes.cdrh3.mean_sequence_aar_percent=41.8838 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/cdr-kong) |
| cdr-sab23 | success | modes.cdrh3.airgen_aar_percent=34.8300；modes.cdrh3.mean_sequence_aar_percent=34.8286；modes.cdrh2.airgen_aar_percent=67.2300；modes.cdrh2.mean_sequence_aar_percent=67.2282；modes.cdrh1.airgen_aar_percent=74.7600；modes.cdrh1.mean_sequence_aar_percent=74.7619；modes.cdrl3.airgen_aar_percent=72.2900；modes.cdrl3.mean_sequence_aar_percent=72.2935；modes.cdrl2.airgen_aar_percent=79.4100 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/cdr-sab23) |
| pairing | running | 运行中 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/pairing) |
| specificity | success | mean.accuracy=0.6451；mean.f1_macro=0.6441；sample_sd_ddof1.accuracy=0.0075；sample_sd_ddof1.f1_macro=0.0077 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/specificity) |
| gdp_a1 | success | properties.AC-SINS_pH7.4.legacy_reference.cv_mean_spearman=0.4983；properties.AC-SINS_pH7.4.legacy_reference.pooled_oof_spearman=0.4879；properties.HIC.legacy_reference.cv_mean_spearman=0.2862；properties.HIC.legacy_reference.pooled_oof_spearman=0.2751；properties.PR_CHO.legacy_reference.cv_mean_spearman=0.4069；properties.PR_CHO.legacy_reference.pooled_oof_spearman=0.4463；properties.Titer.legacy_reference.cv_mean_spearman=0.2614；properties.Titer.legacy_reference.pooled_oof_spearman=0.1973；properties.Tm2.legacy_reference.cv_mean_spearman=0.0925 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/gdp_a1) |
| m396 | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/m396) |
| t1-cdr3b | running | 运行中 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/t1-cdr3b) |
| t1-others_cdr3b | success | eval_sets.seen_test.local.auroc_mean=0.6306；eval_sets.seen_test.local.auroc_std_sample=0.0159；eval_sets.seen_test.local.auprc_precrec_mean=0.6214；eval_sets.seen_test.local.auprc_precrec_std_sample=0.0189；eval_sets.seen_independent.local.auroc_mean=0.5980；eval_sets.seen_independent.local.auroc_std_sample=0.0207；eval_sets.seen_independent.local.auprc_precrec_mean=0.5996；eval_sets.seen_independent.local.auprc_precrec_std_sample=0.0159；eval_sets.unseen_independent.local.auroc_mean=0.5285 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/t1-others_cdr3b) |
| t1-cdr3ab | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/t1-cdr3ab) |
| t1-others_longab | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/t1-others_longab) |
| t2a | success | aligned_points.purity@ret~clusTCR(0.09).purity=0.9800；aligned_points.purity@ret~TCRMatch(0.09).purity=0.9800；aligned_points.purity@ret~GLIPH2(0.22).purity=0.8451；aligned_points.purity@ret~TCRdist3(0.44).purity=0.4523；aligned_points.purity@ret~GIANA(0.25).purity=0.7757；aligned_points.purity@ret~DeepTCR(0.92).purity=0.1790；aligned_points.purity@ret~HD(0.27).purity=0.7351；aligned_points.purity@ret~LD(0.31).purity=0.6464；aligned_points.purity@ret~iSMART(0.18).purity=0.9351 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/t2a) |
| t2b | success | kmeans_nmi_mean=0.1340；kmeans_nmi_best=0.1507；kmeans_purity_mean=0.3075；kmeans_purity_best=0.3282 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/t2b) |
| t3broad | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/t3broad) |
| t3deep | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/t3deep) |
| t4-benchmark14 | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/t4-benchmark14) |
| t4-held20 | running | 运行中 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/t4-held20) |
| uptcr | running | 运行中 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/uptcr) |
| ag-binding | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-binding) |
| ag-spr | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-spr) |
| ag-chimera | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-chimera) |
| ag-abbibench:1mhp_LC_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:1mhp_LC_benchmarking_data) |
| ag-abbibench:1mlc_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:1mlc_benchmarking_data) |
| ag-abbibench:1n8z_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:1n8z_benchmarking_data) |
| ag-abbibench:2fjg_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:2fjg_benchmarking_data) |
| ag-abbibench:3gbn_h1_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:3gbn_h1_benchmarking_data) |
| ag-abbibench:3gbn_h9_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:3gbn_h9_benchmarking_data) |
| ag-abbibench:4d5_her2_benchmarking_data_trimmed | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:4d5_her2_benchmarking_data_trimmed) |
| ag-abbibench:4fqi_h1_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:4fqi_h1_benchmarking_data) |
| ag-abbibench:4fqi_h3_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:4fqi_h3_benchmarking_data) |
| ag-abbibench:5a12_ang2_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:5a12_ang2_benchmarking_data) |
| ag-abbibench:5a12_vegf_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:5a12_vegf_benchmarking_data) |
| ag-abbibench:aayl49_ML_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:aayl49_ML_benchmarking_data) |
| ag-abbibench:aayl49_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:aayl49_benchmarking_data) |
| ag-abbibench:aayl50_LC_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:aayl50_LC_benchmarking_data) |
| ag-abbibench:aayl51_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:aayl51_benchmarking_data) |
| ag-abbibench:aayl52_LC_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:aayl52_LC_benchmarking_data) |
| ag-abbibench:g6_LC_benchmarking_data | pending | 待执行 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/ag-abbibench:g6_LC_benchmarking_data) |
<!-- v5_native_270m_best32000_20261012:end -->

<!-- checkpoint-comparison-20261012:start -->
## 2026-10-12 三个 checkpoint 完整对照

45000、138000、269000各37/37执行成功；138000与269000的iteration各12/12设置完成（各54个CDR格）。12个新云端作业均平台Success。执行成功不表示质量通过或全部子分析可用：SPR pKd探针仍受单位/删失信息限制。300000步训练已完成，未纳入本轮下游对照。
CDR采用AirGen对齐默认Kong4步/SAb2步；TCR生成均为单链禁止dot v2。只按相同协议比较，不用iteration测试集最高值替代默认成绩。

| 指标 | 45000 | 138000 | 269000 |
|---|---|---|---|
| Kong H3 AAR | 40.75% | 40.65% | 40.76% |
| SAb23H2 H3 AAR | 34.70% | 35.82% | 35.15% |
| Specificity accuracy | 62.41% | 64.53% | 65.28% |
| 配对 ImmunoMatch | 0.3823 | 0.4219 | 0.3802 |
| UpTCR zero-shot AUROC | 0.6831 | 0.7379 | 0.7274 |
| UpTCR few-shot AUROC | 0.7516 | 0.7592 | 0.7700 |
| UpTCR majority AUROC | 0.8646 | 0.8677 | 0.8716 |
| AB–Ag 固定抗体分组 AUROC | 0.4536 | 0.5586 | 0.6459 |
| CHIMERA 正确抗原条件配对有效率 | 12.71% | 6.76% | 17.17% |

主要观察：无统一最优checkpoint；H3收益很小，增加iteration没有稳定改善。TCR未见表位binding AUROC仍接近0.5；TCR语法有效率高不代表binding成功。CHIMERA正确/替换抗原条件的有效率接近，尚不能证明可靠的抗原条件生成。AB–Ag采用项目分组协议；预训练重叠未完全排除。
汇总证据：[comparison](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/completed_comparison_20261012.json)；原始结果：[45000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008)、[138000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009)、[269000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009)。
<!-- checkpoint-comparison-20261012:end -->

<!-- v5_cdr_iterations138000_20261009:start -->
## CDR infilling iteration 对照：checkpoint 138000

固定1/2/4/8/16/32步，全部Kong十折和SAb23H2六CDR；表内为AAR (%)。复用AirGen解码，原始位置/X掩码，argmax、temperature1、CFG0、batch64；Kong bf16、SAb fp32。只改变max_iter；默认协议仍为Kong4步/SAb2步。
这是测试集上的敏感性分析，不能挑最高值替换默认主成绩后宣称无偏提升。耗时为预加载模型后的整数据集墙钟时间（含评分与写盘），不作为严格硬件性能基准。
选点：[selection](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_iterations138000_20261009/selection.json)；逐样本预测与折分数：[产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_iterations138000_20261009)。

| Iteration | Kong H1 | H2 | H3 | SAb H1 | H2 | H3 | L1 | L2 | L3 | 状态 | 耗时 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 75.38 | 69.53 | 40.98 | 74.29 | 65.70 | 35.21 | 80.50 | 79.17 | 72.94 | kong: success; sab23: success | kong 56.9s; sab23 3.5s |
| 2 | 75.45 | 69.83 | 40.24 | 74.05 | 66.15 | 35.82 | 80.43 | 78.93 | 72.63 | kong: success; sab23: success | kong 64.5s; sab23 2.7s |
| 4 | 75.51 | 69.77 | 40.65 | 74.52 | 66.60 | 34.62 | 80.45 | 78.69 | 72.42 | kong: success; sab23: success | kong 83.8s; sab23 3.6s |
| 8 | 75.54 | 69.51 | 40.54 | 74.52 | 66.60 | 34.79 | 80.28 | 79.17 | 73.26 | kong: success; sab23: success | kong 125.4s; sab23 5.3s |
| 16 | 75.54 | 69.52 | 40.89 | 74.52 | 66.49 | 34.01 | 80.02 | 79.17 | 72.84 | kong: success; sab23: success | kong 199.3s; sab23 8.9s |
| 32 | 75.54 | 69.52 | 40.88 | 74.52 | 66.49 | 33.65 | 80.02 | 79.17 | 72.84 | kong: success; sab23: success | kong 356.1s; sab23 16.2s |
<!-- v5_cdr_iterations138000_20261009:end -->

<!-- v5_native_best138000_20261009:start -->
## 验证最佳 checkpoint 138000 下游评测

固定选点：2026-10-09T19:31:19.610404+00:00；75.6M，无ESMC，V5数据；epoch 9.26362，eval_loss 1.991442。按训练验证集最低eval_loss选取，与最新269000步独立对照。

同样37项；CDR复用AirGen对齐评测（Kong4步/SAb2步）；TCR采用单链禁止dot v2，CDR3提取/语法有效不代表全长β域或binding有效。原划分、探针、评分器保持；预训练重叠未完全排除。与45000对应修复协议比较，不能与其原无约束失败协议混用。
更新：2026-10-11T12:02:26.129143+00:00；状态：`{'success': 37}`。选点及SHA：[selection](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/selection.json)。

| 任务 | 状态 | 结果／说明 | 证据 |
|---|---|---|---|
| cdr-kong | success | modes.cdrh1.airgen_aar_percent=75.5110；modes.cdrh1.mean_sequence_aar_percent=76.5537；modes.cdrh2.airgen_aar_percent=69.7715；modes.cdrh2.mean_sequence_aar_percent=70.1243；modes.cdrh3.airgen_aar_percent=40.6462；modes.cdrh3.mean_sequence_aar_percent=40.7130 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/cdr-kong) |
| cdr-sab23 | success | modes.cdrh3.airgen_aar_percent=35.8200；modes.cdrh3.mean_sequence_aar_percent=35.8151；modes.cdrh2.airgen_aar_percent=66.1500；modes.cdrh2.mean_sequence_aar_percent=66.1488；modes.cdrh1.airgen_aar_percent=74.0500；modes.cdrh1.mean_sequence_aar_percent=74.0476；modes.cdrl3.airgen_aar_percent=72.6300；modes.cdrl3.mean_sequence_aar_percent=72.6291；modes.cdrl2.airgen_aar_percent=78.9300 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/cdr-sab23) |
| pairing | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/pairing) |
| specificity | success | mean.accuracy=0.6453；mean.f1_macro=0.6442；sample_sd_ddof1.accuracy=0.0147；sample_sd_ddof1.f1_macro=0.0142 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/specificity) |
| gdp_a1 | success | properties.AC-SINS_pH7.4.legacy_reference.cv_mean_spearman=0.4109；properties.AC-SINS_pH7.4.legacy_reference.pooled_oof_spearman=0.4260；properties.HIC.legacy_reference.cv_mean_spearman=0.2849；properties.HIC.legacy_reference.pooled_oof_spearman=0.2548；properties.PR_CHO.legacy_reference.cv_mean_spearman=0.4535；properties.PR_CHO.legacy_reference.pooled_oof_spearman=0.5107；properties.Titer.legacy_reference.cv_mean_spearman=0.2963；properties.Titer.legacy_reference.pooled_oof_spearman=0.2334；properties.Tm2.legacy_reference.cv_mean_spearman=0.1153 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/gdp_a1) |
| m396 | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/m396) |
| t1-cdr3b | success | eval_sets.seen_test.local.auroc_mean=0.7499；eval_sets.seen_test.local.auroc_std_sample=0.0027；eval_sets.seen_test.local.auprc_precrec_mean=0.7649；eval_sets.seen_test.local.auprc_precrec_std_sample=0.0027；eval_sets.seen_independent.local.auroc_mean=0.5817；eval_sets.seen_independent.local.auroc_std_sample=0.0020；eval_sets.seen_independent.local.auprc_precrec_mean=0.5596；eval_sets.seen_independent.local.auprc_precrec_std_sample=0.0021；eval_sets.unseen_independent.local.auroc_mean=0.5410 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/t1-cdr3b) |
| t1-others_cdr3b | success | eval_sets.seen_test.local.auroc_mean=0.6295；eval_sets.seen_test.local.auroc_std_sample=0.0061；eval_sets.seen_test.local.auprc_precrec_mean=0.6206；eval_sets.seen_test.local.auprc_precrec_std_sample=0.0122；eval_sets.seen_independent.local.auroc_mean=0.5918；eval_sets.seen_independent.local.auroc_std_sample=0.0280；eval_sets.seen_independent.local.auprc_precrec_mean=0.6110；eval_sets.seen_independent.local.auprc_precrec_std_sample=0.0163；eval_sets.unseen_independent.local.auroc_mean=0.5236 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/t1-others_cdr3b) |
| t1-cdr3ab | success | eval_sets.seen_test.local.auroc_mean=0.6660；eval_sets.seen_test.local.auroc_std_sample=0.0081；eval_sets.seen_test.local.auprc_precrec_mean=0.6610；eval_sets.seen_test.local.auprc_precrec_std_sample=0.0093；eval_sets.seen_independent.local.auroc_mean=0.5893；eval_sets.seen_independent.local.auroc_std_sample=0.0138；eval_sets.seen_independent.local.auprc_precrec_mean=0.5721；eval_sets.seen_independent.local.auprc_precrec_std_sample=0.0238；eval_sets.unseen_independent.local.auroc_mean=0.5326 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/t1-cdr3ab) |
| t1-others_longab | success | eval_sets.seen_test.local.auroc_mean=0.7884；eval_sets.seen_test.local.auroc_std_sample=0.0076；eval_sets.seen_test.local.auprc_precrec_mean=0.7767；eval_sets.seen_test.local.auprc_precrec_std_sample=0.0071；eval_sets.seen_independent.local.auroc_mean=0.5112；eval_sets.seen_independent.local.auroc_std_sample=0.0062；eval_sets.seen_independent.local.auprc_precrec_mean=0.5022；eval_sets.seen_independent.local.auprc_precrec_std_sample=0.0047；eval_sets.unseen_independent.local.auroc_mean=0.4917 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/t1-others_longab) |
| t2a | success | aligned_points.purity@ret~clusTCR(0.09).purity=0.9885；aligned_points.purity@ret~TCRMatch(0.09).purity=0.9885；aligned_points.purity@ret~GLIPH2(0.22).purity=0.8726；aligned_points.purity@ret~TCRdist3(0.44).purity=0.4517；aligned_points.purity@ret~GIANA(0.25).purity=0.7883；aligned_points.purity@ret~DeepTCR(0.92).purity=0.1731；aligned_points.purity@ret~HD(0.27).purity=0.7236；aligned_points.purity@ret~LD(0.31).purity=0.6601；aligned_points.purity@ret~iSMART(0.18).purity=0.9361 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/t2a) |
| t2b | success | kmeans_nmi_mean=0.1381；kmeans_nmi_best=0.1518；kmeans_purity_mean=0.3159；kmeans_purity_best=0.3277 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/t2b) |
| t3broad | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/t3broad) |
| t3deep | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/t3deep) |
| t4-benchmark14 | success | CDR3语法有效 1374/1400；summary.overall.f1=0.0007；summary.overall.n_scored_by_metric.f1=14.0000；summary.benchmark14.f1=0.0007；summary.benchmark14.n_scored_by_metric.f1=14.0000；summary.historical_bioseq_seen.f1=0.0000；summary.historical_bioseq_seen.n_scored_by_metric.f1=7.0000；summary.historical_bioseq_unseen.f1=0.0000；summary.historical_bioseq_unseen.n_scored_by_metric.f1=6.0000 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/t4-benchmark14) |
| t4-held20 | success | CDR3语法有效 1933/2000；summary.overall.f1=0.0017；summary.overall.n_scored_by_metric.f1=20.0000；summary.held20.f1=0.0017；summary.held20.n_scored_by_metric.f1=20.0000；summary.historical_bioseq_seen.f1=0.0019；summary.historical_bioseq_seen.n_scored_by_metric.f1=18.0000；summary.historical_bioseq_unseen.f1=0.0000；summary.historical_bioseq_unseen.n_scored_by_metric.f1=2.0000 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/t4-held20) |
| uptcr | success | auroc=0.7592；auprc_average_precision=0.4784；auroc=0.8677；auprc_average_precision=0.6836；auroc=0.7379；auprc_average_precision=0.3539；few-shot.auroc=0.7592；few-shot.auprc_average_precision=0.4784；majority.auroc=0.8677 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/uptcr) |
| ag-binding | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-binding) |
| ag-spr | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-spr) |
| ag-chimera | success | novelty.heavy.nearest_retrieved_identity.977fb4d7e199d1cf6439e9ef159c3fe553689606dbff413fa7f2d3494f5f8407=0.9390；novelty.heavy.nearest_retrieved_identity.1bc64c47eadf385d34d454b4d2fbfccc37059dfccf1e3b7148bb8897a8898797=0.9790；novelty.heavy.nearest_retrieved_identity.ccbce9dde6d30fb93caed06cace3bcd8ef8d2f1ba49046e9a3189260f6d6b135=0.9540；novelty.heavy.nearest_retrieved_identity.a5152b2c2926c621f273e6bb32d7e116f9b3f0e4c6cafbaa0938bef6ef106342=0.9020；novelty.heavy.nearest_retrieved_identity.5efa966acf6905a3071d4b2692b78c5d2751836f127d72ea4bfd80d19dd05588=0.9490；novelty.heavy.nearest_retrieved_identity.86a64e5921e1422eac12a3f2ea279416c85faa8f93c0ba304f1bd31dc5fb453e=0.9380；novelty.heavy.nearest_retrieved_identity.93877376f18e849f074e3f6fba92be8 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-chimera) |
| ag-abbibench:1mhp_LC_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:1mhp_LC_benchmarking_data) |
| ag-abbibench:1mlc_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:1mlc_benchmarking_data) |
| ag-abbibench:1n8z_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:1n8z_benchmarking_data) |
| ag-abbibench:2fjg_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:2fjg_benchmarking_data) |
| ag-abbibench:3gbn_h1_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:3gbn_h1_benchmarking_data) |
| ag-abbibench:3gbn_h9_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:3gbn_h9_benchmarking_data) |
| ag-abbibench:4d5_her2_benchmarking_data_trimmed | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:4d5_her2_benchmarking_data_trimmed) |
| ag-abbibench:4fqi_h1_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:4fqi_h1_benchmarking_data) |
| ag-abbibench:4fqi_h3_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:4fqi_h3_benchmarking_data) |
| ag-abbibench:5a12_ang2_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:5a12_ang2_benchmarking_data) |
| ag-abbibench:5a12_vegf_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:5a12_vegf_benchmarking_data) |
| ag-abbibench:aayl49_ML_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:aayl49_ML_benchmarking_data) |
| ag-abbibench:aayl49_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:aayl49_benchmarking_data) |
| ag-abbibench:aayl50_LC_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:aayl50_LC_benchmarking_data) |
| ag-abbibench:aayl51_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:aayl51_benchmarking_data) |
| ag-abbibench:aayl52_LC_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:aayl52_LC_benchmarking_data) |
| ag-abbibench:g6_LC_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/ag-abbibench:g6_LC_benchmarking_data) |
<!-- v5_native_best138000_20261009:end -->

<!-- v5_cdr_iterations269000_20261009:start -->
## CDR infilling iteration 对照：checkpoint 269000

固定1/2/4/8/16/32步，全部Kong十折和SAb23H2六CDR；表内为AAR (%)。复用AirGen解码，原始位置/X掩码，argmax、temperature1、CFG0、batch64；Kong bf16、SAb fp32。只改变max_iter；默认协议仍为Kong4步/SAb2步。
这是测试集上的敏感性分析，不能挑最高值替换默认主成绩后宣称无偏提升。耗时为预加载模型后的整数据集墙钟时间（含评分与写盘），不作为严格硬件性能基准。
选点：[selection](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_iterations269000_20261009/selection.json)；逐样本预测与折分数：[产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_iterations269000_20261009)。

| Iteration | Kong H1 | H2 | H3 | SAb H1 | H2 | H3 | L1 | L2 | L3 | 状态 | 耗时 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 75.49 | 70.12 | 40.85 | 74.52 | 68.20 | 34.68 | 80.90 | 80.12 | 72.79 | kong: success; sab23: success | kong 56.1s; sab23 2.9s |
| 2 | 75.53 | 70.18 | 39.72 | 74.29 | 68.76 | 35.15 | 80.24 | 79.64 | 72.15 | kong: success; sab23: success | kong 64.2s; sab23 2.7s |
| 4 | 75.60 | 70.05 | 40.76 | 74.52 | 67.16 | 33.65 | 80.90 | 79.41 | 71.95 | kong: success; sab23: success | kong 83.4s; sab23 3.6s |
| 8 | 75.55 | 70.04 | 40.29 | 74.52 | 67.60 | 34.85 | 80.48 | 79.88 | 71.81 | kong: success; sab23: success | kong 121.4s; sab23 5.3s |
| 16 | 75.55 | 70.03 | 39.89 | 74.52 | 67.60 | 34.54 | 80.11 | 79.88 | 71.81 | kong: success; sab23: success | kong 197.3s; sab23 8.9s |
| 32 | 75.55 | 70.03 | 40.05 | 74.52 | 67.60 | 35.04 | 80.11 | 79.88 | 71.81 | kong: success; sab23: success | kong 354.4s; sab23 16.0s |
<!-- v5_cdr_iterations269000_20261009:end -->

<!-- v5_native_latest269000_20261009:start -->
## 最新训练 checkpoint 269000 下游评测

固定选点：2026-10-09T19:24:02.492739+00:00；75.6M，无ESMC，V5数据；epoch 18.05733，eval_loss 2.008384。按最新完整checkpoint选取，不是验证最优选点。

同样37项；CDR复用AirGen对齐评测（Kong4步/SAb2步）；TCR采用单链禁止dot v2，CDR3提取/语法有效不代表全长β域或binding有效。原划分、探针、评分器保持；预训练重叠未完全排除。与45000对应修复协议比较，不能与其原无约束失败协议混用。
更新：2026-10-10T17:38:45.276045+00:00；状态：`{'success': 37}`。选点及SHA：[selection](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/selection.json)。

| 任务 | 状态 | 结果／说明 | 证据 |
|---|---|---|---|
| cdr-kong | success | modes.cdrh1.airgen_aar_percent=75.5992；modes.cdrh1.mean_sequence_aar_percent=76.5173；modes.cdrh2.airgen_aar_percent=70.0473；modes.cdrh2.mean_sequence_aar_percent=70.3039；modes.cdrh3.airgen_aar_percent=40.7592；modes.cdrh3.mean_sequence_aar_percent=40.8670 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/cdr-kong) |
| cdr-sab23 | success | modes.cdrh3.airgen_aar_percent=35.1500；modes.cdrh3.mean_sequence_aar_percent=35.1495；modes.cdrh2.airgen_aar_percent=68.7600；modes.cdrh2.mean_sequence_aar_percent=68.7599；modes.cdrh1.airgen_aar_percent=74.2900；modes.cdrh1.mean_sequence_aar_percent=74.2857；modes.cdrl3.airgen_aar_percent=72.1500；modes.cdrl3.mean_sequence_aar_percent=72.1456；modes.cdrl2.airgen_aar_percent=79.6400 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/cdr-sab23) |
| pairing | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/pairing) |
| specificity | success | mean.accuracy=0.6528；mean.f1_macro=0.6515；sample_sd_ddof1.accuracy=0.0048；sample_sd_ddof1.f1_macro=0.0044 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/specificity) |
| gdp_a1 | success | properties.AC-SINS_pH7.4.legacy_reference.cv_mean_spearman=0.4805；properties.AC-SINS_pH7.4.legacy_reference.pooled_oof_spearman=0.4857；properties.HIC.legacy_reference.cv_mean_spearman=0.3695；properties.HIC.legacy_reference.pooled_oof_spearman=0.3380；properties.PR_CHO.legacy_reference.cv_mean_spearman=0.4600；properties.PR_CHO.legacy_reference.pooled_oof_spearman=0.5145；properties.Titer.legacy_reference.cv_mean_spearman=0.2897；properties.Titer.legacy_reference.pooled_oof_spearman=0.2317；properties.Tm2.legacy_reference.cv_mean_spearman=0.0658 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/gdp_a1) |
| m396 | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/m396) |
| t1-cdr3b | success | eval_sets.seen_test.local.auroc_mean=0.7534；eval_sets.seen_test.local.auroc_std_sample=0.0020；eval_sets.seen_test.local.auprc_precrec_mean=0.7685；eval_sets.seen_test.local.auprc_precrec_std_sample=0.0026；eval_sets.seen_independent.local.auroc_mean=0.5909；eval_sets.seen_independent.local.auroc_std_sample=0.0053；eval_sets.seen_independent.local.auprc_precrec_mean=0.5685；eval_sets.seen_independent.local.auprc_precrec_std_sample=0.0050；eval_sets.unseen_independent.local.auroc_mean=0.5250 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/t1-cdr3b) |
| t1-others_cdr3b | success | eval_sets.seen_test.local.auroc_mean=0.6368；eval_sets.seen_test.local.auroc_std_sample=0.0084；eval_sets.seen_test.local.auprc_precrec_mean=0.6248；eval_sets.seen_test.local.auprc_precrec_std_sample=0.0150；eval_sets.seen_independent.local.auroc_mean=0.5950；eval_sets.seen_independent.local.auroc_std_sample=0.0131；eval_sets.seen_independent.local.auprc_precrec_mean=0.6135；eval_sets.seen_independent.local.auprc_precrec_std_sample=0.0069；eval_sets.unseen_independent.local.auroc_mean=0.5349 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/t1-others_cdr3b) |
| t1-cdr3ab | success | eval_sets.seen_test.local.auroc_mean=0.6670；eval_sets.seen_test.local.auroc_std_sample=0.0092；eval_sets.seen_test.local.auprc_precrec_mean=0.6616；eval_sets.seen_test.local.auprc_precrec_std_sample=0.0108；eval_sets.seen_independent.local.auroc_mean=0.5707；eval_sets.seen_independent.local.auroc_std_sample=0.0181；eval_sets.seen_independent.local.auprc_precrec_mean=0.5630；eval_sets.seen_independent.local.auprc_precrec_std_sample=0.0133；eval_sets.unseen_independent.local.auroc_mean=0.5383 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/t1-cdr3ab) |
| t1-others_longab | success | eval_sets.seen_test.local.auroc_mean=0.7933；eval_sets.seen_test.local.auroc_std_sample=0.0134；eval_sets.seen_test.local.auprc_precrec_mean=0.7840；eval_sets.seen_test.local.auprc_precrec_std_sample=0.0123；eval_sets.seen_independent.local.auroc_mean=0.4944；eval_sets.seen_independent.local.auroc_std_sample=0.0149；eval_sets.seen_independent.local.auprc_precrec_mean=0.4915；eval_sets.seen_independent.local.auprc_precrec_std_sample=0.0124；eval_sets.unseen_independent.local.auroc_mean=0.4852 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/t1-others_longab) |
| t2a | success | aligned_points.purity@ret~clusTCR(0.09).purity=0.9826；aligned_points.purity@ret~TCRMatch(0.09).purity=0.9826；aligned_points.purity@ret~GLIPH2(0.22).purity=0.8546；aligned_points.purity@ret~TCRdist3(0.44).purity=0.4413；aligned_points.purity@ret~GIANA(0.25).purity=0.7481；aligned_points.purity@ret~DeepTCR(0.92).purity=0.1783；aligned_points.purity@ret~HD(0.27).purity=0.7196；aligned_points.purity@ret~LD(0.31).purity=0.6350；aligned_points.purity@ret~iSMART(0.18).purity=0.9362 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/t2a) |
| t2b | success | kmeans_nmi_mean=0.1342；kmeans_nmi_best=0.1492；kmeans_purity_mean=0.3108；kmeans_purity_best=0.3282 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/t2b) |
| t3broad | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/t3broad) |
| t3deep | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/t3deep) |
| t4-benchmark14 | success | CDR3语法有效 1357/1400；summary.overall.f1=0.0007；summary.overall.n_scored_by_metric.f1=14.0000；summary.benchmark14.f1=0.0007；summary.benchmark14.n_scored_by_metric.f1=14.0000；summary.historical_bioseq_seen.f1=0.0000；summary.historical_bioseq_seen.n_scored_by_metric.f1=7.0000；summary.historical_bioseq_unseen.f1=0.0000；summary.historical_bioseq_unseen.n_scored_by_metric.f1=6.0000 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/t4-benchmark14) |
| t4-held20 | success | CDR3语法有效 1955/2000；summary.overall.f1=0.0045；summary.overall.n_scored_by_metric.f1=20.0000；summary.held20.f1=0.0045；summary.held20.n_scored_by_metric.f1=20.0000；summary.historical_bioseq_seen.f1=0.0050；summary.historical_bioseq_seen.n_scored_by_metric.f1=18.0000；summary.historical_bioseq_unseen.f1=0.0000；summary.historical_bioseq_unseen.n_scored_by_metric.f1=2.0000 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/t4-held20) |
| uptcr | success | auroc=0.7700；auprc_average_precision=0.4941；auroc=0.8716；auprc_average_precision=0.6914；auroc=0.7274；auprc_average_precision=0.3531；few-shot.auroc=0.7700；few-shot.auprc_average_precision=0.4941；majority.auroc=0.8716 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/uptcr) |
| ag-binding | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-binding) |
| ag-spr | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-spr) |
| ag-chimera | success | novelty.heavy.nearest_retrieved_identity.a0cad9026dae3e8bce98ecd83fe5177b19f19fa1c884b47755759896cfb82500=0.9530；novelty.heavy.nearest_retrieved_identity.820d60d82051374a96899f40bd0f18cac36dbcec55e4277e81c41bea2367d26b=0.9520；novelty.heavy.nearest_retrieved_identity.39eef45403cf735e187fa172b1cb90d319e5a957a64f831f1903a85e888814a3=0.9530；novelty.heavy.nearest_retrieved_identity.e5f193866c9a6e68ef912dfa30f99a851689022bb0210eeef186b7955e322c32=0.9530；novelty.heavy.nearest_retrieved_identity.be015ea5a3be80eafd16731c1ce506f7827b61fc1a4a3079c2f44afed1f39f16=0.9530；novelty.heavy.nearest_retrieved_identity.31708ad7cd987b141f1b176dbd2532112dc3f388a5a3cff8a283269e62ab5652=0.9050；novelty.heavy.nearest_retrieved_identity.af165d468f44026782822a250ba77e6 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-chimera) |
| ag-abbibench:1mhp_LC_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:1mhp_LC_benchmarking_data) |
| ag-abbibench:1mlc_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:1mlc_benchmarking_data) |
| ag-abbibench:1n8z_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:1n8z_benchmarking_data) |
| ag-abbibench:2fjg_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:2fjg_benchmarking_data) |
| ag-abbibench:3gbn_h1_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:3gbn_h1_benchmarking_data) |
| ag-abbibench:3gbn_h9_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:3gbn_h9_benchmarking_data) |
| ag-abbibench:4d5_her2_benchmarking_data_trimmed | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:4d5_her2_benchmarking_data_trimmed) |
| ag-abbibench:4fqi_h1_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:4fqi_h1_benchmarking_data) |
| ag-abbibench:4fqi_h3_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:4fqi_h3_benchmarking_data) |
| ag-abbibench:5a12_ang2_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:5a12_ang2_benchmarking_data) |
| ag-abbibench:5a12_vegf_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:5a12_vegf_benchmarking_data) |
| ag-abbibench:aayl49_ML_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:aayl49_ML_benchmarking_data) |
| ag-abbibench:aayl49_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:aayl49_benchmarking_data) |
| ag-abbibench:aayl50_LC_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:aayl50_LC_benchmarking_data) |
| ag-abbibench:aayl51_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:aayl51_benchmarking_data) |
| ag-abbibench:aayl52_LC_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:aayl52_LC_benchmarking_data) |
| ag-abbibench:g6_LC_benchmarking_data | success | 已完成，详见产物 | [产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/ag-abbibench:g6_LC_benchmarking_data) |
<!-- v5_native_latest269000_20261009:end -->

<!-- v5-repair-20261009 -->
**2026-10-09 修复复测已终态**：T1五折预测仅恢复基线比较表；CHIMERA隔离单条ANARCI异常并重评分；TCR两组采用单链禁止dot v2协议复测，原始3,400条链结构无效候选保留。CDR3可提取/语法有效不等于完整β链有效或binding有效；旧无约束与新受约束结果不能混用。60组T1 fold/split预测独立重算AUROC通过，未见表位弱表现并非此次发现的汇总错误。修复证据：[audit](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/repair_audit_20261009.json)。

<!-- airgen-aligned-cdr:start -->
## 当前模型 CDR infilling：AirGen 协议对齐重评

固定best45000。Kong H1/H2/H3：官方位置字段、10折、argmax/4步；SAb23H2六CDR：原始design FASTA的X掩码、argmax/2步（AirGen示例命令）。batch64、temperature1、CFG0；直接复用AirGen解码方法，按原始logits置信度线性填充及其阈值并列规则。
这是任务/解码协议对齐，不是模型实现相同：保留训练过的V5对象302布局和50词表；AirGen的ESM词表中gap `-` 在本模型中不存在。旧iter1/首次子串匹配结果仅作历史对照，不作为本次对齐结果。Kong仍为既有本地重建划分，不宣称已复现未公开的论文运行参数。

| 数据集 | 状态 | 各CDR AAR (%) | 证据 |
|---|---|---|---|
| kong | success | cdrh1 74.0811927053696；cdrh2 68.10254104402011；cdrh3 40.75437339170653 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_airgen_aligned_20261008/kong) |
| sab23 | success | cdrh3 34.699999999999996；cdrh2 67.54；cdrh1 74.52；cdrl3 71.74000000000001；cdrl2 79.25999999999999；cdrl1 79.63 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_airgen_aligned_20261008/sab23) |
<!-- airgen-aligned-cdr:end -->

<!-- native-v5-best-20261008:start -->
## 当前 V5 数据新模型：best checkpoint 45,000 全下游评测

更新：2026-10-09T22:50:58.861035+00:00。固定选点：训练验证 loss **1.992721**，step **45,000**，epoch **3.02074**；75.6M、无ESMC、V5对象302布局。独立评测，不停止当前训练。

评测范围为现行 Ours AB/TCR 全任务，包含 UpTCR 与 AB–AG 三任务四子集；不重跑外部baseline，不纳入 VHH/Humanization/PPI/NbBench/FLAb。旧表保留为历史对照。
输入适配沿用原划分、标签、探针和评分器；表征排除synthetic位置，CDR补全使用已知区间，生成由模型预测终止符、不给真实目标长度。TCR修复v2在采样时禁止单链任务的dot，原失败输出保留于attempt_history；CDR3提取成功不代表完整β链有效或实验证实binding。新生成协议与旧固定长度结果分开解释。V5预训练重叠未完全排除，不宣称无泄漏。

状态计数：`{"success": 37}`。未完成项没有成绩；失败项保留具体原因。选择与完整计划：[selection.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/selection.json)、[plan.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/plan.json)。

| 任务 | 状态 | 当前指标／说明 | 证据 |
|---|---|---|---|
| cdr-kong | success | average_aar=74.9822；average_aar_all_folds=73.9670；average_aar=41.3530；average_aar_all_folds=41.3532；average_aar=68.2024；average_aar_all_folds=67.8824 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/cdr-kong) |
| cdr-sab23 | success | average_aar=79.5254；average_aar_all_folds=79.5254；average_aar=74.7619；average_aar_all_folds=74.7619；average_aar=71.8814；average_aar_all_folds=71.8814 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/cdr-sab23) |
| pairing | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/pairing) |
| specificity | success | mean.accuracy=0.6241；mean.f1_macro=0.6232；sample_sd_ddof1.accuracy=0.0154；sample_sd_ddof1.f1_macro=0.0150 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/specificity) |
| gdp_a1 | success | properties.AC-SINS_pH7.4.legacy_reference.cv_mean_spearman=0.3582；properties.AC-SINS_pH7.4.legacy_reference.pooled_oof_spearman=0.3521；properties.HIC.legacy_reference.cv_mean_spearman=0.2781；properties.HIC.legacy_reference.pooled_oof_spearman=0.2487；properties.PR_CHO.legacy_reference.cv_mean_spearman=0.3938；properties.PR_CHO.legacy_reference.pooled_oof_spearman=0.4450 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/gdp_a1) |
| m396 | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/m396) |
| t1-cdr3b | success | metrics.auroc=0.7509；metrics.auprc_precrec=0.7656；metrics.accuracy=0.6702；metrics.f1=0.6832；metrics.auroc=0.5124；metrics.auprc_precrec=0.5015 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/t1-cdr3b) |
| t1-others_cdr3b | success | metrics.auroc=0.6300；metrics.auprc_precrec=0.6268；metrics.accuracy=0.5917；metrics.f1=0.5944；metrics.auroc=0.5095；metrics.auprc_precrec=0.5075 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/t1-others_cdr3b) |
| t1-cdr3ab | success | metrics.auroc=0.6617；metrics.auprc_precrec=0.6677；metrics.accuracy=0.6200；metrics.f1=0.5749；metrics.auroc=0.4846；metrics.auprc_precrec=0.4894 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/t1-cdr3ab) |
| t1-others_longab | success | metrics.auroc=0.7807；metrics.auprc_precrec=0.7764；metrics.accuracy=0.6900；metrics.f1=0.7049；metrics.auroc=0.5122；metrics.auprc_precrec=0.5189 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/t1-others_longab) |
| t2a | success | aligned_points.purity@ret~clusTCR(0.09).purity=0.9838；aligned_points.purity@ret~TCRMatch(0.09).purity=0.9838；aligned_points.purity@ret~GLIPH2(0.22).purity=0.8812；aligned_points.purity@ret~TCRdist3(0.44).purity=0.4760；aligned_points.purity@ret~GIANA(0.25).purity=0.7985；aligned_points.purity@ret~DeepTCR(0.92).purity=0.1688 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/t2a) |
| t2b | success | kmeans_nmi_mean=0.1370；kmeans_nmi_best=0.1536；kmeans_purity_mean=0.3107；kmeans_purity_best=0.3268 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/t2b) |
| t3broad | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/t3broad) |
| t3deep | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/t3deep) |
| t4-benchmark14 | success | 单链禁止dot v2；CDR3提取/语法有效 1393/1400（非完整β链或binding验证）；summary.overall.f1=0.0000；summary.overall.n_scored_by_metric.f1=14.0000；summary.benchmark14.f1=0.0000；summary.benchmark14.n_scored_by_metric.f1=14.0000；summary.historical_bioseq_seen.f1=0.0000；summary.historical_bioseq_seen.n_scored_by_metric.f1=7.0000 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/t4-benchmark14) |
| t4-held20 | success | 单链禁止dot v2；CDR3提取/语法有效 1974/2000（非完整β链或binding验证）；summary.overall.f1=0.0069；summary.overall.n_scored_by_metric.f1=20.0000；summary.held20.f1=0.0069；summary.held20.n_scored_by_metric.f1=20.0000；summary.historical_bioseq_seen.f1=0.0076；summary.historical_bioseq_seen.n_scored_by_metric.f1=18.0000 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/t4-held20) |
| uptcr | success | auroc=0.6831；auprc_average_precision=0.2763；auroc=0.7516；auprc_average_precision=0.4591；auroc=0.8646；auprc_average_precision=0.6794 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/uptcr) |
| ag-binding | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-binding) |
| ag-spr | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-spr) |
| ag-chimera | success | novelty.heavy.nearest_retrieved_identity.1726f101739bcce61ac2cf1e2a8d0abd748fada5d5c10d20f2f0b9f9b048e70b=0.8840；novelty.heavy.nearest_retrieved_identity.58ba962e4819bcffcb18813f194a897a9022990894f71fb88b909d9bffdc728c=0.8970；novelty.heavy.nearest_retrieved_identity.e3cd8336dc53563b964655635685733e3984fd0238ac441f1191aa903b5af8f8=0.8970；novelty.heavy.nearest_retrieved_identity.54764d1d960cf0defa0633525e4c4cf7079dcd50f14142bc1e32c40dcff87faf=0.9590；novelty.heavy.nearest_retrieved_identity.ad610f3ade13454fa52884665668d8c9d3fc4cf9f1bf6e087235f1f22 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-chimera) |
| ag-abbibench:1mhp_LC_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:1mhp_LC_benchmarking_data) |
| ag-abbibench:1mlc_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:1mlc_benchmarking_data) |
| ag-abbibench:1n8z_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:1n8z_benchmarking_data) |
| ag-abbibench:2fjg_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:2fjg_benchmarking_data) |
| ag-abbibench:3gbn_h1_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:3gbn_h1_benchmarking_data) |
| ag-abbibench:3gbn_h9_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:3gbn_h9_benchmarking_data) |
| ag-abbibench:4d5_her2_benchmarking_data_trimmed | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:4d5_her2_benchmarking_data_trimmed) |
| ag-abbibench:4fqi_h1_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:4fqi_h1_benchmarking_data) |
| ag-abbibench:4fqi_h3_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:4fqi_h3_benchmarking_data) |
| ag-abbibench:5a12_ang2_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:5a12_ang2_benchmarking_data) |
| ag-abbibench:5a12_vegf_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:5a12_vegf_benchmarking_data) |
| ag-abbibench:aayl49_ML_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:aayl49_ML_benchmarking_data) |
| ag-abbibench:aayl49_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:aayl49_benchmarking_data) |
| ag-abbibench:aayl50_LC_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:aayl50_LC_benchmarking_data) |
| ag-abbibench:aayl51_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:aayl51_benchmarking_data) |
| ag-abbibench:aayl52_LC_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:aayl52_LC_benchmarking_data) |
| ag-abbibench:g6_LC_benchmarking_data | success | 已完成，详见产物 | [目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/ag-abbibench:g6_LC_benchmarking_data) |
<!-- native-v5-best-20261008:end -->

**2026-10-07 V7 两轮生成评测：** [CDR补全](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr)和[轻链生成](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-pairing)已完成并附V5对照；[TCR生成](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#tcr-generation)也已完成；本轮计划的三类生成评测均有实际结果。下方带旧日期的整矩阵记录是历史结果，不代表本次所有任务均已完成。

<a id="result-integrity"></a>

### 2026-09-30 结果完整性与解释边界

本次只生成派生报告并补齐可复算能力，不改历史指标和划分。当前范围引用 [README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md)，报告/依赖/命令见 [执行约定](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md#report-integrity)。

- T1：[论文主列与本地附列](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_integrity_20260930/final/t1/summary_original_baselines.csv) 已按固定 `[P]` 规则生成，缺项不补值；[机器审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_integrity_20260930/final/baseline_audit/baseline_provenance_audit.json) 校验来源一致性。
- T3：论文 ESM2-T6-8M 与本地 150M 分开；[报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_integrity_20260930/final/t3_paper6/_summary.json) 中未知或不一致协议不输出差值/排名。下列历史分数保留作各自协议的参考，不作跨协议 headline。
- CDR：历史只有日志的行是 `aggregate_only`，不能逐样本复算；新运行保存 manifest/predictions/metrics，partial 不作完整成绩。
- GDPa1：同 CV 调参/汇报与标签变换限制继续保留，仅为 `legacy_reference`；Pairing 的长度条件逐 manifest 标注，单 seed 和历史测试集扫参不能支持独立泛化结论。
- SPR 排序与 pKD 回归分别报状态；pKD 的单位/删失问题仍阻塞回归，不能以排序完成代替。所有 endpoint、checkpoint 和预训练暴露证据统一见 [独立报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_integrity_20260930/final/results/integrity_report.json)；未知证据保留 unknown。


> 2026-09-30 UpTCR 原发布三套划分的本地评测已完成，当前成绩与比较边界见 [UpTCR](#uptcr-binding-status)；逐 endpoint 与证据状态见 [独立报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_integrity_20260930/final/results/integrity_report.json)。

> 2026-09-24 范围同步：BioMap ΔG / A2binder 官方 baseline 首轮执行获用户授权，单独记录，不等同 KD/pKD；范围以 [README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md) 为准，状态见 [BioMap 官方运行](#biomap-official-run)。

2026-09-24 新增范围见 [README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md)；抗原条件 paired AB 的新任务状态见 [AB-AG 执行状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-antigen)，历史 AB/TCR 成绩保持各自原协议。

更新：2026-09-27 19:45Z。no-ESMC **150000 × 16 已全部 Success**，主表 150000 行已按 `ab_v2_noesmc_v6_150000_20260928` / `ours_tcr_v2_noesmc_v6_150000_20260928*` 原始 log/JSON 回填；**110000 / ESMC 15000 数字未改**。看护 17:08Z 正常退出。台账：[20260928 submissions](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v2_immune_v6_eval_20260928/submissions.json)。训练未改。

更新：2026-09-28。no-ESMC 最佳 val 更新为 **checkpoint-150000 / 0.709956**（优于已评的 110000 / 0.712413）。已另提 16 条官方 Ours 闲时评测，输出目录与 110000 分开；**主表 110000 数字保留**。台账：[20260928 submissions](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v2_immune_v6_eval_20260928/submissions.json)。训练 `t-20260924185250-w22zf` / `t-20260923185956-wsvpq` 未改。

更新：2026-09-27。V6 官方评测 **32/32 Success**（看护 08:38Z 正常退出）；主表已按原始 log/JSON 回填，含 pairing / Specificity-Fv / m396-Fv / T1 四轨 / T4 held20。最佳 val：ESMC fusion 15000 / 0.739；no-ESMC 110000 / 0.712（现已被 150000 超过，110000 行仍保留）。状态与 task ID 见 [V6 评测状态](#v2-immune-v6-eval)。v5 159000 主行仍是已完成对照，不被 V6 行替换。

更新：2026-09-17。上半部分 AB，下半部分 TCR；每个任务先列 baseline，Ours 在表末，checkpoint 与协议不混用。

**v5 159000 AB＋TCR 最终结果已更新（2026-09-17）**：AB 六项、TCR 十项均完成，使用独立留存 [v5 159000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_159000_full_20260917)，SHA256 `01b3d1446b21f56b3a3658528a405cb34dee7dc889eb1f162e423d64141faf3e`。该轮 Ours 为 **159000**，49000/92000/140000 历史结果保留；新增V7生成结果见顶部更新。Pairing 固定 **p0 / CFG=0 / iter=16**、reference 长度、gumbel_argmax、500×8；140000 为124步，因此差异不能只归因于权重。CDR 为 argmax / iter=2，Specificity 为五折固定末轮100 epoch。TCR 沿用现有协议（T4仍为32步，不随AB pairing改成16步）。[AB产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917)、[TCR任务索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_159000_20260917/all_159000_tcr_submissions.json)、[汇总核对记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_159000_20260917/results_document_audit.json)。本轮核对产物身份、数量、折完整性及部分聚合，不重跑前向、IM评分器或完整逐预测重评分；不宣称严格论文复现或预训练去污染。

49000 整矩阵仍保留为已验收对照。**AB pairing 159000 主行为p0（无前三残基、16步），140000 p0为124步**；49000 主表行仍是 ESMC 反馈修复后的 **p3（有前三残基）**。TCR 49000 完整性以 [最终汇总 JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/comparison_v1/final_49000_vs_baselines_and_29000.json) 为准；92000/140000/159000 数字来自各任务产物 JSON/聚合日志，不是该 49000 comparison 的重算。

- **AB**：[CDR 补全](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr) · [轻链配对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-pairing) · [Specificity / GDPa1 / m396](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-probes)
- **TCR**：[Binding 四轨](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#tcr-binding) · [Clustering A/B](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#tcr-clustering) · [Representation broad/deep](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#tcr-representation) · [Generation](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#tcr-generation)
- **进行中**：[V6 评测状态](#v2-immune-v6-eval)

<a id="v2-immune-v6-eval"></a>
<a id="v6-eval"></a>

### V6 官方评测（2026-09-27 矩阵 32/32 Success；2026-09-28 no-ESMC 150000 16/16 Success，主表已回填）

**版本命名**：按训练数据版本统一称为 **V6**（`immune_v6_binding_only`）。历史文件路径、任务 ID 和 `fixed-canvas v2` 架构标记保留，均不代表另一个数据版本。

对两条仍在训练的 V6 模型评 **RESULTS 官方 Ours 全表**：AB CDR Kong / SAb23H2、pairing p0/CFG0、Specificity、GDPa1、m396；TCR T1 四轨、T2A/T2B、T3 broad/deep、T4 benchmark14/held20。**不评** AB-AG / MAGE / BioMap / S²ALM（那些是官方 baseline 轨，不是这两路模型的 Ours 矩阵），也不评已冻结的 Humanization / PPI / NbBench / FLAb。

选点 = live top-k **最佳 val**，不评 latest（融合 latest 31k eval 0.790 差于 15k 0.739；无 ESMC latest 134k 0.717 差于 110k 0.712）。独立快照 + 训练目录 hardlink `eval_snapshot_*`（nlink=3），避免 top-k 删评测输入。

| 版本 | step / val | SHA256 | 快照 | 训练任务（未动） |
|---|---|---|---|---|
| ESMC fusion，`residue_cond_mode=add`，source_balanced=True | **15000 / 0.739147** | `3748d09e8923a389efac87efa5750c024afde90bb69c5ff52caa45b1252df09e` | [v2_esmc_immune_v6_15000_20260927](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/v2_esmc_immune_v6_15000_20260927) | `t-20260923185956-wsvpq` |
| no-ESMC，`residue_cond_mode=token`，source_balanced=False | 110000 / 0.712413（已评完对照） | `313cc22a3e0fa16b5cac0d817f24042f84e7f7a6c193f08452ec90b128aec3d1` | [v2_noesmc_immune_v6_110000_20260927](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/v2_noesmc_immune_v6_110000_20260927) | `t-20260924185250-w22zf` |
| no-ESMC，同上，新最佳 val | **150000 / 0.709956** | `a6ce5c2c22b2e938f4573eb552b6a8ce1c2f42a3647a345adc98e6a5a052b048` | [v2_noesmc_immune_v6_150000_20260928](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/v2_noesmc_immune_v6_150000_20260928) | `t-20260924185250-w22zf`（未动） |

2026-09-27 的 32 个作业（ESMC 15000 + no-ESMC 110000）均为 `queue012`、1×`ml.pni2.3xlarge`、`Preemptible: true`，已 Success。2026-09-28 另提 **no-ESMC 150000 × 16**，同一队列/规格，输出目录带 `150000_20260928`，不覆盖 110000。YAML：`eval_jobs/eval_{ab,tcr}_v2_noesmc_v6_150000_20260928_*.yml`。索引：[20260928 submissions](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v2_immune_v6_eval_20260928/submissions.json)。看护：`bash scripts/monitor_v2_eval_spot_start.sh`（默认 20260928）；人工 cancel 前 `touch .../v2_immune_v6_eval_20260928/_monitor/STOP` 或 `STOP.<TaskName|task_id>`。150000 pairing **必须新生成**，不复用 110000 的 4000 CSV。Specificity / m396 仍为 `--input-policy fixed-v2-fv`；T4 带 `--restart-incomplete`。

| 任务 | ESMC 15000（已完成） | no-ESMC 110000（已完成） | no-ESMC 150000（已完成） | 回填产物 |
|---|---|---|---|
| AB CDR Kong | `t-20260927050038-wlskk` | `t-20260927050144-p5glc` | `t-20260928002311-z24hp` | `output/downstream_generation/ab_v2_*/cdr-kong/*.log` 的 `Average AAR all folds` |
| AB CDR SAb23 | `t-20260927050048-z6m9b` | `t-20260927050148-6x8f6` | `t-20260928003156-sgfgf` | `.../cdr-sab23/*.log` |
| AB pairing p0/CFG0/16 | `t-20260927141849-dbpr6` | `t-20260927155628-gzcrr` | `t-20260928002317-tnrw8` | `.../pair-p0-cfg0/holdout500_metrics.json` |
| AB Specificity 100ep | `t-20260927134652-2kbcf` | `t-20260927134225-jnds7` | `t-20260928002320-p2gsm` | `.../specificity-fv-v1/metrics.json`（`--input-policy fixed-v2-fv`） |
| AB GDPa1 | `t-20260927050100-dkvvm` | `t-20260927050159-c4n59` | `t-20260928002324-jxtb8` | `.../gdp_a1/metrics.json` 的 `legacy_reference.cv_mean_spearman` |
| AB m396 | `t-20260927143121-2z2w2` | `t-20260927134228-s9sl9` | `t-20260928002327-sfqsw` | `.../m396-fv-v1/metrics.json`（`--input-policy fixed-v2-fv`） |
| T1 cdr3b | `t-20260927142512-4jrq9` | `t-20260927141244-2xbzk` | `t-20260928003810-8btlp` | `downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v2_*_cdr3b/cdr3b/AS/summary.json` |
| T1 others β | `t-20260927135321-vkzdk` | `t-20260927134236-5slj2` | `t-20260928002334-kt5p2` | `.../ours_tcr_v2_*_others_cdr3b/others_cdr3b/AS/summary.json` |
| T1 CDR3αβ | `t-20260927134209-mnqzc` | `t-20260927134239-cp4sx` | `t-20260928002338-qsxx9` | `.../ours_tcr_v2_*_cdr3ab/cdr3ab/AS/summary.json` |
| T1 LongAB | `t-20260927134213-fcw98` | `t-20260927134242-m28dv` | `t-20260928002341-sntvj` | `.../ours_tcr_v2_*_others_longab/others_longab/AS/summary.json` |
| T2A | `t-20260927050122-6h9qk` | `t-20260927050224-j78lg` | `t-20260928002345-jtmjf` | `.../tcr_clustering/ours_tcr_v2_*_runtime_v2/metrics.json` |
| T2B | `t-20260927050125-m9624` | `t-20260927050228-h96g7` | `t-20260928002349-nmdvz` | `.../tcr_clustering_embed/ours_tcr_v2_*_runtime_v2/metrics.json` |
| T3-broad | `t-20260927050129-2ptmp` | `t-20260927050232-tg6xz` | `t-20260928002352-6qqw7` | `.../tcr_representation/ours_tcr_v2_*_runtime_v2/fewshot.json` |
| T3-deep | `t-20260927050133-x98mw` | `t-20260927050237-pzdhc` | `t-20260928002356-sj8p9` | `.../tcr_representation_paper6/ours_tcr_v2_*_runtime_v2/fewshot.json` |
| T4 benchmark14 | `t-20260927050137-qtv4x` | `t-20260927050241-nhfms` | `t-20260928002359-8v56z` | `.../tcr_generation_bench/setting_B/ours_tcr_v2_*_benchmark14_pmhc_beta_v2/metrics.json` |
| T4 held20 | `t-20260927152010-n854g` | `t-20260927050246-6sp5x` | `t-20260928002403-9nfd5` | `.../ours_tcr_v2_*_held20_pmhc_beta_v2/metrics.json` |

回填：`bash scripts/volc-no-proxy.sh ml_task get -i <task_id> -o json` 确认 `Success` 后，只从上述 JSON/log 抄真实数字，禁止编造。闲时可能被抢占；失败则看对应 `run.log` / `run_manifest.json`，不要改训练作业。

读表约定：`[P]` 论文报告值；`[A]` 官方预测重评分；`[R]` 官方代码/权重本地复跑；`[L]` 本地实现；`[C]` 控制。Ours 均为本地评测。论文值优先，本地结果不冒称论文值；“未获取”不等于 0。审计通过不自动证明预训练无重叠、与论文完全同协议或实验功能有效。

已完成对照快照：[v5 49000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913)。历史非 pairing 快照：[v5 92000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_92000_llada_20260915)（SHA256 `2f003a5b42a15498a7d6f2fc51414cad717f96a9cb405dca2388fcef0de3039c`）。历史矩阵快照：[v5 140000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_140000_llada_20260917)（SHA256 `a482afa425ef4f058d88e38d316aca096341af471b504efe948434d3ebc2dfe2`）。TCR 29k 仅作已存在的 others 历史对照，其他任务不借旧结果补空。 当前快照：[v5 159000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_159000_full_20260917)。

这里只保留 AB/TCR 范围内的结果与必要限制。旧排队状态、重复表、失效协议、历史故障分析及已冻结任务移入 [整理前全文存档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_ARCHIVE_20260915.md)；原始预测、日志、权重和审计均未删除。任务定义/操作手册见 [任务目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/README.md)。保留 §0.x 节号以兼容其他文档的引用；展示顺序已改为 AB → TCR。

## 一、AB 结果

<a id="ab-cdr"></a>

### §0.5 AB CDR infilling

**版本区分：历史 V7 f011 50082 与当前使用 V5 数据的 native object302 模型是不同训练运行，不是同一模型的连续 checkpoint。** 历史 f011 的“两轮”指训练两轮，解码为 argmax、**iter=1**；当前 75.6M 模型的 45000 / 138000 / 269000 使用 AirGen 对齐评测，Kong **iter=4**、SAb23H2 **iter=2**。训练数据、模型设置、输入布局和评测实现均有变化，不能把跨版本差异只归因于迭代数。历史 v5 159000 为 **iter=2**。指标为 AAR %。Kong 为 10 折宏平均，SAb23H2 为 60 条抗体。49000/92000 行仍是 **iter=1**，与 140000/159000 不是同一解码协议，不能直接当步数消融。论文模型的训练方式、预训练重叠和解码设置并不完全一致，不据此宣布“干净 zero-shot 超过 baseline”。协议：[AB CDR](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md)；论文值：[Ophiuchus 论文 Table 1–5 / Figure 4](/root/oph_paper/oph.pdf)。

#### SAbDab Kong：H1 / H2 / H3

| 方法 / checkpoint | H1 ↑ | H2 ↑ | H3 ↑ | 来源 / 设置 |
|---|---|---|---|---|
| AntiBERTy | 76.70 | 71.10 | 42.70 | [P] |
| AbLang2 | 76.30 | 70.60 | 42.70 | [P] |
| Ophiuchus-Ab | 75.50 | 70.18 | 43.55 | [P] |
| RADD | 67.64 | 58.87 | 37.71 | [P] fine-tuned |
| ADesigner | 64.34 | 55.52 | 37.37 | [P] fine-tuned |
| dyMEAN | 63.52 | 55.41 | 37.19 | [P] fine-tuned |
| MEAN | 58.29 | 47.15 | 36.38 | [P] fine-tuned |
| AbBFN | 70.30 | 64.90 | 31.50 | [P] |
| Ophiuchus-Ab 官方 checkpoint | 75.69 | 69.96 | 43.70 | [R] 历史固定 iter=1 |
| Ours v5 49000 | 75.9802 | 69.5445 | 42.9223 | 历史对照，固定 iter=1 |
| Ours v5 92000 | 76.2245 | 69.7462 | 43.2944 | 历史对照，固定 iter=1 |
| Ours v5 140000 | 76.9329 | 71.1631 | 42.6172 | 历史，固定 argmax、iter=2 |
| Ours v5 159000 | 76.8701 | 71.5733 | 42.8128 | 历史，固定 argmax、iter=2 |
| Ours V6 ESMC 15000 | 74.4120 | 68.3090 | 40.3803 | Success `t-20260927050038-wlskk`；`Average AAR all folds` |
| Ours V6 no-ESMC 110000 | 75.5013 | 69.0558 | 42.2625 | Success `t-20260927050144-p5glc`；`Average AAR all folds` |
| Ours V6 no-ESMC 150000 | 75.8113 | 69.2353 | 42.8143 | Success `t-20260928002311-z24hp`；`Average AAR all folds` |
| 历史 Ours V7 no-ESMC f011 50082 | 74.5438 | 68.6565 | 42.4700 | 训练两轮（非解码两轮）；历史原生 A4，argmax / iter=1；[逐项产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_cdr/report.json) |
| 当前 V5 数据 native 75.6M 45000 | 74.0812 | 68.1025 | 40.7544 | AirGen 对齐，argmax / iter=4；[metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_airgen_aligned_20261008/kong/metrics.json) |
| 当前 V5 数据 native 75.6M 138000 | 75.5110 | 69.7715 | 40.6462 | AirGen 对齐，argmax / iter=4；[metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009/cdr-kong/kong/metrics.json) |
| 当前 V5 数据 native 75.6M 269000 | 75.5992 | 70.0473 | 40.7592 | AirGen 对齐，argmax / iter=4；[metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/cdr-kong/kong/metrics.json) |

140000 折间 SD（ddof=0）：H1 4.2600、H2 4.0469、H3 2.6097。92000 折间 SD：H1 4.2175、H2 4.0271、H3 2.2569。来源：[159000 三份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917/cdr-kong)、[140000 三份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/cdr-kong)、[92000 三份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/cdr-kong)；49000：[三份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913/cdr-kong)。表值为 `Average AAR all folds`。没有保存逐序列预测，因此是日志/参数核对，不称独立重算全部 AAR。YAML 写明 `--sampling-strategy argmax --max-iter 2`。

#### SAb23H2：六个 CDR

| 方法 / checkpoint | H1 ↑ | H2 ↑ | H3 ↑ | L1 ↑ | L2 ↑ | L3 ↑ |
|---|---|---|---|---|---|---|
| Ophiuchus-Ab [P] | 74.8 | 68.6 | 36.8 | 81.1 | 80.1 | 73.7 |
| IgGM (IgFold) [P] | 74.0 | 64.4 | 36.0 | 75.0 | 74.3 | 63.5 |
| IgGM (AF3) [P] | 73.9 | 63.9 | 33.0 | 73.7 | 73.5 | 60.2 |
| dyMEAN [P] | 74.2 | 62.7 | 29.4 | 63.3 | 63.4 | 57.0 |
| DiffAb (AF3) [P] | 63.7 | 39.4 | 22.6 | 60.8 | 59.9 | 42.4 |
| Ophiuchus-Ab [R]，历史 iter=1 | 73.81 | 68.32 | 35.40 | 80.88 | 80.68 | 72.36 |
| Ours v5 49000，iter=1 | 75.0000 | 67.2718 | 34.4950 | 80.2964 | 78.9304 | 70.9363 |
| Ours v5 92000，iter=1 | 74.2857 | 67.5516 | 34.5570 | 80.7383 | 79.6447 | 70.6512 |
| Ours v5 140000，iter=2 | 75.2381 | 68.9266 | 34.7017 | 80.7319 | 79.8828 | 72.3347 |
| Ours v5 159000，iter=2 | 74.7619 | 68.8710 | 35.0377 | 81.0982 | 79.6447 | 73.2870 |
| Ours V6 ESMC 15000 | 75.0000 | 67.9901 | 34.2420 | 78.3644 | 78.4542 | 70.9139 |
| Ours V6 no-ESMC 110000 | 74.2857 | 66.8452 | 33.4316 | 79.6395 | 78.9304 | 70.8881 |
| Ours V6 no-ESMC 150000 | 74.7619 | 68.3313 | 33.8508 | 80.0189 | 79.1685 | 72.6058 |
| 历史 Ours V7 no-ESMC f011 50082，iter=1 | 75.0000 | 65.0655 | 34.1097 | 79.5654 | 78.6923 | 72.8595 |

来源：[159000 六份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917/cdr-sab23)、[140000 六份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/cdr-sab23)、[92000 六份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/cdr-sab23)、[49000 六份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913/cdr-sab23)。140000 主行是 iter=2，不是 49000/92000 的 iter=1，也不是下方 92000 多步对照里逐 CDR 挑最优。未用每个 CDR 在测试集上挑选的 best-iteration 数值作 Ours 主行；历史多步 sweep 见 [归档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_ARCHIVE_20260915.md)。

<details>
<summary>SAb23H2：ESMC 反馈修复后的 iteration 对照（2026-09-16，本地已完成）</summary>

同一 v5 92000 留存权重，完整 60 条抗体，每次掩一个 CDR；argmax、batch=1、CFG=0、temperature=1、seed=42。仅改变 `max_iter`，共 24 组、1,440 条预测；短 CDR 可提前填完，实际 forward 次数不一定等于上限。指标为逐序列 AAR 的均值，单位 %；下列论文行仅作参考，论文解码步数未明确披露，不等于相同采样协议。

| 方法 / checkpoint | H1 ↑ | H2 ↑ | H3 ↑ | L1 ↑ | L2 ↑ | L3 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Ophiuchus-Ab [P] | 74.8 | 68.6 | 36.8 | 81.1 | 80.1 | 73.7 |
| Ours v5 92000，iter=1 | 74.2857 | 67.5516 | 34.5570 | 80.7383 | 79.6447 | 70.6512 |
| Ours v5 92000，iter=2 | 74.2857 | 66.9405 | 34.8933 | 80.6666 | 79.6447 | 70.9038 |
| Ours v5 92000，iter=4 | 74.5238 | 66.9405 | 32.8751 | 80.4677 | 79.6447 | 70.6089 |
| Ours v5 92000，iter=8 | 74.5238 | 67.2183 | 32.3089 | 80.3289 | 79.8828 | 70.3961 |

结论：本组没有随步数增加而稳定提升。H3 的 iter=2 比 iter=1 高 0.3364 个百分点，但 iter=8 低 2.2481 个百分点；不能用逐 CDR 最优值拼成新主行，原固定 iter=1 主结果保持不变。本轮没有运行 Kong 多步对照，也没有同权重“修复前／后”多步配对实验，不能量化修复本身带来的性能增益。

验收：1,440 条预测均完整且为标准氨基酸，逐条匹配数和 24 格 AAR 独立重算通过；六模式 iter=1 与旧日志完全一致。每模式首条样本、每种步数检查真实 ESMC 输入（共 24 组轨迹），已接受生成残基可见、未接受位置 MASK；另六次隐藏参考替换检查输出不变。这不是对全部 1,440 条进行逐层抓取，也不等于预训练去污染验收。

来源：[完整 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_cdr_iter_20260916/summary.json)、[逐条预测目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_cdr_iter_20260916)、[ESMC 输入审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_cdr_iter_20260916/feedback_audit.json)、[权重／数据／源码身份](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_cdr_iter_20260916/manifest.json)；baseline 仍来自 [论文 Table 1](/root/oph_paper/oph.pdf)，协议见 [AB CDR §4.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md)。

</details>

#### 历史 V7 两轮固定点与既有 V5 对照（2026-10-06T22:28:30.597931+00:00）

[完整收数报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_cdr/report.json)（SHA 8124f22b0a7b5e63470f1db2f6e7347d1c6bb297e315cbf09ff900bf9e16e70c）绑定九份 metrics、manifest、summary、输入审计及历史来源；[生成完成验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/generation_pool_v1/COMPLETE.json)确认九个实际评测和 pool 退出均成功。Kong 每个 CDR 3127 个区域样本、10 折；SAb23 每个 CDR 60 个区域样本、1 折，合计9741个 CDR 区域案例，不是9741个不同抗体。原评分器已对保存预测重算，汇总脚本未再次推理或重评分。

| 数据集 / CDR | V7 AAR % | V5 92k iter1 | 差值 pp | V5 159k iter2 | 差值 pp |
|---|---:|---:|---:|---:|---:|
| kong / CDRH1 | 74.5438 | 76.2245 | -1.6807 | 76.8701 | -2.3263 |
| kong / CDRH2 | 68.6565 | 69.7462 | -1.0897 | 71.5733 | -2.9168 |
| kong / CDRH3 | 42.4700 | 43.2944 | -0.8244 | 42.8128 | -0.3428 |
| sab23 / CDRH1 | 75.0000 | 74.2857 | +0.7143 | 74.7619 | +0.2381 |
| sab23 / CDRH2 | 65.0655 | 67.5516 | -2.4861 | 68.8710 | -3.8056 |
| sab23 / CDRH3 | 34.1097 | 34.5570 | -0.4472 | 35.0377 | -0.9280 |
| sab23 / CDRL1 | 79.5654 | 80.7383 | -1.1729 | 81.0982 | -1.5328 |
| sab23 / CDRL2 | 78.6923 | 79.6447 | -0.9524 | 79.6447 | -0.9524 |
| sab23 / CDRL3 | 72.8595 | 70.6512 | +2.2083 | 73.2870 | -0.4275 |

Kong V7折间SD（ddof=0）：H1 4.7546、H2 4.1394、H3 2.2627。SAb23只有一折，fold SD=0不代表没有样本不确定性。相对V5 92k，同为单步argmax时2项上升、7项下降；相对V5 159k iter2则1项上升、8项下降。当前不支持V7全面优于V5的结论。

本轮是已知长度CDR补全，保留可见框架/另一条链/结束位置；非自主长度生成或抗原条件设计。V7为75,601,176参数、从零训练两轮；与V5的数据、规模、训练预算和输入布局不同，单步协议相同仍不是受控架构消融。未按测试集选择checkpoint或迭代数；双证书范围仍限定observed-pair exact test，不宣称全部query/few-shot或相似度去污染。轻链配对和T4未包含在此九项中。

<a id="ab-pairing"></a>

### §0.6 AB Light-chain pairing

500 条重链，每条生成8候选；IM为全部4,000候选均值。159000 固定 **p0 / CFG=0 / iter=16**，140000 为 **p0 / CFG=0 / iter=124**；均为reference长度、gumbel_argmax、seed42。没有159000/140000的p3或CFG网格结果。49000主表行仍为修复后p3。前缀、步数、权重不同需明确区分。

| 方法 / checkpoint | IM ↑ | 有效率 ↑ | 轻链类型匹配 ↑ | Diversity |
|---|---|---|---|---|
| Ophiuchus-Ab，论文 [P]，有前三残基 | 0.695 | 1.000 | 0.997 | 0.335 |
| Ophiuchus-Ab，论文 [P]，无前缀 | 0.701 | 1.000 | 0.556 | 0.676 |
| Ours v5 140000 p0 / CFG=0 | 0.504681 | 1.000000 | 0.783750 | 0.389908 |
| Ours v5 159000 p0 / CFG=0 / iter=16 | 0.543686 | 1.000000 | 0.792000 | 0.433252 |
| Ours V6 ESMC 15000 p0 / CFG=0 / iter=16 / auto | 0.497320 | 0.990750 | 0.527000 | 0.662486 |
| Ours V6 no-ESMC 110000 p0 / CFG=0 / iter=16 / auto | 0.587728 | 1.000000 | 0.554000 | 0.629925 |
| Ours V6 no-ESMC 150000 p0 / CFG=0 / iter=16 / auto | 0.615300 | 0.999500 | 0.623500 | 0.494591 |
| Ours V7 f011 50082（两轮）p0 / CFG=0 / iter=16 / reference | 0.591934 | 1.000000 | 0.802000 | 0.400775 |


| Ours v5 49000 p3（ESMC 反馈修复后） | 0.442264 | 1.000000 | 0.996500 | 0.203651 |

**V7 两轮实测（2026-10-06T23:35:21.048373+00:00）：** 500×8候选生成和评分均实际exit0，全部4000候选及逐项评分保留、无空候选。相对V5 159000固定p0行，IM增加0.048247，类型匹配增加1.00个百分点，有效率不变，多样性下降0.032477；不能称全面提升。两次使用相同基准字节、reference长度、16步、CFG0、temperature1，但V7为原生A4、逐候选seed、batch1，训练数据/模型规模/训练预算也不同，不是受控架构消融。同一reference的IM相差约0.000008622，未重跑V5。IM与序列有效性均不等于实验结合能力。原始指标和实际退出证据见[完整报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_pairing/report.json)、[metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_pairing_scores_local_gpu_20261006/metrics.json)，协议见[轻链 §4.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。

**2026-09-16 修复后重评已完成并验收（49000 p3）：** ESMC 每轮读取模型已接受的生成残基，pending 保持 MASK，不读取真实后缀；固定同一 49000 权重、数据、reference 长度、iter124、gumbel_argmax、CFG0、seed42。有前缀的 IM 从修复前 0.476541 降至 0.442264，不能把输入错误修复等同于性能必然提升；有效率与轻链类型匹配改善，多样性下降。两组 p0/p3 均有完整 4,000 条生成与评分。验收边界见 [修正协议 §7 k](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。

140000 p0 使用同一修复后 encoder 状态（`committed_generated_residues_pending_mask_no_reference_targets`）。与 49000 p3 的前缀条件不同，也不同于论文原生生成长度。同权重 49000 无前缀 124 步诊断 IM=0.442087 见下方折叠表，不能与 140000 混称为步数扫描。来源：[Ophiuchus-Ab 论文 Table 3](/root/oph_paper/oph.pdf)、[159000 p0 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917/pair-p0-cfg0/holdout500_metrics.json)、[140000 p0 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/pair-p0-cfg0/holdout500_metrics.json)、[修复后 49000 p3 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_pairing_encoder_sync_20260916/pair-p3-cfg0/holdout500_metrics.json)、[修复前历史 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913/pair-p3-cfg0/holdout500_metrics.json)；详细协议见 [AB pairing](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。

<details>
<summary>无前缀 pairing：49000 步数诊断完整结果（2026-09-17 验收）</summary>

六档均为平台Success，24,000条新增候选全部验收；每档同一500条重链×8候选，固定v5 49000、ESMC反馈修复后、无真实轻链初始残基、reference长度、CFG0、gumbel_argmax、temperature1、seed42。124步为此前已验收的同协议独立参照。IM及各率均以全部候选为分母，不删除无效候选、不选best-of-8。论文行来自Table3无前三残基部分，长度为原生生成，与Ours已知长度不同；以下是步数诊断，不替换上方p3主表，也不与140000矩阵混用。

| 方法 / max_iter | IM ↑ | 有效率 ↑ | κ/λ类型匹配 ↑ | Diversity | worker耗时（分钟） |
|---|---:|---:|---:|---:|---:|
| Ophiuchus-Ab [P]，无前缀 | 0.701 | 1.000 | 0.556 | 0.676 | 未报告 |
| Ours v5 49000，8 | 0.485744 | 0.997250 | 0.805250 | 0.553055 | 30.06 |
| Ours v5 49000，16 | 0.496521 | 0.999500 | 0.780250 | 0.497163 | 39.67 |
| Ours v5 49000，32 | 0.479923 | 0.999750 | 0.758750 | 0.480879 | 67.03 |
| Ours v5 49000，64 | 0.462169 | 1.000000 | 0.775750 | 0.458670 | 93.52 |
| Ours v5 49000，96 | 0.447975 | 1.000000 | 0.759250 | 0.457356 | 130.17 |
| Ours v5 49000，124（原参照） | 0.442087 | 1.000000 | 0.766750 | 0.450097 | 未记录同口径worker耗时 |
| Ours v5 49000，128 | 0.438446 | 1.000000 | 0.762750 | 0.452593 | 164.96 |

本轮最高IM出现在16步：比124步高0.054434（相对+12.31%），但仍低于论文参考0.204479。16→32→64→96→128的IM逐档下降，不能假设增加步数必然改善；有效率已接近100%，并未同步解释IM下降。worker耗时含门禁、生成、评分及验收，不含平台排队，不能直接称纯推理吞吐。固定seed不意味着不同步数的逐候选随机数一致；尚无多seed置信区间，也没有在独立验证集选定16步，不能将本次最高分称为普遍最优或直接升级主结果。

2026-09-17只读复核：六组实际反馈gate通过、每组4,000条预测与全部annotation齐全，样本身份／顺序和124参照一致；重新核对指标文件／CSV／manifest哈希、匹配率及有效率分母、CSV审计均通过，已固定源码／数据／参考产物没有漂移。IM与Diversity来自已完成的相同评分器，本次没有重跑评分模型。来源：[完整汇总（含Better、V/J及家族匹配）](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_pairing_iters_p0_20260916/summary.json)、[实验身份与逐条产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_pairing_iters_p0_20260916)、[论文Table3](/root/oph_paper/oph.pdf)；协议见[AB pairing §4.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。

</details>

<a id="ab-probes"></a>

### §0.6a AB 附加表征任务

Ours 编码原生完整 H/L，冻结 ESMC+LLaDA，使用 post-LLaDA 有效残基 global mean；不同任务按既定头/回归流程训练。协议与输入/选模差异见 [AB Native Probes](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md)。

#### Specificity：HD-Flu-CoV 分类

4,398 条样本、五折；Ours 固定取末轮，不按留出折曲线挑最好 epoch。下面是各折均值，不是跨 epoch 平均。

| 方法 / checkpoint | 头训练预算 | Accuracy ↑ | F1 ↑ | MCC ↑ |
|---|---|---|---|---|
| Ophiuchus-Ab [P] | 论文，最终选头配置未披露 | 0.6796 | 0.6790 | 0.5203 |
| Ophiuchus-Ab [R] | 历史独立 100 轮 | 0.677804 | 0.676818 | 0.517054 |
| Ours v5 49000 | 100 | 0.618000 | 0.616288 | 0.428139 |
| Ours v5 49000 | 200 | 0.627325 | 0.625782 | 0.441693 |
| Ours v5 92000 | 200 | 0.643468 | 0.641730 | 0.466119 |
| Ours v5 140000 | 100 | 0.644152 | 0.642641 | 0.466953 |
| Ours v5 159000 | 100 | 0.641652 | 0.640057 | 0.463362 |
| Ours V6 ESMC 15000 | 100 | 0.597995 | 0.597063 | 0.398094 |
| Ours V6 no-ESMC 110000 | 100 | 0.616867 | 0.616058 | 0.425872 |
| Ours V6 no-ESMC 150000 | 100 | 0.614138 | 0.612447 | 0.422339 |

140000/159000 均是标准 100-epoch native、固定末轮，不是 92000 的 200-epoch 预算，也不能当作 200 轮曲线上的中间点。Ours F1 为 macro；论文 Table 5 未标明 averaging。相同头结构不等于相同输入宽度/参数量；历史 baseline 未按本轮全部保存头门禁重跑。来源：[159000 / 100轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917/specificity/metrics.json)、[140000 / 100轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/specificity/metrics.json)、[49000 / 100轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913/specificity/metrics.json)、[49000 / 200轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_specificity_ep200_20260915/metrics.json)、[92000 / 200轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_ep200_20260915/metrics.json)（均保存 fold SD / OOF），以及 [Ophiuchus 论文 Table 1–5 / Figure 4](/root/oph_paper/oph.pdf)。

<details>
<summary>训练预算诊断：完整末轮 loss / 指标对照</summary>

92000/300、400、500轮是独立预算实验；以下新增三行只在平台Success且完整产物验收通过后回填，不用中间轮次。预算与fixed-last规则见[AB Native Probes §4.7](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md)；[实时状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_budget_audit_20260915/status.json)与[逐项验收/最终比较目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_budget_audit_20260915)提供来源。

<!-- ab92000-budget-results:start -->
| 本地诊断运行 | 末轮训练loss | 末轮留出折loss | Accuracy | macro-F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Ophiuchus [R] 50轮 | 0.638430 | 0.774160 | 0.665071 | 0.664036 | 0.498153 |
| Ophiuchus [R] 100轮 | 0.540101 | 0.777501 | 0.677804 | 0.676818 | 0.517054 |
| Ophiuchus [R] 200轮 | 0.407478 | 0.841788 | 0.670304 | 0.669584 | 0.505715 |
| Ophiuchus [R] 300轮 | 0.317621 | 0.945798 | 0.667571 | 0.666570 | 0.501624 |
| Ours v5 step49000 100轮 | 0.790130 | 未记录 | 0.618000 | 0.616288 | 0.428139 |
| Ours v5 step49000 200轮 | 0.742728 | 0.838495 | 0.627325 | 0.625782 | 0.441693 |
| **Ours v5 step92000 200轮** | **0.707176** | **0.821146** | **0.643468** | **0.641730** | **0.466119** |
| Ours v5 step92000 300轮 | 0.670705 | 0.823670 | 0.648926 | 0.647276 | 0.474183 |
| Ours v5 step92000 400轮 | 0.641995 | 0.830413 | 0.648926 | 0.647463 | 0.473956 |
| Ours v5 step92000 500轮 | 0.618489 | 0.839541 | 0.647563 | 0.646208 | 0.471843 |
| Ours v5 step92000 100轮 | 0.759260 | 0.829224 | 0.632781 | 0.631154 | 0.450079 |
<!-- ab92000-budget-results:end -->

不同训练预算是独立运行，会改变 scheduler，不能把独立 100 轮结果当作 200 轮运行的中间点。留出曲线仅作诊断，正式成绩固定末轮。数据/曲线与历史 baseline 路径见 [学习曲线协议与验收 §4.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md#45-specificity-200-epoch-学习曲线诊断2026-09-15)；本节不是采样消融。

</details>

#### GDPa1：属性回归，参考流程 CV

**2026-09-27 debug**：原表分数从特征缓存全部精确复算；数据/折号/标签一致，原 checkpoint 特征抽查通过。未发现解释大幅退化的评测代码错误。V6 与 V5 的采样配方、训练进度、EOS/画布和 encoder 条件不同，不能视为仅数据版本变化。35k 同协议诊断有升有降，未替换主表；见 [完整诊断](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/diagnostics/gdpa1_v6_20260927/REPORT.md)。

| 属性 | n | Ophiuchus-Ab论文 [P] | Ours v5 49000 参考CV | Ours v5 92000 参考CV | Ours v5 140000 参考CV | Ours v5 159000 参考CV | Ours V6 ESMC 15000 | Ours V6 no-ESMC 110000 | Ours V6 no-ESMC 150000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AC-SINS pH7.4 | 242 | 0.511 | 0.490733 | 0.490949 | 0.512870 | 0.526944 | 0.325685 | 0.390761 | 0.401411 |
| HIC | 242 | 0.550 | 0.389040 | 0.472238 | 0.527719 | 0.517856 | 0.363292 | 0.183533 | 0.183557 |
| PR_CHO | 197 | 0.460 | 0.361299 | 0.476785 | 0.360072 | 0.398506 | 0.363871 | 0.419923 | 0.369252 |
| Titer | 239 | 0.359 | 0.267836 | 0.301977 | 0.349780 | 0.357980 | 0.268263 | 0.303170 | 0.291550 |
| Tm2（附加） | 193 | 未报告 | 0.140118 | 0.070526 | 0.131592 | 0.104690 | 0.091427 | 0.095698 | 0.175316 |

V6 GDPa1 已按 `legacy_reference.cv_mean_spearman` 回填：ESMC 15000 Success `t-20260927050100-dkvvm`，no-ESMC 110000 Success `t-20260927050159-c4n59`。

来源：`legacy_reference.cv_mean_spearman`；[159000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917/gdp_a1/metrics.json)、[140000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/gdp_a1/metrics.json)、[49000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913/gdp_a1/metrics.json)、[92000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/gdp_a1/metrics.json)；论文 Table 4。参考流程存在全标签变换与同 CV 选参/报分局限，不当作独立外层测试；不与 raw-label pooled OOF 混用。按用户决定只展示参考流程，历史 nested 诊断已移出主文档，原产物保留。

#### m396：突变体计算能量预测

五个计算能量目标的平均 Spearman，不是实测 Kd。论文 Figure 4 没有可直接转录的六档精确数值，保留缺项，不由本地复跑代填。

| 方法 / checkpoint | 0.5% | 1% | 2% | 5% | 10% | 20% |
|---|---|---|---|---|---|---|
| Ophiuchus-Ab [P] | 未获取 | 未获取 | 未获取 | 未获取 | 未获取 | 未获取 |
| Ophiuchus-Ab [R] | 0.930065 | 0.944155 | 0.952616 | 0.961872 | 0.967179 | 0.970591 |
| Ours v5 49000 | 0.899690 | 0.921741 | 0.934626 | 0.945228 | 0.950440 | 0.954311 |
| Ours v5 92000 | 0.904707 | 0.925033 | 0.934510 | 0.946026 | 0.951057 | 0.953988 |
| Ours v5 140000 | 0.909132 | 0.927589 | 0.937812 | 0.948321 | 0.952641 | 0.955300 |
| Ours v5 159000 | 0.910117 | 0.928021 | 0.937246 | 0.947397 | 0.952155 | 0.954771 |
| Ours V6 ESMC 15000 | 0.899134 | 0.918575 | 0.930190 | 0.942448 | 0.947482 | 0.950682 |
| Ours V6 no-ESMC 110000 | 0.906079 | 0.924764 | 0.936695 | 0.948012 | 0.953353 | 0.956526 |
| Ours V6 no-ESMC 150000 | 0.898766 | 0.924830 | 0.938612 | 0.948263 | 0.953103 | 0.956473 |

有效突变体 86,929；独立完整 H/L 序列 82,311。按 Ophiuchus-Ab 原始代码采用按行随机划分，训练比例按行计、划分 seed=2023，其余行用于测试；各档存在同序列跨 split。按用户决定只展示上述原始划分，附加分组产物不列主表。baseline 特征窗口/pooling 与 Ours 不同。来源：[159000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917/m396/metrics.json)、[140000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/m396/metrics.json)、[92000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/m396/metrics.json)、[49000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913/m396/metrics.json)、[历史 Ophiuchus metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/affinity_m396_metrics.json)、[原始划分代码](/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/downstream/in_silico/finetune_in_silico.py:38)。

## 二、TCR 结果

<a id="uptcr-binding-status"></a>

**UpTCR binding：原发布三套划分本地评测已完成（2026-09-30）。** 按 [任务约定](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md#uptcr-binding) 原样使用发布数据，baseline只需论文公开值。已核验的UpTCR正文AUROC/AUPR [P]：majority 0.905/0.749、few-shot 0.800/0.561、未见抗原50% 0.642/0.296、30% 0.594/0.252，来源 [论文Fig.2](https://www.nature.com/articles/s41467-026-78075-x)。这些是文献参照；Ours 按实际发布划分单列如下。发布majority/zero-shot测试行数与论文Source Data不同，发布包仅一个zero-shot划分，不能将单一 zero-shot Ours 结果称为已复现两套论文阈值设定。本地输入为 FVα/FVβ + peptide、无 HLA，骨干冻结；原 valid 选模、原 test 评分。与论文输入/测试成员/指标实现差异未消除，不计算胜负或跨协议差值。

| Ours：no-ESMC 150000 / 原发布划分 | Test 行数 | AUROC | AUPRC（average precision） | 证据 |
|---|---:|---:|---:|---|
| majority | 25993 | 0.847615 | 0.648445 | [metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/uptcr_noesmc150000_original_20260930/majority/metrics.json) |
| few-shot | 4044 | 0.699060 | 0.356069 | [metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/uptcr_noesmc150000_original_20260930/few-shot/metrics.json) |
| zero-shot（发布包单一划分） | 8958 | 0.608535 | 0.219458 | [metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/uptcr_noesmc150000_original_20260930/zero-shot/metrics.json) |

预训练 unseen 仍未得到 checkpoint→训练 manifest→cohort 完整哈希链证明；不能从此处 zero-shot 名称推导 clean。见 [独立报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_integrity_20260930/final/results/integrity_report.json)。

既有完整 TCR Ours 矩阵为 **v5 159000**；新增V7本轮仅包含T4生成。V5共10项：T1 四轨、T2 两项、T3 两项、T4 两数据集。49000/92000/140000 同行保留对照；49000 完整性仍以 [最终 comparison](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/comparison_v1/final_49000_vs_baselines_and_29000.json) 为准。159000/140000/92000 数字直接取各任务 summary/metrics/fewshot JSON。T1 为保存预测的五折均值，T2/T3 为保存产物聚合，T4 为完整保存序列评分；本轮没有对已写入数字再做一次独立前向重算。

<a id="tcr-binding"></a>

### §0.1 T1 Binding：四种输入分别比较

NM2025 官方 AS 五折；Ours 冻结基础模型、每折训练 MLP，不是全参数微调，也不是关系 token 生成评分。**recognition 位置使用 mask，不给定 `<binding>` 或样本标签。** 原始数据读取后运行时补成 v5 格式，未知区域保留 X；仅观察到的残基参与池化。

每格为 **AUROC / AUPRC（precrec）** 的五折均值；完整折间 sample SD 在各轨 summary.json。论文 [P] 出自 Supplementary Table 7-1…7-9；最终 comparison 保留逐行转录、原表计数与来源 SHA。本地公开文件与论文部分样本数/正负比例不同，尤其影响 AUPRC，不能作为同一行集的严格排名。协议和方法训练方式：[T1 任务](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md)。

#### T1-1 大数据 cdr3b：表位＋CDR3β

| 方法 / checkpoint | Seen test | Seen independent | Unseen independent |
|---|---|---|---|
| TEPCAM [P] | 0.8008 / 0.8163 | 0.6116 / 0.5922 | 0.5053 / 0.5075 |
| epiTCR [P] | 0.8142 / 0.8345 | 0.6556 / 0.6976 | 0.5075 / 0.5106 |
| TEIM [P] | 0.7766 / 0.7932 | 0.6090 / 0.6224 | 0.5326 / 0.5218 |
| AttnTAP [P] | 0.5856 / 0.5911 | 0.5216 / 0.5189 | 0.4939 / 0.4981 |
| ERGO-AE [P] | 0.7388 / 0.7519 | 0.5636 / 0.5711 | 0.5099 / 0.5106 |
| TCRGP [P] | 0.7829 / 0.7982 | 0.6317 / 0.6742 | 未报告 |
| ATM-TCR [P] | 0.7728 / 0.7800 | 0.6028 / 0.5838 | 0.5192 / 0.5136 |
| NetTCR [P] | 0.7372 / 0.7482 | 0.5626 / 0.5563 | 0.5203 / 0.5156 |
| TCR-BERT [P] | 0.7645 / 0.7821 | 0.6208 / 0.6157 | 未报告 |
| ERGO-lstm [P] | 0.5022 / 0.5012 | 0.4987 / 0.4981 | 0.5009 / 0.5049 |
| vibtcr [P] | 0.4995 / 0.5007 | 0.5040 / 0.5056 | 0.4981 / 0.4994 |
| VitTCR [P] | 0.7092 / 0.7277 | 0.5675 / 0.5663 | 0.5126 / 0.5129 |
| TCR-H [P] | 0.6865 / 0.6904 | 0.5593 / 0.5562 | 0.5269 / 0.5267 |
| PiTE [P] | 0.6121 / 0.6256 | 0.5581 / 0.5724 | 0.5127 / 0.5065 |
| TEINet [P] | 0.5770 / 0.5887 | 0.5111 / 0.5164 | 0.5025 / 0.5057 |
| ImRex [P] | 0.5895 / 0.5894 | 0.5295 / 0.5331 | 0.5050 / 0.5041 |
| SETE [P] | 0.5457 / 0.5487 | 0.5412 / 0.5381 | 未报告 |
| DeepTCR [P] | 0.5200 / 0.5091 | 0.5280 / 0.5320 | 0.5113 / 0.5118 |
| TPBTE [P] | 0.4999 / 0.5001 | 0.5011 / 0.5001 | 0.5009 / 0.5008 |
| MCMC [P] | 0.5000 / 0.5000 | 0.5000 / 0.5000 | 0.5000 / 0.5000 |
| TITAN [P] | 0.5041 / 0.5021 | 0.5025 / 0.4983 | 0.5039 / 0.4979 |
| DLpTCR-RESNET [P] | 0.3669 / 0.4079 | 0.4668 / 0.4712 | 0.4918 / 0.4918 |
| DLpTCR-FULL [P] | 0.2535 / 0.3598 | 0.4097 / 0.4341 | 0.4916 / 0.4947 |
| DLpTCR-CNN [P] | 0.2976 / 0.3764 | 0.4394 / 0.4541 | 0.4942 / 0.4941 |
| Ours v5 49000 | 0.7450 / 0.7607 | 0.5755 / 0.5583 | 0.5068 / 0.5181 |
| Ours v5 92000 | 0.7510 / 0.7658 | 0.5822 / 0.5630 | 0.5226 / 0.5236 |
| Ours v5 140000 | 0.7528 / 0.7682 | 0.5814 / 0.5701 | 0.5394 / 0.5357 |
| Ours v5 159000 | 0.7562 / 0.7716 | 0.5909 / 0.5709 | 0.5318 / 0.5312 |
| Ours V6 ESMC 15000 | 0.7466 / 0.7618 | 0.5915 / 0.5815 | 0.5387 / 0.5318 |
| Ours V6 no-ESMC 110000 | 0.7563 / 0.7713 | 0.5966 / 0.5885 | 0.5268 / 0.5210 |
| Ours V6 no-ESMC 150000 | 0.7565 / 0.7709 | 0.6054 / 0.5956 | 0.5282 / 0.5321 |

来源：[159000 五折 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_159000_20260917_cdr3b/cdr3b/AS/summary.json)、[140000 五折 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_140000_20260917_cdr3b/cdr3b/AS/summary.json) 的 `eval_sets.*.local`、[92000 五折 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_92000_20260915_cdr3b/cdr3b/AS/summary.json)、[49000 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_49000_20260913_cdr3b/cdr3b/AS/summary.json)、[49000 逐预测审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v1/binding_cdr3b.json)。未找到 v5 29k 的同协议大数据结果；不把 others β-only 当成本项旧 checkpoint 对照。历史官方 checkpoint 复跑保留于 [归档 §0.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_ARCHIVE_20260915.md)，其中 ERGO-AE 有丢行，不能当逐行对齐；论文 seen 两列与公开复跑存在偏移。

#### T1-2 others β-only：表位＋CDR3β

| 方法 / checkpoint | Seen test | Seen independent | Unseen independent |
|---|---|---|---|
| epiTCR [P] | 0.6667 / 0.6835 | 0.5993 / 0.6315 | 0.5168 / 0.5198 |
| TEPCAM [P] | 0.6697 / 0.6611 | 0.5677 / 0.5697 | 0.5266 / 0.5225 |
| TCRGP [P] | 0.6372 / 0.6517 | 0.6023 / 0.6323 | 未报告 |
| TEIM [P] | 0.6488 / 0.6457 | 0.5977 / 0.6154 | 0.5228 / 0.5196 |
| TCR-BERT [P] | 0.6311 / 0.6388 | 0.5972 / 0.6366 | 未报告 |
| ATM-TCR [P] | 0.6398 / 0.6287 | 0.5602 / 0.5714 | 0.5277 / 0.5323 |
| ERGO-AE [P] | 0.6277 / 0.6218 | 0.4798 / 0.5068 | 0.4890 / 0.4933 |
| TCR-H [P] | 0.6244 / 0.6064 | 0.5362 / 0.5358 | 0.5292 / 0.5203 |
| VitTCR [P] | 0.5834 / 0.5817 | 0.5375 / 0.5482 | 0.5143 / 0.5131 |
| NetTCR [P] | 0.5681 / 0.5643 | 0.5021 / 0.5058 | 0.4883 / 0.4946 |
| ImRex [P] | 0.5472 / 0.5532 | 0.4940 / 0.4909 | 0.4976 / 0.5038 |
| PiTE [P] | 0.5382 / 0.5472 | 0.5394 / 0.5461 | 0.4997 / 0.5012 |
| AttnTAP [P] | 0.5450 / 0.5462 | 0.4880 / 0.5010 | 0.5015 / 0.4992 |
| SETE [P] | 0.5217 / 0.5169 | 0.5350 / 0.5482 | 未报告 |
| TEINet [P] | 0.5129 / 0.5146 | 0.4924 / 0.4941 | 0.4940 / 0.5005 |
| TITAN [P] | 0.5003 / 0.5015 | 0.4736 / 0.4878 | 0.4990 / 0.4950 |
| TPBTE [P] | 0.5001 / 0.5012 | 0.5076 / 0.5058 | 0.5009 / 0.4996 |
| DLpTCR-RESNET [P] | 0.5006 / 0.5005 | 0.4979 / 0.4984 | 0.5012 / 0.5021 |
| MCMC [P] | 0.5000 / 0.5000 | 0.5000 / 0.5000 | 0.5000 / 0.5000 |
| ERGO-lstm [P] | 0.4995 / 0.4997 | 0.5635 / 0.5513 | 0.4992 / 0.5000 |
| vibtcr [P] | 0.4870 / 0.4891 | 0.4841 / 0.4846 | 0.5012 / 0.4981 |
| DLpTCR-CNN [P] | 0.3990 / 0.4251 | 0.5012 / 0.5056 | 0.4865 / 0.4852 |
| DLpTCR-FULL [P] | 0.3846 / 0.4209 | 0.4735 / 0.4867 | 0.4991 / 0.4969 |
| DeepTCR [P] | 0.3214 / 0.3916 | 0.2778 / 0.3753 | 0.5102 / 0.5113 |
| Ours v5 29000（历史） | 0.6212 / 0.6162 | 0.5811 / 0.5920 | 0.5230 / 0.5187 |
| Ours v5 49000 | 0.6335 / 0.6345 | 0.5961 / 0.6051 | 0.5236 / 0.5149 |
| Ours v5 92000 | 0.6314 / 0.6245 | 0.5924 / 0.5976 | 0.5246 / 0.5175 |
| Ours v5 140000 | 0.6362 / 0.6303 | 0.6020 / 0.6149 | 0.5149 / 0.5151 |
| Ours v5 159000 | 0.6346 / 0.6280 | 0.5817 / 0.5989 | 0.5114 / 0.5125 |
| Ours V6 ESMC 15000 | 0.6321 / 0.6279 | 0.5988 / 0.6220 | 0.5286 / 0.5200 |
| Ours V6 no-ESMC 110000 | 0.6230 / 0.6184 | 0.5977 / 0.6088 | 0.5337 / 0.5260 |
| Ours V6 no-ESMC 150000 | 0.6357 / 0.6248 | 0.5862 / 0.5987 | 0.5320 / 0.5309 |

来源：[159000 others β-only summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_159000_20260917_others_cdr3b/others_cdr3b/AS/summary.json)、[140000 others β-only summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_140000_20260917_others_cdr3b/others_cdr3b/AS/summary.json)、[92000 others β-only summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_92000_20260915_others_cdr3b/others_cdr3b/AS/summary.json)、[49000 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_49000_20260913_others_cdr3b/others_cdr3b/AS/summary.json)。这一轨来自 others 行集，不与上表的大数据 cdr3b 合并。

#### T1-3 others CDR3αβ：表位＋CDR3α＋CDR3β

| 方法 / checkpoint | Seen test | Seen independent | Unseen independent |
|---|---|---|---|
| TCRGP-AB [P] | 0.6372 / 0.6517 | 0.6023 / 0.6323 | 未报告 |
| NetTCR-AB [P] | 0.5034 / 0.5063 | 0.4976 / 0.5021 | 0.5023 / 0.5047 |
| vibtcr-AB [P] | 0.4964 / 0.4982 | 0.4967 / 0.4920 | 0.5003 / 0.5023 |
| DeepTCR-ABVJ [P] | 0.4876 / 0.4811 | 0.5002 / 0.4944 | 0.5019 / 0.5046 |
| Ours v5 29000（历史） | 0.6590 / 0.6593 | 0.5414 / 0.5458 | 0.5193 / 0.5111 |
| Ours v5 49000 | 0.6556 / 0.6550 | 0.5717 / 0.5790 | 0.5205 / 0.5143 |
| Ours v5 92000 | 0.6755 / 0.6729 | 0.5472 / 0.5518 | 0.5085 / 0.4998 |
| Ours v5 140000 | 0.6685 / 0.6666 | 0.5472 / 0.5536 | 0.5196 / 0.5178 |
| Ours v5 159000 | 0.6765 / 0.6723 | 0.5540 / 0.5601 | 0.5189 / 0.5092 |
| Ours V6 ESMC 15000 | 0.6341 / 0.6314 | 0.6203 / 0.6203 | 0.5306 / 0.5262 |
| Ours V6 no-ESMC 110000 | 0.6469 / 0.6414 | 0.5813 / 0.5803 | 0.5289 / 0.5227 |
| Ours V6 no-ESMC 150000 | 0.6589 / 0.6530 | 0.5785 / 0.5798 | 0.5332 / 0.5254 |

DeepTCR-ABVJ 额外使用 V/J，不是相同输入对照。来源：[159000 CDR3αβ summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_159000_20260917_cdr3ab/cdr3ab/AS/summary.json)、[140000 CDR3αβ summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_140000_20260917_cdr3ab/cdr3ab/AS/summary.json)、[92000 CDR3αβ summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_92000_20260915_cdr3ab/cdr3ab/AS/summary.json)、[49000 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_49000_20260913_cdr3ab/cdr3ab/AS/summary.json)。

#### T1-4 others LongAB：表位＋LongA＋LongB

| 方法 / checkpoint | Seen test | Seen independent | Unseen independent |
|---|---|---|---|
| TCRconv-fullAB [P] | 0.6915 / 0.7609 | 0.5704 / 0.7321 | 未报告 |
| TCRconv-fullA [P] | 0.6373 / 0.7051 | 0.5342 / 0.6891 | 未报告 |
| TCRconv-fullB [P] | 0.6373 / 0.7051 | 0.5342 / 0.6891 | 未报告 |
| Ours v5 29000（历史） | 0.7419 / 0.7336 | 0.5099 / 0.5105 | 0.5071 / 0.5063 |
| Ours v5 49000 | 0.7595 / 0.7537 | 0.5204 / 0.5380 | 0.5051 / 0.5095 |
| Ours v5 92000 | 0.7619 / 0.7524 | 0.5535 / 0.5429 | 0.4767 / 0.4876 |
| Ours v5 140000 | 0.7732 / 0.7626 | 0.5559 / 0.5431 | 0.5129 / 0.5187 |
| Ours v5 159000 | 0.7737 / 0.7597 | 0.5464 / 0.5354 | 0.5119 / 0.5187 |
| Ours V6 ESMC 15000 | 0.7641 / 0.7573 | 0.5361 / 0.5387 | 0.5015 / 0.5111 |
| Ours V6 no-ESMC 110000 | 0.7336 / 0.7289 | 0.5001 / 0.4967 | 0.4950 / 0.4973 |
| Ours V6 no-ESMC 150000 | 0.7299 / 0.7205 | 0.4903 / 0.4875 | 0.4910 / 0.4936 |

fullA / fullB 分别是仅长链 α / β；论文重复数字按原表保留。Ours 的 CDR3α/β 只用于核验长链，不额外拼接；MHC、V/J 原字段没有输入。来源：[159000 LongAB summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_159000_20260917_others_longab/others_longab/AS/summary.json)、[140000 LongAB summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_140000_20260917_others_longab/others_longab/AS/summary.json)、[92000 LongAB summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_92000_20260915_others_longab/others_longab/AS/summary.json)、[49000 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_49000_20260913_others_longab/others_longab/AS/summary.json)。92000 的 unseen independent AUROC 低于 0.5，按原值记录，不解释成有效反向预测。

others 三轨均使用同一公开 AS 行集，每轨 18,868 个唯一输入。论文 TCRconv seen-independent 原表计数为 834（524 正/310 负），当前公开 others 为 626（313/313），不能因 LongAB seen_test AUROC 较高就宣称超过 TCRconv。长链在两个 independent 集上没有体现优势。

29k/49k 的 others 行身份/标签和输入语义已核对，但源码指纹不同，**仅为历史 checkpoint 对照，不是只改权重的因果实验**。140000 / 92000 / 49000 使用同一冻结协议与官方 AS 行集，但仍不是只改 step 的受控消融。全部 49000 差值、逐折预测 SHA 与配对检查见最终 comparison 的 `old_checkpoint_comparison`；新汇总不混入旧协议或其他系列 Ours 数值。

<a id="tcr-clustering"></a>

### §0.2 T2 Clustering

Ours 两项均仅编码 CDR3β，**不输入表位或 binding 标签**；表位仅用于聚类评分。协议：[T2 任务](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T2_CLUSTERING.md)。

#### T2A：阈值聚类，purity–retention

4,779 个配对身份单位；Ours v5 159000 完整曲线 AUC = **0.416182**（140000：0.424461；92000：0.431322；49000：0.439354）。v2 ESMC 15000 AUC = **0.457516**（Success `t-20260927050122-6h9qk`）；v2 no-ESMC 110000 AUC = **0.412059**（Success `t-20260927050224-j78lg`）。v2 no-ESMC 150000 AUC = **0.412871**（Success `t-20260928002345-jtmjf`）。下表比较各baseline retention附近的Ours点，不将单独purity当作整体排名；AUC不与baseline单点purity混比。v2 九点对照见各 `metrics.json` 的 `aligned_points`，未扩本表列。

| baseline | 单位 | `[P]` retention / purity | `[A]` purity | Ours 92000 实际 retention / purity | Ours 140000 实际 retention / purity | Ours 159000 实际 retention / purity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HD | chain | 0.27 / 0.69 | 0.694 | 0.270140 / 0.736638 | 0.270140 / 0.725794 | 0.270140 / 0.759876 |
| LD | chain | 0.31 / 0.57 | 0.573 | 0.310316 / 0.670937 | 0.310525 / 0.656334 | 0.310316 / 0.697235 |
| TCRMatch | chain | 0.09 / 0.82 | 0.824 | 0.092279 / 0.988662 | 0.090395 / 0.981481 | 0.087675 / 0.988067 |
| iSMART | chain | 0.18 / 0.75 | 0.748 | 0.180372 / 0.940835 | 0.180372 / 0.947796 | 0.180163 / 0.950058 |
| GIANA | chain | 0.25 / 0.65 | 0.649 | 0.250471 / 0.766082 | 0.250052 / 0.767364 | 0.250262 / 0.785953 |
| clusTCR | pair | 0.09 / 0.99 | 0.993 | 0.092279 / 0.988662 | 0.090395 / 0.981481 | 0.087675 / 0.988067 |
| GLIPH2 | pair | 0.22 / 0.80 | 0.798 | 0.220548 / 0.851044 | 0.220130 / 0.866920 | 0.220130 / 0.889734 |
| DeepTCR | pair | 0.92 / 0.63 | 0.632 | 0.920067 / 0.180805 | 0.920067 / 0.183534 | 0.920276 / 0.187358 |
| TCRdist3 | pair | 0.44 / 0.42 | 0.422 | 0.440050 / 0.453162 | 0.440050 / 0.439848 | 0.440469 / 0.474584 |

92000、140000 与159000 各自九点 gap 均小于预设 0.02，但统计单位 chain/pair 和各方法实际输入范围不同，数值可对齐不代表完全同协议。[P] 为论文 Fig 3A，[A] 为作者聚类产物的既有计数校准；精确 tau / gap / comparison_allowed 与完整曲线来源：[159000 T2A metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering/ours_tcr_v5_159000_20260917_runtime_v2/metrics.json)、[140000 T2A metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering/ours_tcr_v5_140000_20260917_runtime_v2/metrics.json)、[92000 T2A metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering/ours_tcr_v5_92000_20260915_runtime_v2/metrics.json)；49000 对照：[T2A metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering/ours_tcr_v5_49000_20260913_runtime_v2/metrics.json)。校准表来源见最终 comparison。

#### T2B：embedding → KMeans

9,033 条 β / 25 表位；K=10…100、步长 5，seed=0；报告 **19 个 K 的均值**，不在评价集上选 best K。

| 方法 / checkpoint | mean ARI ↑ | mean NMI ↑ | mean purity ↑ | 输入 / 来源 |
|---|---|---|---|---|
| SCEPTR [L] | 0.0328 | 0.1587 | 0.3388 | β＋TRBV，历史 |
| TCR-BERT MLM-only [L] | 0.0161 | 0.0948 | 0.2869 | β-only，历史 |
| ESM2-150M [L] | 0.0068 | 0.0676 | 0.2584 | β-only，历史 |
| ProtBERT [L] | 0.0056 | 0.0606 | 0.2478 | β-only，历史 |
| k-mer [C] | 0.0102 | 0.0793 | 0.2680 | β-only，历史 |
| Ours v5 49000 | 0.0204 | 0.1243 | 0.3052 | β-only，历史对照 |
| Ours v5 92000 | 0.0197 | 0.1261 | 0.3065 | β-only，历史对照 |
| Ours v5 140000 | 0.0188 | 0.1241 | 0.3047 | β-only，历史对照 |
| Ours v5 159000 | 0.0191 | 0.1237 | 0.3042 | β-only，当前 |
| Ours V6 ESMC 15000 | 0.0180 | 0.1234 | 0.3010 | β-only，Success `t-20260927050125-m9624` |
| Ours V6 no-ESMC 110000 | 0.0240 | 0.1388 | 0.3138 | β-only，Success `t-20260927050228-h96g7` |
| Ours V6 no-ESMC 150000 | 0.0232 | 0.1406 | 0.3180 | β-only，Success `t-20260928002349-nmdvz` |

本 K-sweep 的对应论文 [P] 精确数值未获取；本地 baseline 是独立参考层，不拿 T2A 或 T3 数字代填。[L]/[C] 未在本轮重跑。来源：[159000 T2B metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering_embed/ours_tcr_v5_159000_20260917_runtime_v2/metrics.json)、[140000 T2B metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering_embed/ours_tcr_v5_140000_20260917_runtime_v2/metrics.json) 的 `kmeans_*_mean`、[92000 T2B metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering_embed/ours_tcr_v5_92000_20260915_runtime_v2/metrics.json)、[49000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering_embed/ours_tcr_v5_49000_20260913_runtime_v2/metrics.json) / [49000 全部 K 点](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering_embed/ours_tcr_v5_49000_20260913_runtime_v2/curve.csv)；历史 baseline 原始路径/SHA 在最终 comparison。没有 v5 29k 同协议 T2 结果。

<a id="tcr-representation"></a>

### §0.3 T3 Representation：few-shot NN AUROC

冻结受体表征、不训练分类头。Ours 输入 CDR3αβ，不输入表位/binding；表位标签用于 support/query 抽样和 AUROC。**本地 broad/deep 不是 SCEPTR 论文同版数据复现。** 协议与 baseline 输入核查：[T3 任务](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md)。

#### T3-broad：24 表位本地任务

Ours 唯一受体 train/test=6,885/1,721，每个有效表位/shot 使用 5 seeds。表内均为 macro AUROC。

| 方法 / checkpoint | k=1 | k=2 | k=5 | k=10 | k=20 | k=50 | k=100 |
|---|---|---|---|---|---|---|---|
| SCEPTR [L] | 0.5869 | 0.6219 | 0.6552 | 0.6919 | 0.7110 | 0.7152 | 0.7413 |
| TCRdist [L] | 0.5772 | 0.6197 | 0.6439 | 0.6740 | 0.6870 | 0.6939 | 0.7282 |
| CDR3-Levenshtein [C] | 0.5545 | 0.5757 | 0.5971 | 0.6265 | 0.6414 | 0.6543 | 0.6834 |
| k-mer [C] | 0.5425 | 0.5702 | 0.5967 | 0.6231 | 0.6432 | 0.6479 | 0.6734 |
| ESM2-150M [L] | 0.5613 | 0.5830 | 0.5894 | 0.6191 | 0.6381 | 0.6419 | 0.6706 |
| ProtBERT [L] | 0.5496 | 0.5761 | 0.5870 | 0.6073 | 0.6273 | 0.6282 | 0.6506 |
| TCR-BERT MLM-only [L] | 0.5470 | 0.5671 | 0.5773 | 0.6034 | 0.6231 | 0.6244 | 0.6414 |
| Ours v5 49000 | 0.5435 | 0.5720 | 0.5949 | 0.6127 | 0.6335 | 0.6180 | 0.6447 |
| Ours v5 92000 | 0.5422 | 0.5748 | 0.5925 | 0.6073 | 0.6258 | 0.6237 | 0.6479 |
| Ours v5 140000 | 0.5445 | 0.5727 | 0.5958 | 0.6126 | 0.6302 | 0.6210 | 0.6497 |
| Ours v5 159000 | 0.5443 | 0.5732 | 0.5949 | 0.6121 | 0.6299 | 0.6226 | 0.6492 |
| Ours V6 ESMC 15000 | 0.5589 | 0.5759 | 0.5829 | 0.5964 | 0.6132 | 0.6110 | 0.6219 |
| Ours V6 no-ESMC 110000 | 0.5554 | 0.5819 | 0.5978 | 0.6142 | 0.6288 | 0.6173 | 0.6376 |
| Ours V6 no-ESMC 150000 | 0.5527 | 0.5793 | 0.5930 | 0.6088 | 0.6208 | 0.6113 | 0.6282 |

Ours 有效表位数依次为 24/24/24/24/24/20/15，高 shot 的均值不是同一个固定人群。历史 baseline 尚未用本轮唯一受体/多标签规则重跑，**只能并列参考，不能据此给修正后协议排名**；其输入字段差异见下方 deep 说明。本地 broad 对应的论文 [P] 数值未获取。

来源：[159000 broad fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation/ours_tcr_v5_159000_20260917_runtime_v2/fewshot.json)、[140000 broad fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation/ours_tcr_v5_140000_20260917_runtime_v2/fewshot.json) 的 `auroc_by_shot` / `n_epitopes_by_shot`、[92000 broad fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation/ours_tcr_v5_92000_20260915_runtime_v2/fewshot.json)、[49000 fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation/ours_tcr_v5_49000_20260913_runtime_v2/fewshot.json)、[49000 support / 聚合核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v1/repr_short.json)。std 是表位间 population SD，不是置信区间；完整 std 保留原 JSON。

#### T3-deep：六表位本地任务

25,816 条 universe；8 档 shots × 100 seeds × 6 targets。以下先保留论文代表值，再列本地完整曲线。

| 论文模型 [P] | Table SI，k=200 六表位宏均值 |
|---|---|
| CDR3-Levenshtein | 0.736833 |
| ESM2-T6-8M | 0.716667 |
| ProtBERT | 0.705333 |
| SCEPTR | 0.798167 |
| TCR-BERT | 0.734667 |
| TCRdist | 0.783167 |

论文宏均值由六个印刷三位小数值计算，来源与逐表位转录在最终 comparison 的 `sceptr_paper_baselines`。ESM2 论文版本为 **T6-8M＋重建长链**，不是本地 ESM2-150M＋CDR3。论文数据与本地不同，不能直接相减解释模型能力差距。

| 本地方法 / checkpoint | k=1 | k=2 | k=5 | k=10 | k=20 | k=50 | k=100 | k=200 |
|---|---|---|---|---|---|---|---|---|
| SCEPTR [L] | 0.6401 | 0.6673 | 0.7020 | 0.7182 | 0.7375 | 0.7602 | 0.7748 | 0.7903 |
| TCRdist [L] | 0.6466 | 0.6724 | 0.7008 | 0.7156 | 0.7325 | 0.7516 | 0.7642 | 0.7770 |
| CDR3-Levenshtein [C] | 0.6256 | 0.6480 | 0.6712 | 0.6836 | 0.6958 | 0.7110 | 0.7223 | 0.7334 |
| k-mer [C] | 0.6074 | 0.6334 | 0.6653 | 0.6777 | 0.6901 | 0.7059 | 0.7173 | 0.7284 |
| ESM2-150M [L] | 0.5914 | 0.6055 | 0.6315 | 0.6429 | 0.6563 | 0.6738 | 0.6848 | 0.6958 |
| ProtBERT [L] | 0.5847 | 0.5992 | 0.6237 | 0.6343 | 0.6480 | 0.6664 | 0.6808 | 0.6944 |
| TCR-BERT MLM-only [L] | 0.5786 | 0.5989 | 0.6309 | 0.6423 | 0.6563 | 0.6699 | 0.6806 | 0.6916 |
| Ours v5 49000 | 0.5810 | 0.6056 | 0.6327 | 0.6452 | 0.6550 | 0.6700 | 0.6792 | 0.6895 |
| Ours v5 92000 | 0.5753 | 0.5985 | 0.6282 | 0.6406 | 0.6514 | 0.6668 | 0.6773 | 0.6871 |
| Ours v5 140000 | 0.5784 | 0.5990 | 0.6297 | 0.6426 | 0.6542 | 0.6692 | 0.6795 | 0.6890 |
| Ours v5 159000 | 0.5794 | 0.5994 | 0.6281 | 0.6413 | 0.6526 | 0.6678 | 0.6785 | 0.6882 |
| Ours V6 ESMC 15000 | 0.5668 | 0.5861 | 0.6063 | 0.6167 | 0.6284 | 0.6458 | 0.6581 | 0.6680 |
| Ours V6 no-ESMC 110000 | 0.5912 | 0.6129 | 0.6382 | 0.6494 | 0.6577 | 0.6716 | 0.6789 | 0.6871 |
| Ours V6 no-ESMC 150000 | 0.5952 | 0.6187 | 0.6423 | 0.6524 | 0.6603 | 0.6730 | 0.6811 | 0.6887 |

历史 SCEPTR / TCRdist 有额外 V 或 V/J 信息；Levenshtein/k-mer/ESM2/ProtBERT 为 CDR3αβ 控制，TCR-BERT MLM-only 为 β-only 控制。历史 baseline 的输入/运行身份未按本轮全部门禁重新锁定，不把 [L]/[C] 升级为论文复现。

来源：[159000 deep fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation_paper6/ours_tcr_v5_159000_20260917_runtime_v2/fewshot.json)、[140000 deep fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation_paper6/ours_tcr_v5_140000_20260917_runtime_v2/fewshot.json)、[92000 deep fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation_paper6/ours_tcr_v5_92000_20260915_runtime_v2/fewshot.json)、[49000 fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation_paper6/ours_tcr_v5_49000_20260913_runtime_v2/fewshot.json)、[49000 deep 身份/覆盖/聚合核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v1/repr_deep.json)。未重跑未保存的全部 NN 距离，不能声称独立重算所有 AUROC。v5 29k 的 broad/deep 同协议成绩均未获取。

<a id="tcr-generation"></a>

### §0.4 T4 Generation：表位条件 CDR3β

**2026-09-16 共享 sampler 修正提示：** 49000/92000 多步 Ours 也调用相同的 ESMC+LLaDA 采样路径，数值仍为修复前协议记录，不作修正版 headline。140000 是修复后代码路径上的新生成（pairing 同日矩阵已标 `committed_generated_residues_pending_mask`）；本轮没有单独做 T4 ESMC 逐轮输入审计，不把 140000 写成已完成 T4 反馈验收。详见 [T4 §7](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。 159000 为修复后路径，真实 GPU gate 通过目标占位符不变性、固定context及评分JSON检查；这不等于对全部正式候选逐轮抓取ESMC输入。

V5 159000 对照的输入为表位＋MHC pseudo（34 aa）；未知框架、CDR1/2 和 α 保持 X，仅 CDR3β core 参与生成。每个目标 **K=100 随机候选＋1 独立 greedy**，max_iter=32，长度来自训练先验，不读取测试参考 CDR3 长度。140000为修复后历史对照；49000/92000为修复前协议记录，不作修正版headline。当前不是 αβ 配对生成，后者仍延期。

表中 d_edit 越低越好，seq-recovery / F1@100 越高越好。序列相似度不等于实验结合，F1=0 只说明未精确命中参考集合。相同目标集合仍可能有 MHC、参考长度、候选筛选和采样预算差异；[T4 协议 / baseline 条件审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) 中的限制继续有效。

#### V7 两轮固定点实测（2026-10-07T00:39:13.894414+00:00）

50082正式权重的34条件全量生成和CPU评分均真实exit0。[完整报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_tcr/report.json)绑定原始候选、[metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_tcr_scores_cpu_20261006/metrics.json)、实际退出记录、源/数据哈希和V5对照；仅重算保存分数的宏平均/分母，没有再次运行模型或重评分序列。3400随机候选全部保留（空串0、均生成终止符），其中3385满足既有CDR3β语法规则，有效率99.5588%；15个语法无效候选也保留在评分中。另34个独立argmax不混入K100，无greedy回退。四个主视图均完整覆盖，未删除目标。

当前V7在四个视图的编辑距离和序列恢复率均劣于V5 159000；F1均为0，不支持TCR生成提升的结论。格式有效性不代表实验结合能力，F1=0也不证明新序列全部不结合。V7输入为表位＋给定34位MHC pseudo、缺失alpha；整段64槽位含锚点/终止符均由模型生成，32步、CFG0、temperature1、batch1。V5固定C/F并使用训练长度先验、batch8；模型规模、数据、训练预算、原生布局和随机数分配也不同，因此不是受控架构消融。

保存输出的长度统计如下，均为“各目标均值再等权平均”，参考数量各目标不同。V7生成明显更长，这是描述性现象，尚未隔离长度对评分差距的因果贡献，未按test调整长度/解码或另选checkpoint。

| 集合 | V7 生成长度均值 | V5 生成长度均值 | 参考长度均值 |
|---|---:|---:|---:|
| benchmark14 | 19.8971 | 14.5243 | 14.6628 |
| held20 | 19.9125 | 14.5085 | 14.5093 |

optional novelty_vs_conditioning因当前条件参考库无对应表位而为null，并非0或已证明新颖/去污；V5该项同样未提供可比数字。历史common6只是同一六目标视图，不证明V7训练未见性。数据审核仍仅为既有observed-pair exact范围，不扩展为全部query/few-shot或近似相似度隔离。协议与输入验收范围见[任务§8.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。

#### T4-1 benchmark14：全部 14 个目标

| 方法 / checkpoint | 覆盖 | d_edit ↓ | seq-rec ↑ | F1@100 ↑ | 来源 |
|---|---|---|---|---|---|
| TCRT5 | 14目标未重评分 | — | — | — | [A] 当前正式评分为13目标，见下表 |
| GRATCR | 14目标未重评分 | — | — | — | [A] 当前正式评分为13目标，见下表 |
| ER | 14目标未重评分 | — | — | — | [A] 当前正式评分为13目标，见下表 |
| TcrDesign | 14/14 | 4.322143 | 0.613834 | 0.001082 | [R] 历史原生权重管线 |
| TCRDiff | 14/14 | 4.719761 | 0.587113 | 0.000922 | [R] 历史原生权重管线 |
| OLGA | 14/14 | 5.931429 | 0.500620 | 0.000000 | [C] 无条件 |
| Ours v5 49000 | 14/14 | 5.674286 | 0.539441 | 0.000000 | 历史对照，完整 β-only |
| Ours v5 92000 | 14/14 | 5.722857 | 0.538980 | 0.000000 | 历史对照，完整 β-only |
| Ours v5 140000 | 14/14 | 4.923571 | 0.579153 | 0.000000 | 历史完整 β-only |
| Ours v5 159000 | 14/14 | 4.882857 | 0.582537 | 0.000000 | V5对照，完整 β-only |
| Ours V6 ESMC 15000 | 14/14 | 6.237598 | 0.478002 | 0.000000 | Success `t-20260927050137-qtv4x`；V6 全长 β+EOS |
| Ours V6 no-ESMC 110000 | 14/14 | 5.413344 | 0.528990 | 0.000000 | Success `t-20260927050241-nhfms`；V6 全长 β+EOS |
| Ours V6 no-ESMC 150000 | 14/14 | 5.218078 | 0.553701 | 0.000000 | Success `t-20260928002359-8v56z`；V6 全长 β+EOS |
| Ours V7 f011 50082（两轮，原生junction） | 14/14 | 8.834286 | 0.402453 | 0.000000 | [本轮报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_tcr/report.json)；原生槽位/自主终止 |

**不是所有模型只测试了六个目标。** 14 是完整集合；13 是排除 `RVRAYTYSK_HLA-A*03:01` 模拟保留目标的论文主视图；6 是其中本轮 v5 训练/验证均未出现的表位子集。官方三种方法的原始存储预测也含 RVR，但当前正式评分产物未对其补成完整 14，不填造分数。来源：[159000 benchmark14 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_159000_20260917_benchmark14_pmhc_beta_v2/metrics.json)、[140000 benchmark14 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_140000_20260917_benchmark14_pmhc_beta_v2/metrics.json)、[92000 benchmark14 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_92000_20260915_benchmark14_pmhc_beta_v2/metrics.json)、[49000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_49000_20260913_benchmark14_pmhc_beta_v2/metrics.json)。

#### 论文主视图 sparse13：官方预测重评分

| 方法 / checkpoint | d_edit ↓ | seq-rec ↑ | F1@100 ↑ | 来源 |
|---|---|---|---|---|
| TCRT5 | 4.467385 | 0.601754 | 0.000296 | [A] 官方预存预测 |
| GRATCR | 4.396615 | 0.595101 | 0.000000 | [A] 官方预存预测 |
| ER | 6.067805 | 0.336948 | 0.000000 | [A] 官方预存预测 |
| Ours v5 49000 | 5.781538 | 0.530099 | 0.000000 | 同一14目标产物的13目标视图 |
| Ours v5 92000 | 5.840000 | 0.529980 | 0.000000 | 同一14目标产物的13目标视图 |
| Ours v5 140000 | 5.053077 | 0.567773 | 0.000000 | 同一14目标产物的13目标视图 |
| Ours v5 159000 | 5.018462 | 0.571268 | 0.000000 | 同一14目标产物的13目标视图 |
| Ours V6 ESMC 15000 | 6.365765 | 0.469397 | 0.000000 | 同一14目标产物的13目标视图 |
| Ours V6 no-ESMC 110000 | 5.566019 | 0.515108 | 0.000000 | 同一14目标产物的13目标视图 |
| Ours V7 f011 50082（两轮，原生junction） | 8.990769 | 0.398318 | 0.000000 | [本轮报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_tcr/report.json)；原生槽位/自主终止 |

这里是论文主目标集合上的 **[A] 重评分**，不把重评分的小数冒充独立转录的 [P]。最终收集器没有单独的生成论文 [P] 精确指标项；TcrDesign / TCRDiff 的原生论文协议也不同，对应 [P] 比较值保留“未获取/未对齐”，不以 held20 本地结果替代论文测试成绩。

#### common6：固定历史六目标子集

| 方法 / checkpoint | d_edit ↓ | seq-rec ↑ | F1@100 ↑ | 来源 |
|---|---|---|---|---|
| TCRT5 | 4.696333 | 0.584639 | 0.000329 | [A] |
| GRATCR | 4.604667 | 0.583890 | 0.000000 | [A] |
| ER | 6.074833 | 0.320566 | 0.000000 | [A] |
| TcrDesign | 4.856667 | 0.573352 | 0.000000 | [R] 历史管线 |
| TCRDiff | 5.157347 | 0.543338 | 0.000000 | [R] 历史管线 |
| OLGA | 6.343333 | 0.469924 | 0.000000 | [C] 无条件 |
| Ours v5 49000 | 5.890000 | 0.519255 | 0.000000 | 历史对照，β-only |
| Ours v5 92000 | 5.975000 | 0.519036 | 0.000000 | 历史对照，β-only |
| Ours v5 140000 | 5.328333 | 0.544756 | 0.000000 | 历史 β-only |
| Ours v5 159000 | 5.211667 | 0.552989 | 0.000000 | V5对照，β-only |
| Ours V6 ESMC 15000 | 6.522574 | 0.454825 | 0.000000 | 同一14目标产物的6目标视图 |
| Ours V6 no-ESMC 110000 | 5.549129 | 0.515363 | 0.000000 | 同一14目标产物的6目标视图 |
| Ours V7 f011 50082（两轮，原生junction） | 9.173333 | 0.384131 | 0.000000 | [本轮报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_tcr/report.json)；原生槽位/自主终止 |

6 个目标是相同目标比较，不是全部模型的最大覆盖交集，也不证明所有生成条件已对齐；历史V5的未见表位标记不自动适用于V7。来源：最终 comparison 的 `generation_baselines` 及同一 benchmark14 metrics。

#### T4-2 held20：官方验证来源的补充集合

| 方法 / checkpoint | 覆盖 | d_edit ↓ | seq-rec ↑ | F1@100 ↑ | 来源 |
|---|---|---|---|---|---|
| TcrDesign | 20/20 | 1.474000 | 0.870222 | 0.152686 | [R] 历史管线 |
| TCRDiff | 20/20 | 2.953920 | 0.731759 | 0.011777 | [R] 历史管线 |
| Ours v5 49000 | 20/20 | 4.100500 | 0.658046 | 0.000500 | 历史对照，β-only |
| Ours v5 92000 | 20/20 | 4.198000 | 0.650140 | 0.000000 | 历史对照，β-only |
| Ours v5 140000 | 20/20 | 2.983500 | 0.737351 | 0.001500 | 历史 β-only |
| Ours v5 159000 | 20/20 | 2.971000 | 0.740938 | 0.002500 | V5对照，β-only |
| Ours V6 ESMC 15000 | 20/20 | 4.661580 | 0.590076 | 0.000000 | Success `t-20260927152010-n854g`；V6 全长 β+EOS |
| Ours V6 no-ESMC 110000 | 20/20 | 3.459913 | 0.690945 | 0.001008 | Success `t-20260927050246-6sp5x`；V6 全长 β+EOS |
| Ours V6 no-ESMC 150000 | 20/20 | 3.252215 | 0.713999 | 0.001010 | Success `t-20260928002403-9nfd5`；V6 全长 β+EOS |
| Ours V7 f011 50082（两轮，原生junction） | 20/20 | 6.752000 | 0.450613 | 0.000000 | [本轮报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_tcr/report.json)；原生槽位/自主终止 |

held20 来自 TCRT5 官方验证参考，不是“我们的 20 条验证样本”，也不是新表位测试。当前正式 TCRT5/GRATCR/ER 官方预测及 OLGA 评分产物未覆盖本集合，不从其他集合借数。来源：[159000 held20 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_159000_20260917_held20_pmhc_beta_v2/metrics.json)、[140000 held20 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_140000_20260917_held20_pmhc_beta_v2/metrics.json)、[92000 held20 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_92000_20260915_held20_pmhc_beta_v2/metrics.json)、[49000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_49000_20260913_held20_pmhc_beta_v2/metrics.json)、[49000 完整序列重评分审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v2/held20.json)。

#### 训练重叠与视图解释（所有 T4 表必读）

| Ours v5 92000 视图 | 目标数 | d_edit ↓ | seq-rec ↑ | F1@100 ↑ | 独立 greedy Char-BLEU |
|---|---|---|---|---|---|
| benchmark14 全量 | 14 | 5.722857 | 0.538980 | 0.000000 | 0.653244 |
| 其中训练已见表位（不含 RVR） | 7 | 5.724286 | 0.539362 | 0.000000 | 0.650969 |
| 其中未见表位（common6） | 6 | 5.975000 | 0.519036 | 0.000000 | 0.612743 |
| held20 全部已见表位 | 20 | 4.198000 | 0.650140 | 0.000000 | 0.912557 |

| Ours v5 140000 视图 | 目标数 | d_edit ↓ | seq-rec ↑ | F1@100 ↑ | 独立 greedy Char-BLEU |
|---|---|---|---|---|---|
| benchmark14 全量 | 14 | 4.923571 | 0.579153 | 0.000000 | 0.739736 |
| 其中训练已见表位（不含 RVR） | 7 | 4.817143 | 0.587502 | 0.000000 | 0.802045 |
| 其中未见表位（common6） | 6 | 5.328333 | 0.544756 | 0.000000 | 0.623664 |
| held20 全部已见表位 | 20 | 2.983500 | 0.737351 | 0.001500 | 0.947890 |

| Ours v5 159000 视图 | 目标数 | d_edit ↓ | seq-rec ↑ | F1@100 ↑ | 独立 greedy Char-BLEU |
|---|---|---|---|---|---|
| benchmark14 全量 | 14 | 4.882857 | 0.582537 | 0.000000 | 0.763420 |
| 其中训练已见表位（不含 RVR） | 7 | 4.852857 | 0.586936 | 0.000000 | 0.801371 |
| 其中未见表位（common6） | 6 | 5.211667 | 0.552989 | 0.000000 | 0.687139 |
| held20 全部已见表位 | 20 | 2.971000 | 0.740938 | 0.002500 | 0.948652 |

这是分层统计，不是额外训练或采样消融。T4 去污染以“β core＋表位”参考配对为键，不是删除整个表位：benchmark14 训练集仍有 7 个表位，但参考 β core / 配对精确命中均为 0。held20 的 20 个表位全部出现，且很多参考 β core 在其他上下文中出现，但参考配对精确命中为 0；不能称全新 β / 全新表位。

完整 source / train / valid 数字、七表位名单及数据成员≠实际 batch 消费的边界，唯一详见 [T4 §7.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) 和 [最终 overlap 报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/audit_v5_overlap_20260913/full_scan_v2/report.json)。不覆盖 ESMC 更早预训练或近同源。

Ours Char-BLEU 使用独立 greedy，不能与历史 `gen[0]` 替代 greedy 的行直接排序。缺少 conditioning 参考时 novelty 为 null，不是新颖性为 0；主指标覆盖完整。没有 v5 29k 同协议 T4 结果；没有当前 αβ 配对质量表。

## 文档与历史证据

- 数字只在本文件维护；协议/缺陷见 [AB 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/AB_BASELINE_EVALUATION_AUDIT.md)、[TCR 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md) 和各任务文档。
- 本次整理前全文（含旧模型、失效采样、PPI/NbBench/FLAb 和历史完整 baseline 表）：[RESULTS_ARCHIVE_20260915.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_ARCHIVE_20260915.md)。归档不作为当前排行榜，不继续追加新数字。
- 执行时间/提交取消日志见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)；文档整理记录见 [PROGRESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROGRESS.md)。2026-09-17 已加入 v5 159000 全部 AB/TCR 最终汇总（AB pairing p0/CFG0/16步；CDR 2步；Specificity 100 epoch；TCR T4 32步），49000/92000/140000历史数字保留。核对边界与30个来源文件SHA见 [汇总核对记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_159000_20260917/results_document_audit.json)。本轮没有重新训练、推理、评分、submit或cancel。

<a id="ab-antigen"></a>
## AB-AG：抗原条件常规配对抗体（2026-09-24）

**范围更新**：FLAb2 已按用户要求排除；MAGE 保留。历史六仓库/CLI 审计是当时快照，不是当前执行清单。MAGE 官方单样本检查已提交：`t-20260924115209-csmbz`，平台 Queue；当前没有生成结果。执行配置见 [AB-AG 手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)，作业状态见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。

**尚无新模型推理、baseline 分数或 Ours 分数；没有完成正式 benchmark。** 本轮只进行原论文/代码比较协议核查、源码获取、静态检查、CLI help 尝试和两个数据子集字段检查。任务定义、每个 baseline 的原比较方法及阻塞项见 [AB-AG 官方协议手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。

| 执行检查 | 实际结果 | 证据 |
|---|---|---|
| 官方源码身份 | 6 个仓库，406 个文件固定 commit/SHA256；来源校验通过 | [source audit](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/source_audit_v2/report.json) |
| 静态可运行性 | AbBiBench 缺 final notebook；FLAb2 缺 helper 且部分源文件有语法缺陷；不能把 source_verified 当作可完成评测 | 同上 static_checks_passed / syntax_errors |
| 原 CLI 尝试 | 21 个入口：2 个 help 通过，19 个因依赖/import 失败；MochiBind 默认混入 VHH，不启动 | [CLI 原始报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/cli_probe/report.json) |
| 来源校验工具测试 | 9 项通过；不涉及生物模型推理 | [execution status](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/execution_status.json) |
| MAGE 权重 | 首次 gated 403 已解除；config/tokenizer 已取得，完整权重已下载及校验；已提交官方 n=1 生成检查，当前 Queue，尚无推理结果 | [获取记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/ab_antigen_official/MAGE_V1/acquisition.json) |
| CoV-UniBind 分类候选 | 原始 632 行均有 VH/VL/Ag；353 positive / 279 negative；598 个唯一序列三元组，未去重重分割 | [数据审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/covunibind_data_audit.json) |
| CoV-UniBind SPR 候选 | 原始 839 行均有 VH/VL/Ag；标签 log10_kd，有 242 行等于最大值 4.301029995663981，检测限含义待核；未转成 pKD | 同上 |

无 VHH 评测。当前所有原论文数值仍须逐表转录/对齐，不以源码或数据样本数填补排行榜；候选方法保留在 [逐项台账](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/comparisons.tsv)。正式复跑只能调用对应 benchmark 作者的原脚本/适配器/划分/权重与统计流程；缺失者保持未运行状态，不采用自写替代。

<a id="s2alm-audit"></a>
## S²ALM：文献值与 benchmark 数据审计（2026-09-24）

来源为 [S²ALM Table 2](https://doi.org/10.34133/research.0721)，转录证据：[paper_tables.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/s2alm_audit/paper_tables.json)。以下全部 `[P]`，括号内容照原表保留；没有本地推理或严格同 split 复现，不据此与 Ours 排名。S²ALM 正文/补充资料未提供可核验的全部随机种子及具体测试索引。复现缺口与任务定义见 [S²ALM 复现审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/S2ALM_REPRODUCIBILITY_AUDIT.md)。

| 方法（均为论文值） | 14H Pearson | 14H Spearman | 14L Pearson | 14L Spearman | BioMap Pearson | BioMap Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla BERT [P] | 0.594 (0.021) | 0.480 (0.025) | 0.607 (0.024) | 0.611 (0.027) | 0.492 (0.036) | 0.498 (0.037) |
| Ens-Grad [P] | 0.601 (0.016) | 0.476 (0.023) | 0.637 (0.019) | 0.645 (0.023) | 0.645 (0.031) | 0.664 (0.033) |
| AbMAP [P] | 0.606 (0.015) | 0.510 (0.015) | 0.674 (0.012) | 0.685 (0.016) | 0.637 (0.027) | 0.673 (0.029) |
| AntiBERTa2 [P] | 0.623 (0.011) | 0.545 (0.008) | 0.673 (0.013) | 0.684 (0.012) | 0.633 (0.022) | 0.670 (0.031) |
| ESM-F [P] | 0.634 (0.007) | 0.516 (0.010) | 0.674 (0.011) | 0.681 (0.014) | 0.628 (0.028) | 0.644 (0.024) |
| A2binder [P] | 0.642 (0.012) | 0.553 (0.011) | 0.683 (0.010) | 0.688 (0.015) | 0.701 (0.024) | 0.746 (0.025) |
| S2ALM [P] | 0.650 (0.013) | 0.558 (0.014) | 0.687 (0.015) | 0.693 (0.017) | 0.701 (0.022) | 0.749 (0.028) |

六个 baseline 行与 PALM/A2binder Table 1 的均值及括号数字逐项一致；这不证明已独立复跑或两文使用相同样本索引。S²ALM 正文称 BioMap Spearman 与 A2binder 相同，但表值分别为 0.749 / 0.746；引用时以表值和该文本不一致记录为准，不作额外显著性判断。

| 官方公开数据字段检查（非预测结果） | 实际读取 |
|---|---|
| BioMap | 1,706 行；两条抗体链及 antigen 均非空；列为 pdb、antibody_seq_a、antibody_seq_b、antigen_seq、delta_g；链长范围分别113–263 / 104–260，抗原3–1,676 aa，VH/VL 边界尚待核查 |
| 14H | 13,922 行；heavy、cdr1/2/3、affinity；发布表无 antigen/配对 light 字段 |
| 14L | 18,708 行；light、cdr1/2/3、affinity；发布表无 antigen/配对 heavy 字段 |

证据：[audit_report.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/s2alm_audit/audit_report.json)，PALM 数据 commit `f44090b3fbe5404ae0a9d196f5429a48ef4de5c5`。未过滤、未训练、未换单位。BioMap 是 ΔG 回归候选；14H/14L 是实验校准的 AlphaSeq 亲和力估计，且原入口只有变化链，不能称为已完成 antigen 条件 pKD benchmark。

<a id="biomap-audit"></a>
## BioMap 发布材料审计（2026-09-24；不是模型成绩）

协议与来源见 [BioMap 官方 baseline 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/BIOMAP_BASELINE_REPRODUCIBILITY_AUDIT.md)。以下仅为真实发布材料核查，不代表本地模型预测。

| 核查对象 | 结果 | 证据 |
|---|---|---|
| 原 CSV 内 PDB 1DQJ | 64 行；对照 RCSB 角色及序列前缀，31 行 a=light/b=heavy，33 行 a=heavy/b=light | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/biomap_audit/1dqj_rows.json`、`pdb_1dqj_1.json` / `pdb_1dqj_2.json`；不是全量链分类 |
| Source Data Fig. S3 | 六种方法各 170 对 True/Predicted affinity；无样本 ID 映射 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/biomap_audit/source_fig_s3.json` |
| Fig. S3 真值集合 | A2binder / ESM-F / Ens-Grad / Vanilla BERT 的多重集合相同；AbMAP、AntiBERTa2 各自不同（6 位小数比较） | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/biomap_audit/audit_report.json`；不能证明六种方法共享具体 test split |
| 原始 Model.zip | HTTP 206 读取末尾 131,072 字节；A2binder_affinity 目录仅三编码器 config/weights，无完整回归 head checkpoint | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/biomap_audit/model_zip_inventory.json`；不是完整权重下载或加载成功 |

未重新计算论文性能分数、未生成 Ours 对照、未修正原数据或新建 split。论文表的 `[P]` 数字仍在 [原表转录](#s2alm-audit)。

<a id="biomap-official-run"></a>
## BioMap A2binder 官方代码单 seed 运行（2026-09-24）

**状态：已提交，平台 Queue；尚无 GPU 训练或测试成绩。** task_id=`t-20260924130337-27l9z`。本条为当前状态；前文“权重未取得/环境未验证”是调研时的历史快照。

| 方法 / 来源 | 协议 | Pearson | Spearman | MSE | 状态 |
|---|---|---:|---:|---:|---|
| A2binder `[R]` 待产出 | 作者原 BioMap CSV，seed=3，官方训练和评分；单 seed | — | — | — | Queue；不能写成已复现 |

- CPU 原 loader 实测：train=1,366 / valid=170 / test=170；保存 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/biomap_seed3/split_membership.csv`，不自行实现或替换 split。
- 三个编码器全部完成 CPU 权重加载。Heavy/Light 无缺失 encoder keys；ESM 只有原代码不使用的 pooler 两个参数随机初始化，`last_hidden_state` 编码器权重正常。未做 CPU forward/training。
- 作者核心依赖版本已匹配；细节见 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/biomap_seed3/environment.freeze.txt`、`preflight.json`。source lock 校验 42 个原样文件通过。
- 当前生成的仅为准备/审计产物；最终成绩由 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/biomap_seed3/metrics.json` 与 `run_report.json` 确认，原始预测由作者入口输出 `results/datasplit/.../test_result.csv`。这些正式结果文件在作业执行前尚不存在。
- 采用原配置最多 50 epochs、validation Pearson 选模、early-stop=15；配置只改路径，未修正 H/L 列不一致，未改变 ΔG 标签。论文多次重复与我们的单 seed 不混同，也不称未见抗原泛化。

执行手册见 [AB-AG §8.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md#86-biomap-a2binder-原代码运行)。

<a id="ab-ag-sequence-v1"></a>

### AB–AG sequence v1 / no-ESMC step150000（2026-09-27）

当前范围以 [README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md) 为准。四子集已构建，GPU smoke `t-20260928062524-wls4h` 已提交，平台查询 **Queue**（2026-09-27T22:39Z）；**暂无 GPU 推理、探针或生成质量分数**，CPU preflight 不算模型结果。

- 原始 CoV-UniBind binding632 / SPR839；当前可用 binding601 / SPR839。AbBiBench215697、CHIMERA2858 为派生可用记录，非模型成绩。逐项依据 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/ab_antigen_sequence_v1/build_audit.json`。
- 权重固定 step150000 SHA `a6ce5c2c22b2e938f4573eb552b6a8ce1c2f42a3647a345adc98e6a5a052b048`；输出 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_sequence_v1_noesmc150000`。
- pKD 指标待单位/检测限核验；SPR 抗原同源组仅两组，严格该轴三分不可用。不得以随机拆分替代。
- 实际 prepared 训练集 Ag/H/L/H3/配对精确重叠已完成；近同源检索仍在运行，原始上游来源未穷尽，不标 decontaminated。

2026-09-28 启动复核（2026-09-28T00:49:34.768573+00:00）：no-ESMC150000 smoke `t-20260928062524-wls4h` 平台仍为 Queue；自动看护存活，尚无模型成绩。本次未重复提交。

### 2026-09-28 本地执行覆盖

用户明确要求本地测试、不提交任务；已停止自动提交并取消 Volc smoke（`t-20260928062524-wls4h`，终态 Killed）。本地 A100 80GB 上启动固定 no-ESMC150000 四子集 smoke，PID833743，日志 [ab_ag_sequence_v1_noesmc150000_local_20260928/run.log](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_sequence_v1_noesmc150000_local_20260928/run.log)。沿用 K20 与正确/替换条件，未改指标协议；启动不等于完成，未获得有效完整模型成绩。

本地进展复核：binding、SPR、AbBiBench 各1条 smoke 的 metrics.json 均为 complete；CHIMERA 已开始输出候选，仍在运行。单样本验证无法给出有意义的 AUC/Spearman，不作为正式 benchmark 分数。

## 2026-09-28 本地正式 AB–Ag 评测启动

用户授权正式运行；启动本地顺序流水线 PID835846，先等待四子集 smoke 产物通过，再执行 binding、SPR、CHIMERA、17个 AbBiBench assay 共20项。无新增 Volc 提交。沿用固定 no-ESMC150000、原数据和指标协议；每项24小时上限，失败停止，保留断点。日志 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_sequence_v1_noesmc150000_local_20260928/full_pipeline.log`，状态 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_sequence_v1_noesmc150000_local_20260928/local_pipeline_state.json`，逐任务日志 `full_<key>.log`。当前启动时为 waiting_for_smoke，不能报告正式结果已完成。

## 2026-09-28 解除正式预测任务对 CHIMERA 检索的依赖

原本地流水线仍为 waiting_for_smoke，未运行任何正式任务；停止等待进程 PID835846，以 PID848460 重启。Binding/SPR/AbBiBench smoke 已通过，因此预测类正式任务不再等待 CHIMERA 新颖性检索；CHIMERA 正式任务仍检查自身完整 smoke。未终止检索、未改变指标、未提交 Volc。状态：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_sequence_v1_noesmc150000_local_20260928/local_pipeline_state.json`。

## 2026-09-30 PLL 批量调整并恢复本地评测

用户要求扩大批量。以剩余4fqi_h1/h3的6条代表记录实测批量2/16/32/64/128；平均每条1.926/1.334/1.379/1.313/1.396秒，分数最大绝对差均为0；batch64峰值allocated显存31.08GiB。选择64，约1.47倍加速（小样本计时，不保证全程同比）。保留4fqi_h1已有20404条完整预测，原评分代码和checkpoint不变，运行参数变化记录在各输出目录batch_execution_change.json。本地流水线PID1614629支持读取既有状态并跳过完成项，续跑h1及h3；未提交集群任务。验证记录：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_sequence_v1_noesmc150000_local_20260928/batch_benchmark.json`。


2026-10-09 修复复测已终态：{"t1-cdr3b": "success", "ag-chimera": "success", "t4-benchmark14": "success", "t4-held20": "success"}。结果及CDR3语法有效率见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/repair_final_20261009.json；原始候选保留，MMseqs临时目录已清理。82项回归及60组T1 AUROC复核通过；未见表位泛化弱仍未由流程修复解决。
