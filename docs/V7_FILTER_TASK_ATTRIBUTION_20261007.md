# V7 两轮过滤与下游任务对应

2026-10-07T05:35:18.340Z。只读已有删除台账与规则，未扫描原始序列、重新注释、修改数据或训练。数字单位为被删语料行，包含train+valid；一条行可能命中多个任务/原因，不可跨行相加。绑定哈希及完整计数见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_filter_attribution_20261007/summary.json；可复算脚本同目录audit.mjs。

## 初版过滤：479684行

由immune_v6_binding_only到immune_v7_clean_r2，原始输入split计数train471506、valid8178；后续保留样本重新按family分组划分，因此不同于两个manifest训练行数净差。主要实际删除来源OTS399768、ASD34315、native34051。cdr3a原因400683中OTS362096、native31656、papers5467、TRAIT1464；cdr3b52290原因中OTS47085。

规则为冻结joint easy-linclust簇：cdr3a/cdr3b/H3 identity0.8、heavy0.95、antigen0.7、peptide0.8，coverage0.8/cov-mode0/cluster-mode1。凡所属簇包含保护序列就标blocked；不能把同簇计数当直接逐对相似度已验证或exact重复数。任一受体链命中即可删整条记录，不要求同一个测试αβ。额外peptide编辑距离筛选，以及抗原包含测试表位子串、抗原双向覆盖至少80%的包含规则。

原保护库涵盖：OTS holdout、T1/tcr_binding和NM2025、T2两聚类集、T3两表征集、旧TCR generation holdout/public beta参考；AB包括SAbDab CDR、OAS轻链配对holdout、NbBench、FLAb、AB-antigen binding/SPR/AbBiBench/Chimera。初版部分AB选择包含valid，FLAb等为历史广范围，并非仅当前三类生成或统一test-only。没有把UpTCR当初版保护来源，也不能将旧T4库冒充当前全部benchmark14/held20联合保护。

初版删除台账只有shard/row/reasons，混合保护库不保存每行命中的具体测试记录/任务。因此这里只能确定规则和任务库来源，不能分配400683到单个benchmark或声称每行均泄漏结合标签。精确恢复该映射需要进一步依据簇与带任务来源的保护键进行只读关联，必要时重扫；此轮未做。annotation.missing_h3=13304、missing_cdr3a=15是注释缺口，不是已证实泄漏；budget5是兼容性；semantic dedup802是库内重复。

## 后续过滤：441532行

父版本是保留初版过滤结果的MHC更新版。paired=8433，只允许exact_ab（同一条测试记录的α/β CDR3均相同）；nonpaired=433099，沿用exact和edit80规则。共joint_exact10508（其中8433为ab，其余2075有ap/bp/bpmhc联合键）、仅component_exact148250、仅near282774。joint也不自动意味着相同MHC/表位/label：exact_ab不要求表位相同，exact_bp不要求MHC相同，所有标记均未证明标签一致。

非配对主要reason：edit80_cdr3b338660、edit80_peptide162595、exact_peptide110587、exact_epitope_in_antigen34768、exact_cdr3b12760。各reason重叠；edit80是归一化Levenshtein相似度至少80%的已有证据。ASD34768全因抗原包含TCR保护表位；不证明抗体目标重复。

| 保护任务cohort | 总删除行 | train | valid | 联合键exact | 仅组分exact | 仅near |
|---|---:|---:|---:|---:|---:|---:|
| tcr_representation_paper6/background_pool | 291022 | 289247 | 1775 | 4580 | 8068 | 278374 |
| uptcr/zero-shot/test | 259030 | 257405 | 1625 | 3123 | 47374 | 208533 |
| uptcr/few-shot/test | 191790 | 190828 | 962 | 2334 | 31545 | 157911 |
| uptcr/majority/test | 294664 | 292859 | 1805 | 6666 | 38433 | 249565 |
| tcr_clustering/tcrs | 258214 | 256632 | 1582 | 0 | 94291 | 163923 |
| tcr_representation/test | 141607 | 140912 | 695 | 2 | 30543 | 111062 |
| tcr_representation_paper6/target_binders | 152033 | 151351 | 682 | 0 | 16556 | 135477 |
| tcr_clustering_embed/tcrs | 235233 | 233879 | 1354 | 0 | 48565 | 186668 |
| nm2025/seen/cdr3b/AS/test.csv | 67031 | 66695 | 336 | 0 | 18328 | 48703 |
| nm2025/seen/cdr3b/HS/test.csv | 49952 | 49739 | 213 | 0 | 18328 | 31624 |
| nm2025/seen/cdr3b/PS/test.csv | 67926 | 67571 | 355 | 0 | 18328 | 49598 |
| nm2025/seen/others/AS/test.csv | 40268 | 40103 | 165 | 0 | 15076 | 25192 |
| tcr_binding/test | 75867 | 75530 | 337 | 0 | 22642 | 53225 |
| nm2025/unseen/cdr3b/AS/test.csv | 11220 | 11183 | 37 | 0 | 0 | 11220 |
| nm2025/unseen/cdr3b/HS/test.csv | 14162 | 14108 | 54 | 0 | 0 | 14162 |
| nm2025/unseen/cdr3b/PS/test.csv | 23381 | 23285 | 96 | 0 | 0 | 23381 |
| nm2025/unseen/others/AS/test.csv | 1759 | 1756 | 3 | 0 | 0 | 1759 |

对应：T1为tcr_binding与NM2025，UpTCR三套test，T2为clustering两集，T3为representation/test及paper6背景/target。paper6背景是实际评测候选池，非全部已知binder。exact组分包括仅peptide或抗原子串，不能读成完整受体/条件重复。后续17个cohort不含AB CDR/轻链或T4生成；其保护来自继承规则及独立T4精确beta-core+epitope范围，不能借用此表宣称有新增T4保护。

## 撤销过滤意味着什么

仅撤销后续过滤：恢复的441532行确有上述记录的重叠/近似命中，但只有部分是联合键重叠。相同受体可能损害未见受体解释；相同表位只对严格未见表位设定构成冲突，不能在seen-epitope任务自动叫答案泄漏；80%近似属于更严格家族泛化隔离。初版过滤不做则撤销已登记AB/TCR库的同源隔离，其中实际exact量及任务归属不能从简化台账量化。两轮都撤销的完整反事实不是此表相加：初版被删行从未进入后续台账，需额外匹配才能知道其后续命中。现有数据保护声明不能原样沿用到撤销后的数据。

没有建议无条件恢复全部数据，也未执行任何恢复。若后续要缩窄范围，应单列联合关系exact、目标序列exact、仅条件exact、近似四类并先确定任务泛化口径。
