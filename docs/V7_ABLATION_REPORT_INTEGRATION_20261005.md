# V7 本地与原池消融结果的汇总接入

<!-- CURRENT_FIXED_ABLATION_REVIEW_BEGIN -->

## 当前固定点：3500（核实于2026-10-07T06:52:19.404575+00:00）

3500步、各112000/120000条曝光；相对各自3250步，平均残基loss/准确率改善8/8、3/8；全目标残基准确率改善3/8，context全目标END准确率改善3/8。

18个同seed固定点预算、验证分母及3250曲线前缀核对通过。固定验证1024条，其中context37条、全目标END44个。两个seed、有限context、混合源码来源及历史退出未知的限制保持；不据中途指标认定统一最优架构或生成/结合能力。原训练继续至3750，下一3750；不改新版本V5数据/模型决定。

变化均相对各自3250步，准确率单位为百分点；CE越低越好。

| 变体 | seed | 平均AA CE | 全目标AA CE | 全目标AA准确率变化 | 全目标END准确率变化 | context全目标AA CE变化 |
|---|---:|---:|---:|---:|---:|---:|
| f010 | 43 | 0.805281 | 1.701762 | -0.4732 | -16.5742 | +0.014825 |
| f101 | 43 | 0.810285 | 1.709188 | -1.3557 | -11.2138 | +0.019114 |
| f110 | 42 | 0.806049 | 1.698793 | +0.6380 | +9.3038 | +0.002415 |
| f110 | 43 | 0.808150 | 1.702628 | -1.0638 | -18.7924 | +0.019772 |
| f111 | 42 | 0.805444 | 1.694561 | +1.0364 | +9.1805 | +0.008189 |
| f111 | 43 | 0.805329 | 1.701566 | -0.5955 | -14.1713 | +0.012845 |
| cv_segment_global | 42 | 0.818753 | 1.700392 | +0.8935 | +13.2471 | +0.004181 |
| cv_segment_global | 43 | 0.813439 | 1.702441 | -0.4169 | -11.0906 | +0.021307 |

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first3500_root_review.json；验收记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first3500_acceptance_receipt.json。仅读取保存指标，没有重跑模型或重绘固定500/3750图。

<!-- CURRENT_FIXED_ABLATION_REVIEW_END -->

## 2026-10-07T06:01:00.599765+00:00 原消融与新工作区输入布局保持分离

因工作区已另行准备V5对象布局，本轮只读核对原消融的源码绑定。八个已启动任务的run_provenance仍为A4，其50份记录源码与原冻结目录逐项SHA一致；四份pending票据与原catalog逐项相同，绑定的原dispatcher以隔离Python入口执行冻结runner。工作区layout.py与冻结layout.py的SHA确实不同，但这些已登记消融没有采用该工作区版本。原22格报告继续对应其原配置和来源；不能把这批结果标为新V5布局的训练结果。

证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/source_isolation_20261007T055915Z.json（SHA 8665060e3208b90daa07be8ab93d26b3915fe88196aa3120477d893974dfeb42，实际e85d0f rc0）。本项核对覆盖磁盘源码、既有运行来源及待执行票据，不是新的内存模块检查，也不替代pending任务的原启动门禁。没有加载模型、读取原始序列、改票据、重启/提交任务或新增checkpoint；当前仍等待同一会话3185的3250步指标。

## 2026-10-07T05:41:32.629026+00:00 八项3000保存指标已核实

参数化读取器c8602a rc0：八项各96000/120000曝光，18个同seed点预算/分母及2750曲线前缀通过。根报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first3000_root_review.json，实际回执 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first3000_acceptance_receipt.json；详细指标和限制见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新3000章节。完整epoch仍10/22（9成功退出、f000/42未知），原8运行/4排队；下一3250/最终3750。观察会话17579实际exit0已结束，原训练继续。固定图未重绘，无模型重跑、任务重启或测试checkpoint。

## 2026-10-07T05:08:27.070594+00:00 八项2750保存指标已核实

参数化读取器首次实际执行7dc862 rc0，八项各88000/120000曝光，18个同seed点预算/分母及2500曲线前缀通过。根报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first2750_root_review.json，回执 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first2750_acceptance_receipt.json；详细指标和限制见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新2750章节。完整epoch仍10/22，原8运行/4排队、下一3000；会话14812实际exit0已结束，原训练继续。固定图未重绘、无模型重跑或测试checkpoint。

## 2026-10-07T04:49:34.372226+00:00 保存指标读取器参数化，2750实际核对待结果齐全

新增 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/read_saved_fixed_point_v1.py（SHA 39a4d499e3e9482eaa40b66f0fea7916a72aabbfdf925698cba1e349a23ab339），显式接收上一点/目标点和上一份报告SHA，曝光数直接计算为step×32、完整epoch固定3750×32=120000，不再全局替换数字。复用原冻结collect读取器；保留原来源绑定、旧曲线前缀、同seed token预算、验证分母、有限数值和8/18格覆盖检查。输出排他创建；新增完整epoch或退出覆盖时先报错交由根端核实；3750最终图仍走原完整报告器。

命令行/语法检查实际522b3c rc0，只证明新读取器可调用，未声称2750指标核对通过，也没有重读已完成点来重复验收。准备回执 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/fixed_point_reader_preparation.json；待同一观察会话14812确认八项2750齐全后执行一次。没有改变训练源码、配置、数据或门禁，无模型重跑或测试checkpoint。

## 2026-10-07T04:34:13.262347+00:00 八项2500保存指标核对完成，继续原完整epoch

冻结报告器bffa43 rc0；新增八项各80000/120000条曝光，18个固定2500点预算/分母核对通过，2250曲线前缀保持。根报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first2500_root_review.json，实际回执 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first2500_acceptance_receipt.json。整体与context的残基/结束标记结果仍分化；详细数字及记录脚本修正唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新2500章节。两seed、context37条、混合来源限制保持；正式f011和下游解码不变。

完整epoch仍10/22（9成功退出、f000/42未知），原8运行/4排队，当前2525–2550，下一2750/最终3750。只读会话71161实际exit0已结束，原训练继续；固定500/3750图未重绘，无模型重跑、提交/重启或测试checkpoint。

## 2026-10-07T03:54:11.238678+00:00 八项2250保存指标核对完成，继续完整epoch

冻结报告器e8b37a rc0；新增八项各72000条曝光，18个现有固定2250点预算/分母核对通过，2000前缀保持。根报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first2250_root_review.json，实际回执 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first2250_acceptance_receipt.json。重建/结束标记及context分组的详细数字唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新2250章节。混合来源、两seed/context样本有限等限制保持，正式f011和下游解码不变；产物核查通过不等于全面质量改善。

完整epoch仍10/22（9成功退出、f000/42未知），原8运行/4排队；下一2500/最终3750，固定500/3750图无新增末点不重绘。原只读会话73314正常结束，训练继续，无模型重跑、任务提交/重启或测试checkpoint。

## 2026-10-07T03:17:48.065803+00:00 八项2000保存指标已核实，质量表现分化

冻结报告器eb610f rc0；新增八项各64000条曝光，18个现有固定2000点预算/分母核对通过，1750前缀保持。根报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first2000_root_review.json，实际回执 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first2000_acceptance_receipt.json。整体与context的重建/结束标记详细数字唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新2000章节；不能将产物验收当质量全面改善。混合来源、两seed及context37条限制保持，正式f011和下游解码不变。

完整epoch仍10/22（9成功退出、f000/42未知），原8运行/4排队；下一2250/最终3750。固定500/3750图无新增末点不重绘；原只读会话31864正常结束，训练继续，无模型重跑、任务提交/重启或测试checkpoint。

## 2026-10-07T02:42:21.260230+00:00 八项1750保存指标已核实，继续原完整epoch

冻结报告器32c43f rc0；新增八项各56000条曝光，18个现有固定1750点的预算/分母核对通过，1500前缀保持。根报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first1750_root_review.json，实际回执 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first1750_acceptance_receipt.json。平均AA改善及全目标/context/END取舍的数字唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新1750章节。混合来源、两seed及context37条限制保留，正式f011和下游解码不变。

完整epoch仍10/22（9成功退出、f000/42未知），原8运行/4排队；下一2000/最终3750，固定500/3750图无新增末点不重绘。原只读会话33333正常结束，训练继续，无模型重跑、提交/重启或测试checkpoint。

## 2026-10-07T02:08:15.750729+00:00 1500验证及18/22同曝光数字对比已完成

实际625c01 rc0，新增八项各48000/120000曝光；18个固定1500点的预算/分母与变体标签源码绑定通过。根报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first1500_root_review.json，逐seedCSV /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/fixed1500_architecture_comparison.csv，实际回执 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first1500_acceptance_receipt.json。9种架构各两seed比较及context/END取舍的数字唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新1500章节；混合来源、两seed/context37条限制及4个缺失控制格均保留。

固定500/3750图未重画，完整epoch仍10/22（9成功退出、f000/42未知）、8运行/4排队。正式f011和原解码保持，下一1750/最终3750；不把产物核查通过当质量全面通过，不因缺失历史exit或混合来源重复已完成实验。有限观察结束、原worker继续，无模型重跑、任务提交/重启或测试checkpoint。

## 2026-10-07T01:29:12.607715+00:00 八项1250验证已验收，继续完整epoch

冻结报告器4ea40e rc0；八项各40000/120000曝光、固定1024验证及18个现有同seed预算/分母通过，原1000前缀未变。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first1250_root_review.json、实际回执 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first1250_acceptance_receipt.json。总体/context平均AA改善与全遮蔽/END分化的详细数据唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新1250章节；context只有37条/全遮蔽44个END，不扩大功能或显著性结论。

完整epoch仍10/22（9成功退出、f000/42未知），8运行/4排队，500和3750固定图未重画；下一1500/最终3750。正式f011和解码保持，混合来源限制保留。只读观察已结束，原worker继续；无模型重跑、提交/重启或测试checkpoint。

## 2026-10-07T00:53:58.799641+00:00 八项1000固定验证已核实，原完整epoch继续

报告器994bef rc0；根产物 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first1000_root_review.json，实际回执/现场状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first1000_acceptance_receipt.json。新增八项每项32000条曝光、1024验证，18个现有1000点同seed预算/分母核对通过。详细指标及升降唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新1000章节。完整epoch仍10/22（9成功退出、f000/42退出未知），8运行/4排队；500和3750图没有新增末点，不重绘。下一1250和最终3750，正式f011/下游解码不改，保留混合来源及两seed限制。只读观察在1000齐全后结束，原worker继续；无模型重跑、测试checkpoint、任务重启或提交。

## 2026-10-07T00:20:18.580374+00:00 新增八项750验证已验收

冻结报告器9c49cf rc0，证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first750_root_review.json、实际回执 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first750_acceptance_receipt.json。每项24000/120000曝光、1024验证及同seed预算已核实。平均AA改善与END回落同时记录在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md；不据早期点更换正式架构，下一固定1000/最终3750。500图18/22和3750图10/22未重画；8运行/4排队，未新增完整epoch或真实退出。

## 2026-10-06T23:42:54.045644+00:00 固定500对比图达到18/22

固定500对比图更新为18/22、仍PARTIAL：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_step500_eight_new/step500_architecture.png；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_step500_eight_new/root_review.json。平均遮蔽与全遮蔽AA CE呈不同取舍，500不是最终3750，来源cohort混合、两个seed不足以给统计显著性；维持正式f011与既定完整epoch计划，不依据早期指标换架构。具体对比产物仅引用以上CSV/JSON。

8项仍运行、4项排队，实际终态尚未新增；原22格完整epoch仍10产物，9成功退出和f000/42退出未知边界保持。已完成检查不重做，后续仍3750完整epoch。实时状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_step500_eight_new/live_status.json。

## 2026-10-06T09:17:22.685086+00:00 f010完成750步验证：平均AA改善，END优势缩小且context回退

正式训练最新6575/25042、完整验证6000；本地f010已验收750步，24000/120000曝光，当前训练750/3750。平均AA改善，全遮蔽CE/END仍有回退，不更换正式f011。详情 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LOCAL_ABLATION_PREPARATION_20261005.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation750_comparison_20261006.json。

完整epoch仍9格产物（8格成功退出、f000/42退出码未知）、1运行、12未启动；固定500图10/22、3750图9/22均不重画。下一本地1000/正式7000，既有安全停线和门禁保持。

## 2026-10-06T08:12:10.109919+00:00 f010完成500步固定点：平均重建与END改善，全遮蔽AA回退

正式八卡f011继续运行，最新训练日志6225/25042步，最近已核实验证/保存6000，当前段目标10016。原监督PID3977587/start2446951610与8rank新鲜心跳正常，无abort；DDP+NCCL、micro32/GA1/global256不变。

本地f010/42已完成500步验证，16000/120000条训练曝光；当前训练500/3750步，原PID80642/start2452750909持续匹配，训练退出码仍未知。固定1024条验证（context37、无context987），曝光/token/学习率与本地f000/42同500步一致；170计数字段/40个mask_ratio匹配，复用已验收250点的来源和计数证据。

同500步：四档平均AA CE1.185176→1.106683、准确率63.2778%→66.2457%；全遮蔽AA CE1.793335→1.807474、准确率40.2420%→39.7036%回退。全遮蔽END token准确率48.0591%→58.7800%改善。含context37条的平均/全遮蔽AA CE均下降，但全遮蔽AA准确率45.1168%→44.3125%。这仍是单seed/早期重建结果，不能认定生成与理解全面收益，保持正式f011及既定3750完整epoch/第二seed计划。

固定500步图已更新至10/22、PARTIAL；完整3750图仍9/22（8格成功退出已核实、f000/42退出码未知），另1格运行、12格未启动。PNG已目视检查，CSV/JSON/SVG/PDF保留。新图 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_step500_f010/step500_architecture.png；对照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation500_comparison_20261006.json（SHA45c0985b46fd2e11477bee20c581ffc98d604ce613c417a2a035c69d4745aa97）；根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation500_root_review_20261006.json（SHAb89bc9b54b12db6ed1537e9dee33a0ce753ddaa5fc24365abfd80ca31ba56007）。

观察48013在实际500验证出现后结束，14次观察、工具2e2859实际rc0；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_bounded_watch_20261006T075807Z.jsonl（SHA26c7d8f95cb0f4e182f86a006167726e9aa65ef216a85805ce840ea66152a3ca）。比较工具0e12f1、绘图工具cfffae均实际rc0；没有新模型运行、平台提交、重启或重复旧点评估。原正式/本地训练与原监督继续，当前没有运行中的有限观察会话。自有测试目录无权重/tmp可删，正式恢复状态保留。下一本地750、正式7000；完整Goal仍active，本轮progress。序列注释停线、独立CUDA暂停、远端403不重试和限定数据/来源门禁保持。

## 2026-10-06T07:04:56.525907+00:00 f010首个250步验证：部分遮蔽与END改善，全遮蔽AA持平

2026-10-06T07:04:56.525907+00:00：f010 seed42已完成250步验证，8000/120000条训练曝光，固定1024条验证（context37/无context987）。与本地f000/42同250步比较，来源记录仅variant不同，数据、初始化摘要、训练配方和源码记录一致；曝光/token/学习率、170个计数字段与40个mask_ratio逐项相同。四档平均AA CE1.538930→1.425135、准确率51.5941%→55.9645%；全遮蔽AA CE1.799719→1.799752、准确率41.2489%→41.0792%，没有同步改善。全遮蔽END准确率23.5983%→29.2052%。这是同seed早期对照，完整epoch尚未完成。

含context37条的全遮蔽AA CE2.017438→1.973137、AA准确率44.7338%→45.5381%、END准确率31.8182%→38.6364%；分来源并非普遍受益，tcr_papers平均/全遮蔽AA CE均回退。tcr_native仅2条/4个END，trait仅1条/1个END，不放大其百分比波动。已有正例关系准确率两组均1，不代表正负binding或生物功能。保持正式f011初步版本及f010的3750完整epoch计划，等待500/3750固定点和第二seed，不据单点换架构。

实际对照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation250_comparison_20261006.json（SHA1e2626c6567f05abf59d4f8914f847a2e7b440635233fe9c5b8fad0fcfd1a34b），根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation250_root_review_20261006.json（SHA8b7544a348083cc11f5fdbd47859dd7f6887f8b7643c670719de95d9fb1eceba）；原读取工具1ad374实际rc0、1.308秒，没有模型重跑。15分钟观察86292与随后等待验证的50137均实际rc0结束，当前无运行中的有限观察会话；原训练和监督仍在运行。正式最新5825/25042步、loss=0.4655，最近完整验证/保存5016；本地最新训练275步，最近完整验证250。原PID/start/argv与8rank新鲜心跳再次核实，无abort或训练退出码。

完整22格仍9格完整epoch产物（8格实际成功退出已核实、f000/42退出码未知）、1运行、12未启动。固定3750图仍9/22、PARTIAL，本轮不重画500/3750图。没有测试权重/tmp、额外模型运行、新平台任务或重启；正式恢复状态保留。原序列注释停线、独立CUDA暂停、远端403不重试及数据/来源门禁保持。完整Goal仍active；下一本地验证500、正式验证6000。

## 2026-10-06T06:00:45.177529+00:00 f010首25步已实际完成，启动观察结束

f010/42已实际更新25步/800条，loss=2.855475，峰值已分配显存16.9678GiB；初始1024条验证与新报告接入通过，但250步验证尚未到达。六分钟9次观察结束、真实进程仍在，证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_launch_package_preparation_20261006/startup_acceptance.json。正式训练继续5475/25042，最近完整保存5016。完整epoch产物仍9/22（其中1格退出码未知），1运行/12未启动；固定末点图不重画。无新增测试权重、平台任务或重启。

## 2026-10-06T05:52:50.913570+00:00 f010 seed42已通过真实准入并单次启动

2026-10-06T05:52:50.913570+00:00：正式八卡f011继续运行，最新核实训练日志为5425/25042步、loss=0.4741，最近完整验证/保存5016，当前段目标10016、下次常规验证6000。原监督PID3977587/start2446951610与八rank新鲜心跳已核实，无abort；DDP+NCCL、micro32/GA1/global256保持。

f010 seed42于2026-10-06T05:50:34Z真实单次启动，持共用格子/设备锁的guard.main二次检查通过，PID80642/start2452750909已exec原冻结训练CLI。当前处于启动初始化，run_provenance已出现、首个训练loss尚未观察到；wrapper rc0只证明tmux启动，训练退出码仍未知。原120000条、3750步完整epoch、micro32/GA1/eval250/eval4、metrics-only不变。此项使用已有授权本地共享GPU，未申请、抢占或重启八卡正式资源，不作受控测速。

完整22格：9格完整epoch产物（8格成功退出已核实、f000/42退出码未知），f010/42运行中，另12格未启动。原3750步图仍9/22、PARTIAL，未重画或声称新末点完成。实际启动记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_launch_package_preparation_20261006/actual_launch_receipt_v2.json；根观察 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_launch_package_preparation_20261006/actual_launch_root_observation.json；新显式选择文件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_selection_v2.json（SHA9a60ebd56551c71c16faf86dd5efd4c44448e71fe3a44de31c86221a089dc0fc），只把f010/42从pending改为该真实local attempt，其他21格保持，旧选择/图保留。

后续直接观察同一tmux任务v7_local_f010_seed42_attempt1及原正式监督，禁止重放本启动包。原训练PID/start/argv与真实guard/header/started绑定已经核实；结束后读取原runner保存的exit_code/done/failed再形成真实exit v2，不把PID消失或COMPLETE等同rc0。元数据发布工具6254d9在三个原件已写入后，仅末尾打印误用native_checkpoint_step键报错；0be65d只读确认正确字段latest_completed_checkpoint_step=5016，未重发启动或改造结果。

本轮progress（实际接续一项原计划消融）；上一问答仅状态答复属no_progress。完整Goal仍active，正式25042及22格最终图尚未完成。序列注释停线、独立CUDA专项暂停、403不重试、原审核及来源门禁保持；未知GPU进程未干预，受控profile仍UNRUN。无测试checkpoint生成，正式恢复状态保留。

## 2026-10-06T05:43:41.648750+00:00 本地f000完整epoch验收与9/22固定末点对照

本地f000 seed42已完成3750/3750步、120000条唯一记录、epoch1/position0。最终输入46043658、eligible18432158、selected AA/END9205174；末次更新使用LR1.8520555061796174e-11，峰值已分配显存18.095971GiB。COMPLETE绑定summary SHA b834155d907317c8fc0b61573fbd335ece2033eda980e6a52067aad1f882d86c与原provenance df6c14a09b0f8cd9074013710deb157ffe97530f31c03cefcc947e500e9060a8，原PID已消失、无FAILURE或自有checkpoint/tmp。实际训练退出码未留存，不能写rc0。读取器在/proc消失时停止的异常与训练失败分开记录。

| 本地指标 | 3500步 | 3750步完整末点 |
|---|---:|---:|
| 四档平均AA CE | 0.826046 | 0.824812 |
| 四档平均AA准确率 | 74.9333% | 74.9618% |
| 全遮蔽AA CE | 1.696441 | 1.693748 |
| 全遮蔽AA准确率 | 43.5153% | 43.5146% |
| 四档平均END准确率 | 81.8718% | 79.5137% |
| 全遮蔽END CE | 1.505086 | 1.595907 |
| 全遮蔽END准确率 | 42.9452% | 33.2717% |

含context的37条验证记录平均AA CE1.219970→1.220773、全遮蔽AA CE1.853081→1.857462、全遮蔽END准确率50%→52.2727%；不含context的987条全遮蔽END准确率42.7486%→32.7422%。已有正例关系准确率1.0不等于正负binding判别或功能能力。新末点170计数/40mask_ratio和1024验证记录保持，只新增3750指标核对；工具5bc4ff实际rc0，无模型重跑。末点原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation3750_trend_20261006.json（SHA0b0254d9141c850a387bfa4226ef4346ae9d0bbf2cafd9d26947f8eca9554288）。

| 条件 | seed | 四档平均AA CE | 全遮蔽AA CE | 全遮蔽END准确率 |
|---|---:|---:|---:|---:|
| f000 | 42 | 0.824812 | 1.693748 | 33.2717% |
| f000 | 43 | 0.821038 | 1.697685 | 42.1442% |
| f001 | 42 | 0.808769 | 1.694904 | 27.2335% |
| f001 | 43 | 0.808351 | 1.701822 | 33.3949% |
| f011 | 42 | 0.802201 | 1.693988 | 30.5607% |
| f011 | 43 | 0.803388 | 1.701324 | 33.8262% |
| f100 | 42 | 0.833389 | 1.697030 | 32.6556% |
| f100 | 43 | 0.830072 | 1.696440 | 31.3001% |
| f101 | 42 | 0.808952 | 1.695659 | 33.4566% |

f011相对f000的平均AA CE在seed42/43分别低0.022612/0.017650，平均AA准确率高0.6812/0.3860个百分点；但全遮蔽AA CE反而高0.000240/0.003639，全遮蔽END准确率低2.7110/8.3179个百分点。seed42比较跨local_metrics_only_v1与legacy_large_v1执行来源，seed43为同一旧来源；仅两seed、不做显著性或孤立因果结论。维持正式f011初步版本，优先补f010两个seed拆解R24固定残差缩放的独立作用，继续分别观察部分遮蔽重建和全遮蔽指标；不能宣布理解与生成两端都提高。

新图 [3750步完整epoch局部图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_epoch1/step3750_architecture.png) 覆盖9/22，仍PARTIAL；9格有完整产物，8格真实成功退出已核实，当前本地1格退出码未知，13格待跑。原选择/来源/训练预算不变，未以3500较好END点替代3750。真实渲染45156d退出0、warnings空，CSV44条设计末点行/7200曲线行/49配对行；PNG已人工查看，双mask面板、missing标签、来源/seed图例可见。本地退出码未知点明确淡化。SVG/PDF/CSV/JSON及根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_epoch1/root_review.json（SHAdfb0191a887268f86297ed0e58251efb97838bfd7a122563d312b450f2b3b613）；旧图和必要证据保留。

完整验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/completion3750_root_review_20261006.json（SHA1c72315c75e9d82434c283a5b12f178381384e999b1c8a61b19b1ac62688a46c）。GPU仍有未知归属PID66091、1680MiB、利用率61%观测，不能用此次共享环境当受控速度对照，profile仍UNRUN。既定本地质量消融允许沿原共享锁/资源合同准备f010/42，不新增全卡idle条件；真实门禁通过前不启动，当前接续包准备中。原八卡继续5400/25042，最近验证/保存5016；原Goal完整范围、安全停线和审核门禁保持。

## 2026-10-05T18:33:58.003212+00:00 guard写入与报告读取接口检查完成

未来本地消融的私有guard、共用格子/设备锁和报告器v2接入已完成本轮工程检查：子agent7项人工检查（5.984秒、rc0），追加实际临时receipt→reader接口1项（1.456秒、rc0）；根端报告器3项定向检查（1.411秒、rc0）。只有锁探针进行了真实CPU exec，训练exec为mock，无新模型/GPU测试。根端源码/原metadata字段/固定绑定和当前4个正式公共发布SHA已复核；没有重复运行既有pool或模型验收。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/local_guard_root_review_20261005.json（SHA932befc2eb38d317595b81ea4680fba06913f8fdf175c78158ffca5ae3220cb9）；报告器检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/guarded_reporter_tests_20261005.json。旧图使用的源码已归档 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/reporter_source_v1，固定500/3750局部图不重绘。人工fixture已清理，无测试权重，正式恢复文件保留。

当前实现源码SHA为cbc6a710d5f030bb591329307e7ba5d6324e9e0503da489f2ae4174312af195d。v2 exit须重复launch的admission、ledger、guard_source、guard_argv和guard_execution绑定，实际训练退出码仍从原tmux runner的PIPESTATUS记录读取；pre-exec记录本身不代表成功。现有真实500/3750报告保持原结果，当前尚无真实guarded运行。


## 2026-10-05 18:29 UTC 未来 guarded tmux v2 记录支持

报告器新增对 `v7.local.tmux_launch.v2` / `v7.local.tmux_exit.v2` 的读取：检查启动前 admission、共用格子认领、guard 源码与实际调用参数，以及实际 pre-exec receipt；后者不能代替训练成功。只有原生完整 epoch 产物与真实 `exit_code` 同时满足，才记完整成功；没有退出证据仍为 `completed_exit_unconfirmed`。读取报告不执行 guard，也不重跑启动检查或模型。

实际定向检查 3 项通过（1.411 秒、rc0）：新 v2 的完成/退出语义、替换凭据拒绝、受影响的原 v1 tmux 退出检查。仅临时人工元数据，未导入 ML 模块，临时 fixture 已清理。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/guarded_reporter_tests_20261005.json。本地 guard 自身的实现检查仍在进行，未启动新训练或创建真实 admission/ledger；不能据本段称真实接续已完成。

原 500/3750 图及其原始 SHA 保持，未重绘。此次修改前、实际用于这些图的报告器和人工测试源码已保存至 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/reporter_source_v1，归档说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/reporter_source_v1/archive.json。下方 15:55 的实现 SHA 对应这份旧来源，不代表当前修改后源码。

更新于 2026-10-05T15:55:59.001982+00:00。状态：独立报告器已实现，5项新增元数据测试通过；500/3750实际局部报告已生成并检查。正式八卡与本地f000继续运行。

[500步局部图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_v1/step500_architecture.png)（9/22），[3750步完整epoch局部图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_v1/step3750_architecture.png)（8/22）。两图均明确PARTIAL，分别展示mask_mean和mask100，每seed一个点；13项尚未启动，本地f000仍在跑。SVG/PDF、CSV/JSON及实际渲染凭据均在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_v1；根端复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_v1/root_review.json。

实现源码SHAa6e2c7f26e0e17aff1dd66149707b93c2e9f269f255c18d2ff7c7f941a60e6e0；当前coverage_complete=false、all_exit_verified=false、source_homogeneous=false，final/legacy_final=false。本地500结果只进入500图，3750图不降档。旧成功记录和旧失败记录均未修改。5项测试仅临时人工元数据，首轮fixture绑定修正后通过，临时图片已清理；真实图由root生成，退出0、无warnings。

下方“尚未实现”表述均为先前准备阶段的历史记录；当前结果以上方实际渲染及根端复核为准。

真实选择清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_selection_v1.json（SHA18703a2541003c807103703e206129d3a49c1907334fe1c6bb97fb04c058e91e），显式选择8个旧成功任务、当前本地f000 seed42和13个pending格；不把旧失败格冒充当前实验。它绑定各自来源清单和共同batch方案；本地exit_evidence=null，未补造退出凭据。下文15:22时的“尚未实现”记录为历史准备状态，实际可用与渲染结果以后续验收为准。

现有比较器 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_large_batch_ablation.py 的collect只读取一个pool、严格比较包含source_sha256的共同身份，终态依赖原生COMPLETE和真实pool succeeded/rc0。保留该行为，不能通过伪造pool结果或删除源码标识把本地结果接入。原账本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/jobs.json 保持不变。

原同格与本地run_provenance的顶层差异为checkpoint_saving和source_sha256；provenance记录的源码字典有5处差异；本地冻结文件为：

- /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/source_snapshot_v1/dllm/pipelines/immune_llada/research/model.py
- /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/source_snapshot_v1/scripts/research/result_retention.py
- /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/source_snapshot_v1/scripts/research/run_v7_architecture_pilot.py
- /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/source_snapshot_v1/scripts/research/run_v7_large_batch_ablation.py
- /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/source_snapshot_v1/scripts/run_immune_research.py

这里的差异数不同于冻结覆盖文件数。原始250步比较 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation250_comparison_20261005.json 记录AA CE及terminal准确率差异；原因尚未确认，不称逐位复现或将跨来源差值全部归为架构收益。

后续采用独立的混合来源汇总入口，显式选择每格的实际attempt，保留旧失败记录。逐run复用selected_plan、identity、read_curve、validate_points；核对各自冻结来源、共同数据/batch/mask/schedule、同seed初始化及曝光计数，再复用derive等指标计算。CSV及图注保留transport、attempt、来源版本和完成证据。旧draw的FINAL标题及时间对比面板不能原样用于跨环境报告；本地wall-time不作为受控架构加速证据。该独立报告器目前尚未实现或运行，不把方案准备当最终图已完成。

完成状态分开呈现：coverage_complete指22格完整3750步/120000条末点齐全；all_exit_verified指实际退出凭据齐全；source_homogeneous指执行来源一致。三者不互相替代。原生COMPLETE只证明程序完成产物及其summary/provenance绑定；不据此或PID消失伪称rc0。本地当前Popen启动者已退出，现有launch_receipt没有未来退出码凭据；若将来只有原生完成产物，标completed_exit_unconfirmed，不冒充旧严格报告的final。当前仍在训练，无需侵入或重启该进程，也不因此自动重复任何已完成实验。

后续13项本地任务复用既有 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_in_tmux.sh，它实际保存PIPESTATUS[0]为exit_code，并写done/failed。命令准备 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/managed_local_launch_plan_v1.json（SHA7413c5fc9783ea2aa81dc37ebbdb5ba25588e09f76cc88190039e1a9de26970e）。基础训练argv、冻结来源、3750步和metrics-only保持；每项使用全新task/log/output身份并绑定真实启动记录，启动前仍核实原数据/来源条件。该runner会清理同名旧哨兵，所以禁止复用已有log/session身份。当前f000不转移、不重启；13项均未启动，下一顺序仍f010 seed42、seed43。

该复核由root和现有只读子agent共同完成，没有执行新模型/GPU/算子测试、数据处理、平台提交或取消。停止的序列注释及独立CUDA暂停保持。

2026-10-05T15:25:26.929022+00:00 命令元数据复核：根端确认13/13底层argv与原计划一致，基础计划/runner SHA匹配；既有只读子agent独立核对环境及当时新task/log/output身份。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/managed_local_launch_plan_review_v1.json。实际启动前仍重查身份不存在；必须读取训练的exit_code与done/failed，外层启动成功或tmux pane返回0均不代替训练退出码。没有执行训练或新GPU测试。
