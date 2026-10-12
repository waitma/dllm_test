# V7 完整消融矩阵接续清单

## 2026-10-06T22:28:30.597931+00:00 原八卡接续12项缺失消融，已有10项不重跑

真实两轮及九项CDR完成、训练/生成监督实际退出后，12个缺失格子已单次发布，当前8运行/4排队/0结果；实际GPU更新尚待验收。清单及资源门禁 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/root_publication_review.json，唯一票据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/enqueue_receipt.json，最终图选择 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/mixed_report_selection_with_local_exit_v1.json。每格120000条/3750步完整epoch，metrics-only；原数据/源码/共享锁和不自动重试保持。完整图仍10/22 PARTIAL，f000/42退出码未知、f010/42真实rc0已接入；不得重发批次或重新运行已完整格子。

## 2026-10-06T22:17:06.607485+00:00 固定3750覆盖新增f010/42：10/22，仍PARTIAL

f010/42完成3750/120000条且实际rc0；新增后10格完整产物、9格成功退出/1格f000/42未知，12格未启动、0格本地运行。当前选择 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_terminal_evidence_20261006/mixed_report_selection_v3.json；固定3750图 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_epoch1_f010/step3750_architecture.png；根核查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_terminal_evidence_20261006/root_full_epoch_review.json。固定500图10/22保持历史版本未重画。未发生新任务启动或其他格子重跑；具体数值/升降只见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LOCAL_ABLATION_PREPARATION_20261005.md。原完整22格与全部生成评测仍未完成。



## 2026-10-06T21:13:43.071416+00:00 本地f010/42新增3500验证

3500同112000条曝光对照已核查；平均AA/END及整体全遮蔽END改善，但context END准确率下降，完整数表与边界见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LOCAL_ABLATION_PREPARATION_20261005.md 本节点，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation3500_comparison_20261006.json。下一3750；22格9完整产物/1运行/12未启动，固定图仍PARTIAL。正式45050后停止，独立恢复包准备中，尚未运行生成或后续缺格。

## 2026-10-06T20:05:14.636358+00:00 本地f010/42新增3250步验证

训练3250/3750，3250同104000条曝光对照已核查；平均与context AA改善，但整体全遮蔽AA/END仍较差，完整表唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LOCAL_ABLATION_PREPARATION_20261005.md 本时间节点。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation3250_comparison_20261006.json（SHA8a0e6b0b3cdbbc581a251d308ea351a5d6db58212ee15e8bb88bd057e8a0476c，4a7c11 rc0）。下一3500；完整epoch格数和固定图保持。

## 2026-10-06T19:03:30.084354+00:00 本地f010/42新增3000步验证

训练3000/3750、3000同曝光对照已核查；平均AA改善但全遮蔽AA CE和END退步，完整表唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LOCAL_ABLATION_PREPARATION_20261005.md 本时间节点。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation3000_comparison_20261006.json（SHA359b69b22cea92f09e3c5c523a1436bc6f59ca08bcb07a22590e96e7aeb6da4a，776e1f rc0）。下一3250；完整epoch格数和固定图保持。


## 2026-10-06T18:02:18.596521+00:00 本地f010/42新增2750步验证

最新训练2750/3750、2750验证已核查；AA优于同曝光基线而整体END更差，完整指标唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LOCAL_ABLATION_PREPARATION_20261005.md 本时间节点。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation2750_comparison_20261006.json（SHA36a31e531f0da9e348664017127d890513fa48ae51d50dcb0cc8b6bdc911b79a）。下一3000；完整epoch格数和500/3750固定图不变。


## 2026-10-06T14:53:38.864998+00:00 本地f010/42新增2000步同曝光对照

新点读取原有曲线并完成聚合核查，未运行额外模型。同seed42、64000/120000条曝光、相同token数量/LR；170项计数和40项mask比例完全一致，固定1024条验证含context37条。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation2000_comparison_20261006.json，SHAeebd520a4dabfcb6c9fa1494b3dc0e68c1ccc3a89c9582c44e80d5c1c3525b4f；工具a28c55实际rc0。

| 指标 | f000基线 | f010固定残差0.1 |
|---|---:|---:|
| 四档平均AA CE | 0.886307 | 0.863236 |
| 四档平均AA准确率 | 73.2962% | 74.0273% |
| 全遮蔽AA CE | 1.761686 | 1.775030 |
| 全遮蔽AA准确率 | 42.2681% | 41.1974% |
| 平均END准确率 | 85.4062% | 83.0622% |
| 全遮蔽END准确率 | 55.5761% | 45.7178% |

平均重建有改善，但全遮蔽AA与END均回退，不能判固定残差全面优于基线。context平均AA CE1.256698→1.187280改善、全遮蔽CE1.854908→1.889831变差；context全遮蔽END由23/44降至22/44。仅已有关系正例的准确率均1，不能称正负binding能力。保持正式f011与原22格/完整3750/第二seed计划，不因这一点改正式配置；下一本地2250。当前9完整产物、1运行、12未启动，固定500/3750图不重画。


## 2026-10-06T12:50:03.929367+00:00 本地f010/42新增1500步验证元数据已核对

读取现有原件，未重跑模型。同seed42、48000/120000条曝光，170计数字段与40个mask_ratio完全对齐；固定1024条验证，其中context37条。平均AA CE 0.931826→0.899168改善；全遮蔽AA CE 1.782721→1.784553略差，准确率40.9893%→40.4453%。平均END准确率78.3659%→77.2258%，全遮蔽END 34.0111%→33.7646%，均回退。

context全遮蔽AA CE 1.927439→1.880236改善，但END准确率38.6364%→27.2727%（17→12/44），本点不能称固定残差全面更好；已有正例关系准确率均1不证明正负binding或功能质量。保持正式f011、完整3750步和第二seed判断计划；下一本地1750。此本地任务仍用原冻结实现，与新版正式where训练的吞吐分开。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation1500_comparison_20261006.json；根记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation1500_root_review_20261006.json。22格仍9完整epoch产物、1运行、12未启动；图500点10/22、3750点9/22均PARTIAL，本次未重画、未造新退出回执或测试ckpt。

## 2026-10-06T11:33:52.924409+00:00 f010完成1250步验证：重建改善，END仍有分化

同seed42、同来源；40000/120000条曝光，input15391562、eligible6157115、selected3096993，LR7.747811222909726e-5。170计数字段与40个mask_ratio一致，现有metadata validator核实新点的四档分子/分母、均值及来源/context聚合。固定1024条验证，context37条。

| 指标 | f000基线 | f010固定残差0.1 |
|---|---:|---:|
| 平均AA CE | 0.961122 | 0.918567 |
| 平均AA准确率(%) | 70.930929 | 72.489807 |
| 全遮蔽AA CE | 1.787297 | 1.776067 |
| 全遮蔽AA准确率(%) | 41.235131 | 41.497486 |
| 平均END准确率(%) | 78.917998 | 79.654119 |
| 全遮蔽END CE | 1.485902 | 1.467244 |
| 全遮蔽END准确率(%) | 36.229205 | 34.627234 |

平均及全遮蔽AA在本点均改善；全遮蔽END CE略降，但准确率36.2292%→34.6272%，并非全面受益。context全遮蔽AA CE1.980387→1.987997略差、准确率42.9337%→43.6997%；context全遮蔽END准确率47.7273%→52.2727%（21→23/44）。已有正例关系准确率均1，不能推出正负binding或功能质量。

f010是R24固定残差0.1单因素；1000与1250点的END表现波动。维持正式f011及3750完整epoch/第二seed计划，下一本地1500。完整22格仍9格完整epoch产物（8格成功退出、f000/42退出码未知）、1运行、12未启动；500固定图10/22、3750图9/22，均PARTIAL，未重画。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation1250_comparison_20261006.json（SHAd0fa3949be9ebb665b7291b9db1a1524bd12d2735cbcbcaf4d7837f0a30b22fc）；根核实 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation1250_root_review_20261006.json。本监督只解析新指标，未重复运行模型。


## 2026-10-06T10:27:22.350189+00:00 f010完成1000步验证：平均AA改善，END指标回退

同seed42、同本地来源，与f000同1000步比较；32000/120000条曝光，170个计数字段及40个mask_ratio一致，固定验证1024条（context37条）。平均AA CE1.008167→0.969030、准确率69.2347%→70.9678%；全遮蔽AA CE1.798585→1.835593变差、准确率40.3622%→40.4934%基本持平。平均END CE0.778868→0.924024、准确率70.9231%→69.9270%；全遮蔽END CE2.082055→2.592752、准确率18.7307%→12.1380%，本点明显回退。context全遮蔽END准确率50%→43.1818%（22→19/44）。已有正例关系准确率均为1，不能推断正负判别或功能质量。

f010是R24固定残差0.1单因素，不是R26；结果进一步表明不能凭平均重建收益宣布全面提升。原正式组合f011与3750完整epoch计划保持；本地下一验证1250。固定500/3750图不替换为1000点，完整22格仍9完整epoch产物（其中1格退出码未知）、1运行、12未启动。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation1000_comparison_20261006.json；根核实 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation1000_root_review_20261006.json。本轮只读取新指标、未重跑模型。发布时正式日志6975/25042，读到的完整checkpoint目录6000；最近根验收仍6000，新checkpoint若出现需单独验收。本地训练1025/3750、验证1000。

## 2026-10-06T09:17:22.685086+00:00 f010完成750步验证：平均AA改善，END优势缩小且context回退

f010/42新增750点（24000/120000曝光），尚未完成3750。对照与结果见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation750_root_review_20261006.json；完整epoch格数和固定500/3750图保持，下一本地1000。

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

本地f000/42完整3750步/120000条epoch已验收，退出码未知如实标注；3750固定末点图已更新为9/22，13项未启动。f011平均AA CE较低，但全遮蔽AA/END没有同步优于基线，不能作理解/生成全面收益结论。正式八卡继续5400/25042，配置不变。f010/42真实接续包准备中，尚未启动；受控测速仍因非独占未执行。详情 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md，根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/completion3750_root_review_20261006.json，新图 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_epoch1/step3750_architecture.png。

核实时间：2026-10-06T20:05:14.636358+00:00。9格完整epoch产物（8退出成功、f000/42退出码未知）；f010/42训练3250、验证3250，另12格未启动。500图10/22，3750图9/22。

正式训练最近已核查验证/保存42000步，最新计时42363/50082；正式与pilot配方独立，不纳入此消融排名。


[最新500步局部图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_step500_f010/step500_architecture.png)（10/22，PARTIAL）；[3750步完整epoch局部图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_epoch1/step3750_architecture.png)（9/22，PARTIAL）。旧快照保留；图中固定500点不变，f010已有3250中途评估，完整epoch及实际退出尚未完成。

| variant | seed42 | seed43 |
|---|---|---|
| f000 | 完整3750/120k产物已验收；退出码未知 | 完整完成 |
| f010 | 本地attempt1运行中（R24固定残差0.1），训练3250/3750；3250验证已核实，PID80642 | 待补 |
| f001 | 完整完成 | 完整完成 |
| f011 | 完整完成 | 完整完成 |
| f100 | 完整完成 | 完整完成 |
| f101 | 完整完成 | 待补 |
| f110 | 待补 | 待补 |
| f111 | 待补 | 待补 |
| cv_segment_global | 待补 | 待补 |
| cv_late_fusion | 待补 | 待补 |
| fixed_gate_control | 待补 | 待补 |

当前f010 seed42已运行，下一项优先f010 seed43；先补齐R20/R24/R26/组合的两seed匹配对照。f000 seed43已有完整结果，不重复运行。其余格保持原变体与seed范围。

每项3750步/120000条，micro32/GA1/global32/eval4，默认metrics-only。新本地冻结来源支持全部11个variant。原本地PID3970535已消失；f010/42真实接续已启动，PID80642。八卡正式联合训练由独立监督器占用，本清单不会抢占或重复入队。

历史13项候选命令（已包含本轮运行中的f010/42，勿照原清单重复启动；当前另12格未启动）：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/remaining_ablation_plan.json。标记planned_not_enqueued的命令只是后续候选，尚未执行。原catalog：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/jobs.json。

原未完成格包括3项写入失败、8项资源时限中断、3项从未启动；旧失败曲线/结果保留，因测试checkpoint已清理，均需独立fresh attempt，不能拼成恢复曲线。新来源清单：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/source_snapshot_v1.json，SHAc840011ab6b01c8b6dd35e61d059e2b7206551749654cf67cc4ebe1c9562db0b。

最终比较图只能合并协议相同且实际完整结束的格；在跑/失败/缺格明确标出，不能用早期loss代替完整epoch结果。


## 2026-10-06T04:27:38.275704+00:00 本地3500步验证：整体重建和结束标记改善，context重建CE略回升

本地f000 seed42完成3500步验证、112000/120000条曝光（93.33%），继续3750完整epoch。平均AA CE0.831506→0.826046，全遮蔽AA CE1.711951→1.696441、准确率42.3560%→43.5153%；整体全遮蔽terminal准确率25.3235%→42.9452%。含context37条的平均/全遮蔽AA CE略升，但全遮蔽terminal准确率45.4545%→50.0000%、CE下降；组别与指标差异均记录，不据单点或已有正例关系准确率推断功能收益。

新点170计数/40mask_ratio、1024验证记录与provenance保持，summary匹配，无终态或测试权重。详细数值见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation3500_trend_20261006.json（SHA83579fc9124cc40dffa8fd769bd18b12c649d913c828ead6db799596350ae5e1）。正式八卡f011继续25042目标，评估独立记录。完整22格仍8完成/1运行/13未启动，固定500/3750图不重画；下一本地3750完整epoch，既有训练和门禁不变。


## 2026-10-06T03:22:05.045022+00:00 本地3250步验证：AA重建回升，整体结束标记继续回落

本地f000 seed42完成3250步验证、104000/120000条曝光（86.67%），继续3750完整epoch。平均AA CE0.849367→0.831506，全遮蔽AA CE1.767363→1.711951、准确率38.5985%→42.3560%；整体全遮蔽terminal准确率29.2052%→25.3235%。含context37条的全遮蔽terminal准确率43.1818%→45.4545%，但CE上升；组别和指标方向分别记录，不据单点或已有正例关系准确率推断功能收益。

新点170计数/40mask_ratio、1024验证记录与provenance保持，summary匹配，无终态或测试权重。详细数值见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation3250_trend_20261006.json（SHA751d8c32272cc014655d5967536d5e0c89eee48e806cb1e82b44c2b6629bd82f）。原八卡f011最近完整保存4000且继续25042目标。完整22格仍8完成/1运行/13未启动，固定500/3750图不重画；下一本地3500、正式5000，既有训练和门禁不变。


## 2026-10-06T02:21:32.688388+00:00 本地3000步验证完成：平均AA CE持平，全遮蔽与结束标记准确率回落

本地f000 seed42完成3000步验证、96000/120000条曝光（80%），继续3750步完整epoch。平均AA CE0.849889→0.849367基本持平；全遮蔽AA CE1.744851→1.767363、准确率41.8326%→38.5985%；全遮蔽terminal准确率39.0018%→29.2052%。含context37条的全遮蔽AA CE上升、AA准确率下降，terminal准确率43.1818%持平。已有正例关系CE下降不等于正负binding能力。

仅核对新3000点，170计数/40mask_ratio、1024验证记录与provenance保持；summary匹配，无终态或测试权重。详细统计见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation3000_trend_20261006.json（SHAb11c26591a09b065f7176f76abf0a42a46d1db24036f2f23b36ed7185ba1e1fa）。原八卡f011最近完整验证保存4000并继续25042目标，两配方不混用。完整22格仍8完成/1运行/13未启动；固定500/3750图保持，下一本地3250/正式5000，既有训练和所有门禁不变。


## 2026-10-06T01:18:25.792578+00:00 本地2750步验证：总体重建回升，context结束标记仍有回落

本地f000 seed42完成2750步验证、88000/120000条曝光，仍继续3750步完整epoch。平均AA CE0.897508→0.849889；全遮蔽AA CE1.898242→1.744851、准确率37.9543%→41.8326%；全遮蔽terminal准确率20.1479%→39.0018%。含context37条的全遮蔽AA CE略升、terminal准确率52.2727%→43.1818%，正例关系CE也回升，不能只报告总体改善或声称结合功能收益。

新点170计数/40mask_ratio与2500相同，原provenance和1024验证记录保持；summary匹配，无测试权重或终态。学习率核对的检查脚本索引偏一已按冻结源修正，训练未受影响。详细数值与限制见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation2750_trend_20261006.json（SHAb2deacc4257cef0180858d94ac99c3b903ccb0351cf2bef769db0420acf59a7a）。完整22格仍8完成/1运行/13未启动，固定500/3750图不按中途点重画。原正式八卡继续完整25042目标；下一本地3000/正式4000。


## 2026-10-05T14:06:47.605732+00:00 本地消融的运行环境观察

本地训练PID3970535实际存活且持续输出进度；A100-80GB的NVML快照同时显示两处计算进程显存分配，约1.64GiB和21.49GiB，相关NVML PID在当前/proc命名空间不可见，尚不能确定归属或是否存在计算竞争；未向它们发送信号。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/local_gpu_environment_observation.json。本地与原8卡池的wall-time不是受控的架构速度比较，不据此宣称某方法加速；最终质量比较仍按相同数据/batch/seed/完整epoch协议核实。此观察不影响远端正式world8任务，也没有新训练或验证。

## 2026-10-05T14:28:32.090226+00:00 本地消融完成首次250步验证，八卡正式训练持续运行

本地f000 seed42已实际验证250/3750步、8000/120000条；全遮蔽AA CE为1.799719、准确率41.2489%，四档mask平均AA CE为1.538930。旧同模型/同seed在250步对应为1.798783/41.1658%及1.537953；全遮蔽AA CE差+0.000936（约+0.0521%），不能把四档平均与旧全遮蔽直接比较。样本曝光、输入/eligible/selected token数、学习率及每档验证计数一致，provenance差异仅checkpoint_saving和source_sha256；已匹配协议，不宣称逐位复现。初始全遮蔽AA CE差+0.000136；250步terminal准确率旧12.6309%/新23.5983%，terminal CE旧2.418198/新2.385153，数值差异来源未确认，不归因于架构收益或声称全部指标已复现。继续原3750步完整epoch，不修改或重启当前实验。

远端原任务t-20261005205851-lqfjg仍Running；观察2026-10-05T14:28:11.098227+00:00中8worker心跳新鲜、无abort，最新正式训练loss=1.4282。监督PID3977587与本地PID3970535的真实argv已核对，监督start_ticks一致；当前5000新增步段仍在运行，下一常规eval/save为1000步，已完整保存并验收的最近段为16步，不把目标5016当成完成。监督器仍是后续接段唯一执行者，原Goal保持active；本回合是运行证据更新和同口径结果比较，没有新平台提交或GPU验收。

比较原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation250_comparison_20261005.json（SHA3d236866d90b79a7d91ada369e314c429bdb8b2736726e664050482f1e35eedc）；实际原曲线 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f000_seed42_attempt1/curve.jsonl。测试目录仅日志/指标/来源记录，无checkpoint或tmp，正式恢复状态保持。完整22格仍8完成、1本地运行、13未启动；本地结束后优先f010 seed42/43。


## 2026-10-05T15:22:05.935892+00:00 消融结果汇总接入与后续本地退出记录

现有比较器只支持原单pool、相同执行来源及pool rc0；本地retention改动带来5处实际执行来源差异，不能假造pool成功或当成同一来源。已敲定独立混合来源报告策略，分别标注完整末点覆盖、实际退出和来源一致性；代码尚未实现，不算最终图完成。当前本地f000照常运行；其原启动者未保留未来wait退出码，不能倒填rc0。后续13项已准备复用既有run_in_tmux并记录实际PIPESTATUS退出值的命令，未启动或重试。训练源码、数据门禁、f011选型和旧账本均保持。

说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ABLATION_REPORT_INTEGRATION_20261005.md；未执行命令 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/managed_local_launch_plan_v1.json（SHA7413c5fc9783ea2aa81dc37ebbdb5ba25588e09f76cc88190039e1a9de26970e）。当前真实运行状态仍以 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_SUPERVISION_20261005.md 及本地progress为准。


## 2026-10-05T15:34:01.700681+00:00 本地500步验证完成，原八卡训练持续运行

本地f000 seed42实际500/3750步、16000/120000条，1024条验证中含37条context。与旧同格500步相比，样本/输入token/eligible/selected token/学习率及各验证分母一致。四档mask平均AA CE新1.185176/旧1.184855（差+0.000321，约+0.0271%）；全遮蔽AA CE新1.793335/旧1.792839（差+0.000496），AA准确率新40.2420%/旧40.4412%（差约-0.1992个百分点）。数值接近但不逐位相同，不作为架构收益或完整epoch结论。

500步全遮蔽terminal准确率新旧均48.0591%，250步该指标的差异在这一观察点没有延续；原因仍未确认，不据此宣称运行逐位复现。全遮蔽terminal CE新1.301628/旧1.306732；四档平均terminal准确率新80.6511%/旧80.3335%。已有正例关系指标不代表正负判别能力。本地与旧池wall-time环境不同，不据此作架构吞吐比较。完整3750步和最终消融图仍未完成，原f011初步选择保持。

原始比较 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation500_comparison_20261005.json（SHAbecc8fd766ecd1bffde60e8dee20ac8c46f0ddd5e35908675f90adfec5aaadd1）；直接曲线 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f000_seed42_attempt1/curve.jsonl。新summary与500末点一致，原生COMPLETE尚无，测试目录无checkpoint/tmp。原正式监督与本地进程的PID/start_ticks/argv均匹配；8worker心跳新鲜、无abort，最新正式训练loss=0.836，下一正式eval/save1000步尚待实际产物。观察session94960完成35次只读核实后自然退出，训练/持续监督未重启。原Goal继续active；本回合有新的500步验证证据和汇总接入问题的记录，没有新训练提交。


## 2026-10-05T16:39:14.429613+00:00 本地750步验证完成，正式训练继续

2026-10-05T16:39:14.429613+00:00 本地f000 seed42已完成750步验证（24000/120000条，20%计划epoch），当前progress为750/3750；完整epoch未完成。正式原8卡继续运行，最新训练loss=0.6997、8worker心跳新鲜、无abort；checkpoint1000已完成验证保存，下一正式节点2000步、本地下一验证1000步。

四档平均AA CE新1.066193/旧1.065402（差+0.000791，约+0.0743%）；全遮蔽AA CE新1.777032/旧1.775615（约+0.0798%），AA准确率新42.0923%/旧42.2900%。全遮蔽terminal准确率新28.4658%/旧31.0536%（差-2.5878个百分点），terminal CE新1.622146/旧1.626278；不因AA指标接近而隐去结束标记差异。四档平均terminal准确率新70.2284%/旧71.5257%。

样本/输入token/eligible/selected token/学习率一致，170个records或*_count字段及40个mask_ratio字段分别核对一致；1024条验证含37条context。逐来源和context分组的计数、分子与聚合已用既有metadata helper核对，没有重跑模型。provenance SHA与500步原件保持，差异仍为checkpoint_saving/source_sha256；不称位级复现，数值差异原因未确认，不归因架构收益、运行速度或功能改进。旧同格为未完成任务的750步历史观察，不作为旧完整epoch终态。原f011初步选择与当前训练保持。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation750_comparison_20261005.json（SHAd33170815d60b977ebe735cbc1926b189442bee02bd0d2b7246ef6cbf2742bae）；实际曲线 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f000_seed42_attempt1/curve.jsonl。summary与750末点一致，无COMPLETE/FAILURE、测试checkpoint或tmp。固定500/3750局部图保持原观察点，不重复渲染；完整22格仍8完成、1运行、13待补。

本回合在5次具体进程身份观察后取得新的750节点证据，分类progress；原Goal保持active，未重启/提交/重复GPU验收或改动训练源码/数据门禁。



## 2026-10-05T17:44:15.579408+00:00 本地1000步验证已核实

2026-10-05T17:44:15.579408+00:00：本地f000 seed42已完成1000步验证（32000/120000条，约26.67%计划epoch），当前progress为1000/3750；完整epoch未完成。正式原8卡继续运行，最新训练loss=0.6312、8worker心跳新鲜、无abort；最近完整验证/保存为1000步，当前5016段和25042目标尚未完成。下一节点：本地1250步、正式2000步。

同架构/seed的1000步比较（按token汇总；不是ESMC对比或架构增益）：

| 指标 | 旧同格历史点 | 本地本轮 | 本轮−旧 |
|---|---:|---:|---:|
| 四档平均AA CE | 1.006546 | 1.008167 | +0.001621 |
| 全遮蔽AA CE | 1.798385 | 1.798585 | +0.000200 |
| 全遮蔽AA准确率 | 40.4124% | 40.3622% | −0.0501个百分点 |
| 全遮蔽terminal CE | 2.156175 | 2.082055 | −0.074121 |
| 全遮蔽terminal准确率 | 19.0388% | 18.7307% | −0.3081个百分点 |
| 四档平均terminal准确率 | 70.9275% | 70.9231% | −0.0044个百分点 |

相较本轮750步，四档平均AA CE从1.066193降至1.008167，但全遮蔽AA CE从1.777032升至1.798585，AA准确率从42.0923%降至40.3622%。不能把均值改善概括为所有条件都改善，也不据单点改架构或调训练协议。

曝光和调度一致：32000条、输入token12336197、eligible4937019、selected2488592、学习率8.57637576121707e-5。170个records/*_count字段与40个mask_ratio字段分别相同；1024条验证含37条context。根端仅用既有metadata helper核对新旧1000点的计数、分子和聚合，未额外重跑模型。两份provenance SHA与此前原件保持，差异仍仅checkpoint_saving/source_sha256；不称位级复现、受控速度比较或生物功能收益。旧点来自最终失败/未完成的历史尝试，不算旧完整epoch成功；关系准确率1.0仍仅为既有正例token结果。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation1000_comparison_20261005.json（SHA244a482284610786a435d3dd81c78e5636ce0766ee4417698b530b5016169e78）；实际曲线 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f000_seed42_attempt1/curve.jsonl。summary与1000点一致，无COMPLETE/FAILURE或测试权重/tmp。固定500/3750图保持原观察点，不重绘；完整22格仍8完成、1本地运行、13未启动。

本回合22次具体进程观察后获得实际1000节点，分类progress；原Goal继续active。CURRENT交接区同时完成消歧，历史原文归档 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/handoff_before_cleanup_20261005T173354Z.md；未重启、重复提交、改变数据/来源/训练配置或放松门禁。


## 2026-10-05T18:47:08.472956+00:00 本地1250步验证完成，继续原完整epoch

2026-10-05T18:47:08.472956+00:00：本地f000 seed42已完成1250步验证，累计40000/120000条（计划epoch的三分之一），当前progress为1250/3750。正式原8卡继续运行，最新训练loss=0.5721，8worker心跳新鲜、无abort；最近完整验证/保存为1000步，当前5016段和完整25042目标尚未完成。下一节点为本地1500步、正式2000步。

同架构/seed的1250步比较（按token汇总）：

| 指标 | 旧同格1250步 | 本地本轮1250步 | 本轮−旧 |
|---|---:|---:|---:|
| 四档平均AA CE | 0.962144 | 0.961122 | -0.001022 |
| 全遮蔽AA CE | 1.787628 | 1.787297 | -0.000331 |
| 全遮蔽AA准确率 | 41.2550% | 41.2351% | -0.0199个百分点 |
| 全遮蔽terminal CE | 1.531905 | 1.485902 | -0.046003 |
| 全遮蔽terminal准确率 | 35.7979% | 36.2292% | +0.4313个百分点 |
| 四档平均terminal准确率 | 78.0813% | 78.9180% | +0.8367个百分点 |

同一当前实验1000→1250步：四档平均AA CE由1.008167降至0.961122，全遮蔽AA CE由1.798585降至1.787297，AA准确率由40.3622%升至41.2351%。全遮蔽terminal准确率由18.7307%升至36.2292%，仍需完整epoch观察；这不是位级复现、架构收益或ESMC比较。

新旧曝光与调度一致：40000条，输入token15391562、eligible6157115、selected3096993、学习率7.747811222909726e-5。170个records/*_count字段和40个mask_ratio字段分别相同；1024条验证含37条context。根端以既有metadata helper核对新旧1250点的计数、分子、逐来源/context分组及聚合，没有重跑模型。两份provenance SHA保持，差异仍仅checkpoint_saving/source_sha256，差异原因未确认。旧点来自最终失败/未完成尝试，不算旧完整epoch成功；已有正例关系token准确率1.0不代表正负判别或功能效果。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation1250_comparison_20261005.json（SHA27549b3190a1ae2c15307fb762c603be70e6f6b25d73a1bf0c593322313bc517）；实际曲线 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f000_seed42_attempt1/curve.jsonl，summary与1250点一致。完整22格仍8完成、1运行、13未启动；固定500/3750局部图保持，不因本次中途节点重绘。测试无checkpoint/tmp，正式恢复文件保留。

本回合14次具体进程观察后获得实际1250验证节点，分类progress；上一回合为progress（私有接续工程完成检查）。原Goal继续active。观察原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_bounded_watch_20261005T183454Z.jsonl，原监督/本地PID、start_ticks、argv持续匹配；有限观察自然退出rc0不是训练终态。未重启、submit/cancel、改变训练来源或数据/架构配置。

2026-10-05T19:52:34.799686+00:00 本地1500步已完成，匹配比较原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation1500_comparison_20261005.json。仍未完成3750步，不改变8完成/1运行/13未启动及固定终点局部图；具体改善和退步均见本地实验记录。

2026-10-05T20:59:44.248817+00:00 本地1750步验证已完成，匹配比较 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation1750_comparison_20261005.json。仍未完成3750步，8完成/1运行/13待启动及固定500/3750局部图保持；不把中途点当成完整格子。


## 2026-10-05T22:01:53.627091+00:00 本地2000步节点

本地f000 seed42已验证2000/3750，64000/120000条；完整22格仍8完成/1运行/13未启动。AA与terminal相较本轮1750改善，详细原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation2000_comparison_20261005.json。旧失败同格曲线到2000结束，后续不补造历史点；固定500/3750局部图不重绘。原八卡正式训练继续，最新完整保存2000，完整目标未完成。下一本地2250、正式3000；训练配置与接续顺序保持。


## 2026-10-05T22:30:51.057390+00:00 正式3000步节点

正式原八卡3000步验证/保存已核实，继续25042目标；详情 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md。本地f000当前2100/3750、最近验证2000；完整22格仍8完成/1运行/13未启动。正式指标与pilot配方独立，不纳入消融排名；固定500/3750局部图不重绘。下一本地2250、正式4000，接续顺序不变。


## 2026-10-05T23:07:29.260939+00:00 本地2250步节点

本地f000 seed42已验证2250/3750，72000/120000条；完整22格仍8完成/1运行/13未启动。平均AA CE下降，但全遮蔽AA与terminal退步，含context子集AA CE也回升，详情 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation2250_trend_20261005.json。2250没有旧同seed对照，只记录本轮时间变化；固定500/3750图不重绘。正式原八卡最近完整保存3000，仍继续25042目标；下一本地2500、正式4000。

## 2026-10-06T00:11:42.114535+00:00 f000 seed42的2500步结果已记录

本地完成80000/120000条曝光，仍继续3750完整epoch。平均AA CE0.897508，全遮蔽AA CE1.898242、准确率37.9543%，全遮蔽terminal准确率20.1479%，较2250退步；较低遮蔽档位AA CE改善。此为同一在跑实验的中间结果，不更新固定500/3750图或作最终排名。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation2500_trend_20261006.json（SHAa9e3a6b66553171173c3977149f542a0d7bf7c3a97ce6848a4224aa388846e76）；完整22格仍8完成/1运行/13未启动，下一本地2750。


## 2026-10-06T14:49:24.133710+00:00 后续publisher已准备，原矩阵仍未完成

重新读取原22格：9完整产物（8退出成功已核实、1退出未知）、本地f010/42继续运行、12尚缺。当前本地最近验证2000/3750；后续候选由 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v1/read_only_preview_20261006.json记录，发布时重新核实，不把候选当成已启动。旧资源3张从未开始的票据已归档；原失败结果不改。正式两轮和9项生成完成后才可使用冻结publisher；当前图不重绘、无新增完整消融结果。
