# V7 正式训练完成后的八卡消融接续

## 2026-10-06T22:30:09.108912+00:00 新八项初始验证和汇总接入已核实

原冻结报告器已实际读取新组合选择通过（3ab774 rc0），证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/initial_actual_report.json（SHA 602b36fad7f179ae5ac1bd5931730518f485d6858b09649a43790bf713c18465）。八个新运行均已有step0的1024条初始验证与真实provenance；目前尚无已核实的训练更新，不能将初始验证说成3750训练通过。旧10完整产物保留，f010/42真实rc0已接入、f000/42退出码仍未知；其余4项仍排队。完整图继续10/22 PARTIAL，混合来源差异保留，不重绘没有新增末点的图。

后续读取实际产物位置 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20261005/runs/<job_id>，不是原本地plan的output_dir；日志在同pool的logs/<job_id>.log。plotter复用post_generation_pool_v1/source_v1，v2没有另复制绘图器。后续核实首25更新/固定250验证及实际退出，不重发12票据。原序列注释停线、数据门禁和独立CUDA暂停不变。

## 2026-10-06T22:28:30.597931+00:00 原八卡接续12项缺失消融，已有10项不重跑

真实两轮及九项CDR完成、训练/生成监督实际退出后，12个缺失格子已单次发布，当前8运行/4排队/0结果；实际GPU更新尚待验收。清单及资源门禁 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/root_publication_review.json，唯一票据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/enqueue_receipt.json，最终图选择 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/mixed_report_selection_with_local_exit_v1.json。每格120000条/3750步完整epoch，metrics-only；原数据/源码/共享锁和不自动重试保持。完整图仍10/22 PARTIAL，f000/42退出码未知、f010/42真实rc0已接入；不得重发批次或重新运行已完整格子。

## 2026-10-06T22:17:06.607485+00:00 f010已实际完成；未来12格发布和报告接入保持

实际退出凭据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_terminal_evidence_20261006/actual_exit_receipt_v2.json，完整验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_terminal_evidence_20261006/root_full_epoch_review.json；固定3750新图10/22。12格未启动，f010/42现应排除为已有完成产物，不再称其运行中。冻结publisher仍读取旧不可变selection_v2，其该格exit ref为null不影响排除已有完整产物；最终pool图须在新组合选择中加入上述实际exit ref。不得改冻结选择/重跑f010或把f000/42未知退出补成0。正式当前49900/50082、生成未开始，当前无消融发布/新监督。


## 2026-10-06T22:01:12.787358+00:00 本地f010未来完整退出的选择文件衔接

/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/record_v7_f010_local_completion.py 已准备，真实进程仍在时拒绝核查通过，尚无退出凭据。现有私有pool发布包仍使用不可变mixed_report_selection_v2，能排除已经有产物或归本地所有的f010格，不能据其exit_evidence=null推定需要重跑。未来实际退出采集会另写 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_terminal_evidence_20261006/mixed_report_selection_v3.json。最终pool混合图应复制新pool选择为另一个明确的新文件，再接入已核实的f010 exit ref；旧f000/42退出仍未知，不能一起补成成功。真实完成前不创建v3或宣称完整退出。当前正式48576/50082，生成和后续12格尚未启动。


## 2026-10-06T21:37:34.475996+00:00 正式46625/50082继续；生成后消融接续更新已核查、未发布

当前原八卡45058→50058，实际计时46625/50082、abort=null；恢复监督319218/start2458293575与新生成等待监督320497/start2458326416存活，8worker新鲜running。46000完整保存/验证已核查，下一47000；当前44000/45000/46000三份完整状态，latest在best3中。恢复后真实300步均值0.721418秒；未新增测速/模型或独立CUDA测试。新生成仍等待50082实际终态与八卡退出，9项未入队、尚无结果；不自动开始第三轮。

后续消融私有包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/package_v1.json（SHAff3ee6d8617a1621522b4cadf1f9d81259e4bf138ed7e905979bc4f28e2a7865）仅改为当前恢复/生成完成证据，原模型/数据/配方/22格和共享锁不变。6项新增边界/源码检查通过（7433e1 rc0，0.629秒、8个STOP/STOP_NOW子场景），原已验收套件未重跑；临时fixture清零，无测试权重。实际只读预览与提前发布拒绝通过（557f05 rc0）：当前12格符合未启动候选条件，9格产物和本地f010/42排除；训练/生成监督仍活着时发布明确失败，公开源码、registry、私有发布文件和claims均未改变。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/root_cold_review_v1.json（SHA5cd95547897b826ba9d7861349958b2e9fdd5b29ea09546f5908cfbdc2645fb2）及 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/lineage_controls_v1.json（SHA92c2ee0d30302a0ac50acf2acbec23f42a2e159e4e2be9fc02905ffbef6fb926）。

当前本地f010/42为3600/3750、最新完整验证3500；3750终态/实际退出仍待核实。完整22格9产物（8成功退出、1未知）/1运行/12未启动，既有固定图仍PARTIAL。后续重新读取真实缺失格子、两轮/9项生成终态、监督退出、八worker空闲和原资源时限后才能单次发布；本轮没有启动新的消融监督。旧失败stage/两个旧监督保持，原安全停线/数据门禁/独立CUDA暂停不变。原Goal active，完整目标及累计用量不改。本轮progress，未提交/取消平台任务或重启容器。


## 2026-10-06T21:32:18.621472+00:00 46000步保存核查通过；消融接续独立版本准备中

46000完整保存和固定验证已实际核查（953224 rc0），原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint46000_root_review.json（SHA9887798690ce93740185301a56057fab2914505f81781546bfb25d21fce5c216）。当前正式长段45058→50058运行，完整目标50082；八卡及恢复监督未重启。主eval_loss由45058的1.865579037降到1.862781763；下游生成尚未开始，不能据此认定生成能力提升。完整验证数字见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md。

后续消融旧准备包仍指向失败的原终态，现新增私有 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/package_v1.json，绑定当前恢复终态与新生成批次；旧包不改、不发布。模型、120k/3750配方、原22格和共享认领锁保持。代码已写入但新边界检查尚未完成；没有新票据、认领、公开入口修改或新监督启动。生成仍等待真实50082完成与八卡退出；消融还要再等9项生成完成。原失败记录/安全停线/审核门禁/独立CUDA暂停保持。


## 2026-10-06T14:49:24.133710+00:00 原八卡已接续过17016；后续消融发布包准备与检查完成

正式12016→17016段实际完成，8个原生恢复/保存回执、8个pool成功退出及8份进程清理记录已核实；原监督正常接续17016→22016。根端最新实测17523/50082，abort=null，三个监督PID/start/源码身份及8worker新鲜心跳匹配。两轮接续和生成监督仍等待，无生成成绩。段验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/segment17016_acceptance.json；本轮根记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v1/root_review_v1.json。

已准备独立后续消融publisher并封存6份代码及依赖清单：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v1/package_v1.json。入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v1/source_v1/publish_v7_post_generation_ablation.py。只读preview重新读取22格与现有attempt，当前12格可作为未来候选、9格完整产物、f010/42运行；这个名单到发布时必须重新读取，不固化当前状态。新选择文件将继续沿用原混合来源报告器和固定3750末点。

发布必须在50082及9项生成真实完成、三个监督实际退出、八卡空闲且队列为空后进行；原数据/来源和完整退出检查保持。剩余时限按原worker max_seconds604200、job_seconds86400核实，用所有波次的完整超时预算保守准入，不修改deadline、不把预算说成速度预测。当前12项需两波，要求至少173400秒剩余。只扩展原公开启动入口，保留全部旧语句与既有registry，原模型/训练快照保持。

6项新检查实际通过（34566；08e413 rc0，1.362秒）：真实现状preview排除已完成和运行格；训练仍在时实际publish调用在任何发布写入前拒绝；新attempt排除与时限检查；纯stdlib子进程实际等待批次释放再获取共享锁；错误批次引用拒绝。新增批次等待确保publisher持锁写票据期间worker不会误报锁冲突，最终enqueued只在释放锁后写入。日志 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v1/publication_tests_v1.log；临时人工目录已清理，无模型/GPU测试或checkpoint。

旧平台t-20261001033327-fmnql只读查询确认为Failed、End=2026-10-03T19:24:33Z，旧8worker均exited。3张从未启动的pending票据已保留原字节归档到 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_adapter_v2/old_unstarted_tickets_archive，清除旧队列占位；未伪造退出/完成结果、未调用平台cancel。原catalog与历史失败证据保留。

新版尚未实际发布、入队或创建正式claims/catalog；当前公开训练入口、已部署生成包、本地v1 guard依赖均未改。下一步先等待原训练/生成，满足真实门禁后在原资源单次发布缺失格子，并核实实际GPU运行和最终22格对比。当前本地最新验证2000/3750。原Goal active，本回合及上回合均progress，完整范围/累计用量不变；安全停线、数据门禁和CUDA暂停保持。


## 2026-10-06T14:35:22.374740+00:00 两轮及生成评测后的消融入口v2：私有元数据检查通过

已新增 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_metrics_only_ablation_v2_dispatch.py 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/v7_post_generation_acceptance.py。新入口要求50082/epoch2实际完成、8rank成功退出与进程清理、9个生成任务完整覆盖和评测/worker双重成功退出，仍调用冻结的原生成数据审核检查。原监督的STOP必须保留；当前阶段STOP/失败继续阻止准入。旧adapter仍由正在运行的本地guard绑定，SHA保持不变。

消融配置及模型不变：原120000条、3750步、micro32/GA1、eval250/eval4、metrics-only。数据/源码/容量检查、worker身份和exec主体与v1的AST一致。共享锁必须在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/guarded_dispatch_v1/cell_locks，避免另建锁目录造成同格并发；旧COMPLETE即使退出码未知也阻止重复，新pool已有成功或失败结果均不自动重跑；尚存生成任务也拒绝启动。

实际13项人工元数据检查通过，0.544秒、rc0，含真实命名空间缺50082完成证据时拒绝而不写文件。入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_post_generation_acceptance.py；日志 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_adapter_v2/metadata_tests_v1.log；根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_adapter_v2/root_review_v2.json。测试用纯人工文件和代理审核调用，不含序列/模型/GPU/真实证书，临时目录均已清理。不能将这些检查当作八卡实际接续通过。

当前正式训练16549/50082、段目标17016、abort=null；三个监督PID/start/源码和8worker新鲜心跳核实。生成仍waiting_two_epochs，公开入口及全部冻结生成包SHA保持。新版消融未发布、未入队、未创建真实claims/ledger；发布代码及终态后真实缺失格子核实仍待完成。不得直接运行旧25042适配器。当前22格9完整产物、1本地运行、12未启动，此数字不固化为未来提交列表。

本回合为progress，上一回合也为progress；原Goal身份、累计用量与完整要求保持active。未submit/cancel或重启，独立CUDA暂停、序列注释停线及原数据门禁不变。


## 2026-10-06T14:15:40.813569+00:00 当前后续顺序已延长到两轮和生成评测之后

用户新增累计两轮50082及该点生成评测；本页旧25042/原SUP终态规则是旧计划，不能用于提前领取卡。正式两轮与9项CDR完成后再准入缺失格子。旧私有pool_metrics_only_ablation_dispatch.py仍指向旧终态身份，尚需独立适配新完成证据及当时公共入口，不能改造旧回执或直接执行。当前22格9完整产物、1本地f010/42运行、12未启动；未来重新核实，不把此计数硬编码为待提交名单。生成衔接 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md。

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

本地f000/42完整3750步/120000条epoch已验收，退出码未知如实标注；3750固定末点图已更新为9/22，13项未启动。f011平均AA CE较低，但全遮蔽AA/END没有同步优于基线，不能作理解/生成全面收益结论。正式八卡继续5400/25042，配置不变。f010/42真实接续包准备中，尚未启动；受控测速仍因非独占未执行。详情 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md，根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/completion3750_root_review_20261006.json，新图 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_epoch1/step3750_architecture.png。

## 2026-10-05T18:33:58.003212+00:00 后续本地消融私有启动保护与报告接入完成检查

2026-10-05T18:33:58.003212+00:00：原八卡正式训练继续5016目标段，最新训练loss=0.5692；最近完整验证/保存为1000步，完整25042步尚未完成。本地f000 seed42为1200/3750步、38400/120000条，最近验证1000步；下一节点仍为本地1250步和正式2000步。

未来本地消融的私有guard、共用格子/设备锁和报告器v2接入已完成本轮工程检查：子agent7项人工检查（5.984秒、rc0），追加实际临时receipt→reader接口1项（1.456秒、rc0）；根端报告器3项定向检查（1.411秒、rc0）。只有锁探针进行了真实CPU exec，训练exec为mock，无新模型/GPU测试。根端源码/原metadata字段/固定绑定和当前4个正式公共发布SHA已复核；没有重复运行既有pool或模型验收。

私有实现尚未用于真实接续，真实admission/认领账本/锁文件未创建；当前本地进程不迁移或重启。后续先等待本地完整epoch和资源释放，再按最新缺失格子启动guarded任务；八卡复用仍等待正式25042终态、全部清理、空闲和剩余时限核实。原数据审核及停止序列注释、独立CUDA暂停保持。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/local_guard_root_review_20261005.json（SHA932befc2eb38d317595b81ea4680fba06913f8fdf175c78158ffca5ae3220cb9）；报告器检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/guarded_reporter_tests_20261005.json。旧图使用的源码已归档 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/reporter_source_v1，固定500/3750局部图不重绘。人工fixture已清理，无测试权重，正式恢复文件保留。

新增本地guard：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/local_metrics_only_ablation_dispatch.py；保持pool适配器原SHA。prepare只读；main必须由未来SHA绑定的tmux命令显式调用，拿锁后重核并写实际pre-exec receipt，再exec原冻结CLI。实际跨环境任务互斥/训练接续仍未验收，人工锁/metadata检查不能替代真实部署。


## 2026-10-05 18:24 UTC 本地共用锁与报告接入补齐中

原八卡正式训练和本地 f000 保持运行。仅为未来新本地任务新增私有启动保护，复用现有 22 格认领记录和空文件锁；已有训练入口、冻结来源、managed plan 与 pool 适配器均保持。启动前的 immutable admission 与启动后的真实 tmux/guard/退出凭据分开，避免启动记录循环依赖。报告器将显式支持新 v2 凭据，仍要求原生完整产物和实际训练退出码，不能把准备或拿锁当训练成功。

本轮实现与人工元数据检查尚在进行；未建立真实认领账本、未启动新训练、未接入运行中的任务。当前正式 25042 终态尚未完成，八卡消融复用仍须按下方门禁等待。

## 2026-10-05T17:12:40.120217+00:00 当前：私有实现与元数据检查完成，未部署

2026-10-05T17:12:40.120217+00:00：正式原八卡训练继续，监督与本地PID/start_ticks/argv匹配，8worker心跳新鲜且无abort；最近完整验证/保存为1000步，最新训练loss=0.6466。当前5016段、完整25042步目标尚未完成。本地f000 seed42为875/3750步、28000/120000条，最近完成750步验证；下一节点为正式2000步、本地1000步。

私有后续消融接入代码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_metrics_only_ablation_dispatch.py 已完成本轮实现和审阅，SHA a9baf1390cdeeba315c7984c8602599e29033a356fd7649b82dcb206899a5fe8；对应人工检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_pool_metrics_only_ablation_dispatch.py，SHA 9b6d70e8844373299857ac9b160b828f3f0846d12afc3c8c90a47b48de772c0a。子agent实际执行6项人工元数据检查，3.467秒、rc0；根端检查了源码和覆盖范围，没有重复运行这些检查。根端另用实际既有来源/旧池容量/样本与父审核元数据运行check_sources_and_batch和check_data，2.153秒、rc0，未导入任何ML模块或读取序列。

已修正resume默认False、worker字符串device、formal_member别名，以及拿锁后重读认领信息和核对锁inode。exec只在人工测试中被mock；真实prepare/main、真实catalog/票据、公共发布和GPU执行均未发生。现有本地启动器尚未共用该锁，后续必须接入后再混合调度；本轮不能宣称跨环境互斥或GPU接入已验收。正式25042及八rank清理/空闲、剩余资源预算和最新缺失格子核实仍是未来发布前置。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_formal_pool_adapter_root_metadata_review_20261005.json（SHA e347b9dece500d8e59b07999a4eed04ce66803ddee0414d2697e7ad92881df2b）；说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_POST_FORMAL_ABLATION_REUSE_20261005.md。当前四个正式公共发布文件SHA不变；自有人工fixture及本地测试权重/tmp均无遗留。正式恢复文件保持。

下方较早章节为过程记录，当前状态以上述本节为准。

## 2026-10-05T17:10:27.763497+00:00 实现检查点

2026-10-05T17:10:27.763497+00:00：正式原八卡仍运行，监督与本地PID/start_ticks/argv匹配，八worker新鲜且无abort。最近完整保存节点1000步，最新训练loss=0.6466，当前5016段和25042完整目标均未完成。本地f000 seed42为875/3750步、28000/120000条；最近验证750步，下一验证1000步。

后续消融的私有接入代码已写入 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_metrics_only_ablation_dispatch.py，尚在检查，未发布/入队或运行模型。根端初审已要求并核对修正resume默认False、worker字符串device、formal_member别名和拿锁后重查。人工元数据检查尚未完成；未来本地启动也要接入共用锁，不能把单个适配器当成跨环境互斥已完成。

下方16:52为最初只读计划，以上是当前实现状态；真实发布/正式终态验收仍未发生。

更新时间：2026-10-05T16:52:36.604444+00:00。状态：代码只读审阅与方案判断完成；新适配器尚未实现、发布或入队。当前八卡正式训练继续，本地 f000 seed42 消融继续。

## 决定与范围

保留现有八卡 worker，后续用薄适配器接入已经准备好的 metrics-only 消融训练入口。正式 25042 步完成、原监督器验收成功且全部子进程清理后，才考虑复用这些 GPU；当前不改公共启动文件、冻结模型、worker 或 Trainer。正式训练期间，本地消融仍按已有顺序推进。

本次审阅原件：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_formal_pool_readonly_review_20261005.json（SHA ef66943741234ca5b4c3ccf5333fe59ccb5366d5fe709bc9eb869d20cbaeab1d）。本次没有运行新模型、GPU 验证或训练启动。

## 实际代码依据

- /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/immune_research_pool.py 已有原子领取任务、单卡 CUDA mask、实际子进程 returncode 和 COMPLETE 检查。它固定调用公共入口，不接受任意训练脚本字段，不能只往队列写一个新脚本路径。
- /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/pool_parent_source_v1/pool_parent_runtime_observation.py 只匹配旧容量测试的唯一 alias 与 job ID。新的 p04 消融任务不会触发这段旧容量观察；此判断来自代码阅读，没有再次执行旧容量测试。
- /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/launcher_source_v1/pool_large_batch_ablation_dispatch.py 固定旧 pool、完整 22 格登记、568 文件快照和旧容量证据。仅替换 pool/catalog 路径不足以接入新的 569 文件 metrics-only 快照；不得篡改原账本或重放旧失败 ID。
- 当前正式公共入口的四个发布文件已与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/entry_20261005/publication.json 核对一致。本次只读检查不替代既有完整来源验收。

## 后续最小实现及实际验收

1. 私有适配器只负责解析任务、核对固定来源/数据/既有 batch 选择证据，再调用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/source_snapshot_v1/scripts/research/run_v7_large_batch_ablation.py。复用 120k、3750 步、micro32/GA1、250 步验证和不保存测试权重的既定配置，不新增 Trainer 或架构。
2. 保留 worker 分配的物理 `CUDA_VISIBLE_DEVICES`，不能复制本地固定 GPU0 的环境到八个任务。训练子进程由既有 worker 持有并记录真实退出码；有 COMPLETE 文件不等于已证明退出码为 0。
3. 实际接入前刷新完整 22 格的结果与运行账本。当下是 8 完成、1 本地运行、13 未启动，但这不是未来固定提交列表。通过共用的格子认领记录排除已经完成或仍在本地运行的 variant/seed；只为确实缺失的格子创建新的尝试 ID、输出目录和实际来源记录。
4. 新代码仅做必要的无 GPU 元数据检查：错误来源/数据/重复格子拒绝，CUDA mask 保持，命令与原 frozen runner 一致，终态证据正确关联。既有容量、正式保存恢复和已通过的消融检查不重复。以上检查尚未执行，不能标记 passed。
5. 发布与入队必须等待正式终态验收、八卡新鲜空闲状态、空队列及足够剩余运行时间；保持原作业时限和数据审核范围。新测试仍不保存权重，正式恢复 checkpoint 保留。若失败，保留原始证据，不能伪造终态或自动重放失败任务。

后续汇总沿用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ABLATION_REPORT_INTEGRATION_20261005.md，分别展示来源、覆盖和实际退出状态。当前 500/3750 步局部图维持原固定观察点，不因这次计划审阅重绘或声称完整消融完成。

停止的序列注释分支不得恢复、重试、转交或绕过；独立 CUDA 算子专项仍暂停。现有数据审核及所有实际启动门禁保持；本方案没有扩大审核结论。
