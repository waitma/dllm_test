# V7 初步联合训练与后续消融

## 2026-10-06T10:41:11.204701+00:00 正式7000步验证与保存已验收

平均AA CE0.746101→0.736503、全遮蔽AA CE1.749312→1.728628改善；全遮蔽END准确率57.4277%→56.3310%、首END准确率16.7498%→16.3011%回退，长度MAE3.220339→3.438404变差（分母2006→2005）。误报END减少，但不能据总体loss宣布全面提升。17原生文件/6JSON归档/8rank一致性、112固定计数、70核心聚合、26结束指标核对通过；不重复载入tensor或运行模型。详细表 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md；根证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint7000_root_observation.json。

新FSDP 2×4 YAML及CPU验收保持上一轮状态，未提交，未重复验收或改写已交付源码。本地f010/42最近验证1000保持，下一1250；固定500/3750图与22格计划保持。

速度补充来源与端点已复核：普通训练区间仍10.1991/10.6956秒/更新，慢化不只发生在验证/保存。冻结代码禁用memory-efficient、可训练bias限制其他融合后端，源码推断math SDPA为优先排查对象，尚无实际kernel trace。根核对 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/steady_metadata_root_review_20261006.json；原安全停线和独立CUDA专项暂停保持。首次文档发布工具在JavaScript解析阶段失败，未执行或写文件；修正后发布，不重跑7000验收。

## 2026-10-06T09:41:42.093903+00:00 V5/current源码对比完成；澄清2048上限与checkpoint策略

23份关键源码对照及解释已写入 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_V5_CODE_COMPARISON_20261006.md，当前源码逐份匹配冻结manifest。V5 FSDP由保存参数确认；历史git版本可复原但未提交补丁缺失，报告明确此限制。新确认层内显式RoPE读取position_ids.max()及component float同步、稠密可训练bias反向、固定target容量与batch padding均值得受控计时；尚未实测瓶颈或切换并行方式。

2048是整条记录上限，并非统一padding。既有1024条小样本/micro32的32批实际S=516–1206，17批为516；固定target容量256/64仍在。当前每1000步及分段边界保存，保留最近3个完整恢复点（5000/5016/6000，合计约2.724GB），不同于V5最优3+最新/slim。新增证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/root_review.json，没有重跑模型、读原始序列、修改训练配置或清理正式恢复状态。文档f010/R24标签已纠正。

当前正式6725/25042、本地850/3750，原进程与8rank心跳正常，完整验证仍6000/750；有限观察41757已实际rc0结束，原训练继续。完整Goal仍active，本轮progress。既有安全停线、独立CUDA暂停、远端403不重试和数据/来源门禁保持。


## 2026-10-06T09:17:22.685086+00:00 f010完成750步验证：平均AA改善，END优势缩小且context回退

正式训练最新6575/25042、完整验证6000；本地f010已验收750步，24000/120000曝光，当前训练750/3750。平均AA改善，全遮蔽CE/END仍有回退，不更换正式f011。详情 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LOCAL_ABLATION_PREPARATION_20261005.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation750_comparison_20261006.json。

完整epoch仍9格产物（8格成功退出、f000/42退出码未知）、1运行、12未启动；固定500图10/22、3750图9/22均不重画。下一本地1000/正式7000，既有安全停线和门禁保持。

## 2026-10-06T08:12:10.109919+00:00 f010完成500步固定点：平均重建与END改善，全遮蔽AA回退

正式八卡f011继续运行，最新训练日志6225/25042步，最近已核实验证/保存6000，当前段目标10016。原监督PID3977587/start2446951610与8rank新鲜心跳正常，无abort；DDP+NCCL、micro32/GA1/global256不变。

本地f010/42已完成500步验证，16000/120000条训练曝光；当前训练500/3750步，原PID80642/start2452750909持续匹配，训练退出码仍未知。固定1024条验证（context37、无context987），曝光/token/学习率与本地f000/42同500步一致；170计数字段/40个mask_ratio匹配，复用已验收250点的来源和计数证据。

同500步：四档平均AA CE1.185176→1.106683、准确率63.2778%→66.2457%；全遮蔽AA CE1.793335→1.807474、准确率40.2420%→39.7036%回退。全遮蔽END token准确率48.0591%→58.7800%改善。含context37条的平均/全遮蔽AA CE均下降，但全遮蔽AA准确率45.1168%→44.3125%。这仍是单seed/早期重建结果，不能认定生成与理解全面收益，保持正式f011及既定3750完整epoch/第二seed计划。

固定500步图已更新至10/22、PARTIAL；完整3750图仍9/22（8格成功退出已核实、f000/42退出码未知），另1格运行、12格未启动。PNG已目视检查，CSV/JSON/SVG/PDF保留。新图 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_step500_f010/step500_architecture.png；对照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation500_comparison_20261006.json（SHA45c0985b46fd2e11477bee20c581ffc98d604ce613c417a2a035c69d4745aa97）；根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation500_root_review_20261006.json（SHAb89bc9b54b12db6ed1537e9dee33a0ce753ddaa5fc24365abfd80ca31ba56007）。

观察48013在实际500验证出现后结束，14次观察、工具2e2859实际rc0；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_bounded_watch_20261006T075807Z.jsonl（SHA26c7d8f95cb0f4e182f86a006167726e9aa65ef216a85805ce840ea66152a3ca）。比较工具0e12f1、绘图工具cfffae均实际rc0；没有新模型运行、平台提交、重启或重复旧点评估。原正式/本地训练与原监督继续，当前没有运行中的有限观察会话。自有测试目录无权重/tmp可删，正式恢复状态保留。下一本地750、正式7000；完整Goal仍active，本轮progress。序列注释停线、独立CUDA暂停、远端403不重试和限定数据/来源门禁保持。

## 2026-10-06T07:40:52.614659+00:00 正式6000步已完成验证/保存

正式八卡f011已完成常规6000步验证/保存（6000/25042，约23.96%）；当前最新训练日志6025/25042，当前段目标10016。DDP+NCCL、micro32/GA1/global256不变。原监督PID3977587/start2446951610和本地f010 PID80642/start2452750909的/proc身份仍匹配，8rank心跳新鲜，无abort。

新点通过原17文件/8rank/6JSON及指标复核。平均AA CE略降，全遮蔽AA CE与长度MAE较5000回退，详表 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md。本地f010训练400/3750、验证250；完整22格仍9完整epoch/1运行/12未启动。原训练继续，下一正式7000/本地500，未修改架构或并行方式。

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

## 2026-10-06T04:39:59.360843+00:00 正式5016段完成并接续10016；段尾评估单独记录

16→5016的5000更新已完成，8个实际退出/进程清理及24项optimizer/scheduler/各rank RNG恢复摘要通过；原监督在同一8卡pool接续5016→10016，新段已有5025日志，完整目标仍25042。新段完整恢复回执尚待结束后核实，不声称与不间断轨迹等价。根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/segment5016_root_review_20261006.json（SHA7dcc51cf7133f82146e92b6c99b50413bcb2bb4548eb98d6ef3d70b7b84fc51c）。

5016是冻结callback强制生成的段尾评估，不只是保存；较5000只增加16步。平均AA CE0.748578→0.754328、全遮蔽AA准确率42.8318%→41.3847%回落；真实END准确率55.6830%→62.1635%上升，首END准确率16.3011%→18.7438%，但AA误报END1945→2258、token END精确率下降。分来源差异和完整表见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md；新点 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint5016_root_observation.json（SHAf13fe5643e7fc844b593a8a85eec9ef80758acd2659c1131889f77a923c4e43f）。不挑较好中间点，也不据单点改变当前架构。

5条外层清理扫描FileNotFoundError与原生进程清理分开记录；8slot无测试权重，正式状态完整，不热改冻结pool。状态读取修正了“5016无新评估”的假设，实际训练未受影响。本地3500已验证，继续3750；下一常规正式6000，22格及全部原门禁保持。


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


## 2026-10-06T00:11:42.114535+00:00 本地2500步验证完成，正式八卡继续运行

本地f000 seed42已训练80000/120000条并验证2500/3750步：平均AA CE0.876467→0.897508，全遮蔽AA CE1.774273→1.898242、准确率41.1548%→37.9543%，terminal准确率27.1103%→20.1479%。较低遮蔽档位与含context37条的平均AA CE改善，不能概括为所有指标同向。继续原完整epoch；正式仍继续5016段/25042总目标，最近完整保存3000步，两套配方不混用。

新点170计数/40mask_ratio与2250相同，来源记录保持；没有旧同seed2500对照。详细结果 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation2500_trend_20261006.json（SHAa9e3a6b66553171173c3977149f542a0d7bf7c3a97ce6848a4224aa388846e76）。完整22格8完成/1运行/13未启动，固定500/3750图保持PARTIAL；下一本地2750、正式4000。原进程继续，无重启或重复提交，本地没有测试权重/tmp。

## 2026-10-05T23:07:29.260939+00:00 本地2250步验证完成：平均AA改善，全遮蔽与结束标记回落

本地f000 seed42已完成2250步验证，曝光72000/120000条（60%）；继续原3750步完整epoch。正式原八卡继续5016段，最近完整验证/保存3000步，完整25042目标未完成。两套配方独立，不混用指标。

| 指标（按token汇总） | 本轮2000步 | 本轮2250步 | 变化 |
|---|---:|---:|---:|
| 四档平均AA CE | 0.886307 | 0.876467 | -0.009839 |
| 全遮蔽AA CE | 1.761686 | 1.774273 | +0.012587 |
| 全遮蔽AA准确率 | 42.2681% | 41.1548% | -1.1133个百分点 |
| 全遮蔽terminal CE | 1.236364 | 1.884492 | +0.648128 |
| 全遮蔽terminal准确率 | 55.5761% | 27.1103% | -28.4658个百分点 |
| 四档平均terminal准确率 | 85.4062% | 78.2787% | -7.1275个百分点 |

平均AA loss与全遮蔽表现方向不同；此前2000步结束标记回升，在2250步再次回落。不能只报告平均loss下降，也不据中途单点改架构、loss、batch或提前挑选最佳步数。四档平均正例关系CE0.000786→0.000736，全遮蔽关系CE0.000455→0.000518，准确率仍1.0；这不是正负binding或功能效果。

| 验证分组 | 记录数 | 目标AA平均CE：2000→2250 | 全遮蔽目标AA CE：2000→2250 | 全遮蔽terminal准确率：2000→2250 |
|---|---:|---:|---:|---:|
| 含context | 37 | 1.256698→1.265946 | 1.854908→1.866306 | 52.2727%→50.0000% |
| 无context | 987 | 0.879368→0.869177 | 1.759984→1.772592 | 55.6681%→26.4725% |


分组指标均针对target，context序列CE仍为0。含context仅37条，不把分组变化当作广泛条件生成/结合能力结论；其四档平均terminal准确率81.9526%→83.4677%，与全遮蔽terminal回落方向不同。总体平均AA CE改善不能替代对context子集的检查。

新点曝光/调度：72000条、input27632537、eligible11062274、selected5547363、学习率3.6236519169490876e-5。既有metadata helper核对2250点的计数/分子、来源/context分组和聚合，170个records/*_count字段与40个mask_ratio字段匹配本轮2000；1024验证记录与provenance SHA保持。2000点复用已核实记录和原始行SHA，未重验旧点、未额外运行模型。

旧同架构/seed历史曲线至2000结束，实际旧文件SHA仍一致；没有旧2250点，不能补造或外推。此次只记录同一当前实验的时间变化，不形成新架构、ESMC、位级复现、速度或生物功能收益结论。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation2250_trend_20261005.json（SHAda1eb090ec4fdda83c8ce822e2c05deb77ed4116c3c4a48485779bb1b7fcb500）；实际曲线 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f000_seed42_attempt1/curve.jsonl，summary与2250点一致，无COMPLETE/FAILURE、checkpoint/tmp。

本回合4次有限观察取得新2250节点，分类progress；原监督/本地身份与8worker心跳再次核实，无abort、重启或重复提交。完整22格仍8完成/1运行/13未启动，固定500/3750图保持；下一本地2500、正式4000。原安全停线、审核门禁及正式恢复状态保留。

## 2026-10-05T22:30:51.057390+00:00 正式3000步验证与保存完成，整体改善与来源差异并存

第3000步由原监督器于2026-10-05T22:26:18.396952+00:00记录完成；原八卡继续5016段，完整25042步未完成。本地当前2100/3750、最近验证2000步；正式和pilot配方独立，不直接比较loss。

| 指标（按token汇总） | 第2000步 | 第3000步 | 变化 |
|---|---:|---:|---:|
| 四档平均AA CE | 0.797064 | 0.773253 | -0.023811 |
| 四档平均AA准确率 | 75.7041% | 76.3480% | +0.6439个百分点 |
| 全遮蔽AA CE | 1.771696 | 1.760729 | -0.010968 |
| 全遮蔽AA准确率 | 41.9116% | 42.1558% | +0.2443个百分点 |
| 四档平均terminal准确率 | 81.3003% | 85.2953% | +3.9949个百分点 |
| 全遮蔽terminal CE | 1.659838 | 1.328663 | -0.331175 |
| 全遮蔽terminal准确率 | 41.4257% | 57.7268% | +16.3011个百分点 |

七来源等权平均AA CE1.260794→1.232357，全遮蔽AA CE1.991063→1.977242；但四档平均terminal准确率84.1577%→82.1536%，其CE0.626438→0.655345。全遮蔽逐来源AA CE在OTS1.901283→1.927323、TCR native1.864420→1.880862变差；terminal准确率在TCR papers91.2409%→69.7080%、TCR repertoire53.5156%→36.7188%变差。保留不同方向的结果，不能把token汇总改善描述为所有来源或生成质量提升。

同一1330条验证记录、四档遮蔽；112计数字段和采样/配置保持，5320次评估非独立样本。Root复算70个聚合值，核对6JSON及归档，8rank参数摘要一致且较2000改变：c4146f35a3c901831aca8c5cfbd258a352397b4d24599eae9d43c93bb825acbd；17原生文件齐全，二进制校验沿用实际保存回调，未重读大权重。默认eval_loss1.923425是全遮蔽来源等权总loss；正例关系token准确率1.0不是正负binding或功能证据。实际从3000恢复尚未发生。

全遮蔽按链首END准确率13.2602%→15.5035%，长度MAE3.310070→2.761715；来源等权分别15.8266%→19.3023%、2.850117→2.396635。未预测END均0/2006。与此同时，token END精确率37.7556%→33.9291%，AA位置误报END比例0.7855%→1.2929%；来源等权误报率则4.8412%→2.5968%，口径不可混用。这是一次argmax去噪前向，不是多步生成评估；真实END位置准确率与首END位置准确率定义不同。新3000点18项诊断复核通过，2000点直接复用已记录证据。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint3000_root_observation.json（SHA88a4133af4a1f1005b84a5133ee7645d59c9ea99f812bb8aa1f99c9899aeff56）及 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint3000_endpoint_diagnostics.json（SHA743b9b80db5eb35acd46ab2798afa4e344fe78bbf8b6c55ea103b2127f834976）。保存现为1000/2000/3000，旧checkpoint-16正常轮替且必要恢复JSON仍在归档，无额外权重副本。继续固定f011/25042协议，下一正式4000、本地2250；无重启、提交或额外模型测试。完整22格和固定500/3750图保持，原安全停线、审核门禁与正式恢复状态保留。

## 2026-10-05T22:01:53.627091+00:00 本地2000步验证完成，AA与结束标记指标改善

本地f000 seed42已完成2000步验证，曝光64000/120000条（53.33%）；完整3750步仍在运行。正式原八卡继续5016段，最近完整验证/保存2000步，完整25042目标尚未完成；两套配方独立，不直接比较loss。

| 指标（按token汇总） | 旧同格2000步 | 本地本轮2000步 | 本轮−旧 |
|---|---:|---:|---:|
| 四档平均AA CE | 0.886866 | 0.886307 | -0.000559 |
| 全遮蔽AA CE | 1.764967 | 1.761686 | -0.003281 |
| 全遮蔽AA准确率 | 41.9721% | 42.2681% | +0.2960个百分点 |
| 全遮蔽terminal CE | 1.266512 | 1.236364 | -0.030148 |
| 全遮蔽terminal准确率 | 54.0357% | 55.5761% | +1.5404个百分点 |
| 四档平均terminal准确率 | 85.2351% | 85.4062% | +0.1711个百分点 |

同一当前实验1750→2000：四档平均AA CE0.907134→0.886307，全遮蔽AA CE1.779121→1.761686、准确率40.4027%→42.2681%；terminal CE1.920001→1.236364、准确率32.4707%→55.5761%，四档平均terminal准确率77.0275%→85.4062%。此前回落后在此点改善，不据单点声称持续改善或改变固定完整epoch。四档平均正例关系CE0.000621→0.000786略升，关系准确率仍1.0；已有正例token指标不能代表正负binding判别或生物功能。

新旧曝光与调度一致：64000条，input24535629、eligible9824278、selected4926301、学习率4.681752459671005e-5。170个records/*_count字段和40个mask_ratio字段相同；1024条验证含37条context。既有metadata helper核对新旧2000点的计数、分子、来源/context分组与聚合，未新增模型运行。provenance SHA保持，顶层差异仍仅checkpoint_saving/source_sha256；不称位级复现或受控速度比较，不将同架构复测归因为架构或ESMC收益。

旧点来自最终失败/未完成尝试，旧曲线到此结束；2250及以后没有旧同seed点可匹配，继续记录本轮真实曲线，不插值/外推。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation2000_comparison_20261005.json（SHA0b7214c684756ff325aeca81365bb3b870cf4dabd3d061d0ad4af90e7c88ce43）；实际曲线 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f000_seed42_attempt1/curve.jsonl，summary与2000点一致，无COMPLETE/FAILURE、checkpoint/tmp。

本回合16次有限观察取得新2000节点，分类progress；原监督/本地身份与8worker心跳再次核实，无abort、重启或重复提交。完整22格仍8完成、1运行、13未启动；固定500/3750图保持，下一本地2250、正式3000。正式恢复状态、原安全停线与审核门禁保持。

## 2026-10-05T20:59:44.248817+00:00 本地1750步验证完成，AA loss与全遮蔽准确率变化不同

本地f000 seed42已完成1750步验证，曝光56000/120000条（46.67%）；当前progress 1750/3750。正式原八卡继续5016段，最近完整保存2000步，完整25042目标未完成；下一节点为本地2000、正式3000。

| 指标（按token汇总） | 旧同格1750步 | 本地本轮1750步 | 本轮−旧 |
|---|---:|---:|---:|
| 四档平均AA CE | 0.905755 | 0.907134 | +0.001379 |
| 全遮蔽AA CE | 1.773030 | 1.779121 | +0.006091 |
| 全遮蔽AA准确率 | 40.7297% | 40.4027% | -0.3269个百分点 |
| 全遮蔽terminal CE | 1.833503 | 1.920001 | +0.086497 |
| 全遮蔽terminal准确率 | 29.2668% | 32.4707% | +3.2039个百分点 |
| 四档平均terminal准确率 | 76.3316% | 77.0275% | +0.6959个百分点 |

同一当前实验1500→1750：四档平均AA CE0.931826→0.907134，全遮蔽AA CE1.782721→1.779121；全遮蔽AA准确率40.9893%→40.4027%，terminal准确率34.0111%→32.4707%，terminal CE1.610497→1.920001。四档平均terminal准确率78.3659%→77.0275%。改善与回落均记录，不以loss下降宣称全面提升；完整3750步尚未完成。

新旧曝光/调度一致：56000条，input21486061、eligible8601527、selected4311743，学习率5.7545315382928044e-5。170个records/*_count字段与40个mask_ratio字段分别相同；1024条验证含37条context。既有metadata helper核对新旧1750点的计数、分子、逐来源/context分组及聚合，没有额外执行模型。provenance SHA保持，差异仍仅checkpoint_saving/source_sha256；不称位级复现或受控速度比较，数值差异原因未确认。

旧点来自最终失败/未完成的历史尝试，不算旧完整epoch成功。正例关系token准确率1.0不是正负binding或生物功能证据；此同架构复测不形成架构收益或ESMC结论。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation1750_comparison_20261005.json（SHAb938476b03478015c2b3b7c6823365affc35857de09722f8a7f9a81919c8d530）；实际曲线 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f000_seed42_attempt1/curve.jsonl，summary与1750点一致，无COMPLETE/FAILURE、checkpoint/tmp。

本回合21次有限观察取得新1750节点，分类progress；原监督/本地进程身份及8worker心跳再次核实，当前无abort。无重启、submit/cancel或训练源码/数据变更。完整22格仍8完成、1运行、13未启动，固定500/3750局部图保持；正式恢复文件、原安全停线及审核门禁保留。

## 2026-10-05T19:52:34.799686+00:00 本地1500步验证完成，继续原3750步完整epoch

本地f000 seed42已完成1500步验证，曝光48000/120000条（40%）；当前progress 1500/3750。正式原八卡继续，最近完整保存2000步，完整25042目标未完成；下一节点为本地1750、正式3000。

| 指标（按token汇总） | 旧同格1500步 | 本地本轮1500步 | 本轮−旧 |
|---|---:|---:|---:|
| 四档平均AA CE | 0.930847 | 0.931826 | +0.000979 |
| 全遮蔽AA CE | 1.782431 | 1.782721 | +0.000289 |
| 全遮蔽AA准确率 | 41.2544% | 40.9893% | -0.2651个百分点 |
| 全遮蔽terminal CE | 1.592687 | 1.610497 | +0.017810 |
| 全遮蔽terminal准确率 | 30.7455% | 34.0111% | +3.2656个百分点 |
| 四档平均terminal准确率 | 76.7469% | 78.3659% | +1.6190个百分点 |

同一当前实验1250→1500：平均AA CE0.961122→0.931826，全遮蔽AA CE1.787297→1.782721；全遮蔽AA准确率41.2351%→40.9893%，terminal准确率36.2292%→34.0111%，terminal CE1.485902→1.610497。不能用loss下降概括全部指标改善；仍按原协议跑完完整epoch。

新旧曝光/调度一致：48000条，input18452209、eligible7384869、selected3707540，学习率6.792509347874236e-5。170个records/*_count字段与40个mask_ratio字段分别相同。1024条验证含37条context；既有metadata helper核对新旧1500点的计数、分子、逐来源/context分组与聚合，无新模型执行。provenance SHA保持，差异仍仅checkpoint_saving/source_sha256，不声称位级复现或受控速度比较。

旧点来自最终失败/未完成的历史尝试，不算旧完整epoch成功。正例关系token准确率1.0不是正负binding或生物功能证据；本次同架构复测不产生架构收益或ESMC结论。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation1500_comparison_20261005.json（SHA4e4afa7f149b2838567c7e3aa8b29eb4861c1f4eaa759f8078d284395e956968）；实际曲线 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f000_seed42_attempt1/curve.jsonl，summary与1500点一致，无COMPLETE/FAILURE、checkpoint/tmp。

本回合有限观察21次自然结束后，继续读取同一真实本地进程，取得1500验证节点，分类progress。原八卡/本地进程身份与心跳再次核实，无abort、重启或重复提交。完整22格仍8完成、1运行、13未启动，固定500/3750局部图保持。正式恢复文件保留，原安全停线和审核门禁不变。

## 2026-10-05T19:32:51.240366+00:00 正式2000步验证与保存完成，AA改善但全遮蔽结束标记退步

第2000步在2026-10-05T19:29:42.200066+00:00由原监督器记录完成；原八卡继续5016段，完整25042步未完成。本地f000 seed42当前1425/3750，最近验证1250步。

| 指标（按token汇总） | 第1000步 | 第2000步 | 变化 |
|---|---:|---:|---:|
| 四档平均AA CE | 0.913768 | 0.797064 | -0.116704 |
| 四档平均AA准确率 | 72.5513% | 75.7041% | +3.1528个百分点 |
| 全遮蔽AA CE | 1.837341 | 1.771696 | -0.065644 |
| 全遮蔽AA准确率 | 40.5602% | 41.9116% | +1.3514个百分点 |
| 四档平均terminal准确率 | 85.1957% | 81.3003% | -3.8954个百分点 |
| 全遮蔽terminal CE | 1.545641 | 1.659838 | +0.114197 |
| 全遮蔽terminal准确率 | 57.1286% | 41.4257% | -15.7029个百分点 |

同一1330条记录、四档遮蔽；5320次记录评估并非独立样本。逐来源全遮蔽AA CE在本次七个来源均下降，但terminal准确率并未一致改善：ASD47.9224%→24.0997%、OAS27.1484%→4.8828%、TCR repertoire78.5156%→53.5156%。这些是当前检查点的观察，原因未确认，不据此中途改架构/配方，也不能称生成能力整体提升。下一正式3000节点继续核对AA与terminal。

8rank参数一致、17原生文件齐全；参数摘要 11e55a4d2e45c6c30a2b63a0977fcead204618be04044fc7bfc5f069445bb468，较1000真实改变，源/数据/调度绑定不变。Root核对6JSON及归档一致、复算70项逐来源聚合与四档均值，112个验证计数字段与1000相同；二进制散列沿用原8rank保存回调，不重复读取模型。实际从2000恢复尚未发生。默认eval_loss1.926993仍是全遮蔽来源等权总loss；正例关系token准确率不是正负binding判别或功能证据，不与pilot配方直接比较。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint2000_root_observation.json（SHAd0d5f91502042a552d60b27e345ca7deb4c174ba0306e5d45581cf48ffe1b654）；详情 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md。此次为新正式训练节点的progress；有限观察自然结束，原训练进程继续。完整22格仍8完成/1运行/13待启动，固定500/3750图不重绘。无重启/提交/额外模型验收，所有审核和安全停线保持；测试无权重/tmp，正式恢复文件正常轮替保留。

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


## 2026-10-05T17:44:15.579408+00:00 本地1000步验证完成

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

## 2026-10-05T16:39:14.429613+00:00 本地750步验证完成，正式训练继续

2026-10-05T16:39:14.429613+00:00 本地f000 seed42已完成750步验证（24000/120000条，20%计划epoch），当前progress为750/3750；完整epoch未完成。正式原8卡继续运行，最新训练loss=0.6997、8worker心跳新鲜、无abort；checkpoint1000已完成验证保存，下一正式节点2000步、本地下一验证1000步。

四档平均AA CE新1.066193/旧1.065402（差+0.000791，约+0.0743%）；全遮蔽AA CE新1.777032/旧1.775615（约+0.0798%），AA准确率新42.0923%/旧42.2900%。全遮蔽terminal准确率新28.4658%/旧31.0536%（差-2.5878个百分点），terminal CE新1.622146/旧1.626278；不因AA指标接近而隐去结束标记差异。四档平均terminal准确率新70.2284%/旧71.5257%。

样本/输入token/eligible/selected token/学习率一致，170个records或*_count字段及40个mask_ratio字段分别核对一致；1024条验证含37条context。逐来源和context分组的计数、分子与聚合已用既有metadata helper核对，没有重跑模型。provenance SHA与500步原件保持，差异仍为checkpoint_saving/source_sha256；不称位级复现，数值差异原因未确认，不归因架构收益、运行速度或功能改进。旧同格为未完成任务的750步历史观察，不作为旧完整epoch终态。原f011初步选择与当前训练保持。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation750_comparison_20261005.json（SHAd33170815d60b977ebe735cbc1926b189442bee02bd0d2b7246ef6cbf2742bae）；实际曲线 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f000_seed42_attempt1/curve.jsonl。summary与750末点一致，无COMPLETE/FAILURE、测试checkpoint或tmp。固定500/3750局部图保持原观察点，不重复渲染；完整22格仍8完成、1运行、13待补。

本回合在5次具体进程身份观察后取得新的750节点证据，分类progress；原Goal保持active，未重启/提交/重复GPU验收或改动训练源码/数据门禁。

## 2026-10-05T16:30:57.845872+00:00 正式1000步验证与完整保存完成

2026-10-05T16:30:57.845872+00:00 checkpoint1000已实际验证并完整保存：8rank参数一致、17原生文件齐全，元数据与监督归档一致；长段5016仍运行，完整目标25042。八worker新鲜、无abort，监督与本地PID/start_ticks/argv匹配。本地f000为725/3750步、23200/120000条，最近验证500步。下一正式节点2000步，本地750步。

1330条记录×四档遮蔽：按token汇总的四档平均AA CE0.913768/准确率72.5513%，全遮蔽AA CE1.837341/准确率40.5602%；七来源等权的对应CE为1.435434/2.096753。默认eval_loss2.026115是全遮蔽来源等权总loss，不能混用。已有正例关系token准确率不代表正负判别，正式结果不直接与pilot消融拼表。

详情 [V7_FORMAL_TRAINING_RESULTS_20261005.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md)；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint1000_root_observation.json；JSON归档 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint_evidence/checkpoint-1000。本回合从25次真实进程观察转为新1000节点证据，分类progress，原Goal继续active。保存完成不等于从1000实际恢复或5016/25042终态；原8→16恢复验收保持，不重复GPU检查或重启/提交。

## 2026-10-05T15:55:59.001982+00:00 当前训练与对比报告

2026-10-05T15:55:59.001982+00:00 实际核实：原八卡任务t-20261005205851-lqfjg为Running，8worker心跳新鲜、无abort；监督与本地进程的PID/start_ticks/argv匹配。正式最新训练loss=0.7595，长段目标5016/完整horizon25042保持；1000步验证和保存尚待产物。本地f000 seed42为575/3750步、18400/120000条，最近验证500步、下一750步，无COMPLETE/FAILURE或测试checkpoint。

[500步局部图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_v1/step500_architecture.png)（9/22），[3750步完整epoch局部图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_v1/step3750_architecture.png)（8/22）。两图均明确PARTIAL，分别展示mask_mean和mask100，每seed一个点；13项尚未启动，本地f000仍在跑。SVG/PDF、CSV/JSON及实际渲染凭据均在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_v1；根端复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_v1/root_review.json。

## 2026-10-05T15:34:01.700681+00:00 本地500步验证完成，原八卡训练持续运行

本地f000 seed42实际500/3750步、16000/120000条，1024条验证中含37条context。与旧同格500步相比，样本/输入token/eligible/selected token/学习率及各验证分母一致。四档mask平均AA CE新1.185176/旧1.184855（差+0.000321，约+0.0271%）；全遮蔽AA CE新1.793335/旧1.792839（差+0.000496），AA准确率新40.2420%/旧40.4412%（差约-0.1992个百分点）。数值接近但不逐位相同，不作为架构收益或完整epoch结论。

500步全遮蔽terminal准确率新旧均48.0591%，250步该指标的差异在这一观察点没有延续；原因仍未确认，不据此宣称运行逐位复现。全遮蔽terminal CE新1.301628/旧1.306732；四档平均terminal准确率新80.6511%/旧80.3335%。已有正例关系指标不代表正负判别能力。本地与旧池wall-time环境不同，不据此作架构吞吐比较。完整3750步和最终消融图仍未完成，原f011初步选择保持。

原始比较 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation500_comparison_20261005.json（SHAbecc8fd766ecd1bffde60e8dee20ac8c46f0ddd5e35908675f90adfec5aaadd1）；直接曲线 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f000_seed42_attempt1/curve.jsonl。新summary与500末点一致，原生COMPLETE尚无，测试目录无checkpoint/tmp。原正式监督与本地进程的PID/start_ticks/argv均匹配；8worker心跳新鲜、无abort，最新正式训练loss=0.836，下一正式eval/save1000步尚待实际产物。观察session94960完成35次只读核实后自然退出，训练/持续监督未重启。原Goal继续active；本回合有新的500步验证证据和汇总接入问题的记录，没有新训练提交。

## 2026-10-05T14:28:32.090226+00:00 本地消融完成首次250步验证，八卡正式训练持续运行

本地f000 seed42已实际验证250/3750步、8000/120000条；全遮蔽AA CE为1.799719、准确率41.2489%，四档mask平均AA CE为1.538930。旧同模型/同seed在250步对应为1.798783/41.1658%及1.537953；全遮蔽AA CE差+0.000936（约+0.0521%），不能把四档平均与旧全遮蔽直接比较。样本曝光、输入/eligible/selected token数、学习率及每档验证计数一致，provenance差异仅checkpoint_saving和source_sha256；已匹配协议，不宣称逐位复现。初始全遮蔽AA CE差+0.000136；250步terminal准确率旧12.6309%/新23.5983%，terminal CE旧2.418198/新2.385153，数值差异来源未确认，不归因于架构收益或声称全部指标已复现。继续原3750步完整epoch，不修改或重启当前实验。

远端原任务t-20261005205851-lqfjg仍Running；观察2026-10-05T14:28:11.098227+00:00中8worker心跳新鲜、无abort，最新正式训练loss=1.4282。监督PID3977587与本地PID3970535的真实argv已核对，监督start_ticks一致；当前5000新增步段仍在运行，下一常规eval/save为1000步，已完整保存并验收的最近段为16步，不把目标5016当成完成。监督器仍是后续接段唯一执行者，原Goal保持active；本回合是运行证据更新和同口径结果比较，没有新平台提交或GPU验收。

比较原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation250_comparison_20261005.json（SHA3d236866d90b79a7d91ada369e314c429bdb8b2736726e664050482f1e35eedc）；实际原曲线 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f000_seed42_attempt1/curve.jsonl。测试目录仅日志/指标/来源记录，无checkpoint或tmp，正式恢复状态保持。完整22格仍8完成、1本地运行、13未启动；本地结束后优先f010 seed42/43。

## 2026-10-05T13:45:46.163722+00:00 单实例持续监督已实际启动，八卡训练继续

监督PID 3977587已由本机/proc核实PID/start_ticks/argv，固定源码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/source_v1/supervise_v7_formal_20261005.py（SHA ea7287182faa6a8f6a6496dc1e7bc7dd61c034c65a2949e0b2796efce4bb32a4）；每30秒读取当前8卡任务，每5分钟只读查询精确平台任务。当前平台Running、8worker心跳新鲜、coordinator无abort，最新正式日志loss=2.7233；本地f000到100步/3200条、loss=2.336218，无测试模型或tmp产物。上一Goal回合为progress（实际八卡原生恢复与长段启动）；本回合为progress（监督接续实际运行），原Goal保持active，完整训练/消融尚未完成。

监督器为后续接段的唯一执行者：当前5016段结束后，只有准确8票据成功/rc0、8rank清理、原生恢复、SEGMENT及最终完整checkpoint全部核实才调用新冻结helper，依次至10016/15016/20016/25016/25042。运行期间其他agent不得手工重复接段；失败即停接续，保留现场。新helper /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/supervisor_launcher_v1/launch_v7_formal_segment_20261005.py 仅增STOP发布前检查与admission_commit；提交前STOP零入队，commit后完成整组8票据并停止后续段，不自动取消当前段。

原监督reader缺失文件改为明确FileNotFoundError，让正常checkpoint轮替/暂时观察失败不被误记训练失败；身份/binding/SHA/ack错误仍硬停。新边界定向控制及真实16步metadata读取消费通过，未重复GPU验收。checkpoint-16的轻量JSON已实际归档供后续恢复证明比对；原训练参数、611来源、2/30/180、数据审核和保存恢复检查均保持。

动态实际状态：/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_SUPERVISION_20261005.md；进程原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_process_identity.json、启动原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/popen_receipt.json、边界检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/control_delta_receipt.json。测试临时fixture均已清理，正式checkpoint保留原策略。未submit/cancel平台任务，也未重启现有容器/worker/监听。

## 2026-10-05T13:40:03.875317+00:00 持续监督与接段控制已实现，尚未启动监督进程

原正式5016长段继续运行；新监督器 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/supervise_v7_formal_20261005.py 只观察已完整发布的checkpoint sidecar，并归档JSON/每段metrics防止正常轮替覆盖证据。只有准确8票据succeeded/rc0、coordinator成功、8child清理、SEGMENT/COMPLETE、8rank原生恢复证明和最终checkpoint共同成立，才调用原冻结helper接续10016/15016/20016/25016/25042；既定来源、审核、batch、25042调度和5000单段上限保持。失败/STOP停止接段；观察失败重读同一任务，不推断训练退出，不重启或重复submit。

5项针对新监督器的metadata控制通过，包括真实8→16证据读取，以及失败rank/abort/缺rank的拒绝；没有重复模型或GPU验收。首轮发现并修复运行worker使用heartbeat而非time，以及自写JSON摘要格式与原driver不同，已改为直接复用原driver digest。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/control_receipt.json，临时metadata fixture全部已删除。此时监督器尚未启动，不把准备当运行。

## 2026-10-05T13:32:42.384239+00:00 正式长段八卡实际运行中

累计16步的训练、完整保存和原生恢复已通过；当前5000新增步（累计目标5016）8个worker实际running、8注册、无abort，最近实际GPU快照8卡均100%利用率，正式作业持有这些GPU。本地f000独立消融已到50步/1600条，loss=2.837418。长段与完整3750步消融均尚未完成；未新增平台提交、重启或重复验收。当前运行原件：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_5016/runtime_observation.json。后续按每1000步真实eval/checkpoint记录关键点，5016成功后按CURRENT接续到完整25042。

## 2026-10-05T13:29:43.942052+00:00 八卡原生恢复验收通过，继续5000步正式训练

8→16实际恢复成功：八个pool结果succeeded/rc0，8rank的native optimizer/scheduler及本rank RNG在返回点与保存值逐项digest一致；随后真实更新8步，八卡参数一致且相比第8步已变化，checkpoint-16原生17文件完整。未证明与不间断训练逐位轨迹等价。恢复原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_16_attempt2/root_segment_acceptance.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_initial_training_20261005/run/RESUME_8_TO_16.json。修复后启动记录没有锁超时（有界时序记录最大hold约0.030秒）；这次运行成功不代表已直接证明旧超时的内部原因。

随后5000新增步已登记，累计目标5016，/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_5016/enqueue_receipt.json。实际观察 {"at_utc": "2026-10-05T13:29:43.942052+00:00", "phase": "started", "terminal": null, "abort": null, "registrations": 8, "next_cumulative_step": 5016, "new_updates": 5000}；f011/global256/25042固定horizon、每1000步eval/save保持。当前完整完成16步，不把入队的5016写成完成。下一成功段后再继续10016，直至25042。8卡用于联合训练，本地f000完整消融独立运行。

续训调用耗时79.8298秒，新增8步/秒=0.100213，名义25.655records/s，含评估保存；HF按全horizon计算的吞吐不采用。HF4.48.1原train_loss=1.683202的分母为累计16步；本段实际8个新更新的rank0均值为3.366403，不能据原数字声称loss减半，也不称全局均值。正式save_total_limit=3正常轮替权重，前8/16的JSON与恢复证据已另归档 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/first_resume_evidence，没有额外复制模型/optimizer/RNG。

## 2026-10-05T13:25:17.487963+00:00 修复后从checkpoint-8开始第二次续训尝试

新8票据已入队，实际状态 {"at_utc": "2026-10-05T13:25:17.487963+00:00", "phase": "started", "terminal": null, "abort": null, "registrations": 8, "exits": 0, "pending": 0, "running": 8}。旧失败组保留且八个子进程清理已核实；新group使用独立stage，checkpoint-8、冻结训练源、数据及固定调度不变。预导入CPU检查通过，未初始化CUDA/NCCL，原已有函数仅rank_child增加预载及abort复查；完整GPU恢复尚待实际结果。

原件：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_16_attempt2/enqueue_receipt.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_16_attempt2/plan.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/transport_preload_v1/preload_check.json。后续5016段必须显式指定新transport并将 --previous-stage 指向实际成功的 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_16_attempt2，不能使用旧失败segment16。当前本地f000已实际到25步、800条、loss2.870623，3750步未完成。

## 2026-10-05T13:24:26.771800+00:00 续训协调启动修复（尚未GPU验收）

新 transport 位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/transport_preload_v1/formal_pool_transport.py，只把第三方冷导入移到注册和心跳线程之前，并在导入后再次检查已有abort；先执行原冻结driver的source隔离，不构造模型/数据集或初始化CUDA/NCCL。原协调器、原NCCL入口与父监督SHA不变，2秒锁/30秒心跳/180秒启动不变。新launcher支持独立attempt和明确previous-stage，先验证旧失败组八个进程均已退出，拒绝覆盖旧stage；正式611来源、数据、batch、调度和checkpoint绑定不变。实际检查和差异原件位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/transport_preload_v1；GPU恢复仍须新实际8→16结果，不能根据CPU预导入检查称为通过。

## 2026-10-05T13:22:33.958207+00:00 续训启动失败已核实，真实累计仍为8步

8→16第一次续训组在协调启动阶段 aborted；8个pool结果均failed，8个child已清理，尚无 SEGMENT_16 或 RESUME_8_TO_16。正式 checkpoint-8 完整保留。此前标题“恢复段在运行”已过时；下文保留历史观察。首次持久abort由rank0/PID4821 heartbeat等待锁2.000061秒触发；rank5 heartbeat临界区持锁2.098672秒，但commit仅0.005539秒。冷导入Torch造成持锁线程停顿目前只是推断。正在准备新transport把重型依赖导入移到注册/heartbeat之前；2秒锁、30秒心跳、180秒启动和审核/来源/abort检查全部保持。旧失败组不重放；修复后用新group/新job ID从真实checkpoint-8续训，不重启容器。

失败原件：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_16/root_failure_observation.json；协调时序：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_16/meeting/coordinator。本地f000完整3750步消融独立运行，完成状态仍待实际结果。

## 2026-10-05T13:17:42.764234+00:00 从实际 checkpoint-8 恢复到16步的正式组已入队

8个新成员已一次登记，复用同一f011/seed42/global256/25042固定调度，累计目标16、新增8步，全部绑定真实checkpoint-8。此前8个pool结果成功及完整原生保存已经核实；当前新组状态 {'at_utc': '2026-10-05T13:17:42.764234+00:00', 'phase': 'terminal', 'terminal': 'aborted', 'abort': {'abort_request_sha256': '1f541210a2da2bdffc5670e52c2a2660199e5818174ae5333ab9be268d3858bb', 'at_monotonic': 24467932.983603936, 'reason': 'Coordinator lock wait expired'}, 'registrations': 8, 'exits': 8, 'pool_running': 0, 'local_summary.json': {'step': 0, 'records': 0, 'train_seconds': 0.0}}。恢复是否通过须以未来真实 RESUME_8_TO_16.json、八rank结果及续步为准，不能据入队或存在checkpoint宣布通过。

登记原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_16/enqueue_receipt.json、实际观察 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_16/runtime_observation.json。登记原件中的 formal_training_started=false 仅代表登记当时尚未观测该段优化步，不是整体当前运行状态。当前正式程序及本地消融均有实际运行handle，不重放或重启。


## 2026-10-05T13:15:45.137987+00:00 真实八卡首段8步完成，单卡补消融已并行启动

8个正式pool result均 succeeded/rc0，8rank实际参数digest相同，checkpoint-8包含模型/Adam/scheduler和8份rank RNG，SEGMENT_8为真实累计8步。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_8/root_segment_acceptance.json；保存恢复只完成“保存”，下一段必须实际恢复至16步后才能记原生恢复通过。rank0 train_loss=3.7763006687164307（本rank训练均值，非全局重算均值），训练调用实际耗时100.266秒。HF因固定horizon提前停止而打印的全量吞吐不能使用；8步/实际耗时=0.079788 updates/s，名义2048记录/100.266秒=20.426 records/s（含该调用内评估/保存）。

本地A100独立新 f000 seed42 已启动，PID3970535实际存活、argv匹配；相同原worker解释器 Torch2.8.0/Transformers4.48.1/NumPy2.1.3，源另冻结为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/source_snapshot_v1.json。120k/3750步、micro32/GA1/global32/eval4，metrics-only，不resume旧失败、不覆盖原账本；实际初始化SHA当前为 44ff26ac8144ece78c84d9660e2d40d8f0746fa4e555a9ae2afeaadbd7d4b6a2，完整epoch尚未完成。启动原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/launch_receipt.json。新八卡继续正式训练，本地消融并行，不再新申请GPU。


## 2026-10-05T13:11:04.146916+00:00 f011 首段八卡联合训练已实际入队

新容器 t-20261005205851-lqfjg 内已一次登记8个正式成员，初次观测 pending=0、running=8。初步f011（R26+R24）、seed42、每卡micro32×GA1×8=global256；同一完整horizon25042、LR1e-4/cosine、warmup1000、每1000步评估/保存，首次实际累计终点8步。正式目标约一遍6410749条训练记录；drop_last及epoch尾部实际曝光以Trainer记录为准，不虚称完全无重复/无遗漏。通过首段完整保存和8rank退出后，恢复至16步，原生optimizer/scheduler/每rank RNG及真实续步须通过原冻结driver的检查。

正式数据为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002，原v3/611来源与限定scope双证书保持并在runtime再检查。当前正式采用自然epoch随机采样（context约4.65%），不是pilot的20.54%context配方；context序列重建CE为0，已有正例关系token监督保持，无新增负例。初步选择依据为实际完成两seed f001/f011，不宣称完整22消融已完成。

计划原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_8/plan.json，8个具体job ID在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_8/enqueue_receipt.json；新容器实际parent七字段/完整argv已由唯一metadata-only job成功捕获。复用旧已冻结slot wrapper/transport与原正式Trainer，没有更改模型、数据审核或保存恢复门禁。正式输出独立为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_initial_training_20261005/run，保留原生恢复checkpoint，测试清理器不覆盖它。

目前仅确认实际入队/worker进程，真实优化步、参数同步、保存和恢复尚待日志/产物验证；不得记作验收通过。未新submit/cancel平台资源或重启容器/worker/监听，停止注释和独立CUDA专项暂停保持。上一Goal回合为progress（新容器实际Running且8worker就绪）；本回合推进正式任务实际启动，原Goal保持active。


## 2026-10-05T13:06:30.253153+00:00 新容器正式入口接入，真实父进程身份采集已入队

新池8个worker均实际待命且心跳<15秒。已冻结新薄入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/entry_20261005/pool_formal_20261005_dispatch.py，仅重绑定新pool/catalog/8别名，复用原冻结正式 prepare/main、wrapper、transport及v3/611训练来源；旧未知任务分发保持。三项必要冷检查通过，无模型/GPU重复验收。

只入队一个实际容器元数据采集 job `1791205590250592385_p06_v7_pool_identity_bed2f0ab`，在新容器读取真实parent PID/start_ticks/boot/argv，结果用于绑定八卡正式计划；它不执行Torch、模型、CUDA专项或读取训练分片，不能记为GPU训练通过。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/entry_20261005/publication.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/entry_20261005/identity_enqueue.json。未新增Volc提交或重启资源；正式模型仍未启动。


2026-10-05T13:01:01Z 实际核实8个worker均 standby_load / maintenance_only，GPU编号0–7齐全，心跳约4秒内；pending=0、running=0。八卡容器已就绪，正式训练和消融尚未开始。原件：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20261005/resource_ready_observation.json。

## 2026-10-05T13:01:01.223411+00:00 资源已申请

新任务 `t-20261005205851-lqfjg` / `immune-v7-pool-8gpu-20261005` 实际平台状态 Running。c20250601 非闲时单实例8卡，配置 /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_pool_c20250601/eight_gpu_pool_20261005.yml，新池 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20261005。用户本轮要求先申请容器；初始化为空实验队列，不自动运行旧测试，正式训练/剩余消融尚未启动。实际 worker 就绪以 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20261005/resource_ready_observation.json 为准，后续训练入口仍须完成原门禁接入。下文“尚未提交”仅为先前准备记录。


更新于 2026-10-05。初步版本选用 `f011`：R26 链内局部/全局注意力交替，加 R24 attention/FFN 固定 0.1 残差缩放。no-ESMC、随机初始化，A4 grammar，8 层、hidden 768、12 heads、FFN 3072、最大渲染长度 2048。token/role/scope embedding、每链重置 RoPE 和同链/其他位置 bias 保持。target 仍有 full/fragment 256/64 容量及 `<protd>`；本版没有宣称已取消固定容量或结束标记。

## 选择依据和范围

原新大 batch 组 `f001` 与 `f011` 都完成 seed42/43 各 3750 步、120000 条一 epoch。氨基酸重建 CE 两 seed 均值分别为 0.8085597562491476 和 0.8027943049451316，组合比纯 R26 低约 0.71%。全遮蔽准确率均值分别约 43.02% 和 42.93%，差约 0.09 个百分点；这是重建代理指标，不是实际生成质量。context 验证只有 37 条既有 binding 正例，没有显著性或正负判别结论。

这是按用户最新要求敲定的初步训练版本，不是完整 22 项消融的最终赢家。完整矩阵保留待办；后续补充新独立实验必须保留原失败/中断记录，不能拼接没有 checkpoint 的中断轨迹或覆盖旧 job。

## 初始准备时状态（历史，当前见顶部）

原资源 `t-20261001033327-fmnql` 平台直接查询为 Failed，End=2026-10-03T19:24:33Z，8 个 worker 均 exited。新大 batch 原 22 项中，8 项成功完成、3 项 checkpoint 写入失败、8 项因资源时限中断、3 项仍为从未启动的 pending 文件。不存在在跑的原八卡训练；正式训练 catalog 尚未创建，八卡联合更新与原生保存/恢复尚未实际验收。

## 训练配方与执行安排

用户已授权申请新的 c20250601 非闲时八卡容器，进入其中运行初步 V7 训练，并在正式训练期间继续消融。本机已只读确认有一张 A100-SXM4-80GB；后续消融可在本机单卡依次执行，和八卡联合训练并行。新资源必须使用独立任务和输出目录，不重启旧资源，不重复提交。

正式数据使用已有审核的 V7 派生目录 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002`。原只读门禁与来源清单 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_manifest_v3.json` 保持；实际启动时仍须核实 ready、两份限定范围审核、来源字节及真实八卡环境，不创建或伪造证书、不扩大审核结论。context 序列不计算重建 loss，已有监督 binding 正例关系 token 纳入训练，不新增负例。

## 保存策略

测试只保存指标、曲线、日志和必要来源证据，不保留模型或临时 checkpoint。2026-10-03 的 metrics-only 策略已覆盖之前保留 p04 latest.pt 的旧策略；原失败测试没有可用恢复 checkpoint。正式训练保留原生模型、optimizer、scheduler 和每 rank RNG，供继续训练；正式输出必须与测试清理目录分开。

停止的序列注释分支不得恢复、重试、转交或绕过；独立 CUDA 算子专项暂停保持。初始准备阶段尚未启动的陈述仅代表历史；当前真实进展以顶部时间戳及原始结果为准，原Goal未完成。
