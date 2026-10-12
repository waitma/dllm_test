# 架构探索索引与验证状态

## 2026-10-03T04:02:11.291493+00:00 新统一batch首250真实结果

2026-10-03T03:59:36.506698+00:00只读核实：新组8 running/14 pending，首8均325/3750步；实际共同完整评估为250，每项8000/120000条曝光（6.67%）。统一micro32×GA1/global32、eval4及原3750完整epoch目标保持，尚无新组终态。

真实8/22部分概览 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/v7_large_batch_ablation_overview.png；R20定义、四条件结果和判读边界见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ARCHITECTURE_INTERIM_DECISION_20261002.md。这是新组四条件×两seed的早期重建结果，8/22 partial、final=false；另14项仍待运行。组合f011的汇总AA CE最低，但全mask准确率未显示同步改善，且仅37条context验证上的AA CE更高；不能认定生成、多链理解、二分类或生物功能提升，不能据250步选正式模型。两个seed不构成显著性证据；旧micro4×GA8结果独立保留。

真实250报告及12图已逐字节独立留存 [/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/report.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/report.json)；[实际概览图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/v7_large_batch_ablation_overview.png)；[渲染/归档记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/render250_20261003.json)（SHA41f067ee719394b508be77c97080b01e06bc367a7d203cb987672e5a1ad981f3）；[两seed派生](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/current_method_comparison_shared250_20261003.json)（SHA514bf2eba5a2ac69603e08e31373a10421690ba740451b260fc5851441863965）。未重绘、未重复模型验证或读取二进制checkpoint。

## 2026-10-03T03:14:48.193324+00:00 最新真实状态：完整22项已登记，首八项已在原八卡运行

2026-10-03T03:10:52.992945+00:00为8 running/14 pending，首批R20/R24/R26及组合各两seed；统一micro32×GA1/global32、eval4、原120k/3750完整epoch。实际计划与固定账本已存在，尚无新组训练结论或完整图。发布/登记/初始原件依次 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v2/publication.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v2/enqueue_receipt.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v2/initial_observation.json；新启动源清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/launcher_source_v2.json，原568模型不改。有限只读监督的后续记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LARGE_BATCH_RUNTIME_WATCH_20261003.md。正式八卡联合训练/恢复未启动，原Goal active、数据/来源/恢复门禁与全部停线保留。以下各带日期段是历史状态，当前以本节、PROJECT顶部和CURRENT为准。

## 2026-10-03T02:36:47Z 最新：原八项完整epoch与容量v2实际核实

原p02八项完整120k/3750核验结束，设计仍8/22 partial；R20基线AA CE0.825971、R26 0.812045（低1.69%），详细定义与真实终态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ARCHITECTURE_INTERIM_DECISION_20261002.md。原p04容量失败保留，独立v2九格已真实成功并根端核实40加载源，micro32三代表有15%余量；协议/单次原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_CAPACITY_TIMEOUT_REVISION_20261003.md。下一32×1统一plan/22账本及新消费者尚未发布，正式联合8卡/恢复仍待，原Goal active。停止注释、独立CUDA暂停、数据门禁和实际恢复checkpoint保持。

## 2026-10-03T01:27:30Z 最新：恢复观测v3已冻结/新来源门禁通过

新增CPU5/5首次通过及实际源码/归档/日志已根端审阅，611文件v3冻结、原helper新来源一次passed59.723秒；真实world8更新/恢复仍待，不重跑旧验收或改数据证书。当前原八项继续3750，共同3000真实AA CE均值R26对应0.826145较同轮基线0.841631低1.84%；未选最终赢家。入口/原件优先读 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md CURRENT及 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_RESUME_OBSERVATION_20261003.md，下方带日期段保留历史。

## 2026-10-03T01:10:58Z 当前接续索引：共同3000新图与原生恢复观测

原Goal实际active，final V7现行限定范围ready/双证书与原已执行门禁一致。p01八项完整epoch全完成，旧p02仅八项继续3750、十四项启动前撤下；共同3000/96000曝光新图已由监督者实际更新，仍partial。唯一容量任务仍等自然空卡，没有推荐microbatch、新22项账本或正式训练启动。

原生恢复薄观测完成一次必要CPU5/5，根端审阅/新冻结待完成，未实际八卡恢复。最新接续读 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md CURRENT块、/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 顶部，再读 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_RESUME_OBSERVATION_20261003.md 和具体job原件。下方日期条目是历史，旧blocked/未审核版本不覆盖当前状态。

## 2026-10-02T23:01:46Z 当前架构判断与新绘图接续

完整首轮优先保留R24固定残差缩放及R26局部/全局交替；新增context第二轮的共同1000步f001（R26同机制）目前 token CE最低，尚未完整epoch或确定最终架构。两seed均值、同轮变化、来源宏平均及后续优先级已实际记录在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ARCHITECTURE_INTERIM_DECISION_20261002.md。

新大batch绘图源码及5项人工metadata/Matplotlib检查完成，已独立冻结两个绘图文件；清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/analysis_source_v1.json。旧plotter仅一行可选标题扩展，旧默认行为不变。尚无新组实际曲线/数值图；容量测量唯一job仍等原卡自然空出，新22项和正式八卡共同训练未启动。源/数据门禁、注释停线、CUDA暂停保持。

## 2026-10-02T22:52:04Z 最新接续入口：容量测量已登记，待原卡自然空出

原任务 t-20261001033327-fmnql 经实际平台查询仍 Running。一次人工输入的容量测量已接入并登记：job1790981433321562451_p04_v7_capacity_febb28e3，账本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_job.json；当前 pending1/running8，测量尚未执行，没有实测推荐 microbatch。首轮 p01 已完成，旧 p02 八项继续原配置，十四项启动前撤下。新 22 项大 batch 训练及正式共同八卡训练均未启动。

容量入口新的四组标准库控制通过、568 源及两份 CPU 证据匹配；公共接入仅增加一个名并保留原 100 名，旧冻结模型源保持。进度/实际接入原件见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md；测量详情与有限监督见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LARGE_BATCH_PROFILE_20261002.md。新组独立消融绘图接入正在实现，需校验统一 batch/eval4/plan 与独立账本，尚未验收。

原 Goal active，保留已有停线、数据门禁和 CUDA 暂停；不重复登记此测量、不补回旧十四项、不重启原资源。后文各日期状态属于当时事实，不能用旧“未发布”或“pending0”覆盖当前状态。

## 2026-10-02T22:39:58Z 最新接续入口

首轮 p01 八项均已完成 100000 条、3125 步完整 epoch，最终图已生成。原 p02 当前八项为 775–850/3750 步，另外十四项已按用户增大 batch 的要求在启动前撤下；当前池 pending=0、running=8。后文“p01仍运行”“p02等待22项”等均是旧时间点记录。

- 实际进度及只读证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md，快照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/project_status_sync_20261002/20261002T223958Z.json。
- 大 batch：新入口、11 YAML、CPU 检查与 568 文件冻结完成；micro16/32 的 GPU 显存/吞吐尚未测量，档位未选、新组未入队。详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LARGE_BATCH_ABLATION_20261002.md 和 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LARGE_BATCH_PROFILE_20261002.md。
- 八卡共同训练：driver v2 冻结及既有父数据门禁核验通过；新增 pool wrapper 8/8、调用入口 4/4 人工 CPU 检查通过，但尚未部署、实际八卡联合更新/保存恢复未验收，正式训练未启动。详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_POOL_INTEGRATION_20261002.md。

后续先等待原卡自然完成并取得真实容量证据，再确定新组 batch；不得把不同 batch 协议混成架构对照，不把 CPU 检查写成 GPU 成功。用户选择的已有 binding 正例/context 序列零重建 loss 保持；原 Goal、注释停线、独立 CUDA 暂停和数据门禁保持。本次只核实并同步对应文档，没有操作队列或重启资源。

## 2026-10-02T21:35:07.714476+00:00 最新接续入口

p01八项2800–2875/3125、共同2750验证，无失败；p02的22项新120k组合/CV实验等待原卡完成。用户确认已有binding正例，不做正负判别；11份配置/冻结入口一致。正式driver发现并修复eval丢尾批，新2项worker CPU人工覆盖通过；transport新8项人工协议通过。源码v2已冻结、原helper新版绑定真实passed，但实际8worker预留/发布接线及联合更新/保存恢复尚未完成，正式训练未启动。后续按 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_EIGHT_GPU_ENTRY_20261002.md 接续。原Goal保持active；保留全部停线/暂停及数据门禁，勿重跑已有验收或改动原任务。


## 2026-10-02T21:07:58.922239+00:00 最新接续入口

p01原八项继续3125步，完整终态尚未到；p02已在同一池实际排入22项120k/3750组合与CV对照，尚未有GPU结果。正式driver已实现，隔离只读原审核真实通过，但世界8联合启动与保存恢复未验收，不能标正式训练完成。下一动作与证据见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_CONTEXT_FACTORIAL_ABLATION_20261002.md、/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_EIGHT_GPU_ENTRY_20261002.md。原Goal保持active、未新建或改目标，安全停线及独立CUDA暂停保持。

## 2026-10-02T20:26:02.091441+00:00 当前交接状态

原八项V7真实架构比较运行至1400–1475/3125步，继续原100k完整epoch；独立新增context正例20k子集已生成并通过加载及逐行mask检查，train120k/valid1024，新组3750步尚未启动。原运行中数据与源码不改。用户已确认仅已有binding正例，context无序列CE、binding关系监督保留；完整记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_GRAMMAR_CONTEXT_LOSS_20261002.md。Goal工具本轮只读返回active，未创建、重置或修改其状态；下方blocked等带日期条目属于历史。

## 2026-10-02T19:52:22.715860+00:00 grammar/context loss核验及context数据增强计划

冻结A4的101024行loss资格核验通过：train/valid context位置697220/5852，eligible与有效label均0；关系监督资格4648/37，实际标签全部binding正例。用户明确要求保留关系监督并增加context，计划原100000追加20000既有context监督记录、原valid不变；尚在实现/未抽样/未启动新组，不修改当前八项。完整epoch索引与第1000步断点接续控制通过，3125步恰好覆盖100000索引。详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_GRAMMAR_CONTEXT_LOSS_20261002.md。

多图/多帧新增阅读与三份固定作者源码已归档；优先研究分段全长/global交替及既有延后融合，其次target/context cross-attention，保持context无序列CE与明确关系监督。新方法未实现/执行；研究取舍见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/CV_MULTIVIEW_VIDEO_ARCHITECTURE_20261002.md。

以下为历史阶段记录，旧数据缺审核状态不代表当前已物化并核验的最终V7版本；当时Goal工具为blocked；不能用该旧快照代表当前状态。

## 2026-10-01T11:47:52.931165+00:00 三方架构决策完成

三个用户授权subagent只读决策已完成：保留R20/A4主干，先单独比较R21/R24并以R26作成本对照；联合任务G02末端adapter为条件候选，G01降为读出/优化控制，硬瓶颈和EMA暂缓。8卡4方法×2seed及后续联合组设计已记录，未启动实验。原指定数据缺双证书/ready绑定，真实小batch未运行；原Goal与资源、门禁、注释停线和CUDA暂停保持。

[决策与具体测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_CHANGE_DECISION_20261001.md)、[三方审查摘要](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_architecture_decision_20261001/agent_review_summaries.json)、[禁用决策矩阵YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/decision_matrix_20261001.yml)。新增发现：G01当前任务仿射可吸收到通用原型独立Linear头，不能预称主干任务特化；G04/G05按role而非任意段ID隔离且关系头存在池化旁路；正式R20输出untied但尚无G00式联合分类头。审核范围计划与旧执行scope还需一致的新版本绑定，本轮不改gate、不签证。以下带时间条目是阶段历史。

## 2026-10-01T11:29:06.399847+00:00 当前GPU工程准备与正式数据选择

通用GPU引擎CPU验收完成：两环境各44个唯一案例分三阶段覆盖，未完整重跑最终44项；修正4项工程边界，固定15文件源码包与8份禁用YAML核对通过。清理50个自有CPU状态文件，验证checkpoint=0。现有8卡执行端尚未接入G入口，新增GPU任务0。用户指定immune_v7_mhc_sequence_v1仅metadata核实，缺双证书且无training_ready=true，真实小batch未启动。

[当前GPU工程协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_JOINT_GPU_SEGMENT_PROTOCOL_20261001.md)、[CPU验收原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/validation_profile_count_fix.json)、[固定源码包及核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/source_package_review.json)、[指定数据门禁](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/requested_training_data_gate.json)、[当前配置SHA](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/updated_plan_manifest.json)。原Goal保持active；保留原注释停线、真实数据审核与独立CUDA暂停条件。正式架构比较优先使用审核通过的指定训练数据小batch及实际模型输入/目标/评价；人工整数仅作工程控制，不替代正式效果。下一步完成原8卡执行接入、独立结果复核与自有续接状态清理工具，不重跑已通过检查。

## 2026-10-01T09:37:59+00:00 当前接续

**G00–G07八个通用整数联合原型已实现，CPU功能验收完成，GPU入口与计划500步曲线待完成**。[原型机制、参数量、实际步数、失败修正与接续](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_CV_JOINT_PROTOTYPE_20261001.md)；[CPU验收原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_prototype_cpu_20261001/validation.json)；[当前8份配置](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_cv_joint_exploration_20261001/README.md)。两环境各35个唯一案例覆盖，worker首轮1项汇总失败的定向修正保留原件；8配置各3个CPU优化窗口不作方法排名。R20/R21/R24/R26的GPU工程验收保持通过，不重跑。真实语料未准入；[统一接手入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)给出原Goal、资源、约束与下一动作。下方日期为当时历史状态。

## 2026-10-01T08:20:56.541520+00:00 当前接续

R20/R21/R24/R26人工Trainer工程验收已通过，新增三项[终态原始结果复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/candidate_terminal_review.json)；CV新8方案仍原型待实现。当前状态、下一步与阻断见[统一接手入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)。下方日期为历史记录。

## 2026-10-01T08:10:49.754641+00:00 CV联合能力探索记录点

CV联合理解/生成研究新增原文和作者代码核读；7份固定源码快照、8份实验YAML规划已写入。新方法实现尚未完成，CPU/GPU未运行，配置未启用；现有R21/R24/R26仍在原8卡合成验证。 [研究与证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CV_UNIFIED_RELATIONAL_EXPLORATION_20261001.md)；[8份配置](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_cv_joint_exploration_20261001/README.md)。

## 2026-10-01T07:48:49.897115+00:00 新增联合目标与关系建模探索

新增[CV多对象关系与理解/生成联合建模记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CV_UNIFIED_RELATIONAL_EXPLORATION_20261001.md)，当前为文献核查和去重；没有新增方法的GPU结果。现有三个候选工程验收仍运行。

更新：2026-10-01 05:52 UTC。v3 R20完整GPU工程验收及根端强复审通过，3/24/768；T20私有准备完成，数据仍未准入，正式训练尚未启动。

## 2026-10-01 05:52 UTC v3完整GPU工程验收通过与剩余数据准入

05:44:51 UTC，v3 D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959 在原GPU3终态succeeded/rc0。根端05:48以worker同一Python的全新-I进程执行原不可变verify_completion/verify_candidate，通过全部raw/summary/COMPLETES/installed source/独立CPU计数重算；禁止本地CUDA初始化，未重跑GPU。runtime manifest仍cc13036fc492109afbf4c9cf9fd11a4c87684fc9702b83ce86c961b2a08294fe，524父文件不变。

实际正式3窗/24 microsteps/768人工记录，formal1054.3747秒<1800，训练loss3.450910→3.448508→3.380331；首窗LR0符合固定warmup，后两窗LR1e-6/2e-6且参数确有更新。独立full oracle706.9962秒<1200、GA/CE/RNG比较通过；四mask评估5.1172秒<600、每mask24条/共96次，eval_batch16含实际尾批及完整2048输入，指标由raw packets独立重算。正式峰值allocated45,350,397,440字节（42.24 GiB），reserved56,555,995,136（52.67 GiB），原85%标准通过；tiny/full/formal清理弱引用释放、最终allocated0。parent1794.0503秒<3600，child rc0、正常退出观察/后代清理true。[强复审原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/terminal_review.json)、[GPU通过原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/GPU_RELEASE_COMPLETE.json)。

两个自有空hf目录已删除，checkpoint=0；raw20,049,785字节、summary/日志/完成哈希是后续strong verifier及训练入口依赖，按必要证据保留，v1/v2失败原件不覆盖。[清理清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/gpu_validation_cleanup.json)。首次root提取曾把字段读到不同COMPLETE schema，读取错误已保存在review中，修正后通过，生产代码/标准未改。原8卡任务Running，0/0/39=36历史成功+1新D01成功+2D01失败，8worker维护；本地监听PID2213164继续watching，无新平台申请。

T20 v3私有两环境各48项通过且根端AST复审，source-map私有各23项通过，未publish/实际stage/apply/pointer/入队。[T20私有检查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_v3_20261001/validation.json)、[桥接复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/t20_v3_root_review.json)、[源映射复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/source_map_root_review.json)。GPU验收仅覆盖R20工程入口，不认证R21/R24/R26同版本训练，不证明架构收益。2026-10-01 05:52 UTC数据metadata仍training_ready=false、双证书缺失；审核后manifest改变须另做明确版本绑定，不能仅补文件绕过False。真实四候选1000→累计5000与最终版本训练未启动，goal仍active；独立CUDA算子保持暂停。

## 2026-10-01 05:31 UTC full梯度与清理实测通过，正式窗口运行中

v3新D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959 在原GPU3已完成独立full gradient_oracle：706.9962秒<1200，1窗/8 microsteps/256人工记录，78参数比较通过，梯度倍率1.00000077、相对L2误差0.00015194，CE/RNG核对通过。11个对象弱引用全部失效；gc/empty-cache后allocated17,039,360字节，清cuBLAS workspace后allocated/reserved均0；新运行tiny与full阶段均实测支持工作区驻留候选，历史v2仍缺残留数值，不追认其根因。[完整oracle局部实测](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/full_oracle_observation.json)。

已进入train_optimizer_step/1800秒预算，正式三窗口和四mask评估仍待完成，整体gpu_accepted/raw_completed=false。524父字节、原架构、数值标准和工作量不变；独立CUDA算子暂停。监听PID2213164/原八卡任务保持；T20 v3私有rebase中、源映射未部署，数据双证书仍缺/候选training_ready=false，真实比较与最终训练未启动。验证checkpoint=0。

## 2026-10-01 05:20 UTC 当前实际v3状态

v3于05:04:13 UTC实际发布8个支持/producer文件，05:09:59构建526文件不可变包，manifest SHA cc13036fc492109afbf4c9cf9fd11a4c87684fc9702b83ce86c961b2a08294fe；524父文件保持。producer必要CPU两环境各30项，support各108项通过，实际隔离导入各8 profiles/54项目模块通过。05:12:16本地监听实际重载PID2213164并核验watching，GPU任务/worker未重启。05:14:40只在原八卡池入队D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959，GPU3实际领取。固定tiny preflight耗时1.8074秒、8 microsteps/8人工记录；11个对象弱引用均释放，gc/empty-cache后allocated17,039,360字节，清原生cuBLAS workspace后allocated/reserved均0。当前完整gradient_oracle进行中，formal三窗口及四mask评估尚未通过；仍gpu_accepted=false，不能把局部tiny通过记作完整验收。[发布原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/publication.json)、[固定运行包](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v3_20261001/pilot_runtime_manifest.json)、[实际启动记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/d01_v3_launch.json)、[tiny阶段实测](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/lifetime_preflight_observation.json)。

T20 source-map私有两环境各23项通过，原28项映射保持；[软件源绑定记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_source_map_cpu_20261001/validation.json)。T20新私有v3正在显式rebase固定root/manifest，未发布/apply/pointer/入队；原数据准入全部保留。候选training_ready仍false、双证书未齐；真实四候选1000→累计5000与最终训练未启动，goal active。独立CUDA算子仍暂停。自有CPU临时目录已清理，人工GPU关闭checkpoint保存，小型指标/失败原件/源码哈希保留。

以下带时间记录属于当时状态。



## 当前结论

基线保留原 LLaDA270 的 **d768、8层、12头、MLP3072** 主干，从头训练且不用 ESMC。原 checkpoint 的约270M参数主要来自大词表；当前50-token版本约75.60M，不能把它描述为270M有效主干。已有链内 RoPE、链角色/范围编码和同链/异链 bias；[参数及接口核查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_LLADA270_BASELINE_AUDIT.md)记录具体差异。

R20–R27 已全部完成 **500步**，八项终态、39份源码和profile核验一致。最后100步R21/R26/R24的mean loss均值为0.186520/0.186906/0.187263，条件full-AA均值98.53%/98.77%/99.00%。保留R20对照，优先比较低额外显存R21/R24，R26作为长输入显存增加28.51%的候选。五个检查点属于同一条曲线，不能当成独立重复或据此确定唯一赢家。partial-mask已接近饱和，无条件full-mask又有已知输入冲突，不为toy排名统一延长全部八项。[完整终态、曲线和成本](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_R20_R27_500STEP_RESULTS.md)。

## 方法覆盖与实际状态

| 研究轴 | 已落地的范围 | 当前证据 / 下一步 |
| --- | --- | --- |
| span / whole-chain mask、链与END权重 | R01–R03，原目标单独对照 | 首轮20步GPU工程诊断；尚无真实语料效果证据，目标不同不能直接按训练loss排名 |
| QK norm、register、局部卷积、噪声调制、有向角色bias | R04–R07、R10；R11为预先指定组合 | 已实现并完成首轮20步；现有CPU/GPU通过不表示各轴已有充分收敛验证 |
| head gate及尺度对照 | R21 vs R22，并含R20对照 | 500步人工诊断完成；区分query相关门控与单纯乘0.5 |
| LayerScale及尺度对照 | R23 vs R24，并含R20对照 | 500步人工诊断完成；学习缩放和固定0.1分别比较 |
| 链内/全局信息流 | R25前4层独立后4层全局；R26局部半径32与全局交替 | 500步人工诊断完成；dense mask改变信息流，不能宣称已实现稀疏加速 |
| 首层value残差 | R27 | 500步人工诊断完成；所有V投影保留，没有跨batch缓存 |
| 表征辅助目标 | R08 VICReg式；独立G06/G07自身EMA对齐与零权重对照；SIGReg/data2vec/iBOT/JEPA为设计 | R08仅短诊断；G06/G07通用整数CPU功能已通过、GPU曲线待完成；其他机制未实现，teacher成本和统计microbatch需控制 |
| 来源采样及loss计数 | R09；自然/等来源曝光审计；record-mean原型 | 整数索引曝光已复核；record-mean通过人工CPU真实HF/Accelerate窗口验证，未接入生产模型或GPU；温度采样只有配额方案 |
| 长条件编码/残差通路 | 独立G04/G05的4/16 register主干瓶颈；Perceiver、双向SSM、Gated DeltaNet、HC/mHC、差分attention | G04/G05通用整数CPU功能已通过、GPU曲线待完成，关系头仍拼接角色池化；其他机制仍设计阶段，保持双向去噪及逐位置输出合同 |
| 自条件、置信度、重新遮蔽 | Analog Bits/SED、RemeDi、ReMDM启发 | 原文与固定作者代码已阅读，未实现；训练辅助与推理采样收益分列 |

R00–R11的12项20步、R12–R19的8项100步、R20–R27的8项25步已完成，共28项。它们的fixture、源版本和预算不同，不能把28项拼成同一架构排行榜。额外八项500步也已完成，共36份终态记录；仍不能跨fixture、源码版本和预算拼成一张排名表。

文献和代码的具体阅读范围、固定提交、未读到的材料及迁移限制分别见：[首轮方法库](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_FOUNDATION_ARCHITECTURE_RESEARCH.md)、[视觉与多模态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_VISION_EXPANSION.md)、[序列与残差架构](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_SEQUENCE_EXPANSION.md)、[去噪方法补充](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_DENOISING_METHOD_EXPANSION.md)。这些文献清单有重复工作，不能把表行数直接相加称为独立论文数。

## 接下来按证据推进

2026-10-01用户确认当前不需要独立CUDA算子验证；它已移到规划的可选支线。主线顺序如下：

1. 数据审核收尾与修正后Trainer GPU验收并行。实际完整窗口、BF16/AdamW、四mask计数、首次END/长度、RNG及显存需要验证；三次更新只证明实现，不能评估学习收益。已有覆盖未决项和数据证书仍待完成。
2. 满足原数据准入和实际GPU验收后，R20对照、R21、R24、R26各1000步，固定训练内验证、每100步评估，报告分层曲线和显存/耗时。
3. 依据训练内验证和成本选候选，恢复完整optimizer/scheduler/RNG到累计5000步，再用第二seed确认；不凭人工成绩或单检查点敲定最终架构。
4. 新loss、来源采样和其它机制分别独立消融，保持主干和评估口径一致。[完整规划](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/protocol.yml)。

当前首轮架构比较保持同一训练目标、自然采样、global256和学习率日程。新目标、新采样、反馈输入或采样器各自独立消融。原目标对microbatch分组敏感：实测推荐档不同的架构，不能仅凭global256相同就视为等权比较；新record-mean原型也不能悄然替换旧目标。[归一化证据与限制](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RECORD_LOSS_FRAMEWORK.md)。

部署准备已补齐可审阅产物：[完整依赖归档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_REFINED_RELEASE_ARCHIVE.md)两环境各22项CPU通过，[固定路由补丁](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_DIAGNOSTIC_DISPATCH_STAGING.md)各38项通过。524文件R20父快照和8项release_map重绑定已完成；worker的.pth/同名namespace隔离缺口正在修正，实际新入口尚未部署。后续新pilot仍须解决live源码与旧profile的绑定差异，不能以此次路由验收替代新评估/训练的GPU证据。

02:30Z进展：固定新入口已实现并有两环境各29项CPU通过；D01运输/来源隔离各34项、typed展示各14项通过，实际28个历史路由均匹配自己的快照。以上新源码尚未应用到BASE3或运行中的daemon，也没有正式runtime/GPU任务。producer原址出现未归属内容变化后已保存字节，在专属私有副本继续完成；发布先复审和核对原址SHA，旧producer成绩不认证当前变化的版本。[入口验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TRAINER_RUNTIME_ENTRY_V1.md)；[运输与展示](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TRAINER_POOL_TRANSPORT_V1.md)。独立算子支线维持不执行。

实际HF窗口审计发现新Trainer因可变关键字签名跳过GA除法，已在未部署原型中修正，旧字节及前后对照保留；两环境各90项回归与5个完整窗口实验通过。修正不改变当前500步手动循环、不引入record-mean新目标；它仍需真实模型BF16/AdamW的GPU验收。[完整窗口修正](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_HF_MICRO_MEAN_FIX.md)。

## 数据、监督和记录入口

独立候选已导出4,781,843条，但仍 `training_ready=false`，缺少通过的双证书；1288条固定验证清单只是索引和字节绑定。native/TRAIT验证各4条，宏平均改善必须同时展示计数和不确定性。[候选与准入状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TRAINING_DATA_READINESS.md)。此前自动检查中断的序列注释分支保持停止，未解决覆盖项继续pending，软件验收不替代该审核。

[实时平台/GPU/任务记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_POOL_LIVE.md)与[证据监督](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_SUPERVISION.md)由后台daemon更新；监督subagent负责解释异常并报告。daemon持续运行不等于聊天窗口或模型推理无限在线。修正只进入独立版本，当前实验不热改；测试checkpoint及时清理，保留小型日志、指标和来源证据。
