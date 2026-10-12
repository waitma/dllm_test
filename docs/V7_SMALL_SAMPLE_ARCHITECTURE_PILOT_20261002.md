# V7 小样本架构比较

2026-10-03T08:52:55Z checkpoint 清理更新：按用户要求，p01 已完成 8 项的 latest.pt 已删除；pool 成功结果、COMPLETE、summary、provenance、curve、日志与最终图保留。已实际核对终态及结果 SHA，绘图不依赖二进制。旧段落中“latest.pt 保留”为当时历史状态；当前不再支持加载或恢复这些旧模型。逐文件证据：[清理记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_checkpoint_cleanup_20261003/cleanup_receipt.json)。本次未改变架构、数据或实验结果。

更新时间：2026-10-02T21:53:57.288545+00:00。当前状态：原p01八项100000条完整epoch终态核验通过、最终图完成；p02另册继续，正式训练暂缓。

用户明确要求暂缓正式训练，从当前已处理的V7真实训练数据抽少量样本，在现有八卡各做独立实验，不等待全量处理完成。本轮没有恢复停止的序列注释分支。新核实的最终V7目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002 已由其他工作流程完成物化，其清单与两份限定范围审核均passed，parent ready=True；该变化不是本研究手改ready，旧16:28缺审核结论不能代表此新版本。parent manifest SHA 28e3cbf8fb860c13b514a11d600579257a233fefe9f70ea070b042d04f8a424c。本轮仍只做用户要求的小样本研究，不据此启动正式训练。

两位子agent只读建议与主agent取舍：R20作共同对照；R24固定残差缩放和R21动态头门控为优先候选；R26局部/全局注意力作为第三项及成本对照。四项×两个真实训练seed（42/43）共八个独立单卡任务；无需共同八卡DDP作为前置。暂不叠加G系列多目标、EMA等新机制。R24可能受有效学习率影响，R21需区分动态门控和常数缩放，R26当前dense mask不保证更省显存；本轮用于筛选，不能宣称最终生物学收益。

采样上限train100000、valid1024，来源分层、确定性随机抽样，保留原split和原行字节及原shard/行号；不构建新注释，不使用test进行选型。子集记录parent清单/审核和自身哈希，标探索用途，不伪造新的正式审核。模型保持d768/L8/H12/MLP3072/max2048，micro4×GA8/global32；每项3125优化步，LR1e-4、warmup100、共同3125步调度。共同参数初始化、样本顺序与遮蔽流匹配，seed42/43是训练种子，不能用profile种子冒充。

报告初始及每250步四mask验证曲线、逐token与来源宏平均重建指标、训练/验证差距、token曝光、GPU秒及峰值显存；不将masked reconstruction当成生成质量或理解下游成绩。不输出新生成的生物序列。若差异小于seed波动或排序反复，不宣布获胜。只保留一个latest恢复checkpoint，运行期间保留恢复状态，完成结果核查后清理自有多余checkpoint。

使用原容器/worker，八个独立单卡实验均已运行；未重启容器或监听、未重复平台提交。独立小样本入口与最小pool分派整合已完成；所有八项运行同一冻结源码快照，公共正式训练审核规则保持。

机器方案：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/study_plan.json。抽样源码：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/sample_v7_architecture_pilot.py（已实现并用于当前实验）；实验源码：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_architecture_pilot.py（已实现并用于当前实验）。此前已通过人工500步不重复。实际启动/结果将在本文与PROJECT_PROCESS同步，不以计划代替执行证据。

用户追加要求（2026-10-02）：抽样扩大到100000 train；已确定实验可先启动。global32下1000步只覆盖32000次记录曝光，因此完整初筛统一3125步/100000次曝光（实际unique数量以记录为准），前1000步仅中间趋势。最终消融图另外纳入单因素、组合及去除组件对照计划。抽样脚本、fixture和实际100000/1024抽样均已完成；下面带日期条目为阶段历史。

## 2026-10-02T19:08:18.446121+00:00 十万条V7抽样完成，完整消融图计划登记

实际抽样train100000/valid1024成功，原行/split保持；source分层、SHA与count已核实，样本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/sample/sample_manifest.json。尚未启动GPU训练。完成五份原始论文/官方指南查阅，消融设计为四项初筛、三因素八条件组合与逐项去除，另列固定门控/同等调参预算对照；最终六面板图及日志字段见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/ARCHITECTURE_ABLATION_FIGURE_PLAN_20261002.md。已确定首批实验不等待组合实现。

## 2026-10-02T19:11:34.656162+00:00 V7入口检查与模型源码冻结

新增runner的CPU功能检查首轮失败于torch动态模块伪文件_classes.py，不涉及模型数值；修复source_hashes仅纳入实际存在文件后，第二轮2步人工入口检查通过，初末验证loss 4.234493→4.141472。只验证新入口，不当作真实效果。已清理该次自有latest.pt并保留日志、完成与失败证据。冻结563个模型/入口Python文件于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/source_snapshot_v1；新增既有pool分派器尚未接入/入队。十万条实际输入布局全扫描进行中，不截断或替换失败行。

## 2026-10-02T19:14:30.135931+00:00 首批V7实际架构实验已进入原八卡池

独立review与8别名参数检查通过；实际101024行布局全通过。已最小更新公共runner入口路径和8个launch-only alias，原70项注册不变。首批R20 seed42、R21 seed42各3125步已入现有池（不是新平台提交），job IDs：1790968470149719891_p01_v7_r20_seed42_838aa052, 1790968470161653699_p01_v7_r21_seed42_ccb7e981。此时只有入队证据，实际优化步待观察；其余6项在相同入口开始初始化后接续入队。没有重启容器/worker/监听。新MD方案及冻结源码参见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_SMALL_SAMPLE_ARCHITECTURE_PILOT_20261002.md。此前未发布的V4 world8 stage属于历史方案，其公共源码前置已改变，不可再直接发布旧stage。

## 2026-10-02T19:17:16.170057+00:00 八项V7探索已全部入现有池

R20 seed42/GPU0、R21 seed42/GPU4已实际初始化并完成step0四mask验证，共同参数初值SHA均44ff26ac8144ece78c84d9660e2d40d8f0746fa4e555a9ae2afeaadbd7d4b6a2，参数量75,601,176/75,674,904，torch2.8.0。其余R24/R26 seed42和四项seed43已依次入队，总8项。每项100000训练样本/1024验证样本，3125优化步，实际进度由progress/curve证明。完整job账本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/jobs.json。real_data_pilot_started=true，formal_training_started=false。未新建平台任务或重启。

## 2026-10-02T19:20:29.257576+00:00 8张GPU均已实际运行V7探索

所有8任务running，均完成step0验证；数据SHA/加载源码映射完全相同，同seed共同参数初值一致。两位独立复核与root状态相符。无失败/OOM证据。下面是此时刻快照，后续实时进度看 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/jobs.json 指向的progress.json/curve.jsonl。

| GPU | 方法与seed | 已记录优化步 |
|---|---|---|
| 0 | p01_v7_r20_seed42 | 100 |
| 1 | p01_v7_r24_seed42 | 50 |
| 2 | p01_v7_r24_seed43 | 50 |
| 3 | p01_v7_r21_seed43 | 50 |
| 4 | p01_v7_r21_seed42 | 100 |
| 5 | p01_v7_r20_seed43 | 50 |
| 6 | p01_v7_r26_seed43 | 50 |
| 7 | p01_v7_r26_seed42 | 50 |

当前没有任何非零步验证终点，因此不能评价架构排序；每250步的验证才能进入公平比较图。重建loss与训练loss分开，完整三因素组合消融仍未运行。

## 2026-10-02T19:24:28.477070+00:00 实测作图读取已通过，等待共同250步验证

作图入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_architecture_pilot.py 已实现并用当前八项真实日志执行。数据/配置/源码/同seed初始化一致性检查通过；返回awaiting_shared_evaluation、common_evaluation_step=null、missing_curves=[]、figures=[]。当前只有step0曲线，正确输出CSV/JSON证据而不生成假效果图。作图证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/figures/v7_pilot.json。

当前优化步：p01_v7_r20_seed42=200, p01_v7_r21_seed42=175, p01_v7_r24_seed42=125, p01_v7_r26_seed42=125, p01_v7_r20_seed43=125, p01_v7_r21_seed43=125, p01_v7_r24_seed43=125, p01_v7_r26_seed43=125。首轮完整3125步仍运行中，未确定胜出架构。根任务在本轮完成方案、抽样、入口整合与实际八卡启动；后续监督负责共同250/1000/2000/3125里程碑及错误告警，不提交新任务或改动运行中源码。


## 2026-10-02T19:30:43+00:00 八项共同250步验证已落盘并生成首张对比图

原8项V7小样本实验均完成250步验证；19:30:30 UTC进度R20/R21 seed42=300，其余六项=250，全部running、心跳新鲜、无失败。每项共同250步对应8000次记录曝光，尚未完成100000条一轮。独立监督原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/watch_receipts/shared250_20261002T193030Z.json。

既有作图脚本实际返回recorded_evidence_available、common_evaluation_step=250、missing_curves=[]；首张SVG/PDF/PNG及CSV/JSON已输出至 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/figures/v7_pilot（对应扩展名）。当前图只比较共同250步，展示四独立机制候选×两seed，不能称三因素完整组合消融、最终架构选定或理解/生成下游效果。源代码与运行任务未变，不提交新任务、不重启、不清理恢复latest.pt；正式训练仍暂缓。

## 2026-10-02T19:52:22.715860+00:00 grammar/context loss核验及context数据增强计划

冻结A4的101024行loss资格核验通过：train/valid context位置697220/5852，eligible与有效label均0；关系监督资格4648/37，实际标签全部binding正例。用户明确要求保留关系监督并增加context，计划原100000追加20000既有context监督记录、原valid不变；尚在实现/未抽样/未启动新组，不修改当前八项。完整epoch索引与第1000步断点接续控制通过，3125步恰好覆盖100000索引。详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_GRAMMAR_CONTEXT_LOSS_20261002.md。

多图/多帧新增阅读与三份固定作者源码已归档；优先研究分段全长/global交替及既有延后融合，其次target/context cross-attention，保持context无序列CE与明确关系监督。新方法未实现/执行；研究取舍见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/CV_MULTIVIEW_VIDEO_ARCHITECTURE_20261002.md。

## 2026-10-02T19:58:16.131168+00:00 用户确认本轮只学习binding正例

异步目标澄清已明确选择“本轮只学习已有binding正例，暂不做正负判别”。据此追加候选只取原V7明确binding=True监督的context记录，保留关系token监督，不引入negative、不生成负例、不请求新负例来源。缺nonbinding不再作为本轮阻断，也不把正例学习描述为正负区分能力。当前八卡样本/源码/3125步不变；独立追加20000正例context仍在实现中。


## 2026-10-02T20:06:40+00:00 八项共同1000步验证完成，仍继续完整100000条epoch

19:25起有限监督至20:06:21 UTC，原8任务全部running，实际优化步1000–1050/3125，心跳新鲜、无失败；所有progress的records=step×32。共同1000步每项只曝光32000条，未将此中间点称作完整epoch。已按用户明确选择维持已有binding正例监督，不做正负判别，context无序列CE协议不变。

| 方法 | seed | token汇总四mask平均AA CE | source宏平均四mask平均AA CE | 记录曝光 |
|---|---|---|---|---|
| R20 | 42 | 1.013248 | 1.616497 | 32000 |
| R21 | 42 | 1.030937 | 1.569563 | 32000 |
| R24 | 42 | 0.951986 | 1.484525 | 32000 |
| R26 | 42 | 0.952572 | 1.418192 | 32000 |
| R20 | 43 | 0.996182 | 1.621853 | 32000 |
| R21 | 43 | 1.018624 | 1.747689 | 32000 |
| R24 | 43 | 0.952117 | 1.595562 | 32000 |
| R26 | 43 | 0.950303 | 1.398512 | 32000 |

上表是固定验证子集的中间重建结果，不等于关系二分类、理解或迭代生成效果；仅两seed、来源规模不同，不按单一指标或单点宣布最终赢家。既有plot已实际更新SVG/PDF/PNG和CSV/JSON，common_evaluation_step=1000、missing_curves=[]。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/watch_receipts/shared1000_20261002T200621Z.json；图及原始汇总 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/figures/v7_pilot.json。原8项继续至3125步，无任务/源码变更，latest.pt保留；正式训练与三因素组合消融未启动。

## 2026-10-02T20:26:02.091441+00:00 context正例新增20000条完成，加载与loss布局检查通过

用户选择本轮只学习已有binding正例、暂不做正负判别。独立新子集 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/sample_context_plus20k 已生成：保留原100000训练行作为逐字节前缀，追加20000现有V7 parent train记录，总120000；valid1024逐字节不变。context从4648/100000（4.648%）增至24648/120000（20.54%）。新增均labels.relation=binding且metadata.relation_supervised=True，未生成标签或负例。原74个train分片的SHA/count全部核对，候选context正例297751条，排除base中4648行后可选293103条，分source比例抽样。

| 新增来源 | 行数 |
|---|---:|
| asd_antibody | 11335 |
| tcr_native | 825 |
| tcr_papers | 7685 |
| trait | 155 |

原冻结loader实际加载120000/1024并通过parent ready/两证书当前SHA绑定检查；新增20000行逐条A4布局通过，最长1299 tokens，无丢行/截断。新增context位置2972878个，label/eligible均0；binding关系监督资格20000个，target监督资格2292707个。独立复核新增原行SHA唯一、与base train/valid原行SHA无交集。这里排重指原行字节与来源身份，不声称做了新的序列相似度清理。eligible仅指可被遮蔽监督，不表示每步都纳入loss。

数据manifest SHA 1efa27ea19d10bc94143c3977c80dba8c3a29bad921c093ca7f24447635b9478；完整计数/来源证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/sample_context_plus20k/sample_manifest.json；加载与mask检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/context_augmentation_preflight.json。增补入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/augment_v7_context_subset.py；独立只读源码review无必须修正项，记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/context_augmentation_code_review.json。没有产生checkpoint或修改正式ready/审核文件、恢复注释分支或独立CUDA算子工作。

当前原8项记录步数1400–1475/3125，继续原100000条完整epoch，原样本/冻结源码/作业不变。新120000条尚未用于训练；global32下下一独立组应完整3750步。旧p01 runner/dispatcher/plot固定3125，不能直接拿它训练120000并称完整epoch；后续必须独立配置3750的训练调度、终态与作图并分组报告。没有平台submit/cancel、容器/worker/监听重启或重复入队。


## 2026-10-02T20:56:35.885636+00:00 原p01八项共同2000步验证完成

20:55:27 UTC原8项全部running，优化步2000–2075/3125；心跳正常、没有失败结果，records=step×32持续一致。共同2000步每项64000次记录曝光，仍未完成原100000条epoch。既有plot实际更新SVG/PDF/PNG与CSV/JSON，common_evaluation_step=2000、missing_curves=[]；数据/来源/配置/同seed初始化与同step token-mask曝光检查通过。

| 方法 | seed | token汇总四mask平均AA CE | source宏平均四mask平均AA CE | 记录曝光 |
|---|---|---|---|---|
| R20 | 42 | 0.895875 | 1.462718 | 64000 |
| R21 | 42 | 0.901744 | 1.403328 | 64000 |
| R24 | 42 | 0.884383 | 1.312384 | 64000 |
| R26 | 42 | 0.876142 | 1.320238 | 64000 |
| R20 | 43 | 0.878229 | 1.397849 | 64000 |
| R21 | 43 | 0.892175 | 1.468787 | 64000 |
| R24 | 43 | 0.856003 | 1.384450 | 64000 |
| R26 | 43 | 0.859800 | 1.274758 | 64000 |

本表仅原p01四候选×两seed在固定1024验证记录上的中间重建结果，维持用户选定的已有binding正例学习，不评价正负判别或生成质量。后续sample_context_plus20k及3750步方案不混入本图。监督证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/watch_receipts/shared2000_20261002T205527Z.json；作图汇总 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/figures/v7_pilot.json。继续原3125步终态，无重试/新提交/运行源码变更，唯一latest.pt继续保留；三因素组合消融及正式训练未启动。


## 2026-10-02T21:53:57.288545+00:00 原p01八项完整epoch终态核验通过

21:49:58 UTC观察8/8 pool succeeded与COMPLETE；21:51:08 UTC独立核验全部rc0、3125优化步、records=unique_train_records=100000、epoch1/position0、14个实际评估点（0、250..3000、3125）、summary/provenance SHA及原sample/source绑定通过。同seed共同初始化一致。每run唯一latest.pt仍完整保留（约907–908MB），没有多余checkpoint目录或latest临时文件；未加载或删除checkpoint。

| 方法与seed | 四mask token AA CE | 四mask source宏平均AA CE | full-mask AA准确率 | 峰allocated GiB |
|---|---|---|---|---|
| r20_seed42 | 0.840692 | 1.328485 | 43.68% | 3.236 |
| r21_seed42 | 0.850393 | 1.328313 | 43.82% | 3.252 |
| r24_seed42 | 0.816158 | 1.254604 | 43.79% | 3.236 |
| r26_seed42 | 0.823796 | 1.256284 | 43.79% | 3.815 |
| r20_seed43 | 0.835106 | 1.352879 | 43.73% | 3.227 |
| r21_seed43 | 0.845567 | 1.400086 | 43.54% | 3.243 |
| r24_seed43 | 0.818527 | 1.325737 | 43.72% | 3.227 |
| r26_seed43 | 0.818080 | 1.235990 | 43.67% | 3.803 |

终态R24与R26两seed的token平均AA CE均低于各自R20；R21两seed均高于R20。R24在token汇总均值略低，R26在source宏平均更低，且full-mask AA准确率各项仅约43.5–43.8%，不能把partial-mask改善当完整生成质量。source宏平均中部分来源验证仅1或2条，分层不确定性很大。两seed、固定子集与固定LR的筛选结果不证明全局最优或机制因果；门控常数缩放、残差优化尺度和三因素组合需独立p02证据。

累计worker wall折算20.1754 GPU小时，训练循环19.6586 GPU小时；当前真实长度峰显存3.227–3.815GiB，与旧2048 profile口径不同。最终SVG/PDF/PNG真实生成且PNG已目视检查：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/figures/v7_pilot.png（同前缀有SVG/PDF/CSV/JSON）。完整哈希/终态/成本原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/watch_receipts/final_p01_20261002.json。

用户确认的binding正例目标保持，未声称正负判别、理解下游或迭代生成效果；正式训练未启动。p01终态图只包含原100k/3125八项，不能拼入120k/3750的p02。原worker已自然接续既有p02队列；21:52:31 UTC为8running/14pending、首批25–100步，未新增提交或重启，后续另册监督。
