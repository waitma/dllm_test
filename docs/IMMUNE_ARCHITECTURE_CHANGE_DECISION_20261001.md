# 架构是否改动：三方审查后的阶段决策

## 2026-10-02T02:02:27.596571+00:00 新seed实际失败及当前边界

最新实际状态为7运行、1失败、0/8完整验收通过。R21 seed43/GPU2在lifetime_preflight独立CE/GA梯度核对报wte.weight不一致；profile正常退出，release子进程rc1、runner rc2，父进程记录进程清理完成，GPU2恢复维护。错误时对象弱引用存活，不把它写作成功阶段清理，也不凭此认定持续泄漏。其余7项仍gradient_oracle，真实训练0。

根因尚未确认，原断言未保存逐参数误差统计；下一步是隔离通用数值/随机流诊断。/root/architecture_keep_or_change只读分析，/root/candidate_gpu_watch只读监督；全部v2活跃源码/私有来源/运行包保持冻结，不热改、重试、放宽阈值或重启。V7仍缺有效ready及双报告，原500学习、完整恢复、数据门禁与安全停线保持，独立CUDA算子暂停，停止注释分支不恢复。

[实际失败复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_launch/r21_seed43_failure_review.json)、[最新逐项状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_launch/observation_20261002T020227Z.json)。

## 2026-10-02T01:57:43.567408+00:00 当前完成边界

四R候选实现保持；新seed执行整合已完成，新增定向CPU两环境各42/42及实际stdlib启动各8/8通过。四运行包已构建、原容器已接入，八项d02人工Trainer验收01:54 UTC一次启动；01:55:50 UTC全部领取，0/8完整终态，真实训练0。真实P1的t20/t21/t24/t26接口、原500学习/完整恢复与当前V7有效数据审核仍是前置；不把软件和人工结果解释为真实效果，不前置扩大G02。

[当前任务原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_launch/launch.json)、[来源与改动](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_publication/publication.json)、[当前总状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)。下方日期章节保留为历史。

## 2026-10-02T01:30:12.523373+00:00 当前首轮完成边界

八身份软件请求接口完成，pllm与worker对应Python各13/13新增定向CPU检查首轮通过；实际核对四v4包的manifest/fullmap/helper、八个显式seed请求、软件1000→5000续接一致性及缺执行准入时无副作用拒绝。源码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/private_sources/scripts/architecture_prototypes/pilot_pool_seed_launcher_v1.py SHA867427c73dd7c2b16e8f419d5e22d1b1735aad54d832a13316e2ff2f0980ea24。独立静态复审1项收尾SHA位置问题已修正，冻结前writer变更使旧SHA记录拒绝，未将过期复审写成通过；原件保留。受保护18项源码/manifest SHA保持，旧core28/execution11/8次完整导入/旧Trainer验收未重跑。两pytest临时根未创建，临时文件删除0，checkpoint创建/删除0；必要日志/JUnit/冻结与复审记录保留。 [新增CPU原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_controller_cpu_results.json)。

本次只完成软件请求接口；完整八身份dispatcher/registry/entry接入、seed-aware runtime/producer与当前源/seed的GPU及规定profile/diagnostic证据仍待完成。未部署/注册/入队/运行新GPU/真实训练。原八卡01:29:00 UTC pending0/running0/results42、8worker维护无job；平台缓存01:28:52 Running、监听PID2213164存活。指定immune_v7_mhc_sequence_v1 manifest SHA未变，无明确training_ready字段及两份当前版本/范围报告。旧V7已有清理；当前缺绑定不能解释为已证实残留泄漏。独立CUDA暂停、停止注释分支与原数据门禁全部保持；原Goal actual active及身份/计量保留。 [资源及V7原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_controller_final_resource.json)。

下一必要工程：私有正式入口/registry/dispatcher和新seed-aware runtime verifier/人工Trainer producer；随后只获取受影响的当前source/seed证据。旧profile/500结果按原来源保留，不替换SHA、不删原检查。有效data/selection和完整checkpoint恢复是实际训练前置；团队数据审核未闭环时不抽取真实小batch、不启动正式训练。首轮仍R20/R21/R24/R26×seed42/43共8独立单卡比较，G02为条件P2。

下方按时间保留历史，本节覆盖同项旧状态。

## 2026-10-02T01:21:06.578730+00:00 当前首轮范围及完成状态

首轮固定R20/R21/R24/R26×seed42/43，八项独立单卡比较；四候选实现和旧版本synthetic Trainer验收已完成。多seed核心、执行来源绑定CPU以及四v4包×两环境的软件导入已完成；八身份软件请求接口正在实现与定向验收，正式容器接入/new-source GPU及有效数据报告仍未完成，真实训练未启动。G02为后续条件研究，不前置扩大本轮。原Goal工具active，旧blocked条目仅描述当时状态。

指定数据仍是immune_v7_mhc_sequence_v1；旧V7已有清理，不重新认定全部未做。当前MHC manifest没有明确ready、两份当前版本/范围报告缺失，不等于已证实残留泄漏；不得补写passing或恢复停止的注释。原八卡容器维持在线但无实际训练job。本轮没有新结构决策、部署、入队或GPU验收。 [当前状态原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_controller_readiness_observation.json)、[统一接手页](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)。

以下按时间保留阶段历史，当前状态以本节和原始JSON为准。

## 2026-10-01T23:41:25.476114+00:00 V7与八卡启动条件核实

用户倾向使用指定V7并已条件授权架构完成后启动原八卡。当前核实：目标为immune_v7_mhc_sequence_v1（train 6,849,243 / valid 52,572），缺明确ready和两份检查报告；四个R候选实现及synthetic Trainer GPU验收完成，私有训练接入未部署，G02正式联合模型未适配。原池0/0/42、八worker维护、监听存活；真实训练未启动。仅metadata/原件/源码核对与MD记录，无重复验收、注释恢复、门禁变更、任务操作或新checkpoint；原Goal blocked保持。 [当前版本、原始验收与资源核实](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/v7_eight_gpu_readiness_20261001T233803Z.json)。用户启动授权已记录，无需重复申请权限；本条未将数据可用意见写成审核通过，也未启动真实训练。下方带日期内容保留为历史。

本回合是当前实现状态核实，未新增架构决策。四个R候选编码完成；G02等联合方案仍为待正式适配的条件研究项，不能把所有探索方案统称为已完成。已有synthetic验收不重复；当前主要剩余条件是有效数据绑定、正式接入发布与实际小batch验收。

更新：2026-10-01T11:50:40.065141+00:00。维护者 `/root`；原Goal `01a0f245-4559-7b62-9dc2-0917a8ed2267`继续active。用户明确要求多个subagent帮助决定“是否改、怎么改、怎样测试”。本页是决策与测试设计，尚未实施这些架构改动或启动真实实验。

**当前决定：保留R20/A4为正式尺寸基线；R21、R24先分别比较，R26作为成本对照；联合理解/生成优先考虑末端小adapter，任务专用norm降为读出/优化对照。硬瓶颈和EMA暂不进入主线。** 没有真实数据效果证据时不选赢家、不组合多个机制。

## 三个独立审查与合并判断

| Agent | 负责问题 | 主要结论 |
|---|---|---|
| `/root/architecture_keep_or_change` | 现有基线是否有充分理由被替换 | 没有；先提高正式评价判别力，优先单独比较R21/R24 |
| `/root/cv_method_decision` | CV方法是否增加有效任务分工、关系能力 | G02实际改变末端路径；G01当前主要为可吸收的仿射重参数化；硬瓶颈及EMA先缓 |
| `/root/architecture_validation_strategy` | 多GPU试验如何区分工程正常和模型提升 | 正式尺寸4方法×2seed先行；联合任务组另设共同对照；按tokens、GPU时间和完整曲线比较 |

三者已返回，仅只读文档、源码及metadata，没有改源码、跑模型/测试、读取shard/序列或操作GPU任务。[审查摘要及证据索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_architecture_decision_20261001/agent_review_summaries.json)。该索引是根端合并摘要，不是数据审核证书或训练成绩。

分歧处理：基线审查最初倾向先试低成本norm；源码复核发现G01没有改变主干任务路径，因此把G01保留为便宜控制，把G02作为有条件的结构候选。没有照多数投票直接升级模块。

## 是否改、怎么改

| 方法/问题 | 现在的决定 | 后续具体动作与理由 |
|---|---|---|
| R20/A4共享主干 | 保持 | 保留d768/L8/H12/MLP3072、小词表、链内RoPE、role/scope编码、同链/异链bias及当前终止协议，作为所有正式尺寸比较的共同对照 |
| 评价合同 | 优先补齐，不改主干 | 分开报告理解、重建、实际多步生成和停止；同时保留来源宏平均、token微平均及来源/长度/链组合分层。END召回100%不能代替正确首次停止 |
| R21注意力头门控 | 单项候选 | 比较动态gate；保持原目标、容量、数据与日程。门控初值涉及尺度，晋级后再单独检查LR敏感性 |
| R24固定残差缩放 | 单项候选 | 与R20单独比较，不先和R21叠加；区分优化尺度适配与稳定泛化收益 |
| R26局部/全局交互 | 成本对照，暂不默认采用 | 观察关系信息收益能否覆盖长输入显存和耗时增加；不能只看训练loss |
| 正式联合理解/生成基线 | 先明确目标与接口 | R20当前主训练目标是去噪，能导出hidden features，不等于已存在G00式联合理解分类头。先确认理解端可信标签/固定探针协议，缺标签就标未就绪 |
| G01任务末端norm | 读出/优化控制 | 不预先声称增加主干任务特化；检查表征评价是否改善以及是否只是头部校准 |
| G02末端小adapter | 有条件的优先结构候选 | 在最后两层分别加入按任务选择的小残差adapter，上投影零初始化；保留共享浅层、位置协议和共同输出范围。先单独比较，不同时加入norm/register/EMA |
| G03 register | 保留研究配置，后移 | 它增加全局工作空间，不限制跨段直接通信；只有明确聚合失败时再试 |
| G04/G05硬瓶颈 | 暂缓 | 当前仅约束主干跨role attention，关系头仍可直接拼接两role池化；不能证明所有关系信息经过瓶颈。实现为dense mask，没有稀疏加速证据 |
| G06/G07自身EMA及计算对照 | 暂缓 | teacher增加参数与前向；只有表示问题成立且能超过同teacher计算零权重对照，才进入确认组 |
| 改loss、来源重采样、扩容 | 各自独立研究，当前不组合 | 改归一化或采样会改变优化问题，不与结构候选一起改后声称是架构收益 |

R20/R21/R24/R26已通过的完整synthetic Trainer验收不重跑。历史人工500步partial-mask已接近饱和，几项尾loss很接近；这些证据不能替代真实任务选择。[历史结果与限制](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_R20_R27_500STEP_RESULTS.md)、[当前工程验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/candidate_terminal_review.json)。

## G01和瓶颈的具体边界

当前[通用原型](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/generic_joint_model.py:81)的两种任务norm只分别学习仿射参数，标准化统计相同；生成头有独立Linear，关系头是两个非空角色的线性均值池化后接独立Linear。**在当前目标实际使用的生成输出与理解输出上**，`W_t(γ_t⊙N(h)+β_t)+b_t`中的任务仿射参数可吸收到各自头部权重/偏置；该推论不覆盖forward返回字典中未被对应任务监督的交叉输出。这是函数结构的推论，**不表示优化轨迹相同，也不表示导出特征的几何不变**。tied/frozen head、非线性池化、改变norm统计或缺失角色时须另核实。

正式R20[冻结构造](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r20_r27_20261001/dllm/pipelines/immune_llada/v3/model.py:53)显式weight_tying=false，但尚无上述联合分类头，不能直接外推整个双头论证。[冻结输出路径](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r20_r27_20261001/dllm/pipelines/immune_llada/research_refined/model.py:74)。

G04/G05的[attention_layout](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/generic_joint_model.py:105)按role相等开放边，原型每role至多一段；多个同role对象会直接通信。其[关系头](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/generic_joint_model.py:139)还有池化旁路。以后若测试严格段间瓶颈，必须明确独立段身份和关系头读出范围。当前不修改该实现、不用其分数宣称完整瓶颈成立。

CV依据只支持设计假设：[Show-o](https://arxiv.org/abs/2408.12528)和[Transfusion](https://arxiv.org/abs/2408.11039)提供共享主干处理不同目标的范例；[UniFork](https://arxiv.org/html/2506.17202v1)支持浅层共享、末端分工的研究方向；[MBT](https://arxiv.org/abs/2107.00135)支持受限融合机制；[SRA](https://arxiv.org/html/2505.02831v1)支持自身低噪声EMA对齐机制。本文的迁移判断不是这些论文对本项目性能的保证，原版依赖和阅读范围见[已有研究](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CV_UNIFIED_RELATIONAL_EXPLORATION_20261001.md)。

## 原八卡容器上的测试顺序

下面是审核与接口完成后的设计，**本次没有入队**。原资源`t-20261001033327-fmnql`、worker和监听PID2213164保持，不重复申请。

| 阶段 | 八卡安排 | 进入条件与目的 |
|---|---|---|
| P1：现有正式尺寸候选 | R20/R21/R24/R26 × seed42/43，一卡一项，共8项 | 数据有效准入、各候选实际训练接入及目标合同齐备；先比较已有低成本机制 |
| P2：联合任务分工 | 正式适配后的共享对照/G01/G02 × 同样两个seed，共6项 | 可信理解目标和共享主干联合基线已建立；其余两卡只安排确有需要的评价/成本控制，不为占满卡重复实验 |
| P3：问题驱动扩展 | register/两档瓶颈成组，或共享对照/EMA/同teacher计算零权重成组 | P2或正式表征评价确实暴露对应问题；不扩成全部G×R组合 |
| 确认 | 对照和入围候选追加第三个匹配seed | 排序、关键分层及成本结果足够支持继续；确认后才敲定阶段架构 |

P1与P2保持各自共同对照和目标。现有d192/L4/H6、16类整数G原型不能直接改data_dir当正式模型使用，不能和R组总loss混排行。

先用准入后的代表性小batch校准真实tokens吞吐、完整Adam状态显存、评价/保存耗时及8项并行存储影响，再冻结预算。profile若单独初始化，其更新次数另计、权重丢弃；若属于正式轨迹，则连续计数和完整续训。工程验收中的oracle和核验耗时不当作普通训练速度。

R组沿用既定global256与原LR/warmup/10000步scheduler horizon；1000步初筛、选定后**累计**5000步是检查节点，不是充分性保证，也不把scheduler horizon改成1000/5000。小batch先体现为所有候选共同可用的microbatch，再用GA保持global。现有micro-mean目标对分组敏感，结构排行需相同micro分组和尾窗规则；不同profile档位不得静默改变目标权重。[既有控制合同](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/protocol.yml:292)。记录归一化若采用，作为另一项明确消融。

每个seed共有参数初始化逐字节一致；新增模块用独立RNG且零初始化adapter不扰动共同模型。各候选同数据版本、记录顺序、mask、合法输入、内部valid、优化器和精度；评价保护训练RNG。主图按相同累计训练tokens，成本图按GPU秒；register、teacher、额外前向与推理成本全部计入。

## 何时停止、延长或晋级

- 功能验收只补改动相关项：零初始化输出一致、分支选择、padding、梯度、完整恢复；已通过35/44项和四候选工程验收不无依据重跑。
- 正式理解只用已有可信标签或锁定的冻结特征协议；不搬整数关系标签，不将未观察配对自动造为负例。缺标签记未就绪。partial-mask重建不等于理解能力。
- 生成同时看四档mask、固定解码预算的实际迭代结果、END precision/误报/首次停止/missing/长度误差，以及来源宏平均和token微平均。理解与生成单列，关键长度/链组合/来源退化单列。
- 数值异常、准入/版本失配、标签泄漏、重复样本消费、恢复不完整等工程错误立即停止并保留失败；新版本只补受影响检查。
- 曲线仍改善、排序不稳或seed波动大于候选差异时，只有在冻结预算允许的条件下，才对对照与候选按相同token节点共同延长。预算到期且证据不足写“未能区分”。
- 晋级必须在预先声明的理解/生成权衡及容忍退化范围内，有匹配seed和多个检查节点的一致证据，成本可接受，第三seed确认。检查点不是独立重复，统计单位为seed和独立验证家族。若仅总loss更低、仅头部更好、等GPU时间收益消失或关键分层持续退化，不进入当前主线。
- 架构及checkpoint选择只用固定内部valid；test/query不反馈结构、步数或loss选择。support暴露按有效审核范围如实登记，不扩大unseen声明。

门槛数值和预算秒数须在准入后按基线噪声、实际吞吐及有效验证家族数预先冻结，本次不编造这些尚未测得的数字。历史expanded候选某些来源只有4个验证家族，不能拆同族记录凑独立样本或据此做稳定来源排名。

## 当前门禁、产物与下一步

用户指定`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1`为正式小batch首选；仅metadata核实：manifest无training_ready字段，两份审核证书缺失。[门禁原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/requested_training_data_gate.json)。原expanded候选training_ready=false也保持。任何真实小batch和正式训练目前均未启动。

此外，最新[去重范围记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/DATA_BENCHMARK_DEDUP_SCOPE_20261001.md)与[tcr_scope_gate.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/tcr_scope_gate.py:9)旧SCOPE/input_specs尚未形成一致的新审核绑定。后续必须将有效范围、输入SHA、gate版本、双证书和最终manifest绑定一致。本轮不改变审核范围、降低门禁、补造证书或恢复已被安全检查停止的序列注释分支。

四候选外部接入已完成私有源码及两环境各24项新增身份/拒绝检查，仍未部署、未准入执行，MHC需有效审核及新的共同数据/selection绑定。[本次工程结果](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CANDIDATE_TRAINING_BRIDGES_20261001.md)。当前直接相关私有整合已收尾；剩余真实训练推进依赖外部审核，不增加无关synthetic曲线填充Goal。新G固定包已构建并完成CPU来源检查，但原池执行接入、独立GPU复核及实际模型适配仍未完成。[GPU工程记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_JOINT_GPU_SEGMENT_PROTOCOL_20261001.md)。独立CUDA算子验证仍暂停。

任务去重绑定源码/配置/数据与证书/valid选择/共有初始化/目标/seed/累计区间/明确前段SHA；成功项不重复提交。续训不找latest、不重置optimizer/scheduler/RNG。只删除明确自有且无复核/续训依赖的产物，保留实际恢复状态、tracked best、阶段锚点和pending/running引用；save_total_limit=3不代替依赖保护。

每个节点写UTC、维护agent、真实阶段/阻断、实际job/GPU/attempt、全部来源与版本SHA、预算和实际更新/records/tokens/GPU秒/显存、两端分层指标和分母、决策理由、失败处理与清理。状态同步[PROJECT_PROCESS.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)，方法同步[BIOSEQ_MODEL_PLAN.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md)，长期约定同步[PROJ_GUIDE.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJ_GUIDE.md)。实际数据格式适配发生后才更新DATA_FORMAT_AUDIT；本轮格式未改。
