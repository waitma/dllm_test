# 四候选正式训练接入：私有工程整合

更新：2026-10-01T16:34:42.531238+00:00。原Goal `01a0f245-4559-7b62-9dc2-0917a8ed2267`恢复后新周期连续3次核实同一数据审核阻断，现已实际设为**blocked**；原目标、身份和累计用量保留，旧周期计数未复用。**四候选私有源码和两环境各24项CPU检查已完成；未部署、未准入、未启动真实小batch或最终训练。** 私有工程检查不证明架构收益或数据审核通过，全部停线和门禁保持。

## 现有能力与实际缺口

三候选固定包已有`execute_admitted`正式训练API，调用原preflight、各自包/模型/完整GPU验收，然后调用原训练实现；无需另建Trainer。原R20外部接入仅适用于控制模型；此次已在私有源码中补齐三候选独立执行身份、各自来源/GPU证明及续训绑定，线上尚未部署。

| 模型 | 已有正式API | 固定manifest SHA |
|---|---|---|
| R20 | 原完整验收包的admitted-pilot API | cc13036fc492109afbf4c9cf9fd11a4c87684fc9702b83ce86c961b2a08294fe |
| R21 | [execute_admitted](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r21_pilot_v1_20261001/scripts/architecture_prototypes/pilot_runtime_entry_r21_v1.py:271) | eaefeda0391d1e93cf58f0f0677905cd8323c190c8b769ff6ff579e044efe5b6 |
| R24 | [execute_admitted](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r24_pilot_v1_20261001/scripts/architecture_prototypes/pilot_runtime_entry_r24_v1.py:271) | ca0831164397882ab71217336f4cfce5c91b1a30a3d93012cd433c321f910325 |
| R26 | [execute_admitted](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r26_pilot_v1_20261001/scripts/architecture_prototypes/pilot_runtime_entry_r26_v1.py:271) | 47f60d1238b1b5aff18a408e8b5be69c047e7c251641c0ba05f9b9d41347b7db |

现startup guard已支持各候选，不需改动或重启worker/监听。普通model train身份保持历史行为；新T21/T24/T26仅作为预定独立外部执行身份，名字不表示数据已经获准。

## 本轮私有改动边界

源码仅放在`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/private_sources`，不复制整个源码树或修改已验固定包：

- [controller](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/private_sources/scripts/architecture_prototypes/pilot_pool_launcher_v1.py)：保留原R20核心函数，新增三候选固定绑定及执行helpers。
- [执行选择](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/private_sources/scripts/architecture_prototypes/pool_release_dispatcher.py)：只增加三项外部身份，既有28模型和D00/D01/三候选人工验收处理保持。
- [源码准备工具](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/private_sources/scripts/architecture_prototypes/stage_pool_pilot_launcher_v1.py)：仅写自有review artifacts，不apply或生成准入pointer。
- [来源映射准备工具](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/private_sources/scripts/architecture_prototypes/stage_pool_pilot_source_map_v1.py)：通用四身份的stage-only协议，保留历史映射和expected-before合同，不含live写入/CAS apply入口。

旧source-map工具仍硬编码R20，且旧manifest正则不接受当前support_v2命名；其旧23项检查不能替代此次新接入检查。[旧工具](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_source_map_cpu_20261001/isolated_worktree/scripts/architecture_prototypes/stage_pool_pilot_source_map_v1.py)。

## 必须保持的门禁与验收

每项绑定自己的runtime、entry、package、producer、profile及已验GPU目录/五项原件SHA。加载用绝对路径runpy及已有namespace隔离，不提前import live项目模块。原assert_ready、明确training_ready is True、preflight、正确package和完整GPU verify全部通过后才写receipt；不能借R20证明替代三候选。

累计1000/5000、5000明确完整resume、fresh同池output和外部provenance不变量保持。后段绑定前段source/data/selection/profile/GPU/optimizer/scheduler/RNG，不找latest、不跨候选恢复。来源工具只准备review差异，旧映射不隐式重绑定。

CPU只运行新改动相关的固定身份/来源/拒绝检查；不用伪造passed审核证书，不读取真实shard或序列、不初始化CUDA、不重复已通过的四候选GPU验收。检查完成后删自有临时目录和验证checkpoint，保留日志、JUnit、差异和SHA；真实训练恢复状态保持。

## 当前真实数据阻断

共享工程拒绝基准仍是旧expanded目录及其冻结manifest/selection，training_ready=false且双证书缺失。用户指定的`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1`仅作为后续正式目标登记：manifest无training_ready字段、双证书缺；不能把expanded selection或审核SHA用于MHC。有效审核完成后须另做一致的范围、selection、gate与最终manifest版本绑定。[指定版本门禁](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/requested_training_data_gate.json)。本轮不创建任何passing pointer、审核证书或可执行准入manifest。

11:54 UTC只读监督观察：原八卡池0/0/42，8worker均maintenance/standby_load，监听PID2213164存活且仍05:12:16启动；平台缓存11:53:21查询原任务Running。本轮没有新的GPU实验。此前安全检查停止的序列注释分支绝不恢复/重试/转交/绕过，独立CUDA算子保持暂停。

实现agent与独立静态复审agent已完成本次工作；root完成新增CPU验收、原件复核、临时控制目录清理及MD同步。没有声称这些agent仍在持续监听。状态更新至[PROJECT_PROCESS.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)与[统一接手页](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)。

## 2026-10-01T12:08:17.780870+00:00 独立静态复审发现

来源准备工具记录了dispatcher expected-before SHA，但未核对实际before字节；已要求在任何output前拒绝漂移并记录observed值，补充定向检查。其余核心门禁/身份/续训检查静态未见新增问题。当前修正及新CPU checks仍进行中，不能视为通过。[复审观察](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/static_review_observation.json)。

## 2026-10-01T12:17:53.844266+00:00 最终私有工程结果

根端独立源码核对通过：原R20 12个核心函数AST与字节保持；既有dispatcher仅selected_variant/route_name/resolve_route/main四个允许位置改变，原candidate/guard函数保持；6份live源码与4份runtime manifest pins保持。[源码核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/source_review.json)、[源码差异](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/private_source_changes.patch)。固定包未改；源码核对原件中的CPU_pending是当时阶段状态，由下列最终CPU原件补充。

独立静态复审发现1项来源缺口，已在任何stage output创建前核对dispatcher实际before SHA并记录observed值。定向静态确认及新增CPU漂移拒绝case通过。[首轮发现](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/static_review_observation.json)、[修正复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/static_correction_review.json)。没有测试失败后降低阈值；首轮实际CPU即全部通过。

| 环境 | 本次case | 原始JUnit总耗时 | 模型/优化更新 | 原件 |
|---|---|---|---|---|
| pllm | 24/24，0失败/0跳过 | 0.212秒 | 0/0 | [日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/pllm_attempt1.log)、[JUnit](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/pllm_attempt1.junit.xml) |
| worker Python | 24/24，0失败/0跳过 | 0.191秒 | 0/0 | [日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/worker_attempt1.log)、[JUnit](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/worker_attempt1.junit.xml) |

24个唯一参数化case/环境只覆盖：原函数AST、新四身份字面绑定及历史普通model意图、候选transport和5000缺resume、指定MHC拒绝、新alias缺pointer、R20 GPU目录跨用拒绝、四身份历史映射字节增量与禁止隐式rebind、无apply、dispatcher before漂移无output。没有模拟passed证书、pointer或完整admission成功，也没有实际stage/apply发布、fresh-child执行或真实Trainer启动验收。旧48/23/35/44以及四候选GPU验收没有重跑。

[预先CPU预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/cpu_budget.json)、[最终独立复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/validation.json)。最终复核SHA：`d2dd362673124d2a8c3e3262a83d157ec76fe18a8d58845e0d05053b1aa2b4a2`。5项当前source SHA如下；它们是工程版本记录，不是审核证书：

| 文件 | SHA256 |
|---|---|
| [pilot_pool_launcher_v1.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/private_sources/scripts/architecture_prototypes/pilot_pool_launcher_v1.py) | 3977f499fd0e363205248b743eaec675ca084eb0a1816f3bd2b0b32664524840 |
| [pool_release_dispatcher.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/private_sources/scripts/architecture_prototypes/pool_release_dispatcher.py) | 3d8928d85eda6a492b330c586a4da20c6424344ab4e6948810a05af64bc1e69c |
| [stage_pool_pilot_launcher_v1.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/private_sources/scripts/architecture_prototypes/stage_pool_pilot_launcher_v1.py) | c2cee38f9eb3bd216273f0fb747f4d07f24a385df0d850cbf7f2b29d0cec98fa |
| [stage_pool_pilot_source_map_v1.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/private_sources/scripts/architecture_prototypes/stage_pool_pilot_source_map_v1.py) | d646d406409c22e178e10befed77eeb229093a2de674913b7ea2564d9f47ff17 |
| [test_candidate_training_bridges.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/private_sources/scripts/tests/immune_llada/test_candidate_training_bridges.py) | 13eeebd77b0b41b3bbe790224b13d128e83262a40694c9161d27d6485dbaa8b6 |

已删自有两个CPU control roots中的30个普通文件/1118字节和10个别名，未跟随别名；验证checkpoint保留0、私有pyc0。日志/JUnit、差异、源码、预算及证明保留，实际训练恢复文件未动。[清理清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/cleanup.json)。

## 下一动作与真实阻断

当前允许且与最终训练直接相关的私有整合已收尾。指定MHC版本仍无training_ready字段且缺audit_certificate.json/downstream_scope_certificate.json；旧expanded仍false且双证书缺。上述两版本manifest SHA没有变化。真实小batch和正式1000→累计5000比较均未启动；CPU拒绝通过不会使数据合格。

后续须有独立有效审核，明确绑定用户要求的测试/query保护范围、输入SHA、gate版本、双证书与最终manifest及selection；此前停止的注释分支绝不恢复/重试/转交/绕过。收到合格版本后再另作四候选一致的新数据绑定和来源发布；旧expanded selection不能搬给MHC。然后在原八卡按已记录候选×匹配seed设计比较，依据理解/生成/成本指标做架构决策。不能只添加字段或文件绕过原门禁。

截至本节点，没有新的GPU实验、submit/cancel、worker/监听重启或live source/map写入。独立CUDA算子继续暂停。剩余真实训练推进依赖外部有效数据审核；不以额外无关人工曲线填充Goal，也不标记Goal complete。当前三方架构决定仍按[决策页](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_CHANGE_DECISION_20261001.md)执行。

## 2026-10-01T12:22:34.503809+00:00 外部审核阻断再次核实

上一Goal回合为实际私有整合/CPU验收/清理进展；本回合仅一次metadata和原池现状核实，不记为新增实验或训练进展。MHC manifest无ready字段、旧expanded为false，双版本均缺双证书且SHA未变。八卡0/0/42、8worker维护，监听和原任务保留；没有实际训练进程可等待。已记录同一外部阻断第二次观察，原Goal仍active且未完成；不重跑测试/模型、不恢复注释、不降低门禁、不做无关验证。[本次只读核实](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/external_review_recheck_2.json)。

## 2026-10-01T12:28:08.781013+00:00 原Goal正式记为blocked

同一外部数据审核阻断已在三个连续Goal回合核实。独立agent确认当前授权通用私有工程已收尾，未发现不依赖审核、直接推进最终真实训练的必要工程项。第三次metadata仍MHC无ready字段/双证书缺、expanded为false/双证书缺且SHA不变；原池0/0/42，无实际训练进程可等待。按原Goal工具实际返回已设blocked，目标身份及累计用量保留，不声称完成。[阻断审计与实际状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/external_review_recheck_3.json)。

后续需有效外部审核与一致的范围/input/gate/manifest/selection/ready绑定，之后才可另做来源发布、fresh-child与真实小batch验收及正式训练；这些后续仍未完成。用户恢复此前blocked Goal时，按新审计周期核实，不能沿用旧三次计数。停止的序列注释不恢复/重试/转交，数据门禁和独立CUDA暂停保持。当前没有重启/取消原八卡或监听，没有新增实验、模型测试、shard读取或checkpoint；原证据、真实恢复状态与资源保持。

## 2026-10-01T16:26:25.076053+00:00 原Goal恢复与用户问题：为什么审核仍未完成

已有清理和部分审计；最终数据审核尚未收尾。旧clean_r2 audit_certificate状态pending，标签unresolved_benchmark_h3；MHC输入更新没有对应新manifest的双证书和明确ready。expanded导出报告仍有role coverage/near-neighbor review、独立parent/expanded scope复扫、双证书及upstream parent-antigen溯源pending。此前补覆盖的序列注释分支被安全检查中止，按用户要求没有恢复；后续模型工程检查并未解决这些数据项。最新test/query保护决定与旧gate SCOPE/input清单也未形成一致的新审核绑定。缺证书反映上述审核工作未完成，不是机器权限申请；这些记录不证明新范围已有残留泄漏。

[恢复后一次原始核实及证据索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/resumed_external_review_1790871604_1.json)、[旧pending证书](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_clean_r2/audit_certificate.json)、[导出pending报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/expanded_candidate_export_report.json)。旧审核证明不能移给新manifest，文件/源码/CPU验收不代替真实数据审核。保留停止注释、数据门禁、原八卡/监听与独立CUDA暂停；本回合无新模型/测试/真实小batch。恢复后的新审计为第1次，不沿用旧3次。

## 2026-10-01T16:28:55.684858+00:00 恢复周期第二次同一阻断核实

上回合为审核原因解释及metadata核实，未推进真实训练。本回合数据与审核状态未变化；MHC无ready字段/双证书，expanded为false/双证书缺，原manifest SHA保持。原八卡0/0/42、8worker维护且无job，原监听和任务存活，无实际训练进程可等待。当前允许且必要的独立工程已收尾，不重跑已过验收、不做无关验证，不恢复停止的注释或修改gate。新恢复周期第二次观察，原Goal仍active，旧三次计数不复用。[本次原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/resumed_external_review_1790871604_2.json)。

## 2026-10-01T16:34:42.531238+00:00 恢复周期数据审核阻断确认

恢复周期连续3次核实同一数据审核阻断，独立复审未发现可独立推进的必要工程；原Goal已实际设为blocked，原身份、目标和累计用量保留，未完成。MHC无明确ready及双证书，expanded仍false且缺双证书，manifest SHA未变。原八卡0/0/42、8worker维护无job、监听PID2213164存活；真实小batch和最终训练未启动。本轮无模型/测试/shard读取、提交/取消/重启或新增checkpoint；不恢复注释、不改门禁、不重复验收。 [本周期第三次审计及实际状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/resumed_external_review_1790871604_3.json)、[Goal工具实际返回](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/resumed_blocked_goal_update_1790871604.json)。当前状态覆盖下方带日期历史。

