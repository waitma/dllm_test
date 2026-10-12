# 真实语料进入训练的准备状态

更新：2026-09-30。本轮核验时间约 20:33–20:44 UTC。结论：**当前两个 v7 候选均不能通过真实训练入口；主要缺口是 H3 的角色/注释判定，以及扩展 TCR/epitope/support/query 范围的清理和重新认证，不是再等一个训练作业。**

后续证据更新：264 项已进一步完成逐位编号、作者角色和两个 ENA 原始参考核验，其中 184 项获得角色或实际观测范围解释、80 项仍有注释缺口。详见 [IMMUNE_V7_OBSERVATION_ROLES.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_V7_OBSERVATION_ROLES.md)。这是新的前置证据，未完成角色近邻扩展、未改数据或签证，因此下面的权威门禁状态不变；第三节的“下一步”应结合新文档阅读。

本轮没有改 prepared 数据、benchmark、证书、清理规则、live 模型源码或共享根 MD，没有提交训练。当前主机进程快照未发现 `prepare_immune_v7`、`audit_tcr_exposure`、`recover_v7_h3` 或 MHC 数据更新正在运行；已有 AB–antigen 评测和 pool 监听继续运行，没有停止/重启。该检查只覆盖当前主机，不能证明其他主机或会话没有尚未启动的工作。

## 1. 权威门禁状态

权威入口：[assert_ready](/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/immune_pretrain_v3.py:39)，扩展范围门禁：[tcr_scope_gate.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/tcr_scope_gate.py)。目录名称中出现 clean/MHC 不等于获准训练。

| 数据目录 | 旧范围 audit_certificate | expanded downstream_scope_certificate | 实际入口结果 |
|---|---|---|---|
| [immune_v7_clean_r2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_clean_r2) | `pending`，唯一 pending 为 `unresolved_benchmark_h3`，264 项 | 缺失 | `ValueError: immune_v7 audit is incomplete` |
| [immune_v7_mhc_sequence_v1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1) | 缺失 | 缺失 | `FileNotFoundError: audit_certificate.json` |

实际调用证据：[actual_gate_calls.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/actual_gate_calls.json)。两次调用都在全分片 hash 扫描前拒绝；本轮没有把它写成新的全量扫描。

两版本均为 train **6,849,243**、valid **52,572**，总计 **6,901,815** 条，81 个逻辑分片。MHC 版本有 71 个分片是指向父版本的相对 symlink，10 个分片重写 MHC；逻辑数据约 12.62 GB。manifest SHA：

- clean_r2：`d9cd4f281705f674733a6f35257fb15c8e1f6fa69657bfc21f58c2c2383cd383`
- MHC：`e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94`

父版本的[独立复扫证书](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_clean_r2/audit_certificate.json)确实记录旧范围 `forbidden_overlap_rows=0`、`family_split_overlap=0`、所有 7 个来源都有 train/valid，但仍有 264 项 H3 覆盖 pending。它不覆盖随后纳入的完整支持/查询范围。

MHC 的 [validation](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1/mhc_validation.json) 已通过数据/loader/layout/backprop 检查；[更新报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1/mhc_update_report.json)声明并检查了行成员、顺序、split、label 和非 MHC 链内容不变。因此它不会消除已发现的 TCR/epitope 暴露，也不能继承另一 manifest 的清理证书。

## 2. 已保护的范围和仍有的边界

旧 registry：[immune_v7_20260930/registry.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/immune_v7_20260930/registry.json)，包含 68 个输入条目；已建六类同源聚类：heavy 95% identity、H3/CDR3a/CDR3b/peptide 80%、antigen 70%，均要求既定的 80% coverage。短 peptide 另有编辑近邻筛查；antigen 精确包含要求双向 coverage≥80%，避免短片段误删长蛋白；held-out epitope-in-antigen 是独立规则。上述 identity 与 normalized Levenshtein 的 80% 不是同一个指标。

旧范围包括已有 SAbDab、light pairing、FLAb、NBbench、OTS holdout、TCR generation 等，以及新增 AB–antigen 数据：

| AB–antigen 任务 | 保护记录数 | 声明为 unseen-antigen 的记录数 |
|---|---:|---:|
| binding | 363 | 271 |
| SPR | 217 | 0 |
| AbBiBench | 215,697 | 0 |
| CHIMERA | 545 | 545 |

binding/SPR 取既定 split 各轴的 val/valid/test 并集；AbBiBench 保护整套 assay；CHIMERA 保护既定 val/valid/test。AbBiBench 不是默认 unseen-antigen 任务，不能把“未将所有 assay antigen 按陌生抗原过滤”误写成漏做其已声明协议。H/L/pair、WT heavy 的纳入逻辑见 [prepare_immune_v7.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/prepare_immune_v7.py:88)。

扩展 registry：[tcr_exposure_20260930_r2/registry.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/tcr_exposure_20260930_r2/registry.json)，scope=`tcr_all_support_query_uptcr_v1`，24 cohorts、374,373 行：

- UpTCR majority/few-shot/zero-shot × train/valid/test，共 9 个池；train 在这里是需保护的 downstream support/adaptation，不作为 foundation 无限制训练数据。
- TCR representation train/support 和 test；paper6 target_binders 与完整 background_pool，避免只保护一次 few-shot 抽样而漏掉后续可被抽中的记录。
- 两类 TCR clustering，TCR binding test，以及 NM2025 seen/unseen 的 8 个 test 文件。
- 分开检测 CDR3a/b、完整 FVa/FVb 精确重复、alpha-beta 配对、alpha/beta–peptide、AB–peptide 和 beta–peptide–MHC 的同行联合键；短 core/peptide 另做穷举 global normalized Levenshtein≥0.8，及精确 epitope-in-antigen。

本轮重新核对 **92 个范围条目、79 个唯一文件、691,838,484 bytes**，全部与冻结 SHA 一致：[benchmark_bindings.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/benchmark_bindings.json)。没有改下游官方划分、few-shot 池或 benchmark 行。

现有 expanded 审计明确限制：完整 FV 只查精确重复，CDR3 才做上述短序列近邻；HLA 名称不猜映射；部分 background 类别没有真实 peptide 元数据；缺少上游 parent-antigen accession/全蛋白映射，不能宣称 parent-protein novelty。已有原始 peptide 独立扫描与语义扫描一致，不能因为 joint pair 没命中，就忽略 epitope 单独暴露。非标准 CDR3 端点的来源语义，以及训练数据中 3 个 missing/invalid beta-key 观察值仍需处理。输入 C/F/W 残基不因审计的 core 归一化而删除或补造。

## 3. 264 个 H3 缺口已经追到原始来源

原覆盖/诊断：[benchmark_h3_coverage.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/immune_v7_20260930/benchmark_h3_coverage.json)、[remaining_h3_diagnostic.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/immune_v7_20260930/remaining_h3_diagnostic.json)。此前 ANARCI 已恢复 9,149 项，2,463 项通过已有受保护 source core/native reference 覆盖；剩余 264 不是单纯少跑一次编号。

本轮对 68 个冻结输入逐行只读追溯，**264/264 全部找到**，落在 6 个 NBbench 来源：[h3_source_provenance.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/h3_source_provenance.json)。该文件提供 sequence SHA、原编号分类、文件、零基行号、字段和显式 core 候选，不自动宣布覆盖已解决。

| 独立序列数 | 具体发现 | 下一步需要验证 |
|---:|---|---|
| 133 | nanobody_type 的 71 个 V-lambda、62 个 V-kappa；与既有 ANARCI L/K 分类一致 | 这是多分类任务的合法轻链类别，应按 light 角色保护，不能要求其拥有 heavy H3 |
| 35 | nanobody_type 的 shark VNAR，57 条源记录；被通用编号诊断标为 no-domain | 不能默认是残缺 VHH；需适用该受体类型的注释/覆盖证据 |
| 4 | CDRInfilling/VRClassification 的两条原记录衍生输入/参考，显式 CDR3 仅 2 AA：`VA`、`EV` | 旧 bank 的 `add_cdr3(min_len=3)` 排除了它们；需核实源注释与短-loop保护策略，不能仅改状态或补造残基 |
| 75 | 有 H domain 但未编号出 CDR3，来自 polyreaction 50、hIL6 18、SARS-CoV-2 7 | 核实完整性、源 native/目标注释，不能把缺失标为 clean |
| 17 | 其余 no-domain：hIL6 12、SARS-CoV-2 5 | 来源、角色与完整性判定 |

作者本地数据 [nanobody_type/README.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_raw/nbbench/hf_data/nanobody_type/README.md:25) 明确 label2=shark VNAR、label3=human V-lambda、label4=human V-kappa。本轮把 README 的 SHA 也记入辅助证据，避免仅凭序列外观判定角色。

**不能只把 light 的 H3 填为 N/A 后放行。** 旧 nanobody reader 把所有类型都加入 heavy bank，export 的 RULES 没有 light 单链近邻筛查，只有 heavy/H3 与精确 H–L pair 等规则。必须同时证明角色纠正后的保护仍覆盖应保护的 light 输入，不能通过角色变化丢掉保护。

作为小成本前置工作，已对 133 个 light 与 35 个 VNAR 在既有全量源 `inventory.sqlite` 的 heavy/light/H3 键上点查询：168 个均未找到精确出现；1,554,051,072-byte SQLite 重新算 SHA，与冻结 near_run 一致，耗时 1.45 秒。[role_specific_inventory_check.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/role_specific_inventory_check.json)。这不是完整角色近邻筛查，没有把这些序列标成 resolved，也没有签发证书。

## 4. 已有清理候选：精确与仅近邻不能混写

已完成的 v7 expanded 全量扫描耗时 534.17 秒，检查了原 manifest 对应的全部 6,901,815 条；其 [report.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/tcr_exposure_20260930_r2/immune_v7_clean_r2/report.json) 为 `overlap_found`。[cleanup_plan.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/tcr_exposure_20260930_r2/cleanup_plan.json) 仍为 **not_applied**，绑定父 clean_r2 manifest，不绑定 MHC manifest。

本轮流式复核既有 230.7 MB row-hit 清单并核对 SHA、分片行号范围及逐来源计数；新增 [existing_hit_breakdown.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/existing_hit_breakdown.json)。下表中“任一精确理由”和“仅近邻”互斥；精确理由可能是单链、同行配对、peptide 或 epitope 包含，不等于整条记录精确复制。

| Train 来源 | 原行数 | 任一精确理由 | 仅近邻，无精确理由 | 候选全并集标记 | 若全排除，剩余 train / valid |
|---|---:|---:|---:|---:|---:|
| OAS | 2,472,502 | 0 | 0 | 0 | 2,472,502 / 24,857 |
| ASD antibody | 203,189 | 35,798 | 0 | 35,798 | 167,391 / 595 |
| OTS | 1,701,193 | 126,417 | 1,497,524 | 1,623,941 | 77,252 / 530 |
| TCR native | 22,508 | 19,869 | 1,993 | 21,862 | 646 / 4 |
| TCR papers | 437,770 | 152,126 | 204,351 | 356,477 | 81,293 / 732 |
| TCR repertoire | 2,001,488 | 1 | 65,474 | 65,475 | 1,936,013 / 19,173 |
| TRAIT | 10,593 | 7,765 | 1,977 | 9,742 | 851 / 4 |

train+valid 的全并集共 **2,119,972** 条，其中任一精确理由 **343,422**、仅近邻 **1,776,550**。OTS train 的精确 alpha-beta 同行配对只有 **253** 条，不能把 1,623,941 条全写成“精确配对泄漏”。ASD 的新增 35,798 条全部属于 epitope 精确包含，不能与 TCR 单链相似度混淆。不同 cohort/reason 可命中同一行，不能把逐任务/逐理由数量相加。

若机械应用全部候选并集，理论剩余 4,781,843 条（train 4,735,948 / valid 45,895），但 native/TRAIT 各仅 4 条 valid，这不足以支持稳定的来源级选优。上述全是**候选影响量**，没有执行删除或清理导出；近邻不是已证实的生物学泄漏比例。

## 5. 到可训练数据的最小完整路径

1. **冻结可执行的范围和角色判定。** 复用本轮已核对的 92 个输入绑定；根据 source label/native reference 处理上述 light/VNAR/2-AA/不完整 H，明确每项的保护角色与证据。对缺失字段做判定或隔离；保留官方 benchmark 行。不能将无法证明的情况直接改为 resolved，也不能仅扩大模型输入中可见的目标信息来解决审计问题。
2. **确定并记录 expanded 排除规则与 validation 设计。** 精确、单链近邻、epitope、同行联合键分别保留理由；沿用已声明阈值，不把高标记比例视为自动放宽阈值的理由。若采用现有严格并集，应先设计来源足量且按整个 `v7_family_group` 划分的 train/valid，再按同一规则导出新的候选目录；不得随机移动个别同族记录填 valid。可以先导出固定、严格筛选的小规模真实语料 pilot，但其 manifest、来源覆盖、家族隔离和全部适用范围证据也必须完整，不能凭子集名称绕过证书。
3. **以新版本导出，保护原数据。** 对 MHC 版本重新绑定排除计划：行号可以利用 MHC 已验证的不变性做映射，但每个新分片 SHA、manifest SHA 和报告必须重新产生。保留去除理由、来源和 split 计数、家族组及 MHC provenance。不要把 clean_r2 的旧 certificate/row-hit manifest SHA 复制到 MHC 目录。
4. **对确切的新产物独立复扫。** 旧 AB/H3/antigen 范围、expanded TCR/epitope/support/query 范围都复扫；核对原/新行号、分片 SHA、非空来源 split、family overlap、遗漏注释和所有禁止命中为零。新数据仅为已查库存的删选/重分组且非 MHC 序列不变时，可复用 hash 绑定的近邻索引；有新序列/规则/输入时才补相应近邻或聚类，不能静默复用过期索引。原近邻三阶段已用时约 162.15+858.18+0.08 秒，全量 source 扫描约 9 分钟，仅供同规模成本参考。
5. **只从通过的实测证据签发两份新证书。** 两份证书绑定同一个最终 manifest；`pending=[]` 必须对应实际缺口已解决，`forbidden_overlap_rows=0`、`family_split_overlap=0`，expanded proof 包含当前每个 benchmark 输入 SHA 与独立 evidence 文件 SHA。上游 parent-protein 来源不足时只能维持明确的 observed-sequence 审计范围，不能把它包装成 parent-antigen novelty 证明。
6. **实际运行 assert_ready，再接入真实 pilot。** 主线程更新受控 plan 的 data_dir 到新目录；所有候选共享同一版本、样本顺序、预算与来源级 validation。通过工程诊断、数据和 measured-profile 三重门禁后再开始 1000 步；5000 步选择以固定 validation 证据为准。这份文档本身不是训练授权证明或证书。

## 6. 可复用工具与需要补齐的实现

| 工具 | 可复用部分 | 不能直接做的事 |
|---|---|---|
| [prepare_immune_v7.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/prepare_immune_v7.py) | freeze/inventory/cluster、键提取、family graph、版本化 export、独立 verify 的机制 | `verify_export` 会写 manifest/certificate，且硬编码 `immune_v7_clean_r2`、依赖 cleaning_report；**不要直接对当前 MHC 目录调用来“补证”**。需要新版本适配器保留 MHC/version 元数据 |
| [recover_v7_h3.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/recover_v7_h3.py) | 已有 heavy ANARCI 编号恢复与 SHA 绑定 | 重跑同一方法不会自动解决 light/VNAR/短-loop语义问题；不能猜锚点 |
| [audit_tcr_exposure.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/audit_tcr_exposure.py) | 当前24池、exact/near/epitope与同行联合键、逐行命中和实际分片验证 | 当前只有 freeze/near/scan，**没有应用清理或签发 scope certificate 的入口**；零命中时也输出 review_required，不能拿扫描成功当正式证书 |
| [update_mhc_inputs.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/update_mhc_inputs.py) | MHC替换、非MHC逐行不变性、symlink复用及兼容性验证 | 不负责去污染认证；当前版本映射固定，不能无适配地套到新 clean 数据目录 |
| [tcr_scope_gate.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/tcr_scope_gate.py) | 当前 scope、manifest、benchmark 和 evidence SHA 校验 | 是验证器，不是证书签发器 |
| [本轮 audit_readonly.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/audit_readonly.py) | 92输入SHA、264项来源定位、精确/近邻分解、证书状态快照；33.68秒 | 不做清理、角色裁定、近邻重算或签证；只向自身输出目录写证据 |

本轮还明确发现：**expanded 清理导出器、与最终 MHC manifest 绑定的双范围独立验证/证书签发流程，需要补实现并测试**。当前 repo 已有检查门禁和审计产物，不等于已有一条完成上述全部步骤的一键命令。

## 7. 本轮证据索引

总状态：[readiness_snapshot.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/readiness_snapshot.json)。实际门禁调用、来源 SHA、H3 行级定位、精确/近邻分解及角色点查询均保留为小型 JSON，没有复制全量语料或保存模型权重。源码、证书、原 prepared 与 benchmark 未修改；没有为了填满 GPU 而绕过数据门禁。共享项目账本与下一阶段实现由主线程同步。
