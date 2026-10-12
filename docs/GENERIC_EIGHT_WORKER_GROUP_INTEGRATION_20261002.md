# 原八worker的单模型八rank接入

## 2026-10-02T11:03:50.603902+00:00 当前原CE500学习验收已闭合

CE R21/profile43原500于10:56:27 UTC succeeded/rc0，原verify_diagnostic(500,require_learning=True)唯一root只读调用及独立stdlib复核均通过，trace25..500共20点。学习seed42：四mask平均loss4.0881584711→0.1847930732，AA准确率6.38465%→92.65651%，terminal100%；完全遮蔽mask100 AA71.54613%、loss0.7071843，平均不能称全遮蔽/生成准确率。39当前source物理SHA、9实际输入及62独立引用哈希、日志/双COMPLETE/实际子退出清理均一致，checkpoint目录0。

截至2026-10-02T11:03:50.603902+00:00，原池pending=0/running=7/results=66；CE已完成验收，其余七项最新步数为375、375、375、375、375、375、375/500，仍需真实终态后各自一次原验收。GPU1回原维护状态，没有重启worker/作业/监听或新增GPU任务。

该证据只证既有当前来源人工单GPU学习，不证真实V7/生物下游效果、单模型八GPU-DDP或正式恢复；profile43不是新学习seed。不同来源、不同步数及相同learningseed42的各profile不能当独立多seed比较或最终选型。原CPU Trainer/通信验收保持完成，不重复；正式资源接入及实际共同GPU保存恢复仍未完成。

本回合重新仅读V7清单元数据：manifest仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94、training_ready字段缺失、audit_pending，两份当前审核均缺。合法当前完整下游资源范围清单及绑定格式缺失，原正式gate无法正确定稿；不造清单/证书、手改ready、降低门禁。真实shard/小batch/审核selection/训练0；停止注释不恢复/重试/转交/绕过、独立CUDA算子暂停、原资源不重启或重复提交。

当前验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_root_acceptance.json；原只读调用尝试 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_root_attempt.json；independent /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_independent_review.json；最新原任务快照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_jobs_after_CE500_independent_closure.json；V7 metadata /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/V7_metadata_after_original_CE500.json。

## 以下为较早记录

当前状态 2026-10-02T10:34:05.741083+00:00

本轮新增八进程通信/清理控制两环境首次各4/4通过：worker5.747秒、pllm7.253秒。每环境8实际parent和8实际CPU/Gloo cold rank全部rc0/wait/自有进程组已清，rank0持有FD直接交TCPStore；倒序member正确映射rank0..7、实际store/all-gather8身份一致、缺员不可逆abort和未登记失败child rc2清理均通过。26旧文件及9冻结source不变，临时目录0、checkpoint0，不重复已完成numeric Trainer。root原始证据与独立只读复核均通过；root复核88原件、独立报告103原件物理SHA，9source和26旧保护文件不变。完整本阶段验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_CPU_transport_root_acceptance.json。

此证据只覆盖新增CPU通信及词表列表元数据格式校验；原生产Trainer类组装仅静态复审完成，未实际构造/训练；新词表callback未通过实际native保存恢复。实际单模型8GPU-DDP完整保存恢复、正式外部资源参数全调用链和最终架构选型仍未完成。

截至2026-10-02 10:38:29 UTC，原同八项500步合成学习仍reported running，poll-gated心跳0.47–1.62秒：CE R21/profile43为450/500，其余七项325/500；pool pending0/running8/results65，八项各自pool result及内外COMPLETE均缺。八项是独立单GPU，不是单模型八GPU训练；不同来源/步数和同learning seed不能直接选优。

[最新原任务与V7快照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_jobs_and_V7_readonly_snapshot.json)；[本轮根端复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_CPU_transport_root_review.json)；[新增工程记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_WORKER_GROUP_INTEGRATION_20261002.md)。

## 以下为来源准备与修订记录

更新 2026-10-02T10:03:47.608079+00:00。上一Goal回合完成现有两环境八CPU Trainer及独立raw/保存恢复复核；本回合继续必要worker组调度/实际原Trainer类组装。当前只私有源码，不发布/入队GPU、不改旧worker/模型/validator/immutable来源。

旧worker自行claim，member不能预先冒充rank。八remote parent先汇合固定exact8成员的真实单卡mask、UUID/PCI、host/boot和父身份，按mask0..7定globalrank；GPU0 parent创建已有coordinator。端口保持loopback socket FD并交rank0 TCPStore(master_listen_fd,use_libuv=False)，不能选号后释放；实际API尚未验。父进程必须监控abort、wait并清自有整个PG。登记前失败保留未登记，另记录八member实际清理，不伪造旧coordinator八exits。

实际TokenValidationTrainer loss/evaluation_loop/_save可原样继承，world1仅外层evaluate由已验factory覆盖；新私有synthetic构造器使用原六protocol/world8 micro32 GA1 global256、原horizon10000/LR1e-4/warm100/cosine/evalsave100、8 RNG callback。它不替代正式data/resource/preflight/selection/execute/resume链；当前资源审核inventory缺失，正式生产接入仍未完成。

当前v2预算最多5source/每环境4个新定向CPU控制、180秒，不重跑已通过numeric Trainer或旧模型验收。来源和预算冻结/静态review先于任何运行；GPU验证0。旧8项500仍运行（最新CE375、其他七项225），先等原终态，不重启/重复提交/加架构波次。

V7 manifest仍audit_pending、training_ready字段和当前双审核缺失；真实分片/小batch/训练0，原所有门禁保留。停止注释不得恢复、重试、转交、绕过，独立CUDA算子暂停。

预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/execution_budget_v2.json；保护源 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/protected_before.json；当前机器状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/EXECUTION_STATE.json。


## 2026-10-02T10:09:29.380497+00:00 原词表列表保存格式的实际兼容修复

原模型_save保存vocabulary.json为list，冻结numeric callback要求dict，无法直接接入。新显式私有回调 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/eight_rank_checkpoint_adapter_v2.py 仅_metadata保留原非空unique字符串列表、其余3dict，其他全部函数AST不变；不用FunctionType/globals或改词表包装，旧源/阈值/门禁/native/8RNG不改。新SHA 21815b3fd54ee51481f72f77cce1879d2fdf9d0e4818665c2cf72531169084b8 不继承旧GPU/CPU通过声明，当前未运行。新增1个必要source，修订预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/execution_budget_v2.json SHA 43ed1ccfc4cbeb71dbaf58d82781e3d1e6c8a473dab63d4b8ccebf0598a6657e；原未执行预算保留，CPU控制仍4/env180秒、GPU/发布/真实训练0。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/checkpoint_vocabulary_source_delta.json。


## 2026-10-02T10:31:00.164239+00:00 冻结与首次执行待开始

5个新私有source及4个复用source已冻结，26旧保护文件未改。A/B接口复审完成；root补自有进程清理helper抛错时的有界fallback，C只修顶部CLI说明。首次每环境4项CPU transport/metadata控制待执行；不重跑已通过Trainer。此阶段无GPU作业/真实数据读取/生产准入，实际八GPU共同Trainer和原native词表保存恢复仍待验证。


## 2026-10-02T10:44:51.097287+00:00 当前收尾与剩余外部资源阻断

新增八进程通信/清理的两环境首次4/4及独立raw复核已闭合：root88与独立103原件哈希重新核实，9冻结source/26旧文件不变，temp及checkpoint均0，不再重复已通过控制。实际单模型八GPU训练/原Trainer构造/native词表保存恢复仍未验证；生产入口未接通，真实语料训练0。

正式接入的真实阻断是缺当前V7合法完整scope/input inventory/registry与报告绑定格式，不仅是两份审核尚未passed。调用链会重复原固定资源gate、请求重建丢附加字段，selection还要求旧payload完整重算相等；自行选短path/SHA清单会缩减原保护范围。不能补假清单/证书、手改ready、monkeypatch旧ROOT或加入永拒placeholder充数。证书未通过本身不禁止通用开发，但没有权威完整资源契约就无法正确定稿这部分formal gate。

外部必要材料：当前scope_version及完整逻辑输入清单；批准的外部根、逻辑ID→绝对路径映射及文件/registry/evidence哈希格式；绑定当前manifest和同一输入全集的双审核字段语义；各候选/selection/执行/续训共同使用的同一版本资源契约。未来真实执行仍要求training_ready is True和两份有效当前审核，不降低任何原门禁。

当前可继续核实同8原任务的实际500终态并各自一次原verify_diagnostic(500,require_learning=True)；它们仍Running时不得因观察超时重启。后续单模型GPU/正式接入须满足当前来源及资源条件，不能用新numeric/GPU测试填补外部资源缺口。待权威契约到位，最小后续六类source为参数化资源gate、selection外envelope、正式request/preflight/build、execution/fullresume、cold entry及immutable package；此处仅明确必要边界，没有创建或执行六类实现。

只读决策 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/formal_resource_remaining_implementation_decision.json（d2631a1f…ba5c），root物理核5原source SHA及10实际调用位置；CPU阶段验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_CPU_transport_root_acceptance.json；10:43实际平台Running原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/platform_task_readonly_continuation_1043.json；10:38同8原worker poll-gated/trace仍CE450、其余325，无目标终态。

此前停止序列注释分支不恢复/重试/转交/绕过；独立CUDA算子暂停。原容器/作业/worker/监听不重启，不重复平台提交；不读取真实shard/抽真实小batch/执行审核selection/开始真实训练。全部原data/scope/source/selection/learning/profile/GPU/full-resume门禁保留。
