# V7 数据与八卡训练的当前状态

## 2026-10-02T11:03:50.603902+00:00 当前原CE500学习验收已闭合

CE R21/profile43原500于10:56:27 UTC succeeded/rc0，原verify_diagnostic(500,require_learning=True)唯一root只读调用及独立stdlib复核均通过，trace25..500共20点。学习seed42：四mask平均loss4.0881584711→0.1847930732，AA准确率6.38465%→92.65651%，terminal100%；完全遮蔽mask100 AA71.54613%、loss0.7071843，平均不能称全遮蔽/生成准确率。39当前source物理SHA、9实际输入及62独立引用哈希、日志/双COMPLETE/实际子退出清理均一致，checkpoint目录0。

截至2026-10-02T11:03:50.603902+00:00，原池pending=0/running=7/results=66；CE已完成验收，其余七项最新步数为375、375、375、375、375、375、375/500，仍需真实终态后各自一次原验收。GPU1回原维护状态，没有重启worker/作业/监听或新增GPU任务。

该证据只证既有当前来源人工单GPU学习，不证真实V7/生物下游效果、单模型八GPU-DDP或正式恢复；profile43不是新学习seed。不同来源、不同步数及相同learningseed42的各profile不能当独立多seed比较或最终选型。原CPU Trainer/通信验收保持完成，不重复；正式资源接入及实际共同GPU保存恢复仍未完成。

本回合重新仅读V7清单元数据：manifest仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94、training_ready字段缺失、audit_pending，两份当前审核均缺。合法当前完整下游资源范围清单及绑定格式缺失，原正式gate无法正确定稿；不造清单/证书、手改ready、降低门禁。真实shard/小batch/审核selection/训练0；停止注释不恢复/重试/转交/绕过、独立CUDA算子暂停、原资源不重启或重复提交。

当前验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_root_acceptance.json；原只读调用尝试 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_root_attempt.json；independent /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_independent_review.json；最新原任务快照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_jobs_after_CE500_independent_closure.json；V7 metadata /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/V7_metadata_after_original_CE500.json。

## 以下为较早记录

核实时间 2026-10-02T10:34:05.741083+00:00；原任务/数据快照时间10:31:15 UTC。用户已有条件开训授权，无需再次申请权限。

V7为immune_v7_mhc_sequence_v1，train6,849,243/valid52,572，manifest SHA e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94。training_ready字段缺失、audit_pending；audit_certificate.json与downstream_scope_certificate.json均缺。缺审核不等于已证明数据泄漏；用户认可或操作权限不能替代当前manifest及新增下游范围审核。真实分片/小batch/语料训练0。

候选R20基础对照/R21注意力头门控/R24固定残差缩放/R26链内局部与全局注意力交替已实现；最终选择尚未完成。

截至2026-10-02 10:38:29 UTC，原同八项500步合成学习仍reported running，poll-gated心跳0.47–1.62秒：CE R21/profile43为450/500，其余七项325/500；pool pending0/running8/results65，八项各自pool result及内外COMPLETE均缺。八项是独立单GPU，不是单模型八GPU训练；不同来源/步数和同learning seed不能直接选优。

此前两环境numeric八CPU Trainer保存恢复、8RNG和指标尾批复核已通过，本轮未重复。本轮新增八进程通信/清理控制两环境首次各4/4通过：worker5.747秒、pllm7.253秒。每环境8实际parent和8实际CPU/Gloo cold rank全部rc0/wait/自有进程组已清，rank0持有FD直接交TCPStore；倒序member正确映射rank0..7、实际store/all-gather8身份一致、缺员不可逆abort和未登记失败child rc2清理均通过。26旧文件及9冻结source不变，临时目录0、checkpoint0，不重复已完成numeric Trainer。root原始证据与独立只读复核均通过；root复核88原件、独立报告103原件物理SHA，9source和26旧保护文件不变。完整本阶段验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_CPU_transport_root_acceptance.json。

此证据只覆盖新增CPU通信及词表列表元数据格式校验；原生产Trainer类组装仅静态复审完成，未实际构造/训练；新词表callback未通过实际native保存恢复。实际单模型8GPU-DDP完整保存恢复、正式外部资源参数全调用链和最终架构选型仍未完成。

两份审核分别覆盖当前V7数据清理/划分与完整实际新增下游test/query及few-shot support保护范围；必须绑定当前manifest。缺文件不是操作权限问题，也不证明数据有泄漏。

下一步收尾原八项500实际终态，各自只执行一次原verify_diagnostic(500,require_learning=True)及原始结果/来源/退出清理核验。原八卡全部空闲后，在单独冻结当前来源与有限预算下，补必要通用单模型GPU组及原Trainer保存恢复证据；不新增架构波次，不重跑旧通过验收。真实数据训练须当前有效审核及原全部data/scope/source/selection/learning/profile/GPU/full-resume条件满足。

停止序列注释不恢复、重试、换agent或绕过；独立CUDA算子暂停。原容器/worker/监听2213164不重启、无重复Volc提交，本轮新增GPU/平台submit/cancel/restart为0；不造证书/手改ready/降低门禁。

[最新原任务与V7快照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_jobs_and_V7_readonly_snapshot.json)；[本轮根端复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_CPU_transport_root_review.json)；[新增工程记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_WORKER_GROUP_INTEGRATION_20261002.md)。


## 2026-10-02T10:44:51.097287+00:00 当前收尾与剩余外部资源阻断

新增八进程通信/清理的两环境首次4/4及独立raw复核已闭合：root88与独立103原件哈希重新核实，9冻结source/26旧文件不变，temp及checkpoint均0，不再重复已通过控制。实际单模型八GPU训练/原Trainer构造/native词表保存恢复仍未验证；生产入口未接通，真实语料训练0。

正式接入的真实阻断是缺当前V7合法完整scope/input inventory/registry与报告绑定格式，不仅是两份审核尚未passed。调用链会重复原固定资源gate、请求重建丢附加字段，selection还要求旧payload完整重算相等；自行选短path/SHA清单会缩减原保护范围。不能补假清单/证书、手改ready、monkeypatch旧ROOT或加入永拒placeholder充数。证书未通过本身不禁止通用开发，但没有权威完整资源契约就无法正确定稿这部分formal gate。

外部必要材料：当前scope_version及完整逻辑输入清单；批准的外部根、逻辑ID→绝对路径映射及文件/registry/evidence哈希格式；绑定当前manifest和同一输入全集的双审核字段语义；各候选/selection/执行/续训共同使用的同一版本资源契约。未来真实执行仍要求training_ready is True和两份有效当前审核，不降低任何原门禁。

当前可继续核实同8原任务的实际500终态并各自一次原verify_diagnostic(500,require_learning=True)；它们仍Running时不得因观察超时重启。后续单模型GPU/正式接入须满足当前来源及资源条件，不能用新numeric/GPU测试填补外部资源缺口。待权威契约到位，最小后续六类source为参数化资源gate、selection外envelope、正式request/preflight/build、execution/fullresume、cold entry及immutable package；此处仅明确必要边界，没有创建或执行六类实现。

只读决策 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/formal_resource_remaining_implementation_decision.json（d2631a1f…ba5c），root物理核5原source SHA及10实际调用位置；CPU阶段验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_CPU_transport_root_acceptance.json；10:43实际平台Running原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/platform_task_readonly_continuation_1043.json；10:38同8原worker poll-gated/trace仍CE450、其余325，无目标终态。

此前停止序列注释分支不恢复/重试/转交/绕过；独立CUDA算子暂停。原容器/作业/worker/监听不重启，不重复平台提交；不读取真实shard/抽真实小batch/执行审核selection/开始真实训练。全部原data/scope/source/selection/learning/profile/GPU/full-resume门禁保留。
