# 正式训练的外部审核资源绑定缺口

## 2026-10-02T10:44:51.097287+00:00 当前收尾与剩余外部资源阻断

新增八进程通信/清理的两环境首次4/4及独立raw复核已闭合：root88与独立103原件哈希重新核实，9冻结source/26旧文件不变，temp及checkpoint均0，不再重复已通过控制。实际单模型八GPU训练/原Trainer构造/native词表保存恢复仍未验证；生产入口未接通，真实语料训练0。

正式接入的真实阻断是缺当前V7合法完整scope/input inventory/registry与报告绑定格式，不仅是两份审核尚未passed。调用链会重复原固定资源gate、请求重建丢附加字段，selection还要求旧payload完整重算相等；自行选短path/SHA清单会缩减原保护范围。不能补假清单/证书、手改ready、monkeypatch旧ROOT或加入永拒placeholder充数。证书未通过本身不禁止通用开发，但没有权威完整资源契约就无法正确定稿这部分formal gate。

外部必要材料：当前scope_version及完整逻辑输入清单；批准的外部根、逻辑ID→绝对路径映射及文件/registry/evidence哈希格式；绑定当前manifest和同一输入全集的双审核字段语义；各候选/selection/执行/续训共同使用的同一版本资源契约。未来真实执行仍要求training_ready is True和两份有效当前审核，不降低任何原门禁。

当前可继续核实同8原任务的实际500终态并各自一次原verify_diagnostic(500,require_learning=True)；它们仍Running时不得因观察超时重启。后续单模型GPU/正式接入须满足当前来源及资源条件，不能用新numeric/GPU测试填补外部资源缺口。待权威契约到位，最小后续六类source为参数化资源gate、selection外envelope、正式request/preflight/build、execution/fullresume、cold entry及immutable package；此处仅明确必要边界，没有创建或执行六类实现。

只读决策 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/formal_resource_remaining_implementation_decision.json（d2631a1f…ba5c），root物理核5原source SHA及10实际调用位置；CPU阶段验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_CPU_transport_root_acceptance.json；10:43实际平台Running原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/platform_task_readonly_continuation_1043.json；10:38同8原worker poll-gated/trace仍CE450、其余325，无目标终态。

此前停止序列注释分支不恢复/重试/转交/绕过；独立CUDA算子暂停。原容器/作业/worker/监听不重启，不重复平台提交；不读取真实shard/抽真实小batch/执行审核selection/开始真实训练。全部原data/scope/source/selection/learning/profile/GPU/full-resume门禁保留。

## 以下为较早源码定位记录

更新：2026-10-02T07:37:29.854839+00:00。当前为源码与路径metadata核实，未执行审核、selection或真实训练。原Goal身份与累计用量保留。

此前“有效V7审核和selection加一个薄controller即可正式训练”的结论不完整。PilotRequest接受外部data_dir/selection_path；冻结scope gate却调用包内input_specs，后者用自身源码所在runtime作ROOT。五个runtime的两类固定评测目录均缺失（10个绝对路径见原件）；原package同时要求resource_aliases为空、源文件集完整且无额外文件/符号链接。因此补合法证书本身不能消除这个软件路径阻断。

实际源码：[数据请求](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/runtime_packages/r21_seed_runtime_v2/scripts/architecture_prototypes/certified_pilot.py:43)，[单进程合同](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/runtime_packages/r21_seed_runtime_v2/scripts/architecture_prototypes/certified_pilot.py:141)，[scope验证](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/runtime_packages/r21_seed_runtime_v2/scripts/data/tcr_scope_gate.py:25)，[固定资源根](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/runtime_packages/r21_seed_runtime_v2/scripts/data/audit_tcr_exposure.py:25)，[缺输入拒绝](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/runtime_packages/r21_seed_runtime_v2/scripts/data/audit_tcr_exposure.py:101)，[原准入检查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/runtime_packages/r21_seed_runtime_v2/examples/llada/immune_pretrain_v3.py:39)。原assert_ready检查双证书及数据内容SHA，但未显式检查training_ready；用户要求的training_ready is True必须由正式外层明确保留，不能利用原遗漏准入。

通用工程的最小下一设计是新版本显式资源合同：固定允许的外部根、完整逻辑输入集合及绝对路径/SHA、审核范围及版本、已有registry/evidence绑定、合同和新软件来源SHA。资源解析要有明确参数传递；不修改旧运行包、补链接/额外文件、monkeypatch审核模块或临时改ROOT。新实现保留全部原data-shard/manifest、双证书passed/no pending、禁重叠和family split为0、全部support/query输入与evidence实际SHA检查；新增下游范围必须被合法当前审核覆盖。合同绑定不是审核结果，也不产生证书或修改ready。

现在只做接口/来源设计；真正审核输入、selection重建及真实训练仍等待有效当前V7审核，不调用已停注释逻辑。新软件对profile/学习/full/resume来源的影响须明确评审，不能迁移旧结果或无依据安排旧GPU验收重跑。七项缺失的synthetic学习仍只为既定原来源前置，模型/validator不变；当前CE500继续。

现PilotRequest要求WORLD_SIZE=1，支持八个独立单卡候选/seed的初筛，不是单模型八卡DDP。最终选定模型的多卡训练是单独的实现与实际证据要求，不能用候选并行代替。两阶段都未真实启动。

[实际源码SHA与10个缺失路径](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/formal_resource_binding_readonly_gap.json)。
