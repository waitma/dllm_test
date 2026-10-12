# 原 pool 到正式八 slot 组的独立接线准备

## 2026-10-03T03:41:39.451832+00:00 根端正式审阅与独立冻结完成，仅准备状态

根端已读取实际源码、三项控制完整stdout/receipt和归档SHA，前后原件一致，未重复执行。审阅 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/entry_v2_root_review.json（SHAae9a7f7a2cad1a4c3ffd92834b1947453dfc31182f76f62c615ac48486d327ff）；两源冻结清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/pool_launcher_source_v2.json（SHAe92de529db8b77f5c0e083ff115186c1419fbfc6c9a53ba61b386d5fde1bb9df）。旧formal wrapper、运输、v3/611 source与parent_gate_v3不改，当前public仍large-batch v2。正式selection/plan/catalog与八票据未创建/入队，真实联合更新和恢复未验收。首次冷控制证明变动接线，不能替代实际world8。新组真实进度 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LARGE_BATCH_RUNTIME_WATCH_20261003.md，预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_RUNTIME_RESOURCE_BUDGET_20261003.md。

当前2026-10-03T02:36:47Z：实际父pool七字段/完整argv/cwd已由原p04启动时获得，metadata原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/logs/1790981433321562451_p04_v7_capacity_febb28e3.pool_parent_runtime_observation.json（SHA96ba2a4bb89e8b4c571a98e135f4f3d9af2d76f2d30bcbb1e56217dc36d1e269），未来wrapper仍须现场重核同identity。独立容量v2九格已成功且根端核实，原8项完整epoch收尾；正式v3来源门禁通过及冻结wrapper保持，formal计划/票据/实际八rank更新与恢复仍未执行。先接续统一batch完整消融；此capture不视作资源预留或正式准入，下方缺capture等表述保留当时历史。

2026-10-03T01:27:30Z 最新driver为独立v3（611文件、原helper新来源一次passed），native恢复返回点的状态观测已完成首次CPU控制与根端审阅/冻结；旧wrapper/dispatcher三文件冻结及7项metadata控制不改，动态driver引用未来使用v3。尚无formal plan/catalog/登记或真实world8恢复；唯一容量任务父身份capture仍待真实领取。详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_RESUME_OBSERVATION_20261003.md。下方各日期状态保留历史，不用旧“源码待准备”覆盖当前。

2026-10-03T00:09:34.673530+00:00更新：前述正式launcher三文件/7项metadata修正冻结保持且仍未发布formal训练。真实pool父身份读取已接入原唯一pending容量job，但其尚未领取，实际capture仍缺失；不能提前把CPU self/parent身份填入正式计划。该hook只读原容器/proc、保留实际相对script argv/cwd，日志输出与profile run隔离；原job/dispatcher/registry不改，无重启/重复入队。发布证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/pool_parent_metadata_deployment_v1/publication.json。

当前状态（2026-10-02T23:49:12.122059+00:00）：新分段/恢复/兼容调用修正已完成并经根端核实，三文件独立冻结；未发布正式入口、构造真实计划或运行共同八卡。下方最初8项/4项检查是当时版本记录，新版证据以本节及末尾修正记录为准。

根端发现新5项fixture与实现共同使用了compact JSON摘要，但实际冻结driver.digest使用带默认空格的JSON序列化。现已由调用方传入真实冻结driver['digest']，不改plan/intent原compact摘要；checkpoint正向fixture也使用真实driver摘要，并加错误compact摘要拒绝。另实际pool YAML以 `python -u scripts/immune_research_pool.py work ...` 启动，脚本argv为相对路径；wrapper现场读取实际父/proc/cwd解析script，保持原始pool_argv精确比较，不推测或改写调用参数。

仅因这些明确源码修正执行新版7项人工metadata控制一次，0.293秒，均通过；真实冻结driver parser/digest参与，模型/transport main/二进制checkpoint/进程/网络/GPU均未执行，临时JSON已清理。第一次5项原件保留，但不足以证明实际driver摘要兼容。最新原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_segment_digest_fix_20261002/receipt.json，SHA256 31e7d63b4ecf51b5666f1da5060389b33c105d617a8fb6b5b74d3b8a987a9833。

实际新版wrapper SHA256 2d4da598e1bceebed2efa2626d79408367d679e3bcdf7cf9966c81b4a1eae40c；dispatcher 21f2ffe71d23950207eb4c067b0cb26cf5dbb875a13e2ba15bdec3a7d3f5ce59。独立三文件冻结 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/pool_launcher_source_v1.json，SHA256 828a0010868da8bef2172b3b2940b18ad1c8b66a89a94db364789cd2aad9f0bb；driver/transport v2字节和610文件清单保持。

真实pool完整身份/argv仍缺失：V3实际旧controller原件只能说明父PID708/start_ticks，不能补齐当前pool七字段或cmdline。正在准备让唯一尚pending的capacity任务启动时做std-only父身份读取，原件将放 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/logs/1790981433321562451_p04_v7_capacity_febb28e3.pool_parent_runtime_observation.json。不能提前写入capacity run目录：冻结profiler限制初始目录只含原日志，否则会拒绝profile。该读取尚未发布/执行，不改568 profiler源或新增任务/监听，读出的旧身份未来也须wrapper现场重验证。

2026-10-02：新增 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/formal_pool_job.py`，源码 SHA256 `7e3cce617a86c05bb5feae726d8eaec2201bacb38295d64b38eda6e5ca396136`。本项只准备新源码和人工 CPU 控制；没有修改公共入口、registry、现有 pool、冻结 source_snapshot_v2 或 transport 字节，没有创建真实 plan/catalog/reservation，未发布、入队、监听或运行 GPU。现有 pilot/队列继续由根端管理，模型/seed 选择不由 wrapper 推断。

## 真实进程关系与入口

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/immune_research_pool.py` 的八个 worker 是同一 pool 进程内的 ThreadPoolExecutor 线程，不是八个独立 pool PID。每个 slot 的 `Popen(start_new_session=True)` 才创建独立 job 进程。未来早期 dispatcher 必须通过 `exec` 调用冻结 wrapper，保持这个 job/controller PID 和进程组不变；wrapper 再作为 frozen transport 的真实父进程监督 rank child。

CLI 保持 `--plan /absolute/plan.json --plan-sha FILE_SHA256 --job-id ACTUAL_POOL_JOB_ID`。根端独立 dispatcher 可在 ML 导入前验证 pool argv/ticket/catalog，再执行 `python -I -B -u FROZEN_WRAPPER ...`。本项未接入任何公共 dispatcher。

八个票据可使用 `p03_v7_formal_member_0..7` 等明确别名，但数字只是票据身份。pool 随机领取票据，rank 必须来自实际 `CUDA_VISIBLE_DEVICES=0..7`。wrapper 现场读取 coordinator process_identity、父 pool 的 `/proc` 身份与完整 cmdline，核对线程 worker 状态文件当前 pid/job、运行票据内容、实际 GPU UUID/PCI/NUMA。transport 的历史字段 `intent.members.worker_process` 在本接线中明确表示八个独立 **slot job/controller PID**；共同 pool 父身份另存在 plan/claim 中。这样 frozen `member_parent` 的 `current == worker_process` 分支成立，不冒充八个 pool 进程。

## 计划字段与统一发布

`prepare(spec)` 只构造并验证内存对象。计划 schema 为 `v7.formal.pool_group_plan.v1`，字段如下：

- `pool_dir`、`meeting_dir`、`group_id`：绝对目录与独立组标识；meeting 的父目录由未来发布方准备。
- `pool_process`：实际 coordinator process_identity 的全部字段；`pool_argv`：真实 `/proc/<pool_pid>/cmdline` 参数列表。不能在开发机伪造容器 PID/boot。
- `pool_source`、`wrapper`、`transport`：各为 `{path, sha256}`，由调用时实际文件重新核对。wrapper 自身路径必须匹配。
- `work`：原 frozen transport 的完整 work 对象，包含已选模型证据、真实 final-parent 审核结果、两根 source manifest、冻结 driver、完整 driver argv/预算。共享 `work.output_dir` 必须位于 pool 的 per-ticket runs 目录之外，防止 slot COMPLETE 或 pool checkpoint 清理干扰共享训练输出。
- `tickets`：`{actual_job_id: exact_original_pool_job_spec}`，严格八项。别名与 rank 没有预设映射。
- `join_seconds<=300`、`startup_seconds<=180`、`duration_seconds`、`cleanup_margin_seconds>=120`、`segment_steps`：汇合、组启动、资源及声明训练段预算。

每个 controller 在 meeting 目录写带 plan digest 的真实 claim，绑定 job/worker/mask/controller PID/start ticks/shared pool/host/boot/GPU。pool 在 Popen 后才补写 PID heartbeat，wrapper 仅允许这段限时初始竞态，不接受错误 job/pid。每轮在同一把有界 flock 下检查所有已有 claim；发布前必须满足：八 expected job 全在 running、pending 为空、没有无关 running、八个当前 worker pid/job 完全匹配、八 controller 与共同 pool 均仍 live、claim/worker heartbeat 在 30 秒内。

可能较慢的源码/审核证据核对在 flock 外执行，然后重新锁定并复核全部身份和队列快照。只有复核通过才写完整 `intent.json` 与 `reservation.json`，最后独占写入 `PUBLISHED.json` 作为唯一提交点。所有 controller 只在该提交点存在且文件 SHA 匹配后调用 frozen member_parent。若中途失败留下未提交文件，不覆盖重试；ABORT 阻止同一 meeting 继续启动。

准确语义是：**八 slot 已被本组 controller 持有后，原子发布 intent/reservation。** 这没有改造 pool 的全局队列事务。后续新 pending 不会抢占已经持有的 slots，运行期不会仅因出现新 pending 而中止训练。公共队列/catalog/registry 发布仍由根端留待当前 pilot/队列完成且模型已选后处理。

## 保守期限与退出

wrapper 从真实 pool argv 读取 `--max-seconds`、`--job-seconds` 和 expected-gpus，使用 pool/controller 实际 `/proc` start_ticks、SC_CLK_TCK 与 CLOCK_BOOTTIME 计算最早硬截止；从进程起点计时比原代码稍后建立的截止更保守，也不会因 suspend 延长资源。组截止取声明 duration 与最早硬截止减清理余量的较小值，必须仍容纳 startup。reservation 发布时重新生成 monotonic 时间，保持原 30 秒新鲜度。

部分票据、无关旧作业、异常或 stale claim 不能无限等待。新 wrapper 使用有界 join/lock，持久化 plan 绑定的 ABORT；发布后的错误同时交由 frozen transport 的 abort 传播。训练监督仍使用原 transport 的 SIGTERM、真实 wait、TERM/KILL、进程组/后代清理与 finally，不只依赖 pool 杀 controller 进程组。wrapper 不增加训练期轮询层。

## 每票据完成证据

controller 必须等共享 coordinator 真正 succeeded、exits 恰好含 0..7 的八条成功真实 wait/cleanup 证据、八份 parent supervision receipts 全部存在。原 coordinator 可能先复制七条 exits，随后在第二次短锁看到 terminal=succeeded；这是一种合法过渡，`completion_ready` 会继续等待，不能直接报失败或提前完成。

随后校验声明 `segment_steps` 对应的真实共享 `SEGMENT_<step>.json` 或最终 `COMPLETE.json`、run_provenance 的模型/两根代码/数据/seed/global budget/horizon、完整 checkpoint sidecar、模型/optimizer/scheduler/八 RNG 等文件 inventory 与 SHA，以及八 rank 实际参数摘要证据。每个 controller 仅在全部成立后，独占写自己的 `pool/runs/<job_id>/COMPLETE.json`，类型明确为 `formal_pool_segment_completion`。

pool ticket 完成不自动等于正式全训练完成。只有 step 等于原 scheduler horizon 且共享 driver 的最终 COMPLETE 成立，`formal_training_complete` 才为 true；中间段为 false。pool 原来的逐 ticket prune 路径不包含共享训练 checkpoint。

## 本次检查与剩余工作

新 wrapper 人工检查 8/8 通过：共享 pool 父身份和八 controller、反序 ticket 到实际 mask/rank、错 pid/job/stale/shared-pool-PID 拒绝、部分/无关队列拒绝、保守资源截止、临时目录原子提交、持久化 abort、七 exit 合法过渡，以及缺 optimizer 时拒绝完成/完整人工段仍为非最终完成。测试里的模型/checkpoint 文件只是临时文本 metadata，不是训练产物，不证明 native optimizer 或 GPU 恢复。socket.socket 在控制期间被替换为调用即失败；未创建子进程/GPU/listener，没有访问真实 pool 状态或数据。

真实 stdout：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_pool_job_evidence_20261002/self_test.stdout.log`。执行范围与 source SHA receipt：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_pool_job_evidence_20261002/receipt.json`。

仍未进行真实 slot 汇合、公共 registry/dispatcher/catalog 发布或入队。实际容器 pool 身份/剩余预算、八票据真实领取与统一发布、父子监督/FD/NCCL、共同 optimizer 更新和保存恢复仍需未来有界整合验收。本次没有重跑旧 Gloo/500 suite，也不恢复任何停线注释或独立 CUDA 验收分支。任务保持已有 binding 正例的 masked reconstruction 范围，不增加负例判别或特定生物功能优化。

## 新 dispatcher 的独立 CPU 边界检查 — 2026-10-02 22:29 UTC

主会话新增但尚未发布的入口为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_formal_dispatch.py，修正版 SHA256 `19f92a73b7c5cd8c2e9448255040d207662bb79317bc26ac352882aeb3bcbea8`。只读审查发现两处必要缺口：合法 job 路径自身或父目录为 symlink 时可能把 slot COMPLETE 写出预期目录；catalog 当前票据正确但其余同组票据可指向别的 plan。主会话已在新入口补充 canonical/nonsymlink 路径检查和同组八票据同一 plan 引用检查，没有修改旧 dispatcher、公共训练入口或冻结 v2。

新增检查源 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_pool_formal_dispatch.py，SHA256 `b09760669aaf4b8927a52a9c69f5bb475f6bda4e571a38c9a250176e9329bd88`。22:29:55 UTC 在 /vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python 首轮四项控制4/4通过，检查段0.077517秒，命令外层限180秒：

- 八份人工票据/catalog/plan绑定及精确 `python -I -B -u wrapper --plan --plan-sha --job-id` 命令。mock拦截 os.execv，票据后缀与声明mask反序仍可委派，命令不携带猜测的rank。此项证明调用约定，不声称实际exec/PID交接已执行。
- 错当前job envelope、少一票据、拆组catalog引用、真实临时plan/wrapper字节SHA变化、缺catalog均拒绝。
- 合法job输出symlink、runs父目录symlink、plan symlink、越界引用和错误data路径均拒绝；没有读取任何真实data目录。
- 旧p02 argv原样交给mock previous main，实际旧源码SHA保持 `0946d193e7f6e5bb080cc79447a72e09095c543df64fa44b36b9f1e98eb923d6`；修改的临时旧源码副本在加载前被拒绝。

所有fixture明确为 synthetic_fakecontrol，仅符合dispatcher所需的人工envelope，不是完整正式计划、审核或GPU准入件。actual wrapper/previous main、模型、socket、subprocess和GPU均未执行；catalog/plan只在自有临时目录生成并已全部删除。测试前后新dispatcher、wrapper和旧p02源码SHA均相同，没有真实catalog/registry/入口发布或任务操作。

## 新增步数、恢复起点与兼容分派修正 — 2026-10-02

仅更新尚未发布的 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/formal_pool_job.py 和 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_formal_dispatch.py；保持plan schema `v7.formal.pool_group_plan.v1`及字段集合不变。原pool、公共入口、真实plan/catalog/queue、冻结transport和driver v2均未改动。

票据job.steps现在明确为本段新增updates，仍遵守原pool的1..5000限制；plan.segment_steps始终为冻结driver的累计stop_after（未指定时为max_steps）。八票据的steps/data_dir/resume必须完全一致，且resume严格等于冻结driver parse_args解析出的实际参数。无resume时先前step为0；有resume时只读共享output内既有checkpoint的formal_checkpoint.json，以及该sidecar.files中SHA绑定的trainer_state.json/run_provenance.json。核对global_step、sidecar.step、binding摘要、world8/rank0/八rank acknowledgments、原模型/数据/source manifest/固定日程，并复核读取期间元数据SHA不变；绝不从checkpoint目录名猜step、不加载二进制checkpoint或补造恢复证据。要求 `累计target - 既有prior == ticket.steps`、`prior < target <= work.max_steps`。例如先前6000、target6500、ticket.steps500可通过原pool，而无需放宽原pool5000上限。

新wrapper公开只读validate_segment(plan, parsed_driver_args)，返回prior/target/新增updates及实际读取的checkpoint JSON SHA。dispatcher在返回exec命令前，通过plan引用SHA加载wrapper和冻结driver，复用prepare、parse_args和validate_segment；wrapper汇合后、统一发布前再次执行相同核对。该早期检查只验证JSON协议，不代替冻结driver后续完整模型/optimizer/scheduler/八rank RNG二进制恢复与全部checkpoint inventory验证。

formal fallback改为已冻结 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/launcher_source_v1/pool_large_batch_ablation_dispatch.py，固定SHA `dc3408ff7a8e74b7944b5ceb17ca18bdaf0baf9a226d270c9cb12fe99058aa0f`。未知alias保留原argv，未来兼容链为formal→large-batch ablation→capacity→p02；本次没有把公共入口接到formal，也没有发布八formal alias。

新增有限控件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_formal_segment_dispatch.py 在pllm/空CUDA mask下仅执行一次，5/5通过，耗时0.249秒，外层限时180秒。实际调用冻结v2 driver的标准库parser、新wrapper metadata helper及dispatcher.prepare，覆盖累计6500/先前6000/新增500的八票据命令、无resume prior0、八票据steps/data/resume不一致、实际driver resume不匹配、错误先前step/差值、trainer JSON SHA漂移、sidecar step不一致、没有剩余更新、固定日程漂移及SHA绑定fallback的capacity/large-batch/p02原argv委派。execv与旧main被mock，socket/subprocess被禁止；全部synthetic_fakecontrol临时计划、catalog和checkpoint JSON已清理，未创建或读取任何native checkpoint二进制。没有模型/GPU/实际训练数据/真实计划或队列操作，没有重跑旧检查。

原始stdout/stderr：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_segment_controls_20261002/stdout.log；命令、解释器、原件和SHA归档：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_segment_controls_20261002/receipt.json。修正版wrapper SHA256 `5e5235b59331cf09c4bf2e59dc7b793010bd925e133bfa784d83100f1d9fbc49`；dispatcher SHA256 `b78dddd6e5d52462d97fe5cc992dbd5a553c33a25c2898c799896d900ffc6ec3`；专用控件SHA256 `e407109596b4738549f4b6e64df82a80278366e1bbffcd2a234137573fb8ea90`。冻结v2 driver仍为 `a751eea2893aea463ab23798b0f685edbf1333a3da8e7a287aa4a7a4f453c94d`，transport仍为 `17bf275958f2cdcbd2a9432dfd2fd0f87ad6df61203b99c558efbae96f7d8346`。真实pool完整身份/argv、正式选择/计划及world8实际保存恢复仍待根端后续处理；本次软件检查不构成GPU验收或发布。

原始日志 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_dispatch_controls_20261002/stdout.log；实际命令/解释器/源码SHA、八条mock命令、拒绝范围与空临时目录证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_dispatch_controls_20261002/result.json，SHA256 `56598a4fee7a94cf5b8cd515b9bb24c820778aea66c884110ccd81cfe190badb`。该结果不替代实际八卡汇合、训练或保存恢复验收。
