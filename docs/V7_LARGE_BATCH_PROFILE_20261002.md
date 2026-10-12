# V7 大 microbatch 模型容量与吞吐 probe

最新2026-10-03T02:36:47Z：本页原p04九格/72更新任务首格非OOM超时永久保留；独立v2九格/36更新任务已实际成功、根端原件及40实际加载源核对通过，micro32三代表均有15%余量。新协议与下一统一batch说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_CAPACITY_TIMEOUT_REVISION_20261003.md。不能用v2结果冒充原p04成功，不能由容量推导真实语料或formal验收。真实父pool身份已经捕获，当前读取CURRENT及PROJECT顶部，下方“仍pending/capture缺失”为各自日期历史。

当前接续（2026-10-03T00:09:34.673530+00:00）：原唯一capacity仍pending，尚无GPU measurement/summary/COMPLETE，也没有真实pool身份capture原件。00:01:12 UTC只发布了SHA-bound stdlib父身份读取hook，等该同一job领取时记录真实七字段/原始argv/实际cwd到pool/logs，随后原冻结profile继续；无新入队/重启，capacity dispatcher/registry/568源及原pending JSON保持。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/pool_parent_metadata_deployment_v1/publication.json。

observer五冷控制原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pool_parent_observation_controls_20261002/receipt.json 已根端读取并匹配各源码/原始日志SHA；实际CPU self/parent不作为容器pool身份。另三项staged早期entry控制使用mock observer/dispatcher，旧p02和原capacity argv保留、错误source SHA先拒绝；没有ML/子进程/网络/GPU。记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/pool_parent_metadata_deployment_v1/early_entry_controls.json。两文件独立冻结清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/pool_parent_source_v1.json。

唯一实际capture目的地为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/logs/1790981433321562451_p04_v7_capacity_febb28e3.pool_parent_runtime_observation.json，不能放在capacity run目录提前污染冻结profiler的初始允许文件集合。它只为后续formal parent/process核对提供metadata，不是profile或GPU成功证据。当前真实资源状态原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/pool_after_parent_hook_20261003.json：Running，原8项2225–2300/3750，1pending/8running/105finished。

当前状态（2026-10-02T22:52:04Z）：容量入口新四组标准库检查通过，已于22:50:33 UTC完成一次实际接入/登记；job1790981433321562451_p04_v7_capacity_febb28e3仍pending，等原卡自然空出。实际GPU容量测量尚未执行，没有峰值、吞吐或推荐batch。原八项继续，未重启资源。当前证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_job.json；原始接入/入队证据在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v1。文末为此唯一任务的有限只读监督记录。

2026-10-02：源码准备完成，尚未运行 GPU。入口为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/profile_v7_large_batch.py`，SHA256 `e44e10f134965299f1c288ed7d606b62eb832436230395112a3e6ddf4c542233`。没有修改原 pool、registry、已有训练或冻结源码，没有提交、入队、监听或恢复停线分支。

2026-10-02T22:39:58Z 接续更新：该脚本已被复制到新建的独立大 batch 冻结目录，原 p01/p02 冻结源不变；清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/source_snapshot_v1.json（SHA256 `2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783`），568 文件及绑定 CPU 原件均已核对。调用脚本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_large_batch_dispatch.py 仅为未部署源码，尚无入口调用检查或实际入队证据；公共入口仍使用原 p02 调用脚本。GPU profile 未执行，不能给出 micro16/32 的实测峰值、吞吐或推荐档位。

## 根端接入接口

激活 conda pllm 后，未来仅在原 A100 80GB 自然空出的单卡上，经根端指定一次 profile job 调用：

```text
python -I -B /absolute/frozen/profile_v7_large_batch.py
  --source-root /absolute/frozen/model-source
  --output-dir /absolute/new-profile
  --variants f000 f111 cv_segment_global
  --micro-batches 4 16 32
  --global-batch 32 --warmup-steps 2 --measure-steps 6
  --candidate-seconds 180 --max-seconds 1800 --seed 42
```

以上换行仅为展示，实际传 argv 列表。`parse_args(argv)` / `run(args)` 提供直接 API；普通调用使用 orchestrator，内部 `--candidate-variant` / `--candidate-micro` 仅用于它启动的独立测量子进程。output 必须新建或仅含已有 worker 日志，不覆盖历史 profile。source-root 显式选择被冻结的真实模型实现，不自动切换当前训练任务源。

默认 3 variants × 3 microbatches ×（2 warmup + 6 measured）= **72 optimizer updates**，包含建立 AdamW state 的更新。参数检查硬拒绝超过 100 个计划更新。global32 对应 micro4/16/32 的 GA8/2/1。每档子进程 wall-time 上限 180 秒，总测量预算 1800 秒；超时后仅允许有界 TERM/KILL 清理余量（最多两次 3 秒 wait），不会自动续跑或重试。

## 模型与人工输入

复用原 FactorialModel、原 forward/loss、AdamW(lr1e-4)、FP32 参数及 Adam 状态、BF16 autocast、whole-layer activation checkpointing，并保留原 pilot 的 clip_grad_norm=1.0/nonfinite 检查。配置固定 A4、d768/L8/H12/MLP3072、full_capacity256、fragment_capacity64、maxlen2048，不缩模型。硬件入口要求仅一个可见 GPU，名称 A100、总显存符合 80GB 档。

仅创建整数 tensor，不读取任何真实记录、序列或注释。每条 2048 个位置全部 attended：8 个前缀位置、1528 个可见 context 位置、两个各 256 的目标区段。目标区段各含一个真实监督位置意义上的 END 整数 ID，context/prefix labels=-100，固定 100% 目标 masking，形成保守的满长 dense-attention 形状。它不是实际 corpus 的长度/链数/任务分布，也不声称涵盖所有实际数据内存峰值。

没有 collator/CPU preprocessing/H2D 时间进入 measured timing：输入已预分配到 GPU，测量 forward/backward、梯度裁剪、AdamW update、同步的训练循环。吞吐同时报告 attended input tokens/sec、target tokens/sec 和 records/sec；不同 micro 的有效每 update 输入预算相同。constant LR 仅服务有限容量/吞吐测量，不复现 pilot cosine 日程或语料学习曲线。

## 输出、失败与推荐

每档独立子进程，复用现有 profile 的进程隔离方式，避免前档 OOM/allocator 缓存影响下一档。`<variant>_micro<N>/measurement.json` 记录 warmup 和总 peak allocated/reserved、显存总量/开始空闲量、optimizer state tensor bytes、完成 update 数、6 个有限标量 loss、测量耗时/吞吐、模型参数量及实际执行源码 SHA。`candidate.log` 留存诊断；不输出生物序列、预测 logits 或模型权重，不保存 checkpoint。

CUDA OOM 是该档容量结果，记录 `status=oom`、发生 phase、已完成 updates 和可读取峰值，后续不同档位继续一次。非 OOM 异常、nonfinite、source drift 或 timeout 使此次 sweep 失败，保留现场，不重试。timeout 无法取得的指标明确为 null，不补造峰值。父进程处理 SIGTERM/SIGINT，finally 回收独立测量子进程；实际 GPU 清理行为尚待未来这次 profile 运行观测。

`summary.json` 保留所有测量和机器推荐：每 variant 在 allocated/reserved 均不超过总显存 85% 的通过档中选 synthetic 吞吐最快者，另报告全部代表 variant 都通过且保留 15% 余量的最大共同 micro，以及实际 OOM 档位。没有测量则推荐 null；没有任意档位通过也不会伪造推荐。`COMPLETE.json` 仅表示有限 synthetic profile 完成，明确 training_completed=false、actual_corpus_test=false、checkpoint_created=false。

推荐只支持后续选择容量候选；它不是实际语料学习效果、生物学效果或自动修改现有任务 batch 的许可。micro4 是同形状吞吐 control，micro16/32 是更大 batch 候选；最终是否修改未来任务仍由根端结合真实测量和运行预算决定。

## 本次 CPU 证据与未完成项

本次仅在 CUDA_VISIBLE_DEVICES='' 下执行参数/整数 shape 检查：默认72更新、超过100拒绝、micro4/16/32 的 [B,2048] 全字段形状、512目标/2 END/context loss mask、global32算术、无数据推荐为null，以及人工 OOM 排除共同micro32推荐。没有构造 FactorialModel、forward/backward、optimizer update 或 GPU 操作。首次检查日志保留为历史，最终源码补齐与原 pilot 一致的梯度裁剪后进行了同一有限检查。

最终 stdout：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_profile_source_20261002/cpu_checks_final.stdout.log`。receipt：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_profile_source_20261002/receipt.json`。本轮不重跑旧测试或独立 CUDA 算子 benchmark。

当前剩余：完整九档显存/吞吐/OOM测量。唯一调用已于2026-10-03 01:42:46 UTC实际失败：首档f000/micro4触及180秒候选上限，其余八档未启动，尚无合格容量或推荐batch；详见文末终态证据。本轮没有自动重试或扩大预算。

## Dispatcher 独立入口检查 — 2026-10-02 22:48 UTC

审查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_large_batch_dispatch.py（SHA256 `f259fb0f0946fb7434c1f78b06f64e5d186ece7b781fd29dac7f3f0fe0e5d866`）与实际冻结清单后，未发现阻断该单项 synthetic profile 接入的必要源码修正。此结论只覆盖 dispatcher 软件边界，不表示已经发布、入队或取得 GPU 容量结果。

按要求只调用一次绑定真实清单的 `prepare(argv)`，使用符合格式且不存在的人工 job ID `9999999999999999999_p04_v7_capacity_c0decafe`。实际568份冻结源码、2份CPU证据SHA全部匹配；将返回的命令交给冻结 profile 脚本的真实标准库 `parse_args`，确认 f000/f111/cv_segment_global × micro4/16/32 × (warmup2+measure6)=72，global32，candidate180秒、total1800秒、seed42，CONFIG长度声明2048。没有调用 main/run、导入Torch或创建该 pool output 目录；没有执行旧 profile self-test。

一次有限标准库执行另外检查：只接受 profile-synthetic/steps72，拒绝train、71/73步、data/resume/自定义micro参数；空mask、多mask及8均拒绝；正确单mask通过 mock os.execv，命令保留 `python -I -B -u`，没有启动真实process。临时人工文件实际修改验证 source/evidence/manifest SHA拒绝，以及输出目录/冻结源 symlink拒绝。未知p02 alias保持argv原样交给mock previous main，并在调用前检查实际旧dispatcher SHA；旧main从未执行。

新增源码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_pool_large_batch_dispatch.py（SHA256 `8a9b0bb66b8fa915ddbd75edbcab6aa5cff797d9676873fa59cbe6c5d02c2097`）。22:48:11 UTC 在 /vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python 首轮四组控制通过，实际段0.574728秒，外层限180秒，CUDA_VISIBLE_DEVICES为空。测试仅在自有目录使用明确 synthetic_fakecontrol 文件，临时目录已全部清理。实际新dispatcher/冻结清单/旧p02 dispatcher检查前后SHA保持不变。

原始 stdout /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_dispatch_controls_20261002/stdout.log；实际命令、代码SHA、逐项范围和空临时目录证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_dispatch_controls_20261002/result.json（SHA256 `e8f6af7c7e1886be10c27203e3e032f1afacbaf9654520f281dd78d117b5d86c`）。本检查未修改公共入口/registry/queue或任何冻结文件，未运行模型、GPU、真实数据、注释或独立CUDA算子。

## 唯一容量任务的有限只读监督

主会话已于2026-10-02 22:50:33 UTC完成一次实际发布/登记，job=`1790981433321562451_p04_v7_capacity_febb28e3`。真实账本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_job.json；发布与入队原件位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v1。这里只跟踪同一个任务，不重复profile、不重启/改任务，也不启动OS监听或daemon；观察最迟截止2026-10-03 08:50:33 UTC，观察到期不等于任务失败。

22:52:04 UTC实际首次快照：pool pending=1/running=8/results=105，唯一capacity仍在pending，run目录尚不存在。八个原任务的worker均为running，poll-gated heartbeat距观察时刻0.25–1.02秒，表明原pool仍在实际监督这些进程。平台已有最新查询为22:45:44 UTC、原任务t-20261001033327-fmnql状态Running；该平台时间与本次worker观察时间分开记录，不把旧平台cache称为新查询。当前没有capacity领取卡号/PID、测量、峰值或推荐值。

原始状态与输入SHA：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_readonly_watch/observation_20261002T225204Z.json。后续仅在实际领取、候选结果、错误或终态变化时追加证据。OOM按容量结果记录，其他异常如实报告，不自动重试或绕过。

## 唯一 capacity 任务父 pool 身份读取源码（2026-10-02）

新增标准库入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_parent_runtime_observation.py，API为observe(argv=None)->bool。当前仅源码和冷控件完成，未修改公共入口或部署。未来由根端在唯一capacity任务启动时调用，不增加任务、监听或子进程。仅匹配p04_v7_capacity及job1790981433321562451_p04_v7_capacity_febb28e3；其他argv立即False，无文件或环境读取/写入。匹配后仅允许原variant/mode=profile-synthetic/steps72/profile-dir/output-dir五选项各一次；路径须canonical、无symlink，单CUDA mask为0..7。helper只读取该mask，不读取环境快照或credentials，不查询GPU、不读取数据。

最初拟将原件写入capacity run_dir，但预览发现冻结profile_v7_large_batch.py在run入口仅允许train.log/worker.log/stdout.log/stderr.log四种既有文件。预先写其他metadata会使原profile拒绝启动。因此按根端确认，唯一输出改为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/logs/1790981433321562451_p04_v7_capacity_febb28e3.pool_parent_runtime_observation.json；run_dir不创建、不改变。先写并fsync独占临时文件，再以hardlink独占发布原件，绝不覆盖既有目的文件，最后清理临时文件；payload仍绑定原run_dir/job。

复用SHA绑定V4 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/eight_rank_coordinator_v2.py 的process_identity；SHA保持dbaf3e8536370309a42fd2c33d23b020224b62d981451b2038387a21009e4001。实际读取当前controller与os.getppid父pool的七字段身份、父/proc/cmdline及/proc/cwd，前后全字段、argv和cwd须一致；核对controller父PID/start_ticks/host/boot。父argv必须是Python调用原immune_research_pool.py work，仅允许pool-dir、expected-gpus、idle-seconds、max-seconds、job-seconds及standby-load；8GPU与同pool为必需语义，不接受无关进程选项。实际YAML脚本token为相对的scripts/immune_research_pool.py，故用现场读取的父cwd解析到唯一原source，保留原始pool_argv，另存pool_cwd与resolved_pool_source；不猜历史PID708或cwd、不改写argv。

payload记录真实controller/pool_process、完整已校验pool_argv、cwd、源路径/SHA、observed_at、metadata_only=true和训练/GPU/数据/credentials未执行声明。source_sha256仅表示观测时磁盘源字节，不冒充父进程已加载模块证明；该原件不是资源reservation或训练准入证书。正式wrapper未来仍须现场验证父进程身份与argv，不能把历史观察当作当前live证明。

新增控件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_pool_parent_runtime_observation.py 在pllm/空CUDA mask下仅执行一次，5/5通过，耗时0.051秒，外层180秒。覆盖noncapacity严格零读取/写入/环境访问、人工完整身份和原始相对script argv捕获、身份变化/父关联/cwd/选项/路径/symlink/重复选项拒绝、独占不覆盖、run_dir不存在或既有日志字节保持不变；并通过实际冻结profiler的AST核对其四个允许初始文件名。仅一次真实V4 process_identity接口核查当前本地CPU控件及父进程，范围明确为本地开发机进程而非GPU容器pool。exec/socket/subprocess被禁止；全部synthetic_fakecontrol临时原件已清理，无GPU、模型、真实数据、任务或队列操作。

原始stdout/stderr：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pool_parent_observation_controls_20261002/stdout.log。命令、解释器、结果和SHA归档：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pool_parent_observation_controls_20261002/receipt.json。helper SHA256为8d13c32c6c8e0b6ee65df4573a14e784947e98602e50c7ef10cf2459393e79c8。原capacity dispatcher及568冻结源未改；公共接线、helper冻结和实际容器原件留给根端后续处理，本次尚无真实pool身份观察原件。

## 实际自然领取 — 2026-10-03 01:40 UTC

同一 job `1790981433321562451_p04_v7_capacity_febb28e3` 于01:39:42 UTC被原worker自然领取，设备0、外层PID733814。01:40:29 UTC只读观察时该worker的poll-gated heartbeat距读取1.19秒，pool为pending0/running8/results106。实际summary为running、planned_optimizer_updates=72、global_batch=32、原d768/L8/H12/max_length2048配置；此时measurement数量0且无COMPLETE或终态，尚不能推荐micro batch或声称容量通过。

领取原件：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_readonly_watch/observation_20261003T014029Z.json。实际运行目录：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790981433321562451_p04_v7_capacity_febb28e3。

主会话已于00:01:12 UTC发布父身份读取hook；该唯一任务启动时已产生真实metadata原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/logs/1790981433321562451_p04_v7_capacity_febb28e3.pool_parent_runtime_observation.json（SHA256 `96ba2a4bb89e8b4c571a98e135f4f3d9af2d76f2d30bcbb1e56217dc36d1e269`）。其observed_at为01:39:42.156156 UTC，controller PID733814、父pool PID708、同host/boot关联、单mask0及完整原pool argv均留在原件；这仅是当时的进程身份与命令metadata，不是GPU profile结果或正式训练准入。没有污染profile运行目录、修改冻结测量源或新增第二次profile。

## 实际失败终态 — 2026-10-03 01:42:46 UTC

同一任务的pool结果明确为failed、returncode=1，实际01:39:42.052687开始、01:42:46.178768结束（184.126秒）。summary为failed，保留`subprocess.TimeoutExpired`的180秒候选上限错误；summary总耗时182.820秒，总预算1800秒、计划72次更新和100次硬上限均未改变。没有COMPLETE。此为实际任务失败，不是观察窗口到期，也不是OOM。

| 条件 | micro4 | micro16 | micro32 |
| --- | --- | --- | --- |
| f000 | 180秒超时；完成warmup2+measure2，共4次更新 | 未启动 | 未启动 |
| f111 | 未启动 | 未启动 | 未启动 |
| cv_segment_global | 未启动 | 未启动 | 未启动 |

唯一子测量报告GPU为NVIDIA A100-SXM4-80GB、total=79.151001GiB、75,601,176参数。其不完整执行期间原始peak allocated=6.280078GiB、reserved=7.330078GiB；这些是失败首档的部分观测，不是通过档位峰值。父summary将超时档的合格峰值保留为null。六次计划测量只完成两次，没有`measured_seconds`、吞吐或推荐；micro16/32和其余模型均无数据，不能从这份记录外推。

`candidate.log`将stdout/stderr合并，末写时间01:42:45.023109 UTC；外层日志末写时间01:42:45.933376 UTC。候选记录elapsed=176.326543秒，计时开始于import和初始GPU检查之后、模型构建之前。没有单独加载、warmup或两次完成测量的时间，不能用总耗时拆分或补算吞吐。原候选在measurement阶段收到SIGTERM15，finally保留了异常、部分峰值和`allocated_gib_after_cleanup=0.579147`；没有把非零值写成零。

冻结父流程执行有界TERM/wait3/KILL/wait3后保留原TimeoutExpired，pool确实观察到外层rc1，GPU0 worker随后为standby_load。这些原件没有单独候选returncode、所有后代或PGgone证明，因此不称已独立证明完整进程组清理。运行目录仅有summary和首档log/measurement，没有checkpoint文件；本监督没有删除任何文件、执行模型、改变源/预算、重启或重试。有限观察已在该真实终态停止。

终态汇总和九格缺失状态：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_readonly_watch/terminal_observation_20261003T0144Z.json（SHA256 `4b4d1bafde53373954220e4b022533bf57469dfba8f5bc79eda268cd442a2f51`）。原始终态快照：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_readonly_watch/observation_20261003T014316Z.json。

原始pool结果 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790981433321562451_p04_v7_capacity_febb28e3.json（SHA256 `d3244877fd1584c4face5066bc00f336b6b9a5fde7da90e3c5b31f49179a69c3`）；summary /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790981433321562451_p04_v7_capacity_febb28e3/summary.json（SHA256 `ff8551e5117f14f7219a4e35875ae41791929bc8ee72bc3bc9d6ae9d91c74693`）；首档measurement /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790981433321562451_p04_v7_capacity_febb28e3/f000_micro4/measurement.json（SHA256 `f986ebf3fe9b607ebc46574d26e51bdba116bc6aca3dc17abcb1c20fd4ad9e33`）。日志SHA和原始metadata均保留在终态汇总，历史capacity_job.json保持原样。
