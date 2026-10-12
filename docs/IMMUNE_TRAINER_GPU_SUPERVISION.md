# Trainer GPU 软件诊断监督记录

Last updated: 2026-10-01 07:00:58 UTC

最新终态：v3人工Trainer单GPU软件功能验收已通过独立原件/绑定核对。job `1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959` 于05:44:51.491409 UTC实测result=succeeded/rc0，3正式窗口/24micro/768records和四mask eval完成；parent receipt child rc0、退出观察/后代清理true、1794.050316144秒内完成。05:53:09 UTC监督以stdlib核对26项全部通过，固定v3的526项源字节及全产物哈希链匹配；05:58:55补核对7项installed源码、four-mask绑定及v1/v2共14份失败证据均通过。没有重新执行GPU/模型或独立重算梯度；root另有固定worker Python新-I CPU语义验证通过的原件。checkpoint/保存权重0，8worker恢复maintenance、8GPU各639MiB，队列0/0/39；LOCAL monitor2213164 watching，06:02:47直接平台get仍Running。仅人工软件功能接受；真实语料双证书仍缺，T20未部署/未启动，data/corpus flags false，v1/v2失败原件保留。

新增R21/R24/R26：root已发布NEW8并构建三个source-only固定包，各528文件；06:46:25独立读审1584项新增包源字节、8项live及11项protected通过。07:00:58实际队列仍0/0/39、8worker维护，三个manifest身份稳定，尚无这三个candidate的GPU job/接受结果；后续等root实际job通知再核对执行证据。

监督范围仅限现有 `t-20261001033327-fmnql`、队列 `c20250601`、非闲时单实例 8×A100-SXM4-80GB 工作池。当前只读检查进程、心跳、队列、平台和产物；不提交、扩步、重启、取消任务，不执行 CUDA attention/operator D00 专项，不读取真实生物序列或恢复停止的 annotation 分支。

工作池绝对路径：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930`。

本记录由监督分支维护；主项目 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md` 由主分支在同一回合同步。现有 runner、registry、release map 和固定快照均不由监督分支改动。后续只有主分支完成 Trainer GPU 人工验收并入队固定软件诊断后，才观察该诊断的运行和终态产物。

## 2026-10-01 01:55 UTC：初始状态核验

| 核验 | 证据 | 判断 |
| --- | --- | --- |
| 本地监控进程 | `ps -p 2033525 -o pid,ppid,lstart,etime,stat,args`：PID 2033525、PPID 1、STAT `Ss`，启动于 2026-10-01 00:06:13 UTC；argv 为 `pllm/bin/python` 运行 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/monitor_immune_research_pool.py`，interval 30 秒 | 实际进程存在 |
| 平台直接查询 | 2026-10-01 01:55:59.493088 UTC，激活 `/vepfs-mlp2/c20250601/251105016/conda/envs/pllm`，执行 `bash /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/volc-no-proxy.sh ml_task get -i t-20261001033327-fmnql -o json`；20 秒超时；退出码 0，JSON 匹配 JobId | `Status=Running`，1 worker replica，`ml.pni2.28xlarge` |
| 监控心跳 | 2026-10-01 01:55:22.679193 UTC 读取 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/heartbeat.json`；`state=watching`，PID 2033525，time age 10.767 秒 | 心跳新鲜 |
| 队列和历史结果 | 同次 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/state.json`：pending 0、running 0、results 36，state age 13.288 秒，errors 空数组 | 无排队或运行中的实验 |
| 8 worker | 同次 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/workers/`：设备 0–7 均 `standby_load`、`maintenance_only=true`、job_id 未设置；`time` 年龄约 0–4 秒 | 维护心跳持续更新；这些记录不是远端 ps 的替代证明 |
| GPU 遥测 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/gpu_status.json` 及 state telemetry：8 GPU 各 81920 MiB，已用各 639 MiB；采样 time 1790819702.2298725，error null | GPU 总利用率来自维护负载，不记作实验利用率 |

8 worker JSON 中记录的 PID（按设备 0–7）：83079、83020、82960、82921、82987、82937、82959、83010。采样前后文件可能继续原子更新；一个 worker 的 time 比本次读取开始时刻稍晚（0.068 秒），属于读取期间更新，不据此认定时钟故障。

首次查询尝试使用不存在的 `/vepfs-mlp2/c20250601/251105016/conda/etc/profile.d/conda.sh`，conda 激活失败；CLI 输出前导文字也不能整体 `json.loads`。随后使用实际 `/vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh` 初始化并激活 pllm，按现有 monitor 的 JSONDecoder 方式提取匹配 JobId 的 JSON，得到以上直接平台证据。前一尝试不是平台异常证据。

本次未出现监督异常。R20–R27 的历史合成工程诊断结果维持既有终态，不从维护负载、历史 mode=train 或 loss 数字作真实训练/生物学效果结论。

## 2026-10-01 02:05 UTC：D01 等待和现有监听显示限制

主分支确认拟使用固定 D01：`variant=d01_pilot_runtime_gpu_r20_v1`，`kind=trainer_gpu_runtime_functional_diagnostic`，预期 3 个人工 optimizer windows，非真实语料训练。尚未收到 job_id 或入队指令。2026-10-01 02:04:10 UTC 只读 checkpoint 仍为 pending 0、running 0、results 36；daemon 心跳 age 15.72 秒，8 worker 维护心跳最大 age 4.29 秒，平台缓存 `Running`、age 50.64 秒、query_error null。

现有监控 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/monitor_immune_research_pool.py` 在模块顶层导入 `is_diagnostic` 和 `case_progress`。PID 2033525 于 00:06:13 UTC 启动；当前读取到的 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/immune_diagnostic_reporting.py` 仅识别 D00，未包含热重载逻辑。后续磁盘新增 D01 分支不会自动更新该进程已经导入的函数；旧 LIVE 表可能把 D01 transport 的 steps=1 显示为 step 进度。这个显示限制不代表任务异常，也不能作为 optimizer windows 完成证明。

监督分支保持现有 daemon 不变，后续依据原始 summary、COMPLETE、GPU_RELEASE_COMPLETE、raw、process evidence 与源码身份独立记录 D01。D01 typed reporting/monitor 的代码兼容分支由 trainer_pool_entry 负责；监督分支仅维护本文件。已将显示限制通知主分支和实现分支。

## 2026-10-01 02:16 UTC：固定父快照独立校验

02:16:23.594898 UTC 直接读取 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r20_r27_20261001/release_manifest.json` 并逐文件校验其中 524 个 `copied_files_sha256` 项。所有路径均位于该快照内且实际 SHA256 匹配；缺失/内容不匹配为 0。manifest 实际 SHA256 为 `46399204c7018d00726e3ae08142bd4eb28475c8a9a2991b3a5c90725c8824f7`，与当前人工 GPU producer 源码的固定父身份一致。

02:16:38.813969 UTC 读取 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/release_map.json`：schema_version 1，variants 28。R20–R27 的 8 个映射均指向上述固定父快照。本次仅校验现有源码文件及其 hash，未读取语料、写入快照或改动 release map。

源码预审观察到 D01 producer 的正式诊断为 3 个完整 HF Trainer optimizer windows，并以独立模型另执行 1 个 unclipped gradient oracle window；前者才是 summary 预期 3 windows 的口径。producer 仍由实现分支迭代，预审不代替固定 runtime 或 GPU 终态验收。

## 2026-10-01 02:24 UTC：并发源码漂移记录

主分支报告 producer 实现分支在 02:21:31 再次观察到非其当前实现操作的写入，涉及 exercise 的 dataset len check 和 first-window 字段；主分支声明未编辑 producer，实现分支已暂停写入。这个写入归属来自主分支/实现分支的报告，监督分支不能仅凭 mtime 或代码内容识别 writer。

02:24:56.583165 UTC 的独立只读证据：

| 文件 | 实际 mtime UTC | 大小 bytes | 实际 SHA256 |
| --- | --- | --- | --- |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pilot_gpu_release_producer_v1.py` | 2026-10-01 02:21:31.196149 | 57549 | `37541f3a8d67b78c1e2f88f253cab8282812bd2dfe63268c07844dfa45dbf63e` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_pilot_gpu_release_producer_v1.py` | 2026-10-01 02:11:14.230819 | 9673 | `326a4da87d2f48b22137233f6353950ca7d5e11dfc31bb6fd7090d7ed2e46cbd` |

当前 producer 源码确有 `len(training) >= optimizer_steps*global_batch` 与整窗口取模检查，以及 `first_window_preclip_match`、`first_window_inputs_match` 字段。较早的只读源码预览（02:08–02:10 UTC，发生在 02:10:50 下一轮 watcher 启动前）实际读取到 `len(training) == optimizer_steps*global_batch` 的精确窗口检查。这两次工具读取独立证实该检查内容变化；较早预览未记录文件 SHA，不能据此完整还原旧版本或识别 writer。监督分支只读取这些字段；未编辑、执行或测试 producer。

同次实测 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v1_20261001` 和 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/stage_trainer_v1` 均不存在。02:24:58.284588 UTC 监督 checkpoint：pending 0、running 0、results 36，8 个 worker 均维护状态；daemon 心跳 age 21.64 秒，worker 最大 age 4.11 秒；平台 `Running`、缓存 age 24.19 秒、query_error null。实际 D01 尚未建立或执行。

本监督分支至此的文件写入仅为本文件的 Markdown 补充。所有源码/测试查看均使用 rg、sed 或只读 SHA 校验；没有项目模块导入或测试执行，没有任何入队或平台 task 变更。本地临时只读 watcher（02:20:57 启动的 exec session 45188、pllm inline Python PID 2116658）只读取工作池的 heartbeat/platform/workers/pending/running/results，并用 `os.kill(2033525, 0)` 核对监控进程存在；代码没有 write/rename/unlink，也不读取或修改 producer/测试。它每 30 秒采样，在 600 秒后自然结束；此 watcher 不属于 GPU task 或现有 monitor daemon。

遵照主分支指令，继续绝对只读监督；不修改源码，不入队。并发漂移期间不将当前 mutable producer 源码视为已固定的 GPU 验收身份。

## 2026-10-01 02:33 UTC：只读复核

02:32:52.070748 UTC 再次在实际 pllm 环境，通过同一 no-proxy `ml_task get -i t-20261001033327-fmnql -o json`、20 秒超时直接查询平台：退出码 0，JSON 的 JobId 匹配，`Status=Running`，无查询错误。02:33:26 UTC 的 `ps` 仍显示 PID 2033525、PPID 1、原启动时间 00:06:13 UTC、STAT `Ss`，argv 和 interval 30 秒相同。

02:33:06.709008 UTC 本地 checkpoint 为 pending 0、running 0、results 36，8 个 worker 均维护状态，无监督异常；daemon 心跳 age 20.43 秒，worker 最大 age 4.32 秒，平台缓存 age 22.96 秒且 query_error null。

02:32:51.428084 UTC 再读取 producer：SHA256 与 02:24:56 的 `37541f3a8d67b78c1e2f88f253cab8282812bd2dfe63268c07844dfa45dbf63e` 相同，mtime 仍为 02:21:31.196149 UTC。固定 runtime 与 stage_trainer_v1 在本次采样仍均不存在。以上是两次采样端点相同的证据，不用于识别 writer。

02:20:57 的临时只读 watcher 已按 600 秒上限自然结束，退出码 0；02:31:06 起下一轮同样的只读 watcher 继续每 30 秒观察现有工作池。现有 monitor daemon 与 GPU task 没有重启或调整。

## 2026-10-01 02:40 UTC：主目标更新后的只读复核

主分支告知 active goal 已转为完成最终预训练整合并启动训练；目前仍无 D01 或实际语料训练。主分支已向用户询问团队已有审核与可用数据路径，因为原双证书缺失。监督分支不查找或读取真实序列，不恢复已停止的序列/annotation 分支；这里仅记录主分支的状态和现有资源证据。

02:40:55.095699 UTC 实测 live producer SHA 仍为 `37541f3a8d67b78c1e2f88f253cab8282812bd2dfe63268c07844dfa45dbf63e`，mtime 仍为 02:21:31.196149 UTC。`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes` 父目录不存在，因而没有该目录下的新 runtime；固定 runtime 与 stage_trainer_v1 也不存在。实际队列 pending 0、running 0、results 36；对这三个目录中的 job variant 核对，D01 条目为 0，已检查的工作池 GPU_RELEASE_COMPLETE 标记路径为 0。

02:37:10 UTC GPU 遥测仍有效：age 23.83 秒、host `t-20261001033327-fmnql-worker-0`、error null、8 GPU；worker 全为 maintenance_only，各显存已用 639 MiB/总量 81920 MiB。此采样属于维护负载。

后续只认可主分支明确给定的真实 job_id。若发现未经主分支告知的 GPU 验收、D01 队列项或新 runtime，立即报告原始路径/状态/时间，并把 caller 标为未知；不执行、修复、重试、入队或更改任何资源。继续维护本监督文件，主项目 MD 仍由主分支在同一回合同步。

02:43:04 UTC 起的临时只读 watcher 按主分支要求增加 live producer SHA/mtime、runtime 目录元数据、工作池 runs 内递归 GPU_RELEASE_COMPLETE 文件名检查；保留最初 36 个历史 result ID 作为固定观察基线，因此快速完成的新任务也不会在下一轮 watcher 初始化时混入历史基线。采样间隔仍为 30 秒，每轮本地 watcher 的生命期为 600 秒，不创建任何项目 Python 文件，不导入或执行被观察的源码。初始采样为 0/0/36、无新结果或 GPU marker、runtime 父目录不存在、producer hash 未变、全部心跳/遥测有效。

## 2026-10-01 02:53 UTC：源码漂移后的父快照复核

因 live producer 的并发漂移发生在 02:16 父快照检查之后，02:53:09.622115 UTC 额外进行一次与该事件相关的只读复核。先验证固定父 manifest 实际 SHA 仍匹配 `46399204c7018d00726e3ae08142bd4eb28475c8a9a2991b3a5c90725c8824f7`，再逐文件验证其 524 个 copied_files_sha256；缺失/不匹配 0。R20–R27 的 8 个 release_map 映射仍全部指向 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r20_r27_20261001`。

此检查确认上述采样点的固定父源码身份；不认定 live producer 漂移的 writer，也不替代未执行的 GPU 验收或实际语料训练。没有执行项目脚本、测试或读取语料。

## 2026-10-01 03:01 UTC：root 已审核的原子发布

02:59:04.715386 UTC watcher 捕获 live producer SHA 从 `37541f3a8d67b78c1e2f88f253cab8282812bd2dfe63268c07844dfa45dbf63e` 变为 `5843eb2a65e9637455c31731fa9709c4fdf0c62e84bba02cdc0df77ac041550b`，mtime 02:59:01.611218 UTC，并立即向主分支报告观测时的 writer 归属未识别。当时无 GPU 执行/队列事件；0/0/36、全部 worker 维护、无新结果/marker、runtime/stage 不存在。

主分支随后明确确认此次 caller=root：已核验私有 final_review/diff、24-case CPU 测试、env/log/JUnit 与原 live SHA/inode/mtime/size，然后逐文件原子发布 producer 和测试。监督分支据此将归属记为 root 已确认，不把它记为新的未知竞写。主分支告知下一步创建固定 source-only runtime；尚未执行 GPU 或给出真实 job_id。

03:01:26.734109 UTC 独立读取 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_gpu_producer_cpu_20261001/root_reviewed_merge.json`：实际 SHA `aa927e70f910d29175ed0009fb9c366fae809d2d8d0fc9319cd9e39f3cb4b9ad`；schema `immune.reviewed_gpu_producer_merge.v1`，status `reviewed_original_sha_checked_atomic_publication`。其中 before/after SHA 与监督分支 02:24/02:59 的实际观测相符。

同次 live byte 校验：

| 文件 | 实际 SHA256 | 实际 mtime UTC | bytes |
| --- | --- | --- | --- |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pilot_gpu_release_producer_v1.py` | `5843eb2a65e9637455c31731fa9709c4fdf0c62e84bba02cdc0df77ac041550b` | 02:59:01.611218 | 71383 |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pilot_runtime_entry_v1.py` | `50cf109b482296a9036ea5f428618b2a5049671f24ee9d248faac294ba357713` | 02:09:07.589791 | 16580 |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_pilot_gpu_release_producer_v1.py` | `e6eb22b35133a435844627cff303a3bb27560ce5710c6e04dfcac1da89920ff4` | 02:59:01.612434 | 14890 |

producer 与 wrapper 的两 additions SHA 均匹配主分支给定的固定值。监督分支没有执行这 24 项 CPU 测试或 GPU 验收，只核对审计文件、源字节和现有资源状态。该发布不等于 GPU 通过证明或实际语料训练开始。

## 2026-10-01 03:04–03:15 UTC：固定 runtime、监听重载与唯一 D01

03:01:04 UTC watcher 首次观察到主分支已告知的 source-only runtime 创建，caller=root 已确认。03:04:30.167821 UTC 独立检查 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v1_20261001/pilot_runtime_manifest.json`，实测 SHA `baeac98b962650770cf17c2f83014654c2520622e2785ead17db680a1228421a`；526 个 runtime_files_sha256 全部匹配，缺失/不符 0；两 additions 匹配 root 给定的 5843/50cf 固定值；父 manifest SHA 匹配 463992 固定身份。runtime 内无 GPU_RELEASE_COMPLETE。源包 schema 是 `immune.pilot_runtime_source_package.v1`；03:15:27 读取其创建状态仍为 `source_package_only_not_gpu_or_data_acceptance`，gpu_accepted/data_accepted/training_started 均 false。这是源包元数据，不用它推断后来 D01 的实际窗口完成量。

主分支已告知两环境 fresh inspection 各通过 8 个旧 profile 与 54 imports；该结果由主分支执行并报告，监督分支未导入运行时或构建模型。

03:15:27.585821 UTC 独立校验 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/stage_trainer_v1/application.json` 的 after_sha256，以下 3 个 live 文件实际 bytes 均匹配应用记录：

| 文件 | 应用后实测 SHA256 |
| --- | --- |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_immune_research.py` | `89e154f682e7ced2680f38bd0895eea2d18fd3447eb2e8aa0cd4184afc34fd30` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research/model.py` | `2b0b4de27f6c1dac950af510c50eb49158ae577e31e0d323d2e1be9ff370bd84` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/immune_research_pool.py` | `f3522eca58f7fedf145e8d61986d5124e0cdd9e2718c6062c53c94e563ba832f` |

stage deployment_manifest 的实际 SHA `2a7d04c846b15a6eec8fc762d1e8efc5c184e9b44dfb85ce2b41f94625633124` 与 application 记录相同。application.json 自身 SHA 为 `e30fa2899bc3111c8aa261a1604ef3fbb74e555bacd4d0b43962ab48310e8e36`。这三处应用由 root 控制，监督分支未写入源码。

本地监听的短暂重载已由主分支提前告知。`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/stage_trainer_v1/monitor_refresh.json` 实际 SHA `8340e3d8cd31dae2a16ff59dd2033cf8d9cdadae7100ec894854125ecf6972d4`，记录旧 2033525 → 新 2143672；03:09:54 实际 ps 仅见新 PID 2143672、PPID 1、STAT Ss、启动于 03:05:19 UTC；heartbeat watching。旧本地 monitor 退出是此次 root 已告知的 SIGTERM 重载；GPU task/worker 未据此重启。监督 observer 已改为核对 root 明确给定的新 PID，未把重载判断为平台故障。

主分支唯一授权 D01 job_id 为 `1790824088674261408_d01_pilot_runtime_gpu_r20_v1_9af03156`，caller=root；`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/stage_trainer_v1/d01_launch.json` 实测 SHA `cc4acc1ac2db406b72cb5d044bd387761f65c4b0c8b30ec63a31f31b78132308`，记录入队时间 03:08:08.676452 UTC、既有平台 task、无新平台提交。正式人工目标为 3 optimizer windows、24 microsteps、768 records；独立 oracle 另 1 window。biological_quality_claim=false、corpus_training_started=false。

03:09:54 实测 GPU5 worker 已运行此 ID，子进程 PID 107050，heartbeat 03:09:53 UTC；实际 argv 为 protenix_abtcr Python → live runner → 固定 D01 transport `profile-synthetic/steps=1`。该 steps=1 只用于 transport，不能记作训练步数。其余 7 worker 为维护负载。

run 的 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790824088674261408_d01_pilot_runtime_gpu_r20_v1_9af03156/request.json` 实测 SHA `5dabe5f1b0d753683e9f023ac8e6f54e5982afbaacc44e84ebc8742b8d92f35b`，绑定固定 runtime manifest、R20 profile、3 windows、1200 秒上限和 false biological claim。固定 entry 源码在写 request 和启动 child 前检查全 runtime bytes、隔离导入路径和精确 transport；独立检查的 526 个 source bytes、已观测 request 与 root launch 身份一致。真正 parent exit receipt、candidate raw 校验及 GPU_RELEASE_COMPLETE/COMPLETE 尚待终态原始文件验证。

03:13:31 实际目录含 hf_artificial、producer.log、request.json，未见 progress、FAILED 或完成 marker。producer.log 仅包含 Accelerate 对 kernel 5.4.250 的原始版本警告；未见异常栈。GPU5 利用率 100%、已用 51401 MiB。该警告与显存/利用率不作为 hang、窗口完成或功能通过证明；继续观察固定任务，不自动重试、减小 batch、fallback 或追加步骤。实际 T20/语料训练仍等待数据认证，不由监督分支启动。

## 2026-10-01 03:20–03:25 UTC：D01 v1 原始失败终态

03:19:47.406786 UTC 的只读 watcher 仍观察到 0 pending、1 running、36 results。03:20:17.466651 UTC 首次观察到 0/0/37，新增结果仅为 root 已告知的唯一 D01；8 worker 全部回到 standby_load，工作池 runs 未出现 GPU_RELEASE_COMPLETE。03:23:22.578882 UTC 直接读取原始结果和终态文件后立即向主分支报告失败；未触发重试、改 batch、fallback、取消、扩步或任何源代码写入。

结果 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790824088674261408_d01_pilot_runtime_gpu_r20_v1_9af03156.json` 记录 state=failed、runner returncode=2，device=5，started=1790824089.239825，finished=1790824813.5390358（03:20:13.539035 UTC）。父进程 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790824088674261408_d01_pilot_runtime_gpu_r20_v1_9af03156/process_evidence.json` 记录 child returncode=1、cleanup_completed=true、completion_observed_by_parent=true、elapsed_seconds=716.1638979390264、timeout_seconds=1200；receipt 中 producer_log_sha256 与实际日志字节 SHA 完全相同。FAILED 的阶段是 producer_execution，外层错误为 `Producer did not exit normally with verified descendant cleanup`。固定 wrapper 151–157 行要求 rc=0 且清理/观察为 true；本次是 rc=1 被拒绝，不能从该通用错误文字推断清理失败。

原始 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790824088674261408_d01_pilot_runtime_gpu_r20_v1_9af03156/producer.log` 的 traceback 指向固定 producer 第 946 行：`ValueError: Gradient oracle exceeded fixed budget`。固定源码 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v1_20261001/scripts/architecture_prototypes/pilot_gpu_release_producer_v1.py` 第 149 行规定 gradient_oracle=240 秒，第 944–946 行先运行 oracle 并检查该阶段耗时，第 948 行才调用 formal。因此依据固定源码与实际失败位置，本次未进入正式三窗口执行；总进程 716 秒不是 1200 秒总超时。日志中的 HF `train_runtime=701.6219`、loss 等通用训练摘要不代替 oracle 原始计时、正式窗口记录或验收证明；没有持久化 raw 可核验 oracle 计时的精确数值。

03:23:22 的 run 内实际文件仅为 FAILED.json、process_evidence.json、producer.log、request.json；未发现 checkpoint 文件，观察到 checkpoint 数量为 0。gpu_release_raw.json、gpu_release_summary.json、PRODUCER_SUCCESS.json、producer_progress.json、GPU_RELEASE_COMPLETE.json、summary.json、COMPLETE.json 均不存在。父进程未签发 GPU 验收标记；不能记作三窗口完成、功能验收通过或生物学效果结论。FAILED 的 biological_quality_claim=false、corpus_training_started=false。

独立读取的完整文件身份如下：

| 绝对路径 | SHA256 |
| --- | --- |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790824088674261408_d01_pilot_runtime_gpu_r20_v1_9af03156.json` | `68923e8ffd2d870d2bd2515df722bd00a4df1d935efba62e7503ea7a99a18b90` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790824088674261408_d01_pilot_runtime_gpu_r20_v1_9af03156/FAILED.json` | `8abeaa76e0e8dbbb532500cef9bbae92abb803cfbc3471f2c27242d84e1326ac` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790824088674261408_d01_pilot_runtime_gpu_r20_v1_9af03156/process_evidence.json` | `681b02476bcf7f7c6cc92e74df5720589a038ed0c84231b2d17b770415c2a3a7` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790824088674261408_d01_pilot_runtime_gpu_r20_v1_9af03156/producer.log` | `7130160ffeb044419ac27cbb4a4e521ffff13d475feec82bd738380073a352c7` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790824088674261408_d01_pilot_runtime_gpu_r20_v1_9af03156/request.json` | `5dabe5f1b0d753683e9f023ac8e6f54e5982afbaacc44e84ebc8742b8d92f35b` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/logs/1790824088674261408_d01_pilot_runtime_gpu_r20_v1_9af03156.log` | `f886866ba9cfe21334e2270051149002b399f07bcd94249510e63a94619f0cf4` |

03:24:26 UTC 再次逐文件复核端到端源码身份：父 release_manifest SHA 仍为 `46399204c7018d00726e3ae08142bd4eb28475c8a9a2991b3a5c90725c8824f7`，其 524 个 copied_files_sha256 无缺失、不匹配或逃逸路径；固定 pilot_runtime_manifest SHA 仍为 `baeac98b962650770cf17c2f83014654c2520622e2785ead17db680a1228421a`，其 526 个 runtime_files_sha256 无缺失、不匹配或逃逸路径。live producer/wrapper 仍为 5843/50cf；runner/registry/pool 三处仍分别匹配上节应用记录的 89e154/2b0b4d/f3522e。此终态校验是观察时刻的源字节核验；缺少候选 raw 和源码 before/after receipt，不宣称完成了成功路径上的 post-execution producer 验收。

同次激活 pllm，通过固定 no-proxy 的 `ml_task get -i t-20261001033327-fmnql -o json`（20 秒超时）独立查询，退出码 0，匹配 JobId 的 Status=Running，仍为 1 worker replica、ml.pni2.28xlarge、RoleRestartPolicy=Never。实际 ps 见 PID2143672、PPID1、启动时间03:05:19、STAT Rs、原 monitor argv；heartbeat state=watching、counts=0/0/37，age32.10秒。8 worker 的维护心跳最大 age4.66秒，全部 maintenance_only=true；GPU5 已用显存回落到639MiB，各8GPU均639/81920MiB，遥测age24.59秒。GPU总利用率对应当前维护负载，不记作实验利用率。

主分支随后确认同一失败证据，计划以实测耗时审查另一个显式版本的不可变 runtime，再单独授权一次验收；v1 与其失败证据必须保留不改。本节记录的是主分支计划，尚未收到新 job_id；监督分支继续只读观察既有资源，不自行启动新版本。真实语料/T20 仍未通过数据认证。

03:28:32.389157 UTC 再读 request.json，实际 SHA 与上述表格一致；monitor/state.json counts=0/0/37、errors=[]。临时只读 watcher session20938 已按600秒上限自然结束、退出码0；继续使用原每30秒只读观察方式，初始36份历史结果ID基线保持不变，新D01失败结果不混入历史基线。

03:39:32.517332 UTC 持续监督 checkpoint：0/0/37，全部8worker maintenance_only，heartbeat age22.58秒、worker最大age4.84秒，平台Running/query_error null、缓存age90.50秒；无新runtime或GPU marker。live producer SHA仍5843固定值，v1 runtime manifest实际SHA仍baeac98b固定值。上一轮只读 watcher session57336按600秒自然结束、退出码0；新的session10116同样仅每30秒读取元数据/哈希，增加runtime manifest实际SHA观察，仍无项目模块执行或源码/任务操作。

## 2026-10-01 03:50 UTC：失败证据保留及现有平台复核

03:50:40.689689 UTC 重读上表六份终态文件，实际SHA全部与03:23初读一致；run文件清单仍仅FAILED.json、process_evidence.json、producer.log、request.json，无checkpoint或验收完成标记。随后在pllm环境执行同一固定no-proxy非交互`ml_task get -i t-20261001033327-fmnql -o json`、20秒超时：返回码0、stderr空，匹配JobId的Status=Running、1 worker replica、ml.pni2.28xlarge、RoleRestartPolicy=Never。

同次新一轮只读watcher session35206起始采样：0/0/37，8个worker全部standby_load/maintenance_only、未设置job_id；heartbeat age3.50秒、worker最大age3.81秒，GPU遥测age1.06秒。runtime父目录仍仅固定v1，manifest SHA仍baeac98b；live producer SHA仍5843；工作池runs未出现GPU_RELEASE_COMPLETE。session10116已按600秒正常结束、退出码0。尚未收到新版本job_id，不执行额外GPU任务或修改现有v1。

03:52 的持续监督期间，额外只读核对自动生成的 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_POOL_LIVE.md`：D01表格行kind为trainer_gpu_runtime_functional_diagnostic、state=failed、进度为`?/3 optimizer windows`、loss等指标为空；文件明确corpus_training_started=false。原始raw缺失时显示未知完成量，没有把transport steps=1记作正式窗口完成或将oracle日志loss记作功能/生物学通过证明。监督分支没有改动这个自动报告文件。

## 2026-10-01 03:53–04:02 UTC：root发布固定v2及唯一新验收

03:53:10.962745 UTC watcher捕获live producer实际SHA变为 `ae1a60cbf0f155482f93dd2b1a612883b3a9de06b8be4b07458fc39360da175d`、mtime03:53:00.552697；03:54:11.032213 UTC首次观察到 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v2_20261001`，manifest实际SHA为 `3805679f1d7370042cd39741d4839ad1c6acc3805be899bed3b1f95872dace12`。先按观测事实向主分支报告，未凭mtime识别writer。主分支随后明确确认caller=root：03:53:00.558 CAS发布9源码/测试，03:54:07.621创建固定v2；这次变更归属root已确认。

03:56:52.737126 UTC分别逐文件核对v1、v2 runtime，各526个runtime_files_sha256全部匹配、无缺失/不符/逃逸路径。v1仍保留原5843/50cf两additions与baeac98b manifest；v2两additions如下，父身份仍463992固定manifest：

| v2固定文件 | SHA256 |
| --- | --- |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v2_20261001/scripts/architecture_prototypes/pilot_runtime_entry_v1.py` | `370d34a5a68dff3a8ba411bf658f5cda8e32157e729d35b0133ffaf0602f71e9` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v2_20261001/scripts/architecture_prototypes/pilot_gpu_release_producer_v1.py` | `ae1a60cbf0f155482f93dd2b1a612883b3a9de06b8be4b07458fc39360da175d` |

实际 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_budget_v2_cpu_20261001/publication.json` SHA为 `4406134215f6efb9de5836a8a314513ce35f5683070fc72207e2d84e00075991`，schema immune.root_runtime_budget_publication.v1、status published_verified；03:57:30.519265 UTC逐一核对其中9项live after hash，全部匹配。实际 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_package_20261001/build_v2_evidence.json` SHA为 `d9ef081fcf8df25e71dc495a8a38917d5cd323f6c37205341b872b4e72216161`，绑定同一publication、v2 manifest、524父文件+2additions，status built_source_package_only；gpu_accepted/data_accepted/training_started均false。这是固定源包的创建元数据，不推断之后人工D01是否运行。

03:59:32.885298 UTC因新发布事件再次独立核对实际父快照524个copied_files_sha256，全部匹配463992固定manifest。两runtime的base_copied_files_sha256均精确等于父map，runtime_files_sha256均精确等于父map加各自2additions；仅有manifest声明的两个resource symlink，无新增源/资源别名。只读检查symlink元数据，未进入数据目标或读取真实序列。v1六份失败原件实际SHA仍与03:23初读一致。

固定v2源码保持GLOBAL_BATCH=256、STEPS=3、scheduler_horizon=10000；新增阶段证据，oracle1200秒、正式训练1800秒、四mask评估600秒、总3600秒。三个正式窗口仍为24microsteps/768records，独立oracle另1window；没有修改旧v1正在执行的预算或其失败证据。v2 entry仍使用相同transport variant `d01_pilot_runtime_gpu_r20_v1` 和schema immune.pool_trainer_gpu_runtime.v1，因此版本必须由实际job_id、request runtime_root和manifest共同识别，不由variant后缀猜测。

主分支明确确认只重载LOCAL monitor，不重启GPUworker/task。04:00:57.176332 UTC实际ps仅见新PID2172045、PPID1、STAT Ss、启动时间03:56:42；旧2143672不在进程表；heartbeat watching。实际 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_budget_v2_cpu_20261001/monitor_refresh_v2.json` SHA `46ecc5cb43f3ca60f1b9aff5677ec499145dd4a0b5df6d1e54fa1c06701178e4` 与这些事实相符，记录初次检查早于首轮heartbeat、随后实际watching已验证。旧只读watcher session35206在03:57之后核对旧PID得到ProcessLookupError，原因是其固定旧PID未更新，不认作平台/新daemon故障；它已按600秒自然结束、退出码0，新session36164核对明确的新PID2172045。

04:00:57首次独立观察到新D01在GPU7运行，job_id `1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a`、worker child PID118833、started04:00:42.617068 UTC。主分支随后明确确认caller=root：04:00:41.094523仅在现有池入队此一个job；原平台task不变，无新平台提交。实际 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_budget_v2_cpu_20261001/d01_v2_launch.json` SHA `96c73118e060d774642d4d1dc3ebd6fa6947a0fe88b7ed8f2d11ff1fb29e95c2`，绑定v2固定entry/manifest、source publication、两个CPU inspection proof、新monitor及1200/1800/600/3600预算。两环境actual fresh inspection exit0/noCUDA由root执行并报告，监督分支没有执行CPU inspection或导入项目模块。

04:02:43.769350 UTC实际run request `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a/request.json` SHA `4b0d4ff231e246f79f3a4078dd8728d10b71458f8ccd3ef4b5ae061f428ffe94`，正确绑定v2 root、3805679f manifest、R20原profile、3600秒、expected_optimizer_windows=3和false biological_quality_claim；与caller已确认的launch一致。固定entry在写request和启动child前核对完整source map及导入隔离；实际source和request符合预期，尚待终态raw/parent exit及完成marker独立验收。

同次run内实际文件仅request.json、producer_progress.json、producer.log、producer_stage_evidence.json；stage evidence实际SHA `a21dabc79dcdc9e4373cecb57cae7397ea694e22f868c7f9fa1c24e253777eaa`，声明artificial_runtime_partial_stage_observations、runtime_package_name=v2、phase预算1200/1800/600，gradient_oracle status=starting，gpu_accepted=false/raw_completed=false；progress实际SHA `f2a5b2c2dad859ee60c65c9f71f7ffa7c2f33311043f0cdb6ddcd94581ab9cbf`，同为oracle starting。此处是动态阶段原件在该采样点的hash，后续正常进展可更新，不能当作固定终态或3正式窗口完成证明。FAILED/GPU_RELEASE_COMPLETE/COMPLETE尚不存在；producer.log仍仅kernel5.4.250版本警告，无异常栈。GPU7遥测100%、51401/81920MiB，其余7GPU各639MiB维护负载；不能由利用率/显存认定功能通过。

继续只读观察此已明确授权的新job，后续按实际阶段、raw、退出清理、完整source guard和parent完成marker记录；不热改runtime/source、不自动retry或fallback。v1失败证据保留，实际语料/T20仍因数据未认证而未由本监督分支启动。

## 2026-10-01 04:12–04:18 UTC：v2生命周期检查失败及独立终态审计

04:13:17.071211 UTC直接读取v2阶段文件和日志，观察到gradient_oracle status=execution_failed、FAILED存在，并立即向主分支报告原始错误。主分支随后确认同一失败事实，保持v2 immutable/live不热改、不重试或优化分支；仅授权实现分支做另一个isolated v3的Python/HF生命周期诊断，待主分支审查后另行给出真实job。T20私有v2保持hold。

04:15:03.187015 UTC直接读取v2 result：state=failed、runner returncode=2，finished=1790827961.027274（04:12:41.027274 UTC），worker GPU7。process_evidence的argv明确指向固定v2 producer及runtime；child returncode=1、cleanup_completed=true、completion_observed_by_parent=true、elapsed_seconds=709.2714709825814、timeout_seconds=3600；producer_log_sha256与实际log完全匹配。该失败不是1200秒oracle/3600秒总预算超时，也不是父进程后代清理失败。

原始阶段记录保存了额外oracle的局部观测：1个optimizer window、8microsteps、256records，elapsed_seconds=695.0527676977217，小于1200秒预算；oracle.passed=true，relative_l2_error=0.00015193938714112447，parameters_compared=78，rng_replay_equal=true。这里仅转述该局部原件，并未独立重新执行/比较梯度；gpu_accepted=false、raw_completed=false，正式三窗口不是这个额外oracle窗口。数值局部passed不能作为整体GPU功能验收或生物学质量证明。

producer.log traceback指向固定v2 producer第1044行的`torch.cuda.memory_allocated(device) == 0`检查，错误为 `Oracle left GPU tensors resident before formal measurement`；第1045行才调用formal。因此依据实际失败位置和固定源码，正式三窗口未进入执行。阶段文件未给出该失败检查时memory_allocated的精确值或GPU对象归属，监督分支不据此认定某个对象是根因。raw/summary未生成，成功路径后续framework/source before-after验收没有完成。

04:15采样的run实际文件仅FAILED.json、process_evidence.json、producer.log、producer_progress.json、producer_stage_evidence.json、request.json，checkpoint文件数为0。gpu_release_raw.json、gpu_release_summary.json、PRODUCER_SUCCESS.json、summary.json、COMPLETE.json、GPU_RELEASE_COMPLETE.json全部不存在。原始终态文件身份如下；FAILED和外层log因相同通用错误内容可与v1拥有相同SHA，具体job/runtime身份由其各自绝对路径、request与child argv共同确认：

| v2终态文件绝对路径 | SHA256 |
| --- | --- |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a.json` | `a50dbf1ff3fae065d384a22c50acbdf248c571c2bf447282ee114cf484fe4f33` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a/FAILED.json` | `8abeaa76e0e8dbbb532500cef9bbae92abb803cfbc3471f2c27242d84e1326ac` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a/process_evidence.json` | `20ac1c3511e20636cadc98d10a38ebcae94cddff39c58e518f7ef86bb540d172` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a/producer_stage_evidence.json` | `a45ec18473027d0367e6c409bc4035a83e580040782308e137880d8c0f163026` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a/producer_progress.json` | `593e808369f381dedcc2cc079432ac335a7e59bc7af2215e5e0ba6d003b2322c` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a/request.json` | `4b0d4ff231e246f79f3a4078dd8728d10b71458f8ccd3ef4b5ae061f428ffe94` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a/producer.log` | `ef02d47febc1d3ebf222fafb5bce5c12c1392d19a4198a736983bd6ff6444bdb` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/logs/1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a.log` | `f886866ba9cfe21334e2270051149002b399f07bcd94249510e63a94619f0cf4` |

04:15实际ps见LOCAL monitor PID2172045、PPID1、STAT Ss，原03:56:42启动时间和argv；heartbeat watching、counts=0/0/38。8worker已全部standby_load/maintenance_only，各心跳最大age约4.89秒；8GPU显存全部639/81920MiB。GPU7维护子进程PID现为121551，这个JSON记载不作为GPUworker控制器重启或远端完整ps证明；原诊断已结束、parent receipt清理记录及显存回落彼此一致。维护利用率不计作实验利用率。

04:17:54.567936 UTC独立逐文件检查父快照524项、v1 runtime526项、v2 runtime526项：全部SHA匹配，无缺失/不符/逃逸；三个manifest分别仍为463992、baeac98b、3805679f的上述完整固定SHA。publication记录的9个live after值也逐一全部匹配；v1六份失败证据SHA仍未变。此为终态采样的源字节身份核验，不替代未发生的成功候选/installed framework before-after证明。

同次在pllm环境执行固定no-proxy非交互`ml_task get -i t-20261001033327-fmnql -o json`、20秒超时：returncode=0、stderr空、匹配JobId的Status=Running，原1replica/ml.pni2.28xlarge/Never restart policy。LIVE的v2行仍为kind trainer_gpu_runtime_functional_diagnostic、state failed、`?/3 optimizer windows`、指标空，没有将额外oracle的1窗口记入正式3窗口验收。

新只读watcher session89184于04:17:54.591621开始，每30秒读取现有资源/metadata，并增加明确获准v2阶段/receipt/marker的实际SHA观察，初始counts=0/0/38、全部8worker维护、无监督资源异常。它不导入项目模块、构建模型或执行GPU代码；没有写入源码/runner/registry/runtime或入队。保留36份历史result基线，新增两份D01失败结果仍分别显示，不混入历史已完成记录。等待主分支另行明确的新版本/job。

04:30:20.678094 UTC只读核对主分支的 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_budget_v2_cpu_20261001/terminal_review.json`，实际SHA `65ab35af8069109d7914c0cbd698af3da4d371824eac936646415d5ec3d889cb`，job_id对应同一v2。主分支已完成独立终态复核并同步其11份MD/计划；后续遵其指令不循环重复1576个完整源码文件校验，仅在实质异常或下一actual job时针对性核验。v3仍为新私有CPU生命周期定位，暂无GPU job。

04:30:21.079903 UTC最新只读心跳/队列采样为0/0/38，全部8worker maintenance_only；monitor age11.56秒、worker最大age4.95秒，平台Running/query_error null、缓存age79.49秒，GPU遥测age28.25秒。live producer AE固定值、v1/v2 manifest及v2终态阶段/receipt文件SHA均未变，无GPU marker或新runtime。session89184已按600秒自然退出，session25971继续每30秒只读观察；常态checkpoint仅精简输出心跳/队列，状态变化和异常保留完整metadata，不改现有monitor源码或任务。

04:45:06.103493 UTC持续心跳/队列checkpoint仍为0/0/38、全部8worker维护，无资源异常；monitor age10.87秒、worker最大age4.71秒，平台Running/query_error null、缓存age78.86秒，GPU遥测age7.87秒，受观察metadata/signature未变。04:41:05起的session93655延续同一30秒只读观察；上轮session25971按600秒自然结束。没有新增实际GPU job/runtime/marker，也没有重复1576项源码扫描；等待root明确下一版本及真实job。

04:55:23.556365 UTC持续监督仍为0/0/38、全部8worker维护，资源异常为空；monitor age4.53秒、worker最大age4.42秒，平台Running/query_error null、缓存age7.17秒，GPU遥测age21.79秒，受监测状态未变。session93655已按600秒自然结束，04:53:23起session70993继续相同30秒只读metadata观察；未新增GPUjob/runtime/marker，未执行v3代码或重复完整源码扫描。

05:03:50.350782 UTC在pllm环境独立执行 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/volc-no-proxy.sh ml_task get -i t-20261001033327-fmnql -o json`（只读、20秒超时）：exit_code=0、stderr空，精确匹配JobId的Status仍Running，stdout实测SHA为 `07f2934f80bef56b702fad78f994ff1f85d445b5747ed792c9c690ec1ae4cd03`。同次实际ps仍见LOCAL monitor PID2172045、PPID1、STAT Ss，03:56:42启动时间与原monitor argv未变。

05:04:06.158254 UTC资源采样仍0/0/38、8worker standby_load/maintenance_only，monitor age2.14秒、worker最大age4.35秒，平台缓存Running/query_error null/age37.42秒，GPU遥测age1.16秒；live producer、两个runtime manifest与v2终态阶段/receipt SHA均保持原固定值，无新runtime/job/完成marker。session70993按600秒自然结束，session74251继续同一30秒只读观察；本次未重新扫描1576项源码，也未执行v3或GPU代码。

05:04:36.182549 UTC只读watcher捕获live producer变化：由ae1a60cb变为 `57352e48e7d06e146700c194f59ea6c6858191e85b24ea4b27bafca0aceac723`，实际mtime为05:04:13.765500 UTC；立即向root报告。当时尚未收到该次发布的caller明确确认；不根据文件名推断实际写入者。固定v1/v2 manifest和v2失败证据未变，0/0/38、8worker维护，未发现新runtime/job/完成marker。本分支工具仅写本MD，未编辑producer或测试。

05:06:22.913405 UTC针对性读取三个producer源码文件并核对读前后inode/size/mtime稳定：live为上述57352e48值（size94680、inode25873721272），固定v1仍5843eb2a、固定v2仍ae1a60cb，二者均与原不可变包预期匹配。未重复1576项旧源码扫描，也未进入任何外部数据symlink。

05:06:50.001214 UTC独立读取 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/publication.json`，SHA为 `d9f2840dd033ab99058f58f73b6ec2c766f46b0bd6c4495ad3fd369829918c23`，schema=immune.root_runtime_v3_publication.v1、at=05:04:13.768563 UTC、status=published_verified。该原件描述8项发布、runtime_name=r20_r27_pilot_v3_20261001，并自述gpu_execution=false、corpus_training_started=false；自述不替代实际caller确认或GPU验收。

05:07:25.423785 UTC对publication所列8处live源码/测试字节逐项独立SHA核对，全部匹配after值，各次读取期间inode/size/mtime均稳定。只读取.py源码，没有执行测试、项目模块或GPU代码；对应身份如下：

| live文件绝对路径 | 实测SHA256（与publication after匹配） |
| --- | --- |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pilot_runtime_package.py` | `999e471a7a6faf4c054f45a642680ce122b26c9a671a5bdd184cc2c7540235e8` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pilot_runtime_entry_v1.py` | `05574a849a5b2b4f1ce0205ebab0f1ad5bab1f53c6e1c4eb1932f1b84bcb4bda` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pool_release_dispatcher.py` | `32bec9e3dc6e5f4ef94ae9f3a2c2ae3d75be0966fb35f4a448ba21c69e61976c` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/release_startup/sitecustomize.py` | `bf95c573db0d11703ff41854806b5696c196c187e19da6e20c8313b360ce8976` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/immune_diagnostic_reporting.py` | `68f5bb768bee7decd2a36427d659c9e59d7d1e83d50a266b07e79a9e1d2d7483` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_typed_trainer_runtime_reporting.py` | `ff5b2a04684c60df9938523532b01ff6fe6770c3c86a04147c5f25918fdb20e2` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pilot_gpu_release_producer_v1.py` | `57352e48e7d06e146700c194f59ea6c6858191e85b24ea4b27bafca0aceac723` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_pilot_gpu_release_producer_v1.py` | `6a9d1937bd76c59fbc1a851f15ebdabb4f507c13c9e90948264ba4fe83aa0428` |

截至05:09:05 UTC，root已通过协作消息明确确认其为05:04:13.768563 UTC八文件CAS发布的实际caller，publication SHA与上述独立读取一致；前述caller待确认状态已解决。root说明后续由其构建固定v3、做两环境实际package导入检查、刷新LOCAL monitor并另发真实manifest/job；该消息是计划授权说明，不能视为GPU已执行或通过。继续每30秒只读观察，保留v1/v2失败；05:08:36.577546 UTC采样仍0/0/38、无异常、新runtime/job/marker。

05:10:06.687936 UTC实际观察到已告知的新固定包 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v3_20261001`；manifest SHA为 `cc13036fc492109afbf4c9cf9fd11a4c87684fc9702b83ce86c961b2a08294fe`，created_at=05:09:58.612521 UTC。05:11:36.397771 UTC仅对这个新包逐文件独立核验526项：全部canonical/contained且SHA匹配，无缺失、逃逸、读间漂移，manifest读取前后稳定；base524映射与实际固定父manifest一致。两处外部alias仅读取symlink metadata、目标与manifest一致，不读取数据内容；没有重复旧三包1576项扫描。两新增项仍是wrapper05574a84、producer57352e48；创建manifest的gpu/data/training=false是source-only创建信息，不能用于推断后续实际job进度。

静态读取固定v3源码确认：wrapper仍仅准入固定五项D01 transport；实际子进程正常退出、parent观察/后代清理receipt、strict candidate及runtime unchanged均满足后，才由parent写GPU_RELEASE_COMPLETE，随后summary/COMPLETE。producer将额外tiny native Trainer生命周期预检限制为120秒、1窗口/8条记录，counts_toward_formal=false，另有原模型额外oracle；正式目标仍3窗口/24micro/768records、micro32/GA8/global256及10000 scheduler horizon，phase预算1200/1800/600、总3600秒。严格候选核对preflight及oracle/formal的actual CUDA cleanup、记录owners weakrefs全部释放、清workspace前后实测、最终allocated精确0；CPU预检不能授予CUDA通过。这些是固定源码的校验要求，尚非本分支执行/实际GPU结果。

05:12:36.892659 UTC旧临时helper因仍固定观察2172045而报ProcessLookupError；root已预告此次LOCAL reload，不据此认定平台或新daemon故障。实际ps随后见新LOCAL monitor PID2213164、PPID1、STAT Ss、05:12:16启动时间和原argv。05:13:30.441749 UTC直接读heartbeat/pid均对应2213164、watching、0/0/38；`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/monitor_refresh_v3.json`记载2172045→2213164、新started05:12:16.691962 UTC，GPU workers/task重启及new platform submission均false。监督仅更正自己的临时只读helper观察PID，不执行任何daemon/GPUworker操作，等待root明确actual v3 job。

root明确实际caller已于05:14:40.610538 UTC在原pool only入队唯一v3 job `1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959`，无平台提交，并确认LOCAL刷新为2213164。05:15:51.414956 UTC独立读取 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/d01_v3_launch.json`，SHA为 `65f3b9d2ecddc036c72e13c3c83ae2c83646da2ca3d54c27e72ff0e9168eac9a`，job_id/runtime_root/manifest/publication均与root消息和本分支实测一致。此launch记录绑定build evidence SHA41b3a90f、pllm/worker两actual inspection SHA a8e7ef25/2c8a9113；这些CPU检查由root执行，监督未代为执行，也不作为CUDA验收。

该次实际request路径为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/request.json`，SHA `f70d18212e26e223de99be3d299803faa6210a288c02435f6a87d84491ea5376`；确指固定v3/cc13036f manifest、原R20 profile、3600秒总预算、expected3 optimizer windows。实际worker JSON记录GPU3、runner PID135405、started05:14:42.519749 UTC、state running/新鲜heartbeat；该JSON不是远端完整ps证明。05:15:06持续资源观察为0/1/38，其余7worker standby_load/maintenance_only，LOCAL心跳/平台缓存正常。

05:15:51 UTC实际stage文件SHA `30f56698aed9cdbff0418f000c49520d29dc83ca4519e24b59d776ca6102e2f4`（后续阶段更新会改变），lifetime_preflight局部status=observed_within_budget、elapsed=1.8073859699070454<120秒，1窗口/8micro/8records。实际CUDA cleanup记录initial allocated=0、gc/empty_cache后allocated=17039360/reserved=20971520字节、workspace clear_called=true、clear后及最终allocated=0，released=17039360；11种owners weakrefs均false、alive_object_types空。该额外tiny预检的GA/CE局部passed=true只转述阶段原件，不计正式3窗口或完整GPU通过，也不据此反推v2未记录的对象根因。

同次progress SHA `f2a5b2c2dad859ee60c65c9f71f7ffa7c2f33311043f0cdb6ddcd94581ab9cbf`，stage=gradient_oracle/status=starting；full oracle phase实际allocated_before_model_bytes=0、budget1200秒。gpu_accepted=false、raw_completed=false，正式三窗口尚未开始；FAILED/parent receipt/COMPLETE尚不存在。session76271从05:14:36开始继续30秒只读资源观察，观察新任务状态；待后续阶段及终态再核对raw候选和parent完成凭据。

05:18:08.355751 UTC独立读取实际GPU遥测：GPU3 utilization=100%、memory_used=51401MiB（约50.2GiB）/81920MiB；其余7GPU各639MiB并对应维护worker，不计作诊断实验利用率。实际遥测time=1790831859.308179，非推断训练吞吐/质量。session76271及临时专属stage观察session70148分别按600/300秒自然结束；05:25:31起session32650合并同一新v3阶段/cleanup/receipt只读观察，仍每30秒、无实际daemon/worker操作。

05:27:01.945534 UTC独立捕获实质阶段更新并立即报告root：stage文件SHA `ee8b858bf219252fbad97c3d39471f35248e3ff7b69d1fccb15efbf50146b4cb`，full gradient_oracle status=observed_within_budget、elapsed=706.9961670227349<1200秒，1窗口/8micro/256records、oracle.passed=true。该阶段实际cleanup同样11个weakrefs全false、alive_object_types空；initial allocated=0、gc/empty_cache后allocated=17039360/reserved=41943040字节、workspace clear_called=true、clear后及最终allocated=0、released=17039360。此为本次full oracle阶段的原始局部观测，不是v2根因反推、正式3窗口完成或最终GPU验收。

同次progress SHA `b87575eac1826f661105ba737ec6ba38768a6fe272d95f3006e9ac96326af0eb`，stage=train_optimizer_step/status=starting/budget1800秒；据实际阶段进展，正式阶段已开始，而非此前尚未进入formal状态。formal/eval及正常child退出、parent candidate/source与GPU_RELEASE_COMPLETE/COMPLETE仍待后续实际证据；gpu_accepted/raw_completed均false。GPU3 running，队列0/1/38，LOCAL monitor/worker心跳、平台Running和遥测新鲜度均正常；受观察producer/三个manifest身份未变。

05:30:25.446740 UTC独立只读补核对launch绑定的现有CPU/build/reload凭据（未重新执行检查）：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/inspection_pllm.json`实测SHA `a8e7ef25476f491e2e72c1b79824fd3b32d561a8c68ec74df468bc0ce4c2935f`，同目录inspection_worker.json实测SHA `2c8a91132da20addd7c6146f08370ff5251334c79707ffc545cb1f3e394414b2`，均匹配launch；两report均runtime模式、8原profile/54imports、module_roots指固定v3、model_constructed/gpu_started/training_started/data_gate_executed=false。这些标志仅描述当时CPU检查，不否定其后的实际人工GPU诊断。

同目录 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/runtime_build.json`实测SHA `41b3a90f5032daf9c47ea35ae6ff65ef9b4d6cda16bad635ec9a5d876fda5d08`，与launch绑定值一致；记录的v3 manifest/父SHA、524+2文件计数、两新增源码及publication身份均与本分支先前源字节审查一致。LOCAL monitor_refresh_v3.json实测SHA `d146ff7b43903c72eb7eaf47d01732073996266a8b8dfb0333855977a7989d95`。此补核对为证据身份一致性，没有授予GPU/数据接受或执行项目模块。

05:33:02.557303 UTC只读watcher捕获第1个正式窗口并立即向root报告。05:34:08.525896 UTC针对性直接读取同一stage原件（SHA `782f804b32111f64fff3b0717eead5fc84fba5d04dc13e42e9d52e4fcb7a3706`）：train_optimizer_step status=running，optimizer_steps=1/microsteps=8/records=256/window_count=1，elapsed=353.9542229473591<1800秒；progress SHA `ff2364ed181bc072c96bfaa449b248346ecfb7e891961297f98e242d0f9f85f3`。这是formal第1/3窗口，不是额外preflight/oracle。

该正式窗口原件记录78组参数梯度有限且非零、observed gradient norm=0.9999999075594033、preclip norm=25.093288745417127、AdamW state78组/604809720字节；learning_rate=0、parameter_delta_l2=0与固定原scheduler首个warmup窗口一致，不能据此误判窗口未执行。这里只转述partial stage原件，未独立重算梯度，也不以loss/梯度作为生物学质量结论。后续2窗口/eval/parent正常退出和完成guard仍待实际终态；gpu_accepted/raw_completed保持false，队列0/1/38、GPU3运行、无资源异常。

05:39:01.384773 UTC只读watcher捕获第2个正式窗口并立即报告root。05:40:48.757275 UTC直接核对stage（SHA `28343610174f671abf4039571bee7344bc10369ace3a7944096bf34cebc90286`）：train_optimizer_step仍running，2optimizer windows/16micro/512records、elapsed=705.5507740676403<1800秒；progress SHA `6d0008f5b90469404f713327ae704ce74c7ae0d5b42b431a1aba3b15e761d2ce`。第2窗口为8micro/256records、78组梯度finite/nonzero、preclip norm24.94294147355261/observed norm0.9999999304060149，learning_rate=1.0000000000000002e-06、parameter_delta_l2=0.008336347685184582；AdamW state仍78组/604809720字节。该参数变化为阶段软件运行原件的局部观察，未独立重算、也不证明生物学质量或完整GPU验收。

session32650按600秒自然结束，05:38:01起session88066继续同一30秒只读阶段/资源观察。第3窗口、四mask eval、parent正常退出/候选与完成marker仍待；gpu_accepted/raw_completed=false，GPU3 running、其余7worker维护、0/1/38，producer/manifest身份未变，无监督资源异常。

05:44:31.926817 UTC实际观察到正式3窗口/24micro/768records已记录、four_mask_evaluation运行；05:45:01.989005 UTC实际观察到0/0/39、parent GPU_RELEASE_COMPLETE passed及pool COMPLETE completed、child receipt rc0/清理/parent观察true，随即向root报告并继续原件核对。05:46:48.817428 UTC直接读取实际result：state=succeeded、returncode=0、finished=1790833491.4914086（05:44:51.491409 UTC）、GPU3；marker status=completed与pool result state=succeeded是各自协议字段，不混用。

05:53:09.763270 UTC监督仅用stdlib读取严格JSON（拒绝重复key/非有限常量/证据symlink），核对26项全部通过：实际终态/request与固定v3身份；raw、候选summary、producer readiness、parent GPU marker、pool summary/COMPLETE完整SHA链；计划/fixture digest；模型/adapter/execution source_binding每项对固定runtime映射；raw source before/after与526项manifest映射；installed源码before/after；正常child退出/清理/argv/log；原batch/horizon、正式计数和额外窗口区分；三个CUDA stage的owners/零allocated；phase预算；证据读前后稳定及无FAILED。仅对新v3再次逐字节读取526项，全部canonical/contained、SHA匹配，无缺失/逃逸/读间漂移；没有重新检查旧三包1576项，也未执行项目模块/测试/GPU代码。

终态实际正式输入为[32,2048]、micro32/GA8/global256、10000 scheduler horizon；24个实际batch逐项形状匹配。第1窗口LR0/delta0，后两个窗口有正LR和参数变化，属于固定原warmup行为；三个窗口各8micro/256records、78组梯度finite/nonzero、AdamW state78组/604809720字节。原件如下，不作为生物学质量或真实语料效果结论：

| 正式窗口 | learning_rate | parameter_delta_l2 |
| --- | --- | --- |
| 1 | 0 | 0 |
| 2 | 1.0000000000000002e-06 | 0.008336347685184582 |
| 3 | 2.0000000000000003e-06 | 0.016472997697818015 |

| 独立阶段 | 终态原件elapsed_seconds | 预声明budget_seconds |
| --- | --- | --- |
| 额外tiny生命周期预检（不计formal） | 1.8073859699070454 | 120 |
| 额外full gradient oracle（不计formal） | 706.9961670227349 | 1200 |
| 正式3窗口 | 1054.3746531940997 | 1800 |
| 四mask evaluation | 5.117227207869291 | 600 |

parent receipt总1794.050316143781秒，pool summary含parent候选核对后总1800.9288187921047秒，均属于3600秒总体预算；不将总体耗时与1800秒formal阶段预算混比。tiny/oracle/formal三个实际CUDA cleanup均记录11种weakrefs全false、alive_object_types空、initial allocated=0、gc/empty_cache后allocated=17039360字节、workspace清理后/最终allocated=0，released=17039360；raw最终retained_checkpoints=[]、final_cuda_allocated_bytes=0。GPU原件UUID fb245801-a17e-a87d-2a22-0db07f1c5025与GPU3遥测对应。

v3终态身份（绝对路径与实际SHA）：

| 文件绝对路径 | SHA256 |
| --- | --- |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959.json` | `d17d8b23cc5c2f75a201a181a088bcd5798168dc565bfb9bba0fe4f7ff6e0050` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/request.json` | `f70d18212e26e223de99be3d299803faa6210a288c02435f6a87d84491ea5376` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/gpu_release_raw.json` | `c4b9f0988c7794cb508fe0886bad62faf065cb718c7f56eb06aac4ae7e544ede` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/gpu_release_summary.json` | `3750829f739f63e1d6c557a67536282c02fabdfec8d8190ece48114d312a7630` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/PRODUCER_SUCCESS.json` | `075372f76cdb6c46e0cd107c0120d72a7a7cfe2daa31b1bcc83903d5852f8fa8` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/process_evidence.json` | `0e68ee0d7666e324821d6573b56d73ffe3dbf016cc9ffcef17887424839ea4b1` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/GPU_RELEASE_COMPLETE.json` | `8d57b5de3b1e131e9fc2dc1ac4647dc25097425576bd6015bc6783fdbd4f290f` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/summary.json` | `6861a208eeb822d4d6dde4956b5ef493e47a33857d7606b133ae01b938a34e2f` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/COMPLETE.json` | `3f7e8abe0d4273421536b0aeb90d220de0074656df6b01c5becc38ecb284c955` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/producer_stage_evidence.json` | `933772904c8f5021bcc2d2b437c590c26a9fba01a45a916b4533b7e859c730e7` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/producer_progress.json` | `8b3fbdd0600fa9d24426b244a4333bc0272419380f2e9972c6fbbc9e4a3f09b4` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/producer.log` | `9ac021286d32833114b89937ce6e3886b55cf3dd1f65a8f678fa2deedccd9687` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/logs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

05:58:55.139855 UTC补核对raw所列7个实际installed-source文件SHA全部匹配，且raw before/after一致：4处固定v3内部源码及HF Trainer/Torch AdamW/Accelerate 3处 `/vepfs-mlp2/c20250601/251105016/conda/envs/flow/lib/python3.12/site-packages/`源码，未导入这些项目/框架模块。候选与raw evaluation绑定一致：mask levels [.25,.5,.75,1]、12 actual eval batches、完整输入[16,2048]、原TokenValidationTrainer，8个raw population key覆盖4mask×procedural_mixed/full_capacity；root的实际fresh CPU verifier另对recount作语义核验。本分支未独立重算GPU梯度或重新运行模型。

同次独立核对v1六份、v2八份原失败证据SHA全部仍匹配本记录原值，两个旧job的COMPLETE/GPU marker仍不存在。root已明确仅清理本v3下两个空hf_*目录；实际现两个目录均不存在，保留全部11个job证据文件，无checkpoint或保存权重文件、无FAILED。原始partial stage文件的gpu_accepted/raw_completed=false及source-only创建manifest false保持各自观察/创建语义；最终接受以实际result及parent完成哈希链为准，不据这些旧false字段误判终态失败。

root报告已于05:48:03.846156 UTC使用worker同一Python新-I CPU进程执行固定verify_completion/producer.verify_candidate，通过且local_cuda_initialized=false。监督实际读取 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/terminal_review.json` 的SHA为 `26983ce22d96f73624e86b10e9793c7817c5c385602ebd1f6e531e327459319b`，与root消息一致。root说明曾在首次提取时误从COMPLETE读corpus字段，后改为实际wrapper summary；监督独立检查summary的corpus_training_started=false，未改任何生产验收/标准。root另已同步其11份MD/protocol；本分支只写本专属MD。

05:53:09实际8worker均standby_load/maintenance_only、8GPU各639MiB；05:57:30持续观察为0/0/39、LOCAL2213164 watching、心跳新鲜、平台Running/query_error null、无新source/runtime/job。session88066已自然结束，session11253从05:57:30以相同30秒间隔继续只读metadata观察，观察段900秒以减少临时helper换接；现有daemon/worker/task均未操作。遵root终态指令，后续无新job/source/pub或实质异常时不重复526/旧1576或任何CPU/GPU验收。

06:02:47.755798 UTC在pllm/no-proxy下独立平台非交互get（20秒超时）exit0、stderr空，精确JobId=t-20261001033327-fmnql、Status=Running，stdout SHA `2d701dbeccf537f232e3826242c5b1a8839b2c61970fc46ec46f53140a5a725d`。同次实际ps仍PID2213164、PPID1、STAT Ss、05:12:16原启动时间与argv。此后保持现有资源30秒只读监督；真实语料双证书仍缺、T20未部署/未启动，没有新增任务或生物学质量结论。

06:13:46.444416 UTC仅持续资源/metadata观察：队列0/0/39、8worker均维护，LOCAL monitor age21.50秒、worker最大age4.50秒，平台Running/query_error null/缓存age56.96秒，GPU遥测age19.02秒；受观察source/manifest/parent终态marker身份未变，无新job或资源异常。session11253按900秒自然结束，session51049继续相同30秒只读观察。没有执行CPU/GPU验证、项目模块或重复源码扫描；等待root明确下一实质任务/变化。

06:24:55 UTC记录root最新计划通知：producer/entry两个既有工程分支仅在全新私有树准备R21/R24/R26同Trainer人工GPU检查，尚无live发布、固定runtime构建或入队。预计runtime路径分别为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r21_pilot_v1_20261001`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r24_pilot_v1_20261001`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r26_pilot_v1_20261001`；这些为计划路径，并非已建立/已验收事实。root说明后续仅复用已有各model variant的profile-synthetic/steps1精确任务格式，当前不增加diagnostic variant、不改registry/pool/runner、不重启GPUworkers；等root明确实际publication/runtime manifest/job后再针对性读取核对。R20 v3及v1/v2原件保持，数据双证书仍缺、T20仍私有未部署。

session51049持续30秒只读metadata观察至06:23:47.339296 UTC：pending0/running0/results39，heartbeat age25.95秒、worker最大age4.11秒、平台Running/query_error null/缓存age61.85秒、遥测age16.59秒；受观察身份未变、issues=[]。本次只记录计划通知和资源观察，没有新增source扫描或CPU/GPU验证。

session51049按900秒自然结束，session64690从06:29:12.708933 UTC继续同一30秒只读观察；续接首次event=change是观察helper初始状态，与前段已知身份一致，并非新源码/任务变化。root随后要求近5分钟资源回报，确认三候选仍未发布/排队、后续由root审核统一发布。06:29:12、06:31:12及06:33:13.090619 UTC状态均pending0/running0/results39、issues=[]、受观察身份未变；后者GPU遥测age9.20秒且平台缓存Running/query_error null。

06:33:44.101289 UTC独立读取 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/heartbeat.json` 与workers目录：LOCAL PID2213164 watching、heartbeat age约27.7秒，8worker设备0–7全standby_load/maintenance_only=true，job_id及runner_pid均null，worker心跳age0.544–2.905秒。worker心跳所报PID按设备顺序为83079/83020/82960/142281/82987/109787/82959/121551；这里是心跳原件，不冒充容器内实际ps。LOCAL monitor另有实际ps：PID2213164、PPID1、STAT Ss、05:12:16原启动时间及原argv。维护负载不记作验收实验或实际语料训练。

06:33:45.026234 UTC在pllm/no-proxy下独立平台非交互get（timeout20秒）exit0/stderr空、JobId=t-20261001033327-fmnql、Status=Running，stdout SHA `0d40e175995712df77123533b11a9558219d66727e790d41655496041e9fb9d8`。监督已向root回报近5分钟无异常；没有重跑历史验收、启动任务或扫描私有候选源码。

## 2026-10-01 06:46 UTC：R21/R24/R26新增source-only包

06:42:43.951235 UTC持续metadata观察首次发现R21/R24/R26三个新runtime目录，队列仍0/0/39、8worker维护、issues=[]；监督先报告实际目录出现及caller/build通知待确认。06:43:46.470420 UTC独立读取正确的pilot_runtime_manifest.json身份。root随后明确实际caller=root，NEW8已发布并完成三份528文件source-only包构建，两个环境fresh CPU隔离导入仍进行中，尚无新GPU job；监督解除caller未知状态。最初辅助reader猜测runtime_manifest.json得到的false只是文件名读取错误，已用rg定位pilot_runtime_manifest.json并核正，不是包manifest缺失。

实际新增包身份如下，均加入后续30秒只读metadata观察；manifest内gpu_accepted/data_accepted/training_started=false属于source-only创建状态，尚不构成GPU接受：

| candidate variant | manifest绝对路径 | 实际SHA256 |
| --- | --- | --- |
| r21_refined_head_gate | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r21_pilot_v1_20261001/pilot_runtime_manifest.json` | `eaefeda0391d1e93cf58f0f0677905cd8323c190c8b769ff6ff579e044efe5b6` |
| r24_refined_fixed_scale | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r24_pilot_v1_20261001/pilot_runtime_manifest.json` | `ca0831164397882ab71217336f4cfce5c91b1a30a3d93012cd433c321f910325` |
| r26_refined_local_global | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r26_pilot_v1_20261001/pilot_runtime_manifest.json` | `47f60d1238b1b5aff18a408e8b5be69c047e7c251641c0ba05f9b9d41347b7db` |

06:46:25.995802 UTC对本次实际新增发布/包进行独立stdlib读取审核，全部通过。严格JSON拒绝重复key/非有限常量，原件与源码读前后stat稳定，实际8份live源码匹配publication after SHA；publication的11个protected before/after完全一致，当前11份文件SHA亦匹配（仅11个guard文件，没有重扫R20旧包）。三新包各528项源字节全部canonical/contained/regular、SHA匹配；每包524父map+4additions无重叠且精确组成runtime map，39 measured条目匹配；build的variant/runtime/manifest/计数/4新增身份全绑定。合计仅本次新三包1584项，没有再次扫描旧526/1576。每包两个external alias只核对readlink与manifest，未打开、遍历或hash真实数据目标。

独立读到root发布/构建原件：

| 原件绝对路径 | 实际SHA256 | 原件状态 |
| --- | --- | --- |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/publication_sources.json` | `ddf8ab715c54856be11e77b8cb36eaa411149682e25fe969dfd47ed9a6858c67` | published_new_sources_only_verified，at06:41:48.422458 UTC，8项新增、before null |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/runtime_build.json` | `061ce1ff9af0008195b9c9bbd64572e7a460c51f50e0d3a4a1af432212de9d8e` | built_verified_source_only，at06:42:23.613370 UTC，gpu_acceptance/corpus false，publication SHA绑定 |

publication的routing_published/runtime_built=false是其发布时状态，后续另有runtime_build原件，不按旧字段否认已构建事实。三包各四新增包含本candidate wrapper/producer及共用candidate package/inspect：共用package SHA `643553c27188c179e9313235cc2425ae390b79bfe86f2127f7a6734a25fa4c46`，inspect SHA `3f6a8f3f86a7a89e085d47324a229e5c3034b289d30b52983cec8e8b7dc48b53`；R21 producer/wrapper为 `b16219c4d5073a5fa4116bf6a3f1475f4ac952054ce47237db9cd52840831e6c` / `f63355f076d09d037b0408680c1a20949465f576fd073540c443b8a76c29b8df`，R24为 `059d1346b2c2ad6c3d876a69ca0660314f76c03c29d3c414e3e7d082bf02081b` / `854b68acefca3ab05bffb1898c480df8ec431dacdc76ea3dbe0f820c66f2e1f6`，R26为 `a8f4c45dd6901e0d7c4427c41652b2e3e7262472bd04fff500bc217545959ce9` / `f46ab2fe6b0cda0a1736486411f5371e883a009138e87ba8b322d426a8778697`。这些为源码身份，不是实际GPU梯度或训练质量结果。

session64690自然结束；session55785从06:45:13.488654 UTC以同样30秒观察三新manifest及原资源，初始0/0/39、8维护、LOCAL2213164新鲜watching、平台Running/query_error null，R20原producer/manifest/终态标记保持。监督没有导入项目模块、执行CPU测试/导入验收或GPU/模型代码，没有操作daemon/worker/task。等待root实际job通知，数据双证书仍缺、T20仍私有未启动。

06:47–06:59 UTC持续30秒资源/manifest观察未发现异常或新job，定时checkpoint均issues=[]。session55785按900秒自然结束；session80693从07:00:58.720261 UTC继续同一只读观察，pending0/running0/results39、8worker维护、heartbeat age12.73秒、worker最大age4.80秒、平台Running/query_error null/缓存age15.85秒、GPU遥测age14.98秒，R20原标记与三新manifest身份均未变。仅续接stdlib观察helper，没有重跑源码全量检查、CPU/GPU验收或操作实际daemon/worker/task。
