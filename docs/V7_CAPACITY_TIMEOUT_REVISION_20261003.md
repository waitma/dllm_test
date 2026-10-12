# 容量超时后的独立协议修订

根端2026-10-03T02:36:47Z已读取实际九格及四终态原件，核对精确命令/4更新/2有限loss/Adam状态/全长2048/A10080/余量，并验证40实际加载源来自同568冻结且SHA匹配。审阅 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_v2_root_terminal_review.json（SHAb6ae90637728f22bef378e71d0d2c943e50ee74e4ad52165b1f0a66d95970c5b）。后续统一micro32×GA1，但实际plan/jobs/明确v2消费者与训练发布仍未完成；原p04失败、旧8更新要求及其他门禁不改，没有重复模型或控制执行。

最新状态（2026-10-03 02:27:52 UTC）：新固定v2任务已实际succeeded/rc0，九格均完成本协议4更新/2有限loss，COMPLETE的summary SHA匹配。原profiler报告最大共同人工容量micro32；这是人工全长形状容量证据，不是实际语料效果、正式训练准入或自动batch计划变更。下文早期源码/未发布状态保留为历史。

2026-10-03T01:57:22Z：新 dispatcher 源码及唯一一轮必要冷控制完成，尚未发布或入队。本项不重试旧任务，不运行模型/GPU，不改既有 profiler、568 份冻结源、数据门禁、pool/public 入口、registry 或队列。主会话独立负责真实账本、后续部署决定及全局项目文档。

## 实际失败依据

旧唯一任务 `1790981433321562451_p04_v7_capacity_febb28e3` 的 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790981433321562451_p04_v7_capacity_febb28e3.json` 为 failed、returncode 1。其运行目录 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790981433321562451_p04_v7_capacity_febb28e3` 中，`summary.json` 只有首格 f000/micro4 timeout，单格预算 180 秒、总预算 1800 秒；不是 OOM。

该目录 `f000_micro4/measurement.json` 记录 completed_optimizer_steps=4、原 warmup_steps=2、measure_steps=6、两个有限 measured_losses，状态 failed，原因是父超时清理发出的 SIGTERM/InterruptedError。它证实首格完成两次 warmup 和两次测量，但不是原六次测量协议通过，不能从截断结果填造 throughput/九格完成或推荐 batch。

新协议独立使用 p05_v7_capacity_v2，pool steps=36，九格仍是 f000/f111/cv_segment_global × micro4/16/32；每格两次 warmup 加两次测量，单格 240 秒，总预算仍 1800 秒，optimizer hard cap 100。总预算优先于 9×240 秒，后续格实际等待仍受剩余总时间限制；不保证九格一定完成。原全长 2048、d768/L8/H12、AdamW 状态建立、BF16、whole-layer checkpointing 和 15% 显存余量推荐逻辑均来自原冻结 profiler，未修改。四次更新只用于有限容量/时延观测，不用于学习效果或模型选择结论。

## 接口和单次绑定

新源码 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_large_batch_capacity_v2_dispatch.py` 提供只读 `prepare(argv) -> (args, command)` 和未来 pool 的 `main(argv)`。五个唯一选项为：`--variant p05_v7_capacity_v2 --mode profile-synthetic --steps 36 --profile-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/profiles/p05_v7_capacity_v2 --output-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790992416352344471_p05_v7_capacity_v2_a4854542`。不接受 data/resume、重复选项或外部修改 sweep。

根端独立创建 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_job_v2.json`，SHA256 `3429ab434375501e56860e52dd83ebe6702f436a6b18b54d26752b7ff2efc152`。dispatcher 固定此 SHA，因此不能靠改账本换新 job 重试；job ID 是 `1790992416352344471_p05_v7_capacity_v2_a4854542`。账本 schema 为 v7.capacity.enqueue_intent.v2，包含 exact pool_dir/job/run_dir、source_freeze、protocol、四项 previous_failure 原件 path/SHA、automatic_retry=false、max_attempts=1。旧失败四引用的身份、failed/rc1、首格180秒 timeout 和真实2+2进展同时做语义核对。既有新任务终态或非初始日志输出会拒绝。源码不创建或改写账本/计划/通过件。

prepare 仅调用 SHA-bound `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_large_batch_dispatch.py` 的冷 prepare，复用它对原 568 源及 CPU 证据的核验，不执行旧 main 或旧任务。所得原 command 只替换 output-dir、measure-steps=2、candidate-seconds=240。未知 alias 原 argv 委派该旧 capacity dispatcher（SHA256 f259fb0f0946fb7434c1f78b06f64e5d186ece7b781fd29dac7f3f0fe0e5d866），其原 p02 链保持。未来匹配新 alias 时仍要求原 pool 分配的单 mask 0..7，用 execv 保持 PID。

## 首次必要冷控制原件

激活 pllm 后执行 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_pool_large_batch_capacity_v2_dispatch.py`，唯一一轮五组控制 passed，0.783 秒，exit 0。仅一次读取根端真实新 intent 并调用旧冷 prepare/原 profiler parse_args；没有执行 profiler 的 run/candidate/main，没有模型/ML 导入或 GPU 初始化。旧 source 漂移控制和旧 suite 未重跑。

人工 TemporaryDirectory 控制覆盖缺失/变更账本、错误协议、selected-only sweep、重复/data/resume 参数、复用旧 output、假 prior passed 即使重算 SHA 也拒绝、已有 profile 输出或终态、unknown argv 原样委派及 SHA 拒绝。单 mask/exec 边界使用 mocked prepare/exec，旧 main 始终 mocked；socket/Popen 禁用。临时 fixture 全部清理；真实新 run 仍不存在，原 ledger/source SHA 前后相同。

证据目录 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_capacity_v2_dispatch_controls_20261003`：`receipt.json` 为首次执行原件及执行后 stdout 绑定，`stdout.log` 为完整原 stdout/stderr，`source` 归档两份本轮源码。不创建 fake GPU passed、batch plan 或实际队列记录。

| 原件 | SHA256 |
| --- | --- |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_large_batch_capacity_v2_dispatch.py` | `998dc1c8f6debbaf1b811a0bb30bebbdd2b450749653f6c92ac5b7f6b02a44d6` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_pool_large_batch_capacity_v2_dispatch.py` | `2098c9b7a7048169510f5e59a9a472312a6f7aa199087564d0b39590dad47a86` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_capacity_v2_dispatch_controls_20261003/stdout.log` | `c53699c4e3b0985f5017d24465257c51c475cfaefddc1167e9c0e9ab852f4159` |

后续消费者仍保留旧 8 更新/6 loss 的协议校验，本项没有修改 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_large_batch_ablation_dispatch.py`。未来 v2 真正完成后必须明确整合新协议，不能把4更新/2loss结果冒充旧协议成功。root 审核后才决定单次新协议部署；所有停止的注释分支、独立 CUDA 专项暂停及数据门禁保持。

## 实际单次发布与只读观察 — 2026-10-03 02:07 UTC

主会话已于02:05:07 UTC实际发布并单次登记新固定job `1790992416352344471_p05_v7_capacity_v2_a4854542`。原件 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_v2_deployment_v1/enqueue_receipt.json` 记录enqueue_count=1、automatic_retry=false、max_attempts=1；历史prepared intent及旧失败记录保持原字节，状态变化由独立publication/enqueue原件记录。

02:05:09.160118 UTC，原pool的GPU2 worker实际领取该任务，外层PID739548。02:07:26 UTC只读核实池pending0/running1/results114，该任务poll-gated heartbeat距读取1.27秒；summary为running，当前测量文件0、没有COMPLETE或终态。不得将领取或新协议冷控制通过称为九格容量通过。

独立原始快照 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_v2_readonly_watch/observation_20261003T020726Z.json`。本观察只读同一任务的worker、日志、measurement、summary、COMPLETE和pool result，最晚02:45 UTC停止；不启动第二次profile、OS监听、不改源/预算/队列或清理checkpoint。关键档位与真实终态将在本节之后追加。

02:07:47 UTC首格f000/micro4实际status=passed，完整4次更新、2个有限measured_losses；peak allocated/reserved/total分别6.280078/7.330078/79.151001GiB，measured_seconds=70.919994，input_tokens_per_second=1848.167105，records_per_second=0.902425。测量原件 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790992416352344471_p05_v7_capacity_v2_a4854542/f000_micro4/measurement.json`（SHA256 `71a3e262ac46d22172841ee0cdc4fc5ac358b382d557bfceb43b565fd76b6e57`）；阶段快照 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_v2_readonly_watch/observation_20261003T020747Z.json`。此时整体仍running，九格只完成一格，不作最终通过或推荐结论。

02:12:53 UTC，f000的micro4/16/32三档均实际passed，每档4次更新及2个有限loss。micro16的allocated/reserved=21.680095/26.472656GiB、1853.436664 input tokens/s；micro32为41.995689/49.638672GiB、1837.654946 input tokens/s，总显存均79.151001GiB。此时完成3/9，其他两个模型尚无完成结果，无共同档位推荐。这只是两次测量更新的人工2048形状吞吐，不推断真实语料加速或统计显著差异。阶段全原件和各measurement SHA：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_v2_readonly_watch/observation_20261003T021253Z.json`。

02:20:23 UTC，f111三档也均实际passed、各4更新/2有限loss，整体完成6/9且无OOM或timeout。f111的micro4/16/32峰allocated分别7.796802/27.743694/54.156778GiB，reserved分别8.699219/31.728516/61.884766GiB；对应input tokens/s为1843.437454/1850.960057/1877.379282。总显存79.151001GiB。此时cv_segment_global三档尚待完成，summary仍running，不将6/9称为共同容量通过。实际阶段快照和SHA：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_v2_readonly_watch/observation_20261003T022023Z.json`。

## 实际完整终态与九格数值 — 2026-10-03 02:27:52 UTC

同一v2任务在GPU2于02:05:09.160118开始、02:27:52.089226结束，pool明确succeeded/returncode=0；外层耗时1362.929108秒，profiler耗时1362.319632秒，小于原总预算1800秒。九个唯一档均status=passed，分别完成warmup2+measure2，共36次更新，18个测量loss均有限；无OOM、timeout或缺格。每档总显存均为79.151001GiB。

| 条件 | micro | allocated / reserved峰值（GiB） | input tokens/s | records/s |
| --- | ---: | ---: | ---: | ---: |
| f000 | 4 | 6.280 / 7.330 | 1848.167 | 0.902425 |
| f000 | 16 | 21.680 / 26.473 | 1853.437 | 0.904998 |
| f000 | 32 | 41.996 / 49.639 | 1837.655 | 0.897292 |
| f111 | 4 | 7.797 / 8.699 | 1843.437 | 0.900116 |
| f111 | 16 | 27.744 / 31.729 | 1850.960 | 0.903789 |
| f111 | 32 | 54.157 / 61.885 | 1877.379 | 0.916689 |
| cv_segment_global | 4 | 7.772 / 8.854 | 1843.657 | 0.900223 |
| cv_segment_global | 16 | 27.649 / 31.082 | 1846.318 | 0.901522 |
| cv_segment_global | 32 | 53.968 / 61.846 | 1886.828 | 0.921302 |

原summary报告largest_common_synthetic_capacity_micro=32；三模型各自本次短测量中最快且保留15%余量的档位分别为f000/micro16、f111/micro32、cv_segment_global/micro32。此处仅转述真实原profiler输出，不声称两次测量更新的小差异具有统计显著性，不推断实际语料吞吐或学习效果。旧8更新/6loss消费合同不会因此自动变成4更新/2loss合同，后续整合和batch选择由主会话独立处理。

`COMPLETE.json`为synthetic_capacity_profile_only/completed，其summary_sha256=`bebc288b710af65c4b6486db7211874f1e9fa3bb1bddf4157d2da9921c2ac70d`与真实summary字节一致；training_completed=false、actual_corpus_test=false、checkpoint_created=false。九份measurement均在原finally记录allocated_gib_after_cleanup=0.015869617462158203，未写成零；外层rc0且GPU2 worker随后standby_load。原父流程逐档wait并执行有界session cleanup，但没有单独每个子进程rc或所有后代PGgone原件，本观察不冒称独立完整PG清理证明。实际运行目录没有checkpoint文件，本观察没有删除任何产物。

终态原件汇总 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_v2_readonly_watch/terminal_observation_20261003T0227Z.json`（SHA256 `9654e236ed6035652ff47ba88ca8c5dc51ba52401b19e12c2d3bac3c73798084`）保留九格精确数值、两个loss、measurement与log SHA、全部metadata、清理证据边界及历史prepared intent SHA未变。最后阶段原始快照 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_v2_readonly_watch/observation_20261003T022752Z.json`。有限观察已在真实终态停止，没有重试、第二次profile、source/预算/队列修改或真实数据操作。

运行目录 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790992416352344471_p05_v7_capacity_v2_a4854542`：summary SHA256=`bebc288b710af65c4b6486db7211874f1e9fa3bb1bddf4157d2da9921c2ac70d`；COMPLETE SHA256=`f2a7a4621e4b12ea3db1783998a4a9c392e5dfb7d995483df84c1e2e62c35b77`。pool结果 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790992416352344471_p05_v7_capacity_v2_a4854542.json` 的SHA256=`11b352c842a1b41414b58a34eee7b30d602e0b845fd1820f5b2541d4095d319d`。
