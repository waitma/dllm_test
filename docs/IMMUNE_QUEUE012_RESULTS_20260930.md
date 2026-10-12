# queue012 no-ESMC 运行结果复核（2026-09-30）

Last checked: **2026-09-30T19:25:31Z**。只读调用 `volc ml_task list/get -o json`，逐项核对共享盘结果；本次未 submit/cancel，没有停止或重启其他会话的看护进程。机器可见范围内，A0–A4 合成 20-step 已全部 Success；新的 R00–R11 research 方案尚未查到实际提交或云端运行产物。

证据快照：[audit.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_queue012_results_20260930/audit.json)。其中保存精简平台字段、原始 summary/COMPLETE、SHA256、源码一致性核验与数据证书存在性。该文件不包含凭证、环境变量或训练权重。

## 1. 平台终态与时间

以下 queue012 任务均为 **Preemptible=true**、1×`ml.pni2.3xlarge`、队列 `q-20260524172355-rnqtf`。这是已经提交的闲时任务的事实；用户新提出的八卡长期实验池应单独使用 `Preemptible=false`，不能把历史任务写成非闲时。

| 实验 | task_id | 终态 | LaunchTime（UTC） | End（UTC） | 平台 Elapsed 秒 |
|---|---|---|---|---|---:|
| A0 合成 20-step，重提后 | `t-20261001025258-6gndd` | Success | 18:53:29 | 18:56:01 | 182 |
| A1 合成 20-step | `t-20261001024401-qxxcr` | Success | 18:44:35 | 18:45:53 | 111 |
| A2 合成 20-step | `t-20261001024405-9pzqd` | Success | 18:44:42 | 18:45:24 | 78 |
| A3 合成 20-step，重提后 | `t-20261001025254-2m5f5` | Success | 18:53:26 | 18:55:56 | 181 |
| A4 合成 20-step | `t-20261001024357-44ks8` | Success | 18:45:05 | 18:51:08 | 431 |
| A0 原任务 | `t-20261001024414-rvwzz` | Killed | 18:44:46 | 18:46:55 | 161 |
| A3 原任务 | `t-20261001024409-874wl` | Killed | 18:44:47 | 18:46:56 | 166 |
| A0–A4 显存 profile | `t-20261001020115-zvt6f` | Success | 18:01:55 | 18:16:52 | 936 |

所有时间均为 2026-09-30 UTC。平台 Elapsed 从提交/创建起算，包含启动等待；不是纯训练耗时。原 A0/A3 为 Killed，不能因其 summary 仍留 `status=running` 而认为还在跑。平台本轮返回的 ExitCode 不能取代终态判断。

另核实 c20250601 的旧 profile `t-20260930225817-8r8sx` 已 Killed，End=`2026-09-30T18:32:04Z`，Elapsed=12,826 秒、LaunchTime 为空；没有启动 GPU 计算。`immune-v3` 的非终态列表本轮为空。

## 2. 五个合成 20-step 的实际结果

这些实验反复输入同一条人工双链记录，seed=42、global batch=256、学习率=1e-4、20 个 optimizer steps。`COMPLETE.json` 五份均为 completed，summary 的轨迹均到 step 20，loss_decreased=true，biological_quality_claim=false；记录的十个生产源文件 SHA256 均与当前工作区匹配。这里的结果既不是 1M 语料训练，也不是下游测试成绩。

| 变体 | 初始 → 最终总 loss | 最终 AA loss | AA accuracy | end accuracy | micro-bs × accum | 峰值 GiB | 脚本计时秒 |
|---|---|---:|---:|---:|---|---:|---:|
| A0 | 14.093841 → 0.414990 | 6.015907 | 0 | 1 | 16 × 16 | 12.69 | 126.17 |
| A1 | 5.837228 → 0.328653 | 4.321488 | 0 | 1 | 256 × 1 | 7.36 | 13.29 |
| A2 | 5.164647 → 0.252067 | 3.528610 | 0 | 1 | 256 × 1 | 7.36 | 11.74 |
| A3 | 5.169685 → 0.185864 | 2.147263 | 0.461538 | 1 | 32 × 8 | 2.45 | 126.82 |
| A4 | 4.774529 → 1.119085 | 0.959949 | 0.692308 | 0 | 32 × 8 | 4.24 | 324.78 |

A0–A3 的本轮输入 actual_length=310，A4=516。A0 约 270M 参数，其余约 76M；词表、布局、结束标记和损失目标有差异。该表不能按总 loss 排架构名次，也不能把计时差异全部归因于某个组件。峰值显存占用也不等于 GPU 利用率；本轮产物没有 GPU utilization 时间序列。

有用的诊断是：A0/A1/A2 的总 loss 很低，但 AA accuracy 仍为 0，说明单看总 loss 会掩盖氨基酸恢复问题；A4 的 AA accuracy 较高，但 end accuracy 仍为 0，20 步尚不足以证明可靠结束行为。A3/A4 适合继续验证，当前证据不足以敲定最终架构。A0 也是从头训练的对照，不是旧 ESMCv5 checkpoint，因此不能把该表表述成与 ESMCv5 的效果比较。

成功原始结果：

- [A0 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v3_explore_queue012/synthetic20/A0/20260930T185327165633657/summary.json)
- [A1 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v3_explore_queue012/synthetic20/A1/20260930T184433096992764/summary.json)
- [A2 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v3_explore_queue012/synthetic20/A2/20260930T184440213469520/summary.json)
- [A3 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v3_explore_queue012/synthetic20/A3/20260930T185324186289793/summary.json)
- [A4 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v3_explore_queue012/synthetic20/A4/20260930T184504148507558/summary.json)

## 3. 已有显存基线仍成立

profile 的 COMPLETE=passed，summary SHA256=`d55a8eeb7b36f8307a4d1498b16893e6f79db3458d23859255d6250ffd9e148e`，本次重新计算一致。设备为 NVIDIA A100-SXM4-80GB；配置 max_length=2048，执行序列为相应布局实际长度。每档三次优化，仅用于显存和数值可运行性检查。

| 变体 | 推荐 micro-bs | accumulation（单卡 global=256） | 推荐档峰值 GiB | 首个 OOM 档 |
|---|---:|---:|---:|---:|
| A0 | 16 | 16 | 58.24 | 32 |
| A1 | 256 | 1 | 43.59 | 未出现，扫至 256 |
| A2 | 256 | 1 | 43.59 | 未出现，扫至 256 |
| A3 | 32 | 8 | 41.80 | 64 |
| A4 | 32 | 8 | 41.76 | 64 |

扫档 OOM 是预期停止条件，未导致该任务 Failed。新 R00–R11 修改了方法/架构，需要各自重新 profile，不能直接沿用 A4 显存档位。证据：[profile summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v3_profile_20261001_queue012/summary.json)。

## 4. 新 research 与真实数据状态

本轮按 `immune-research` 查询所有八类平台状态，返回空列表；`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_queue012` 没有云端结果。结论限于当前 CLI 可见范围和既定命名，不能推断其他账号、其他命名的任务绝对不存在。

R00–R11 的 36 个 YAML 已准备，初筛 1,000 步、优选后累计 5,000 步的预算与固定验证协议记录在 [research 说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_queue012/README.md) 和 [架构研究记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_FOUNDATION_ARCHITECTURE_RESEARCH.md)。目前不能报告这些方案的 GPU 质量结论。

本轮检查 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1` 仍同时缺少 audit_certificate.json 与 downstream_scope_certificate.json。已有合成试验通过不意味着数据去重门禁通过。新八卡环境可以先进行 profile/合成工程验证；真实训练继续执行已有 assert_ready 门禁，待对应清洁数据完成审计后运行。

## 5. 产物保留与可清理范围

已检查以下五个相关输出目录，没有 `.pt/.pth/.bin/.safetensors` 权重或 checkpoint。合成诊断本来就不保存模型权重，不存在大量 smoke checkpoint 堆积：

| 目录（均在项目 output 下） | 本次文件总字节数 |
|---|---:|
| immune_v3_explore_queue012 | 68,203 |
| immune_v3_profile_20261001_queue012 | 101,818 |
| immune_research_local_20260930 | 16,639 |
| immune_research_local_worker_env_20260930 | 10,549 |
| immune_research_budget_update_20260930 | 102,716 |

大小为本次读取时的文件逻辑大小；看护日志后续可能继续增加。成功 summary、COMPLETE、profile 与精简日志是结果和批量配置依据，保留它们成本很小。

原 Killed A0/A3 的不完整结果只剩约 2.9 KB/份 summary，已包含在本次 audit.json 快照中；如主执行线程按用户要求清理，可以删除以下两个目录，无需保留重复的半成品：

- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v3_explore_queue012/synthetic20/A0/20260930T184444299189906`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v3_explore_queue012/synthetic20/A3/20260930T184445511213754`

本报告任务本身只做核查、归档与清理建议，没有删除文件。未扫描或删除旧正式训练、下游特征、生产 checkpoint，也没有删除看护 PID/ledger。后续真实 pilot 的 checkpoint 需按最新训练恢复/优选保留策略清理，不能依据本次 synthetic 核查推断其已无用途。

本轮后续清理：上列两份Killed半成品目录已按用户要求删除，原summary保存在audit.json；同时删除本地重复生成的YAML测试fixture。删除清单：[cleanup](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/cleanup_20260930.json)。成功结果、profile和日志保留。
