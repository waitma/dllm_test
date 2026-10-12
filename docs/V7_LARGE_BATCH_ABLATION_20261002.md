# V7 independent large-batch ablation — 2026-10-02

## 2026-10-03T09:09:20Z 用户指定测试不保留 checkpoint

本轮架构测试的模型/恢复文件已按用户要求全部清理，指标、曲线、来源、日志及图保留。当前 frozen runner 不改，后台清理每 10 秒处理新发布的 checkpoint，活跃 .tmp 写入完成前不删除；新公共测试入口默认只写指标。旧段落中的“latest.pt 保留”和测试恢复计划为当时历史，不再代表当前策略；没有模型文件时不得声称能恢复这些测试。动态状态：[清理状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_checkpoint_metrics_only_20261003/state.json)，[政策与初次清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_checkpoint_metrics_only_20261003/policy.json)。6 项控制和两步人工 CPU 入口通过，未新增或重启 GPU 任务。

## 2026-10-03T04:32:08.720556+00:00 新共同500实际图已归档，组合收益收窄与恢复入口缺口明确

2026-10-03T04:24:25.624UTC，首8的真实curve/summary均完成500步、每项16000/120000条唯一曝光（13.33%）；仍8running/14pending，无失败或3750终态。统一micro32×GA1/global32、eval4及原完整22范围保持。

| 条件 | 两seed平均AA CE | 相对R20降低 | 全mask AA CE | 全mask AA准确率 | context子集AA CE |
|---|---:|---:|---:|---:|---:|
| R20/f000 | 1.179252 | 0.00% | 1.790015 | 41.13% | 1.999002 |
| R24/f010 | 1.105767 | 6.23% | 1.799239 | 40.62% | 1.976865 |
| R26/f001 | 1.063047 | 9.85% | 1.831919 | 40.13% | 1.962080 |
| R24+R26/f011 | 1.043684 | 11.50% | 1.837388 | 39.73% | 1.969006 |

f011两seed汇总AA CE均低于匹配R20，均值优势由250的20.58%收窄到500的11.50%；f001为9.85%、f010为6.23%。全mask AA准确率当前R20最高，f011低1.40个百分点，且全mask CE更高；context子集三候选均值略低，但仅37条正例验证。不能将部分遮蔽收益解释为生成或多链理解已改善，不凭早期汇总选正式模型。两seed无显著性结论，仍8/22 partial/final=false，旧4×8协议独立。

[500概览图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared500_20261003/v7_large_batch_ablation_overview.png)；原报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared500_20261003/report.json（SHAd8d1ae3b9c8f6e33592f5c7478a494ad944826fe61ed4b579878054ef0b7ac94）；[12图及实际渲染归档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/render500_20261003.json)（SHA9d19428788fad9579a9196c1c0053a071ab296c99ad63184013869d9b50a1af4）；[两seed派生](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/current_method_comparison_shared500_20261003.json)（SHAcf0571f33cbb9358ae15e7fda91ef744e1496e0dff7c626d2842bdc5a4b6d0a4）。250独立归档保留，500真实endpoint只渲染一次。

250→500实测8.806–9.000秒/步，评估13.929–14.495秒。直接get同一资源真实Running，原始查询 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/resource_budget_v1/platform_after_shared250.raw.json；按该Elapsed在04:24:25的pool剩余proxy约14.98小时。以首8代表未启动14项，剩余三波纯训练预测26.29–26.87小时，仍不计验证、保存或正式；新预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/resource_budget_v1/budget_review_common500.json（SHAee127987daf70ee63acda7475dd2b7fcf66590b9982d82160cc4e15d02596ea3）。原范围/队列/配方未改，不因预算不足假称完成。

只读审阅确认：原冻结核心已有--resume，恢复模型/Adam/scheduler/CPU-CUDA RNG、epoch/position和累计计数；现有pool启动脚本却不能传递到此入口，且outer新attempt必须有自己的COMPLETE。正在准备独立通用接续模块，保留原账本、完整来源/plan/capacity/data检查和同cell持锁；实际票据必须等原job terminal及checkpoint保存边界核实后创建。本回合未读取或散列实际latest.pt、未恢复模型，不能将存在文件当恢复通过。原保存替换latest→追加curve→summary，若硬停致末步不一致，原恢复拒绝保持，不补写证据绕过。

原Goal身份/累计用量保持active；上一回合分类progress（250真实图/原件归档/比较派生与MD同步），本回合500实际新证据改变选型判读并核实必要接续缺口。正式八卡联合更新/原生恢复未启动或通过，final V7限定范围ready/双证书及全部选择/来源/恢复门禁保持；停注释及独立CUDA专项保持暂停。无submit/cancel、资源/worker/监听重启或重复登记；实际恢复文件保留。

500观察已自然结束，GPU任务继续。下方其他日期段是历史。

## 2026-10-03T04:02:11.291493+00:00 当前共同250，四条件×两seed实际图已归档

2026-10-03T03:59:36.506698+00:00只读核实：新组8 running/14 pending，首8均325/3750步；实际共同完整评估为250，每项8000/120000条曝光（6.67%）。统一micro32×GA1/global32、eval4及原3750完整epoch目标保持，尚无新组终态。

| 条件 | 两seed平均AA CE | 相对R20降低 | 全mask AA准确率 | context子集AA CE |
|---|---:|---:|---:|---:|
| R20基线/f000 | 1.565687 | 0.00% | 40.58% | 1.997667 |
| R24/f010 | 1.449769 | 7.40% | 41.04% | 1.972678 |
| R26/f001 | 1.269924 | 18.89% | 40.59% | 2.064495 |
| R24+R26/f011 | 1.243442 | 20.58% | 40.51% | 2.118483 |

这是新组四条件×两seed的早期重建结果，8/22 partial、final=false；另14项仍待运行。组合f011的汇总AA CE最低，但全mask准确率未显示同步改善，且仅37条context验证上的AA CE更高；不能认定生成、多链理解、二分类或生物功能提升，不能据250步选正式模型。两个seed不构成显著性证据；旧micro4×GA8结果独立保留。

真实250报告及12图已逐字节独立留存 [/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/report.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/report.json)；[实际概览图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/v7_large_batch_ablation_overview.png)；[渲染/归档记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/render250_20261003.json)（SHA41f067ee719394b508be77c97080b01e06bc367a7d203cb987672e5a1ad981f3）；[两seed派生](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/current_method_comparison_shared250_20261003.json)（SHA514bf2eba5a2ac69603e08e31373a10421690ba740451b260fc5851441863965）。未重绘、未重复模型验证或读取二进制checkpoint。

实际25→250净训练8.475–8.652秒/步；250评估13.877–14.506秒、当时worker墙钟2238.7–2242.3秒。若未启动14项沿用首8代表速度，剩余三波仅训练约25.90–26.44小时；03:46:50按既有platform elapsed估计资源余15.61小时，本次未重新查询平台，正式准入仍须核实际pool lifetime。独立保存耗时缺失，保留null，checkpoint stat不证明保存步数或恢复。新的[250预算原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/resource_budget_v1/budget_review_common250.json)（SHAc11b683760ebf34c7d90c33e68d04e5a65d19f915ace6b84db54527cfce6b72e）。原22范围与顺序保持，不静默撤下、缩短、改batch或重放。

下方其他各带日期段为历史状态；本节和PROJECT顶部为当前依据。

## 2026-10-03T03:14:48.193324+00:00 最新真实状态：完整22项已登记，首八项已在原八卡运行

2026-10-03T03:10:52.992945+00:00为8 running/14 pending，首批R20/R24/R26及组合各两seed；统一micro32×GA1/global32、eval4、原120k/3750完整epoch。实际计划与固定账本已存在，尚无新组训练结论或完整图。发布/登记/初始原件依次 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v2/publication.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v2/enqueue_receipt.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v2/initial_observation.json；新启动源清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/launcher_source_v2.json，原568模型不改。有限只读监督的后续记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LARGE_BATCH_RUNTIME_WATCH_20261003.md。正式八卡联合训练/恢复未启动，原Goal active、数据/来源/恢复门禁与全部停线保留。以下各带日期段是历史状态，当前以本节、PROJECT顶部和CURRENT为准。

当前更新2026-10-03T02:27:52Z：旧p04容量首格非OOM超时已永久保留；独立v2 job1790992416352344471_p05_v7_capacity_v2_a4854542单次在原GPU2执行，九格实际passed/rc0，三个micro32各有≥15%余量。原568模型/profiler不改，v2只测2warmup+2measure/格、36更新、单格240/总仍1800；详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_CAPACITY_TIMEOUT_REVISION_20261003.md。下方旧8更新消费者为历史未部署协议，不能以v2原件冒充通过；下一节点需明确新协议消费者/实际plan/完整22账本再发布。此时新batch plan/jobs尚不存在，新22项/正式组均未启动，原数据审核与安全门禁保持。

绘图接续更新：新独立绘图入口已通过一次5项人工metadata/Matplotlib检查，根端核对实际源码与stdout SHA，并以原1000步图的源码SHA证明旧plotter只有一行可选标题扩展。绘图两文件已单独冻结到 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/analysis_source_v1；清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/analysis_source_v1.json，SHA256 `0e9634f9398a46c08fb6f4a30500fa5b608ec00dd401cdb8c21bdc8176368b0d`。这是metadata绘图源，不修改模型568文件清单；新实际训练曲线尚未出现，人工图件已清理，不将它们当实验图。

当前状态（2026-10-02T22:52:04Z）：源码、11 份 existing-pool 实验清单、四项新参数控制和一次两步 CPU 人工入口检查已完成；568 文件已独立冻结并逐项核对 SHA。一次独立 synthetic 容量测量已接入并登记到原池，但仍 pending，GPU 容量/吞吐尚未测量，统一 microbatch 未选定；22 项新训练尚未发布或入队。容量账本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_job.json。原 p02 八项继续原配置，十四项未启动任务已撤下并保留原件；撤下记录位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/batch_revision_20261002/withdrawal_receipt.json。

冻结目录：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/source_snapshot_v1。清单：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/source_snapshot_v1.json，SHA256 `2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783`。包含原 p02 冻结的 566 份源以及新增独立入口和容量测量脚本，原 p02 源保持不变。根端只读核对记录：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/project_status_sync_20261002/20261002T223958Z.json。

入口：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_large_batch_ablation.py。清单目录：/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_large_batch_ablation_20261002。每份 YAML 对应一个条件、seed42/43；它们不是 volc 资源申请配置，不能作为资源提交命令的 conf。

## 独立协议

study 为 `v7_context_factorial_120k_large_batch_v1`。完整 11 条件 × 两 seed，统一选择 micro16×GA2 或 micro32×GA1；global32、120000 train、1024 valid、3750 updates、warmup100、LR1e-4、eval250 均沿用原 core。验证单独固定 micro4，四档按记录固定的 mask 和来源/context/关系计数复用既有 evaluator。

训练保留 `existing_micro_mean`：每个物理 microbatch 内 selected-token mean，再按 GA 等权累积；没有新增 loss。训练 mask seed 仍为 `seed*1000003+(step-1)*grad_accum+physical_micro_index`。改变 GA、batch 和 padding 形状会改变逐记录随机 mask 与损失权重，所以不与 p02 4×8 声称梯度或 mask 相同，也不能把两组拼成单一 factorial 矩阵。固定记录顺序、global32 与3750步仍保留一遍120k记录的目标。

只复用原 FactorialModel、原 core 和原 context evaluator；模型结构、既有正例 relation objective、context_sequence_loss=false、negative_pair_construction=false 保持。原 parent-ready、双证书、subset SHA、context input preflight 检查均不删除。没有审核/注释逻辑实现或运行，没有真实数据读取。

## CLI 与共同档位

必须显式给出 `--variant --seed --micro-batch --grad-accum --output-dir`。非 CPU 运行另需 `--data-dir --batch-plan --batch-plan-sha256`；默认 `--steps 3750 --eval-every 250`。允许同协议未完成 checkpoint 使用 `--resume`，先根据 JSON provenance 拒绝旧 p02 / 不同档位，再由 core 严格比较完整 provenance、来源、模型和日程；不允许修改旧 checkpoint 跨协议恢复。

共同 batch-plan 是 profile 完成后由主会话实际选择的 JSON，当前不存在。schema=`v7.large_batch_selection.v1`，study 为本项，status=`selected`，固定字段：micro_batch、grad_accum、global_batch=32、evaluation_micro_batch=4、sample_manifest_sha256、variants（与入口顺序一致的11项）、seeds=[42,43]、profile_evidence（非空实际绝对 path/SHA 列表）。入口校验计划 SHA 与每个证据文件实际 SHA，把全部引用写入 study provenance。该检查绑定实际选择记录，并不把任意被引用 JSON 当成 GPU 验收通过；profile 内容的真实核查、共同档位批准和来源冻结仍由主会话完成。

YAML 当前 micro_batch/grad_accum/batch_plan/memory_peak_gib/source_freeze_sha256 为 null，profile_evidence 为空，status=profile_pending、queued=false。这表示未确定档位，并不是可执行的已通过配置。

## 本地验收边界

新增四个标准库参数控制：两档组合和非法参数拒绝、CPU人工边界、固定验证micro4、跨协议/跨档恢复拒绝。它们不 import 模型、读取真实数据或写 profile 通过件。

CPU 人工入口命令：

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_large_batch_ablation.py --self-test
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_large_batch_ablation.py --variant f111 --seed 42 --micro-batch 16 --grad-accum 2 --cpu-smoke --steps 2 --output-dir /absolute/new_cpu_check
```

CPU 人工检查只验证新入口连接和记录口径，不是实际 75M/GPU 显存、BF16、真实数据效果或完整3750步验收。临时 CPU checkpoint 在保留 JSON/日志证据后删除。

## 实际新增验收

2026-10-02：在 pllm 的 /vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python、Torch 2.8.0+cu126 中各执行一次，CUDA_VISIBLE_DEVICES 为空，每命令限180秒。四个新标准库控制4/4通过。新 f111/micro16/GA2 人工入口返回0，实际优化2步、曝光64行、模型24680参数，验证仍采用micro4，累计训练段0.203368秒。micro32/GA1只通过参数组合约束，本次未运行该档模型，更未验证GPU容量。

入口源码 SHA256：`0ca5f4694eec011baa7b87e00d68296f07af7de40876813fd4cd07862947b25c`。未改公共 core，其实际 SHA仍为 `81266e9b51a35b284a10e84f72b586a1987269368732cf97b3086d38f691eb10`，与p02冻结core一致。

原件：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/parameter_contract.stdout.log、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/cpu_entry.stdout.log、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/cpu_entry_evidence.json。后者绑定实际命令、解释器、源码、11份YAML与CPU JSON/日志SHA；run_provenance、summary、COMPLETE、curve保留。仅本次自有CPU latest.pt（342667字节）已删除，SHA与清理路径记录于该证据，剩余临时checkpoint为0。没有执行旧套件、GPU profile、真实数据、平台或pool任务操作。

## 独立大 batch 消融绘图入口（2026-10-02）

新增 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_large_batch_ablation.py，只读 JSON/JSONL 元数据。复用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_context_ablation.py 的分母聚合校验、validate_points、derive、write_csv 和四页 draw；旧脚本仅将图标题扩展为 report.get('figure_title', 原 p02 标题)，旧 identity/collect/study 和默认输出不变。本项源码未加入模型冻结源清单，未发布或入队。

运行接口（实际 selected plan 和独立 jobs 账本存在后使用；下列 /absolute 路径为占位）：

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_large_batch_ablation.py --jobs-file /absolute/jobs.json --pool-dir /absolute/pool --batch-plan /absolute/selection.json --batch-plan-sha256 HEX64 --output-prefix /absolute/figures/large_batch
```

账本顶层必须含 study=`v7_context_factorial_120k_large_batch_v1`、batch_plan={path,sha256}，与显式 CLI 完全一致。jobs 各项沿用 variant/seed/job_id/alias/job；job.variant 必须等于 alias，alias 以 `_v7_{variant}_seed{seed}` 结尾且不能为旧 p02_，不预设新队列编号。每个条件、job ID 和 alias 唯一，job 为 train/3750/绝对 data_dir。若 pool pending/running 原件存在，必须与账本 job 一致；pool result 必须绑定对应 job_id。

绘图核对 plan 本体 SHA、selected 字段、完整11条件×两seed与固定数据 SHA，以及每个 profile 引用原件 SHA；只认定字节身份，不将任意 profile JSON 当成 GPU 通过证明。各 run 的实际 provenance 必须匹配统一 micro16×GA2 或 micro32×GA1、eval micro4、global32、3750日程、模型/源码/数据、输入preflight和同seed公共初始化；旧 p02 或混合 batch、eval 非4、计划漂移均拒绝。四档验证聚合口径和来源/context/关系分母复用旧校验，不读取序列、模型或 checkpoint。

输出四页 overview/factorial/strata/separate_controls，各 SVG/PDF/PNG；主 CSV、paired/interactions/removal CSV、额外 runs 状态 CSV 和报告 JSON 均绑定 study、micro、GA、global batch、eval micro及plan SHA。共同正步取已有可比较 run 的交集，不齐22格明确 endpoint_scope=partial、all_22_common_step=null；图注明真实参与数/22，缺失格留空，八factorial格不足不计算对应seed的效应。只有22项共同3750、COMPLETE/summary/provenance SHA及各pool成功原件齐备才final。空输入或仅step0不产生数值图，并列出未更新的既有同prefix图件。两seed没有CI/显著性或生物效果声明，正例关系指标不表述为二分类判别。旧14项withdrawn标注仍是旧plotter独立待办，本入口不读入或改写旧组。

新增人工检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_plot_v7_large_batch_ablation.py 已在pllm、CUDA_VISIBLE_DEVICES为空下仅执行一次，5/5通过，耗时6.667秒，限时180秒。范围为：空/初值不出图、部分共同终点、完整22格终点与pool成功原件、study/batch/eval/源码不匹配拒绝、plan/profile SHA与账本/alias漂移拒绝。完整人工矩阵实际生成并检查四页共12个图件及全部CSV字段；原始日志保留人工图件大小/SHA，TemporaryDirectory退出后全部临时原件和图件清理，未创建任何checkpoint。没有模型导入、真实数据读取、GPU运行、真实plan或jobs创建，未重跑旧套件。

真实stdout/stderr原件：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_plot_evidence_20261002/metadata_controls.stdout.log。命令、解释器、结果及源/日志SHA归档：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_plot_evidence_20261002/receipt.json。新增入口SHA256 `868984422b186b13d5ae6a88cd4d3e9b61701f50988585fb1a4dd0374ec08d93`；专用检查SHA256 `9c0c54846a5b41eb09a466cf4453596bcb20f51422bd0edc4847a31ea9ad0d37`。尚无本项真实大batch学习结果，人工fixture不代表GPU容量或实际训练完成。

## 大 batch 消融分派源码准备（2026-10-02）

新增 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_large_batch_ablation_dispatch.py；状态为源码和人工控制完成，未部署、注册alias或入队。公共入口、registry及568份冻结源均未修改。未知alias保留原argv，委派 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_large_batch_dispatch.py，严格核对SHA `f259fb0f0946fb7434c1f78b06f64e5d186ece7b781fd29dac7f3f0fe0e5d866`。

固定22个alias为 `p04_v7_{method}_seed42/43`，method使用本项11条件。未来pool在ML导入前调用 main(original_argv)，也可仅调用 prepare(original_argv) 返回 args、command 做只读核查。只接受原pool的variant、mode=train、steps=3750、data-dir、profile-dir、output-dir各一次，不接受外部micro/GA/plan/resume或额外参数；路径必须为绝对、规范、无符号链接路径。data-dir固定 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/sample_context_plus20k；profile/output固定原pool对应alias/job目录。main要求CUDA_VISIBLE_DEVICES为单个0..7，最后execv冻结runner，保持原slot job PID。prepare不创建输出或进程。

未来实际共同计划固定 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/batch_selection_v1.json；完整账本固定 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/jobs.json。本项未创建这两个实际文件。账本包含study、batch_plan={path,sha256}、source_freeze={path,sha256}及22个显式jobs；每项含alias、variant、seed、预分配job_id和精确job={variant:alias,mode:train,steps:3750,data_dir:固定路径}。先写完整账本后才允许未来发布pending，避免领取时缺账本。公共registry未来需一次发布22个literal条目，每项launcher_only=True、model_variant、seed、study、jobs_catalog_sha256、batch_plan_sha256、source_freeze_sha256完全匹配；新dispatcher仅AST读取literal VARIANTS，不import模型，不修改registry。

source manifest固定SHA `2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783`；每次prepare逐项核对568源和2份CPU证据，并复用该冻结runner的checked_batch_plan、parse_args、validate_parameters。selected plan沿原schema，统一16×2或32×1/global32/eval4；profile_evidence必须包含以下四个固定path/SHA引用，另有引用也须匹配真实文件SHA：

- /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_job.json
- /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790981433321562451_p04_v7_capacity_febb28e3/summary.json
- /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790981433321562451_p04_v7_capacity_febb28e3/COMPLETE.json
- /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790981433321562451_p04_v7_capacity_febb28e3.json

capacity_job是原始入队快照，其pending字段不当作最新状态；真正成功必须来自同一唯一job的pool succeeded/returncode0、精确job与原pool argv。COMPLETE必须绑定summary SHA及training_completed/actual_corpus_test/checkpoint_created三个false。summary必须完成3代表模型×micro4/16/32九格、无重复、状态仅passed或显式容量OOM；核对A10080、2048完整shape、768/8/12/3072/full256/frag64、BF16/whole-layer、AdamW、8更新和6个有限实测loss及已建立optimizer state。实际source_sha256逐项必须属于冻结源并匹配，profile/model实际执行源不可缺。选中micro在三个代表模型均须passed，allocated与reserved分别不超过total的85%；未选中的OOM允许保留。该语义核验拒绝把任意引用JSON当作GPU通过，也不凭profile推断真实语料学习效果。

专用人工控件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_pool_large_batch_ablation_dispatch.py 已在pllm、空CUDA mask下执行一次，5/5通过，耗时6.623秒，限时180秒。覆盖22项正向参数绑定、两档组合、mock execv、缺计划/profile、选档OOM/显存超限/优化步或loss不足/硬件错误、九格重复、源漂移、跨组SHA/seed、输出symlink、pool失败/argv变化、外部覆盖、GPU mask和未知alias委派。控件只在自有synthetic_fakecontrol临时目录写人工metadata和568个小源码fixture，复用冻结runner文本的标准库协议；没有执行任何模型、实际profile、旧suite、真实数据或旧dispatcher main，subprocess/socket被禁止，execv被mock，临时目录全部清理。通过仅是软件边界验收，非实际GPU验收或运行授权。

原始stdout/stderr：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_dispatch_controls_20261002/stdout.log。命令、解释器、原件和SHA归档：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_dispatch_controls_20261002/receipt.json。dispatcher SHA256 `dc3408ff7a8e74b7944b5ceb17ca18bdaf0baf9a226d270c9cb12fe99058aa0f`；控件SHA256 `18f63fdd0d964ef808476eaa8e91f6aff799d2846d9fd4da3cdd1fce9b824351`。后续仍需实际capacity成功证据、主会话选档、完整真实账本与registry发布，再由根端审查接入；本项未进行这些操作。

2026-10-02T23:25:23.615132+00:00 根端已完成上述源码、完整stdout/receipt及逐文件SHA核对，另与实际pool.command和冻结profiler输出字段对照；未重复执行控制。两文件独立冻结清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/launcher_source_v1.json，SHA256 a7cad07e42638134bfb19c6b94d280a24ad36b22de823871bb20c18204fc673f。公共入口/registry/568模型源保持，无真实plan/jobs或训练发布。工程审阅完成，实际容量终态与档位仍待原job，不从fixture或引用SHA推断GPU成功。
