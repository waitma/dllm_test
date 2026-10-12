# 22项统一大batch消融的有限只读监督

本记录仅针对2026-10-03 03:09:26 UTC主会话发布的新study `v7_context_factorial_120k_large_batch_v1`，不混入旧p02八项或人工容量profile的成绩。账本 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/jobs.json` 的SHA256为 `987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d`；统一batch选择 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/batch_selection_v1.json` 的SHA256为 `42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04`，实际读取均匹配。

观察范围是固定22个job的pool/worker、progress、run_provenance、curve、summary与日志metadata；约每55秒读取一次，最晚03:49:26 UTC停止，或首批八项产生共同250步完整评估时提前报告。本观察不提交、取消、重启、重试或改源码/队列/门禁，不运行绘图/模型/测试或读取真实数据行；`latest.pt`只确认存在，绝不加载、散列或清理其内容。停止的序列注释分支和独立CUDA专项暂停保持。核心项目MD由主会话统一同步。

## 实际启动与初始进展

03:11:01 UTC实际新组8 running/14 pending，全池历史results115。首批身份与设备映射如下，各外层PID由原worker的poll-gated heartbeat佐证，不能把alias编号当作GPU编号。

| 条件 | seed | GPU | 外层PID |
| --- | ---: | ---: | ---: |
| f000 | 42 | 0 | 753969 |
| f000 | 43 | 4 | 753966 |
| f010 | 42 | 5 | 753965 |
| f010 | 43 | 2 | 753977 |
| f001 | 42 | 3 | 754007 |
| f001 | 43 | 7 | 754008 |
| f011 | 42 | 1 | 754013 |
| f011 | 43 | 6 | 754012 |

03:14:24 UTC八项均实际发布step25、records800，累计训练206.711–232.386秒、峰allocated14.348–21.416GiB；每项当前完成的评估曲线只有step0，不是250步，更不是3750步终态。八项训练恢复`latest.pt`均存在且保留，没有本组失败或COMPLETE。其余14项尚未开始，指标保持缺失。

八份新provenance均记录micro32×GA1/global32、evalmicro4、horizon3750、120000目标训练记录、同一新study、同一数据manifest SHA `1efa27ea19d10bc94143c3977c80dba8c3a29bad921c093ca7f24447635b9478` 与上述batch-plan引用。source-map规范化metadata摘要八项相同：`78650012e2d9d4356abc56c88377b918d57f7d601c5ca9127af8a2f5fda9217c`。同seed共同初始化在四条件间分别一致：seed42为 `44ff26ac8144ece78c84d9660e2d40d8f0746fa4e555a9ae2afeaadbd7d4b6a2`，seed43为 `783464ece77f32a21af3ce08c3d4dd5d096aee4c8f8eae9940d1cbf6b6f41af6`。

八份metadata的context_sequence_loss和negative_pair_construction均为false，input_preflight引用SHA均为 `21c640f870284cf7f09ebf8766db63ec7e7d311b8e94e0f2567798f53474a5f6`。这里只核已有引用一致性，没有重新执行审核、读取数据行或把metadata核对变成新的数据准入证明；formal_training=false、generation=false。

初始完整快照及各输入原件SHA：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/readonly_watch_v1/observation_20261003T031424Z.json`。后续快照均保存在同一个`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/readonly_watch_v1`目录，不覆盖历史CAP或p02记录。报告实际训练耗时和速度时不默认与旧micro4×GA8梯度、mask流或padding开销相等。

03:25:31 UTC，首批八项都已实际发布训练step100、records3200，累计train_seconds为827.904–891.384秒，峰allocated16.957–22.695GiB。八worker心跳距读取0.21–1.93秒，新组仍8 running/14 pending、无失败或终态。完整评估曲线仍仅step0，不能把progress100当作250评估或完整训练结果。此阶段原件：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/readonly_watch_v1/observation_20261003T032531Z.json`。共同250到来时将用实际train_seconds减去已保留的25步值；原日志若无独立checkpoint耗时，不由文件mtime差补造该计时。

## 首批八项共同250步：本轮观察完成

2026-10-03 03:46:50 UTC，f000/f010/f001/f011各seed42、43的真实curve均已完整发布step250评估，共同评估点为[0,250]。当前本组仍8 running/14 pending，八worker心跳距读取0.84–1.80秒；没有3750步COMPLETE或本组失败。每项实际曝光8000条、unique_train_records=8000、subset_train_records=120000。这允许绘制四条件×两seed的早期partial结果，不能填补另14项、完整八格因素矩阵或将250步当作终点。本agent未运行绘图。

| 条件 | seed | 25→250净训练秒 | 250评估秒 | 250累计训练秒 | 本次观测worker墙钟秒 | 峰allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f000 | 42 | 1943.133 | 13.970 | 2175.045 | 2242.033 | 17.945 |
| f000 | 43 | 1907.652 | 14.351 | 2114.364 | 2242.129 | 16.957 |
| f010 | 42 | 1942.978 | 13.877 | 2174.865 | 2242.274 | 17.945 |
| f010 | 43 | 1906.895 | 14.067 | 2113.626 | 2241.820 | 16.957 |
| f001 | 42 | 1946.687 | 14.194 | 2179.073 | 2239.031 | 22.695 |
| f001 | 43 | 1909.081 | 14.314 | 2116.124 | 2238.924 | 21.416 |
| f011 | 42 | 1944.520 | 14.506 | 2176.665 | 2238.674 | 22.695 |
| f011 | 43 | 1909.233 | 14.360 | 2116.258 | 2238.717 | 21.416 |

25→250是各项实际train_seconds之差，均覆盖225次更新、7200条新增曝光；本观察沿用runner原字段，未据此重新拆分评估或保存耗时。worker墙钟从真实worker.started到共同metadata快照，包含各自启动、初始评估、训练及其余开销，也可能包含250后已继续的时间，不能作为250完成的精确耗时。seed42四条件的净训练均值为1944.330秒，seed43为1908.215秒。

八个真实恢复latest.pt均仍存在并保留；收尾仅记录路径、size和mtime，不读/散列/加载内容，不证明checkpoint步数或恢复正确。当前日志未单独记录保存耗时，故报告将其保留为null，不用mtime构造耗时。

完整收尾原件：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/readonly_watch_v1/first_common250_observation_20261003T034650Z.json`，SHA256 `4b96a4cb77324f887045a4aa331f939bb23761a36d250a78bfb7f296304ae08e`。其内绑定25步基准快照、250步快照、账本、batch plan、发布与入队原件SHA，并保留各job原metadata输入摘要。

本轮观察在03:46:50共同250节点自然结束（早于03:49:26上限）；没有停止、重启、重复提交或修改任何任务。训练继续，正式八卡准入、数据审核、安全停线和独立CUDA暂停均未变。

## 下一共同500步的有界观察

主会话授权继续观察相同22个固定ID，本轮从2026-10-03 04:03:17 UTC开始，最晚05:03:17 UTC停止，或八项真实curve与summary共同完成500评估、或出现明确失败/异常时提前报告结束；单次等待不超过45秒。不会重复250验收、绘图或创建OS监听。

04:04:58 UTC首批八项均实际发布训练step375，已完成评估仍为[0,250]；本组8 running/14 pending，无COMPLETE或失败。worker心跳距读取0.26–1.79秒，保留原设备和PID。原件 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/readonly_watch_v1/observation_common500_20261003T040458Z.json`，SHA256 `61731cc1c749070fd858f294b0ea12da76cbf05e146e8d57b1f839d1982772b2`。500评估此时尚未完整，375训练进度不能替代评估结果或3750步终态。

## 04:24:25 UTC共同500评估：本轮观察完成

实际八份curve均含完整validation500，summary均为500，共同完成评估点[0,250,500]；首批八项每项实际records/record_exposures/unique_train_records=16000、subset_train_records=120000。仍8 running/14 pending，无本组失败或3750步COMPLETE；worker心跳距读取0.76–1.48秒。这里是四条件×两seed的中途8/22 partial，不能称完整八格因素矩阵、完整3750周期、正式八卡训练或全22项最终结论。

| 条件 | seed | 250→500 train_seconds差 | 500评估秒 | 500累计训练秒 | 峰allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| f000 | 42 | 2247.504 | 14.013 | 4422.549 | 17.945 |
| f000 | 43 | 2202.531 | 14.288 | 4316.894 | 17.945 |
| f010 | 42 | 2247.450 | 13.929 | 4422.315 | 17.945 |
| f010 | 43 | 2201.387 | 14.086 | 4315.013 | 17.945 |
| f001 | 42 | 2249.917 | 14.302 | 4428.990 | 22.695 |
| f001 | 43 | 2203.879 | 14.187 | 4320.003 | 22.695 |
| f011 | 42 | 2249.147 | 14.495 | 4425.812 | 22.695 |
| f011 | 43 | 2204.425 | 14.284 | 4320.682 | 22.695 |

250→500的原train_seconds差在seed42四条件均值为2248.505秒，seed43为2203.055秒；每项增加250次更新、8000条曝光。沿用实际runner字段，不据此重拆评估或checkpoint保存耗时。实际worker.started到共同快照墙钟、每项输入token差和吞吐均保留在收尾JSON。

八个latest.pt均仍存在且保留。仅在共同评估快照之后读取stat路径/size/mtime，没有加载或散列其内容；不能据此证明文件已经保存到500步，也未执行恢复验证。独立保存耗时没有原字段，继续保留null。

共同500原始快照：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/readonly_watch_v1/observation_common500_20261003T042425Z.json`，SHA256 `5f8926d2961d1b3d8863fcebfa2a7bbf4fa6448a113749a7ade396156ea9c695`。收尾原件：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/readonly_watch_v1/first_common500_observation_20261003T042425Z.json`，SHA256 `8ce69d9467587459d2bde40e53a54f9eac31f797e0524fbc88f9f3c43420d4f1`。

本轮在实际共同500节点自然结束，未达到一小时上限；没有作图、重复250验收、停止/重启/重试/提交任务，未改训练来源、配置、队列或审核门禁。后续图与核心MD由主会话处理，训练任务继续。

## 下一共同1000步的有界观察

沿用原22个固定ID，于2026-10-03 04:50:32 UTC开始新一轮只读观察，最晚06:20:32 UTC停止。共同750评估只记录并通知一次；共同1000须首八项curve含完整validation且summary至少1000。明确异常或终态也会提前收尾，不将观察截止当作任务失败。单次等待不超过45秒，没有OS监听、模型/CUDA执行或任务操作。

04:52:16 UTC实际仍8 running/14 pending：seed42四项训练progress675，seed43四项700；完整评估仍[0,250,500]，750尚未齐。原worker PID保持，心跳距读取0.53–1.24秒，无失败或COMPLETE。原件 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/readonly_watch_v1/observation_common1000_20261003T045216Z.json`，SHA256 `668d29cd190017f272c9cb72e4b010383f3c590fd16ba4d9af0d21936f3fe72f`。本轮不重复500验收或绘图，保留实际latest.pt且绝不读取/散列其内容。独立恢复脚本的新文件不视为已发布或正在执行的任务。

05:00:23.193 UTC，首八项curve均已发布完整validation750，summary均750，共同评估点[0,250,500,750]。本组仍8 running/14 pending，累计train_seconds6473.260–6554.559秒，worker心跳距读取1.14–1.81秒，没有失败、异常或COMPLETE。中间750节点只通知主会话一次，不作图、不重新执行验收，继续原定共同1000目标与06:20:32截止。原件 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/readonly_watch_v1/observation_common1000_20261003T050023Z.json`，SHA256 `7453e88a49b7a72c862ab1b333fddcf0401a343bcd6284fb5269fab8e66dcada`。这仍是8/22中途结果，未完成3750周期。
