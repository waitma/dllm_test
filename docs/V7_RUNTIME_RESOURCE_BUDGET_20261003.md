# 实际训练耗时与当前资源预算

## 2026-10-03T04:32:08.720556+00:00 当前500实测预算，接续能力必须完整核实

2026-10-03T04:24:25.624UTC，首8的真实curve/summary均完成500步、每项16000/120000条唯一曝光（13.33%）；仍8running/14pending，无失败或3750终态。统一micro32×GA1/global32、eval4及原完整22范围保持。

250→500实测8.806–9.000秒/步，评估13.929–14.495秒。直接get同一资源真实Running，原始查询 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/resource_budget_v1/platform_after_shared250.raw.json；按该Elapsed在04:24:25的pool剩余proxy约14.98小时。以首8代表未启动14项，剩余三波纯训练预测26.29–26.87小时，仍不计验证、保存或正式；新预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/resource_budget_v1/budget_review_common500.json（SHAee127987daf70ee63acda7475dd2b7fcf66590b9982d82160cc4e15d02596ea3）。原范围/队列/配方未改，不因预算不足假称完成。

只读审阅确认：原冻结核心已有--resume，恢复模型/Adam/scheduler/CPU-CUDA RNG、epoch/position和累计计数；现有pool启动脚本却不能传递到此入口，且outer新attempt必须有自己的COMPLETE。正在准备独立通用接续模块，保留原账本、完整来源/plan/capacity/data检查和同cell持锁；实际票据必须等原job terminal及checkpoint保存边界核实后创建。本回合未读取或散列实际latest.pt、未恢复模型，不能将存在文件当恢复通过。原保存替换latest→追加curve→summary，若硬停致末步不一致，原恢复拒绝保持，不补写证据绕过。

保存独立耗时仍null；Checkpoint只stat，恢复步数/实际native状态尚未验证。下面250/175是历史预算。

## 2026-10-03T04:02:11.291493+00:00 当前250复核，预算冲突保持

2026-10-03T03:59:36.506698+00:00只读核实：新组8 running/14 pending，首8均325/3750步；实际共同完整评估为250，每项8000/120000条曝光（6.67%）。统一micro32×GA1/global32、eval4及原3750完整epoch目标保持，尚无新组终态。

实际25→250净训练8.475–8.652秒/步；250评估13.877–14.506秒、当时worker墙钟2238.7–2242.3秒。若未启动14项沿用首8代表速度，剩余三波仅训练约25.90–26.44小时；03:46:50按既有platform elapsed估计资源余15.61小时，本次未重新查询平台，正式准入仍须核实际pool lifetime。独立保存耗时缺失，保留null，checkpoint stat不证明保存步数或恢复。新的[250预算原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/resource_budget_v1/budget_review_common250.json)（SHAc11b683760ebf34c7d90c33e68d04e5a65d19f915ace6b84db54527cfce6b72e）。原22范围与顺序保持，不静默撤下、缩短、改batch或重放。

250原评估和checkpoint metadata见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/readonly_watch_v1/first_common250_observation_20261003T034650Z.json；二进制checkpoint未读，原watch已自然结束，训练继续。下面03:41段为首175步历史预算。

更新 2026-10-03T03:41:39.451832+00:00。原Goal范围保持完整11条件×两seed×120k/3750epoch及后续正式八卡初训/原生恢复。当前是在运行任务的条件预测，不是外部审核阻断，不改变scope或把partial当完成。

2026-10-03T03:41:16.404111+00:00固定新组实际8 running/14 pending，首8为200–200/3750步、6400–6400条记录；完整共同评估点仍[0]。真实原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/readonly_watch_v1/observation_20261003T034116Z.json（SHA102c50223a69406674451099e89a68cd681e2cc061726da4974dad957ba4f14f），没有新组终态、最终架构或完整22图结论。

| 协议 / 时段 | 实际训练秒/更新 | 证据范围 |
|---|---:|---|
| 旧micro4×GA8，完整3750 | 3.676–3.698 | 同120k旧八项实际全epoch，每项worker墙钟14042–14131秒 |
| 新micro32×GA1，首25 | 8.268–9.295 | 800条，仅0.67%epoch，可能有首次更新/形状波动 |
| 新micro32×GA1，25→175 | 8.417–8.573 | 八项实际净150更新；未计验证、保存或启动 |

冻结core计时从取样/组batch开始，到backward、clip、Adam、scheduler及CUDA同步完成。初始和后续eval、checkpoint、进度日志均在计时外，不能从累计train_seconds再减step0验证。旧与新改变micro/GA及训练mask/损失加权，不宣称逐步梯度等价；此处仅比较已有真实运行耗时。未测实际padding后每batch形状，暂不归因具体瓶颈。

预算原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/resource_budget_v1/budget_review.json（SHA9e31330742a3e6ef1cdf35c553e89bb1c970c1c41e6b6d6e49166be4c5c0bbac），绑定原旧完整3750各summary、first25/175和新的实际platform get。记录时原pool估计剩15.75小时；以新净窗口代表尚未运行14项，三波剩余纯训练约25.89–26.38小时，不包括验证/保存/正式两段。未运行条件的速度仍是代表预测，不能伪装直接测量。formal wrapper未来还会按真实pool start_ticks与resource_deadline检查，不以此platform elapsed估计取代原准入。

250阶段只读已有真实产物：(C250-C25)/225逐method/seed；curve/summary250完整评估、evaluation_seconds和checkpoint发布；实际worker started到完成墙钟。保存独立耗时若原日志没有，则明确缺失，不用mtime差补造。同步真实资源剩余；只有预算覆盖原准入和两段保存/恢复/续步/清理才启动正式。

可选顺序问题已提出：完整消融优先、正式接后续资源窗口；或首8完整后当前窗口优先正式、其余消融后续补齐。尚未有新的选择，按原完整22→正式顺序继续，不撤下/改batch/缩短/重放任何任务。若资源始终不足，公开记录仍未完成范围及真实恢复状态；不能将首8变作完整22，不能借时间不足伪造选型或降低审核/来源/恢复门禁。

正式v2启动衔接源码审阅冻结已完成，/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_V2_ENTRY_INTEGRATION_20261003.md；formal计划/八票据尚未创建。停止序列注释、独立CUDA专项暂停，现行limited-scope ready/双证书及完整门禁保持；没有提交、取消、重启或读取二进制checkpoint。
