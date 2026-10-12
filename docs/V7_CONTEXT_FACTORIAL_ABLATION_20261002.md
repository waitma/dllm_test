# V7 context增补后的完整组合与CV对照

2026-10-03T08:52:55Z checkpoint 清理更新：按用户要求，旧 p02 已完成 8 项的 latest.pt 已删除；pool 成功结果、COMPLETE、summary、provenance、curve、日志与终态图保留。已实际核对完整 3750 步终态、summary/provenance SHA 与最后 curve 点一致，绘图不依赖二进制。旧段落中“latest.pt 保留”为当时历史状态；当前不再支持加载或恢复这 8 个旧模型。新 p04 的 11 个 latest.pt 均保留。逐文件证据：[清理记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_checkpoint_cleanup_20261003/cleanup_receipt.json)。本次不改变原消融协议或已记录结果。

更新时间：2026-10-03T01:54:08Z。状态：旧p02原8项3750步完整120000条epoch终态核验通过；22项设计仍partial、14withdrawn-before-start；终态12图已更新。

本阶段复用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/sample_context_plus20k，train120000/valid1024，context正例24648（20.54%）；原valid只有37条context，单独报告分母，不据此夸大关系泛化。全部候选micro4×GA8/global32，3750步完整epoch；warmup100、LR1e-4、固定3750 cosine。8个fGSL格×seed42/43覆盖G门控、S固定0.1残差、L交替半径32局部/global的所有组合；另加入cv_segment_global、cv_late_fusion与fixed_gate_control各两个seed，共22项初筛。原三seed确认仍待完成，两seed图只标初筛；不能把旧100k结果补进新120k矩阵。

模型 /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research_refined/factorial.py 复用旧attention block和缓存denoise，新增分段全长/global仅改变mask；prefix不在同段层偷传，padding keys不开放。24项CPU整数合同首次通过（12.99秒），其中旧6对应项数值一致、组合checkpoint梯度、隔离与保存恢复；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/factorial_architecture_cpu_20261002/pytest.xml。无真实训练/新注释/功能设计。

入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_context_ablation.py 实际f111两步CPU人工检查通过，最后loss3.95745，验证总CE3.93411；只证明入口可执行，不证明学习收益。原公共pilot增加可注入horizon/factory/evaluator，默认p01不变、在跑p01使用旧冻结源。新末态要求records120000且epoch1/position0才COMPLETE；3750前只SEGMENT。仅自有CPU latest.pt已删除，保留曲线/完成标志/清理SHA，原GPU恢复状态保留。

指标 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/context_pilot_metrics.py 在同一次前向分别累积source、有/无context和关系token的numerator/count；2项整数张量测试通过，验证关系与AA/END分离、context不遮蔽、缺关系分母返回None及RNG/训练模式保持。正例关系CE不是正负判别；不输出序列或logits。

YAML目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_context_ablation_20261002 下11份方法配置是已有资源池的实验清单，不是申请新资源的Volc配置，不能用volc submit直接提交。实际入队已完成，由原pool登记22个唯一job ID；不重启原worker、不重复申请容器。机器计划 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/study_plan.json；空白矩阵 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/ablation_matrix.csv。

接续：原卡完成p01后再领取这些3750步任务；实现只读22项矩阵/交互图。正式8rank单模型入口并行整合，正式架构选择和GPU保存恢复仍未完成，不能将八份单卡任务冒充八卡单模型训练。

## 2026-10-02T21:01:39.089377+00:00 22项120k组合/CV实验已一次入原池

p02冻结566份源，复用p01完整源再显式覆写4个已验新/共享文件；冻结清单SHA 0a88b6a1b38b85fdaa9491444286da6b1bd023c721ce4828d7f5ca0b0b0e53ab。22个launch参数检查通过；只更新公共入口指向新分派并追加22个launch-only名，原78个名及p01分派保持。发布前后字节证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/deployment_v1/publication.json。

已实际登记22个唯一job ID到原pool（pending22、原p01 running8）：账本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/jobs.json。先排baseline/原局部global/整段global/延后融合各两seed，再排其余GSL组合与固定门控对照；每项120000条、3750步。只是已入队，尚无p02 GPU初始化/优化步/完成证据，不宣称新方法有效。原卡自然完成p01后领取，不重启或取消在跑任务，不新平台submit。真实平台查询返回同一task Running；按原worker 258600秒预算估计入队时余22.41小时，原始CLI输出带版本提示行，已提取真正JSON记录而未把提示当任务故障。

正式八rank driver已落盘：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_formal.py；分离冻结模型源与真实审核资源，显式支持11组合/CV模型及原4候选、global=micro×GA×8、rank0写共享状态、八rank实际参数SHA与native optimizer/scheduler/RNG完整恢复。新协议CPU9项通过（子agent执行报告；原始日志归档待根端读取），仍未联合GPU运行或接入pool组启动；不将此写成正式训练已启动。接口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_EIGHT_GPU_ENTRY_20261002.md。

## 2026-10-02T21:07:58.922239+00:00 正式入口隔离审核首次真实执行通过，p02作图已登记

冻结正式模型/Trainer与真实审核资源的分离接口已首次执行：原helper在python -I -B、CUDA隐藏的子进程调用原assert_ready，真实校验final V7所有分片及现行scope/evidence，54.28秒返回passed；parent清单与两证书SHA前后不变，实际门禁模块39份。没有创建审核证书或改ready、恢复注释或启动模型。源清单1137项、SHA 2c5a2a8a5ba9841dbffc569f1318a09d2497eed39a9ccc7ba5ed8e99104172b1；事实原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_first_execution.stdout.json 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_first_execution.json。此处是新隔离执行接口的首次实际核验，非重复签证。

root已读取原工具stdout归档，正式driver新增协议测试9/9、3AST通过；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_driver_evidence_20261002/receipt.json。正式入口源已冻结，但launcher还没接通，联合NCCL更新、实参同步、保存恢复实际GPU证据仍缺，formal_training_started=false。正式长程预算需结合原资源剩余时间设置可恢复阶段；不允许因全部小实验排完便把8份独立模型视为world8联合训练成功。

p02作图 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_context_ablation.py 新4项纯数学/结构控制通过；首次读真实22项账本输出JSON及4CSV，awaiting_shared_evaluation、missing22、figures=[]。没有用假效果值画图。数据列包含source/context分母、关系CE/accuracy及正交G/S/L对比，完整图页的真实渲染待共同评估出现；说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_CONTEXT_ABLATION_FIGURES_20261002.md。原p01监督完成后续看p02，至多12小时或22项全终态，无新增OS监听/平台任务。


## 2026-10-02T21:53:57.288545+00:00 首批p02已实际运行，p01与p02结果分开

21:52:31 UTC只读核实p02 8running/14pending，首批均完成step0并实际25–100步。GPU0=f000/42（100步），GPU4=f001/42（75），GPU1=cv_segment_global/42（50），GPU5=cv_late_fusion/42（50），GPU2=f000/43（50），GPU6=f001/43（50），GPU7=cv_segment_global/43（50），GPU3=cv_late_fusion/43（25）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/first_active_20261002T215231Z.json。原worker自然接续，没有新平台提交/容器或监听重启。

p01原8项3125步/100000记录已全部完成并核验，最终图另保留；p02使用独立120k样本/3750 horizon，不混用p01数值补矩阵。监督最多12小时或22项全终态，约55秒只读原账本与aggregate；异常先记录/汇报，不自行重试，全部sole latest.pt保留。已验p02作图入口待至少两个实际匹配run出现共同非零评估后首次渲染，缺格保持缺失。


## 2026-10-02T22:08:15.189175+00:00 p02首批共同250步真实图首次渲染通过

22:06:28 UTC首批8项f000/f001/cv_segment_global/cv_late_fusion×seed42/43均完成250步验证（每项8000记录曝光），实际训练250–300/3750，8running/14pending，无失败/心跳或曝光异常。既有p02作图首次真实渲染rc0，status=partial、final=false、endpoint_step=250、8/22实测格、14缺格，warnings=[]。原sample/source/配置/同seed初始化与同step曝光检查通过。

实际overview/factorial/strata/separate_controls四页各SVG/PDF/PNG共12图已确认存在、非空并记录SHA；overview/factorial PNG目视通过，组合缺格及尚不可计算的交互/移除对照保持空白。图前缀 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/figures/v7_context_ablation；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/first_render250_20261002.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/first8_shared250_20261002T220628Z.json。

当前是新120k上的首批早期正例学习/遮蔽重建结果，不能称22项完整消融完成、正负判别或最终架构选定。旧p01最终图未覆盖；全部sole latest.pt保留。继续原22项队列，不新增/重试任务、不改运行源码。


## 2026-10-02T22:19:34.974599+00:00 旧p02监督范围随用户batch调整收窄为原8运行项

根端22:15:25 UTC已将原14个未启动pending原件原子移至batch_revision_20261002/withdrawn_pending，原22项jobs账本保留。独立监督核对14份移后原件SHA与withdrawal_receipt一致，均无run目录、评估或pool result，状态明确为withdrawn-before-start，非失败或仍pending；不补回队列。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/batch_revision_20261002/withdrawal_receipt.json；独立状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/withdrawal_status_independent_20261002.json。

旧p02仅原8个micro4/GA8任务继续3750步，终态图保留8/22 partial与14未执行缺格，缺失原因为用户要求提高后续batch。既有模型、样本、日程、checkpoint与worker未改；新large-batch组须另册登记及作图，不和旧p02拼矩阵。旧p02监督停止点改为原8项全终态，或先到原12小时观察上限。


## 2026-10-02T22:55:13.494412+00:00 旧p02原8运行项共同1000步验证与图已实际更新

22:53:55 UTC旧8项全部running，实际1000–1075/3750，共同1000每项32000记录曝光；14原未启动项仍withdrawn-before-start并逐份核SHA，没有失败或心跳/曝光异常。一个另行登记的容量probe属于独立任务，不是这14项重入队。

| 条件/seed | 四mask token AA CE | 四mask正例关系CE | 四mask正例关系准确率 | full-mask关系token数 |
|---|---|---|---|---|
| cv_late_fusion/s42 | 0.991087 | 0.000898 | 100.00% | 37 |
| cv_late_fusion/s43 | 0.975321 | 0.000863 | 100.00% | 37 |
| cv_segment_global/s42 | 0.985494 | 0.000747 | 100.00% | 37 |
| cv_segment_global/s43 | 0.976937 | 0.000555 | 100.00% | 37 |
| f000/s42 | 1.026356 | 0.011798 | 100.00% | 37 |
| f000/s43 | 1.011729 | 0.000958 | 100.00% | 37 |
| f001/s42 | 0.969643 | 0.000754 | 100.00% | 37 |
| f001/s43 | 0.962069 | 0.000679 | 100.00% | 37 |

所有关系监督均为用户明确选定的binding正例；正例关系准确率高不能解释为正负判别能力，完整关系验证分母仅37。上表仅中间重建/正例学习结果，不选最终赢家，不将32000曝光称完整120000条epoch。

既有作图实际rc0，endpoint_step=1000、8/22 partial、final=false、14缺格；12份SVG/PDF/PNG均存在非空、SHA写入 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/render1000_20261002.json。监督快照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/active8_shared1000_20261002T225355Z.json。原图前缀保持 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/figures/v7_context_ablation；14缺格原因是用户batch修订撤下，不冒充完整三因素矩阵或失败。旧8任务继续原micro4/GA8/3750，全部latest.pt保留，不混后续large-batch组，不新增任务或重启。


## 2026-10-02T23:57:10.918944+00:00 旧p02原8项共同2000步验证与图已落盘

23:55:40 UTC旧8项全部running，实际2000–2075/3750；共同2000每项64000记录曝光，仍不是120000条完整epoch。14未启动项保持withdrawn-before-start，原件SHA与无运行证据持续核对通过；既有8项无失败/心跳/曝光异常。

| 条件/seed | 四mask token AA CE | 四mask正例关系CE | 四mask正例关系准确率 | 记录曝光 |
|---|---|---|---|---|
| cv_late_fusion/s42 | 0.894573 | 0.000292 | 100.00% | 64000 |
| cv_late_fusion/s43 | 0.883733 | 0.000277 | 100.00% | 64000 |
| cv_segment_global/s42 | 0.892453 | 0.000307 | 100.00% | 64000 |
| cv_segment_global/s43 | 0.878854 | 0.000338 | 100.00% | 64000 |
| f000/s42 | 0.904568 | 0.001676 | 100.00% | 64000 |
| f000/s43 | 0.895628 | 0.000948 | 100.00% | 64000 |
| f001/s42 | 0.887570 | 0.000331 | 100.00% | 64000 |
| f001/s43 | 0.873039 | 0.000314 | 100.00% | 64000 |

这是旧micro4/GA8组的中间正例学习/遮蔽重建指标，关系验证完整分母仍37，不解释为正负判别或生物功能收益。既有plot实际rc0，endpoint_step=2000、8/22 partial、final=false；12份图已确认存在非空并记录SHA，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/render2000_20261002.json。绘图入口近期仅兼容figure_title，记录此次实际plot源码SHA，不把标题扩展冒充运行模型/配置改变。

监督证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/active8_shared2000_20261002T235540Z.json；图前缀 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/figures/v7_context_ablation。14缺格因用户batch修订撤下，不补旧值或新组值。继续原8项3750终态、保留全部sole latest.pt，不发布/入队/重启；新大batch/容量probe/正式组各自独立。


## 2026-10-03T00:59:14.039482+00:00 旧p02原8项共同3000步验证与图已更新

2026-10-03 00:58:01 UTC旧8项全部running，实际3000–3075/3750，共同3000每项96000记录曝光；仍欠24000条曝光，未写成完整120000条epoch。14withdrawn-before-start保持、原件SHA及无执行证据持续一致。无失败、心跳失联或records/step偏差。

| 条件/seed | 四mask token AA CE | 四mask正例关系CE | 记录曝光 |
|---|---|---|---|
| cv_late_fusion/s42 | 0.841672 | 0.000180 | 96000 |
| cv_late_fusion/s43 | 0.826449 | 0.000220 | 96000 |
| cv_segment_global/s42 | 0.841070 | 0.000175 | 96000 |
| cv_segment_global/s43 | 0.826294 | 0.000188 | 96000 |
| f000/s42 | 0.848733 | 0.001052 | 96000 |
| f000/s43 | 0.834529 | 0.000365 | 96000 |
| f001/s42 | 0.830354 | 0.000213 | 96000 |
| f001/s43 | 0.821936 | 0.000165 | 96000 |

上述仅原micro4/GA8组中间遮蔽重建及既有binding正例学习；关系完整验证分母37，不能据此声称正负判别、生成或生物功能收益。既有plot实际rc0，endpoint_step=3000、8/22 partial、final=false；12份SVG/PDF/PNG存在非空并记录SHA，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/render3000_20261003.json。监督快照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/active8_shared3000_20261003T005801Z.json。

图前缀 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/figures/v7_context_ablation；14缺格按用户batch修订撤下保留，不使用别组补齐。继续原8项3750终态、保留全部sole latest.pt，无发布/入队/重启；独立容量probe与新大batch/正式组均不纳入旧图。


## 2026-10-03T01:54:08Z 旧p02原8项完整epoch终态已核验；22项设计仍partial

2026-10-03 01:46:03 UTC首次观测原8项全部pool succeeded/returncode0及COMPLETE。独立终态核验全部通过：每项3750步、records=unique_train_records=subset_train_records=120000、epoch1/position0；16个共同验证点0及250至3750齐全，summary/provenance SHA与COMPLETE绑定一致。49个实际加载源码SHA逐一匹配566文件冻结快照，8项加载源码映射和样本manifest SHA一致，同seed四项公共初始化SHA一致。每run唯一latest.pt（907355403 bytes）原样保留，无临时checkpoint或额外checkpoint目录；未加载或修改checkpoint。

14项仍withdrawn-before-start：逐一原件SHA匹配、无pending/run/result，因用户batch变更撤下而非失败。原8项有限监督至此结束，不补回、不重试、不启动新训练。

3750步四mask token-pooled AA CE实测如下：

| 条件 | seed42 AA CE | seed43 AA CE | 两seed均值 |
|---|---:|---:|---:|
| f000 | 0.828634 | 0.823307 | 0.825971 |
| f001 | 0.813223 | 0.810867 | 0.812045 |
| cv_segment_global | 0.819328 | 0.814601 | 0.816965 |
| cv_late_fusion | 0.820687 | 0.816286 | 0.818486 |

这是遮蔽重建的两seed描述性比较；fullmask AA准确率范围42.64%–43.77%，不能把CE改善泛化为所有指标或最终架构优胜。验证context只有37条（无context987条），fullmask关系分母37，且全为已有binding正例；不声称正负判别、生成品质、理解或生物功能提升。G/S组合、固定gate对照及交互/移除消融仍缺，不能称完整22项消融。

训练循环合计30.7660 GPU小时，worker wall合计31.3443 GPU小时；实际长度峰显存3.293–3.888 GiB。01:46:43重绘报告明确status=partial、final=false、endpoint3750、8/22有值/14缺格；12份SVG/PDF/PNG与JSON/CSV均实际存在且已记录SHA，overview PNG已目视检查。最终图前缀：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/figures/v7_context_ablation；终态/成本/图SHA证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/final_active8_p02_20261003.json。旧p01图未覆盖，新大batch组未混入。

并行容量探测另据根端01:48:46报告：原唯一p04 job第一档180秒超时、returncode1，非OOM，无COMPLETE或推荐；资源仍Running，8worker standby_load、pending0/running0。新容量协议仅工程准备，新22项和正式八卡训练未启动；该容量失败不计入旧p02八项结果。
