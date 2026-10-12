# 当前架构筛选结果与接续决定

当前执行 2026-10-03T04:32:08.720556+00:00：2026-10-03T04:24:25.624UTC，首8的真实curve/summary均完成500步、每项16000/120000条唯一曝光（13.33%）；仍8running/14pending，无失败或3750终态。统一micro32×GA1/global32、eval4及原完整22范围保持。 f011两seed汇总AA CE均低于匹配R20，均值优势由250的20.58%收窄到500的11.50%；f001为9.85%、f010为6.23%。全mask AA准确率当前R20最高，f011低1.40个百分点，且全mask CE更高；context子集三候选均值略低，但仅37条正例验证。不能将部分遮蔽收益解释为生成或多链理解已改善，不凭早期汇总选正式模型。两seed无显著性结论，仍8/22 partial/final=false，旧4×8协议独立。 当前500证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/render500_20261003.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/current_method_comparison_shared500_20261003.json。

更新于 2026-10-03T02:27:52Z（R20定义保持，第二轮原八项已完整3750步；22格设计仍partial）。当前是重建指标的阶段性判断，正式模型尚未选定。两 seed 均值不构成显著性结论；100k 完整首轮与新增 context 的120k第二轮分别比较，不能直接比较两轮绝对 loss。

## R20 的基线定义：2026-10-02T23:19:52Z 冻结源码核实

R20 是本轮 no-ESMC、随机初始化的共同对照模型，实际名称 `r20_refined_control`，继承 `r12_architecture_control` 且 options 为空。其数据编码使用 A4；A4 是输入布局名称，R20 是用于后续改动对比的模型版本名称。实际首轮 provenance 记录参数 75,601,176，未加载 ESMC 或其他预训练蛋白编码器。

| 部分 | R20 实际设置 |
|---|---|
| Backbone | LLaDA 双向 Transformer，8 层、hidden 768、12 attention heads、FFN 3072；RMSNorm、SiLU 门控 FFN，最大渲染长度 2048 |
| 输入编码 | 紧凑氨基酸/grammar token embedding，加链 role embedding、观测 scope embedding |
| 位置编码 | RoPE，链起始标记位置 0、链内内容从 1 编号，每条链重新开始 |
| 链间编码 | 每个 head 有两项可学习 attention bias，区分同链与其他位置；两项均从 0 初始化，其他位置也包括 grammar prefix |
| 注意力 | 全部 8 层为全局双向注意力；padding 不作为 key；没有新增动态 head gate 或局部/全局交替 |
| 残差 | attention 和 FFN 残差系数均为 1；没有 R24 的固定 0.1 缩放 |
| 训练目标 | 在 eligible 位置抽样遮蔽并计算 CE；context 序列不遮蔽、不计算序列重建 CE；已有 `relation_supervised` 关系 token 可参与遮蔽监督，当前数据配方只使用已有 binding 正例 |
| 长度与结束标记 | target full/fragment 各预留 256/64 个内容位置，容量包含一个 `<protd>` 结束标记；真实内容后的未标注 mask 位置仍在 attention 中；context 按实际观测长度渲染，批间 padding 动态至本 batch 最大渲染长度 |

因此当前 R20 并未完全取消固定 target 容量或结束标记训练；动态 batch padding 不能等同于 target 已无固定容量。该 forward 没有显式 diffusion timestep embedding；遮蔽率用于采样 corruption。这里是现有源码定义核实，没有新架构、数据或训练改动。

R24 相对 R20 改 attention/FFN 固定残差缩放；R26 相对 R20 改链内局部/全局注意力交替。同 seed 首轮共享同一共同参数初始化、数据顺序及训练配方，用于解释各项改动的影响。首轮 R20 采用 micro4×GA8/global32，100000 train、1024 valid、3125 步完整一 epoch、seed42/43；两 seed 的四档遮蔽氨基酸重建 CE 均值为 0.837899。此 CE 不含关系/结束标记的汇总，也不代表实际生成或下游理解质量。

实际运行配置：[R20 seed42 provenance](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790968470149719891_p01_v7_r20_seed42_838aa052/run_provenance.json)。冻结模型：[R20 模型定义](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/source_snapshot_v1/dllm/pipelines/immune_llada/research_refined/model.py)、[A4 输入布局](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/source_snapshot_v1/dllm/pipelines/immune_llada/v3/layout.py)。本次五份冻结源码 SHA 描述原件：[核实记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/r20_baseline_source_description_20261002.json)，SHA256 f3ab0a6373f0f1d4b2c701548a38dcdb3c92510d4ef9fd28cee4426c4526dc6a。

## 2026-10-03T04:02:11.291493+00:00 新统一batch共同250：尚未确定最终架构

| 条件 | 两seed平均AA CE | 相对R20降低 | 全mask AA准确率 | context子集AA CE |
|---|---:|---:|---:|---:|
| R20基线/f000 | 1.565687 | 0.00% | 40.58% | 1.997667 |
| R24/f010 | 1.449769 | 7.40% | 41.04% | 1.972678 |
| R26/f001 | 1.269924 | 18.89% | 40.59% | 2.064495 |
| R24+R26/f011 | 1.243442 | 20.58% | 40.51% | 2.118483 |

这是新组四条件×两seed的早期重建结果，8/22 partial、final=false；另14项仍待运行。组合f011的汇总AA CE最低，但全mask准确率未显示同步改善，且仅37条context验证上的AA CE更高；不能认定生成、多链理解、二分类或生物功能提升，不能据250步选正式模型。两个seed不构成显著性证据；旧micro4×GA8结果独立保留。

真实250报告及12图已逐字节独立留存 [/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/report.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/report.json)；[实际概览图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/v7_large_batch_ablation_overview.png)；[渲染/归档记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/render250_20261003.json)（SHA41f067ee719394b508be77c97080b01e06bc367a7d203cb987672e5a1ad981f3）；[两seed派生](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/current_method_comparison_shared250_20261003.json)（SHA514bf2eba5a2ac69603e08e31373a10421690ba740451b260fc5851441863965）。未重绘、未重复模型验证或读取二进制checkpoint。

## 2026-10-03T04:32:08.720556+00:00 新共同500，正式模型仍未选定

| 条件 | 两seed平均AA CE | 相对R20降低 | 全mask AA CE | 全mask AA准确率 | context子集AA CE |
|---|---:|---:|---:|---:|---:|
| R20/f000 | 1.179252 | 0.00% | 1.790015 | 41.13% | 1.999002 |
| R24/f010 | 1.105767 | 6.23% | 1.799239 | 40.62% | 1.976865 |
| R26/f001 | 1.063047 | 9.85% | 1.831919 | 40.13% | 1.962080 |
| R24+R26/f011 | 1.043684 | 11.50% | 1.837388 | 39.73% | 1.969006 |

f011两seed汇总AA CE均低于匹配R20，均值优势由250的20.58%收窄到500的11.50%；f001为9.85%、f010为6.23%。全mask AA准确率当前R20最高，f011低1.40个百分点，且全mask CE更高；context子集三候选均值略低，但仅37条正例验证。不能将部分遮蔽收益解释为生成或多链理解已改善，不凭早期汇总选正式模型。两seed无显著性结论，仍8/22 partial/final=false，旧4×8协议独立。

[500概览图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared500_20261003/v7_large_batch_ablation_overview.png)；原报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared500_20261003/report.json（SHAd8d1ae3b9c8f6e33592f5c7478a494ad944826fe61ed4b579878054ef0b7ac94）；[12图及实际渲染归档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/render500_20261003.json)（SHA9d19428788fad9579a9196c1c0053a071ab296c99ad63184013869d9b50a1af4）；[两seed派生](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/current_method_comparison_shared500_20261003.json)（SHAcf0571f33cbb9358ae15e7fda91ef744e1496e0dff7c626d2842bdc5a4b6d0a4）。250独立归档保留，500真实endpoint只渲染一次。

## 首轮完整 epoch：100000 条、3125 步、seed42/43

下表均为四档遮蔽率的氨基酸重建 CE，越低越好。token 汇总平均较偏重大来源；source 宏平均让每个来源权重相同，但部分来源验证仅一两条，估计不稳定。

| 改动 | token 汇总 CE，两 seed 平均 | 相对同轮基线下降 | 来源宏平均 CE，两 seed 平均 | 来源宏平均相对基线下降 |
|---|---:|---:|---:|---:|
| R20 基线 | 0.837899 | — | 1.340682 | — |
| R24 固定残差缩放 | 0.817343 | 2.45% | 1.290171 | 3.77% |
| R26 交替链内局部/全局注意力 | 0.820938 | 2.02% | 1.246137 | 7.05% |
| R21 动态 attention head 门控 | 0.847980 | -1.20% | 1.364200 | -1.75% |

R24 与 R26 的两个 seed 的 token CE 均低于各自基线。R24 的 token 汇总 CE 最低，R26 的来源宏平均最低；R21 在这个配方下未显示 token CE 改善。full-mask 氨基酸准确率均约43.7%，差异很小，不能把平均遮蔽重建 CE 改善解释为生成优势已得到证明。

真实终态原件：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/watch_receipts/final_p01_20261002.json。最终图：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/figures/v7_pilot.png（同前缀 SVG/PDF/CSV/JSON）。

## 第二轮原八项完整epoch：3750步、120000条，22格设计仍partial

原八项终态独立核验通过，每项3750步、120000条唯一记录、epoch1；COMPLETE绑定summary/provenance，实际加载源与566冻结、同seed初始化、唯一latest.pt均核验一致。十四项启动前撤下而未执行，图仍partial/8of22/final=false；不能当完整组合消融。原micro4×GA8/global32保持。

| 改动 | token 汇总 AA CE，两 seed 平均 | 相对同轮基线下降 |
|---|---:|---:|
| f000，R20基线 | 0.825971 | — |
| f001，R26局部/全局交替 | 0.812045 | 1.69% |
| CV整段链内/全局交替 | 0.816965 | 1.09% |
| CV延迟跨链融合 | 0.818486 | 0.91% |

三项改动的两个seed均低于匹配基线；R26在本组四方案中token重建CE最低，R24及组合f011仍待新统一batch组。只有37条context正例验证，当前指标不能支持正负判别、实际生成或生物功能改善结论。

独立终态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/final_active8_p02_20261003.json（SHA6e5795f08e73e7a4f534812f5779ee412b686da098419d1a44693d9096101be3）；最终12图前缀 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/figures/v7_context_ablation。根端逐字节保存实际报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/report_shared3750_20261003.json（SHA6eafd2574efa1e4da0b9d61332d97c81764cdcde0e508d84fdc17aa2ed5d1c72），两seed派生 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/current_method_comparison_shared3750_20261003.json（SHA41970a9198eef06485e281097b054b32674a1069d247a3ac273e9467f08aba90）。没有新增checkpoint或重跑模型。

## 第二轮历史共同3000步：96000条曝光，尚未完整120000条epoch

根端读取监督者真实报告，8项都具有共同3000终点评估及96000条唯一训练曝光。原micro4×GA8/global32保持，14项启动前撤下而未运行；不是完整消融矩阵，不与未来独立大batch组混比。

| 改动 | token 汇总 AA CE，两 seed 平均 | 相对同轮基线下降 |
|---|---:|---:|
| f000 基线 | 0.841631 | — |
| f001，R26局部/全局交替 | 0.826145 | 1.84% |
| CV整段链内/全局交替 | 0.833682 | 0.94% |
| CV延迟跨链融合 | 0.834060 | 0.90% |

R26对应两个seed都优于匹配基线，仍是本轮优先候选；1000/2000/3000的相对改善由5.22%/2.20%收窄到1.84%，不能将早期收益当最终值。3750完整epoch和f011组合仍待；这些重建/已有正例关系指标不证明实际生成、正负判别或生物功能改进。

不会被后续绘图覆盖的真实原报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/report_shared3000_20261003.json，SHA256 14c6b7abfe797d283e7f031ff1608d35f117db7a1e5430775620bd7eaebb597f。派生两seed均值 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/current_method_comparison_shared3000_20261003.json，SHA256 4967fdcf5462e56e99d570bccf58df2adb9fd5325a319b6df205f9da97d6e382；实际render原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/render3000_20261003.json。未重绘或新增checkpoint。

## 第二轮历史共同2000步：120000 条配方，尚未完整 epoch

本轮增加既有 context 正例，当前共同曝光每项64000条。各项仍采用原micro4×GA8/global32，不能与未来新大batch组混为一个矩阵。实际已重绘12图，仍为8/22 partial，原14项撤下而未运行。

| 改动 | token 汇总 CE，两 seed 平均 | 相对同轮基线下降 |
|---|---:|---:|
| f000 基线 | 0.900098 | — |
| f001，对应 R26 局部/全局交替 | 0.880305 | 2.20% |
| CV 启发的整段链内/全局交替 | 0.885653 | 1.60% |
| CV 启发的延迟跨链融合 | 0.889153 | 1.22% |

此共同中间点 f001 的两个 seed 均优于同 seed 基线，并且 token 平均最低；其相对改善由1000步的5.22%收窄到2000步的2.20%，不能把早期幅度当作最终收益。完整3750步结论尚未到，组合f011也未验证，不能据此确定最终赢家。关系监督仍为已有binding正例，验证全遮蔽关系token分母仅37，100%正例关系准确率不表示具备正负判别或实际使用context的证据。

原图和汇总：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/figures/v7_context_ablation.json，2000步汇总SHA256 ebde7475b316aac968c9fb1f70666d251516783871d367455ed1a71e0387e26d。实际图件记录：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/render2000_20261002.json。根端两seed均值与百分比原件：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/current_method_comparison_shared2000_20261003.json，SHA256 a7f307668e464767dd9e2e0df612e5da6e5d27baa1992e933255125374a780c0。

共同2000步报告已于2026-10-03T00:16:27Z独立归档：[2000步原报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/report_shared2000_20261003.json)，4,419,007 bytes，SHA与上述汇总一致。原件记录：[归档 receipt](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/report_shared2000_archive_20261003.json)。后续 figures 报告可随真实里程碑更新，本归档保留当时8/22 partial、final=false的数值证据，不新增训练或checkpoint。

## 2026-10-03T02:27:52Z 历史取舍与接续计划（新22登记前）

保留 R26 与 R24 为优先候选。R26 在首轮的来源宏平均及新context配方的中间 token CE 均有较好表现；R24 在完整首轮的 token CE 最好且没有新增动态门控参数。新统一大batch组应优先覆盖基线f000、R24对应f010、R26对应f001和组合f011各两seed，再补齐其他门控组合和CV对照；全11条件/两seed的消融设计保持，未实际登记新22项。

f011 组合收益尚未验证，不能假定两项改善可相加。R24 也可能与当前学习率/优化尺度相关；R21保留在消融中用于检验门控作用，尚不作为默认改动。完整矩阵、第三seed确认、实际八卡共同更新和保存恢复、理解/生成的下游效果仍待验证。原容量首格非OOM超时保持失败；独立v2容量九档实际passed/pool rc0，可据新协议核实后选共同micro32，不能冒充旧8更新协议通过。新22项与正式训练未启动。

历史首次比较（2026-10-02T23:01:46Z，首轮完整epoch与第二轮共同1000步）的均值/百分比派生记录，含当时两个原始 JSON 的 SHA：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/current_method_comparison_20261002T230146Z.json（SHA256 ef9d8c1d80a7ce975c4734efe99a08ebebe25670b3cfe29e8c2f58c14be9309c）。第二轮当前比较使用上文3750完整epoch原件；1000/2000/3000保留历史。架构数值核实没有新增真实语料训练、序列注释、负例或数据改动。
