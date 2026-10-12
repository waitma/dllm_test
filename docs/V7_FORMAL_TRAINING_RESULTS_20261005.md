# V7 正式训练实际结果

## 2026-10-06T22:28:30.597931+00:00 累计两轮50082正式终态验收完成

/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint50082_root_review.json（188b61 rc0）和 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/segment50082_root_review.json（6fe75e rc0）确认epoch2、最终原生保存及八rank成功退出；保留45000/46000/47000＋50082四份完整恢复点。最后24步含恢复/验证/保存/退出的150.614396秒不是稳态速度。完整5000步墙钟口径继续见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/SPEED_ANALYSIS.md；原模型及日程未再改。

| 指标 | 50058 | 50082 |
|---|---:|---:|
| eval_loss | 1.864237411957972 | 1.8642526092601706 |
| eval_mask_mean_token_micro_aa_loss | 0.7002642121078781 | 0.7002715578516201 |
| eval_mask_mean_token_micro_aa_accuracy | 0.783447716091204 | 0.7834443711091101 |
| eval_mask_mean_token_micro_terminal_loss | 0.45167712094471657 | 0.4515824635807853 |
| eval_mask_mean_token_micro_terminal_accuracy | 0.8491383645646087 | 0.8491393546430328 |
| eval_mask100_token_micro_aa_loss | 1.6935274471985269 | 1.6935243471274863 |
| eval_mask100_token_micro_aa_accuracy | 0.43742331288343556 | 0.43741757926724384 |
| eval_mask100_token_micro_terminal_loss | 1.365666478942215 | 1.365107962190608 |
| eval_mask100_token_micro_terminal_accuracy | 0.536889332003988 | 0.5378863409770688 |
| eval_mask100_token_micro_first_terminal_accuracy | 0.16550348953140578 | 0.16650049850448653 |
| eval_mask100_token_micro_terminal_missing_rate | 0.0 | 0.0 |
| eval_mask100_token_micro_length_mae | 3.3125623130608175 | 3.310069790628116 |
| eval_mask100_token_micro_terminal_precision | 0.36807928913192073 | 0.3681337427499147 |
| eval_mask100_token_micro_terminal_false_positive_rate | 0.0106014563385127 | 0.010618657187087896 |
| eval_mask100_token_micro_length_compared_count | 2006 | 2006 |
| eval_mask100_token_micro_chain_count | 2006 | 2006 |

两轮后的九项实际CDR已完成，数字唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr；正式训练停在固定两轮，不从test选其他checkpoint。

## 2026-10-06T22:21:08.237756+00:00 50058保存与长段实际退出验收

50058完整保存/验证7f53af rc0，/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint50058_root_review.json（SHA75b00695af94dcda7267d1adf45ee5a1b2d46c6abc44df46c60a522ff1641103）；45058→50058整段49a0a3 rc0，/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/segment50058_root_review.json（SHA14301194f8e75fddc96cfb7e6dee613b44864e0ab932e8d9a5ce410033682f05）。5000个实际更新连续齐全、24项optimizer/scheduler/各rank RNG原生恢复、八实际pool成功退出/子进程清理通过；45058旧权重已正常轮替，其原生JSON归档用于校验，没有重载旧权重或伪造恢复。当前last50058＋best3 45000/46000/47000共4份完整状态，50000权重正常轮替。

| 指标 | 50000 | 50058 |
|---|---:|---:|
| eval_loss | 1.8642699545095784 | 1.864237411957972 |
| eval_mask_mean_token_micro_aa_loss | 0.700276524467664 | 0.7002642121078781 |
| eval_mask_mean_token_micro_aa_accuracy | 0.7834257359018514 | 0.783447716091204 |
| eval_mask_mean_token_micro_terminal_loss | 0.4516863617814663 | 0.45167712094471657 |
| eval_mask_mean_token_micro_terminal_accuracy | 0.8488901023997626 | 0.8491383645646087 |
| eval_mask100_token_micro_aa_loss | 1.6935363214306969 | 1.6935274471985269 |
| eval_mask100_token_micro_aa_accuracy | 0.43733157502436787 | 0.43742331288343556 |
| eval_mask100_token_micro_terminal_loss | 1.36507420232337 | 1.365666478942215 |
| eval_mask100_token_micro_terminal_accuracy | 0.536889332003988 | 0.536889332003988 |
| eval_mask100_token_micro_first_terminal_accuracy | 0.16550348953140578 | 0.16550348953140578 |
| eval_mask100_token_micro_terminal_missing_rate | 0.0 | 0.0 |
| eval_mask100_token_micro_length_mae | 3.313060817547358 | 3.3125623130608175 |
| eval_mask100_token_micro_terminal_precision | 0.3682051282051282 | 0.36807928913192073 |
| eval_mask100_token_micro_terminal_false_positive_rate | 0.010595722722320967 | 0.0106014563385127 |
| eval_mask100_token_micro_length_compared_count | 2006 | 2006 |
| eval_mask100_token_micro_chain_count | 2006 | 2006 |

原监督已单次接续最后50058→50082，当前读数50058；最终两轮和下游生成仍须其实际证据。5000更新从实际入队到验收3753.498933秒、平均0.750699787秒/步，包含初始化/恢复、验证、保存和退出；不是额外测速，也不等同只有训练循环的0.721418秒窗口。


## 2026-10-06T22:19:25.314881+00:00 50000保存/验证通过

实际13ffb3 rc0；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint50000_root_review.json（SHA21d34fcdd1504b6af446b755a50590fe703050f113ad336e2e41c63ab7984740），17项原生文件/8rank摘要/615源绑定及112/70/26指标检查通过。当前best3为45000/46000/47000，latest50000另存；49000权重正常轮替，原JSON保留。主eval_loss略升，AA与END指标有升降，不用该点替代固定50082的生成评测。

| 指标 | 49000 | 50000 |
|---|---:|---:|
| eval_loss | 1.8639524893489763 | 1.8642699545095784 |
| eval_mask_mean_token_micro_aa_loss | 0.7001980334282351 | 0.700276524467664 |
| eval_mask_mean_token_micro_aa_accuracy | 0.7834095928165133 | 0.7834257359018514 |
| eval_mask_mean_token_micro_terminal_loss | 0.45563168893814315 | 0.4516863617814663 |
| eval_mask_mean_token_micro_terminal_accuracy | 0.8490137384429736 | 0.8488901023997626 |
| eval_mask100_token_micro_aa_loss | 1.6930647332742146 | 1.6935363214306969 |
| eval_mask100_token_micro_aa_accuracy | 0.4373545094891348 | 0.43733157502436787 |
| eval_mask100_token_micro_terminal_loss | 1.3776927358374058 | 1.36507420232337 |
| eval_mask100_token_micro_terminal_accuracy | 0.5363908275174477 | 0.536889332003988 |
| eval_mask100_token_micro_first_terminal_accuracy | 0.16600199401794616 | 0.16550348953140578 |
| eval_mask100_token_micro_terminal_missing_rate | 0.0 | 0.0 |
| eval_mask100_token_micro_length_mae | 3.3155533399800596 | 3.313060817547358 |
| eval_mask100_token_micro_terminal_precision | 0.3692518874399451 | 0.3682051282051282 |
| eval_mask100_token_micro_terminal_false_positive_rate | 0.010538386560403647 | 0.010595722722320967 |
| eval_mask100_token_micro_length_compared_count | 2006 | 2006 |
| eval_mask100_token_micro_chain_count | 2006 | 2006 |


## 2026-10-06T22:09:02.283505+00:00 49000保存/验证通过，latest与best3分开保留

新节点b9be88 rc0；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint49000_root_review.json（SHA587966599581f94e394c8a9bddf3640465b070f2b4f0bd0e5e6a979b4bfd0ce0）。17项原生清单、8rank参数摘要、615源绑定及112/70/26计数/聚合检查通过；根端未重新加载权重或重跑模型，前一48000验收不重复。

| 指标 | 48000 | 49000 |
|---|---:|---:|
| eval_loss | 1.8636487733774219 | 1.8639524893489763 |
| eval_mask_mean_token_micro_aa_loss | 0.7000110901580132 | 0.7001980334282351 |
| eval_mask_mean_token_micro_aa_accuracy | 0.783514221833544 | 0.7834095928165133 |
| eval_mask_mean_token_micro_terminal_loss | 0.4506606157015345 | 0.45563168893814315 |
| eval_mask_mean_token_micro_terminal_accuracy | 0.8502186869197976 | 0.8490137384429736 |
| eval_mask100_token_micro_aa_loss | 1.6941515729060959 | 1.6930647332742146 |
| eval_mask100_token_micro_aa_accuracy | 0.43771572730921393 | 0.4373545094891348 |
| eval_mask100_token_micro_terminal_loss | 1.3616658210215329 | 1.3776927358374058 |
| eval_mask100_token_micro_terminal_accuracy | 0.5388833499501495 | 0.5363908275174477 |
| eval_mask100_token_micro_first_terminal_accuracy | 0.16350947158524426 | 0.16600199401794616 |
| eval_mask100_token_micro_terminal_missing_rate | 0.0 | 0.0 |
| eval_mask100_token_micro_length_mae | 3.323529411764706 | 3.3155533399800596 |
| eval_mask100_token_micro_terminal_precision | 0.36323924731182794 | 0.3692518874399451 |
| eval_mask100_token_micro_terminal_false_positive_rate | 0.010865202683332378 | 0.010538386560403647 |
| eval_mask100_token_micro_length_compared_count | 2006 | 2006 |
| eval_mask100_token_micro_chain_count | 2006 | 2006 |

平均AA和END指标略退，全遮蔽AA CE、first-END、长度MAE、END precision和误报率有改善；主eval_loss略高于48000，不能称全面改善。当前best3是45000/46000/47000，latest49000另存，合计4份完整权重/优化器/调度器/8rank RNG；这正是best3+latest取并集，不能继续写“latest已在best3中”。继续固定50082评测，不按测试表现换checkpoint或解码。


## 2026-10-06T21:56:00.268243+00:00 48000保存/验证通过，latest与best3分开保留

新节点a14188 rc0；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint48000_root_review.json（SHA8b873755583471f357ab6429042ac1cdc5a3d18509bb9a5dc27dfc02effb950c）。17项原生清单、8rank参数摘要、615源绑定及112/70/26计数/聚合检查通过；根端未重新加载权重或重跑模型，前一47000验收不重复。

| 指标 | 47000 | 48000 |
|---|---:|---:|
| eval_loss | 1.8634856055982953 | 1.8636487733774219 |
| eval_mask_mean_token_micro_aa_loss | 0.7000837134265516 | 0.7000110901580132 |
| eval_mask_mean_token_micro_aa_accuracy | 0.7831521832067155 | 0.783514221833544 |
| eval_mask_mean_token_micro_terminal_loss | 0.4447665798493376 | 0.4506606157015345 |
| eval_mask_mean_token_micro_terminal_accuracy | 0.8514659382145727 | 0.8502186869197976 |
| eval_mask100_token_micro_aa_loss | 1.6928995481811533 | 1.6941515729060959 |
| eval_mask100_token_micro_aa_accuracy | 0.4370850295281234 | 0.43771572730921393 |
| eval_mask100_token_micro_terminal_loss | 1.3428790921457343 | 1.3616658210215329 |
| eval_mask100_token_micro_terminal_accuracy | 0.5448654037886341 | 0.5388833499501495 |
| eval_mask100_token_micro_first_terminal_accuracy | 0.16201395812562314 | 0.16350947158524426 |
| eval_mask100_token_micro_terminal_missing_rate | 0.0 | 0.0 |
| eval_mask100_token_micro_length_mae | 3.073778664007976 | 3.323529411764706 |
| eval_mask100_token_micro_terminal_precision | 0.36036927134849983 | 0.36323924731182794 |
| eval_mask100_token_micro_terminal_false_positive_rate | 0.011123215411960324 | 0.010865202683332378 |
| eval_mask100_token_micro_length_compared_count | 2006 | 2006 |
| eval_mask100_token_micro_chain_count | 2006 | 2006 |

平均AA略改善，但END CE/准确率和长度MAE回退；完整48000的主eval_loss略高于47000。当前best3是45000/46000/47000，latest48000另存，合计4份完整权重/优化器/调度器/8rank RNG；这正是best3+latest取并集，不能继续写“latest已在best3中”。继续固定50082评测，不按测试表现换checkpoint或解码。


## 2026-10-06T21:43:49.470154+00:00 47000完整保存与固定验证核查

实际工具97030c rc0，17项原生保存清单、8rank参数摘要一致、615冻结源码绑定、112计数/70聚合/26结束指标检查通过；完整张量字节核验来自8rank原生保存回执，根端未重载张量。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint47000_root_review.json（SHA96978302587b0cfc6f50e1df3c5b4c54d13aa74027e1e15a3147d5aee72ac031）。前一46000节点没有重跑。

| 指标 | 46000 | 47000 |
|---|---:|---:|
| eval_loss | 1.862781762676079 | 1.8634856055982953 |
| eval_mask_mean_token_micro_aa_loss | 0.7011582623495837 | 0.7000837134265516 |
| eval_mask_mean_token_micro_aa_accuracy | 0.7830569929080398 | 0.7831521832067155 |
| eval_mask_mean_token_micro_terminal_loss | 0.44655486814511575 | 0.4447665798493376 |
| eval_mask_mean_token_micro_terminal_accuracy | 0.8514687573621602 | 0.8514659382145727 |
| eval_mask100_token_micro_aa_loss | 1.6932334169691867 | 1.6928995481811533 |
| eval_mask100_token_micro_aa_accuracy | 0.43771572730921393 | 0.4370850295281234 |
| eval_mask100_token_micro_terminal_loss | 1.3528858718768357 | 1.3428790921457343 |
| eval_mask100_token_micro_terminal_accuracy | 0.5438683948155534 | 0.5448654037886341 |
| eval_mask100_token_micro_first_terminal_accuracy | 0.16650049850448653 | 0.16201395812562314 |
| eval_mask100_token_micro_terminal_missing_rate | 0.0 | 0.0 |
| eval_mask100_token_micro_length_mae | 3.060817547357926 | 3.073778664007976 |
| eval_mask100_token_micro_terminal_precision | 0.3642737896494157 | 0.36036927134849983 |
| eval_mask100_token_micro_terminal_false_positive_rate | 0.010916805229057967 | 0.011123215411960324 |
| eval_mask100_token_micro_length_compared_count | 2006 | 2006 |
| eval_mask100_token_micro_chain_count | 2006 | 2006 |

主eval_loss略升，平均AA CE和全遮蔽AA/END CE下降；全遮蔽AA准确率、first-END、长度MAE及END precision略退，不能概括为全面改善。继续固定50082评测，不按该点改checkpoint或解码。保留完整45000/46000/47000（latest与best3重合），44000权重由原策略轮替、JSON证据保留。此为训练内固定验证，非下游生成成绩。


## 2026-10-06T21:32:18.621472+00:00 正式46000固定验证与完整恢复保存

证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint46000_root_review.json（953224 rc0，SHA9887798690ce93740185301a56057fab2914505f81781546bfb25d21fce5c216）；当前保留44000/45000/46000，latest与best3重合。45058权重正常轮换、JSON证据保留。

| 指标 | 45058 | 46000 |
|---|---:|---:|
| eval_loss | 1.8655790374361092 | 1.862781762676079 |
| eval_mask_mean_token_micro_aa_loss | 0.701955342490939 | 0.7011582623495837 |
| eval_mask_mean_token_micro_aa_accuracy | 0.7827942710499857 | 0.7830569929080398 |
| eval_mask_mean_token_micro_terminal_loss | 0.45888131266614485 | 0.44655486814511575 |
| eval_mask_mean_token_micro_terminal_accuracy | 0.8470666715315139 | 0.8514687573621602 |
| eval_mask100_token_micro_aa_loss | 1.696012761587485 | 1.6932334169691867 |
| eval_mask100_token_micro_aa_accuracy | 0.43646579897941634 | 0.43771572730921393 |
| eval_mask100_token_micro_terminal_loss | 1.380817137680315 | 1.3528858718768357 |
| eval_mask100_token_micro_terminal_accuracy | 0.5289132602193419 | 0.5438683948155534 |
| eval_mask100_token_micro_first_terminal_accuracy | 0.16201395812562314 | 0.16650049850448653 |
| eval_mask100_token_micro_terminal_missing_rate | 0.0 | 0.0 |
| eval_mask100_token_micro_length_mae | 3.370887337986042 | 3.060817547357926 |
| eval_mask100_token_micro_terminal_precision | 0.372280701754386 | 0.3642737896494157 |
| eval_mask100_token_micro_terminal_false_positive_rate | 0.010257439367008772 | 0.010916805229057967 |
| eval_mask100_token_micro_length_compared_count | 2006 | 2006 |
| eval_mask100_token_micro_chain_count | 2006 | 2006 |

平均和全遮蔽重建及长度MAE改善，END precision下降且误报增加；这是固定验证，非下游生成。未改架构或配方。



## 2026-10-06T21:22:17.810232+00:00 45058独立恢复保存与真实8步验收通过

/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint45058_root_review.json（SHAf2ef99d3eccc2276169b1d3b5fa9056a9c689e8b38793a6762ba499fe279e337，833ade rc0）证明17文件/615源码/112计数/70聚合/26结束指标；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/segment45058_root_review.json（e43403 rc0）另证明实际8更新、8成功退出/清理和24项原生恢复。

| 指标 | 45050 | 45058 |
|---|---:|---:|
| 主eval_loss | 1.865641 | 1.865579 |
| 平均AA CE | 0.701833 | 0.701955 |
| 全遮蔽AA CE | 1.695817 | 1.696013 |
| 全遮蔽AA准确率(%) | 43.675248 | 43.646580 |
| 全遮蔽END准确率(%) | 51.645065 | 52.891326 |
| 首END准确率(%) | 16.849452 | 16.201396 |
| 长度MAE | 3.340977 | 3.370887 |
| END precision | 0.391979 | 0.372281 |
| AA误报END率(%) | 0.921392 | 1.025744 |

主loss微降、全遮蔽END位置准确率改善，但AA CE/准确率、首END、长度误差和END误报均有回退。仅8步恢复节点，不据此调整模型或声称生成收益。新长段继续50058，累计50082后的生成尚未开始；最佳3份为45000/44000/45058，最新重合，均完整恢复状态。

## 2026-10-06T20:45:09.009724+00:00 45050完整段通过；下一50050段启动锁超时失败

40050→45050的5000更新已完成：8个pool实际rc0/child退出与进程组清理、24项optimizer/scheduler/各rank RNG恢复通过。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/segment45050_root_review.json（SHA50675bf75f5322f5e792c4b687df077507b00516450ebba1b29d03c5bdb171be，f364e2 rc0）。整段3771.846267秒，平均0.7543692534秒/更新，含恢复/验证/保存/退出。该报告明确status=actual_epoch2_45050_segment_passed_next_attempt_aborted，不把下一段入队当作训练成功。

45000与45050原生保存/固定指标聚合均已核查，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint45000_root_review.json（SHAb6d97e678512ba8228239a52959495d9fdfab9a2c4d51b53fac71bfd6d3a5c3d，833fb1 rc0）；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint45050_root_review.json（SHA4821afc13ecb1bebff4b2a226bd408779aafda1e30b877347c6f75fe051f64cf，c3b31e rc0）。

| 指标 | 44000 | 45000 | 45050 |
|---|---:|---:|---:|
| 主eval_loss | 1.864429 | 1.863106 | 1.865641 |
| 平均AA CE | 0.702444 | 0.700918 | 0.701833 |
| 全遮蔽AA CE | 1.695927 | 1.695324 | 1.695817 |
| 全遮蔽AA准确率(%) | 43.778453 | 43.748638 | 43.675248 |
| 全遮蔽END准确率(%) | 55.383848 | 53.140578 | 51.645065 |
| 首END准确率(%) | 16.849452 | 16.400798 | 16.849452 |
| 长度MAE | 3.296112 | 3.349452 | 3.340977 |
| END precision | 0.365581 | 0.372467 | 0.391979 |
| AA误报END率(%) | 1.105441 | 1.029757 | 0.921392 |

45000较44000主loss与平均/全遮蔽AA CE改善，但END位置/首END准确率与长度回退。45050多50个更新后主loss、AA重建和真实END位置准确率回退，首END、长度MAE、precision与误报改善；三点漏END均0，长度分母均2006。当前最低主loss为45000的1.863106227，最终保留best3=45000、44000、45050，latest45050重合，共3份完整恢复状态。父24000与41000权重已正常轮替，历史指标和恢复证据保留。未完成两轮，不能将这些点替代固定50082下游评测。

下一50050段20:38Z实际启动后，因Coordinator lock wait expired中止，8个pool失败结果和8个child清理已核实。训练监督20:38:22Z rc1退出，生成监督20:38:24Z因上游停止rc1退出，两个真实PID均缺席。没有45050之后的新训练更新，9项CDR未开始。故障 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/segment50050_failure_root_observation.json；启动锁分析 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/segment50050_lock_diagnosis.json（SHAb11ce534bfb665ce547d5204c0baa054dfd5d5bce2b5529e15d5befe2a70a158，0f40f5 rc0）：474条完成计时中，两次heartbeat持锁2.364763/2.350461秒超出2秒等待阈值，最长一次提交阶段2.362406秒；3次heartbeat锁超时，首次abort来自rank7。尚无系统调用证据确定具体文件操作或调度延迟，不能直接归因GPU/模型/数据。详细工程交接 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_STARTUP_LOCK_FAILURE_20261006.md。

现有8worker待命且心跳新鲜，本轮未重启、重试、重提或删除失败标记。原数据门禁、安全停线和CUDA暂停保持。本地f010/42当前3400/3750、已核查3250、下一3500；22格9完整产物/1运行/12未启动。原Goal active，本轮完成新验收与故障定位，完整两轮/生成/消融目标未完成。

## 2026-10-06T20:26:12.396130+00:00 44000主loss成为当前最佳，多项改善但END误报增加

原八卡当前44095/50082，40050→45050正常、abort=null。44000新原生保存的17文件/8rank摘要、615源码/transition、epoch/50082日程、112计数/70核心聚合/26结束聚合及best3＋latest检查通过（569a1c rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint44000_root_review.json（SHA51067a59e5eee0d2fb4f9b3c501e267f572f48102283c902592637390d3205d1）。仅读取现有新保存与验证元数据，没有额外模型测试。

| 指标 | 43000 | 44000 |
|---|---:|---:|
| 主eval_loss | 1.866805 | 1.864429 |
| 平均AA CE | 0.704133 | 0.702444 |
| 全遮蔽AA CE | 1.700102 | 1.695927 |
| 全遮蔽AA准确率(%) | 43.578350 | 43.778453 |
| 全遮蔽END准确率(%) | 52.442672 | 55.383848 |
| 首END准确率(%) | 16.650050 | 16.849452 |
| 长度MAE | 3.305583 | 3.296112 |
| END precision | 0.370945 | 0.365581 |
| AA误报END率(%) | 1.022877 | 1.105441 |

较43000，主loss、平均/全遮蔽AA CE和准确率、END CE/真实位置准确率、首END准确率及长度MAE改善；END precision下降，AA误报END率上升。两点漏END均0、长度比较分母均2006。主loss1.864429053低于前最佳41000的1.865988852，为当前候选最低值，但不据单点声称统计显著收益。较25050起点，平均AA CE0.702444优于0.704848，全遮蔽AA准确率43.778453%略高于43.761826%；全遮蔽AA CE1.695927仍略高于1.693868，长度MAE3.296112仍高于3.176969，首END及误报指标未全面更优。整体生成能力仍需固定50082后的下游评测。

保留策略已实际更新：best3为44000、41000、24000，latest44000与best重合，因此并集3份完整恢复状态。25050和43000权重按既定规则轮替，第二轮起点指标、原生receipt归档和必要证据继续保留；不额外复制权重。固定50082评测点不随best变化。下一45000常规保存，再核实45050段尾及既有监督接续50050。

本地f010/42为3325/3750，最近已核查3250、下一3500；22格9完整产物/1运行/12未启动，图PARTIAL，9项CDR未开始。原Goal active，上轮/本轮progress，完整范围/累计用量及全部安全停线/数据门禁/CUDA暂停保持，无提交/重启或新测试权重。

## 2026-10-06T20:13:57.241107+00:00 43000主loss与重建回升，长度误差仍回退

原八卡当前43080/50082，40050→45050正常、abort=null。43000新原生保存的17文件/8rank摘要、615源码/transition、epoch/50082日程、112计数/70核心聚合/26结束聚合和best3＋latest检查通过（007808 rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint43000_root_review.json（SHAf7fdaa6181239129d85ea8dfb95b4d03c0f8a561468c1604ef7587e8527d2baf）。仅读取新保存/验证元数据，没有额外模型运行。

| 指标 | 42000 | 43000 |
|---|---:|---:|
| 主eval_loss | 1.877881 | 1.866805 |
| 平均AA CE | 0.705883 | 0.704133 |
| 全遮蔽AA CE | 1.707583 | 1.700102 |
| 全遮蔽AA准确率(%) | 43.479158 | 43.578350 |
| 全遮蔽END准确率(%) | 49.002991 | 52.442672 |
| 首END准确率(%) | 14.107677 | 16.650050 |
| 长度MAE | 3.138086 | 3.305583 |
| END precision | 0.361132 | 0.370945 |
| AA误报END率(%) | 0.997076 | 1.022877 |

较42000，主loss、平均/全遮蔽AA CE及准确率、真实END位置与首END准确率、END precision改善；长度MAE与AA误报END略差，平均END CE也略升。两点漏END均0、长度比较分母均2006。平均AA CE0.704133略优于25050的0.704848，但全遮蔽AA CE1.700102与准确率43.578350%仍未优于25050的1.693868/43.761826%；主loss仍高于当前最佳41000的1.865988852。不能将多数指标较上个点恢复当作全面生成收益。

best3仍41000、24000、25050，加latest43000共4份完整恢复状态；42000权重正常轮替，JSON及必要证据保留。保持原冻结配置和日程，下一44000；45050段尾由现有监督验收接续，固定50082后的9项CDR尚未开始。本地f010/42为3275/3750，最新已核查验证3250，下一3500；22格9完整产物/1运行/12未启动，图PARTIAL。原Goal active，上轮/本轮progress，完整范围/累计用量和全部安全停线/数据门禁/CUDA暂停保持，无提交/重启或测试权重。

## 2026-10-06T20:02:27.061757+00:00 42000主loss与结束位置回退，平均AA CE略改善

原八卡当前42130/50082，40050→45050正常、abort=null。42000新原生保存的17文件/8rank摘要、615源码/transition、epoch/50082日程、112计数/70核心聚合/26结束聚合和best3＋latest核查通过（c2f3ab rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint42000_root_review.json（SHAf812421e42e06e538e3cf192137cf6ddc6e0233e9aff56491cd3701a0718e7b3）。仅读取已产生的保存/验证元数据，没有额外模型测试。

| 指标 | 41000 | 42000 |
|---|---:|---:|
| 主eval_loss | 1.865989 | 1.877881 |
| 平均AA CE | 0.707093 | 0.705883 |
| 全遮蔽AA CE | 1.702583 | 1.707583 |
| 全遮蔽AA准确率(%) | 43.653460 | 43.479158 |
| 全遮蔽END准确率(%) | 54.735793 | 49.002991 |
| 首END准确率(%) | 16.550349 | 14.107677 |
| 长度MAE | 2.877866 | 3.138086 |
| END precision | 0.373596 | 0.361132 |
| AA误报END率(%) | 1.055559 | 0.997076 |

较41000，平均AA CE和准确率略改善，但主loss、全遮蔽AA CE/准确率、真实END位置/首END准确率和长度MAE回退。END precision下降，AA误报END率略降；两点漏END均0、长度比较分母均2006，因此长度回退不是分母变化。当前结果显示训练中验证波动，不能据平均AA CE单项或执行正常宣称第二轮整体改善，也不据单点改变冻结架构/日程。

best3仍为41000、24000、25050，加上latest42000，实际保留4份完整恢复状态；各点指标和恢复证据保留。固定50082下游评测点不变，9项CDR尚未开始。下一43000，45050段尾由原监督验收接续。本地f010/42为3250/3750，最新完整验证3000，下一3250；22格9完整产物/1运行/12未启动，图PARTIAL。原Goal active，上轮/本轮progress，完整范围/累计用量、全部安全停线/数据门禁/CUDA暂停保持，无提交/重启或新测试权重。

## 2026-10-06T19:49:54.390189+00:00 41000主loss进入当前最佳，AA改善但END位置准确率回落

原八卡当前41097/50082，40050→45050正常、abort=null。41000新原生保存的17文件/8rank摘要、615源码/transition、epoch/50082日程、112固定计数/70核心聚合/26结束聚合与best3＋latest核查通过（2883d4 rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint41000_root_review.json（SHA209d749b9b2c48c7d7ad4e207b1fcccc25835c11fd52dd84c1a3c9f23003f577）。仅读取新保存元数据，没有额外模型测试。

| 指标 | 40050 | 41000 |
|---|---:|---:|
| 主eval_loss | 1.872695 | 1.865989 |
| 平均AA CE | 0.708743 | 0.707093 |
| 全遮蔽AA CE | 1.708164 | 1.702583 |
| 全遮蔽AA准确率(%) | 43.411502 | 43.653460 |
| 全遮蔽END准确率(%) | 56.430708 | 54.735793 |
| 首END准确率(%) | 17.647059 | 16.550349 |
| 长度MAE | 2.895314 | 2.877866 |
| END precision | 0.361085 | 0.373596 |
| AA误报END率(%) | 1.148443 | 1.055559 |

较40050，主loss、平均/全遮蔽AA CE及准确率改善，长度MAE略降、END precision提高、AA误报END减少；平均/全遮蔽END CE变差，真实END位置及首END准确率下降。两点漏END均0、长度比较分母均2006。当前主loss1.865988852低于父24000的1.866277015及第二轮起点25050的1.866348194，为当前保留候选中的最低值；差值很小，不等于统计显著改善。平均AA CE0.707093、全遮蔽AA CE1.702583仍高于25050的0.704848/1.693868，全遮蔽AA准确率43.653460%仍略低于25050的43.761826%，不将主loss排序当作整体生成收益。

保留策略已实际更新：best3为41000、24000、25050，latest41000与best重合，故并集为3份完整恢复状态。40050与父25000权重按既定规则轮替，原始指标JSON及40050整段恢复/退出证据保留；未手动删正式恢复状态。此排序不改变固定50082下游评测点，9项CDR仍待累计两轮和实际训练退出。

原配置/日程保持，下一42000，45050段尾按原监督验收接续。本地f010/42为3175/3750，最新完整验证3000，下一3250；22格9完整产物/1运行/12未启动，图PARTIAL。原Goal active，上轮/本轮progress，完整范围/累计用量与全部安全停线、数据门禁、CUDA暂停保持，无提交/重启或新测试权重。

## 2026-10-06T19:36:40.075718+00:00 40050整段通过并接续45050；结束位置改善，AA与误报仍有代价

原八卡35050→40050的5000更新已完成，8个pool实际退出码0、8个训练child退出/进程组清理和24项optimizer/scheduler/各rank RNG恢复核查通过（c5d308 rc0）。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/segment40050_root_review.json（SHAfa75e09146c9f05710bf242588272b4f8f2734923f77f8c1973c2c33f6e59395）。前置35050权重按保留策略已轮替，核对使用已验收原生receipt归档，不重新加载已删权重。现有监督已单次接续40050→45050，最新计时40050/50082、abort=null；下一段完整恢复/退出验收仍待实际结束。

本5000更新整段3770.872869秒，平均0.7541745738秒/步（含初始化/恢复、验证、保存和八rank退出），上一段0.7540041532秒/步，实际速度保持接近。未新增测速实验。

40000和40050各自通过17原生文件/8rank摘要、615源码/transition、epoch/50082日程、112计数/70核心聚合/26结束聚合及best3＋latest核查。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint40000_root_review.json（SHA6184749d81f8b5b0707a042f889cb85b9d8d8f7b554909707c0cdb13adf572bd，a4a927 rc0）；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint40050_root_review.json（SHA870eb2f8b59b8e3b5cf58dbc9cf6a234bd5854a31bd3f494c6e63f19949691f3，788212 rc0）。

| 指标 | 39000 | 40000 | 40050 |
|---|---:|---:|---:|
| 主eval_loss | 1.874295 | 1.873590 | 1.872695 |
| 平均AA CE | 0.708813 | 0.708121 | 0.708743 |
| 全遮蔽AA CE | 1.705025 | 1.705839 | 1.708164 |
| 全遮蔽AA准确率(%) | 43.714810 | 43.613898 | 43.411502 |
| 全遮蔽END准确率(%) | 51.495513 | 53.589232 | 56.430708 |
| 首END准确率(%) | 17.347956 | 17.298106 | 17.647059 |
| 长度MAE | 4.036391 | 3.404786 | 2.895314 |
| END precision | 0.379500 | 0.371587 | 0.361085 |
| AA误报END率(%) | 0.968408 | 1.042371 | 1.148443 |

40000较39000：主loss/平均AA CE/长度MAE下降、真实END准确率提升；全遮蔽AA CE/准确率略回退，END误报增多。40050比40000多50个训练更新后，主loss、真实END位置和首END准确率、长度MAE继续改善；AA CE与准确率回退、END precision下降且误报继续增加。三点长度比较分母均2006、漏END均0。较25050起点，40050长度与部分END指标改善，但主loss和AA重建仍未全面更优；不能将段尾单点或保存成功称作整体生成收益。

best3父24000/25000、新25050＋latest40050；39000、40000权重正常轮替，归档JSON/段恢复/退出证据保留。原冻结配置和日程保持，下一41000；固定50082后的9项CDR尚未运行。本地f010/42为3125/3750，最新完整验证3000，下一3250；22格9完整产物/1运行/12未启动，图PARTIAL。原Goal active，上轮/本轮progress，完整范围/累计用量及全部安全停线、数据门禁、CUDA暂停保持。

## 2026-10-06T19:23:48.395378+00:00 39000准确率回升，但AA CE与长度误差变差

原八卡当前39145/50082，35050→40050正常、abort=null。39000新原生保存的17文件/8rank摘要、615源码/transition、epoch/50082日程、112固定计数/70核心聚合/26结束聚合及best3＋latest检查通过（9bdef6 rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint39000_root_review.json（SHAae8fbe48792161eab0aa4fc64ef9a347abdf628dcb35bd390c12cb329256e5e8）。仅核查新保存和已产生的验证元数据，没有额外模型测试。

| 指标 | 38000 | 39000 |
|---|---:|---:|
| 主eval_loss | 1.875248 | 1.874295 |
| 平均AA CE | 0.706701 | 0.708813 |
| 全遮蔽AA CE | 1.700618 | 1.705025 |
| 全遮蔽AA准确率(%) | 43.390287 | 43.714810 |
| 全遮蔽END准确率(%) | 48.654038 | 51.495513 |
| 首END准确率(%) | 16.251246 | 17.347956 |
| 长度MAE | 3.036391 | 4.036391 |
| END precision | 0.382895 | 0.379500 |
| AA误报END率(%) | 0.901898 | 0.968408 |

相较38000，主loss、全遮蔽AA/真实END位置准确率及首END准确率改善；平均/全遮蔽AA CE、长度MAE和END误报变差，precision略降。两点漏END均0、长度比较分母均2006，长度MAE的增长不是分母变化。归档中的分来源记录显示，OAS的长度MAE5.810547→8.771484（512条链、绝对误差和增加1516），ASD antibody为2.349030→3.814404（361条链、绝对误差和增加529）；其余来源合计减少39，净增加2006，对应整体MAE增加1。仅凭汇总不能判断具体样本的误差分布或原因。只读分组查询d0c809 rc0；前次f9148f因误把无来源的总指标当来源键而KeyError，已改为使用绑定的7个validation_indices来源，未改原结果或重跑模型。

较第二轮起点25050，主loss、AA CE和全遮蔽AA准确率仍未全面优于该点；单次argmax诊断不等于下游生成收益。原冻结配置/日程保持，下一40000常规保存，40050段尾由现有监督实际验收并接续45050。best3父24000/25000、新25050＋latest39000，38000权重按既定策略轮替、JSON证据保留。

本地f010/42当前3075/3750，最新完整验证3000、下一3250。22格9完整产物/1运行/12未启动，图PARTIAL；固定50082后的9项CDR未开始。原Goal active，上轮/本轮progress，完整范围/累计用量、全部安全停线和数据门禁保持，无提交/重启或新测试权重。

## 2026-10-06T19:11:07.549687+00:00 38000重建与长度改善，真实END位置准确率下降

正式八卡当前38090/50082，35050→40050正常、abort=null。38000新原生保存的17文件/8rank摘要、615源码/transition、epoch/50082日程、112固定计数/70核心聚合/26结束聚合及best3＋latest检查通过（708c25 rc0）。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint38000_root_review.json（SHA0edef80ceba0b3896190471de8e9eb271602f3254b0e018f097a3d0ba8c8478b），没有重复模型测试。

| 指标 | 25050 | 37000 | 38000 |
|---|---:|---:|---:|
| 主eval_loss | 1.866348 | 1.877963 | 1.875248 |
| 平均AA CE | 0.704848 | 0.713145 | 0.706701 |
| 全遮蔽AA CE | 1.693868 | 1.717913 | 1.700618 |
| 全遮蔽AA准确率(%) | 43.761826 | 43.228599 | 43.390287 |
| 全遮蔽END准确率(%) | 54.187438 | 55.832502 | 48.654038 |
| 首END准确率(%) | 17.298106 | 15.054835 | 16.251246 |
| 长度MAE | 3.176969 | 3.108175 | 3.036391 |
| END precision | 0.373411 | 0.363400 | 0.382895 |
| AA误报END率(%) | 1.045812 | 1.124935 | 0.901898 |

相较37000，主loss与平均/全遮蔽AA CE下降、AA准确率略升。首END准确率、长度MAE、END precision和AA误报END率改善；两点漏END均0、长度分母均2006。但是平均/全遮蔽END CE变差，平均END准确率85.2756%→83.4953%、全遮蔽END准确率55.8325%→48.6540%下降。平均AA CE0.706701已接近25050的0.704848，但主loss、AA CE和AA准确率尚未恢复至该起点，不宣称整体生成收益。

best3父24000/25000、新25050＋latest38000，37000权重正常轮替、JSON证据保留。原配置与日程保持，下一39000，40050段尾按既有监督验收并接续。固定50082后的9项CDR尚未运行。本地f010/42为3025/3750，最新验证3000、下一3250；22格9完整产物/1运行/12未启动，图PARTIAL。原Goal active，本轮progress，完整范围/累计用量及全部安全停线、数据门禁、CUDA暂停保持，无提交/重启或新测试权重。


## 2026-10-06T18:58:47.645501+00:00 37000主loss与END略改善，AA CE和长度误差略回退

原八卡第二轮当前37083/50082，35050→40050正常、abort=null。37000新原生保存的17文件/8rank摘要、615源码/transition、epoch/50082日程、112固定计数/70核心聚合/26结束聚合和best3＋latest核查通过（8b665a rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint37000_root_review.json（SHAf47088a40e4a9fa55bd2ca992b73cc315ce2dd5a833e7ca399f7389e48b1c833），没有模型重跑。

| 指标 | 25050 | 36000 | 37000 |
|---|---:|---:|---:|
| 主eval_loss | 1.866348 | 1.881713 | 1.877963 |
| 平均AA CE | 0.704848 | 0.712647 | 0.713145 |
| 全遮蔽AA CE | 1.693868 | 1.713322 | 1.717913 |
| 全遮蔽AA准确率(%) | 43.761826 | 43.176423 | 43.228599 |
| 全遮蔽END准确率(%) | 54.187438 | 53.539382 | 55.832502 |
| 首END准确率(%) | 17.298106 | 14.556331 | 15.054835 |
| 长度MAE | 3.176969 | 3.062313 | 3.108175 |
| END precision | 0.373411 | 0.352941 | 0.363400 |
| AA误报END率(%) | 1.045812 | 1.128949 | 1.124935 |

相较36000，主loss略降、平均/全遮蔽AA准确率略升，但平均AA CE0.712647→0.713145、全遮蔽AA CE1.713322→1.717913变差。END CE、真实END位置准确率、首END准确率与precision改善，误报率略降；长度MAE3.062313→3.108175略差（漏END均0、分母均2006）。与25050相比主loss和AA CE仍更差，不能称第二轮整体收益已确认。

best3父24000/25000、新25050＋latest37000；36000权重正常轮替、JSON证据保留。保持冻结配置和日程，下一38000常规验证，40050段尾由原监督验收后接续。本地f010/42当前2975/3750，最新完整验证2750，下一3000；22格9完整产物/1运行/12未启动，图PARTIAL。50082后的9项CDR未开始。原Goal active、本轮progress，完整范围/累计用量和全部安全停线/数据门禁/CUDA暂停保持，无提交/重启或新测试权重。


## 2026-10-06T18:47:06.719394+00:00 36000 AA CE下降但准确率回退，首END与误报仍不稳定

原八卡第二轮当前36128/50082，35050→40050正常、abort=null；监督真实身份及8worker新鲜心跳核实。36000新原生保存的17文件/8rank摘要、615源码/transition、epoch/50082日程、112计数/70核心聚合/26结束聚合及best3＋latest检查通过（8f5afd rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint36000_root_review.json（SHA71bfbd15c68576398445faa8f8984f128f922cac8a8d16a32fc2b2cbdb27fcfa），未重复模型测试。

| 指标 | 25050 | 35050 | 36000 |
|---|---:|---:|---:|
| 主eval_loss | 1.866348 | 1.881045 | 1.881713 |
| 平均AA CE | 0.704848 | 0.716656 | 0.712647 |
| 全遮蔽AA CE | 1.693868 | 1.713822 | 1.713322 |
| 全遮蔽AA准确率(%) | 43.761826 | 43.543375 | 43.176423 |
| 全遮蔽END准确率(%) | 54.187438 | 53.240279 | 53.539382 |
| 首END准确率(%) | 17.298106 | 17.248255 | 14.556331 |
| 长度MAE | 3.176969 | 3.053838 | 3.062313 |
| END precision | 0.373411 | 0.378053 | 0.352941 |
| AA误报END率(%) | 1.045812 | 1.007396 | 1.128949 |

相较35050，主loss略升0.000668，平均/全遮蔽AA CE下降，但平均AA准确率77.9795%→77.9529%、全遮蔽AA准确率43.5434%→43.1764%下降。平均/全遮蔽END CE及真实位置准确率改善，但首END准确率17.2483%→14.5563%下降，END precision0.378053→0.352941下降、AA误报END率1.0074%→1.1289%增加。两点漏END均0、长度分母均2006，长度MAE3.053838→3.062313略差。主loss与AA CE较25050仍未恢复，不能仅凭CE下降认定整体生成收益。

best3父24000/25000、新25050＋latest36000；35050权重正常轮替，完整段退出/恢复及指标JSON证据保留。原冻结配置与日程保持，下一常规验证37000，40050段尾由既有监督验收并接续。固定50082后的9项CDR尚未开始。

本地f010/42为2925/3750，最新验证2750已对照、下一3000；22格9完整产物/1运行/12未启动，图PARTIAL。原Goal active，本轮progress，完整范围/累计用量保持；所有安全停线/数据门禁/CUDA暂停不变，无提交/重启或新测试权重。


## 2026-10-06T18:34:57.137689+00:00 35050完整段通过并接续40050；35000回退、段尾多项回升

原八卡30050→35050的5000更新已完成，8个pool实际退出码0、8个训练child退出/进程组清理及24项optimizer/scheduler/各rank RNG恢复核查通过（11e587 rc0）。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/segment35050_root_review.json（SHA9c4f58a5e477e48e11d5a541d492bca180fc99521d908f3785690f0398717273）。前置30050权重已按best3＋latest正常轮替，本次使用其已验收原生receipt归档逐项核对恢复记录，没有重新加载已删权重、补造回执或重跑模型。

整段enqueue到实际退出验收耗时3770.020766秒（62分50.02秒）、平均0.7540041532秒/步，含初始化恢复、验证、保存与退出；上一5000段0.7519159536秒/步。真实长段吞吐保持接近，非新增测速实验，也不等于效果改善。现有监督已单次接续35050→40050，当前35115/50082、abort=null；所有原始数据/来源门禁保持。

35000和35050分别通过17原生文件/8rank保存摘要、615源码/transition、epoch/50082日程、112计数/70核心聚合/26结束聚合及best3＋latest核查。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint35000_root_review.json（SHAe2c85c456f4e5ad39b18d697f72b1ce8f701ca651eede093542ba11b81e948f7，5ab037 rc0）；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint35050_root_review.json（SHA9e4dc65e85c0516cf9be3f748b7666465c901175aacf737601bcd28fcd8f630f，4baff8 rc0）。

| 指标 | 25050 | 34000 | 35000 | 35050 |
|---|---:|---:|---:|---:|
| 主eval_loss | 1.866348 | 1.871820 | 1.895401 | 1.881045 |
| 平均AA CE | 0.704848 | 0.712913 | 0.718372 | 0.716656 |
| 全遮蔽AA CE | 1.693868 | 1.710736 | 1.731800 | 1.713822 |
| 全遮蔽AA准确率(%) | 43.761826 | 43.597844 | 42.078436 | 43.543375 |
| 全遮蔽END准确率(%) | 54.187438 | 54.336989 | 47.507478 | 53.240279 |
| 首END准确率(%) | 17.298106 | 15.304088 | 13.858425 | 17.248255 |
| 长度MAE | 3.176969 | 3.535892 | 3.922233 | 3.053838 |
| END precision | 0.373411 | 0.366017 | 0.363740 | 0.378053 |
| AA误报END率(%) | 1.045812 | 1.082507 | 0.955794 | 1.007396 |

35000相对34000主loss、AA CE/准确率、END/首END与长度MAE均回退，只有误报率减少等局部改善。35050只多50更新，多项重建与结束指标较35000回升，但AA误报END率0.9558%→1.0074%增大；不挑段尾较好数字替代35000。上述四点漏END均0、长度分母均2006。35050主loss/AA CE仍差于25050，且相对34000也尚未恢复；不能宣布第二轮整体收益。固定50082下游评测仍待真实训练终态。

best3父24000/25000、新25050＋latest35050；34000/35000权重已按既定策略轮替，全部核查和JSON指标证据保留。下一36000常规验证，40050段末再核实实际退出/接续。当前本地f010/42为2875/3750、已核查验证2750、下一3000；22格9完整产物/1运行/12未启动。原Goal active，上轮/本轮progress，完整范围和累计用量保持，安全停线/数据门禁/CUDA暂停不变；无提交/重启或新测试权重。


## 2026-10-06T18:20:54.761969+00:00 34000重建继续小幅改善，结束位置与长度指标回退

原八卡第二轮当前34152/50082，30050→35050正常、abort=null，监督真实身份及8worker新鲜心跳核实。34000新原生保存的17文件/8rank摘要、615源码/transition、epoch/50082日程、112计数/70核心聚合/26结束聚合及best3＋latest检查通过（38093c rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint34000_root_review.json（SHA756ef47ce5b6621c6f513a60b3ae0ebfb2654b0e03f7f3af890a5cb46d39ba29），没有重复模型测试。

| 指标 | 25050 | 33000 | 34000 |
|---|---:|---:|---:|
| 主eval_loss | 1.866348 | 1.872551 | 1.871820 |
| 平均AA CE | 0.704848 | 0.714318 | 0.712913 |
| 全遮蔽AA CE | 1.693868 | 1.713657 | 1.710736 |
| 全遮蔽AA准确率(%) | 43.761826 | 43.349005 | 43.597844 |
| 全遮蔽END准确率(%) | 54.187438 | 58.973081 | 54.336989 |
| 首END准确率(%) | 17.298106 | 17.298106 | 15.304088 |
| 长度MAE | 3.176969 | 2.613161 | 3.535892 |
| END precision | 0.373411 | 0.354935 | 0.366017 |
| AA误报END率(%) | 1.045812 | 1.232727 | 1.082507 |

相较33000，主loss、平均/全遮蔽AA CE及AA准确率改善；但平均/全遮蔽END CE变差、准确率下降，首END准确率17.2981%→15.3041%、长度MAE2.613161→3.535892回退（两点漏END均0/2006、长度分母均2006）。END precision改善、AA误报END率下降，仍不能据单类指标判断整体生成收益。主loss和AA CE较25050起点仍更差，固定两轮后的下游评测尚未发生。

best3父24000/25000、新25050＋latest34000，33000权重正常轮替、JSON证据保留。配置与日程保持，下一35000常规验证；随后35050有段尾保存/实际退出，接近时缩短只读观察间隔，避免正常保留策略轮替后误称中间checkpoint缺失为训练故障。现有监督负责接续，不手动重复入队。

本地f010/42为2825/3750，最新验证2750已核查、下一3000；22格9完整产物/1运行/12未启动，图PARTIAL。原Goal active、本轮progress，完整范围/累计用量及全部安全停线/数据门禁/CUDA暂停保持。没有提交/重启或新测试权重，9项CDR仍等待50082/两轮/实际退出。


## 2026-10-06T18:08:27.169153+00:00 33000主loss/AA CE和长度误差改善，AA准确率与END误报仍分化

原八卡第二轮当前33132/50082，30050→35050正常、abort=null，现有监督真实身份及8worker新鲜心跳核实。33000新保存的原生17文件/8rank摘要、615源码/transition、epoch/50082日程、112计数/70核心聚合/26结束聚合及best3＋latest检查通过（1d68ef rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint33000_root_review.json（SHAcc59b13f33e4e2c9310688c11ffd0fc6a3a6f3ebfaafab7ab11b71d5d621cbea），未重复模型验收。

| 指标 | 25050 | 32000 | 33000 |
|---|---:|---:|---:|
| 主eval_loss | 1.866348 | 1.889075 | 1.872551 |
| 平均AA CE | 0.704848 | 0.716941 | 0.714318 |
| 全遮蔽AA CE | 1.693868 | 1.718900 | 1.713657 |
| 全遮蔽AA准确率(%) | 43.761826 | 43.448197 | 43.349005 |
| 全遮蔽END准确率(%) | 54.187438 | 52.293121 | 58.973081 |
| 首END准确率(%) | 17.298106 | 17.447657 | 17.298106 |
| 长度MAE | 3.176969 | 3.134596 | 2.613161 |
| END precision | 0.373411 | 0.380624 | 0.354935 |
| AA误报END率(%) | 1.045812 | 0.978728 | 1.232727 |

主eval_loss、平均/全遮蔽AA CE较32000继续下降，但全遮蔽AA准确率43.4482%→43.3490%略降。真实END位置准确率52.2931%→58.9731%上升，长度MAE3.134596→2.613161改善（两点漏END均0、分母均2006）；但首END准确率17.4477%→17.2981%下降，END precision0.380624→0.354935下降、AA误报END率0.9787%→1.2327%增加。较25050起点，主loss与AA CE仍更差。不能把CE和长度改善当作整体生成质量已提高，继续固定两轮后的真实下游评测。

best3父24000/25000、新25050＋latest33000，32000权重正常轮替，JSON证据保留。冻结配置和日程保持，下一常规验证34000，35050段末由既有监督核实退出再接续；50082后的9项CDR尚未运行。本地f010/42为2775/3750，最新已核查验证2750、下一3000；22格9完整产物/1运行/12未启动。原Goal active，本轮progress、完整范围与累计用量保持，安全停线/数据门禁/CUDA暂停不变，无提交/重启或测试权重。


## 2026-10-06T17:56:42.214267+00:00 32000重建与首END改善，漏检清零；真实END准确率回落

原八卡第二轮当前32144/50082，30050→35050正常运行；现有监督实际身份与8worker新鲜心跳已核实，abort=null。32000新原生保存的17文件/8rank摘要、615源码/transition、epoch/50082日程、112固定计数/70核心聚合/26结束聚合和best3＋latest检查通过（b98f5b rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint32000_root_review.json（SHAc8ab29a60af7bc5f3086ef32795f8fc9d308e2e340af40ccf0df89ad30c25c94），没有重复模型验收。

| 指标 | 25050 | 31000 | 32000 |
|---|---:|---:|---:|
| 主eval_loss | 1.866348 | 1.897838 | 1.889075 |
| 平均AA CE | 0.704848 | 0.722642 | 0.716941 |
| 全遮蔽AA CE | 1.693868 | 1.734588 | 1.718900 |
| 全遮蔽AA准确率(%) | 43.761826 | 42.831260 | 43.448197 |
| 平均END准确率(%) | 85.038844 | 85.808369 | 84.382570 |
| 全遮蔽END准确率(%) | 54.187438 | 57.627119 | 52.293121 |
| 首END准确率(%) | 17.298106 | 15.952144 | 17.447657 |
| 长度MAE | 3.176969 | 3.245000 | 3.134596 |
| 漏END率(%) | 0.000000 | 0.299103 | 0.000000 |
| END precision | 0.373411 | 0.353733 | 0.380624 |
| AA误报END率(%) | 1.045812 | 1.210940 | 0.978728 |

相较31000，主loss、AA CE与AA准确率改善，首END准确率上升；漏END由6/2006降至0/2006，长度MAE分母2000→2006且误差3.245000→3.134596降低。END precision上升、AA误报END减少。但是平均END准确率85.8084%→84.3826%、全遮蔽END准确率57.6271%→52.2931%下降，相应END CE升高。因此效果仍分化，不能笼统声称生成质量提升。较第二轮起点25050，主loss与AA CE仍更差，AA准确率也尚未恢复。

保留best3父24000/25000、新25050及latest32000四份完整状态；31000权重依原策略轮替，JSON/指标证据保留。冻结架构与日程保持，下一常规验证33000，35050段末由原监督验收后接续。固定50082后的9项CDR尚未运行。

本地f010/42训练2725/3750，最新验证2500，下一2750；22格9完整产物/1运行/12未启动，最终图仍PARTIAL。原Goal active、上一轮与本轮progress，完整目标/累计用量保持。所有安全停线、数据门禁及独立CUDA暂停保持，无新提交/重启或测试checkpoint。交接 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json。


## 2026-10-06T17:44:00.531486+00:00 31000重建回退；长度误差降低但漏END增加

原八卡第二轮当前31101/50082，30050→35050正常运行，abort=null；现有训练/生成监督和8worker实际存活、心跳新鲜。31000的原生17文件/8rank保存摘要、615源码、封存transition、epoch/50082日程、112固定计数/70核心聚合/26结束聚合及best3＋latest核对通过（7b1721 rc0）。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint31000_root_review.json（SHA2807b5135f9ca8bb630ca2708c8a768aaebe42dd280ce0080419ac9b05eabb18）；只核对新产生的保存元数据，没有重跑模型验收。

| 指标 | 25050 | 30050 | 31000 |
|---|---:|---:|---:|
| 主eval_loss | 1.866348 | 1.885701 | 1.897838 |
| 平均AA CE | 0.704848 | 0.720643 | 0.722642 |
| 全遮蔽AA CE | 1.693868 | 1.717349 | 1.734588 |
| 全遮蔽AA准确率(%) | 43.761826 | 43.112780 | 42.831260 |
| 全遮蔽END准确率(%) | 54.187438 | 57.577268 | 57.627119 |
| 首END准确率(%) | 17.298106 | 15.852443 | 15.952144 |
| 长度MAE | 3.176969 | 4.922655 | 3.245000 |
| 漏END率(%) | 0.000000 | 0.099701 | 0.299103 |
| END precision | 0.373411 | 0.325903 | 0.353733 |
| AA误报END率(%) | 1.045812 | 1.369761 | 1.210940 |

相较30050，主loss、平均/全遮蔽AA CE和AA准确率均退步；与25050相比重建指标仍未恢复。真实END位置和首END准确率略升，END precision改善、误报率下降；但漏END从2/2006增至6/2006，长度MAE的计算分母2004→2000。长度MAE4.922655→3.245000不能忽略这4条新增漏检，也不能单独当作整体生成质量改善的证据。

最佳3仍父24000/25000与新25050，另保留latest31000；30050权重按既定策略轮替，段退出、恢复和指标JSON证据保留。保持冻结架构与学习率日程，下一常规验证32000；现有35050段末再核实实际退出。两轮50082及后续9项CDR生成尚未完成。

本地f010/42训练2675/3750、最近已核查2500，下一2750；22格仍9完整产物/1运行/12未启动。原Goal active、本轮progress，完整范围和累计用量保持；全部安全停线、数据门禁、CUDA暂停不变，无新提交/重启或测试权重。交接 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json。


## 2026-10-06T17:32:33.406977+00:00 第二轮首个5000步段完成，已接续35050；结束位置仍不稳定

原八卡25050→30050完成，实际8个pool退出码均0、8个训练child退出/进程组清理回执通过，24项optimizer/scheduler/各rank RNG恢复摘要与保存状态一致。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/segment30050_root_review.json（SHA3dca9004ec0fdb614f5525a9a8d8fe5f5b1b9d806cc1bb2d9c5b87e6b7a57110，8ec528 rc0）。原监督已单次接续30050→35050，当前30156/50082、abort=null；不能把本段完成当作累计两轮完成或连续训练轨迹位级等价。

完整5000更新从16:26:37.018819到17:29:16.598587 UTC，耗时3759.579768秒，平均0.7519159536秒/步；含初始化恢复、验证保存和八rank退出，非新增测速实验。下一段仍使用原f011、micro32/GA1/world8/global256与冻结第二轮日程。

新30000及30050分别通过原生17文件/8rank保存摘要、615源码、epoch与50082日程、112固定计数/70核心聚合/26结束聚合及best3＋latest核对。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint30000_root_review.json（SHAdd4f72d583f597a08aba59d769bbaf5f05e4832093740e15112825ab1c7ff8fc，98b699 rc0）；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint30050_root_review.json（SHA7c6c53e6b22990fb5ba237530cd7e48223a20e0a2a0fbb73db5749a9712a83c5，9e0ce4 rc0）。

| 指标 | 25050 | 29000 | 30000 | 30050 |
|---|---:|---:|---:|---:|
| 主eval_loss | 1.866348 | 1.891922 | 1.888945 | 1.885701 |
| 平均AA CE | 0.704848 | 0.719833 | 0.724029 | 0.720643 |
| 全遮蔽AA CE | 1.693868 | 1.725481 | 1.722199 | 1.717349 |
| 全遮蔽AA准确率(%) | 43.761826 | 43.310590 | 43.033083 | 43.112780 |
| 全遮蔽END准确率(%) | 54.187438 | 53.888335 | 44.366899 | 57.577268 |
| 首END准确率(%) | 17.298106 | 16.201396 | 16.001994 | 15.852443 |
| 长度MAE | 3.176969 | 3.377866 | 3.547904 | 4.922655 |
| 长度MAE分母 | 2006.000000 | 2006.000000 | 2004.000000 | 2004.000000 |
| 漏END率(%) | 0.000000 | 0.000000 | 0.099701 | 0.099701 |
| END precision | 0.373411 | 0.347477 | 0.402715 | 0.325903 |
| AA误报END率(%) | 1.045812 | 1.163924 | 0.756837 | 1.369761 |

30000主loss较29000下降，但平均AA CE、AA准确率及END准确率回退；误报减少却伴随漏END和更高长度误差。30050再增加50步，主loss和AA CE较30000改善，真实END位置准确率44.3669%→57.5773%，但首END准确率16.0020%→15.8524%、长度MAE3.547904→4.922655变差，漏检均2/2006、长度分母均2004；END precision0.402715→0.325903下降、AA误报END率0.7568%→1.3698%增加。不能挑段尾END准确率作为整体生成改善的证据；较第二轮开始25050，主loss和AA CE仍更差。

best3仍父24000/25000及新25050，另保留latest30050。29000和30000权重依原策略轮替，已核实的JSON/指标归档保留；不复制或额外删除正式恢复状态。下一常规验证31000；生成9项仍等待固定50082及真实训练退出。本地f010/42为2650/3750，最近核查2500、下一2750；22格仍9完整产物/1运行/12未启动。原Goal与全范围保持active，无新提交、容器/worker重启或模型测试；原数据门禁、序列注释停线和CUDA暂停保持。


## 2026-10-06T17:18:35.535925+00:00 29000重建指标回升，结束位置与误报仍退步

原八卡第二轮当前29187/50082，25050→30050正常运行，abort=null；8worker新鲜心跳和训练/生成监督进程身份已核实。29000原生17文件、8rank保存摘要、615源码/封存transition、实际epoch、112计数/70核心聚合/26结束聚合及best3＋latest检查通过（c3712f rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint29000_root_review.json（SHA88d981457317fbd2a6634ccfe413280adc28e125be6bce4fea1403581efab790）；这是已生成保存元数据的核对，没有重复模型测试。

| 验证指标 | 25050 | 28000 | 29000 |
|---|---:|---:|---:|
| 主eval_loss | 1.866348 | 1.898669 | 1.891922 |
| 平均AA CE | 0.704848 | 0.728084 | 0.719833 |
| 全遮蔽AA CE | 1.693868 | 1.751431 | 1.725481 |
| 全遮蔽AA准确率(%) | 43.761826 | 43.117940 | 43.310590 |
| 全遮蔽END准确率(%) | 54.187438 | 50.498504 | 53.888335 |
| 首END准确率(%) | 17.298106 | 17.796610 | 16.201396 |
| 长度MAE | 3.176969 | 3.197906 | 3.377866 |

相较28000，主loss、平均/全遮蔽AA CE和AA准确率改善，结束了此前AA CE连续三个新点的回退；但仍未超过25050的重建指标。全遮蔽END准确率上升，首END准确率下降；长度MAE3.197906→3.377866变差（分母均2006、漏检均0），END precision0.374769→0.347477下降、AA误报END率0.9690%→1.1639%增加。因此不能称第二轮整体质量已提高，也不能用保存核查通过替代下游生成效果。

保留最佳父24000/25000、新25050及latest29000四份完整恢复状态；28000权重按既定策略轮替，JSON证据保留。继续原冻结日程，下一常规验证30000；30050实际段退出和接续由现有监督执行，不手动重复入队。下游9项CDR评测仍等待累计两轮50082和实际训练退出，尚未生成结果。

本地f010/42训练2600/3750、已核查验证2500，下一2750；22格仍9完整产物/1运行/12未启动，最终图PARTIAL。原Goal active、身份/累计用量/完整范围保持，本轮新增29000实际结果；安全停线、数据门禁和独立CUDA暂停保持，无平台提交/重启或新测试权重。机器交接 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json。


## 2026-10-06T17:07:09.789953+00:00 28000主loss回落，但AA CE继续退步

原八卡第二轮当前28235/50082、25050→30050正常运行，abort=null；现有监督与8worker新鲜心跳再次核实。28000的原生17文件、8rank保存摘要、615源码/封存transition、实际epoch与50082日程，以及固定1330×4验证的112计数/70核心聚合/26结束聚合和best3＋latest核查通过（c76ce2 rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint28000_root_review.json（SHA1f44edb1a955ab34901036d6f106da39c7b01bb289e16d9ba60d78b5ff9ffbe7），仅核对实际保存元数据，没有新增模型测试。

| 验证指标 | 25050 | 26000 | 27000 | 28000 |
|---|---:|---:|---:|---:|
| 主eval_loss | 1.866348 | 1.887508 | 1.903813 | 1.898669 |
| 平均AA CE | 0.704848 | 0.719752 | 0.726757 | 0.728084 |
| 全遮蔽AA CE | 1.693868 | 1.719164 | 1.734098 | 1.751431 |
| 全遮蔽AA准确率(%) | 43.761826 | 43.360472 | 43.089272 | 43.117940 |
| 全遮蔽END准确率(%) | 54.187438 | 46.410768 | 49.750748 | 50.498504 |
| 首END准确率(%) | 17.298106 | 15.403789 | 14.456630 | 17.796610 |
| 长度MAE | 3.176969 | 3.671820 | 3.061815 | 3.197906 |

主eval_loss较27000改善0.005144，但仍高于25050；平均AA CE和全遮蔽AA CE连续三个新点变差。AA准确率较27000略升，不能据此忽略CE恶化。全遮蔽END和首END准确率回升，长度MAE3.061815→3.197906变差、分母均2006且漏检0；END precision0.384438→0.374769下降，误报0.9162%→0.9690%增加。因此不能声称第二轮整体恢复或已产生收益。

最佳状态仍为父24000/25000及新25050，另保留最新28000，共4份完整恢复状态；27000权重按原策略轮替、JSON证据保留。现有来源/恢复/epoch检查均正常，保持既定日程，不把效果回退称为执行失败或擅自改变训练条件；下一验证29000。已核对的27000 post-warmup实际学习率检查沿用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint27000_schedule_observation.json，本轮没有重复该检查或额外测速。

本地f010/42训练2550/3750、2500对照已核查，下一2750；22格9完整产物/1运行/12未启动，固定末点图仍PARTIAL。累计两轮50082、9项生成和全部必要消融未完成；后续消融包仍待真实终态。原Goal active，原身份/累计用量与完整范围不变，上轮与本轮progress。全部安全停线、数据审核门禁、独立CUDA暂停保持，无提交/重启或新增测试checkpoint。实时交接 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json。



## 2026-10-06T16:56:35.848503+00:00 27000主loss与AA继续回退；本地2500对照已更新

正式八卡当前27366/50082、25050→30050运行，abort=null；现有监督和8worker新鲜心跳核实。27000原生17文件/8rank摘要、615源码、封存transition、实际epoch=1.0782317000119803、固定1330×4验证与112计数/70核心聚合/26结束指标及best3＋latest检查通过（41515会话，9e3509 rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint27000_root_review.json（SHA26aa485b1ccd35eeea44838225838f65d4832f0f7060269a7363f3ca1794f1cf），只解析已产生的元数据。

| 验证指标 | 25050恢复点 | 26000 | 27000 |
|---|---:|---:|---:|
| 主eval_loss | 1.866348 | 1.887508 | 1.903813 |
| 平均AA CE | 0.704848 | 0.719752 | 0.726757 |
| 全遮蔽AA CE | 1.693868 | 1.719164 | 1.734098 |
| 全遮蔽END准确率(%) | 54.187438 | 46.410768 | 49.750748 |
| 首END准确率(%) | 17.298106 | 15.403789 | 14.456630 |
| 长度MAE | 3.176969 | 3.671820 | 3.061815 |
| 长度分母 | 2006.000000 | 2005.000000 | 2006.000000 |

主验证loss与AA CE连续两个点退步，第二轮目前尚未显示效果收益。27000的真实END准确率、长度MAE及漏检较26000改善，但首END准确率继续下降；不能只取其中改善的指标。END漏检0.04985%→0，precision0.396170→0.384438下降，误报0.8136%→0.9162%增加。

新40条训练日志loss/LR/grad_norm均有限，其中39条已进入原定cosine阶段；逐条学习率与冻结的epoch_restart_factor吻合。27000 LR=9.960867775716144e-5、训练loss=0.4386、grad_norm=0.362459；学习率仍接近预定峰值。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint27000_schedule_observation.json（SHAf6aa5e9714f2076e2dcc7c4ec77b8569a301fec7412db2fb09d17f0d6757b96b，627c5d rc0）。没有实际日程/恢复错误证据，不能从相关性指定回退原因；保留既定两轮运行并继续观察28000。best3仍父24000/25000与新25050，加latest27000，共4份完整恢复状态；26000权重正常轮替、JSON保留。


完整本地对照单独记录于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md；本节不将两个数据配方的数值混比。



## 2026-10-06T16:43:44.442769+00:00 26000实际保存验收通过，但验证效果回退

第二轮八卡当前26302/50082、25050→30050正常运行，abort=null；现有226559/234066监督及8worker新鲜心跳核实。新只读核查工具 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/review_v7_epoch2_checkpoint.py 首次实际执行通过（16684会话，43e674 rc0）：26000原生17文件/8rank摘要、615项来源、封存transition、原生epoch=1.0382971926041291与50082日程、固定1330×4验证、112计数/70核心聚合/26结束聚合及best3＋latest均核查。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint26000_root_review.json（SHA13162dedba13cfb58cd1cc1b9113948cd52527e3427b1c271e56e175fc10e8fe）。没有新增模型评测。

| 验证指标 | 25050恢复验收点 | 26000 |
|---|---:|---:|
| 主eval_loss | 1.866348 | 1.887508 |
| 平均AA CE | 0.704848 | 0.719752 |
| 全遮蔽AA CE | 1.693868 | 1.719164 |
| 全遮蔽AA准确率 | 43.7618% | 43.3605% |
| 全遮蔽END准确率 | 54.1874% | 46.4108% |
| 首END准确率 | 17.2981% | 15.4038% |
| END漏检率 | 0% | 0.04985% |
| 长度MAE | 3.176969 | 3.671820 |
| 长度比较分母 | 2006 | 2005 |
| END precision | 0.373411 | 0.396170 |
| END误报率 | 1.0458% | 0.8136% |

主loss、AA重建、END准确率及长度退步；precision与误报改善。不能把接续工程验收通过解释为效果提升，也不能据这一点宣布延长训练有益。实际39条新训练日志loss/LR/grad_norm有限，学习率逐条匹配既定第二阶段warmup：25050为8e-7，26000为9.58e-5，预定26042达到1e-4后cosine下降。末条训练loss=0.4343、grad_norm=0.474614。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint26000_schedule_observation.json（SHA4afdce610fe46f9b45e0697efbf7f94cd55458bc036be1692acab7c14f9b78fb）。退步与学习率升温同时发生，但没有因果对照，不指定单一原因；当前无恢复/日程/数据epoch错误证据，保持既定运行并观察27000新点，不热改冻结源或调度。

best3＋latest为父24000/25000、新25050＋latest26000，共4份完整恢复状态，最佳状态均保留。第二轮300步训练间隔均速0.735978秒/更新已记录；性能统计与效果结果分开，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/speed_first300_updates_root_review.json。下一常规保存27000，累计50082及9项生成尚未完成。

本地f010/42训练2450/3750、已核查2250对照，下一2500；22格9完整产物/1运行/12未启动，最终图PARTIAL。原Goal active、完整范围/身份/累计用量不变，上轮与本轮progress。序列注释停线、原数据审核门禁与独立CUDA暂停保持，无新资源提交/重启或测试checkpoint。新元数据工具不属于训练冻结源，既有源及旧核查工具未改。实时交接 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json。



## 2026-10-06T16:33:29.100254+00:00 第二轮25050真实恢复验收通过，已继续25050→30050

原八卡累计25465/50082，当前25050→30050段有新训练更新、abort=null；现有第二轮监督PID226559/start2455552329和生成监督PID234066/start2455753425身份、8worker新鲜心跳核实。父阶段25042已真实完成且原PID212174退出，勿重启；其原生epoch=1.0000399345074078，完整终态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/segment25042_root_review.json。

新25042→25050首段的8pool实际rc0、8份进程/进程组清理、24项optimizer/scheduler/各rank RNG原生恢复摘要均通过；旧绑定与新绑定通过MIGRATION_RESTORE和封存transition明确连接，615项冻结源哈希保持，模型仍f011/A4、micro32/GA1/world8/global256。25043–25050实际八个更新均有计时，保存参数与25042不同；17项原生文件及八rank摘要一致，实际epoch=1.0003594105666707、max_steps=50082。固定1330×4验证的112计数与70核心聚合核查通过；本首段未单独重算26项结束指标，不声称额外范围验证。

根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/initial25050_root_review.json（SHA69f1a7c2cbcea87eebeed2327982ed30dc5cd4d9a0e30bbc1b47f652e511c8cf，0f8f4e rc0）。首段eval_loss=1.8663481936605006；这8步用于实际接续验收，不据此评价方法收益或声称八卡连续轨迹位级等价。现保留best3＋latest并集：父输出24000/25000及新输出25050，共3份完整状态；父25042权重已正常轮替，原生归档与恢复证据保留。

原数据检查真实passed，/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/parent_gate.stdout.json；封存 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/seal_receipt.json，未生成审核证书或降低门禁。只读根核查最初12565d rc1，原因是将归档引用路径与原保存路径作整体比较；2ef2a0确认内容SHA一致，修正仅比较归档内容哈希后通过。未改模型/训练源码、未重跑模型或重放任务，初次失败原因保留在根复核JSON。

当前正式输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/run，冻结源 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/source_snapshot_v1（615项），状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/supervision_state.json；旧where目录是已完成的父阶段。原where专用核查脚本只接受旧目录/25042日程，后续新点须针对第二轮原生元数据核查，不能误用旧脚本或重跑已通过的25050验收。下一常规保存26000，累计两轮终点50082及之后9项生成仍待完成。

本地f010/42训练2425/3750、2250对照已核查，下一2500；22格9完整产物、1运行、12未启动，末点图仍PARTIAL。后续消融发布包继续等待真实两轮/生成终态。原Goal active，原身份/累计用量/完整范围保持；上轮与本轮progress。全部安全停线、数据门禁和独立CUDA暂停保持；无新资源提交、容器/worker重启或测试checkpoint。交接 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json。



## 2026-10-06T16:26:23.061725+00:00 25042父阶段完成；第二轮已在原八卡单次启动

父阶段25042于16:22:59.665284Z实际结束，8pool均rc0、8rank进程/进程组清理与24项optimizer/scheduler/RNG恢复摘要通过；原监督PID212174已消失，B/COMPLETE有效且无SUPERVISION_STOPPED。原生epoch=1.0000399345074078。根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/segment25042_root_review.json（SHA8e2c73528223e33b3f0466f631c070e0e749d4a8a589213c6df9f1f74981b91a，95639d rc0）；监督进程自身exit code未独立捕获，不伪称rc0。

25042的17文件、八rank保存摘要、112固定计数/70核心聚合/26结束聚合核查通过（7467e9 rc0），/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/checkpoint25042_root_review.json（SHA4b01ab550f141d44445ede637e59716a864ca7788492a87d4ec6dffb146d62f0）。主eval_loss=1.866527，较24000的1.866277略升；平均AA CE0.704739→0.704843、全遮蔽AA CE1.693416→1.693879略升；全遮蔽END准确率53.7886%→54.0379%，首END17.1486%不变，长度MAE3.297607→3.302094、分母2006且漏检0。仍有分化，不是下游生成结论。此时保留24000/25000/25042三份完整状态；25000已有原生记录但未另作根端完整复核。

现有两轮监督在原数据检查真实passed后于16:24:11.942046Z封存transition，16:24:14.220637Z单次入队25042→25050。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/seal_receipt.json；实际数据检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/parent_gate.stdout.json；入队 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/launches_v1/segment_25050/enqueue_receipt.json。全部seal引用哈希已核实，未生成/修改审核证书或降低门禁。八worker已执行新段，当前25050、阶段started，GPU原生恢复/25050保存退出验收仍待实际证据，不提前宣称通过。

当前训练状态应读取 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/supervision_state.json，父阶段终态保留在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/COMPLETE.json；原212174不要重启。累计目标50082，固定该点9项生成仍waiting_two_epochs。原Goal active，完整范围及累计用量保持；无新submit/cancel、无容器/worker重启，无额外模型测试。原序列注释停线、独立CUDA暂停及数据门禁保持。最新交接 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json。



## 2026-10-06T16:12:29.308522+00:00 24000新保存核查通过，继续等待当前阶段结束

原八卡当前24189/50082，22016→25042正常、abort=null，三个监督与本地训练的实际PID/start、8worker新鲜心跳已核实。新24000的17原生文件、八rank保存摘要、1330条×4遮蔽档、112计数/70核心聚合/26结束指标核查通过（b3bf93 rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/checkpoint24000_root_review.json，SHAda30c9094d2c5f5b66dea7d34f05cbaa5de8ebe6666dd288e2fdbd94b5a95deb。仅解析训练实际产物，没有额外模型验证。

| 指标 | 23000 | 24000 |
|---|---:|---:|
| 主eval_loss | 1.868110 | 1.866277 |
| 平均AA CE | 0.705711 | 0.704739 |
| 全遮蔽AA CE | 1.696361 | 1.693416 |
| 全遮蔽END准确率(%) | 54.237288 | 53.788634 |
| 首END准确率(%) | 17.048853 | 17.148554 |
| 长度MAE | 3.185444 | 3.297607 |
| END precision | 0.369942 | 0.373486 |
| END误报率(%) | 1.062439 | 1.037785 |

主loss和AA重建改善；全遮蔽END准确率下降、长度MAE上升，分母均2006且漏检均0。首END准确率、precision及误报率改善；结果仍有分化，不代替下游生成评测或证明架构优势。

best3＋latest并集为21000/23000/24000三份完整状态，latest已在best3；18000权重按既定策略轮替、JSON保留。下一常规保存25000，父阶段25042完成后由已部署的两轮监督执行25042→25050首段恢复验收，再继续累计两轮50082和9项生成。当前仍waiting_parent/waiting_two_epochs，没有提前开始生成或重启任何进程。

本地f010/42训练2325/3750、已核查2250对照，下一2500；完整22格仍9完整产物、1运行、12未启动。原Goal active、身份/累计用量和完整范围保持，上轮及本轮progress。原数据审核门禁、序列注释停线、独立CUDA暂停保持，无新增测试checkpoint。实时交接 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json；已关闭观察57608不再poll。



## 2026-10-06T16:00:00.441187+00:00 23000原生保存核查通过，原八卡继续当前阶段

正式八卡实测23161/50082，当前22016→25042段正常、abort=null。三个原监督、本地训练的真实PID/start与8worker新鲜心跳再次核实；直接观察现有进程，没有新建或重启监听、容器、worker或平台任务。此前有限观察57608已关闭，不再poll。原Goal active、范围与累计用量保持，上轮及本轮均progress。

新23000原生17文件、8rank参数/保存摘要、固定1330条×四mask、112计数/70核心聚合/26结束位置聚合核查通过（cf716c，rc0）；仅读取实际保存元数据，没有额外模型验证。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/checkpoint23000_root_review.json（SHA90193625471fcb0e25b33e7f8e9d9fb318ee1b243c70f5ac8521852a116e94c1）。

| 验证指标 | 22016段尾 | 23000 |
|---|---:|---:|
| 主eval_loss | 1.871544 | 1.868110 |
| 平均AA CE | 0.706036 | 0.705711 |
| 全遮蔽AA CE | 1.698330 | 1.696361 |
| 全遮蔽AA准确率 | 43.6248% | 43.7418% |
| 全遮蔽END准确率 | 54.3868% | 54.2373% |
| 首END准确率 | 16.9990% | 17.0489% |
| 长度MAE（分母均2006） | 3.316550 | 3.185444 |
| END precision | 0.368457 | 0.369942 |
| END误报率 | 1.0722% | 1.0624% |

主loss、残基重建、长度误差及误报改善；全遮蔽END准确率略降、漏检均0，不能声称所有指标提高或代替实际生成评测。best3＋latest现为18000/21000/23000三份完整恢复状态，latest已在best3；19000及22016权重按既定策略轮替，JSON与恢复/退出证据保留，未额外复制或手动删除正式状态。

下一常规保存24000，父阶段25042完成后仍由已有两轮监督接到50082，之后固定该点运行9项生成评测。当前两轮接续waiting_parent、生成waiting_two_epochs，均尚未完成；本地训练2275/3750、2250比较已记录，下一2500。完整22格及最终图仍未完成，不更改既定架构。序列注释停线、原数据审核门禁、CUDA专项暂停保持，无测试checkpoint新增。实时交接 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json。



## 2026-10-06T15:48:10.321315+00:00 22016整段通过八卡恢复/保存/退出，原任务已接续25042

17016→22016的5000更新于2026-10-06T15:44:08.778170Z完成。8份实际pool结果均succeeded/rc0，8份parent回执均确认wait、进程与进程组消失；24项optimizer/scheduler/各rank RNG恢复摘要与原生保存一致。22016原生17文件、八rank保存摘要、112固定计数/70核心聚合/26结束指标核查通过。根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/segment22016_root_review.json（SHA9e12b444fb0b964fe13f6be93d55be3f1863db5af4171088dcca6840da2bc103，1abebb rc0）；指标复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/checkpoint22016_root_review.json（SHA567b7443ce607773d93dd38c4fdcd09d7d2d86157f7cbc57990ba9758a8218bc，7f885b rc0）。仅核查实际产物，没有新增模型验证。

原监督于15:44:12.485859Z单次接续22016→25042，下一段已经产生真实训练更新；当前22185/50082，abort=null。平台原任务t-20261005205851-lqfjg在15:44:09只读查询仍Running，原三个监督PID/start及8worker新鲜心跳已核实。25042只是当前父阶段终点，之后仍由既有两轮监督接到50082，再运行9项生成任务；当前生成仍waiting_two_epochs、没有成绩。

该5000步完整段从14:41:17.746781入队到15:44:08.778170验收，耗时3771.031389秒（62分51秒），平均0.754206秒/更新，含初始化/原生恢复、验证、保存与退出验收；上一完整段0.752044秒/更新，两窗口速度相近。不是独立CUDA测试或整轮耗时保证。

| 验证指标 | 21000 | 22016段尾 |
|---|---:|---:|
| 主eval_loss | 1.868686 | 1.871544 |
| 平均AA CE | 0.706948 | 0.706036 |
| 全遮蔽AA CE | 1.701091 | 1.698330 |
| 全遮蔽END准确率 | 53.7388% | 54.3868% |
| 首END准确率 | 17.3978% | 16.9990% |
| 长度MAE（分母均2006） | 3.288136 | 3.316550 |
| END precision | 0.380784 | 0.368457 |
| END误报率 | 1.0051% | 1.0722% |

AA重建与真实END准确率改善，但主loss、首END、长度及误报有回退；漏检均0。段尾是原callback产生的额外评估，不把不同步数结果当成严格架构对照，也不推断下游生成提升。

best3为18000/19000/21000，加latest22016，共4份完整恢复状态。常规22000的eval_loss=1.871649、不在最佳3份中，已在22016保存时按原策略删除权重；JSON与指标仍在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/run/checkpoint_retention_evidence/checkpoint-22000_2860001fec43。根工具cc3c03读取22000活动目录时rc1（FileNotFoundError），原因是已正常轮替，不是训练失败；未伪称该工具通过、未重跑模型，随后核查仍保留的22016成功。

有限只读观察57608于实际22016验收后正常结束（4ab28d rc0，PID246927已消失），64次观察从19448到22016。结束记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/root_watch_to22016_completion.json（SHAd8d12ae566830c5c954489ca4718a5529e438d4a2928af9221203a4f4fc8d9f5）；勿再poll该已关闭句柄或将观察结束视为训练停止。原三个持续监督继续运行。当前没有额外未结束的有限观察会话，不重启旧监听/容器/worker，不重复提交。

本地f010/42最近实际观察训练2225/3750、最近验证2000，下一2250；原22格进度保持9完整产物、1运行、12未启动。原Goal active、完整范围与累计用量不变；上一轮和本轮均progress。全部审核门禁、序列注释停线和独立CUDA暂停保持，无自有测试checkpoint新增。实时交接 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json。



## 2026-10-06T15:33:35.637977+00:00 21000步新保存核查通过，验证指标有所改善

原八卡正式训练当前21147/50082，17016→22016段继续、abort=null；三个持续监督、同一只读观察57608和本地消融均通过真实PID/start核实，8worker心跳新鲜。上一轮verified_wait，本轮取得新的21000保存核查证据，属于progress；原Goal身份/累计用量与完整范围保持active。

21000原生17项文件、8rank参数/保存摘要、固定1330条×四档遮蔽、112计数、70核心聚合及26结束位置聚合核查通过（285995，rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/checkpoint21000_root_review.json，SHAb97ec1080317e67e175368e658dec7e89638f3ab709536e20bf3daaba4dfb157；仅读取新保存的元数据，不重复运行模型。

| 验证指标 | 20000 | 21000 |
|---|---:|---:|
| 主eval_loss | 1.874000 | 1.868686 |
| 平均AA CE | 0.708002 | 0.706948 |
| 全遮蔽AA CE | 1.703542 | 1.701091 |
| 全遮蔽AA准确率 | 43.5864% | 43.6064% |
| 全遮蔽END准确率 | 53.6391% | 53.7388% |
| 首END准确率 | 17.2483% | 17.3978% |
| END漏检率 | 0% | 0% |
| 长度MAE（分母均2006） | 3.546361 | 3.288136 |
| END误报率 | 1.0034% | 1.0051% |

主要重建和长度指标改善，END误报略升；不能用训练验证指标代替实际下游生成或证明架构优势。best3＋latest并集现为18000/19000/21000三份完整状态，latest已在best3；20000权重按既定轮替删除，原生JSON与审阅证据保留。没有新测试checkpoint。

有限观察57608继续等待22016真实保存/退出验收，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/root_watch_to22016_20261006T151255Z.jsonl；不重启观察、训练或容器。累计两轮与9项生成监督仍等待，下一常规保存22000；本地训练2175/3750、最近验证2000，下一2250。完整22格仍9完整产物、1运行、12未启动，原图PARTIAL不重画。数据门禁、序列注释停线、独立CUDA暂停保持。



## 2026-10-06T15:21:37.726456+00:00 20000步原生保存与验证核查通过

正式八卡当前20156/50082，17016→22016段继续、abort=null。新20000点只读核查通过（工具135133，rc0）：17原生文件、8rank保存摘要、固定1330条×4档遮蔽、112计数/70核心聚合/26结束位置聚合保持。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/checkpoint20000_root_review.json，SHA74e045c298e24f52d1e1faa61d66c19d2fffff503d4da22717749543d7fc83a8。只核查训练已产生的元数据，没有重复运行模型。

较19000，平均AA CE0.709208→0.708002改善；主eval_loss1.871489→1.874000、全遮蔽AA CE1.702343→1.703542略升，全遮蔽END准确率54.8355%→53.6391%回退。END漏检0.2991%→0、precision0.366789→0.380750、误报1.0888%→1.0034%改善；长度MAE3.180500→3.546361且分母2000→2006。指标有升有降，不能推出实际下游生成质量已改善，仍按原计划使用累计两轮50082模型评测。

本次best3＋latest并集为18000/19000/20000三份完整恢复状态，latest包含在best3；17000按原规则轮替，JSON证据保留。有限观察57608仍在、PID246927/start2456125035身份有效，继续等待22016段验收；该观察和原三个监督均未重启。两轮与生成监督继续等待，9项生成尚未运行。本地训练2125/3750、最近验证2000；完整22格进度未变。原Goal active，本轮已有新保存核查进展，全部安全停线及数据门禁保持。



## 2026-10-06T13:05:08.199910+00:00 正式9000步保存与验证已核实，持续到9303

where-DDP实际9303/25042，当前段目标12016；监督PID212174/start2455180199和本地PID80642/start2452750909仍在，八卡25次有界观察均有新鲜正在执行心跳、无abort。9000原生八rank保存、17文件/JSON归档、源码绑定、112固定计数、70核心聚合和26结束指标核对通过，实际检查会话70352退出0（c33975）。模型和验证集合未重跑。新工具 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/review_v7_where_checkpoint.py 已对真实9000点执行通过；后续仅核查新产生的点，不重复旧验收。段内保存验收不等于12016段退出验收。

| 指标 | 8000步 | 9000步 |
|---|---:|---:|
| 主eval_loss | 1.889376 | 1.890182 |
| 平均AA CE | 0.732485 | 0.734610 |
| 全遮蔽AA CE | 1.723888 | 1.740789 |
| 全遮蔽END准确率(%) | 55.682951 | 53.688933 |
| 首END准确率(%) | 17.497507 | 16.400798 |
| 未预测出END比例(%) | 1.994018 | 1.246261 |
| 长度MAE | 2.820448 | 3.249369 |
| 长度MAE分母 | 1966.000000 | 1981.000000 |

主eval_loss和AA CE略回升；END CE下降、漏检减少，但END准确率、首END、长度MAE及误报有回退，不能宣布整体质量提升。MAE分母1966→1981随漏检改变，不能脱离分母解读。继续原f011/25042日程与后续验证，不因一个点改架构。

best3 7000/7008/7016＋latest9000实际共4份完整恢复状态；8000权重已由训练策略删除，保留前JSON证据仍可读取，9000核查成功复用了8000归档，未因删除权重丢失比较依据。旧5016/6000及自有临时验证checkpoint保持已清理；不额外删除保留恢复点。根记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/checkpoint9000_root_review.json（SHA85df3e257935ccb5506df67b38ad5d34be246ddb4a879ef949a3eec02dadc2d1）。

新增已完成1000更新墙钟窗口8001→9001为722.879864秒，平均0.722880秒/步，包含一次完整常规验证/保存/源码检查/保留轮换；8002–9000的999个普通更新均值0.713420秒，中位数0.763121秒。9000非训练边界约9.44秒；此窗口没有段启动或恢复，不能当完整epoch ETA。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/speed_after_checkpoint9000_root_review.json。此前300步0.740862秒的证据保留，不覆盖原统计。

本地f010/42进度1600/3750、51200/120000条；最近已验收1500，下一1750，无真实退出回执。22格仍9完整epoch产物、1运行、12未启动，500图10/22、3750图9/22均PARTIAL。只读nvidia-smi显示本地GPU利用率100%，其中PID为设备/宿主侧视图，未据此映射或干预未知进程。

有界观察63484已于13:01:31 UTC正常退出0（220e9c），25次从8505观察到9000，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/root_watch_to9000_20261006T125530Z.jsonl（SHA88f955363dec278284d4902449df5379112021c737fea57dc625bfae72f0afbe）；结束的是观察窗口，训练/持续监督未停止。上一Goal回合和本回合均progress；原Goal身份/累计用量及25042＋22格完整范围保持，未完成。下一正式10000、本地1750；独立CUDA暂停、序列注释停线、原数据审核门禁保持，未提交/重启容器或任务。

## 2026-10-06T12:52:51.652079+00:00 新版常规8000步验证/保存通过，继续训练到8278

同一八卡where-DDP仍运行，当前8278/25042、当前段目标12016，总目标不变；新监督PID212174/start2455180199和8个正在执行的worker心跳再次核实，无abort。8000完整17文件、八rank参数与保存模型一致、六JSON归档与SHA、112固定验证计数、70核心聚合和26结束指标核对通过；固定1330条×4遮蔽档保持。此为段内checkpoint验收，整段7016→12016尚未结束；最近通过退出验收的段仍7016。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/checkpoint8000_root_review.json。

| 指标 | 旧代码7000 | 新代码8000 |
|---|---:|---:|
| 主eval_loss | 1.884267 | 1.889376 |
| 平均AA CE | 0.736503 | 0.732485 |
| 全遮蔽AA CE | 1.728628 | 1.723888 |
| 全遮蔽END准确率(%) | 56.331007 | 55.682951 |
| 首END准确率(%) | 16.301097 | 17.497507 |
| 未预测出END比例(%) | 0.049850 | 1.994018 |
| 长度MAE（有END的链） | 3.438404 | 2.820448 |

AA CE略改善，END准确率/漏检有回退，首END和有END链的长度MAE改善。长度MAE的比较分母随漏检变化，不能只据该数值判断生成长度变好；整体主eval_loss略高，不据单点宣布效果提高或更改架构。此表跨显式where代码迁移边界、相隔1000训练更新，并非同step效果对照。

best3＋latest实际保留7000、7008、7016、8000四份完整状态：前三份为主eval_loss最低，8000是最新；本次未删除这四份之一。原5016/6000已在先前轮换删除，JSON证据保留。保存完成后已观察到8001及更后训练计时，运行继续。下一常规验证/保存9000。300步稳态均值0.740862秒（旧10.199–10.696秒，约13.77–14.44倍）的统计范围不变，详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/speed_300_updates_root_review.json；不把验证保存耗时混为训练步时。

正式当前输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/run；持续监督 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/supervision_state.json。新独立2×4 FSDP的best3版本已完成CPU元数据/源码/CLI检查，未提交，文件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_where_best3_fsdp_2node4gpu_20261006.yml。旧容器未重建、未重复提交；独立CUDA暂停、序列注释停线与原数据审核门禁保持。原Goal身份和累计用量保持，25042正式训练与22格消融尚未全部完成。

## 2026-10-06T12:42:38.746417+00:00 新代码已完成八卡接续，300步实测平均0.740862秒/步

原容器同DDP正式训练已切换到独立where-only冻结源。7000→7008与7008→7016均通过实际8rank原生optimizer/scheduler/RNG恢复、参数/完整保存和8slot退出检查；随后7016→12016于2026-10-06T12:35:30.046083+00:00单次启动，当前实际计时到7450/25042。新监督PID212174/start2455180199已通过/proc核实，8worker心跳新鲜，abort=null。平台仍为t-20261005205851-lqfjg，没有重复submit/cancel或重建容器。原监督PID3977587已按用户授权停止，不重启。

| 计时窗口 | 更新数 | 平均秒/步 |
|---|---:|---:|
| 旧版5475→5875 | 400 | 10.199121 |
| 旧版6125→6850 | 725 | 10.695606 |
| where新版7027–7326 | 300 | 0.740862 |

新版中位数0.785829秒、P95 1.057530秒，global256对应约345.54条/秒；相对上述旧窗口观察到13.77–14.44倍提速。跳过本段最初10步；计时含数据与DDP的训练循环墙钟，不含验证/保存/初始化。旧窗口使用日志首次观察端点，有轮询误差且batch不同；不能将此作为同batch严格配对或端到端epoch耗时。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/speed_300_updates_root_review.json，含300条原始计时和来源哈希；未新增模型或CUDA专项测试。

保存已实际采用主eval_loss最小3份＋最新完整恢复点取并集，当前保留7000、7008、7016（最新与best重合）。原5016、6000权重已按新规则删除，删除前的JSON/指标证据保留在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/run/checkpoint_retention_evidence；每份保留完整优化器/调度器/8rank RNG。当前7016的eval_loss=1.889094435879232；旧7000=1.8842671278774004，短接续验收不证明效果提高。原旧段最后打印7600，只有7000完整保存，约600个已打印未保存更新已从7000重算，不宣称无进度回退。

接续证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/segment7008_root_review.json 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/segment7016_acceptance.json；新监督 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/supervisor_v1/supervise_v7_where_20261006.py，实时状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/supervision_state.json；当前输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/run。后续每段5000更新，经实际恢复/保存/退出验收后接到总25042；遇错误停止继续准入，不重复提交。当前仍为f011、micro32/GA1/world8/global256、BF16 autocast、math attention；仅替换等价chain_bias展开表达式，未启用高效SDPA或切FSDP。原数据门禁、序列注释停线、独立CUDA暂停保持；原Goal仍active，完整正式训练和22格消融未完成。

更新时间：2026-10-06T10:41:11.204701+00:00。最新常规验证/保存7000，最近段尾5016；当前段目标10016，完整25042未完成。

初步架构f011（R26+R24），no-ESMC，world8、micro32/GA1/global256，V7自然配方；6410749条训练记录。每1000步验证/保存，固定25042步调度不变。数据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002；原限定范围审核与来源门禁保持。





## 2026-10-06T10:41:11.204701+00:00 正式7000步：残基重建改善，END与长度指标回退

第7000步为原Trainer常规验证/保存，距6000为1000次更新。固定1330条/四档5320次记录评估保持；17原生文件、6JSON归档、8rank参数及保存摘要一致，参数较6000确有更新，数据/源码/日程绑定未变。根端检查JSON哈希及原生文件存在/大小，复用八rank回调的张量inventory，不重新载入模型权重。112个固定计数字段、70个核心聚合指标及26项结束指标/计数核对通过。

| 指标（按token汇总） | 6000 | 7000 |
|---|---:|---:|
| 四档平均AA CE | 0.746101 | 0.736503 |
| 四档平均AA准确率 | 77.1554% | 77.4181% |
| 全遮蔽AA CE | 1.749312 | 1.728628 |
| 全遮蔽AA准确率 | 42.8869% | 43.2234% |
| 四档平均END准确率 | 85.7244% | 85.4231% |
| 全遮蔽END CE | 1.365877 | 1.406400 |
| 全遮蔽END准确率 | 57.4277% | 56.3310% |

来源等权的平均AA CE1.192123→1.188161、全遮蔽AA CE1.971760→1.946087改善；全遮蔽END CE1.411773→1.430857、准确率60.1239%→59.6900%回退。默认全遮蔽来源等权总eval_loss1.909531→1.884267。已有正例关系准确率仍1，不能视为正负判别或功能性能。

| 结束位置/误报指标（全遮蔽） | 6000 | 7000 |
|---|---:|---:|
| 首END位置准确率 | 16.7498% | 16.3011% |
| 长度MAE | 3.220339 | 3.438404 |
| END token精确率 | 35.1971% | 36.8679% |
| AA误报END率 | 1.2161% | 1.1095% |

AA误报END2121→1935减少，token精确率提高，但首END正确链336→327下降。7000点有1条未预测END，长度MAE分母2006→2005；不能把缺失链从分母移除后的MAE当同一固定总体。它们是单次全遮蔽argmax指标，不是迭代生成质量。

| 来源 | 验证记录数 | 全遮蔽AA CE | 全遮蔽END准确率 |
|---|---:|---:|---:|
| asd_antibody | 256 | 1.593342→1.582908 | 37.1191%→36.0111% |
| oas | 256 | 1.642656→1.627813 | 41.2109%→39.8438% |
| ots | 256 | 1.885178→1.861566 | 72.2656%→75.3906% |
| tcr_native | 42 | 1.856747→1.738459 | 83.5443%→81.0127% |
| tcr_papers | 256 | 2.351109→2.338686 | 83.2117%→73.7226% |
| tcr_repertoire | 256 | 2.226806→2.217469 | 53.5156%→53.5156% |
| trait | 8 | 2.246484→2.255708 | 50.0000%→58.3333% |

6/7来源的全遮蔽AA CE下降，trait仅8条且CE略升；END方向分化。正式f011、micro32/GA1/global256及25042总预算保持，下一常规点8000，原10016段恢复/退出验收仍等自然结束。

原件 [checkpoint7000_root_observation.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint7000_root_observation.json)（SHAab877ea7d0bf5748432206b8741d0a4630b51cf21ee60617901ea3ef0295bf8d）与 [checkpoint7000_endpoint_diagnostics.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint7000_endpoint_diagnostics.json)（SHAfb9a311a13cff2b58a0d2757032186bc7c32aac0a495f893d60a5c855c42b5e2）。元数据复核入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/review_v7_formal_checkpoint.py，实际工具f4b88e rc0。仅用于新的原DDP保存点，不重复验收旧点或处理新FSDP schema。本轮无新模型测试、提交、重启、冻结训练更改或测试权重。


## 2026-10-06T07:40:52.614659+00:00 正式6000步：平均重建略改善，全遮蔽CE与长度误差仍有回退

第6000步是原Trainer常规验证/保存，距离5000常规点1000次更新、距离5016段尾点984次更新。固定1330条验证记录（四档共5320次记录评估，并非5320个独立样本）及数据、来源绑定保持。当前训练继续到既定25042步，原段目标10016；全程配置未改。

| 指标（按token汇总） | 5000（常规） | 5016（段尾） | 6000（常规） |
|---|---:|---:|---:|
| 四档平均AA CE | 0.748578 | 0.754328 | 0.746101 |
| 四档平均AA准确率 | 76.9604% | 76.6386% | 77.1554% |
| 全遮蔽AA CE | 1.737691 | 1.759509 | 1.749312 |
| 全遮蔽AA准确率 | 42.8318% | 41.3847% | 42.8869% |
| 四档平均END准确率 | 85.2601% | 86.7531% | 85.7244% |
| 全遮蔽END CE | 1.361929 | 1.256616 | 1.365877 |
| 全遮蔽END准确率 | 55.6830% | 62.1635% | 57.4277% |

来源等权的平均AA CE1.198948→1.192123下降，但全遮蔽AA CE1.968865→1.971760上升，全遮蔽END准确率62.6696%→60.1239%下降。token汇总与来源等权方向不同，不能只报告token汇总的END改善。默认全遮蔽来源等权总eval_loss1.900775→1.909531。已有正例关系准确率仍1.0，仅代表现有正例token重建，不代表正负binding判别或功能能力。

| 来源 | 验证记录数 | 全遮蔽AA CE：5000→6000 | 全遮蔽END准确率：5000→6000 |
|---|---:|---:|---:|
| asd_antibody | 256 | 1.579897→1.593342 | 40.7202%→37.1191% |
| oas | 256 | 1.636402→1.642656 | 23.0469%→41.2109% |
| ots | 256 | 1.869169→1.885178 | 72.2656%→72.2656% |
| tcr_native | 42 | 1.808495→1.856747 | 82.2785%→83.5443% |
| tcr_papers | 256 | 2.367319→2.351109 | 86.1314%→83.2117% |
| tcr_repertoire | 256 | 2.255484→2.226806 | 67.5781%→53.5156% |
| trait | 8 | 2.265293→2.246484 | 66.6667%→50.0000% |

| 全遮蔽结束位置诊断 | 5000 | 5016 | 6000 |
|---|---:|---:|---:|
| 首END准确率（按链） | 16.3011% | 18.7438% | 16.7498% |
| 长度MAE（按链） | 2.760219 | 2.524925 | 3.220339 |
| token END精确率 | 36.4794% | 35.5777% | 35.1971% |
| AA位置误报END率 | 1.1152% | 1.2947% | 1.2161% |

按链首END正确数327→376→336/2006，未预测END均0、长度比较分母均2006。AA位置误报END1945→2258→2121/174410；6000较5016减少，但仍高于5000。来源等权首END准确率18.7267%→19.2983%、END精确率39.7300%→42.1857%，而来源等权长度MAE2.449411→2.746532变差。以上仍是单次argmax重建诊断，不能据此认定多步生成或下游理解能力。

17项原生清单存在且非空，8rank参数摘要相同并与保存摘要一致，参数已较5000/5016改变。根仅读取JSON并检查文件存在/大小，未重新读取tensor；原Trainer回调的inventory记录与6份JSON归档一致。112计数字段与已验收点保持、70聚合及18结束位置核对通过。实际新验收工具033c9f退出0（0.358秒），复用了旧点已验收证据，没有旧点重跑或额外GPU验证。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint6000_root_observation.json（SHA0d66adf5c774abf132bfa342795913ff9f46d8bc99e7cfe8b7cb8305fff2e3d7）及 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint6000_endpoint_diagnostics.json（SHA93758c41eeeb3c89b440f584ff0093e66441ac88755cbbd13074af86320d4a71）。

原规则当前保留checkpoint-5000/5016/6000；4000权重已自然轮替，/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint_evidence/checkpoint-4000 内6份JSON保留。没有手动删除恢复状态或复制权重。初步f011及既定预算保持，继续看7000和本地f010的500/3750固定点，不据单点评估切架构。完整22格9个完整epoch产物、1运行、12未启动；完整Goal尚未完成。


## 2026-10-06T04:39:59.360843+00:00 正式5016段完成并接续10016；段尾评估单独记录

原监督于2026-10-06T04:30:01.140001+00:00验收16→5016的5000个新增更新，04:30:03.074248已在同一pool提交下一组8个内部票据，累计目标10016、从checkpoint-5016接续，完整目标仍25042。根只读复核实际8个pool结果均succeeded/rc0，8份rank父进程记录均正常退出、已wait、进程及进程组消失且清理完成；原生optimizer/scheduler及每rank RNG共24项内存/保存状态摘要相同。归档的checkpoint-16绑定与恢复证据一致。没有重新申请或重启平台资源；SEGMENT_5016.json的status=paused仅表示原程序的该段停止点，不是用户暂停Goal或整个任务。

新段的8个实际worker票据与enqueue记录一致、心跳新鲜，plan仅按既定规则更新resume/stop位置；同一pool进程、来源manifest、f011、micro32/GA1/global256和25042调度保持。新段已有5025训练日志，读取器按当前stage和原生历史/学习率定位，不从epoch小数推算。完整5016→10016恢复回执将在该段结束后再核实；当前不能声称其完整验收或与不间断训练位级等价。

冻结driver /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/scripts/research/run_v7_formal.py:225–228在段尾同时强制evaluate/save/stop，因此5016有独立的新评估，距离常规5000点仅16个更新。初次根读取曾假设只有5000评估，断言在写回执前停止；已按真实native log_history和冻结callback修正（失败工具ad7c2c，修正后a48493实际rc0）。这不是训练错误，也没有重跑模型。状态记录区分最新训练日志、已保存步数、已评估步数。

| 指标（按token汇总） | 第5000步（常规） | 第5016步（段尾） | 变化 |
|---|---:|---:|---:|
| 四档平均AA CE | 0.748578 | 0.754328 | +0.005750 |
| 四档平均AA准确率 | 76.9604% | 76.6386% | -0.3217个百分点 |
| 全遮蔽AA CE | 1.737691 | 1.759509 | +0.021818 |
| 全遮蔽AA准确率 | 42.8318% | 41.3847% | -1.4472个百分点 |
| 四档平均terminal准确率 | 85.2601% | 86.7531% | +1.4931个百分点 |
| 全遮蔽terminal CE | 1.361929 | 1.256616 | -0.105313 |
| 全遮蔽terminal准确率 | 55.6830% | 62.1635% | +6.4806个百分点 |

默认全遮蔽来源等权总eval_loss1.900775→1.925438上升。来源等权平均AA CE1.198948→1.197756略降，但全遮蔽AA CE1.968865→1.991178、准确率37.3531%→35.1347%变差；不能只保留5000的较好结果。TCR native全遮蔽AA准确率40.1954%→29.2086%，是此点明显回落；其他来源和terminal方向有分化，16步间隔不提供稳定趋势或架构对照。已有正例关系CE下降、准确率1.0仍不等于正负binding或功能能力。

| 来源（全遮蔽） | AA CE：5000→5016 | terminal准确率：5000→5016 |
|---|---:|---:|
| asd_antibody | 1.579897→1.606175 | 40.7202%→65.3740% |
| oas | 1.636402→1.640288 | 23.0469%→35.9375% |
| ots | 1.869169→1.893166 | 72.2656%→72.2656% |
| tcr_native | 1.808495→1.990564 | 82.2785%→83.5443% |
| tcr_papers | 2.367319→2.348208 | 86.1314%→76.2774% |
| tcr_repertoire | 2.255484→2.222226 | 67.5781%→67.5781% |
| trait | 2.265293→2.237617 | 66.6667%→75.0000% |

| 全遮蔽结束位置诊断 | 第5000步 | 第5016步 |
|---|---:|---:|
| 首END准确率（按链） | 16.3011% | 18.7438% |
| 首END准确率（来源等权） | 18.7267% | 18.4179% |
| 长度MAE（按链） | 2.760219 | 2.524925 |
| 长度MAE（来源等权） | 2.449411 | 2.310267 |
| token END精确率 | 36.4794% | 35.5777% |
| AA位置误报END率 | 1.1152% | 1.2947% |
| AA位置误报END率（来源等权） | 4.3436% | 4.4265% |

首END正确数327→376/2006、按链长度MAE改善，但来源等权首END准确率略降；未预测END均0、长度分母均2006。AA位置误报END1945→2258/174410，token END精确率继续下降。仍是单次全遮蔽argmax诊断，不是多步生成或功能验证。新点70聚合、112计数字段和18结束位置检查通过，固定1330验证记录/四档不变；复用5000证据，没有重复旧点评估。

checkpoint-5016的17原生文件、6JSON归档及8rank一致参数摘要已核实，参数较5000改变：b9c69a3d3ab4ecf72e0e0917e6589330ca25feac356c6e5e27659336b0194db8。binding保持，当前原规则保留4000/5000/5016，3000权重正常轮替且JSON仍归档；没有新增权重副本。恢复/接段原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/segment5016_root_review_20261006.json（SHA7dcc51cf7133f82146e92b6c99b50413bcb2bb4548eb98d6ef3d70b7b84fc51c）；评估 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint5016_root_observation.json（SHAf13fe5643e7fc844b593a8a85eec9ef80758acd2659c1131889f77a923c4e43f）、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint5016_endpoint_diagnostics.json（SHA22a48524672fabc308ebe27fc41d504645992af934e10572bd78f84d8339e20a）。段尾评估读取cc201b实际rc0；新长段恢复复核a48493实际rc0，仅元数据操作。

如实保留5条外层pool结果的cleanup_error=FileNotFoundError：代码位置是训练后prune_checkpoints扫描pending/running票据，不是rank进程清理。8个自有slot目录均无checkpoint/tensor产物，正式checkpoint完整；与其他worker同时rename票据造成扫描竞争相符，但异常未保留具体丢失路径，不能声称已证明唯一原因。没有通过忽略原生恢复/清理门禁强行接段，没有热改受保护pool源码或重试；当前作为非阻断清理诊断保留，冻结源运行期间不部署修改。

下一常规正式评估6000、下一累计段目标10016；本地f000最近验证3500，继续3750完整epoch。22格仍8完成/1运行/13未启动，固定500/3750对照图不变。原Goal active，本轮progress，完整训练/消融未完成；序列注释停线、独立CUDA暂停、限定数据审核和实际恢复状态全部保持。


## 2026-10-06T04:31:36.542828+00:00 正式5000步验证与保存完成：重建CE和真实END准确率改善，首END与误报仍分化

原监督器于2026-10-06T04:27:00.647840+00:00记录checkpoint-5000完成（19.97%计划步数），原八卡DDP继续完整25042目标；5000保存不表示当前5016段或整轮结束。本地f000已另行核实3500步验证，仍继续3750完整epoch；两套配方独立。

| 指标（按token汇总） | 第4000步 | 第5000步 | 变化 |
|---|---:|---:|---:|
| 四档平均AA CE | 0.759145 | 0.748578 | -0.010567 |
| 四档平均AA准确率 | 76.8552% | 76.9604% | +0.1052个百分点 |
| 全遮蔽AA CE | 1.748737 | 1.737691 | -0.011046 |
| 全遮蔽AA准确率 | 42.9299% | 42.8318% | -0.0980个百分点 |
| 四档平均terminal准确率 | 83.6137% | 85.2601% | +1.6464个百分点 |
| 全遮蔽terminal CE | 1.641390 | 1.361929 | -0.279461 |
| 全遮蔽terminal准确率 | 48.7537% | 55.6830% | +6.9292个百分点 |

来源等权平均AA CE1.205054→1.198948、全遮蔽AA CE1.976021→1.968865，默认全遮蔽来源等权总eval_loss1.913330→1.900775。但token汇总全遮蔽AA准确率略降；OAS、TCR papers/repertoire与trait全遮蔽AA CE上升。全遮蔽terminal准确率多数来源改善，TCR repertoire/trait持平；TCR repertoire对应terminal CE上升。已有正例关系token平均CE0.000261→0.000129、全遮蔽0.000275→0.000144，准确率1.0仍不是正负binding或功能证据。

| 来源（全遮蔽） | AA CE：4000→5000 | terminal准确率：4000→5000 |
|---|---:|---:|
| asd_antibody | 1.607184→1.579897 | 31.5789%→40.7202% |
| oas | 1.631109→1.636402 | 20.8984%→23.0469% |
| ots | 1.882474→1.869169 | 56.2500%→72.2656% |
| tcr_native | 1.876151→1.808495 | 79.7468%→82.2785% |
| tcr_papers | 2.342065→2.367319 | 82.1168%→86.1314% |
| tcr_repertoire | 2.239460→2.255484 | 67.5781%→67.5781% |
| trait | 2.253703→2.265293 | 66.6667%→66.6667% |

| 全遮蔽结束位置诊断 | 第4000步 | 第5000步 |
|---|---:|---:|
| 首END准确率（按链） | 16.8495% | 16.3011% |
| 首END准确率（来源等权） | 17.3963% | 18.7267% |
| 长度MAE（按链） | 4.073815 | 2.760219 |
| 长度MAE（来源等权） | 3.306467 | 2.449411 |
| token END精确率 | 38.5191% | 36.4794% |
| AA位置误报END率 | 0.8950% | 1.1152% |
| AA位置误报END率（来源等权） | 4.1269% | 4.3436% |

首END正确数338→327/2006下降，真实END位置准确率上升并不意味着首个预测结束位置更准。未预测END1/2006→0/2006，长度MAE分母2005→2006；不能把两次长度MAE当完整同分母逐链对照。AA位置误报END1561→1945/174410，token END精确率下降；来源等权误报率也上升。以上是一次全遮蔽argmax前向，不是多步生成或生物功能验证，各指标搜索范围按原定义保留。

仅复算新点：70个聚合值、112计数字段与4000匹配，固定1330条验证记录/四档、5320次呈现非独立样本；18项结束位置聚合核对通过。原4000复用已验回执，未重跑模型。6JSON与监督归档SHA一致，17原生文件非空，8rank参数摘要一致且较4000改变：5b25b0588c08bdfab4117d68e2134ec1349ebafbecf4b229005689d970ac12ca；numel75601176，binding与配置/验证子集保持。大权重校验沿用原保存回调，根未重读GB张量；未声称从5000实际恢复。新元数据读取工具97cf69实际rc0、0.2561秒，无GPU测试。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint5000_root_observation.json（SHA4e51f7c8d2006dd5221d548aa6cdd04ea1ab2ca36b1245de6ab846a5b0e99b64）；结束位置 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint5000_endpoint_diagnostics.json（SHAb94916684d8607e474bf8a98b9155cbdcc3dc2e3476c1cad90f0a3abc4dd08e2）。5000完成时按原save_total_limit=3保留checkpoint-3000/4000/5000，2000权重正常轮替，必要JSON仍归档；不额外复制测试权重或删除正式恢复状态。4000→5000保存事件相隔10926.702977秒，约10.9267秒/步（含验证/保存与观察延迟），不作为新的GPU纯计算测速或慢化归因。

下一常规正式评估6000步；5016原生接段仍由唯一原监督器按8票据成功、清理和恢复门禁处理，实际接续结果另行核实。本地下一3750完整epoch，完整22格仍8完成/1运行/13未启动，固定500/3750图保持。当前状态更新曾因检测到新5000节点而按门禁拒绝仅刷新旧状态，这是观察逻辑的预期保护，训练无异常；现已完成新节点核对。原Goal active，本轮progress；序列注释停线、独立CUDA暂停与原限定审核门禁保持。


## 2026-10-06T01:27:15.290670+00:00 正式4000步验证与保存完成：AA改善，结束标记与长度误差仍波动

原监督器于2026-10-06T01:24:53.944863+00:00记录checkpoint-4000完成（15.97%计划步数），原八卡DDP继续5016段和25042总目标，没有新提交/重启/停训。本地f000最近验证2750，原3750步完整epoch继续；两套配方独立。

| 指标（按token汇总） | 第3000步 | 第4000步 | 变化 |
|---|---:|---:|---:|
| 四档平均AA CE | 0.773253 | 0.759145 | -0.014108 |
| 四档平均AA准确率 | 76.3480% | 76.8552% | +0.5072个百分点 |
| 全遮蔽AA CE | 1.760729 | 1.748737 | -0.011991 |
| 全遮蔽AA准确率 | 42.1558% | 42.9299% | +0.7740个百分点 |
| 四档平均terminal准确率 | 85.2953% | 83.6137% | -1.6816个百分点 |
| 全遮蔽terminal CE | 1.328663 | 1.641390 | +0.312727 |
| 全遮蔽terminal准确率 | 57.7268% | 48.7537% | -8.9731个百分点 |

来源等权平均AA CE1.232357→1.205054；全遮蔽AA CE1.977242→1.976021仅小幅变化，全遮蔽AA准确率37.4475%→36.9851%下降。来源等权平均terminal准确率82.1536%→84.0003%，与token汇总方向不同。ASD、TCR papers/repertoire全遮蔽AA CE变差；抗体来源的terminal准确率下降、若干TCR来源上升，不能概括为全部来源/生成效果提升。默认eval_loss1.913330是全遮蔽来源等权总loss，正例关系token准确率1.0仍不能当正负binding能力。

| 来源（全遮蔽） | AA CE：3000→4000 | terminal准确率：3000→4000 |
|---|---:|---:|
| asd_antibody | 1.578471→1.607184 | 42.9363%→31.5789% |
| oas | 1.644222→1.631109 | 47.6562%→20.8984% |
| ots | 1.927323→1.882474 | 79.8828%→56.2500% |
| tcr_native | 1.880862→1.876151 | 75.9494%→79.7468% |
| tcr_papers | 2.337932→2.342065 | 69.7080%→82.1168% |
| tcr_repertoire | 2.215678→2.239460 | 36.7188%→67.5781% |
| trait | 2.256208→2.253703 | 41.6667%→66.6667% |

结束位置诊断：按链首END准确率15.5035%→16.8495%，但来源等权19.3023%→17.3963%；按链长度MAE2.761715→4.073815，来源等权2.396635→3.306467，均变差。未预测END0/2006→1/2006，长度MAE分母2006→2005，不能当逐链完整同分母对照。token END精确率33.9291%→38.5191%，AA位置误报END2255→1561/174410（1.2929%→0.8950%）；来源等权误报率2.5968%→4.1269%变差。以上是一次全遮蔽argmax前向，不是多步生成或功能验证。

新点复算70个聚合值、112计数字段与3000匹配，固定1330条/四档、5320次呈现非独立样本；结束位置18项新聚合核对通过。原3000结果复用已验回执，未重跑模型。6JSON与监督归档SHA一致，17原生文件非空，8rank参数摘要一致且较3000改变：3878eb226874e09e68b500433c06aea25291fabf9607389288688bacd8587159；binding与配置/验证子集不变。大权重校验沿用原8rank保存回调，根未重读GB张量；实际4000恢复尚未发生。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint4000_root_observation.json（SHA98f9e2b5e386ef0d3cb3fabe9d9a8ddc8bcc1dad5e935a57fb7d3c5be68c91a9）；结束位置 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint4000_endpoint_diagnostics.json（SHAbc7170739dd205bbabad27213e8bd1592ab1ddf2bec5c73b431ae0101d404e11）。当前保留checkpoint-2000/3000/4000；旧1000权重由原save_total_limit=3正常轮替，其必要JSON仍归档，没有额外测试权重副本或删除实际恢复状态。3000→4000保存事件相隔10715.547911秒，约10.7155秒/步（含验证/保存与观察延迟），没有新的GPU纯计算测速证据。

下一正式5000、本地3000；原完整22格仍8完成/1运行/13未启动，固定500/3750图保持。原Goal active，上一回合verified_wait、本轮progress；独立CUDA暂停、序列注释停线、限定范围审核与来源门禁保持。


## 2026-10-05T22:30:51.057390+00:00 正式3000步验证与保存完成，整体改善与来源差异并存

第3000步由原监督器于2026-10-05T22:26:18.396952+00:00记录完成；原八卡继续5016段，完整25042步未完成。本地当前2100/3750、最近验证2000步；正式和pilot配方独立，不直接比较loss。

| 指标（按token汇总） | 第2000步 | 第3000步 | 变化 |
|---|---:|---:|---:|
| 四档平均AA CE | 0.797064 | 0.773253 | -0.023811 |
| 四档平均AA准确率 | 75.7041% | 76.3480% | +0.6439个百分点 |
| 全遮蔽AA CE | 1.771696 | 1.760729 | -0.010968 |
| 全遮蔽AA准确率 | 41.9116% | 42.1558% | +0.2443个百分点 |
| 四档平均terminal准确率 | 81.3003% | 85.2953% | +3.9949个百分点 |
| 全遮蔽terminal CE | 1.659838 | 1.328663 | -0.331175 |
| 全遮蔽terminal准确率 | 41.4257% | 57.7268% | +16.3011个百分点 |

七来源等权平均AA CE1.260794→1.232357，全遮蔽AA CE1.991063→1.977242；但四档平均terminal准确率84.1577%→82.1536%，其CE0.626438→0.655345。全遮蔽逐来源AA CE在OTS1.901283→1.927323、TCR native1.864420→1.880862变差；terminal准确率在TCR papers91.2409%→69.7080%、TCR repertoire53.5156%→36.7188%变差。保留不同方向的结果，不能把token汇总改善描述为所有来源或生成质量提升。

同一1330条验证记录、四档遮蔽；112计数字段和采样/配置保持，5320次评估非独立样本。Root复算70个聚合值，核对6JSON及归档，8rank参数摘要一致且较2000改变：c4146f35a3c901831aca8c5cfbd258a352397b4d24599eae9d43c93bb825acbd；17原生文件齐全，二进制校验沿用实际保存回调，未重读大权重。默认eval_loss1.923425是全遮蔽来源等权总loss；正例关系token准确率1.0不是正负binding或功能证据。实际从3000恢复尚未发生。

全遮蔽按链首END准确率13.2602%→15.5035%，长度MAE3.310070→2.761715；来源等权分别15.8266%→19.3023%、2.850117→2.396635。未预测END均0/2006。与此同时，token END精确率37.7556%→33.9291%，AA位置误报END比例0.7855%→1.2929%；来源等权误报率则4.8412%→2.5968%，口径不可混用。这是一次argmax去噪前向，不是多步生成评估；真实END位置准确率与首END位置准确率定义不同。新3000点18项诊断复核通过，2000点直接复用已记录证据。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint3000_root_observation.json（SHA88a4133af4a1f1005b84a5133ee7645d59c9ea99f812bb8aa1f99c9899aeff56）及 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint3000_endpoint_diagnostics.json（SHA743b9b80db5eb35acd46ab2798afa4e344fe78bbf8b6c55ea103b2127f834976）。保存现为1000/2000/3000，旧checkpoint-16正常轮替且必要恢复JSON仍在归档，无额外权重副本。继续固定f011/25042协议，下一正式4000、本地2250；无重启、提交或额外模型测试。完整22格和固定500/3750图保持，原安全停线、审核门禁与正式恢复状态保留。

### 第3000步逐来源结果

| 来源（全遮蔽） | AA CE：2000→3000 | terminal准确率：2000→3000 |
|---|---:|---:|
| asd_antibody | 1.627708→1.578471 | 24.0997%→42.9363% |
| oas | 1.665358→1.644222 | 4.8828%→47.6562% |
| ots | 1.901283→1.927323 | 50.1953%→79.8828% |
| tcr_native | 1.864420→1.880862 | 84.8101%→75.9494% |
| tcr_papers | 2.385832→2.337932 | 91.2409%→69.7080% |
| tcr_repertoire | 2.225059→2.215678 | 53.5156%→36.7188% |
| trait | 2.267780→2.256208 | 66.6667%→41.6667% |

### 第3000步结束位置诊断

| 全遮蔽结束位置诊断 | 第2000步 | 第3000步 |
|---|---:|---:|
| 按链首END准确率 | 13.2602% | 15.5035% |
| 来源等权首END准确率 | 15.8266% | 19.3023% |
| 按链长度MAE | 3.310070 | 2.761715 |
| 来源等权长度MAE | 2.850117 | 2.396635 |
| token END精确率 | 37.7556% | 33.9291% |
| 按token AA误报END比例 | 0.7855% | 1.2929% |
| 来源等权AA误报END比例 | 4.8412% | 2.5968% |
| 未预测END | 0/2006 | 0/2006 |
| 长度误差分母 | 2006 | 2006 |
| 首END预测正确数 | 266/2006 | 311/2006 |

terminal准确率只看真实END位置；首END准确率在全部可见target槽位（含容量尾部）搜索。token END精确率和AA误报比例不计容量尾部，不能与首END搜索范围混为一谈。数据与验证子集绑定保持；这些汇总不提供逐条生成或功能结论。

## 2026-10-05T19:35:46.431403+00:00 正式2000步：补充结束位置诊断（只读已有结果）

复核冻结验证源码并重新汇总原1000/2000日志，新增模型运行0。当前评估是一次全遮蔽去噪前向的argmax，不是多步生成或生物功能验证。“terminal准确率”仅看真实结束标记位置；“首个结束位置准确率”在包含容量尾部的全部可见target槽位查找第一个预测END，按链计数。

| 全遮蔽指标 | 第1000步 | 第2000步 |
|---|---:|---:|
| 按链汇总首个END位置准确率 | 16.1515%（324/2006） | 13.2602%（266/2006） |
| 七来源等权首个END位置准确率 | 13.2088% | 15.8266% |
| 按链汇总长度绝对误差 | 2.463268（2001条可比较链） | 3.310070（2006条可比较链） |
| 七来源等权长度绝对误差 | 2.563148 | 2.850117 |
| 未预测出END的链数 | 5/2006 | 0/2006 |
| AA位置误预测END比例 | 1.2241%（2135/174410） | 0.7855%（1370/174410） |
| token END精确率 | 34.9284% | 37.7556% |

首个END指标在日志中归入token_micro，但实际分母为链数；不能与七来源等权混用。长度误差只在存在预测END的链中统计，两点分母不同，不能声称逐链误差对照。token精确率和AA误报率不计容量尾部，与首个END的搜索范围也不同。总体首个结束位置退步与来源等权改善并存；两种长度误差均变差，缺失END减少。原因未确认，不能据此认定训练目标、架构或解码出现特定故障，也不称生成质量改善。

继续原f011/25042步协议，在下一既定3000步验证观察同一组指标。没有重新验证旧checkpoint、改变训练、启动额外任务或重启。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint2000_endpoint_diagnostics.json（SHAc6986c542ef2e87340f2593b562061e83c015010a87f0d9331681680507e6596），36项计数/聚合复核；定义源码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/scripts/architecture_prototypes/validation_metrics.py，SHA2a5382be35c364140143d65060d60a41bba7aa609b51fc29846db3d1160d23ab。正式训练与本地消融继续，完整任务尚未完成。

## 第2000步：同口径结果与保存证据

| 指标（按token汇总） | 第1000步 | 第2000步 | 变化 |
|---|---:|---:|---:|
| 四档平均AA CE | 0.913768 | 0.797064 | -0.116704 |
| 四档平均AA准确率 | 72.5513% | 75.7041% | +3.1528个百分点 |
| 全遮蔽AA CE | 1.837341 | 1.771696 | -0.065644 |
| 全遮蔽AA准确率 | 40.5602% | 41.9116% | +1.3514个百分点 |
| 四档平均terminal准确率 | 85.1957% | 81.3003% | -3.8954个百分点 |
| 全遮蔽terminal CE | 1.545641 | 1.659838 | +0.114197 |
| 全遮蔽terminal准确率 | 57.1286% | 41.4257% | -15.7029个百分点 |

七来源等权的四档平均AA CE为1.260794/准确率61.0355%，全遮蔽AA CE为1.991063/准确率36.7143%；默认eval_loss=1.926993为全遮蔽来源等权总loss。验证记录/采样索引及112计数字段与1000一致，Root复算70个聚合值。四档平均terminal准确率下降，全遮蔽terminal退步更明显，不能用AA改善掩盖。

| 来源（全遮蔽） | AA CE：1000→2000 | terminal准确率：1000→2000 |
|---|---:|---:|
| asd_antibody | 1.742504→1.627708 | 47.9224%→24.0997% |
| oas | 1.701707→1.665358 | 27.1484%→4.8828% |
| ots | 1.946336→1.901283 | 61.1328%→50.1953% |
| tcr_native | 2.024929→1.864420 | 89.8734%→84.8101% |
| tcr_papers | 2.512811→2.385832 | 89.7810%→91.2409% |
| tcr_repertoire | 2.277643→2.225059 | 78.5156%→53.5156% |
| trait | 2.471341→2.267780 | 25.0000%→66.6667% |

8rank统一参数摘要 11e55a4d2e45c6c30a2b63a0977fcead204618be04044fc7bfc5f069445bb468，较1000改变；17原生文件+formal_checkpoint.json齐全，6JSON与原回调清单/监督归档相符。Root未重读二进制权重；未实际从2000恢复，原8→16恢复证据保持。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint2000_root_observation.json（SHAd0d5f91502042a552d60b27e345ca7deb4c174ba0306e5d45581cf48ffe1b654），JSON归档 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint_evidence/checkpoint-2000。2000步训练日志loss0.5503，与验证口径不同。

继续既定f011、自然配方、25042步和1000步eval/save周期；下一3000步观察两类指标，不因单点波动中途改配置。既有正例关系准确率1.0不代表正负binding或功能验证。当前磁盘保留checkpoint16/1000/2000，旧8按save_total_limit3轮替，必要JSON历史保留。完整22格消融未完成，正式与pilot不直接拼表。

## 第1000步：验证结果

同一批1330条验证记录分别用25%、50%、75%、100%遮蔽，共5320次记录评估；不是5320条独立样本。来源数量：{"asd_antibody": 256, "oas": 256, "ots": 256, "tcr_native": 42, "tcr_papers": 256, "tcr_repertoire": 256, "trait": 8}。

| 汇总方式 | 遮蔽口径 | AA CE | AA准确率 |
|---|---|---:|---:|
| 按token汇总 | 四档等权平均 | 0.913768 | 72.5513% |
| 按token汇总 | 全遮蔽 | 1.837341 | 40.5602% |
| 七个来源等权平均 | 四档等权平均 | 1.435434 | 57.1472% |
| 七个来源等权平均 | 全遮蔽 | 2.096753 | 34.6309% |

原日志默认eval_loss=2.026115是全遮蔽的来源等权总loss，不是按token汇总的AA CE，也不是四档均值。Root从逐来源AA count/loss/accuracy复算了每档token加权与来源等权结果，并核对四档均值。第1000步训练日志loss=0.7011；它与验证口径不同，不直接相减或计算收益。

全遮蔽共有174410个AA、2006个terminal、562个既有正例关系token；关系准确率1.0仅是已有正例token的结果，不代表正负binding判别或生物功能/生成能力。正式与pilot的数据配方、验证子集及batch不同，不能把这里的数值直接拼入3750步消融排名；初步f011选择保持。

## 第1000步：保存与运行证据

原八rank保存回调已核对原生17文件、源/数据/调度绑定、八卡参数一致及保存权重一致。共有75601176个参数，统一参数摘要 6aaa912b2d68aed85f71dc29b6c6096dd268e18e869c4aa2dd22fc0f3e33a762；相比checkpoint16真实变化。Checkpoint目录含17个原生文件加1个formal_checkpoint.json。Root复核6份JSON与回调清单及监督归档逐字节一致，没有重复加载或散列模型/optimizer二进制。

当前checkpoint1000仅完成保存检查，尚未实际从该点恢复；原8→16原生恢复已通过。本次没有新GPU验收、重启、submit/cancel、数据或训练源码改动。原八卡组继续训练至5016，由既有监督器唯一接续后续段；目标25042及完整消融仍未完成。

实际原生checkpoint：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_initial_training_20261005/run/checkpoint-1000。JSON归档：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint_evidence/checkpoint-1000。Root观察：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint1000_root_observation.json（SHA1b793648858826aeaff56c0d5553ef959f21a946604269635b8ecc30f90672c6）。自动事件：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/events.jsonl；当前状态：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/state.json。

正式恢复文件保留save_total_limit=3，正常轮替；仅归档必要JSON而不另拷贝权重。测试仍metrics-only，无新测试checkpoint或临时模型文件。

2026-10-05T18:47:08.472956+00:00 后续状态：正式继续向2000步验证保存推进；本地1250步验证已完成，下一验证1500步，当前1250/3750。完整22格仍8项完成、1项运行、13项待补。本地新结果 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation1250_comparison_20261005.json，不与本页不同配方的正式验证直接比较。

2026-10-05T19:52:34.799686+00:00 并行本地消融完成1500步验证（48000/120000条），当前1500/3750；下一本地1750。正式最近保存仍2000、下一3000。新的同格比较 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation1500_comparison_20261005.json，属于不同pilot配方，不与本页正式训练直接比较。

2026-10-05T20:59:44.248817+00:00 并行本地消融完成1750步验证（56000/120000条），当前1750/3750；下一本地2000。正式最近保存仍2000、下一3000。新的同格比较 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation1750_comparison_20261005.json 属于不同pilot配方，不与本页正式训练直接拼表。


## 2026-10-06T14:59:56.446079+00:00 18000保存验证与新5000步完整段速度

正式八卡最新实测18378/50082，当前17016→22016，最近段退出验收17016，abort=null。18000原生保存的17项文件清单、八rank参数摘要、固定1330条×四mask、112计数/70核心聚合/26结束位置聚合核查通过（10144；226971 rc0）。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/checkpoint18000_root_review.json，SHAfb3fc78ac4ddff3f5e79eb0a0d3599f5e943f0b0cb197de4c0d891382ecc4fbc。保留best3主eval_loss＋latest并集，此点实际为17000/17016/18000三份完整状态，latest已在best3中，因此无需凑到4份；旧权重由既定策略轮替，JSON证据保留。

较15000，主eval_loss 1.885322→1.869832，平均AA CE0.715557→0.711576、全遮蔽AA CE1.710584→1.706185下降；全遮蔽END准确率51.4457%→56.8794%上升、漏检0.1496%→0。长度MAE3.690964→3.708873且分母2003→2006；END precision0.375683→0.357905、误报0.9833%→1.1737%回退。不用部分指标代替真实下游结果。

新完整5000步段12016→17016的实际入队13:38:33.712836至验收14:41:13.933865共3760.221秒（62分40秒），平均0.752044秒/更新，名义340.41条曝光/秒。此值包含初始化/原生恢复、常规验证保存及8rank退出验收；初始化至首更新111.578秒、末更新至验收21.185秒。12017→17016两个训练端点间4999次更新平均0.725637秒，4994个普通训练间隔平均0.716325秒。速度在新5000步窗口保持，但不是整轮剩余时间保证或CUDA内核计时。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/segment17016_speed_root_review.json，SHA829b581301f8df7eaa4a17554dd0a0cae99eb6d03334d74c892404d0a5047546；只读工具e19a1a rc0，没有额外模型/测速实验。

三个原监督已再次通过真实/proc身份核实，八worker心跳新鲜；两轮和生成衔接继续等待，后续消融发布包已准备但未入队。本地f010/42最新验证2000，下一2250；3750末点消融仍9/22、PARTIAL。原Goal active，上回合与本回合均progress；未重复运行已验收模型、未submit/cancel或重启。原数据门禁、序列注释停线及CUDA暂停保持。


## 2026-10-06T15:10:11.005917+00:00 19000实际保存与有限观察结束

原八卡正式训练当前19215/50082，17016→22016段继续、abort=null。只读观察53743在真实19000保存/轮替证据出现后正常rc0结束，14次观察从18473到19006，原三个监督进程身份和8worker心跳持续有效。观察结束不是训练终态，不重启观察或训练。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/root_watch_to19000_20261006T150107Z.jsonl；结束记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/root_watch_to19000_completion.json。当前三个监督再次/proc核实存活。

19000新点的原生17文件、八rank保存摘要、固定计数及指标聚合核查通过（0d67ae rc0），/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/checkpoint19000_root_review.json，SHA2b7e160412e3a212613eb2484e217ac32f433074aea6b427f1de18e586d93a25。较18000，平均AA CE0.711576→0.709208、全遮蔽AA CE1.706185→1.702343改善；主eval_loss1.869832→1.871489小幅上升，全遮蔽END准确率56.8794%→54.8355%、漏检0→0.2991%回退。长度MAE3.708873→3.180500改善，但分母2006→2000；END precision0.357905→0.366789改善，误报1.1737%→1.0888%下降。这是混合变化，不能代替两轮后的实际生成评测。

此点best3＋latest并集为17000/18000/19000三份完整状态；latest已在best3中。17016权重按既定轮替删除，原生恢复与阶段验收JSON证据保留，无额外自有测试checkpoint。两轮与生成监督继续等待，后续消融发布包已准备但未发布。原Goal active、完整范围和累计用量保持；本回合与上回合均progress，等待期间有实际进程/句柄证据。独立CUDA暂停、序列注释停线、数据审核门禁保持。
