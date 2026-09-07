# IRBench 结果排行榜 (Leaderboard)

> 📌 **评测范围（2026-08-29 收窄，强制）：只做 AB（抗体）与 TCR 两个维度，其余一律不考虑。**
>
> 在范围内 = TCR T1–T4（§0.1–§0.4）+ AB CDR infilling（§0.5）+ AB Light pairing（§0.6）。
> ⛔ 已冻结 = MINT GeneralPPI、STRING 90/90 PPI、NbBench 纳米抗体、FLAb 抗体属性回归。
> 本文件下方 **P0 / P1 / A1 / AB-FLAB 各节及其数字仅作存档**，不再更新、不进 headline、不写入论文表。
> 范围定义与逐任务锚点见 [`README.md`](README.md) §0 / §3。

> 🛑 **审计未收口（2026-08-27，2026-08-28 部分修复）—— §0 部分数字已确认有误，引用前先读审计**
>
> 本文主体最后修改于 2026-08-27 06:19，**早于当日审计**，因此除下面标 ✅ 的项外尚未反映审计结论。已确认（逐位核对）的错误：
> - ✅ **已修（08-28）** **§0.1 两个 8B 行是跨 checkpoint 拼接**：AUROC 取自 `checkpoint-40000`，AUPRC 取自 43000/45000。两个 ckpt 输出目录均在盘上，已从单一 ckpt 重抄，无需重跑。seen 与 unseen 两张表都受影响，均已修；顺带修正 unseen AUPRC 列把最低值标成最优的加粗错误。
> - ✅ **已修（08-30）** **§0.4 的 n_pmhc 把类别成员数当成了参与平均的数量**：`_macro` 先丢 NaN 再平均，`n_pmhc` 却数成员数。在 `*_t4_unseen` 两个 run（`designs.jsonl` 只有 9 条，只对 unseen 生成）里，benchmark14 印 14 实为 **7**、`overall` 印 34 实为 **9**、`bioseq_seen` 印 25 实为 **0**（全 NaN）。已补 `n_pmhc_scored` 并让所有打印/汇总改用 `scored/total`。
>   **本条原表述有两处需更正**：(1)「held20 标 20 实际只有 2」说的是 `_t4_unseen` run 的 held20 *视图*；独立的 `*_t4_held20` run 是真正 20/20 完整（零 NaN），两者是不同 run。(2)「完整 14-pMHC 视图从未生成」漏了限定词：对 **fusion ckpt** 确实从未生成（本条成立，故 §0.4 该行标 7/14），但 `ours_bioseq` 等 17 个 run 覆盖全部 34 个 pMHC 且逐个打分成功，所以不能一般化为「从未生成」。
> - ✅ **已修（08-30）** **§0.4 held20 表里的 TCRDiff 取了它的 benchmark14 行**（0.0009），同 split 的 `tcrdiff_held20` run 一直在盘上且 20/20 完整，正确值为 **F1 0.0118**（低报 12.8×）、d_edit 2.954、seq-rec 0.732。更正后 TCRDiff 明显强于我们，不再是「同档」。
> - 🚨 **新发现（08-30）**：`bioseq_unseen` 一栏**跨模型不可比**，其成员随各 run 的覆盖范围变化（2 / 6 / 8 个 pMHC 三种）。已新增同集合视图 `bioseq_unseen_common`（固定 6 个 pMHC）。在这个唯一可比的口径下，**我们最好的模型与无条件随机基线 OLGA 无统计差别**（Δ=−0.047 d_edit，配对 t=−0.55，df=5，不显著；08-31 收紧，原写「只比 OLGA 好 0.046」），落后全部已发表条件生成器。详见 §0.4。
> - ✅ **已修（08-30）** **§0.2 的 baseline 参考值把组均值当成了逐方法值**：HD / LD / GIANA / iSMART 四行共享 0.25/0.66，那是论文正文对这四者**联合**给出的组统计（0.25 ± 0.06 / 0.66 ± 0.09），不是各自的值。已从 Fig 3A 柱上标注取回逐方法真值（并经论文正文、Table 1 定性分箱、补充表 S2 计数三重交叉验证），**我们的复算是对的、参考表是错的**。连带：吻合数 6/9 → 7/9；iSMART 变为精确吻合、旧注「仓库输出为较小一次运行」已撤回；LD purity 偏差 −0.097 → −0.007。校准判据实为 `|Δ|≤0.03`，非「吻合到两位小数」，已写明。另披露论文自身 Fig 3A 与补充表 S2 在 HD/LD retention 上互相矛盾。
> - ✅ **已修（08-30，同日第二处）** **§0.2 的 curation 一直多留了 589 个配对**，此前被归因为「数据库版本漂移」——**归因是错的**。真因是 `curate()` 里一处 pandas/dplyr 语义差异：`dplyr::filter` 丢弃判据为 `NA` 的行，故 R 原文的 `filter(PubMed_ID != <10x url>)` 也删掉了 `PubMed_ID` 缺失的 634 行，而 pandas 的 `!=` 对 `NaN` 返回 `True` 全部保留。修这一行后 universe 由 5,368 变为 **4,779**，论文四个硬编码常量（4,779 / 8,395 / 4,103 / 4,292）**逐一精确命中**，校准由 7/9 提升到 **9/9**（最大 |Δ| 0.005）——HD / LD 此前卡在判据外正是同一根因。§T2(a)(b) 已在新 universe 重算；**§T2(c) 与 §0.2 基础 A 的嵌入曲线仍在旧 universe 上，已标记不可引用、待重跑**（每行需重做 GPU 嵌入，Ours 侧留给 v2 ckpt）。
> - **§0.2 / §0.3 / §0.4 存在训练-评测泄露**（有效配对泄露 86.6% / 64.7% / 43.5%），聚类与表征的 blocklist **从未存在**。§0.1 T1 与 §0.4 benchmark14 为 0.0%，可引用。
> - **§0.5 的「干净打平」表述不成立**：`--asd_antibody_benchmark_blocklist ""` 使抗体侧去污染被完全禁用。**（08-30 已实测，污染确认）** 对 ASD-antibody 语料（708k 唯一重链）实测：SAbDab Kong 折 3,127 条测试抗体中 **54.4% 的 CDR-H3 原样出现在训练语料里**（全链同一性 ≥0.95 独立给出 55.3%；胚系背景对照仅 0.83%，富集 52 倍）；SAb23H2 60 条中 25% 重链完全相同。§0.5 主任务正是 CDR-H3 infilling，故其 Ours 行**只能作为上界**，与 Ophiuchus（无重叠 zero-shot）的「打平」在方向上不对称。详见 §0.5 与 `outputs/ab_cdr_leakage_report.json`。
>
> 🔧 **数据侧已修复（2026-08-28），但这些修复只对重训后的 270m 生效——下文所有数字仍是旧语料下的产物：**
> - 新增聚类 + 表征 blocklist（`t2t3_eval_blocklist.txt`，8,377 键，Lev≤1 扩展），已接入 `trait` / `tcr_native` / `tcr_papers` 过滤链。
> - `load_exclusion_keys` 改 **fail-fast**：空串与缺失文件从「静默 no-op」改为抛异常。这正是上一轮 ASD 与 T4 去污染双双失效的机制。
> - 270m 两个 yml 去掉 `--asd_antibody_benchmark_blocklist ""`、加 `tcr_papers`、改 `_v2` 输出目录、去掉从污染 ckpt 续训。
> - eval 采样从「取前 2,000 行」改为**按源随机抽样**：`valid.csv` 按 `source` 排序，旧口径下 `minervina` / `tenx` / `covidvac` 从未参与过任何 `eval_loss`。
> - valid / holdout 中与 train 编辑距离 ≤1 的行已移入 train（`tcr_native` 42–43%、`tcr_papers` 46–47%），复验后两个 split 的 exact 与 Lev≤1 均为 **0.0%**。
>
> 审计与重训方案见 [`audit_2026_08_27/`](audit_2026_08_27/)：
> [`VERIFICATION_RESULTS.md`](audit_2026_08_27/VERIFICATION_RESULTS.md)（复验结论，权威）·
> [`D_leakage_audit.md`](audit_2026_08_27/D_leakage_audit.md)（泄露实测 + 文献惯例）·
> [`RETRAIN_PLAN.md`](audit_2026_08_27/RETRAIN_PLAN.md)（重训数据方案 + checklist）·
> [`SALVAGED_FINDINGS.md`](audit_2026_08_27/SALVAGED_FINDINGS.md)（未复核的中途结论）
>
> ⚠️ **VOID（2026-08-15，2026-08-20 仍有效）**：文中所有 **Ours-BioSeq / grammar_v2 / cmp500k / mint / integrated 7L** 数字全部作废。当前架构是 `examples/llada` immune fusion。Headline ckpt：8B BERT `checkpoint-43000`、8B diffusion `checkpoint-45000`、270m BERT `checkpoint-49000`、270m diffusion `checkpoint-42000`（此处原写 8B 两条均为 `checkpoint-40000`，2026-08-28 依 `topk_val_manifest.json` 修正）。T1 主协议 = 冻骨干 + 五折 MLP retrain（非 zero-shot）；BERT=表征-only；diffusion=表征+生成。Humanization / GDPa1 / HD-Flu-CoV / m396 本轮不做。现行入口 [`../tasks/README.md`](../tasks/README.md) + 本文件 §0。[`../AB_TCR_EVAL_SUMMARY.md`](../AB_TCR_EVAL_SUMMARY.md) 与 [`../DATA_READY_EVAL_PROGRESS.md`](../DATA_READY_EVAL_PROGRESS.md) 已 ARCHIVE。下文 T1–T4 / A1 各节仅历史。
>
> ✅ **本轮正式数字已落盘（2026-08-25 收口）**：T1/T2/T3（BERT+diffusion）与 T4/CDR/pairing（diffusion-only）统一汇总在 **§0**。CDR 用 Kong 版 SAbDab + SAb23H2 sweep；pairing 用 `max_iter=124` + 官方 cfg sweep。encoder 泄露 / 长度泄露 / argmax 塌缩均已修，旧虚高数字作废。**§0 是本轮唯一权威表**；旧节不再更新。
>
> 持续更新。每个任务一张表；空缺表示尚未运行。
> `Ours-BioSeq` 列在免疫受体基础模型训练完成后填充；当前以公开 baseline + 通用蛋白 LM(ESM2) 作参考。
> 本地运行数字可由 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/` 下的预测文件复算；论文 fallback 数字没有本地 predictions，必须标注“论文值，未本地复现”并给出表格位置。所有行同时标注 split / 数据版本 / 负采样比 / 随机种子或论文来源。
>
> 复现命令见每个表下方。数据版本：IMMREP23 paired-chain VDJdb (commit 06d85be)，neg ratio 5:1，seed 0。
>
> **呈现规约**：所有对比表 / 图中，**我们的模型（Ours-BioSeq）一律排在 baseline 之后** —— 行式对比 = 表末的 Ours 行块、列式 = 最右列、图 = 图例末项；**即使分数领先也不与 baseline 穿插排序**（靠加粗 / 高亮强调）。详见 `PROJGUIDE.md §2.4`。
>
> **Baseline 来源规约（2026-07-21，排行榜构造部分已于 2026-09-02 反转）**：`[P]`=论文报告值，`[A]`=官方 artifact 重评分，`[R]`=官方代码本地重跑，`[L]`=本地重实现，`[C]`=本地控制。`[L]/[C]` 只作独立 diagnostic。
> **🔻 论文值优先（现行）**：每个已列 baseline 必须带上它自己的最佳已发表值作为 `[P]` **主榜行**；`[R]` 降为**附行**，永不取代 `[P]`；复跑失败也不得删行（标 `未获取` + 原因）；协议不一致时同行披露协议差。**旧口径**是「协议一致时以 `[R]` 为主、T1 更是官方环节可用即取 `[R]`」，已作废。理由：只摆本地复跑会系统性抬高 Ours 相对身位 —— T1 `seen_test` 上 8 个 baseline 有 7 个复跑低于论文约 0.010（见 §0.1）。规则详见 [`PROJGUIDE.md`](PROJGUIDE.md) §0.2.2。来源表：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/external/paper_reported_baselines.csv`；审计：同目录 `baseline_provenance_audit.json`。旧 MINT 回归 Pearson/RMSE、旧 FLAb Spearman 与 NbBench sklearn probe 均不属于“论文可比”层。

---

## §0 本轮结果 · immune fusion (BERT vs diffusion)（2026-08-20 跑完）

> **范围**：只评本地已有数据。`Ours-BERT` = 表征-only；`Ours-Diffusion` = 表征 + 生成。Humanization / GDPa1 / HD-Flu-CoV / m396 **无数据或按计划排除，未跑**。
>
> **锁定 ckpt（不追后续 step）**
>
> | Tag | Run | Ckpt | eval_loss |
> |---|---|---|---:|
> | `ours_fusion_270m_bert_49000` | `protein_esmc_llada270m_bert_immune` | `checkpoint-49000` | 0.2578 |
> | `ours_fusion_270m_diff_42000` | `protein_esmc_llada270m_diffusion_immune` | `checkpoint-42000` | 0.5638 |
> | `ours_fusion_8b_bert_43000` | `protein_esmc_llada8b_bert_immune` | `checkpoint-43000` | 0.2566 |
> | `ours_fusion_8b_diff_45000` | `protein_esmc_llada8b_diffusion_immune` | `checkpoint-45000` | 0.5348 |
> | `ours_fusion_v3_diff_27000` | `protein_esmc_llada270m_diffusion_immune_v3` | `checkpoint-27000` | 0.7676 |
> | `ours_fusion_v3_4gpu_diff_42000` | `protein_esmc_llada270m_diffusion_immune_v3_4gpu` | `checkpoint-42000` | 0.7548 |
> | `ours_fusion_v3_allchains_33000` | `protein_esmc_llada270m_diffusion_allchains_immune_v3_4gpu` | `checkpoint-33000` | 0.6866 |
> | `ours_fusion_v3_allchains_8gpu2m_18000` | `protein_esmc_llada270m_diffusion_allchains_immune_v3_8gpu_2m` | `checkpoint-18000` | 0.7044 |
> | `ours_fusion_v3_allchains_8gpu2m_26000` | `protein_esmc_llada270m_diffusion_allchains_immune_v3_8gpu_2m` | `checkpoint-26000` | 0.6902 |
> | `ours_fusion_v3_allchains_8gpu2m_151000` | `protein_esmc_llada270m_diffusion_allchains_immune_v3_8gpu_2m` | `eval_snapshot_151000` | 0.6326 |
> | `ours_fusion_v3_genonly_8gpu2m_44000` | `protein_esmc_llada270m_diffusion_immune_v3_8gpu_2m` | `eval_snapshot_44000` | 0.7520 |
> | `ours_fusion_v3_bert_1m_105000` | `protein_esmc_llada270m_bert_immune_v3_1m` | `eval_snapshot_105000` | 0.4263 |
>
> **v3 generated-only 两行（2026-09-01 收数）** 是七源、去污后、generated-only / global 128 的 diffusion，与上表前四行**不是同一配方**。
> **全链两行（2026-09-02 收数，8/8 Success）**：4 卡 50k cosine 的 33000，以及从它 weights-only 续出的 8 卡 2M polynomial 的 18000。
> **续训 26000（2026-09-02 收数，4/4 Success）**：eval 0.6902。数字在 §0.1–§0.6 / §0.8 组 D。
> v3 BERT 表征已评（`ours_fusion_v3_bert_final`）；spot 本轮不评。
>
> ⚠️ **末三行（151000 / 44000 / bert-105000）目前只有部分数字**（2026-09-07）：
> 前两行的 **T4 与 CDR 已回填**（§0.4 / §0.5 / §0.8），是**本地单卡跑**的
> —— `queue012` 上九条单卡评测排 41 分钟零起跑，四个 T4/CDR 阶段改本机 A100-80G 串行跑完
> （env 与 eval YAML 逐行一致，56 分钟，4/4 exit 0）。
> **两行的 pairing 仍在 `queue012` 排队**（`t-20260907044307-btjxr` / `t-20260907044321-ht8b6`），
> 而 pairing 是 headline 指标，故这两行**尚未收口，不得当整体结论引用**。
> `bert_1m_105000` 那一行的表征作业已 cancel、**未跑**，表内暂无其数字。
> 过程见 [`$ROOT/PROJECT_PROCESS.md`](../../PROJECT_PROCESS.md) 2026-09-07 节。
>
> **两个 8B 行已于 2026-08-28 修正。** 原先写 `checkpoint-40000` / eval_loss 0.2624 / 0.5411，与 §0.1–§0.6 全部表格所用的 `8B@43000` / `8B@45000` 标签矛盾。依据各 run 的 `topk_val_manifest.json`，top-k 选出的最优 ckpt 确为 **43000（0.25656）** 与 **45000（0.53479）**，40000 并非最优（8B diffusion @40000 的 0.5411 高于 45000 的 0.5348）。故表格标签是对的、本表原先是错的。
>
> **统一表征接口**：grammar-v2 record → `LLaDAEsmcFusion` post-LLaDA hidden → 全局 mean-pool（`grammar:decoder:global:<ckpt_dir>`）。生成走 fusion `_denoise` + inverse remap，**仅 diffusion**。
>
> **来源层**：本轮 Ours 全部为 `[L]`（本地口径本地跑）。论文 `[P]` 与官方 ckpt 复跑 `[R]` 保持分层，**不与 `[L]` 合并排序**。
>
> **作业**：`eval-immune-{270m,8b}-{bert,diff}-repr` + `eval-immune-{270m,8b}-diff-gen`，队列 `c20250601`。日志 `output/downstream_generation/eval_<tag>{,_gen}.log`。

### §0.0 已确认的缺陷（读表前必看）

> 🔴 **(g) 抗体配对落后的根因：`heavy2light` 训练桶从未启用，训练与推理的条件化模式失配（2026-08-31 定位）**
>
> 此前 §0.6 把 pairing 的差距定位到「跨链条件化强度只有官方 30%、且集中在 V 区」，但**没有回答为什么**。
> 现在查到了，而且是配置事实而非推测。
>
> `sample_chain_conditioned_timesteps`（`dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py:452`）实现了
> **Ophiuchus 的五个多链训练桶**——函数 docstring 原话是 "Sample per-token ``t`` from Ophiuchus
> multi-chain ratio buckets"，其中 `heavy2light_loss_ratio` 正是「重链 `t=0` 保持干净、轻链高 `t` 掩掉，
> 学从重链推轻链」这一模式，也正是 pairing 推理时的设定。
>
> **但四次 immune fusion 训练一个都没开。** `train_jobs/protein_esmc_llada{270m,8b}_diffusion_immune.yml`
> 里不出现任何 chain-ratio 参数，因此全部走 `protein_pretrain_esmc.py:467-470` 的默认值：
>
> | 桶 | 默认 | 含义 |
> |---|---:|---|
> | `single_chain_ratio` | 0.0 | 单链 |
> | **`heavy2light_loss_ratio`** | **0.0** | **重链→轻链（= pairing 推理模式）** |
> | `light2heavy_loss_ratio` | 0.0 | 轻链→重链 |
> | `independent_loss_ratio` | 0.0 | 两链各自独立 `t` |
> | **`joint_loss_ratio`** | **1.0** | **全部 eligible token 共享一个 `t`** |
>
> 于是实际训练目标退化为：每条序列抽一个 `t ~ U(ε,1)`，然后对**所有链的所有位置** i.i.d. 以概率 `t` 掩码。
> `PROTEIN_PRETRAIN_PROGRESS.md:406` 也记着这一点（"默认 `joint_loss_ratio=1.0` 下……给全部 eligible
> token 发同一个 `t`"）。
>
> ⚠️ **重要限定：该失配只适用于抗体配对，不适用于 TCR 生成。** 本条早期版本把同一机制套到 T4 上，
> 算出「表位干净→CDR3β 全掩」的训练概率 10⁻⁷·⁷ 并据此解释 T4 的失败，**那是错的，已撤回**。
> 原因：`data/grammar.py:463-465` 把表位块与 MHC 块渲染为 **`is_fixed=True`**，
> 而 `diffusion_mask = [int(not is_fixed) for is_fixed in fixed]`（同文件 506 行），
> 所以**表位与 MHC 永不被掩码、永不计入 loss**——训练时表位就已经是干净条件，与推理完全一致，不存在失配。
> T4 的真实根因是数据配比，见下面的 (h)。
>
> **对抗体链失配有多严重可以精确算。** 抗体重链与轻链渲染为 `is_fixed=False`（`grammar.py:485-491`
> 的 `antibody_pair`），两条链共享同一个 `t`。在 joint 模式下，「A 侧 n₁ 残基全干净、B 侧 n₂ 残基全掩码」
> 的出现概率是 `∫₀¹ (1-t)^n₁ · t^n₂ dt = B(n₂+1, n₁+1)`：
>
> | 推理时的条件化模式 | 训练中出现概率 | 性质 |
> |---|---:|---|
> | **AB pairing：重链干净(120) → 轻链全掩(110)** | **10⁻⁷⁰** | 需要**纯跨链推断**，训练中从未出现 → 真失配 |
> | AB CDR-H3：其余干净(108) → H3 全掩(12) | 10⁻¹⁸ | 需要的只是**局部空缺补全**，`t≈0.1` 时训练中大量存在（仅位置散布而非成段）→ 非质变 |
>
> **两者性质不同，这正对应表现差异**：
> - **pairing 需要的能力在训练中不存在**。轻链 110 个残基全掩、没有任何局部锚点（prompt3 只给 3 个），
>   必须靠重链信息。结果条件化强度仅官方 30%，且缺口精确落在 **V 区**（VH–VL 界面，纯跨链信息），
>   而 J 区（可由轻链自身语法预测）与官方持平——**缺口位置与该假设的预测一致**。
> - **AB CDR 需要的能力在训练中充分存在**（i.i.d. 掩码在 `t≈0.1` 时平均掩掉约 12 个位置，正是 H3 的规模，
>   只是散布而非集中成段），所以它是 span-mask vs i.i.d.-mask 的量的差别，不是质的缺失。
>   该任务确实「基本打平」（1.3 pp 内）——但注意其中 **54.4% 测试抗体的 CDR-H3 原样在训练语料中**（见 §0.5），
>   且 H3 仅 12 残基、两端有保守 C…W/F 锚点、周围 108 残基全可见，所以这个「打平」由污染与任务局部性共同撑起。
>
> **为什么「局部语法强、跨链依赖弱」是 joint 桶的必然结果**：i.i.d. 掩码下任一被掩位置的紧邻残基
> 有 `1-t` 概率可见，所以最优策略始终是**先用局部邻居**；跨链信息在训练时几乎总是冗余的（同链邻居给出更强信号），
> 模型没有任何压力去学它。这也解释了 §0.6 那张表里 `max_iter` 32→124 我们只涨 0.002–0.003 而官方涨 0.027——
> **加迭代步数无法补上一个压根没学过的依赖。**
>
> ⚠️ **定性边界：以上是「配置事实 + 概率计算 + 缺口位置吻合」，尚未做反事实训练验证，因此仍是假设。**
> 两个可检验预测，都不需要重训：
> 1. **扫轻链可见比例**（0 / 3 残基 / 10% / 25% / 50%）。若失配假设成立，我们的 ImmunoMatch 曲线应比官方
>    **陡得多**——因为我们依赖局部上下文而官方依赖跨链条件化；且在可见比例足够高时应向官方收敛。
> 2. **给我们自己扫 cfg**。目前 Ours 只有 `cfg=0`。注意这条**不足以解释主要差距**：官方 `cfg=0` 已达 0.668
>    （占特异性空间 91.4%），`cfg 0→1.5` 只值 +0.038，而我们 `cfg=0` 仅 29.2%。
>
> 若假设成立，修复路径是明确的：**启用 `heavy2light_loss_ratio` / `light2heavy_loss_ratio` 做一轮续训**，
> 机制已在代码里、无需新架构。

> 🔴 **(h) TCR 生成差的根因：拟合失败，不是条件化失败（2026-08-31 实验定论；原配比诊断已推翻）**
>
> 与 (g) 是**两个独立的根因**。TCR 侧不存在掩码失配（表位是 `is_fixed=True` 固定上下文，训练时就已干净）。
> 本条先记录最初的配比论证（下半段），再记录推翻它的 infilling 实验（本条后半 ⛔ 段，结论以后者为准）。
> 起点是数据配比：`--dataset_args oas+ots+asd_antibody+trait+tcr_native` 里只有两个源带表位标注。
>
> | 数据源 | 训练行数 | 占比 | 带表位条件 |
> |---|---:|---:|:--:|
> | `oas`（抗体重链） | 2,486,442 | 43.99% | ✗ |
> | `ots`（配对全长 TCR，无表位） | 2,102,722 | 37.20% | ✗ |
> | `asd_antibody`（抗体） | 850,134 | 15.04% | ✗ |
> | **`tcr_native`**（TCR–表位） | 143,391 | 2.54% | **✓** |
> | **`trait`**（TCR–pMHC） | 69,251 | 1.23% | **✓** |
> | 合计 | 5,651,940 | | |
>
> **带表位条件的样本 212,642 / 5,651,940 = 3.76%；其余 96.24% 没有表位块。**
> （两个源的 `epitope_seq` 均 100% 非空，故该占比不含空值折扣。）
>
> **为什么 3.76% 不够**：那 96.24% 的样本里压根没有表位块，模型在它们上面学的是**无条件**生成。
> 当 96% 的梯度都在教「不看表位也要生成合法序列」时，把表位当噪声忽略就是全局最优解——
> 一个强无条件先验已经能压低绝大部分 loss，而表位条件化只能改善那 3.76%。
>
> **这个预测与实测精确吻合，且吻合方式很特殊**：我们在 §0.4 主对照里与 **OLGA 仅差 0.046 编辑距离 /
> 0.013 序列回复率**，而 OLGA **正是一个纯无条件先验**（VDJ 重组采样器，见 §0.4 说明）。
> 对照 TCRT5：它是专门在 TCR–表位配对上训练的 seq2seq，**100%** 的训练信号都是表位条件化，
> 于是比 OLGA 好 1.65 编辑距离，是我们的 36 倍。
>
> ---
>
> ### ⛔ 上述「条件化失败」诊断已被 2026-08-31 的 infilling 实验**推翻**，改判为拟合失败
>
> 完整证据见 `audit_2026_08_29/T4_infill_diagnostic.md`。核心做法：把表位从任务里**彻底拿掉**，
> 给定真实 held-out CDR3β、掩掉中间一段让模型自填，与「不看上下文、只按 (长度, 位置) 查训练边际」
> 的 PWM 对照比。若原诊断（模型学成了一个无条件先验）成立，无表位任务应至少打平 PWM。
>
> **实测：11 组条件全部显著低于 PWM**（McNemar 配对精确检验，p ∈ [6.6e-04, ~0]，长度保真 100%）。
> 摘录关键行，完整表见审计报告：
>
> | 条件 | 模型 | 掩码宽度 | ours | pwm | **ours − pwm** | p |
> |---|---|---|---|---|---|---|
> | `tcr_single`，holdout（与训练**零交集**） | 270m | 1（两侧全可见） | 25.65 | 32.70 | **−7.05** | 3.9e-09 |
> | `tcr_single`，holdout | 270m | 3 | 17.20 | 28.72 | **−11.52** | 1.7e-73 |
> | `tcr_single`，holdout | 270m | 全掩（均宽 11） | 27.90 | 40.37 | **−12.47** | ~0 |
> | `tcr_single`，holdout | **8B** | 1 | **28.55** | 32.70 | **−4.15** | 6.6e-04 |
> | `tcr_single`，holdout | **8B** | 3 | 19.27 | 29.93 | **−10.66** | 2.9e-17 |
> | `tcr_pair`（训练占 **37.2%** 的 layout） | 270m | 1 | 24.71 | 33.02 | **−8.31** | 1.1e-08 |
> | `tcr_pair`，**训练时见过的 616 行** | 270m | 1 | 31.49 | 38.96 | **−7.47** | 显著 |
> | `tcr_pair`，**训练时见过的 616 行** | 270m | 3 | 22.29 | 33.60 | **−11.31** | 显著 |
>
> 六类替代解释被逐一排除：**(1) 不是条件化问题**——无表位任务同样输；**(2) 不是 grammar layout
> 没见过**——训练占 37.2% 的 `tcr_pair` 结果相同；**(3) 不是泛化失败**——模型背过的行照样输
> 7.5–11.3 pp；**(4) 不是污染或解码**——`tcr_single` holdout 零交集，`tcr_pair` 去掉 30.8%
> 污染行后结论不变，长度保真 100%；**(5) 不是模型太小**——8B 在同一批 n=2000 上把差距从
> −7.05 缩到 −4.15，方向未变且仍显著，按此斜率（30× 参数换 2.90 pp）外推还需再 40–100×
> 参数才能打平；**(6) 不是统计假象**——检验用配对 McNemar（二者在同一批位点评分，非配对
> z 检验会低估证据），11 组方向一致。
>
> **反面结果（同日，此前从未做过的对照）**：带表位的训练样本 99.03% 渲染为 `tcr_pmhc`，而
> Setting B 主榜默认用 `tcr_epitope`——后者只占全部训练量的 **0.036%**，前者占 3.726%（102 倍）。
> 据此本应预期 `--with-mhc` 改善，实测**六个视图一致变差**（bioseq_unseen d_edit 6.4075 → 7.1688，
> seq_recovery −0.086，char_BLEU −0.080）。「切到训练实际用的 layout 就会好」同样被证伪。
>
> **改判后的结论**：这是 **TCR 侧的拟合失败**——模型的 CDR3β 序列建模能力低于「按长度和位置查表」
> 这一最弱的无条件基线，在训练见过的序列上亦然。比原诊断严重一档：原诊断说模型退化成了一个无条件
> 先验，实测它连位置边际这个最弱的无条件先验都达不到。相应地 §0.4 中「我们 ≈ OLGA」和 Setting A
> 的 JSD 0.265 应重新解读为同一个拟合失败的两个投影，而**不是**「条件信号被淹没」。
>
> ⚠️ **仍未定位「为何拟合失败」**。规模已排除（见上 (5)），但**抗体侧同口径对照尚未做**——
> 若抗体 CDR-H3 infilling 同样输给 PWM，则这是全局拟合问题，连 §0.5 抗体 CDR 的 43.70 AAR
> 也需重新解释；该对照须先去掉已知 54.4% 的 ASD 污染才有意义。训练轨迹也未扫（只测了各自
> 最新 ckpt），因此「继续训能否收窄」未知。上文的配比论证作为**动机**保留，但它已不足以解释
> 实测，提高 `trait` / `tcr_native` 采样权重也因此不再是有把握的修复方案。

> 🔴 **(f) 最后一层特征各向异性极重，使 §0.1–§0.3 既低估绝对值、又污染两臂比较（2026-08-30 实测）**
>
> **口径未变：headline 仍是 raw `hidden_states[-1]` + L2 归一化，重标定已评估并决定不采用（见 f-1 末）。**
> 本条记录的是这个读出的**已量化代价**，供引用时加限定，不是要改表。
>
> 此前 §0.1–§0.3 一直标注「不受缺陷 (0) 影响」——那句话仍然成立（表征是单次前向，没有迭代解码泄露），
> 但**不等于读出没有代价**。`grammar:decoder:global:` 把 `hidden_states[-1]`（post-`ln_f` 的最后一层）
> 直接 L2 归一化后送去聚类/kNN —— 而这恰好是全模型**线性可解码性最差、各向异性最重**的一层。
>
> 270m 两条 best ckpt 上逐层实测（`scripts/diagnostics/diag_repr_layers.py`，T2 basis-B 同一份
> 9,033 CDR3β / 25 表位，K sweep 均值；先验证 harness 口径无误：**layer8 raw ARI 复现出
> BERT 0.0187 vs 本表 0.0186、diffusion 0.0275 vs 本表 0.0277**）。逐层 probe 显示
> **最后一层是最差的一层**：BERT 0.734（layer0）→ 0.646（layer8）、diffusion 0.737 → 0.683，
> 逐层单调下降。
>
> **各向异性是根因**：mean pairwise cosine 从 layer0 的 0.71/0.58 涨到最后一层的
> **0.994/0.986**，effective rank 从 38/42 掉到 17/20。§0.2 里记的「8B BERT 余弦挤在 0.9999、
> tau 分辨率退化」**不是 8B 特例，是全部四条的共性**，270m 上同样存在，只是没触发阈值曲线溢出。
> 后果是余弦/K-means 看不穿这个几何 —— 有用的方差只占总方差极小一部分。
>
> #### (f-1) 用重标定作为**探针**测出这个几何代价有多大（后处理已评估并**决定不采用**）
>
> ⚠️ **口径声明：headline 保持 raw，不做任何重标定。** 下面这组数是**诊断探针**，
> 用来量化「原始几何有多不适合直接做度量」，**不是我们主张的性能**。
> 一次性加过的 `post=` spec token 已从 `common/model_api.py` **移除**，代码回到 raw-only。
> 探针做法：同一份最后一层 feature（不换层、不用 ESMC 特征），只在送进余弦/K-means 前
> 换一种缩放，跑**官方** `run_paper6.py` / `run_embed_bench.py`。
>
> **T3 deep k=200 macro NN AUROC**（官方脚本，6 pMHC / universe 25,816 / 100 seeds）
>
> | 探针读出 | BERT 270m@49000 | Diffusion 270m@42000 |
> |---|---:|---:|
> | **raw（headline，§0.3 已记录）** | **0.712** | **0.709** |
> | 仅去中心化 | 0.696 | 0.693 |
> | PCA 白化（256 维） | 0.760 | 0.755 |
>
> **T2 basis-B K-means ARI**（官方脚本，同一份 9,033 CDR3β）
>
> | 探针读出 | BERT | Diffusion |
> |---|---:|---:|
> | **raw（headline）** | **0.019** | **0.028** |
> | 仅去中心化 | 0.023 | 0.038 |
> | PCA 白化（256 维） | 0.019 | 0.023 |
>
> 三条可引用的结论（**都不依赖采用后处理**）：
>
> 1. **§0.1–§0.3 的绝对数值是这一读出下的下界，不是模型表征能力的上限。**
>    仅换缩放就能在 T3 上多拿 ~0.047，**比表里任何模型间差异都大**
>    （此前最大是 8B diff vs 270m diff 的 0.011），也比 top-k 波动带（≤0.004）大一个数量级。
>    这不是调参余量，是几何退化的度量。
> 2. **两臂差距的成因分两种，须分开陈述**，见 (f-2)：probe 上的差距是几何造成的
>    （BERT 信息量相当），**但 ARI 上的差距是真实的、未被解释掉**。
>    ⚠️ 这条最重要，因为 §0.1–§0.3 的用途就是比较两臂。
> 3. **不同度量对缩放的偏好相反**（T2 吃去中心化、T3 吃白化），这是各向异性的典型症状：
>    白化把所有方向拉平，利于局部邻域检索，但会把噪声方向放大到与信号方向同量级、破坏大尺度簇结构。
>    (f-2) 的配对实测直接证实了后半句：白化使 diffusion 的 ARI **显著下降 4.1 sd**。
>
> 🚫 **明确不成立、不得引用的说法**：
> - ❌「我们越过了 CDR3-Levenshtein（0.733）/ k-mer(3)（0.728）」—— headline 是 0.709–0.712，没有越过。
> - ❌「与 TCRdist 的差距是 0.017」—— headline 下仍是 **0.065**；与 SCEPTR 仍是 **0.078**。
> - ❌ 任何对 ESM2-150M / ProtBERT / TCR-BERT 的「反超」—— 这三条在本表里同样是
>   **未重标定的 mean-pool transformer 嵌入**，很可能同样大幅受益，从未在同一处理下比过。
>
> **为什么不采用**（决策记录，2026-08-31）：重标定是 post-hoc 补救，回答不了
> 「为什么模型本身产不出可直接度量的空间」。我们的预训练目标是 token 级重建，
> 从未约束嵌入空间的几何，所以长成这样是**预期行为**；SCEPTR 不需要重标定，
> 是因为对比学习（InfoNCE + L2 归一化）在训练时就把空间约束成近似各向同性了。
> **durable 的修法在训练侧（加空间约束/对比项），不在读出侧。**
>
> #### (f-2) 🔴 两臂差距：加上误差棒后（2026-08-31 重测，**推翻本条前一版**）
>
> ⚠️ **前一版曾写「probe 差距 +0.0363 塌到 −0.0055，kNN@1 排序直接翻转」并据此说 BERT 反超 ——
> 那是把噪声当信号，已撤回。** 原先的 probe 是**单次** `train_test_split(random_state=0)`、
> ARI 是**单个** K-means 种子，无任何方差列。补测方法：两臂在**同一重采样**上打分
> （配对统计量），probe 20 次分层划分 / ARI 8 个种子 / kNN@1 2000 次配对 bootstrap。
>
> **七个臂间比较里只有两个可分辨**（Δ = diffusion − BERT）：
>
> | 指标 | 读出 | Δ | 配对 sd | |
> |---|---|---:|---:|---|
> | 25 类 probe AUROC | **raw** | **+0.0330** | 0.0064 | ✅ **5.1 sd，20/20 同号** |
> | 25 类 probe AUROC | 白化 | −0.0039 | 0.0056 | ❌ 0.7 sd = 0 |
> | K-sweep ARI | **raw** | **+0.0092** | 0.0012 | ✅ **7.6 sd，8/8 同号** |
> | K-sweep ARI | 白化 | +0.0027 | 0.0018 | ❌ 1.5 sd |
> | kNN@1 | raw | +0.0061 | 0.0045 | ❌ 1.4 sd，CI 跨 0 |
> | kNN@1 | 白化 | −0.0043 | 0.0038 | ❌ 1.1 sd，CI 跨 0 |
>
> 🔴 **所以 raw headline 下 diffusion 确实真的领先，§0.2 那个读数不是假象。**
> 但"差在哪"分两种情况，**不能混说**：
>
> - ✅ **线性可解码性：BERT 不缺信息、缺的是几何。** 同一份特征只做方差均衡，
>   **BERT 涨 +0.0818（9.2 sd）、diffusion 只涨 +0.0449（6.2 sd）**，probe 差距随即抹平到
>   −0.004（0.7 sd）。BERT 涨幅是臂间差距（0.033）的 **2.5 倍**。
>   且这不是 transductive 泄漏 —— 改成只在训练划分上拟合白化，数值几乎不变
>   （BERT 0.7307→0.7314、diff 0.7269→0.7285）。
> - 🔴 **全局簇结构：diffusion 的优势没有被解释掉。** ARI 差距缩小**不是因为 BERT 变好**
>   （+0.0014，1.3 sd，不可分辨），**而是因为白化把 diffusion 弄坏了**（−0.0052，4.1 sd）。
>   **没有证据说 BERT 的簇结构和 diffusion 一样好。**
>
> 🚫 **随之作废**：~~「局部邻域类度量 BERT ≥ diffusion」~~（kNN@1 从未可分辨）、
> ~~「白化后排序翻转」~~（是抹平到 0，不是反转）。
> 另外 §0.3 那个 **24 类 probe（0.804 vs 0.794）也是无误差棒的单点值**，按上表 sd≈0.006
> 只有约 1.6 sd，**不应作为证据引用**；T3 deep/broad 有 100 seeds 波动带，可以引用。
>
> 🟠 **作用域警告：以上只对 270m 成立，8B 上三项指标全部反向。** 见 §0.3 表下那句
> 「270m 上 BERT 反超 diffusion，8B 上 diffusion 反超 BERT——这个交叉超出波动带」：
> 8B deep k=200 是 BERT 0.7093–0.7133 vs diff **0.7198–0.7204**、broad k=100 是 0.669 vs **0.680**、
> 24 类 probe 是 0.816 vs **0.828**。**探针没有在 8B 上跑过**，所以
> **本条不支持任何跨规模的「哪个预训练目标表征更好」结论。**
> 一个**未验证的**假设是这仍与几何一致：§0.2 记录 **8B BERT 的余弦挤到 0.9999**（四条最极端），
> 即 8B BERT 受的几何惩罚比 270m BERT 更重；若如此，8B 上 diffusion 的领先也有一部分来自几何。
> 要区分「BERT 被几何压制」与「8B 上 diffusion 真的更好」，**必须在 8B 两条上补跑探针**。
>
> ⚠️ **本条早期版本曾写「BERT 臂全模型最好的表征是 ESMC encoder 自身、LLaDA decoder 净负贡献」，
> 该结论基于 raw 特征的 probe 读数，已撤回** —— 换缩放后 decoder feature 的 T3 探针达 0.760，
> 远高于任何 ESMC-only 读数。
>
> 一个可能的解释（**未做反事实训练验证**）：BERT 只掩 15%，`residue_cond_mode=add` 下可训练的
> ESMC-300M 看得到 85% 干净序列，任务偏局部，学出的是好的局部邻域结构（对应 `eval_loss`
> 0.2578 远低于 diffusion 0.5638）；diffusion 的 `t~U(0,1)` 经常掩掉大半序列、逼迫 decoder
> 建全局结构，于是簇更成形。支持性观测：BERT 最后一个 block + `ln_f` 把上下文推回输入附近
> （CKA 对 layer0：layer7 0.478 → layer8 0.809），最终层 **92.3%** 方差可由 ESMC 线性预测
> （diffusion 89.0%），BERT 的 eff-rank 单调塌到 11.3 而 diffusion 中间层升到 55.0。
>
> #### (f-3) 已排除的 bug（都查过，都是干净的）
>
> 1. **权重加载**：两臂 `model.safetensors` 键集**逐个相同**（各 386 张量）。
>    ✅ **2026-08-31 已加 fail-fast 守卫**：原先 missing key 只 `logger.warning`，而 decoder 由
>    `LLaDAModelLM(cfg, init_params=False)` 建（分配后不初始化）—— 真缺 key 会拿
>    **未初始化内存**当权重、静默给随机模型打分。现在加载前把所有浮点参数灌 NaN、
>    加载后残留 NaN 即抛错，**权重绑定自动放过、无需白名单**。
>    四条 headline ckpt（270m×2 + 8B×2，后者含 `weight_tying`）全部 **residual NaN = 0，无误报**；
>    删一个权重的负向测试正确抛错。**本表所有已落盘数字因此确认是完整加载下产生的。**
> 2. **padding 污染**：同一条序列 alone / 同长 batch / 被 10 倍长序列 pad 到 125 宽，
>    三种情况下最后一层 pooled feature 余弦 **0.999998**（残差为 bf16 数值噪声）。
>    LLaDA `forward` 把 `attention_mask` 正确转成 additive `-inf` bias；训练与评测同走
>    collator 的 mask（`protein_pretrain_esmc.py` 里没有 `NoAttentionMaskWrapper`）。**无泄漏。**
> 3. **打分口径**：独立 harness 复现出 T2 ARI BERT **0.0184–0.019** / diffusion **0.0270–0.028**
>    （官方 19 点 K sweep），对上本表的 0.0186 / 0.0277。
>
> **本条不改任何已落盘数字，也不改口径 —— headline 保持 raw，重标定已评估并决定不采用。**
> 但引用 §0.1–§0.3 时必须说明：表内数值是
> 「冻骨干 + decoder 最后一层 + **未做任何重标定** 的 mean-pool」下的结果，因此
> ① **绝对值是下界**（T3 上还有约 0.047 的几何损失）；
> ② **两臂差距的成因不统一** —— 线性 probe 上的差距可由几何解释（BERT 信息量相当），
> 但 **T2 聚类上的差距不能**（(f-2) 实测：白化并未提升 BERT 的 ARI）。
> ③ **不能推出「哪个预训练目标更适合表征」**：8B 上胜负与 270m 相反，
> 每格只有 **n=1 次训练**，且两条 best ckpt 步数不同（BERT@49000 vs diff@42000）。
> 📄 **完整调试报告（含两张图、全部原始数字、复现命令）：
> [`debug/BERT_REPR_DEBUG.md`](../../debug/BERT_REPR_DEBUG.md)**。
> 原始数据：`output/repr_diagnostics/{layer_sweep.csv,final_readout_ladder.csv,layer_cka.csv}`、
> `outputs/tcr_representation_paper6/diag_t3_*/`、`outputs/tcr_clustering_embed/diag_*/`；
> 图 `debug/figures/{readout_effect_official,repr_layer_diagnosis}.png`。
> **T1 未测**（单 ckpt 成本高）；其 MLP 头原则上能自己学到重标定，但带 dropout 0.3 + weight decay
> 未必学得动，属未验证的预测。

> 🔧 **(e) 解码步数用错了：argparse 默认 ≠ 论文实际参数（2026-08-25）**
>
> AirGen 的 [`run/zero_shot_test.sh`](../../../airgen/AirGen-Dev/run/zero_shot_test.sh) 才是论文真实设置，各脚本的 argparse 默认值不是：
>
> | 任务 | 官方 | 我们（旧） | 现已改为 |
> |---|---|---|---|
> | CDR infilling | `max_iter 2` + argmax | 4（基线）/ **8**（我们模型） | **2** |
> | Light pairing | **`max_iter 124`** + gumbel + cfg **0/1/1.5** | 32 + gumbel + cfg 0 | **124** + cfg 扫三档 |
>
> **AAR 是逐位准确率**，迭代解码会在已提交（可能错误）的 token 上继续条件化，误差传播会**拉低** AAR。官方 ckpt 实测 SAbDab H3 随步数单调下降：`max_iter` 1/2/4/8 → **42.00 / 41.43 / 40.96 / 40.70**。
>
> **SAb23H2 在 `max_iter=2` 上已复现论文 Table 1**（H2 68.59 vs 68.6、H3 36.70 vs 36.8，六项全在 ±0.9 内），证明模型权重、解码路径、CDR 定义、AAR 算法都正确——原先那 2.3 pp 缺口纯粹是默认值 4。
>
> **SAbDab H3 的剩余缺口已用 Kong 版数据收口**（见 §0.5）：换 3,127 条 Kong 划分后，官方 ckpt H3 43.70 vs 论文 43.55。旧快照 3320 行的诊断不再需要。
>
> **对齐重跑已全部 Success**：CDR Kong + SAb23H2 sweep、pairing `max_iter=124` + 官方 cfg 0/1/1.5。现行数字以 §0.5 / §0.6 为准；`max_iter=8 / 32` 那批不再引用。
> 缺陷 (0) 的 encoder 泄露结论**不受影响**（那是 10+ pp 量级，`max_iter` 只值 1–2 pp）。T4 仍用 `max_iter=32`（TCRT5 是另一套协议，无官方对应值）。

> 🚨 **泄露修复前的 §0.4 / §0.5 / §0.6 旧数字已作废**，根因见下方 (0)。修复后重跑已全部 Success，现行表是修完 encoder 泄露 + 对齐官方 `max_iter` 后的数字。
> **§0.1–§0.3（T1/T2/T3 表征）不受 (0) 影响**——它们是单次前向 mean-pool，没有迭代解码。

**(0) 🚨 ESMC encoder 条件流在迭代解码中泄露参考序列（最严重，所有生成数字作废）**

[`sampling_bioseq.py::_model_logits`](../../dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py) 原先把随解码收缩的 `corruption_mask` 传给 `apply_decoder_corruption_to_encoder`。该函数只在掩码置位处写 `<mask>`，其余位置保留 `batch["encoder_input_ids"]` —— 而那是 collator 从**干净 record** 构造的张量。于是**每提交一个位置，ESMC 条件流就重新暴露该位置的真实参考残基**，decoder 下一步照抄。

实测（单条 light-pairing record，104 个待生成残基）：

| 解码进度 | ESMC 可见的真实参考残基 |
|---|---:|
| 0%（起始） | 0 / 104 |
| 50% 已提交 | **52 / 104** |
| 90% 已提交 | **93 / 104** |

训练不受影响：训练是单次前向，`corruption_mask` 恰好等于全部 target 集合。只有迭代推理才会出现「掩码随步数收缩」。

**修复**：推理改为镜像**整个 `generation_mask`**（全轨迹恒定），ESMC 在任何一步都看不到 target 残基；修复后实测全轨迹 0/104。回归测试 `test_encoder_never_sees_target_residues_during_decoding`。

**这解释了 §0.5 的异常**：CDR AAR 比论文高出 11–14 pp（H3）不是（只是）OAS 语料重叠，而是**解码过程被喂了答案**。同理 §0.6 的 ImmunoMatch 0.648 与 0.946 相同度、§0.4 的 seq-recovery 都不可信。

---

以下 (a)–(d) 是先前定位的问题，(a)(b) 已修复。

**(a) Light pairing 目标长度泄露 → 近重建，不是 de-novo 配对。**
[`grammar/light_chain_pairing.py`](../grammar/light_chain_pairing.py) 用 `antibody_pair_record(heavy, light)` 把**真实 light 装进 batch**，[`grammar/masks.py`](../grammar/masks.py) 的 `light_chain_generation_partial_mask` 只把这些位置改成 mask，因此挖洞数 = 参考 light 长度。实测（4000 对）：

| 诊断 | Ours 8B diff | Ours 270m diff | Ophiuchus `[R]` |
|---|---:|---:|---:|
| 生成长度 == 参考长度 | **100.0%** | **100.0%** | 40.6% |
| 与参考平均相同度 | **0.946** | **0.944** | 0.816 |
| 完全逐字复制 | 4.7% | 9.7% | 0.0% |
| 每条 heavy 的 8 条候选唯一数 | 7.15 | 6.92 | **1.00** |

后果：pairing 的 `chain-match` / `V-gene` / `V-family` / `length` 被平凡抬高，**ImmunoMatch 的天花板就是 ref 的 0.699**。这几格不能当能力，也不能和论文 `[P]` 比。

**(b) Ophiuchus `[R]` 基线 argmax 塌缩 → n=8 是假的。**
[`ophiuchus_eval/light_pairing.py`](../ophiuchus_eval/) 用确定性 `argmax`，每条 heavy 的 8 条候选完全相同（`diversity_mean` = 4.88e-16，唯一数 = 1.00）。论文报的 diversity 是 0.335，说明论文并非纯 argmax。**所以「`[R]` 0.352 vs `[P]` 0.701」很可能是解码配置差异，不是模型差距。**
注：该移植本身是忠实的——它和官方一样用固定 128 槽位 + `<eos>` 截断，**不泄露长度**（实测长度一致率仅 40.6%）。缺陷只在采样策略。

**✅ (a)(b) 已修复；iter=124 对齐重跑已收口（2026-08-25）**

对照官方 [`AirGen-Dev/downstream/comp_chain/generate_light_from_csv.py`](../../../airgen/AirGen-Dev/downstream/comp_chain/generate_light_from_csv.py)：官方 `LightMaskingCollate` 用**固定 128-token light 缓冲区**，`light_tokens[i, 4:]` 全部置 mask，解码后按 `<eos>` 截断，长度由模型自己决定。

- **(a) 修复**：[`grammar/light_chain_pairing.py`](../grammar/light_chain_pairing.py) 新增 `--light-length-mode`，默认 **`prior`** —— 从 OAS **train** split（300 万行）统计的长度直方图采样，与本行参考无关（先验 `data/downstream/comp_chain/oas_train_light_length_prior.json`，mean 108.86 / median 108 / 范围 91–141）。grammar 块没有链内终止符，所以用「与参考无关的长度先验」替代官方的「模型自己吐 EOS」。产物新增 `light_length_mode` / `target_light_length` / `ref_light_length` 三列可审计。旧的 `reference` 模式保留但会打印泄露警告。
- **(b) 修复**：[`ophiuchus_eval/run_eval.sh`](../ophiuchus_eval/run_eval.sh) 新增 `PAIRING_SAMPLING`，默认 `gumbel_argmax`（与我们同口径），产物按策略命名，旧 argmax 结果保留作 provenance。
- **自动化防回归**：[`scripts/downstream/pairing_leakage_diagnostic.py`](../../scripts/downstream/pairing_leakage_diagnostic.py) 每次 pairing 跑完自动输出 `same_length_frac` / `mean_identity` / `exact_copy_frac` / `unique_per_heavy` 与 verdict，不再依赖人工抽查。
**✅ (b) 重跑已完成 —— 基线缺口基本消失（`t-20260824234946-r9bx2` Success）**

把 Ophiuchus 官方 ckpt 的解码从 `argmax` 换成 `gumbel_argmax` 后（其余协议不变），本地复跑与论文 Table 3 已经基本吻合：

| Ophiuchus-Ab prompt3 | ImmunoMatch↑ | Better↑ | Chain | V | J | V-fam | Div.↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 论文 `[P]` | 0.695 | 49.2% | 99.7% | 0.356 | 0.273 | 0.894 | 0.335 |
| `[R]` argmax（旧，**塌缩**） | 0.352 | 24.4% | 99.8% | 0.392 | 0.350 | 0.904 | 4.9e-16 |
| **`[R]` gumbel_argmax（新）** | **0.641** | 43.2% | 99.7% | 0.316 | 0.275 | 0.862 | **0.374** |

诊断：`same_length_frac` 0.353、`unique_gen_per_heavy` **7.98/8**、`length_leak_suspected=false`、`sampler_collapse_suspected=false`。

> 结论：**先前 0.352 vs 论文 0.701 的巨大缺口完全是解码配置造成的**，不是模型差距。换成随机采样后 ImmunoMatch 差距收窄到 −0.054，J-gene（0.275 vs 0.273）与 chain-match（99.7% vs 99.7%）几乎完全一致。这条现在是可用的官方基线。

- **Ours 侧重跑**：因随后发现缺陷 (0)，pairing 与 CDR/T4 一并并入完整 gen 作业重跑（见 §0.0 顶部任务号）。旧的两条 pairing-only 作业已取消/删除。

**(d) ⚠️ 8B `checkpoint-40000` 已被删除，§0 的 8B 行不可复现。**
8B 两条训练随后跑完 50k，top-k 剪枝按 `save_top_k=3` 删掉了 40000。现存最优是 **8B diffusion `checkpoint-45000`（eval_loss 0.5348，优于旧 0.5411）** 与 **8B BERT `checkpoint-43000`（0.2566，优于旧 0.2624）**。270m 两条（BERT 49000 / diffusion 42000）仍在磁盘上。因此：**§0.1–§0.5 里的 8B 数字对应的权重已不存在**，数值本身有效但无法复算；pairing 重跑已改用 45000，其余 8B 任务需在新 ckpt 上重跑后才能与 270m 并列。

排除掉的嫌疑（已核查，ImmunoMatch 打分器本身无误）：4000 行全部被 abnumber 注释成功（2528 κ / 1472 λ），未触发「无法注释记 0」分支；κ/λ 模型按被打分序列自身类型路由；`max_length=256` 截断未触发（最长 251）。

**(c) ✅ CDR AAR 偏高已解释清楚，不再是悬案。** 当初的怀疑是「OAS 语料重叠」（Ophiuchus 原文即把 AntiBERTy 的 H1/H2 优势归因于此）。实测结论是**两个更平凡的原因**，都已修复：

1. **encoder 泄露**（缺陷 (0)）——解码过程被喂了答案。修掉后 SAbDab H3 从 57.95 掉到 42.03，SAb23H2 H3 从 51.31 掉到 35.24。
2. **SAbDab 数据版本**（详见 §0.5）——我们用的是 2024+ 快照，含 44% 的 2023 年后长 H3 结构。换成论文口径的 Kong 版数据后，官方 ckpt 复现论文（H3 43.70 vs 43.55）。

修完这两项，Ours 与官方 ckpt 在两个数据集上**基本打平（±1.5 pp）**，不再有当初那种 11–14 pp 的异常优势。

> ⚠️ **但「不需要专项泄露报告」这个结论要撤回（2026-08-30）。** 原推理是「既然没有异常优势，就不必再查语料重叠」。这个推理无效：本轮 ckpt 训练时 `--asd_antibody_benchmark_blocklist ""` 使**抗体侧去污染被完全禁用**（见顶部审计警告），所以训练语料**可能**含 SAbDab Kong 折与 SAb23H2 的测试抗体，而我们从未实测过重叠率。「没看到异常尖峰」不等于「数据干净」——只说明若存在污染，它没制造出可见的异常。
>
> 这一点对结论的方向很关键：Ophiuchus 在论文里是**无重叠 zero-shot**，我们是**去污染被禁用**。因此「基本打平」应读作我们干净性能的**上界**，而不是一个对称的平局。§0.5 里任何形式的「干净打平」措辞都不成立，已从本节移除。
>
> 🚨 **实测已完成（2026-08-30），污染是实的且量级很大。** 脚本 `scripts/ab_cdr_leakage.py`，产物 `outputs/ab_cdr_leakage_report.json`。训练混合是 `oas+ots+asd_antibody+trait+tcr_native`，其中 `ots` / `trait` 是 TCR 源，抗体侧只有 **ASD-antibody**（897,364 行 → 707,800 唯一重链）与 **OAS**（2,498,995 行 → 2,457,282 唯一重链）两个。
>
> | 判据 | Kong 折 / ASD | Kong 折 / OAS | SAb23H2 / ASD | SAb23H2 / OAS |
> |---|:---:|:---:|:---:|:---:|
> | 重链完全相同 | 89（2.85%） | 0（0%） | **15（25.00%）** | 0（0%） |
> | 重链 Lev≤1 | 1,357（43.40%） | 0（0%） | 16（26.67%） | 0（0%） |
> | **CDR-H3 core 精确相同** | **1,754（56.09%）** | 81（2.59%） | 15（25.00%） | 0（0%） |
> | CDR-H3（测试文件注释序列） | **1,702（54.43%）** | 75（2.40%） | n/a（只有 fasta） | n/a |
> | 全链同一性 ≥0.99 | 1,357（43.40%） | 0（0%） | 16（26.67%） | 0（0%） |
> | 全链同一性 ≥0.95 | **1,728（55.26%）** | 125（4.00%） | 16（26.67%） | 2（3.33%） |
> | 〔对照〕≥0.99 且 **H3 不同** | 26（**0.83%**） | 0（0%） | 1（1.67%） | 0（0%） |
> | 〔对照〕≥0.95 且 **H3 不同** | 346（11.06%） | 125（**4.00%**） | 2（3.33%） | 2（**3.33%**） |
> | 〔对照〕最高同一性中位 | 0.9831 **vs 0.8667** | 0.8760 **vs 0.8760** | 0.8664 vs 0.8559 | 0.8571 vs 0.8571 |
>
> **胚系背景对照是这份实测的关键。** 抗体重链共享胚系框架，无关抗体之间同一性本就有 ~0.87，所以「同一性 ≥0.95」单独看毫无说服力。对照做法：只在 **CDR-H3 不同**（即可证明不是同一克隆）的训练抗体里取最高同一性。两个语料给出完全相反的读数：
> - **ASD**：≥0.99 的背景仅 **0.83%**，实际 43.40%——**52 倍富集**；最高同一性中位 0.9831 vs 背景 0.8667。
> - **OAS**：观测值与对照**逐档完全相同**（≥0.95 都是 4.00%，中位都是 0.8760）。即 OAS 的高同一性命中**全部**是胚系背景，**零抗体级泄露**。那 2.59% 的 H3 命中在全链同一性上没有任何信号，说明是短/收敛 H3 落在不同框架上，不是同一抗体。
>
> 也就是说，**可测得的抗体侧污染全部来自 ASD-antibody**，与 OAS 无关——这一点是可操作的：只对 ASD 启用 blocklist 即可消除。
>
> **为什么不能只看「完全相同 2.85%」**：两边对可变域的截取边界不同，同一抗体常以偏移形式出现。实例 Kong `6myy`——测试文件存 `VQLVQSGAEVKKPGASVKVSC…`（118 aa），ASD 存 `QVQLVQSGAEVKKPGASVKVSC…`（121 aa），差一个 N 端残基加一个内部替换（Lev=2），`exact_heavy` 与 `lev1_heavy` 都会漏掉，但它显然是同一抗体。所以必须用对边界不敏感的判据。两个互相独立的判据（H3 精确 54.43% 与全链 ≥0.95 55.26%）在 ASD 上收敛到同一个 ~55%，结论不是判据选择的产物。
>
> 另注：Kong 折的 3,127 个条目只有 **2,134 条不同的重链**（同一抗体对多个抗原成像），所以任何按重链去重的统计都不能直接当抗体数用。
>
> **对 §0.5 的影响**：SAbDab Kong 折上**过半**测试抗体的 CDR-H3 原样出现在抗体侧训练语料里，而 §0.5 的主任务正是 CDR-H3 infilling。这不是「可能有污染」而是「已确认污染，且覆盖过半评测集」。§0.5 的 Ours 行**不能**作为泛化性能引用，只能作为上界；且与 Ophiuchus（无重叠 zero-shot）的「基本打平」在方向上完全不对称。SAb23H2 侧污染面小得多（25–27%，且全部来自 ASD），其最高同一性中位已与背景持平（0.8664 vs 0.8559），说明剩下的 ~73% 是干净的，该表比 SAbDab 表更可引用。
> 仍待做：**T4-held20** 的泄露报告（那批 `bioseq_seen`/`bioseq_unseen` 分区是按旧 grammar_v2 语料算的，对本轮 immune 语料无效，见 §0.4）。

### §0.1 TCR T1 Binding — retrain 五折（冻骨干 + 每 fold MLP）

> 数据 = Figshare `retrain.zip` 五个 AS fold。Baseline = 官方发布的 retrained checkpoint 本地推理（**我们不重训它们**）；Ours = 每 fold 用该 fold 的 `i_1_1train.csv` 只训 MLP 头。mean±sample-std over 5 folds。
> **这是 T1 主协议。** original-only（零训练现成 ckpt）见 §T1，**禁止混排**。

| Method | seen_test AUROC | seen_test AUPRC | seen_ind AUROC | seen_ind AUPRC | 类型 |
|---|---:|---:|---:|---:|---|
| epiTCR | **0.8142±0.0013** | **0.8345±0.0014** | **0.6556±0.0027** | **0.6976±0.0014** | `[P]` Supp Table 7 |
| TEIM | 0.7766±0.0028 | 0.7932±0.0032 | 0.6090±0.0071 | 0.6224±0.0088 | `[P]` |
| ATM-TCR | 0.7728±0.0013 | 0.7800±0.0049 | 0.6028±0.0077 | 0.5838±0.0080 | `[P]` |
| ERGO-AE | 0.7388±0.0033 | 0.7519±0.0035 | 0.5636±0.0102 | 0.5711±0.0086 | `[P]` |
| NetTCR | 0.7372±0.0053 | 0.7482±0.0054 | 0.5626±0.0106 | 0.5563±0.0100 | `[P]` |
| TCR-H | 0.6865±0.0043 | 0.6904±0.0056 | 0.5593±0.0059 | 0.5562±0.0071 | `[P]` |
| TEINet | 0.5770±0.0695 | 0.5887±0.0801 | 0.5111±0.0191 | 0.5164±0.0183 | `[P]` |
| ERGO-lstm | 0.5022±0.0027 | 0.5012±0.0005 | 0.4987±0.0038 | 0.4981±0.0033 | `[P]` |
| epiTCR（官方 ckpt 复跑） | 0.8047±0.0015 | 0.8253±0.0016 | 0.6662±0.0030 | 0.7080±0.0007 | `[R]` 附行；seen 相对论文偏移 |
| TEIM（官方 ckpt 复跑） | 0.7641±0.0029 | 0.7802±0.0035 | 0.6171±0.0042 | 0.6303±0.0073 | `[R]` |
| ATM-TCR（官方 ckpt 复跑） | 0.7644±0.0017 | 0.7707±0.0044 | 0.6067±0.0071 | 0.5881±0.0083 | `[R]` |
| ERGO-AE（官方 ckpt 复跑） | 0.7286±0.0040 | 0.7405±0.0043 | 0.5701±0.0104 | 0.5788±0.0090 | `[R]` 静默丢行 |
| NetTCR（官方 ckpt 复跑） | 0.7271±0.0050 | 0.7371±0.0056 | 0.5713±0.0116 | 0.5630±0.0099 | `[R]` |
| TCR-H（官方 ckpt 复跑） | 0.6786±0.0044 | 0.6805±0.0054 | 0.5674±0.0067 | 0.5596±0.0075 | `[R]` |
| TEINet（官方 ckpt 复跑） | 0.5818±0.0768 | 0.5906±0.0848 | 0.5226±0.0211 | 0.5250±0.0245 | `[R]` |
| ERGO-lstm（官方 ckpt 复跑） | 0.5016±0.0018 | 0.4999±0.0007 | 0.4998±0.0022 | 0.4981±0.0038 | `[R]` |
| **Ours-BERT** 270m@49000 | 0.7366±0.0036 | 0.7522±0.0050 | 0.5778±0.0064 | 0.5693±0.0063 | `[L]` frozen+MLP |
| **Ours-Diffusion** 270m@42000 | 0.7527±0.0032 | 0.7678±0.0035 | 0.5853±0.0053 | 0.5852±0.0068 | `[L]` |
| **Ours-BERT** 8B@43000 | 0.7549±0.0047 | 0.7695±0.0053 | 0.5907±0.0091 | 0.5611±0.0052 | `[L]` |
| **Ours-Diffusion** 8B@45000 | **0.7617±0.0034** | **0.7776±0.0031** | **0.6037±0.0040** | **0.5977±0.0052** | `[L]` |
| **Ours-Diffusion** v3@27000 | 0.7468±0.0037 | 0.7627±0.0038 | 0.5945±0.0077 | 0.5886±0.0077 | `[L]` 七源，≠旧 270m |
| **Ours-Diffusion** v3@42000 | 0.7510±0.0021 | 0.7666±0.0028 | 0.5752±0.0040 | 0.5646±0.0091 | `[L]` 七源，≠旧 270m |
| **Ours-Diffusion** allch@33000 | 0.7496±0.0028 | 0.7656±0.0029 | 0.5984±0.0101 | 0.5883±0.0127 | `[L]` 全链，≠组 C |
| **Ours-Diffusion** allch@18k | 0.7451±0.0008 | 0.7606±0.0019 | 0.5951±0.0085 | 0.5933±0.0043 | `[L]` 全链续训快照 |
| **Ours-Diffusion** allch@26k | 0.7450±0.0023 | 0.7609±0.0023 | 0.5921±0.0078 | 0.5783±0.0057 | `[L]` 全链续训最新 |

> **🔻 2026-09-02 论文值优先**：上表 baseline **主榜是 `[P]`（Supplementary Table 7，`PAPER_EXPECTED_AS`）**；原先独占主表的官方 ckpt 复跑降为 `[R]` 附行。seen_test 上 7/8 `[R]` 低于论文约 0.010，只摆复跑会抬高 Ours 身位。补上 `[P]` 后 TEIM 0.7932 / ATM-TCR 0.7800 / epiTCR 0.8345，Ours-Diffusion 8B 的 0.7776 不动，**落后面扩大**（现低于 ATM-TCR 论文值）。seen 两列仍不得声称与论文对齐（系统偏移未定论）；`[R]` 附行就是用来显示该偏移的。

> **两个 8B 行已于 2026-08-28 修正（原为跨 checkpoint 拼接）。** 修正前 AUROC 抄自已被 top-k 淘汰的 `checkpoint-40000`、AUPRC 抄自行标所示的 43000/45000，AUPRC 一律缺 ± 正是因为两列来源不同。现全部取自单一 checkpoint 的 `summary.json`（`outputs/tcr_binding_nm2025_retrained/ours_fusion_8b_{bert_43000,diff_45000}/cdr3b/AS/summary.json`），无需重跑。改动幅度 ≤0.004，结论不变。

> ⚠️ **两个 seen 集上基线复现存在系统性偏移，方向相反，机制未定（2026-08-30 调查）。** 同一批官方权重、官方推理代码、官方数据文件，三个评测集表现出三种行为：
>
> | 评测集 | 8 个基线的 Δ(local−paper) AUROC | 同向数 | std 比值 |
> |---|---|---:|---|
> | `unseen_independent` | −0.0056 … +0.0011（6 个为 ±0.0000） | — | 6/8 恰为 1.00 |
> | `seen_test` | −0.0125 … +0.0048 | **7/8 为负**（≈−0.010） | 多数 >1（0.66–1.33） |
> | `seen_independent` | +0.0011 … +0.0115 | **8/8 为正**（≈+0.008） | 0.59–1.14 |
>
> **已排除的解释**：
> - **不是聚合口径**：unseen 用同一套「逐折算 AUROC 再平均」的实现，均值与 std 都与论文逐位相同，说明该口径正确。
> - **不是数据泄露**：实测五折 `test ∩ train` 精确 (CDR3β, Epitope) 配对**全部为 0**；`seen_independent`（5,882 行 / 80 表位）与 `unseen_independent`（3,162 行 / 211 表位）对五折 train 并集的精确配对重叠也都是 **0**。表位重叠符合设计：seen 侧 80/80 全重叠，unseen 侧 0/211。（偏移与泄露无关；但见下方近重复一节——精确重叠为零**并不**等于 seen 侧干净。）
> - **不是文件选错**：官方 zip 的 AS 目录下每个 fragment 只有**一个** `1_1_1independent_test.csv`，`resolve_test_member` 无歧义；`seen_test` 用逐折 `{fold}_1_1test.csv`，也唯一。
> - **不是单模型训练噪声**：符号在 **8 个架构完全不同的方法**上一致（seen_test 7/8 负、seen_independent 8/8 正）。若来自各自的训练随机性，符号应大致各半。
>
> - **不是「汇总池化 vs 逐折平均」**：已实测。把五折预测拼成一条曲线再算 AUROC，与逐折平均的差异 |Δ| ≤ 0.005（多数 <0.002），完全无法解释 0.010 的偏移。该假设**已否**。
> - **不是行覆盖缺失**：逐折比对产物 `predictions.csv` 与官方测试文件的行数，8 个基线里 **7 个三集五折全部逐一相等**（seen_test 30670/30660/30384/30282/30268、seen_ind 5882、unseen_ind 3162）。
>
> **顺带发现（单独问题）：ERGO-AE 会静默丢行。** 它是唯一行数对不上的方法——seen_test 每折少 10–34 行、seen_independent 少 32 行、unseen_independent 少 12 行，应为其自带编码器拒绝了部分序列。这不解释系统性偏移（其余 7 个方法零丢行却同样偏移），但 ERGO-AE 的三个数值都是在略小的子集上算的，引用时需注明。
>
> **剩余候选**（未验证）：发布的 checkpoint 与生成 Supplementary Table 7 的那一次训练可能不是同一批；或论文对 seen 侧另有未公开的后处理。**在定论前，§0.1 的 seen 两列只应作为「同一口径下的相对比较」使用，不应声称与论文数值对齐。** unseen 一栏不受影响，可直接引用。

> 🚨 **近重复实测：精确重叠为 0，但 seen 侧有 12–13% 的近重复（2026-08-30 新增）。** 「exact overlap = 0」此前被当作切分干净的证据。实测 Lev≤1 后这个结论只对 unseen 成立。判据：表位精确相同 **且** CDR3β 与该表位下某条训练 CDR3β 的编辑距离 ≤1（换表位是换任务，故只在同表位组内比对）。脚本 `scripts/t1_retrained_near_duplicate.py`，产物 `outputs/t1_retrained_near_duplicate.json`。
>
> | 评测集 | 精确重叠 | **Lev≤1 近重复** |
> |---|---:|---:|
> | seen_test fold1–5 | 0 / 0.00% | **3,809–3,975 / 12.42%–13.13%** |
> | seen_independent | 0 / 0.00% | **260 / 4.42%** |
> | unseen_independent | 0 / 0.00% | **0 / 0.00%** |
>
> **seen_test 每 8 条里就有 1 条与某个训练样本只差一个残基**，seen_independent 为 4.42%。这不是切分错误——官方按精确配对去重，做到了它声称的事——但**「零重叠」这句话确实高估了 seen 侧的洁净度**，任何基于 seen 列的泛化声称都必须带上这个数字。
> **unseen 侧则是真正干净的**：3,162 条配对的表位**全部**不在训练集中（`n_test_pairs_with_unseen_epitope = 3162`），所以 Lev≤1 近重复必然为 0。这与上文「unseen 复现到 5×10⁻⁵」一起，使 unseen 成为本节唯一可放心引用的一栏。

> ⚠️ **AUPRC 口径：本节用的是梯形 PR-AUC（`precrec`），它单向地略微抬高我们（2026-08-30）。** 产物同时记录了 `auprc_precrec_mean`（梯形积分）与 `average_precision_mean`（阶梯式 AP，sklearn 推荐、无插值偏置），表里取的是前者。45 组对照下两者中位差仅 3.9×10⁻⁴，但**分布不对称**：官方基线上均值 **−0.0001**（24 组里仅 6 组为正，基本中性），我们的模型上均值 **+0.0012**、最大 **+0.0091**（`bioseq7l` seen_test 0.6699 vs 0.6608；8B BERT seen_test 0.7712 vs 0.7680）。梯形法在 PR 空间做线性插值本身是有偏的，得分并列越多偏得越大，我们的「冻骨干 + MLP 头」并列更多，所以吃到了这份偏置。**幅度远小于表内差距（≥0.01），结论不变**，但若要报单一权威数字应改用 `average_precision`，或至少注明口径。论文用哪种未公开，故 Δ 列同样带这份不确定性。

**unseen_independent（新表位泛化，随机≈0.5）**

> ✅ **八个官方基线已补齐（2026-08-30）**。此前本表只有 Ours 四行，无法判断我们在 unseen 上的位置。源：`outputs/tcr_binding_nm2025_retrained/*/cdr3b/AS/summary.json`，同一批产物、无需重跑。

| Model | AUROC | AUPRC | 论文 AUROC | Δ AUROC / AUPRC | 类型 |
|---|---:|---:|---:|---:|---|
| TEIM | **0.5337±0.0060** | **0.5233±0.0054** | 0.5326±0.0052 | +0.0011 / +0.0015 | `[R]` |
| TCR-H | 0.5269±0.0122 | 0.5267±0.0098 | 0.5269±0.0122 | +0.0000 / −0.0000 | `[R]` |
| **Ours-BERT** 270m@49000 | 0.5245±0.0100 | 0.5145±0.0095 | — | — | `[L]` |
| NetTCR | 0.5203±0.0156 | 0.5156±0.0125 | 0.5203±0.0156 | −0.0000 / +0.0000 | `[R]` |
| **Ours-Diffusion** 270m@42000 | 0.5202±0.0139 | 0.5134±0.0108 | — | — | `[L]` |
| ATM-TCR | 0.5192±0.0108 | 0.5136±0.0125 | 0.5192±0.0108 | −0.0000 / −0.0000 | `[R]` |
| **Ours-BERT** 8B@43000 | 0.5156±0.0120 | 0.5143±0.0110 | — | — | `[L]` |
| **Ours-Diffusion** 8B@45000 | 0.5128±0.0094 | 0.5104±0.0075 | — | — | `[L]` |
| **Ours-Diffusion** v3@27000 | 0.5337±0.0108 | 0.5242±0.0081 | — | — | `[L]` 七源 |
| **Ours-Diffusion** v3@42000 | 0.5201±0.0122 | 0.5156±0.0090 | — | — | `[L]` 七源 |
| **Ours-Diffusion** allch@33000 | 0.5138±0.0096 | 0.5131±0.0097 | — | — | `[L]` 全链 |
| **Ours-Diffusion** allch@18k | 0.5010±0.0122 | 0.5001±0.0104 | — | — | `[L]` 全链续训快照 |
| **Ours-Diffusion** allch@26k | 0.5178±0.0091 | 0.5171±0.0087 | — | — | `[L]` 全链续训最新 |
| ERGO-AE | 0.5099±0.0088 | 0.5106±0.0024 | 0.5099±0.0088 | −0.0000 / −0.0000 | `[R]` |
| epiTCR | 0.5075±0.0086 | 0.5106±0.0060 | 0.5075±0.0086 | +0.0000 / −0.0000 | `[R]` |
| ERGO-lstm | 0.5009±0.0011 | 0.5049±0.0048 | 0.5009±0.0011 | +0.0000 / −0.0000 | `[R]` |
| TEINet | 0.4969±0.0085 | 0.4995±0.0066 | 0.5025±0.0097 | −0.0056 / −0.0062 | `[R]` |

> ★ **这一栏的复现精度是全项目最高的**：6/8 基线的 AUROC **与 AUPRC 的均值和标准差全部与论文逐位相同**（Δ ≤ 5×10⁻⁵，即在论文发布的四位小数下完全一致；std 比值恰为 1.00）。TEIM +0.0011、TEINet −0.0056 是唯二例外。这**不是巧合也不是自我比较**：论文值硬编码在 `tcr_binding/retrained_protocol.py::PAPER_EXPECTED_AS`，取自 Supplementary Table 7，与本地计算路径完全独立。它同时证明「固定 independent 文件 + 5 个官方 fold 权重 + 逐折 AUROC 后平均」这套聚合口径**就是论文的原口径**。
>
> **我们在 unseen 上排第 3 / 第 5 / 第 7 / 第 8（共 12 行）**，全部落在 0.5128–0.5245，与中位基线同档。**但整栏都贴着随机**（最好的 TEIM 只有 0.5337，最差的 TEINet 0.4969 低于随机），所以名次不具区分度——这是该基准公认的难点，不是我们的成绩。**不应把「排第 3」写进任何结论**；可写的是「与全部已发表方法一样，在新表位上接近随机」。
>
> 两个 8B 行的 AUROC 原同样抄自 `checkpoint-40000`，已改回单一 checkpoint。另修正加粗——原先把该列**最低**的 0.5104 标为最优，现改标真正的最优值。

> 结论（**2026-09-02 按 `[P]` 主榜重述**）：Ours 最好一档（8B diffusion，seen_test AUPRC 0.778）**低于全部已发表最强方法的论文值**（epiTCR 0.835 / TEIM 0.793 / ATM-TCR 0.780），也低于 TEIM/ATM-TCR 的 `[R]` 复跑（0.780 / 0.771）。seen_independent 远低于 epiTCR 论文值。**unseen 全部 ≈0.51–0.52，接近随机**。diffusion 略优于 BERT；8B 相对 270m 增益很小。瓶颈不在规模而在「冻骨干 + 全局 mean-pool」。
> T1 只在四条 best ckpt 上跑过（单 ckpt ~113 分钟），未做 top-k 波动带；按 T2/T3 的波动（≤0.003）推测应同样稳定，但未实测。

### §0.2 TCR T2 Clustering

**基础 B — TCREmbedding embedding→K-means（9,033 CDR3β / 25 表位，K∈{10..100}）**

| Model | step | ARI mean-over-K | ARI best | NMI mean | NMI best | Purity mean | Purity best | 类型 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SCEPTR（TCR 专用对比学习） | — | **0.033** | 0.063 | **0.159** | 0.186 | **0.339** | 0.370 | `[L]` 参照 |
| TCR-BERT（TCR 专用 LM） | — | 0.016 | 0.021 | 0.095 | 0.120 | 0.287 | 0.304 | `[L]` 参照 |
| k-mer(2) 组成 | — | 0.010 | — | 0.079 | — | 0.268 | — | `[C]` |
| ESM2-150M | — | 0.007 | — | 0.068 | — | 0.258 | — | `[L]` 参照 |
| ProtBERT | — | 0.006 | — | 0.061 | — | 0.248 | — | `[L]` 参照 |
| One-hot AA 组成 | — | 0.007 | — | 0.064 | — | 0.254 | — | `[C]` |
| BLOSUM62 | — | 0.004 | — | 0.056 | — | 0.240 | — | `[C]` |
| Atchley 五因子 | — | 0.002 | — | 0.047 | — | 0.233 | — | `[C]` |
| Ours-BERT 270m | @41000 | 0.0184 | — | 0.1149 | — | 0.2968 | — | `[L]` |
| | @44000 | 0.0184 | — | 0.1145 | — | 0.2975 | — | `[L]` |
| | **@49000** ★ | 0.0186 | 0.0292 | 0.1160 | 0.1380 | 0.2979 | 0.3182 | `[L]` |
| | @50000 | 0.0187 | — | 0.1154 | — | 0.2975 | — | `[L]` |
| Ours-Diffusion 270m | @39000 | 0.0288 | — | 0.1353 | — | 0.3122 | — | `[L]` |
| | **@42000** ★ | **0.0277** | **0.0437** | 0.1347 | 0.1497 | 0.3117 | 0.3274 | `[L]` |
| | @47000 | 0.0283 | — | 0.1347 | — | 0.3122 | — | `[L]` |
| | @50000 | 0.0269 | — | 0.1346 | — | 0.3110 | — | `[L]` |
| Ours-BERT 8B | **@43000** ★ | 0.0214 | — | 0.1305 | — | 0.3109 | — | `[L]` |
| | @44000 | 0.0217 | — | 0.1306 | — | 0.3105 | — | `[L]` |
| | @46000 | 0.0220 | — | 0.1291 | — | 0.3113 | — | `[L]` |
| | @50000 | 0.0215 | — | 0.1292 | — | 0.3100 | — | `[L]` |
| Ours-Diffusion 8B | @44000 | 0.0241 | — | 0.1423 | — | 0.3178 | — | `[L]` |
| | **@45000** ★ | 0.0249 | — | **0.1452** | — | 0.3224 | — | `[L]` |
| | @48000 | 0.0249 | — | 0.1430 | — | 0.3197 | — | `[L]` |
| | @50000 | 0.0244 | — | 0.1448 | — | **0.3238** | — | `[L]` |
| Ours-Diffusion v3 | **@27000** | 0.0169 | 0.0230 | 0.1079 | 0.1332 | 0.2911 | 0.3093 | `[L]` 七源 |
| | **@42000** | 0.0205 | 0.0320 | 0.1188 | 0.1406 | 0.3023 | 0.3207 | `[L]` 七源 |
| Ours-Diffusion allch | **@33000** | 0.0224 | 0.0368 | 0.1295 | 0.1509 | 0.3094 | 0.3322 | `[L]` 全链 |
| | **@18000** | 0.0225 | 0.0362 | 0.1230 | 0.1431 | 0.3038 | 0.3235 | `[L]` 全链续训快照 |
| | **@26000** | 0.0222 | 0.0342 | 0.1237 | 0.1468 | 0.3043 | 0.3259 | `[L]` 全链续训最新 |

★ = 该 run 内 `eval_loss` 最低那一步（headline）。**top-k 波动带**：同一 run 换 step，ARI ±0.001–0.002、NMI ±0.002、Purity ±0.003。模型间差异（diffusion vs BERT 在 270m 上 +0.009 ARI）**大于波动带，是真实差异**。

**基础 A — NAR-GAB 阈值曲线（`embed-threshold`）**
> **不并入官方 9 方法排序**。官方 9 方法是论文 Retention/Purity `[P]`（clusTCR 0.09/0.99、TCRMatch 0.09/0.82、GLIPH2 0.22/0.80、GIANA 0.25/0.65、iSMART 0.18/0.75、TCRdist3 0.44/0.42、DeepTCR 0.92/0.63、HD 0.27/0.69、LD 0.31/0.57，取自 Fig 3A；**9/9 本地复算吻合**，最大 |Δ| 0.005，见 §T2(a)）；下表是我们 embed 曲线在同 retention 处取点。
>
> 🚨 **下表数字已过期（2026-08-30），待重跑，暂不可引用。** 它跑在错误的 **5,368 配对** universe 上——`curate()` 有一处 pandas/dplyr 语义差异漏掉了 `PubMed_ID` 缺失行的剔除，多留 634 行 / 589 个配对。修复后 universe 为 **4,779 配对**，与论文四个常量逐一精确一致（详见 §T2 开头）。官方九方法的 `[P]` 值与 §T2(a)(b) 已在新 universe 上重算，本表因每行都需重做 GPU 嵌入而未重跑（Ours 侧数字统一留给 v2 checkpoint）。取点锚 `ret≈0.19/0.39` 也已随新 universe 上移，故列名同样不再对应。

| Model | purity @ ret≈0.19 (GLIPH2 点) | purity @ ret≈0.39 (TCRdist3 点) |
|---|---:|---:|
| 官方 GLIPH2 `[P]` | 0.80 (ret 0.22) | — |
| 官方 TCRdist3 `[P]` | — | 0.42 (ret 0.44) |
| **Ours-BERT** 270m@49000 | 0.968 (ret 0.191) | 0.702 (ret 0.388) |
| **Ours-Diffusion** 270m@42000 | 0.972 (ret 0.191) | 0.726 (ret 0.390) |
| **Ours-BERT** 8B@43000 | 0.965 (ret 0.197) | 0.607 (ret 0.432) |
| **Ours-Diffusion** 8B@45000 | **0.975** (ret 0.189) | 0.703 (ret 0.391) |

**v3 基础 A（4,779 配对新 universe，可引用；勿与上表旧 5,368 行混排）**

| Model | purity @ ret≈0.19 (GLIPH2) | purity @ ret≈0.39 (TCRdist3) |
|---|---:|---:|
| 官方 GLIPH2 `[P]` | 0.80 (ret 0.22) | — |
| 官方 TCRdist3 `[P]` | — | 0.42 (ret 0.44) |
| **Ours v3@27000** | 0.948 (ret 0.190) | 0.626 (ret 0.386) |
| **Ours v3@42000** | 0.930 (ret 0.192) | 0.588 (ret 0.389) |
| **Ours allch@33000** | 0.961 (ret 0.192) | 0.676 (ret 0.388) |
| **Ours allch@18k** | 0.955 (ret 0.190) | 0.632 (ret 0.391) |
| **Ours allch@26k** | 0.950 (ret 0.190) | 0.619 (ret 0.389) |

> 结论：ARI 全部落在 **0.018–0.029**，**高于所有通用 PLM（ESM2 0.007 / ProtBERT 0.006 / k-mer 0.010）与 TCR 专用的 TCR-BERT（0.016），但低于 SCEPTR（0.033）**——与「TCR 专用对比学习 > 自监督通用表征」一致。diffusion 稳定优于 BERT；**8B 相对 270m 无增益，diffusion 端甚至略低（0.025 vs 0.028）**。8B BERT 在阈值曲线上 tau 分辨率退化（余弦相似度挤在 0.9999 附近，TCRdist3 点 retention 溢出到 0.432、purity 掉到 0.607），是**嵌入各向异性**问题，不是聚类算法问题。

### §0.3 TCR T3 Representation（few-shot per-epitope NN AUROC，冻表征、不训头）

**deep track — paper6 六 pMHC，universe 25,816，100 seeds，macro AUROC**

> ⚠️ **各方法输入字段不同，比较前先看「输入」列**（2026-08-30 逐源码核对，详见 T3 节的输入字段对照）。

| Method | 类型 | 输入 | k=1 | k=5 | k=20 | k=100 | k=200 |
|---|---|---|---:|---:|---:|---:|---:|
| **SCEPTR** | 专用对比学习(官方权重) | β+α**+V** | **0.640** | **0.702** | **0.738** | **0.775** | **0.790** |
| TCRdist (tcrdist3) | 序列比对(官方源码) | β+α**+V+J** | 0.647 | 0.701 | 0.733 | 0.764 | 0.777 |
| CDR3-Levenshtein-NN | 序列比对(训练free) | β+α | 0.626 | 0.671 | 0.696 | 0.722 | 0.733 |
| k-mer(3) 组成 | 训练free | β+α | 0.607 | 0.665 | 0.690 | 0.717 | 0.728 |
| ESM2-150M | 通用蛋白 LM | β+α | 0.591 | 0.631 | 0.656 | 0.685 | 0.696 |
| ProtBERT | 通用蛋白 LM | β+α | 0.585 | 0.624 | 0.648 | 0.681 | 0.694 |
| TCR-BERT | TCR 专用 LM | β only | 0.579 | 0.631 | 0.656 | 0.681 | 0.692 |
| **Ours-BERT** 270m@49000 ★ | `[L]` fusion mean-pool | β+α | 0.591 | 0.642 | 0.671 | 0.700 | 0.712 |
| **Ours-Diffusion** 270m@42000 ★ | `[L]` | β+α | 0.594 | 0.648 | 0.668 | 0.695 | 0.709 |
| **Ours-BERT** 8B@43000 ★ | `[L]` | β+α | 0.593 | 0.642 | 0.671 | 0.703 | 0.713 |
| **Ours-Diffusion** 8B@45000 ★ | `[L]` | β+α | 0.587 | 0.647 | 0.675 | 0.705 | **0.720** |
| **Ours-Diffusion** v3@27000 | `[L]` 七源 | β+α | 0.584 | 0.636 | 0.664 | 0.692 | 0.705 |
| **Ours-Diffusion** v3@42000 | `[L]` 七源 | β+α | 0.586 | 0.636 | 0.663 | 0.694 | 0.709 |
| **Ours-Diffusion** allch@33000 | `[L]` 全链 | β+α | 0.590 | 0.643 | 0.666 | 0.691 | 0.703 |
| **Ours-Diffusion** allch@18k | `[L]` 全链续训快照 | β+α | 0.587 | 0.640 | 0.667 | 0.696 | 0.710 |
| **Ours-Diffusion** allch@26k | `[L]` 全链续训最新 | β+α | 0.586 | 0.646 | 0.668 | 0.694 | 0.707 |

**deep k=200 的 top-k 波动带**（同 run 换 step）

| Run | steps 上的 k=200 范围 | 宽度 |
|---|---|---|
| 270m BERT | 0.7117 – 0.7128 | 0.0011 |
| 270m diffusion | 0.7089 – 0.7094 | 0.0005 |
| 8B BERT | 0.7093 – 0.7133 | 0.0040 |
| 8B diffusion | 0.7198 – 0.7204 | 0.0006 |

**broad track — 24 表位 macro AUROC**

| Method | 类型 | k=1 | k=5 | k=20 | k=100 |
|---|---|---:|---:|---:|---:|
| **SCEPTR** | 专用对比学习 | **0.587** | **0.655** | **0.711** | **0.741** |
| TCRdist (tcrdist3) | 序列比对 | 0.577 | 0.644 | 0.687 | 0.728 |
| CDR3-Levenshtein-NN | 序列比对(训练free) | 0.555 | 0.597 | 0.641 | 0.683 |
| k-mer(3) 组成 | 训练free | 0.543 | 0.597 | 0.643 | 0.673 |
| Ophiuchus-Ab | 抗体权重 | 0.576 | 0.612 | 0.664 | 0.677 |
| ESM2-150M | 通用蛋白 LM | 0.561 | 0.589 | 0.638 | 0.671 |
| ProtBERT | 通用蛋白 LM | 0.550 | 0.587 | 0.627 | 0.651 |
| TCR-BERT | TCR 专用 LM | 0.547 | 0.577 | 0.623 | 0.641 |
| **Ours-BERT** 270m@49000 ★ | `[L]` | 0.576 | 0.611 | 0.659 | 0.663 |
| **Ours-Diffusion** 270m@42000 ★ | `[L]` | 0.567 | 0.608 | 0.643 | 0.652 |
| **Ours-BERT** 8B@43000 ★ | `[L]` | 0.562 | 0.607 | 0.644 | 0.669 |
| **Ours-Diffusion** 8B@45000 ★ | `[L]` | 0.578 | **0.618** | 0.656 | **0.680** |
| **Ours-Diffusion** v3@27000 | `[L]` 七源 | 0.560 | 0.602 | 0.647 | 0.656 |
| **Ours-Diffusion** v3@42000 | `[L]` 七源 | 0.554 | 0.601 | 0.644 | 0.648 |
| **Ours-Diffusion** allch@33000 | `[L]` 全链 | 0.560 | 0.608 | 0.643 | 0.648 |
| **Ours-Diffusion** allch@18k | `[L]` 全链续训快照 | 0.555 | 0.595 | 0.636 | 0.649 |
| **Ours-Diffusion** allch@26k | `[L]` 全链续训最新 | 0.561 | 0.600 | 0.638 | 0.642 |

broad k=100 波动带：270m BERT 0.6622–0.6634、270m diff 0.6501–0.6521、8B BERT 0.6604–0.6694、8B diff 0.6752–0.6801。

**24-way probe（辅，非主协议）**

| Model | probe-AUROC | probe-Acc | kNN-top1 |
|---|---:|---:|---:|
| Ours 旧 grammar_v2 300m（历史参照） | 0.793 | — | — |
| **Ours-BERT** 270m@49000 | 0.804 | 0.415 | 0.351 |
| **Ours-Diffusion** 270m@42000 | 0.794 | 0.402 | 0.366 |
| **Ours-BERT** 8B@43000 | 0.816 | 0.458 | 0.379 |
| **Ours-Diffusion** 8B@45000 | **0.828** | **0.459** | 0.378 |
| **Ours-Diffusion** v3@27000 | 0.794 | 0.411 | 0.352 |
| **Ours-Diffusion** v3@42000 | 0.786 | 0.401 | 0.379 |
| **Ours-Diffusion** allch@33000 | 0.798 | 0.426 | 0.382 |
| **Ours-Diffusion** allch@18k | 0.798 | 0.408 | 0.373 |
| **Ours-Diffusion** allch@26k | 0.782 | 0.417 | 0.373 |
| SCEPTR（kNN-top1 对照） | — | — | **0.506** |

> 结论（**2026-08-30 按输入档位重述**）：deep k=200 四条落在 **0.709–0.720**。在**同为 CDR3β+α 两字段**的这一档里，我们高于三个 PLM（ESM2 0.696 / ProtBERT 0.694 / TCR-BERT 0.692，后者只有 β 一个字段），但**仍低于两个零训练的序列比对基线**（CDR3-Levenshtein 0.733、k-mer 组成 0.728）——这是同输入下的直接落后，无法用信息量差异解释。
> 而**唯一两个明显超过我们的方法都多拿了胚系基因**：SCEPTR 0.790（多 TRBV/TRAV）、TCRdist 0.777（多 TRBV/TRAV/TRBJ/TRAJ）。所以原先「低于 TCRdist/SCEPTR 约 0.06–0.07」这句**不能直接读成模型能力差距**，它混入了 V/J 信息的贡献；同样也不能用它开脱，因为同档位的比较仍然对我们不利。broad k=100 为 0.652–0.680，与 Levenshtein（0.683）持平、低于 TCRdist（0.728），结构相同。
> **波动带 ≤0.004，模型间差异（8B diffusion 比 270m diffusion 高 0.011 deep / 0.028 broad）是真实的。** 但 270m 上 BERT 反超 diffusion（deep 0.712 vs 0.709、broad 0.663 vs 0.652），8B 上 diffusion 反超 BERT——这个交叉超出波动带，是真实现象而非噪声。
> **规模收益近零**：8B vs 270m 在 deep 上只差 ≤0.011，diffusion 端 T2 ARI 甚至更低。参数量涨 30 倍，表征几乎不动，说明瓶颈在「冻骨干 + 全局 mean-pool」的读出方式或预训练目标，而非容量。

### §0.4 TCR T4 Generation — Setting B 表位条件 CDR3β（K=100，**diffusion-only**）

> BERT 按口径不跑生成。**2026-08-30 已离线重评分**（`scripts/rescore_tcr_generation_offline.py`，纯从已存 `per_pmhc` 重算，无任何模型推理），修了三处口径问题，见表下说明。
>
> **Char-BLEU 一栏不可跨组比较，已灰标。** `common/metrics.py:937` 的 `bleu_hyp = greedy if greedy else gen[0]`：char_bleu 只取**单条**假设序列。TCRT5 / GRATCR / ER 官方行有论文预存的 greedy 解码；Ours / TcrDesign / TCRDiff / OLGA 的 `designs.jsonl` **零条带 greedy**，一律退化成 `gen[0]`，即 K=100 个采样里按种子发射顺序的任意一条。拿后者比前者等于「随机一条采样」比「greedy 解码」。**同组内可比，跨组不可比。**

**★ 跨模型主对照 — 共同 unseen 集合（6 个 pMHC，所有模型完全同集合）**

> 这是本节唯一**跨模型有效**的表。`bioseq_unseen` 一栏此前**不可比**：其成员取决于各 run 恰好跑了什么（只跑 held20 的覆盖 2 个、跑 benchmark14 的 6 个、跑全 34 的 8 个），把两行并排等于比不同的 pMHC。现固定为 `bioseq_unseen_common` = benchmark14 ∩ BioSeq-unseen − 保留 pMHC = `FTDALGIDEY_A*01:01` / `KINMPMSVK_A*03:01` / `NENLDLQEL_B*40:01` / `SALPTNADLY_A*01:01` / `TPSVSSSISSL_B*07:02` / `TSDACMMTMY_A*01:01`。该视图**只在 run 完整覆盖全部 6 个时才输出**，其存在本身即可比性保证。

> **🔻 已列 baseline 的最佳已发表值（论文值优先；与下表 Setting-B K=100 不可合并排序）**。TCRT5 论文主榜是 sparse-13（见本节稍后 `[A]` 表），不是 held20，也不是本表的 6-pMHC 公共 unseen。下表里 TCRT5/GRATCR/ER 的 `[A]` 是官方预存预测在**同一 6-pMHC 集合**上的重评分，用来做同集合比较；它们的**代表论文值**仍是 sparse-13。TcrDesign / TCRDiff 无 sparse-13 论文行，代表值取其已发表/官方管线在可复现协议上的最好数字（held20 是验证集，不得当测试主榜）。

| Method | 最佳已发表 | 协议 | 出处 |
|---|---|---|---|
| TCRT5 | F1@100 0.0003 · d_edit 4.467 · seq-rec 0.602 | 论文 sparse-13（13 pMHC） | `[P]`/`[A]` 见下方 sparse-13 表 |
| GRATCR | F1@100 0.0000 · d_edit 4.397 · seq-rec 0.595 | 同上 | `[P]`/`[A]` |
| ER-Transformer | F1@100 0.0000 · d_edit 6.068 · seq-rec 0.337 | 同上 | `[P]`/`[A]` |
| TcrDesign | 论文全长/β 管线；本地 `[R]` held20 F1 0.1527（**验证集**） | 与 sparse-13 / Setting-B K=100 均不同 | `[P]` 未与 sparse-13 合表；`[R]` 见 held20 表 |
| TCRDiff | bioRxiv 2026 条件扩散；本地 `[R]` common-6 d_edit 5.157 | 非 TCRT5 主协议 | `[P]` 未转录进 sparse-13；`[R]` 见下表 |

| Method | d_edit↓ | seq-rec↑ | F1@100 | Char-BLEU | n | 类型 |
|---|---:|---:|---:|---:|:--:|---|
| **GRATCR**（官方预存） | **4.605** | 0.584 | 0.0000 | 0.673 ᵍ | 6/6 | `[A]` |
| **TCRT5**（官方预存） | 4.696 | **0.585** | 0.0003 | 0.629 ᵍ | 6/6 | `[A]` |
| **TcrDesign-G** | 4.857 | 0.573 | 0.0000 | 0.537 | 6/6 | `[R]` |
| **TCRDiff**（2026 扩散） | 5.157 | 0.543 | 0.0000 | 0.416 | 6/6 | `[R]` |
| **ER-Transformer**（官方预存） | 6.075 | 0.321 | 0.0000 | 0.605 ᵍ | 6/6 | `[A]` |
| **Ours-Diffusion** 8B@45000 | 6.297 | 0.483 | 0.0000 | 0.376 | 6/6 | `[L]` |
| ⚠️ **OLGA**（无条件随机对照） | 6.343 | 0.470 | 0.0000 | 0.494 | 6/6 | `[C]` |
| **Ours-Diffusion** 270m@42000 | 6.985 | 0.420 | 0.0000 | 0.323 | 6/6 | `[L]` |
| **Ours-Diffusion** v3@27000 | 8.403 | 0.242 | 0.0000 | 0.104 | 6/6 | `[L]` 七源 |
| **Ours-Diffusion** v3@42000 | 8.580 | 0.215 | 0.0000 | 0.217 | 6/6 | `[L]` 七源 |
| **Ours-Diffusion** allch@33000 | 8.617 | 0.211 | 0.0000 | 0.242 | 6/6 | `[L]` 全链 |
| **Ours-Diffusion** allch@18k | 8.728 | 0.213 | 0.0000 | 0.154 | 6/6 | `[L]` 全链续训快照 |
| **Ours-Diffusion** allch@26k | 8.577 | 0.212 | 0.0000 | 0.185 | 6/6 | `[L]` 全链续训最新 |
| **Ours-Diffusion** allch@121k | 8.778 | 0.210 | 0.0000 | 0.188 | 6/6 | `[L]` 全链 |
| **Ours-Diffusion** allch@151k | 8.800 | 0.218 | 0.0000 | 0.221 | 6/6 | `[L]` 全链最好 val |
| **Ours-Diffusion** genonly-2M@44k | 8.610 | 0.221 | 0.0000 | 0.231 | 6/6 | `[L]` generated-only 最好 val |

ᵍ = 用论文预存 greedy 解码；其余用 `gen[0]`，两者不可比。

> **allch@121k / 151k 与 genonly-2M@44k 三行为 2026-09-07 补入**（151k / 44k 本地单卡跑，
> 121k 为此前平台跑；过程见 [`$ROOT/PROJECT_PROCESS.md`](../../PROJECT_PROCESS.md) 2026-09-07 节）。
> 产物 `downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_fusion_v3_*_t4_unseen/metrics.json`
> 的 `summary.bioseq_unseen_common`（三条均 n=6/6，故可进本表）。
>
> **长跑到 151k 没有改善 T4，两臂也打平。** 全链 26k→121k→151k 的 d_edit 是 8.577 → 8.778 → 8.800，
> **不降反微升**；151k 对 generated-only 44k 是 8.800 vs 8.610，**落后 0.19**。三行都比 §0.0 早期的
> 270m@42000（6.985）和 8B@45000（6.297）差得多，也远不如不看表位的 OLGA（6.343）。
> 结合上文那条主结论：**七源 v3 系列在 unseen 表位上的条件信号本就接近不存在，继续长跑并未改变这一点。**

> **这是 T4 最重要的一条结论，且不利于我们**：在唯一同集合可比的口径下，我们最好的模型
> （8B@45000，d_edit 6.297）**与 OLGA 无统计差别**。OLGA 是完全不看表位的 VDJ 重组随机采样器，
> 不具备任何条件生成能力。我们落后每一个已发表的条件生成器（GRATCR −1.69 / TCRT5 −1.60 /
> TcrDesign −1.44 / TCRDiff −1.14），也落后 ER-Transformer（−0.22）。
> **表位条件信号在 unseen 表位上接近不存在。**
>
> ⚠️ **「只比 OLGA 好 0.046」这一表述已于 2026-08-31 收紧为「无统计差别」。** 该视图只有 6 个 pMHC，
> 逐 pMHC 配对 t 检验（df=5，|t|>2.57 才在 0.05 显著）给出：
>
> | 对比 | Δ d_edit | SE | t | 判定 |
> |---|---:|---:|---:|---|
> | Ours 8B − **OLGA** | **−0.047** | 0.085 | **−0.55** | **不显著** |
> | Ours 8B − TCRT5 | +1.600 | 0.139 | +11.55 | 显著 |
> | Ours 8B − GRATCR | +1.692 | 0.379 | +4.46 | 显著 |
>
> 逐 pMHC 的配对差是 −0.12 / **+0.33** / −0.27 / **+0.02** / −0.16 / −0.08——六个里有两个我们更差。
> 所以本表对真实强弱差距有分辨率（vs TCRT5 t=11.55），但**分辨不出我们与一个不看表位的随机
> 重组采样器**，因此不得把那 0.046 读成「仍有微弱优势」。
> 这也是根因诊断必须另做实验的原因：6 个 pMHC 的观测量说不了根因，见 §0.0(h) 的 infilling 诊断
> （56,034 个被掩位点、11 组条件、逐位点 McNemar）。

**held20（20 pMHC；⚠️ 是 TCRT5 的 *validation* split，非测试集）**

> manifest 记载 `held20.source = topk_val_df.csv` —— 这是 TCRT5 的 top-k **验证**集，论文主榜是 sparse-13 而非 held20。本表仅作 target-rich 补充读数，**不能当作论文口径的对外结论**。另 held20 对 BioSeq 有 ~48% 训练 pair 泄露。

| Method | F1@100 | d_edit↓ | seq-rec | Char-BLEU | div | n | 类型 |
|---|---:|---:|---:|---:|---:|:--:|---|
| **TcrDesign-G** (beta) | **0.1527** | **1.474** | **0.870** | 0.986 | 0.995 | 20/20 | `[R]` |
| **TCRT5** (HF beam) | 0.0855 | 1.821 | 0.834 | 0.947 ᵍ | 1.000 | 20/20 | `[R]` |
| **TCRDiff**（2026 扩散，pMHC-only） | 0.0118 | 2.954 | 0.732 | 0.582 | 0.996 | 20/20 | `[R]` |
| **Ours-Diffusion** 8B@45000 | 0.0000 | 5.178 | 0.506 | 0.575 | 1.000 | 20/20 | `[L]` |
| **Ours-Diffusion** 270m@42000 | 0.0000 | 5.601 | 0.474 | 0.547 | 1.000 | 20/20 | `[L]` |
| **Ours-Diffusion** v3@27000 | 0.0000 | 6.426 | 0.335 | 0.524 | 1.000 | 20/20 | `[L]` 七源 |
| **Ours-Diffusion** v3@42000 | 0.0000 | 6.751 | 0.282 | 0.517 | 1.000 | 20/20 | `[L]` 七源 |
| **Ours-Diffusion** allch@33000 | 0.0000 | 6.732 | 0.267 | 0.540 | 1.000 | 20/20 | `[L]` 全链 |
| **Ours-Diffusion** allch@18k | 0.0000 | 6.762 | 0.274 | 0.550 | 1.000 | 20/20 | `[L]` 全链续训快照 |
| **Ours-Diffusion** allch@26k | 0.0000 | 6.693 | 0.273 | 0.567 | 1.000 | 20/20 | `[L]` 全链续训最新 |
| **Ours-Diffusion** allch@121k | 0.0000 | 6.744 | 0.278 | 0.531 | 1.000 | 20/20 | `[L]` 全链 |
| **Ours-Diffusion** allch@151k | 0.0000 | 6.751 | 0.288 | 0.542 | 1.000 | 20/20 | `[L]` 全链最好 val |
| **Ours-Diffusion** genonly-2M@44k | 0.0000 | **6.722** | **0.294** | 0.517 | 1.000 | 20/20 | `[L]` generated-only 最好 val |
| Ours 旧 grammar_v2 BioSeq（历史） | 0.0000 | 6.36 | 0.454 | 0.411 | — | — | VOID |

> **TCRDiff 行已更正（2026-08-30）。** 原表填 `F1=0.0009 / seq-rec 0.587 / d_edit —`，那是 TCRDiff **benchmark14** run 的数字被放进了 held20 表。同 split 的 `tcrdiff_held20` run 一直在盘上且 20/20 完整：**F1 0.0118（低报 12.8×）、d_edit 2.954、seq-rec 0.732**。更正后 TCRDiff 从「看起来和我们同档」变成**明显强于我们**（d_edit 领先 2.22、seq-rec 领先 0.23）。

**benchmark14 / unseen 视图（Ours，`*_t4_unseen` run）**

| Model | 视图 | F1@100 | d_edit↓ | seq-rec | Char-BLEU | n（打分/成员） |
|---|---|---:|---:|---:|---:|:--:|
| **Ours-Diffusion** 8B@45000 | benchmark14 | 0.0000 | 6.277 | 0.474 | 0.399 | **7/14** |
| **Ours-Diffusion** 270m@42000 | benchmark14 | 0.0000 | 6.914 | 0.419 | 0.358 | **7/14** |
| **Ours-Diffusion** 8B@45000 | unseen | 0.0000 | 5.838 | 0.517 | 0.398 | 8/8 |
| **Ours-Diffusion** 270m@42000 | unseen | 0.0000 | 6.408 | 0.463 | 0.407 | 8/8 |
| **Ours-Diffusion** 8B@45000 | overall | 0.0000 | 5.873 | 0.506 | 0.413 | **9/34** |
| **Ours-Diffusion** 270m@42000 | overall | 0.0000 | 6.417 | 0.457 | 0.425 | **9/34** |

> **n 列已改为「实际参与平均 / 类别成员」。** 旧表在这些行印 `n=14` 和 `n=9`，但 `_macro` 会丢掉 NaN 再平均，而 `n_pmhc` 数的是类别成员数。这两个 run 的 `designs.jsonl` **只有 9 条**（只对 unseen pMHC 生成），所以 benchmark14 实际只有 7 个进了平均、`overall` 只有 9 个进了平均却印着 34。已在 `scoring.py::_macro` 补 `n_pmhc_scored`，并让 `run.py` / `score_official.py` / 汇总脚本一律打印 `scored/total`。**这不是打分 bug，是 run 本身只覆盖 unseen，旧代码把覆盖范围报大了。**
>
> **unseen 由 9 个改为 8 个。** `RVRAYTYSK/HLA-A*03:01` 被 TCRT5 论文保留给独立 simulation 并排除在主榜之外，却仍留在这一栏里。现已从 `bioseq_seen`/`bioseq_unseen` 两个跨切视图剔除（`benchmark14` 控制视图保留它，那一栏本就标注为 14-pMHC control）。`bioseq_unseen_pmhc.json` 未改动——它记录的是「该表位确实不在 BioSeq 训练语料中」这一泄漏事实，改它会让文件名变成谎言，排除动作放在视图构造处。
> **剔除它对 baseline 的伤害远大于对我们**：该靶点有 895 条参考序列（14 个 target 中最多），是最易命中的目标。剔除后 TCRT5 unseen d_edit 4.343→4.696（+0.35）、TCRDiff 4.882→5.157（+0.27）、TcrDesign 4.010→4.203（+0.19），而我们只 5.873→5.838（−0.036）。**差距是朝对我们有利的方向收窄的，此处明确声明我们不主张这部分改善。**

**论文 sparse-13 主榜 `[A]`（官方预存预测，13 pMHC，尚未与上表合排）**

| Method | F1@100 | d_edit↓ | seq-rec | Char-BLEU | div | n |
|---|---:|---:|---:|---:|---:|:--:|
| TCRT5 | 0.0003 | 4.467 | 0.602 | 0.701 ᵍ | 1.000 | 13/13 |
| GRATCR | 0.0000 | 4.397 | 0.595 | 0.677 ᵍ | 0.081 | 13/13 |
| ER-Transformer | 0.0000 | 6.068 | 0.337 | 0.594 ᵍ | 0.994 | 13/13 |

> **修 encoder 泄露后 T4 反而全面变好**（held20：270m d_edit 7.07→5.60、seq-rec 0.384→0.474、BLEU 0.186→0.547）。原因是旧泄露把某条特定参考 binder 喂进 ESMC 条件流，逼模型去抄它，反而偏离表位共性 motif。
> 但 **F1 在所有视图上仍全为 0，一条真 binder 都没找回**。**表位条件设计是真实失败，不是打分 bug**，且在上面的同集合对照里已量化到「仅优于无条件随机 0.046」。8B 稳定优于 270m（seq-rec +0.03～0.05）。
> ⚠️ held20 的 `bioseq_seen`/`bioseq_unseen` 分区沿用旧 grammar_v2 语料的泄露标注，**对本轮 immune 语料无效**；新泄露报告未做。

### §0.5 AB CDR infilling（AAR % ↑，**diffusion-only**）

> **口径已对齐**：argmax + encoder 泄露已修。SAbDab 用 **Kong 版 3,127 条**；SAb23H2 用官方 `IgGM_Test_set`。两表均取 `max_iter ∈ {1,2,4,8}` 各自最优。

#### SAbDab 10-fold — ✅ 数据版本问题已定位并解决

**根因**：我们原先用的 `data/.../sabdab/` 是 **2024 年后的 SAbDab 快照**（3,320 行 / 3,131 unique PDB，其中 **1,079 行是 8xxx/9xxx 的 2023+ 结构**）。论文引的 Kong et al. 快照冻结在 2022-11-12，里面 8xxx 开头只有 1 个。两份数据交集仅 1,681 个 PDB，且 **fold 分配一致率只有 9–12%（≈随机）**。新结构的 CDR-H3 平均长 0.67 个残基（15.23 vs 14.56），这只压 H3、不动定长的 H1/H2——正是我们观测到的偏差特征。

**已重建 Kong 版数据**：`data/downstream/cdr_infilling/sabdab_kong/`，**3,127 条**（与论文一致），0 个 2023+ 结构。序列取自本地 `data/sabdab.zip`（已是 MEAN 管线输出），10 折用官方 `THUNLP-MT/MEAN:data/split.py` 原样生成（`mmseqs --min-seq-id 0.4` + `seed 2022` + cluster-wise 分折，聚类数 cdrh1 766 / cdrh2 1104 / cdrh3 1648）。`--force` 重建 30 个 `test.json` md5 全一致，确定性已验证。一键复现：`bash scripts/downstream/build_sabdab_kong_split.sh`。
> Kong et al. **没有** release 处理好的数据（dyMEAN 唯一 release 只含 checkpoint，MEAN release 数为 0），只发了冻结 summary + 切分脚本，所以 fold 划分无法与作者逐字节对证；3,127 这个总数与论文吻合是现有最强证据。

**Kong 版数据结果（n=3,127，各模型逐 CDR 取自身 max_iter 最优 — 见下方口径警告）**

| Model | CDR-H1 | CDR-H2 | CDR-H3 | 类型 |
|---|---:|---:|---:|---|
| AntiBERTy | 76.70 | 71.10 | 42.70 | `[P]` 语料可能重叠 |
| AbLang2 | 76.30 | 70.60 | 42.70 | `[P]` 语料可能重叠 |
| Ophiuchus-Ab | 75.50 | 70.18 | 43.55 | `[P]` 无重叠, zero-shot |
| RADD | 67.64 | 58.87 | 37.71 | `[P]` fine-tuned |
| ADesigner | 64.34 | 55.52 | 37.37 | `[P]` fine-tuned |
| dyMEAN | 63.52 | 55.41 | 37.19 | `[P]` fine-tuned |
| MEAN | 58.29 | 47.15 | 36.38 | `[P]` fine-tuned |
| AbBFN | 70.30 | 64.90 | 31.50 | `[P]` |
| Ophiuchus-Ab（官方 ckpt 复跑） | 75.83 | 70.24 | **43.70** | `[R]` |
| **Ours-Diffusion** 270m@42000 | 76.60 | 70.35 | 43.39 | `[L]` |
| **Ours-Diffusion** 8B@45000 | **76.86** | **71.22** | 43.46 | `[L]` |

> ⚠️ **上表的官方行用了三个不同的 `max_iter`**：H1 取 it=8、H2 取 it=2、H3 取 it=1（逐 CDR 在测试集上取最优）。我们的行同样如此（8B 的 H1 取 it=2/4/8、H3 取 it=2）。此前本节写过「SAbDab 用固定 iter=1」，**那是错的**——只是 H3 恰好在 it=1 达到最优。固定口径的版本见下表。

**固定 `iter=1`（无测试集选参，AAR%，`average_aar_all_folds`）**

| Model | CDR-H1 | CDR-H2 | CDR-H3 | 类型 |
|---|---:|---:|---:|---|
| Ophiuchus-Ab（论文） | 75.50 | 70.18 | 43.55 | `[P]` |
| Ophiuchus-Ab（官方 ckpt，iter=1） | 75.69 | 69.96 | **43.70** | `[R]` |
| **Ours-Diffusion** 270m@42000 | 76.51 | 70.35 | 43.39 | `[L]` |
| **Ours-Diffusion** 8B@45000 | **76.84** | **71.22** | 43.39 | `[L]` |

> **固定 iter=1 的复现比取最优更紧**：官方 ckpt 对论文 +0.19 / −0.22 / +0.15，**三项全在 ±0.22 pp 内**（取最优时最大偏差 0.33 pp）。所以「已复现」这个结论在无选参口径下依然成立，而且更强。原先那 −2.66 pp 缺口 = 数据版本（+1.68）+ 解码步数（+0.80）。
> **8B 对官方（固定 iter=1）**：H1 **+1.15**、H2 **+1.26**、H3 −0.31。取最优时是 +1.03 / +0.98 / −0.24。

**max_iter 敏感度（Kong 版，AAR%，argmax 解码）**

| Model | CDR | it=1 | it=2 | it=4 | it=8 | it=16 | 带宽 |
|---|---|---:|---:|---:|---:|---:|---:|
| Ophiuchus `[R]` | H1 | 75.69 | 75.66 | 75.81 | **75.83** | 75.77 | 0.17 |
| Ophiuchus `[R]` | H2 | 69.96 | **70.25** | 70.09 | 70.09 | 70.10 | 0.28 |
| Ophiuchus `[R]` | H3 | **43.70** | 43.11 | 42.31 | 42.22 | 42.16 | **1.56** |
| Ours 270m | H1 | 76.51 | 76.59 | **76.60** | 76.60 | 76.55 | 0.09 |
| Ours 270m | H2 | **70.35** | 70.28 | 70.29 | 70.28 | 70.30 | 0.07 |
| Ours 270m | H3 | **43.39** | 43.33 | 43.34 | 43.34 | 43.36 | 0.06 |
| Ours 8B | H1 | 76.84 | **76.86** | 76.86 | 76.86 | 76.85 | 0.02 |
| Ours 8B | H2 | **71.22** | 71.19 | 71.21 | 71.21 | 71.20 | 0.04 |
| Ours 8B | H3 | 43.39 | **43.46** | 43.44 | 43.41 | 43.40 | 0.07 |

> AAR 是逐位准确率，迭代解码每次提交都会固化预测并作为后续条件，**错误会传播**；`iter=1` 是纯一次性边缘 argmax，逐位期望准确率最高。这个效应只在最难的 H3 上显著（官方跨 1.56 pp），H1/H2 两边都平（≤0.28）。
> **官方 ckpt 在 H3 上跨 1.56 pp，我们的模型只有 0.06–0.07。** 我们的逐步预测自洽（多轮 refine 既不改善也不破坏），但也可能说明迭代解码没真正利用已提交的上下文——这是个待解释的差异，不是优点。
>
> **哪些比较对 `max_iter` 稳健，哪些不稳健**：我们在 H1/H2 上的 +1.15 / +1.26 优势**大于官方自身的带宽（0.17 / 0.28）**，所以这两项的领先在该参数任何合理取值下都成立，是可声称的。H3 上我们落后 0.31，而官方带宽 1.56 是它的 **5 倍**——**H3 的排序不稳健，不能声称任何一方领先**。
>
> 口径约定：**本节及 SAb23H2 节统一以固定 `iter=1` 为报告口径**，逐 CDR 取最优的表仅作波动带参考。聚合方式统一用 `average_aar_all_folds`（逐折宏平均）；注意它与 `average_aar`（逐序列微平均）在 H1 上可差 1.02 pp（75.69 vs 76.71），论文的 75.50 明显更贴宏平均，这也支持该选择。

**旧快照结果（3,320 行，仅作历史对照，不与上表混排）**

| Model | H1 | H2 | H3 |
|---|---:|---:|---:|
| Ophiuchus-Ab `[R]` iter=2 | 75.44 | 70.71 | 41.43 |
| Ours-Diffusion 270m@42000 iter=2 | 76.24 | 70.92 | 41.91 |
| Ours-Diffusion 8B@45000 iter=2 | 76.38 | 71.81 | 42.06 |
| Ours-Diffusion v3@27000 iter=2 | 74.44 | 68.60 | 40.75 |
| Ours-Diffusion v3@42000 iter=2 | 75.41 | 69.82 | 41.62 |
| Ours-Diffusion allch@33000 iter=2 | 75.23 | 69.68 | 41.63 |
| Ours-Diffusion allch@18k iter=2 | 74.93 | 69.58 | 41.44 |
| Ours-Diffusion allch@26k iter=2 | 75.32 | 69.99 | 41.68 |
| **Ours-Diffusion allch@151k iter=2** | **75.88** | **71.33** | **41.92** |
| **Ours-Diffusion genonly-2M@44k iter=2** | 75.60 | 70.88 | 41.79 |

> **151k / 44k 两行是 2026-09-07 本地单卡跑出的**（`queue012` 排不上，四个 T4/CDR 阶段改本机
> A100-80G 串行跑，env 与 eval YAML 逐行一致；过程见
> [`$ROOT/PROJECT_PROCESS.md`](../../PROJECT_PROCESS.md) 2026-09-07 节）。
> 产物：`output/downstream_generation/ours_fusion_v3_{allchains_8gpu2m_151000,genonly_8gpu2m_44000}_cdrh{1,2,3}_10fold_iter2.log`。
>
> **两臂在 CDR 上打平，全链训练无可见收益。** 151k − 44k = H1 +0.28 / H2 +0.45 / H3 +0.13，
> 三项都小于同臂不同 ckpt 间的波动（如 allch 18k→26k 的 H2 +0.41）。151k 相对本臂
> 早期点确有小幅上行（vs 26k：H1 +0.56 / H2 +1.34 / H3 +0.24），但**这是 2M 计划的 7.6%
> 进度，不足以当长跑结论**。SAb23H2 侧同样打平，见下节。

top-k 波动带（旧快照，8 个 diffusion ckpt，iter=2）：270m（39k/42k/47k/50k）H1 76.08–76.24 / H2 70.92–71.08 / H3 41.78–41.96；8B（44k/45k/48k/50k）H1 76.33–76.39 / H2 71.81–71.87 / H3 42.04–42.13。波动 ≤0.2 pp。

#### SAb23H2（n=60，六 CDR）

> SAb23H2 **不存在数据版本问题**——它随官方仓库以 `IgGM_Test_set` 发布，不是自建快照。
> ✅ **Ours sweep 已收口**：`eval-immune-270m-diff-sab23h2`（`t-20260825223259-52wlr`，Success 9.6 min）、`eval-immune-8b-diff-sab23h2`（`t-20260825223303-nwckl`，Success 23.3 min）。产物 `output/downstream_generation/sab23h2_fixed/`。
>
> ⚠️ **下表是「各模型逐 CDR 取自身 `max_iter ∈ {1,2,4,8}` 最优」，这是在测试集上逐指标选超参，不能当作复现口径。** 官方行六个格子把四档设置全用上了：L2 取 it=1、H1/H2/H3 取 it=2、L1 取 it=4、L3 取 it=8。三行都这么处理所以至少是对称的，但它同时抬高了所有人。固定口径的版本见下面「固定 `iter=1`」表。

| Model | L1 | L2 | L3 | H1 | H2 | H3 | 类型 |
|---|---:|---:|---:|---:|---:|---:|---|
| Ophiuchus-Ab | **81.1** | **80.1** | **73.7** | 74.8 | **68.6** | **36.8** | `[P]` |
| IgGM (IgFold) | 75.0 | 74.3 | 63.5 | 74.0 | 64.4 | 36.0 | `[P]` |
| IgGM (AF3) | 73.7 | 73.5 | 60.2 | 73.9 | 63.9 | 33.0 | `[P]` |
| dyMEAN | 63.3 | 63.4 | 57.0 | 74.2 | 62.7 | 29.4 | `[P]` |
| DiffAb (AF3) | 60.8 | 59.9 | 42.4 | 63.7 | 39.4 | 22.6 | `[P]` |
| Ophiuchus-Ab（官方 ckpt 复跑，best of 1/2/4/8） | **81.19** | **80.68** | **73.28** | 74.52 | **68.59** | **36.70** | `[R]` |
| **Ours-Diffusion** 270m@42000 | 79.92 | 79.41 | 72.37 | 74.76 | 67.85 | 35.03 | `[L]` |
| **Ours-Diffusion** 8B@45000 | 80.31 | 79.64 | 72.06 | **75.00** | 68.23 | 35.60 | `[L]` |

**max_iter sweep（argmax，AAR%）**

| Model | CDR | it=1 | it=2 | it=4 | it=8 | best |
|---|---|---:|---:|---:|---:|---|
| Ophiuchus `[R]` | L1 | 80.88 | 80.92 | **81.19** | 80.80 | it=4 |
| Ophiuchus `[R]` | L2 | **80.68** | 80.21 | 79.64 | 79.88 | it=1 |
| Ophiuchus `[R]` | L3 | 72.36 | 72.81 | 72.48 | **73.28** | it=8 |
| Ophiuchus `[R]` | H1 | 73.81 | **74.52** | 74.05 | 74.05 | it=2 |
| Ophiuchus `[R]` | H2 | 68.32 | **68.59** | 67.32 | 67.60 | it=2 |
| Ophiuchus `[R]` | H3 | 35.40 | **36.70** | 34.65 | 34.83 | it=2 |
| Ours 270m | L1 | 79.92 | 79.92 | 79.92 | 79.92 | flat |
| Ours 270m | L2 | 79.41 | 79.41 | 79.41 | 79.41 | flat |
| Ours 270m | L3 | 72.37 | 72.37 | 72.37 | 72.37 | flat |
| Ours 270m | H1 | 74.52 | **74.76** | 74.76 | 74.76 | it=2 |
| Ours 270m | H2 | 67.85 | 67.85 | 67.85 | 67.85 | flat |
| Ours 270m | H3 | 35.03 | 35.03 | 35.03 | 35.03 | flat |
| Ours 8B | L1 | 80.19 | **80.31** | 80.31 | 80.31 | it=2 |
| Ours 8B | L2 | 79.41 | **79.64** | 79.64 | 79.64 | it=2 |
| Ours 8B | L3 | 72.00 | **72.06** | 72.06 | 72.06 | it=2 |
| Ours 8B | H1 | 75.00 | 75.00 | 75.00 | 75.00 | flat |
| Ours 8B | H2 | 67.95 | 67.95 | **68.23** | 68.23 | it=4 |
| Ours 8B | H3 | 34.82 | 35.24 | **35.60** | 35.60 | it=4 |

**固定 `iter=1`（与 SAbDab 表同口径，无测试集选参，AAR%）**

| Model | L1 | L2 | L3 | H1 | H2 | H3 | 类型 |
|---|---:|---:|---:|---:|---:|---:|---|
| Ophiuchus-Ab（论文） | 81.1 | 80.1 | 73.7 | 74.8 | 68.6 | 36.8 | `[P]` |
| Ophiuchus-Ab（官方 ckpt，iter=1） | 80.88 | 80.68 | 72.36 | 73.81 | 68.32 | 35.40 | `[R]` |
| **Ours-Diffusion** 270m@42000 | 79.92 | 79.41 | 72.37 | 74.52 | 67.85 | 35.03 | `[L]` |
| **Ours-Diffusion** 8B@45000 | 80.19 | 79.41 | 72.00 | **75.00** | 67.95 | 34.82 | `[L]` |

**v3 固定 `iter=2`（脚本默认；SAb23H2 与旧 270m iter=2 可比，勿与上表 iter=1 / best-of 混排）**

| Model | L1 | L2 | L3 | H1 | H2 | H3 |
|---|---:|---:|---:|---:|---:|---:|
| Ours 270m@42000 iter=2 | 79.92 | 79.41 | 72.37 | 74.76 | 67.85 | 35.03 |
| **Ours v3@27000** | 79.36 | 78.93 | 70.58 | 74.76 | 66.34 | 34.11 |
| **Ours v3@42000** | 80.19 | 79.41 | 71.47 | 74.76 | 65.54 | 34.79 |
| **Ours allch@33000** | 80.30 | 79.64 | 71.34 | 74.05 | 66.65 | 35.45 |
| **Ours allch@18k** | 80.12 | 80.36 | 70.93 | 75.24 | 66.63 | 34.75 |
| **Ours allch@26k** | 79.17 | 79.64 | 71.51 | 74.76 | 66.33 | 34.79 |
| **Ours allch@151k** | 80.32 | 79.64 | **72.74** | 74.52 | **67.92** | 34.56 |
| **Ours genonly-2M@44k** | **80.65** | 79.64 | 71.20 | **75.24** | 67.83 | **35.15** |

> 151k / 44k 两行同为 2026-09-07 本地单卡跑（见上方 SAbDab 旧快照表下的说明）。产物
> `output/downstream_generation/ours_fusion_v3_{allchains_8gpu2m_151000,genonly_8gpu2m_44000}_sab23h2_cdr{h,l}{1,2,3}_iter2.log`。
> **两臂互有胜负、无一方领先**：allch 赢 L3 +1.54 / H2 +0.09，genonly 赢 L1 +0.33 / H1 +0.72 / H3 +0.59，
> L2 完全相同 —— 六 CDR 均值 **68.29 vs 68.28**，差 0.01。
> 全部落在本节已登记的 `max_iter` 协议不确定性带宽内（官方 ckpt 自身带宽 L2 1.04 / L3 0.92 / H3 2.05 pp），
> 因此**不得据此声称全链与 generated-only 谁更好**。

> v3 的 SAbDab 走的是**旧快照**（`data/downstream/cdr_infilling/sabdab`），不是 Kong 3,127。Kong 主表不要填 v3。

> **口径换了，「谁赢」的细节就变。** 8B 对官方：固定 iter=1 时 L1 −0.69 / L2 −1.27 / L3 −0.36 / H1 **+1.19** / H2 −0.37 / H3 −0.58；逐 CDR 取最优时则是 L1 −0.88 / L2 −1.04 / L3 −1.22 / H1 +0.48 / H2 −0.36 / H3 −1.10。两套口径下「幅度 ≤1.3 pp、基本打平」的结论都成立，但逐项符号和大小都会动，所以**必须先声明口径再比较**。
>
> **论文未公布 `max_iter`，而两个数据集指向不同取值。** SAbDab 上官方 ckpt 在 iter=1 最贴论文（H3 43.70 vs 43.55，Δ +0.15；iter=2 则 43.11，Δ −0.44）；SAb23H2 上却是 iter=2 最贴（六项 Δ 为 −0.18/+0.11/−0.89/−0.28/−0.01/−0.10，最大 0.89 pp），iter=1 下最大偏差扩到 1.40 pp（L3 −1.34、H3 −1.40）。没有单一取值能同时贴合两个数据集，所以残差应记为**协议不确定性**，不应说成「已在 ±0.6 pp 内复现」——那个 ±0.6 是逐 CDR 取最优后的产物。
>
> ⚠️ **决定性的一点：`max_iter` 这一个未公布参数对官方模型的影响，比我们与官方的整条差距还大。** 官方 ckpt 在 iter ∈ {1,2,4,8} 上的带宽为 L1 0.39 / L2 1.04 / L3 0.92 / H1 0.71 / H2 1.27 / **H3 2.05** pp，而我们与它的差距六项都 ≤1.3 pp。相比之下我们自己的带宽极小（270m：除 H1 的 0.24 外全为 0；8B：≤0.78）。**因此在 SAb23H2 上不能声称任何一方领先**——排序在该参数的合理取值范围内就会翻转。
>
> 我们的模型对 iter 几乎不敏感，这一点本身仍待解释（见 SAbDab 节同一现象）：多轮 refine 既不改善也不破坏，可能说明迭代解码没真正利用已提交的上下文，不应算作稳定性优点。

> **Ophiuchus 复现状态：两个数据集都已复现，但「±0.6 pp」这个数字要收紧。** SAbDab 三项 ±0.4 pp（换 Kong 版数据后，固定 iter=1，无选参，站得住）。SAb23H2 原报的 ±0.6 pp 是**逐 CDR 在测试集上取最优**得到的；在单一固定 iter 下，最好情况（iter=2）是 ±0.89 pp，iter=1 则是 ±1.40 pp。模型权重、解码路径、评测口径本身是对的，但复现紧度应按固定口径报。
> **Ours vs Ophiuchus**：SAbDab 上我们 H1/H2 略优、H3 略逊（−0.24）；SAb23H2 上固定 iter=1 时 H1 +1.19、其余 −0.36…−1.27，逐 CDR 取最优时 H1 +0.48、其余 −0.36…−1.22。**幅度都在 1.3 pp 内。结论是「基本打平」，且由于 `max_iter` 单参数就能让官方模型移动 0.4–2.05 pp（大于整条差距），任何一方都不能声称领先。**
> 8B 稳定优于 270m 但幅度极小（SAb23H2 上 0.2–0.6 pp）；旧快照 top-k 波动带 ≤0.2 pp，所以这个小幅优势是真实的。
> 修 encoder 泄露前这两张表的 Ours 曾虚高 3–16 pp（SAbDab H3 曾报 54.5/57.9、SAb23H2 H3 曾报 48.5/51.3），那批数字已作废。

### §0.6 AB Light-chain pairing（OAS holdout500，prompt3，n=8，**diffusion-only**）

> ✅ **`max_iter=124` 已对齐并收口**：`eval-immune-270m-diff-pairing`（`t-20260825150942-whmkl`）、`eval-immune-8b-diff-pairing`（`t-20260825150946-sxd64`）、`eval-ophiuchus-pairing-official`（`t-20260825150949-7f74h`，cfg 0/1/1.5）。Ours 走 `light_length_mode=prior` + gumbel；官方走固定 128 槽 + 自吐 EOS。长度泄露与 argmax 塌缩均已修。

> **论文 Table 3 已取回并逐位核对（2026-08-30）**，来源 bioRxiv 全文 PDF（DOI `10.64898/2026.02.02.703197`）。该表分**两个设定**，三条 `[P]` 行**逐位对应「有三残基提示」那一区**（Ophiuchus 0.695 / 49.2% / 99.7% / 0.356 / 0.273 / 0.894 / 0.335 / 100.0% 全中），与本节 `prompt3` 协议一致，**对齐正确**。论文协议：500 条 held-out 重链，每条生成 8 条光链。

| Model | 解码 | ImmunoMatch↑ | Better↑ | Chain | V | J | V-fam | J-fam | Div.↑ | valid | W_prop↓ | 类型 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ophiuchus-Ab | 论文，prompt3 | **0.695** | **49.2%** | 99.7% | 0.356 | 0.273 | 0.894 | 0.276 | 0.335 | 100.0% | **0.066** | `[P]` |
| p-IgGen | 论文，prompt3 | 0.669 | 45.3% | 98.4% | 0.313 | 0.283 | 0.874 | 0.288 | 0.390 | 99.2% | 0.118 | `[P]` |
| LiChen | 论文，prompt3 | 0.633 | 42.7% | 99.7% | 0.298 | 0.290 | 0.890 | 0.291 | 0.254 | 99.9% | 0.150 | `[P]` |
| Ophiuchus-Ab（官方 ckpt，gumbel，cfg=1.5）☆ | iter 124 | 0.706 | 50.5% | 99.8% | 0.336 | 0.280 | 0.893 | 0.290 | 0.329 | 100.0% | — | `[R]` |
| Ophiuchus-Ab（官方 ckpt，gumbel，cfg=1.0） | iter 124 | 0.704 | 50.3% | 99.8% | 0.335 | 0.284 | 0.890 | 0.293 | 0.333 | 100.0% | — | `[R]` |
| Ophiuchus-Ab（官方 ckpt，gumbel，cfg=0） | iter 124 | 0.668 | 45.1% | 99.7% | 0.319 | 0.282 | 0.863 | 0.289 | 0.359 | 100% | — | `[R]` |
| p-IgGen（官方权重本地复跑） | top_p 0.95, T 1.2 | 0.683 | 47.3% | 99.0% | 0.320 | 0.282 | 0.880 | 0.287 | 0.368 | 99.7% | — | `[R]` |
| LiChen（官方 Zenodo 权重本地复跑） | 默认 | 0.631 | 42.1% | 99.8% | 0.292 | 0.276 | 0.887 | 0.279 | 0.254 | 100.0% | — | `[R]` |
| **Ours-Diffusion** 8B@45000 | iter 124, prior, cfg=0 | 0.440 | 26.1% | 97.6% | 0.197 | 0.302 | 0.688 | 0.306 | 0.524 | 98.2% | — | `[L]` |
| **Ours-Diffusion** 270m@42000 | iter 124, prior, cfg=0 | 0.417 | 24.6% | 94.8% | 0.207 | 0.306 | 0.651 | 0.311 | 0.577 | 95.3% | — | `[L]` |
| **Ours-Diffusion** v3@27000 | iter 124, prior, cfg=0 | 0.353 | 21.5% | 88.1% | 0.179 | 0.272 | 0.571 | 0.276 | 0.651 | 88.7% | — | `[L]` 七源 |
| **Ours-Diffusion** v3@42000 | — | — | — | — | — | — | — | — | — | — | — | CUDA Failed |
| **Ours-Diffusion** allch@33000 | iter 124, prior, cfg=0 | 0.436 | 25.9% | 94.6% | 0.204 | 0.299 | 0.671 | 0.302 | 0.551 | 95.9% | — | `[L]` 全链 |
| **Ours-Diffusion** allch@18k | iter 124, prior, cfg=0 | 0.416 | 24.8% | 94.4% | 0.189 | 0.295 | 0.598 | 0.299 | 0.588 | 95.4% | — | `[L]` 全链续训快照 |
| **Ours-Diffusion** allch@26k | iter 124, prior, cfg=0 | 0.429 | 26.3% | 94.6% | 0.185 | 0.296 | 0.638 | 0.300 | 0.546 | 95.3% | — | `[L]` 全链续训最新 |

> **三个方法的论文值都已本地复现，不只是 Ophiuchus。** 此前只复跑了官方 Ophiuchus 权重；实际 `output/downstream_generation/pairing_baselines/` 下还有 p-IgGen 与 LiChen 的官方权重本地复跑（prompt3 与无提示各一份），从未进入本表。LiChen 复现极准（ImmunoMatch −0.0015、Better −0.6 pp、V −0.007、V-fam −0.003、Div +0.0005，九项 |Δ|≤0.015）；p-IgGen 稍松（ImmunoMatch **+0.014**、Better **+2.0 pp**，其余 |Δ|≤0.023），偏差方向与其随机解码（top_p 0.95 / T 1.2）一致，属采样噪声而非口径错误。**这把「harness 正确」的证据从一个方法扩到三个方法、两种设定**，所以下面 Ours 的落后不能再归因于评测实现。
>
> **`W_property` 是真实覆盖缺口**：`comp_chain/eval_scripts/generation_eval.py` 里没有任何 wasserstein / 属性分布代码，故 `[R]`/`[L]` 行留空；补齐前不能声称与 Table 3 指标全对齐。这一项尤其值得补——论文里 Ophiuchus 的领先幅度在此最大（0.066 vs p-IgGen 0.118 / LiChen 0.150）。
> （早先在本节写过「J-fam 我们也没算」，**该判断是错的**：`overall_j_gene_family_match_rate` 一直在产出，只是没进表，现已全部填入。）
>
> **论文「无提示」设定（供对照，勿与上表混排）**：LiChen 0.657 / 46.2% / 62.3% / 0.049 / 0.156 / 0.215 / 0.157 / 0.515 / 100.0% / 0.140；p-IgGen 0.674 / 50.5% / 50.5% / 0.045 / 0.139 / 0.171 / 0.142 / 0.709 / 99.1% / 0.103；Ophiuchus-Ab 0.701 / 51.6% / 55.6% / 0.056 / 0.150 / 0.169 / 0.152 / 0.676 / 100.0% / 0.072。
> **两个设定不可互换，且方向相反**：三残基提示把 Ophiuchus 的 chain 一致性从 55.6% 拉到 99.7%、V-fam 从 0.169 拉到 0.894（**5.3 倍**），但 ImmunoMatch 从 0.701 **降到** 0.695。论文自己的解释是额外条件把可行光链空间收窄到了记录中的那一条参考序列。**所以任何本地行在与论文比较前必须先声明属于哪个设定**——本节全部是 prompt3。
>
> 参考 light 自身 ImmunoMatch = **0.699**（同一 holdout，是「完美复制」上界）。
>
> ⚠️ **「官方 cfg=1.5 的 0.706 略超论文 0.695」这一表述已收紧为 ☆ 而非 ★。** cfg ∈ {0, 1.0, 1.5} 是我们扫出来的，1.5 之所以被选中正是因为它在**同一测试集**上最高（0.668 / 0.704 / 0.706）；论文未公布其 cfg 取值，0.695 落在我们 cfg=1.0（0.704）与 cfg=0（0.668）之间。**在测试集上挑选超参后再宣称「超过论文」是不成立的**；可以成立的说法是：官方权重 + 官方解码路径在 cfg 取 1.0–1.5 时落在论文值附近（+0.009 ~ +0.011），口径已复现。这不影响下面「我们落后 0.26」的结论——那个差距在任何 cfg 下都成立。

#### 落后 0.26 的机制（2026-08-30 定位）

之前只知道差距大小，不知道差距来自哪里。补了一个原表和论文都没有的**零假设锚点**：把同一批 500 条参考轻链与**打乱后的重链**配对（无固定点排列，3 个独立种子）打 ImmunoMatch。此时轻链本身完全天然，唯一错的就是配对关系，所以它给出任何「看起来像真轻链」的生成器免费继承的地板。脚本 `scripts/pairing_null_controls.py`，产物 `outputs/ab_pairing_null_controls.json`。自检：同一打分器在**正确**配对上得 **0.6993**，与产物里的 `ref_immunomatch_mean` 0.69934 逐位吻合。

| 锚点 | ImmunoMatch |
|---|---:|
| 正确配对（参考轻链 ↔ 真实重链） | 0.6993 |
| **错配配对（参考轻链 ↔ 打乱重链）** | **0.3328 ± 0.0183**（3 种子） |
| 可用的特异性空间 | 0.3665 |

**结论一：ImmunoMatch 确实是强配对特异性指标，地板是 0.33 而不是 0.66。** 我一度以为它主要在测「像不像天然轻链」（因为论文无提示行 V 基因匹配仅 0.049 却仍有 0.657），这个猜想被此对照否掉了。正确解读是：ImmunoMatch 测的是**与该重链的相容性**，而 V 基因匹配测的是**是否复原了记录中的那一条**；无提示方法能生成相容但非记录伙伴的轻链，所以两者可以同时成立。

**结论二：归一化后差距比原始 0.26 更大。** 把地板设为 0，正确配对设为 1：

| Method | IM（全行） | IM（仅有效行，估） | 占特异性空间 |
|---|---:|---:|---:|
| Ophiuchus 官方 cfg=1.5 | 0.7061 | 0.7065 | **101.8%** |
| Ophiuchus 官方 cfg=1.0 | 0.7038 | 0.7039 | 101.2% |
| Ophiuchus 官方 cfg=0 | 0.6677 | 0.6677 | 91.4% |
| p-IgGen 本地复跑 | 0.6831 | 0.6855 | 95.6% |
| LiChen 本地复跑 | 0.6315 | 0.6315 | 81.5% |
| **Ours 8B@45000** | 0.4400 | 0.4479 | **29.2%** |
| **Ours 270m@42000** | 0.4173 | 0.4381 | 23.1% |

三个基线拿到 82%–102%，我们只拿到 23%–29%。**所以「落后 0.26」是偏轻的说法，实际是「只走完不到三分之一」。**

**结论三：有效性只解释 2.9%，不是主因。** `immunomatch_score.py` 把 abnumber 无法定型的序列打分置 **0** 并计入均值（第 424 行），`generation_eval.py` 再对全部 4000 行取均值，所以该指标把「序列无效」和「配对差」混在一个数里。我们 8B 有 1.78% 无效、270m 有 4.75%，基线 ≤0.35%。上表「仅有效行」列按 `mean_all / valid_rate` 反推（*近似*：`gen_valid_rate` 来自 ANARCI 解析，置零来自 abnumber 定型，两者都包 ANARCI 但不是同一集合）。修正后 8B 的差距从 0.2661 缩到 0.2585——**有效性只占 2.9%**，270m 占 7.8%。差距的主体不在这里。

**结论四（最直接的一条）：把重链打乱，官方掉 0.390，我们只掉 0.117。** 同一个错配手术也施加到各模型**自己生成的**轻链上（两臂用同一套 κ/λ 路由，故绝对值带一点路由近似，但「正确−错配」的落差不受影响）：

| Model | 正确重链 | 打乱重链 | 落差 = 条件化强度 |
|---|---:|---:|---:|
| 参考轻链（天然） | 0.6993 | 0.3328 | 0.3665 |
| Ophiuchus 官方 cfg=1.5 | 0.7065 | 0.3164 | **0.3901** |
| **Ours 8B@45000** | 0.4421 | 0.3253 | **0.1169**（官方的 30.0%） |
| **Ours 270m@42000** | 0.4299 | 0.3071 | 0.1227（官方的 31.5%） |

这一条不依赖任何归一化假设，直接测「换掉重链会损失多少分」。**我们只拿到官方条件化强度的 30%。**

**而且它把「天然度」这个嫌疑彻底排除了**：三个生成器错配后都落在同一地板（我们 0.3253 / 官方 0.3164 / 天然参考 0.3328，彼此相差 ≤0.017）。也就是说，就 ImmunoMatch 的地板而言，**我们生成的轻链和真实轻链一样「像轻链」**；差距 100% 来自重链条件化，不来自序列质量。8B 与 270m 之间的 0.012 差距也可以拆开：条件化几乎相同（0.1169 vs 0.1227，270m 反而略高），差异主要在天然度（0.3253 vs 0.3071，+0.018）。

顺带独立验证了上表的「仅有效行」近似：本对照的打分器从不置零，其正确配对值为 8B 0.4421 / 270m 0.4299 / 官方 0.7065，与按 `mean_all / valid_rate` 反推的 0.4479 / 0.4381 / 0.7065 相差 0.006 / 0.008 / **0.0000**。近似成立。

**结论五：失效在 V 区，不在 J 区。** 用参考胚系的边际碰撞率 Σp(v)² 去掉偶然命中（V 基因 0.0414 / V 家族 0.1461 / J 基因 0.1428 / J 家族 0.1457，来自同一 500 条的 `l_v_call`/`l_j_call`）：

| 位点 | 类别数 | 偶然 | Ours 8B | 官方 cfg1.5 | 去偶然后 Ours / 官方 |
|---|---:|---:|---:|---:|---:|
| J 基因 | 11 | 0.143 | 0.302 | 0.280 | 0.186 / 0.160 → **116%** |
| J 家族 | 8 | 0.146 | 0.306 | 0.290 | 0.187 / 0.169 → 111% |
| V 家族 | 32 | 0.146 | 0.688 | 0.893 | 0.635 / 0.875 → 73% |
| V 基因 | 66 | 0.041 | 0.197 | 0.336 | 0.162 / 0.307 → **53%** |

**在 J 端我们与官方持平（略高），在 V 端只有官方的一半。** 这是个自洽的机制：轻链 J 区是 C 端十来个残基、只有 11 种选择，基本可由轻链自身语法预测，不需要重链信息；V 区是 N 端近百个残基，VH–VL 界面残基正落在这里，是真正决定配对的部分。我们学到了轻链语法而没学到重链条件化，于是 ImmunoMatch 落在错配地板附近。**而且 prompt3 协议本身就把参考轻链的前 3 个残基（属 V 区 N 端）交给了模型，我们在有这点提示的情况下 V 端仍只有官方一半，问题更显著。**

顺带否掉另一个猜想：**我们并没有完全忽略重链**——V 基因匹配 0.197 是边际碰撞率 0.0414 的 **4.8 倍**，模型确实携带了重链特异信息，只是量级不足；这与结论四测到的「30% 条件化强度」是同一件事的两种测法，量级也吻合。多样性方向也一致：我们 0.524 vs 官方 0.329，8 条样本铺得更开，符合「V 区条件化不够自信」的图像。

**合起来的诊断**：问题不在解码、不在长度处理、不在评测实现、也不在序列天然度，而在**重链→轻链的跨链条件化强度，且集中在轻链 V 区**（VH–VL 界面所在）。可验证的下一步（都不需要重训）：给我们自己的模型扫 classifier-free guidance（目前只有 cfg=0，官方从 cfg=0 到 1.5 涨 0.038，若我们同样敏感则部分差距只是解码设置）；出逐区段（FR1/CDR-L1/FR2/CDR-L2/FR3/CDR-L3/FR4）一致性，看 V 区缺口落在界面骨架还是 CDR。

**`max_iter` 32 → 124 几乎不动（历史对照，不与上表混排）**

| Run | iter 32 IM | iter 124 IM | Δ |
|---|---:|---:|---:|
| Ophiuchus gumbel cfg=0 | 0.641 | 0.668 | +0.027 |
| Ours 8B@45000 | 0.438 | 0.440 | +0.002 |
| Ours 270m@42000 | 0.414 | 0.417 | +0.003 |

加步数只帮了官方基线（+2.7 pp，再加 cfg=1.5 到 +6.5 pp），**帮不了我们**。argmax 塌缩那行（IM 0.352 / diversity ≈0）已作废，不再列入。

**泄露诊断（iter=124 产物，均为干净）**

| Run | 长度一致率 | 与参考相同度 | 逐字复制 | 每 heavy 唯一数 | 判定 |
|---|---:|---:|---:|---:|---|
| Ophiuchus cfg=1.5 | 0.353 | 0.796 | 0.15% | 7.91/8 | 无泄露、无塌缩 |
| Ophiuchus cfg=0 | 0.350 | 0.793 | 0.20% | 7.94/8 | 无泄露、无塌缩 |
| Ours 8B@45000 | 0.155 | 0.824 | 0.05% | 8.00/8 | 无泄露、无塌缩 |
| Ours 270m@42000 | 0.155 | 0.823 | 0.05% | 7.99/8 | 无泄露、无塌缩 |

> 结论（口径已干净且步数已对齐）：**我们与官方基线的真实差距是 0.26–0.29 ImmunoMatch**（0.417/0.440 vs 0.706），与论文差 0.26。V-gene 一致性 0.20 vs 官方 0.34、V-family 0.65–0.69 vs 官方 0.89，说明生成的 light 在**基因家族层面就偏离**。diversity 反而更高（0.52–0.58 vs 官方 0.329），即更发散但配对兼容性更差。8B 略优于 270m（+0.023）。
> 修复前那批数字（Ours 0.648–0.650）是长度泄露导致的近重建（长度 100% 一致、相同度 0.946），已作废。

### §0.7 本轮小结

各任务的完整对照表见 §0.1–§0.6，每张表内已含全部 baseline 与 Ours，不再另做跨任务合表。

> ⚠️ **本节曾把六项任务全部记为「口径干净、可直接引用」，那是 2026-08-29/30 baseline 审计之前的判断，已作废。**
> 审计逐项推翻了其中四条：T1 的 AUPRC 估计量用错（sklearn AP vs 论文的 `precrec`）且官方 unseen 切分自带污染；
> AB CDR 的 best-of-sweep 等于测试集调参、且抗体侧去污染被禁用；T2 的 (c) 区仍在旧 universe 上；
> T3 的头部 baseline 多拿 V/J 基因，不同输入字段不可直接同列。**可引用性的唯一权威是
> [`README.md`](README.md) §3.1 锚点登记表**，本节只做指向，不再复述判定。

**口径状态**（判定依据见 §3.1 登记表；🚨 = 不可引用，⚠️ = 有条件，✅ = 可引用）

| 任务 | 状态 | 限制条件（引用时必须同时说明） |
|---|:---:|---|
| T1 Binding | ⚠️ | AUPRC 必须报 `precrec` 口径；original unseen 须附 `TTAATHREK` 污染注（kNN 的 unseen 信号 100% 来自污染）；4 个 baseline 的 `[R]` 标签偏乐观 0.003–0.045 |
| T2 Clustering | ⚠️ | (a)(b) 可引用（校准 9/9，最大 \|Δ\| 0.005）；**(c) 区嵌入曲线仍在旧 5,368 universe 上，不可引用** |
| T3 Representation | ⚠️ | 必须按输入字段分层——SCEPTR/TCRdist 多拿 V(/J) 基因；且见下方缺陷 (f)，现表低估我方约 0.047 |
| T4 Generation | ⚠️ baseline / 🚨 Ours | baseline 侧：跨模型只能用 `bioseq_unseen_common`（固定 6 pMHC）；Char-BLEU 的 greedy/sampled 口径跨模型不一致；Setting A 的 `ours_bioseq` 有出处缺陷。**Ours 侧（08-31 改判）：不得表述为「学到了不错的无条件先验」或「优于 OLGA 下限」**——去掉表位的 infilling 里我们在 11 组条件下全部显著输给不看上下文的 (长度,位置) PWM（−4.15…−12.47 pp，含 8B 与训练见过的行），属拟合失败，见 §0.0(h) |
| AB CDR infilling | 🚨 | **Ours 行只能作上界**：Kong 折 54.4% 测试抗体的 CDR-H3 原样在训练语料中（胚系对照 0.83%）。已改固定 `iter=1`，best-of-sweep 已撤回。SAb23H2（污染面 25%）相对可引用 |
| AB Light pairing | ✅ | 唯一口径完全对齐的任务。须声明属 Table 3 的哪一种条件化设定（两者方向相反）；`W_property` 未实现，覆盖 9/10 |

- 另有横跨 T1/T2/T3 的读出口径缺陷 **(f)**（见 §0.0）：表内数值是「最后一层 + 未做任何重标定」，
  已实测低估我方 T3 约 0.047，**该缺陷不影响 baseline 之间的比较，但使 Ours 的绝对水平偏低**。
- 本轮不评：AB Humanization（按计划排除）、GDPa1 / Specificity / m396（无本地数据）。

**待办**

1. §T2(c) 与 §0.2 基础A 阈值曲线在修正后的 4,779 universe 上重跑（需重做 GPU 嵌入，留给 v2 ckpt）。
2. 是否对 ASD 启用 blocklist 并重训——实测已确认 ASD 是唯一污染源，OAS 干净。
3. 是否重建一个去污染的 CDR 评测子集（Kong 折中约 45% 无 H3 命中且无 ≥0.95 同一性命中的部分）。
4. T1 的 top-k 波动带（单 ckpt 约 113 分钟，本轮有意跳过）。
5. 按缺陷 (f) 的结论用 `pcaw256` / `center` 重出 T2/T3 的 Ours 行。
6. v3 4 卡 pairing / T4 首次 CUDA Failed，已重提；补数后回填 §0.4 / §0.6 / §0.8。

### §0.8 v3 完整对照（2026-09-01）

> **这一节是跨任务总表。** 逐任务细节与 baseline 全表仍在 §0.1–§0.6。
>
> **三组不能混着排序，只能同组内比、跨组看方向：**
>
> | 组 | 是什么 | 互比？ |
> |---|---|---|
> | A. 官方 / 专用 baseline | epiTCR、TEIM、SCEPTR、Ophiuchus、TCRT5、OLGA 等 | 与 `[L]` 分层，不合并排名 |
> | B. 旧 immune 四条 | 六源、去污前；270m/8B × BERT/diffusion | 组内可比 |
> | C. v3 diffusion | 七源、去污后、generated-only / global 128；27k vs 42k **先是进度差，不是卡数** | 两条 v3 可比；**≠ 组 B 的 270m@42000** |
> | D. v3 all-chains | 七源、去污后、`--diffusion_all_chains` / global 256；4 卡@33000（50k cosine）vs 8 卡 2M@18000 / @26000（polynomial 重开） | 下游可并排看方向；**`eval_loss` 不可比**；**≠ 组 C** |
>
> 组 D：33000 / 18000 / **26000 均已收口**（26000 四条 2026-09-02 02:43–05:27Z Success）。spot 本轮不评。

**表征（T1 / T2 / T3）**

| 指标 | 最强 baseline | 旧 270m BERT | 旧 270m diff | 旧 8B diff | v3@27k | v3@42k | allch@33000 | allch@18k | allch@26k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 seen AUPRC | epiTCR **0.825** / TEIM 0.780 | 0.752 | 0.768 | **0.778** | 0.763 | 0.767 | 0.766 | 0.761 | 0.761 |
| T1 seen_ind AUPRC | epiTCR **0.708** | 0.569 | 0.585 | 0.598 | 0.589 | 0.565 | 0.588 | 0.593 | 0.578 |
| T1 unseen AUROC | TEIM 0.534（全体≈随机） | 0.525 | 0.520 | 0.513 | 0.534 | 0.520 | 0.514 | 0.501 | 0.518 |
| T2 ARI mean | SCEPTR **0.033** / TCR-BERT 0.016 | 0.019 | **0.028** | 0.025 | 0.017 | 0.021 | 0.022 | 0.022 | 0.022 |
| T3 deep k=200 | SCEPTR 0.790（+V）；同输入 Lev 0.733 | 0.712 | 0.709 | **0.720** | 0.705 | 0.709 | 0.703 | 0.710 | 0.707 |
| T3 broad k=100 | SCEPTR 0.741（+V）；Lev 0.683 | 0.663 | 0.652 | **0.680** | 0.656 | 0.648 | 0.648 | 0.649 | 0.642 |
| T3 probe-AUROC | — | 0.804 | 0.794 | **0.828** | 0.794 | 0.786 | 0.798 | 0.798 | 0.782 |

**生成（T4 / CDR / pairing）**

| 指标 | 最强 baseline | 旧 270m diff | 旧 8B diff | v3@27k | v3@42k | allch@33000 | allch@18k | allch@26k | **allch@121k** | **allch@151k** | **genonly-2M@44k** |
|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|
| T4 common d_edit↓ | GRATCR **4.61** / OLGA 6.34 | 6.99 | 6.30 | **8.40** | **8.58** | 8.62 | 8.73 | 8.58 | 8.78 | 8.80 | 8.61 |
| T4 common seq-rec↑ | TCRT5 **0.585** / OLGA 0.470 | 0.420 | 0.483 | 0.242 | 0.215 | 0.211 | 0.213 | 0.212 | 0.210 | 0.218 | 0.221 |
| T4 F1@100 | TCRT5 0.0003 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| T4 held20 d_edit↓ | TcrDesign **1.47** | 5.60 | 5.18 | 6.43 | 6.75 | 6.73 | 6.76 | 6.69 | 6.74 | 6.75 | 6.72 |
| SAbDab 旧快照 H3（iter=2） | — | 41.91 | 42.06 | 40.75 | 41.62 | 41.63 | 41.44 | 41.68 | — | **41.92** | 41.79 |
| SAb23H2 H3 AAR（iter=2） | Ophiuchus 36.7 | 35.0 | 35.2 | 34.1 | 34.8 | 35.4 | 34.8 | 34.8 | — | 34.6 | 35.1 |
| SAb23H2 六 CDR 均值（iter=2） | — | 68.22 | — | 67.35 | 67.69 | 67.91 | 68.01 | 67.70 | — | 68.29 | 68.28 |
| Pairing ImmunoMatch | Ophiuchus **0.706** | 0.417 | 0.440 | **0.353** | Failed | 0.436 | 0.416 | 0.429 | 待补 | **排队中** | **排队中** |
| Pairing Better | Ophiuchus 50.5% | 24.6% | 26.1% | 21.5% | Failed | 25.9% | 24.8% | 26.3% | 待补 | **排队中** | **排队中** |
| Pairing valid | 官方 ≥99.7% | 95.3% | 98.2% | 88.7% | Failed | 95.9% | 95.4% | 95.3% | 待补 | **排队中** | **排队中** |

> SAb23H2 六 CDR 均值 = (L1+L2+L3+H1+H2+H3)/6，只作组内速读。SAbDab Kong 主表无 v3（本轮走旧快照）。
>
> **151k / 44k 两列的 T4 与 CDR 是 2026-09-07 本地单卡跑**（`queue012` 排不上，改本机 A100 串行；
> env 与 eval YAML 逐行一致，过程见 [`$ROOT/PROJECT_PROCESS.md`](../../PROJECT_PROCESS.md) 2026-09-07 节）。
> **两列的 pairing 仍在 `queue012` 排队**（`t-20260907044307-btjxr` / `t-20260907044321-ht8b6`），
> 未跑，故本表该三行标「排队中」而非留空 —— 这两列目前是**半套数字**，不得据此写整体结论。

**相对旧 270m diffusion@42000 的 Δ（v3 − 旧；正=更好。跨配方，只看方向）**

| | v3@27k | v3@42k |
|---|---:|---:|
| T1 seen AUPRC | −0.005 | −0.001 |
| T1 seen_ind AUPRC | +0.003 | −0.021 |
| T1 unseen AUROC | +0.014 | 0.000 |
| T2 ARI | −0.011 | −0.007 |
| T3 deep k=200 | −0.004 | 0.000 |
| T3 broad k=100 | +0.004 | −0.004 |
| T4 common d_edit（负=更差） | −1.42 | −1.60 |
| SAb23H2 H3 | −0.9 pp | −0.2 pp |
| Pairing IM | −0.064 | — |

**读法**

1. **Seen binding**：旧 8B 仍是我们最好的一档（0.778），卡在 TEIM 与 epiTCR 之间。v3 42k（0.767）≈ 旧 270m，没有超过。
2. **Unseen binding**：全员贴随机。v3 27k 均值碰到 TEIM，std 更大（0.011 vs 0.006），不能写成赢。
3. **表征几何（T2）**：v3 掉到 TCR-BERT 附近（0.017–0.021），旧 diffusion 0.028 的优势没保住。42k > 27k，仍低于 SCEPTR。
4. **T3**：v3 ≈ 旧 270m diffusion，规模/配方都没拉开。同输入仍低于 Levenshtein。
5. **T4**：v3 27k / 42k 都比旧 270m 差（8.40 / 8.58 vs 6.99），也差于无条件 OLGA（6.34）。F1 仍是 0。
6. **CDR**：SAb23H2 上 v3 略低于旧 270m（H3 −0.2…−0.9 pp），仍在「和 Ophiuchus 同档、只能当上界」的区间。
7. **Pairing**：v3 27k ImmunoMatch 0.353，贴错配地板 0.333（约占特异性空间 6%）；旧 270m 是 23%。比旧条更差，不是打平。
8. **组 D 全链（33000 / 18k / 26k）**：表征仍与 generated-only 同档。26k 相对 18k：**pairing 回一点**（IM 0.416→0.429，仍低于源点 0.436），T4 common 8.73→8.58（回到组 C 42k 附近），T1 seen 持平 0.761。相对 33000 **仍无系统性变好**。这是 2M 的 1.3% 进度，不是长跑结论。
9. **长跑到 151k（2M 的 7.6%）仍未改变任何结论，两臂也打平**（2026-09-07 补，**T4 + CDR 部分，pairing 未到**）：
   - **T4 不降反微升**：全链 26k → 121k → 151k 的 common d_edit 是 8.58 → 8.78 → 8.80，seq-rec 0.212 → 0.210 → 0.218。仍远差于旧 270m（6.99）与不看表位的 OLGA（6.34）。**表位条件信号在 unseen 上接近不存在这条主结论，长跑没有撼动。**
   - **CDR 小幅上行但在噪声内**：SAbDab 旧快照 H3 26k 41.68 → 151k 41.92（+0.24），H2 +1.34；SAb23H2 六 CDR 均值 67.70 → 68.29。幅度与同臂 ckpt 间波动（18k→26k 的 H2 +0.41）同量级。
   - **全链 vs generated-only 打平**：151k − 44k 在 SAbDab H1/H2/H3 为 +0.28/+0.45/+0.13，SAb23H2 几乎完全相同（六 CDR 均值 68.29 vs 68.28，差 0.01），T4 则是 151k **落后** 0.19（8.80 vs 8.61）。**`--diffusion_all_chains` 至今无可见收益。**
   - ⚠️ 两列 pairing 未跑（仍在 `queue012` 排队），而 pairing 恰是 headline 指标。**这条读法只覆盖 T4 与 CDR，不构成本轮收口。**

**还缺**

- 组 D 33000 / 18000 / 26000 已收口。
- **本轮最好 val 点**：全链 151000 与 generated-only 44000 的 **T4 + CDR 已回填**（2026-09-07 本地跑）；
  **两条 pairing 仍在 `queue012` 排队**（`t-20260907044307-btjxr` / `t-20260907044321-ht8b6`），
  到齐才算收口。BERT 1M 105000 **只表征**，对应任务已 cancel，未跑，待定是否本地补。
- 全链 121000 的 pairing 数字亦未并入本表（作业 `t-20260906132314-lqbrp` Killed）。
- v3@42k 的 pairing（重提 `t-20260901011610-tb4hh`）；T4 已补（8.58 / 0.215）
- v3 BERT 表征数字尚未并入本表（作业已 Success，tag `ours_fusion_v3_bert_final`）
- v3 / 全链 SAbDab Kong（本轮走了旧快照）

---

## T1 — TCR-Epitope Binding

### T1 · Nature Methods 2025 官方 split（主协议）

> **文献锚定**：主基础 Nature Methods 2025《Assessment of computational methods in predicting TCR–epitope binding recognition》([s41592-025-02910-0](https://www.nature.com/articles/s41592-025-02910-0)，官方仓库 `SuoLab-GZLab/TCREpitopeBenchmark` commit `ec832b47`)；近似去重/低-FPR 补强 arXiv 2606.04994 (2026)。**主指标 = overall（all_values）AUPRC**；macro-AUPRC / AUROC / macro-AUC0.1 辅报。
> **数据边界**：external original-model baseline 只使用 figshare `original.zip`（seen 3 表位 / unseen 40 表位；主 neg=AS）派生 test，不读取 GitHub `train.csv`、不 retrain。`train.csv`（cdr3b 82,065 行）只供 Ours frozen-head、kNN 与 L1 sensitivity 等本地诊断，不属于 external baseline 协议。
> ⚠️ **shipped split 缺陷（2026-08-30 实测，见下方 §L3）**：官方标为 unseen 的 40 个表位里，`TTAATHREK` 实际出现在 `train.csv`（94 训练行），`unseen_test` 中该表位 66 行里有 **30 行是原样照抄的训练配对**（全部正例）。这是 shipped split 自身的标注缺陷，非本地处理引入。受影响的只有用 `train.csv` 训练的方法（Ours frozen-head / kNN）；**kNN 的 unseen 数字整体由此污染驱动**。
> **我们的模型（headline = post-LLaDA globalfeat）**：冻结 grammar_v2-LLaDA decoder 末层 → **完整** `tcr_peptide` record → **全局 mean-pool 单向量** → 官方 train 上 **MLP head**（`run_nm2025.py --method embed --embedder grammar:decoder:global: --head mlp`，符合 PROJ_GUIDE.md post-LLaDA 强制口径）；3 checkpoint（esmc300m-cmp500k / esmc600m-cmp500k / esmc300m-mint），tag `ours_globalfeat_*`。encoder-only（`grammar:encoder:`）与 segment-concat（`ours_postllada_*`）仅作 control。
> **baseline**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025/summary_original_baselines.csv` 是 canonical 来源表：完整可用时取官方 original checkpoint + 官方推理代码 + `original.zip` 的 `[R]`；不可运行时取 Supplementary Table 4 `[P]` 并标 **“论文值，未本地复现”**。当前 13 个模型为 `[R]`，SETE/TEPCAM 为 `[P]` fallback。同目录 `summary_local_runs.csv` 与 `baseline_paper_audit.csv` 仅作运行/差异诊断。kNN / Random 为 `[C]` 控制。

#### cdr3b · AS · overall AUPRC（主排名，随机≈0.5；Ours = post-LLaDA globalfeat headline）

> **AUPRC 口径 = `precrec::evalmod`（论文口径，2026-08-30 换定）。** 本表此前用 sklearn `average_precision_score`，与论文 Supplementary Table 4 的 `precrec` 不是同一个估计量，且**单侧偏高**（详见 §L4）。已按 `precrec` 重抄，AUROC / AUC0.1 两列不受影响（原本就吻合）。

| Method | Seen AUPRC↑ | Unseen AUPRC↑ | Seen AUROC | Unseen AUROC | Seen AUC0.1 | Unseen AUC0.1 | 类型 |
|--------|:-----------:|:-------------:|:----------:|:------------:|:---:|:---:|------|
| **ATM-TCR** | **0.6957** | 0.5208 | **0.6480** | 0.5218 | 0.663 | 0.674 | [R] original-only |
| **TEIM** | 0.6756 | 0.5159 | 0.6025 | 0.5161 | 0.676 | 0.622 | [R] original-only |
| **TEPCAM** | 0.6693 | 0.5128 | 0.5988 | 0.5120 | — | — | [P] 论文值，未本地复现 |
| **NetTCR** | 0.6110 | 0.5027 | 0.5903 | 0.4971 | 0.569 | 0.642 | [R] original-only |
| **SETE** | 0.5629 | — | 0.5463 | — | — | — | [P] 论文值，未本地复现 |
| epiTCR | 0.5577 | 0.4792 | 0.5357 | 0.4822 | 0.517 | 0.626 | [R] original-only |
| PanPep | 0.4980 | 0.4771 | 0.5065 | 0.4826 | 0.518 | 0.669 | [R] original-only |
| CDR3-kNN (k=5) | 0.558 | 0.544 ⚠️ | 0.507 | 0.504 ⚠️ | 0.551 | 0.511 | 训练free |
| Random | 0.511 | 0.500 | 0.517 | 0.520 | 0.496 | 0.694 | floor |
| **Ours-BioSeq** esmc300m-mint | 0.596 | 0.499 | 0.574 | 0.529 | 0.578 | 0.696 | [L] LLaDA-dec globalfeat+MLP |
| **Ours-BioSeq** esmc300m-cmp500k | 0.583 | 0.505 | 0.552 | 0.502 | 0.585 | 0.712 | [L] LLaDA-dec globalfeat+MLP |
| **Ours-BioSeq** esmc600m-cmp500k | 0.576 | 0.495 | 0.543 | 0.494 | 0.593 | 0.624 | [L] LLaDA-dec globalfeat+MLP |

#### 口径对照：headline globalfeat vs 两组 control（Ours-BioSeq · cdr3b · AS · 消融对照）

| Checkpoint | 口径 | Seen AUPRC | Unseen AUPRC | 状态 |
|------------|------|:----------:|:------------:|------|
| esmc300m-cmp500k | **post-LLaDA globalfeat（headline，`grammar:decoder:global:`）** | 0.583 | 0.505 | ✅ |
| esmc600m-cmp500k | **post-LLaDA globalfeat（headline）** | 0.576 | 0.495 | ✅ |
| esmc300m-mint | **post-LLaDA globalfeat（headline）** | 0.596 | 0.499 | ✅ |
| esmc300m-cmp500k | post-LLaDA segment-concat（control-A，`ours_postllada_*`） | 0.618 | 0.511 | ✅ |
| esmc600m-cmp500k | post-LLaDA segment-concat（control-A） | 0.538 | 0.515 | ✅ |
| esmc300m-mint | post-LLaDA segment-concat（control-A） | 0.493 | 0.516 | ✅ |
| esmc300m-cmp500k | encoder-only（control-B，`grammar:encoder:`） | 0.564 | 0.532 | ✅ |
| esmc600m-cmp500k | encoder-only（control-B） | 0.578 | 0.508 | ✅ |
| esmc300m-mint | encoder-only（control-B） | 0.569 | 0.518 | ✅ |

> AUPRC 口径同主表 = `precrec`（§L4）。

> **诚实结论**：globalfeat 是 PROJ_GUIDE.md 规定的 headline 口径，但其 unseen AUPRC（0.495–0.505）实际**略低于** encoder-only control（0.508–0.532）与 segment-concat control（0.511–0.516）——三口径 unseen 全部落在近随机带（0.49–0.53），post-LLaDA 相比 encoder-only 在 T1 unseen 上**未见提升**（如实标注）；seen 上 globalfeat（0.58–0.60）与其他口径相当。换 `precrec` 口径后三组各自下移约 0.002，相对结论不变。源：`outputs/tcr_binding_nm2025/{ours_globalfeat_*,ours_postllada_*,ours_biseq_mlp*}` + `auprc_convention_audit.csv`。

> **caveat**：① macro-AUPRC / macro-AUC0.1 unseen 40 表位、~17 行/表位方差大，不作主排名（如 Random unseen AUC0.1=0.694 属小样本假象）。② **ERGO-II** 原生用 CDR3β+α+V/J+MHC，本 track 仅喂 CDR3β（其余 UNK，同论文 partial-feature 口径），seen 近随机属预期。③ 汇总 `outputs/tcr_binding_nm2025/summary_main.csv`。④ ⚠️ **kNN 的 unseen 两栏（AUPRC 0.544 / AUROC 0.504）不可作为泛化能力引用**：它对 624 条非污染 unseen 行输出恒为 0.0 分，全部表观信号来自单个污染表位 `TTAATHREK`，剔除后 AUROC 精确塌到 0.5000（§L3）。

#### Original-model baseline 扩充：TEINet / ERGO 变体 / TPBTE（cdr3b · AS · overall AUPRC）

> 下表均已通过官方 original checkpoint + 官方推理代码 + `original.zip` 本地重跑，故选择 `[R]`；论文数值与 delta 仍保存在 audit 表中。AUPRC 口径同主表 = `precrec`（§L4）。

| Method | Seen AUPRC↑ | Unseen AUPRC↑ | Seen AUROC | Unseen AUROC | 类型 |
|--------|:-----------:|:-------------:|:----------:|:------------:|------|
| TEINet-large | 0.6663 | 0.5026 | 0.6006 | 0.5023 | [R] original-only |
| TEINet-small | 0.6559 | 0.4985 | 0.5923 | 0.5053 | [R] original-only |
| ERGO-lstm (vdj) | 0.6279 | 0.5171 | 0.5770 | 0.5116 | [R] original-only |
| ERGO-AE (vdj) | 0.6150 | 0.5041 | 0.5428 | 0.4994 | [R] original-only |
| ERGO-lstm (mc) | 0.6091 | 0.5067 | 0.5614 | 0.5108 | [R] original-only |
| ERGO-AE (mc) | 0.5798 | 0.5206 | 0.5152 | 0.5051 | [R] original-only |
| TPBTE (mc) | 0.4999 | 0.4983 | 0.5055 | 0.5037 | [R] original-only |
| TPBTE (vdj) | 0.5000 | 0.5000 | 0.5000 | 0.5000 | [R] original-only |

> **结论（与 NM2025 一致）**：TEINet/ERGO 变体 seen 有信号（AUPRC 0.58–0.67）、**unseen 全部回落近随机（0.50–0.52）**；TPBTE 两变体 seen/unseen 均≈随机（0.50）。SETE 的官方 pickle 在 seen/AS 报输入 433 维而模型期待 1 维，故使用论文 seen 值；论文未给 SETE unseen 值，保持空缺而不补造。TEPCAM wrapper 尚未接通，seen/unseen 均明确使用论文值。

#### L2 · 负样本来源对照（AS/PS/HS · overall AUPRC；`summary_neg_source.csv`）

| Method | seen AS | seen PS | seen HS | unseen AS | unseen PS | unseen HS |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| ATM-TCR | 0.696 | 0.682 | 0.666 | 0.522 | 0.508 | 0.519 |
| NetTCR-2.2(b) | 0.611 | 0.622 | 0.578 | 0.505 | 0.572 | 0.526 |
| TEIM | 0.676 | 0.700 | 0.741 | 0.518 | 0.536 | 0.596 |
| ERGO-II | 0.545 | 0.556 | 0.554 | 0.510 | 0.507 | 0.509 |
| epiTCR(noMHC) | 0.557 | 0.541 | 0.543 | 0.482 | 0.500 | 0.490 |
| PanPep | 0.499 | 0.504 | 0.590 | 0.479 | 0.448 | 0.510 |
| CDR3-kNN(k=5) | 0.553 | 0.526 | 0.649 | 0.544 | 0.544 | 0.544 |
| Ours esmc300m-cmp500k (globalfeat) | 0.583 | 0.560 | 0.657 | 0.507 | 0.484 | 0.547 |
| Ours esmc600m-cmp500k (globalfeat) | 0.576 | 0.521 | 0.650 | 0.496 | 0.471 | 0.518 |
| Ours esmc300m-mint (globalfeat) | 0.596 | 0.535 | 0.639 | 0.501 | 0.478 | 0.508 |

> **本地 sensitivity `[L]`**：换负样本来源会明显移动 AUPRC。该表不替代 canonical original-only 主表；主 baseline 固定 AS，并逐模型选择完整官方重跑 `[R]` 或带标签的论文 fallback `[P]`。

#### L1 · 近似去重（train↔test CDR3β Levenshtein≤k；`near_dedup_summary.csv`）

去掉与官方 train.csv 任一 CDR3β 距离 ≤k 的 test 行后在同一硬子集复算 overall AUPRC：

| | k=0(原) | k≤1 | k≤2 | k≤3 |
|---|:---:|:---:|:---:|:---:|
| 去掉 unseen 比例 | 0%(690) | 19.0%(559) | 67.4%(225) | 92.6%(51) |
| 去掉 seen 比例 | 0%(1956) | 9.5%(1770) | 57.3%(836) | 87.5%(244) |
| kNN unseen AUPRC | 0.544 | 0.487 | 0.499 | 0.490 |
| ATM-TCR unseen | 0.522 | 0.523 | 0.564 | 0.542 |
| TEIM unseen | 0.518 | 0.507 | 0.528 | 0.560 |
| Ours-cmp500k unseen (enc-only control) | 0.534 | 0.520 | 0.511 | 0.534 |

> **结论**：k≤2 去掉 unseen 67.4%（落 arXiv 2606.04994 的 40–70%）；去重后 unseen 仍全体≈0.5（"泛化失败"是真结论）；但 **kNN 表观优势 k≤1 即坍缩**（0.544→0.487），距离法吃近重复红利被证实。k≤3 仅剩 51 行不作主张，原口径全保留。

> ⚠️ **口径区别（不要与下一节混用）**：上表 L1 的判据是「CDR3β 对**全** train.csv 的任一 CDR3β 距离 ≤k，**忽略配对的是哪个表位**」——这是刻意放宽的判据（某条 CDR3β 在训练里配表位 X，拿去测表位 Y 时仍算近重复）。下一节 L3 用的是**严格**判据：完整 (CDR3β, 表位) 二元组是否原样出现在 train.csv 里。两者数值不可互相引用。

#### L3 · original 协议 split 的配对级污染（`original_unseen_contamination.json`，2026-08-30 新增）

> **动机**：官方 `original.zip` 把 40 个表位标为 "unseen"。若这个标签成立，`unseen_test` 与 `train.csv` 的精确 (CDR3β, 表位) 配对重叠应当**恒为 0**——出现任何命中都是 shipped split 自身的标注缺陷，而非某个模型的建模选择。脚本 `scripts/t1_original_unseen_contamination.py`（不改动任何既有产物）。

| split | n | 表位数 | **精确 (CDR3β, 表位) 命中 train.csv** | 标签一致 | 命中表位 | 该表位在 train.csv 的行数 |
|---|:---:|:---:|:---:|:---:|---|:---:|
| `seen_test` | 1,956 | 3 | 11 / **0.56%** | 11/11 | 3 个 seen 表位（符合设计） | — |
| `unseen_test` | 690 | 40 | **30 / 4.35%** | 30/30 | **`TTAATHREK` 独占全部 30 条** | 94 |

> **结论 1：`TTAATHREK` 不是 unseen 表位。** 40 个所谓 unseen 表位里有 1 个（`TTAATHREK`）本身就在 train.csv 的 73 个表位中，带 94 条训练行。该表位在 `unseen_test` 里占 **66 行（9.57%）**，其中 **30 行（占该表位 45.5%）** 是原样照抄的训练配对，且**全部是正例、标签全部一致**——即对任何用 train.csv 训练的方法来说，这 30 行是可以直接背下来的。`seen_test` 的 11 条（0.56%）同理但量级小得多，且 seen 侧共享表位本就是设计意图。

> **结论 2：AUPRC 的视图差不能直接读——它主要是正例率假象。** 被污染的行全是正例，剔掉后正例率从 0.500 掉到 0.477，于是**连 Random 的 AUPRC 都会跟着掉**。故下表报 `auprc_lift = AUPRC − 正例率`，并同时给出对正例率不敏感的 AUROC。剔掉整个 `TTAATHREK` 表位（624 行 / 312 正例）恰好让正例率回到 0.500，是最干净的一个视图。

| Method（训练语料） | AUROC 全集 | AUROC 剔 30 配对 | AUROC 剔整个表位 | AUPRC-lift 全集 | AUPRC-lift 剔整个表位 |
|---|:---:|:---:|:---:|:---:|:---:|
| **CDR3-kNN (k=5)**（train.csv） | 0.5039 | 0.4567 | **0.5000** | 0.0440 | **0.0000** |
| Ours globalfeat cmp500k（train.csv） | 0.5020 | 0.4791 | 0.4926 | 0.0070 | 0.0016 |
| Ours globalfeat mint（train.csv） | 0.5291 | 0.5141 | 0.5234 | 0.0013 | −0.0038 |
| Ours-BioSeq esmc600m（train.csv） | 0.5019 | 0.4836 | 0.4816 | 0.0109 | −0.0018 |
| Random（无训练） | 0.5196 | 0.5225 | 0.5254 | 0.0025 | 0.0089 |
| 11 个官方 baseline（各自私有语料）区间 | 0.4822–0.5161 | 0.4582–0.5419 | 0.4700–0.5196 | −0.021–0.023 | −0.026–0.028 |

> 🚨 **结论 3：kNN 在 original 协议 unseen 上的全部表观信号来自这一个污染表位。** kNN 对 624 条非 `TTAATHREK` 的 unseen 行给出的分值**唯一且恰好为 0.0**（无任何训练邻居落进阈值），只有 `TTAATHREK` 那 66 行有非零分（正例均分 0.226 vs 负例 0.053）。因此剔掉该表位后 kNN 的 AUROC 精确塌到 0.5000、AUPRC-lift 精确塌到 0.0000。**§T1 主表里 kNN 的 unseen AUPRC 0.544 应当读作「近重复+配对污染红利」，不是泛化能力**；这与上一节 L1「k≤1 即坍缩」互相独立地指向同一结论。

> **结论 4：除 kNN 外，本节不改变任何主表数字，也不改变 T1 的定性结论。** 24 个 Ours checkpoint 剔表位后 AUROC 变化 −0.0203…+0.0039（中位 −0.0077），三视图全部落在 0.4625–0.5335 的近随机带；11 个用私有语料训练的官方 baseline 视图差**无方向性**（6 升 5 降，−0.0157…+0.0083），符合「它们没见过 train.csv」的预期。主表继续按官方 shipped split 报，但引用 unseen 一栏时**必须**附注 `TTAATHREK` 污染；若审稿人要求纯净 unseen，用「剔整个表位」视图（624 行 / 39 表位 / 正例率 0.500）。

> **未对齐的 3 个方法**：`atmtcr`、`tpbte_mc`、`tpbte_vdj` 的 `predictions.csv` 用位置索引 id（`unseen_AS_<i>`）且 wrapper 重排了输出行，逐行 (label, group) 校验不通过，故**不猜测**、直接排除（脚本会显式报告）。其余 37 个方法全部通过校验完成对齐。

#### L4 · AUPRC 估计量口径：论文用的是 `precrec`，不是 sklearn AP（`auprc_convention_audit.csv`，2026-08-30 换定）

> **问题**：NM2025 的 AUPRC 由 R `precrec::evalmod` + `precrec::auc` 算出（tie 感知的非线性 PR 插值 + 梯形积分）；sklearn `average_precision_score` 是逐步求和，**两者不是同一个估计量**。retrained 协议（`run_retrained_*.py`）早已用 precrec 移植版，但 original 协议（`run_nm2025.py` → `common.metrics.official_binding_report`）一直报 sklearn AP，而 §T1 主表把这些 `[R]` 本地值和 `[P]` 论文值放在**同一列**里。脚本 `scripts/t1_original_auprc_convention.py`。

> **归因分解**：13 个有论文值的 baseline × 2 split = 26 行。用 **AUROC 是否吻合**把「口径差」和「跑数差」分开——AUROC 在两边都是同一个 sklearn 实现，若 AUROC 都对不上，说明是这一次本地重跑本身与论文那次不同，不能归因于 AUPRC 口径。

| 行集合 | n | `precrec` 平均 \|Δ\| | sklearn AP 平均 \|Δ\| | AP 偏差方向 |
|---|:---:|:---:|:---:|:---:|
| seen，AUROC 吻合（\|Δ\|≤0.001） | 9 | **0.000083** | 0.000398 | 8 正 / **0 负** |
| unseen，AUROC 吻合 | 10 | **0.000240** | 0.001803 | 9 正 / **0 负** |
| AUROC 本身对不上（跑数差） | 7 | — | — | NetTCR、epiTCR、TEINet-large/small |

> **结论 1：`precrec` 就是论文口径，且能精确复现。** 在 AUROC 也吻合的 19 行上，`precrec` 把论文值复现到平均 \|Δ\| = 8.3×10⁻⁵（seen）/ 2.4×10⁻⁴（unseen），多数行**恰好为 0.0000**；sklearn AP 差 4.0×10⁻⁴ / 1.8×10⁻³，且 **17/19 行为正、0 行为负**——不是随机噪声而是**单侧系统性偏高**。

> **结论 2：偏差对 Ours 同样成立，量级 ~+0.001（seen）/ ~+0.002（unseen）。** 54 个无论文值的行（全部 Ours checkpoint + kNN + Random）上 AP − precrec 平均 **+0.0010**、52/54 为正。因为 unseen 全体挤在 0.48–0.52 这条窄带里，+0.002 相当于「随机以上全部空间」的约 10%，不可忽略。

> **结论 3：已换口径，排名基本不动。** §T1 主表、globalfeat/control 对照表、TEINet/ERGO/TPBTE 扩充表的 AUPRC 列已全部按 `precrec` 重抄（AUROC / AUC0.1 两列原本就吻合，未动）。unseen 降序排名只有 ATM-TCR 与 ERGO-AE-mc 互换第 2/3 位，而两者 precrec 值差 0.0002，本就是并列。同时 `common/metrics.py` 的 `_confusion_row` 已新增 `prc_auc_precrec` 键（保留原 `prc_auc` 不动，不破坏既有消费方），后续 original 协议跑数会原生同时给出两个估计量。

> ⚠️ **尚未换算的三张辅助表**：上文 **L1 近似去重**、**L2 负样本来源（PS/HS）**、**L3 污染视图** 三表仍为 sklearn AP。它们都是「同一口径内部的相对比较」（k 之间、AS/PS/HS 之间、视图之间），单侧偏移不影响其结论方向，故本轮未重算；引用绝对值时需减去约 0.001–0.002，或直接查 `auprc_convention_audit.csv`。

> **4 个跑数差异（与口径无关，需单独追）**：epiTCR seen Δ_AUPRC −0.045 且 Δ_AUROC −0.021（其分值 tie 率高达 0.91，仅 175 个唯一分值）；NetTCR seen Δ_AUPRC −0.035 但 Δ_AUROC **+0.002**；TEINet-large seen Δ_AUPRC +0.0075 且 Δ_AUROC +0.0077；TEINet-small seen Δ −0.003。这四个的本地重跑与论文那一次不是同一个配置，`[R]` 标签对它们**偏乐观**，引用时应注明「本地重跑，与论文值有 0.003–0.045 的未解释差异」。

> 复现：
> ```bash
> ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
> export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH; cd benchmark
> $ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/prepare_nm2025_binding.py --tests-only
> $ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_nm2025.py --method random --track cdr3b
> $ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_nm2025.py --method knn --track cdr3b
> $ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_official_baseline.py --name epitcr --tag epitcr_nomhc --track cdr3b --neg-source AS
> $ENV/bin/python tcr_binding/run_nm2025.py --method embed --track cdr3b \
>     --embedder grammar:encoder:/abs/grammar_v2_esmc300m_cmp500k_llada/best.pt \
>     --head mlp --tag ours_biseq_mlp
> for b in atmtcr nettcr teim panpep; do
>   $ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_official_baseline.py --name $b --track cdr3b --neg-source AS; done
> $ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_official_baseline.py --name teinet --tag teinet_small --track cdr3b --neg-source AS
> $ENV/bin/python scripts/nm2025_near_dedup.py --splits seen unseen --neg AS --dedup-ref all
> $ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/summarize_nm2025.py
> ```

#### T1 retrained 五折 · baseline 与 Ours（独立诊断，不替换 original-only 主表）

> 数据直接取 Figshare `retrain.zip` 的五个 AS fold。8 个 baseline 使用发布的 official retrained checkpoint 本地推理；Ours 每 fold 用自己的 `i_1_1train.csv` 训练 frozen-backbone head。两张表分别展示 seen test 与 seen independent；baseline 均按 headline AUPRC 从高到低排列，Ours 固定在最后。当前展示先不包含 unseen。

##### Seen test

| Method | AUROC | AUPRC |
|---|---:|---:|
| epiTCR | **0.8047±0.0015 (#1)** | **0.8253±0.0016 (#1)** |
| TEIM | 0.7641±0.0029 (#3) | 0.7802±0.0035 (#2) |
| ATM-TCR | 0.7644±0.0017 (#2) | 0.7707±0.0044 (#3) |
| ERGO-AE | 0.7286±0.0040 (#4) | 0.7405±0.0043 (#4) |
| NetTCR | 0.7271±0.0050 (#5) | 0.7371±0.0056 (#5) |
| TCR-H | 0.6786±0.0044 (#6) | 0.6805±0.0054 (#6) |
| TEINet | 0.5818±0.0768 (#8) | 0.5906±0.0848 (#8) |
| ERGO-lstm | 0.5016±0.0018 (#9) | 0.4999±0.0007 (#9) |
| **Ours BioSeq frozen global-mean + MLP** | **0.6532±0.0026 (#7)** | **0.6699±0.0026 (#7)** |

##### Seen independent

| Method | AUROC | AUPRC |
|---|---:|---:|
| epiTCR | **0.6662±0.0030 (#1)** | **0.7080±0.0007 (#1)** |
| TEIM | 0.6171±0.0042 (#2) | 0.6303±0.0073 (#2) |
| ATM-TCR | 0.6067±0.0071 (#3) | 0.5881±0.0083 (#3) |
| ERGO-AE | 0.5701±0.0104 (#5) | 0.5788±0.0090 (#4) |
| NetTCR | 0.5713±0.0116 (#4) | 0.5630±0.0099 (#5) |
| TCR-H | 0.5674±0.0067 (#6) | 0.5596±0.0075 (#6) |
| TEINet | 0.5226±0.0211 (#8) | 0.5250±0.0245 (#8) |
| ERGO-lstm | 0.4998±0.0022 (#9) | 0.4981±0.0038 (#9) |
| **Ours BioSeq frozen global-mean + MLP** | **0.5462±0.0026 (#7)** | **0.5558±0.0020 (#7)** |

> 分表产物：`comparison_with_ours_seen_test_local_release_AS.{md,csv}` 与 `comparison_with_ours_seen_independent_local_release_AS.{md,csv}`。CSV 中保留 numeric mean/SD/rank、来源和 ERGO-AE official tail-drop 行数注释。复现入口：`tcr_binding/run_retrained_ours.py`。

---

### T1 · IMMREP23 辅基准（历史，与 NM2025 不可直接比较）

**数据集（IRBench 辅基准）**：IMMREP23 paired-chain VDJdb 正样本 + 自生成参考负样本(Lev>3, 5:1) 作训练；
官方 `solutions.csv`(含负样本) 作测试。**防泄露**：训练集中与测试克隆型(CDR3a|CDR3b)重叠的 60 条正样本已剔除，
最终 clonotype/pair overlap = 0（见 `data/tcr_binding/leakage_report.json`）。
测试按表位是否在训练集出现分为 **seen(13 表位/2418 行)** 与 **unseen(7 表位/1066 行)**。
主排名 = **unseen-epitope**（泛化到新表位，最难也最有意义）。

> **主指标口径（IMMREP23 历史口径）**：unseen-epitope macro-AUPRC 为主榜，AUROC/AUC0.1 辅报。

### Unseen-epitope split (主排名，2026-07-02 全量复跑)
| Method | macro-AUPRC | macro-AUROC | macro-AUC0.1 | 类型 | 备注 |
|--------|-------------|-------------|--------------|------|------|
| **epiTCR (官方 RF)** | **0.195** | **0.525** | **0.509** | 官方源码+权重 | 迄今 unseen 最好，仍近随机 |
| Random | 0.192 | 0.473 | 0.504 | floor | |
| ESM2-150M + linear probe | 0.189 | 0.470 | 0.496 | 通用蛋白LM | 均值池化拼接，线性不建模交互 |
| CDR3-kNN (TCRdist式) | 0.167 | 0.500 | 0.500 | 距离/训练free | 无参考结合子→必然随机 |
| **Ours-BioSeq** | _pending_ | | | 扩散免疫基础模型 | 训练完成后填 |

### Seen-epitope split
| Method | macro-AUROC | macro-AUPRC | macro-AUC0.1 | 备注 |
|--------|-------------|-------------|--------------|------|
| CDR3-kNN (TCRdist式) | **0.694** | **0.539** | **0.689** | 有同表位参考结合子 |
| epiTCR (官方 RF) | 0.612 | 0.352 | 0.584 | 官方源码 baseline |
| Random | 0.512 | 0.207 | 0.511 | |
| ESM2-150M + linear probe | 0.489 | 0.191 | 0.494 | 线性 probe 不足以建模 binding |
| **Ours-BioSeq** | _pending_ | | | |

> **关键结论**：CDR3-kNN 在 seen 表位有效(AUROC 0.69)，但在 unseen 表位**恰好随机(0.50)**——
> 这正是泄露受控的 unseen-split 的价值：它如实暴露了距离类方法无法泛化到新表位。
> 通用 ESM2 均值嵌入 + 线性 probe 在 binding 上不足（接近随机），说明 binding 需要建模 TCR×表位交互
> 或专门预训练——这是免疫受体基础模型的动机。
>
> 复现：
> ```bash
> python scripts/prepare_tcr_binding.py --neg-ratio 5 --seed 0
> python tcr_binding/run.py --method random
> python tcr_binding/run.py --method knn --knn-k 5
> python tcr_binding/run.py --method embed --embedder esm2_150m
> ```

### 外部参考：IMMREP22 官方排行榜（不同数据集，仅作定位）
> 来源：`baselines/IMMREP_2022_TCRSpecificity/methods_results/`（17 个表位，指标 MicroAUC / Average Rank）。
> **注意**：这是 IMMREP22 数据集上的官方结果，**与上方 IRBench-T1(IMMREP23) 不可直接比较**，仅用于展示传统方法量级。

| Method (官方) | avg MicroAUC | avg Rank |
|---------------|--------------|----------|
| tcrexab | 0.847 | 3.28 |
| TCRGP | 0.847 | 3.49 |
| tcrdist3 | 0.840 | 3.58 |
| netTCR_cdr123ab | 0.825 | 3.82 |
| TCRAI | 0.823 | — |
| sonia_paired | 0.806 | 4.54 |
| pMTnet | 0.774 | — |
| TCR-BERT | 0.754 | — |
| TITAN | 0.746 | 9.12 |
| SETE | 0.739 | — |
| random | 0.490 | 9.12 |

> 完整 22 行见 `outputs/external/immrep22_official.csv`（由 `scripts/import_immrep22_results.py` 生成）。

---

## T2 — TCR Clustering（双基础：NAR-GAB 2025 主 + Brief Bioinformatics 2025 第二基础）

> **两个构建基础**：**基础A = NAR-GAB 2025**（配对 TCR + 官方 9 方法 + Purity–Retention–Sensitivity，见 (a)(b)(c)）；**基础B = Brief Bioinformatics 2025 / TCREmbedding**（9,033 唯一 CDR3β/25 表位 + embedding→clustering 的 ARI/NMI/Purity，见 (d)）。基础A 提供"官方方法级 baseline"权威来源；基础B 提供 Ours-BioSeq 最对口的"嵌入→聚类"同口径 + 第二独立数据集 + ARI/NMI 完整指标。两基础**分区呈现、互不混表**（配对口径 vs 单链 CDR3β 口径）。

> **数据（基础A，主）**：直接复用 NAR Genomics & Bioinformatics 2025《Benchmarking unsupervised methods for inferring TCR specificity》开源仓库 [`i3-unit/TCR_Unsupervised_Benchmark`](https://github.com/i3-unit/TCR_Unsupervised_Benchmark)（commit `ac767882`）的 curated pooled DB（IEDB+McPAS+VDJdb），照其 `database_processing.R` 复算 CD8/VS=2/AIS>4.3/配对唯一口径，得 **4,779 唯一配对 TCR / 8,395 唯一单链序列 / 4,103 唯一 CDR3α / 4,292 唯一 CDR3β / 366 表位**（`data/tcr_clustering/{tcrs.csv,meta.json}`）。
>
> ✅ **universe 已精确复现论文（2026-08-30 修复）**。论文的四个可核对常量（4,779 配对 / 8,395 序列 / 4,103 α / 4,292 β）现已**逐一精确命中**，6 个高亮表位的配对/链计数也照旧逐一一致（GIL 706/484/456…）。
>
> 🚨 **此前报的 5,368 配对 / 9,370 序列 / 374 表位是错的，根因已定位并修复。** 旧叙事把 +589 的差异归给「仓库 README 警告的数据库版本漂移」——**这个归因是错的**。实际是 `curate()` 里的一处 pandas/dplyr 语义差异：`dplyr::filter` 会丢掉判据为 `NA` 的行，所以 R 原文的 `filter(PubMed_ID != <10x url>)` 连带删掉了 `PubMed_ID` 缺失的行；pandas 的 `!=` 对 `NaN` 返回 `True`，把它们全留下了。那是 **634 行**，恰好贡献了多出来的 589 个配对，且两侧配对集合**完全嵌套**（pandas 多 589、R 多 0、共享 4,779）。只修这一处即精确复现论文全部常量。逐步骤证据：`scripts/t2_curate_step_align.py` → `outputs/tcr_clustering/curate_step_align.json`。
>
> 顺带说明 4,779 / 8,395 的性质：它们并不是论文分析里现算的，而是 `benchmark_tools_pooled_database_CD8.Rmd` 第 217–226 行**硬编码的分母常量**（注释写明 "divided by the number of unique pairs / sequences"）。所以「能否复现这两个数」本身就是对 curation 链是否忠实的一次强检验，而我们此前一直没通过。
> **评测单元**：唯一**配对 TCR**（`pair_id`）。paired 方法（clusTCR/GLIPH2/DeepTCR/TCRdist3）用其配对簇；chain-separate 方法（HD/LD/TCRMatch/iSMART/GIANA）用 **β 链簇映射到配对**（同一 β 归同簇）。**防泄露**：表位标签只用于评测，不参与聚类。
> **指标**（`common/metrics.py::clustering_metrics`，NAR-GAB 口径）：Purity / Retention / Sensitivity / %clusters purity>0.9 / %seqs in high-purity clusters（+ NMI/ARI + cluster-size>3/5/10 复算）。
>
> **口径归属（重要）**：论文原生 Retention/Purity `[P]` 是主 baseline。七种方法有作者 assignment artifact；HD/LD 是本地距离图重建。九种方法被投影到**已与论文一致的 4,779-pair universe**（修复前是错误的 5,368-pair universe，那时统一 pair 表只能算 `mismatched` diagnostic）；(b) 区仍是 `[A]/[L]`，因为投影与 HD/LD 重建不是论文原生产物，但 universe 本身不再是偏离项。
> - **我们的模型 / 参照嵌入（(c) 区）**：ESM2/k-mer/TCR-VALID/**Ours-BioSeq** 无"官方聚类输出"可用（论文本身也是对 ESM/tcrBERT 做 embedding+聚类），故用**我们统一评测框架**的阈值式聚类壳（`embed-threshold`）。这些是"我们的模型/参照嵌入"，**与官方 9 方法分区呈现、不混为一栏**。
> - HD/LD 明确标作 `local_reimplementation`；2026-07-14 修复了 LD indel 分支曾静默退化成 HD 的错误，旧 LD 数值不再引用。

### (a) 论文主 baseline `[P]` + 本地校准 diagnostic（`outputs/tcr_clustering/calibration_report.csv`）

| 方法 | 复算 Ret/Pur | 论文 Ret/Pur | 一致 |
|------|-------------|-------------|------|
| clusTCR | 0.089 / 0.993 | 0.09 / 0.99 | ✓ |
| DeepTCR | 0.922 / 0.632 | 0.92 / 0.63 | ✓ |
| TCRMatch | 0.089 / 0.824 | 0.09 / 0.82 | ✓ |
| GLIPH2 | 0.224 / 0.798 | 0.22 / 0.80 | ✓ |
| TCRdist3 | 0.438 / 0.422 | 0.44 / 0.42 | ✓ |
| GIANA | 0.253 / 0.649 | 0.25 / 0.65 | ✓ |
| iSMART | 0.183 / 0.748 | 0.18 / 0.75 | ✓ |
| HD | 0.268 / 0.694 | 0.27 / 0.69 | ✓（ret −0.002，pur +0.004） |
| LD | 0.305 / 0.573 | 0.31 / 0.57 | ✓（ret −0.005，pur +0.003） |

> ✅ **9/9 全部吻合**（2026-08-30），判据是 `|Δret| ≤ 0.03 且 |Δpur| ≤ 0.03`（`import_clustering_baselines.py` 的 `match` 列），不是「吻合到两位小数」。实际精度远高于判据：最大 `|Δret| = 0.005`、最大 `|Δpur| = 0.004`。覆盖 purity 0.42–0.99、retention 0.09–0.92 全范围 → 指标实现与输出解析正确。
>
> **HD / LD 从「不通过」翻到通过，与 +589 是同一个根因。** 这两个恰好是官方仓库里**唯一没有预计算输出文件**的方法（论文用 stringdist 现算），我们同样现算，跑在本地 curate 出的数据库上；此前 curate 因为 `PubMed_ID` 缺失行未被剔除而多出 975 条唯一序列，分母又沿用论文硬编码的 8,395，于是 retention 系统性偏高 +0.033 / +0.037，刚好卡在 0.03 判据之外。修掉那一行后 retention 偏差变为 **−0.002 / −0.005**。所以旧文里「HD/LD 因数据库多出序列而无法吻合」这个说法不再适用——那不是数据库的问题，是我们的过滤实现的问题。
>
> **2026-08-30 修正了论文参考列。** 此前 HD / LD / iSMART / GIANA 四行的论文值都写成 `0.25 / 0.66`，那是论文正文对这四个方法**联合**给出的组统计（"similar low retention (0.25 ± 0.06) and intermediate purity scores (0.66 ± 0.09)"），不是逐方法值；等于九行里有四行的参考值是同一个组均值。现改为 Fig 3A 柱上标注的逐方法值。连带两处结论翻转：
> - **iSMART 由「不一致」变为精确吻合**（0.183/0.748 vs 0.18/0.75）。旧注「仓库输出文件为较小一次运行」**予以撤回**——补充表 S2 明确记载 iSMART 聚类了 1,538 条唯一序列，1538/8395 = 0.183，正是我们复算出的值，仓库文件是完整的。
> - **LD 的 purity 偏差由 −0.097 收窄到 −0.007**（缩小 14 倍）。
>
> **披露论文自身的内部矛盾**：Fig 3A 标 HD retention 0.27、LD 0.31，而补充表 S2 的计数除以同一分母给出 HD 2564/8395 = 0.31、LD 2248/8395 = 0.27，两者在 HD/LD 上互换；其余七个方法两处逐位一致。此处采用 Fig 3A（正文主结果图，且其顺序符合 LD 容许 indel、应比 HD 聚进更多序列的算法学预期）。若改采 S2 口径，HD/LD 的 retention 偏差会分别变成 −0.007 与 +0.077。

### (b) 统一 pair-universe 表（N=4,779，与论文同一 universe）

七种为作者 assignment artifact 投影；HD/LD 为本地重建。**2026-08-30 已在修复后的 4,779-pair universe 上全部重算**（旧表是 N=5,368，已作废）：

| Method（官方输出来源） | Purity | Retention | Sens(top6) | %clu pur>0.9 | 聚类配对数 |
|--------|--------|-----------|-----------|--------------|:---:|
| clusTCR | **0.991** | 0.089 | 0.135 | 0.964 | 426 |
| TCRMatch | 0.968 | 0.118 | 0.127 | 0.866 | 564 |
| GIANA | 0.932 | 0.196 | 0.290 | 0.714 | 939 |
| iSMART | 0.918 | 0.196 | 0.281 | 0.683 | 935 |
| GLIPH2 | 0.904 | 0.209 | 0.277 | 0.638 | 1,001 |
| HD（本地重建） | 0.865 | 0.248 | 0.344 | 0.588 | 1,183 |
| LD（本地重建，indel 已修） | 0.771 | 0.272 | 0.256 | 0.518 | 1,298 |
| TCRdist3 | 0.421 | 0.435 | 0.004 | 0.070 | 2,079 |
| DeepTCR | 0.630 | **0.922** | **0.376** | 0.194 | 4,405 |

> 修复前后的方向：分母由 5,368 降到 4,779，**retention 全部上移**（如 DeepTCR 0.821→0.922、TCRdist3 0.387→0.435），purity 变化很小（最大 LD +0.011）。方法间排序不变。
>
> 与 (a) 区的关系：(a) 是论文原生协议（单链/配对各按论文的固定分母），(b) 是把九个方法统一投影到同一个配对 universe 后重算，因此两表的 retention 数值不同是设计使然，不是矛盾。现在两者跑在同一个 curation 结果上。LD 是本地 Levenshtein≤1 连通分量重建，不称为官方 artifact。
>
> ⚠️ **标签口径偏离（(b)(c) 两区共有，已量化）**：本表与 (c) 区用的 shipped 数据集 `data/tcr_clustering/tcrs.csv` 经 `prepare_tcr_clustering.py::build_pairs` 把**多标签塌缩成多数标签**（一个配对只保留一个表位，366 表位），而论文自己的 `purity_function` 保留一个配对观察到的**全部**表位——它的 `merge` 会把多特异性 CDR3 展开成每表位一行，分子按多标签计数、分母仍是去重后的簇大小（这也是论文 purity 理论上可以超过 1 的原因）。
>
> **偏离规模已实测**（`scripts/t2_label_collapse_effect.py` → `outputs/tcr_clustering/label_collapse_effect.csv`）：4,779 个配对中多特异性的有 **88 个（1.84%）**。tie-break 为**字典序**取多数表位；实测 **0 个**塌缩标签落在该配对自身观察到的表位集合之外。在同一批簇上用两种口径重算 purity：
>
> | 方法 | purity（塌缩，即本表口径） | purity（论文多标签口径） | Δ |
> |---|:---:|:---:|:---:|
> | clusTCR | 0.9906 | 0.9930 | +0.0023 |
> | TCRMatch | 0.9681 | 0.9699 | +0.0018 |
> | HD | 0.8648 | 0.8664 | +0.0017 |
> | LD | 0.7712 | 0.7727 | +0.0015 |
> | GIANA | 0.9318 | 0.9329 | +0.0011 |
> | iSMART | 0.9176 | 0.9187 | +0.0011 |
> | DeepTCR | 0.6304 | 0.6316 | +0.0011 |
> | GLIPH2 | 0.9041 | 0.9051 | +0.0010 |
> | TCRdist3 | 0.4214 | 0.4218 | +0.0005 |
>
> 塌缩口径**一致地略微低估** purity，但幅度上限 **0.0023**，比方法间差距（0.42–0.99）小两个数量级，也小于本表任何一处排序间隔。所以 (b)(c) 两区的排序与结论不受这个偏离影响；但凡把本表数字与论文 `[P]` 值并列时必须声明它是塌缩口径。（(a) 区校准不受影响——`import_clustering_baselines.py` 的校准路径本来就用论文的多标签口径，见 §T2(a)。）该脚本同时用塌缩口径精确复现了本表全部 9 个 purity 值，作为实现一致性检查。

### (c) 我们的模型 / 参照嵌入 —— 统一评测框架的阈值式聚类（**非论文官方方法 baseline**，与 (a)(b) 分区）

> 这些方法无论文聚类输出，使用本地 `embed-threshold` 壳。表中的 retention 坐标仅用于取点，不构成与论文九方法的数值排名。
>
> 🚨 **本表数字已过期（2026-08-30），待重跑。** 它是在错误的 5,368-pair universe 上算的（根因见本节开头）。(a)(b) 两区已在修复后的 4,779-pair universe 上重算，本区**没有**——因为每一行都要重新做 GPU 嵌入（`bioseq:` / ESM2 / TCR-VALID），而本轮审计的范围是 baseline 与协议对齐，Ours 侧数字统一留给 v2 checkpoint。取点用的 retention 锚（`ret≈0.19/0.25/0.39/0.82`）也已随 (b) 表上移（见 (b) 区说明），所以列名里的锚值同样不再对应。**在重跑前本表任何数字都不可引用**，包括其中的相对排序——因为各行 universe 相同、但锚点位置变了，取点位置会整体偏移。
>
> ⚠️ 重跑后本区仍将与 (b) 区共用 shipped 数据集，因此**同样是多数标签塌缩口径**（88/4,779 配对受影响，purity 被一致低估 ≤0.0023）。完整量化见 (b) 区末尾的披露。

| 嵌入方法 | Pur@ret≈0.19 (GLIPH2) | Pur@ret≈0.25 (HD/LD/GIANA) | Pur@ret≈0.39 (TCRdist3) | Pur@ret≈0.82 (DeepTCR) | 曲线AUC |
|----------|:---:|:---:|:---:|:---:|:---:|
| k-mer(3) 组成(训练free) | 0.931 | 0.842 | 0.664 | 0.188 | 0.420 |
| ESM2-150M（通用蛋白 LM） | 0.949 | 0.860 | 0.633 | 0.180 | 0.408 |
| TCR-VALID（我们口径复用其 latent） | **0.983** | **0.953** | **0.815** | 0.263 | **0.521** |
| **Ours-BioSeq** esmc300m-cmp500k (enc-only 消融) | 0.964 | 0.912 | 0.661 | 0.207 | 0.441 |
| **Ours-BioSeq** esmc600m-cmp500k (enc-only 消融) | 0.968 | 0.917 | 0.666 | 0.184 | 0.419 |
| **Ours-BioSeq** esmc300m-mint (enc-only 消融) | 0.959 | 0.874 | 0.602 | 0.176 | 0.390 |
| **Ours post-LLaDA** esmc300m-cmp500k | 0.963 | 0.853 | 0.613 | 0.211 | 0.426 |
| **Ours post-LLaDA** esmc600m-cmp500k | 0.969 | 0.888 | 0.615 | 0.193 | 0.434 |
| **Ours post-LLaDA** esmc300m-mint | 0.936 | 0.824 | 0.572 | 0.184 | 0.392 |

> Ophiuchus-Ab 嵌入按逐序列路径极慢（占位抗体权重），本轮未在时限内产出曲线，如实标注（其余两个通用 LM 参照 esm2_150m/kmer 已产出）。ret≈0.08（clusTCR 的超低保留/超高纯度区）阈值法不可达（这些嵌入的余弦图最低保留约 0.10，仅剩完全重复序列的簇），如实留空。

### (d) 第二基础 —— embedding→clustering 的 ARI/NMI/Purity（Brief Bioinformatics 2025 / TCREmbedding）

> **为何加第二基础**：NAR-GAB 2025（基础A）给的是**配对 TCR + 官方 9 方法 + Purity–Retention 权衡**，但缺 ARI/NMI 完整一致性指标、也只有一个数据集。Brief Bioinformatics 2025《A comprehensive benchmarking for evaluating TCR embeddings in modeling TCR-epitope interactions》（bbaf030, PMID 39883514；官方仓库 [`deepomicslab/TCREmbedding`](https://github.com/deepomicslab/TCREmbedding) commit `d5bf911`）正是**"嵌入→聚类"评测口径**：固定标注数据集（GIANA 项目 curated，**9,033 唯一 CDR3β / 25 抗原表位**，≥110/类，`dataset/clustering/TCRantigenData_unique_test.csv`）→ 标准 sklearn 聚类（**K∈{10,15,…,100} 扫描**取均值以消目标簇数偏置）→ **ARI / NMI / Purity**。这正是 Ours-BioSeq embed 方法最对口的同口径基准，补齐了基础A缺的 ARI/NMI + 第二独立数据集。
> **数据/脚本**：`scripts/prepare_tcrembedding_clustering.py` → `data/tcr_clustering_embed/{tcrs.csv,meta.json}`；`tcr_clustering/run_embed_bench.py`（嵌入 CDR3β → L2 归一 → K-means 主 + 层次(PCA-50 Ward) 辅 → ARI/NMI/Purity）；`scripts/summarize_tcrembedding_clustering.py`。**防泄露**：表位标签仅评测。**GPU 嵌入在火山引擎作业跑**（task_id `t-20260705211729-crvhj`，State Success；ESM2/k-mer 参照本地跑）。
> **口径与官方方法的关系**：基础B 是"我们的模型/参照嵌入"分区（与基础A 官方 9 方法互不混表）。参照 = **ESM2-150M / ProtBERT（通用蛋白 LM，官方 HF 权重）** + **TCR-BERT（TCR 专用 LM，仅 CDR3β）** + **k-mer(2) 组成（训练free）**；论文本身评的 19 个嵌入方法（GIANA/clusTCR/catELMo/理化…）在 Supplementary Table 2，其**定性结论**是"专用/特异性感知嵌入 > 数据驱动，通用 ESM 在 CDR3β 短序列上不占优"——我们的数字复现了这一趋势（ESM2/ProtBERT 垫底，TCR-BERT 介于 k-mer 与 Ours 之间），未擅自转录论文未复核的逐方法数值。

| 分区 | 嵌入方法 | ARI (mean/best) | NMI (mean/best) | Purity (mean/best) |
|------|----------|:---:|:---:|:---:|
| 参照 | **SCEPTR**（专用对比学习 TCR 模型，CDR3β+TRBV partial） | **0.033 / 0.063** | **0.159 / 0.186** | **0.339 / 0.370** |
| 参照 | TCR-BERT（TCR 专用 LM，仅 CDR3β） | 0.016 / 0.021 | 0.095 / 0.120 | 0.287 / 0.304 |
| 参照 | k-mer(2) 组成（训练free） | 0.010 / 0.013 | 0.079 / 0.112 | 0.268 / 0.294 |
| 参照 | ESM2-150M（通用蛋白 LM） | 0.007 / 0.008 | 0.068 / 0.093 | 0.258 / 0.279 |
| 参照 | ProtBERT（通用蛋白 LM） | 0.006 / 0.008 | 0.061 / 0.087 | 0.248 / 0.269 |
| **Ours post-LLaDA** | esmc300m-cmp500k（headline） | 0.024 / 0.045 | 0.140 / 0.152 | 0.317 / 0.332 |
| **Ours post-LLaDA** | esmc600m-cmp500k | 0.022 / 0.036 | 0.129 / 0.147 | 0.306 / 0.322 |
| **Ours post-LLaDA** | esmc300m-mint | 0.004 / 0.006 | 0.055 / 0.076 | 0.242 / 0.257 |
| **Ours** | **BioSeq** esmc300m-cmp500k（enc-only 对照） | 0.021 / 0.036 | 0.128 / 0.146 | 0.306 / 0.321 |
| **Ours** | **BioSeq** esmc600m-cmp500k（enc-only） | 0.018 / 0.027 | 0.109 / 0.136 | 0.293 / 0.315 |
| **Ours** | **BioSeq** esmc300m-mint（enc-only） | 0.006 / 0.009 | 0.064 / 0.089 | 0.257 / 0.274 |

> post-LLaDA 行来自 Volc `t-20260705215519-nck2s`（T2 部分成功，`bioseq-llada:` tag `*_postllada`）；**SCEPTR 行本地跑**（`run_embed_bench.py --embedder sceptr`，14.7s，`outputs/tcr_clustering_embed/sceptr/`）。源：`outputs/tcr_clustering_embed/_summary.csv`。

> 表内为 **K-means** mean/best over K；层次(Ward on PCA-50)趋势一致（esmc300m-cmp500k ARI 0.025/NMI 0.138/Purity 0.314），完整曲线见 `outputs/tcr_clustering_embed/<tag>/curve.csv`。**绝对值偏低是单链 CDR3β 表位聚类的固有难度**（论文各方法 ARI 亦普遍在 0.0x 量级）；关键是**同数据同协议下的相对次序**。
>
> **加入 SCEPTR 后的诚实次序（2026-07-08）**：专用对比学习 **SCEPTR 三项均最高（ARI 0.033 / NMI 0.159 / Purity 0.339），超过我们的模型**；我们的 post-LLaDA cmp500k（ARI 0.024 headline）领先所有通用/自监督 PLM（TCR-BERT 0.016 > k-mer 0.010 > ESM2 0.007 > ProtBERT 0.006）但**低于 SCEPTR**——与 T3 及 SCEPTR 论文"对比学习专用模型 > 通用/自监督表征"的结论一致、互相印证。**口径注**：SCEPTR 用 CDR3β+TRBV（其原生 partial 输入口径，α 缺失；比纯 CDR3β 多 V 基因信息），其余参照仅 CDR3β。

### 核心结论
- **口径校验通过**（**9/9** 官方方法逐一复现论文 Purity/Retention，最大 |Δ| 0.005）→ 我们的 NAR-GAB 指标实现可信；数字全部可由 `outputs/tcr_clustering/` 复现。
- **Ours-BioSeq post-LLaDA headline 已接入 T2**（`bioseq-llada:`，Volc `t-20260705215519-nck2s`）：基础A 仍优于 ESM2/k-mer 但**略低于 enc-only 对照**（ret≈0.25：600m post-LLaDA 0.888 vs enc 0.917）；基础B cmp500k post-LLaDA 领先所有通用/自监督 PLM（ARI 0.024 vs ESM2 0.007/k-mer 0.010），但**低于专用对比学习 SCEPTR（0.033，2026-07-08 新增参照）**。
- **两类范式各擅胜场（与论文一致的权衡）**：阈值式嵌入聚类（含 Ours-BioSeq）在**中低保留区**能达到甚至超过 GLIPH2/GIANA 的纯度，并在 TCRdist3 的 retention（0.39）处纯度远超之（0.66 vs 0.42）；但**高保留区**（DeepTCR 的 0.82）仍是 DeepTCR 独有优势（阈值法此时坍缩为巨簇、纯度骤降）。

### 复现命令
```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
cd benchmark
# 1) 数据（需先放置 baselines/TCR_Unsupervised_Benchmark，gitignore）
$ENV/bin/python scripts/prepare_tcr_clustering.py
# 2) 导入 9 官方方法（作者预计算输出）+ 口径校验
$ENV/bin/python scripts/import_clustering_baselines.py
$ENV/bin/python tcr_clustering/run.py --method precomputed --all
# 3) 我们的模型 / 参照嵌入的统一阈值曲线（非官方 baseline）
$ENV/bin/python tcr_clustering/run.py --method embed-threshold --embedder esm2_150m --tag esm2_150m_thr
$ENV/bin/python tcr_clustering/run.py --method embed-threshold --embedder kmer --tag kmer_thr
$ENV/bin/python tcr_clustering/run.py --method embed-threshold --embedder bioseq:/abs/best.pt --tag ours_esmc300m_cmp500k
# 4) TCR-VALID（独立 tcrvalid TF env）：原生协议 + 我们口径
conda run -n tcrvalid python scripts/tcrvalid_embed_cluster.py
$ENV/bin/python tcr_clustering/run.py --method embed-threshold \
    --embedder emb:TCR-VALID:$(pwd)/outputs/tcr_clustering/tcrvalid/latents.npy --tag tcrvalid
# 5) 汇总
$ENV/bin/python scripts/summarize_tcr_clustering.py

# ===== 第二基础 (d)：Brief Bioinformatics 2025 / TCREmbedding embedding→clustering =====
# 6) 数据（需先放置 baselines/TCREmbedding/dataset/clustering/TCRantigenData_unique_test.csv，gitignore）
$ENV/bin/python scripts/prepare_tcrembedding_clustering.py
# 7) 参照嵌入本地跑（轻量）
$ENV/bin/python tcr_clustering/run_embed_bench.py --embedder kmer      --tag kmer      --algos kmeans,hierarchical
$ENV/bin/python tcr_clustering/run_embed_bench.py --embedder esm2_150m --tag esm2_150m --algos kmeans,hierarchical
# 8) Ours-BioSeq 嵌入（GPU）→ 提交火山引擎作业（no-proxy 包装脚本）
/root/.codex/skills/volc-no-proxy/scripts/volc-no-proxy.sh ml_task submit --conf eval_jobs/eval_t2_clustering_embed_bioseq.yml
# 9) 汇总（ours vs 参照）
$ENV/bin/python scripts/summarize_tcrembedding_clustering.py
```

### 补充 baseline：TCR-VALID（Nat Commun 2024, `peterghawkins-regn/tcrvalid`, commit `475ed964`, Apache-2.0）
- **原生协议**（`outputs/tcr_clustering/tcrvalid_native/`）：其 VAE latent 上 DBSCAN(manhattan) 扫 eps + 其自带 `clustering_scoring`（mean_purity/well_clustered/percent_clustered，均为 %）。如 eps=3.0：mean_purity 80.5% / retention 36.8%；eps=0.5：98.4% / 18.1%。
- **本地统一口径 `[L]`**（上表 (c)）：把 TCR-VALID 的 16 维 latent 灌入相同 embed-threshold 后，其 AUC 为 0.521；这只是在该本地阈值曲线中的比较，不作为 NAR-GAB 论文协议下的领先结论。
- **诚实标注的差异**：TCR-VALID 用 **CDR2β-CDR3β_core** 特征（V 基因经 `TRBV_reference.csv` → CDR2，99.5% 配对可映射），与 NAR-GAB 的 **CDR3-only** 聚类口径不同，故列为**补充表征 baseline**、非 NAR-GAB 协议复跑；其原生 mean-purity 也与 NAR-GAB 加权 purity 定义不同。

## T3 — TCR Representation (few-shot per-epitope NN AUROC)

**锚定论文 = SCEPTR (Cell Systems 2024)**。两条互补 track：**deep 6-pMHC（主口径，严格照 Table SI III.1）** + **broad 24-epitope（互补的大表位面视图）**。

### T3-deep — SCEPTR-compatible 本地评测 `[L]`（6-pMHC deep few-shot）

> **协议**：脚本 `tcr_representation/run_paper6.py`，输出 `outputs/tcr_representation_paper6/`。6 target pMHC（GILGFVFTL/NLVPMVATV/SPRWYFYYL/TFEYVSQPFLMDLE/TTDPSFLGRY/YLQPRTFLL），universe=**25,816**（3,941 binder + 21,875 背景 VDJdb 全库负样本），k∈{1,2,5,10,20,50,100,200}、**100 seeds**、min-NN、per-pMHC AUROC → 6-pMHC macro。数据 = era-matched **VDJdb 2023-10**（与论文同期，`prepare_tcr_representation_paper6.py`）。列口径同 broad track：嵌入类 `cdr3b cdr3a`，**TCR-BERT 仅 CDR3β**。
> **提交方式**：baseline 由 `eval_paper6_tcr_representation.yml`（task `t-20260705204639-l56dp`, ExitCode 0，其 encoder-only 伪双链输出已按 PROJ_GUIDE.md 删除）；decoder+占位peptide 中间态由 `eval_paper6_ours_decoder.yml`（task `t-20260705214312-k8ktb`, ExitCode 0）；**我们 3 个模型的最终正确口径(真双链 tcr_pair)**由 `eval_jobs/eval_paper6_ours_tcrpair.yml`（`scripts/run_paper6_ours_tcrpair.sh`，task `t-20260705221018-trjnv`, ExitCode 0）重跑，输出直写共享 vepfs。
>
> **我们模型的表征口径（headline = post-LLaDA globalfeat）**：主表 **Ours** = `GrammarEmbedder` + `grammar:decoder:global:/abs/best.pt`——β+α 渲染为无 peptide 的 `tcr_pair`，干净联合前向 LLaDA 解码器，**整条 record global mean-pool** → 单向量（960 维 @300m）。tag `ours_globalfeat_<run>`；`fewshot.json` 含 `checkpoint` 溯源（`common/provenance.py`）。旧 `grammar_tcrpair_*` / `bioseq_llada_*`（07-05 ckpt）仅历史参照。

#### deep few-shot macro AUROC vs shots（真实数字，`_summary.json`；随机=0.5；"我们的模型"=post-LLaDA globalfeat · Volc `t-20260707172056-fwktl`）
| Method | 类型 | k=1 | k=5 | k=20 | k=100 | k=200 |
|--------|------|-----|-----|------|-------|-------|
| **SCEPTR** | 专用对比学习(官方权重) | **0.640** | **0.702** | **0.738** | **0.775** | **0.790** |
| TCRdist (tcrdist3) | 序列比对法(官方源码) | 0.647 | 0.701 | 0.733 | 0.764 | 0.777 |
| CDR3-Levenshtein-NN | 序列比对法(训练free) | 0.626 | 0.671 | 0.696 | 0.722 | 0.733 |
| k-mer(3) 组成 | 训练free 组成基线 | 0.607 | 0.665 | 0.690 | 0.717 | 0.728 |
| ESM2-150M | 通用蛋白LM(官方权重) | 0.591 | 0.631 | 0.656 | 0.685 | 0.696 |
| ProtBERT | 通用蛋白LM(官方权重) | 0.585 | 0.624 | 0.648 | 0.681 | 0.694 |
| TCR-BERT | TCR专用LM(官方权重,仅CDR3β) | 0.579 | 0.631 | 0.656 | 0.681 | 0.692 |
| **Ours** esmc300m-cmp500k | post-LLaDA globalfeat | 0.594 | 0.646 | 0.678 | 0.707 | **0.719** |
| **Ours** esmc600m-cmp500k | post-LLaDA globalfeat | 0.585 | 0.626 | 0.654 | 0.689 | 0.703 |
| **Ours** esmc300m-mint | post-LLaDA globalfeat | 0.557 | 0.585 | 0.606 | 0.633 | 0.645 |

#### ★ 输入字段对照（2026-08-30 逐 baseline 核对源码）

这张表此前不存在，而它改变上表的读法：**各方法拿到的输入字段并不相同。**

| Method | 输入字段 | 字段数 | k=200 |
|---|---|---:|---:|
| **SCEPTR** | CDR3β + CDR3α + **TRBV + TRAV** | 4 | **0.790** |
| TCRdist (tcrdist3) | CDR3β + CDR3α + **TRBV + TRAV + TRBJ + TRAJ** | **6** | 0.777 |
| CDR3-Levenshtein-NN | CDR3β + CDR3α | 2 | 0.733 |
| k-mer(3) 组成 | CDR3β + CDR3α | 2 | 0.728 |
| **Ours** esmc300m-cmp500k | CDR3β + CDR3α | 2 | **0.719** |
| ESM2-150M | CDR3β + CDR3α | 2 | 0.696 |
| ProtBERT | CDR3β + CDR3α | 2 | 0.694 |
| TCR-BERT | CDR3β | 1 | 0.692 |

> **唯一两个超过我们的方法（SCEPTR 0.790、TCRdist 0.777）都拿到了我们没有的胚系基因**：SCEPTR 多 TRBV/TRAV，TCRdist 还额外多 TRBJ/TRAJ。数据里这些字段近 100% 有值（`va`/`vb` 100%、`ja` 97.6%、`jb` 99.7%），不是空占位。核对来源：`common/model_api.py::ScepterSource._to_sceptr_df`（构造 `TRAV/CDR3A/TRBV/CDR3B` 四列）与 `TcrdistSource`（构造 `cdr3_a_aa/v_a_gene/j_a_gene/cdr3_b_aa/v_b_gene/j_b_gene` 六列）。
>
> **所以 §0.3 的结论要按输入档位分层重述**：在**同为 CDR3β+α（2 字段）**的这一档里，我们 0.719 高于三个通用/专用 PLM（ESM2 0.696 / ProtBERT 0.694 / TCR-BERT 0.692，后者只有 1 字段），但**仍低于两个训练-free 的序列比对基线**（CDR3-Levenshtein 0.733、k-mer 组成 0.728）。**「低于 SCEPTR/TCRdist 约 0.06–0.07」这句话不能直接当作模型能力差距**——它混入了胚系信息的贡献；反过来也不能拿它当借口，因为同档位里两个零训练基线仍然赢我们。
>
> ⚠️ **产物字段曾误导，已修**：`--columns` 只作用于逐列 embed / 编辑距离两条路径；`method=sceptr` 与 `method=tcrdist` 走 native cdist，直接从 dataframe 读 `va/vb/ja/jb`，与 `--columns` 无关。但 `run_paper6.py` 原先无条件把 `args.columns` 写进 `fewshot.json`，于是这两个方法的产物都记着 `columns=['cdr3b','cdr3a']`，会被读成"等输入比较"。已新增 `input_fields` 字段记录真实消耗的字段，并回填 46 个已有产物（SCEPTR/TCRdist 两行带 `input_fields_note` 说明）。
>
> **我们这一侧无表位泄露，已逐行核实**：`grammar_pair_embed` 传 `peptide=None`，`embed_pairs` 把它变成空串，且 `order=("beta","alpha")` 里根本没有 `peptide` 段——表位是**完全不进渲染记录**，不是"常量占位"。V/J 基因我们从未喂入。（`fewshot.py` 与 `run_paper6.py` 的两处 docstring 原写"constant-placeholder peptide"且称"逐链段池化后只保留 β(+α) 段"，与实现不符：实际是整条记录全局 mean-pool（`pool_mode="global"`）。两处已改正。）

> ckpt：300m-cmp500k `best.pt @2026-07-07 02:08`（sha256 `abb85171c7b7…`，稳定）；600m `@2026-07-06 06:55`（`c6cc944bf223…`，稳定）；mint 快照 `best.t3snap_20260707T110558Z.pt`（源 `best.pt @2026-07-07 09:24`，sha256 `900176b10f94…`）。源：`outputs/tcr_representation_paper6/ours_globalfeat_*/fewshot.json`（mint 本地重刷 @2026-07-07 12:25 UTC）。
>
> ⚠️ **mint = 当前最新 ckpt 快照**：训练仍 Running（Volc `t-20260707105105-9x4hq`），T3 数字（deep 0.645 / broad 0.595）对应该时刻冻结快照，不必等训练结束；`best.pt` 更新后 `FORCE=1 bash scripts/downstream/rerun_t3_mint.sh` 可重刷。cmp500k ckpt 已固定。

#### 旧 ckpt 对照（resume 前，仅历史参照）
| Checkpoint | 旧 deep k=200 | 新 deep k=200 | 旧 broad k=100 | 新 broad k=100 |
|------------|---------------|---------------|----------------|----------------|
| esmc300m-cmp500k | 0.710 (`grammar_tcrpair_seq_*`) | **0.719** | 0.657 (`bioseq_llada_*`) | **0.662** |
| esmc600m-cmp500k | 0.718 | **0.703** | 0.643 | **0.613** |
| esmc300m-mint | 0.645 | 0.645 | 0.613 | **0.595** |

#### 口径校验：本地 baseline vs SCEPTR Table SI `[P]`（k=200 per-pMHC，Δ=local−paper）
| 复现 baseline | Δ 范围 | 平均 \|Δ\| | 判定 |
|------|------|------|------|
| CDR3-Levenshtein | −0.016 … +0.006 | ≈0.006 | ✓ 高度吻合 |
| TCRdist | −0.013 … +0.003 | ≈0.007 | ✓ 高度吻合 |
| SCEPTR | −0.017 … −0.000 | ≈0.008 | ✓ 高度吻合 |
| ProtBERT | −0.029 … +0.010 | ≈0.014 | ✓ 吻合 |
| ESM2-150M | −0.052 … +0.009 | ≈0.024 | ○ 基本吻合 |
| TCR-BERT | −0.113 … +0.042 | ≈0.057 | △ 排序一致，逐点更宽松 |

> **5/6 baseline 的本地偏差较小**（逐 pMHC 平均约 0.006–0.024）；TCR-BERT 偏差明显更大。该结果说明 harness 大体兼容，但主表仍标 `[L]`，不会把重跑值冒充论文原值。
>
> ✅ **上表已独立复算并逐位吻合（2026-08-30）**：用登记表里的 36 个论文值对上 `outputs/tcr_representation_paper6/*/fewshot.json` 的 `per_epitope_by_shot["200"]`，六行的 Δ 范围与平均 |Δ| 全部重现（0.0058 / 0.0072 / 0.0079 / 0.0143 / 0.0237 / 0.0571）。
>
> ⚠️ **但有个此前未披露的模式：偏差方向系统性偏负。** 36 个 Δ 里只有 5 个为正，SCEPTR 六个 pMHC **全部 ≤ 0**（−0.017 … −0.000），TCRdist 5/6 为负，ESM2 5/6 为负。若只是随机噪声，正负应大致各半。这指向一个**共享的协议差异**（候选：我们的 universe 为 25,816 且 `max_ref_pool=2000` 对每个 target 的候选参考池设了上限；seeds=100；背景池构成），而不是逐方法的实现错误——因为它同向作用于全部六个方法。**结论方向不受影响**（各方法排序与论文一致），但「高度吻合」应理解为「一致偏低约 0.006–0.024 的系统偏移」，不是无偏复现。逐 pMHC 明细：SCEPTR GIL −0.012 / NLV −0.012 / SPR −0.006 / TFE −0.000 / TTD −0.017 / YLQ −0.000。
>
> **deep track 核心结论（2026-08-30 按输入档位重述）**：原排序 **SCEPTR ≥ TCRdist/CDR3-Lev ≥ ESM2/ProtBERT/TCR-BERT** 在数值上没错（k=200：SCEPTR 0.790 ≳ TCRdist 0.777 > Lev 0.733 > 三个 PLM ~0.69），但**它跨越了两个不同的输入档位**：前两名多拿了 V（SCEPTR）或 V+J（TCRdist）胚系基因，其余都只有 CDR3。
> **Ours globalfeat**：300m-cmp500k k=200 **0.719**、600m **0.703**（resume 后略低于旧 ckpt 0.718）、mint **0.645**。在同输入（CDR3β+α）档位内，300m 超三个 PLM 约 0.03，但**低于 Levenshtein 0.733 与 k-mer 0.728 这两个零训练基线**；与 TCRdist/SCEPTR 的 0.06–0.07 差距不是纯能力差距，含胚系信息贡献。

复现：
```bash
cd benchmark
$ENV/bin/python tcr_representation/run_paper6.py --method sceptr        # 或 tcrdist / levenshtein
$ENV/bin/python tcr_representation/run_paper6.py --method embed --embedder tcrbert --columns cdr3b   # TCR-BERT 官方口径仅 CDR3β
# 我们的模型 —— headline：post-LLaDA globalfeat
$ENV/bin/python tcr_representation/run_paper6.py --method embed \
  --embedder grammar:decoder:global:/abs/best.pt --columns cdr3b cdr3a \
  --tag ours_globalfeat_<run>
# Volc 批量：eval_jobs/eval_t3_representation_globalfeat.yml
$ENV/bin/python scripts/summarize_tcr_representation_paper6.py          # 汇总 + Table SI k=200 逐 pMHC Δ
bash scripts/run_paper6_all.sh              # 仅 baseline（tcrbert/kmer）+ 汇总；encoder-only ours 已移除
bash scripts/run_paper6_ours_decoder.sh     # 3×decoder+占位peptide 中间态（t-20260705214312-k8ktb）
bash scripts/run_paper6_ours_tcrpair.sh     # 3×真双链 tcr_pair 最终口径 + 汇总（t-20260705221018-trjnv）
```

### T3-broad — 24-epitope few-shot（互补视图，更大表位面）

> **主协议 = SCEPTR (Cell Systems 2024) few-shot per-epitope NN AUROC**：对每个目标表位、每个 shots k，从 train 池随机抽 k 个该表位参考 binder（R=5 seed），把整个 test 集按到支持集的最近邻距离打分（min-NN 取负），算该表位 AUROC，对表位内 seed 取均值、再对 24 表位 macro 平均（mean±std）。距离后端：native cdist（SCEPTR / TCRdist 真实序列/距离矩阵）、embedding cosine（ESM2 / ProtBERT / TCR-BERT / kmer / Ophiuchus / Ours-BioSeq）、editdist（CDR3β+α Levenshtein）。ProtBERT/ESM2 编码 CDR3β+CDR3α（与 SCEPTR Table SI 通用 PLM 口径一致）；TCR-BERT 按官方用法仅编码 CDR3β（`--columns cdr3b`，其上下文窗口 64、专为 CDR3 训练）。
> **数据**：与 T2 共享 clonotype 隔离 split（train=6887 / test=1722 / 24 表位，overlap=0）。k∈{1,2,5,10,20,50,100}；k≥50 时部分表位 train binder 不足被跳过（k=50 剩 20 表位、k=100 剩 15 表位，如实标注 n_epitopes）。
> **引擎自检（oracle）**：支持集含 query 正样本 → mean AUROC=1.0000 / min=0.9997（`python -m common.fewshot`，口径正确）。

#### few-shot AUROC vs shots（主表，macro mean over epitopes；随机=0.5）
| Method | 类型 | k=1 | k=5 | k=20 | k=100 |
|--------|------|-----|-----|------|-------|
| **SCEPTR** | 专用对比学习(官方权重) | **0.587** | **0.655** | **0.711** | **0.741** |
| TCRdist (tcrdist3) | 序列比对法(官方源码) | 0.577 | 0.644 | 0.687 | 0.728 |
| CDR3-Levenshtein-NN | 序列比对法(训练free) | 0.555 | 0.597 | 0.641 | 0.683 |
| k-mer(3) 组成 | 训练free | 0.543 | 0.597 | 0.643 | 0.673 |
| ESM2-150M | 通用蛋白LM | 0.561 | 0.589 | 0.638 | 0.671 |
| ProtBERT | 通用蛋白LM(官方权重) | 0.550 | 0.587 | 0.627 | 0.651 |
| TCR-BERT | TCR专用LM(官方权重) | 0.547 | 0.577 | 0.623 | 0.641 |
| Ophiuchus-Ab | 抗体权重 | 0.576 | 0.612 | 0.664 | 0.677 |
| **Ours** esmc300m-cmp500k | post-LLaDA globalfeat | 0.569 | 0.609 | 0.648 | **0.662** |
| **Ours** esmc600m-cmp500k | post-LLaDA globalfeat | 0.556 | 0.580 | 0.616 | 0.613 |
| **Ours** esmc300m-mint | post-LLaDA globalfeat | 0.541 | 0.555 | 0.593 | 0.595 |

> Ours 三行 = **post-LLaDA headline**（`grammar:decoder:global:`，tag `ours_globalfeat_*`；cmp500k/600m Volc `t-20260707172056-fwktl`，mint 本地快照重刷 @2026-07-07 12:25 UTC；probe AUROC 300m **0.793** / 600m **0.796** / mint **0.811**）。旧 `bioseq_llada_*`（07-05 ckpt）仅历史参照。

> 全部 7 个 shots（k=1/2/5/10/20/50/100）见 `outputs/tcr_representation/_summary.json` 与各 `<tag>/fewshot.json`。
>
> **本地结论 `[L]`（与 SCEPTR 论文方向一致，但不是论文数值复现）**：
> 1. **SCEPTR ≥ 序列比对法（TCRdist / Levenshtein）**：每个 shots 上 SCEPTR 均居首（如 k=20：0.711 vs TCRdist 0.687 vs Lev 0.641），符合 SCEPTR 论文"专用对比学习反超"结论。
> 2. **通用/专用 LM（ProtBERT / TCR-BERT / ESM2）≤ CDR3 Levenshtein，低于 TCRdist/SCEPTR**；这只描述本地 broad split 的排序。
> 3. **我们的模型（Ours-BioSeq, post-LLaDA）**：esmc300m-cmp500k k=20 达 0.641（≈ Lev 0.641，高于 ESM2 0.638 / ProtBERT 0.627 / TCR-BERT 0.623），但仍低于 TCRdist/SCEPTR——与"通用/自监督 PLM 表征打不过专门为 TCR 特异性设计的对比/距离法"一致；扩散预训练过 LLaDA 解码器的 mean-pool 表征尚未针对表位特异性优化。
>
> **口径对照 SCEPTR Table SI `[P]`**（论文 k=200、6 表位均值，仅供量级参照，split/表位数不同）：ProtBERT≈0.705 / TCR-BERT≈0.735 / ESM2(8M)≈0.717 / CDR3-Lev≈0.737 / TCRdist≈0.783 / SCEPTR≈0.798。不得与 24 表位本地 k=100 数值直接排名。
>
> 复现：
> ```bash
> python -m common.fewshot                                   # oracle 自检
> python tcr_representation/run.py --method levenshtein       # 零依赖主对照
> python tcr_representation/run.py --method tcrdist           # tcrdist3
> python tcr_representation/run.py --method sceptr            # SCEPTR
> python tcr_representation/run.py --method embed --embedder protbert                    # ProtBERT (Rostlab/prot_bert)
> python tcr_representation/run.py --method embed --embedder tcrbert --columns cdr3b     # TCR-BERT (wukevin/tcr-bert-mlm-only)
> python tcr_representation/run.py --method embed --embedder esm2_150m   # 或 kmer / ophiuchus
> python tcr_representation/run.py --method embed --embedder grammar:decoder:global:/abs/best.pt --tag ours_globalfeat_<run>
> python scripts/summarize_tcr_representation.py             # 汇总主表
> ```

### 辅报：24-way linear probe / 1-NN（向后兼容，非主榜）
> 冻结主干 + 线性 probe（逻辑回归）/ 1-NN，24-way 表位分类（随机 acc≈0.04）。距离法（Lev/TCRdist）无 probe 特征，故不列。

| Method | probe-AUROC(macro-OVR) | probe-Acc | kNN top-1 |
|--------|------------------------|-----------|-----------|
| ESM2-150M | 0.806 | 0.438 | 0.409 |
| ProtBERT | 0.793 | 0.405 | 0.393 |
| TCR-BERT | 0.748 | 0.416 | 0.431 |
| Ophiuchus-Ab | 0.825 | 0.452 | 0.443 |
| k-mer(3) 组成 | 0.800 | 0.430 | 0.447 |
| SCEPTR (64-d) | 0.803 | 0.425 | **0.506** |
| **Ours-BioSeq** esmc300m-cmp500k | **0.825** | 0.451 | 0.438 |
| **Ours-BioSeq** esmc600m-cmp500k | **0.825** | **0.468** | 0.432 |
| **Ours-BioSeq** esmc300m-mint | 0.811 | 0.433 | 0.411 |

> 辅报里各法 probe-AUROC 量级相近（0.80–0.825）：24-way probe 有充足监督，能从任意合理嵌入里线性读出表位特异性，区分度弱于 few-shot（这正是补 few-shot 主协议的原因）。SCEPTR 的 kNN-top1（0.506）显著领先，与其表征更"表位聚类"一致。
> **BioSeq 接入说明**：`grammar_v2_*_llada` 为 ESMC-300M/600M 扩散(LLaDA)微调 checkpoint，接入时把微调后 `encoder.esmc.*` 权重载入独立 ESMC 骨架（**0 missing / 0 unexpected 键**，非随机初值），mean-pool 末层表征。见 `common/model_api.py::EsmcBioSeqEmbedder`。

## T4 — TCR Generation（TCRT5 锚定 · 三 setting）

> **2026-07-05 重构**：主口径改为 **TCRT5 (Nat Mach Intell 2025)** 三 setting（A 无条件 / B 表位条件 / C 全长 α/β）。统一 harness = `tcr_generation_bench/run.py`；指标 = `common/metrics.py` 的 `tcrt5_*` 套件（与官方 `src/evaluation.py` 逐函数一致，**42 block 交叉验证 0 mismatch**）。完整设计见 **`TCR_GENERATION_BENCHMARK.md`**。
> 旧 IMMREP23/Markov/PWM/CondPool 口径（`tcr_generation/run.py` + `tcr_design/run.py`）保留为辅视图，本表不再作为主榜。
> **论文 baseline 主榜 = sparse-13 `[A]`**。`RVRAYTYSK/HLA-A*03:01` 是论文 simulation 保留项；旧 benchmark14 只作 `[14-pMHC control]`。**held20 是 TCRT5 的 validation split**（manifest 记载 `source = topk_val_df.csv`），不是测试集，且对 BioSeq 有 ~48% 训练 pair 泄露，仅作 seen control。**跨模型 unseen 对照请用 `bioseq_unseen_common`（固定 6 个 pMHC）**，`bioseq_unseen` 一栏的成员随 run 覆盖范围变化、不可比。
>
> ⛔ **引用本节我们自己的行之前，先读 §0.0(h) 与 `audit_2026_08_29/T4_infill_diagnostic.md`。**
> 2026-08-31 的 infilling 诊断表明：**把表位完全去掉**、只让模型从两侧残基补回真实 held-out CDR3β 的
> 中间一段，我们比「不看上下文、只按 (长度, 位置) 查训练边际」的 PWM 低 **4.15–12.47 pp**，
> **11 组条件全部 McNemar 显著**。该结论在零重叠 holdout、在训练占 37.2% 的 `tcr_pair` layout、
> 在**模型训练时见过的 616 行**（−7.47 / −11.31 pp）、以及在 **8B** 上都成立。因此本节
> 「我们与 OLGA 仅差 0.046」**不能**解读为「学到了一个不错的无条件先验」——它是 TCR 侧
> **拟合失败**在条件任务上的投影。
> 另：把 Setting B 切到训练中占比高 102 倍的 `tcr_pmhc` layout（`--with-mhc`）**六个视图一致变差**。

### Setting A — 无条件 CDR3β repertoire（n=5000）

> **2026-08-30 已换用修复版参考并重算**（`rescore_tcr_generation_offline.py --setting-a`，从 `samples.txt` 离线重算，无模型推理）。参考改为 `data/tcr_generation_fullref`：holdout **7,874** 条（旧版 9,767 条里有 **1,893 条（19.38%）实际存在于全量 OTS 训练语料**，只是旧 holdout 仅对截断后的 10.53% 子集去过重，所以「看起来干净」；新 holdout 正是旧版精确减去这 1,893 条，零误差）。novelty 参照改为**全量 1,718,935 条精确比对**（此前先被数据层截断到 10.53%、再被 `train_ref_cap=20000` 抽样，有效参照只有 1.16%）。NN 距离仍用 20,000 抽样（O(n_gen × n_train) 编辑距离，全量不可行），已在产物中记录 `n_nn_reference` 使近似性可见。

| Method | 类型 | JSD↓ | novelty | NN dist | pgen+ | med log10 pgen | unique |
|--------|------|------|---------|---------|-------|----------------|--------|
| **soNNia/SONIA** | 官方 post-select | **0.0483** | 0.806 | 2.36 | 1.00 | -8.84 | 4987/5000 |
| **OLGA** human_T_beta | 官方 VDJ 模型 | 0.0619 | 0.907 | 3.07 | 1.00 | -9.89 | 4998/5000 |
| **Ours-BioSeq** 最优 ckpt（step41000_retro） | 我们(扩散) | 0.265 | **1.000** | 5.30 | 0.662 | -18.81 | 5000/5000 |
| Ours-BioSeq 12 个 ckpt 波动带 | 我们(扩散) | 0.265–0.528 | 0.9996–1.000 | 4.92–6.64 | 0.66–1.00 | -21.3…-16.3 | 4848–5000 |

> **novelty 的修正只打到官方基线，没打到我们**：soNNia 0.992 → **0.806**（高报 0.186）、OLGA 0.996 → **0.907**（高报 0.088），我们各 ckpt 仍是 0.9996–1.0000。截断参照让「已被复现的真实序列」被误判为新颖。
>
> **但这不是我们的优势。** 高 novelty 在无条件生成里本身不是优点：soNNia / OLGA 的 JSD 只有 0.048 / 0.062，即贴合真实库分布，它们「不够新」恰恰因为在复现真实存在的序列；我们 novelty≈1.0 而 JSD 0.265–0.528，是**因为偏离分布才显得新**。修正后这个对比反而更锐利。pgen 校准同向：官方 pgen+ = 1.00 / 中位 log10 pgen ≈ −9，我们 0.66–1.00 / −16 ~ −21，即大量序列的 V(D)J 重组概率极低——符合扩散模型未对重组统计建模的预期。
>
> ⚠️ **`setting_A/ours_bioseq` 一行已作废，不得引用。** 该目录的 `samples.txt` 与 `..._step41000_retro/samples.txt` **md5 完全相同**（`272baf9b80ae…`），即其原始生成物已被覆盖丢失；而 `metrics.json` 原先仍保留着原始样本算出的 JSD 0.295187 / NN 6.2738。用存盘时所用的截断参考重算，**14 个 run 里 13 个的 JSD 与存盘值逐位相同**（指标实现与参考加载均正确），唯独这一个不符，且其重算值恰好等于 step41000_retro。该行现已写入 `provenance_defect` 字段；因 pgen 沿用原始块而分布指标已改为 step41000_retro 样本，它是**混合出处**，整行不可用。旧表里那个 JSD 0.295 / NN 6.27 就来自这一行。

### Setting B — 表位条件 CDR3β 设计 ★主榜（K=100）

> 数据：论文 sparse-13 + 单独的 reserved-simulation control；held20 为 target-rich 补充。

**论文 sparse-13（官方 stored predictions，主 baseline）**

| Method | 类型 | F1@100 | prec | recall | d_edit↓ | seq-rec | Char-BLEU | div |
|--------|------|--------|------|--------|---------|---------|-----------|-----|
| **TCRT5** | [A] paper sparse-13 | 0.0003 | 0.0002 | 0.0075 | 4.467 | 0.602 | 0.701 | 1.000 |
| **GRATCR** | [A] paper sparse-13 | 0.0000 | 0.0000 | 0.0000 | 4.397 | 0.595 | 0.677 | 0.081 |
| **ER-Transformer** | [A] paper sparse-13 | 0.0000 | 0.0000 | 0.0000 | 6.068 | 0.337 | 0.594 | 0.994 |

> TCRDiff/TcrDesign/OLGA/Ours 的现有数值属于 14-pMHC local control，不与上述论文 baseline 合表。
>
> ⚠️ **「离线重评分到 sparse-13」做不到，该待办已改写（2026-08-31 核实）。** Ours 的两个 t4 run 只对 **9 个表位**生成过
> （`designs.jsonl` 各 9 行），与 sparse-13 的交集只有 **6 个**：`FTDALGIDEY` / `KINMPMSVK` / `NENLDLQEL` /
> `SALPTNADLY` / `TPSVSSSISSL` / `TSDACMMTMY`。缺的 7 个（`HPNGYKSLSTL` / `QEIRTFSF` / `QMMVKAGL` /
> `RTATKAYNV` / `RTATKQYNV` / `TDLGQNLLY` / `YERMCNIL`）盘上**没有任何生成物**，离线重算无从取数，
> 必须重新跑模型推理（属 Ours 侧，超出 baseline-only 范围）。
> 我方独有的 3 个是 `LVVDFSQFSR` / `STLPETAVVRR` 与保留项 `RVRAYTYSK`。
>
> **这同时反过来印证了 §0.4 主对照的取法**：`bioseq_unseen_common` 那 6 个 pMHC 恰好就是
> 「Ours 覆盖 ∩ sparse-13」的**全部**，即现有数据下跨模型可比的**最大**子集，不是随手挑的。

**held20（TCRT5 论文 held-out；BioSeq ⚠️ ~48% pair 泄露）**

| Method | F1@100 | prec | recall | d_edit↓ | seq-rec | Char-BLEU |
|--------|--------|------|--------|---------|---------|-------------|
| **TCRT5** (HF beam, 本次生成) | **0.0855** | 0.0855 | 0.0855 | 1.82 | 0.834 | 0.947 |
| **TcrDesign-G** (beta, 本次生成) | **0.1527** | 0.1540 | 0.1515 | 1.47 | 0.870 | 0.986 |
| **Ours-BioSeq** ⚠️seen | 0.000 | 0.000 | 0.000 | 6.36 | 0.454 | 0.411 |

> † `bioseq_unseen` = epitope 不在 BioSeq 训练集的 pMHC，零 pair 泄露。**但这一栏跨模型不可比**（成员随各 run 覆盖范围变化：只跑 held20 的 2 个、跑 benchmark14 的 6 个、跑全 34 的 8 个），且原先误含论文保留给 simulation 的 `RVRAYTYSK/HLA-A*03:01`（现已从跨切视图剔除，8 个）。**跨模型请用 §0.4 的 `bioseq_unseen_common`（固定 6 个 pMHC）。**
> 结论：TCRT5 beam 在 held20 上 F1=0.086、seq-recovery=0.83；**TcrDesign-G 同 split F1=0.153、seq-rec 0.87**，为当前最强官方 baseline。BioSeq 表位条件 CDR3β 设计尚未找回任何真 binder（F1=0）。**「seq-recovery 仍优于 OLGA 下限」这一表述已撤回**：在同集合口径下 d_edit 之差为 −0.047，配对 t 检验 t=−0.55（df=5）**不显著**，即分辨不出与 OLGA 的差别，不存在「微弱优势」，见 §0.4 主对照。benchmark14 上所有方法 F1≈0（稀疏 ref），以 seq-recovery / edit dist 为主读数；**TCRDiff**（pMHC-only 扩散对照）benchmark14 F1=0.0009 / seq-rec 0.587，**held20 F1=0.0118 / d_edit 2.954 / seq-rec 0.732**（两个 split 的值此前被混用）。

### Setting C — 全长 α/β 组装

> 两种口径：(1) **无条件全长**（BioSeq，OTS 分布）：validity（合法 AA + CDR3β 可提取率）→ ANARCI 提取 CDR3β 后套 Setting A 分布指标；(2) **表位条件全长**（TcrDesign，SOTA）：官方 `tcrdesign.py` pipeline，全长 α/β + CDR3β 提取后套 Setting B 指标。

**(1) 无条件全长（n=500，BioSeq；Volc 任务 `t-20260705205817-8mvdh` 已完成）**

| Method | cond | frac_valid_aa (β/α) | frac_cdr3_extracted | 提取CDR3β JSD↓ | novelty | NN dist | pgen+ | med log10 pgen |
|--------|------|---------------------|----------------------|----------------|---------|---------|-------|----------------|
| **Ours-BioSeq** fulllength | 无条件 | 0.998 / 0.996 | **0.038** (19/499) | 0.846 | 1.000 | 8.47 | 0.789 | -23.99 |

> 结论（诚实标注能力边界）：BioSeq 全长 α/β 的**单残基合法率高（99.6–99.8% 合法 AA）**，但**只有 3.8% 的 β 链能被 ANARCI 提取出规范 CDR3β**——说明模型能产出"看起来像蛋白"的氨基酸串，却**尚不能组装出带保守 Cys/J-motif 的真实 TCR 全长结构**（生成物富含重复 Ala，见 `outputs/.../setting_C/ours_bioseq/full.jsonl`）。可提取出的 19 条 CDR3β 相对 OTS holdout 的 k-mer JSD=0.85、多为低重组概率序列，进一步印证全长生成远未达真实库分布。**BioSeq 的强项是 CDR3 级、非全长 de novo**——这是 TcrDesign（表位条件全长 SOTA）的定位。

**(2) 表位条件全长（TcrDesign，SOTA baseline）**

| Method | 状态 | 备注 |
|--------|------|------|
| **TcrDesign-G** (`tcrdesign_G.py -mode beta`) | ✅ | Setting B：34 pMHC k=100；held20 F1=0.153 / seq-rec 0.870 |
| **TcrDesign** (`tcrdesign.py` 全长) | ✅ | Setting C：34 pMHC k=100 已评分 → overall F1 **0.061** / prec 0.133 / recall 0.040 / seq-rec **0.772** / d_edit **2.58** / Char-BLEU 0.84（`setting_C/tcrdesign_full/metrics.json`；`bash scripts/run_tcrdesign_baseline.sh full all 100 tcrdesign_full`） |
| **TCRDiff** (2026 扩散) | ✅ benchmark14 + held20 | pMHC-only 公平口径；benchmark14 F1=0.0009 / seq-rec 0.587；**held20 F1=0.0118 / d_edit 2.954 / seq-rec 0.732**（`tcrdiff_held20`，20/20，早已在盘上，「held20 待补」是过期判定） |
| **TCR-epiDiff** (2025 扩散) | ⛔ 仅引用 | 官方生成代码非 turnkey-reproducible（`from TCR-epiDiff_model import *` 非法模块名；硬编码未发布 pkl 输入；采样为 denoise-from-real-TCR 非 de-novo；4 通道编码 + 未定义 `nucleotide_to_index`）→ 跑通需 reimplement，违反"官方代码 only"；本地权重仅作 provenance，不接入、不编造数字（同 LSMTCR/TCRGen）。详见 `TCR_GENERATION_BENCHMARK.md §0.3/§7` |

> 扩散 baseline 与 BioSeq 同族对照；TCRDiff 因 eval 集无 V 基因信息，wrapper 使用官方 **pMHC-only** 模式（V 留空 → CDR1/2 特征置零），与 TCRT5/Ours 条件一致。

> 口径交叉验证：`scripts/test_tcrt5_metrics.py` gold 单测全 PASS + 对官方 `ModelEvaluator` **42 block 0 mismatch**。
> 复现：`python scripts/prepare_tcr_generation_bench.py` → `python tcr_generation_bench/run.py --setting {A,B,C} --method {olga,sonnia,tcrt5,file}` → Setting C 无条件：`bash scripts/run_bioseq_setC.sh 500`（或提交 `eval_jobs/eval_tcr_generation_setC_bioseq.yml`）→ `python scripts/summarize_tcr_generation_bench.py`

## P1 — PPI (STRING 90/90)

### P0 — MINT GeneralPPI

> **当前五任务范围（2026-07-21）**：HumanPPI、YeastPPI、Gold-standard PPI、MutationalPPI、SKEMPI。选表规则是前三项使用论文 Source Data `[P]`，后两项使用本地固定 split 的全 8 模型重跑 `[L]`；论文未公开精确 fold，因此 `[L]` 不冒充论文复现，也不拿论文值覆盖。审计见 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/mint_tasks/OFFICIAL_REBUILD_AUDIT.md`。
>
> | 方法 / 来源 | HumanPPI Accuracy `[P]` | YeastPPI Accuracy `[P]` | Gold-standard AUPRC `[P]` | MutationalPPI AUPRC `[L]` | SKEMPI Pearson `[L]` |
> |---|---:|---:|---:|---:|---:|
> | MINT | **0.8796±0.0069** | **0.6870±0.0106** | **0.6872±0.0033** | 0.6769±0.0683 | 0.3491±0.1908 |
> | ESM2-150M | 0.8537±0.0026 | 0.5973±0.0012 | 0.6775±0.0040 | 0.7349±0.0811 | **0.3493±0.1324** |
> | ESM2-650M | 0.8704±0.0146 | 0.6074±0.0084 | 0.6639±0.0065 | 0.7294±0.0824 | 0.3035±0.1658 |
> | ESM-1b | 0.8407±0.0172 | 0.6074±0.0032 | 0.6630±0.0032 | 0.7336±0.0738 | 0.3476±0.1397 |
> | ESM2-3B | 0.8759±0.0094 | 0.6218±0.0136 | 0.6513±0.0033 | **0.7520±0.0731** | 0.2965±0.1372 |
> | ProGen2-Large | 0.8204±0.0343 | 0.5838±0.0110 | 0.6146±0.0014 | 0.7140±0.0880 | 0.2733±0.0631 |
> | ProtT5-UniRef | 0.8704±0.0131 | 0.5880±0.0093 | 0.6546±0.0097 | 0.7264±0.0778 | 0.2736±0.1689 |
> | ProtT5-BFD | 0.8537±0.0146 | 0.6125±0.0176 | 0.6536±0.0039 | 0.7165±0.0812 | 0.3067±0.0883 |
> | BioSeq step121000 `[C]` | 0.6778±0.0181 | 0.6024±0.0078 | 0.5921±0.0014† | — | — |
>
> `[L]` 最终任务：MutationalPPI `t-20260722032251-pd4qq`、SKEMPI `t-20260722032319-9wfwt`，均为 **Success**；早期任务的成功格由 manifest 保留，最终 retry 只补缺失模型。机器可读混合表：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/mint_tasks/_selected_baseline_results.csv`；audit 为 expected/observed=`16/16`、missing=`[]`。
>
> BioSeq `[C]` 使用用户指定的冻结 `step121000`、joint two-chain PPI grammar record、每链最多 1,024 residues、final decoder residue global mean，以及论文 Methods 的 960→640→1 / 100 epochs / 3 repetitions probe。HumanPPI 与 YeastPPI 是 exact public fixed split；†Gold-standard 使用公开 notebook 的 163,192 个 train rows，比论文 Table 1 的 163,019 多 173，因此该格必须视为可审计本地 control、不是 paper-exact reproduction。三任务完整的 8×`[P]`+Ours 表与逐任务差值见 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/mint_tasks/_ours_step121000_official3ppi/REPORT.md` 和 `ours_vs_best_paper.csv`。
>
> **历史本地 port `[L]`**：`max_train=3000`、不同 head/数据；Bernett=IRBench subset，非论文 Gold-standard PPI。不得把下面的本地表称作 MINT 论文复现，也不得将它当成当前官方重建数据上的结果。
> **HumanPPI** = SaProt PEER split（test 237）；**Bernett** = IRBench STRING 90/90 split（test 2644，train/test 蛋白零重叠）。

#### 本地 diagnostic：AUROC / AUPRC（3-rep mean）

| 任务 ↓ \ 方法 → | MINT checkpoint `[L]` | ESM2-650M `[L]` | Ours-BioSeq `[L]` |
|-----------------|:-------------:|:---------:|:-----------:|
| **HumanPPI** | 0.938 / 0.910 | **0.939** / **0.941** | **0.830** / **0.811** |
| **Bernett** (90/90) | 0.739 / **0.748** | **0.786** / **0.767** | **0.609** / **0.616** |

#### 辅表：Accuracy / F1（3-rep mean）

| 任务 | MINT checkpoint `[L]` | ESM2-650M `[L]` | Ours-BioSeq `[L]` |
|------|:-------------:|:---------:|:-----------:|
| HumanPPI Acc / F1 | **0.876** / **0.874** | 0.851 / 0.855 | **0.727** / **0.675** |
| Bernett Acc / F1 | **0.661** / **0.625** | **0.709** / **0.686** | **0.531** / **0.165** |

> **运行位置**：MINT checkpoint local port = Volc `t-20260705210235-5k5zz`；ESM2 = 本地 sep；Ours = local globalfeat。三者仅在本地 capped grid 内比较。
> 复现：Volc → `eval_jobs/eval_mint_ppi_benchmark.yml`；本地 → `downstream/mint_tasks/_run_bench.sh` 或 `mint_tasks/README.md` Quickstart。

#### 本地 PLM diagnostic（非 MINT Fig.2 论文口径）

> Volc **`t-20260707165326-5qg4d`**（Success，2026-07-07，`eval_mint_ppi_baselines.yml`）。数字溯源 `mint_tasks/<task>/<stem>_sep_t3000/*_metrics.json`。

| 方法 | HumanPPI AUROC / AUPRC | Bernett AUROC / AUPRC |
|------|:----------------------:|:---------------------:|
| ESM-1b | 0.937 / 0.931 | 0.727 / 0.711 |
| ESM2-3B | 0.938 / 0.938 | 0.711 / 0.694 |
| **ProtT5-XL** | **0.939** / **0.941** | **0.799** / **0.777** |
| ProGen2-base | 0.925 / 0.906 | 0.622 / 0.603 |

> 该排序只描述本地 cap/Bernett 数据，不作“符合 MINT 论文趋势”的复现主张。

#### GeneralPPI 新任务（YeastPPI / MutationalPPI / SKEMPI）

> Volc **`t-20260707172352-znrkl`**（Success，2026-07-07，`eval_mint_ppi_newtasks.yml`；首轮 `t-20260707165937-zz8gr` 因 esm2 cache 路径 bug Failed 已修复重提）。9/9 格，`--max_train 3000`。Ours = post-LLaDA global pool（无 `--sep_chains`）；MINT/ESM2 = `--sep`。

**YeastPPI**（binding 分类，MLP head，test `best_test_*`）

| 方法 | AUROC | AUPRC |
|------|:-----:|:-----:|
| **MINT checkpoint `[L]`** | **0.678** | **0.717** |
| ESM2-650M | 0.648 | 0.674 |
| Ours esmc300m-cmp500k (globalfeat) | 0.601 | 0.609 |
| Ours esmc600m-cmp500k (globalfeat) | 0.567 | 0.579 |
| Ours esmc300m-mint (globalfeat) | 0.582 | 0.590 |

**MutationalPPI**（10-fold CV；类别极不平衡 pos/neg≈1473/11139 → 看 AUPRC，F1≈0 无参考价值）

| 方法 | AUROC | AUPRC |
|------|:-----:|:-----:|
| ESM2-650M | **0.725** | **0.250** |
| MINT checkpoint `[L]` | 0.643 | 0.186 |
| Ours esmc600m-cmp500k (globalfeat) | 0.621 | 0.183 |
| Ours esmc300m-mint (globalfeat) | 0.619 | 0.178 |
| Ours esmc300m-cmp500k (globalfeat) | 0.614 | 0.177 |

> ⚠️ **口径脚注**：本地 MutationalPPI 的 Y2H 标签**已是 {0,1} 二值**（来自 SWING CSV 内嵌列），与 MINT 论文 Methods（Y2H 0–4 + cutoff=2 二值化 + UniProt 序列 lookup）不完全同构；数值仅在本口径内可比。

**SKEMPI 回归状态**：旧 Pearson/RMSE 在 fold-specific PowerTransformer 空间，已撤回；旧 Spearman 只作 local diagnostic。新的完整数据、raw-unit、固定 complex folds 的 8-baseline `[L]` 重跑已全部完成并进入上方主表；每个模型为三个预定义 complex-held-out fold + weighted aggregate，共覆盖 6,706 个 test rows。

#### 排除项历史记录：PDB-Bind

> PDB-Bind 已从当前 MINT 五任务面板移除。以下只保留历史原因：HF `proteinea/ppb_affinity` 的 3267 行本地子集与论文数据构建不同，旧 Pearson/RMSE 已撤回，不得进入当前排行榜。

---

> 数据：STRING model-org 90/90 正样本 + 同 split 内平衡随机负样本(1:1)。**防泄露**：train/test 蛋白重叠=0（90/90 序列相似性划分，见 `data/ppi/leakage_report.json`）。

| Method | AUROC | AUPRC | n_test | 备注 |
|--------|-------|-------|--------|------|
| ESM2-150M + Hadamard + LR | **0.815** | **0.826** | 2644 | 通用蛋白LM 嵌入（max-train 8000） |
| k-mer(3) + Hadamard + LR | 0.562 | 0.602 | 2644 | 训练free 组成特征 |
| **Ours-BioSeq** | _pending_ | | | |

> 结论：90/90 防泄露划分下 k-mer 仅 AUROC 0.56，而 ESM2-150M 嵌入升到 0.82——预训练蛋白表征在消除相似性泄露后仍显著有效，为免疫基础模型设定了强参照。
> 复现：`python scripts/prepare_ppi.py` → `python ppi/run.py --embedder kmer` / `--embedder esm2_150m --max-train 8000`

---

## A1 — 抗体/纳米抗体（NbBench，冻结主干 + 线性 probe）

> 数据：`data/nanobody_raw/nbbench/hf_data/<task>/` 官方 train/val/test split（NbBench, Zhang & Tsuda, *Mach. Learn.: Sci. Technol.* **6**(4):040502, 2025; DOI 10.1088/2632-2153/ae20ec; arXiv:2505.02022）。
> 方法：冻结嵌入 + sklearn probe（分类 LogReg / 回归 Ridge / 位点级逐残基 head），**无主干微调**。
> **全部 12 个 NbBench 任务已跑通**（8 序列级标量 + vhh_affinity-seq + 3 位点级/生成：VRClassification / Paratope / CDRInfilling）。
> 复现：`python nbbench/run.py --task all --embedder esm2_150m`（标量）、`python nbbench/run_residue.py --task all --embedder esm2_150m`（位点级）。

### 论文主表 `[P]`：NbBench Table 5（MLP head，3 seeds）

| Task | Paper metric | ESM2-150M `[P]` | Best of 11 `[P]` |
|---|---|---:|---:|
| VRClassification | Accuracy | 0.9989 | 0.9989 (ESM2-150M/AntiBERTa2) |
| CDRInfilling | BLOSUM62 recovery | 1.499 | 1.551 (ESM2-650M) |
| SARS-CoV-2 | AUROC | 0.834 | 0.884 (AbLang-H) |
| hIL6 | AUROC | 0.850 | 0.925 (AbLang-H) |
| Paratope | AUROC | 0.922 | 0.939 (AntiBERTa2-CSSP) |
| thermo-seq | Spearman | 0.389 | 0.587 (AntiBERTa2) |
| thermo-tm | Spearman | 0.301 | 0.593 (AntiBERTa2-CSSP) |
| polyreaction | AUROC | 0.833 | 0.842 (ESM2-650M) |
| nanobody_type | Accuracy | 0.994 | 0.999 (AntiBERTy) |
| vhh_affinity-seq | Spearman | 0.170 | 0.184 (IgBert) |
| vhh_affinity-score | Spearman | 0.063 | 0.128 (AbLang-L) |

> 本地 ESM2/k-mer/Ours 使用 sklearn 单 seed probe（validation 未用于训练），不是论文 MLP/3-seed 协议，统一放在 `_leaderboard.json.local_diagnostic`，不得和上表排名。Paratope 本地 primary=AUPRC、论文=AUROC，尤其不能混列。hTNFa 是本地额外任务，论文 Table 5 不含。

> **⚠ thermo 近似重复告警（诚实标注）**：`thermo-tm` 有 **52%** 测试样本到训练集最短编辑距离 ≤2、`thermo-seq` 约 32%（NbBench 采用 75% 相似度切分，非严格去冗余）。故组成敏感的 **k-mer 在 thermo 上 spearman 虚高（0.84/0.73）**，甚至超过官方所有 PLM——这是数据近似重复而非表征优势。回归任务的可信信号仍以 **affinity**（无此问题、各法均低）为准；报告 thermo 数字时须连同该注解。


### Ours-BioSeq post-LLaDA（NbBench · 12 探针 + 生成原生 · 2026-07-05 23:01（UTC+8））
> embedder = `grammar:decoder:/abs/best.pt`；SARS-CoV-2 抗原 **head-1024 截断**。Resume Volc：`t-20260705224330-dcd5w` / `t-20260705224329-984qk` / `t-20260705224330-zf7zf` → **Success**；源：`outputs/nbbench/**/ours_*_llada/metrics.json` + `CDRInfilling/gen_ours_*_llada/`。

| Checkpoint | VRCls acc | Paratope AUPRC | SARS-CoV-2 AUROC | CDRInf (probe BR) | CDRInf gen (EM / BR) |
|------------|-----------|----------------|------------------|-------------------|----------------------|
| esmc600m-cmp500k | 0.9975 | 0.6843 | **0.847** | 1.406 | 0.400 / **1.569** |
| esmc300m-cmp500k | 0.9982 | 0.6819 | **0.874** | 1.440 | 0.375 / 1.476 |
| esmc300m-mint | 0.9959 | **0.7106** | 0.852 | 1.473 | **0.409** / **1.594** |

> 全榜见 `_leaderboard.json.local_diagnostic`。这些值只用于同一 local sklearn harness 内的诊断，不与论文 Table 5 作数值高低结论。

### AB 抗体生成（grammar_v2 · SabDab CDR + OAS light pairing · post-LLaDA 扩散解码）

> 协议：`scripts/downstream/run_grammar_v2_variant_downstream_eval.sh`；CDR = SabDab 10-fold Average AAR all folds；pairing = OAS holdout500 prompt3 n=8，ImmunoMatch + ANARCI。源：`output/downstream_generation/grammar_v2_esmc{300,600}m_cmp500k_llada_*`。7-05 Volc：`768fw`（300M Failed @ tf-keras）/ `zwgdk`（600M Success）/ `mrmzv`（300M metrics-only Success）。

#### CDR infilling（Average AAR all folds ↑）

| Checkpoint | CDR-H1 | CDR-H2 | CDR-H3 | 状态 |
|------------|:------:|:------:|:------:|:----:|
| **Ours** esmc300m-cmp500k | **68.76** | **63.64** | **44.68** | ✅ |
| **Ours** esmc600m-cmp500k | **71.88** | **67.27** | **45.25** | ✅ |

> vs 7-04 旧 ckpt：300M 64.72/57.93/40.54 → 上表（Δ +4.0/+5.7/+4.1 pp）；600M 68.95/62.34/41.80 → 上表（Δ +2.9/+4.9/+3.5 pp）。

#### CDR infilling 论文 baseline `[P]`（Ophiuchus-Ab Table 2）

| Model | CDR-H1 | CDR-H2 | CDR-H3 | 来源/注记 |
|-------|:------:|:------:|:------:|:----:|
| AntiBERTy | 76.70 | 71.10 | 42.70 | [P] overlap flagged |
| AbLang2 | 76.30 | 70.60 | 42.70 | [P] overlap flagged |
| Ophiuchus-Ab | 75.50 | 70.18 | **43.55** | [P] no overlap; zero-shot |

> 本地 AntiBERTy/AbLang2 masked-fill 和 Ours 是另一协议，保留在 `cdr_baselines/_summary.csv`，不覆盖上表。Ophiuchus 路径已修，local rerun pending。

#### Light-chain pairing（holdout500 · prompt3 · n=8）

| Checkpoint | gen ImmunoMatch↑ | ref ImmunoMatch | gen>ref ratio | chain match | V gene match | diversity | 状态 |
|------------|:----------------:|:---------------:|:-------------:|:-----------:|:------------:|:---------:|:----:|
| **Ours** esmc300m-cmp500k | **0.633** | 0.699 | **0.393** | **1.000** | **0.835** | 0.163 | ✅ `mrmzv` |
| **Ours** esmc600m-cmp500k | **0.607** | 0.699 | **0.368** | **1.000** | **0.835** | 0.140 | ✅ |

> vs 7-04：300M gen ImmunoMatch **0.430 → 0.633**；600M **0.455 → 0.607**；V gene 均 **→ 0.835**（旧 0.577）。源：`grammar_v2_esmc{300,600}m_cmp500k_llada_light_pairing_holdout500_prompt3_metrics.json`（mtime 2026-07-05）。

#### Light pairing baseline（Ophiuchus-Ab Table 3 口径 · OAS holdout500 × n=8）

外部 baseline **均为自跑**（论文未提供 p-IgGen/LICHEN 数字）；`de-novo`（无 light prompt）与 `prompt3`（前 3 个 reference light 残基作 prompt，与 grammar_v2/Ophiuchus 同口径）两套。grammar_v2 亦补跑 de-novo(prompt0) 以隔离 chain-match 成因。

| Model | 口径 | gen ImmunoMatch↑ | ref | chain match | V(exact/fam) | J | diversity | 状态 |
|-------|:----:|:----------------:|:---:|:-----------:|:------------:|:---:|:---------:|:----:|
| **p-IgGen** | de-novo | **0.686** | 0.699 | 0.552 | 0.050/0.179 | 0.150 | 0.695 | ✅ |
| **p-IgGen** | **prompt3** | **0.683** | 0.699 | **0.990** | 0.320/0.880 | 0.282 | 0.368 | ✅ |
| **LICHEN** | de-novo | 0.666 | 0.699 | 0.622 | 0.053/0.206 | 0.152 | 0.519 | ✅ |
| **LICHEN** | **prompt3** | 0.632 | 0.699 | **0.998** | 0.292/0.887 | 0.276 | 0.255 | ✅ |
| **Ours** esmc300m-cmp500k | **de-novo** | 0.604 | 0.699 | **1.000** | **0.846**/0.964 | 0.693 | 0.168 | ✅ |
| **Ours** esmc300m-cmp500k | prompt3 | 0.633 | 0.699 | **1.000** | **0.835**/0.964 | 0.628 | 0.163 | ✅ |
| **Ours** esmc600m-cmp500k | **de-novo** | 0.629 | 0.699 | **1.000** | **0.882**/0.964 | 0.735 | 0.147 | ✅ |
| **Ours** esmc600m-cmp500k | prompt3 | 0.607 | 0.699 | **1.000** | **0.835**/0.965 | 0.703 | 0.140 | ✅ |

> **全部自跑**（p-IgGen/LICHEN 非论文数字）。外部 baseline de-novo 未上 prompt3，故 chain 仅 0.55/0.62；**补 prompt3 后 → 0.990/0.998**。**⚠️ grammar_v2 的 chain=1.000/V≈0.85 不是 prompt3 造成的**：de-novo(prompt0) 下仍 chain 1.000 / V 0.85+ / diversity 0.15，因为 mask-fill 用 mask 数=reference 长度，泄漏目标长度→生成 light 与 ref 长度 100% 一致、同一性 0.92（定长近重建），prompt3 对 chain/V 几乎无增量。与外部模型真变长 de-novo 是范式差异，匹配率不可横比。ImmunoMatch：外部专用模型最强（0.68），grammar_v2 0.60-0.63。源：`output/downstream_generation/{pairing_baselines/{piggen,lichen}_holdout500{,_prompt3}, grammar_v2_esmc*_light_pairing_holdout500_{denovo,prompt3*}}_metrics.json`。grammar_v2 de-novo=当前 best.pt，prompt3=7-05 ckpt（同 ckpt prompt3 复跑待回填）。

### AB-FLAB 论文 baseline `[P]`（MINT Figure 3b · nested 10×5-fold R²）

> 论文主指标是 R²；历史 local sweep 是 5-fold pooled Spearman，不能替代本表。

| Model `[P]` | g6 Kd R² | g6 ER R² | trastuzumab R² | d44 R² |
|-------|:-----:|:-----:|:--------------:|:------:|
| AbLang | 0.244 | 0.439 | 0.293 | 0.246 |
| AntiBERTy | 0.199 | 0.401 | 0.239 | 0.217 |
| IgBert | 0.174 | 0.400 | 0.306 | 0.131 |
| IgT5 | 0.179 | 0.548 | 0.274 | 0.297 |
| MINT | **0.253** | **0.657** | **0.398** | **0.379** |

> 完整 7-method mean±std 见 `paper_reported_baselines.csv`。历史 32-run local Spearman sweep 仅为 diagnostic；paper-aligned nested-R² runner 已修，尚待重跑。

### 位点级任务：ESM2 vs 训练free one-hot 对照（可复现）
> **动机**：位点级任务此前只有 ESM2 一列，缺训练free 下限。新增 `onehot`（滑窗 ±3 逐残基 one-hot，纯局部组成、无预训练上下文）作对照，隔离"预训练主干相对局部 motif 的增益"——是标量 k-mer 基线的位点级类比。

| 任务 | 本地主指标 | ESM2-150M `[L]` | one-hot `[L]` | 论文 ESM2-150M `[P]` | 本地增益 |
|------|--------|-----------|-------------|----------------|------------------------|
| VRClassification | acc | **0.9989** | 0.8708 | 0.9989 | +0.128（区域判别强依赖上下文） |
| Paratope | AUPRC（AUROC） | **0.6817**（0.9231） | 0.4485（0.7923） | **AUROC 0.922** | +0.233 AUPRC |
| CDRInfilling | BR（EM） | **1.467**（0.396） | 1.124（0.329） | 1.499（—） | +0.34 BR / +0.067 EM |

> 结论：三项位点级任务上 ESM2 均显著超训练free one-hot（尤以 VRCls +0.13 acc、Paratope +0.23 AUPRC 最明显），说明**位点级任务确实需要预训练主干的上下文表征**，与标量任务里 thermo 之外 ESM2>k-mer 的趋势一致——这为基础模型在位点级任务留出了明确、真实的提升空间。

### 逐任务全指标（ESM2-150M，可复现）
| 任务 | 全部指标 |
|------|----------|
| VRClassification | acc 0.9989 / macro-P 0.9982 / macro-R 0.9985 / macro-F1 0.9983（n_pos=345679, 4 类） |
| Paratope | auprc 0.6817 / auroc 0.9231 / acc 0.8861 / P 0.670 / R 0.605 / F1 0.636（pos_rate 0.164） |
| CDRInfilling | br_masked 1.467 / aar_masked 0.3959 / exact_seq_rate 0.000（n_masked_pos=89440, n_seq=2846） |

> 复现命令：
> ```bash
> ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
> LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
> $ENV/bin/python nbbench/run.py --task all --embedder esm2_150m     # 9 标量
> $ENV/bin/python nbbench/run.py --task all --embedder kmer          # k-mer 对照
> $ENV/bin/python nbbench/run_residue.py --task all --embedder esm2_150m  # VR/Paratope/CDRInf
> $ENV/bin/python nbbench/run_residue.py --task all --embedder onehot     # 位点级 训练free 对照
> $ENV/bin/python scripts/import_nbbench_results.py                       # 固化官方参照 CSV
> ```
> 官方数字来源：NbBench (Zhang & Tsuda, *Mach. Learn.: Sci. Technol.* **6**(4):040502, 2025, DOI 10.1088/2632-2153/ae20ec; arXiv:2505.02022 v1, Tables 5–8) → `outputs/external/nbbench_official.csv`。
