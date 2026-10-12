# AB Light-Chain Pairing

> 2026-09-30 历史作业配置整理：已归档 YAML 的位置见 [归档索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/README.md)；文中的旧提交命令及原路径文字保留作历史记录，复用配置前请先按索引定位。

> **2026-09-17 无前缀步数扫描完成验收**。49000／ESMC反馈修复版本的iter=8/16/32/64/96/128六档均Success，完整24,000条新增候选验收通过；旧iter124作为参照，不重生成。协议及入口见§4.5，完整数值见RESULTS的独立诊断表；不自动替换p3主结果，不与140000结果混用。

> Agent 进入本任务时：先读 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md`，再读本文，数字只写 [`RESULTS.md`](../benchmark/RESULTS.md) §0.6。
> 模板：[`_TEMPLATE.md`](_TEMPLATE.md)。规则：[`downstream-doc-sync.mdc`](../../.cursor/rules/downstream-doc-sync.mdc)。

---

## 0. 元信息

| 字段 | 值 |
|---|---|
| 任务 ID | AB-PAIR |
| 锚点论文 | Zhu et al., *Ophiuchus-Ab*, bioRxiv 2026, [10.64898/2026.02.02.703197](https://doi.org/10.64898/2026.02.02.703197) **Table 3** |
| RESULTS 锚点 | `RESULTS.md` **§0.6** |
| 主指标 | ImmunoMatch ↑（`gen_immunomatch_mean`） |
| 数据根 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/comp_chain/test_data_oas_holdout.csv`（500 heavies） |
| 我们的入口 | `python -m downstream.grammar.light_chain_pairing` |
| 可引用性 | 2026-09-16 同 49000 权重的 ESMC 反馈修复后 p0/p3 全量重评及产物验收完成，见 §7 k；v3/v4 旧数字仅为修复前历史。长度仍为 reference，不是原生 EOS 同协议复现；`W_property` 仍缺失 |
| 最后更新 | 2026-09-16 |

---

## 1. 任务定义与范围

给定 held-out 重链，设计其配对轻链。共同设置为 **prompt3**（轻链前 3 残基提示），每条 heavy 生成 n=8。**BERT 禁止。**

**2026-09-13 用户确定的现行 Ours 协议**：给定 reference 轻链长度，评测已知长度条件生成（§4.3）。决策入口见 [AB 测评审计 §9](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/AB_BASELINE_EVALUATION_AUDIT.md)。Ophiuchus baseline 保留原生 EOS 生成协议；两者长度条件必须分别披露。sampler 已修、CPU 回归通过，pilot 与完整 reference-length 重评均已完成；单因素修复贡献尚未隔离（§7 h）。

### 1.1 模型适用性矩阵（硬约束）

| 我们的模型臂 | 本任务是否适用 | 原因 |
|---|:--:|---|
| Ours-Diffusion | yes | 条件生成 |
| Ours-BERT | **no** | 无迭代解码 |

### 1.2 不在本文档范围内的相邻任务

- CDR infilling：见 `AB_CDR_INFILLING.md`。
- 无提示（no-prompt）Table 3 另一区：方向相反，禁止与 prompt3 混排。
- IgCraft humanization：无关。

---

## 2. 参考文献调研（锚点论文做了什么）

### 2.1 论文实验全景

| 论文实验 | 数据 / split | 指标 | 论文结论 | 我们是否对照 | 不对照的原因 |
|---|---|---|---|:--:|---|
| Table 3 prompt3 | OAS 500 held-out heavy，每条 8 条 light | ImmunoMatch / Better / Chain / V / J / V-fam / J-fam / Div / Valid / W_property | Ophiuchus 领先；提示抬高 chain/V-fam、略降 IM | 是（headline） | — |
| Table 3 no-prompt | 同数据，无 3 残基提示 | 同上 | IM 更高、chain/V 更低 | 对照注释，不合主表 | 方向相反 |
| Table 1/2 CDR | SAbDab / SAb23H2 | AAR | — | 否 | 另一任务 |
| 其它表/图 | GDPa1 等 | — | — | 否 | 范围外 |

### 2.2 指标定义（逐字对齐）

| 论文名 | JSON key | 我们的实现 | 已知差异 |
|---|---|---|---|
| ImmunoMatch↑ | `gen_immunomatch_mean`（参考 `ref_immunomatch_mean`） | `comp_chain/eval_scripts/generation_eval.py` κ/λ heads | 无 |
| Better↑ | `gen_better_ratio` | 同 | 无 |
| Chain / V / J / V-fam / J-fam | `overall_*_match_rate` | ANARCI | J-fam 一直有，只是曾没进表 |
| Div.↑ | `diversity_mean` | 同 | 官方 argmax 会塌成 0，须 gumbel |
| Valid.↑ | `gen_valid_rate` | 同 | 无 |
| W_property↓ | — | **未实现** | 覆盖 9/10；论文里 Ophiuchus 领先幅度在此最大 |

Ours 当前主协议配置：`gumbel_argmax`，`max_iter=124`，`light_length_mode=reference`，`cfg=0`。本次全量已按此配置完成；CLI/wrapper 默认与完整约束见 §4.3，成绩见 RESULTS §0.6。

### 2.3 论文未公开的关键细节

论文未公布 cfg。本地 cfg∈{0,1.0,1.5} 在**测试集**上扫出 1.5 最高，不得宣称「超过论文」。只可说官方权重在 cfg 1.0–1.5 落在论文值附近。

---

## 3. 官方产物 release 与获取状态

| 产物 | 是否公开 | License | 本地绝对路径 | SHA256 | 获取方式 | 状态 |
|---|:--:|---|---|---|---|---|
| Ophiuchus-Ab ckpt | yes | — | Zenodo 18478480 / `/c20250601/mj/model_weights/ophiuchus_ab/` | — | Zenodo | `[R]` 已跑 cfg sweep |
| p-IgGen | yes | — | HF `ollieturnbull/p-IgGen` | — | HF | `[R]` IM 略高于论文，属采样噪声 |
| LiChen / LICHEN | yes | — | `downstream/pairing_baselines/weights/` Zenodo | — | Zenodo | `[R]` 九项 \|Δ\|≤0.015 |
| 评分器 ImmunoMatch | yes | — | generation_eval 依赖 | — | — | 已跑通 |
| 数据 | 内部 holdout | — | `data/downstream/comp_chain/test_data_oas_holdout.csv` | — | OAS 派生 | 已落地 |
| 长度先验 | 本地防泄露 | — | `.../oas_train_light_length_prior.json` | — | 由 OAS train 直方图 | 已落地 |

---

## 4. 我们的协议

### 4.1 数据与入口

| 项 | 值 |
|---|---|
| CSV | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/comp_chain/test_data_oas_holdout.csv` |
| 入口 | `python -m downstream.grammar.light_chain_pairing --csv-path ... --checkpoint-path ... --num-seqs 8 --light-prompt-tokens 3 --light-length-mode reference` |
| wrapper | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_immune_fusion_pairing.sh` 第四个参数传 `reference`；或 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_immune_fusion_gen.sh` 显式设 `GEN_STAGES=pairing PAIR_MAX_ITER=124 PAIR_LENGTH_MODE=reference` |
| Volc | `eval_jobs/eval_immune_{270m,8b}_diff_pairing.yml`；官方 `eval_ophiuchus_pairing_official_params.yml` |
| 新协议输出 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/<全新tag>_light_pairing_holdout500_prompt3_lenreference_iter124_{n8.csv,metrics.json}`；不得覆盖历史产物 |

### 4.2 口径切割线（⚠️ 引用前必读）

| 分组 A | 分组 B | 为什么不可混同 | 证据 |
|---|---|---|---|
| prompt3 | no-prompt | 指标方向相反（chain 55.6%→99.7%，IM 0.701→0.695） | `RESULTS.md:980-981` |
| Ours `reference` | `prior` / 原生 EOS 生成 | reference 是明确允许的长度条件；可并列展示但必须分协议标记，不能视为同输入条件 | §4.3；旧代码注释未同步，不再作为禁止 reference 的依据 |
| gumbel (+cfg) | 纯 argmax | argmax diversity≈0 | `RESULTS.md:429-451` |
| 官方 cfg=1.5 `[R]` | 论文 `[P]` 0.695 | 测试集选超参 | `RESULTS.md:985` |
| `max_iter=124` | argparse 默认 32 | 官方 AirGen 实际 124 | `run_immune_fusion_pairing.sh`；`downstream/README.md` 推荐表已改 124 |
| BERT pairing | diffusion | BERT 不适用 | §1.1 |
| v3@42000 CUDA Failed | 其它行 | 无数字 | `RESULTS.md:973` |

### 4.3 Reference-length 条件生成协议（已实现，全量重评已验收）

- **输入契约**：使用清洗后 reference light 的残基长度分配轻链槽位，末尾固定 `<protd>`；PAD 仍不参与 attention。完整重链可见，prompt3 只保留 reference light 的前三个残基。其余真实轻链残基必须在 decoder 和 ESMC 两路全部遮蔽。缺失、空或不满足 prompt 的 reference 必须明确拒绝/记录，不能静默切回 prior。
- **边界**：真实长度是协议允许条件，不要求模型预测长度，也不要求先做自主 EOS 训练；不把高长度匹配率本身判作泄露。仍需记录槽位数与实际解析长度，提前产生 `<protd>` 或非法 token 不能靠静默补齐/截断掩盖。
- **优先级**：先修 §7 h 并验收，再用同 checkpoint、同数据、同 seed、同 reference 长度做 sampler 受控对照。prior 可选保留作另一个长度控制实验，不作为该协议的强制 gate。
- **防答案泄露验收**：保持 heavy、reference 长度和前三残基不变，只替换 reference 剩余残基；模型实际收到的 decoder token、ESMC token 以及 attention/position/chain masks 必须完全一致。检查对象是遮蔽后真正送入模型的张量，不是仍含 labels/reference 的原始记录。
- **产物契约**：独立新 tag，建议 `protocol_id=ab_pairing_ref_length_prompt3_v1`，显式记录 `light_length_mode=reference`、`length_condition=reference`、prompt=3、目标槽数/实际生成长度、checkpoint、seed、采样器版本/修复状态和全部解码参数。更改 sampler 时同步更新 generation protocol version；不得续用旧进度文件混合新旧实现。
- **比较契约**：官方 Ophiuchus-Ab 原生复跑/论文值保持原来源层与 EOS 协议；本地 Ours 单独标 `known-reference-length`。允许附列比较，但不隐去条件差异，不称完全同协议复现或据此直接声称优于 baseline。若后来另做同长度条件 Ophiuchus 对照，另命名适配协议，不覆盖原生 baseline。
- **当前代码状态（2026-09-13 实施）**：CLI 与两个 wrapper 默认已切换为 reference，CLI 默认 iter124；模块说明/警告与 CSV 审计按已知长度协议解释。两种长度模式都先构建 prompt + 常量 placeholder，reference 后缀不进入模型记录。`GENERATION_PROTOCOL_VERSION=3`，progress signature 含关键源码 SHA，manifest 保存权重/输入 CSV/源码 SHA 与协议字段；完成产物拒绝覆盖。历史 reference 结果仍不自动恢复有效。

---

### 4.4 新授权：prompt0 / prompt3 × CFG（原生适配 v4，全部完成）

用户要求同时测试不提供任何轻链前缀与提供前三残基，并测试不同 CFG；所有组均保留已批准的 reference 长度条件。固定矩阵为 prompt={0,3} × cfg={0,1.0,1.5}，6 组各自一个非闲时、非抢占单卡作业；同一个用户指定的最新 v5 独立快照、同一 holdout500、n8、iter124、gumbel_argmax、seed42。无前缀不等于无重链条件，也不等于未知长度生成。所有组完整报告，不在测试集挑最好 CFG 冒充盲评。

- v4 CLI 已接入 `--cfg-scale`，prompt0 已通过隐藏全部 reference 内容的输入不变性检查；CSV、manifest、resume signature 都包含 prompt/CFG，旧 v3 进度不能混入。
- CFG 明确定义为 `cond + s*(cond-uncond)`，s=0 为普通条件生成。uncond 在 ESMC 和 decoder 两路遮蔽已观察的残基（重链，以及 prompt3 的轻链前缀），保留 grammar/type/终止符/关系标记/PAD；两次 forward 的已生成轻链状态相同。这是本地 residue-condition CFG，不声称与 AirGen 条件删除完全相同。
- 旧非零 CFG 路径有重复 `input_ids`/`encoder_input_ids` 关键字，且会遮掉全部 partial-mask 结构标记。CPU 复现为4 failed/2 passed；已修复。与原生探针相关套件共30项通过（protenix_abtcr），不等于全仓测试或 GPU 模型成绩通过。
- 配置生成器和独立 GPU gate 见 [原生 AB 探针 §8](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md)。用户指定最新 v5 后，真实模型六种配置的短迭代执行检查通过，6组全量作业已分别提交；权重 SHA、任务 ID、平台状态见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。尚无完整新性能结果；现有 v3 已完成结果保持原来源和解释边界。

**2026-09-14完成更新（取代上句提交时状态）**：6组正式任务全部Success，每组4000候选与评分齐全；来源、完整数字、计数核验及长度不一致候选见 [RESULTS §0.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。声明前缀全部保持，目标槽数全部等于reference；仍有少量候选解析长度较短，未静默修正。不能把prompt3下CFG的小幅IM提升说成全面改善，也不能把先前“不接CFG”当当前结果差距的充分解释；未进行新的训练/打乱重链因果对照。

### 4.5 无初始轻链残基的 iteration 扫描（2026-09-16）

- **用户问题**：检验沿用AirGen的长步数预算是否适合我们的模型，以及步数对无前缀生成IM的影响。新增固定六档`max_iter=8,16,32,64,96,128`，每档完整holdout500×8，共24,000条候选；保留已验收的修复后p0／iter124作为独立参照，不复用修复前产物。
- **控制条件**：独立留存v5 49000、相同500条重链及顺序、heavy batch=2、prompt0、reference轻链长度、gumbel_argmax、temperature=1、CFG=0、seed=42、protocol_version=5。完整重链可见，轻链没有任何真实初始残基；每轮已接受生成残基回填ESMC，pending保持MASK。模型／训练／生成器／评分器不改。
- **统计边界**：每档均报全部4,000候选的IM均值，不选best-of-8、不删除无效候选；同时保留有效率、类型／V／J匹配、多样性和参考IM，帮助解释分数变化。单seed扫描只能作为诊断，不能当成多seed显著性结论；相同seed不表示不同步数消耗了相同的逐候选随机数。`max_iter`为上限，提前填完可结束，128也不等于每条恰好128次forward。
- **代码与身份冻结**：[pairing_iter_sweep.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/pairing_iter_sweep.py)包装原有生成CLI及原有IM／ANARCI评分入口。准备阶段校验参考产物和实际权重SHA；实验manifest固定权重、输入、九份生成源码、评分源码、门禁／runner、六份YAML及旧124步参考产物SHA，worker启动与阶段切换检查，身份变化即拒绝继续，不混合版本。
- **真实门禁与验收**：每档先运行相同的两条heavy×两候选×四步真实权重反馈／隐藏参考扰动gate，再执行指定步数的全量推理；四步gate不是该档的性能结果。完成时检查4000行／500组／每组variant0–7、与124参照相同的样本身份、长度元数据、annotation顺序、全部候选的五项匹配率和有效率分母，重算CSV审计。无效／提前结束序列保留在分母内，不靠过滤改善IM。IM及Diversity直接使用同一个现有评分器。
- **产物**：[实验根](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_pairing_iters_p0_20260916)，含`experiment_manifest.json`、每档`iterN/status.json`、gate／generation／scoring日志、生成CSV／manifest／metrics。每档结束后带文件锁更新`summary.json`，只有六档完整验收才`complete=true`；该字段表示产物完成，不替代平台Success查询。失败状态和日志保留，禁止覆盖历史输出。
- **执行**：六份`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_pairing_iters_p0_20260916_iter{8,16,32,64,96,128}.yml`分别单卡、c20250601、`Preemptible:false`。首次准备运行`python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/pairing_iter_sweep.py prepare`，目录已存在则拒绝覆盖；后续只读校验／刷新汇总用同脚本`summarize`。先激活pllm，实际worker切protenix_abtcr。CPU回归45项通过，其中17项新增扫描配置／身份／完整性／汇总门禁。
- **展示**：六档及124参考全部保留，不以测试集IM最高档直接替换p3主表；不恢复CFG扫描。Ophiuchus-Ab对照只引用论文Table3的无前三残基部分，并披露原生生成长度与Ours已知长度的差异。
- **完成验收（2026-09-17）**：六档平台均Success，summary为complete=true／pending=[]／errors={}；本轮重新对全部24,000候选执行完整性、原始身份顺序、长度元数据、annotation顺序、各率分母及CSV审计核对，各档指标／CSV／manifest哈希与summary一致。六份真实反馈gate及隐藏参考扰动检查均通过，冻结源码／输入／原124产物均无漂移。IM／Diversity沿用原评分器的完整输出，本轮未重新推理／评分；结果与边界只维护于 [RESULTS §0.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-pairing)。

### 4.6 V7 no-ESMC原生轻链适配（4000预测与评分完成）

当前 2026-10-06T23:35:21.048373+00:00：完整预测和评分均真实exit0，4000逐项IM和详细分母复核通过；最终报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_pairing/report.json，根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/score_actual_completion_root_review.json。指标只见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-pairing。保留原始候选及评分，清理自有缓存；不得重放该attempt。下方按日期保留过程历史。

当前 2026-10-06T23:24:18.465554+00:00：500×8预测已完整保存并实际exit0，/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/full_local_completion_root_review.json。评分已单次启动，PID346252/start2459060767，输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_pairing_scores_local_gpu_20261006，实际日志 /vepfs-mlp2/c20250601/251105016/project/dllm_test/logs/v7_pairing50082_score_local_gpu_20261006；根观察 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/score_local_launch_root_observation.json。尚无评分完成/metrics，不重复启动或把预测完备当质量通过。下方按日期保留准备阶段历史。

2026-10-06T22:59:41.079874+00:00 评分入口已准备但未执行：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/score_v7_pairing.py --plan /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/scoring_plan_v1.json --output-root /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_pairing_scores_local_gpu_20261006。先activate pllm，使用worker-compatible Python、plan固定environment及已有local_gpu0锁；须等待当前预测真实exit0，不能并发抢占该预测。计划绑定3份现有评分源和本地权重，CPU并发上限4；直接调用原generation_eval，未更换评分算法。保留4000逐候选IM/详细比较和原CSV。IM/整体匹配与有效率分母为4000；原Diversity按非空序列分heavy聚合，保留其既有定义。评分中的既有AB基准链型判断不属于此前被停的VNAR/source-H3覆盖分支，禁止调用或恢复后者。准备根证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/scorer_preparation_root_review.json；本次尚未调用任何注释/评分。


当前更新 2026-10-06T22:54:53.864578+00:00：canonical清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/benchmark_inventory_v2.json，CSV字节未变；旧v1和下方带日期记录为历史。真实50082 CPU/GPU各4例的逐轮输入及隐藏reference扰动检查通过，全量预测已单次运行，评分尚未执行。入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_v7_pairing_eval.py；输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_pairing_local_gpu_20261006；根证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/actual_input_and_full_launch_root_review.json。候选seed固定42+row_index*8+variant，batch1；不同于V5的batch随机轨迹。完整输出必须覆盖500×8并保留空/提前终止候选，通过PREDICTIONS_COMPLETE及实际runner exit0后才进入单独评分。不得重放已开始attempt。

2026-10-06T22:39:04.095965+00:00 已核实并固定基准清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/benchmark_inventory_v1.json：完整500行，原列h_sequence/l_sequence、原协议去连字符、不过滤；两链均在A4槽位容量内。继承的AB来源存在于registry，当前新scope直接输入表不包含该CSV；该清单不等于新的去重证书。真实预测/评分未运行。

独立模块 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v7_pairing.py 复用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v7_native.py 的完整原生checkpoint加载接口。必须先在干净进程载入冻结50082/epoch2再导入模型依赖，不使用fusion转换；本轮未加载正式权重。固定未来第一组为p0、已知reference长度、500条重链×8候选、iter16、gumbel_argmax、confidence-deterministic-linear、temperature1、CFG0，与既有V5 159000同主要解码设置。p3仅为现有协议兼容测试，不进行test选步数/前缀。

pairing_input只接受heavy、light_length和声明的prefix；隐藏参考后缀不能传入。维持原A4完整canvas和已知终点；原生denoise仅接收7个feature，无label/residue_mask/terminal_mask。逐轮输入只包含固定条件、已提交生成token及pending MASK。预测提前protd不修复，raw token全部保存；解释序列按首次protd结束，须在未来全部候选分母中保留无效/空候选。采样原语复用训练快照的sampling_bioseq，词表/布局和未来候选seed分配仍须单列，不能声称与V5逐样本随机轨迹相同。

CPU人工验证 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/synthetic_root_review.json（d57ea6 rc0）：原生tiny f011、p0/p3逐轮输入/条件/RNG检查、单步argmax等价、提前终止保留与6类非法请求拒绝。没有真实数据、GPU、生产权重、checkpoint写入或注释调用；不得重复当模型能力测试。后续需固定benchmark/来源/范围证据、真实50082输入检查、全量预测与原评分器接入/验收；当前无新分数、无执行包或任务发布。此前停止的源数据VNAR/H3覆盖注释分支不得恢复；本准备不运行任何注释。

---

## 5. Baseline 清单与排行榜构造

### 5.1 论文值优先阶梯（强制，全项目统一）

§0.6 已合规：三条 `[P]` 在上，三个方法的 `[R]` 附行，Ours 表末。

### 5.2 Baseline 表

| 方法 | 出处层 | 最佳已发表值来源 | 官方产物是否拿到 | 是否真的本地跑通 | 协议差 |
|---|:--:|---|:--:|:--:|---|
| Ophiuchus-Ab | `[P]` + `[R]` | Table 3 prompt3 | yes | yes | cfg 未公开；`W_property` 本地空 |
| p-IgGen | `[P]` + `[R]` | Table 3 | yes HF | yes | 随机解码 IM +0.014 |
| LiChen | `[P]` + `[R]` | Table 3 | yes Zenodo | yes | 近精确复现 |
| 打乱配对 null | `[C]` 诊断 | 非论文 | 本地脚本 | yes | 不进主榜；IM 地板 ≈0.33 |

---

## 6. 结果写入位置

| 内容 | 唯一权威位置 |
|---|---|
| 排行榜数字 | `RESULTS.md` §0.6（§0.8 有 v3 行） |
| 任务锚点 | `benchmark/README.md` §3.1 AB pairing 行 |
| changelog | `benchmark/PROGRESS.md` |
| Volc | `PROJECT_PROCESS.md` |
| 本任务协议 | **本文件** |

产物：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/`

---

## 7. 已知缺陷与不可比口径

2026-09-30 报告继续保留历史协议；长度条件必须逐条读 manifest：`reference/auto` 记录参考长度依赖，`fixed_v2` 记录 fixed-canvas/eos，不混称自由长度生成。单 seed、历史测试集扫参以及选模独立性未验证等限制须随成绩导出，不仅留在注释中。见 [独立完整性报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_integrity_20260930/final/results/integrity_report.json)；本轮未更改采样器或重跑。

### 2026-09-16：ESMC 逐轮生成状态回填（k，当前优先）

用户授权修复并重新 inference。v3/v4 的 decoder 已使用 committed state，但 ESMC 每轮从原始 encoder buffer 重新遮蔽**整个 generation_mask**，从未读取已接受的生成残基。此问题与 §7 h 已修的 pending-token 状态错误不同，也不是已经证明无前缀 IM 下降的唯一原因。

v5 在 [共享 sampler](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py) 每次 forward 前重建 ESMC 的目标位置：仅从当前 `output_tokens` 取已接受残基，经 `llada_to_grammar_ids` 转到 ESMC 词表；pending／无法映射的 token 保持 encoder MASK。原始目标 buffer 在所有 generation 位置被覆盖，绝不因 pending mask 缩小而恢复参考答案。重链、允许的 prompt、终止符、PAD 和 attention 结构不变；CFG 两次 forward 共享同一生成状态，无条件分支只删除可见条件残基。

- pairing 的隐藏参考后缀仍在 collator 前替换为占位；真实后缀只留在评分 metadata。p0 不提供任何真实轻链残基，p3 只允许前三残基；两者均允许 reference 长度。
- 协议版本升为 **5**；manifest/metrics 登记 `encoder_state`，旧 progress 禁止续接新协议。训练 loss／权重未改，已运行的 Python 进程不会热更新。共享 sampler 的其他多步调用会继承此修改，但不据此声称已完成其他任务重评；单步没有后续 forward，当前 CDR iter=1 不受回填改变影响。
- CPU 两份生成 suite 28 项通过；覆盖逐轮实际 `_denoise` 输入、不同词表、变长 batch、固定／已接受 token、CFG、proxy mapping、非法输出保护，以及改变隐藏参考后缀后**每轮**两路输入不变。额外训练 corruption 路径 3 项通过；实际执行环境 protenix_abtcr 的生成及 AB probes 三套共 38 项通过。
- 真实权重门禁：[preflight_pairing_encoder_feedback.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/preflight_pairing_encoder_feedback.py)，逐轮检查两路残基一致、ESMC 可见生成残基递增、改变隐藏答案后输入和生成结果不变；2 条 heavy × 2 候选 × 4 步，仅协议门禁，不是性能 pilot。
- 两组真实 A100／49000 门禁已通过；门禁与 full manifest 的权重／数据／九份生成源码 SHA 均核对一致，且与修复前的权重、数据及采样配置相同。完整 IM／ANARCI 评分已完成；平台终态与耗时只在 PROJECT_PROCESS 维护。
- 本轮固定独立留存 **v5 49000**、相同 holdout500 × 8、heavy batch=2、reference 长度、iter124、gumbel_argmax、temperature1、CFG0、seed42；只重跑 p0/p3 两组，分别单卡非闲时，不扫描 CFG、不换训练权重。
- 入口：[run_pairing_encoder_sync.sh](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_pairing_encoder_sync.sh)；两份 YAML 位于 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_encoder_sync_20260916_p{0,3}.yml`。独立输出根 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_pairing_encoder_sync_20260916/`，先真实门禁，再全量生成与原有 IM／ANARCI 评分；执行状态见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。未完成不得填写新分数；旧产物完整保留。
- RESULTS 继续遵循用户简化要求：仅有前缀 Ours 与 Ophiuchus-Ab 论文主对照；p0 用于本次诊断，不恢复 CFG 扫描或消融主表。
- 完整产物验收（2026-09-16）：p0/p3 各4,000行、500条原始heavy身份、每条variant 0–7完整；标准氨基酸、reference长度及p3前缀检查通过。逐条ANARCI记录数量／顺序一致，五项匹配率和有效率按全部4,000候选重算一致，CSV审计重算一致；九份源码SHA与gate／manifest／metrics一致，前后权重与数据SHA及采样参数一致。IM／Diversity采用已完成评分器的聚合输出，本轮没有重新执行评分模型或ANARCI，也没有声称独立重算IM。
- 新数值与对比结论只维护于 [RESULTS §0.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-pairing)；[p0诊断产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_pairing_encoder_sync_20260916/pair-p0-cfg0/holdout500_metrics.json) 保留。此轮为同权重、同配置的一次固定seed复跑，不能推广为所有模型／seed的提升保证，仍未补齐`W_property`。

| # | 缺陷 | 状态 | 后果 | 证据 |
|---|---|:--:|---|---|
| a | 历史 reference 长度条件未与自由生成区分 | 现协议允许 reference 长度，见 §4.3 | 不再因长度已知单独判无效；历史 VOID 结果仍需排除其他问题，不能自动恢复 | §4.3；历史记录保留于 RESULTS |
| b | 官方 argmax 塌缩 | 已修（gumbel+cfg） | 旧「0.352 vs 0.701」是解码不是模型 | `RESULTS.md:429-451` |
| c | encoder 泄露 | 已修 | 旧数字 VOID | `RESULTS.md:391-410` |
| d | 历史 checkpoint 的 H→L 训练模式覆盖不足假说 | 本次未改训练 | 旧“仅捕获官方约 30%”估计受旧 sampler 混杂，不能外推为新协议或当前 v5 的结论；需另做训练/条件化对照 | RESULTS 历史分析；本节 f/h |
| e | `W_property` 缺失 | 未实现 | 不能声称与 Table 3 指标全对齐 | `RESULTS.md:977` |
| f | 历史 IM / V 区差距被解释为纯能力问题 | 因 h 的生成器混杂，撤回“不是 harness”的排他结论 | 先做同 checkpoint 修正前后对照，再判断训练贡献 | 本节 h；历史数字仍见 RESULTS §0.6 |
| g | CLI 默认 `max_iter=32` | 已改 124 | 新配置仍需显式记录 | pairing CLI 与新评测 YAML |
| h | 多步解码中 pending mask 与实际输入 token 脱离 | **代码已修，CPU 回归与 reference-length 全量验收通过；单因素归因未完成** | 新结果按新协议报告；旧多步产物不自动恢复可引用性，不能把改长度和修 sampler 的联合变化全部归因于 sampler | 下节实施记录 |
| i | 将长度先验解释为“模型没有终止 token / 长度必定等于槽位数” | **解释和代码说明已纠正；采用 §4.3 已知长度协议** | 不能据此声称已评测自主长度生成 | 下节长度与终止符复核 |
| j | 历史 pairing 未传 CFG，非零分支存在重复传参/结构遮蔽问题 | **v4 已通过 CPU 与真实 GPU 门禁；全量 sweep 已提交** | v3 历史结果仍为0；新实验见 §4.4，不将差距归因提前关闭 | 下节历史 CFG 复核及 §4.4 |

### 2026-09-13：共享 grammar sampler 的 pending-token 状态错误（h，P0）

**最新实施记录（优先于下方修复前诊断）**：用户授权开始评测后，主循环已分离 proposal 与 committed state，history 现在记录真正进入下一轮的状态。reference 模式先用常量占位替换隐藏答案后缀；CLI/wrapper 默认和 v3 协议同步。

- 新增 [CPU 回归测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_pairing_generation_protocol.py)：真实主循环 1/4/16 步、变长 batch、逐轮 pending mask 计数、fixed/committed token 不变、history 与下一轮输入一致，以及替换 reference 隐藏后缀不改变两路模型输入。修复前 2 failed / 2 passed；首批修复后连同基础 suite 为 10 passed（GPU worker 启动时亦通过）。进一步补充实际 `_denoise` 边界捕获、不同 decoder/encoder ID 空间、非法短 reference 拒绝和共享 mask-ID helper fallback 后，本地共 **16 passed**。
- 扩展运行 `test_gidd_ophiuchus_edit.py` 合计 19 项中 18 passed；唯一失败是既有 `test_ophiuchus_ratio_buckets_and_val_independent`，它假定精确桶数量，与当前训练的逐样本随机抽桶不符，未修改此无关测试/训练逻辑。不得宣称全仓测试通过。
- 实际 270M@42000 权重 CPU load、tokenizer remap 与 collator 检查通过，shape=(1,24)、d_model=768、8 layers。加载器之前为了 mask ID 导入整个训练入口；现将原样 helper 移到 fusion 公共模块，训练与推理复用，行为不变。
- 环境预检：pllm 缺 Biohub ESMC 与 ANARCI；CPU 单测仍用 pllm，真实模型与评分使用既有 protenix_abtcr 环境（未安装/升级共享依赖）。新 [Volc 配置](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_immune_270m_42000_pairing_reference_v3.yml) 顺序运行 pilot32 与 full500、n8、iter124、cfg0、seed42。作业 ID/平台状态只在 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md) 维护。
- 独立产物目录：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_pairing_reference_v3_20260913_270m42000/`。先落 manifest 和 generation log，完成后分别落 CSV / metrics；pilot 不能代替完整结果。新 scorer 入口为 [score_pairing_generation.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/score_pairing_generation.py)，支持生成/评分环境分离而不重新生成。
- pilot 与完整 holdout500 的生成、ImmunoMatch / ANARCI 评分均已完成；数值与解释边界只维护在 [RESULTS §0.6 全量](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。全量 CSV 与原始数据的重轻链配对集合一致，每组候选数正确；prompt、长度、字母表及逐条匹配率分母复核通过，full 的前缀行与 pilot 完全一致。实际权重、输入 CSV、manifest 记录的九份源码 SHA256 复核一致；本次复跑两份相关 CPU suite 仍为 **16 passed**。
- **验收分层**：实现状态不变量与本次已知长度全量产物已通过；这不等于已完成相同 reference 长度下“只改变 sampler”的受控实验。与历史 prior 的差分是联合协议变化，不足以关闭单因素归因问题；也不自动关闭旧 CDR 结果复核、训练模式假说或 `W_property` 缺项。

下方保留修复前诊断证据：

证据入口：[本地主循环](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py:438)、
[本地 confidence 更新](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py:251)、
[AirGen 主循环](/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/src/byprot/models/lm/dplm_multichain.py:836)。

AirGen 将 **旧 output_tokens 的 clone** 与 **本轮 proposal 的 clone** 分开传给 `_decoding`，只把选中的位置写入旧状态，其他位置保持 mask；原函数还断言待生成标记等于实际 mask token 的位置。
修复前本地主循环先用 proposals 填满所有 still_masked 位置，再把填满后的同一状态同时作为 output_tokens 和 cur_tokens 传入 `_confidence_decoding`。该函数只缩小 still_masked，却没有把尚未接受的位置恢复为 mask；下一轮 `_model_logits` 直接读取这个候选已填满的 output_tokens。

**已执行的只读 CPU 复现**：在 pllm 环境中用 AST 提取两侧实际 confidence 函数及本地 deterministic topk 函数，不导入模型、不加载权重、不用 GPU；初始四个 mask（token ID 32），proposal=[5,6,7,8]，score=[0.9,0.8,0.2,0.1]，step=1/max_step=2。

| 路径 | 下一轮输入 | pending 标记 | pending == 实际 mask |
|---|---|---|---|
| AirGen（旧状态 + proposal 分离） | [5,6,32,32] | [false,false,true,true] | true |
| 当前本地调用方式 | [5,6,7,8] | [false,false,true,true] | false |

诊断退出码 0（断言确认两种行为不同）。相同主循环已存在于仓库 commit `c6ed2c9`，不是当前未提交的 relation-mask 修改引入。具体历史产物仍需逐个绑定运行代码版本，不能仅凭当前源码断言每个历史结果都受影响。

**边界**：这是实际实现差异及状态不变量破坏的证据，不是修正后会涨多少分的证据。`max_iter=1` 没有后续迭代读取错误状态，不能机械判为同样受影响；CDR 的多步行也需核查。Ophiuchus baseline 走另一条生成路径，不因本问题作废。

**原修正建议（已采用 proposal/state 分离实现）**：保留旧 token/score 状态，只将 proposal 交给 confidence 选择；或者在返回前显式恢复 pending 位置的 mask。必须保持已接受位置与 fixed context 不变，并测试每一步 `still_masked == (output_tokens == mask_id)`（限生成位置）、mask 数单调下降和最后归零。
用同 checkpoint、相同 heavy 子集/候选数/seed/长度样本/解码参数做修正前后对照，冻结训练参数；先查 invariant，再测指标。逐轮 history 应区分 proposal 与进入下一轮的实际状态。

**其他尚不能排除的因素**：本地 prior 预分配槽位与 AirGen 自吐 EOS 的差异、CFG 调用差异见下节 i/j；ESMC 分支在整个生成区保持遮蔽，与只遮 CDR 的可见上下文差异很大。训练默认 joint 桶与 AirGen train.sh 分链桶不同，但已存在 chainratio 对照 YAML，不能称作“项目完全没有尝试”；必须另查运行和 checkpoint provenance。以上因素都不能在 sampler 修正对照前被宣称为唯一根因。

### 2026-09-13：长度、终止符与 CFG 复核（i/j）

**旧注释已纠正**：[pairing 模块说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/light_chain_pairing.py:11) 原 “no terminator” 和 “slot count is generated length” 不准确；用户授权实施后已改为槽位预算/可能提前 `<protd>` 的描述。

1. **训练布局与 padding**：[renderer](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/grammar.py:386) 将重轻链放在同一 `<prots><ab> H . L <protd>` 块内；当前无条件 AB 记录前面还有固定 null-context 块。grammar 层分隔符为 `.`，经 [LLaDA remap](/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_fusion_model.py:239) 变为 `<chainsep>`，不是字面 `<split>`。[collator](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/grammar.py:792) 按 batch 最长记录补右侧 PAD，位置在最终 `<protd>` 后，PAD 的 attention/训练 eligibility 为 0、labels 为 -100。ESMC 是另一路逐链 `<cls> seq <eos> PAD…` 输入，不能与 decoder 的 `<protd>` 混同。
2. **有 token 不等于所有 checkpoint 都学过自主结束**：当前 generated-only 保留 renderer 的显式 eligibility，目标块的 `<protd>` 可以被遮蔽、作为恢复目标；固定 null-context 的 `<protd>` 不参与。`diffusion_all_chains=True` 则在 [compute_loss](/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_fusion_model.py:643) 用 [all_residue_eligible_mask](/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_fusion_model.py:173) 覆盖 eligibility，排除结构 token，包含 `<protd>`。当前 v5 YAML 显式为 False；历史 checkpoint 必须查实际训练参数/代码版本，不能从词表存在推断学会了长度建模。即使纳入恢复目标，也尚未验证它在未知长度窗口内的终止能力。
3. **当前推理只是预分配布局，不是完整的未知长度协议**：[generate_for_batch](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/light_chain_pairing.py:256) 从训练先验抽槽位数、构建 placeholder light，再由 [partial-mask helper](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/masks.py:106) 仅开放轻链残基位；[resolve_partial_mask](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py:47) 强制结构 token 可见，所以预设的末尾 `<protd>` 保持固定，PAD 也不开放生成。不过 [chain parser](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/masks.py:82) 遇到目标块内更早的 `<protd>` 会结束提取，因此不能断言输出长度必然等于抽样槽数。
4. **本轮只读 CPU 验证**：调用当前实际 collator、pairing mask 与链解析器；AST 提取当前 all-chains eligibility helper，不加载权重。两条玩具 AB 记录分别为 H=`ACDE`/L=`FGH` 与 H=`ACDEFG`/L=`HIKLM`：短记录的四个 PAD 全在最终 `<protd>` 后；两条目标 `<protd>` 的 generated-only eligibility=true、all-chains eligibility=false、pairing generation=false。将第二条轻链第三个残基手工替换为 `<protd>` 后解析结果为 `HI`。所有断言通过、退出码 0。注入 token 仅测试解析行为，不是模型生成实验。
5. **未知长度窗口只是待验证研究方向，不是现有 checkpoint 的直接修复方案**：用户进一步指出 PAD 不参与 attention，这一约束确实构成训练/推理失配。[噪声函数](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py:352) 只遮 eligible token、不修改 attention；[训练 forward](/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_fusion_model.py:677) 继续传原始 attention，最终 [LLaDA attention bias](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/llada/models/modeling_llada.py:1307) 排除 PAD keys。因此即使目标 `<protd>` 被遮住，有效 attention 范围仍由真实序列末端确定，尾部 PAD 从不成为可见、受监督的噪声位。把超长窗口全部置为 attention=1 的 MASK，会新增双向可见位置且没有对应尾部恢复训练；若仍设 attention=0，又会被当前 generation mask 排除，不能自动扩展生成。前者是分布失配，尚未实测必然失败或失分幅度；不是 PAD 机制本身有 bug。
6. **AirGen 的固定窗口有不同的训练依据**：[run/train.sh](/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/run/train.sh:9) 选择 `ab/mint_650m_stage1`，其 [配置](/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/configs/experiment/ab/mint_650m_stage1.yaml:7) 指向 poas。实际 [MyCollateFn](/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/src/byprot/datamodules/dataset/poas.py:255) 使用 H=150/L=128 固定缓冲区，`tokens.fill_(eos_idx)`，尾部是重复 EOS 而非 PAD；[maskable 判定](/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/src/byprot/models/lm/dplm_multichain.py:579) 不排除 EOS，[q_sample_comp](/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/src/byprot/models/lm/dplm_multichain.py:174) 可将这些位置遮住并返回 loss mask。其 [MINT encoder](/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/src/byprot/models/lm/modules/mint/model/esm2.py:90) 只按 PAD ID 排除 padding。因此不能只照搬它的推理缓冲区，却忽略训练尾部的参与方式。以上为本地原脚本链路证据，不等同于确认论文 checkpoint 的完整训练 provenance。
7. **后续方案边界（被用户新决定更新）**：本阶段采用 §4.3 的 reference-length 协议，先隔离 §7 h；此前建议“必须 prior/不暴露长度”不再约束该协议。未知长度生成仍可作为另一个研究任务，需训练/适配与推理一致的窗口设计；不作为当前 pairing 评测的阻塞条件。prior 与自由 EOS 旧产物作为各自协议保留，不改标签或覆盖数值。

**AirGen 的 CFG 是实际开启的**：[原运行脚本](/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/run/heavy2light.sh:7) 固定 `max_iter=124`、n=8、gumbel_argmax，扫 `cfg_scale ∈ {1.5,1.25,1.0}`。[调用入口](/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/downstream/comp_chain/generate_light_from_csv.py:202) 将该参数传给 `model.generate`；[forward_decoder](/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/src/byprot/models/lm/dplm_multichain.py:514) 用同一个模型计算有条件与遮掉 partial-mask 条件的两个 logits，再按 `guided = cond + cfg_scale * (cond - uncond)` 合成。这里 0 表示普通条件生成，1 表示 `2*cond - uncond`，不是外接分类器或 ImmunoMatch 参与生成。本地 pairing 调用不传 CFG，而 [公共包装器](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/common.py:139) 默认 0；函数存在参数不代表 CLI 已接通或分支已验收。原脚本的非零设置也不能证明论文最终使用了哪档或哪次产物。

---

## 8. Agent 执行手册

### 8.1 提交前 gate（逐条打勾，任一不过就不要提交）

- [ ] 先满足 §7 h 的状态不变量、双路遮蔽及产物验收；若声称 sampler 的修复收益，须额外完成同长度的单因素对照，不能用联合改动代替
- [ ] **diffusion only**
- [ ] 显式记录前缀：历史有前缀主行 `--light-prompt-tokens 3`；140000 及本轮 159000 对照为 `--light-prompt-tokens 0 --cfg-scale 0`。两种条件分行比较，不混用。
- [ ] 明确步数：历史默认及 140000 为 `--max-iter 124`；2026-09-17 用户指定 159000 为 `--max-iter 16`，不得沿用模板的 124；对比时披露此差异。
- [ ] Ours 显式 `--light-length-mode reference`，按 §4.3 标记已知长度协议；官方原生 baseline 不随之更改
- [ ] 通过 §4.3 双路遮蔽不变量测试；已知长度不等于允许真实轻链内容可见
- [ ] 全新输出 tag/protocol 记录，不能恢复旧 sampler 的 progress 或挪用旧 reference 分数
- [ ] 解码 gumbel_argmax；不要纯 argmax 当 headline
- [ ] 收数 9 个已实现指标；`W_property` 留空
- [ ] `test -f ${CKPT}/model.safetensors`

### 8.2 提交命令

本轮实际配置及实施记录见 §7 h，作业账本见 PROJECT_PROCESS；下列为旧模板，不能重复提交当成新运行。已将当前 CLI/wrapper 默认改为 reference，但旧 YAML 若显式传 prior 仍会覆盖它，提交时必须核对参数与新 tag。

```bash
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
bash scripts/volc-no-proxy.sh ml_task submit --conf eval_jobs/eval_immune_270m_diff_pairing.yml
```

本地：

```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
export PYTHONPATH=/vepfs-mlp2/c20250601/251105016/project/dllm_test
export LD_LIBRARY_PATH=${ENV}/lib:${LD_LIBRARY_PATH:-}
export GEN_STAGES=pairing
export PAIR_MAX_ITER=124
export PAIR_LENGTH_MODE=reference
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
bash scripts/downstream/run_immune_fusion_gen.sh \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune/checkpoint-42000 \
  ours_fusion_270m_diff_42000_reflen_v1 4 8
```

### 8.3 收数

读 `*_iter124_metrics.json`：

- `gen_immunomatch_mean` / `gen_better_ratio` / `overall_chain_match_rate`
- `overall_v_gene_match_rate` / `overall_j_gene_match_rate`
- `overall_v_gene_family_match_rate` / `overall_j_gene_family_match_rate`
- `diversity_mean` / `gen_valid_rate`

### 8.4 填表

1. 确认属于 prompt3，并显式标记每行的长度条件。`[P]` 在上，原生 `[R]` 附行（不要把 cfg=1.5 标成超过论文）。Ours 的 reference-length 结果在表末单列协议，不替换历史 prior 分数。
2. changelog → `PROGRESS.md`。

### 8.5 同步哪些文档

`RESULTS.md` §0.6 + `PROGRESS.md` + 本文件 + `PROJECT_PROCESS.md`。

---

## 9. Changelog

| 日期 | 变更 |
|---|---|
| 2026-09-02 | 初版。锁定 prompt3 + iter=124 + prior；登记 heavy2light 失配与 W_property 缺口。 |

### 2026-09-27 completed-generation scoring retries

The v2 eval YAMLs validate existing generation via `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/validate_pairing_reuse.py` before skipping generation and rerunning scoring. Validation covers checkpoint and input SHA256, generation parameters, exact 500×8 row count, reference H/L identities, variant order and nonempty outputs. Original generation provenance is retained; current scorer does not relabel old generation code. Both archived CSVs were validated and copied back; no samples were regenerated.
