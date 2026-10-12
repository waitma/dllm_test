# p02 context 组合消融图

更新：2026-10-03T01:54:08Z。终态图仍partial（8/22，3750步）；14项启动前撤下。实现入口：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_context_ablation.py。

本工具只读 p02 的 jobs.json、run_provenance.json、progress.json、curve.jsonl、summary.json、COMPLETE.json 和 pool result 元数据，不读训练/验证序列，不导入模型，不修改训练、队列或旧 p01 作图。p02 固定 120000 train、1024 valid、global32、3750 updates，11 条件 × seed42/43。公共源码、完整配置、数据 manifest、日程、目标及同 seed 公共初始化必须一致；相同 seed 在共同评估点的记录/token/mask 曝光也必须一致。

## 运行

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_context_ablation.py \
  --jobs-file /absolute/p02/jobs.json \
  --pool-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930 \
  --output-prefix /absolute/p02/figures/context_ablation
```

jobs.json 顶层为 `{"jobs": [...]}`，每项包含 `job_id/alias/variant/seed/job`，job 内含 `variant=alias、mode=train、steps=3750、data_dir`。空 jobs 或只有预注册而尚无实测曲线时，仍输出缺格清单、CSV 表头与 JSON；不生成数值图。脚本不替用户选取重复尝试，同条件同 seed 的多次尝试须先显式确定唯一 job。

## 真实评估、缺失和最终结果

- 按已存在非零评估的运行求共同已记录 step；至少两项具有共同非零点才画图。图题和 JSON 明示 `PARTIAL`、实际参与条件数、step 及全部缺格。学习曲线展示所有已完成评估点，不插值；step0 不冒充训练终点。
- 只有全部 22 项存在真实 3750 点，records=120000、epoch=1、position=0，COMPLETE 的 summary/provenance SHA 与原件一致，且每项 pool result 为 succeeded/returncode0，才标 `final=true`。早期/缺格/完成后尚未观察到 pool 退出均不是最终结果。
- 每 mask 的 metric 必须与原 numerator/count 一致；计数为零时保留 null。source/context 的 records、token counts 和各 numerator 必须能还原 pooled 值。末尾尚未完整写完的 JSONL 行明确忽略并记录 warning。
- 输入文件 SHA 与读取字节数、候选状态、原始评估点、指标、缺格和脚本 SHA 都保存在 JSON。这里的 running 只表示 metadata 文件存在，不声称做了活性监督。

## 图与表

每页输出 SVG/PDF/PNG，前缀后缀分别为：

1. `_overview`：实测学习曲线；同 seed f000 的四 mask mean AA CE 差值；训练时间—重建指标；共同点峰值 allocated 显存。时间沿用 train loop 口径，明确不含验证和 checkpoint 时间。
2. `_factorial`：f000…f111 × 两 seed 的完整设计格（缺格空白）；G/S/L 主效应、GS/GL/SL 二阶及 GSL 三阶对比；分别移除 G/S/L 相对 f111 的差值。
3. `_strata`：来源及 with_context/without_context 分层四 mask mean AA CE，标签附验证记录分母，各 mask 真实 token 分母在 CSV/JSON 中逐项保留。
4. `_separate_controls`：cv_segment_global、cv_late_fusion、fixed_gate_control 相对同 seed f000 的 AA loss、relation loss、relation accuracy 差值。这三项不混入 2³ 因子交互计算。

主 CSV 包含 pooled/source/context × 各 mask 原始 metric、numerator、count，以及 mask_mean（它没有唯一 token 分母，因此 count 列留空）。另输出 `_paired.csv`、`_interactions.csv`、`_removal.csv`。所有表保留 seed，不用缺失值补零。

## 对比定义与结论界限

fGSL 位序与模型一致：G 动态 head gate；S 固定 residual 0.1；L 同段半径 32/global 交替。单 seed 因子交互必须有全部八格实际共同评估值才计算。统一采用 `sum(product(signs) * observed_value) / 4`：正号四格均值减负号四格均值，主效应/二阶/三阶采用相同 convention。该交互量是完整因子对比，不是拟合回归系数或显著性检验。

去除效应为 f011−f111（去 G）、f101−f111（去 S）、f110−f111（去 L），只在相应实际同 seed 配对存在时记录。所有差值是 candidate−baseline；loss 较低和 accuracy 较高的方向不同，图使用明确指标名，不混成统一胜出分数。

两 seed 不报告置信区间或显著性。当前 relation 来自已有正例监督 token，不代表正负判别能力。重建指标也不代表下游生物功能、生成品质或最终架构优胜。当前无 p02 实测共同点评估，代码交付不等于已得到效果图。

## 验证状态

2026-10-02T21:04:33Z，pllm 实际完成 4 项纯标准库检查：实际 Factorial 源码的 11 条件一致；已知纯数学三因素表的 7 个正交对比及常数零效应；缺格/nonfinite 拒绝；空设计保持缺失且未导入 Torch/NumPy/Matplotlib。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/plot_numeric_contract_check.json。这里的数值仅为内存中的数学控制，没有写入任何实测 CSV/图或模型效果记录。

随后使用 root 已实际登记的 22 项真实账本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/jobs.json 首次运行，返回 `awaiting_shared_evaluation`、`final=false`、`endpoint_step=null`、22 个 missing endpoint、`figures=[]`。实际输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/figures/v7_context_ablation.json 及同前缀 CSV；此时未出现可比较的真实数值，正确不生成效果图。

脚本 SHA256：`e099e902418aa74a48383cfd75af2beb3ace023e4386303f1dac64ac9895e574`。SVG/PDF/PNG 的实数据绘图路径仍须等待共同评估点存在，没有用假生物效果 fixture 预填图页，也不将未执行的渲染路径称为已通过。无训练、模型、GPU、序列读取或队列修改。


## 2026-10-02T22:08:15.189175+00:00 p02首批共同250步真实图首次渲染通过

22:06:28 UTC首批8项f000/f001/cv_segment_global/cv_late_fusion×seed42/43均完成250步验证（每项8000记录曝光），实际训练250–300/3750，8running/14pending，无失败/心跳或曝光异常。既有p02作图首次真实渲染rc0，status=partial、final=false、endpoint_step=250、8/22实测格、14缺格，warnings=[]。原sample/source/配置/同seed初始化与同step曝光检查通过。

实际overview/factorial/strata/separate_controls四页各SVG/PDF/PNG共12图已确认存在、非空并记录SHA；overview/factorial PNG目视通过，组合缺格及尚不可计算的交互/移除对照保持空白。图前缀 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/figures/v7_context_ablation；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/first_render250_20261002.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/first8_shared250_20261002T220628Z.json。

当前是新120k上的首批早期正例学习/遮蔽重建结果，不能称22项完整消融完成、正负判别或最终架构选定。旧p01最终图未覆盖；全部sole latest.pt保留。继续原22项队列，不新增/重试任务、不改运行源码。


## 2026-10-02T22:19:34.974599+00:00 旧p02监督范围随用户batch调整收窄为原8运行项

根端22:15:25 UTC已将原14个未启动pending原件原子移至batch_revision_20261002/withdrawn_pending，原22项jobs账本保留。独立监督核对14份移后原件SHA与withdrawal_receipt一致，均无run目录、评估或pool result，状态明确为withdrawn-before-start，非失败或仍pending；不补回队列。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/batch_revision_20261002/withdrawal_receipt.json；独立状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/withdrawal_status_independent_20261002.json。

旧p02仅原8个micro4/GA8任务继续3750步，终态图保留8/22 partial与14未执行缺格，缺失原因为用户要求提高后续batch。既有模型、样本、日程、checkpoint与worker未改；新large-batch组须另册登记及作图，不和旧p02拼矩阵。旧p02监督停止点改为原8项全终态，或先到原12小时观察上限。


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
