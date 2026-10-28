> 2026-10-29（+08:00）补充生成区分诊断：A5正确目标整链AAR为37.02%，同来源/链类型其他不同目标的重评分为35.42%，差值+1.60个百分点且主要来自TCR native。提示保守序列成分贡献较大，不能直接声称条件特异性或抗体设计改善；分数重配对不是条件干预。6条重复TCR beta输出对应相同MASK输入，不能据此判为模式坍塌。继续原九组三种子确认，阈值、源码与数据保持。方法与证据见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> 2026-10-29（+08:00）筛选输出核查：八架构936条整链预测的AAR独立重算一致。A5相对A0参考位置覆盖率33.60%→95.82%，在两者共同覆盖的同一组位置上准确率37.87%→53.64%（均来源宏平均）；不足真实长度一半的输出107/117→0/117。诊断支持继续确认A5，但不能只用长度或单种子筛选宣称架构已胜出。整链CDR位置评分与有上下文CDR补全分开解释；原三种子确认和统计阈值保持不变。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> 2026-10-29（+08:00）八组1epoch筛选全部结束，按预定规则冻结 A5（生成候选）/A4（表征候选）与 A0 对照。进入原计划的3epochs×seeds42/43/44，共9组、每组3828步；用户要求全部在当前开发机单GPU串行执行。各组重新随机初始化，以相同seed的A0作配对参照，使用独立confirm评测和AB specificity二级验证。既有阈值、数据形式和训练源码SHA保持；尚未得到三种子确认结论，不据筛选结果替换正式1B架构。

> 2026-10-28 架构探索：批准矩阵 A0 原版、A1 QK norm、A2 SDPA 后门控、A3 组合、A4 可学习残差尺度、A5 全全局、A6 同参数 FFN taper、A7 固定 .5 门控。8 层/768/12 头约 75.8M，各组从头训练，先 1 epoch 筛选，再 A0+两个候选 3 epochs×3 seeds；生成与表征等权。细节、研究来源与选择边界见 [架构探索实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

# 当前模型方案

2026-10-28：用户要求1B直接正式长训；28层/1536维/24头/FFN6144、V8 R2和183类family目标保持，随机初始化300000步、global512=8×micro16×GA4，LR1e-4、warmup15000、cosine。保留已跑通的NCCL2.31.2及确定性算法，跳过额外资格和200步测试。原micro8测试已完成200步，micro16只有单卡实测；新长训入口已准备但八卡尚未启动。[正式方案](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/BERT_1B_DIRECT_TRAIN_20261028.md)。

2026-10-27：新增独立1B工程测试版，28层/1536维/24头/FFN6144，实际1,057,514,727参数；保留V8 R2及183类family目标，随机初始化、不继承75.6M权重，200步而非长训。global512、micro8/4/2/1实测、逐层重计算；八卡最长形状DDP与BF16/FP32恢复门禁后才进入测试。尚未GPU执行，不能宣称稳定性或规模收益。[实施与边界](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/BERT_1B_TEST_20261027.md)。

2026-10-27 用户明确nonbinding可以预测family，现取消family目标对relation的过滤，schema升级v3。loss=token_selected_mean+0.1×eligible_chain_family_mean；nonbinding的token目标只含关系符，family头及BERT共享表征正常接收辅助梯度。链存在但无可靠来源注释→unknown；缺失链或无观测残基的合成占位槽→-100，不制造unknown链。例如heavy-only记录没有提供L，β-only没有提供α；VHH没有L。183类和known/unknown分项指标保持。[全部关系family审计](/project_c20250601/251105016/project/dllm_test/output/v8_v_family_labels_all_relations_20261027/audit.json)。

2026-10-27 用户确定显式unknown类别，取代此前缺失/过滤标签一律忽略的策略。V-family schema升级v2，共183类；缺失、歧义、词表外标注的有观测受体链监督unknown，unknown表示注释状态而非生物家族。H/VHH共用第1池化槽，其他槽为L/β/α。缺链和全合成占位链仍-100；nonbinding仍不参与family loss。辅助权重0.1与独立全局有效链归一化保持。验证额外记录known-only accuracy/macro-F1与unknown链数/召回率，避免只看混合准确率。[unknown标签审计](/project_c20250601/251105016/project/dllm_test/output/v8_v_family_labels_unknown_20261027/audit.json)。

2026-10-27 V-family辅助目标已实现，替代此前“尚未加入”的状态：在每条H/L/β/α的观测残基隐藏状态均值上共享一个Linear(d_model,182)，同一次noisy BERT forward完成MLM与family CE。类为species:V-family，human78/mouse72/rat32；小鼠品系合并。仅来源注释通过一致性检查的非负例链参与，VHH没有来源标签则忽略；不新增J头。辅助权重0.1，DDP/梯度累计分别按全局token数与有效链数归一化，空标签rank保持head计算图。记录family loss/accuracy以及固定噪声开发集上有真值支持类别的macro-F1和支持类别数。分类头参数进入检查点及优化器，spec随checkpoint保存。[当前family训练YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_unbinding_v8_vfamily_scratch_300k_8gpu_20261027.yml)。

2026-10-27 V8 R2新增独立随机初始化训练方案，暂按300000步、LR1e-4、5% warmup，原75.6M架构与MLM/负例仅关系符目标保持。已有287000续训包不变；未添加V/J-family辅助分类头或CLIP。family预测拟从各链表征接分类头，属于待实现方案，不能将现有来源注释视为已参与loss。[从零训练配置](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_unbinding_v8_scratch_300k_8gpu_20261027.yml)。

2026-10-27 用户确定下一阶段两项探索：在既有模型上设置有/无Unbinding预训练的对照，沿用一致的数据划分和评测协议，观察结合预测及序列建模表现；另尝试扩大模型参数规模，比较训练稳定性、资源成本和下游结果。作为下一步计划写入汇报，未指定新参数规模、未启动新训练。当前论文展示的BERT247000与Diffusion76500是分别固定的已有评测状态，后者从BERT287000初始化，不能将两行差值当作严格同起点的Diffusion训练增益。[汇报计划原图](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/figures/compact_next_plan.drawio)、[BERT结果身份](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/data/bert_result_availability.json)。

2026-10-27 Protenix探索汇报按当前在线实现单独说明：8层各有独立Pair Encoder，最后一个兼用于最终distogram读出；不能套用历史ESMFold2共享Encoder图。Pair评估按完整原始latent矩阵→同样例教师/学生distogram→实验几何接触的顺序，教师版本与残基映射固定。原始矩阵固定同一通道及共同色标；LayerNorm后的范数接近常数，不用其纹理判断学习。PCA按链/序列位置分组不能证明真实接触，已从当前汇报主分析移出。三个既有Fab仅支持当前案例的接触排序改善，仍弱于教师，独立结构泛化及下游收益待验证。本轮未改模型或训练实现。[矩阵数据与口径](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/data/protenix_matrix/summary.json)、[来源与核验](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/references/protenix_matrix_evidence.json)。

2026-10-26 用户确认模型当前名称为 **Ophiuchus-AIR**，总体路线为：①BERT掩码预训练获得通用序列表征；②继承BERT权重进行Diffusion生成训练；③后续针对抗体Design等任务微调专用模型。架构与训练路线分开在相邻汇报页呈现；第三步是计划，不作为已完成实验或效果结论。本轮仅更新汇报及图示，未启动训练或改动模型实现。[整体思路](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/figures/02_整体思路.drawio)。

<!-- four-relations-20261026 -->
2026-10-26 四类负例采用来源标签与推断错配分开记录：新51668筛选阴性、125154推断。train-only置换保留H/L、α/β及pMHC整体，同MHC组内取样，排除已知正例、重复与heldout，并同步重建交互标识。VHH显式objective保留<nb>/role及单链，参数形状兼容BERT287000；负例仅关系符监督、正例full-token MLM。仍为单位relation-token权重，未实现置信度降权，未宣称模型提升。[四类负例探索、证据与R2入口](/project_c20250601/251105016/project/dllm_test/docs/V8_FOUR_RELATION_NEGATIVES_20261026.md)
<!-- /four-relations-20261026 -->

2026-10-26 V8抗体扩充采用实测重复一致的SPR标签：全部重复无结合才作为严格负例，混合结果隔离；已测构建体按来源固定VH/VL框架恢复，抗原明确标为参考上下文。沿用现有测试保护后新增28条，全部为负例；同来源255条正例因重叠排除，不额外提高单一负类来源权重。负例仍仅关系CE，正例保留MLM；VHH另行清点，保持既有reader边界。[来源、筛选与限制](/project_c20250601/251105016/project/dllm_test/docs/V8_ANTIBODY_NEGATIVE_EXPANSION_20261026.md)。

2026-10-26 Diffusion新增固定框架的单CDR和多CDR条件恢复任务：只在有完整真实区域标注的抗体mask分支内路由，默认chain/singleCDR/multiCDR=50/25/25；目标外受体序列保持可见，loss按选定目标AA长度归一化。CDR任务25%全MASK起点、其余沿用diffusion时间采样；关系辅助loss保留。新增各CDR及多CDR内部验证，原guard指标保留。为给定loop跨度的补全目标，尚无长期生成收益结论。[CDR任务与当前入口](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_FIXED_FRAME_CDR_20261026.md) 当前执行规模按用户要求为4×A10080，global512保持；每卡256互补视图按48×5＋16累积，四rank资格独立核验。

2026-10-26 V8负样本MHC采用与V7相同的来源证据补全规则：明确等位基因、唯一且覆盖一致的参考及pseudo一致性才接受；不从34aa反推等位基因，不猜缺失链。56399条参考class I上下文完成，非实验实测序列。新增MHC不进入负样本序列loss；整条负样本仍仅显式关系CE，正样本保留MLM，模型不变。[V8 MHC补全与当前训练入口](/project_c20250601/251105016/project/dllm_test/docs/V8_MHC_COMPLETION_20261026.md)。

2026-10-26 当前Diffusion修复候选改为原BERT287000重新初始化新Diffusion（step0、AdamW/warmup重置），保留任务/噪声/损失，source采用alpha=.75且最多4倍自然比例；TRAIT份额由14.286%降至0.1251%。独立验证剔除4条train受体精确重叠，885条重新建立guard参考。200k新步、峰值LR2e-5、warmup1000，保留验证降LR与真实clip。此为有审计依据的候选，尚无长期训练收益证据；从76500恢复方案撤销。[数据审计与新提交入口](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_BERT_RESTART_DATA_REVIEW_20261026.md)

2026-10-26 更正：用户要求从原 BERT287000 重新开始 Diffusion。此前 resume76500 仅为保留旧进度的修补方案，已撤销；旧 YAML 与 train.sh 已设置立即退出，尚未提交该任务。新实验准备中：Diffusion step0、新 AdamW/warmup、独立输出，并开展全量数据与验证重叠审计。

2026-10-26 Diffusion梯度回升复核支持先处理固定验证退化：现有日志未显示后期梯度爆炸；TRAIT小来源等权曝光为风险线索。独立候选保留模型、任务ratio、噪声损失和采样，从76500完整状态续接，初始LR上限减半，每3次验证无相对0.1%改善再减半/下限2e-6；控制器状态完整恢复，累计预算200k不变。新增裁剪前后观测并实际使用配置clip。CPU和单卡数值检查通过，八卡及长期验证未执行，不能宣称已改善梯度/泛化。[方案与边界](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_GRADIENT_REVIEW_20261026.md)。

2026-10-26 新增V8 BERT unbinding独立目标：从原始BERT287000逐张量继承，重新初始化优化器/调度器；nonbinding整条记录仅关系token CE，其余序列/EOS/语法不扰动且labels=-100；binding继续原full-token MLM，bind/unbind关系均始终MASK预测；unknown/无标签维持原目标。复用现有词表与75.6M架构、全局selected-token mean、不重采样负样本；probe严格隐藏关系。默认新增13000步，独立schema与冻结包，尚无正式训练收益。[方案与验证](/project_c20250601/251105016/project/dllm_test/docs/BERT_V8_UNBINDING_20261026.md)。

2026-10-25 为便于直接查看 v2 新增内容，将第 3、4 页分别导出为单页原生 draw.io：[Token/Role/Scope 与 h](/project_c20250601/251105016/project/dllm_test/docs/figures/pair_training_simple_20261025_v2_03_inputs.drawio)、[Pair Feature/Bias 与参数共享](/project_c20250601/251105016/project/dllm_test/docs/figures/pair_training_simple_20261025_v2_04_pair_sharing.drawio)。两文件均直接提取自四页 v2，保留原有可编辑对象与连接线。

2026-10-25 架构图字段核对：`FixedChainCollator` 对受体固定槽设置 `scope_ids=0`，图中 Scope 加注此限制；通用 Scope embedding 接口保留，但不能把该槽位画成按 full/fragment 等类别编码。Pair Encoder 接收逐位置 hidden，内部执行单位置投影、成对组合及 Pair MLP；共享版跨层复用这些参数，64→12 的 bias 投影逐层独立。

2026-10-25 按用户要求补充简洁的 [draw.io 可编辑架构图（v2，四页）](/project_c20250601/251105016/project/dllm_test/docs/figures/pair_training_simple_20261025_v2.drawio)：第一页对比原 BERT 与 Pair 联合训练；第二页展开原单层 Attention、FFN、残差与 Pair Bias 接入；第三页展开 Token/Role/Scope 的含义、embedding 相加及逐位置 h；第四页逐个画出五组 64 维特征、Concat、Pair MLP、Bias 投影，并以多层调用图展示共享 Encoder θ 与独立 W1–W8、最终距离读出。图绑定已有 ESMFold2、8 层共享 Pair Encoder、链内 KD 实验。原生框、文字、连接线可编辑，旧文件链接同步更新；[PDF 预览](/project_c20250601/251105016/project/dllm_test/docs/figures/pair_training_simple_20261025_v2.pdf)。

2026-10-25 周报进一步按用户要求删去pipeline图，方法说明只保留ESMFold2路线及50万条已完成训练的结果。此处对应8层共享Pair网络的历史对照，不能将候选每层独立Pair或Protenix跨链方案写成该实验架构；候选实现与原始研究记录不变。[精简后的正文](/project_c20250601/251105016/project/dllm_test/docs/WEEKLY_REPORT_20261025.md)。

2026-10-25 周报方法图口径按用户纠正：feature对齐从ESMFold2教师出发，主图描述干净序列教师、MLM扰动学生、残基映射、链内distogram KL与Pair attention反馈；图内不包含完成/待验证等进度标签或diffusion成绩。Protenix-Mini-ESM仅作为当前测试的候选教师，其跨链及PDE置信度加权单独在正文说明，不表述为正式替换ESMFold2。此为报告表述修正，不变更训练实现或实验结论。[方法图](/project_c20250601/251105016/project/dllm_test/docs/figures/weekly_report_20261025/feature_alignment_pipeline.svg)。

2026-10-25T07:59:05Z 用户明确当前BERT应以表征下游为评价主线。已核对现有评估没有生成：伙伴/恢复属于原MLM head诊断，不能据其低分否定hidden表征；结果解释限定于实际读出。[依据与边界](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINED_MODEL_EVALUATION_REVIEW_20261025.md#stage1-bert-evaluation-boundary-20261025)。

2026-10-25T02:31:26Z 现有最终9770步两模型冻结效果比较三条主线均已执行：GDP、天然伙伴/区域恢复、实验结构检索。结果支持部分表示改善，不能宣称整套Pair＋链内KD在所有下游上更强；原始跨链训练消融未因此完成。推理期伙伴遮蔽与分层诊断不改变模型参数。[完整结论与证据](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINED_MODEL_EVALUATION_REVIEW_20261025.md)。

2026-10-25T01:54:37Z 结构检索增加事后近同源敏感性：90%/95%双阈值全部保留，过滤query及gallery，不重新提特征；候选池变化使跨阈值绝对性能不能直接比较。伙伴任务完成后使用原候选池的query分层诊断，不能声称严格无同源测试。[协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_EXPERIMENTAL_STRUCTURE_RETRIEVAL.md)、[执行](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#stage1-native-partner-20261025)。

2026-10-25T01:28:00Z 现有权重效果验证保持三条主线，主模型全部冻结：GDP固定alpha读出已完成；[伙伴检索](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PARTNER_RETRIEVAL.md)采用八组确定性mask覆盖全部残基，双向原始条件分数及遮蔽伙伴增益完整报告，不冒称单位置PLL；[结构检索](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_EXPERIMENTAL_STRUCTURE_RETRIEVAL.md)使用实验CA、H/L联合框架对齐及同H3长度候选，比较相同全局hidden读出和序列基线。它们是文献启发的自建协议，不重新预训练、不改变历史benchmark，也不照搬paired/unpaired论文收益。[文献依据](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINED_MODEL_EVALUATION_REVIEW_20261025.md)。

2026-10-25 用户明确选择50万条在线蒸馏，教师产物不落盘。已接入逐micro内存目标、干净H/L联合教师、完整链内/跨链KL和PDE≤2Å双向均值降权，权重不进入分母；学生从零8层主干，新计划启用每层独立Pair网络，最后一套读出最终hidden，没有跨层持久Pair state。教师目标/RNG隔离，断点恢复重新计算目标；新源码已通过CPU及两进程Gloo检查，GPU联合资格待执行。当前复用已完成9770步scratch MLM，只新增加权Pair训练；数据、随机主干、mask和预算一致，软件环境差异留待统一终点评测复核，旧MLM权重不用于初始化Pair。未加权KD、Pair但KD=0及科学置信度/独立表征验证仍待做，不能声称已有独立策略收益。[在线方案与状态](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PROTENIX_AB500K_CONFIDENCE_PLAN_20261025.md)。

2026-10-25 通用教师首轮24例GPU工程验证完成，三条路线各8/8通过。当前配置下Protenix-Mini-ESM四配对例batch=1延迟约为ESMFold2的1/2.61，Fab接触AP接近，但1AKE链内质量低于基线；Boltz无MSA较慢且若干通用例退化。下一阶段优先扩大Protenix与ESMFold2的独立结构质量及批处理吞吐对照，纳入相同序列链的等价置换评价。未证明泛化或学生收益，不更换正式教师、学生或缓存；[首轮实测与配置边界](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_GENERAL_TEACHER_VALIDATION_20261024.md)。

<!-- stage1-general-teacher-landscape-20261024 -->
2026-10-24T23:04:07+00:00 通用教师候选扩至AlphaFold、RoseTTAFold、Protenix、Boltz及单序列路线；优先比较ESMFold2现有基线与Protenix-Mini-ESM/Boltz-2的完整联合distogram，RF3追加、AF3备选，trRosetta仅作链内对照。截断坐标分支须保持原trunk和头的输出；不同bin/原子定义或contact/坐标/token不可直接替换。在线免标签落盘将多epoch计算重复，尚无新模型同卡收益实测；未更换正式教师或修改AB100K完整链内＋跨链候选。[源码依据与筛选](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_GENERAL_STRUCTURE_TEACHER_FEASIBILITY_20261024.md)。
<!-- /stage1-general-teacher-landscape-20261024 -->

<!-- stage1-scratch-fast-20261024 -->
2026-10-24T12:36:34.657206+00:00 当前对称Pair新增可配置执行后端，优化包使用bias_implementation=symmetric、autocast_pair_embeddings=true、symmetric_pair_chunk=4096、checkpoint_pair_chunks=false；默认调用方仍dense原行为。网络参数、全部8层共享反馈、链内distogram目标和随机主干不变；BF16梯度累计不宣称逐位相等。原始跨链/其余消融目标保留，优化包不扩大当前监督范围。[优化版启动与核验](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_SCRATCH_FAST_20261024.md)。
2026-10-24T19:08:10.020221+00:00 从零对照已取得共同9000步固定验证：Pair AA CE0.302593、MLM0.303221，Pair相对低0.207%；Pair任务9200步时终止，尚缺共同终点，不将小幅单次差异视为确定结构收益。[只读进度快照](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_scratch_fast_20261024/diagnostics/status_20261024/snapshot.json)。
<!-- /stage1-scratch-fast-20261024 -->

<!-- stage1-contact-residues-20261024 -->
2026-10-24T11:33:55.652866+00:00 教师跨链位置级检查：三个既有Fab的真实接触对能由已有distogram head从Pair表示读出近/远距差异，展开固定同H的接触/远距L位置作图；不能将单通道纹理解释为具体物理作用，也不构成raw Pair蒸馏优于distogram或未见结构泛化证据。[接触残基与256维教师特征](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_CONTACT_RESIDUE_FEATURES_20261024.md)。
<!-- /stage1-contact-residues-20261024 -->

<!-- stage1-cross-chain-diagnostic-20261024 -->
2026-10-24T08:13:09.252819+00:00 完整distogram诊断显示教师在三个既有Fab结构上含有准确H–L接触信息，但当前链内KD学生跨链距离预测差，六条valid跨链KL均退化。伙伴链干预仍显示上下文依赖，不能说模型完全没有跨链信息。补齐原A–F，尤其链内/链内＋跨链KD对照；raw Pair State对齐保持独立可选实验。当前结果不证明新架构或跨链KD必然提升MLM。[完整矩阵、实验接触与Pair特征诊断](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_CROSS_CHAIN_DISTOGRAM_DIAGNOSTIC_20261024.md)。
<!-- /stage1-cross-chain-diagnostic-20261024 -->

<!-- stage1-validation-scope-correction-20261024 -->
2026-10-24 用户明确保留Stage 1原始完整初步验证目标：仅MLM、仅KD、仅Pair反馈、反馈＋KD、链内/链内＋跨链和Mask-aware等对照。已完成AB500K只覆盖MLM与反馈＋链内KD，不能代表跨链蒸馏验证；G/H的ESMC/raw Pair State仍为原文可选项。现有缓存仅保存链内抽样distogram，跨链目标需补生成，容量约束不能静默缩减研究目标。质量筛选与目标精度须显式报告；本次更正文档和核查数据，未改冻结训练包。[目标与覆盖范围](/project_c20250601/251105016/project/dllm_test/docs/Stage1_Multichain_BERT_ESMFold2_Distillation.md)。
<!-- /stage1-validation-scope-correction-20261024 -->

<!-- stage1-ab500k-offline-pilot-20261022 -->
2026-10-22 AB500K离线验证代码及冻结提交包已完成，33项CPU测试通过：保留当前8层Pair反馈，以相同BERT287000比较纯MLM与Pair＋链内KD，各3000步/global512。4个独立四卡任务预计算；FP16＋16位索引＋Zstd分片，保留每链1024对/64bins，总预算150GB。固定4096原valid记录、逐记录同MASK；以共同验证MLM CE及步数/时间比较，不比较MLM与联合总loss。初始checkpoint-0和latest2＋best均可恢复。结构准确性仍unvalidated，尚无训练收益结论；GPU任务由用户提交。[计划与操作](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_OFFLINE_PILOT_PLAN_20261022.md)。
2026-10-23T21:40:24.124992+00:00 AB500K长训预算讨论已核算：500000/global512保留尾批时每epoch977步，5epoch=4885、10epoch=9770、20倍总60000=61.412672epoch；按8卡现速总时长约MLM6.9小时/Pair50小时。建议当前Pair完成3000后，两组匹配续接到总9770先看趋势，主干/Pair峰值1e-5/1e-4、500新增步再warmup为候选；尚未改运行计划或生成/提交长训作业。[候选方案](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_OFFLINE_PILOT_PLAN_20261022.md#长训预算与学习率候选2026-10-23讨论方案)。
2026-10-23T21:57:29.079264+00:00 AB500K中文曲线及机制复核完成：21:51:44 UTC快照MLM3000、Pair800；共同验证750时Pair AA CE比MLM高0.062796%，教师KD由3.414345降至1.304292（−61.80%），尚无MLM收益。checkpoint750的4条CPU探针确认8层反馈已更新且梯度连通，加权KD/MLM主干梯度范数约2.81%，仅为小样本线索。提出KD-only/feedback-only对照及单因素Pair容量、跨层状态候选；未改训练或提交GPU。中文PNG/PDF、158行训练CSV、17行验证CSV及数值核验通过。[Pair特征与中文曲线复核](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_PAIR_FEATURE_REVIEW_20261023.md)。
2026-10-24T06:48:07.307840+00:00 AB500K两组3000步均COMPLETE=success（各约3.07epoch）；固定4096条×3mask终点AA CE：MLM 0.288281123、Pair 0.289064492，Pair相对高0.271738%。KD从3.414345降至0.584299（−82.887%），但两组最佳AA CE仍是step0；本轮没有MLM收益。训练会话耗时MLM20.74分钟、Pair150.52分钟（7.26倍，不含教师生成及启动资格）。已刷新[完整中文曲线](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/loss_completed_20261024/loss_explained_zh.png)及[共同终点数据](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/loss_completed_20261024/summary.json)。仅本地读取与绘图，未改训练或提交作业。
2026-10-24T07:09:47.671073+00:00 Pair架构探索完成：对照Graphormer、Uni-Mol、ESMFold v1、AF2/AF3原论文及官方代码，区分跨层权重共享、Pair状态传递、recycle复用。当前共享动态bias缺少显式跨层z状态；共享本身未证实是退化原因。优先候选为低成本持久Pair状态与独立层级更新，并保留仅解除共享的对照；后者按指定构造估计总76.58M，不天然加速。候选未实现或提交训练。[架构对照与实验设计](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PAIR_ARCHITECTURE_EXPLORATION_20261024.md)。
2026-10-24T07:24:38.997184+00:00 按用户要求准备AB500K从零训练：两组旧学生权重继承0，直接随机初始化seed42，共享78个state张量逐值一致，2条真实样本初始logits一致；MLM/Pair参数75.60M/75.73M，保留当前共享Pair架构及已验收缓存。新独立包默认9770步=10epoch、主干/Pair LR均1e-4、warmup500；23项CPU测试、两组84分片/缓存入场、冻结源码和shell/YAML核验通过。GPU profile/8rank短测待用户提交，本轮无GPU或云端submit。[新计划与两个启动命令](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_SCRATCH_PLAN_20261024.md)。
2026-10-24T07:47:46.897119+00:00 Pair速度诊断确认主要成本为每层稠密反馈，权重共享并不消除重复计算。当前对称特征可只算上三角再还原，叠加提前dtype转换的实测单卡更新速度约1.7倍；关闭Pair重计算约2.05倍、allocated50.11GiB。保持参数/监督及当前架构，有限批次logits/loss逐值一致、最大参数组梯度相对L2约1.41%；不声称长训或多卡等价。编译候选未过既定数值门槛，不默认采用；正式接入前另做八卡profile和恢复资格。[性能诊断与候选方案](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PAIR_SPEED_EXPLORATION_20261024.md)。
2026-10-24T08:04:15.397210+00:00 按用户“每层一个Pair网络”补充官方源码复核：ESMFold ModuleList、AF2/AF3 layer_stack均支持层级独立参数，并持续传递pair状态；Uni-Mol提供较轻量的逐层关系更新参照。建议独立PairUpdate＋残差z作为下一版候选，保留仅解除共享的U对照；CPU实例化核对8个encoder均120832、合计966656参数且存储独立，指定U总模型约76.58M。已将旧文档的父BERT初始化建议改为遵循用户要求的同一随机主干。独立权重不自动提速；有向/三角更新不能照搬对称加速。仅研究与CPU参数核算，未改训练或提交任务。[参考及适配方案](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PAIR_ARCHITECTURE_EXPLORATION_20261024.md#8-针对每层有一个pair网络的官方实现复核)。
2026-10-24T20:50:34.352551+00:00 原生Pair容量已核算：教师256维pair＋64维distogram，扫描504096条请求真实长度，500k训练平均H+L231.503。全有向L²、最终一份教师状态、16位未压缩本体为训练17.157TB、加验证17.297TB；FP32两者均原样存储则翻倍。合格OAS训练池实际2160824条，外推约74.145TB；400万同长度场景约137.253TB，非已核准全量AB计数。4096有向位置抽样为另选项，不静默替代全位置目标；学生8层共用教师最终目标无需8份缓存。未生成数据或提交GPU。[容量依据与范围](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PAIR_FEATURE_STORAGE_20261024.md)。
2026-10-24T21:16:05.107257+00:00 AB100K仅distogram候选已设计：10万训练＋2000dev＋2000test按既有长度外推FP16全方阵713.72GB，对称验收后上三角358.40GB，保留链内/跨链完整位置。候选A/B/C/D区分MLM、KD和独立层反馈；同随机主干、20epoch3920步筛查及50epoch9800步延长候选。主证据为冻结hidden的新实验结构接触探针和功能探针，KD仅诊断；原Intra/Inter及Mask-aware验证范围保留，严格划分/教师质量/三seed须实际执行。尚未抽样、生成缓存、实现新模型或提交作业。[AB100K候选方案](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB100K_REPRESENTATION_PLAN_20261024.md)。
<!-- /stage1-ab500k-offline-pilot-20261022 -->

<!-- stage1-online-20261022 -->
2026-10-22 用户选择Stage 1按需教师蒸馏：原始未加MASK的完整真实链进入冻结ESMFold2；缺缓存时生成原生distogram目标并持久化，相同请求后续复用，学生继续独立随机MASK。沿用BERT287000、全部8层Pair反馈、global512、自然比例采样、初轮10k步及固定验证pilot；完整beta可做链内KD，片段/合成链保持MLM-only。在线模式移除预生成训练缓存覆盖门槛；教师结构准确性保留unvalidated，作为明确的探索性预训练。设备改为教师/学生同卡分进程常驻，提交入口自动重新profile及验证在线恢复。[实现与实验边界](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_ONLINE_DISTILLATION_20261022.md)。
<!-- /stage1-online-20261022 -->

<!-- stage1-cache-compression-20261022 -->
2026-10-22 用户要求先分析缓存压缩，未授权实际切换。候选优先固定容量缓存，满额淘汰/再次使用重算，保留当步全部合格样本KD；本轮仅0.819个epoch，永久保存全部请求的本轮复用收益有限。紧凑FP16＋范围检查的16位索引＋可逆字节重排＋Zstd样本外推约715GB；减少每链pair另属监督消融。保留64 bins，缓存codec/校验需版本化，冻结运行包不热改。[压缩候选方案](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_ONLINE_DISTILLATION_20261022.md#缓存压缩候选方案2026-10-22仅分析)。
<!-- /stage1-cache-compression-20261022 -->

<!-- stage1-lmdb-feasibility-20261022 -->
2026-10-22 LMDB候选已做本地64条压缩读写/重开往返验证；LMDB负责KV组织，FP16/字节重排/Zstd负责缩减，尚未接入训练。实际持久化目录为共享GPFS，结合LMDB官方远程文件系统约束，建议共享盘压缩分片＋索引，本地磁盘可选有界LMDB；8教师推理在写事务外，单environment短事务writer。容量约束覆盖各存储层，满额保留当步KD、复访重算。[LMDB候选设计](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_ONLINE_DISTILLATION_20261022.md#lmdb候选存储与实际挂载约束)。
<!-- /stage1-lmdb-feasibility-20261022 -->

<!-- stage1-cache-alternatives-20261022 -->
2026-10-22 缓存其他路线调研：共享盘优先候选为独立Zstd记录＋不可变分片＋偏移索引，次选Zarr v3 sharding；HDF5按文件单写/离线封存，RocksDB需单进程服务或拆库，均未进行本项目性能验证。存储容器、编码精度和固定容量分别评估；先维持pair数量及64 bins，比较FP32无损与FP16编码，去重需实测重复率和重新明确教师seed/位置映射契约。仅更新候选设计，训练不变。[其他路线与验证方向](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_ONLINE_DISTILLATION_20261022.md#lmdb之外的探索路线)。
<!-- /stage1-cache-alternatives-20261022 -->

<!-- stage1-timing-diagnosis-20261022 -->
2026-10-22 在线Stage1速度诊断：教师每卡逐条串行、批内约45个请求，近千残基输入单条73–77秒造成跨卡等待；前三步教师准备区间约占墙钟87–89%，为日志/metadata重建而非GPU profiler。后续候选优先低频分段计时、保持global batch成员的教师负载调度/预取和短输入批推理；仅压缩磁盘缓存无法解决首次推理主成本。任何调度/attention实现/数值变化需重新核验；现有运行未改。[证据与候选](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_ONLINE_DISTILLATION_20261022.md#训练速度诊断2026-10-22)。
20:01Z核对源码/config SHA与运行契约一致：教师80层ESMC仅执行一次；num_loops=3实际4轮，每轮4＋24层pair更新，最后2层，共114次共享pair block执行。triangle更新含O(L³)收缩，是长输入计算成本的重要候选来源；ESMC与trunk耗时占比待分模块计时，未改变循环次数或模型实现。[计算结构核对](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/online/log_review/teacher_compute_audit_20261022.json)。
20:16Z核对官方循环入口：绑定Fast模型卡示例显式3，而同revision配置默认20，当前官方仓库/SDK也使用或默认20；参考实现与本地均按num_loops+1执行。当前训练显式3覆盖20，与Fast快速示例一致，仍需独立验证目标质量；未改循环设置。[官方来源核对](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/online/log_review/official_loop_settings_20261022.json)。
<!-- /stage1-timing-diagnosis-20261022 -->

<!-- stage1-training-build-20261022 -->
2026-10-22 Stage 1训练采用BERT287000初始化、全部8层Pair反馈、原full-token MLM与链对归一化KD；全量自然比例无放回采样，与diffusion等来源双视图配方分开。实现分组LR、KD warmup和独立MLM/结构验证；新模型必须按2048长度及稀疏pair上限重新profile。初始10k/global512等值为候选，尚未验证收益；跨链监督仍需来源和质量依据。[Stage 1训练构建与提交](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINING_PIPELINE_20261022.md)。
<!-- /stage1-training-build-20261022 -->

2026-10-22按用户反馈将Stage 1架构主图重画为左侧整体架构、右侧A/B/C/D模块内部特征线路，放大字号，显式标出Q/K/V分头、残差、FFN双路门控、五路Pair特征与两个输出头的张量形状。原固定槽位／教师对齐图作为补充保留；本次仅修改图件与架构说明，模型行为未改动。[图文说明](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_STRUCTURE_IMPLEMENTATION_20261022.md#当前实现的架构图) · [特征流主图PDF](/project_c20250601/251105016/project/dllm_test/docs/figures/stage1_structure_architecture_20261022_feature_flow.pdf)。

2026-10-22 Stage 1全层结构增强已完成本地工程验收：full-token BERT287000全部参数继承，共享Pair MLP在全部8层生成内容相关attention bias，出口零初始化，保留原链内／全局交替、RoPE和0.1残差。原MLM＋链对归一化distogram KL已用真实ESMFold2-Fast缓存完成联合更新和精确恢复；权重／环境／数据契约见[实现记录](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_STRUCTURE_IMPLEMENTATION_20261022.md)。后续比较A原MLM、B无反馈KD、C反馈无KD、D反馈＋KD；当前不推断预训练或扩散生成收益。

2026-10-19 CoV-UniBind最新约束为作者原划分；恢复论文lineage/抗体分组与20/4门槛，不以全局分折或精确抗原序列组替代。原始fold缺口显式登记，新增方向未启动。[协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md#covunibind-author-split-audit-20261019)。

2026-10-19 用户确认CoV-UniBind改为全部lineage合并、每任务每折共享Ridge、全局抗体双链隔离，整体OOF计分；旧per-lineage口径仅作历史参考。[当前协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md#covunibind-pooled-5cv-20261019)。

2026-10-20按用户要求将BERT287000→diffusion/edit扩展至累计200000步；已从latest10000恢复模型、AdamW、RNG和数据游标并实际续训。任务／数据／gamma0／global512沿用；LR从旧终点2e-6经过1000步升回2e-5，再cosine衰减至累计200000步。原best2500保留，13项CPU及八rank恢复／重放核查通过。[配方、接续调度与检查](/project_c20250601/251105016/project/dllm_test/docs/BERT_TO_DIFFUSION_EDIT_20261019.md)。

2026-10-19 CoV-UniBind PLM比较协议更正：已核实论文Ridge 5CV，历史Ours标为非同协议且不作headline；已完成8组Ag+H+L Ridge五折评估及独立验收，详见[执行记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#covunibind-aghl-ridge5cv-20261019)。见[协议核查与重评要求](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md#covunibind-5cv-20261019)。

2026-10-17 GDPa1层级诊断支持中间层读出作为待验证方向：HIC层差经float64与训练折内尺度控制仍存在，但属性间表现不同。保留现有正式末层/global默认，不在已检视测试折上选层；旧V5测试暴露已确认，未来泛化结论须绑定实际去重语料和独立选点。[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#gdpa1-layers-results-20261017)、[数据审计与解释边界](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#ab-exposure-and-layers-20261017)。

2026-10-17 GDPa1继续保留原生最终decoder/global默认；AirGen方法对齐诊断未支持改成分链拼接来修复HIC。回归实现一致性和未决机制见[协议核查](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md#gdpa1-airgen-audit-20261017)、[诊断结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#gdpa1-airgen-audit-results-20261017)。

2026-10-17 最新用户决定：BERT 改为全有效 token MLM，除 `<prots>` / `<protd>` 外，真实受体／抗原／peptide／MHC 残基、类型、relation、分隔符、EOS 和 `<null>` 均可选中预测；PAD／合成 MASK／空槽位没有真实标签。保持 15%＋80/10/10、V7 多来源自然比例及 300000 步，从头初始化新 run。结合分类探针隐藏 relation，避免标签泄漏。此前 AA-only 运行已停止，68000 完整权重保留。[当前配方与评估约束](/project_c20250601/251105016/project/dllm_test/docs/BERT_FULL_TOKEN_PRETRAINING_20261017.md)。

2026-10-16 来源策略复核：旧fixed-chain探索／全量LoRA确实使用等来源采样，LoRA完成的33981步是balanced coverage epoch；当前V7 BERT采用全量自然比例无放回遍历，每轮12207步，两者不能混用。实际冻结sampler的CPU八rank全轮重放确认6,249,956条各出现一次，尾批484条、恢复前缀与源计数均正确；冻结代码／索引哈希匹配。当前仍为全局selected-token mean，新增温度／record候选未启用。[复核与历史证据](/project_c20250601/251105016/project/dllm_test/docs/BERT_SOURCE_BALANCING_RESEARCH_20261016.md)。

2026-10-16 多来源平衡调研：基于当前V7冻结训练索引，OAS+OTS占68.05%记录／95.27%真实受体AA；TCR repertoire占29.25%记录／2.86%真实AA。当前全局selected-token mean让长链得到更大CE系数，不能仅靠等源采样解决。建议独立比较自然／有4x上限的alpha=.75温度采样与token mean／半token半record目标的2×2消融；record目标按0.15×真实AA数归一化，保留零选中记录语义。DoReMi、DoGE、RegMix与ESM簇采样为后续路线，尚未验证收益。只是候选方案，没有修改或提交训练。[统计、论文与实验方案](/project_c20250601/251105016/project/dllm_test/docs/BERT_SOURCE_BALANCING_RESEARCH_20261016.md)。

2026-10-16 历史 AA-only 配方（目标已由上方全 token 配方替代）：BERT式表征预训练改用新清理并补全MHC的V7，75.6M随机初始化、全参数更新，目标恰好300000 optimizer steps。train6,249,956、global512、每轮12207步，共24完整epoch＋第25轮7032步（24.576轮）；warmup15000＋cosine。AA-only动态15%选中、80/10/10恢复和链内／全局交替主干保持；V5 10-epoch启动已中止，不继承其权重。MLM验证改用V7；外部关系开发探针按V7测试保护规则重筛，仅TCR papers/TRAIT两来源有足够双类样本；它们只评估、不给主干梯度，valid受体与V7训练全量精确隔离。AB表征需后续独立下游验证，当前两来源macro AUROC不代表AB效果。详见[方案](/project_c20250601/251105016/project/dllm_test/docs/BERT_RESIDUE_EDIT_PRETRAINING_20261016.md)。

2026-10-16 早期方案（预算已由上述用户授权10epoch取代）：A1，自己的双向 75.6M 从头全参数训练，动态 15% 选中＋80/10/10，选中受体 AA 上 CE；AdamW 1e-4、WD0.01、warmup5%＋cosine，八卡 global512，先按 3 个完整无放回 epoch 验收。按既有 V5 manifest 估算约 44,691 步，新 V7 仅另列 36,621 步的版本对照，未自动换数据。用冻结表征开发任务选点，增加替换比例为后续同预算对照。本段是启动前的历史建议；当前实施状态以首段及PROJECT_PROCESS为准。[完整首轮配方](/project_c20250601/251105016/project/dllm_test/docs/BERT_RESIDUE_EDIT_PRETRAINING_20261016.md#first-bert-run)。

2026-10-16 GIDD 原方法补充核查：完整方法需要时间相关的 MASK／uniform 混合过程、加权 KL＋IS 目标和对应反向转移，不能等同于随机替换＋普通 CE。原文混合噪声改善生成与自纠错，但语言理解成绩略低于纯 MASK；因此当前表征配比仍以冻结下游验证选择，不因论文名称直接扩大替换比例。作者源码与论文动态权重存在表达差异，已按版本记录。[原方法与适配边界](/project_c20250601/251105016/project/dllm_test/docs/BERT_RESIDUE_EDIT_PRETRAINING_20261016.md#gidd-original-method)。未切换训练。

2026-10-16 用户选择先训练表征模型，阶段 A 就加入随机残基修复。建议复用双向 75.6M 主干，以 15% 选中＋80/10/10 建立 BERT 式基线，再比较提高替换比例；选中位置统一恢复原残基，同时验证冻结表征、修复率和误改率。LLaDA2.1 在 CPT/SFT 都使用 M2T/T2T，但具体蛋白配比为我们的待验证设计。主实验沿用多来源范围，OAS-only 保持独立对照。当前完成方法核查与方案记录，新训练代码、冻结启动配置及作业尚未实施。[专项方案与原文依据](/project_c20250601/251105016/project/dllm_test/docs/BERT_RESIDUE_EDIT_PRETRAINING_20261016.md)；后续生成／插删等候选见[完整调研](/project_c20250601/251105016/project/dllm_test/docs/REPRESENTATION_TO_EDIT_GENERATION_REVIEW_20261015.md)。

2026-10-15 已实施 MHC 数据补全：沿用 167/135 受体槽位和 receptor-only loss，对 67,648 条现有记录恢复有来源证明的双链 MHC 条件，其中 34 条为 MHC-II。未引入 pMHC-only 辅助目标或 IEGR，后两项仍为独立研究建议；已有冻结训练不自动迁移。[实现与验收](/project_c20250601/251105016/project/dllm_test/docs/V7_MHC_COMPLETION_20261015.md)。

2026-10-15 DecoderTCR 调研建议：先按覆盖一致性重审双链 MHC 参考映射，保持受体 167/135；再独立评估 peptide/MHC 辅助 MLM 与固定 CDR 区域的熵排序/随机块重采样。现有 receptor-only loss 不能直接使用无受体 pMHC 语料，原生 EOS 生成也不等于论文固定长度 IEGR。论文 ESM2 与官方 V0.3 ESMC 的训练/评测口径分开；以上为待实施建议。[依据与适配方案](/project_c20250601/251105016/project/dllm_test/docs/DECODERTCR_RESEARCH_20261015.md)。

2026-10-15 新增 OAS-only scratch 对照：用户指定 queue012 单节点八卡，随机初始化、全参数训练；复用原 fixed-chain 75.6M 冻结实现及 300k / global512 / LR1e-4 调度，训练和验证来源均限制为 OAS。模型代码逐文件与原冻结版一致，数据选择写入每个 checkpoint 的 provenance；[完整设置及版本对应](/project_c20250601/251105016/project/dllm_test/configs/models/v5_fixed_chain_75m_oas_20261015/README.md)。

2026-10-15 V7 数据范围已更正并验收：从 V5 保留 binding 与未标注／unknown repertoire，仅按标签排除明确 nonbinding，再对实际 test/query 清理。新数据 6,318,945 条，当前 design/fixed-chain 与 CDR3β scaffold 原样保留；旧 194,237 条严格 binding 输出撤回为当前候选。[数据与验收](/project_c20250601/251105016/project/dllm_test/docs/V7_FROM_V5_REBUILD_20261015.md)。

2026-10-15 推理研究优先级：现权重先研究remask纠错与自适应提交，再评估近似缓存；可靠多步教师形成后才考虑蒸馏。连续latent flow为独立训练路线，不替换当前离散decoder。[依据与实验设计](/project_c20250601/251105016/project/dllm_test/docs/INFERENCE_FLOW_DIFFUSION_REVIEW_20261015.md)。

2026-10-15 CDR解码首轮诊断：原AirGen为不可回退的逐步提交；归一化置信度未证明稳定改善，生产默认保持。后续须用独立验证集评估排序与纠错策略，[机制与决策](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#cdr-steps-debug-20261015)。

2026-10-15 fixed reference轻链配对评测默认切换为已验收的AirGen Gumbel/margin v2；范围、生产入口及复现约束见[任务协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md#86-fixed-chain--v7-采样器对齐复现)。

2026-10-14T12:22:38.287065+00:00 正式后训练准备：85原始分片全哈希一致；全量索引保留7,626,748条，排除与原valid完全重合的137条受体记录，原JSONL未改。实际全量loader及7来源collator检查通过。新增八卡native Trainer入口复用已有完整optimizer/scheduler/8-rank RNG检查；任务混合与LoRA的CPU中断恢复均与连续训练逐参数、逐优化器状态一致。尚未选定最终方案，正式启动包不能在确认决策缺失时构建。

2026-10-14T12:00:53.833720+00:00 八模型冻结表征诊断已排在r7长程训练之后；每来源/类别最多256条训练记录，C仅在训练内受体分组划分选择，512条screen记录报告来源AUROC与训练标签打乱对照。通过两项冻结测试（关系标签翻转不改变输入；screen标签不参与选C）。不访问确认/benchmark测试集。

2026-10-14T11:55:14.879733+00:00 确认评测实现已完成准备：保存逐记录统计，按来源内受体tuple聚类与配对训练种子做5000次bootstrap，四候选Bonferroni校正；主指标为全mask、条件补链、CDR三域AA-CE相对匹配continued的平均变化，另设来源/部分mask/生成有效率与长度退步门限。阈值在打开确认集前记录，尚未运行确认评测。 [规则](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/selection_protocol.json)。

2026-10-14T11:43:59.633330+00:00 r6七组均通过真实CUDA前后向并已产生更新；r7五方案×三种子（42/43/44）共15组、各3000步已排队。方案为continued、task_mix、observed_eos、observed_cdr、LoRA；通过53项冻结测试及子计划哈希门禁，确认集未打开。

2026-10-14 新增弱标签后训练对照：全遮蔽条件下比较观测残基平均NLL相对固定112000参考的改善，β=1/10、固定随机方向负对照和正例SFT；均以0.1辅助权重叠加observed_cdr重建。该目标为denoising surrogate，不等同序列DPO或验证过的结合奖励。原始架构/输入布局保持，冻结后单独实验。

2026-10-14 第二轮八组后训练对照已部署到现有容器队列：仅真实观测终点监督、CDR-aware masking、固定112000教师四步到单步分布蒸馏、隐藏关系token的结合标签辅助分类及组合。旧输入布局/词表/容量保持，训练额外mask与label不进入denoise；同时保留legacy与observed-only验证视图。尚未最终选型。[回执](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/enqueue_r5.json)。

2026-10-14 保持当前V5固定双链输入，改用112000权重开展12组架构/后训练对照；短程结果已开始落盘，尚未选定最终方案。必须区分单步恢复和多轮生成，并补充synthetic模板终点与真实观测终点监督对照、CDR-aware SFT及蒸馏研究。后续多种子/独立确认/下游验证后才部署最终训练。[完整方案与研究来源](/project_c20250601/251105016/project/dllm_test/docs/FIXED_CHAIN_EXPLORATION_20261014.md)。






2026-10-13 用户最终选图：仅保留英文 a 图（输入 → 去噪模型 → 序列输出），去掉单层结构与注意力拓扑子图。交付 [原生 draw.io 文件](/project_c20250601/251105016/project/dllm_test/docs/figures/research_refined_architecture_a.drawio)、[PNG](/project_c20250601/251105016/project/dllm_test/docs/figures/research_refined_architecture_a.png)、[SVG](/project_c20250601/251105016/project/dllm_test/docs/figures/research_refined_architecture_a.svg)、[PDF](/project_c20250601/251105016/project/dllm_test/docs/figures/research_refined_architecture_a.pdf)。draw.io 包含 156 个独立可编辑对象，未嵌入整张图片；XML 结构校验通过，当前环境未进行 draw.io 浏览器渲染验证。[导出脚本](/project_c20250601/251105016/project/dllm_test/scripts/export_research_architecture_drawio.py) 复用英文绘图布局。

## 固定双链槽位恢复（2026-10-15，新 V7 已实施）

用户再次确认新数据使用先前的 **167/135** 槽位（含EOS），残基上限166/134。新V7配置的 `full_capacity=304`，包含两链槽位及dot/protd；配置显式保存在新数据目录的 `immune_config.json` 并由manifest绑定SHA256。全量扫描原no-negative V7的6,318,945条，隔离4条超长训练记录，保留6,318,941条，不截断序列。新入口与验收见[重建说明](/project_c20250601/251105016/project/dllm_test/docs/V7_FROM_V5_REBUILD_20261015.md)。既有冻结训练配置维持原绑定。

以下为2026-10-13的历史记录，槽位缩短已在上述新V7派生版落实：

用户要求先记录此项，当前训练继续使用187/154槽位。

- 修改方向：缩短H/β、L/α的固定槽位，拟恢复为 **167/135**（均包含EOS，可容纳166/134个残基），避免为极少数超长记录增加全部样本的计算开销。
- 历史依据：[原始统计](/project_c20250601/251105016/project/dllm_test/scripts/chain_length_stats.json)的抗体H/L P99为165/131，8对齐建议为168/136；历史encoder含CLS，decoder对应167/135。该统计为历史混合来源，不能当作当前V5分位数。
- 当前覆盖证据：[V5初始容量审计](/project_c20250601/251105016/project/dllm_test/output/v5_fixed_chain_f011_20261012/initial_capacity_audit.json)中，7,729,193条train+valid记录仅5条超出167/135槽位：抗体H 1条、TCRβ 1条、TCRα 3条，约0.00006469%。现行187/154是为全部容纳这5条而扩大。
- 实施前明确超长记录的处理方式并保留按来源/split的计数；不能静默截断。过滤或单独处理尚未落实。
- 此项仅为后续修改记录，未修改运行中的源码、训练配置、权重、数据或作业；尾部重复EOS监督是另一项讨论，不随槽位缩短自动变更。

## 既有方案与实现记录

2026-10-13 架构图英文重绘：按用户反馈，以 DPLM / EvoDiff 的序列方格、网络堆叠与局部展开方式替代说明卡片。新版只展示原 A4 / `f011` 的输入、去噪网络、输出、单层结构与注意力拓扑；具体机制未变，示意序列不代表实验生成结果。首选 [英文论文版 PNG](/project_c20250601/251105016/project/dllm_test/docs/figures/research_refined_architecture_paper.png)、[SVG](/project_c20250601/251105016/project/dllm_test/docs/figures/research_refined_architecture_paper.svg)、[PDF](/project_c20250601/251105016/project/dllm_test/docs/figures/research_refined_architecture_paper.pdf)；[图注与参考](/project_c20250601/251105016/project/dllm_test/docs/figures/research_refined_architecture_paper.txt)，[绘图源码](/project_c20250601/251105016/project/dllm_test/scripts/draw_research_refined_paper.py)。上一版中文教学图保留供对照。

2026-10-13 原探索架构图：按用户指定的 `research_refined / FactorialModel` 绘制 A4 / `f011` 示例，覆盖输入序列与元数据、随机遮蔽、8 层局部/全局交替 Transformer、单层残差结构、训练 loss 和迭代生成输出。图中全长/片段容量为 256/64，局部半径为 32；这些是原探索配置，当前固定双链训练方案仍以上下文对应配置为准。提供 [PNG 预览](/project_c20250601/251105016/project/dllm_test/docs/figures/research_refined_architecture.png)、[可编辑 SVG](/project_c20250601/251105016/project/dllm_test/docs/figures/research_refined_architecture.svg)、[矢量 PDF](/project_c20250601/251105016/project/dllm_test/docs/figures/research_refined_architecture.pdf) 和 [绘图源码](/project_c20250601/251105016/project/dllm_test/scripts/draw_research_refined_architecture.py)。图示依据原 checkpoint-50082 配置以及 `v3/layout.py`、`research/model.py`、`research_next/model.py`、`research_refined/model.py`、`research_refined/factorial.py`；仅新增图和文档，不修改模型实现。

2026-10-13 架构代码可读性重构：工作区 [fixed_chain/model.py](/project_c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/fixed_chain/model.py) 集中定义单层 Transformer、模型初始化、去噪前向、注意力规则和训练 loss；`FixedChainModel` 直接继承 `nn.Module`，移除对历史 research 模型继承链的依赖，继续复用 LLaDA 基础模块及原遮蔽采样器。模型数学定义、参数名/顺序、checkpoint schema 与数据布局保持；当前训练仍使用原冻结源码。阅读顺序及官方设计参考见[当前版本说明](/project_c20250601/251105016/project/dllm_test/configs/models/v5_fixed_chain_f011_75m_20261012/README.md)。CPU 等价性检查包含原有 13 项及新增 16 项测试；完整核对记录见 [verification.json](/project_c20250601/251105016/project/dllm_test/output/model_readability_refactor_20261013/verification.json)。

2026-10-13 任务设计更正：模型缺少epitope/结构能力时保留官方目标与数据划分，按输入能力标注差异；CHIMERA改用官方H3序列任务，见[CHIMERA现行协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。

2026-10-13 评测协议决定：遵照用户要求保留原始评测记录，不通过去重改变样本权重；AB–Ag实现见 [AB–Ag 当前原始行协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。

2026-10-12 新版本已正式运行：75,601,176参数，V5训练集7,626,885条；残差0.1、双链独立RoPE、链内全注意力/全局交替，固定槽位187/154。300000步/global512=8×micro64×GA1，LR1e-4/warmup2000/cosine，随机从头初始化。2026-10-12T18:23:05.641727+00:00核实已到65步，最近记录训练loss=2.9047，梯度有限；13项CPU测试、8/8整模GPU检查、8/8 NCCL、85分片全哈希门禁通过。每1000步验证/保存，首次验证尚未到达。证据：[启动验收](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_300k_b512_supervision/startup_acceptance.json)。旧270M已按用户要求停止（最后约72279步，完整保存72000保留），原V7 f011此前已完成。

以下object302/270M说明属于此前已启动版本；新方案以以上配置为准。

更新：2026-10-10T08:12:07.932612+00:00

本轮采用**V5数据集＋约270M新模型**（实际268,590,112参数）；前一轮75.6M已完成。选择V5数据不等于恢复旧V5的ESMC、大词表或旧模型权重。

| 项目 | 当前决定 |
|---|---|
| 数据 | /project_c20250601/251105016/project/dllm_test/data/prepared/immune_v5_receptor_completion |
| 范围 | 常规抗体H/L、TCR；不纳入VHH |
| 训练预算 | 300,000更新步；global batch512=8卡×micro64×累积1；LR1e-4、warmup2000、cosine；八卡synthetic显存检查已通过 |
| 初始化 | 随机初始化，无ESMC |
| 主干 | 16层、1024维、16头、FFN 4096，紧凑50词表；268,590,112参数 |
| 残差 | 系数1.0，取消原0.1缩放 |
| 注意力 | 偶数层蛋白对象内全注意力，奇数层全局；对象由prots/protd划分 |
| 输入顺序 | V5对象布局：context→关系→受体，双链用`.`分隔 |
| 生成 | 对象容量302（原固定decoder总槽位167+135），模型预测`.`及`<protd>` |
| 长度特征 | 对象内位置连续，`.`只分隔输出序列，不切换attention段或重置RoPE |
| CDR-only | 保留已构建全长scaffold；synthetic位置输入mask、不计损失 |

当前配置：/project_c20250601/251105016/project/dllm_test/configs/models/v5_noesmc_270m_20261010/research_config.json；参数预算与数据身份见同目录training_config.json及version.json。下方带2026-10-07日期及launch_v1/v2/v3的说明属于75.6M历史记录。

基础实现原有97项CPU测试、V5全部7,729,193条布局审核保持；270M新增五项门禁CPU检查及八卡synthetic通过，启动时85分片完整哈希通过，现已从头开始正式训练；270M已提交验证最佳32000步的37项下游评测，尚无完成成绩；状态见PROJECT_PROCESS.md。报告 /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/full_audit.json，启动入口 /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v1/README.md。

支持整体生成及已知前链→生成后链；已知后链反向补链需另定条件协议。实验记录与更正只维护在[实验记录](/project_c20250601/251105016/project/dllm_test/docs/EXPERIMENT_LOG.md)，不在本文件重复堆叠过程日志。

详细接口：[configs/models/v7_v5_layout_chain_global_20261007/README.md](/project_c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/README.md)。旧方案与完整推导：[完整历史](/project_c20250601/251105016/project/dllm_test/docs/archive/project_docs_20261007T061603Z/BIOSEQ_MODEL_PLAN.md)。

运行配置：/project_c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/training_config.json；完整启动参数：/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v1/launch_training_args.json。步数以历史下游150,000步为参照翻倍，样本处理量为原150,000×256的四倍；首次直接提交启动失败，后已在持续八卡容器内从头开始正式训练。

2026-10-07启动整合：当前入口为 /project_c20250601/251105016/project/dllm_test/train_jobs/v5_object302_scratch_8gpu_20261007.yml，源码包 /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v2。初始化明确为随机从头训练，所有模型参数可训练；新增split_loss/split_accuracy用于单独统计`.`，总验证CE仍覆盖全部合法监督token，不改训练loss。

运行超参数以 /project_c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/training_config.json 为准；当前参数清单逐项匹配。launch_v3为当前容器内启动版本，保留训练目标、模型主体、数据及300000/global512配置，已通过八卡synthetic和通信启动检查并开始正式训练；已完成多轮验证/保存；最新步数与验收见 [PROJECT_PROCESS.md](/project_c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。

2026-10-07历史loss对照口径：当前V5新no-ESMC与已定位旧V6 binding-only no-ESMC的数据、目标token、遮蔽协议、loss聚合及样本预算不同，暂不据原始曲线更改架构或宣称收益。训练loss更低和全遮蔽验证loss更高均不足以直接排名；同口径对照要求与证据见 [V5_CURRENT_VS_LEGACY_NOESMC_LOSS_20261007.md](/project_c20250601/251105016/project/dllm_test/docs/V5_CURRENT_VS_LEGACY_NOESMC_LOSS_20261007.md)。本次只读对照，不新增评测任务。


### 2026-10-09 评测修复

下游单β生成采用显式单链词表约束（禁止dot）的独立v2评测协议；这是推理协议修复，不是训练架构/布局改动。训练中的蛋白对象内attention、可学习split与protd保持原样。

## 2026-10-10 用户要求扩大到270M

新实验使用16层、1024维、16头、FFN4096，CPU实例化核实268,590,112个可训练参数（约270M）；旧模型实际75,601,176参数。保留50词表、no-ESMC、V5数据、对象容量302、对象内/全局交替attention、残差1.0、原mask目标及验证口径，从头初始化。配置 /project_c20250601/251105016/project/dllm_test/configs/models/v5_noesmc_270m_20261010/research_config.json。

训练预算保持300000步/global512/LR1e-4/warmup2000/cosine，候选micro64×GA1×8卡（同global预算下最大micro）；先在原持续八卡容器验收整模synthetic前后向/显存与通信，再用独立票据正式启动。新数据绑定门禁仅将架构固定值切换为新尺寸，85分片内容哈希、原布局审核与training_ready停线保持；未重写数据审核或改旧冻结包。原75.6M产物保留，用于同口径对照。

270M实际已开训；以 /project_c20250601/251105016/project/dllm_test/configs/models/v5_noesmc_270m_20261010/training_config.json 和运行包 /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_270m_train_bs64_20261010 为当前入口。micro64/GA1保持global512，GPU验收和原生数据门禁均通过；首个训练日志loss有限。模型维度以外的数据、目标、注意力模式、验证采样及总训练预算继承75.6M；微批次/累积变化需在比较训练loss时注明。


2026-10-12历史草案已被75.6M固定双链实现替代；不再待用户选择链边界。

## 2026-10-13 固定双链checkpoint50000下游适配

沿用原训练冻结实现与187/154独立链槽位。新增评测桥接 /project_c20250601/251105016/project/dllm_test/downstream/fixed_chain_eval.py，复用原数据/探针/评分器；CDR按角色和链内位置映射AirGen网格，T1关系token统一mask、仅观测残基池化。整链任务用原生每链EOS生成（pairing16步、T4 32步），保留失败候选；该解码与object302不同，结果需标注。36任务计划及CPU验收：/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/plan.json、/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/cpu_adapter_checks.json、/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/cpu_gate/cpu_checks.json。训练模型和训练数据均未修改。


2026-10-14T12:43:50.280649+00:00 确认候选保持continued/task_mix/observed_eos/observed_cdr/LoRA，均三种子3000步。表征筛选未确立需扩展的辅助损失收益；阈值不因screen结果而调整。最终方案等待16模型独立确认。[冻结依据](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/candidate_freeze_r1.json)。


2026-10-14T12:50:48.718337+00:00 当前最终部署方案：保留75.6M固定双链187/154输入与主干，112000权重初始化，rank16 LoRA仅训练2260992参数，使用task_mix去噪任务；选择依据为预先冻结的16模型独立确认（唯一全门限合格）。不加入当前蒸馏、弱偏好、额外结合头或新attention拓扑。全量阶段global512/LR2e-5/warmup200/cosine10000与pilot batch32/3000步不同，须单独监测；保存best3+latest完整状态。[决策](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/confirmation_r1/selection.json)，[正式包](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/prepared_final_r1.json)。


2026-10-14T13:02:15.800552+00:00 选定LoRA方案已实际部署：50→100原生八卡状态恢复及checkpoint验收完成，主训练已继续更新。该结论仅完成方案选择/布置训练，不代表10000步结束或下游全面提升。[统一部署验收](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/deployment_complete.json)。


2026-10-15 用户要求LoRA完整跑一轮：累计目标33981步，由最大来源2485433×7/global512向上取整确定。延续原10000步模型/AdamW/RNG/均衡数据流，新增200步预热至原2e-5，再余弦至终点；这是明确新增的学习率阶段，不声称等同从头33981步单段余弦。模型、LoRA rank16、任务混合、数据、验证协议与batch保持；复制原冻结源码，仅加入续训入口和调度支持，原产物独立保留。[冻结计划](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_lora_epoch1_20261015/continuation_plan.json)。

## 2026-10-15 CDR 免训练多轮对照

执行推理调研第一阶段：固定权重、词表、EOS 规则与 fp32 精度，原 AirGen 解码保留。比较原始/归一化置信度、root/cosine 提交、自适应阈值/熵预算、greedy 重遮蔽及带 substitution carryover 的 ReMDM-cap 概率转移；eta=0 是对应随机 MDLM 控制。greedy 方法是本地适配，不宣称论文完整复现。额外 oracle 仅回填已提交位置的上下文，并重放基线提交位置；最终预测仍由模型产生，不用于候选选择。细节见[任务协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md#cdr-inference-rounds-20261015)。

免训练第一轮尚未证明可部署策略优于单步，冻结候选的确认评测已完成，保持生产默认；oracle结果用于定位错误条件传播，不能作为可用方法。证据见[诊断结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-inference-rounds-results-20261015)。

CDR采样方法选择以目标指标为准，不要求步数与AAR单调增加；完美边缘后验下单步逐位置argmax已最大化条件期望AAR。这个数学性质不证明当前模型已达到最优，实际策略仍以冻结确认实验判断。[指标解释](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md)。

三轮免训练评测完成：H3的退化可复现，当前候选未形成跨数据集/checkpoint稳定收益；H1/H2有局部收益，后续不能强求所有CDR共享同一个最优步数。模型权重与正式协议均保持。[三轮诊断结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-inference-rounds-results-20261015)。

## 2026-10-15 排除已试采样策略后的新探索

按新的用户目标继续免训练探索。搁置上轮logit/log-prob排序、线性/root/cosine提交、自适应阈值/熵预算及greedy/random/ReMDM重遮蔽，改测三类概率聚合：多个预测上下文的遮蔽位点概率平均；所有位置逐一留一遮蔽后与原始边缘概率融合；枚举单个上下文位点候选并按原始概率加权。它们都是本地待检验假设，不宣称精确边缘后验或论文完整复现。模型权重固定。

新探索池来自较新SAbDab快照，排除上一轮PDB与完整重链重复，整组保留确认样本。只对仍为MASK的位点使用条件概率；不采用自身可见位点的高置信度。52种配置先在H3探索，候选若无正增益就继续新方向，禁止把负扫参当作本轮目标完成。正候选必须冻结后做确认和跨checkpoint复核。[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

## 2026-10-15 checkpoint概率集成探索

前一轮条件概率聚合仅探索提升0.0394个百分点，未确认为改善。新增已有50000/112000单步概率的算术与几何融合，权重仅在318条探索H3选择；两个单模型均作对照，融合按两次前向计费。保留确认集未评分，不改变权重或正式默认。[冻结方案](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015/ensemble_plan.json)。

## 2026-10-15 上下文强度新探索

概率集成未超过较强单模型，见[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-marginal-results-20261015)。继续单checkpoint条件强度对照：完整输入与分别mask可见轻链/可见重链/全部可见残基的两次预测做log概率对比；固定位置、链身份、EOS与目标全mask。系数预设−0.5/−0.25/0.25/0.5/1/2，仅探索集选参。该方向是待验证推断，不宣称模型已有严格CFG训练或保证改善。[冻结方案](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015/contrast_plan.json)。

## 2026-10-15 软embedding上下文探索

条件强度未改善，见[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-marginal-results-20261015)。新假设：用模型自身氨基酸分布的期望embedding代替硬填残基，逐位置留一遮蔽，只读取被遮蔽位置输出；插值强度、温度和与初始概率混合权重预设后探索。模型参数冻结，不宣称与离散边际化等价。真实模型先检查空hook与onehot embedding精确等价，再跑探索；额外前向如实计数。[冻结方案](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015/soft_plan.json)。

## 2026-10-15 冻结新推理候选进入确认

选定新探索方向中探索增益最高的软embedding配置，冻结温度/强度/先验混合权重；保留全部失败方向。主检验为112000在预留现代SAbDab H3上的分组配对区间，辅以H1/H2、50000同配置及历史Kong/SAb H3。确认不调参，若失败如实保留，后续方法不得复用该确认集选参。[预注册](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015/confirmation_plan.json)。

## 2026-10-15 新推理冻结确认结论

软上下文主检验存在微弱正向信号，但跨checkpoint区间及成本不支持实用替换；保留为研究结果，正式默认不变，稳定改善目标仍active。不得用原多步退化对照掩盖相对单步收益极小的事实。确认集已使用，后续改法应另设未用于选参的验证来源，并优先解决额外前向成本。[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-marginal-results-20261015)。

## 2026-10-15 分组软上下文：降低免训练推理成本

将目标CDR划为2或3组：每次保持整组待评分位置为MASK，仅在其余目标位置注入初始预测分布的期望embedding，再读取本组输出。所有组使用同一初始分布，可见上下文保持固定；模型权重冻结，无需训练。初始预测加分组预测，每个候选共3或4次前向（NFE）。固定网格为groups 2/3、interleaved/contiguous划分、temperature 0.5/1、embedding插值strength 0.25/0.5/1、输出概率的prior weight 0/0.5/0.8，共72个候选，仅在原318条H3开发集选择。

已在新OAS评分前冻结开发集winner，并完成50000与112000两checkpoint的确认及单步/多步对照。此前已使用的SAbDab确认集不能重新标为未见验证集；新数据数量和排除范围统一见[数据审计说明](/project_c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md#cdr-block-data-20261015)及[独立审计](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_context_20261015/data_independent_audit.json)。这轮验证成本和AAR改善是否可同时成立，不预设更多前向必然改善，也未替换正式默认。[运行与选参协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md#cdr-block-context-20261015)。

2026-10-15 分组方法的成本目标达到，跨checkpoint稳定提升标准未通过，保留负结果并维持正式默认。[结果和判断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-block-results-20261015)。

## 2026-10-16 分组soft组件归因

冻结完整方法后，采用softness×组间反馈×输出anchor的因子消融，并分开检查MASK混合与温度。将各因素保持在同一分组顺序和前向次数，避免把硬提交强度或anchor写入反馈bank混成多个因素。使用新预留数据、跨checkpoint成对比较并校正主要检验，不据测试结果选最优变体。[方法及统计协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md#cdr-block-ablation-20261016)。

2026-10-16 消融已完成：保留软分布的组件作用得到跨checkpoint支持；当前数据不足以把收益归因于组间独立或最终anchor。完整方法未稳定超过单步，不据消融结果选新winner或变更正式默认。[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-block-ablation-results-20261016)。

2026-10-16 用户要求追加新方法8前向验证：固定其余设置比较3组与7组，沿用已评分样本作为后续敏感性分析；预算与上下文划分同时改变，结论限定于该配置比较。[协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md#cdr-block-budget-20261016)。

2026-10-16 4/8分组预算验证完成，增加组查询没有明确H3收益证据，不能把更多前向当成质量保证；未选新配置，正式默认保持。[完整结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-block-budget-results-20261016)。

2026-10-16 严格从左到右hard argmax补全已完成，保留双向框架/轻链上下文；本轮H3在两checkpoint均低于单步，未获得改善证据，正式默认保持。[结果与边界](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-left-to-right-results-20261016)。

2026-10-17 按用户要求，对V7全有效token BERT当前best48828开展既有AB/TCR表征与MLM打分下游验证；编码器固定，保留各任务读出及探针协议。生成训练目标不在此轮变更。[评测绑定与范围](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#bert-full-token-representation-20261017)。


### 2026-10-17 固定第二层表征验证

按用户选择，将LoRA seed42/3000第2个Transformer block作为固定读出，对原有其他表征任务开展对照；冻结骨干、复用原头及划分。定义、任务范围和验收统一见[执行协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#lora-layer2-representation-20261017)。该实验评估较浅层是否具有跨任务收益，不能预设所有任务均改善。


### 2026-10-18 fixed50000第二层对照

按用户指定测试未做LoRA的fixed-chain50000，固定block2读出并与其自身末层对比。用于判断此前浅层收益是否也出现在这个预训练checkpoint；它不是LoRA3000的直接父checkpoint，不能据此单独归因LoRA训练。[执行与方法](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#fixed50000-layer2-20261018)。


### 2026-10-18 GDPa1池化验证

按用户建议检验分链拼接和结构token均值，未观察跨属性统一优势，保留原默认；特殊token原已参与attention，本次改变读出集合。[执行协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#fixed50000-gdpa1-pooling-20261018)；[完整结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#fixed50000-gdpa1-pooling-20261018)。


2026-10-18 层间机制研究：保留原默认；优先验证去噪＋固定第二层教师的末层表征约束，直接评测第八层而非拼接浅层。MASK-key和拓扑干预已揭示交互，不能相加后直接部署。[证据、文献及方案](/project_c20250601/251105016/project/dllm_test/docs/LAYER_REPRESENTATION_DIAGNOSIS_20261018.md)。


2026-10-18 已执行首个末层保持pilot：固定第二层教师的中心化关系损失直接约束第八层，比较CE-only／joint／repr-only，不拼接浅层或添加推理头；尚无完整结论。[方法与运行](/project_c20250601/251105016/project/dllm_test/docs/LAYER_REPRESENTATION_DIAGNOSIS_20261018.md)。


2026-10-18 更新：关系与固定坐标约束完成实际pilot；整体CKA改善不足以达到第二层HIC，待跨任务保持性收齐后检验有界谱加权，不改原默认。[证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-preservation-pilot-20261018)、[方案](/project_c20250601/251105016/project/dllm_test/docs/LAYER_REPRESENTATION_DIAGNOSIS_20261018.md)。


2026-10-18 teacher-PCA消融确认较弱方向对当前GDP读出有贡献；已执行lambda_64有界谱加权与同预算uniform续训比较，尚未选优。[方法](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-spectral-pilot-20261018)、[实测依据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#teacher-spectral-readout-20261018)。


2026-10-18T03:00:38.310252+00:00：末层普通／谱加权等预算续训完成并复核，弱方向和HIC部分恢复但存在属性权衡；继续MASK-key交互与跨任务验证，未完成目标。[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-spectral-results-20261018)、[研究解释](/project_c20250601/251105016/project/dllm_test/docs/LAYER_REPRESENTATION_DIAGNOSIS_20261018.md)。


2026-10-18T03:02:47.624857+00:00：MASK-key与末层训练交互验证完成，HIC已接近第二层但其他性质仍未达到；关系300的m396保持性完成。尚缺谱加权组合的跨任务确认和整体性质恢复，不能结束目标。[唯一数值页](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-mask-interaction-20261018)。下一步需针对Titer等尚未恢复的方向检查教师目标与实际末层约束，而非继续只优化HIC或只看CKA。


2026-10-18T03:11:48.865836+00:00：原始梯度审计不支持强烈整体CE／表征冲突，暂不把解冻输出头作为优先干预。新对照只改变student表征训练attention，固定原生教师和CE，恢复同一optimizer，分离训练／推理视图对齐因素。另做OAS-only线性可恢复性诊断，以区分读出几何与不可恢复损失；诊断映射不算末层改善。[实测](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-gradient-audit-20261018)、[固定方案](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-attention-alignment-20261018)。


2026-10-18T03:16:11.322722+00:00：OAS-only线性恢复没有全面恢复GDP属性；同拟合映射在GDP的误差更大，而教师距离分布仍重叠，需区分方向误差与粗粒度分布偏离。条件OOF配对区间不支持把近似均值当成非劣证据；Titer五fold均下降。继续已启动的attention训练对照，不将诊断投影作为目标完成。[权威证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-linear-recoverability-20261018)。


2026-10-18T03:20:37.465233+00:00：attention训练／推理对齐对Titer有同预算收益，但没有全面达到第二层；单纯延长当前池化目标也不是充分解。下一阶段应针对保留信息的粒度和泛化检查，例如分链／token级教师约束与更多预训练样本的受控比较，而不是继续仅延长同一个全局均值损失。本轮所有自有任务已结束，目标保持active；尚缺实际末层整体恢复与固定候选独立确认。[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-attention-alignment-results-20261018)。


2026-10-18T03:27:54.818307+00:00：新增受控分链差异保持实验，比较增加全局损失强度与增加H/L差异信息，保留同起点、optimizer、数据及attention。仅训练辅助目标改变，末层读出与维度不变；当前未有成绩。[方法](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-chain-preservation-20261018)。


2026-10-18T03:39:07.622508+00:00：冻结教师目标的九组诊断完成；输出特征白化伤害部分属性，不能与误差谱加权混为一谈。保留自然方向尺度的混合教师仍有属性权衡，没有全面优于第二层；不将教师投影结果计为实际末层改进。[唯一数值页](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#multilayer-teacher-target-diagnostics-20261018)。


2026-10-18T03:43:02.038536+00:00：分链约束降低了目标重建误差，却未改善整体GDP；不再沿此单一方向增加损失。启动严格嵌套4096→8192 OAS覆盖量对照，保留原全局损失及校准，检验此前OAS到GDP恢复差距是否受样本覆盖影响，尚无新分数。[完成证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-chain-preservation-results-20261018)、[新固定方案](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-diversity-pilot-20261018)。


2026-10-18T03:54:53.883492+00:00：新实验已完成。扩大OAS覆盖量、改取最终归一化前输出、切换后训练全局attention均未解决末层整体差距；不升级为默认方案。[末层最新实验结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-latest-audits-20261018)。


2026-10-18T03:56:36.970382+00:00：新增谱加权训练CE反向消融，直接检验生成约束与表征恢复的取舍。[CE消融固定方案](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-ce-ablation-20261018)。


2026-10-18T14:44:09.795452+00:00：原CE进程已确认终止且无中途checkpoint，新目录重放同一固定对照；补做折内性质预测方向误差诊断。[CE消融恢复及方向审计](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-ce-ablation-recovery-20261018)。


2026-10-18T14:47:52.675674+00:00：CE0恢复对照已完成，移除原CE约束仍未整体恢复第二层；不将去噪冲突作为当前主要解释。检验同起点更大优化幅度能否减少残余对齐误差，固定预算并保留去噪监测。[CE消融实测](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-ce-ablation-results-20261018)、[学习率对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-optimization-pilot-20261018)。


2026-10-18T14:55:49.307895+00:00：CE0对照不支持原CE是主要瓶颈；更大优化幅度改善HIC但仍无法恢复全部性质。高整体CKA不能替代性质方向核查。固定遮蔽输入读出未整体改善，不升级默认方案。下一步需检验当前池化监督是否足以约束token层面的信息保持，并继续以实际下游和生成保持性验收。[学习率对照实测](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-optimization-results-20261018)。

2026-10-18T15:03:28.749188+00:00：启动token方向约束的32位置／全残基覆盖对照，检查池化监督下的token信息保持是否改善未恢复性质；并补查高LR候选的Specificity。结论以固定终点下游实测为准。[token方向对齐协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-token-direction-20261018)。

2026-10-18T15:06:54.737443+00:00：逐token约束的稀疏组仍未恢复Titer；增加固定预算谱权重下限对照，检验弱方向损失分配，保留原Gram目标以隔离干预。[谱权重下限对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-spectral-floor-20261018)。

2026-10-18T15:13:45.829044+00:00：token方向约束改善HIC，但密集约束仍未补回Titer；不能用局部对齐误差改善推断性质信号完整恢复。继续既定谱下限对照和跨任务验证，目标active。[token方向对齐实测](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-token-direction-results-20261018)。

2026-10-18T15:16:06.343634+00:00：提高最弱方向权重确实降低其重建误差，却增加主要方向误差且没有恢复整体性质；不继续盲目加大该权重。token方向约束也仅部分有效。下一步先完成已有Specificity验证，再检查较高LR方案是否尚未充分收敛及怎样同时保留强弱方向；不把现有HIC改善当成全面完成。[谱下限实测与分方向核查](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-spectral-floor-results-20261018)。

2026-10-18T15:19:31.952372+00:00：已有高LR1700终点screen误差仍下降，启动同损失／学习率／数据的固定5000预算对照，恢复权重和优化器，不按中途GDP选模型；Specificity旧进程继续。[固定5000步收敛对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-convergence-20261018)。

2026-10-18T15:24:14.085161+00:00：研究报告当前结论已合并CE直接消融、方向恢复取舍与尚未充分收敛的边界；续训起点复现及首个恢复包读取通过。[当前结论](/project_c20250601/251105016/project/dllm_test/docs/LAYER_REPRESENTATION_DIAGNOSIS_20261018.md)、[执行核查](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-convergence-20261018)。

2026-10-18T15:31:24.191026+00:00：高LR Gram1700在HIC改善的同时保持／提高Specificity平均成绩，但Titer未恢复；该证据只适用于这组native配置，不能代替正在运行的dense-token5000验证。目标active。[高LR末层Specificity实测](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-optimization-specificity-20261018)。

2026-10-18T15:34:39.490181+00:00：5000续训继续原进程；新增无标签的train／screen／GDP方向误差核查，先建立1700基线，待5000固定终点后比较，区分拟合与泛化。[泛化分方向核查](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-generalization-20261018)。

2026-10-18T15:37:29.563164+00:00：1700 train／screen／GDP无标签方向审计完成；增加固定权重FP32推理对照，核查残余弱信号误差是否受BF16推理舍入影响。5000续训沿原进程推进。[FP32推理对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-inference-precision-20261018)。

2026-10-18T15:41:06.365537+00:00：FP32推理未实质补回dense-token1700的性质差距，不把BF16推理舍入当作主要解释；5000固定终点结果仍待完成。[末层推理精度实测](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-inference-precision-results-20261018)。

2026-10-18T15:48:49.145256+00:00：更充分拟合部分补回Titer，但不能同时保住所有性质；高CKA仍不等于完整迁移，留出弱方向有剩余泛化误差且去噪明显退化。先用高LR长预算的直接CE1／CE0对照检验生成约束，不将低LR短预算的结论过度外推，也不将当前5000候选升级默认。[5000步实测与泛化核查](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-convergence-results-20261018)、[同预算CE1对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-joint-convergence-20261018)。

2026-10-18T15:56:37.943675+00:00：1700／5000性质方向审计完成；Titer存在能较好迁移的teacher线性方向，AC退化也不能解释为所测方向误差增大。统一加密Ridge网格诊断已启动，原报告不改；CE1仍运行。[长训性质方向诊断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-long-training-directions-20261018)、[Ridge网格诊断协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-alpha-resolution-20261018)。

2026-10-18T16:01:04.361293+00:00：统一Ridge网格加密只解释部分分数变化，未消除主要差距。原第二层＋全global末层的互补教师目标诊断启动，仅评目标潜力；CE1原进程继续。[Ridge网格诊断结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-alpha-resolution-results-20261018)、[互补教师目标协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#global-teacher-targets-20261018)。

2026-10-18T16:04:28.838715+00:00：互补教师目标完成诊断，四个正相关性质改善而Tm2更差；这是目标设计证据，非真实学生成绩。当前先完成CE1配对实验；若仍不能整体恢复，可用同维数early512控制与混合教师进行实际蒸馏对照，不能直接将拼接目标升级为结果。[互补教师目标诊断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#global-teacher-target-results-20261018)。


2026-10-18T16:16:39.460186+00:00：CE1固定5000完成并通过配对／方向／泛化核查；去噪保持改善，末层整体目标尚未完成。新同维数教师蒸馏已本地运行。[CE1固定终点与对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-joint-convergence-results-20261018)；[互补教师实际蒸馏协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-complementary-teacher-20261018)。


2026-10-18T16:22:11.386155+00:00：CE1/no-mask-keys的Specificity、m396、T3-broad及native Kong生成保持性已本地启动；三表征任务GDP读出预核查通过。互补教师蒸馏原session44393继续。[CE1跨任务保持性协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-joint-cross-task-20261018)。


2026-10-18T16:26:54.676371+00:00：CE1/no-mask-keys T3-broad完成，七档点估计均略低于原native末层；启动两个缺失attention控制单元分离因素，其余既定任务继续。[CE1 T3-broad保持性结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-joint-tcr-results-20261018)；[TCR attention配对控制](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-joint-tcr-controls-20261018)。


2026-10-18T16:32:27.274268+00:00：TCR四格对照及两组距离层核查完成，固定attention后训练也均下降，不能归因于MASK排除单因素。CPU完成OTS4096／126无标签回放池，为后续保持约束做准备；原四个GPU任务继续。[TCR权重与attention完整对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-joint-tcr-factorial-results-20261018)；[无标签TCR回放数据准备](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#tcr-preservation-replay-20261018)。


2026-10-18T16:41:34.841449+00:00：CE1真实Kong三个CDR均下降，原始预测和协议重算通过；TCR原末层教师缓存及漂移诊断完成。连续CDR单步诊断KL数值问题已定位并用独立v2修复重跑，其余任务继续。[CE1真实CDR保持性](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-joint-cdr-results-20261018)；[TCR教师目标与漂移诊断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#tcr-preservation-targets-20261018)。


2026-10-18T16:54:48.651248+00:00：m39612设置保持／改善，early512固定终点完成但Titer未达第二层；连续CDR和置信度反事实诊断完成，单改排序不足以恢复OAS四步生成。Specificity和互补教师另一arm继续。[CE1 m396结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-joint-m396-results-20261018)；[末层保持性与置信度诊断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-span-confidence-results-20261018)；[early512固定终点](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-early512-results-20261018)。

- 2026-10-18T17:02:18.805844+00:00：保持性证据补齐[Specificity](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-joint-specificity-results-20261018)；下一步需覆盖条件token分布而非仅随机mask准确率/置信度偏移，先准备[CDR原模型缓存](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#cdr-preservation-targets-20261018)。互补教师固定终点待验，尚无全面成功结论。

- 2026-10-18T17:11:57.906581+00:00：对已确认TCR/CDR下降的CE1候选，准备同起点、同原目标与前向、仅切换回放反传的保持性对照；先运行无更新preflight，不预判回放会恢复任务。[协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-retention-pilot-20261018)。

- 2026-10-18T17:14:32.175761+00:00：互补教师实际末层固定终点完成，补回Titer点估计而HIC/AC接近第二层；待确认保持性和稳定性，不能宣称目标完成。启动候选native跨任务核验与独立CE1回放对照。[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-complementary-results-20261018)、[保持性执行](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-complementary-cross-task-20261018)。

- 2026-10-18T17:20:32.160531+00:00：新候选全层与冻结控制完成，后层性质确有改善，但Titer均值收益集中于一个fold；条件区间不能证明稳定达标。TCR同native七档点估计均下降，保持训练和其余任务继续。[逐层证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-complementary-depth-results-20261018)、[TCR保持性](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-complementary-tcr-results-20261018)。

- 2026-10-18T17:26:36.923921+00:00：词表头nullspace目标可行性与等距控制完成：提供将语义目标放入输出不敏感方向并保留原均值head分量的候选设计；只是目标级证据，非token或生成保持，未训练且不改现有live实验。[结果与限制](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#head-nullspace-target-results-20261018)、[协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#head-nullspace-target-protocol-20261018)。

- 2026-10-18T17:31:22.500601+00:00：互补候选完整Kong已核验，GDP改善伴随真实生成损失；继续原保持性对照，预备固定终点审计，不将新目标可行性或monitor当作恢复证据。[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-complementary-cdr-results-20261018)。

- 2026-10-18T17:46:41.120328+00:00：互补候选四任务核验齐全，GDP/AB探针改善未带来TCR/CDR保持；不升级默认。继续既定CE1同起点回放对照，固定终点后联合评估GDP与真实任务；是否迁移到互补目标须由实际结果决定。[证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-complementary-specificity-results-20261018)。

- 2026-10-19T01:07:07.354144+00:00：当前BF16训练执行路径不具逐位重复性，精确恢复参数/AdamW不等于后续轨迹逐位相同；已有两次同seed20步诊断。保留此限制，不能将恢复容差当作任务稳定性证明；固定终点仍需实际GDP和TCR/CDR验收。[诊断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#retention-recovery-repeatability-20261019)。

- 2026-10-19T01:15:39.243968+00:00：回放同预算对照已完成，screen目标误差改善但GDP仍有性质取舍；当前配置不能直接升级。先完成既定实际任务以区分监测恢复与真实保持，随后再决定目标方向/约束设计；尚不启动nullspace新训练。[证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-retention-endpoint-results-20261019)。

- 2026-10-19T01:20:20.563515+00:00：实际native TCR显示MSE回放仅部分恢复；缓存子空间反事实提示no-mask HIC损失更多伴随头null分量变化。尚不能分离TCR/CDR两个损失的因果贡献，不能只用head敏感方向冲突解释，也不将混合缓存当作输出方案。完成剩余真实任务后再选下一项训练干预。[证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#retention-native-tcr-results-20261019)、[子空间](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#retention-head-subspaces-20261019)。

- 2026-10-19T01:24:14.531527+00:00：两种attention的实际TCR均未全面恢复，当前MSE回放不足；下一设计须在实际few-shot任务上验证，不能依据更小的feature MSE宣称保持成功。先完成正在运行的CDR以判断生成端效果，再选择训练干预。[证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#retention-no-mask-tcr-results-20261019)。


2026-10-19T01:33:21.258505+00:00：本轮不直接训练逐token head/norm投影目标，其池化后GDP仍有取舍。先验证序列均值目标能否在逐token约束下实现；这只决定下一训练目标的可行性，不以离线目标替代实际末层。[方法与运行](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#token-head-norm-targets-20261019)。


2026-10-19T01:35:01.243994+00:00：序列均值固定的逐token约束构造可行，直接逐token归一化的Titer损失并非由head/norm约束必然导致。下一步将此构造扩展到OAS训练/screen，并检查互补教师目标可行性；实际训练须从原参数验证，不把目标探针当成模型修复。先完成当前保持性跨任务评估。[证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#token-mean-head-norm-targets-20261019)。


2026-10-19T01:43:54.432221+00:00：将离线可行目标推进真实参数学习，原fixed50000同起点配对：pool-only与pool+feasible-token；最终RMSNorm冻结以保持目标坐标，实际只更新blocks3–8，生成/TCR保持项两组相同。末层需实际GDP和跨任务共同验证；不把目标分数当作模型性能，不把逐token clean目标保持外推masked decoding。[完整设计与限制](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#feasible-final-pilot-20261019)。
