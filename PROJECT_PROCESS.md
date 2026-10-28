> 2026-10-29 07:25 +08:00，按用户要求停止持续人工轮询，待之后再查看。已启动的本地串行确认队列保持后台运行，未取消或重启训练。最后一次整体核查（2026-10-28T23:25:25Z）：A0–A7筛选8/8完成；三种子确认0/9完成，首组confirm_A0_s42为3393/3828步，训练PID122910和控制器PID122889均通过实际命令核验；其余8组尚未开始。已有控制器继续执行后续试验及既定评测、记录和发布流程。此处暂停的是主动监督，不是GPU训练。

> Last updated: 2026-10-28T23:08:57.254861+00:00 (UTC)。本地A0确认step3250完整screen评测与checkpoint已核验：token CE0.675133、CDR AAR80.74%、整链AAR10.17%、宏AUROC0.855206、宏AUPRC0.656580。相较3000步，token/CDR/条件CE降低，整链AAR和表征指标小幅回升，但长度MAE增大；best生成500/best表征1250保持。六文件SHA、数据游标及原冻结绑定通过，完整曲线/CSV和回执已更新。当前3254/3828步，下一次完整评估为3500步，九组确认继续在本机执行；原V8形式和固定10%子集保持，尚无三种子确认结论。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T22:39:17.219344+00:00 (UTC)。本地A0确认step3000完整screen评测与checkpoint已核验：token CE0.677267、CDR AAR80.40%、整链AAR9.93%、宏AUROC0.850580、宏AUPRC0.648004。相较2750步，token/CDR CE降低，但整链AAR基本持平、条件CE升高、AUROC/AUPRC回落；best生成500/best表征1250保持。六文件SHA、数据游标及原冻结绑定通过，完整曲线/CSV和回执已更新。当前3003/3828步，下一次完整评估为3250步，九组确认继续在本机执行；原V8形式和固定10%子集保持，尚无三种子确认结论。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T22:08:49.854633+00:00 (UTC)。本地A0确认step2750完整screen评测与checkpoint已核验：token CE0.680204、CDR AAR80.56%、整链AAR9.95%、宏AUROC0.854481、宏AUPRC0.662842。相较2552步，整链AAR及AUROC小幅升高，条件CE降低，但CDR CE升高、AUPRC回落；best生成500/best表征1250保持。六文件SHA、数据游标及原冻结绑定通过，完整曲线/CSV和回执已更新。当前2751/3828步，下一次完整评估为3000步，九组确认继续在本机执行；原V8形式和固定10%子集保持，尚无三种子确认结论。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T21:44:41.470773+00:00 (UTC)。本地A0确认第二轮末step2552完整screen评测与checkpoint已核验：token CE0.683204、CDR AAR80.05%、整链AAR9.80%、宏AUROC0.853470、宏AUPRC0.674383。相较2500步，token CE下降、表征指标回升，整链AAR回落；best生成500/best表征1250保持。六文件SHA、两轮数据游标及原冻结绑定通过，完整曲线/CSV和回执已更新。当前2556/3828步，下一次完整评估为2750步，九组确认继续在本机执行；原V8形式和固定10%子集保持，尚无三种子确认结论。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T21:38:28.782527+00:00 (UTC)。本地A0确认step2500完整screen评测与checkpoint已核验：token CE0.684500、CDR AAR80.21%、整链AAR10.16%、宏AUROC0.850866、宏AUPRC0.653701。CDR指标改善，整链AAR小幅回升但条件CE及长度误差增大，表征指标回落；best生成500/best表征1250保持。六文件SHA、数据游标及原冻结绑定通过，完整曲线/CSV和回执已更新。当前2505/3828步，下一次完整评估为第二轮末2552步，九组确认继续在本机执行；原V8形式和固定10%子集保持，尚无三种子确认结论。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T21:07:40.080651+00:00 (UTC)。本地A0确认step2250完整screen评测与checkpoint已核验：token CE0.689420、CDR AAR79.92%、整链AAR9.93%、宏AUROC0.853613、宏AUPRC0.685070。整链AAR和长度较step2000部分恢复，但条件CE升至5.289235；best生成500/best表征1250保持。六文件SHA、数据游标及原冻结绑定通过，完整曲线/CSV和回执已更新。当前2255/3828步，九组确认继续在本机执行；原V8形式和固定10%子集保持，尚无三种子确认结论。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T20:37:56.314765+00:00 (UTC)。本地A0确认step2000完整screen评测与checkpoint已核验：token CE0.696573、CDR AAR79.26%、整链AAR8.58%、宏AUROC0.851152、宏AUPRC0.651114。token和CDR CE继续降低，整链AAR及长度仍退化；best生成500/best表征1250保持。六文件SHA、数据游标及原冻结绑定通过，完整曲线/CSV和回执已更新。当前2006/3828步，九组确认继续在本机执行；原V8形式和固定10%子集保持，尚无三种子确认结论。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T20:08:07.828564+00:00 (UTC)。本地A0确认step1750完整screen评测与checkpoint已核验：token CE0.700540、CDR AAR79.92%、整链AAR9.29%、宏AUROC0.850284。token CE继续降低，整链AAR和长度再次退化，表征AUROC部分回升；best生成500/best表征1250保持。六文件SHA、数据游标及原冻结绑定通过，完整曲线/CSV和回执已更新。当前1757/3828步，九组确认继续在本机执行；原V8形式和固定10%子集保持，尚无三种子确认结论。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T19:40:53.908345+00:00 (UTC)。本地A0确认step1500完整screen评测与checkpoint已核验：token CE0.710401、CDR AAR80.42%、整链AAR10.67%、宏AUROC0.836977。整链指标较首轮末部分恢复，表征AUROC回落；best生成500/best表征1250保持。六文件SHA、数据游标及原冻结绑定通过，完整曲线/CSV与回执已更新。当前1529/3828步，九组确认继续在本机执行；原V8形式、固定10%子集保持，尚无三种子确认结论。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T19:11:15.354565+00:00 (UTC)。本地A0确认第一轮完成，step1250/1276完整screen评估与checkpoint已核验并更新曲线。1276步token CE0.717845、整链AAR9.25%、CDR AAR80.00%、宏AUROC0.861855；1250步AUROC峰值0.873140，best生成仍为500、best表征为1250。首轮游标epoch=1/next_batch=0、累计653288条与日志一致；原V8形式、10%子集和冻结源码保持。当前1288/3828步，第二轮继续，九组确认尚未完成。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T18:37:31.920543+00:00 (UTC)。本地A0确认step1000完整screen评估与checkpoint完成：token CE0.725498、整链AAR10.65%、CDR AAR79.50%、宏AUROC0.851295。token/CDR继续改善，整链AAR基本持平而条件CE升高；best生成/表征仍为step500。六文件SHA、数据游标和冻结绑定通过，已更新完整曲线/CSV及回执。训练当前1014/3828步，九组队列继续在本机执行，原V8形式与固定10%子集保持。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T18:07:18.314642+00:00 (UTC)。本地A0确认step750完整screen评估与checkpoint完成：token CE0.752463、整链AAR10.61%、CDR AAR78.45%、宏AUROC0.849416；整链AAR与长度继续退化，原多种子确认协议保持。六文件SHA、数据游标和冻结绑定通过，best生成/表征仍保留step500；已刷新曲线和回执。训练当前765/3828步，九组队列继续在本机执行，原V8数据形式与固定10%子集不变。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T17:36:36.590596+00:00 (UTC)。本地A0确认step500完整screen评估与checkpoint完成：token CE0.816243、整链AAR16.28%、CDR AAR76.41%、宏AUROC0.851734。相较250步整链AAR下降而token/CDR/表征改善，保持原多种子确认与终点评测规则，不据中间点改选。六文件SHA、数据游标和冻结绑定通过；完整曲线/CSV及回执已更新，训练继续至513/3828步。九组均在本机串行，原V8数据形式与10%子集保持。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T17:27:34.378829+00:00 (UTC)。新增CPU模型层蛋白块换序诊断，A0/A4/A5筛选终点权重SHA及18项检查通过；最大logit误差1.90735e-5，chain_id重编号输出逐位一致。证据和限定条件已同步记录，未使用确认结果或改冻结训练源码。单GPU确认A0 seed42仍运行，当前439/3828步；原V8数据形式、固定10%子集和9组队列保持。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T17:17:16.934419+00:00 (UTC)。CPU核查A4筛选权重step500/1250/1276，三个checkpoint模型/binding/state SHA与48个残差尺度向量通过检查，导出层间尺度图和CSV；FFN首层均值0.095983、末层0.109889，作为方法诊断而非确认胜出证据。无新增GPU任务，本地A0确认仍运行，当前354/3828步；冻结源码、10%数据索引与V8形式保持。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T17:06:45.251027+00:00 (UTC)。本地确认首组A0 seed42完成step250完整screen评测和checkpoint；验证CE1.050245、整链AAR18.57%、宏AUROC0.796904。六文件SHA、CPU严格权重/Adam加载、scheduler与数据游标一致性检查通过，训练继续；未运行CUDA重放或独立confirm终点评测。已刷新全曲线与保存验证回执，9组队列、冻结源码、原10%数据与V8形式保持。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T16:56:49.032358+00:00 (UTC)。新增CPU生成输入/目标区分诊断，八架构936条原AAR重算通过；A5正确目标与其他目标AAR分别37.02%/35.42%，需将通用序列恢复与条件特异性分开解释。核对6条TCR beta重复输出来自完全相同的MASK输入。产物、定义及输入SHA已记录；无额外GPU推理、未读confirm评测、未改原选择规则。本地九组确认仍运行，A0 seed42当前187/3828步；冻结源码及10%索引SHA核验保持。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T16:49:32Z (UTC)。八组筛选936条整链输出独立重算与已发布AAR一致；A5相对A0参考位置覆盖率33.60%→95.82%，相同共同覆盖位置准确率37.87%→53.64%，过短输出107/117→0/117。属单种子筛选诊断，尚不构成确认胜出。已保存PNG/PDF/SVG、逐任务CSV及来源SHA；冻结训练源码和原10%数据索引SHA不变。九组本地确认队列继续运行，A0 seed42已至124/3828步，micro56/global512，未启动重复GPU任务或远端任务。详见[生成核查与实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-28T16:36:35Z (UTC，2026-10-29 +08:00)。用户要求后续架构探索全部在当前开发机执行。八组screen均已完成，按冻结规则选A5生成候选、A4表征候选及A0对照；本机GPU0已启动9组串行确认（3架构×seeds42/43/44，各3epochs/3828步）。首组confirm_A0_s42已真实训练至15/3828步，micro56/global512；控制器PID122888、suite122889、worker122910，后台进程已脱离终端。原10%子集653,288条和V8数据格式保持，源码/数据SHA及八组筛选证据已核验。每组结束自动confirm/AB评测、曲线、commit/push探索分支；无新增远端提交、未改1B训练。恢复命令与运行回执见[本地确认队列](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)及[控制器回执](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_v8_20261028/local_confirmation_controller.json)。

> Last updated: 2026-10-28T14:54Z (UTC)。按用户请求刷新架构消融结果并生成可视化。A0–A6 已完成 1276 步及最终评测；本地 A2 于 12:51:20Z 结束。A7 快照时训练 602/1276 步，最新完整评测 step500；不参与终点排名，八组候选选择和三种子确认尚未执行。A5 整链 AAR 37.02%（A0 12.18%）、长度 MAE 5.61（A0 76.50）；A4 宏 AUROC 0.847814 最高，A6 验证 CE 0.716032 最低，均为单种子小样本筛选观测。交付 PNG/SVG/PDF、原始点 CSV 和带日志 SHA 的数据快照；所有完成组最终训练记录及评测步数核对通过，未改冻结训练源码或原 V8 数据形式，没有启动额外训练/评测。详见[实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)和[终点对比图](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/snapshot_20261028T1454Z/final_comparison.png)。下列时间条目保留为历史记录。

> Last updated: 2026-10-28T10:28:12Z (UTC)。GitHub 设备登录已由用户完成，当前开发机身份 `waitma`、仓库写权限验证通过。探索分支首次推送已成功，4ecc061 本地/远端 SHA 一致，先前凭据阻塞已解除；首次 HTTP 408 经 GitHub 专用 HTTP/1.1/缓冲配置重试解决。最新发布回执：[publication_status.json](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_v8_20261028/publication_status.json)。A2 本地训练及远端 A3–A7 分工保持，未操作 main。

> Last updated: 2026-10-28T10:14:19Z (UTC)。按用户分工，开发机 GPU0 已启动 A2（注意力门控）筛选，控制器 PID 92411、worker PID 92426；最大长度 profile 选择 micro56，global512，初始完整评估和 checkpoint-0 完成，真实训练已到 5/1276 步。A0/A1 已完成，四卡任务由用户提交 A3–A7，尚未收到远端提交回执。四卡入口默认列表同步改为 A3–A7，避免与本地 A2 重叠；原 V8 数据形式、固定 10% 索引和训练源码 SHA 保持。控制器信息：[local_controller_A2.json](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_v8_20261028/local_controller_A2.json)，分工：[execution_assignment.json](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_v8_20261028/execution_assignment.json)。

> Last updated: 2026-10-28T10:07:06Z (UTC)。已交付独立四卡入口 [launch_arch70m_4gpu.sh](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_4gpu.sh)，默认 screen；支持 confirm、多机分片和 dry-run。语法、参数转发、不同物理卡数量校验及真实单卡拒绝启动通过，当前仅一卡，尚无四卡实跑结果。入口检查曾因 nvidia-smi 忽略无效卡号触发 CUDA 初始化失败，未新增训练步数；A1 被调度器误标失败后，已核验 A0/A1 最终 checkpoint 全部文件 SHA 并恢复已完成状态，检查证据保留，详见实验日志。原数据及训练源码哈希保持。

> Last updated: 2026-10-28T09:56:24Z (UTC)。本副本属于独立探索分支 `explore/arch70m-v8-20261028`。本地 A0/A1 均完成 1 epoch、1276/1276 步，分别于 05:05:02Z / 07:59:16Z 完成，末轮生成/表征评估及 checkpoint 已保存；每组 17 个验证记录点，其中 7 次完整任务评估。筛选进度 2/8，A2–A7 与三种子确认尚未开始；本地控制器已退出、GPU 空闲。训练成功，控制器最终非零仅来自 GitHub 推送凭据缺失；结果已分别本地提交 10f2a3a / 097d88f。当前数据形式保持原 V8。下方现役 1B 历史状态为迁入快照，不作实时观测。详细结果与命令：[架构探索实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

> Last updated: 2026-10-27T19:00:11.170851+00:00 (UTC)。2026-10-28 03:00:11 +08:00：按用户“不要测试，直接开始训练”，准备1.057B V8正式随机初始化300000步包，micro16×GA4×8/global512，LR1e-4、warmup15000、cosine；沿用正式方案每1000步验证/保存、每25步日志，保留最近两份＋最佳。原micro16命令已转接正式入口，跳过独立NCCL资格/profile/smoke/200步测试；保留源码、数据、设备、认证及checkpoint完整性检查。语法、冻结输入与121分片检查通过；未新增GPU测试。当前开发机仅1卡且无平台控制CLI，未提交或开始八卡训练，等待用户平台执行。[正式入口与配置](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/BERT_1B_DIRECT_TRAIN_20261028.md)。

> 2026-10-27 ownkaifa02 文件同步已完成，训练／下游清单全量 SHA-256 通过；最新 V-family 启动包与现役 AB/TCR 测试输入已补齐。仅文件迁移，未启动训练／评测；目标运行路径和环境尚待适配。证据：[OWNKAIFA02_TRANSFER_20261027.md](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/OWNKAIFA02_TRANSFER_20261027.md)。

2026-10-27T05:26:50.702506+00:00 用户授权76500 light-chain pairing探索已启动本地tmux `pairing76500_explore`；CPU/CUDA真实模型检查及评分器8候选冒烟通过。协议见[任务手册](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md#2026-10-2776500-stepscfgprefix-探索)，实时状态见[status.json](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_76500_pairing_explore_20261027/status.json)。后台顺序生成/评分，无云任务，完成前不写榜单数字。

2026-10-27T05:21:51.297681+00:00 用户授权76500 αβ联合生成探索性对照，本地tmux启动；CPU五项和真实CUDA门禁通过，自动生成→独立复核→报告。任务协议与状态见 [T4 §7.11](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#711-2026-10-27αβ联合生成对照)，[输出](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_76500_joint_ab_local_20261027)；旧结果保留，无云任务。

2026-10-27T05:13:20.085933+00:00 用户授权的76500本地α占位修正重跑完成：held20与benchmark14共3434条原始候选全部生成、独立重放/重评分通过，旧产物哈希不变；结果总表/简表四个视图已新增修正行，旧行保留并标明输入问题。held20覆盖19/20，仍有一目标CDR3格式全部失败；benchmark14/sparse13/common6分别14/14、13/13、6/6。详见 [T4 §7.10](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#710-2026-10-27用户授权本地α占位修正重跑)；[验收回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_76500_alpha_context_local_20261027/verification.json)。本地GPU作业已正常结束，未提交云任务。

2026-10-27T04:44:41.520502+00:00 用户授权76500 held20/benchmark14在本地修正重跑；新增α全MASK上下文适配与仅目标β输出校验，4项CPU测试与真实CUDA门禁通过；本地生成已启动，先held20后benchmark14。状态见 [T4 §7.10](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#710-2026-10-27用户授权本地α占位修正重跑)，[新输出](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_76500_alpha_context_local_20261027)。未提交云任务，旧结果保留。

2026-10-27T04:39:23.876548+00:00 核对76500 sparse13采样链路与缺失α占位的影响范围，原始记录与视图分母核验通过；结论统一见 [T4 §7.9](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#79-2026-10-27提前eos的案例输入布局与提交机制诊断)。结果表补正式引用限制，历史指标保持；未新生成或提交任务。

2026-10-27T04:24:11.595306+00:00 held20提前EOS根因追踪完成：原权重CPU逐轮、冻结训练index扫描、40项输入因子对照及20目标全MASK α槽位干预均已复核；结论与边界统一见 [T4 §7.9](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#79-2026-10-27提前eos的案例输入布局与提交机制诊断)。新增诊断脚本及证据回执；未改正式模型/采样器/评测入口或历史分数，无GPU训练/云任务。

2026-10-27T04:36:28.400835+00:00 V8 family任务 t-20261027122841-gp69p 在data_gate失败，八rank NCCL初始化完成。复现根因：预检间接导入旧去重模块，硬编码/vepfs-mlp2路径读取records.py；本地旧挂载别名掩盖问题。新增冻结bundle_pathfix与launch_pathfix.sh，原网页入口及两份当前YAML已指向新包；旧bundle保留。禁止旧挂载访问的完整数据预检通过（121分片/6,604,459条），10项启动测试通过，W&B导入通过。新增torchrun之前CPU预检，rank_zero保留完整异常栈与文件名。未submit/cancel，GPU profile/replay仍待用户重提。

2026-10-27T03:42:03.146102+00:00 修正汇报第18页总结无法编辑的问题：矩阵底图移除总结文字，右侧标题和三条结论改为4个独立PPT原生文本框，文字/字体/字号/位置均可编辑。原生shape类型及逐字核对、底图右侧透明空白、导出PDF可选文字和实际版面核验通过；其余20页XML不变，21页PPT/PDF与20页draw.io保持。总验收267份来源通过；外部台账本轮新增held20限定说明及日志，保留live文件，以旧哈希完全一致的来源快照验收既有表图，原数字未变，本轮不改论文结果。旧稿已归档。[PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/20261027_项目进度汇报_逐页重做.pptx)、[文本框核验](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/data/native_summary_textboxes_verification.json)。

2026-10-27T04:08:38.107064+00:00 修复V8 family网页提交EnvironmentNameNotFound：原脚本按名称激活pllm依赖开发机Conda搜索配置，现对齐成功diffusion任务，以绝对路径 /project_c20250601/251105016/conda/envs/pllm 激活。新增 /project_c20250601/251105016/project/dllm_test/output/bert_v8_vfamily_all_relations_scratch_300k_20261027/launch_envfix.sh；旧launch与冻结bundle保持。三个网页入口及当前两份train_jobs YAML重定向修复脚本并同步入口SHA，生成器同步修复。训练解释器仍为原已验证protenix_abtcr绝对路径；模型、数据、8卡及loss配置不变。未提交任务。 验证：隔离Conda环境搜索目录的绝对路径激活、bash语法及原网页入口--verify-only均通过；121分片/6,604,459条数据和冻结plan SHA核验通过。GPU启动待用户重提。

2026-10-27T03:35:02.511904+00:00 完成Diffusion/edit 76500 held20保存产物debug：全量token解析及ANARCI重提取复核通过；空评分串来源、EOS与覆盖口径更正统一见 [T4 §7.8](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#78-2026-10-27diffusionedit-76500-held20-空评分串定位)。新增可复用离线审计入口及JSON证据，结果表增加子集均值限制；未改冻结生成/历史指标，未训练或提交任务。

2026-10-27T03:21:14.442223+00:00 汇报第18页关系矩阵右侧增加“本例观察”三点：所示Pair通道随训练明显改变、学生已复现部分链内短距离模式、跨链热点仍较弱并转下页结构验证。仅平移五幅矩阵以留旁注区，131921个原生矩形的颜色/尺寸及色阶逐项不变；限定valid_01单样本与固定通道0，不把单通道激活当接触准确性，下一页为其他3个Fab结构样例。同步可编辑draw.io、透明PNG/SVG/矢量PDF及21页PPT/PDF，修复独立PDF中文字体嵌入；其余20页XML不变，263份来源核验及目视复核通过。未改论文数值、未新增模型推理或训练。[PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/20261027_项目进度汇报_逐页重做.pptx)、[单页预览](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/preview/18_Pair%20feature%20与残基关系矩阵.png)、[旁注依据](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/references/pair_matrix_observations_evidence.json)。

2026-10-27T03:31:17.906861+00:00 按用户纠正，nonbinding也参与V-family预测：负例token loss仍仅关系符、序列上下文保持干净，另计算family CE（可靠注释或unknown）。schema v3，183类；全量重扫train有效链目标11055744/valid106343。43项CPU回归测试通过，新增负例known/unknown非零family梯度及无序列MLM验证；新包bert_v8_vfamily_all_relations_scratch_300k_20261027已准备并转接原网页Shell/YAML，旧冻结包保留。真实四类负例CPU检查及入口预检在新run目录，未提交/未GPU训练。[原网页Shell（已更新）](/project_c20250601/251105016/project/dllm_test/output/bert_unbinding_v8_vfamily_scratch_300k_20261027/web_submission/entrypoint.sh)。

2026-10-27T03:26:57.121447+00:00 按用户要求把缺失family标签改为unknown类别：当前183类=182个已知family+unknown，train有10581439条有效链目标（unknown2335377），valid102012（unknown60214）。缺链/完全合成占位链/nonbinding仍忽略family loss；有观测序列的VHH正例监督unknown。41项CPU测试通过；新冻结包bert_unbinding_v8_vfamily_unknown_scratch_300k_20261027，旧用户侧网页Shell/YAML已转接，旧182类冻结包保留。CPU真实模型检查及入口预检回执在新run目录；未提交/未GPU训练。[原网页Shell入口（已转接unknown版）](/project_c20250601/251105016/project/dllm_test/output/bert_unbinding_v8_vfamily_scratch_300k_20261027/web_submission/entrypoint.sh)。

2026-10-27T03:22:19.421970+00:00 按用户要求改为用户自行网页提交，不调用volc：参考既有nonspot网页入口，新增V8 V-family scratch web_submission/entrypoint.sh，检查当前挂载及冻结launch哈希、设置证书路径、将stdout/stderr同时写平台日志和entrypoint.log，再调用现有八卡profile→smoke→train入口。Shell语法检查通过；CPU --verify-only检查结果以entrypoint.log为准。无提交/取消或GPU训练。[网页提交Shell](/project_c20250601/251105016/project/dllm_test/output/bert_unbinding_v8_vfamily_scratch_300k_20261027/web_submission/entrypoint.sh)。

2026-10-27T03:10:32.336218+00:00 按用户要求将论文及汇报结果表统一为两位小数（ROUND_HALF_UP），保留原始账本/指标及raw_rendered_rows精度，底色仍按未取整成绩确定；m396删除grouped展示，仅保留主要row split。当前18张动态论文表中14张进入8页结果截图，固定历史参考附表也统一两位展示；完整汇报仍21页、合并draw.io20页，13个非结果页XML不变。重新编译论文、截取透明表图并导出PPT/PDF，261份来源、45份指标、两位展示及数值映射核验通过；实际PDF第9/10页目视无遮挡。未运行新训练或评测，canonical RESULTS.md SHA保持；两位小数前版本完整归档。[当前PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/20261027_项目进度汇报_逐页重做.pptx)、[论文](/project_c20250601/251105016/project/dllm_test/paper/main.pdf)、[格式核验](/project_c20250601/251105016/project/dllm_test/paper/tables/display_precision_audit.json)、[汇报验收](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/preview/validation.json)。

2026-10-27T03:20:43.874477+00:00 用户要求加入V-family预测头，已接入共享链均值池化→182类Linear头；总loss=原token均值loss+0.1×有注释链均值CE。train/valid分别8246062/41798个合格链标签；nonbinding仍仅关系符监督。39项CPU测试（含2-rank Gloo空辅助标签rank、梯度累计、保存重载、验证指标）及新冻结包121分片/6604459行CPU预检通过；真实全尺寸CPU检查另存run目录。独立random-init/300000步包bert_unbinding_v8_vfamily_scratch_300k_20261027，原用户侧scratch YAML已转接新包，旧冻结baseline保留。新八卡显存profile/replay仍待作业内执行，未submit。[当前family训练YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_unbinding_v8_vfamily_scratch_300k_8gpu_20261027.yml)。

2026-10-27T02:53:06.290030+00:00 本次汇报按最新要求精简为21页，前4页确认稿XML及18个交大母版部件保留；正式论文先更新BERT247000与Diffusion76500双模型行、精简baseline，再将15张真实PDF表格截图组合为8页。BERT既有结果恢复到canonical RESULTS.md共9表，全部新增单元格核对既有产物，157个保留baseline/Diffusion数值不变；BERT缺失生成成绩显示“—”，Diffusion父模型是BERT287000，未冒充同父模型阶段消融。TCR生成仅Sparse13/held20；T2/T3不进本次PPT。任务合并1页保留两个7帧透明GIF，Pair机制/完整曲线/矩阵压缩展示，中文负样本方法与扩充结果1页，Next plan加入Unbinding对照与扩大参数两项。21页PPT/PDF、20页可编辑draw.io、无备注、18pt下限、透明截图及259份来源/45份指标产物检查通过，相关实际PDF页已独立目视复核；旧36页完整归档并按SHA清理活动旧图。仅整理已有结果、修改论文与展示，未新运行训练、评测或云端任务。[当前PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/20261027_项目进度汇报_逐页重做.pptx)、[论文](/project_c20250601/251105016/project/dllm_test/paper/main.pdf)、[验收](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/preview/validation.json)、[BERT账本回执](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/data/bert_ledger_sync_verification.json)。

2026-10-27T02:13:48.866728+00:00 按用户新增明确授权，已整体删除两个旧8B输出目录 protein_esmc_llada8b_bert_immune / protein_esmc_llada8b_diffusion_immune（含正式权重及目录内记录）；删除前du合计249.23 GiB，两目录不存在核验通过。原保留规则对此两目录被本次授权覆盖，其他模型保留规则不变；历史8B权重路径不可再加载。[删除清单与回执](/project_c20250601/251105016/project/dllm_test/output/output_cleanup_20261027/legacy_8b/completion.json)。

2026-10-27T02:46:58.303100+00:00 按用户要求准备V8 R2从零训练入口：随机初始化75.6M BERT，独立输出bert_unbinding_v8_scratch_300k_20261027；预算暂按300000步、LR1e-4/warmup15000，global512、初始micro32×GA2×8。复用冻结R2数据/训练源码，启动器新增显式scratch校验及CPU verify-only；13项CPU测试和打包数据门禁通过。八rank显存profile及恢复重放必须在作业内完成，未提交、未运行GPU训练；family头尚未实现/启用。[从零训练配置](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_unbinding_v8_scratch_300k_8gpu_20261027.yml)。

2026-10-27T01:54:03.564728+00:00 output清理：删除两组已通过历史BERT smoke（AA-only smoke_r4、full-token smoke）的checkpoint-1权重/优化器/RNG/scheduler，共22文件、1.690 GiB；保留并哈希核验报告、配置及原始回执，新增WEIGHTS_REMOVED标记。正式权重、共享索引/vendor、教师缓存及评测结果保留。未查询实时云任务，仅核对本地打开文件及静态引用。[清理回执](/project_c20250601/251105016/project/dllm_test/output/output_cleanup_20261027/completion.json)、[逐文件清单](/project_c20250601/251105016/project/dllm_test/output/output_cleanup_20261027/deletion_plan.json)。

2026-10-27T01:52:10.551560+00:00 汇报按用户修订完成36页：前4页交大确认稿保持，新增真实论文PDF表格截图16页（17原表/19裁片，原精度/全部行，浅底标记合规已发表参考）、两处任务GIF、两页清楚的Pair构建/注意力整合、完整MLM及KL轨迹和原始Pair→教师学生→实验接触三页完整矩阵；Unbinding改为具体token输入/监督例子，负样本以NetTCR-2.2思路的整受体错配和筛选/推断来源分别解释。35页合并原生draw.io与透明/矢量素材齐备，36页PPT/PDF、20pt正文、18母版部件、前4页XML、无备注及248项来源检查通过；2个7帧GIF已嵌入。论文仅表格样式/版面改变，18个指标产物SHA及原数值不变；T2/T3仍只从本次汇报排除。旧22页成品归档，活动旧图/预览清理；本轮仅展示修改与只读CPU矩阵诊断，无训练/云任务或新负样本构建。[当前PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/20261027_项目进度汇报_逐页重做.pptx)、[说明](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/说明.txt)、[验收](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/preview/validation.json)。

2026-10-27T01:23:24.325617+00:00 按用户要求从本次汇报的任务展示与结果中移除TCR clustering与representation（T2聚类、T3少样本表征）；保留T1/UpTCR结合预测、TCR生成及全部抗体任务。完整稿现为22页，任务介绍第7页重新排版，其他21页按旧新页码映射XML逐项不变；合并draw.io为21页、透明结果表6页，PPT/PDF与源图核验通过。旧24页稿及移除图已归档，活动图/预览清理并重建；仅改汇报范围，未改评测协议、权威结果或模型。[当前PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/20261027_项目进度汇报_逐页重做.pptx)、[验收](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/preview/validation.json)、[范围与页序](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/说明.txt)。

2026-10-27T01:13:41+00:00 项目汇报扩为24页：保留前4页交大封面/Pipeline/架构/路线，新增BERT、Diffusion、两页任务定义、八页统一结果表、六页Protenix-Mini探索与两页Unbinding/已有负样本构建。正文20pt起，结果表为透明截图并附逐格可编辑draw.io，23页合并源图及PNG/SVG/矢量PDF齐备；24页PPT/PDF文字、透明、矢量、来源与字号检查通过，前4页XML和18个母版部件不变，无备注。仅冻结既有结果与本地训练日志，另完成8条验证记录的只读CPU Pair诊断和真实特征PCA；未训练、提交/取消任务或构建新负样本。主线76500成绩、Diffusion重启与scratch Pair探索分开呈现，排行榜未改。[当前PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/20261027_项目进度汇报_逐页重做.pptx)、[交付与证据说明](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/说明.txt)、[验收](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/preview/validation.json)。

2026-10-27T00:32:11.996071+00:00 汇报架构按IgGM2模块层次重画：恢复浅紫/浅黄绿分组、蓝色堆叠模块与白色算子，右侧展开完整Transformer残差结构；架构47节点＋20连线。三阶段训练图全部英文，输入/模型/预测输出对齐、蓝色由浅至深，权重传递明确，第三阶段标记Planned；92节点＋39连线。实际PPT目视修正多行文字贴底、推理回路穿字及训练页底边间距，正文20pt起。当前4页PPT/PDF导出验证通过，交大母版18部件及封面/Pipeline XML不变、8份模型依据文件SHA不变、无scope/备注。3页合并draw.io及各图透明PNG/SVG/矢量PDF同步更新。保存IgGM2/旧稿/新稿对照和可重现配色统计（不当作审美分数）；旧稿完整归档。仅展示变更，无模型/训练实现变更。[当前PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/20261027_项目进度汇报_逐页重做.pptx)、[可编辑原稿](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/figures/Ophiuchus-AIR_论文图源文件.drawio)、[并排对照](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/preview/style_review/architecture_reference_comparison.png)、[验收](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/preview/validation.json)。

2026-10-27 汇报新增模型Pipeline页：插入交大封面之后、模型架构之前，当前顺序为封面→Pipeline→模型架构→整体思路，共4页。新增图展示抗体与TCR/pMHC结构域示意、受体/可选条件序列、条件/关系/类型/链槽位与角色格式、表征读出及迭代生成两条推理用途。结构图表示分子对象；实际模型输入为序列，输出为表征或候选受体序列。新图131个原生可编辑对象，附独立draw.io/SVG/矢量PDF/透明PNG，合并draw.io扩为3页。四页PPT/PDF文字、字号、对齐、透明与矢量检查通过，原封面/架构/整体思路三页XML逐项不变；源码依据与独立科学复核通过，无模型/训练代码改动。[当前PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/20261027_项目进度汇报_逐页重做.pptx)、[Pipeline原图](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/figures/00_模型Pipeline.drawio)、[Pipeline预览](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/preview/02_Pipeline.png)、[输入/推理依据](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/references/pipeline_sources.json)。

2026-10-27 汇报图按Nature系列论文视觉重画：实际查看RFdiffusion（Nature 2023）、Chroma（Nature 2023）、InterPLM（Nature Methods 2025）Figure 1，保存原图、来源与SHA。两图采用白底留白、细线与局部低饱和强调，架构改横向主流程并显示真实残差旁路；训练图统一输入/模型/预测输出对齐，保留同蓝系浅→深三阶段。交大封面/母版、3页顺序与无备注约定不变。新增两页合并draw.io原稿、独立矢量PDF/SVG、透明PNG与图形样式说明，全部图元可编辑；科学依据文件SHA未变，无模型或训练改动。旧稿在本次目录归档。[当前PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/20261027_项目进度汇报_逐页重做.pptx)、[合并原稿](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/figures/Ophiuchus-AIR_论文图源文件.drawio)、[样式与编辑说明](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/图形样式与编辑说明.txt)。

2026-10-27 汇报架构与整体思路重画：按用户反馈恢复完整模块与单层结构，图内模块以英文为主、页标题中文；统一20/22/24/26pt字号，避免为追求大字删掉关键细节。整体思路补齐三阶段训练输入→Ophiuchus-AIR模型标识→输出，浅/中/深蓝分别表示BERT预训练、Diffusion生成训练、后续专用模型微调，并用权重传递箭头连接。架构页配色不约束后续页。保留交大封面，共3页；两图同步原生可编辑PPT/draw.io及透明PNG/SVG，无scope及演讲者备注。保留原母版18个XML部件，旧简版归档。本轮仅更新展示，不改模型或训练代码。[当前PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/20261027_项目进度汇报_逐页重做.pptx)、[架构原图](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/figures/01_当前模型架构.drawio)、[整体思路原图](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/figures/02_整体思路.drawio)。

2026-10-26 汇报开头页核验：活动目录现有1份3页PPT及对应3页PDF，首页均为完整交大封面，含Ophiuchus-AIR、项目主题、马坚及2026年10月27日；两张封面标识与原模板一致。按用户强调，将每份完整汇报固定保留封面纳入生成校验，现有三页内容保持；PPT/PDF首页字段逐项核验通过。[当前PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/20261027_项目进度汇报_逐页重做.pptx)、[封面预览](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/preview/01_封面.png)。

2026-10-26 汇报新增整体思路页：按用户命名，当前模型展示名称为 Ophiuchus-AIR。当前汇报共3页（交大封面、模型架构、整体思路）；架构页只讲结构，下一页按左到右BERT预训练→Diffusion生成训练→抗体Design等任务专用模型微调排列，以两条竖向虚线分隔并用箭头相连，第三阶段明确为后续计划。正文至少24pt、无页面备注及演讲者备注；两张图均提供原生可编辑PPT/draw.io和透明PNG/SVG，三页导出文本完整、母版18个部件保持一致。[当前PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/20261027_项目进度汇报_逐页重做.pptx)、[整体思路原图](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/figures/02_整体思路.drawio)。

2026-10-26 汇报大字精简：按用户最新意见恢复交大主题封面，当前稿共2页（封面＋架构）。架构正文至少24pt，移除单层细节展开、张量/词表小字及辅助说明；scope继续省略。PPT页脚和全部演讲者备注已移除，draw.io/透明PNG/SVG同步更新。原架构细节版存入本次目录_旧稿归档/架构细节版_修改前。[当前PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/20261027_项目进度汇报_逐页重做.pptx)、[可编辑原图](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/figures/01_当前模型架构.drawio)。

2026-10-26 汇报逐页重做：按用户要求先仅保留“当前模型架构”1页，继续沿用交大母版，图形风格参考IgGM2 Figure 2。当前图显示Token+Role、链内全注意力/全局交替×4、8层768维12头、一个Token head与表征分支；scope从图中省略，不修改模型代码。提供原生可编辑draw.io、PPT及透明PNG/SVG。今后本次汇报全部资料统一保存在 [20261027_项目进度汇报](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报)；[当前PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/20261027_项目进度汇报_逐页重做.pptx)、[draw.io原图](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/figures/01_当前模型架构.drawio)。旧稿其他页移出当前稿，20个根条目/212个文件归档至该目录_旧稿归档，保留恢复清单；原始三份汇报模板不动。

# 项目进度

2026-10-26 按用户更正，10月27日汇报已切换为交大主题：直接沿用用户提供20260804模板的4个母版、版式、主题、校门封面及联合标识；重排42页完整版与19页主讲版，表格强调色改为交大红且RGBA透明保留。18个母版/版式/主题XML部分一致，原模板与旧稿未改，结果单元格完全一致；导出PDF并检查字体换行。[交大主讲PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/_旧稿归档/261027_项目进度汇报_免疫受体基础模型_交大主题主讲版.pptx)、[交大完整版](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/_旧稿归档/261027_项目进度汇报_免疫受体基础模型_交大主题完整版.pptx)。

2026-10-26 明日汇报PPT完成：沿用8月原汇报主题，重新采用写实科研风格与真实PDB结构；已生成19页主讲版、42页含备份完整版及PDF。统一21张V7-76500结果表和7张Stage1源表，所有嵌入表图/结构图均为RGBA透明PNG，保留可编辑文字与架构。完成逐页渲染、透明/图片比例/原生文字/数值与模型边界核查；原PPT未修改。[主讲PPT](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/_旧稿归档/261027_项目进度汇报_免疫受体基础模型_主讲版.pptx)、[完整版](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/_旧稿归档/261027_项目进度汇报_免疫受体基础模型_写实版.pptx)、[说明](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/_旧稿归档/261027_汇报文件说明.txt)。

<!-- four-relations-20261026 -->
2026-10-26T16:49:11.514915+00:00 四类负例R2扩充完成：新增176822负例（51668筛选阴性、125154推断）与706阳性，负例107990→284812；新121分片共6604459行。全部新行collator/独立来源及配对标识核验、24项CPU与BERT287000真实44条四任务批次A100检查通过；R1辅助ID问题已修正并禁用R1入口。独立八卡续训包默认新增13000步，micro32/max2048 reserved43.916GiB；八rank资格待作业内运行。无submit/cancel、无正式训练。[四类负例探索、证据与R2入口](/project_c20250601/251105016/project/dllm_test/docs/V8_FOUR_RELATION_NEGATIVES_20261026.md)
<!-- /four-relations-20261026 -->

2026-10-26 论文 Results 同步：定位并更新 paper/main.tex，当前21张结果表统一绑定 Diffusion/Edit V7 step76500（15任务完成），另保留5张冻结论文基线表；主稿模型说明同步为8层768维native模型。生成器逐项校验当前模型指标并记录来源，原稿完整备份。数值权威源仍为 [RESULTS.md](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)，未新增评测或改写其数值。[新版PDF](/project_c20250601/251105016/project/dllm_test/paper/main.pdf)、[更新说明](/project_c20250601/251105016/project/dllm_test/paper/RESULTS_UPDATE_20261026.txt)、[核验](/project_c20250601/251105016/project/dllm_test/output/paper_results_update_20261026/validation.json)。

2026-10-26T13:51:19.790378+00:00 Ophiuchus-Ab三个冻结表征任务已全部完成：Specificity/m396按用户授权本机运行并真实exit0；全部新特征、保存预测及独立重评分核验通过，结果已写入原任务表。m396额外float64诊断完成；原输入/pooling适配与GDP参考CV局限保留。本轮无新增云submit/cancel。[完整执行与证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#ophiuchus-probe-audit-20261026)。

2026-10-26T13:21:02Z V8抗体负例扩充完成：新增28条IgDesign实测且带VH/VL的负例（train23/valid5），抗体负例2956→2984；1148条候选因测试集重叠排除。新96分片语料/index与独立BERT287000续训包已准备；15项CPU测试、逐新增行collator/独立来源审计及真实父权重混合批次单A100更新通过。VHH557236条负标签仅清点，现有train15507条尚未接入。默认新增13000步，未submit/cancel或正式训练。[结果、限制与新版入口](/project_c20250601/251105016/project/dllm_test/docs/V8_ANTIBODY_NEGATIVE_EXPANSION_20261026.md)。

2026-10-26T12:53:21.515079+00:00 Ophiuchus-Ab Specificity本机完整执行exit0：4398条新特征、5×100轮固定末轮分类头、OOF与独立混淆矩阵重算通过；进入m396全86929条及WT新特征提取和两套六档Ridge评测。仍沿用冻结参数，无新云任务。[Specificity指标](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_probe_audit_20261026/specificity/metrics.json)。

2026-10-26T23:52:54+00:00 Diffusion四卡修复包已实际训练：用户任务t-20261027004403-495cm平台Running，四rank smoke通过，从BERT287000初始化后完成7425/200000步，完整checkpoint7000。300条已记录更新无NaN/Inf，warmup后梯度均值0.2783、P95 0.3288、最大0.4111，无已记录裁剪；6000步验证平台期触发LR减半至9.986e-6，最佳验证6500=1.101507，7000=1.102351。128条内部已知长度CDR恢复暂未明显改善，H3准确率42.483%与初始化相同；数值稳定不等同CDR能力提升。旧任务t-20261027001920-m9ct7的cp312/py310 W&B失败现场保留；本次只读审查，无submit/cancel或训练设置修改。[曲线与统计](/project_c20250601/251105016/project/dllm_test/output/diffusion_bert287000_cdr_4gpu_wandbfix_20261026/diagnostics/training_review_20261026/review.json)、[当前入口与历史修复](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_FIXED_FRAME_CDR_20261026.md)

2026-10-26T12:45:34.672166+00:00 用户明确授权“需要跑完，在本地跑就可以”。Ophiuchus-Ab Specificity/m396改用当前空闲本机A10080顺序执行，沿用既有冻结计划和生产读出；云提交权限限制不再阻塞本轮。GDPa1已完成结果保留，无新云submit/cancel。[本机授权与起始资源](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_probe_audit_20261026/local_authorization.json)。

2026-10-26T11:58:35.649512+00:00 V8无MHC的6752条负样本来源追踪完成：全部来自TcrDesign-B官方Zenodo14545852的pMHC_TCR_train.tsv，train6676/valid76；逐条匹配原始6794行，MHC均34个X、label0。原发布无单条数据库accession，不能进一步强归于IEDB/VDJdb等；作者存在MHC遮盖增强，不能把全X一律解释为原始实验未注释。本轮仅审计，无训练数据修改。[来源逐行审计](/project_c20250601/251105016/project/dllm_test/output/mhc_completion_v8_20261026/missing_mhc_origin/summary.json)。

2026-10-26T09:39:38.359633+00:00 V8新增负样本MHC补全完成：105006条TCR记录均找到上游，56399条依明确等位基因补全参考class I重链+B2M；41855条保留pseudo，6752条上游无等位基因仍缺失。全部6426903行独立验收，62项CPU及真实287000权重/26条混合样本单A100更新通过；新数据索引和独立冻结训练包已准备，默认新增13000步，未submit/cancel或正式训练。[V8 MHC补全与当前训练入口](/project_c20250601/251105016/project/dllm_test/docs/V8_MHC_COMPLETION_20261026.md)。

2026-10-26T09:24:58.837459+00:00 Diffusion重启包完成：按用户要求改为BERT287000权重初始化、Diffusion step0/新AdamW，旧76500入口禁用。全量84分片/6,318,941行审计完成；小来源等权过曝、观测完整度分布差异与token_valid4条受体精确重叠被确认。新采样alpha.75/4倍自然上限，独立885条验证；50项CPU＋冻结12项＋A100初始化/真实批次/max2048更新通过，reserved69.006/79.151GiB。8rank资格由新YAML作业内执行，未submit/cancel或正式训练。[数据审计与新提交入口](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_BERT_RESTART_DATA_REVIEW_20261026.md)

2026-10-26 更正：用户要求从原 BERT287000 重新开始 Diffusion。此前 resume76500 仅为保留旧进度的修补方案，已撤销；旧 YAML 与 train.sh 已设置立即退出，尚未提交该任务。新实验准备中：Diffusion step0、新 AdamW/warmup、独立输出，并开展全量数据与验证重叠审计。

2026-10-26T09:13:37.736570+00:00 Ophiuchus-Ab GDPa1新提特征/生产读出已完成：全部768权重张量与官方checkpoint一致，原AirGen骨干抽查8条前向逐值一致，原回归同特征五项分数及alpha一致；全部OOF独立重算通过，五项分数与10月17日复跑逐值相同。新核验器初次误用raw标签核对CV，HIC因变换浮点并列排名触发断言；只修正核验器为变换后标签，预测/成绩未变，旧plan及runner保留。10项既有CPU测试通过。Specificity/m396尚未运行；云创建权限被拒，等待用户确认本地空闲A100运行例外。[状态与证据](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_probe_audit_20261026/status.json)。

2026-10-26T09:06:31.810594+00:00 Ophiuchus-Ab Specificity单卡非闲时作业提交被平台IAM拒绝（ml_platform:CreateCustomTask，RequestID=202610261705180483F23A63FF00E05EA3），未创建task ID，不加入Active表；未尝试其他账号/队列。GDPa1已在本机CPU启动完整新复跑，session34785；其余运行等待可用执行方式。[拒绝回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_probe_audit_20261026/submission_denied.json)。

2026-10-26T09:04:45.919209+00:00 Ophiuchus-Ab三个冻结表征任务全面复核准备：采用官方权重重新提取特征，直接调用当前Specificity/GDPa1/m396生产读出；冻结100轮分类头、legacy GDP CV及m396原始/分组两套划分。独立输出保存逐条预测并重评分；新GPU作业尚未提交。[执行记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#ophiuchus-probe-audit-20261026)。

2026-10-26T08:53:09Z 按用户询问准备Diffusion修复版单任务提交YAML：`train_jobs/diffusion_guard_resume76500_8gpu_20261026.yml`，ZENITH_MLP2_TASK非闲时8×A10080，从76500完整恢复到累计200000，作业内自动先smoke再train。冻结代码长度2048、每rank128视图按48+48+32累计的单卡两次完整更新通过，峰值allocated54.454/reserved65.957/total79.151GiB；global512原记录/1024视图不变。YAML及shell语法、冻结plan/profile身份、凭据文件存在和新输出检查通过。本轮未提交/取消云任务、未启动正式训练；八rank资格待作业内运行。[提交预检](/project_c20250601/251105016/project/dllm_test/output/diffusion_guard_repair_20261026/submission_preflight.json)。

2026-10-26T02:57:07Z Diffusion梯度审计和独立修复包完成：3088条训练记录/154个验证点、21条重放去重；后期梯度中位数0.323→0.642，已记录点低于clip1，固定验证CE1.104→1.269（+14.93%）。TRAIT仅1971条、累计曝光约2839遍为过拟合风险，尚无因果消融。修复clip配置硬编码和前后梯度日志，新增可恢复验证降LR；新包从完整76500续接，初始LR减半、连续3次无改善再减半，目标仍200k。30项CPU回归、真实权重单卡更新等价/减半与BF16检查、16文件及数据预检通过。旧任务最后监控为Killed，本轮无submit/cancel或正式重训，新包八卡资格和长期收益待验。[诊断与证据](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_GRADIENT_REVIEW_20261026.md)。

2026-10-26T02:56:11Z V8 BERT unbinding测试版本已准备：完整继承原BERT287000权重；显式nonbinding仅关系token监督，binding保留full-token MLM。V8恢复107962条负样本（train105568/valid2394），总train6355524/valid71379；239738条测试重叠负样本继续排除。94分片入场、24项CPU测试、真实权重混合样本单A100 forward/backward通过。默认新增13000步、global512/micro32×GA2×8；长度2048峰值reserved39.678GiB，micro64余量不足。冻结脚本/YAML已准备，八rank profile/replay在启动时运行；本轮无submit/cancel、无正式训练。[配置、数据与启动](/project_c20250601/251105016/project/dllm_test/docs/BERT_V8_UNBINDING_20261026.md)。

2026-10-25T23:01:45Z 按用户要求仅补飞书现有表格：以用户最新删改后的revision6508为基准，8张表新增BERT247000共10行（Specificity/GDPa1/m396/UpTCR三划分/T1四组），Light-chain pairing补Ophiuchus-Ab论文无前缀/原生EOS一行。9个局部表块写入至revision6517，原表头及原有单元格保留，全部非目标块逐块完全一致；用户删除的介绍、协议文字未恢复。未新增任务表、正文或pipeline，未进行训练/评测。Word与Markdown同步当前正文。[本次验收](/project_c20250601/251105016/project/dllm_test/output/weekly_report_20261025/feishu_sync/baseline_addition_verification.json)。

2026-10-25T22:47:39Z 按用户意见精简本周飞书周报至revision6497：11张结果表恢复上周表头/全部对照行，仅追加最新diffusion76500；移除上周未列的T2/T3/TCR生成补充。删除pipeline、教师筛选与复杂公式，探索部分仅为ESMFold2架构改动、50万条同初始化训练10epoch/9770步、训练MLM与KD曲线、最终GDPa1冻结表征表。曲线复用已完成日志并处理恢复重放，终点验证AA CE为0.302662/0.302179，KD为3.218→0.154；HIC改善、Titer下降，未表述整体优势。12张表全部单元格与曲线字节读回通过，周报之外原文完全保留；本地Word/Markdown/11页PDF同步。无新训练/评测或云任务操作。[飞书本周节](https://rcntyhvf331j.feishu.cn/docx/QLALdzCrjoV5y1xLNB7cu4jxn0e#doxcnaCXbEjMDdx7AuVBJebbjff)、[精简版验收](/project_c20250601/251105016/project/dllm_test/output/weekly_report_20261025/feishu_sync/simplification_verification.json)。

2026-10-25T22:35:48Z 飞书pipeline更正：按用户要求移除方法图中的已完成主线、进度状态与diffusion checkpoint，改用ESMFold2教师→链内distogram对齐→学生Pair/attention反馈→MLM＋KD联合优化；Protenix-Mini-ESM仅在正文标为候选教师测试。同步本地PNG/SVG/PDF/Mermaid、周报Markdown/Word/PDF与生成脚本；远端4个局部块更新至revision6476，读回revision6479，图字节一致、全部表格未变；同期外部一处文字修改保留。7页导出与13张本地表格复核通过。[更正验收](/project_c20250601/251105016/project/dllm_test/output/weekly_report_20261025/feishu_sync/pipeline_correction_verification.json)。

2026-10-25T22:29:49Z 飞书周报更新完成：官方 Lark CLI v1.0.98 用户身份登录成功，读取原文revision6471后，沿用日期倒序、编号章节、任务表头与Next风格，在20261019之前新增20261025节，写入revision6473。18张表含129条数据行，覆盖统一diffusion/edit76500的20条普通表项和9个T2A点，并保留上周关键对照；补充Pair feature对齐pipeline、局部收益及在线新版待验证状态。读回全部表格与正文精确一致，下载pipeline图片与原图字节一致，原696个顶层内容单元完全保留。未启动训练/评测或submit/cancel。[飞书本周节](https://rcntyhvf331j.feishu.cn/docx/QLALdzCrjoV5y1xLNB7cu4jxn0e#doxcnaCXbEjMDdx7AuVBJebbjff)、[远端验收](/project_c20250601/251105016/project/dllm_test/output/weekly_report_20261025/feishu_sync/verification.json)。授权凭据不写入项目文档。

2026-10-25T22:04:09Z 本周周报草稿已完成：以已核验的diffusion/edit76500统一更新结果，并补充结构教师→残基映射→distogram KL→Pair attention反馈→冻结表征评估的整体pipeline。交付可编辑Word、7页PDF、项目Markdown、SVG/Mermaid图及全部结果替换行；18份指标SHA、13张原生表格、Markdown列宽和PDF边界核验通过。飞书链接GET跳登录页，未读取上周正文/样式，也未写入远端，模板对齐仍待原文访问。此为周报快照，不替代RESULTS主表；无新训练、评测或平台提交。[周报](/project_c20250601/251105016/project/dllm_test/docs/WEEKLY_REPORT_20261025.md)、[交付与验收](/project_c20250601/251105016/project/dllm_test/output/weekly_report_20261025/verification.json)。

2026-10-25T02:35:07Z 已训练Stage1效果验证完成并通过最终验收：GDP、天然伙伴检索/恢复、实验结构检索及同源敏感性均有完整产物；所有主模型冻结，原运行session真实exit0。结论为局部改善，尚不足证明Pair整体优于纯MLM。无教师调用、新预训练或submit/cancel。[完成审计与结论](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINED_MODEL_EVALUATION_REVIEW_20261025.md#7-完成审计与当前结论)、[执行证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#stage1-native-partner-20261025)。

<!-- protenix-confidence-distillation-20261025 -->
2026-10-25T22:02:41.525876+00:00 周报只读核查发现loader修复版已有八rank联合profile与完整恢复短测passed回执，正式metrics已记录至step3；尚无完整训练或下游收益结果。本轮未查询新云任务ID/实时平台状态，不据本地文件更新Active云任务表；旧任务此前Killed的历史记录保留。未submit/cancel或启动训练。[本次带时间的本地快照](/project_c20250601/251105016/project/dllm_test/output/weekly_report_20261025/verification.json)、[修复版运行目录](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/train_fix_loader_20261025/pair_kd)、[旧任务快照](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/diagnostics/training_status_latest.json)。
<!-- /protenix-confidence-distillation-20261025 -->

<!-- general-teacher-validation-20261024 -->
2026-10-25T00:02:01+00:00 通用教师本地A100工程验证完成：ESMFold2/Protenix-Mini-ESM/Boltz-2各8/8 GPU例通过、进程正常退出；完整路径对照两新教师distogram最大误差均0。Protenix确定性r2修复重放问题，未降低门槛；等价链评价修正2HHB编号歧义。四配对例平均前向分别1.430/0.547/3.633秒，三Fab跨链AP19为0.9690/0.9745/0.9704，不能宣称泛化或学生收益。三次云提交均未创建任务，本轮无活动GPU任务；候选优先扩大Protenix验证。[协议、结果及独立启动脚本](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_GENERAL_TEACHER_VALIDATION_20261024.md)。
<!-- /general-teacher-validation-20261024 -->

<!-- stage1-general-teacher-landscape-20261024 -->
2026-10-24T23:04:07+00:00 完成通用结构教师源码可行性调研：15个官方仓库固定revision，并保留trRosetta作者源码及SHA；Protenix-Mini-ESM、Boltz-2可优先验证trunk＋distogram，RF3追加候选，AF3作备选，trRosetta作链内对照。现有ESMFold2已是trunk-only；AF2系/RFAA/OmegaFold有坐标反馈，不能一概删去全部3D计算。核算在线重算与完整分布缓存取舍；5个权重端点HEAD可访问，尚未下载新权重、运行新教师或证明速度/质量收益。训练与缓存保持现状。[调研与接入边界](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_GENERAL_STRUCTURE_TEACHER_FEASIBILITY_20261024.md)。
<!-- /stage1-general-teacher-landscape-20261024 -->

<!-- stage1-scratch-fast-20261024 -->
2026-10-24T12:36:34.657206+00:00 按用户要求完成Pair加速正式接入并冻结独立scratch_fast包：上三角镜像＋autocast前embedding转换＋关闭Pair重计算，保留主干checkpoint；两组仍随机seed42、9770步/global512及原链内KD。50项工作区CPU测试、冻结包10项集成测试、两组84分片/初始化/缓存入场通过；旧包源码SHA未变。新两组sh/YAML可运行，作业内自动每rank profile和8rank恢复门禁；本轮未新跑GPU或提交云端，局部约2.05倍仅引用此前候选实测。[优化版启动与核验](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_SCRATCH_FAST_20261024.md)。
2026-10-24T19:41:59.284628+00:00 从零训练曲线已生成：MLM完成9770步，Pair日志至9275、验证与完整checkpoint至9250，尚无COMPLETE；本地最后心跳19:24 UTC，不据此断言仍在运行。共同9250步AA CE为MLM0.302923586、Pair0.302463270，相对改善0.151958%；链内KL从3.218140降至0.155180（下降95.18%）。核验随机初始化/数据/MASK/训练条件一致，恢复重放剔除17行，保留最新轨迹；PNG/PDF含完整曲线与后期放大，CSV及快照保存。单seed且Pair未完成，不作稳定收益或跨链结论；未改训练或submit/cancel。[整体曲线](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_scratch_fast_20261024/diagnostics/curves_20261024_1945/scratch_overview_zh.png)、[链内KD](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_scratch_fast_20261024/diagnostics/curves_20261024_1945/scratch_kd_zh.png)、[数值快照](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_scratch_fast_20261024/diagnostics/curves_20261024_1945/summary.json)。
<!-- /stage1-scratch-fast-20261024 -->

<!-- stage1-contact-residues-20261024 -->
2026-10-24T11:33:55.652866+00:00 完成教师跨链接触位置级CPU诊断：复用1JHL/1MLC/1VFB实验坐标与完整缓存，定位52/51/59共162对接触，导出真实序列/PDB编号及全部接触特征。固定各例接触最多的H残基，对比其全部接触与同H远距L对照；三例接触组短距离均分0.9364/0.9433/0.9025、远距组均<0.00011。6组PNG/PDF、原始向量NPZ及CSV约2.96MB已持久化，坐标重算/特征对称和/概率归一化/图像检查通过；未训练或新GPU推理。[接触残基与256维教师特征](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_CONTACT_RESIDUE_FEATURES_20261024.md)。
<!-- /stage1-contact-residues-20261024 -->

<!-- stage1-cross-chain-diagnostic-20261024 -->
2026-10-24T08:13:09.252819+00:00 按用户授权完成本地完整矩阵诊断：6条随机valid＋3个预先指定实验结构，教师9份和学生18份全矩阵推理完成；教师跨链接触AP均值0.9692，学生step3000为0.00855。六例masked链内KL 3.4056→0.5475，跨链3.9530→7.4530且全部变差，当前链内KD不能代表跨链学习成功。生成5组PNG/PDF、57份二维数值图、逐例指标与SHA回执，27份张量有限值/形状和学生对称性核验通过；共享盘quota触发后完整约1.084GB张量保留容器/tmp，小型产物约13.65MB已持久化。仅诊断，不更新权重/正式缓存，不提交云端任务；三个相关Fab仅案例证据。[完整矩阵、实验接触与Pair特征诊断](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_CROSS_CHAIN_DISTOGRAM_DIAGNOSTIC_20261024.md)。
<!-- /stage1-cross-chain-diagnostic-20261024 -->

<!-- stage1-ab500k-offline-pilot-20261022 -->
2026-10-24T07:24:30Z 验证范围更正：用户要求保留原始Stage 1完整初步验证，AB500K两组仅标记为MLM/链内KD子集，跨链及其余原定对照未被覆盖。重新核对504096条SQLite索引、训练缓存SHA绑定、train/valid负载及存储：仅HH/LL各1024对，无HL/raw Pair State；正式缓存和辅助文件108.199GB。研究方案与AB500K文档已明确缺口；未修改训练/缓存或submit/cancel。[核查证据](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/supervision_scope_audit_20261024.json)。
2026-10-23T21:34:37.469844+00:00 AB500K本地loss曲线已生成：纯MLM已完成3000/3000（COMPLETE=success），Pair/KD快照475/3000；两组共同step的MLM loss几乎重合。MLM固定验证AA CE初始0.288198632、最终0.288281123，best仍step0；Pair仅到step250验证，尚无训练收益结论。新增只读绘图脚本，输出完整/共同step训练曲线、验证/蒸馏分项图、PNG/PDF、原始CSV及冻结快照；粗线为5个已记录批次的移动平均，验证不平滑。数据行数、目标函数分解及图像检查通过，无训练代码/配置修改或GPU提交。[训练曲线](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/loss_curves_20261023/training_loss_curves.png)、[验证及KD曲线](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/loss_curves_20261023/validation_and_kd_curves.png)、[数据快照](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/loss_curves_20261023/snapshot.json)。
2026-10-23T21:40:24.124992+00:00 AB500K长训预算讨论已核算：500000/global512保留尾批时每epoch977步，5epoch=4885、10epoch=9770、20倍总60000=61.412672epoch；按8卡现速总时长约MLM6.9小时/Pair50小时。建议当前Pair完成3000后，两组匹配续接到总9770先看趋势，主干/Pair峰值1e-5/1e-4、500新增步再warmup为候选；尚未改运行计划或生成/提交长训作业。[候选方案](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_OFFLINE_PILOT_PLAN_20261022.md#长训预算与学习率候选2026-10-23讨论方案)。
2026-10-23T21:57:29.079264+00:00 AB500K中文曲线及机制复核完成：21:51:44 UTC快照MLM3000、Pair800；共同验证750时Pair AA CE比MLM高0.062796%，教师KD由3.414345降至1.304292（−61.80%），尚无MLM收益。checkpoint750的4条CPU探针确认8层反馈已更新且梯度连通，加权KD/MLM主干梯度范数约2.81%，仅为小样本线索。提出KD-only/feedback-only对照及单因素Pair容量、跨层状态候选；未改训练或提交GPU。中文PNG/PDF、158行训练CSV、17行验证CSV及数值核验通过。[Pair特征与中文曲线复核](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_PAIR_FEATURE_REVIEW_20261023.md)。
2026-10-24T06:48:07.307840+00:00 AB500K两组3000步均COMPLETE=success（各约3.07epoch）；固定4096条×3mask终点AA CE：MLM 0.288281123、Pair 0.289064492，Pair相对高0.271738%。KD从3.414345降至0.584299（−82.887%），但两组最佳AA CE仍是step0；本轮没有MLM收益。训练会话耗时MLM20.74分钟、Pair150.52分钟（7.26倍，不含教师生成及启动资格）。已刷新[完整中文曲线](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/loss_completed_20261024/loss_explained_zh.png)及[共同终点数据](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/loss_completed_20261024/summary.json)。仅本地读取与绘图，未改训练或提交作业。
2026-10-24T07:09:47.671073+00:00 Pair架构探索完成：对照Graphormer、Uni-Mol、ESMFold v1、AF2/AF3原论文及官方代码，区分跨层权重共享、Pair状态传递、recycle复用。当前共享动态bias缺少显式跨层z状态；共享本身未证实是退化原因。优先候选为低成本持久Pair状态与独立层级更新，并保留仅解除共享的对照；后者按指定构造估计总76.58M，不天然加速。候选未实现或提交训练。[架构对照与实验设计](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PAIR_ARCHITECTURE_EXPLORATION_20261024.md)。
2026-10-24T07:24:38.997184+00:00 按用户要求准备AB500K从零训练：两组旧学生权重继承0，直接随机初始化seed42，共享78个state张量逐值一致，2条真实样本初始logits一致；MLM/Pair参数75.60M/75.73M，保留当前共享Pair架构及已验收缓存。新独立包默认9770步=10epoch、主干/Pair LR均1e-4、warmup500；23项CPU测试、两组84分片/缓存入场、冻结源码和shell/YAML核验通过。GPU profile/8rank短测待用户提交，本轮无GPU或云端submit。[新计划与两个启动命令](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_SCRATCH_PLAN_20261024.md)。
2026-10-24T07:47:46.897119+00:00 AB500K Pair单卡速度探索完成：A10080、micro64/长度310，两批真实输入确认原版约2.46秒/更新；对称计算＋提前dtype转换约1.43–1.45秒、allocated5.44GiB，再关闭Pair分块重计算约1.20秒、allocated50.11GiB。8项CPU测试及两批非零反馈完整模型输出/梯度检查通过；编译候选未过预设logits门槛并保留失败结果。单卡更新速度约1.7/2.05倍，不代表八卡端到端性能。已保存脚本、原始测量、证据SHA和PNG/PDF；未改scratch冻结包或提交云端任务，八卡资格待正式接入。[性能诊断与候选方案](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PAIR_SPEED_EXPLORATION_20261024.md)。
2026-10-24T08:04:15.397210+00:00 按用户“每层一个Pair网络”补充官方源码复核：ESMFold ModuleList、AF2/AF3 layer_stack均支持层级独立参数，并持续传递pair状态；Uni-Mol提供较轻量的逐层关系更新参照。建议独立PairUpdate＋残差z作为下一版候选，保留仅解除共享的U对照；CPU实例化核对8个encoder均120832、合计966656参数且存储独立，指定U总模型约76.58M。已将旧文档的父BERT初始化建议改为遵循用户要求的同一随机主干。独立权重不自动提速；有向/三角更新不能照搬对称加速。仅研究与CPU参数核算，未改训练或提交任务。[参考及适配方案](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PAIR_ARCHITECTURE_EXPLORATION_20261024.md#8-针对每层有一个pair网络的官方实现复核)。
2026-10-24T20:50:34.352551+00:00 原生Pair容量已核算：教师256维pair＋64维distogram，扫描504096条请求真实长度，500k训练平均H+L231.503。全有向L²、最终一份教师状态、16位未压缩本体为训练17.157TB、加验证17.297TB；FP32两者均原样存储则翻倍。合格OAS训练池实际2160824条，外推约74.145TB；400万同长度场景约137.253TB，非已核准全量AB计数。4096有向位置抽样为另选项，不静默替代全位置目标；学生8层共用教师最终目标无需8份缓存。未生成数据或提交GPU。[容量依据与范围](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PAIR_FEATURE_STORAGE_20261024.md)。
2026-10-24T21:16:05.107257+00:00 AB100K仅distogram候选已设计：10万训练＋2000dev＋2000test按既有长度外推FP16全方阵713.72GB，对称验收后上三角358.40GB，保留链内/跨链完整位置。候选A/B/C/D区分MLM、KD和独立层反馈；同随机主干、20epoch3920步筛查及50epoch9800步延长候选。主证据为冻结hidden的新实验结构接触探针和功能探针，KD仅诊断；原Intra/Inter及Mask-aware验证范围保留，严格划分/教师质量/三seed须实际执行。尚未抽样、生成缓存、实现新模型或提交作业。[AB100K候选方案](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB100K_REPRESENTATION_PLAN_20261024.md)。
<!-- /stage1-ab500k-offline-pilot-20261022 -->

<!-- diffusion-edit-latest-single-gpu -->
2026-10-23T21:19:56Z diffusion/edit76500的15/15项评测已完成；按用户要求将全部成绩回填RESULTS与RESULTS_SUMMARY原任务表，21个表项与18份指标核对通过，保留原有结果及来源。PROGRESS已同步，本轮无GPU启动或云端submit/cancel。[发布与验收](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#diffusion-edit-step76500-local-20261023)。
<!-- /diffusion-edit-latest-single-gpu -->

<!-- stage1-online-20261022 -->
2026-10-22T19:27:24Z 在线Stage1任务`t-20261023004236-rwrkk`在19:24Z平台查询仍Running；第79批373/373目标齐全、最后metadata19:27:15Z，第80批当次扫描0/353，据同步执行顺序推断至少完成78次更新，非直接优化器读数。正式日志仍step3，后续每100步记录；目标10k步，每500步验证/保存。新增只读速度诊断：20个完整批、7275条metadata，近千残基请求单条73–77秒，第58批各卡目标完成相差80.22秒；前三步教师准备区间重建占87–89%，首批前启动约18.1分钟、前79批平均约111.3秒/批。主要等待来自逐条在线教师及长输入，未取得算子profile/实时GPU利用率；日志无Traceback/OOM。micro16/global512及64.21GiB资格结果沿用，无submit/cancel、运行包或缓存修改。[速度诊断与限制](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_ONLINE_DISTILLATION_20261022.md#训练速度诊断2026-10-22)、[只读证据](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/online/log_review/timing_diagnosis_20261022.json)。
<!-- /stage1-online-20261022 -->

<!-- stage1-cache-capacity-20261022 -->
2026-10-22T18:08:26.931853+00:00 Stage 1缓存容量只读核验：12,750条实测分配7.24 GB，平均0.568 MB/条；logits落盘FP32。按前三批70.38%可监督比例估计，本轮10k×512约360万请求、2.05 TB，访问全训练集一次约2.50 TB；建议本轮预留2.5–3 TB，仅缓存、不含权重/checkpoint。比例与分布仍为早期估计，共享df不代表项目quota；未改变运行代码/精度或提交任务。[容量实测](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/online/log_review/cache_storage_estimate_20261022.json)。
<!-- /stage1-cache-capacity-20261022 -->

<!-- stage1-cache-compression-20261022 -->
2026-10-22T18:16:48.111459+00:00 仅分析缓存压缩：64条既有缓存CPU往返试验完成，未改训练/缓存文件。保留FP32的索引缩窄＋字节重排＋Zstd外推约1.52TB；紧凑FP16方案约715GB，目标分布KL均值1.30e-6。建议按空间预算设计容量上限，复访重算仍保留当步KD；尚未选定/实施。[压缩候选方案](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_ONLINE_DISTILLATION_20261022.md#缓存压缩候选方案2026-10-22仅分析)、[64条缓存压缩核验](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/online/log_review/cache_compression_feasibility_20261022.json)。
<!-- /stage1-cache-compression-20261022 -->

<!-- stage1-lmdb-feasibility-20261022 -->
2026-10-22T18:22:17.466028+00:00 本地临时LMDB PoC通过：64条压缩写入、关闭重开、随机读取/解码64/64成功，35.73MB→13.00MB，外推本轮约734GB；FP16有损，与原FP32并非逐值一致。只验证开发容器/tmp，未验证训练容器或共享GPFS；原缓存/运行包未改，临时库已删除。[本地LMDB往返验证](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/online/log_review/lmdb_compressed_cache_poc_20261022.json)、[LMDB候选设计](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_ONLINE_DISTILLATION_20261022.md#lmdb候选存储与实际挂载约束)。
<!-- /stage1-lmdb-feasibility-20261022 -->

<!-- stage1-validation-complete-20261022 -->
2026-10-22T15:30:29Z 合并工程验证完成：profile micro24/reserved65.05 GiB，8/8恢复通过、Adam重放差0；教师8worker通过，train768/valid761共1529条、2,257,088对，全部同seed重跑/对称差0，教师峰值allocated23.22 GiB。核对总回执/审计/索引哈希；正式长训未启动，仍缺实验结构质量评估及训练缓存覆盖（目前0.012288%）。按用户要求新增60秒控制台汇总，后续训练默认100步记录；14项日志/合并流程测试通过。没有新submit/cancel或重跑GPU。[最新核验](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/validation_review_20261022.json)、[执行与日志](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINING_PIPELINE_20261022.md)。
<!-- /stage1-validation-complete-20261022 -->

<!-- stage1-combined-checks-20261022 -->
2026-10-22 按用户要求合并Stage 1后续检查为一个[launch_checks入口](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/launch_checks/entrypoint.sh)：8卡smoke→ESMFold2 pilot（同seed重跑）→原子建索引→缓存工程审计，同一8卡节点顺序执行。完成的smoke可复用，缓存可校验后续接。新增7项CPU测试通过；合并入口84分片/父BERT/教师6分片权重CPU预检通过，未代提交或启动GPU。现有实验结构质量评估未接入，报告保留unvalidated及实际覆盖不足，不自动启动正式训练。[执行说明](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINING_PIPELINE_20261022.md)。
<!-- /stage1-combined-checks-20261022 -->

<!-- stage1-profile-smoke-20261022 -->
2026-10-22 Stage 1用户运行的8卡profile已通过：A10080/max_length2048/每记录8192对，micro24 allocated56.26/reserved65.05 GiB；micro32 reserved77.58 GiB余量不足，micro64/48 OOM。已按真实回执新建 `launch_smoke`，84分片/父权重/资格CPU预检通过、blockers为空，下一步由用户提交8卡训练/完整恢复短测；正式global512计划每卡24+24+16。教师缓存目前仍只有工程训练2条，正式教师质量、覆盖与验证缓存未就绪。本次未代提交，未知平台任务ID，不新增Active条目。[profile回执](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/profile/COMPLETE.json)、[smoke入口](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/launch_smoke/entrypoint.sh)、[执行说明](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINING_PIPELINE_20261022.md)。
<!-- /stage1-profile-smoke-20261022 -->

<!-- stage1-training-build-20261022 -->
2026-10-22 Stage 1训练管线已构建：全量V7自然比例读取、SQLite教师缓存索引、DDP全局损失、warmup、独立验证及完整状态恢复；16项CPU/教师特征测试通过，含两进程Gloo不均匀KD与精确恢复。冻结profile入口完成84分片及BERT权重预检；教师作业固定6分片权重。已生成profile、teacher pilot、smoke/train候选YAML，均未提交；正式候选明确缺新GPU profile、8卡smoke、质量与覆盖资格。新增教师请求1785条/可用完整链1529条，推理尚未执行。[Stage 1训练构建与提交](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINING_PIPELINE_20261022.md)。
<!-- /stage1-training-build-20261022 -->

<!-- stage1-structure-local -->
2026-10-22 Stage 1本地工程验收完成：全8层Pair Bias从BERT287000完整继承75,601,176原参数，新增131,136参数；ESMFold2-Fast固定revision真实加载6.534B冻结参数，严格无缺失权重。40项测试通过，两条原生OAS配对教师缓存6144对，输出对称且固定seed重放完全一致；真实MLM+KD联合更新3步，全8层反馈及主干获得梯度；checkpoint2→3恢复与连续运行的模型、Adam、RNG和游标完全一致。尚未验证预训练收益，当前仅工程pilot；[实现记录](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_STRUCTURE_IMPLEMENTATION_20261022.md)、[验收回执](/project_c20250601/251105016/project/dllm_test/output/stage1_structure_20261022/verification.json)。
<!-- /stage1-structure-local -->

Last updated (UTC): 2026-10-27T05:36:55.899292+00:00

<!-- diffusion-edit-downstream-16500 -->
2026-10-20T08:56:18.791328+00:00 diffusion/edit16500下游完成15/15；6组非闲时评测：g0=Success, g1=Success, g2=Success, g3=Success, g4=Success, g5=Success；本地交接状态handed_off。[执行](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#diffusion-edit-step16500-20261020)、[作业](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/jobs.json)。
<!-- /diffusion-edit-downstream-16500 -->

2026-10-19 CoV-UniBind按用户最新要求转为作者划分核查：公开资产缺原fold信息，新方向训练未启动；已完成官方分数复核，既有整体结果改列补充。[执行](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#covunibind-author-split-audit-20261019)。

<!-- bert-diffusion-continuation-status -->
2026-10-22T10:22:08.761565+00:00 用户网页新任务 `t-20261022181600-wcfcx`（`cq9rf89httddx5ka`）平台Running；八rank已于10:18:49 UTC从完整checkpoint28000恢复模型、AdamW、scheduler与各rank RNG，恢复回执8/8通过。当前观察28075/200000，world_size=8；新checkpoint28025的16文件SHA与八rank参数一致已验证。沿用W&B `diffedit287k200k1020`；服务端旧日志游标28201，高于恢复点28000，故暂时忽略较小step的重复日志；本地metrics继续保存且训练更新正常，超过该游标后继续显示新点。每60秒只读监控已绑定当前任务，依赖开发机进程存活。[监控](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/monitor_status.json)、[新保存点验收](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/restart_checkpoint_verification.json)。
<!-- /bert-diffusion-continuation-status -->

Last updated (UTC): 2026-10-27T05:26:50.702506+00:00

2026-10-19 CoV-UniBind整体共享Ridge评估及独立验收完成，当前可用缓存共23项；5项历史缓存缺口明确保留，结果已回填任务表，无GPU作业变更。[执行](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#covunibind-pooled-5cv-20261019)。

2026-10-19 CoV-UniBind Ag+H+L Ridge五折重评完成：8组完整缓存均已训练和验收，结果写入两份任务结果表；本次为CPU缓存评估，无平台作业变更。[执行记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#covunibind-aghl-ridge5cv-20261019)。

2026-10-19T00:56:04.027713+00:00 OAS-only 续训未在跑：r3 `t-20261019021523-8psvh` 仍 Queue（已等约6.7小时），r2 `t-20261017122656-nzn5w` 已于 2026-10-18T04:26:08Z Failed。训练输出无新步，完整保存仍是 checkpoint-11000。[平台查询](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_resume11000_20261018/platform_status.json)。

<!-- bert-task-table-sync-20261018 -->
2026-10-18T14:31:55+00:00 BERT48828/107000结果已回填各任务原表；按用户要求移除结果页顶部更新说明/对照及重复附录，只在对应任务展示。任务表数字保留，未改变评测任务。[排版核对](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_representation_review_20261018/task_layout_verification.json)、[任务榜单](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_SUMMARY.md)。
<!-- /bert-task-table-sync-20261018 -->

<!-- final-layer-ce-ablation-20261018 -->
2026-10-19T02:26:15.829111+00:00：pool-only1000已完成；feasible-token组2573065继续，两个100步实际bundle/optimizer/冻结norm及留出监测配对核验通过。当前监测不是下游结果；旧m396及新终点任务队列继续，目标未达。[证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#feasible-final-paired-screen100-20261019)。
<!-- /final-layer-ce-ablation-20261018 -->

<!-- final-layer-chain-preservation-20261018 -->
2026-10-18T03:54:53.883492+00:00：分链、8192覆盖量、后训练归一化和全局注意力审计均完成，相关自有进程已终态。HIC改善但尚未整体追平第二层，目标未完成。[末层最新实验结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-latest-audits-20261018)。
<!-- /final-layer-chain-preservation-20261018 -->

<!-- final-layer-attention-alignment-20261018 -->
2026-10-18T03:20:37.465233+00:00 attention训练对齐两组1700与20项OOF／重载完成；梯度、OAS线性恢复、分布误差、条件OOF区间均完成。对齐在同预算下改善Titer但仍未达到第二层，全部本轮自有进程终态，目标active。[结果与范围](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-attention-alignment-results-20261018)。
<!-- /final-layer-attention-alignment-20261018 -->

<!-- final-layer-spectral-pilot-20261018 -->
2026-10-18T03:04:11.935021+00:00 本轮普通／谱加权1000步、MASK交互、关系300的Specificity／m396全部本地完成并验证，所有自有session终态；HIC接近第二层，但其他属性及候选确认尚未满足，目标active。[完整结果与范围](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-mask-interaction-20261018)。
<!-- /final-layer-spectral-pilot-20261018 -->

<!-- final-layer-preservation-pilot-20261018 -->
2026-10-18T03:04:11.935021+00:00 四种300步训练与关系联合模型保持性全部完成；Specificity、m396、T3-broad提高，Kong H3略降。不同模型证据分开，不替代谱加权候选确认。[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-preservation-pilot-20261018)。
<!-- /final-layer-preservation-pilot-20261018 -->

<!-- fixed50000-depth-diagnosis-20261018 -->
2026-10-18T02:22:37.280163+00:00 层间机制定位两轮本地实验完成，155项OOF复核通过；识别MASK尾部attention影响及第八层几何变化，完成一手文献与后训练方案，目标仍active。未改变默认模型或提交Volc。[研究记录](/project_c20250601/251105016/project/dllm_test/docs/LAYER_REPRESENTATION_DIAGNOSIS_20261018.md)、[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#fixed50000-depth-diagnosis-20261018)。
<!-- /fixed50000-depth-diagnosis-20261018 -->

<!-- fixed50000-gdpa1-pooling-20261018 -->
2026-10-18T02:06:14.743425+00:00 fixed50000 GDPa1六组池化诊断完成，特征/30组OOF及数值控制通过；原12项评测继续独立运行，无新Volc提交。[完整结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#fixed50000-gdpa1-pooling-20261018)。
<!-- /fixed50000-gdpa1-pooling-20261018 -->

<!-- bert-step122070-representation-20261018 -->
2026-10-18T17:48:09.253689+00:00 BERT122070本地评测已复核7/13项并回填对应任务表，继续运行中。[结果与状态](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#bert-step122070-representation-20261018)。
<!-- /bert-step122070-representation-20261018 -->

<!-- bert-step166000-representation-20261018 -->
2026-10-18T17:48:09.253689+00:00 BERT166000本地评测已复核9/13项并回填对应任务表，继续运行中。[结果与状态](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#bert-step166000-representation-20261018)。
<!-- /bert-step166000-representation-20261018 -->

<!-- bert-step107000-representation-20261018 -->
2026-10-18T14:18:10.874036+00:00 BERT107000本地13/13成功，结束于05:46:08 UTC；保存预测复核通过。[完整对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#bert-checkpoint-comparison-20261018)、[48828复核](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_representation_review_20261018/verified_48828.json)、[107000复核](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_representation_review_20261018/verified_107000.json)。原5组云端Killed，未重提。
<!-- /bert-step107000-representation-20261018 -->


<!-- fixed50000-layer2-20261018 -->
2026-10-18T14:18:10.874036+00:00 fixed50000第二层12/12任务success、6组exit0，driver于03:52:55 UTC正常结束；无Volc提交。[状态快照](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_representation_review_20261018/status_snapshot.json)。本次核对执行终态，不宣称新完成全部分数复算。
<!-- /fixed50000-layer2-20261018 -->

<!-- lora-layer2-representation-20261017 -->
2026-10-18T01:01:17.438928+00:00 第二层本地13入口结束，7组exit0；SPR探针按原协议未拟合，完整模型PLL保持。指标/划分核查完成，[全部对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#lora-layer2-complete-results-20261018)、[终态](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_layer2_20261017/summary.json)。原7个Volc任务均已取消，本轮无新提交。
<!-- /lora-layer2-representation-20261017 -->

2026-10-17 旧V5/fixed112000→LoRA3000完整测试暴露审计及GDPa1层级CPU诊断完成；发现Specificity/GDPa1预训练配对重叠，结果页已明确不可作严格隔离headline；层级读出提供新诊断证据，正式默认与历史分数未改。无GPU提交。[执行、复核与限制](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#ab-exposure-and-layers-20261017)。

<!-- bert-representation-suite-20261017 -->
2026-10-18T14:18:10.874036+00:00 核查：BERT107000为13/13完成；best48828为27/29完成，其中13个表征/结合入口全部完成，AbBiBench14/16完成。4fqi_h1正在本地计算，4fqi_h3待执行；当前无失败任务。fixed50000第二层12/12完成。[状态快照](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_representation_review_20261018/status_snapshot.json)。 当前4fqi_h1=13141/65094，下一项4fqi_h3共65535行。原9组云任务均终态；本地driver及GPU仍正常运行。[完整对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#bert-checkpoint-comparison-20261018)。
<!-- /bert-representation-suite-20261017 -->


<!-- ophiuchus-gdpa1-reproduction-20261017 -->
2026-10-17 Ophiuchus-Ab GDPa1本地重新复现完成；官方源码/权重全量新特征、逐token输入、保存预测与双回归入口复核通过，结果已回填；无GPU提交。[执行与实时状态](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#ophiuchus-gdpa1-reproduction-20261017)。
<!-- /ophiuchus-gdpa1-reproduction-20261017 -->

2026-10-17 GDPa1 / AirGen 方法核查及LoRA全量CPU读出诊断完成；无GPU提交或训练变更。[执行与验收](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#gdpa1-airgen-audit-20261017)。

<!-- bert-full-token-status -->
2026-10-19T17:03Z 按用户要求停止BERT预训练，最后日志287400步；最新完整287000已复制并核验独立推理快照，作为diffusion/edit续训起点。原训练被主动中断（supervisor interrupted=true），不宣称300000步完成；八卡容器standby并保留，PAUSE_TRAINING阻止自动接续。用户已确认条件序列不mask；新目标已通过八卡验证并完成首阶段10000步，当前状态见上方续训记录。[停止证据与起点](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_20261019/pretraining_stop.json)。
采样／调度／EOS focal工具及6项CPU测试完成；用户已澄清，进入八卡profile及保存恢复短测。[续训准备](/project_c20250601/251105016/project/dllm_test/docs/BERT_TO_DIFFUSION_EDIT_20261019.md)。
<!-- /bert-full-token-status -->

2026-10-16T11:14:43.384403+00:00 BERT V7正式运行已核实推进至125/300000步，global512、world8，累计访问64000条；checkpoint50所有文件SHA、8份RNG及rank参数一致性检查通过。新账号waitma的pressure_jian/dllm-immune正式run已从服务器读回训练历史。数据、从头初始化、300k终止条件与同版评估均已落实。[启动核实](/project_c20250601/251105016/project/dllm_test/output/bert_mlm_v7_300k_20261016/preparation/startup_verification.json)；[W&B](https://wandb.ai/pressure_jian/dllm-immune/runs/bertv7mlm300k1016)；[训练说明](/project_c20250601/251105016/project/dllm_test/configs/models/v7_bert_mlm_75m_300k_20261016/README.md)。


2026-10-16T11:11:55.347946+00:00 BERT V7正式票据已提交既有八卡容器：slots167/135、MHC-completed，train6,249,956条，300000步（24.576轮），global512=micro32×GA2×8，随机初始化。84分片完整哈希、11项测试、八卡max_length2048显存、8rank保存恢复、FP32/math逐位一致复算、484条尾批和开发验证均通过，smoke_r4真实exit0。新W&B正式run为pressure_jian/dllm-immune/bertv7mlm300k1016；训练和测试代码绑定同一冻结包。[配置与同版测试](/project_c20250601/251105016/project/dllm_test/configs/models/v7_bert_mlm_75m_300k_20261016/README.md)；[YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_fixed_chain_75m_v7_300k_8gpu_20261016.yml)。本次复用已分配容器，没有重复提交平台资源。


2026-10-16 BERT 训练范围更正：按用户最新指令，V5 10-epoch 试运行已于 10:50 UTC 通过 PAUSE_TRAINING 中止（supervisor interrupted=true，子进程已回收），既有八卡容器继续 standby。新正式运行改用 V7 slots167/135、MHC-completed 数据，目标恰好 300000 optimizer steps；从头初始化。新配置、完整数据绑定及八卡检查正在准备。


2026-10-16T10:45:07.158708+00:00 历史V5启动记录（10:50已按用户更正中止）：BERT十轮正式训练曾启动并核实推进到150/148970步（loss=2.5830，global512，八rank）。checkpoint-50完整保存，8份rank RNG和参数一致性检查通过；新账号waitma在pressure_jian/dllm-immune的正式run已从服务器读回训练日志。主实验V5七来源，10完整epoch，从头训练75.6M；当前仍在第1轮。[启动核验证据](/project_c20250601/251105016/project/dllm_test/output/bert_mlm_10epoch_20261016/preparation/startup_verification.json)；[W&B](https://wandb.ai/pressure_jian/dllm-immune/runs/bertv5mlm1016)；[训练与同版测试说明](/project_c20250601/251105016/project/dllm_test/configs/models/v5_bert_mlm_75m_10epoch_20261016/README.md)。


2026-10-16T10:41:36.221577+00:00 正式BERT训练票据已入现有八卡容器：`bert_mlm_train_20261016.json`，平台容器`t-20261007153431-vhx2f`；V5多来源75.6M从头全参数训练，10完整epoch/148,970步，global512=micro32×GA2×8。CPU7项、85分片全哈希、八卡长度2048 profile、八rank原生恢复与133条尾批/真实spawn DataLoader检查均通过，r3票据exit0。W&B正式新run绑定waitma/pressure_jian/dllm-immune（bertv5mlm1016）。[YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_fixed_chain_75m_v5_10epoch_8gpu_20261016.yml)；[冻结启动包](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_bert_mlm_train_20261016)。复用已分配资源，本次没有重复提交新的平台容器。


2026-10-16 BERT smoke r2差异定位：恢复后的loss、selected_count和LR完全一致，参数最大差值7.45e-9，梯度范数相对差约3.6e-7；严格逐位比较把并行浮点舍入差异判为失败。r3增加恢复模型/AdamW/调度器/RNG逐项精确检查与实际noised batch精确比较，对后续更新采用显式atol1e-8/rtol1e-6容差；旧失败票据与诊断保留，新增r3票据已入池。正式训练未启动。


2026-10-16 BERT八卡启动检查：初次smoke已验证新W&B账号waitma可写入pressure_jian/dllm-immune，但逐位续训比较未通过；旧票据已失败归档，不原样重试。已修正AdamW恢复时step计数器的设备映射，新增逐参数差值诊断和真实133条DDP尾批／spawn DataLoader检查，CPU7项通过；修订票据 `bert_mlm_smoke_r2_20261016.json` 已进入同一空闲容器。正式10epoch训练尚未启动。


2026-10-16 用户授权新 BERT 表征预训练至少10完整epoch，正在实施并启动检查：V5主实验多来源7,626,885条，global512，148,970步，从头训练固定双链75.6M；15%选中、80/10/10残基恢复，85分片哈希及全量合格索引完成。八卡显存检查已成功，采用micro32×GA2×8/global512；七项CPU测试通过，正在准备原生保存／恢复和W&B在线检查，尚未正式开始十轮训练。W&B新身份waitma已核实，entity/project与新版SDK连接检查进行中。[准备证据](/project_c20250601/251105016/project/dllm_test/output/bert_mlm_10epoch_20261016/preparation)。


2026-10-16 早期表征方案记录（已由上方10epoch正式训练取代）：用户选择先做表征模型；A1 建议采用动态 15% 选中＋80/10/10、选中受体 AA 上 CE，首轮八卡 global512／3 完整 epoch，按冻结表征开发任务选点。已补查 BERT、LLaDA2.1、GIDD 与当前数据规模；论文依据、预算估算和待实施项见[专项方案](/project_c20250601/251105016/project/dllm_test/docs/BERT_RESIDUE_EDIT_PRETRAINING_20261016.md#first-bert-run)。本次没有新训练实现或提交，现有作业不自动切换。

2026-10-16 调研范围纠错完成：OAS-only 按用户澄清保留为数据来源对照，撤回对主实验的 OAS-only 限定；没有切换或启动训练。[更正记录](/project_c20250601/251105016/project/dllm_test/docs/EXPERIMENT_LOG.md)。

2026-10-16 OAS-only 已从 checkpoint-11000 续训并进入 queue012：task_id=`t-20261016162026-4rq8w`，平台状态 `Queue`，1 × `ml.pni2.28xlarge`，`Preemptible: false`。沿用原冻结源码与 300k/global512；W&B 使用当前 `waitma` 凭据、显式项目 `dllm-immune` / entity `models`，新 run `fixed-chain-75m-oas-resume-11000`，不接旧 run。提交前输出目录写探测通过。[YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/v5_fixed_chain_75m_oas_resume11000_8gpu_queue012_20261016.yml)；[平台查询](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_resume11000_20261016/platform_status.json)。

2026-10-16 已按用户授权清理旧 8B 评测结果与当前汇总表已收录的特征缓存，删除及保留验收通过；新 CDR 作业及其父实验依赖保留，未提交或取消任务。[清单、空间与核验](/project_c20250601/251105016/project/dllm_test/output/feature_cache_cleanup_20261016/README.md)。

<!-- cdr-left-to-right-status -->
2026-10-16 严格从左到右CDR验证完成：两checkpoint作业均Success，全部预测、真实轨迹与来源/成本审计通过；H3均明确低于单步，未缓解退化，正式默认保持。[完整结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-left-to-right-results-20261016)。
<!-- /cdr-left-to-right-status -->

2026-10-16 W&B 新账号 `waitma` 登录及训练凭据同步完成；身份验证通过。用户已取消 W&B 清理，本次未创建、删除或修改远端 run。[验证记录](/project_c20250601/251105016/project/dllm_test/output/wandb_account_switch_20261016/login_verification.json)。

<!-- sample-cleanup-status -->
2026-10-16 已记录采样产物清理完成：最终移除227文件、约311.18 GiB，指标、模型与特征缓存保留。期间误删两个并发CDR任务需要的父预测，现已按原完整SHA256恢复，两项原冻结CPU审计均passed；恢复GPU作业Success，原预算作业Failed历史保留。[完整清理、事故及核验](/project_c20250601/251105016/project/dllm_test/output/sample_cleanup_20261016/README.md)。
<!-- /sample-cleanup-status -->

<!-- cdr-block-budget-status -->
2026-10-16 新方法4/8前向验证与完整审计已完成：父依赖已按原SHA恢复，原冻结审计在CPU通过；两GPU任务审计阶段Failed终态保留。增加前向未显示明确收益，未改变正式默认。[完整结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-block-budget-results-20261016)。
<!-- /cdr-block-budget-status -->

2026-10-16 W&B 清理已由用户取消：此前仅完成旧账号 codema 的只读盘点（主项目92个run、两个早期项目各3个），未执行远端删除。历史候选与未完成的备份核查仅供记录，不继续执行。[审计目录](/project_c20250601/251105016/project/dllm_test/output/wandb_cleanup_20261016)。

2026-10-16 按用户授权清理V7之前未测试的旧checkpoint：106个旧系列中保留37个，清理69个的252份模型/训练状态文件；扣除保留硬链接后释放文件块255.07 GiB。配置、日志、指标、来源及272份目录内元数据保留；172个保留权重和两份结果表检查通过。最近native/fixed-chain/OAS/LoRA及V7未纳入本次删除，无任务提交或取消。[完整删除/保留清单与核验](/project_c20250601/251105016/project/dllm_test/output/checkpoint_cleanup_20261016/README.md)。

2026-10-16 用户新提供 TCRvdb 文件核查完成：1 个 CSV、2,034 条成对 TCR 记录，910 阳性 / 872 阴性 / 252 ambiguous，五种 peptide、两种 HLA-A。序列/长度及来源哈希检查通过；全部 2,034 行命中冻结测试 CDR3 配对保护键（其中 1,974 命中 T4 β-core＋peptide），按现有规则不加入训练。原始文件与训练数据未改。[字段、统计与命中证据](/project_c20250601/251105016/project/dllm_test/docs/TCRVDB_INSPECTION_20261016.md)。

<!-- cdr-block-ablation-status -->
2026-10-16 分组机制消融双checkpoint作业均Success，13策略全量预测、反馈和来源审计通过，固定主要检验与描述性结果已完整交付，无winner选择。软分布组件获支持，完整方法仍未超过单步，正式默认保持。[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-block-ablation-results-20261016)。
<!-- /cdr-block-ablation-status -->

2026-10-16 八卡训练效果核实：OAS-only最新11000步在固定256条验证集上，全mask AA准确率45.14%、首EOS位置正确13.48%、长度MAE6.19 aa；6000步后验证loss约1.62，未持续下降。LoRA已成功完成33981步、1个来源均衡采样epoch（八rank success、COMPLETE及票据exit0一致），最佳22000步loss=1.807105，最终1.807499；持续八卡资源容器仍Running、内部standby。两种验证数据不同，不能直接比较loss；上述单步重建指标不是下游生成成绩。[状态、曲线与证据](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_oas_300k_b512_20261015_supervision/review_20261016/training_status_and_metrics.json)。

2026-10-15 表征预训练→可编辑生成调研已形成文档：覆盖 25 项原始工作，核对当前模型加噪、时间输入、布局和生成限制，完成核心论文与方案的独立复核；实验仍为建议，未提交训练或修改数据。[报告与交付核查](/project_c20250601/251105016/project/dllm_test/docs/REPRESENTATION_TO_EDIT_GENERATION_REVIEW_20261015.md)。

<!-- cdr-block-status -->
2026-10-15 分组软上下文开发与双checkpoint确认两个Volc作业均Success，全量预测/来源/概率审计通过。新方案降低前向成本，但未通过跨checkpoint稳定优于单步的标准；本轮验证完成，整体改善问题尚未解决，正式默认保持。[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-block-results-20261015)。
<!-- /cdr-block-status -->

2026-10-15 MHC 补全完成并发布独立新 V7：67,648 条（train 64,369 / valid 3,279），含 MHC-I 67,614、MHC-II 34；总样本 6,318,941、167/135 槽位及原 split 保持。39 项测试、全部改变记录的真实 collator、全数据逐行独立比较、8 个来源文件 104,720 条证据回查及两套 reader 验收通过，最长 845 token。未切换已有训练。[新数据与验收](/project_c20250601/251105016/project/dllm_test/docs/V7_MHC_COMPLETION_20261015.md)。

2026-10-15 DecoderTCR 调研完成：核对 ICML 正式论文、固定 commit 官方实现及全部 88 个 MHC 参考键，复现 Class II allele 查询接口 45/45 失败；确认双链 MHC、pMHC 组分监督与 IEGR 的适配边界，区分论文与 V0.3 版本。报告给出覆盖感知 MHC 恢复、pMHC 辅助任务和独立解码对照的建议顺序；本轮未改训练数据或运行模型。[报告与来源](/project_c20250601/251105016/project/dllm_test/docs/DECODERTCR_RESEARCH_20261015.md)。

2026-10-15 双链MHC数量解释更正：112条为旧参考字符串一致性粗筛的样本数，并非112种重链或可接入上限；全部对应E01:03的一种358 aa序列。A02:01的504个FASTA条目对应10种字符串，最长365 aa序列唯一，其余均为精确连续子串，重叠无冲突。原“歧义”分类混入部分覆盖差异，已在[探索报告](/project_c20250601/251105016/project/dllm_test/docs/DUAL_MHC_DATA_EXPLORATION_20261015.md)更正；其他allele及全量候选需重新按覆盖一致性评估，尚未声称新的可入库总数。

2026-10-15 双链MHC数据探索完成：已核实本地MHC-I heavy+B2M恢复来源、TCR3d/RCSB结构来源和VDJdb MHC-II记录，查阅IEDB两链allele筛选入口。当前V7可唯一匹配39,048条双链上下文提案，全部通过167/135真实collator检查；严格完整MHC参考唯一映射仅112条（train79/valid33）。TCR3d人源MHC-II 57个结构中45个命中当前精确配对/T4测试键，余12个已下载mmCIF并提取实体序列，尚待结构/融合/可变域及最终入库核查。VDJdb人源2,234个唯一配对条件中，两条参考均唯一的37个全部命中测试保护集合。候选清单、来源哈希、脚本语法和12个结构下载完整性复核通过；没有修改训练数据或启动训练。[详细探索与接入顺序](/project_c20250601/251105016/project/dllm_test/docs/DUAL_MHC_DATA_EXPLORATION_20261015.md)。

<!-- cdr-marginal-status -->
2026-10-15 新方向探索与冻结确认五个作业全部Success，完整预测/参考/历史对照验收通过。软上下文有受限微弱正向信号，跨checkpoint及成本尚不足以认定实用修复；正式默认保持，探索目标仍active。[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-marginal-results-20261015)。
<!-- /cdr-marginal-status -->

2026-10-16 OAS-only scratch 失败根因更正：task_id=`t-20261015124455-xrns4` 平台终态 `Failed`（End=2026-10-15T16:05:37Z）。rank0 写 `step_timing.jsonl` 时触发 `OSError: [Errno 122] Disk quota exceeded`，离开训练后其余七rank等待 NCCL BROADCAST 1800s 超时；旧尾日志漏掉写盘异常。最后记录11845步，完整保存11000，best3+latest 为6000/7000/9000/11000。应先处理输出路径适用的存储配额，再准备从11000恢复；未重提。[完整证据及更正](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_queue012_20261015/failure_review.json)。

<!-- cdr-inference-rounds-status -->
CDR免训练三轮评测已完成：三个Volc作业Success、全部预测和轨迹核对通过，冻结配置完成跨checkpoint确认。H3尚无稳定优于单步的候选，正式默认保持。[三轮诊断结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-inference-rounds-results-20261015)。
<!-- /cdr-inference-rounds-status -->

2026-10-15 V5→V7 数据范围纠错完成：恢复 repertoire 参与清理，新版本 train 6,249,960 / valid 68,985，合计 6,318,945。全量选择复算、原始字节及实际 reader 验收通过；旧 194,237 条严格 binding 输出标为历史错误范围。保留 design 布局，未启动或切换训练。[数据与验收](/project_c20250601/251105016/project/dllm_test/docs/V7_FROM_V5_REBUILD_20261015.md)。

2026-10-15 Flow/diffusion推理调研完成：18篇一手论文、源码兼容性和分阶段实验方案已核对。优先ReMDM纠错与EB自适应提交；连续flow与蒸馏另需训练。本轮只调研，未运行新sampler或修改默认。[报告](/project_c20250601/251105016/project/dllm_test/docs/INFERENCE_FLOW_DIFFUSION_REVIEW_20261015.md)。

Last updated (UTC): 2026-10-25T13:29:47.387664+00:00

2026-10-15 CDR步数退化首轮诊断完成：fixed50000完整SAb23H2 H3八组真实CPU对照及480条独立复算通过，原始2步与历史60条逐token一致；归一化排序未证明稳定改善，生产默认保持。[机制、证据与剩余问题](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#cdr-steps-debug-20261015)。

2026-10-15T02:58:11.613007+00:00 fixed reference默认采样器对齐已完成，两个真实checkpoint核查通过；定向回归3项、各500条输入及各2例冻结实现CPU逐token比对通过，历史指标及冻结作业保持。[协议与验证记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#pairing-alignment-20261015)。

2026-10-14T12:47:21.898974+00:00 pairing debug四组全量验收完成：两方向采样器交换、置信度单因素及跨checkpoint控制均真实exit0；逐行/独立统计与最终来源校验通过。原因定位到解码置信度排序不一致；50000对齐后接近V7，112000仍存在残余差距，未归因为训练。历史成绩保留，新增对齐行及复现入口已同步；本地队列无剩余任务。[最终证据与边界](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#pairing-sampler-debug-20261014)。

2026-10-14T12:13:02.403309+00:00 pairing采样轨迹核查：前两条重链的16候选第一步提交预算相同，原始概率排序显著偏向各位置最高概率token；仅作为机制证据，完整IM结论仍待两项在跑对照。[诊断明细](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#pairing-sampler-debug-20261014)。

2026-10-14T12:08:10.361815+00:00 pairing诊断文档已整理：移除当前结论中失效的排队/评分中状态，原过程另存；两份榜单的已发布对齐行与完整metrics逐项核对一致。任务§7登记历史解码不可比口径，§8.6补齐复现与验收入口。剩余两项已按/proc确认实际运行；整体目标仍待其完整结果。[权威诊断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#pairing-sampler-debug-20261014)。

2026-10-14 pairing双向全量对照已确认fixed50000原IM差距几乎全部来自采样器：fixed同权重对齐V7后0.445515→0.591688，V7同权重换native后0.591934→0.458792。两组各4000候选真实exit0、独立验收通过；fixed对齐行已并列回填两份结果表。gumbel_prob单因素与fixed112000跨checkpoint对照正在本地接续运行，整体debug仍active。[完整证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#pairing-sampler-debug-20261014)。

2026-10-14 pairing首个完整反向对照验收：V7同权重换native概率排序，4000候选IM从0.591934降至0.458792，差−0.133142，重链配对bootstrap 95%区间[−0.153682,−0.112390]。进程真实exit0，逐行与独立统计核验通过；整体debug仍进行中。fixed50000/legacy已完整生成、正在评分；gumbel_prob已由本地队列接续启动，fixed112000/legacy待槽位。[完整诊断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#pairing-sampler-debug-20261014)。

2026-10-14 pairing原评分全量复核完成：fixed50000/112000各4000候选CPU重算与历史GPU均值差分别−7.34×10⁻⁷、4.88×10⁻⁶；进程真实exit0。重链替换检查两模型各16/16条轻链logits变化，未见条件通路整体断开。两项生成继续运行、另外两项待本地队列，最终质量结论仍待四组全量验收；[完整诊断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#pairing-sampler-debug-20261014)。

2026-10-14 pairing诊断协议已补齐：冻结源码四个采样核心函数一致；新增第三组控制用于区分置信度公式与并列分数提交规则。两份结果表的fixed reference历史行补标“原始概率排序”，所有指标单元格核对保持原值。四项全量对照继续本地队列执行，未发布未完成结果；[协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md#2026-10-14-同权重采样器诊断)。

2026-10-14T13:02:15.800552+00:00 探索与独立确认完成，选定LoRA；全量八卡主训练已从100步恢复并推进至240/10000，50→100原生恢复及完整checkpoint验收通过。所选pilot权重的36个下游任务已提交10作业，当前全部Queue。[部署验收](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/deployment_complete.json)。

2026-10-14 pairing四组受控诊断均迁入本地两并发队列（fixed50000/legacy、V7/native运行；fixed50000/gumbel_prob、fixed112000/legacy待前序结束），对应云任务均已Killed。队列PID3396646/会话22878，最多4小时；失败不重试，检测到其他GPU实验时暂缓新启动。原baseline CPU评分PID3388058继续。两组当前生成均精确复现历史GPU首候选，fixed迁移前后209条原始token一致；两版各500条输入全量审计通过。[队列状态](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/pairing_sampler_debug_20261014/local_queue_status.json)，[诊断过程](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#pairing-sampler-debug-20261014)。



2026-10-14 轻链IM诊断第二项已提交：fixed50000 / gumbel_prob，任务t-20261014185159-2422w，平台Queue。保留V7相同Gumbel提案，仅用原始token概率排序提交位置，以隔离置信度规则的影响。

2026-10-14 轻链配对IM受控debug进行中：针对fixed-chain与V7 f011差距，固定权重/500条输入/每条8候选/16步/reference条件，交换采样及置信度提交规则。原生fixed-chain CPU复现检查通过；独立目录[实验产物](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/pairing_sampler_debug_20261014)。

2026-10-14 11:06 UTC诊断进展：三组历史候选各4000条完整对齐/多样性复算通过；所有fixed候选均跑满16步。中断作业的26重链×8候选CPU同子集IM：原fixed50000=0.444309、仅换legacy=0.540737，V7历史同子集0.587133；采样器交换恢复本子集67.5%的差距，18/26重链改善。该子集非随机且不完整，尚未给出完整500条结论。两组完整baseline CPU评分仍运行，四个正式受控任务均平台Queue；[核查与局限](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#pairing-sampler-debug-20261014)。



2026-10-14 UpTCR低分debug首轮完成：已有预测复算、正式fixed-chain快照CPU输入与前向检查及39项现有测试通过；具体异常指标“IM”待用户明确。诊断和局限统一见[UpTCR输入输出核查](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#uptcr-debug-20261014)。

2026-10-14T10:38:29.740885+00:00 best112000下游评测状态核实：10个作业中8个Success、2个Running；36子任务30个success、2个running、4个pending，0失败。20项非AbBiBench任务全部完成；AbBiBench完成10/16，4fqi_h1与4fqi_h3运行中，其余4项等待同组前序任务。当前日志进度分别为25571/65094（39.3%）、24911/65535（38.0%）。[平台查询](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_best112000_20261014/platform_status.json)。

2026-10-14 2026-10-14T05:21:52.828565+00:00 核实：75.6M固定双链继续训练，114438/300000步（38.15%，约7.68 epoch）；最新保存114000，最佳112000 eval_loss=1.844108。已验证114次；最近训练loss=0.378。最近10000步含验证/保存均速1.100秒/步，估算剩余56.7小时。最新全mask/source-macro AA准确率38.67%、首EOS位置完全正确9.77%、长度MAE=4.67；属于单次前向指标。训练心跳正常，最新checkpoint所列文件与八rank保存回执齐全。[本次核实](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_300k_b512_supervision/training_review_20261014.json)。

2026-10-14 UpTCR binding已补齐论文全部7个比较模型，统一提取Fig.2官方Source Data的5次运行均值；纠正原正文单次数值与均值混用，Ours已有成绩未改。见[UpTCR baseline核查](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md#uptcr-baseline-audit-20261014)。

2026-10-14 T4 sparse13 baseline来源核查完成，已修正候选预算/F1标签及后续评分入口的严格对齐误标。未改原始预测、历史metrics或结果数值；差异及独立重算见[任务核查](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#sparse13-baseline-source-audit)。

2026-10-14 T4 sparse13差异核查完成：保存序列的距离/恢复率重算通过，新增长度统计和ANARCI整链抽样证据；已修正详细结果页单链有效性的解释。成绩与简洁版数值未变，无模型重跑。见[任务核查](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#t4-length-audit-20261014)。

2026-10-14 按用户要求，从详细版与简洁版结果汇总删除 S²ALM / BioMap 补充论文表：该表没有同协议 Ours 对照，不属于当前任务成绩比较。历史转录与审计继续保存在[过程文档](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#s2alm-audit)，已更新相关来源链接。

2026-10-14 按用户要求，从 RESULTS.md 和 RESULTS_SUMMARY.md 删除 CHIMERA 历史整链生成区块及四个 checkpoint 结果行；当前官方 H3 任务结果保留，原始实验产物未改动。

2026-10-14 270M / checkpoint32000 的 AbBiBench 16项已于01:50 UTC完成；修正详细版/简洁版此前滞留的11/16状态。已核验各项完成状态、metrics SHA256、范围、样本数及等权均值；[发布核验](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/270m32000_abbibench_publish_20261014/verification.json)，成绩见[RESULTS](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。本轮未重跑模型；下方2026-10-13状态表为历史快照。

2026-10-14 AbBiBench 论文/仓库榜单来源核查完成；差异及验证证据见[任务协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md#abbibench-paper-scope)。已补充详细结果页与出处登记，保持用户指定的16项范围及已有分数；简洁版继续只列结果。未重跑模型。

2026-10-14 T3 representation 当前协议核查完成：broad/deep仍为本地任务，论文数据、sampling和baseline recipe未对齐；汇总聚合及αβ身份分离检查通过，但不等于完整复现。未重跑或修改分数，修正旧入口说明。权威结论见 [T3 §7.4](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md#74-2026-10-14当前-fixed-chain50000-协议复核)。

2026-10-14 按用户要求移除 T2B KMeans 评测：两版结果表删除 MiniBatchKMeans 与完整KMeans区块；任务生成器/批量入口跳过T2B，直接运行和重建发布入口禁用。T2A保留。历史结果区块及移除记录见 [存档](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/t2b_retired_20261014)，原始特征/预测/评分不删除。范围以 [README](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/README.md#0-评测范围2026-08-29-收窄强制) 为准。旧冻结作业的源文件哈希不改写，不将旧计划当作新评测计划复用。

2026-10-14 fixed-chain75.6M checkpoint50000 Kong iter1补测完成；同权重、数据/掩码、batch64、bf16、argmax/temperature1/CFG0，唯一变化为4步改1步。3个CDR各3127条、共9381条预测独立重算通过，已回填详细与简洁结果表。证据：[独立核验](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_cdr_iter1_20261014/verification.json)、[原始指标](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_cdr_iter1_20261014/kong/metrics.json)。

2026-10-14 fixed-chain50000 Kong iter1本地补测启动；仅改变迭代次数，复用同checkpoint/数据/掩码/精度。[运行目录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_cdr_iter1_20261014)。

2026-10-13 T2B 作者发布包核查与旧入口协议说明修正完成；3个公开编码器全量特征一致、历史算法回归通过，结果数字未变。证据见[代码核查](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/t2b_paper_alignment_20261013/code_release_audit/AUDIT.md)。

2026-10-13 T2B本地评测4/4完成，结果与独立重算见[产物](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/t2b_paper_alignment_20261013/summary.json)；严格论文复现仍未通过。

2026-10-13 T2B 论文协议核查本地运行：3个作者编码器的CPU评测完成；进一步核实GPU仅剩占卡脚本，实际实验已退出，fixed-chain50000已从本轮CPU提取切换为本地CUDA bf16。CPU部分结果单独保存，无混用；未停止占卡进程，未提交云端作业。任务状态见 [独立运行目录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/t2b_paper_alignment_20261013)。

2026-10-13 T2B 实际实现与论文协议差异已登记，结果页名称已更正；见 [更正记录](/project_c20250601/251105016/project/dllm_test/docs/EXPERIMENT_LOG.md)。未修改结果数值或重跑评测。

2026-10-13 T2A 论文引用与比较口径说明已更正，见 [更正记录](/project_c20250601/251105016/project/dllm_test/docs/EXPERIMENT_LOG.md)。仅修改文档，未重跑评测或修改结果数值。

2026-10-13T13:14:33.796499+00:00 固定双链75.6M checkpoint50000：36项中30项success、2项running、4项pending，0失败；平台g0–g7 Success、g8/g9 Running。20项非AbBiBench任务全部完成；AbBiBench官方16项完成10项，未发布完整均值。已回填RESULTS.md及RESULTS_SUMMARY.md的对应任务榜单，保留V5/V6/V7和原native成绩。CDR 9741条预测独立重算通过，T1四种输入各五折齐全。证据：[fixed_chain_75m_step50000_20261013](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/published_results_audit.json)。

2026-10-13 新增任务结果简洁版[RESULTS_SUMMARY.md](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_SUMMARY.md)；详细版RESULTS.md保持原样，未新增评测。

2026-10-13 下游成绩已回填各任务原有对比表；结果页与评测过程分离。见[结果表](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)、[评测过程](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md)；未变更训练/评测作业。

2026-10-13 AB–Ag修正评测已提交六个单GPU作业，最近查询均Queue；两个旧4fqi作业继续Running，CPU核验/汇总已启动。三份可用模型快照通过CPU原生加载和H3接口检查；269000缺失权重，Binding/H3未启动。[CHIMERA现行协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。

2026-10-13 2026-10-13T04:32:53.707950+00:00核实：新75.6M固定双链版本继续训练，33281/300000步（约2.23 epoch）。已完成33次验证；最佳checkpoint-29000 eval_loss=1.8604366，最新完整checkpoint-33000 eval_loss=1.8642278；最近训练loss=0.3918。固定验证子集为七来源各256条，共1792条；主指标为全mask/source-macro。最佳AA准确率38.31%，EOS单点80.93%，首EOS位置完全正确10.41%，长度MAE4.39；后两项为全mask单次预测，不能当多轮生成成绩。新版本尚无已回填的下游benchmark成绩。最近10000步含验证/保存平均1.101秒/步，按此速度剩余约81.6小时（未计后续中断）。证据：[本次核实](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_300k_b512_supervision/training_review_20261013.json)。

2026-10-12 新版本已正式运行：75,601,176参数，V5训练集7,626,885条；残差0.1、双链独立RoPE、链内全注意力/全局交替，固定槽位187/154。300000步/global512=8×micro64×GA1，LR1e-4/warmup2000/cosine，随机从头初始化。2026-10-12T18:23:05.641727+00:00核实已到65步，最近记录训练loss=2.9047，梯度有限；13项CPU测试、8/8整模GPU检查、8/8 NCCL、85分片全哈希门禁通过。每1000步验证/保存，首次验证尚未到达。证据：[启动验收](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_300k_b512_supervision/startup_acceptance.json)。旧270M已按用户要求停止（最后约72279步，完整保存72000保留），原V7 f011此前已完成。

以下object302/270M说明属于此前已启动版本；新方案以以上配置为准。

Last updated (UTC): 2026-10-25T13:29:47.387664+00:00

## 当前决定

当前主实验为新V7 MHC-completed数据上的75.6M全有效token BERT式表征预训练，除prots/protd边界及无真实标签的位置外均参与MLM，从头更新全部参数，目标300000步。下表同时保留历史生成／LoRA实验摘要；当前运行状态以Active表及顶部启动证据为准。

| 工作 | 当前状态 | 下一步 |
|---|---|---|
| BERT V7 75.6M 全有效token表征预训练 | 八卡运行，187175/300000，15.333 epoch；最新完整187000，内部开发最佳122070 | [核查](/project_c20250601/251105016/project/dllm_test/output/bert_full_token_v7_300k_20261017/review_20261018_1909.json) |
| OAS-only 75.6M从头训练 | r1–r3 Failed；r4已提交，2026-10-24 07:21 UTC为Staging；最新保存11000，尚无新增训练步 | 等待r4实际恢复更新；[只读状态核验](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_oas_300k_b512_20261015_supervision/review_20261024/status.json) |
| 固定双链LoRA完整epoch | 成功完成33981步，最佳验证loss位于22000；该训练已退出，容器现用于BERT | 用本次权重完成下游验证，不能套用旧3000步成绩 |
| AB–Ag修正评测 / CHIMERA H3 | 六个新作业Queue、两个旧assay继续；269000新推理缺权重 | [执行状态](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_corrected_20261013/platform_status.json)；结果完成后按新协议汇总 |
| 新75.6M固定双链训练 | 已主动停止；完整恢复点131000保留 | 复用112000探索后训练 |
| 75.6M正式训练 | 300000/300000完成；最终完整保存核对通过，退出码0；八卡standby已恢复 | 原产物保留 |
| 270M新训练 | 容器Running；synthetic 8/8；正式72279/300000步；训练failed；训练子任务报错，检查原日志 | 核实已保存产物，并监督73000步验证/保存 |
| 新模型全下游评测 | best45000本地修复已终态；详见repair_final_20261009.json | [新状态](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/dedicated_status_check.json) |
| 历史V5生成数据 | V5全量7,729,193条通过；302容量覆盖；当时启动绑定通过 | [启动参数与回执](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v3/README.md) |
| V7 f011正式训练 | 已完成两轮，checkpoint-50082 | 保留历史结果，不重跑 |
| V7三类生成评测 | CDR、轻链、TCR已完成 | 数字见[downstream/benchmark/RESULTS.md](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md) |
| 原22格消融 | 已按用户要求停止：8项停于3600–3650步、4项未启动；原曲线保留 | 不自动续跑；[停止记录](/project_c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/user_pause_20261007.json) |

## Active Volc Training Tasks

本表记录当前作业最近的实际平台查询与训练观察；时间及原始证据见各行。

Last updated (UTC): 2026-10-26T23:52:54+00:00

| 任务 | ID | 最后记录 | 证据 |
|---|---|---|---|
| Diffusion BERT287000 / CDR四卡修复包 | `t-20261027004403-495cm` | 2026-10-26T23:52Z平台Running；正式7425/200000步，checkpoint7000；梯度稳定，内部CDR验证尚无明显收益 | [训练审查](/project_c20250601/251105016/project/dllm_test/output/diffusion_bert287000_cdr_4gpu_wandbfix_20261026/diagnostics/training_review_20261026/review.json) |
| OAS-only 75.6M续训 r6 | `t-20261025194633-xrh4w` | 2026-10-25T13:29:47.387664+00:00 Running；训练32560，完整checkpoint32000；全mask验证loss1.621/AA45.4%，约0.34秒/步 | [训练日志审查](/project_c20250601/251105016/project/dllm_test/output/oas_resume15000_r6_20261025/log_review_latest.json) |
| AB500K纯MLM | `t-20261024014536-ccpc2` | 容器最近仍Running；rank0首步日志TypeError，其他rank随后NCCL超时；只有checkpoint0 | [实际异常](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/code_review_20261023/mlm_rank0_failure.log) |
| Stage 1 在线结构蒸馏 | `t-20261023004236-rwrkk` | 2026-10-22T19:24Z平台Running；19:27Z第79批目标齐全，推断至少完成78步；约111秒/批，教师长输入导致跨卡等待；正式日志仍step3 | [速度诊断](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/online/log_review/timing_diagnosis_20261022.json) |


历史实验仍使用各自冻结的数据／源码，不随当前BERT改用V7而切换。完成状态以实际退出及产物为准；不将单次进度观察当终态。

### 2026-10-16 清理前任务终态核实

平台查询确认旧 LoRA 下游 g9（`t-20261014205213-5299k`）与 g8（`t-20261014205209-85z9v`）已于 2026-10-15 分别在 15:47:20Z、15:52:23Z 进入 Failed；移出此前滞留的 Active 行。[查询证据](/project_c20250601/251105016/project/dllm_test/output/feature_cache_cleanup_20261016/stale_task_status.json)。本次未调查失败原因或重启任务。

## 下一步

OAS-only 续训 r1–r3 均 Failed，r4已Failed，r5（`t-20261024152826-krr7n`）已Running并完成实际训练更新；恢复checkpoint-11000，目标300000步。

已完成LoRA完整epoch配置：[冻结续训包](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_lora_epoch1_20261015/prepared.json)。日志：[main](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_lora_epoch1_20261015/torchrun_stage2_runtime.log)，输出：[完整状态](/project_c20250601/251105016/project/dllm_test/output/immune_fixed_chain_lora_epoch1_20261015)。累计33981步成功退出，八rank success与COMPLETE一致；保留20000/21000/22000及33981完整状态，原10000步后训练和131000步预训练产物独立保留。

## 文档入口

- [BIOSEQ_MODEL_PLAN.md](/project_c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md)：当前模型方案。
- [DATA_FORMAT_AUDIT.md](/project_c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md)：V5数据及输入规则。
- [PROJ_GUIDE.md](/project_c20250601/251105016/project/dllm_test/PROJ_GUIDE.md)：日常操作约定。
- [实验与更正](/project_c20250601/251105016/project/dllm_test/docs/EXPERIMENT_LOG.md)：关键里程碑、证据、已纠正结论。
- [完整历史](/project_c20250601/251105016/project/dllm_test/docs/archive/project_docs_20261007T061603Z/PROJECT_PROCESS.md)：整理前逐次日志。

## 2026-10-07 数据选择与文档整理

按用户要求切换新版本数据选择到V5；归档重复过程记录并精简当前文档。保留历史实验数据、指标、checkpoint来源与原作业状态。数据绑定现已完成；实际训练参数现已配置；运行仍需显存检查与资源调度。

## 2026-10-07 按用户要求暂停消融并切换新训练

已向原八卡池写入STOP_NOW，请求终止在跑消融并阻止四项排队消融启动。指标、原始日志、排队票据及正式训练恢复checkpoint保留。此刻只确认停止请求，实际退出与平台终态另核实。新训练采用当前V5冻结包；未重复提交。证据：/project_c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/user_pause_20261007.json。

原池已于2026-10-07T07:08:26Z终止，八worker均exited、running目录清空；平台标为Failed，原因为用户主动中断，不是新发现模型故障。task_id=t-20261005205851-lqfjg，Preemptible=false，YAML=/project_c20250601/251105016/project/dllm_test/train_jobs/immune_research_pool_c20250601/eight_gpu_pool_20261005.yml。未调用cancel或重启；终态任务已移出Active表。

## 2026-10-07 新训练启动整合

YAML已写 /project_c20250601/251105016/project/dllm_test/train_jobs/v5_object302_scratch_8gpu_20261007.yml；使用 /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v2。权重从头随机初始化，源码固定副本不代表冻结参数。提交前补齐V5分隔符验证统计，否则首次eval会将`.`误归入binding校验而失败；训练目标及模型架构不变。保留best3＋latest完整状态、速度与显存记录。GPU实测待任务启动。

## 2026-10-07 提交V5新模型八卡训练

操作：submit；task_id=['t-20261007151807-cq4fb']；任务名=immune-v5-object302-scratch-8gpu-20261007；YAML=/project_c20250601/251105016/project/dllm_test/train_jobs/v5_object302_scratch_8gpu_20261007.yml；初始状态=submitted_pending_platform_query；Preemptible=false；命令returncode=0。global512=8×micro8×GA8，300000步，DDP；全部权重随机初始化。显存待所分配GPU上synthetic实测，不将提交成功当作已训练。证据：/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v2/submission.json。

## 2026-10-07 新训练首次启动失败

任务t-20261007151807-cq4fb于07:20:18Z Failed；rank4/6 SIGSEGV(-11)，发生在NCCL初始化后、synthetic报告前。正式训练0步，无新增checkpoint。平台终态已移出Active表；失败任务不重启、原始日志保留，不将CPU测试通过当八卡启动通过。证据：/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v2/failure_review.json。

## 2026-10-07 配置确认与原容器状态

用户指定training_config.json与实际launch_v2／候选launch_v3的全部训练超参数逐项一致，见 /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/training_config_agreement_20261007.json。状态字段已更正：曾提交、Failed、正式0步；超参数未改。用户希望复用原训练容器，但原池STOP_NOW使八worker及容器任务终止，无法直接进入已释放资源。用户随后明确持续容器方式，已新申请t-20261007153431-vhx2f；候选修正v3将通过容器内票据执行，不提交v3直接训练YAML，也不重启失败任务。

## 2026-10-07 按用户纠正恢复持续容器方式

按最新要求，改为先申请持续八卡容器，训练在容器内执行；训练退出/失败不退出容器，占卡进程必须在训练前全部停止并wait。资源YAML=/project_c20250601/251105016/project/dllm_test/train_jobs/immune_training_container_8gpu_20261007.yml；控制目录=/project_c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261007。当前仅完成准备、空训练队列；未提交候选v3直接训练YAML。

## 2026-10-07 提交持续八卡容器

操作submit；task_id=['t-20261007153431-vhx2f']；任务名=immune-v5-devpool-8gpu-20261007；YAML=/project_c20250601/251105016/project/dllm_test/train_jobs/immune_training_container_8gpu_20261007.yml；Preemptible=false；returncode=0；初始状态待平台查询。只提交资源容器，训练队列仍空；不重复提交已失败直接训练任务。17个CPU fixture子进程全部wait回收，已验证训练失败后容器继续、无自动重试、standby训练前停止。证据：/project_c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261007/submission.json。

## 2026-10-07 容器内发布正式训练票据

持续资源t-20261007153431-vhx2f Running，host=t-20261007153431-vhx2f-worker-0，supervisor PID737，八个standby PID770–777。已发布一次v5_scratch_launch_v3_20261007；训练初始化scratch，参数与用户指定training_config一致。此为容器内执行，不是第二次直接训练平台submit。占卡回收、GPU启动验收、正式更新步需看实际结果。证据：/project_c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261007/training_enqueue.json。

## 2026-10-07 持续容器内启动检查通过

八卡均完成3步synthetic前后向，micro8／长度1218，峰值allocated最大4.48 GiB；八个rank_nccl_ready全部生成，已越过首次失败位置。无synthetic checkpoint。容器state为training、standby_pids为空。随后已通过原生V5数据门禁并产生正式更新步；实时步数以step_timing.jsonl为准。证据：/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/attempt2/startup_review.json。首次SIGSEGV确切根因仍未单独归因。

## 2026-10-07 正式V5训练已开始

实际至121/300000步；单机八卡DDP，micro8×GA8/global512，完全随机初始化；数据train 7,626,885条，使用launch_v3代码。最近25个普通更新步中位耗时0.798秒。八卡显存及通信已通过，正式门禁回执随run_provenance保存。尚未到首次1000步完整验证/保存；不提前宣称长程、checkpoint或下游成绩通过。证据：/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/formal_start_review.json。

## 2026-10-07 首次1000步验证与保存通过

07:54:20Z只读验收通过：17个完整状态文件重新核对SHA256，八rank参数/保存确认一致；四个mask等级、七数据源的90项核心指标聚合一致。主eval_loss=2.1790857（mask100/source-macro）；当前best与latest均为checkpoint-1000，符合best3＋latest并集规则。训练随后继续超过1100步，没有重启或额外提交；已保存可恢复所需状态，本次未执行恢复重放。证据：[1000步核对报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint1000_root_review.json)。

首个完整1000更新窗口（已完成step1→1001）实际803.212秒，含一次验证/保存，平均0.803212秒/步；训练callback计时平均0.793750秒/步，其外合计9.462秒，不能进一步拆成单独验证或保存耗时。此为当前DDP/global512配置的实际吞吐，不是与旧版本的同配置对照。证据：[完整窗口计时](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/speed_first1000_updates_root_review.json)。

## 2026-10-07 2000步验证与保存通过

第二份checkpoint的17个状态文件、八rank参数/保存确认及90项核心聚合通过，参数相对1000步已更新；训练已继续超过2000步。主eval_loss（mask100/source-macro）2.179086→2.132296，mask100/token-micro AA准确率31.27%→33.54%；同口径关系标签重建准确率80.76%→69.34%，需继续观察，不能宣称全面改善或下游binding判别提升。保留完整checkpoint-1000与checkpoint-2000，best/latest当前为2000；未执行恢复重放。证据：[2000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint2000_root_review.json)。

关系标签回退主要来自tcr_native（96.48%→37.50%）；四个有关系标签来源的数量均保持各256，尚无原因定论。分源原件与数值见[实验记录](/project_c20250601/251105016/project/dllm_test/docs/EXPERIMENT_LOG.md)，3000步继续观察，训练配置不变。

## 2026-10-07 3000步验证与保存通过

第三份完整保存通过17文件哈希、八rank一致性、90项核心聚合和保留目录核对；主eval_loss=2.072132，mask100/token-micro AA准确率34.35%、关系标签准确率71.48%。tcr_native关系标签准确率36.72%，较2000步37.50%未恢复，不能以总体指标回升掩盖分源回退。最优/最新为3000，完整保留1000/2000/3000，无超额权重待清理；恢复重放未执行。证据：[3000步核对报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint3000_root_review.json)。继续同一配置，下一4000步。

## 2026-10-07 4000步验证与首次checkpoint轮换

4000步完整保存/验证检查通过。主eval_loss=2.108269，较3000步回升；mask100/token-micro AA准确率33.70%、关系标签准确率78.81%，其中tcr_native回升至56.25%。当前最优仍为3000、最新为4000，按best3＋latest保留2000/3000/4000三份完整状态。1000步权重已由训练原生保留逻辑删除；其六份JSON档案与删除前记录哈希一致，指标及来源证据保留。未重启训练或执行恢复重放。证据：[4000步核对报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint4000_root_review.json)、[清理后档案核对](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/retention_after4000_archive_review.json)。下一5000步，配置不变。

## 2026-10-07 5000步验证与保存通过

5000步17文件、八rank、90项核心聚合、参数更新及实际保留目录均通过。主eval_loss=2.091514，较4000步改善但最佳仍为3000；mask100/token-micro AA准确率34.27%、关系标签准确率75.78%，tcr_native关系标签44.92%，继续观察。现保留3000/4000/5000三份完整状态；2000步权重自动清理，其六份JSON档案哈希检查已随本次完整验收通过。未重启训练、恢复checkpoint或修改配置。证据：[5000步核对报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint5000_root_review.json)。下一6000步。

## 2026-10-07 6000步验证与保存通过

6000步完整状态与验证检查通过，主eval_loss=2.050985，为当前最低；mask100/token-micro AA准确率35.15%、关系标签准确率79.00%，tcr_native关系标签58.59%。分隔符准确率较5000步略退，不宣称全面改善。最优与最新均为6000，保留3000/5000/6000三份完整状态；4000步权重自动清理，六份JSON档案哈希检查通过。原容器与训练继续，配置不变，无恢复重放。证据：[6000步核对报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint6000_root_review.json)。下一7000步。

## 2026-10-07 7000步验证与保存通过

7000步完整状态与验证检查通过，主eval_loss=2.102113，较6000步回退；mask100/token-micro AA准确率34.17%、关系标签准确率71.88%，tcr_native关系标签35.55%。当前最优仍为6000；最佳三份3000/5000/6000与最新7000取并集，实际保留四份完整状态，本次没有旧权重被清理。该数量符合best3＋latest规则。配置保持不变，无训练重启或恢复重放。证据：[7000步核对报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint7000_root_review.json)。下一8000步。

## 2026-10-07 8000步验证与保存通过

8000步完整状态/验证检查通过，主eval_loss=2.070347，较7000改善，最佳仍6000；mask100/token-micro AA准确率34.52%、关系标签准确率77.25%，tcr_native关系标签52.34%。终止符与分隔符准确率较7000回退，继续如实记录。best3为6000/8000/3000，latest8000与best3重合，实际保留3000/6000/8000三份完整状态；5000/7000权重自动清理，两点共12份JSON档案哈希检查通过。配置及训练进程保持，未执行恢复重放。证据：[8000步核对报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint8000_root_review.json)。下一9000步。

## 2026-10-07 9000步验证与保存通过

9000步完整状态/验证检查通过，主eval_loss=2.033069，为当前最低；mask100/token-micro AA准确率36.05%、关系标签准确率74.12%，tcr_native关系标签39.45%。主loss、AA及边界指标较8000改善，关系标签指标回退，不宣称全面提升。最优与最新均为9000，保留6000/8000/9000三份完整状态；3000步权重自动清理，其六份JSON档案哈希检查通过。保持原配置，无额外训练启动或恢复重放。证据：[9000步核对报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint9000_root_review.json)。下一10000步。

## 2026-10-07 10000步验证与保存通过

10000步完整状态/验证检查通过，主eval_loss=2.028133，为当前最低；mask100/token-micro AA准确率36.53%、终止符75.61%、关系标签71.29%、分隔符78.03%。AA与分隔符提高，终止符及关系标签回退；tcr_native关系标签32.81%，不宣称全面改善或下游效果。最优与最新均为10000，实际保留6000/9000/10000三份完整状态；8000步权重自动清理，其六份JSON档案哈希核对通过。原容器内训练继续，未改配置或执行恢复重放。证据：[10000步核对报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint10000_root_review.json)。下一11000步。

## 2026-10-07 11000步验证与保存通过

11000步完整状态/验证检查通过，主eval_loss=2.025401，为当前最低；mask100/token-micro AA准确率35.61%、终止符75.95%、关系标签74.71%、分隔符75.42%。终止符与关系标签回升，AA和分隔符回退；tcr_native关系标签42.19%，仍有波动。best与latest均为11000，保留9000/10000/11000三份完整状态；6000步权重由原生策略清理，其六份JSON档案哈希核对通过。原容器内训练继续，配置保持，未执行恢复重放。证据：[11000步核对报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint11000_root_review.json)。下一12000步。

## 2026-10-07 12000步验证与保存通过

12000步完整状态/验证检查通过，主eval_loss=2.025920，略高于11000步；mask100/token-micro AA准确率34.97%、终止符84.49%、关系标签73.83%、分隔符76.41%。终止符与分隔符回升，AA及关系标签回退；tcr_native关系标签38.28%。最优仍为11000，最新12000；保留10000/11000/12000三份完整状态，9000步权重自动清理，其六份JSON档案哈希核对通过。继续原配置，未执行恢复重放。证据：[12000步核对报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint12000_root_review.json)。下一13000步。

## 2026-10-07 13000步验证回退与保存通过

13000步完整保存及指标核对通过，主eval_loss=2.122830，高于12000步2.025920；最优仍11000。终止符/关系标签准确率上升，AA和分隔符回退。按source-macro等权分解，tcr_native loss1.900705→2.532334贡献本次总增量约93.11%，各来源全遮蔽计数未变，原因尚未确定。best3为11000/12000/10000，加最新13000，实际保留四份完整状态，本次无清理。下一14000步继续观察，原配置与容器保持。证据：[13000步核对](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint13000_root_review.json)、[分来源增量](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint13000_source_delta_review.json)。

## 2026-10-07 14000步验证恢复与保存通过

14000步完整保存及指标核对通过，主eval_loss=2.018878，为新最佳。tcr_native loss2.532334→1.873917、AA准确率33.23%→43.07%，本点恢复；该来源关系标签43.36%→34.38%回退。整体全遮蔽AA36.93%、终止符82.76%、关系标签71.97%、分隔符78.09%，存在分项回退。实际保留11000/12000/14000三份完整状态，10000/13000权重由原生策略清理，两点共12份JSON档案哈希通过。继续原配置至15000步。证据：[14000步核对](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint14000_root_review.json)、[分来源后续核对](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint14000_source_followup_review.json)。

## 2026-10-07 15000步验证与首轮训练记录

15000步完整保存及指标核对通过，原生trainer_state记录epoch=1.006914。主eval_loss=2.044152，较14000步回升；全遮蔽AA34.92%、终止符85.83%、关系标签71.48%、分隔符75.48%，最佳仍14000。tcr_native loss1.873917→2.022110、AA43.07%→33.91%，仍有波动。best3=14000/11000/12000加最新15000，实际保留四份完整状态，本次无清理。首轮完成不代表300000步目标完成；保持配置，下一16000步。证据：[15000步核对报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint15000_root_review.json)。

## 2026-10-07 16000步验证与保存通过

16000步完整保存及指标核对通过，主eval_loss=2.052560，较15000步小幅回升，最佳仍14000。全遮蔽AA34.47%、终止符84.60%下降，关系标签76.46%提高，分隔符75.48%持平。保留11000/12000/14000/16000四份完整状态；15000旧权重自动清理，六份JSON档案哈希通过，首轮训练记录保留。继续原配置至17000步，未执行恢复重放。证据：[16000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint16000_root_review.json)。

## 2026-10-07 17000步验证与保存通过

17000步完整保存及指标核对通过，主eval_loss=2.046143，较16000步改善，最佳仍14000。全遮蔽AA34.79%、分隔符78.03%提高，终止符82.92%、关系标签75.29%回退。保留11000/12000/14000/17000四份完整状态；16000旧权重自动清理，六份JSON档案哈希通过。现有训练及配置保持，下一18000步。证据：[17000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint17000_root_review.json)。

## 2026-10-07 18000步验证与保存通过

18000步完整保存及指标核对通过，主eval_loss=2.073688，较17000步回升；最佳仍14000。全遮蔽AA34.81%略升，终止符81.75%、关系标签70.61%、分隔符75.36%回退；tcr_native loss2.043230→2.229023。保留11000/12000/14000/18000四份完整状态；17000旧权重自动清理，六份JSON档案哈希通过。保持原配置至19000步。证据：[18000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint18000_root_review.json)。

## 2026-10-07 19000步验证与保存通过

19000步完整保存及指标核对通过，主eval_loss=2.064763，较18000步改善；最佳仍14000。全遮蔽AA34.82%、终止符82.92%、关系标签75.10%、分隔符75.48%提高，AA均值75.94%略退；分源仍有回退。保留11000/12000/14000/19000四份完整状态，18000旧权重自动清理，六份JSON档案哈希通过。保持原配置至20000步。证据：[19000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint19000_root_review.json)。

## 2026-10-07 20000步验证与保存通过

20000步完整保存及指标核对通过，主eval_loss=2.040837，较19000步改善，最佳仍14000。全遮蔽AA34.90%、关系标签75.78%提高，终止符81.47%、分隔符75.42%回退；分源变化见实验记录。现保留11000/12000/14000/20000四份完整状态，19000旧权重自动清理，其六份JSON档案哈希一致。训练继续超过20000步，保持原配置，下一21000步。证据：[20000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint20000_root_review.json)。

## 2026-10-07 21000步验证与保存通过

21000步完整保存及指标核对通过，主eval_loss=2.027940，较20000步改善，最佳仍14000。全遮蔽AA35.05%、终止符84.54%、分隔符77.96%提高，关系标签75.68%略退。现保留11000/12000/14000/21000四份完整状态，20000旧权重自动清理，六份JSON档案哈希一致。原训练继续，配置保持，下一22000步。证据：[21000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint21000_root_review.json)。

## 2026-10-07 22000步验证与保存通过

22000步完整保存及指标核对通过，主eval_loss=2.052559，较21000步回升；最佳仍14000。全遮蔽AA34.92%、终止符80.41%、分隔符75.42%回退，关系标签79.49%提高。现保留11000/12000/14000/22000四份完整状态，21000旧权重自动清理、六份JSON档案哈希一致。原训练继续，配置保持，下一23000步。证据：[22000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint22000_root_review.json)。

## 2026-10-07 23000步验证与保存通过

23000步完整保存及指标核对通过，主eval_loss=2.025071，改善并进入最佳三份，最佳仍14000。全遮蔽AA35.61%、分隔符75.54%提高，终止符80.25%、关系标签72.85%回退。best3为14000/23000/11000，最新23000与之重合，实际保留11000/14000/23000三份完整状态；12000和22000旧权重自动清理，两点共12份JSON档案哈希一致。原训练继续，配置保持，下一24000步。证据：[23000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint23000_root_review.json)。

## 2026-10-07 24000步验证与保存通过

24000步完整保存及指标核对通过，主eval_loss=2.024323，较23000步小幅改善，当前第二、最佳仍14000。终止符81.47%、关系标签75.68%、分隔符78.27%提高，全遮蔽AA35.53%、AA均值76.20%略退。best3为14000/24000/23000，最新24000已在best3内，实际保留14000/23000/24000三份完整状态；11000旧权重自动清理，六份JSON档案哈希一致。原训练继续，配置保持，下一25000步。证据：[24000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint24000_root_review.json)。

## 2026-10-07 25000步验证与保存通过

25000步完整保存及指标核对通过，主eval_loss=2.089885，较24000步回升；按分源loss差值计算，tcr_native贡献主loss净增量约97.69%，仅为算术分解、原因未定。全遮蔽AA34.63%、关系标签74.51%、分隔符74.92%回退，终止符81.53%略升。最佳三份仍14000/24000/23000，加latest25000，实际保留14000/23000/24000/25000四份完整状态；本次无旧权重删除。原训练继续，配置保持，下一26000步。证据：[25000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint25000_root_review.json)。

## 2026-10-07 26000步验证与保存通过

26000步完整保存及指标核对通过，主eval_loss=2.061376，较25000步改善；tcr_native的loss/AA仅部分恢复，终止符恢复到99.22%，分隔符仍71.48%。全遮蔽AA34.87%、关系标签74.61%、分隔符75.48%提高，总体终止符79.07%回退。最佳仍14000；实际保留14000/23000/24000/26000四份完整状态，25000旧权重自动清理、六份JSON档案哈希一致。原训练继续，配置保持，下一27000步。证据：[26000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint26000_root_review.json)。

## 2026-10-07 27000步刷新主loss最佳值

27000步完整保存及指标核对通过，主eval_loss=2.011242，优于原14k的2.018878，为当前主loss最佳。全遮蔽AA35.64%、关系标签84.96%提高，分隔符75.48%持平；终止符降至70.93%，七来源均回退，不称为全面改善。best3=27000/14000/24000，latest已在其中，实际保留14000/24000/27000三份完整状态；23000与26000旧权重自动清理，共12份JSON档案哈希一致。原训练及配置保持，下一28000步。证据：[27000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint27000_root_review.json)。

## 2026-10-07 28000步验证与保存通过

28000步完整保存及指标核对通过，主eval_loss=2.012737，当前第二，最佳仍27000。七来源终止符均回升，总体79.74%；分隔符78.09%提高，全遮蔽AA35.59%、关系标签74.80%回退。tcr_native终止符仍低于26k水平，关系标签87.50%→38.28%下降，不能称为全面恢复。实际保留14000/27000/28000三份完整状态；24000旧权重自动清理，六份JSON档案哈希一致。原训练及配置保持，下一29000步。证据：[28000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint28000_root_review.json)。

## 2026-10-07 29000步验证与保存通过

29000步完整保存及指标核对通过，主eval_loss=2.019129，最佳仍27000。全遮蔽AA35.59%→36.69%、终止符79.74%→80.69%提高，关系标签74.80%→71.68%、分隔符78.09%→75.48%回退。tcr_native终止符恢复到99.22%，但oas降至44.53%；分源表现仍有波动。best3保持27000/28000/14000，最新29000不在best3，实际保留14000/27000/28000/29000四份完整状态，本次无旧状态待删。原训练及配置保持，下一30000步。证据：[29000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint29000_root_review.json)。

## 2026-10-07 30000步保存通过并越过两轮

30000步完整保存及指标核对通过，Trainer epoch=2.013829，完成计划步数的10%；仍按原300000步计划训练。主eval_loss=2.039798回升，最佳仍27000；全遮蔽AA34.90%、终止符75.00%、关系标签71.48%、分隔符74.18%均较29k下降。tcr_native AA41.41%→33.80%，asd_antibody分隔符26.67%→1.33%，需继续观察。best3保持27000/28000/14000，实际保留14000/27000/28000/30000四份完整状态；29000旧权重自动清理，六份JSON档案核对通过。原训练及配置保持，下一31000步。证据：[30000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint30000_root_review.json)。

## 2026-10-07 31000步验证与保存通过

31000步完整保存及指标核对通过，主eval_loss=2.042209略升，最佳仍27000。全遮蔽AA34.90%→35.03%、终止符75.00%→80.30%、分隔符74.18%→75.42%提高，关系标签71.48%→71.00%下降。asd_antibody分隔符1.33%→26.67%、oas终止符19.53%→53.91%恢复；tcr_native的AA及关系标签准确率继续下降。实际保留14000/27000/28000/31000四份完整状态；30000旧权重自动清理，六份JSON档案核对通过，首次越过两轮的证据保留。原训练及配置保持，下一32000步。证据：[31000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint31000_root_review.json)。

## 2026-10-07 32000步验证与保存通过

32000步完整保存及指标核对通过，主eval_loss=2.061845；27k后连续五个保存点上升，最佳仍27000。终止符80.30%→82.92%、关系标签71.00%→71.19%、分隔符75.42%→75.48%提高，全遮蔽AA35.03%→35.02%略降。tcr_native AA及关系准确率略升，整体loss继续回升；暂不作原因判断或改动配置。实际保留14000/27000/28000/32000四份完整状态；31000旧权重自动清理，六份JSON档案核对通过。原训练继续，下一33000步。证据：[32000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint32000_root_review.json)。

## 2026-10-07 33000步验证与保存通过

33000步完整保存及指标核对通过，主eval_loss=2.026425下降，结束27k后的连续五次上升，最佳仍27000。全遮蔽AA35.14%、终止符83.71%、分隔符78.09%提高；关系标签71.19%→66.89%下降，tcr_native32.81%→12.11%明显回退。各源目标计数保持一致，原因未定，继续跟踪。实际保留14000/27000/28000/33000四份完整状态；32000旧权重自动清理，六份JSON档案核对通过。原训练及配置保持，下一34000步。证据：[33000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint33000_root_review.json)。

## 2026-10-07 34000步验证与保存通过

34000步完整保存及指标核对通过，主eval_loss=2.066092回升，最佳仍27000。全遮蔽AA35.20%、分隔符80.01%提高，终止符79.69%、关系标签64.16%下降。tcr_native关系标签12.11%→3.91%继续回退；其25%/50%/75%遮蔽准确率为75.71%/74.29%/78.84%，各等级数量与此前一致，全遮蔽回退更剧烈。原因未定，继续跟踪。实际保留14000/27000/28000/34000四份完整状态；33000旧权重自动清理，六份JSON档案通过。原配置保持，下一35000步。证据：[34000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint34000_root_review.json)。

## 2026-10-07 35000步验证与保存通过

35000步完整保存及指标核对通过，主eval_loss=2.032845下降，最佳仍27000。关系标签64.16%→75.88%提高；tcr_native全遮蔽关系3.91%→43.36%、其他三遮蔽等级也回升，但四等级仍低于27k。终止符79.69%→75.17%、分隔符80.01%→78.09%、全遮蔽AA35.20%→35.18%下降。实际保留14000/27000/28000/35000四份完整状态；34000旧权重自动清理、六份JSON档案通过。只读观察20150实际exit0后接续66245，训练未重启。下一36000步，原配置保持。证据：[35000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint35000_root_review.json)。

## 2026-10-07 36000步验证与保存通过

36000步完整保存及指标核对通过，主eval_loss=2.022820继续下降，最佳仍27000。终止符75.17%→80.64%提高；关系标签75.88%→73.63%、全遮蔽AA35.18%→34.98%、分隔符78.09%→75.42%下降。native四遮蔽关系准确率均较35k下降，全遮蔽43.36%→39.45%，各等级计数不变。实际保留14000/27000/28000/36000四份完整状态；35000旧权重自动清理、六份JSON档案通过。原训练与只读观察66245继续，下一37000步，配置保持。证据：[36000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint36000_root_review.json)。

## 2026-10-07 37000步验证与保存通过

37000步完整保存及指标核对通过，主eval_loss=2.056276回升，最佳仍27000。全遮蔽AA34.98%→34.39%、终止符80.64%→76.23%、关系标签73.63%→73.34%下降，分隔符75.42%→75.54%略升。native四遮蔽关系准确率均回升，全遮蔽39.45%→40.23%，各等级计数保持不变。实际保留14000/27000/28000/37000四份完整状态；36000旧权重自动清理、六份JSON档案通过。原训练与观察66245继续，下一38000步，配置保持。证据：[37000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint37000_root_review.json)。

## 2026-10-07 38000步验证与保存通过

38000步完整保存及指标核对通过，主eval_loss=2.057750略升，最佳仍27000。全遮蔽AA34.39%→35.16%、终止符76.23%→83.09%提高；关系标签73.34%→71.88%、分隔符75.54%→75.42%下降。native四遮蔽关系准确率均下降，全遮蔽40.23%→32.03%，各等级计数不变。实际保留14000/27000/28000/38000四份完整状态；37000旧权重自动清理、六份JSON档案通过。当前仍在已申请的八卡容器内训练，standby_pids为空；未重新申请容器或提交训练。下一39000步，配置保持。证据：[38000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint38000_root_review.json)。

## 2026-10-07 39000步验证与保存通过

39000步完整保存及指标核对通过，主eval_loss=2.040339下降，最佳仍27000。全遮蔽关系标签71.88%→74.12%提高；AA35.16%→35.05%、终止符83.09%→79.24%、分隔符75.42%→74.49%下降。native关系准确率在mask25持平，其他三等级提高，全遮蔽32.03%→39.06%；各等级计数不变。实际保留14000/27000/28000/39000四份完整状态，38000旧权重自动清理、六份JSON档案核对通过。原容器及训练继续，下一40000步，配置保持。证据：[39000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint39000_root_review.json)。

## 2026-10-07 40000步验证成为当前主loss最佳

40000步完整保存及指标核对通过，主eval_loss=2.000729，优于原27000步的2.011242。全遮蔽AA35.05%→36.15%、分隔符74.49%→75.36%提高，终止符79.24%→76.90%、关系标签74.12%→71.00%下降；native终止符99.22%→82.81%、关系39.06%→31.25%回退。best3=40000/27000/28000，latest与best重合，实际保留27000/28000/40000三份完整状态；14000及39000旧权重自动清理，共12份JSON档案核对通过。只读窗口66245自然结束后接续5765，原容器与训练未重启。下一41000步，配置保持。证据：[40000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint40000_root_review.json)。

## 2026-10-07 41000步验证与保存通过

41000步完整保存及指标核对通过，主eval_loss=2.011555，较40k回升，按主loss排名第三。全遮蔽关系标签71.00%→81.74%、终止符76.90%→80.58%提高，AA36.15%→35.73%、分隔符75.36%→74.49%下降；native关系31.25%→73.83%、终止符82.81%→99.22%回升。best3=40000/27000/41000，latest已在其中，实际保留27000/40000/41000三份完整状态；28000旧权重自动清理、六份JSON档案核对通过。原容器、训练及只读观察5765继续，下一42000步，配置保持。证据：[41000步报告](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint41000_root_review.json)。

## 2026-10-07 23:33 UTC 当前状态与观察间隔补核对

只读观察5765在17:29:45自然结束；23:33:32重新查询已到71395步、平台和训练均Running。原容器、训练PID及配置保持，占卡进程为空；未重新提交。42k–70k的29点保存元数据、2610项聚合与保留记录已补核对，旧已删除二进制不宣称复核；当前42k/45k/68k及最新71k完整文件哈希通过。最佳45k主loss1.992721，最新71k为2.011193；保留42k/45k/68k/71k四份完整状态。新只读观察26248，下一72000步。证据和指标见 [实验记录](/project_c20250601/251105016/project/dllm_test/docs/EXPERIMENT_LOG.md)。

## 2026-10-07 72000步验收与旧no-ESMC曲线核对

72k完整保存通过17文件哈希、八rank和90项核心聚合核对；主eval_loss=1.999634，当前最佳仍45k。按best3＋latest保留42k/45k/68k/72k；71k权重由原生逻辑清理，六份JSON证据已核对。报告：[checkpoint72000_root_review.json](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint72000_root_review.json)。

已从原始Trainer记录整理两版训练和验证曲线。旧no-ESMC实际为V6 binding-only，当前为V5；验证mask、EOS目标和日志尺度不同，不能以原始loss排名。当前10k→72k主验证下降约1.41%，旧版自身验证同期约9.71%，仅为各自协议下的趋势。图表与限定见 [V5_CURRENT_VS_LEGACY_NOESMC_LOSS_20261007.md](/project_c20250601/251105016/project/dllm_test/docs/V5_CURRENT_VS_LEGACY_NOESMC_LOSS_20261007.md)。未改变正在运行的配置、未新增GPU实验。

## 2026-10-08 新模型下游评测提交

固定best45000，37任务/4组。首次queue012提交被平台拒绝：CreateCustomTask无权限，未产生作业；改用当前已授权c20250601队列的独立单卡评测资源。原八卡训练不变。

- submit：`t-20261008075358-bkn9q`，v5-native-best45000-all-g0-20261008；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g0_20261008.yml；初始Submitted，c20250601，Preemptible=true。

- submit：`t-20261008075414-ldm2d`，v5-native-best45000-all-g1-20261008；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g1_20261008.yml；初始Submitted，c20250601，Preemptible=true。

- submit：`t-20261008075418-pbx7q`，v5-native-best45000-all-g2-20261008；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g2_20261008.yml；初始Submitted，c20250601，Preemptible=true。

- submit：`t-20261008075421-nsd4d`，v5-native-best45000-all-g3-20261008；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g3_20261008.yml；初始Submitted，c20250601，Preemptible=true。

- 2026-10-08T17:40:22.173688+00:00 cancel：t-20261008075358-bkn9q（闲时g0，原YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g0_20261008.yml），按用户授权切换非闲时；请求成功，等待终态，不重复提交。

- 2026-10-08T17:40:22.659598+00:00 cancel：t-20261008075414-ldm2d（闲时g1，原YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g1_20261008.yml），按用户授权切换非闲时；请求成功，等待终态，不重复提交。

- 2026-10-08T17:40:23.443411+00:00 cancel：t-20261008075418-pbx7q（闲时g2，原YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g2_20261008.yml），按用户授权切换非闲时；请求成功，等待终态，不重复提交。

- 2026-10-08T17:40:23.817528+00:00 cancel：t-20261008075421-nsd4d（闲时g3，原YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g3_20261008.yml），按用户授权切换非闲时；请求成功，等待终态，不重复提交。

## 2026-10-08 授权切换非闲时下游评测

4个旧闲时作业均已终止，未执行评测；终态证据 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/spot_terminal_states.json。使用原best45000、相同37项任务和划分，改为4个独立单卡非闲时作业，不修改训练。

- submit：t-20261009014055-jwbqq，v5-native-best45000-all-g0-dedicated-20261008；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g0_20261008_dedicated.yml；初始Submitted，c20250601，Preemptible=false。

- submit：t-20261009014058-nwkcf，v5-native-best45000-all-g1-dedicated-20261008；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g1_20261008_dedicated.yml；初始Submitted，c20250601，Preemptible=false。

- submit：t-20261009014102-gd7sw，v5-native-best45000-all-g2-dedicated-20261008；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g2_20261008_dedicated.yml；初始Submitted，c20250601，Preemptible=false。

- submit：t-20261009014105-lgdcg，v5-native-best45000-all-g3-dedicated-20261008；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g3_20261008_dedicated.yml；初始Submitted，c20250601，Preemptible=false。

## 2026-10-08 17:40 UTC 训练过半与153000步核对

实际查询至153097/300000步（51.03%），容器及训练均Running；无可见报错，占卡进程为空。最新153k验证loss=2.008281，最佳138k=1.991442，比此前45k最佳仅改善约0.064%。验证仍在平台区间，不能宣称显著效果提升。最新完整保存核对通过；进度、实际耗时和验收范围见 [本次状态证据](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/status_20261008T1740_root_review.json)。未改变训练配置、提交或重启任务。

## 2026-10-08 训练结束后恢复八卡占卡

按用户要求核实已有自动切换逻辑；无需重启容器、改训练或新增GPU负载。当前训练继续，standby为空。已启动本地只读监督PID1100460、会话90153，上限50小时；跟踪训练状态，并在训练退出后确认八卡占卡新进程和初始化日志，记录实际结果。监督准备及CPU边界检查通过，证据 [占卡恢复监督](/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/training_standby_watch_20261008/setup_review.json)。此刻训练未完成，不提前称占卡已恢复。

## 2026-10-08 Kong失败修复并重提

用户授权重提失败项；只有Kong失败。独立重试入口将空轻链表示为缺失链，不删样本、不补造序列；CPU真实样本前向、隐藏答案隔离、双链路径一致性通过。原4组源码哈希不变。submit t-20261009015628-wqmrg；任务名v5-native-best45000-kong-retry1-20261008；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_kong_retry1_20261008.yml；初始Submitted，Preemptible=false，c20250601。旧失败产物归档于 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/attempt_history/cdr-kong_attempt1。

## 2026-10-08 全部CDR协议对齐

按用户要求，新增Kong三CDR/十折与SAb23H2六CDR官方design掩码评测，复用AirGen原解码逻辑；原iter1成绩保留历史。全量输入9381+360条核查通过；独立best45000评测快照，其他37项原作业代码哈希保持。规范见 /project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md。

- submit：t-20261009020953-x4xxj；v5-airgen-aligned-cdr-kong-45000-20261008；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_airgen_aligned_cdr_kong_20261008.yml；初始Submitted，c20250601，Preemptible=false。

- submit：t-20261009020956-sj4dd；v5-airgen-aligned-cdr-sab23-45000-20261008；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_airgen_aligned_cdr_sab23_20261008.yml；初始Submitted，c20250601，Preemptible=false。

## 2026-10-09 下游失败修复

原四组评测平台状态为Failed/Failed/Success/Failed，两个AirGen对齐CDR作业均Success，均移出Active表。T1修复基线表路径，只恢复已有五折结果的比较表；CHIMERA隔离ANARCI单序列异常，保留候选及失败原因。本地运行评分恢复与TCR beta-only禁止dot的新协议复测；原始失败输出归档，302容量、learned protd、checkpoint及种子不变。22项回归测试通过。结果以 /project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md 为准。

本地收尾器已启动（scripts/downstream/finalize_v5_repairs.py），等待两个修复进程实际退出后再同步终态/RESULTS并删除CHIMERA novelty_tmp目录；未提交新云端资源，未修改训练进程。TCR当前已完成目标的部分进度见repair_audit_20261009.json，不能当作全量成绩。

## 2026-10-09 最新269000步评测准备

按用户要求固定最新完整checkpoint-269000（选择时刻2026-10-09T19:24:02Z），8 rank及六文件SHA通过，独立硬链接仅保留推理权重/元数据。验证最佳仍为138000步。37项任务沿用修复协议，CDR为AirGen对齐，输出 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009；本轮与45000分开记录，完整后释放评测权重链接，不保留探针ckpt。

2026-10-09 submit：latest269000 group0，task_id=['t-20261010032615-hpdkc']，returncode=0，Preemptible=false；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_latest269000_g0_20261009.yml。

2026-10-09 submit：latest269000 group1，task_id=['t-20261010032618-5nptj']，returncode=0，Preemptible=false；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_latest269000_g1_20261009.yml。

2026-10-09 submit：latest269000 group2，task_id=['t-20261010032622-nx9nd']，returncode=0，Preemptible=false；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_latest269000_g2_20261009.yml。

2026-10-09 submit：latest269000 group3，task_id=['t-20261010032626-sdw56']，returncode=0，Preemptible=false；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_latest269000_g3_20261009.yml。


### 2026-10-09 CDR iteration 对照

固定269000步开展CDR infilling iteration敏感性实验：1/2/4/8/16/32步，Kong三CDR十折和SAb六CDR全量，共12个数据集×步数设置、54个CDR对照格。独立权重硬链接防训练轮转，默认主评测继续；结果与逐样本预测保存于 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_iterations269000_20261009。

2026-10-09 submit：CDR iterations kong，id=['t-20261010033011-gjsw2']，returncode=0，非闲时1GPU，YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_cdr_iterations269000_kong_20261009.yml；六种步数CPU检查全部通过。

2026-10-09 submit：CDR iterations sab23，id=['t-20261010033015-85bgb']，returncode=0，非闲时1GPU，YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_cdr_iterations269000_sab23_20261009.yml；六种步数CPU检查全部通过。


### 2026-10-09 验证最佳138000评测

固定验证最佳checkpoint-138000，eval_loss=1.991442，8 rank与推理权重/元数据SHA核对通过。对齐269000运行37项全下游及CDR 1/2/4/8/16/32步敏感性分析。独立源码副本保持已有作业代码哈希不变；独立硬链接评测快照仅包含推理权重和元数据。目录 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009 与 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_iterations138000_20261009。

2026-10-09 submit：v5_native_best138000_g0_20261009，id=['t-20261010033209-q99gz']，returncode=0，Preemptible=false；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best138000_g0_20261009.yml。

2026-10-09 submit：v5_native_best138000_g1_20261009，id=['t-20261010033213-6fkx8']，returncode=0，Preemptible=false；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best138000_g1_20261009.yml。

2026-10-09 submit：v5_native_best138000_g2_20261009，id=['t-20261010033216-8j2s8']，returncode=0，Preemptible=false；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best138000_g2_20261009.yml。

2026-10-09 submit：v5_native_best138000_g3_20261009，id=['t-20261010033220-q4fgp']，returncode=0，Preemptible=false；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best138000_g3_20261009.yml。

2026-10-09 submit：v5_cdr_iterations138000_kong_20261009，id=['t-20261010033224-djh66']，returncode=0，Preemptible=false；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_cdr_iterations138000_kong_20261009.yml。

2026-10-09 submit：v5_cdr_iterations138000_sab23_20261009，id=['t-20261010033228-m7rng']，returncode=0，Preemptible=false；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_cdr_iterations138000_sab23_20261009.yml。


2026-10-09 修复复测已终态：{"t1-cdr3b": "success", "ag-chimera": "success", "t4-benchmark14": "success", "t4-held20": "success"}。结果及CDR3语法有效率见 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/repair_final_20261009.json；原始候选保留，MMseqs临时目录已清理。82项回归及60组T1 AUROC复核通过；未见表位泛化弱仍未由流程修复解决。


## 2026-10-10T02:07:51.760263+00:00 训练至占卡只读监督结束

监督结果=training_exited_standby_observed；训练票据状态=succeeded；八卡占卡恢复观察=True。证据：/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/training_standby_watch_20261008/exit.json。未变更GPU进程或票据；完整训练产物最终验收另行核对。

### 2026-10-10 评测进度核对

45000修复后37/37成功；269000全下游31/37成功、2项运行、4项等待；138000全下游7/37成功、1项运行、29项等待，另三组平台Queue。269000 iteration 12/12设置完成，138000 iteration两个作业仍Queue。已完成的四个评测作业移出Active表。训练300000步已成功结束，验证最优仍138000，当前新评测固定269000/138000，不自动切换300000。证据 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/progress_20261010.json。

## 2026-10-10 75.6M完成及270M准备

旧训练真实成功退出、COMPLETE为300000；最终17文件/八rank/90项指标核对通过，最终主loss2.003477，best138k=1.991442，保留138k/45k/190k及latest300k。容器仍运行且八个占卡已恢复。最终保存报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint300000_root_review.json。

用户要求新训270M，已完成参数核实与独立配置；复用当前容器，不重复提交资源。配置 /project_c20250601/251105016/project/dllm_test/configs/models/v5_noesmc_270m_20261010/training_config.json；候选整模profile包 /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_270m_profile_bs64_20261010。当前尚未发布新票据，不将准备完成称为开训。

## 2026-10-10 发布270M整模synthetic检查票据

向现有容器t-20261007153431-vhx2f发布一次v5_270m_profile_bs64_20261010.json；仅整模synthetic micro64×3步/每GPU及通信检查，不访问语料训练。五项CPU门禁测试与启动元数据核对通过。容器先回收八个standby再执行；本次无平台submit、无容器重启。正式训练尚未启动。证据 /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_270m_profile_bs64_20261010/ticket_published.json。

## 2026-10-10 发布270M正式训练票据

v5_270m_scratch_bs64_20261010.json已发布到原持续容器，未平台submit/重启。八卡synthetic及NCCL通过，allocated峰值42.78797GiB、reserved48.87891GiB；8卡参数均268,590,112，3步有限loss，profile实际退出码0，无测试checkpoint。正式训练复用这份匹配源码/尺寸/batch的检查证据，不重复synthetic；原生数据全哈希门禁保留。每卡64、GA1、global512为同预算可用最大micro。参数和参考YAML见 /project_c20250601/251105016/project/dllm_test/configs/models/v5_noesmc_270m_20261010/README.md；发布证据 /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_270m_train_bs64_20261010/ticket_published.json。

## 2026-10-10 270M正式更新步已确认

2026-10-10T08:14:43.885827+00:00实读67/300000步；训练loss=2.8957有限，8rank通信及85分片全内容哈希通过，真实训练峰值allocated=40.79GiB。监督会话26867已实际输出并持续运行；训练结束后的占卡恢复仍由原容器负责。启动证据 [270M正式训练](/project_c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/formal_start_review.json)。本阶段尚无1000步验证/保存，不宣称质量收益。

## 2026-10-11 270M训练进度与28000步保存核对

2026-10-11本次核实已到28082/300000步（约9.36%）；最新完整保存28000步/epoch1.8797，累计28次原生验证。主eval_loss（全遮蔽/source-macro）最佳17000步1.993521，最新28000步2.092312，近期波动，尚无持续下降结论。最近10000更新墙钟平均2.8864秒/步，包含窗口内验证/保存；不中断约余9.1天。 训练和容器均running，占卡为空，原只读监督仍运行。保留best3（17000/16000/14000）＋latest28000共4份完整状态，其余旧权重按规则轮换。28000步完整核对通过，未执行恢复重放；本次无提交、重启或训练配置变动。证据：[状态与计时](/project_c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/status_20261011_checkpoint28000_root_review.json)、[28000步核对](/project_c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/checkpoint28000_root_review.json)。

## 2026-10-12 270M训练进度与60000步保存核对

2026-10-12本次核实已到60305/300000步（约20.10%）；最新完整保存60000步/epoch4.0279，累计60次原生验证。主eval_loss（全遮蔽/source-macro）最佳32000步1.993467，最新60000步2.007204；相对17000步旧最佳仅改善0.0027%，整体仍波动。最近10000更新墙钟平均2.8940秒/步，包含窗口内验证/保存；不中断约余8.0天。 训练和容器均running，standby为空，原只读监督PID1737482/exec26867仍运行。保留best3（32000/17000/43000）＋latest60000共4份完整状态，其余旧权重按规则轮换；正式恢复状态保留。本次无提交、重启或训练配置变动，未执行恢复重放。证据：[状态与计时](/project_c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/status_20261012_checkpoint60000_root_review.json)、[60000步核对](/project_c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/checkpoint60000_root_review.json)。

### 2026-10-12 下游评测全部终态

三组各37/37成功；两组iteration各12/12设置完成。核实12个新云端作业均Success并移出Active表；本轮没有待跑/失败的评测任务。所有质量结论与对照数字统一见 /project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md；不能将任务执行成功当作模型质量达标。


2026-10-12：澄清 CDR 主表版本：f011 50082 的 74.5438 / 68.6565 / 42.4700 为历史原生 A4、iter=1 成绩，非当前 V5 数据 native object302 权重；“两轮”指训练轮数。保留历史数值，在 Kong 主表补入当前 75.6M 三个 checkpoint 的 AirGen 对齐 iter=4 原始 metrics，并明确跨版本不构成受控消融。仅修正文档展示，未重新评测。


2026-10-12：用户确认 CDR 评测以 Ophiuchus-Ab 官方权重复现论文为验收目标，不以代码形式对齐代替结果验证。历史复跑与当前入口验收分开；当前端到端官方权重对照尚待完成。协议见 /project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md 。本次仅明确验收规则，未新增评测或修改成绩。


## 2026-10-12T09:27:37.255160+00:00 270M 下游提交 g0

操作 submit；task_id=['t-20261012172738-sfpmb']；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_270m_best32000_g0_20261012.yml；returncode=0；初始状态 submitted_pending_platform_query；Preemptible=false。固定验证最佳32000步，37项分四组，独立评测快照为硬链接；训练不受影响。


## 2026-10-12T09:27:40.815762+00:00 270M 下游提交 g1

操作 submit；task_id=['t-20261012172742-pbvpt']；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_270m_best32000_g1_20261012.yml；returncode=0；初始状态 submitted_pending_platform_query；Preemptible=false。固定验证最佳32000步，37项分四组，独立评测快照为硬链接；训练不受影响。


## 2026-10-12T09:27:44.305878+00:00 270M 下游提交 g2

操作 submit；task_id=['t-20261012172745-kxt5s']；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_270m_best32000_g2_20261012.yml；returncode=0；初始状态 submitted_pending_platform_query；Preemptible=false。固定验证最佳32000步，37项分四组，独立评测快照为硬链接；训练不受影响。


## 2026-10-12T09:27:48.146769+00:00 270M 下游提交 g3

操作 submit；task_id=['t-20261012172749-tx78f']；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_270m_best32000_g3_20261012.yml；returncode=0；初始状态 submitted_pending_platform_query；Preemptible=false。固定验证最佳32000步，37项分四组，独立评测快照为硬链接；训练不受影响。


2026-10-12：270M 首次下游固定验证最佳32000步（epoch 2.14823，eval_loss 1.993467），四组37任务已提交，CPU真实权重CDR门禁通过；独立结果见 /project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md 。选点与平台状态见 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012 。


2026-10-12：按用户要求先提交当前代码快照e8ad5fb（未push），再准备相同270M checkpoint32000的旧CDR入口对照。旧run_v5_cdr_eval及六个依赖SHA与45000历史manifest一致；原V7评测共用argmax和评分器，输入适配不同。协议差异见 /project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md 。不更改运行中AirGen源码与成绩。


## 2026-10-12T09:40:19.756460+00:00 270M 下游提交 g0

操作 submit；task_id=['t-20261012174021-jrcbb']；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_270m_legacycdr32000_g0_20261012.yml；returncode=0；初始状态 submitted_pending_platform_query；Preemptible=false。固定验证最佳32000步，9类CDR分两组，独立评测快照为硬链接；训练不受影响。


## 2026-10-12T09:40:23.536049+00:00 270M 下游提交 g1

操作 submit；task_id=['t-20261012174026-thrps']；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_native_270m_legacycdr32000_g1_20261012.yml；returncode=0；初始状态 submitted_pending_platform_query；Preemptible=false。固定验证最佳32000步，9类CDR分两组，独立评测快照为硬链接；训练不受影响。


2026-10-12：按用户要求在本地A100启动Ophiuchus-Ab官方权重采样验证，不提交云任务。固定CFG0、temperature1、seed42、batch64；两数据集全量比较argmax iter1/2/4与vanilla/gumbel_argmax iter1，另查SAb argmax iter2 bf16精度差异。执行AirGen原始decoder/collator方法，backbone使用现有checkpoint兼容移植实现并严格加载权重；不冒充未经验证的全仓官方环境。原始预测/指标见 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_sampling_audit_20261012 。


2026-10-12：Ophiuchus-Ab本地采样复现核查完成，11组数据集设置/51个CDR指标/49,065条区域预测独立重算通过。官方权重严格加载，11份对应主干源码与AirGen一致；本次范围内Kong argmax1/bf16、SAb argmax2/fp32最接近论文，仍非逐项精确复现。数字见RESULTS.md，方案和证据见 /project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md 。本地运行退出码0，无新checkpoint、无云提交。


2026-10-12：用户要求单步测试。75.6M 138000/269000已有完整iter1产物；本地新增270M固定32000的Kong/SAb iter1评测，仅改迭代数，坐标/X掩码、精度及batch64保持。无云提交。产物 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_single_step270m_20261012 。此前旧子串入口Kong因空轻链输入失败，不能与此次新掩码单步测试混称。


2026-10-12：270M固定32000新坐标/X掩码的单步CDR本地评测完成；9741条区域预测独立重算通过，75.6M两组既有单步也重算通过。结果及原多步差值已回填RESULTS.md。模型硬链接自动释放，原训练权重未动。证据 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_single_step270m_20261012 。


2026-10-12历史草案已被75.6M固定双链实现替代；不再待用户选择链边界。


## 2026-10-12 切换至75.6M固定双链新版本

用户要求停止旧训练并启动新版本。旧V7 f011已结束；实际运行的270M票据v5_270m_scratch_bs64_20261010已通过PAUSE_TRAINING请求停止，保留容器t-20261007153431-vhx2f及历史checkpoint。待确认退出后启动新固定双链版本；本次不停止下游评测。请求证据：/project_c20250601/251105016/project/dllm_test/output/v5_fixed_chain_f011_20261012/old_training_stop_request.json。


## 2026-10-12T18:17:24.162084+00:00 训练至占卡只读监督结束

监督结果=training_exited_standby_observed；训练票据状态=failed；八卡占卡恢复观察=True。证据：/project_c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/training_standby_watch_20261010/exit.json。未变更GPU进程或票据；完整训练产物最终验收另行核对。

2026-10-12T18:20:20.324869+00:00：已发布唯一新训练票据 v5_fixed_chain_75m_scratch_20261012.json，复用现有八卡容器，未新增平台资源。13项CPU测试及启动配置检查通过；启动包独立冻结，GPU显存与原生全量哈希检查进行中，尚不计为正式训练步。旧270M interrupted=true，退出码1为主动停止。


## 2026-10-12 新固定双链75.6M正式训练已启动

2026-10-12 新版本已正式运行：75,601,176参数，V5训练集7,626,885条；残差0.1、双链独立RoPE、链内全注意力/全局交替，固定槽位187/154。300000步/global512=8×micro64×GA1，LR1e-4/warmup2000/cosine，随机从头初始化。2026-10-12T18:23:05.641727+00:00核实已到65步，最近记录训练loss=2.9047，梯度有限；13项CPU测试、8/8整模GPU检查、8/8 NCCL、85分片全哈希门禁通过。每1000步验证/保存，首次验证尚未到达。证据：[启动验收](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_300k_b512_supervision/startup_acceptance.json)。旧270M已按用户要求停止（最后约72279步，完整保存72000保留），原V7 f011此前已完成。

本次通过现有容器票据执行，无新增volc submit/cancel。此前已终态的270M legacy g1 Success、legacy g0 Failed、主评测g1 Success已移出Active表，其余三组评测继续运行；本次未取消评测。

## 2026-10-13 AbBiBench默认任务范围修复

默认任务清单、评测入口和聚合器修复完成；CPU验收及既有产物复核见[过程记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/PROGRESS.md)。无新增GPU提交、取消或模型推理。

## 2026-10-13 提交AB–Ag修正评测 75m45000 core

操作submit；task_id=['t-20261013170100-4dzcc']；任务名=ab-ag-fixed-75m45000-core-20261013；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/ab_ag_corrected_75m45000_core_20261013.yml；初始状态=submitted_pending_platform_query；Preemptible=False；returncode=0。任务=['ag-binding', 'ag-chimera']。独立输出=/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_corrected_20261013/75m45000；使用固定权重，原训练不变。

## 2026-10-13 提交AB–Ag修正评测 75m138000 core

操作submit；task_id=['t-20261013170647-5fw8p']；任务名=ab-ag-fixed-75m138000-core-20261013；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/ab_ag_corrected_75m138000_core_20261013.yml；初始状态=submitted_pending_platform_query；Preemptible=False；returncode=0。任务=['ag-binding', 'ag-chimera']。独立输出=/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_corrected_20261013/75m138000；使用固定权重，原训练不变。

## 2026-10-13 提交AB–Ag修正评测 270m32000 core

操作submit；task_id=['t-20261013170651-48nk2']；任务名=ab-ag-fixed-270m32000-core-20261013；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/ab_ag_corrected_270m32000_core_20261013.yml；初始状态=submitted_pending_platform_query；Preemptible=False；returncode=0。任务=['ag-binding', 'ag-chimera']。独立输出=/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_corrected_20261013/270m32000；使用固定权重，原训练不变。

## 2026-10-13 提交AB–Ag修正评测 270m32000 abbi0

操作submit；task_id=['t-20261013170654-2jrmb']；任务名=ab-ag-fixed-270m32000-abbi0-20261013；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/ab_ag_corrected_270m32000_abbi0_20261013.yml；初始状态=submitted_pending_platform_query；Preemptible=False；returncode=0。任务=['ag-abbibench:aayl49_benchmarking_data']。独立输出=/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_corrected_20261013/270m32000；使用固定权重，原训练不变。

## 2026-10-13 提交AB–Ag修正评测 270m32000 abbi1

操作submit；task_id=['t-20261013170658-mm925']；任务名=ab-ag-fixed-270m32000-abbi1-20261013；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/ab_ag_corrected_270m32000_abbi1_20261013.yml；初始状态=submitted_pending_platform_query；Preemptible=False；returncode=0。任务=['ag-abbibench:aayl49_ML_benchmarking_data']。独立输出=/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_corrected_20261013/270m32000；使用固定权重，原训练不变。

## 2026-10-13 提交AB–Ag修正评测 270m32000 abbi2

操作submit；task_id=['t-20261013170701-mjjjj']；任务名=ab-ag-fixed-270m32000-abbi2-20261013；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/ab_ag_corrected_270m32000_abbi2_20261013.yml；初始状态=submitted_pending_platform_query；Preemptible=False；returncode=0。任务=['ag-abbibench:aayl52_LC_benchmarking_data']。独立输出=/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_corrected_20261013/270m32000；使用固定权重，原训练不变。

## 2026-10-13 AB–Ag旧队列移交与自动汇总

仅向旧评测目录写入STOP，当前两个4fqi worker继续完成，阻止后续按旧清单重复启动；三项缺失正式assay已独立提交，g6不列入正式范围。没有取消Volc作业。CPU汇总进程PID=2991054已启动，每60秒核验并接入旧作业产物；启动回执=/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_corrected_20261013/watch_process.json；全部模型结果目录=/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_corrected_20261013。

## 2026-10-13 核实评测终态

t-20261012172745-kxt5s（v5-270m-best32000-g2-20261012）平台状态Success，已移出Active；End=2026-10-12T19:51:31Z。查询证据：/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_corrected_20261013/platform_status.json。

## 2026-10-13 修正评测启动验收

六个新作业均已取得平台task ID并确认为Queue；两个旧4fqi任务仍Running，旧g2已确认Success并移出Active。三份快照通过CPU加载和单样本H3接口检查；四模型复用部分通过独立原始行、Spearman和正式范围验收。269000新推理因原权重及快照缺失而阻断。CPU监控已存活并产生首轮汇总。证据：/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_corrected_20261013/cpu_startup_validation.json、/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_corrected_20261013/independent_reuse_validation.json、/project_c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_corrected_20261013/platform_status.json；成绩唯一维护在下游RESULTS新协议节。

## 2026-10-13 75.6M checkpoint45000 单步 CDR 对照准备

用户要求补测one step；固定native object302 checkpoint45000（非新固定双链训练）。Kong三CDR十折、SAb23H2六CDR全量；仅max_iter改为1，batch64/argmax/temperature1/CFG0及Kong bf16/SAb fp32保持。权重SHA与既有多步结果一致；独立硬链接快照、评测脚本及云作业已准备。CPU检查与提交待执行。目录：/project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_single_step45000_20261013；YAML：/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_cdr_single_step45000_20261013.yml。

## 2026-10-13 提交75.6M checkpoint45000 单步CDR评测

操作submit；task_id=t-20261013174432-vx8dp；任务名v5-cdr-single-step-75m45000-20261013；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_cdr_single_step45000_20261013.yml；初始状态submitted_pending_platform_query；Preemptible=false。CPU检查通过：Kong含空轻链4例、SAb六模式6例，非目标上下文保持、隐藏参考替换不改变输入；全量数据清单Kong9381、SAb360区域。完成后独立重算并回填详细及简洁结果表。

## 2026-10-13 75.6M checkpoint45000 单步CDR完成

作业t-20261013174432-vx8dp平台Success，已移出Active表；Preemptible=false，YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/v5_cdr_single_step45000_20261013.yml。9741条预测独立重算、数据与掩码对齐、同权重哈希核对通过。Kong H1/H2/H3=73.9703/67.8281/41.3273；SAb H1/H2/H3/L1/L2/L3=74.76/66.65/35.53/79.66/79.64/71.88。单步结果已并列回填RESULTS.md与RESULTS_SUMMARY.md，原多步行保留；详细过程见EVALUATION_PROCESS.md。证据：/project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_single_step45000_20261013/independent_score_audit.json。

## 2026-10-13 新固定双链50000全下游评测准备

用户指定新固定双链checkpoint50000。已建立独立推理权重硬链接和元数据快照，SHA核验通过；原训练继续。需使用fixed-chain原生布局及生成接口，不能沿用object302的拼接槽位映射。尚未提交GPU作业。快照：/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/checkpoint；选点：/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/selection.json。

2026-10-13 固定双链50000适配完成，CPU实际checkpoint检查通过；36任务分10个单GPU组（AB/TCR6组、AbBiBench4组），先提交g0进行GPU适配检查及Kong/T1/T2/T3，检查通过后提交其余。权重快照保留，不自动释放。/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013

## 2026-10-13 提交固定双链50000评测首组

操作submit；任务名fixed-chain-75m-50000-g0-20261013；task_id=t-20261013182852-cttwd；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_step50000_g0_20261013.yml；初始状态submitted_pending_platform_query；Preemptible=false。先GPU适配检查，再Kong/T1-cdr3b/T2A/T3-broad。

## 2026-10-13 提交固定双链50000评测g1

操作submit；任务名fixed-chain-75m-50000-g1-20261013；task_id=t-20261013183024-7zr8w；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_step50000_g1_20261013.yml；初始状态submitted_pending_platform_query；Preemptible=false。GPU适配检查已通过；任务清单见/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/plan.json。

## 2026-10-13 提交固定双链50000评测g2

操作submit；任务名fixed-chain-75m-50000-g2-20261013；task_id=t-20261013183029-l59t9；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_step50000_g2_20261013.yml；初始状态submitted_pending_platform_query；Preemptible=false。GPU适配检查已通过；任务清单见/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/plan.json。

## 2026-10-13 提交固定双链50000评测g3

操作submit；任务名fixed-chain-75m-50000-g3-20261013；task_id=t-20261013183033-nv7x4；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_step50000_g3_20261013.yml；初始状态submitted_pending_platform_query；Preemptible=false。GPU适配检查已通过；任务清单见/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/plan.json。

## 2026-10-13 提交固定双链50000评测g4

操作submit；任务名fixed-chain-75m-50000-g4-20261013；task_id=t-20261013183036-zkpcv；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_step50000_g4_20261013.yml；初始状态submitted_pending_platform_query；Preemptible=false。GPU适配检查已通过；任务清单见/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/plan.json。

## 2026-10-13 提交固定双链50000评测g5

操作submit；任务名fixed-chain-75m-50000-g5-20261013；task_id=t-20261013183041-rkfkr；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_step50000_g5_20261013.yml；初始状态submitted_pending_platform_query；Preemptible=false。GPU适配检查已通过；任务清单见/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/plan.json。

## 2026-10-13 提交固定双链50000评测g6

操作submit；任务名fixed-chain-75m-50000-g6-20261013；task_id=t-20261013183044-vhft5；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_step50000_g6_20261013.yml；初始状态submitted_pending_platform_query；Preemptible=false。GPU适配检查已通过；任务清单见/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/plan.json。

## 2026-10-13 提交固定双链50000评测g7

操作submit；任务名fixed-chain-75m-50000-g7-20261013；task_id=t-20261013183048-bcnst；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_step50000_g7_20261013.yml；初始状态submitted_pending_platform_query；Preemptible=false。GPU适配检查已通过；任务清单见/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/plan.json。

## 2026-10-13 提交固定双链50000评测g8

操作submit；任务名fixed-chain-75m-50000-g8-20261013；task_id=t-20261013183052-l9kjr；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_step50000_g8_20261013.yml；初始状态submitted_pending_platform_query；Preemptible=false。GPU适配检查已通过；任务清单见/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/plan.json。

## 2026-10-13 提交固定双链50000评测g9

操作submit；任务名fixed-chain-75m-50000-g9-20261013；task_id=t-20261013183056-57tvg；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_step50000_g9_20261013.yml；初始状态submitted_pending_platform_query；Preemptible=false。GPU适配检查已通过；任务清单见/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/plan.json。

## 2026-10-13 固定双链50000评测全组已提交

共36项、10个单GPU作业。平台状态{'Running': 10}；GPU实际checkpoint检查通过（T1三输入、T2/T3、AB–Ag PLL、pairing/T4生成）。已启动完整CDR任务。过程状态自动写EVALUATION_PROCESS.md；结果待完成核验后回填任务表，不将运行状态写入成绩表。运行目录：/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013。

## 2026-10-13 固定双链50000首批下游成绩回填

2026-10-13T13:14:33.796499+00:00 固定双链75.6M checkpoint50000：36项中30项success、2项running、4项pending，0失败；平台g0–g7 Success、g8/g9 Running。20项非AbBiBench任务全部完成；AbBiBench官方16项完成10项，未发布完整均值。已回填RESULTS.md及RESULTS_SUMMARY.md的对应任务榜单，保留V5/V6/V7和原native成绩。CDR 9741条预测独立重算通过，T1四种输入各五折齐全。证据：[fixed_chain_75m_step50000_20261013](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_20261013/published_results_audit.json)。

## 2026-10-13 提交fixed-chain50000 reference长度配对对照

操作submit；task_id=t-20261014003602-r5tsx；任务名=fixed-chain-75m-50000-pair-ref-20261013；YAML=/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_step50000_pairing_reference_20261013.yml；初始状态submitted_pending_platform_query；Preemptible=false。CPU500条输入及实际前向检查通过；GPU先复现原生EOS首候选，再执行500×8 reference长度生成和原评分器。当前训练继续。[协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)；[产物](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_pairing_reference_20261013)。

## 2026-10-13 reference配对：本地资源核查

2026-10-13T16:39:18.320345+00:00 用户要求本地空闲优先测试。当前仅可见1张A100-80GB，GPU利用率100%，占用约14.3GiB；Protenix-v2仍有3个DockQ实验进程。尚未在本地启动或停止其他实验；本次云任务t-20261014003602-r5tsx最新为Queue，保留任务。[核查证据](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_pairing_reference_20261013/local_availability.json)；长期调度偏好已写PROJ_GUIDE.md。


## 2026-10-14 Submitted fixed-chain best112000 g0

操作submit；task_id=t-20261014133005-n4t4m；任务名=fixed-chain-75m-best112000-g0-20261014；YAML=[fixed_chain_75m_best112000_g0_20261014.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_best112000_g0_20261014.yml)；初始状态=Initialized (submitted; platform query pending)；Preemptible=false；单GPU。任务范围：cdr-kong, t1-cdr3b, t2a, t3broad。独立模型快照与校验记录：[运行目录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_best112000_20261014)。当前训练继续，未提交训练作业或修改训练进程。


## 2026-10-14 Submitted fixed-chain best112000 g1

操作submit；task_id=t-20261014133009-h2lwb；任务名=fixed-chain-75m-best112000-g1-20261014；YAML=[fixed_chain_75m_best112000_g1_20261014.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_best112000_g1_20261014.yml)；初始状态=Initialized (submitted; platform query pending)；Preemptible=false；单GPU。任务范围：cdr-sab23, specificity, t1-others_cdr3b。独立模型快照与校验记录：[运行目录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_best112000_20261014)。当前训练继续，未提交训练作业或修改训练进程。


## 2026-10-14 Submitted fixed-chain best112000 g2

操作submit；task_id=t-20261014133012-6gm5k；任务名=fixed-chain-75m-best112000-g2-20261014；YAML=[fixed_chain_75m_best112000_g2_20261014.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_best112000_g2_20261014.yml)；初始状态=Initialized (submitted; platform query pending)；Preemptible=false；单GPU。任务范围：gdp_a1, m396, t1-cdr3ab, t3deep, uptcr。独立模型快照与校验记录：[运行目录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_best112000_20261014)。当前训练继续，未提交训练作业或修改训练进程。


## 2026-10-14 Submitted fixed-chain best112000 g3

操作submit；task_id=t-20261014133016-wqdk9；任务名=fixed-chain-75m-best112000-g3-20261014；YAML=[fixed_chain_75m_best112000_g3_20261014.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_best112000_g3_20261014.yml)；初始状态=Initialized (submitted; platform query pending)；Preemptible=false；单GPU。任务范围：ag-chimera, ag-binding, ag-spr, t1-others_longab。独立模型快照与校验记录：[运行目录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_best112000_20261014)。当前训练继续，未提交训练作业或修改训练进程。


## 2026-10-14 Submitted fixed-chain best112000 g4

操作submit；task_id=t-20261014133019-xq5xz；任务名=fixed-chain-75m-best112000-g4-20261014；YAML=[fixed_chain_75m_best112000_g4_20261014.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_best112000_g4_20261014.yml)；初始状态=Initialized (submitted; platform query pending)；Preemptible=false；单GPU。任务范围：pairing, pairing-reference。独立模型快照与校验记录：[运行目录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_best112000_20261014)。当前训练继续，未提交训练作业或修改训练进程。


## 2026-10-14 Submitted fixed-chain best112000 g5

操作submit；task_id=t-20261014133023-jl99f；任务名=fixed-chain-75m-best112000-g5-20261014；YAML=[fixed_chain_75m_best112000_g5_20261014.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_best112000_g5_20261014.yml)；初始状态=Initialized (submitted; platform query pending)；Preemptible=false；单GPU。任务范围：t4-benchmark14, t4-held20。独立模型快照与校验记录：[运行目录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_best112000_20261014)。当前训练继续，未提交训练作业或修改训练进程。


## 2026-10-14 Submitted fixed-chain best112000 g6

操作submit；task_id=t-20261014133027-wn6db；任务名=fixed-chain-75m-best112000-g6-20261014；YAML=[fixed_chain_75m_best112000_g6_20261014.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_best112000_g6_20261014.yml)；初始状态=Initialized (submitted; platform query pending)；Preemptible=false；单GPU。任务范围：ag-abbibench:1mhp_LC_benchmarking_data, ag-abbibench:3gbn_h1_benchmarking_data, ag-abbibench:4d5_her2_benchmarking_data_trimmed, ag-abbibench:aayl49_benchmarking_data。独立模型快照与校验记录：[运行目录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_best112000_20261014)。当前训练继续，未提交训练作业或修改训练进程。


## 2026-10-14 Submitted fixed-chain best112000 g7

操作submit；task_id=t-20261014133031-kjh8m；任务名=fixed-chain-75m-best112000-g7-20261014；YAML=[fixed_chain_75m_best112000_g7_20261014.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_best112000_g7_20261014.yml)；初始状态=Initialized (submitted; platform query pending)；Preemptible=false；单GPU。任务范围：ag-abbibench:1mlc_benchmarking_data, ag-abbibench:3gbn_h9_benchmarking_data, ag-abbibench:5a12_ang2_benchmarking_data, ag-abbibench:aayl49_ML_benchmarking_data。独立模型快照与校验记录：[运行目录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_best112000_20261014)。当前训练继续，未提交训练作业或修改训练进程。


## 2026-10-14 Submitted fixed-chain best112000 g8

操作submit；task_id=t-20261014133035-5wv8h；任务名=fixed-chain-75m-best112000-g8-20261014；YAML=[fixed_chain_75m_best112000_g8_20261014.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_best112000_g8_20261014.yml)；初始状态=Initialized (submitted; platform query pending)；Preemptible=false；单GPU。任务范围：ag-abbibench:1n8z_benchmarking_data, ag-abbibench:4fqi_h1_benchmarking_data, ag-abbibench:5a12_vegf_benchmarking_data, ag-abbibench:aayl51_benchmarking_data。独立模型快照与校验记录：[运行目录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_best112000_20261014)。当前训练继续，未提交训练作业或修改训练进程。


## 2026-10-14 Submitted fixed-chain best112000 g9

操作submit；task_id=t-20261014133038-4n4jv；任务名=fixed-chain-75m-best112000-g9-20261014；YAML=[fixed_chain_75m_best112000_g9_20261014.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/fixed_chain_75m_best112000_g9_20261014.yml)；初始状态=Initialized (submitted; platform query pending)；Preemptible=false；单GPU。任务范围：ag-abbibench:2fjg_benchmarking_data, ag-abbibench:4fqi_h3_benchmarking_data, ag-abbibench:aayl50_LC_benchmarking_data, ag-abbibench:aayl52_LC_benchmarking_data。独立模型快照与校验记录：[运行目录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_best112000_20261014)。当前训练继续，未提交训练作业或修改训练进程。


## 2026-10-14 Evaluation submission platform check

2026-10-14T05:31:05.260908+00:00；10个评测任务提交完成，当前{'Running': 7, 'Staging': 3}；逐任务平台回执见[platform_status.json](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_best112000_20261014/platform_status.json)。

## 2026-10-14T10:38:29.740885+00:00 best112000下游状态核实

2026-10-14T10:38:29.740885+00:00 best112000下游评测状态核实：10个作业中8个Success、2个Running；36子任务30个success、2个running、4个pending，0失败。20项非AbBiBench任务全部完成；AbBiBench完成10/16，4fqi_h1与4fqi_h3运行中，其余4项等待同组前序任务。当前日志进度分别为25571/65094（39.3%）、24911/65535（38.0%）。[平台查询](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_best112000_20261014/platform_status.json)。

已从Active表移除终态作业g0–g7：t-20261014133005-n4t4m, t-20261014133009-h2lwb, t-20261014133012-6gm5k, t-20261014133016-wqdk9, t-20261014133019-xq5xz, t-20261014133023-jl99f, t-20261014133027-wn6db, t-20261014133031-kjh8m。本轮仅查询状态，未提交、取消或重启作业。

## 2026-10-14T10:41:42.925362+00:00 转入架构与后训练探索

2026-10-14T10:41:42.925362+00:00 按用户新的架构/后训练探索目标，请求停止固定双链预训练，保留现有八卡容器；131000完整恢复点（含优化器/RNG）逐文件哈希通过，112000作为后训练共同初始化独立保存。当前仅stop_requested，等待训练子进程真实退出。[证据](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/stop_request.json)。

## 2026-10-14T10:51:19.052802+00:00 启动固定双链架构与后训练筛选

2026-10-14T10:51:19.052802+00:00 固定双链架构/后训练首轮12组对照已入队现有八卡容器t-20261007153431-vhx2f；每组112000初始化、1000步、batch32。原预训练已确认主动退出，131000完整恢复点保留。85个原始分片哈希与原审核一致；筛选/确认集分离；CPU检查33项及新增验证视图检查通过。[执行证据](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/enqueue.json)，[实验方案](/project_c20250601/251105016/project/dllm_test/docs/FIXED_CHAIN_EXPLORATION_20261014.md)。

复用原非闲时资源YAML /project_c20250601/251105016/project/dllm_test/train_jobs/immune_training_container_8gpu_20261007.yml；本轮无新平台submit/cancel。仅归档本轮PAUSE_TRAINING以允许新票据，原预训练不自动恢复。

## 2026-10-14 轻链IM受控debug：fixed50000 legacy采样

操作submit；task_id=t-20261014185128-s98d5；任务名pairing-debug-fixed50000-legacy-20261014；[评测YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/pairing_debug_fixed50000_legacy_20261014.yml)；初始Submitted（待平台查询），Preemptible=true。只改变采样器，正式快照及原始成绩保持；worker先精确复现原GPU首候选，再执行500×8生成和原评分器。

## 2026-10-14 轻链IM受控debug：fixed50000概率排序对照

操作submit；task_id=t-20261014185159-2422w；任务名pairing-debug-fixed50000-gumbel-prob-20261014；[评测YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/pairing_debug_fixed50000_gumbel_prob_20261014.yml)；平台Queue，Preemptible=true。运行冻结runner_v1，500×8；同Gumbel提案、不同置信度提交规则。

fixed50000 legacy任务t-20261014185128-s98d5平台已于10:52:16 UTC进入Killed终态，已移出Active；生成日志到26/500。保留部分输出，正在核查终止原因，尚未自动重试。

## 2026-10-14 轻链IM受控debug：fixed112000 legacy采样

操作submit；task_id=t-20261014185500-5bm4j；任务名pairing-debug-fixed112000-legacy-20261014；[评测YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/pairing_debug_fixed112000_legacy_20261014.yml)；Submitted待查询，Preemptible=true。独立完整500×8验证，以检验采样器问题是否跨checkpoint存在。

V7诊断加载增加显式allow_snapshot_relocations选项：仅允许历史工作区文件解析至manifest独立列出、SHA256完全一致的冻结源码，严格校验实际导入路径；默认仍拒绝变更。原检查拒绝的是已修改工作区layout/model，而冻结副本完整。正在进行CPU复现，未改权重、原provenance或历史预测。

## 2026-10-14 轻链IM诊断：资源回收后独立恢复

平台instance给出原legacy任务终止原因：`reclaimable queue needs to return the borrowed resource`；非模型异常。原任务已Killed，不复用其26条部分输出。操作submit；新task_id=t-20261014185646-bhfxb；任务名pairing-debug-fixed50000-legacy-dedicated-20261014；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/pairing_debug_fixed50000_legacy_dedicated_20261014.yml)；Submitted，Preemptible=false；使用原冻结runner_v1、全新输出目录，完整500×8。

## 2026-10-14 轻链IM诊断：V7反向采样器对照

操作submit；task_id=t-20261014185717-bx9fg；任务名pairing-debug-v7-native-dedicated-20261014；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/pairing_debug_v7_native_dedicated_20261014.yml)；平台Queue，Preemptible=false。冻结runner_v2；V7权重及A4输入不变，仅采用fixed-chain概率提交规则。CPU旧解码器复现已通过，GPU必须先复现历史首候选。10:57 UTC其余三个活跃诊断任务均Queue。

## 2026-10-14 轻链IM诊断：迁移首项到本地GPU

操作cancel；task_id=t-20261014185646-bhfxb；任务名pairing-debug-fixed50000-legacy-dedicated-20261014；Preemptible=false；[原YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/pairing_debug_fixed50000_legacy_dedicated_20261014.yml)。11:08:39 UTC平台Killed，已移出Active。逐进程设备句柄和脚本确认本地GPU仅运行随机矩阵占位，无训练/评测；占位进程PID1909保留。使用同YAML冻结入口在本地运行完整500×8，云未产生候选；[资源核验](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/pairing_sampler_debug_20261014/local_resource_check.json)。

本地首项已于11:09:43 UTC启动，shell PID3392934，入口会话21735；GPU历史首候选复现通过并开始生成。环境同YAML，ANARCI/Joblib限制4 workers；[运行身份](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/pairing_sampler_debug_20261014/local_fixed50000_execution.json)。

## 2026-10-14 轻链IM诊断：迁移V7反向对照

操作cancel；task_id=t-20261014185717-bx9fg；任务名pairing-debug-v7-native-dedicated-20261014；Preemptible=false；[原YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/pairing_debug_v7_native_dedicated_20261014.yml)；cancel已受理，待终态后以同冻结入口在本地运行。最多两个本轮GPU进程并行，各自独立RNG与输出；不影响其他训练容器，不停止本地占卡脚本。

11:10:43 UTC已确认V7云任务Killed，移出Active；本地首项23/500，GPU总显存2574MiB，只有本轮生成与已核实占卡进程。反向对照在全新v7_native输出目录执行，不能把之前CPU检查当GPU历史复现通过。

## 2026-10-14T10:55:05.328066+00:00 修正探索启动依赖

2026-10-14T10:55:05.328066+00:00 首轮后训练探索修正版r3已入队现有八卡容器；旧attempt因冻结包缺examples依赖在导入阶段失败、无权重更新，已确认退出。r3补齐依赖并通过隔离入口与22项冻结源码测试，同时按分片来源区分tcr_papers/tcr_native别名；12组对照、112000初始化及预算不变。[回执](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/enqueue_r3.json)。

复用原非闲时八卡容器及既有train_jobs/immune_training_container_8gpu_20261007.yml，无新平台提交。r2仅准备未执行，旧失败票据不覆盖、不重复执行。

## 2026-10-14T11:05:31.699861+00:00 repair门禁失败修正并排队

2026-10-14T11:05:31.699861+00:00 r3中7组通过CUDA门禁并已更新约410–430步，4组等待前序；repair在CUDA反向门禁失败、0更新。已复现并修正no_grad首次预测污染autocast权重缓存的问题，24项CPU测试（含两次前向+checkpoint+bf16回归）通过；仅修正后的repair票据r4排在当前r3之后，其余实验继续。[状态](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/current.json)。

仅新修正版票据入队现有非闲时容器；无新平台submit/cancel，不重跑成功或正在运行的候选。

## 2026-10-14T11:18:27.540790+00:00 轻链IM诊断迁移本地待运行队列

操作cancel；task_id=t-20261014185159-2422w；任务名pairing-debug-fixed50000-gumbel-prob-20261014；Preemptible=true；[原YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/pairing_debug_fixed50000_gumbel_prob_20261014.yml)；cancel已受理。终态确认后由本地两并发队列执行同冻结入口，等待当前两项结束；旧输出不复用。

t-20261014185159-2422w 已核实平台Killed（End=2026-10-14T11:18:29Z），从Active移除。

## 2026-10-14T11:18:32.475637+00:00 轻链IM诊断迁移本地待运行队列

操作cancel；task_id=t-20261014185500-5bm4j；任务名pairing-debug-fixed112000-legacy-20261014；Preemptible=true；[原YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/pairing_debug_fixed112000_legacy_20261014.yml)；cancel已受理。终态确认后由本地两并发队列执行同冻结入口，等待当前两项结束；旧输出不复用。

t-20261014185500-5bm4j 已核实平台Killed（End=2026-10-14T11:18:34Z），从Active移除。

## 2026-10-14T11:22:26.703013+00:00 第二轮后训练筛选入队

2026-10-14T11:22:26.703013+00:00 首轮r3的11个健康候选均完成1000步；repair修正版r4已通过CUDA并运行。第二轮r5八组（task_mix对照、observed_eos、CDR、组合、四步教师蒸馏及结合标签辅助）通过40项CPU与冻结包测试并入队；保持输入布局，额外注释仅用于loss/评分，确认集仍未打开。[回执](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/enqueue_r5.json)。

复用现有非闲时容器与既有train_jobs/immune_training_container_8gpu_20261007.yml，未新增平台作业。真实数据预检：91861个有注释链全部区域拼接精确一致，192对标签翻转不改变分类器输入。

## 2026-10-14T11:41:24.250929+00:00 弱标签后训练对照入队

2026-10-14T11:41:24.250929+00:00 r5八组均完成1000步；r6七组（continued/task_mix/observed_cdr基线、弱偏好β1/β10、固定打乱方向、正例SFT）已入队现有八卡容器。52项冻结CPU测试、实际112000权重与21对screen配对前向通过；真实CUDA门禁待运行。确认集仍未打开。 [回执](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/enqueue_r6.json)。

## 2026-10-14T11:43:59.633330+00:00 多种子长程验证入队

2026-10-14T11:43:59.633330+00:00 r6七组均通过真实CUDA前后向并已产生更新；r7五方案×三种子（42/43/44）共15组、各3000步已排队。方案为continued、task_mix、observed_eos、observed_cdr、LoRA；通过53项冻结测试及子计划哈希门禁，确认集未打开。 [回执](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/enqueue_r7.json)。

## 2026-10-14T11:55:14.879733+00:00 确认评测准备

2026-10-14T11:55:14.879733+00:00 确认评测实现已完成准备：保存逐记录统计，按来源内受体tuple聚类与配对训练种子做5000次bootstrap，四候选Bonferroni校正；主指标为全mask、条件补链、CDR三域AA-CE相对匹配continued的平均变化，另设来源/部分mask/生成有效率与长度退步门限。阈值在打开确认集前记录，尚未运行确认评测。 逐记录验证54项测试通过；确认统计与模型回归组合49项通过（两组有重叠，不累计计数）。后训练checkpoint的下游桥接正在实际CPU检查。

2026-10-14T11:57:48.119928+00:00 pairing本地队列：v7_native 终态 {'success': True, 'exit_code': 0}；无自动重试。

2026-10-14T11:57:48.659668+00:00 pairing本地队列：fixed50000_gumbel_prob 启动，shell PID3484101；同冻结YAML入口，云t-20261014185159-2422w已Killed；[产物](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/pairing_sampler_debug_20261014/fixed50000_gumbel_prob)。

2026-10-14T11:59:48.822418+00:00 pairing本地队列：fixed50000_legacy_dedicated 终态 {'success': True, 'exit_code': 0}；无自动重试。

2026-10-14T11:59:49.378341+00:00 pairing本地队列：fixed112000_legacy 启动，shell PID3504983；同冻结YAML入口，云t-20261014185500-5bm4j已Killed；[产物](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/pairing_sampler_debug_20261014/fixed112000_legacy)。

## 2026-10-14T12:00:53.833720+00:00 冻结表征对照入队

2026-10-14T12:00:53.833720+00:00 八模型冻结表征诊断已排在r7长程训练之后；每来源/类别最多256条训练记录，C仅在训练内受体分组划分选择，512条screen记录报告来源AUROC与训练标签打乱对照。通过两项冻结测试（关系标签翻转不改变输入；screen标签不参与选C）。不访问确认/benchmark测试集。 [回执](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/enqueue_representation_probe_r1.json)。

## 2026-10-14T12:22:38.287065+00:00 全量后训练与恢复机制准备

2026-10-14T12:22:38.287065+00:00 正式后训练准备：85原始分片全哈希一致；全量索引保留7,626,748条，排除与原valid完全重合的137条受体记录，原JSONL未改。实际全量loader及7来源collator检查通过。新增八卡native Trainer入口复用已有完整optimizer/scheduler/8-rank RNG检查；任务混合与LoRA的CPU中断恢复均与连续训练逐参数、逐优化器状态一致。尚未选定最终方案，正式启动包不能在确认决策缺失时构建。 [数据验收](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/full_data_loader_preflight.json)，[恢复测试](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/formal_post_training_tests.log)。


## 2026-10-14T12:40:33.785852+00:00 三种子长程筛选完成

r7五方案×三种子共15组均成功完成3000步，权重导出验收进行中；容器自动接续八模型表征诊断。确认集仍未打开，尚未选定最终方案。[三种子筛选汇总](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/screen_r7_summary.json)。


## 2026-10-14T12:43:50.280649+00:00 独立确认评测入队

15组长程训练及8模型表征诊断均成功；15份导出参数/实际logits一致并绑定原训练源码。保持原五family、三种子及预先阈值，连同112000 parent共16模型冻结确认评测已入队既有非闲时八卡容器；无新平台submit/cancel。结合辅助的筛选AUROC小幅增益配对区间跨零，不宣称已确认收益。[入队](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/enqueue_confirmation_r1.json)，[候选冻结依据](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/candidate_freeze_r1.json)。

2026-10-14T12:43:52.144800+00:00 pairing本地队列：fixed50000_gumbel_prob 终态 {'success': True, 'exit_code': 0}；无自动重试。

2026-10-14T12:45:52.293058+00:00 pairing本地队列：fixed112000_legacy 终态 {'success': True, 'exit_code': 0}；无自动重试。


## 2026-10-14T12:46:24.883Z 历史下游平台终态核对

Active表中13个历史下游作业已逐项查询：11个Success移出Active，best112000的g8/g9仍Running。仅修正文档状态，无取消或重跑。[原始平台结果](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/legacy_downstream_platform_status.json)。


## 2026-10-14T12:50:48.718337+00:00 确认选定LoRA并启动正式后训练

16模型独立确认全部成功，唯一通过全部预设门限的方案为rank16 LoRA+task_mix；相对continued的三种子平均主指标变化−0.50198%，多重校正区间[−0.75886%,−0.27262%]。正式包69项冻结测试和元数据检查通过，首50步已入队现有非闲时八卡容器；112000初始化，7626748条可用全量索引，global512，LR2e-5，warmup200，总horizon10000，eval/save500。50→100→后续10000共用同一schedule/source，后两段待实际完整恢复点验收；当前不能宣称真实训练步已完成。[确认决策](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/confirmation_r1/selection.json)，[入队回执](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/enqueue_final_r1_stage1.json)。无新增平台训练提交。


## 2026-10-14T12:51:36.839424+00:00 已确认LoRA下游g0提交

操作submit；task_id=['t-20261014205134-bkrbt']；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_r1_g0_20261014.yml)；初始Submitted待查询；Preemptible=false。固定所选family的seed42/3000步权重，既有AB/TCR协议不调参。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_r1_20261014/submissions.json)。


## 2026-10-14T12:51:42.484397+00:00 已确认LoRA下游g1提交

操作submit；task_id=['t-20261014205140-rsp8w']；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_r1_g1_20261014.yml)；初始Submitted待查询；Preemptible=false。固定所选family的seed42/3000步权重，既有AB/TCR协议不调参。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_r1_20261014/submissions.json)。


## 2026-10-14T12:51:47.298200+00:00 已确认LoRA下游g2提交

操作submit；task_id=['t-20261014205145-gbwrd']；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_r1_g2_20261014.yml)；初始Submitted待查询；Preemptible=false。固定所选family的seed42/3000步权重，既有AB/TCR协议不调参。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_r1_20261014/submissions.json)。


## 2026-10-14T12:51:53.811207+00:00 已确认LoRA下游g3提交

操作submit；task_id=['t-20261014205151-vlhvs']；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_r1_g3_20261014.yml)；初始Submitted待查询；Preemptible=false。固定所选family的seed42/3000步权重，既有AB/TCR协议不调参。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_r1_20261014/submissions.json)。


## 2026-10-14T12:51:57.278775+00:00 已确认LoRA下游g4提交

操作submit；task_id=['t-20261014205155-vdrg8']；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_r1_g4_20261014.yml)；初始Submitted待查询；Preemptible=false。固定所选family的seed42/3000步权重，既有AB/TCR协议不调参。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_r1_20261014/submissions.json)。


## 2026-10-14T12:52:01.068381+00:00 已确认LoRA下游g5提交

操作submit；task_id=['t-20261014205158-zdwcq']；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_r1_g5_20261014.yml)；初始Submitted待查询；Preemptible=false。固定所选family的seed42/3000步权重，既有AB/TCR协议不调参。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_r1_20261014/submissions.json)。


## 2026-10-14T12:52:04.743775+00:00 已确认LoRA下游g6提交

操作submit；task_id=['t-20261014205202-8ccb2']；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_r1_g6_20261014.yml)；初始Submitted待查询；Preemptible=false。固定所选family的seed42/3000步权重，既有AB/TCR协议不调参。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_r1_20261014/submissions.json)。


## 2026-10-14T12:52:08.328561+00:00 已确认LoRA下游g7提交

操作submit；task_id=['t-20261014205206-f228j']；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_r1_g7_20261014.yml)；初始Submitted待查询；Preemptible=false。固定所选family的seed42/3000步权重，既有AB/TCR协议不调参。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_r1_20261014/submissions.json)。


## 2026-10-14T12:52:11.992451+00:00 已确认LoRA下游g8提交

操作submit；task_id=['t-20261014205209-85z9v']；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_r1_g8_20261014.yml)；初始Submitted待查询；Preemptible=false。固定所选family的seed42/3000步权重，既有AB/TCR协议不调参。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_r1_20261014/submissions.json)。


## 2026-10-14T12:52:15.327628+00:00 已确认LoRA下游g9提交

操作submit；task_id=['t-20261014205213-5299k']；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_r1_g9_20261014.yml)；初始Submitted待查询；Preemptible=false。固定所选family的seed42/3000步权重，既有AB/TCR协议不调参。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_r1_20261014/submissions.json)。


## 2026-10-14T12:55:23.863254+00:00 LoRA正式50步验收与恢复验证

首段真实50更新、exit0；17个checkpoint文件哈希、八rank一致性、冻结模型strict reload、optimizer step50及scheduler50均核验通过。第二段从checkpoint50完整恢复至100已入队，不重置优化器/调度。八卡S1136/micro64 profile峰值约15.14GiB，固定global512下GA=1已是最大micro；真实训练峰值约16.17GiB。[验收](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/validated_checkpoint_50.json)，[入队](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/enqueue_final_r1_stage2.json)。


## 2026-10-14T12:58:18.554119+00:00 LoRA原生恢复验收与主段入队

50→100实际恢复训练exit0，八rank在原生恢复函数返回时的optimizer/scheduler/RNG内存状态逐项与保存状态哈希一致；checkpoint100完整17文件及实际八卡参数一致性再核验通过。100→10000正式主段已入队同一容器、同一source/schedule。未声称八卡连续轨迹逐bit等价。[原生恢复证据](/project_c20250601/251105016/project/dllm_test/output/immune_fixed_chain_post_training_final_r1_20261014/RESUME_50_TO_100.json)，[验收](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/validated_checkpoint_100.json)，[主段](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/enqueue_final_r1_stage3.json)。


## 2026-10-14T13:02:15.800552+00:00 最终方案训练部署验收完成

LoRA独立确认、完整50/100步checkpoint、真实八卡50→100状态恢复、主段从100继续至240步均验收通过。选型及布置训练这一目标完成；10000步训练仍在运行，36任务下游仍待平台执行，不能声称训练结束或下游已提升。[统一验收](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/deployment_complete.json)。

## 2026-10-15T03:32:47.066512+00:00 LoRA续训至完整覆盖epoch

原LoRA10000步及容器票据成功退出已核实；八卡此前standby。按用户要求将累计终点设为33981步（剩余23981），复用现有非闲时容器，未另申请资源。冻结CPU恢复检查7项通过；从原完整checkpoint恢复模型、optimizer、scheduler、八rank RNG与数据游标，新增200步预热至2e-5后余弦衰减。10000→10050阶段已入队，真实GPU恢复尚待验收。[计划](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_lora_epoch1_20261015/continuation_plan.json)，[回执](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_lora_epoch1_20261015/enqueue_stage1.json)。

2026-10-15T03:37:04.101972+00:00 首段10000→10050实际成功退出；八rank原生optimizer/scheduler/RNG恢复均匹配，17份完整checkpoint文件核对通过，112个LoRA张量发生更新、基座张量逐值不变，LR=5e-6且AdamW累计步数10050。主段10050→33981已入队同一容器。[验收](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_lora_epoch1_20261015/stage1_acceptance.json)。

2026-10-15T03:39:56.498595+00:00 主段GPU检查及通信8/8通过，已从10050继续推进至10117/33981，当前standby为空；由原生回调在33981步验证/保存后正常退出。完成恢复验收和续训部署，尚未跑完一个epoch。[当前验收](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_lora_epoch1_20261015/startup_acceptance.json)。

## 2026-10-15 CDR 免训练多轮评测准备

新增隔离采样器和冻结评测包，真实 checkpoint CPU 门禁通过；第一轮 GPU 作业已成功，独立审计通过；第二、三轮确认均已成功，使用同一冻结配置并完成最终核验。第一轮探索、第二轮保留组确认、第三轮 checkpoint112000 跨权重确认。产物：[cdr_inference_rounds_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_inference_rounds_20261015)。

## 2026-10-15 CDR免训练第1轮 submit（补记即时查询可见性延迟）

- task_id=`t-20261015115717-nkf6t`；状态 `Running`；记录时间 `2026-10-15T03:58:04.079797+00:00`；`Preemptible: false`。
- 配置：[cdr_inference_round1_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_inference_round1_20261015.yml)；执行产物：[cdr_inference_rounds_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_inference_rounds_20261015)。

## 2026-10-15 活跃表终态核对

按具体任务 ID 查询，移除已终态历史行；没有取消或重启其他任务。证据：[逐 ID 查询](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_inference_rounds_20261015/active_task_reconciliation.json)。

- `t-20261014133035-5wv8h`：`Failed`。
- `t-20261014133038-4n4jv`：`Failed`。
- `t-20261014205134-bkrbt`：`Success`。
- `t-20261014205140-rsp8w`：`Success`。
- `t-20261014205145-gbwrd`：`Success`。
- `t-20261014205151-vlhvs`：`Success`。
- `t-20261014205155-vdrg8`：`Success`。
- `t-20261014205158-zdwcq`：`Success`。
- `t-20261014205202-8ccb2`：`Success`。
- `t-20261014205206-f228j`：`Success`。

## 2026-10-15 CDR免训练第1轮 终态

- task_id=`t-20261015115717-nkf6t`；状态 `Success`；记录时间 `2026-10-15T04:01:37.316433+00:00`；`Preemptible: false`。
- 配置：[cdr_inference_round1_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_inference_round1_20261015.yml)；执行产物：[cdr_inference_rounds_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_inference_rounds_20261015)。

## 2026-10-15 CDR免训练第2轮 submit

- task_id=`t-20261015120224-vvpvl`；状态 `提交成功，平台状态待查询`；记录时间 `2026-10-15T04:02:26.570471+00:00`；`Preemptible: false`。
- 配置：[cdr_inference_round2_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_inference_round2_20261015.yml)；执行产物：[cdr_inference_rounds_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_inference_rounds_20261015)。

## 2026-10-15 CDR免训练第2轮 初始状态确认

- task_id=`t-20261015120224-vvpvl`；状态 `Queue`；记录时间 `2026-10-15T04:02:28.701533+00:00`；`Preemptible: false`。
- 配置：[cdr_inference_round2_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_inference_round2_20261015.yml)；执行产物：[cdr_inference_rounds_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_inference_rounds_20261015)。

## 2026-10-15 CDR免训练第3轮 submit

- task_id=`t-20261015120450-qcq8s`；状态 `提交成功，平台状态待查询`；记录时间 `2026-10-15T04:04:53.250984+00:00`；`Preemptible: false`。
- 配置：[cdr_inference_round3_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_inference_round3_20261015.yml)；执行产物：[cdr_inference_rounds_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_inference_rounds_20261015)。

## 2026-10-15 CDR免训练第3轮 初始状态确认

- task_id=`t-20261015120450-qcq8s`；状态 `Queue`；记录时间 `2026-10-15T04:04:53.751857+00:00`；`Preemptible: false`。
- 配置：[cdr_inference_round3_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_inference_round3_20261015.yml)；执行产物：[cdr_inference_rounds_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_inference_rounds_20261015)。

2026-10-15：R1 oracle成对统计已补齐；R2/R3确认作业继续运行。最终核验入口见[执行诊断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md)。

## 2026-10-15 CDR免训练第2轮 终态

- task_id=`t-20261015120224-vvpvl`；状态 `Success`；记录时间 `2026-10-15T04:28:05.968340+00:00`；`Preemptible: false`。
- 配置：[cdr_inference_round2_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_inference_round2_20261015.yml)；执行产物：[cdr_inference_rounds_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_inference_rounds_20261015)。

## 2026-10-15 CDR免训练第3轮 终态

- task_id=`t-20261015120450-qcq8s`；状态 `Success`；记录时间 `2026-10-15T04:30:55.950434+00:00`；`Preemptible: false`。
- 配置：[cdr_inference_round3_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_inference_round3_20261015.yml)；执行产物：[cdr_inference_rounds_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_inference_rounds_20261015)。

## 2026-10-15 CDR免训练三轮验收完成

全部实验及审计完成；正式默认、历史主榜与两套权重保持。[三轮诊断结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-inference-rounds-results-20261015)；[逐项完成审计](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_inference_rounds_20261015/completion_audit.json)。

## 2026-10-15 提交 OAS-only scratch 八卡训练

操作 submit；task_id=`t-20261015124455-xrns4`；任务名=`immune-fixed75m-oas-scratch-8gpu-queue012-20261015`；YAML=/project_c20250601/251105016/project/dllm_test/train_jobs/v5_fixed_chain_75m_oas_scratch_8gpu_queue012_20261015.yml；初始状态=`Queue`；`Preemptible: false`；队列=`queue012`（`q-20260524172355-rnqtf`）；1 × `ml.pni2.28xlarge`。returncode=0。尚未分配 GPU，正式训练 0 步；launcher 启动后先做八卡 synthetic 显存检查。证据：[submission](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_queue012_20261015/submission.json)，[平台查询](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_queue012_20261015/platform_status.json)。

## 2026-10-15 CDR概率聚合新探索112000 submit

- task_id=`t-20261015125726-2sthb`；状态 `提交成功，平台状态待查询`；记录时间 `2026-10-15T04:57:28.428153+00:00`；`Preemptible: false`。
- 配置：[cdr_marginal_explore112000_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_marginal_explore112000_20261015.yml)；执行产物：[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

## 2026-10-15 CDR概率聚合新探索112000 初始状态确认

- task_id=`t-20261015125726-2sthb`；状态 `Queue`；记录时间 `2026-10-15T04:57:28.937368+00:00`；`Preemptible: false`。
- 配置：[cdr_marginal_explore112000_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_marginal_explore112000_20261015.yml)；执行产物：[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

## 2026-10-15 CDR概率聚合新探索112000 终态

- task_id=`t-20261015125726-2sthb`；状态 `Success`；记录时间 `2026-10-15T05:02:10.123089+00:00`；`Preemptible: false`。
- 配置：[cdr_marginal_explore112000_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_marginal_explore112000_20261015.yml)；执行产物：[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

## 2026-10-15 CDR checkpoint概率集成探索 submit

- task_id=`t-20261015130746-jhj9c`；状态 `提交成功，平台状态待查询`；记录时间 `2026-10-15T05:07:49.123374+00:00`；`Preemptible: false`。
- 配置：[cdr_checkpoint_ensemble_probe_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_checkpoint_ensemble_probe_20261015.yml)；执行产物：[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

## 2026-10-15 CDR checkpoint概率集成探索 初始状态确认

- task_id=`t-20261015130746-jhj9c`；状态 `Queue`；记录时间 `2026-10-15T05:07:51.366718+00:00`；`Preemptible: false`。
- 配置：[cdr_checkpoint_ensemble_probe_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_checkpoint_ensemble_probe_20261015.yml)；执行产物：[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

## 2026-10-15 CDR checkpoint概率集成探索 终态

- task_id=`t-20261015130746-jhj9c`；状态 `Success`；记录时间 `2026-10-15T05:09:08.233419+00:00`；`Preemptible: false`。
- 配置：[cdr_checkpoint_ensemble_probe_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_checkpoint_ensemble_probe_20261015.yml)；执行产物：[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

## 2026-10-15 CDR上下文强度探索 submit

- task_id=`t-20261015131106-57bfk`；状态 `提交成功，平台状态待查询`；记录时间 `2026-10-15T05:11:09.141835+00:00`；`Preemptible: false`。
- 配置：[cdr_context_contrast_probe_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_context_contrast_probe_20261015.yml)；执行产物：[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

## 2026-10-15 CDR上下文强度探索 初始状态确认

- task_id=`t-20261015131106-57bfk`；状态 `Queue`；记录时间 `2026-10-15T05:11:11.296703+00:00`；`Preemptible: false`。
- 配置：[cdr_context_contrast_probe_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_context_contrast_probe_20261015.yml)；执行产物：[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

## 2026-10-15 CDR上下文强度探索 终态

- task_id=`t-20261015131106-57bfk`；状态 `Success`；记录时间 `2026-10-15T05:12:43.846312+00:00`；`Preemptible: false`。
- 配置：[cdr_context_contrast_probe_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_context_contrast_probe_20261015.yml)；执行产物：[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

## 2026-10-15 CDR软embedding上下文探索 submit

- task_id=`t-20261015131452-mkndw`；状态 `提交成功，平台状态待查询`；记录时间 `2026-10-15T05:14:54.349438+00:00`；`Preemptible: false`。
- 配置：[cdr_soft_context_probe_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_soft_context_probe_20261015.yml)；执行产物：[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

## 2026-10-15 CDR软embedding上下文探索 初始状态确认

- task_id=`t-20261015131452-mkndw`；状态 `Queue`；记录时间 `2026-10-15T05:14:56.520331+00:00`；`Preemptible: false`。
- 配置：[cdr_soft_context_probe_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_soft_context_probe_20261015.yml)；执行产物：[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

## 2026-10-15 CDR新推理冻结独立确认 submit

- task_id=`t-20261015131845-fgtzf`；状态 `提交成功，平台状态待查询`；记录时间 `2026-10-15T05:18:47.779834+00:00`；`Preemptible: false`。
- 配置：[cdr_new_inference_confirmation_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_new_inference_confirmation_20261015.yml)；执行产物：[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

## 2026-10-15 CDR新推理冻结独立确认 初始状态确认

- task_id=`t-20261015131845-fgtzf`；状态 `Queue`；记录时间 `2026-10-15T05:18:50.498337+00:00`；`Preemptible: false`。
- 配置：[cdr_new_inference_confirmation_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_new_inference_confirmation_20261015.yml)；执行产物：[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

## 2026-10-15 CDR软embedding上下文探索 终态

- task_id=`t-20261015131452-mkndw`；状态 `Success`；记录时间 `2026-10-15T05:19:23.346335+00:00`；`Preemptible: false`。
- 配置：[cdr_soft_context_probe_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_soft_context_probe_20261015.yml)；执行产物：[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

## 2026-10-15 CDR新推理冻结独立确认 终态

- task_id=`t-20261015131845-fgtzf`；状态 `Success`；记录时间 `2026-10-15T05:38:28.105081+00:00`；`Preemptible: false`。
- 配置：[cdr_new_inference_confirmation_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_new_inference_confirmation_20261015.yml)；执行产物：[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。
# 2026-10-15 新V7恢复先前167/135槽位：完成

用户确认按先前约定缩短新V7固定链槽位。已发布 `/project_c20250601/251105016/project/dllm_test/data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_20261015`，manifest绑定独立 `immune_config.json`：H/β167、L/α135（含EOS），full_capacity304。原no-negative父版本中4条超长训练记录整条隔离并保存来源/原记录/原因；最终train6,249,956、valid68,985，共6,318,941。全量两遍检查保留字节与选择结果，源和输出分片哈希通过；168例真实reader检查和5项槽位边界/scaffold测试通过。原V5、旧V7父版本、模型默认值及已冻结训练均未修改；没有启动新训练。[规则、产物与验收](/project_c20250601/251105016/project/dllm_test/docs/V7_FROM_V5_REBUILD_20261015.md)。

## 2026-10-15 CDR分组软上下文开发探索 submit

- task_id=`t-20261015151244-jpxjq`；状态 `Initialized`；记录时间 `2026-10-15T07:12:46.570031+00:00`；`Preemptible: false`。
- 配置：[cdr_block_explore_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_explore_20261015.yml)；执行产物：[cdr_block_context_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_context_20261015)。

## 2026-10-15 CDR分组软上下文开发探索 初始状态确认

- task_id=`t-20261015151244-jpxjq`；状态 `Waiting`；记录时间 `2026-10-15T07:12:47.112248+00:00`；`Preemptible: false`。
- 配置：[cdr_block_explore_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_explore_20261015.yml)；执行产物：[cdr_block_context_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_context_20261015)。

## 2026-10-15 CDR分组软上下文开发探索 终态

- task_id=`t-20261015151244-jpxjq`；状态 `Success`；记录时间 `2026-10-15T07:15:33.711299+00:00`；`Preemptible: false`。
- 配置：[cdr_block_explore_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_explore_20261015.yml)；执行产物：[cdr_block_context_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_context_20261015)。

## 2026-10-15 CDR分组软上下文冻结OAS确认 submit

- task_id=`t-20261015151535-cj8pn`；状态 `Initialized`；记录时间 `2026-10-15T07:15:37.426156+00:00`；`Preemptible: false`。
- 配置：[cdr_block_confirm_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_confirm_20261015.yml)；执行产物：[cdr_block_context_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_context_20261015)。

## 2026-10-15 CDR分组软上下文冻结OAS确认 初始状态确认

- task_id=`t-20261015151535-cj8pn`；状态 `Staging`；记录时间 `2026-10-15T07:15:39.579143+00:00`；`Preemptible: false`。
- 配置：[cdr_block_confirm_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_confirm_20261015.yml)；执行产物：[cdr_block_context_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_context_20261015)。

## 2026-10-15 CDR分组软上下文冻结OAS确认 终态

- task_id=`t-20261015151535-cj8pn`；状态 `Success`；记录时间 `2026-10-15T07:25:44.960863+00:00`；`Preemptible: false`。
- 配置：[cdr_block_confirm_20261015.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_confirm_20261015.yml)；执行产物：[cdr_block_context_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_context_20261015)。

## 2026-10-16 OAS-only scratch 平台 Failed

操作查询；task_id=`t-20261015124455-xrns4`；任务名=`immune-fixed75m-oas-scratch-8gpu-queue012-20261015`；YAML=/project_c20250601/251105016/project/dllm_test/train_jobs/v5_fixed_chain_75m_oas_scratch_8gpu_queue012_20261015.yml；终态=`Failed`；End=2026-10-15T16:05:37Z；`Preemptible: false`。已从 Active 表删除。实际训练到 step 11845，最新完整保存 11000，最佳仍 6000（eval_loss=1.620522）。worker 日志：rank 3 NCCL BROADCAST 超时 1800s，随后 rank 5/7 SIGABRT。未自动重提。证据：[平台查询](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_queue012_20261015/platform_status.json)；[失败核对](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_queue012_20261015/failure_review.json)。

## 2026-10-16 CDR分组机制消融112000 submit

- task_id=`t-20261016144817-xtkvw`；状态 `Initialized`；记录时间 `2026-10-16T06:48:19.691598+00:00`；`Preemptible: false`。
- 配置：[cdr_block_ablation_112000_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_ablation_112000_20261016.yml)；执行产物：[cdr_block_ablation_20261016](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_ablation_20261016)。

## 2026-10-16 CDR分组机制消融50000 submit

- task_id=`t-20261016144828-rqwkp`；状态 `Initialized`；记录时间 `2026-10-16T06:48:30.542240+00:00`；`Preemptible: false`。
- 配置：[cdr_block_ablation_50000_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_ablation_50000_20261016.yml)；执行产物：[cdr_block_ablation_20261016](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_ablation_20261016)。

## 2026-10-16 CDR分组机制消融112000 终态

- task_id=`t-20261016144817-xtkvw`；状态 `Success`；记录时间 `2026-10-16T07:03:22.620112+00:00`；`Preemptible: false`。
- 配置：[cdr_block_ablation_112000_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_ablation_112000_20261016.yml)；执行产物：[cdr_block_ablation_20261016](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_ablation_20261016)。

## 2026-10-16 CDR分组机制消融50000 终态

- task_id=`t-20261016144828-rqwkp`；状态 `Success`；记录时间 `2026-10-16T07:03:23.219612+00:00`；`Preemptible: false`。
- 配置：[cdr_block_ablation_50000_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_ablation_50000_20261016.yml)；执行产物：[cdr_block_ablation_20261016](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_ablation_20261016)。

## 2026-10-16 CDR新方法4与8前向112000 submit

- task_id=`t-20261016152037-trxg7`；状态 `Initialized`；记录时间 `2026-10-16T07:20:40.080467+00:00`；`Preemptible: false`。
- 配置：[cdr_block_budget_112000_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_budget_112000_20261016.yml)；执行产物：[cdr_block_budget_20261016](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_budget_20261016)。

## 2026-10-16 CDR新方法4与8前向112000 初始状态确认

- task_id=`t-20261016152037-trxg7`；状态 `Queue`；记录时间 `2026-10-16T07:20:40.726909+00:00`；`Preemptible: false`。
- 配置：[cdr_block_budget_112000_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_budget_112000_20261016.yml)；执行产物：[cdr_block_budget_20261016](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_budget_20261016)。

## 2026-10-16 CDR新方法4与8前向50000 submit

- task_id=`t-20261016152042-bddll`；状态 `Initialized`；记录时间 `2026-10-16T07:20:44.337543+00:00`；`Preemptible: false`。
- 配置：[cdr_block_budget_50000_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_budget_50000_20261016.yml)；执行产物：[cdr_block_budget_20261016](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_budget_20261016)。

## 2026-10-16 CDR新方法4与8前向50000 初始状态确认

- task_id=`t-20261016152042-bddll`；状态 `Queue`；记录时间 `2026-10-16T07:20:44.923532+00:00`；`Preemptible: false`。
- 配置：[cdr_block_budget_50000_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_budget_50000_20261016.yml)；执行产物：[cdr_block_budget_20261016](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_budget_20261016)。

## 2026-10-16 CDR新方法4与8前向112000 终态

- task_id=`t-20261016152037-trxg7`；状态 `Failed`；记录时间 `2026-10-16T07:27:46.528610+00:00`；`Preemptible: false`。
- 配置：[cdr_block_budget_112000_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_budget_112000_20261016.yml)；执行产物：[cdr_block_budget_20261016](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_budget_20261016)。

## 2026-10-16 CDR新方法4与8前向50000 终态

- task_id=`t-20261016152042-bddll`；状态 `Failed`；记录时间 `2026-10-16T07:27:47.567938+00:00`；`Preemptible: false`。
- 配置：[cdr_block_budget_50000_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_block_budget_50000_20261016.yml)；执行产物：[cdr_block_budget_20261016](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_budget_20261016)。

## 2026-10-16 CDR清理误删依赖恢复 submit

- task_id=`t-20261016152832-lnqrj`；状态 `Staging`；记录时间 `2026-10-16T07:29:09.020672+00:00`；`Preemptible: false`。
- 配置：[cdr_prediction_restore_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_prediction_restore_20261016.yml)；执行产物：[sample_cleanup_20261016](/project_c20250601/251105016/project/dllm_test/output/sample_cleanup_20261016)。

## 2026-10-16 CDR清理误删依赖恢复 终态

- task_id=`t-20261016152832-lnqrj`；状态 `Success`；记录时间 `2026-10-16T07:34:56.881213+00:00`；`Preemptible: false`。
- 配置：[cdr_prediction_restore_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_prediction_restore_20261016.yml)；执行产物：[sample_cleanup_20261016](/project_c20250601/251105016/project/dllm_test/output/sample_cleanup_20261016)。

## 2026-10-16 CDR从左到右解码112000 submit

- task_id=`t-20261016155317-zlctk`；状态 `Initialized`；记录时间 `2026-10-16T07:53:20.308120+00:00`；`Preemptible: false`。
- 配置：[cdr_left_to_right_112000_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_left_to_right_112000_20261016.yml)；执行产物：[cdr_left_to_right_20261016](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_left_to_right_20261016)。

## 2026-10-16 CDR从左到右解码50000 submit

- task_id=`t-20261016155431-649zl`；状态 `Initialized`；记录时间 `2026-10-16T07:54:33.387718+00:00`；`Preemptible: false`。
- 配置：[cdr_left_to_right_50000_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_left_to_right_50000_20261016.yml)；执行产物：[cdr_left_to_right_20261016](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_left_to_right_20261016)。

## 2026-10-16 CDR从左到右解码50000 初始状态确认

- task_id=`t-20261016155431-649zl`；状态 `Queue`；记录时间 `2026-10-16T07:54:34.025364+00:00`；`Preemptible: false`。
- 配置：[cdr_left_to_right_50000_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_left_to_right_50000_20261016.yml)；执行产物：[cdr_left_to_right_20261016](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_left_to_right_20261016)。

## 2026-10-16 CDR从左到右解码112000 终态

- task_id=`t-20261016155317-zlctk`；状态 `Success`；记录时间 `2026-10-16T07:59:38.673327+00:00`；`Preemptible: false`。
- 配置：[cdr_left_to_right_112000_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_left_to_right_112000_20261016.yml)；执行产物：[cdr_left_to_right_20261016](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_left_to_right_20261016)。

## 2026-10-16 CDR从左到右解码50000 终态

- task_id=`t-20261016155431-649zl`；状态 `Success`；记录时间 `2026-10-16T08:00:49.576391+00:00`；`Preemptible: false`。
- 配置：[cdr_left_to_right_50000_20261016.yml](/project_c20250601/251105016/project/dllm_test/eval_jobs/cdr_left_to_right_50000_20261016.yml)；执行产物：[cdr_left_to_right_20261016](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_left_to_right_20261016)。

## 2026-10-16 提交 OAS-only 从11000续训

操作 submit；task_id=`t-20261016162026-4rq8w`；任务名=`immune-fixed75m-oas-resume11000-8gpu-queue012-20261016`；YAML=/project_c20250601/251105016/project/dllm_test/train_jobs/v5_fixed_chain_75m_oas_resume11000_8gpu_queue012_20261016.yml；初始状态=`Queue`；`Preemptible: false`；队列=`queue012`。从 checkpoint-11000 原生恢复，超参与冻结源码不变。W&B：当前 waitma 密钥、项目 `dllm-immune`、entity `models`、新 run `fixed-chain-75m-oas-resume-11000`，不设置旧 `WANDB_RUN_ID`。输出目录写探测通过。证据：[submission](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_resume11000_20261016/submission.json)，[平台查询](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_resume11000_20261016/platform_status.json)。

## 2026-10-17 BERT V7 best48828 全套表征评测提交

固定 best checkpoint48828（第4 epoch，内部开发集选点），独立复制推理权重并绑定训练源码。29项非生成评测分9个单GPU作业，队列c20250601，Preemptible=true；原八卡训练继续。真实CPU模型/标签隔离与全量输入容量检查通过，GPU门禁在每个作业开始时执行。

- submit g0；task_id=t-20261018005950-bh5mn；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g0_20261017.yml)；initial status=submitted_pending_platform_query；Preemptible=true；UTC=2026-10-17T16:59:49.460513+00:00；returncode=0。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/submit_g0.log)。
- submit g1；task_id=t-20261018005954-phvj8；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g1_20261017.yml)；initial status=submitted_pending_platform_query；Preemptible=true；UTC=2026-10-17T16:59:53.321356+00:00；returncode=0。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/submit_g1.log)。
- submit g2；task_id=t-20261018005958-hzjf7；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g2_20261017.yml)；initial status=submitted_pending_platform_query；Preemptible=true；UTC=2026-10-17T16:59:56.445399+00:00；returncode=0。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/submit_g2.log)。
- submit g3；task_id=t-20261018010001-t7skb；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g3_20261017.yml)；initial status=submitted_pending_platform_query；Preemptible=true；UTC=2026-10-17T17:00:00.194432+00:00；returncode=0。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/submit_g3.log)。
- submit g4；task_id=t-20261018010004-g5j2m；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g4_20261017.yml)；initial status=submitted_pending_platform_query；Preemptible=true；UTC=2026-10-17T17:00:03.553465+00:00；returncode=0。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/submit_g4.log)。
- submit g5；task_id=t-20261018010007-p9qpr；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g5_20261017.yml)；initial status=submitted_pending_platform_query；Preemptible=true；UTC=2026-10-17T17:00:06.816795+00:00；returncode=0。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/submit_g5.log)。
- submit g6；task_id=t-20261018010010-khsm6；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g6_20261017.yml)；initial status=submitted_pending_platform_query；Preemptible=true；UTC=2026-10-17T17:00:09.869688+00:00；returncode=0。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/submit_g6.log)。
- submit g7；task_id=t-20261018010014-mc9nd；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g7_20261017.yml)；initial status=submitted_pending_platform_query；Preemptible=true；UTC=2026-10-17T17:00:13.146758+00:00；returncode=0。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/submit_g7.log)。
- submit g8；task_id=t-20261018010018-chszf；YAML=[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g8_20261017.yml)；initial status=submitted_pending_platform_query；Preemptible=true；UTC=2026-10-17T17:00:17.009499+00:00；returncode=0。[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/submit_g8.log)。

2026-10-17T17:02:04.252580+00:00 提交后查询：{'Queue': 6, 'Running': 3}；g0–g2实际GPU门禁passed，Specificity及T1两个输入任务已启动，后续组排队。[平台查询](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/platform_status.json)、[任务摘要](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/summary.json)。


## 2026-10-17 submit LoRA block2 representation g0

2026-10-17T17:22:45.670954+00:00 提交 `post-lora-s42-layer2-g0-20261017`，task_id=`t-20261018012214-qlg28`，平台接受创建，初态待查询，Preemptible=true；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_layer2_g0_20261017.yml)。独立评测目录，不更改训练任务。


## 2026-10-17 submit LoRA block2 representation g1

2026-10-17T17:22:49.429635+00:00 提交 `post-lora-s42-layer2-g1-20261017`，task_id=`t-20261018012246-d8g85`，平台接受创建，初态待查询，Preemptible=true；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_layer2_g1_20261017.yml)。独立评测目录，不更改训练任务。


## 2026-10-17 submit LoRA block2 representation g2

2026-10-17T17:22:53.294560+00:00 提交 `post-lora-s42-layer2-g2-20261017`，task_id=`t-20261018012250-dxqrb`，平台接受创建，初态待查询，Preemptible=true；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_layer2_g2_20261017.yml)。独立评测目录，不更改训练任务。


## 2026-10-17 submit LoRA block2 representation g3

2026-10-17T17:22:57.142066+00:00 提交 `post-lora-s42-layer2-g3-20261017`，task_id=`t-20261018012254-hpfmt`，平台接受创建，初态待查询，Preemptible=true；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_layer2_g3_20261017.yml)。独立评测目录，不更改训练任务。


## 2026-10-17 submit LoRA block2 representation g4

2026-10-17T17:23:00.590844+00:00 提交 `post-lora-s42-layer2-g4-20261017`，task_id=`t-20261018012258-t6cqd`，平台接受创建，初态待查询，Preemptible=true；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_layer2_g4_20261017.yml)。独立评测目录，不更改训练任务。


## 2026-10-17 submit LoRA block2 representation g5

2026-10-17T17:23:04.218634+00:00 提交 `post-lora-s42-layer2-g5-20261017`，task_id=`t-20261018012301-dlnf7`，平台接受创建，初态待查询，Preemptible=true；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_layer2_g5_20261017.yml)。独立评测目录，不更改训练任务。


## 2026-10-17 submit LoRA block2 representation g6

2026-10-17T17:23:08.088915+00:00 提交 `post-lora-s42-layer2-g6-20261017`，task_id=`t-20261018012305-7pxmw`，平台接受创建，初态待查询，Preemptible=true；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_layer2_g6_20261017.yml)。独立评测目录，不更改训练任务。


## 2026-10-17 LoRA 第二层作业提交后核实

2026-10-17T17:24:14.402378+00:00 七组全部平台读回成功，状态{'Queue': 7}；[逐任务初态](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_layer2_20261017/platform_status.json)。CPU门禁passed，GPU门禁和任务结果以实际产物为准。


## 2026-10-17 cancel LoRA block2 g0：用户改为本地

2026-10-17T17:25:35.568176+00:00 `post-lora-s42-layer2-g0-20261017`，`t-20261018012214-qlg28`，Queue → Queue；Preemptible=true；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_layer2_g0_20261017.yml)；[操作与查询](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_layer2_20261017/local_migration_platform.json)。仅撤销本轮第二层任务。


## 2026-10-17 cancel LoRA block2 g1：用户改为本地

2026-10-17T17:25:37.082408+00:00 `post-lora-s42-layer2-g1-20261017`，`t-20261018012246-d8g85`，Queue → Queue；Preemptible=true；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_layer2_g1_20261017.yml)；[操作与查询](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_layer2_20261017/local_migration_platform.json)。仅撤销本轮第二层任务。


## 2026-10-17 cancel LoRA block2 g2：用户改为本地

2026-10-17T17:25:38.902596+00:00 `post-lora-s42-layer2-g2-20261017`，`t-20261018012250-dxqrb`，Queue → Queue；Preemptible=true；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_layer2_g2_20261017.yml)；[操作与查询](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_layer2_20261017/local_migration_platform.json)。仅撤销本轮第二层任务。


## 2026-10-17 cancel LoRA block2 g3：用户改为本地

2026-10-17T17:25:40.616150+00:00 `post-lora-s42-layer2-g3-20261017`，`t-20261018012254-hpfmt`，Queue → Queue；Preemptible=true；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_layer2_g3_20261017.yml)；[操作与查询](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_layer2_20261017/local_migration_platform.json)。仅撤销本轮第二层任务。


## 2026-10-17 cancel LoRA block2 g4：用户改为本地

2026-10-17T17:25:42.294787+00:00 `post-lora-s42-layer2-g4-20261017`，`t-20261018012258-t6cqd`，Queue → Queue；Preemptible=true；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_layer2_g4_20261017.yml)；[操作与查询](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_layer2_20261017/local_migration_platform.json)。仅撤销本轮第二层任务。


## 2026-10-17 cancel LoRA block2 g5：用户改为本地

2026-10-17T17:25:44.112156+00:00 `post-lora-s42-layer2-g5-20261017`，`t-20261018012301-dlnf7`，Queue → Queue；Preemptible=true；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_layer2_g5_20261017.yml)；[操作与查询](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_layer2_20261017/local_migration_platform.json)。仅撤销本轮第二层任务。


## 2026-10-17 cancel LoRA block2 g6：用户改为本地

2026-10-17T17:25:45.603352+00:00 `post-lora-s42-layer2-g6-20261017`，`t-20261018012305-7pxmw`，Queue → Queue；Preemptible=true；[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/post_training_lora_s42_layer2_g6_20261017.yml)；[操作与查询](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_layer2_20261017/local_migration_platform.json)。仅撤销本轮第二层任务。


## 2026-10-17 第二层远端作业全部取消完成

2026-10-17T17:26:36.771211+00:00 七个本轮作业均核实为Killed，已从Active表移除；用户指定改用本地A100。[终态证据](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_layer2_20261017/local_platform_terminal.json)。


## 2026-10-17 本地启动第二层表征评测

2026-10-17T17:27:16.844837+00:00 用户明确要求本地执行，优先于默认Volc评测规则。本地A100 80GB、CUDA设备0、单组顺序运行，先g0再g4/2/1/3/5/6；[执行绑定](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_layer2_20261017/local_execution.json)、[本地driver](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/post_training_lora_s42_layer2_20261017/local_driver.json)。不修改既有权重、任务配置或其他进程。

## 2026-10-18 BERT best48828 下游状态核实与收数

2026-10-18T01:05:38.018505+00:00 只读平台查询确认2作业Success、4 Killed、3 Queue；Killed发生在2026-10-17T17:20:36–17:20:37Z，syslog显示同时结束，但未给出终止原因，不擅自标为模型异常或资源抢占。10项完整任务可收数，UpTCR的majority/few-shot独立完成；其他未完成项不填完整成绩。本次无submit/cancel；终态移出Active，原提交YAML/Preemptible=true保留在历史行。[查询](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/platform_status.json)、[状态更正](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/review_20261018/platform_reconciliation.json)。

- g0：t-20261018005950-bh5mn，Success；结束UTC=2026-10-17T17:12:19Z；Preemptible=true。[原YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g0_20261017.yml)。
- g1：t-20261018005954-phvj8，Killed；结束UTC=2026-10-17T17:20:37Z；Preemptible=true。[原YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g1_20261017.yml)。
- g2：t-20261018005958-hzjf7，Success；结束UTC=2026-10-17T17:04:40Z；Preemptible=true。[原YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g2_20261017.yml)。
- g3：t-20261018010001-t7skb，Killed；结束UTC=2026-10-17T17:20:36Z；Preemptible=true。[原YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g3_20261017.yml)。
- g4：t-20261018010004-g5j2m，Killed；结束UTC=2026-10-17T17:20:36Z；Preemptible=true。[原YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g4_20261017.yml)。
- g5：t-20261018010007-p9qpr，Killed；结束UTC=2026-10-17T17:20:37Z；Preemptible=true。[原YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g5_20261017.yml)。
- g6：t-20261018010010-khsm6，Queue；结束UTC=None；Preemptible=true。[原YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g6_20261017.yml)。
- g7：t-20261018010014-mc9nd，Queue；结束UTC=None；Preemptible=true。[原YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g7_20261017.yml)。
- g8：t-20261018010018-chszf，Queue；结束UTC=None；Preemptible=true。[原YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g8_20261017.yml)。

2026-10-18 本轮完成项保存预测重算/聚合核查passed，已回填RESULTS与RESULTS_SUMMARY；未新增推理作业或改变正式协议。[验收](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/review_20261018/verified_results.json)。

## 2026-10-18 BERT best48828 g6 迁移本地：cancel

- UTC=2026-10-18T01:16:33.937734+00:00；task_id=t-20261018010010-khsm6；任务名=bert-full-token-75m-best48828-g6-20261017；原状态=Queue；cancel exit=0；Preemptible=true。用户授权排队/失败评测迁移本机。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g6_20261017.yml)；[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/local_20261018/cancel_g6.log)。

## 2026-10-18 BERT best48828 g7 迁移本地：cancel

- UTC=2026-10-18T01:16:34.282624+00:00；task_id=t-20261018010014-mc9nd；任务名=bert-full-token-75m-best48828-g7-20261017；原状态=Queue；cancel exit=0；Preemptible=true。用户授权排队/失败评测迁移本机。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g7_20261017.yml)；[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/local_20261018/cancel_g7.log)。

## 2026-10-18 BERT best48828 g8 迁移本地：cancel

- UTC=2026-10-18T01:16:34.680958+00:00；task_id=t-20261018010018-chszf；任务名=bert-full-token-75m-best48828-g8-20261017；原状态=Queue；cancel exit=0；Preemptible=true。用户授权排队/失败评测迁移本机。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_best48828_g8_20261017.yml)；[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/local_20261018/cancel_g8.log)。

迁移前平台复核：t-20261018010018-chszf=Queue；t-20261018010014-mc9nd=Queue；t-20261018010010-khsm6=Queue。

BERT 本地迁移终态确认：原9组全部终态（2 Success、7 Killed），其中本次cancel的g6/g7/g8均已Killed。19项本地执行清单及旧缓存身份校验完成，待GPU检查及启动。[平台终态](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/local_20261018/platform_terminal.json)；[迁移绑定](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_best48828_20261017/local_20261018/manifest.json)。


## 2026-10-18 本地启动fixed50000第二层评测

2026-10-18T01:22:26.355721+00:00 用户指定模型，CPU门禁通过后启动本地driver PID 1951564；原12项协议及该模型末层基线保持，无平台提交。[启动证据](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_layer2_20261018/local_execution.json)。

## 2026-10-18 BERT step107000 g0 submit

- UTC=2026-10-18T01:57:00.191611+00:00；task_id=t-20261018095701-ptw5t；任务名=bert-full-token-75m-step107000-g0-20261018；初始状态=submitted_pending_platform_query；Preemptible=true；queue=c20250601；submit exit=0。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step107000_g0_20261018.yml)；[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step107000_20261018/submit_g0.log)。固定用户指定107000独立快照，沿用48828同版表征协议。

## 2026-10-18 BERT step107000 g1 submit

- UTC=2026-10-18T01:57:03.446284+00:00；task_id=t-20261018095704-g62kk；任务名=bert-full-token-75m-step107000-g1-20261018；初始状态=submitted_pending_platform_query；Preemptible=true；queue=c20250601；submit exit=0。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step107000_g1_20261018.yml)；[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step107000_20261018/submit_g1.log)。固定用户指定107000独立快照，沿用48828同版表征协议。

## 2026-10-18 BERT step107000 g2 submit

- UTC=2026-10-18T01:57:06.800912+00:00；task_id=t-20261018095708-g2c9q；任务名=bert-full-token-75m-step107000-g2-20261018；初始状态=submitted_pending_platform_query；Preemptible=true；queue=c20250601；submit exit=0。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step107000_g2_20261018.yml)；[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step107000_20261018/submit_g2.log)。固定用户指定107000独立快照，沿用48828同版表征协议。

## 2026-10-18 BERT step107000 g3 submit

- UTC=2026-10-18T01:57:10.304674+00:00；task_id=t-20261018095711-695gc；任务名=bert-full-token-75m-step107000-g3-20261018；初始状态=submitted_pending_platform_query；Preemptible=true；queue=c20250601；submit exit=0。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step107000_g3_20261018.yml)；[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step107000_20261018/submit_g3.log)。固定用户指定107000独立快照，沿用48828同版表征协议。

## 2026-10-18 BERT step107000 g4 submit

- UTC=2026-10-18T01:57:13.444054+00:00；task_id=t-20261018095714-gdq7v；任务名=bert-full-token-75m-step107000-g4-20261018；初始状态=submitted_pending_platform_query；Preemptible=true；queue=c20250601；submit exit=0。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step107000_g4_20261018.yml)；[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step107000_20261018/submit_g4.log)。固定用户指定107000独立快照，沿用48828同版表征协议。

## 2026-10-18 BERT step107000 g0 cancel：转本地

- UTC=2026-10-18T02:02:49.631919+00:00；task_id=t-20261018095701-ptw5t；任务名=bert-full-token-75m-step107000-g0-20261018；原状态=Queue；cancel exit=0；Preemptible=true。用户明确要求本机执行。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step107000_g0_20261018.yml)；[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step107000_20261018/local_20261018/cancel_g0.log)。

## 2026-10-18 BERT step107000 g1 cancel：转本地

- UTC=2026-10-18T02:02:49.938212+00:00；task_id=t-20261018095704-g62kk；任务名=bert-full-token-75m-step107000-g1-20261018；原状态=Queue；cancel exit=0；Preemptible=true。用户明确要求本机执行。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step107000_g1_20261018.yml)；[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step107000_20261018/local_20261018/cancel_g1.log)。

## 2026-10-18 BERT step107000 g2 cancel：转本地

- UTC=2026-10-18T02:02:50.319390+00:00；task_id=t-20261018095708-g2c9q；任务名=bert-full-token-75m-step107000-g2-20261018；原状态=Queue；cancel exit=0；Preemptible=true。用户明确要求本机执行。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step107000_g2_20261018.yml)；[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step107000_20261018/local_20261018/cancel_g2.log)。

## 2026-10-18 BERT step107000 g3 cancel：转本地

- UTC=2026-10-18T02:02:50.664312+00:00；task_id=t-20261018095711-695gc；任务名=bert-full-token-75m-step107000-g3-20261018；原状态=Queue；cancel exit=0；Preemptible=true。用户明确要求本机执行。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step107000_g3_20261018.yml)；[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step107000_20261018/local_20261018/cancel_g3.log)。

## 2026-10-18 BERT step107000 g4 cancel：转本地

- UTC=2026-10-18T02:02:51.303914+00:00；task_id=t-20261018095714-gdq7v；任务名=bert-full-token-75m-step107000-g4-20261018；原状态=Queue；cancel exit=0；Preemptible=true。用户明确要求本机执行。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step107000_g4_20261018.yml)；[回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step107000_20261018/local_20261018/cancel_g4.log)。

2026-10-18T02:06:11.473520+00:00 本轮step107000迁移终态复核：5组云端任务全部Killed，从Active表移除；本地启动前不再有重复平台任务。[终态](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step107000_20261018/local_20261018/platform_terminal.json)。

## 2026-10-18 OAS-only 两次续训失败根因更正

2026-10-18T18:05:38.009014+00:00 查询确认 r2 `t-20261017122656-nzn5w` 于2026-10-18T04:26:08Z Failed，已移出Active表。r1/r2完整可取平台日志均显示 rank0 在 `WandbCallback.setup` 调用 `model.config.to_dict()` 时抛出 `AttributeError: 'FixedChainConfig' object has no attribute 'to_dict'`；其他七rank的Gloo RNG集合通信1800s超时为继发错误。原scratch关闭W&B，续训launcher开启后触发；更换entity/runtime没有修复接口。两次续训新增0步，11845为原scratch历史记录，完整恢复点仍11000。本轮只诊断并同步证据，未改训练源码或提交任务。[原始日志](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_oas_300k_b512_20261015_supervision/review_20261018/r2_platform_failure.log:633)；[诊断](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_oas_300k_b512_20261015_supervision/review_20261018/failure_review.json)。

## 2026-10-18 OAS-only 续训 W&B 配置序列化修复

2026-10-18T18:09:27.498029+00:00 已在 [launch_oas_resume.py](/project_c20250601/251105016/project/dllm_test/scripts/research/launch_oas_resume.py) 为实际导入的冻结 `FixedChainConfig` 安装 `to_dict=dataclasses.asdict` 运行时适配；安装前校验源码路径，启动回执记录源码SHA，`--check-only`覆盖适配和已导入模块的源码清单。原冻结源码文件和模型字段未改。两项真实W&B离线回归分别针对工作区和冻结源码，均先复现原异常，再通过on_train_begin/on_log/on_train_end，包含错误源码拒绝、重复安装、CPU隔离检查。checkpoint-11000全部17份文件SHA匹配，r3新runtime/run名及launcher/spec哈希、YAML与bash语法通过。r1/r2旧spec绑定旧launcher哈希，不再使用；r3沿用300k/global512、原数据和模型。本轮修复及准备完成，未提交/取消GPU任务，分布式续训尚未验证。[检查证据](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_resume11000_20261018/verification.json)；[新配置](/project_c20250601/251105016/project/dllm_test/train_jobs/v5_fixed_chain_75m_oas_resume11000_8gpu_queue012_20261018.yml)。

## 2026-10-18 OAS-only11000 cdr 非闲时评测 submit

2026-10-18T18:21:15.142193+00:00 操作submit；task_id=`t-20261019022112-54lqt`；任务名=`oas-only-75m-step11000-cdr-20261018`；初始状态`Queue`；Preemptible=false；c20250601 / 1×ml.pni2.3xlarge。用户明确授权非闲时资源；固定最新完整11000快照。CPU九CDR输入/前向、500条配对输入、实际采样器对齐及3项回归通过。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/oas_only_75m_step11000_cdr_20261018.yml)；[提交回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/oas_only_75m_step11000_generation_20261018/cdr_submission.json)；[协议与执行](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#oas-only-generation-20261018)。

## 2026-10-18 OAS-only11000 pairing 非闲时评测 submit

2026-10-18T18:21:19.183618+00:00 操作submit；task_id=`t-20261019022116-x85nt`；任务名=`oas-only-75m-step11000-pairing-20261018`；提交成功，首次即时get未返回JSON，随后list确认`Running`；Preemptible=false；c20250601 / 1×ml.pni2.3xlarge。未重复提交。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/oas_only_75m_step11000_pairing_20261018.yml)；[提交回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/oas_only_75m_step11000_generation_20261018/pairing_submission.json)；[协议与执行](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#oas-only-generation-20261018)。

## 2026-10-18 OAS-only11000 CDR评测 Success

2026-10-18T18:45:31.386352+00:00 查询确认`t-20261019022112-54lqt`于2026-10-18T18:24:08Z Success；非闲时单卡，已移出Active表。Kong9381与SAb23H2共360条区域预测独立核验通过，已回填对应原表。配对任务4000候选生成完成，仍在详细评分。[核验](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/oas_only_75m_step11000_generation_20261018/cdr/verification.json)；[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr)。

## 2026-10-18 OAS-only11000生成评测全部完成

2026-10-18T19:01:54.193826+00:00 配对任务`t-20261019022116-x85nt`于2026-10-18T18:50:35Z Success，已移出Active表；CDR任务此前Success。CDR全部9741条区域预测及配对4000候选均核验通过，aligned采样器一致，指标已回填原任务表。本轮没有新提交或取消。[完整核验](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/oas_only_75m_step11000_generation_20261018/latest_review.json)；[结果表](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr)。

## 2026-10-19 BERT step247000 下游 g0 submit

- 任务：`bert-full-token-75m-step247000-g0-20261019`；ID `t-20261019162448-mhdkn`；初始状态 `Queue`；Preemptible: true。
- YAML：[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step247000_g0_20261019.yml)；提交记录：[平台记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step247000_20261019/platform_g0.txt)。

## 2026-10-19 BERT step247000 下游 g1 submit

- 任务：`bert-full-token-75m-step247000-g1-20261019`；ID `t-20261019162521-hdzwx`；初始状态 `Queue`；Preemptible: true。
- YAML：[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step247000_g1_20261019.yml)；提交记录：[平台记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step247000_20261019/platform_g1.txt)。

## 2026-10-19 BERT step247000 下游 g2 submit

- 任务：`bert-full-token-75m-step247000-g2-20261019`；ID `t-20261019162525-nsf6k`；初始状态 `Queue`；Preemptible: true。
- YAML：[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step247000_g2_20261019.yml)；提交记录：[平台记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step247000_20261019/platform_g2.txt)。

## 2026-10-19 BERT step247000 下游 g3 submit

- 任务：`bert-full-token-75m-step247000-g3-20261019`；ID `t-20261019162529-469fs`；初始状态 `Queue`；Preemptible: true。
- YAML：[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step247000_g3_20261019.yml)；提交记录：[平台记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step247000_20261019/platform_g3.txt)。

## 2026-10-19 BERT step247000 下游 g4 submit

- 任务：`bert-full-token-75m-step247000-g4-20261019`；ID `t-20261019162602-v2vxt`；初始状态 `Submitted（平台状态尚未可见）`；Preemptible: true。
- YAML：[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step247000_g4_20261019.yml)；提交记录：[平台记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step247000_20261019/platform_g4.txt)。

## 2026-10-19 BERT step247000 g0 cancel

- 用户要求切本地，cancel `t-20261019162448-mhdkn`（bert-full-token-75m-step247000-g0-20261019）；操作已受理，终态待查；Preemptible: true。YAML：[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step247000_g0_20261019.yml)；[取消回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step247000_20261019/cancel_g0.txt)。

## 2026-10-19 BERT step247000 g1 cancel

- 用户要求切本地，cancel `t-20261019162521-hdzwx`（bert-full-token-75m-step247000-g1-20261019）；操作已受理，终态待查；Preemptible: true。YAML：[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step247000_g1_20261019.yml)；[取消回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step247000_20261019/cancel_g1.txt)。

## 2026-10-19 BERT step247000 g2 cancel

- 用户要求切本地，cancel `t-20261019162525-nsf6k`（bert-full-token-75m-step247000-g2-20261019）；操作已受理，终态待查；Preemptible: true。YAML：[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step247000_g2_20261019.yml)；[取消回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step247000_20261019/cancel_g2.txt)。

## 2026-10-19 BERT step247000 g3 cancel

- 用户要求切本地，cancel `t-20261019162529-469fs`（bert-full-token-75m-step247000-g3-20261019）；操作已受理，终态待查；Preemptible: true。YAML：[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step247000_g3_20261019.yml)；[取消回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step247000_20261019/cancel_g3.txt)。

## 2026-10-19 BERT step247000 g4 cancel

- 用户要求切本地，cancel `t-20261019162602-v2vxt`（bert-full-token-75m-step247000-g4-20261019）；操作已受理，终态待查；Preemptible: true。YAML：[配置](/project_c20250601/251105016/project/dllm_test/eval_jobs/bert_full_token_step247000_g4_20261019.yml)；[取消回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step247000_20261019/cancel_g4.txt)。

### 2026-10-19T13:40:07.994159+00:00 step247000 云评测终态核实

五组均已终态，已从Active表移除，改用本地执行；[平台终态](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/bert_full_token_75m_step247000_20261019/local_20261019/platform_terminal.json)。

## 2026-10-20 diffusion/edit16500 下游 g0 非闲时 submit

2026-10-20T08:27:28.142788+00:00 操作submit；task_id=`t-20261020162723-wprhh`；任务名=`diffedit-step16500-g0-20261020`；初始查询状态`Queue`；Preemptible=false；c20250601 / 单卡ml.pni2.3xlarge。用户明确授权非闲时评测，复用原冻结checkpoint与协议，仅执行pairing。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/diffusion_edit_step16500_g0_nonpreempt_20261020.yml)、[提交回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/g0/submission.txt)、[作业记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/jobs.json)。

## 2026-10-20 diffusion/edit16500 下游 g1 非闲时 submit

2026-10-20T08:27:34.589442+00:00 操作submit；task_id=`t-20261020162730-7bph4`；任务名=`diffedit-step16500-g1-20261020`；初始查询状态`Queue`；Preemptible=false；c20250601 / 单卡ml.pni2.3xlarge。用户明确授权非闲时评测，复用原冻结checkpoint与协议，仅执行tcr-benchmark14, tcr-held20。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/diffusion_edit_step16500_g1_nonpreempt_20261020.yml)、[提交回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/g1/submission.txt)、[作业记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/jobs.json)。

## 2026-10-20 diffusion/edit16500 下游 g2 非闲时 submit

2026-10-20T08:27:38.850823+00:00 操作submit；task_id=`t-20261020162735-fh4r4`；任务名=`diffedit-step16500-g2-20261020`；初始查询状态`Queue`；Preemptible=false；c20250601 / 单卡ml.pni2.3xlarge。用户明确授权非闲时评测，复用原冻结checkpoint与协议，仅执行t1-cdr3b。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/diffusion_edit_step16500_g2_nonpreempt_20261020.yml)、[提交回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/g2/submission.txt)、[作业记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/jobs.json)。

## 2026-10-20 diffusion/edit16500 下游 g3 非闲时 submit

2026-10-20T08:27:43.167354+00:00 操作submit；task_id=`t-20261020162740-jfhhp`；任务名=`diffedit-step16500-g3-20261020`；初始查询状态`Waiting`；Preemptible=false；c20250601 / 单卡ml.pni2.3xlarge。用户明确授权非闲时评测，复用原冻结checkpoint与协议，仅执行t1-others_cdr3b。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/diffusion_edit_step16500_g3_nonpreempt_20261020.yml)、[提交回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/g3/submission.txt)、[作业记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/jobs.json)。

## 2026-10-20 diffusion/edit16500 下游 g4 非闲时 submit

2026-10-20T08:27:49.068895+00:00 操作submit；task_id=`t-20261020162744-mhjb4`；任务名=`diffedit-step16500-g4-20261020`；初始查询状态`Queue`；Preemptible=false；c20250601 / 单卡ml.pni2.3xlarge。用户明确授权非闲时评测，复用原冻结checkpoint与协议，仅执行t1-cdr3ab, t1-others_longab。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/diffusion_edit_step16500_g4_nonpreempt_20261020.yml)、[提交回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/g4/submission.txt)、[作业记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/jobs.json)。

## 2026-10-20 diffusion/edit16500 下游 g5 非闲时 submit

2026-10-20T08:27:55.010133+00:00 操作submit；task_id=`t-20261020162750-58znw`；任务名=`diffedit-step16500-g5-20261020`；初始查询状态`Queue`；Preemptible=false；c20250601 / 单卡ml.pni2.3xlarge。用户明确授权非闲时评测，复用原冻结checkpoint与协议，仅执行uptcr。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/diffusion_edit_step16500_g5_nonpreempt_20261020.yml)、[提交回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/g5/submission.txt)、[作业记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/jobs.json)。

## 2026-10-20 diffusion/edit16500 g3 平台终态

2026-10-20T08:31:08.770186+00:00 查询`t-20261020162740-jfhhp`（`diffedit-step16500-g3-20261020`）进入`Success`，已移出Active表；Preemptible=false。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/diffusion_edit_step16500_g3_nonpreempt_20261020.yml)、[平台记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/g3/latest_platform.txt)。本次未提交或取消任务。

## 2026-10-20 diffusion/edit16500 g4 平台终态

2026-10-20T08:31:42.488580+00:00 查询`t-20261020162744-mhjb4`（`diffedit-step16500-g4-20261020`）进入`Success`，已移出Active表；Preemptible=false。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/diffusion_edit_step16500_g4_nonpreempt_20261020.yml)、[平台记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/g4/latest_platform.txt)。本次未提交或取消任务。

## 2026-10-20 diffusion/edit16500 g5 平台终态

2026-10-20T08:45:48.563705+00:00 查询`t-20261020162750-58znw`（`diffedit-step16500-g5-20261020`）进入`Success`，已移出Active表；Preemptible=false。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/diffusion_edit_step16500_g5_nonpreempt_20261020.yml)、[平台记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/g5/latest_platform.txt)。本次未提交或取消任务。

## 2026-10-20 diffusion/edit16500 g1 平台终态

2026-10-20T08:49:00.677534+00:00 查询`t-20261020162730-7bph4`（`diffedit-step16500-g1-20261020`）进入`Success`，已移出Active表；Preemptible=false。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/diffusion_edit_step16500_g1_nonpreempt_20261020.yml)、[平台记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/g1/latest_platform.txt)。本次未提交或取消任务。

## 2026-10-20 diffusion/edit16500 g2 平台终态

2026-10-20T08:55:47.735431+00:00 查询`t-20261020162735-fh4r4`（`diffedit-step16500-g2-20261020`）进入`Success`，已移出Active表；Preemptible=false。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/diffusion_edit_step16500_g2_nonpreempt_20261020.yml)、[平台记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/g2/latest_platform.txt)。本次未提交或取消任务。

## 2026-10-20 diffusion/edit16500 g0 平台终态

2026-10-20T08:56:18.791328+00:00 查询`t-20261020162723-wprhh`（`diffedit-step16500-g0-20261020`）进入`Success`，已移出Active表；Preemptible=false。[YAML](/project_c20250601/251105016/project/dllm_test/eval_jobs/diffusion_edit_step16500_g0_nonpreempt_20261020.yml)、[平台记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_75m_step16500_20261020/cloud/g0/latest_platform.txt)。本次未提交或取消任务。

## 2026-10-20 旧八卡容器终态与恢复准备

2026-10-20T10:56:34.663527+00:00 查询旧容器`t-20261007153431-vhx2f`已于2026-10-20T10:11:49Z进入Success（容器exit0），已移出Active表；训练票据为interrupted/returncode1，并未完成20万步。最后日志20800，最新完整保存20500。按用户明确要求准备新八卡非闲时常驻容器，训练退出恢复standby。[平台终态](/project_c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261020/previous_terminal.json)、[旧训练票据](/project_c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261007/results/diffusion_edit287000_extend200k_train_20261020.json)。

## 2026-10-20 八卡常驻容器重新申请 submit

2026-10-20T10:57:06.076523+00:00 操作submit；任务`immune-diffedit-devpool-8gpu-20261020`；task_id=`t-20261020185703-8f7th`；初始状态`Queue`；Preemptible=false；c20250601 / 1×ml.pni2.28xlarge（8×A10080）。复用原占卡与训练切换监督器，CPU失败隔离/17个子进程回收检查通过，训练退出后恢复8个standby。[YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/immune_training_container_8gpu_20261020.yml)、[回执](/project_c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261020/submission.json)。

## 2026-10-20 diffusion/edit20500 完整状态恢复入队

2026-10-20T11:00:33.241582+00:00 在新容器`t-20261020185703-8f7th`中仅发布一次恢复票据；原容器已终态，原训练中断。新容器8个standby初始化通过，监督器会在DDP前全部停止回收。原训练数据/模型/配方/20万目标/学习率schedule/W&B run均保留，恢复AdamW、scheduler和八rank RNG；未落盘的20501至20800步重算。[训练票据](/project_c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261020/training_enqueue.json)、[启动包](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_diffusion_edit287000_resume20500_20261020/plan.json)、[完整校验](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/restart_20261020/preparation.json)。

## 2026-10-20 八卡容器恢复与首个保存验收

2026-10-20T11:04:02.313874+00:00 新容器`t-20261020185703-8f7th`平台Running，真实训练已至20575/200000；八rank完整恢复与学习率连续性通过，checkpoint20525的16份文件SHA一致、八rank参数一致且权重相对20500已更新。沿用W&B `diffedit287k200k1020`；重算未落盘步骤时W&B会跳过较旧step，主训练正常。监督器保持训练退出恢复八卡standby的原行为。[验收](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/restart_20261020/startup_verification.json)。


## 2026-10-22 diffusion/edit 闲时续训准备与凭据阻塞

2026-10-22T07:25:48.353897+00:00 当前diffusion/edit已中断：旧容器本地state=exited（2026-10-20T15:24:04Z），最后训练日志28225，最新完整checkpoint28000；不能把残留status=training当作仍在运行。用户要求闲时续训至累计200000；新8×A10080、Preemptible=true配置与完整状态恢复包已准备，16个checkpoint文件／874源码／1567 vendor文件哈希及8项CPU测试通过。当前Volc凭据缺失，平台查询失败，尚未submit、无新task_id；旧任务平台终态待凭据恢复后核实。本地只读监督每60秒更新，等待提交回执后监控云任务。[闲时YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_diffusion_edit_resume28000_spot_20261022.yml)、[准备验收](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/spot_20261022/preparation.json)、[当前监督](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/spot_20261022/monitor_status.json)、[阻塞](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/spot_20261022/submission_blocked.json)。

恢复原模型／AdamW／scheduler／8-rank RNG和累计采样游标；目标200000、global512记录=1024视图、每卡48+48+32及原W&B run不变。保留独立28000快照；实例回收最多20次平台重试，仅InstanceReclaimed，启动时重新选择最新完整保存点；训练结束释放资源。监控仅观察，不提交／取消／自动重提平台任务，停滞只记录告警。停止监控使用 `/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/spot_20261022/STOP_MONITOR`；保存后停训使用原训练目录的STOP。初次启动前仍须核实平台无同run活跃任务，当前未完成GPU实际恢复验收。


## 2026-10-22 diffusion/edit 闲时续训 submit

2026-10-22T08:04:21.223840+00:00 已提交闲时续训任务`t-20261022160418-jwww4`；1×ml.pni2.28xlarge（8×A10080）、Preemptible=true、c20250601。从完整28000恢复模型／AdamW／scheduler／8-rank RNG，目标累计200000；原冻结源码、数据、batch及W&B run沿用。平台状态正在回读，尚未宣称实际开始训练。每60秒只读监控已运行。[YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_diffusion_edit_resume28000_spot_20261022.yml)、[回执](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/spot_20261022/submission.json)、[实时监督](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/spot_20261022/monitor_status.json)。

操作submit；任务名`bert-diffedit-resume28000-200k-spot-20261022`；初始状态提交成功待平台回读。旧训练容器`t-20261020185703-8f7th`平台Success（2026-10-20T15:24:20Z），旧OAS r3`t-20261019021523-8psvh`平台Failed（2026-10-20T08:58:30Z），已移出Active表。旧容器Success只表示容器正常退出，diffusion训练未达到20万步。[提交前平台核对](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/spot_20261022/pre_submit_platform.json)。


## 2026-10-22 diffusion/edit 闲时任务状态变化

2026-10-22T08:04:51.074361+00:00 task_id=`t-20261022160418-jwww4`；平台`Queue`；训练观察`queue`；最后日志28225，最新完整28000；Preemptible=true。[监控](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/spot_20261022/monitor_status.json)。


## 2026-10-22 闲时队列剩余资源核查

2026-10-22T08:21:00.739831+00:00 闲时续训`t-20261022160418-jwww4`仍Queueing。API实查本队列c20250601 GPU配额0；同资源组可出借且Running的队列为xts001_temp（24配额／19已分配／SharedQuotaAllocated=5）和devinstance-queue3（32／32／0）。仅按配额减已分配计算的余量上限为5张，小于单节点8卡需求；尚未取得节点级物理空闲卡数。平台任务提示“当前队列中有配额，可能存在碎片问题”是调度诊断，不能替代上述配额统计。[查询证据](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/spot_20261022/queue_diagnostic.json)。


## 2026-10-22 非闲时续训切换准备

用户要求改为非闲时申请；已准备独立控制目录 /project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022，复用原完整状态恢复器，监控状态按实际Preemptible区分。训练配方与200000目标不变。尚未取消旧任务或提交新任务，正在核对用户补充的队列选择。


## 2026-10-22 diffusion/edit 闲时任务状态变化

2026-10-22T08:28:22.389297+00:00 task_id=`t-20261022160418-jwww4`；平台`Queue`；训练观察`queue`；最后日志28225，最新完整28000；Preemptible=true。[监控](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/spot_20261022/monitor_status.json)。


## 2026-10-22 cancel 闲时任务，改投用户指定非闲时队列

2026-10-22T08:28:22.744607+00:00 cancel `t-20261022160418-jwww4`（`bert-diffedit-resume28000-200k-spot-20261022`），回读状态 `Queue`，Preemptible=true；原配置 `/project_c20250601/251105016/project/dllm_test/train_jobs/bert_diffusion_edit_resume28000_spot_20261022.yml`。用户明确改用 `ZENITH_MLP2_TASK`（`q-20261009101456-4m9np`），下一次提交使用 Preemptible=false；不同时运行两个写入同一训练目录的任务。原只读监控已停止。


## 2026-10-22 diffusion/edit 闲时任务状态变化

2026-10-22T08:28:57.854523+00:00 task_id=`t-20261022160418-jwww4`；平台`Killed`；训练观察`stopped_or_failed`；最后日志28225，最新完整28000；Preemptible=true。[监控](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/spot_20261022/monitor_status.json)。


## 2026-10-22 submit 非闲时 ZENITH 被权限拒绝

2026-10-22T08:29:50.240397+00:00 submit `bert-diffedit-resume28000-200k-nonspot-zenith-20261022`，YAML `/project_c20250601/251105016/project/dllm_test/train_jobs/bert_diffusion_edit_resume28000_nonspot_zenith_20261022.yml`，Preemptible=false，8×A10080。CLI退出1，错误 `User is not authorized to perform: ml_platform:CreateCustomTask`，资源 `trn:ml_platform:cn-beijing:2106112826:resourcequeue/q-20261009101456-4m9np`，RequestID=`2026102216285806DB5D5AC9A3BB1EBC29`。事后按完整唯一任务名查无任务，未分配task_id；Active表不新增。旧闲时任务已确认Killed并移出Active表。配置、恢复状态与提交证据保留，需为当前Key对应身份授权该队列创建训练任务，或由用户配置已获授权的Key后再提交。


## 2026-10-22 尝试通过 ZENITH 网页提交

2026-10-22T08:42:25.831126+00:00 用户要求直接访问 `https://zenith.test.bza.edu.cn/my/jobs` 提交同一非闲时续训。网页访问工具无法访问；开发机直连DNS报 `Name or service not known`，环境代理访问报TLS EOF，未进入登录或提交页面。未进行新submit/cancel，无新任务ID。已将既定YAML的镜像、挂载、8卡资源和原入口脚本整理为 `/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/web_submission/fields.json` 与 `/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/web_submission/entrypoint.sh`，网页字段尚待可访问后核对。需确认平台的校园网/VPN连接方式；网页访问失败不能用来判断该平台账号是否具有提交权限。[访问证据](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/web_access_status.json)。


## 2026-10-22 网页入口路径修正

2026-10-22T09:01:17.865507+00:00 已核实当前项目实际路径为 `/project_c20250601/251105016/project/dllm_test`，共享存储挂载根为 `/project_c20250601/251105016`。网页入口修正为 `bash /project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/web_submission/entrypoint.sh`；脚本使用当前挂载并在必要时建立旧前缀兼容链接供冻结checkpoint与conda使用。恢复器对源码根路径先resolve，再检验加载模块归属，修复符号链接导致的错误拒绝；冻结训练计划、数据、优化器与原200000目标保持。新入口 --verify-only 通过16份checkpoint／874源码／1567依赖哈希检查；未启动训练、未新提交任务。[路径验收](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/web_submission/mount_fix.json)。


## 2026-10-22 统一当前存储、代码与历史日志访问路径

2026-10-22T09:08:28.512124+00:00 按用户要求更新当前diffusion/edit启动配置、恢复／监控代码及维护文档链接到 `/project_c20250601/251105016`。199份维护Markdown和4份规则文件完成路径替换；原日志文件、checkpoint/provenance与冻结训练计划不改。新入口输出位于 `/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/entrypoint.log`；历史BERT、首阶段diffusion、200k扩展及20500恢复日志统一列入[路径表](/project_c20250601/251105016/project/dllm_test/docs/BERT_TO_DIFFUSION_EDIT_20261019.md#当前存储代码与日志路径)。没有submit/cancel或启动训练。验收通过：8项CPU测试、旧别名监控链接检查、CPU模拟启动边界检查、新入口完整恢复校验；16份checkpoint／874源码／1567依赖哈希通过，5份冻结计划／回执与15份历史日志哈希保持。[验收](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/path_migration_20261022/verification.json)。


## 2026-10-22 diffusion/edit 非闲时任务状态变化

2026-10-22T09:42:28.581523+00:00 task_id=`t-20261022172952-vshml`；平台`Failed`；训练观察`stopped_or_failed`；最后日志28225，最新完整28000；Preemptible=false。[监控](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/monitor_status.json)。


## 2026-10-22 网页续训失败定位及数据路径兼容修复

2026-10-22T09:44:07.786097+00:00 网页任务 `t-20261022172952-vshml`（`mr3w2d46j1idw66u`，ZENITH_MLP2_TASK，Preemptible=false）平台确认 `Failed`，结束于09:31:23 UTC；09:30:37的verified仅代表恢复包哈希通过。实际在数据门禁报 `Shard resolves outside the release and its immutable parent`：新挂载解析路径与旧V7不可变绑定的路径比较不一致，未进入模型恢复／首个训练更新。已加入独立哈希绑定的路径适配，保留允许目录、目标一致性和内容哈希检查；原84分片及索引、开发集、父BERT287000和完整checkpoint28000实际CPU检查通过，17项回归测试通过。原任务已退出，需在网页用同一入口重新提交。[修复验收](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/data_path_fix_20261022/verification.json)、[实际数据检查](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/data_preflight.json)、[失败日志](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/attempts/20261022T093034_13d9a827/torchrun.log)。 网页使用 `bash /project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/web_submission/entrypoint.sh`；等价配置 `/project_c20250601/251105016/project/dllm_test/train_jobs/bert_diffusion_edit_resume28000_nonspot_zenith_20261022.yml`。这是用户网页提交的实际任务回读，本轮没有CLI submit/cancel或启动GPU训练，终态任务不留在Active表。新训练标准输出和错误会同时写入平台日志、entrypoint.log及attempts下torchrun.log。


## 2026-10-22 存储改名完成

2026-10-22T10:08:17.985973+00:00 按用户要求完成当前存储路径的实际迁移：入口、Conda、训练源码、数据manifest/binding/index、checkpoint provenance/receipt均使用 `/project_c20250601/251105016`；删除入口的旧挂载链接逻辑和路径适配模块，使用原标准数据门禁。禁止旧挂载访问的同一sh完整预检通过，旧路径访问0次；84分片全量哈希、父BERT287000、两份完整checkpoint28000校验通过，全部84分片reader/collator抽读及17项CPU回归测试通过。权重、AdamW/scheduler/8-rank RNG文件哈希与训练配方保持，累计目标200000。上次网页任务 `t-20261022172952-vshml` 已Failed，本轮未提交或启动GPU训练，需在网页用同一入口重新提交。[迁移验收](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/storage_rename_20261022/verification.json)。


## 2026-10-22 diffusion/edit 非闲时任务状态变化

2026-10-22T10:22:08.808267+00:00 task_id=`t-20261022181600-wcfcx`；平台`Running`；训练观察`training`；最后日志28075，最新完整28025；Preemptible=false。[监控](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/monitor_status.json)。


## 2026-10-23 diffusion/edit 非闲时任务状态变化

2026-10-23T13:55:24.074893+00:00 task_id=`t-20261022181600-wcfcx`；平台`Killed`；训练观察`stopped_or_failed`；最后日志76900，最新完整76500；Preemptible=false。[监控](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/monitor_status.json)。


## 2026-10-24 OAS-only续训 r4 resubmit

操作 resubmit；task_id=`t-20261024152107-fm8ws`；任务名=`immune-fixed75m-oas-resume11000-8gpu-queue012-20261023-r4`；YAML=/project_c20250601/251105016/project/dllm_test/train_jobs/v5_fixed_chain_75m_oas_resume11000_8gpu_queue012_20261023.yml；提交后状态`Queue`，queue012、1×ml.pni2.28xlarge、Preemptible=false。来源检查704项与checkpoint-11000的17份文件SHA均通过；共享工作区3个已变化文件以任务专属挂载提供逐字节匹配副本；训练配置和恢复点保持原绑定。W&B切为offline，规避r3在在线登录时flags=null导致rank0在step0退出的问题。仅进队列，尚未开始训练更新。[准备/预检](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_resume11000_20261023)；[云任务配置](/project_c20250601/251105016/project/dllm_test/train_jobs/v5_fixed_chain_75m_oas_resume11000_8gpu_queue012_20261023.yml)。


## 2026-10-24 OAS-only r4 Failed：父数据绑定检查

任务`t-20261024152107-fm8ws`于2026-10-24T07:23:48Z Failed，已从Active表移除。八rank profile/NCCL与704项源码校验通过；正式训练前父数据gate报Data binding SHA differs，原始V5 binding和audit曾随存储路径迁移同步更新。原checkpoint11000仍完整，新增训练步数0。正在任务专属目录重建逐字节匹配的旧元数据并执行完整父数据gate；不修改当前共享报告。[原始日志](/project_c20250601/251105016/project/dllm_test/output/oas_resume11000_r5_20261024/r4_platform_failure.log)。


## 2026-10-24 OAS-only续训 r5 resubmit

操作resubmit；task_id=`t-20261024152826-krr7n`；任务名`immune-fixed75m-oas-resume11000-8gpu-queue012-20261024-r5`；YAML=/project_c20250601/251105016/project/dllm_test/train_jobs/v5_fixed_chain_75m_oas_resume11000_8gpu_queue012_20261024.yml；初始状态`Staging`；queue012，1×ml.pni2.28xlarge，Preemptible=false。保持checkpoint11000完整状态、原OAS数据与global512/300k设置，W&B离线。恢复3份迁移过的历史父绑定/audit为逐字节匹配副本并仅挂载于该任务。提交前真正执行完整父数据gate：85分片内容SHA、704源码、全部父绑定通过；checkpoint17文件SHA通过。证据 /project_c20250601/251105016/project/dllm_test/output/oas_resume11000_r5_20261024 。尚不据平台Running宣称已更新训练步。


## 2026-10-24 OAS-only r5 已恢复真实训练

2026-10-24T07:31:09.825549+00:00 查询平台Running；训练step_timing已推进到11166，时间戳1792827069.8011532，较恢复点11000新增166次更新，当前约0.338秒/步。r5通过完整父数据gate后进入训练，W&B离线启动成功。最新完整checkpoint仍11000（每1000步保存），不能把未保存更新作为可恢复点。[状态证据](/project_c20250601/251105016/project/dllm_test/output/oas_resume11000_r5_20261024/training_status.json)。

## 2026-10-24 通用教师验证云提交未创建任务

- 操作：三次submit尝试；配置 `/project_c20250601/251105016/project/dllm_test/eval_jobs/stage1_general_teacher_esmfold2_20261024.yml`；单A100、Preemptible=false。
- c20250601（23:43 UTC）资源总量0；ZENITH_MLP2_TASK（23:45 UTC）与queue012（23:47 UTC）均返回当前身份无CreateCustomTask权限。三次均无task_id，不加入Active表；原始提交日志位于 `/project_c20250601/251105016/project/dllm_test/output/general_teacher_validation_20261024`。
- 本轮改为当前空闲A100上的本地串行工程诊断；没有成功提交、取消或修改既有云任务。


## 2026-10-25 OAS-only r5 Failed：磁盘配额

2026-10-25T11:43:58.565616+00:00 查询任务`t-20261024152826-krr7n`已于2026-10-24T08:30:20Z Failed，已移出Active表。实际训练11000→15979，最后更新2026-10-24T07:59:00Z，最新完整checkpoint15000（13000/14000/15000的optimizer和scheduler文件均在）。rank0追加step_timing.jsonl时触发OSError [Errno122] Disk quota exceeded，随后其余rank退出/超时。目标300k未完成；本轮仅查询与保存失败证据，尚未重提。[原始日志](/project_c20250601/251105016/project/dllm_test/output/oas_resume11000_r5_20261024/failure_20261025.log)；[当前状态](/project_c20250601/251105016/project/dllm_test/output/oas_resume11000_r5_20261024/training_status.json)。


## 2026-10-25 OAS-only r6 从15000步续训 resubmit

操作resubmit；task_id=`t-20261025194633-xrh4w`；任务名`immune-fixed75m-oas-resume15000-8gpu-queue012-20261025-r6`；YAML=/project_c20250601/251105016/project/dllm_test/train_jobs/v5_fixed_chain_75m_oas_resume15000_8gpu_queue012_20261025.yml；初始状态`Staging`；queue012、1×ml.pni2.28xlarge、Preemptible=false。用户确认配额足够并明确要求重提；checkpoint15000的17份恢复文件SHA和8rank完整回执通过，输出目录1MiB写入/fsync通过，完整85分片父数据gate和15000恢复metadata检查均通过。任务专属launcher改为显式绑定resume_step=15000，原训练源码/数据、global512、300k目标及W&B offline保持原配置；沿用r5的逐字节匹配历史源码/元数据视图。[检查日志与spec](/project_c20250601/251105016/project/dllm_test/output/oas_resume15000_r6_20261025)。尚未依据平台状态宣称训练已更新。


## 2026-10-25 OAS-only r6 实际恢复训练

2026-10-25T11:48:55.297846+00:00：task_id=t-20261025194633-xrh4w 平台Running，启动后step_timing已从15000增长至15049（时间戳1792928935.0221605），确认实际优化更新，约0.353秒/步。最新完整checkpoint15000，后续每1000步保存；W&B离线模式。状态证据 /project_c20250601/251105016/project/dllm_test/output/oas_resume15000_r6_20261025/training_status.json。

## 2026-10-25 Protenix在线Pair速度检查与loader修复

2026-10-25T12:50:56.972685+00:00 在线Pair任务`t-20261025182020-v6lz2`平台Running，但自10:36 UTC八rank均停在creating_loader/step0，未产出正式metrics。spawn子进程因本项目scripts命名空间与Protenix同名常规包冲突而导入失败；已将本项目scripts设为显式包，并同步冻结器。独立loader修复包8项CPU回归、原生路径旧失败/新成功对照、4worker读取128条真实输入和完整SHA校验均通过。原plan/source/checkpoint0保留，原公共Pair sh转接新包；新输出train_fix_loader_20261025，仍同seed42从零初始化，MLM复用，教师目标不落盘。新包未启动，GPU资格待重跑；当前任务未取消。本轮未submit/cancel。历史最大形状profile约83.12秒/512条（2.22万条/小时），正式速度尚未测得。[现场与速度](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/diagnostics/loader_failure_20261025/status.json)；[修复与启动](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/launch_fix_loader_20261025/README.md)。


## 2026-10-25 OAS-only r6 32000步日志与验证趋势

2026-10-25T13:29:47.387664+00:00 训练约32560步、完整保存32000；当前训练loss约0.41–0.43，梯度范数约0.10–0.15，LR约9.75e-5。15000→32000的全mask验证loss1.61768→1.62057，AA准确率45.107%→45.409%，首EOS准确率13.281%→14.062%，长度MAE2.246→4.727aa；验证效果未显示持续改善。训练随机mask统计与全mask验证不应直接比较loss绝对值；该AA准确率不是下游CDR生成AAR。近期抓取日志未见Traceback、OOM或quota异常。证据 /project_c20250601/251105016/project/dllm_test/output/oas_resume15000_r6_20261025/log_review_latest.json。

## 2026-10-25 Protenix在线Pair终态核实

2026-10-25T15:56:06.533546+00:00 在线Pair旧任务`t-20261025182020-v6lz2`已于2026-10-25 13:27:53 UTC结束，平台Killed；正式训练仍为step0/seen0、只有checkpoint0。最新loader修复版输出目录尚不存在，未见profile、正式metrics或新checkpoint，当前未观察到该组训练启动。修复包与8项CPU/真实四worker输入验收保持就绪；复用已完成MLM，教师目标不落盘。本轮仅查询，无submit/cancel。[状态快照](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/diagnostics/training_status_latest.json)；[修复版启动入口](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/launch_fix_loader_20261025/README.md)。

## 2026-10-26 Diffusion四卡W&B启动失败

用户任务 `t-20261027001920-m9ct7` 查询为Failed，smoke passed、正式优化0步；Python3.10/cp312 vendor ABI冲突。修复包与原入口转接已验证，未代为submit/cancel，终态任务不加入Active表。[故障与当前四卡入口](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_FIXED_FRAME_CDR_20261026.md)
