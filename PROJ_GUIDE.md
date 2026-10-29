> 2026-10-29 UTC：本地层次架构续跑入口 /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_hierarchy.sh --gpu 0；输出 /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_hierarchy_m32_20261029，共同micro32/global512(16次累积)、seed43/1epoch。先四组逐个做3次maxlen2048真实GPU资格检查，全部通过才正式训练；源码位于architecture_hierarchy独立冻结快照，模型复用原architecture_round2。旧数据/输出/源码SHA保持；共同资格和profile文件SHA绑定、原checkpoint恢复；每组结束发布曲线/CSV/报告到探索分支。启动后不持续人工监督。

> 2026-10-29（UTC）本地后续架构入口 /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_round2.sh --gpu 0；输出独立为 /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_round2_20261029。B1/B2/B3只seed43、各1epoch/1276步，统一micro56/global512，最大长度2048显存门槛85%；顺序单GPU，无DDP/NCCL。architecture_round2在新进程适配原冻结trainer，旧模型/训练/data reader文件保持。第二轮源码快照SHA与原SHA同时记录，数据index字节相同。每组结束自动发布完整曲线/CSV和报告，仅探索分支；失败push独立pending。持续人工监督仍暂停，当前本地GPU队列授权不等于重启旧三种子确认。

> 2026-10-29（+08:00）8卡启动器修复：本平台NVML UUID与CUDA UUID不一致，CUDA_VISIBLE_DEVICES使用数字卡号，设置CUDA_DEVICE_ORDER=PCI_BUS_ID；NVML UUID仅用于物理设备去重审计。worker启动时实际cuda.init并运行tiny kernel，不能只信NVML device_count。发布时按当前环境代理或直连覆盖Git命令，忽略连不上的本机loopback代理，不修改共享Git全局配置；推送失败记录pending并保留提交，训练原始错误与退出码独立显示。可用 --publish-only 稍后同步结果，无GPU工作；--no-push保留本地提交。13项回归＋本机真实单卡验证通过，8卡重试待执行。

> 2026-10-29（+08:00）当前架构复核入口为 [launch_arch70m_8gpu.sh](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_8gpu.sh)：仅seed43、A0–A7各单卡1epoch。统一micro56，9个56加一个8条micro组成global512；更改时通过 --micro-batch 对八组统一指定。每卡在最大长度2048上做3次真实更新核验85%显存门槛，再由全新进程从头训练；不逐架构自动改变micro。训练器和数据仍用已冻结版本。当前8卡脚本由用户执行，本机未启动新任务；旧三种子调度已停止，当前已启动的A0继续收尾，持续监督暂停。报告位于 docs/arch70m_results/seed43_validation，逐试验commit/push探索分支。

> 2026-10-28 用户再次明确：架构探索始终沿用当前 V8 数据形式。原 prepared JSONL 字段、链序列与顺序、角色/元数据、relation/family 标签、tokenizer/词表/特殊 token、固定链输入布局保持；10% 子集只选择原记录的 offsets，独立实验索引不构成训练数据格式升级。后续架构变体遵守同一约束。

> 2026-10-28 GitHub 身份：当前开发机已通过用户本人设备验证登录 `waitma`，API 已确认对 `waitma/dllm_test` 有 push 权限；GitHub CLI credential helper 已配置，CLI 位于 `/root/.local/share/github-cli/2.103.3/gh`。探索结果仅推送 `explore/arch70m-v8-20261028`。这是当前开发机的认证状态，其他作业容器需具备自己的可用认证；实验文档与提交仅记录账号和发布回执，不保存认证令牌。

> 当前开发机的 GitHub 传输按 URL 配置 HTTP/1.1、64 MiB postBuffer 和现有本机 HTTP 代理，首次推送遇到 HTTP 408 后使用该组合成功；配置前值与新值存于探索 output 的 `github_auth_setup.json`。模型与训练源码未改动，后续自动发布沿用 Git credential helper。

> 2026-10-28 四卡交付入口：[launch_arch70m_4gpu.sh](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_4gpu.sh)，默认 screen 的 A3–A7，开发机单卡负责 A2，A0/A1 已完成；`--stage confirm` 在全部筛选结束后进入确认。四个独立单卡 worker，支持 `--gpus`、`--trials`、`--machine-index/--machine-count`、`--dry-run`。入口放在工作树根部，仅转发既有 suite，不改冻结训练代码与 A0/A1 源码哈希。

> 2026-10-28 探索工作树规则：本目录 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore` 仅提交/推送 `explore/arch70m-v8-20261028`。所有架构试验使用独立单 GPU worker 和不可变源码快照，四卡并行试验，不调用 DDP/NCCL。global batch 512 固定，micro 由最大长度三次实际 optimizer update 的显存 profile 决定。原 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_test` 训练版本只读；长训不受改动。入口、恢复、发布规则见 [架构探索实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。


> ownkaifa02 当前项目根为 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_test`，存储根为 `/mnt/project-storage/fset-rN4kfh/251105016`。文件已同步；冻结元数据及入口仍按源端原文保存，须显式路径／哈希迁移并通过标准门禁后运行。详见 [OWNKAIFA02_TRANSFER_20261027.md](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/OWNKAIFA02_TRANSFER_20261027.md)。

2026-10-27 用户明确授权删除两个旧8B训练输出目录；现已删除 `/project_c20250601/251105016/project/dllm_test/output/protein_esmc_llada8b_bert_immune` 与 `/project_c20250601/251105016/project/dllm_test/output/protein_esmc_llada8b_diffusion_immune` 全部内容，历史正式评测权重保留规则对此两目录不再适用。其他模型保留规则继续有效。历史文档及清单中的8B权重路径仅为来源记录，不代表文件仍可加载。[删除回执](/project_c20250601/251105016/project/dllm_test/output/output_cleanup_20261027/legacy_8b/completion.json)。

2026-10-27T04:36:28.400835+00:00 V8启动便携性要求：预检导入的旧数据脚本也不得读取/vepfs-mlp2挂载；去重normalizer使用当前包导入，保证冻结reader身份。当前用户网页入口指向 output/bert_v8_vfamily_all_relations_scratch_300k_20261027/launch_pathfix.sh 和 bundle_pathfix；启动先运行CPU完整预检，再进入8rank profile/replay/train。回归检查必须禁止旧挂载访问，不能只在存在旧路径别名的开发机上验证。

2026-10-27T04:08:38.107064+00:00 修复V8 family网页提交EnvironmentNameNotFound：原脚本按名称激活pllm依赖开发机Conda搜索配置，现对齐成功diffusion任务，以绝对路径 /project_c20250601/251105016/conda/envs/pllm 激活。新增 /project_c20250601/251105016/project/dllm_test/output/bert_v8_vfamily_all_relations_scratch_300k_20261027/launch_envfix.sh；旧launch与冻结bundle保持。三个网页入口及当前两份train_jobs YAML重定向修复脚本并同步入口SHA，生成器同步修复。训练解释器仍为原已验证protenix_abtcr绝对路径；模型、数据、8卡及loss配置不变。未提交任务。 验证：隔离Conda环境搜索目录的绝对路径激活、bash语法及原网页入口--verify-only均通过；121分片/6,604,459条数据和冻结plan SHA核验通过。GPU启动待用户重提。

2026-10-27 最新用户侧网页Shell/YAML转接到/project_c20250601/251105016/project/dllm_test/output/bert_v8_vfamily_all_relations_scratch_300k_20261027，用户原bash命令保持。183类schema v3，包含negative family supervision；300000步随机初始化、global512/初始micro32×GA2×8、LR1e-4、warmup15000、max2048不变。新GPU显存profile与八rank恢复资格待作业内执行，旧包不覆盖、无自动提交。[原网页Shell（已更新）](/project_c20250601/251105016/project/dllm_test/output/bert_unbinding_v8_vfamily_scratch_300k_20261027/web_submission/entrypoint.sh)。

2026-10-27 V-family unknown版当前包为/project_c20250601/251105016/project/dllm_test/output/bert_unbinding_v8_vfamily_unknown_scratch_300k_20261027；原用户网页Shell与两份scratch YAML已指向新包，提交命令不必改。随机初始化300000步、LR1e-4、warmup15000、global512、max2048保持；head改183类，GPU显存profile与八rank恢复资格必须重新在新包运行。无自动提交；旧182类冻结bundle不覆盖。[原网页Shell入口（已转接unknown版）](/project_c20250601/251105016/project/dllm_test/output/bert_unbinding_v8_vfamily_scratch_300k_20261027/web_submission/entrypoint.sh)。

2026-10-27 用户选择自行网页提交V8 V-family随机初始化训练，当前入口为[网页提交Shell](/project_c20250601/251105016/project/dllm_test/output/bert_unbinding_v8_vfamily_scratch_300k_20261027/web_submission/entrypoint.sh)。网页启动命令使用bash加该绝对路径，无需volc；单节点8×A100-80G，共享存储挂载/project_c20250601/251105016。默认执行全部资格检查及训练，--verify-only仅CPU检查；日志写output/bert_unbinding_v8_vfamily_scratch_300k_20261027/entrypoint.log。原冻结模型、数据、300000步配方不变。

2026-10-27 当前从零训练入口切至V-family版：/project_c20250601/251105016/project/dllm_test/output/bert_unbinding_v8_vfamily_scratch_300k_20261027/launch.sh。专用vfamily YAML及原scratch YAML均指向新冻结包；旧baseline输出/train.yml保留。300000步、LR1e-4、warmup15000、global512、初始micro32×GA2×8、max2048；新增头后GPU显存待profile，不沿用旧资格。--phase all强制新八rank profile（由micro64向下搜索）与恢复重放；smoke末32条覆盖有标签OAS/OTS，保证各rank有family标签。--verify-only仅CPU，不启动GPU或提交。[当前family训练YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_unbinding_v8_vfamily_scratch_300k_8gpu_20261027.yml)。

2026-10-27 V8 R2从零训练入口：/project_c20250601/251105016/project/dllm_test/output/bert_unbinding_v8_scratch_300k_20261027/launch.sh；--verify-only只跑CPU哈希/数据门禁，--phase all执行八rank profile→恢复重放→正式训练。随机初始化明确禁止parent/resume字段；独立输出目录拒绝覆盖。global512、初始micro32×GA2×8，max2048；新作业显存待profile，从micro64开始选择全rank可用的最大micro，不能以旧单卡结果宣称本作业已验收。[从零训练配置](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_unbinding_v8_scratch_300k_8gpu_20261027.yml)。

2026-10-27 汇报当前入口：精简稿21页，前4页确认稿XML保留，合并draw.io含20个可编辑页面。统一目录 `/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报`；完整重建用 `/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/scripts/build_progress_report.py`，导出PDF后由 `/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/scripts/verify_progress_report.py` 验收。旧build_architecture.py/verify_report.py仅适用于四页基础稿。结果先更新正式论文，再将14张真实论文表格截图组合为8页；各表固定展示BERT V7 step247000和Diffusion/Edit V7 step76500，缺少已验证结果的单元格显示“—”，不补造数据。BERT既有评测先恢复到canonical RESULTS.md再同步论文；Diffusion来自BERT287000，所展示BERT247000是另一既有评测快照，两行不作为同一父模型的因果阶段消融。按本次用户要求精简baseline但保留强参照，已发表主指标领先行用浅蓝底，PPT结果页不加阅读指南或额外注释；不同协议仍由表标题区分。论文和PPT结果表统一以ROUND_HALF_UP显示两位小数，原始精度在RESULTS.md及manifest.raw_rendered_rows保留，底色按原始值选择；m396仅保留主要row split，不展示grouped表。TCR生成仅Sparse13和held20；任务及结果页排除TCR clustering/representation（T2/T3），不改变项目评测范围。下游任务合并一页并保留两个透明GIF；Pair机制、完整loss曲线及矩阵对照压缩展示；中文负样本扩充仅一页，Next plan列Unbinding对照和扩大参数两项。PPT放真实透明论文截图，draw.io另附原生可编辑副本；冻结资产复用并核验SHA。live台账增加说明时，既有汇报可用与原SHA完全一致的来源快照验收，差异及当前SHA必须显式记录在data/report_source_snapshots.json，不能改写当前台账或静默跳过核验。矩阵页右侧标题与三条总结使用PPT原生文本框，底图不含总结文字，可直接编辑字体与内容；右侧用“本例观察”概括所示Pair通道变化、部分链内短距离模式与较弱跨链热点；valid_01无实验接触真值，下一页用另3个Fab结构样例验证。矩阵及少量既有案例不作为独立泛化证据。按用户允许缩放，本版内容字号下限18pt，表格有效字号最低18.92pt；无备注、交大主题保持。原36页稿完整归档于 `_旧稿归档/双模型精简前_36页_20261027_020821`。本段替代此前36页、全baseline及统一20pt展示约束。[编辑说明](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/图形样式与编辑说明.txt)、[文件清单](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/data/report_manifest.json)。

2026-10-27 架构与训练图最新视觉约定（用户再次修订）：架构优先贴近IgGM2 Figure 2a及本项目早期模块版，采用浅色分组→蓝色模块→白色运算层的明确层次；不再把架构画成平铺的横向文字框。主模型与单层展开分别清楚呈现，借鉴样式不复制参考模型运算。架构和三阶段训练图内均使用简洁英文；正文保持至少20pt，训练图同一蓝系浅→深、输入/模型/输出对齐，第三阶段标记Planned。交大封面、Pipeline位置、无备注、所有图原生draw.io与透明/矢量导出约定继续有效。交付前实际检查PPT渲染，并与参考及前版并排审图；可计算明确可重现的配色/着色覆盖统计，但不能把这些统计当作美观分数或编造审美相似度百分比。当前对照证据在 `/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/preview/style_review`。本条替代上一轮架构“只局部着色、减少大底板”的选择，其他图按内容配色。

2026-10-27 汇报顺序补充（用户要求）：在模型架构前增加模型Pipeline页，当前完整汇报顺序为交大封面→Pipeline→模型架构→整体思路。Pipeline讲输入对象、序列输入形式、推理与输出用途，可用可编辑分子结构域示意帮助说明对象；当前序列模型的结构图标不表示三维坐标输入或直接结构预测。沿用论文图风格、大字、无备注、统一目录及所有图均有draw.io原稿的约定。新增源图 `/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/figures/00_模型Pipeline.drawio`；合并源稿现有3页。

2026-10-27 论文原图与汇报图风格约定（用户最新要求）：参考实际发表在Nature系列期刊的图，重视整体布局、留白、字体层级和配色；本次依据RFdiffusion、Chroma、InterPLM Figure 1建立白底细线、局部低饱和强调的绘图规范。统一字体/线宽/间距不等于每张图都用同一色板；三阶段训练流程仍按同一蓝系浅→深。所有图都交付draw.io可编辑版本，文字、模块、标识和连线保留独立图元，另附SVG、矢量PDF和透明PNG，便于后续论文组图。完整汇报继续保留交大封面和母版，独立论文图不带交大页眉。当前合并原稿为 `/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/figures/Ophiuchus-AIR_论文图源文件.drawio`；样式、色值与编辑方法见 `/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/图形样式与编辑说明.txt`。

2026-10-27 汇报最新视觉要求：架构图需恢复关键模块和单层细节，图内模块以英文为主；保留清楚的字号层级，避免零碎小字，也不能过度简化。当前正文采用20/22/24/26pt；此为排版实现选择，替代先前“至少24pt、只保留少量主流程”的处理。整体思路页每阶段均展示输入、Ophiuchus-AIR名称标识和输出，用同一蓝色系由浅到深表达三阶段，专用模型微调标记为后续计划。借鉴参考图的视觉风格不等于所有页面沿用架构图配色，后续配色按内容选择。继续保留交大封面/母版、可编辑draw.io、透明图及无备注约定，资料统一放在 `/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报`。

2026-10-26 开头页保留约定（用户再次强调）：每份完整汇报PPT都必须以交大主题封面作为第1页，包含项目进度汇报、Ophiuchus-AIR、免疫受体基础模型、汇报人和本次日期；PDF也从同一封面开始。后续逐页添加/修改正文时持续保留封面。当前生成脚本已加入封面必需字段检查。

2026-10-26 用户明确当前模型名称为 **Ophiuchus-AIR**（用户输入：ophiuchus-air）。本次汇报顺序为封面→当前模型架构→整体思路；整体思路横向三阶段BERT预训练→Diffusion生成训练→抗体Design等任务的专用模型微调，第三阶段按后续计划表述。沿用交大主题、大字、无备注、draw.io可编辑及本次汇报目录统一存放的约定。该命名用于项目汇报，不改动既有checkpoint路径或外部基线模型名称。

2026-10-26 汇报排版补充（历史版本，详略与字号已由2026-10-27要求更新）：本次汇报需有交大主题开头页；当时按大字、少量主流程重排。页面不添加备注式说明，演讲者备注保持为空。当时为封面＋架构2页，仍在 `/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报` 内逐页维护。

2026-10-26 本次汇报文件约定（用户最新要求）：本次2026-10-27汇报的PPT、draw.io、图片、结果表、脚本、参考资料和说明统一放在 `/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报`，不要再散放PPT根目录。当前逐页重做，只保留用户当前要求的范围；当前仅架构1页，后续按用户反馈逐步补充。架构必须保留可编辑draw.io原图，scope不在图中展示；交大母版继续使用，架构视觉参考IgGM2 Figure 2。先前自动生成的全稿已移入新目录_旧稿归档，不作为活动汇报。

# 项目操作约定

2026-10-28 1B正式启动：用户明确要求跳过额外测试；当前入口为 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/output/bert_v8_1b_mb16_300k_20261028/launch.sh`，原micro16的launch_deterministic.sh和launch.sh已转接。micro16×GA4×8/global512，300000步随机初始化，warmup15000，每1000步验证/保存；不运行独立NCCL资格/profile/smoke/200步测试，不伪造通过回执。保留正常输入/设备/认证及checkpoint完整性检查；已有train目录拒绝覆盖，没有自动续训。当前单卡51.25/55.38 GiB测量不能当八卡显存资格。正式八卡任务由用户平台启动；助手不得把准备完毕写成已开训。[正式说明](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/BERT_1B_DIRECT_TRAIN_20261028.md)。

2026-10-28 1B重试与W&B身份：从头重跑必须创建独立run ID；冻结plan中的run_id仅作前缀，UUID不消耗训练RNG。实际ID由rank0创建并保存为wandb_run.json，随checkpoint一并保存并纳入哈希回执；恢复时验证该文件，使用同一ID及resume="must"。本地归档不能清除云端run，禁止把resume="allow"当作从零重试的通用修复。新trainer拒绝缺失/冲突的checkpoint身份；旧checkpoint沿用原冻结代码。原micro8重试入口为 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/output/bert_v8_1b_test_20261027/launch_deterministic.sh --phase all`，该scratch入口尚不支持自动正式续训。修订/验收：[1B测试说明](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/BERT_1B_TEST_20261027.md)。

W&B日志保持简洁：当前1B包按显式标量白名单上传关键训练指标、五项总体验证指标及总体probe AUROC，不自动扩展分来源/计数/诊断曲线；详细数据写入本地metrics.jsonl、validation/probe JSON。日志展示调整不改变验证计算、损失或采样。[当前白名单](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/BERT_1B_TEST_20261027.md)。

2026-10-27 ownkaifa02 1B测试：独立Python环境 `/mnt/project-storage/fset-rN4kfh/251105016/conda/envs/pllm`，入口 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/output/bert_v8_1b_test_20261027/launch.sh`。W&B使用该包交互式 `--login --entity … --project …` 重新验证身份和写权限，禁止继承旧vendor/凭据路径或将key写入YAML。默认八卡profile＋恢复门禁＋200步；新平台挂载/队列/镜像确认后才生成最终提交YAML，不沿用旧Vepfs ID。其他包的历史路径不自动适配。[使用说明](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/BERT_1B_TEST_20261027.md)。

2026-10-26 汇报视觉约定：用户明确项目汇报使用交大主题，以 [20260804项目进度汇报_免疫语言模型.pptx](/project_c20250601/251105016/project/dllm_test/PPT/20260804项目进度汇报_免疫语言模型.pptx) 为默认模板，保留其母版、联合标识、红色标题栏及校门水印封面；不自行替换为独立科研配色主题。结果表与截图采用透明背景，正文/架构尽量保持可编辑；用户后续指定优先。

<!-- four-relations-20261026 -->
2026-10-26 V8 BERT当前四任务R2数据为 `/project_c20250601/251105016/project/dllm_test/data/prepared/immune_v8_four_relations_20261026_r2`；入口 `/project_c20250601/251105016/project/dllm_test/train_jobs/bert_unbinding_v8_from287000_8gpu_20261026_four_relations_r2.yml`，输出 `/project_c20250601/251105016/project/dllm_test/output/bert_unbinding_v8_four_relations_from287000_20261026_r2`。原287000权重新增13000步，global512/micro32×GA2×8；max2048单A100 reserved43.916GiB，八rank资格由作业内执行。目标固定binding_mlm_nonbinding_relation_only_vhh；R1入口已禁用，禁止绕过SUPERSEDED标记。推断错配不能称实测；AVIDa遵循源CC-BY-NC-4.0，源NbBench val/test不导入train。未提交。[四类负例探索、证据与R2入口](/project_c20250601/251105016/project/dllm_test/docs/V8_FOUR_RELATION_NEGATIVES_20261026.md)
<!-- /four-relations-20261026 -->

2026-10-27 Git 备份已同步：GitHub 设备授权完成，已将此前 4 个本地提交推送至 https://github.com/waitma/dllm_test 的 main；远端实时核验为 `e61446995da26ada16cd843bff0bfb8fb2c0fc27`，与本地 HEAD 一致。包含 `ac887a0`（取消 383 项非代码跟踪）和 `e614469`（669 个代码文件的当前改动）；652 个 Python 文件与 17 个 Shell 脚本通过语法检查，未执行完整功能或训练测试。JSON/YAML、Markdown、论文/图片（含 Mermaid `.mmd`）等仍仅在本地维护，历史提交不改写。授权等待期间另有 9 个已跟踪 Python 文件修改、10 个未跟踪 Python 文件及子模块工作区变化，保留在本地，未纳入这次推送。Git 身份沿用 waitma，仅配置本仓库；认证由 GitHub CLI 管理，不把令牌写进项目文件。

2026-10-26 Git 跟踪范围：当前保留源码（Python/Shell）及必要的 Git、依赖和构建文件；JSON/YAML、Markdown/MDC 文档、数据/日志/权重、论文、图片、字体和环境杂项仅在本地保留。通过 `git rm --cached` 取消已有非代码文件的跟踪，保留本地文件，历史提交不改写。忽略规则维护在 /project_c20250601/251105016/project/dllm_test/.gitignore；后续仍按文档同步规则维护本地 Markdown，当前不要用 `git add -f` 重新加入上述文件。

2026-10-26 Diffusion四卡稳定入口 `/project_c20250601/251105016/project/dllm_test/output/diffusion_bert287000_cdr_4gpu_20261026/launch.sh` 已转接Python3.10 W&B修复包 `/project_c20250601/251105016/project/dllm_test/output/diffusion_bert287000_cdr_4gpu_wandbfix_20261026/launch.sh`，旧vendor中的cp312扩展不能用于pllm3.10；依赖须与实际执行Python匹配并在GPU启动前做真实导入/ABI/Settings检查。新依赖版本及SHA锁在 `/project_c20250601/251105016/project/dllm_test/output/diffusion_wandb_py310_repair_20261026/requirements.lock`，不覆盖共享环境/旧vendor。global512/每rank256视图/micro48×5＋16保持；修复包A10080 max2048 allocated54.453/reserved69.246GiB。四卡YAML同步转接，旧失败输出保留，修复包独立smoke/train/W&B。[故障与当前四卡入口](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_FIXED_FRAME_CDR_20261026.md)

2026-10-26 V8 BERT上一版抗体扩充数据入口为 `/project_c20250601/251105016/project/dllm_test/data/prepared/immune_v8_unbinding_slots167_135_mhc_ab_expanded_20261026`，训练YAML为 `/project_c20250601/251105016/project/dllm_test/train_jobs/bert_unbinding_v8_from287000_8gpu_20261026_mhc_ab.yml`，独立输出 `/project_c20250601/251105016/project/dllm_test/output/bert_unbinding_v8_mhc_ab_from287000_20261026`。继承MHC补全版并新增28条配对抗体阴性；VHH不在本包。原V8/MHC-only包保留各自绑定；从BERT287000权重新增13000步。global512/micro32×GA2×8；同模型/max2048复用已验单卡reserved39.678GiB（micro64余量不足），本次另验新增抗体/补全MHC真实批次；八rank资格由作业内执行，未提交。[抗体扩充、来源与新版入口](/project_c20250601/251105016/project/dllm_test/docs/V8_ANTIBODY_NEGATIVE_EXPANSION_20261026.md)。

2026-10-26 Diffusion当前提交入口为 `/project_c20250601/251105016/project/dllm_test/train_jobs/diffusion_bert287000_restart_8gpu_20261026.yml`，从原BERT287000权重新开Diffusion；旧resume76500 YAML/train.sh已禁用。global512原记录/1024视图、8卡每rank128视图按48+48+32累积，当前冻结代码A10080/max2048实测allocated54.452/reserved69.006/total79.151GiB；作业内先8rank smoke再从BERT正式初始化。新输出/W&B独立，未提交。[数据审计与新提交入口](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_BERT_RESTART_DATA_REVIEW_20261026.md)

2026-10-26 更正：用户要求从原 BERT287000 重新开始 Diffusion。此前 resume76500 仅为保留旧进度的修补方案，已撤销；旧 YAML 与 train.sh 已设置立即退出，尚未提交该任务。新实验准备中：Diffusion step0、新 AdamW/warmup、独立输出，并开展全量数据与验证重叠审计。

2026-10-26 Diffusion独立修复提交入口：[diffusion_guard_resume76500_8gpu_20261026.yml](/project_c20250601/251105016/project/dllm_test/train_jobs/diffusion_guard_resume76500_8gpu_20261026.yml)。global512原记录=1024互补视图，单节点8卡；每rank128视图按48+48+32累积。当前冻结代码A10080长度2048两次完整更新显存短测passed，allocated54.454/reserved65.957/total79.151GiB；旧micro64曾余量不足，保留48。作业内先执行本包八rank恢复重放，再正式从76500续训至200000；新W&B/输出，不复写旧实验。首次提交脚本不支持直接重复执行；已有输出应使用冻结launcher的--resume恢复该修复run的完整checkpoint。[profile回执](/project_c20250601/251105016/project/dllm_test/output/diffusion_guard_repair_20261026/max_shape_profile.json)。当前仅准备，未submit/cancel。

2026-10-26 V8 unbinding采用独立[八卡YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_unbinding_v8_from287000_8gpu_20261026.yml)与[冻结启动包](/project_c20250601/251105016/project/dllm_test/output/bert_unbinding_v8_from287000_20261026/bundle/plan.json)。只从原BERT287000权重开始新阶段，optimizer/scheduler重置；默认新增13000步。global512=32×GA2×8，长度2048单卡reserved39.678GiB；micro64 allocated71.660/79.151GiB超过90%门槛，32是下一个整除档位。启动必须通过本包八卡profile和正负混合恢复短测，不能复用V7资格或手动改冻结plan/SHA。该V8绑定独立于旧V7硬编码release SHA。[操作与限制](/project_c20250601/251105016/project/dllm_test/docs/BERT_V8_UNBINDING_20261026.md)。

2026-10-25 Protenix在线方案固定teacher_target_storage=none；只新增Pair，复用已完成MLM。当前[最新启动包](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/launch_fix_loader_20261025/README.md)修复spawn worker与原生Protenix的scripts包冲突：项目scripts必须保留__init__.py并纳入source冻结。原公共launch/train_pair_kd.sh与qualify_pair_kd.sh已转接，新输出`/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/train_fix_loader_20261025`。sampler修复任务已于13:27 UTC终止（Killed），停于step0；最新loader修复版尚无运行输出。8项CPU回归、四worker真实128条输入和完整输入校验通过，新包八卡资格待运行；相同seed42随机初始化，后续仅续本包latest。历史联合profile micro64×1×8/global512、峰值22.174/79.151 GiB；64是固定global512的每rank有效batch上限。正式YAML：`/project_c20250601/251105016/project/dllm_test/train_jobs/stage1-protenix-online-ab500k-fix-loader-20261025-train-pair-kd.yml`。新Pair与旧MLM环境不同，最终统一评测复核。


2026-10-24 AB500K从零学生训练改用 /project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_scratch_20261024/launch/train_mlm.sh 和 train_pair_kd.sh；各单节点8卡，首次随机初始化，后续仅续接该实验latest。CPU预检追加 --verify-only；原续训入口仍保留。两个作业由用户分别提交，项目盘必须挂载 /project_c20250601/251105016。[操作计划](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_SCRATCH_PLAN_20261024.md)。

2026-10-23 AB500K：NCCL初始化后的多worker DataLoader显式使用spawn并设置读批超时；资格短测须覆盖正式加载与日志路径。当前训练入口为 `/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/launch_training_fix_20261023`，旧包保留供审计，避免直接复用存在首步日志错误的入口。[代码复核](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_TRAINING_REVIEW_20261023.md)。

<!-- stage1-ab500k-offline-pilot-20261022 -->
2026-10-22 AB500K冻结包已完成：四个teacher_cache shell各单节点4卡，train_mlm/train_pair_kd各单节点8卡；check_cache与report为CPU入口，共8个sh、6份GPU YAML，由用户手动提交。所有入口从冻结源码启动；教师复用完整分片，学生默认latest恢复。global512不变，每arm提交后实际profile选择最大安全micro；逐记录MASK不受micro影响。33项CPU测试和teacher CPU运行时预检通过，训练在缓存未完成时返回not_ready；GPUprofile/恢复检查仍待提交后执行。[提交入口](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/launch/README.md)；[实验合同](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_OFFLINE_PILOT_PLAN_20261022.md)。
网页任务必须将Vepfs `vepfs-cnbj2c98dea54433`的`c20250601/251105016`以读写方式挂载到`/project_c20250601/251105016`；仅有personal/public挂载无法访问脚本。六份任务YAML已包含此配置。
<!-- /stage1-ab500k-offline-pilot-20261022 -->

<!-- stage1-online-20261022 -->
2026-10-22 当前Stage 1入口 `/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/launch_online_v2/entrypoint.sh` 已由用户运行，单节点8×A10080。同卡profile已选micro16，global512=16×4×8，联合显存峰值估算64.21 GiB；micro24估算76.56 GiB余量不足。8卡在线缓存及恢复短测通过，训练输出 `/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/online/train`，同包自动恢复latest，STOP位于该目录。动态缓存独立持久化；教师Python入口须保留虚拟环境软链接路径。训练前3步/每100步/结束记录，教师进度最多每60秒汇总；初始缓存覆盖不再是在线模式门槛，结构质量仍未验证。[操作及回执](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_ONLINE_DISTILLATION_20261022.md)。
<!-- /stage1-online-20261022 -->

<!-- stage1-logging-20261022 -->
2026-10-22 Stage 1合并工程检查已通过，不再要求重跑profile/smoke/teacher pilot。后续控制台常规进度按60秒汇总，阶段与诊断即时输出，逐缓存metadata/results继续保留；独立日志入口 `/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/launch_checks_quiet/entrypoint.sh` 保持原教师contract/资格回执。新训练包默认 `prepare_stage1_training --log-every 100`，仅rank0记录前3步、每100步、最后一步；每500步验证/保存不变。正式训练尚需教师质量与足够覆盖，旧candidate不能直接使用。[说明](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINING_PIPELINE_20261022.md)。
<!-- /stage1-logging-20261022 -->

<!-- stage1-combined-checks-20261022 -->
2026-10-22 Stage 1当前统一提交入口为 `/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/launch_checks/entrypoint.sh`，资源仍为非闲时单节点8×A10080。smoke、教师pilot、索引、审计在同一作业顺序执行；教师与学生沿用各自冻结源码及独立环境。`--verify-only`仅CPU校验，完成记录位于同目录；工程检查完成不等于实验结构准确性通过，也不自动开始长训。[操作说明](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINING_PIPELINE_20261022.md)。
<!-- /stage1-combined-checks-20261022 -->

<!-- stage1-training-build-20261022 -->
2026-10-22 Stage 1独立训练输出根 `/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022`，学生pllm/教师独立环境；提交沿用ZENITH非闲时8×A10080。冻结入口支持 `--verify-only`、同包 `--resume latest` 与输出目录STOP。新profile已通过：max_length2048、每记录8192对、micro24，allocated56.26/reserved65.05 GiB；正式global512应按每卡24+24+16累计。新的 `launch_smoke/entrypoint.sh` 已绑定回执，global8用于短程8卡更新/恢复验证；旧候选包缺回执时拒绝运行，不能手改plan解锁。当前日志为本地JSON，尚未绑定Stage 1 W&B。[Stage 1训练构建与提交](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINING_PIPELINE_20261022.md)。
<!-- /stage1-training-build-20261022 -->

<!-- stage1-environments -->
2026-10-22 Stage 1结构增强的学生沿用pllm；ESMFold2-Fast教师使用独立Python `/project_c20250601/251105016/conda/envs/stage1_esmfold2/bin/python`（transformers5.20.2），不得升级已有训练环境代替隔离。固定教师目录 `/project_c20250601/251105016/model_weights/esmfold2_fast/45fe8656f5b3ef493c17fcf9abe9a2968902e712`。教师必须经本项目Fast兼容加载入口，按checkpoint禁用MSA encoder且继续严格核对全部权重。本地入口已验收单设备有限pilot与精确恢复，正式GPU训练仍走既有Volc流程。[实现记录](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_STRUCTURE_IMPLEMENTATION_20261022.md)。
<!-- /stage1-environments -->


2026-10-22 当前存储根 `/project_c20250601/251105016`，项目根 `/project_c20250601/251105016/project/dllm_test`。维护文档中的代码／数据／历史日志链接统一使用当前前缀；历史事件、指标及原始日志保留；运行所需checkpoint路径元数据及关联哈希已同步迁移。网页入口、冻结源码与各阶段日志的唯一操作索引见[当前存储、代码与日志路径](/project_c20250601/251105016/project/dllm_test/docs/BERT_TO_DIFFUSION_EDIT_20261019.md#当前存储代码与日志路径)。续训YAML、控制配置、源码、数据索引、Conda及checkpoint元数据直接使用当前挂载。禁止通过旧挂载别名或运行时前缀适配继续读取；路径改名须同步更新哈希依赖并通过原标准门禁。

2026-10-22 当前续训目标队列为用户明确指定的 `ZENITH_MLP2_TASK`（`q-20261009101456-4m9np`），Preemptible=false；[非闲时YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_diffusion_edit_resume28000_nonspot_zenith_20261022.yml)。直接CLI提交曾被队列权限拒绝；先前网页任务 `t-20261022172952-vshml` 已Failed，新网页任务 `t-20261022181600-wcfcx` 已Running，八rank从28000完整恢复并保存新checkpoint28025，详见[过程](/project_c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。控制目录 `/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022`，从最新完整≥28000恢复至累计200000，保留原训练计算、数据分片与W&B身份，源码路径及绑定哈希已迁移；模型／AdamW／scheduler／8-rank RNG及累计采样游标完整恢复。沿用8×A10080/max_length2048/global512记录=1024视图、micro48+48+32的原显存profile（allocated53.99/reserved58.80GiB；micro64 reserved77.61GiB未接受），未改模型计算。非闲时配置不启用实例回收重试，训练退出释放资源。保留原spot控制目录下独立checkpoint-28000作为恢复依赖，不可删除。恢复器使用独占锁与完整状态哈希校验；再次提交前核对唯一任务名与提交回执，避免重复。网页环境须把存储直接挂载到 `/project_c20250601/251105016`。每60秒只读监控已绑定当前网页任务，它依赖开发机存活。

2026-10-20当前八卡常驻容器为`t-20261020185703-8f7th`，共享控制目录[immune_training_container_c20250601_20261020](/project_c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261020)，资源[YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/immune_training_container_8gpu_20261020.yml)。训练退出恢复8卡standby；PAUSE_TRAINING只停训练，STOP释放容器。diffusion/edit从20500完整状态续跑的冻结包和原源码绑定见[恢复记录](/project_c20250601/251105016/project/dllm_test/docs/BERT_TO_DIFFUSION_EDIT_20261019.md)。旧20261007容器已终态，仅保留历史证据。


2026-10-20 diffusion/edit当前累计目标200000，接续产物位于`/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020`。使用[扩展启动入口](/project_c20250601/251105016/project/dllm_test/scripts/research/prepare_diffusion_extension.py)及其冻结包，恢复模型、AdamW、RNG与累计采样游标；旧10000阶段checkpoint与best2500继续保留，budget／LR扩展记录在provenance.extension。新W&B为diffedit287k200k1020，同版评估与当前停止路径见[续训方案](/project_c20250601/251105016/project/dllm_test/docs/BERT_TO_DIFFUSION_EDIT_20261019.md)。

2026-10-19 条件diffusion/edit的batch以视图计：global512原始记录×2互补视图=1024；八卡每rank128视图，micro48+48+32三个累积块，统一全局记录视图分母。max_length2048实测最大allocated53.99GiB/reserved58.80GiB，64视图reserved77.61GiB未接受。复用已有八卡容器，profile和八rank恢复已通过。新schema为`immune.fixed_chain.conditional_diffusion_edit.v1`，使用checkpoint绑定的diffusion_continuation_objective加载及evaluate_checkpoint入口验证。W&B仅3项train、6项核心valid与7项来源loss，其余诊断落地JSON；启动、同版评估及保存后停止的操作见[方案](/project_c20250601/251105016/project/dllm_test/docs/BERT_TO_DIFFUSION_EDIT_20261019.md)。

2026-10-19 BERT W&B 验证日志精简：[训练入口](/project_c20250601/251105016/project/dllm_test/scripts/research/run_bert_mlm.py)仅上传 `valid/loss`、`valid/mask_accuracy` 和 `valid/<source>/loss`；当前七来源共9项，原288项诊断仍完整写入本地 `metrics.jsonl` 与 `validation_step*.json`。训练及 probe 指标和最佳 checkpoint 选择保持原逻辑。修改仅对加载新版代码的进程生效，既有冻结启动包、运行中进程和 W&B 历史曲线不会自动更新。

下游结果排版按用户要求仅更新对应任务表，结果页顶部不另放汇总、状态或更新说明；具体约定见[下游结果维护](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md)。

2026-10-17 当前 BERT 为全有效token MLM：[配置与同版测试](/project_c20250601/251105016/project/dllm_test/configs/models/v7_bert_full_token_75m_300k_20261017/README.md)；[八卡YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_full_token_75m_v7_300k_8gpu_20261017.yml)；[冻结包](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_bert_full_token_v7_train_20261017)。输出为 `output/bert_full_token_v7_300k_20261017`，W&B为pressure_jian/dllm-immune/bertv7full300k1017。启动准备使用 `prepare_bert_mlm_v7_launch --objective all_observed_tokens`。原容器直接复用，YAML的queue012仅为等价新申请，不重复提交。新checkpoint必须使用绑定的evaluate_bert.py及新schema；结合分类隐藏relation真值。共享只读V7索引及W&B vendor仍在旧输出目录，不能随旧run清理；vendor依赖使用protenix_abtcr的Python，勿混用pllm解释器。八卡长度2048实测micro32×GA2×8/global512，最大reserved39.68GiB。

2026-10-16 历史 AA-only BERT 入口（2026-10-17已停止）：[配置与同版测试](/project_c20250601/251105016/project/dllm_test/configs/models/v7_bert_mlm_75m_300k_20261016/README.md)、[冻结启动包](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_bert_mlm_v7_train_20261016)、[输出](/project_c20250601/251105016/project/dllm_test/output/bert_mlm_v7_300k_20261016)。使用新V7 slots167/135、MHC-completed，随机从头训练75.6M，300000步/global512（micro32×GA2×8）；warmup15000，覆盖24.576数据轮。V7八卡A10080长度2048 profile：micro64 reserved78.49GiB余量不足，micro32最大38.47GiB。既有容器t-20261007153431-vhx2f复用，legacy v5目录／launcher名仅作传输，新包执行V7门禁。W&B waitma / pressure_jian/dllm-immune / bertv7mlm300k1016，SDK0.30及依赖仍引用[隔离vendor](/project_c20250601/251105016/project/dllm_test/output/bert_mlm_10epoch_20261016/vendor)，勿随旧启动目录删除。MLM checkpoint必须使用冻结evaluate_bert.py，不套用旧diffusion/EOS验证入口。


2026-10-16 历史V5 BERT启动（已被V7 300k替代）：`output/bert_mlm_10epoch_20261016`，V5七来源10完整epoch/global512；八卡A10080长度2048实测micro64 allocated71.66/reserved78.49GiB余量不足，采用micro32×GA2×8（reserved最大38.47GiB），global batch不增加。新账号waitma可见团队pressure_jian；本轮已成功向pressure_jian/dllm-immune在线写入并从API读回训练指标，不能把API默认字段models当成已验证的写入位置。W&B0.30.0及依赖仅安装在该实验vendor目录，旧共享环境不修改。独立MLM checkpoint必须使用绑定的 `evaluate_bert.py` / `bert_mlm_eval.py` 测试，不使用旧diffusion/EOS验证入口。

2026-10-16 实验范围约定：OAS-only 是用户指定的数据来源对照，不是主实验；不能将该分支的“只用 OAS”约束推广到主实验或表征预训练阶段。训练目标（MLM／生成／编辑）与数据范围分别登记，主实验沿用已确定范围并绑定具体版本；不得据调研建议自动切换数据或训练。[说明与更正](/project_c20250601/251105016/project/dllm_test/docs/REPRESENTATION_TO_EDIT_GENERATION_REVIEW_20261015.md#data-scope-correction-20261016)。

2026-10-16 特征缓存清理规则：用户授权删除成绩已记录在 RESULTS_SUMMARY.md 的对应缓存，并额外授权直接删除旧 8B 评测结果。非 8B 必须核对 checkpoint、任务/划分、输入协议及指标，保留指标、配置、来源和评测头；缺少对应证据的缓存继续保留。运行中和冻结计划的依赖始终优先保留，删除前复查新增作业。带 FEATURE_CACHE_REMOVED.json 的目录已移除特征数组，再训练评测头或复算表征需按原版本重新提取。[本次范围与清单](/project_c20250601/251105016/project/dllm_test/output/feature_cache_cleanup_20261016/README.md)。

2026-10-16 W&B 按训练任务选择项目：登录凭据只决定身份，不全局固定 `WANDB_PROJECT` 或 `WANDB_ENTITY`。用户明确要求取消旧统一默认项目名；六份现有 W&B 训练 YAML 均要求显式提供非空 `WANDB_PROJECT`，遗漏时在进入训练前报错，不回退到其他项目。每个研究项目指定自己的 Project，同一项目内用 `--run_name` 区分实验；`WANDB_DIR` 仅控制本地日志目录。云任务在其 YAML 的 Entrypoint 中、项目名检查之前明确设置，或通过平台实际注入到 worker 的环境变量设置；提交机前缀环境变量不保证传入云端。近期 `report_to='none'` 的训练仍需单独启用 W&B 集成才会上传，修改项目名本身不会启用上传。具体示例见 [训练配置说明](/project_c20250601/251105016/project/dllm_test/train_jobs/README.md#wandb-项目与实验名称)。

2026-10-16 W&B 凭据已按用户要求切换至 `waitma`（API 返回默认 entity：`models`）。本机 `/root/.netrc` 与训练脚本读取的 `/project_c20250601/251105016/.secrets/wandb_api_key` 已同步，权限均为0600；服务器身份验证及保存凭据一致性检查通过。后续读取这些凭据的新进程使用新账号；已有训练进程未重启，project名称及显式entity配置未修改。密钥不写入项目文档。[验证记录](/project_c20250601/251105016/project/dllm_test/output/wandb_account_switch_20261016/login_verification.json)。

2026-10-16 采样清理规则：已汇总不代表原始文件不再被其他实验使用。正在运行、已冻结或计划复用的依赖（尤其父实验 predictions.jsonl）必须保留；删除前重新检查并发任务与所有冻结计划的引用，出现新提交时停止清理并重核。小型对照预测优先保留，空间优先回收已聚合的大型比对明细。指标、配置、来源、训练数据和特征缓存不随采样一起删除。此次误删依赖按原SHA256恢复，不能改哈希绕过审计。[清理与恢复台账](/project_c20250601/251105016/project/dllm_test/output/sample_cleanup_20261016/README.md)。

2026-10-16 旧checkpoint保留规则：按用户要求，V7之前已用于测试、结果展示或对照实验的正式权重保留，其余旧训练权重可删除。此次范围仅历史 `protein_esmc_*` / `protein_llada*` 系列，最近native/fixed-chain/OAS/LoRA及V7仍保留。清理保留配置、训练曲线与来源；有 `WEIGHTS_REMOVED.json` 的目录只剩记录，历史top-k/resume清单不代表权重仍可加载。逐路径保留/删除依据见[清理台账](/project_c20250601/251105016/project/dllm_test/output/checkpoint_cleanup_20261016/README.md)。此处“测试权重”指用于实际评测的正式权重，不改变synthetic测试产物的清理规则。

2026-10-17 OAS-only 当前提交入口为 [从11000续训 r2 YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/v5_fixed_chain_75m_oas_resume11000_8gpu_queue012_20261017.yml)；20261016 resume YAML 与原 scratch YAML 均已 Failed，不能重提。冻结源码与 300k/global512 不变。W&B 必须在该 YAML Entrypoint 中显式设置 `WANDB_PROJECT=dllm-immune` 与 `WANDB_ENTITY=pressure_jian`，使用当前 waitma 密钥，并 `unset WANDB_RUN_ID`；原 scratch 为 `report_to=none`，没有旧 run 可续。后续测试先执行[版本检查器](/project_c20250601/251105016/project/dllm_test/scripts/research/check_oas_training_contract.py)，使用 checkpoint 绑定的冻结模型源码；[路径与命令](/project_c20250601/251105016/project/dllm_test/configs/models/v5_fixed_chain_75m_oas_20261015/README.md)。

2026-10-15 新 V7 数据约定：“不要负样本”只排除明确 nonbinding；保留未标注／unknown repertoire，不改标为 binding。当前已完成 MHC 补全的目录为 `/project_c20250601/251105016/project/dllm_test/data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_mhc_completed_20261015`（6,318,941 条，其中 67,648 条恢复双链 MHC 条件）。**新V7使用167/135槽位（含EOS），显式加载目录内manifest绑定的immune_config.json；不可套用旧187/154默认配置。** 4条超长记录整条隔离且保留审计，不截断。沿用现有 FixedChainCollator 与 CDR3β scaffold；保护实际 test/query，train-only 不因训练身份被排除。新目录未变 shard 链接到不可变父目录 `immune_v7_from_v5_no_negative_design_slots167_135_20261015`，两者应一起保留；既有训练不自动迁移。严格binding历史版本已撤回。[槽位与清理](/project_c20250601/251105016/project/dllm_test/docs/V7_FROM_V5_REBUILD_20261015.md)、[MHC 补全与验收](/project_c20250601/251105016/project/dllm_test/docs/V7_MHC_COMPLETION_20261015.md)。

2026-10-15 fixed reference轻链配对的默认入口、v2采样协议及旧实验复现方式见[任务§8.6](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md#86-fixed-chain--v7-采样器对齐复现)；后续计划按该入口生成。

本地评测优先（2026-10-13）：用户要求本地没有运行实验、GPU空闲时直接在本地测试，避免一直等待云队列。启动前核查GPU利用率、计算进程和现有实验；迁移同一评测前确认并取消重复排队任务，记录平台终态及本地进程身份。不要仅凭显存有余量认定GPU空闲，也不因此停止其他实验。实际资源核查与执行状态维护在PROJECT_PROCESS.md。

结果简洁版（2026-10-13）：[RESULTS_SUMMARY.md](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_SUMMARY.md)仅保留任务标题和结果表，由RESULTS.md派生；更新详细版数字后同步简洁版，数字仍以详细版为权威。

结果展示约定（2026-10-13）：模型最终成绩按任务更新[结果表](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)原表，保留历史模型并列；监督状态、诊断与执行明细写入[评测过程](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md)。AbBiBench结果页仅展示官方16 assay；不展示额外assay或其合并均值。

2026-10-13 评测约定：不能因模型不支持某种条件而改变baseline的输出目标或测试划分；候选预算、编号、聚合须显式绑定。CHIMERA执行入口及历史整链隔离规则见[CHIMERA现行协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。

2026-10-13 用户约定：现阶段评测数据不做去重，保留原始重复行、标签和出现次数；既有成绩不追溯改写。AB–Ag落地入口及限制见 [AB–Ag 当前原始行协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。

项目根目录：/project_c20250601/251105016/project/dllm_test

## 当前入口

| 内容 | 唯一维护位置 |
|---|---|
| 当前状态和下一步 | [PROJECT_PROCESS.md](/project_c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md) |
| 模型方案 | [BIOSEQ_MODEL_PLAN.md](/project_c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md) |
| 表征预训练到可编辑生成的调研 | [研究报告与实现边界](/project_c20250601/251105016/project/dllm_test/docs/REPRESENTATION_TO_EDIT_GENERATION_REVIEW_20261015.md) |
| 数据与格式 | [DATA_FORMAT_AUDIT.md](/project_c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md) |
| 实验里程碑与更正 | [实验记录](/project_c20250601/251105016/project/dllm_test/docs/EXPERIMENT_LOG.md) |
| 下游分数和比较口径 | [downstream/benchmark/RESULTS.md](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md) |

当前新训练数据是 /project_c20250601/251105016/project/dllm_test/data/prepared/immune_v5_receptor_completion。配置身份仍为v7_v5_layout_chain_global_20261007，名称不代表数据必须来自V7。

当前attention分段单位是prots/protd蛋白对象；split仅用于输出链解析。实现证据见 /project_c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/capacity302_regression.log。

## 操作规则

- 运行Python前激活pllm；测试仅保存报告／指标，不保留测试checkpoint和临时文件。正式恢复点不能按测试文件删除。
- 正式checkpoint按best3＋latest自动轮换后，核对被删除点的JSON档案与原生回执哈希一致；历史指标及来源证据保留。只读核对入口为 /project_c20250601/251105016/project/dllm_test/scripts/research/review_v5_checkpoint.py，不重新运行已通过的整模检查来验证档案。
- 使用绝对路径。提交训练走volc ml_task submit --conf及训练YAML；评测走评测YAML。提交／取消后更新进度表和简短事件记录。
- 当前已有作业使用冻结源码和数据；不修改运行中的绑定、不重复提交已完成或正在运行的任务，不将工作区新版本注入旧训练进程。
- 保留既有STOP／abort／监督停线；禁止自动移除它们或重复启动失败作业。先前暂停的独立CUDA诊断、序列注释／VNAR分支不因文档整理自动恢复。
- 原平台池按任务名immune-v7-pool-8gpu-20261005查询；状态查询需非交互JSON。2026-10-07用户要求暂停后已停止，不自动恢复原消融或占卡负载。
- 历史V5容器内生成训练使用 /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v3/launch_data_args.json 及固定源码副本；data-binding和SHA256为必填。原生门禁重新哈希全部85分片，不能只替换路径绕过。旧A4继续原V7门禁。
- 下游协议及指标保留各自版本、数据、checkpoint和生成条件，不能因切换新数据覆盖历史成绩。

- 2026-10-08用户授权下游评测优先使用闲时资源（Preemptible=true），也可本地GPU执行。迁移到本地前核实资源占用，并避免与已提交作业重复执行；不抢占当前训练或其他正在运行的任务。

## 文档保持简单

当前文档只写结论、状态、下一步和证据链接。同一实验更新原行；逐次轮询、心跳、重复进度保留在output原始记录中，不每次新增大段章节。只有数据／方案决定、启动／终态、结果和更正进入实验记录。同一事实不复制到四份md。

详细老约定和历史特殊情况：[完整历史](/project_c20250601/251105016/project/dllm_test/docs/archive/project_docs_20261007T061603Z/PROJ_GUIDE.md)。归档不是当前方案；有新的用户决定时以当前文档为准。

历史直接训练YAML /project_c20250601/251105016/project/dllm_test/train_jobs/v5_object302_scratch_8gpu_20261007.yml 对应已失败的首次启动，不能作为当前运行入口重提。当前global512=8×8×8；八卡3步synthetic启动检查已通过，证据 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/attempt2。启动前如发现明确training_ready=false则拒绝，不创建或伪造证书。训练权重不从synthetic或历史checkpoint初始化。

2026-10-07用户明确采用持续容器：/project_c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261007，资源YAML=/project_c20250601/251105016/project/dllm_test/train_jobs/immune_training_container_8gpu_20261007.yml。训练由容器内supervisor执行，每个票据一次；PAUSE_TRAINING只停训练，STOP才释放容器。子任务失败保留资源，正式训练前停止全部standby进程。后续不再将暂停实验解释为结束整个容器。

持续容器平台Running与训练完成/失败必须分开核实：同时检查实际进度和 /project_c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261007/results 下对应训练票据的退出结果。观察窗口到时退出不代表训练停止；完整训练需最终300000步产物与真实成功退出共同证明。

2026-10-07本版本显存实测：8×A10080、micro8、synthetic总长1218，峰值allocated最大4.48 GiB；实际用户指定global512配置保持。尚未对更大micro做显存搜索，不称micro8为吞吐最优。证据：/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/attempt2/startup_review.json。

2026-10-08用户进一步授权本轮37项下游评测使用非闲时资源；4组替代作业已设Preemptible=false，原闲时作业确认终止后才提交，不重复执行。其他评测仍优先闲时/本地。

2026-10-08用户明确：持续监督当前训练，训练退出后继续八卡占卡、保留当前容器。现有容器监督器已具备此切换：训练期间占卡全部停止；训练子进程wait回收、结果归档后，恢复每卡一个standby。失败也保留资源且不自动重试训练；STOP/资源到期仍按原规则退出，不能移除停线。已核对启动绑定的supervisor/helper/standby哈希一致。

当前只读训练至占卡观察入口：/project_c20250601/251105016/project/dllm_test/scripts/research/watch_v5_training_to_standby.py；本次有界50小时（180000秒），状态 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/training_standby_watch_20261008/latest.json，退出记录 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/training_standby_watch_20261008/exit.json。观察器不控制GPU、不重复发训练票据；以训练真实退出、state新心跳、八个不同PID和各卡新初始化日志共同确认占卡恢复。最终完整训练产物仍须另行核对。禁止重复启动同一个观察器，文件锁防重入。

2026-10-10新增270M实验配置目录 /project_c20250601/251105016/project/dllm_test/configs/models/v5_noesmc_270m_20261010。实际268,590,112参数；候选batch64×GA1×8/global512，先完成八卡A10080整模profile，显存数据尚待实测。旧75.6M冻结包及原门禁文件保持；新270M门禁使用 /project_c20250601/251105016/project/dllm_test/scripts/research/check_v5_270m_formal_parent.py 和 /project_c20250601/251105016/project/dllm_test/scripts/research/v5_270m_data_binding.py，仍强制原审核、85分片字节哈希、架构精确匹配及training_ready停线。

270M最终batch采用64×GA1×8/global512：八卡A10080实测长度1218，allocated42.79GiB/reserved48.88GiB；micro64已达保持global512时的上限，不额外增大全局batch。实际从容器内票据v5_270m_scratch_bs64_20261010.json执行。参考YAML /project_c20250601/251105016/project/dllm_test/train_jobs/v5_noesmc_270m_8gpu_20261010.yml不重复提交；运行包 /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_270m_train_bs64_20261010。

270M当前监督入口 /project_c20250601/251105016/project/dllm_test/scripts/research/watch_v5_270m_training_to_standby.py，状态 /project_c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/training_standby_watch_20261010/latest.json，退出结果同目录exit.json；本轮会话26867最多15天，不重复启动。保存核对入口 /project_c20250601/251105016/project/dllm_test/scripts/research/review_v5_270m_checkpoint.py --step STEP；与75.6M核对入口分开，实际参数数268590112、micro64/GA1固定检查。


2026-10-12历史固定双链生成训练入口：/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_20261012。模型为75.6M固定双链全链/全局attention，V5数据；不得使用旧object302或270M启动票据。停止当前训练用容器PAUSE_TRAINING，STOP会释放整个容器。


2026-10-14T12:53:45.346342+00:00 已确认LoRA全量后训练使用冻结原生八卡入口，global512=64×1×8；micro64已是该global/world组合允许的最大值。八卡S1136真实synthetic profile通过，optimizer状态随后丢弃，正式训练重新加载112000。按本次固定优化预算保持global batch，不为提高显存占用改变batch。[实测profile](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/final_gpu_profile.json)，[启动包](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/prepared_final_r1.json)。


2026-10-15 历史LoRA完整epoch续训入口：[准备与两个原生分段](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_lora_epoch1_20261015/prepared.json)，主段源码冻结于/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_lora_epoch1_20261015_stage1/source，输出/project_c20250601/251105016/project/dllm_test/output/immune_fixed_chain_lora_epoch1_20261015。复用现有八卡持续容器，累计目标33981；原10k checkpoint是只读父状态，新增200步预热/余弦阶段由绑定计划显式记录。原停止/失败不自动重试规则继续适用；当前状态以PROJECT_PROCESS为准。


### 2026-10-17 本轮第二层评测的执行位置

用户明确要求本轮第二层表征测试直接在本地运行，无需Volc提交；此会话授权优先于通用GPU评测提交规则。仅适用当前LoRA第二层实验，[绑定及本地入口](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#lora-layer2-representation-20261017)；原7个作业已取消，其他任务调度保持。


2026-10-18 用户继续指定fixed50000第二层对照，沿用本会话本地GPU测试授权；本轮无需Volc提交。[绑定](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#fixed50000-layer2-20261018)。

2026-10-18 用户明确BERT step107000下游表征测试在本机执行，不排云端资源；原5组已取消。13项串行、共享本机GPU，保留其他正在运行的评测及八卡训练。[执行与状态](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#bert-step107000-representation-20261018)。


2026-10-22 存储路径迁移已验收。网页入口使用 flow 环境实际路径，Conda脚本和SSL配置使用当前存储根。manifest/binding/index、provenance及receipt依赖哈希同步重算，原始模型张量和训练分片保持；迁移前文件另有备份。`--verify-only`直接执行标准数据门禁、父模型及完整恢复文件校验；训练stdout/stderr继续同步到平台和entrypoint.log。[迁移验收](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/storage_rename_20261022/verification.json)。


2026-10-27 1B八卡启动故障：新重试入口为 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/output/bert_v8_1b_test_20261027/launch_nccl_retry.sh`，先独立NCCL检查，再调用原冻结训练。Socket回退尚待八卡验证，配置仅限此入口；不全局更改环境或绕过原profile/smoke门禁。诊断记录见同目录nccl_retry/attempt_*，不上传W&B。详见docs/BERT_1B_TEST_20261027.md。

2026-10-28：1B测试新增隔离NCCL2.31.2入口 `output/bert_v8_1b_test_20261027/launch_nccl2312.sh`（项目根目录下）。仅此入口LD_PRELOAD固定校验的共享库，保持pllm安装包不变。旧Socket入口已在用户八卡任务中失败；新版本仍需八卡资格验证，原训练门禁完整保留。覆盖库证据在nccl_2312/receipt.json，不能以Python包freeze冒充实际NCCL版本。

2026-10-28：1B当前重试入口launch_deterministic.sh（output/bert_v8_1b_test_20261027内）；启用确定性计算，重新profile和smoke，保留原严格恢复阈值。执行前归档已有profile/smoke，已有train则拒绝启动；仅当前隔离入口生效。实际通信库和外部确定性模式均需结合revision记录，不能仅看冻结plan或Python包版本。
