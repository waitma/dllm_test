# V7 正式八 rank 入口（2026-10-02）

2026-10-03T01:27:30Z 最新源码为独立v3，611文件已根端审阅冻结，新身份原helper一次现行门禁passed59.723秒；v1/v2不改。新source manifest /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_manifest_v3.json（SHA992bc3c5718b71c379c7a2ee4f39d8e5866f974266500d2c2ec69d5ab4db700b）。下方v2/待审阅是当时状态；实际八卡更新/恢复仍待，formal未发布或入队，当前路径/原件见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_RESUME_OBSERVATION_20261003.md。

2026-10-03T01:10:58Z 当前接续：v2冻结和原helper门禁通过保留；wrapper修正版7项metadata控制及独立冻结完成，formal入口/计划/票据仍未发布。新公共driver增加原生恢复返回点只读状态观测，必要CPU5/5首次通过，根端审阅及新v3冻结待完成；详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_RESUME_OBSERVATION_20261003.md。源码准备不授予world8/实际恢复通过，v1/v2不改。候选/正式配方/统一实测batch待选择，原p02八项继续3750。

当前状态（2026-10-02T22:39:58Z）：driver v2 已冻结，验证尾批修复及原 helper 的新版绑定只读审核已通过。新增八 slot 启动 wrapper 已实现并通过 8/8 人工 CPU 检查，调用入口已实现并通过 4/4 人工 CPU 检查；详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_POOL_INTEGRATION_20261002.md。接线源码尚未部署，真实 plan/catalog 未创建，实际八 GPU 联合更新及完整保存恢复尚未验收，正式训练未启动。首轮 p01 已完成，原 p02 八项仍运行；候选模型与最终配方未选定。CPU mock/人工 checkpoint 证据不能替代 GPU 验收，冻结 v1/v2 及数据证书保持原件。

入口：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_formal.py`。只读审核 helper：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/check_v7_formal_parent.py`。测试：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_formal_protocol.py`。

## 外部 launcher 协议

外部 launcher 必须在调用 `main(argv)` 的同一进程中，预先完成真实 NCCL process group 初始化：world_size=8、global rank=0..7、每个进程只暴露一个互不重复的物理 GPU、LOCAL_RANK=0，并提供一致的 RANK/WORLD_SIZE 环境变量。driver 不初始化主进程组、不认领 GPU，不负责池调度。仅运行八次独立脚本不满足该协议。全局 rank 唯一性和物理 GPU 唯一性仍由 launcher 协调器负责。

必须显式传入 `--source-root /absolute/frozen-code --repo-root /absolute/real-repository --data-dir /absolute/final-parent --output-dir /absolute/run --variant NAME --seed N --micro-batch N --grad-accum N --global-batch N --max-steps N --source-manifest /absolute/sources.json --source-manifest-sha256 SHA256`。`micro_batch × grad_accum × 8` 必须等于所声明的 global_batch，例如 4×8×8=256。可设置 `--warmup-steps`、`--learning-rate`、`--eval-every`、`--save-every`、`--valid-per-source`；`--stop-after` 只停止当前段，不改变 cosine horizon。`--resume-from-checkpoint` 必须指向完整 native Trainer checkpoint。

source manifest 格式为 `{"files": {"/absolute/source.py": "sha256", ...}}`。同时绑定两个根：source-root 下的 driver、两个复用适配器及所有实际导入的训练模块；真实 repo-root 下的审核 helper、实际导入的门禁源码，以及旧适配器固定绝对路径读取的只读 metadata contract。训练用 driver 自身也必须从 source-root 执行，不能在公共目录执行后仅改 import 路径。两个根及审核实际执行源码摘要均写入 provenance。此清单是源码身份，不是新的训练/数据审核证书。

父进程在导入训练模块前拒绝来自 source-root 之外的 dllm/scripts/examples module cache，移除真实仓库 import 路径；只从冻结 source-root 导入模型和 Trainer。审核由 rank0 启动 `python -I -B` 独立子进程，cwd 为真实 repo-root，CUDA 隐藏，清除 rank/PYTHONPATH 环境。helper 原样调用真实 repo 的 assert_ready，不改 scope.ROOT、不签证、不改 ready、不输出序列；返回 metadata SHA 和实际执行门禁源码 SHA，rank0 校验后全 rank 接收并核对各自可见的 parent metadata。helper 全部所需实际导入源码必须列在同一 SHA 清单中，遗漏或漂移拒绝。

## 数据、模型与验证

仅接受 manifest SHA256 为 `28e3cbf8fb860c13b514a11d600579257a233fefe9f70ea070b042d04f8a424c` 的现有 final V7 parent。隔离子进程的原 assert_ready 包括全 shard SHA、audit 和现行 downstream scope/evidence；其他 rank 必须看到相同 metadata。探索性 100k/120k sample_manifest 不能伪装为正式数据。

显式支持四个 Refined 名称 `r20/r21/r24/r26`，以及 FactorialModel 的 11 个名称：`f000/f001/f010/f011/f100/f101/f110/f111/cv_segment_global/cv_late_fusion/fixed_gate_control`。Factorial 直接 import/构建，保存其原 schema/options；不经过旧 ResearchModel registry 分派。新 driver 的小型 Trainer 子类覆盖 sampler 并复用原 EpochRandomSampler，避免旧 sampler 对 factorial 名称查 registry 失败；最新修正版另为原 HF eval loader 构造临时设置 drop_last=False，并用 try/finally 恢复训练参数。训练 compute_loss、保存及恢复仍继承原实现。未预设获胜者，也未重跑已有 Factorial CPU suite。

训练用原 PreparedImmuneDataset；valid 按原 manifest source，每源用 seed42 固定选择至多 256 条（可显式改变上限，绑定到 provenance）。四 mask 25/50/75/100 的验证复用 TokenValidationTrainer 和 eight_rank_trainer_adapter_v1，不改变原训练 loss。全模型保持 A4、768/8/12/3072、maxlen2048、full_capacity256、fragment64，fp32 参数和 bf16 autocast。只做已有 masked reconstruction，不生成序列、不引入负例、不将已有正例描述为正负判别效果。

训练 sampler 继承 EpochRandomSampler，data_seed=seed，分布式划分交给原 Trainer/Accelerate。训练保留 drop_last=True；验证单独 drop_last=False，由原 Accelerate 为逻辑 rank 补齐 batch 后在 gather_for_metrics 去除尾部重复。不能只用 HF output.num_samples==len(dataset) 证明覆盖，因为该 num_samples 来自数据集长度。global_batch 指完整 accumulation window，训练 epoch 尾部可能存在不足 GA 的窗口，provenance 明确记录这一点，不能将 max_steps×global_batch 无条件声称为精确曝光。后续运行验证应报告实际批次/曝光。没有增加特殊功能优化或注释分支。

## 保存和恢复

训练继承原 ResearchTrainer→ImmuneTrainer，评估只用已有 factory 包装。共享 provenance、模型、optimizer、scheduler、Trainer state 由 rank0 写；native HF 每个 rank 写自身 `rng_state_0.pth` 至 `rng_state_7.pth`。on_save 在全 rank barrier 后，逐 rank 从自身实际 named_parameters 获取参数名、shape、dtype、原始 bytes 摘要，要求八个摘要完全一致；再对比落盘 model.safetensors 同名参数摘要。sidecar 分别保存八 rank 参数证据和落盘摘要，共享文件 hash 相同本身不是参数同步证明。随后核查全部 native 文件、optimizer state、scheduler step、Python/NumPy/CPU/CUDA RNG、共同配置和源码 SHA，再由 rank0 写完整文件 inventory sidecar。正式 horizon 与停止段不受旧适配器限定的 1000/5000 stage 约束。

旧 checkpoint_adapter_v2 的完整 callback 固定了历史合成日程，本版仅复用其通用 `_files`、`_atomic_json_once` 等文件处理函数。正式 callback 是新代码，尚无实际八 GPU 保存恢复证据。恢复必须先通过 sidecar 全文件 SHA、binding 和 step 检查，随后交给原 `Trainer.train(resume_from_checkpoint=...)` 恢复权重、optimizer、scheduler、每 rank RNG 和数据跳过位置。p01 的 latest.pt 不兼容，也不接受仅权重续训冒充完整恢复。

## 验收边界

本次仅执行新增协议测试及必要 CPU 人工接口检查，不复跑旧 suite。helper 初版及 v2 源绑定均已实际 passed，包含尾批修正版的两根源码清单也已冻结。当前还必须完成：外部协调器接线与真实物理 GPU 唯一性；选定模型/配方/日程；在该新源码上一次有界 world8 联合 optimizer 更新、四 mask/source tail 汇总、实际参数摘要一致性、完整保存与恢复续步；对应 microbatch 的实际显存与资源剩余时间确认。下述历史 9 项协议检查针对之前字节，不是修正版全部验收；新两项 loader 检查也不替代联合 GPU 验收。旧 V4 stage 或旧 CPU 证据不替代这些新集成检查。

本轮结果：新 stdlib 协议测试 4/4 通过；另用 CPU 人工 `torch.nn.Linear(1,1)` 构造原 TokenValidationTrainer 的八 rank 包装类，确认继承 compute_loss、_save、_load_rng_state，model_accepts_loss_kwargs=False，并正确拒绝未初始化的 world1。该人工检查没有 forward/backward、训练数据或 GPU 操作。运行环境产生已有 Linux 5.4 kernel 提示；它不是八 GPU 实测结果，也不构成重启/升级授权。

冻结接口增补：仅增加 helper 子进程隔离协议、两根/module-cache 拒绝、11 候选 CLI、真实门禁源码闭包/漂移，以及 CPU 人工 Linear 参数与落盘摘要差异检测。没有执行真实数据审核、Factorial forward、GPU 或旧验收套件。此前 4 项测试记录保留为历史；本版 driver 专用测试 9/9 通过，3 个 Python 文件 AST 解析通过。

证据归档（未重跑）：原结果此前仅在工具输出中，现原样保存为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_driver_evidence_20261002/protocol_9_tests_and_ast.stdout.log` 和 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_driver_evidence_20261002/cpu_original_trainer_construction.stdout.log`。同目录 `receipt.json` 记录原始命令、工具 chunk ID、exit code、来源和边界；CPU Trainer 构造属于较早接口检查，不是最终 driver 的八 GPU 验收。本次未修改冻结 Python 源码。

## 独立 formal transport 源码准备

新增 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/formal_pool_transport.py`，源码 SHA256 为 `17bf275958f2cdcbd2a9432dfd2fd0f87ad6df61203b99c558efbae96f7d8346`。本项未修改 driver/helper、任何冻结包或 source_manifest_v1，未发布、入队、创建真实 intent/reservation/group/listener，也未运行 GPU。

`prepare(spec)` / `validate_intent(intent)` 只做内存结构检查。正式 intent 绑定明确的模型选择记录、实际只读 final-parent 审核结果文件、冻结 driver、两根 source manifest、完整 driver argv/预算、8 个唯一 worker/job/GPU mask/UUID/进程身份及共同 host/boot。driver SHA 由 intent 显式提供，不硬编码某一 driver 版本；新 v2 清单必须有原 helper 真实运行形成的匹配审核证据，不继承或伪写 v1 passed。原有 v1 的验证尾批问题及其修复/新冻结由根端另行处理，本 transport CPU 结果不使 v1 获得正式发布资格。

未来 pool 调用 `member_parent(intent_path, intent_file_sha, rank, reservation_path, reservation_file_sha)`。intent_file_sha 是文件字节 SHA；协调器 work_binding 和 reservation 的 intent_sha256 是 canonical JSON digest，二者需区分。调用进程必须是预留 worker 本身或其直接 job 子进程，环境中的 `FORMAL_POOL_WORKER_ID`、`FORMAL_POOL_JOB_ID` 和物理 CUDA mask 必须匹配。全 8 个实际 worker 进程必须仍活着。真实 GPU UUID/PCI/NUMA 在未来 parent/child 执行时才由旧硬件观察函数核验，本次 CPU 检查未调用它。

外部 reservation 格式由 `validate_reservation` 明确：schema=`v7.formal.all_eight_pool_reservation.v1`、status=`reserved_all_eight`，绑定完整 members/intent digest/host/boot，previous_jobs_finished=true、exclusive=true，reserved_monotonic 距调用不超过 30 秒，并携带有限 group_deadline_monotonic。**八 worker 原子 reservation 的生产逻辑及实际组发布/队列接线尚未实现。** 不允许先塞部分 rank 等其他实验完成；必须由未来 pool 整组确认八 worker 全部可用后，一次性预留并发布。

transport 使用 SHA 固定的 V4 `eight_rank_coordinator_v2.py` 通用 scope/work binding；仅复用 cold-rank 的 held loopback FD、`initialize_explicit_store`、只读状态监督、实际 wait/进程组回收函数。没有修改旧 synthetic FALSE_FLAGS、旧 WORK_BINDING 或旧 stage。旧监督证据中保留的 false 字段只代表旧通用组件不授予训练许可，不会被翻成 true。rank0 在未来真实调用时持有监听 FD 并通过 pass_fds 给隔离 child；各 child 保持原物理 mask、LOCAL_RANK=0，经同一 NCCL startup 后调用受 SHA 绑定的冻结 `formal driver.main(argv)`。

只读 group_status 锁超时仍仅对原异常类型/目录/contract 绑定做有界重试，不刷新失败观察时间或心跳；30 秒 freshness 不延长。真实失败发布绑定 intent 的不可逆 abort marker，并对已建立的 coordinator 广播 abort。父进程保留实际 child 身份，负责 wait、TERM/KILL 和后代回收、原 coordinator exit receipt。child 保留原心跳和进程组清理顺序。实际监督、FD 继承、NCCL 初始化、清理和八 rank 参数/保存恢复仍待未来一次有界 GPU 整合验收。

新增 CPU 协议检查 8/8 通过，覆盖预算/30秒界限、模型未选择拒绝、worker/job/GPU 重复、部分/过期/仍忙的 reservation 拒绝、单物理 mask 和 LOCAL_RANK0、旧 synthetic schema 拒绝、不可变 coordinator 对通用 contract 的接收。全程使用人工内存 metadata，并将 socket.socket 替换为调用即失败；没有运行旧 Gloo/500 suite、真实数据审核、进程监督或训练。直接捕获日志：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_transport_evidence_20261002/self_test.stdout.log`；范围和真实命令 receipt：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_transport_evidence_20261002/receipt.json`。

## 2026-10-02T21:07:58.922239+00:00 正式入口隔离审核首次真实执行通过，p02作图已登记

冻结正式模型/Trainer与真实审核资源的分离接口已首次执行：原helper在python -I -B、CUDA隐藏的子进程调用原assert_ready，真实校验final V7所有分片及现行scope/evidence，54.28秒返回passed；parent清单与两证书SHA前后不变，实际门禁模块39份。没有创建审核证书或改ready、恢复注释或启动模型。源清单1137项、SHA 2c5a2a8a5ba9841dbffc569f1318a09d2497eed39a9ccc7ba5ed8e99104172b1；事实原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_first_execution.stdout.json 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_first_execution.json。此处是新隔离执行接口的首次实际核验，非重复签证。

root已读取原工具stdout归档，正式driver新增协议测试9/9、3AST通过；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_driver_evidence_20261002/receipt.json。正式入口源已冻结，但launcher还没接通，联合NCCL更新、实参同步、保存恢复实际GPU证据仍缺，formal_training_started=false。正式长程预算需结合原资源剩余时间设置可恢复阶段；不允许因全部小实验排完便把8份独立模型视为world8联合训练成功。

p02作图 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_context_ablation.py 新4项纯数学/结构控制通过；首次读真实22项账本输出JSON及4CSV，awaiting_shared_evaluation、missing22、figures=[]。没有用假效果值画图。数据列包含source/context分母、关系CE/accuracy及正交G/S/L对比，完整图页的真实渲染待共同评估出现；说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_CONTEXT_ABLATION_FIGURES_20261002.md。原p01监督完成后续看p02，至多12小时或22项全终态，无新增OS监听/平台任务。

## 2026-10-02T21:29:33Z 验证尾批最小修复，worker CPU 两项定向控制通过

只读审查确认 worker 原 HF get_eval_dataloader 使用 TrainingArguments.dataloader_drop_last；冻结 factory 直接调用该方法、没有自行构造尾批。因此旧全局 True 会在 Accelerate 八 rank batch 分片时丢弃不满一组的验证记录，而 output.num_samples 仍可能报告原数据集长度。

只改公共 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_formal.py 的 FormalBaseTrainer.get_eval_dataloader：保存原参数，try 内设 False 后调用原 super 方法，finally 无条件还原。训练仍 True。公共修正版 SHA256=`a751eea2893aea463ab23798b0f685edbf1333a3da8e7a287aa4a7a4f453c94d`；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v1/scripts/research/run_v7_formal.py 仍为原 SHA256=`24735d705cd1b847a6bd52b5be005e67dfebabe12dc816a0de395edb51d06d13`，未修改冻结包/helper，也未重新冻结或部署。

新测试 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_formal_eval_tail.py（SHA256=`766cb9510c14b5d0bc290cc387df8dbfb51f7f87f4cfb685e9f9c73bbf814aba`）从实际 driver AST 提取原 override，再调用实际 HF Trainer loader，没有复制替代实现。先激活 pllm 后显式使用 worker Python /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python，仅执行新增 **2/2 cases，14.569 秒**；没有再跑本机环境或历史 suite。实际库版本 Torch 2.8.0、Transformers 4.48.1、Accelerate 1.14.0；源码真实 resolve 到 flow 环境目录，实际路径与 SHA 已完整记录，未误写成本机 pllm 库证据。

覆盖证据：对 13/17 条纯整数 ID，分别构造原 HF eval loader，再用 8 个真实 BatchSamplerShard/DataLoaderShard（micro4、split_batches=False、even_batches=True）获取 rank-major 输出；每逻辑 rank 都恰好 1 次迭代。手工拼接这些实际 shard 输出以替代网络通信，再调用原 Accelerator.gather_for_metrics，使用实际 active loader remainder 13/17 去掉补齐重复，结果精确为 0..12/0..16、分母 13/17、无漏行或重复。训练 loader 33 行仍产出 8×4=32、drop_last=True；原 HF 无 eval_dataset 异常真实抛出后参数仍恢复 True。自有临时输出目录已自动删除。

stdout 原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eval_tail_20261002/worker.stdout.log；包含每逻辑 rank 原始 ID、拼接与去尾 ID、分母、版本、安装源码 SHA 及限制的 JSON /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eval_tail_20261002/worker/result.json。已有 Linux 5.4 kernel 提示原样保留。

边界：1 个 CPU 进程中的 8 个真实 sampler 对象，不是 8 进程 collective，也不是 GPU/DDP 验收；只构造人工 Linear，不做 forward/backward/train，不读真实数据，不初始化分布式组或 CUDA。不能据此宣称正式模型尾批/保存恢复已获实际八卡通过。


## 2026-10-02T21:35:07.714476+00:00 验证尾批修复、新源码冻结与当前接续

根端已读原始证据：正式入口的验证 loader 现在单独使用 drop_last=False，并在 finally 恢复训练的 True。worker 环境（torch2.8.0/transformers4.48.1/accelerate1.14.0）新增 2/2 人工整数检查通过：真实 HF loader、8 个逻辑 BatchSamplerShard/DataLoaderShard 及原 gather_for_metrics 完整保留 13/17 行；训练 33 行实际保留 32，异常后参数恢复。没有 forward、训练、真实数据或分布式通信；临时输出自动删除。结果 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eval_tail_20261002/worker/result.json 与同级 worker.stdout.log。旧 9 项未重复执行，不把它们算作新版全验收。

独立 transport 的 8/8 新 CPU metadata 检查通过，根端已读原日志 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_transport_evidence_20261002/self_test.stdout.log。复用已冻结通用协调器、held FD、进程监督与清理，旧 synthetic 标志不变；测试禁用 socket。尚未实现实际 pool 的八 worker 一次预留/发布接线，未创建 intent/reservation/group/listener 或新 GPU/平台任务。不能把 CPU 协议通过称作八 GPU 启动通过。

新版冻结 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v2，driver SHA a751eea2893aea463ab23798b0f685edbf1333a3da8e7a287aa4a7a4f453c94d，原 v1 字节保留。清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_manifest_v2.json，SHA 566bec35f5ee0ea65047a3e69d864e729cff2de2ad1a861929401551cc368799，共 610 文件：570 冻结源码、39 份已实际执行审核源码和固定 metadata contract。没有继续绑定与审核无关的可变研究文件；实际审核执行源码仍逐项拒绝遗漏/漂移，数据门禁未修改。

因为修复使源清单改变，原 helper 对新版绑定真实执行一次，58.79 秒 passed，stderr 空；原 parent manifest 与两证书 SHA 保持相同，未签发或改写证书。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_v2.stdout.json，执行记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_v2.json。不可把旧 passed 直接换上新 SHA，本次是有实际源码修正依据的绑定核验。

21:34:15 原 p01 八项 2800–2875/3125，共同 2750 验证，未观察到失败/失联；p02 已入队 22 项仍待空卡，目标保留用户选择的已有 binding 正例/context 序列零重建 loss。正式模型与配方最终选择、八 worker 统一启动接线、实际联合更新/完整 checkpoint 保存恢复仍待完成。正式训练未启动，原 Goal 保持 active；不重启容器/worker/监听，不重复提交，不恢复注释停线或独立 CUDA 算子。
