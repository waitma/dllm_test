# 固定 R20 人工 GPU runtime 验收 producer

当前状态（2026-10-01 03:26 UTC）：下文14项是首版历史记录。后续实现已完成两环境各24项并复审发布，见[当前实现](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PILOT_GPU_RELEASE_PRODUCER_IMPLEMENTATION.md)。实际D01 v1因oracle701.6219秒超出240秒阶段预算失败，无通过标记；正在准备事前预算修订的不可变v2，见[整合与失败记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PRETRAIN_INTEGRATION_20261001.md)。旧记录不认证新源码。

2026-10-01，首版实现的两环境新测试各14项通过。随后发现根线程正在并发扩展producer的容差/来源/预算合同，本分支已停止修改源码；以下14项结果不能认证正在变化的当前版本，须由接手线程完成扩展后重新绑定来源与验收。未运行 GPU、未提交任务、未读取真实数据，没有生成通过证书。

实现：[pilot_gpu_release_producer_v1.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pilot_gpu_release_producer_v1.py)。测试：[test_pilot_gpu_release_producer_v1.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_pilot_gpu_release_producer_v1.py)。本轮新增文件不修改39项模型来源、已归档524文件、旧指标或pilot执行器。

窄接口为 `produce(profile_dir, output_dir)`，固定 `r20_refined_control`。只能从父 manifest SHA `46399204c7018d00726e3ae08142bd4eb28475c8a9a2991b3a5c90725c8824f7` 对应 524 文件快照派生的 runtime 启动。先以标准库核验并隔离 runtime，再导入其原 `make_model`、`training_arguments` 和修正后的 `TokenValidationTrainer`；不改 ROOT，不覆盖父快照，不调用真实 `build_trainer` 或数据门禁。

独立首窗 oracle 暂停梯度裁剪，以免裁剪掩盖 GA 倍率错误。真实 HF 窗口观察输入和 RNG，在更新前用同一未更新权重逐 micro 重放 `loss/GA.backward()`，逐参数比较未裁剪梯度。此阶段释放后重新构造模型，正式三个 global256 完整窗口保留原裁剪、AdamW、BF16、自然采样和 10000 步 LR horizon，并单独测量显存。三个窗口仅验软件，不要求 loss 下降、不宣称学习质量。

oracle与正式阶段都使用同一完整768条人工训练集合、相同初始化和data_seed；oracle只提前停在首个完整窗口，不换成256条小集合而改变随机采样。报告逐micro核对两阶段首窗的完整输入哈希、shape和计数相同。

评价调用真正四档 `evaluate()`；观察型子类的 `compute_loss` 只记录实际评价batch后调用原方法，`evaluation_loop` 只在原调用返回后复制总量。`train`、`training_step` 和 `accelerator.backward` 使用实际框架实现，没有复制训练或评估算法。人工来源固定为7条短混合记录，以及eval_batch+1条满长压力记录，大小不同，覆盖双容量、短输入 padding、单/双目标、relation 有/无；核对 CPU 计数/聚合、None 和callback前后Python/NumPy/Torch CPU/CUDA RNG。训练与评估都保留实际GPU输入形状、槽位、FF输出层观察到的autocast/dtype、FP32参数/AdamW状态和峰值。完整 GPU 验收尚未发生。

父进程只使用固定本文件的 worker 启动，复用已验收进程组清理工具；worker 原始报告不是通过证书。只有真实 GPU worker 正常退出、清理完成、全部数值/计数/来源校验通过后才可写末尾 `GPU_RELEASE_COMPLETE.json`。CPU mock 只测试拒绝和接口，不签发该文件。GPU producer 与未来真实 pilot wrapper 分开；原双数据证书仍是实际训练的前置要求。

## 输出与固定范围

`request.json` 绑定固定计划和完整runtime来源；`worker_report.json` 包含人工fixture内容哈希、逐参数梯度误差、三个窗口/实际batch/AdamW状态、四mask原始总量及全部指标、来源和运行环境。父进程将正常退出/清理与原始报告SHA写入 `gpu_release_summary.json`，其原字段兼容已归档GPU合同，扩展字段为 `producer_schema`、`producer_sources`、`request_sha256`、`raw_report_sha256`、`child_process`。最后才写 `GPU_RELEASE_COMPLETE.json`；OOM、超时、漂移、计数或数值失败保留日志/FAILED，不能缩小输入后继续沿用原身份。

父manifest固定为本轮524文件版本。实际producer和未来wrapper都必须作为新路径被runtime的additions map绑定，无法从活动仓库直接运行生产分支。整个child预算1200秒；只允许1张可见CUDA卡、固定R20、profile推荐micro、global256、三个完整窗，无data/resume/架构/步数任意参数。任意其它变体、尾窗口、恢复训练、数值学习质量、分布式训练均未由此脚本认证。

梯度容差在计划中预先固定为 `atol=2e-6, rtol=2e-4`，对所有有梯度参数逐元素检查；不会失败后放宽。oracle本身使用首个warmup更新，其LR为0，因此不要求该窗改变权重。正式三窗必须有有限非零参数变化、AdamW step全部为3、scheduler horizon仍10000，且保留原clip=1、warmup=100、AdamW配置和自然采样。oracle的模型、优化器与临时梯度释放并确认GPU张量归零后，才重新构造正式模型/reset峰值。

## CPU 验收与限制

首版最后一次pllm（Transformers4.57.6）**14 passed，9.92秒**；worker对应环境（4.48.1）**14 passed，10.10秒**。两者使用tiny RefinedModel、实际whole-layer checkpoint、真实Trainer及AdamW；global batch仅8，GA=1/2/4，分别验证首窗逐参数manual对照和三个正式窗口，未冒充global256的GPU测量。每个case比较24个参数，CPU最大梯度绝对差均为0，oracle与正式阶段首窗输入相同；这些值仅适用于本次CPU人工case。另覆盖活动ROOT拒绝且不创建输出、不完整窗、错误缩放flag、非有限loss、CPU/来源不符报告拒绝和评价计数损坏。四档评估真实执行，含None和会消耗RNG的callback。日志中旧模型的autocast CPU弃用提醒不影响通过；冻结模型未为此改动。

并发扩展交接点：review时当前 `fixed_plan.gradient_tolerance` 已扩展为 `atol/rtol/relative_l2/scale` 四键，而当时的 `exercise` 仍只生成前两键，生产validator要求完全相等，因而会拒绝实际成功worker。顶层docstring新增的 `--runtime-root` 当时也尚未在parser注册。这是正在编辑中的版本，不能通过删字段掩盖尚未实现的扩展。复审未发现首版核心HF循环、同权重/RNG重放和真实四档evaluate方向的额外阻断；接手线程仍需最终正向worker proof合同测试，避免仅有拒绝case遗漏字段不一致。root消息工具连续返回active-writer错误，交接证据留在本目录并通过final返回。

证据目录：[immune_pilot_gpu_producer_cpu_20261001](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_gpu_producer_cpu_20261001)。CPU通过不证明实际BF16梯度、GPU显存、1200秒预算足够、完整runtime新入口可部署或真实数据已经认证。正式runtime仍须等两个新增文件冻结后构建，再执行人工GPU验收；本轮没有创建runtime、数据链接或GPU通过文件。
