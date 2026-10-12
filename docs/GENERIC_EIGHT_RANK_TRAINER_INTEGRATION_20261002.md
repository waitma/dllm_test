# 实际八rank Trainer 接入与数值 CPU 验收

## 当前结论（2026-10-02T09:54:10.064871+00:00）

两环境各一次实际八CPU/Gloo numeric Trainer主case及独立复核全部通过：worker19.747秒、pllm22.008秒，连续3步与原生checkpoint2→3恢复比较，参数最大误差0、Adam最大1.862645149230957e-9、scheduler0；16 rank原生RNG及下一批IDs一致。384组source/mask/rank-phase、2256指标由stdlib独立重算，误差0；独立float32损失oracle最大1.375491803790041e-8，低于原1e-6。16 child均rc0/wait/进程组清理完成，自有numeric临时checkpoint及二进制剩余0，18旧保护文件保持。

此验收仅证明纯数字CPU fixture的新增训练器工程，不证明免疫生产Trainer、真实GPU-DDP、1000→5000续训、架构生物效果或数据准入。已清理二进制未被独立review重读，证据来自执行时保留的原值/哈希及冻结source；保留全部原失败/raw/source/log与实际训练恢复状态。

候选R20/R21/R24/R26代码已实现，最终架构选型未完成。生产单模型八GPU入口、原pool八rank组调度、正式外部审核资源参数在preflight/selection/build_trainer/execute/resume的贯通及新来源实际GPU完整恢复仍未完成；原controller launch拒绝实际执行，旧pilot/metrics仍限world1、生产checkpoint仍单份rng_state.pth，冻结包十资源目录缺失。新CPU模块未接入生产包，不能将通过声明移给新GPU来源。

下一步只收尾当前八项500的实际终态并按各自原verify_diagnostic(500,require_learning=True)核实，再完成必要通用生产调用链及八rank冷入口；不新增架构波次、不重复已通过CPU/旧模型验收、不启动无关验证。现八GPU仍被原任务占用，任何必要同模型GPU组验收等待全部原worker空闲并满足新来源原门禁；真实数据训练需有效当前审核和全部前置满足。

[本轮根端验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_CPU_device_repair_20261002/actual_CPU_conformance_root_acceptance.json)；[最新数据与原任务状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_CPU_device_repair_20261002/user_training_readiness_after_strong_reviews.json)；[必要八卡接入缺口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EIGHT_GPU_TRAINING_INTEGRATION_GAPS_20261002.md)。

以下带日期内容为历史阶段记录，旧“待执行/复核中”不代表当前状态。


本轮继续原Goal，不新增模型候选、不更改旧source或既有门禁。上一Goal回合为progress（八rank通用接口实际实现/两环境首次四控制通过）。本阶段将已验协调和checkpoint元数据接口接入真正HF/Accelerate Trainer；仍只纯数字CPU fixture，不读取/训练任何生物序列或正式分片，不执行注释/审核/selection/GPU job。

四新source以内：checkpoint callback（A）、distributed validation Trainer factory（B）、numeric child（C）、stdlib CPU parent（root）。独立预算每环境一次8真实CPU rank conformance、600秒；先连续3步，再实际保存2步并恢复到3步，原optimizer/scheduler horizon10000、LR1e-4/warmup100/cosine及eval100/save100保留；2/3步的save/stop属于显式验收额外instrumentation，不声称1000或5000训练完成。四mask/two numeric source13/17的尾批计数检查原gather_for_metrics去重，已经全局totals不再reduce。

所有CPU子进程effective CUDA_VISIBLE_DEVICES为空，HF保存RNG不访问GPU；coordinator visible index0..7只为CPU fixture声明，不能冒充GPU硬件验证。原八GPU任务、worker、容器及监听不动，新GPU/平台提交取消重启为零。旧WORLD_SIZE1入口保持，实际新的world8 factory必须运行真实distributed而不是checker后调用旧evaluate。

真实HF保存8份rank RNG/optimizer/scheduler/Trainerstate，callback全rank保存屏障、rank0唯一公共metadata/hash sidecar、实际8acks；恢复前重读完整状态/来源、再调用nativeHF实际RNG restore。安装源已核worker HF4.48.1/Torch2.8有NumPy safe_globals，不预先放宽torch.load或全局绕过。CPU的真实数值文件/参数比对仅支持通用Trainer conformance；当前formal资源链、审核/真实数据、immutable源码准入、真实GPU/DDP/最终架构仍未完成。原单卡legacy sidecar不能直接称DDP已兼容，toy完整状态validator与formal gate各自明确，不能假造旧生产证明。

当前状态：四个私有源码实现中，未sourcefreeze/实际CPU执行/生产发布。原8项500人工GPU学习仍实活，最新本轮起始CE200、其余七项75，无终态。最终V7 ready字段/双审核仍缺，所有data-scope-source-selection-learning-GPU-full-resume门禁、安全停线、独立CUDA暂停保持。

本轮起始时间 2026-10-02T08:44:04.665526+00:00。预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/execution_budget.json；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/existing_eight500_readonly_start.json；旧接口证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_integration_20261002/CPU_control_results.json。

## 2026-10-02T09:04:38.182055+00:00 当前可接续的真实状态

Checkpoint callback与Trainer factory已实际交付；parent setup异常与leader已退残余PG漏清理两项已修，独立只读复审确认。三文件AST可解析、9保护旧源不变；numeric child仍实现中，四文件未冻结，实际CPU验收0/未生产接入。不要将源码复审记为DDP或恢复通过。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/static_source_review_current.json；Goal实际工具累计 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/goal_actual_after_status_and_source_review.json。下一步只完成child并做必要保存恢复及尾批数字CPU验收。

## 2026-10-02T09:16:09.780145+00:00 四文件已交付，等待独立复审后首次执行

唯一numeric child已落盘570行，root实际通读主case/原native restore观察/安全RNGload/16文件gate/4mask×13与17尾批/模型与Adam及scheduler原数值/main-parent API；四文件均AST解析，9旧源保持。A专审checkpoint/native load，B专审实际HF启动与评价collectives；尚未freeze/CPU执行/发布。当前budget不变，每环境一次600秒CPU case尚未用。

root曾误把headnode的/proc PID查询当GPU空间，实际host为di-20260121220916-9rvmj，与remote容器不同，因此0个本地PID并不证明远程任务退出；该inconclusive原件保留 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/local_proc_namespace_probe_inconclusive.json，修正说明在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/existing_eight500_verified_live_during_fixture_integration.json。真实只读平台get返回原t-20261001033327-fmnql Status Running，证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/platform_task_readonly_current.json；远程worker poll-gated metadata和已有trace仍更新。没有重启/提交/取消或新GPU任务。V7全部原审核/ready及安全停线继续。

## 2026-10-02T09:19:16.320175+00:00 四source冻结与首次CPU准执行

root AST/逐行及A checkpoint-native restore/B CPU启动与指标collectives/parent清理独立复审完成，实际四源码SHA固定于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/source_freeze_before_CPU.json；budget79cef8da/9保护旧源不变。即将依次worker、pllm各一次原numeric 8rank case，600秒parent总界；CPU初始processgroup不会直接使用TA ddp_timeout60，不能假报已验证该60秒。CPU当前尚未执行，任一失败保留日志与真实exit/cleanup，不自动重跑。原GPU/真实数据/生产发布0，所有原门禁与停线保持。

## 2026-10-02T09:27:18.509373+00:00 worker实际CPU失败与最小修正决策

首次8CPU在21.64秒失败：原HF4.48.1 world8恢复optimizer使用args.device=cpu:0，原Torch CPU storage dispatch只接受cpu。原3连续/2保存与尾批执行，恢复前集体gate已返回，尚未执行native RNGrestore，不称整轮通过；二进制已清理、阶段原数值因仅成功return留存存在缺口，明确不补造。8child实际reap/PGgone，自有fixture删除，源/sidecar/日志/错误原件保留 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/actual_CPU_failure_root_review.json。v1预算worker1已用、pllm0，pllm只从源预测同路径问题，不记录实测失败，旧版本不自动重跑。

最小新来源预算只2new source，复用冻结callback/factory/parent，不改v1/安装HF或Torch/native save-load/safe allowlist/数值阈值。NumericTrainingArguments.device getter仅CPU规范化；逐phase原数值及真实pretrain verification立即存JSON。每环境一次同case/600秒/GPU0，预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_CPU_device_repair_20261002/execution_budget.json。新case未实现/未执行/未发布；所有原数据门禁和安全停线保持。

## 2026-10-02T09:32:44.629446+00:00 CPU最小修正版两源冻结

只新child/device及raw修正、root薄入口，共2source；冻结A/B/parent三源复用且旧v1/失败原件保持。NumericTrainingArguments.device限定numericCPU guard，原cache/Gloo/native optimizer/scheduler/RNG/safe allowlist与阈值不改；子类通过真实模块globals/qualname按原HF pickle，未替换序列化。成功phase及真实verify_resume返回立即JSON保存，即使后续失败也保留。root实际diff/AST/A/B独立复审通过，来源冻结 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_CPU_device_repair_20261002/source_freeze_before_CPU.json；即将首次repair worker→pllm，各一次600秒，CPU未执行/通过，GPU/生产/真实数据0。

## 2026-10-02T09:35:11.198509+00:00 worker修正版实际通过，pllm进行中

worker first case 19.747秒，model0/Adam1.862645149e-9/scheduler0、native8RNG及tails通过，实际8rc0且own fixture已清理；raw保存完整。pllm首次已启动53613，未终态声明。此为真正numericCPU Trainer证明，仍不是GPU/DDP/真实训练准入，待root及独立原数值复核。

## 2026-10-02T09:41:24.876311+00:00 两环境实际结束与独立复核

worker19.747秒/pllm22.008秒，各8实际CPUrc0/父wait/PGgone/ownfixture删除；同原case参数0差、Adam1.862645149e-9/scheduler0、native8RNG和四mask两source尾批。每环境62JSON raw实际保留；A/B只读独立过程与数值强核对正在进行，不再model/CPUcase重跑，不作生产发布/正式训练声明。原GPU CE325/七项200无终态；数据及原安全门禁未改变。


## 2026-10-02T09:54:10.064871+00:00 独立复核完成及根端原件绑定

两实际环境及独立process/checkpoint/RNG、stdlib numeric raw复核完成；root170/38输入文件SHA、5 source和18旧保护路径匹配。根端结果 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_CPU_device_repair_20261002/actual_CPU_conformance_root_acceptance.json，process复核SHA41515b34ec303b061b38713c1963a751f3f220a96decef8068dc42ddde813ee4，numeric复核SHAc4008b7c9a9b9534ac14e5742c01863e4ebd7f4a17e46544ef44ef6928464796。全部自有numeric checkpoint清理完成，必要raw/日志与V1失败原件保留；未重跑验收、没有生产发布或GPU/数据准入。

原八项500步合成学习：CE350/500，其余七项225/500，池pending=0/running=8/results=65；09:50只读平台查询Running，09:51远程poll-gated状态及原trace无终态。八项均独立单GPU，不能称单模型八GPU训练。

真实shard/小batch/语料训练0；原strict training_ready is True、当前双审核及全部data/scope/source/selection/learning/profile/GPU/full-resume门禁保留。停止序列注释不恢复、重试、转交或绕过；独立CUDA算子暂停；原容器/worker/监听2213164不重启、不重复Volc提交，新GPU/平台submit/cancel/restart0。
