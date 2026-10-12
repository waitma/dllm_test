# 通用整数GPU验证：100步分段与连续500步协议

更新：2026-10-01T11:29:06.399847+00:00。维护agent `/root`，原Goal `01a0f245-4559-7b62-9dc2-0917a8ed2267`继续active。本轮仅通用工程与人工整数任务。新GPU protocol/engine/entry源码已实现；**两个环境各44个唯一CPU案例在三个阶段完成覆盖，15文件固定源码包及8份禁用配置已核对；尚未接入8卡执行端，没有新增GPU任务或真实数据训练**。此前八原型CPU功能验收保持，不重跑。

源码：[protocol](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/generic_joint_gpu_protocol_v1.py)、[engine](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/generic_joint_gpu_engine_v1.py)、[entry](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/generic_joint_gpu_entry_v1.py)。首轮结果和静态复审：[原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/first_CPU_and_static_review.json)。

CPU控制使用d12/L4/H6、FP32；G00和G06均实际验证连续102窗与100窗保存后恢复2窗的student/EMA/投影、optimizer/scheduler、数据摘要与RNG一致，global128。它不是d192 CUDA/BF16效果证据或500步学习曲线。

当前[CPU验收原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/validation_profile_count_fix.json)，SHA `946a5929ff2e1a00c248f135836444f3d954242edd75f8166d8f7bab7d043342`。每环境首轮26项通过，三项边界修正后定向20项通过（8项重查、12项新增），随后计数修正新增6项通过；合计44个唯一案例，不声称最终源码完整重跑44项。原38项[阶段验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/validation.json)及全部日志/JUnit保持。

复审要求补齐旧worker依赖的COMPLETE执行完成文件（独立验收仍pending），逐参数Adam moments/step/shape/dtype与完整scheduler字段拒绝，以及前段result/review/admission的canonical/nonsymlink检查。原版本实测会接受Adam state为空和scheduler内部计数重置，[定位原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/missing_state_pre_fix_diagnosis.json)保留。首次定位包装器误按pytest短目录名选择方法，在模型/训练之前停止，改用payload身份后定位成功；[包装器记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/diagnosis_attempt1.json)。原4项源码字节和首轮日志/JUnit保留。

后续只读监督发现profile训练成功后若评价OOM，旧异常分支会漏记已完成更新。[发现记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/supervisor_followup_static_review.json)。已改为optimizer.step返回后立即计数，后续scheduler/EMA/评价异常不清零，OOM记录失败阶段。新增6项包括四个模拟profile异常/正常案例及两个真实d12单窗CPU计数控制；模拟profile产物不属于GPU证据，已删除。

固定包[manifest](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_runtimes/generic_joint_gpu_v1_20261001/generic_gpu_manifest.json)，SHA `f9073b972aa8b59e3c2ccd2fbd6d36ea18ccbb2c2e02266093b228922ff7bf92`，仅认证15项来源文件；[构建记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/source_package_construction.json)。两个环境均核对来源字节和8份配置，且未有startup绑定时入口在Torch导入前拒绝：[pllm核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/pllm_source_package_check.json)、[worker核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/worker_source_package_check.json)。初次核对因Python -I忽略PYTHONDONTWRITEBYTECODE，生成2个额外pyc而停止；加显式-B后通过，自有pyc已删，[失败与修正](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/source_package_check_attempt1.json)。未构造模型/优化器、未初始化CUDA，不能当作GPU完成。

## 原因与工作量

现有worker每次读取磁盘注册表，因此未来可在不重启worker的前提下加入独立G任务；但其人工模式每job最多100步，train模式要求真实数据目录。不能借用R/D身份、填写假目录或把500次更新藏在1/100的预算中。

明确采用每方法一个profile身份及五个不同segment身份：`__s001_100`、`__s101_200`、`__s201_300`、`__s301_400`、`__s401_500`。每个segment外层steps=100，实际也只执行100次optimizer.step，连续累计到500；profile最多4次独立优化器更新，单独计数，不加入训练曲线。只读监督已复审该原则和必要边界，不声称GPU通过。

共同计划global128、seed42、LR3e-4、整程500步/25步warmup的连续余弦日程；本轮明确共同gradient clip norm=1.0。数据种子按累计步数推进；恢复不能重置日程、优化器或重复前100步样本。显存profile先测16/32/64/128微批，实际训练档位须绑定已核验profile。固定协议源码：[generic_joint_gpu_protocol_v1.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/generic_joint_gpu_protocol_v1.py)。

## 续接与证据边界

后段仅接受明确前段job、同方法/源码包/配置/seed/日程绑定及前段结果核验SHA；不寻找latest，不跳段、不跨方法或重复续接。状态包括student、EMA teacher、投影、optimizer、scheduler及CPU/CUDA RNG；加载构造完成后恢复RNG。第一至第四段完成不等于500步完成。

协议将临时内部续接状态明确计入验证产物。它不同于常规按间隔保存用户checkpoint；不声称分段期间零状态文件。后段核验前保留必要恢复副本，最终五段核验完成后删除自有验证状态，保留原始指标、版本SHA、失败和清理证据；真实训练恢复状态不受影响。续接保存/恢复已写入新engine；现有8份YAML常规checkpoint_save=false保留，临时续接约定已同步，GPU终态强复核/最终清理工具仍待实现。[当前配置SHA](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/updated_plan_manifest.json)；旧版本和快照保留。

本轮五个自有CPU目录中的54个常规文件/8,637,834字节已删除，其中50个状态文件：[清理原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/cleanup.json)。计数补查的两个自有目录另删16个模拟结果文件：[补查清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/profile_count_fix_cleanup.json)。首次补查清理因pytest自有current别名停止，随后记录别名并不跟随地删除：[包装器记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/profile_cleanup_wrapper_attempt1.json)。保留验证checkpoint=0，日志/JUnit/源码与失败证据、实际训练恢复文件保持。

## 用户指定正式数据与当前门禁

用户要求后续尽量在同一个8卡容器并行验证，并用正式训练数据的小batch做架构判断。指定目录为`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1`；本轮只核实metadata，manifest SHA `e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94`。该manifest没有training_ready字段，audit_certificate.json和downstream_scope_certificate.json均缺，不能解释为training_ready=true。[只读原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/user_designated_training_dataset_metadata.json)。没有读取shard/序列或启动真实batch训练。

该目录已登记为正式小batch验证的首选版本；仍需有效现有审核及明确版本绑定，且实际模型输入/目标/评价适配尚未完成。G计划继续只接受16类人工整数，不直接把真实语料塞入其数学标签任务。人工500步曲线尚未运行；它不代替正式模型效果比较，后续须明确工程GPU控制与真实任务比较的目的，不能靠无关synthetic实验填充Goal。

## 当前接续

1. 新独立GPU engine复用已验整数模型/目标/评价；不改旧CPU五项来源SHA，不执行真实语料。
2. 必要CPU验收重点为连续执行与断点恢复的参数、optimizer、EMA、日程、数据/RNG等价，以及固定任务/错误状态拒绝；不重复旧35项模型语义验收。
3. 完整源码包已构建并经两环境元数据/配置核对；下一步完成G注册表、执行选择/启动隔离的最小增量及独立结果复核工具。当前live注册表/dispatcher/guard/worker/监听均未改。GPU显存与500步曲线尚无成绩；真实小batch依赖有效审核和真实模型适配。

training_ready=false、双证书缺失时真实语料不训练；停止的序列注释分支不恢复/重试/转交/绕过，独立CUDA算子测试暂停。当前[接手状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)及[总账](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)同步此节点。

本轮只读监督已确认计数修正及固定包逐项来源，结果见[源码包复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/source_package_review.json)；监督已返回，不声称仍在后台持续运行。生产启动须保留原guard的环境加载方式，不能因元数据检查用-I -B就直接替换生产参数；-I也忽略PYTHONPATH。
