# Trainer 运行包 v3：对象生命周期与原生工作区

更新：2026-10-01 05:52 UTC。v3 R20完整GPU工程验收及根端强复审通过，3/24/768；T20私有准备完成，数据仍未准入，正式训练尚未启动。

## 2026-10-01 05:52 UTC v3完整GPU工程验收通过与剩余数据准入

05:44:51 UTC，v3 D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959 在原GPU3终态succeeded/rc0。根端05:48以worker同一Python的全新-I进程执行原不可变verify_completion/verify_candidate，通过全部raw/summary/COMPLETES/installed source/独立CPU计数重算；禁止本地CUDA初始化，未重跑GPU。runtime manifest仍cc13036fc492109afbf4c9cf9fd11a4c87684fc9702b83ce86c961b2a08294fe，524父文件不变。

实际正式3窗/24 microsteps/768人工记录，formal1054.3747秒<1800，训练loss3.450910→3.448508→3.380331；首窗LR0符合固定warmup，后两窗LR1e-6/2e-6且参数确有更新。独立full oracle706.9962秒<1200、GA/CE/RNG比较通过；四mask评估5.1172秒<600、每mask24条/共96次，eval_batch16含实际尾批及完整2048输入，指标由raw packets独立重算。正式峰值allocated45,350,397,440字节（42.24 GiB），reserved56,555,995,136（52.67 GiB），原85%标准通过；tiny/full/formal清理弱引用释放、最终allocated0。parent1794.0503秒<3600，child rc0、正常退出观察/后代清理true。[强复审原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/terminal_review.json)、[GPU通过原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/GPU_RELEASE_COMPLETE.json)。

两个自有空hf目录已删除，checkpoint=0；raw20,049,785字节、summary/日志/完成哈希是后续strong verifier及训练入口依赖，按必要证据保留，v1/v2失败原件不覆盖。[清理清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/gpu_validation_cleanup.json)。首次root提取曾把字段读到不同COMPLETE schema，读取错误已保存在review中，修正后通过，生产代码/标准未改。原8卡任务Running，0/0/39=36历史成功+1新D01成功+2D01失败，8worker维护；本地监听PID2213164继续watching，无新平台申请。

T20 v3私有两环境各48项通过且根端AST复审，source-map私有各23项通过，未publish/实际stage/apply/pointer/入队。[T20私有检查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_v3_20261001/validation.json)、[桥接复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/t20_v3_root_review.json)、[源映射复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/source_map_root_review.json)。GPU验收仅覆盖R20工程入口，不认证R21/R24/R26同版本训练，不证明架构收益。2026-10-01 05:52 UTC数据metadata仍training_ready=false、双证书缺失；审核后manifest改变须另做明确版本绑定，不能仅补文件绕过False。真实四候选1000→累计5000与最终版本训练未启动，goal仍active；独立CUDA算子保持暂停。

## 2026-10-01 05:31 UTC full梯度与清理实测通过，正式窗口运行中

v3新D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959 在原GPU3已完成独立full gradient_oracle：706.9962秒<1200，1窗/8 microsteps/256人工记录，78参数比较通过，梯度倍率1.00000077、相对L2误差0.00015194，CE/RNG核对通过。11个对象弱引用全部失效；gc/empty-cache后allocated17,039,360字节，清cuBLAS workspace后allocated/reserved均0；新运行tiny与full阶段均实测支持工作区驻留候选，历史v2仍缺残留数值，不追认其根因。[完整oracle局部实测](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/full_oracle_observation.json)。

已进入train_optimizer_step/1800秒预算，正式三窗口和四mask评估仍待完成，整体gpu_accepted/raw_completed=false。524父字节、原架构、数值标准和工作量不变；独立CUDA算子暂停。监听PID2213164/原八卡任务保持；T20 v3私有rebase中、源映射未部署，数据双证书仍缺/候选training_ready=false，真实比较与最终训练未启动。验证checkpoint=0。

## 2026-10-01 05:20 UTC 当前实际v3状态

v3于05:04:13 UTC实际发布8个支持/producer文件，05:09:59构建526文件不可变包，manifest SHA cc13036fc492109afbf4c9cf9fd11a4c87684fc9702b83ce86c961b2a08294fe；524父文件保持。producer必要CPU两环境各30项，support各108项通过，实际隔离导入各8 profiles/54项目模块通过。05:12:16本地监听实际重载PID2213164并核验watching，GPU任务/worker未重启。05:14:40只在原八卡池入队D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959，GPU3实际领取。固定tiny preflight耗时1.8074秒、8 microsteps/8人工记录；11个对象弱引用均释放，gc/empty-cache后allocated17,039,360字节，清原生cuBLAS workspace后allocated/reserved均0。当前完整gradient_oracle进行中，formal三窗口及四mask评估尚未通过；仍gpu_accepted=false，不能把局部tiny通过记作完整验收。[发布原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/publication.json)、[固定运行包](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v3_20261001/pilot_runtime_manifest.json)、[实际启动记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/d01_v3_launch.json)、[tiny阶段实测](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/lifetime_preflight_observation.json)。

T20 source-map私有两环境各23项通过，原28项映射保持；[软件源绑定记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_source_map_cpu_20261001/validation.json)。T20新私有v3正在显式rebase固定root/manifest，未发布/apply/pointer/入队；原数据准入全部保留。候选training_ready仍false、双证书未齐；真实四候选1000→累计5000与最终训练未启动，goal active。独立CUDA算子仍暂停。自有CPU临时目录已清理，人工GPU关闭checkpoint保存，小型指标/失败原件/源码哈希保留。

以下带时间记录属于当时状态。



v2于04:12:41终态failed：实际独立梯度/CE核对通过，但正式三窗口前的allocated==0检查失败。原运行包及失败原件保留；[终态原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_budget_v2_cpu_20261001/terminal_review.json)。v2未记录具体残留字节，因此不能事后确定泄漏对象。

两个环境的真实tiny RefinedModel完整GA8/whole_layer checkpoint CPU实验，在run_stage finally内的首次gc后已释放model、Trainer、Accelerator、callback、AdamW wrapper/state与parameter；worker的BF16包装和workers2组合也释放。[CPU精确时序](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_gpu_lifetime_v3_cpu_20261001/diagnosis_worker_amp_workers_exact/case.json)。它只排查Python持有者，不证明CUDA清理正常。

PyTorch 2.8的cuBLAS工作区会保留到专用清理调用，其固定版本源码使用原生map和CUDA caching allocator分配；这是可能解释CPU与CUDA不同的候选，尚未实测确认本次v2根因。[2.8官方说明](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cublas-workspaces)、[v2.8.0源码](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/cuda/CublasHandlePool.cpp)。候选修订先记录所有具名owner弱引用和after-GC/empty-cache字节，只有owner全释放才在自有child同步后清cuBLAS workspace并再记录字节；仍要求实际allocated严格等于0。不得设置workspace禁用环境、改变训练kernel/精度、删除Tensor来伪装结果或重置全局Accelerate State。

v3固定canonical目录为r20_r27_pilot_v3_20261001，仍保留524父文件与原主干。拟新增固定tiny生命周期preflight：d16/L2/H2/MLP32、length64、full8/fragment4，micro1×GA8/global8，FP32+BF16/AdamW、whole_layer checkpoint；一窗native+独立manual8核对，独立120秒上限。其工作量单独记录，不能计入原full oracle或三个正式窗口。原2048输入、micro32×GA8/global256、full oracle一窗加formal三窗以及1200/1800/600、整体3600秒和全部数值/85%显存标准保持。candidate verifier还须强验tiny实际CUDA/计数/梯度/清理证据，失败不得产出GPU通过marker。

根端private support只调整固定v3路径、guard SHA、readonly白名单，并保留v1/v2识别和未知/非canonical/symlink拒绝。两环境package26+entry29+transport34+typed19，共108项分别17.51/15.90秒通过，真实普通Python child隔离检查保留；没有执行GPU/model/data。[必要CPU验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/validation.json)、[review patch](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/review.patch)。

首次测试因显式basetemp父目录未建立，两环境各11 passed/97 setup errors；保存原始日志后只创建自有目录，未改实现/断言，复测108全通过。[失败设置记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/setup_failure.json)。两个自有测试目录2626文件/9,972,831字节已删除，没有真实GPU或训练checkpoint；小型指标、JUnit、哈希和源代码复审保留。[清理清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/cleanup.json)。

producer修订及其必要CPU断言仍在新私有树完成。T20 v1/v2原证据保留、未部署；另一个私有软件任务补明确release_map源码绑定，后续v3 rebase须等待根端复审。现有十个prepared manifest的只读metadata目录检查未找到任何downstream_scope_certificate.json，候选training_ready仍false；不切换历史版本冒充新范围已准入。双证书和真实候选比较未完成，[整合状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PRETRAIN_INTEGRATION_20261001.md)。
