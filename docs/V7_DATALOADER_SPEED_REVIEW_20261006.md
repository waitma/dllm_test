# V7 数据加载 CPU 速度诊断

2026-10-06，当前状态：冻结实现、元数据核实与一次有界 CPU 计时已完成，原进程退出码为0。正式八卡与本地 f000 继续运行，本诊断不修改训练代码或参数。

正式绑定为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_initial_training_20261005/run/run_provenance.json。当前数据为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002。三份 SHA 已与正式绑定核对：manifest `28e3cbf8fb860c13b514a11d600579257a233fefe9f70ea070b042d04f8a424c`、audit `5c7b8a80387b5fbb9cbd58fc498ddea11a1751b283ecaa064c3f62ac11cdfb94`、scope `b868e2adce6a90b7ccf7f5efbbc517cfbd649e2b41b712c6680a90d81d9453fd`。ready=true，两证书 passed/pending=[]；限定 observed-pair scope 保持，不扩大为所有下游或相似度排除结论。

冻结来源 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3；原611清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_manifest_v3.json 的 SHA 为 `992bc3c5718b71c379c7a2ee4f39d8e5866f974266500d2c2ec69d5ab4db700b`。只核对本诊断实际导入源码，不重复完整611验收或数据审核。

## 只读结论

- /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/immune_llada/data/dataset.py 的构造函数逐分片 readline/tell 扫描全 train，建立 uint64 offsets；没有持久化索引缓存。全训练为6,410,749条、74个shard，每rank初始化都会扫描；offset数组本身约48.91 MiB/rank。这是初始化成本，不能直接解释稳态每步约10.6–11秒。
- 原 getitem 使用缓存的文件句柄、seek、单行 JSON decode 和 prepared object 恢复；没有运行时来源解析、筛选、补样或新序列注释。
- /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/immune_llada/v3/layout.py 的 A4 每条恢复既有观察范围，构造11份字段列表并逐字段 torch.tensor；按当前batch最大渲染长度padding，上限2048。target full/fragment容量256/64内部空槽仍被attention看见，必须与batch外部padding分开统计。
- /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/scripts/research/run_v7_formal.py 使用 num_workers=0，取样/拼批在训练进程。尚未得到训练容器内数据等待、forward/backward、通信与保存的分段耗时，不能据此认定 loader 是主瓶颈。

## 已授权有界计时方法

脚本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/data_loader_review/probe_data_loader_cpu.py，激活 pllm 后用原 protenix_abtcr Python、-I -B、CUDA_VISIBLE_DEVICES=''、CPU线程1。

不调用会全扫描的真实 constructor。仅从七来源各选择一个固定shard前缀，最少8条/来源、余量按manifest记录数分配，得到1024个不同记录；用1024行读取建立临时内存索引，再通过原 getitem 读取1024次，总计2048行读取。该局部索引初始化与完整初始化明确区分，没有创建训练数据副本、索引缓存或审核文件。

单次读取逐记录及32组计时，恢复对象保留在CPU内存，用同一打乱顺序分别实际拼 batch32/64；64组getitem为共享32读数配对求和，并非独立第二次加载。各拼批前两组记warmup，稳态报告p50/p95与样本数；输出实际tensor长度、外部padding、被attention看见的target内部空槽。只写时序/计数/哈希，不打印序列，不实例化模型、forward或GPU操作。

局限：来源最少数和shard前缀不是全语料随机抽样；索引扫描先预热页缓存；本地主机单CPU进程不代表八rank容器的FS/CPU争用。getter+collate之和是CPU分项代理，不是端到端训练或batch64加速证据。停止的序列注释分支和独立CUDA专项不恢复。主MD由根端同步。

## 实际结果与证据

实际执行时间为2026-10-06T00:40:12.575998+00:00，主机为 `di-20260121220916-9rvmj`，不是正式八卡容器。解释器为 /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python，torch=2.8.0。运行前激活pllm；命令使用 `CUDA_VISIBLE_DEVICES='' USE_TF=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`、原解释器与 `-I -B`。退出码0，总墙钟6.3004秒；结果明确记录 CUDA未初始化、无模型实例化/forward/训练。

结果原件为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/data_loader_review/result.json，SHA256=`98e28f54dba6dd665459f3bfc73b0c425762ff170370597d707e7d4aabb4bf75`。诊断脚本SHA256=`60c87fc57360d07c155d5f35d893af5a543d70202bdf909069b9d8b3a61c14b7`。实际导入的23个冻结源码文件逐项与611清单匹配，并在运行后保持一致；三份数据元数据再次核实未变。未重新扫描全语料或伪造完整constructor耗时。

实际工具session43498的退出码0与stdout结果引用已转录保存到 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/data_loader_review/execution_receipt.json。该回执明确标为工具观察转录，不是训练回执；没有为保存回执重新执行探针。

1024条不同记录的来源计数为：asd_antibody=34、oas=381、ots=265、tcr_native=10、tcr_papers=25、tcr_repertoire=301、trait=8。临时前缀索引读取1024行，原getitem读取1024行，总2048次记录行读取。索引构造0.04142秒，collator构造0.000092秒，冻结模块首次导入3.6393秒；完整dataset constructor未执行，完整初始化耗时仍未知。单条getitem稳态960条，p50=0.04151毫秒、p95=0.06269毫秒。

| CPU稳态项目 | batch32 | batch64 |
| --- | ---: | ---: |
| getter p50 / p95，毫秒 | 1.48 / 2.14 | 2.96 / 4.14（32组配对和） |
| 实际collate p50 / p95，毫秒 | 36.04 / 49.06 | 78.84 / 102.16 |
| getter+collate代理 p50 / p95，毫秒 | 37.46 / 50.58 | 81.79 / 105.28 |
| 稳态batch数，排除前2组 | 30 | 14 |
| 最大渲染长度 p50 / p95 | 539 / 1130.45 | 743 / 1168.30 |
| 全部batch外部padding比例 | 43.58% | 51.27% |
| attended位置中的内部target空槽比例 | 53.08% | 53.08% |

两档共用相同1024条记录，每条渲染长度70–1206。全batch32 tensor位置总数679,040，其中attended=383,083、外部padding=295,957；batch64 tensor位置总数786,112，attended仍为383,083，外部padding=403,029。两档attended内部target空槽均为203,347个位置。这些槽是当前生成接口的一部分，不能与外部padding混为一项，也不据此改变训练目标。

局部证据支持：本样本CPU时间主要花在拼批而不是单行读取。batch32代理约0.0375秒，与正式包含验证/保存的10.6–11秒/步量级相比很小；这不是容器内端到端分项测量，不能据此排除共享FS/CPU争用或认定GPU主因。把batch增加到64时，同样记录的tensor位置增加约15.77%、外部padding增多；不能据此断言batch64训练更快或更慢，正式global batch与模型计算均未测量。

当前不修改运行中source、dataloader参数、batch、数据或审核门禁。若继续定位正式慢因，下一项必要证据是在原已授权范围内、资源可用的训练边界记录容器内数据等待、forward/backward、同步、验证与保存耗时；不再重复本CPU探针，也不恢复独立CUDA算子验证。
