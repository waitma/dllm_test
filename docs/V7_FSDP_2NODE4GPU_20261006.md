# V7 f011 两节点四卡 FSDP 接入

最新交付：含where性能修复的新独立配置见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FSDP_WHERE_2NODE4GPU_20261006.md。本文件保留原未优化FSDP版本，两个YAML均尚未提交；不要混用其manifest或恢复目录。

更新 2026-10-06T11:33:52.924409+00:00：本地真实V7整模型测速已完成，bias索引反向为主要热点；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/native_speed_evidence_root_review_20261006.json。本YAML仍绑定原冻结f011及source_manifest_v2，未接入where优化，不能将新测速当作该FSDP任务的速度承诺。此处仅同步文档，YAML/绑定源码未改；两节点GPU/NCCL验收仍待实际任务。

状态：2026-10-06T10:29:39.027411+00:00，YAML与独立入口已准备，CPU工程验收通过；尚未提交，两节点GPU/NCCL和目标显存验收待新任务执行。

用户要求在另一组两节点、每节点四卡资源上运行FSDP。原八卡DDP和本地f010均保持；正式架构仍f011（R24固定残差0.1＋R26交替局部/全局radius32），没有新证据支持换架构。新任务是独立训练对照，不接续或覆盖原DDP checkpoint。

新入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_fsdp.py；适配器 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/v7_fsdp_trainer.py；合成CPU检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_fsdp.py。

沿用冻结模型 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3；使用FSDP1 FULL_SHARD/use_orig_params，按ArchitectureBlock包装，根forward内执行原token验证，FSDP内部非重入activation checkpointing替代外层重复包装。模型与loss协议不新增生物学目标。源文件新增独立hash清单，不修改已有冻结manifest。

新YAML将先运行同配置合成张量训练、完整保存及原生恢复，成功后清理自有合成checkpoint，保留JSON证据；再执行原始只读数据门禁，使用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002。training_ready或原审核/范围证书不通过即停止；不补造证书，不降低门禁。新机器资源队列/GPU型号尚待用户提供，先参考已有两节点四卡规格。

速度主因已由单卡完整模型实测定位为chain_bias重复索引反向；本文件的原YAML未接入该修复。新入口增加host分阶段计时和有界整模型CPU/CUDA trace；host嵌套时间不能相加，trace窗口有额外开销。独立CUDA算子验证继续暂停。FSDP跨节点通信也有成本，不承诺比当前DDP快。

实现依据包括本机PyTorch2.8.0、Transformers4.48.1和Accelerate1.14.0实际源码。根forward限制见 [PyTorch 2.8 FSDP](https://docs.pytorch.org/docs/2.8/fsdp.html)；分片/状态保存机制见 [Accelerate官方说明](https://huggingface.co/docs/accelerate/usage_guides/fsdp)。

## 可提交文件与参数

YAML：[v7_f011_fsdp_2node4gpu_20261006.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_fsdp_2node4gpu_20261006.yml)。默认ResourceQueueName=c20250601，2×ml.pni2.14xlarge，每节点4卡，非闲时、不自动重试。另一资源队列或GPU型号需替换资源字段，共享挂载与固定Python环境必须可见；尚未得到不同资源信息。

每卡32、GA1、world8，global256；25042步、cosine LR1e-4、warmup1000，验证/保存每1000步、最终步补保存，最近3份完整恢复点。保持当前A4上限2048/动态batch padding/容量256与64及f011。每卡32是global256下GA为正整数时的最大micro；目标GPU峰值仍待profile，不能拿CPU或原DDP的显存冒充。若合成2048长度OOM即停止，不静默缩短/截断输入或跳过验收。

任务先在同一真实2×4拓扑进行合成4步保存、恢复到8步（同一8步日程），检查原生optimizer/scheduler/RNG恢复、四mask与19条尾批计数、全长显存与有限loss。8步仅是工程验收，不是架构有效性试验。通过才写独立ADMISSION.json，随后删除该合成目录的checkpoint，保留small JSON证据，再执行原数据审核进入真实训练。合成通过不等于数据审核通过。

正式新输出为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_2node4gpu_20261006/job_attempt1/train；合成目录为同根synthetic。已有DDP checkpoint不能直接传给此入口；独立从头训练，不迁移旧optimizer。显式恢复仅支持本新run的完整FSDP checkpoint，绑定数据/源码/参数/日程并核验原生文件与真实rank的RNG保存确认。当前没有自动寻找恢复点或故障重试。

CPU验收：loss/梯度与原模型完全一致，四档指标一致；尾批3条按3条统计；原生CPU Trainer在第2步保存/恢复后跑到4步与不中断4步逐参数相等；另有真实2-rank Gloo FSDP分片、activation checkpointing、平均梯度更新与四档评估/完整模型优化器状态往返通过。CPU测试不能替代2节点NCCL/BF16与生产显存验收。自有临时测试checkpoint已自动删除。

FSDP使用原生BF16混合精度与内部非重入activation checkpointing，和当前DDP的FP32参数/autocast及外层重计算不是逐算子精度、通信或重计算路径完全相同的运行；新任务是工程对照，不能单凭墙钟差异宣称单一因素因果或模型收益。

计时证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_2node4gpu_20261006/job_attempt1/train/timing_progress.json 分开累计取batch、训练步、前向/loss、验证、原生保存、保存核验的host时间；这些是嵌套区间，不能相加。whole_model_trace_from_0.json仅在开头wait2/warmup1/active4步记录整模型CPU/CUDA事件，可用于检查attention后端、CPU同步、通信和算子耗时；trace窗口不当稳态速度。未部署独立CUDA算子测试。

当前正式V7约10.9–11.2秒/步，旧V5约2.73秒/步，均global256但运行协议不同。最新实测已确认chain_bias重复索引反向为主要模型计算热点；其余端到端开销仍需新任务测量。原容器远端执行接口此前403，不重试或绕过；后续单卡有界测速已完成并恢复原消融，详见速度根因报告。新资源的trace将补充实际证据；FSDP跨节点有通信代价，不承诺提速。

验收原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_2node4gpu_20261006/root_review_v2.json；绑定清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_2node4gpu_20261006/source_manifest_v2.json。v1是未启动的准备稿，YAML只引用v2。

提交命令（由用户执行）：

```bash
volc ml_task submit --conf /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_fsdp_2node4gpu_20261006.yml
```

提交后将真实task ID与状态加入 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 的Active表及dated日志；当前未提交，不能预填任务ID。
