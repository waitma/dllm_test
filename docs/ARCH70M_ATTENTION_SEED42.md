# 五架构新增一个训练种子：seed42

2026-11-03 用户要求再做一组种子对照。本轮只新增seed42，完整复跑C0(A5)、C1(QK归一化)、C2(输出门控)、C3(组合)、A0(原始架构)；与正在完成的seed43配对，不追加seed44或其他种子。

每组从随机权重开始，1epoch/1276步，micro32/global512，原学习率1e-4、AdamW、5% warmup、目标、噪声、解码和评估保持。每组仍是653,288条固定10%数据、214,482,921个非pad token。不能复用早期micro56/96或三epoch日程下的seed42结果，也不继续训练seed43检查点。

模型直接调用冻结的`architecture_attention.models.build(settings, variant, 42)`，训练仍调用原冻结trainer；新代码只负责seed传递、profile/恢复绑定、排队和双种子报告。seed43源码SHA保持`28609d7f776013915561eb038396d54181dcee7b1c32d071be0a9da5dc12faad`。原V8形式、字段、链顺序/角色、metadata、标签、词表/tokenizer、固定链布局及reader/collator不变，数据索引字节相同，SHA保持`4beaac6b54a1f46492533aeb77c2de2c6ac2a0bd8266fd57a6bb312679435c51`。

## 队列和资格检查

入口：

```bash
bash /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/launch_arch70m_seed42.sh --gpu 0
```

独立输出 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/output/arch70m_attention_seed42_20261103`。控制器先准备不可变快照，随后等待与原seed43控制器相同的GPU文件锁。等待时不分配GPU、不打断或重启现有训练；原seed43全部五组完成且完整checkpoint核验通过后，自动继续。

seed42五组重新执行maxlen2048/micro32、每组三次真实optimizer更新，并检查原85%显存门槛。全部通过后用新进程按C0→C1→C2→C3→A0顺序从头训练；profile权重不用于正式训练。只有本机单GPU，不提交远程作业。源代码、旧输出和生产1B训练只读。

训练保存逻辑保持原完整六文件检查点、优化器/调度器/RNG/数据游标和绑定核验；恢复时显式绑定seed42，不接受seed43的profile或检查点。原先的seed43模块不做修改，新适配单独冻结。

## 比较和记录

只用原screen分区，比较固定最终step1276，不按最佳seed或中途最好checkpoint选择结果。保留两seed各自完整loss/生成/表征曲线及原始CSV。全部10组完成后生成同seed相对C0、A0的差值以及每模型的双seed均值/样本标准差，检查CDR、AUROC/AUPRC、整链AAR、CE和长度误差的增益/退步方向是否复现。

双种子只是重复性复核，不把两seed当成原三seed稳定显著性确认；原稳定性门槛不变，旧confirm不当作新的未使用留出。此次没有新增confirm评估。

报告 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/attention_seed42_20261103/REPORT.md`；PNG/PDF/SVG、CSV和配对结果逐组更新并commit/push `explore/arch70m-v8-20261028`。推送失败只记录publication pending，保留训练真实状态；后台排队与运行，无需持续人工轮询。

CPU检查覆盖同一模型/训练实现、42种子传递、新旧profile拒绝、完整恢复绑定/权重校验、全矩阵配对、真实GPU文件锁排队、旧源码保护和快照导入。实际通过状态、控制器PID和队列阶段见同目录validation.json、queue_verified.json与PROJECT_PROCESS.md。

## 实际排队核验

2026-11-03T05:23:33Z UTC：seed42控制器PID74329，实际命令及cwd已核验；状态waiting_for_gpu，内核等待点locks_lock_inode_wait，未创建GPU worker。原seed43 A0核验时step1174，保持正常运行。7项CPU回归、shell语法、dry-run与快照导入通过；新source SHA `258dd11e8895c3b6d4fbb9c00e526de67e6bc9ccbef43948970790f5a00d8fed`。GPU资格和正式训练尚未开始，将在旧队列结束后自动运行。
