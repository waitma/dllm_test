# 固定 Trainer runtime 入口：实现与 CPU 验证

更新：2026-10-01 04:07 UTC。v1阶段时限失败保留。入口SHA370d34a5…已发布于不可变v2，总体3600秒；打包/入口/运输两环境各89项通过，新D01在GPU7运行，尚无GPU通过。[实际版本证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PILOT_RUNTIME_BUDGET_V2.md)。脚本名/API/schema保持v1合同；下文50cf SHA、29项CPU/1200秒为v1历史，不能认证新源码。

[入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pilot_runtime_entry_v1.py)仅接收固定 D01 profile-synthetic/steps=1 的工作池运输参数。steps=1 是运输合同，实际计划为三个 global256 完整 optimizer windows，即 micro32×GA8×3=768 次人工记录呈现；没有任意 data、resume、命令或架构参数。runtime 必须继承 manifest SHA 46399204c7018d00726e3ae08142bd4eb28475c8a9a2991b3a5c90725c8824f7 的 524 文件固定父快照，新增 producer 与本入口均须绑定完整源码哈希。

父进程核验 runtime 后，以固定脚本启动人工 producer，使用已有通用进程组/后代清理工具，期限 1200 秒；不执行该工具文件里的 attention/CUDA 专项测试。只有实际正常退出、后代清理完成、producer 的原始记录和强核验全部满足，才写 GPU_RELEASE_COMPLETE，随后写工作池 summary/COMPLETE。异常保留 FAILED 与日志，不写通过记录。producer 自己仅能写候选产物，没有父进程完成权限。

真实 pilot 的 execute_admitted API 先调用原数据 preflight，再核验完整 runtime 与同版本 GPU 完成证据；第一版仅覆盖 R20。该 API 未执行，不能据此声称 R21/R24/R26 已获新 Trainer GPU 验收。原数据门禁、模型快照、历史 39 份测量源码和证书保持原样。

[CPU 测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_pilot_runtime_entry.py)在 pllm 与实际 worker 环境分别 29 passed（7.68/6.99 秒）。包含重复/额外参数、错误目标、跨池路径、非有限/重复 JSON、原数据拒绝先于模型/runtime、副进程失败、以及真实 CPU 子进程退出 0 但没有 GPU 证明时仍拒绝。没有使用正向 mock 生成 GPU 通过标记。

入口 SHA=50cf109b482296a9036ea5f428618b2a5049671f24ee9d248faac294ba357713；测试 SHA=a93ace05cb1fbccea40b6d5940c2cbf59daf097e694d9464edf92db496b2567f。日志/JUnit 的 SHA 与核验范围记录于 [validation](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_runtime_entry_20261001/validation.json)。两个自有 refusal-test 临时目录共 58 个文件已删除，checkpoint=0；[清理记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_runtime_entry_20261001/cleanup.json)。worker 初次命令因输出目录尚未建立而在 pytest 前退出，纠正目录依赖后才得到上述通过记录，不计为执行过测试。

producer已在私有副本完成两环境各24项CPU检查并由根线程复审发布；此前来源漂移的原件保留。实际v1人工GPU已失败，完整BF16/AdamW、指标及正式三窗口验收仍未通过。v2必须重新绑定源码/manifest，不能把本文旧CPU或旧失败结果当成新版本证据。CPU拒绝检查不代替实际GPU或数据审核。
