# T20 R20 工作池桥：runtime v2 私有 rebase

2026-10-01。本版仅在 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_v2_20261001/isolated_worktree` 实现和测试。未 publish/apply live support、生成实际最终 stage、创建 passing pointer/GPU/data certificate、入队、执行 GPU 或读取真实生物数据。父代理维护 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md`，本文件是独立实现记录。

原 T20 v1 源码与两环境 48 项证据保留在 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_20261001`，本轮逐 SHA 复核未改。原 D01 runtime v1 失败冻结保留，不能作为通过证明。

## 精确 rebase 来源

此 rebase 使用根端 2026-10-01 03:53:00 UTC 已 CAS 发布的 support。来源记录 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_budget_v2_cpu_20261001/publication.json`，SHA `4406134215f6efb9de5836a8a314513ce35f5683070fc72207e2d84e00075991`。五个实际 source bytes 在复制前核对如下：

| source | 已发布 SHA256 |
| --- | --- |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pool_release_dispatcher.py` | `abfff84782ca1ecf72c51d543bf27712d496dd344ed17704ffd96f2b7bd5c075` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/release_startup/sitecustomize.py` | `25a8223bcc6da8318a95d7173dcd1a818546333d7eb310a11955df0c1574a00d` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pilot_runtime_entry_v1.py` | `370d34a5a68dff3a8ba411bf658f5cda8e32157e729d35b0133ffaf0602f71e9` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pilot_runtime_package.py` | `47bbe9c68279a7a5b441434f39f8d80f3a4d59a2d72cb82c62eb95478bd1dacb` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pilot_gpu_release_producer_v1.py` | `ae1a60cbf0f155482f93dd2b1a612883b3a9de06b8be4b07458fc39360da175d` |

guard、entry、package、producer 字节在本私有树中原样保留。只在新 private dispatcher 叠加已验收的 T20 分支；不会修改已经建立的 runtime v2 或其 additions。

与冻结 T20 v1 相比，最小变更为 fixed runtime `r20_r27_pilot_v2_20261001`、guard SHA pin、dispatcher before SHA pin、外部 launcher manifest 的 v2 review edition，以及 positive tiny CPU fixture 中对应的 fixed runtime path。transport 测试文件字节未改，API/准入顺序和其它条款未放宽。新 runtime 的整体 3600、oracle 1200、formal 1800、evaluation 600 秒预算由根端 runtime support 负责，桥不改预算或测量定义。

## 固定接口与真实限制

仍使用 `t20_r20_certified_pilot_v1` transport variant、`trainer_runtime_v1` route、`pilot_pool_launcher_v1.py`/`pilot_runtime_entry_v1.py` 文件名和原 v1 schemas；这是 API 命名，runtime source edition 明确为 v2。

固定 runtime 为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v2_20261001`，外部 launcher manifest 改为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pool_launchers/t20_r20_pilot_v2_20261001/launcher_manifest.json`。此新 review edition 必须绑定确切外部 launcher/dispatcher source SHA 和 root-reviewed pointer SHA；不能复用 v1 review 记录来声明桥已被原 GPU source map 覆盖。

唯一 pointer 名仍是 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/pilot_admission_r20_v1.json`，当前未创建。GPU evidence 必须来自该 pointer 固定绑定的同 pool D01 directory；尚无通过 GPU 目录，不能依靠 oracle 耗时、旧 failed v1、CPU mock 或 profile 来凑准入。

固定数据 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_expanded_candidate_20260930` 的 manifest SHA 仍为 `601db781ebf81ee857dde98ddcfb2d1da8e121a4c57ff2d80c864d420a4ebfce`、`training_ready=false`。固定现有 selection `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/expanded_candidate_validation_selection.json` SHA 仍为 `659ef88421dba606fb327e8221091b0feb5f939cb78c19a643da6aeac35b94f2`（1288 个既有索引）。不存在可接受的真实 data certificate；两个文件存在仅是缺文件快速拒绝检测，不能替代原双证书/分片/scope 检查。

执行仍先 stdlib source/pointer/runtime 验证及继承 guard，再 runpy 非 `__main__` 加载固定 entry 并调用其 stdlib `install_runtime_search`，随后原 unchanged `assert_ready` 必须先于 GPU evidence acceptance、model construction、training/output。要求 `training_ready is True`，再调用 unchanged `execution.preflight`、root stronger completion 和 root `execute_admitted`。R20-only、1000/5000 累计 stop、5000 显式完整同 pool resume、真实 R20 profile、三 formal optimizer windows、五 GPU artifacts 和数据/selection/source binding 均保留。

外部 admission receipt、output 的 `pool_launcher_provenance.json` 与 `POOL_LAUNCHER_COMPLETE.json` 独立绑定 bridge source/manifest/pointer、runtime 和精确输入 bytes；训练前后均复核。原 R20 executor 负责自己的 `COMPLETE.json`。外部证据有漂移时不能冒充 T20 完成。普通 Python profile/probe child 继续继承同 runtime v2 root/namespace/guard，不引入任意 shell/module/target 或不同架构。

## Stage 与发布前提

`stage_pool_pilot_launcher_v1.proposed_sources(root)`/`stage(output_dir)` 的 before 仍是 current deployed D01 BASE3：runner `89e154f682e7ced2680f38bd0895eea2d18fd3447eb2e8aa0cd4184afc34fd30`、registry `2b0b4de27f6c1dac950af510c50eb49158ae577e31e0d323d2e1be9ff370bd84`、pool `f3522eca58f7fedf145e8d61986d5124e0cdd9e2718c6062c53c94e563ba832f`。dispatcher upgrade before 改为已发布 v2 的 `abfff84782ca1ecf72c51d543bf27712d496dd344ed17704ffd96f2b7bd5c075`，guard 精确沿用 `25a8223bcc6da8318a95d7173dcd1a818546333d7eb310a11955df0c1574a00d`。

T20 registry 项仍为 `launcher_only`、不可构造模型、init 排除；D01、28 个历史 model options 及旧 R20 train frozen route 不改。这里只有 pytest 独立临时目录中的 review patch，未生成实际最终 stage 或 apply。根端等 D01 v2 terminal 后独立 review/apply；data 仍无证时即使 GPU 通过也不能训练。

发布必要项还包括 T20 的明确 source provenance/runtime map 或独立只读 typed 分支，避免普通监督回退 live source root 造成 immutable runtime 指纹误报。本轮没有扩大 typed 展示、模型或其它测试范围，也没有刷新 LOCAL daemon 或 GPU worker。

## 新 CPU 证据

仅运行前版 T20 的两个专属文件、共 48 项必要 rebase 检查，两环境均通过。验证固定 envelope、缺 pointer/cert 无副作用、原真实 data gate 与 readiness 拒绝、current BASE3 patch、constructor/init 排除、继承 runtime v2 guard 的真实 ordinary children、R20-only gate、完整 resume、外部来源/input 漂移，以及 unchanged root API/executor 上真实 tiny CPU 1000→5000 continuation。positive admission interfaces 明确 mock，所有 fixture proof 为 not acceptance；不是 GPU/data acceptance 或真实语料训练。

归档目录 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_v2_20261001` 保留 `source_before.json`、`t20_v1_source_before`、`published_source_before`、`t20_v2_rebase.patch`、`private_dispatcher_review.diff`、`pllm_launcher_pytest.log`/`pllm_launcher_junit.xml`、`worker_launcher_pytest.log`/`worker_launcher_junit.xml`、`cleanup.json` 和 `validation.json`。cleanup 只清除此版本明确指定的两套 pytest basetemp，source/test/doc/diff/log/JUnit SHA 由 validation 绑定。旧 v1 源码和 48 项证据逐 SHA 复核保留。
