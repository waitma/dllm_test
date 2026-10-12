# T20 R20 工作池桥：私有 CPU 验收候选

2026-10-01。此版本仅在 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_20261001/isolated_worktree` 实现和测试；没有发布、应用 live patch、生成实际 admission pointer、入队、GPU 执行、真实数据读取或证书签发。父代理维护 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md`，本文件是此实现的专属记录。

工作池桥已经调用既有固定 runtime 的 `execute_admitted(PilotRequest(...), gpu_release_dir)`，不是训练规划占位。`t20_r20_certified_pilot_v1` 是外部 launcher 变体，实际模型固定为 `r20_refined_control`。已有 `r20_refined_control` 工作池 train 路由仍指向其原始 R20 frozen snapshot；D01 路由独立保留。immutable runtime 的 entry/producer 两个 additions 没有改动，桥不属于它们的 GPU source map。

## 固定输入与准入顺序

预期发布根目录是 `/vepfs-mlp2/c20250601/251105016/project/dllm_test`。数据固定为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_expanded_candidate_20260930`，manifest SHA 为 `601db781ebf81ee857dde98ddcfb2d1da8e121a4c57ff2d80c864d420a4ebfce`。现有 selection 固定为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/expanded_candidate_validation_selection.json`，文件 SHA 为 `659ef88421dba606fb327e8221091b0feb5f939cb78c19a643da6aeac35b94f2`，已有 1288 个索引；桥从不生成 selection。

当前 expanded manifest 的 `training_ready=false` 是实际限制。此 v1 使用固定旧 SHA，即使补两个证书文件也不能成为训练准入。数据审核改变 manifest 后，根端必须明确复核并版本化 dataset/selection 的固定绑定；桥不会自动接受新 SHA、切回旧 MHC 数据或另造未经审核的 selection。缺文件检测仅是快速拒绝，不能替代完整原始 `assert_ready`。

实际执行顺序如下：

1. stdlib 校验固定六个 CLI 选项，及唯一可选 `--resume-from-checkpoint`；mode 只能是 `train`，累计 stop 只能是整数 1000/5000。
2. 读取固定 pool admission pointer 和外部 launcher manifest，核对 launcher/dispatcher 两个源文件 SHA、pointer SHA、固定 paths/runtime/protocol。没有 pointer 或两证书之一时，在 runtime entry 执行、模型构造及输出创建前拒绝。
3. stdlib dispatcher 核验原 R20 snapshot 和固定 runtime 全部 source bytes，再普通 Python exec。沿用 `trainer_runtime_v1` guard；bridge 自身仅按经过验证的固定 path 用 runpy 执行，项目 namespace 仍来自 runtime。
4. runpy 加载固定 runtime entry，run_name 为非 `__main__`，调用其 stdlib `install_runtime_search`，不调用 `load_runtime` 或 D01 CLI。随后 import runtime 项目函数。
5. 首个实际数据/训练准入是运行时里的**原始 unchanged `assert_ready`**，包含 audit certificate、全部 shard SHA 和 downstream scope 的原始检查。它通过后，额外要求 manifest 的 `training_ready is True`。
6. 调用 unchanged `execution.preflight`，包含原 selection/profile、500-step learning diagnostic、GPU source/runtime/objective contract、完整 checkpoint sidecar；再调用 root entry 的更强 `verify_completion`。所有这些发生在模型构造和训练输出创建前。
7. 核对 root-reviewed input bytes，写独立外部 admission receipt，再调用 unchanged root `execute_admitted`。该 API 再执行自己的 original preflight、R20-only gate、完整 GPU completion verification 和原始 executor。
8. 训练完成后再次检查 bridge source、pointer/manifest 与全部已绑定输入。核对原始 runtime `COMPLETE.json` 的 R20 variant 和精确累计步数，最后写 T20 的独立 provenance 与完成 marker。

运行时项目函数在 stdlib isolation 完成后 import 是允许的；torch/transformers 的模块 import 不等于模型构造或训练准入。此桥未复制、抽取或替换原始 data gate。

## 现有工作池接口与外部来源绑定

唯一工作池是 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930`。旧 worker 仍构造既有命令形式：`--variant --mode --steps --profile-dir --output-dir`，train 时附加 `--data-dir` 与可选 resume。T20 transport profile path 必须是 pool 的 `profiles/t20_r20_certified_pilot_v1`；实际 `PilotRequest.profile_dir` 强制为既有 `profiles/r20_refined_control`。没有新增 GPU evidence、selection、shell、module 或任意 target job 字段。

GPU evidence 位置来自固定 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/pilot_admission_r20_v1.json`；只接受同 pool/runs 下根端审核的 D01 evidence 目录。此 pointer 尚未实际创建。其 schema 为 `immune.pool_pilot_admission.v1`、status 为 `root_reviewed_fixed_r20_admission_pointer`，固定 R20、T20、D01 transport variant、source/data/selection/profile/runtime path 和 runtime manifest SHA。

pointer 的 `input_sha256` 必须精确包含下列 11 个 key，不能省略或新增自由输入：

| 绑定对象 | key |
| --- | --- |
| 数据与原双证书 | `dataset_manifest.json`, `audit_certificate.json`, `downstream_scope_certificate.json` |
| 已审核 selection | `selection.json` |
| 原 R20 profile | `profile/COMPLETE.json`, `profile/summary.json` |
| 实际 D01 GPU proof | `GPU_RELEASE_COMPLETE.json`, `gpu_release_summary.json`, `gpu_release_raw.json`, `process_evidence.json`, `PRODUCER_SUCCESS.json` |

外部 launcher manifest 固定为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pool_launchers/t20_r20_pilot_v1_20261001/launcher_manifest.json`，schema 为 `immune.pool_pilot_launcher.v1`，status 为 `reviewed_transport_only_not_gpu_or_data_acceptance`。它绑定精确 source map（launcher + dispatcher）以及 admission pointer 文件 SHA。它是根端的 transport review 记录，不是 GPU/data acceptance。

准入后独立 receipt 写在 pool 的 `launcher_provenance/<job_id>.json`，output 中写 `pool_launcher_provenance.json`，schema 为 `immune.pool_launcher_provenance.v1`。这些记录 bridge source/manifest/pointer SHA、runtime/data/selection/profile/GPU evidence 及 exact input SHA；明确注明外部 bridge 没有被原 runtime GPU map 覆盖。T20 成功标记为 `POOL_LAUNCHER_COMPLETE.json`；原始 `COMPLETE.json` 仍由 unchanged R20 executor 负责。若 bridge/source/input 在执行中变化，只记录独立 launcher failure，不冒充完整 T20 成功，也不改写 runtime 的训练证据。

5000 必须显式提供同 pool prior run 的 checkpoint。原完整状态/sidecar 验证检查 model、optimizer、scheduler、RNG 等原始文件。桥还要求 prior run 的外部 receipt invariant 与当前完全相同；累计 stop/output/resume 字段除外。1000 支持原协议允许的中断 checkpoint 续跑，不能复用其他架构、数据、runtime 或桥版本的 receipt。

## 私有 stage 与发布边界

私有实现文件：

- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_20261001/isolated_worktree/scripts/architecture_prototypes/pilot_pool_launcher_v1.py`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_20261001/isolated_worktree/scripts/architecture_prototypes/pool_release_dispatcher.py`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_20261001/isolated_worktree/scripts/architecture_prototypes/stage_pool_pilot_launcher_v1.py`

`stage_pool_pilot_launcher_v1.proposed_sources(root)` 和 `stage(output_dir)` 只从**已经部署的 D01 baseline** 生成 review patch。before SHA 精确为：

| live baseline 文件 | SHA256 |
| --- | --- |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_immune_research.py` | `89e154f682e7ced2680f38bd0895eea2d18fd3447eb2e8aa0cd4184afc34fd30` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research/model.py` | `2b0b4de27f6c1dac950af510c50eb49158ae577e31e0d323d2e1be9ff370bd84` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/immune_research_pool.py` | `f3522eca58f7fedf145e8d61986d5124e0cdd9e2718c6062c53c94e563ba832f` |

runner bytes 保持 D01 early dispatcher；registry 增加 `launcher_only` T20，不可构造 model；pool strict validator 新增固定 T20 train envelope，init 排除 T20，原 28 模型 options 与 D01 单独 envelope 保持。独立 dispatcher upgrade 的 before SHA 为 `2e23bdfdbb92f61855f3b638a9f5141be01b6b48745bafdb886c821ea7cd741b`。guard 沿用原字节与 SHA `a385edb951e146d55ccf23da7078f970e9d738fdc78c5c364e17829b4e444582`，不改 immutable runtime。

当前只写了 pytest 临时目录中的 review patch。没有生成实际最终 T20 stage。根端 review 后可以把外部 launcher/stager 和单独 dispatcher upgrade 作为 transport 版本发布，并在核对 before SHA 后从部署位置调用 `stage(output_dir)`；最终 apply/pointer/入队另由根端决定。发布时必须先确认 D01 已 idle，不能在 D01 执行中改 live support。

原始监控可以读取真实 trainer_state 的累计步数；T20 不新增 typed 展示。根端发布时必须明确加入 T20 的 source provenance/runtime 映射或独立 typed read-only 分支，否则现有监督可能回退 live source root，对 immutable runtime 指纹产生误报。此候选未改 live reporting、LOCAL daemon 或 GPU worker。

固定 runtime v1 是 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v1_20261001`。父代理已核实该 D01 v1 在 240 秒 phase budget 下失败：oracle `train_runtime=701.6219` 秒，parent 约 716 秒、rc=1、cleanup=true，未产生 raw/GPU completion。该 oracle 时间不是 formal three windows 或通过证明。因此本 T20 v1 当前没有合格 GPU evidence；根端若选择新的 versioned runtime，必须新 review/version 该固定 route/pointer/source 绑定，不能热改 v1 或自动把 pointer 指向另一 runtime。

## 新 CPU 验证与冻结证据

仅运行新 T20 组，没有重跑已通过的 D01 组。最终 48 项在 pllm Python 3.10.19 和 worker Python 3.12.12 各通过；18.34 秒与 21.45 秒。测试覆盖固定 envelope、原真实 data gate 拒绝、training_ready=false、缺 pointer/cert 无 runtime/model/output 副作用、D01 baseline patch、constructor/init 排除、root R20-only gate、source/selection/raw 漂移、完整-state resume 与独立 provenance。

实际 subprocess 与 `os.posix_spawn` child 通过真实 site startup，在外部 regular packages 与 `.pth` 污染存在时继承同 runtime root/namespace/guard。该 source-only sentinel 使用明确的拒绝 gate interface，无任何 ML。另一个真实 tiny CPU 2→3 线性分类器经 unchanged root API/executor 跑 1000→5000 累计训练，恢复实际 weights、optimizer state、scheduler step，并保存 RNG/sidecar；positive admission interfaces 明确 mock，所有 toy GPU/data JSON 都标记 not acceptance，不是签发证书或真实语料训练。

最终日志/JUnit：

- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_20261001/pllm_launcher_v2_pytest.log`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_20261001/pllm_launcher_v2_junit.xml`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_20261001/worker_launcher_v2_pytest.log`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_20261001/worker_launcher_v2_junit.xml`

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_20261001/validation.json` 记录冻结 source/test/doc SHA、两环境日志/JUnit SHA 与 reviewed live baseline，状态为 CPU-only private candidate；它不代表 source 已发布或 data/GPU 已接受。后续新 runtime 工作必须保留这一版冻结证据。
