# D01 integration path and root deployment review

2026-10-01 UTC. This is a read-only follow-up review of the implemented D01 transport. No source files changed, passed CPU tests were not repeated, and no final stage, package, source application, queue entry, monitor replacement, GPU or biological-data operation was performed. The root will review the producer's private candidate, compare and swap it into the paused live source, and explicitly declare both additions frozen before the commands below become eligible.

The existing CPU evidence remains [validation.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/trainer_transport_validation/validation.json): 34 transport and 14 typed-reporting tests per environment. Its implementation-document hash refers to the unchanged [IMMUNE_TRAINER_POOL_TRANSPORT_V1.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TRAINER_POOL_TRANSPORT_V1.md). This additional review does not change that archived evidence.

## Route and ordinary child inheritance

The new independent stage changes exactly three measured files. The live `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_immune_research.py` gains an early `__main__` route before `argparse`, Torch and model imports. It runs the live stdlib dispatcher, which validates the fixed D01 five-option envelope and the entire fixed runtime source package before replacing Python with a guarded fresh process. The registry adds only D01 as `diagnostic_only`; model construction rejects it. The pool validator accepts only `{variant: d01_pilot_runtime_gpu_r20_v1, mode: profile-synthetic, steps: 1}` with exact integer steps and excludes it from init.

For D01, `IMMUNE_RELEASE_ROUTE=trainer_runtime_v1` selects only `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v1_20261001`. The live guard SHA is `a385edb951e146d55ccf23da7078f970e9d738fdc78c5c364e17829b4e444582`. `PYTHONPATH` contains the live guard directory and that selected root; `scripts` and `examples` are empty namespace packages whose paths are exactly their runtime directories. The fresh dispatcher checks the per-process guard marker and then executes only the runtime's fixed `pilot_runtime_entry_v1.py`.

The root wrapper's producer command is ordinary Python: `sys.executable -u <runtime>/scripts/architecture_prototypes/pilot_gpu_release_producer_v1.py --profile-dir <pool>/profiles/r20_refined_control --output-dir <pool>/runs/<job> --runtime-root <runtime>`. The audited `run_case_process` helper calls `os.posix_spawn` with `dict(os.environ)` and a child process group in the same session. It preserves the guard environment, selected root and namespace startup. The wrapper and producer accept the guard's already installed `scripts` and `examples` namespaces only at that exact root. The previously passed CPU tests exercised real ordinary subprocess and `posix_spawn` children with external `.pth` contamination; no GPU computation is inferred from those tests.

The frozen refined runner's `profile_unlocked` launches `[sys.executable, str(Path(__file__)), '--mode', 'profile-candidate', ...]` with no interpreter isolation flags or environment replacement. `__file__` is the selected snapshot/runtime runner file. Consequently an ordinary profile candidate inherits the same route and obtains modules from the same selected root. The legacy and architecture profile runners likewise use their own `__file__`. A runtime import of `run_immune_research` does not enter the staged live `__main__` branch: the runtime retains its unchanged parent copy. The refined runner also directs profile candidates to its own refined entry, so it does not re-enter the live model dispatcher.

The generic probe `worker_command` also uses `sys.executable` and its own resolved file without `-I`/`-S`; the same process helper preserves its environment. This confirms the ordinary-child path by source inspection only. No attention probe is part of this D01 sequence and none was executed. D01 itself reads the existing R20 profile evidence and does not start a profile scan. Standalone `-I`/`-S` launches remain outside the inherited guard contract.

The currently paused live producer still contains an old `produce()` worker launch with `-I`, an old CLI lacking `--runtime-root`, and old publication behavior. Those bytes are not a reviewed final D01 candidate and cannot establish end-to-end readiness. The private producer candidate must replace or remove that path, accept the wrapper's exact invocation, emit the root-approved candidate artifacts, and accept the guard's exact namespaces before the root declares the additions frozen. This review does not edit or run either producer copy.

## Exact stage command after the additions freeze

Use the local `pllm` environment and the canonical original repository. The final stage directory must not already exist; the helper never overwrites a stage.

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/stage_pool_trainer_runtime_routing.py \
  --output-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/stage_trainer_v1
```

Equivalent API, from the canonical original repository:

```python
from pathlib import Path
from scripts.architecture_prototypes import stage_pool_trainer_runtime_routing as staging

manifest = staging.stage(Path(
    '/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/stage_trainer_v1'))
```

`proposed_sources()` is the read-only preview API. `stage()` emits `routing.patch`, three complete sources under `proposed`, and `deployment_manifest.json`. It checks all three original baseline SHA values before writing and records the entry/producer, dispatcher, guard and typed-reporting SHA values actually reviewed. It does not create or verify an existing runtime package, apply the patch, enqueue D01 or refresh a daemon. Stage generation is independent of the historical D00 stages.

## Actual conditions before applying and executing the reviewed route

1. The root has reconciled the unexplained producer drift, reviewed the private producer diff and protocol, compared the expected paused live SHA immediately before replacement, and confirmed the final entry and producer SHA values stay unchanged. The final additions are exactly `scripts/architecture_prototypes/pilot_runtime_entry_v1.py` and `scripts/architecture_prototypes/pilot_gpu_release_producer_v1.py`. The parent already contains the package verifier and inspector; no parent source may be replaced.
2. The three live baseline SHA values still match the staged originals: runner `711e7201250be7cc554022a18a555a95c0e10655440b5563499282c142afb6fa`, registry `2cad8a5ffb787cb108f7ab22702e7476011b42a1c837f0dc21b0803ed6c66092`, and pool `f1997a0e4d15a24637fb7030500ee92ff2641107dd595872429d9bfd0875e5b2`. The root checks the emitted patch SHA, proposed source SHA values and every required live support-file hash in the deployment manifest before applying its separately reviewed measured-source change.
3. The fixed R20 parent archive remains at manifest SHA `46399204c7018d00726e3ae08142bd4eb28475c8a9a2991b3a5c90725c8824f7`; all 524 copied files, 39 measured-source hashes, eight 500-step terminal identities and original evidence bindings remain intact. The eight R20 release-map entries still point to that frozen snapshot; the other 20 historical mappings remain unchanged. No model/diagnostic job may be using the three mutable sources during the change.
4. The separately authorized package builder has produced the fixed canonical runtime and its full manifest from that parent plus the two reviewed additions. The full source map is therefore 526 files. The manifest retains source-only acceptance flags, exact parent maps, terminal identities and the two original-root resource-alias identities. Both `pilot_runtime_package.verify_runtime(runtime)` and the dispatcher source preflight must accept the final bytes. The new additions must hash-match their canonical original-root reviewed sources.
5. The actual existing pool `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930` remains initialized to that original source root and has a healthy allocated worker available, no pool `STOP`/`STOP_NOW`, an empty D01 output destination and no competing D01 invocation. The worker's child environment must expose one assigned GPU. D01 validates the unchanged `profiles/r20_refined_control` certificate, 39-source map, profile summary, production configuration, measured GPU identity and the required micro-batch/global256 plan. It does not profile into the D01 transport directory.
6. The final producer's own targeted CPU review is complete and its source/protocol matches the wrapper. Successful D01 publication requires the observed normal producer exit, descendant cleanup, verified raw three-window evidence, current runtime/producer/wrapper SHA bindings, and the wrapper's last-written `GPU_RELEASE_COMPLETE.json` and pool `COMPLETE.json`. A worker return code or a summary alone is insufficient. No D00 GPU proof is a prerequisite.
7. The local monitor has loaded the current typed D01 functions before interpreting D01 artifacts. The existing GPU workers need not restart: their in-memory old validator reads the live registry dynamically, and the new dispatcher plus runtime entry enforce the strict envelope even with that older validator.

The conditions above are a review sequence; no execution is performed by this document. Resource submission or further allocation is not required by the D01 transport itself.

## Minimal replacement of the local monitor only

Read-only observation at this review found the local monitor as PID `2033525`, parent PID `1`, started `2026-10-01 00:06:13 UTC`, hostname `di-20260121220916-9rvmj`. Its executable is the local `pllm` Python, working directory is the canonical original repository, and `/proc/2033525/cmdline` identifies only `monitor_immune_research_pool.py` with this pool, the existing LIVE document and interval 30. The latest observed heartbeat was `watching`, pending 0, running 0, results 36. These are observations, not a PID lease; re-read the identity immediately before signaling.

The root should read `monitor/pid.json`, `monitor/heartbeat.json` and the corresponding `/proc/<pid>/cmdline`, executable and start time, requiring the same local hostname, monitor script, pool and document. Send SIGTERM to only that verified monitor PID and wait for it to exit and release `monitor.lock`. Its handler sets its stop event; a currently running read-only platform query can take up to its 20-second timeout before the daemon finishes. Keep the existing monitor state, event history, logs and LIVE/SUPERVISION documents. Do not signal the GPU task, GPU workers or their groups, and do not create the pool-level `STOP` or `STOP_NOW` files. No monitor `STOP` sentinel needs to be created for this replacement.

After confirming the old monitor has exited and there is no `monitor/STOP`, start exactly one fresh local daemon with the existing arguments:

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/monitor_immune_research_pool.py \
  --pool-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930 \
  --document /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_POOL_LIVE.md \
  --interval 30 --start
```

`--start` creates a separate local session and the daemon takes the existing shared-filesystem flock. It reloads the current reporting/supervision imports and reads the old `state.json` so history is preserved. Confirm the returned new PID matches `monitor/pid.json`, its fresh heartbeat is `watching`, the log has no lock/read failure, and historical records retain their correct release roots. When D01 artifacts exist, verify the LIVE and SUPERVISION outputs show `3/3 optimizer windows`, kind `trainer_gpu_runtime_functional_diagnostic`, artificial training and `corpus_training_started=false`. Starting a second daemon while the old lock is held is not a refresh. This document has not stopped or started a monitor.

## Remaining real pretraining integration boundary

D01 is a bounded artificial Trainer GPU acceptance invocation. Its five-option pool route cannot carry corpus data, resume checkpoints or cumulative 1000/5000 pilot budgets. An ordinary `variant=r20_refined_control, mode=train` pool job still routes to the immutable refined runner and its legacy `_REAL_TRAIN`; this staged patch does not redirect it to the corrected `TokenValidationTrainer` pilot adapter or the new GPU-release admission gate.

The stronger admitted-pilot API is the fixed runtime's `pilot_runtime_entry_v1.execute_admitted(PilotRequest(...), gpu_release_dir)`, called after verified runtime isolation. It starts with the unchanged original data gate, then verified validation selection, original profile and 500-step evidence, current corrected GPU release and checkpoint admission. It finally delegates to the existing executor. Connecting that API to a final corpus-pretraining startup requires the root's separate fixed-entry/transport decision; D01 success alone neither enqueues a real pilot nor certifies real data. This review does not inspect biological dataset contents or change either existing data certificate gate.
