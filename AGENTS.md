## Architecture exploration scope (2026-10-28)

The user explicitly authorized local single-GPU ablations and publication to `explore/arch70m-v8-20261028`. This worktree is `/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore`. Keep the production worktree and active training bundles read-only. Update documentation in this exploration worktree. Four GPUs run independent trials. See docs/ARCH70M_EXPLORATION.md for the approved protocol.

User constraint reaffirmed 2026-10-28: preserve the current V8 prepared JSONL record format and model input conventions throughout architecture exploration. Keep chain sequences/order/roles, metadata, relation/family labels, tokenizer/vocabulary, special tokens and fixed-chain layout unchanged. The approved 10% sampling changes only selected record offsets; experiment manifests and evaluation bookkeeping are separate from the training record schema. Reuse the frozen reader/collator. Do not introduce data-format migrations as part of architecture trials.

Current execution allocation (latest user instruction 2026-10-29, UTC+8): provide `launch_arch70m_8gpu.sh` for A0-A7 on eight independent single-GPU processes, adding only seed43. Each new trial is one epoch, matched to the completed seed42 screening, with global512 and a common micro56 (configurable uniformly). The original 10% V8 subset and frozen source SHA remain binding. The earlier nine-trial, three-seed confirmation queue is superseded: its dispatcher PID122889 was stopped on 2026-10-29T00:00:55Z; its already-running A0 seed42 worker was retained to finish. Do not restart that queue or schedule seed44. The old selection.json and confirmation artifacts are historical evidence, not the new execution plan. The eight-GPU script is delivered for user execution; do not submit remote jobs or start duplicate local workers. Continuous active monitoring remains paused at the user's request. Publish only on the exploration branch.

# Markdown Guidelines

- This ownkaifa02 copy now includes V8 four-relations r2, current V7 diffusion inputs, the latest V-family scratch launch bundles, and current AB/TCR evaluation inputs. Transfer receipts and runtime path/hash migration requirements: `/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/OWNKAIFA02_TRANSFER_20261027.md`. No training/evaluation was started by the transfer.

- Current storage root: `/mnt/project-storage/fset-rN4kfh/251105016`; project root: `/mnt/project-storage/fset-rN4kfh/251105016/dllm_test`. Use these actual paths in code, data/config/checkpoint metadata, and docs. Do not add old-mount aliases or runtime prefix adapters. A path rename must update dependent integrity hashes together and pass the normal data/checkpoint gates.

- Use **absolute paths** instead of relative paths.
- After any change, update the matching project md in the **same turn**: task/experiment status → `PROJECT_PROCESS.md`; architecture/method/plan decisions → `BIOSEQ_MODEL_PLAN.md`; long-term rules/paths → `PROJ_GUIDE.md`; data format/layout → `DATA_FORMAT_AUDIT.md`. See `.cursor/rules/doc-sync.mdc`.

# Python Guidelines

- At the top of each file, include a **docstring** with simple instructions on how to run the code.
- When writing new code: preview existing code first, reuse existing modules where possible, and keep the new code’s style consistent with the codebase.
- Before running scripts: activate conda env `pllm` 
- For GPU training jobs: `volc ml_task submit --conf train_jobs/<job>.yml`.
- For GPU eval jobs (downstream / pairing): `volc ml_task submit --conf eval_jobs/<job>.yml`.
- After every `volc ml_task submit` or `cancel`, update `/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/PROJECT_PROCESS.md` (Active table + dated log). Remove task IDs from the Active table when jobs finish. See `.cursor/rules/volc-train-task-log.mdc`.
