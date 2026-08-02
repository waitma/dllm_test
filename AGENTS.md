# Markdown Guidelines

- Use **absolute paths** instead of relative paths.
- After any change, update the matching project md in the **same turn**: task/experiment status → `PROJECT_PROCESS.md`; architecture/method/plan decisions → `BIOSEQ_MODEL_PLAN.md`; long-term rules/paths → `PROJ_GUIDE.md`; data format/layout → `DATA_FORMAT_AUDIT.md`. See `.cursor/rules/doc-sync.mdc`.

# Python Guidelines

- At the top of each file, include a **docstring** with simple instructions on how to run the code.
- When writing new code: preview existing code first, reuse existing modules where possible, and keep the new code’s style consistent with the codebase.
- Before running scripts: activate conda env `pllm` 
- For GPU training jobs: `volc ml_task submit --conf train_jobs/<job>.yml`.
- For GPU eval jobs (downstream / pairing): `volc ml_task submit --conf eval_jobs/<job>.yml`.
- After every `volc ml_task submit` or `cancel`, update `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md` (Active table + dated log). Remove task IDs from the Active table when jobs finish. See `.cursor/rules/volc-train-task-log.mdc`.
