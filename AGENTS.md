# Markdown Guidelines

- Use **absolute paths** instead of relative paths.

# Python Guidelines

- At the top of each file, include a **docstring** with simple instructions on how to run the code.
- When writing new code: preview existing code first, reuse existing modules where possible, and keep the new code’s style consistent with the codebase.
- Before running scripts: activate conda env `pllm` 
- For tasks requiring a GPU, use the following command: `volc ml_task submit --conf train_jobs/<job>.yml`.
- After every `volc ml_task submit` or `cancel`, update `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md` (Active table + dated log). Remove task IDs from the Active table when jobs finish. See `.cursor/rules/volc-train-task-log.mdc`.
