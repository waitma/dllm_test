"""Independent where-bias FSDP entrypoint with the original audit/resume gates.

After conda activate pllm, inspect:
  python -B /project_c20250601/251105016/project/dllm_test/scripts/research/run_v7_fsdp_where.py --help
Use its separate YAML and source manifest. Old DDP/FSDP checkpoints are not
accepted as same-run resumes; the original entry and frozen model are unedited.
"""
import importlib.util
from pathlib import Path
import sys

ROOT = Path('/project_c20250601/251105016/project/dllm_test')
ENTRY = ROOT/'scripts/research/run_v7_fsdp.py'
ADAPTER = ROOT/'scripts/research/v7_fsdp_where_trainer.py'
BASE_ADAPTER = ROOT/'scripts/research/v7_fsdp_trainer.py'
spec = importlib.util.spec_from_file_location('_v7_where_original_entry', ENTRY)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
base.ADAPTER = ADAPTER
original_verify_sources = base.verify_sources


def verify_sources(args, loaded=False):
    files = original_verify_sources(args, loaded=loaded)
    for path in (Path(__file__).resolve(), BASE_ADAPTER):
        base.require(files.get(str(path)) == base.sha(path),
                     'Missing or changed where integration source binding: '+str(path))
    return files


base.verify_sources = verify_sources


if __name__ == '__main__':
    base.main()
