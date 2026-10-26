"""Download the pinned ESMFold2-Fast snapshot without importing student modules.

In pllm: HF_HUB_DISABLE_XET=1 python -B -m scripts.research.download_stage1_teacher
  --output /project_c20250601/251105016/model_weights/esmfold2_fast/45fe8656f5b3ef493c17fcf9abe9a2968902e712
The teacher loader independently verifies every shard against Hub SHA metadata.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from scripts.research.stage1_teacher import REPOSITORY, REVISION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    root=Path(args.output).resolve()
    snapshot_download(REPOSITORY,revision=REVISION,local_dir=root,
                      allow_patterns=["*.json","*.safetensors","README.md"],max_workers=6)
    (root / "download_provenance.json").write_text(json.dumps(dict(
        repository=REPOSITORY,revision=REVISION,local_dir=str(root)),indent=2)+"\n")
    print(str(root))


if __name__ == "__main__":
    main()
