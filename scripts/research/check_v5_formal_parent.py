"""Activate pllm; python -I -B /absolute/path/check_v5_formal_parent.py --help.

Read-only gate for the user-selected V5 parent, separate from the old V7 gate.
Full shard hashing is mandatory for this training gate. No training is launched.
"""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root','data-dir','data-binding','source-manifest'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('data-binding-sha256','source-manifest-sha256','variant'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--object-capacity',type=int,required=True)
    args=p.parse_args(argv)
    sys.path.insert(0,str(args.repo_root))
    from scripts.research.v5_data_binding import verify_binding,sha,require
    require(Path(__file__).resolve()==args.repo_root/'scripts/research/check_v5_formal_parent.py','Gate must run from the real repo')
    require(sha(args.source_manifest)==args.source_manifest_sha256,'Source manifest SHA differs')
    files=json.loads(args.source_manifest.read_text())['files']
    executed=[str(args.repo_root/'scripts/research'/f) for f in ('check_v5_formal_parent.py','v5_data_binding.py')]
    require(all(name in files for name in executed),'Source manifest omits V5 gate code')
    for name,expected in files.items():
        path=Path(name)
        require(path.is_absolute() and path==path.resolve() and not path.is_symlink() and sha(path)==expected,'Bound source changed: '+name)
    result=verify_binding(args.data_binding,args.data_binding_sha256,args.data_dir,args.object_capacity,args.variant)
    audit=json.loads(Path(json.loads(args.data_binding.read_text())['audit']['path']).read_text())
    require(audit['manifest_sha256']=='890d44b2ee57649d03209aa61460eaac838daf8796e5c3b76b67f1b2744c9622', 'Requires the user-selected V5 parent')
    for name,expected in audit['source_sha256'].items():
        require(files.get(name)==expected and sha(name)==expected,'Audit source changed or not bound: '+name)
    result.update(repo_root=str(args.repo_root),source_manifest_sha256=args.source_manifest_sha256,
                  executed_sources={name:files[name] for name in executed+list(audit['source_sha256'])})
    print(json.dumps(result,sort_keys=True))
    return result


if __name__=='__main__':main()
