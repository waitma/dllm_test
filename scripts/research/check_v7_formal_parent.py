"""Read-only parent gate in a clean process rooted at the real repository.

After conda activate pllm: python -I -B /absolute/real/repo/scripts/research/
check_v7_formal_parent.py --repo-root /absolute/real/repo --data-dir /absolute/data
--source-manifest /absolute/sources.json --source-manifest-sha256 SHA256.
The formal driver invokes this helper with CUDA hidden. It calls the original
assert_ready, writes no files/certificates, and prints only metadata/source hashes.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

GATE_FILES = ('examples/llada/immune_pretrain_v3.py', 'scripts/data/tcr_scope_gate.py',
              'scripts/data/immune_test_scope_policy.py', 'scripts/data/audit_tcr_exposure.py',
              'scripts/data/prepare_immune_v7.py', 'scripts/research/check_v7_formal_parent.py')
FINAL_MANIFEST = '28e3cbf8fb860c13b514a11d600579257a233fefe9f70ea070b042d04f8a424c'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_sources(args, loaded=False):
    require(sha(args.source_manifest) == args.source_manifest_sha256, 'Source manifest SHA differs')
    files = json.loads(args.source_manifest.read_text())['files']
    executed = {str(args.repo_root/name) for name in GATE_FILES}
    if loaded:
        for name, module in tuple(sys.modules.items()):
            filename = getattr(module, '__file__', None)
            if filename and name.split('.')[0] in ('dllm', 'scripts', 'examples'):
                path = Path(filename).resolve()
                require(path.is_relative_to(args.repo_root), 'Audit imported another source root')
                if path.suffix == '.py':
                    executed.add(str(path))
    require(executed <= files.keys(), 'Source manifest omits real gate imports: '+str(sorted(executed-files.keys())))
    for name, expected in files.items():
        path = Path(name)
        require(path.is_absolute() and path == path.resolve() and path.is_file()
                and not path.is_symlink() and sha(path) == expected, 'Bound source changed: '+name)
    return {name: files[name] for name in sorted(executed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root', 'data-dir', 'source-manifest'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    args = parser.parse_args(argv)
    require(all(path.is_absolute() and path == path.resolve() for path in
        (args.repo_root, args.data_dir, args.source_manifest)), 'Canonical absolute paths required')
    require(Path(__file__).resolve() == args.repo_root/GATE_FILES[-1], 'Helper must execute from the real repo')
    check_sources(args)
    sys.path.insert(0, str(args.repo_root))
    names = ('dataset_manifest.json', 'audit_certificate.json', 'downstream_scope_certificate.json')
    before = {name: sha(args.data_dir/name) for name in names}
    require(before[names[0]] == FINAL_MANIFEST, 'Requires the final audited V7 parent')
    with contextlib.redirect_stdout(io.StringIO()):
        from examples.llada.immune_pretrain_v3 import assert_ready
        from scripts.data import immune_test_scope_policy as scope
        require(scope.ROOT.resolve() == args.repo_root, 'Original scope resources use another root')
        manifest, _ = assert_ready(args.data_dir)
        require(manifest.get('training_ready') is True, 'Parent is not ready')
    executed = check_sources(args, loaded=True)
    require({name: sha(args.data_dir/name) for name in names} == before, 'Parent metadata changed during audit')
    result = {'schema': 'v7.formal.read_only_parent_check.v1', 'status': 'passed',
        'repo_root': str(args.repo_root), 'data_dir': str(args.data_dir), 'metadata': before,
        'source_manifest_sha256': args.source_manifest_sha256, 'executed_sources': executed}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
