"""Fetch pinned teacher sources and official weights for a small validation run.

Activate pllm, then run this absolute file. Downloads are resumable, are stored
under a new experiment directory, and never modify existing model weights.
"""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request

ROOT = Path('/project_c20250601/251105016/project/dllm_test/output/general_teacher_validation_20261024')
WEIGHTS = Path('/project_c20250601/251105016/model_weights/general_teacher_validation_20261024')
SOURCES = {
    'protenix': ('bytedance/Protenix', '85767b811c40ed46e73a9b39519cf6bfca8701ba'),
    'boltz': ('jwohlwend/boltz', 'b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc'),
}
BASE = 'https://protenix.tos-cn-beijing.volces.com/checkpoint/'
ASSETS = [
    ('protenix_mini_esm_v0.5.0.pt', BASE + 'protenix_mini_esm_v0.5.0.pt', 541640990),
    ('esm2_t36_3B_UR50D.pt', BASE + 'esm2_t36_3B_UR50D.pt', 5678116398),
    ('esm2_t36_3B_UR50D-contact-regression.pt', BASE + 'esm2_t36_3B_UR50D-contact-regression.pt', None),
    ('boltz2_conf.ckpt', 'https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt', 2286561469),
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(item):
    name, url, expected_size = item
    destination = WEIGHTS / name
    partial = destination.with_name(name + '.part')
    row = dict(name=name, url=url, path=str(destination))
    try:
        if not destination.exists():
            subprocess.run(['curl', '-fL', '--retry', '3', '--connect-timeout', '30',
                            '--max-time', '2400', '--continue-at', '-', '--silent',
                            '--show-error', '--output', str(partial), url], check=True)
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise ValueError('Unexpected size: ' + str(partial.stat().st_size))
            partial.replace(destination)
        if expected_size is not None and destination.stat().st_size != expected_size:
            raise ValueError('Existing file has unexpected size')
        row.update(status='downloaded', bytes=destination.stat().st_size,
                   sha256=digest(destination))
    except Exception as exc:
        row.update(status='failed', error=str(exc))
    print(json.dumps(row), flush=True)
    return row


def source(key, repo, revision):
    destination = ROOT / 'sources' / key
    url = f'https://codeload.github.com/{repo}/tar.gz/{revision}'
    archive = ROOT / (key + '.tar.gz')
    with urllib.request.urlopen(url, timeout=60) as response, archive.open('wb') as f:
        while chunk := response.read(1024 * 1024):
            f.write(chunk)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar:
            relative = Path(*Path(member.name).parts[1:])
            if not member.isfile() or '..' in relative.parts:
                continue
            if member.size > 20_000_000:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
    return dict(repo=repo, revision=revision, path=str(destination),
                archive_sha256=digest(archive), url=url)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(), sources={}, weights=[])
    for key, (repo, revision) in SOURCES.items():
        report['sources'][key] = source(key, repo, revision)
        (ROOT / 'assets.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(event='source_ready', key=key)), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(download, ASSETS):
            report['weights'].append(row)
            (ROOT / 'assets.json').write_text(json.dumps(report, indent=2) + '\n')
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report['status'] = 'passed' if all(x['status'] == 'downloaded' for x in report['weights']) else 'failed'
    (ROOT / 'assets.json').write_text(json.dumps(report, indent=2) + '\n')
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
