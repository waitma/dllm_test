"""Acquire immutable official sequence-benchmark assets.

Run after activating pllm: python -m downstream.benchmark.ab_antigen.acquire_sequence
Downloads author data, never rewrites author code or fabricates missing labels.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import requests
import argparse
import time

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / 'data/downstream/ab_antigen_sequence_v1/raw'
SOURCES = {
    'abbibench': ('AbBibench/Antibody_Binding_Benchmark_Dataset', '556fd6913aa231c0d342a8658818be8f963fd582'),
    'chimera': ('mansoorbaloch/chimera-bench', '54d5a2a1e59a329db7a6c09184652ed225188940'),
}


def fetch(args):
    name, repo, rev, filename = args
    path = DEST / name / filename
    url = f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{filename}'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        for attempt in range(4):
            try:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        tmp = path.with_suffix(path.suffix + '.part')
        tmp.write_bytes(response.content)
        tmp.replace(path)
    return dict(source=name, revision=rev, file=str(path.relative_to(ROOT)), url=url,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', action='store_true')
    args = parser.parse_args()
    tasks = []
    for name, (repo, rev) in SOURCES.items():
        response = requests.get(f'https://huggingface.co/api/datasets/{repo}/revision/{rev}', timeout=60)
        response.raise_for_status()
        for item in response.json()['siblings']:
            f = item['rfilename']
            if f.endswith(('.md', '.json', '.csv', '.fasta')) or (name == 'abbibench' and f.endswith('.pdb')) or (args.features and f.startswith('complex_features/') and f.endswith('.pt')):
                tasks.append((name, repo, rev, f))
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(fetch, tasks))
    (DEST / 'acquisition.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps({'files': len(rows), 'bytes': sum(r['bytes'] for r in rows)}))


if __name__ == '__main__':
    main()
