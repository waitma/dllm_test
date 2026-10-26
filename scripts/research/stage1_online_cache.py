"""Atomic, append-only online teacher cache shared by independent rank workers.

Activate pllm; run python -B -m pytest scripts/tests/immune_llada/test_stage1_online.py -q.
This module uses only the standard library, so both teacher and student can import it.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


def request_digest(request):
    payload = {k: v for k, v in request.items() if k != "request_sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if request.get("request_sha256") != digest:
        raise ValueError("Online teacher request hash differs")
    return digest


def cache_path(root, digest):
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid online cache key")
    return Path(root) / digest[:2] / digest[2:4] / digest


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def cache_lock(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep lock files: unlinking them can allow two different inodes to be locked.
    with destination.with_suffix(".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def ensure_cache(root, request, build, verify):
    """Publish only complete targets; recheck after locking to deduplicate ranks."""
    destination = cache_path(root, request_digest(request))
    with cache_lock(destination):
        if destination.exists():
            verify(destination)
            return destination, False
        # A dead writer cannot leave a complete entry. Clean its private staging
        # directory only while holding this request's lock.
        for abandoned in destination.parent.glob(destination.name + ".partial.*"):
            shutil.rmtree(abandoned)
        temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".partial.", dir=destination.parent))
        try:
            build(temporary)
            verify(temporary)
            for name in ("targets.safetensors", "metadata.json"):
                with (temporary / name).open("rb") as stream:
                    os.fsync(stream.fileno())
            fsync_directory(temporary)
            temporary.rename(destination)
            fsync_directory(destination.parent)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    return destination, True
