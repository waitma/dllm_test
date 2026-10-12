"""Persist test metrics without model files; imported by the V7 test runners.

Activate conda pllm. This module uses only the standard library and does not
load a model, dataset, or checkpoint. Explicit checkpoint saving is optional.
"""
import json
from pathlib import Path


def record_observation(directory, point, *, save_checkpoint=False, checkpoint_writer=None):
    """Record every evaluation even when checkpoint saving is disabled."""
    directory = Path(directory)
    if save_checkpoint:
        if checkpoint_writer is None:
            raise ValueError('Checkpoint saving requires an explicit writer')
        checkpoint_writer()
    with (directory / 'curve.jsonl').open('a') as handle:
        handle.write(json.dumps(point, allow_nan=False) + '\n')
    temporary = directory / 'summary.json.tmp'
    temporary.write_text(json.dumps(point, indent=2, allow_nan=False) + '\n')
    temporary.replace(directory / 'summary.json')
