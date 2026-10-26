"""Observe native checkpoint restoration without changing training state.

Imported by /project_c20250601/251105016/project/dllm_test/scripts/research/
run_v7_formal.py after frozen-source isolation. No standalone training command.
Activate conda pllm before running the companion CPU control script. Importing
this module uses only stdlib; tensor libraries are imported inside observations.
"""
import hashlib
import inspect
import json
import math
from pathlib import Path
import random


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Expected a regular nonsymlink file')
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def method_source(method):
    method = getattr(method, '__func__', method)
    path = Path(inspect.getsourcefile(method)).resolve()
    return {'path': str(path), 'file_sha256': sha(path),
            'method': method.__qualname__,
            'method_sha256': hashlib.sha256(inspect.getsource(method).encode()).hexdigest()}


def state_digest(value):
    """Hash all typed state contents; device placement is intentionally excluded."""
    import numpy as np
    import torch
    result = hashlib.sha256()
    def emit(tag, data=b''):
        result.update(tag.encode()+b':'+str(len(data)).encode()+b':'+data)
    def visit(item):
        if torch.is_tensor(item):
            value = item.detach().cpu().contiguous()
            require(bool(torch.isfinite(value).all()), 'Nonfinite tensor state')
            emit('tensor', json.dumps([str(value.dtype), list(value.shape)]).encode())
            emit('bytes', value.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            require(not item.dtype.hasobject and bool(np.isfinite(item).all()), 'Invalid numpy state')
            emit('ndarray', json.dumps([item.dtype.str, list(item.shape)]).encode())
            emit('bytes', item.tobytes(order='C'))
        elif isinstance(item, np.generic):
            emit('numpy_scalar', str(item.dtype).encode())
            visit(item.item())
        elif type(item) is dict:
            require(all(type(key) in (int, str) for key in item), 'Unsupported state key')
            emit('dict', str(len(item)).encode())
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
        elif type(item) in (list, tuple):
            emit(type(item).__name__, str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            require(item is None or type(item) in (str, int, float, bool), 'Unsupported state value')
            require(type(item) is not float or math.isfinite(item), 'Nonfinite scalar state')
            emit(type(item).__name__, json.dumps(item, allow_nan=False).encode())
    visit(value)
    return result.hexdigest()


def rng_state(include_cuda=True):
    import numpy as np
    import torch
    result = {'python': random.getstate(), 'numpy': np.random.get_state(),
              'cpu': torch.random.get_rng_state()}
    if include_cuda:
        # The formal launcher already requires exactly one initialized GPU.
        result['cuda'] = torch.cuda.random.get_rng_state_all()
    return result


class ResumeObservation:
    """One rank's observations; construction follows the driver's resume gate."""
    def __init__(self, checkpoint, sidecar_sha256, rank, binding_sha256, native_methods):
        self.checkpoint = Path(checkpoint)
        require(self.checkpoint.is_absolute() and self.checkpoint == self.checkpoint.resolve(),
                'Checkpoint must be canonical')
        require(type(rank) is int and 0 <= rank < 8, 'Invalid observation rank')
        self.rank, self.binding_sha256 = rank, binding_sha256
        self.sidecar_sha256 = sidecar_sha256
        self.sidecar = self._sidecar()
        require(self.sidecar['binding_sha256'] == binding_sha256 and self.sidecar['world_size'] == 8
                and self.sidecar['rank_acknowledgments'] == list(range(8)), 'Resume binding differs')
        self.native_methods = native_methods
        self.sources = {key: method_source(value) for key, value in native_methods.items()}
        self.events = []

    def _sidecar(self):
        path = self.checkpoint/'formal_checkpoint.json'
        require(sha(path) == self.sidecar_sha256, 'Resume sidecar SHA changed')
        return json.loads(path.read_text())

    def _read(self, name):
        import torch
        self._sidecar()
        path = self.checkpoint/name
        expected = self.sidecar['files'][name]
        require(sha(path) == expected, 'Saved state SHA differs: '+name)
        value = torch.load(path, map_location='cpu', weights_only=False)
        require(sha(path) == expected, 'Saved state changed while reading: '+name)
        return value

    def observe(self, kind, trainer, checkpoint, native_method, *, include_cuda=True):
        require(Path(checkpoint) == self.checkpoint, 'Native method used another checkpoint')
        require(getattr(native_method, '__func__', native_method) is self.native_methods[kind],
                'Native recovery method was replaced')
        require(method_source(native_method) == self.sources[kind], 'Native method source changed')
        before = state_digest(rng_state(include_cuda))
        if kind == 'optimizer_scheduler':
            actual = {'optimizer.pt': trainer.optimizer.state_dict(),
                      'scheduler.pt': trainer.lr_scheduler.state_dict()}
        elif kind == 'rng':
            require(trainer.args.process_index == self.rank, 'Native RNG rank differs')
            actual = {f'rng_state_{self.rank}.pth': rng_state(include_cuda)}
        else:
            raise ValueError('Unknown observation kind')
        matches = {}
        for name, value in actual.items():
            saved_digest, memory_digest = state_digest(self._read(name)), state_digest(value)
            require(saved_digest == memory_digest, 'Native restored memory differs: '+name)
            matches[name] = {'file_sha256': self.sidecar['files'][name],
                             'saved_state_sha256': saved_digest, 'memory_state_sha256': memory_digest}
        require(before == state_digest(rng_state(include_cuda)), 'Observation changed RNG state')
        self._sidecar()
        self.events.append({'kind': kind, 'native_return_observed': True,
                            'native_source': self.sources[kind], 'matches': matches})

    def receipt(self, final_step):
        require(type(final_step) is int and final_step > self.sidecar['step'], 'No actual resumed updates')
        require({event['kind'] for event in self.events} == {'optimizer_scheduler', 'rng'},
                'Both native recovery observations must occur')
        return {'rank': self.rank, 'checkpoint': str(self.checkpoint),
                'sidecar_sha256': self.sidecar_sha256, 'binding_sha256': self.binding_sha256,
                'prior_step': self.sidecar['step'], 'final_step': final_step, 'events': self.events}


def complete_receipt(receipts):
    require(len(receipts) == 8 and [item['rank'] for item in receipts] == list(range(8)),
            'All eight actual rank receipts are required')
    common = ('checkpoint', 'sidecar_sha256', 'binding_sha256', 'prior_step', 'final_step')
    for item in receipts:
        require(all(item[key] == receipts[0][key] for key in common), 'Rank resume identity differs')
        require(item['final_step'] > item['prior_step'] and
                {event['kind'] for event in item['events']} == {'optimizer_scheduler', 'rng'},
                'Missing native restore or resumed update')
        for event in item['events']:
            reference = next(value for value in receipts[0]['events'] if value['kind'] == event['kind'])
            require(event['native_return_observed'] is True and
                    event['native_source'] == reference['native_source'], 'Native source differs across ranks')
            expected = {'optimizer.pt', 'scheduler.pt'} if event['kind'] == 'optimizer_scheduler' else {
                f'rng_state_{item["rank"]}.pth'}
            require(set(event['matches']) == expected and all(
                value['saved_state_sha256'] == value['memory_state_sha256'] for value in event['matches'].values()),
                'Incomplete or mismatched restored state evidence')
    return {'schema': 'v7.formal.native_resume_observation.v1', 'world_size': 8,
            **{key: receipts[0][key] for key in common}, 'ranks': receipts,
            'claim': 'Native restored memory matched saved state at method return; resumed updates completed.',
            'continuous_trajectory_equivalence_tested': False}
