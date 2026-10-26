"""Test current-path data binding and rejection of changed bytes or shard targets.

Activate pllm; python -B /ABS/scripts/tests/immune_llada/test_diffusion_resume_data.py.
Uses only temporary files and synthetic manifest metadata, with no GPUs.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import os
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('USE_TF', '0')
from scripts.research import run_bert_mlm as runtime, bert_mlm_v7_data as binding


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ResumeDataPathTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.current = self.base / 'current'
        self.current.mkdir()
        self.addCleanup(patch.stopall)

    def fixture(self):
        release = self.current / 'release'
        parent = self.current / 'parent'
        (release / 'train').mkdir(parents=True)
        (parent / 'train').mkdir(parents=True)
        shard = parent / 'train/one.jsonl'
        shard.write_text('{}\n')
        (release / 'train/one.jsonl').symlink_to(shard)
        model = dict(first_chain_capacity=167, second_chain_capacity=135,
                     full_capacity=304, max_length=2048)
        save(release / 'config.json', model)
        save(parent / 'dataset_manifest.json', {})
        layout = self.current / 'dllm/pipelines/immune_llada/fixed_chain/layout.py'
        layout.parent.mkdir(parents=True)
        layout.write_text('# bound layout\n')
        manifest = dict(reader_contract=dict(layout='immune.fixed_chain.f011.v1',
            a4_allowed=False, config_path='config.json', config_sha256=sha(release / 'config.json')),
            mhc_completion=dict(parent=str(parent), parent_manifest_sha256=sha(parent / 'dataset_manifest.json')),
            splits=dict(train=dict(records=1, shards=[dict(path='train/one.jsonl', source='test',
                records=1, sha256=sha(shard))]), valid=dict(records=0, shards=[])))
        save(release / 'dataset_manifest.json', manifest)
        release_sha = sha(release / 'dataset_manifest.json')
        for name in ('verification.json', 'independent_acceptance.json', 'reader_acceptance.json'):
            save(release / name, dict(status='passed', dataset_manifest_sha256=release_sha,
                 config_sha256=sha(release / 'config.json'), collator_sha256=sha(layout)))
        patch.object(binding, 'ROOT', self.current).start()
        patch.object(binding, 'RELEASE_SHA', release_sha).start()
        bound = binding.release_binding(release)
        bound['shards'][0]['resolved_path'] = str(shard)
        save(self.current / 'binding.json', bound)
        offset = self.current / 'offset.npy'
        offset.write_bytes(b'synthetic offsets')
        development = self.current / 'dev.jsonl'
        development.write_text('{}\n')
        index = dict(binding_kind=binding.BINDING_SCHEMA, binding=str(self.current / 'binding.json'),
            binding_sha256=sha(self.current / 'binding.json'), manifest=str(release / 'dataset_manifest.json'),
            manifest_sha256=release_sha, records=dict(train=1, valid=0),
            shards=[dict(bound['shards'][0], index=str(offset), index_sha256=sha(offset))],
            files=dict(dev=dict(path=str(development), sha256=sha(development))))
        save(self.current / 'index.json', index)
        plan = dict(data_index=str(self.current / 'index.json'),
                    data_index_sha256=sha(self.current / 'index.json'), model=model)
        return plan, release, shard

    def test_accepts_current_paths_and_parent_shard_with_all_hashes(self):
        plan, release, shard = self.fixture()
        self.assertEqual(runtime.data_gate(plan)['train'], 1)

    def test_rejects_retarget_to_identical_bytes(self):
        plan, release, shard = self.fixture()
        alternate = shard.with_name('other.jsonl')
        alternate.write_bytes(shard.read_bytes())
        link = release / 'train/one.jsonl'
        link.unlink()
        link.symlink_to(alternate)
        with self.assertRaisesRegex(ValueError, 'symlink target changed'):
            runtime.data_gate(plan)

    def test_rejects_outside_release_even_with_identical_bytes(self):
        plan, release, shard = self.fixture()
        outside = self.base / 'outside/train/one.jsonl'
        outside.parent.mkdir(parents=True)
        outside.write_bytes(shard.read_bytes())
        link = release / 'train/one.jsonl'
        link.unlink()
        link.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'outside the release'):
            runtime.data_gate(plan)

    def test_rejects_changed_shard_bytes(self):
        plan, release, shard = self.fixture()
        shard.write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'Shard/index content changed'):
            runtime.data_gate(plan)

    def test_rejects_changed_index_bytes(self):
        plan, release, shard = self.fixture()
        (self.current / 'offset.npy').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Shard/index content changed'):
            runtime.data_gate(plan)

    def test_rejects_changed_development_bytes(self):
        plan, release, shard = self.fixture()
        (self.current / 'dev.jsonl').write_text('changed\n')
        with self.assertRaisesRegex(ValueError, 'Development set changed'):
            runtime.data_gate(plan)

    def test_rejects_changed_data_index(self):
        plan, release, shard = self.fixture()
        (self.current / 'index.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Frozen data index changed'):
            runtime.data_gate(plan)

    def test_rejects_changed_binding(self):
        plan, release, shard = self.fixture()
        (self.current / 'binding.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'data binding changed'):
            runtime.data_gate(plan)


if __name__ == '__main__':
    unittest.main()
