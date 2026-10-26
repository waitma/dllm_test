"""Check post-generation publication without publishing or training real jobs.

Activate conda pllm, then run:
python -B /project_c20250601/251105016/project/dllm_test/scripts/research/test_publish_v7_post_generation_ablation.py
Tests read actual preparation metadata and use disposable artificial stdlib
children for the publication/lock boundary. No sequences, weights or GPU calls.
"""
import contextlib
import copy
import fcntl
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path('/project_c20250601/251105016/project/dllm_test')
BASE = ROOT/'output/v7_local_ablation_20261005/post_generation_pool_v1'
spec = importlib.util.spec_from_file_location('frozen_post_generation_publisher',
    BASE/'source_v1/publish_v7_post_generation_ablation.py')
api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')
    return api.ref(path)


class PublicationBoundary(unittest.TestCase):
    def test_actual_preview_excludes_complete_and_running_original_cells(self):
        before = {p: api.sha(p) for p in BASE.rglob('*') if p.is_file()}
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(api.main(['--action', 'preview']), 0)
        value = json.loads(stdout.getvalue())
        self.assertEqual(value['status'], 'read_only_preview_not_admitted')
        self.assertNotIn(['f000', 42], value['selected_cells'])
        self.assertNotIn(['f010', 42], value['selected_cells'])
        self.assertEqual(len(value['selected_cells'])+len(value['exclusions']), 22)
        self.assertEqual(before, {p: api.sha(p) for p in BASE.rglob('*') if p.is_file()})
        self.assertFalse(any(n.split('.')[0] in {'torch','numpy','transformers','dllm'} for n in sys.modules))

    def test_actual_live_predecessor_refuses_before_publication_mutation(self):
        if not any(api.live(d) for d in (api.PARENT, api.EPOCH, api.GENERATION)):
            self.skipTest('Predecessors are now terminal; early-live refusal no longer applies')
        before = {p: api.sha(p) for p in BASE.rglob('*') if p.is_file()}
        public_before, registry_before = api.sha(api.PUBLIC), api.sha(api.REGISTRY)
        with self.assertRaisesRegex(ValueError, 'supervisor is still live'):
            api.main(['--action', 'publish'])
        self.assertEqual(before, {p: api.sha(p) for p in BASE.rglob('*') if p.is_file()})
        self.assertEqual(public_before, api.sha(api.PUBLIC)); self.assertEqual(registry_before, api.sha(api.REGISTRY))
        self.assertFalse(api.CATALOG.exists()); self.assertFalse(api.LEDGER.exists())

    def test_resource_admission_covers_all_waves_and_preserves_limits(self):
        platform = {'task':dict(JobId='t-20261005205851-lqfjg',Status='Running',Start='2026-10-05T00:00:00Z')}
        parent = {'pool_argv':['python','--max-seconds','604200','--job-seconds','86400']}
        epoch = api.datetime.fromisoformat('2026-10-05T00:00:00+00:00').timestamp()
        result = api.budget(platform,parent,12,epoch+3600)
        self.assertEqual(result['waves'],2); self.assertEqual(result['required_seconds'],173400)
        with self.assertRaisesRegex(ValueError,'Insufficient original allocation'):
            api.budget(platform,parent,12,epoch+604200-173399)
        parent['pool_argv'][-1]='90000'
        with self.assertRaisesRegex(ValueError,'deadlines changed'):
            api.budget(platform,parent,12,epoch+3600)


class ActualEntryBarrier(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(prefix='metadata_post_publish_',dir=ROOT/'output')
        self.addCleanup(tmp.cleanup); r=self.r=Path(tmp.name)
        self.lock=r/'cell.lock'; self.lock.touch(); self.marker=r/'entered.json'
        self.adapter=r/'adapter.py'; self.public=r/'public.py'; self.catalog=r/'catalog.json'; self.ledger=r/'claims.json'
        write(self.catalog,{'fixture':True}); write(self.ledger,{'fixture':True})
        self.adapter.write_text('''"""Artificial execution boundary; run only through the test launcher."""
import fcntl
import json
from pathlib import Path
def main(argv, **kwargs):
    with Path(@LOCK_PATH@).open('r') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Path(@MARKER_PATH@).write_text(json.dumps({'lock_acquired': True, 'argv': argv}))
    return 0
'''.replace('@LOCK_PATH@',repr(str(self.lock))).replace('@MARKER_PATH@',repr(str(self.marker))))
        context=patch.dict(api.__dict__,dict(BASE=r,ADAPTER=self.adapter,CATALOG=self.catalog,LEDGER=self.ledger,PUBLIC=self.public))
        context.start(); self.addCleanup(context.stop)
        before=b'"""Artificial old launcher; must not execute for the chosen alias."""\nraise RuntimeError("unexpected fallthrough")\n'
        self.public.write_bytes(api.public_extension(before,['fixture_alias'],api.sha(self.adapter),api.sha(self.catalog),api.sha(self.ledger)))
        self.receipt=r/'enqueue_receipt.json'
        write(self.receipt,{'state':'publishing','catalog':api.ref(self.catalog)})

    def launch(self):
        proc=subprocess.Popen([sys.executable,'-I','-B',str(self.public),'--variant','fixture_alias'],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        def cleanup():
            if proc.poll() is None: proc.kill()
            proc.communicate()
        self.addCleanup(cleanup)
        return proc

    def test_real_child_waits_until_batch_release_then_acquires_existing_lock(self):
        with self.lock.open('r') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            proc=self.launch(); time.sleep(.25)
            self.assertIsNone(proc.poll()); self.assertFalse(self.marker.exists())
        temporary=self.receipt.with_suffix('.new')
        write(temporary,{'state':'enqueued','catalog':api.ref(self.catalog)}); temporary.replace(self.receipt)
        stdout,stderr=proc.communicate(timeout=5)
        self.assertEqual(proc.returncode,0,stderr)
        self.assertTrue(api.read(self.marker)['lock_acquired'])

    def test_released_receipt_for_wrong_catalog_is_rejected(self):
        write(self.receipt,{'state':'enqueued','catalog':{'path':str(self.catalog),'sha256':'0'*64}})
        proc=self.launch(); stdout,stderr=proc.communicate(timeout=5)
        self.assertNotEqual(proc.returncode,0); self.assertIn('batch binding changed',stderr)
        self.assertFalse(self.marker.exists())


class InventoryBoundary(unittest.TestCase):
    def test_newer_attempt_and_completed_cells_are_never_selected(self):
        with tempfile.TemporaryDirectory(prefix='metadata_post_inventory_',dir=ROOT/'output') as tmp:
            root=Path(tmp); methods=('f000','f001','f010','f011','f100','f101','f110','f111','cv_segment_global','cv_late_fusion','fixed_gate_control')
            cells=[(v,s) for v in methods for s in (42,43)]
            rows=[dict(variant=v,seed=s,artifact_complete=False,transport='pending',status='pending') for v,s in cells]
            rows[0].update(artifact_complete=True,transport='local',status='completed_exit_unconfirmed')
            rows[1].update(transport='local',status='observed_incomplete_exit_unconfirmed')
            managed={'commands':[dict(variant=v,seed=s,output_dir=str(root/f'{v}_{s}'),log_dir=str(root/f'log_{v}_{s}')) for v,s in cells]}
            mf=root/'managed.json'; write(mf,managed)
            ledger=root/'prior.json'; write(ledger,{'cells':[dict(variant=v,seed=s,state='unclaimed') for v,s in cells]})
            (root/'f001_42').mkdir()
            write(root/'pool/results/failed.json',{'job':{'variant':'p04_v7_f001_seed43'},'state':'failed'})
            report={'runs':rows}; helpers={'MANAGED':mf,'MANAGED_SHA':api.sha(mf),'read':lambda path,digest:api.read(path)}
            with patch.dict(api.__dict__,dict(PRIOR_LEDGER=ledger,POOL=root/'pool')):
                actual,selected,excluded,prior=api.inventory(helpers,{'collect':lambda path,step:copy.deepcopy(report)})
            self.assertEqual(len(selected),18)
            self.assertEqual({tuple(x['cell']) for x in excluded},set(cells[:4]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
