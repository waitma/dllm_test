"""CPU protocol regressions: python -B -m architecture_hierarchy.tests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from .common import ROOT,VARIANTS,BASE_OUT,PREVIOUS_OUT,PREVIOUS_SHA,DATA_SHA,read,write,sha,digest
from .common import prepare,snapshot,previous_sources,source_binding,valid_profile,valid_qualification,completed


def profile(variant,signature='test'):
    return dict(variant=variant,micro_batch=32,global_batch=512,seed=43,length=2048,
        grad_accum=16,final_micro_batch=32,source_sha256=signature,data_sha256=DATA_SHA,
        candidates=[dict(status='passed',updates=3,safe=True,reserved_bytes=70,external_bytes=1,total_bytes=100)])


class ProtocolTests(unittest.TestCase):
    def test_profile_rejects_wrong_budget_signature_or_unsafe_peak(self):
        original=profile('B2');valid_profile(original,'B2','test')
        cases=[]
        for k,v in [('micro_batch',56),('data_sha256','wrong'),('source_sha256','wrong'),('global_batch',256)]:
            p=copy.deepcopy(original);p[k]=v;cases.append(p)
        for k,v in [('updates',2),('safe',False),('reserved_bytes',90)]:
            p=copy.deepcopy(original);p['candidates'][0][k]=v;cases.append(p)
        for candidate in cases:
            with self.assertRaises(ValueError):valid_profile(candidate,'B2','test')

    def test_common_qualification_detects_tampered_or_missing_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);profiles={}
            for variant in VARIANTS:
                p=root/'trials'/f'screen_{variant}_s43/profile.json';write(p,profile(variant));profiles[variant]=sha(p)
            q=dict(status='passed',micro_batch=32,source_sha256='test',data_sha256=DATA_SHA,profiles=profiles)
            write(root/'qualification.json',q);valid_qualification(root,'test')
            path=root/'trials/screen_B3_s43/profile.json';p=read(path);p['micro_batch']=56;write(path,p)
            with self.assertRaises(ValueError):valid_qualification(root,'test')
            write(path,profile('B3'));del q['profiles']['B0'];write(root/'qualification.json',q)
            with self.assertRaises(ValueError):valid_qualification(root,'test')

    def test_completed_skip_checks_final_weights_and_data_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);trial=out/'trials/screen_B2_s43';write(trial/'profile.json',profile('B2'))
            binding=dict(variant='B2',seed=43,epochs=1,total_steps=1276,micro_batch=32,source_sha256='test',
                data_sha256=DATA_SHA,settings=read(ROOT/'experiments/arch70m/base.json'),profile_sha256=sha(trial/'profile.json'))
            write(trial/'binding.json',binding);write(trial/'status.json',dict(status='completed'))
            cp=trial/'checkpoints/step_001276';cp.mkdir(parents=True)
            for name in ('model.safetensors','optimizer.pt','rng.pt','scheduler.pt'):(cp/name).write_bytes(b'test fixture')
            write(cp/'binding.json',binding);write(cp/'state.json',dict(step=1276,seen_records=653288,epoch=1,next_batch=0))
            write(cp/'receipt.json',dict(status='complete',files={p.name:sha(p) for p in cp.iterdir()}))
            write(trial/'metrics/screen_001276.json',dict(step=1276));self.assertTrue(completed(out,'B2','test'))
            (cp/'model.safetensors').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):completed(out,'B2','test')

    def test_protected_outputs_are_rejected_and_frozen_adapter_imports(self):
        for root in (BASE_OUT,PREVIOUS_OUT,BASE_OUT/'trials/new'):
            with self.assertRaises(ValueError):prepare(root)
        self.assertEqual(digest(previous_sources()),PREVIOUS_SHA)
        with tempfile.TemporaryDirectory() as tmp:
            source,signature=snapshot(Path(tmp))
            code="from architecture_hierarchy.runner import trainer; from architecture_hierarchy.common import source_binding,digest,ROOT; import architecture_round2.models as m; import json; t=trainer(); print(json.dumps(dict(root=str(ROOT),signature=digest(t.source_binding()),model=m.__file__)))"
            env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES='',USE_TF='0')
            output=subprocess.check_output([sys.executable,'-B','-c',code],cwd=source,env=env,text=True)
            receipt=json.loads(output.strip().splitlines()[-1])
            self.assertEqual(receipt['root'],str(source));self.assertEqual(receipt['signature'],signature)
            self.assertTrue(receipt['model'].startswith(str(source)))
            self.assertEqual(signature,digest(source_binding()))


if __name__=='__main__':unittest.main()
