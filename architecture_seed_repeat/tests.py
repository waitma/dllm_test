"""CPU seed replication checks: python -B -m architecture_seed_repeat.tests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from .common import ROOT, OLD, SEED, VARIANTS, PREVIOUS_SHA, DATA_SHA, COUNTS
from .common import read,write,sha,digest,previous_sources,source_binding,snapshot,valid_profile,qualified,completed,trial,validate_output
from .runner import trainer,profile_one,command


def profile(v):
    return dict(variant=v,seed=42,micro_batch=32,global_batch=512,length=2048,grad_accum=16,final_micro_batch=32,
        source_sha256='test',data_sha256=DATA_SHA,candidates=[dict(status='passed',updates=3,safe=True,reserved_bytes=70,external_bytes=1,total_bytes=100)])


class RepeatTests(unittest.TestCase):
    def test_same_frozen_model_and_training_implementation(self):
        import torch
        from architecture_attention.models import build
        from experiments.arch70m import train
        torch.set_num_threads(2)
        settings=read(ROOT/'experiments/arch70m/base.json')
        settings['model'].update(d_model=48,n_heads=4,n_layers=8,mlp_hidden_size=192,
            first_chain_capacity=16,second_chain_capacity=12,full_capacity=30,max_length=128)
        adapted=trainer();self.assertIs(adapted,train);self.assertIs(adapted.build,build)
        for v in VARIANTS:
            a,_=adapted.construct(settings,v,42,torch.device('cpu'));b=build(settings,v,42);old=build(settings,v,43)
            self.assertEqual(set(a.state_dict()),set(b.state_dict()))
            for k,x in a.state_dict().items():self.assertTrue(torch.equal(x,b.state_dict()[k]))
            self.assertTrue(any(not torch.equal(x,old.state_dict()[k]) for k,x in a.state_dict().items()))
        self.assertEqual(digest(previous_sources()),PREVIOUS_SHA)

    def test_profile_delegation_uses_seed42_and_no_profile_weight_continuation(self):
        from architecture_hierarchy import runner as memory
        from argparse import Namespace
        names=('SEED','STEPS','MICRO','GLOBAL','trainer','source_binding','valid_profile')
        previous={n:getattr(memory,n) for n in names};seen={}
        def record(args):
            seen.update(seed=memory.SEED,steps=memory.STEPS,micro=memory.MICRO,variant=args.profile_one)
        try:
            with patch.object(memory,'profile_one',record):profile_one(Namespace(output_root=Path('/tmp/repeat_fixture'),profile_one='C3'))
            self.assertEqual(seen,dict(seed=42,steps=1276,micro=32,variant='C3'))
        finally:
            for n,x in previous.items():setattr(memory,n,x)
        cmd=command(Path('/tmp/source'),Path('/tmp/out'),'C3')
        self.assertIn('architecture_seed_repeat.runner',cmd);self.assertIn('--train-one',cmd)
        self.assertNotIn('--resume',cmd)

    def test_profiles_reject_seed43_and_unsafe_or_partial_qualification(self):
        original=profile('C0');valid_profile(original,'C0','test')
        wrong=copy.deepcopy(original);wrong['seed']=43
        with self.assertRaises(ValueError):valid_profile(wrong,'C0','test')
        wrong=copy.deepcopy(original);wrong['candidates'][0]['reserved_bytes']=90
        with self.assertRaises(ValueError):valid_profile(wrong,'C0','test')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);hashes={}
            for v in VARIANTS:
                p=trial(out,v)/'profile.json';write(p,profile(v));hashes[v]=sha(p)
            q=dict(status='passed',seed=42,source_sha256='test',data_sha256=DATA_SHA,profiles=hashes)
            write(out/'qualification.json',q);qualified(out,'test')
            del q['profiles']['A0'];write(out/'qualification.json',q)
            with self.assertRaises(ValueError):qualified(out,'test')

    def test_completed_receipt_binds_seed_budget_weights_and_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);t=trial(out,'C0');write(t/'profile.json',profile('C0'))
            binding=dict(variant='C0',seed=42,epochs=1,total_steps=1276,micro_batch=32,source_sha256='test',data_sha256=DATA_SHA,
                settings=read(ROOT/'experiments/arch70m/base.json'),profile_sha256=sha(t/'profile.json'))
            write(t/'binding.json',binding);write(t/'status.json',dict(status='completed',step=1276,parameters=COUNTS['C0']))
            cp=t/'checkpoints/step_001276';cp.mkdir(parents=True)
            for name in ('model.safetensors','optimizer.pt','rng.pt','scheduler.pt'):(cp/name).write_bytes(b'fixture')
            write(cp/'binding.json',binding);write(cp/'state.json',dict(step=1276,epoch=1,next_batch=0,seen_records=653288,seen_tokens=214482921))
            write(cp/'receipt.json',dict(status='complete',files={p.name:sha(p) for p in cp.iterdir()}))
            write(t/'metrics/screen_001276.json',dict(step=1276,partition='screen',token={'all':{'loss':.5}},generation={'conditional_ce':3.},representation={'macro_auroc':.8}))
            write(t/'metrics/screen_001276_observations.json',dict(generation=[{'fixture':1}],representation=[{'fixture':1}]))
            self.assertTrue(completed(out,'C0','test'))
            binding['seed']=43;write(t/'binding.json',binding)
            with self.assertRaises(ValueError):completed(out,'C0','test')
            binding['seed']=42;write(t/'binding.json',binding);(cp/'model.safetensors').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):completed(out,'C0','test')

    def test_paired_reports_wait_for_all_models_and_preserve_seed_directions(self):
        from .report import paired,FIELDS,matched_binding
        values={(v,s):{key:float(i)+(s==42)*.2 for key in FIELDS} for i,v in enumerate(VARIANTS) for s in (42,43)}
        incomplete=dict(values);del incomplete['A0',42]
        self.assertEqual(paired(incomplete),([],[]))
        differences,means=paired(values)
        self.assertTrue(differences);self.assertTrue(means)
        c1=[r for r in means if r['variant']=='C1'][0]
        self.assertTrue(c1['both_seeds_positive']);self.assertAlmostEqual(c1['difference_from_C0_seed42'],1.)
        binding=dict(variant='C0',seed=42,epochs=1,total_steps=1276,micro_batch=56,source_sha256='test',data_sha256=DATA_SHA,
            settings=read(ROOT/'experiments/arch70m/base.json'))
        with self.assertRaises(ValueError):matched_binding(binding,'C0',42,'test')

    def test_local_gpu_lease_actually_queues_without_overlapping_work(self):
        from experiments.arch70m.common import lock
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'gpu.lock';receipt=Path(tmp)/'acquired'
            code="from experiments.arch70m.common import lock;from pathlib import Path;import sys;print('ready',flush=True)\nwith lock(Path(sys.argv[1]),blocking=True):Path(sys.argv[2]).write_text('acquired')"
            with lock(p,blocking=False):
                child=subprocess.Popen([sys.executable,'-B','-c',code,str(p),str(receipt)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=subprocess.PIPE,text=True)
                try:
                    self.assertEqual(child.stdout.readline().strip(),'ready')
                    with self.assertRaises(subprocess.TimeoutExpired):child.wait(timeout=.2)
                    self.assertFalse(receipt.exists())
                except BaseException:child.kill();child.wait();raise
            self.assertEqual(child.wait(timeout=10),0);child.stdout.close();self.assertEqual(receipt.read_text(),'acquired')

    def test_snapshot_and_dry_run_preserve_old_sources_and_output(self):
        for out in (OLD,OLD/'new',ROOT/'output/arch70m_v8_20261028'):
            with self.assertRaises(ValueError):validate_output(out)
        self.assertEqual(digest(previous_sources()),PREVIOUS_SHA)
        with tempfile.TemporaryDirectory() as tmp:
            source,signature=snapshot(Path(tmp));env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES='',USE_TF='0')
            code="from architecture_seed_repeat.runner import trainer;from architecture_seed_repeat.common import ROOT,digest,source_binding,SEED;import architecture_attention.models as m;import json;print(json.dumps(dict(root=str(ROOT),signature=digest(source_binding()),seed=SEED,model=m.__file__)))"
            r=json.loads(subprocess.check_output([sys.executable,'-B','-c',code],cwd=source,env=env,text=True).strip().splitlines()[-1])
            self.assertEqual(r['root'],str(source));self.assertEqual(r['signature'],signature);self.assertEqual(r['seed'],42)
            self.assertTrue(r['model'].startswith(str(source)))
            output=Path(tmp)/'not_created'
            subprocess.run([sys.executable,'-B','-m','architecture_seed_repeat.runner','--dry-run','--output-root',str(output)],cwd=source,env=env,check=True,capture_output=True)
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
