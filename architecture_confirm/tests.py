"""CPU confirmation checks: python -B -m architecture_confirm.tests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from experiments.arch70m.models import build as original_build
from architecture_round2.models import build as round2_build
from .models import build
from .common import ROOT,JOBS,PREVIOUS,PREVIOUS_SHA,DATA_SHA,read,write,sha,digest,source_binding,previous_sources,snapshot,prepare,trial,valid_profile,qualified
from .decision import supported,quality_guard,calculate,decision


def profile(v,s):
    return dict(variant=v,seed=s,micro_batch=32,global_batch=512,length=2048,grad_accum=16,final_micro_batch=32,
        source_sha256='test',data_sha256=DATA_SHA,
        candidates=[dict(status='passed',updates=3,safe=True,reserved_bytes=70,external_bytes=1,total_bytes=100)])


class ConfirmationTests(unittest.TestCase):
    def test_model_adapter_preserves_original_initial_weights_for_each_seed(self):
        torch.set_num_threads(2);settings=read(ROOT/'experiments/arch70m/base.json')
        settings['model'].update(d_model=48,n_heads=4,n_layers=8,mlp_hidden_size=192,
            first_chain_capacity=16,second_chain_capacity=12,full_capacity=30,max_length=128)
        for seed in (42,43,44):
            for v in ('A0','B0','B1'):
                actual=build(settings,v,seed);reference=(original_build if v=='A0' else round2_build)(settings,v,seed)
                self.assertEqual(set(actual.state_dict()),set(reference.state_dict()))
                for k,value in reference.state_dict().items():self.assertTrue(torch.equal(value,actual.state_dict()[k]),(v,seed,k))
        with self.assertRaises(ValueError):build(settings,'B2',43)

    def test_seed_profile_and_all_nine_qualification_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);hashes={}
            for v,s in JOBS:
                p=trial(out,v,s)/'profile.json';write(p,profile(v,s));valid_profile(read(p),v,s,'test');hashes[f'{v}_s{s}']=sha(p)
            write(out/'qualification.json',dict(status='passed',source_sha256='test',profiles=hashes))
            qualified(out,'test')
            p=trial(out,'B1',44)/'profile.json';wrong=read(p);wrong['seed']=43;write(p,wrong)
            with self.assertRaises(ValueError):qualified(out,'test')
            with self.assertRaises(ValueError):valid_profile(wrong,'B1',44,'test')

    def test_pending_trials_cannot_produce_a_stability_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for v,s in JOBS:write(trial(out,v,s)/'status.json',dict(status='completed' if (v,s)!=('B1',44) else 'training'))
            result=decision(out);self.assertEqual(result['decision'],'not_assessed');self.assertEqual(result['remaining'],['B1_s44'])

    def test_effect_size_consistency_and_decoder_guards_reject_false_winners(self):
        x=dict(per_seed_gain=[.03,.04,.05],mean_gain=.04,adjusted_ci=[.01,.08])
        self.assertTrue(supported(x,'generation',3))
        self.assertFalse(supported(x,'generation',1))
        mixed=dict(x,per_seed_gain=[.1,-.001,.05]);self.assertFalse(supported(mixed,'generation',3))
        self.assertFalse(supported(dict(x,mean_gain=.01),'generation',3))
        self.assertFalse(supported(dict(x,adjusted_ci=[-.001,.1]),'representation',auprc_gain=.02))
        self.assertFalse(supported(x,'representation',auprc_gain=-.001))
        base=dict(generation=dict(conditional_length_mae=10,conditional_validity=1.,conditional_aar=.3,cdr_aar=.8))
        worse=copy.deepcopy(base);worse['generation']['conditional_aar']=.29
        self.assertFalse(quality_guard(base,worse));self.assertTrue(quality_guard(base,base))

    def test_paired_bootstrap_detects_known_effects_and_rejects_misalignment(self):
        observations={};metrics={}
        for v,s in JOBS:
            generation=[];representation=[]
            for k in range(12):
                for kind in ('conditional','cdr'):
                    generation.append(dict(source='fixture',key=f'r{k}',role='antibody_heavy',kind=kind,
                                           ce={'A0':2.,'B0':1.8,'B1':1.7}[v]))
                label=k%2
                probability=(.8 if label else .2) if v!='A0' else (.2 if label else .8)
                representation.append(dict(source='fixture',key=f'r{k}',label=label,probability=probability))
            observations[v,s]=dict(generation=generation,representation=representation)
            metrics[v,s]=dict(generation=dict(conditional_length_mae=5,conditional_validity=1.,conditional_aar=.3,cdr_aar=.8),
                              representation=dict(macro_auprc=.8 if v!='A0' else .4))
        result=calculate(observations,metrics,repetitions=40)
        self.assertTrue(result['pairs']['B1_vs_A0']['generation']['stable_improvement'])
        self.assertTrue(result['pairs']['B1_vs_A0']['representation']['stable_improvement'])
        self.assertFalse(result['pairs']['B1_vs_B0']['representation']['stable_improvement'])
        observations['B1',44]['generation'][0]['key']='unpaired'
        with self.assertRaises(ValueError):calculate(observations,metrics,repetitions=2)

    def test_previous_outputs_protected_and_snapshot_import_is_frozen(self):
        with self.assertRaises(ValueError):prepare(PREVIOUS)
        self.assertEqual(digest(previous_sources()),PREVIOUS_SHA)
        with tempfile.TemporaryDirectory() as tmp:
            source,signature=snapshot(Path(tmp))
            code="from architecture_confirm.runner import trainer; from architecture_confirm.common import ROOT,digest; import architecture_confirm.models as m; import json; print(json.dumps(dict(root=str(ROOT),signature=digest(trainer().source_binding()),model=m.__file__)))"
            env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES='',USE_TF='0')
            r=json.loads(subprocess.check_output([sys.executable,'-B','-c',code],cwd=source,env=env,text=True).strip().splitlines()[-1])
            self.assertEqual(r['root'],str(source));self.assertEqual(r['signature'],signature);self.assertTrue(r['model'].startswith(str(source)))
            self.assertEqual(signature,digest(source_binding()))


if __name__=='__main__':unittest.main()
