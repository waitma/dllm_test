"""CPU architecture/protocol regressions: python -B -m architecture_attention.tests."""
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from safetensors.torch import load_file, save_file
from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyCollator, corrupt
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
from experiments.arch70m.models import build as original_build
from experiments.arch70m.train import update, save_checkpoint, restore, make_scheduler
from .models import build, FACTORS
from .common import ROOT, VARIANTS, COUNTS, DATA_SHA, PREVIOUS_SHA, BASE
from .common import read, write, sha, digest, previous_sources, source_binding, snapshot, prepare
from .common import trial, valid_profile, qualified, completed


def profile(v, signature='test'):
    return dict(variant=v, micro_batch=32, global_batch=512, seed=43, length=2048,
        source_sha256=signature, data_sha256=DATA_SHA, grad_accum=16, final_micro_batch=32,
        candidates=[dict(status='passed', updates=3, safe=True, reserved_bytes=70, external_bytes=1, total_bytes=100)])


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings = read(ROOT / 'experiments/arch70m/base.json')
        cls.settings['model'].update(d_model=48, n_heads=4, n_layers=8, mlp_hidden_size=192,
            first_chain_capacity=16, second_chain_capacity=12, full_capacity=30, max_length=128)
        cls.records = [BioSeqRecord([BioSeqChain('LAGVSERT','antibody_heavy'),BioSeqChain('IDPKQN','antibody_light')],
            'antibody','oas_paired',labels={'relation':'binding'}) for _ in range(2)]

    def clean(self, model): return VFamilyCollator(model.config, family_spec=model.family_spec)(self.records)

    def test_shared_initial_weights_inputs_and_global_masks_preserved(self):
        for seed in (42,43,44):
            original = original_build(self.settings, 'A5', seed)
            clean = self.clean(original)
            for v in VARIANTS:
                reference_model = original_build(self.settings, 'A0', seed) if v == 'A0' else original
                before = torch.get_rng_state().clone(); model = build(self.settings, v, seed)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                for name, value in original.state_dict().items(): self.assertTrue(torch.equal(value, model.state_dict()[name]), (v,name))
                for name, value in clean.items(): self.assertTrue(torch.equal(value,self.clean(model)[name]),(v,name))
                for i in range(8): self.assertEqual(model.mask_key(i),reference_model.mask_key(i))
                biases = model._build_attention_biases(clean['chain_ids'],clean['position_ids'],clean['attention_mask'])
                reference = reference_model._build_attention_biases(clean['chain_ids'],clean['position_ids'],clean['attention_mask'])
                self.assertEqual(set(biases),set(reference))
                for key in biases:self.assertTrue(torch.equal(biases[key],reference[key]))

    def test_gate_is_identity_at_initialization_for_both_norm_settings(self):
        clean = self.clean(build(self.settings, 'C0'))
        for a, b in (('C0','C2'),('C1','C3')):
            first, second = build(self.settings,a), build(self.settings,b)
            noisy = corrupt(clean,first.tokenizer,torch.Generator().manual_seed(7))
            torch.testing.assert_close(first(**noisy)['logits'],second(**noisy)['logits'],atol=0,rtol=0)
            for block in second.backbone.transformer.blocks:
                self.assertEqual(float(block.output_gate.weight.abs().sum()),0.)
                self.assertEqual(float(block.output_gate.bias.abs().sum()),0.)
                self.assertTrue(torch.equal(block.attention_scale,torch.full((48,),.1)))

    def test_both_new_factors_receive_finite_nonzero_gradients(self):
        for v in ('C1','C2','C3'):
            model = build(self.settings,v)
            batch = corrupt(self.clean(model),model.tokenizer,torch.Generator().manual_seed(9))
            result = model(**batch);(result['loss_sum']+.1*result['family_loss_sum']).backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad,(v,name));self.assertTrue(torch.isfinite(parameter.grad).all(),(v,name))
                if any(key in name for key in ('head_norm','output_gate')):
                    self.assertGreater(float(parameter.grad.abs().sum()),0.,(v,name))

    def test_checkpointed_backward_and_optimizer_rng_resume_match(self):
        plain = build(self.settings,'C3');checkpointed = copy.deepcopy(plain)
        checkpointed.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
        batch = corrupt(self.clean(plain),plain.tokenizer,torch.Generator().manual_seed(11))
        for model in (plain,checkpointed):
            result=model(**batch);(result['loss_sum']+.1*result['family_loss_sum']).backward()
        for (name,a),(_,b) in zip(plain.named_parameters(),checkpointed.named_parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-5,msg=name)
        a,b=build(self.settings,'C3'),build(self.settings,'C3')
        optimizers=[torch.optim.AdamW(m.parameters(),lr=1e-4) for m in (a,b)]
        schedules=[make_scheduler(o,10,1) for o in optimizers];noise=torch.Generator().manual_seed(13)
        update(a,batch,optimizers[0],schedules[0],1,torch.device('cpu'))
        with tempfile.TemporaryDirectory() as tmp:
            cp=save_checkpoint(Path(tmp),a,optimizers[0],schedules[0],noise,dict(step=1),dict(test='attention'))
            update(a,batch,optimizers[0],schedules[0],1,torch.device('cpu'))
            restore(cp,b,optimizers[1],schedules[1],noise,dict(test='attention'))
            update(b,batch,optimizers[1],schedules[1],1,torch.device('cpu'))
            for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)

    def test_protein_block_equivariance_and_new_parameter_roundtrip(self):
        for v in ('C1','C2','C3'):
            model=build(self.settings,v).eval()
            for block in model.backbone.transformer.blocks:
                if hasattr(block,'output_gate'):torch.nn.init.normal_(block.output_gate.weight,std=.02)
            features=model.feature_args(self.clean(model));ids=features['chain_ids'][0];chunks=[];start=0
            for i in range(1,len(ids)):
                if ids[i]!=ids[i-1]:chunks.append(torch.arange(start,i));start=i
            chunks.append(torch.arange(start,len(ids)));perm=torch.cat(chunks[::-1]);inverse=perm.argsort()
            changed={key:value[:,perm] for key,value in features.items()}
            a,h=model.denoise(**features,output_hidden_states=True);b,j=model.denoise(**changed,output_hidden_states=True)
            torch.testing.assert_close(a,b[:,inverse],atol=5e-5,rtol=1e-5)
            torch.testing.assert_close(h,j[:,inverse],atol=5e-5,rtol=1e-5)
            with tempfile.TemporaryDirectory() as tmp:
                path=str(Path(tmp)/'model.safetensors');save_file(model.state_dict(),path)
                other=build(self.settings,v,123).eval();other.load_state_dict(load_file(path),strict=True)
                torch.testing.assert_close(a,other.denoise(**features)[0],atol=0,rtol=0)

    def test_full_parameter_budget_and_unchanged_ffn(self):
        settings=read(ROOT/'experiments/arch70m/base.json')
        for v in VARIANTS:
            model=build(settings,v)
            self.assertEqual(sum(p.numel() for p in model.parameters()),COUNTS[v],v)
            for block in model.backbone.transformer.blocks:self.assertEqual(block.base.ff_proj.out_features,3072)
            del model;gc.collect()

    def test_all_five_profiles_and_memory_budget_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);hashes={}
            for v in VARIANTS:
                p=trial(out,v)/'profile.json';write(p,profile(v));hashes[v]=sha(p);valid_profile(read(p),v,'test')
            write(out/'qualification.json',dict(status='passed',source_sha256='test',data_sha256=DATA_SHA,micro_batch=32,profiles=hashes))
            qualified(out,'test')
            bad=profile('C3');bad['candidates'][0]['reserved_bytes']=90
            with self.assertRaises(ValueError):valid_profile(bad,'C3','test')
            write(trial(out,'C3')/'profile.json',bad)
            with self.assertRaises(ValueError):qualified(out,'test')

    def test_completed_receipt_rejects_corruption_and_wrong_token_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);t=trial(out,'C2');write(t/'profile.json',profile('C2'))
            binding=dict(variant='C2',seed=43,epochs=1,total_steps=1276,micro_batch=32,source_sha256='test',data_sha256=DATA_SHA,
                settings=read(ROOT/'experiments/arch70m/base.json'),profile_sha256=sha(t/'profile.json'))
            write(t/'binding.json',binding);write(t/'status.json',dict(status='completed',step=1276,parameters=COUNTS['C2']))
            cp=t/'checkpoints/step_001276';cp.mkdir(parents=True)
            for name in ('model.safetensors','optimizer.pt','rng.pt','scheduler.pt'):(cp/name).write_bytes(b'test fixture')
            write(cp/'binding.json',binding);write(cp/'state.json',dict(step=1276,seen_records=653288,epoch=1,next_batch=0,seen_tokens=214482921))
            def receipt():write(cp/'receipt.json',dict(status='complete',files={p.name:sha(p) for p in cp.iterdir() if p.name!='receipt.json'}))
            receipt();write(t/'metrics/screen_001276.json',dict(step=1276,partition='screen'))
            self.assertTrue(completed(out,'C2','test'))
            state=read(cp/'state.json');state['seen_tokens']-=1;write(cp/'state.json',state);receipt()
            with self.assertRaises(ValueError):completed(out,'C2','test')
            (cp/'model.safetensors').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):completed(out,'C2','test')

    def test_historical_bindings_snapshot_import_and_dry_run(self):
        self.assertEqual(digest(previous_sources()),PREVIOUS_SHA)
        for out in (BASE,BASE/'new',ROOT/'output/arch70m_confirm_4gpu_20261031',ROOT.parent/'dllm_test/output/new'):
            with self.assertRaises(ValueError):prepare(out)
        with tempfile.TemporaryDirectory() as tmp:
            source,signature=snapshot(Path(tmp))
            code="from architecture_attention.runner import trainer;from architecture_attention.common import ROOT,digest;import architecture_attention.models as m;import json;print(json.dumps(dict(root=str(ROOT),signature=digest(trainer().source_binding()),model=m.__file__)))"
            env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES='',USE_TF='0')
            value=json.loads(subprocess.check_output([sys.executable,'-B','-c',code],cwd=source,env=env,text=True).strip().splitlines()[-1])
            self.assertEqual(value['root'],str(source));self.assertEqual(value['signature'],signature)
            self.assertTrue(value['model'].startswith(str(source)))
            pending=Path(tmp)/'dry_run_output'
            subprocess.run([sys.executable,'-B','-m','architecture_attention.runner','--dry-run','--output-root',str(pending)],cwd=source,env=env,check=True,capture_output=True)
            self.assertFalse(pending.exists())


if __name__=='__main__': unittest.main()
