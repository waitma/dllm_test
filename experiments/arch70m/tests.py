"""Meaningful CPU regressions: python -B -m experiments.arch70m.tests."""
import copy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import torch
from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyModel, VFamilyCollator, corrupt
from scripts.research.run_bert_mlm import make_scheduler
from .common import ROOT, read
from .models import build
from .train import update, save_checkpoint, restore
from .data import identity, partition
from .evaluate import make_tasks, task_batch, generation

class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings=read(ROOT/'experiments/arch70m/base.json')
        cls.settings['model'].update(d_model=64,n_heads=4,n_layers=8,mlp_hidden_size=256,
                                     first_chain_capacity=16,second_chain_capacity=12,full_capacity=30,max_length=128)
        cls.records=[BioSeqRecord([BioSeqChain('LAGVSERT','antibody_heavy'),BioSeqChain('IDPKQN','antibody_light')],
                       'antibody','oas_paired',labels={'relation':'binding'}) for _ in range(2)]

    def batch(self,model):
        return VFamilyCollator(model.config,family_spec=model.family_spec)(self.records)

    def test_original_forward_and_gradients(self):
        torch.manual_seed(42)
        original=VFamilyModel(FixedChainConfig(**self.settings['model']),family_spec=self.settings['family_prediction'])
        model=build(self.settings,'A0',42)
        for n,p in original.state_dict().items(): self.assertTrue(torch.equal(p,model.state_dict()[n]),n)
        batch=corrupt(self.batch(model),model.tokenizer,torch.Generator().manual_seed(1))
        for m in (original,model):
            value=m(**batch); (value['loss_sum']+.1*value['family_loss_sum']).backward()
        self.assertTrue(torch.equal(original(**batch)['logits'],model(**batch)['logits']))
        for (n,p),(_,q) in zip(original.named_parameters(),model.named_parameters()):
            if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0,msg=n)

    def test_variants_shapes_gradients_and_initialization(self):
        base=build(self.settings,'A0'); parent=base.state_dict(); counts={}
        for variant in ('A0','A1','A2','A3','A4','A5','A6','A7'):
            model=build(self.settings,variant); counts[variant]=sum(p.numel() for p in model.parameters())
            for n,p in parent.items():
                q=model.state_dict()[n]
                if q.shape==p.shape: self.assertTrue(torch.equal(p,q),(variant,n))
            batch=corrupt(self.batch(model),model.tokenizer,torch.Generator().manual_seed(9))
            values=model(**batch); (values['loss_sum']+.1*values['family_loss_sum']).backward()
            for n,p in model.named_parameters():
                self.assertIsNotNone(p.grad,(variant,n)); self.assertTrue(torch.isfinite(p.grad).all(),(variant,n))
        self.assertEqual(counts['A0'],counts['A6'])
        self.assertEqual(counts['A0'],counts['A7'])

    def test_accumulation_and_resume(self):
        a=build(self.settings); b=build(self.settings)
        opts=[torch.optim.AdamW(m.parameters(),lr=1e-4) for m in (a,b)]
        scheds=[make_scheduler(o,10,1) for o in opts]
        batch=corrupt(self.batch(a),a.tokenizer,torch.Generator().manual_seed(9))
        for m,o,s,micro in zip((a,b),opts,scheds,(2,1)): update(m,batch,o,s,micro,torch.device('cpu'))
        # The first scheduler step has LR=0; check gradients and Adam moments too.
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=1e-7,rtol=1e-5)
            torch.testing.assert_close(opts[0].state[p]['exp_avg'],opts[1].state[q]['exp_avg'],atol=1e-8,rtol=1e-5)
        for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q,atol=1e-7,rtol=1e-6)
        noise=torch.Generator().manual_seed(9)
        with tempfile.TemporaryDirectory() as tmp:
            path=save_checkpoint(Path(tmp),a,opts[0],scheds[0],noise,dict(step=1),dict(test=True))
            update(a,batch,opts[0],scheds[0],2,torch.device('cpu'))
            restore(path,b,opts[1],scheds[1],noise,dict(test=True))
            update(b,batch,opts[1],scheds[1],2,torch.device('cpu'))
            for p,q in zip(a.parameters(),b.parameters()): torch.testing.assert_close(p,q,atol=0,rtol=0)

    def test_relation_label_blindness_and_topology(self):
        model=build(self.settings); collate=VFamilyCollator(model.config,family_spec=model.family_spec,hide_relation=True)
        other=copy.deepcopy(self.records); other[0].labels['relation']='nonbinding'
        a=collate(self.records); b=collate(other)
        for n,v in model.feature_args(a).items(): self.assertTrue(torch.equal(v,model.feature_args(b)[n]),n)
        segment=model.layer_allowed(0,a['chain_ids'],a['position_ids'],a['attention_mask'])
        glob=model.layer_allowed(1,a['chain_ids'],a['position_ids'],a['attention_mask'])
        self.assertTrue((segment<=glob).all()); self.assertTrue((segment!=glob).any())
        alt=build(self.settings,'A5'); self.assertEqual(alt.mask_key(0),('global',))
        self.assertTrue(torch.equal(alt.layer_allowed(0,a['chain_ids'],a['position_ids'],a['attention_mask']),glob))

    def test_constant_gate_control_and_serialization(self):
        from safetensors.torch import save_file,load_file
        gate=build(self.settings,'A2'); control=build(self.settings,'A7')
        for block in gate.backbone.transformer.blocks:
            torch.nn.init.zeros_(block.gate.weight); torch.nn.init.zeros_(block.gate.bias)
        batch=self.batch(gate)
        torch.testing.assert_close(gate(**batch)['logits'],control(**batch)['logits'],atol=0,rtol=0)
        for v in ('A1','A2','A3','A4','A6'):
            model=build(self.settings,v)
            with tempfile.TemporaryDirectory() as tmp:
                path=str(Path(tmp)/'model.safetensors'); save_file(model.state_dict(),path)
                restored=build(self.settings,v,123); restored.load_state_dict(load_file(path))
                torch.testing.assert_close(model(**batch)['logits'],restored(**batch)['logits'],atol=0,rtol=0)

    def test_machine_partition_and_cluster_pairing(self):
        from .suite import assigned_trials
        from .statistics import align,statistic
        trials=list(range(9)); a=assigned_trials(trials,0,2); b=assigned_trials(trials,1,2)
        self.assertFalse(set(a)&set(b)); self.assertEqual(set(a+b),set(trials))
        base=[dict(source='s',key=str(i//2),kind='cdr',role='h',ce=2.) for i in range(4)]
        candidate=[dict(x,ce=1.) for x in base]; groups=align(base,candidate,'generation')
        self.assertEqual(statistic(base,candidate,groups,'generation'),.5)
        with self.assertRaises(ValueError): align(base,candidate[::-1],'generation')

    def test_max_length_profile_corruption(self):
        from .train import profile_batch
        model=build(self.settings); batch=profile_batch(self.batch(model),3,128,model.tokenizer)
        noisy=corrupt(batch,model.tokenizer,torch.Generator().manual_seed(9100))
        self.assertEqual(tuple(noisy['input_ids'].shape),(3,128))
        self.assertTrue(torch.isfinite(model(**noisy)['loss_sum']))

    def test_generation_target_length_hidden(self):
        from dllm.pipelines.immune_llada.fixed_chain.layout import FixedChainCollator
        collate=FixedChainCollator(FixedChainConfig(**self.settings['model']),mode='generation',target_roles=['antibody_heavy'])
        other=[replace(self.records[0], chains=[replace(self.records[0].chains[0], sequence='LAGV'), self.records[0].chains[1]])]
        a=collate([self.records[0]]); b=collate([other[0]])
        model=build(self.settings)
        for n,v in model.feature_args(a).items(): self.assertTrue(torch.equal(v,model.feature_args(b)[n]),n)

if __name__=='__main__': unittest.main()
