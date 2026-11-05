"""Meaningful V11 regressions: USE_TF=0 python -B -m unittest v11_a5.tests -v."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
import torch
from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyCollator,VFamilyModel,corrupt
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
from experiments.arch70m.models import build
from v11_a5.model import V11A5Model,install_trainer_adapter
from v11_a5.release import CONFIG,read,validate_config,rebind_data,checked_plan,write,sha


class V11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.release=read(CONFIG);cls.settings=deepcopy(cls.release)
        cls.settings['model'].update(d_model=48,n_layers=8,n_heads=4,mlp_hidden_size=192,
            first_chain_capacity=16,second_chain_capacity=12,full_capacity=30,max_length=128)
        cls.record={'chains':['LAGVSERT','IDPKQN'],'chain_roles':['antibody_heavy','antibody_light'],
            'task_type':'antibody','source':'oas_paired','split':'train','labels':{'relation':'binding'},
            'metadata':{'species':'human'},'chain_metadata':{'0':{'h_v_call':'IGHV1-2*01'}}}

    def model(self):
        torch.manual_seed(42)
        return V11A5Model(FixedChainConfig(**self.settings['model']),family_spec=self.settings['family_prediction'])

    def test_same_a5_weights_logits_inputs_and_gradients(self):
        a=build(self.settings,'A5',42);b=self.model()
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        for name,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.state_dict()[name]),name)
        collate=VFamilyCollator(b.config,family_spec=b.family_spec)
        clean=collate([self.record]);batch=corrupt(clean,b.tokenizer,torch.Generator().manual_seed(9))
        ao,bo=a(**batch),b(**batch)
        torch.testing.assert_close(ao['logits'],bo['logits'],atol=0,rtol=0)
        for m,o in [(a,ao),(b,bo)]:
            loss=o['loss_sum']+.1*o['family_loss_sum'];loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        for (n,p),(nn,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(n,nn);torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)

    def test_all_layers_can_read_other_chains_and_hide_padded_keys(self):
        m=self.model();ids=torch.tensor([[0,0,1,1,-1]]);positions=torch.tensor([[1,2,1,2,0]])
        attention=torch.tensor([[True,True,True,True,False]])
        for layer in range(8):
            self.assertEqual(m.mask_key(layer),('global',))
            allowed=m.layer_allowed(layer,ids,positions,attention)
            self.assertTrue(allowed[0,0,2]);self.assertTrue(allowed[0,2,0]);self.assertFalse(allowed[:,:,4].any())
        for block in m.backbone.transformer.blocks:
            self.assertTrue(torch.equal(block.attention_scale,torch.full((48,),.1)))
            self.assertTrue(torch.equal(block.ffn_scale,torch.full((48,),.1)))

    def test_checkpointing_preserves_backward(self):
        a=self.model();b=deepcopy(a);b.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
        collate=VFamilyCollator(a.config,family_spec=a.family_spec)
        batch=corrupt(collate([self.record]),a.tokenizer,torch.Generator().manual_seed(22))
        for m in [a,b]:
            o=m(**batch);(o['loss_sum']+.1*o['family_loss_sum']).backward()
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)

    def test_v11_checkpoint_cannot_silently_load_as_a0(self):
        m=self.model().eval();collate=VFamilyCollator(m.config,family_spec=m.family_spec)
        batch=collate([self.record])
        with tempfile.TemporaryDirectory() as tmp:
            m.save_pretrained(tmp);restored=V11A5Model.from_pretrained(tmp).eval()
            torch.testing.assert_close(m(**batch)['logits'],restored(**batch)['logits'],atol=0,rtol=0)
            with self.assertRaises(ValueError):VFamilyModel.from_pretrained(tmp)
            p=Path(tmp)/'research_config.json';v=read(p);v['options']['attention_schedule']='alternating';write(p,v)
            with self.assertRaises(ValueError):V11A5Model.from_pretrained(tmp)

    def test_training_and_data_exactly_match_1b_except_model_binding(self):
        parent=read(self.release['reference_1b_plan']);validate_config(self.release,parent)
        bad=deepcopy(self.release);bad['training']['max_steps']=1276
        with self.assertRaises(ValueError):validate_config(bad,parent)
        old=read(parent['data_index'])
        with tempfile.TemporaryDirectory() as tmp:
            new=rebind_data(parent,self.release['model'],Path(tmp),Path(tmp))
            self.assertEqual(new['shards'],old['shards']);self.assertEqual(new['files'],old['files'])
            self.assertEqual(new['records'],{'train':6533037,'valid':71422})
            self.assertEqual(read(Path(tmp)/'binding.json')['model'],self.release['model'])
            self.assertNotEqual(new['binding_sha256'],old['binding_sha256'])
        self.assertEqual(self.release['training']['micro_batch'],16)
        self.assertEqual(self.release['training']['grad_accum'],4)
        self.assertEqual(self.release['training']['max_steps'],300000)

    def test_full_size_count_and_trainer_adapter(self):
        torch.manual_seed(42)
        model=V11A5Model(FixedChainConfig(**self.release['model']),family_spec=self.release['family_prediction'])
        self.assertEqual(sum(p.numel() for p in model.parameters()),75741903)
        from scripts.research import run_bert_mlm as original
        old=original.components
        try:
            t=install_trainer_adapter();collator,factory,noise,recipe=t.components(self.release['objective'],self.release['family_prediction'])
            self.assertIsInstance(factory(FixedChainConfig(**self.settings['model'])),V11A5Model)
            self.assertEqual(recipe,self.release['recipe'])
        finally:original.components=old

    def test_frozen_plan_detects_dependency_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp);(b/'source').mkdir();dep=b/'source/dependency.py';dep.write_text('x=1\n')
            index=b/'index.json';write(index,{})
            plan=dict(schema='immune.v11.a5.training.v1',architecture='a5_all_global',version='V11',source_root=str(b/'source'),
                training=self.release['training'],reference_1b={'training':self.release['training']},expected_parameters=75741903,
                execution={'world_size':8},sources={str(dep):sha(dep)},assets={},data_index=str(index),data_index_sha256=sha(index))
            write(b/'plan.json',plan);write(b/'release.json',{'plan_sha256':sha(b/'plan.json')})
            checked_plan(b);dep.write_text('x=2\n')
            with self.assertRaises(ValueError):checked_plan(b)


if __name__=='__main__':unittest.main()
