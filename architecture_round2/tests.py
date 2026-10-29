"""CPU regressions: python -B -m architecture_round2.tests from the exploration checkout."""
import copy
import gc
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import save_file, load_file
from dllm.pipelines.immune_llada.data.records import BioSeqRecord, BioSeqChain
from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyCollator, corrupt
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
from experiments.arch70m.models import build as original_build
from experiments.arch70m.train import update, save_checkpoint, restore, make_scheduler
from .common import ROOT, read, digest, base_source_binding, BASE_SOURCE_SHA
from .models import build


class RoundTwoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.settings=read(ROOT/'experiments/arch70m/base.json')
        cls.settings['model'].update(d_model=48,n_heads=4,n_layers=8,mlp_hidden_size=192,
            first_chain_capacity=16,second_chain_capacity=12,full_capacity=30,max_length=128)
        cls.records=[BioSeqRecord([BioSeqChain('LAGVSERT','antibody_heavy'),BioSeqChain('IDPKQN','antibody_light')],
            'antibody','oas_paired',labels={'relation':'binding'}) for _ in range(2)]

    def batch(self,model):
        return VFamilyCollator(model.config,family_spec=model.family_spec)(self.records)

    def test_control_and_initial_global_layerscale_match_A5_exactly(self):
        reference=original_build(self.settings,'A5',43); clean=self.batch(reference)
        noisy=corrupt(clean,reference.tokenizer,torch.Generator().manual_seed(7))
        for variant in ('B0','B1'):
            model=build(self.settings,variant,43)
            for key,value in reference.state_dict().items():self.assertTrue(torch.equal(value,model.state_dict()[key]),key)
            torch.testing.assert_close(reference(**noisy)['logits'],model(**noisy)['logits'],atol=0,rtol=0)
        self.assertEqual(digest(base_source_binding()),BASE_SOURCE_SHA)

    def test_all_variants_have_finite_gradients_and_identical_input_features(self):
        reference=build(self.settings,'B0'); clean=self.batch(reference)
        for variant in ('B1','B2','B3'):
            model=build(self.settings,variant)
            for key,value in clean.items():self.assertTrue(torch.equal(value,self.batch(model)[key]),(variant,key))
            noisy=corrupt(clean,model.tokenizer,torch.Generator().manual_seed(9))
            value=model(**noisy);(value['loss_sum']+.1*value['family_loss_sum']).backward()
            for name,param in model.named_parameters():
                self.assertIsNotNone(param.grad,(variant,name));self.assertTrue(torch.isfinite(param.grad).all(),(variant,name))
            if variant!='B1':
                self.assertEqual(model.backbone.transformer.blocks[0].base.ff_proj.out_features,128)

    def test_hierarchical_masks_have_chain_local_then_global_visibility(self):
        model=build(self.settings,'B2');batch=self.batch(model)
        local,glob=model._build_attention_biases(batch['chain_ids'],batch['position_ids'],batch['attention_mask'])[('hierarchical',)]
        same=batch['chain_ids'][:,:,None].eq(batch['chain_ids'][:,None,:])
        valid=batch['attention_mask'].bool();different=~same&valid[:,:,None]&valid[:,None,:]
        self.assertTrue((local[:,0][different]<-1e20).all())
        self.assertTrue((glob[:,0][different]>-1e20).all())
        for i in range(8):self.assertEqual(model.mask_key(i),('hierarchical',))

    def test_whole_layer_checkpoint_recomputes_tuple_bias_gradients(self):
        plain=build(self.settings,'B3');checkpointed=copy.deepcopy(plain)
        checkpointed.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
        batch=corrupt(self.batch(plain),plain.tokenizer,torch.Generator().manual_seed(10))
        for model in (plain,checkpointed):
            model.train();v=model(**batch);(v['loss_sum']+.1*v['family_loss_sum']).backward()
        for (name,a),(_,b) in zip(plain.named_parameters(),checkpointed.named_parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-5,msg=name)

    def test_protein_blocks_remain_permutation_equivariant(self):
        for variant in ('B1','B2','B3'):
            model=build(self.settings,variant).eval();batch=self.batch(model);features=model.feature_args(batch)
            ids=features['chain_ids'][0]; chunks=[];start=0
            for i in range(1,len(ids)):
                if ids[i]!=ids[i-1]:chunks.append(torch.arange(start,i));start=i
            chunks.append(torch.arange(start,len(ids)));perm=torch.cat(chunks[::-1]);inverse=perm.argsort()
            changed={key:value[:,perm] for key,value in features.items()}
            a,h=model.denoise(**features,output_hidden_states=True)
            b,j=model.denoise(**changed,output_hidden_states=True)
            torch.testing.assert_close(a,b[:,inverse],atol=5e-5,rtol=1e-5)
            torch.testing.assert_close(h,j[:,inverse],atol=5e-5,rtol=1e-5)

    def test_new_parameters_roundtrip_through_safetensors(self):
        for variant in ('B1','B2','B3'):
            model=build(self.settings,variant);batch=self.batch(model)
            with tempfile.TemporaryDirectory() as tmp:
                file=str(Path(tmp)/'weights.safetensors');save_file(model.state_dict(),file)
                other=build(self.settings,variant,123);other.load_state_dict(load_file(file),strict=True)
                torch.testing.assert_close(model(**batch)['logits'],other(**batch)['logits'],atol=0,rtol=0)

    def test_new_architecture_optimizer_and_rng_checkpoint_resume(self):
        a=build(self.settings,'B3');b=build(self.settings,'B3')
        optimizers=[torch.optim.AdamW(m.parameters(),lr=1e-4) for m in (a,b)]
        schedulers=[make_scheduler(o,10,1) for o in optimizers]
        noise=torch.Generator().manual_seed(11)
        batch=corrupt(self.batch(a),a.tokenizer,noise)
        update(a,batch,optimizers[0],schedulers[0],1,torch.device('cpu'))
        with tempfile.TemporaryDirectory() as tmp:
            cp=save_checkpoint(Path(tmp),a,optimizers[0],schedulers[0],noise,dict(step=1),dict(test='round2'))
            update(a,batch,optimizers[0],schedulers[0],1,torch.device('cpu'))
            restore(cp,b,optimizers[1],schedulers[1],noise,dict(test='round2'))
            update(b,batch,optimizers[1],schedulers[1],1,torch.device('cpu'))
            for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)

    def test_full_size_parameter_budget(self):
        settings=read(ROOT/'experiments/arch70m/base.json')
        expected={'B0':75741903,'B1':75754191,'B2':75748047,'B3':75766479}
        for variant,count in expected.items():
            model=build(settings,variant)
            self.assertEqual(sum(p.numel() for p in model.parameters()),count,variant)
            del model;gc.collect()


if __name__=='__main__':unittest.main()
