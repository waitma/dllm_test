"""Adapters used by run_general_teacher_validation.py; activate pllm first.

All adapters preserve native distance-head weights and use complete protein
chains. The Protenix adapter computes ESM embeddings in memory on every call.
"""
from copy import deepcopy
from pathlib import Path
import json
import os
import sys

import numpy as np
import torch

ROOT = Path('/project_c20250601/251105016/project/dllm_test/output/general_teacher_validation_20261024')
WEIGHTS = Path('/project_c20250601/251105016/model_weights/general_teacher_validation_20261024')


def on_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone().to(device)
    if isinstance(obj, dict):
        return {k: on_device(v, device) for k,v in obj.items()}
    if isinstance(obj, list):
        return [on_device(v, device) for v in obj]
    return obj


class ESMFold:
    def __init__(self, device, load_model=True):
        import stage1_teacher as teacher
        self.teacher, self.device = teacher, device
        self.contract = dict(num_loops=3, msa=False, dtype='bfloat16', mode='native_trunk')
        if load_model:
            path = '/project_c20250601/251105016/model_weights/esmfold2_fast/45fe8656f5b3ef493c17fcf9abe9a2968902e712'
            self.model, loaded, self.loading = teacher.load_teacher(path, device)
            self.contract.update(loaded)

    def prepare(self, example):
        return self.teacher.prepare_multichain_features(example['sequences'], 'cpu')

    def predict(self, features, full=False):
        kwargs=self.teacher.trunk_arguments(on_device(features, self.device))
        return self.model(**kwargs, num_loops=3).distogram_logits[0]


class ProtenixMini:
    def __init__(self, device, load_model=True):
        os.environ['PROTENIX_ROOT_DIR'] = '/project_c20250601/251105016/project/Protenix-v2/data'
        os.environ['LAYERNORM_TYPE'] = 'torch'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        sys.path.insert(0, str(ROOT/'sources/protenix'))
        from configs.configs_base import configs as base
        from configs.configs_data import data_configs
        from configs.configs_inference import inference_configs
        from configs.configs_model_type import model_configs
        from protenix.config.config import parse_configs
        from protenix.data.inference.infer_dataloader import InferenceDataset
        from protenix.model.protenix import Protenix
        self.device = device
        raw = deepcopy({**base, 'data': data_configs, **inference_configs})
        def merge(a,b):
            for k,v in b.items():
                if isinstance(v,dict) and isinstance(a.get(k),dict): merge(a[k],v)
                else:a[k]=v
        merge(raw, model_configs['protenix_mini_esm_v0.5.0'])
        args = f'--model_name protenix_mini_esm_v0.5.0 --input_json_path {ROOT}/protenix_inputs.json --dump_dir {ROOT}/protenix_work --load_checkpoint_dir {WEIGHTS} --triangle_attention torch --triangle_multiplicative torch --load_strict true --use_template false --use_msa false --mc_dropout_apply_rate 0 --sample_diffusion.N_sample 1'
        self.config = parse_configs(raw, arg_str=args, fill_required_with_null=True)
        self.processor = InferenceDataset.__new__(InferenceDataset)
        self.processor.configs = self.config
        self.processor.use_msa = self.processor.use_template = self.processor.use_rna_msa = False
        self.processor.msa_pair_as_unpair = True
        self.processor.online_template_featurizer = None
        self.processor.esm_enable = False
        self.contract = dict(revision='85767b811c40ed46e73a9b39519cf6bfca8701ba',
            model='protenix_mini_esm_v0.5.0', cycles=4, msa=False, templates=False,
            folding_dtype='bfloat16_autocast', esm_dtype='float32', kernels='torch',
            mc_dropout=False, deterministic_algorithms=True, coordinates_for_parity=dict(samples=1, steps=5),
            distogram_bins=64, breaks=[2.3125,21.6875,63], esm_cache=False)
        if load_model:
            # The official fair-esm checkpoint includes configuration objects.
            from esm import pretrained
            checkpoint = torch.load(WEIGHTS/'protenix_mini_esm_v0.5.0.pt', map_location='cpu', weights_only=False)
            state = {k.removeprefix('module.'):v for k,v in checkpoint['model'].items()}
            self.model = Protenix(self.config)
            self.loading = str(self.model.load_state_dict(state, strict=True))
            del checkpoint,state
            self.model.eval().requires_grad_(False).to(device)
            # Match upstream's compatibility setting for fair-esm on Torch >=2.6.
            os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
            self.esm, alphabet = pretrained.load_model_and_alphabet_local(str(WEIGHTS/'esm2_t36_3B_UR50D.pt'))
            self.esm.eval().requires_grad_(False).to(device)
            self.converter = alphabet.get_batch_converter()
            self.contract['parameters'] = sum(p.numel() for p in self.model.parameters())
            self.contract['esm_parameters'] = sum(p.numel() for p in self.esm.parameters())

    def prepare(self, example):
        sample = dict(name=example['id'], sequences=[dict(proteinChain=dict(sequence=s,count=1)) for s in example['sequences']])
        # Use the upstream featurizer and verify its token-to-chain mapping.
        data, atoms, _ = self.processor.process_one(sample)
        # The official ESM featurizer assigns embeddings by entity and residue ID.
        centers = atoms[atoms.centre_atom_mask.astype(bool)]
        order = []
        for entity,resid in zip(centers.label_entity_id,centers.res_id):
            order.append((int(entity)-1,int(resid)-1))
        expected = [(c,i) for c,s in enumerate(example['sequences']) for i in range(len(s))]
        if order != expected:
            raise ValueError('Protenix residue order differs from requested chain order')
        return dict(features=data['input_feature_dict'], sequences=example['sequences'], order=order)

    def _input_features(self, prepared):
        f = on_device(prepared['features'], self.device)
        _,_,tokens = self.converter([(str(i),s) for i,s in enumerate(prepared['sequences'])])
        result = self.esm(tokens.to(self.device), repr_layers=[36], return_contacts=False)['representations'][36]
        f['esm_token_embedding'] = torch.cat([result[i,1:len(s)+1] for i,s in enumerate(prepared['sequences'])]).float()
        return f

    def predict_with_confidence(self, prepared, confidence_threshold=4.0):
        """One native five-step sample; PDE is coordinate-conditioned, not free.

        Return native distance logits and the predicted probability of pair
        distance error <= 4 A. This confidence is not calibrated for distogram
        correctness and must be evaluated before promoting it to a training gate.
        """
        f = self._input_features(prepared)
        representative = f['distogram_rep_atom_mask'].bool().clone()
        captured = []
        handle = self.model.distogram_head.register_forward_hook(
            lambda module,args,output:captured.append(output.detach().clone()))
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction,_,_ = self.model(input_feature_dict=f, label_full_dict=None,
                    label_dict=None, mode='inference', mc_dropout_apply_rate=0.0)
        finally:
            handle.remove()
        if len(captured)!=1 or prediction['pde'].shape[0]!=1:
            raise ValueError('Expected one native head evaluation and one coordinate sample')
        logits = captured[0].float()
        pde = prediction['pde'][0].float()
        expected = self.config.loss.pde
        if (expected.min_bin,expected.max_bin,expected.no_bins)!=(0,32,64):
            raise ValueError('PDE bin definition changed')
        if confidence_threshold not in (1.,2.,4.):
            raise ValueError('Confidence threshold must be a declared native PDE bin edge')
        # Native PDELoss assigns 0..4 A to the first eight half-Angstrom bins.
        cdf = pde.softmax(-1).cumsum(-1).clamp(0,1)
        confidence = cdf[...,int(confidence_threshold*2)-1]
        return dict(logits=logits,confidence=confidence,
            confidence_cdf=torch.stack([cdf[...,int(t*2)-1] for t in (1.,2.,4.)],dim=-1),
            coordinates=prediction['coordinate'][0,representative].float(),
            confidence_kind=f'native_PDE_probability_distance_error_le_{confidence_threshold:g}A',
            confidence_calibrated_for_distogram=False)

    def predict(self, prepared, full=False):
        from protenix.model.protenix import update_input_feature_dict
        f = self._input_features(prepared)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if full:
                captured = []
                handle = self.model.distogram_head.register_forward_hook(lambda m,a,o:captured.append(o.detach().clone()))
                try:
                    self.model(input_feature_dict=f, label_full_dict=None, label_dict=None,
                               mode='inference', mc_dropout_apply_rate=0.0)
                finally:
                    handle.remove()
                if len(captured)!=1:raise ValueError('Expected one native distance-head call')
                return captured[0]
            f = self.model.relative_position_encoding.generate_relp(f)
            f = update_input_feature_dict(f)
            _,_,z = self.model.get_pairformer_output(f, N_cycle=4, inplace_safe=True,
                        chunk_size=self.model._get_dynamic_chunk_size(len(prepared['order'])), mc_dropout=False)
            return self.model.distogram_head(z)


class BoltzTeacher:
    def __init__(self, device, load_model=True):
        sys.path.insert(0, str(ROOT/'sources/boltz/src'))
        from boltz.model.models.boltz2 import Boltz2
        self.device = device
        self.contract = dict(revision='b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc',
            model='boltz2_conf', recycling_steps=3, actual_trunk_passes=4, msa='empty/query-only',
            templates=False, dtype='bfloat16_autocast', kernels=False,
            coordinates_for_parity=dict(samples=1, steps=200), distogram_bins=64,
            confidence_for_trunk=False, affinity=False)
        if load_model:
            from dataclasses import asdict
            from boltz.main import Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs, BoltzSteeringParams
            self.model = Boltz2.load_from_checkpoint(str(WEIGHTS/'boltz2_conf.ckpt'),
                map_location='cpu', strict=True, use_kernels=False, ema=False,
                diffusion_process_args=asdict(Boltz2DiffusionParams()),
                pairformer_args=asdict(PairformerArgsV2()),
                msa_args=asdict(MSAModuleArgs()),steering_args=asdict(BoltzSteeringParams()),
                predict_args=dict(recycling_steps=3,sampling_steps=200,diffusion_samples=1,
                    max_parallel_samples=1,write_confidence_summary=True,write_full_pae=False,write_full_pde=False))
            self.loading = 'strict=True load_from_checkpoint completed'
            self.model.eval().requires_grad_(False).to(device)
            self.contract['parameters'] = sum(p.numel() for p in self.model.parameters())

    def prepare(self, example):
        import yaml
        from boltz.main import process_inputs
        from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
        from boltz.data.types import Manifest
        work = ROOT/'boltz_features'/example['id']
        work.mkdir(parents=True,exist_ok=True)
        path=work/'input.yaml'
        path.write_text(yaml.safe_dump(dict(version=1,sequences=[dict(protein=dict(id=chr(65+i),sequence=s,msa='empty')) for i,s in enumerate(example['sequences'])])))
        process_inputs([path], work, WEIGHTS/'unused_ccd.pkl', WEIGHTS/'mols',
                    msa_server_url='', msa_pairing_strategy='greedy', use_msa_server=False,
                    boltz2=True, preprocessing_threads=1)
        manifest=Manifest.load(work/'processed/manifest.json')
        if len(manifest.records)!=1:raise ValueError('Boltz preprocessing did not yield exactly one record')
        data=Boltz2InferenceDataModule(manifest,work/'processed/structures',work/'processed/msa',
            WEIGHTS/'mols',num_workers=0,constraints_dir=work/'processed/constraints',
            template_dir=work/'processed/templates',extra_mols_dir=work/'processed/mols')
        f=next(iter(data.predict_dataloader()))
        lengths=example['chain_lengths']
        valid=f['token_pad_mask'][0].bool()
        chain=f['asym_id'][0][valid].tolist()
        residues=f['residue_index'][0][valid].tolist()
        # Native parsing groups copies of identical entities (e.g. A,C,B,D).
        # Restore the declared A,B,C,D order before comparing with coordinates.
        names={c.chain_id:c.chain_name for c in manifest.records[0].chains}
        lookup={(names[c],r):i for i,(c,r) in enumerate(zip(chain,residues))}
        expected=[(chr(65+c),i) for c,L in enumerate(lengths) for i in range(L)]
        if set(lookup)!=set(expected) or len(lookup)!=int(valid.sum()):
            raise ValueError('Boltz token/residue mapping is not bijective')
        order=[lookup[x] for x in expected]
        from Bio.Data.PDBData import protein_letters_3to1
        from boltz.data import const
        observed=''.join(protein_letters_3to1[const.tokens[int(i)]] for i in f['res_type'][0][valid].argmax(-1)[order])
        if observed!=''.join(example['sequences']):
            raise ValueError('Boltz residue identity check failed')
        return dict(features=f,order=order)

    def predict(self, features, full=False):
        self.model.skip_run_structure = not full
        self.model.confidence_prediction = full
        self.model.affinity_prediction = False
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result=self.model(on_device(features['features'],self.device),recycling_steps=3,
                              num_sampling_steps=200,diffusion_samples=1)
        x=result['pdistogram']
        if x.ndim!=5 or x.shape[0]!=1 or x.shape[-2]!=1:
            raise ValueError('Unexpected native Boltz distance-head shape '+str(x.shape))
        valid=features['features']['token_pad_mask'][0].bool().to(x.device)
        order=torch.tensor(features['order'],device=x.device)
        x=x[0,:,:,0,:][valid][:,valid]
        return x.index_select(0,order).index_select(1,order)


ADAPTERS = {'esmfold2':ESMFold, 'protenix':ProtenixMini, 'boltz2':BoltzTeacher}
