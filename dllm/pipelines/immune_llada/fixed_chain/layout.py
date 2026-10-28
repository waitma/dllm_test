"""Keep V5 outer grammar but reserve separate chain regions; test with test_fixed_chain.py."""
from dataclasses import replace
import torch
from ..data.records import BioSeqChain, BioSeqRecord, CHAIN_ROLE_TO_ID
from ..v3.layout import ImmuneCollator, GenerationRequest, ordered_chains
from .model import FixedChainConfig

ROLES=('antibody_heavy','antibody_light','tcr_beta','tcr_alpha')
FIRST={'antibody_heavy','tcr_beta'}
PAIRS={('antibody_heavy',),('antibody_light',),('antibody_heavy','antibody_light'),
       ('tcr_beta',),('tcr_alpha',),('tcr_beta','tcr_alpha')}

def capacity(config, role):
    return config.first_chain_capacity if role in FIRST else config.second_chain_capacity

class FixedChainCollator:
    def __init__(self, config=None, *, mode='train', target_roles=None):
        self.config=config or FixedChainConfig()
        if not isinstance(self.config,FixedChainConfig): raise TypeError('FixedChainConfig required')
        if mode not in ('train','generation','representation'): raise ValueError(mode)
        self.mode=mode
        self.target_roles=set(ROLES if target_roles is None else target_roles)
        if not self.target_roles <= set(ROLES): raise ValueError('Unsupported receptor role')
        self.base=ImmuneCollator('V5', max_length=self.config.max_length,
                                full_capacity=self.config.full_capacity,mode='representation')
        self.tokenizer=self.base.tokenizer

    @classmethod
    def from_config(cls, config, **kwargs): return cls(config, **kwargs)

    def row(self, item):
        tok=self.tokenizer
        if isinstance(item,dict):item=BioSeqRecord.from_prepared_dict(item)
        if isinstance(item,GenerationRequest):
            if self.mode!='generation':raise ValueError('GenerationRequest needs generation mode')
            targets={r for r,_ in item.targets}
            chains=list(item.conditions)+[BioSeqChain('X',r) for r,_ in item.targets]
            is_tcr=any(c.role.startswith('tcr_') for c in chains)
            task='tcr' if is_tcr else 'antibody_antigen' if any(c.role=='antigen' for c in chains) else 'antibody'
            record=BioSeqRecord(chains,task,'fixed_chain_generation')
        else:
            record=item;targets=self.target_roles & set(record.chain_roles)
        receptors=[c for c in ordered_chains(record) if c.role in ROLES]
        if tuple(c.role for c in receptors) not in PAIRS or 'nanobody_vhh' in record.chain_roles:
            raise ValueError('Unsupported receptor signature; VHH is excluded')
        if self.mode=='generation' and not targets:raise ValueError('No generation target')
        # Do not render any hidden target length/metadata, even for validation.
        safe=replace(record,chains=[BioSeqChain('X',c.role) if c.role in ROLES else c for c in record.chains])
        rendered=self.base._legacy([safe])
        start=int((rendered['input_ids'][0]==tok.special_id('<prots>')).nonzero()[-1])
        row={k:v[0,:start].long().tolist() for k,v in rendered.items()}
        uid=max(row['chain_ids'],default=-1)+1
        def add(token, *, chain=-1,pos=0,role=0,slot=False,label=-100,residue=False,terminal=False):
            values=dict(input_ids=token,labels=label,attention_mask=1,role_ids=role,scope_ids=0,
                        chain_ids=chain,position_ids=pos,target_slot_mask=slot,eligible_mask=label!=-100,
                        residue_mask=residue,terminal_mask=terminal)
            for k in row:row[k].append(int(values[k]))
        add(tok.special_id('<prots>'))
        add(tok.special_id('<tcr>' if receptors[0].role.startswith('tcr_') else '<ab>'))
        for i,c in enumerate(receptors):
            if i:add(tok.chain_separator_id())  # structural condition, never predicted
            cap=capacity(self.config,c.role);hidden=self.mode=='generation' and c.role in targets
            slot=self.mode!='representation' and c.role in targets
            if hidden:
                encoded=[];synthetic=[]
            else:
                encoded=tok.encode_residues(c.sequence)
                synthetic=c.metadata.get('synthetic_residue_mask',[0]*len(encoded))
                if len(synthetic)!=len(encoded) or any(x not in (0,1,False,True) for x in synthetic):
                    raise ValueError('Invalid synthetic provenance')
                if len(encoded)+1>cap:raise ValueError('chain_capacity_overflow:'+c.role)
            observed=any(not flag for flag in synthetic)
            for j in range(cap):
                token=tok.mask_token_id;label=-100;residue=False;terminal=False
                if not hidden and j<len(encoded):
                    if not synthetic[j]:
                        token=encoded[j];residue=True
                        if self.mode=='train' and slot:label=token
                elif not hidden and j==len(encoded) and observed:
                    token=tok.eos_token_id;terminal=True
                    if self.mode=='train' and slot:label=token
                add(token,chain=uid+i,pos=j+1,role=CHAIN_ROLE_TO_ID[c.role],slot=slot,
                    label=label,residue=residue,terminal=terminal)
        add(tok.special_id('<protd>'))
        if len(row['input_ids'])>self.config.max_length:raise ValueError('sequence_budget_overflow')
        return row

    def __call__(self, records):
        rows=[self.row(r) for r in records]
        if not rows:raise ValueError('Empty batch')
        size=max(len(r['input_ids']) for r in rows)
        fill={'input_ids':self.tokenizer.pad_token_id,'labels':-100,'chain_ids':-1}
        result={k:torch.tensor([r[k]+[fill.get(k,0)]*(size-len(r[k])) for r in rows],dtype=torch.long) for k in rows[0]}
        for k in ('attention_mask','target_slot_mask','eligible_mask','residue_mask','terminal_mask'):result[k]=result[k].bool()
        return result
