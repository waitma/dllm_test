"""CPU compatibility checks: activate pllm, then python -B SCRIPT --checkpoint ABS."""
import argparse,json,sys
from pathlib import Path
ROOT=Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test');sys.path.insert(0,str(ROOT))
from downstream.v5_native_eval import load,NativeBackbone,cdr_input

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    model,identity=load(a.checkpoint,'cpu');backbone=NativeBackbone(model,identity)
    import numpy as np
    from downstream.benchmark.tcr_binding.query_protocol import TCRBindingQueryProtocol,TCRBindingEmbedder
    from downstream.grammar.tcr_features import TCRRuntimeEmbedder
    reports={}
    for mode in ('cdr3b','cdr3ab','longab'):
        protocol=TCRBindingQueryProtocol(input_mode=mode)
        b='CASSLGQETQYF';alpha=None if mode=='cdr3b' else 'CAVRDSNYQLIW'
        if mode=='longab':
            rec=TCRBindingQueryProtocol(input_mode='cdr3ab').record('GILGFVFTL',b,alpha)
            b,alpha=[c.sequence for c in rec.chains[-2:]]
        emb=TCRBindingEmbedder(backbone,protocol)
        vectors=emb.embed_pairs([b],alpha=[alpha] if alpha else None,peptide=['GILGFVFTL'])
        assert vectors.shape==(1,768) and np.isfinite(vectors).all();reports[mode]='passed'
    e=TCRRuntimeEmbedder(backbone=backbone,batch_size=8)
    x=e.embed_pairs(['CASSLGQETQYF','CASSIRSSYEQYF'],['CAVRDSNYQLIW','CAVRDSNYQLIW'])
    e=TCRRuntimeEmbedder(backbone=backbone,batch_size=1)
    y=e.embed_pairs(['CASSLGQETQYF','CASSIRSSYEQYF'],['CAVRDSNYQLIW','CAVRDSNYQLIW'])
    assert np.allclose(x,y,rtol=1e-4,atol=1e-5)
    h='EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRGYYFDYWGQGTLVTVSS'
    l='DIQMTQSPSSLSASVGDRVTITCRASQSIYSSLAWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPYTFGQGTKVEIK'
    f,m=cdr_input(model,h,l,'antibody_heavy',25,32)
    g,n=cdr_input(model,h[:25]+'G'*7+h[32:],l,'antibody_heavy',25,32)
    assert all(f[k].equal(g[k]) for k in f) and m.equal(n) and int(m.sum())==7
    from downstream.v7_native import cdr_argmax
    tokens,audit=cdr_argmax(model,f,m);assert audit['labels_provided'] is False
    from dllm.pipelines.immune_llada.v3.layout import GenerationRequest
    from dllm.pipelines.immune_llada.data.records import BioSeqChain
    from downstream.v5_layout import inputs
    for conditions,targets in [([BioSeqChain(h,'antibody_heavy')],[('antibody_light','full')]),([BioSeqChain('GILGFVFTL','peptide')],[('tcr_beta','full')]),([BioSeqChain('ACDEFGHIK','antigen')],[('antibody_heavy','full'),('antibody_light','full')])]:
        f=inputs(model,[GenerationRequest(tuple(conditions),tuple(targets))],mode='generation')
        slots=f['target_slot_mask'];assert int(slots.sum())==302
        assert not ((f['input_ids']==model.tokenizer.special_id('<protd>')) & slots).any()
    reports.update(tcr_batch_invariance='passed',cdr_hidden_reference_invariance='passed',generation_no_reference_terminal='passed')
    Path(a.output).write_text(json.dumps({'status':'passed','checks':reports,'model':identity},indent=2)+'\n');print(reports)
if __name__=='__main__':main()
