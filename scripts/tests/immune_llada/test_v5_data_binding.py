"""Run python -B -m pytest scripts/tests/immune_llada/test_v5_data_binding.py -q."""
from pathlib import Path
import json
import os
import tempfile
import pytest

from scripts.research.v5_data_binding import verify_binding,sha
from scripts.research.run_v7_formal import parse_args,parent_gate
from dllm.pipelines.immune_llada.v3.v5_contract import geometry
from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator
from dllm.pipelines.immune_llada.data.records import BioSeqRecord,BioSeqChain


@pytest.mark.parametrize('task,chains',[
    ('antibody',[('AC','antibody_heavy'),('DE','antibody_light')]),
    ('antibody_antigen',[('FG','antigen'),('AC','antibody_heavy'),('DE','antibody_light')]),
    ('antibody',[('AC','antibody_heavy')]),
    ('tcr',[('AC','tcr_beta'),('DE','tcr_alpha')]),
    ('tcr_epitope',[('AC','peptide'),('DE','tcr_beta')]),
    ('tcr_pmhc',[('AC','mhc'),('DE','mhc'),('FG','peptide'),('HI','tcr_beta'),('KL','tcr_alpha')]),
    ('tcr_pmhc',[('AC','mhc'),('DE','tcr_beta')])])
def test_arithmetic_contract_matches_real_renderer(task,chains):
    row=BioSeqRecord([BioSeqChain(*c) for c in chains],task,'test').to_dict()
    g=geometry(row);batch=ImmuneCollator('V5',full_capacity=302)([row])
    assert not g['reasons']
    assert int(batch['attention_mask'].sum())==g['total_length']
    assert int(batch['eligible_mask'][batch['target_slot_mask']].sum())==g['object_length']


def test_contract_rejects_scope_overflow_and_bad_provenance():
    def record(seq,role='tcr_beta',**meta):
        return BioSeqRecord([BioSeqChain(seq,role,metadata=meta)],'tcr','test').to_dict()
    assert geometry(record('A'*302))['reasons']==['object_capacity_overflow']
    assert geometry(record('AA',synthetic_residue_mask=[1]))['reasons']==['invalid_synthetic_provenance']
    assert geometry(record('AA',role='nanobody_vhh'))['excluded']=='vhh'
    missing=geometry(record('XXX',synthetic_residue_mask=[1,1,1]))
    assert missing['missing_chains']==1 and missing['observed_residues']==0


@pytest.fixture
def binding_case():
    with tempfile.TemporaryDirectory(prefix='v5-binding-test-') as directory:
        root=Path(directory);data=root/'data';data.mkdir()
        shard=data/'train.jsonl';shard.write_text('{"sample":1}\n');stat=shard.stat()
        manifest={'schema_version':'immune_llada.semantic.v1','format':'jsonl','splits':{
            'train':{'records':1,'shards':[{'path':'train.jsonl','records':1,'source':'test'}]},
            'valid':{'records':0,'shards':[]}}}
        (data/'dataset_manifest.json').write_text(json.dumps(manifest))
        audit=dict(schema='immune.v5.full_layout_audit.v1',status='passed',data_dir=str(data),capacity=302,max_length=2048,
            counts={'valid_records':1,'records':1},reasons={},splits={'train':1,'valid':0},
            manifest_sha256=sha(data/'dataset_manifest.json'),shards=[dict(relative_path='train.jsonl',path=str(shard),
                split='train',source='test',counts={'records':1},sha256=sha(shard),stat={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns})])
        audit_path=root/'audit.json';audit_path.write_text(json.dumps(audit))
        repo=Path(__file__).resolve().parents[3]
        model=root/'research_config.json';model.write_bytes((repo/'configs/models/v7_v5_layout_chain_global_20261007/research_config.json').read_bytes())
        binding=dict(schema='immune.v5.launch_data_binding.v1',status='ready',data_dir=str(data),capacity=302,max_length=2048,
            input_layout='V5',variant='cv_segment_global',manifest_sha256=audit['manifest_sha256'],
            audit={'path':str(audit_path),'sha256':sha(audit_path)},model_config={'path':str(model),'sha256':sha(model)})
        path=root/'binding.json';path.write_text(json.dumps(binding))
        yield path,data,shard,model
    assert not root.exists()


def test_binding_passes_and_rejects_tampering(binding_case):
    path,data,shard,model=binding_case;expected=sha(path)
    assert verify_binding(path,expected,data,302,'cv_segment_global')['content_hashes_verified']
    with pytest.raises(ValueError,match='capacity'):
        verify_binding(path,expected,data,512,'cv_segment_global')
    with pytest.raises(ValueError,match='SHA'):
        verify_binding(path,'0'*64,data,302,'cv_segment_global')
    before=shard.stat();shard.write_text('{"sample":2}\n');os.utime(shard,ns=(before.st_atime_ns,before.st_mtime_ns))
    # Some mounted filesystems round restored mtimes. Match the fresh stat in
    # the report while keeping the ORIGINAL content hash: the byte check must
    # still reject this same-sized edit independently of the stat check.
    binding=json.loads(path.read_text());audit_path=Path(binding['audit']['path'])
    audit=json.loads(audit_path.read_text());audit['shards'][0]['stat']['mtime_ns']=shard.stat().st_mtime_ns
    audit_path.write_text(json.dumps(audit));binding['audit']['sha256']=sha(audit_path)
    path.write_text(json.dumps(binding));expected=sha(path)
    with pytest.raises(ValueError,match='content changed'):
        verify_binding(path,expected,data,302,'cv_segment_global')


def test_binding_rejects_failed_audit_even_if_rehashed(binding_case):
    path,data,_,_=binding_case;b=json.loads(path.read_text());a=Path(b['audit']['path'])
    report=json.loads(a.read_text());report['reasons']={'object_capacity_overflow':1};a.write_text(json.dumps(report))
    b['audit']['sha256']=sha(a);path.write_text(json.dumps(b))
    with pytest.raises(ValueError,match='Unresolved'):
        verify_binding(path,sha(path),data,302,'cv_segment_global')


def test_driver_v5_binding_required_and_a4_gate_unchanged(binding_case):
    path,data,_,_=binding_case;root=path.parent
    argv=[]
    for name in ('source-root','repo-root','output-dir'):
        argv+=['--'+name,str(root)]
    argv+=['--data-dir',str(data),'--source-manifest',str(path),'--source-manifest-sha256',sha(path),
           '--variant','cv_segment_global','--seed','42','--micro-batch','1','--grad-accum','1',
           '--global-batch','8','--max-steps','200']
    a4=parse_args(argv)
    assert parent_gate(a4)=='scripts/research/check_v7_formal_parent.py'
    with pytest.raises(ValueError,match='full-audit'):
        parse_args(argv+['--input-layout','V5'])
    v5=parse_args(argv+['--input-layout','V5','--data-binding',str(path),'--data-binding-sha256',sha(path)])
    assert parent_gate(v5)=='scripts/research/check_v5_formal_parent.py'
    assert v5.object_capacity==302
