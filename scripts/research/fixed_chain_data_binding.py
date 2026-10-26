"""Activate pllm; use check_fixed_chain_parent.py to verify this read-only data binding."""
import json
from pathlib import Path
from scripts.research.v5_data_binding import sha, require, verify_binding as verify_original


def verify_binding(path, expected_sha, data_dir, capacity, variant, *, full_hash=True):
    path=Path(path);data_dir=Path(data_dir)
    require(path.is_absolute() and path==path.resolve() and sha(path)==expected_sha,'Binding identity differs')
    b=json.loads(path.read_text())
    require(b['schema']=='immune.fixed_chain.launch_binding.v1' and b['status']=='ready','Binding not ready')
    require(str(data_dir)==b['data_dir'] and capacity==343 and variant=='fixed_chain_full_global','Wrong training preset')
    require(b['capacity']==capacity and b['variant']==variant and b['max_length']==2048,'Bound layout differs')
    # The previous audit proves schema/provenance for the same immutable V5 rows.
    original=b['original_v5_binding']
    verify_original(original['path'],original['sha256'],data_dir,302,'cv_segment_global',full_hash=False)
    manifest_path=data_dir/'dataset_manifest.json';manifest=json.loads(manifest_path.read_text())
    require(sha(manifest_path)==b['manifest_sha256'],'Manifest changed')
    require(manifest.get('training_ready') is not False,'Explicit training_ready=false remains a stop')
    audit_path=Path(b['audit']['path']);require(sha(audit_path)==b['audit']['sha256'],'Fixed-chain audit changed')
    a=json.loads(audit_path.read_text());expected_caps={'antibody_heavy':187,'antibody_light':154,'tcr_beta':187,'tcr_alpha':154}
    require(a['status']=='passed' and not a['reasons'] and a['data_dir']==str(data_dir),'Failed or wrong audit')
    require(a['layout_contract']=='immune.fixed_chain.f011.v1' and a['chain_capacities']==expected_caps,'Wrong chain layout')
    require(a['max_total']==1136 and a['renderer_parity']==170,'Audit geometry coverage differs')
    entries={str(data_dir/x['path']):x for s in manifest['splits'].values() for x in s['shards']}
    original_binding=json.loads(Path(original['path']).read_text())
    old_audit=json.loads(Path(original_binding['audit']['path']).read_text())
    old_hashes={x['path']:x['sha256'] for x in old_audit['shards']}
    seen=set()
    for x in a['shard_results']:
        name=x['path'];f=Path(name)
        require(name in entries and name not in seen and f==f.resolve() and f.is_relative_to(data_dir),'Shard identity differs')
        require(not x['reasons'] and x['records']==entries[name]['records'],'Shard count/errors differ')
        require(x['sha256']==old_hashes[name],'Fixed-chain audit did not cover original V5 bytes')
        if full_hash: require(sha(f)==x['sha256'],'Shard content changed: '+name)
        seen.add(name)
    require(seen==set(entries) and len(seen)==a['shards']==85,'Incomplete shard coverage')
    require(sum(x['records'] for x in a['shard_results'])==a['records']==7729193,'Incomplete record coverage')
    model=Path(b['model_config']['path']);require(sha(model)==b['model_config']['sha256'],'Model config changed')
    c=json.loads(model.read_text())
    expected=dict(ablation='V5',d_model=768,n_layers=8,n_heads=12,mlp_hidden_size=3072,max_length=2048,
                  full_capacity=343,fragment_capacity=64,schema_version='immune_llada.noesmc.v3',
                  layout_contract='immune.fixed_chain.f011.v1',first_chain_capacity=187,second_chain_capacity=154)
    require(c['schema']==expected['layout_contract'] and c['variant']==variant and c['base']==expected
            and c['options']=={'residual_scale':.1,'segment_global':True},'Unexpected model architecture')
    return dict(schema='v5.formal.read_only_parent_check.v1',status='passed',data_dir=str(data_dir),
                metadata={'dataset_manifest.json':b['manifest_sha256']},data_binding_sha256=expected_sha,
                audit_sha256=b['audit']['sha256'],splits={k:v['records'] for k,v in manifest['splits'].items()},
                shards_checked=len(seen),content_hashes_verified=full_hash,model_config_sha256=b['model_config']['sha256'])
