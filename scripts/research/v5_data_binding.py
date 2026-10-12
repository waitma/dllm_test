"""Validate immutable V5 audit/data bindings; run test_v5_data_binding.py via pytest.

No dataset certificates are written and no V7 decontamination claim is implied.
"""
import hashlib
import json
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def require(ok,message):
    if not ok:raise ValueError(message)


def verify_binding(path, expected_sha, data_dir, capacity, variant, *, full_hash=True):
    path=Path(path);data_dir=Path(data_dir)
    require(path.is_absolute() and path==path.resolve(),'Binding path must be canonical absolute')
    require(sha(path)==expected_sha,'Data binding SHA differs')
    binding=json.loads(path.read_text())
    require(binding.get('schema')=='immune.v5.launch_data_binding.v1' and binding.get('status')=='ready','V5 binding is not ready')
    require(str(data_dir)==binding['data_dir'] and data_dir==data_dir.resolve(),'Wrong V5 data path')
    require(capacity==binding['capacity'] and variant==binding['variant'],'V5 model/capacity binding differs')
    require(binding['input_layout']=='V5' and binding['max_length']==2048,'V5 layout binding differs')
    audit_path=Path(binding['audit']['path'])
    require(sha(audit_path)==binding['audit']['sha256'],'Audit report changed')
    audit=json.loads(audit_path.read_text())
    require(audit['schema']=='immune.v5.full_layout_audit.v1' and audit['status']=='passed','Audit not passed')
    require(audit['data_dir']==str(data_dir) and audit['capacity']==capacity and audit['max_length']==2048,'Audit layout/data mismatch')
    counts=audit['counts']
    require(not audit['reasons'] and counts.get('invalid_records',0)==0 and counts.get('excluded_vhh',0)==0,'Unresolved data exclusions/errors')
    require(counts['valid_records']==counts['records']==sum(audit['splits'].values()),'Incomplete audit coverage')
    manifest_path=data_dir/'dataset_manifest.json'
    require(sha(manifest_path)==binding['manifest_sha256']==audit['manifest_sha256'],'V5 manifest changed')
    manifest=json.loads(manifest_path.read_text())
    require(manifest['schema_version']=='immune_llada.semantic.v1' and manifest['format']=='jsonl','Manifest schema differs')
    expected={}
    for split in ('train','valid'):
        require(manifest['splits'][split]['records']==audit['splits'][split],'Split count differs')
        for row in manifest['splits'][split]['shards']:
            require(row['path'] not in expected,'Duplicate manifest shard')
            expected[row['path']]=(split,row)
    require(len(audit['shards'])==len(expected),'Shard coverage differs')
    seen=set()
    for shard in audit['shards']:
        relative=shard['relative_path'];require(relative in expected and relative not in seen,'Unexpected or repeated audit shard')
        seen.add(relative);split,entry=expected[relative]
        file=data_dir/relative
        require(file.resolve().is_relative_to(data_dir) and not file.is_symlink(),'Noncanonical shard path')
        require(shard['path']==str(file) and shard['split']==split and shard['source']==entry['source'],'Shard identity differs')
        require(shard['counts']['records']==entry['records'],'Shard count differs')
        stat=file.stat()
        require(stat.st_size==shard['stat']['size'] and stat.st_mtime_ns==shard['stat']['mtime_ns'],'Shard stat changed: '+relative)
        if full_hash:require(sha(file)==shard['sha256'],'Shard content changed: '+relative)
    model=Path(binding['model_config']['path'])
    require(sha(model)==binding['model_config']['sha256'],'Model config changed')
    config=json.loads(model.read_text())
    require(config['variant']==variant and config['base']['ablation']=='V5' and config['base']['full_capacity']==capacity,'Bound model differs')
    expected_base=dict(ablation='V5',d_model=768,n_layers=8,n_heads=12,mlp_hidden_size=3072,
                       max_length=2048,full_capacity=capacity,fragment_capacity=64,schema_version='immune_llada.noesmc.v3')
    require(config['base']==expected_base,'Bound architecture differs from the formal driver')
    require(config.get('schema')=='immune_architecture.factorial.v1' and config.get('options')=={'segment_global':True}
            and config.get('factor_order')==['G','S','L'] and variant=='cv_segment_global','Wrong new-version architecture')
    return dict(schema='v5.formal.read_only_parent_check.v1',status='passed',data_dir=str(data_dir),
        metadata={'dataset_manifest.json':binding['manifest_sha256']},data_binding_sha256=expected_sha,
        audit_sha256=binding['audit']['sha256'],splits=audit['splits'],shards_checked=len(seen),
        content_hashes_verified=full_hash,model_config_sha256=binding['model_config']['sha256'])
