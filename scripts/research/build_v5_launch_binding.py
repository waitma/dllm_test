"""Activate pllm; run python -B -m scripts.research.build_v5_launch_binding --help.

Freeze code/config and bind a passed full V5 audit. Does not submit a GPU job.
"""
import argparse
import json
from pathlib import Path
import shutil

from scripts.research.v5_data_binding import sha,require,verify_binding


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--model-config',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];audit_path=args.audit.resolve();audit=json.loads(audit_path.read_text())
    require(audit['status']=='passed' and not audit['reasons'],'Full audit not passed')
    for name,expected in audit['source_sha256'].items():require(sha(name)==expected,'Audit code changed: '+name)
    output=args.output_dir.resolve();require(not output.exists(),'Refusing to replace a launch package')
    output.mkdir(parents=True)
    source=output/'source';source.mkdir()
    files={}
    for directory in ('dllm','scripts','examples'):
        for path in sorted((root/directory).rglob('*.py')):
            relative=path.relative_to(root);dest=source/relative
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
            files[str(dest)]=sha(dest)
    adapters=['output/generic_eight_rank_trainer_integration_20261002/eight_rank_trainer_adapter_v1.py',
              'output/generic_eight_worker_group_integration_20261002/eight_rank_checkpoint_adapter_v2.py']
    for relative in adapters:
        dest=source/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/relative).read_bytes())
        files[str(dest)]=sha(dest)
    for path in [root/'scripts/research/check_v5_formal_parent.py',root/'scripts/research/v5_data_binding.py',
                 root/'output/generic_eight_rank_integration_20261002/eight_rank_contracts_v1.py',
                 *(Path(name) for name in audit['source_sha256'])]:
        files[str(path)]=sha(path)
    model=output/'research_config.json';model.write_bytes(args.model_config.read_bytes());config=json.loads(model.read_text())
    binding=dict(schema='immune.v5.launch_data_binding.v1',status='ready',data_dir=audit['data_dir'],
        capacity=audit['capacity'],max_length=audit['max_length'],input_layout='V5',variant=config['variant'],
        manifest_sha256=audit['manifest_sha256'],audit=dict(path=str(audit_path),sha256=sha(audit_path)),
        model_config=dict(path=str(model),sha256=sha(model)),
        scope='Original V5 data; no new V7 decontamination; VHH absent in the full scan',
        source_fingerprint=audit['source_sha256'])
    bindpath=output/'data_binding.json';bindpath.write_text(json.dumps(binding,indent=2)+'\n')
    # Full hashes were computed in the full audit; this construction check is
    # explicit stat-only. The subsequent native parent gate always rehashes all.
    verify_binding(bindpath,sha(bindpath),Path(audit['data_dir']),audit['capacity'],config['variant'],full_hash=False)
    manifest=output/'source_manifest.json';manifest.write_text(json.dumps({'files':files},indent=2,sort_keys=True)+'\n')
    argv=['--source-root',str(source),'--repo-root',str(root),'--data-dir',audit['data_dir'],
          '--source-manifest',str(manifest),'--source-manifest-sha256',sha(manifest),
          '--data-binding',str(bindpath),'--data-binding-sha256',sha(bindpath),
          '--input-layout','V5','--variant',config['variant'],'--object-capacity',str(audit['capacity'])]
    (output/'launch_data_args.json').write_text(json.dumps(argv,indent=2)+'\n')
    receipt=dict(status='data_bound_pending_native_parent_preflight',training_submitted=False,source_files=len(files),
                 data_binding=str(bindpath),data_binding_sha256=sha(bindpath),
                 source_manifest=str(manifest),source_manifest_sha256=sha(manifest),
                 driver=str(source/'scripts/research/run_v7_formal.py'),data_arguments=str(output/'launch_data_args.json'),
                 remaining_runtime='Existing eight-rank NCCL launcher and explicit run/output/schedule/batch parameters are required')
    (output/'package.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
