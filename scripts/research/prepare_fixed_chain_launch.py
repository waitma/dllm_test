"""Activate pllm; python -B -m scripts.research.prepare_fixed_chain_launch.

Freeze the audited 75.6M full-chain/global model with the established eight-rank
Trainer. Prepares one unique launch package; does not enqueue or submit it.
"""
import json
import shutil
from pathlib import Path
from scripts.research.v5_data_binding import sha, require
from scripts.research.fixed_chain_data_binding import verify_binding
ROOT=Path('/project_c20250601/251105016/project/dllm_test')
BASE=ROOT/'output/v5_full_binding_20261007/launch_v3'
BUNDLE=BASE.parent/'launch_fixed_chain_75m_20261012'
CONFIG=ROOT/'configs/models/v5_fixed_chain_f011_75m_20261012'
AUDIT=ROOT/'output/v5_fixed_chain_f011_20261012/full_audit.json'


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')


def replace(path,old,new):
    s=path.read_text();require(s.count(old)==1,'Patch anchor differs: '+str(path)+' / '+old);path.write_text(s.replace(old,new))


def main():
    require(not BUNDLE.exists(),'Refusing to overwrite a launch package')
    original=json.loads((BASE/'source_manifest.json').read_text())['files']
    for path,expected in original.items():require(sha(path)==expected,'Baseline changed: '+path)
    BUNDLE.mkdir();source=BUNDLE/'source'
    shutil.copytree(BASE/'source',source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(ROOT/'dllm/pipelines/immune_llada/fixed_chain',source/'dllm/pipelines/immune_llada/fixed_chain',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('fixed_chain_data_binding.py','check_fixed_chain_parent.py','fixed_chain_validation.py'):
        shutil.copy2(ROOT/'scripts/research'/name,source/'scripts/research'/name)
    driver=source/'scripts/research/run_v7_formal.py'
    s=driver.read_text().replace('check_v5_formal_parent.py','check_fixed_chain_parent.py').replace('scripts/research/v5_data_binding.py','scripts/research/fixed_chain_data_binding.py')
    s=s.replace("choices=(*VARIANTS, *FACTORIAL)","choices=('fixed_chain_full_global',)")
    s=s.replace('from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator','from dllm.pipelines.immune_llada.fixed_chain import FixedChainCollator, FixedChainConfig, FixedChainModel')
    s=s.replace('from scripts.architecture_prototypes import validation_metrics as metrics','from scripts.research import fixed_chain_validation as metrics')
    s=s.replace('config = ImmuneConfig(ablation=args.input_layout','config = FixedChainConfig(ablation=args.input_layout')
    s=s.replace('model = FactorialModel(config, args.variant) if args.variant in FACTORIAL else RefinedModel(config, VARIANTS[args.variant])',"model = FixedChainModel(config)\n        require(sum(p.numel() for p in model.parameters()) == 75601176, 'Parameter count differs')")
    s=s.replace('collator = ImmuneCollator.from_config(config)','collator = FixedChainCollator(config)')
    driver.write_text(s)
    launcher=source/'scripts/research/launch_v5_direct.py'
    s=launcher.read_text().replace("args.variant == 'cv_segment_global'","args.variant == 'fixed_chain_full_global'")
    s=s.replace('from dllm.pipelines.immune_llada.v3.layout import ImmuneCollator','from dllm.pipelines.immune_llada.fixed_chain import FixedChainCollator, FixedChainConfig, FixedChainModel')
    s=s.replace('from scripts.architecture_prototypes.validation_metrics import selected_token_metrics, summarize','from scripts.research.fixed_chain_validation import selected_token_metrics, summarize')
    s=s.replace("config = ImmuneConfig(ablation='V5'","config = FixedChainConfig(ablation='V5'").replace('full_capacity=302','full_capacity=343')
    s=s.replace('collator = ImmuneCollator.from_config(config)','collator = FixedChainCollator(config)').replace('length = 1218','length = 1136').replace('1218 is the bound full-V5 observed maximum','1136 is the audited fixed-chain full-V5 maximum')
    s=s.replace('model = FactorialModel(config, args.variant).cuda().train()','model = FixedChainModel(config).cuda().train()')
    s=s.replace("require(scores['split_count'] == 1, 'V5 split validation failed')","require(scores['split_count'] == 0 and scores['terminal_count'] == 2 and scores['chain_count'] == 2, 'Fixed-chain validation failed')\n    require(sum(p.numel() for p in model.parameters()) == 75601176, 'Parameter count differs')")
    launcher.write_text(s)
    shutil.copy2(CONFIG/'research_config.json',BUNDLE/'research_config.json')
    binding=dict(schema='immune.fixed_chain.launch_binding.v1',status='ready',data_dir=str(ROOT/'data/prepared/immune_v5_receptor_completion'),
        capacity=343,max_length=2048,input_layout='V5',variant='fixed_chain_full_global',
        manifest_sha256=sha(ROOT/'data/prepared/immune_v5_receptor_completion/dataset_manifest.json'),
        audit=dict(path=str(AUDIT),sha256=sha(AUDIT)),model_config=dict(path=str(BUNDLE/'research_config.json'),sha256=sha(BUNDLE/'research_config.json')),
        original_v5_binding=dict(path=str(BASE/'data_binding.json'),sha256=sha(BASE/'data_binding.json')),
        scope='User-selected original V5 rows; audited fixed-chain rendering; no new V7 decontamination claim')
    write(BUNDLE/'data_binding.json',binding)
    verify_binding(BUNDLE/'data_binding.json',sha(BUNDLE/'data_binding.json'),Path(binding['data_dir']),343,'fixed_chain_full_global',full_hash=False)
    files={str(p):sha(p) for p in sorted(source.rglob('*.py'))}
    for path,expected in original.items():
        if not Path(path).is_relative_to(BASE/'source'):files[path]=expected
    for name in ('fixed_chain_data_binding.py','check_fixed_chain_parent.py','prepare_fixed_chain_launch.py','audit_fixed_chain.py'):
        p=ROOT/'scripts/research'/name;files[str(p)]=sha(p)
    write(BUNDLE/'source_manifest.json',dict(files=files))
    argv=json.loads((BASE/'launch_training_args.json').read_text())
    changes={'--source-root':source,'--source-manifest':BUNDLE/'source_manifest.json','--source-manifest-sha256':sha(BUNDLE/'source_manifest.json'),
             '--data-binding':BUNDLE/'data_binding.json','--data-binding-sha256':sha(BUNDLE/'data_binding.json'),
             '--variant':'fixed_chain_full_global','--object-capacity':343,'--micro-batch':64,'--grad-accum':1,
             '--output-dir':ROOT/'output/immune_v5_fixed_chain_75m_300k_b512'}
    for key,value in changes.items():argv[argv.index(key)+1]=str(value)
    write(BUNDLE/'launch_training_args.json',argv)
    write(BUNDLE/'direct_launch.json',dict(schema='immune.v5.direct_launch.v1',initialization='random_from_scratch',
        model_parameters_frozen=False,runtime_dir=str(ROOT/'output/immune_v5_fixed_chain_75m_300k_b512_supervision/startup'),
        arguments_sha256=sha(BUNDLE/'launch_training_args.json'),launcher_sha256=sha(launcher),
        checkpoint_retention='best3_eval_loss_plus_latest_full',data_audit_scope=binding['scope']))
    changed=[str(p.relative_to(source)) for p in source.rglob('*.py') if not (BASE/'source'/p.relative_to(source)).exists() or sha(p)!=sha(BASE/'source'/p.relative_to(source))]
    write(BUNDLE/'preparation.json',dict(status='prepared',parent=str(BASE),changed_source_files=sorted(changed),
        initialization='scratch',parameters=75601176,training_submitted=False,global_batch=512,max_steps=300000,
        remaining='Eight-rank synthetic profile and mandatory native full content hash check occur before formal training'))
    print(BUNDLE)

if __name__=='__main__':main()
