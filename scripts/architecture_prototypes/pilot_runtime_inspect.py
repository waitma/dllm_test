"""Fixed CPU import/profile inspector, not a pilot or training launcher.

Activate pllm; call pilot_runtime_package.inspect_fresh() on a verified runtime,
or an explicitly labelled temporary source-copy rehearsal. No model construction,
GPU, dataset/scope gate, diagnostic generation, or training operation is performed.
"""
import argparse
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]


def isolated_search_path(runtime, original, paths, loaded_names):
    """Remove live-root .pth search entries before the first project import."""
    if any(name.split('.')[0] in ('scripts','dllm','examples') for name in loaded_names):
        raise ValueError('Inspection requires a fresh process with no project modules loaded')
    runtime,original=Path(runtime).resolve(),Path(original).resolve()
    kept=[]
    for value in paths:
        if not value:continue
        path=Path(value).resolve()
        if path==runtime:continue
        if path.is_relative_to(original) and not path.is_relative_to(runtime):continue
        kept.append(value)
    return [str(runtime),*kept]


def project_namespace_specs(runtime):
    """Pin our namespace packages; site-packages may contain another 'scripts'."""
    result={}
    for name in ('scripts','examples'):
        directory=Path(runtime)/name
        if not directory.is_dir():raise ValueError('Missing project package: '+name)
        if (directory/'__init__.py').is_file():continue
        spec=ModuleSpec(name,loader=None,is_package=True)
        spec.submodule_search_locations=[str(directory)]
        result[name]=spec
    return result


def inspect(mode,pool):
    metadata_name='pilot_runtime_manifest.json' if mode=='runtime' else 'SOURCE_COPY_REHEARSAL_ONLY.json'
    metadata=json.loads((ROOT/metadata_name).read_text())
    original=metadata['source_root'] if mode=='runtime' else metadata['original_source_root']
    sys.path[:]=isolated_search_path(ROOT,original,sys.path,list(sys.modules))
    for name,spec in project_namespace_specs(ROOT).items():sys.modules[name]=module_from_spec(spec)
    from scripts.architecture_prototypes import pilot_runtime_package as package
    if mode=='runtime':
        before=package.verify_runtime(ROOT)
        files=before['runtime_files_sha256']
    else:
        before=package.read_json(ROOT/package.REHEARSAL)
        package.require(before.get('kind')=='temporary_source_copy_rehearsal_only'
                        and before.get('production_acceptance') is False,'Invalid rehearsal identity')
        files=before['source_files_sha256']
    for name,value in files.items():
        package.require(package.sha(package.archive.relative_file(ROOT,name))==value,'Inspection source differs: '+name)
    import torch
    def forbidden(*args,**kwargs):raise RuntimeError('CPU source inspection forbids model/GPU/data construction')
    torch.cuda.init=forbidden;torch.cuda._lazy_init=forbidden
    from scripts import run_immune_research,run_immune_architecture,run_immune_refined,enqueue_immune_refined
    from scripts.architecture_prototypes import certified_pilot,certified_pilot_execution
    from dllm.pipelines.immune_llada.research.model import ResearchModel
    from dllm.pipelines.immune_llada.research_next.model import ArchitectureModel
    from dllm.pipelines.immune_llada.research_refined.model import RefinedModel
    from dllm.pipelines.immune_llada.v3.model import ImmuneModel
    for model in (ResearchModel,ArchitectureModel,RefinedModel,ImmuneModel):model.__init__=forbidden
    roots={module.__name__:str(module.ROOT) for module in
           (run_immune_research,run_immune_architecture,run_immune_refined,enqueue_immune_refined,certified_pilot)}
    package.require(all(value==str(ROOT) for value in roots.values()),'A model/adapter ROOT escaped the runtime')
    sources=run_immune_refined.fingerprints()
    package.require(all(files.get(name)==value for name,value in sources.items()),'Profile source map escaped runtime')
    profiles={}
    for variant in package.archive.FAMILY:
        summary=enqueue_immune_refined.verify_profile_evidence(pool,variant)
        profiles[variant]={'summary_sha256':package.sha(pool/'profiles'/variant/'summary.json'),
                           'complete_sha256':package.sha(pool/'profiles'/variant/'COMPLETE.json'),
                           'recommended_micro_batch':summary['recommended_micro_batch']}
    binding=certified_pilot_execution.source_binding()
    package.require(binding['model']==sources,'Execution and model sources differ')
    for group in binding.values():
        package.require(all(files.get(name)==value for name,value in group.items()),'Adapter/execution source outside runtime')
    imported={}
    for name,module in list(sys.modules.items()):
        if name.split('.')[0] not in ('scripts','dllm','examples'):continue
        file=getattr(module,'__file__',None)
        if file:
            path=Path(file).resolve()
            package.require(path.is_relative_to(ROOT),'Imported project module escaped runtime: '+name)
            relative=str(path.relative_to(ROOT))
            package.require(files.get(relative)==package.sha(path),'Imported module lacks runtime source binding: '+name)
            imported[name]=relative
        for path in getattr(module,'__path__',[]):
            package.require(Path(path).resolve().is_relative_to(ROOT),'Project package namespace escaped runtime: '+name)
    for name,value in files.items():
        package.require(package.sha(ROOT/name)==value,'Source changed during CPU inspection')
    if mode=='runtime':package.require(package.verify_runtime(ROOT)==before,'Runtime manifest changed during inspection')
    else:package.require(package.read_json(ROOT/package.REHEARSAL)==before,'Rehearsal manifest changed during inspection')
    return dict(kind='cpu_import_and_original_profile_compatibility_only',inspection_mode=mode,
                production_acceptance=False,model_constructed=False,gpu_started=False,data_gate_executed=False,
                diagnostic500_executed=False,training_started=False,runtime_root=str(ROOT),module_roots=roots,
                measured_source_sha256=sources,execution_source_binding=binding,profiles=profiles,
                imported_modules=imported,metric_protocol=certified_pilot.metric_protocol(),
                versions=certified_pilot.software_versions())


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--inspection-mode',choices=['runtime','source-copy-rehearsal'],required=True)
    parser.add_argument('--pool-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(inspect(args.inspection_mode,args.pool_dir.resolve()),indent=2,allow_nan=False))


if __name__=='__main__':main()
