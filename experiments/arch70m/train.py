"""Single-GPU architecture trainer; launch via run.sh --stage screen --gpus 0."""
import argparse
from importlib.metadata import version
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback
import uuid
import numpy as np
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file, save_file
from dllm.pipelines.immune_llada.fixed_chain.v_family import VFamilyCollator, corrupt
from dllm.pipelines.llada.models.configuration_llada import ActivationCheckpointingStrategy
from scripts.research.bert_mlm_data import CompleteEpochBatchSampler, IndexedPreparedDataset
from scripts.research.run_bert_mlm import make_scheduler, capture_rng, restore_rng
from .common import ROOT, read, write, append, sha, digest, source_binding, commit_id, stamp, lock
from .models import build
from .evaluate import evaluate

def setup(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.set_device(0); torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)

def construct(settings, variant, seed, device):
    model = build(settings, variant, seed).to(device)
    model.backbone.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
    model.train()
    s = settings['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=s['learning_rate'], betas=tuple(s['adam_betas']),
                                  eps=s['adam_epsilon'], weight_decay=s['weight_decay'])
    return model, optimizer

def update(model, batch, optimizer, scheduler, micro, device):
    """Normalize accumulated gradients by whole-step token and chain populations."""
    token_count = int(batch['labels'].ne(-100).sum()); family_count = int(batch['family_labels'].ne(-100).sum())
    if not token_count and not family_count: raise ValueError('No loss population')
    optimizer.zero_grad(set_to_none=True); token_sum = 0.; family_sum = 0.
    for start in range(0, len(batch['input_ids']), micro):
        part = {k: v[start:start+micro].to(device, non_blocking=True) for k,v in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
            result = model(**part)
            loss = result['loss_sum']/max(1, token_count) + model.family_loss_weight*result['family_loss_sum']/max(1, family_count)
        loss.backward(); token_sum += float(result['loss_sum'].detach()); family_sum += float(result['family_loss_sum'].detach())
        del part, result, loss
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    learning_rate = optimizer.param_groups[0]['lr']; optimizer.step(); scheduler.step()
    return dict(loss=token_sum/max(1,token_count), family_loss=family_sum/max(1,family_count),
                total_loss=token_sum/max(1,token_count)+model.family_loss_weight*family_sum/max(1,family_count),
                grad_norm=norm, learning_rate=learning_rate, selected_tokens=token_count, family_chains=family_count)

def profile_batch(clean, size, max_length, tokenizer):
    row = int(clean['attention_mask'].sum(1).argmax()); length = clean['input_ids'].shape[1]
    result = {}
    for key, value in clean.items():
        value = value[row:row+1].clone()
        if value.ndim == 2 and value.shape[1] == length:
            fill = 0
            if key in ('input_ids', 'labels'): fill = tokenizer.encode_residues('A')[0]
            elif key in ('attention_mask', 'eligible_mask', 'residue_mask', 'context_residue_mask'): fill = 1
            elif key == 'chain_ids': fill = -1
            value = torch.nn.functional.pad(value, (0,max_length-length), value=fill)
        result[key] = value.repeat(size, *([1]*(value.ndim-1)))
    return result

def profile(settings, variant, seed, clean, device, output):
    model, optimizer = construct(settings, variant, seed, device)
    scheduler = make_scheduler(optimizer, 100, 5)
    total = torch.cuda.get_device_properties(device).total_memory
    candidates = []; chosen = None; upper = None; micro = 8
    while micro <= settings['training']['global_batch']:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        safe=False; batch=None
        try:
            batch = profile_batch(clean, micro, settings['model']['max_length'], model.tokenizer)
            batch = corrupt(batch, model.tokenizer, torch.Generator().manual_seed(9100))
            start = time.monotonic()
            for _ in range(3): update(model, batch, optimizer, scheduler, micro, device)
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated(); reserved = torch.cuda.max_memory_reserved()
            free, capacity = torch.cuda.mem_get_info()
            external = capacity-free-torch.cuda.memory_reserved()
            safe = reserved+max(0,external) <= .85*total
            item = dict(micro_batch=micro,status='passed',allocated_bytes=allocated,reserved_bytes=reserved,
                        external_bytes=max(0,external),total_bytes=total,seconds_per_step=(time.monotonic()-start)/3,safe=safe)
            candidates.append(item); print(json.dumps(dict(stage='profile',variant=variant,**item)), flush=True)
            del batch
        except torch.cuda.OutOfMemoryError:
            candidates.append(dict(micro_batch=micro,status='oom'))
        if safe: chosen=micro
        else: upper=micro
        write(output.with_name('profile_attempts.json'),dict(variant=variant,candidates=candidates))
        if upper is None:
            if micro==settings['training']['global_batch']: break
            micro=min(micro*2,settings['training']['global_batch'])
        else:
            if chosen is None or upper-chosen<=8: break
            micro=8*((chosen+upper)//16)
        optimizer.zero_grad(set_to_none=True)
    del model, optimizer, scheduler; gc.collect(); torch.cuda.empty_cache()
    if chosen is None: raise RuntimeError('No safe microbatch >=8; inspect profile and available memory')
    result = dict(variant=variant,micro_batch=chosen,grad_accum=math.ceil(settings['training']['global_batch']/chosen),
                  final_micro_batch=settings['training']['global_batch']%chosen or chosen,
                  length=settings['model']['max_length'],candidates=candidates,device=torch.cuda.get_device_name(),
                  note='Three real optimizer updates at padded maximum length. Fresh initialization after profiling.')
    write(output, result); return result

def save_checkpoint(out, model, optimizer, scheduler, noise, state, binding):
    directory = out/'checkpoints'; directory.mkdir(exist_ok=True)
    target = directory/f"step_{state['step']:06d}"
    if target.exists(): raise FileExistsError(target)
    temporary = directory/('.pending_'+uuid.uuid4().hex); temporary.mkdir()
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}, str(temporary/'model.safetensors'))
    torch.save(optimizer.state_dict(),temporary/'optimizer.pt')
    torch.save(scheduler.state_dict(),temporary/'scheduler.pt')
    torch.save(capture_rng(noise),temporary/'rng.pt')
    write(temporary/'state.json',state); write(temporary/'binding.json',binding)
    write(temporary/'receipt.json',dict(status='complete',files={p.name:sha(p) for p in temporary.iterdir()}))
    temporary.rename(target); write(out/'latest.json',dict(path=str(target),step=state['step']))
    keep = set(sorted(directory.glob('step_*'))[-2:]) | {target}
    for key in ('best_generation_step','best_representation_step'):
        if state.get(key) is not None: keep.add(directory/f"step_{state[key]:06d}")
    keep.update(directory/f'step_{x:06d}' for x in state.get('epoch_end_steps',[]))
    for path in directory.glob('step_*'):
        if path not in keep: shutil.rmtree(path)
    return target

def restore(path, model, optimizer, scheduler, noise, binding):
    receipt=read(path/'receipt.json')
    if receipt['status']!='complete' or any(sha(path/n)!=h for n,h in receipt['files'].items()): raise ValueError('Checkpoint checksum failed')
    if read(path/'binding.json') != binding: raise ValueError('Cannot resume changed code/data/architecture/microbatch/environment')
    model.load_state_dict(load_file(str(path/'model.safetensors')),strict=True)
    optimizer.load_state_dict(torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False))
    scheduler.load_state_dict(torch.load(path/'scheduler.pt',map_location='cpu',weights_only=False))
    restore_rng(torch.load(path/'rng.pt',map_location='cpu',weights_only=False),noise)
    return read(path/'state.json')

def run(args):
    device=torch.device('cuda:0'); setup(args.seed)
    root=Path(args.output_root); out=root/'trials'/f'{args.stage}_{args.variant}_s{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    with lock(out/'trial.lock',blocking=False):
        settings=read(ROOT/'experiments/arch70m/base.json'); index_path=root/'data/index.json'; index=read(index_path)
        source=source_binding(); epochs=1 if args.stage=='screen' else 3
        source_sha=digest(source); data_sha=sha(index_path)
        frozen=dict(settings=settings,variant=args.variant,seed=args.seed,epochs=epochs,source_sha256=source_sha,data_sha256=data_sha)
        if (out/'status.json').exists() and read(out/'status.json')['status']=='completed':
            prior=read(out/'binding.json')
            if any(prior[k]!=v for k,v in frozen.items()): raise ValueError('Completed trial binding differs')
            print('Verified completed trial; skipping '+str(out),flush=True); return
        dataset=IndexedPreparedDataset(index_path)
        collate=VFamilyCollator(FixedChainConfigProxy(settings),family_spec=settings['family_prediction'])
        clean=collate([dataset[i] for i in range(min(512,len(dataset)))])
        profile_path=out/'profile.json'
        write(out/'status.json',dict(status='profiling',step=0,updated=stamp(),pid=os.getpid()))
        prof=read(profile_path) if profile_path.exists() else profile(settings,args.variant,args.seed,clean,device,profile_path)
        del clean
        micro=prof['micro_batch']; steps_per_epoch=math.ceil(len(dataset)/512); total=epochs*steps_per_epoch
        model,optimizer=construct(settings,args.variant,args.seed,device)
        scheduler=make_scheduler(optimizer,total,math.ceil(.05*total)); noise=torch.Generator().manual_seed(args.seed+90000)
        binding={**frozen,'micro_batch':micro,'total_steps':total,'torch':torch.__version__,'cuda':torch.version.cuda,
                 'device':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'sdpa':model.sdpa_backend,
                 'packages':{n:version(n) for n in ('numpy','transformers','safetensors','scikit-learn')},
                 'profile_sha256':sha(profile_path)}
        if (out/'binding.json').exists() and read(out/'binding.json')!=binding: raise ValueError('Existing attempt has a different binding')
        write(out/'binding.json',binding); write(out/'sources.json',source)
        state=dict(step=0,epoch=0,next_batch=0,seen_records=0,seen_tokens=0,elapsed_train_seconds=0.,
                   best_generation=None,best_representation=None,epoch_end_steps=[],wandb_id=uuid.uuid4().hex)
        resuming=(out/'latest.json').exists()
        if args.stage=='confirm':
            freeze=read(root/'selection.json')
            if args.variant not in freeze['confirm_variants'] or freeze['source_sha256']!=source_sha or freeze['data_sha256']!=data_sha:
                raise ValueError('Confirmation requires a matching frozen candidate selection')
        if resuming:
            state=restore(Path(read(out/'latest.json')['path']),model,optimizer,scheduler,noise,binding)
            # A crash after logging but before checkpointing must not duplicate plotted steps.
            for name in ('train.jsonl','eval.jsonl'):
                if (out/name).exists():
                    kept=[json.loads(x) for x in (out/name).read_text().splitlines() if x.strip()]
                    (out/name).write_text(''.join(json.dumps(x)+'\n' for x in kept if x['step']<=state['step']))
        metrics_dir=out/'metrics'; metrics_dir.mkdir(exist_ok=True)
        write(out/'status.json',dict(status='initial_evaluation',step=state['step'],total_steps=total,updated=stamp(),pid=os.getpid(),parameters=sum(p.numel() for p in model.parameters())))
        wb=None
        if args.wandb:
            try:
                import wandb
                wb=wandb.init(entity='pressure_jian',project='dllm-immune',group='arch70m-v8-20261028',
                    id=state['wandb_id'],name=out.name,resume='allow' if resuming else 'never',
                    dir=str(out),config=binding,settings=wandb.Settings(init_timeout=30))
            except Exception as exc: write(out/'wandb_unavailable.json',dict(error=str(exc),local_logging='active'))
        try:
            if not resuming:
                value=evaluate(model,index,'screen',device,metrics_dir,0,tasks=True); append(out/'eval.jsonl',value)
                save_checkpoint(out,model,optimizer,scheduler,noise,state,binding)
            for epoch in range(state['epoch'],epochs):
                start_batch=state['next_batch'] if epoch==state['epoch'] else 0
                sampler=CompleteEpochBatchSampler(len(dataset),512,0,1,args.seed,epoch,start_batch=start_batch)
                loader=DataLoader(dataset,batch_sampler=sampler,collate_fn=collate,num_workers=2,pin_memory=True)
                for batch_number,clean in enumerate(loader,start_batch):
                    batch=corrupt(clean,model.tokenizer,noise)
                    torch.cuda.reset_peak_memory_stats(); start=time.monotonic()
                    values=update(model,batch,optimizer,scheduler,micro,device); torch.cuda.synchronize()
                    seconds=time.monotonic()-start; state['step']+=1; state['seen_records']+=len(clean['input_ids'])
                    state['seen_tokens']+=int(clean['attention_mask'].sum()); state['elapsed_train_seconds']+=seconds
                    state['epoch']=epoch; state['next_batch']=batch_number+1
                    epoch_end=state['next_batch']==steps_per_epoch
                    if epoch_end:
                        state['epoch']=epoch+1; state['next_batch']=0; state['epoch_end_steps'].append(state['step'])
                    values.update(step=state['step'],epoch=epoch,seen_records=state['seen_records'],seen_tokens=state['seen_tokens'],
                        gpu_hours=state['elapsed_train_seconds']/3600,seconds_per_step=seconds,micro_batch=micro,
                        records=len(clean['input_ids']),
                        allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
                    append(out/'train.jsonl',values)
                    if state['step']%25==0 or state['step']==1: print(json.dumps(dict(stage='train',variant=args.variant,**values)),flush=True)
                    if wb:
                        try: wb.log({'train/'+k:v for k,v in values.items()},step=state['step'])
                        except Exception as exc:
                            write(out/'wandb_unavailable.json',dict(error=str(exc),local_logging='active')); wb=None
                    do_tasks=state['step']%250==0 or epoch_end
                    if state['step']%100==0 or do_tasks:
                        value=evaluate(model,index,'screen',device,metrics_dir,state['step'],tasks=do_tasks)
                        append(out/'eval.jsonl',value)
                        if do_tasks:
                            gen=value['generation']; terms=[gen[k] for k in ('conditional_ce','cdr_ce') if k in gen]
                            g=sum(terms)/len(terms) if terms else None; r=value['representation']['macro_auroc']
                            if g is not None and (state['best_generation'] is None or g<state['best_generation']):
                                state.update(best_generation=g,best_generation_step=state['step'])
                            if state['best_representation'] is None or r>state['best_representation']:
                                state.update(best_representation=r,best_representation_step=state['step'])
                        print(json.dumps(dict(stage='evaluated',step=state['step'],tasks=do_tasks)),flush=True)
                    if do_tasks: save_checkpoint(out,model,optimizer,scheduler,noise,state,binding)
                    write(out/'status.json',dict(status='training',step=state['step'],total_steps=total,updated=stamp(),pid=os.getpid(),metrics=values))
            if args.stage=='confirm':
                freeze=read(root/'selection.json')
                if args.variant not in freeze['confirm_variants']: raise ValueError('Candidate not frozen')
                value=evaluate(model,index,'confirm',device,metrics_dir,state['step'],tasks=True); append(out/'eval.jsonl',value)
                from .ab_specificity import evaluate_ab
                evaluate_ab(model,device,out/'ab_specificity',index['protected_ab_binding'])
            write(out/'status.json',dict(status='completed',step=state['step'],total_steps=total,updated=stamp(),
                                         parameters=sum(p.numel() for p in model.parameters()),source_sha256=source_sha))
        except BaseException:
            write(out/'status.json',dict(status='failed',step=state['step'],total_steps=total,updated=stamp(),error=traceback.format_exc()))
            raise
        finally:
            if wb: wb.finish()

def FixedChainConfigProxy(settings):
    from dllm.pipelines.immune_llada.fixed_chain.model import FixedChainConfig
    return FixedChainConfig(**settings['model'])

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output-root',required=True)
    parser.add_argument('--stage',choices=('screen','confirm'),required=True); parser.add_argument('--variant',required=True)
    parser.add_argument('--seed',type=int,default=42); parser.add_argument('--wandb',action='store_true')
    run(parser.parse_args())

if __name__=='__main__': main()
