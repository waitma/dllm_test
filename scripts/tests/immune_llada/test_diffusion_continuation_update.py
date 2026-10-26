"""Two/four-rank CPU oracles for global record-view normalization and accumulation.

Activate pllm; USE_TF=0 python -B -m pytest THIS_FILE -q.
"""
import torch
from torch.nn.parallel import DistributedDataParallel
from scripts.research.run_diffusion_continuation import update


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(.2))

    def forward(self,input_ids,weights,**metadata):
        values=((self.weight*input_ids).square()*weights).sum(1)
        loss=values.sum()
        zero=loss.detach()*0
        return dict(loss_sum=loss,ce_sum=loss.detach(),eos_ce_sum=zero,
                    tokens=torch.tensor(input_ids.numel()),eos_tokens=zero,
                    view_loss_sums=values.detach())


def worker(rank,path):
    torch.distributed.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:
        model=DistributedDataParallel(Toy())
        optimizer=torch.optim.SGD(model.parameters(),lr=.05)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
        rows=3 if rank==0 else 5
        x=torch.arange(1,rows+1,dtype=torch.float)[:,None]*.1
        weights=torch.zeros_like(x) if rank==0 else torch.ones_like(x)
        update(model,dict(input_ids=x,weights=weights),optimizer,scheduler,2,2,bf16=False)
        expected=.2-.05*(2*.2*float((torch.arange(1,6)*.1).square().sum())/8)
        torch.testing.assert_close(model.module.weight,torch.tensor(expected),rtol=1e-6,atol=1e-8)
        # Different local CDR populations must still reduce to one global view mean.
        model.module.weight.data.fill_(.2)
        scope=torch.tensor([0,1,1] if rank==0 else [0,0,2,2,2])[:,None]
        edited=torch.tensor([False,False,False] if rank==0 else [True,False,False,False,False])[:,None]
        grouped=update(model,dict(input_ids=x,weights=weights,mask_scope=scope,edit_task=edited,
            cdr_full_mask=scope.ne(0),cdr_eligible=torch.ones_like(scope,dtype=torch.bool)),
            optimizer,scheduler,2,2,bf16=False)
        torch.testing.assert_close(model.module.weight,torch.tensor(expected),rtol=1e-6,atol=1e-8)
        assert [grouped['task_views_'+name] for name in ('edit','chain_mask','cdr_single','cdr_multi')]==[1,2,2,3]
        assert grouped['task_loss_cdr_single']==0.
        reconstructed=sum(grouped['task_views_'+name]*grouped['task_loss_'+name]
                          for name in ('edit','chain_mask','cdr_single','cdr_multi'))/8
        assert abs(reconstructed-grouped['loss'])<1e-8
        # Force clipping and verify the actual update uses the configured limit.
        model.module.weight.data.fill_(.2)
        large = torch.full((rows, 1), 10.)
        result = update(model, dict(input_ids=large, weights=torch.ones_like(large)),
                        optimizer, scheduler, 2, 2, bf16=False, gradient_clip=.25)
        torch.testing.assert_close(model.module.weight, torch.tensor(.2 - .05 * .25),
                                   rtol=1e-6, atol=1e-8)
        assert abs(result['grad_norm_pre_clip'] - 40.) < 1e-4
        assert result['grad_norm'] == result['grad_norm_pre_clip']
        assert result['grad_norm_post_clip'] <= .250001
        assert result['grad_clipped']
        assert abs(result['grad_clip_coefficient'] - .25 / 40.) < 1e-8
        # A nonfinite gradient must fail before any optimizer update.
        before = model.module.weight.detach().clone()
        try:
            update(model, dict(input_ids=large * float('nan'), weights=torch.ones_like(large)),
                   optimizer, scheduler, 2, 2, bf16=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Nonfinite gradient was accepted')
        torch.testing.assert_close(model.module.weight, before, rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


def test_distributed_view_mean_includes_zero_contribution_records(tmp_path):
    torch.multiprocessing.spawn(worker,args=(str(tmp_path/'gloo'),),nprocs=2,join=True)


def four_rank_worker(rank, path):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://'+path, rank=rank, world_size=4)
    try:
        model = DistributedDataParallel(Toy())
        optimizer = torch.optim.SGD(model.parameters(), lr=.05)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        full = torch.arange(1, 1025, dtype=torch.float)[:, None] / 1024
        x = full[rank::4]
        result = update(model, dict(input_ids=x, weights=torch.ones_like(x)),
                        optimizer, scheduler, 48, 4, bf16=False)
        expected = .2 - .05 * (2 * .2 * full.square().mean())
        torch.testing.assert_close(model.module.weight, expected, rtol=1e-6, atol=1e-8)
        assert result['record_views'] == 1024
        assert abs(result['loss'] - float((.2*full).square().mean())) < 1e-8
    finally:
        torch.distributed.destroy_process_group()


def test_four_rank_512_records_with_partial_last_microbatch(tmp_path):
    torch.multiprocessing.spawn(four_rank_worker, args=(str(tmp_path/'gloo4'),), nprocs=4, join=True)
