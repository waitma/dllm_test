"""Evaluate fixed-framework, full-span CDR recovery on the bound internal holdout.

Activate pllm; invoke through diffusion_continuation_eval.py --checkpoint ABS
--index ABS --output ABS --device cuda. CDR diagnostics do not replace legacy CE.
"""
import torch
from torch.nn import functional as F

from scripts.research.bert_mlm_eval import autocast
from scripts.research.diffusion_cdr_tasks import CDRS, cdr_spans
from scripts.research.diffusion_continuation_objective import DiffusionEditCollator


@torch.no_grad()
def evaluate_cdr(model, samples, device, rank=0, world=1, batch_size=8):
    from scripts.research.diffusion_continuation_eval import fixed_batch
    eligible = {i: {group for group, _, _, _ in cdr_spans(row["record"])}
                for i, row in enumerate(samples)}
    eligible = {i: groups for i, groups in eligible.items() if groups}
    collator = DiffusionEditCollator(model.config)
    modes = [(name, "cdr_single", [name], 1.) for name in CDRS]
    modes += [("multi_subset", "cdr_multi", None, 1.),
              ("all_six", "cdr_multi", list(CDRS), 1.),
              ("all_six_half_noise", "cdr_multi", list(CDRS), .5)]
    results = {}
    was_training = model.training
    model.eval()
    try:
        for name, scope, targets, ratio in modes:
            wanted = {CDRS.index(n)+1 for n in targets} if targets else set()
            indices = [i for i, groups in eligible.items() if wanted.issubset(groups)]
            local = indices[rank::world]
            totals = torch.zeros(4, device=device, dtype=torch.float64)
            for start in range(0, len(local), batch_size):
                selected = local[start:start+batch_size]
                batch = fixed_batch(collator, samples, selected, model.recipe, "joint", ratio, False,
                                    mask_scope=scope, target_cdrs=targets)
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast(device):
                    logits, _ = model.denoise(**model.feature_args(batch))
                target = batch["noise_kind"].eq(1)
                if target.any():
                    scores = logits[target].float()
                    labels = batch["labels"][target]
                    totals[0] += F.cross_entropy(scores, labels, reduction="sum").double()
                    totals[1] += scores.argmax(-1).eq(labels).sum()
                    totals[2] += target.sum()
                totals[3] += len(selected)
            if world > 1:
                torch.distributed.all_reduce(totals)
            ce, correct, tokens, records = totals.cpu().tolist()
            results[name] = dict(loss=ce/tokens if tokens else None,
                accuracy=correct/tokens if tokens else None, ce_sum=ce,
                correct=int(correct), tokens=int(tokens), records=int(records))
    finally:
        model.train(was_training)
    def avg(names, metric):
        values = [results[n][metric] for n in names if results[n][metric] is not None]
        return sum(values)/len(values) if values else None
    summary = dict(cdr_single_loss=avg(CDRS, "loss"), cdr_single_accuracy=avg(CDRS, "accuracy"),
                   cdr_multi_loss=results["multi_subset"]["loss"],
                   cdr_multi_accuracy=results["multi_subset"]["accuracy"],
                   cdr_all_six_loss=results["all_six"]["loss"],
                   cdr_all_six_accuracy=results["all_six"]["accuracy"])
    return dict(summary=summary, modes=results, eligible_records=len(eligible),
        scope="Internal known-length CDR recovery; framework and other loops fixed; not binding or experimental generation validation")
