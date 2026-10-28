"""FP32 forward KL with pair -> chain-pair -> record normalization.

Activate pllm; run:
  USE_TF=0 python -B -m pytest scripts/tests/immune_llada/test_structure_mlm.py -q
Return sums and eligible-record counts separately for exact global DDP and
microbatch normalization. Intra/inter coefficients are independent: missing
inter supervision never silently increases the intra coefficient.
"""
import torch
from torch.nn import functional as F


def chain_aware_kd(student, teacher, pair_indices, chain_ids, noise_kind,
                   temperature=1., mask_weights=None, pair_confidence=None):
    """Optionally attenuate each pair with a detached confidence in [0, 1].

    Denominators remain the original eligible pair/group/record counts, including
    zero-confidence pairs. Dividing by the sum of confidences would cancel the
    desired reduction when an entire interface has low teacher confidence.
    """
    if temperature <= 0 or student.shape != teacher.shape or student.ndim != 2:
        raise ValueError("Invalid distogram shapes or temperature")
    if pair_indices.shape != (len(student), 3):
        raise ValueError("Distograms and pair indices disagree")
    if mask_weights is not None and (len(mask_weights) != 3 or min(mask_weights) < 0
                                    or sum(mask_weights) <= 0):
        raise ValueError("Invalid VV/VM/MM weights")
    if pair_confidence is not None:
        if (pair_confidence.shape != (len(student),)
                or not torch.isfinite(pair_confidence).all()
                or (pair_confidence < 0).any() or (pair_confidence > 1).any()):
            raise ValueError("Pair confidence must be finite [N] values in [0, 1]")
        pair_confidence = pair_confidence.detach().to(device=student.device, dtype=torch.float32)
    zero = student.sum() * 0
    result = {"intra_sum": zero, "inter_sum": zero,
              "intra_count": student.new_zeros(()), "inter_count": student.new_zeros(()),
              "pair_count": len(student)}
    if not len(student):
        return result
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("Non-finite distogram logits")
    row, i, j = pair_indices.unbind(-1)
    if (i == j).any():
        raise ValueError("Self pairs are excluded")
    ci, cj = chain_ids[row, i], chain_ids[row, j]
    if (ci < 0).any() or (cj < 0).any():
        raise ValueError("Non-chain positions cannot carry structure targets")
    teacher_log = F.log_softmax(teacher.detach().float() / temperature, dim=-1)
    student_log = F.log_softmax(student.float() / temperature, dim=-1)
    loss = (teacher_log.exp() * (teacher_log - student_log)).sum(-1) * temperature**2
    if pair_confidence is not None:
        result.update(unweighted_pair_kl_sum=loss.detach().sum(),
                      confidence_sum=pair_confidence.sum(), confidence_pair_count=len(student))
        loss = loss * pair_confidence
    corrupted = noise_kind.eq(1) | noise_kind.eq(2)
    mask_group = corrupted[row, i].long() + corrupted[row, j].long()
    keys = torch.stack([row, torch.minimum(ci, cj), torch.maximum(ci, cj)], dim=1)
    groups, inverse = torch.unique(keys, dim=0, return_inverse=True)
    group_loss, active_groups = [], []
    for index in range(len(groups)):
        members = inverse.eq(index)
        if mask_weights is None:
            value = loss[members].mean()
        else:
            numerator, denominator = zero, 0.
            for kind, weight in enumerate(mask_weights):
                subset = members & mask_group.eq(kind)
                if weight > 0 and subset.any():
                    numerator = numerator + weight * loss[subset].mean()
                    denominator += weight
            if denominator == 0:
                continue
            value = numerator / denominator
        group_loss.append(value)
        active_groups.append(index)
    if not group_loss:
        return result
    group_loss = torch.stack(group_loss)
    groups = groups[active_groups]
    for name, use in (("intra", groups[:, 1].eq(groups[:, 2])),
                      ("inter", groups[:, 1].ne(groups[:, 2]))):
        per_record = [group_loss[use & groups[:, 0].eq(sample)].mean()
                      for sample in groups[use, 0].unique()]
        if per_record:
            result[name + "_sum"] = torch.stack(per_record).sum()
            result[name + "_count"] = student.new_tensor(float(len(per_record)))
    for kind, name in enumerate(("vv", "vm", "mm")):
        subset = mask_group.eq(kind)
        result[name + "_sum"] = loss[subset].detach().sum()
        result[name + "_count"] = subset.sum()
    return result


def normalized_objective(mlm_sum, mlm_count, kd, intra_weight=.05, inter_weight=.05,
                         denominators=None, world_size=1):
    """For DDP pass counts over the whole optimizer step and all ranks.

    DDP averages gradients, hence multiply each local numerator by world_size.
    Callers sum microbatch contributions without another accumulation divisor.
    """
    if min(intra_weight, inter_weight) < 0 or world_size < 1:
        raise ValueError("Invalid objective coefficients")
    if denominators is None:
        denominators = (mlm_count, kd["intra_count"], kd["inter_count"])
    terms = ((mlm_sum, denominators[0], 1.),
             (kd["intra_sum"], denominators[1], intra_weight),
             (kd["inter_sum"], denominators[2], inter_weight))
    return sum(value * (weight * world_size) / torch.as_tensor(
        count, device=value.device, dtype=value.dtype).clamp_min(1)
        for value, count, weight in terms)
