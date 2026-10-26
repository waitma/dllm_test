"""Training-free probability aggregation; run test_cdr_marginal_inference.py in pllm.

The predictor receives token IDs only. Reference residues are never accepted.
Every conditional probability used for a target is computed while that target
is MASK. These are local hypotheses, not claims of exact posterior inference.
"""
import hashlib

import torch


def row_generators(tokens, seed):
    """Independent visible-input-derived streams, invariant to batch composition."""
    generators = []
    for row in tokens.cpu().tolist():
        digest = hashlib.sha256((str(seed)+':'+','.join(map(str, row))).encode()).digest()
        generators.append(torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little') % (2**63-1)))
    return generators


def random_rows(generators, shape, device):
    return torch.stack([torch.rand(shape, generator=g) for g in generators]).to(device)


def base_distribution(predict, batch, target, mask_id):
    tokens = batch['input_ids']
    assert tokens[target].eq(mask_id).all()
    probs = predict(tokens)
    assert torch.isfinite(probs).all() and (probs >= 0).all()
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs[..., 0]), atol=1e-5)
    return probs


def aggregate(predict, batch, target, mask_id, prior, family, observer=None):
    """Yield (name, posterior, counts, forward-equivalents, metadata).

Each yielded posterior is an alternative to the prior. Anchor mixtures are
formed outside this function. Shared prefixes reduce grid-search work; the
reported forward-equivalents include the prior and every conditional query
needed to produce an individual result, including padded inactive rows.
"""
    tokens = batch['input_ids']
    greedy = prior.argmax(-1)
    if observer is None:
        observer = lambda *args: None

    def query(inputs, scored, tag):
        assert torch.equal(inputs[~target], tokens[~target])
        assert not (scored & ~target).any()
        assert inputs[scored].eq(mask_id).all(), 'Self-visible probabilities must not enter the aggregate'
        probs = predict(inputs)
        assert torch.isfinite(probs).all() and (probs >= 0).all()
        assert torch.allclose(probs.sum(-1), torch.ones_like(probs[..., 0]), atol=1e-5)
        observer(tag, inputs, scored, probs)
        return probs

    kind = family['kind']
    if kind == 'mc_context':
        generators = row_generators(tokens, family.get('seed', 42))
        sums = torch.zeros_like(prior)
        counts = torch.zeros_like(tokens, dtype=torch.float32)
        for draw in range(1, max(family['draws'])+1):
            reveal = target & (random_rows(generators, tokens.shape[1:], tokens.device) < family['reveal'])
            # Consume the same stream for every temperature, including greedy.
            u = random_rows(generators, prior.shape[1:], tokens.device).clamp_(1e-7, 1-1e-7)
            temp = family['temperature']
            proposed = greedy if temp == 0 else (prior.clamp_min(1e-30).log()/temp - (-u.log()).log()).argmax(-1)
            inputs = tokens.clone(); inputs[reveal] = proposed[reveal]
            scored = target & ~reveal
            probs = query(inputs, scored, f'draw{draw}')
            sums += probs * scored[..., None]
            counts += scored
            if draw in family['draws']:
                posterior = torch.where((counts > 0)[..., None], sums/counts.clamp_min(1)[..., None], prior)
                yield f"{family['name']}_k{draw}", posterior, counts.clone(), draw+1, dict(
                    missing_views=int((target & counts.eq(0)).sum()), prior_fallback_for_missing=True)
    elif kind == 'leave_one_out':
        filled = tokens.clone(); filled[target] = greedy[target]
        positions = [row.nonzero().flatten() for row in target]
        posterior = prior.clone(); counts = torch.zeros_like(tokens, dtype=torch.float32)
        for index in range(max(map(len, positions))):
            inputs = filled.clone(); scored = torch.zeros_like(target)
            for i, pos in enumerate(positions):
                if index < len(pos):
                    inputs[i, pos[index]] = mask_id; scored[i, pos[index]] = True
            probs = query(inputs, scored, f'position{index}')
            posterior[scored] = probs[scored]; counts += scored
        assert counts[target].eq(1).all()
        yield family['name'], posterior, counts, 1+max(map(len, positions)), dict(
            padded_queries_counted=True, max_target_length=max(map(len, positions)))
    elif kind == 'branch_context':
        # Enumerate alternatives at one anchor site at a time, then marginalize
        # with prior probabilities; unresolved tail mass keeps the prior.
        confidence = prior.max(-1).values
        entropy = -(prior * prior.clamp_min(1e-30).log()).sum(-1)
        rank = confidence if family['anchor_rule'] == 'confidence' else entropy
        order = rank.masked_fill(~target, -torch.inf).argsort(dim=-1, descending=True, stable=True)
        sums = torch.zeros_like(prior); counts = torch.zeros_like(tokens, dtype=torch.float32)
        forwards = 0
        for anchor in range(family['anchors']):
            valid = target.sum(-1) > anchor
            pos = order[:, anchor]
            weights, candidates = prior[torch.arange(len(tokens), device=tokens.device), pos].topk(family['topk'], dim=-1)
            estimate = prior.clone()
            scored = target & valid[:, None]
            scored[torch.arange(len(tokens), device=tokens.device), pos] = False
            for k in range(family['topk']):
                inputs = tokens.clone()
                for i in range(len(tokens)):
                    if valid[i]: inputs[i, pos[i]] = candidates[i, k]
                probs = query(inputs, scored, f'anchor{anchor}_candidate{k}')
                estimate += weights[:, k, None, None]*(probs-prior)
                forwards += 1
            # The anchor site's self-visible outputs are excluded completely.
            sums += estimate * scored[..., None]; counts += scored
        posterior = torch.where((counts > 0)[..., None], sums/counts.clamp_min(1)[..., None], prior)
        assert (posterior >= -1e-6).all()
        posterior = posterior.clamp_min(0)
        posterior /= posterior.sum(-1, keepdim=True)
        yield family['name'], posterior, counts, forwards+1, dict(
            tail='prior weighted by unenumerated mass', anchor_sites_excluded=True)
    else:
        raise ValueError(kind)


def mix_prediction(tokens, target, prior, posterior, prior_weight):
    if not 0 <= prior_weight <= 1:
        raise ValueError(prior_weight)
    distribution = prior_weight*prior+(1-prior_weight)*posterior
    assert torch.isfinite(distribution).all() and (distribution >= 0).all()
    assert torch.allclose(distribution.sum(-1), torch.ones_like(distribution[..., 0]), atol=1e-5)
    prediction = tokens.clone(); prediction[target] = distribution.argmax(-1)[target]
    assert torch.equal(prediction[~target], tokens[~target])
    return prediction, distribution
