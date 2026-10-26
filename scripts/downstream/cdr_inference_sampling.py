"""Inference-only CDR samplers; run tests via test_cdr_inference_sampling.py.

All functions operate on the frozen AirGen bridge, never on reference labels.
The explicitly named oracle diagnostic alone accepts reference tokens. Custom
greedy repair and entropy-budget decoding are adaptations, not paper replicas.
"""
import math

import torch


def transition_rates(step, steps, eta):
    """Linear clean-probability schedule, ReMDM-cap with substitution carryover."""
    if not 0 <= eta <= 1 or not 0 <= step < steps:
        raise ValueError('Invalid transition parameters')
    alpha_t, alpha_s = step / steps, (step + 1) / steps
    sigma = min(eta, (1 - alpha_s) / alpha_t) if alpha_t else eta
    if step == steps-1:
        sigma = 0.
    unmask = (alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t)
    assert -1e-7 <= unmask <= 1 + 1e-7
    return sigma, min(1., max(0., unmask))


def adaptive_positions(pending, probabilities, kind, value, force=False):
    """Select a progress-making prefix; entropy bound is sum(H) - max(H)."""
    selected = torch.zeros_like(pending)
    confidence = probabilities.max(-1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
    for i in range(len(pending)):
        pos = pending[i].nonzero().flatten()
        if not len(pos):
            continue
        order = confidence[i, pos].argsort(descending=True, stable=True)
        pos = pos[order]
        if force:
            count = len(pos)
        elif kind == 'threshold':
            count = max(1, int((confidence[i, pos] >= value).sum()))
        elif kind == 'entropy':
            h = entropy[i, pos]
            bound = h.cumsum(0) - h.cummax(0).values
            count = max(1, int((bound <= value + 1e-7).sum()))
        else:
            raise ValueError(kind)
        selected[i, pos[:count]] = True
    return selected


def sample(bridge, batch, target, config, seed=42, gold=None, replay=None):
    """Return predictions and per-step traces, including billed batched forwards.

Repair scores are stored when a position was masked, never read from an
unmasked self-visible prediction. Random policies use an isolated generator.
Oracle replays original commit positions but feeds gold for previous commits;
reported predictions themselves are never replaced with reference tokens.
"""
    kind, steps = config['kind'], config['steps']
    if steps < 1:
        raise ValueError('steps must be positive')
    if kind == 'oracle':
        if gold is None or replay is None or len(replay) != steps:
            raise ValueError('Oracle requires reference and frozen commit trace')
    elif gold is not None:
        raise ValueError('Non-oracle sampler must not receive reference')
    tokens = batch['input_ids'].clone()
    scores = torch.zeros_like(tokens, dtype=torch.float32)
    stored_conf = torch.zeros_like(scores)
    generator = torch.Generator(device=tokens.device).manual_seed(seed)
    traces = []
    for step in range(steps):
        before = tokens.clone()
        pending = target & tokens.eq(bridge.mask_id)
        remasked = torch.zeros_like(target)
        if kind in ('repair', 'random_repair') and step:
            for i in range(len(tokens)):
                positions = target[i].nonzero().flatten()
                count = max(1, math.ceil(len(positions) * config['fraction']))
                rank = (torch.rand(len(positions), device=tokens.device, generator=generator)
                        if kind == 'random_repair' else stored_conf[i, positions])
                chosen = positions[rank.argsort(stable=True)[:count]]
                remasked[i, chosen] = True
            tokens[remasked] = bridge.mask_id
            pending = remasked
        if kind in ('threshold', 'entropy') and not pending.any():
            break
        inputs = tokens.clone()
        if kind == 'oracle':
            accepted = target & ~pending
            inputs[accepted] = gold[accepted]
        logits = bridge.net(input_ids=inputs, chain_ids=batch['chain_ids'])['logits'].float()
        logits[..., bridge.special_token_list] = -math.inf
        logits[..., bridge.eos_id] += 1e-3 * math.log(step / steps + 1e-3)
        raw, proposal = logits.max(-1)
        probs = logits.softmax(-1)
        confidence = probs.gather(-1, proposal[..., None]).squeeze(-1)
        if kind == 'scheduled':
            cur_tokens, cur_scores = tokens.clone(), scores.clone()
            cur_tokens[pending] = proposal[pending]
            ranking = raw if config.get('confidence', 'raw') == 'raw' else confidence.clamp_min(1e-30).log()
            cur_scores[pending] = ranking[pending]
            masks, tokens, scores = bridge._decoding(
                output_tokens=tokens, output_scores=scores, cur_tokens=cur_tokens,
                cur_scores=cur_scores,
                decoding_strategy='confidence-deterministic-' + config.get('schedule', 'linear'),
                xt_neq_x0=pending, non_special_sym_mask=target,
                t=step+1, max_step=steps, noise=bridge.mask_id)
            assert torch.equal(masks, tokens.eq(bridge.mask_id))
            commit = pending & ~masks
        elif kind in ('threshold', 'entropy'):
            commit = adaptive_positions(pending, probs, kind, config['value'], step == steps-1)
            tokens[commit] = proposal[commit]
        elif kind in ('repair', 'random_repair'):
            commit = pending
            tokens[commit] = proposal[commit]
        elif kind == 'oracle':
            commit = replay[step]['commit'].to(tokens.device)
            assert not (commit & ~pending).any()
            tokens[commit] = proposal[commit]
        elif kind == 'remdm':
            sigma, unmask = transition_rates(step, steps, config['eta'])
            # Independent Bernoulli transition and categorical residue draw.
            u = torch.rand(tokens.shape, generator=generator, device=tokens.device)
            remasked = target & ~pending & (u < sigma)
            commit = pending & (u < unmask)
            # Gumbel-max on log probabilities; no random confidence ranking.
            u_cat = torch.rand(logits.shape, generator=generator, device=tokens.device).clamp_(1e-7, 1-1e-7)
            proposal = (logits - (-u_cat.log()).log()).argmax(-1)
            confidence = probs.gather(-1, proposal[..., None]).squeeze(-1)
            tokens[remasked] = bridge.mask_id
            tokens[commit] = proposal[commit]
        else:
            raise ValueError(kind)
        stored_conf[commit] = confidence[commit]
        assert torch.equal(tokens[~target], batch['input_ids'][~target])
        if kind not in ('repair', 'random_repair', 'remdm'):
            assert torch.equal(tokens[target & ~pending], before[target & ~pending])
        traces.append({k: v.detach().cpu().clone() for k, v in
                       dict(commit=commit, remasked=remasked, before=before, after=tokens,
                            proposal=proposal, pending=pending, confidence=confidence).items()})
    assert not tokens[target].eq(bridge.mask_id).any()
    return tokens, traces
