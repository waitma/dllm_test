"""V5 object decoding; run pytest scripts/tests/immune_llada/test_v5_canvas.py."""
import math
import torch


def parse_object(tokenizer, tokens, *, expected_chains=None):
    """Stop at the first terminal; report missing terminal/invalid grammar explicitly."""
    tokens = [int(t) for t in tokens]
    end_id = tokenizer.special_id('<protd>')
    if end_id not in tokens:
        return dict(status='capacity_exhausted', chains=[], terminal_offset=None)
    end = tokens.index(end_id)
    before = tokens[:end]
    if tokenizer.mask_token_id in before:
        return dict(status='incomplete', chains=[], terminal_offset=end)
    amino = set(tokenizer.encode_residues('LAGVSERTIDPKQNFYMHWCXBUZO'))
    separator = tokenizer.chain_separator_id()
    parts = [[]]
    for token in before:
        if token == separator:
            parts.append([])
        elif token in amino:
            parts[-1].append(tokenizer.token(token))
        else:
            return dict(status='invalid_token', chains=[], terminal_offset=end)
    if any(not part for part in parts) or (expected_chains is not None and len(parts) != expected_chains):
        return dict(status='invalid_chain_structure', chains=[], terminal_offset=end)
    return dict(status='ok', chains=[''.join(part) for part in parts], terminal_offset=end)


@torch.no_grad()
def generate_features(model, features, *, steps=32, temperature=0.0, generator=None, expected_chains=None):
    """Confidence-ordered absorbing-mask decoding over a fixed object capacity.

    Return parsed results plus raw canvas IDs. No checkpoint or temporary files
    are written. Tokens after the first valid terminal are ignored, including
    unresolved masks. Caller may declare the required number of output chains.
    """
    if type(steps) is not int or steps < 1 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError('steps must be positive; temperature must be finite and nonnegative')
    device = next(model.parameters()).device
    features = {k:v.to(device) for k,v in features.items()}
    tok = model.tokenizer
    ids = features['input_ids'].clone()
    canvas = features['target_slot_mask'] & features['attention_mask']
    if ((ids == tok.special_id('<protd>')) & canvas).any():
        raise ValueError('Generation canvas contains a supplied terminal; use generation-mode inputs')
    if not canvas.any(1).all():
        raise ValueError('Each generation request needs a target canvas')
    counts = ([expected_chains]*ids.shape[0] if expected_chains is None or isinstance(expected_chains, int)
              else list(expected_chains))
    if len(counts) != ids.shape[0] or any(n is not None and n not in (1, 2) for n in counts):
        raise ValueError('expected_chains must declare one or two chains per object')
    initial = ((ids == tok.mask_token_id) & canvas).sum(1)
    allowed = torch.zeros(tok.vocab_size, dtype=torch.bool, device=device)
    allowed[tok.encode_residues('LAGVSERTIDPKQNFYMHWCXBUZO')] = True
    allowed[tok.chain_separator_id()] = True
    allowed[tok.special_id('<protd>')] = True
    done = [False]*ids.shape[0]
    was_training = model.training
    used_steps = 0
    model.eval()
    try:
        for step in range(steps):
            used_steps = step+1
            logits, _ = model.denoise(**{**features, 'input_ids':ids})
            logits = logits.float().masked_fill(~allowed, float('-inf'))
            for b in range(len(done)):
                if done[b]:
                    continue
                positions = ((ids[b] == tok.mask_token_id) & canvas[b]).nonzero().flatten()
                if len(positions):
                    scores = logits[b, positions]
                    if not torch.isfinite(scores[:, allowed]).all():
                        raise ValueError('Nonfinite generation logits')
                    probs = (scores / (temperature or 1.0)).softmax(-1)
                    predictions = (probs.argmax(-1) if temperature == 0 else
                                   torch.multinomial(probs, 1, generator=generator).squeeze(-1))
                    confidence = probs.gather(1, predictions[:,None]).squeeze(1)
                    left = int(initial[b]) * (steps-step-1) // steps
                    count = max(0, len(positions)-left)
                    chosen = confidence.topk(count).indices
                    ids[b, positions[chosen]] = predictions[chosen]
                parsed = parse_object(tok, ids[b,canvas[b]].tolist(), expected_chains=counts[b])
                done[b] = parsed['status'] == 'ok'
            if all(done):
                break
    finally:
        model.train(was_training)
    return [dict(**parse_object(tok, ids[b,canvas[b]].tolist(), expected_chains=counts[b]),
                 canvas_ids=ids[b,canvas[b]].tolist(), steps=used_steps) for b in range(ids.shape[0])]
