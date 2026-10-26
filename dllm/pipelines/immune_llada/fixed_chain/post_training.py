"""Controlled post-training recipes on the unchanged fixed-chain input layout.

Run: python -m pytest scripts/tests/immune_llada/test_fixed_chain_post_training.py -q
These are exploratory adaptations, not reproductions of published SFT/RL methods.
"""
import math
from dataclasses import asdict
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .model import FixedChainModel, FixedChainConfig
from .post_training_data import observed_end_batch, relation_blind_features


RECIPES = {
    'continued': {},
    'task_mix': {'mask': 'tasks'},
    'high_mask': {'mask': 'high'},
    'inverse_t': {'inverse_t': True},
    'eos_balance': {'mask': 'tasks', 'eos_weight': 0.1},
    'endpoint': {'mask': 'tasks', 'endpoint_weight': 0.1},
    'repair': {'mask': 'tasks', 'repair': True},
    'global': {'mask': 'tasks', 'topology': 'global'},
    'late_fusion': {'mask': 'tasks', 'topology': 'late'},
    'layerscale': {'mask': 'tasks', 'learn_scale': True},
    'lora': {'mask': 'tasks', 'lora_rank': 16},
    'contrastive': {'mask': 'tasks', 'contrastive_weight': 0.05},
    'observed_eos': {'mask': 'tasks', 'observed_eos': True},
    'cdr_mix': {'mask': 'tasks', 'cdr_mask': True},
    'observed_cdr': {'mask': 'tasks', 'observed_eos': True, 'cdr_mask': True},
    'distill': {'mask': 'tasks', 'distill_weight': 0.2, 'teacher_steps': 4},
    'observed_cdr_distill': {'mask': 'tasks', 'observed_eos': True, 'cdr_mask': True,
                             'distill_weight': 0.2, 'teacher_steps': 4},
    'binding_aux': {'mask': 'tasks', 'binding_weight': 0.1},
    'observed_cdr_binding': {'mask': 'tasks', 'observed_eos': True, 'cdr_mask': True,
                            'binding_weight': 0.1},
    'weak_preference': {'mask': 'tasks', 'observed_eos': True, 'cdr_mask': True,
                        'preference_weight': 0.1, 'preference_beta': 1.0},
    'weak_preference_strong': {'mask': 'tasks', 'observed_eos': True, 'cdr_mask': True,
                               'preference_weight': 0.1, 'preference_beta': 10.0},
    'weak_preference_shuffle': {'mask': 'tasks', 'observed_eos': True, 'cdr_mask': True,
                                'preference_weight': 0.1, 'preference_beta': 1.0,
                                'shuffle_preference': True},
    'positive_sft': {'mask': 'tasks', 'observed_eos': True, 'cdr_mask': True,
                     'preference_weight': 0.1, 'positive_only': True},
}


class LoRALinear(nn.Module):
    """Zero-initialized update; initial predictions exactly match the parent."""
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b)


class PostTrainingModel(FixedChainModel):
    """Load parent weights first, then install one explicitly recorded intervention."""
    def __init__(self, config, recipe='continued'):
        super().__init__(config)
        self.recipe = recipe
        self.post_options = dict(RECIPES[recipe])

    def install_adaptation(self):
        if self.post_options.get('binding_weight'):
            self.binding_head = nn.Linear(self.config.d_model, 2)
        if self.post_options.get('learn_scale'):
            for block in self.backbone.transformer.blocks:
                for name in ('attention_scale', 'ffn_scale'):
                    initial = getattr(block, name).clone()
                    delattr(block, name)
                    block.register_parameter(name, nn.Parameter(initial))
        if self.post_options.get('lora_rank'):
            self.requires_grad_(False)
            for block in self.backbone.transformer.blocks:
                for name in ('q_proj', 'k_proj', 'v_proj', 'attn_out', 'ff_proj', 'up_proj', 'ff_out'):
                    setattr(block.base, name, LoRALinear(getattr(block.base, name), self.post_options['lora_rank']))

    def save_pretrained(self, directory):
        """Export a distinct, strict-loadable post-training checkpoint contract."""
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.save_research_config(directory)
        (directory / 'immune_config.json').write_text(json.dumps(asdict(self.config), indent=2) + '\n')
        (directory / 'vocabulary.json').write_text(json.dumps(
            [self.tokenizer.token(i) for i in range(self.tokenizer.vocab_size)], indent=2) + '\n')
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
                  str(directory / 'model.safetensors'))

    def save_research_config(self, directory):
        """Also preserve the recipe when the native Trainer writes the checkpoint."""
        settings = {'schema': 'immune.fixed_chain.post_training.v1', 'recipe': self.recipe,
                    'options': self.post_options, 'base': asdict(self.config)}
        (Path(directory) / 'research_config.json').write_text(json.dumps(settings, indent=2) + '\n')

    @classmethod
    def from_pretrained(cls, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        settings = json.loads((directory / 'research_config.json').read_text())
        if (settings.get('schema') != 'immune.fixed_chain.post_training.v1'
                or settings.get('recipe') not in RECIPES
                or settings['options'] != RECIPES[settings['recipe']]
                or settings['base'] != json.loads((directory / 'immune_config.json').read_text())):
            raise ValueError('Post-training checkpoint schema/options mismatch')
        model = cls(FixedChainConfig(**settings['base']), settings['recipe'])
        model.install_adaptation()
        expected = [model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]
        if json.loads((directory / 'vocabulary.json').read_text()) != expected:
            raise ValueError('Post-training vocabulary mismatch')
        model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
        return model

    def mask_key(self, layer):
        topology = self.post_options.get('topology')
        if topology == 'global':
            return ('global',)
        if topology == 'late':
            return ('chain',) if layer < self.config.n_layers // 2 else ('global',)
        return super().mask_key(layer)

    def layer_allowed(self, layer, chain_ids, position_ids, attention_mask):
        valid = attention_mask.bool()
        allowed = valid[:, None, :].expand(-1, chain_ids.shape[1], -1)
        if self.mask_key(layer) != ('global',):
            same = chain_ids[:, :, None] == chain_ids[:, None, :]
            allowed = (allowed & same) | (~valid[:, :, None] & valid[:, None, :])
        return allowed

    def _build_attention_biases(self, chain_ids, position_ids, attention_mask):
        same = ((chain_ids[:, :, None] == chain_ids[:, None, :]) &
                (chain_ids[:, :, None] >= 0) & (chain_ids[:, None, :] >= 0))
        base = torch.where(same[:, None], self.chain_bias[1][None, :, None, None],
                           self.chain_bias[0][None, :, None, None])
        biases = {}
        for layer in range(self.config.n_layers):
            key = self.mask_key(layer)
            if key not in biases:
                allowed = self.layer_allowed(layer, chain_ids, position_ids, attention_mask)
                biases[key] = base.masked_fill(~allowed[:, None], torch.finfo(base.dtype).min)
        return biases

    def training_corruption(self, batch):
        """Keep conditions intact; select only observed supervision positions."""
        ids, eligible = batch['input_ids'], batch['eligible_mask']
        if self.recipe == 'continued':
            noised, _, selected = self.corrupt(batch)
            return noised, selected, torch.ones((len(ids), 1), device=ids.device)
        ratio = torch.empty((len(ids), 1), device=ids.device).uniform_(0.001, 1)
        if self.post_options.get('inverse_t'):
            ratio = ratio.clamp_min(0.05)  # Explicit variance-limited surrogate.
        selected = (torch.rand(ids.shape, device=ids.device) < ratio) & eligible
        mode = self.post_options.get('mask')
        for row in range(len(ids)):
            draw = torch.rand((), device=ids.device).item() if mode else 1.0
            if mode == 'high' and draw < 0.5 or mode == 'tasks' and draw < 0.1:
                selected[row] = eligible[row]
            elif mode == 'tasks' and draw < 0.4:
                chains = batch['chain_ids'][row][eligible[row]].unique()
                if len(chains):
                    uid = chains[torch.randint(len(chains), (), device=ids.device)]
                    selected[row] = eligible[row] & batch['chain_ids'][row].eq(uid)
            elif mode == 'tasks' and draw < 0.6:
                chains = batch['chain_ids'][row][eligible[row]].unique()
                uid = chains[torch.randint(len(chains), (), device=ids.device)] if len(chains) else -1
                positions = (eligible[row] & batch['chain_ids'][row].eq(uid)).nonzero().flatten()
                if len(positions):
                    start = int(torch.randint(len(positions), (), device=ids.device))
                    size = max(1, len(positions) // 4)
                    selected[row].zero_()
                    selected[row, positions[start:start + size]] = True
            if not selected[row].any() and eligible[row].any():
                positions = eligible[row].nonzero().flatten()
                selected[row, positions[torch.randint(len(positions), (), device=ids.device)]] = True
        if self.post_options.get('cdr_mask'):
            for row in range(len(ids)):
                # Half the records use annotated CDR reconstruction, otherwise task_mix.
                if batch['cdr_mask'][row].any() and torch.rand((), device=ids.device) < 0.5:
                    selected[row] = batch['cdr_mask'][row] & eligible[row]
        return ids.masked_fill(selected, self.tokenizer.mask_token_id), selected, ratio

    def endpoint_loss(self, logits, batch, selected):
        """First-EOS hazard likelihood, only on fully hidden, fully observed chains.

        Penalize EOS before the true end and reward EOS at the true end. Never
        provide length to denoise, or supervise synthetic completion residues.
        """
        logp = logits.float().log_softmax(-1)[..., self.terminal_id]
        values = []
        for row in range(len(logits)):
            for uid in batch['chain_ids'][row][batch['terminal_mask'][row]].unique():
                chain = batch['chain_ids'][row].eq(uid)
                eos = (chain & batch['terminal_mask'][row]).nonzero().flatten()
                if len(eos) != 1:
                    continue
                end = int(eos[0])
                before = chain & (torch.arange(logits.shape[1], device=logits.device) < end)
                observed = before | (chain & batch['terminal_mask'][row])
                if not selected[row, observed].all() or not batch['residue_mask'][row, before].all():
                    continue
                non_end = torch.log1p(-logp[row, before].exp().clamp(max=1 - 1e-6))
                values.append(-logp[row, end] - non_end.sum())
        return torch.stack(values).mean() if values else logp.sum() * 0

    @torch.no_grad()
    def teacher_distribution(self, teacher, features):
        """Refine the same visible input for four steps; never read truth masks/lengths."""
        ids = features['input_ids'].clone()
        canvas = features['target_slot_mask'] & features['attention_mask']
        steps = self.post_options['teacher_steps']
        for step in range(steps):
            logits = teacher.denoise(**{**features, 'input_ids': ids})[0]
            probs = logits.float()[..., self.allowed_target].softmax(-1)
            confidence, index = probs.max(-1)
            tokens = self.allowed_target.nonzero().flatten()[index]
            for row in range(len(ids)):
                pending = (canvas[row] & ids[row].eq(self.tokenizer.mask_token_id)).nonzero().flatten()
                if len(pending):
                    count = math.ceil(len(pending) / (steps-step))
                    chosen = pending[confidence[row, pending].topk(count).indices]
                    ids[row, chosen] = tokens[row, chosen]
        return probs

    def forward(self, teacher=None, preference_batch=None, preference_directions=None, **batch):
        if self.post_options.get('observed_eos'):
            batch = observed_end_batch(batch, self.tokenizer.mask_token_id)
        noised, selected, ratio = self.training_corruption(batch)
        features = {**self.feature_args(batch), 'input_ids': noised}
        if self.post_options.get('repair'):
            # A detached first prediction supplies errors; only selected sites change.
            # Do not cache no-grad cast weights for the later gradient-bearing pass.
            # Reusing those weights breaks checkpoint recomputation under autocast.
            with torch.no_grad(), torch.autocast(
                noised.device.type, enabled=torch.is_autocast_enabled(noised.device.type),
                dtype=torch.get_autocast_dtype(noised.device.type), cache_enabled=False,
            ):
                guessed = self.denoise(**features)[0].argmax(-1)
            replace = selected & (torch.rand_like(noised.float()) < 0.25)
            features['input_ids'] = torch.where(replace, guessed, noised)
        logits, hidden = self.denoise(**features, output_hidden_states=True)
        labels = batch['labels'].masked_fill(~selected, -100)
        losses = F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100, reduction='none')
        loss = losses.sum() / selected.sum().clamp_min(1)
        if self.post_options.get('inverse_t'):
            loss = (losses.sum(1) / ratio[:, 0] / batch['eligible_mask'].sum(1).clamp_min(1)).mean()
        eos = selected & batch['terminal_mask']
        if self.post_options.get('eos_weight'):
            loss = loss + self.post_options['eos_weight'] * losses[eos].sum() / eos.sum().clamp_min(1)
        if self.post_options.get('endpoint_weight'):
            loss = loss + self.post_options['endpoint_weight'] * self.endpoint_loss(logits, batch, selected)
        if self.post_options.get('distill_weight'):
            if teacher is None:
                raise ValueError('Distillation requires the frozen parent teacher')
            # Keep the detached teacher's autocast cache separate from student gradients.
            with torch.autocast(noised.device.type,
                                enabled=torch.is_autocast_enabled(noised.device.type),
                                dtype=torch.get_autocast_dtype(noised.device.type), cache_enabled=False):
                target = self.teacher_distribution(teacher, features)
            student_logp = logits.float()[..., self.allowed_target].log_softmax(-1)
            divergence = F.kl_div(student_logp, target, reduction='none').sum(-1)
            loss = loss + self.post_options['distill_weight'] * divergence[selected].sum() / selected.sum().clamp_min(1)
        if self.post_options.get('binding_weight'):
            blind = relation_blind_features(self, batch)
            _, receptor_hidden = self.denoise(**blind, output_hidden_states=True)
            mask = batch['residue_mask'] & batch['target_slot_mask']
            pooled = (receptor_hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            scores = self.binding_head(pooled)
            known = batch['binding_labels'] >= 0
            if known.any():
                loss = loss + self.post_options['binding_weight'] * F.cross_entropy(scores[known].float(), batch['binding_labels'][known])
        if self.post_options.get('contrastive_weight') and len(noised) > 1:
            other, _, _ = self.training_corruption(batch)
            _, second = self.denoise(**{**features, 'input_ids': other}, output_hidden_states=True)
            mask = batch['residue_mask'][..., None]
            def pool(value):
                return F.normalize((value.float() * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1)
            similarity = pool(hidden) @ pool(second).T / 0.1
            # Identical receptor labels are not negatives.
            same = batch['labels'][:, None, :].eq(batch['labels'][None, :, :]).all(-1)
            same.fill_diagonal_(False)
            similarity = similarity.masked_fill(same, float('-inf'))
            loss = loss + self.post_options['contrastive_weight'] * F.cross_entropy(
                similarity, torch.arange(len(noised), device=noised.device))
        if self.post_options.get('preference_weight'):
            from .preference import preference_objective
            if preference_batch is None:
                raise ValueError('Preference recipes require matched training pairs')
            if self.post_options.get('shuffle_preference') and preference_directions is None:
                raise ValueError('Shuffled preference control requires explicit directions')
            loss = loss + self.post_options['preference_weight'] * preference_objective(
                self, teacher, preference_batch, beta=self.post_options.get('preference_beta', 1.),
                positive_only=self.post_options.get('positive_only', False),
                directions=preference_directions)
        return {'loss': loss, 'selected_count': selected.sum().detach()}
