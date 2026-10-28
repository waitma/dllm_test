"""当前固定双链模型：配置 → Transformer 层 → 模型前向 → 训练目标。

Run after activating pllm:
    python -B -m pytest scripts/tests/immune_llada/test_fixed_chain*.py -q

输入布局见 layout.py，多轮生成见 generation.py。这里直接组合 LLaDA 的基础
模块，不再继承历史 research 模型；保留原有参数名和 checkpoint 格式。
张量记号：B = batch，S = token 数，D = hidden size，H = attention heads。
"""

from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from dllm.pipelines.llada.models.configuration_llada import ModelConfig
from dllm.pipelines.llada.models.modeling_llada import LLaDAModel
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import sample_bioseq_diffusion_noise
from ..data.records import CHAIN_ROLE_TO_ID
from ..v3.layout import SCOPES
from ..v3.model import ImmuneConfig
from ..v3.vocabulary import make_tokenizer

SCHEMA = "immune.fixed_chain.f011.v1"
VARIANT = "fixed_chain_full_global"
RESIDUES = "LAGVSERTIDPKQNFYMHWCXBUZO"
RECEPTOR_ROLES = ("antibody_heavy", "antibody_light", "tcr_beta", "tcr_alpha")


@dataclass
class FixedChainConfig(ImmuneConfig):
    """继承基础尺寸配置；每条链的容量包含一个待预测的 EOS。"""

    ablation: str = "V5"
    full_capacity: int = 343  # 187 + 154 + 固定的 dot、protd
    layout_contract: str = SCHEMA
    first_chain_capacity: int = 187  # H / β
    second_chain_capacity: int = 154  # L / α

    def __post_init__(self):
        super().__post_init__()
        if self.ablation != "V5" or self.layout_contract != SCHEMA:
            raise ValueError("Expected fixed-chain V5 f011 configuration")
        if min(self.first_chain_capacity, self.second_chain_capacity) < 2:
            raise ValueError("Each chain needs residue and EOS capacity")
        if self.full_capacity != self.first_chain_capacity + self.second_chain_capacity + 2:
            raise ValueError("Capacity must include both chain regions, dot and protd")


class FixedChainBlock(nn.Module):
    """Pre-norm 双向 Transformer：attention 和 FFN 残差均固定缩放 0.1。"""

    def __init__(self, base):
        super().__init__()
        # 保留 .base 这层命名，使已有 checkpoint 可直接 strict load。
        self.base = base
        self.sdpa_backend = "auto"
        for name in ("attention_scale", "ffn_scale"):
            self.register_buffer(
                name, torch.full((base.config.d_model,), 0.1), persistent=False
            )

    def set_activation_checkpointing(self, strategy):
        self.base.set_activation_checkpointing(strategy)

    def forward(self, hidden_states, attention_bias, position_ids):
        """[B, S, D] → [B, S, D]；链间可见性由 attention_bias 决定。"""
        block = self.base

        # 1. 归一化并投影 Q/K/V，再拆成多个注意力头。
        normalized = block.attn_norm(hidden_states)
        query = block.q_proj(normalized)
        key = block.k_proj(normalized)
        value = block.v_proj(normalized)
        dtype = key.dtype
        if block.q_norm is not None and block.k_norm is not None:
            query = block.q_norm(query).to(dtype)
            key = block.k_norm(key).to(dtype)
        batch_size, seq_length, width = query.shape
        num_heads = block.config.n_heads
        head_dim = width // num_heads
        kv_heads = block.config.effective_n_kv_heads
        query = query.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, kv_heads, head_dim).transpose(1, 2)

        # 2. 每链独立的位置编号进入 RoPE；全局层仍允许不同链互相注意。
        query, key = block.rotary_emb(query, key, position_ids=position_ids)
        attention_bias = block._cast_attn_bias(attention_bias, dtype)
        backends = [SDPBackend.MATH]
        if query.is_cuda and self.sdpa_backend in ("auto", "efficient"):
            backends = [SDPBackend.EFFICIENT_ATTENTION]
            if self.sdpa_backend == "auto":
                backends.append(SDPBackend.MATH)
        # 将后端选择限制在本次调用内，梯度重计算时也使用相同策略。
        with sdpa_kernel(backends):
            attended = block._scaled_dot_product_attention(
                query, key, value, attn_mask=attention_bias,
                dropout_p=block.config.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
        attended = attended.transpose(1, 2).contiguous()
        attended = block.attn_out(attended.reshape(batch_size, seq_length, width))
        hidden_states = hidden_states + block.dropout(attended * self.attention_scale)

        # 3. SwiGLU 前馈网络，再加回缩放后的残差。
        normalized = block.ff_norm(hidden_states)
        feedforward = block.ff_out(
            block.act(block.ff_proj(normalized)) * block.up_proj(normalized)
        )
        return hidden_states + block.dropout(feedforward * self.ffn_scale)


class FixedChainModel(nn.Module):
    """固定双链掩码模型；denoise 预测 token，forward 加噪并计算训练 loss。"""

    def __init__(self, config: FixedChainConfig):
        super().__init__()
        if not isinstance(config, FixedChainConfig):
            raise TypeError("FixedChainConfig is required; object302 checkpoints are incompatible")
        self.config = config
        self.variant = VARIANT
        self.options = {"residual_scale": 0.1, "segment_global": True}
        self.tokenizer = make_tokenizer(config.ablation)
        self.terminal_id = self.tokenizer.eos_token_id

        # 复用 LLaDA 的投影、归一化、RoPE 和初始化；默认 8 层 / 768 维 / 12 头。
        backbone_config = ModelConfig(
            d_model=config.d_model, n_layers=config.n_layers, n_heads=config.n_heads,
            n_kv_heads=config.n_heads, mlp_hidden_size=config.mlp_hidden_size,
            vocab_size=self.tokenizer.vocab_size,
            embedding_size=self.tokenizer.vocab_size,
            max_sequence_length=config.max_length,
            rope=True, rope_theta=500000.0, block_type="llama",
            activation_type="silu", layer_norm_type="rms", weight_tying=False,
            init_device="cpu", init_fn="mitchell", init_std=0.02,
            scale_logits=False, bias_for_layer_norm=False,
            attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0,
        )
        self.backbone = LLaDAModel(backbone_config, init_params=True)
        self.role_embeddings = nn.Embedding(
            len(CHAIN_ROLE_TO_ID), config.d_model, padding_idx=0
        )
        self.scope_embeddings = nn.Embedding(len(SCOPES), config.d_model, padding_idx=0)
        self.chain_bias = nn.Parameter(torch.zeros(2, config.n_heads))
        for embedding in (self.role_embeddings, self.scope_embeddings):
            nn.init.normal_(embedding.weight, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        self.backbone.transformer.blocks = nn.ModuleList(
            [FixedChainBlock(block) for block in self.backbone.transformer.blocks]
        )
        self.configure_sdpa_backend()

        # 受体槽位只预测残基或 EOS；dot/protd 由输入布局固定提供。
        allowed = torch.zeros(self.tokenizer.vocab_size, dtype=torch.bool)
        allowed[self.tokenizer.encode_residues(RESIDUES)] = True
        allowed[self.terminal_id] = True
        self.register_buffer("allowed_target", allowed, persistent=False)

    def denoise(
        self, input_ids, attention_mask, role_ids, scope_ids, chain_ids,
        position_ids, target_slot_mask, *, output_hidden_states=False,
    ):
        """一次去噪：只读取输入特征，不读取真实标签或真实链长。

        输入特征均为 [B, S]；返回 logits [B, S, vocab]，以及可选的最终表征。
        attention_mask 标记有效输入；target_slot_mask 标记待生成的受体区域。
        """
        self._validate_target_features(role_ids, chain_ids, position_ids, target_slot_mask)
        valid = attention_mask.bool()
        if not valid.any(1).all():
            raise ValueError("Each record needs at least one attended token")

        # 1. 输入表示：残基/语法 token + 链角色 + 序列范围。
        hidden_states = (
            self.backbone.transformer.wte(input_ids)
            + self.role_embeddings(role_ids)
            + self.scope_embeddings(scope_ids)
        )
        hidden_states = self.backbone.transformer.emb_drop(hidden_states)

        # 2. 同一次前向只构造两种 bias；各层共享 bias，不共享 Q/K/V。
        biases = self._build_attention_biases(chain_ids, position_ids, attention_mask)
        checkpoint_layers = (
            self.backbone.activation_checkpointing_strategy is not None and self.training
        )
        for layer_index, block in enumerate(self.backbone.transformer.blocks):
            attention_bias = biases[self.mask_key(layer_index)]
            if checkpoint_layers:
                hidden_states = self.backbone._activation_checkpoint_fn(
                    block, hidden_states, attention_bias, position_ids
                )
            else:
                hidden_states = block(hidden_states, attention_bias, position_ids)

        # 3. 输出每个位置的 token 分数；受体区域屏蔽非法语法 token。
        hidden_states = self.backbone.transformer.ln_f(hidden_states)
        logits = self.backbone.transformer.ff_out(hidden_states)
        forbidden = target_slot_mask[:, :, None] & ~self.allowed_target[None, None]
        logits = logits.masked_fill(forbidden, float("-inf"))
        return logits, hidden_states if output_hidden_states else None

    def forward(self, token_t=None, **batch):
        """随机遮蔽 → 去噪 → 仅对被遮蔽的监督位置计算交叉熵。"""
        if not self.training and token_t is None:
            token_t = torch.ones_like(batch["input_ids"], dtype=torch.float)
        noised, targets, selected = self.corrupt(batch, token_t)
        if not torch.equal(targets[selected], batch["labels"][selected]):
            raise ValueError("Input/label mismatch")
        features = self.feature_args(batch)
        features["input_ids"] = noised
        logits, _ = self.denoise(**features)
        token_loss = F.cross_entropy(
            logits.transpose(1, 2).float(), targets, ignore_index=-100, reduction="none"
        )
        aa_selected = selected & batch["residue_mask"]
        eos_selected = selected & batch["terminal_mask"]

        def average(mask):
            return (token_loss * mask).sum() / mask.sum().clamp_min(1)

        loss = average(selected)
        # 保留验证器使用的逐样本加和/计数；训练 loss 本身不额外加权 EOS。
        sample_metrics = torch.stack(
            ((token_loss * aa_selected).sum(1), aa_selected.sum(1),
             (token_loss * eos_selected).sum(1), eos_selected.sum(1)), dim=-1
        ).detach()
        return {
            "loss": loss,
            "aa_loss": average(aa_selected).detach(),
            "terminal_loss": average(eos_selected).detach(),
            "aa_count": aa_selected.sum().detach(),
            "terminal_count": eos_selected.sum().detach(),
            "auxiliary_loss": loss.new_zeros(()),
            "sample_metrics": sample_metrics,
        }

    def corrupt(self, batch, token_t=None):
        """复用原遮蔽采样器；synthetic/padding 由 eligible_mask 排除。"""
        noise_batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "diffusion_eligible_mask": batch["eligible_mask"],
            "diffusion_loss_mask": batch["eligible_mask"],
            "residue_mask": batch["residue_mask"],
        }
        noised, targets, selected, _ = sample_bioseq_diffusion_noise(
            noise_batch, self.tokenizer.mask_token_id, token_t=token_t
        )
        return noised, targets, selected

    @staticmethod
    def feature_args(batch):
        """显式限制模型可见特征，避免将标签或真实 EOS 位置送入网络。"""
        names = (
            "input_ids", "attention_mask", "role_ids", "scope_ids",
            "chain_ids", "position_ids", "target_slot_mask",
        )
        return {name: batch[name] for name in names}

    @staticmethod
    def mask_key(layer):
        """0-based：偶数层链内全注意力，奇数层全局注意力。"""
        return ("segment",) if layer % 2 == 0 else ("global",)

    def layer_allowed(self, layer, chain_ids, position_ids, attention_mask):
        """返回 [B, S, S] 可见性；position_ids 保留供现有调用方使用。"""
        valid = attention_mask.bool()
        if layer % 2:
            return valid[:, None, :].expand(-1, chain_ids.shape[1], -1)
        same_chain = chain_ids[:, :, None] == chain_ids[:, None, :]
        allowed = same_chain & valid[:, None, :]
        # padding query 的输出会丢弃；允许其读取有效 key，避免全屏蔽行。
        return allowed | (~valid[:, :, None] & valid[:, None, :])

    def _build_attention_biases(self, chain_ids, position_ids, attention_mask):
        """可学习的同链/异链偏置 + 不可见位置的屏蔽值。"""
        same_chain = (
            (chain_ids[:, :, None] == chain_ids[:, None, :])
            & (chain_ids[:, :, None] >= 0) & (chain_ids[:, None, :] >= 0)
        )
        base_bias = torch.where(
            same_chain[:, None], self.chain_bias[1][None, :, None, None],
            self.chain_bias[0][None, :, None, None],
        )  # [B, H, S, S]
        biases = {}
        for layer in range(min(2, self.config.n_layers)):
            key = self.mask_key(layer)
            if key == ("global",) and bool(attention_mask.bool().all()):
                biases[key] = base_bias
            else:
                allowed = self.layer_allowed(layer, chain_ids, position_ids, attention_mask)
                biases[key] = base_bias.masked_fill(
                    ~allowed[:, None], torch.finfo(base_bias.dtype).min
                )
        return biases

    @staticmethod
    def _validate_target_features(role_ids, chain_ids, position_ids, target_slot_mask):
        valid_role = torch.zeros_like(target_slot_mask)
        for role in RECEPTOR_ROLES:
            valid_role |= role_ids.eq(CHAIN_ROLE_TO_ID[role])
        invalid = ~valid_role | (chain_ids < 0) | (position_ids < 1)
        if (target_slot_mask & invalid).any():
            raise ValueError("FixedChainCollator is required for fixed-chain targets")

    def configure_sdpa_backend(self):
        """后端属于运行设置，不写入模型权重；CPU 始终使用 math。"""
        backend = os.environ.get("IMMUNE_SDPA_BACKEND", "auto")
        if backend not in ("auto", "math", "efficient"):
            raise ValueError("IMMUNE_SDPA_BACKEND must be auto, math, or efficient")
        self.sdpa_backend = backend
        for block in self.backbone.transformer.blocks:
            block.sdpa_backend = backend
        logging.getLogger(__name__).info("Fixed-chain SDPA policy=%s", backend)

    @torch.no_grad()
    def represent(self, batch):
        """对 representation 布局返回残基表征、各链均值及整体均值。"""
        _, hidden = self.denoise(**self.feature_args(batch), output_hidden_states=True)
        mask = batch["residue_mask"].bool() & batch["attention_mask"].bool()
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        chain_ids = batch["chain_ids"]
        chains = []
        for row in range(hidden.shape[0]):
            chains.append({
                int(uid): hidden[row][mask[row] & (chain_ids[row] == uid)].mean(0)
                for uid in chain_ids[row][mask[row]].unique(sorted=True)
            })
        return {
            "residue_embeddings": hidden, "residue_mask": mask,
            "chain_embeddings": chains, "pooled": pooled,
        }

    @torch.no_grad()
    def generate(self, batch, *, steps=32, temperature=0.0, seed=42, expected_chains=None):
        """多轮填充和每链首个 EOS 的解析集中在 generation.py。"""
        from .generation import generate
        return generate(
            self, batch, steps=steps, temperature=temperature, seed=seed,
            expected_chains=expected_chains,
        )

    def save_pretrained(self, directory):
        """保留原 checkpoint 文件名和参数键。"""
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "immune_config.json").write_text(
            json.dumps(asdict(self.config), indent=2) + "\n"
        )
        vocabulary = [self.tokenizer.token(i) for i in range(self.tokenizer.vocab_size)]
        (directory / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2) + "\n")
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()},
            str(directory / "model.safetensors"),
        )
        self.save_research_config(directory)

    def save_research_config(self, directory):
        settings = {
            "schema": SCHEMA, "variant": self.variant,
            "options": self.options, "base": asdict(self.config),
        }
        (Path(directory) / "research_config.json").write_text(
            json.dumps(settings, indent=2) + "\n"
        )

    @classmethod
    def from_pretrained(cls, directory):
        """严格检查配置和词表后加载已有 fixed-chain 权重。"""
        from safetensors.torch import load_file
        directory = Path(directory)
        settings = json.loads((directory / "research_config.json").read_text())
        if (
            settings.get("schema") != SCHEMA or settings.get("variant") != VARIANT
            or settings.get("options") != {"residual_scale": 0.1, "segment_global": True}
        ):
            raise ValueError("Not a fixed-chain f011 checkpoint")
        if settings["base"] != json.loads((directory / "immune_config.json").read_text()):
            raise ValueError("Checkpoint configuration mismatch")
        model = cls(FixedChainConfig(**settings["base"]))
        vocabulary = [model.tokenizer.token(i) for i in range(model.tokenizer.vocab_size)]
        if json.loads((directory / "vocabulary.json").read_text()) != vocabulary:
            raise ValueError("Checkpoint vocabulary mismatch")
        model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        return model
