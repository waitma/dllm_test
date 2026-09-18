"""Smoke-test LLaDAEsmcFusion for residue_cond_mode in {token, feature, add}.

Uses a tiny from-scratch LLaDA decoder (d_model=128) and a real local ESMC
encoder when importable; otherwise a stub encoder that returns random features.

Run from dllm_test:
    PYTHONPATH=. python scripts/debug/smoke_protein_fusion.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Must precede huggingface_hub / transformers imports.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn as nn
import transformers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
from dllm.pipelines.llada.models.modeling_llada import (
    LLaDAModel,
    LLaDAModelLM,
    create_model_config_from_pretrained_config,
)
from dllm.pipelines.immune_llada.data import (
    BioSeqChain,
    BioSeqRecord,
    GrammarBioSeqCollator,
    GrammarTokenizer,
    HuggingFaceEsmTokenizerAdapter,
)
from examples.llada.protein_fusion_model import (
    LLaDAEsmcFusion,
    RemapCollator,
    build_remap_lookup,
    expand_llada_tokenizer_for_esmc_grammar,
)
from examples.llada.protein_pretrain_esmc import FusionTrainer, _decoder_mask_token_id

ESMC_PATH = PROJECT_ROOT / "model_weights" / "esmc" / "ESMC-300M"


class StubESMCEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=int(hidden_size))
        self._probe = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None, **_):  # noqa: ANN001
        batch, seq = input_ids.shape
        hidden = torch.randn(
            batch,
            seq,
            self.config.hidden_size,
            device=input_ids.device,
            dtype=self._probe.dtype,
        )
        if attention_mask is not None:
            hidden = hidden * attention_mask.to(hidden.dtype).unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden)


def _build_tiny_llada(vocab_size: int, mask_token_id: int, pad_token_id: int) -> LLaDAModelLM:
    llada_config = LLaDAConfig(
        d_model=128,
        n_heads=2,
        n_layers=2,
        mlp_hidden_size=256,
        activation_type="silu",
        block_type="llama",
        layer_norm_type="rms",
        rms_norm_eps=1e-5,
        rope=True,
        rope_theta=10000.0,
        alibi=False,
        include_bias=False,
        include_qkv_bias=False,
        weight_tying=False,
        input_emb_norm=False,
        scale_logits=False,
        max_sequence_length=512,
        vocab_size=int(vocab_size),
        embedding_size=int(vocab_size),
        pad_token_id=int(pad_token_id),
        mask_token_id=int(mask_token_id),
        use_cache=False,
        init_device="cpu",
    )
    model_config = create_model_config_from_pretrained_config(llada_config)
    model_config.init_device = "cpu"
    inner = LLaDAModel(model_config, init_params=True)
    return LLaDAModelLM(llada_config, model=inner)


def _fabricate_records() -> list[BioSeqRecord]:
    return [
        BioSeqRecord(
            chains=[
                BioSeqChain("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS", "antibody_heavy"),
                BioSeqChain("DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK", "antibody_light"),
            ],
            task_type="antibody",
            source="smoke_oas",
        ),
        BioSeqRecord(
            chains=[
                BioSeqChain("NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSYVGNTGELFFGEGSRLTVL", "tcr_beta"),
                BioSeqChain("KQEVTQIPAALSVPEGENLVLNCSFTDSAIYNLQWFRQDPGKGLTSLLLIQSSQREQTSGRLNASLDKSSGRSTLYIAASQPGDSATYLCAVTNQAGTALIFGKGTTLSVSP", "tcr_alpha"),
            ],
            task_type="tcr",
            source="smoke_ots",
        ),
        BioSeqRecord(
            chains=[
                BioSeqChain("QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARDRGYYYGMDVWGQGTTVTVSS", "antibody_heavy"),
                BioSeqChain("EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPLTFGGGTKVEIK", "antibody_light"),
            ],
            task_type="antibody",
            source="smoke_oas2",
        ),
    ]


def _load_encoder(hidden_size: int) -> nn.Module:
    try:
        from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import load_local_esmc_encoder

        encoder = load_local_esmc_encoder(ESMC_PATH)
        print(f"Using real ESMC encoder from {ESMC_PATH}")
        return encoder
    except Exception as exc:
        print(f"ESMC load failed ({type(exc).__name__}: {exc}); using StubESMCEncoder")
        return StubESMCEncoder(hidden_size)


def smoke_one_mode(
    mode: str,
    tok: transformers.PreTrainedTokenizer,
    gtok: GrammarTokenizer,
    lookup: torch.Tensor,
    records: list[BioSeqRecord],
    encoder_hidden: int,
) -> None:
    decoder = _build_tiny_llada(
        vocab_size=len(tok),
        mask_token_id=_decoder_mask_token_id(tok),
        pad_token_id=int(tok.pad_token_id),
    )
    encoder = _load_encoder(encoder_hidden)
    model = LLaDAEsmcFusion(
        decoder=decoder,
        encoder=encoder,
        encoder_hidden_size=encoder_hidden,
        decoder_mask_token_id=_decoder_mask_token_id(tok),
        encoder_mask_token_id=int(gtok.mask_token_id),
        residue_cond_mode=mode,
        condition_norm=True,
        freeze_encoder=True,
    )
    collator = RemapCollator(
        GrammarBioSeqCollator(tokenizer=gtok, max_sequence_length=512, max_protein_length=256),
        lookup,
    )

    out_dir = Path(tempfile.mkdtemp(prefix=f"smoke_fusion_{mode}_"))
    try:
        args = transformers.TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=1,
            max_steps=3,
            learning_rate=1e-3,
            logging_steps=1,
            report_to="none",
            remove_unused_columns=False,
            label_names=[],
            save_strategy="no",
            dataloader_num_workers=0,
        )
        trainer = FusionTrainer(
            model=model,
            processing_class=tok,
            train_dataset=records,
            args=args,
            data_collator=collator,
        )
        train_out = trainer.train()
        loss = float(train_out.training_loss)
        assert torch.isfinite(torch.tensor(loss)), f"non-finite loss for mode={mode}: {loss}"
        save_dir = out_dir / "checkpoint-final"
        trainer.save_model(str(save_dir))
        assert save_dir.exists() or any(out_dir.iterdir()), "save_model produced no files"
        print(f"PASS mode={mode} loss={loss:.4f}")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def main() -> None:
    with (ESMC_PATH / "config.json").open() as handle:
        encoder_hidden = int(json.load(handle)["d_model"])

    tok = transformers.AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        padding_side="right",
        local_files_only=True,
    )
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"mask_token": "<|mdm_mask|>"})

    base_esmc = HuggingFaceEsmTokenizerAdapter.from_pretrained(
        ESMC_PATH, local_files_only=True
    )
    gtok = GrammarTokenizer(base_esmc)
    remap, _ = expand_llada_tokenizer_for_esmc_grammar(tok, gtok)
    lookup = build_remap_lookup(remap)
    records = _fabricate_records()

    for mode in ("token", "feature", "add"):
        smoke_one_mode(mode, tok, gtok, lookup, records, encoder_hidden)
    print("ALL MODES PASS")


if __name__ == "__main__":
    main()
