"""Focused checkpoint-policy tests.

Run with::

    PYTHONPATH=. conda run -n pllm python -m pytest scripts/tests/immune_llada/test_v2_checkpoint_policy.py -q
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from torch import nn

from examples.llada import protein_pretrain_esmc as pretrain
from examples.llada.load_fusion_checkpoint import _read_fusion_policy
from examples.llada.protein_fusion_model import (
    LLaDAEsmcFusion,
    build_inverse_remap,
    load_fusion_config,
    save_fusion_config,
)
from examples.llada.protein_pretrain_esmc import (
    _validate_resume_policy,
)


CHECKPOINT = Path("/tmp/fusion-checkpoint")
FIXED_CANVAS_METADATA = {
    "fixed_heavy_encoder_length": 168,
    "fixed_light_encoder_length": 136,
    "fixed_heavy_decoder_slots": 167,
    "fixed_light_decoder_slots": 135,
    "fixed_heavy_residue_max": 166,
    "fixed_light_residue_max": 134,
}


class _StubDecoder(nn.Module):
    def __init__(self, eos_token_id: int | None) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(16, 4)
        self.config = SimpleNamespace(
            pad_token_id=1, eos_token_id=eos_token_id, bos_token_id=0, vocab_size=16
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings


def _tiny_fusion(*, native_eos_id: int | None = 1, **policy) -> LLaDAEsmcFusion:
    return LLaDAEsmcFusion(
        decoder=_StubDecoder(native_eos_id),
        encoder=nn.Linear(4, 4),
        encoder_hidden_size=4,
        decoder_mask_token_id=2,
        encoder_mask_token_id=2,
        residue_token_ids=[3, 4],
        **policy,
    )


@pytest.fixture
def fixed_config() -> dict:
    return {
        **FIXED_CANVAS_METADATA,
        "fixed_receptor_lengths": True,
        "predict_eos": True,
        "decoder_chain_eos_token_id": 7,
        "decoder_chain_eos_policy": "chain_eos",
    }


def _write_resume_config(tmp_path: Path, config: dict) -> Path:
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "fusion_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    return checkpoint


@pytest.mark.parametrize(
    ("native_eos_id", "eos_policy"), [(1, "chain_eos"), (7, "native_eos")]
)
def test_construct_save_read_fixed_policy_roundtrip(
    tmp_path: Path, native_eos_id: int, eos_policy: str
) -> None:
    model = _tiny_fusion(
        native_eos_id=native_eos_id,
        fixed_receptor_lengths=True,
        predict_eos=True,
        decoder_chain_eos_token_id=7,
    )
    for key, value in FIXED_CANVAS_METADATA.items():
        assert getattr(model.config, key) == value
    assert model.config.decoder_chain_eos_policy == eos_policy

    save_fusion_config(tmp_path, model)
    config = load_fusion_config(tmp_path)
    assert config["fusion_config_version"] == 2
    for key, value in FIXED_CANVAS_METADATA.items():
        assert config[key] == value
    policy = _read_fusion_policy(config, tmp_path)
    assert policy["fixed_receptor_lengths"] is True
    assert policy["predict_eos"] is True
    assert policy["decoder_chain_eos_token_id"] == 7
    assert policy["decoder_chain_eos_policy"] == eos_policy

    # The eval loader reconstructs with the EOS ID, without passing its policy.
    reloaded = _tiny_fusion(
        native_eos_id=native_eos_id,
        **{key: value for key, value in policy.items() if key != "decoder_chain_eos_policy"},
    ).eval()
    save_fusion_config(tmp_path / "resaved", reloaded)
    assert _read_fusion_policy(
        load_fusion_config(tmp_path / "resaved"), tmp_path / "resaved"
    ) == policy


@pytest.mark.parametrize("native_eos_id", [None, 7])
def test_constructor_preserves_explicit_eos_policy(
    tmp_path: Path, native_eos_id: int | None
) -> None:
    model = _tiny_fusion(
        native_eos_id=native_eos_id,
        fixed_receptor_lengths=True,
        predict_eos=True,
        decoder_chain_eos_token_id=7,
        decoder_chain_eos_policy="chain_eos",
    )
    assert model.config.decoder_chain_eos_policy == "chain_eos"
    save_fusion_config(tmp_path, model)
    assert _read_fusion_policy(load_fusion_config(tmp_path), tmp_path)[
        "decoder_chain_eos_policy"
    ] == "chain_eos"


def test_writer_backfills_canvas_metadata_and_uses_tokenizer_policy(tmp_path: Path) -> None:
    model = _tiny_fusion(
        native_eos_id=None,
        fixed_receptor_lengths=True,
        predict_eos=True,
        decoder_chain_eos_token_id=7,
    )
    assert model.config.decoder_chain_eos_policy is None
    for key in FIXED_CANVAS_METADATA:
        if hasattr(model.config, key):
            delattr(model.config, key)
    tokenizer = SimpleNamespace(
        _fusion_decoder_chain_eos_token_id=7,
        _fusion_decoder_chain_eos_policy="chain_eos",
    )
    save_fusion_config(tmp_path, model, tokenizer=tokenizer)
    config = load_fusion_config(tmp_path)
    for key, value in FIXED_CANVAS_METADATA.items():
        assert config[key] == value
    policy = _read_fusion_policy(config, tmp_path)
    assert policy["decoder_chain_eos_token_id"] == 7
    assert policy["decoder_chain_eos_policy"] == "chain_eos"


def test_construct_save_read_legacy_policy_roundtrip(tmp_path: Path) -> None:
    save_fusion_config(tmp_path, _tiny_fusion())
    assert _read_fusion_policy(load_fusion_config(tmp_path), tmp_path) == (
        _read_fusion_policy({}, CHECKPOINT)
    )


def test_sidecarless_checkpoint_keeps_legacy_policy_defaults() -> None:
    policy = _read_fusion_policy({}, CHECKPOINT)

    assert policy == {
        "fixed_receptor_lengths": False,
        "predict_eos": False,
        "decoder_chain_eos_token_id": None,
        "decoder_chain_eos_policy": None,
        "condition_norm": True,
        "residue_cond_mode": "add",
    }


def test_fixed_policy_requires_v2_constants_and_eos() -> None:
    with pytest.raises(ValueError, match="fixed_heavy_encoder_length"):
        _read_fusion_policy(
            {
                "fixed_receptor_lengths": True,
                "predict_eos": True,
                "decoder_chain_eos_token_id": 123,
                "decoder_chain_eos_policy": "chain_eos",
            },
            CHECKPOINT,
        )


def test_fixed_policy_rejects_legacy_eos_alias() -> None:
    with pytest.raises(ValueError, match="predict_eos"):
        _read_fusion_policy(
            {
                "fixed_receptor_lengths": True,
                "predict_eos": False,
                "decoder_chain_eos_token_id": 123,
                "decoder_chain_eos_policy": "legacy_eos_alias",
                "fixed_heavy_encoder_length": 168,
                "fixed_light_encoder_length": 136,
                "fixed_heavy_decoder_slots": 167,
                "fixed_light_decoder_slots": 135,
                "fixed_heavy_residue_max": 166,
                "fixed_light_residue_max": 134,
            },
            CHECKPOINT,
        )


def test_legacy_inverse_remap_prefers_grammar_pad_on_eos_alias() -> None:
    inverse = build_inverse_remap(
        {1: 7, 2: 7}, preferred_source_ids=(1,), size=8
    )

    assert int(inverse[7]) == 1


def test_resume_rejects_legacy_to_v2_policy_switch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "fusion_config.json").write_text(
        '{"fixed_receptor_lengths": false}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="policy mismatch"):
        _validate_resume_policy(True, str(tmp_path), fixed_receptor_lengths=True)


def test_resume_accepts_matching_legacy_policy(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "fusion_config.json").write_text(
        '{"fixed_receptor_lengths": false}\n', encoding="utf-8"
    )

    _validate_resume_policy(True, str(tmp_path), fixed_receptor_lengths=False)


@pytest.mark.parametrize("eos_policy", ["chain_eos", "native_eos"])
@pytest.mark.parametrize("auto_resume", [False, True])
def test_resume_accepts_matching_fixed_eos_policy(
    tmp_path: Path, fixed_config: dict, eos_policy: str, auto_resume: bool
) -> None:
    fixed_config["decoder_chain_eos_policy"] = eos_policy
    checkpoint = _write_resume_config(tmp_path, fixed_config)
    _validate_resume_policy(
        True if auto_resume else str(checkpoint),
        str(tmp_path),
        fixed_receptor_lengths=True,
        decoder_chain_eos_token_id=7,
        decoder_chain_eos_policy=eos_policy,
    )


@pytest.mark.parametrize(
    ("eos_id", "eos_policy", "error"),
    [
        (8, "chain_eos", "EOS id mismatch"),
        (7, "native_eos", "EOS policy mismatch"),
        (None, "chain_eos", "current tokenizer"),
        (7, None, "current tokenizer"),
        (7, "legacy_eos_alias", "current tokenizer"),
    ],
)
def test_resume_rejects_current_tokenizer_eos_mismatch(
    tmp_path: Path, fixed_config: dict, eos_id: int | None, eos_policy: str | None,
    error: str,
) -> None:
    _write_resume_config(tmp_path, fixed_config)
    with pytest.raises(ValueError, match=error):
        _validate_resume_policy(
            True,
            str(tmp_path),
            fixed_receptor_lengths=True,
            decoder_chain_eos_token_id=eos_id,
            decoder_chain_eos_policy=eos_policy,
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("decoder_chain_eos_token_id", None),
        ("decoder_chain_eos_policy", None),
        ("decoder_chain_eos_policy", "legacy_eos_alias"),
    ],
)
def test_resume_rejects_missing_or_unsafe_saved_eos(
    tmp_path: Path, fixed_config: dict, key: str, value
) -> None:
    fixed_config[key] = value
    _write_resume_config(tmp_path, fixed_config)
    with pytest.raises(ValueError, match="missing a safe decoder chain EOS policy"):
        _validate_resume_policy(
            True,
            str(tmp_path),
            fixed_receptor_lengths=True,
            decoder_chain_eos_token_id=7,
            decoder_chain_eos_policy="chain_eos",
        )


@pytest.mark.parametrize("with_sidecar", [False, True])
def test_legacy_resume_does_not_require_matching_eos_metadata(
    tmp_path: Path, with_sidecar: bool
) -> None:
    if with_sidecar:
        _write_resume_config(
            tmp_path,
            {
                "fixed_receptor_lengths": False,
                "decoder_chain_eos_token_id": 1,
                "decoder_chain_eos_policy": "legacy_eos_alias",
            },
        )
    else:
        (tmp_path / "checkpoint-10").mkdir()
    _validate_resume_policy(
        True,
        str(tmp_path),
        fixed_receptor_lengths=False,
        decoder_chain_eos_token_id=7,
        decoder_chain_eos_policy="chain_eos",
    )


@pytest.mark.parametrize(
    ("eos_id", "eos_policy", "error"),
    [(8, "chain_eos", "EOS id mismatch"), (7, "native_eos", "EOS policy mismatch")],
)
def test_training_rejects_resume_eos_mismatch_before_loading_weights(
    tmp_path: Path, fixed_config: dict, monkeypatch: pytest.MonkeyPatch,
    eos_id: int, eos_policy: str, error: str,
) -> None:
    _write_resume_config(tmp_path, fixed_config)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    model_args = SimpleNamespace(
        decoder_init="scratch", load_in_4bit=False, model_name_or_path="stub"
    )
    data_args = SimpleNamespace(
        esmc_path=str(tmp_path), dataset_args="oas", prepared_data_dir=str(tmp_path),
        train_split="train", max_length=1024, max_protein_length=1024,
    )
    training_args = SimpleNamespace(
        residue_cond_mode="add", fixed_receptor_lengths=True, train_objective="diffusion",
        relation_aux="none", dry_run=False, resume_from_checkpoint=True,
        output_dir=str(tmp_path),
    )
    parser = SimpleNamespace(
        parse_args_into_dataclasses=lambda: (model_args, data_args, training_args)
    )
    monkeypatch.setattr(pretrain.transformers, "HfArgumentParser", lambda *args: parser)
    monkeypatch.setattr(pretrain.dllm.utils, "print_args_main", lambda *args: None)
    monkeypatch.setattr(pretrain.dllm.utils, "initial_training_setup", lambda *args: None)
    monkeypatch.setattr(pretrain, "_resolve_esmc_path", lambda *args: tmp_path)

    class StubTokenizer(SimpleNamespace):
        def __len__(self) -> int:
            return 16

    tokenizer = StubTokenizer(
        _fusion_decoder_chain_eos_token_id=eos_id,
        _fusion_decoder_chain_eos_policy=eos_policy,
    )
    monkeypatch.setattr(pretrain, "_load_llada_tokenizer", lambda *args: tokenizer)
    monkeypatch.setattr(
        pretrain.HuggingFaceEsmTokenizerAdapter, "from_pretrained", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(pretrain, "GrammarTokenizer", lambda *args: None)
    monkeypatch.setattr(
        pretrain, "expand_llada_tokenizer_for_esmc_grammar", lambda *args, **kwargs: ({0: 0}, 0)
    )
    monkeypatch.setattr(pretrain, "parse_sources", lambda *args: ["oas"])
    monkeypatch.setattr(pretrain, "load_prepared_dataset", lambda *args, **kwargs: [])
    monkeypatch.setattr(pretrain, "GrammarBioSeqCollator", lambda **kwargs: None)
    load_decoder = Mock(side_effect=AssertionError("weights loaded before policy validation"))
    monkeypatch.setattr(pretrain, "_load_llada_decoder", load_decoder)

    with pytest.raises(ValueError, match=error):
        pretrain.train()
    load_decoder.assert_not_called()
