from pathlib import Path

import torch

from dllm.pipelines.bioseq.ophiuchus import (
    MultiChainOphiuchusAbModel,
    OphiuchusAbInferenceCollator,
    OphiuchusAbTrainingCollator,
    compute_ophiuchus_ab_training_loss,
)
from dllm.pipelines.bioseq.ophiuchus.model import OphiuchusAbBackbone


def test_ophiuchus_training_collator_matches_airgen_batch_layout():
    collator = OphiuchusAbTrainingCollator()
    batch = collator(
        [
            {
                "chains": ["EVQLVESGGGLVQPGGSLRLSCAASG", "DIQMTQSPSSLSASVGDRVTITC"],
                "task_type": "antibody",
            }
        ]
    )
    assert batch["heavy_tokens"]["targets"].shape == (1, 150)
    assert batch["light_tokens"]["targets"].shape == (1, 128)
    assert batch["heavy_tokens"]["chain_ids"][0, 0].item() == 0
    assert batch["light_tokens"]["chain_ids"][0, 0].item() == 1
    assert batch["weights"].shape == (1, 1)


def test_ophiuchus_multichain_training_loss_smoke():
    collator = OphiuchusAbTrainingCollator()
    batch = collator(
        [
            {"chains": ["ACDE", "FGHI"], "task_type": "antibody"},
            {"chains": ["KLMN", "PQRS"], "task_type": "antibody"},
        ]
    )
    backbone = OphiuchusAbBackbone()
    backbone.init_multimer_attention()
    model = MultiChainOphiuchusAbModel(net=backbone)
    result = compute_ophiuchus_ab_training_loss(model, batch)
    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.heavy_loss)
    assert torch.isfinite(result.light_loss)


def test_ophiuchus_generate_matches_airgen_interface():
    collator = OphiuchusAbInferenceCollator()
    batch = collator(
        [
            {
                "chains": ["EVQLVESGGGLVQPGGSLRLSCAASG", "DIQMTQSPSSLSASVGDRVTITC"],
                "task_type": "antibody",
            }
        ]
    )
    backbone = OphiuchusAbBackbone()
    backbone.init_multimer_attention()
    model = MultiChainOphiuchusAbModel(net=backbone)
    model.eval()
    with torch.no_grad():
        tokens, scores = model.generate(
            batch,
            max_iter=4,
            sampling_strategy="gumbel_argmax",
        )
    assert tokens.shape == (1, 278)
    assert scores.shape == tokens.shape
    assert tokens[0, 0].item() == 0
    assert tokens[0, 150].item() == 0


def test_ophiuchus_checkpoint_can_be_loaded_when_present():
    checkpoint = Path("/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt")
    if not checkpoint.exists():
        return
    model = MultiChainOphiuchusAbModel()
    missing, unexpected = model.net.model.load_state_dict(
        torch.load(checkpoint, map_location="cpu").get("state_dict", {}),
        strict=False,
    )
    assert len(missing) == 0
    assert len(unexpected) == 0
