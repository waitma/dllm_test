import torch

from dllm.pipelines.bioseq.ophiuchus import MultiChainOphiuchusAbModel
from dllm.pipelines.bioseq.ophiuchus.model import OphiuchusAbBackbone
from dllm.pipelines.bioseq.ophiuchus.multichain import OphiuchusAbTrainConfig
from scripts.tests.bioseq.airgen_reference import AirGenMultiChainReference, AirGenTrainConfig


def _make_synthetic_batch(bsz: int, heavy_len: int, light_len: int):
    heavy = torch.randint(4, 20, (bsz, heavy_len))
    light = torch.randint(4, 20, (bsz, light_len))
    heavy[:, 0] = 0
    heavy[:, -1] = 2
    light[:, 0] = 0
    light[:, -1] = 2
    heavy_region = torch.ones_like(heavy)
    light_region = torch.ones_like(light)
    return heavy, light, heavy_region, light_region


def _build_models():
    cfg = AirGenTrainConfig(
        single_chain_ratio=0.0,
        heavy2light_loss_ratio=0.0,
        light2heavy_loss_ratio=0.0,
        joint_loss_ratio=0.5,
        independent_loss_ratio=0.5,
    )
    ref = AirGenMultiChainReference(cfg=cfg, mask_id=32)
    backbone = OphiuchusAbBackbone()
    migrated = MultiChainOphiuchusAbModel(
        net=backbone,
        train_config=OphiuchusAbTrainConfig(
            single_chain_ratio=cfg.single_chain_ratio,
            heavy2light_loss_ratio=cfg.heavy2light_loss_ratio,
            light2heavy_loss_ratio=cfg.light2heavy_loss_ratio,
            joint_loss_ratio=cfg.joint_loss_ratio,
            independent_loss_ratio=cfg.independent_loss_ratio,
        ),
    )
    return ref, migrated


def test_construct_x_t_matches_airgen_reference():
    ref, migrated = _build_models()
    heavy, light, heavy_region, light_region = _make_synthetic_batch(8, 24, 20)

    torch.manual_seed(20260613)
    ref_heavy, ref_light, ref_mask_heavy, ref_mask_light = ref.construct_x_t(
        heavy, light, heavy_region, light_region
    )
    torch.manual_seed(20260613)
    mig_heavy, mig_light, mig_mask_heavy, mig_mask_light = migrated.construct_x_t(
        heavy, light, heavy_region, light_region
    )

    assert torch.equal(ref_heavy["t"], mig_heavy["t"])
    assert torch.equal(ref_light["t"], mig_light["t"])
    assert torch.equal(ref_heavy["x_t"], mig_heavy["x_t"])
    assert torch.equal(ref_light["x_t"], mig_light["x_t"])
    assert torch.equal(ref_heavy["mask"], mig_heavy["mask"])
    assert torch.equal(ref_light["mask"], mig_light["mask"])
    assert torch.equal(ref_mask_heavy, mig_mask_heavy)
    assert torch.equal(ref_mask_light, mig_mask_light)


def test_construct_x_t_val_stage_matches_airgen_reference():
    ref, migrated = _build_models()
    heavy, light, heavy_region, light_region = _make_synthetic_batch(4, 16, 14)

    torch.manual_seed(7)
    ref_heavy, ref_light, _, _ = ref.construct_x_t(
        heavy, light, heavy_region, light_region, stage="val"
    )
    torch.manual_seed(7)
    mig_heavy, mig_light, _, _ = migrated.construct_x_t(
        heavy, light, heavy_region, light_region, stage="val"
    )

    assert torch.equal(ref_heavy["x_t"], mig_heavy["x_t"])
    assert torch.equal(ref_light["x_t"], mig_light["x_t"])


def test_decoding_mask_transition_smoke():
    backbone = OphiuchusAbBackbone()
    model = MultiChainOphiuchusAbModel(net=backbone)
    bsz, seq_len = 2, 32
    output_tokens = torch.randint(4, 20, (bsz, seq_len))
    output_tokens[:, 0] = 0
    output_tokens[:, 5:10] = model.mask_id
    output_scores = torch.randn(bsz, seq_len)
    cur_tokens = torch.randint(4, 20, (bsz, seq_len))
    cur_scores = torch.randn(bsz, seq_len)
    xt_neq_x0 = output_tokens.eq(model.mask_id)
    non_special = model.get_non_special_sym_mask(output_tokens)

    torch.manual_seed(0)
    mask1, tokens1, scores1 = model._decoding(
        output_tokens.clone(),
        output_scores.clone(),
        cur_tokens,
        cur_scores,
        "confidence-deterministic-linear",
        xt_neq_x0.clone(),
        non_special,
        t=10,
        max_step=100,
        noise=model.mask_id,
    )
    assert mask1.dtype == torch.bool
    assert tokens1.shape == output_tokens.shape
    assert scores1.shape == output_scores.shape
    assert mask1.any()
