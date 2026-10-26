"""Activate pllm; python -B -m pytest THIS_FILE -q.

Check four-rank qualification, global-batch partitioning and profile identity.
"""
import json

import pytest

from scripts.research.launch_diffusion_bert_restart import production_world_size, verify_profile, verify_smoke
from scripts.research.diffusion_source_sampling import CappedSourceBatchSampler


def plan_for(world=4):
    return dict(world_size=world, model=dict(max_length=2048),
                training=dict(global_batch=512, micro_batch_views=48, complementary_views=2),
                recipe={}, sources={}, data_index_sha256="data", parent_model_sha256="bert")


def test_legacy_world_and_invalid_topology():
    plan = plan_for()
    del plan["world_size"]
    assert production_world_size(plan) == 8
    for value in (0, 1, 3, 16, True, "4"):
        plan["world_size"] = value
        with pytest.raises(ValueError):
            production_world_size(plan)
    plan["world_size"] = 4
    plan["training"]["global_batch"] = 513
    with pytest.raises(ValueError):
        production_world_size(plan)


def test_four_rank_smoke_requires_exact_topology_and_every_rank(tmp_path):
    plan = plan_for()
    plan["smoke_output"] = str(tmp_path)
    done = dict(status="passed", world_size=4, starting_step=0,
                training=plan["training"], recipe=plan["recipe"])
    provenance = {k: plan[k] for k in ("world_size", "sources", "data_index_sha256", "parent_model_sha256")}
    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value))
    write("COMPLETE.json", done)
    write("run_provenance.json", provenance)
    for rank in range(4):
        write(f"smoke_rank{rank}.json", dict(status="passed"))
    verify_smoke(plan)
    write("COMPLETE.json", {**done, "world_size": 8})
    with pytest.raises(ValueError, match="4-rank"):
        verify_smoke(plan)
    write("COMPLETE.json", done)
    write("run_provenance.json", {**provenance, "world_size": 8})
    with pytest.raises(ValueError, match="provenance"):
        verify_smoke(plan)
    write("run_provenance.json", provenance)
    (tmp_path / "smoke_rank3.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_smoke(plan)


@pytest.mark.parametrize("world,chunks", [(4, [48]*5+[16]), (8, [48,48,32])])
def test_profile_must_match_production_batch(world, chunks):
    plan = plan_for(world)
    profile = dict(status="passed", plan_sha256="plan", production_world_size=world,
                   per_rank_records=512//world, per_rank_views=1024//world,
                   micro_views=chunks, max_length=2048)
    verify_profile(plan, profile, "plan")
    for key, bad in (("production_world_size", 12-world), ("plan_sha256", "old"),
                     ("per_rank_records", 1), ("per_rank_views", 2),
                     ("micro_views", [32]), ("max_length", 1024)):
        with pytest.raises(ValueError, match="profile"):
            verify_profile(plan, {**profile, key: bad}, "plan")


def test_four_and_eight_ranks_preserve_exact_global_sampling_stream():
    def stream(world):
        streams = [list(CappedSourceBatchSampler([100, 400, 500], sampling=dict(period=1024),
            global_batch=512, rank=r, world_size=world, seed=42, start_step=1, max_steps=4))
            for r in range(world)]
        assert all(len(batch) == 512//world for batches in streams for batch in batches)
        return [[x for group in zip(*batches) for x in group] for batches in zip(*streams)]
    assert stream(4) == stream(8) == stream(1)
