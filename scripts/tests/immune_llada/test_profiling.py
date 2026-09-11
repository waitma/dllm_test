"""CPU-only profiler tests, including actual grammar/remap with spawned workers.

Run with the protenix_abtcr Python, ``-B -m pytest -p no:cacheprovider`` and
``--basetemp refactor_baseline/local_profile_20260909/pytest_tmp``. No model
weights, production prepared shard scan, CUDA allocation, or real-model training
is used. Autocast/backward regression uses a tiny CPU linear layer only.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
profiler = importlib.import_module("scripts.debug.profile_immune_llada")
GPU_UUID = "GPU-11111111-1111-1111-1111-111111111111"
OTHER_GPU_UUID = "GPU-22222222-2222-2222-2222-222222222222"


@pytest.fixture(autouse=True)
def no_cuda_initialization(monkeypatch):
    import torch

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    def forbidden(*args, **kwargs):
        pytest.fail("CPU profiler tests must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    yield
    assert not torch.cuda.is_initialized()


def snapshot(*, memory=0, util=0, apps=None, index=0, uuid=GPU_UUID):
    return dict(error=None, gpus=[dict(index=index, uuid=uuid, memory_used_mib=memory,
                                     utilization_pct=util)], compute_apps=apps or [])


def test_safe_defaults_and_real_model_configuration(tmp_path):
    cfg = profiler.parse_args(["--run"])
    assert cfg["workers"] == [0, 1, 2, 4]
    assert cfg["warmup"] == 24 and cfg["steps"] == 96
    cli = profiler.train_cli(cfg, tmp_path)
    flags = dict(zip(cli[1::2], cli[2::2]))
    assert flags["--decoder_init"] == "scratch"
    assert flags["--decoder_d_model"] == "768"
    assert flags["--decoder_n_layers"] == "8"
    assert flags["--decoder_n_heads"] == "12"
    assert flags["--decoder_mlp_hidden"] == "3072"
    assert flags["--decoder_grad_ckpt"] == "True"
    assert flags["--freeze_encoder"] == "True"
    assert flags["--save_strategy"] == "no"
    assert flags["--dataset_args"].split("+") == list(profiler.SOURCES)
    assert profiler.parse_args(["--smoke"])["steps"] == 8


@pytest.mark.parametrize("arguments", [
    ["--workers", "0", "0"], ["--steps", "1"], ["--warmup", "0"],
    ["--samples-per-source", "2"], ["--wait-seconds", "0"],
    ["--candidates-per-source", "3", "--samples-per-source", "4"],
])
def test_reject_invalid_configuration(arguments):
    with pytest.raises(SystemExit):
        profiler.parse_args(arguments)


@pytest.mark.parametrize("state", [
    dict(error="query failed", gpus=[], compute_apps=[]),
    dict(error=None, gpus=[], compute_apps=[]), snapshot(memory=1693, util=63),
    snapshot(apps=[dict(uuid=GPU_UUID, pid=3074807, name="[Not Found]")]),
    snapshot(util=float("nan")),
])
def test_gpu_gate_fails_closed(state):
    assert profiler.idle_reasons(state, 0)


def test_idle_gpu_and_missing_io_permission(tmp_path):
    assert profiler.idle_reasons(snapshot(), 0) == []
    marker = tmp_path / "IO_CLEAR"
    assert not profiler.io_is_clear(marker)
    marker.write_text("wrong permission")
    assert not profiler.io_is_clear(marker)
    marker.write_text("main-agent-cleared\n")
    assert profiler.io_is_clear(marker)


def test_gpu_query_parses_host_pid_and_unknown_utilization(monkeypatch):
    values = iter([f"0, {GPU_UUID}, A100, 1693, 81920, 63", f"{GPU_UUID}, 3074807, [Not Found], 1680"])
    monkeypatch.setattr(profiler.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=next(values)))
    state = profiler.gpu_snapshot()
    assert state["compute_apps"][0]["pid"] == 3074807
    assert profiler.idle_reasons(state, 0)
    monkeypatch.setattr(profiler.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=f"0, {GPU_UUID}, A100, 0, 81920, [N/A]"))
    assert profiler.gpu_snapshot()["error"]


@pytest.mark.parametrize("uuid", [None, "", "0", "GPU-test", GPU_UUID + ",0"])
def test_gpu_gate_rejects_invalid_uuid(uuid):
    assert profiler.idle_reasons(snapshot(uuid=uuid), 0)


def test_gpu_uuid_selection_survives_index_reordering(monkeypatch):
    state = snapshot(index=7)
    state["gpus"] += snapshot(index=0, uuid=OTHER_GPU_UUID, memory=2000)["gpus"]
    state["compute_apps"] = [dict(uuid=OTHER_GPU_UUID, pid=123)]
    assert profiler.idle_reasons(state, 0)
    assert not profiler.idle_reasons(state, gpu_uuid=GPU_UUID)
    assert profiler.gpu_card(state, gpu_uuid=GPU_UUID)["index"] == 7
    cfg = dict(gpu_index=0, gpu_uuid=GPU_UUID)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    assert not profiler.child_gate_reasons(state, cfg)
    state["gpus"].append(dict(state["gpus"][0], index=8))
    assert profiler.child_gate_reasons(state, cfg)


@pytest.mark.parametrize("visible, verified", [("0", GPU_UUID), (OTHER_GPU_UUID, GPU_UUID), (GPU_UUID, None)])
def test_child_rejects_unverified_binding_before_import(monkeypatch, tmp_path, visible, verified):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(profiler, "gpu_snapshot", snapshot)
    marker = tmp_path / "IO_CLEAR"
    marker.write_text("main-agent-cleared")
    cfg = dict(profiler.parse_args([]), gpu_uuid=verified, io_clear_file=str(marker))
    with pytest.raises(RuntimeError, match="pre-import safety gate.*verified GPU UUID"):
        profiler.run_child(cfg, tmp_path)


def test_supervisor_requires_same_uuid_streak_and_binds_child(monkeypatch, tmp_path):
    cfg = profiler.parse_args(["--run", "--wait-seconds", "30"])
    marker = tmp_path / "IO_CLEAR"
    marker.write_text("main-agent-cleared")
    cfg["io_clear_file"] = str(marker)
    observations = [snapshot(), snapshot(uuid=OTHER_GPU_UUID), snapshot(uuid=OTHER_GPU_UUID), snapshot(uuid=OTHER_GPU_UUID)]
    calls = []
    def query():
        state = observations[len(calls)]
        calls.append(state)
        return state
    child = SimpleNamespace(pid=123456, wait=lambda timeout: 0)
    launches, stopped = [], []
    def launch(command, **kwargs):
        assert len(calls) == 4  # A different UUID resets the three-observation gate.
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == OTHER_GPU_UUID
        assert kwargs["start_new_session"]
        saved = json.loads((tmp_path / "configuration.json").read_text())
        assert saved["gpu_uuid"] == OTHER_GPU_UUID
        launches.append(command)
        return child
    monkeypatch.setattr(profiler, "gpu_snapshot", query)
    monkeypatch.setattr(profiler.subprocess, "Popen", launch)
    monkeypatch.setattr(profiler, "stop_owned_group", lambda owned: stopped.append(owned))
    monkeypatch.setattr(profiler.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(profiler.signal, "signal", lambda *args: None)
    monkeypatch.setattr(profiler.signal, "setitimer", lambda *args: None)
    assert profiler.supervise(cfg, tmp_path) == 0
    assert len(launches) == 1 and stopped == [child]
    assert json.loads((tmp_path / "status.json").read_text())["gpu_uuid"] == OTHER_GPU_UUID


def test_context_snapshot_label_and_uuid_attribution(monkeypatch, tmp_path):
    state = snapshot(index=7, apps=[dict(uuid=GPU_UUID, pid=123), dict(uuid=OTHER_GPU_UUID, pid=456)])
    state["gpus"] += snapshot(uuid=OTHER_GPU_UUID)["gpus"]
    monkeypatch.setattr(profiler, "gpu_snapshot", lambda: state)
    monitor = profiler.Monitor(tmp_path, 2, GPU_UUID, tmp_path / "absent")
    monitor.establish_own_gpu_processes()
    assert monitor.allowed_pids == {123} and monitor.baseline_ready
    assert json.loads((tmp_path / "gpu_after_context_init.json").read_text()) == state
    assert not (tmp_path / "gpu_after_model_load.json").exists()


@pytest.mark.parametrize("case", ["reindexed", "missing", "foreign"])
def test_monitor_tracks_uuid_not_original_index(monkeypatch, tmp_path, case):
    import psutil

    marker = tmp_path / "IO_CLEAR"
    marker.write_text("main-agent-cleared")
    monitor = profiler.Monitor(tmp_path, 2, GPU_UUID, marker)
    monitor.allowed_pids = {123}
    monitor.baseline_ready = True
    state = snapshot(index=7, apps=[dict(uuid=GPU_UUID, pid=123)])
    state["gpus"] += snapshot(uuid=OTHER_GPU_UUID)["gpus"]
    if case == "missing":
        state["gpus"] = state["gpus"][1:]
    elif case == "foreign":
        state["compute_apps"].append(dict(uuid=GPU_UUID, pid=456))
    process = SimpleNamespace(pid=123, children=lambda recursive: [], oneshot=nullcontext,
                              cpu_times=lambda: SimpleNamespace(user=0, system=0),
                              memory_info=lambda: SimpleNamespace(rss=0))
    monkeypatch.setattr(psutil, "Process", lambda: process)
    monkeypatch.setattr(psutil, "cpu_percent", lambda: 0)
    monkeypatch.setattr(profiler, "gpu_snapshot", lambda: state)
    monkeypatch.setattr(monitor.stop, "wait", lambda timeout: monitor.stop.set())
    monitor.run()  # One synchronous, entirely mocked resource observation.
    assert monitor.samples == 1
    if case == "reindexed":
        assert monitor.problem is None
    elif case == "missing":
        assert "not uniquely visible" in monitor.problem
    else:
        assert "foreign GPU process" in monitor.problem


def test_semantic_sampling_and_plan_cover_all_sources_lengths():
    from dllm.pipelines.immune_llada.data import BioSeqChain, BioSeqRecord
    parts = [[BioSeqRecord([BioSeqChain("A" * (5 + (i * 19) % 130), "peptide")], "generic", source)
              for i in range(150)] for source in profiler.SOURCES]
    selected, report = profiler.select_samples(parts, profiler.SOURCES, 17, 100, 18)
    again, _ = profiler.select_samples(parts, profiler.SOURCES, 17, 100, 18)
    assert selected == again
    for source, rows in selected.items():
        lengths = [x["residues"] for x in rows]
        assert lengths == sorted(lengths) and len(set(lengths)) > 3
        assert len({x["index"] for x in rows}) == 18
        assert report[source]["candidate_count"] == 100
        assert max(x["local_index"] for x in rows) > 100
    plan = profiler.make_plan(selected, 2, 24, 96, 19)
    assert plan == profiler.make_plan(selected, 2, 24, 96, 19)
    for phase in ("warmup", "measured"):
        rows = [x for x in plan if x["phase"] == phase]
        assert {x["group"] for x in rows} == set(profiler.SOURCES) | {"mixed"}
        for source in profiler.SOURCES:
            assert {x["length_bucket"] for x in rows if x["group"] == source} == {"short", "middle", "long"}
    for row in plan:
        assert len(row["indices"]) == 2


def test_throughput_is_ratio_of_totals_not_mean_of_rates():
    rows = [dict(step_time_s=1, next_wait_s=.1, collate_s=.2, samples=2, residues=100),
            dict(step_time_s=3, next_wait_s=.3, collate_s=.4, samples=2, residues=300)]
    result = profiler.aggregate(rows)
    assert result["samples_per_sec"] == 1
    assert result["residues_per_sec"] == 100
    assert result["exposed_next_fraction"] == pytest.approx(.1)


def test_atomic_json_and_nan_rejection(tmp_path):
    path = tmp_path / "status.json"
    profiler.write_json(path, dict(status="blocked"))
    assert json.loads(path.read_text())["status"] == "blocked"
    assert not path.with_suffix(".json.tmp").exists()
    with pytest.raises(ValueError):
        profiler.write_json(path, dict(value=float("nan")))


def test_supervisor_busy_never_spawns_child(monkeypatch, tmp_path):
    cfg = profiler.parse_args(["--run", "--wait-seconds", "1"])
    cfg["io_clear_file"] = str(tmp_path / "absent")
    monkeypatch.setattr(profiler, "gpu_snapshot", lambda: snapshot(memory=1693, util=63))
    monkeypatch.setattr(profiler.subprocess, "Popen", lambda *a, **k: pytest.fail("busy GPU must not spawn child"))
    monkeypatch.setattr(profiler.signal, "signal", lambda *a: None)
    monkeypatch.setattr(profiler.signal, "setitimer", lambda *a: None)
    assert profiler.supervise(cfg, tmp_path) == 3
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "blocked"
    assert not (tmp_path / "child.pid").exists()


def test_owned_group_timeout_escalates_only_own_pid(monkeypatch):
    signals = []
    monkeypatch.setattr(profiler.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    class Child:
        pid = 123456
        waits = 0
        def poll(self):
            return None
        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("owned test child", timeout)
            return -9
    profiler.stop_owned_group(Child())
    assert signals == [(123456, signal.SIGTERM), (123456, signal.SIGKILL)]


def test_child_gate_precedes_any_training_import(monkeypatch, tmp_path):
    monkeypatch.setattr(profiler, "gpu_snapshot", lambda: snapshot(memory=1693))
    cfg = profiler.parse_args([])
    with pytest.raises(RuntimeError, match="pre-import safety gate"):
        profiler.run_child(cfg, tmp_path)


@pytest.mark.parametrize("fail_forward", [False, True])
def test_compute_loss_uses_accelerator_autocast_and_unwinds(fail_forward):
    import torch
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True, mixed_precision="bf16")
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    inputs = torch.ones(2, 4)
    labels = torch.tensor([0, 1])
    events = []
    @contextmanager
    def loss_context():
        events.append("loss_context_enter")
        try:
            yield
        finally:
            events.append("loss_context_exit")
    def compute_loss(actual_model, batch):
        assert events == ["loss_context_enter"]
        assert torch.is_autocast_enabled("cpu")
        logits = actual_model(batch)
        assert logits.dtype == torch.bfloat16
        if fail_forward:
            raise ValueError("synthetic forward failure")
        return torch.nn.functional.cross_entropy(logits, labels)
    def backward_hook(gradient):
        assert not torch.is_autocast_enabled("cpu")
        assert events == ["loss_context_enter", "loss_context_exit"]
        return gradient
    model.weight.register_hook(backward_hook)
    trainer = SimpleNamespace(accelerator=accelerator, compute_loss_context_manager=loss_context,
                              compute_loss=compute_loss)
    if fail_forward:
        with pytest.raises(ValueError, match="synthetic forward failure"):
            profiler.compute_profile_loss(trainer, model, inputs)
    else:
        before = model.weight.detach().clone()
        loss = profiler.compute_profile_loss(trainer, model, inputs)
        assert loss.dtype == torch.float32 and torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(model.weight.grad).all()
        optimizer.step()
        assert not torch.equal(before, model.weight)
    assert not torch.is_autocast_enabled("cpu")
    assert events == ["loss_context_enter", "loss_context_exit"]


def test_timed_collator_digest_covers_every_model_tensor():
    import torch
    from dllm.pipelines.immune_llada.data import BioSeqChain, BioSeqRecord, GrammarBioSeqCollator, GrammarTokenizer

    records = [BioSeqRecord([BioSeqChain("ACDEF", "antibody_heavy"), BioSeqChain("LAG", "antibody_light")],
                            "antibody", "oas_paired")]
    collator = profiler.TimedCollator(GrammarBioSeqCollator(GrammarTokenizer()))
    batch = collator(records)
    baseline = batch["_profile"]["tensor_sha256"]
    assert baseline == profiler.batch_tensor_digest(batch)
    tested = set()
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor) or not value.numel():
            continue
        changed = value.clone()
        flat = changed.reshape(-1)
        flat[0] = not bool(flat[0]) if value.dtype == torch.bool else int(flat[0]) + 1
        assert profiler.batch_tensor_digest(dict(batch, **{key: changed})) != baseline, key
        tested.add(key)
    assert {"attention_mask", "encoder_attention_mask", "encoder_chain_mask", "chain_ids", "position_ids_inner"} <= tested
    reordered = dict(reversed(list(batch.items())))
    reordered["sources"] = ["metadata is not part of the tensor digest"]
    assert profiler.batch_tensor_digest(reordered) == baseline


def test_batch_digest_covers_schema_and_supports_bfloat16():
    import torch

    value = torch.zeros(2, 2, dtype=torch.int32)
    baseline = profiler.batch_tensor_digest({"tensor": value})
    assert baseline != profiler.batch_tensor_digest({"renamed": value})
    assert baseline != profiler.batch_tensor_digest({"tensor": value.reshape(4)})
    assert baseline != profiler.batch_tensor_digest({"tensor": value.view(torch.float32)})
    value = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).T
    assert not value.is_contiguous()
    assert profiler.batch_tensor_digest({"tensor": value}) == profiler.batch_tensor_digest({"tensor": value.contiguous()})
    assert profiler.batch_tensor_digest({"tensor": torch.tensor(1., dtype=torch.bfloat16)}) != profiler.batch_tensor_digest({"tensor": torch.tensor(2., dtype=torch.bfloat16)})


@pytest.fixture
def short_socket_tmp(tmp_path, monkeypatch):
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    alias = f"/proc/{os.getpid()}/fd/{directory_fd}"
    monkeypatch.setenv("TMPDIR", alias)
    monkeypatch.setattr(tempfile, "tempdir", alias)
    try:
        yield alias
    finally:
        os.close(directory_fd)


def test_real_semantic_prepared_collator_spawned_workers(tmp_path, monkeypatch, short_socket_tmp):
    """Tiny fixture only; actual load/grammar/remap/reconstruction in every worker."""
    import torch
    from torch.utils.data import DataLoader
    from dllm.pipelines.immune_llada.data import (
        BioSeqChain, BioSeqRecord, GrammarBioSeqCollator, GrammarTokenizer, load_prepared_dataset,
    )
    from dllm.pipelines.immune_llada.data.preprocessing.validators import SCHEMA_VERSION
    from examples.llada.protein_fusion_model import RemapCollator, build_remap_lookup

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    records = [BioSeqRecord([BioSeqChain("ACD" * (i + 1), "antibody_heavy"),
                            BioSeqChain("LAGV" * (i + 1), "antibody_light")], "antibody", "oas_paired")
               for i in range(4)]
    shard = tmp_path / "oas.jsonl"
    shard.write_text("".join(json.dumps(dict(record.to_dict(), schema_version=SCHEMA_VERSION)) + "\n" for record in records))
    manifest = dict(schema_version=SCHEMA_VERSION, format="jsonl", splits={"train": {"shards": [dict(path="oas.jsonl", source="oas", records=4)]}})
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(manifest))
    dataset = load_prepared_dataset(tmp_path)
    gtok = GrammarTokenizer()
    base = GrammarBioSeqCollator(gtok, max_sequence_length=128)
    # A deterministic nonidentity lookup exercises the production RemapCollator;
    # this test does not claim the local LLaDA tokenizer/model was loaded.
    remap = RemapCollator(base, build_remap_lookup({i: i + 100 for i in range(gtok.vocab_size)}))
    collator = profiler.TimedCollator(remap)
    calls = []
    original = base.renderer.encode
    with monkeypatch.context() as local:
        local.setattr(base.renderer, "encode", lambda record: (calls.append(record), original(record))[1])
        collator([dataset[0], dataset[3]])
        collator([dataset[0], dataset[3]])
    assert len(calls) == 4  # No reuse of pretokenized/model-ready batches.
    baseline = None
    for workers in (0, 1, 2, 4):
        kwargs = dict(num_workers=workers, collate_fn=collator, batch_sampler=[[0, 3], [1, 2]],
                      generator=torch.Generator().manual_seed(17), worker_init_fn=profiler.seed_worker)
        if workers:
            kwargs.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2, timeout=45)
        loader = DataLoader(dataset, **kwargs)
        iterator = iter(loader)
        try:
            batches = list(iterator)
        finally:
            if workers:
                iterator._shutdown_workers()
        hashes = [x["_profile"]["tensor_sha256"] for x in batches]
        if baseline is None:
            baseline = hashes
        assert hashes == baseline
        for batch in batches:
            assert batch["encoder_input_ids"].ndim == 3
            assert batch["encoder_chain_mask"].any()
            assert batch["_profile"]["padding_tokens"] > 0
            assert batch["_profile"]["residues"] > 0
            assert batch["input_ids"].device.type == "cpu"
    assert not torch.cuda.is_initialized()


def test_local_tokenizers_remap_and_diffusion_encoder_corruption(monkeypatch):
    import torch
    from dllm.pipelines.immune_llada.data import (
        BioSeqChain, BioSeqRecord, GrammarBioSeqCollator, GrammarTokenizer,
        HuggingFaceEsmTokenizerAdapter,
    )
    from examples.llada.protein_pretrain_esmc import _load_llada_tokenizer, _decoder_mask_token_id
    from examples.llada.protein_fusion_model import (
        RemapCollator, build_remap_lookup, expand_llada_tokenizer_for_esmc_grammar,
        sample_bioseq_diffusion_noise, apply_decoder_corruption_to_encoder,
    )
    if not Path(profiler.SNAPSHOT).is_dir():
        pytest.skip("local LLaDA snapshot unavailable")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    tok = _load_llada_tokenizer(profiler.SNAPSHOT)
    gtok = GrammarTokenizer(HuggingFaceEsmTokenizerAdapter.from_pretrained(ROOT / "model_weights/esmc/ESMC-300M", local_files_only=True))
    remap, _ = expand_llada_tokenizer_for_esmc_grammar(tok, gtok)
    collator = profiler.TimedCollator(RemapCollator(GrammarBioSeqCollator(gtok), build_remap_lookup(remap)))
    records = [
        BioSeqRecord([BioSeqChain("ACDEFGHIKLMNPQRSTVWY", "antibody_heavy"), BioSeqChain("LAGVSERT", "antibody_light")], "antibody", "oas_paired"),
        BioSeqRecord([BioSeqChain("CASSLGQETQYF", "tcr_beta"), BioSeqChain("GILGFVFTL", "peptide")], "tcr", "trait"),
    ]
    batch = collator(records)
    assert batch["_profile"]["padding_tokens"] > 0
    tokens = tok.convert_ids_to_tokens(batch["input_ids"][0].tolist())
    assert "<prots>" in tokens and "<protd>" in tokens and "<ab>" in tokens
    assert "<chainsep>" in tokens
    mask_id = _decoder_mask_token_id(tok)
    torch.manual_seed(19)
    first = sample_bioseq_diffusion_noise(batch=batch, mask_token_id=mask_id)
    torch.manual_seed(19)
    second = sample_bioseq_diffusion_noise(batch=batch, mask_token_id=mask_id)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    noised, labels, corrupted, _ = first
    assert corrupted.any()
    assert not (corrupted & batch["fixed_context_mask"]).any()
    # Production diffusion includes generated grammar tokens, not just residues.
    assert not (corrupted & ~batch["diffusion_eligible_mask"]).any()
    assert not (corrupted & ~batch["attention_mask"]).any()
    assert (corrupted & batch["structure_token_mask"]).any()
    assert (noised[corrupted] == mask_id).all()
    encoder = apply_decoder_corruption_to_encoder(batch=batch, corruption_mask=corrupted, mask_token_id=gtok.mask_token_id)
    assert (encoder == gtok.mask_token_id).sum() == (corrupted & batch["residue_mask"]).sum()
    assert not torch.cuda.is_initialized()


def test_real_training_argument_parser_cpu_only(tmp_path):
    import torch
    from transformers import HfArgumentParser
    from examples.llada.protein_pretrain_esmc import ModelArguments, DataArguments, TrainingArguments
    cli = profiler.train_cli(profiler.parse_args([]), tmp_path)[1:]
    # Parse the actual production dataclasses while explicitly avoiding CUDA setup.
    cli[cli.index("--bf16") + 1] = "False"
    cli += ["--use_cpu", "True"]
    model, data, training = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)).parse_args_into_dataclasses(args=cli)
    assert model.decoder_mlp_hidden == 3072 and model.decoder_d_model == 768
    assert data.dataset_args.split("+") == list(profiler.SOURCES)
    assert training.freeze_encoder and str(training.device) == "cpu"
    assert not torch.cuda.is_initialized()
