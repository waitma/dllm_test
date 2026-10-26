"""Load the pinned real ESMFold2 teacher and cache protein-only multichain logits.

Activate pllm, then use the isolated teacher Python:
  /project_c20250601/251105016/conda/envs/stage1_esmfold2/bin/python -B \
    -m scripts.research.stage1_teacher --model-dir MODEL --requests JSON --output DIR
Uses the official Transformers per-chain atom featurizer, merges chain and
entity identities explicitly, and calls trunk forward without diffusion.
No student imports: Transformers 5 and the student's Transformers 4 stay isolated.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import json
import math
import os
import errno
from pathlib import Path
import shutil
import tempfile
import time

import torch
from safetensors.torch import load_file, save_file


REPOSITORY = "biohub/ESMFold2-Fast"
REVISION = "45fe8656f5b3ef493c17fcf9abe9a2968902e712"
ATOM_FIELDS = ("ref_pos", "ref_charge", "atom_attention_mask", "ref_element",
               "ref_atom_name_chars", "ref_space_uid", "atom_to_token")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_multichain_features(sequences, device="cpu"):
    """Assemble official per-chain residue/atom features, without inter-chain bonds."""
    from transformers.models.esmfold2.protein_utils import prepare_protein_features
    if not sequences or any(not s or not set(s) <= set("ACDEFGHIKLMNPQRSTVWY") for s in sequences):
        raise ValueError("Teacher requires nonempty complete canonical protein chains")
    parts = [prepare_protein_features(sequence) for sequence in sequences]
    token_fields = ("res_type", "mol_type", "input_ids", "attention_mask", "deletion_mean")
    msa_fields = ("msa", "msa_attention_mask", "has_deletion", "deletion_value")
    result = {name: torch.cat([p[name] for p in parts], dim=1) for name in token_fields}
    result.update({name: torch.cat([p[name] for p in parts], dim=2) for name in msa_fields})
    atoms = {name: [] for name in ATOM_FIELDS}
    chain_ids, entities, symmetries, residue_indices, representatives = [], [], [], [], []
    entity_ids, copies = {}, Counter()
    token_offset, atom_offset = 0, 0
    for chain_id, (sequence, part) in enumerate(zip(sequences, parts)):
        entity = entity_ids.setdefault(sequence, len(entity_ids) + 1)
        symmetry = copies[entity]
        copies[entity] += 1
        count = int(part["atom_attention_mask"].sum())
        for name in ATOM_FIELDS:
            value = part[name][:, :count].clone()
            if name in {"atom_to_token", "ref_space_uid"}:
                value += token_offset
            atoms[name].append(value)
        representatives.append(part["distogram_atom_idx"] + atom_offset)
        chain_ids.extend([chain_id] * len(sequence))
        entities.extend([entity] * len(sequence))
        symmetries.extend([symmetry] * len(sequence))
        residue_indices.extend(range(len(sequence)))
        token_offset += len(sequence)
        atom_offset += count
    padded_atoms = math.ceil(atom_offset / 32) * 32
    for name, values in atoms.items():
        value = torch.cat(values, dim=1)
        padding = value.new_zeros((1, padded_atoms - atom_offset, *value.shape[2:]))
        result[name] = torch.cat((value, padding), dim=1)
    result.update(token_index=torch.arange(token_offset)[None],
                  residue_index=torch.tensor(residue_indices)[None],
                  asym_id=torch.tensor(chain_ids)[None], entity_id=torch.tensor(entities)[None],
                  sym_id=torch.tensor(symmetries)[None],
                  token_bonds=torch.zeros(1, token_offset, token_offset, 1),
                  distogram_atom_idx=torch.cat(representatives, dim=1))
    return {name: value.to(device) for name, value in result.items()}


def trunk_arguments(features):
    from transformers.models.esmfold2.modeling_esmfold2 import EsmFold2AtomInputs
    features = dict(features)
    features.pop("distogram_atom_idx")
    features["atom_inputs"] = EsmFold2AtomInputs(**{name: features.pop(name) for name in ATOM_FIELDS})
    return features


def sample_pairs(request, per_chain_pair, seed):
    generator = torch.Generator().manual_seed(seed)
    chains = request["chains"]
    permitted = {tuple(x["chains"]) for x in request["inter_evidence"]}
    offsets, offset = [], 0
    for chain in chains:
        offsets.append(offset)
        offset += len(chain["sequence"])
    selected = []
    for a, left in enumerate(chains):
        for b in range(a, len(chains)):
            right = chains[b]
            if a != b and tuple(sorted((left["record_chain"], right["record_chain"]))) not in permitted:
                continue
            na, nb = len(left["sequence"]), len(right["sequence"])
            if a == b:
                pairs = torch.triu_indices(na, na, offset=1).T + offsets[a]
            else:
                pairs = torch.cartesian_prod(torch.arange(na), torch.arange(nb))
                pairs += torch.tensor([offsets[a], offsets[b]])
            if len(pairs) > per_chain_pair:
                pairs = pairs[torch.randperm(len(pairs), generator=generator)[:per_chain_pair]]
            if len(pairs):
                selected.append(pairs)
    if not selected:
        raise ValueError("No valid teacher pairs")
    return torch.cat(selected)


def fast_teacher_class():
    """Honor the release Fast checkpoint's disabled MSA module.

    Transformers 5.20.2 always constructs/calls the optional MSA encoder. The
    reference esm 3.4.1.post1 release model (model.py) sets it to None when
    config.msa_encoder.enabled is false. Remove it before checkpoint accounting
    and skip its call. Keep release profile features and the nested LM per-loop
    dropout exactly as implemented upstream; top-level experimental-model flags
    must not be applied to the release model.
    """
    from transformers import EsmFold2Model

    class FastEsmFold2Model(EsmFold2Model):
        def __init__(self, config):
            if (getattr(config,"type",None) != "release"
                    or getattr(config.msa_encoder,"enabled",True) is not False):
                raise ValueError("Fast compatibility class requires release + disabled MSA encoder")
            super().__init__(config)
            self.msa_encoder = None

        def _build_msa_kwargs(self, *args, **kwargs):
            return None

    return FastEsmFold2Model


def load_teacher(model_dir, device="cuda", dtype=torch.bfloat16):
    import transformers
    from transformers import EsmFold2Model
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    model_dir = Path(model_dir).resolve()
    provenance = json.loads((model_dir / "download_provenance.json").read_text())
    if provenance["repository"] != REPOSITORY or provenance["revision"] != REVISION:
        raise ValueError("Teacher weights are not the bound revision")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    if any(key.startswith("msa_encoder.") for key in index["weight_map"]):
        raise ValueError("Expected the pinned Fast checkpoint without MSA parameters")
    # Match downloaded LFS content hashes, rather than trusting a revision label alone.
    weight_hashes = {}
    for filename in sorted(set(index["weight_map"].values())):
        value = sha(model_dir / filename)
        metadata_path = model_dir / ".cache/huggingface/download" / (filename + ".metadata")
        metadata_lines = metadata_path.read_text().splitlines()
        if metadata_lines[0] != REVISION or metadata_lines[1] != value:
            raise ValueError("Teacher shard fails pinned Hub content hash: " + filename)
        weight_hashes[filename] = value
    model, info = fast_teacher_class().from_pretrained(
        str(model_dir), local_files_only=True, dtype=dtype, output_loading_info=True,
        attn_implementation="eager")
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Teacher checkpoint did not load completely: " + str(info))
    info = {key: sorted(value) if isinstance(value,set) else value for key,value in info.items()}
    model.eval().requires_grad_(False)
    model.to(device)
    contract = dict(repository=REPOSITORY, revision=REVISION,
                    transformers=transformers.__version__, torch=torch.__version__,
                    model_source_sha256=sha(inspect.getfile(EsmFold2Model)),
                    fast_compat_source_sha256=sha(__file__),
                    fast_compat_reference="esm==3.4.1.post1 release model.py; optional MSA module",
                    msa_encoder_present=model.msa_encoder is not None,
                    lm_encoder_dropout=model.config.lm_encoder.lm_dropout,
                    per_loop_lm_dropout=model.config.lm_encoder.per_loop_lm_dropout,
                    config_sha256=sha(model_dir / "config.json"), weight_sha256=weight_hashes,
                    distogram_bins=model.config.structure_head.num_distogram_bins,
                    representative_atom="CB; CA for glycine",
                    bin_semantics="native checkpoint class order; no rebinned targets",
                    dtype=str(dtype), mode="trunk_only", parameters=sum(p.numel() for p in model.parameters()),
                    deterministic_algorithms=True,
                    all_parameters_frozen=all(not p.requires_grad for p in model.parameters()))
    return model, contract, info


def iter_requests(path, worker_index=0, worker_count=1):
    """Stream corpus envelopes; preserve compatibility with the original pilot."""
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError("Invalid teacher worker partition")
    with Path(path).open() as stream:
        requests = (json.loads(line)["request"] for line in stream if line.strip()) if Path(path).suffix == ".jsonl" else iter(json.load(stream)["requests"])
        for number, request in enumerate(requests):
            if number % worker_count != worker_index:
                continue
            payload = {k: v for k, v in request.items() if k != "request_sha256"}
            if content_sha(payload) != request["request_sha256"]:
                raise ValueError("Request hash differs")
            yield request


def verify_existing_cache(destination, request, contract):
    metadata = json.loads((destination / "metadata.json").read_text())
    if (metadata.get("schema") != "immune.structure_teacher.sparse_cache.v1"
            or metadata["request_sha256"] != request["request_sha256"]
            or metadata["teacher_contract"] != contract
            or metadata["targets_sha256"] != sha(destination / "targets.safetensors")):
        raise ValueError("Existing cache differs from request/teacher/content")
    tensors = load_file(str(destination / "targets.safetensors"))
    if (tensors["student_positions"].tolist() != [p for c in request["chains"] for p in c["student_positions"]]
            or not torch.isfinite(tensors["teacher_logits"]).all()):
        raise ValueError("Existing cache mapping/values invalid")


def run(args):
    from transformers.models.esmfold2.protein_utils import prepare_protein_features
    requests_path, output = Path(args.requests).resolve(), Path(args.output).resolve()
    worker_index, worker_count = getattr(args, "worker_index", 0), getattr(args, "worker_count", 1)
    requests = iter_requests(requests_path, worker_index, worker_count)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(json.dumps(dict(event="loading_teacher", revision=REVISION)), flush=True)
    model, contract, loading = load_teacher(args.model_dir, args.device)
    contract.update(num_loops=args.num_loops, base_seed=args.seed,
                    pairs_per_chain_pair=args.pairs_per_chain_pair,
                    featurizer_source_sha256=sha(inspect.getfile(prepare_protein_features)))
    report = dict(teacher_contract=contract, loading_info=loading, requests_sha256=sha(requests_path),
                  results=[], load_seconds=time.time() - started, worker_index=worker_index,
                  worker_count=worker_count, counts=dict(cached=0, reused=0, mlm_only=0))
    suffix = f"_worker{worker_index}" if worker_count > 1 else ""
    (output / f"teacher_loaded{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(event="teacher_loaded", parameters=contract["parameters"],
                          seconds=report["load_seconds"])), flush=True)
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for request in requests:
        if not request["chains"]:
            report["counts"]["mlm_only"] += 1
            continue
        destination = output / request["request_sha256"]
        if destination.exists():
            if not getattr(args, "resume_cache", False):
                raise FileExistsError("Use --resume-cache to verify and reuse existing targets: " + str(destination))
            verify_existing_cache(destination, request, contract)
            report["counts"]["reused"] += 1
            continue
        directory = Path(tempfile.mkdtemp(prefix=request["request_sha256"] + ".partial.", dir=output))
        sequences = [chain["sequence"] for chain in request["chains"]]
        features = prepare_multichain_features(sequences, args.device)
        seed = (args.seed + int(request["request_sha256"][:8], 16)) % (2**31)
        kwargs = trunk_arguments(features)
        def predict():
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                return model(**kwargs, num_loops=args.num_loops).distogram_logits[0].float().cpu()
        before = time.time()
        logits = predict()
        length = sum(map(len, sequences))
        if logits.shape != (length, length, contract["distogram_bins"]) or not torch.isfinite(logits).all():
            raise ValueError("Teacher output shape/values invalid")
        symmetry_error = float((logits - logits.transpose(0, 1)).abs().max())
        if symmetry_error > 1e-5:
            raise ValueError("Teacher distogram is not symmetric")
        repeat_error = None
        if args.verify_repeat:
            repeat_error = float((logits - predict()).abs().max())
            if repeat_error > 1e-5:
                raise ValueError("Seeded teacher replay differs; max_abs=" + str(repeat_error))
        pairs = sample_pairs(request, args.pairs_per_chain_pair, seed)
        selected = logits[pairs[:, 0], pairs[:, 1]]
        positions = torch.tensor([p for c in request["chains"] for p in c["student_positions"]])
        save_file(dict(teacher_logits=selected.contiguous(), teacher_pair_indices=pairs.contiguous(),
                       student_positions=positions), str(directory / "targets.safetensors"))
        metadata = dict(schema="immune.structure_teacher.sparse_cache.v1",
                        request_sha256=request["request_sha256"], teacher_contract=contract,
                        targets_sha256=sha(directory / "targets.safetensors"), seed=seed,
                        chain_lengths=list(map(len, sequences)), pair_count=len(pairs),
                        symmetry_max_abs=symmetry_error, repeat_max_abs=repeat_error,
                        seconds=time.time() - before, teacher_requires_grad=logits.requires_grad,
                        quality_status=request["quality_status"],
                        asym_id=features["asym_id"][0].cpu().tolist(),
                        entity_id=features["entity_id"][0].cpu().tolist(),
                        sym_id=features["sym_id"][0].cpu().tolist())
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        try:
            directory.rename(destination)
        except OSError as exc:
            # Identical corpus rows can land on different workers. Only reuse a
            # complete, byte-verified result; never overwrite another worker.
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not getattr(args, "resume_cache", False):
                raise
            verify_existing_cache(destination, request, contract)
            shutil.rmtree(directory)
        result = dict(request=request["request_sha256"], directory=str(destination), status="cached",
                      pairs=len(pairs), length=length, symmetry_max_abs=symmetry_error,
                      repeat_max_abs=repeat_error, seconds=metadata["seconds"])
        report["counts"]["cached"] += 1
        # JSONL jobs can contain millions of records. Keep detailed logs on disk.
        if requests_path.suffix != ".jsonl":
            report["results"].append(result)
        with (output / f"results{suffix}.jsonl").open("a") as stream:
            stream.write(json.dumps(result)+"\n")
        print(json.dumps(result), flush=True)
    report.update(status="passed", seconds=time.time() - started,
                  peak_cuda_allocated_gb=torch.cuda.max_memory_allocated()/2**30
                  if torch.device(args.device).type == "cuda" else None)
    (output / f"verification{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-loops", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs-per-chain-pair", type=int, default=1024)
    parser.add_argument("--verify-repeat", action="store_true")
    parser.add_argument("--resume-cache", action="store_true")
    parser.add_argument("--worker-index", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--worker-count", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    args = parser.parse_args()
    if args.num_loops < 1 or args.pairs_per_chain_pair < 1:
        parser.error("num-loops and pairs-per-chain-pair must be positive")
    if "LOCAL_RANK" in os.environ and args.device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        args.device = "cuda:" + os.environ["LOCAL_RANK"]
    run(args)
