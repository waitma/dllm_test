"""Explicit full-sequence teacher requests and verified sparse target alignment.

Activate pllm; run tests with:
  USE_TF=0 python -B -m pytest scripts/tests/immune_llada/test_structure_mlm.py -q
Requests are created before corruption. Incomplete chains are never completed
with synthetic scaffold residues; inter-chain eligibility needs explicit evidence.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from dllm.pipelines.immune_llada.data.records import BioSeqRecord, CHAIN_ROLE_TO_ID
from dllm.pipelines.immune_llada.fixed_chain.full_token_mlm import FullTokenMLMCollator
from dllm.pipelines.immune_llada.fixed_chain.structure_mlm import file_sha256
from dllm.pipelines.immune_llada.v3.layout import ordered_chains


REQUEST_SCHEMA = "immune.structure_teacher.request.v1"
CACHE_SCHEMA = "immune.structure_teacher.sparse_cache.v1"
CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_teacher_request(record, config, *, chain_indices=None, inter_evidence=None,
                         quality_status="unvalidated"):
    if isinstance(record, dict):
        record = BioSeqRecord.from_prepared_dict(record)
    if quality_status not in {"unvalidated", "validated"}:
        raise ValueError("Unknown teacher quality status")
    collator = FullTokenMLMCollator(config)
    clean = collator([record])
    rendered_chains = clean["chain_ids"][0][clean["chain_ids"][0] >= 0].unique(sorted=True)
    ordered = ordered_chains(record)
    if len(rendered_chains) != len(ordered):
        raise ValueError("Rendered chain mapping is ambiguous")
    record_to_uid = {}
    for uid, chain in zip(rendered_chains.tolist(), ordered):
        matches = [i for i, original in enumerate(record.chains) if original is chain]
        if len(matches) != 1:
            raise ValueError("Record chain identity is ambiguous")
        record_to_uid[matches[0]] = uid
    if chain_indices is None:
        chain_indices = list(range(len(record.chains)))
    if len(set(chain_indices)) != len(chain_indices) or any(
            index < 0 or index >= len(record.chains) for index in chain_indices):
        raise ValueError("Invalid requested chain indices")
    accepted, excluded = [], []
    for index in chain_indices:
        chain = record.chains[index]
        synthetic = chain.metadata.get("synthetic_residue_mask", [0] * len(chain.sequence))
        scope = chain.metadata.get("sequence_scope", "full")
        reason = None
        if len(synthetic) != len(chain.sequence) or any(flag not in (0, 1) for flag in synthetic):
            raise ValueError("Invalid synthetic provenance")
        if any(synthetic):
            reason = "synthetic_or_partial_sequence"
        elif scope not in {"full", "fv"}:
            reason = "fragment_scope"
        elif not set(chain.sequence) <= CANONICAL:
            reason = "noncanonical_residue_or_embedded_separator"
        if reason:
            excluded.append(dict(record_chain=index, reason=reason))
            continue
        uid = record_to_uid[index]
        positions = (clean["chain_ids"][0].eq(uid) & clean["residue_mask"][0]).nonzero().flatten()
        expected = torch.tensor(collator.tokenizer.encode_residues(chain.sequence))
        if not torch.equal(clean["input_ids"][0, positions], expected):
            raise ValueError("Student residues differ from teacher sequence")
        if not clean["role_ids"][0, positions].eq(CHAIN_ROLE_TO_ID[chain.role]).all():
            raise ValueError("Student chain role differs")
        accepted.append(dict(record_chain=index, student_chain=uid, role=chain.role,
                             sequence=chain.sequence, student_positions=positions.tolist()))
    inter = []
    accepted_ids = {x["record_chain"] for x in accepted}
    for item in inter_evidence or []:
        a, b = item["chains"]
        if a == b or a not in accepted_ids or b not in accepted_ids or not item.get("evidence"):
            raise ValueError("Inter-chain evidence must name two accepted chains")
        pair = sorted((a, b))
        if any(x["chains"] == pair for x in inter):
            raise ValueError("Duplicate inter-chain evidence")
        inter.append(dict(chains=pair, evidence=str(item["evidence"])))
    payload = dict(schema=REQUEST_SCHEMA, record=record.to_dict(), config=asdict(config),
                   chains=accepted, excluded=excluded, inter_evidence=inter,
                   quality_status=quality_status)
    return {**payload, "request_sha256": content_sha(payload)}


def verify_request(request):
    payload = {k: v for k, v in request.items() if k != "request_sha256"}
    if request.get("schema") != REQUEST_SCHEMA or content_sha(payload) != request.get("request_sha256"):
        raise ValueError("Teacher request content hash differs")


def load_teacher_cache(directory, request, *, allow_unvalidated=False):
    verify_request(request)
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if (metadata.get("schema") != CACHE_SCHEMA
            or metadata["request_sha256"] != request["request_sha256"]):
        raise ValueError("Teacher cache/request identity differs")
    if not allow_unvalidated and request["quality_status"] != "validated":
        raise ValueError("Teacher quality unvalidated; only explicit engineering smoke may use this cache")
    if file_sha256(directory / "targets.safetensors") != metadata["targets_sha256"]:
        raise ValueError("Teacher target file changed")
    tensors = load_file(str(directory / "targets.safetensors"))
    expected_positions = [p for chain in request["chains"] for p in chain["student_positions"]]
    if tensors["student_positions"].tolist() != expected_positions:
        raise ValueError("Teacher-to-student residue mapping differs")
    pairs = tensors["teacher_pair_indices"]
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype != torch.long
            or not len(pairs) or pairs.min() < 0 or pairs.max() >= len(expected_positions)
            or not (pairs[:, 0] < pairs[:, 1]).all()
            or len(pairs.unique(dim=0)) != len(pairs)):
        raise ValueError("Invalid cached teacher pair indices")
    logits = tensors["teacher_logits"]
    if logits.shape != (len(pairs), metadata["teacher_contract"]["distogram_bins"]) or not torch.isfinite(logits).all():
        raise ValueError("Invalid teacher distogram shape/values")
    teacher_chains = torch.tensor([c["record_chain"] for c in request["chains"] for _ in c["sequence"]])
    permitted = {tuple(x["chains"]) for x in request["inter_evidence"]}
    for a, b in torch.stack((teacher_chains[pairs[:, 0]], teacher_chains[pairs[:, 1]]), 1).unique(dim=0).tolist():
        if a != b and tuple(sorted((a, b))) not in permitted:
            raise ValueError("Cached inter pair lacks explicit evidence")
    return dict(metadata=metadata, tensors=tensors, request=request)


def batch_teacher_targets(caches, clean, device):
    """Join per-record caches; MLM-only records use None and remain in the batch."""
    if len(caches) != clean["input_ids"].shape[0]:
        raise ValueError("Cache batch size differs")
    indices, targets, contract = [], [], None
    for row, cache in enumerate(caches):
        if cache is None:
            continue
        current = cache["metadata"]["teacher_contract"]
        if contract is not None and contract != current:
            raise ValueError("Mixed teacher contracts in batch")
        contract = current
        positions = cache["tensors"]["student_positions"]
        teacher_pairs = cache["tensors"]["teacher_pair_indices"]
        endpoints = positions[teacher_pairs]
        if endpoints.max() >= clean["input_ids"].shape[1] or not clean["residue_mask"][row, endpoints].all():
            raise ValueError("KD endpoints include non-observed residues")
        tokenizer = FullTokenMLMCollator().tokenizer
        expected = torch.tensor(tokenizer.encode_residues(
            "".join(c["sequence"] for c in cache["request"]["chains"])))
        if not torch.equal(clean["input_ids"][row, positions].cpu(), expected):
            raise ValueError("Cache applied to a different sequence")
        for chain in cache["request"]["chains"]:
            if not clean["chain_ids"][row, chain["student_positions"]].eq(chain["student_chain"]).all():
                raise ValueError("Cache chain mapping differs from clean batch")
        indices.append(torch.cat((torch.full((len(endpoints), 1), row), endpoints), dim=1))
        targets.append(cache["tensors"]["teacher_logits"])
    if not indices:
        return torch.empty((0, 3), dtype=torch.long, device=device), None, None
    return torch.cat(indices).to(device), torch.cat(targets).to(device), contract
