"""Persistent ESMFold2 worker for on-demand labels, isolated from student imports.

Activate pllm; the client starts this file with the stage1_esmfold2 Python:
  python -I -B THIS_FILE --plan ABS_PLAN --device cuda:0 --parent-pid PID
Stdin/stdout carry one JSON request/response per line; diagnostics go to stderr.
"""
import argparse
from contextlib import redirect_stdout
import ctypes
import gc
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from safetensors.torch import save_file

from scripts.research import stage1_teacher as teacher
from scripts.research.stage1_online_cache import ensure_cache, request_digest


def runtime_report():
    import transformers
    from transformers import EsmFold2Model
    from transformers.models.esmfold2.protein_utils import prepare_protein_features
    return dict(transformers=transformers.__version__, torch=torch.__version__,
        model_source_sha256=teacher.sha(inspect.getfile(EsmFold2Model)),
        featurizer_source_sha256=teacher.sha(inspect.getfile(prepare_protein_features)),
        fast_compat_source_sha256=teacher.sha(teacher.__file__))


class TeacherWorker:
    def __init__(self, plan, device):
        from transformers.models.esmfold2.protein_utils import prepare_protein_features
        self.plan, self.device = plan, torch.device(device)
        torch.cuda.set_device(self.device)
        self.model, self.contract, _ = teacher.load_teacher(plan["online_teacher"]["model_dir"], device)
        expected = plan["teacher_contract"]
        self.contract.update(num_loops=expected["num_loops"], base_seed=expected["base_seed"],
            pairs_per_chain_pair=expected["pairs_per_chain_pair"],
            featurizer_source_sha256=teacher.sha(inspect.getfile(prepare_protein_features)))
        if self.contract != expected:
            raise ValueError("Online teacher differs from the frozen seed-cache contract")
        torch.cuda.empty_cache()

    def predict(self, sequences, seed):
        features = teacher.prepare_multichain_features(sequences, self.device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            logits = self.model(**teacher.trunk_arguments(features),
                num_loops=self.contract["num_loops"]).distogram_logits[0].float().cpu()
        length = sum(map(len, sequences))
        if logits.shape != (length, length, self.contract["distogram_bins"]) or not torch.isfinite(logits).all():
            raise ValueError("Invalid online teacher logits")
        symmetry = float((logits - logits.transpose(0, 1)).abs().max())
        if symmetry > 1e-5:
            raise ValueError("Online teacher distogram is not symmetric")
        identities = {k: features[k][0].cpu().tolist() for k in ("asym_id", "entity_id", "sym_id")}
        return logits, symmetry, identities

    def generate(self, request, directory):
        started = time.monotonic()
        sequences = [c["sequence"] for c in request["chains"]]
        if not sequences or sum(map(len, sequences)) > self.plan["model"]["max_length"]:
            raise ValueError("Online teacher sequence length outside the profiled bound")
        seed = (self.contract["base_seed"] + int(request["request_sha256"][:8], 16)) % (2**31)
        logits, symmetry, identities = self.predict(sequences, seed)
        pairs = teacher.sample_pairs(request, self.contract["pairs_per_chain_pair"], seed)
        if len(pairs) > self.plan["max_pairs_per_record"]:
            raise ValueError("Online target pair count exceeds the profiled bound")
        positions = torch.tensor([p for c in request["chains"] for p in c["student_positions"]])
        save_file(dict(teacher_logits=logits[pairs[:, 0], pairs[:, 1]].contiguous(),
            teacher_pair_indices=pairs.contiguous(), student_positions=positions), str(directory / "targets.safetensors"))
        metadata = dict(schema="immune.structure_teacher.sparse_cache.v1", request_sha256=request["request_sha256"],
            teacher_contract=self.contract, targets_sha256=teacher.sha(directory / "targets.safetensors"),
            seed=seed, chain_lengths=list(map(len, sequences)), pair_count=len(pairs),
            symmetry_max_abs=symmetry, repeat_max_abs=None, seconds=time.monotonic()-started,
            teacher_requires_grad=False, quality_status=request["quality_status"], **identities)
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def dispatch(self, message):
        started = time.monotonic()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        external = total - free - torch.cuda.memory_reserved(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        if message["op"] == "profile":
            # Allocation probe only; it never becomes a biological training label.
            # W has the largest canonical residue atom count. The longest allowed
            # total residue count bounds both the pair grid and atom input buffers.
            logits, _, _ = self.predict(["W" * self.plan["model"]["max_length"]], 42)
            del logits
            result = dict(status="profiled", length=self.plan["model"]["max_length"])
        elif message["op"] == "ensure":
            request = message["request"]
            request_digest(request)
            destination, created = ensure_cache(self.plan["online_teacher"]["cache_root"], request,
                lambda path: self.generate(request, path),
                lambda path: teacher.verify_existing_cache(path, request, self.contract))
            result = dict(status="generated" if created else "reused", directory=str(destination))
        else:
            raise ValueError("Unknown teacher operation")
        torch.cuda.synchronize(self.device)
        result.update(seconds=time.monotonic()-started,
            worker_peak_allocated_gib=torch.cuda.max_memory_allocated(self.device)/2**30,
            worker_peak_reserved_gib=torch.cuda.max_memory_reserved(self.device)/2**30,
            joint_peak_reserved_estimate_gib=(external + torch.cuda.max_memory_reserved(self.device))/2**30)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--verify-runtime", action="store_true")
    args = parser.parse_args()
    if args.verify_runtime:
        actual = runtime_report()
        contract = json.loads(args.plan.read_text())["teacher_contract"]
        if any(contract[k] != value for k, value in actual.items()):
            raise ValueError("Teacher runtime differs from the pinned contract")
        print(json.dumps(dict(status="passed", **actual)), flush=True)
        return
    if args.parent_pid is None:
        parser.error("--parent-pid is required for the persistent worker")
    # torchrun can kill a failed rank without allowing Python finally blocks.
    # Terminate this worker as well instead of leaking a 6.5B model on its GPU.
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
        raise OSError(ctypes.get_errno(), "Unable to set teacher parent-death signal")
    if os.getppid() != args.parent_pid:
        return
    protocol = sys.stdout
    def respond(value):
        protocol.write(json.dumps(value, allow_nan=False) + "\n")
        protocol.flush()
    try:
        with redirect_stdout(sys.stderr):
            worker = TeacherWorker(json.loads(args.plan.read_text()), args.device)
        respond(dict(status="ready", contract=worker.contract))
        for line in sys.stdin:
            message = json.loads(line)
            if message.get("op") == "close":
                break
            try:
                with redirect_stdout(sys.stderr):
                    result = worker.dispatch(message)
            except torch.cuda.OutOfMemoryError:
                traceback.print_exc(file=sys.stderr)
                result = dict(status="error", kind="cuda_oom", message="Online teacher CUDA out of memory")
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                result = dict(status="error", kind=type(exc).__name__, message=str(exc))
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            # A reply transfers GPU use back to the student. Release the teacher
            # allocator's unused blocks before acknowledging completion.
            respond(result)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        respond(dict(status="error", kind=type(exc).__name__, message=str(exc)))
        raise


if __name__ == "__main__":
    main()
