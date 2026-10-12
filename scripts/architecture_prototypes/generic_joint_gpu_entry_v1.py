"""Fixed-package entry for artificial integer GPU profiles and 100-step segments.

Activate conda pllm; check transport rejection via:
python -m pytest scripts/tests/immune_llada/test_generic_joint_gpu_engine.py -q
Only the reviewed dispatcher may launch this installed package in the existing
GPU pool. No CLI for real data, checkpoint input, weights, or arbitrary code.
"""
import sys


def main(argv=None):
    from scripts.architecture_prototypes.generic_joint_gpu_protocol_v1 import validate_installed_entry, execution_completion, sha
    work, output, profile, manifest, admission = validate_installed_entry(
        list(sys.argv[1:] if argv is None else argv))
    # Transport, package, preparation and artificial-only scope precede Torch.
    from scripts.architecture_prototypes.generic_joint_gpu_engine_v1 import run_profile, run_segment, write_json
    result = (run_profile if work.kind == 'profile' else run_segment)(work, output, manifest, admission)
    receipt = execution_completion(work, result)
    result_path = output/('GENERIC_PROFILE_RESULT.json' if work.kind == 'profile' else 'GENERIC_SEGMENT_RESULT.json')
    receipt['result_sha256'] = sha(result_path)
    write_json(output/'COMPLETE.json', receipt)
    print({'transport': work.transport, 'status': result['status'],
           'actual_optimizer_steps': result['actual_optimizer_steps']})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
