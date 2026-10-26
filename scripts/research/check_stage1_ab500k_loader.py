"""Check real subset/cache loading on CPU without modifying training outputs.

Activate pllm; python -B THIS_FILE --plan ABS --output ABS_JSON.
Checks the first two rank-zero batches with four spawned workers and compares
their rows and decoded tensors with direct reads. No model or GPU is loaded.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import time
import torch

from scripts.research.bert_mlm_data import CompleteEpochBatchSampler, write_json
from scripts.research.run_stage1_ab500k import PilotDataset, training_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(args.plan.read_text())
    subset = json.loads(Path(plan["subset_manifest"]).read_text())
    results = {}
    for arm in ("mlm", "pair_kd"):
        started = time.monotonic()
        dataset = PilotDataset(subset, plan["cache_manifest"], targets=arm == "pair_kd")
        dataset[0]  # Open files/SQLite in the parent before spawning, as profile does.
        sampler = CompleteEpochBatchSampler(len(dataset), 512, 0, 8, 42, 0, max_batches=2)
        expected_batches = list(sampler)
        loader = training_loader(dataset, sampler, plan["training"], 0, 0)
        compared = 0
        for positions, batch in zip(expected_batches, loader):
            for position, item in zip(positions, batch):
                direct = dataset[position]
                assert item[0] == direct[0] and item[1].to_dict() == direct[1].to_dict()
                if arm == "pair_kd":
                    assert item[2]["request"] == direct[2]["request"]
                    for key, value in direct[2]["tensors"].items():
                        torch.testing.assert_close(item[2]["tensors"][key], value, rtol=0, atol=0)
                else:
                    assert item[2] is None
                compared += 1
        if dataset.store is not None:
            dataset.store.close()
        assert compared == 128
        results[arm] = dict(status="passed", compared_records=compared, workers=4,
                            context=loader.multiprocessing_context.get_start_method(),
                            seconds=time.monotonic()-started)
        print(json.dumps({arm: results[arm]}), flush=True)
    write_json(args.output, dict(status="passed", scope="CPU real rows and decoded targets; no GPU training", arms=results))


if __name__ == "__main__":
    main()
