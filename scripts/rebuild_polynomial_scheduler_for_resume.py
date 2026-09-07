#!/usr/bin/env python3
"""Rebuild HF polynomial scheduler.pt for an extended max_steps resume.

Matches AirGen: linear polynomial (power=1) from base_lr to lr_end, then hold.
Do not load the original cosine scheduler.pt — its last_lr is 0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers.optimization import get_polynomial_decay_schedule_with_warmup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--new-max-steps", type=int, default=1_000_000)
    parser.add_argument("--last-epoch", type=int, default=50_000)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--base-lr", type=float, default=4e-5)
    parser.add_argument("--lr-end", type=float, default=1e-5)
    parser.add_argument("--power", type=float, default=1.0)
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    sched_path = ckpt / "scheduler.pt"
    old = torch.load(sched_path, map_location="cpu")
    print(
        "[scheduler] old last_epoch",
        old.get("last_epoch"),
        "last_lr",
        old.get("_last_lr"),
    )

    dummy = torch.nn.Parameter(torch.zeros(1))
    dummy2 = torch.nn.Parameter(torch.zeros(1))
    opt = AdamW(
        [
            {"params": [dummy], "lr": args.base_lr},
            {"params": [dummy2], "lr": args.base_lr},
        ]
    )
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]
    sched = get_polynomial_decay_schedule_with_warmup(
        opt,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.new_max_steps,
        lr_end=args.lr_end,
        power=args.power,
        last_epoch=args.last_epoch - 1,
    )
    torch.save(sched.state_dict(), sched_path)
    print(
        "[scheduler] new polynomial last_epoch",
        sched.last_epoch,
        "last_lr",
        sched.get_last_lr(),
        "base_lr",
        args.base_lr,
        "lr_end",
        args.lr_end,
        "warmup",
        args.warmup_steps,
        "max_steps",
        args.new_max_steps,
    )

    state_path = ckpt / "trainer_state.json"
    state = json.loads(state_path.read_text())
    state["max_steps"] = args.new_max_steps
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print("[trainer_state] global_step", state.get("global_step"), "max_steps", state["max_steps"])


if __name__ == "__main__":
    main()
