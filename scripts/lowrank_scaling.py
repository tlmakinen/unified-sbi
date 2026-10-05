#!/usr/bin/env python
"""Scaling study on the hidden-2D-Rosenbrock-in-8D benchmark, with the (k, m) screen.

    python scripts/lowrank_scaling.py --out results/lowrank --quick         # smoke test
    python scripts/lowrank_scaling.py --out results/lowrank                 # all cells
    python scripts/lowrank_scaling.py --out results/lowrank --task-id $SLURM_ARRAY_TASK_ID
    python scripts/lowrank_scaling.py --out results/lowrank_fixed --selection fixed --k 2 --m 2

Finished cells are skipped, so a crashed sweep can be relaunched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from unified_sbi.lowrank.study import LowRankConfig, run_and_save
from unified_sbi.utils import atomic_write_json, resolve_device


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("results/lowrank"))
    p.add_argument("--config", type=Path, help="JSON LowRankConfig; flags override it.")
    p.add_argument("--budgets", type=int, nargs="+")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--arms", nargs="+", help="raw mean summary eta structured oracle gaussian prior")
    p.add_argument("--discovery", choices=["fraction", "fixed"])
    p.add_argument("--discovery-value", type=float, help="fraction of N (e.g. 0.4) or a count (e.g. 500)")
    p.add_argument("--reuse", action="store_true", help="Map-based arms also train NPE on discovery sims.")
    p.add_argument("--selection", choices=["screen", "fixed"])
    p.add_argument("--k", type=int, help="k when --selection fixed")
    p.add_argument("--m", type=int, help="m when --selection fixed")
    p.add_argument("--k-values", type=int, nargs="+")
    p.add_argument("--m-mode", choices=["standard", "all", "identity"])
    p.add_argument("--n-test", type=int)
    p.add_argument("--n-post", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int)
    p.add_argument("--task-id", type=int)
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args(argv)


def build_config(args) -> LowRankConfig:
    base = json.loads(args.config.read_text()) if args.config else {}
    if args.quick:
        base.update(budgets=[500, 1000], seeds=[0], n_test=48, n_post=128, n_reference=256,
                    k_values=[1, 2, 3], maps=dict(max_steps=500, eval_every=25, patience_steps=250),
                    npe=dict(max_epochs=80, stop_after_epochs=10))
    flags = dict(budgets=args.budgets, seeds=args.seeds, arms=args.arms, discovery_mode=args.discovery,
                 discovery_value=args.discovery_value, selection=args.selection, fixed_k=args.k,
                 fixed_m=args.m, k_values=args.k_values, m_mode=args.m_mode, n_test=args.n_test,
                 n_post=args.n_post)
    base.update({k: v for k, v in flags.items() if v is not None})
    if args.reuse:
        base["reuse"] = True
    base["device"] = resolve_device(args.device)
    return LowRankConfig(**base)


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)
    tasks = cfg.tasks()
    if args.list_tasks:
        for i, (s, b) in enumerate(tasks):
            print(f"{i:3d}: seed={s} N_total={b}")
        print(f"{len(tasks)} tasks -> use --array=0-{len(tasks) - 1}")
        return
    torch.set_num_threads(args.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args.out.mkdir(parents=True, exist_ok=True)
    cfg_path = args.out / "config.json"
    current = json.loads(json.dumps(cfg.to_dict()))
    if cfg_path.exists() and not args.overwrite:
        saved = json.loads(cfg_path.read_text())
        if any(saved.get(k) != current[k] for k in current if k not in ("device", "seeds", "budgets")):
            sys.exit(f"{cfg_path} differs from this run's config; use a new --out or --overwrite.")
    else:
        atomic_write_json(cfg_path, current)
    selected = [tasks[args.task_id]] if args.task_id is not None else tasks
    print(f"device={cfg.device} selection={cfg.selection} grid={cfg.grid()} "
          f"discovery={cfg.discovery_mode}:{cfg.discovery_value} reuse={cfg.reuse}")
    start = time.time()
    for seed, n in selected:
        run_and_save(cfg, seed, n, args.out, args.overwrite)
    print(f"done in {time.time() - start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
