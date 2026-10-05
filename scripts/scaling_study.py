#!/usr/bin/env python
"""Simulation-budget scaling study: our discovery + NPE pipeline against NPE on raw data.

Examples
--------
All cells sequentially (Colab / workstation):
    python scripts/scaling_study.py --out results/scaling

One cell per SLURM array task (see slurm/scaling_array.sbatch):
    python scripts/scaling_study.py --out results/scaling --task-id $SLURM_ARRAY_TASK_ID

Minute-long smoke test:
    python scripts/scaling_study.py --out results/smoke --quick

Cells already on disk are skipped, so a crashed or pre-empted sweep can simply be relaunched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from unified_sbi.study import DEFAULT_BUDGETS, StudyConfig, run_and_save
from unified_sbi.utils import atomic_write_json, resolve_device


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("results/scaling"))
    p.add_argument("--config", type=Path, help="JSON StudyConfig; command-line flags override it.")
    p.add_argument("--simulator", default=None, help="Preset name, e.g. rosenbrock8_hidden2d.")
    p.add_argument("--budgets", type=int, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--arms", nargs="+", default=None, help="raw xbar theta_t eta_t")
    p.add_argument("--n-discovery", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--n-post", type=int, default=None)
    p.add_argument("--map-steps", type=int, default=None)
    p.add_argument("--learn-scales", action="store_true", help="Per-observation latent scales s(x).")
    p.add_argument("--no-reuse", action="store_true",
                   help="Ablation: map-based arms train NPE only on simulations not used for discovery.")
    p.add_argument("--npe-max-epochs", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: SLURM_CPUS_PER_TASK or 4)")
    p.add_argument("--task-id", type=int, default=None, help="Run only this (seed, budget) cell.")
    p.add_argument("--list-tasks", action="store_true", help="Print the task grid and exit.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quick", action="store_true", help="Tiny budgets and training for a smoke test.")
    return p.parse_args(argv)


def build_config(args) -> StudyConfig:
    base = json.loads(args.config.read_text()) if args.config else {}
    if args.quick:
        base.update(budgets=[250, 500, 1000], seeds=[0], n_test=64, n_post=128, exact_grid=120,
                    maps=dict(max_steps=400, eval_every=25, patience_steps=200),
                    npe=dict(max_epochs=60, stop_after_epochs=10))
    flags = dict(simulator=args.simulator, budgets=args.budgets, seeds=args.seeds, arms=args.arms,
                 n_discovery=args.n_discovery, val_fraction=args.val_fraction,
                 n_test=args.n_test, n_post=args.n_post)
    base.update({k: v for k, v in flags.items() if v is not None})
    if args.no_reuse:
        base["reuse_discovery"] = False
    maps, npe = dict(base.get("maps", {})), dict(base.get("npe", {}))
    if args.map_steps is not None:
        maps["max_steps"] = args.map_steps
    if args.learn_scales:
        maps["learn_scales"] = True
    if args.npe_max_epochs is not None:
        npe["max_epochs"] = args.npe_max_epochs
    base.update(maps=maps, npe=npe, device=resolve_device(args.device))
    return StudyConfig(**base)


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)
    tasks = cfg.tasks()
    if args.list_tasks:
        for i, (s, b) in enumerate(tasks):
            print(f"{i:3d}: seed={s} N_total={b}")
        print(f"{len(tasks)} tasks -> use --array=0-{len(tasks) - 1}")
        return
    threads = args.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    torch.set_num_threads(threads)

    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out / "config.json"
    if config_path.exists() and not args.overwrite:
        saved = json.loads(config_path.read_text())
        current = json.loads(json.dumps(cfg.to_dict()))
        keys = [k for k in current if k not in ("device", "seeds", "budgets")]
        if any(saved.get(k) != current[k] for k in keys):
            sys.exit(f"{config_path} differs from this run's config; use a new --out or --overwrite.")
    else:
        atomic_write_json(config_path, cfg.to_dict())

    selected = [tasks[args.task_id]] if args.task_id is not None else tasks
    print(f"device={cfg.device} threads={threads} simulator={cfg.simulator} arms={list(cfg.arms)}")
    start = time.time()
    for seed, n_total in selected:
        run_and_save(cfg, seed, n_total, args.out, overwrite=args.overwrite)
    print(f"done in {time.time() - start:.0f}s -> {args.out}/rows")


if __name__ == "__main__":
    main()
