#!/usr/bin/env python3
"""Few-run conditional surrogate study on the Hasegawa-Wakatani testbed.

    python scripts/surrogate_study.py --out results/surrogate                      # full ablation
    python scripts/surrogate_study.py --out results/surrogate --design             # + design experiment (P5)
    python scripts/surrogate_study.py --out /tmp/surr_quick --quick                # ~minutes on a GPU
    python scripts/surrogate_study.py --out results/surrogate --list-tasks         # grid for SLURM arrays
    python scripts/surrogate_study.py --out results/surrogate --task-id 3          # one (seed, budget) task

See docs/plasma_surrogate_conditioning.md (sections 3, 4, 8) for the method and the arms.
"""
import argparse
import json
from pathlib import Path

from unified_sbi.surrogate.hw import HWConfig
from unified_sbi.surrogate.study import ARMS, StudyConfig, run_design, run_study

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--out", required=True)
p.add_argument("--cache", default="cache/surrogate")
p.add_argument("--config", help="JSON file of StudyConfig overrides (an 'hw' key overrides HWConfig)")
p.add_argument("--budgets", type=int, nargs="+")
p.add_argument("--seeds", type=int, nargs="+")
p.add_argument("--arms", nargs="+", choices=list(ARMS))
p.add_argument("--steps", type=int)
p.add_argument("--n", type=int, help="grid side (default 64)")
p.add_argument("--K", type=int)
p.add_argument("--device", help="cpu | cuda | auto")
p.add_argument("--threads", type=int)
p.add_argument("--design", action="store_true", help="also run the design experiment")
p.add_argument("--quick", action="store_true", help="small, fast configuration (software check)")
p.add_argument("--list-tasks", action="store_true")
p.add_argument("--task-id", type=int)
a = p.parse_args()

over = json.loads(Path(a.config).read_text()) if a.config else {}
hw = HWConfig().to_dict()
hw.update(over.pop("hw", {}))
if a.n:
    hw["n"] = a.n
if a.quick:
    hw.update(n_save=60)
    over.update(pool_size=64, n_val=8, n_test=16, budgets=(16, 48), seeds=(0,), steps=2000,
                width=128, depth=4, heads=4, eval_every=250, design_budgets=(32,))
for k in ("budgets", "seeds", "arms", "steps", "K", "device", "threads"):
    v = getattr(a, k)
    if v is not None:
        over[k] = tuple(v) if isinstance(v, list) else v
cfg = StudyConfig(hw=hw, **over)

tasks = [(s, n) for s in cfg.seeds for n in cfg.budgets]
if a.list_tasks:
    for i, (s, n) in enumerate(tasks):
        print(i, f"seed={s}", f"budget={n}")
    raise SystemExit
cells = None
if a.task_id is not None:
    s, n = tasks[a.task_id]
    cells = [(s, n, arm) for arm in cfg.arms]
data = run_study(cfg, a.out, a.cache, cells)
if a.design:
    run_design(cfg, a.out, a.cache, data=data)
