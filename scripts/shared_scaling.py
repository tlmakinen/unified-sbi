#!/usr/bin/env python3
"""Shared parameter/data representation scaling study on the tomographic lensing toy.

    python scripts/shared_scaling.py --out results/shared_gauss                    # Gaussian field + exact oracle
    python scripts/shared_scaling.py --out results/shared_ln --field lognormal     # non-Gaussian field
    python scripts/shared_scaling.py --out results/shared_s8 --params sigma8       # (Omega_m, sigma_8) coordinates
    python scripts/shared_scaling.py --out results/shared_gauss --list-tasks       # grid for SLURM arrays
    python scripts/shared_scaling.py --out results/shared_gauss --task-id 3        # one (seed, budget) task
    python scripts/shared_scaling.py --out /tmp/smoke --quick                      # ~1 min software check
    python scripts/shared_scaling.py --out results/shared_cat --catalogue prior_S8_L_250_N_128_Nz_512.npz

Arms: the named ones in unified_sbi.shared.study.ARMS, or parametric emulator-study arms
hybrid_{quad|nce|hyv}_b{beta}[_B{batch}], e.g. hybrid_nce_b0.1_B256.
"""
import argparse
import json
from pathlib import Path

from unified_sbi.shared.lensing import LensingConfig
from unified_sbi.shared.study import ARMS, StudyConfig, is_arm, run_study

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--out", required=True)
p.add_argument("--cache", default="cache/shared")
p.add_argument("--config", help="JSON file of StudyConfig overrides")
p.add_argument("--budgets", type=int, nargs="+")
p.add_argument("--seeds", type=int, nargs="+")
p.add_argument("--arms", nargs="+", help="named arms (%s) or hybrid_{quad|nce|hyv}_b{beta}[_B{batch}]" % ", ".join(ARMS))
p.add_argument("--field", choices=["gaussian", "lognormal"])
p.add_argument("--params", choices=["S8", "sigma8"], help="second parameter (prior box: lensing.param_box)")
p.add_argument("--noise-amp", type=float)
p.add_argument("--n", type=int, help="map side in pixels (default 64)")
p.add_argument("--K", type=int)
p.add_argument("--critic", choices=["gauss", "expfam"])
p.add_argument("--base", choices=["prior", "jeffreys"])
p.add_argument("--threads", type=int)
p.add_argument("--device", help="cpu | cuda | auto")
p.add_argument("--cnn-width", type=int)
p.add_argument("--catalogue", help="NPZ with prior_sims/prior_theta (e.g. the hybrid-statistics demo archive)")
p.add_argument("--catalogue-noise", type=float, default=0.125, help="noise amplitude for --catalogue")
p.add_argument("--quick", action="store_true")
p.add_argument("--list-tasks", action="store_true")
p.add_argument("--task-id", type=int)
a = p.parse_args()
for arm in a.arms or ():
    if not is_arm(arm):
        p.error(f"unknown arm {arm}")

over = json.loads(Path(a.config).read_text()) if a.config else {}
lens = LensingConfig().to_dict()
lens.update(over.pop("lensing", {}))
for k, v in (("field", a.field), ("params", a.params), ("noise_amp", a.noise_amp), ("n", a.n)):
    if v is not None:
        lens[k] = v
for k in ("budgets", "seeds", "arms", "K", "critic", "base", "threads", "device", "cnn_width"):
    v = getattr(a, k)
    if v is not None:
        over[k] = tuple(v) if isinstance(v, list) else v
if a.quick:
    over.setdefault("budgets", (250, 1000))
    over.update(test_n=32, max_epochs=5, patience=5, posterior_samples=200)
cfg = StudyConfig(lensing=lens, **over)

tasks = [(s, n) for s in cfg.seeds for n in cfg.budgets]
if a.list_tasks:
    for i, (s, n) in enumerate(tasks):
        print(i, f"seed={s}", f"budget={n}")
    raise SystemExit
cells = None
if a.task_id is not None:
    s, n = tasks[a.task_id]
    cells = [(s, n, arm) for arm in cfg.arms]
run_study(cfg, a.out, a.cache, cells, catalogue=a.catalogue, catalogue_noise=a.catalogue_noise)
