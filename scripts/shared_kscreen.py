#!/usr/bin/env python3
"""Screen the shared-space dimension K (and critic) on the Gaussian lensing toy.

Tests prediction P3 (docs/shared_space.md): with statistics linear in the measured spectrum
(the exponential-family form), the excess KL should fall with K toward the 6-bin information
floor, and the best K should grow with N.

    python scripts/shared_kscreen.py --out results/shared_kscreen --Ks 2 3 4 8 16 --budgets 1000 4000
"""
import argparse
import json
from pathlib import Path

from unified_sbi.shared.lensing import LensingConfig
from unified_sbi.shared.study import StudyConfig, run_study

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--cache", default="cache/shared")
p.add_argument("--Ks", type=int, nargs="+", default=[2, 3, 4, 8, 16])
p.add_argument("--critics", nargs="+", default=["gauss"])
p.add_argument("--arms", nargs="+", default=["pklin_shared", "pk_shared"])
p.add_argument("--budgets", type=int, nargs="+", default=[1000, 4000])
p.add_argument("--seeds", type=int, nargs="+", default=[0])
p.add_argument("--field", default="gaussian")
p.add_argument("--threads", type=int, default=1)
a = p.parse_args()

lens = LensingConfig(field=a.field).to_dict()
for critic in a.critics:
    for K in a.Ks:
        cfg = StudyConfig(lensing=lens, K=K, critic=critic, budgets=tuple(a.budgets), seeds=tuple(a.seeds),
                          arms=tuple(a.arms), threads=a.threads)
        print(f"=== K={K} critic={critic}", flush=True)
        run_study(cfg, Path(a.out) / f"K{K}_{critic}", a.cache)
