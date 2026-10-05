#!/usr/bin/env python3
"""Shear-calibration nuisance study: theta = (Omega_m, sigma8, m), m ~ N(0, m_sigma^2) (unified_sbi.shared.nuisance).

    python scripts/nuisance_m_study.py --out results/nuisance_m_gauss_cpu --budgets 250 1000 4000
    python scripts/nuisance_m_study.py --out /tmp/smoke --quick
"""
import argparse
import json
from pathlib import Path

from unified_sbi.shared.lensing import LensingConfig
from unified_sbi.shared.nuisance import ARMS, NuisanceData, run_cell
from unified_sbi.shared.study import StudyConfig

p = argparse.ArgumentParser()
p.add_argument("--out", required=True); p.add_argument("--cache", default="cache/shared")
p.add_argument("--budgets", type=int, nargs="+", default=[250, 1000, 4000])
p.add_argument("--seeds", type=int, nargs="+", default=[0])
p.add_argument("--arms", nargs="+", default=list(ARMS))
p.add_argument("--m-sigma", type=float, default=0.05)
p.add_argument("--threads", type=int, default=1)
p.add_argument("--quick", action="store_true")
a = p.parse_args()
over = dict(eval_frozen=False, threads=a.threads, budgets=tuple(a.budgets), seeds=tuple(a.seeds), arms=tuple(a.arms))
if a.quick:
    over.update(test_n=32, max_epochs=3, patience=3, posterior_samples=200)
# the pool is shared with the sigma8 study, so its budget ladder must keep max = 4000
cfg = StudyConfig(lensing=LensingConfig(params="sigma8").to_dict(), **over)
cfg.budgets = (250, 500, 1000, 2000, 4000)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "config.json").write_text(json.dumps(dict(cfg.to_dict(), m_sigma=a.m_sigma, run_budgets=a.budgets, run_arms=a.arms), indent=1))
data = NuisanceData(cfg, a.cache, m_sigma=a.m_sigma)
data.test(); data.oracle()
for s in a.seeds:
    for n in a.budgets:
        for arm in a.arms:
            try:
                run_cell(cfg, data, s, n, arm, out)
            except Exception as exc:
                print(f"FAILED seed {s} N {n} {arm}: {type(exc).__name__}: {exc}", flush=True)
