#!/usr/bin/env python3
"""Post-hoc diagnostics and evaluation B for saved shared-space models (any arm with an eta tower).

    python scripts/shared_posthoc.py --results results/shared_gauss --cache cache/shared [--no-frozen]

Writes posthoc.json (and test_B.npz) next to each metrics.json: J^T J and J^T Sigma^-1 J log-det ratios
against the exact Fisher (median, IQR), Sigma-hat eigenvalues, binned residual bias, residual norm,
canonical correlations / harmonics R^2, and the frozen-summary MAF evaluation (B_* keys).
"""
import argparse
import json
from pathlib import Path

from unified_sbi.shared.study import Data, StudyConfig, posthoc_cell

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--cache", default="cache/shared")
ap.add_argument("--no-frozen", action="store_true")
ap.add_argument("--arms", nargs="+")
ap.add_argument("--force", action="store_true")
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()
res = Path(a.results)
raw = json.loads((res / "config.json").read_text())
cfg = StudyConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in StudyConfig.__dataclass_fields__})
cfg.device, cfg.threads = "cpu", a.threads
data = Data(cfg, a.cache)
for mp in sorted(res.glob("seed*/N*/*/model.pt")):
    dest = mp.parent
    arm, budget, seed = dest.name, int(dest.parent.name[1:]), int(dest.parent.parent.name[4:])
    if (a.arms and arm not in a.arms) or ((dest / "posthoc.json").exists() and not a.force):
        continue
    m = posthoc_cell(cfg, data, seed, budget, arm, dest, frozen=not a.no_frozen)
    print(f"[posthoc seed {seed} N {budget} {arm}]", {k: m.get(k) for k in
          ("fisher_logdet_ratio", "fisher_sigma_logdet_ratio", "B_excess_nll", "B_excess_nll_se", "B_gap")}, flush=True)
