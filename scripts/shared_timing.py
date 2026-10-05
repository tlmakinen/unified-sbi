#!/usr/bin/env python3
"""ms per training step (batch 64 unless the arm sets B, one CPU thread) and parameter counts.

    python scripts/shared_timing.py --arms hybrid_maf hybrid_rect hybrid_shared hybrid_nce_b0.1 hybrid_hyv_b0.1
"""
import argparse
import json
import time

import numpy as np
import torch

from unified_sbi.shared.lensing import HI, LO
from unified_sbi.shared.study import Pipeline, StudyConfig, arm_spec

ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", required=True)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--out")
a = ap.parse_args()
torch.set_num_threads(1)
torch.set_flush_denormal(True)
cfg = StudyConfig()
rows = {}
for arm in a.arms:
    torch.manual_seed(0)
    bs = arm_spec(arm).get("batch", cfg.batch_size)
    m = Pipeline(arm, cfg, a.n, np.full((4, 1, 1), 0.01, np.float32))
    x = 0.01 * torch.randn(bs, 4, a.n, a.n)
    th = torch.as_tensor(LO + (HI - LO) * np.random.default_rng(0).random((bs, 2)), dtype=torch.float32)
    m.fit_preprocessing(x)
    m.train()
    if getattr(m.head, "emu", None) is not None:
        m.head.beta = arm_spec(arm)["beta"]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    ts = []
    for i in range(a.steps + 5):
        t0 = time.perf_counter()
        loss = m.loss(th, x).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        ts.append(time.perf_counter() - t0)
    ms = 1000 * np.median(ts[5:])
    rows[arm] = dict(ms_per_step=float(ms), batch=bs, n_params=int(sum(p.numel() for p in m.parameters())))
    print(f"{arm:28s} {ms:7.1f} ms/step  B={bs}  params={rows[arm]['n_params']}", flush=True)
if a.out:
    open(a.out, "w").write(json.dumps(rows, indent=1))
