#!/usr/bin/env python3
"""Tables for the embedding-objective study (stop-gradient emulator, nce, hyv, gmi; Gaussian toy).

    python scripts/shared_emu_report.py --results results/toy_ref results/toy_emu --out results/toy_emu/report.md

Reads metrics.json (+ posthoc.json), history.json, test.npz (evaluation A) and test_B.npz
(evaluation B: MAF on the frozen t) from every seed*/N*/arm cell in the given directories.
Paired differences are per test observation against each ``--baselines`` arm, same seed and N,
under the same evaluation (A vs A, B vs B).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", nargs="+", required=True)
ap.add_argument("--baselines", nargs="+", default=["hybrid_quad_b0", "hybrid_shared"])
ap.add_argument("--K", type=int, default=4)
ap.add_argument("--alarm-tol", type=float, default=0.05, help="tolerance on the coiling alarm threshold (nats)")
ap.add_argument("--out")
a = ap.parse_args()

rows, logq, logqB, traces = [], {}, {}, {}
for r in a.results:
    for mj in Path(r).glob("seed*/N*/*/metrics.json"):
        m = json.loads(mj.read_text())
        ph = mj.parent / "posthoc.json"
        if ph.exists():
            for k, v in json.loads(ph.read_text()).items():
                m.setdefault(k, v)
        m["dir"] = r
        rows.append(m)
        key = (m["arm"], m["seed"], m["budget"])
        tz, tb = mj.parent / "test.npz", mj.parent / "test_B.npz"
        if tz.exists() and "log_q" in np.load(tz).files:
            logq[key] = np.load(tz)["log_q"].astype(np.float64)
        if tb.exists():
            logqB[key] = np.load(tb)["log_q"].astype(np.float64)
        hj = mj.parent / "history.json"
        if hj.exists():
            h = json.loads(hj.read_text()).get("main", [])
            tr = [(e[0], e[3]) for e in h if len(e) > 3]
            if tr:
                traces[key] = (tr, m.get("best_epoch"))
df = pd.DataFrame(rows).sort_values(["arm", "budget"])
out = []


def table(col, fmt="{:.3f}", title=None, data=None):
    d = df if data is None else data
    if col not in d or d[col].isna().all():
        return
    p = d.pivot_table(index="arm", columns="budget", values=col, aggfunc="first")
    out.append(f"\n### {title or col}\n")
    out.append(p.map(lambda v: "" if (not isinstance(v, str) and pd.isna(v)) else fmt.format(v)).to_markdown())


def fmt_list(col, f="{:.2f}"):
    if col in df:
        df[col + "_s"] = df[col].map(lambda v: " ".join(f.format(x) for x in v) if isinstance(v, list) else "")


def paired(store, base, title):
    pr = []
    for (arm, s, n), lq in store.items():
        b = store.get((base, s, n))
        if b is None or arm == base:
            continue
        d = b - lq                      # > 0: arm worse than baseline (lower log q at the truth)
        pr.append(dict(arm=arm, budget=n, s=f"{d.mean():+.3f}±{d.std(ddof=1) / np.sqrt(len(d)):.3f}"))
    if pr:
        P = pd.DataFrame(pr).pivot_table(index="arm", columns="budget", values="s", aggfunc="first")
        out.append(f"\n### {title} (arm − {base}, nats, = difference in held-out −log q; ± paired SE; >0 = worse)\n")
        out.append(P.fillna("").to_markdown())


out.append("## Evaluation A — the arm's own quadrature-normalised posterior")
table("excess_nll", title="Excess KL, evaluation A")
if "excess_nll" not in df:
    table("nll", title="Held-out NLL, evaluation A")
for b in a.baselines:
    paired(logq, b, "Paired excess-KL difference, evaluation A")
table("joint_cov_mae", title="Joint HPD coverage MAE, evaluation A")

out.append("\n## Evaluation B — MAF on the frozen, standardised t(x)")
table("B_excess_nll", title="Excess KL, evaluation B")
if "B_excess_nll" not in df:
    table("B_nll", title="Held-out NLL, evaluation B")
for b in a.baselines:
    paired(logqB, b, "Paired excess-KL difference, evaluation B")
table("B_joint_cov_mae", title="Joint HPD coverage MAE, evaluation B")
table("B_gap", "{:+.3f}", "Frozen-MAF train − validation NLL gap (fixed-noise training maps; very negative = t overfit)")

out.append("\n## Geometry and emulator calibration (test set)")
table("fisher_logdet_ratio", "{:+.2f}", "Fisher log-det ratio, JᵀJ vs exact (median; η_emu for gmi)")
table("fisher_sigma_logdet_ratio", "{:+.2f}", "Fisher log-det ratio, J_emuᵀΣ̂⁻¹J_emu vs exact (median)")
for col in ("fisher_logdet_ratio_iqr", "fisher_sigma_logdet_ratio_iqr"):
    fmt_list(col, "{:+.2f}")
    table(col + "_s", "{}", col.replace("_", " ").replace("iqr", "IQR"))
fmt_list("sigma_eig")
table("sigma_eig_s", "{}", "Σ̂ eigenvalues (test residual second moment; the critic assumes I)")
if "bias_chi2" in df:
    df["bias_s"] = [f"{c:.0f}/{d:.0f}" if pd.notna(c) else "" for c, d in zip(df["bias_chi2"], df["bias_dof"])]
    table("bias_s", "{}", "Binned residual bias χ²/dof (5×5 prior bins; ≈ 1 per dof if unbiased)")
table("residual_norm_median", "{:.2f}", f"Median ‖t − η_emu(θ*)‖ (√K = {np.sqrt(a.K):.0f} if t ~ N(η, I))")
fmt_list("canon_corr")
table("canon_corr_s", "{}", "Canonical correlations of t with η(θ*) (the information tower for gmi)")
fmt_list("harmonics_r2")
table("harmonics_r2_s", "{}", "Harmonics R² of canonical variates 2..K on cubic polynomials of the earlier ones (chance ≈ p/256)")
table("gmi_info", "{:.2f}", "gmi: implied Gaussian information −𝓛_GMI on the test set (nats)")

if traces:
    thr = -0.5 * (a.K - 2) * np.log(2 * np.pi)
    tr_rows = []
    for (arm, s, n), (tr, be) in traces.items():
        v = np.array([x[1] for x in tr]); ep = np.array([x[0] for x in tr])
        sel = v[ep <= (be if be is not None and be >= 0 else ep.max())]
        tr_rows.append(dict(arm=arm, budget=n, s=f"{v[0]:.2f}→{v[-1]:.2f} (max {v.max():.2f}{' ALARM' if sel.max() > thr + a.alarm_tol else ''})"))
    P = pd.DataFrame(tr_rows).pivot_table(index="arm", columns="budget", values="s", aggfunc="first")
    out.append(f"\n### E log Z(t) on 64 test maps over training: first → last (max); ALARM if > {thr:.2f} + {a.alarm_tol:g} (quadrature tolerance) up to the selected epoch\n")
    out.append(P.fillna("").to_markdown())

out.append("\n## Training")
if "epochs" in df:
    df["ep"] = df["epochs"].map(lambda v: v.get("main") if isinstance(v, dict) else v)
    table("ep", "{:.0f}", "Epochs trained (embedding)")
table("best_epoch", "{:.0f}", "Selected epoch")
table("B_epochs", "{:.0f}", "Epochs trained (frozen MAF)")
table("seconds", "{:.0f}", "Wall seconds per cell incl. evaluation B (1 thread, shared 2-core box)")
table("n_params", "{:.0f}", "Parameters")
text = "\n".join(out)
print(text)
if a.out:
    Path(a.out).write_text(text + "\n")
    df.drop(columns=[c for c in df if c.endswith("_s")]).to_csv(Path(a.out).with_suffix(".csv"), index=False)
