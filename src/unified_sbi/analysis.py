"""Aggregate scaling-study rows: seed means, power-law fits and simulation-savings factors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = {
    # name: (column, higher_is_better)
    "val_log_prob": ("val_log_prob", True),
    "test_log_prob": ("test_log_prob", True),
    "crps": ("crps", False),
}


def load_rows(results_dir: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(results_dir, "rows").glob("*.json")):
        rows.extend(json.loads(path.read_text()))
    if not rows:
        raise FileNotFoundError(f"No rows under {results_dir}/rows")
    df = pd.DataFrame(rows)
    df["excess_crps"] = df["crps"] - df["oracle_crps"]
    df["excess_nll"] = df["oracle_test_log_prob"] - df["test_log_prob"]
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["val_log_prob", "test_log_prob", "crps", "excess_crps", "excess_nll",
            "coverage68", "coverage90", "npe_seconds", "map_seconds"]
    g = df.groupby(["arm", "n_total"])[cols]
    out = g.mean().add_suffix("_mean").join(g.sem().add_suffix("_sem")).join(g.size().rename("n_seeds"))
    return out.reset_index()


def power_law_fit(n: np.ndarray, y: np.ndarray):
    """Fit y = A * N^(-alpha) to positive y by least squares in log-log. Returns (A, alpha)."""
    ok = (y > 0) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(np.log(n[ok]), np.log(y[ok]), 1)
    return float(np.exp(intercept)), float(-slope)


def equivalent_budget(n_ref: np.ndarray, y_ref: np.ndarray, y: float, higher_is_better: bool):
    """Budget at which the reference curve reaches value y, interpolating linearly in log N.

    Uses the running best of the reference curve so it is monotone. Returns (budget, bound):
    bound is "=" inside the sweep, "<=" if the reference already beats y at its smallest budget
    (budget = that smallest budget), and ">=" if it never reaches y (budget = its largest budget).
    """
    order = np.argsort(n_ref)
    n_ref, y_ref = np.asarray(n_ref, dtype=float)[order], np.asarray(y_ref)[order]
    sign = 1.0 if higher_is_better else -1.0
    best = np.maximum.accumulate(sign * y_ref)
    target = sign * y
    if target <= best[0]:
        return float(n_ref[0]), "<="
    if target > best[-1]:
        return float(n_ref[-1]), ">="
    k = int(np.searchsorted(best, target))
    lo, hi = best[k - 1], best[k]
    frac = 0.0 if hi == lo else (target - lo) / (hi - lo)
    return float(np.exp(np.log(n_ref[k - 1]) + frac * (np.log(n_ref[k]) - np.log(n_ref[k - 1])))), "="


def savings_table(agg: pd.DataFrame, baseline: str = "raw") -> pd.DataFrame:
    """For each arm and budget: the baseline budget needed to match it, and the ratio.

    "savings" = (baseline budget to match) / N_total, with its bound: ">=" means the baseline
    never matches inside the sweep, "<=" means the arm is worse than the baseline's smallest budget.
    """
    base = agg[agg.arm == baseline].sort_values("n_total")
    rows = []
    for _, r in agg[agg.arm != baseline].iterrows():
        row = dict(arm=r.arm, n_total=int(r.n_total))
        for name, (col, hib) in METRICS.items():
            n_eq, bound = equivalent_budget(base.n_total.values, base[f"{col}_mean"].values,
                                            r[f"{col}_mean"], hib)
            row[f"{name}: {baseline} budget to match"] = n_eq
            row[f"{name}: savings"] = n_eq / r.n_total
            row[f"{name}: bound"] = bound
            row[f"{name}: savings (text)"] = f"{'' if bound == '=' else bound}{n_eq / r.n_total:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def power_law_table(agg: pd.DataFrame, min_budget: int = 0) -> pd.DataFrame:
    rows = []
    for arm, g in agg[agg.n_total >= min_budget].groupby("arm"):
        for col in ("excess_crps", "excess_nll"):
            a, alpha = power_law_fit(g.n_total.values.astype(float), g[f"{col}_mean"].values)
            rows.append(dict(arm=arm, quantity=col, A=a, alpha=alpha))
    return pd.DataFrame(rows)
