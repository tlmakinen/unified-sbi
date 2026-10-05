#!/usr/bin/env python3
"""Figures and tables for the few-run surrogate study.

    python scripts/plot_surrogate.py --results results/surrogate

Writes <results>/figures/{scaling,sensitivity_by_param,design}.png and
<results>/tables/{summary,paired_vs_A0,design}.csv.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# reference categorical palette, fixed order (validated: CVD and normal-vision checks pass;
# contrast WARN on aqua/yellow/magenta -> every series also gets a marker + line style and
# the tables carry the numbers)
ARM_STYLE = {
    "A0_film":    ("#2a78d6", "--", "o", "A0 FiLM on θ"),
    "A1_g":       ("#eb6834", "--", "s", "A1 learned g"),
    "A2_iso":     ("#1baf7a", "-",  "D", "A2 + R_iso (self-Fisher)"),
    "A3_iso_wn":  ("#eda100", "-",  "^", "A3 A2 + whitened noise"),
    "A4_iso_jac": ("#e87ba4", "-",  "v", "A4 A2 + Jacobian sup. (phys-Fisher)"),
    "A5_iso_in":  ("#008300", ":",  "X", "A5 A2 + isotropic noise"),
    "A6_full":    ("#4a3aa7", "-",  "P", "A6 full method"),
}
DESIGN_COL = {"random": "#2a78d6", "lhs": "#eb6834", "maximin_theta": "#1baf7a",
              "maximin_g": "#eda100", "maximin_geo": "#e87ba4"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PARAMS = ["log α", "κ", "log ν̂", "log D̂", "k0"]

PANELS = [
    ("onestep_nrmse", "one-step NRMSE (held-out runs)", True),
    ("rollout_nrmse_h10", "rollout NRMSE at 10 steps", True),
    ("flux_rel_err", "flux |Γ̂/Γ − 1| (rollouts)", True),
    ("flux_spearman", "flux rank corr. across runs ↑", False),
    ("jac_rel_err", "Jacobian rel. error vs branches", True),
    ("fisher_logdet_ratio", "regularised Fisher log-det ratio (0 = right)", False),
]


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def load(results):
    rows = []
    for p in Path(results).glob("seed*/N*/*/metrics.json"):
        m = json.loads(p.read_text())
        row = {k: v for k, v in m.items() if not isinstance(v, (list, dict))}
        for key in ("jac_rel_err_per_param", "jac_cos_per_param"):
            for i, v in enumerate(m.get(key, [])):
                row[f"{key}_{i}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def scaling_figure(df, out):
    arms = [a for a in ARM_STYLE if a in set(df.arm)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), constrained_layout=True)
    for ax, (key, title, logy) in zip(axes.flat, PANELS):
        if key not in df:
            ax.set_visible(False)
            continue
        for a in arms:
            g = df[df.arm == a].groupby("budget")[key]
            m, lo, hi = g.mean(), g.min(), g.max()
            c, ls, mk, lab = ARM_STYLE[a]
            ax.errorbar(m.index, m.values, yerr=[m - lo, hi - m], color=c, ls=ls, marker=mk,
                        ms=6, lw=2, capsize=0, elinewidth=1, label=lab)
        ax.set_xscale("log")
        if key == "fisher_logdet_ratio":
            ax.axhline(0, color=MUTED, lw=1)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(sorted(df.budget.unique()))
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("training runs N_θ", fontsize=8, color=MUTED)
        style(ax)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=min(4, len(arms)), frameon=False, fontsize=9)
    fig.suptitle("Held-out-θ accuracy and sensitivity vs number of runs (mean over seeds, bars = min–max)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.savefig(out / "scaling.png", dpi=150)
    plt.close(fig)


def sensitivity_figure(df, out):
    if "jac_rel_err_per_param_0" not in df:
        return
    Nmax = df.budget.max()
    arms = [a for a in ARM_STYLE if a in set(df.arm)]
    sub = df[df.budget == Nmax]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    w = 0.8 / len(arms)
    x = np.arange(len(PARAMS))
    for ax, key, title in [(axes[0], "jac_rel_err_per_param", f"Jacobian rel. error per parameter (N_θ = {Nmax})"),
                           (axes[1], "jac_cos_per_param", f"cosine(Ĵ_i, J_i) per parameter ↑ (N_θ = {Nmax})")]:
        for j, a in enumerate(arms):
            vals = [sub[sub.arm == a][f"{key}_{i}"].mean() for i in range(len(PARAMS))]
            c, _, _, lab = ARM_STYLE[a]
            ax.bar(x + (j - (len(arms) - 1) / 2) * w, vals, w * 0.9, color=c, label=lab,
                   edgecolor="white", linewidth=1)
        ax.set_xticks(x, PARAMS)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        style(ax)
    axes[0].set_yscale("log")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=min(4, len(arms)), frameon=False, fontsize=8)
    fig.savefig(out / "sensitivity_by_param.png", dpi=150)
    plt.close(fig)


def design_figure(results, out, tables):
    rows = []
    for p in Path(results).glob("design/seed*/N*/*/metrics.json"):
        m = json.loads(p.read_text())
        rows.append({k: v for k, v in m.items() if not isinstance(v, (list, dict))})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.groupby(["budget", "design"]).mean(numeric_only=True).to_csv(tables / "design.csv")
    keys = [("onestep_nrmse", "one-step NRMSE"), ("rollout_nrmse_h10", "rollout NRMSE @10"),
            ("jac_rel_err", "Jacobian rel. error")]
    budgets = sorted(df.budget.unique())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for ax, (k, t) in zip(axes, keys):
        if k not in df or df[k].isna().all():
            ax.set_visible(False)
            continue
        x = np.arange(len(budgets))
        present = [g for g in DESIGN_COL if g in set(df.design)]
        w = 0.8 / len(present)
        for j, g in enumerate(present):
            s = df[df.design == g].groupby("budget")[k]
            m = s.mean().reindex(budgets)
            lo, hi = s.min().reindex(budgets), s.max().reindex(budgets)
            ax.bar(x + (j - (len(present) - 1) / 2) * w, m, w * 0.92, color=DESIGN_COL[g], label=g,
                   edgecolor="white",
                   yerr=[m - lo, hi - m], error_kw=dict(elinewidth=1, ecolor=MUTED))
        ax.set_xticks(x, [f"N_θ = {b}" for b in budgets])
        ax.set_title(t, fontsize=10, color=INK, loc="left")
        style(ax)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, frameon=False, fontsize=9)
    fig.savefig(out / "design.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    res = Path(a.results)
    figs, tables = res / "figures", res / "tables"
    figs.mkdir(exist_ok=True, parents=True); tables.mkdir(exist_ok=True)
    df = load(res)
    if len(df):
        df.to_csv(tables / "all_cells.csv", index=False)
        num = df.drop(columns=["seed"]).groupby(["arm", "budget"]).mean(numeric_only=True)
        num.to_csv(tables / "summary.csv")
        # paired seed-level differences against the FiLM baseline (positive = better than A0)
        keys = [k for k, _, _ in PANELS if k in df]
        # the log-det ratio's target is 0, so compare its magnitude
        absd = df.assign(**({"fisher_logdet_ratio": df.fisher_logdet_ratio.abs()}
                            if "fisher_logdet_ratio" in df else {}))
        base = absd[absd.arm == "A0_film"].set_index(["seed", "budget"])[keys]
        out = []
        for arm in sorted(set(df.arm) - {"A0_film"}):
            d = absd[absd.arm == arm].set_index(["seed", "budget"])[keys]
            diff = (base - d).dropna(how="all")
            if "flux_spearman" in diff:
                diff["flux_spearman"] *= -1
            g = diff.groupby("budget")
            t = g.mean().add_suffix("_gain").join(g.sem().add_suffix("_se"))
            t["arm"] = arm
            out.append(t.reset_index())
        if out:
            pd.concat(out).to_csv(tables / "paired_vs_A0.csv", index=False)
        scaling_figure(df, figs)
        sensitivity_figure(df, figs)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            cols = [k for k, _, _ in PANELS if k in num]
            print(num[cols].round(4))
    design_figure(res, figs, tables)
    print("figures in", figs)


if __name__ == "__main__":
    main()
