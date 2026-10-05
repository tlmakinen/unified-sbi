#!/usr/bin/env python3
"""Figures and tables for the shared-space scaling study.

    python scripts/plot_shared.py --results results/shared_gauss
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# colour = features, line style = head (reference categorical palette, fixed order)
COL = {"pk": "#2a78d6", "cnn": "#eb6834", "hybrid": "#1baf7a", "hybrid_split": "#4a3aa7",
       "rect": "#e34948", "rect2": "#e87ba4"}
STYLE = {
    "pk_maf": (COL["pk"], "--", "o"), "pk_shared": (COL["pk"], "-", "o"),
    "cnn_maf": (COL["cnn"], "--", "s"), "cnn_shared": (COL["cnn"], "-", "s"),
    "hybrid_maf": (COL["hybrid"], "--", "D"), "hybrid_shared": (COL["hybrid"], "-", "D"),
    "hybrid_split_shared": (COL["hybrid_split"], "-", "^"),
    "hybrid_nested_shared": (COL["hybrid_split"], "-.", "^"), "pklin_shared": ("#008300", "-", "o"),
    "hybrid_manifold_shared": (COL["hybrid"], "-.", "X"),
    "hybrid_rect": (COL["rect"], ":", "v"), "hybrid_rect_2stage": (COL["rect2"], ":", "P"),
    # K = d arms of the sigma8 study: own hues so they never collide with the K = 4 arms above
    "hybrid_shared_K2": ("#008300", "-", "o"), "hybrid_affine_shared_K2": ("#eda100", "-", "v"),
    "hybrid_rect_K2": (COL["rect"], ":", "P"), "cnn_shared_K2": (COL["cnn"], "-.", "^"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# embedding-objective arms (hybrid_{quad|nce|hyv}_b{beta}, hybrid_gmi): colour = objective, solid = beta 0
OBJ_COL = {"quad": COL["hybrid"], "nce": COL["pk"], "hyv": COL["cnn"], "gmi": COL["hybrid_split"]}


def style(arm):
    if arm in STYLE:
        return STYLE[arm]
    import re
    m = re.match(r"hybrid_(quad|nce|hyv|gmi)(?:_b([0-9.]+))?", arm)
    if not m:
        return ("#777", "-", "o")
    beta = float(m.group(2) or 0)
    return (OBJ_COL[m.group(1)], "-" if beta == 0 or m.group(1) == "gmi" else "-.", "*" if beta > 0 else "o")


def load(results):
    rows = []
    for p in Path(results).glob("seed*/N*/*/metrics.json"):
        m = json.loads(p.read_text())
        ps = m.get("post_std", [np.nan, np.nan])
        rows.append({k: v for k, v in m.items() if not isinstance(v, (list, dict))}
                    | {"post_std_Om": ps[0], "post_std_S8": ps[1]})
    return pd.DataFrame(rows)


def style_axes(ax):
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def curve(ax, df, metric, arms, logy=False):
    for arm in arms:
        d = df[df.arm == arm].groupby("budget")[metric].agg(["mean", "std", "count"]).reset_index()
        if d.empty:
            continue
        c, ls, mk = style(arm)
        ax.plot(d.budget, d["mean"], ls=ls, marker=mk, color=c, lw=2, ms=6, label=arm,
                mec="white", mew=1)
        if (d["count"] > 1).any():
            se = d["std"] / np.sqrt(d["count"])
            ax.fill_between(d.budget, d["mean"] - se, d["mean"] + se, color=c, alpha=0.12, lw=0)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ns = sorted(df.budget.unique())
    ax.set_xticks(ns, [str(n) for n in ns]); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    style_axes(ax)


def savings(df, metric, pairs):
    """Budget the baseline needs to match the challenger (log-log interpolation)."""
    out = []
    for base, chal in pairs:
        b = df[df.arm == base].groupby("budget")[metric].mean()
        c = df[df.arm == chal].groupby("budget")[metric].mean()
        if b.empty or c.empty:
            continue
        lb, vb = np.log(b.index.values.astype(float)), b.values
        for n, v in c.items():
            if v < vb.min():
                need, bound = np.exp(lb[np.argmin(vb)]), ">="
            elif v > vb.max():
                need, bound = np.exp(lb[np.argmax(vb)]), "<="
            else:      # baseline curve assumed decreasing; use the monotone envelope
                env = np.minimum.accumulate(vb)
                i = np.searchsorted(-env, -v)
                i = min(max(i, 1), len(env) - 1)
                f = (v - env[i - 1]) / (env[i] - env[i - 1]) if env[i] != env[i - 1] else 0
                need, bound = np.exp(lb[i - 1] + f * (lb[i] - lb[i - 1])), "="
            out.append(dict(baseline=base, challenger=chal, budget=n, value=v,
                            baseline_budget_needed=need, bound=bound, factor=need / n))
    return pd.DataFrame(out)


def kscreen(screen, ref):
    """Excess NLL against K for pklin_shared, with the MAF baselines of ``ref`` as reference lines."""
    import re
    rows = []
    for p in Path(screen).glob("K*_*/seed*/N*/*/metrics.json"):
        m = json.loads(p.read_text())
        rows.append(dict(K=int(re.search(r"K(\d+)_", str(p)).group(1)), budget=m["budget"], arm=m["arm"],
                         excess=m["excess_nll"], se=m["excess_nll_se"]))
    df = pd.DataFrame(rows).sort_values("K")
    df.to_csv(Path(screen) / "kscreen.csv", index=False)
    refdf = load(ref) if ref else pd.DataFrame()
    budgets = sorted(df.budget.unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(5.2 * len(budgets), 4.2), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], budgets):
        d = df[(df.budget == n) & (df.arm == "pklin_shared")]
        ax.errorbar(d.K, d.excess, d.se, color="#008300", marker="o", lw=2, capsize=3, label="pklin_shared (t = W·Pk)")
        for arm, ls, lab in (("pk_maf", "--", "pk_maf"), ("hybrid_maf", "-.", "hybrid_maf"),
                             ("pk_shared", ":", "pk_shared (MLP summary, K=4)")):
            r = refdf[(refdf.arm == arm) & (refdf.budget == n)] if not refdf.empty else []
            if len(r):
                ax.axhline(r.excess_nll.mean(), color=STYLE[arm][0], ls=ls, lw=1.5, label=lab)
        ax.set_xscale("log", base=2); ax.set_xticks(sorted(df.K.unique()), [str(k) for k in sorted(df.K.unique())])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(f"N = {n}", color=INK); ax.set_xlabel("shared-space dimension K", color=INK)
        style_axes(ax)
    axes[0, 0].set_ylabel("excess NLL = E KL(p || q) [nats]", color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Statistics linear in the measured spectrum: more K, less bias", color=INK)
    fig.tight_layout(); out = Path(screen) / "kscreen.png"; fig.savefig(out, dpi=160); plt.close(fig)
    print(df.pivot_table(index="K", columns="budget", values="excess").round(3).to_string())


def main():
    p = argparse.ArgumentParser(); p.add_argument("--results")
    p.add_argument("--kscreen", help="results of scripts/shared_kscreen.py"); p.add_argument("--ref")
    a = p.parse_args()
    if a.kscreen:
        kscreen(a.kscreen, a.ref); return
    res = Path(a.results); fig_dir = res / "figures"; fig_dir.mkdir(exist_ok=True)
    df = load(res)
    if df.empty:
        print("no results yet"); return
    df.to_csv(res / "metrics.csv", index=False)
    has_oracle = "excess_nll" in df
    metric = "excess_nll" if has_oracle else "nll"
    arms = [a for a in STYLE if a in set(df.arm)] + sorted(set(df.arm) - set(STYLE))
    arms = [a for a in arms if df[df.arm == a].budget.nunique() > 1]     # curves only (drops one-N pilots/ablations)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    curve(axes[0], df, metric, arms, logy=has_oracle)
    axes[0].set_ylabel("excess NLL = E KL(p || q) [nats]" if has_oracle else "test NLL [nats]", color=INK)
    if not has_oracle:
        pass
    curve(axes[1], df, "joint_cov_mae", arms)
    axes[1].set_ylabel("joint HPD coverage error", color=INK)
    rect_arms = [a for a in arms if "rect" in a and "2stage" not in a]
    curve(axes[2], df[df.arm.isin(rect_arms)], "mean_logZ", rect_arms)
    axes[2].axhline(0, color=MUTED, lw=1, ls=":")
    K_cfg = json.loads((res / "config.json").read_text()).get("K", 4)
    for K in sorted({2 if a.endswith("_K2") else K_cfg for a in rect_arms}):
        if K > 2:      # K = d: the flat-manifold value is 0, the dotted line
            axes[2].axhline(-0.5 * (K - 2) * np.log(2 * np.pi), color=MUTED, lw=1, ls="--")
            axes[2].text(axes[2].get_xlim()[0], -0.5 * (K - 2) * np.log(2 * np.pi) + 0.05,
                         f"  K = {K}, flat manifold, t on it: -(K-d)/2 log 2π", color=MUTED, fontsize=8)
    axes[2].set_ylabel("E log ∫ q_rect dθ  (0 if normalised)", color=INK)
    for ax in axes:
        ax.set_xlabel("simulations N", color=INK)
    oracle = res / "oracle.json"
    title = "Shared-space vs MAF heads, same simulations"
    if oracle.exists():
        o = json.loads(oracle.read_text()); title += f"   (exact posterior NLL {o['nll']:.2f})"
    fig.suptitle(title, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(fig_dir / "scaling.png", dpi=160); plt.close(fig)

    if "B_" + metric in df:          # evaluation B: MAF on the frozen t(x)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        armsB = [a for a in arms if df[df.arm == a]["B_" + metric].notna().any()]
        curve(axes[0], df, "B_" + metric, armsB, logy=has_oracle)
        if "hybrid_maf" in set(df.arm):
            curve(axes[0], df, metric, ["hybrid_maf"], logy=has_oracle)
        axes[0].set_ylabel(("excess NLL" if has_oracle else "test NLL") + " of a MAF on the frozen t(x)", color=INK)
        curve(axes[1], df, "B_joint_cov_mae", armsB)
        axes[1].set_ylabel("joint HPD coverage error (evaluation B)", color=INK)
        for ax in axes:
            ax.set_xlabel("simulations N", color=INK)
        fig.suptitle("Evaluation B: frozen summaries + downstream MAF (hybrid_maf: end-to-end)", color=INK)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.12, 1, 0.95))
        fig.savefig(fig_dir / "scaling_B.png", dpi=160); plt.close(fig)

    pairs = [("pk_maf", "pk_shared"), ("cnn_maf", "cnn_shared"), ("hybrid_maf", "hybrid_shared"),
             ("hybrid_maf", "hybrid_split_shared"), ("hybrid_rect_2stage", "hybrid_shared"),
             ("hybrid_maf", "hybrid_rect")]
    sv = savings(df, metric, pairs)
    sv.to_csv(res / "savings.csv", index=False)
    tab = df.pivot_table(index="arm", columns="budget", values=metric, aggfunc="mean").reindex(arms)
    tab.round(3).to_csv(res / f"table_{metric}.csv")
    print(tab.round(3).to_string())
    if "fisher_logdet_ratio" in df:
        print("\nlearned/exact Fisher log-det ratio (median over test):")
        print(df.pivot_table(index="arm", columns="budget", values="fisher_logdet_ratio").round(2).to_string())
    print("\nparameters:", df.groupby("arm").n_params.first().to_dict())
    if not sv.empty:
        print("\nsavings (baseline budget needed / challenger budget):")
        print(sv.assign(factor=sv.bound + sv.factor.round(2).astype(str))
              .pivot_table(index=["baseline", "challenger"], columns="budget", values="factor", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
