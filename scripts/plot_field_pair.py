#!/usr/bin/env python3
"""Side-by-side Gaussian and lognormal convergence maps with the same phases and parameters.

    python scripts/plot_field_pair.py --out results/figures/field_pair.png --om 0.3 --s8 0.8 --bin 3
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from unified_sbi.shared.lensing import LensingConfig, LensingSimulator

INK, MUTED = "#0b0b0b", "#52514e"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8a1f1e"])   # gray midpoint = zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures/field_pair.png")
    ap.add_argument("--om", type=float, default=0.3); ap.add_argument("--s8", type=float, default=0.8,
                    help="sigma_8 (the maps use params='sigma8')")
    ap.add_argument("--bin", type=int, default=3, help="tomographic bin 0-3 (z = 0.5, 0.8, 1.1, 1.5)")
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise", action="store_true", help="add shape noise")
    a = ap.parse_args()
    theta = np.array([[a.om, a.s8]])
    maps = {}
    for field in ("gaussian", "lognormal"):
        cfg = LensingConfig(field=field, params="sigma8", n=a.n)
        sim = LensingSimulator(cfg)
        x = sim.simulate(theta, np.random.default_rng(a.seed))       # same seed -> same Gaussian phases
        if a.noise:
            x = sim.add_noise(x, np.random.default_rng(a.seed + 1))
        maps[field] = x[0, a.bin].astype(np.float64)
    zs = LensingConfig().z_sources[a.bin]
    v = np.percentile(np.abs(np.concatenate([m.ravel() for m in maps.values()])), 99.5)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), constrained_layout=True)
    L = LensingConfig().field_deg
    for ax, (field, m) in zip(axes, maps.items()):
        im = ax.imshow(m.T, origin="lower", cmap=DIVERGING, vmin=-v, vmax=v, extent=(0, L, 0, L),
                       interpolation="nearest")
        skew = float(((m - m.mean()) ** 3).mean() / m.std() ** 3)
        ax.set_title(f"{field}   (std {m.std():.4f}, skewness {skew:+.2f})", color=INK, fontsize=10)
        ax.set_xlabel("deg", color=MUTED); ax.set_ylabel("deg", color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(MUTED)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(r"convergence $\kappa$", color=INK); cb.ax.tick_params(colors=MUTED, labelsize=8)
    noise = "with shape noise" if a.noise else "noise-free"
    fig.suptitle(f"Same phases, $\\Omega_m$ = {a.om}, $\\sigma_8$ = {a.s8}, source bin z = {zs} ({a.n}², {L:g}°, {noise})",
                 color=INK)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170); plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
