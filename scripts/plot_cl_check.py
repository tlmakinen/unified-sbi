#!/usr/bin/env python3
"""Check the toy's lensing C_ell against CCL for a single source plane.

    python scripts/plot_cl_check.py --zs 1.0 --out results/figures/cl_check_z1.png

Toy model (lensing.lensing_cl): linear BBKS power with Gamma = Omega_m h, Carroll-Press-Turner growth,
convergence Limber integral with k = (ell + 1/2) / chi. Reference: pyccl (CMBLensingTracer = single-plane
convergence) with the same BBKS linear power (Omega_b -> 1e-4 so CCL's Sugiyama shape factor is ~1), plus
Eisenstein-Hu linear and halofit (Omega_b = 0.049) to show what the toy leaves out.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl

from unified_sbi.shared.lensing import LensingConfig, lensing_cl, noise_sigma

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs", type=float, default=1.0); ap.add_argument("--om", type=float, default=0.3)
    ap.add_argument("--s8", type=float, default=0.8, help="sigma_8")
    ap.add_argument("--out", default="results/figures/cl_check_z1.png")
    a = ap.parse_args()
    cfg = LensingConfig(z_sources=(a.zs,), params="sigma8")
    ell = np.geomspace(10, 5000, 120)
    ours = lensing_cl(np.array([a.om, a.s8]), ell, cfg)[:, 0, 0]

    def ref(tf, pk, ob):
        c = ccl.Cosmology(Omega_c=a.om - ob, Omega_b=ob, h=cfg.h, sigma8=a.s8, n_s=cfg.n_s,
                          transfer_function=tf, matter_power_spectrum=pk)
        t = ccl.CMBLensingTracer(c, z_source=a.zs)
        return ccl.angular_cl(c, t, t, ell)
    bbks = ref("bbks", "linear", 1e-4)
    eh = ref("eisenstein_hu", "linear", 0.049)
    hf = ref("eisenstein_hu", "halofit", 0.049)

    # multipole range of the toy maps and the shape-noise level used in the toy (white, per bin)
    ell_f = 2 * np.pi / np.deg2rad(cfg.field_deg)
    nbar = cfg.n_gal_per_bin * (60 * 180 / np.pi) ** 2            # per steradian
    N_ell = (cfg.noise_amp * cfg.sigma_e) ** 2 / nbar
    D = lambda c: ell * (ell + 1) * c / (2 * np.pi)

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.4, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3]))
    for n, alpha in ((64, 0.10), (128, 0.05)):
        ax.axvspan(ell_f, ell_f * n / 2 * np.sqrt(2), color=MUTED, alpha=alpha, lw=0)
        axr.axvspan(ell_f, ell_f * n / 2 * np.sqrt(2), color=MUTED, alpha=alpha, lw=0)
    ax.plot(ell, D(ours), color=C1, lw=2.2, label="toy (lensing.py): BBKS linear, CPT growth")
    ax.plot(ell, D(bbks), color=C2, lw=1.8, ls="--", label="CCL: BBKS linear")
    ax.plot(ell, D(eh), color=MUTED, lw=1.4, ls="-.", label="CCL: Eisenstein–Hu linear (Ω_b = 0.049)")
    ax.plot(ell, D(hf), color=C3, lw=1.8, ls=":", label="CCL: Eisenstein–Hu + halofit")
    ax.plot(ell, D(np.full_like(ell, N_ell)), color=MUTED, lw=1.0, label="toy shape noise (σ_e = 0.26, 2.5 arcmin⁻²)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylabel(r"$\ell(\ell+1)\,C_\ell^{\kappa\kappa}/2\pi$", color=INK)
    ax.set_title(f"Convergence power, single source plane z = {a.zs}, $\\Omega_m$ = {a.om}, $\\sigma_8$ = {a.s8}", color=INK, fontsize=10)
    ax.text(ell_f * 1.08, ax.get_ylim()[0] * 1.6 if ax.get_ylim()[0] > 0 else 1e-6, "64² map range (darker), 128² (lighter)",
            color=MUTED, fontsize=7.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    r = ours / bbks
    axr.plot(ell, r, color=C1, lw=2)
    axr.axhline(1, color=MUTED, lw=0.8, ls=":")
    axr.fill_between(ell, 0.98, 1.02, color=C2, alpha=0.12, lw=0)
    axr.set_ylim(0.95, 1.05)
    axr.set_ylabel("toy / CCL BBKS", color=INK); axr.set_xlabel(r"multipole $\ell$", color=INK)
    for x in (ax, axr):
        x.grid(color=GRID, lw=0.7, which="both"); x.set_axisbelow(True)
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
        x.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170); plt.close(fig)
    i = lambda L: np.argmin(abs(ell - L))
    print({L: dict(toy_over_ccl=round(float(r[i(L)]), 4), halofit_over_linear=round(float(hf[i(L)] / eh[i(L)]), 2)) for L in (36, 100, 300, 1000, 1600)})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
