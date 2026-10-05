"""Tomographic weak-lensing toy with an exact posterior.

Parameters ``theta = (Omega_m, S8)`` with the box prior of the hybrid-statistics paper
(arXiv:2410.07548): Omega_m in [0.15, 0.70], S8 in [0.35, 1.52]. With ``params="sigma8"`` the
second parameter is sigma_8 instead, on the box Omega_m in [0.15, 0.70], sigma_8 in [0.40, 1.40]
(same physics; the S8 degeneracy is then a curved banana that a method has to discover).

Physics (deliberately simple, fully vectorised over theta):
  * flat LCDM background, h = 0.7, n_s = 0.96, BBKS transfer with Gamma = Omega_m h;
  * sigma_8 = S8 / sqrt(Omega_m / 0.3), linear growth from the Carroll-Press-Turner fit;
  * four single-plane source bins, Limber integral over a redshift grid;
  * periodic ``n x n`` maps of side ``field_deg`` degrees.

Fourier convention: with ``X = fft2(x)`` (numpy, unnormalised), each mode has
``E[X_l X_l^H] = Npix * Sigma_l`` where ``Sigma_l = C(l)/Omega_pix + sigma_n^2 I`` (4x4).
The signal power is constant on unit-width annuli ``b = round(|k|)`` and evaluated at the
annulus mean multipole, so the per-annulus 4x4 sample cross-spectra are exactly sufficient
for the Gaussian field.

``field="gaussian"``: the likelihood is a product of complex Gaussians over Fourier modes, so
``exact_loglik`` and ``exact_fisher`` are exact (no approximation beyond float64).
``field="lognormal"``: shifted-lognormal transform per bin (FLASK-style correlation-function
mapping). No closed-form likelihood; information beyond the power spectrum exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

LO = np.array([0.15, 0.35])
HI = np.array([0.70, 1.52])
PARAM_NAMES = ["Omega_m", "S8"]
# prior box and names per parametrisation; LO/HI above are the default ("S8") box
BOXES = {"S8": (LO, HI), "sigma8": (np.array([0.15, 0.40]), np.array([0.70, 1.40]))}
NAMES = {"S8": ["Omega_m", "S8"], "sigma8": ["Omega_m", "sigma8"]}


def param_box(params: str = "S8"):
    """(lo, hi) of the uniform prior box for ``params`` in {"S8", "sigma8"}."""
    if params not in BOXES:
        raise ValueError(f"params must be one of {list(BOXES)}, got {params!r}")
    lo, hi = BOXES[params]
    return lo.copy(), hi.copy()
C_KMS = 299792.458


@dataclass
class LensingConfig:
    n: int = 64                       # pixels per side
    field_deg: float = 10.0           # side length in degrees
    z_sources: tuple = (0.5, 0.8, 1.1, 1.5)
    n_gal_per_bin: float = 2.5        # galaxies / arcmin^2 / bin
    sigma_e: float = 0.26
    noise_amp: float = 1.0            # multiplies the shape-noise standard deviation
    field: str = "gaussian"           # "gaussian" | "lognormal"
    lognormal_shift: tuple = (0.008, 0.012, 0.016, 0.020)
    h: float = 0.7
    n_s: float = 0.96
    nz_limber: int = 48
    params: str = "S8"                # second parameter: "S8" or "sigma8" (see param_box)

    def to_dict(self):
        return asdict(self)

    @property
    def n_bins(self):
        return len(self.z_sources)


# ----------------------------------------------------------------------------- geometry

def fourier_geometry(cfg: LensingConfig):
    """Integer |k| annulus index per full-plane mode, annulus mean multipole, counts."""
    n = cfg.n
    k = np.fft.fftfreq(n) * n
    kk = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    ann = np.rint(kk).astype(int)                 # 0 is DC
    n_ann = ann.max() + 1
    counts = np.bincount(ann.ravel(), minlength=n_ann)
    kmean = np.bincount(ann.ravel(), weights=kk.ravel(), minlength=n_ann) / np.maximum(counts, 1)
    ell_f = 2 * np.pi / np.deg2rad(cfg.field_deg)
    ell = kmean * ell_f
    omega_pix = (np.deg2rad(cfg.field_deg) / n) ** 2
    return dict(ann=ann, n_ann=n_ann, counts=counts, ell=ell, omega_pix=omega_pix, kk=kk)


def noise_sigma(cfg: LensingConfig) -> float:
    pix_arcmin2 = (cfg.field_deg * 60 / cfg.n) ** 2
    return cfg.noise_amp * cfg.sigma_e / np.sqrt(cfg.n_gal_per_bin * pix_arcmin2)


# ----------------------------------------------------------------------------- cosmology

def _bbks(k, gamma):
    q = k / gamma
    return np.log1p(2.34 * q) / (2.34 * q) * (1 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4) ** -0.25


def _growth_g(om):
    ol = 1 - om
    return 2.5 * om / (om ** (4 / 7) - ol + (1 + om / 2) * (1 + ol / 70))


def lensing_cl(theta, ell, cfg: LensingConfig):
    """C_ij(ell) for theta (..., 2) and multipoles ell (L,). Returns (..., L, nb, nb)."""
    theta = np.asarray(theta, dtype=np.float64)
    om, p2 = theta[..., 0], theta[..., 1]
    sig8 = p2 if getattr(cfg, "params", "S8") == "sigma8" else p2 / np.sqrt(om / 0.3)
    h, ns = cfg.h, cfg.n_s
    gamma = om * h
    # sigma_8 normalisation of P_lin(k) = A k^ns T^2 (k in h/Mpc)
    lk = np.linspace(np.log(1e-4), np.log(1e2), 400)
    kq = np.exp(lk)
    x = kq * 8.0
    w = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    T = _bbks(kq, gamma[..., None])
    integrand = kq ** (3 + ns) * T ** 2 * w ** 2 / (2 * np.pi ** 2)
    sig2_unit = np.trapezoid(integrand, lk, axis=-1)
    amp = sig8 ** 2 / sig2_unit                                       # (...)
    # background on a z grid up to the furthest source
    zs = np.asarray(cfg.z_sources)
    zg = np.linspace(1e-3, zs.max(), cfg.nz_limber * 8)
    Ez = np.sqrt(om[..., None] * (1 + zg) ** 3 + 1 - om[..., None])   # (..., Z)
    chi_g = np.concatenate([np.zeros(Ez.shape[:-1] + (1,)),
                            np.cumsum(0.5 * (1 / Ez[..., 1:] + 1 / Ez[..., :-1]) * np.diff(zg), -1)], -1)
    chi_g = chi_g * C_KMS / 100.0                                     # Mpc/h
    chi_src = _interp_last(zs, zg, chi_g)                              # (..., nb)
    # Limber nodes (midpoint in z)
    edges = np.linspace(0, zs.max(), cfg.nz_limber + 1)
    zl = 0.5 * (edges[1:] + edges[:-1]); dz = np.diff(edges)
    El = np.sqrt(om[..., None] * (1 + zl) ** 3 + 1 - om[..., None])
    chi_l = _interp_last(zl, zg, chi_g)                               # (..., Zl)
    dchi = C_KMS / 100.0 * dz / El
    # growth: D(a) = a g(Om(a)) / g(Om0)
    a = 1 / (1 + zl)
    om_a = om[..., None] / (om[..., None] + (1 - om[..., None]) * a ** 3)
    D = a * _growth_g(om_a) / _growth_g(om)[..., None]
    # lensing efficiency q_i(z) = 1.5 Om (H0/c)^2 (1+z) chi (chi_i - chi)/chi_i
    H0c = 100.0 / C_KMS
    qi = 1.5 * om[..., None, None] * H0c ** 2 * (1 + zl) * chi_l[..., None, :] * \
        np.clip(chi_src[..., :, None] - chi_l[..., None, :], 0, None) / chi_src[..., :, None]   # (..., nb, Zl)
    # P(k = ell/chi, z)
    ell = np.asarray(ell, dtype=np.float64)
    kl = (ell[:, None] + 0.5) / chi_l[..., None, :]                   # (..., L, Zl)
    Tl = _bbks(kl, gamma[..., None, None])
    P = amp[..., None, None] * kl ** ns * Tl ** 2 * D[..., None, :] ** 2
    weight = dchi / chi_l ** 2                                        # (..., Zl)
    cl = np.einsum("...iz,...jz,...lz,...z->...lij", qi, qi, P, weight)
    return cl


# ----------------------------------------------------------------------------- simulator

class LensingSimulator:
    """Noise-free signal maps; noise is added separately so it can be redrawn per epoch."""

    def __init__(self, cfg: LensingConfig):
        self.cfg = cfg
        self.geo = fourier_geometry(cfg)
        self.sigma_n = noise_sigma(cfg)
        self.lo, self.hi = param_box(getattr(cfg, "params", "S8"))

    def prior_sample(self, n, rng):
        return self.lo + (self.hi - self.lo) * rng.random((n, 2))

    def signal_sigma(self, theta):
        """Per-annulus signal covariance in map units, (..., n_ann, nb, nb); DC set to zero."""
        g = self.geo
        cl = lensing_cl(theta, g["ell"], self.cfg) / g["omega_pix"]
        cl[..., 0, :, :] = 0.0
        return cl

    def simulate(self, theta, rng, batch=256):
        theta = np.atleast_2d(theta)
        out = np.empty((len(theta), self.cfg.n_bins, self.cfg.n, self.cfg.n), np.float32)
        for i in range(0, len(theta), batch):
            out[i:i + batch] = self._simulate_batch(theta[i:i + batch], rng)
        return out

    def _simulate_batch(self, theta, rng):
        cfg, g = self.cfg, self.geo
        nb, n = cfg.n_bins, cfg.n
        S = self.signal_sigma(theta)                                  # (B, A, nb, nb)
        B = len(theta)
        if cfg.field == "gaussian":
            L = _psd_sqrt(S)                                          # (B, A, nb, nb)
            Lmode = L[:, g["ann"]]                                    # (B, n, n, nb, nb)
            eps = rng.standard_normal((B, nb, n, n))
            E = np.fft.fft2(eps)                                      # (B, nb, n, n)
            X = np.einsum("bxyij,bjxy->bixy", Lmode, E)
            return np.fft.ifft2(X).real
        if cfg.field == "lognormal":
            lam = np.asarray(cfg.lognormal_shift)
            Cmode = S[:, g["ann"]]                                    # (B, n, n, nb, nb)
            xi = np.fft.ifft2(np.moveaxis(Cmode, (1, 2), (-2, -1))).real   # (B, nb, nb, n, n)
            lamij = lam[:, None] * lam[None, :]
            xi_g = np.log1p(xi / lamij[None, :, :, None, None])
            Cg = np.fft.fft2(xi_g).real                               # (B, nb, nb, n, n)
            Cg = np.moveaxis(Cg, (-2, -1), (1, 2))                    # (B, n, n, nb, nb)
            Cg = 0.5 * (Cg + np.swapaxes(Cg, -1, -2))
            Lg = _psd_sqrt(Cg)
            eps = rng.standard_normal((B, nb, n, n))
            E = np.fft.fft2(eps)
            G = np.fft.ifft2(np.einsum("bxyij,bjxy->bixy", Lg, E)).real
            var_g = np.einsum("biixy->bi", xi_g[..., :1, :1])           # xi_g at zero lag
            return lam[None, :, None, None] * (np.exp(G - 0.5 * var_g[:, :, None, None]) - 1)
        raise ValueError(cfg.field)

    def add_noise(self, x, rng):
        return x + self.sigma_n * rng.standard_normal(x.shape).astype(x.dtype)

    # ------------------------------------------------------------- exact (Gaussian field)

    def sufficient_stats(self, x):
        """Per-annulus summed cross-power  S_b = sum_{l in b} Re(X_l X_l^H) / Npix, (B, A, nb, nb)."""
        g = self.geo
        X = np.fft.fft2(np.asarray(x, np.float64))                    # (B, nb, n, n)
        P = np.einsum("bixy,bjxy->bxyij", X, X.conj()).real / (self.cfg.n ** 2)
        A = g["n_ann"]
        out = np.zeros((len(x), A) + P.shape[-2:])
        flat = P.reshape(len(x), -1, *P.shape[-2:])
        np.add.at(out, (slice(None), g["ann"].ravel()), flat)
        return out

    def loglik_from_stats(self, stats, theta, chunk=4096):
        """Exact log p(x | theta) up to a theta-independent constant.

        stats: (A, nb, nb) for one observation; theta: (M, 2). Returns (M,).
        """
        g = self.geo
        cnt = g["counts"].astype(np.float64)
        out = np.empty(len(theta))
        eye = np.eye(self.cfg.n_bins)
        for i in range(0, len(theta), chunk):
            Sig = self.signal_sigma(theta[i:i + chunk]) + self.sigma_n ** 2 * eye
            Lc = np.linalg.cholesky(Sig)
            logdet = 2 * np.log(np.diagonal(Lc, axis1=-2, axis2=-1)).sum(-1)       # (m, A)
            inv = np.linalg.inv(Sig)
            quad = np.einsum("maij,aji->ma", inv, stats)
            out[i:i + chunk] = -0.5 * (quad + cnt * logdet).sum(-1)
        return out

    def exact_fisher(self, theta, step=1e-4):
        """Exact Fisher matrix per observation: 1/2 sum_l tr(S^-1 dS_a S^-1 dS_b). (M, 2, 2)."""
        theta = np.atleast_2d(theta)
        cnt = self.geo["counts"].astype(np.float64)
        eye = np.eye(self.cfg.n_bins)
        Sig = self.signal_sigma(theta) + self.sigma_n ** 2 * eye
        inv = np.linalg.inv(Sig)
        dS = []
        for a in range(2):
            d = np.zeros(2); d[a] = step * (self.hi[a] - self.lo[a])
            dS.append((self.signal_sigma(theta + d) - self.signal_sigma(theta - d)) / (2 * d[a]))
        F = np.zeros((len(theta), 2, 2))
        for a in range(2):
            for b in range(2):
                F[:, a, b] = 0.5 * np.einsum("a,maij,majk,makl,mali->m", cnt, inv, dS[a], inv, dS[b])
        return F


def _interp_last(znew, zgrid, values):
    """Linear interpolation along the last axis of ``values`` (shared grid)."""
    idx = np.clip(np.searchsorted(zgrid, znew) - 1, 0, len(zgrid) - 2)
    w = (np.asarray(znew) - zgrid[idx]) / (zgrid[idx + 1] - zgrid[idx])
    return values[..., idx] * (1 - w) + values[..., idx + 1] * w


def _psd_sqrt(S):
    """Symmetric square root via eigendecomposition with negative eigenvalues clipped."""
    w, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.clip(w, 0, None))[..., None, :]) @ np.swapaxes(V, -1, -2)
