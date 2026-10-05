"""Batched pseudo-spectral solver for the modified Hasegawa-Wakatani (HW) model, in torch.

This is the Tier-1 testbed of ``docs/plasma_surrogate_conditioning.md`` (section 8.1): a cheap
reduced plasma-turbulence model with ``d = 5`` control parameters, any number of runs, any
design, and branch restarts that give the *true* one-step sensitivity ds_{t+1}/dtheta.

Equations (x radial, y poloidal; zonal = ky == 0 modes; tilde = non-zonal part):

    d_t Omega + [phi, Omega] = alpha (phi~ - n~)                - nu k^{2p} Omega
    d_t n     + [phi, n]     = alpha (phi~ - n~) - kappa d_y phi - D  k^{2p} n
    Omega = lap phi,          [a, b] = d_x a d_y b - d_y a d_x b

on a doubly periodic box of side L = 2 pi / k0 with ``n x n`` points. Time stepping is RK4
with an integrating factor for the (diagonal) hyperdiffusion, 2/3-rule dealiasing of the
bracket, and fixed ``dt``. Every run in a batch has its own parameters.

Parameters (natural coordinates, uniform prior on a box; ``standardise`` maps the box to zero
mean and unit variance per coordinate):

    theta = (log10 alpha, kappa, log10 nu_hat, log10 D_hat, k0)

The hyperdiffusion coefficients are grid-relative: nu_hat and D_hat are the damping rates at
the dealiased cutoff k_c = k0 n / 3, i.e. nu = nu_hat / k_c^(2p). This keeps grid-scale
dissipation in a controlled range for every box size.

The surrogate's state is the pair of real fields (n, phi). A restart from a stored (n, phi)
rebuilds Omega = lap phi with the *restart's* k0, so the one-step map (state array, theta) ->
next state array is well defined for every theta, which is exactly what the surrogate learns.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

NAMES = ("log10_alpha", "kappa", "log10_nu_hat", "log10_D_hat", "k0")


@dataclass
class HWConfig:
    n: int = 64                    # grid points per side
    p: int = 3                     # hyperdiffusion order: damping nu k^(2p)
    dt: float = 0.02               # solver step
    save_dt: float = 1.0           # time between stored snapshots (= one surrogate step)
    t_spin: float = 250.0          # spin-up before the first stored snapshot
    n_save: int = 120              # stored snapshots per run
    ic_amp: float = 0.2            # amplitude of the random initial condition
    batch: int = 64                # runs integrated together
    # prior box in natural coordinates (see NAMES)
    lo: tuple = (-1.0, 0.7, math.log10(0.3), math.log10(0.3), 0.15)
    hi: tuple = (0.0, 1.4, 1.0, 1.0, 0.25)

    @property
    def d(self):
        return len(self.lo)

    @property
    def steps_per_save(self):
        return int(round(self.save_dt / self.dt))

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------------------------
# parameters


def box(cfg: HWConfig, dtype=torch.float64):
    return torch.tensor(cfg.lo, dtype=dtype), torch.tensor(cfg.hi, dtype=dtype)


def standardise(theta, cfg: HWConfig):
    """Natural coordinates -> zero mean, unit variance under the uniform prior."""
    lo, hi = box(cfg, theta.dtype)
    lo, hi = lo.to(theta.device), hi.to(theta.device)
    return (theta - (lo + hi) / 2) / ((hi - lo) / 2 / math.sqrt(3))


def unstandardise(u, cfg: HWConfig):
    lo, hi = box(cfg, u.dtype)
    lo, hi = lo.to(u.device), hi.to(u.device)
    return u * ((hi - lo) / 2 / math.sqrt(3)) + (lo + hi) / 2


def sample_prior(n, cfg: HWConfig, rng: np.random.Generator):
    lo, hi = np.array(cfg.lo), np.array(cfg.hi)
    return lo + (hi - lo) * rng.random((n, cfg.d))


def latin_hypercube(n, cfg: HWConfig, rng: np.random.Generator):
    lo, hi = np.array(cfg.lo), np.array(cfg.hi)
    u = (np.stack([rng.permutation(n) for _ in range(cfg.d)], 1) + rng.random((n, cfg.d))) / n
    return lo + (hi - lo) * u


def physical(theta, cfg: HWConfig):
    """(B, 5) natural coordinates -> dict of (B, 1, 1) physical parameters."""
    th = theta[:, :, None, None]
    kc = (th[:, 4] * cfg.n / 3.0) ** (2 * cfg.p)
    return dict(alpha=10 ** th[:, 0], kappa=th[:, 1], nu=10 ** th[:, 2] / kc, D=10 ** th[:, 3] / kc,
                k0=th[:, 4])


# ---------------------------------------------------------------------------------------------
# spectral operators


@dataclass
class Grid:
    kx: torch.Tensor      # (B, n, 1)
    ky: torch.Tensor      # (B, 1, n//2+1)
    k2: torch.Tensor      # (B, n, n//2+1)
    inv_k2: torch.Tensor
    dealias: torch.Tensor  # (n, n//2+1) bool
    nonzonal: torch.Tensor  # (1, n//2+1) real
    n: int = field(default=0)


def make_grid(k0, n, device, dtype):
    """k0: (B, 1, 1). Wavenumbers are k0 * integer."""
    ix = torch.fft.fftfreq(n, d=1.0 / n, device=device, dtype=dtype)       # integers
    iy = torch.arange(n // 2 + 1, device=device, dtype=dtype)
    kx = k0 * ix[None, :, None]
    ky = k0 * iy[None, None, :]
    k2 = kx ** 2 + ky ** 2
    inv_k2 = torch.where(k2 > 0, 1.0 / k2.clamp_min(1e-30), torch.zeros_like(k2))
    cut = n / 3.0
    dealias = (ix.abs()[:, None] < cut) & (iy[None, :] < cut)
    nonzonal = (iy > 0).to(dtype)[None, :]
    return Grid(kx, ky, k2, inv_k2, dealias, nonzonal, n)


def _deriv(fh, g: Grid):
    """Real-space d_x f, d_y f from the spectrum fh."""
    n = g.n
    fx = torch.fft.irfft2(1j * g.kx * fh, s=(n, n))
    fy = torch.fft.irfft2(1j * g.ky * fh, s=(n, n))
    return fx, fy


def rhs(nh, wh, par, g: Grid):
    """Nonlinear + coupling + drive tendencies (hyperdiffusion handled by the integrating factor)."""
    ph = -wh * g.inv_k2
    phx, phy = _deriv(ph * g.dealias, g)
    nx, ny = _deriv(nh * g.dealias, g)
    wx, wy = _deriv(wh * g.dealias, g)
    br_n = torch.fft.rfft2(phx * ny - phy * nx) * g.dealias
    br_w = torch.fft.rfft2(phx * wy - phy * wx) * g.dealias
    coup = par["alpha"] * (ph - nh) * g.nonzonal
    dn = coup - br_n - par["kappa"] * 1j * g.ky * ph
    dw = coup - br_w
    return dn, dw


class HWState:
    """Spectral state for a batch of runs with per-run parameters."""

    def __init__(self, theta, cfg: HWConfig, device="cpu", dtype=torch.float32):
        self.cfg, self.dtype, self.device = cfg, dtype, device
        theta = torch.as_tensor(theta, dtype=dtype, device=device)
        self.theta = theta
        self.par = physical(theta, cfg)
        self.g = make_grid(self.par["k0"], cfg.n, device, dtype)
        kp = self.g.k2 ** cfg.p
        h = cfg.dt
        self.E_n = torch.exp(-self.par["D"] * kp * h / 2)      # half-step integrating factors
        self.E_w = torch.exp(-self.par["nu"] * kp * h / 2)
        self.nh = self.wh = None

    # --- initial conditions and conversions -------------------------------------------------
    def random_ic(self, seeds):
        n, B = self.cfg.n, self.theta.shape[0]
        fields = []
        for s in seeds:
            gen = torch.Generator().manual_seed(int(s))
            fields.append(self.cfg.ic_amp * torch.randn(2, n, n, generator=gen, dtype=torch.float64))
        f = torch.stack(fields).to(self.device, self.dtype)
        f = f - f.mean((-2, -1), keepdim=True)
        self.nh = torch.fft.rfft2(f[:, 0]) * self.g.dealias
        self.wh = torch.fft.rfft2(f[:, 1]) * self.g.dealias
        assert self.nh.shape[0] == B
        return self

    def set_fields(self, n_phi):
        """n_phi: (B, 2, n, n) real fields (n, phi) -> spectral (n, Omega) with this batch's k0."""
        f = torch.as_tensor(n_phi, device=self.device, dtype=self.dtype)
        self.nh = torch.fft.rfft2(f[:, 0])
        self.wh = -self.g.k2 * torch.fft.rfft2(f[:, 1])
        return self

    def fields(self):
        """(B, 2, n, n) real (n, phi)."""
        n = self.cfg.n
        ph = -self.wh * self.g.inv_k2
        return torch.stack([torch.fft.irfft2(self.nh, s=(n, n)), torch.fft.irfft2(ph, s=(n, n))], 1)

    # --- time stepping ----------------------------------------------------------------------
    def step(self):
        """One IF-RK4 step of size dt."""
        h, En, Ew, par, g = self.cfg.dt, self.E_n, self.E_w, self.par, self.g
        n0, w0 = self.nh, self.wh
        a_n, a_w = rhs(n0, w0, par, g)
        n1, w1 = En * (n0 + h / 2 * a_n), Ew * (w0 + h / 2 * a_w)
        b_n, b_w = rhs(n1, w1, par, g)
        n2, w2 = En * n0 + h / 2 * b_n, Ew * w0 + h / 2 * b_w
        c_n, c_w = rhs(n2, w2, par, g)
        n3, w3 = En * (En * n0 + h * c_n), Ew * (Ew * w0 + h * c_w)
        d_n, d_w = rhs(n3, w3, par, g)
        self.nh = En * En * n0 + h / 6 * (En * En * a_n + 2 * En * (b_n + c_n) + d_n)
        self.wh = Ew * Ew * w0 + h / 6 * (Ew * Ew * a_w + 2 * Ew * (b_w + c_w) + d_w)
        return self

    def advance(self, n_steps):
        for _ in range(n_steps):
            self.step()
        return self

    def finite(self):
        return torch.isfinite(self.nh).all((-2, -1)) & torch.isfinite(self.wh).all((-2, -1))


# ---------------------------------------------------------------------------------------------
# diagnostics


def flux(n_phi, k0):
    """Radial particle flux Gamma_n = -< n d_y phi > per field. n_phi: (..., 2, n, n); k0: (...)."""
    f = torch.as_tensor(n_phi)
    k0 = torch.as_tensor(k0, dtype=f.dtype, device=f.device)
    n = f.shape[-1]
    ky = torch.arange(n // 2 + 1, device=f.device, dtype=f.dtype)
    phy = torch.fft.irfft(1j * ky * torch.fft.rfft(f[..., 1, :, :], dim=-1), n=n, dim=-1)
    return -(f[..., 0, :, :] * phy).mean((-2, -1)) * k0


def energy(n_phi, k0):
    """E = 1/2 < n^2 + |grad phi|^2 >."""
    f = torch.as_tensor(n_phi)
    k0 = torch.as_tensor(k0, dtype=f.dtype, device=f.device)
    n = f.shape[-1]
    ph = torch.fft.rfft2(f[..., 1, :, :])
    ix = torch.fft.fftfreq(n, d=1.0 / n, device=f.device, dtype=f.dtype)
    iy = torch.arange(n // 2 + 1, device=f.device, dtype=f.dtype)
    gx = torch.fft.irfft2(1j * ix[:, None] * ph, s=(n, n))
    gy = torch.fft.irfft2(1j * iy[None, :] * ph, s=(n, n))
    k0 = k0[..., None, None]
    return 0.5 * (f[..., 0, :, :] ** 2 + k0 ** 2 * (gx ** 2 + gy ** 2)).mean((-2, -1))


# ---------------------------------------------------------------------------------------------
# runs and branch restarts


def simulate(theta, seeds, cfg: HWConfig, device="cpu", dtype=torch.float32, log=None,
             record_spinup_every=0):
    """Integrate runs from random initial conditions.

    Returns a dict with ``states`` (R, n_save, 2, n, n) float32 (n, phi) at times
    t_spin + j * save_dt, ``flux`` (R, n_save), ``finite`` (R,) bool, and optionally the energy
    during spin-up (to check saturation)."""
    theta = np.asarray(theta, dtype=np.float64)
    R = len(theta)
    out = torch.zeros(R, cfg.n_save, 2, cfg.n, cfg.n, dtype=torch.float32)
    fl = torch.zeros(R, cfg.n_save)
    ok = torch.ones(R, dtype=torch.bool)
    spin = []
    spin_steps = int(round(cfg.t_spin / cfg.dt))
    for b0 in range(0, R, cfg.batch):
        sl = slice(b0, min(R, b0 + cfg.batch))
        st = HWState(theta[sl], cfg, device, dtype).random_ic(seeds[sl])
        k0 = st.par["k0"].reshape(-1)
        es = []
        if record_spinup_every:
            for i in range(0, spin_steps, record_spinup_every):
                st.advance(min(record_spinup_every, spin_steps - i))
                es.append(energy(st.fields(), k0).cpu())
            spin.append(torch.stack(es, 1))
        else:
            st.advance(spin_steps)
        for j in range(cfg.n_save):
            if j:
                st.advance(cfg.steps_per_save)
            f = st.fields()
            out[sl, j] = f.float().cpu()
            fl[sl, j] = flux(f, k0).float().cpu()
        ok[sl] = st.finite().cpu() & torch.isfinite(out[sl]).flatten(1).all(1)
        if log:
            log(f"  simulated runs {sl.start}-{sl.stop - 1} / {R}")
    res = dict(states=out, flux=fl, finite=ok)
    if record_spinup_every:
        res["spinup_energy"] = torch.cat(spin)
    return res


def branch_jacobian(states, theta, cfg: HWConfig, eps=0.02, device="cpu", dtype=torch.float64,
                    batch=None):
    """True one-step sensitivity by central differences of branch restarts.

    states: (S, 2, n, n) stored (n, phi); theta: (S, d) natural coordinates.
    ``eps`` is in *standardised* units. Returns J (S, d, 2, n, n) = d s_{t+1} / d theta_std and
    the unperturbed next state (S, 2, n, n)."""
    states = torch.as_tensor(states)
    th = torch.as_tensor(np.asarray(theta), dtype=torch.float64)
    S, d = th.shape
    scale = ((torch.tensor(cfg.hi) - torch.tensor(cfg.lo)) / 2 / math.sqrt(3)).double()
    # rows: base, then +e_i, -e_i for each i
    pert = [th]
    for i in range(d):
        for sgn in (1, -1):
            t = th.clone()
            t[:, i] += sgn * eps * scale[i]
            pert.append(t)
    P = torch.stack(pert, 1).reshape(-1, d)                      # (S*(2d+1), d)
    X = states[:, None].expand(S, 2 * d + 1, *states.shape[1:]).reshape(-1, *states.shape[1:])
    batch = batch or 4 * cfg.batch
    Y = torch.empty(X.shape, dtype=torch.float64)
    for b0 in range(0, len(P), batch):
        sl = slice(b0, min(len(P), b0 + batch))
        st = HWState(P[sl].numpy(), cfg, device, dtype).set_fields(X[sl].to(dtype))
        st.advance(cfg.steps_per_save)
        Y[sl] = st.fields().cpu().double()
    Y = Y.reshape(S, 2 * d + 1, *states.shape[1:])
    J = (Y[:, 1::2] - Y[:, 2::2]) / (2 * eps)
    return J.float(), Y[:, 0].float()
