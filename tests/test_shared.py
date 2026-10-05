"""Checks for the shared-space lensing study (fast, CPU)."""
import math

import numpy as np
import pytest
import torch

from unified_sbi.shared.heads import SharedHead
from unified_sbi.shared.lensing import HI, LO, LensingConfig, LensingSimulator
from unified_sbi.shared.nets import CrossSpectrum, ParamEmbed
from unified_sbi.shared.quadrature import integrate

LOt, HIt = torch.tensor(LO), torch.tensor(HI)


def test_quadrature_truncated_gaussians():
    torch.set_default_dtype(torch.float64)
    try:
        mu = torch.tensor([[0.3, 0.8], [0.16, 0.36]]); sd = torch.tensor([[0.01, 0.002], [0.02, 0.01]])
        logf = lambda p: (-0.5 * ((p - mu[:, None]) / sd[:, None]) ** 2 - torch.log(sd[:, None] * math.sqrt(2 * math.pi))).sum(-1)
        c = integrate(logf, LOt, HIt, 2, n0=48, n=24, levels=2, shared_level0=False)
        from scipy.stats import norm
        expect = norm.sf((0.15 - 0.16) / 0.02) * norm.sf((0.35 - 0.36) / 0.01)
        assert abs(c.logZ[0].exp().item() - 1) < 1e-4
        assert abs(c.logZ[1].exp().item() - expect) < 1e-3
    finally:
        torch.set_default_dtype(torch.float32)


def test_simulator_power_and_exact_loglik():
    cfg = LensingConfig(n=32)
    sim = LensingSimulator(cfg)
    rng = np.random.default_rng(0)
    th = np.array([[0.3, 0.8]])
    x = sim.simulate(np.repeat(th, 200, 0), rng)
    st = sim.sufficient_stats(x) / np.maximum(sim.geo["counts"], 1)[None, :, None, None]
    S = sim.signal_sigma(th)[0]
    r = st.mean(0)[3:12].diagonal(axis1=-2, axis2=-1) / S[3:12].diagonal(axis1=-2, axis2=-1)
    assert np.abs(r - 1).max() < 0.1
    # exact log-likelihood equals a dense multivariate-normal log density (up to a constant)
    xo = sim.add_noise(x[:1], rng).astype(np.float64)
    thetas = np.array([[0.3, 0.8], [0.5, 0.6], [0.2, 1.1]])
    ll = sim.loglik_from_stats(sim.sufficient_stats(xo)[0], thetas)
    # dense covariance of the flattened map via the Fourier construction
    n, nb = cfg.n, cfg.n_bins
    dense = []
    for t in thetas:
        Sig = sim.signal_sigma(t[None])[0][sim.geo["ann"]]                  # (n, n, nb, nb)
        xi = np.fft.ifft2(np.moveaxis(Sig, (0, 1), (-2, -1))).real          # (nb, nb, n, n)
        idx = np.arange(n)
        dx = (idx[:, None] - idx[None, :]) % n
        C = xi[:, :, dx[:, None, :, None], dx[None, :, None, :]]           # (nb, nb, n, n, n, n)
        C = np.transpose(C, (0, 2, 3, 1, 4, 5)).reshape(nb * n * n, nb * n * n)
        C += sim.sigma_n ** 2 * np.eye(len(C))
        sign, ld = np.linalg.slogdet(C)
        v = xo[0].reshape(-1)
        dense.append(-0.5 * (v @ np.linalg.solve(C, v) + ld))
    dense = np.array(dense)
    np.testing.assert_allclose(ll - ll[0], dense - dense[0], rtol=1e-6, atol=1e-5)


def test_cross_spectrum_shape():
    cs = CrossSpectrum(32, 4, 6)
    assert cs(torch.randn(3, 4, 32, 32)).shape == (3, 60)


def test_log_volume_matches_autograd():
    torch.manual_seed(0)
    for coords in ("box", "logit"):
        e = ParamEmbed(4, LO, HI, coords=coords).double()
        with torch.no_grad():
            for p in e.net.parameters():
                p.add_(0.3 * torch.randn_like(p))
        th = torch.tensor([[0.3, 0.8], [0.6, 1.4]], dtype=torch.float64)
        z, lv = e.log_volume(th)
        J = torch.stack([torch.autograd.functional.jacobian(e, th[i:i + 1])[0, :, 0] for i in range(2)])
        expect = 0.5 * torch.logdet(J.transpose(-1, -2) @ J)
        torch.testing.assert_close(lv, expect, rtol=1e-6, atol=1e-6)


def test_shared_head_is_normalised_and_rect_is_not_bounded():
    torch.manual_seed(0)
    head = SharedHead(4, LO, HI).double()
    with torch.no_grad():
        for p in head.eta.parameters():
            p.mul_(8)                    # sharp, curved embedding
    t = torch.randn(3, 4, dtype=torch.float64) * 3
    fine = integrate(lambda p: head.log_f(p, t), head.lo.double(), head.hi.double(), 3, n0=128, n=64, levels=3)
    coarse = head.cells(t)
    torch.testing.assert_close(coarse.logZ, fine.logZ, atol=2e-3, rtol=0)
    # helix: the rectangular loss decreases without bound while the normalised NLL does not
    lo, hi = torch.tensor([-1.0, -1.0], dtype=torch.float64), torch.tensor([1.0, 1.0], dtype=torch.float64)
    t = torch.tensor([[0.6, 0.0, 0.0]], dtype=torch.float64)
    th = torch.tensor([[0.2, 0.3]], dtype=torch.float64)
    rect, norm = [], []
    for w in (1.0, 16.0, 256.0):
        def logf(p, w=w):
            z = torch.stack([3 * p[..., 0], torch.cos(w * p[..., 1]), torch.sin(w * p[..., 1])], -1)
            return -0.5 * (z - t[:, None]).square().sum(-1) - 1.5 * math.log(2 * math.pi) + math.log(3 * w)
        lf = logf(th[:, None])[:, 0]
        rect.append(-lf.item())
        norm.append(-(lf - integrate(logf, lo, hi, 1, n0=64, n=64, levels=2).logZ).item())
    assert rect[0] > rect[1] > rect[2] and rect[0] - rect[2] > 5
    assert max(norm) - min(norm) < 0.05


# ------------------------------------------------------------ stop-gradient emulator study

def test_hyvarinen_helix_is_frequency_invariant():
    """Plain theta, no prior: eta = (a th1, R cos w th2, R sin w th2), t = (a th1_hat, 0, 0).
    Per-sample loss = 1/2 a^4 (th1 - th1_hat)^2 - a^2 for every w (curvature cancels tr J^T J)."""
    from unified_sbi.shared.heads import hyvarinen_loss
    a, R, th1_hat = 3.0, 2.0, 0.4
    y = torch.tensor([[0.1, 0.3], [-0.5, 0.9], [0.7, -0.2]], dtype=torch.float64)
    t = torch.tensor([[a * th1_hat, 0.0, 0.0]], dtype=torch.float64).expand(3, 3)
    expect = 0.5 * a ** 4 * (y[:, 0] - th1_hat) ** 2 - a ** 2
    for w in (1.0, 16.0, 256.0):
        f = lambda p, w=w: torch.stack([a * p[:, 0], R * torch.cos(w * p[:, 1]), R * torch.sin(w * p[:, 1])], -1)
        L, _ = hyvarinen_loss(f, y, t)
        torch.testing.assert_close(L, expect, rtol=1e-9, atol=1e-6)


def test_hyvarinen_prior_score_matches_autograd():
    """Logit-coordinate prior score: d/dy log p(y) for u = sigmoid(s y) ~ U(0, 1)."""
    from unified_sbi.shared.heads import LOGIT_SCALE, logit_prior_score
    y = torch.linspace(-3, 3, 7, dtype=torch.float64).requires_grad_(True)
    s = LOGIT_SCALE
    logp = (torch.log(torch.sigmoid(s * y)) + torch.log(torch.sigmoid(-s * y))).sum()
    g = torch.autograd.grad(logp, y)[0]
    torch.testing.assert_close(logit_prior_score(y), g)


def test_emulator_term_stops_gradient_to_t():
    """L_emu reaches only the emulator tower: not t, and (gmi) not the GMI eta tower."""
    from unified_sbi.shared.study import Pipeline, StudyConfig
    torch.manual_seed(0)
    cfg = StudyConfig()
    for arm in ("hybrid_quad_b0.1", "hybrid_hyv_b0.1", "hybrid_nce_b0.1", "hybrid_gmi"):
        model = Pipeline(arm, cfg, 32, np.full((4, 1, 1), 0.01, np.float32))
        model.fit_preprocessing(torch.randn(16, 4, 32, 32))
        model.train()
        th = torch.as_tensor(LO + (HI - LO) * np.random.default_rng(0).random((8, 2)), dtype=torch.float32)
        t = model.t(torch.randn(8, 4, 32, 32))
        emu_tower = model.emulator_eta()
        model.head.emu(emu_tower(th), t).mean().backward()
        frozen = list(model.summary.named_parameters())
        if arm == "hybrid_gmi":
            frozen += list(model.head.eta.named_parameters())
        for name, p in frozen:
            assert p.grad is None or torch.all(p.grad == 0), (arm, name)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in emu_tower.parameters())
        # and the full training loss does reach t (the information term)
        model.zero_grad()
        model.head.beta = 0.5
        model.loss(th, torch.randn(8, 4, 32, 32)).mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.summary.parameters())


def test_infonce_limit_matches_quadrature_logZ():
    """log (1/B) sum_j exp c(eta(theta_j), t), theta_j ~ prior, -> log Z(t) - log|Theta|."""
    torch.manual_seed(1)
    head = SharedHead(4, LO, HI).double().eval()
    with torch.no_grad():
        for p in head.eta.parameters():
            p.mul_(2)
    t = head.eta(torch.tensor([[0.3, 0.8], [0.5, 1.2]], dtype=torch.float64)) + 0.3 * torch.randn(2, 4, dtype=torch.float64)
    B = 10_000
    th = LOt.double() + (HIt - LOt).double() * torch.rand(B, 2, dtype=torch.float64)
    with torch.no_grad():
        lf = head.log_f(th[None], t)                                           # (2, B)
        mc = torch.logsumexp(lf, -1) - math.log(B)
        w = (lf - lf.max(-1, keepdim=True).values).exp()
        se = (w.std(-1) / w.mean(-1)) / math.sqrt(B)                           # delta-method SE of log mean
        quad = head.cells(t).logZ - math.log(float(((HIt - LOt).prod())))
    assert torch.all((mc - quad).abs() < 4 * se + 1e-3), (mc, quad, se)


def test_infonce_loss_form():
    from unified_sbi.shared.heads import infonce_loss
    torch.manual_seed(0)
    eta, t = torch.randn(5, 3), torch.randn(5, 3)
    c = -0.5 * torch.cdist(t, eta) ** 2
    expect = -c.diagonal() + torch.logsumexp(c, -1) - math.log(5)
    torch.testing.assert_close(infonce_loss(eta, t), expect, rtol=1e-5, atol=1e-5)


def test_gmi_linear_gaussian_optimum():
    """theta ~ N(0,1), x = theta + sigma eps, eta = theta, t = c x: min_c L_GMI = 1/2 log(sigma^2/(1+sigma^2))."""
    from scipy.optimize import minimize_scalar
    from unified_sbi.shared.heads import gmi_loss
    g = torch.Generator().manual_seed(0)
    B = 100_000
    for sigma in (0.3, 1.0, 3.0):
        th = torch.randn(B, 1, generator=g, dtype=torch.float64)
        x = th + sigma * torch.randn(B, 1, generator=g, dtype=torch.float64)
        f = lambda c: gmi_loss(th, c * x, eps=0.0).item()
        r = minimize_scalar(f, bounds=(0.0, 2.0), method="bounded", options=dict(xatol=1e-8))
        expect = 0.5 * math.log(sigma ** 2 / (1 + sigma ** 2))
        # MC error of 1/2 log of a variance ratio with 1e5 samples is ~ 1e-2 at most
        assert abs(r.fun - expect) < 0.01, (sigma, r.fun, expect)
        assert abs(r.x - 1 / (1 + sigma ** 2)) < 0.02


def test_gmi_helix_invariance():
    """A coil in a direction the data don't constrain contributes log 1 = 0: L_GMI equal across w."""
    from unified_sbi.shared.heads import gmi_loss
    g = torch.Generator().manual_seed(1)
    B, a, R, sigma = 100_000, 2.0, 1.5, 0.5
    th = 2 * torch.rand(B, 2, generator=g, dtype=torch.float64) - 1
    x = th[:, :1] + sigma * torch.randn(B, 1, generator=g, dtype=torch.float64)   # x carries theta_1 only
    vals = []
    for w in (1.0, 16.0, 256.0):
        eta = torch.stack([a * th[:, 0], R * torch.cos(w * th[:, 1]), R * torch.sin(w * th[:, 1])], -1)
        X = torch.cat([torch.ones(B, 1, dtype=torch.float64), x], 1)
        coef = torch.linalg.lstsq(X, eta).solution                                    # best linear predictor
        vals.append(gmi_loss(eta, X @ coef, eps=0.0).item())
    expect = 0.5 * math.log(1 - (1 / 3) / (1 / 3 + sigma ** 2))                        # theta_1 ~ U(-1, 1)
    assert max(vals) - min(vals) < 0.02, vals
    assert abs(vals[0] - expect) < 0.02, (vals, expect)


def test_hyvarinen_projected_residual_is_bounded_under_scaling():
    """Under (eta, t) -> s (eta, t): 1/2||eta - t||^2 - tr Lambda runs to -inf; the proper
    (projected-residual) Hyvarinen loss 1/2||J^T(eta - t)||^2 - tr Lambda does not."""
    from unified_sbi.shared.heads import hyvarinen_loss
    torch.manual_seed(0)
    f0 = lambda y: torch.stack([y[:, 0] + 0.3 * y[:, 1] ** 2, y[:, 1], torch.sin(y[:, 0]), 0.5 * y[:, 0] * y[:, 1]], -1)
    y = torch.randn(64, 2, dtype=torch.float64)
    t0 = f0(y).detach() + 0.3 * torch.randn(64, 4, dtype=torch.float64)
    full, proper = [], []
    for s in (1.0, 10.0, 100.0):
        f = lambda z, s=s: s * f0(z)
        L, z = hyvarinen_loss(f, y, s * t0)                       # 1/2||J^T(t - eta)||^2 + div, div = -tr Lambda
        with torch.enable_grad():
            yy = y.clone().requires_grad_(True)
            sc = torch.autograd.grad((-0.5 * (f(yy) - s * t0).square().sum(-1)).sum(), yy, create_graph=True)[0]
        L_full = L - 0.5 * sc.square().sum(-1) + 0.5 * (z - s * t0).square().sum(-1)
        full.append(L_full.mean().item()); proper.append(L.mean().item())
    assert full[0] > full[1] > full[2] and full[2] < -1e3, full
    assert proper[2] > proper[1] and proper[2] > 0, proper
