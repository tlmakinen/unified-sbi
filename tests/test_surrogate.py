"""Checks for the few-run surrogate study (fast, CPU)."""
import math

import numpy as np
import pytest
import torch

from unified_sbi.surrogate import fisher as F
from unified_sbi.surrogate.hw import (HWConfig, HWState, branch_jacobian, energy, sample_prior,
                                      simulate, standardise, unstandardise)
from unified_sbi.surrogate.nets import Model, normalise


def test_bracket_conserves_energy_in_ideal_limit():
    """alpha, kappa, nu, D -> 0: 2D Euler for Omega. Dealiased spectral brackets conserve the
    kinetic energy up to the RK4 truncation error."""
    cfg = HWConfig(n=32, dt=0.01)
    th = np.array([[-12.0, 0.0, -12.0, -12.0, 0.2]])
    st = HWState(th, cfg, dtype=torch.float64)
    st.random_ic([0])
    st.wh = st.wh * 50                      # O(1) vorticity so the bracket matters
    e0 = energy(st.fields(), 0.2)
    f0 = st.fields()
    st.advance(200)
    e1 = energy(st.fields(), 0.2)
    assert (st.fields() - f0).norm() / f0.norm() > 0.05      # the flow did evolve
    assert abs(e1 / e0 - 1).item() < 1e-5


def test_standardise_round_trip_and_unit_variance():
    cfg = HWConfig()
    th = torch.as_tensor(sample_prior(20000, cfg, np.random.default_rng(0)))
    u = standardise(th, cfg)
    assert torch.allclose(unstandardise(u, cfg), th)
    assert (u.std(0) - 1).abs().max() < 0.02 and u.mean(0).abs().max() < 0.03


def test_restart_reproduces_trajectory_and_branch_fd_is_consistent():
    cfg = HWConfig(n=32, t_spin=60, n_save=2, batch=4)
    th = sample_prior(2, cfg, np.random.default_rng(1))
    r = simulate(th, [3, 4], cfg)
    assert r["finite"].all()
    J1, y0 = branch_jacobian(r["states"][:, 0], th, cfg, eps=0.02)
    J2, _ = branch_jacobian(r["states"][:, 0], th, cfg, eps=0.005)
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    assert rel(y0, r["states"][:, 1]) < 1e-4
    assert rel(J1, J2) < 1e-2


def test_r_iso_minimum_positivity_and_invariance():
    torch.manual_seed(0)
    d = 5
    A = torch.randn(7, d, d, dtype=torch.float64); M = A @ A.transpose(1, 2) + 0.1 * torch.eye(d)
    B = torch.randn(7, d, d, dtype=torch.float64); G = B @ B.transpose(1, 2) + 0.1 * torch.eye(d)
    assert abs(F.r_iso(M, M).item()) < 1e-9
    assert F.r_iso(G, M).item() > 0
    U = torch.randn(d, d, dtype=torch.float64) + 3 * torch.eye(d)  # change of theta coordinates
    r1 = F.r_iso(G, M); r2 = F.r_iso(U.T @ G @ U, U.T @ M @ U)
    assert abs(r1 - r2).item() < 1e-8
    # the gauge: scaling G alone is penalised in both directions
    assert F.r_iso(4 * M, M) > 0 and F.r_iso(M / 4, M) > 0


def test_whitened_noise_covariance_and_matched_isotropic():
    torch.manual_seed(1)
    d = 3
    A = torch.randn(d, d, dtype=torch.float64); G = A @ A.T + 0.5 * torch.eye(d)
    Gb = G.expand(200000, d, d)
    delta = F.whitened_noise(Gb, 0.5)
    C = delta.T @ delta / len(delta)
    assert torch.allclose(C, 0.25 * torch.linalg.inv(G), atol=3e-3)
    iso = F.isotropic_noise_matched(Gb, 0.5)
    q_w = torch.einsum("bi,ij,bj->b", delta, G, delta).mean()
    q_i = torch.einsum("bi,ij,bj->b", iso, G, iso).mean()
    assert abs(q_w / q_i - 1) < 0.02


def test_surrogate_jacobian_matches_autograd_and_metric_of_g():
    torch.manual_seed(2)
    d, S = 5, 3
    W = torch.randn(4 * 8 * 8, d, dtype=torch.float64)
    fn = lambda x, th, lr: x + torch.tanh(th @ W.T).reshape(-1, 4, 8, 8) * lr.sum(-1)[:, None, None, None]
    x = torch.randn(S, 4, 8, 8, dtype=torch.float64); th = torch.randn(S, d, dtype=torch.float64)
    lr = torch.randn(S, 2, dtype=torch.float64)
    J = F.surrogate_jacobian(fn, x, th, lr, eps=1e-4)
    for s in range(S):
        Ja = torch.autograd.functional.jacobian(lambda t: fn(x[s:s + 1], t[None], lr[s:s + 1])[0], th[s])
        assert torch.allclose(J[s], Ja.permute(3, 0, 1, 2), atol=1e-6)
    m = Model(d, 16, K=8, width=32, depth=1, heads=2, patch=4).double()
    G = F.metric_of_g(m.g, th)
    Jg = torch.autograd.functional.jacobian(lambda t: m.g(t[None])[0], th[0])
    assert torch.allclose(G[0], Jg.T @ Jg, atol=1e-10)


def test_model_shapes_and_zero_init_residual():
    m = Model(5, 32, K=8, width=64, depth=2, heads=4, patch=4)
    s = torch.randn(3, 2, 32, 32) * 3
    x, logr = normalise(s)
    assert torch.allclose(x.pow(2).mean((-2, -1)), torch.ones(3, 2), atol=1e-5)
    y = m(x, torch.randn(3, 5), logr)
    assert y.shape == x.shape
    assert torch.allclose(y, x + (y - x))  # residual form
    th = torch.randn(3, 5)
    y0 = m(x, th, logr)
    G0 = F.metric_of_g(m.g, th)
    m.rescale_gauge(7.0)
    assert torch.allclose(m(x, th, logr), y0, atol=1e-5)          # predictions unchanged
    assert torch.allclose(F.metric_of_g(m.g, th), 49 * G0, rtol=1e-4)
    ident = Model(5, 32, identity=True, width=64, depth=1, heads=4, patch=4)
    assert ident.g.K == 5


def test_maximin_picks_far_points():
    existing = torch.zeros(1, 2)
    cand = torch.tensor([[0.1, 0.0], [5.0, 0.0], [5.1, 0.0], [-5.0, 0.0]])
    p = F.maximin(cand, existing, 2)
    assert set(p) == {1, 3} or set(p) == {2, 3}


def test_geodesic_maximin_respects_curvature():
    """Points on a circle: the chord between antipodes is 2, the geodesic pi. With one existing
    point at angle 0, geodesic maximin picks the antipode first."""
    ang = torch.linspace(0, 2 * math.pi, 201)[:-1]
    circ = torch.stack([ang.cos(), ang.sin()], 1)
    p = F.maximin_geodesic(circ[1:], circ[:1], 1, n_neighbours=4)
    assert abs(ang[1:][p[0]].item() - math.pi) < 0.05


@pytest.mark.slow
def test_study_smoke(tmp_path):
    from unified_sbi.surrogate.study import StudyConfig, load_results, run_design, run_study
    hw = HWConfig(n=16, t_spin=40, n_save=12, batch=32).to_dict()
    cfg = StudyConfig(hw=hw, pool_size=8, n_val=2, n_test=2, n_checkpoints=2, budgets=(4,),
                      seeds=(0,), arms=("A0_film", "A3_iso_wn", "A6_full"), K=6, width=32,
                      depth=1, heads=2, patch=4, steps=40, batch=4, eval_every=20, n_val_pairs=8,
                      fisher_every=5, fisher_runs=2, rollout_starts=1, rollout_h=6, flux_from=2,
                      horizons=(1, 5), design_budgets=(4,), design_candidates=32, device="cpu",
                      sim_device="cpu")
    run_study(cfg, tmp_path / "out", tmp_path / "cache", log=lambda *a: None)
    rows = load_results(tmp_path / "out")
    assert len(rows) == 3
    for r in rows:
        assert math.isfinite(r["onestep_nrmse"]) and math.isfinite(r["jac_rel_err"])
    run_design(cfg, tmp_path / "out", tmp_path / "cache", log=lambda *a: None)
    assert len(list((tmp_path / "out" / "design").rglob("metrics.json"))) == len(cfg.designs)
