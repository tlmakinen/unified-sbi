import numpy as np
import pytest
import torch

from unified_sbi.lowrank.maps import (LatentGaussian, StructuredMapConfig, StructuredParameterMap,
                                      cholesky_from_extras, fit_structured_maps, max_extra,
                                      check_structured_maps)
from unified_sbi.lowrank.reference import analytic_normaliser_check, converged_reference
from unified_sbi.lowrank.screen import candidate_grid, select_one_se
from unified_sbi.lowrank.simulator import HiddenRosenbrock
from unified_sbi.lowrank.study import LowRankConfig, run_cell


def test_simulator_and_sufficient_statistic():
    sim = HiddenRosenbrock()
    theta, x = sim.simulate(6, seed=0)
    assert theta.shape == (6, 8) and x.shape == (6, 4, 8) and sim.sufficient(x).shape == (6, 3)
    assert np.allclose(sim.Q @ sim.Q.T, np.eye(8)) and np.allclose(sim.O @ sim.O.T, np.eye(8))


def test_reference_normaliser_matches_independent_1d_integral():
    sim = HiddenRosenbrock()
    _, x = sim.simulate(3, seed=1)
    for s in sim.sufficient(x):
        ref = converged_reference(sim, s)
        assert analytic_normaliser_check(sim, s, ref["log_z"]) < 1e-4


@pytest.mark.parametrize("k", [1, 2, 3, 8])
def test_structured_map_invertible_with_exact_logdet(k):
    torch.manual_seed(0)
    phi = StructuredParameterMap(8, k)
    with torch.no_grad():
        phi.rotation_raw.normal_(0, 0.3)
        for p in phi.flow.parameters():
            p.normal_(0, 0.1)
    th = torch.randn(5, 8)
    eta, ld = phi(th)
    back, ild = phi.inverse(eta)
    assert torch.allclose(back, th, atol=1e-5)
    jac = torch.autograd.functional.jacobian(lambda u: phi(u)[0], th[0])
    assert torch.allclose(torch.linalg.slogdet(jac)[1], ld[0], atol=1e-5)
    assert torch.allclose(phi.rotation() @ phi.rotation().T, torch.eye(8), atol=1e-5)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_latent_gaussian_matches_torch_mvn(k):
    torch.manual_seed(1)
    for n_extra in range(max_extra(k) + 1):
        t = torch.randn(4, k + n_extra)
        lat = LatentGaussian(t, k)
        z = torch.randn(4, k)
        mvn = torch.distributions.MultivariateNormal(t[:, :k], scale_tril=lat.L)
        assert torch.allclose(lat.log_prob(z), mvn.log_prob(z), atol=1e-5)


def test_candidate_grid_and_one_se_rule():
    assert candidate_grid([1, 2], "standard") == [(1, 1), (1, 2), (2, 2), (2, 4), (2, 5)]
    assert candidate_grid([2], "all") == [(2, 2), (2, 3), (2, 4), (2, 5)]
    rng = np.random.default_rng(0)
    base = rng.normal(size=200)
    per = {(2, 2): base + 1.0 + 0.3 * rng.normal(size=200),      # clearly worse
           (2, 3): base + 0.01 + 0.3 * rng.normal(size=200),     # ties the best within 1 paired SE
           (2, 5): base}
    import pandas as pd
    table = pd.DataFrame([dict(k=k, m=m, val_nll=v.mean()) for (k, m), v in per.items()])
    assert select_one_se(table, per) == (2, 3)


def test_short_fit_prefers_true_plane_over_k1():
    sim = HiddenRosenbrock()
    theta, x = sim.simulate(1500, seed=2)
    th, xx = torch.tensor(theta), torch.tensor(x)
    cfg = StructuredMapConfig(max_steps=800, eval_every=50)
    fits = {k: fit_structured_maps(th[:1200], xx[:1200], th[1200:], xx[1200:], k, k, cfg) for k in (1, 2)}
    check_structured_maps(fits[2], th[:8])
    assert fits[2].best_val_nll < fits[1].best_val_nll


def test_run_cell_smoke(tmp_path):
    cfg = LowRankConfig(budgets=(600,), seeds=(0,), n_test=12, n_post=64, n_reference=128,
                        k_values=(1, 2), m_mode="identity",
                        maps=dict(max_steps=120, eval_every=20), npe=dict(max_epochs=15, stop_after_epochs=5))
    rows, screen = run_cell(cfg, seed=0, n_total=600, out_dir=tmp_path, verbose=False)
    assert {r["arm"] for r in rows} == set(cfg.arms)
    assert set(zip(screen.k, screen.m)) == {(1, 1), (2, 2)} and screen.selected.sum() == 1
    for r in rows:
        assert np.isfinite([r["posterior_kl"], r["crps"], r["sw_active"], r["val_log_prob"]]).all()
    learned = [r for r in rows if r["arm"] in ("summary", "eta", "structured")]
    assert all(r["n_npe_train"] + r["n_npe_val"] == 600 - 240 for r in learned)   # disjoint by default
    assert (tmp_path / "reference.npz").exists()
