import itertools

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from unified_sbi.exact import chain_posterior, gl_grid, log_likelihood_exponent
from unified_sbi.maps import MapConfig, check_maps, fit_maps
from unified_sbi.metrics import crps_ensemble, crps_gaussian
from unified_sbi.simulators import RosenbrockChain, make_simulator
from unified_sbi.analysis import equivalent_budget, power_law_fit


def test_simulator_shapes_and_curvature():
    sim = make_simulator("rosenbrock8_hidden2d")
    theta, x = sim.simulate(5, seed=0)
    assert theta.shape == (5, 8) and x.shape == (5, 8, 15)
    assert sim.raw_dim == 120
    np.testing.assert_array_equal(sim.curvature, [1, 0, 0, 0, 0, 0, 0])
    assert make_simulator("rosenbrock8_full").curvature.sum() == 7


@pytest.mark.parametrize("curved", [(0,), "all"])
def test_chain_quadrature_matches_brute_force(curved):
    sim = RosenbrockChain(dim=3, curved_links=curved)
    theta, x = sim.simulate(1, seed=3)
    x = x.astype(float)
    post = chain_posterior(sim, x[0], n_grid=60)
    grid, w = gl_grid(60, sim.a_box)
    mesh = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1)
    lp = log_likelihood_exponent(sim, mesh, x[0].mean(0)) + sum(
        np.log(w).reshape([-1 if k == j else 1 for k in range(3)]) for j in range(3))
    joint = np.exp(lp - logsumexp(lp))
    for j in range(3):
        other = tuple(k for k in range(3) if k != j)
        np.testing.assert_allclose(post.mass_1d[j], joint.sum(other), atol=1e-12)
    m2 = post.mass_2d()
    np.testing.assert_allclose(m2[2, 0], joint.sum(1), atol=1e-12)
    assert np.isclose(post.log_z, logsumexp(lp))


def test_exact_crps_matches_sampling_from_marginal():
    sim = make_simulator("rosenbrock8_hidden2d")
    theta, x = sim.simulate(1, seed=5)
    post = chain_posterior(sim, x[0], n_grid=200)
    rng = np.random.default_rng(0)
    exact = post.crps(theta[0].astype(float))
    samples = []
    for j in range(sim.dim):   # inverse-CDF sampling of each marginal
        z = np.linspace(-sim.a_box, sim.a_box, 20001)
        samples.append(np.interp(rng.random(20000), post.cdf_1d(j, z), z))
    mc = crps_ensemble(np.stack(samples, -1)[None], theta[:1].astype(float))[0]
    np.testing.assert_allclose(exact, mc, rtol=0.03, atol=2e-3)


def test_crps_ensemble_matches_gaussian():
    rng = np.random.default_rng(1)
    y = np.array([[0.3, -1.0]])
    draws = rng.normal([0.0, 0.5], [1.0, 2.0], size=(1, 40000, 2))
    np.testing.assert_allclose(crps_ensemble(draws, y)[0],
                               crps_gaussian(np.array([0.0, 0.5]), np.array([1.0, 2.0]), y[0]), rtol=0.02)


def test_maps_invertible_after_short_fit():
    sim = make_simulator("rosenbrock8_hidden2d")
    theta, x = sim.simulate(300, seed=0)
    th, xx = torch.tensor(theta), torch.tensor(x)
    maps = fit_maps(th[:250], xx[:250], th[250:], xx[250:], MapConfig(max_steps=100, eval_every=20))
    check_maps(maps, th[:8])
    assert maps.summarise(xx).shape == (300, 8)


def test_equivalent_budget_and_power_law():
    n = np.array([100, 1000, 10000.0])
    y = np.array([1.0, 0.1, 0.01])
    a, alpha = power_law_fit(n, y)
    assert np.isclose(alpha, 1.0) and np.isclose(a, 100.0)
    # Lower-is-better curve: value 0.1 is reached at N = 1000; 0.001 never.
    value, bound = equivalent_budget(n, y, 0.1, higher_is_better=False)
    assert np.isclose(value, 1000.0) and bound == "="
    # Linear in the metric between neighbouring budgets, linear in log N.
    value, bound = equivalent_budget(n, y, 0.055, higher_is_better=False)
    assert np.isclose(value, np.sqrt(1e7)) and bound == "="
    assert equivalent_budget(n, y, 0.001, higher_is_better=False) == (10000.0, ">=")
    assert equivalent_budget(n, y, 2.0, higher_is_better=False) == (100.0, "<=")
