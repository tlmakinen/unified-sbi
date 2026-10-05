"""End-to-end: one tiny scaling cell for every arm, checking the budget accounting."""

import numpy as np

from unified_sbi.study import StudyConfig, run_cell, simulation_pool, split_indices


def tiny_config(**kw):
    base = dict(budgets=(250, 600), seeds=(0,), arms=("raw", "xbar", "theta_t", "eta_t"),
                n_discovery=500, n_test=16, n_post=64, exact_grid=80,
                maps=dict(max_steps=60, eval_every=20), npe=dict(max_epochs=60, stop_after_epochs=10))
    base.update(kw)
    return StudyConfig(**base)


def test_budget_accounting_is_nested_and_shared():
    cfg = tiny_config()
    sim = cfg.make_simulator()
    _, _, is_val = simulation_pool(sim, cfg, seed=0)
    small = split_indices(is_val, 250, 250)
    big = split_indices(is_val, 600, 500)
    # Discovery sims are reused by NPE, and a sim never changes role between budgets.
    assert set(big["disc_train"]) <= set(big["npe_train"])
    assert set(small["npe_val"]) <= set(big["npe_val"])
    assert len(big["npe_train"]) + len(big["npe_val"]) == 600
    assert max(big["disc_train"].max(), big["disc_val"].max()) < 500


def test_run_cell_all_arms(tmp_path):
    cfg = tiny_config()
    rows = run_cell(cfg, seed=0, n_total=600, out_dir=tmp_path, verbose=False)
    assert [r["arm"] for r in rows] == list(cfg.arms)
    for r in rows:
        assert np.isfinite([r["val_log_prob"], r["test_log_prob"], r["crps"]]).all()
        assert r["n_npe_train"] + r["n_npe_val"] == 600
        assert r["n_discovery"] == (500 if r["arm"] in ("theta_t", "eta_t") else 0)
    assert (tmp_path / "oracle.json").exists()
    assert len(list((tmp_path / "cache").glob("maps_*.pt"))) == 1
    # Second call at the same discovery size reuses the cached maps.
    rows2 = run_cell(cfg, seed=0, n_total=600, out_dir=tmp_path, verbose=False)
    assert rows2[2]["map_seconds"] == 0.0
