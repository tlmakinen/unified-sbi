"""Scaling study for the rank-deficient benchmark, with an automatic (k, m) screen.

Budget accounting (all at one total budget N):

* Each seed draws one pool; budget N uses its first N simulations (nested budgets).
* Each simulation is marked validation with probability ``val_fraction``, fixed per seed, and
  keeps that role across budgets and stages.
* Discovery uses the first n_disc simulations: ``fraction * N`` (default 0.4, as in the source
  notebook) or ``min(fixed, N)``. The screen (or fixed-(k, m) fit) uses their training members
  and is scored on their validation members.
* ``reuse=False`` (default, as in the source notebook): map-based arms train NPE on the
  remaining N - n_disc simulations only. ``reuse=True``: on all N.
* raw, mean and oracle use all N. gaussian uses only the n_disc discovery simulations; prior uses 0.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..npe import NPEConfig
from ..utils import atomic_torch_save, atomic_write_json, seed_everything, to_tensor
from .arms import ALL_ARMS, USES_MAPS, Arm
from .diagnostics import evaluate, reference_crps
from .maps import (StructuredMapConfig, check_structured_maps, fit_structured_maps,
                   load_structured_maps)
from .reference import ExactReference
from .screen import candidate_grid, run_screen
from .simulator import HiddenRosenbrock

DEFAULT_BUDGETS = (250, 500, 1000, 2000, 5000, 10000)


@dataclass
class LowRankConfig:
    simulator: dict = field(default_factory=dict)       # HiddenRosenbrock overrides
    budgets: tuple = DEFAULT_BUDGETS
    seeds: tuple = (0, 1, 2)
    arms: tuple = ALL_ARMS
    discovery_mode: str = "fraction"                    # "fraction" | "fixed"
    discovery_value: float = 0.4                        # fraction of N, or a fixed count
    reuse: bool = False
    val_fraction: float = 0.15
    selection: str = "screen"                           # "screen" | "fixed"
    fixed_k: int = 2
    fixed_m: int = 2
    k_values: tuple = (1, 2, 3, 4)
    m_mode: str = "standard"                            # see screen.candidate_grid
    m_values: tuple | None = None
    pool_size: int | None = None
    n_test: int = 500
    n_post: int = 500
    n_reference: int = 1000
    test_seed: int = 9001
    maps: StructuredMapConfig = field(default_factory=StructuredMapConfig)
    npe: NPEConfig = field(default_factory=NPEConfig)
    device: str = "cpu"

    def __post_init__(self):
        if isinstance(self.maps, dict):
            self.maps = StructuredMapConfig(**self.maps)
        if isinstance(self.npe, dict):
            self.npe = NPEConfig(**self.npe)
        self.budgets = tuple(int(b) for b in self.budgets)
        self.seeds = tuple(int(s) for s in self.seeds)
        self.arms = tuple(self.arms)
        self.k_values = tuple(int(k) for k in self.k_values)
        if self.m_values is not None:
            self.m_values = tuple(int(m) for m in self.m_values)
        unknown = set(self.arms) - set(ALL_ARMS)
        if unknown:
            raise ValueError(f"Unknown arms {unknown}; options {ALL_ARMS}")
        if self.discovery_mode not in ("fraction", "fixed") or self.selection not in ("screen", "fixed"):
            raise ValueError("discovery_mode must be fraction|fixed and selection screen|fixed.")
        if self.pool_size is None:
            self.pool_size = max(self.budgets)

    def make_simulator(self) -> HiddenRosenbrock:
        return HiddenRosenbrock(**self.simulator)

    def n_discovery(self, n_total: int) -> int:
        if self.discovery_mode == "fraction":
            return int(self.discovery_value * n_total)
        return int(min(self.discovery_value, n_total))

    def grid(self):
        if self.selection == "fixed":
            return [(self.fixed_k, self.fixed_m)]
        return candidate_grid(self.k_values, self.m_mode, self.m_values)

    def tasks(self):
        return [(s, b) for s in self.seeds for b in self.budgets]

    def to_dict(self):
        d = asdict(self)
        for key in ("budgets", "seeds", "arms", "k_values"):
            d[key] = list(d[key])
        d["m_values"] = list(self.m_values) if self.m_values is not None else None
        return d


# ------------------------------------------------------------------------ data and reference
def simulation_pool(sim, cfg, seed):
    theta, x = sim.simulate(cfg.pool_size, seed=1000 * seed + 11)
    is_val = np.random.default_rng(1000 * seed + 12).random(cfg.pool_size) < cfg.val_fraction
    return theta, x, is_val


def load_reference(sim, cfg, out_dir: Path | None) -> ExactReference:
    path = Path(out_dir) / "reference.npz" if out_dir else None
    if path and path.exists():
        z = np.load(path)
        if z["theta"].shape[0] == cfg.n_test and z["samples"].shape[1] == cfg.n_reference:
            grids = [dict(mean=m) for m in z["grid_means"]]
            return ExactReference(sim, z["theta"], z["x"], grids, z["true_log_prob"], z["samples"])
    theta, x = sim.simulate(cfg.n_test, seed=cfg.test_seed)
    ref = ExactReference.build(sim, theta, x, cfg.n_reference)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, theta=ref.theta, x=ref.x, true_log_prob=ref.true_log_prob,
                            samples=ref.samples, grid_means=np.stack([g["mean"] for g in ref.grids]))
        tmp.replace(path)
    return ref


# ------------------------------------------------------------------------ discovery
def _cache_key(sim, cfg, seed, n_disc) -> str:
    blob = json.dumps(dict(sim=sim.to_dict(), maps=cfg.maps.to_dict(), grid=cfg.grid(), seed=seed,
                           n_disc=n_disc, pool=cfg.pool_size, val=cfg.val_fraction), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def discovery(sim, cfg, seed, theta_t, x_t, is_val, n_disc, cache_dir: Path | None, verbose=True):
    """Screen (or fixed fit) on the discovery simulations. Returns (maps, screen table, cached)."""
    idx = np.arange(n_disc)
    tr, va = idx[~is_val[:n_disc]], idx[is_val[:n_disc]]
    if len(va) < 2 or len(tr) < 2:
        raise ValueError(f"Discovery split too small ({len(tr)} train / {len(va)} val).")
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"screen_seed{seed}_n{n_disc}_{_cache_key(sim, cfg, seed, n_disc)}.pt"
        if path.exists():
            state = torch.load(path, map_location=cfg.device, weights_only=False)
            maps = load_structured_maps(state["maps"], sim.dim, x_t[tr], cfg.maps)
            return maps, pd.DataFrame(state["table"]), True
    true_rows = sim.Q[:2]
    if cfg.selection == "screen":
        if verbose:
            print(f"  screening {len(cfg.grid())} (k, m) candidates on {len(tr)} + {len(va)} sims")
        res = run_screen(theta_t[tr], x_t[tr], theta_t[va], x_t[va], cfg.grid(), cfg.maps,
                         seed=seed, true_rows=true_rows, verbose=verbose)
        maps, table = res.selected_maps(), res.table
    else:
        maps = fit_structured_maps(theta_t[tr], x_t[tr], theta_t[va], x_t[va],
                                   cfg.fixed_k, cfg.fixed_m, cfg.maps, seed=seed)
        table = pd.DataFrame([dict(k=maps.k, m=maps.m, val_nll=maps.best_val_nll, selected=True)])
    check_structured_maps(maps, theta_t[tr[:8]])
    if path is not None:
        atomic_torch_save(dict(maps=maps.state(), table=table.to_dict("list")), path)
    return maps, table, False


# ------------------------------------------------------------------------ one cell
def run_cell(cfg: LowRankConfig, seed: int, n_total: int, out_dir: Path | None = None,
             verbose: bool = True, return_artifacts: bool = False):
    device = cfg.device
    seed_everything(seed)
    sim = cfg.make_simulator()
    theta, x, is_val = simulation_pool(sim, cfg, seed)
    theta_t, x_t = to_tensor(theta[:n_total], device), to_tensor(x[:n_total], device)
    val = is_val[:n_total]
    ref = load_reference(sim, cfg, out_dir)
    th_test, x_test = to_tensor(ref.theta, device), to_tensor(ref.x, device)
    floor = reference_crps(ref)

    n_disc = cfg.n_discovery(n_total)
    maps, table, cached = None, None, False
    if any(USES_MAPS[a] for a in cfg.arms):
        maps, table, cached = discovery(sim, cfg, seed, theta_t, x_t, is_val, n_disc,
                                        Path(out_dir) / "cache" if out_dir else None, verbose)
        if verbose:
            print(f"[seed {seed} | N={n_total}] discovery on {n_disc} sims "
                  f"({'cached' if cached else 'fitted'}): selected k={maps.k}, m={maps.m}")

    idx = np.arange(n_total)
    all_tr, all_va = idx[~val], idx[val]
    learned_idx = idx if cfg.reuse else idx[n_disc:]
    lr_tr, lr_va = learned_idx[~val[learned_idx]], learned_idx[val[learned_idx]]

    rows, artifacts = [], dict(maps=maps, screen=table, ref=ref, samples={}, delta={})
    for arm_name in cfg.arms:
        uses = USES_MAPS[arm_name]
        arm = Arm(arm_name, sim, maps if uses else None, device)
        tr, va = (lr_tr, lr_va) if (uses and arm_name != "gaussian") else (all_tr, all_va)
        if arm_name == "gaussian":   # the one-step family itself: scored on the discovery validation sims
            va = idx[:n_disc][val[:n_disc]]
        if arm.name not in ("gaussian", "prior") and (len(tr) < 2 or len(va) < 2):
            print(f"[seed {seed} | N={n_total}] {arm_name}: too few NPE simulations; skipped")
            continue
        arm.fit(theta_t[tr], x_t[tr], theta_t[va], x_t[va], cfg.npe, seed=seed + 17)
        samples = arm.sample(x_test, cfg.n_post)
        log_q = arm.log_prob(th_test, x_test)
        metrics, delta = evaluate(samples, log_q, ref)
        used = {"gaussian": n_disc, "prior": 0}.get(arm_name, n_total)
        row = dict(seed=seed, n_total=n_total, arm=arm_name, simulations_used=used,
                   n_discovery=n_disc if uses else 0, reuse=cfg.reuse,
                   k=maps.k if uses else None, m=maps.m if uses else None,
                   n_npe_train=arm.fit_info.get("n_train", 0), n_npe_val=arm.fit_info.get("n_val", 0),
                   val_log_prob=arm.fit_info["best_val_log_prob"],
                   npe_epochs=arm.fit_info.get("epochs"), npe_seconds=arm.fit_info.get("seconds"),
                   oracle_crps=floor["crps"], **metrics)
        rows.append(row)
        artifacts["samples"][arm_name], artifacts["delta"][arm_name] = samples, delta
        if verbose:
            print(f"[seed {seed} | N={n_total}] {arm_name:10s} KL {row['posterior_kl']:7.3f} "
                  f"± {row['posterior_kl_se']:.3f} | CRPS {row['crps']:.4f} (floor {floor['crps']:.4f}) | "
                  f"SW active {row['sw_active']:.3f} | val logq {row['val_log_prob']:8.3f}")
    screen_rows = None if table is None else table.assign(seed=seed, n_total=n_total, n_discovery=n_disc)
    return (rows, screen_rows, artifacts) if return_artifacts else (rows, screen_rows)


def cell_paths(out_dir: Path, seed: int, n_total: int):
    base = Path(out_dir)
    return base / "rows" / f"seed{seed}_n{n_total}.json", base / "screens" / f"seed{seed}_n{n_total}.csv"


def run_and_save(cfg, seed, n_total, out_dir, overwrite=False):
    rows_path, screen_path = cell_paths(out_dir, seed, n_total)
    if rows_path.exists() and not overwrite:
        print(f"skip existing {rows_path}")
        return None
    rows, screen = run_cell(cfg, seed, n_total, out_dir)
    if screen is not None:
        screen_path.parent.mkdir(parents=True, exist_ok=True)
        screen.to_csv(screen_path, index=False)
    atomic_write_json(rows_path, rows)
    return rows
