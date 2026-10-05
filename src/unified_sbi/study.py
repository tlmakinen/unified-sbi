"""Simulation-budget scaling study: one (seed, total budget) cell at a time.

Budget accounting (all arms see the same N_total simulations and the same split):

* A pool of ``pool_size`` simulations is drawn per seed. A budget N_total uses its first N_total
  entries, so budgets are nested within a seed.
* Each simulation is assigned to validation with probability ``val_fraction`` (fixed per seed),
  so a simulation keeps its role across budgets and across both stages.
* Discovery (our method) uses the first n_disc = min(n_discovery, N_total) simulations: their
  training members fit (eta_phi, eta_psi), their validation members early-stop it.
* NPE then trains on all N_total training members, REUSING the discovery simulations, and
  early-stops on all N_total validation members. The raw baseline uses exactly the same split.
* Metrics on a fixed, independent test set (not counted in any budget) sit alongside the
  end-of-training validation log-probability, together with exact-posterior oracles.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import methods
from .exact import exact_test_metrics
from .maps import MapConfig, check_maps, fit_maps, load_maps
from .metrics import central_coverage, crps_ensemble
from .npe import NPEConfig, train_npe
from .simulators import RosenbrockChain, make_simulator
from .utils import atomic_torch_save, atomic_write_json, seed_everything, to_tensor

DEFAULT_BUDGETS = (250, 500, 1000, 2000, 5000, 10000)


@dataclass
class StudyConfig:
    simulator: str = "rosenbrock8_hidden2d"
    simulator_overrides: dict = field(default_factory=dict)
    budgets: tuple = DEFAULT_BUDGETS
    seeds: tuple = (0, 1, 2)
    arms: tuple = ("raw", "theta_t", "eta_t")
    n_discovery: int = 500
    reuse_discovery: bool = True      # False: map-based arms train NPE only on non-discovery sims
    val_fraction: float = 0.15
    pool_size: int | None = None      # defaults to max(budgets)
    n_test: int = 500
    n_post: int = 500                 # posterior draws per test observation
    test_seed: int = 10_000
    exact_grid: int = 200
    maps: MapConfig = field(default_factory=MapConfig)
    npe: NPEConfig = field(default_factory=NPEConfig)
    device: str = "cpu"

    def __post_init__(self):
        if isinstance(self.maps, dict):
            self.maps = MapConfig(**self.maps)
        if isinstance(self.npe, dict):
            self.npe = NPEConfig(**self.npe)
        self.budgets = tuple(int(b) for b in self.budgets)
        self.seeds = tuple(int(s) for s in self.seeds)
        self.arms = tuple(self.arms)
        unknown = set(self.arms) - set(methods.ARMS)
        if unknown:
            raise ValueError(f"Unknown arms {unknown}; options {methods.ARMS}")
        if self.pool_size is None:
            self.pool_size = max(self.budgets)

    def make_simulator(self) -> RosenbrockChain:
        return make_simulator(self.simulator, **self.simulator_overrides)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["budgets"], d["seeds"], d["arms"] = list(self.budgets), list(self.seeds), list(self.arms)
        return d

    def tasks(self) -> list[tuple[int, int]]:
        """All (seed, budget) cells; a SLURM array index selects one."""
        return [(s, b) for s in self.seeds for b in self.budgets]


# --------------------------------------------------------------------------- data
def simulation_pool(sim: RosenbrockChain, cfg: StudyConfig, seed: int):
    theta, x = sim.simulate(cfg.pool_size, seed=1000 * seed + 1)
    is_val = np.random.default_rng(1000 * seed + 2).random(cfg.pool_size) < cfg.val_fraction
    return theta, x, is_val


def split_indices(is_val: np.ndarray, n_total: int, n_disc: int) -> dict:
    idx = np.arange(n_total)
    val = is_val[:n_total]
    split = dict(npe_train=idx[~val], npe_val=idx[val],
                 disc_train=idx[:n_disc][~val[:n_disc]], disc_val=idx[:n_disc][val[:n_disc]])
    for k, v in split.items():
        if len(v) < 2:
            raise ValueError(f"Split '{k}' has {len(v)} simulations; raise the budget or val_fraction.")
    return split


def test_set(sim: RosenbrockChain, cfg: StudyConfig):
    return sim.simulate(cfg.n_test, seed=cfg.test_seed)


def oracle_metrics(sim: RosenbrockChain, cfg: StudyConfig, out_dir: Path | None = None) -> dict:
    """Exact-posterior test metrics, cached in out_dir/oracle.json."""
    path = Path(out_dir) / "oracle.json" if out_dir else None
    if path and path.exists():
        return json.loads(path.read_text())
    theta_test, x_test = test_set(sim, cfg)
    ex = exact_test_metrics(sim, x_test, theta_test, cfg.exact_grid)
    oracle = dict(test_log_prob=float(ex["log_prob"].mean()),
                  test_log_prob_se=float(ex["log_prob"].std(ddof=1) / np.sqrt(cfg.n_test)),
                  crps=float(ex["crps"].mean()),
                  crps_se=float(ex["crps"].mean(1).std(ddof=1) / np.sqrt(cfg.n_test)),
                  crps_per_dim=ex["crps"].mean(0).tolist(),
                  n_test=cfg.n_test, test_seed=cfg.test_seed, exact_grid=cfg.exact_grid)
    if path:
        atomic_write_json(path, oracle)
    return oracle


def _maps_cache_key(sim: RosenbrockChain, cfg: StudyConfig, seed: int, n_disc: int) -> str:
    blob = json.dumps(dict(sim=sim.to_dict(), maps=cfg.maps.to_dict(), seed=seed, n_disc=n_disc,
                           pool=cfg.pool_size, val=cfg.val_fraction), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def discovery(sim, cfg, seed, theta_t, x_t, split, n_disc, cache_dir: Path | None):
    """Fit (or load cached) maps on the discovery simulations."""
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"maps_seed{seed}_n{n_disc}_{_maps_cache_key(sim, cfg, seed, n_disc)}.pt"
        if path.exists():
            state = torch.load(path, map_location=cfg.device, weights_only=False)
            return load_maps(state, sim.dim, x_t[split["disc_train"]], cfg.maps), True
    tr, va = split["disc_train"], split["disc_val"]
    maps = fit_maps(theta_t[tr], x_t[tr], theta_t[va], x_t[va], cfg.maps, seed=seed)
    check_maps(maps, theta_t[tr[:8]])
    if path is not None:
        atomic_torch_save(maps.state(), path)
    return maps, False


# --------------------------------------------------------------------------- one cell
def run_cell(cfg: StudyConfig, seed: int, n_total: int, out_dir: Path | None = None,
             verbose: bool = True, return_artifacts: bool = False):
    """Train every arm at one (seed, N_total) and return one metrics row per arm.

    With ``return_artifacts`` also return a dict holding the maps, the trained estimators,
    the test set and every arm's posterior draws, for diagnostics.
    """
    device = cfg.device
    seed_everything(seed)
    sim = cfg.make_simulator()
    theta, x, is_val = simulation_pool(sim, cfg, seed)
    n_disc = min(cfg.n_discovery, n_total)
    split = split_indices(is_val, n_total, n_disc)
    theta_t, x_t = to_tensor(theta[:n_total], device), to_tensor(x[:n_total], device)
    theta_test, x_test = test_set(sim, cfg)
    theta_test_t, x_test_t = to_tensor(theta_test, device), to_tensor(x_test, device)
    oracle = oracle_metrics(sim, cfg, out_dir)

    maps, maps_cached = None, False
    if any(methods.USES_MAPS[a] for a in cfg.arms):
        maps, maps_cached = discovery(sim, cfg, seed, theta_t, x_t, split, n_disc,
                                      Path(out_dir) / "cache" if out_dir else None)
        if verbose:
            print(f"[seed {seed} | N={n_total}] discovery on {n_disc} sims "
                  f"({'cached' if maps_cached else f'{maps.seconds:.1f}s'}), "
                  f"best one-step val loss {maps.best_val_loss:.3f} at step {maps.best_step}")

    rows = []
    artifacts = dict(maps=maps, fits={}, draws={}, theta_test=theta_test, x_test=x_test,
                     split=split, simulator=sim)
    tr, va = split["npe_train"], split["npe_val"]
    for arm in cfg.arms:
        arm_maps = maps if methods.USES_MAPS[arm] else None
        context = methods.context_for(arm, x_t, arm_maps)
        target, logdet = methods.target_for(arm, theta_t, arm_maps)
        arm_tr = tr
        if arm_maps is not None and not cfg.reuse_discovery:
            arm_tr = np.setdiff1d(tr, split["disc_train"])
            if len(arm_tr) < 2:
                print(f"[seed {seed} | N={n_total}] {arm}: no non-discovery sims to train on; skipped")
                continue
        fit = train_npe(target[arm_tr], context[arm_tr], target[va], context[va], cfg.npe, seed=seed + 17)
        val_log_prob_theta = fit.best_val_log_prob + logdet[va].mean().item()

        test_context = methods.context_for(arm, x_test_t, arm_maps)
        draws = methods.sample_posteriors(fit.net, arm, test_context, cfg.n_post, arm_maps, sim.a_box)
        test_lp = methods.log_prob_theta(fit.net, arm, theta_test_t, test_context, arm_maps,
                                         acceptance=draws.acceptance)
        crps = crps_ensemble(draws.samples, theta_test)
        row = dict(
            seed=seed, n_total=n_total, arm=arm, simulator=cfg.simulator,
            n_discovery=n_disc if methods.USES_MAPS[arm] else 0,
            n_npe_train=len(arm_tr), n_npe_val=len(va),
            val_log_prob=val_log_prob_theta,
            test_log_prob=float(test_lp.mean()),
            test_log_prob_se=float(test_lp.std(ddof=1) / np.sqrt(len(test_lp))),
            crps=float(crps.mean()),
            crps_se=float(crps.mean(1).std(ddof=1) / np.sqrt(len(crps))),
            crps_per_dim=crps.mean(0).tolist(),
            coverage68=float(central_coverage(draws.samples, theta_test, 0.68).mean()),
            coverage90=float(central_coverage(draws.samples, theta_test, 0.90).mean()),
            acceptance=float(draws.acceptance.mean()),
            n_obs_clipped=int(draws.n_topped_up),
            npe_epochs=fit.epochs_trained, npe_best_epoch=fit.best_epoch, npe_seconds=fit.seconds,
            map_seconds=(maps.seconds if (arm_maps and not maps_cached) else 0.0),
            map_best_val_loss=(maps.best_val_loss if arm_maps else None),
            oracle_test_log_prob=oracle["test_log_prob"], oracle_crps=oracle["crps"],
        )
        rows.append(row)
        artifacts["fits"][arm], artifacts["draws"][arm] = fit, draws.samples
        if verbose:
            print(f"[seed {seed} | N={n_total}] {arm:8s} val logq {row['val_log_prob']:8.3f} | "
                  f"test logq {row['test_log_prob']:8.3f} (oracle {oracle['test_log_prob']:.3f}) | "
                  f"CRPS {row['crps']:.4f} (oracle {oracle['crps']:.4f}) | "
                  f"{fit.epochs_trained} epochs, {fit.seconds:.0f}s")
    return (rows, artifacts) if return_artifacts else rows


def cell_path(out_dir: Path, seed: int, n_total: int) -> Path:
    return Path(out_dir) / "rows" / f"seed{seed}_n{n_total}.json"


def run_and_save(cfg: StudyConfig, seed: int, n_total: int, out_dir: Path,
                 overwrite: bool = False) -> list[dict] | None:
    path = cell_path(out_dir, seed, n_total)
    if path.exists() and not overwrite:
        print(f"skip existing {path}")
        return None
    rows = run_cell(cfg, seed, n_total, out_dir)
    atomic_write_json(path, rows)
    return rows
