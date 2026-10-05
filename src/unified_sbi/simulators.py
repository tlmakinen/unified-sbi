"""Rosenbrock-chain simulators.

theta ~ U([-a, a]^D). Each observation has ``n_rep`` Gaussian replicates of

    mu(theta) = (theta_1, ..., theta_D,  theta_{j+1} - b * c_j * theta_j^2  for j = 1..D-1),

with c_j = 1 for links in ``curved_links`` and 0 otherwise. The default, ``curved_links=(0,)``
with ``b=1``, hides one 2D Rosenbrock banana in (theta_1, theta_2) inside an otherwise linear
Gaussian chain. ``curved_links="all"`` gives the fully curved chain used in unified_sbi.ipynb.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class RosenbrockChain:
    dim: int = 8
    n_rep: int = 8
    a_box: float = 3.0
    var_direct: float = 4.0
    var_link: float = 0.25
    curved_links: tuple | str = (0,)
    b: float = 1.0

    def __post_init__(self):
        if self.curved_links == "all":
            self.curved_links = tuple(range(self.dim - 1))
        self.curved_links = tuple(int(j) for j in self.curved_links)
        if any(j < 0 or j >= self.dim - 1 for j in self.curved_links):
            raise ValueError(f"curved_links must lie in [0, {self.dim - 2}].")

    # ------------------------------------------------------------------ shapes
    @property
    def x_dim(self) -> int:
        """Channels per replicate: D direct measurements plus D-1 links."""
        return 2 * self.dim - 1

    @property
    def raw_dim(self) -> int:
        """Length of one flattened, uncompressed observation."""
        return self.n_rep * self.x_dim

    @property
    def noise(self) -> np.ndarray:
        return np.sqrt([self.var_direct] * self.dim + [self.var_link] * (self.dim - 1))

    @property
    def curvature(self) -> np.ndarray:
        c = np.zeros(self.dim - 1)
        c[list(self.curved_links)] = 1.0
        return self.b * c

    # --------------------------------------------------------------- simulate
    def mu(self, theta: np.ndarray) -> np.ndarray:
        links = theta[..., 1:] - self.curvature * theta[..., :-1] ** 2
        return np.concatenate([theta, links], axis=-1)

    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-self.a_box, self.a_box, (n, self.dim))

    def simulate(self, n: int, seed: int):
        """Return theta (n, D) and x (n, n_rep, 2D-1), both float32."""
        rng = np.random.default_rng(seed)
        theta = self.sample_prior(n, rng)
        noise = rng.normal(size=(n, self.n_rep, self.x_dim)) * self.noise
        x = self.mu(theta)[:, None, :] + noise
        return theta.astype("float32"), x.astype("float32")

    def to_dict(self) -> dict:
        return asdict(self)


def make_simulator(name: str = "rosenbrock8_hidden2d", **overrides) -> RosenbrockChain:
    """Named presets. Keyword overrides replace preset fields."""
    presets = {
        "rosenbrock8_hidden2d": dict(dim=8, curved_links=(0,)),
        "rosenbrock8_full": dict(dim=8, curved_links="all"),
        "rosenbrock4_full": dict(dim=4, curved_links="all"),
    }
    if name not in presets:
        raise KeyError(f"Unknown simulator '{name}'. Options: {sorted(presets)}")
    return RosenbrockChain(**{**presets[name], **overrides})
