"""Fisher machinery of docs/plasma_surrogate_conditioning.md, section 3.

All Jacobians are with respect to *standardised* theta and are measured in the
instance-normalised output frame y = s_{t+1} / r_t, so the output metric W is the identity in
that frame (a relative-error metric, see section 3.1).

    surrogate_jacobian   central differences of the surrogate in theta    (section 4.1)
    fisher_from_jacobian A = J^T J / D, pooled over a run's states          (section 3.2)
    FisherBank           per-run EMA of A, normalised target M = A / max(s, eps_s) + lam I
    metric_of_g          G_eps = J_g^T J_g + eps_g I for the embedding (differentiable)
    r_iso                tr P - log det P - d,  P = G_eps^{-1} M              (section 3.3)
    whitened_noise       delta ~ N(0, sigma^2 G^{-1})                        (section 3.5)
    maximin              greedy maximin selection in an embedding            (section 7)
"""

from __future__ import annotations

import torch
from torch.func import jacrev, vmap


def perturbations(theta, eps):
    """(S, d) -> (S, 2d, d): theta + eps e_i, theta - eps e_i for i = 1..d."""
    S, d = theta.shape
    E = torch.eye(d, device=theta.device, dtype=theta.dtype) * eps
    P = torch.stack([theta[:, None] + E, theta[:, None] - E], 2)    # (S, d, 2, d)
    return P.reshape(S, 2 * d, d)


def surrogate_jacobian(fn, x, theta, logr, eps=0.02, chunk=256):
    """J (S, d, f, n, n) of fn(x, theta, logr) by central differences in standardised theta.

    Gradients flow if called outside ``torch.no_grad`` (used for Jacobian supervision)."""
    S, d = theta.shape
    P = perturbations(theta, eps).reshape(-1, d)
    X = x.repeat_interleave(2 * d, 0)
    L = logr.repeat_interleave(2 * d, 0)
    outs = [fn(X[i:i + chunk], P[i:i + chunk], L[i:i + chunk]) for i in range(0, len(P), chunk)]
    Y = torch.cat(outs).reshape(S, d, 2, *x.shape[1:])
    return (Y[:, :, 0] - Y[:, :, 1]) / (2 * eps)


def fisher_from_jacobian(J):
    """(S, d, f, n, n) -> (S, d, d) with A = J^T J / D (D = f n n)."""
    Jf = J.flatten(2)
    return Jf @ Jf.transpose(1, 2) / Jf.shape[-1]


class FisherBank:
    """Per-run exponential moving average of the surrogate's Fisher matrix."""

    def __init__(self, n_runs, d, decay=0.7, device="cpu"):
        self.A = torch.zeros(n_runs, d, d, device=device)
        self.seen = torch.zeros(n_runs, dtype=torch.bool, device=device)
        self.decay, self.d = decay, d

    def update(self, run_ids, A):
        """run_ids: (S,) run index per sample; A: (S, d, d). Samples of one run are averaged."""
        run_ids = run_ids.to(self.A.device)
        u = torch.unique(run_ids)
        for r in u.tolist():
            a = A[run_ids == r].mean(0)
            if self.seen[r]:
                self.A[r] = self.decay * self.A[r] + (1 - self.decay) * a
            else:
                self.A[r], self.seen[r] = a, True

    def set(self, A):
        self.A = A.to(self.A.device).clone()
        self.seen[:] = True

    def scale(self):
        """s_bar = mean_r tr A_r / d over the runs seen so far."""
        return self.A[self.seen].diagonal(dim1=-2, dim2=-1).sum(-1).mean() / self.d

    def target(self, lam, eps_s=1e-12):
        """M = A / max(s_bar, eps_s) + lam I (section 3.2)."""
        eye = torch.eye(self.d, device=self.A.device)
        return self.A / self.scale().clamp_min(eps_s) + lam * eye


def jacobian_of_g(g, theta):
    """(N, d) -> J_g (N, K, d), differentiable w.r.t. the parameters of g."""
    return vmap(jacrev(lambda t: g(t[None])[0]))(theta)


def metric_of_g(g, theta, eps_g=0.0):
    """G_eps = J_g^T J_g + eps_g I  (N, d, d)."""
    Jg = jacobian_of_g(g, theta)
    G = Jg.transpose(1, 2) @ Jg
    return G + eps_g * torch.eye(G.shape[-1], device=G.device, dtype=G.dtype) if eps_g else G


def r_iso(G, M):
    """Mean over runs of tr P - log det P - d, P = G^{-1} M; zero iff G = M, >= 0.

    Computed as tr(G^{-1} M) + log det G - log det M - d with Cholesky solves (no explicit
    inverse). Pass G_eps = J_g^T J_g + eps_g I (``metric_of_g(..., eps_g)``) so the loss stays
    defined if J_g is temporarily rank-deficient."""
    d = G.shape[-1]
    L = torch.linalg.cholesky(G)
    tr = torch.cholesky_solve(M, L).diagonal(dim1=-2, dim2=-1).sum(-1)
    logdet_G = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    logdet_M = torch.linalg.slogdet(M)[1]
    return (tr - (logdet_M - logdet_G) - d).mean()


def whitened_noise(G, sigma, generator=None):
    """delta ~ N(0, sigma^2 G^{-1}) per row. G: (B, d, d) positive definite (use G_eps)."""
    B, d, _ = G.shape
    L = torch.linalg.cholesky(G)
    xi = torch.randn(B, d, 1, device=G.device, dtype=G.dtype, generator=generator)
    delta = torch.linalg.solve_triangular(L.transpose(1, 2), xi, upper=True)[..., 0]
    return torch.as_tensor(sigma, device=G.device, dtype=G.dtype).reshape(-1, 1) * delta


def isotropic_noise_matched(G, sigma, generator=None):
    """delta ~ N(0, s^2 I) with s^2 = sigma^2 d / tr G, so E[delta^T G delta] matches the
    whitened noise (the control arm A5)."""
    B, d, _ = G.shape
    s = torch.as_tensor(sigma, device=G.device, dtype=G.dtype).reshape(-1) * torch.sqrt(
        d / G.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12))
    xi = torch.randn(B, d, device=G.device, dtype=G.dtype, generator=generator)
    return s[:, None] * xi


def nn_spacing(c):
    """Median nearest-neighbour distance between rows of c (N, K)."""
    D = torch.cdist(c, c)
    D.fill_diagonal_(float("inf"))
    return D.min(1).values.median()


def maximin(cand, existing, k):
    """Greedy maximin: pick k rows of ``cand`` (M, K) maximising the min Euclidean (chord)
    distance to ``existing`` (E, K) and to the rows already picked. Indices into cand."""
    dist = lambda i: (cand - cand[i]).norm(dim=1)
    d0 = torch.cdist(cand, existing).min(1).values if len(existing) else None
    return _greedy(len(cand), d0, dist, k)


def maximin_geodesic(cand, existing, k, n_neighbours=10):
    """Greedy maximin on graph shortest-path distances along the learned manifold (section 7):
    a symmetric k-nearest-neighbour graph over cand + existing in the embedding, weighted by
    chord length; geodesic ~ shortest path. Falls back to the chord for disconnected pairs."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    X = torch.cat([cand, existing]).double().cpu()
    D = torch.cdist(X, X)
    nn = D.topk(n_neighbours + 1, largest=False).indices[:, 1:]
    rows = torch.arange(len(X))[:, None].expand_as(nn).reshape(-1)
    W = csr_matrix((D[rows, nn.reshape(-1)].numpy(), (rows.numpy(), nn.reshape(-1).numpy())),
                   shape=(len(X), len(X)))
    W = W.maximum(W.T)
    Dg = torch.as_tensor(dijkstra(W, directed=False))
    Dg = torch.where(torch.isfinite(Dg), Dg, D)
    M = len(cand)
    d0 = Dg[:M, M:].min(1).values if len(existing) else None
    return _greedy(M, d0, lambda i: Dg[:M, i], k)


def _greedy(M, d0, dist, k):
    dmin = d0.clone() if d0 is not None else torch.full((M,), float("inf"))
    dmin = dmin.double().cpu()
    picked = []
    for _ in range(k):
        i = int(torch.argmax(dmin))
        picked.append(i)
        dmin = torch.minimum(dmin, dist(i).double().cpu())
        dmin[i] = -1.0
    return picked


def affine_invariant_distance(A, B, jitter=1e-10):
    """|| log(A^{-1/2} B A^{-1/2}) ||_F per pair, for SPD (N, d, d)."""
    d = A.shape[-1]
    eye = torch.eye(d, dtype=A.dtype, device=A.device)
    ev = torch.linalg.eigvals(torch.linalg.solve(A + jitter * eye, B + jitter * eye)).real
    return torch.log(ev.clamp_min(1e-30)).pow(2).sum(-1).sqrt()
