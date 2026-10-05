"""Adaptive midpoint quadrature over a 2D parameter box, batched over observations.

Used for (i) the normaliser Z(t) of shared-space posteriors during training and evaluation and
(ii) the exact lensing posterior. Level 0 is a global ``n0 x n0`` grid shared by the batch;
each further level replaces the cells of the previous level that lie inside a per-item
bounding box of the significant cells (expanded by one cell) by an ``n x n`` grid. The cells
that are kept form an exact partition of the box, so

    log Z = logsumexp_cells(log f(centre) + log area)

is a consistent quadrature rule (stratified Monte Carlo when ``jitter=True``, which makes
Z unbiased cell by cell).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Cells:
    logf: torch.Tensor      # (B, M) log integrand at the cell points
    logarea: torch.Tensor   # (B, M) log cell area (-inf for cells that were refined away)
    points: torch.Tensor    # (B, M, 2)
    width: torch.Tensor     # (B, M, 2) cell widths
    logZ: torch.Tensor      # (B,)

    @property
    def logmass(self):
        return self.logf + self.logarea - self.logZ[:, None]

    def hpd_level(self, logf_true):
        """Posterior mass with density above the density at the truth (per item)."""
        above = (self.logf > logf_true[:, None]) & torch.isfinite(self.logarea)
        lm = torch.where(above, self.logmass, torch.full_like(self.logmass, -torch.inf))
        return torch.logsumexp(lm, -1).exp().clamp(0, 1)

    def moments(self):
        w = self.logmass.exp()
        mean = (w[..., None] * self.points).sum(1)
        var = (w[..., None] * (self.points - mean[:, None]) ** 2).sum(1) + (w[..., None] * self.width ** 2 / 12).sum(1)
        return mean, var.sqrt()

    def sample(self, n, generator=None):
        p = self.logmass.exp().nan_to_num(0)
        idx = torch.multinomial(p, n, replacement=True, generator=generator)        # (B, n)
        c = torch.gather(self.points, 1, idx[..., None].expand(-1, -1, 2))
        w = torch.gather(self.width, 1, idx[..., None].expand(-1, -1, 2))
        u = torch.rand(c.shape, generator=generator, dtype=c.dtype, device=c.device) - 0.5
        return c + u * w

    def marginal_cdf_at(self, value, dim):
        """P(theta_dim < value) per item, treating mass as uniform within each cell."""
        lo = self.points[..., dim] - self.width[..., dim] / 2
        frac = ((value[:, None] - lo) / self.width[..., dim]).clamp(0, 1)
        return (self.logmass.exp() * frac).sum(-1)


def _grid(lo, hi, m, jitter, generator=None):
    """lo, hi: (B, 2). Returns points (B, m*m, 2) and widths (B, 2)."""
    B = lo.shape[0]
    width = (hi - lo) / m
    i = torch.arange(m, device=lo.device, dtype=lo.dtype)
    ij = torch.stack(torch.meshgrid(i, i, indexing="ij"), -1).reshape(-1, 2)    # (m*m, 2)
    if jitter:
        off = torch.rand((B, m * m, 2), generator=generator, device=lo.device, dtype=lo.dtype)
    else:
        off = torch.full((1, 1, 2), 0.5, device=lo.device, dtype=lo.dtype)
    pts = lo[:, None, :] + (ij[None] + off) * width[:, None, :]
    return pts, width, ij


def integrate(logf, lo, hi, batch, n0=48, n=24, levels=2, tol=1e-5, jitter=False,
              generator=None, shared_level0=True):
    """Adaptive quadrature of exp(logf) over the box [lo, hi].

    logf(points) maps (B, M, 2) -> (B, M). If ``shared_level0`` the level-0 points are the
    same for every item, and logf receives them with B = 1 broadcastable shape (1, M, 2)
    and may return (B, M) (e.g. a shared parameter embedding compared with B summaries).
    """
    lo = lo.reshape(1, 2).expand(batch, 2)
    hi = hi.reshape(1, 2).expand(batch, 2)
    pieces = []
    for level in range(levels + 1):
        m = n0 if level == 0 else n
        if level == 0 and shared_level0:
            pts, width, ij = _grid(lo[:1], hi[:1], m, jitter, generator)
            lf = logf(pts).expand(batch, -1)
            pts = pts.expand(batch, -1, -1); width = width.expand(batch, 2)
        else:
            pts, width, ij = _grid(lo, hi, m, jitter, generator)
            lf = logf(pts)
        logarea = torch.log(width.prod(-1))[:, None].expand_as(lf)
        keep = torch.ones_like(lf, dtype=torch.bool)
        if level < levels:
            with torch.no_grad():
                lw = lf + logarea
                w = torch.softmax(torch.nan_to_num(lw, nan=-torch.inf), -1)
                sig = w > tol
                sig = sig | (w == w.max(-1, keepdim=True).values)                # at least one cell
                big = torch.tensor(m, device=lf.device)
                ix, iy = ij[:, 0][None].expand_as(sig), ij[:, 1][None].expand_as(sig)
                ilo_x = torch.where(sig, ix, big).min(-1).values - 1
                ihi_x = torch.where(sig, ix, -1).max(-1).values + 1
                ilo_y = torch.where(sig, iy, big).min(-1).values - 1
                ihi_y = torch.where(sig, iy, -1).max(-1).values + 1
                ilo = torch.stack([ilo_x, ilo_y], -1).clamp(0, m - 1)
                ihi = torch.stack([ihi_x, ihi_y], -1).clamp(0, m - 1)
                inside = (ij[None] >= ilo[:, None]).all(-1) & (ij[None] <= ihi[:, None]).all(-1)
                keep = ~inside
                new_lo = lo + ilo.to(lo.dtype) * width
                new_hi = lo + (ihi + 1).to(lo.dtype) * width
        logarea = torch.where(keep, logarea, torch.full_like(logarea, -torch.inf))
        pieces.append((lf, logarea, pts, width[:, None, :].expand(-1, lf.shape[1], -1)))
        if level < levels:
            lo, hi = new_lo, new_hi
    lf = torch.cat([p[0] for p in pieces], 1)
    la = torch.cat([p[1] for p in pieces], 1)
    pts = torch.cat([p[2] for p in pieces], 1)
    wd = torch.cat([p[3] for p in pieces], 1)
    logZ = torch.logsumexp(lf + la, -1)
    return Cells(lf, la, pts, wd, logZ)
