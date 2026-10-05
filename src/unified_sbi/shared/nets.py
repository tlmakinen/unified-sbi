"""Networks for the shared-space study: power spectrum, CNN, parameter embedding, MAF head."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class CrossSpectrum(nn.Module):
    """Signed binned tomographic cross-spectra Re(X_i X_j^*)/Npix in ``n_bins`` log |k| bins.

    Unlike the demo's |F_i F_j^*|, this is the actual (real part of the) cross-spectrum, so
    it is an unbiased estimator of the binned per-mode covariance. Output: (B, n_pairs*n_bins).
    """

    def __init__(self, n: int, n_tomo: int = 4, n_bins: int = 6):
        super().__init__()
        k = np.fft.fftfreq(n) * n
        kk = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2).ravel()
        edges = np.geomspace(0.5, kk.max() + 1e-6, n_bins + 1)
        b = np.searchsorted(edges, kk, side="right") - 1
        H = np.zeros((n_bins, n * n), np.float32)
        for j in range(n_bins):
            sel = (b == j) & (kk > 0)
            H[j, sel] = 1.0 / max(sel.sum(), 1)
        self.register_buffer("H", torch.from_numpy(H))
        self.pairs = [(i, j) for i in range(n_tomo) for j in range(i, n_tomo)]
        self.n2 = n * n
        self.dim = len(self.pairs) * n_bins

    @torch.no_grad()
    def forward(self, x):
        X = torch.fft.fft2(x.float())
        P = torch.stack([(X[:, i] * X[:, j].conj()).real.flatten(1) for i, j in self.pairs], 1) / self.n2
        return torch.einsum("bpf,kf->bpk", P, self.H).flatten(1)


class Standardise(nn.Module):
    def __init__(self, dim, shape=None):
        super().__init__()
        shape = shape or (dim,)
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("std", torch.ones(shape))

    def fit(self, x, dims=0):
        self.mean.copy_(x.mean(dims).reshape(self.mean.shape))
        self.std.copy_(x.std(dims).clamp_min(1e-12).reshape(self.std.shape))
        return self

    def forward(self, x):
        return (x - self.mean) / self.std


def mlp(din, dout, width=64, depth=2, act=nn.GELU):
    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        d = width
    layers.append(nn.Linear(d, dout))
    return nn.Sequential(*layers)


class CNN(nn.Module):
    """Stride-2 circular convs, GELU, doubling channels down to 8x8, then dense 128 -> 64 -> out."""

    def __init__(self, n: int, out: int, width: int = 8, n_tomo: int = 4):
        super().__init__()
        layers, c, f, s = [], n_tomo, width, n
        while s > 8:
            layers += [nn.Conv2d(c, f, 3, stride=2, padding=1, padding_mode="circular"), nn.GELU()]
            c, f, s = f, 2 * f, s // 2
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * 64, 128), nn.GELU(),
                                  nn.Linear(128, 64), nn.GELU(), nn.Linear(64, out))

    def forward(self, x):
        return self.head(self.conv(x))


class DataSummary(nn.Module):
    """t(x) for features in {"pk", "cnn", "hybrid", "hybrid_split"}.

    * pk:            MLP(Pk) -> out
    * pk_linear:     W Pk -> out   (exponential-family form: statistics linear in the spectrum)
    * cnn:           CNN(x) -> out
    * hybrid:        MLP([Pk, CNN_c(x)]) -> out       (fusion, as in the agent's new arm)
    * hybrid_split:  [Linear(Pk) -> k_pk, CNN(x) -> out - k_pk]   (interpretable blocks)
    """

    def __init__(self, features, out, n, pk_dim, cnn_width=8, cnn_extra=8, k_pk=2):
        super().__init__()
        self.features, self.out = features, out
        if features == "pk":
            self.f = mlp(pk_dim, out)
        elif features == "pk_linear":
            self.f = nn.Linear(pk_dim, out)
        elif features == "cnn":
            self.cnn = CNN(n, out, cnn_width)
        elif features == "hybrid":
            self.cnn = CNN(n, cnn_extra, cnn_width)
            self.f = mlp(pk_dim + cnn_extra, out)
        elif features == "hybrid_split":
            if not 0 < k_pk < out:
                raise ValueError("hybrid_split needs 0 < k_pk < out")
            self.lin = nn.Linear(pk_dim, k_pk)
            self.cnn = CNN(n, out - k_pk, cnn_width)
            self.k_pk = k_pk
        else:
            raise ValueError(features)
        self.uses_maps = features not in ("pk", "pk_linear")

    def forward(self, im, pk):
        if self.features in ("pk", "pk_linear"):
            return self.f(pk)
        if self.features == "cnn":
            return self.cnn(im)
        if self.features == "hybrid":
            return self.f(torch.cat([pk, self.cnn(im)], -1))
        return torch.cat([self.lin(pk), self.cnn(im)], -1)


class ParamEmbed(nn.Module):
    """eta(theta): R^2 -> R^K, linear skip plus smooth MLP on box (or logit) coordinates."""

    def __init__(self, K, lo, hi, width=64, coords="box", act=nn.GELU):
        super().__init__()
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.as_tensor(hi, dtype=torch.float32))
        self.coords = coords
        self.lin = nn.Linear(2, K)
        self.net = mlp(2, K, width, act=act)
        nn.init.normal_(self.net[-1].weight, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def to_coords(self, theta):
        u = (theta - self.lo) / (self.hi - self.lo)
        if self.coords == "box":
            return 2 * u - 1, torch.log(2 / (self.hi - self.lo)).sum().expand(theta.shape[:-1])
        s = math.pi / math.sqrt(3)
        y = (torch.log(u) - torch.log1p(-u)) / s
        ld = (-torch.log(self.hi - self.lo) - torch.log(u) - torch.log1p(-u) - math.log(s)).sum(-1)
        return y, ld

    def f(self, y):
        return self.lin(y) + self.net(y)

    def forward(self, theta):
        return self.f(self.to_coords(theta)[0])

    def log_volume(self, theta):
        """eta(theta) and 0.5 log det(J^T J) with J = d eta / d theta (exact, forward mode)."""
        y, ld = self.to_coords(theta)
        cols = []
        for a in range(2):
            e = torch.zeros_like(y); e[..., a] = 1
            z, jv = torch.func.jvp(self.f, (y,), (e,))
            cols.append(jv)
        J = torch.stack(cols, -1).double()                       # (..., K, 2) in y coordinates
        G = J.transpose(-1, -2) @ J
        lv = 0.5 * torch.logdet(G)
        return z, (lv + ld.double()).to(z.dtype)


class LogitBox:
    """theta in the box <-> y in R^2 (for flows), with log|dy/dtheta|."""

    def __init__(self, lo, hi):
        self.lo = torch.as_tensor(lo, dtype=torch.float32)
        self.hi = torch.as_tensor(hi, dtype=torch.float32)

    def forward(self, theta):
        lo, hi = self.lo.to(theta), self.hi.to(theta)
        u = ((theta - lo) / (hi - lo)).clamp(1e-7, 1 - 1e-7)
        y = torch.log(u) - torch.log1p(-u)
        ld = (-torch.log(hi - lo) - torch.log(u) - torch.log1p(-u)).sum(-1)
        return y, ld

    def inverse(self, y):
        lo, hi = self.lo.to(y), self.hi.to(y)
        return lo + (hi - lo) * torch.sigmoid(y)


class MAFHead(nn.Module):
    """zuko MAF on logit-box coordinates, conditioned on a context vector."""

    def __init__(self, context, lo, hi, transforms=5, hidden=50):
        super().__init__()
        import zuko
        self.flow = zuko.flows.MAF(features=2, context=context, transforms=transforms,
                                   hidden_features=[hidden, hidden])
        self.box = LogitBox(lo, hi)

    def log_prob(self, theta, c):
        y, ld = self.box.forward(theta)
        return self.flow(c).log_prob(y) + ld

    @torch.no_grad()
    def sample_and_log_prob(self, c, n):
        d = self.flow(c)
        y, lp = d.rsample_and_log_prob((n,))                     # (n, B, 2), (n, B)
        th = self.box.inverse(y)
        _, ld = self.box.forward(th)
        return th.transpose(0, 1), (lp + ld).transpose(0, 1)
