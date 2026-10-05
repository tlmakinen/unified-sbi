"""Networks: the parameter embedding g and a conditional patch-transformer surrogate.

The surrogate mirrors the production layout (grid2mesh encoder -> attention processors -> z ->
mesh2grid decoder) at Colab scale:

    x_t (2 x n x n, instance-normalised) -> circular conv + patchify (+ c)       "grid2mesh"
      -> L transformer blocks with adaLN-Zero conditioning on c                  "processors"
      -> final adaLN + linear unpatchify + circular conv refinement              "mesh2grid"
      -> residual prediction  y_hat = x_t + delta

The conditioning vector enters (i) every token at the encoder, (ii) every processor block and
(iii) the decoder, as in section 2 of docs/plasma_surrogate_conditioning.md. Everything that
reads the conditioning (g, the conditioning MLP and all modulation projections) is exposed by
``conditioning_parameters`` so that it can get its own weight decay.

Scale handling: the state is divided by its own per-field RMS r_t (a property of the input,
not of theta), the target is s_{t+1} / r_t, and log r_t is appended to the conditioning so the
network still knows the absolute amplitude (which sets the strength of the nonlinearity).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ParamEmbed(nn.Module):
    """g: standardised theta (d) -> c (K). ``identity=True`` gives the raw-theta baseline."""

    def __init__(self, d, K=8, hidden=128, depth=2, identity=False):
        super().__init__()
        self.identity, self.d = identity, d
        self.K = d if identity else K
        if not identity:
            layers, w = [], d
            for _ in range(depth):
                layers += [nn.Linear(w, hidden), nn.SiLU()]
                w = hidden
            layers.append(nn.Linear(w, K))
            self.net = nn.Sequential(*layers)

    def forward(self, theta):
        return theta if self.identity else self.net(theta)


class Block(nn.Module):
    """DiT-style transformer block with adaLN-Zero modulation from the conditioning vector."""

    def __init__(self, w, heads, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(w, 3 * w)
        self.proj = nn.Linear(w, w)
        h = int(w * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(w, h), nn.GELU(approximate="tanh"), nn.Linear(h, w))
        self.mod = nn.Linear(w, 6 * w)
        nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)

    def forward(self, x, cvec):
        B, T, w = x.shape
        s1, b1, g1, s2, b2, g2 = self.mod(F.silu(cvec))[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        q, k, v = self.qkv(h).reshape(B, T, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, w)
        x = x + g1 * self.proj(a)
        h = self.n2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class Surrogate(nn.Module):
    def __init__(self, n=64, fields=2, c_dim=8, width=192, depth=6, heads=6, patch=4,
                 extra_cond=2):
        super().__init__()
        assert n % patch == 0 and width % heads == 0
        self.n, self.f, self.p, self.w = n, fields, patch, width
        self.T = (n // patch) ** 2
        # conditioning: [c, log r_t] -> cvec
        self.cond = nn.Sequential(nn.Linear(c_dim + extra_cond, width), nn.SiLU(),
                                  nn.Linear(width, width))
        # grid2mesh
        self.stem = nn.Conv2d(fields, width // 4, 3, padding=1, padding_mode="circular")
        self.patchify = nn.Conv2d(width // 4, width, patch, stride=patch)
        self.pos = nn.Parameter(torch.randn(1, self.T, width) * 0.02)
        self.tok_cond = nn.Linear(width, width)
        # processors
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        # mesh2grid
        self.n_out = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mod_out = nn.Linear(width, 2 * width)
        nn.init.zeros_(self.mod_out.weight); nn.init.zeros_(self.mod_out.bias)
        self.unpatch = nn.Linear(width, patch * patch * fields)
        self.refine = nn.Sequential(
            nn.Conv2d(2 * fields, 32, 3, padding=1, padding_mode="circular"), nn.GELU(),
            nn.Conv2d(32, fields, 3, padding=1, padding_mode="circular"))
        nn.init.zeros_(self.refine[-1].weight); nn.init.zeros_(self.refine[-1].bias)

    def conditioning_parameters(self):
        mods = [self.cond, self.tok_cond, self.mod_out] + [b.mod for b in self.blocks]
        return [p for m in mods for p in m.parameters()]

    def forward(self, x, c, logr):
        """x: (B, f, n, n) normalised state; c: (B, c_dim); logr: (B, extra_cond)."""
        B = x.shape[0]
        cvec = self.cond(torch.cat([c, logr], -1))
        h = self.patchify(F.gelu(self.stem(x))).flatten(2).transpose(1, 2)    # (B, T, w)
        h = h + self.pos + self.tok_cond(F.silu(cvec))[:, None]
        for blk in self.blocks:
            h = blk(h, cvec)
        s, b = self.mod_out(F.silu(cvec))[:, None].chunk(2, -1)
        h = self.unpatch(self.n_out(h) * (1 + s) + b)                          # (B, T, p*p*f)
        g = self.n // self.p
        d = h.reshape(B, g, g, self.p, self.p, self.f).permute(0, 5, 1, 3, 2, 4)
        d = d.reshape(B, self.f, self.n, self.n)
        d = d + self.refine(torch.cat([d, x], 1))
        return x + d


class Model(nn.Module):
    """g + surrogate. Works in the instance-normalised frame (see module docstring)."""

    def __init__(self, d, n, K=8, identity=False, g_hidden=128, **kw):
        super().__init__()
        self.g = ParamEmbed(d, K, g_hidden, identity=identity)
        self.net = Surrogate(n=n, c_dim=self.g.K, **kw)

    def conditioning_parameters(self):
        return list(self.g.parameters()) + self.net.conditioning_parameters()

    def forward(self, x, theta, logr):
        return self.net(x, self.g(theta), logr)

    @torch.no_grad()
    def rescale_gauge(self, a):
        """c -> a c with the first conditioning layer absorbing 1/a: predictions are unchanged
        exactly, but the pullback metric of g scales by a^2 (the gauge of section 3.3)."""
        if self.g.identity:
            return
        last = self.g.net[-1]
        last.weight.mul_(a); last.bias.mul_(a)
        self.net.cond[0].weight[:, :self.g.K].div_(a)

    def forward_c(self, x, c, logr):
        return self.net(x, c, logr)


def normalise(s, eps=1e-8):
    """s: (B, f, n, n) -> (x = s / r, log r) with r the per-field RMS."""
    r = s.pow(2).mean((-2, -1)).sqrt().clamp_min(eps)
    return s / r[..., None, None], r.log()


def n_params(m):
    return sum(p.numel() for p in m.parameters())
