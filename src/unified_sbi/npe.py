"""Neural posterior estimation with sbi's MAF, trained by an explicit loop.

The architecture is the one ``ltu-ili``'s ``load_nde_sbi(model="maf")`` builds (sbi's
``posterior_nn``). Training is done here rather than through sbi's trainer so that the
train/validation split is ours: every method sees exactly the same split, and validation
log-probabilities can be reported in physical theta coordinates.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sbi.neural_nets import posterior_nn


@dataclass
class NPEConfig:
    model: str = "maf"
    hidden_features: int = 50
    num_transforms: int = 5
    batch_size: int = 128
    lr: float = 5e-4
    clip_norm: float = 5.0
    stop_after_epochs: int = 20
    max_epochs: int = 2000

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainedNPE:
    net: torch.nn.Module
    best_val_log_prob: float   # mean over the validation split, in the estimator's own coordinates
    best_epoch: int
    epochs_trained: int
    history: list              # (epoch, train log prob, val log prob)
    seconds: float


@torch.no_grad()
def mean_log_prob(net, target, context, batch: int = 4096) -> float:
    total = 0.0
    for i in range(0, len(target), batch):
        total += net.log_prob(target[None, i:i + batch], context[i:i + batch]).sum().item()
    return total / len(target)


def train_npe(target_train, context_train, target_val, context_val,
              cfg: NPEConfig, seed: int = 0) -> TrainedNPE:
    """Maximum-likelihood NPE with early stopping on the validation log-probability."""
    torch.manual_seed(seed)
    build = posterior_nn(model=cfg.model, hidden_features=cfg.hidden_features,
                         num_transforms=cfg.num_transforms)
    net = build(target_train, context_train).to(target_train.device)   # z-scores from training set
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    n = len(target_train)
    best, best_epoch, best_state, history = -np.inf, 0, None, []
    start = time.time()
    epoch = 0
    for epoch in range(cfg.max_epochs):
        net.train()
        perm = torch.randperm(n, device=target_train.device)
        train_total = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = net.loss(target_train[idx], context_train[idx]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_norm)
            opt.step()
            train_total -= loss.item() * len(idx)
        net.eval()
        val = mean_log_prob(net, target_val, context_val)
        history.append((epoch, train_total / n, val))
        if val > best:
            best, best_epoch, best_state = val, epoch, copy.deepcopy(net.state_dict())
        elif epoch - best_epoch >= cfg.stop_after_epochs:
            break
    net.load_state_dict(best_state)
    net.eval().requires_grad_(False)
    return TrainedNPE(net, best, best_epoch, epoch + 1, history, time.time() - start)
