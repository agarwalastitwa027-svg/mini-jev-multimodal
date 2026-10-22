"""Training losses.

  soft cross-entropy  -sum_j t_j log p_j     (t = one-hot, or soft targets from data/teacher)
  RPS (score only)    sum_k (CDF_p(k) - CDF_t(k))^2 / (K - 1)
                      ranked probability score: predicting 4 when the answer is 5 costs less than
                      predicting 1; a strictly proper scoring rule, so honest probabilities minimise it
  Brier (optional)    sum_j (p_j - t_j)^2

mode "hard" reproduces the hard-label recipe (hard cross-entropy on the argmax target only) for the ablation.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def rps(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    K = p.shape[-1]
    if K < 2:
        return p.new_zeros(())
    return ((p.cumsum(-1) - t.cumsum(-1)) ** 2)[..., :-1].sum(-1) / (K - 1)


def decision_loss(logits: List[torch.Tensor], targets: List[List[float]], types: List[str],
                  mode: str = "soft", rps_weight: float = 1.0, brier_weight: float = 0.0,
                  label_smoothing: float = 0.0) -> torch.Tensor:
    total = logits[0].new_zeros((), dtype=torch.float32)
    for lg, tgt, ty in zip(logits, targets, types):
        lg = lg.float()
        t = torch.tensor(tgt, dtype=torch.float32, device=lg.device)
        if mode == "hard":
            total = total + F.cross_entropy(lg[None], t.argmax()[None])
            continue
        if label_smoothing > 0:
            t = (1 - label_smoothing) * t + label_smoothing / t.numel()
        logp = F.log_softmax(lg, -1)
        loss = -(t * logp).sum()
        p = logp.exp()
        if ty == "score" and rps_weight > 0:
            loss = loss + rps_weight * rps(p, t)
        if brier_weight > 0:
            loss = loss + brier_weight * ((p - t) ** 2).sum()
        total = total + loss
    return total / len(logits)
