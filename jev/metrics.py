"""Decision metrics: accuracy, chance-adjusted skill, NLL, Brier, ECE, selective accuracy, AURC.

Every metric works on a list of (probabilities, target) pairs, where target is the gold option index.
Rows a model could not answer (unsupported / error) are passed as probs=None and count as wrong,
as on the Decision Index.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


def summarize(items: Sequence[tuple], bins: int = 15, thresholds=(0.5, 0.7, 0.9)) -> Dict:
    """items: (probs or None, gold_index). Returns a metrics dict."""
    n = len(items)
    if n == 0:
        return {"n": 0}
    correct, conf, nll, brier, chance, unanswered = [], [], 0.0, 0.0, 0.0, 0
    for p, y in items:
        k = len(p) if p is not None else None
        if p is None:
            unanswered += 1
            correct.append(0)
            conf.append(0.0)
            nll += math.log(1e6)
            brier += 2.0
            continue
        chance += 1.0 / k
        j = max(range(k), key=p.__getitem__)
        correct.append(int(j == y))
        conf.append(float(p[j]))
        nll -= math.log(max(p[y], 1e-12))
        brier += sum((pi - (1.0 if i == y else 0.0)) ** 2 for i, pi in enumerate(p))
    answered = n - unanswered
    chance = chance / answered if answered else 0.0
    acc = sum(correct) / n
    out = {"n": n, "unanswered": unanswered, "accuracy": acc, "chance": chance,
           "skill": (acc - chance) / (1 - chance) if chance < 1 else 0.0,
           "nll": nll / n, "brier": brier / n, "ece": ece(conf, correct, bins),
           "aurc": aurc(conf, correct)}
    for t in thresholds:
        sel = [c for c, f in zip(correct, conf) if f >= t]
        out[f"coverage@{t}"] = len(sel) / n
        out[f"sel_acc@{t}"] = sum(sel) / len(sel) if sel else None
    return out


def ece(conf: List[float], correct: List[int], bins: int = 15) -> float:
    n = len(conf)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            avg = sum(conf[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - avg)
    return total


def aurc(conf: List[float], correct: List[int]) -> float:
    """Area under the risk-coverage curve (lower is better): answer most-confident first."""
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    risk, errs = 0.0, 0
    for r, i in enumerate(order, 1):
        errs += 1 - correct[i]
        risk += errs / r
    return risk / len(order) if order else 0.0


def reliability_bins(conf: List[float], correct: List[int], bins: int = 10) -> List[Dict]:
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo < c <= hi]
        if idx:
            out.append({"lo": lo, "hi": hi, "n": len(idx),
                        "confidence": sum(conf[i] for i in idx) / len(idx),
                        "accuracy": sum(correct[i] for i in idx) / len(idx)})
    return out


def mean_std(values: List[Optional[float]]):
    v = [x for x in values if x is not None]
    if not v:
        return None, None
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, s
