"""Fit one temperature per question type on validation data, then evaluate test files.

    python calibrate.py --run runs/full_seed0 --val data/mix/val.jsonl \
        --test data/mix/test_text.jsonl data/mix/test_vision.jsonl data/mix/test_zeroshot.jsonl

Writes into the run folder:
    head_calibrated.pt   the answer head with fitted temperatures (loaded automatically afterwards)
    calibration.json     temperatures + metrics before/after for every test file
    reliability.png      reliability diagram (if matplotlib is installed)
"""
import argparse
import json
import math
import os

import torch

from eval.evaluate import collect, report, rescale
from jev.data import read_jsonl
from jev.schema import TYPES, TYPE_INDEX


def fit_temperature(pairs, min_n=1):
    """pairs: (probs, gold). Minimise NLL of softmax(log p / T) over T > 0 (grid + golden section)."""
    pairs = [(p, y) for p, y in pairs if p is not None]
    if len(pairs) < min_n:
        return 1.0

    def nll(T):
        return -sum(math.log(max(rescale(p, T)[y], 1e-12)) for p, y in pairs) / len(pairs)
    grid = [0.05 * 1.15 ** i for i in range(42)]  # 0.05 .. ~16
    best = min(grid, key=nll)
    lo, hi = best / 1.15, best * 1.15
    g = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--val-limit", type=int, default=3000)
    ap.add_argument("--test-limit", type=int, default=None)
    ap.add_argument("--min-per-type", type=int, default=50,
                    help="types with fewer validation questions share the pooled temperature")
    ap.add_argument("--image-root", default=".")
    a = ap.parse_args()

    from eval.predictors import MiniJevPredictor
    pred = MiniJevPredictor(a.run, image_root=a.image_root)
    pred.d.model.head.temperature.fill_(1.0)  # raw probabilities while fitting

    recs = collect(pred, read_jsonl(a.val, a.val_limit))
    pooled = fit_temperature([(r["probs"], r["gold"]) for r in recs])
    temps = {}
    for t in TYPES:
        pairs = [(r["probs"], r["gold"]) for r in recs if r["type"] == t and r["probs"] is not None]
        temps[t] = fit_temperature(pairs) if len(pairs) >= a.min_per_type else pooled
    print("temperatures:", {k: round(v, 3) for k, v in temps.items()}, "| pooled", round(pooled, 3))

    head = pred.d.model.head
    for t, v in temps.items():
        head.temperature[TYPE_INDEX[t]] = v
    torch.save({k: v.cpu() for k, v in head.state_dict().items()}, os.path.join(a.run, "head_calibrated.pt"))

    out = {"temperatures": temps, "pooled": pooled, "val": {"before": report(recs), "after": report(recs, temps)},
           "test": {}}
    for f in a.test:
        head.temperature.fill_(1.0)
        trecs = collect(pred, read_jsonl(f, a.test_limit))
        before, after = report(trecs), report(trecs, temps)
        out["test"][os.path.basename(f)] = {"before": before, "after": after}
        b, c = before["all"], after["all"]
        print(f"{os.path.basename(f)}: acc {c['accuracy']:.4f} | ECE {b['ece']:.4f} -> {c['ece']:.4f} | "
              f"NLL {b['nll']:.4f} -> {c['nll']:.4f} | Brier {b['brier']:.4f} -> {c['brier']:.4f} | "
              f"sel_acc@0.9 {c['sel_acc@0.9']} (coverage {c['coverage@0.9']:.2f})")
    json.dump(out, open(os.path.join(a.run, "calibration.json"), "w"), indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        for name, d in [("val", out["val"]["after"])] + [(k, v["after"]) for k, v in out["test"].items()]:
            bins = d["_reliability"]
            ax.plot([x["confidence"] for x in bins], [x["accuracy"] for x in bins], "o-", label=name, ms=3)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(a.run, "reliability.png"), dpi=150)
    except ImportError:
        pass
    print(f"wrote {a.run}/head_calibrated.pt and calibration.json")


if __name__ == "__main__":
    main()
