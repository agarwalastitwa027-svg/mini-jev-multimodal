"""Compare groups of runs (e.g. 3 seeds of recipe A vs 3 seeds of recipe B) with a paired bootstrap.

    python -m eval.compare_groups --file test_confirm.jsonl --ref narrow \
        --groups narrow=a0.json,a1.json,a2.json broad=b0.json,b1.json,b2.json --out results/groups_confirm.md

Per group: the mean and standard deviation over its runs of each metric (seed-to-seed spread), and a bootstrap
95 % interval for the group mean (questions resampled within each dataset, the same resample for every run, so
the difference between groups is paired). Reports the difference of group means vs --ref with a p-value.
Temperatures stored in a run file are applied first (see eval.compare).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from eval.compare import arrays, ece_of, load, pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--groups", nargs="+", required=True, help="name=run1.json,run2.json,...")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-options", type=int, default=None, help="only questions with at most this many options")
    a = ap.parse_args()

    groups = {}
    for spec in a.groups:
        name, paths = spec.split("=", 1)
        groups[name] = [load(p, a.file, a.max_options) for p in paths.split(",")]
    allruns = [r for g in groups.values() for r in g]
    keys = sorted(set.intersection(*[set(r) for r in allruns]))
    src = {k: allruns[0][k][0] for k in keys}
    keys = sorted(keys, key=lambda k: (src[k], k))
    sources = sorted(set(src.values()))
    cols = {s: [i for i, k in enumerate(keys) if src[k] == s] for s in sources}
    rng = np.random.default_rng(a.seed)
    idx = np.concatenate([np.array(cols[s])[rng.integers(0, len(cols[s]), (a.boot, len(cols[s])))] for s in sources], 1)
    bounds, o = {}, 0
    for s in sources:
        bounds[s] = (o, o + len(cols[s]))
        o += len(cols[s])

    def stats(run, boot):
        c, f, l = arrays(run, keys)
        if not boot:
            return {"micro": c.mean(), "macro": np.mean([c[cols[s]].mean() for s in sources]),
                    "ece": float(ece_of(c, f)), "nll": l.mean()}
        cb, fb, lb = c[idx], f[idx], l[idx]
        return {"micro": cb.mean(1), "macro": np.mean([cb[:, bounds[s][0]:bounds[s][1]].mean(1) for s in sources], 0),
                "ece": ece_of(cb, fb), "nll": lb.mean(1)}

    pt = {n: [stats(r, False) for r in g] for n, g in groups.items()}
    bt = {n: {m: np.mean([stats(r, True)[m] for r in g], 0) for m in ("micro", "macro", "ece", "nll")}
          for n, g in groups.items()}
    L = [f"# {a.file}: {len(keys)} questions, {len(sources)} datasets, groups of runs, {a.boot} bootstrap resamples", "",
         "| Group | runs | Acc micro % | Acc macro % | ECE | NLL |", "|---|---|---|---|---|---|"]
    fm = {"micro": lambda x: pct(x), "macro": lambda x: pct(x), "ece": lambda x: f"{x:.3f}", "nll": lambda x: f"{x:.2f}"}
    for n in groups:
        row = [n, str(len(groups[n]))]
        for m in ("micro", "macro", "ece", "nll"):
            v = np.array([p[m] for p in pt[n]])
            lo, hi = np.percentile(bt[n][m], [2.5, 97.5])
            sd = f" ± {fm[m](v.std(ddof=1))}" if len(v) > 1 else ""
            row.append(f"{fm[m](v.mean())}{sd} [{fm[m](lo)}, {fm[m](hi)}]")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("Cells: mean over runs ± sd across runs [95% bootstrap interval of the mean].")
    if a.ref:
        L += ["", f"Difference of group means vs `{a.ref}` (positive accuracy better, negative ECE/NLL better):", "",
              "| Group | d Acc micro (pts) | d Acc macro (pts) | d ECE | d NLL |", "|---|---|---|---|---|"]
        for n in groups:
            if n == a.ref:
                continue
            row = [n]
            for m, sc, fmt in (("micro", 100, "{:+.1f}"), ("macro", 100, "{:+.1f}"), ("ece", 1, "{:+.3f}"), ("nll", 1, "{:+.2f}")):
                d = bt[n][m] - bt[a.ref][m]
                est = (np.mean([p[m] for p in pt[n]]) - np.mean([p[m] for p in pt[a.ref]])) * sc
                lo, hi = np.percentile(d * sc, [2.5, 97.5])
                p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / a.boot)
                row.append(f"{fmt.format(est)} [{fmt.format(lo)}, {fmt.format(hi)}] p={p:.3f}")
            L.append("| " + " | ".join(row) + " |")
    text = "\n".join(L) + "\n"
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
