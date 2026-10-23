"""Compare evaluation runs on the same questions, with bootstrap confidence intervals.

    python -m eval.compare --file test_more.jsonl --ref base \
        --runs base=results/exp/base_T.json@raw ours=results/exp/broad_seed1.json --out results/study/x.md

Each run file is the JSON written by `python -m eval.evaluate --out`. Records are matched on
(row id, question key); only questions every run answered or refused are compared. The bootstrap resamples
questions within each dataset (stratified), 95 % percentile intervals. "Macro" averages the per-dataset
accuracies, so a big dataset cannot dominate. Differences are paired: the same resampled questions are used
for every run, so the interval on (run - ref) is much tighter than two independent intervals.
Temperatures stored in a run file (post-hoc scaling) are applied to its probabilities first.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

BINS = 15


def load(path, file, max_options=None):
    raw = path.endswith("@raw")  # "run=file.json@raw": ignore the temperatures stored in the file
    d = json.load(open(path[:-4] if raw else path))
    temps = {} if raw else (d.get("temps") or {})
    recs = d["files"][file]["records"]
    out = {}
    for r in recs:
        if max_options and r.get("n_options", 0) > max_options:
            continue
        p = r["probs"]
        if p is not None and temps.get(r["type"], 1.0) != 1.0:
            lg = np.log(np.maximum(np.array(p), 1e-12)) / temps[r["type"]]
            e = np.exp(lg - lg.max())
            p = (e / e.sum()).tolist()
        out[(str(r["id"]), r["key"])] = (r["source"], r["gold"], p)
    return out


def arrays(recs, keys):
    corr, conf, nll = [], [], []
    for k in keys:
        _, gold, p = recs[k]
        if p is None:
            corr.append(0.0); conf.append(0.0); nll.append(math.log(1e6))
            continue
        j = int(np.argmax(p))
        corr.append(float(j == gold)); conf.append(float(p[j])); nll.append(-math.log(max(p[gold], 1e-12)))
    return np.array(corr), np.array(conf), np.array(nll)


def ece_of(corr, conf):
    """Works on (N,) or (B, N) arrays."""
    b = np.clip(np.ceil(conf * BINS).astype(int) - 1, 0, BINS - 1)
    total = 0.0
    for k in range(BINS):
        m = (b == k)
        total = total + np.abs(((corr - conf) * m).sum(-1))
    return total / corr.shape[-1]


def pct(x):
    return f"{100 * x:.1f}"


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="test file name inside the run JSONs, e.g. test_more.jsonl")
    ap.add_argument("--runs", nargs="+", required=True, help="name=path/to/result.json")
    ap.add_argument("--ref", default=None, help="run name to compute paired differences against")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-options", type=int, default=None, help="only questions with at most this many options")
    a = ap.parse_args()

    runs = {}
    for spec in a.runs:
        name, path = spec.split("=", 1)
        runs[name] = load(path, a.file, a.max_options)
    keys = sorted(set.intersection(*[set(r) for r in runs.values()]))
    dropped = max(len(r) for r in runs.values()) - len(keys)
    src = {k: next(iter(runs.values()))[k][0] for k in keys}
    order = sorted(range(len(keys)), key=lambda i: (src[keys[i]], i))
    keys = [keys[i] for i in order]
    names = list(runs)
    A = {n: arrays(runs[n], keys) for n in names}
    sources = sorted(set(src.values()))
    cols = {s: [i for i, k in enumerate(keys) if src[k] == s] for s in sources}

    rng = np.random.default_rng(a.seed)
    idx = np.concatenate([np.array(cols[s])[rng.integers(0, len(cols[s]), (a.boot, len(cols[s])))] for s in sources], 1)
    bounds, o = {}, 0
    for s in sources:
        bounds[s] = (o, o + len(cols[s]))
        o += len(cols[s])

    def boot_stats(n):
        c, f, l = (x[idx] for x in A[n])
        micro = c.mean(1)
        macro = np.mean([c[:, bounds[s][0]:bounds[s][1]].mean(1) for s in sources], 0)
        return {"micro": micro, "macro": macro, "ece": ece_of(c, f), "nll": l.mean(1)}

    B = {n: boot_stats(n) for n in names}
    point = {n: {"micro": A[n][0].mean(),
                 "macro": np.mean([A[n][0][cols[s]].mean() for s in sources]),
                 "ece": float(ece_of(A[n][0], A[n][1])), "nll": A[n][2].mean()} for n in names}

    L = [f"# {a.file}: {len(keys)} questions, {len(sources)} datasets, {a.boot} bootstrap resamples", ""]
    if dropped:
        L += [f"({dropped} questions were not answered by every run and are left out.)", ""]
    L += ["| Run | Acc micro % [95% CI] | Acc macro % [95% CI] | ECE [95% CI] | NLL [95% CI] |", "|---|---|---|---|---|"]
    for n in names:
        row = [n]
        for m, f in (("micro", pct), ("macro", pct), ("ece", lambda x: f"{x:.3f}"), ("nll", lambda x: f"{x:.2f}")):
            lo, hi = ci(B[n][m])
            row.append(f"{f(point[n][m])} [{f(lo)}, {f(hi)}]")
        L.append("| " + " | ".join(row) + " |")
    if a.ref:
        L += ["", f"Paired difference vs `{a.ref}` (run - ref; positive accuracy is better, negative ECE/NLL is better). "
              "p = two-sided bootstrap p-value.", "",
              "| Run | d Acc micro (pts) | d Acc macro (pts) | d ECE | d NLL |", "|---|---|---|---|---|"]
        for n in names:
            if n == a.ref:
                continue
            row = [n]
            for m, scale, fmt in (("micro", 100, "{:+.1f}"), ("macro", 100, "{:+.1f}"), ("ece", 1, "{:+.3f}"), ("nll", 1, "{:+.2f}")):
                d = B[n][m] - B[a.ref][m]
                lo, hi = ci(d * scale)
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                row.append(f"{fmt.format((point[n][m] - point[a.ref][m]) * scale)} [{fmt.format(lo)}, {fmt.format(hi)}] p={max(p, 1 / a.boot):.3f}")
            L.append("| " + " | ".join(row) + " |")
    L += ["", "Per dataset (accuracy %, 95% CI):", "",
          "| Dataset | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for s in sources:
        lo_, hi_ = bounds[s]
        cells = []
        for n in names:
            c = A[n][0][cols[s]]
            bs = np.array(A[n][0])[idx][:, lo_:hi_].mean(1)
            lo, hi = ci(bs)
            cells.append(f"{pct(c.mean())} [{pct(lo)}, {pct(hi)}]")
        L.append(f"| {s} | {len(cols[s])} | " + " | ".join(cells) + " |")
    text = "\n".join(L) + "\n"
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
