"""Per-dataset table (accuracy, calibration error, log-loss) for several runs on one test file.

    python -m eval.report_table --file test_more.jsonl \
        --runs "Base=results/exp/base_T.json@raw" "Base + temp. scaling=results/exp/base_T.json" "mini-Jev=results/exp/broad_seed1.json" \
        --calib 1 2 --out results/final_test_more.md

--calib lists the run indices (0-based) that also get ECE and NLL columns. Questions with more options than
a run can handle (the 26-letter base readout) count as wrong, as on the Decision Index.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from eval.compare import arrays, ece_of, load, pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="Label=path.json[@raw]")
    ap.add_argument("--calib", nargs="*", type=int, default=[])
    ap.add_argument("--max-options", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    labels, runs = [], []
    for spec in a.runs:
        lab, path = spec.split("=", 1)
        labels.append(lab)
        runs.append(load(path, a.file, a.max_options))
    keys = sorted(set.intersection(*[set(r) for r in runs]))
    src = {k: runs[0][k][0] for k in keys}
    sources = sorted(set(src.values()))
    arr = [arrays(r, keys) for r in runs]
    chance = []
    for k in keys:
        chance.append(1.0)
    # chance level = mean of 1/n_options, taken from the first run's probabilities (or its gold row)
    nopt = {}
    for r in runs:
        for k, (_, _, p) in r.items():
            if p is not None:
                nopt.setdefault(k, len(p))
    head = ["Dataset", "n", "Chance %"] + [f"{l} acc" for l in labels] + \
           [f"{labels[i]} ECE" for i in a.calib] + [f"{labels[i]} NLL" for i in a.calib]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    def row(name, sel):
        n = len(sel)
        ch = 100 * np.mean([1.0 / nopt.get(keys[i], 2) for i in sel])
        cells = [name, str(n), f"{ch:.0f}"] + [pct(arr[j][0][sel].mean()) for j in range(len(runs))]
        cells += [f"{float(ece_of(arr[i][0][sel], arr[i][1][sel])):.3f}" for i in a.calib]
        cells += [f"{arr[i][2][sel].mean():.2f}" for i in a.calib]
        lines.append("| " + " | ".join(cells) + " |")

    for s in sources:
        row(s, np.array([i for i, k in enumerate(keys) if src[k] == s]))
    row("**ALL**", np.arange(len(keys)))
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
