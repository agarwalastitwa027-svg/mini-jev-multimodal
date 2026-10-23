"""Figures for the paper, drawn from the per-question result files and the study tables.

    python paper/make_figures.py --records results/exp --study results/study --out paper/figures

`results/exp` holds the JSON written by `python -m eval.evaluate --out` (they are published in the Hugging Face
results dataset under study/records/). Colours: one fixed colour per method in every figure, taken in order from a
colour-vision-safe categorical palette; the raw base model is neutral grey.
"""
import argparse
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.compare import arrays, ece_of, load  # noqa: E402

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C = {"base_raw": "#898781", "base_T": "#2a78d6", "narrow": "#eb6834", "released": "#1baf7a", "perm4": "#eda100"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300})


def parse_row(path, name):
    """First 'd Acc micro' cell of the paired-difference row for `name`: (estimate, lo, hi)."""
    for line in open(path, encoding="utf-8"):
        if line.startswith(f"| {name} |") and "p=" in line:
            m = re.search(r"([+-]\d+\.\d) \[([+-]\d+\.\d), ([+-]\d+\.\d)\]", line)
            return tuple(float(x) for x in m.groups())
    raise KeyError((path, name))


def fig_forest(study, out):
    rows = [("16 open datasets\n(3,678 questions)", "final_vs_base_test_more.md"),
            ("11 confirmation datasets\n(2,306 questions)", "final_vs_base_test_confirm_le26.md"),
            ("2 fresh datasets\n(600 questions)", "final_vs_base_test_fresh.md"),
            ("3 zero-shot sources\n(720 questions)", "final_vs_base_test_zeroshot.md")]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ys = np.arange(len(rows))[::-1]
    for off, key, lab in ((0.14, "narrow_mix", "Original 12-source mix"), (-0.14, "mini_jev", "Released model (broad mix)")):
        col = C["narrow"] if key == "narrow_mix" else C["released"]
        for y, (_, f) in zip(ys, rows):
            est, lo, hi = parse_row(os.path.join(study, f), key)
            ax.plot([lo, hi], [y + off] * 2, color=col, lw=1.6, solid_capstyle="round", zorder=3)
            ax.plot(est, y + off, "o", ms=5.5, color=col, mec="#fcfcfb", mew=1.0, zorder=4,
                    label=lab if y == ys[0] else None)
            ax.text(hi + 0.5, y + off, f"{est:+.1f}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.9, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Accuracy gain over base + temperature scaling (points; 95% CI)")
    ax.set_xlim(-4.5, 22)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_gain_over_baseline.png"))
    plt.close(fig)


def conf_correct(run, keys):
    c, f, _ = arrays(run, keys)
    return f, c


def fig_reliability(rec, out):
    f = "test_more.jsonl"
    runs = {"Base model (raw)": (os.path.join(rec, "base_T.json") + "@raw", C["base_raw"], "base_raw"),
            "Base + temperature scaling": (os.path.join(rec, "base_T.json"), C["base_T"], "base_T"),
            "Released model": (os.path.join(rec, "broad_seed1.json"), C["released"], "released")}
    data = {n: load(p, f) for n, (p, _, _) in runs.items()}
    keys = sorted(set.intersection(*[set(d) for d in data.values()]))
    fig, ax = plt.subplots(figsize=(3.9, 3.9))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1.0, ls="--", zorder=1)
    for n, (_, col, _) in runs.items():
        conf, corr = conf_correct(data[n], keys)
        ece = float(ece_of(corr, conf))
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 15:
                xs.append(conf[m].mean())
                ys.append(corr[m].mean())
        ax.plot(xs, ys, "-o", color=col, lw=1.6, ms=4.5, mec="#fcfcfb", mew=0.8, label=f"{n} (ECE {ece:.3f})", zorder=3)
    ax.set_xlabel("Confidence of the top option")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.2, 1.0)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_reliability.png"))
    plt.close(fig)


def raw_records(path):
    return json.load(open(path))["files"]


def fig_options(rec, out):
    files = ("test_more.jsonl", "test_fresh.jsonl", "test_zeroshot.jsonl")
    runs = {"Base + temperature scaling": ("base_T.json", C["base_T"]), "Original mix": ("full_seed0.json", C["narrow"]),
            "Released model": ("broad_seed1.json", C["released"]), "Released + 4 option orders": ("final_perm4.json", C["perm4"])}
    data = {}
    for n, (fn, _) in runs.items():
        d = raw_records(os.path.join(rec, fn))
        data[n] = {(f, str(r["id"]), r["key"]): (r["n_options"], float(r["probs"] is not None and int(np.argmax(r["probs"])) == r["gold"]))
                   for f in files for r in d[f]["records"]}
    keys = sorted(set.intersection(*[set(v) for v in data.values()]))
    buckets = [("2 (yes / no)", 2, 2), ("3-4", 3, 4), ("5-10", 5, 10), ("more than 10", 11, 999)]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    w = 0.2
    for j, (n, (_, col)) in enumerate(runs.items()):
        vals = []
        for _, lo, hi in buckets:
            ks = [k for k in keys if lo <= data["Base + temperature scaling"][k][0] <= hi]
            vals.append(100 * np.mean([data[n][k][1] for k in ks]))
        x = np.arange(len(buckets)) + (j - 1.5) * (w + 0.015)
        ax.bar(x, vals, w, color=col, label=n, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.7, f"{v:.0f}", ha="center", fontsize=7, color=INK2)
    counts = [sum(lo <= data["Base + temperature scaling"][k][0] <= hi for k in keys) for _, lo, hi in buckets]
    ax.set_xticks(np.arange(len(buckets)))
    ax.set_xticklabels([f"{b[0]}\nn = {c:,}" for b, c in zip(buckets, counts)], fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Number of options per question (16 + 2 + 3 held-out datasets pooled)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_by_option_count.png"))
    plt.close(fig)


def acc_of(path, f):
    recs = raw_records(path)[f]["records"]
    return 100 * np.mean([r["probs"] is not None and int(np.argmax(r["probs"])) == r["gold"] for r in recs])


def fig_breadth(rec, out):
    cols = [("16 open\ndatasets", "test_more.jsonl", ["full_seed%d.json" % s for s in range(3)],
             ["broad_seed0.json", "broad_seed1.json", "broad_seed2.json"]),
            ("2 fresh\ndatasets", "test_fresh.jsonl", ["full_seed%d.json" % s for s in range(3)],
             ["broad_seed0.json", "broad_seed1.json", "broad_seed2.json"]),
            ("3 zero-shot\nsources", "test_zeroshot.jsonl", ["full_seed%d.json" % s for s in range(3)],
             ["broad_seed0.json", "broad_seed1.json", "broad_seed2.json"]),
            ("27-100\noptions", "test_manyopt.jsonl", ["many_full_seed%d.json" % s for s in range(3)],
             ["many_broad_seed0.json", "broad_seed1.json", "broad_seed2.json"]),
            ("13 confirmation\ndatasets", "test_confirm.jsonl", ["conf_full_seed%d.json" % s for s in range(3)],
             ["conf_broad_seed0.json", "broad_seed1.json", "broad_seed2.json"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, (lab, f, nar, bro) in enumerate(cols):
        for k, (files, col, name) in enumerate(((nar, C["narrow"], "Original 12-source mix"), (bro, C["released"], "Broad mix (released recipe)"))):
            v = [acc_of(os.path.join(rec, x), f) for x in files]
            x = i + (-0.17 if k == 0 else 0.17)
            ax.plot([x - 0.1, x + 0.1], [np.mean(v)] * 2, color=col, lw=2.4, zorder=3)
            ax.plot([x] * len(v), v, "o", ms=4.5, color=col, mec="#fcfcfb", mew=0.8, zorder=4,
                    label=name if i == 0 else None)
        d = np.mean([acc_of(os.path.join(rec, x), f) for x in cols[i][3]]) - np.mean([acc_of(os.path.join(rec, x), f) for x in cols[i][2]])
        ax.annotate(f"{d:+.1f}", (i, max(np.mean([acc_of(os.path.join(rec, x), f) for x in cols[i][3]]), 0)),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(40, 83)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_breadth_three_seeds.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="results/exp")
    ap.add_argument("--study", default="results/study")
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_forest(a.study, a.out)
    fig_reliability(a.records, a.out)
    fig_options(a.records, a.out)
    fig_breadth(a.records, a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
