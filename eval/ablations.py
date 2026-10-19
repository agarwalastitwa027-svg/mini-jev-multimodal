"""Run every comparison in one go and write a results table.

    python -m eval.ablations --config eval/ablations.json --out results

The config lists test files and entries. An entry with several `paths` (one run per seed) is
reported as mean ± std across seeds. Entries whose runs are missing, or whose server (Ollama)
is not reachable, are skipped with a note rather than stopping the whole table.

Outputs
    results/raw/<entry>__<seed>.json   per-question records (probabilities, gold, latency)
    results/ablations.json             all metrics
    results/ablations.md               the table for the README / model card
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re

from eval.evaluate import collect, report
from jev.data import read_jsonl
from jev.metrics import mean_std

# (metric key, column title, scale, decimals)
COLUMNS = [("accuracy", "Acc", 100, 1), ("skill", "Skill", 100, 1), ("ece", "ECE", 1, 3), ("brier", "Brier", 1, 3),
           ("nll", "NLL", 1, 3), ("sel_acc@0.9", "Acc@conf≥0.9", 100, 1), ("coverage@0.9", "Cov@0.9", 100, 1),
           ("row_ms_p50", "p50 ms", 1, 0)]
HUB_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


def slug(s):
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()


def free():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_entry(entry, tests, limit, raw_dir, image_root):
    from eval.predictors import make_predictor
    paths = entry.get("paths") or [entry.get("path")]
    per_seed = []
    for i, p in enumerate(paths):
        spec = dict(entry, image_root=image_root)
        spec.pop("paths", None)
        if entry["kind"] == "minijev":
            if not p or not (os.path.isdir(p) or (HUB_ID.match(p) and not p.startswith(("runs/", "./")))):
                print(f"  [skip] {entry['name']}: run not found: {p}")
                continue
            spec["path"] = p
        try:
            pred = make_predictor(spec)
        except Exception as e:
            print(f"  [skip] {entry['name']}: {type(e).__name__}: {e}")
            continue
        res = {}
        for tname, f in tests.items():
            if not os.path.exists(f):
                continue
            recs = collect(pred, read_jsonl(f, limit), progress=0)
            rep = report(recs)
            res[tname] = rep
            json.dump({"entry": entry["name"], "path": p, "test": tname, "records": recs},
                      open(os.path.join(raw_dir, f"{slug(entry['name'])}__{i}__{tname}.json"), "w"))
            al = rep["all"]
            print(f"  {entry['name']} [{i}] {tname}: acc {al['accuracy']:.4f} ece {al['ece']:.4f}", flush=True)
        per_seed.append(res)
        del pred
        free()
    return per_seed


def aggregate(per_seed):
    """{test: {group: {metric: (mean, std)}}} across seeds."""
    out = {}
    if not per_seed:
        return out
    for t in per_seed[0]:
        out[t] = {}
        for g in per_seed[0][t]:
            if g.startswith("_"):
                continue
            out[t][g] = {}
            for k in per_seed[0][t][g]:
                vals = [s[t][g].get(k) for s in per_seed if t in s and g in s[t]]
                if all(isinstance(v, (int, float)) or v is None for v in vals):
                    out[t][g][k] = mean_std(vals)
    return out


def fmt(ms, scale, dec):
    m, s = ms if isinstance(ms, (tuple, list)) else (None, None)
    if m is None:
        return "–"
    txt = f"{m * scale:.{dec}f}"
    return f"{txt} ± {s * scale:.{dec}f}" if s else txt


def to_markdown(results, tests):
    lines = []
    for t in tests:
        for group in ("all", "modality:text", "modality:vision"):
            rows = [(name, r[t][group]) for name, r in results.items() if t in r and group in r[t]]
            if not rows:
                continue
            lines.append(f"\n### {t} — {group.replace('modality:', '')}\n")
            lines.append("| Model | n | " + " | ".join(c[1] for c in COLUMNS) + " |")
            lines.append("|---|---|" + "---|" * len(COLUMNS))
            for name, m in rows:
                n = m.get("n", (None,))[0]
                lines.append(f"| {name} | {int(n) if n else '–'} | " +
                             " | ".join(fmt(m.get(k), sc, d) for k, _, sc, d in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="eval/ablations.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", nargs="*", help="entry names (prefix match) to run")
    ap.add_argument("--image-root", default=".")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    raw = os.path.join(a.out, "raw")
    os.makedirs(raw, exist_ok=True)
    path = os.path.join(a.out, "ablations.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    for e in cfg["entries"]:
        if a.only and not any(e["name"].startswith(o) for o in a.only):
            continue
        print(f"== {e['name']}")
        per_seed = run_entry(e, cfg["tests"], cfg.get("limit"), raw, a.image_root)
        if per_seed:
            results[e["name"]] = aggregate(per_seed)
            results[e["name"]]["_seeds"] = len(per_seed)
        json.dump(results, open(path, "w"), indent=1)
    md = to_markdown(results, cfg["tests"])
    open(os.path.join(a.out, "ablations.md"), "w").write(
        "# Ablations\n\nMean ± std across seeds where an entry has several runs. Acc, Skill, Acc@conf≥0.9 and "
        "Cov@0.9 in %. Skill = (acc − chance) / (1 − chance). Unanswered rows count as wrong.\n" + md)
    print(md)


if __name__ == "__main__":
    main()
