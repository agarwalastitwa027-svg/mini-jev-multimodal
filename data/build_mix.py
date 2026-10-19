"""Combine v2 source files into the training mix and the test sets.

    python data/build_mix.py --inputs data/v2/src/*.jsonl --out data/v2

Outputs in --out:
    train.jsonl, val.jsonl          training / checkpoint selection (both modalities)
    test_text.jsonl                 held-out text rows from sources seen in training
    test_vision.jsonl               held-out image rows from sources seen in training
    test_zeroshot.jsonl             rows from sources never trained on (split "zeroshot")
    mix_report.json                 counts per source, modality and split

Splits: a row's own `split` field wins (train/val/test/zeroshot; "bench" rows are skipped here).
Rows without one are assigned by a hash of their `group` (or id), so every question about the same
image or passage lands in the same split. Rows from Decision Index benchmarks are refused.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.common import is_di_family, split_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="data/v2")
    ap.add_argument("--val", type=float, default=0.05)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--max-train-per-source", type=int, default=0, help="0 = no cap")
    ap.add_argument("--holdout-sources", nargs="*", default=[], help="extra sources moved to zero-shot test")
    ap.add_argument("--max-test-per-source", type=int, default=300)
    a = ap.parse_args()

    paths = sorted({p for pat in a.inputs for p in glob.glob(pat)})
    out = {k: [] for k in ("train", "val", "test_text", "test_vision", "test_zeroshot")}
    counts = defaultdict(lambda: defaultdict(int))
    per_source_train = defaultdict(int)
    per_source_test = defaultdict(int)
    for p in paths:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            src = r.get("source", os.path.basename(p))
            if is_di_family(src):
                raise SystemExit(f"{p}: source {src!r} is a Decision Index benchmark; remove it from training")
            mod = r.get("modality") or ("vision" if r.get("images") else "text")
            r["modality"] = mod
            sp = r.get("split") or split_of(r.get("group") or r.get("id") or line, a.val, a.test)
            if src in a.holdout_sources:
                sp = "zeroshot"
            if sp == "bench":
                continue
            if sp == "train":
                if a.max_train_per_source and per_source_train[src] >= a.max_train_per_source:
                    continue
                per_source_train[src] += 1
                key = "train"
            elif sp == "val":
                key = "val"
            elif sp in ("test", "zeroshot"):
                if per_source_test[(src, sp)] >= a.max_test_per_source:
                    continue
                per_source_test[(src, sp)] += 1
                key = "test_zeroshot" if sp == "zeroshot" else f"test_{mod}"
            else:
                raise SystemExit(f"unknown split {sp!r} in {p}")
            out[key].append(r)
            counts[f"{src} ({mod})"][key] += 1

    trained = {r["source"] for r in out["train"]}
    leaked = {r["source"] for r in out["test_zeroshot"]} & trained
    if leaked:
        raise SystemExit(f"zero-shot sources also appear in training: {sorted(leaked)}")

    os.makedirs(a.out, exist_ok=True)
    for k, rows in out.items():
        with open(os.path.join(a.out, f"{k}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    q = {k: sum(len(r["questions"]) for r in rows) for k, rows in out.items()}
    mods = {k: {m: sum(len(r["questions"]) for r in rows if r["modality"] == m) for m in ("text", "vision")}
            for k, rows in out.items()}
    report = {"rows": {k: len(v) for k, v in out.items()}, "questions": q, "questions_by_modality": mods,
              "by_source": {s: dict(c) for s, c in sorted(counts.items())}, "inputs": paths}
    json.dump(report, open(os.path.join(a.out, "mix_report.json"), "w"), indent=1)
    for k in out:
        print(f"{k:14} rows {report['rows'][k]:6}  questions {q[k]:6}  {mods[k]}")


if __name__ == "__main__":
    main()
