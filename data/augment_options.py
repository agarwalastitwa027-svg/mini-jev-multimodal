"""Option-count augmentation: extra training copies of closed-label questions with many more options.

    python data/augment_options.py --inputs data/broad/train_clean.jsonl --out data/broad_aug/extra.jsonl

Why: the answer head scored below the base model on questions with more than 10 options (position and
length bias that the 2-5 option training questions never exposed). For every source whose options come from a
closed label set (e.g. 37 pet breeds shown 4 at a time), this writes copies of its questions padded with extra
labels from the same set up to 12 / 20 / 40 options. The gold answer is unchanged; nothing is dropped.
Only `choice` questions with multi-character option keys are padded (A/B/C/D-style keys cannot be extended).
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", nargs="+", type=int, default=[12, 20, 40])
    ap.add_argument("--min-pool", type=int, default=16, help="sources with fewer distinct option keys are skipped")
    ap.add_argument("--max-extra", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = [json.loads(l) for p in a.inputs for l in open(p, encoding="utf-8") if l.strip()]
    pool, nq = defaultdict(Counter), Counter()
    for r in rows:
        for q in r["questions"].values():
            if q["type"] == "choice" and isinstance(q.get("criteria"), dict):
                nq[r["source"]] += 1
                for k, v in q["criteria"].items():
                    pool[r["source"]][k] += 1
    closed = {s for s, c in pool.items()
              if len(c) >= a.min_pool and len(c) <= 0.25 * nq[s] and all(len(k) > 1 for k in c)}
    print("closed-label sources:", {s: len(pool[s]) for s in sorted(closed)})
    desc = {}  # key -> description, per source (descriptions of closed labels are the labels themselves)
    for r in rows:
        if r["source"] in closed:
            for q in r["questions"].values():
                if q["type"] == "choice" and isinstance(q.get("criteria"), dict):
                    for k, v in q["criteria"].items():
                        desc.setdefault((r["source"], k), v)
    cands = [r for r in rows if r["source"] in closed and r.get("split", "train") == "train"]
    rng.shuffle(cands)
    out, counts = [], Counter()
    for r in cands:
        if len(out) >= a.max_extra:
            break
        new_q, ok = {}, False
        for key, q in r["questions"].items():
            if q["type"] != "choice" or not isinstance(q.get("criteria"), dict) or any(len(k) <= 1 for k in q["criteria"]):
                new_q[key] = q
                continue
            labels = list(pool[r["source"]])
            have = list(q["criteria"])
            target = min(rng.choice(a.targets), len(labels))
            if target <= len(have):
                new_q[key] = q
                continue
            extra = rng.sample([l for l in labels if l not in q["criteria"]], target - len(have))
            crit = {**q["criteria"], **{k: desc[(r["source"], k)] for k in extra}}
            items = list(crit.items())
            rng.shuffle(items)
            new_q[key] = {**q, "criteria": dict(items)}
            ok = True
        if ok:
            nr = dict(r)
            nr["questions"] = new_q
            nr["id"] = r["id"] + "#aug"
            nr["group"] = r.get("group", r["id"]) + "#aug"
            out.append(nr)
            counts[r["source"]] += 1
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} augmented rows -> {a.out}", dict(counts))


if __name__ == "__main__":
    main()
