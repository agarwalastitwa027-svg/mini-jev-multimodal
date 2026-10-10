"""Split a synthetic JSONL by task (whole tasks held out -> honest test of unseen tasks).

    python data/split.py data/synthetic.jsonl --val 0.05 --test 0.10
    -> data/synthetic_train.jsonl, data/synthetic_val.jsonl, data/synthetic_test.jsonl
"""
import argparse
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--val", type=float, default=0.05)
ap.add_argument("--test", type=float, default=0.10)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.path, encoding="utf-8") if l.strip()]
tasks = sorted({r.get("task_id", i) for i, r in enumerate(rows)})
random.Random(0).shuffle(tasks)
n_test, n_val = int(len(tasks) * a.test), int(len(tasks) * a.val)
test, val = set(tasks[:n_test]), set(tasks[n_test:n_test + n_val])
out = {k: open(a.path.replace(".jsonl", f"_{k}.jsonl"), "w", encoding="utf-8")
       for k in ["train", "val", "test"]}
counts = {"train": 0, "val": 0, "test": 0}
for i, r in enumerate(rows):
    tid = r.get("task_id", i)
    k = "test" if tid in test else "val" if tid in val else "train"
    out[k].write(json.dumps(r, ensure_ascii=False) + "\n")
    counts[k] += 1
print(counts)
