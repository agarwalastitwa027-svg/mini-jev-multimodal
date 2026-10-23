"""Fresh held-out test set: datasets that appear nowhere in mini-jev's train/val/test files."""
import json
import random
import sys

from datasets import load_dataset

R = random.Random(7)
N = 300
rows = []


def add(source, i, state, qs, exp):
    rows.append({"id": f"{source}/fresh/{i}", "group": f"{source}/fresh/{i}", "source": source, "modality": "text",
                 "split": "test", "state": state, "images": [], "questions": qs, "expected": exp})


arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").shuffle(seed=7)
n = 0
for r in arc:
    labels, texts = r["choices"]["label"], r["choices"]["text"]
    if r["answerKey"] not in labels:
        continue
    add("arc_challenge", n, "", {"answer": {"type": "choice", "instructions": r["question"],
                                            "criteria": dict(zip(labels, texts))}}, {"answer": r["answerKey"]})
    n += 1
    if n == N:
        break

irony = load_dataset("cardiffnlp/tweet_eval", "irony", split="test").shuffle(seed=7)
for i, r in enumerate(irony.select(range(min(N, len(irony))))):
    add("tweet_irony", i, r["text"], {"ironic": {"type": "noul", "instructions": "Is this tweet ironic or sarcastic?"}},
        {"ironic": bool(r["label"])})

out = sys.argv[1]
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "rows ->", out)
