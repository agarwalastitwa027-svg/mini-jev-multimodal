"""Convert public Hugging Face datasets into mini-Jev decision examples.

Output: JSONL lines {"context","question","options","answer","source"}
    data/train.jsonl, data/val.jsonl, data/test.jsonl
    data/test_zeroshot.jsonl  (datasets held out entirely -> tests generalisation to unseen tasks)

Usage:
    python data/make_public.py --per-dataset 4000 --holdout emotion
"""
import argparse
import json
import os
import random

from datasets import load_dataset

R = random.Random(0)
MAX_OPTS = 8  # big label sets (banking77 / clinc) are sub-sampled to the gold + distractors


def pick(qs):
    return R.choice(qs)


def subsample(options, gold):
    options = list(dict.fromkeys(options))
    if len(options) > MAX_OPTS:
        others = [o for o in options if o != gold]
        options = [gold] + R.sample(others, MAX_OPTS - 1)
    R.shuffle(options)
    return options


def names_of(ds, col):
    return ds.features[col].names


# ---------- one converter per dataset: (hf_name, config, split_map, fn) ----------

def conv_ag_news(ds):
    names = ["World", "Sports", "Business", "Science/Technology"]
    for r in ds:
        yield {"context": r["text"],
               "question": pick(["What is the topic of this news article?",
                                 "Which newspaper section does this belong in?",
                                 "Classify this article by subject."]),
               "options": subsample(names, names[r["label"]]), "answer": names[r["label"]]}


def conv_emotion(ds):
    names = names_of(ds, "label")
    for r in ds:
        yield {"context": r["text"],
               "question": pick(["What emotion does the writer express?",
                                 "Which feeling best describes this message?"]),
               "options": subsample(names, names[r["label"]]), "answer": names[r["label"]]}


def conv_yelp(ds):
    stars = ["1", "2", "3", "4", "5"]
    for r in ds:
        yield {"context": r["text"],
               "question": pick(["How many stars (1-5) did this reviewer give?",
                                 "Rate the customer's satisfaction from 1 (worst) to 5 (best)."]),
               "options": stars, "answer": stars[r["label"]]}


def conv_boolq(ds):
    for r in ds:
        q = r["question"].strip()
        yield {"context": r["passage"], "question": q[0].upper() + q[1:] + "?",
               "options": ["yes", "no"], "answer": "yes" if r["answer"] else "no"}


def conv_banking77(ds):
    names = sorted(set(ds["label_text"]))
    names = [n.replace("_", " ") for n in names]
    for r in ds:
        gold = r["label_text"].replace("_", " ")
        yield {"context": r["text"],
               "question": pick(["What does the bank customer want?",
                                 "Which support intent matches this message?"]),
               "options": subsample(names, gold), "answer": gold}


def conv_clinc(ds):
    names = [n.replace("_", " ") for n in names_of(ds, "intent")]
    for r in ds:
        gold = names[r["intent"]]
        yield {"context": r["text"],
               "question": pick(["What is the user asking the assistant to do?",
                                 "Which intent fits this request?"]),
               "options": subsample(names, gold), "answer": gold}


def conv_mnli(ds):
    opts = ["B follows from A", "B might or might not be true", "B contradicts A"]
    for r in ds:
        if r["label"] not in (0, 1, 2):
            continue
        yield {"context": f"Statement A: {r['premise']}\nStatement B: {r['hypothesis']}",
               "question": "Given statement A, what can we say about statement B?",
               "options": subsample(opts, opts[r["label"]]), "answer": opts[r["label"]]}


def conv_sst2(ds):
    for r in ds:
        pos = r["label"] == 1
        if R.random() < 0.5:
            yield {"context": r["sentence"], "question": "Is the sentiment of this text positive?",
                   "options": ["yes", "no"], "answer": "yes" if pos else "no"}
        else:
            o = ["positive", "negative"]
            yield {"context": r["sentence"], "question": "What is the sentiment?",
                   "options": subsample(o, o[0] if pos else o[1]), "answer": o[0] if pos else o[1]}


def conv_csqa(ds):
    for r in ds:
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] not in labels:
            continue
        gold = texts[labels.index(r["answerKey"])]
        yield {"context": r["question"], "question": "Which answer is most sensible?",
               "options": subsample(texts, gold), "answer": gold}


SOURCES = {
    # name: (hf id, config, train split, eval split, converter)
    "ag_news":   ("fancyzhx/ag_news", None, "train", "test", conv_ag_news),
    "emotion":   ("dair-ai/emotion", "split", "train", "test", conv_emotion),
    "yelp":      ("Yelp/yelp_review_full", None, "train", "test", conv_yelp),
    "boolq":     ("google/boolq", None, "train", "validation", conv_boolq),
    "banking77": ("mteb/banking77", None, "train", "test", conv_banking77),
    "clinc":     ("clinc/clinc_oos", "plus", "train", "test", conv_clinc),
    "mnli":      ("nyu-mll/glue", "mnli", "train", "validation_matched", conv_mnli),
    "sst2":      ("stanfordnlp/sst2", None, "train", "validation", conv_sst2),
    "csqa":      ("tau/commonsense_qa", None, "train", "validation", conv_csqa),
}


def take(ds, n):
    n = min(n, len(ds))
    return ds.shuffle(seed=0).select(range(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=4000, help="train examples per dataset")
    ap.add_argument("--eval-per-dataset", type=int, default=400)
    ap.add_argument("--only", nargs="*", default=None, help="subset of: " + " ".join(SOURCES))
    ap.add_argument("--holdout", nargs="*", default=["emotion"],
                    help="datasets kept out of training (zero-shot test)")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = {k: open(os.path.join(a.out, f"{k}.jsonl"), "w", encoding="utf-8")
             for k in ["train", "val", "test", "test_zeroshot"]}
    counts = {}
    for name, (hf, cfg, tr, ev, fn) in SOURCES.items():
        if a.only and name not in a.only:
            continue
        try:
            train = load_dataset(hf, cfg, split=tr)
            evals = load_dataset(hf, cfg, split=ev)
        except Exception as e:  # dataset moved / renamed: skip, keep going
            print(f"[skip] {name}: {e}")
            continue
        def dump(rows, f):
            n = 0
            for ex in rows:
                ex["source"] = name
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n += 1
            return n
        ev_rows = list(fn(take(evals, 2 * a.eval_per_dataset)))
        if a.holdout and name in a.holdout:
            counts[name] = {"test_zeroshot": dump(ev_rows[: a.eval_per_dataset], files["test_zeroshot"])}
        else:
            counts[name] = {
                "train": dump(fn(take(train, a.per_dataset)), files["train"]),
                "val": dump(ev_rows[: a.eval_per_dataset // 2], files["val"]),
                "test": dump(ev_rows[a.eval_per_dataset // 2: a.eval_per_dataset], files["test"]),
            }
        print(name, counts[name])
    for f in files.values():
        f.close()


if __name__ == "__main__":
    main()
