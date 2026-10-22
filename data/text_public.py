"""Public text datasets -> rows (several typed questions per state where the data allows).

None of these is scored by the Jev Decision Index (see data/common.py: DI_FAMILIES); the
banking77 and clinc converters were removed for that reason.

    python data/text_public.py --out data/mix/src/text_public.jsonl --per-dataset 3000

Each row carries a `split`: train/val come from the dataset's train split, test from its official
eval split. Datasets in --holdout are written with split "zeroshot" only (never trained on).
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.common import is_di_family, split_of  # noqa: E402

R = random.Random(0)


def choice(instr, opts):
    return {"type": "choice", "instructions": instr, "criteria": {o: "" for o in opts}}


def labels_of(ds, col):
    return ds.features[col].names


# ------------------------------------------------------------------ converters: ds rows -> rows

def conv_ag_news(ds):
    names = ["World", "Sports", "Business", "Science/Technology"]
    for r in ds:
        yield r["text"], {"topic": choice(R.choice(["What is the topic of this news article?",
                                                    "Which newspaper section does this belong in?"]), names)}, \
            {"topic": names[r["label"]]}


def conv_dbpedia(ds):
    names = labels_of(ds, "label")
    for r in ds:
        yield f"{r['title']}: {r['content']}", \
            {"kind": choice("What kind of entity does this text describe?", names)}, {"kind": names[r["label"]]}


def conv_emotion(ds):
    names = labels_of(ds, "label")
    for r in ds:
        yield r["text"], {"emotion": choice("Which emotion does the writer express?", names)}, \
            {"emotion": names[r["label"]]}


def conv_yelp(ds):
    stars = ["1", "2", "3", "4", "5"]
    for r in ds:
        s = r["label"] + 1
        yield r["text"], {
            "stars": {"type": "score", "instructions": "How many stars did this reviewer give?",
                      "criteria": {"1": "very negative", "2": "", "3": "mixed", "4": "", "5": "very positive"}},
            "recommend": {"type": "noul", "instructions": "Would this reviewer recommend the business?"},
        }, {"stars": stars[r["label"]], "recommend": s >= 4} if s != 3 else {"stars": "3"}


def conv_boolq(ds):
    for r in ds:
        q = r["question"].strip()
        yield r["passage"], {"answer": {"type": "noul", "instructions": q[0].upper() + q[1:] + "?"}}, \
            {"answer": bool(r["answer"])}


def conv_mnli(ds):
    opts = {"entailment": "B follows from A", "neutral": "B might or might not be true",
            "contradiction": "B contradicts A"}
    keys = list(opts)
    for r in ds:
        if r["label"] not in (0, 1, 2):
            continue
        yield f"Statement A: {r['premise']}\nStatement B: {r['hypothesis']}", {
            "relation": {"type": "choice", "instructions": "Given statement A, what can we say about statement B?",
                         "criteria": opts},
            "follows": {"type": "noul", "instructions": "Does statement B follow from statement A?"},
        }, {"relation": keys[r["label"]], "follows": r["label"] == 0}


def conv_sst2(ds):
    for r in ds:
        pos = r["label"] == 1
        yield r["sentence"], {
            "positive": {"type": "noul", "instructions": "Is the sentiment of this text positive?"},
            "sentiment": choice("What is the sentiment?", ["positive", "negative"]),
        }, {"positive": pos, "sentiment": "positive" if pos else "negative"}


def conv_csqa(ds):
    for r in ds:
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] not in labels:
            continue
        yield "", {"answer": {"type": "choice", "instructions": r["question"],
                              "criteria": dict(zip(labels, texts))}}, {"answer": r["answerKey"]}


def conv_obqa(ds):
    for r in ds:
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] not in labels:
            continue
        yield "", {"answer": {"type": "choice", "instructions": r["question_stem"],
                              "criteria": dict(zip(labels, texts))}}, {"answer": r["answerKey"]}


SOURCES = {
    # name: (hf id, config, train split, eval split, converter)
    "ag_news":    ("fancyzhx/ag_news", None, "train", "test", conv_ag_news),
    "dbpedia":    ("fancyzhx/dbpedia_14", None, "train", "test", conv_dbpedia),
    "emotion":    ("dair-ai/emotion", "split", "train", "test", conv_emotion),
    "yelp":       ("Yelp/yelp_review_full", None, "train", "test", conv_yelp),
    "boolq":      ("google/boolq", None, "train", "validation", conv_boolq),
    "mnli":       ("nyu-mll/glue", "mnli", "train", "validation_matched", conv_mnli),
    "sst2":       ("stanfordnlp/sst2", None, "train", "validation", conv_sst2),
    "csqa":       ("tau/commonsense_qa", None, "train", "validation", conv_csqa),
    "openbookqa": ("allenai/openbookqa", "main", "train", "test", conv_obqa),
}


def take(ds, n):
    return ds.shuffle(seed=0).select(range(min(n, len(ds))))


def main():
    from datasets import load_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mix/src/text_public.jsonl")
    ap.add_argument("--per-dataset", type=int, default=3000, help="train rows per dataset")
    ap.add_argument("--eval-per-dataset", type=int, default=300)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--only", nargs="*", default=None, help="subset of: " + " ".join(SOURCES))
    ap.add_argument("--holdout", nargs="*", default=["emotion", "openbookqa"],
                    help="datasets kept out of training entirely (zero-shot test)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n_all = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for name, (hf, cfg, tr, ev, fn) in SOURCES.items():
            if a.only and name not in a.only:
                continue
            assert not is_di_family(name), f"{name} is a Decision Index benchmark; refusing to train on it"
            try:
                train = load_dataset(hf, cfg, split=tr)
                evals = load_dataset(hf, cfg, split=ev)
            except Exception as e:
                print(f"[skip] {name}: {e}")
                continue
            counts = {}
            parts = [("zeroshot", take(evals, a.eval_per_dataset))] if name in (a.holdout or []) else \
                [("train", take(train, a.per_dataset)), ("test", take(evals, a.eval_per_dataset))]
            for split, ds in parts:
                for i, (state, qs, exp) in enumerate(fn(ds)):
                    sp = split_of(f"{name}/{i}", a.val_frac, 0.0) if split == "train" else split
                    row = {"id": f"{name}/{split}/{i}", "group": f"{name}/{split}/{i}", "source": name,
                           "modality": "text", "split": sp, "state": state, "images": [],
                           "questions": {k: v for k, v in qs.items()}, "expected": exp}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[sp] = counts.get(sp, 0) + 1
                    n_all += 1
            print(name, counts, flush=True)
    print(f"wrote {n_all} rows -> {a.out}")


if __name__ == "__main__":
    main()
