"""Confirmation test set: datasets never used for any design decision or training.

    python data/make_confirm_tests.py data/mix/test_confirm.jsonl

SuperGLUE CB / COPA / WiC, tweet_eval emotion / sentiment / stance / emoji (20 options), Rotten Tomatoes,
ARC-Easy, and images: STL-10, GTSRB traffic signs (43 options), RESISC45 remote sensing (45 options),
FER2013 facial expressions, SVHN digits. Built after the main comparison was run, so the final claim
is tested on data that did not influence any choice.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset  # noqa: E402
from data.common import criteria_from, save_image  # noqa: E402

N, NV = 250, 200
IMG = "data/images/confirm"
rows = []


def add(src, i, state, qs, exp, images=(), modality="text"):
    rows.append({"id": f"{src}/confirm/{i}", "group": f"{src}/confirm/{i}", "source": src, "modality": modality,
                 "split": "test", "state": state, "images": list(images), "questions": qs, "expected": exp})


def choice(instr, names):
    return {"type": "choice", "instructions": instr, "criteria": criteria_from(names)}


def noul(instr):
    return {"type": "noul", "instructions": instr}


def run(name, fn):
    before = len(rows)
    try:
        fn()
        print(f"[ok] {name}: {len(rows) - before} rows", flush=True)
    except Exception as e:  # noqa: BLE001
        del rows[before:]
        print(f"[skip] {name}: {type(e).__name__}: {str(e)[:110]}", flush=True)


def sg(cfg, split="validation"):
    return load_dataset("aps/super_glue", cfg, split=split).shuffle(seed=0)


def cb():
    names = ["entailment", "contradiction", "neutral"]
    for i, r in enumerate(sg("cb")):
        add("cb", i, f"Statement A: {r['premise']}\nStatement B: {r['hypothesis']}",
            {"relation": choice("Given statement A, what can we say about statement B?", names)}, {"relation": names[r["label"]]})


def copa():
    for i, r in enumerate(sg("copa")):
        q = "What was the cause?" if r["question"] == "cause" else "What happened as a result?"
        q2 = choice(f"{r['premise']} {q}", [r["choice1"], r["choice2"]])
        keys = list(q2["criteria"])
        add("copa", i, "", {"answer": q2}, {"answer": keys[r["label"]]})


def wic():
    for i, r in enumerate(sg("wic").select(range(N))):
        add("wic", i, f"Sentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}",
            {"same": noul(f"Is the word '{r['word']}' used with the same meaning in both sentences?")}, {"same": bool(r["label"])})


def tweet(cfg, instr, name, binary_pos=None):
    def f():
        ds = load_dataset("cardiffnlp/tweet_eval", cfg, split="test").shuffle(seed=0)
        names = ds.features["label"].names
        for i, r in enumerate(ds.select(range(min(N, len(ds))))):
            add(name, i, r["text"], {"answer": choice(instr, names)}, {"answer": names[r["label"]]})
    return f


def rotten():
    ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test").shuffle(seed=0)
    for i, r in enumerate(ds.select(range(N))):
        add("rotten_tomatoes", i, r["text"], {"positive": noul("Is this movie review positive?")}, {"positive": r["label"] == 1})


def arc_easy():
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").shuffle(seed=0)
    n = 0
    for r in ds:
        labs, txt = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] not in labs:
            continue
        add("arc_easy", n, "", {"answer": {"type": "choice", "instructions": r["question"], "criteria": dict(zip(labs, txt))}},
            {"answer": r["answerKey"]})
        n += 1
        if n == N:
            break


def vision(name, hf, split, img_col, lab_col, instr, cfg=None, ext="png"):
    def f():
        ds = (load_dataset(hf, cfg, split=split) if cfg else load_dataset(hf, split=split)).shuffle(seed=0)
        names = [str(n).replace("_", " ") for n in ds.features[lab_col].names]
        for i, r in enumerate(ds.select(range(min(NV, len(ds))))):
            p = save_image(r[img_col], f"{IMG}/{name}/{i:04d}.{ext}", 512)
            add(name, i, "", {"answer": choice(instr, names)}, {"answer": names[r[lab_col]]}, [p], "vision")
    return f


run("cb", cb)
run("copa", copa)
run("wic", wic)
run("tweet_emotion", tweet("emotion", "Which emotion does this tweet express?", "tweet_emotion"))
run("tweet_sentiment", tweet("sentiment", "What is the sentiment of this tweet?", "tweet_sentiment"))
run("tweet_stance", tweet("stance_feminist", "What is the author's stance toward feminism?", "tweet_stance"))
run("tweet_emoji", tweet("emoji", "Which emoji best fits this tweet?", "tweet_emoji"))
run("rotten_tomatoes", rotten)
run("arc_easy", arc_easy)
run("stl10", vision("stl10", "tanganke/stl10", "test", "image", "label", "What object is shown in the image?"))
run("gtsrb", vision("gtsrb", "tanganke/gtsrb", "test", "image", "label", "Which traffic sign is shown?"))
run("resisc45", vision("resisc45", "tanganke/resisc45", "test", "image", "label", "What kind of scene is shown in this aerial image?"))
run("fer2013", vision("fer2013", "tanganke/fer2013", "test", "image", "label", "Which facial expression is shown?"))
run("svhn", vision("svhn", "ufldl-stanford/svhn", "test", "image", "label", "Which digit is shown in the image?", cfg="cropped_digits"))
with open(sys.argv[1], "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "rows ->", sys.argv[1])
