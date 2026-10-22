"""More held-out tests: open datasets that appear nowhere in mini-jev's training data."""
import json
import os
import sys
import traceback

sys.path.insert(0, "/home/astitwa/projects/mini-jev")
from datasets import load_dataset  # noqa: E402
from data.common import criteria_from, save_image  # noqa: E402

N = int(os.environ.get("N", 250))
NV = int(os.environ.get("NV", 200))
IMG = "/home/astitwa/projects/mini-jev/data/images/more"
rows = []


def add(src, i, state, qs, exp, images=(), modality="text"):
    rows.append({"id": f"{src}/more/{i}", "group": f"{src}/more/{i}", "source": src, "modality": modality,
                 "split": "test", "state": state, "images": list(images), "questions": qs, "expected": exp})


def choice(instr, opts):
    return {"type": "choice", "instructions": instr, "criteria": criteria_from(opts)}


def noul(instr):
    return {"type": "noul", "instructions": instr}


def load(*a, **k):
    return load_dataset(*a, **k).shuffle(seed=0)


def run(name, fn):
    before = len(rows)
    try:
        fn()
        print(f"[ok] {name}: {len(rows) - before} rows", flush=True)
    except Exception as e:  # noqa: BLE001
        del rows[before:]
        print(f"[skip] {name}: {type(e).__name__}: {str(e)[:120]}", flush=True)


def sciq():
    ds = load("allenai/sciq", split="test")
    for i, r in enumerate(ds.select(range(min(N, len(ds))))):
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        q = choice(r["question"], opts)
        add("sciq", i, r["support"].strip(), {"answer": q}, {"answer": list(q["criteria"])[0]})


def qasc():
    ds = load("allenai/qasc", split="validation")
    for i, r in enumerate(ds.select(range(min(N, len(ds))))):
        labs, txt = r["choices"]["label"], r["choices"]["text"]
        add("qasc", i, "", {"answer": {"type": "choice", "instructions": r["question"], "criteria": dict(zip(labs, txt))}},
            {"answer": r["answerKey"]})


def glue_pair(cfg, a, b, pos_label, instr, name):
    def f():
        ds = load("nyu-mll/glue", cfg, split="validation")
        for i, r in enumerate(ds.select(range(min(N, len(ds))))):
            add(name, i, f"Text A: {r[a]}\nText B: {r[b]}", {"answer": noul(instr)}, {"answer": r["label"] == pos_label})
    return f


def snli():
    ds = load("stanfordnlp/snli", split="test").filter(lambda r: r["label"] in (0, 1, 2))
    names = ["entailment", "neutral", "contradiction"]
    crit = {"entailment": "B follows from A", "neutral": "B might or might not be true", "contradiction": "B contradicts A"}
    for i, r in enumerate(ds.select(range(min(N, len(ds))))):
        add("snli", i, f"Statement A: {r['premise']}\nStatement B: {r['hypothesis']}",
            {"relation": {"type": "choice", "instructions": "Given statement A, what can we say about statement B?",
                          "criteria": crit}}, {"relation": names[r["label"]]})


def imdb():
    ds = load("stanfordnlp/imdb", split="test").filter(lambda r: len(r["text"]) < 2500)
    for i, r in enumerate(ds.select(range(min(N, len(ds))))):
        add("imdb", i, r["text"], {"positive": noul("Is the sentiment of this review positive?")}, {"positive": r["label"] == 1})


def tweet(cfg, instr, name):
    def f():
        ds = load("cardiffnlp/tweet_eval", cfg, split="test")
        for i, r in enumerate(ds.select(range(min(N, len(ds))))):
            add(name, i, r["text"], {"answer": noul(instr)}, {"answer": r["label"] == 1})
    return f


def fin_news():
    ds = load("zeroshot/twitter-financial-news-sentiment", split="validation")
    names = ["bearish", "bullish", "neutral"]
    for i, r in enumerate(ds.select(range(min(N, len(ds))))):
        add("fin_news", i, r["text"], {"sentiment": choice("What is the market sentiment of this news?", names)},
            {"sentiment": names[r["label"]]})


def newsgroups():
    ds = load("SetFit/20_newsgroups", split="test").filter(lambda r: 40 < len(r["text"]) < 2500)
    names = sorted(set(ds["label_text"]))
    for i, r in enumerate(ds.select(range(min(N, len(ds))))):
        add("newsgroups20", i, r["text"], {"group": choice("Which newsgroup was this posted to?", names)},
            {"group": r["label_text"]})


def trec():
    ds = load("CogComp/trec", split="test")
    names = ds.features["coarse_label"].names
    for i, r in enumerate(ds.select(range(min(N, len(ds))))):
        add("trec", i, "", {"type": choice(f"What kind of answer does this question expect? \"{r['text']}\"", names)},
            {"type": names[r["coarse_label"]]})


def vision(name, hf, split, img_col, lab_col, instr, cfg=None, names=None, ext="png"):
    def f():
        ds = load(hf, cfg, split=split) if cfg else load(hf, split=split)
        labs = names or ds.features[lab_col].names
        for i, r in enumerate(ds.select(range(min(NV, len(ds))))):
            p = save_image(r[img_col], f"{IMG}/{name}/{i:04d}.{ext}", 512)
            add(name, i, "", {"answer": choice(instr, labs)}, {"answer": labs[r[lab_col]]}, [p], "vision")
    return f


run("sciq", sciq)
run("qasc", qasc)
run("rte", glue_pair("rte", "sentence1", "sentence2", 0, "Does text B follow from text A?", "rte"))
run("qnli", glue_pair("qnli", "question", "sentence", 0, "Does text B answer the question in text A?", "qnli"))
run("mrpc", glue_pair("mrpc", "sentence1", "sentence2", 1, "Do text A and text B mean the same thing?", "mrpc"))
run("snli", snli)
run("imdb", imdb)
run("tweet_hate", tweet("hate", "Is this tweet hateful toward immigrants or women?", "tweet_hate"))
run("tweet_offensive", tweet("offensive", "Is this tweet offensive?", "tweet_offensive"))
run("fin_news", fin_news)
run("newsgroups20", newsgroups)
run("trec", trec)
run("cifar10", vision("cifar10", "uoft-cs/cifar10", "test", "img", "label", "What object is shown in the image?"))
run("fashion_mnist", vision("fashion_mnist", "zalando-datasets/fashion_mnist", "test", "image", "label",
                            "What kind of clothing item is shown?"))
run("mnist", vision("mnist", "ylecun/mnist", "test", "image", "label", "Which digit is shown in the image?"))
run("cifar100_super", vision("cifar100_super", "uoft-cs/cifar100", "test", "img", "coarse_label",
                             "Which category does the object in the image belong to?"))
run("beans", vision("beans", "AI-Lab-Makerere/beans", "test", "image", "labels",
                    "What is the condition of this bean leaf?", ext="jpg"))

with open(sys.argv[1], "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "rows ->", sys.argv[1])
