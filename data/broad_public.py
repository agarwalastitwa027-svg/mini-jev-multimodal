"""More public training sources for the broader mix (task-breadth experiment).

    python data/broad_public.py --out data/broad/src/broad_public.jsonl --per-dataset 1000

New text sources:   CoLA (grammar), STS-B (similarity score), RACE (reading comprehension), SMS spam,
                    Yahoo Answers topics, Amazon polarity, QuaRTz, CosmosQA, SST-5 (score)
New image sources:  Oxford-IIIT Pets, Oxford Flowers, EuroSAT, DTD textures (4-way choices)

Held out on purpose, never trained on even in the broad mix (so they test generalisation to a new
task *family*, not only a new dataset): paraphrase (MRPC), hate / offensive / irony tweets, financial news
sentiment, and the digit, clothing and leaf-disease image sets. The builder refuses any source that is
a Decision Index benchmark or one of the held-out names.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.common import criteria_from, is_di_family, save_image, split_of  # noqa: E402

R = random.Random(0)
HELD_OUT = {"mrpc", "qqp", "paws", "tweet_hate", "tweet_offensive", "tweet_irony", "fin_news", "mnist",
            "fashion_mnist", "beans", "svhn", "emotion", "openbookqa", "sciq", "qasc", "snli", "rte", "qnli",
            "imdb", "newsgroups20", "cifar10", "cifar100_super", "arc_challenge", "food101_zeroshot"}


def noul(instr):
    return {"type": "noul", "instructions": instr}


def choice(instr, opts):
    return {"type": "choice", "instructions": instr, "criteria": criteria_from(opts)}


# ------------------------------------------------------------------ text converters: ds -> (state, questions, expected)

def conv_cola(ds):
    for r in ds:
        yield r["sentence"], {"acceptable": noul("Is this sentence grammatically acceptable English?")}, \
            {"acceptable": r["label"] == 1}


def conv_stsb(ds):
    crit = {"0": "unrelated", "1": "", "2": "", "3": "", "4": "", "5": "same meaning"}
    for r in ds:
        yield f"Sentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}", \
            {"similarity": {"type": "score", "instructions": "How semantically similar are the two sentences?",
                            "criteria": crit}}, {"similarity": str(max(0, min(5, int(round(r["label"])))))}


def conv_race(ds):
    for r in ds:
        opts = r["options"]
        if len(opts) != 4 or r["answer"] not in "ABCD":
            continue
        yield r["article"], {"answer": {"type": "choice", "instructions": r["question"],
                                        "criteria": dict(zip("ABCD", opts))}}, {"answer": r["answer"]}


def conv_sms(ds):
    for r in ds:
        yield r["sms"], {"spam": noul("Is this text message spam?")}, {"spam": r["label"] == 1}


def conv_yahoo(ds):
    names = ds.features["topic"].names
    for r in ds:
        txt = f"{r['question_title']}\n{r['question_content']}".strip()
        yield txt, {"topic": choice("Which topic does this question belong to?", names)}, {"topic": names[r["topic"]]}


def conv_amazon(ds):
    for r in ds:
        yield f"{r['title']}. {r['content']}", {"positive": noul("Is this product review positive?")}, \
            {"positive": r["label"] == 1}


def conv_quartz(ds):
    for r in ds:
        labs, txt = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] not in labs:
            continue
        yield r["para"], {"answer": {"type": "choice", "instructions": r["question"],
                                     "criteria": dict(zip(labs, txt))}}, {"answer": r["answerKey"]}


def conv_cosmos(ds):
    for r in ds:
        if r["label"] not in (0, 1, 2, 3):
            continue
        opts = [r["answer0"], r["answer1"], r["answer2"], r["answer3"]]
        yield r["context"], {"answer": {"type": "choice", "instructions": r["question"],
                                        "criteria": dict(zip("ABCD", opts))}}, {"answer": "ABCD"[r["label"]]}


def conv_sst5(ds):
    crit = {"1": "very negative", "2": "negative", "3": "neutral", "4": "positive", "5": "very positive"}
    for r in ds:
        yield r["text"], {"sentiment": {"type": "score", "instructions": "How positive is the sentiment of this text?",
                                        "criteria": crit}}, {"sentiment": str(r["label"] + 1)}


TEXT = {
    "cola":     ("nyu-mll/glue", "cola", "train", conv_cola),
    "stsb":     ("nyu-mll/glue", "stsb", "train", conv_stsb),
    "race":     ("ehovy/race", "all", "train", conv_race),
    "sms_spam": ("ucirvine/sms_spam", None, "train", conv_sms),
    "yahoo_topics": ("community-datasets/yahoo_answers_topics", None, "train", conv_yahoo),
    "amazon_polarity": ("fancyzhx/amazon_polarity", None, "train", conv_amazon),
    "quartz":   ("allenai/quartz", None, "train", conv_quartz),
    "cosmos_qa": ("allenai/cosmos_qa", None, "train", conv_cosmos),
    "sst5":     ("SetFit/sst5", None, "train", conv_sst5),
}

# name -> (hf id, config, split, image column, label column, question)
VISION = {
    "oxford_pets": ("timm/oxford-iiit-pet", None, "train", "image", "label", "Which breed is the animal in the image?"),
    "oxford_flowers": ("nelorth/oxford-flowers", None, "train", "image", "label", "Which flower is shown in the image?"),
    "eurosat": ("tanganke/eurosat", None, "train", "image", "label", "What kind of land does this satellite image show?"),
    "dtd_textures": ("tanganke/dtd", None, "train", "image", "label", "Which texture is shown in the image?"),
}


def pretty(n):
    return str(n).replace("_", " ").strip()


def main():
    from datasets import load_dataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/broad/src/broad_public.jsonl")
    ap.add_argument("--img-dir", default="data/images/broad")
    ap.add_argument("--per-dataset", type=int, default=1000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    total = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for name, (hf, cfg, split, fn) in TEXT.items():
            if a.only and name not in a.only:
                continue
            assert name not in HELD_OUT and not is_di_family(name), name
            counts = {}
            try:
                ds = (load_dataset(hf, cfg, split=split) if cfg else load_dataset(hf, split=split)).shuffle(seed=0)
                ds = ds.select(range(min(a.per_dataset, len(ds))))
                for i, (state, qs, exp) in enumerate(fn(ds)):
                    sp = split_of(f"{name}/{i}", a.val_frac, 0.0)
                    f.write(json.dumps({"id": f"{name}/train/{i}", "group": f"{name}/train/{i}", "source": name,
                                        "modality": "text", "split": sp, "state": state, "images": [],
                                        "questions": qs, "expected": exp}, ensure_ascii=False) + "\n")
                    counts[sp] = counts.get(sp, 0) + 1
                    total += 1
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {name}: {type(e).__name__}: {str(e)[:100]}")
            print(name, counts, flush=True)
        for name, (hf, cfg, split, img, lab, instr) in VISION.items():
            if a.only and name not in a.only:
                continue
            assert name not in HELD_OUT and not is_di_family(name), name
            counts = {}
            try:
                ds = (load_dataset(hf, cfg, split=split) if cfg else load_dataset(hf, split=split)).shuffle(seed=0)
                ds = ds.select(range(min(a.per_dataset, len(ds))))
                names = [pretty(n) for n in ds.features[lab].names]
                for i, r in enumerate(ds):
                    gold = names[r[lab]]
                    others = R.sample([n for n in names if n != gold], min(3, len(names) - 1))
                    opts = [gold] + others
                    R.shuffle(opts)
                    path = save_image(r[img], f"{a.img_dir}/{name}/{i:04d}.jpg", 512)
                    sp = split_of(f"{name}/{i}", a.val_frac, 0.0)
                    f.write(json.dumps({"id": f"{name}/train/{i}", "group": f"{name}/train/{i}", "source": name,
                                        "modality": "vision", "split": sp, "state": "", "images": [path],
                                        "questions": {"answer": choice(instr, opts)},
                                        "expected": {"answer": gold}}, ensure_ascii=False) + "\n")
                    counts[sp] = counts.get(sp, 0) + 1
                    total += 1
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {name}: {type(e).__name__}: {str(e)[:100]}")
            print(name, counts, flush=True)
    print(f"wrote {total} rows -> {a.out}")


if __name__ == "__main__":
    main()
