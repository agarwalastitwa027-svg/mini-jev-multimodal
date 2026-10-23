"""Held-out many-option test sets (every question lists 27-100 options).

    python data/make_manyopt_tests.py data/mix/test_manyopt.jsonl

cifar100_fine (100 classes, images), banking77 (77 intents), go_emotions (28 labels, single-label rows).
None of these is in any training mix. Used to measure the many-option weakness and option-count augmentation.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset  # noqa: E402
from data.common import criteria_from, save_image  # noqa: E402

N, NV = 250, 200
IMG = "data/images/manyopt"
rows = []


def add(src, i, state, instr, names, gold, images=(), modality="text"):
    rows.append({"id": f"{src}/manyopt/{i}", "group": f"{src}/manyopt/{i}", "source": src, "modality": modality,
                 "split": "test", "state": state, "images": list(images),
                 "questions": {"answer": {"type": "choice", "instructions": instr, "criteria": criteria_from(names)}},
                 "expected": {"answer": gold}})


def run(name, fn):
    before = len(rows)
    try:
        fn()
        print(f"[ok] {name}: {len(rows) - before} rows", flush=True)
    except Exception as e:  # noqa: BLE001
        del rows[before:]
        print(f"[skip] {name}: {type(e).__name__}: {str(e)[:120]}", flush=True)


def cifar100():
    ds = load_dataset("uoft-cs/cifar100", split="test").shuffle(seed=0)
    names = [n.replace("_", " ") for n in ds.features["fine_label"].names]
    for i, r in enumerate(ds.select(range(NV))):
        p = save_image(r["img"], f"{IMG}/cifar100/{i:04d}.png", 512)
        add("cifar100_fine", i, "", "What object is shown in the image?", names, names[r["fine_label"]], [p], "vision")


def banking():
    last = None
    for hf in ("PolyAI/banking77", "mteb/banking77"):
        try:
            ds = load_dataset(hf, split="test").shuffle(seed=0)
            break
        except Exception as e:  # noqa: BLE001
            last = e
    else:
        raise last
    col = "label_text" if "label_text" in ds.column_names else None
    names = [n.replace("_", " ") for n in (sorted(set(ds["label_text"])) if col else ds.features["label"].names)]
    for i, r in enumerate(ds.select(range(N))):
        gold = (r["label_text"] if col else ds.features["label"].names[r["label"]]).replace("_", " ")
        add("banking77", i, r["text"], "Which banking intent does this message express?", names, gold)


def go_emotions():
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test").filter(lambda r: len(r["labels"]) == 1)
    ds = ds.shuffle(seed=0)
    names = ds.features["labels"].feature.names
    for i, r in enumerate(ds.select(range(N))):
        add("go_emotions", i, r["text"], "Which emotion does the writer express?", names, names[r["labels"][0]])


run("cifar100_fine", cifar100)
run("banking77", banking)
run("go_emotions", go_emotions)
with open(sys.argv[1], "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "rows ->", sys.argv[1])
