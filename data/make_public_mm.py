"""Real photos with exact labels: Food-101 as an image-classification-as-decision dataset.

    python data/make_public_mm.py --classes 16 --held-out 6 --per-class 20

Writes data/images/public/*.jpg and
    data/mm_food_train.jsonl      seen classes   (question "What dish is shown?", 4 options)
    data/mm_food_zeroshot.jsonl   held-out classes never used for training (zero-shot test)
Distractor options come from the same group (seen or held-out). The stream is class-sorted, so we read
(without decoding) until `--classes` classes have been collected, keeping `--per-class` images each.
The images in data/images/public can also be fed to `make_synthetic_mm.py propose`.
"""
import argparse
import io
import json
import os
import random

from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image


def pretty(name):
    return name.replace("_", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=16)
    ap.add_argument("--held-out", type=int, default=6, help="how many of the classes go to the zero-shot file")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--img-dir", default="data/images/public")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    R = random.Random(a.seed)
    os.makedirs(a.img_dir, exist_ok=True)

    ds = load_dataset("ethz/food101", split="train", streaming=True)
    names = ds.features["label"].names
    ds = ds.shuffle(seed=a.seed, buffer_size=64).cast_column("image", HFImage(decode=False))

    kept, done = {}, set()
    for ex in ds:
        lab = ex["label"]
        if lab in done:
            continue
        lst = kept.setdefault(lab, [])
        if len(lst) < a.per_class:
            lst.append(ex["image"]["bytes"])
        if len(lst) == a.per_class:
            done.add(lab)
            print(f"collected {len(done)}/{a.classes}: {names[lab]}", flush=True)
            if len(done) == a.classes:
                break

    labs = sorted(done)
    R.shuffle(labs)
    groups = {"zeroshot": labs[: a.held_out], "train": labs[a.held_out:]}
    for g, glabs in groups.items():
        rows = []
        for lab in glabs:
            for i, raw in enumerate(kept[lab][: a.per_class]):
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                im.thumbnail((a.max_side, a.max_side))
                path = f"{a.img_dir}/food_{names[lab]}_{i:03d}.jpg"
                im.save(path, quality=90)
                opts = [names[lab]] + R.sample([names[x] for x in glabs if x != lab], min(3, len(glabs) - 1))
                R.shuffle(opts)
                rows.append({"image": path, "context": "", "question": "What dish is shown in the image?",
                             "options": [pretty(o) for o in opts], "answer": pretty(names[lab]),
                             "task_id": path, "source": f"food101_{g}", "kind": "photo"})
        out = f"data/mm_food_{g}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{g}: {len(glabs)} classes, {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
