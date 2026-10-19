"""Public image datasets -> v2 rows.

    python data/vision_public.py --out data/v2/src/vision_public.jsonl --per-dataset 3000
    python data/vision_public.py --bench-only --out data/v2/bench_scienceqa.jsonl     # ScienceQA test, zero-shot

Sources (none is part of the Decision Index vision board, see data/common.py):
  scienceqa  derek-thomas/ScienceQA   multiple choice; rows without an image become text rows
  aokvqa     HuggingFaceM4/A-OKVQA    multiple choice over COCO photos
  chartqa    HuggingFaceM4/ChartQA    chart questions; free-form answers become 4-way choices with
                                      distractors (numeric answers: perturbed values; text answers:
                                      other answers from the same split), yes/no answers become noul
Images are saved under --img-dir, longest side capped at --max-side.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.common import criteria_from, is_di_family, save_image, split_of  # noqa: E402

R = random.Random(0)
NUM = re.compile(r"^-?\d+(?:\.\d+)?%?$")


def numeric_distractors(ans: str, k: int = 3):
    pct = ans.endswith("%")
    x = float(ans.rstrip("%"))
    dec = len(ans.rstrip("%").split(".")[1]) if "." in ans else 0
    cands = set()
    for f in (0.5, 0.75, 1.25, 1.5, 2.0, 0.8, 1.1, 0.9):
        v = round(x * f, dec) if x != 0 else round(R.uniform(1, 10), dec)
        cands.add(v)
    for d in (1, 2, 5, 10):
        cands.update({round(x + d, dec), round(x - d, dec)})
    cands.discard(round(x, dec))
    def fmt(v):
        return f"{v:.{dec}f}" if dec else str(int(round(v)))
    out = [fmt(v) + ("%" if pct else "") for v in cands]
    out = [o for o in dict.fromkeys(out) if o != ans]
    R.shuffle(out)
    return out[:k]


def conv_scienceqa(ds, img_dir, max_side, split_name):
    for i, ex in enumerate(ds):
        opts = [str(c).strip() for c in ex["choices"]]
        crit = criteria_from(opts)
        keys = list(crit)
        images = []
        if ex.get("image") is not None:
            images = [save_image(ex["image"], f"{img_dir}/scienceqa/{split_name}_{i:05d}.png", max_side)]
        yield {"id": f"scienceqa/{split_name}/{i}", "source": "scienceqa",
               "modality": "vision" if images else "text", "state": (ex.get("hint") or "").strip(),
               "images": images,
               "questions": {"answer": {"type": "choice", "instructions": ex["question"].strip(), "criteria": crit}},
               "expected": {"answer": keys[ex["answer"]]}, "subject": ex.get("subject")}


def conv_aokvqa(ds, img_dir, max_side, split_name):
    for i, ex in enumerate(ds):
        if ex.get("correct_choice_idx") is None or ex.get("image") is None:
            continue
        crit = criteria_from(ex["choices"])
        keys = list(crit)
        path = save_image(ex["image"], f"{img_dir}/aokvqa/{split_name}_{i:05d}.jpg", max_side)
        yield {"id": f"aokvqa/{split_name}/{i}", "source": "aokvqa", "modality": "vision", "state": "",
               "images": [path],
               "questions": {"answer": {"type": "choice", "instructions": ex["question"].strip(), "criteria": crit}},
               "expected": {"answer": keys[ex["correct_choice_idx"]]}}


def conv_chartqa(ds, img_dir, max_side, split_name):
    rows = list(ds)
    text_answers = [str(r["label"][0]).strip() for r in rows
                    if r.get("label") and not NUM.match(str(r["label"][0]).strip())
                    and str(r["label"][0]).strip().lower() not in ("yes", "no")]
    for i, ex in enumerate(rows):
        if not ex.get("label") or ex.get("image") is None:
            continue
        ans = str(ex["label"][0]).strip()
        q = ex["query"].strip()
        if ans.lower() in ("yes", "no"):
            question, expected = {"type": "noul", "instructions": q}, ans.lower() == "yes"
        else:
            if NUM.match(ans):
                distract = numeric_distractors(ans)
            else:
                pool = [t for t in text_answers if t.lower() != ans.lower()]
                distract = R.sample(pool, min(3, len(pool)))
            if len(distract) < 3:
                continue
            opts = [ans] + distract
            R.shuffle(opts)
            crit = criteria_from(opts)
            question, expected = {"type": "choice", "instructions": q, "criteria": crit}, list(crit)[opts.index(ans)]
        path = save_image(ex["image"], f"{img_dir}/chartqa/{split_name}_{i:05d}.png", max_side)
        yield {"id": f"chartqa/{split_name}/{i}", "source": "chartqa", "modality": "vision", "state": "",
               "images": [path], "questions": {"answer": question}, "expected": {"answer": expected}}


SOURCES = {
    # name: (hf id, config, train split, eval split, converter)
    "scienceqa": ("derek-thomas/ScienceQA", None, "train", "validation", conv_scienceqa),
    "aokvqa":    ("HuggingFaceM4/A-OKVQA", None, "train", "validation", conv_aokvqa),
    "chartqa":   ("HuggingFaceM4/ChartQA", None, "train", "val", conv_chartqa),
}


def stream(hf, cfg, split, n):
    from datasets import load_dataset
    ds = load_dataset(hf, cfg, split=split, streaming=True)
    return ds.shuffle(seed=0, buffer_size=2000).take(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v2/src/vision_public.jsonl")
    ap.add_argument("--img-dir", default="data/images")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--per-dataset", type=int, default=3000)
    ap.add_argument("--eval-per-dataset", type=int, default=300)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--only", nargs="*", default=None, help="subset of: " + " ".join(SOURCES))
    ap.add_argument("--holdout", nargs="*", default=[], help="sources written as zero-shot test only")
    ap.add_argument("--bench-only", action="store_true",
                    help="write only the ScienceQA *test* image questions (the v1 public benchmark)")
    ap.add_argument("--bench-n", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    if a.bench_only:
        from datasets import load_dataset
        ds = load_dataset("derek-thomas/ScienceQA", split="test", streaming=True)
        n = 0
        with open(a.out, "w", encoding="utf-8") as f:
            for row in conv_scienceqa((x for x in ds if x["image"] is not None), a.img_dir, a.max_side, "test"):
                row["split"] = "bench"
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                if n >= a.bench_n:
                    break
        print(f"wrote {n} ScienceQA test image questions -> {a.out}")
        return

    total = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for name, (hf, cfg, tr, ev, fn) in SOURCES.items():
            if a.only and name not in a.only:
                continue
            assert not is_di_family(name), f"{name} is a Decision Index benchmark; refusing to train on it"
            parts = [("zeroshot", ev, a.eval_per_dataset)] if name in a.holdout else \
                [("train", tr, a.per_dataset), ("test", ev, a.eval_per_dataset)]
            counts = {}
            try:
                for split, hf_split, n in parts:
                    for row in fn(stream(hf, cfg, hf_split, n), a.img_dir, a.max_side, split):
                        row["group"] = row["id"]
                        row["split"] = split_of(row["id"], a.val_frac, 0.0) if split == "train" else split
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        counts[row["split"]] = counts.get(row["split"], 0) + 1
                        total += 1
            except Exception as e:
                print(f"[skip/partial] {name}: {type(e).__name__}: {e}")
            print(name, counts, flush=True)
    print(f"wrote {total} rows -> {a.out}")


if __name__ == "__main__":
    main()
