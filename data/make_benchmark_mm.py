"""ScienceQA (image questions of the official test split) -> decision JSONL, for zero-shot evaluation.

    python data/make_benchmark_mm.py --n 300 --out data/bench_scienceqa.jsonl
Only questions that have an image are kept. The `hint` field (extra text) becomes `context`.
Option text is kept as is; duplicates within one question get a letter suffix so the answer stays unique.
"""
import argparse
import json
import os

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/bench_scienceqa.jsonl")
    ap.add_argument("--img-dir", default="data/images/scienceqa")
    ap.add_argument("--max-side", type=int, default=768)
    a = ap.parse_args()
    os.makedirs(a.img_dir, exist_ok=True)
    ds = load_dataset("derek-thomas/ScienceQA", split="test", streaming=True)
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if ex["image"] is None:
                continue
            opts = [str(c).strip() for c in ex["choices"]]
            if len(set(o.lower() for o in opts)) != len(opts):
                opts = [f"{o} ({chr(65 + j)})" for j, o in enumerate(opts)]
            im = ex["image"].convert("RGB")
            im.thumbnail((a.max_side, a.max_side))
            path = f"{a.img_dir}/sqa_{i:05d}.png"
            im.save(path)
            f.write(json.dumps({"image": path, "context": ex["hint"].strip(), "question": ex["question"].strip(),
                                "options": opts, "answer": opts[ex["answer"]], "task_id": path,
                                "source": "scienceqa", "subject": ex["subject"]}, ensure_ascii=False) + "\n")
            n += 1
            if n >= a.n:
                break
    print(f"{n} questions -> {a.out}", flush=True)
    os._exit(0)  # the streaming dataset's background threads can keep the process alive after break


if __name__ == "__main__":
    main()
