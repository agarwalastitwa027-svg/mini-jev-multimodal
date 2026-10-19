"""Convert v1 decision rows ({context, question, options, answer, image?}) into v2 rows.

Rows that share the same image and context become ONE v2 row with several questions, which is
what the shared-state cache is for (the procedural charts have ~6 questions per image).

    python data/convert_v1.py data/synthetic/mm_enriched.jsonl data/v2/src/procedural.jsonl --modality vision
    python data/convert_v1.py data/mm_food_train.jsonl data/v2/src/food101.jsonl --modality vision
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.common import expected_for, question_from_options  # noqa: E402


def convert(rows, modality=None, max_questions=16):
    groups = OrderedDict()
    for r in rows:
        key = (r.get("image") or "", (r.get("context") or "").strip())
        groups.setdefault(key, []).append(r)
    out = []
    for (image, context), rs in groups.items():
        for start in range(0, len(rs), max_questions):  # very large groups are split into several rows
            part = rs[start:start + max_questions]
            qs, exp = {}, {}
            for i, r in enumerate(part):
                q = question_from_options(r["question"].strip(), r["options"])
                k = f"q{i}"
                qs[k] = q
                if r.get("answer") is not None:
                    exp[k] = expected_for(q, r["answer"], r["options"])
            first = part[0]
            out.append({
                "id": f"{first.get('task_id', image or context[:40])}#{start}",
                "group": str(first.get("task_id", image or context[:80])),
                "source": first.get("source", "v1"),
                "modality": modality or ("vision" if image else "text"),
                "state": context,
                "images": [image] if image else [],
                "questions": qs,
                "expected": exp,
                **({"split": first["split"]} if "split" in first else {}),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--modality", choices=["text", "vision"], default=None)
    ap.add_argument("--split", default=None, help="force a split label (train/val/test/bench)")
    ap.add_argument("--max-questions", type=int, default=16)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    if a.split:
        for r in rows:
            r["split"] = a.split
    out = convert(rows, a.modality, a.max_questions)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    nq = sum(len(r["questions"]) for r in out)
    print(f"{a.inp}: {len(rows)} v1 rows -> {len(out)} v2 rows, {nq} questions -> {a.out}")


if __name__ == "__main__":
    main()
