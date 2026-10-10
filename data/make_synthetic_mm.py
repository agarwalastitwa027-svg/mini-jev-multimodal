"""Multimodal (image + text) synthetic decision data with a local Ollama vision model.

Mode 1  enrich   rows with an exact label (data/make_images.py output):
    the teacher LOOKS at the image and (a) rewords the question, (b) writes a short realistic text that
    came "with" the image (chat message, ticket note...) without revealing the answer. The new
    wording is kept only if the teacher, answering blind, still gets the known answer;
    otherwise the original question is used. The label always stays the programmatic one.

Mode 2  propose  a folder of real images:
    the teacher invents decision questions (choice / yes-no / score) about each image with an answer;
    a second blind pass must agree (same keep-if-agree rule as data/make_synthetic.py).

    python data/make_synthetic_mm.py enrich  --in data/mm_procedural.jsonl --out data/mm_enriched.jsonl
    python data/make_synthetic_mm.py propose --images data/images/public --out data/mm_proposed.jsonl
"""
import argparse
import base64
import io
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_synthetic import Teacher  # noqa: E402

R = random.Random(0)
MAX_SIDE = 768  # downscale what we send to the teacher; keeps Ollama fast


def b64(path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def verify(t, img, question, options, context=""):
    schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": options}},
              "required": ["answer"]}
    ctx = f"Text that came with the image:\n{context}\n\n" if context else ""
    p = (f"Look at the image carefully.\n{ctx}Question: {question}\n"
         f"Options: {json.dumps(options)}\nPick the single best option.")
    return (t.ask_json(p, schema, temperature=0.0, images=[img]) or {}).get("answer")


def same_question(t, original, reworded):
    """Text-only judge: does the reworded question ask for exactly the same thing?"""
    schema = {"type": "object", "properties": {"same": {"type": "boolean"}}, "required": ["same"]}
    p = (f"Question A: {original}\nQuestion B: {reworded}\n"
         "Do A and B ask for exactly the same information, so that the correct answer to A is always "
         "the correct answer to B? Be strict: a different attribute (e.g. 'largest' vs 'a colour'), "
         "extra conditions, or a changed comparison direction means false.")
    return bool((t.ask_json(p, schema, temperature=0.0) or {}).get("same"))


def verify_text_only(t, question, options, context):
    """Answer from the accompanying text alone (no image) -> detects answer leaks in `context`."""
    schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": options}},
              "required": ["answer"]}
    p = (f"Text: {context}\nQuestion: {question}\nOptions: {json.dumps(options)}\n"
         "Pick the single best option using only the text above.")
    return (t.ask_json(p, schema, temperature=0.0) or {}).get("answer")


REWRITE_SCHEMA ={"type": "object", "properties": {"question": {"type": "string"},
                                                   "context": {"type": "string"}},
                  "required": ["question", "context"]}


def enrich_row(t, row, imgs):
    img = imgs.setdefault(row["image"], b64(row["image"]))
    p = (f"An automated system must answer a question about this image.\n"
         f"Original question: \"{row['question']}\"\nAllowed answers: {json.dumps(row['options'])}\n"
         "1. Reword the question naturally (keep the exact same meaning and keep it answerable with the "
         "allowed answers; do not add facts).\n"
         "2. Write `context`: 1-2 sentences of realistic text that might accompany this image "
         "(a chat message, ticket note, report caption). It must NOT state or hint the answer, "
         "and must not describe specific values in the image.")
    out = t.ask_json(p, REWRITE_SCHEMA, temperature=0.8, images=[img]) or {}
    q, ctx = (out.get("question") or "").strip(), (out.get("context") or "").strip()
    row = {**row, "paraphrased": False}
    if q and same_question(t, row["question"], q) and verify(t, img, q, row["options"], ctx) == row["answer"]:
        # text-only leak check: if the answer is guessable from the text alone, drop the text
        if ctx and verify_text_only(t, q, row["options"], ctx) == row["answer"]:
            ctx = ""
        row.update(question=q, context=ctx, paraphrased=True)
    row["teacher_agrees_original"] = None  # filled lazily only when paraphrase failed
    if not row["paraphrased"]:
        row["teacher_agrees_original"] = verify(t, img, row["question"], row["options"]) == row["answer"]
    return row


TASKS_SCHEMA = {"type": "object", "properties": {"tasks": {"type": "array", "items": {
    "type": "object", "properties": {
        "question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}},
        "answer": {"type": "string"}},
    "required": ["question", "options", "answer"]}}}, "required": ["tasks"]}


def propose_image(t, path, per_image, imgs):
    img = imgs.setdefault(path, b64(path))
    p = (f"Look at this image. Invent {per_image} different decision questions that software could be "
         "asked about it, each with the correct answer. Mix kinds:\n"
         " - choice: 3 to 6 short, mutually exclusive options (1-4 words each)\n"
         " - yes/no: options exactly [\"yes\", \"no\"]\n"
         " - score: options exactly [\"1\",\"2\",\"3\",\"4\",\"5\"] and the question explains the scale\n"
         "Only ask what is clearly visible; `answer` must be exactly one of `options`.")
    out = t.ask_json(p, TASKS_SCHEMA, temperature=1.0, images=[img]) or {}
    rows = []
    for x in out.get("tasks", []):
        opts = [str(o).strip() for o in x.get("options", []) if str(o).strip()]
        ans = str(x.get("answer", "")).strip()
        if not (2 <= len(opts) <= 8 and len(set(o.lower() for o in opts)) == len(opts) and x.get("question")):
            continue
        match = [o for o in opts if o.lower() == ans.lower()]
        if not match:
            continue
        if verify(t, img, x["question"], opts) != match[0]:
            continue  # blind second pass disagrees -> drop
        R.shuffle(opts)
        rows.append({"image": path, "context": "", "question": x["question"].strip(), "options": opts,
                     "answer": match[0], "task_id": path, "source": "teacher_image"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["enrich", "propose"])
    ap.add_argument("--in", dest="inp", help="enrich: JSONL from make_images.py")
    ap.add_argument("--images", help="propose: folder of images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max rows (enrich) / images (propose)")
    ap.add_argument("--per-image", type=int, default=3)
    ap.add_argument("--enrich-frac", type=float, default=0.4,
                    help="enrich: fraction of rows sent through the teacher (rest keep plain questions)")
    ap.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    a.backend, a.api_key_env = "ollama", ""
    t = Teacher(a)
    imgs = {}

    if a.mode == "enrich":
        rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
        if a.limit:
            rows = rows[: a.limit]
        # rows not selected pass through with their exact label (plain question, no teacher call)
        jobs = [(lambda r=r: [enrich_row(t, r, imgs)]) if R.random() < a.enrich_frac else (lambda r=r: [r])
                for r in rows]
    else:
        paths = sorted(os.path.join(a.images, f).replace("\\", "/") for f in os.listdir(a.images)
                       if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
        if a.limit:
            paths = paths[: a.limit]
        jobs = [lambda p=p: propose_image(t, p, a.per_image, imgs) for p in paths]

    n = 0
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as pool:
        futs = [pool.submit(j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            for r in fu.result():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
            if i % 5 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} jobs, {n} rows written", flush=True)
            imgs.pop(next(iter(imgs)), None) if len(imgs) > 64 else None  # bound memory
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
