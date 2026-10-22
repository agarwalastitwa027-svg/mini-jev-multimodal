"""Remove training rows that overlap with the Jev Decision Index suites.

    # get the suites first (inside a clone of github.com/apolinario/decision-index):
    #   python -m decision_index suite download --dir suite
    #   decision-index suite rebuild --vision --work work          (optional, for image hashes)
    python data/decontam.py --train data/mix/train.jsonl --suite-dir ../decision-index/suite \
        --vision-dir ../decision-index/work/vision --out data/mix/train.jsonl

Text: word 13-grams of the normalised state + question text, sampled consistently (a gram is kept
when its hash is divisible by --sample, on both sides, so overlaps are still found). A training row
is dropped when >= --threshold of its sampled grams occur in the suite, or when its whole
normalised text (short rows) occurs verbatim. Images: exact sha256 of the file bytes.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jev.schema import as_text  # noqa: E402

WORD = re.compile(r"[a-z0-9]+")


def norm_words(text: str):
    return WORD.findall(text.lower())


def grams(words, n, sample):
    out = set()
    for i in range(len(words) - n + 1):
        h = int.from_bytes(hashlib.blake2b(" ".join(words[i:i + n]).encode(), digest_size=8).digest(), "big")
        if h % sample == 0:
            out.add(h)
    return out


def row_text(r):
    parts = [as_text(r.get("state", ""))]
    for q in (r.get("questions") or {}).values():
        parts.append(as_text(q.get("instructions", "")))
        crit = q.get("criteria") or {}
        parts += [f"{k} {as_text(v)}" for k, v in (crit.items() if isinstance(crit, dict) else [])]
    return " ".join(parts)


def iter_suite(suite_dir):
    files = glob.glob(os.path.join(suite_dir, "**", "*.jsonl.gz"), recursive=True) + \
        glob.glob(os.path.join(suite_dir, "**", "*.jsonl"), recursive=True)
    for p in files:
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--vision-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--sample", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--image-root", default=".")
    a = ap.parse_args()

    index, exact, n_suite = set(), set(), 0
    for r in iter_suite(a.suite_dir):
        w = norm_words(row_text(r))
        index |= grams(w, a.n, a.sample)
        if len(w) < 3 * a.n:
            exact.add(" ".join(w))
        n_suite += 1
    print(f"suite rows {n_suite}, sampled grams {len(index)}, short texts {len(exact)}")

    img_hashes = set()
    if a.vision_dir:
        for p in glob.glob(os.path.join(a.vision_dir, "**", "*"), recursive=True):
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
                img_hashes.add(sha256_file(p))
        print(f"suite images {len(img_hashes)}")

    rows = [json.loads(l) for l in open(a.train, encoding="utf-8") if l.strip()]
    keep, dropped = [], Counter()
    for r in rows:
        w = norm_words(row_text(r))
        g = grams(w, a.n, a.sample)
        hit = bool(g) and len(g & index) / len(g) >= a.threshold
        if not hit and not r.get("images"):  # image rows reuse templated questions; judge them by image
            hit = " ".join(w) in exact and len(w) >= 5
        if not hit and img_hashes:
            for p in r.get("images") or []:
                full = p if os.path.isabs(p) else os.path.join(a.image_root, p)
                if os.path.exists(full) and sha256_file(full) in img_hashes:
                    hit = True
                    break
        if hit:
            dropped[r.get("source", "?")] += 1
        else:
            keep.append(r)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"kept {len(keep)} / {len(rows)} rows; dropped by source: {dict(dropped)}")
    json.dump({"kept": len(keep), "total": len(rows), "dropped": dict(dropped),
               "suite_rows": n_suite, "suite_images": len(img_hashes)},
              open(os.path.splitext(a.out)[0] + "_decontam.json", "w"), indent=1)


if __name__ == "__main__":
    main()
