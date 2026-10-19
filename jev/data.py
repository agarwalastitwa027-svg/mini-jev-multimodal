"""Reading v2 rows and turning them into per-question training examples."""
from __future__ import annotations

import gzip
import json
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from jev.schema import options_of, target_distribution


def read_jsonl(paths, limit: Optional[int] = None, seed: int = 0) -> List[Dict]:
    if isinstance(paths, str):
        paths = [paths]
    rows = []
    for p in paths:
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt", encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    if limit and len(rows) > limit:
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


def write_jsonl(path: str, rows: Iterable[Dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def modality(row: Dict) -> str:
    return row.get("modality") or ("vision" if row.get("images") else "text")


@dataclass
class Example:
    row: int            # index into the rows list
    key: str            # question key
    qtype: str
    target: List[float]
    modality: str
    source: str


def flatten(rows: List[Dict], modalities: Optional[List[str]] = None) -> List[Example]:
    """One Example per labelled question; rows outside `modalities` are skipped."""
    out = []
    for i, r in enumerate(rows):
        m = modality(r)
        if modalities and m not in modalities:
            continue
        exp, soft = r.get("expected") or {}, r.get("target_probs") or {}
        for k, q in r["questions"].items():
            t = target_distribution(q, exp.get(k), soft.get(k))
            if t is None:
                continue
            out.append(Example(i, k, q["type"], t, m, r.get("source", "?")))
    return out


def shuffled_order(q: Dict, rng: random.Random) -> List[int]:
    """Random option order (stops the model learning 'the answer is usually first').

    Score levels keep their natural order: their order carries meaning.
    """
    n = len(options_of(q)[0])
    order = list(range(n))
    if q["type"] != "score":
        rng.shuffle(order)
    return order


def balanced_order(examples: List[Example], text_frac: Optional[float], rng: random.Random) -> List[int]:
    """One epoch of example indices, resampled so text makes up `text_frac` of the steps.

    With text_frac=None the natural mix is used. The minority modality is oversampled (with
    replacement) rather than the majority being dropped, so no data is thrown away.
    """
    idx = list(range(len(examples)))
    if text_frac is None:
        rng.shuffle(idx)
        return idx
    text = [i for i in idx if examples[i].modality == "text"]
    vis = [i for i in idx if examples[i].modality != "text"]
    if not text or not vis:
        rng.shuffle(idx)
        return idx
    n = len(idx)
    nt, nv = int(round(n * text_frac)), n - int(round(n * text_frac))

    def draw(pool, k):
        out = pool[:]
        rng.shuffle(out)
        while len(out) < k:
            out += rng.sample(pool, min(len(pool), k - len(out)))
        return out[:k]
    order = draw(text, nt) + draw(vis, nv)
    rng.shuffle(order)
    return order
