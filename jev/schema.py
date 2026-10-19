"""Typed decision questions and the v2 data format.

A request (and a training row) is one *state* plus several typed questions about it:

    {
      "id": "...", "source": "ag_news", "modality": "text" | "vision",
      "state": "free text" | {"any": "json"},          # may be "" when images carry the content
      "images": ["path/or/data-url", ...],              # 0 to 4 images
      "questions": {
        "topic":  {"type": "choice", "instructions": "What is the topic?",
                   "criteria": {"world": "World news", "sports": "Sports"}},
        "urgent": {"type": "noul",   "instructions": "Is this urgent?"},
        "stars":  {"type": "score",  "instructions": "How satisfied is the customer?",
                   "criteria": {"1": "very unhappy", "2": "", "3": "", "4": "", "5": "very happy"}}
      },
      "expected":     {"topic": "sports", "urgent": true, "stars": "4"},   # training / eval only
      "target_probs": {"topic": {"world": 0.1, "sports": 0.9}}              # optional soft labels
    }

Question types
  choice  pick one option key from `criteria` (2 to 255 options) -> probability per option
  noul    yes/no -> P(yes); internally the options are "yes" and "no"
  score   ordered levels (criteria keys, in order) -> probability per level + expected score

`choice` and `noul` match the Jev / Decision Index wire format exactly. `score` is an extension:
the Decision Index sends ordinal questions as `choice`, which this model also accepts.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple

TYPES = ("choice", "noul", "score")
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}
NOUL_OPTIONS = {"yes": "yes", "no": "no"}
MAX_OPTIONS = 255
MAX_IMAGES = 8


class Unsupported(ValueError):
    """A declared capacity limit was hit (context length, option count, image count, memory).

    Never truncate or drop options to make a request fit; raise this instead.
    The message contains 'maximum context length' or a similar marker so HTTP clients
    (the Decision Index `http` engine) classify it as unsupported rather than an error.
    """


def as_text(x) -> str:
    """State and instructions may be strings or JSON objects; render objects compactly."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def options_of(q: Dict) -> Tuple[List[str], List[str]]:
    """Return (keys, descriptions) in order for any question type."""
    if q["type"] == "noul":
        return ["yes", "no"], ["yes", "no"]
    crit = q.get("criteria")
    if not crit:
        raise ValueError("choice/score questions need non-empty 'criteria'")
    if isinstance(crit, list):  # convenience: a plain list of option strings
        crit = {str(c): str(c) for c in crit}
    keys = [str(k) for k in crit]
    descs = [as_text(v) for v in crit.values()]
    return keys, descs


def validate_questions(questions: Dict) -> None:
    if not isinstance(questions, dict) or not questions:
        raise ValueError("'questions' must be a non-empty object")
    for k, q in questions.items():
        t = q.get("type")
        if t not in TYPES:
            raise Unsupported(f"unsupported question type {t!r} for {k!r}")
        if t != "noul":
            keys, _ = options_of(q)
            if len(keys) < 2:
                raise ValueError(f"question {k!r} needs at least 2 options")
            if len(keys) > MAX_OPTIONS:
                raise Unsupported(f"{len(keys)} options per choice exceeds the declared limit of {MAX_OPTIONS}")
            if len(set(keys)) != len(keys):
                raise ValueError(f"question {k!r} has duplicate option keys")


def score_values(keys: List[str]) -> List[float]:
    """Numeric value of each score level: the key itself if numeric, else its index."""
    try:
        return [float(k) for k in keys]
    except ValueError:
        return [float(i) for i in range(len(keys))]


def target_distribution(q: Dict, expected=None, soft=None) -> List[float] | None:
    """Training target over the question's options (in option order).

    `soft` (a {key: p} map) wins over `expected`. For noul, `expected` may be a bool,
    "yes"/"no", or a probability. Returns None when there is no label.
    """
    keys, _ = options_of(q)
    if soft:
        p = [float(soft.get(k, 0.0)) for k in keys]
        s = sum(p)
        if s > 0:
            return [x / s for x in p]
    if expected is None:
        return None
    if q["type"] == "noul":
        if isinstance(expected, bool):
            py = 1.0 if expected else 0.0
        elif isinstance(expected, (int, float)):
            py = float(expected)
        else:
            py = 1.0 if str(expected).strip().lower() in ("yes", "true", "1") else 0.0
        return [py, 1.0 - py]
    norm = [k.strip().lower() for k in keys]
    e = str(expected).strip().lower()
    if e not in norm:
        raise ValueError(f"expected answer {expected!r} not among options {keys}")
    return [1.0 if n == e else 0.0 for n in norm]


def answer_from_probs(q: Dict, probs: List[float]) -> Dict:
    """Build the wire-format answer for one question from option probabilities."""
    keys, _ = options_of(q)
    s = sum(probs)
    probs = [p / s for p in probs] if s > 0 else [1.0 / len(probs)] * len(probs)
    if q["type"] == "noul":
        return {"type": "noul", "noul": float(probs[0])}
    k = max(range(len(probs)), key=probs.__getitem__)
    out = {"type": q["type"], "choice": keys[k], "probabilities": {kk: float(p) for kk, p in zip(keys, probs)}}
    if q["type"] == "score":
        vals = score_values(keys)
        out["score"] = float(sum(v * p for v, p in zip(vals, probs)))
    return out


def predicted_probs(q: Dict, answer: Dict) -> List[float]:
    """Inverse of answer_from_probs: option probabilities in option order."""
    keys, _ = options_of(q)
    if q["type"] == "noul":
        p = float(answer["noul"])
        return [p, 1.0 - p]
    return [float(answer["probabilities"][k]) for k in keys]


def entropy(p: List[float]) -> float:
    return -sum(x * math.log(x) for x in p if x > 0)
