"""Helpers shared by the data builders."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re

# Benchmarks scored by the Jev Decision Index (text 0.2/0.3 and vision 0.3). We never train on any
# split of these, so leaderboard scores measure generalisation, not memorised benchmark formats.
DI_FAMILIES = {
    # text suite
    "acos", "anli", "api-bank", "arc", "arc-challenge", "arc-easy", "amazon esci", "banking77", "bfcl", "bpomp",
    "bright", "boxoban", "clinc", "clinc150", "clinc_oos", "cladder", "cruxeval", "chessbench", "codenames",
    "contractnli", "finentity", "forecastbench", "gpqa", "gsm8k", "hle", "habermas", "hanabi", "hellaswag",
    "humicroedit", "mmlu", "miniwob", "musr", "nli4ct", "pop909", "rtfm", "routerbench", "sata", "sgd",
    "scienceworld", "simplebench", "toolret", "vast", "winogrande", "cfcolor", "isarcasmeval", "ragtruth",
    "phishnchips",
    # vision suite
    "cv-bench", "blink", "realworldqa", "charxiv", "infographicvqa", "docvqa", "mind2web", "winoground",
    "cord", "funsd", "hateful_memes", "r-bench", "mmmu", "mmmu-pro",
}


def is_di_family(source: str) -> bool:
    s = source.lower().replace(" ", "")
    return any(f.replace(" ", "") in s for f in DI_FAMILIES)


def stable_hash(*parts) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "big")


def split_of(key, val: float, test: float) -> str:
    """Deterministic train/val/test assignment from a group key."""
    u = (stable_hash(key) % 10_000) / 10_000
    return "test" if u < test else "val" if u < test + val else "train"


def save_image(im, path: str, max_side: int = 1024) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    im.save(path)
    return path.replace("\\", "/")


def criteria_from(options) -> dict:
    """Option strings -> criteria map; duplicate strings get a letter suffix so keys stay unique."""
    opts = [str(o).strip() for o in options]
    if len({o.lower() for o in opts}) != len(opts):
        opts = [f"{o} ({chr(65 + i)})" for i, o in enumerate(opts)]
    return {o: o for o in opts}


def question_from_options(instructions: str, options) -> dict:
    """Pick the question type from the options: yes/no -> noul, consecutive integers -> score, else choice."""
    opts = [str(o).strip() for o in options]
    low = [o.lower() for o in opts]
    if sorted(low) == ["no", "yes"]:
        return {"type": "noul", "instructions": instructions}
    if all(re.fullmatch(r"-?\d+", o) for o in opts):
        nums = [int(o) for o in opts]
        if sorted(nums) == list(range(min(nums), max(nums) + 1)) and len(nums) >= 3:
            ordered = [str(n) for n in sorted(nums)]
            return {"type": "score", "instructions": instructions, "criteria": {k: "" for k in ordered}}
    return {"type": "choice", "instructions": instructions, "criteria": criteria_from(opts)}


def expected_for(q: dict, answer, options) -> object:
    if q["type"] == "noul":
        return str(answer).strip().lower() == "yes"
    if q["type"] == "choice":
        keys = list(q["criteria"])
        opts = [str(o).strip() for o in options]
        return keys[opts.index(str(answer).strip())] if str(answer).strip() in opts else str(answer).strip()
    return str(answer).strip()


R = random.Random(0)
