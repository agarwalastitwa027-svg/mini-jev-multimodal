"""Write a tiny dataset (text + image rows, all three question types) for tests and CI."""
from __future__ import annotations

import json
import os
import random


def make(out_dir: str, n: int = 24, seed: int = 0) -> str:
    from PIL import Image, ImageDraw
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    rng = random.Random(seed)
    rows = []
    colors = {"red": (220, 30, 30), "green": (30, 180, 30), "blue": (30, 30, 220)}
    for i in range(n):
        if i % 2 == 0:  # text row with three questions about one state
            angry = rng.random() < 0.5
            state = ("My package arrived damaged and nobody answers. I want a refund now!" if angry
                     else "Thanks, the package arrived on time and works great.")
            rows.append({
                "id": f"t{i}", "source": "tiny_text", "modality": "text", "state": state, "images": [],
                "questions": {
                    "queue": {"type": "choice", "instructions": "Which queue should handle this?",
                              "criteria": {"billing": "refunds and payments", "shipping": "deliveries",
                                           "general": "anything else"}},
                    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
                    "stars": {"type": "score", "instructions": "How satisfied is the customer?",
                              "criteria": {"1": "very unhappy", "2": "", "3": "", "4": "", "5": "very happy"}},
                },
                "expected": {"queue": "billing" if angry else "general", "urgent": angry,
                             "stars": "1" if angry else "5"},
            })
        else:  # image row: a coloured square
            c = rng.choice(list(colors))
            path = os.path.join(out_dir, "img", f"sq_{i}.png")
            im = Image.new("RGB", (96, 64), "white")
            ImageDraw.Draw(im).rectangle([20, 12, 60, 52], fill=colors[c])
            im.save(path)
            rows.append({
                "id": f"v{i}", "source": "tiny_vision", "modality": "vision", "state": "", "images": [path],
                "questions": {
                    "color": {"type": "choice", "instructions": "What colour is the square?",
                              "criteria": {k: k for k in colors}},
                    "is_red": {"type": "noul", "instructions": "Is the square red?"},
                },
                "expected": {"color": c, "is_red": c == "red"},
                "target_probs": {"color": {k: (0.9 if k == c else 0.05) for k in colors}},
            })
    path = os.path.join(out_dir, "tiny.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


if __name__ == "__main__":
    import sys
    print(make(sys.argv[1] if len(sys.argv) > 1 else "runs/tiny-data"))
