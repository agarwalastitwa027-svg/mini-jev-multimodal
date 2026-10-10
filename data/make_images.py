"""Procedurally draw images whose answers are known exactly (no teacher needed for the label).

    python data/make_images.py --n 400 --out data/mm_procedural.jsonl --img-dir data/images/proc

Kinds: bar chart (highest / lowest / A-vs-B), pie chart (largest slice), line chart (trend),
shape scene (count of a colour+shape, colour of the largest shape, is-there-a-X).
Every row: {image, context, question, options, answer, task_id, source, kind}; task_id = image id,
so data/split.py keeps all questions about one image in the same split.
"""
import argparse
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

R = random.Random(0)

LABEL_SETS = [
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    ["North", "South", "East", "West", "Central"],
    ["Apples", "Bananas", "Cherries", "Grapes", "Mangoes", "Pears"],
    ["Q1", "Q2", "Q3", "Q4"],
    ["Email", "Phone", "Chat", "Social", "Walk-in"],
    ["Alpha", "Beta", "Gamma", "Delta", "Omega"],
    ["Delhi", "Mumbai", "Pune", "Chennai", "Kolkata", "Jaipur"],
    ["Python", "Java", "Go", "Rust", "C++", "Ruby"],
]
TITLES = ["Sales", "Tickets resolved", "Revenue (k)", "Signups", "Complaints", "Units sold",
          "Visitors", "Orders", "Errors per day", "Satisfaction score"]
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
SHAPE_COLORS = {"red": (220, 40, 40), "blue": (40, 80, 220), "green": (40, 170, 70),
                "yellow": (240, 210, 40), "purple": (140, 60, 180), "orange": (240, 140, 30)}


def spread_values(n, lo=5, hi=100, min_gap=8):
    """Distinct integers so the max / min is unambiguous."""
    while True:
        v = [R.randint(lo, hi) for _ in range(n)]
        s = sorted(v)
        if len(set(v)) == n and s[-1] - s[-2] >= min_gap and s[1] - s[0] >= min_gap:
            return v


def save_fig(fig, path):
    fig.savefig(path, dpi=R.choice([70, 90, 110]))
    plt.close(fig)


def bar_chart(path):
    labels = R.choice(LABEL_SETS)
    labels = R.sample(labels, R.randint(4, min(6, len(labels))))
    vals = spread_values(len(labels))
    fig, ax = plt.subplots(figsize=(R.uniform(5, 7), R.uniform(3.5, 5)))
    horizontal = R.random() < 0.3
    color = R.choice(COLORS)
    (ax.barh if horizontal else ax.bar)(labels, vals, color=color)
    ax.set_title(R.choice(TITLES))
    if R.random() < 0.6:
        for i, v in enumerate(vals):
            (ax.text(v + 1, i, str(v), va="center") if horizontal else ax.text(i, v + 1, str(v), ha="center"))
    save_fig(fig, path)
    hi, lo = labels[vals.index(max(vals))], labels[vals.index(min(vals))]
    a, b = R.sample(range(len(labels)), 2)
    return [
        ("Which category has the highest value?", labels, hi),
        ("Which category has the lowest value?", labels, lo),
        (f"Is the value for {labels[a]} higher than the value for {labels[b]}?", ["yes", "no"],
         "yes" if vals[a] > vals[b] else "no"),
    ]


def pie_chart(path):
    labels = R.sample(R.choice(LABEL_SETS), 4)
    vals = spread_values(4, 10, 60)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(vals, labels=labels, colors=R.sample(COLORS, 4), autopct="%1.0f%%" if R.random() < .5 else None)
    ax.set_title(R.choice(TITLES))
    save_fig(fig, path)
    return [("Which slice is the largest?", labels, labels[vals.index(max(vals))])]


def line_chart(path):
    n = R.randint(6, 10)
    trend = R.choice(["rising", "falling", "flat"])
    base = R.randint(30, 60)
    step = {"rising": R.uniform(4, 8), "falling": -R.uniform(4, 8), "flat": 0}[trend]
    ys = [base + step * i + R.uniform(-2, 2) for i in range(n)]
    fig, ax = plt.subplots(figsize=(R.uniform(5, 7), R.uniform(3.5, 5)))
    ax.plot(range(n), ys, marker="o", color=R.choice(COLORS))
    ax.set_ylim(0, max(ys) * 1.4 + 5)
    ax.set_title(R.choice(TITLES))
    save_fig(fig, path)
    return [("Is the overall trend rising, falling or flat?", ["rising", "falling", "flat"], trend),
            ("Is the last value higher than the first value?", ["yes", "no"], "yes" if ys[-1] > ys[0] else "no")]


def shape_scene(path):
    W = H = 384
    im = Image.new("RGB", (W, H), R.choice([(255, 255, 255), (245, 245, 235), (230, 235, 245)]))
    d = ImageDraw.Draw(im)
    objs, placed = [], []
    for _ in range(R.randint(3, 8)):
        for _try in range(50):
            size = R.randint(30, 90)
            x, y = R.randint(5, W - size - 5), R.randint(5, H - size - 5)
            if all(abs(x - px) > (size + ps) * 0.75 or abs(y - py) > (size + ps) * 0.75 for px, py, ps in placed):
                break
        else:
            continue
        color, shape = R.choice(list(SHAPE_COLORS)), R.choice(["circle", "square", "triangle"])
        box = [x, y, x + size, y + size]
        if shape == "circle":
            d.ellipse(box, fill=SHAPE_COLORS[color])
        elif shape == "square":
            d.rectangle(box, fill=SHAPE_COLORS[color])
        else:
            d.polygon([(x + size / 2, y), (x, y + size), (x + size, y + size)], fill=SHAPE_COLORS[color])
        placed.append((x, y, size))
        objs.append((color, shape, size))
    im.save(path)
    c, s = R.choice(objs)[:2]
    count = sum(1 for o in objs if o[0] == c and o[1] == s)
    sizes = sorted(o[2] for o in objs)
    qs = [(f"How many {c} {s}s are in the image?", [str(i) for i in range(0, 6)], str(min(count, 5)))]
    if len(sizes) < 2 or sizes[-1] - sizes[-2] >= 8:  # unambiguous largest shape
        big = max(objs, key=lambda o: o[2])
        qs.append(("What colour is the largest shape?", list(SHAPE_COLORS), big[0]))
    q_c, q_s = R.choice(list(SHAPE_COLORS)), R.choice(["circle", "square", "triangle"])
    present = any(o[0] == q_c and o[1] == q_s for o in objs)
    qs.append((f"Is there a {q_c} {q_s} in the image?", ["yes", "no"], "yes" if present else "no"))
    return qs


MAKERS = {"bar": bar_chart, "pie": pie_chart, "line": line_chart, "shapes": shape_scene}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="number of images")
    ap.add_argument("--out", default="data/mm_procedural.jsonl")
    ap.add_argument("--img-dir", default="data/images/proc")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    R.seed(a.seed)
    os.makedirs(a.img_dir, exist_ok=True)
    kinds = list(MAKERS)
    n_rows = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(a.n):
            kind = kinds[i % len(kinds)]
            path = os.path.join(a.img_dir, f"{kind}_{a.seed}_{i:05d}.png").replace("\\", "/")
            for q, opts, ans in MAKERS[kind](path):
                opts = list(opts)
                R.shuffle(opts) if opts != ["yes", "no"] and not opts[0].isdigit() else None
                f.write(json.dumps({"image": path, "context": "", "question": q, "options": opts,
                                    "answer": ans, "task_id": f"{a.seed}_{i}", "source": "procedural",
                                    "kind": kind}, ensure_ascii=False) + "\n")
                n_rows += 1
    print(f"{a.n} images, {n_rows} questions -> {a.out}")


if __name__ == "__main__":
    main()
