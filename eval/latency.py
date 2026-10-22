"""Latency, throughput and memory of a run, with and without the shared-state cache.

    python -m eval.latency --run runs/full_seed0 --out results/latency.json

For each state size (text tokens, and with one image) it times one request with 1 question and with
N questions about the same state, cached (state encoded once) and uncached (each question re-reads
the state, the naive behaviour). Reports p50 / p95 over --reps timed calls after warm-up, and peak VRAM.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

FILLER = ("The customer wrote to support about an order that arrived late and partly damaged. "
          "They asked for a refund for the broken item and a discount on the next order. ")


def questions(n):
    qs = {}
    for i in range(n):
        if i % 3 == 0:
            qs[f"q{i}"] = {"type": "choice", "instructions": f"Which queue should handle this? ({i})",
                           "criteria": {"billing": "payments, refunds", "shipping": "deliveries",
                                        "technical": "product faults", "general": "anything else"}}
        elif i % 3 == 1:
            qs[f"q{i}"] = {"type": "noul", "instructions": f"Is the customer asking for a refund? ({i})"}
        else:
            qs[f"q{i}"] = {"type": "score", "instructions": f"How upset is the customer? ({i})",
                           "criteria": {str(k): "" for k in range(1, 6)}}
    return qs


def make_image(path, w=1024, h=768):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    for i in range(6):
        d.rectangle([60 + i * 150, h - 80 - i * 90, 160 + i * 150, h - 60], fill=(40 * i, 120, 200 - 30 * i))
    im.save(path)
    return path


def bench(dec, state, qs, images, reps):
    dec.decide(state, qs, images)  # warm-up
    dec.decide(state, qs, images)
    times = []
    for _ in range(reps):
        dec.synchronize()
        t = time.perf_counter()
        dec.decide(state, qs, images)
        dec.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"p50_ms": statistics.median(times), "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
            "input_tokens": dec.last_stats["input_tokens"], "prefix_tokens": dec.last_stats["prefix_tokens"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--state-tokens", nargs="+", type=int, default=[100, 1000, 4000])
    ap.add_argument("--questions", nargs="+", type=int, default=[1, 8])
    ap.add_argument("--image-pixels", type=int, default=1_638_400, help="image cap for the image case")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="results/latency.json")
    a = ap.parse_args()
    from jev.decide import Decider
    dec = Decider.from_pretrained(a.run, max_pixels=a.image_pixels)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    img = make_image(os.path.join(os.path.dirname(a.out) or ".", "latency_probe.png"))
    words_per_token = 0.75
    cases = []
    for n_tok in a.state_tokens:
        state = (FILLER * (1 + int(n_tok * words_per_token / len(FILLER.split()))))
        cases.append((f"text {n_tok} tok", state, []))
    if dec.vision is not None:
        cases.append(("image + 100 tok", FILLER, [img]))
    if dec.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    results = []
    for name, state, images in cases:
        for nq in a.questions:
            row = {"case": name, "questions": nq}
            for cached in (True, False):
                dec.cache_prefix = cached
                try:
                    r = bench(dec, state, questions(nq), images, a.reps)
                except Exception as e:
                    r = {"error": f"{type(e).__name__}: {e}"}
                row["cached" if cached else "uncached"] = r
            results.append(row)
            c, u = row["cached"], row["uncached"]
            print(f"{name:18} Q={nq:<2} cached p50 {c.get('p50_ms', float('nan')):7.1f} ms | "
                  f"uncached p50 {u.get('p50_ms', float('nan')):7.1f} ms | prefix {c.get('prefix_tokens')} tok")
    dec.cache_prefix = True
    out = {"run": a.run, "device": torch.cuda.get_device_name(0) if dec.device.type == "cuda" else "cpu",
           "dtype": str(dec.dtype), "results": results}
    if dec.device.type == "cuda":
        out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    json.dump(out, open(a.out, "w"), indent=1)
    md = ["| Case | Questions | Cached p50 / p95 (ms) | Uncached p50 / p95 (ms) |", "|---|---|---|---|"]
    for r in results:
        c, u = r["cached"], r["uncached"]
        f = lambda d: f"{d['p50_ms']:.0f} / {d['p95_ms']:.0f}" if "p50_ms" in d else "–"
        md.append(f"| {r['case']} | {r['questions']} | {f(c)} | {f(u)} |")
    open(a.out.replace(".json", ".md"), "w").write(
        f"# Latency ({out['device']}, {out['dtype']})\n\n" + "\n".join(md) +
        (f"\n\nPeak VRAM: {out['peak_vram_gb']} GB\n" if "peak_vram_gb" in out else "\n"))
    print("\n".join(md))


if __name__ == "__main__":
    main()
