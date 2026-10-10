"""Head-to-head: mini-Jev vs an LLM in Ollama with structured output (same test set).

    ollama pull qwen3:1.7b        # same base size = fair comparison
    python bench_vs_ollama.py --run runs/qwen3-1.7b --test data/test.jsonl --n 200 --ollama-model qwen3:1.7b

Measures accuracy and per-item latency (batch size 1) for both.
Tip: on an 8 GB GPU run the two halves separately (--skip-jev / --skip-ollama) so they
don't fight for VRAM, then compare the printed numbers.
"""
import argparse
import json
import random
import statistics
import time

import requests


def bench_jev(run, rows):
    import torch
    from infer import JevRunner
    jev = JevRunner(run)
    correct, lat = 0, []
    for i, ex in enumerate(rows):
        qs = {"q": ex["options"]}
        if jev.device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        r = jev.decide(ex.get("context", ""), qs, {"q": ex["question"]}, image=ex.get("image"))
        if jev.device.type == "cuda":
            torch.cuda.synchronize()
        if i:  # first call is warm-up
            lat.append(time.perf_counter() - t)
        correct += str(r["q"]["answer"]).lower() == str(ex["answer"]).lower()
    return correct / len(rows), lat


def load_b64(path, max_side=768):
    import base64
    import io

    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def bench_ollama(model, url, rows, think):
    correct, lat, bad = 0, [], 0
    for i, ex in enumerate(rows):
        schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": ex["options"]}},
                  "required": ["answer"]}
        ctx = f"Context:\n{ex['context']}\n\n" if ex.get("context") else ""
        prompt = (f"{ctx}Question: {ex['question']}\n"
                  f"Options: {json.dumps(ex['options'])}\nAnswer with the single best option.")
        msg = {"role": "user", "content": prompt}
        if ex.get("image"):  # same image preprocessing cost as a real deployment: downscale + base64
            msg["images"] = [load_b64(ex["image"])]
        t = time.perf_counter()
        r = requests.post(f"{url}/api/chat", timeout=600, json={
            "model": model, "stream": False, "think": think, "format": schema,
            "options": {"temperature": 0}, "messages": [msg]})
        dt = time.perf_counter() - t
        if i:
            lat.append(dt)
        try:
            ans = json.loads(r.json()["message"]["content"])["answer"]
        except Exception:
            ans, bad = None, bad + 1
        correct += str(ans).lower() == str(ex["answer"]).lower()
    return correct / len(rows), lat, bad


def summary(name, acc, lat, extra=""):
    lat_ms = sorted(x * 1000 for x in lat) or [0]
    p95 = lat_ms[int(0.95 * (len(lat_ms) - 1))]
    print(f"{name:28} acc {acc:.4f} | median {statistics.median(lat_ms):7.0f} ms | p95 {p95:7.0f} ms {extra}")
    return statistics.median(lat_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--test", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ollama-model", default="qwen3.5:9b-q4_K_M")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--think", action="store_true", help="let the Ollama model reason first (slower)")
    ap.add_argument("--skip-jev", action="store_true")
    ap.add_argument("--skip-ollama", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.test, encoding="utf-8") if l.strip()]
    random.Random(0).shuffle(rows)
    rows = rows[: a.n]
    print(f"{len(rows)} test items\n")
    med = {}
    if not a.skip_jev:
        acc, lat = bench_jev(a.run, rows)
        med["jev"] = summary("mini-Jev (prefill only)", acc, lat)
    if not a.skip_ollama:
        acc, lat, bad = bench_ollama(a.ollama_model, a.ollama_url, rows, a.think)
        med["llm"] = summary(f"Ollama {a.ollama_model}", acc, lat, f"| unparseable {bad}")
    if len(med) == 2 and med["jev"]:
        print(f"\nspeed-up (median latency): {med['llm'] / med['jev']:.1f}x")


if __name__ == "__main__":
    main()
