"""Run one predictor over test files and report decision metrics, per modality and per source.

    python -m eval.evaluate --kind minijev --path runs/full_seed0 --test data/mix/test_text.jsonl data/mix/test_vision.jsonl
    python -m eval.evaluate --kind base --model Qwen/Qwen3-VL-2B-Instruct --test ...
    python -m eval.evaluate --kind ollama --model qwen3.5:9b-q4_K_M --test ... --limit 300

Rows the predictor refuses (Unsupported) or fails on count as wrong, as on the Decision Index.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import traceback
from collections import defaultdict
from typing import Dict, List

from jev.data import modality, read_jsonl
from jev.metrics import reliability_bins, summarize
from jev.schema import Unsupported, options_of, target_distribution


def collect(pred, rows: List[Dict], warmup: int = 1, progress: int = 200) -> List[Dict]:
    from eval.predictors import timed
    recs = []
    for r in rows[:warmup]:  # CUDA kernels / caches; not timed
        try:
            pred.answer(r)
        except Exception:
            pass
    for i, r in enumerate(rows):
        status, out, ms = "ok", {}, None
        try:
            out, ms = timed(pred, r)
        except Unsupported as e:
            status, err = "unsupported", str(e)
        except Exception as e:  # keep going; count as wrong
            status, err = "error", f"{type(e).__name__}: {e}"
            if os.environ.get("JEV_EVAL_DEBUG"):
                traceback.print_exc()
        exp, soft = r.get("expected") or {}, r.get("target_probs") or {}
        for k, q in r["questions"].items():
            t = target_distribution(q, exp.get(k), soft.get(k))
            if t is None:
                continue
            recs.append({"id": r.get("id", i), "source": r.get("source", "?"), "modality": modality(r),
                         "key": k, "type": q["type"], "n_options": len(options_of(q)[0]),
                         "gold": max(range(len(t)), key=t.__getitem__), "status": status,
                         "probs": out.get(k) if status == "ok" else None,
                         "ms": ms / max(1, len(r["questions"])) if ms is not None else None,
                         "row_ms": ms})
        if progress and (i + 1) % progress == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    return recs


def rescale(p, T: float):
    if p is None or T == 1.0:
        return p
    lg = [math.log(max(x, 1e-12)) / T for x in p]
    m = max(lg)
    e = [math.exp(x - m) for x in lg]
    s = sum(e)
    return [x / s for x in e]


def report(recs: List[Dict], temps: Dict[str, float] | None = None) -> Dict:
    temps = temps or {}
    groups = defaultdict(list)
    lat = defaultdict(list)
    for r in recs:
        p = rescale(r["probs"], temps.get(r["type"], 1.0))
        for g in ("all", f"modality:{r['modality']}", f"source:{r['source']}", f"type:{r['type']}"):
            groups[g].append((p, r["gold"]))
            if r["row_ms"] is not None:
                lat[g].append(r["row_ms"])
    out = {}
    for g, items in sorted(groups.items()):
        m = summarize(items)
        if lat[g]:
            v = sorted(lat[g])
            m["row_ms_p50"] = statistics.median(v)
            m["row_ms_p95"] = v[min(len(v) - 1, int(0.95 * len(v)))]
        out[g] = m
    conf, corr = [], []
    for r in recs:
        p = rescale(r["probs"], temps.get(r["type"], 1.0))
        if p is not None:
            j = max(range(len(p)), key=p.__getitem__)
            conf.append(p[j])
            corr.append(int(j == r["gold"]))
    out["_reliability"] = reliability_bins(conf, corr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["minijev", "base", "ollama"], required=True)
    ap.add_argument("--path", help="run folder or Hub repo (minijev)")
    ap.add_argument("--model", help="base model id (base) or Ollama model (ollama)")
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--out", default=None, help="write records + metrics JSON here")
    ap.add_argument("--n-perm", type=int, default=1, help="minijev: average choice questions over this many option orders")
    ap.add_argument("--fit-temps", nargs="+", default=None,
                    help="validation file(s): fit one temperature per question type on them (post-hoc scaling) and apply it")
    ap.add_argument("--val-limit", type=int, default=1500)
    ap.add_argument("--min-per-type", type=int, default=50)
    a = ap.parse_args()
    from eval.predictors import make_predictor
    spec = {"kind": a.kind, "path": a.path, "model": a.model, "image_root": a.image_root}
    if a.max_pixels and a.kind != "ollama":
        spec["max_pixels"] = a.max_pixels
    if a.n_perm > 1:
        spec["n_perm"] = a.n_perm
    pred = make_predictor(spec)
    temps = None
    if a.fit_temps:
        from calibrate import fit_temperature
        from jev.schema import TYPES
        vrecs = collect(pred, read_jsonl(a.fit_temps, a.val_limit))
        pooled = fit_temperature([(r["probs"], r["gold"]) for r in vrecs])
        temps = {}
        for t in TYPES:
            pairs = [(r["probs"], r["gold"]) for r in vrecs if r["type"] == t and r["probs"] is not None]
            temps[t] = fit_temperature(pairs) if len(pairs) >= a.min_per_type else pooled
        print("fitted temperatures:", {k: round(v, 3) for k, v in temps.items()}, "| pooled", round(pooled, 3))
    result = {"predictor": pred.name + (" + temperature scaling" if temps else ""), "temps": temps, "files": {}}
    for f in a.test:
        rows = read_jsonl(f, a.limit)
        recs = collect(pred, rows)
        rep = report(recs, temps)
        result["files"][os.path.basename(f)] = {"metrics": rep, "records": recs}
        al = rep["all"]
        print(f"{os.path.basename(f)}: acc {al['accuracy']:.4f} skill {al['skill']:.4f} ece {al['ece']:.4f} "
              f"nll {al['nll']:.4f} unanswered {al['unanswered']}/{al['n']} p50 {al.get('row_ms_p50', 0):.0f} ms")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(result, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
