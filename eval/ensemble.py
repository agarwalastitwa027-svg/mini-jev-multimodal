"""Average the probabilities of several evaluation runs (e.g. training seeds) into one run file.

    python -m eval.ensemble --out results/seeds_ens.json results/seed0.json results/seed1.json results/seed2.json

Each input is the JSON from `python -m eval.evaluate --out`; temperatures stored in an input are applied
first. The output has the same format, so eval.compare can score it like any other run.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def scaled(p, T):
    if p is None or not T or T == 1.0:
        return None if p is None else np.array(p)
    lg = np.log(np.maximum(np.array(p), 1e-12)) / T
    e = np.exp(lg - lg.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["mean", "geo"], default="mean", help="arithmetic or geometric mean of probabilities")
    a = ap.parse_args()
    runs = [json.load(open(p)) for p in a.inputs]
    out = {"predictor": f"ensemble of {len(runs)} runs ({a.mode})", "temps": None, "files": {}}
    for f in runs[0]["files"]:
        by_run = []
        for r in runs:
            temps = r.get("temps") or {}
            by_run.append({(str(x["id"]), x["key"]): (x, scaled(x["probs"], temps.get(x["type"], 1.0)))
                           for x in r["files"][f]["records"]})
        recs = []
        for k, (x, _) in by_run[0].items():
            ps = [br[k][1] for br in by_run if k in br]
            if len(ps) != len(by_run) or any(p is None for p in ps):
                y = dict(x, probs=None, status="unsupported")
            else:
                m = np.mean(ps, 0) if a.mode == "mean" else np.exp(np.mean(np.log(np.maximum(ps, 1e-12)), 0))
                y = dict(x, probs=(m / m.sum()).tolist(), status="ok")
            recs.append(y)
        out["files"][f] = {"records": recs}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
