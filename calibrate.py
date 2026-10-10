"""Calibrate (temperature scaling) and evaluate a trained mini-Jev run.

    python calibrate.py --run runs/qwen3-1.7b --val data/val.jsonl \
        --test data/test.jsonl data/test_zeroshot.jsonl data/synthetic_test.jsonl

Reports, per test file, before and after calibration:
    accuracy | NLL | Brier | ECE (15 bins) | selective accuracy at confidence thresholds
and writes a reliability diagram PNG into the run folder.
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from jev.format import collate, encode_example
from jev.model import MiniJev, load_tokenizer, load_vision, pick_dtype


@torch.no_grad()
def raw_logits(model, tok, rows, bs, max_ctx, device, dtype, vision=None):
    out, ys = [], []
    for i in range(0, len(rows), bs):
        batch = [encode_example(tok, ex, max_ctx, vision) for ex in rows[i:i + bs]]
        b = collate(batch, tok.pad_token_id)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            lg = model.run(b, device)
        out += [l.float().cpu() for l in lg]
        ys += b["labels"].tolist()
    return out, ys


def fit_temperature(logits, ys):
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        t = log_t.exp()
        loss = sum(F.cross_entropy((l / t)[None], torch.tensor([y])) for l, y in zip(logits, ys)) / len(ys)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp())


def metrics(logits, ys, T=1.0, bins=15):
    conf, correct, nll, brier = [], [], 0.0, 0.0
    for l, y in zip(logits, ys):
        p = F.softmax(l / T, -1)
        c, k = p.max(0)
        conf.append(float(c))
        correct.append(int(k) == y)
        nll -= float(torch.log(p[y] + 1e-12))
        onehot = torch.zeros_like(p)
        onehot[y] = 1
        brier += float(((p - onehot) ** 2).sum())
    n = len(ys)
    ece, rel = 0.0, []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        idx = [j for j in range(n) if lo < conf[j] <= hi]
        if idx:
            acc = sum(correct[j] for j in idx) / len(idx)
            avg = sum(conf[j] for j in idx) / len(idx)
            ece += len(idx) / n * abs(acc - avg)
            rel.append((avg, acc, len(idx)))
    selective = {}
    for th in [0.5, 0.7, 0.8, 0.9, 0.95]:
        idx = [j for j in range(n) if conf[j] >= th]
        selective[th] = (len(idx) / n, (sum(correct[j] for j in idx) / len(idx)) if idx else None)
    return {"acc": sum(correct) / n, "nll": nll / n, "brier": brier / n, "ece": ece,
            "selective": selective, "reliability": rel, "n": n}


def plot(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    for name, m in results.items():
        xs = [r[0] for r in m["reliability"]]
        ys = [r[1] for r in m["reliability"]]
        ax.plot(xs, ys, "o-", label=f"{name} (ECE {m['ece']:.3f})")
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("actual accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-ctx", type=int, default=384)
    ap.add_argument("--limit", type=int, default=2000)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype()
    tok = load_tokenizer(a.run)
    model = MiniJev.load(a.run, device=device)
    vision = load_vision(a.run)
    model.head.temperature.fill_(1.0)

    def load(paths):
        rows = []
        for p in paths:
            rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        return rows[: a.limit]

    vl, vy = raw_logits(model, tok, load(a.val), a.bs, a.max_ctx, device, dtype, vision)
    T = fit_temperature(vl, vy)
    print(f"fitted temperature T = {T:.3f}")

    report, plots = {"temperature": T}, {}
    for p in a.test:
        if not os.path.exists(p):
            print(f"[skip] {p} not found")
            continue
        lg, ys = raw_logits(model, tok, load([p]), a.bs, a.max_ctx, device, dtype, vision)
        before, after = metrics(lg, ys, 1.0), metrics(lg, ys, T)
        name = os.path.basename(p)
        print(f"\n{name} (n={before['n']})")
        print(f"  {'':10} {'acc':>7} {'nll':>7} {'brier':>7} {'ECE':>7}")
        for tag, m in [("raw", before), ("calibrated", after)]:
            print(f"  {tag:10} {m['acc']:7.4f} {m['nll']:7.4f} {m['brier']:7.4f} {m['ece']:7.4f}")
        print("  selective accuracy (calibrated): threshold -> coverage, accuracy")
        for th, (cov, acc) in after["selective"].items():
            print(f"    >= {th:.2f}: {cov:6.1%} of items, acc {acc if acc is None else round(acc, 4)}")
        report[name] = {"raw": before, "calibrated": after}
        plots[name] = after

    model.head.temperature.fill_(T)
    torch.save(model.head.state_dict(), os.path.join(a.run, "head_calibrated.pt"))
    with open(os.path.join(a.run, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    plot(plots, os.path.join(a.run, "reliability.png"))
    print(f"\nsaved temperature into {a.run}/head_calibrated.pt (head.pt left untouched), report -> eval_report.json, plot -> reliability.png")


if __name__ == "__main__":
    main()
