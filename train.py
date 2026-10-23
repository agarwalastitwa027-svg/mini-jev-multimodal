"""Train mini-Jev: LoRA on the backbone + a fresh answer head, typed losses over the options.

8 GB laptop GPU (RTX 5070 Laptop), the default recipe:
    python train.py --model Qwen/Qwen3-VL-2B-Instruct \
        --train data/mix/train.jsonl --val data/mix/val.jsonl --out runs/full_seed0 --seed 0

Ablations (same data, see docs/METHODOLOGY.md):
    --lora-r 0                 head only, backbone frozen            (B)
    --loss hard                hard-label recipe: hard cross-entropy only    (C)
    --modalities text          text-only training                    (E)
    --modalities vision        vision-only training                  (E)
    --head-type letters        LM-head letter readout instead of the answer head   (F)

CPU smoke test (seconds):
    python tests/tiny_model.py runs/tiny-base
    python train.py --model runs/tiny-base --train tests/data/tiny.jsonl --val tests/data/tiny.jsonl \
        --out runs/smoke --max-steps 4 --bs 2 --grad-accum 1 --no-grad-ckpt
"""
import argparse
import json
import math
import os
import random
import time
from collections import OrderedDict, defaultdict

import torch
from transformers import get_cosine_schedule_with_warmup

from jev.data import balanced_order, flatten, read_jsonl, shuffled_order
from jev.encode import collate_full, encode_prefix, encode_suffix, letter_token_ids, load_tokenizer, load_vision
from jev.losses import decision_loss
from jev.model import MiniJev, pick_dtype


class PrefixCache:
    """Small LRU of encoded prefixes (images are the expensive part); rows have several questions."""

    def __init__(self, tok, vision, size=256):
        self.tok, self.vision, self.size, self.d = tok, vision, size, OrderedDict()

    def get(self, i, row):
        if i in self.d:
            self.d.move_to_end(i)
            return self.d[i]
        p = encode_prefix(self.tok, row.get("state", ""), row.get("images") or [], self.vision)
        self.d[i] = p
        if len(self.d) > self.size:
            self.d.popitem(last=False)
        return p


def encode_batch(exs, rows, cache, tok, rng=None, style="options"):
    items, targets, types = [], [], []
    for e in exs:
        q = rows[e.row]["questions"][e.key]
        order = shuffled_order(q, rng) if rng is not None else None
        suf = encode_suffix(tok, e.key, q, order, style)
        items.append((cache.get(e.row, rows[e.row]), suf))
        targets.append([e.target[j] for j in order] if order is not None else e.target)
        types.append(e.qtype)
    return collate_full(items, tok.pad_token_id), targets, types


@torch.no_grad()
def evaluate(model, rows, exs, cache, tok, a, device, dtype):
    style = a.head_type
    model.eval()
    stats = defaultdict(lambda: [0, 0.0, 0])  # modality -> correct, nll, n
    for i in range(0, len(exs), a.eval_bs):
        part = exs[i:i + a.eval_bs]
        b, targets, types = encode_batch(part, rows, cache, tok, None, style)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits = model.forward_full(b, device, types, calibrated=False)
        for e, lg, t in zip(part, logits, targets):
            p = torch.softmax(lg.float(), -1)
            y = max(range(len(t)), key=t.__getitem__)
            for m in (e.modality, "all"):
                s = stats[m]
                s[0] += int(int(p.argmax()) == y)
                s[1] -= float(torch.log(p[y] + 1e-12))
                s[2] += 1
    model.train()
    return {m: {"acc": c / n, "nll": l / n, "n": n} for m, (c, l, n) in stats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", default="runs/model")
    ap.add_argument("--name", default="mini-jev-multimodal")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=0, help="optimizer steps; overrides --epochs")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=16, help="0 = freeze the backbone, train the head only")
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--head-type", choices=["options", "letters"], default="options",
                    help="options: learned answer head over option text; letters: LM-head letter readout (baseline)")
    ap.add_argument("--loss", choices=["soft", "hard"], default="soft")
    ap.add_argument("--rps-weight", type=float, default=1.0)
    ap.add_argument("--brier-weight", type=float, default=0.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--modalities", nargs="*", default=None, help="text and/or vision (default: both)")
    ap.add_argument("--text-frac", type=float, default=0.5, help="share of steps on text; -1 = natural mix")
    ap.add_argument("--max-train-tokens", type=int, default=2048,
                    help="skip (never truncate) training examples longer than this")
    ap.add_argument("--max-pixels", type=int, default=256 * 32 * 32,
                    help="per-image pixel cap during training (inference can raise it)")
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--select", choices=["acc", "nll"], default="nll",
                    help="checkpoint selection on validation accuracy or log-loss")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    rng = random.Random(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype() if device.type == "cuda" else torch.float32
    print("device:", torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "| dtype:", dtype)

    tok = load_tokenizer(a.model)
    vision = load_vision(a.model, a.max_pixels, root=a.image_root)
    rows = read_jsonl(a.train, a.limit, a.seed)
    vrows = read_jsonl(a.val, None)
    if vision is None:
        rows = [r for r in rows if not r.get("images")]
        vrows = [r for r in vrows if not r.get("images")]
        print("text-only backbone: rows with images dropped")
    exs = flatten(rows, a.modalities)
    vexs = flatten(vrows, a.modalities)
    random.Random(0).shuffle(vexs)
    vexs = vexs[: a.val_limit]
    cache, vcache = PrefixCache(tok, vision), PrefixCache(tok, vision)

    # drop over-long training examples instead of truncating them
    keep = []
    for e in exs:
        r = rows[e.row]
        n = len(cache.get(e.row, r).ids) + len(encode_suffix(tok, e.key, r["questions"][e.key], None, a.head_type).ids)
        if n <= a.max_train_tokens:
            keep.append(e)
    print(f"train questions={len(keep)} (skipped {len(exs) - len(keep)} longer than {a.max_train_tokens} tokens) "
          f"| val questions={len(vexs)}")
    by_mod = defaultdict(int)
    for e in keep:
        by_mod[e.modality] += 1
    print("train mix:", dict(by_mod))
    exs = keep

    model = MiniJev.create(a.model, lora_r=a.lora_r, head_dim=a.head_dim, grad_ckpt=not a.no_grad_ckpt,
                           load_4bit=a.load_4bit, dtype=dtype if device.type == "cuda" else torch.float32,
                           head_type=a.head_type,
                           letter_ids=letter_token_ids(tok) if a.head_type == "letters" else None)
    model.meta.update({"name": a.name, "max_pixels": a.max_pixels, "modalities": a.modalities or ["text", "vision"],
                       "train_args": vars(a)})
    if a.load_4bit:
        model.head.to(device)
    else:
        model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.2f}M")

    head_params = list(model.head.parameters())
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    groups = ([{"params": head_params, "lr": a.head_lr}] if head_params else []) + \
        ([{"params": body, "lr": a.lr}] if body else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(exs) / (a.bs * a.grad_accum))
    total = a.max_steps or max(1, int(steps_per_epoch * a.epochs))
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total), total)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))
    text_frac = None if a.text_frac < 0 else a.text_frac

    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "w")
    model.train()
    step, micro, t0, run_loss, order = 0, 0, time.time(), 0.0, []
    best = None
    while step < total:
        if len(order) < a.bs:
            order += balanced_order(exs, text_frac, rng)
        idx, order = order[:a.bs], order[a.bs:]
        b, targets, types = encode_batch([exs[i] for i in idx], rows, cache, tok, rng, a.head_type)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits = model.forward_full(b, device, types, calibrated=False)
        loss = decision_loss(logits, targets, types, a.loss, a.rps_weight, a.brier_weight, a.label_smoothing)
        scaler.scale(loss / a.grad_accum).backward()
        run_loss += loss.item()
        micro += 1
        if micro % a.grad_accum:
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        if step % 10 == 0 or step == total:
            mem = f" | vram {torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if device.type == "cuda" else ""
            k = 10 if step % 10 == 0 else (step % 10)
            print(f"step {step}/{total} | loss {run_loss / (k * a.grad_accum):.4f} | "
                  f"{(time.time() - t0) / step:.2f}s/step{mem}", flush=True)
            run_loss = 0.0
        if step % a.eval_every == 0 or step == total:
            ev = evaluate(model, vrows, vexs, vcache, tok, a, device, dtype)
            print(f"== eval step {step}: " + " | ".join(
                f"{m} acc {v['acc']:.4f} nll {v['nll']:.4f} (n={v['n']})" for m, v in sorted(ev.items())), flush=True)
            log.write(json.dumps({"step": step, "eval": ev}) + "\n")
            log.flush()
            score = ev["all"]["acc"] if a.select == "acc" else -ev["all"]["nll"]
            if best is None or score > best:
                best = score
                model.meta["best_step"], model.meta["best_val"] = step, ev
                model.save(a.out)
                tok.save_pretrained(a.out)
                if vision is not None:
                    vision.ip.save_pretrained(a.out)
                print(f"   saved best -> {a.out}")
    log.close()
    meta_path = os.path.join(a.out, "meta.json")  # the best checkpoint's meta; add timing info
    meta = json.load(open(meta_path))
    meta["train_minutes"] = round((time.time() - t0) / 60, 1)
    meta["total_steps"] = total
    if device.type == "cuda":
        meta["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        meta["gpu"] = torch.cuda.get_device_name(0)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done. Next: python calibrate.py --run {a.out} --val {' '.join(a.val)} --test <test files>")


if __name__ == "__main__":
    main()
