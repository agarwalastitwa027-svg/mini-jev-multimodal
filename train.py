"""Train mini-Jev: LoRA on the backbone + a fresh answer head, cross-entropy over the options.

Laptop RTX 5070 (8 GB) defaults:
    python train.py --model Qwen/Qwen3-1.7B --train data/train.jsonl data/synthetic_train.jsonl \
        --val data/val.jsonl --out runs/qwen3-1.7b
Quick smoke test (minutes):
    python train.py --model Qwen/Qwen3-0.6B --train data/train.jsonl --val data/val.jsonl \
        --max-steps 200 --out runs/smoke
Bigger backbone in 8 GB:
    python train.py --model Qwen/Qwen3-4B --load-4bit --bs 2 --grad-accum 16 ...
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from jev.format import collate, encode_example
from jev.model import MiniJev, load_tokenizer, load_vision, pick_dtype


def read_jsonl(paths, limit=None):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    if limit:
        random.Random(0).shuffle(rows)
        rows = rows[:limit]
    return rows


def shuffled(ex, rng):
    """Shuffle option order every time (stops the model learning 'answer is usually first')."""
    opts = list(ex["options"])
    rng.shuffle(opts)
    return {**ex, "options": opts}


@torch.no_grad()
def evaluate(model, tok, rows, args, device, dtype, vision=None):
    model.eval()
    correct = nll = 0.0
    for i in range(0, len(rows), args.eval_bs):
        batch = [encode_example(tok, ex, args.max_ctx, vision) for ex in rows[i:i + args.eval_bs]]
        b = collate(batch, tok.pad_token_id)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits = model.run(b, device)
        for lg, y in zip(logits, b["labels"].tolist()):
            correct += int(lg.argmax().item() == y)
            nll += F.cross_entropy(lg[None].float(), torch.tensor([y], device=device)).item()
    model.train()
    return correct / len(rows), nll / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", default="runs/minijev")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="optimizer steps; overrides epochs")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--eval-bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=16, help="0 = freeze backbone, train head only")
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--max-ctx", type=int, default=384, help="max context tokens")
    ap.add_argument("--max-pixels", type=int, default=256 * 28 * 28,
                    help="image resolution cap for vision-language backbones (e.g. Qwen/Qwen3-VL-2B-Instruct)")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pick_dtype()
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), "| dtype:", dtype)
    else:
        print("WARNING: no CUDA GPU found, training on CPU (only for tiny tests)")

    tok = load_tokenizer(a.model)
    train_rows = read_jsonl(a.train, a.limit)
    val_rows = read_jsonl(a.val, a.val_limit)
    print(f"train={len(train_rows)} val={len(val_rows)}")

    vision = load_vision(a.model, a.max_pixels)
    model = MiniJev.create(a.model, load_4bit=a.load_4bit, lora_r=a.lora_r,
                           head_dim=a.head_dim, grad_ckpt=not a.no_grad_ckpt)
    if vision is not None:
        model.meta["max_pixels"] = a.max_pixels
    if not a.load_4bit:
        model.to(device)
    else:
        model.head.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e6:.1f}M")

    head_params = list(model.head.parameters())
    body_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    opt = torch.optim.AdamW([{"params": body_params, "lr": a.lr},
                             {"params": head_params, "lr": a.head_lr}], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_rows) / (a.bs * a.grad_accum))
    total = a.max_steps or max(1, int(steps_per_epoch * a.epochs))
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total), total)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    model.train()
    step, micro, best, t0 = 0, 0, -1.0, time.time()
    run_loss = 0.0
    order = []
    while step < total:
        if not order:
            order = list(range(len(train_rows)))
            rng.shuffle(order)
        idx, order = order[:a.bs], order[a.bs:]
        batch = [encode_example(tok, shuffled(train_rows[i], rng), a.max_ctx, vision) for i in idx]
        b = collate(batch, tok.pad_token_id)
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits = model.run(b, device)
        loss = sum(F.cross_entropy(lg[None].float(), y[None].to(device))
                   for lg, y in zip(logits, b["labels"])) / len(logits)
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
        if step % 10 == 0:
            mem = f" | vram {torch.cuda.max_memory_allocated()/2**30:.1f}GB" if device.type == "cuda" else ""
            print(f"step {step}/{total} | loss {run_loss/(10*a.grad_accum):.4f} | "
                  f"{(time.time()-t0)/step:.2f}s/step{mem}")
            run_loss = 0.0
        if step % a.eval_every == 0 or step == total:
            acc, nll = evaluate(model, tok, val_rows, a, device, dtype, vision)
            print(f"== eval step {step}: val acc {acc:.4f} | val nll {nll:.4f}")
            if acc > best:
                best = acc
                model.save(a.out)
                tok.save_pretrained(a.out)
                print(f"   saved best -> {a.out}")
    print(f"done. best val acc {best:.4f}. Next: python calibrate.py --run {a.out}")


if __name__ == "__main__":
    main()
