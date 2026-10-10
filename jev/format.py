"""Prompt format + tokenization for mini-Jev.

Every decision is one example:
    {"context": str, "question": str, "options": [str, ...], "answer": str}

Bool questions use options ["yes", "no"]; score questions use ["1","2","3","4","5"].
Everything is a choice, so one head handles all three Jev output types.

We render:

    Context:
    <context>

    Question: <question>
    Options:
    - <opt 1>
    - <opt 2>
    ...
    Answer:

and record the token span of every option plus the index of the final token.
The final token has (causally) seen the context, the question and ALL options,
so it is the "query"; each option span gives that option's representation.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch


class Vision:
    """Turns an image path into Qwen-VL image tokens + pixel tensors (image+text mode).

    Tokens are built by hand (<vision_start> <image_pad>*n <vision_end>) and put in front of the
    normal text prompt, so option token spans only need to be shifted by the prefix length.
    """

    def __init__(self, image_processor, start_id, pad_id, end_id, max_pixels, root="."):
        self.ip, self.start_id, self.pad_id, self.end_id = image_processor, start_id, pad_id, end_id
        self.max_pixels, self.root = max_pixels, root
        self.merge = getattr(image_processor, "merge_size", 2)

    def encode(self, path: str):
        from PIL import Image
        im = Image.open(os.path.join(self.root, path)).convert("RGB")
        out = self.ip(images=[im], return_tensors="pt", max_pixels=self.max_pixels)
        thw = out["image_grid_thw"][0]
        n = int(thw.prod().item()) // (self.merge ** 2)
        ids = [self.start_id] + [self.pad_id] * n + [self.end_id]
        return ids, out["pixel_values"], out["image_grid_thw"]


def build_prompt(context: str, question: str, options: List[str]):
    """Return prompt text and character spans (start, end) of each option."""
    head = f"Context:\n{context.strip()}\n\n" if context.strip() else ""
    parts = [f"{head}Question: {question.strip()}\nOptions:\n"]
    spans = []
    pos = len(parts[0])
    for opt in options:
        opt = str(opt).strip()
        line_prefix = "- "
        start = pos + len(line_prefix)
        end = start + len(opt)
        parts.append(f"{line_prefix}{opt}\n")
        spans.append((start, end))
        pos += len(line_prefix) + len(opt) + 1
    parts.append("Answer:")
    return "".join(parts), spans


def truncate_context(tokenizer, context: str, max_tokens: int) -> str:
    ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return context
    # keep the head and the tail of long contexts
    head = max_tokens * 2 // 3
    tail = max_tokens - head
    return (tokenizer.decode(ids[:head]) + " ... " + tokenizer.decode(ids[-tail:]))


def encode_example(tokenizer, ex: Dict, max_ctx_tokens: int = 384,
                   vision: Optional[Vision] = None) -> Dict:
    context = truncate_context(tokenizer, ex.get("context", ""), max_ctx_tokens)
    text, char_spans = build_prompt(context, ex["question"], ex["options"])
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    tok_spans = []
    for (cs, ce) in char_spans:
        idx = [i for i, (s, e) in enumerate(offsets) if e > cs and s < ce]
        if not idx:  # extremely short option merged into a neighbour token
            idx = [min(range(len(offsets)), key=lambda i: abs(offsets[i][0] - cs))]
        tok_spans.append((idx[0], idx[-1] + 1))
    input_ids, extra = enc["input_ids"], {}
    if ex.get("image"):
        if vision is None:
            raise ValueError("example has an image but this model/run has no vision support")
        img_ids, pv, thw = vision.encode(ex["image"])
        input_ids = img_ids + input_ids
        tok_spans = [(a + len(img_ids), b + len(img_ids)) for a, b in tok_spans]
        extra = {"pixel_values": pv, "image_grid_thw": thw, "image_pad_id": vision.pad_id}
    label = -100
    if ex.get("answer") is not None:
        opts = [str(o).strip().lower() for o in ex["options"]]
        label = opts.index(str(ex["answer"]).strip().lower())
    return {"input_ids": input_ids, "spans": tok_spans, "label": label, **extra}


def collate(batch: List[Dict], pad_id: int) -> Dict:
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):  # right padding
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        mask[i, :n] = 1
    out = {
        "input_ids": ids,
        "attention_mask": mask,
        "last_idx": mask.sum(1) - 1,
        "spans": [b["spans"] for b in batch],
        "labels": torch.tensor([b["label"] for b in batch]),
    }
    with_img = [b for b in batch if "pixel_values" in b]
    if with_img:  # Qwen-VL layout: patches of all images concatenated, one grid row per image
        out["pixel_values"] = torch.cat([b["pixel_values"] for b in with_img])
        out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in with_img])
        out["mm_token_type_ids"] = (ids == with_img[0]["image_pad_id"]).long()
    return out
