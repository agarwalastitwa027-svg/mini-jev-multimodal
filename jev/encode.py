"""Fixed prompt rendering, tokenization and image handling.

One rendering is used for every request and every benchmark (no per-task prompts).
A request is split into a shared *prefix* (images + state) and one *suffix* per question:

    prefix:  [<vision_start> <image_pad>*n <vision_end> "\\n"] * images
             "State:\\n<state>\\n\\n"
    suffix:  "Question: <instructions>\\nOptions:\\n- <key>: <description>\\n...Answer:"

The prefix and each suffix are tokenized separately and concatenated. Training uses exactly the
same concatenation, so running the prefix once and reusing its KV cache for every question at
inference time gives the same numbers as running each full sequence (see tests/test_tiny.py).
"""
from __future__ import annotations

import base64
import binascii
import io
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from jev.schema import MAX_IMAGES, Unsupported, as_text, options_of


# ----------------------------------------------------------------------------------- images

def load_image(x, root: str = "."):
    """Accept a path, a data: URL, a raw base64 string, bytes, or a PIL image."""
    from PIL import Image
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")
    if isinstance(x, str):
        if x.startswith("data:"):
            return Image.open(io.BytesIO(base64.b64decode(x.split(",", 1)[1]))).convert("RGB")
        path = x if os.path.isabs(x) else os.path.join(root, x)
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
        try:  # raw base64
            return Image.open(io.BytesIO(base64.b64decode(x, validate=True))).convert("RGB")
        except (binascii.Error, ValueError, OSError):
            raise FileNotFoundError(f"image not found and not base64: {x[:80]}")
    raise TypeError(f"unsupported image input {type(x)}")


class Vision:
    """Qwen-VL style image tokens + pixel tensors.

    max_pixels caps each image (aspect ratio kept). Training uses a small cap to save memory;
    the Decision Index vision board asks for native resolution up to 1,638,400 pixels.
    """

    def __init__(self, image_processor, start_id: int, pad_id: int, end_id: int,
                 max_pixels: int = 256 * 32 * 32, root: str = "."):
        self.ip = image_processor
        self.start_id, self.pad_id, self.end_id = start_id, pad_id, end_id
        self.max_pixels, self.root = max_pixels, root
        self.merge = getattr(image_processor, "merge_size", 2)

    def encode(self, images: Sequence, max_pixels: Optional[int] = None):
        ims = [load_image(x, self.root) for x in images]
        cap = max_pixels or self.max_pixels
        size = getattr(self.ip, "size", None)
        try:
            shortest = size["shortest_edge"] if isinstance(size, dict) else getattr(size, "shortest_edge", None)
        except (KeyError, TypeError):
            shortest = None
        if shortest:  # newer processors: size={"shortest_edge", "longest_edge"} (both in pixels)
            out = self.ip(images=ims, return_tensors="pt",
                          size={"shortest_edge": min(int(shortest), cap), "longest_edge": cap})
        else:
            out = self.ip(images=ims, return_tensors="pt", max_pixels=cap)
        thw = out["image_grid_thw"]
        n_tok = [int(t.prod().item()) // (self.merge ** 2) for t in thw]
        return n_tok, out["pixel_values"], thw, [im.size for im in ims]


def load_vision(name_or_dir: str, max_pixels: int, root: str = ".",
                processor_dir: Optional[str] = None) -> Optional[Vision]:
    """Return a Vision helper if the backbone is a vision-language model, else None.

    `name_or_dir` holds the model config; `processor_dir` (a run folder) may hold a saved
    image processor + tokenizer, which then take precedence.
    """
    from transformers import AutoConfig, AutoImageProcessor, AutoTokenizer
    cfg = AutoConfig.from_pretrained(name_or_dir)
    if not hasattr(cfg, "vision_config"):
        return None
    src = processor_dir if processor_dir and os.path.exists(
        os.path.join(processor_dir, "preprocessor_config.json")) else name_or_dir
    ip = AutoImageProcessor.from_pretrained(src)
    tok = AutoTokenizer.from_pretrained(src)
    ids = [tok.convert_tokens_to_ids(t) for t in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")]
    if any(i is None or i == tok.unk_token_id for i in ids):
        raise ValueError("tokenizer lacks Qwen-VL vision tokens")
    return Vision(ip, *ids, max_pixels=max_pixels, root=root)


# ----------------------------------------------------------------------------------- text

def load_tokenizer(name_or_dir: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name_or_dir)
    if not getattr(tok, "is_fast", False):
        raise ValueError("a fast tokenizer is required (offset mapping for option spans)")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


@dataclass
class Prefix:
    ids: List[int]
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    image_pad_id: Optional[int] = None
    image_pixels: List[int] = field(default_factory=list)


@dataclass
class Suffix:
    key: str
    qtype: str
    keys: List[str]          # option keys in the order they were rendered
    ids: List[int]
    spans: List[tuple]       # (start, end) token span of each option, suffix coordinates


def encode_prefix(tok, state, images: Sequence = (), vision: Optional[Vision] = None,
                  max_pixels: Optional[int] = None) -> Prefix:
    ids: List[int] = []
    pv = thw = None
    pixels: List[int] = []
    images = list(images or [])
    if images:
        if vision is None:
            raise Unsupported("images were supplied but this model has no vision support")
        if len(images) > MAX_IMAGES:
            raise Unsupported(f"{len(images)} images exceeds the declared limit of {MAX_IMAGES} images per request")
        n_tok, pv, thw, sizes = vision.encode(images, max_pixels)
        nl = tok("\n", add_special_tokens=False)["input_ids"]
        for n in n_tok:
            ids += [vision.start_id] + [vision.pad_id] * n + [vision.end_id] + nl
        pixels = [int(t[1] * t[2]) * (vision.ip.patch_size ** 2) if hasattr(vision.ip, "patch_size") else 0
                  for t in thw]
    text = f"State:\n{as_text(state).strip()}\n\n"
    ids += tok(text, add_special_tokens=False)["input_ids"]
    return Prefix(ids, pv, thw, vision.pad_id if (images and vision) else None, pixels)


def render_suffix(instructions, keys: List[str], descs: List[str]):
    head = f"Question: {as_text(instructions).strip()}\nOptions:\n"
    parts, spans, pos = [head], [], len(head)
    for k, d in zip(keys, descs):
        k, d = str(k).strip(), str(d).strip()
        body = k if (not d or d == k) else f"{k}: {d}"
        start = pos + 2  # after "- "
        parts.append(f"- {body}\n")
        spans.append((start, start + len(body)))
        pos += 2 + len(body) + 1
    parts.append("Answer:")
    return "".join(parts), spans


def encode_suffix(tok, key: str, q: Dict, order: Optional[List[int]] = None) -> Suffix:
    keys, descs = options_of(q)
    if order is not None:  # option shuffling during training
        keys, descs = [keys[i] for i in order], [descs[i] for i in order]
    text, char_spans = render_suffix(q.get("instructions", ""), keys, descs)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    spans = []
    for cs, ce in char_spans:
        idx = [i for i, (s, e) in enumerate(offs) if e > cs and s < ce]
        if not idx:
            idx = [min(range(len(offs)), key=lambda i: abs(offs[i][0] - cs))]
        spans.append((idx[0], idx[-1] + 1))
    return Suffix(key, q["type"], keys, enc["input_ids"], spans)


# ----------------------------------------------------------------------------------- batching

def collate_full(items: List[tuple], pad_id: int) -> Dict:
    """Training/eval batch of full sequences: items are (Prefix, Suffix) pairs.

    Returns padded ids, attention mask, last-token index, option spans in full-sequence
    coordinates, and the concatenated image tensors for every item that has images.
    """
    seqs = [p.ids + s.ids for p, s in items]
    L = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, x in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    spans = [[(a + len(p.ids), b + len(p.ids)) for a, b in s.spans] for p, s in items]
    out = {"input_ids": ids, "attention_mask": mask, "last_idx": mask.sum(1) - 1, "spans": spans}
    with_img = [p for p, _ in items if p.pixel_values is not None]
    if with_img:
        out["pixel_values"] = torch.cat([p.pixel_values for p in with_img])
        out["image_grid_thw"] = torch.cat([p.image_grid_thw for p in with_img])
        out["mm_token_type_ids"] = (ids == with_img[0].image_pad_id).int()
    return out
