"""Inference: typed questions about one state, with the state encoded once.

    from jev.decide import Decider
    jev = Decider.from_pretrained("AstitwaAgarwal/multimodal-jev")      # or a local run folder
    out = jev.decide(
        state="My package arrived damaged and I want a refund.",
        questions={
            "queue":  {"type": "choice", "instructions": "Which queue?", "criteria": {"billing": "", "shipping": ""}},
            "urgent": {"type": "noul",   "instructions": "Is this urgent?"},
        },
        images=["photo.jpg"],     # optional, 0 to 8 images
    )

How a request runs
  1. the prefix (images + state) goes through the model once and its KV cache is kept;
  2. the questions are answered in chunks: each chunk copies the cache along the batch dimension
     and runs all its question suffixes in one batched pass;
  3. chunk size is chosen so (prefix + suffix) x chunk stays under `token_budget` tokens in flight,
     and halves on CUDA out-of-memory, down to one question; if one still does not fit, the request
     is refused as Unsupported. Nothing is ever truncated.
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict, Optional, Sequence

import torch

from jev.encode import Vision, encode_prefix, encode_suffix, load_tokenizer, load_vision, collate_full
from jev.model import MiniJev, pick_dtype
from jev.schema import Unsupported, answer_from_probs, validate_questions

DEFAULT_MAX_TOKENS = 16384


def _crop_to(cache, P: int) -> None:
    """Keep only the first P positions of every cache layer (works across transformers versions)."""
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            k = getattr(layer, "keys", None)
            if k is not None and k.numel() and k.shape[-2] > P:
                layer.keys, layer.values = layer.keys[..., :P, :], layer.values[..., :P, :]
    else:  # older DynamicCache with key_cache / value_cache lists
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][..., :P, :]
            cache.value_cache[i] = cache.value_cache[i][..., :P, :]


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


class Decider:
    def __init__(self, model: MiniJev, tok, vision: Optional[Vision], device, dtype,
                 max_tokens: int = DEFAULT_MAX_TOKENS, token_budget: int = 32768,
                 cache_prefix: bool = True, max_pixels: Optional[int] = None, name: str = "mini-jev"):
        self.model, self.tok, self.vision = model, tok, vision
        self.device, self.dtype = device, dtype
        self.max_tokens, self.token_budget = max_tokens, token_budget
        self.cache_prefix = cache_prefix
        self.max_pixels = max_pixels
        self.name = name
        self.last_stats: Dict = {}

    @classmethod
    def from_pretrained(cls, run_or_repo: str, device: Optional[str] = None, revision: Optional[str] = None,
                        image_root: str = ".", **kw):
        run = run_or_repo
        if not os.path.isdir(run):
            from huggingface_hub import snapshot_download
            run = snapshot_download(run_or_repo, revision=revision)
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = kw.pop("dtype", None) or (pick_dtype() if dev.type == "cuda" else torch.float32)
        model = MiniJev.load(run, device=dev, dtype=dtype, attn=kw.pop("attn", "sdpa"))
        meta = model.meta
        tok = load_tokenizer(run)
        merged = os.path.join(run, "merged")
        cfg_src = merged if os.path.isdir(merged) else meta["base_model"]
        vision = load_vision(cfg_src, meta.get("max_pixels", 256 * 32 * 32), root=image_root,
                             processor_dir=run) if model.is_vl else None
        return cls(model, tok, vision, dev, dtype, name=meta.get("name", "mini-jev"), **kw)

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def decide(self, state, questions: Dict, images: Sequence = (), max_pixels: Optional[int] = None) -> Dict:
        validate_questions(questions)
        t0 = time.perf_counter()
        prefix = encode_prefix(self.tok, state, images, self.vision, max_pixels or self.max_pixels)
        suffixes = [encode_suffix(self.tok, k, q) for k, q in questions.items()]
        P = len(prefix.ids)
        longest = max(len(s.ids) for s in suffixes)
        if P + longest > self.max_tokens:
            raise Unsupported(f"request needs {P + longest} tokens; maximum context length is {self.max_tokens}")
        if self.cache_prefix and len(suffixes) > 1:  # one question: a single full pass is cheaper
            logits = self._run_cached(prefix, suffixes)
        else:
            logits = self._run_full(prefix, suffixes)
        answers = {}
        for s, lg in zip(suffixes, logits):
            probs = torch.softmax(lg.float(), -1).cpu().tolist()
            q = questions[s.key]
            answers[s.key] = answer_from_probs(q, probs)
        self.last_stats = {"prefix_tokens": P, "questions": len(suffixes),
                           "input_tokens": P + sum(len(s.ids) for s in suffixes),
                           "image_pixels": prefix.image_pixels,
                           "wall_ms": (time.perf_counter() - t0) * 1000}
        return {"model": self.name, "answers": answers, "usage": {"input_tokens": self.last_stats["input_tokens"]}}

    # ------------------------------------------------------------------ internals
    def _autocast(self):
        return torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda")

    def _run_cached(self, prefix, suffixes):
        try:
            with self._autocast():
                cache, P, next_pos = self.model.forward_prefix(prefix, self.device)
        except RuntimeError as e:
            if _is_oom(e):
                self._free()
                raise Unsupported(f"prefix of {len(prefix.ids)} tokens does not fit in device memory "
                                  f"(maximum context length on this device)") from e
            raise
        longest = max(len(s.ids) for s in suffixes)
        chunk = max(1, min(len(suffixes), self.token_budget // (P + longest)))
        out, i = [], 0
        while i < len(suffixes):
            part = suffixes[i:i + chunk]
            try:
                with self._autocast():
                    if len(part) == 1:
                        lg = self.model.forward_suffix(cache, P, next_pos, part, self.tok.pad_token_id, self.device)
                        _crop_to(cache, P)  # drop the suffix tokens; the prefix cache is reused
                    else:
                        c = copy.deepcopy(cache)
                        c.batch_repeat_interleave(len(part))
                        lg = self.model.forward_suffix(c, P, next_pos, part, self.tok.pad_token_id, self.device)
                        del c
            except RuntimeError as e:
                if not _is_oom(e):
                    raise
                self._free()
                _crop_to(cache, P)
                if chunk == 1:
                    raise Unsupported("question does not fit in device memory with this context "
                                      "(maximum context length on this device)") from e
                chunk = max(1, chunk // 2)
                continue
            out += lg
            i += len(part)
        return out

    def _run_full(self, prefix, suffixes):
        """Reference path without the prefix cache (each question re-reads the state)."""
        out = []
        for s in suffixes:
            b = collate_full([(prefix, s)], self.tok.pad_token_id)
            with self._autocast():
                out += self.model.forward_full(b, self.device, [s.qtype])
        return out

    def _free(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
