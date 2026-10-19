"""mini-Jev v2: a vision-language backbone with the LM head removed, plus an answer head.

    backbone  = AutoModel (e.g. Qwen3VLModel): transformer layers, NO lm_head, LoRA on the text layers
    q         = W_q * h[last token of the question suffix]   (has read images, state, question, options)
    o_j       = W_o * mean(h[tokens of option j])
    logit_j   = <q, o_j> / sqrt(d) / T[type]                  (T fitted per question type by calibrate.py)
    p         = softmax over the supplied options              (cannot answer outside them)

One head serves choice, noul (options yes/no) and score (ordered levels).
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from jev.schema import TYPE_INDEX, TYPES

RENDER_VERSION = 2


class AnswerHead(nn.Module):
    def __init__(self, hidden: int, d: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.q = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.o = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.d = d
        # one temperature per question type; not trained by gradient, set by calibrate.py
        self.register_buffer("temperature", torch.ones(len(TYPES)))

    def forward(self, hs: torch.Tensor, last_idx, spans, types: Optional[List[str]] = None,
                calibrated: bool = True) -> List[torch.Tensor]:
        """hs: (B, L, hidden). Returns a list of (n_options,) raw logits, one per row."""
        hs = self.norm(hs.float())
        out = []
        for b in range(hs.size(0)):
            h = hs[b]
            q = self.q(h[int(last_idx[b])])
            opt = torch.stack([h[s:e].mean(0) for (s, e) in spans[b]])
            lg = (self.o(opt) @ q) / math.sqrt(self.d)
            if calibrated and types is not None:
                lg = lg / self.temperature[TYPE_INDEX[types[b]]]
            out.append(lg)
        return out


def pick_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def hidden_size(cfg):
    """Text hidden size; vision-language configs nest it under text_config."""
    return getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size


def _4bit_kwargs(dtype):
    from transformers import BitsAndBytesConfig
    return {"quantization_config": BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["visual"]),
        "device_map": {"": 0}}


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class MiniJev(nn.Module):
    def __init__(self, backbone, head: AnswerHead, meta: Dict):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.meta = meta

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(cls, model_name: str, lora_r: int = 16, head_dim: int = 256, grad_ckpt: bool = True,
               load_4bit: bool = False, attn: str = "sdpa", dtype=None):
        from transformers import AutoModel
        dtype = dtype or pick_dtype()
        kw = {"dtype": dtype, "attn_implementation": attn}
        if load_4bit:
            kw.update(_4bit_kwargs(dtype))
        backbone = AutoModel.from_pretrained(model_name, **kw)
        if load_4bit:
            from peft import prepare_model_for_kbit_training
            backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=grad_ckpt)
        elif grad_ckpt:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.config.use_cache = False
        if lora_r > 0:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                             target_modules=LORA_TARGETS, task_type="FEATURE_EXTRACTION")
            backbone = get_peft_model(backbone, cfg)
        else:
            for p in backbone.parameters():
                p.requires_grad_(False)
        head = AnswerHead(hidden_size(backbone.config), head_dim)
        meta = {"base_model": model_name, "lora_r": lora_r, "head_dim": head_dim,
                "load_4bit": load_4bit, "render_version": RENDER_VERSION}
        model = cls(backbone, head, meta)
        for p in model.parameters():  # trainable params in fp32 (stable under bf16/fp16 autocast)
            if p.requires_grad:
                p.data = p.data.float()
        return model

    @classmethod
    def load(cls, run_dir: str, device=None, merge: bool = True, dtype=None, attn: str = "sdpa"):
        from transformers import AutoModel
        meta = json.load(open(os.path.join(run_dir, "meta.json")))
        dtype = dtype or pick_dtype()
        merged_dir = os.path.join(run_dir, "merged")
        kw = {"dtype": dtype, "attn_implementation": attn}
        if os.path.isdir(merged_dir):  # published with merged weights: no base download, no peft
            backbone = AutoModel.from_pretrained(merged_dir, **kw)
        else:
            if meta.get("load_4bit"):
                kw.update(_4bit_kwargs(dtype))
                merge = False
            backbone = AutoModel.from_pretrained(meta["base_model"], **kw)
            if meta["lora_r"] > 0:
                from peft import PeftModel
                backbone = PeftModel.from_pretrained(backbone, os.path.join(run_dir, "lora"))
                if merge:
                    backbone = backbone.merge_and_unload()
        head = AnswerHead(hidden_size(backbone.config), meta["head_dim"])
        cal = os.path.join(run_dir, "head_calibrated.pt")
        state = torch.load(cal if os.path.exists(cal) else os.path.join(run_dir, "head.pt"), map_location="cpu")
        if state.get("temperature") is not None and state["temperature"].numel() == 1:  # v1 head
            state["temperature"] = state["temperature"].repeat(len(TYPES))
        head.load_state_dict(state)
        model = cls(backbone, head, meta)
        model.eval()
        if device is not None:
            if meta.get("load_4bit") and not os.path.isdir(merged_dir):
                model.head.to(device)
            else:
                model.to(device)
        return model

    def save(self, out_dir: str, merged: bool = False):
        os.makedirs(out_dir, exist_ok=True)
        if self.meta["lora_r"] > 0 and hasattr(self.backbone, "save_pretrained"):
            if merged:
                self.backbone.merge_and_unload().save_pretrained(os.path.join(out_dir, "merged"))
            else:
                self.backbone.save_pretrained(os.path.join(out_dir, "lora"))
        torch.save(self.head.state_dict(), os.path.join(out_dir, "head.pt"))
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    # ------------------------------------------------------------------ forward passes
    def core(self):
        """The underlying transformers model (unwraps PEFT)."""
        b = self.backbone
        return b.get_base_model() if hasattr(b, "get_base_model") else b

    @property
    def is_vl(self) -> bool:
        return hasattr(self.core().config, "vision_config")

    def forward_full(self, batch: Dict, device, types: Optional[List[str]] = None, calibrated: bool = True):
        """Whole sequences (prefix + suffix), used for training and as the reference path."""
        kw = {k: batch[k].to(device) for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids") if k in batch}
        out = self.backbone(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device), use_cache=False, **kw)
        return self.head(out.last_hidden_state, batch["last_idx"], batch["spans"], types, calibrated)

    @torch.no_grad()
    def forward_prefix(self, prefix, device):
        """Run the shared prefix once. Returns (cache, prefix_len, next_rope_position)."""
        ids = torch.tensor([prefix.ids], device=device)
        P = ids.shape[1]
        kw, position_ids = {}, None
        if prefix.pixel_values is not None:
            mm = (ids == prefix.image_pad_id).int()
            kw = {"pixel_values": prefix.pixel_values.to(device),
                  "image_grid_thw": prefix.image_grid_thw.to(device), "mm_token_type_ids": mm}
            rope, _ = self.core().get_rope_index(ids, mm_token_type_ids=mm,
                                                 image_grid_thw=prefix.image_grid_thw.to(device))
            text_pos = torch.arange(P, device=device).view(1, 1, P)
            position_ids = torch.cat([text_pos, rope], 0)  # (4, 1, P): text row + 3 M-RoPE rows
            next_pos = int(rope.max().item()) + 1
        else:
            next_pos = P
        from transformers import DynamicCache
        cache = DynamicCache(config=self.core().config)
        out = self.backbone(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=position_ids,
                            past_key_values=cache, use_cache=True, **kw)
        return out.past_key_values, P, next_pos

    @torch.no_grad()
    def forward_suffix(self, cache, P: int, next_pos: int, suffixes: List, pad_id: int, device,
                       calibrated: bool = True):
        """Answer several questions against a cached prefix in one batched pass.

        `cache` must already hold len(suffixes) copies of the prefix along the batch dimension.
        """
        B = len(suffixes)
        L = max(len(s.ids) for s in suffixes)
        ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        smask = torch.zeros((B, L), dtype=torch.long, device=device)
        for i, s in enumerate(suffixes):
            ids[i, :len(s.ids)] = torch.tensor(s.ids, device=device)
            smask[i, :len(s.ids)] = 1
        mask = torch.cat([torch.ones((B, P), dtype=torch.long, device=device), smask], 1)
        ar = torch.arange(L, device=device)
        text_pos = (P + ar).view(1, 1, L).expand(1, B, L)
        rope_pos = (next_pos + ar).view(1, 1, L).expand(3 if self.is_vl else 0, B, L)
        position_ids = torch.cat([text_pos, rope_pos], 0) if self.is_vl else text_pos[0]
        out = self.backbone(input_ids=ids, attention_mask=mask, position_ids=position_ids,
                            past_key_values=cache, use_cache=True)
        last = smask.sum(1) - 1
        return self.head(out.last_hidden_state, last, [s.spans for s in suffixes],
                         [s.qtype for s in suffixes], calibrated)
