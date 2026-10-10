"""mini-Jev: a decoder LLM with the LM head removed, plus an answer head.

    backbone  = AutoModel (e.g. Qwen3Model)  -> transformer layers, NO lm_head
    one forward pass (prefill only, no decode loop)
    q_i       = W_q * h[last token]                (has seen context + question + options)
    o_j       = W_o * mean(h[option j tokens])
    logit_j   = <q, o_j> / sqrt(d) / T            (T = temperature, fitted in calibrate.py)
    p         = softmax over the allowed options -> can never answer outside the options
"""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class AnswerHead(nn.Module):
    def __init__(self, hidden: int, d: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.q = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.o = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.d = d
        # temperature is NOT trained by gradient; calibrate.py sets it
        self.register_buffer("temperature", torch.tensor(1.0))

    def forward(self, hs, last_idx, spans):
        hs = self.norm(hs.float())
        logits = []
        for b in range(hs.size(0)):
            h = hs[b]
            q = self.q(h[last_idx[b]])                                     # (d,)
            opt = torch.stack([h[s:e].mean(0) for (s, e) in spans[b]])     # (n_opt, hidden)
            o = self.o(opt)                                                # (n_opt, d)
            logits.append((o @ q) / math.sqrt(self.d) / self.temperature)
        return logits  # list of (n_opt,) tensors; option counts differ per example


def pick_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class MiniJev(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    @classmethod
    def create(cls, model_name, load_4bit=False, lora_r=16, head_dim=256, grad_ckpt=True):
        dtype = pick_dtype()
        kwargs = {"dtype": dtype}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["visual"])  # keep the vision tower in 16-bit
            kwargs["device_map"] = {"": 0}
        # AutoModel = the base transformer WITHOUT the LM head ("remove the LM head")
        backbone = AutoModel.from_pretrained(model_name, **kwargs)
        if load_4bit:
            from peft import prepare_model_for_kbit_training
            backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=grad_ckpt)
        elif grad_ckpt:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.config.use_cache = False
        if lora_r > 0:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"],
                             task_type="FEATURE_EXTRACTION")
            backbone = get_peft_model(backbone, cfg)
        head = AnswerHead(hidden_size(backbone.config), head_dim)
        model = cls(backbone, head)
        model.meta = {"base_model": model_name, "lora_r": lora_r, "head_dim": head_dim,
                      "load_4bit": load_4bit}
        # keep every trainable parameter in fp32 (stable with bf16 or fp16 autocast)
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        return model

    def forward(self, input_ids, attention_mask, last_idx, spans, **vision_kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **vision_kwargs)
        return self.head(out.last_hidden_state, last_idx, spans)

    def run(self, b, device):
        """Forward a collated batch (text-only or with images) -> list of per-example logits."""
        extra = {k: b[k].to(device) for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids")
                 if k in b}
        return self(b["input_ids"].to(device), b["attention_mask"].to(device),
                    b["last_idx"].to(device), b["spans"], **extra)

    # ---------- save / load ----------
    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        if self.meta["lora_r"] > 0:
            self.backbone.save_pretrained(os.path.join(out_dir, "lora"))
        torch.save(self.head.state_dict(), os.path.join(out_dir, "head.pt"))
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    @classmethod
    def load(cls, out_dir, device=None, merge=True):
        meta = json.load(open(os.path.join(out_dir, "meta.json")))
        dtype = pick_dtype()
        kwargs = {"dtype": dtype}
        if meta.get("load_4bit"):  # e.g. a 4B backbone: keep it 4-bit so it fits in 8 GB
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["visual"])
            kwargs["device_map"] = {"": 0}
            merge = False
        backbone = AutoModel.from_pretrained(meta["base_model"], **kwargs)
        if meta["lora_r"] > 0:
            from peft import PeftModel
            backbone = PeftModel.from_pretrained(backbone, os.path.join(out_dir, "lora"))
            if merge:
                backbone = backbone.merge_and_unload()  # no LoRA overhead at inference
        head = AnswerHead(hidden_size(backbone.config), meta["head_dim"])
        cal = os.path.join(out_dir, "head_calibrated.pt")  # written by calibrate.py
        head_path = cal if os.path.exists(cal) else os.path.join(out_dir, "head.pt")
        head.load_state_dict(torch.load(head_path, map_location="cpu"))
        model = cls(backbone, head)
        model.meta = meta
        model.eval()
        if device and not meta.get("load_4bit"):
            model.to(device)
        elif device:
            model.head.to(device)
        return model


def hidden_size(cfg):
    """Text hidden size; vision-language configs nest it under text_config."""
    return getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size


def load_vision(name, max_pixels=256 * 28 * 28, root="."):
    """Return a Vision helper if `name` (hub id or run dir) is a vision-language model, else None."""
    from transformers import AutoConfig, AutoProcessor

    from jev.format import Vision
    if os.path.isdir(name):  # a run dir: read the base model from meta.json
        meta = json.load(open(os.path.join(name, "meta.json")))
        name, max_pixels = meta["base_model"], meta.get("max_pixels") or max_pixels
    if not hasattr(AutoConfig.from_pretrained(name), "vision_config"):
        return None
    proc = AutoProcessor.from_pretrained(name)
    tok = proc.tokenizer
    ids = [tok.convert_tokens_to_ids(t) for t in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")]
    return Vision(proc.image_processor, *ids, max_pixels=max_pixels, root=root)


def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok
