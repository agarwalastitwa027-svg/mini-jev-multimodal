"""Everything the evaluation compares, behind one interface.

    predictor.answer(row) -> {question_key: [p_option_1, p_option_2, ...]}   (schema option order)

  MiniJevPredictor    a trained run (local folder or Hub repo), shared-state cached inference
  BaseZeroShot        the untouched backbone: same state/question/options, options lettered A, B, C...,
                      one forward pass, probabilities read from the next-token logits of the letters.
                      This is the fair "is fine-tuning better than the base model?" baseline.
  OllamaPredictor     a generative model with structured output (enum over option keys), e.g. the
                      9B teacher or a same-size model. Returns one-hot probabilities (no confidence).
"""
from __future__ import annotations

import base64
import io
import json
import time
from typing import Dict, List

import torch

from jev.schema import Unsupported, as_text, options_of, predicted_probs

LETTERS = [chr(65 + i) for i in range(26)]


class MiniJevPredictor:
    def __init__(self, run: str, device=None, max_pixels=None, image_root=".", n_perm: int = 1, **kw):
        from jev.decide import Decider
        self.n_perm = n_perm
        self.d = Decider.from_pretrained(run, device=device, image_root=image_root, max_pixels=max_pixels, **kw)
        self.name = run + (f" (n_perm={n_perm})" if n_perm > 1 else "")

    def answer(self, row: Dict) -> Dict[str, List[float]]:
        out = self.d.decide(row.get("state", ""), row["questions"], row.get("images") or [], n_perm=self.n_perm)
        return {k: predicted_probs(row["questions"][k], a) for k, a in out["answers"].items()}

    def sync(self):
        self.d.synchronize()


class BaseZeroShot:
    def __init__(self, model_name: str, device=None, max_pixels: int = 256 * 32 * 32, image_root=".",
                 max_tokens: int = 16384):
        from transformers import AutoConfig
        from jev.encode import load_tokenizer, load_vision
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) \
            if self.device.type == "cuda" else torch.float32
        cfg = AutoConfig.from_pretrained(model_name)
        self.is_vl = hasattr(cfg, "vision_config")
        if self.is_vl:
            from transformers import AutoModelForImageTextToText as Auto
        else:
            from transformers import AutoModelForCausalLM as Auto
        self.model = Auto.from_pretrained(model_name, dtype=self.dtype).to(self.device).eval()
        self.tok = load_tokenizer(model_name)
        self.vision = load_vision(model_name, max_pixels, root=image_root) if self.is_vl else None
        self.max_tokens = max_tokens
        self.name = f"{model_name} (zero-shot)"
        self.letter_ids = []
        for L in LETTERS:
            ids = self.tok(L, add_special_tokens=False)["input_ids"]
            self.letter_ids.append(ids[0])
        self.has_template = bool(getattr(self.tok, "chat_template", None))

    def _ids(self, state, images, instructions, keys, descs):
        lines = "\n".join(f"{LETTERS[i]}. {k}" + (f": {d}" if d and d != k else "")
                          for i, (k, d) in enumerate(zip(keys, descs)))
        text = (f"State:\n{as_text(state).strip()}\n\nQuestion: {as_text(instructions).strip()}\n"
                f"Options:\n{lines}\nAnswer with the option's letter from the given choices directly.")
        marker = "§IMG§"
        if self.has_template:
            s = self.tok.apply_chat_template([{"role": "user", "content": marker + text}],
                                             tokenize=False, add_generation_prompt=True)
            before, after = s.split(marker)
        else:
            before, after = "", text + "\nAnswer:"
        ids = self.tok(before, add_special_tokens=False)["input_ids"]
        kw = {}
        if images:
            n_tok, pv, thw, _ = self.vision.encode(images)
            for n in n_tok:
                ids += [self.vision.start_id] + [self.vision.pad_id] * n + [self.vision.end_id]
            kw = {"pixel_values": pv.to(self.device), "image_grid_thw": thw.to(self.device)}
        ids += self.tok(after, add_special_tokens=False)["input_ids"]
        return ids, kw

    @torch.no_grad()
    def answer(self, row: Dict) -> Dict[str, List[float]]:
        out = {}
        images = row.get("images") or []
        if images and not self.is_vl:
            raise Unsupported("text-only base model cannot read images")
        for k, q in row["questions"].items():
            keys, descs = options_of(q)
            if len(keys) > len(LETTERS):
                raise Unsupported(f"{len(keys)} options exceeds the 26-letter zero-shot format")
            ids, kw = self._ids(row.get("state", ""), images, q.get("instructions", ""), keys, descs)
            if len(ids) > self.max_tokens:
                raise Unsupported(f"maximum context length {self.max_tokens} exceeded")
            t = torch.tensor([ids], device=self.device)
            if kw:
                kw["mm_token_type_ids"] = (t == self.vision.pad_id).int()
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
                logits = self.model(input_ids=t, attention_mask=torch.ones_like(t), **kw).logits[0, -1]
            lg = logits.float()[self.letter_ids[:len(keys)]]
            out[k] = torch.softmax(lg, -1).cpu().tolist()
        return out

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()


class OllamaPredictor:
    def __init__(self, model: str, url: str = "http://localhost:11434", think: bool = False,
                 max_side: int = 768, image_root: str = "."):
        self.model, self.url, self.think, self.max_side = model, url.rstrip("/"), think, max_side
        self.image_root = image_root
        self.name = f"ollama:{model}"
        import requests
        try:  # fail fast (the entry is skipped) if the server is down or the model is not pulled
            tags = requests.get(f"{self.url}/api/tags", timeout=5).json().get("models", [])
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama not reachable at {self.url}") from e
        names = {t.get("name") for t in tags} | {t.get("model") for t in tags}
        if model not in names and f"{model}:latest" not in names:
            raise RuntimeError(f"Ollama model {model!r} not pulled (run: ollama pull {model})")

    def _b64(self, x):
        from jev.encode import load_image
        im = load_image(x, self.image_root)
        im.thumbnail((self.max_side, self.max_side))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def answer(self, row: Dict) -> Dict[str, List[float]]:
        import requests
        out = {}
        imgs = [self._b64(x) for x in (row.get("images") or [])]
        for k, q in row["questions"].items():
            keys, descs = options_of(q)
            opts = "\n".join(f"- {kk}" + (f": {d}" if d and d != kk else "") for kk, d in zip(keys, descs))
            prompt = (f"State:\n{as_text(row.get('state', '')).strip()}\n\nQuestion: {as_text(q.get('instructions', ''))}\n"
                      f"Options:\n{opts}\nAnswer with the key of the single best option.")
            msg = {"role": "user", "content": prompt}
            if imgs:
                msg["images"] = imgs
            schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": keys}}, "required": ["answer"]}
            r = requests.post(f"{self.url}/api/chat", timeout=600, json={
                "model": self.model, "messages": [msg], "format": schema, "stream": False,
                "think": self.think, "options": {"temperature": 0}})
            r.raise_for_status()
            try:
                ans = json.loads(r.json()["message"]["content"])["answer"]
            except (KeyError, ValueError):
                ans = None
            out[k] = [1.0 if kk == ans else 0.0 for kk in keys] if ans in keys else [1.0 / len(keys)] * len(keys)
        return out

    def sync(self):
        pass


def make_predictor(spec: Dict):
    kind = spec["kind"]
    opts = {k: v for k, v in spec.items() if k not in ("kind", "name", "path", "model", "paths")}
    if kind == "minijev":
        return MiniJevPredictor(spec["path"], **opts)
    if kind == "base":
        return BaseZeroShot(spec["model"], **opts)
    if kind == "ollama":
        return OllamaPredictor(spec["model"], **opts)
    raise ValueError(f"unknown predictor kind {kind!r}")


def timed(pred, row):
    pred.sync()
    t = time.perf_counter()
    out = pred.answer(row)
    pred.sync()
    return out, (time.perf_counter() - t) * 1000
