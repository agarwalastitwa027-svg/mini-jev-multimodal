"""Engine for the Jev Decision Index kit (github.com/apolinario/decision-index).

Text board (from inside a clone of the kit, with this repo on PYTHONPATH):
    python -m decision_index pipeline --engine di_engine:MiniJevEngine \
        --option run=AstitwaAgarwal/multimodal-jev --out runs/mini-jev-multimodal

Vision board:
    decision-index vision pipeline --dir work/vision --engine di_engine:MiniJevEngine \
        --option run=AstitwaAgarwal/multimodal-jev --out runs/vision-mini-jev-multimodal

Options (all optional except run): run, revision, device, max_tokens (declared context limit,
default 16384), token_budget, max_pixels (default 1,638,400: the board's native-resolution cap),
cache_prefix (true).

House rules this engine follows: one fixed rendering for every benchmark; every option scored;
nothing truncated: requests over the declared limit, or that do not fit in GPU memory after
splitting the question batch down to one question, raise Unsupported.
"""
from __future__ import annotations

import os
import subprocess

try:
    from decision_index.engines import Engine, Unsupported as DIUnsupported
except ImportError:  # the kit is not installed: minimal stand-ins so this module still imports
    class Engine:  # type: ignore[no-redef]
        name = "engine"
        provenance: dict = {}
        latency = ""

        def __init__(self, **options):
            self.options = options

    class DIUnsupported(ValueError):  # type: ignore[no-redef]
        pass

from jev.schema import Unsupported


def _bool(x, default=True):
    if x is None:
        return default
    return str(x).strip().lower() not in ("0", "false", "no", "off")


def _git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


class MiniJevEngine(Engine):
    name = "mini-jev-multimodal"
    latency = ("In-process request wall time: prompt construction, image preprocessing, one prefix pass "
               "and batched question passes; excludes model loading.")

    def __init__(self, run="AstitwaAgarwal/multimodal-jev", revision=None, device=None, max_tokens=16384,
                 token_budget=32768, max_pixels=1_638_400, cache_prefix=True, **options):
        super().__init__(run=run, revision=revision, device=device, max_tokens=max_tokens,
                         token_budget=token_budget, max_pixels=max_pixels, cache_prefix=cache_prefix, **options)
        from jev.decide import Decider
        self.decider = Decider.from_pretrained(
            run, device=device, revision=revision, max_tokens=int(max_tokens), token_budget=int(token_budget),
            cache_prefix=_bool(cache_prefix), max_pixels=int(max_pixels) if max_pixels not in (None, "null") else None)
        self.provenance = {"repo": run, "revision": revision, "engine_commit": _git_commit(),
                           "base_model": self.decider.model.meta.get("base_model"),
                           "declared_limits": {"max_tokens": int(max_tokens), "max_options": 255, "max_images": 8},
                           "max_pixels": max_pixels}

    def __call__(self, state, questions, images=(), prompt=None, **inputs):
        # prompt.options_in_image / readout (vision board) need no special handling: options are scored
        # by the answer head from their key text; the image carries the option descriptions.
        try:
            out = self.decider.decide(state, questions, list(images or []))
        except Unsupported as e:
            raise DIUnsupported(str(e)) from e
        raw = {"stats": dict(self.decider.last_stats)}
        return out, raw

    def runtime(self):
        import torch
        d = self.decider.device
        return {"device": str(d), "dtype": str(self.decider.dtype),
                "gpu": torch.cuda.get_device_name(0) if d.type == "cuda" else None}

    def synchronize(self):
        self.decider.synchronize()

    def close(self):
        del self.decider
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
