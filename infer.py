"""Use a trained mini-Jev like Jev: many typed questions about one context, one forward pass.

    python infer.py --run runs/qwen3-1.7b

In code:
    from infer import JevRunner
    jev = JevRunner("runs/qwen3-1.7b")
    jev.decide("My package arrived damaged and I want a refund.", {
        "queue": ["billing", "shipping", "technical", "general"],
        "urgent": bool,
        "refund_requested": bool,
        "sentiment": (1, 5),
    })
"""
import argparse
import time

import torch

from jev.format import collate, encode_example
from jev.model import MiniJev, load_tokenizer, load_vision, pick_dtype

DEFAULT_QUESTIONS = {
    "queue": "Which support queue should handle this?",
    "urgent": "Is this urgent?",
    "refund_requested": "Is the customer asking for a refund?",
    "sentiment": "How satisfied is the customer, from 1 (very unhappy) to 5 (very happy)?",
}


class JevRunner:
    def __init__(self, run_dir, max_ctx=384):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = pick_dtype()
        self.tok = load_tokenizer(run_dir)
        self.model = MiniJev.load(run_dir, device=self.device)
        self.vision = load_vision(run_dir)  # None for text-only runs
        self.max_ctx = max_ctx

    @staticmethod
    def _options(spec):
        if spec is bool:
            return ["yes", "no"], "bool"
        if isinstance(spec, tuple) and len(spec) == 2:
            return [str(i) for i in range(spec[0], spec[1] + 1)], "score"
        return list(spec), "choice"

    @torch.no_grad()
    def decide(self, context, questions, question_text=None, image=None):
        """questions: {name: options-list | bool | (lo, hi)}; question_text: {name: wording};
        image: optional image path (vision-language runs only); context may be "" with an image"""
        question_text = {**DEFAULT_QUESTIONS, **(question_text or {})}
        names, kinds, rows = [], [], []
        for name, spec in questions.items():
            opts, kind = self._options(spec)
            q = question_text.get(name, name.replace("_", " ") + "?")
            rows.append(encode_example(self.tok, {"context": context, "question": q, "image": image,
                                                  "options": opts, "answer": None},
                                       self.max_ctx, self.vision))
            names.append(name)
            kinds.append((kind, opts))
        # all questions go through the GPU together as one batch = one parallel pass
        b = collate(rows, self.tok.pad_token_id)
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            logits = self.model.run(b, self.device)
        out = {}
        for name, (kind, opts), lg in zip(names, kinds, logits):
            p = torch.softmax(lg.float(), -1).cpu().tolist()
            k = max(range(len(p)), key=p.__getitem__)
            ans = opts[k]
            if kind == "bool":
                ans = ans == "yes"
            elif kind == "score":
                ans = int(ans)
            out[name] = {"answer": ans, "confidence": round(p[k], 4),
                         "probs": {o: round(x, 4) for o, x in zip(opts, p)}}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--text", default="My package arrived damaged and I want a refund. "
                                      "This is the second time this month!")
    ap.add_argument("--image", default=None, help="image path (vision-language runs)")
    a = ap.parse_args()
    jev = JevRunner(a.run)
    qs = {"queue": ["billing", "shipping", "technical", "general"], "urgent": bool,
          "refund_requested": bool, "sentiment": (1, 5)}
    jev.decide(a.text, qs, image=a.image)  # warm-up (CUDA kernels, caches)
    if jev.device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    res = jev.decide(a.text, qs, image=a.image)
    if jev.device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) * 1000
    print(f"\nInput: {a.text}\n")
    for k, v in res.items():
        print(f"  {k:18} -> {str(v['answer']):10} conf {v['confidence']:.2f}   {v['probs']}")
    print(f"\n{len(qs)} decisions in {ms:.0f} ms (one batched forward pass)")


if __name__ == "__main__":
    main()
