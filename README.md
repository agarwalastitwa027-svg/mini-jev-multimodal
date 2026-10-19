# mini-jev-multimodal

A small **text + image decision model** that trains and runs on an 8 GB laptop GPU. Give it a state
(text or JSON, and up to 8 images) and typed questions; it returns a probability for every allowed
option of every question from one forward pass, with no text generation.

An independent, open reproduction of the idea behind System One decision models such as Jev
(TypeSafe AI). It is **not affiliated with or derived from** that product.

**Model:** <https://huggingface.co/AstitwaAgarwal/multimodal-jev> (v2 on `main`, v1 on the `v1` branch)

```
images + state ──► Qwen3-VL-2B (prefill only, LoRA-tuned, LM head removed) ──► KV cache   ← state encoded once
for each question:  "Question / Options / Answer:" ──► hidden states (batched over questions)
        q   = W_q · h[last token]          o_j = W_o · mean(h[option j])
        p_j = softmax_j( q·o_j / √d / T[type] )      T fitted per type: choice / noul / score
```

## What's new in v2

| | v1 | v2 |
|---|---|---|
| Inputs | 1 image + text | text-only or 0–8 images + text, JSON states |
| Questions | each question re-read the image and text | state encoded **once**, questions answered against its KV cache (exactly equal, tested) |
| Types | choice; yes/no and 1–5 as plain choices | `choice`, `noul` (P(yes)), `score` (probabilities + expected value) |
| Loss | hard cross-entropy | soft targets + ranked probability score for ordinal questions |
| Calibration | one temperature | one temperature per question type |
| Long inputs | cut to 384 tokens | never truncated; over the declared limit → `Unsupported` |
| Data | synthetic charts, Food-101, (text incl. BANKING77/CLINC) | balanced text + image mix, **no Decision Index benchmark in training**, decontaminated |
| Evaluation | accuracy vs a 9B teacher | base zero-shot, head-only, v1 recipe, modality ablations, generative baselines; 3 seeds; calibration + latency |
| Serving | Python only | Jev-compatible `POST /v1/systemone` server and a Decision Index engine |

## Quick start

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 50-series needs CUDA 12.8+
pip install -r requirements.txt
python infer.py --run AstitwaAgarwal/multimodal-jev
```

```python
from jev.decide import Decider
jev = Decider.from_pretrained("AstitwaAgarwal/multimodal-jev")
out = jev.decide(
    state="My package arrived damaged and I want a refund.",
    questions={
        "queue":  {"type": "choice", "instructions": "Which queue?",
                   "criteria": {"billing": "payments, refunds", "shipping": "deliveries"}},
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
        "stars":  {"type": "score", "instructions": "How satisfied is the customer?",
                   "criteria": {"1": "very unhappy", "2": "", "3": "", "4": "", "5": "very happy"}},
    },
    images=["photo.jpg"],      # optional
)
# out["answers"]["queue"]  -> {"type": "choice", "choice": "shipping", "probabilities": {...}}
# out["answers"]["urgent"] -> {"type": "noul", "noul": 0.71}
# out["answers"]["stars"]  -> {"type": "score", "choice": "1", "score": 1.4, "probabilities": {...}}
```

Server: `python serve.py --run AstitwaAgarwal/multimodal-jev --port 8000`, then `POST /v1/systemone`
with `{"model", "state", "questions", "images"?}`.

## Train and evaluate on your laptop

Tested target: Samsung Galaxy Book6 Ultra, NVIDIA GeForce RTX 5070 Laptop GPU (8 GB GDDR7, 90 W),
Intel Core Ultra 7 356H, 32 GB RAM, Windows 11. Any 8 GB NVIDIA GPU with CUDA 12.8+ should work.

```bash
python -m venv .venv && .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
nvidia-smi                                                # check the GPU is visible
pytest -q tests                                           # 30 s, tiny model, no downloads

python run_all.py --dry-run                               # see every command first
python run_all.py                                         # data → train → calibrate → ablations → latency → release
python run_all.py --steps publish                         # upload to Hugging Face (hf auth login first)
python run_all.py --steps di_text di_vision --di-kit ../decision-index   # leaderboard runs
```

Rough timings on the RTX 5070 Laptop: data 1–2 h (downloads), ~1.5 h per training run
(3 seeds of v2 + 4 ablation runs ≈ 10–12 h, good overnight), evaluation 1–2 h, full Decision Index
text suite many hours. Every step resumes and skips finished work.

Individual steps:

```bash
python train.py --model Qwen/Qwen3-VL-2B-Instruct --train data/v2/train.jsonl --val data/v2/val.jsonl --out runs/v2_seed0
python calibrate.py --run runs/v2_seed0 --val data/v2/val.jsonl --test data/v2/test_text.jsonl data/v2/test_vision.jsonl
python -m eval.evaluate --kind base --model Qwen/Qwen3-VL-2B-Instruct --test data/v2/test_vision.jsonl
python -m eval.ablations --config eval/ablations.json --out results
python -m eval.latency --run runs/v2_seed0
```

## Comparisons

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md). In short:

| ID | Model | Question it answers |
|---|---|---|
| A | base Qwen3-VL-2B, zero-shot letter probabilities | is fine-tuning better than the base model? |
| B | head only (backbone frozen) | how much does the head alone add? |
| C | v1 recipe (hard cross-entropy) | do soft targets + RPS help? |
| D | v2, 3 seeds | the release model, with seed spread |
| E1/E2 | text-only / image-only training | does mixing modalities help or hurt each? |
| F1/F2 | 9B teacher and same-size 2B, generative via Ollama | accuracy and speed vs generative models |

Metrics: accuracy, chance-adjusted skill, ECE, Brier, NLL, accuracy/coverage at confidence ≥ 0.9, AURC,
p50/p95 latency (1 vs 8 questions per state, cached vs uncached), peak VRAM. Results land in
`results/ablations.md` and `results/latency.md` and are copied into the model card.

## Jev Decision Index

`di_engine.py` plugs this model into the leaderboard kit (text and vision boards), following its
rules: one fixed prompt for every benchmark, every option scored, nothing truncated. How to run and
submit: [docs/SUBMITTING.md](docs/SUBMITTING.md).

## Repository

```
jev/schema.py        question types, validation, answers                jev/encode.py   prompt rendering, image tokens
jev/model.py         backbone + LoRA + answer head, prefix/suffix passes jev/decide.py   cached inference, chunking, limits
jev/losses.py        soft CE, RPS, Brier                                jev/metrics.py  accuracy, skill, ECE, AURC, ...
train.py  calibrate.py  infer.py  serve.py  di_engine.py  publish_hf.py  run_all.py
eval/                predictors (mini-Jev, base zero-shot, Ollama), evaluate, ablations, latency
data/                text_public, vision_public, procedural images, Food-101, convert_v1, build_mix, decontam
tests/               tiny random Qwen3-VL: cached = full, training, server, leaderboard engine
```

## v1 results (for reference)

v1 (Qwen3-VL-2B, synthetic charts + 10 food classes, 244 steps) on an RTX 5070 Laptop: 90.2 % on held-out
procedural images, 96.7 % on unseen Food-101 dishes (4 options), 81.3 % zero-shot on ScienceQA image questions
(teacher qwen3.5:9b: 95.7 %), 86 ms per question vs 1,132 ms for the teacher. Small single-seed test sets.

## Licence

Apache-2.0, see [LICENSE](LICENSE). The base model `Qwen/Qwen3-VL-2B-Instruct` is Apache-2.0. Public datasets
are downloaded by the scripts under their own licences and are not redistributed.
