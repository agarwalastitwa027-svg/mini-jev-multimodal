# mini-jev-multimodal

A small **text + image decision model** that trains and runs on an 8 GB laptop GPU. Give it a state (text or
JSON, and up to 8 images) and typed questions; it returns a probability for every allowed option of every
question from one forward pass, with no text generation.

An independent, open reproduction of the idea behind System One decision models such as Jev (TypeSafe AI).
It is **not affiliated with or derived from** that product.

- **Model:** <https://huggingface.co/AstitwaAgarwal/multimodal-jev> (LoRA adapters + answer head, Apache-2.0)
- **Raw Decision Index run (unofficial):** <https://huggingface.co/datasets/AstitwaAgarwal/decision-index-results>

```
images + state ──► Qwen3-VL-2B (prefill only, LoRA-tuned, LM head removed) ──► KV cache   ← state encoded once
for each question:  "Question / Options / Answer:" ──► hidden states (batched over questions)
        q   = W_q · h[last token]          o_j = W_o · mean(h[option j])
        p_j = softmax_j( q·o_j / √d / T[type] )      T fitted per type: choice / noul / score
```

## Base model vs mini-Jev (the headline result)

The **base model** is `Qwen/Qwen3-VL-2B-Instruct` with no fine-tuning, scored zero-shot by reading the
probabilities of the option letters. **mini-Jev** is that same model after LoRA fine-tuning plus the answer
head and per-type calibration. Both were tested on the **same 3,678 questions from 16 open datasets that
appear nowhere in training** (11 text, 5 image). "Hard-label" is the same recipe trained with plain
cross-entropy instead of soft targets. Accuracy in %; ECE (calibration error) and NLL (log-loss): lower is better.

| Dataset | n | Chance | Base acc | Hard-label acc | **mini-Jev acc** | Base ECE | mini-Jev ECE | Base NLL | mini-Jev NLL |
|---|---|---|---|---|---|---|---|---|---|
| beans | 128 | 33 | 58.6 | 60.2 | **55.5** | 0.394 | 0.222 | 2.38 | 0.88 |
| cifar10 | 200 | 10 | 96.0 | 96.5 | **98.0** | 0.026 | 0.064 | 0.10 | 0.13 |
| cifar100_super | 200 | 5 | 62.0 | 64.5 | **65.5** | 0.311 | 0.077 | 2.29 | 0.97 |
| fashion_mnist | 200 | 10 | 68.5 | 62.5 | **61.0** | 0.217 | 0.111 | 1.58 | 0.95 |
| fin_news | 250 | 33 | 42.4 | 69.2 | **72.8** | 0.528 | 0.053 | 5.19 | 0.60 |
| imdb | 250 | 50 | 94.4 | 93.6 | **93.2** | 0.055 | 0.041 | 0.53 | 0.13 |
| mnist | 200 | 10 | 95.0 | 95.5 | **94.5** | 0.026 | 0.036 | 0.31 | 0.21 |
| mrpc | 250 | 50 | 73.2 | 74.0 | **74.0** | 0.262 | 0.154 | 2.38 | 0.73 |
| newsgroups20 | 250 | 5 | 56.0 | 44.8 | **44.8** | 0.364 | 0.091 | 3.69 | 1.64 |
| qasc | 250 | 12 | 73.2 | 72.8 | **74.8** | 0.195 | 0.102 | 1.86 | 0.82 |
| qnli | 250 | 50 | 83.2 | 75.2 | **79.2** | 0.164 | 0.080 | 1.34 | 0.49 |
| rte | 250 | 50 | 70.0 | 82.4 | **82.8** | 0.269 | 0.117 | 2.14 | 0.57 |
| sciq | 250 | 25 | 96.8 | 96.8 | **96.0** | 0.029 | 0.019 | 0.37 | 0.08 |
| snli | 250 | 33 | 68.8 | 89.6 | **88.0** | 0.200 | 0.071 | 1.21 | 0.31 |
| tweet_hate | 250 | 50 | 60.4 | 65.2 | **66.8** | 0.326 | 0.108 | 1.91 | 0.61 |
| tweet_offensive | 250 | 50 | 40.0 | 75.6 | **74.4** | 0.562 | 0.053 | 4.74 | 0.50 |
| **ALL** | 3678 | 31 | 71.1 | 76.5 | **76.8** | 0.240 | 0.037 | 2.04 | 0.59 |
| **text** | 2750 | 37 | 68.9 | 76.3 | **77.0** | 0.265 | 0.045 | 2.31 | 0.59 |
| **vision** | 928 | 12 | 77.4 | 77.0 | **76.4** | 0.171 | 0.050 | 1.25 | 0.61 |

What this shows, honestly:
- **Accuracy:** 76.8 % overall against 71.1 % for the base model (text 77.0 vs 68.9). mini-Jev wins on 9 of 16
  datasets, mostly where the base model's letter readout is weak: offensive tweets (+34 points), financial news
  sentiment (+30), SNLI (+19, partly because MNLI was in training), RTE (+13). It is slightly worse on 20-way
  newsgroups, Fashion-MNIST and bean-leaf images, and level on SciQ, IMDB and MNIST. On images overall it is on
  par with the base model (76.4 vs 77.4).
- **Calibration is the clearest gain:** overall ECE falls from 0.240 to 0.037 and NLL from 2.04 to 0.59, so the
  probabilities can be used as confidences. A base model that is "90 % sure" is often wrong far more than 10 %
  of the time; here it is not.
- **Soft vs hard targets:** tied (76.8 vs 76.5). The benefit comes from fine-tuning and the calibrated head.
- Single seed, 128–250 questions per dataset: differences of a few points are within noise.

Earlier held-out check (two further new datasets plus the three sources withheld from training; base vs
mini-Jev, same questions):

| Dataset | n | Base acc | mini-Jev acc | Base ECE | mini-Jev ECE | Base NLL | mini-Jev NLL |
|---|---|---|---|---|---|---|---|
| ARC-Challenge | 300 | 76.7 | 78.0 | 0.196 | 0.047 | 1.62 | 0.64 |
| tweet_eval irony | 300 | 45.7 | 60.3 | 0.524 | 0.147 | 4.90 | 0.69 |
| emotion | 300 | 52.3 | 54.0 | 0.420 | 0.149 | 4.07 | 1.24 |
| OpenBookQA | 300 | 72.3 | 69.3 | 0.221 | 0.054 | 1.58 | 0.88 |
| Food-101 unseen dishes | 120 | 98.3 | 96.7 | 0.016 | 0.025 | 0.06 | 0.08 |

On the in-distribution test sets (sources seen in training), 3-seed means: text **87.8 %** vs base 79.6 %,
images **83.5 %** vs 80.6 %, ECE on text 0.015 vs 0.177. The base model is as good on ScienceQA test images
(90.3 % vs 89.9 %); there fine-tuning adds trustworthy confidence, not accuracy. Full ablation table (base,
head-only, hard-label, text-only, image-only, three seeds): [results/ablations.md](results/ablations.md).
Method and exact test sets: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## What it can and cannot do

**Can**
- Pick one of 2–255 options for a state of text and/or up to 8 images, with a calibrated probability for each.
- Answer yes/no questions (`noul`, returns P(yes)) and ordered scores (`score`, probabilities plus expected value).
- Answer several questions about the same state at once; the state is encoded once and reused.
- Run fast: about 74 ms per question on an RTX 5070 Laptop (p50 across the validation mix), median 61 ms across
  150k Decision Index requests. Peak 6.2 GB VRAM in training; it runs within 8 GB (6.7-7.9 GB seen while serving the longest benchmark requests).
- Run text-only (no image) or image + text.

**Cannot**
- Write free text, captions or explanations, or answer outside the options you supply.
- Handle audio or video.
- Replace a general model: it is a 2B model. It is weak on chess, music, routing, expert-level science and
  long-context retrieval (see the Decision Index section below).
- Take requests above the declared limit (16,384 tokens by default, 255 options, 8 images): those return
  `Unsupported`. Nothing is silently truncated.

## Quick start

Needs Python 3.10+ and an NVIDIA GPU. RTX 50-series cards need a PyTorch build for CUDA 12.8 or newer.

```bash
git clone https://github.com/agarwalastitwa027-svg/mini-jev-multimodal.git
cd mini-jev-multimodal
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python infer.py --run AstitwaAgarwal/multimodal-jev    # smoke test: 4 decisions with probabilities
```

The LoRA adapters and answer head download from the Hugging Face repo on first use, and the base model
`Qwen/Qwen3-VL-2B-Instruct` is fetched from the Hub too (about 4.5 GB). No Hugging Face login is needed to run it.

### From Python

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
    images=["photo.jpg"],      # optional: paths or PIL images, up to 8; omit for text-only
)
# out["answers"]["queue"]  -> {"type": "choice", "choice": "shipping", "probabilities": {...}}
# out["answers"]["urgent"] -> {"type": "noul", "noul": 0.71}
# out["answers"]["stars"]  -> {"type": "score", "choice": "1", "score": 1.4, "probabilities": {...}}
```

Question types:

| Type | Options | You get back |
|---|---|---|
| `choice` | 2–255 keys in `criteria` (key → short description) | probability per key and the top `choice` |
| `noul` | yes / no | `noul` = P(yes) |
| `score` | ordered levels (the `criteria` keys, in order) | probability per level, top `choice`, expected `score` |

The state can be plain text or a JSON-serialisable object. Put what the model should know in `state`
and the rules for each option in `criteria`; the more concrete the descriptions, the better the answers.

### As an HTTP server (Jev-compatible)

```bash
python serve.py --run AstitwaAgarwal/multimodal-jev --port 8000
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "model": "mini-jev", "state": "My package arrived damaged.",
  "questions": {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}}'
```

`POST /v1/systemone` takes `{"model", "state", "questions", "images"?}` and returns one answer per question.

## Train and evaluate on your laptop

Tested on: Samsung Galaxy Book6 Ultra, NVIDIA GeForce RTX 5070 Laptop GPU (8 GB, 90 W), Core Ultra 7 356H,
32 GB RAM, Windows 11 and Ubuntu on WSL2. Any 8 GB NVIDIA GPU with CUDA 12.8+ should work.

```bash
pytest -q tests                      # 30 s, tiny random model, no downloads
python run_all.py --dry-run          # print every command first
python run_all.py                    # data → train (3 seeds + 4 comparison runs) → calibrate → ablations → latency → release
python run_all.py --steps publish    # upload to Hugging Face (hf auth login first)
```

Rough timings on the RTX 5070 Laptop: data 1–2 h (downloads), about 1.5 h per training run (the released model
trained 1,211 steps in 93 minutes at 6.2 GB peak), evaluation 1–2 h. Every step skips finished work, so you can
stop and resume. Single steps:

```bash
python train.py --model Qwen/Qwen3-VL-2B-Instruct --train data/mix/train.jsonl --val data/mix/val.jsonl --out runs/full_seed0
python calibrate.py --run runs/full_seed0 --val data/mix/val.jsonl --test data/mix/test_text.jsonl data/mix/test_vision.jsonl
python -m eval.evaluate --kind base --model Qwen/Qwen3-VL-2B-Instruct --test data/mix/test_vision.jsonl   # the base model
python -m eval.evaluate --kind minijev --path runs/full_seed0 --test data/mix/test_vision.jsonl           # mini-Jev
python -m eval.ablations --config eval/ablations.json --out results
python -m eval.latency --run runs/full_seed0
python data/make_more_tests.py data/mix/test_more.jsonl     # the 16-dataset wider test used above
```

Training data (14,997 rows, 19,643 questions): AG News, DBpedia-14, Yelp, BoolQ, MNLI, SST-2, CommonsenseQA,
ScienceQA, A-OKVQA, ChartQA, procedural charts and shapes, Food-101 seen classes. Emotion, OpenBookQA and
Food-101 held-out classes are never trained on. No Decision Index benchmark is used for training; 0 of the
14,997 training rows overlapped the Decision Index suites by 13-gram text overlap (image hashes were not
checked). Details in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Results summary

| Where | What | Result |
|---|---|---|
| [results/more_datasets.md](results/more_datasets.md) | 16 unseen open datasets, base vs hard-label vs mini-Jev | 76.8 % vs 71.1 % base; ECE 0.037 vs 0.240 |
| [results/base_vs_finetuned.md](results/base_vs_finetuned.md) | 5 held-out sets, base vs mini-Jev | better calibrated on 4 of 5; accuracy +14.6 on irony |
| [results/ablations.md](results/ablations.md) | base, head-only, hard-label, text-only, image-only, 3 seeds | in-distribution text 87.8 %, images 83.5 % |
| [results/latency.md](results/latency.md) | cached vs uncached latency and VRAM | state encoded once; see table |
| [results/decision_index_text_unofficial.md](results/decision_index_text_unofficial.md) | Decision Index text board, local rebuild | index 17.4 (unofficial) |

### Jev Decision Index (text board) — unofficial

Run through the kit at github.com/apolinario/decision-index (edition 0.2.1): 149,432 requests, all answered,
**Decision Index 17.43** (chance-corrected, 0 = random guessing), raw 36.8 %, median latency 61 ms. Strongest
area: Tools & Automation (skill 25.5); weakest: Arts & Human Taste (7.9). **This is not a leaderboard entry:** the
official suite is not downloadable, so it was rebuilt from public sources and the core rows are not byte-identical,
and ToolRet-retrieval and BRIGHT-retrieval were not run (they cannot finish on an 8 GB GPU, so they count as wrong).
Per-benchmark scores: [results/decision_index_text_scores_unofficial.json](results/decision_index_text_scores_unofficial.json).
The vision board was not run. How to run it: [docs/SUBMITTING.md](docs/SUBMITTING.md).

## Repository

```
jev/schema.py        question types, validation, answers                jev/encode.py   prompt rendering, image tokens
jev/model.py         backbone + LoRA + answer head, prefix/suffix passes jev/decide.py   cached inference, chunking, limits
jev/losses.py        soft CE, RPS, Brier (and the hard-label ablation)   jev/metrics.py  accuracy, skill, ECE, AURC, ...
train.py  calibrate.py  infer.py  serve.py  di_engine.py  publish_hf.py  run_all.py
eval/                predictors (mini-Jev, base zero-shot, Ollama), evaluate, ablations, latency
data/                text_public, vision_public, procedural images, Food-101, convert_legacy, build_mix, decontam,
                     make_more_tests (the 16-dataset wider test)
docs/                METHODOLOGY.md (design and comparisons), SUBMITTING.md (publishing and the leaderboard)
results/             every table and score quoted above
tests/               tiny random Qwen3-VL: cached = full, training, server, leaderboard engine
```

## Licence

Apache-2.0, see [LICENSE](LICENSE). The base model `Qwen/Qwen3-VL-2B-Instruct` is Apache-2.0. Public datasets are
downloaded by the scripts under their own licences and are not redistributed.
