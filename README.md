# mini-jev-multimodal

A small **text + image decision model** that trains and runs on an 8 GB laptop GPU. Give it a state (text or
JSON, and up to 8 images) and typed questions; it returns a probability for every allowed option of every
question from one forward pass, with no text generation.

An independent, open reproduction of the idea behind System One decision models such as Jev (TypeSafe AI).
It is **not affiliated with or derived from** that product.

- **Model:** <https://huggingface.co/AstitwaAgarwal/multimodal-jev> (LoRA adapters + answer head, Apache-2.0)
- **Full report:** [docs/REPORT.md](docs/REPORT.md) (what fine-tuning buys, with baselines, seeds and confidence intervals)
- **Raw per-question results of every run:** <https://huggingface.co/datasets/AstitwaAgarwal/decision-index-results>

```
images + state ──► Qwen3-VL-2B (prefill only, LoRA-tuned, LM head removed) ──► KV cache   ← state encoded once
for each question:  "Question / Options / Answer:" ──► hidden states (batched over questions)
        q   = W_q · h[last token]          o_j = W_o · mean(h[option j])
        p_j = softmax_j( q·o_j / √d / T[type] )      T fitted per type: choice / noul / score
```

## Base model vs mini-Jev (the headline result)

The **base model** is `Qwen/Qwen3-VL-2B-Instruct` with no fine-tuning, scored zero-shot by reading the probabilities
of the option letters. Because raw base-model probabilities are badly overconfident, it is also shown **with
post-hoc temperature scaling** (one temperature per question type, fitted on the same validation set), the fair
baseline for calibration. **mini-Jev** is the same model after LoRA fine-tuning on a broad mix of 26k training rows
plus the answer head and per-type calibration. All numbers are on questions from datasets that appear nowhere in
training. Accuracy in %; ECE (calibration error) and NLL (log-loss): lower is better. 95 % intervals are paired
bootstrap intervals over questions.

| Test collection | Questions | Base | Base + temp. scaling | **mini-Jev** | mini-Jev minus base + temp. scaling |
|---|---|---|---|---|---|
| 16 open datasets (`test_more`) | 3,678 | 71.1 | 71.1 | **78.0** | **+6.9 [+5.5, +8.3]** |
| 11 further datasets never used for any decision (`test_confirm`; the two with 43-45 options left out) | 2,306 | 69.2 | 69.2 | **72.0** | **+2.8 [+1.2, +4.4]** |
| ARC-Challenge, tweet irony (`test_fresh`) | 600 | 61.2 | 61.2 | **76.2** | +15.0 [+11.2, +18.8] |
| emotion, OpenBookQA, unseen Food-101 dishes (`test_zeroshot`) | 720 | 68.3 | 68.3 | **70.0** | +1.7 [-1.3, +4.3] (not significant) |
| 27-100 options: CIFAR-100, BANKING77, GoEmotions (`test_manyopt`) | 700 | cannot answer (> 26 options) | cannot answer | **48.4** | n/a |

| Calibration | Base | Base + temp. scaling | **mini-Jev** |
|---|---|---|---|
| ECE on the 16 datasets | 0.240 | 0.068 | **0.043** |
| NLL on the 16 datasets | 2.04 | 0.80 | **0.58** |
| ECE on the confirmation set (<= 26 options) | 0.255 | 0.099 | **0.044** |
| NLL on the confirmation set (<= 26 options) | 2.66 | 0.96 | **0.81** |

What this shows, honestly:
- **Accuracy:** fine-tuning helps, but how much depends on the task. +6.9 points on 16 unseen datasets, +2.8 on 11 more
  that influenced no decision, and no significant gain on three zero-shot sources. The gains are largest on yes / no
  and few-option questions (offensive tweets +36, financial news sentiment +34, SNLI +22, RTE +11 points).
- **Calibration:** temperature scaling alone fixes **75-87 %** of the raw base model's calibration error, so most of
  the calibration improvement is *not* due to fine-tuning. Fine-tuning still has the lowest ECE and NLL.
- **Task breadth is the change that helped most:** three seeds of the broad training mix beat three seeds of the
  original 12-source mix on every held-out collection (+1.4 [+0.8, +2.0] on the 16 datasets, +1.3 [+0.5, +2.1] on the
  confirmation set, +4.6 [+2.8, +6.5] on the fresh sets, +5.1 on many-option questions).
- **Known weakness:** on questions with more than 10 options mini-Jev is *below* the base model (54.2 % vs 58.7 %,
  pooled). Averaging over four option orders (`n_perm=4`, see below) recovers part of it (56.9 %), not all.
- **Head design:** a plain letter-readout head matches the learned answer head on accuracy and generalises better to
  unseen sources, but is calibrated worse and cannot take more than 26 options. See the report.

Per dataset, the 16 open datasets (accuracy %, ECE, NLL; chance is the average of 1 / options):

| Dataset | n | Chance % | Base acc | Base + temp. scaling acc | mini-Jev acc | Base + temp. scaling ECE | mini-Jev ECE | Base + temp. scaling NLL | mini-Jev NLL |
|---|---|---|---|---|---|---|---|---|---|
| beans | 128 | 33 | 58.6 | 58.6 | 61.7 | 0.165 | 0.153 | 0.96 | 0.77 |
| cifar10 | 200 | 10 | 96.0 | 96.0 | 97.0 | 0.272 | 0.045 | 0.42 | 0.11 |
| cifar100_super | 200 | 5 | 62.0 | 62.0 | 68.0 | 0.167 | 0.069 | 1.50 | 1.02 |
| fashion_mnist | 200 | 10 | 68.5 | 68.5 | 67.5 | 0.176 | 0.097 | 1.12 | 0.78 |
| fin_news | 250 | 33 | 42.4 | 42.4 | 76.8 | 0.416 | 0.052 | 1.62 | 0.53 |
| imdb | 250 | 50 | 94.4 | 94.4 | 93.6 | 0.040 | 0.042 | 0.18 | 0.14 |
| mnist | 200 | 10 | 95.0 | 95.0 | 95.5 | 0.227 | 0.037 | 0.46 | 0.18 |
| mrpc | 250 | 50 | 73.2 | 73.2 | 77.2 | 0.114 | 0.139 | 0.59 | 0.69 |
| newsgroups20 | 250 | 5 | 56.0 | 56.0 | 43.2 | 0.092 | 0.125 | 1.59 | 1.65 |
| qasc | 250 | 12 | 73.2 | 73.2 | 74.0 | 0.059 | 0.079 | 0.87 | 0.81 |
| qnli | 250 | 50 | 83.2 | 83.2 | 75.2 | 0.041 | 0.146 | 0.41 | 0.67 |
| rte | 250 | 50 | 70.0 | 70.0 | 80.8 | 0.104 | 0.132 | 0.59 | 0.56 |
| sciq | 250 | 25 | 96.8 | 96.8 | 97.2 | 0.017 | 0.012 | 0.12 | 0.06 |
| snli | 250 | 33 | 68.8 | 68.8 | 90.4 | 0.050 | 0.069 | 0.72 | 0.30 |
| tweet_hate | 250 | 50 | 60.4 | 60.4 | 68.4 | 0.110 | 0.092 | 0.68 | 0.62 |
| tweet_offensive | 250 | 50 | 40.0 | 40.0 | 76.4 | 0.384 | 0.075 | 1.05 | 0.49 |
| **ALL** | 3678 | 31 | 71.1 | 71.1 | 78.0 | 0.068 | 0.043 | 0.80 | 0.58 |

Per dataset, the confirmation set (the base model cannot answer GTSRB and RESISC45, which have 43 and 45 options):

| Dataset | n | Chance % | Base acc | Base + temp. scaling acc | mini-Jev acc | Base + temp. scaling ECE | mini-Jev ECE | Base + temp. scaling NLL | mini-Jev NLL |
|---|---|---|---|---|---|---|---|---|---|
| arc_easy | 250 | 25 | 95.6 | 95.6 | 94.8 | 0.029 | 0.077 | 0.17 | 0.22 |
| cb | 56 | 33 | 66.1 | 66.1 | 82.1 | 0.151 | 0.124 | 0.74 | 0.52 |
| copa | 100 | 50 | 87.0 | 87.0 | 89.0 | 0.064 | 0.052 | 0.35 | 0.30 |
| gtsrb | 200 | 2 | 0.0 | 0.0 | 54.0 | 0.000 | 0.143 | 13.82 | 1.65 |
| resisc45 | 200 | 2 | 0.0 | 0.0 | 58.0 | 0.000 | 0.130 | 13.82 | 1.29 |
| rotten_tomatoes | 250 | 50 | 86.0 | 86.0 | 88.8 | 0.043 | 0.052 | 0.37 | 0.24 |
| stl10 | 200 | 10 | 99.5 | 99.5 | 98.5 | 0.205 | 0.027 | 0.25 | 0.05 |
| svhn | 200 | 10 | 74.5 | 74.5 | 79.5 | 0.272 | 0.064 | 1.10 | 0.66 |
| tweet_emoji | 250 | 5 | 19.6 | 19.6 | 19.6 | 0.213 | 0.124 | 2.77 | 3.14 |
| tweet_emotion | 250 | 25 | 80.8 | 80.8 | 75.6 | 0.103 | 0.101 | 0.62 | 0.62 |
| tweet_sentiment | 250 | 33 | 52.8 | 52.8 | 72.4 | 0.341 | 0.074 | 1.47 | 0.67 |
| tweet_stance | 250 | 33 | 66.4 | 66.4 | 60.4 | 0.117 | 0.106 | 0.91 | 0.99 |
| wic | 250 | 50 | 48.0 | 48.0 | 56.0 | 0.374 | 0.175 | 1.12 | 0.79 |
| **ALL** | 2706 | 25 | 58.9 | 58.9 | 69.6 | 0.084 | 0.041 | 2.86 | 0.91 |

On the in-distribution test sets (sources seen in training; 3-seed means of the earlier mix) text accuracy is
87.8 % against the base model's 79.6 % and image accuracy 83.5 % against 80.6 %, with ScienceQA test images level
with the base model (89.9 % vs 90.3 %). Full ablation table (base, head-only, hard-label, text-only, image-only,
three seeds): [results/ablations.md](results/ablations.md). Method and test sets:
[docs/METHODOLOGY.md](docs/METHODOLOGY.md); every comparison with intervals: [results/study/](results/study/).

## What it can and cannot do

**Can**
- Pick one of 2-255 options for a state of text and/or up to 8 images, with a calibrated probability for each.
- Answer yes/no questions (`noul`, returns P(yes)) and ordered scores (`score`, probabilities plus expected value).
- Answer several questions about the same state at once; the state is encoded once and reused.
- Run fast on an RTX 5070 Laptop: median 69 ms per request on the validation mix; 39 ms for a short text question,
  89 ms for 8 questions about one short text, 434 ms for an image plus 8 questions
  ([results/latency.md](results/latency.md)). Peak VRAM 6.2 GB in training and 7.4 GB in the latency benchmark
  (4,000-token inputs); it runs within 8 GB.
- Run text-only (no image) or image + text.

**Cannot**
- Write free text, captions or explanations, or answer outside the options you supply.
- Handle audio or video.
- Replace a general model: it is a 2B model. It is weak on chess, music, routing, expert-level science and
  long-context retrieval (see the Decision Index section below).
- Take requests above the declared limit (16,384 tokens by default, 255 options, 8 images): those return
  `Unsupported`. Nothing is silently truncated.
- Be fully trusted on questions with many (> 10) options without option-order averaging (see below).

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
| `choice` | 2-255 keys in `criteria` (key → short description) | probability per key and the top `choice` |
| `noul` | yes / no | `noul` = P(yes) |
| `score` | ordered levels (the `criteria` keys, in order) | probability per level, top `choice`, expected `score` |

The state can be plain text or a JSON-serialisable object. Put what the model should know in `state`
and the rules for each option in `criteria`; the more concrete the descriptions, the better the answers.

**Many options?** Pass `n_perm=4` to average each `choice` question over four option orders (identity, reversed,
two seeded shuffles). It reduces position bias: +1.0 [+0.4, +1.5] points on the 16 datasets and better log-loss
elsewhere, but no significant accuracy gain on the other collections, and a higher latency (the state is still
encoded once). Yes/no and score questions are never permuted:

```python
out = jev.decide(state, questions, n_perm=4)
```

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
python run_all.py                    # data → 3 seeds of the released recipe → calibrate → latency → release
python run_all.py --mix narrow       # the original 12-source mix used for the ablation study
python run_all.py --steps publish    # upload to Hugging Face (hf auth login first)
```

Rough timings on the RTX 5070 Laptop: data 1-2 h (downloads), about 2.5-3 h per training run of the broad mix
(1,925 steps; 6.2 GB peak), evaluation 1-2 h. Every step skips finished work, so you can stop and resume. Single steps:

```bash
python train.py --model Qwen/Qwen3-VL-2B-Instruct --train data/broad/train_clean.jsonl --val data/broad/val.jsonl --out runs/broad_seed0
python calibrate.py --run runs/broad_seed0 --val data/broad/val.jsonl --test data/mix/test_text.jsonl data/mix/test_vision.jsonl
python -m eval.evaluate --kind base --model Qwen/Qwen3-VL-2B-Instruct --fit-temps data/mix/val.jsonl --test data/mix/test_more.jsonl --out base.json
python -m eval.evaluate --kind minijev --path runs/broad_seed0 --test data/mix/test_more.jsonl --out ours.json
python -m eval.compare --file test_more.jsonl --ref base_T --runs base_T=base.json ours=ours.json   # paired bootstrap CIs
python -m eval.latency --run runs/broad_seed0
```

Training data (broad mix, 26,413 rows / 31,059 questions): AG News, DBpedia-14, Yelp, BoolQ, MNLI, SST-2,
CommonsenseQA, ScienceQA, A-OKVQA, ChartQA, procedural charts and shapes, Food-101 seen classes, CoLA, STS-B, RACE,
SMS spam, Yahoo Answers topics, Amazon polarity, QuaRTz, SST-5, Oxford pets, Oxford flowers, EuroSAT, DTD textures.
Held out on purpose, never trained on: paraphrase (MRPC), hate / offensive / irony tweets, financial news
sentiment, digit / clothing / leaf-disease images, plus emotion, OpenBookQA and unseen Food-101 classes. No Decision Index
benchmark is used for training; no training row overlapped the Decision Index suites by 13-gram text overlap (image
hashes were not checked).

### Optional inference techniques studied

| Technique | Effect (see [report](docs/REPORT.md)) |
|---|---|
| `n_perm=4` option-order averaging | released model: +1.0 [+0.4, +1.5] on the 16 datasets, lower log-loss elsewhere, > 10 options 54.2 → 56.9 %; no effect on yes / no questions; original model: up to +1.3 and a full repair of the > 10-option gap |
| 3-seed probability ensemble (`eval/ensemble.py`) | +0.8 [+0.1, +1.5] on the 16 datasets at 3x inference cost |
| Option-count augmentation of the training data | no benefit (negative result) |

## Jev Decision Index (text board): unofficial, earlier model

An earlier (original 12-source mix) version of this model was run through the kit at
github.com/apolinario/decision-index (edition 0.2.1): 149,432 requests, all answered, **Decision Index 17.43**
(chance-corrected, 0 = random guessing), raw 36.8 %, median latency 61 ms. **The released model was not re-run.**
This is not a leaderboard entry: the official suite is not downloadable, so it was rebuilt from public sources and
the core rows are not byte-identical, and ToolRet-retrieval and BRIGHT-retrieval were not run (they cannot finish
on an 8 GB GPU, so they count as wrong). Details: [results/decision_index_text_unofficial.md](results/decision_index_text_unofficial.md).
The vision board was not run. How to run both: [docs/SUBMITTING.md](docs/SUBMITTING.md).

## Repository

```
jev/schema.py        question types, validation, answers                jev/encode.py   prompt rendering, image tokens
jev/model.py         backbone + LoRA + answer head / letter head         jev/decide.py   cached inference, chunking, option-order averaging
jev/losses.py        soft CE, RPS, Brier (and the hard-label ablation)   jev/metrics.py  accuracy, skill, ECE, AURC, ...
train.py  calibrate.py  infer.py  serve.py  di_engine.py  publish_hf.py  run_all.py
eval/                predictors (mini-Jev, base zero-shot, Ollama), evaluate, ablations, latency,
                     compare / compare_groups (paired bootstrap), ensemble, report_table
data/                text_public, vision_public, broad_public, augment_options, procedural images, Food-101,
                     convert_legacy, build_mix, decontam, and the test-set builders (make_more_tests, make_fresh_tests,
                     make_manyopt_tests, make_confirm_tests)
docs/                REPORT.md (the study), METHODOLOGY.md (design), SUBMITTING.md (publishing, leaderboard)
results/             tables quoted above; results/study/ has every paired comparison with intervals
tests/               tiny random Qwen3-VL: cached = full, training, letter head, option averaging, server, engine
```

## Licence

Apache-2.0, see [LICENSE](LICENSE). The base model `Qwen/Qwen3-VL-2B-Instruct` is Apache-2.0. Public datasets are
downloaded by the scripts under their own licences and are not redistributed.
