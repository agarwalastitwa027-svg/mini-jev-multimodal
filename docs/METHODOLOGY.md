# Methodology and comparisons

This document says how the model works and was trained, and exactly which comparisons the results tables report.

## 1. What a decision model is here

Input: a **state** (text or JSON, and 0–8 images) plus typed **questions**. Output: a probability for
every allowed option of every question, from one prefill pass per request, with no generated tokens.

| Type | Options | Output |
|---|---|---|
| `choice` | 2–255 keys from `criteria` | probability per key, argmax `choice` |
| `noul` | yes / no | `noul` = P(yes) |
| `score` | ordered levels (criteria keys in order) | probability per level, argmax, expected `score` |

`choice` and `noul` match the Jev / Decision Index wire format. The Decision Index sends ordinal
questions as `choice`; `score` is an extension for our own data.

## 2. Model

- **Backbone:** `Qwen/Qwen3-VL-2B-Instruct`, one model for text-only and image+text inputs. The LM head
  is removed (`AutoModel`); LoRA r=16 on the text layers' attention and MLP projections; the vision
  tower is frozen.
- **Answer head:** `q = W_q·h[last]`, `o_j = W_o·mean(h[option j tokens])`, `p = softmax(q·o_j/√d/T[type])`.
  Options are scored through their own text, so new options and new tasks need no new parameters.
- **Shared state:** the prefix (images + state) runs once and its KV cache is reused for every question.
  Prefix and suffix are tokenized separately and concatenated identically in training and inference,
  so the cached path equals the full path exactly (tested to 1e-5 in float32: `tests/test_tiny.py`).
  M-RoPE positions for the question tokens continue from the prefix's largest position, as the model
  would compute them for the full sequence.
- **Memory:** questions are batched in chunks so (prefix + suffix) × chunk stays under `token_budget`;
  on CUDA out-of-memory the chunk halves down to one question, then the request is refused.
- **No truncation:** requests longer than the declared `max_tokens` (default 16,384) raise `Unsupported`.

## 3. Training

| Setting | Value | Why |
|---|---|---|
| Loss | soft cross-entropy (+ RPS for `score`) | soft targets carry label uncertainty; RPS is a proper scoring rule that respects order |
| Option order | shuffled every step (not for `score`) | removes position bias |
| Modality mix | 50 % of steps text, 50 % image (`--text-frac`) | the minority modality is oversampled, nothing dropped |
| Long examples | skipped above 2,048 tokens during training | never truncated |
| Images | ≤ 262,144 px during training | memory on 8 GB; inference uses native resolution up to 1.6 MP |
| Selection | best validation log-loss | favours calibrated checkpoints over lucky accuracy |
| Calibration | one temperature per type, fitted on validation | fixes over/under-confidence without changing accuracy |

### Data (`run_all.py --steps data`)

| Modality | Source | Questions per state | Notes |
|---|---|---|---|
| text | AG News, DBpedia-14, Yelp (score + noul), BoolQ, MNLI (choice + noul), SST-2 (noul + choice), CommonsenseQA | 1–2 | train split → train/val, official eval split → test |
| text (zero-shot) | Emotion, OpenBookQA | 1 | never trained on |
| image | procedural charts and shapes (exact labels from the drawing code, teacher-reworded) | up to 6 | earlier synthetic data, converted |
| image | Food-101 seen classes | 1 | earlier synthetic data, converted |
| image | ScienceQA train (image rows; text rows go to the text side), A-OKVQA, ChartQA (4-way from free-form) | 1 | |
| image (zero-shot) | Food-101 held-out classes | 1 | never trained on |
| benchmark | ScienceQA test image questions (300) | 1 | public benchmark, zero-shot |

**Decontamination.** No split of any Decision Index benchmark (text or vision board) is trained on;
`data/build_mix.py` refuses such sources. `data/decontam.py` then removes training rows whose sampled
13-grams overlap the downloaded suite by ≥ 50 %, short rows that occur verbatim, and images whose sha256
matches a suite image. The BANKING77 and CLINC150 converters were removed because both are on the board.

## 4. Comparisons

All comparisons use the same test files. Unanswered rows count as wrong.

| ID | Model | Question it answers |
|---|---|---|
| A | Base Qwen3-VL-2B, zero-shot: same state/question, options lettered A–Z, next-token letter probabilities | Is fine-tuning better than the base model? |
| B | Head only (`--lora-r 0`) | How much does the head alone get, without adapting the backbone? |
| C | Hard-label recipe (`--loss hard`: hard cross-entropy) | Do soft targets and RPS help accuracy or calibration? |
| D | Full recipe (mini-Jev), 3 seeds | The release model; seed spread shows run-to-run noise |
| E1 / E2 | Trained on text only / image only | Does mixing modalities help or hurt each one? |
| F1 | Teacher `qwen3.5:9b` via Ollama (generative, enum output) | Accuracy gap vs a 4.5× larger generative model, and the speed gap |
| F2 | `qwen3-vl:2b` via Ollama (generative) | Same size, generative vs decision head |

External reference points come from the published leaderboards, not reruns on 8 GB: Jev 1.13,
Clef-flash, JevK5 and Tiny-Jev on the Jev Decision Index.

### Metrics

- **Accuracy** and **skill** = (accuracy − chance) / (1 − chance), chance = mean of 1/#options.
- **Calibration:** ECE (15 bins), Brier, NLL, reliability diagram.
- **Selective prediction:** accuracy and coverage at confidence ≥ 0.5 / 0.7 / 0.9, AURC.
- **Speed:** row latency p50/p95 (one request), and `eval/latency.py`: 1 vs 8 questions per state,
  cached vs uncached, text states of 100 / 1,000 / 4,000 tokens and one 1.6 MP image; peak VRAM.

### Test sets

| File | What it measures |
|---|---|
| `test_text.jsonl` | held-out rows of the text sources seen in training |
| `test_vision.jsonl` | held-out rows of the image sources seen in training |
| `test_zeroshot.jsonl` | sources never trained on (Emotion, OpenBookQA, Food-101 held-out classes) |
| `bench_scienceqa.jsonl` | ScienceQA test image questions |
| Decision Index text suite | 37 benchmarks, 5 areas, the public leaderboard |
| Decision Index vision board | 11 public benchmarks (50 % of the Full score; maintainers run the private half) |

## 5. What to expect, honestly

- **A vs D:** fine-tuning should clearly win on in-domain tests and calibration; the zero-shot and
  ScienceQA gains will be smaller because those mostly reflect the pretrained backbone.
- **E:** a mixed model may lose a little on each modality versus a single-modality model of the same
  size, or gain from shared skills; the table measures it rather than assuming.
- **Leaderboard:** a 2B model will trail 9B–31B decision models on Knowledge & Reasoning; it can be
  competitive on Retrieval & Classification and Tools, and its latency is far lower.
