# mini-jev-multimodal

A small image + text **decision model** that runs on an 8 GB laptop GPU. You give it an image (optional),
some text, a question and a list of allowed options. It returns one probability per option from a single
forward pass, with no text generation.

It is a small, independent, open reproduction of the idea behind decision models such as Jev
(TypeSafe AI). It is **not affiliated with or derived from** that product.

```
image + context + question + options ──► Qwen3-VL-2B (prefill only, LoRA-tuned) ──► hidden states
                                                                                    │
            q   = W_q · h[last token]          (has read image, context, question and every option)
            o_j = W_o · mean(h[option j])      (one vector per option)
            p_j = softmax_j( q·o_j / √d / T )  ← T = temperature fitted by calibration
```

The LM head is removed, so the model cannot answer outside the options. Yes/no questions are options
`["yes","no"]` and 1–5 scores are options `["1".."5"]`, so one head covers choice, bool and score.

## What it can and cannot do

**Can**
- Pick one of several options for an image plus text, and give a probability for each option.
- Answer yes/no questions and 1–5 ratings the same way.
- Return calibrated confidence: in testing, high-confidence answers were more often correct.
- Run fast: about 86 ms per question on an RTX 5070 Laptop (batch size 1), about 13x faster than the 9B
  Ollama model it was compared against.
- Run text-only (no image), because the text path is unchanged.

**Cannot**
- Write free text, captions or explanations, or answer outside the options you supply.
- Handle audio or video. Image + text only.
- Be treated as a general vision model. It was trained on synthetic charts and shapes plus 10 food
  classes. Its ScienceQA score comes mostly from the pretrained Qwen3-VL-2B backbone.
- Match a 9B generative model on accuracy (see results).
- Share the image encoding across questions yet: `decide()` batches questions, but each one re-reads the
  image. True shared-prefix parallelism is not built.
- Train in 4-bit with images: `--load-4bit` is untested for the vision model.

## Results

Measured on an RTX 5070 Laptop (8 GB), bf16, 4.8 GB peak training memory. 244 optimizer steps, about
16 minutes.

| Test set | n | Accuracy |
|---|---|---|
| Held-out procedural images (charts, shapes) | 164 | 90.2% |
| Food-101 dishes never seen in training (4 options, chance 25%) | 120 | 96.7% |
| ScienceQA image questions, zero-shot (chance 36.1%) | 300 | 81.3% |

Against the Ollama teacher `qwen3.5:9b-q4_K_M` on the same questions:

| Test | mini-Jev (2B) | Teacher (9B) |
|---|---|---|
| Mixed test items, n=60 (accuracy / median latency) | 88.3% / 86 ms | 98.3% / 1132 ms |
| ScienceQA, n=300 (accuracy) | 81.3% | 95.7% |

Calibration (ScienceQA): ECE 0.087; the 48% of questions answered with confidence ≥ 0.9 were 92.3%
correct. Temperature scaling helped log-loss on held-out procedural images (0.52 → 0.34) but not on every
set. These are small test sets from a single run and seed, so treat the numbers as indicative.

## How it was built

1. **Images with exact labels.** `data/make_images.py` draws bar, pie and line charts and shape scenes;
   the answer is computed by the drawing code. `data/make_public_mm.py` takes Food-101 photos, with some
   dish classes held out for a zero-shot test.
2. **Ollama vision teacher.** `data/make_synthetic_mm.py` rewords questions and writes accompanying text.
   A rewording is kept only if a text-only judge says it asks the same thing, a blind answer on the image
   still matches the exact label, and the text does not leak the answer. Labels are never taken from the
   teacher.
3. **Train.** LoRA (r=16) on the Qwen3-VL-2B text layers plus a new answer head; vision tower frozen.
4. **Calibrate and evaluate.** Temperature scaling, ECE / Brier / selective accuracy, ScienceQA, and a
   latency comparison with the teacher.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The first run downloads `Qwen/Qwen3-VL-2B-Instruct` (about 4 GB) from Hugging Face. To rebuild the data
you also need [Ollama](https://ollama.com) and a vision model: `ollama pull qwen3.5:9b-q4_K_M`
(any Ollama vision model works with `--model`).

## Reproduce

```bash
# 1. images + labelled questions (deterministic for a given --seed)
python data/make_images.py --n 500 --seed 1 --out data/mm_procedural.jsonl --img-dir data/images/proc
python data/make_public_mm.py                       # Food-101: seen classes + zero-shot classes

# 2. teacher-enriched version of the procedural questions (about 1 h on an 8 GB GPU)
python data/make_synthetic_mm.py enrich --in data/mm_procedural.jsonl --out data/mm_enriched.jsonl --enrich-frac 0.4
#    or skip step 2: data/synthetic/mm_enriched.jsonl is included; run step 1 first so its images exist

# 3. split by image, then train
python data/split.py data/mm_enriched.jsonl --val 0.1 --test 0.15
python data/split.py data/mm_food_train.jsonl --val 0.1 --test 0.1
python train.py --model Qwen/Qwen3-VL-2B-Instruct \
    --train data/mm_enriched_train.jsonl data/mm_food_train_train.jsonl \
    --val data/mm_enriched_val.jsonl data/mm_food_train_val.jsonl \
    --out runs/mm_v1 --epochs 4 --bs 4 --grad-accum 4 --eval-bs 8 --eval-every 20 --lr 2e-4

# 4. calibrate + evaluate
python calibrate.py --run runs/mm_v1 --val data/mm_enriched_val.jsonl data/mm_food_train_val.jsonl \
    --test data/mm_enriched_test.jsonl data/mm_food_train_test.jsonl data/mm_food_zeroshot.jsonl --bs 8

# 5. public benchmark (zero-shot)
python data/make_benchmark_mm.py --n 300 --out data/bench_scienceqa.jsonl
python calibrate.py --run runs/mm_v1 --val data/mm_enriched_val.jsonl --test data/bench_scienceqa.jsonl --bs 4

# 6. compare with an Ollama model (run each half separately, they compete for VRAM)
python bench_vs_ollama.py --run runs/mm_v1 --test data/bench_scienceqa.jsonl --n 300 --skip-ollama
python bench_vs_ollama.py --test data/bench_scienceqa.jsonl --n 300 --skip-jev
```

`data/synthetic/mm_enriched.jsonl` regenerates exactly from `--seed 1`: all 1,076 labels matched when
checked.

## Use a trained model

```python
from infer import JevRunner

jev = JevRunner("runs/mm_v1")          # a folder containing lora/, head*.pt, meta.json, tokenizer files
out = jev.decide(
    "",                                  # text context; may be empty when you pass an image
    {"top": ["Apples", "Pears", "Grapes"], "any_pears": bool, "quality": (1, 5)},
    {"top": "Which category has the highest value?",
     "any_pears": "Is there a Pears bar?",
     "quality": "How readable is this chart, from 1 (poor) to 5 (excellent)?"},
    image="chart.png",
)
print(out["top"])   # {'answer': 'Apples', 'confidence': 0.99, 'probs': {...}}
```

All questions about one image go through the GPU as one batch. Text-only use is the same call without
`image=`.

## Files

```
jev/format.py                 prompt layout, option-span tokenization, image tokens, batching
jev/model.py                  MiniJev = AutoModel (no LM head) + LoRA + AnswerHead; save/load
train.py                      training loop (bf16/fp16, grad accumulation, checkpointing)
calibrate.py                  temperature scaling, ECE / Brier / selective accuracy, reliability plot
infer.py                      JevRunner.decide(...)
bench_vs_ollama.py            accuracy + latency vs an Ollama model with structured output
data/make_images.py           procedural charts / shape scenes with exact labels
data/make_public_mm.py        Food-101 photos (+ zero-shot held-out classes)
data/make_synthetic_mm.py     Ollama vision teacher: enrich labelled images / propose tasks for photos
data/make_benchmark_mm.py     ScienceQA image questions -> decision JSONL
data/split.py                 split by task / image id
data/synthetic/               teacher-enriched procedural questions (text only, no images)
data/make_public.py           (text-only) public datasets -> decision JSONL
data/make_synthetic.py        (text-only) Ollama teacher: tasks -> contexts -> verified
```

The text-only pipeline from the original code is still here and only smoke-tested (a few training steps),
not trained at scale.

## Notes

- Trained weights are not part of this repository. Rebuild them with the steps above (about 1.5 hours
  end to end on an 8 GB laptop GPU, most of it teacher generation).
- ScienceQA data and the Food-101 photos are not redistributed; the scripts download them.
- The teacher model's outputs are used as training text. Check the model's licence for your use.
