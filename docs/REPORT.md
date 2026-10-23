# What fine-tuning a small vision-language model into a decision model buys: an empirical report

Everything here was run on one laptop (RTX 5070 Laptop GPU, 8 GB) with one base model, `Qwen/Qwen3-VL-2B-Instruct`.
All tables come from files in [results/](../results/); the commands to regenerate them are in section 8. This
is an independent study, not affiliated with TypeSafe AI's Jev, and it makes no novelty claim for the
architecture (see section 7).

## 1. Summary

1. **Fine-tuning helps accuracy on unseen datasets, by a modest amount that depends on how close the task is to
   the training families.** Against the strongest baseline (the base model with post-hoc temperature scaling),
   the released model gains **+6.9 points [+5.5, +8.3]** on 16 unseen open datasets, but **+2.8 points
   [+1.2, +4.4]** on 11 further datasets that influenced no decision (confirmation set, questions the base model can
   answer at all), and **no significant gain** on three zero-shot sources (+1.7 [-1.3, +4.3]).
2. **Most of the calibration gain is not due to fine-tuning.** Temperature scaling the *base* model cuts its
   expected calibration error (ECE) from 0.240 to 0.068. Fine-tuning reaches 0.043 and a lower log-loss
   (0.58 vs 0.80), a real but smaller edge.
3. **Training on more tasks is the one change that clearly helped.** Three seeds of a broader training mix beat
   three seeds of the narrower one on every held-out collection: +1.4 [+0.8, +2.0] (16 datasets),
   +4.6 [+2.8, +6.5] (fresh), +1.6 [+0.2, +3.2] (zero-shot), +5.1 [+3.5, +6.8] (27-100 options) and
   +1.3 [+0.5, +2.1] (confirmation).
4. **The learned answer head is not more accurate than a plain letter readout.** Inside the same framework,
   same data and prompts, a letter-readout head matches it on accuracy and generalises better to unseen sources
   (+4.4 [+1.8, +6.9]); the answer head is better calibrated (ECE 0.047 vs 0.073) and is the only one that handles
   more than 26 options (up to 255).
5. **Known weakness:** on questions with more than 10 options the model is *below* the base model (54.2% vs 58.7%
   in the pooled test). Averaging over 4 option orders at test time recovers part of it (56.9%) and adds +1.0
   [+0.4, +1.5] on the 16 datasets; it fully closed the gap for the earlier narrow-mix model but not for the released
   one. Option-count augmentation of the training data did not help.

## 2. Setup

**Model.** `Qwen3-VL-2B-Instruct` with its LM head removed, LoRA (r=16) on the text layers, and an answer head
that scores each option from its text; probabilities come from one prefill pass with the images and state encoded
once (README, [METHODOLOGY](METHODOLOGY.md)). Calibration is one temperature per question type fitted on
validation data.

**Baselines.**
- **B0, base model, zero-shot:** same state, question and options, options lettered A, B, C...; probabilities
  from the next-token logits of the letters. Limited to 26 options (more count as wrong).
- **B1, base + temperature scaling:** B0 with one temperature per question type fitted on the validation set.
  This is the fair calibration baseline; without it, calibration comparisons against B0 are misleading.
- **B2, letter-readout head:** the same LoRA fine-tuning, data, prompts, cache and calibration, but the answer is
  read from the tied LM-head logits of the option letters instead of the learned answer head. Isolates the head.

**Training mixes.** *Narrow*: 12 sources, 14,997 rows (AG News, DBpedia, Yelp, BoolQ, MNLI, SST-2, CommonsenseQA,
ScienceQA, A-OKVQA, ChartQA, procedural charts, Food-101 seen classes). *Broad* (the released recipe): narrow plus 12
sources, 26,413 rows (text: CoLA, STS-B, RACE, SMS spam, Yahoo topics, Amazon polarity, QuaRTz, SST-5; images:
Oxford pets, Oxford flowers, EuroSAT, DTD textures; CosmosQA was planned but could not be loaded). Whole task families are held out of both: paraphrase (MRPC), hate / offensive /
irony tweets, financial sentiment, and digit / clothing / leaf-disease images. No Decision Index benchmark is used;
0 of the training rows overlapped the Decision Index suites by 13-gram text overlap.

**Test collections** (none overlaps any training mix):

| Collection | Questions | Datasets | Role |
|---|---|---|---|
| `test_more` | 3,678 | 16 (11 text, 5 image) | built before the experiments; used to compare variants |
| `test_fresh` | 600 | ARC-Challenge, tweet irony | used to compare variants |
| `test_zeroshot` | 720 | emotion, OpenBookQA, Food-101 unseen dishes | used to compare variants |
| `test_manyopt` | 700 | CIFAR-100 (100 options), BANKING77 (77), GoEmotions (28) | used to study the many-option weakness |
| **`test_confirm`** | 2,706 | 13 (SuperGLUE CB / COPA / WiC, 4 tweet tasks, Rotten Tomatoes, ARC-Easy, STL-10, GTSRB, RESISC45, SVHN) | **built after the comparison; influenced no decision** |

**Statistics.** Paired, dataset-stratified bootstrap (1,000 resamples) with 95 % percentile intervals; the same
resampled questions are used for every run, so differences are paired. Group comparisons average over seeds
(3 seeds per recipe) and report the spread between seeds. "Acc micro" pools questions; "macro" averages datasets.
The released model is the seed with the lowest validation log-loss (rule fixed in advance).

## 3. Fine-tuning against strong baselines

Accuracy %, ECE and NLL (log-loss); lower is better for the last two. 95 % intervals in `results/study/`.

| Test collection | n | B0 base | B1 base + temp. scaling | Narrow-mix model | **Released model** | Released - B1 |
|---|---|---|---|---|---|---|
| `test_more` (16 datasets) | 3,678 | 71.1 | 71.1 | 76.8 | **78.0** | **+6.9 [+5.5, +8.3]** |
| `test_confirm`, <= 26 options | 2,306 | 69.2 | 69.2 | 70.9 | **72.0** | **+2.8 [+1.2, +4.4]** |
| `test_fresh` (2 datasets) | 600 | 61.2 | 61.2 | 69.2 | **76.2** | +15.0 [+11.2, +18.8] |
| `test_zeroshot` (3 sources) | 720 | 68.3 | 68.3 | 67.5 | **70.0** | +1.7 [-1.3, +4.3] (n.s.) |
| `test_confirm`, all | 2,706 | 58.9 | 58.9 | 68.4 | **69.6** | +10.7 (base cannot answer >26 options) |
| `test_manyopt` (27-100 options) | 700 | 0.0 | 0.0 | 46.4 | **48.4** | base cannot answer |

| Calibration (`test_more` / `test_confirm` <= 26) | B0 | B1 | Released |
|---|---|---|---|
| ECE | 0.240 / 0.255 | 0.068 / 0.099 | 0.043 / 0.044 |
| NLL | 2.04 / 2.66 | 0.80 / 0.96 | 0.58 / 0.81 |

Reading it: the gain is large where the task resembles the training families (NLI, sentiment, offensive-language
tweets), smaller on datasets that do not, and absent on the three zero-shot sources. The honest like-for-like
estimate for new datasets is the `test_confirm` <= 26 row. The comparison on the full confirmation set (+10.7) is
inflated by 400 questions with 43-45 options that the 26-letter baseline cannot answer; it is shown only to be
complete. Temperature scaling alone closes 87 % (`test_more`) to 74 % (`test_confirm`, <= 26 options) of the gap
between the raw base model and the released model's ECE, so fine-tuning's calibration claim is limited to the
remainder.

By number of options per question (pooled `test_more` + `test_fresh` + `test_zeroshot`, 4,998 questions):

| Options | n | B1 | Narrow model | Narrow + 4 option orders | Released | Released + 4 option orders |
|---|---|---|---|---|---|---|
| 2 (yes / no) | 1,800 | 66.1 | 75.4 | 75.4 | 77.4 | 77.4 |
| 3-4 | 1,597 | 72.6 | 79.5 | 80.0 | 82.2 | 83.2 |
| 5-10 | 1,151 | 74.7 | 74.5 | 75.2 | 76.4 | 77.5 |
| > 10 | 450 | **58.7** | 54.0 | 59.8 | 54.2 | 56.9 |

The gain comes from yes / no and few-option questions; for more than 10 options the fine-tuned model is worse
than the base model, and option-order averaging closes the gap only for the narrow-mix model.

## 4. Task breadth (three seeds per recipe)

Mean +- standard deviation over three seeds; difference of means with a paired bootstrap interval. `narrow` is the
original 12-source mix, `broad` the released recipe.

| Collection | Narrow acc | Broad acc | Difference | Broad ECE vs narrow | Broad NLL vs narrow |
|---|---|---|---|---|---|
| `test_more` | 76.2 +- 0.6 | 77.6 +- 0.3 | **+1.4 [+0.8, +2.0]** | 0.050 vs 0.049 (n.s.) | -0.04 [-0.05, -0.03] |
| `test_fresh` | 68.8 +- 1.8 | 73.3 +- 2.6 | **+4.6 [+2.8, +6.5]** | -0.049 [-0.064, -0.027] | -0.13 [-0.16, -0.10] |
| `test_zeroshot` | 68.8 +- 1.4 | 70.4 +- 0.6 | **+1.6 [+0.2, +3.2]** | +0.016 (n.s.) | +0.02 (n.s.) |
| `test_manyopt` | 45.2 +- 2.1 | 50.3 +- 2.7 | **+5.1 [+3.5, +6.8]** | +0.019 (n.s.) | -0.26 [-0.30, -0.21] |
| `test_confirm` (all) | 68.7 +- 0.4 | 70.1 +- 0.4 | **+1.3 [+0.5, +2.1]** | +0.004 (n.s.) | -0.01 (n.s.) |

Accuracy improves on every collection, including the confirmation set. Calibration error does not change
consistently. Seed spread is 0.3-2.6 points, which is why single-seed differences below about 2 points should not be
trusted; the original model's ECE of 0.037 on `test_more` was a lucky seed (the other two scored 0.060 and 0.050).

## 5. Learned answer head vs letter readout

Same data (narrow mix), prompts, LoRA, cache and calibration; only the readout differs (one seed each for the
letter head).

| | `test_more` | `test_fresh` | `test_zeroshot` | `test_confirm` <= 26 | options > 26 |
|---|---|---|---|---|---|
| Accuracy: letter head minus answer head | -0.2 [-1.4, +0.9] (tie) | +0.2 (tie) | **+4.4 [+1.8, +6.9]** | 70.7 vs 71.1, 3-seed mean (tie) | 0.0 vs 46.4 |
| ECE, answer head / letter head | 0.037 / 0.073 | 0.070 / 0.108 | 0.054 / 0.074 | 0.047 / 0.073 | n/a |

The answer head's measurable advantages are calibration error and the ability to take up to 255 options; the letter
readout is as accurate, simpler, and generalises better to unseen sources. If a use case never exceeds 26
options, a letter-readout fine-tune is a serious alternative.

## 6. Other techniques tried

| Technique | Result |
|---|---|
| **Average over 4 option orders** at test time (`Decider.decide(..., n_perm=4)`) | Narrow model: +1.3 [+0.7, +1.8] on `test_more`, +0.8 [0.0, +1.6] on `test_confirm`; no effect on yes / no questions (not permuted) or on `test_fresh` / `test_zeroshot`. Fixes the > 10-option weakness of the narrow model (54.0 -> 59.8). Released-model result: see section 6.1. |
| **Seed ensemble** (mean of 3 seeds' probabilities) | +0.8 [+0.1, +1.5] on `test_more`, +1.9 [+0.3, +3.6] on `test_zeroshot`; costs 3x inference. |
| **Option-count augmentation** (extra training copies padded to 12-40 options) | **Negative.** No better than the broad mix alone (27-100 options: 52.4 vs 53.4; `test_more`: 77.6 vs 77.3). |
| **Temperature scaling of the base model** | Recovers 74-87 % of the base model's ECE gap (section 3). |

### 6.1 Option-order averaging on the released model

Released model with and without `n_perm=4` (paired differences; positive accuracy and negative NLL are better):

| Collection | Acc without | Acc with 4 orders | Difference | NLL difference |
|---|---|---|---|---|
| `test_more` | 78.0 | 78.9 | **+1.0 [+0.4, +1.5]** | -0.02 [-0.02, -0.01] |
| `test_fresh` | 76.2 | 76.0 | -0.2 [-1.5, +1.2] (n.s.) | 0.00 (n.s.) |
| `test_zeroshot` | 70.0 | 71.0 | +1.0 [-1.0, +2.8] (n.s.) | 0.00 (n.s.) |
| `test_manyopt` | 48.4 | 49.9 | +1.4 [-0.9, +3.6] (n.s.) | -0.06 [-0.09, -0.03] |
| `test_confirm` | 69.6 | 70.0 | +0.4 [-0.4, +1.1] (n.s.) | -0.02 [-0.03, -0.01] |

So for the released model option-order averaging is a small, consistent improvement in log-loss and a significant
accuracy gain only on the 16 datasets. Latency rises: with it the median request takes 70-93 ms on the first three
collections, 80 ms on the confirmation set and 322 ms on the 27-100-option set (69 ms without it on the validation
mix). It is optional and off by default. Tables: `results/study/final_perm4_*.md`.

## 7. Limitations and threats to validity

- **Novelty.** Learned option heads, letter-logit readouts and calibrated one-pass decisions already exist (e.g.
  SALSA, classification-head fine-tuning papers, and several open "Jev-like" models such as decider-2b). This work
  is an empirical comparison, not a new architecture, and it was not compared against those other released models.
- **One base model, one GPU, LoRA r=16 only**; no full fine-tune, no larger or smaller model, no learning-rate sweep.
- **Seeds:** three per recipe for the main comparison, one for the letter head, the augmentation run and the
  per-technique results.
- **Selection effects:** the comparison collections guided choices (which variant to keep); only `test_confirm`
  was untouched, and its gains are smaller than on `test_more`, as expected after selection.
- **Pretraining contamination of the base model** on the test datasets (ARC, SNLI, IMDB, ...) was not checked.
- **Task-family overlap:** some test datasets are close to training tasks (SNLI vs MNLI, RTE vs MNLI), so part of
  the large gains is transfer within a family. Held-out families (MRPC, tweet hate / offensive, financial news,
  digits, clothing, leaf disease) are listed in section 2.
- **The Decision Index run is unofficial and was done on the earlier (narrow-mix) model**, not the released one:
  index 17.43 on a locally rebuilt, non-byte-identical suite, with ToolRet and BRIGHT not run
  ([results/decision_index_text_unofficial.md](../results/decision_index_text_unofficial.md)). The vision board was
  not run.
- **Decontamination** checked text overlap only (13-grams); image hashes were not checked.
- The letter readout is capped at 26 options and the base model's zero-shot baseline likewise; questions beyond
  that count as wrong, which is why those rows are reported separately.

## 8. Reproduce

```bash
python run_all.py --mix narrow --steps data train calibrate ablations   # the narrow-mix study (ablations)
python run_all.py                                                       # data, 3 seeds of the released recipe, calibrate, release
python -m eval.evaluate --kind base --model Qwen/Qwen3-VL-2B-Instruct --fit-temps data/mix/val.jsonl \
    --test data/mix/test_more.jsonl --out results/exp/base_T.json       # B0 and B1 (raw probabilities are stored)
python train.py --head-type letters --train data/mix/train.jsonl --val data/mix/val.jsonl --out runs/letters_seed0   # B2
python -m eval.evaluate --kind minijev --path runs/release --n-perm 4 --test data/mix/test_more.jsonl --out x.json
python -m eval.compare --file test_more.jsonl --ref base_T --runs base_T=a.json mini=b.json      # paired bootstrap
python -m eval.compare_groups --file test_confirm.jsonl --ref narrow --groups narrow=a0.json,a1.json,a2.json broad=b0.json,b1.json,b2.json
python -m eval.ensemble --out ens.json s0.json s1.json s2.json
```

Test sets: `data/make_more_tests.py`, `make_fresh_tests.py`, `make_manyopt_tests.py`, `make_confirm_tests.py`.
Raw per-question records for every run in this report are in the Hugging Face results dataset.
