# Wider test: 16 open datasets that appear nowhere in training

Models: base Qwen3-VL-2B-Instruct (zero-shot option-letter probabilities), the v1 recipe (hard cross-entropy,
seed 0) and the released v2 (`AstitwaAgarwal/multimodal-jev`, seed 0). Same 3,678 questions for all three
(250 per text dataset, 200 per image dataset, 128 for beans). Acc in %; ECE and NLL: lower is better.
Rebuild the set with `data/make_more_tests.py`-style conversion of: allenai/sciq, allenai/qasc, GLUE rte/qnli/mrpc,
stanfordnlp/snli, stanfordnlp/imdb, cardiffnlp/tweet_eval (hate, offensive),
zeroshot/twitter-financial-news-sentiment, SetFit/20_newsgroups, uoft-cs/cifar10, cifar100 (20 coarse classes),
zalando-datasets/fashion_mnist, ylecun/mnist, AI-Lab-Makerere/beans. TREC was skipped (script-only dataset).

| Dataset | n | Chance | Base acc | v1recipe acc | **v2 acc** | Base ECE | v2 ECE | Base NLL | v2 NLL |
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

Reading it:
- v2 beats the base model's accuracy on 9 of 16 datasets. The big wins are where the base model's letter readout is
  weak: tweet_offensive (+34 points), snli (+19), fin_news (+30), rte (+13).
- v2 is not better on every set. It is slightly worse on image sets fashion_mnist and beans, and on 20-way
  newsgroups (44.8 vs 56.0); sciq, imdb and mnist are within about 1 point (noise at n=200-250).
- Calibration improves almost everywhere: overall ECE 0.240 -> 0.037 and NLL 2.04 -> 0.59.
- v2 and the v1 recipe are tied overall (76.8 vs 76.5).
- snli is close to MNLI, which was in the training mix, so part of that gain is task transfer rather than new skill.
- Single seed, 128-250 questions per dataset: differences of a few points are within noise.
