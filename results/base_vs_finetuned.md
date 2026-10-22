# Base Qwen3-VL-2B vs the fine-tuned model on data neither has seen

Same questions for both. The base model answers by reading option-letter probabilities (zero-shot);
the fine-tuned model is `AstitwaAgarwal/multimodal-jev` (seed 0). Acc in %. ECE and NLL: lower is better.
`tweet_irony` and `arc_challenge` come from `data/mix/test_fresh.jsonl` (300 questions each, datasets that
appear nowhere in training); the others are the zero-shot (held-out) sources. Each set has only 120-300
questions, so differences of 1-3 points are within noise.

| Dataset | n | Base acc | Tuned acc | Base ECE | Tuned ECE | Base NLL | Tuned NLL |
|---|---|---|---|---|---|---|---|
| ARC-Challenge (new) | 300 | 76.7 | 78.0 | 0.196 | 0.047 | 1.62 | 0.64 |
| tweet_eval irony (new) | 300 | 45.7 | 60.3 | 0.524 | 0.147 | 4.90 | 0.69 |
| emotion (held out) | 300 | 52.3 | 54.0 | 0.420 | 0.149 | 4.07 | 1.24 |
| OpenBookQA (held out) | 300 | 72.3 | 69.3 | 0.221 | 0.054 | 1.58 | 0.88 |
| Food-101 unseen dishes (held out) | 120 | 98.3 | 96.7 | 0.016 | 0.025 | 0.06 | 0.08 |
| **New 600-question set overall** | 600 | 61.2 | 69.2 | 0.357 | 0.070 | 3.26 | 0.66 |

Takeaway: fine-tuning mainly makes the probabilities trustworthy (ECE and NLL improve on four of five
sets); accuracy improves clearly only on tweet irony, where the base model is at chance.
