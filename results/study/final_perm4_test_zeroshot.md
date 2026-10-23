# test_zeroshot.jsonl: 720 questions, 3 datasets, 1000 bootstrap resamples

| Run | Acc micro % [95% CI] | Acc macro % [95% CI] | ECE [95% CI] | NLL [95% CI] |
|---|---|---|---|---|
| released | 70.0 [66.7, 73.3] | 75.5 [72.6, 78.3] | 0.070 [0.057, 0.107] | 0.91 [0.83, 1.01] |
| released_perm4 | 71.0 [67.8, 74.2] | 76.3 [73.6, 78.8] | 0.061 [0.055, 0.102] | 0.91 [0.82, 1.00] |

Paired difference vs `released` (run - ref; positive accuracy is better, negative ECE/NLL is better). p = two-sided bootstrap p-value.

| Run | d Acc micro (pts) | d Acc macro (pts) | d ECE | d NLL |
|---|---|---|---|---|
| released_perm4 | +1.0 [-1.0, +2.8] p=0.334 | +0.8 [-0.8, +2.2] p=0.320 | -0.009 [-0.027, +0.020] p=0.768 | -0.00 [-0.02, +0.01] p=0.664 |

Per dataset (accuracy %, 95% CI):

| Dataset | n | released | released_perm4 |
|---|---|---|---|
| emotion | 300 | 57.7 [52.0, 63.3] | 56.7 [51.0, 62.0] |
| food101_zeroshot | 120 | 97.5 [94.2, 100.0] | 97.5 [94.2, 100.0] |
| openbookqa | 300 | 71.3 [66.0, 76.3] | 74.7 [69.7, 79.7] |
