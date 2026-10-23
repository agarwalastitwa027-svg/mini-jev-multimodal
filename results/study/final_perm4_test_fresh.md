# test_fresh.jsonl: 600 questions, 2 datasets, 1000 bootstrap resamples

| Run | Acc micro % [95% CI] | Acc macro % [95% CI] | ECE [95% CI] | NLL [95% CI] |
|---|---|---|---|---|
| released | 76.2 [72.8, 79.3] | 76.2 [72.8, 79.3] | 0.045 [0.038, 0.083] | 0.56 [0.51, 0.63] |
| released_perm4 | 76.0 [72.7, 79.3] | 76.0 [72.7, 79.3] | 0.040 [0.035, 0.079] | 0.56 [0.50, 0.62] |

Paired difference vs `released` (run - ref; positive accuracy is better, negative ECE/NLL is better). p = two-sided bootstrap p-value.

| Run | d Acc micro (pts) | d Acc macro (pts) | d ECE | d NLL |
|---|---|---|---|---|
| released_perm4 | -0.2 [-1.5, +1.2] p=0.918 | -0.2 [-1.5, +1.2] p=0.918 | -0.005 [-0.023, +0.020] p=0.780 | -0.00 [-0.02, +0.01] p=0.720 |

Per dataset (accuracy %, 95% CI):

| Dataset | n | released | released_perm4 |
|---|---|---|---|
| arc_challenge | 300 | 80.7 [76.0, 85.0] | 80.3 [76.0, 84.7] |
| tweet_irony | 300 | 71.7 [66.7, 76.3] | 71.7 [66.7, 76.3] |
