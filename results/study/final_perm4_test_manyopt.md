# test_manyopt.jsonl: 700 questions, 3 datasets, 1000 bootstrap resamples

| Run | Acc micro % [95% CI] | Acc macro % [95% CI] | ECE [95% CI] | NLL [95% CI] |
|---|---|---|---|---|
| released | 48.4 [45.1, 51.7] | 50.5 [47.1, 53.8] | 0.056 [0.047, 0.101] | 2.02 [1.90, 2.14] |
| released_perm4 | 49.9 [46.6, 53.1] | 51.8 [48.5, 55.1] | 0.072 [0.064, 0.116] | 1.96 [1.85, 2.08] |

Paired difference vs `released` (run - ref; positive accuracy is better, negative ECE/NLL is better). p = two-sided bootstrap p-value.

| Run | d Acc micro (pts) | d Acc macro (pts) | d ECE | d NLL |
|---|---|---|---|---|
| released_perm4 | +1.4 [-0.9, +3.6] p=0.228 | +1.3 [-0.9, +3.4] p=0.246 | +0.017 [-0.016, +0.047] p=0.320 | -0.06 [-0.09, -0.03] p=0.001 |

Per dataset (accuracy %, 95% CI):

| Dataset | n | released | released_perm4 |
|---|---|---|---|
| banking77 | 250 | 48.0 [41.6, 54.4] | 52.8 [46.8, 59.2] |
| cifar100_fine | 200 | 79.5 [73.5, 85.0] | 79.0 [73.5, 84.5] |
| go_emotions | 250 | 24.0 [18.8, 29.2] | 23.6 [18.8, 28.8] |
