# test_manyopt.jsonl: 700 questions, 3 datasets, groups of runs, 1000 bootstrap resamples

| Group | runs | Acc micro % | Acc macro % | ECE | NLL |
|---|---|---|---|---|---|
| narrow | 3 | 45.2 ± 2.1 [42.2, 48.3] | 47.4 ± 2.1 [44.4, 50.5] | 0.059 ± 0.003 [0.058, 0.091] | 2.24 ± 0.03 [2.12, 2.36] |
| broad | 3 | 50.3 ± 2.7 [47.4, 53.2] | 52.3 ± 2.6 [49.3, 55.2] | 0.078 ± 0.030 [0.074, 0.112] | 1.98 ± 0.09 [1.86, 2.11] |

Cells: mean over runs ± sd across runs [95% bootstrap interval of the mean].

Difference of group means vs `narrow` (positive accuracy better, negative ECE/NLL better):

| Group | d Acc micro (pts) | d Acc macro (pts) | d ECE | d NLL |
|---|---|---|---|---|
| broad | +5.1 [+3.5, +6.8] p=0.001 | +5.0 [+3.3, +6.7] p=0.001 | +0.019 [-0.005, +0.040] p=0.098 | -0.26 [-0.30, -0.21] p=0.001 |
