# test_fresh.jsonl: 600 questions, 2 datasets, groups of runs, 1000 bootstrap resamples

| Group | runs | Acc micro % | Acc macro % | ECE | NLL |
|---|---|---|---|---|---|
| narrow | 3 | 68.8 ± 1.8 [65.4, 72.2] | 68.8 ± 1.8 [65.4, 72.2] | 0.107 ± 0.033 [0.092, 0.144] | 0.72 ± 0.05 [0.65, 0.78] |
| broad | 3 | 73.3 ± 2.6 [70.1, 76.4] | 73.3 ± 2.6 [70.1, 76.4] | 0.058 ± 0.019 [0.053, 0.092] | 0.59 ± 0.04 [0.53, 0.65] |

Cells: mean over runs ± sd across runs [95% bootstrap interval of the mean].

Difference of group means vs `narrow` (positive accuracy better, negative ECE/NLL better):

| Group | d Acc micro (pts) | d Acc macro (pts) | d ECE | d NLL |
|---|---|---|---|---|
| broad | +4.6 [+2.8, +6.5] p=0.001 | +4.6 [+2.8, +6.5] p=0.001 | -0.049 [-0.064, -0.027] p=0.001 | -0.13 [-0.16, -0.10] p=0.001 |
