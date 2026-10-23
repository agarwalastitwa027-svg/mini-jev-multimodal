# test_more.jsonl: 3678 questions, 16 datasets, groups of runs, 1000 bootstrap resamples

| Group | runs | Acc micro % | Acc macro % | ECE | NLL |
|---|---|---|---|---|---|
| narrow | 3 | 76.2 ± 0.6 [75.1, 77.3] | 75.8 ± 0.5 [74.7, 76.9] | 0.049 ± 0.011 [0.043, 0.060] | 0.62 ± 0.02 [0.59, 0.64] |
| broad | 3 | 77.6 ± 0.3 [76.6, 78.7] | 77.3 ± 0.4 [76.2, 78.5] | 0.050 ± 0.012 [0.046, 0.062] | 0.58 ± 0.01 [0.55, 0.61] |

Cells: mean over runs ± sd across runs [95% bootstrap interval of the mean].

Difference of group means vs `narrow` (positive accuracy better, negative ECE/NLL better):

| Group | d Acc micro (pts) | d Acc macro (pts) | d ECE | d NLL |
|---|---|---|---|---|
| broad | +1.4 [+0.8, +2.0] p=0.001 | +1.5 [+0.9, +2.1] p=0.001 | +0.000 [-0.005, +0.008] p=0.692 | -0.04 [-0.05, -0.03] p=0.001 |
