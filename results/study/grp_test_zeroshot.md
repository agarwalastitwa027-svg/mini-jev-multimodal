# test_zeroshot.jsonl: 720 questions, 3 datasets, groups of runs, 1000 bootstrap resamples

| Group | runs | Acc micro % | Acc macro % | ECE | NLL |
|---|---|---|---|---|---|
| narrow | 3 | 68.8 ± 1.4 [65.9, 71.7] | 74.6 ± 1.3 [72.2, 76.9] | 0.058 ± 0.009 [0.052, 0.088] | 0.89 ± 0.03 [0.82, 0.97] |
| broad | 3 | 70.4 ± 0.6 [67.5, 73.5] | 75.8 ± 0.5 [73.3, 78.4] | 0.074 ± 0.014 [0.063, 0.102] | 0.91 ± 0.02 [0.82, 1.00] |

Cells: mean over runs ± sd across runs [95% bootstrap interval of the mean].

Difference of group means vs `narrow` (positive accuracy better, negative ECE/NLL better):

| Group | d Acc micro (pts) | d Acc macro (pts) | d ECE | d NLL |
|---|---|---|---|---|
| broad | +1.6 [+0.2, +3.2] p=0.018 | +1.2 [+0.0, +2.5] p=0.050 | +0.016 [-0.005, +0.028] p=0.138 | +0.02 [-0.01, +0.05] p=0.338 |
