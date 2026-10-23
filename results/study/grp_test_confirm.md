# test_confirm.jsonl: 2706 questions, 13 datasets, groups of runs, 1000 bootstrap resamples

| Group | runs | Acc micro % | Acc macro % | ECE | NLL |
|---|---|---|---|---|---|
| narrow | 3 | 68.7 ± 0.4 [67.4, 70.1] | 70.9 ± 0.5 [69.5, 72.3] | 0.045 ± 0.005 [0.041, 0.060] | 0.93 ± 0.02 [0.89, 0.96] |
| broad | 3 | 70.1 ± 0.4 [68.7, 71.4] | 72.0 ± 0.5 [70.5, 73.3] | 0.050 ± 0.008 [0.045, 0.064] | 0.92 ± 0.04 [0.88, 0.96] |

Cells: mean over runs ± sd across runs [95% bootstrap interval of the mean].

Difference of group means vs `narrow` (positive accuracy better, negative ECE/NLL better):

| Group | d Acc micro (pts) | d Acc macro (pts) | d ECE | d NLL |
|---|---|---|---|---|
| broad | +1.3 [+0.5, +2.1] p=0.002 | +1.1 [+0.2, +1.8] p=0.014 | +0.004 [-0.006, +0.013] p=0.406 | -0.01 [-0.02, +0.01] p=0.364 |
