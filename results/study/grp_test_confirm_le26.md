# test_confirm.jsonl: 2306 questions, 11 datasets, groups of runs, 1000 bootstrap resamples

| Group | runs | Acc micro % | Acc macro % | ECE | NLL |
|---|---|---|---|---|---|
| base_T | 1 | 69.2 [67.6, 70.7] | 70.6 [68.7, 72.3] | 0.099 [0.088, 0.117] | 0.96 [0.92, 1.00] |
| letters | 1 | 70.7 [69.2, 72.2] | 73.0 [71.3, 74.7] | 0.073 [0.061, 0.089] | 0.79 [0.76, 0.83] |
| narrow | 3 | 71.1 ± 0.6 [69.6, 72.6] | 73.9 ± 0.8 [72.3, 75.3] | 0.047 ± 0.007 [0.042, 0.062] | 0.80 ± 0.01 [0.76, 0.83] |
| broad | 3 | 72.1 ± 0.3 [70.8, 73.6] | 74.5 ± 0.4 [73.0, 76.0] | 0.051 ± 0.006 [0.046, 0.065] | 0.82 ± 0.04 [0.79, 0.86] |

Cells: mean over runs ± sd across runs [95% bootstrap interval of the mean].

Difference of group means vs `base_T` (positive accuracy better, negative ECE/NLL better):

| Group | d Acc micro (pts) | d Acc macro (pts) | d ECE | d NLL |
|---|---|---|---|---|
| letters | +1.6 [+0.1, +3.0] p=0.046 | +2.4 [+0.8, +4.2] p=0.004 | -0.026 [-0.045, -0.011] p=0.001 | -0.16 [-0.19, -0.13] p=0.001 |
| narrow | +2.0 [+0.6, +3.3] p=0.002 | +3.3 [+1.9, +4.8] p=0.001 | -0.052 [-0.066, -0.036] p=0.001 | -0.16 [-0.19, -0.13] p=0.001 |
| broad | +2.9 [+1.6, +4.2] p=0.001 | +3.9 [+2.5, +5.3] p=0.001 | -0.048 [-0.061, -0.032] p=0.001 | -0.13 [-0.17, -0.10] p=0.001 |
