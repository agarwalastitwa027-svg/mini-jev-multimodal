# test_confirm.jsonl: 2706 questions, 13 datasets, 1000 bootstrap resamples

| Run | Acc micro % [95% CI] | Acc macro % [95% CI] | ECE [95% CI] | NLL [95% CI] |
|---|---|---|---|---|
| released | 69.6 [68.0, 71.1] | 71.4 [69.8, 73.0] | 0.041 [0.033, 0.058] | 0.91 [0.87, 0.95] |
| released_perm4 | 70.0 [68.4, 71.5] | 71.8 [70.1, 73.3] | 0.034 [0.030, 0.054] | 0.89 [0.85, 0.92] |

Paired difference vs `released` (run - ref; positive accuracy is better, negative ECE/NLL is better). p = two-sided bootstrap p-value.

| Run | d Acc micro (pts) | d Acc macro (pts) | d ECE | d NLL |
|---|---|---|---|---|
| released_perm4 | +0.4 [-0.4, +1.1] p=0.406 | +0.3 [-0.5, +1.2] p=0.454 | -0.006 [-0.015, +0.008] p=0.526 | -0.02 [-0.03, -0.01] p=0.001 |

Per dataset (accuracy %, 95% CI):

| Dataset | n | released | released_perm4 |
|---|---|---|---|
| arc_easy | 250 | 94.8 [92.0, 97.2] | 95.2 [92.0, 97.6] |
| cb | 56 | 82.1 [71.4, 91.1] | 80.4 [69.6, 89.3] |
| copa | 100 | 89.0 [83.0, 95.0] | 92.0 [87.0, 97.0] |
| gtsrb | 200 | 54.0 [47.0, 61.0] | 57.0 [50.0, 64.0] |
| resisc45 | 200 | 58.0 [51.5, 64.5] | 59.0 [52.0, 65.5] |
| rotten_tomatoes | 250 | 88.8 [84.8, 92.4] | 88.8 [84.8, 92.4] |
| stl10 | 200 | 98.5 [96.5, 100.0] | 99.0 [97.5, 100.0] |
| svhn | 200 | 79.5 [74.0, 85.0] | 74.5 [68.5, 80.5] |
| tweet_emoji | 250 | 19.6 [15.2, 24.8] | 20.0 [15.6, 25.2] |
| tweet_emotion | 250 | 75.6 [70.0, 80.4] | 80.0 [74.8, 84.4] |
| tweet_sentiment | 250 | 72.4 [66.8, 78.0] | 71.2 [65.6, 76.8] |
| tweet_stance | 250 | 60.4 [53.6, 66.0] | 60.0 [53.2, 65.6] |
| wic | 250 | 56.0 [50.4, 62.0] | 56.0 [50.4, 62.0] |
