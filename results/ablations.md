# Ablations

Mean ± std across seeds where an entry has several runs. Acc, Skill, Acc@conf≥0.9 and Cov@0.9 in %. Skill = (acc − chance) / (1 − chance). Unanswered rows count as wrong.

### test_text — all

| Model | n | Acc | Skill | ECE | Brier | NLL | Acc@conf≥0.9 | Cov@0.9 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| A. Base Qwen3-VL-2B, zero-shot letters | 3085 | 79.6 | 68.3 | 0.177 | 0.370 | 1.591 | 83.1 | 92.1 | 67 |
| B. Head only (backbone frozen) | 3085 | 79.0 | 67.3 | 0.023 | 0.288 | 0.506 | 97.1 | 44.7 | 71 |
| C. Hard-label recipe (hard CE) | 3085 | 87.8 | 81.1 | 0.016 | 0.173 | 0.306 | 97.9 | 61.6 | 64 |
| D. This model (soft + ordinal loss, text+vision) | 3085 | 87.8 ± 0.3 | 81.0 ± 0.4 | 0.015 ± 0.007 | 0.178 ± 0.004 | 0.315 ± 0.008 | 98.0 ± 0.4 | 60.8 ± 1.1 | 65 ± 1 |
| E1. Trained on text only | 3085 | 88.0 | 81.4 | 0.012 | 0.169 | 0.296 | 97.4 | 66.5 | 67 |
| E2. Trained on vision only | 3085 | 75.7 | 62.2 | 0.039 | 0.330 | 0.586 | 95.9 | 40.8 | 67 |

### test_text — text

| Model | n | Acc | Skill | ECE | Brier | NLL | Acc@conf≥0.9 | Cov@0.9 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| A. Base Qwen3-VL-2B, zero-shot letters | 3085 | 79.6 | 68.3 | 0.177 | 0.370 | 1.591 | 83.1 | 92.1 | 67 |
| B. Head only (backbone frozen) | 3085 | 79.0 | 67.3 | 0.023 | 0.288 | 0.506 | 97.1 | 44.7 | 71 |
| C. Hard-label recipe (hard CE) | 3085 | 87.8 | 81.1 | 0.016 | 0.173 | 0.306 | 97.9 | 61.6 | 64 |
| D. This model (soft + ordinal loss, text+vision) | 3085 | 87.8 ± 0.3 | 81.0 ± 0.4 | 0.015 ± 0.007 | 0.178 ± 0.004 | 0.315 ± 0.008 | 98.0 ± 0.4 | 60.8 ± 1.1 | 65 ± 1 |
| E1. Trained on text only | 3085 | 88.0 | 81.4 | 0.012 | 0.169 | 0.296 | 97.4 | 66.5 | 67 |
| E2. Trained on vision only | 3085 | 75.7 | 62.2 | 0.039 | 0.330 | 0.586 | 95.9 | 40.8 | 67 |

### test_vision — all

| Model | n | Acc | Skill | ECE | Brier | NLL | Acc@conf≥0.9 | Cov@0.9 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| A. Base Qwen3-VL-2B, zero-shot letters | 886 | 80.6 | 73.1 | 0.103 | 0.289 | 0.670 | 91.8 | 75.7 | 136 |
| B. Head only (backbone frozen) | 886 | 79.1 | 71.0 | 0.040 | 0.278 | 0.513 | 97.3 | 53.4 | 124 |
| C. Hard-label recipe (hard CE) | 886 | 83.6 | 77.3 | 0.046 | 0.214 | 0.420 | 97.2 | 69.2 | 121 |
| D. This model (soft + ordinal loss, text+vision) | 886 | 83.5 ± 0.2 | 77.1 ± 0.3 | 0.039 ± 0.001 | 0.223 ± 0.002 | 0.427 ± 0.007 | 96.9 ± 0.4 | 66.3 ± 1.8 | 121 ± 1 |
| E1. Trained on text only | 886 | 78.6 | 70.3 | 0.043 | 0.288 | 0.574 | 94.6 | 47.9 | 123 |
| E2. Trained on vision only | 886 | 82.8 | 76.2 | 0.024 | 0.222 | 0.418 | 97.8 | 57.2 | 124 |

### test_vision — vision

| Model | n | Acc | Skill | ECE | Brier | NLL | Acc@conf≥0.9 | Cov@0.9 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| A. Base Qwen3-VL-2B, zero-shot letters | 886 | 80.6 | 73.1 | 0.103 | 0.289 | 0.670 | 91.8 | 75.7 | 136 |
| B. Head only (backbone frozen) | 886 | 79.1 | 71.0 | 0.040 | 0.278 | 0.513 | 97.3 | 53.4 | 124 |
| C. Hard-label recipe (hard CE) | 886 | 83.6 | 77.3 | 0.046 | 0.214 | 0.420 | 97.2 | 69.2 | 121 |
| D. This model (soft + ordinal loss, text+vision) | 886 | 83.5 ± 0.2 | 77.1 ± 0.3 | 0.039 ± 0.001 | 0.223 ± 0.002 | 0.427 ± 0.007 | 96.9 ± 0.4 | 66.3 ± 1.8 | 121 ± 1 |
| E1. Trained on text only | 886 | 78.6 | 70.3 | 0.043 | 0.288 | 0.574 | 94.6 | 47.9 | 123 |
| E2. Trained on vision only | 886 | 82.8 | 76.2 | 0.024 | 0.222 | 0.418 | 97.8 | 57.2 | 124 |

### zeroshot — all

| Model | n | Acc | Skill | ECE | Brier | NLL | Acc@conf≥0.9 | Cov@0.9 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| A. Base Qwen3-VL-2B, zero-shot letters | 720 | 68.3 | 59.6 | 0.266 | 0.565 | 2.364 | 74.3 | 85.3 | 38 |
| B. Head only (backbone frozen) | 720 | 56.7 | 44.8 | 0.060 | 0.556 | 1.098 | 88.7 | 18.5 | 33 |
| C. Hard-label recipe (hard CE) | 720 | 68.2 | 59.5 | 0.064 | 0.434 | 0.899 | 93.7 | 32.9 | 34 |
| D. This model (soft + ordinal loss, text+vision) | 720 | 68.8 ± 1.4 | 60.3 ± 1.8 | 0.058 ± 0.009 | 0.438 ± 0.014 | 0.892 ± 0.035 | 95.1 ± 0.3 | 31.3 ± 2.3 | 34 ± 1 |
| E1. Trained on text only | 720 | 69.2 | 60.7 | 0.089 | 0.452 | 1.024 | 86.4 | 40.8 | 33 |
| E2. Trained on vision only | 720 | 65.4 | 55.9 | 0.054 | 0.471 | 0.922 | 97.4 | 21.0 | 36 |

### zeroshot — text

| Model | n | Acc | Skill | ECE | Brier | NLL | Acc@conf≥0.9 | Cov@0.9 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| A. Base Qwen3-VL-2B, zero-shot letters | 600 | 62.3 | 52.4 | 0.317 | 0.672 | 2.825 | 68.3 | 82.7 | 38 |
| B. Head only (backbone frozen) | 600 | 49.3 | 36.0 | 0.077 | 0.646 | 1.276 | 70.8 | 8.0 | 33 |
| C. Hard-label recipe (hard CE) | 600 | 62.3 | 52.4 | 0.078 | 0.513 | 1.062 | 87.9 | 20.7 | 33 |
| D. This model (soft + ordinal loss, text+vision) | 600 | 63.1 ± 1.5 | 53.3 ± 1.9 | 0.071 ± 0.009 | 0.519 ± 0.015 | 1.056 ± 0.040 | 90.1 ± 1.0 | 18.7 ± 2.6 | 33 ± 1 |
| E1. Trained on text only | 600 | 64.0 | 54.5 | 0.111 | 0.529 | 1.204 | 79.3 | 31.3 | 32 |
| E2. Trained on vision only | 600 | 59.0 | 48.2 | 0.067 | 0.558 | 1.090 | 90.5 | 7.0 | 35 |

### zeroshot — vision

| Model | n | Acc | Skill | ECE | Brier | NLL | Acc@conf≥0.9 | Cov@0.9 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| A. Base Qwen3-VL-2B, zero-shot letters | 120 | 98.3 | 97.8 | 0.016 | 0.029 | 0.060 | 99.2 | 98.3 | 135 |
| B. Head only (backbone frozen) | 120 | 93.3 | 91.1 | 0.053 | 0.104 | 0.206 | 98.8 | 70.8 | 123 |
| C. Hard-label recipe (hard CE) | 120 | 97.5 | 96.7 | 0.024 | 0.040 | 0.085 | 100.0 | 94.2 | 121 |
| D. This model (soft + ordinal loss, text+vision) | 120 | 97.8 ± 1.0 | 97.0 ± 1.3 | 0.021 ± 0.005 | 0.033 ± 0.009 | 0.071 ± 0.013 | 100.0 | 94.4 ± 0.5 | 120 ± 1 |
| E1. Trained on text only | 120 | 95.0 | 93.3 | 0.035 | 0.067 | 0.124 | 99.1 | 88.3 | 122 |
| E2. Trained on vision only | 120 | 97.5 | 96.7 | 0.013 | 0.035 | 0.079 | 100.0 | 90.8 | 122 |

### scienceqa — all

| Model | n | Acc | Skill | ECE | Brier | NLL | Acc@conf≥0.9 | Cov@0.9 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| A. Base Qwen3-VL-2B, zero-shot letters | 300 | 90.3 | 84.9 | 0.069 | 0.162 | 0.371 | 94.3 | 88.0 | 107 |
| B. Head only (backbone frozen) | 300 | 79.3 | 67.7 | 0.047 | 0.266 | 0.453 | 97.4 | 38.3 | 94 |
| C. Hard-label recipe (hard CE) | 300 | 90.3 | 84.9 | 0.040 | 0.144 | 0.266 | 97.6 | 70.7 | 92 |
| D. This model (soft + ordinal loss, text+vision) | 300 | 89.9 ± 0.7 | 84.2 ± 1.1 | 0.047 ± 0.009 | 0.153 ± 0.006 | 0.267 ± 0.020 | 98.0 ± 0.4 | 67.9 ± 1.9 | 92 ± 1 |
| E1. Trained on text only | 300 | 80.3 | 69.2 | 0.057 | 0.264 | 0.458 | 94.0 | 49.7 | 95 |
| E2. Trained on vision only | 300 | 85.7 | 77.6 | 0.056 | 0.188 | 0.325 | 97.6 | 55.0 | 95 |

### scienceqa — vision

| Model | n | Acc | Skill | ECE | Brier | NLL | Acc@conf≥0.9 | Cov@0.9 | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| A. Base Qwen3-VL-2B, zero-shot letters | 300 | 90.3 | 84.9 | 0.069 | 0.162 | 0.371 | 94.3 | 88.0 | 107 |
| B. Head only (backbone frozen) | 300 | 79.3 | 67.7 | 0.047 | 0.266 | 0.453 | 97.4 | 38.3 | 94 |
| C. Hard-label recipe (hard CE) | 300 | 90.3 | 84.9 | 0.040 | 0.144 | 0.266 | 97.6 | 70.7 | 92 |
| D. This model (soft + ordinal loss, text+vision) | 300 | 89.9 ± 0.7 | 84.2 ± 1.1 | 0.047 ± 0.009 | 0.153 ± 0.006 | 0.267 ± 0.020 | 98.0 ± 0.4 | 67.9 ± 1.9 | 92 ± 1 |
| E1. Trained on text only | 300 | 80.3 | 69.2 | 0.057 | 0.264 | 0.458 | 94.0 | 49.7 | 95 |
| E2. Trained on vision only | 300 | 85.7 | 77.6 | 0.056 | 0.188 | 0.325 | 97.6 | 55.0 | 95 |
