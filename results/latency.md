# Latency (NVIDIA GeForce RTX 5070 Laptop GPU, torch.bfloat16)

| Case | Questions | Cached p50 / p95 (ms) | Uncached p50 / p95 (ms) |
|---|---|---|---|
| text 100 tok | 1 | 43 / 55 | 44 / 48 |
| text 100 tok | 8 | 108 / 131 | 325 / 333 |
| text 1000 tok | 1 | 147 / 150 | 147 / 149 |
| text 1000 tok | 8 | 254 / 258 | 1176 / 1178 |
| text 4000 tok | 1 | 596 / 600 | 594 / 597 |
| text 4000 tok | 8 | 1197 / 1210 | 4707 / 4725 |
| image + 100 tok | 1 | 333 / 422 | 340 / 350 |
| image + 100 tok | 8 | 440 / 446 | 2647 / 2656 |

Peak VRAM: 7.43 GB
