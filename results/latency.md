# Latency (NVIDIA GeForce RTX 5070 Laptop GPU, torch.bfloat16)

| Case | Questions | Cached p50 / p95 (ms) | Uncached p50 / p95 (ms) |
|---|---|---|---|
| text 100 tok | 1 | 39 / 43 | 37 / 39 |
| text 100 tok | 8 | 89 / 95 | 279 / 288 |
| text 1000 tok | 1 | 109 / 111 | 108 / 112 |
| text 1000 tok | 8 | 194 / 197 | 885 / 910 |
| text 4000 tok | 1 | 478 / 483 | 491 / 496 |
| text 4000 tok | 8 | 1077 / 1107 | 4758 / 4761 |
| image + 100 tok | 1 | 338 / 403 | 338 / 359 |
| image + 100 tok | 8 | 434 / 439 | 2585 / 2636 |

Peak VRAM: 7.43 GB
