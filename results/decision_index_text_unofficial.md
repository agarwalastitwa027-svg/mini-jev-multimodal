# Decision Index text run (unofficial)

Model: the **earlier version** of `AstitwaAgarwal/multimodal-jev` (seed 0 of the original 12-source training mix),
**not the currently released broad-mix model, which was not re-run on the Decision Index.** Run with
`di_engine:MiniJevEngine`, `max_tokens=16384`, on an RTX 5070 Laptop (8 GB). Edition 0.2.1 of the kit at
github.com/apolinario/decision-index.

**This is not a leaderboard entry.** The official suite dataset is not downloadable, so the suite was rebuilt
from the pinned public sources with `suite rebuild`:
- The seven benchmarks added in 0.2 rebuilt byte-identical to the official hashes.
- The core 0.1 rows did **not** (sha256 differs), so the kit skipped the 0.2 cut and the 0.3 GSM8K rows. The run
  used the rebuilt 0.1 rows with verification off (`--no-verify`).
- **ToolRet-retrieval (8,032 rows) and BRIGHT-retrieval (1,384 rows) were not run**: on an 8 GB GPU they take
  minutes per request. They count as wrong, so the index is lower than the model would otherwise score. The
  retrieval and tools areas show coverage 0.82.

Results: 149,432 requests completed, all answered, no errors.

| | |
|---|---|
| Decision Index (chance-corrected, balanced) | **17.43** |
| Raw (uncorrected) | 36.76 |
| Latency (ms) | median 60.8, mean 241.5, p95 770.7 |

| Area | Skill | Raw | Coverage | Benchmarks |
|---|---|---|---|---|
| Tools & Automation | 25.5 | 32.0 | 0.82 | 5 |
| Language Understanding | 19.9 | 46.4 | 1.00 | 10 |
| Retrieval & Classification | 17.0 | 34.9 | 0.82 | 6 |
| Knowledge & Reasoning | 13.4 | 32.9 | 1.00 | 10 |
| Arts & Human Taste | 7.9 | 34.2 | 1.00 | 7 |

Skill = (score - chance) / (1 - chance), clipped to 0-1, then weighted by the kit; 0 is random guessing.
Per-benchmark scores are in `decision_index_text_scores_unofficial.json`. Vision board not run.

Full per-request results (233 MB): https://huggingface.co/datasets/AstitwaAgarwal/decision-index-results
