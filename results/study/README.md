# Study tables

Every table here was produced by `eval/compare.py`, `eval/compare_groups.py` or `eval/report_table.py` from the
per-question result files of the runs (stored in the Hugging Face results dataset, `study/records/`). Intervals are
95 % paired, dataset-stratified bootstrap intervals (1,000 resamples). See [../../docs/REPORT.md](../../docs/REPORT.md).

| File | What it compares |
|---|---|
| `final_vs_base_<collection>.md` | released model vs the narrow-mix model vs the base model with and without temperature scaling, one file per test collection (`test_more`, `test_fresh`, `test_zeroshot`, `test_manyopt`, `test_confirm`, and `test_confirm_le26` = only questions with at most 26 options) |
| `table_<collection>.md` | per-dataset accuracy, ECE and NLL of base, base + temperature scaling and the released model (the tables in the README) |
| `grp_<collection>.md` | 3 seeds of the original 12-source mix against 3 seeds of the broad mix (mean, spread between seeds, paired difference) |
| `grp_test_confirm_le26.md` | the confirmation set restricted to <= 26 options: base, base + temperature scaling, letter-readout head, narrow and broad seeds |
| `cmp1_<collection>.md` | stage 1: base vs base + temperature scaling vs the three narrow-mix seeds, their ensemble and option-order averaging |
| `cmp2_<collection>.md` | stage 2-3 (one seed each): narrow vs letter-readout head vs broad mix vs broad mix + option-count augmentation; `cmp2_manyopt.md` also has option-order averaging |
| `cmp3_confirm.md` | confirmation set, per dataset: base + temperature scaling, narrow, broad, letter head |
| `final_perm4_*.md` | released model with and without option-order averaging (`n_perm=4`) |

Run names: `base_raw` = base model, `base_T` = base model + temperature scaling, `narrow` / `seedN` = original 12-source
mix, `broad` = released recipe, `letters` = letter-readout head, `broadaug` = broad mix + option-count augmentation,
`perm4` = four option orders averaged, `ens3` = mean of three seeds.
