# Paper

**What Does Fine-Tuning Buy a Small Vision-Language Decision Model? A controlled study against calibrated baselines.**

- [mini_jev_paper.docx](mini_jev_paper.docx): the manuscript (17 pages: 7 sections, 4 figures, 12 tables, 67 references,
  five appendices).
- [mini_jev_paper.pdf](mini_jev_paper.pdf): the same document exported from Word, for readers without Word.
- `figures/`: the four figures (PNG, 300 dpi).

Every number in the paper comes from the result files in [../results/](../results/) (summaries) and from the per-question
records in the Hugging Face dataset [AstitwaAgarwal/decision-index-results](https://huggingface.co/datasets/AstitwaAgarwal/decision-index-results)
(`study/records/`).

## Rebuild

```bash
pip install python-docx matplotlib            # in the project's virtual environment
# put the per-question records where the scripts expect them (they are gitignored):
huggingface-cli download AstitwaAgarwal/decision-index-results --repo-type dataset --include "study/records/*" --local-dir /tmp/rec
mkdir -p results/exp && cp /tmp/rec/study/records/*.json results/exp/

python paper/make_figures.py        # figures from results/exp and results/study
python paper/build_paper.py         # paper/mini_jev_paper.docx
python paper/verify_paper.py        # structure, leftover markup, citation numbering, text overlap
```

`verify_paper.py` checks that no markup is left, that every reference is listed and cited, and measures how much of the
text overlaps with this repository's own README and report (1.8 % when the paper was built). It is not a plagiarism service;
run your institution's checker before submitting.

## Before submitting

- The related-work arXiv references were checked against their abstract pages; the dataset citations in Appendix C were
  written from knowledge of the original papers and should be spot-checked.
- The paper's claims are deliberately conservative (see Section 6); it is an empirical study, not a new method.
- The authorship, affiliation and the AI-assistance statement should be reviewed and edited by the author.

## Manager copy

[manager_copy/](manager_copy/) holds `Paper_Abstract_and_Summary.docx` / `.pdf`: a four-page plain-language abstract for readers who only
need the research (the problems with the untouched model and with letter labels, the answer-picking part, results by number of choices and on unseen datasets,
temperature scaling with a calibration graph, and next steps), with no links or code references.
