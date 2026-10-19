# Publishing and the Jev Decision Index

## 1. Hugging Face model

```bash
hf auth login                                   # token with write access to AstitwaAgarwal/*
python run_all.py --steps publish               # = python publish_hf.py --run runs/release --repo AstitwaAgarwal/multimodal-jev
```

`publish_hf.py` first copies the repo's current `main` (the v1 weights) to a `v1` branch, then uploads
v2 to `main` with a generated model card (training data, recipe, calibration, comparison and latency
tables, leaderboard scores when available). `--dry-run` builds the folder and card without uploading.
`--merged` also uploads merged full weights (~4.5 GB) so loading needs no PEFT.

## 2. Text board

The leaderboard re-runs your model from the Hub, so publish first.

```bash
git clone https://github.com/apolinario/decision-index ../decision-index
cd ../decision-index && pip install -e ".[transformers]" && cd -
# Windows:  set HF_HUB_DISABLE_XET=1      Linux/macOS: export HF_HUB_DISABLE_XET=1
python run_all.py --steps di_text --di-kit ../decision-index
```

This runs, inside the kit:

```bash
python -m decision_index pipeline --engine di_engine:MiniJevEngine \
  --option run=AstitwaAgarwal/multimodal-jev --option max_tokens=16384 \
  --out runs/mini-jev-multimodal-v2 --compact --upload AstitwaAgarwal/decision-index-results --public
```

- About 140k requests. On an RTX 5070 Laptop expect many hours (most requests are short; a few are
  long). Runs checkpoint to `results.jsonl` and resume, so you can stop and continue.
- Requests over 16,384 tokens or that do not fit in 8 GB are recorded as `unsupported` (scored wrong),
  never truncated. A bigger GPU can declare a larger `max_tokens`.
- Alternative: the kit's `hf-job` command runs the whole pipeline as one Hugging Face Job on an RTX PRO 6000
  (paid). A custom engine needs this repo installed inside the job, so check `python -m decision_index hf-job --help`
  for how to add packages before relying on it; the local run above needs nothing extra.

## 3. Vision board

```bash
cd ../decision-index
pip install -e ".[vision]"
hf auth login                       # accept the Winoground terms on the Hub first (gated)
decision-index suite rebuild --vision --work work --font "C:\Windows\Fonts\arialbd.ttf"   # Windows path to Arial Bold
decision-index vision verify --dir work/vision
cd -
python run_all.py --steps di_vision --di-kit ../decision-index
```

Images run at native resolution capped at 1,638,400 pixels (the board's rule); our engine's default
`max_pixels` is exactly that. The board also requires median, mean and p80 latency ≤ 1,000 ms per request
on one RTX PRO 6000; a 2B model is far below that.

## 4. Pull request

Fork `apolinario/decision-index`, add one line to `submissions/README.md` (create it if missing), and open
a PR. `run_all.py` prints the line; it contains:

- model name and Hub repo: `mini-jev-multimodal-v2`, https://huggingface.co/AstitwaAgarwal/multimodal-jev
- results dataset: https://huggingface.co/datasets/AstitwaAgarwal/decision-index-results (`runs/<name>/scores.json`, `"complete": true`)
- engine and commit: `di_engine:MiniJevEngine` from this repo at the exact commit you ran
- hardware: NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB
- declared limits: `max_tokens=16384`, 255 options per question, 8 images
- vision: mark the line **vision**, give the results link for `vision-scores.json`, and state
  "native resolution up to 1.6 MP"

The maintainers re-score the results file, and for vision they run the private half themselves from
the Hub weights and this engine.
