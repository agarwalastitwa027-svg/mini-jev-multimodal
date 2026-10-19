"""Publish a trained run to the Hugging Face Hub with a generated model card.

    hf auth login                      # once, with a token that can write to the repo
    python publish_hf.py --run runs/release --repo AstitwaAgarwal/multimodal-jev --results results
    python publish_hf.py ... --dry-run # build the upload folder and README only, upload nothing

What it does
  1. keeps the current `main` of an existing repo on a branch (default "v1") before replacing it;
  2. stages the run (LoRA adapters, answer head with calibrated temperatures, tokenizer, image processor,
     meta, calibration report) plus results tables, and optionally merged full weights (--merged);
  3. writes README.md (model card) from meta.json, mix_report.json, results/ and Decision Index scores;
  4. uploads the folder in one commit.
"""
import argparse
import json
import os
import shutil

GITHUB = "https://github.com/agarwalastitwa027-svg/mini-jev-multimodal"
RUN_FILES = ["head.pt", "head_calibrated.pt", "meta.json", "calibration.json", "reliability.png", "train_log.jsonl",
             "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
             "vocab.json", "merges.txt", "preprocessor_config.json", "video_preprocessor_config.json"]


def read(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f) if path.endswith(".json") else f.read()
    except (OSError, ValueError):
        return default


def di_summary(path):
    s = read(path)
    if not s:
        return None
    idx = s.get("index") or s.get("decision_index") or {}
    val = idx.get("index") if isinstance(idx, dict) else idx
    if val is None:
        val = (s.get("scores") or {}).get("balanced_skill")
    return {"index": val, "edition": s.get("edition"), "complete": s.get("complete"),
            "areas": {k: (v.get("skill") if isinstance(v, dict) else v) for k, v in (idx.get("areas") or {}).items()}
            if isinstance(idx, dict) else {}}


TEST_NAMES = {"test_text.jsonl": "Text, sources seen in training",
              "test_vision.jsonl": "Images, sources seen in training",
              "test_zeroshot.jsonl": "Zero-shot (sources never trained on)",
              "bench_scienceqa.jsonl": "ScienceQA test images (zero-shot)"}


def pct(x):
    return f"{100 * x:.1f}" if isinstance(x, (int, float)) else "–"


def calibration_table(cal):
    """Accuracy + calibration rows from calibration.json (validation and every test file)."""
    if not cal or not cal.get("test"):
        return []
    rows = [("Validation", cal.get("val", {}))] + [(TEST_NAMES.get(f, f), d) for f, d in cal["test"].items()]
    out = ["### Accuracy and calibration", "",
           "Per-type temperatures fitted on validation data. ECE is shown before → after calibration. "
           "Acc@conf≥0.9 is the accuracy on questions answered with confidence ≥ 0.9 (coverage = their share). "
           "Acc, Skill and coverage in %; Skill = (acc − chance) / (1 − chance).", "",
           "| Test set | Group | n | Acc | Skill | ECE | NLL | Acc@conf≥0.9 (coverage) |",
           "|---|---|---|---|---|---|---|---|"]
    for name, d in rows:
        before, after = d.get("before", {}), d.get("after", {})
        groups = [g for g in ("modality:text", "modality:vision") if g in after]
        for g in (["all"] if len(groups) < 2 else ["all"] + groups):
            a, b = after.get(g), before.get(g, {})
            if not a:
                continue
            label = "all" if g == "all" else g.split(":")[1]
            if g == "all" and len(groups) == 1:
                label = groups[0].split(":")[1]
            out.append(f"| {name} | {label} | {a.get('n')} | {pct(a.get('accuracy'))} | {pct(a.get('skill'))} | "
                       f"{b.get('ece', 0):.3f} → {a.get('ece', 0):.3f} | {a.get('nll', 0):.3f} | "
                       f"{pct(a.get('sel_acc@0.9'))} ({pct(a.get('coverage@0.9'))}) |")
    return out + [""]


def model_card(meta, mix, results_dir, cal, di_text, di_vision, repo, lora, decontam=None):
    base = meta.get("base_model", "Qwen/Qwen3-VL-2B-Instruct")
    ta = meta.get("train_args", {})
    sources = sorted({s.split(" (")[0] for s in (mix or {}).get("by_source", {})})
    abl = read(os.path.join(results_dir, "ablations.md"), "") if results_dir else ""
    lat = read(os.path.join(results_dir, "latency.md"), "") if results_dir else ""
    lines = ["---", "license: apache-2.0", f"base_model: {base}"]
    if lora:
        lines.append("library_name: peft")
    lines += ["tags:", "- decision-model", "- system-one", "- jev", "- typed-decisions", "- calibration",
              "- multimodal", "- vision-language", "language:", "- en", "---", "",
              "# mini-jev-multimodal (v2)", "",
              "A small **decision model** for text and images. Give it a state (text and/or up to 8 images) and "
              "typed questions; it returns a probability for every allowed option in one forward pass, with no "
              "text generation. It cannot answer outside the options you supply.", "",
              "An independent, open reproduction of the idea behind System One decision models such as TypeSafe "
              "AI's Jev. Not affiliated with or derived from that product.", "",
              f"Code, training and evaluation: <{GITHUB}>", "",
              "## How it works", "", "```",
              f"images + state ──► {base.split('/')[-1]} (prefill, LoRA-tuned, LM head removed) ──► KV cache (state encoded once)",
              "for each question:  suffix 'Question / Options / Answer:' ──► hidden states",
              "    q   = W_q · h[last token]        o_j = W_o · mean(h[option j])",
              "    p_j = softmax_j( q·o_j / √d / T[type] )      T fitted per type (choice / noul / score)",
              "```", "",
              "- **Types:** `choice` (2–255 options), `noul` (yes/no → P(yes)), `score` (ordered levels → "
              "probabilities + expected score).",
              "- **Shared state:** the images and state are encoded once and reused for every question in the "
              "request (exactly equal to re-reading them, see the repo tests).",
              "- **No truncation:** requests over the declared limit (16,384 tokens by default) are refused, not cut.",
              "", "## Use", "", "```bash", f"pip install git+{GITHUB}@v2", "```", "", "```python",
              "from jev.decide import Decider", f'jev = Decider.from_pretrained("{repo}")',
              "out = jev.decide(", '    state="My package arrived damaged and I want a refund.",', "    questions={",
              '        "queue":  {"type": "choice", "instructions": "Which queue?",',
              '                   "criteria": {"billing": "payments, refunds", "shipping": "deliveries"}},',
              '        "urgent": {"type": "noul", "instructions": "Is this urgent?"},',
              '        "stars":  {"type": "score", "instructions": "How satisfied is the customer?",',
              '                   "criteria": {"1": "very unhappy", "2": "", "3": "", "4": "", "5": "very happy"}},',
              "    },", '    images=["photo.jpg"],      # optional', ")", "```", "",
              f"Jev-compatible server: `python serve.py --run {repo}` → `POST /v1/systemone`.", "",
              "## Training", "",
              f"- Base: `{base}`, LoRA r={meta.get('lora_r')} on the text layers (vision tower frozen), answer head "
              f"d={meta.get('head_dim')}.",
              f"- Loss: {'soft cross-entropy + ranked probability score for ordinal questions' if ta.get('loss', 'v2') == 'v2' else 'cross-entropy'}"
              f"; option order shuffled every step; text share of steps {ta.get('text_frac', 0.5)}.",
              f"- {meta.get('total_steps', '?')} optimizer steps, batch {ta.get('bs', '?')} × accumulation "
              f"{ta.get('grad_accum', '?')}, lr {ta.get('lr', '?')}, images capped at {meta.get('max_pixels')} px "
              f"during training; {meta.get('train_minutes', '?')} min on {meta.get('gpu', 'one GPU')}"
              f"{', peak ' + str(meta['peak_vram_gb']) + ' GB VRAM' if meta.get('peak_vram_gb') else ''}.",
              "- Calibration: one temperature per question type fitted on validation data: "
              + (", ".join(f"{k} {v:.2f}" for k, v in (cal or {}).get("temperatures", {}).items()) or "see calibration.json")
              + ".", ""]
    if mix:
        q = mix.get("questions_by_modality", {}).get("train", {})
        lines += [f"- Data: {mix['questions'].get('train', '?')} training questions "
                  f"({q.get('text', 0)} text, {q.get('vision', 0)} image) from: {', '.join(sources)}.",
                  "- **Decontamination:** no split of any Jev Decision Index benchmark is used for training"
                  + (f"; training rows were checked against the Decision Index suites by 13-gram text overlap"
                     + (" and exact image hash" if decontam.get("suite_images") else
                        " (image hashes were not checked: the vision suite was not built)")
                     + f", and {decontam.get('total', 0) - decontam.get('kept', 0)} of "
                     f"{decontam.get('total', 0)} rows overlapped and were removed. The text check used a locally "
                     "rebuilt copy of the suite that is not byte-identical to the leaderboard's." if decontam else
                     "; the overlap check against the downloaded Decision Index suites has not been run yet."), ""]
    lines += ["## Results", ""]
    if di_text or di_vision:
        lines += ["### Jev Decision Index", "", "| Board | Score | Edition | Complete |", "|---|---|---|---|"]
        for name, d in (("Text", di_text), ("Vision (public half)", di_vision)):
            if d:
                v = d.get("index")
                lines.append(f"| {name} | {v:.2f} | {d.get('edition') or '–'} | {d.get('complete')} |" if isinstance(v, (int, float))
                             else f"| {name} | – | {d.get('edition') or '–'} | {d.get('complete')} |")
        lines.append("")
    lines += calibration_table(cal)
    if abl:
        lines += [abl.replace("# Ablations", "### Comparisons (own test sets)"), ""]
    if lat:
        lines += [lat.replace("# Latency", "### Latency"), ""]
    lines += ["## Limitations", "",
              "- Decides only among the options you give it; it cannot write, explain, or say 'none of these' unless "
              "that is an option.",
              "- Small (2B) backbone: weaker than large generative models on knowledge and multi-step reasoning.",
              "- Confidence is calibrated on its own validation distribution; check calibration on your data before "
              "using thresholds to act automatically.",
              "- Image inputs only (no audio or video).", "",
              "## Licence", "",
              f"Apache-2.0. The base model `{base}` is Apache-2.0. Training data come from public datasets under their "
              "own licences; images from them are not redistributed here.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", default="AstitwaAgarwal/multimodal-jev")
    ap.add_argument("--results", default="results")
    ap.add_argument("--mix-report", default="data/v2/mix_report.json")
    ap.add_argument("--di-text-scores", default=None, help="scores.json from the text board run")
    ap.add_argument("--di-vision-scores", default=None, help="vision-scores.json from the vision board run")
    ap.add_argument("--merged", action="store_true", help="also upload merged full weights (about 4.5 GB)")
    ap.add_argument("--keep-branch", default="v1", help="copy the repo's current main to this branch first ('' = skip)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--staging", default="runs/_hf_upload")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    meta = read(os.path.join(a.run, "meta.json"))
    if not meta:
        raise SystemExit(f"no meta.json in {a.run}")
    if os.path.exists(a.staging):
        shutil.rmtree(a.staging)
    os.makedirs(a.staging)
    for f in RUN_FILES:
        p = os.path.join(a.run, f)
        if os.path.exists(p):
            shutil.copy2(p, a.staging)
    lora = os.path.isdir(os.path.join(a.run, "lora"))
    if lora:
        shutil.copytree(os.path.join(a.run, "lora"), os.path.join(a.staging, "lora"))
    if a.merged:
        from jev.model import MiniJev
        m = MiniJev.load(a.run, merge=True)
        m.core().save_pretrained(os.path.join(a.staging, "merged"))
    if a.results and os.path.isdir(a.results):
        os.makedirs(os.path.join(a.staging, "results"), exist_ok=True)
        for f in ("ablations.md", "ablations.json", "latency.md", "latency.json"):
            if os.path.exists(os.path.join(a.results, f)):
                shutil.copy2(os.path.join(a.results, f), os.path.join(a.staging, "results"))
    mix = read(a.mix_report)
    if mix:
        shutil.copy2(a.mix_report, os.path.join(a.staging, "mix_report.json"))
    decontam = read(os.path.join(os.path.dirname(a.mix_report), "train_decontam.json"))
    card = model_card(meta, mix, a.results, read(os.path.join(a.run, "calibration.json")),
                      di_summary(a.di_text_scores) if a.di_text_scores else None,
                      di_summary(a.di_vision_scores) if a.di_vision_scores else None, a.repo, lora, decontam)
    open(os.path.join(a.staging, "README.md"), "w", encoding="utf-8").write(card)
    print(f"staged {a.staging}: {sorted(os.listdir(a.staging))}")
    if a.dry_run:
        print("dry run: nothing uploaded")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    exists = api.repo_exists(a.repo)
    if not exists:
        api.create_repo(a.repo, private=a.private)
    elif a.keep_branch:
        refs = {b.name for b in api.list_repo_refs(a.repo).branches}
        if a.keep_branch not in refs:
            api.create_branch(a.repo, branch=a.keep_branch, revision="main")
            print(f"kept previous main on branch '{a.keep_branch}'")
    old = {f for f in api.list_repo_files(a.repo)} if exists else set()
    new = {os.path.relpath(os.path.join(d, f), a.staging).replace(os.sep, "/")
           for d, _, fs in os.walk(a.staging) for f in fs}
    stale = [f for f in old - new if f != ".gitattributes"]
    api.upload_folder(folder_path=a.staging, repo_id=a.repo, commit_message="mini-jev-multimodal v2",
                      delete_patterns=stale or None)
    print(f"uploaded to https://huggingface.co/{a.repo}" + (f" (removed {len(stale)} v1 files from main)" if stale else ""))


if __name__ == "__main__":
    main()
