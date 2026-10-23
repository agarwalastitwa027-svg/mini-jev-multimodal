"""The whole pipeline on one machine, step by step (Windows, Linux or macOS).

    python run_all.py --dry-run                       # print every command, run nothing
    python run_all.py --steps data train calibrate    # just these steps
    python run_all.py                                 # everything except publish / leaderboard
    python run_all.py --steps publish di_text di_vision --di-kit ../decision-index

Steps (in order)
  data        build text + vision sources, mix, split, decontaminate      ~1-2 h (downloads)
  train       full model x seeds, plus ablations B, C, E1, E2                      ~1.5 h per run on an RTX 5070 Laptop
  calibrate   per-type temperatures + test metrics for every run
  ablations   comparison table (base zero-shot, ablations, Ollama models)  results/ablations.md
  latency     cached vs uncached latency + VRAM for the best run        results/latency.md
  release     copy the best seed (lowest validation NLL) to runs/release
  publish     upload runs/release to the Hugging Face repo (needs `hf auth login`)
  di_text     full Decision Index text suite with di_engine (needs --di-kit; many hours)
  di_vision   Decision Index vision board, public half (needs the rebuilt vision suite)

Every step skips work that is already done (existing run folders, results files), so it can be
re-run after an interruption.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["data", "train", "calibrate", "ablations", "latency", "release", "publish", "di_text", "di_vision"]
DEFAULT = ["data", "train", "calibrate", "ablations", "latency", "release"]
VARIANTS = {  # name: extra train.py arguments
    "full": [],
    "headonly": ["--lora-r", "0"],
    "hardlabel": ["--loss", "hard"],
    "textonly": ["--modalities", "text"],
    "visiononly": ["--modalities", "vision"],
}


def prefix(a):
    return "broad" if a.mix == "broad" else "full"


def mix_files(a):
    """(train, val) for the selected mix. The broad mix prefers the decontaminated train file when it exists."""
    if a.mix == "broad":
        clean = "data/broad/train_clean.jsonl"
        return (clean if os.path.exists(clean) else "data/broad/train.jsonl"), "data/broad/val.jsonl"
    return "data/mix/train.jsonl", "data/mix/val.jsonl"


class Runner:
    def __init__(self, dry, env=None):
        self.dry, self.env = dry, {**os.environ, **(env or {})}

    def __call__(self, cmd, cwd=HERE, env=None, check=True):
        shown = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
        print(f"\n$ {shown}" + (f"   (in {cwd})" if cwd != HERE else ""), flush=True)
        if self.dry:
            return 0
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, env={**self.env, **(env or {})})
        if check and r.returncode:
            raise SystemExit(f"step failed ({r.returncode}): {shown}")
        return r.returncode


def step_data(run, a):
    src = "data/mix/src"
    os.makedirs(src, exist_ok=True)
    # procedural charts/shapes: draw the images the included teacher-enriched questions refer to
    run([PY, "data/make_images.py", "--n", "500", "--seed", "1", "--out", "data/mm_procedural.jsonl",
         "--img-dir", "data/images/proc"])
    run([PY, "data/convert_legacy.py", "data/synthetic/mm_enriched.jsonl", f"{src}/procedural.jsonl", "--modality", "vision"])
    # Food-101: seen classes for training, held-out classes as a zero-shot test
    run([PY, "data/make_public_mm.py"])
    run([PY, "data/convert_legacy.py", "data/mm_food_train.jsonl", f"{src}/food101.jsonl", "--modality", "vision"])
    run([PY, "data/convert_legacy.py", "data/mm_food_zeroshot.jsonl", f"{src}/food101_zeroshot.jsonl",
         "--modality", "vision", "--split", "zeroshot"])
    run([PY, "data/text_public.py", "--out", f"{src}/text_public.jsonl", "--per-dataset", str(a.per_dataset)])
    run([PY, "data/vision_public.py", "--out", f"{src}/vision_public.jsonl", "--per-dataset", str(a.per_dataset)])
    run([PY, "data/vision_public.py", "--bench-only", "--out", "data/mix/bench_scienceqa.jsonl"])
    run([PY, "data/build_mix.py", "--inputs", f"{src}/*.jsonl", "--out", "data/mix"])
    # held-out test sets that no training mix contains (see docs/REPORT.md)
    run([PY, "data/make_fresh_tests.py", "data/mix/test_fresh.jsonl"])
    run([PY, "data/make_more_tests.py", "data/mix/test_more.jsonl"])
    run([PY, "data/make_manyopt_tests.py", "data/mix/test_manyopt.jsonl"])
    run([PY, "data/make_confirm_tests.py", "data/mix/test_confirm.jsonl"])
    if a.mix == "broad":  # the released recipe: 11 more sources, whole task families held out
        run([PY, "data/broad_public.py", "--out", "data/broad/src/broad_public.jsonl", "--per-dataset", "1000"])
        run([PY, "data/build_mix.py", "--inputs", f"{src}/*.jsonl", "data/broad/src/*.jsonl", "--out", "data/broad"])
    suite = os.path.join(a.di_kit, "suite-0.3") if a.di_kit else None
    if suite and (os.path.isdir(suite) or a.dry_run):
        tr = "data/broad/train.jsonl" if a.mix == "broad" else "data/mix/train.jsonl"
        out_tr = "data/broad/train_clean.jsonl" if a.mix == "broad" else tr
        cmd = [PY, "data/decontam.py", "--train", tr, "--suite-dir", suite, "--out", out_tr]
        vis = os.path.join(a.di_kit, "work", "vision")
        if os.path.isdir(vis):
            cmd += ["--vision-dir", vis]
        run(cmd)
    else:
        print("\n[note] decontamination skipped. To check training data against the leaderboard suite, run "
              "`python -m decision_index suite download --dir suite-0.3` inside a clone of "
              "github.com/apolinario/decision-index, then re-run this step with --di-kit <that clone>.")


def run_dirs(a):
    out = [(f"{prefix(a)}_seed{s}", "full", s) for s in a.seeds]
    for v in ("headonly", "hardlabel", "textonly", "visiononly"):
        out += [(f"{v}_seed{s}", v, s) for s in a.ablation_seeds]
    return out


def step_train(run, a):
    for name, variant, seed in run_dirs(a):
        out = f"runs/{name}"
        if os.path.exists(os.path.join(out, "meta.json")) and not a.dry_run:
            print(f"[done] {out}")
            continue
        tr, va = mix_files(a)
        run([PY, "train.py", "--model", a.model, "--train", tr, "--val", va,
             "--out", out, "--seed", str(seed), "--epochs", str(a.epochs)] + VARIANTS[variant] + a.train_extra)


def tests():
    return [f for f in ("data/mix/test_text.jsonl", "data/mix/test_vision.jsonl", "data/mix/test_zeroshot.jsonl",
                        "data/mix/bench_scienceqa.jsonl") if os.path.exists(f)]


def step_calibrate(run, a):
    for name, _, _ in run_dirs(a):
        d = f"runs/{name}"
        if not a.dry_run and (not os.path.exists(os.path.join(d, "meta.json")) or os.path.exists(os.path.join(d, "calibration.json"))):
            continue
        run([PY, "calibrate.py", "--run", d, "--val", mix_files(a)[1], "--test"] + (tests() or ["data/mix/test_text.jsonl"]))


def step_ablations(run, a):
    run([PY, "-m", "eval.ablations", "--config", "eval/ablations.json", "--out", "results"])


def best_run(a):
    best, score = None, None
    for s in a.seeds:
        d = f"runs/{prefix(a)}_seed{s}"
        cal = os.path.join(d, "calibration.json")
        if os.path.exists(cal):
            nll = json.load(open(cal))["val"]["after"]["all"]["nll"]
            if score is None or nll < score:
                best, score = d, nll
    return best or f"runs/{prefix(a)}_seed{a.seeds[0]}"


def step_latency(run, a):
    run([PY, "-m", "eval.latency", "--run", best_run(a), "--out", "results/latency.json"])


def step_release(run, a):
    src = best_run(a)
    print(f"\nrelease <- {src}")
    if not a.dry_run:
        if os.path.exists("runs/release"):
            shutil.rmtree("runs/release")
        shutil.copytree(src, "runs/release")


def step_publish(run, a):
    cmd = [PY, "publish_hf.py", "--run", "runs/release", "--repo", a.hf_repo, "--results", "results"]
    if a.mix == "broad":
        cmd += ["--mix-report", "data/broad/mix_report.json"]
    t = os.path.join(a.di_kit or "", "runs", a.di_name, "scores.json")
    v = os.path.join(a.di_kit or "", "runs", f"vision-{a.di_name}", "vision-scores.json")
    if a.di_kit and os.path.exists(t):
        cmd += ["--di-text-scores", t]
    if a.di_kit and os.path.exists(v):
        cmd += ["--di-vision-scores", v]
    run(cmd + a.publish_extra)


def kit_env():
    return {"PYTHONPATH": HERE + os.pathsep + os.environ.get("PYTHONPATH", ""), "HF_HUB_DISABLE_XET": "1"}


def step_di_text(run, a):
    if not a.di_kit:
        raise SystemExit("--di-kit is required (a clone of https://github.com/apolinario/decision-index, "
                         "installed with: pip install -e .)")
    cmd = [PY, "-m", "decision_index", "pipeline", "--engine", "di_engine:MiniJevEngine",
           "--option", f"run={a.release_ref}", "--option", f"max_tokens={a.max_tokens}",
           "--out", f"runs/{a.di_name}", "--compact"]
    if a.di_results_repo:  # public, so the maintainers can open and re-score it
        cmd += ["--upload", a.di_results_repo, "--public"]
    run(cmd, cwd=a.di_kit, env=kit_env())


def step_di_vision(run, a):
    if not a.di_kit:
        raise SystemExit("--di-kit is required")
    vis = os.path.join(a.di_kit, "work", "vision")
    if not os.path.isdir(vis) and not a.dry_run:
        font = r"C:\Windows\Fonts\arialbd.ttf" if os.name == "nt" else "/path/to/Arial Bold.ttf"
        raise SystemExit("build the vision suite first (inside the kit):\n"
                         "  pip install -e .[vision]\n  hf auth login   (accept the Winoground terms on the Hub first)\n"
                         f"  decision-index suite rebuild --vision --work work --font \"{font}\"\n"
                         "  decision-index vision verify --dir work/vision")
    run(["decision-index", "vision", "pipeline", "--dir", "work/vision", "--engine", "di_engine:MiniJevEngine",
         "--option", f"run={a.release_ref}", "--option", "max_pixels=1638400",
         "--out", f"runs/vision-{a.di_name}"], cwd=a.di_kit, env=kit_env())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mix", choices=["broad", "narrow"], default="broad",
                    help="broad = the released recipe (more sources); narrow = the original 12-source mix used for "
                         "the ablation study (eval/ablations.json refers to its runs)")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="seeds for the main runs")
    ap.add_argument("--ablation-seeds", nargs="+", type=int, default=[0], help="seeds for ablations B/C/E")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per-dataset", type=int, default=1500, help="training rows per public dataset")
    ap.add_argument("--train-extra", nargs=argparse.REMAINDER, default=[], help="extra train.py args (put last)")
    ap.add_argument("--hf-repo", default="AstitwaAgarwal/multimodal-jev")
    ap.add_argument("--publish-extra", nargs="*", default=[])
    ap.add_argument("--di-kit", default=None, help="path to a clone of apolinario/decision-index")
    ap.add_argument("--di-name", default="mini-jev-multimodal")
    ap.add_argument("--di-results-repo", default="AstitwaAgarwal/decision-index-results")
    ap.add_argument("--release-ref", default=None,
                    help="what the leaderboard engine loads (default: the Hub repo, so maintainers can rerun it)")
    ap.add_argument("--max-tokens", type=int, default=16384, help="declared context limit for the leaderboard run")
    a = ap.parse_args()
    a.release_ref = a.release_ref or a.hf_repo
    if a.di_kit:
        a.di_kit = os.path.abspath(a.di_kit)
    os.chdir(HERE)
    run = Runner(a.dry_run)
    for s in STEPS:
        if s in a.steps:
            print(f"\n===== {s} =====", flush=True)
            globals()[f"step_{s}"](run, a)
    if "di_text" in a.steps or "di_vision" in a.steps:
        print("\nNext: open a pull request on https://github.com/apolinario/decision-index adding to "
              "submissions/README.md:\n"
              f"  {a.di_name} | weights: https://huggingface.co/{a.hf_repo} | results: "
              f"https://huggingface.co/datasets/{a.di_results_repo} | engine: di_engine:MiniJevEngine "
              f"(https://github.com/agarwalastitwa027-svg/mini-jev-multimodal, commit <sha>) | "
              f"hardware: NVIDIA GeForce RTX 5070 Laptop GPU 8 GB | declared limits: max_tokens={a.max_tokens}, "
              "255 options, 8 images | images: native up to 1.6 MP")


if __name__ == "__main__":
    main()
