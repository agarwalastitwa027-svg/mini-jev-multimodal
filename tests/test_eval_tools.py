"""Tests for the comparison / ensemble tools on small synthetic result files (no model needed)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def records(acc_a, acc_b, n=80, seed=0, temp=None):
    """Two datasets of n yes/no questions; the model is right with the given rates, confident at 0.9."""
    rng = np.random.default_rng(seed)
    out = []
    for src, acc in (("a", acc_a), ("b", acc_b)):
        for i in range(n):
            ok = rng.random() < acc
            p = [0.9, 0.1] if ok else [0.1, 0.9]
            out.append({"id": f"{src}{i}", "key": "q", "source": src, "modality": "text", "type": "noul",
                        "n_options": 2, "gold": 0, "status": "ok", "probs": p, "ms": 1.0, "row_ms": 1.0})
    return {"predictor": "fake", "temps": temp, "files": {"t.jsonl": {"records": out}}}


@pytest.fixture()
def runs(tmp_path):
    paths = {}
    for name, (a, b, seed) in {"weak": (0.55, 0.60, 1), "strong": (0.85, 0.90, 2), "strong2": (0.85, 0.88, 3)}.items():
        paths[name] = str(tmp_path / f"{name}.json")
        json.dump(records(a, b, seed=seed), open(paths[name], "w"))
    return paths


def run(*args):
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def test_compare_detects_a_real_difference(runs, tmp_path):
    out = run("eval.compare", "--file", "t.jsonl", "--ref", "weak", "--boot", "300",
              "--runs", f"weak={runs['weak']}", f"strong={runs['strong']}")
    row = [l for l in out.splitlines() if l.startswith("| strong |") and "p=" in l][0]
    assert "p=0.00" in row and "+3" in row.split("|")[2]  # roughly +30 points, significant


def test_compare_groups(runs):
    out = run("eval.compare_groups", "--file", "t.jsonl", "--ref", "w", "--boot", "300",
              "--groups", f"w={runs['weak']}", f"s={runs['strong']},{runs['strong2']}")
    assert "| s | 2 |" in out and "| w | 1 |" in out
    diff = [l for l in out.splitlines() if l.startswith("| s |") and "p=" in l][0]
    assert "p=0.00" in diff


def test_max_options_filter_drops_questions(runs):
    from eval.compare import load
    assert len(load(runs["weak"], "t.jsonl", max_options=2)) == 160
    assert len(load(runs["weak"], "t.jsonl", max_options=1)) == 0


def test_temperature_is_applied_and_raw_ignores_it(tmp_path):
    from eval.compare import load
    p = str(tmp_path / "t.json")
    json.dump(records(0.9, 0.9, temp={"noul": 3.0}), open(p, "w"))
    scaled = load(p, "t.jsonl")
    raw = load(p + "@raw", "t.jsonl")
    k = next(iter(raw))
    assert raw[k][2][0] in (0.9, 0.1) and abs(scaled[k][2][0] - raw[k][2][0]) > 0.05
    assert abs(sum(scaled[k][2]) - 1) < 1e-9


def test_ensemble_averages_probabilities(runs, tmp_path):
    out = str(tmp_path / "ens.json")
    run("eval.ensemble", "--out", out, runs["weak"], runs["strong"])
    d = json.load(open(out))["files"]["t.jsonl"]["records"]
    w = json.load(open(runs["weak"]))["files"]["t.jsonl"]["records"]
    s = json.load(open(runs["strong"]))["files"]["t.jsonl"]["records"]
    for r, a, b in list(zip(d, w, s))[:20]:
        assert r["probs"] == pytest.approx([(x + y) / 2 for x, y in zip(a["probs"], b["probs"])])


def test_report_table_has_every_dataset(runs, tmp_path):
    out = run("eval.report_table", "--file", "t.jsonl", "--calib", "1", "--runs", f"W={runs['weak']}", f"S={runs['strong']}")
    assert "| a |" in out and "| b |" in out and "**ALL**" in out and "S ECE" in out
