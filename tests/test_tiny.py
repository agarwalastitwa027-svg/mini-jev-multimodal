"""End-to-end tests on a tiny random Qwen3-VL (CPU, no downloads). Run: pytest -q"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from jev.decide import Decider  # noqa: E402
from jev.encode import load_tokenizer, load_vision  # noqa: E402
from jev.model import MiniJev  # noqa: E402
from jev.schema import Unsupported, answer_from_probs, target_distribution  # noqa: E402

QS = {
    "queue": {"type": "choice", "instructions": "Which queue?",
              "criteria": {"billing": "money", "shipping": "packages", "general": ""}},
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "stars": {"type": "score", "instructions": "How satisfied?",
              "criteria": {"1": "bad", "2": "", "3": "", "4": "", "5": "great"}},
}


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    from tiny_model import build_tiny
    from make_tiny_data import make
    base = build_tiny(str(tmp_path_factory.mktemp("tiny")))
    data = make(str(tmp_path_factory.mktemp("data")))
    return base, data


@pytest.fixture(scope="session")
def trained(tiny, tmp_path_factory):
    base, data = tiny
    out = str(tmp_path_factory.mktemp("run"))
    cmd = [sys.executable, "train.py", "--model", base, "--train", data, "--val", data, "--out", out,
           "--max-steps", "80", "--bs", "8", "--grad-accum", "1", "--no-grad-ckpt", "--eval-every", "20",
           "--lora-r", "8", "--head-dim", "32", "--lr", "3e-3", "--head-lr", "3e-3"]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    subprocess.run([sys.executable, "calibrate.py", "--run", out, "--val", data, "--min-per-type", "5"],
                   cwd=ROOT, check=True, capture_output=True)
    return out


def decider(base, **kw):
    torch.manual_seed(0)
    m = MiniJev.create(base, lora_r=4, head_dim=32, grad_ckpt=False, dtype=torch.float32).eval()
    return Decider(m, load_tokenizer(base), load_vision(base, 64 * 32 * 32), torch.device("cpu"), torch.float32, **kw)


def flat(r):
    out = []
    for v in r["answers"].values():
        out += [v["noul"]] if v["type"] == "noul" else list(v["probabilities"].values())
    return np.array(out)


def image(path, color):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (96, 64), "white")  # same layout as tests/make_tiny_data.py
    ImageDraw.Draw(im).rectangle([20, 12, 60, 52], fill=color)
    im.save(path)
    return path


def test_cached_equals_full(tiny, tmp_path):
    base, _ = tiny
    a = image(str(tmp_path / "a.png"), (200, 30, 30))
    d_cached = decider(base, cache_prefix=True, token_budget=10**6)
    d_chunk1 = decider(base, cache_prefix=True, token_budget=1)
    d_full = decider(base, cache_prefix=False)
    for imgs in ([], [a], [a, a]):
        ref = flat(d_full.decide("My package arrived damaged.", QS, imgs))
        assert np.abs(flat(d_cached.decide("My package arrived damaged.", QS, imgs)) - ref).max() < 1e-5
        assert np.abs(flat(d_chunk1.decide("My package arrived damaged.", QS, imgs)) - ref).max() < 1e-5


def test_answers_are_valid_distributions(tiny):
    base, _ = tiny
    out = decider(base).decide({"order": 42, "text": "late"}, QS)
    for k, v in out["answers"].items():
        if v["type"] == "noul":
            assert 0 <= v["noul"] <= 1
        else:
            assert set(v["probabilities"]) == set(QS[k]["criteria"])
            assert abs(sum(v["probabilities"].values()) - 1) < 1e-4
            assert v["choice"] in QS[k]["criteria"]
    assert 1 <= out["answers"]["stars"]["score"] <= 5


def test_limits_raise_unsupported(tiny):
    base, _ = tiny
    d = decider(base, max_tokens=120)
    with pytest.raises(Unsupported, match="maximum context length"):
        d.decide("word " * 200, QS)
    with pytest.raises(Unsupported, match="options per choice"):
        d.decide("x", {"q": {"type": "choice", "instructions": "?", "criteria": {str(i): "" for i in range(300)}}})
    with pytest.raises(Unsupported):
        d.decide("x", {"q": {"type": "ranking", "instructions": "?"}})


def test_training_learns_and_images_matter(trained, tmp_path):
    d = Decider.from_pretrained(trained, device="cpu")
    qs = {"color": {"type": "choice", "instructions": "What colour is the square?",
                    "criteria": {"red": "red", "green": "green", "blue": "blue"}}}
    red = d.decide("", qs, [image(str(tmp_path / "r.png"), (220, 30, 30))])["answers"]["color"]
    blue = d.decide("", qs, [image(str(tmp_path / "b.png"), (30, 30, 220))])["answers"]["color"]
    assert red["choice"] == "red" and blue["choice"] == "blue"
    cal = json.load(open(os.path.join(trained, "calibration.json")))
    assert set(cal["temperatures"]) == {"choice", "noul", "score"}
    assert os.path.exists(os.path.join(trained, "head_calibrated.pt"))


def test_server_and_di_engine(trained, tiny):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from serve import build_app
    _, data = tiny
    row = json.loads(open(data).readlines()[1])
    c = TestClient(build_app(Decider.from_pretrained(trained, device="cpu", max_tokens=200)))
    r = c.post("/v1/systemone", json={"model": "x", "state": "", "questions": row["questions"], "images": row["images"]})
    assert r.status_code == 200 and set(r.json()["answers"]) == set(row["questions"])
    r = c.post("/v1/systemone", json={"model": "x", "state": "w " * 400, "questions": QS})
    assert r.status_code == 413 and "maximum context length" in r.json()["error"]["message"]

    from di_engine import MiniJevEngine
    e = MiniJevEngine(run=trained, device="cpu")
    resp, _ = e("", row["questions"], images=row["images"], prompt={"options_in_image": False, "readout": None})
    try:
        from decision_index.engines.base import validate
    except ImportError:
        validate = None
    if validate:
        validate(row["questions"], resp)


def test_schema_helpers():
    q = QS["stars"]
    a = answer_from_probs(q, [0, 0, 0, 0.5, 0.5])
    assert a["choice"] in ("4", "5") and abs(a["score"] - 4.5) < 1e-9
    assert target_distribution(QS["urgent"], True) == [1.0, 0.0]
    assert target_distribution(QS["urgent"], 0.3) == pytest.approx([0.3, 0.7])
    assert target_distribution(QS["queue"], None, {"billing": 3, "general": 1}) == [0.75, 0.0, 0.25]


def test_converters_and_metrics(tmp_path):
    from data.common import is_di_family, question_from_options
    from data.convert_legacy import convert
    from jev.metrics import summarize
    assert question_from_options("?", ["yes", "no"])["type"] == "noul"
    assert question_from_options("?", ["1", "2", "3", "4", "5"])["type"] == "score"
    assert question_from_options("?", ["cat", "dog"])["type"] == "choice"
    rows = convert([{"image": "a.png", "context": "", "question": "Q1", "options": ["x", "y"], "answer": "y"},
                    {"image": "a.png", "context": "", "question": "Q2", "options": ["yes", "no"], "answer": "no"}])
    assert len(rows) == 1 and len(rows[0]["questions"]) == 2 and rows[0]["expected"] == {"q0": "y", "q1": False}
    assert is_di_family("BANKING77") and is_di_family("mmlu_pro") and not is_di_family("ag_news")
    m = summarize([([0.9, 0.1], 0), ([0.2, 0.8], 1), (None, 0)])
    assert m["accuracy"] == pytest.approx(2 / 3) and m["unanswered"] == 1


def letters_decider(base, **kw):
    from jev.encode import letter_token_ids
    torch.manual_seed(0)
    tok = load_tokenizer(base)
    m = MiniJev.create(base, lora_r=4, grad_ckpt=False, dtype=torch.float32, head_type="letters",
                       letter_ids=letter_token_ids(tok)).eval()
    return Decider(m, tok, load_vision(base, 64 * 32 * 32), torch.device("cpu"), torch.float32, **kw)


def test_letter_head_cached_equals_full(tiny):
    base, _ = tiny
    cached = letters_decider(base, cache_prefix=True, token_budget=10**6)
    full = letters_decider(base, cache_prefix=False)
    assert cached.style == "letters"
    ref = flat(full.decide("My package arrived damaged.", QS))
    assert np.abs(flat(cached.decide("My package arrived damaged.", QS)) - ref).max() < 1e-5
    many = {"q": {"type": "choice", "instructions": "?", "criteria": {str(i): "" for i in range(27)}}}
    with pytest.raises(Unsupported, match="at most 26 options"):
        cached.decide("x", many)


def test_letter_head_trains_and_loads(tiny, tmp_path):
    base, data = tiny
    out = str(tmp_path / "letters")
    subprocess.run([sys.executable, "train.py", "--model", base, "--train", data, "--val", data, "--out", out,
                    "--max-steps", "6", "--bs", "4", "--grad-accum", "1", "--no-grad-ckpt", "--eval-every", "3",
                    "--lora-r", "4", "--head-type", "letters"], cwd=ROOT, check=True, capture_output=True)
    d = Decider.from_pretrained(out, device="cpu")
    assert d.style == "letters"
    out = d.decide("My package arrived damaged.", QS)
    assert abs(sum(out["answers"]["queue"]["probabilities"].values()) - 1) < 1e-4


def test_permutation_averaging(tiny):
    base, _ = tiny
    d = decider(base, cache_prefix=True)
    plain = d.decide("My package arrived damaged.", QS)
    perm = d.decide("My package arrived damaged.", QS, n_perm=2)
    # noul and score questions are never permuted
    assert perm["answers"]["urgent"]["noul"] == pytest.approx(plain["answers"]["urgent"]["noul"], abs=1e-5)
    assert perm["answers"]["stars"]["probabilities"] == pytest.approx(plain["answers"]["stars"]["probabilities"], abs=1e-5)
    # a choice question becomes the mean of the identity and the reversed order
    rev_q = {"queue": {**QS["queue"], "criteria": dict(reversed(list(QS["queue"]["criteria"].items())))}}
    rev = d.decide("My package arrived damaged.", rev_q)["answers"]["queue"]["probabilities"]
    exp = {k: (plain["answers"]["queue"]["probabilities"][k] + rev[k]) / 2 for k in rev}
    assert perm["answers"]["queue"]["probabilities"] == pytest.approx(exp, abs=1e-5)
    assert abs(sum(perm["answers"]["queue"]["probabilities"].values()) - 1) < 1e-5
