"""Quick demo: several typed questions about one text (and optionally one image).

    python infer.py --run AstitwaAgarwal/multimodal-jev
    python infer.py --run runs/release --image chart.png --text ""
"""
import argparse
import json

from jev.decide import Decider

DEMO_QUESTIONS = {
    "queue": {"type": "choice", "instructions": "Which support queue should handle this?",
              "criteria": {"billing": "payments and refunds", "shipping": "deliveries and parcels",
                           "technical": "product faults", "general": "anything else"}},
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "satisfaction": {"type": "score", "instructions": "How satisfied is the customer?",
                     "criteria": {"1": "very unhappy", "2": "", "3": "neutral", "4": "", "5": "very happy"}},
}
IMAGE_QUESTIONS = {
    "kind": {"type": "choice", "instructions": "What kind of image is this?",
             "criteria": {"chart": "a chart or graph", "photo": "a photograph", "document": "a document or text",
                          "other": "something else"}},
    "has_text": {"type": "noul", "instructions": "Does the image contain readable text?"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder or Hugging Face repo id")
    ap.add_argument("--text", default="My package arrived damaged and I want a refund. Second time this month!")
    ap.add_argument("--image", default=None)
    ap.add_argument("--questions", default=None, help="JSON file with your own questions")
    a = ap.parse_args()
    jev = Decider.from_pretrained(a.run)
    qs = json.load(open(a.questions)) if a.questions else (IMAGE_QUESTIONS if a.image else DEMO_QUESTIONS)
    images = [a.image] if a.image else []
    jev.decide(a.text, qs, images)  # warm-up
    out = jev.decide(a.text, qs, images)
    for k, v in out["answers"].items():
        if v["type"] == "noul":
            print(f"  {k:14} P(yes) = {v['noul']:.3f}")
        else:
            extra = f"  expected {v['score']:.2f}" if "score" in v else ""
            print(f"  {k:14} {v['choice']:<10} p = {v['probabilities'][v['choice']]:.3f}{extra}")
    s = jev.last_stats
    print(f"\n{s['questions']} decisions, {s['input_tokens']} input tokens, {s['wall_ms']:.0f} ms "
          f"(state encoded once: {s['prefix_tokens']} tokens)")


if __name__ == "__main__":
    main()
