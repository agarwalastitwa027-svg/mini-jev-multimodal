"""Generate synthetic decision data with a local Ollama model (or any OpenAI-compatible API).

Three stages (the same idea Jev reports using: AI-generated training data):
  1. TASKS     teacher invents decision tasks: {domain, question, options}
  2. CONTEXTS  for each task and each target option, teacher writes a realistic input
               (email, chat, ticket, log line, review, transcript...) whose answer is that option
  3. VERIFY    teacher answers the question again, blind; keep only examples where it agrees

Ollama (local, free):
    ollama pull qwen3:8b
    python data/make_synthetic.py --tasks 200 --per-option 3 --out data/synthetic.jsonl

OpenAI-compatible API (e.g. a free-tier key; check the provider's terms on using outputs for training):
    python data/make_synthetic.py --backend openai --base-url https://.../v1 \
        --model <model-name> --api-key-env MY_API_KEY --workers 4
"""
import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

R = random.Random(0)

DOMAINS = [
    "customer support", "e-commerce orders", "banking and payments", "insurance claims",
    "HR and recruiting", "IT helpdesk", "software logs and incidents", "content moderation",
    "spam and phishing detection", "healthcare appointment triage (non-diagnostic)",
    "travel bookings", "food delivery", "real estate inquiries", "education and student questions",
    "legal intake (routing only)", "sales leads", "product reviews", "social media comments",
    "telecom complaints", "logistics and shipping", "smart home commands", "game NPC decisions",
    "AI agent tool routing", "prompt injection detection", "code review comments",
    "meeting transcripts", "news and articles", "finance news sentiment", "manufacturing QA reports",
    "Indian languages mixed with English (Hinglish) messages",
]
FORMATS = ["email", "chat message", "support ticket", "app review", "log excerpt",
           "call transcript snippet", "tweet/social post", "form submission", "SMS", "internal note"]


class Teacher:
    def __init__(self, a):
        self.a = a
        self.lock = threading.Lock()

    def ask_json(self, prompt, schema, temperature=0.9, images=None):
        """images: optional list of base64 strings (Ollama vision models only)."""
        a = self.a
        for attempt in range(3):
            try:
                if a.backend == "ollama":
                    msg = {"role": "user", "content": prompt}
                    if images:
                        msg["images"] = images
                    r = requests.post(f"{a.base_url.rstrip('/')}/api/chat", timeout=300, json={
                        "model": a.model, "stream": False, "think": False, "format": schema,
                        "options": {"temperature": temperature},
                        "messages": [msg]})
                    r.raise_for_status()
                    text = r.json()["message"]["content"]
                else:
                    if images:
                        raise NotImplementedError("images are only supported with --backend ollama")
                    key = os.environ.get(a.api_key_env, "")
                    r = requests.post(f"{a.base_url.rstrip('/')}/chat/completions", timeout=300,
                                      headers={"Authorization": f"Bearer {key}"}, json={
                        "model": a.model, "temperature": temperature,
                        "response_format": {"type": "json_object"},
                        "messages": [{"role": "user", "content":
                                      prompt + "\n\nReply with JSON only, matching this schema:\n"
                                      + json.dumps(schema)}]})
                    r.raise_for_status()
                    text = r.json()["choices"][0]["message"]["content"]
                text = re.sub(r"^```(json)?|```$", "", text.strip()).strip()
                return json.loads(text)
            except Exception as e:
                err = e
        print(f"  [teacher error] {err}", file=sys.stderr)
        return None


TASK_SCHEMA = {"type": "object", "properties": {"tasks": {"type": "array", "items": {
    "type": "object", "properties": {
        "question": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}}},
    "required": ["question", "options"]}}}, "required": ["tasks"]}

CTX_SCHEMA = {"type": "object", "properties": {"context": {"type": "string"}},
              "required": ["context"]}


def gen_tasks(t, domain, n):
    p = (f"You design decisions that software must make automatically in the domain: {domain}.\n"
         f"Invent {n} different decision questions about a single incoming text (message, document, log...).\n"
         "Mix three kinds:\n"
         " - choice: 3 to 8 short, mutually exclusive options (labels of 1-4 words)\n"
         " - yes/no: options exactly [\"yes\", \"no\"]\n"
         " - score: options exactly [\"1\",\"2\",\"3\",\"4\",\"5\"] and the question explains the scale\n"
         "Questions must be answerable from the text alone. No duplicates.")
    out = t.ask_json(p, TASK_SCHEMA, temperature=1.0)
    tasks = []
    for x in (out or {}).get("tasks", []):
        opts = [str(o).strip() for o in x.get("options", []) if str(o).strip()]
        if 2 <= len(opts) <= 8 and len(set(o.lower() for o in opts)) == len(opts) and x.get("question"):
            tasks.append({"domain": domain, "question": x["question"].strip(), "options": opts})
    return tasks


def gen_context(t, task, target, hard):
    fmt = R.choice(FORMATS)
    style = ("Make it subtle and realistic: the answer should be inferable but not stated "
             "with the exact option words; you may include a distracting detail."
             if hard else "Make it realistic and natural, 1-6 sentences.")
    p = (f"Domain: {task['domain']}. Write one realistic {fmt}.\n"
         f"A system will read it and answer: \"{task['question']}\" "
         f"with one of: {json.dumps(task['options'])}.\n"
         f"The correct answer for your text must be: \"{target}\".\n{style}\n"
         "Do NOT mention the question, the options or the answer. Return only the text as `context`.")
    out = t.ask_json(p, CTX_SCHEMA, temperature=1.0)
    return (out or {}).get("context", "").strip() or None


def verify(t, context, task):
    schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": task["options"]}},
              "required": ["answer"]}
    p = (f"Text:\n{context}\n\nQuestion: {task['question']}\n"
         f"Options: {json.dumps(task['options'])}\nPick the single best option.")
    out = t.ask_json(p, schema, temperature=0.0)
    return (out or {}).get("answer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--model", default="qwen3.5:9b-q4_K_M")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--tasks", type=int, default=200, help="total number of tasks to invent")
    ap.add_argument("--per-option", type=int, default=3, help="contexts per (task, option)")
    ap.add_argument("--hard-frac", type=float, default=0.4)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help=">1 only helps with cloud APIs")
    ap.add_argument("--out", default="data/synthetic.jsonl")
    a = ap.parse_args()
    t = Teacher(a)

    # 1. tasks
    tasks, per_call = [], 8
    while len(tasks) < a.tasks:
        dom = R.choice(DOMAINS)
        new = gen_tasks(t, dom, per_call)
        tasks += new
        print(f"tasks: {len(tasks)}/{a.tasks} (+{len(new)} from {dom})")
    tasks = tasks[: a.tasks]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out.replace(".jsonl", "_tasks.json"), "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=1, ensure_ascii=False)

    # 2 + 3. contexts, verified
    jobs = [(ti, opt, R.random() < a.hard_frac)
            for ti, task in enumerate(tasks) for opt in task["options"] for _ in range(a.per_option)]
    R.shuffle(jobs)
    kept = dropped = 0

    def run(job):
        ti, target, hard = job
        task = tasks[ti]
        ctx = gen_context(t, task, target, hard)
        if not ctx:
            return None
        if not a.no_verify and verify(t, ctx, task) != target:
            return False
        return {"context": ctx, "question": task["question"], "options": task["options"],
                "answer": target, "source": "synthetic", "domain": task["domain"],
                "task_id": ti, "hard": hard}

    with open(a.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as pool:
        futs = [pool.submit(run, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            ex = fu.result()
            if ex:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
            elif ex is False:
                dropped += 1
            if i % 25 == 0 or i == len(jobs):
                print(f"contexts {i}/{len(jobs)}  kept={kept}  rejected_by_verifier={dropped}")
    print(f"done -> {a.out}")


if __name__ == "__main__":
    main()
