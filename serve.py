"""Jev-compatible HTTP server: POST /v1/systemone.

    python serve.py --run AstitwaAgarwal/multimodal-jev --port 8000
    curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
      "model": "mini-jev-multimodal",
      "state": "My package arrived damaged and I want a refund.",
      "questions": {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}}'

Request   {"model", "state", "questions", "images"?: [data URL | base64 | path], "max_pixels"?: int}
Response  {"model", "answers": {key: {"type": "choice", "choice", "probabilities"} | {"type": "noul", "noul"}}, "usage"}

Requests over a declared limit get HTTP 413 with a message containing "maximum context length"
(or "options per choice"), which the Decision Index `http` engine records as unsupported.
Set MINI_JEV_API_KEY to require `Authorization: Bearer <key>`.
"""
import argparse
import os
import threading

from jev.schema import Unsupported


def build_app(decider):
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="mini-jev-multimodal", version="2")
    lock = threading.Lock()  # one GPU, one request at a time
    key = os.environ.get("MINI_JEV_API_KEY")

    @app.get("/health")
    def health():
        return {"ok": True, "model": decider.name, "vision": decider.vision is not None,
                "max_tokens": decider.max_tokens}

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: str | None = Header(default=None)):
        if key and authorization != f"Bearer {key}":
            raise HTTPException(401, "invalid API key")
        body = await request.json()
        if "questions" not in body:
            raise HTTPException(422, "missing 'questions'")
        try:
            with lock:
                out = decider.decide(body.get("state", ""), body["questions"], body.get("images") or [],
                                     body.get("max_pixels"))
        except Unsupported as e:
            return JSONResponse({"error": {"type": "unsupported", "message": str(e)}}, status_code=413)
        except (ValueError, TypeError, FileNotFoundError) as e:
            return JSONResponse({"error": {"type": "invalid_request", "message": str(e)}}, status_code=400)
        return out

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder or Hugging Face repo id")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-pixels", type=int, default=1_638_400)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    import uvicorn
    from jev.decide import Decider
    dec = Decider.from_pretrained(a.run, device=a.device, max_tokens=a.max_tokens, max_pixels=a.max_pixels)
    uvicorn.run(build_app(dec), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
