from __future__ import annotations

import json
import logging
import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

import hmac

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ai_core.config import get_settings
from ai_core.graph import run_research
from ai_core.llm import get_router

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpcore", "openai", "primp", "ddgs"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.getLogger("instructor").setLevel(logging.CRITICAL)

app = FastAPI(title="Multi-Agent Research API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get("FRONTEND_ORIGINS", "").split(",") if o] or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    depth: Literal["quick", "deep"] = "quick"
    document_text: str = Field(default="", max_length=2_000_000)
    document_name: str = "document"


@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version}


def require_api_key(x_api_key: str = Header(default="")):
    expected = get_settings().api_key
    if not expected:
        raise HTTPException(status_code=403, detail="API access is disabled. Set API_KEY on the server to enable it.")
    if not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


@app.get("/api/providers", dependencies=[Depends(require_api_key)])
def providers():
    s = get_settings()
    r = get_router()
    return {
        "llm_providers": r.provider_names,
        "llm_calls": dict(r.stats.calls),
        "llm_failures": dict(r.stats.failures),
        "search": {
            "tavily": bool(s.tavily_api_key),
            "duckduckgo": True,
            "openalex": True,
            "arxiv": True,
            "semantic_scholar": True,
            "wikipedia": True,
            "jina_reader": True,
        },
    }


@app.post("/api/research", dependencies=[Depends(require_api_key)])
async def research_sync(req: ResearchRequest):
    report = None
    async for kind, payload in run_research(req.topic, req.depth, req.document_text, req.document_name):
        if kind == "report":
            report = payload
    return report.model_dump()


@app.post("/api/research/stream", dependencies=[Depends(require_api_key)])
async def research_stream(req: ResearchRequest):
    async def events():
        async for kind, payload in run_research(req.topic, req.depth, req.document_text, req.document_name):
            if kind == "progress":
                yield {"event": "progress", "data": json.dumps(payload)}
            elif kind == "report":
                yield {"event": "report", "data": payload.model_dump_json()}
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(events())


from chainlit.utils import mount_chainlit  # noqa: E402

mount_chainlit(app=app, target="frontend/app.py", path="/")
