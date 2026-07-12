import os
import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()
from ai_core.graph import research_graph

app = FastAPI(title="Multi-Agent Research API")
frontend_origins = os.environ.get(
    "FRONTEND_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    topic: str
    document_text: str = ""


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "multi-agent-research-api"}


@app.post("/api/research")
async def run_research_sync(request: ResearchRequest):
    topic_to_research = request.topic
    initial_state = {
        "topic": topic_to_research,
        "iterations": 0,
        "document_text": request.document_text,
    }
    final_report_md = "Report generation failed."
    async for output in research_graph.astream(initial_state):
        if "synthesis" in output:
            report = output["synthesis"].get("report")
            if report:
                final_report_md = f"# {report.title}\n\n"
                for s in report.sections:
                    final_report_md += f"## {s.title}\n{s.content}\n\n"
    return {"topic": request.topic, "status": "success", "report": final_report_md}


@app.post("/api/research/stream")
async def stream_research(request: ResearchRequest):
    print(
        f"[API] Received streaming research request for topic: '{request.topic[:50]}...'"
    )
    topic_to_research = request.topic

    async def event_generator():
        print(f"[API] Starting LangGraph execution for '{topic_to_research[:50]}...'")
        initial_state = {
            "topic": topic_to_research,
            "iterations": 0,
            "document_text": request.document_text,
        }
        cached_report = None
        try:
            async for output in research_graph.astream(initial_state):
                for node_name, state_update in output.items():
                    log_msg = f"Completed node: {node_name}"
                    if node_name == "init":
                        personas = state_update.get("personas", [])
                        log_msg = f"Generated {len(personas)} expert personas."
                    elif node_name == "research":
                        facts = state_update.get("facts", [])
                        log_msg = f"Gathered {len(facts)} facts from the internet."
                    elif node_name == "synthesis":
                        log_msg = "Synthesized draft report."
                        if "report" in state_update:
                            cached_report = state_update["report"]
                    elif node_name == "critique":
                        issues = state_update.get("actionable_issues", False)
                        log_msg = f"Critique complete. Issues found: {issues}"
                    yield {"event": "log", "data": json.dumps({"message": log_msg})}
                    if node_name == "critique":
                        issues = state_update.get("actionable_issues", False)
                        iterations = state_update.get("iterations", 0)
                        if not issues or iterations >= 2:
                            if cached_report:
                                md = f"# {cached_report.title}\n\n"
                                for s in cached_report.sections:
                                    md += f"## {s.title}\n{s.content}\n\n"
                                yield {
                                    "event": "report",
                                    "data": json.dumps(
                                        {
                                            "markdown": md,
                                            "references": [
                                                ref.model_dump()
                                                for ref in getattr(
                                                    cached_report, "references", []
                                                )
                                            ],
                                        }
                                    ),
                                }
                await asyncio.sleep(0.1)
        except Exception as e:
            yield {
                "event": "log",
                "data": json.dumps(
                    {"message": f"Critical Error during research: {str(e)}"}
                ),
            }
            yield {
                "event": "report",
                "data": json.dumps(
                    {
                        "markdown": f"# System Error\n\nThe backend encountered a critical error: {e}",
                        "references": [],
                    }
                ),
            }
            yield {"event": "done", "data": "Finished"}
            print(f"[API] Stream finished for '{topic_to_research[:50]}...'")

    return EventSourceResponse(event_generator())


from chainlit.utils import mount_chainlit

mount_chainlit(app=app, target="frontend/app.py", path="/")
