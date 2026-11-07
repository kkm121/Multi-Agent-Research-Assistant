"""LangGraph orchestration (STORM-style multi-perspective research + grounded reflexion).

    plan ──Send()──▶ research(persona 1) ─┐
         ├────────▶ research(persona 2) ─┼─▶ outline ─▶ write ─▶ critique ─┬─▶ finalize
         └────────▶ research(persona N) ─┘                ▲               │
                                                          └── revise ◀────┘

Progress is streamed with `stream_mode="custom"` events: {"agent", "message", "kind"}.
"""

from __future__ import annotations

import asyncio
import logging
import operator
from typing import Annotated, Literal, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import agents
from .config import get_settings
from .llm import LLMNotConfigured, LLMUnavailable
from .models import Critique, Fact, Outline, Persona, ResearchReport, Source
from .search import search

log = logging.getLogger(__name__)

DEPTHS = {
    "quick": {"personas": 3, "rounds": 1, "queries": 2, "facts_per_round": 6, "words": "800-1100"},
    "deep": {"personas": 4, "rounds": 2, "queries": 2, "facts_per_round": 7, "words": "1300-1800"},
}


def merge_sources(a: dict[str, Source], b: dict[str, Source]) -> dict[str, Source]:
    return {**a, **b}


class ResearchState(TypedDict, total=False):
    topic: str
    depth: Literal["quick", "deep"]
    document_text: str
    document_name: str
    personas: list[Persona]
    facts: Annotated[list[Fact], operator.add]
    sources: Annotated[dict[str, Source], merge_sources]
    outline: Outline
    draft: str
    critique: Critique
    revisions: int
    report: ResearchReport
    error: str


class PersonaTask(TypedDict):
    topic: str
    depth: str
    persona: Persona
    document_text: str
    document_name: str


def emit(agent: str, message: str, kind: str = "info") -> None:
    try:
        get_stream_writer()({"agent": agent, "message": message, "kind": kind})
    except Exception:
        pass  # not running under a streaming invocation


# ------------------------------------------------------------------ nodes
async def plan(state: ResearchState) -> dict:
    cfg = DEPTHS[state.get("depth", "quick")]
    emit("Orchestrator", "Assembling an expert panel…")
    personas = await agents.plan_personas(state["topic"], cfg["personas"])
    emit("Orchestrator", "Panel: " + ", ".join(f"{p.name} ({p.role})" for p in personas), "result")
    return {"personas": personas, "revisions": 0}


def fan_out(state: ResearchState) -> list[Send]:
    return [
        Send(
            "research",
            PersonaTask(
                topic=state["topic"],
                depth=state.get("depth", "quick"),
                persona=p,
                document_text=state.get("document_text", ""),
                document_name=state.get("document_name", "document"),
            ),
        )
        for p in state["personas"]
    ]


async def research(task: PersonaTask) -> dict:
    cfg = DEPTHS[task["depth"]]
    p = task["persona"]
    facts: list[Fact] = []
    sources: dict[str, Source] = {}
    for rnd in range(cfg["rounds"]):
        try:
            plan_ = await agents.plan_queries(task["topic"], p, [f.statement for f in facts], cfg["queries"])
        except LLMUnavailable:
            raise
        except Exception as e:
            emit(p.name, f"Query planning failed: {e}", "warn")
            break
        emit(p.name, f"Round {rnd + 1}: searching {', '.join(repr(q) for q in plan_.queries)} ({plan_.mode})")
        groups = await asyncio.gather(*(search(q, plan_.mode) for q in plan_.queries))
        found: list[Source] = []
        seen = set(sources)
        for g in groups:
            for s in g[:5]:
                if s.url not in seen:
                    seen.add(s.url)
                    found.append(s)
        if task["document_text"]:
            for q in plan_.queries:
                for s in agents.document_sources(task["document_text"], task["document_name"], q, task["topic"]):
                    if s.url not in seen:
                        seen.add(s.url)
                        found.insert(0, s)  # uploaded doc gets priority
        if not found:
            emit(p.name, "No sources found this round.", "warn")
            continue
        found = found[:10]
        try:
            new_facts = await agents.extract_facts(task["topic"], p, found, cfg["facts_per_round"])
        except LLMUnavailable:
            raise
        except Exception as e:
            emit(p.name, f"Fact extraction failed: {e}", "warn")
            continue
        for s in found:
            sources[s.url] = s
        facts.extend(new_facts)
        origins = sorted({s.origin for s in found})
        emit(p.name, f"Read {len(found)} sources ({', '.join(origins)}) → {len(new_facts)} facts", "result")
    used = {f.source_url for f in facts}
    return {"facts": facts, "sources": {u: s for u, s in sources.items() if u in used}}


async def outline(state: ResearchState) -> dict:
    facts = state.get("facts", [])
    if not facts:
        return {"error": "No evidence could be gathered. Search backends or LLM providers may be unavailable."}
    emit("Planner", f"Organising {len(facts)} facts from {len(state.get('sources', {}))} sources into an outline…")
    ids = agents.assign_ids(facts, state["sources"])
    evidence = agents._evidence_block(facts, ids, state["sources"])
    o = await agents.make_outline(state["topic"], evidence)
    emit("Planner", "Outline: " + " → ".join(s.title for s in o.sections), "result")
    return {"outline": o}


def _evidence(state: ResearchState) -> tuple[dict[str, int], str]:
    ids = agents.assign_ids(state["facts"], state["sources"])
    return ids, agents._evidence_block(state["facts"], ids, state["sources"])


async def write(state: ResearchState) -> dict:
    _, evidence = _evidence(state)
    crit = state.get("critique")
    feedback = ""
    if crit and crit.needs_revision:
        feedback = "\n".join(
            [f"- Unsupported: {c}" for c in crit.unsupported_claims] + [f"- Improve: {i}" for i in crit.improvements]
        )
        emit("Writer", "Revising the draft based on fact-check feedback…")
    else:
        emit("Writer", "Drafting the report from the evidence…")
    target = DEPTHS[state.get("depth", "quick")]["words"]
    try:
        draft = await agents.write_report(state["topic"], state["outline"], evidence, feedback, state.get("draft", ""), target)
    except LLMUnavailable:
        if not feedback:
            raise
        # A failed revision must never lose a good first draft.
        emit("Writer", "Revision unavailable right now — keeping the fact-checked first draft", "warn")
        return {"draft": state["draft"], "revisions": state.get("revisions", 0) + 1, "critique": Critique(needs_revision=False)}
    emit("Writer", f"Draft ready ({len(draft.split())} words)", "result")
    return {"draft": draft, "revisions": state.get("revisions", 0) + (1 if feedback else 0)}


async def review(state: ResearchState) -> dict:
    _, evidence = _evidence(state)
    emit("Fact-checker", "Checking every claim against the evidence…")
    try:
        crit = await agents.critique(state["draft"], evidence)
    except LLMUnavailable:
        raise
    except Exception as e:
        emit("Fact-checker", f"Review skipped: {e}", "warn")
        crit = Critique(needs_revision=False)
    if crit.needs_revision:
        emit(
            "Fact-checker",
            f"Found {len(crit.unsupported_claims)} unsupported claim(s), {len(crit.improvements)} improvement(s)",
            "warn",
        )
    else:
        emit("Fact-checker", "Draft is grounded in the evidence ✓", "result")
    return {"critique": crit}


def should_revise(state: ResearchState) -> str:
    crit = state.get("critique")
    if crit and crit.needs_revision and state.get("revisions", 0) < get_settings().max_revisions:
        return "write"
    return "finalize"


async def finalize(state: ResearchState) -> dict:
    if state.get("error"):
        return {"report": ResearchReport(title="Research could not be completed", markdown=state["error"])}
    ids, _ = _evidence(state)
    body, refs, invalid = agents.finalize_citations(state["draft"], ids, state["sources"])
    if invalid:
        emit("Editor", f"Removed {invalid} citation(s) that pointed to no real source", "warn")
    emit("Editor", f"Final report with {len(refs)} verified references", "result")
    return {"report": ResearchReport(title=state["outline"].title, markdown=body, references=refs)}


def after_outline(state: ResearchState) -> str:
    return "finalize" if state.get("error") else "write"


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("plan", plan)
    g.add_node("research", research)
    g.add_node("outline", outline)
    g.add_node("write", write)
    g.add_node("review", review)
    g.add_node("finalize", finalize)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["research"])
    g.add_edge("research", "outline")
    g.add_conditional_edges("outline", after_outline, ["write", "finalize"])
    g.add_edge("write", "review")
    g.add_conditional_edges("review", should_revise, ["write", "finalize"])
    g.add_edge("finalize", END)
    return g.compile()


research_graph = build_graph()


async def run_research(topic: str, depth: str = "quick", document_text: str = "", document_name: str = "document"):
    """Async generator yielding ("progress", dict) events and finally ("report", ResearchReport)."""
    state = {"topic": topic, "depth": depth, "document_text": document_text, "document_name": document_name}
    report = None
    try:
        async for mode, chunk in research_graph.astream(state, stream_mode=["custom", "updates"]):
            if mode == "custom":
                yield "progress", chunk
            elif mode == "updates":
                for node, update in chunk.items():
                    if node == "finalize" and update and "report" in update:
                        report = update["report"]
    except LLMUnavailable as e:
        log.error("research aborted: %s", e)
        yield "progress", {"agent": "System", "message": str(e), "kind": "error"}
        if isinstance(e, LLMNotConfigured):
            report = ResearchReport(
                title="No AI provider configured",
                markdown=(
                    "The server has no LLM API key set. Add at least one free key (`GEMINI_API_KEY`, "
                    "`GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GITHUB_TOKEN` or `MISTRAL_API_KEY`) to the "
                    "environment and restart. See the README for links."
                ),
            )
        else:
            report = ResearchReport(
                title="AI providers unavailable",
                markdown=(
                    "Every configured AI provider is currently rate-limited or unreachable, so the report "
                    f"could not be written.\n\n```\n{e}\n```\n\nWait a minute and try again, or add another "
                    "provider key (see the README)."
                ),
            )
    if report is None:
        report = ResearchReport(title="Research failed", markdown="The pipeline ended without a report.")
    yield "report", report
