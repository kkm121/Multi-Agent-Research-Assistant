from __future__ import annotations

import asyncio
import logging
import operator
import re
from typing import Annotated, Literal, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import agents
from .config import get_settings
from .llm import LLMNotConfigured, LLMUnavailable
from .models import Critique, Fact, Outline, Persona, ResearchReport, Source
from .rank import hybrid_rank
from .search import search

log = logging.getLogger(__name__)

RECENT_RE = re.compile(r"\b(recent|latest|newest|current|emerging|state of the art|this year|20(2[4-9]|3\d))\b", re.I)

DEPTHS = {
    "quick": {"personas": 3, "rounds": 1, "queries": 2, "facts_per_round": 6, "words": "700-1000", "keep_facts": 20},
    "deep": {"personas": 4, "rounds": 2, "queries": 2, "facts_per_round": 7, "words": "1200-1600", "keep_facts": 34},
}

CORE_PERSONA = Persona(
    name="Core Researcher",
    role="Reference researcher",
    focus="The direct answer to the question: the central facts, who, what, when, how, and the main consequences",
)


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
    prior_facts: list[Fact]
    selected: list[Fact]
    prior_sources: dict[str, Source]


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
        pass


async def plan(state: ResearchState) -> dict:
    cfg = DEPTHS[state.get("depth", "quick")]
    emit("Orchestrator", "Assembling an expert panel…")
    personas = [CORE_PERSONA] + await agents.plan_personas(state["topic"], cfg["personas"] - 1)
    emit("Orchestrator", "Panel: " + ", ".join(f"{p.name} ({p.role})" for p in personas), "result")
    seeded = {}
    prior = state.get("prior_facts") or []
    if prior:
        emit("Orchestrator", f"Carrying {len(prior)} facts from earlier in this chat into the comparison", "result")
        seeded = {"facts": prior, "sources": state.get("prior_sources") or {}}
    return {"personas": personas, "revisions": 0, **seeded}


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
        if p.name == CORE_PERSONA.name and rnd == 0:
            plan_.queries = [task["topic"], f"{task['topic']} overview key facts"] + [
                q for q in plan_.queries if q != task["topic"]
            ][:1]
            plan_.mode = "both"
        emit(p.name, f"Round {rnd + 1}: searching {', '.join(repr(q) for q in plan_.queries)} ({plan_.mode})")
        recent = plan_.recent or bool(RECENT_RE.search(task["topic"]))
        groups = await asyncio.gather(*(search(q, plan_.mode, recent=recent) for q in plan_.queries))
        candidates: list[Source] = []
        seen = set(sources)
        for g in groups:
            for s in g:
                if s.url not in seen:
                    seen.add(s.url)
                    candidates.append(s)
        found: list[Source] = []
        if candidates:
            query = f"{task['topic']} {p.focus} {' '.join(plan_.queries)}"
            order = await hybrid_rank(query, [f"{s.title}. {s.snippet or s.content[:500]}" for s in candidates])
            per_origin: dict[str, int] = {}
            for i in order:
                o = candidates[i].origin
                if per_origin.get(o, 0) < 4:
                    per_origin[o] = per_origin.get(o, 0) + 1
                    found.append(candidates[i])
                if len(found) == 8:
                    break
        if task["document_text"]:
            for q in plan_.queries:
                for s in agents.document_sources(task["document_text"], task["document_name"], q, task["topic"]):
                    if s.url not in seen:
                        seen.add(s.url)
                        found.insert(0, s)
        if not found:
            emit(p.name, "No sources found this round.", "warn")
            continue
        found = found[:10]
        try:
            limit = cfg["facts_per_round"] + (4 if p.name == CORE_PERSONA.name else 0)
            new_facts = await agents.extract_facts(task["topic"], p, found, limit)
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
    keep = DEPTHS[state.get("depth", "quick")]["keep_facts"]
    order = await hybrid_rank(state["topic"], [f.statement for f in facts])
    core = [f for f in facts if f.persona == CORE_PERSONA.name]
    ranked = [facts[i] for i in order if facts[i].persona != CORE_PERSONA.name][: keep + 8]
    chosen = await agents.filter_relevant(state["topic"], [f.statement for f in ranked]) if ranked else []
    others = [ranked[i] for i in sorted(chosen)] if chosen else ranked
    selected, seen = [], set()
    for f in core + others:
        if f.statement not in seen:
            seen.add(f.statement)
            selected.append(f)
    selected = selected[:keep]
    emit(
        "Planner",
        f"Kept the {len(selected)} most relevant of {len(facts)} facts; organising them into an outline…",
    )
    ids = agents.assign_ids(selected, state["sources"])
    evidence = agents._evidence_block(selected, ids, state["sources"])
    o = await agents.make_outline(state["topic"], evidence)
    emit("Planner", "Outline: " + " → ".join(s.title for s in o.sections), "result")
    return {"outline": o, "selected": selected}


def _evidence(state: ResearchState) -> tuple[dict[str, int], str]:
    facts = state.get("selected") or state["facts"]
    ids = agents.assign_ids(facts, state["sources"])
    return ids, agents._evidence_block(facts, ids, state["sources"])


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
        emit("Fact-checker", "Draft is grounded in the evidence", "result")
    return {"critique": crit}


def should_revise(state: ResearchState) -> str:
    crit = state.get("critique")
    if crit and crit.needs_revision and state.get("revisions", 0) < get_settings().max_revisions:
        return "write"
    return "finalize"


async def finalize(state: ResearchState) -> dict:
    if state.get("error"):
        return {"report": ResearchReport(title="Research could not be completed", markdown=state["error"])}
    ids, evidence = _evidence(state)
    draft = state["draft"]
    try:
        draft, removed = await agents.prune_unsupported(draft, evidence)
        if removed:
            emit("Editor", f"Removed {removed} sentence(s) the evidence does not support", "warn")
    except LLMUnavailable:
        pass
    body, refs, invalid = agents.finalize_citations(draft, ids, state["sources"])
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


async def run_research(
    topic: str,
    depth: str = "quick",
    document_text: str = "",
    document_name: str = "document",
    prior_facts: list[Fact] | None = None,
    prior_sources: dict[str, Source] | None = None,
):
    state = {
        "topic": topic,
        "depth": depth,
        "document_text": document_text,
        "document_name": document_name,
        "prior_facts": prior_facts or [],
        "prior_sources": prior_sources or {},
    }
    report = None
    final: dict = {}
    try:
        async for mode, chunk in research_graph.astream(state, stream_mode=["custom", "values"]):
            if mode == "custom":
                yield "progress", chunk
            elif mode == "values":
                final = chunk
                if chunk.get("report") is not None:
                    report = chunk["report"]
    except LLMUnavailable as e:
        log.error("research aborted: %s", e)
        yield "progress", {"agent": "System", "message": str(e), "kind": "error"}
        if isinstance(e, LLMNotConfigured):
            report = ResearchReport(
                title="No AI provider configured",
                markdown=(
                    "The server has no LLM API key set. Add at least one free key (`GEMINI_API_KEY`, "
                    "`GROQ_API_KEY`, `OPENROUTER_API_KEY`, `NVIDIA_API_KEY` or `MISTRAL_API_KEY`) to the "
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
    yield "evidence", {
        "facts": final.get("facts", []),
        "selected": final.get("selected") or final.get("facts", []),
        "sources": final.get("sources", {}),
    }
    yield "report", report
