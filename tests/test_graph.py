import asyncio
import re

import pytest

from ai_core import graph as G
from ai_core import llm as L
from ai_core.models import (
    Critique,
    ExtractedFact,
    ExtractedFacts,
    Outline,
    OutlineSection,
    Persona,
    PersonaList,
    QueryPlan,
    Source,
)


class FakeRouter:
    def __init__(self):
        self.calls = []

    async def structured(self, model, messages, tier="fast", max_tokens=None):
        self.calls.append(model.__name__)
        if model is PersonaList:
            return PersonaList(personas=[Persona(name=f"P{i}", role="r", focus="f") for i in range(5)])
        if model is QueryPlan:
            return QueryPlan(queries=["q1", "q2"], mode="both")
        if model is ExtractedFacts:
            ids = [int(x) for x in re.findall(r"^\[(\d+)\]", messages[1]["content"], re.M)]
            return ExtractedFacts(facts=[ExtractedFact(statement=f"fact from {i}", source_id=i) for i in ids[:2]]
                                  + [ExtractedFact(statement="ghost", source_id=99)])
        if model is Outline:
            return Outline(title="Test Report", sections=[OutlineSection(title="A", description="d")])
        if model is Critique:
            first = self.calls.count("Critique") == 1
            return Critique(unsupported_claims=["x"] if first else [], needs_revision=first)
        if model.__name__ == "RelevantFacts":
            return model(ids=[])
        if model.__name__ == "UnsupportedSentences":
            return model(sentences=[])
        raise AssertionError(model)

    async def text(self, messages, tier="strong", max_tokens=None, validate=None, temperature=None):
        self.calls.append("text")
        return "Summary [1].\n\n## A\nClaim [1][2]. Invented [42]."


async def fake_search(query, mode="both", read_pages=2, recent=False):
    return [Source(title=f"{query}-{i}", url=f"https://ex.com/{query}/{i}", content="c" * 50, origin="Stub") for i in range(3)]


@pytest.fixture
def patched(monkeypatch):
    fake = FakeRouter()
    monkeypatch.setattr(L, "_router", fake)
    monkeypatch.setattr(G, "search", fake_search)
    return fake


def _run(depth, **kw):
    async def go():
        events, report = [], None
        async for kind, payload in G.run_research("test topic", depth, **kw):
            (events.append(payload) if kind == "progress" else None)
            report = payload if kind == "report" else report
        return events, report

    return asyncio.run(go())


def test_quick_pipeline_end_to_end(patched):
    events, report = _run("quick")
    assert report.title == "Test Report"
    assert "[42]" not in report.markdown
    assert len(report.references) == 2
    assert all(r.url.startswith("https://ex.com/") for r in report.references)
    assert patched.calls.count("PersonaList") == 1
    assert patched.calls.count("QueryPlan") == 3
    assert patched.calls.count("text") == 2
    agents_seen = {e["agent"] for e in events}
    assert {"Orchestrator", "Planner", "Writer", "Fact-checker", "Editor"} <= agents_seen


def test_deep_uses_more_personas_and_rounds(patched):
    _, report = _run("deep")
    assert patched.calls.count("QueryPlan") == 8
    assert report.references


def test_uploaded_document_becomes_a_source(patched):
    doc = "Graphene conducts heat extremely well. " * 50
    _, report = _run("quick", document_text=doc, document_name="paper.pdf")
    assert any(r.url.startswith("document://paper.pdf") for r in report.references)


def test_no_evidence_gives_clean_error(monkeypatch):
    fake = FakeRouter()
    monkeypatch.setattr(L, "_router", fake)

    async def empty_search(*a, **k):
        return []

    monkeypatch.setattr(G, "search", empty_search)
    _, report = _run("quick")
    assert report.title == "Research could not be completed"
    assert "text" not in fake.calls
