import asyncio
import json

from ai_core import conversation as C
from ai_core import graph as G
from ai_core import llm as L
from ai_core.models import Fact, Reference, ResearchReport, Source
from ai_core.rank import hybrid_rank, rrf
from test_graph import FakeRouter, fake_search


def _memory():
    m = C.Memory()
    src = {
        "https://a.org/battery": Source(title="Battery review", url="https://a.org/battery", content="Solid-state cells reach 400 Wh/kg."),
        "https://b.org/cost": Source(title="Cost study", url="https://b.org/cost", content="Costs are $800-1500 per kWh."),
    }
    facts = [
        Fact(statement="Solid-state cells reach 400 Wh/kg", source_url="https://a.org/battery"),
        Fact(statement="Solid-state battery costs are $800-1500 per kWh", source_url="https://b.org/cost"),
    ]
    report = ResearchReport(title="Solid-state batteries", markdown="Energy density is high [1]. Costs are high [2].",
                            references=[Reference(id=1, title="Battery review", url="https://a.org/battery")])
    m.add_turn("user", "solid state batteries")
    m.add_turn("assistant", "# Solid-state batteries\n\nFull previous answer text")
    m.add_research("solid state batteries", report, facts, src)
    return m


def test_memory_roundtrips_through_thread_metadata():
    m = _memory()
    from frontend.app import _memory_from_thread

    restored = _memory_from_thread({"metadata": json.dumps({"memory": m.model_dump(mode="json")})})
    assert restored.facts == m.facts and restored.sources.keys() == m.sources.keys()
    assert restored.last_assistant() == "# Solid-state batteries\n\nFull previous answer text"


def test_memory_rebuilt_from_steps_for_old_threads():
    from frontend.app import _memory_from_thread

    thread = {"metadata": {}, "steps": [{"type": "user_message", "output": "q1"}, {"type": "assistant_message", "output": "a1"}]}
    m = _memory_from_thread(thread)
    assert [t.content for t in m.turns] == ["q1", "a1"]


def test_first_message_skips_router():
    intent = asyncio.run(C.route("What are recent advances in AI?", C.Memory()))
    assert intent.kind == "research"


def test_followup_cites_only_remembered_sources(monkeypatch):
    class R(FakeRouter):
        async def text(self, messages, tier="strong", max_tokens=None, validate=None):
            assert "400 Wh/kg" in messages[1]["content"]
            return "They reach 400 Wh/kg [1] but cost a lot [2]. Made up [7]."

    monkeypatch.setattr(L, "_router", R())
    ans = asyncio.run(C.answer_followup("how dense are solid state batteries", _memory()))
    assert "[7]" not in ans.markdown
    assert {r.url for r in ans.references} <= set(_memory().sources)


def test_comparison_seeds_previous_evidence(monkeypatch):
    monkeypatch.setattr(L, "_router", FakeRouter())
    monkeypatch.setattr(G, "search", fake_search)
    m = _memory()

    async def go():
        evidence = None
        async for kind, payload in G.run_research("compare with sodium-ion", "quick", prior_facts=m.facts, prior_sources=m.sources):
            if kind == "evidence":
                evidence = payload
        return evidence

    ev = asyncio.run(go())
    assert any(f.source_url == "https://a.org/battery" for f in ev["facts"])
    assert "https://b.org/cost" in ev["sources"]


def test_rrf_and_hybrid_rank_without_embeddings():
    assert rrf([[0, 1, 2], [2, 0, 1]])[0] == 0
    docs = ["penguins live in antarctica", "lithium battery electrolyte chemistry", "football scores"]
    assert asyncio.run(hybrid_rank("battery electrolyte", docs))[0] == 1


def test_memory_survives_chainlit_session_persistence():
    from chainlit.session import clean_metadata

    m = _memory()
    session = {"memory_obj": m, "memory": m.compact().model_dump(mode="json"), "chat_profile": "Quick research"}
    persisted = clean_metadata(session)
    assert persisted["memory"]["facts"]
    assert persisted["memory_obj"] is None


def test_memory_compacts_under_chainlit_size_limit():
    m = C.Memory()
    big = "x" * 1200
    for r in range(12):
        facts = [Fact(statement=f"fact {r}-{i} " + "y" * 200, source_url=f"https://s{r}-{i}.org") for i in range(40)]
        srcs = {f.source_url: Source(title="t", url=f.source_url, content=big) for f in facts}
        m.add_research(f"q{r}", ResearchReport(title=f"r{r}", markdown="z" * 8000), facts, srcs)
    m.compact()
    assert len(m.model_dump_json()) <= C.MAX_BYTES
    assert m.reports[-1].title == "r11"
