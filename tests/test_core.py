import asyncio

import httpx
import openai
import pytest

from ai_core import agents
from ai_core.config import Settings
from ai_core.llm import LLMRouter, LLMUnavailable
from ai_core.models import Fact, Source
from ai_core.retrieval import chunk_text, search_document
from ai_core.search import _openalex_abstract, normalize_url
from frontend.auth import hash_password, verify_password


def _sources():
    return {
        "https://a.com": Source(title="A", url="https://a.com", origin="Tavily"),
        "https://b.com": Source(title="B", url="https://b.com", origin="OpenAlex"),
        "https://c.com": Source(title="C (unused)", url="https://c.com"),
    }


def test_assign_ids_only_cited_sources_in_order():
    facts = [Fact(statement="x", source_url="https://b.com"), Fact(statement="y", source_url="https://a.com"),
             Fact(statement="z", source_url="https://b.com"), Fact(statement="q", source_url="https://ghost.com")]
    assert agents.assign_ids(facts, _sources()) == {"https://b.com": 1, "https://a.com": 2}


def test_finalize_citations_renumbers_and_drops_invented_ids():
    ids = {"https://b.com": 1, "https://a.com": 2}
    md = "Claim one [2]. Claim two [1][2]. Made up [9]. Mixed [1, 7]."
    body, refs, invalid = agents.finalize_citations(md, ids, _sources())
    assert body == "Claim one [1]. Claim two [2][1]. Made up. Mixed [2]."
    assert [(r.id, r.title) for r in refs] == [(1, "A"), (2, "B")]
    assert invalid == 2


def test_evidence_block_groups_by_source():
    facts = [Fact(statement="f1", source_url="https://a.com"), Fact(statement="f2", source_url="https://a.com")]
    ids = agents.assign_ids(facts, _sources())
    block = agents._evidence_block(facts, ids, _sources())
    assert block.splitlines() == ["[1] A", "    - f1", "    - f2"]


def test_bm25_finds_relevant_chunk():
    text = ("Penicillin was discovered by Alexander Fleming in 1928. " * 5 + "filler words " * 300
            + "Photosynthesis converts light energy into chemical energy in chloroplasts. " * 5)
    hits = search_document(text, "chloroplast photosynthesis light", k=1)
    assert hits and "Photosynthesis" in hits[0]


def test_chunking_overlaps():
    chunks = chunk_text(" ".join(str(i) for i in range(500)), words_per_chunk=200, overlap=50)
    assert len(chunks) == 4 and chunks[1].split()[0] == "150"


def test_openalex_abstract_reconstruction():
    assert _openalex_abstract({"world": [1], "hello": [0], "again": [2]}) == "hello world again"


def test_normalize_url():
    assert normalize_url("http://www.Example.com/path/") == normalize_url("https://example.com/path")


def test_password_hashing():
    h = hash_password("s3cret-pw")
    assert h.startswith("scrypt$") and "s3cret" not in h
    assert verify_password("s3cret-pw", h)
    assert not verify_password("wrong", h)
    assert hash_password("s3cret-pw") != h


def _status_error(cls, code, headers=None):
    req = httpx.Request("POST", "https://x")
    return cls("err", response=httpx.Response(code, request=req, headers=headers or {}), body=None)


def test_router_without_keys_raises():
    r = LLMRouter(Settings(_env_file=None))
    with pytest.raises(LLMUnavailable):
        asyncio.run(r.text([{"role": "user", "content": "hi"}]))


def _router(**kw):
    return LLMRouter(Settings(_env_file=None, **kw))


def test_router_fails_over_and_benches_rate_limited_model():
    r = _router(gemini_api_key="k1", groq_api_key="k2", provider_order="gemini,groq",
                gemini_fast_models="g1", groq_fast_models="q1")
    calls = []

    async def call(p, model):
        calls.append(p.name)
        if p.name == "gemini":
            raise _status_error(openai.RateLimitError, 429, {"retry-after": "120"})
        return "ok from " + p.name

    assert asyncio.run(r._run(call, "fast", "t")) == "ok from groq"
    assert asyncio.run(r._run(call, "fast", "t")) == "ok from groq"
    assert calls == ["gemini", "groq", "groq"]


def test_router_skips_bad_key_and_payment_required():
    r = _router(groq_api_key="bad", nvidia_api_key="pay", mistral_api_key="ok", provider_order="groq,nvidia,mistral")

    async def call(p, model):
        if p.name == "groq":
            raise _status_error(openai.AuthenticationError, 401)
        if p.name == "nvidia":
            raise _status_error(openai.APIStatusError, 402)
        return p.name

    assert asyncio.run(r._run(call, "fast", "t")) == "mistral"
    assert r._providers[0].benched_until > 0 and r._providers[1].benched_until > 0


def test_router_falls_back_to_next_model_when_retired():
    r = _router(gemini_api_key="k", gemini_fast_models="old-model,new-model", provider_order="gemini")
    tried = []

    async def call(p, model):
        tried.append(model)
        if model == "old-model":
            raise _status_error(openai.NotFoundError, 404)
        return model

    assert asyncio.run(r._run(call, "fast", "t")) == "new-model"
    assert asyncio.run(r._run(call, "fast", "t")) == "new-model"
    assert tried == ["old-model", "new-model", "new-model"]


def test_router_unwraps_instructor_wrapped_rate_limit():
    from instructor.core.exceptions import InstructorRetryException

    r = _router(gemini_api_key="k1", groq_api_key="k2", provider_order="gemini,groq", gemini_fast_models="g1")

    async def call(p, model):
        if p.name == "gemini":
            try:
                raise _status_error(openai.RateLimitError, 429)
            except openai.RateLimitError as inner:
                raise InstructorRetryException("wrapped", n_attempts=1, total_usage=0) from inner
        return p.name

    assert asyncio.run(r._run(call, "fast", "t")) == "groq"
    assert "g1" in r._providers[0].model_benched_until


def test_rate_limit_benches_only_that_model_and_parses_retry_delay():
    r = _router(gemini_api_key="k", gemini_fast_models="m1,m2", provider_order="gemini")
    tried = []

    async def call(p, model):
        tried.append(model)
        if model == "m1":
            raise _status_error(openai.RateLimitError, 429)
        return model

    assert asyncio.run(r._run(call, "fast", "t")) == "m2"
    p = r._providers[0]
    assert p.benched_until == 0
    assert "m1" not in p.ready_models("fast") and "m2" in p.ready_models("fast")
    err = _status_error(openai.RateLimitError, 429)
    err.args = ("Error code: 429 - ... Please retry in 51.119078467s.",)
    assert 51 < r._retry_after(err) < 53


def test_check_draft_rejects_degenerate_output():
    good = " ".join(["word"] * 300) + " [1] [2] [3]"
    agents.check_draft(good)
    with pytest.raises(ValueError):
        agents.check_draft("Too short [1][2][3].")
    with pytest.raises(ValueError):
        agents.check_draft(" ".join(["word"] * 300))
    with pytest.raises(ValueError):
        agents.check_draft(good, previous=" ".join(["w"] * 1000))
