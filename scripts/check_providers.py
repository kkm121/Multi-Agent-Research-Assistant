from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from pydantic import BaseModel  # noqa: E402

from ai_core import search as S  # noqa: E402
from ai_core.config import get_settings  # noqa: E402
from ai_core.llm import LLMRouter  # noqa: E402


class Ping(BaseModel):
    answer: str
    number: int


async def check_llms(list_models: bool) -> bool:
    router = LLMRouter()
    if not router._providers:
        print("LLM: no provider keys configured")
        return False
    ok_any = False
    for p in router._providers:
        if list_models:
            try:
                models = await p.client.models.list()
                ids = sorted(m.id for m in models.data)
                print(f"  {p.name}: {len(ids)} models: {', '.join(ids[:40])}{' …' if len(ids) > 40 else ''}")
            except Exception as e:
                print(f"  {p.name}: could not list models ({type(e).__name__})")
        for tier, model in [(t_, m_) for t_ in ("fast", "strong") for m_ in p.models[t_]]:
            t = time.perf_counter()
            try:
                r = await p.structured.chat.completions.create(
                    model=model,
                    response_model=Ping,
                    messages=[{"role": "user", "content": "Reply with answer='pong' and number=7."}],
                    max_retries=1,
                )
                good = r.number == 7
                ok_any |= good
                print(f"  {'OK ' if good else 'BAD'} {p.name:<10} {tier:<6} {model:<45} {time.perf_counter() - t:5.1f}s")
            except Exception as e:
                print(f"  ERR {p.name:<10} {tier:<6} {model:<45} {type(e).__name__}: {str(e)[:140]}")
    return ok_any


async def check_search() -> None:
    q = "solid state battery"
    async with S._client() as c:
        backends = {
            "tavily": S.tavily(c, q),
            "openalex": S.openalex(c, q),
            "arxiv": S.arxiv(c, q),
            "semantic_scholar": S.semantic_scholar(c, q),
            "wikipedia": S.wikipedia(c, q),
            "duckduckgo": S.duckduckgo(q),
        }
        for name, coro in backends.items():
            t = time.perf_counter()
            try:
                res = await coro
                note = "(no key — skipped)" if name == "tavily" and not get_settings().tavily_api_key else ""
                print(f"  {'OK ' if res or note else 'EMPTY'} {name:<17} {len(res)} results {time.perf_counter() - t:5.1f}s {note}")
            except Exception as e:
                print(f"  ERR {name:<17} {type(e).__name__}: {str(e)[:120]}")
        t = time.perf_counter()
        text = await S.jina_read(c, "https://en.wikipedia.org/wiki/Solid-state_battery")
        print(f"  {'OK ' if len(text) > 500 else 'ERR'} jina_reader       {len(text)} chars {time.perf_counter() - t:5.1f}s")


async def main():
    print("Configured LLM providers:", get_settings().configured_providers() or "none")
    print("\nLLM providers")
    ok = await check_llms("--models" in sys.argv)
    print("\nSearch backends")
    await check_search()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
