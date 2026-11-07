"""Async multi-source search.

Sources (all free):
  * Tavily            — web search built for agents (optional key, 1k credits/month free)
  * DuckDuckGo (ddgs) — keyless web fallback
  * OpenAlex          — 250M+ scholarly works, keyless
  * arXiv             — preprints, keyless
  * Semantic Scholar  — keyless (key optional for higher limits)
  * Wikipedia         — keyless background knowledge
  * Jina Reader       — turns a web page into clean text (keyless, key optional)

Every backend is isolated: one failing never breaks a search.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Literal
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from .config import get_settings
from .models import Source

log = logging.getLogger(__name__)

Kind = Literal["web", "academic", "reference"]
USER_AGENT = "MultiAgentResearchAssistant/2.0 (+https://github.com/kkm121/Multi-Agent-Research-Assistant)"
_s2_lock = asyncio.Lock()
_s2_last = 0.0
_cache: dict[tuple[str, str], list[Source]] = {}


def normalize_url(url: str) -> str:
    try:
        parts = urlsplit(url.strip())
        host = parts.netloc.lower().removeprefix("www.")
        path = parts.path.rstrip("/")
        return urlunsplit(("https", host, path, parts.query, ""))
    except Exception:
        return url.strip()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=8.0),
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )


# --------------------------------------------------------------------------- web
async def tavily(client: httpx.AsyncClient, query: str, n: int = 5) -> list[Source]:
    key = get_settings().tavily_api_key
    if not key:
        return []
    r = await client.post(
        "https://api.tavily.com/search",
        json={"query": query, "max_results": n, "search_depth": "basic", "include_raw_content": "text"},
        headers={"Authorization": f"Bearer {key}"},
    )
    r.raise_for_status()
    out = []
    for item in r.json().get("results", []):
        content = (item.get("raw_content") or item.get("content") or "").strip()
        out.append(
            Source(
                title=item.get("title") or item.get("url", "Untitled"),
                url=item.get("url", ""),
                snippet=(item.get("content") or "")[:500],
                content=content[:6000],
                kind="web",
                origin="Tavily",
            )
        )
    return out


async def duckduckgo(query: str, n: int = 5) -> list[Source]:
    def _run():
        from ddgs import DDGS

        return list(DDGS().text(query, max_results=n))

    results = await asyncio.wait_for(asyncio.to_thread(_run), timeout=20)
    return [
        Source(
            title=r.get("title", "Untitled"),
            url=r.get("href", ""),
            snippet=r.get("body", ""),
            content=r.get("body", ""),
            kind="web",
            origin="DuckDuckGo",
        )
        for r in results
        if r.get("href")
    ]


async def jina_read(client: httpx.AsyncClient, url: str, max_chars: int = 6000) -> str:
    headers = {"X-Return-Format": "text"}
    key = get_settings().jina_api_key
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        r = await client.get(f"https://r.jina.ai/{url}", headers=headers, timeout=20)
        r.raise_for_status()
        return re.sub(r"\n{3,}", "\n\n", r.text)[:max_chars]
    except Exception as e:
        log.debug("Jina read failed for %s: %s", url, e)
        return ""


# --------------------------------------------------------------------------- academic
def _openalex_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    positions = [(pos, word) for word, idxs in inv.items() for pos in idxs]
    return " ".join(w for _, w in sorted(positions))


async def openalex(client: httpx.AsyncClient, query: str, n: int = 4) -> list[Source]:
    params = {
        "search": query,
        "per-page": n,
        "select": "id,doi,title,publication_year,abstract_inverted_index,authorships,cited_by_count",
    }
    email = get_settings().contact_email
    if email:
        params["mailto"] = email
    r = await client.get("https://api.openalex.org/works", params=params)
    r.raise_for_status()
    out = []
    for w in r.json().get("results", []):
        abstract = _openalex_abstract(w.get("abstract_inverted_index"))
        if not abstract:
            continue
        authors = ", ".join(
            a.get("author", {}).get("display_name", "") for a in (w.get("authorships") or [])[:3]
        )
        out.append(
            Source(
                title=f"{w.get('title')} ({w.get('publication_year')})",
                url=w.get("doi") or w.get("id", ""),
                snippet=abstract[:500],
                content=f"Authors: {authors}\nCitations: {w.get('cited_by_count', 0)}\nAbstract: {abstract}",
                kind="academic",
                origin="OpenAlex",
            )
        )
    return out


async def arxiv(client: httpx.AsyncClient, query: str, n: int = 3) -> list[Source]:
    r = await client.get(
        "https://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "max_results": n, "sortBy": "relevance"},
    )
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    out = []
    for e in root.findall("a:entry", ns):
        title = " ".join((e.findtext("a:title", "", ns) or "").split())
        summary = " ".join((e.findtext("a:summary", "", ns) or "").split())
        year = (e.findtext("a:published", "", ns) or "")[:4]
        url = e.findtext("a:id", "", ns) or ""
        if title and summary:
            out.append(
                Source(
                    title=f"{title} ({year})",
                    url=url,
                    snippet=summary[:500],
                    content=f"Abstract: {summary}",
                    kind="academic",
                    origin="arXiv",
                )
            )
    return out


async def semantic_scholar(client: httpx.AsyncClient, query: str, n: int = 3) -> list[Source]:
    global _s2_last
    headers = {}
    key = get_settings().semantic_scholar_api_key
    if key:
        headers["x-api-key"] = key
    async with _s2_lock:  # keyless pool allows ~1 req/s
        wait = 1.1 - (time.monotonic() - _s2_last)
        if wait > 0:
            await asyncio.sleep(wait)
        _s2_last = time.monotonic()
        r = await client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": n, "fields": "title,url,abstract,year,authors,citationCount"},
            headers=headers,
        )
    r.raise_for_status()
    out = []
    for p in r.json().get("data", []) or []:
        if not p.get("abstract"):
            continue
        authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:3])
        out.append(
            Source(
                title=f"{p.get('title')} ({p.get('year')})",
                url=p.get("url", ""),
                snippet=p["abstract"][:500],
                content=f"Authors: {authors}\nCitations: {p.get('citationCount', 0)}\nAbstract: {p['abstract']}",
                kind="academic",
                origin="Semantic Scholar",
            )
        )
    return out


async def wikipedia(client: httpx.AsyncClient, query: str, n: int = 2) -> list[Source]:
    r = await client.get(
        "https://en.wikipedia.org/w/rest.php/v1/search/page", params={"q": query, "limit": n}
    )
    r.raise_for_status()
    out = []
    for page in r.json().get("pages", []):
        key = page.get("key")
        if not key:
            continue
        s = await client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(key, safe='')}")
        if s.status_code != 200:
            continue
        data = s.json()
        extract = data.get("extract", "")
        if extract:
            out.append(
                Source(
                    title=f"{data.get('title', key)} — Wikipedia",
                    url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    snippet=extract[:500],
                    content=extract,
                    kind="reference",
                    origin="Wikipedia",
                )
            )
    return out


# --------------------------------------------------------------------------- orchestration
async def _safe(name: str, coro) -> list[Source]:
    try:
        return await coro
    except Exception as e:
        log.info("search backend %s failed: %s", name, str(e)[:150])
        return []


async def search(query: str, mode: Literal["web", "academic", "both"] = "both", read_pages: int = 2) -> list[Source]:
    """Run all relevant backends in parallel, dedupe by URL, and enrich top web hits with full text."""
    cache_key = (query.lower().strip(), mode)
    if cache_key in _cache:
        return _cache[cache_key]

    async with _client() as client:
        tasks = []
        if mode in ("web", "both"):
            tasks.append(_safe("tavily", tavily(client, query)))
            tasks.append(_safe("wikipedia", wikipedia(client, query)))
        if mode in ("academic", "both"):
            tasks.append(_safe("openalex", openalex(client, query)))
            tasks.append(_safe("arxiv", arxiv(client, query)))
            tasks.append(_safe("semantic_scholar", semantic_scholar(client, query)))
        groups = await asyncio.gather(*tasks)

        results: list[Source] = [s for g in groups for s in g]
        has_web = any(s.kind == "web" for s in results)
        if mode in ("web", "both") and not has_web:
            results.extend(await _safe("duckduckgo", duckduckgo(query)))

        # Dedupe
        seen: set[str] = set()
        unique: list[Source] = []
        for s in results:
            if not s.url:
                continue
            key = normalize_url(s.url)
            if key in seen:
                continue
            seen.add(key)
            unique.append(s)

        # Enrich thin web results with full page text (Tavily already returns raw content).
        thin = [s for s in unique if s.kind == "web" and len(s.content) < 800][:read_pages]
        texts = await asyncio.gather(*(jina_read(client, s.url) for s in thin))
        for s, text in zip(thin, texts):
            if len(text) > len(s.content):
                s.content = text

    _cache[cache_key] = unique
    if len(_cache) > 256:
        _cache.pop(next(iter(_cache)))
    return unique
