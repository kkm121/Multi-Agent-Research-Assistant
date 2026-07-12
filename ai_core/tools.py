import os
import time
import requests
import threading
from functools import lru_cache
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
import concurrent.futures

_scholar_lock = threading.Lock()


def jina_reader(url: str) -> str:
    try:
        jina_url = f"https://r.jina.ai/{url}"
        headers = {}
        api_key = os.environ.get("JINA_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.get(jina_url, headers=headers, timeout=5)
        response.raise_for_status()
        text = response.text
        return text[:5000] if len(text) > 5000 else text
    except Exception:
        return ""


def duckduckgo_fallback(query: str) -> str:
    try:
        from duckduckgo_search import DDGS

        results = DDGS().text(query, max_results=3)
        formatted_results = []

        def process_result(r):
            title = r.get("title", "Unknown Title")
            url = r.get("href", "No URL")
            body = r.get("body", "No Abstract available")
            full_text = jina_reader(url)
            content = full_text if full_text else body
            return f"Title: {title}\nURL: {url}\nContent: {content}\n"

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            formatted_results = list(executor.map(process_result, results))
        return "\n---\n".join(formatted_results)
    except Exception as e:
        print(f"DuckDuckGo search failed: {e}")
        return ""


@retry(
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type(Exception),
)
def fetch_semantic_scholar(query: str) -> str:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": 3, "fields": "title,url,abstract,authors,year"}
    headers = {"User-Agent": "MultiAgentResearchBot/1.0 (mailto:admin@example.com)"}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    with _scholar_lock:
        time.sleep(1.0)
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
    data = response.json()
    results = data.get("data", [])
    formatted_results = []

    def process_scholar_result(r):
        title = r.get("title", "Unknown Title")
        url = r.get("url", "No URL")
        year = r.get("year", "Unknown Year")
        abstract = r.get("abstract", "No Abstract available")
        authors_list = r.get("authors", [])
        authors = ", ".join([a.get("name", "") for a in authors_list])
        full_text = jina_reader(url)
        content = full_text if full_text else abstract
        return f"Title: {title} ({year})\nAuthors: {authors}\nURL: {url}\nContent: {content}\n"

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        formatted_results = list(executor.map(process_scholar_result, results))
    if not formatted_results:
        raise ValueError("No results found in Semantic Scholar.")
    return "\n---\n".join(formatted_results)


@lru_cache(maxsize=100)
def _cached_research_tool(query: str) -> str:
    try:
        print(f"[SEARCHING] Querying Semantic Scholar for: {query}")
        return fetch_semantic_scholar(query)
    except Exception as e:
        print(
            f"[FALLBACK] Semantic Scholar failed ({str(e)}). Using DuckDuckGo for: {query}"
        )
        return duckduckgo_fallback(query)


def research_tool(query: str) -> str:
    return _cached_research_tool(query)
