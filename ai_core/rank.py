from __future__ import annotations

import hashlib
import logging
import math

from openai import AsyncOpenAI

from .config import get_settings
from .retrieval import BM25Index

log = logging.getLogger(__name__)

EMBED_MODELS = ("gemini-embedding-001", "gemini-embedding-2")
EMBED_DIM = 768
_cache: dict[str, list[float]] = {}
_client: AsyncOpenAI | None = None


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


async def embed(texts: list[str]) -> list[list[float]] | None:
    global _client
    key = get_settings().gemini_api_key
    if not key or not texts:
        return None
    if _client is None:
        _client = AsyncOpenAI(
            api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/", timeout=20, max_retries=0
        )
    clipped = [t[:2000] or " " for t in texts]
    missing = [t for t in dict.fromkeys(clipped) if _key(t) not in _cache]
    for model in EMBED_MODELS:
        try:
            for i in range(0, len(missing), 96):
                batch = missing[i : i + 96]
                resp = await _client.embeddings.create(model=model, input=batch, dimensions=EMBED_DIM)
                for t, d in zip(batch, resp.data):
                    _cache[_key(t)] = d.embedding
            break
        except Exception as e:
            log.info("embedding with %s failed: %s", model, str(e)[:120])
    else:
        return None
    if len(_cache) > 20000:
        _cache.clear()
    try:
        return [_cache[_key(t)] for t in clipped]
    except KeyError:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + pos + 1)
    return sorted(scores, key=lambda i: -scores[i])


async def hybrid_rank(query: str, docs: list[str], top_k: int | None = None) -> list[int]:
    if not docs:
        return []
    bm25 = BM25Index(docs)
    lexical = [i for i, _ in bm25.search(query, k=len(docs))]
    lexical += [i for i in range(len(docs)) if i not in set(lexical)]
    rankings = [lexical]
    vectors = await embed([query] + docs)
    if vectors:
        q, rest = vectors[0], vectors[1:]
        sims = [_cosine(q, v) for v in rest]
        rankings.append(sorted(range(len(docs)), key=lambda i: -sims[i]))
    order = rrf(rankings)
    return order[:top_k] if top_k else order
