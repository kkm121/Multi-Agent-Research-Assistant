"""Lightweight retrieval over an uploaded document.

Replaces the v1 `BAAI/bge-m3` + FAISS stack (~2.5 GB RAM, would not fit on a free host) with
Okapi BM25 — pure Python, no model download, millisecond latency, and strong on keyword-heavy
research queries.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were with "
    "which what how why when who their there these those not but can will into than then also".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def chunk_text(text: str, words_per_chunk: int = 220, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, words_per_chunk - overlap)
    return [" ".join(words[i : i + words_per_chunk]) for i in range(0, len(words), step) if words[i : i + words_per_chunk]]


class BM25Index:
    def __init__(self, chunks: list[str], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        q = tokenize(query)
        scores = []
        for i, (doc, dl) in enumerate(zip(self.docs, self.lengths)):
            s = 0.0
            for t in q:
                tf = doc.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            if s > 0:
                scores.append((i, s))
        return sorted(scores, key=lambda x: -x[1])[:k]


@lru_cache(maxsize=8)
def _index_for(digest: str, text: str) -> BM25Index:
    return BM25Index(chunk_text(text))


def search_document(text: str, query: str, k: int = 3) -> list[str]:
    if not text.strip():
        return []
    digest = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
    idx = _index_for(digest, text)
    return [idx.chunks[i] for i, _ in idx.search(query, k)]
