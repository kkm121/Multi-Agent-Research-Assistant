from __future__ import annotations

import hashlib
import logging
import re

from pydantic import BaseModel, Field

from .llm import LLMUnavailable, get_router
from .models import (
    Critique,
    ExtractedFacts,
    Fact,
    Outline,
    OutlineSection,
    Persona,
    PersonaList,
    QueryPlan,
    Reference,
    Source,
)
from .retrieval import chunk_text, search_document

log = logging.getLogger(__name__)

FALLBACK_PERSONAS = [
    Persona(name="Analyst", role="Domain researcher", focus="Core concepts, definitions and current state"),
    Persona(name="Historian", role="Historian of the field", focus="Origins, key milestones and how thinking evolved"),
    Persona(name="Critic", role="Skeptical reviewer", focus="Limitations, controversies, risks and open questions"),
]


async def plan_personas(topic: str, n: int = 3) -> list[Persona]:
    try:
        result = await get_router().structured(
            PersonaList,
            [
                {
                    "role": "system",
                    "content": (
                        f"You assemble a research panel. First identify every part of the question (for example "
                        "'how does X work' and 'what are its risks' are two parts). Then return exactly "
                        f"{n} expert personas so that every part is covered, the core explanatory part first, "
                        "with clearly different angles and no overlap in focus."
                    ),
                },
                {"role": "user", "content": f"Topic: {topic}"},
            ],
            tier="fast",
        )
        personas = result.personas[:n]
        if personas:
            return personas
    except LLMUnavailable:
        raise
    except Exception as e:
        log.warning("persona generation failed, using defaults: %s", e)
    return FALLBACK_PERSONAS[:n]


async def plan_queries(topic: str, persona: Persona, known: list[str], n: int = 2) -> QueryPlan:
    known_txt = "\n".join(f"- {k}" for k in known[-15:]) or "(nothing yet)"
    plan = await get_router().structured(
        QueryPlan,
        [
            {
                "role": "system",
                "content": (
                    f"You are {persona.name}, a {persona.role}. Your focus: {persona.focus}. "
                    f"Write {n} short search queries that would find authoritative evidence for your focus. "
                    "Do not repeat what is already known; target gaps. Choose the search mode, and set recent=true "
                    "when the topic is about recent, latest or current developments."
                ),
            },
            {"role": "user", "content": f"Topic: {topic}\n\nAlready known:\n{known_txt}"},
        ],
        tier="fast",
    )
    plan.queries = [q.strip() for q in plan.queries if q.strip()][:n] or [topic]
    return plan


def document_sources(document_text: str, document_name: str, query: str, topic: str = "", k: int = 2) -> list[Source]:
    chunks = search_document(document_text, f"{query} {topic}", k=k)
    if not chunks:
        chunks = chunk_text(document_text)[:k]
    return [
        Source(
            title=f"{document_name} (excerpt)",
            url=f"document://{document_name}#{hashlib.sha1(c.encode()).hexdigest()[:10]}",
            snippet=c[:300],
            content=c,
            kind="document",
            origin="Uploaded document",
        )
        for c in chunks
    ]


async def extract_facts(topic: str, persona: Persona, sources: list[Source], max_facts: int = 6) -> list[Fact]:
    if not sources:
        return []
    numbered = {i + 1: s for i, s in enumerate(sources)}
    blocks = "\n\n".join(
        f"[{i}] {s.title} ({s.origin})\n{(s.content or s.snippet)[:2200]}" for i, s in numbered.items()
    )
    result = await get_router().structured(
        ExtractedFacts,
        [
            {
                "role": "system",
                "content": (
                    f"You are {persona.name}, a {persona.role}, focusing on: {persona.focus}.\n"
                    f"Extract up to {max_facts} distinct, specific facts relevant to the topic from the sources. "
                    "Rules: each fact must be explicitly stated in the source you cite; keep numbers, dates and "
                    "names exact; ignore navigation text, ads and off-topic content. Prefer facts that directly "
                    "answer the topic as asked (central definitions, mechanisms, key people, dates, causes and "
                    "consequences) over narrow findings from a single study, species or experiment, unless the "
                    "topic asks about those. Use the source's [id]."
                ),
            },
            {"role": "user", "content": f"Topic: {topic}\n\nSources:\n{blocks}"},
        ],
        tier="fast",
    )
    facts = []
    for f in result.facts[:max_facts]:
        src = numbered.get(f.source_id)
        if src and f.statement.strip():
            facts.append(Fact(statement=f.statement.strip(), source_url=src.url, persona=persona.name))
    return facts


def _evidence_block(facts: list[Fact], ids: dict[str, int], sources: dict[str, Source]) -> str:
    by_src: dict[int, list[str]] = {}
    for f in facts:
        if f.source_url in ids:
            by_src.setdefault(ids[f.source_url], []).append(f.statement)
    lines = []
    for sid in sorted(by_src):
        src = sources[next(u for u, i in ids.items() if i == sid)]
        lines.append(f"[{sid}] {src.title}")
        lines.extend(f"    - {s}" for s in dict.fromkeys(by_src[sid]))
    return "\n".join(lines)


async def make_outline(topic: str, evidence: str) -> Outline:
    try:
        return await get_router().structured(
            Outline,
            [
                {
                    "role": "system",
                    "content": (
                        "Plan a research report. Produce 4-6 sections that tell a coherent story "
                        "(e.g. background → mechanisms/findings → applications/impact → limitations → outlook). "
                        "Only plan sections the evidence can actually support. Do not include a references section."
                    ),
                },
                {"role": "user", "content": f"Topic: {topic}\n\nEvidence:\n{evidence[:12000]}"},
            ],
            tier="fast",
        )
    except LLMUnavailable:
        raise
    except Exception as e:
        log.warning("outline failed, using default: %s", e)
        return Outline(
            title=topic,
            sections=[
                OutlineSection(title="Overview", description="Background and key concepts"),
                OutlineSection(title="Key Findings", description="The main evidence"),
                OutlineSection(title="Limitations and Open Questions", description="Gaps and debates"),
            ],
        )


async def write_report(
    topic: str, outline: Outline, evidence: str, feedback: str = "", previous: str = "", target_words: str = "800-1100"
) -> str:
    outline_txt = "\n".join(f"## {s.title}\n   {s.description}" for s in outline.sections)
    revision = (
        f"\n\nYou are revising a previous draft. Fix every reviewer note; if a flagged claim cannot be "
        f"supported by the evidence, delete it rather than rephrase it.\nReviewer notes:\n{feedback}\n\n"
        f"Previous draft:\n{previous[:12000]}"
        if feedback
        else ""
    )
    return await get_router().text(
        [
            {
                "role": "system",
                "content": (
                    "You are a meticulous research writer. Write the report body in Markdown following the outline.\n"
                    "STRICT RULES:\n"
                    "1. Use ONLY the evidence provided. Never add facts, numbers or sources from memory. Do not "
                    "add examples, materials, equipment, mechanisms, names or causes that the cited evidence does "
                    "not state, even if they are well known; a shorter accurate sentence beats a richer one.\n"
                    "2. Every factual sentence ends with a citation like [3] or [2][5] using the evidence ids.\n"
                    "3. Use '## ' headings for sections exactly as in the outline. Do not write a title, "
                    "a references list, or a closing summary of sources.\n"
                    "4. Start with a 2-3 sentence paragraph that directly answers the question as asked, before "
                    "the first heading.\n"
                    "5. Every section must help answer the question. Leave out evidence that is off the point, "
                    "however interesting. If evidence is thin for a section, say so briefly instead of padding.\n"
                    f"6. Aim for {target_words} words.\n"
                    "7. Stay close to the wording of the evidence; paraphrase only to connect findings. Write "
                    "mainly in clear paragraphs; use a short list or table only where it genuinely helps."
                ),
            },
            {
                "role": "user",
                "content": f"Topic: {topic}\n\nOutline:\n{outline_txt}\n\nEvidence:\n{evidence[:16000]}{revision}",
            },
        ],
        tier="strong",
        max_tokens=6000,
        temperature=0.2,
        validate=lambda text: check_draft(text, previous if feedback else ""),
    )


class UnsupportedSentences(BaseModel):
    sentences: list[str] = Field(
        default_factory=list,
        description="Sentences copied exactly from the draft that the evidence does not support.",
    )


def _norm(sentence: str) -> str:
    return " ".join(_CITE.sub("", sentence).lower().split()).strip(" .")


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\[(\"'])", text) if s.strip()]


async def prune_unsupported(draft: str, evidence: str) -> tuple[str, int]:
    result = await get_router().structured(
        UnsupportedSentences,
        [
            {
                "role": "system",
                "content": (
                    "You verify a research report sentence by sentence against an evidence list. Return every "
                    "sentence (copied exactly) that states something the evidence does not support: claims, "
                    "numbers, names, dates, causes or added details that are not in the evidence. Do not return "
                    "headings, transitions or sentences that only summarise supported points."
                ),
            },
            {"role": "user", "content": f"Evidence:\n{evidence[:14000]}\n\nReport:\n{draft[:14000]}"},
        ],
        tier="strong",
    )
    targets = {_norm(s) for s in result.sentences if len(s.split()) >= 4}
    if not targets:
        return draft, 0
    removed = 0
    lines = []
    for line in draft.split("\n"):
        if line.lstrip().startswith("#"):
            lines.append(line)
            continue
        kept = []
        for sent in split_sentences(line):
            if _norm(sent) in targets:
                removed += 1
            else:
                kept.append(sent)
        lines.append(" ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)), removed


def check_draft(text: str, previous: str = "") -> None:
    words = len(text.split())
    cites = sum(len(re.findall(r"\d+", group)) for group in _CITE.findall(text))
    if words < 250:
        raise ValueError(f"draft too short ({words} words)")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first.startswith("#"):
        raise ValueError("draft does not open with a direct answer")
    if cites < 3:
        raise ValueError(f"draft has only {cites} citations")
    body = re.sub(r"^#.*$", "", text, flags=re.M)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", body) if len(s.split()) >= 8]
    cited = sum(1 for s in sentences if _CITE.search(s))
    if sentences and cited / len(sentences) < 0.5:
        raise ValueError(f"only {cited}/{len(sentences)} sentences cite evidence")
    if previous and words < 0.6 * len(previous.split()):
        raise ValueError(f"revision shrank the draft to {words} words")


async def critique(draft: str, evidence: str) -> Critique:
    return await get_router().structured(
        Critique,
        [
            {
                "role": "system",
                "content": (
                    "You are a strict fact-checker. Compare the draft against the evidence list sentence by "
                    "sentence. List any claim that is not supported by the evidence it cites (or cites nothing), "
                    "including specific details the evidence does not mention (added examples, materials, "
                    "equipment, mechanisms, numbers or causes), and "
                    "up to 4 concrete improvements. Set needs_revision=true only for unsupported claims, "
                    "wrong citations, or major structural gaps — not for style nitpicks."
                ),
            },
            {"role": "user", "content": f"Evidence:\n{evidence[:12000]}\n\nDraft:\n{draft[:12000]}"},
        ],
        tier="fast",
    )


_CITE = re.compile(r"\[(\d+(?:\s*(?:,|\]\s*\[)\s*\d+)*)\]")


def assign_ids(facts: list[Fact], sources: dict[str, Source]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for f in facts:
        if f.source_url in sources and f.source_url not in ids:
            ids[f.source_url] = len(ids) + 1
    return ids


def finalize_citations(
    markdown: str, ids: dict[str, int], sources: dict[str, Source]
) -> tuple[str, list[Reference], int]:
    by_id = {i: sources[u] for u, i in ids.items()}
    remap: dict[int, int] = {}
    invalid = 0

    def repl(m: re.Match) -> str:
        nonlocal invalid
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        out = []
        for n in nums:
            if n not in by_id:
                invalid += 1
                continue
            if n not in remap:
                remap[n] = len(remap) + 1
            out.append(f"[{remap[n]}]")
        return "".join(dict.fromkeys(out))

    body = _CITE.sub(repl, markdown)
    body = re.sub(r"[ \t]+([.,;:])", r"\1", body)
    refs = [
        Reference(id=new, title=by_id[old].title, url=by_id[old].url, origin=by_id[old].origin)
        for old, new in sorted(remap.items(), key=lambda kv: kv[1])
    ]
    return body, refs, invalid


class RelevantFacts(BaseModel):
    ids: list[int] = Field(default_factory=list, description="Numbers of the facts that help answer the question.")


async def filter_relevant(topic: str, statements: list[str]) -> list[int] | None:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
    try:
        result = await get_router().structured(
            RelevantFacts,
            [
                {
                    "role": "system",
                    "content": (
                        "Select the facts that help answer the question directly. Drop facts about narrow side "
                        "topics, single experiments or tangents the question does not ask about. Keep all facts "
                        "needed for a complete answer. Return their numbers."
                    ),
                },
                {"role": "user", "content": f"Question: {topic}\n\nFacts:\n{numbered}"},
            ],
            tier="fast",
        )
    except Exception as e:
        log.warning("relevance filter failed, keeping ranked facts: %s", e)
        return None
    keep = [i - 1 for i in dict.fromkeys(result.ids) if 1 <= i <= len(statements)]
    return keep if len(keep) >= 4 else None
