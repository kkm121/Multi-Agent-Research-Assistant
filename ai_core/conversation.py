from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

from . import agents
from .llm import get_router
from .models import Fact, Reference, ResearchReport, Source
from .rank import hybrid_rank

log = logging.getLogger(__name__)

MAX_SOURCE_CHARS = 1200
MAX_FACTS = 300
MAX_BYTES = 700_000


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ReportMemory(BaseModel):
    question: str
    title: str
    markdown: str
    references: list[Reference] = Field(default_factory=list)


class Memory(BaseModel):
    turns: list[Turn] = Field(default_factory=list)
    reports: list[ReportMemory] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    sources: dict[str, Source] = Field(default_factory=dict)

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))
        self.turns = self.turns[-40:]

    def add_research(self, question: str, report: ResearchReport, facts: list[Fact], sources: dict[str, Source]) -> None:
        self.reports.append(
            ReportMemory(question=question, title=report.title, markdown=report.markdown, references=report.references)
        )
        self.reports = self.reports[-10:]
        known = {(f.statement, f.source_url) for f in self.facts}
        self.facts.extend(f for f in facts if (f.statement, f.source_url) not in known)
        self.facts = self.facts[-MAX_FACTS:]
        for url, s in sources.items():
            self.sources[url] = s.model_copy(update={"content": (s.content or s.snippet)[:MAX_SOURCE_CHARS]})
        used = {f.source_url for f in self.facts}
        self.sources = {u: s for u, s in self.sources.items() if u in used}

    def transcript(self, limit: int = 12, chars: int = 700) -> str:
        lines = []
        for t in self.turns[-limit:]:
            text = t.content if len(t.content) <= chars else t.content[:chars] + " …"
            lines.append(f"{t.role.upper()}: {text}")
        return "\n".join(lines)

    def compact(self) -> "Memory":
        for i, t in enumerate(self.turns[:-2]):
            if len(t.content) > 4000:
                self.turns[i] = Turn(role=t.role, content=t.content[:4000] + " …")
        while len(self.model_dump_json()) > MAX_BYTES and (len(self.reports) > 1 or len(self.facts) > 50):
            if len(self.reports) > 1:
                self.reports.pop(0)
            self.facts = self.facts[len(self.facts) // 5 :]
            used = {f.source_url for f in self.facts}
            self.sources = {u: s for u, s in self.sources.items() if u in used}
        return self

    def last_assistant(self) -> str | None:
        for t in reversed(self.turns):
            if t.role == "assistant":
                return t.content
        return None


class Intent(BaseModel):
    kind: Literal["research", "followup", "repeat", "chat"] = Field(
        description=(
            "research: needs new evidence from the web/papers (a new topic, or comparing earlier findings with "
            "something not yet researched). followup: can be answered from the earlier reports in this chat "
            "(clarify, summarise, explain, list sources, what did you find about X). repeat: asks to repeat or "
            "show again the previous answer. chat: greetings, thanks, or questions about the assistant itself."
        )
    )
    standalone_query: str = Field(
        description=(
            "The user's request rewritten as a complete, self-contained question that resolves references like "
            "'that', 'the previous findings' or 'it' using the conversation."
        )
    )
    uses_previous: bool = Field(
        default=False, description="True if answering needs the earlier findings of this chat (e.g. a comparison)."
    )


_GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|cool|great)\b[\s!.]*$", re.I)


async def route(message: str, memory: Memory) -> Intent:
    if not memory.turns:
        if _GREETING.match(message):
            return Intent(kind="chat", standalone_query=message)
        return Intent(kind="research", standalone_query=message)
    reports = "\n".join(f"- {r.title} (asked: {r.question})" for r in memory.reports[-5:]) or "(none)"
    try:
        return await get_router().structured(
            Intent,
            [
                {
                    "role": "system",
                    "content": (
                        "You route messages in a research assistant chat. Decide what the latest user message needs. "
                        "Prefer followup when the earlier reports already cover the question; choose research only "
                        "when genuinely new evidence is required."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Reports so far:\n{reports}\n\nConversation:\n{memory.transcript()}\n\nLatest message: {message}",
                },
            ],
            tier="fast",
        )
    except Exception as e:
        log.warning("routing failed, defaulting to research: %s", e)
        return Intent(kind="research", standalone_query=message)


async def answer_followup(question: str, memory: Memory) -> ResearchReport:
    facts = [f for f in memory.facts if f.source_url in memory.sources]
    order = await hybrid_rank(question, [f.statement for f in facts], top_k=25) if facts else []
    picked = [facts[i] for i in order]
    ids = agents.assign_ids(picked, memory.sources)
    evidence = agents._evidence_block(picked, ids, memory.sources) or "(no stored evidence)"
    last = memory.reports[-1].markdown[:4000] if memory.reports else ""
    body = await get_router().text(
        [
            {
                "role": "system",
                "content": (
                    "You are a research assistant continuing a conversation. Answer the user's question using ONLY "
                    "the evidence list from earlier research in this chat. Cite every factual sentence with the "
                    "evidence ids like [2]. If the evidence does not cover the question, say so plainly and suggest "
                    "running new research. Be direct and well structured; use headings only for long answers."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation so far:\n{memory.transcript()}\n\nMost recent report (for context):\n{last}\n\n"
                    f"Evidence:\n{evidence}\n\nQuestion: {question}"
                ),
            },
        ],
        tier="strong",
        max_tokens=2500,
    )
    markdown, refs, _ = agents.finalize_citations(body, ids, memory.sources)
    return ResearchReport(title="", markdown=markdown, references=refs)


async def chat_reply(message: str, memory: Memory) -> str:
    return await get_router().text(
        [
            {
                "role": "system",
                "content": (
                    "You are the Research Assistant: a multi-agent system that researches questions across the web "
                    "and academic databases and writes cited reports. Reply briefly and helpfully. Do not state "
                    "facts about the world here; offer to research instead."
                ),
            },
            {"role": "user", "content": f"Conversation:\n{memory.transcript()}\n\nMessage: {message}"},
        ],
        tier="fast",
        max_tokens=400,
    )
