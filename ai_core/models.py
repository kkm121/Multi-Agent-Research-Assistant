from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Source(BaseModel):
    id: int = 0
    title: str

    @field_validator("title")
    @classmethod
    def _one_line(cls, v: str) -> str:
        return " ".join(v.split())[:300] or "Untitled"
    url: str
    snippet: str = ""
    content: str = ""
    kind: Literal["web", "academic", "reference", "document"] = "web"
    origin: str = ""


class Fact(BaseModel):
    statement: str
    source_url: str
    persona: str = ""


class Persona(BaseModel):
    name: str = Field(description="Short name for the expert, e.g. 'Dr. Lee'.")
    role: str = Field(description="Expertise, e.g. 'Battery chemist'.")
    focus: str = Field(description="The specific angle this expert investigates, one sentence.")


class PersonaList(BaseModel):
    personas: list[Persona] = Field(description="Distinct experts with non-overlapping angles.")


class QueryPlan(BaseModel):
    queries: list[str] = Field(description="Concise search-engine queries (3-8 words each).")
    mode: Literal["web", "academic", "both"] = Field(
        default="both",
        description=(
            "'academic' only for narrow technical questions answered by papers; 'web' for news, products or "
            "current events; 'both' for explanatory questions (how something works, overviews) and anything mixed."
        ),
    )
    recent: bool = Field(
        default=False,
        description="True if the topic asks about recent, latest, current or new developments.",
    )


class ExtractedFact(BaseModel):
    statement: str = Field(description="One self-contained factual claim, with numbers/dates when present.")
    source_id: int = Field(description="The [id] of the source that states this claim.")


class ExtractedFacts(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)


class OutlineSection(BaseModel):
    title: str
    description: str = Field(description="What this section covers and which facts it uses.")


class Outline(BaseModel):
    title: str = Field(description="Title of the report.")
    sections: list[OutlineSection]


class Critique(BaseModel):
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the draft that no listed fact supports, or that cite the wrong source.",
    )
    improvements: list[str] = Field(
        default_factory=list, description="Specific, actionable fixes to structure, depth or clarity."
    )
    needs_revision: bool = Field(description="True only if there are unsupported claims or serious gaps.")


class Reference(BaseModel):
    id: int
    title: str
    url: str
    origin: str = ""


class ResearchReport(BaseModel):
    title: str
    markdown: str
    references: list[Reference] = Field(default_factory=list)
