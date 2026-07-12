import os
import instructor
from openai import OpenAI
from cerebras.cloud.sdk import Cerebras
from groq import Groq
from typing import List, Any
from .models import (
    PersonaList,
    SearchQueryList,
    FactList,
    ResearchReport,
    CriticFeedback,
    Persona,
    Section,
    Outline,
)
from .tools import research_tool
import hashlib
import numpy as np
import faiss
from tenacity import retry, stop_after_attempt, wait_exponential
import threading
import time
import logging

logging.getLogger("tenacity").setLevel(logging.CRITICAL)
_cerebras_sem = threading.Semaphore(int(os.environ.get("CEREBRAS_CONCURRENCY", 2)))
_sambanova_sem = threading.Semaphore(int(os.environ.get("SAMBANOVA_CONCURRENCY", 2)))
_groq_sem = threading.Semaphore(int(os.environ.get("GROQ_CONCURRENCY", 2)))
_document_cache = {}
_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer("BAAI/bge-m3")
    return _embedding_model


def get_document_chunks(text: str):
    model = get_embedding_model()
    doc_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    if doc_hash in _document_cache:
        return (_document_cache[doc_hash], model)
    try:
        from indicnlp.tokenize import sentence_tokenize

        sentences = sentence_tokenize.sentence_split(text, lang="en")
    except Exception as e:
        sentences = text.split(". ")
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        words = sentence.split()
        if current_length + len(words) > 300:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(words)
        else:
            current_chunk.append(sentence)
            current_length += len(words)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    if not chunks:
        return (([], None), model)
    embeddings = model.encode(chunks).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    _document_cache[doc_hash] = (chunks, index)
    return (_document_cache[doc_hash], model)
    words = text.split()
    chunks = [" ".join(words[i : i + 300]) for i in range(0, len(words), 200)]
    if not chunks:
        return (([], None), model)
    embeddings = model.encode(chunks).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    _document_cache[doc_hash] = (chunks, index)
    return (_document_cache[doc_hash], model)


def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
    client = Groq(api_key=api_key)
    return instructor.from_groq(client, mode=instructor.Mode.TOOLS)


def get_sambanova_client():
    api_key = os.environ.get("SAMBANOVA_API_KEY", "your_sambanova_api_key_here")
    client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=api_key, timeout=15)
    return instructor.from_openai(client, mode=instructor.Mode.JSON)


def get_cerebras_client():
    api_key = os.environ.get("CEREBRAS_API_KEY", "your_cerebras_api_key_here")
    client = Cerebras(api_key=api_key)
    return instructor.from_cerebras(client, mode=instructor.Mode.JSON)


def generate_personas(topic: str) -> PersonaList:
    messages = [
        {
            "role": "system",
            "content": "You are an expert AI orchestrator. Given a research topic, generate 3 diverse expert personas who would ask insightful questions about this topic.",
        },
        {"role": "user", "content": f"Topic: {topic}"},
    ]
    try:
        with _groq_sem:
            client = get_groq_client()
            return client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                response_model=PersonaList,
                messages=messages,
            )
    except Exception:
        try:
            with _cerebras_sem:
                client = get_cerebras_client()
                return client.chat.completions.create(
                    model="gemma-4-31b", response_model=PersonaList, messages=messages
                )
        except Exception:
            with _sambanova_sem:
                client = get_sambanova_client()
                return client.chat.completions.create(
                    model="Meta-Llama-3.3-70B-Instruct",
                    response_model=PersonaList,
                    messages=messages,
                )


def _generate_queries(
    persona_name: str, persona_role: str, topic: str, context: str = ""
) -> SearchQueryList:
    messages = [
        {
            "role": "system",
            "content": f"You are {persona_name}, a {persona_role}. Generate 1 precise search query to research the topic. If context is provided, ask follow-up questions to fill gaps.",
        },
        {
            "role": "user",
            "content": f"Topic: {topic}\n\nContext/Previous Findings:\n{context}",
        },
    ]
    try:
        with _groq_sem:
            client = get_groq_client()
            return client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                response_model=SearchQueryList,
                messages=messages,
            )
    except Exception:
        try:
            with _cerebras_sem:
                client = get_cerebras_client()
                return client.chat.completions.create(
                    model="gemma-4-31b",
                    response_model=SearchQueryList,
                    messages=messages,
                )
        except Exception:
            with _sambanova_sem:
                client = get_sambanova_client()
                return client.chat.completions.create(
                    model="Meta-Llama-3.3-70B-Instruct",
                    response_model=SearchQueryList,
                    messages=messages,
                )


def conduct_research(
    topic: str, persona_name: str, persona_role: str, document_text: str = ""
) -> FactList:
    all_facts = []
    accumulated_context = ""
    for round_num in range(1):
        try:
            queries = _generate_queries(
                persona_name, persona_role, topic, accumulated_context
            )
            client = get_cerebras_client()
            for q in queries.queries:
                raw_text = research_tool(q.query)
                if document_text:
                    try:
                        (chunks, index), model = get_document_chunks(document_text)
                        if len(chunks) > 0 and index is not None:
                            q_emb = model.encode([q.query]).astype("float32")
                            distances, indices = index.search(q_emb, 2)
                            for idx in indices[0]:
                                if idx != -1:
                                    raw_text += (
                                        f"\n\n[From Uploaded Document]: {chunks[idx]}"
                                    )
                    except Exception as e:
                        print(f"RAG Chunking Error: {e}")
                if raw_text and len(raw_text) > 20:
                    truncated_text = raw_text[:3500]
                    try:
                        with _cerebras_sem:
                            facts = client.chat.completions.create(
                                model="gemma-4-31b",
                                response_model=FactList,
                                messages=[
                                    {
                                        "role": "system",
                                        "content": f"Extract 1 key factual statement from the text that is strictly relevant to the topic: '{topic}'. Ignore all irrelevant website sidebars, ads, and unrelated news. Include the source URL if available (use 'Uploaded Document' if from a document).",
                                    },
                                    {
                                        "role": "user",
                                        "content": f"Text: {truncated_text}",
                                    },
                                ],
                            )
                        all_facts.extend(facts.facts)
                        for f in facts.facts:
                            accumulated_context += f"- {f.statement}\n"
                    except Exception:
                        try:
                            with _groq_sem:
                                groq = get_groq_client()
                                facts = groq.chat.completions.create(
                                    model="llama-3.3-70b-versatile",
                                    response_model=FactList,
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": f"Extract 1 key factual statement from the text that is strictly relevant to the topic: '{topic}'. Ignore all irrelevant website sidebars, ads, and unrelated news. Include the source URL if available (use 'Uploaded Document' if from a document).",
                                        },
                                        {
                                            "role": "user",
                                            "content": f"Text: {truncated_text}",
                                        },
                                    ],
                                )
                            all_facts.extend(facts.facts)
                            for f in facts.facts:
                                accumulated_context += f"- {f.statement}\n"
                        except Exception:
                            try:
                                with _sambanova_sem:
                                    sambanova = get_sambanova_client()
                                    facts = sambanova.chat.completions.create(
                                        model="Meta-Llama-3.3-70B-Instruct",
                                        response_model=FactList,
                                        messages=[
                                            {
                                                "role": "system",
                                                "content": f"Extract 1 key factual statement from the text that is strictly relevant to the topic: '{topic}'. Ignore all irrelevant website sidebars, ads, and unrelated news. Include the source URL if available (use 'Uploaded Document' if from a document).",
                                            },
                                            {
                                                "role": "user",
                                                "content": f"Text: {truncated_text}",
                                            },
                                        ],
                                    )
                                all_facts.extend(facts.facts)
                                for f in facts.facts:
                                    accumulated_context += f"- {f.statement}\n"
                            except Exception as e:
                                print(
                                    f"Fact extraction failed completely for query '{q.query}': {e}"
                                )
        except Exception as e:
            print(f"Error in research round {round_num}: {e}")
    if not all_facts:
        print(
            "Warning: All fact extraction calls failed (likely due to rate limits). Injecting fallback fact."
        )
        from .models import Fact

        all_facts.append(
            Fact(
                statement=f"The research system was unable to retrieve factual information about '{topic}' due to API rate limits from the LLM providers.",
                source_url="System Alert: Rate Limit Exceeded",
            )
        )
    return FactList(facts=all_facts)


def critique_report(report: ResearchReport, facts: List[Any]) -> CriticFeedback:
    draft = "\n".join([f"## {s.title}\n{s.content}" for s in report.sections])
    messages = [
        {
            "role": "system",
            "content": "You are a harsh but fair academic critic. Review the following draft report and provide highly specific feedback to improve its accuracy, depth, and structure.",
        },
        {"role": "user", "content": f"Draft:\n{draft}"},
    ]
    try:
        with _cerebras_sem:
            client = get_cerebras_client()
            return client.chat.completions.create(
                model="gemma-4-31b", response_model=CriticFeedback, messages=messages
            )
    except Exception:
        try:
            with _groq_sem:
                client = get_groq_client()
                return client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    response_model=CriticFeedback,
                    messages=messages,
                )
        except Exception:
            with _sambanova_sem:
                client = get_sambanova_client()
                return client.chat.completions.create(
                    model="Meta-Llama-3.3-70B-Instruct",
                    response_model=CriticFeedback,
                    messages=messages,
                )


def generate_outline(topic: str, facts: List[Any]) -> Outline:
    client = get_cerebras_client()
    fact_strings = "\n".join([f"- {f.statement}" for f in facts[:50]])
    try:
        return client.chat.completions.create(
            model="gemma-4-31b",
            response_model=Outline,
            messages=[
                {
                    "role": "system",
                    "content": "You are a master planner. Create a highly structured outline for an academic report based on the provided topic and facts.",
                },
                {
                    "role": "user",
                    "content": f"Topic: {topic}\n\nFacts:\n{fact_strings}",
                },
            ],
            max_tokens=4000,
        )
    except Exception:
        try:
            sambanova = get_sambanova_client()
            return sambanova.chat.completions.create(
                model="Meta-Llama-3.3-70B-Instruct",
                response_model=Outline,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a master planner. Create a highly structured outline for an academic report based on the provided topic and facts.",
                    },
                    {
                        "role": "user",
                        "content": f"Topic: {topic}\n\nFacts:\n{fact_strings}",
                    },
                ],
            )
        except Exception:
            groq = get_groq_client()
            return groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                response_model=Outline,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a master planner. Create a highly structured outline for an academic report based on the provided topic and facts.",
                    },
                    {
                        "role": "user",
                        "content": f"Topic: {topic}\n\nFacts:\n{fact_strings}",
                    },
                ],
            )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def synthesize_report(
    topic: str, facts: List[Any], outline: Outline, feedback: str = ""
) -> ResearchReport:
    fact_strings = "\n".join(
        [f"- {f.statement} (Source: {f.source_url})" for f in facts]
    )
    outline_text = "\n".join(
        [f"## {s.title}\n{s.description}" for s in outline.sections]
    )
    messages = [
        {
            "role": "system",
            "content": "You are an expert researcher writing an academic report. Write a detailed, comprehensive report based ONLY on the provided facts, following the outline strictly. Include inline citations like [1], [2] referencing the source URLs.",
        },
        {
            "role": "user",
            "content": f"Topic: {topic}\n\nOutline Structure:\n{outline_text}\n\nFacts:\n{fact_strings}"
            + (
                f"\n\nCritic Feedback (Implement these revisions):\n{feedback}"
                if feedback
                else ""
            ),
        },
    ]
    try:
        with _cerebras_sem:
            client = get_cerebras_client()
            return client.chat.completions.create(
                model="gemma-4-31b",
                response_model=ResearchReport,
                messages=messages,
                max_tokens=6000,
            )
    except Exception:
        try:
            with _groq_sem:
                client = get_groq_client()
                return client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    response_model=ResearchReport,
                    messages=messages,
                    max_tokens=6000,
                )
        except Exception:
            with _sambanova_sem:
                client = get_sambanova_client()
                return client.chat.completions.create(
                    model="Meta-Llama-3.3-70B-Instruct",
                    response_model=ResearchReport,
                    messages=messages,
                    max_tokens=6000,
                )
