# Multi-Agent AI Research Assistant

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-1.2-green.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/UI-Chainlit%202.12-orange.svg" alt="Chainlit">
  <img src="https://img.shields.io/badge/Backend-FastAPI-red.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/APIs-100%25%20free%20tier-brightgreen.svg" alt="Free APIs">
  <a href="https://multi-agent-research-assistant-kbzb.onrender.com"><img src="https://img.shields.io/badge/Live_Demo-Render-brightgreen.svg"></a>
</p>

> **Live demo:** [multi-agent-research-assistant-kbzb.onrender.com](https://multi-agent-research-assistant-kbzb.onrender.com)

A panel of AI expert agents researches your question across the web and academic databases,
extracts evidence, writes a report, and **fact-checks every claim against the sources it cites**
before you see it. It runs entirely on **free, no-credit-card APIs** and fits in a free 512 MB host.

---

### Contents
- [What's new in v2](#whats-new-in-v2)
- [How it works](#how-it-works)
- [Free APIs used](#free-apis-used)
- [Run it locally](#run-it-locally)
- [Deploy](#deploy)
- [API](#api)
- [Evaluation](#evaluation)
- [Project layout](#project-layout)
- [Design decisions](#design-decisions)

---

## What's new in v2

| Area | v1 | v2 |
|---|---|---|
| LLMs | Cerebras (now returns *402 Payment required*), Groq, SambaNova (keys rejected) | **Gemini, Groq, OpenRouter, NVIDIA NIM, Mistral**: all free, no card, any subset works |
| Failover | Same try/except copy-pasted ~8×, no rate-limit awareness | One async **router** with an ordered model list per provider: benches only the rate-limited *model* (free quotas are per model), reads the provider's retry delay, skips retired models and bad keys, rejects degenerate drafts |
| Search | Semantic Scholar → DuckDuckGo (`duckduckgo_search`, now broken) | **Tavily, OpenAlex, arXiv, Semantic Scholar, Wikipedia, DuckDuckGo (`ddgs`)** in parallel, deduped, + Jina Reader for full page text |
| Citations | `[1]` numbers invented by the LLM, so references could be hallucinated | Facts are tied to real retrieved sources; citations to non-existent sources are **removed in code**; the reference list is built from the source registry, not by the LLM |
| Reflexion | Critic never saw the evidence | Fact-checker compares the draft to the evidence list; unsupported claims trigger a rewrite |
| Research depth | 1 query, 1 fact per persona | Quick (3 agents) or Deep (4 agents × 2 gap-filling rounds) |
| PDF RAG | `bge-m3` + torch + FAISS (~2.5 GB RAM, OOMs on free hosts) | In-process **BM25**: zero model downloads, milliseconds |
| Orchestration | LangGraph 0.0.x, personas in threads | LangGraph 1.2 with **`Send` fan-out** per persona and custom stream events |
| UI ↔ backend | Chainlit called its own server over HTTP | Chainlit runs the graph in-process; REST/SSE API kept for external clients |
| Auth | Plaintext passwords sent to the browser; users with no password could log in with **any** password | Salted **scrypt** hashes in a separate table, automatic migration of v1 accounts |
| Thread history | Silently broken on SQLite (list binding error) | Fixed via a small data-layer adapter |
| UI | Default Chainlit | Custom theme and logo, research profiles, starters, live task list, citation chips, downloadable Markdown report, "Go deeper" action |
| Image | torch, sentence-transformers, FAISS, ragas, langchain-* | No ML frameworks; all deps pinned; non-root; health check |
| Tests | None | 20 offline tests (router, citations, BM25, auth flow, full graph with a fake LLM) + live provider check + grounding eval |

## How it works

```mermaid
graph LR
    U((Question<br/>+ optional PDF)) --> P[Orchestrator<br/>expert panel]
    P -->|Send| R1[Expert 1]
    P -->|Send| R2[Expert 2]
    P -->|Send| R3[Expert N]
    R1 & R2 & R3 --> O[Planner<br/>outline]
    O --> W[Writer]
    W --> F{Fact-checker}
    F -- unsupported claims --> W
    F -- grounded --> E[Editor<br/>verify citations]
    E --> Rep((Report +<br/>references))

    subgraph each expert, per round
      Q[plan queries] --> S[parallel search<br/>web + academic + PDF] --> X[extract facts<br/>tied to source ids]
    end
```

1. **Orchestrator** creates 3–4 experts with non-overlapping angles (STORM-style multi-perspective research).
2. **Experts run in parallel** (`Send`). Each plans queries, picks web/academic/both, searches every backend at once, and extracts facts. The LLM only sees local source ids, which are mapped back to URLs in code, so a fact can't point at a source that wasn't retrieved. In Deep mode a second round targets gaps.
3. **Planner** builds an outline from the evidence.
4. **Writer** drafts using only the numbered evidence, citing `[n]`.
5. **Fact-checker** compares the draft with the evidence. Unsupported claims or wrong citations send it back to the writer (up to `MAX_REVISIONS`).
6. **Editor** renumbers citations by first use, deletes any that don't map to a real source, and builds the reference list.

Progress from every agent streams live into the UI.

## Free APIs used

All of these work **without a credit card**. You need **at least one LLM key**. More keys give more headroom, since the router fails over automatically.
Status was verified live on 2026-11-07 with `scripts/check_providers.py`.

| Purpose | Provider | Get a key | Default models (fallback order) | Status 2026-11-07 |
|---|---|---|---|---|
| LLM | Google Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | 3.5-flash-lite, flash-lite-latest, 2.5-flash / 2.5-flash, 3.5-flash | ✅ primary (≈15 RPM per model) |
| LLM | Groq | [console.groq.com/keys](https://console.groq.com/keys) | gpt-oss-20b, qwen3.8-27b / gpt-oss-120b | ✅ fastest (<1 s) |
| LLM | OpenRouter | [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) | `openrouter/free`, nemotron-3-ultra `:free` | ✅ |
| LLM | NVIDIA NIM | [build.nvidia.com](https://build.nvidia.com/settings/api-keys) | mistral-large-4 | ✅ slower (~8 s) |
| LLM | Mistral | [console.mistral.ai](https://console.mistral.ai/api-keys) | mistral-small / medium | ⚠️ was rate-limited when tested |
| Search | Tavily *(optional)* | [app.tavily.com](https://app.tavily.com) | | ✅ 1,000 searches/month |
| Search | OpenAlex, arXiv, Wikipedia, DuckDuckGo | none | | ✅ keyless |
| Search | Semantic Scholar *(key recommended)* | [request a key](https://www.semanticscholar.org/product/api#api-key-form) | | ✅ with key; keyless pool often 429s |
| Reader | Jina Reader *(key optional)* | [jina.ai](https://jina.ai/api-dashboard) | | ✅ |

**Not used, and why:** Cerebras ended its no-card free tier in July 2026 (API now returns 402), GitHub Models was retired on 30 July 2026, and SambaNova rejected every tested key.

Free-tier models get retired often (for example `gemini-2.5-flash-lite` in 2026). When a provider reports a model as gone, the router moves to the next model in that provider's list. To see what your keys can use right now:

```bash
python scripts/check_providers.py --models
```

Model lists can be overridden per provider in `.env` (see [`.env.example`](.env.example)).

## Run it locally

```bash
git clone https://github.com/kkm121/Multi-Agent-Research-Assistant.git
cd Multi-Agent-Research-Assistant
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add at least one LLM key
chainlit create-secret      # paste the value into CHAINLIT_AUTH_SECRET in .env
python init_chainlit_db.py
uvicorn backend.main:app --port 8000
```

Open http://localhost:8000. The first login with a new username creates the account
(set `ALLOW_SIGNUP=false` to disable).

Run the tests (no keys or network needed):

```bash
pip install pytest pytest-asyncio
pytest
```

## Deploy

The repo ships a `Dockerfile` that listens on `$PORT` (default `7860`).

**Render** (current live demo): Web Service → Docker → set the env vars from `.env.example`.
Pushing to the branch Render watches (`main`) triggers an automatic redeploy when Auto-Deploy is on.

**Hugging Face Spaces**: Docker Space, add the keys as *Secrets*.

> On free hosts with an ephemeral disk, the SQLite file (accounts and chat history) is reset on every
> redeploy. Users can log in again with any password to reclaim their username.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness |
| `GET` | `/api/providers` | Configured LLM providers, call/failure counts, search backends |
| `POST` | `/api/research` | `{"topic", "depth": "quick"\|"deep", "document_text"?}` → full report JSON |
| `POST` | `/api/research/stream` | Same body; SSE events `progress` → `report` → `done` |

```bash
curl -N -X POST localhost:8000/api/research/stream \
  -H "Content-Type: application/json" -d '{"topic": "perovskite solar cell stability"}'
```

## Evaluation

`scripts/eval_grounding.py` runs the pipeline and uses an LLM judge to check every cited sentence
against the text of the source it cites:

```bash
python scripts/eval_grounding.py "The discovery and impact of penicillin" quick
```

It reports **citation faithfulness** (supported cited sentences / cited sentences) and
**citation coverage** (cited sentences / all sentences), and saves them to `eval_results.json`.

Measured on 2026-11-07 (free-tier models, single runs, so treat as indicative):

| Question | Mode | Time | References | Cited sentences | Faithfulness | Coverage |
|---|---|---|---|---|---|---|
| Long-term effects of GLP-1 agonists (semaglutide) | Deep | 5 min 41 s* | 25 | 52 / 71 | **0.94** | 0.73 |
| Solid-state batteries: state and blockers | Quick | 55 s | 10 | — | — | — |
| Small vs large language models on-device | Quick (via UI) | 35 s | 13 | — | — | — |

<sub>*Includes the evaluation itself (52 sequential judge calls); the research pipeline alone is shorter.</sub>

## Project layout

```
ai_core/
  config.py      settings from env
  llm.py         async multi-provider router (failover, rate-limit benching, structured output)
  search.py      parallel search backends + reader
  retrieval.py   BM25 over uploaded documents
  agents.py      persona / query / extraction / outline / writer / critic + citation verification
  graph.py       LangGraph pipeline and streaming runner
  models.py      Pydantic models
backend/main.py  FastAPI (REST + SSE) with the Chainlit UI mounted at /
frontend/        Chainlit app + password hashing
public/          theme, CSS, logos
scripts/         provider checker, grounding evaluation
tests/           offline test suite
docs/UPGRADE_TRACKER.md   v2 audit and verification log
```

## Design decisions

- **OpenAI-compatible endpoints only.** Every provider here speaks the OpenAI API, so one client type plus `instructor` covers all of them. Adding a provider is one line in `BASE_URLS`.
- **Benching instead of retry storms.** A 429 removes the provider from rotation until `retry-after` passes. Requests flow to the next provider instead of hammering one that is already limited.
- **Small structured calls, free-text writing.** Planning and extraction use small Pydantic schemas that small free models fill reliably. The long report is plain Markdown, avoiding fragile JSON-escaped essays.
- **Citations are verified in code, not by prompt.** Prompts reduce hallucinated citations, but only the source registry guarantees that every `[n]` in the final report points to a document the agents actually read.
- **BM25 over dense embeddings for uploads.** For a single uploaded document, keyword retrieval is fast, accurate enough, and needs no 2 GB model, which is what makes the app fit a free host.
