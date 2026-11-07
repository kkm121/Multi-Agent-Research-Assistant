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
| Conversation | Every message started a new, unrelated report ("repeat that" produced a random report) | **Chat memory**: an intent router sends each message to new research, a follow-up answered from earlier evidence, a repeat, or small talk; comparisons reuse earlier findings; memory is stored with the thread and survives reloads |
| Retrieval | First results from each search engine, as returned | **Hybrid ranking**: BM25 + Gemini embeddings fused with Reciprocal Rank Fusion picks the most relevant sources before extraction and the most relevant remembered facts for follow-ups; "recent/latest" questions filter academic search to the last 2 years |
| PDF RAG | `bge-m3` + torch + FAISS (~2.5 GB RAM, OOMs on free hosts) | In-process **BM25**: zero model downloads, milliseconds |
| Orchestration | LangGraph 0.0.x, personas in threads | LangGraph 1.2 with **`Send` fan-out** per persona and custom stream events |
| UI ↔ backend | Chainlit called its own server over HTTP | Chainlit runs the graph in-process; REST/SSE API kept for external clients |
| Auth | Plaintext passwords sent to the browser; users with no password could log in with **any** password; REST API open to anyone | Salted **scrypt** hashes; reserved owner accounts (`AUTH_USERS`), open signup or invite-only, per-user daily research limit; REST API requires an `X-API-Key`; other users can never open your chats |
| Thread history | Silently broken on SQLite (list binding error) | Fixed via a small data-layer adapter |
| UI | Default Chainlit | ChatGPT-style dark theme plus a light theme, research profiles, starters, live task list, citation chips, downloadable Markdown report, "Go deeper" action |
| Image | torch, sentence-transformers, FAISS, ragas, langchain-* | No ML frameworks; all deps pinned; non-root; health check |
| Tests | None | 30 offline tests (router, citations, retrieval, memory, auth and API access, full graph with a fake LLM) + live provider check + grounding eval + v1/v2 benchmark |

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

### Conversation memory

Each chat keeps a memory of its turns, reports, extracted facts and source excerpts, stored with the
thread so it survives reloads and resumed chats. Every new message is classified first:

| Message | What happens |
|---|---|
| A new topic | Full research run |
| A question about earlier results ("what did you find about X?", "which source said that?") | Answered from the stored evidence with hybrid retrieval and citations, no new search |
| A comparison with something new ("compare that with bio-science") | New research, seeded with the earlier facts so the report covers both |
| "Repeat that" / "show it again" | The previous answer, unchanged |
| Greetings, thanks, questions about the assistant | Short reply |

## Free APIs used

All of these work **without a credit card**. You need **at least one LLM key**. More keys give more headroom, since the router fails over automatically.
Status was verified live on 2026-11-07 with `scripts/check_providers.py`.

| Purpose | Provider | Get a key | Default models (fallback order) | Status 2026-11-07 |
|---|---|---|---|---|
| LLM | Google Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | 3.5-flash-lite, flash-lite-latest, 2.5-flash / 2.5-flash, 3.5-flash | Working, primary (≈15 RPM per model) |
| LLM | Groq | [console.groq.com/keys](https://console.groq.com/keys) | gpt-oss-20b, qwen3.8-27b / gpt-oss-120b | Working, fastest (<1 s) |
| LLM | OpenRouter | [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) | `openrouter/free`, nemotron-3-ultra `:free` | Working |
| LLM | NVIDIA NIM | [build.nvidia.com](https://build.nvidia.com/settings/api-keys) | mistral-large-4 | Working, slower (~8 s) |
| LLM | Mistral | [console.mistral.ai](https://console.mistral.ai/api-keys) | mistral-small / medium | Was rate-limited when tested |
| Search | Tavily *(optional)* | [app.tavily.com](https://app.tavily.com) | | Working, 1,000 searches/month |
| Search | OpenAlex, arXiv, Wikipedia, DuckDuckGo | none | | Working, keyless |
| Search | Semantic Scholar *(key recommended)* | [request a key](https://www.semanticscholar.org/product/api#api-key-form) | | Working with key; keyless pool often 429s |
| Reader | Jina Reader *(key optional)* | [jina.ai](https://jina.ai/api-dashboard) | | Working |

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

**Accounts.** Create your own owner entry and set it as `AUTH_USERS` on the host. Owner accounts
are reserved (nobody can register that name), have no usage limit, and survive redeploys because
the credentials live in the environment:

```bash
python scripts/create_account.py
```

| Setting | Who can use the app |
|---|---|
| `ALLOW_SIGNUP=true` (default) | Anyone can create their own account; each sees only their own chats; non-owners get `DAILY_RESEARCH_LIMIT` research runs per day (default 10, Deep counts as 2, follow-ups are free) |
| `ALLOW_SIGNUP=false` | Invite-only: only the `AUTH_USERS` accounts can log in |

> On free hosts with an ephemeral disk, the SQLite file (chat history and self-registered accounts)
> is reset on every redeploy. Owner accounts in `AUTH_USERS` keep working; other users sign up again.

## API

All endpoints except `/api/health` require the `X-API-Key` header to match the server's `API_KEY`.
If `API_KEY` is not set, the REST API is disabled (the web UI does not use it).

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness |
| `GET` | `/api/providers` | Configured LLM providers, call/failure counts, search backends |
| `POST` | `/api/research` | `{"topic", "depth": "quick"\|"deep", "document_text"?}` → full report JSON |
| `POST` | `/api/research/stream` | Same body; SSE events `progress` → `report` → `done` |

```bash
curl -N -X POST localhost:8000/api/research/stream -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" -d '{"topic": "perovskite solar cell stability"}'
```

## Evaluation

### v1 vs v2 benchmark (2026-11-07)

Same 4 questions, same API keys, one judge for both (see [`benchmarks/`](benchmarks/)). The judge
re-fetches every cited source independently (paper abstracts through the OpenAlex, Semantic Scholar
and arXiv APIs; web pages through Jina Reader) and checks a sample of up to 12 cited sentences per
report against that text. It is deliberately strict: a sentence fails if it adds any detail the
source does not state.

| | v1 as deployed | v1, patched to run* | v2 (two runs, 8 reports) |
|---|---|---|---|
| Reports produced | **0 / 4** (every provider failed) | 3 / 4 | **8 / 8** |
| References that point to a source actually retrieved | n/a | 22% (fake links such as `system.alert/ratelimit`, `example.com`) | **100%** (0 invented) |
| Cited sentences supported by the cited source | n/a | 0% (writer worked from model memory) | **73% average** (range 42–100%) |
| Relevance to the question (1–5) | n/a | 2.3 | **4.9** |
| Typical time, Quick mode | n/a | 1–2.5 min | 1–3.5 min |

<sub>*v1 code with only two changes so it could run at all: Groq's retired `llama-3.3-70b-versatile`
replaced by `gpt-oss-120b`, and Groq structured output switched from tool mode (which crashed on
the first schema mismatch) to JSON mode. Cerebras (402) and SambaNova (401) still failed, so v1 fell
back to its placeholder fact "System Alert: Rate Limit Exceeded" and wrote the reports from the
model's own memory.</sub>

How to read the v2 numbers: every v2 reference is real and every claim is tied to retrieved
evidence, but roughly 1 in 4 cited sentences adds a detail the source abstract does not state
explicitly (or paraphrases beyond it). Part of that is the judge seeing only an abstract where the
pipeline read more text; part is genuine embellishment, which the writer and fact-checker prompts now
explicitly forbid. Free-tier models and single runs make these numbers indicative, not precise.

### Ragas (same setup as the v1 README)

The v1 README reported Ragas scores of 0.95 faithfulness and 0.92 answer relevancy. Those numbers were
never produced by a Ragas run: the evaluation crashed and the figures were entered by hand. Running the
original `ragas_eval.py` today also fails, because the v1 pipeline has no working LLM provider.

Below is a real Ragas 0.4.3 run on v2 with the original script's setup: same topic ("The discovery and
implications of Penicillin"), same ground truth, same four metrics, `BAAI/bge-m3` embeddings and a Groq
judge (`gpt-oss-120b`, since `llama-3.3-70b-versatile` has been retired).

| Metric | v1 README (not measured) | v2 (measured, 2026-11-07) |
|---|---|---|
| Faithfulness | 0.95 | **0.57** |
| Answer relevancy | 0.92 | **0.55** |
| Context precision | 0.83 | **0.15** |
| Context recall | 0.88 | **0.50** |

Notes: single run on one topic. Ragas faithfulness compares the report with the one-line extracted
facts rather than the full source text, so wording beyond those facts counts against it (the source-text
benchmark above gives 73%). On this topic the experts focused on the "implications" half (antibiotic
resistance and costs), so the evidence missed parts of the ground truth (Fleming by name, WWII), which
lowers relevancy, precision and recall.

### Grounding check for a single question

`scripts/eval_grounding.py` runs the pipeline and checks every cited sentence against the text the
pipeline itself retrieved (a more lenient check than the benchmark above):

```bash
python scripts/eval_grounding.py "The discovery and impact of penicillin" quick
```

| Question | Mode | References | Cited sentences | Faithfulness | Coverage |
|---|---|---|---|---|---|
| Long-term effects of GLP-1 agonists (semaglutide) | Deep | 25 | 52 / 71 | 0.94 | 0.73 |

## Project layout

```
ai_core/
  config.py      settings from env
  llm.py         async multi-provider router (failover, rate-limit benching, structured output)
  rank.py        hybrid retrieval: BM25 + embeddings with Reciprocal Rank Fusion
  conversation.py chat memory, intent routing, follow-up answers
  search.py      parallel search backends + reader
  retrieval.py   BM25 over uploaded documents
  agents.py      persona / query / extraction / outline / writer / critic + citation verification
  graph.py       LangGraph pipeline and streaming runner
  models.py      Pydantic models
backend/main.py  FastAPI (REST + SSE) with the Chainlit UI mounted at /
frontend/        Chainlit app + password hashing
public/          theme, CSS, logos
scripts/         provider checker, grounding evaluation, account creation
tests/           offline test suite
docs/UPGRADE_TRACKER.md   v2 audit and verification log
```

## Design decisions

- **OpenAI-compatible endpoints only.** Every provider here speaks the OpenAI API, so one client type plus `instructor` covers all of them. Adding a provider is one line in `BASE_URLS`.
- **Benching instead of retry storms.** A 429 removes the provider from rotation until `retry-after` passes. Requests flow to the next provider instead of hammering one that is already limited.
- **Small structured calls, free-text writing.** Planning and extraction use small Pydantic schemas that small free models fill reliably. The long report is plain Markdown, avoiding fragile JSON-escaped essays.
- **Citations are verified in code, not by prompt.** Prompts reduce hallucinated citations, but only the source registry guarantees that every `[n]` in the final report points to a document the agents actually read.
- **BM25 over dense embeddings for uploads.** For a single uploaded document, keyword retrieval is fast, accurate enough, and needs no 2 GB model, which is what makes the app fit a free host.
