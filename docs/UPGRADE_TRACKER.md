# v2 Upgrade Tracker

Working log for the v2 upgrade. Every item is only ticked after it was actually run/verified
locally — "verified" notes say how. Nothing here is assumed.

## Problems found in v1 (audit, 2026-11-07)

| # | Problem | Where | Severity |
|---|---------|-------|----------|
| 1 | Cerebras removed its no-card free tier (Jul 2026) — primary provider for most agents is gone | `ai_core/agents.py` | **Breaking** |
| 2 | `BAAI/bge-m3` + torch + sentence-transformers (~2.5 GB RAM) cannot run on Render free (512 MB) — PDF RAG would OOM | `agents.py` | **Breaking** |
| 3 | Login: users with no stored password log in with *any* password; passwords stored in plaintext; anyone can auto-register | `frontend/app.py` | **Security** |
| 4 | Citations `[1]`, `[2]` are invented by the LLM — facts are not numbered, so references are hallucinated | `synthesize_report` | Correctness |
| 5 | Critic never sees the facts, so it cannot detect unsupported claims (the "reflexion" was cosmetic) | `critique_report` | Correctness |
| 6 | Same 3-provider fallback copy-pasted ~8× (≈300 lines); no rate-limit awareness, sequential timeouts | `agents.py` | Maintainability |
| 7 | Sync LLM/HTTP calls in threads; blocking `time.sleep` inside a global lock for Semantic Scholar | `tools.py` | Performance |
| 8 | SSE `done` event only sent on error (indentation bug) | `backend/main.py` | Bug |
| 9 | Chainlit calls its *own* server over HTTP loopback to run the graph | `frontend/app.py` | Architecture |
| 10 | `duckduckgo_search` package renamed to `ddgs`; old one is broken | `tools.py` | **Breaking** |
| 11 | Unpinned deps; `ragas`, vertexai, langchain-* in runtime image (huge, unused at runtime) | `requirements.txt` | Ops |
| 12 | Only one research round (`range(1)`) and one query per persona — very thin evidence | `conduct_research` | Quality |
| 13 | SQLite on Render free disk is wiped on every redeploy — users/chat history lost | deploy | Ops |

## Provider choice (all free, no credit card)

| Role | Provider | Why | Key needed |
|------|----------|-----|------------|
| LLM primary | Google Gemini (AI Studio) | Strong models, no card | `GEMINI_API_KEY` |
| LLM fast/fallback | Groq | ~1k req/day/model, very fast, no card | `GROQ_API_KEY` |
| LLM fallback | OpenRouter `:free` models | Many models, no card (50 req/day) | `OPENROUTER_API_KEY` |
| LLM fallback (optional) | GitHub Models | Already have a GitHub account | `GITHUB_TOKEN` |
| LLM fallback (optional) | Mistral (Experiment plan) | Phone verification, no card | `MISTRAL_API_KEY` |
| Web search | Tavily | 1,000 credits/month, no card | `TAVILY_API_KEY` (optional) |
| Web search fallback | DuckDuckGo (`ddgs`) | No key | — |
| Academic | OpenAlex, arXiv, Semantic Scholar | No key | — |
| Page reader | Jina Reader | Works keyless; key raises limits | `JINA_API_KEY` (optional) |
| Doc retrieval | In-process BM25 | Zero RAM/network, replaces bge-m3 | — |

Model IDs are env-configurable; `scripts/check_providers.py` verifies each one live.

## Tasks

- [x] Branch `upgrade/v2-free-apis`
- [x] Pinned, slim `requirements.txt` (ragas dropped; replaced by `scripts/eval_grounding.py`)
- [x] `ai_core/config.py` — settings from env
- [x] `ai_core/llm.py` — async provider router (bench on 429, skips unconfigured/bad keys, instructor)
- [x] `ai_core/search.py` — async multi-source search + reader, dedup
- [x] `ai_core/retrieval.py` — BM25 for uploaded docs
- [x] `ai_core/agents.py` — rewritten agents using router
- [x] `ai_core/graph.py` — LangGraph 1.2, `Send` fan-out, grounded critic, custom stream events
- [x] Deterministic references built from source registry (no invented citations)
- [x] `backend/main.py` — SSE `done` always sent, `/api/providers`, Chainlit mounted
- [x] `frontend/app.py` — graph in-process, task list, profiles, starters, download, "Go deeper"
- [x] Auth: scrypt hashes in separate table, v1 migration, no passwordless login
- [ ] ~~Optional Postgres~~ — deferred: can't verify locally without a Postgres instance; SQLite only
- [x] UI theme, CSS, logos, favicon, login image, `.chainlit/config.toml`
- [x] Dockerfile (python 3.12-slim, non-root, healthcheck) — **not built locally (no Docker here)**
- [x] `scripts/check_providers.py`
- [x] Tests — 20 passing
- [x] **Local E2E with real keys** — Quick + Deep + UI runs (see log)
- [x] Default model IDs re-checked live; replaced retired ones
- [x] Local UI: login, starters, profile switch, failure path, thread history, resume
- [x] README rewrite
- [ ] Push branch (only after user OK) → merge to `main` triggers Render auto-deploy

## Verification log

- 2026-11-07 — Deps installed into Python 3.11 venv; pinned versions recorded in requirements.txt.
- 2026-11-07 — `pytest`: 11 unit tests pass (router failover/benching, bad-key skip, citations renumber +
  invented-id removal, BM25, chunking, OpenAlex abstract, URL normalise, scrypt).
- 2026-11-07 — `check_providers.py` (no LLM keys): OpenAlex OK (3), arXiv OK (3), Wikipedia OK (2),
  DuckDuckGo OK (5), Jina Reader OK (6000 chars). Semantic Scholar keyless → 429 (shared pool);
  pipeline continues without it. Tavily skipped (no key).
- 2026-11-07 — Server boots; `/api/health` 200, `/api/providers` lists backends.
- 2026-11-07 — Browser: theme/logo applied, login creates account, starters render; with no LLM key a
  run shows "No AI provider configured" and task list marks failed (no crash).
- 2026-11-07 — **Bug found & fixed**: Chainlit SQLAlchemy layer binds the thread tag list → SQLite
  `type 'list' is not supported` → threads never saved (same bug existed in v1). Added adapter; verified
  thread appears in sidebar, tags stored as `["Quick research"]`, resume shows history.
- 2026-11-07 — **Bug found & fixed**: uploaded doc ignored when queries share no keywords with it; now
  matches query+topic and falls back to opening chunks. Covered by `test_uploaded_document_becomes_a_source`.
- 2026-11-07 — Graph integration tests (fake LLM, stub search): quick & deep fan-out counts, revision
  loop, invented `[42]` stripped, doc source cited, empty-evidence path. 4 pass.
- 2026-11-07 — **Edge case fixed**: after a DB reset with a live session cookie Chainlit recreates a
  credential-less user row → user would be locked out forever. Now reclaimable. `test_auth_flow` covers
  signup/login/wrong pw/v1 migration/orphan reclaim — passes.

### Live verification with real keys (2026-11-07)

- Probed every key the user supplied (pasted + harness `.env`), never printing values:
  - Gemini ✅, Groq ✅, OpenRouter ✅, NVIDIA ✅ (only `mistral-large-4` answers; others 404/timeout)
  - **Cerebras ❌ 402 Payment required on all keys** (confirms the July 2026 free-tier change)
  - **SambaNova ❌ 401 on pasted and harness keys**
  - Mistral ⚠️ 429 on every key (account-level quota) — kept as last fallback
  - GitHub Models: retired 2026-07-30 per harness notes → provider removed
  - `gemini-2.5-flash-lite` → 404 "no longer available"; Groq `llama-3.1-8b-instant` gone → defaults replaced
  - Tavily ✅, Jina ✅, Semantic Scholar ✅ with key
- One key per provider is used. Multiple keys from separate accounts were deliberately **not** rotated
  (providers' terms generally forbid multiplying free quota across accounts).
- **Bug found & fixed**: `instructor` wraps provider errors in `InstructorRetryException`, so the router never
  recognised 429s on structured calls. Added `root_api_error()` unwrapping + test.
- **Bug found & fixed**: Gemini (15 RPM) and Groq (TPM) quotas are *per model*; router benched the whole
  provider. Now benches only the model, parses "retry in 51s" from the body. Tests added.
- **Bug found & fixed**: a free fallback model returned a 4-sentence "revision" that replaced a good draft
  (Deep run produced 1 reference from 31 sources). Added `check_draft` gate (≥250 words, ≥3 citations, a
  revision may not shrink >40%) and "keep first draft if revision fails". Tests added.
- **Bug found & fixed**: citation counter undercounted adjacent `[1] [2]` groups (caught by new test).
- **Bug found & fixed**: test isolation — a test loaded the real `.env`, so the "no keys" test made a live call.
  Added autouse fixture clearing provider env vars.
- Gemini 5xx are now treated as model-specific (try next model) instead of benching the provider.
- Quick E2E (script): solid-state batteries — 55 s, 3 experts, 15 facts, 10 refs, revision loop ran, Gemini→Groq failover observed.
- Deep E2E + grounding eval: GLP-1 — 25 refs, 71 sentences, 52 cited, **faithfulness 0.942**, coverage 0.732
  (5 m 41 s including 52 judge calls). Report read manually: accurate, peer-reviewed sources, exact figures.
- UI E2E in browser: login → starter → live agent stream + task list → report in 35 s, 13 refs,
  23 citation pills linking to sources with title tooltips, References list, `.md` download, "Go deeper" button.
- `pytest`: 20 passed.

## Known limitations (honest)

- SQLite on Render free disk resets on redeploy (accounts + history). Fix = external DB (follow-up).
- Downloadable `.md` attachment is shown live but not persisted in resumed threads (Chainlit needs blob
  storage for elements); the report text itself is persisted.
- Chainlit logs a harmless `chat_profiles was never awaited` warning from its own config-override path.
