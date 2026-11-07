# v1 vs v2 benchmark

Same questions, same keys, one judge for both versions.

```bash
git worktree add ../v1 70ebb61          # v1 code (needs its own venv and .env)
python benchmarks/run_v1.py ../v1 v1.json '["question 1", "question 2"]'
python benchmarks/run_v2.py . v2.json '["question 1", "question 2"]'
python benchmarks/judge.py . v1.json v2.json
```

Metrics per report:

| Metric | Meaning |
|---|---|
| `real_reference_pct` | References whose URL the pipeline actually retrieved |
| `faithfulness_pct` | Sampled cited sentences supported by the cited source (abstract via OpenAlex / Semantic Scholar / arXiv APIs for papers, page text via Jina Reader otherwise) |
| `unverifiable_sentences` | Cited sentences whose source text could not be fetched (excluded from faithfulness) |
| `cited_sentence_pct` | Sentences that carry a citation |
| `relevance_1to5` | How directly the report answers the question |
