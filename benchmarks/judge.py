import asyncio, json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
ROOT = sys.argv[1]
FILES = sys.argv[2:] if __name__ == "__main__" else []
os.chdir(ROOT); sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))
import logging
logging.basicConfig(level=logging.ERROR)
from pydantic import BaseModel, Field
from ai_core.llm import get_router
from ai_core.search import _client, jina_read, normalize_url

SAMPLE = 12
_pages: dict[str, str] = {}


class Verdict(BaseModel):
    supported: bool


class Relevance(BaseModel):
    score: int = Field(description="1-5: how directly and completely the report answers the question")


def sentences(md: str) -> list[str]:
    text = re.sub(r"^#+ .*$", "", md, flags=re.M)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", text)
    return [p.strip(" -*|") for p in parts if len(p.split()) >= 6]


async def _abstract(client, url: str) -> str:
    from ai_core.search import _openalex_abstract
    m = re.search(r"semanticscholar\.org/paper/(?:.*/)?([0-9a-f]{40})", url)
    if m:
        h = {"x-api-key": os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")}
        for _ in range(3):
            r = await client.get(f"https://api.semanticscholar.org/graph/v1/paper/{m.group(1)}",
                                 params={"fields": "title,abstract,tldr"}, headers=h)
            if r.status_code == 429:
                await asyncio.sleep(2); continue
            if r.status_code == 200:
                d = r.json()
                return f"{d.get('title','')}\n{d.get('abstract') or ''}\n{(d.get('tldr') or {}).get('text','')}"
            break
        return ""
    m = re.search(r"doi\.org/(10\..+)$", url)
    if m or "openalex.org/W" in url:
        key = f"doi:{m.group(1)}" if m else url.rsplit("/", 1)[1]
        r = await client.get(f"https://api.openalex.org/works/{key}")
        if r.status_code == 200:
            d = r.json()
            return f"{d.get('title','')}\n{_openalex_abstract(d.get('abstract_inverted_index'))}"
        return ""
    m = re.search(r"arxiv\.org/abs/([^\s?#]+)", url)
    if m:
        r = await client.get("https://export.arxiv.org/api/query", params={"id_list": re.sub(r"v\d+$", "", m.group(1))})
        s = re.search(r"<summary>(.*?)</summary>", r.text, re.S)
        t = re.search(r"<entry>.*?<title>(.*?)</title>", r.text, re.S)
        return f"{t.group(1) if t else ''}\n{s.group(1) if s else ''}"
    return None


async def page(client, url: str) -> str:
    if url not in _pages:
        text = ""
        if url.startswith("http"):
            try:
                text = await _abstract(client, url)
            except Exception:
                text = ""
            if text is None:
                text = await jina_read(client, url, max_chars=12000)
        _pages[url] = text if len(text.split()) >= 30 else ""
    return _pages[url]


async def judge_report(client, r: dict) -> dict:
    if r.get("error") or not r.get("markdown"):
        return {"question": r["question"], "failed": r.get("error", "no report")}
    refs = {int(x["id"]): x.get("url", "") for x in r["references"]}
    retrieved = {normalize_url(u) for u in (r.get("retrieved_urls") or [f["source_url"] for f in r["facts"]]) if u}
    real_refs = sum(1 for u in refs.values() if u.startswith("http") and normalize_url(u) in retrieved)
    sents = sentences(r["markdown"])
    cited = [(s, [int(n) for n in re.findall(r"\[(\d+)\]", s)]) for s in sents]
    cited = [(s, ns) for s, ns in cited if ns]
    step = max(1, len(cited) // SAMPLE)
    sample = cited[::step][:SAMPLE]
    router = get_router()
    verdicts = []
    unverifiable = 0
    for s, ns in sample:
        urls = [refs.get(n, "") for n in ns]
        texts = [await page(client, u) for u in urls]
        ctx = "\n\n".join(f"SOURCE {u}\n{t[:5000]}" for u, t in zip(urls, texts) if t)
        if not ctx:
            unverifiable += 1
            continue
        claim = re.sub(r"\[\d+\]", "", s)
        try:
            v = await router.structured(Verdict, [
                {"role": "system", "content": "Does the source text explicitly support the claim? Be strict about numbers, dates and names."},
                {"role": "user", "content": f"Claim: {claim}\n\n{ctx}"},
            ])
            verdicts.append(v.supported)
        except Exception as e:
            print("judge error", str(e)[:100])
    rel = await router.structured(Relevance, [
        {"role": "system", "content": "Rate how directly and completely this report answers the question (1-5). Ignore style."},
        {"role": "user", "content": f"Question: {r['question']}\n\nReport:\n{r['markdown'][:9000]}"},
    ])
    return {
        "question": r["question"],
        "seconds": r["seconds"],
        "words": len(r["markdown"].split()),
        "references": len(refs),
        "real_reference_pct": round(100 * real_refs / max(1, len(refs)), 1),
        "invented_references": len(refs) - real_refs,
        "cited_sentence_pct": round(100 * len(cited) / max(1, len(sents)), 1),
        "faithfulness_pct": round(100 * sum(verdicts) / max(1, len(verdicts)), 1),
        "judged_sentences": len(verdicts),
        "unverifiable_sentences": unverifiable,
        "relevance_1to5": rel.score,
    }


async def main():
    out = {}
    async with _client() as client:
        for f in FILES:
            rows = [await judge_report(client, r) for r in json.load(open(f, encoding="utf-8"))]
            out[os.path.basename(f)] = rows
            print(f"\n== {f}")
            for row in rows:
                print(json.dumps(row))
    json.dump(out, open(os.path.join(os.path.dirname(FILES[0]), "judged.json"), "w"), indent=1)

if __name__ == "__main__":
    asyncio.run(main())
