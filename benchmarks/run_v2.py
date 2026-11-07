import asyncio, json, logging, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
ROOT, OUT = sys.argv[1], sys.argv[2]
QUESTIONS = json.loads(sys.argv[3])
os.chdir(ROOT); sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))
logging.basicConfig(level=logging.ERROR)
from ai_core.graph import run_research


async def one(q):
    t = time.time()
    report, ev = None, {"facts": [], "sources": {}}
    async for kind, p in run_research(q, "quick"):
        if kind == "report":
            report = p
        elif kind == "evidence":
            ev = p
    return {
        "question": q,
        "seconds": round(time.time() - t, 1),
        "markdown": f"# {report.title}\n\n{report.markdown}",
        "references": [r.model_dump() for r in report.references],
        "facts": [{"statement": f.statement, "source_url": f.source_url} for f in ev["facts"]],
        "retrieved_urls": list(ev["sources"].keys()),
    }


async def main():
    results = []
    for q in QUESTIONS:
        r = await one(q)
        print(json.dumps({k: r.get(k) for k in ("question", "seconds")}), flush=True)
        results.append(r)
        await asyncio.sleep(20)
    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=1)

asyncio.run(main())
