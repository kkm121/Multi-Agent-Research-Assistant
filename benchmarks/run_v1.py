import asyncio, json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
ROOT, OUT = sys.argv[1], sys.argv[2]
QUESTIONS = json.loads(sys.argv[3])
os.chdir(ROOT); sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))
from ai_core.graph import research_graph


async def one(q):
    t = time.time()
    report, facts = None, []
    try:
        async for out in research_graph.astream({"topic": q, "iterations": 0, "document_text": ""}):
            for node, st in out.items():
                if st and st.get("report") is not None:
                    report = st["report"]
                if st and st.get("facts"):
                    facts.extend(st["facts"])
    except Exception as e:
        return {"question": q, "error": f"{type(e).__name__}: {e}"[:300], "seconds": time.time() - t}
    md = f"# {report.title}\n\n" + "".join(f"## {s.title}\n{s.content}\n\n" for s in report.sections) if report else ""
    return {
        "question": q,
        "seconds": round(time.time() - t, 1),
        "markdown": md,
        "references": [r.model_dump() for r in getattr(report, "references", [])] if report else [],
        "facts": [{"statement": f.statement, "source_url": f.source_url} for f in facts],
    }


async def main():
    results = []
    for q in QUESTIONS:
        r = await one(q)
        print(json.dumps({k: r.get(k) for k in ("question", "seconds", "error")}), flush=True)
        results.append(r)
        await asyncio.sleep(20)
    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=1)

asyncio.run(main())
