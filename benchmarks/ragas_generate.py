import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.ERROR)

from ai_core.graph import run_research  # noqa: E402


async def one(item: dict) -> dict:
    t = time.time()
    report, ev = None, {"selected": []}
    async for kind, payload in run_research(item["question"], "quick"):
        if kind == "report":
            report = payload
        elif kind == "evidence":
            ev = payload
    return {
        **item,
        "answer": report.markdown,
        "contexts": [f.statement for f in ev["selected"]],
        "references": len(report.references),
        "seconds": round(time.time() - t, 1),
    }


async def main(src: str, out: str, only: list[int]) -> None:
    items = json.load(open(src, encoding="utf-8"))
    rows = []
    for i, item in enumerate(items):
        if only and i not in only:
            continue
        row = await one(item)
        print(json.dumps({k: row[k] for k in ("question", "references", "seconds")}), flush=True)
        rows.append(row)
        await asyncio.sleep(15)
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]))
