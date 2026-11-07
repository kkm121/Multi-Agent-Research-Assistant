"""Measure how well report claims are grounded in the sources they cite.

For every cited sentence in the final report, an LLM judge reads the cited source text and decides
whether it supports the sentence. Reports:
  * citation_faithfulness — supported cited sentences / all cited sentences
  * citation_coverage     — sentences with a citation / all factual-looking sentences

    python scripts/eval_grounding.py "The discovery of penicillin" [quick|deep]
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from pydantic import BaseModel  # noqa: E402

from ai_core import agents  # noqa: E402
from ai_core.graph import research_graph  # noqa: E402
from ai_core.llm import get_router  # noqa: E402


class Verdict(BaseModel):
    supported: bool
    reason: str


def sentences(md: str) -> list[str]:
    text = re.sub(r"^#+ .*$", "", md, flags=re.M)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", text)
    return [p.strip(" -*|") for p in parts if len(p.split()) >= 6]


async def main(topic: str, depth: str) -> None:
    state = await research_graph.ainvoke({"topic": topic, "depth": depth})
    report = state["report"]
    ids = agents.assign_ids(state["facts"], state["sources"])
    # finalize renumbered citations; rebuild a map from final ref id -> source
    by_url = state["sources"]
    ref_src = {r.id: by_url[r.url] for r in report.references if r.url in by_url}

    sents = sentences(report.markdown)
    cited = [(s, [int(n) for n in re.findall(r"\[(\d+)\]", s)]) for s in sents]
    cited = [(s, ns) for s, ns in cited if ns]
    router = get_router()

    async def judge(sentence: str, nums: list[int]) -> bool:
        ctx = "\n\n".join(
            f"[{n}] {ref_src[n].title}\n{(ref_src[n].content or ref_src[n].snippet)[:3000]}" for n in nums if n in ref_src
        )
        if not ctx:
            return False
        claim = re.sub(r"\[\d+\]", "", sentence)
        v = await router.structured(
            Verdict,
            [
                {"role": "system", "content": "Decide if the sources explicitly support the claim. Be strict about numbers and names."},
                {"role": "user", "content": f"Claim: {claim}\n\nSources:\n{ctx}"},
            ],
        )
        return v.supported

    verdicts = []
    for s, ns in cited:  # sequential: stays within free-tier rate limits
        try:
            verdicts.append(await judge(s, ns))
        except Exception as e:
            print("judge error:", e)
    result = {
        "topic": topic,
        "depth": depth,
        "sources_used": len(ids),
        "references": len(report.references),
        "sentences": len(sents),
        "cited_sentences": len(cited),
        "citation_coverage": round(len(cited) / max(1, len(sents)), 3),
        "citation_faithfulness": round(sum(verdicts) / max(1, len(verdicts)), 3),
        "llm_calls": dict(router.stats.calls),
    }
    print(json.dumps(result, indent=2))
    Path("eval_results.json").write_text(json.dumps(result, indent=2))
    refs = "\n".join(f"{r.id}. [{r.origin}] {r.title} — {r.url}" for r in report.references)
    Path("eval_report.md").write_text(f"# {report.title}\n\n{report.markdown}\n\n## References\n{refs}\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "The discovery and impact of penicillin",
                     sys.argv[2] if len(sys.argv) > 2 else "quick"))
