import json
import sys
from types import ModuleType

mock = ModuleType("langchain_community.chat_models.vertexai")
mock.ChatVertexAI = None
sys.modules["langchain_community.chat_models.vertexai"] = mock

from datasets import Dataset  # noqa: E402
from dotenv import dotenv_values  # noqa: E402
from langchain_community.embeddings import HuggingFaceBgeEmbeddings  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from ragas import RunConfig, evaluate  # noqa: E402
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness  # noqa: E402

rows = json.load(open(sys.argv[1], encoding="utf-8"))
env = dotenv_values(sys.argv[2])
judge = sys.argv[3] if len(sys.argv) > 3 else "nvidia"
ds = Dataset.from_dict({
    "question": [r["question"] for r in rows],
    "answer": [r["answer"] for r in rows],
    "contexts": [r["contexts"] for r in rows],
    "ground_truth": [r["ground_truth"] for r in rows],
})
if judge == "groq":
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=env["GROQ_API_KEY"], temperature=0, max_tokens=8192)
else:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="mistralai/mistral-large-4-1t-a52b",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=env["NVIDIA_API_KEY"],
        temperature=0,
        max_tokens=4096,
    )
emb = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")
answer_relevancy.strictness = 1
result = evaluate(
    dataset=ds,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=llm,
    embeddings=emb,
    run_config=RunConfig(max_workers=1, timeout=900, max_retries=15, max_wait=70),
)
df = result.to_pandas()
cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
for _, r in df.iterrows():
    print("ROW", json.dumps({"question": r.get("user_input", r.get("question")), **{c: round(float(r[c]), 3) for c in cols}}))
print("MEAN", json.dumps({c: round(float(df[c].mean()), 3) for c in cols}))
