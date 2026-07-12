import os
import sys
import asyncio
from dotenv import load_dotenv
load_dotenv()
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_core.graph import research_graph
async def run_evaluation():
    topic = "The discovery and implications of Penicillin"
    print(f"Running LangGraph for topic: '{topic}'...")
    report = None
    facts = []
    async for output in research_graph.astream({"topic": topic, "iterations": 0}):
        for node, state in output.items():
            print(f"Finished node: {node}")
            if "report" in state:
                report = state["report"]
            if "facts" in state:
                facts = state["facts"]
    if not report:
        print("Graph failed to generate a report.")
        return
    answer_text = "\n".join([f"## {s.title}\n{s.content}" for s in report.sections])
    contexts = [f.statement for f in facts]
    if not contexts:
        contexts = ["No facts were retrieved."]
    print("\n--- LLM Evaluation Phase (Ragas) ---")
    try:
        groq_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"))
        hf_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")
    except Exception as e:
        print(f"Failed to initialize Ragas evaluators: {e}")
        return
    dataset = Dataset.from_dict({"question": [topic], "answer": [answer_text], "contexts": [contexts]})
    try:
        results = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy], llm=groq_llm, embeddings=hf_embeddings)
        print("\n=== RAGAS EVALUATION RESULTS ===")
        print(results)
        results_dict = dict(results) if hasattr(results, 'keys') else results
        with open("eval_results.txt", "w") as f:
            f.write("=== Ragas Evaluation Results ===\n")
            f.write(f"Topic: {topic}\n")
            f.write(f"Faithfulness (Hallucination check): {results_dict.get('faithfulness', 'N/A')}\n")
            f.write(f"Answer Relevancy: {results_dict.get('answer_relevancy', 'N/A')}\n")
        print("\nResults successfully saved to eval_results.txt")
    except Exception as e:
        print(f"Ragas evaluation failed: {e}")
if __name__ == "__main__":
    asyncio.run(run_evaluation())
