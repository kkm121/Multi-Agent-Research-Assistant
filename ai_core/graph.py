from typing import TypedDict, List, Annotated, Optional
from langgraph.graph import StateGraph, END
import operator
from .models import Persona, Fact, ResearchReport, Outline
from .agents import (
    generate_personas,
    conduct_research,
    synthesize_report,
    critique_report,
    generate_outline,
)


class GraphState(TypedDict):
    topic: str
    personas: List[Persona]
    facts: Annotated[List[Fact], operator.add]
    outline: Optional[Outline]
    report: Optional[ResearchReport]
    feedback: str
    actionable_issues: bool
    iterations: int
    document_text: str


def init_research(state: GraphState):
    personas_list = generate_personas(state["topic"])
    return {"personas": personas_list.personas, "iterations": 0}


async def research_phase(state: GraphState):
    import asyncio

    all_facts = []

    async def run_persona(p):
        return await asyncio.to_thread(
            conduct_research,
            state["topic"],
            p.name,
            p.role,
            state.get("document_text", ""),
        )

    tasks = [run_persona(p) for p in state.get("personas", [])]
    results = await asyncio.gather(*tasks)
    for facts_list in results:
        all_facts.extend(facts_list.facts)
    return {"facts": all_facts}


def outline_phase(state: GraphState):
    outline = generate_outline(state["topic"], state.get("facts", []))
    return {"outline": outline}


def synthesis_phase(state: GraphState):
    try:
        report = synthesize_report(
            state["topic"],
            state.get("facts", []),
            state["outline"],
            state.get("feedback", ""),
        )
    except Exception as e:
        from .models import ResearchReport, Section

        report = ResearchReport(
            title=f"Research on {state.get('topic', 'Topic')}",
            sections=[
                Section(
                    title="Generation Failed",
                    content=f"The system was unable to synthesize the final report due to LLM API limitations or validation errors.\n\nError details: {e}",
                )
            ],
            references=[],
        )
    return {"report": report}


def critique_phase(state: GraphState):
    report = state.get("report")
    if not report:
        return {
            "feedback": "",
            "actionable_issues": False,
            "iterations": state.get("iterations", 0) + 1,
        }
    feedback = critique_report(report, state.get("facts", []))
    return {
        "feedback": feedback.feedback if feedback.actionable_issues else "",
        "actionable_issues": feedback.actionable_issues,
        "iterations": state.get("iterations", 0) + 1,
    }


def should_revise(state: GraphState):
    if state.get("actionable_issues") and state.get("iterations", 0) < 2:
        return "revise"
    return "end"


workflow = StateGraph(GraphState)
workflow.add_node("init", init_research)
workflow.add_node("research", research_phase)
workflow.add_node("outline", outline_phase)
workflow.add_node("synthesis", synthesis_phase)
workflow.add_node("critique", critique_phase)
workflow.set_entry_point("init")
workflow.add_edge("init", "research")
workflow.add_edge("research", "outline")
workflow.add_edge("outline", "synthesis")
workflow.add_edge("synthesis", "critique")
workflow.add_conditional_edges(
    "critique", should_revise, {"revise": "synthesis", "end": END}
)
research_graph = workflow.compile()
