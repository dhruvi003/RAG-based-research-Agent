from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    planner_node,
    retriever_node,
    critic_node,
    synthesizer_node
)

max_retries = 2

def should_retry(state:AgentState) -> str:
    """
    After critic evaluates context:
    - If sufficient → move to synthesizer
    - If not sufficient and retries left → loop back to retriever
    - If not sufficient and no retries left → synthesize anyway
    """
    if state["context_sufficient"]:
        print("\n✅ [Router] Context approved → Synthesizer")
        return "synthesize"
    
    if state["retry_count"] < max_retries:
        print(f"\n🔄 [Router] Context insufficient → Retry #{state['retry_count'] + 1}")
        return "retry"
    
    print("\n⚠️  [Router] Max retries reached → Synthesizer anyway")
    return "synthesize"


def increment_retry(state: AgentState) -> AgentState:
    current = state.get("retry_count") or 0
    return {"retry_count": current + 1}


def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("critic", critic_node)
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("synthesizer", synthesizer_node)

    # Entry point
    graph.set_entry_point("planner")

    # Edges
    graph.add_edge("planner", "retriever")
    
    graph.add_edge("retriever", "critic")

    # Conditional edge after critic
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {
            "synthesize": "synthesizer",
            "retry": "increment_retry"
        }
    )

    # Retry loop back to retriever
    graph.add_edge("increment_retry", "retriever")

    # End after synthesizer
    graph.add_edge("synthesizer", END)

    return graph.compile()