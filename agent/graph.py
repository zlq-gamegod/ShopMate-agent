from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from agent.state import AgentState
from agent.nodes import route_node, retrieve_node, agent_node, handoff_node
from agent.tools import TOOLS


def _after_retrieve(state: AgentState) -> str:
    return "handoff" if not (state.get("retrieved") or "").strip() else "agent"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("handoff", handoff_node)

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        lambda s: s["route"],
        {"policy": "retrieve", "other": "agent", "boundary": "handoff"},
    )
    g.add_conditional_edges("retrieve", _after_retrieve,
                            {"agent": "agent", "handoff": "handoff"})
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    g.add_edge("handoff", END)

    return g.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    g = build_graph()
    out = g.invoke(
        {"messages": [{"role": "user", "content": "我要去12315投诉你们"}]},  # pyright: ignore[reportArgumentType]
        config={"configurable": {"thread_id": "t1"}, "recursion_limit": 10},
    )
    print("route:", out.get("route"))
    print("escalated:", out.get("escalated"))
    print("msgs:", len(out["messages"]))
    print("answer:", out["messages"][-1].content)
