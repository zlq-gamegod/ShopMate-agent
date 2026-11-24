from agent.graph import build_graph
from langchain_core.messages import AIMessage, HumanMessage

g = build_graph()


def run_once(question: str, thread_id: str = "default") -> dict:
    out = g.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
    )
    ai_msgs = [m for m in out["messages"] if isinstance(m, AIMessage)]
    return {
        "answer": out["messages"][-1].content,
        "route": out.get("route"),
        "escalated": out.get("escalated", False),
        "tool_calls": [tc["name"] for m in ai_msgs for tc in (m.tool_calls or [])],
        "turns": len(ai_msgs),
        "messages": out["messages"],
    }
if __name__ == "__main__":
    tid = "cli-1"                      
    while True:
        q = input("你> ").strip()
        if q in {"q", "quit", "exit"}:
            break
        r = run_once(q, thread_id=tid)
        print(f"[{r['route']}] {r['answer']}\n")
