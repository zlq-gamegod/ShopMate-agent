from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from RAG.rag_tools import retriever
from model import model
from agent.state import AgentState
from agent.lexicon import classify
from agent.tools import transfer_to_human_impl, TOOLS
import re
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately

ORDER_RE = re.compile(r"ORD\d{12}")   


MAX_CTX_CHARS = 2200

SYSTEM_TMPL = """你是闲鱼售后客服。语气简洁，别靠猜!!!只回答对方问的那一件事。


【当前会话】
订单号：{order_id}

【本轮规则依据】
{context}

铁律：
1. 条款、天数、票数、金额只能来自上面的依据或工具返回值，不得编造。
2. 消息里出现订单号时，先调 query_orders 拿到订单事实再作答；没拿到事实前不许下结论（"随时可退""肯定能退"这类断言禁止）。
3. 时效问题先报工具返回的签收天数（days_since_signed），再报超期天数，形如"签收 X 天，已超出售后期 Y 天"；X、Y 都只能用工具返回值，不许自己相减。
4. 问钱在谁手上（是否已放款给卖家、是否退回买家）必须查 query_guarantee，不能凭订单状态推断。
5. 用户还没申请退款、但问"能退多少"时，按用户描述的问题类型直接试算 calc_refund_amount；查不到退款单不是转人工的理由。
6. 只有规则类问题在依据为空时才回答"这个问题我需要为您转接人工"；订单、物流、金额类即使依据为空也要先查工具再决定。
7. 工具返回的状态词必须原样引用（如「已放款」「担保中」「待发货」「审核中」），不得改写成"货款已经打给卖家"这类口语说法。
8. 结论是"不能/不支持/超期"时，必须同时给出规则允许的替代路径（与卖家协商、申请客服介入、举报违规、走司法途径等），不能只回一句否定。
"""

def _last_human_text(state: AgentState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else ""
    return ""

def route_node(state: AgentState) -> dict:
    text = _last_human_text(state)
    upd = {"route": classify(text)}
    hit = ORDER_RE.findall(text.upper())   
    if hit:
        upd["order_id"] = hit[-1]          
    return upd

def retrieve_node(state: AgentState) -> dict:
    docs = retriever.invoke(_last_human_text(state))
    ctx = "\n\n".join(d.page_content for d in docs)
    return {"retrieved": ctx[:MAX_CTX_CHARS]}

def agent_node(state: AgentState) -> dict:
    system = SystemMessage(content=SYSTEM_TMPL.format(
        order_id=state.get("order_id") or "（未提供，涉及订单事实时先向用户索取）",
        context=state.get("retrieved") or "（无）",
    ))
    history = trim_messages(
        list(state["messages"]),
        max_tokens=1500,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
        include_system=False,
        allow_partial=False,
    )
    resp = model.bind_tools(TOOLS).invoke([system] + history)
    return {"messages": [resp]}

def handoff_node(state: AgentState) -> dict:
    info = transfer_to_human_impl(
        reason=state.get("route") or "unspecified",
        user_message=_last_human_text(state),
    )
    return {"messages": [AIMessage(content=info)], "escalated": True}
