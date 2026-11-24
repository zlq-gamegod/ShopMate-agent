from typing import Annotated, NotRequired, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: NotRequired[str]
    retrieved: NotRequired[str]
    order_id: NotRequired[str]
    escalated: NotRequired[bool]
