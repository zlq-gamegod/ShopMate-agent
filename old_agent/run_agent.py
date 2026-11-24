from langchain.agents import create_agent  # pyright: ignore[reportMissingImports]
from langchain_core import messages
from langchain_core.tools import tool  # pyright: ignore[reportMissingImports]
from model import model
from RAG.rag_tools import search_knowledge
from agent.tools import get_current_time, query_orders, query_logistics



agent = create_agent(
    model = model,
    tools =[get_current_time , query_orders, query_logistics, search_knowledge], 
    system_prompt ="你是一个闲鱼的售后客服，用非常简洁的语言回答问题，当用于问涉及售后的内容时请先获取用用户订单号通过订单号再调用相关工具来解答用户的提问，规则类问题直接查知识库；只有需要判断订单事实时才向用户要订单号"
)


response = agent.invoke({  
    "messages" : [{"role": "user", "content": "12315"}]
})

print(response["messages"][-1].content)

