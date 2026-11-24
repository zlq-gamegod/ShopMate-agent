from json import load
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_core import vectorstores
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) 
from model import model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool

load_dotenv()
rule_docs = Docx2txtLoader("ShopMate-闲鱼售后规则详细版.docx").load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    separators=["\n\n\n","\n\n", "\n", "。", "！", "？", "，", " "],
    chunk_overlap = 50,
    add_start_index=True
)
chunks = text_splitter.split_documents(documents=rule_docs)


embedding = OpenAIEmbeddings(
    model = "text-embedding-v3",
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    check_embedding_ctx_length=False,
    chunk_size=10
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding,
    collection_name="shopmate",
)

if vectorstore._collection.count() == 0:
    vectorstore = Chroma.from_documents(      
        documents=chunks,
        embedding=embedding,
        persist_directory="./chroma_db",
        collection_name="shopmate",
    )

bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=jieba.lcut)
bm25_retriever.k = 5

dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.5, 0.5],
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "仅根据以下资料回答，没有依据就说不知道，并注明来源。\n\n{context}"),
    ("human", "{question}"),
])
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = ({"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
         | prompt | model | StrOutputParser())

@tool
def search_knowledge(question: str):
    """查询 ShopMate 平台规则知识库。当用户询问售后政策、退货退款规则、
    交易流程、特殊类目规范等问题时调用此工具；查询订单状态或物流请勿调用。"""
    return chain.invoke(question)

retriever = hybrid_retriever