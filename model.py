import os
from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]

load_dotenv()
model = ChatOpenAI(
    model='qwen-plus',
    temperature=float(os.getenv('MODEL_TEMPERATURE', '0.7')),
    timeout=60,
    max_retries=2,
    api_key=os.getenv('DASHSCOPE_API_KEY'),  # pyright: ignore[reportArgumentType]
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',

)

