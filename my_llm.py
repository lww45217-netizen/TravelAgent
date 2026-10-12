from langchain_openai import ChatOpenAI

from env_utils import DASHSCOPE_API_KEY, DASHSCOPE_API_URL, DASHSCOPE_MODEL


# 百炼提供 OpenAI 兼容接口，因此统一使用 ChatOpenAI，避免引入多套 SDK。
llm = ChatOpenAI(
    model=DASHSCOPE_MODEL,
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_API_URL,
)
