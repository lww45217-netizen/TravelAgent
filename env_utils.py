import os

from dotenv import load_dotenv

load_dotenv()


def require_env(name: str) -> str:
    """读取必需的环境变量，并在缺失时给出明确错误。"""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}，请在项目根目录的 .env 中配置")
    return value


# 当前项目使用百炼 OpenAI 兼容接口和高德地图 MCP；模型名由环境变量配置。
DASHSCOPE_API_KEY = require_env("DASHSCOPE_API_KEY")
DASHSCOPE_API_URL = require_env("DASHSCOPE_API_URL")
DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "deepseek-v4.1-flash")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")
