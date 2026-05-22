"""
网页搜索工具
使用 SearchAPI.io 调用百度搜索引擎获取最新信息
"""
import json
import requests
from loguru import logger
from langchain_core.tools import tool
from .rate_limiter import rate_limited

# SearchAPI 的搜索接口地址
SEARCH_API_URL = "https://www.searchapi.io/api/v1/search"


@tool
@rate_limited("web_search")
def search_web(query: str, api_key: str = "") -> str:
    """
    搜索网络获取最新信息（使用百度搜索引擎）

    Args:
        query: 搜索关键词
        api_key: SearchAPI 的 API Key（可选，优先使用配置中的 API Key）

    Returns:
        搜索结果 JSON 格式字符串
    """
    logger.info(f"执行网页搜索: {query}")

    # 优先使用传入的 api_key，如果没有则尝试从配置获取
    if not api_key:
        try:
            from config.settings import settings
            # 假设配置中有 search_api_key 字段，如果没有可以在 .env 中添加
            api_key = getattr(settings, "search_api_key", "")
        except Exception:
            pass

    if not api_key:
        return "搜索失败: 未配置 SearchAPI Key，请在 .env 中配置 SEARCH_API_KEY"

    try:
        params = {
            "q": query,
            "api_key": api_key,
            "engine": "baidu"
        }

        response = requests.get(SEARCH_API_URL, params=params, timeout=10)
        response.raise_for_status()

        json_data = response.json()

        # 提取 organic_results 部分，取前 5 条
        if "organic_results" not in json_data:
            return "未找到相关搜索结果。"

        organic_results = json_data["organic_results"]
        top_results = organic_results[:5]

        # 返回 JSON 格式字符串
        return json.dumps(top_results, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"搜索失败: {e}")
        return f"搜索失败: {str(e)}"
