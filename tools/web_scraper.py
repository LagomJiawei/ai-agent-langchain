"""
网页内容抓取工具
"""
import requests
from bs4 import BeautifulSoup
from loguru import logger
from langchain_core.tools import tool
from .rate_limiter import rate_limited


@tool
@rate_limited("web_scraper")
def scrape_web_page(url: str, max_length: int = 3000) -> str:
    """
    抓取指定 URL 的网页内容

    Args:
        url: 网页地址
        max_length: 返回内容最大长度，默认3000字符

    Returns:
        网页文本内容
    """
    logger.info(f"抓取网页: {url}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # 移除脚本、样式、导航等不需要的元素
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # 获取文本内容
        text = soup.get_text(separator="\n", strip=True)

        # 清理多余空行
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        # 限制长度
        if len(text) > max_length:
            text = text[:max_length] + "..."

        if not text:
            return "无法获取网页内容。"

        return f"网页内容 ({url}):\n{text}"

    except Exception as e:
        logger.error(f"网页抓取失败: {e}")
        return f"网页抓取失败: {str(e)}"
