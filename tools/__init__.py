"""
工具系统
包含 7 种工具 + 限流保护机制
"""
from typing import List

from langchain_core.tools import BaseTool

from .web_search import search_web
from .web_scraper import scrape_web_page
from .file_ops import file_read, file_write, list_files
from .terminal import terminal_exec
from .downloader import download_file
from .pdf_generator import generate_pdf
from .terminate import do_terminate
from .rag_tool import search_knowledge_base
from .rate_limiter import rate_limited, RateLimiter, get_rate_limiter


def get_all_tools() -> List[BaseTool]:
    """获取所有可用工具"""
    return [
        search_web,
        scrape_web_page,
        file_read,
        file_write,
        list_files,
        terminal_exec,
        download_file,
        generate_pdf,
        search_knowledge_base,
        do_terminate,
    ]


__all__ = [
    "search_web",
    "scrape_web_page",
    "file_read",
    "file_write",
    "list_files",
    "terminal_exec",
    "download_file",
    "generate_pdf",
    "search_knowledge_base",
    "do_terminate",
    "get_all_tools",
    "rate_limited",
    "RateLimiter",
    "get_rate_limiter",
]
