"""工具系统：所有工具在导入时注册进 harness 默认 registry。"""
from harness.registry import register_tool
from harness.subagent import dispatch_subagent, dispatch_subagents

from .downloader import download_file
from .file_ops import file_read, file_write, list_files
from .pdf_generator import generate_pdf
from .rag_tool import search_knowledge_base
from .rate_limiter import RateLimiter, get_rate_limiter
from .terminal import terminal_exec
from .terminate import do_terminate
from .web_scraper import scrape_web_page
from .web_search import search_web

# 默认 scope 分组：web / fs / system / kb / control
_REGISTRATIONS = [
    (search_web, "web"),
    (scrape_web_page, "web"),
    (download_file, "web"),
    (file_read, "fs"),
    (file_write, "fs"),
    (list_files, "fs"),
    (terminal_exec, "system"),
    (generate_pdf, "fs"),
    (search_knowledge_base, "kb"),
    (do_terminate, "control"),
    (dispatch_subagent, "control"),
    (dispatch_subagents, "control"),
]

for _tool, _scope in _REGISTRATIONS:
    register_tool(_tool, scope=_scope)

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
    "dispatch_subagent",
    "dispatch_subagents",
    "RateLimiter",
    "get_rate_limiter",
]
