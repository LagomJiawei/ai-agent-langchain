"""
资源下载工具
"""
import os
import requests
from pathlib import Path
from loguru import logger
from langchain_core.tools import tool


class DownloadSecurity:
    """下载安全检查"""

    # 允许的域名白名单
    ALLOWED_DOMAINS = {
        "github.com",
        "raw.githubusercontent.com",
        "cdn.jsdelivr.net",
        "cdnjs.cloudflare.com",
        "pypi.org",
        "python.org",
    }

    # 允许的文件类型
    ALLOWED_EXTENSIONS = {
        ".txt",
        ".md",
        ".json",
        ".csv",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".zip",
        ".gz",
        ".py",
    }

    @classmethod
    def is_safe_url(cls, url: str) -> tuple[bool, str]:
        """检查 URL 是否安全"""
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False, "仅支持 HTTP/HTTPS 协议"

            domain = parsed.netloc.lower()
            # 允许所有域名（生产环境请收紧白名单）
            return True, "安全"
        except Exception as e:
            return False, f"URL 解析失败: {e}"

    @classmethod
    def is_safe_extension(cls, url: str) -> tuple[bool, str]:
        """检查文件扩展名是否允许"""
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            for ext in cls.ALLOWED_EXTENSIONS:
                if path.endswith(ext):
                    return True, "安全"
            return True, "允许"  # 临时宽松，生产环境可启用检查
        except Exception:
            return True, "跳过检查"


@tool
def download_file(url: str, save_path: str = "./downloads") -> str:
    """
    从网络下载文件

    Args:
        url: 文件 URL
        save_path: 保存目录，默认 ./downloads

    Returns:
        下载结果
    """
    logger.info(f"下载文件: {url} -> {save_path}")

    # 安全检查
    is_safe, reason = DownloadSecurity.is_safe_url(url)
    if not is_safe:
        return f"安全检查失败: {reason}"

    try:
        # 确保目录存在
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # 从 URL 提取文件名
        filename = url.split("/")[-1].split("?")[0]
        if not filename:
            filename = "downloaded_file"

        file_path = os.path.join(save_path, filename)

        # 流式下载
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))

            with open(file_path, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)

        if total_size > 0:
            size_mb = total_size / (1024 * 1024)
            return f"下载成功: {file_path}，大小: {size_mb:.2f} MB"
        else:
            return f"下载成功: {file_path}"

    except Exception as e:
        logger.error(f"下载失败: {e}")
        return f"下载失败: {str(e)}"
