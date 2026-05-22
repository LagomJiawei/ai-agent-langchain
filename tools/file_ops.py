"""
文件操作工具
支持读取和写入文件
"""
import os
from pathlib import Path
from loguru import logger
from langchain_core.tools import tool
from .rate_limiter import rate_limited


class FileSecurity:
    """文件安全检查"""

    # 允许的目录（相对路径）
    ALLOWED_DIRS = {
        "./data",
        "./documents",
        "./output",
        "./downloads",
    }

    @classmethod
    def is_safe_path(cls, path: str) -> bool:
        """检查路径是否在允许范围内"""
        try:
            base_dir = Path.cwd()
            target_path = (base_dir / path).resolve()

            # 检查是否在当前目录的子目录下
            if not str(target_path).startswith(str(base_dir)):
                return False

            # 检查是否在允许的目录中
            for allowed_dir in cls.ALLOWED_DIRS:
                allowed_path = (base_dir / allowed_dir).resolve()
                if str(target_path).startswith(str(allowed_path)):
                    return True

            return False
        except Exception:
            return False

    @classmethod
    def ensure_dir(cls, path: str) -> bool:
        """确保目录存在"""
        try:
            dir_path = Path(path).parent
            dir_path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False


@tool
@rate_limited("file_read")
def file_read(path: str) -> str:
    """
    读取文件内容

    Args:
        path: 文件路径（相对路径）

    Returns:
        文件内容
    """
    logger.info(f"读取文件: {path}")

    if not FileSecurity.is_safe_path(path):
        return f"错误: 不允许访问路径 {path}，请使用允许的目录: {FileSecurity.ALLOWED_DIRS}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return f"文件 {path} 内容:\n{content}"
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return f"读取文件失败: {str(e)}"


@tool
@rate_limited("file_write")
def file_write(path: str, content: str) -> str:
    """
    将内容写入文件

    Args:
        path: 文件路径（相对路径）
        content: 要写入的内容

    Returns:
        操作结果
    """
    logger.info(f"写入文件: {path}")

    if not FileSecurity.is_safe_path(path):
        return f"错误: 不允许访问路径 {path}，请使用允许的目录: {FileSecurity.ALLOWED_DIRS}"

    try:
        FileSecurity.ensure_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"成功写入文件: {path}，共 {len(content)} 字符"
    except Exception as e:
        logger.error(f"写入文件失败: {e}")
        return f"写入文件失败: {str(e)}"


@tool
@rate_limited("file_list")
def list_files(directory: str = "./") -> str:
    """
    列出目录中的文件

    Args:
        directory: 目录路径，默认当前目录

    Returns:
        文件列表
    """
    logger.info(f"列出目录: {directory}")

    if not FileSecurity.is_safe_path(directory + "/dummy"):
        return f"错误: 不允许访问目录 {directory}"

    try:
        path = Path(directory)
        if not path.exists():
            return f"目录不存在: {directory}"

        files = []
        for item in path.iterdir():
            item_type = "📁" if item.is_dir() else "📄"
            size = f" ({item.stat().st_size} bytes)" if item.is_file() else ""
            files.append(f"{item_type} {item.name}{size}")

        return f"目录 {directory} 内容:\n" + "\n".join(files)
    except Exception as e:
        logger.error(f"列出目录失败: {e}")
        return f"列出目录失败: {str(e)}"
