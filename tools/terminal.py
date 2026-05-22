"""
终端命令执行工具
注意：仅允许安全的白名单命令
"""
import subprocess
import shlex
from loguru import logger
from langchain_core.tools import tool
from .rate_limiter import rate_limited


class TerminalSecurity:
    """终端安全白名单"""

    # 允许执行的命令白名单
    ALLOWED_COMMANDS = {
        # 文件操作
        "ls",
        "dir",
        "pwd",
        "cat",
        "echo",
        "head",
        "tail",
        "wc",
        "find",
        "grep",
        # 系统信息
        "date",
        "time",
        "whoami",
        "uname",
        "hostname",
        "uptime",
        "free",
        "df",
        "du",
        # 网络
        "ping",
        "curl",
        "wget",
        "ip",
        "ifconfig",
        # Python 相关（安全子集）
        "python",
        "python3",
        "pip",
        "pip3",
    }

    # 禁止的危险参数
    FORBIDDEN_FLAGS = {
        "--delete",
        "--remove",
        "-rf",
        "-rm",
        ">",
        ">>",
        "|",
        ";",
        "&",
        "&&",
        "||",
        "$(",
        "`",
    }

    @classmethod
    def is_safe_command(cls, command: str) -> tuple[bool, str]:
        """
        检查命令是否安全

        Returns:
            (是否安全, 原因说明)
        """
        # 检查危险字符
        for forbidden in cls.FORBIDDEN_FLAGS:
            if forbidden in command:
                return False, f"包含危险字符: {forbidden}"

        # 解析命令
        try:
            parts = shlex.split(command)
        except Exception as e:
            return False, f"命令解析失败: {e}"

        if not parts:
            return False, "空命令"

        cmd = parts[0].lower()

        # 检查是否在白名单中
        if cmd not in cls.ALLOWED_COMMANDS:
            return (
                False,
                f"命令 '{cmd}' 不在白名单中。允许的命令: {', '.join(sorted(cls.ALLOWED_COMMANDS))}",
            )

        return True, "安全"


@tool
@rate_limited("terminal_exec")
def terminal_exec(command: str) -> str:
    """
    执行终端命令（仅允许白名单内的安全命令）

    Args:
        command: 要执行的终端命令

    Returns:
        命令执行结果
    """
    logger.info(f"执行终端命令: {command}")

    # 安全检查
    is_safe, reason = TerminalSecurity.is_safe_command(command)
    if not is_safe:
        return f"安全检查失败: {reason}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = []
        if result.stdout:
            output.append(f"输出:\n{result.stdout}")
        if result.stderr:
            output.append(f"错误:\n{result.stderr}")
        output.append(f"退出码: {result.returncode}")

        return "\n".join(output)

    except subprocess.TimeoutExpired:
        return "命令执行超时（30秒限制）"
    except Exception as e:
        logger.error(f"命令执行失败: {e}")
        return f"命令执行失败: {str(e)}"
