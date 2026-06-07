#!/usr/bin/env python3
"""
LiCaiManus 启动脚本
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

if __name__ == "__main__":
    print("=" * 60)
    print("  LiCaiManus - LangChain + Harness 主循环版本")
    print("=" * 60)
    print()
    print("启动服务...")
    print(f"API 文档: http://localhost:{os.getenv('APP_PORT', '8000')}/docs")
    print(f"健康检查: http://localhost:{os.getenv('APP_PORT', '8000')}/api/health")
    print()
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=os.getenv("DEBUG", "true").lower() == "true",
    )
