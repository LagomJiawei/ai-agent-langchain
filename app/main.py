"""
FastAPI 主应用
理财顾问 AI 服务接口
"""
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loguru import logger

from config import settings
from .services import get_financial_service


# 请求/响应模型
class ChatRequest(BaseModel):
    """对话请求"""
    message: str
    chat_id: str = "default"
    use_memory: bool = True


class RagChatRequest(BaseModel):
    """RAG 对话请求"""
    message: str
    chat_id: str = "default"


class AgentChatRequest(BaseModel):
    """Agent 对话请求"""
    message: str
    force_plan_execute: bool = False


class ChatResponse(BaseModel):
    """对话响应"""
    success: bool
    answer: str
    chat_id: Optional[str] = None


class AgentResponse(BaseModel):
    """Agent 响应"""
    success: bool
    answer: str
    steps: int
    tool_calls: int


class CacheStatsResponse(BaseModel):
    """缓存统计响应"""
    enabled: bool
    count: Optional[int] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    # 启动时初始化
    logger.info("LiCaiManus 理财顾问服务启动中...")
    logger.info(f"配置: Agent 模式 = {settings.agent.mode}")

    yield

    # 关闭时清理
    logger.info("服务关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="LiCaiManus API",
    description="基于 LangChain + LangGraph 的智能化理财顾问服务",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
)

app.mount("/swagger-assets", StaticFiles(directory="static/swagger"), name="swagger-assets")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/swagger-assets/swagger-ui-bundle.js",
        swagger_css_url="/swagger-assets/swagger-ui.css",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


# 健康检查
@app.get("/api/health", tags=["System"])
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "service": "LiCaiManus"}


# 普通对话接口
@app.post("/api/chat", tags=["Chat"], response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    普通对话接口（带对话记忆）

    - **message**: 用户消息
    - **chat_id**: 对话 ID，用于区分不同对话
    - **use_memory**: 是否使用对话记忆
    """
    try:
        service = get_financial_service()
        answer = service.chat(
            message=request.message,
            chat_id=request.chat_id,
            use_memory=request.use_memory,
        )

        return ChatResponse(
            success=True,
            answer=answer,
            chat_id=request.chat_id,
        )
    except Exception as e:
        logger.error(f"对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# RAG 增强对话接口
@app.post("/api/chat/rag", tags=["Chat"], response_model=ChatResponse)
async def chat_with_rag(request: RagChatRequest):
    """
    RAG 知识库增强对话接口

    - **message**: 用户消息
    - **chat_id**: 对话 ID
    """
    try:
        service = get_financial_service()
        answer = service.chat_with_rag(
            message=request.message,
            chat_id=request.chat_id,
        )

        return ChatResponse(
            success=True,
            answer=answer,
            chat_id=request.chat_id,
        )
    except Exception as e:
        logger.error(f"RAG 对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Agent 任务处理接口
@app.post("/api/chat/agent", tags=["Chat"], response_model=AgentResponse)
async def chat_with_agent(request: AgentChatRequest):
    """
    Agent 智能任务处理接口

    - **message**: 用户消息或任务描述
    - **force_plan_execute**: 强制使用 Plan-and-Execute 模式
    """
    try:
        service = get_financial_service()
        result = service.chat_with_agent(
            message=request.message,
            force_plan_execute=request.force_plan_execute,
        )

        return AgentResponse(
            success=result["success"],
            answer=result["answer"],
            steps=result["steps"],
            tool_calls=result["tool_calls"],
        )
    except Exception as e:
        logger.error(f"Agent 任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 流式对话接口
@app.get("/api/chat/stream", tags=["Chat"])
async def stream_chat(message: str, chat_id: str = "default"):
    """
    流式对话接口（SSE）

    - **message**: 用户消息
    - **chat_id**: 对话 ID
    """

    async def generate_response() -> AsyncGenerator[str, None]:
        """流式生成响应"""
        try:
            service = get_financial_service()

            # 先执行 RAG 检索
            from rag import get_rag_pipeline
            pipeline = get_rag_pipeline()
            context = pipeline.retrieve_context_only(message)

            yield f"event: context\ndata: {context[:100]}...\n\n"

            # 生成答案（模拟流式）
            answer = service.chat(message, chat_id, use_memory=True)

            # 逐字输出
            for char in answer:
                yield f"data: {char}\n\n"

            yield "event: end\ndata: \n\n"

        except Exception as e:
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# 缓存统计接口
@app.get("/api/cache/stats", tags=["Cache"], response_model=CacheStatsResponse)
async def get_cache_stats():
    """获取缓存统计信息"""
    try:
        service = get_financial_service()
        stats = service.get_cache_stats()
        return CacheStatsResponse(**stats)
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 清空对话记忆接口
@app.delete("/api/memory/{chat_id}", tags=["Memory"])
async def clear_chat_memory(chat_id: str):
    """
    清空指定对话的记忆

    - **chat_id**: 对话 ID
    """
    try:
        service = get_financial_service()
        service.clear_chat_memory(chat_id)
        return {"success": True, "message": f"对话 {chat_id} 的记忆已清空"}
    except Exception as e:
        logger.error(f"清空记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 根路径重定向到文档
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "LiCaiManus API", "docs": "/docs"}

