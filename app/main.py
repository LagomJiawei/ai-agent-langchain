"""
FastAPI 主应用
理财顾问 AI 服务接口
"""
import json
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from loguru import logger

from config import settings
from .security import Principal, resolve_principal
from .services import get_financial_service
from .sse import with_keepalive


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
    chat_id: str = "default"


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
    logger.info(f"配置: Agent 最大迭代 = {settings.agent.max_iterations}")
    logger.info(
        f"配置: 已注册 API Key 数 = {len(settings.security.api_keys)}, "
        f"允许匿名 = {settings.security.allow_anonymous}"
    )

    yield

    # 关闭时清理
    logger.info("服务关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="LiCaiManus API",
    description="基于 LangChain + 自研 Harness 主循环的智能化理财顾问服务",
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
    """健康检查接口（不需要 X-API-Key）。"""
    return {"status": "healthy", "service": "LiCaiManus"}


# 普通对话接口
@app.post("/api/chat", tags=["Chat"], response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    principal: Principal = Depends(resolve_principal),
):
    """
    普通对话接口（带对话记忆）

    - **message**: 用户消息
    - **chat_id**: 对话 ID（在服务端会被命名空间化为 ``<principal>:<chat_id>``）
    - **use_memory**: 是否使用对话记忆
    - **X-API-Key**: 可选 header，未带且允许匿名时映射为 anonymous
    """
    try:
        service = get_financial_service()
        effective_chat_id = principal.namespace_chat_id(request.chat_id)
        answer = service.chat(
            message=request.message,
            chat_id=effective_chat_id,
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
async def chat_with_rag(
    request: RagChatRequest,
    principal: Principal = Depends(resolve_principal),
):
    """
    RAG 知识库增强对话接口

    - **message**: 用户消息
    - **chat_id**: 对话 ID（命名空间化后传给 RAG 日志）
    - **X-API-Key**: 可选 header，未带且允许匿名时映射为 anonymous
    """
    try:
        service = get_financial_service()
        effective_chat_id = principal.namespace_chat_id(request.chat_id)
        answer = service.chat_with_rag(
            message=request.message,
            chat_id=effective_chat_id,
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
async def chat_with_agent(
    request: AgentChatRequest,
    principal: Principal = Depends(resolve_principal),
):
    """
    Agent 智能任务处理接口（Harness 主循环）

    - **message**: 用户消息或任务描述
    - **chat_id**: 会话 ID（命名空间化后用于 trace 分桶）
    - **X-API-Key**: 可选 header，未带且允许匿名时映射为 anonymous
    """
    try:
        service = get_financial_service()
        effective_chat_id = principal.namespace_chat_id(request.chat_id)
        result = service.chat_with_agent(
            message=request.message, chat_id=effective_chat_id
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
async def stream_chat(
    message: str,
    chat_id: str = "default",
    principal: Principal = Depends(resolve_principal),
):
    """
    流式对话接口（SSE，真 token 流）

    - **message**: 用户消息
    - **chat_id**: 对话 ID（命名空间化后传给下游）
    - **X-API-Key**: 可选 header，未带且允许匿名时映射为 anonymous
    """

    effective_chat_id = principal.namespace_chat_id(chat_id)

    async def _inner() -> AsyncGenerator[str, None]:
        try:
            service = get_financial_service()
            async for token in service.astream_chat(message, effective_chat_id):
                # SSE data 字段必须不含裸换行；用 JSON 包一层保险
                yield f"data: {json.dumps({'delta': token}, ensure_ascii=False)}\n\n"
            yield "event: end\ndata: {}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"event: error\ndata: {json.dumps({'message': str(e)}, ensure_ascii=False)}\n\n"

    async def generate_response() -> AsyncGenerator[str, None]:
        async for chunk in with_keepalive(
            _inner(), interval=settings.app.sse_keepalive_interval
        ):
            yield chunk

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# Agent 流式接口
@app.get("/api/chat/agent/stream", tags=["Chat"])
async def stream_chat_with_agent(
    message: str,
    chat_id: str = "default",
    principal: Principal = Depends(resolve_principal),
):
    """
    Agent 流式任务处理接口（SSE 事件流）

    - **message**: 用户消息或任务描述
    - **chat_id**: 会话 ID（命名空间化后用于 trace 分桶）
    - **X-API-Key**: 可选 header，未带且允许匿名时映射为 anonymous

    输出事件类型：run_start / thinking_token / tool_call / tool_result /
    final_text / run_end / error
    """

    effective_chat_id = principal.namespace_chat_id(chat_id)

    async def _inner() -> AsyncGenerator[str, None]:
        try:
            service = get_financial_service()
            async for event in service.astream_agent(message, chat_id=effective_chat_id):
                yield (
                    f"event: {event.type}\n"
                    f"data: {json.dumps(event.data, ensure_ascii=False)}\n\n"
                )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Agent 流式接口异常: {e}")
            yield (
                "event: error\n"
                f"data: {json.dumps({'message': str(e)}, ensure_ascii=False)}\n\n"
            )

    async def generate_events() -> AsyncGenerator[str, None]:
        async for chunk in with_keepalive(
            _inner(), interval=settings.app.sse_keepalive_interval
        ):
            yield chunk

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# RAG 流式接口
@app.get("/api/chat/rag/stream", tags=["Chat"])
async def stream_chat_with_rag(
    message: str,
    chat_id: str = "default",
    principal: Principal = Depends(resolve_principal),
):
    """
    RAG 流式问答接口（SSE 事件流，不走缓存）

    - **message**: 用户问题
    - **chat_id**: 会话 ID（命名空间化后仅用于日志）
    - **X-API-Key**: 可选 header，未带且允许匿名时映射为 anonymous

    输出事件类型：retrieval_started / retrieval_done / generation_token /
    done / error
    """

    effective_chat_id = principal.namespace_chat_id(chat_id)

    async def _inner() -> AsyncGenerator[str, None]:
        try:
            service = get_financial_service()
            async for event in service.astream_rag(message, chat_id=effective_chat_id):
                yield (
                    f"event: {event.type}\n"
                    f"data: {json.dumps(event.data, ensure_ascii=False)}\n\n"
                )
        except Exception as e:  # noqa: BLE001
            logger.error(f"RAG 流式接口异常: {e}")
            yield (
                "event: error\n"
                f"data: {json.dumps({'message': str(e)}, ensure_ascii=False)}\n\n"
            )

    async def generate_events() -> AsyncGenerator[str, None]:
        async for chunk in with_keepalive(
            _inner(), interval=settings.app.sse_keepalive_interval
        ):
            yield chunk

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# 缓存统计接口
@app.get("/api/cache/stats", tags=["Cache"], response_model=CacheStatsResponse)
async def get_cache_stats(
    principal: Principal = Depends(resolve_principal),
):
    """获取缓存统计信息

    - **X-API-Key**: 可选 header，未带且允许匿名时映射为 anonymous
    """
    try:
        service = get_financial_service()
        stats = service.get_cache_stats()
        return CacheStatsResponse(**stats)
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 清空对话记忆接口
@app.delete("/api/memory/{chat_id}", tags=["Memory"])
async def clear_chat_memory(
    chat_id: str,
    principal: Principal = Depends(resolve_principal),
):
    """
    清空指定对话的记忆

    - **chat_id**: 对话 ID（命名空间化为 ``<principal>:<chat_id>``，
      principal A 永远删不到 principal B 的会话）
    - **X-API-Key**: 可选 header，未带且允许匿名时映射为 anonymous
    """
    try:
        service = get_financial_service()
        effective_chat_id = principal.namespace_chat_id(chat_id)
        service.clear_chat_memory(effective_chat_id)
        return {"success": True, "message": f"对话 {chat_id} 的记忆已清空"}
    except Exception as e:
        logger.error(f"清空记忆失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 根路径重定向到文档
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "LiCaiManus API", "docs": "/docs"}
