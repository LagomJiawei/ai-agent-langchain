"""
LLM 模型配置与工厂
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from config.settings import settings


def create_chat_model(
    model_name: str = None,
    temperature: float = None,
    timeout: int = None,
) -> BaseChatModel:
    """创建聊天模型"""
    return ChatOpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        model=model_name or settings.llm.model_name,
        temperature=temperature if temperature is not None else settings.llm.temperature,
        timeout=timeout or settings.llm.timeout,
    )


def create_embeddings_model(
    model_name: str = None,
) -> OpenAIEmbeddings:
    """创建 Embedding 模型"""
    return OpenAIEmbeddings(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        model=model_name or settings.llm.embedding_model_name,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
    )
