"""
项目配置文件
使用 Pydantic Settings 管理配置
"""
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class LLMSettings(BaseSettings):
    """LLM 模型配置"""
    api_key: str = Field(default="", alias="OPENAI_API_KEY")
    base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", alias="OPENAI_BASE_URL")
    model_name: str = Field(default="qwen-turbo", alias="OPENAI_MODEL_NAME")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")
    timeout: int = Field(default=60, alias="TIMEOUT")
    embedding_model_name: str = Field(default="text-embedding-v3", alias="EMBEDDING_MODEL_NAME")


class VectorStoreSettings(BaseSettings):
    """向量数据库配置"""
    store_type: str = Field(default="inmemory", alias="VECTOR_STORE_TYPE")  # inmemory / milvus
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_collection_name: str = Field(default="rag_documents", alias="MILVUS_COLLECTION_NAME")
    milvus_dimension: int = Field(default=1536, alias="MILVUS_DIMENSION")
    faiss_persist_dir: str = Field(default="./vector-index/faiss", alias="FAISS_PERSIST_DIR")


class RedisSettings(BaseSettings):
    """Redis 配置"""
    enabled: bool = Field(default=False, alias="REDIS_ENABLED")
    host: str = Field(default="localhost", alias="REDIS_HOST")
    port: int = Field(default=6379, alias="REDIS_PORT")
    db: int = Field(default=0, alias="REDIS_DB")
    password: Optional[str] = Field(default=None, alias="REDIS_PASSWORD")


class ChatMemorySettings(BaseSettings):
    """聊天记忆配置"""
    store_type: str = Field(default="file", alias="CHAT_MEMORY_STORE_TYPE")  # memory / file / redis
    max_messages: int = Field(default=20, alias="CHAT_MEMORY_MAX_MESSAGES")
    base_dir: str = Field(default="./chat-memory", alias="CHAT_MEMORY_BASE_DIR")


class RagSettings(BaseSettings):
    """RAG 配置"""
    top_k: int = Field(default=5, alias="RAG_TOP_K")
    recall_k: int = Field(default=50, alias="RAG_RECALL_K")
    quality_threshold: float = Field(default=0.35, alias="RAG_QUALITY_THRESHOLD")


class AgentSettings(BaseSettings):
    """Agent 配置"""
    max_iterations: int = Field(default=10, alias="AGENT_MAX_ITERATIONS")
    mode: str = Field(default="auto", alias="AGENT_MODE")  # auto / react / plan_execute


class ToolRateLimitSettings(BaseSettings):
    """工具限流配置"""
    enabled: bool = Field(default=True, alias="TOOL_RATE_LIMIT_ENABLED")
    qps: int = Field(default=5, alias="TOOL_RATE_LIMIT_QPS")
    max_concurrent: int = Field(default=10, alias="TOOL_RATE_LIMIT_MAX_CONCURRENT")
    circuit_breaker_threshold: int = Field(default=5, alias="TOOL_RATE_LIMIT_CIRCUIT_BREAKER_THRESHOLD")


class AppSettings(BaseSettings):
    """应用配置"""
    host: str = Field(default="0.0.0.0", alias="APP_HOST")
    port: int = Field(default=8000, alias="APP_PORT")
    debug: bool = Field(default=True, alias="APP_DEBUG")


class Settings(BaseSettings):
    """全局配置"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    chat_memory: ChatMemorySettings = Field(default_factory=ChatMemorySettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    tool_rate_limit: ToolRateLimitSettings = Field(default_factory=ToolRateLimitSettings)
    app: AppSettings = Field(default_factory=AppSettings)
    search_api_key: str = Field(default="", alias="SEARCH_API_KEY")


# 全局配置实例
settings = Settings()
