"""
配置管理模块
"""
from .settings import settings
from .llm import create_chat_model, create_embeddings_model
from .vector_store import create_vector_store, create_embeddings, create_faiss_vector_store, create_milvus_vector_store, load_internal_documents

__all__ = [
    "settings",
    "create_chat_model",
    "create_embeddings_model",
    "create_vector_store",
    "create_embeddings",
    "create_faiss_vector_store",
    "create_milvus_vector_store",
    "load_internal_documents",
]
