"""
向量存储配置与工厂
支持内存存储 (FAISS) 和 Milvus 分布式存储
"""
import re
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS, Milvus
from langchain_openai import OpenAIEmbeddings
from config.settings import settings


_faiss_vector_store: Optional[FAISS] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _project_root() / resolved
    return resolved


def _split_text(text: str, max_chars: int = 1200) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars))
        elif current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def _fallback_markdown_documents(path: Path, content: str, title: str) -> list[Document]:
    sections = re.split(r"(?=^#{1,6}\s+)", content, flags=re.MULTILINE)
    documents = []
    chunk_index = 1
    for section in sections:
        section = section.strip()
        if not section:
            continue
        heading_match = re.match(r"^#{1,6}\s+(.+)$", section, re.MULTILINE)
        section_title = heading_match.group(1).strip() if heading_match else title
        for chunk in _split_text(section):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "title": section_title,
                        "source": path.name,
                        "chunk_id": f"chunk-{chunk_index}",
                    },
                )
            )
            chunk_index += 1
    return documents


def load_internal_documents(document_dir: Optional[Path] = None) -> list[Document]:
    document_dir = document_dir or _project_root() / "document"
    if not document_dir.exists():
        return []

    documents = []
    for path in sorted(document_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        title_match = re.search(r"^####\s*(.+)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem
        matches = list(
            re.finditer(
                r"Q(\d+)：\s*(.*?)(?:\n\s*)+A：\s*(.*?)(?=\n\s*Q\d+：|\Z)",
                content,
                re.DOTALL,
            )
        )
        if not matches:
            documents.extend(_fallback_markdown_documents(path, content, title))
            continue

        for match in matches:
            qa_id = f"Q{match.group(1)}"
            question = match.group(2).strip()
            answer = match.group(3).strip()
            for chunk_index, answer_chunk in enumerate(_split_text(answer), 1):
                metadata = {
                    "title": title,
                    "source": path.name,
                    "qa_id": qa_id,
                }
                if chunk_index > 1:
                    metadata["chunk_id"] = f"{qa_id}-{chunk_index}"
                documents.append(
                    Document(
                        page_content=f"问题：{question}\n回答：{answer_chunk}",
                        metadata=metadata,
                    )
                )

    return documents


def create_embeddings() -> Embeddings:
    """创建 Embedding 模型"""
    return OpenAIEmbeddings(
        openai_api_key=settings.llm.api_key,
        openai_api_base=settings.llm.base_url,
        model=settings.llm.embedding_model_name,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
        chunk_size=8,
    )


def create_faiss_vector_store(
    embeddings: Optional[Embeddings] = None,
    texts: Optional[list] = None,
    persist_dir: Optional[str] = None,
) -> FAISS:
    """创建 FAISS 内存向量存储"""
    global _faiss_vector_store

    if embeddings is None:
        embeddings = create_embeddings()

    if texts is not None and len(texts) > 0:
        return FAISS.from_texts(texts, embeddings)

    if _faiss_vector_store is not None:
        return _faiss_vector_store

    index_dir = _resolve_path(persist_dir or settings.vector_store.faiss_persist_dir)
    index_file = index_dir / "index.faiss"
    metadata_file = index_dir / "index.pkl"

    if index_file.exists() and metadata_file.exists():
        _faiss_vector_store = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return _faiss_vector_store

    documents = load_internal_documents()
    if documents:
        _faiss_vector_store = FAISS.from_documents(documents, embeddings)
    else:
        _faiss_vector_store = FAISS.from_texts(["_init_"], embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    _faiss_vector_store.save_local(str(index_dir))
    return _faiss_vector_store


def create_milvus_vector_store(
    embeddings: Optional[Embeddings] = None,
    collection_name: Optional[str] = None,
) -> Milvus:
    """创建 Milvus 分布式向量存储"""
    if embeddings is None:
        embeddings = create_embeddings()

    if collection_name is None:
        collection_name = settings.vector_store.milvus_collection_name

    return Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={
            "host": settings.vector_store.milvus_host,
            "port": settings.vector_store.milvus_port,
        },
        drop_old=False,
        auto_id=True,
    )


def create_vector_store(embeddings: Optional[Embeddings] = None):
    """根据配置创建向量存储"""
    store_type = settings.vector_store.store_type

    if store_type == "inmemory":
        return create_faiss_vector_store(embeddings)
    elif store_type == "milvus":
        return create_milvus_vector_store(embeddings)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
