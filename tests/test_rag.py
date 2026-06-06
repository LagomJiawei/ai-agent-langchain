"""
RAG 测试
"""
import json
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag import (
    QueryTranslationTransformer,
    QueryRewritingTransformer,
    DocumentReranker,
    RerankStrategy,
    is_chinese_query,
    transform_query_for_retrieval,
)
from rag.transformer import _strip_query_label


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text) % 10), float(i % 10), 1.0] for i, text in enumerate(texts)]

    def embed_query(self, text):
        return [float(len(text) % 10), 0.0, 1.0]


class FakeVectorStore:
    def __init__(self, results):
        self.results = results

    def similarity_search_with_score(self, query, k):
        return self.results[:k]


def test_translation_transformer_initialization():
    """测试查询翻译转换器初始化"""
    transformer = QueryTranslationTransformer()
    assert transformer is not None


def test_rewriting_transformer_initialization():
    """测试查询重写转换器初始化"""
    transformer = QueryRewritingTransformer()
    assert transformer is not None


def test_reranker_initialization():
    """测试重排序器初始化"""
    reranker = DocumentReranker()
    assert reranker is not None


def test_rerank_strategy_enum():
    """测试重排序策略枚举"""
    assert RerankStrategy.COSINE_SIMILARITY.value == "cosine"
    assert RerankStrategy.HYBRID_SCORE.value == "hybrid"
    assert RerankStrategy.RRF.value == "rrf"


def test_retriever_returns_candidates_without_similarity_filter():
    """召回阶段不使用固定阈值硬过滤候选文档"""
    from rag.retriever import DocumentRetriever

    vector_store = FakeVectorStore(
        [
            (Document(page_content="低分但候选仍应保留", metadata={"source": "a.md"}), 99.0),
            (Document(page_content="另一个候选", metadata={"source": "b.md"}), 49.0),
        ]
    )
    retriever = DocumentRetriever(vector_store=vector_store)

    docs = retriever.retrieve("测试查询")

    assert len(docs) == 2
    assert docs[0].metadata["score"] == docs[0].metadata["vector_score"]
    assert docs[0].metadata["vector_distance"] == 99.0
    assert docs[0].metadata["retrieval_rank"] == 1


def test_reranker_persists_scores_to_metadata():
    """重排序结果保留 hybrid 质量信号"""
    reranker = DocumentReranker()
    docs = [
        Document(page_content="基金定投适合长期投资", metadata={"score": 0.2, "vector_score": 0.2}),
        Document(page_content="保险用于风险保障", metadata={"score": 0.9, "vector_score": 0.9}),
    ]

    ranked = reranker.rerank(docs, "基金定投 长期投资", RerankStrategy.HYBRID_SCORE)

    assert "rerank_score" in ranked[0].metadata
    assert "bm25_score" in ranked[0].metadata
    assert "keyword_score" in ranked[0].metadata
    assert ranked[0].metadata["rerank_strategy"] == RerankStrategy.HYBRID_SCORE.value



def test_is_chinese_query():
    """测试公共中文查询检测"""
    assert is_chinese_query("什么是基金定投？") is True
    assert is_chinese_query("Hello World") is False
    assert is_chinese_query("基金 fund 混合") is True
    assert is_chinese_query("") is False
    assert is_chinese_query("   ") is False


def test_strip_query_label():
    """清理查询变换输出中的标签前缀"""
    assert _strip_query_label("优化后的查询: 理财新手 财务规划") == "理财新手 财务规划"
    assert _strip_query_label("优化后的检索查询：基金定投 定期定额投资") == "基金定投 定期定额投资"
    assert _strip_query_label("理财新手 财务规划") == "理财新手 财务规划"


def test_translation_transformer_strips_label():
    """翻译器输出不保留标签前缀"""
    transformer = QueryTranslationTransformer.__new__(QueryTranslationTransformer)
    transformer.chain = Mock()
    transformer.chain.invoke.return_value = Mock(content="优化后的查询: 理财新手 财务规划")

    assert transformer.transform("financial plan") == "理财新手 财务规划"


def test_rewriting_transformer_strips_label():
    """重写器输出不保留标签前缀"""
    transformer = QueryRewritingTransformer.__new__(QueryRewritingTransformer)
    transformer.chain = Mock()
    transformer.chain.invoke.return_value = Mock(content="优化后的检索查询：理财新手 财务规划")

    assert transformer.transform("理财新手") == "理财新手 财务规划"


def test_chinese_query_transform_skips_translation():
    """中文查询跳过翻译，直接重写"""
    with patch("rag.transformer.get_translation_transformer") as get_translator:
        with patch("rag.transformer.get_rewriting_transformer") as get_rewriter:
            translator = Mock()
            rewriter = Mock()
            rewriter.transform.return_value = "基金定投 定期定额投资"
            get_translator.return_value = translator
            get_rewriter.return_value = rewriter

            result = transform_query_for_retrieval("什么是基金定投？")

            translator.transform.assert_not_called()
            rewriter.transform.assert_called_once_with("什么是基金定投？")
            assert result.original_query == "什么是基金定投？"
            assert result.translated_query is None
            assert result.rewritten_query == "基金定投 定期定额投资"


def test_non_chinese_query_transform_translates_then_rewrites():
    """非中文查询先翻译，再基于翻译结果重写"""
    with patch("rag.transformer.get_translation_transformer") as get_translator:
        with patch("rag.transformer.get_rewriting_transformer") as get_rewriter:
            translator = Mock()
            rewriter = Mock()
            translator.transform.return_value = "什么是基金定投？"
            rewriter.transform.return_value = "基金定投 定期定额投资"
            get_translator.return_value = translator
            get_rewriter.return_value = rewriter

            result = transform_query_for_retrieval("What is dollar cost averaging?")

            translator.transform.assert_called_once_with("What is dollar cost averaging?")
            rewriter.transform.assert_called_once_with("什么是基金定投？")
            assert result.original_query == "What is dollar cost averaging?"
            assert result.translated_query == "什么是基金定投？"
            assert result.rewritten_query == "基金定投 定期定额投资"


def test_load_internal_documents_from_document_folder():
    """从 document 目录解析内部知识库问答文档"""
    from config.vector_store import load_internal_documents

    documents = load_internal_documents()

    assert len(documents) >= 15
    assert any("月光族" in doc.page_content for doc in documents)
    first_doc = documents[0]
    assert "title" in first_doc.metadata
    assert "source" in first_doc.metadata
    assert "qa_id" in first_doc.metadata
    assert first_doc.metadata["source"].endswith(".md")


def test_create_embeddings_limits_batch_size():
    """Embedding 批量大小不超过 DashScope 限制"""
    from config.vector_store import create_embeddings

    embeddings = create_embeddings()

    assert embeddings.chunk_size == 8


def test_load_internal_documents_falls_back_for_plain_markdown(tmp_path):
    """非 Q&A Markdown 也能作为内部知识库文档加载"""
    from config.vector_store import load_internal_documents

    doc_dir = tmp_path / "document"
    doc_dir.mkdir()
    (doc_dir / "plain.md").write_text(
        "# 资产配置\n\n长期投资应关注风险承受能力。\n\n## 保险\n\n保险主要用于风险转移。",
        encoding="utf-8",
    )

    documents = load_internal_documents(doc_dir)

    assert len(documents) >= 2
    assert any("风险承受能力" in doc.page_content for doc in documents)
    assert all("chunk_id" in doc.metadata for doc in documents)



def test_create_faiss_vector_store_loads_internal_documents(tmp_path):
    """默认 FAISS 向量库加载 document 内部文档而不是仅初始化 dummy 文本"""
    import config.vector_store as vector_store

    vector_store._faiss_vector_store = None
    store = vector_store.create_faiss_vector_store(
        embeddings=FakeEmbeddings(),
        persist_dir=str(tmp_path / "faiss"),
    )
    docs = list(store.docstore._dict.values())

    assert any("月光族" in doc.page_content for doc in docs)
    assert not all(doc.page_content == "_init_" for doc in docs)

    vector_store._faiss_vector_store = None



def test_create_faiss_vector_store_persists_and_reloads_internal_documents(tmp_path):
    """FAISS 本地索引保存后可再次加载内部知识库文档"""
    import config.vector_store as vector_store

    persist_dir = tmp_path / "faiss"
    vector_store._faiss_vector_store = None
    vector_store.create_faiss_vector_store(
        embeddings=FakeEmbeddings(),
        persist_dir=str(persist_dir),
    )

    assert (persist_dir / "index.faiss").exists()
    assert (persist_dir / "index.pkl").exists()

    vector_store._faiss_vector_store = None
    reloaded = vector_store.create_faiss_vector_store(
        embeddings=FakeEmbeddings(),
        persist_dir=str(persist_dir),
    )
    docs = list(reloaded.docstore._dict.values())

    assert any("月光族" in doc.page_content for doc in docs)

    vector_store._faiss_vector_store = None



def test_create_faiss_vector_store_with_texts_does_not_persist(tmp_path):
    """显式传入 texts 时保持临时构建，不写入持久化目录"""
    import config.vector_store as vector_store

    persist_dir = tmp_path / "faiss"
    store = vector_store.create_faiss_vector_store(
        embeddings=FakeEmbeddings(),
        texts=["temporary text"],
        persist_dir=str(persist_dir),
    )
    docs = list(store.docstore._dict.values())

    assert docs[0].page_content == "temporary text"
    assert not persist_dir.exists()



def test_search_knowledge_base_returns_quality_signal():
    """知识库检索工具返回 Agent 可判断的信息充足性信号"""
    from tools.rag_tool import search_knowledge_base

    retriever = Mock()
    reranker = Mock()
    doc = Document(
        page_content="基金定投适合长期投资，通过定期投入平滑市场波动。",
        metadata={
            "title": "基金定投",
            "source": "knowledge_base",
            "score": 0.8,
            "rerank_score": 0.8,
            "keyword_score": 0.6,
        },
    )
    retriever.retrieve.return_value = [doc]
    reranker.rerank.return_value = [doc]

    with patch("tools.rag_tool.transform_query_for_retrieval") as transform:
        with patch("tools.rag_tool.get_document_retriever", return_value=retriever):
            with patch("tools.rag_tool.get_document_reranker", return_value=reranker):
                transform.return_value = Mock(rewritten_query="基金定投 定期投资")

                payload = json.loads(search_knowledge_base.invoke({"query": "什么是基金定投？"}))

    assert payload["original_query"] == "什么是基金定投？"
    assert payload["rewritten_query"] == "基金定投 定期投资"
    assert payload["doc_count"] == 1
    assert payload["quality_score"] >= 0.5
    assert payload["sufficiency"] == "adequate"
    assert "基金定投适合长期投资" in payload["context"]
    retriever.retrieve.assert_called_once_with("基金定投 定期投资")
    reranker.rerank.assert_called_once()



def test_rag_pipeline_truncates_after_rerank():
    """RAG pipeline 在 rerank 后按 top_k 截断上下文"""
    from rag.pipeline import RagPipeline

    retriever = Mock()
    reranker = Mock()
    generation_chain = Mock()
    candidates = [Document(page_content=f"候选 {i}") for i in range(8)]
    reranked = [Document(page_content=f"排序后 {i}") for i in range(8)]
    retriever.retrieve.return_value = candidates
    reranker.rerank.return_value = reranked
    generation_chain.invoke.return_value = Mock(content="answer")

    pipeline = RagPipeline(retriever=retriever, reranker=reranker, cache=None)
    pipeline.generation_chain = generation_chain

    with patch("rag.pipeline.transform_query_for_retrieval") as transform:
        transform.return_value = Mock(rewritten_query="基金定投")
        assert pipeline.execute("什么是基金定投？") == "answer"

    context = generation_chain.invoke.call_args.args[0]["context"]
    assert "排序后 4" in context
    assert "排序后 5" not in context



def test_rag_pipeline_execute_chinese_uses_rewritten_query_without_translation():
    """串行 RAG 中文查询不翻译，检索使用重写结果"""
    from rag.pipeline import RagPipeline

    retriever = Mock()
    reranker = Mock()
    generation_chain = Mock()
    retriever.retrieve.return_value = []
    reranker.rerank.return_value = []
    generation_chain.invoke.return_value = Mock(content="answer")

    pipeline = RagPipeline(retriever=retriever, reranker=reranker, cache=None)
    pipeline.generation_chain = generation_chain

    with patch("rag.transformer.get_translation_transformer") as get_translator:
        with patch("rag.transformer.get_rewriting_transformer") as get_rewriter:
            translator = Mock()
            rewriter = Mock()
            rewriter.transform.return_value = "基金定投 定期定额投资"
            get_translator.return_value = translator
            get_rewriter.return_value = rewriter

            assert pipeline.execute("什么是基金定投？") == "answer"

            translator.transform.assert_not_called()
            rewriter.transform.assert_called_once_with("什么是基金定投？")
            retriever.retrieve.assert_called_once_with("基金定投 定期定额投资", False)
            reranker.rerank.assert_called_once_with([], "基金定投 定期定额投资", RerankStrategy.HYBRID_SCORE)


# ============================================================================
# P8: 质量函数公开 API
# ============================================================================


def test_calculate_quality_score_and_judge_sufficiency_are_public():
    """从 rag 顶层 import 即可使用，rag_tool 与 pipeline 共用同一份实现。"""
    from langchain_core.documents import Document

    from rag import calculate_quality_score, judge_sufficiency

    empty_score = calculate_quality_score([])
    assert empty_score == 0.0
    assert judge_sufficiency([], empty_score) == "insufficient"

    docs = [
        Document(
            page_content="x",
            metadata={"rerank_score": 0.9, "keyword_score": 0.8},
        )
        for _ in range(3)
    ]
    score = calculate_quality_score(docs)
    assert 0.0 < score <= 1.0
    # adequate 与否取决于配置的 threshold；只验证两种取值之一
    assert judge_sufficiency(docs, score) in ("adequate", "insufficient")


if __name__ == "__main__":
    print("测试 RAG 组件初始化...")
    test_translation_transformer_initialization()
    test_rewriting_transformer_initialization()
    test_reranker_initialization()
    test_rerank_strategy_enum()
    test_is_chinese_query()
    print("所有 RAG 测试通过！")
