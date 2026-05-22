# CLAUDE.md

## 项目协作规则
- 默认使用中文沟通；代码、命令、变量名使用英文。
- 遵守全局红线：删除文件、修改 `.env`、改 CI/CD、数据库迁移、git push/rebase/reset、安装全局依赖、公开发布前必须先确认。
- 修改依赖、运行入口或配置约定时，先更新本文件中的规则，再修改实际文件。

## 依赖管理
- `pyproject.toml` 是依赖的唯一权威声明。
- `requirements.txt` 是兼容 `pip install -r requirements.txt` 的安装入口，内容必须与 `pyproject.toml` 的运行依赖保持一致。
- 新增、删除、升级依赖时，先修改 `pyproject.toml`，再同步 `requirements.txt`。
- 源码直接 `import` 的第三方包必须在 `pyproject.toml` 中显式声明，不能只依赖传递依赖。
- 禁止只写 `>=` 下限；每个依赖必须使用兼容上界，例如 `>=1.3,<1.4`、`>=2.13,<2.14`、`>=2.34,<3.0`。
- 不再引入 `redis-py-cluster`，Redis 相关功能使用官方 `redis` Python 客户端。
- Python 生产基线为 `>=3.12,<3.14`。

## 版本策略
- LangChain / LangGraph 生态按当前主版本和次版本收窄，例如 `langchain>=1.3,<1.4`。
- FastAPI、Pydantic、Uvicorn 等框架依赖按当前兼容次版本收窄。
- Requests、PyYAML、Loguru、ReportLab 等基础工具库保留主版本上界。
- FAISS 与 Milvus 依赖必须显式声明；若后续改成可选安装，需要同步调整代码路径和安装说明。

## 运行入口
- 本地开发唯一推荐启动入口是 `python run.py`。
- `app/main.py` 只暴露 FastAPI 应用对象 `app`，不维护重复的脚本启动逻辑。
- 不新增平台专用启动脚本；需要平台差异时优先写入 README 或 PyCharm Run Configuration。

## RAG 实现
- RAG 仅保留串行 `RagPipeline` 实现，不维护并行 RAG pipeline、并行配置或并行 API 参数。
- 查询变换统一使用 `transform_query_for_retrieval`：中文跳过翻译直接重写，非中文先翻译再基于翻译结果重写。
- `document/*.md` 是默认内部知识库来源；本地 FAISS 持久化索引默认保存到 `./vector-index/faiss`，可用 `FAISS_PERSIST_DIR` 覆盖。
- RAG 检索采用“候选召回 → rerank → top_k 截断 → sufficiency 判断”流程；不要在向量召回阶段使用固定相似度阈值硬过滤候选文档。
- `RAG_RECALL_K` 控制候选召回数量，`RAG_TOP_K` 控制最终上下文数量，`RAG_QUALITY_THRESHOLD` 仅用于 Agentic RAG 的信息充足性判断。
- `vector-index/` 是生成产物目录，不进入版本管理；文档变更或解析策略变化后如需重建索引，先清理该目录或后续实现显式重建命令。

## 验证命令
依赖变更后必须运行：

```bash
python -m pip check
python -c "from app.main import app; print('import OK')"
python -m pytest tests/ -v
```

如果验证失败，先定位根因；不要通过注释代码、跳过测试或添加绕过标记来让验证通过。
