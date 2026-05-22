# AI Agent LangChain - LiCaiManus 理财顾问

LiCaiManus 是一个基于 **FastAPI + LangChain + LangGraph** 的理财顾问 Agent 项目。它提供普通对话、RAG 知识库问答、Agent 工具调用、SSE 流式响应、FAISS 本地向量索引、可选 Redis 缓存与多后端会话记忆封装。

项目当前更适合作为「AI Agent 后端应用样板」：代码已经覆盖 API 层、LLM 工厂、RAG pipeline、Agent 执行器、工具系统、测试用例和本地启动脚本，但部分能力仍是框架级封装，尚未完全接入主业务链路。

## 核心能力

| 能力 | 当前状态 | 说明 |
|---|---:|---|
| 普通理财对话 | 已接入 | `/api/chat` 调用 LLM，以中文理财顾问身份回答 |
| 会话记忆 | 部分接入 | 普通对话使用轻量内存窗口；文件/Redis 存储后端已实现，但主链路未持久化使用 |
| RAG 问答 | 已接入 | `/api/chat/rag` 基于 `document/*.md` 构建本地知识库问答 |
| Agent 任务 | 已接入 | `/api/chat/agent` 支持 ReAct 与 Plan-and-Execute 自动选择 |
| Agentic RAG 工具 | 已接入 | Agent 可调用 `search_knowledge_base` 检索本地理财知识库 |
| SSE 流式响应 | 已接入 | `/api/chat/stream` 返回 SSE；当前是取完整答案后逐字输出，不是真正 LLM token streaming |
| FAISS 本地索引 | 已接入 | 默认使用 `./vector-index/faiss` 持久化索引 |
| Milvus | 可配置 | 代码支持 `VECTOR_STORE_TYPE=milvus`，需要外部 Milvus 服务 |
| Redis 语义缓存 | 可选 | `REDIS_ENABLED=true` 时启用 RAG 结果缓存 |
| 工具限流 | 已接入 | 工具调用支持 QPS、并发和熔断保护 |

## 技术栈

- Python：`>=3.12,<3.14`
- Web：FastAPI、Uvicorn、SSE
- Agent：LangChain、LangGraph
- LLM：OpenAI 兼容接口，默认配置指向通义千问 DashScope 兼容模式
- Embedding：OpenAI 兼容 Embedding 接口，默认 `text-embedding-v3`
- 向量库：FAISS 本地持久化；可切换 Milvus
- 缓存/记忆：Redis 可选；会话记忆支持 memory/file/redis 存储抽象
- 文档解析：内置 Markdown Q&A 解析和普通 Markdown 分块
- 测试：pytest、pytest-asyncio

## 项目结构

```text
ai-agent-langchain/
├── app/                    # FastAPI 应用层
│   ├── main.py             # API 路由与 app 对象
│   └── services.py         # FinancialAdvisorService 主服务编排
├── agents/                 # Agent 系统
│   ├── react.py            # ReAct Agent：think -> act -> think
│   ├── plan_execute.py     # Plan-and-Execute Agent：先规划后执行
│   ├── selector.py         # Agent 模式选择器
│   └── state.py            # Agent 状态对象
├── config/                 # 配置、LLM 工厂、向量库工厂
│   ├── settings.py
│   ├── llm.py
│   └── vector_store.py
├── rag/                    # RAG 检索增强系统
│   ├── transformer.py      # 查询翻译/重写
│   ├── retriever.py        # 候选召回与去重
│   ├── reranker.py         # 重排序
│   ├── pipeline.py         # 串行 RAG pipeline
│   └── cache.py            # Redis RAG 结果缓存
├── tools/                  # Agent 工具集合
│   ├── web_search.py
│   ├── web_scraper.py
│   ├── file_ops.py
│   ├── terminal.py
│   ├── downloader.py
│   ├── pdf_generator.py
│   ├── rag_tool.py
│   ├── terminate.py
│   └── rate_limiter.py
├── memory/                 # 会话记忆抽象与存储实现
├── document/               # 默认内部理财知识库 Markdown
├── static/swagger/         # 本地 Swagger UI 静态资源
├── tests/                  # pytest 测试
├── scripts/                # 初始化脚本
├── run.py                  # 本地开发启动入口
├── pyproject.toml          # 依赖权威声明
├── requirements.txt        # pip install -r 兼容入口
└── README.md
```

生成产物目录：

- `vector-index/`：FAISS 本地索引，已在 `.gitignore` 中忽略
- `chat-memory/`：文件型会话记忆目录，已在 `.gitignore` 中忽略
- `venv/`、`.env`、`__pycache__/`：本地环境产物，不进入版本管理

## 架构流程

### API 主链路

```text
HTTP 请求
  -> app.main FastAPI route
  -> FinancialAdvisorService
  -> LLM / RAG Pipeline / Agent Selector
  -> 结构化响应
```

主要入口：

- `app/main.py`：只暴露 FastAPI `app` 和路由定义
- `run.py`：本地开发唯一推荐启动入口
- `app/services.py`：编排普通对话、RAG、Agent 和缓存统计

### RAG 流程

```text
用户问题
  -> transform_query_for_retrieval
       中文：直接重写
       非中文：先翻译为中文，再重写
  -> DocumentRetriever 召回 RAG_RECALL_K 个候选
  -> DocumentReranker 重排
  -> 截断到 RAG_TOP_K
  -> 拼接上下文
  -> LLM 基于资料生成答案
```

关键约束：

- RAG 只维护串行 `RagPipeline`
- 召回阶段不使用固定相似度阈值硬过滤候选文档
- `RAG_RECALL_K` 控制候选召回数量
- `RAG_TOP_K` 控制最终进入上下文的文档数量
- `RAG_QUALITY_THRESHOLD` 仅用于 Agentic RAG 工具的信息充足性判断

### Agent 流程

```text
用户任务
  -> AgentSelector
       AGENT_MODE=react        强制 ReAct
       AGENT_MODE=plan_execute 强制 Plan-and-Execute
       AGENT_MODE=auto         规则 + LLM 分类自动选择
  -> 工具调用
  -> 循环防御 / 最大步数限制
  -> 最终答案
```

已注册工具：

| 工具 | 说明 |
|---|---|
| `search_web` | 通过 SearchAPI 搜索网页 |
| `scrape_web_page` | 抓取网页正文 |
| `file_read` | 读取文件 |
| `file_write` | 写入文件 |
| `list_files` | 列出文件 |
| `terminal_exec` | 执行终端命令 |
| `download_file` | 下载文件 |
| `generate_pdf` | 生成 PDF |
| `search_knowledge_base` | 检索本地理财知识库 |
| `do_terminate` | 显式终止 Agent 任务 |

## 快速开始

### 1. 创建虚拟环境

Windows PowerShell：

```powershell
py -3.13 -m venv venv
```

macOS / Linux：

```bash
python3.13 -m venv venv
```

### 2. 安装依赖

Windows PowerShell：

```powershell
./venv/Scripts/python.exe -m pip install -r requirements.txt
```

macOS / Linux：

```bash
./venv/bin/python -m pip install -r requirements.txt
```

### 3. 创建环境变量文件

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

macOS / Linux：

```bash
cp .env.example .env
```

至少配置：

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL_NAME=qwen-turbo
EMBEDDING_MODEL_NAME=text-embedding-v3
```

常用可选项：

```dotenv
SEARCH_API_KEY=your_searchapi_key_here

VECTOR_STORE_TYPE=inmemory
FAISS_PERSIST_DIR=./vector-index/faiss

REDIS_ENABLED=false
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

CHAT_MEMORY_STORE_TYPE=file
CHAT_MEMORY_MAX_MESSAGES=20
CHAT_MEMORY_BASE_DIR=./chat-memory

RAG_RECALL_K=50
RAG_TOP_K=5
RAG_QUALITY_THRESHOLD=0.35

AGENT_MODE=auto
AGENT_MAX_ITERATIONS=10

TOOL_RATE_LIMIT_ENABLED=true
TOOL_RATE_LIMIT_QPS=5
TOOL_RATE_LIMIT_MAX_CONCURRENT=10
TOOL_RATE_LIMIT_CIRCUIT_BREAKER_THRESHOLD=5

APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=true
```

### 4. 验证环境

Windows PowerShell：

```powershell
./venv/Scripts/python.exe -m pip check
./venv/Scripts/python.exe -c "from app.main import app; print('import OK')"
./venv/Scripts/python.exe -m pytest tests/ -v
```

macOS / Linux：

```bash
./venv/bin/python -m pip check
./venv/bin/python -c "from app.main import app; print('import OK')"
./venv/bin/python -m pytest tests/ -v
```

### 5. 启动服务

Windows PowerShell：

```powershell
./venv/Scripts/python.exe run.py
```

macOS / Linux：

```bash
./venv/bin/python run.py
```

默认访问：

- API 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/api/health`

也可以使用初始化脚本：

```powershell
./scripts/setup.ps1
```

```bash
bash scripts/setup.sh
```

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 服务信息 |
| `GET` | `/docs` | 本地 Swagger UI |
| `GET` | `/api/health` | 健康检查 |
| `POST` | `/api/chat` | 普通理财顾问对话 |
| `POST` | `/api/chat/rag` | RAG 知识库增强问答 |
| `POST` | `/api/chat/agent` | Agent 任务处理 |
| `GET` | `/api/chat/stream` | SSE 流式响应 |
| `GET` | `/api/cache/stats` | RAG 缓存统计 |
| `DELETE` | `/api/memory/{chat_id}` | 清理指定会话记忆；当前服务层仍是占位实现 |

## 请求示例

### 普通对话

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"我有 10 万元，应该怎么做资产配置？","chat_id":"demo","use_memory":true}'
```

### RAG 问答

```bash
curl -X POST http://localhost:8000/api/chat/rag \
  -H "Content-Type: application/json" \
  -d '{"message":"什么是基金定投？","chat_id":"demo"}'
```

### Agent 任务

```bash
curl -X POST http://localhost:8000/api/chat/agent \
  -H "Content-Type: application/json" \
  -d '{"message":"对比基金定投和一次性买入的优缺点","force_plan_execute":false}'
```

### SSE 流式响应

```bash
curl "http://localhost:8000/api/chat/stream?message=什么是指数基金&chat_id=demo"
```

### 缓存统计

```bash
curl http://localhost:8000/api/cache/stats
```

## 知识库维护

默认知识库来源是：

```text
document/*.md
```

解析规则：

1. 优先识别中文 Q&A 格式：

```text
Q1：问题内容

A：回答内容
```

2. 如果不是 Q&A 文档，则按 Markdown 标题和段落回退分块。
3. 文档会写入 FAISS 本地索引目录 `./vector-index/faiss`。
4. 修改 `document/*.md` 或分块策略后，需要清理 `vector-index/` 让系统重新构建索引。

## 测试覆盖重点

当前测试主要覆盖：

- 查询语言检测、翻译与重写逻辑
- RAG 召回不做固定阈值硬过滤
- 重排器元数据保留
- Markdown 知识库解析
- FAISS 索引持久化与重载
- Agent 初始化与模式选择
- ReAct 循环防御
- Agentic RAG 工具质量信号
- RAG pipeline 在 rerank 后再按 `top_k` 截断

运行：

```bash
python -m pytest tests/ -v
```

## 依赖与开发约定

- `pyproject.toml` 是依赖唯一权威声明。
- `requirements.txt` 是兼容 `pip install -r requirements.txt` 的安装入口，必须与 `pyproject.toml` 运行依赖保持一致。
- 新增、删除、升级依赖时，先改 `pyproject.toml`，再同步 `requirements.txt`。
- 不使用只有下限的依赖版本；必须带兼容上界。
- Redis 相关能力使用官方 `redis` Python 客户端，不引入 `redis-py-cluster`。
- 本地开发启动入口统一为 `python run.py`。
- `app/main.py` 只暴露 FastAPI 应用对象，不维护重复启动逻辑。
- RAG 只保留串行 `RagPipeline`，不引入并行 RAG pipeline 或并行 API 参数。

## 当前实现边界

这部分是接手项目时最容易误判的地方：

1. `/api/chat` 的 `ChatMemory` 是请求内新建的轻量内存对象，不是跨请求持久化会话。
2. `memory/` 下的 memory/file/redis 存储后端已经实现，但 `FinancialAdvisorService.chat()` 当前没有使用这些后端。
3. `/api/memory/{chat_id}` 当前调用 `clear_chat_memory()`，服务层方法仍是占位逻辑。
4. `/api/chat/stream` 是完整生成后逐字返回，不是模型原生 token stream。
5. `search_web` 需要 `SEARCH_API_KEY`，否则搜索工具不可用。
6. `VECTOR_STORE_TYPE=milvus` 需要可用 Milvus 服务和匹配的向量维度配置。
7. 首次 RAG 调用会构建 FAISS 索引，并调用 Embedding API，速度取决于文档量和模型服务。

## License

MIT
