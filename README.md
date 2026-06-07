# AI Agent LangChain - LiCaiManus 理财顾问

LiCaiManus 是一个基于 **FastAPI + LangChain + 自研 Harness** 的理财顾问 Agent 项目。它提供普通对话、RAG 知识库问答、Agent 工具调用、SSE 流式响应、FAISS 本地向量索引、可选 Redis 缓存与多后端会话记忆封装。

代码覆盖 API 层、LLM 工厂、RAG pipeline、Agent 执行器、工具系统、Hook 总线、身份与多租户隔离、SSE 心跳、两层 RAG 缓存、多后端会话记忆，可直接作为 AI Agent 后端应用模板使用。当前测试 **219 passed**。

## 核心能力

| 能力 | 当前状态 | 说明 |
|---|---:|---|
| 普通理财对话 | 已接入 | `/api/chat` 调用 LLM，以中文理财顾问身份回答 |
| 会话记忆 | 已接入 | 普通对话按 `CHAT_MEMORY_STORE_TYPE` 使用 memory/file/redis store 跨请求保存；RAG 与 Agent 路径不接会话记忆 |
| RAG 问答 | 已接入 | `/api/chat/rag` 基于 `document/*.md` 构建本地知识库问答 |
| Agent 任务 | 已接入 | `/api/chat/agent` 由项目自研 Harness 主循环驱动（think -> act -> observe） |
| Agentic RAG 工具 | 已接入 | Agent 可调用 `search_knowledge_base` 检索本地理财知识库 |
| SSE 流式响应 | 已接入 | `/api/chat/stream`、`/api/chat/agent/stream`、`/api/chat/rag/stream` 三个端点全部内置 keepalive 心跳（默认 15s） |
| FAISS 本地索引 | 已接入 | 默认使用 `./vector-index/faiss` 持久化索引 |
| Milvus | 可配置 | 代码支持 `VECTOR_STORE_TYPE=milvus`，需要外部 Milvus 服务 |
| Redis 两层缓存 | 可选 | `REDIS_ENABLED=true` 时启用：answer 缓存（24h）+ retrieval 缓存（7d，流式路径也用） |
| 工具限流 | 已接入 | QPS、并发、熔断三件套；hook 体系驱动，线程安全 + 重发幂等 |
| 身份与多租户 | 已接入 | `X-API-Key` 头识别 principal，chat_id 自动命名空间隔离 |
| Trace 落盘 | 已接入 | 每次 `Harness.run()` / `Harness.astream()` 写一份带 `schema_version` 的 JSON trace，按 `chat_id` 分桶 |

## 技术栈

- Python：`>=3.12,<3.14`
- Web：FastAPI、Uvicorn、SSE
- Agent：LangChain（消息与工具抽象）+ 自研 Harness 主循环
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
│   ├── services.py         # FinancialAdvisorService 主服务编排
│   ├── security.py         # 身份与多租户隔离（X-API-Key → Principal）
│   └── sse.py              # SSE keepalive 心跳 helper
├── harness/                # Agent 执行框架
│   ├── loop.py             # Harness 主循环：think → act → observe
│   ├── registry.py         # ToolRegistry：按 scope 注册 / 过滤工具
│   ├── context.py          # ConversationContext：消息上下文 + token 预算裁剪
│   ├── hooks.py            # HookBus：横切关注点（限流/权限/审计）
│   ├── builtin_hooks.py    # 内置 hook 实现（RateLimit/LoopGuard/Allowlist/TraceWriter）
│   ├── subagent.py         # dispatch_subagent / dispatch_subagents 并发派发
│   ├── turn.py             # Turn / ToolCall / ToolResult 数据模型
│   ├── trace.py            # HarnessTrace JSON 序列化（含 schema_version）
│   ├── token_counter.py    # tiktoken 包装的 token 计数
│   ├── _message_utils.py   # extract_chunk_text / extract_message_text 统一文本提取
│   ├── events.py           # 事件工厂（run_start / thinking_token / …）
│   └── prompts/            # 主 system prompt
├── config/                 # 配置、LLM 工厂、向量库工厂
│   ├── settings.py
│   ├── llm.py
│   └── vector_store.py
├── rag/                    # RAG 检索增强系统
│   ├── transformer.py      # 查询翻译/重写
│   ├── retriever.py        # 候选召回与去重
│   ├── reranker.py         # 重排序
│   ├── pipeline.py         # 串行 RAG pipeline（同步 + 流式）
│   ├── cache.py            # Redis RAG answer 缓存
│   ├── retrieval_cache.py  # retrieval 级缓存（流式/同步共用）
│   └── events.py           # RAG 事件工厂（RetrievalStarted / RagEvent / …）
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
- `traces/`：Harness 执行 trace，按 `chat_id` 分桶，已在 `.gitignore` 中忽略
- `venv/`、`.env`、`__pycache__/`：本地环境产物，不进入版本管理

## 架构流程

### API 主链路

```text
HTTP 请求
  -> app.main FastAPI route
  -> FinancialAdvisorService
  -> LLM / RAG Pipeline / Harness
  -> 结构化响应或 SSE 事件流
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
  -> Harness.run()
       PreToolUse hooks → think：LLM 推理（带工具）
       act：从 ToolRegistry 取工具执行（被 hook 拦截则跳过真实执行）
       observe：PostToolUse hooks 改写结果 → 注入 ToolMessage 到上下文
  -> 循环防御 / 最大步数限制 / do_terminate
  -> OnStop hooks → 最终答案（stopped_reason: final_text / terminate_tool / loop_guard / max_iterations / error）
```

`Harness` 同时暴露 `run()`（同步）/ `arun()`（async 公开入口）/ `astream()`（async 事件迭代器）三种调用方式，三者共享单一事实源（`run()` 内部委托 `arun()`，`arun()` 委托 `astream()`），外部并发场景请直接 `await sub.arun(...)`。

### Hook 系统

横切关注点（限流、权限、循环防御、审计）由 `harness/hooks.py` 的 hook 总线统一处理，工具体内不做安全校验。Hook 既支持 `def` 也支持 `async def` —— async hook 由 `HookBus.arun_pre / arun_post / arun_stop` 原生 `await`，sync hook 自动 `asyncio.to_thread` 派到工作线程，**不阻塞 event loop**（避免 RateLimit 的 blocking semaphore acquire 冻结主循环）。

默认装配的 PreToolUse 链：

| 顺序 | Hook | 职责 |
|---|---|---|
| 1 | `RateLimitPreHook` | 占令牌、查熔断；耗尽即拒绝；线程安全 + 同 `call.id` 重发幂等 |
| 2 | `TerminalAllowlistHook` | `terminal_exec` 的白名单校验 |
| 3 | `FilePathAllowlistHook` | `file_read` / `file_write` / `list_files` 的路径白名单 |
| 4 | `LoopGuardHook` | 相同 `(tool, args)` 累计 2 次拒绝并停止主循环 |

PostToolUse 链：`RateLimitPostHook` 反馈结果给熔断器并归还信号量。

OnStop 链：

| 顺序 | Hook | 职责 |
|---|---|---|
| 1 | `RateLimitSweeperStopHook` | 异常路径兜底归还所有未配对释放的 semaphore，防泄漏 |
| 2 | `TraceWriterHook` | 把 `HarnessTrace` 写入 `AGENT_TRACE_DIR/<chat_id>/<trace_id>.json` |

测试可通过 `Harness(hooks=HookBus())` 绕过默认装配跑裸 loop。

### 子 agent 派发

主 agent 在 think 阶段判断"这是个独立子任务"时，可调用 `dispatch_subagent(task, allowed_scopes, max_iterations=6)`：

- 工具体内开一个新 `Harness` 实例，registry 按 `allowed_scopes` 多 scope 并集过滤（如 `["kb", "web"]` 同时给知识库和联网能力）。
- 子 agent **始终强制注入 `do_terminate`**，无论 `allowed_scopes` 是否含 `control`，确保可显式收尾。
- 子 agent **永远拿不到 `dispatch_subagent` 自身**，防递归爆栈。
- 子 Harness 与主 Harness 共享全局 `RateLimiter` 单例，但循环计数与 trace 各自独立。
- 主 Harness 在 `run()` 开始时把自己的 `trace_id` 发布到 `ContextVar`；子 Harness 从中读取作为 `parent_trace_id` 写入子 trace，事后可从 `./traces/` 按 `parent_trace_id` 拼出完整调用树。

### 并发子 agent

主 agent 一次性派发多个**彼此独立**的子任务时调用 `dispatch_subagents(tasks, max_concurrency=4)`：

- `tasks` 是 dict 列表，每项 `{"task": "...", "allowed_scopes": ["kb"], "max_iterations": 6}`。
- 实现：`asyncio.gather` + `asyncio.Semaphore` 控制并发，默认 4，硬上限 16 个任务 / 16 并发。
- 单个子任务失败不影响兄弟，返回 JSON 数组里该项是 `{"index": N, "task": "...", "error": "..."}`。
- 子 agent 内部禁止再调 `dispatch_subagent` / `dispatch_subagents`（静态排除防嵌套）。
- 依赖关系任务请用多次 `dispatch_subagent` 顺序调用，不要塞进 `tasks`。

### 流式接口（SSE）

三个 SSE 端点都内置 keepalive 心跳：流上游 idle 超过 `SSE_KEEPALIVE_INTERVAL`（默认 15s）时自动发出 SSE comment 行 `: keepalive <unix_ts>\n\n`，符合 WHATWG SSE 规范，标准 EventSource 客户端自动忽略。

- `GET /api/chat/stream?message=...` —— 轻量对话真 token 流。每条 `data:` 是 `{"delta": "..."}`，最后 `event: end`。
- `GET /api/chat/agent/stream?message=...&chat_id=default` —— Agent 事件流，事件类型固定 7 种：

| `event:` | `data:` 字段 | 说明 |
|---|---|---|
| `run_start` | `trace_id` / `parent_trace_id` / `user_query` / `started_at` | 主循环开始 |
| `thinking_token` | `turn_index` / `delta` | LLM 流式 token 增量 |
| `tool_call` | `turn_index` / `call` | LLM 决定调用某工具 |
| `tool_result` | `turn_index` / `result` | 工具返回（或 hook 拦截结果） |
| `final_text` | `final_text` | 最终答案（仅正常路径发，异常路径跳过） |
| `run_end` | `stopped_reason` / `total_tool_calls` / `finished_at` | 主循环结束，`stopped_reason` ∈ `final_text` / `terminate_tool` / `loop_guard` / `max_iterations` / `error` |
| `error` | `message` | 异常信息（异常路径事件序列：`error` → OnStop 写 trace → `run_end(stopped_reason="error")`，客户端仍能确定结束） |

底层：`Harness.astream()` 是头等异步接口，`Harness.run()` 内部委托给它，保证两条路径行为完全一致。

- `GET /api/chat/rag/stream?message=...&chat_id=default` —— RAG 流式问答，事件类型固定 5 种：

| `event:` | `data:` 字段 | 说明 |
|---|---|---|
| `retrieval_started` | `original_query` / `rewritten_query` | 查询变换完成，开始检索 |
| `retrieval_done` | `doc_count` / `quality_score` / `sufficiency` / `titles` / `from_cache` | 检索 + 重排完成，含质量信号；`from_cache=true` 表示命中 retrieval 缓存 |
| `generation_token` | `delta` | LLM 流式 token 增量 |
| `done` | `finished_at` | 生成结束 |
| `error` | `message` | 异常信息 |

流式 RAG **不写 answer 缓存**（保持 token 流体验）但**读写 retrieval 缓存**（命中时 `retrieval_done.from_cache=true`，跳过 retriever + reranker，token 流仍正常发）；同步 `/api/chat/rag` 路径先 answer cache、未命中走 retrieval cache，生成后写两层。

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
| `dispatch_subagent` | 把单个独立子任务派发到隔离子 Harness（按 scope 白名单授权工具集） |
| `dispatch_subagents` | 一次并发派发多个独立子任务，``asyncio.gather`` 实现，默认 4 并发、硬上限 16 |

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
RAG_RETRIEVAL_CACHE_TTL=604800

AGENT_MAX_ITERATIONS=10
AGENT_CONTEXT_MAX_TOKENS=5734
AGENT_TRACE_ENABLED=true
AGENT_TRACE_DIR=./traces

TOOL_RATE_LIMIT_ENABLED=true
TOOL_RATE_LIMIT_QPS=5
TOOL_RATE_LIMIT_MAX_CONCURRENT=10
TOOL_RATE_LIMIT_CIRCUIT_BREAKER_THRESHOLD=5

APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=true
# 注意：当前 run.py 的 reload 开关读取 DEBUG，不读取 APP_DEBUG
DEBUG=true
SSE_KEEPALIVE_INTERVAL=15

# 多租户（可选）：JSON 映射 {api_key: principal_id}
# API_KEYS={"sk-alice":"alice","sk-bob":"bob"}
ALLOW_ANONYMOUS=true
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

## 身份与多租户隔离

所有业务路由通过 HTTP 头 `X-API-Key` 识别调用方：

- 服务启动时从 `API_KEYS` env JSON 映射加载 `{key: principal_id}`。
- 请求带合法 key → 解析为对应 `principal_id`；带未知 key → 401。
- 请求未带 key 且 `ALLOW_ANONYMOUS=true`（默认）→ 解析为 `anonymous` principal。
- 请求未带 key 且 `ALLOW_ANONYMOUS=false` → 401。

路由层把客户端传入的 `chat_id` 自动命名空间化为 `<principal>:<chat_id>`：

- 会话历史按 principal 隔离（`anonymous:default` ≠ `alice:default`）。
- Trace 文件按 principal 自动分到 `traces/<principal>_<chat_id>/` 子目录（`_sanitize_chat_id` 把 `:` 替换为 `_`）。
- `DELETE /api/memory/<chat_id>` 永远只能删自己 principal 名下的会话。

不强制鉴权的路由：`GET /api/health`、`/docs`、`/swagger-assets/*`、`/`（运维与文档需要）。

不引入 RBAC、JWT、per-principal 限流、用户表 —— 当前是最小可用身份层。

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 服务信息 |
| `GET` | `/docs` | 本地 Swagger UI |
| `GET` | `/api/health` | 健康检查 |
| `POST` | `/api/chat` | 普通理财顾问对话 |
| `POST` | `/api/chat/rag` | RAG 知识库增强对话（同步，走缓存） |
| `POST` | `/api/chat/agent` | Agent 任务处理（同步）；``chat_id`` 用于 trace 分桶 |
| `GET` | `/api/chat/stream` | 轻量对话真 token 流（SSE）；``?message=...&chat_id=...`` |
| `GET` | `/api/chat/rag/stream` | RAG 流式问答（SSE，含心跳）；不写 answer 缓存但读写 retrieval 缓存 |
| `GET` | `/api/chat/agent/stream` | Agent 事件流（SSE，含工具调用过程 + 心跳）；``?message=...&chat_id=...`` |
| `GET` | `/api/cache/stats` | RAG answer 缓存统计（不包含 retrieval 缓存数量） |
| `DELETE` | `/api/memory/{chat_id}` | 清理指定会话记忆（从 ``memory/`` store 实际删除） |

## 请求示例

> 以下示例默认走 `anonymous` 身份。如需多租户隔离，给每个请求加 `-H "X-API-Key: sk-xxx"` 并在 `.env` 中配置 `API_KEYS={"sk-xxx":"alice"}`。

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
  -d '{"message":"对比基金定投和一次性买入的优缺点"}'
```

### SSE 流式响应

```bash
curl "http://localhost:8000/api/chat/stream?message=什么是指数基金&chat_id=demo"
```

### 缓存统计

```bash
curl http://localhost:8000/api/cache/stats
```

该接口统计的是 RAG answer 缓存；retrieval 缓存目前没有单独暴露统计 API。

### RAG 流式问答

```bash
curl -N "http://localhost:8000/api/chat/rag/stream?message=什么是基金定投&chat_id=demo"
```

### Agent 事件流

```bash
curl -N "http://localhost:8000/api/chat/agent/stream?message=检索知识库并说明基金定投适合什么人&chat_id=demo"
```

### 清理会话记忆

```bash
curl -X DELETE http://localhost:8000/api/memory/demo
```

多租户示例：

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-alice" \
  -d '{"message":"继续刚才的话题","chat_id":"demo","use_memory":true}'
```

服务端实际使用的会话键是 `alice:demo`，响应中仍返回客户端传入的 `demo`。

## 会话记忆

普通 `/api/chat` 路径在 `use_memory=true` 时会读写会话记忆：

```text
路由层 chat_id
  -> Principal.namespace_chat_id()
  -> <principal>:<chat_id>
  -> memory store
  -> 最近 10 轮进入 LLM 上下文
```

可选后端：

| `CHAT_MEMORY_STORE_TYPE` | 说明 |
|---|---|
| `memory` | 进程内临时存储，进程退出即丢失 |
| `file` | 默认，写入 `CHAT_MEMORY_BASE_DIR`，适合本地开发 |
| `redis` | 使用 Redis 保存，适合多进程 / 容器场景 |

边界：

- `/api/chat/stream` 是轻量 token 流，不读写会话记忆。
- `/api/chat/rag` 与 `/api/chat/rag/stream` 不读写会话记忆；RAG 是基于知识库的字典式查询。
- `/api/chat/agent` 与 `/api/chat/agent/stream` 不读写会话记忆；Agent 是任务粒度执行，`chat_id` 只用于审计和 trace 分桶。
- `DELETE /api/memory/{chat_id}` 只清理当前 principal 命名空间下的会话。

## Trace 与审计

默认 `AGENT_TRACE_ENABLED=true`，每次 `Harness.run()` / `Harness.astream()` 都会在 OnStop 阶段写入 JSON trace：

```text
AGENT_TRACE_DIR/
└── <sanitized_chat_id>/
    └── <trace_id>.json
```

Trace 顶层包含 `schema_version`，当前版本由 `harness.trace.TRACE_SCHEMA_VERSION` 定义。改动 trace schema 时必须同步递增该常量和项目 `CLAUDE.md` 中的版本说明。

主子 agent trace 通过 `parent_trace_id` 关联：主 Harness 开始时把当前 `trace_id` 发布到 `ContextVar`，`dispatch_subagent` / `dispatch_subagents` 创建的子 Harness 会继承并写入 `parent_trace_id`。

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
- Agent 初始化与 Harness 主循环停止原因（final_text / terminate_tool / loop_guard / max_iterations / error）
- Harness hook 总线、工具限流、白名单、循环防御与异常兜底释放
- Harness async 入口、SSE 事件流、上下文 token 裁剪、LLM bind_tools 缓存
- 子 agent 单任务 / 多任务派发、chat_id ContextVar 继承、trace 父子关联
- 身份与多租户 chat_id 命名空间隔离
- 普通对话跨请求会话记忆读写与清理
- Agentic RAG 工具质量信号
- RAG pipeline 在 rerank 后再按 `top_k` 截断
- RAG 流式事件、retrieval 缓存、SSE keepalive 心跳
- 消息 chunk 文本提取工具函数

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
- Agent 只保留 `harness/` 单一主循环，不恢复 ReAct / Plan-and-Execute / LangGraph 并行实现。
- 工具通过 `harness.register_tool` 注册到 `default_registry`；新增高危工具时要配套 permission hook 并在 `default_hooks()` 注册。
- 工具体内不做横切安全校验；限流、权限、循环防御、审计统一通过 hook 总线实现。
- 多个独立子任务用 `dispatch_subagents` 并发派发；有依赖关系的子任务用多次 `dispatch_subagent` 顺序执行。
- RAG 只保留串行 `RagPipeline`，不引入并行 RAG pipeline 或并行 API 参数。
- SSE 心跳只放在 `app/sse.py` 传输层，不下沉到 Harness 或 RAG pipeline。

## 当前实现边界

这部分是接手项目时最容易误判的地方：

1. `/api/chat/stream` 是模型原生 token 流，不是后处理的逐字回放；它是轻量对话，不接 RAG，也不接跨请求记忆。
2. `/api/chat/rag/stream` 不写 answer 缓存，但会读写 retrieval 缓存；同步 `/api/chat/rag` 会先查 answer 缓存，未命中再走 retrieval 缓存。
3. `/api/chat/agent` 和 `/api/chat/agent/stream` 不接会话记忆；`chat_id` 用于审计标签和 trace 分桶。
4. `search_web` 需要 `SEARCH_API_KEY`，否则搜索工具不可用。
5. `VECTOR_STORE_TYPE=milvus` 需要可用 Milvus 服务；当前 `MILVUS_DIMENSION` 配置存在于 settings 中，但 `create_milvus_vector_store()` 未显式使用该字段，集合维度主要由 embedding 写入时决定。
6. 首次 RAG 调用会构建 FAISS 索引，并调用 Embedding API，速度取决于文档量和模型服务。
7. `vector-index/`、`chat-memory/`、`traces/` 都是运行产物目录，不应提交。
8. `ALLOW_ANONYMOUS=true` 是向后兼容默认值；生产场景应配置 `API_KEYS` 并按需关闭匿名访问。
9. Trace JSON 有 schema 版本，离线分析脚本读取时应按 `schema_version` 做兼容。

## License

MIT
