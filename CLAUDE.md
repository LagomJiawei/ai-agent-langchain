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
- 源码不直接 `import` `langgraph` 与 `langchain-classic`（这两个包目前仍作为 `langchain` / `langchain-community` 的传递依赖被装入虚拟环境，无法只靠依赖声明移除，属于上游约束）。Agent 执行使用项目内 `harness/` 单一主循环；会话记忆使用项目自有的 `memory/` 抽象，不使用 `langchain-classic` 提供的 `ConversationBufferWindowMemory` 等组件。
- `tiktoken` 由 `harness/token_counter.py` 直接 import 做 token 计数，必须在 `pyproject.toml` 与 `requirements.txt` 显式声明，不依赖 langchain 的传递关系。
- Python 生产基线为 `>=3.12,<3.14`。

## 版本策略
- LangChain 生态按当前主版本和次版本收窄，例如 `langchain>=1.3,<1.4`。`langgraph` 与 `langchain-classic` 是 `langchain` / `langchain-community` 的传递依赖，不在 `pyproject.toml` 与 `requirements.txt` 中显式声明；本项目源码不允许直接 import 它们。
- FastAPI、Pydantic、Uvicorn 等框架依赖按当前兼容次版本收窄。
- Requests、PyYAML、Loguru、ReportLab 等基础工具库保留主版本上界。
- FAISS 与 Milvus 依赖必须显式声明；若后续改成可选安装，需要同步调整代码路径和安装说明。

## 运行入口
- 本地开发唯一推荐启动入口是 `python run.py`。
- `app/main.py` 只暴露 FastAPI 应用对象 `app`，不维护重复的脚本启动逻辑。
- 不新增平台专用启动脚本；需要平台差异时优先写入 README 或 PyCharm Run Configuration。

## Agent 实现
- Agent 仅保留 `harness/` 单一主循环（think -> act -> observe）。不再维护 ReAct 与 Plan-and-Execute 两套并行实现，不再使用 LangGraph 编排。
- 工具通过 `harness.register_tool` 注册到 `default_registry`；`tools/__init__.py` 在 import 时一次性注册全部工具。不再提供 `get_all_tools()`。
- 主循环停止原因显式为四种：`final_text` / `terminate_tool` / `loop_guard` / `max_iterations`。循环防御阈值为相同 `(tool, args)` 累计 2 次。
- 配置项只保留 `AGENT_MAX_ITERATIONS`，不再有 `AGENT_MODE` / `force_plan_execute`。
- 工具横切关注点（限流、权限、循环防御、审计）通过 `harness/hooks.py` 的 hook 总线挂载：`PreToolUseHook` 返回 `ToolResult` 即拦截、返回 `None` 即放行；`PostToolUseHook` 链式改写结果；`OnStopHook` 用于审计落盘。工具体内不做安全校验、不使用装饰器。
- 新工具默认获得限流、循环防御保护；涉及文件系统、shell、网络写入等高危工具需要为其配套提供 PreToolUse permission hook，并在 `harness/builtin_hooks.py::default_hooks()` 中注册。
- `Harness(hooks=...)` 显式传入空 `HookBus()` 可跑裸 loop（测试场景）；不传则使用 `default_hooks()` 装配。
- `ConversationContext.snapshot()` 在估算 token 数超过 `AGENT_CONTEXT_MAX_TOKENS` 时原地裁剪：永远保留 SystemMessage、首条 HumanMessage、最近 2 个 AIMessage 及其紧跟的 ToolMessage；只裁中间的 ToolMessage（先截到 200 字符，仍超阈值则按从旧到新整条删）。AIMessage 永不裁，避免孤儿 tool_call_id。
- 每次 `Harness.run()` 默认通过 `TraceWriterHook` 写一份 JSON trace 到 `AGENT_TRACE_DIR`（默认 `./traces/`），可通过 `AGENT_TRACE_ENABLED=false` 关闭。`traces/` 是产物目录，已在 `.gitignore` 中忽略，不进版本管理。
- 复杂任务通过 `dispatch_subagent(task, allowed_scopes, max_iterations=6)` 派发到隔离子 Harness 执行；子 agent 不能再调 `dispatch_subagent`（由 `harness/subagent.py::_build_sub_registry` 静态排除），但始终自动获得 `do_terminate`。
- 子 Harness 与主 Harness 共享全局 `RateLimiter` 单例（限流/熔断仍跨主子统一），但循环计数、context、trace 各自独立。
- trace 通过 `parent_trace_id` 关联父子：主 Harness 在 `run()` 开始处把 `trace_id` 发布到 `harness.loop._current_trace_id_var` (`ContextVar`)；`dispatch_subagent` 工具读取该 var 作为子 Harness 的 `parent_trace_id`。事后可从 `./traces/` 按 `parent_trace_id` 拼调用树。
- Harness 主循环对外暴露异步迭代器 `Harness.astream()`，同步 `Harness.run()` 内部委托给它（`asyncio.run`），单一事实源避免行为分裂。
- LLM 通过 `llm_with_tools.astream()` 流式产出 token chunk；工具调用仍同步执行，由 `asyncio.to_thread` 在 event loop 里包装。Hook 同样保持同步。
- 事件流类型固定 7 种：`run_start` / `thinking_token` / `tool_call` / `tool_result` / `final_text` / `run_end` / `error`，由 `harness/events.py` 工厂函数构造。
- OnStop hook 在 `run_end` 事件**之前**触发，确保 trace 文件已落地后客户端才收到结束信号。
- 子 agent (`dispatch_subagent`) 内部仍调同步 `Harness.run()`，子 agent 内部事件不暴露到主流；主流只见一次 `tool_call`/`tool_result`。
- `chat()` 路径通过 `memory.get_memory_store()` 实现跨请求会话记忆（按 `CHAT_MEMORY_STORE_TYPE` 选 memory/file/redis）；`chat_with_agent` 与 RAG 路径不接 store（agent 是任务粒度，RAG 是字典式查询）。
- `chat_id` 从 FastAPI 路由层传入到 `Harness(chat_id=...)`；Harness 主循环不消费它，只作审计标签 + trace 分桶维度；子 Harness 通过 `harness.loop.current_chat_id()` ContextVar 自动继承父 chat_id。
- Trace 文件按 chat_id 落到 `AGENT_TRACE_DIR/<sanitized_chat_id or _default>/<trace_id>.json`；chat_id 含 `/`、`..` 等会被 `harness.trace._sanitize_chat_id` 替换为 `_`，防止写穿 base_dir。
- `Harness.arun(query)` 是 async 公开入口；`Harness.run()` 是 `asyncio.run(self.arun(...))` 同步外壳。两路径单一事实源，外部并发场景请直接 `await sub.arun(...)`。
- 多个独立子任务用 `dispatch_subagents(tasks, max_concurrency=4)` 并发派发；并发用 `asyncio.gather` + `asyncio.Semaphore`，硬上限 16 个任务、16 并发。子 agent 不能再调 `dispatch_subagent` / `dispatch_subagents`（由 `_build_sub_registry` 静态排除）。
- 选择规则：单任务用具体工具；2+ 有依赖任务用多次 `dispatch_subagent` 顺序调用；2+ 独立任务用 `dispatch_subagents` 一次并发。

## 身份与多租户隔离
- HTTP 请求通过 `X-API-Key` 头识别调用方。`app/security.py::resolve_principal` 是 FastAPI dependency，从 `settings.security.api_keys`（`API_KEYS` env JSON 映射）查 principal_id。
- `Principal.namespace_chat_id(raw)` 把客户端 chat_id 命名空间化为 `<principal>:<raw>`。命名空间化**只发生在路由层**，下游 Harness / memory store / trace 把命名空间化后的字符串当透明 chat_id，零感知、零接口改动。
- `ALLOW_ANONYMOUS=true`（默认）：未带 key 的请求映射为 `principal_id="anonymous"`，落到 `anonymous:*` 命名空间，向后兼容旧调用方。`ALLOW_ANONYMOUS=false`：未带或带未知 key 一律 401。
- principal A 永远访问不到 principal B 的会话或 trace（命名空间不同即不同 store key 与不同 trace 子目录），无需额外权限检查。
- `/api/health` 不强制鉴权（容器 probe 需要）；`/docs`、`/swagger-assets/*`、`/` 同样开放。其余业务路由全部走 `Depends(resolve_principal)`。
- **不引入** RBAC、per-principal 限流、JWT/OAuth、用户表 —— 当且仅当真实多租户需求出现再加。`Principal.is_anonymous` 字段已预留给未来 hook 用。

## RAG 实现
- RAG 仅保留串行 `RagPipeline` 实现，不维护并行 RAG pipeline、并行配置或并行 API 参数。
- 查询变换统一使用 `transform_query_for_retrieval`：中文跳过翻译直接重写，非中文先翻译再基于翻译结果重写。
- `document/*.md` 是默认内部知识库来源；本地 FAISS 持久化索引默认保存到 `./vector-index/faiss`，可用 `FAISS_PERSIST_DIR` 覆盖。
- RAG 检索采用“候选召回 → rerank → top_k 截断 → sufficiency 判断”流程；不要在向量召回阶段使用固定相似度阈值硬过滤候选文档。
- `RAG_RECALL_K` 控制候选召回数量，`RAG_TOP_K` 控制最终上下文数量，`RAG_QUALITY_THRESHOLD` 仅用于 Agentic RAG 的信息充足性判断。
- `vector-index/` 是生成产物目录，不进入版本管理；文档变更或解析策略变化后如需重建索引，先清理该目录或后续实现显式重建命令。
- `RagPipeline.astream_execute(query)` 是异步事件流入口（5 类事件：`retrieval_started` / `retrieval_done` / `generation_token` / `done` / `error`）；同步 `RagPipeline.execute()` 保留并仍走缓存。流式路径**不走缓存**，每次完整跑 retrieval + generation。
- `calculate_quality_score` / `judge_sufficiency` 是 `rag` 包公开 API，`rag_tool.py` 工具与流式 pipeline 共用同一份实现，避免漂移。

## 验证命令
依赖变更后必须运行：

```bash
python -m pip check
python -c "from app.main import app; print('import OK')"
python -m pytest tests/ -v
```

如果验证失败，先定位根因；不要通过注释代码、跳过测试或添加绕过标记来让验证通过。
