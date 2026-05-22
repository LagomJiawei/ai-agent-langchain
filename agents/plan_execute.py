"""
Plan-and-Execute Agent
先规划所有步骤，再按批次并行执行
"""
import json
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import BaseTool

from config import create_chat_model, settings
from tools import get_all_tools
from .state import AgentState
from .react import ReActAgent


class PlanStep:
    """规划步骤"""

    def __init__(
        self,
        step_id: int,
        description: str,
        tool: str,
        params: Dict[str, Any],
        parallel: bool = False,
        confidence: float = 0.5,
    ):
        self.id = step_id
        self.description = description
        self.tool = tool
        self.params = params
        self.parallel = parallel
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "tool": self.tool,
            "params": self.params,
            "parallel": self.parallel,
            "confidence": self.confidence,
        }


class ExecutionResult:
    """执行结果"""

    def __init__(
        self,
        step_id: int,
        description: str,
        success: bool,
        result: str = "",
        error: str = "",
    ):
        self.step_id = step_id
        self.description = description
        self.success = success
        self.result = result
        self.error = error


class PlanAndExecuteAgent:
    """Plan-and-Execute Agent 实现"""

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        max_steps: int = 15,
    ):
        self.tools = tools or get_all_tools()
        self.max_steps = max_steps
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.llm = create_chat_model(temperature=0.7)
        # 降级用的 ReAct Agent
        self.react_agent = ReActAgent(tools, max_iterations=max_steps)

    def _get_planning_prompt(self, user_query: str) -> str:
        """获取规划 Prompt"""
        tool_names = ", ".join([t.name for t in self.tools])
        return f"""你是智能任务规划专家。请将用户的请求拆分为具体的执行步骤。

用户请求: {user_query}

可用工具: {tool_names}

规划要求：
1. 每个步骤描述清晰，可以用工具完成
2. 涉及理财知识、基金、股票、债券、保险、资产配置等本地知识库问题时，优先使用 search_knowledge_base
3. 对比、规划、综合分析类问题可拆成多个 search_knowledge_base 子查询
4. 可以并行执行的步骤标记为 parallel: true
5. 有依赖关系的按顺序排列，标记为 parallel: false
6. 预估每个步骤的置信度 (0-1)

只返回 JSON 格式，不要其他内容：
{{
  "steps": [
    {{
      "id": 1,
      "description": "步骤描述",
      "tool": "search_web",
      "params": {{"query": "关键词"}},
      "parallel": false,
      "confidence": 0.9
    }}
  ],
  "estimated_steps": 3,
  "strategy": "并行执行可独立步骤"
}}"""

    def _get_synthesis_prompt(self, user_query: str, results: List[ExecutionResult]) -> str:
        """获取答案合成 Prompt"""
        results_text = "\n".join(
            [
                f"步骤 {r.step_id}: {r.description}\n  结果: {r.result if r.success else '失败: ' + r.error}"
                for r in results
            ]
        )
        return f"""请基于以下执行结果回答用户问题。

用户原始问题: {user_query}

执行结果汇总:
{results_text}

要求：
1. 基于真实结果回答，不要编造信息
2. 结构清晰，分点说明
3. 标注信息来源
4. 用中文回答"""

    def _parse_plan(self, response_text: str) -> List[PlanStep]:
        """解析规划结果"""
        try:
            # 提取 JSON 部分
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]

            data = json.loads(json_str)
            steps = []
            for step_data in data.get("steps", []):
                step = PlanStep(
                    step_id=step_data.get("id", 0),
                    description=step_data.get("description", ""),
                    tool=step_data.get("tool", ""),
                    params=step_data.get("params", {}),
                    parallel=step_data.get("parallel", False),
                    confidence=step_data.get("confidence", 0.5),
                )
                steps.append(step)
            return steps
        except Exception as e:
            logger.warning(f"解析规划失败，使用默认规划: {e}")
            # 返回默认规划：使用搜索
            return [
                PlanStep(
                    step_id=1,
                    description=f"搜索关于 '{user_query}' 的相关信息",
                    tool="search_web",
                    params={"query": user_query},
                    parallel=False,
                    confidence=0.5,
                )
            ]

    def _plan(self, user_query: str) -> List[PlanStep]:
        """阶段 1：规划步骤"""
        logger.info("【规划阶段】开始规划任务步骤")

        prompt = self._get_planning_prompt(user_query)
        response = self.llm.invoke(prompt)
        steps = self._parse_plan(response.content)

        logger.info(f"【规划阶段】完成，共 {len(steps)} 步")
        return steps

    def _group_by_dependencies(self, steps: List[PlanStep]) -> List[List[PlanStep]]:
        """按依赖关系分组"""
        batches = []
        current_batch = []

        for step in steps:
            if step.parallel:
                current_batch.append(step)
            else:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                batches.append([step])

        if current_batch:
            batches.append(current_batch)

        return batches

    def _execute_single_step(self, step: PlanStep, state: AgentState) -> ExecutionResult:
        """执行单个步骤"""
        logger.debug(f"执行步骤 {step.id}: {step.description}")

        # 循环检测
        if state.is_in_loop(step.tool, json.dumps(step.params, sort_keys=True)):
            return ExecutionResult(
                step_id=step.id,
                description=step.description,
                success=False,
                error="循环检测拦截",
            )

        state.record_tool_call(step.tool, json.dumps(step.params, sort_keys=True))

        try:
            tool = self.tool_map.get(step.tool)
            if tool is None:
                raise ValueError(f"未知工具: {step.tool}")
            result = tool.invoke(step.params)

            return ExecutionResult(
                step_id=step.id,
                description=step.description,
                success=True,
                result=str(result),
            )
        except Exception as e:
            return ExecutionResult(
                step_id=step.id,
                description=step.description,
                success=False,
                error=str(e),
            )

    def _execute_batch(
        self,
        batch: List[PlanStep],
        state: AgentState,
    ) -> List[ExecutionResult]:
        """按批次执行步骤"""
        if len(batch) == 1:
            # 单步执行
            return [self._execute_single_step(batch[0], state)]

        # 并行执行
        results = []
        with ThreadPoolExecutor(max_workers=min(10, len(batch))) as executor:
            future_to_step = {
                executor.submit(self._execute_single_step, step, state): step
                for step in batch
            }
            for future in as_completed(future_to_step):
                results.append(future.result())

        # 按 step_id 排序返回结果
        return sorted(results, key=lambda r: r.step_id)

    def _should_terminate_early(
        self,
        results: List[ExecutionResult],
        user_query: str,
    ) -> bool:
        """早期终止检测"""
        # 成功结果数量阈值
        success_count = sum(1 for r in results if r.success)
        if success_count >= 3:
            return True

        # 结果内容长度阈值（信息充足）
        total_length = sum(
            len(r.result) for r in results if r.success and r.result is not None
        )
        return total_length > 1000

    def _synthesize_answer(
        self,
        user_query: str,
        results: List[ExecutionResult],
    ) -> str:
        """合成最终答案"""
        prompt = self._get_synthesis_prompt(user_query, results)
        response = self.llm.invoke(prompt)
        return response.content

    def execute(self, user_query: str) -> AgentState:
        """
        执行 Plan-and-Execute 模式

        Args:
            user_query: 用户查询

        Returns:
            Agent 状态（包含执行结果）
        """
        start_time = time.time()
        logger.info(f"【Plan-and-Execute】开始处理: {user_query}")

        # 初始化状态
        state = AgentState(
            user_query=user_query,
            max_steps=self.max_steps,
        )

        try:
            # ========== 阶段 1: 规划 ==========
            steps = self._plan(user_query)

            # ========== 阶段 2: 按批次执行 ==========
            all_results: List[ExecutionResult] = []
            batches = self._group_by_dependencies(steps)

            for i, batch in enumerate(batches, 1):
                batch_num = i
                logger.info(
                    f"【Plan-and-Execute】执行批次 {batch_num}/{len(batches)}, 并行度: {len(batch)}"
                )

                batch_results = self._execute_batch(batch, state)
                all_results.extend(batch_results)

                # 早期终止检测
                if self._should_terminate_early(all_results, user_query):
                    logger.info("【Plan-and-Execute】检测到信息充足，提前终止")
                    break

                state.increment_step()
                if state.is_max_steps_reached():
                    logger.warning("【Plan-and-Execute】达到最大步数限制")
                    break

            # ========== 阶段 3: 答案合成 ==========
            final_answer = self._synthesize_answer(user_query, all_results)
            state.is_finished = True
            state.final_answer = final_answer

            elapsed = (time.time() - start_time) * 1000
            logger.info(
                f"【Plan-and-Execute】完成，总耗时: {elapsed:.0f}ms，执行步数: {len(all_results)}"
            )

            return state

        except Exception as e:
            logger.error(f"【Plan-and-Execute】异常，降级为 ReAct 模式: {e}")
            return self.react_agent.execute(user_query)
