"""Least-privilege tool filtering and pre-invocation runtime guarding."""

from __future__ import annotations

import inspect
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from pydantic import ConfigDict, Field

from trip_planner.harness.core import TravelAgentHarness


class ToolPolicyError(ToolException):
    """Raised before a tool side effect when allowlist or budget policy fails."""


class GuardedTool(BaseTool):
    """A BaseTool proxy that preserves schema while guarding the raw execution."""

    wrapped_tool: BaseTool = Field(exclude=True)
    harness: TravelAgentHarness = Field(exclude=True)
    agent_role: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        *,
        wrapped_tool: BaseTool,
        harness: TravelAgentHarness,
        agent_role: str,
    ):
        super().__init__(
            name=wrapped_tool.name,
            description=wrapped_tool.description,
            args_schema=wrapped_tool.args_schema,
            return_direct=wrapped_tool.return_direct,
            response_format=wrapped_tool.response_format,
            metadata=wrapped_tool.metadata,
            handle_tool_error=wrapped_tool.handle_tool_error or True,
            wrapped_tool=wrapped_tool,
            harness=harness,
            agent_role=agent_role,
        )

    async def _arun(
        self,
        *args: Any,
        config: RunnableConfig | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        async def invoke_original() -> Any:
            method = self.wrapped_tool._arun
            parameters = inspect.signature(method).parameters
            delegated_kwargs = dict(kwargs)
            if "config" in parameters:
                delegated_kwargs["config"] = config or {}
            if "run_manager" in parameters:
                delegated_kwargs["run_manager"] = run_manager
            return await method(*args, **delegated_kwargs)

        return await self.harness.call_tool(
            self.agent_role,
            self.name,
            invoke_original,
        )

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Guarded MCP tools must be invoked asynchronously")


def prepare_agent_tools(
    tools: list[BaseTool],
    agent_role: str,
    harness: TravelAgentHarness,
) -> list[GuardedTool]:
    """Filter first, then wrap only the tools visible to this agent role."""
    return [
        GuardedTool(wrapped_tool=tool, harness=harness, agent_role=agent_role)
        for tool in tools
        if harness.is_tool_allowed(agent_role, tool.name)
    ]
