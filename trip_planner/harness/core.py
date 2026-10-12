"""Request-scoped policy, tracing, metrics and lifecycle infrastructure."""

from __future__ import annotations

import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, TypeVar

from langchain_core.callbacks import AsyncCallbackHandler

from trip_planner.harness.tracing import JsonlTraceWriter, TraceEvent


DEFAULT_TOOL_ALLOWLISTS: dict[str, frozenset[str]] = {
    "weather": frozenset({"maps_weather"}),
    "attraction": frozenset(
        {"maps_text_search", "maps_search_detail", "maps_around_search"}
    ),
    "hotel": frozenset(
        {"maps_text_search", "maps_search_detail", "maps_around_search"}
    ),
    "planner": frozenset(),
}


@dataclass(frozen=True)
class HarnessPolicy:
    """Small policy surface: least-privilege tools and per-run budgets."""

    max_total_tool_calls: int = 10
    max_attraction_tool_calls: int = 2
    max_weather_tool_calls: int = 2
    max_hotel_tool_calls: int = 2
    tool_allowlists: Mapping[str, frozenset[str]] = field(
        default_factory=lambda: dict(DEFAULT_TOOL_ALLOWLISTS)
    )

    def __post_init__(self) -> None:
        values = (
            self.max_total_tool_calls,
            self.max_attraction_tool_calls,
            self.max_weather_tool_calls,
            self.max_hotel_tool_calls,
        )
        if any(value < 0 for value in values):
            raise ValueError("Tool call budgets must be non-negative")

    def role_budget(self, agent_role: str) -> int:
        budgets = {
            "attraction": self.max_attraction_tool_calls,
            "weather": self.max_weather_tool_calls,
            "hotel": self.max_hotel_tool_calls,
        }
        return budgets.get(agent_role, 0)


@dataclass
class RunContext:
    """All mutable counters belong to one request, never to global state."""

    run_id: str
    started_at: float
    total_tool_calls: int = 0
    tool_calls_by_agent: dict[str, int] = field(default_factory=dict)
    tool_calls_by_tool: dict[str, int] = field(default_factory=dict)
    agent_invocations_by_agent: dict[str, int] = field(default_factory=dict)
    llm_calls: int = 0
    planner_replans: int = 0
    repair_count: int = 0
    events: list[TraceEvent] = field(default_factory=list)
    model_started_at: dict[str, float] = field(default_factory=dict, repr=False)
    model_run_ids: set[str] = field(default_factory=set, repr=False)


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    success: bool
    total_latency_ms: float
    total_tool_calls: int
    tool_calls_by_agent: dict[str, int]
    tool_calls_by_tool: dict[str, int]
    agent_invocations: int
    agent_invocations_by_agent: dict[str, int]
    llm_calls: int
    planner_replans: int
    repair_count: int
    validation_passed: bool
    final_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


class HarnessModelCallback(AsyncCallbackHandler):
    """Count actual LangChain model runs, including agent tool-loop turns."""

    def __init__(self, harness: TravelAgentHarness):
        self.harness = harness

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self.harness.record_model_start(str(run_id))

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self.harness.record_model_start(str(run_id))

    async def on_llm_end(self, response: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
        self.harness.record_model_end(str(run_id), "success")

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self.harness.record_model_end(str(run_id), "error")


class TravelAgentHarness:
    """Infrastructure layer around the existing graph, agents and tools."""

    _SENSITIVE_PARTS = (
        "api_key",
        "authorization",
        "secret",
        "access_token",
        "headers",
        "prompt",
        "response",
    )

    def __init__(
        self,
        policy: HarnessPolicy | None = None,
        log_dir: str | Path = "logs/runs",
    ):
        self.policy = policy or HarnessPolicy()
        self.trace_writer = JsonlTraceWriter(log_dir)
        # ContextVar 会随 asyncio 任务传播，但不同请求拥有各自的值。
        self._active_run: ContextVar[RunContext | None] = ContextVar(
            f"travel_agent_run_{id(self)}",
            default=None,
        )
        self.model_callback = HarnessModelCallback(self)

    @staticmethod
    def canonical_tool_name(tool_name: str) -> str:
        return tool_name.removeprefix("amap_")

    def is_tool_allowed(self, agent_role: str, tool_name: str) -> bool:
        allowed = self.policy.tool_allowlists.get(agent_role, frozenset())
        return self.canonical_tool_name(tool_name) in allowed

    def current_context(self, *, required: bool = False) -> RunContext | None:
        context = self._active_run.get()
        if required and context is None:
            raise RuntimeError("Tool execution requires an active Harness RunContext")
        return context

    def start_run(self) -> tuple[RunContext, Token[RunContext | None]]:
        context = RunContext(
            run_id=f"trip_{uuid.uuid4().hex}",
            started_at=time.perf_counter(),
        )
        token = self._active_run.set(context)
        self.record("harness", "run_start", "started")
        return context, token

    def reset_run(self, token: Token[RunContext | None]) -> None:
        self._active_run.reset(token)

    async def run_workflow(
        self,
        workflow: Any,
        initial_state: dict[str, Any],
    ) -> tuple[dict[str, Any], RunSummary]:
        """Execute one graph invocation with an isolated runtime context."""
        context, token = self.start_run()
        try:
            result = await workflow.ainvoke(initial_state)
            final_status = str(result.get("validation_status", "unknown"))
            summary = self.finish_run(
                context,
                success=final_status in {"passed", "degraded"},
                final_status=final_status,
            )
            return result, summary
        except BaseException as exc:
            self.finish_run(
                context,
                success=False,
                final_status="error",
                metadata={"error_type": type(exc).__name__},
            )
            raise
        finally:
            self.reset_run(token)

    def finish_run(
        self,
        context: RunContext,
        *,
        success: bool,
        final_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> RunSummary:
        total_latency_ms = self._elapsed_ms(context.started_at)
        self.record(
            "harness",
            "run_end",
            "success" if success else "error",
            latency_ms=total_latency_ms,
            metadata=metadata,
        )
        summary = RunSummary(
            run_id=context.run_id,
            success=success,
            total_latency_ms=total_latency_ms,
            total_tool_calls=context.total_tool_calls,
            tool_calls_by_agent=dict(context.tool_calls_by_agent),
            tool_calls_by_tool=dict(context.tool_calls_by_tool),
            agent_invocations=sum(context.agent_invocations_by_agent.values()),
            agent_invocations_by_agent=dict(context.agent_invocations_by_agent),
            llm_calls=context.llm_calls,
            planner_replans=context.planner_replans,
            repair_count=context.repair_count,
            validation_passed=final_status == "passed",
            final_status=final_status,
        )
        self.record(
            "harness",
            "run_summary",
            "success" if success else "error",
            metadata=summary.to_dict(),
        )
        return summary

    async def run_traced(
        self,
        component: str,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        if self.current_context() is None:
            return await func(*args, **kwargs)
        self.record(component, "node_start", "started")
        started_at = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
        except BaseException as exc:
            self.record(
                component,
                "node_error",
                "error",
                latency_ms=self._elapsed_ms(started_at),
                metadata={"error_type": type(exc).__name__},
            )
            raise
        self.record(
            component,
            "node_end",
            "success",
            latency_ms=self._elapsed_ms(started_at),
        )
        return result

    def traced_node(
        self,
        component: str,
        func: Callable[..., Awaitable[dict[str, Any]]],
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        async def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return await self.run_traced(component, func, *args, **kwargs)

        return wrapped

    async def invoke_agent(self, agent_role: str, agent: Any, value: Any) -> Any:
        context = self.current_context()
        if context is None:
            return await agent.ainvoke(value)
        self._increment(context.agent_invocations_by_agent, agent_role)
        started_at = time.perf_counter()
        try:
            result = await agent.ainvoke(value)
        except BaseException as exc:
            self.record(
                agent_role,
                "agent_invocation",
                "error",
                latency_ms=self._elapsed_ms(started_at),
                metadata={"agent": agent_role, "error_type": type(exc).__name__},
            )
            raise
        self.record(
            agent_role,
            "agent_invocation",
            "success",
            latency_ms=self._elapsed_ms(started_at),
            metadata={"agent": agent_role},
        )
        return result

    def instrument_agent(self, agent: Any) -> Any:
        """Attach the callback that observes each actual LangChain model run."""
        return agent.with_config({"callbacks": [self.model_callback]})

    async def call_tool(
        self,
        agent_role: str,
        tool_name: str,
        invoke: Callable[[], Awaitable[T]],
    ) -> T:
        """Reserve permission/budget before the delegated side effect begins."""
        context = self.current_context(required=True)
        assert context is not None
        canonical_name = self.canonical_tool_name(tool_name)

        block_reason: str | None = None
        if not self.is_tool_allowed(agent_role, tool_name):
            block_reason = "tool_not_allowed"
        elif context.total_tool_calls >= self.policy.max_total_tool_calls:
            block_reason = "total_tool_budget_exceeded"
        elif (
            context.tool_calls_by_agent.get(agent_role, 0)
            >= self.policy.role_budget(agent_role)
        ):
            block_reason = "agent_tool_budget_exceeded"

        if block_reason:
            self.record(
                tool_name,
                "tool_call_blocked",
                "blocked",
                metadata={
                    "tool": tool_name,
                    "agent": agent_role,
                    "reason": block_reason,
                },
            )
            from trip_planner.harness.tool_guard import ToolPolicyError

            raise ToolPolicyError(
                f"Tool call blocked ({block_reason}): {agent_role}/{tool_name}"
            )

        # 检查与占用之间没有 await，避免同一 Run 的并发调用重复消费额度。
        context.total_tool_calls += 1
        self._increment(context.tool_calls_by_agent, agent_role)
        self._increment(context.tool_calls_by_tool, canonical_name)
        self.record(
            tool_name,
            "tool_call_start",
            "started",
            metadata={"tool": tool_name, "agent": agent_role},
        )
        started_at = time.perf_counter()
        try:
            result = await invoke()
        except BaseException as exc:
            self.record(
                tool_name,
                "tool_call_end",
                "error",
                latency_ms=self._elapsed_ms(started_at),
                metadata={
                    "tool": tool_name,
                    "agent": agent_role,
                    "error_type": type(exc).__name__,
                },
            )
            raise
        self.record(
            tool_name,
            "tool_call_end",
            "error" if self._tool_result_is_error(result) else "success",
            latency_ms=self._elapsed_ms(started_at),
            metadata={"tool": tool_name, "agent": agent_role},
        )
        return result

    def record_validation(self, status: str, issue_codes: list[str]) -> None:
        self.record(
            "validator",
            "validation_result",
            status,
            metadata={"issue_codes": issue_codes},
        )

    def record_route(self, route: str) -> None:
        self.record("router", "route", "selected", metadata={"route": route})

    def record_repair(self, source: str) -> None:
        context = self.current_context()
        if context is None:
            return
        context.repair_count += 1
        self.record(
            f"repair_{source}",
            "repair",
            "started",
            metadata={"source": source},
        )

    def record_replan(self) -> None:
        context = self.current_context()
        if context is None:
            return
        context.planner_replans += 1
        self.record("replan", "replan", "started")

    def record_model_start(self, model_run_id: str) -> None:
        context = self.current_context()
        if context is None or model_run_id in context.model_run_ids:
            return
        context.model_run_ids.add(model_run_id)
        context.model_started_at[model_run_id] = time.perf_counter()
        context.llm_calls += 1
        self.record("llm", "llm_call_start", "started")

    def record_model_end(self, model_run_id: str, status: str) -> None:
        context = self.current_context()
        if context is None:
            return
        started_at = context.model_started_at.pop(model_run_id, None)
        self.record(
            "llm",
            "llm_call_end",
            status,
            latency_ms=self._elapsed_ms(started_at) if started_at else None,
        )

    def record(
        self,
        component: str,
        event_type: str,
        status: str,
        *,
        latency_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        context = self.current_context()
        if context is None:
            return
        event = TraceEvent(
            run_id=context.run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            component=component,
            event_type=event_type,
            status=status,
            latency_ms=latency_ms,
            metadata=self._sanitize_metadata(metadata or {}),
        )
        context.events.append(event)
        self.trace_writer.append(event)

    @classmethod
    def _sanitize_metadata(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            sanitized: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                if any(part in key_text.lower() for part in cls._SENSITIVE_PARTS):
                    sanitized[key_text] = "[REDACTED]"
                else:
                    sanitized[key_text] = cls._sanitize_metadata(item)
            return sanitized
        if isinstance(value, (list, tuple, set)):
            return [cls._sanitize_metadata(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @staticmethod
    def _increment(values: dict[str, int], key: str) -> None:
        values[key] = values.get(key, 0) + 1

    @staticmethod
    def _tool_result_is_error(result: Any) -> bool:
        candidates = result if isinstance(result, tuple) else (result,)
        return any(getattr(item, "status", None) == "error" for item in candidates)

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return max(0.0, (time.perf_counter() - started_at) * 1000)
