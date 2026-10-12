import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.tools import StructuredTool

from eval.cases.fakes import FakeTool
from trip_planner.harness import (
    GuardedTool,
    HarnessPolicy,
    ToolPolicyError,
    TravelAgentHarness,
    prepare_agent_tools,
)
from trip_planner.trip_planner_agent import MultiAgentTripPlanner


class HarnessPolicyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="harness-policy-")
        self.addCleanup(self.temp_dir.cleanup)

    def harness(self, policy=None, child="run"):
        return TravelAgentHarness(
            policy=policy,
            log_dir=Path(self.temp_dir.name) / child,
        )

    async def test_agent_tool_allowlists_filter_before_agent_creation(self):
        async def invoke(city: str = "") -> str:
            return city

        tools = [
            StructuredTool.from_function(
                coroutine=invoke, name=name, description=name
            )
            for name in (
                "maps_weather",
                "maps_text_search",
                "maps_search_detail",
                "maps_around_search",
                "maps_direction_driving",
            )
        ]
        harness = self.harness()

        weather = prepare_agent_tools(tools, "weather", harness)
        attraction = prepare_agent_tools(tools, "attraction", harness)
        hotel = prepare_agent_tools(tools, "hotel", harness)
        planner = prepare_agent_tools(tools, "planner", harness)

        self.assertEqual([tool.name for tool in weather], ["maps_weather"])
        expected_search = {
            "maps_text_search",
            "maps_search_detail",
            "maps_around_search",
        }
        self.assertEqual({tool.name for tool in attraction}, expected_search)
        self.assertEqual({tool.name for tool in hotel}, expected_search)
        self.assertEqual(planner, [])

    async def test_initialize_passes_only_role_tools_to_each_agent(self):
        async def invoke(city: str = "") -> str:
            return city

        tools = [
            StructuredTool.from_function(
                coroutine=invoke, name=name, description=name
            )
            for name in (
                "maps_weather",
                "maps_text_search",
                "maps_search_detail",
                "maps_around_search",
                "maps_direction_driving",
            )
        ]

        class FakeClient:
            async def get_tools(self):
                return tools

        class FakeCreatedAgent:
            def with_config(self, config):
                return self

        agent_tool_names = []

        def fake_create_agent(model, tools=None, system_prompt=None):
            agent_tool_names.append([tool.name for tool in (tools or [])])
            return FakeCreatedAgent()

        planner = MultiAgentTripPlanner(
            llm=object(),
            amap_api_key="fake-key",
            harness=self.harness(child="initialize"),
        )
        with (
            patch(
                "trip_planner.trip_planner_agent.MultiServerMCPClient",
                return_value=FakeClient(),
            ),
            patch(
                "trip_planner.trip_planner_agent.create_agent",
                side_effect=fake_create_agent,
            ),
            patch(
                "socket.socket.connect",
                side_effect=AssertionError("network forbidden"),
            ),
        ):
            await planner.initialize()

        self.assertEqual(agent_tool_names[0], ["maps_weather"])
        self.assertEqual(
            set(agent_tool_names[1]),
            {"maps_text_search", "maps_search_detail", "maps_around_search"},
        )
        self.assertEqual(
            set(agent_tool_names[2]),
            {"maps_text_search", "maps_search_detail", "maps_around_search"},
        )
        self.assertEqual(agent_tool_names[3], [])

    async def test_guarded_allowed_tool_executes_once(self):
        calls = 0

        async def weather(city: str) -> str:
            nonlocal calls
            calls += 1
            return f"weather:{city}"

        original = StructuredTool.from_function(
            coroutine=weather,
            name="maps_weather",
            description="weather",
        )
        harness = self.harness()
        guarded = GuardedTool(
            wrapped_tool=original,
            harness=harness,
            agent_role="weather",
        )
        context, token = harness.start_run()
        try:
            result = await guarded.ainvoke({"city": "测试城市"})
            summary = harness.finish_run(
                context, success=True, final_status="passed"
            )
        finally:
            harness.reset_run(token)

        self.assertEqual(result, "weather:测试城市")
        self.assertEqual(calls, 1)
        self.assertEqual(summary.total_tool_calls, 1)

    async def test_guard_preserves_content_and_artifact_tool_contract(self):
        async def weather(city: str):
            return f"weather:{city}", {"source": "fake"}

        original = StructuredTool.from_function(
            coroutine=weather,
            name="maps_weather",
            description="weather",
            response_format="content_and_artifact",
        )
        harness = self.harness(child="artifact")
        guarded = GuardedTool(
            wrapped_tool=original,
            harness=harness,
            agent_role="weather",
        )
        context, token = harness.start_run()
        try:
            result = await guarded.ainvoke(
                {
                    "type": "tool_call",
                    "name": "maps_weather",
                    "args": {"city": "测试城市"},
                    "id": "fake-tool-call",
                }
            )
            harness.finish_run(context, success=True, final_status="passed")
        finally:
            harness.reset_run(token)

        self.assertEqual(result.content, "weather:测试城市")
        self.assertEqual(result.artifact, {"source": "fake"})

    async def test_unauthorized_tool_is_blocked_before_side_effect(self):
        calls = 0

        async def driving(origin: str = "") -> str:
            nonlocal calls
            calls += 1
            return origin

        original = StructuredTool.from_function(
            coroutine=driving,
            name="maps_direction_driving",
            description="driving",
        )
        harness = self.harness()
        guarded = GuardedTool(
            wrapped_tool=original,
            harness=harness,
            agent_role="weather",
        )
        context, token = harness.start_run()
        try:
            result = await guarded.ainvoke({"origin": "A"})
            harness.finish_run(context, success=True, final_status="passed")
        finally:
            harness.reset_run(token)

        self.assertEqual(calls, 0)
        self.assertIn("tool_not_allowed", result)
        self.assertEqual(context.total_tool_calls, 0)
        self.assertIn("tool_call_blocked", {event.event_type for event in context.events})

    async def test_weather_budget_allows_two_and_blocks_third(self):
        harness = self.harness(HarnessPolicy(max_weather_tool_calls=2))
        tool = FakeTool("maps_weather")
        context, token = harness.start_run()
        try:
            for _ in range(2):
                await harness.call_tool(
                    "weather", tool.name, lambda: tool.ainvoke({})
                )
            with self.assertRaises(ToolPolicyError):
                await harness.call_tool(
                    "weather", tool.name, lambda: tool.ainvoke({})
                )
            harness.finish_run(context, success=True, final_status="passed")
        finally:
            harness.reset_run(token)

        self.assertEqual(tool.invocation_count, 2)
        self.assertEqual(context.total_tool_calls, 2)

    async def test_total_budget_blocks_fourth_cross_agent_call(self):
        harness = self.harness(HarnessPolicy(max_total_tool_calls=3))
        attraction = FakeTool("maps_text_search")
        weather = FakeTool("maps_weather")
        hotel = FakeTool("maps_text_search")
        context, token = harness.start_run()
        try:
            await harness.call_tool(
                "attraction", attraction.name, lambda: attraction.ainvoke({})
            )
            await harness.call_tool(
                "weather", weather.name, lambda: weather.ainvoke({})
            )
            await harness.call_tool("hotel", hotel.name, lambda: hotel.ainvoke({}))
            with self.assertRaises(ToolPolicyError):
                await harness.call_tool(
                    "attraction", attraction.name, lambda: attraction.ainvoke({})
                )
            harness.finish_run(context, success=True, final_status="passed")
        finally:
            harness.reset_run(token)

        self.assertEqual(context.total_tool_calls, 3)
        self.assertEqual(
            attraction.invocation_count
            + weather.invocation_count
            + hotel.invocation_count,
            3,
        )
