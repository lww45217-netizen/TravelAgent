import asyncio
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from eval.cases.fakes import (
    FakeAgent,
    FakeTool,
    attraction_json,
    hotel_json,
    make_request,
    make_valid_plan,
    weather_json,
)
from trip_planner.harness import TravelAgentHarness
from trip_planner.schemas import TripPlan
from trip_planner.trip_planner_agent import MultiAgentTripPlanner


class HarnessSummaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="harness-summary-")
        self.addCleanup(self.temp_dir.cleanup)

    def harness(self, child="run"):
        return TravelAgentHarness(log_dir=Path(self.temp_dir.name) / child)

    async def test_summary_counts_tools_agents_replan_and_repair(self):
        harness = self.harness()
        tools = {
            "attraction": FakeTool("maps_text_search"),
            "weather": FakeTool("maps_weather"),
            "hotel": FakeTool("maps_text_search"),
        }
        fake_agent = FakeAgent("ok")
        context, token = harness.start_run()
        try:
            await harness.call_tool(
                "attraction",
                tools["attraction"].name,
                lambda: tools["attraction"].ainvoke({}),
            )
            for _ in range(2):
                await harness.call_tool(
                    "weather",
                    tools["weather"].name,
                    lambda: tools["weather"].ainvoke({}),
                )
            await harness.call_tool(
                "hotel", tools["hotel"].name, lambda: tools["hotel"].ainvoke({})
            )
            await harness.invoke_agent("planner", fake_agent, {})
            harness.record_replan()
            harness.record_repair("weather")
            summary = harness.finish_run(
                context, success=True, final_status="passed"
            )
        finally:
            harness.reset_run(token)

        self.assertEqual(summary.total_tool_calls, 4)
        self.assertEqual(summary.tool_calls_by_agent["weather"], 2)
        self.assertEqual(summary.agent_invocations, 1)
        self.assertEqual(summary.planner_replans, 1)
        self.assertEqual(summary.repair_count, 1)
        self.assertTrue(summary.validation_passed)

    async def test_model_callback_counts_real_model_run_ids_without_double_counting(self):
        harness = self.harness("model")
        context, token = harness.start_run()
        run_id = uuid.uuid4()
        try:
            await harness.model_callback.on_chat_model_start(
                {}, [[]], run_id=run_id
            )
            await harness.model_callback.on_llm_start(
                {}, ["ignored fallback"], run_id=run_id
            )
            await harness.model_callback.on_llm_end(None, run_id=run_id)
            summary = harness.finish_run(
                context, success=True, final_status="passed"
            )
        finally:
            harness.reset_run(token)

        self.assertEqual(summary.llm_calls, 1)

    async def test_instrumented_fake_chat_model_emits_real_llm_callback(self):
        harness = self.harness("fake-model")
        model = harness.instrument_agent(FakeListChatModel(responses=["first", "second"]))
        context, token = harness.start_run()
        try:
            await harness.invoke_agent("planner", model, "hello")
            await harness.invoke_agent("planner", model, "again")
            summary = harness.finish_run(
                context, success=True, final_status="passed"
            )
        finally:
            harness.reset_run(token)

        self.assertEqual(summary.agent_invocations, 2)
        self.assertEqual(summary.llm_calls, 2)

    async def test_concurrent_runs_have_independent_contexts(self):
        harness = self.harness("concurrent")
        tool = FakeTool("maps_weather")

        class FakeWorkflow:
            def __init__(self, calls):
                self.calls = calls

            async def ainvoke(self, state):
                for _ in range(self.calls):
                    await harness.call_tool(
                        "weather", tool.name, lambda: tool.ainvoke({})
                    )
                    await asyncio.sleep(0)
                return {"validation_status": "passed"}

        (_, summary_a), (_, summary_b) = await asyncio.gather(
            harness.run_workflow(FakeWorkflow(2), {}),
            harness.run_workflow(FakeWorkflow(1), {}),
        )

        self.assertNotEqual(summary_a.run_id, summary_b.run_id)
        self.assertEqual(summary_a.total_tool_calls, 2)
        self.assertEqual(summary_b.total_tool_calls, 1)
        self.assertTrue(harness.trace_writer.path_for(summary_a.run_id).exists())
        self.assertTrue(harness.trace_writer.path_for(summary_b.run_id).exists())

    async def test_concurrent_compiled_langgraph_runs_remain_isolated(self):
        harness = self.harness("concurrent-graph")
        planner = MultiAgentTripPlanner(
            llm=object(), amap_api_key="fake-key", harness=harness
        )
        planner.attraction_agent = FakeAgent(attraction_json())
        planner.weather_agent = FakeAgent(weather_json())
        planner.hotel_agent = FakeAgent(hotel_json())
        planner.planner_agent = FakeAgent(make_valid_plan().model_dump_json())
        planner.workflow = planner._build_workflow()

        (result_a, summary_a), (result_b, summary_b) = await asyncio.gather(
            harness.run_workflow(planner.workflow, {"request": make_request()}),
            harness.run_workflow(planner.workflow, {"request": make_request()}),
        )

        self.assertEqual(result_a["validation_status"], "passed")
        self.assertEqual(result_b["validation_status"], "passed")
        self.assertNotEqual(summary_a.run_id, summary_b.run_id)
        self.assertEqual(summary_a.agent_invocations, 4)
        self.assertEqual(summary_b.agent_invocations, 4)
        self.assertNotEqual(
            harness.trace_writer.path_for(summary_a.run_id),
            harness.trace_writer.path_for(summary_b.run_id),
        )

    async def test_plan_trip_still_returns_trip_plan(self):
        harness = self.harness("api")
        planner = MultiAgentTripPlanner(
            llm=object(), amap_api_key="fake-key", harness=harness
        )
        planner.attraction_agent = FakeAgent(attraction_json())
        planner.weather_agent = FakeAgent(weather_json())
        planner.hotel_agent = FakeAgent(hotel_json())
        planner.planner_agent = FakeAgent(make_valid_plan().model_dump_json())
        planner.workflow = planner._build_workflow()

        with (
            patch(
                "socket.socket.connect",
                side_effect=AssertionError("network forbidden"),
            ),
            patch(
                "trip_planner.trip_planner_agent.MultiServerMCPClient.get_tools",
                side_effect=AssertionError("real MCP forbidden"),
            ),
        ):
            result = await planner.plan_trip(make_request())

        self.assertIsInstance(result, TripPlan)
        self.assertEqual(result.city, "测试城市")
