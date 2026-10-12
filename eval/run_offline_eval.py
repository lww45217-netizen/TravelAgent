"""Nine-case offline regression evaluation for V2 plus the runtime harness."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from eval.cases.fakes import (
    FakeAgent,
    FakeTool,
    attraction_json,
    hotel_json,
    make_request,
    make_valid_plan,
    weather_json,
)
from trip_planner.harness import HarnessPolicy, ToolPolicyError, TravelAgentHarness
from trip_planner.trip_planner_agent import MultiAgentTripPlanner


async def _call(harness, role: str, tool: FakeTool):
    return await harness.call_tool(role, tool.name, lambda: tool.ainvoke({}))


async def _normal_workflow(log_dir: Path):
    harness = TravelAgentHarness(log_dir=log_dir)
    planner = MultiAgentTripPlanner(
        llm=object(), amap_api_key="fake-key", harness=harness
    )
    planner.attraction_agent = FakeAgent(attraction_json())
    planner.weather_agent = FakeAgent(weather_json())
    planner.hotel_agent = FakeAgent(hotel_json())
    planner.planner_agent = FakeAgent(make_valid_plan().model_dump_json())
    planner.workflow = planner._build_workflow()
    result, summary = await harness.run_workflow(
        planner.workflow, {"request": make_request()}
    )
    return harness, result, summary


async def _policy_cases(root: Path) -> dict[str, bool]:
    results: dict[str, bool] = {}

    allowed_harness = TravelAgentHarness(log_dir=root / "allow")
    allowed = FakeTool("maps_weather")
    context, token = allowed_harness.start_run()
    try:
        await _call(allowed_harness, "weather", allowed)
        results["case_2_weather_allowed"] = allowed.invocation_count == 1
        allowed_harness.finish_run(context, success=True, final_status="passed")
    finally:
        allowed_harness.reset_run(token)

    denied_harness = TravelAgentHarness(log_dir=root / "deny")
    denied = FakeTool("maps_direction_driving")
    context, token = denied_harness.start_run()
    try:
        try:
            await _call(denied_harness, "weather", denied)
        except ToolPolicyError:
            pass
        results["case_3_weather_unauthorized"] = denied.invocation_count == 0
        denied_harness.finish_run(context, success=True, final_status="passed")
    finally:
        denied_harness.reset_run(token)

    role_harness = TravelAgentHarness(
        HarnessPolicy(max_weather_tool_calls=2), log_dir=root / "role_budget"
    )
    weather = FakeTool("maps_weather")
    context, token = role_harness.start_run()
    try:
        await _call(role_harness, "weather", weather)
        await _call(role_harness, "weather", weather)
        try:
            await _call(role_harness, "weather", weather)
        except ToolPolicyError:
            pass
        results["case_4_weather_budget"] = (
            weather.invocation_count == 2 and context.total_tool_calls == 2
        )
        role_harness.finish_run(context, success=True, final_status="passed")
    finally:
        role_harness.reset_run(token)

    total_harness = TravelAgentHarness(
        HarnessPolicy(max_total_tool_calls=3), log_dir=root / "total_budget"
    )
    attraction = FakeTool("maps_text_search")
    weather = FakeTool("maps_weather")
    hotel = FakeTool("maps_text_search")
    context, token = total_harness.start_run()
    try:
        await _call(total_harness, "attraction", attraction)
        await _call(total_harness, "weather", weather)
        await _call(total_harness, "hotel", hotel)
        try:
            await _call(total_harness, "attraction", attraction)
        except ToolPolicyError:
            pass
        results["case_5_total_budget"] = (
            context.total_tool_calls == 3
            and attraction.invocation_count
            + weather.invocation_count
            + hotel.invocation_count
            == 3
        )
        total_harness.finish_run(context, success=True, final_status="passed")
    finally:
        total_harness.reset_run(token)

    return results


async def _summary_case(root: Path) -> bool:
    harness = TravelAgentHarness(log_dir=root / "summary")
    tools = {
        "attraction": FakeTool("maps_text_search"),
        "weather": FakeTool("maps_weather"),
        "hotel": FakeTool("maps_text_search"),
    }
    context, token = harness.start_run()
    try:
        await _call(harness, "attraction", tools["attraction"])
        await _call(harness, "weather", tools["weather"])
        await _call(harness, "weather", tools["weather"])
        await _call(harness, "hotel", tools["hotel"])
        harness.record_replan()
        harness.record_repair("weather")
        summary = harness.finish_run(context, success=True, final_status="passed")
    finally:
        harness.reset_run(token)
    return (
        summary.total_tool_calls == 4
        and summary.tool_calls_by_agent.get("weather") == 2
        and summary.planner_replans == 1
        and summary.repair_count == 1
    )


async def _v2_regression_case(root: Path) -> bool:
    harness = TravelAgentHarness(log_dir=root / "v2_regression")
    planner = MultiAgentTripPlanner(
        llm=object(), amap_api_key="fake-key", harness=harness
    )
    planner.attraction_agent = FakeAgent(attraction_json())
    planner.weather_agent = FakeAgent(
        weather_json(complete=False), weather_json(complete=True)
    )
    planner.hotel_agent = FakeAgent(hotel_json())
    plan_json = make_valid_plan().model_dump_json()
    planner.planner_agent = FakeAgent(plan_json, plan_json)
    planner.workflow = planner._build_workflow()

    result, summary = await harness.run_workflow(
        planner.workflow, {"request": make_request()}
    )
    trace_rows = [
        json.loads(line)
        for line in harness.trace_writer.path_for(summary.run_id).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    route_values = [
        row["metadata"].get("route")
        for row in trace_rows
        if row["event_type"] == "route"
    ]
    return (
        result["validation_status"] == "passed"
        and result["weather_retry_count"] == 1
        and "repair_weather" in route_values
        and summary.repair_count == 1
        and summary.planner_replans == 1
    )


async def _isolation_case(root: Path) -> bool:
    harness = TravelAgentHarness(log_dir=root / "isolation")
    tool = FakeTool("maps_weather")

    class FakeWorkflow:
        def __init__(self, calls: int):
            self.calls = calls

        async def ainvoke(self, state):
            for _ in range(self.calls):
                await _call(harness, "weather", tool)
                await asyncio.sleep(0)
            return {"validation_status": "passed"}

    (_, summary_a), (_, summary_b) = await asyncio.gather(
        harness.run_workflow(FakeWorkflow(2), {}),
        harness.run_workflow(FakeWorkflow(1), {}),
    )
    return (
        summary_a.run_id != summary_b.run_id
        and summary_a.total_tool_calls == 2
        and summary_b.total_tool_calls == 1
    )


async def run_offline_eval() -> dict:
    """Run all cases with hard network/MCP/real-Agent guards."""
    with tempfile.TemporaryDirectory(prefix="trip-harness-eval-") as temp_dir:
        root = Path(temp_dir)
        with (
            patch(
                "socket.socket.connect",
                side_effect=AssertionError("network forbidden in offline eval"),
            ),
            patch(
                "trip_planner.trip_planner_agent.MultiServerMCPClient.get_tools",
                side_effect=AssertionError("real MCP forbidden in offline eval"),
            ),
            patch(
                "trip_planner.trip_planner_agent.create_agent",
                side_effect=AssertionError("real Agent creation forbidden in offline eval"),
            ),
        ):
            harness, normal_result, normal_summary = await _normal_workflow(
                root / "normal"
            )
            normal_events = [
                json.loads(line)
                for line in harness.trace_writer.path_for(
                    normal_summary.run_id
                ).read_text(encoding="utf-8").splitlines()
            ]
            event_types = {event["event_type"] for event in normal_events}
            node_latencies = [
                event["latency_ms"]
                for event in normal_events
                if event["event_type"] == "node_end"
            ]
            policy_cases = await _policy_cases(root)
            cases = {
                "case_1_normal_workflow": (
                    normal_result["validation_status"] == "passed"
                    and normal_summary.success
                ),
                "case_2_weather_allowed": policy_cases[
                    "case_2_weather_allowed"
                ],
                "case_3_weather_unauthorized": policy_cases[
                    "case_3_weather_unauthorized"
                ],
                "case_4_weather_budget": policy_cases["case_4_weather_budget"],
                "case_5_total_budget": policy_cases["case_5_total_budget"],
                "case_6_trace": (
                    {"run_start", "node_start", "node_end", "run_end"}
                    <= event_types
                    and node_latencies
                    and all(value is not None and value >= 0 for value in node_latencies)
                ),
                "case_7_summary": await _summary_case(root),
                "case_8_v2_regression": await _v2_regression_case(root),
                "case_9_concurrent_isolation": await _isolation_case(root),
            }

    def rate(names: list[str]) -> float:
        return sum(bool(cases[name]) for name in names) / len(names)

    metrics = {
        "business_pass_rate": rate(["case_1_normal_workflow"]),
        "routing_accuracy": rate(
            ["case_1_normal_workflow", "case_8_v2_regression"]
        ),
        "tool_policy_pass_rate": rate(
            ["case_2_weather_allowed", "case_3_weather_unauthorized"]
        ),
        "tool_budget_compliance": rate(
            ["case_4_weather_budget", "case_5_total_budget"]
        ),
        "trace_completeness": rate(["case_6_trace"]),
        "summary_consistency": rate(["case_7_summary"]),
        "retry_limit_compliance": rate(["case_8_v2_regression"]),
    }
    return {"cases": cases, "metrics": metrics, "all_passed": all(cases.values())}


def main() -> None:
    print(json.dumps(asyncio.run(run_offline_eval()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
