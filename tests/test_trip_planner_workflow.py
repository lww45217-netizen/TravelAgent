import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from trip_planner.schemas import (
    Attraction,
    AttractionEvidence,
    AttractionEvidenceItem,
    Budget,
    DayPlan,
    Hotel,
    HotelEvidence,
    Location,
    Meal,
    TripPlan,
    TripRequest,
    WeatherEvidence,
    WeatherInfo,
)
from trip_planner.trip_planner_agent import MultiAgentTripPlanner


class FakeAgent:
    """Deterministic agent double; it never imports a client or touches network."""

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls = 0
        self.inputs = []

    async def ainvoke(self, value):
        self.inputs.append(value)
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]


def make_request() -> TripRequest:
    return TripRequest(
        city="测试城市",
        start_date="2026-10-03",
        end_date="2026-10-04",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化", "美食"],
    )


def make_attraction(name: str, longitude: float) -> Attraction:
    return Attraction(
        name=name,
        address=f"{name}地址",
        location=Location(longitude=longitude, latitude=45.0),
        visit_duration=120,
        description=f"{name}介绍",
        ticket_price=10,
    )


def make_meals() -> list[Meal]:
    return [
        Meal(type="breakfast", name="早餐", estimated_cost=20),
        Meal(type="lunch", name="午餐", estimated_cost=40),
        Meal(type="dinner", name="晚餐", estimated_cost=60),
    ]


def make_valid_plan() -> TripPlan:
    hotel = Hotel(name="H1", address="H1地址", estimated_cost=200)
    return TripPlan(
        city="测试城市",
        start_date="2026-10-03",
        end_date="2026-10-04",
        days=[
            DayPlan(
                date="2026-10-03",
                day_index=0,
                description="day 1",
                transportation="公交",
                accommodation="经济型酒店",
                hotel=hotel.model_copy(deep=True),
                meals=make_meals(),
                attractions=[make_attraction("A1", 126.1), make_attraction("A2", 126.2)],
            ),
            DayPlan(
                date="2026-10-04",
                day_index=1,
                description="day 2",
                transportation="公交",
                accommodation="经济型酒店",
                hotel=hotel.model_copy(deep=True),
                meals=make_meals(),
                attractions=[make_attraction("A3", 126.3), make_attraction("A4", 126.4)],
            ),
        ],
        weather_info=[
            WeatherInfo(date="2026-10-03", day_weather="晴"),
            WeatherInfo(date="2026-10-04", day_weather="多云"),
        ],
        overall_suggestions="测试建议",
        budget=Budget(
            total_attractions=40,
            total_hotels=400,
            total_meals=240,
            total_transportation=100,
            total=780,
        ),
    )


def make_attraction_evidence(count: int = 4) -> AttractionEvidence:
    return AttractionEvidence(
        attractions=[
            AttractionEvidenceItem(name=f"A{index}", address=f"A{index}地址")
            for index in range(1, count + 1)
        ]
    )


def make_weather_evidence(complete: bool = True) -> WeatherEvidence:
    dates = ["2026-10-03", "2026-10-04"]
    return WeatherEvidence(
        requested_dates=dates,
        available_dates=dates if complete else [dates[0]],
        complete=complete,
        weather_items=[WeatherInfo(date=value, day_weather="晴") for value in (dates if complete else dates[:1])],
    )


def make_hotel_evidence(count: int = 1) -> HotelEvidence:
    return HotelEvidence(
        hotels=[Hotel(name=f"H{index}", address=f"H{index}地址") for index in range(1, count + 1)]
    )


def make_state(plan: TripPlan | None = None) -> dict:
    return {
        "request": make_request(),
        "trip_plan": plan or make_valid_plan(),
        "attraction_text": "fake attraction text",
        "weather_text": "fake weather text",
        "hotel_text": "fake hotel text",
        "attraction_evidence": make_attraction_evidence(),
        "weather_evidence": make_weather_evidence(),
        "hotel_evidence": make_hotel_evidence(),
        "validation_errors": [],
        "unavailable_sources": [],
        "planner_retry_count": 0,
        "weather_retry_count": 0,
        "hotel_retry_count": 0,
        "attraction_retry_count": 0,
    }


def attraction_agent_json(count: int = 4) -> str:
    return json.dumps(
        {"attractions": [{"name": f"A{index}", "address": f"A{index}地址"} for index in range(1, count + 1)]},
        ensure_ascii=False,
    )


def weather_agent_json(complete: bool = True) -> str:
    dates = ["2026-10-03", "2026-10-04"]
    available = dates if complete else dates[:1]
    return json.dumps(
        {
            "requested_dates": dates,
            "available_dates": available,
            "complete": complete,
            "weather_items": [{"date": value, "day_weather": "晴"} for value in available],
        },
        ensure_ascii=False,
    )


def hotel_agent_json(count: int = 1) -> str:
    return json.dumps(
        {"hotels": [{"name": f"H{index}", "address": f"H{index}地址"} for index in range(1, count + 1)]},
        ensure_ascii=False,
    )


class TripPlannerWorkflowOfflineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        network_guard = patch(
            "socket.socket.connect",
            side_effect=AssertionError("network forbidden in offline tests"),
        )
        mcp_guard = patch(
            "trip_planner.trip_planner_agent.MultiServerMCPClient.get_tools",
            side_effect=AssertionError("real MCP forbidden in offline tests"),
        )
        agent_guard = patch(
            "trip_planner.trip_planner_agent.create_agent",
            side_effect=AssertionError("real Agent creation forbidden in offline tests"),
        )
        network_guard.start()
        mcp_guard.start()
        agent_guard.start()
        self.addCleanup(network_guard.stop)
        self.addCleanup(mcp_guard.stop)
        self.addCleanup(agent_guard.stop)
        self.planner = MultiAgentTripPlanner(llm=object(), amap_api_key="fake-key")

    async def validate_and_route(self, state: dict):
        update = await self.planner._validator_node(state)
        updated_state = {**state, **update}
        return updated_state, self.planner._route_after_validation(updated_state)

    async def test_case_1_valid_plan_and_evidence_reaches_end(self):
        """planner -> validator -> END with only fake agents."""
        self.planner.attraction_agent = FakeAgent(attraction_agent_json())
        self.planner.weather_agent = FakeAgent(weather_agent_json())
        self.planner.hotel_agent = FakeAgent(hotel_agent_json())
        self.planner.planner_agent = FakeAgent(make_valid_plan().model_dump_json())
        self.planner.workflow = self.planner._build_workflow()

        result = await self.planner.workflow.ainvoke({"request": make_request()})

        self.assertEqual(result["validation_status"], "passed")
        self.assertEqual(result["validation_errors"], [])
        self.assertEqual(self.planner.planner_agent.calls, 1)

    async def test_case_2_planner_error_routes_to_replan(self):
        plan = make_valid_plan()
        plan.days[0].attractions = plan.days[0].attractions[:1]
        state, route = await self.validate_and_route(make_state(plan))

        matching = [issue for issue in state["validation_errors"] if issue.code == "daily_attraction_count"]
        self.assertTrue(matching)
        self.assertTrue(all(issue.source == "planner" for issue in matching))
        self.assertEqual(route, "replan")
        prompt = self.planner._build_replan_query(state)
        self.assertIn("daily_attraction_count", prompt)
        self.assertIn("Attraction Evidence", prompt)

    async def test_case_3_incomplete_weather_routes_to_weather_repair(self):
        state = make_state()
        state["weather_evidence"] = make_weather_evidence(complete=False)
        state, route = await self.validate_and_route(state)

        self.assertIn("weather_dates_missing", [issue.code for issue in state["validation_errors"]])
        self.assertEqual(route, "repair_weather")

    async def test_case_4_empty_hotel_evidence_routes_to_hotel_repair(self):
        state = make_state()
        state["hotel_evidence"] = make_hotel_evidence(count=0)
        state, route = await self.validate_and_route(state)

        self.assertIn("hotel_evidence_missing", [issue.code for issue in state["validation_errors"]])
        self.assertEqual(route, "repair_hotel")

    async def test_case_5_insufficient_attraction_evidence_routes_to_attraction_repair(self):
        state = make_state()
        state["attraction_evidence"] = make_attraction_evidence(count=1)
        state, route = await self.validate_and_route(state)

        self.assertIn(
            "insufficient_attraction_evidence",
            [issue.code for issue in state["validation_errors"]],
        )
        self.assertEqual(route, "repair_attraction")

    async def test_case_6_budget_total_is_repaired_without_replan(self):
        plan = make_valid_plan()
        plan.budget.total = 1
        state, route = await self.validate_and_route(make_state(plan))

        self.assertEqual(state["trip_plan"].budget.total, 780)
        self.assertEqual(state["validation_errors"], [])
        self.assertEqual(route, "end")
        self.assertEqual(state["planner_retry_count"], 0)

    async def test_case_7_weather_retry_limit_degrades_and_terminates(self):
        """An exhausted weather retry must not call the weather repair path."""
        planner = self.planner
        planner.attraction_agent = FakeAgent(attraction_agent_json())
        planner.weather_agent = FakeAgent(weather_agent_json(complete=False))
        planner.hotel_agent = FakeAgent(hotel_agent_json())
        original = make_valid_plan()
        replanned = make_valid_plan()
        planner.planner_agent = FakeAgent(
            original.model_dump_json(),
            replanned.model_dump_json(),
        )
        planner.workflow = planner._build_workflow()

        result = await planner.workflow.ainvoke(
            {"request": make_request(), "weather_retry_count": 1}
        )

        self.assertEqual(planner.weather_agent.calls, 1, "weather repair must not run")
        self.assertIn("weather", result["unavailable_sources"])
        self.assertEqual(result["validation_status"], "degraded")
        self.assertEqual(result["weather_retry_count"], 1)
        self.assertEqual(result["trip_plan"].weather_info, [])

    async def test_case_8_evidence_error_has_priority_over_planner_error(self):
        plan = make_valid_plan()
        plan.days[0].attractions = plan.days[0].attractions[:1]
        state = make_state(plan)
        state["weather_evidence"] = make_weather_evidence(complete=False)
        state, route = await self.validate_and_route(state)

        sources = {issue.source for issue in state["validation_errors"]}
        self.assertIn("weather", sources)
        self.assertIn("planner", sources)
        self.assertEqual(route, "repair_weather")

    async def test_validator_checks_day_meal_hotel_and_date_rules(self):
        plan = make_valid_plan()
        plan.start_date = "2026-10-02"
        plan.days = plan.days[:1]
        plan.days[0].date = "2026-10-02"
        plan.days[0].meals = plan.days[0].meals[:2]
        plan.days[0].hotel = None

        state, _ = await self.validate_and_route(make_state(plan))
        codes = {issue.code for issue in state["validation_errors"]}
        self.assertTrue(
            {
                "travel_days_mismatch",
                "plan_date_range_mismatch",
                "day_dates_mismatch",
                "daily_meals_missing",
                "daily_hotel_missing",
            }.issubset(codes)
        )

    async def test_weather_repair_reuses_research_helper_and_increments_retry(self):
        state = make_state()
        state["weather_evidence"] = make_weather_evidence(complete=False)
        state, route = await self.validate_and_route(state)
        self.assertEqual(route, "repair_weather")

        self.planner.weather_agent = FakeAgent(weather_agent_json(complete=True))
        update = await self.planner._repair_weather_node(state)

        self.assertEqual(update["weather_retry_count"], 1)
        self.assertTrue(update["weather_evidence"].complete)
        prompt = self.planner.weather_agent.inputs[0]["messages"][0][1]
        self.assertIn("weather_dates_missing", prompt)

    async def test_planner_retry_limit_finalizes_without_loop(self):
        invalid_plan = make_valid_plan()
        invalid_plan.days[0].attractions = invalid_plan.days[0].attractions[:1]

        self.planner.attraction_agent = FakeAgent(attraction_agent_json())
        self.planner.weather_agent = FakeAgent(weather_agent_json())
        self.planner.hotel_agent = FakeAgent(hotel_agent_json())
        self.planner.planner_agent = FakeAgent(invalid_plan.model_dump_json())
        self.planner.workflow = self.planner._build_workflow()

        result = await self.planner.workflow.ainvoke({"request": make_request()})

        self.assertEqual(self.planner.planner_agent.calls, 3)
        self.assertEqual(result["planner_retry_count"], 2)
        self.assertEqual(result["validation_status"], "degraded")
        self.assertEqual(result["last_route"], "finalize")


if __name__ == "__main__":
    unittest.main()
