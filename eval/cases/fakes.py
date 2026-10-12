"""Deterministic fake agents, tools and travel data for offline evaluation."""

from __future__ import annotations

import json

from trip_planner.schemas import (
    Attraction,
    Budget,
    DayPlan,
    Hotel,
    Location,
    Meal,
    TripPlan,
    TripRequest,
    WeatherInfo,
)


class FakeAgent:
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls = 0

    async def ainvoke(self, value):
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]


class FakeTool:
    def __init__(self, name: str):
        self.name = name
        self.invocation_count = 0

    async def ainvoke(self, value=None):
        self.invocation_count += 1
        return {"ok": True, "tool": self.name}


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


def _attraction(name: str, longitude: float) -> Attraction:
    return Attraction(
        name=name,
        address=f"{name}地址",
        location=Location(longitude=longitude, latitude=45.0),
        visit_duration=120,
        description=f"{name}介绍",
        ticket_price=10,
    )


def _meals() -> list[Meal]:
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
                meals=_meals(),
                attractions=[_attraction("A1", 126.1), _attraction("A2", 126.2)],
            ),
            DayPlan(
                date="2026-10-04",
                day_index=1,
                description="day 2",
                transportation="公交",
                accommodation="经济型酒店",
                hotel=hotel.model_copy(deep=True),
                meals=_meals(),
                attractions=[_attraction("A3", 126.3), _attraction("A4", 126.4)],
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


def attraction_json() -> str:
    return json.dumps(
        {
            "attractions": [
                {"name": f"A{index}", "address": f"A{index}地址"}
                for index in range(1, 5)
            ]
        },
        ensure_ascii=False,
    )


def weather_json(*, complete: bool = True) -> str:
    dates = ["2026-10-03", "2026-10-04"]
    available = dates if complete else dates[:1]
    return json.dumps(
        {
            "requested_dates": dates,
            "available_dates": available,
            "complete": complete,
            "weather_items": [
                {"date": value, "day_weather": "晴"} for value in available
            ],
        },
        ensure_ascii=False,
    )


def hotel_json() -> str:
    return json.dumps(
        {"hotels": [{"name": "H1", "address": "H1地址"}]},
        ensure_ascii=False,
    )
