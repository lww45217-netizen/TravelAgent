"""Deterministic business validation and repairs for generated trip plans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from trip_planner.schemas import (
    AttractionEvidence,
    HotelEvidence,
    TripPlan,
    TripRequest,
    ValidationIssue,
    WeatherEvidence,
)


@dataclass
class ValidationOutcome:
    trip_plan: TripPlan
    issues: list[ValidationIssue]
    repaired_codes: list[str]


def requested_dates(request: TripRequest) -> list[str]:
    """Return the inclusive travel date range; invalid input remains safely empty."""
    try:
        current = date.fromisoformat(request.start_date)
        end = date.fromisoformat(request.end_date)
    except ValueError:
        return []
    if end < current:
        return []

    values: list[str] = []
    while current <= end:
        values.append(current.isoformat())
        current += timedelta(days=1)
    return values


def _issue(source: str, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(source=source, code=code, message=message)


def _append_unavailable_notice(plan: TripPlan, source: str) -> None:
    labels = {"weather": "天气", "hotel": "酒店", "attraction": "景点"}
    notice = f"{labels[source]}数据暂不可获取，请在出行前再次确认。"
    if notice not in plan.overall_suggestions:
        separator = "\n" if plan.overall_suggestions else ""
        plan.overall_suggestions += f"{separator}{notice}"


def validate_trip_plan(
    request: TripRequest,
    trip_plan: TripPlan,
    attraction_evidence: AttractionEvidence,
    weather_evidence: WeatherEvidence,
    hotel_evidence: HotelEvidence,
    unavailable_sources: list[str] | None = None,
) -> ValidationOutcome:
    """Validate business rules and apply only deterministic Python repairs."""
    unavailable = set(unavailable_sources or [])
    plan = trip_plan.model_copy(deep=True)
    issues: list[ValidationIssue] = []
    repaired_codes: list[str] = []

    # When evidence is explicitly unavailable, remove unverifiable facts instead
    # of allowing a planner to invent replacements.
    if "weather" in unavailable and plan.weather_info:
        plan.weather_info = []
        repaired_codes.append("weather_unavailable_clear")
        _append_unavailable_notice(plan, "weather")
    if "hotel" in unavailable:
        changed = any(day.hotel is not None for day in plan.days)
        for day in plan.days:
            day.hotel = None
        if changed:
            repaired_codes.append("hotel_unavailable_clear")
        _append_unavailable_notice(plan, "hotel")
    if "attraction" in unavailable:
        changed = any(day.attractions for day in plan.days)
        for day in plan.days:
            day.attractions = []
        if changed:
            repaired_codes.append("attraction_unavailable_clear")
        _append_unavailable_notice(plan, "attraction")

    if len(plan.days) != request.travel_days:
        issues.append(
            _issue(
                "planner",
                "travel_days_mismatch",
                f"行程包含 {len(plan.days)} 天，用户要求 {request.travel_days} 天",
            )
        )

    if plan.start_date != request.start_date or plan.end_date != request.end_date:
        issues.append(
            _issue(
                "planner",
                "plan_date_range_mismatch",
                "行程起止日期与用户请求不一致",
            )
        )

    expected_dates = requested_dates(request)
    actual_dates = [day.date for day in plan.days]
    if expected_dates and actual_dates != expected_dates:
        issues.append(
            _issue(
                "planner",
                "day_dates_mismatch",
                f"每日日期 {actual_dates} 未与旅行日期 {expected_dates} 按顺序对齐",
            )
        )

    evidence_attraction_names = {
        item.name.strip() for item in attraction_evidence.attractions if item.name.strip()
    }
    evidence_hotel_names = {
        item.name.strip() for item in hotel_evidence.hotels if item.name.strip()
    }

    for day in plan.days:
        if "attraction" not in unavailable:
            if not 2 <= len(day.attractions) <= 3:
                issues.append(
                    _issue(
                        "planner",
                        "daily_attraction_count",
                        f"{day.date} 安排了 {len(day.attractions)} 个景点，应为 2-3 个",
                    )
                )
            unknown_attractions = [
                item.name
                for item in day.attractions
                if evidence_attraction_names and item.name not in evidence_attraction_names
            ]
            if unknown_attractions:
                issues.append(
                    _issue(
                        "planner",
                        "attraction_not_in_evidence",
                        f"{day.date} 包含 Evidence 中不存在的景点: {', '.join(unknown_attractions)}",
                    )
                )

        meal_types = {meal.type.strip().lower() for meal in day.meals}
        missing_meals = sorted({"breakfast", "lunch", "dinner"} - meal_types)
        if missing_meals:
            issues.append(
                _issue(
                    "planner",
                    "daily_meals_missing",
                    f"{day.date} 缺少餐食: {', '.join(missing_meals)}",
                )
            )

        if "hotel" not in unavailable:
            if day.hotel is None:
                issues.append(
                    _issue("planner", "daily_hotel_missing", f"{day.date} 缺少酒店安排")
                )
            elif evidence_hotel_names and day.hotel.name not in evidence_hotel_names:
                issues.append(
                    _issue(
                        "planner",
                        "hotel_not_in_evidence",
                        f"{day.date} 的酒店 {day.hotel.name} 不在 Hotel Evidence 中",
                    )
                )

    if "weather" not in unavailable:
        available_dates = set(weather_evidence.available_dates)
        missing_dates = [value for value in expected_dates if value not in available_dates]
        if not weather_evidence.complete or missing_dates:
            issues.append(
                _issue(
                    "weather",
                    "weather_dates_missing",
                    f"天气数据未覆盖日期: {', '.join(missing_dates) or '完整性标记为 false'}",
                )
            )

    if "attraction" not in unavailable:
        required_count = request.travel_days * 2
        if attraction_evidence.count < required_count:
            issues.append(
                _issue(
                    "attraction",
                    "insufficient_attraction_evidence",
                    f"仅有 {attraction_evidence.count} 个候选景点，至少需要 {required_count} 个",
                )
            )

    if "hotel" not in unavailable:
        valid_hotels = [item for item in hotel_evidence.hotels if item.name.strip()]
        if not valid_hotels:
            issues.append(
                _issue("hotel", "hotel_evidence_missing", "Hotel Evidence 中没有有效酒店")
            )

    if plan.budget is None:
        issues.append(_issue("planner", "budget_missing", "行程缺少预算汇总"))
    else:
        expected_total = (
            plan.budget.total_attractions
            + plan.budget.total_hotels
            + plan.budget.total_meals
            + plan.budget.total_transportation
        )
        if plan.budget.total != expected_total:
            plan.budget.total = expected_total
            repaired_codes.append("budget_total_recalculated")

    return ValidationOutcome(
        trip_plan=plan,
        issues=issues,
        repaired_codes=repaired_codes,
    )
