"""Pure-Python parsing helpers for research-agent structured evidence."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from trip_planner.schemas import (
    AttractionEvidence,
    AttractionEvidenceItem,
    Hotel,
    HotelEvidence,
    WeatherEvidence,
    WeatherInfo,
)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object without relying on prose keywords."""
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates.append(text.strip())

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return {}


def parse_attraction_evidence(text: str) -> AttractionEvidence:
    data = extract_json_object(text)
    items: list[AttractionEvidenceItem] = []
    raw_items = data.get("attractions", [])
    if not isinstance(raw_items, list):
        raw_items = []
    for raw_item in raw_items:
        try:
            items.append(AttractionEvidenceItem.model_validate(raw_item))
        except (ValidationError, TypeError):
            continue
    return AttractionEvidence(attractions=items)


def parse_weather_evidence(
    text: str,
    requested_dates: list[str],
) -> WeatherEvidence:
    data = extract_json_object(text)
    raw_items = data.get("weather_items", data.get("weather", data.get("forecasts", [])))
    if not isinstance(raw_items, list):
        raw_items = []
    items: list[WeatherInfo] = []
    for raw_item in raw_items:
        try:
            items.append(WeatherInfo.model_validate(raw_item))
        except (ValidationError, TypeError):
            continue

    available_dates = data.get("available_dates") or [item.date for item in items]
    if not isinstance(available_dates, list):
        available_dates = []
    available_dates = sorted({str(value) for value in available_dates})
    covers_all_dates = set(requested_dates).issubset(set(available_dates))
    declared_complete = data.get("complete")
    complete = covers_all_dates if declared_complete is None else bool(declared_complete) and covers_all_dates
    return WeatherEvidence(
        requested_dates=requested_dates,
        available_dates=available_dates,
        complete=complete,
        weather_items=items,
    )


def parse_hotel_evidence(text: str) -> HotelEvidence:
    data = extract_json_object(text)
    items: list[Hotel] = []
    raw_items = data.get("hotels", [])
    if not isinstance(raw_items, list):
        raw_items = []
    for raw_item in raw_items:
        try:
            items.append(Hotel.model_validate(raw_item))
        except (ValidationError, TypeError):
            continue
    return HotelEvidence(hotels=items)
