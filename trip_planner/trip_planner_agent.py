import asyncio
import json
from typing import TypedDict

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

from trip_planner.evidence import (
    extract_json_object,
    parse_attraction_evidence,
    parse_hotel_evidence,
    parse_weather_evidence,
)
from trip_planner.harness import (
    TravelAgentHarness,
    prepare_agent_tools,
)
from trip_planner.prompts import (
    ATTRACTION_AGENT_PROMPT,
    HOTEL_AGENT_PROMPT,
    PLANNER_AGENT_PROMPT,
    WEATHER_AGENT_PROMPT,
)
from trip_planner.schemas import (
    AttractionEvidence,
    HotelEvidence,
    TripPlan,
    TripRequest,
    ValidationIssue,
    WeatherEvidence,
)
from trip_planner.validation import requested_dates, validate_trip_plan


class TripPlannerState(TypedDict, total=False):
    """Shared state for the complete LangGraph trip-planning workflow."""

    request: TripRequest
    attraction_text: str
    weather_text: str
    hotel_text: str
    trip_plan: TripPlan

    attraction_evidence: AttractionEvidence
    weather_evidence: WeatherEvidence
    hotel_evidence: HotelEvidence

    validation_errors: list[ValidationIssue]
    validation_status: str
    last_route: str
    unavailable_sources: list[str]

    planner_retry_count: int
    weather_retry_count: int
    hotel_retry_count: int
    attraction_retry_count: int


class MultiAgentTripPlanner:
    """Multi-agent planner orchestrated by a parent LangGraph workflow."""

    MAX_PLANNER_RETRIES = 2
    MAX_WEATHER_RETRIES = 1
    MAX_HOTEL_RETRIES = 1
    MAX_ATTRACTION_RETRIES = 1

    def __init__(
        self,
        llm=None,
        amap_api_key: str | None = None,
        harness: TravelAgentHarness | None = None,
    ):
        # Configuration is loaded lazily in initialize(). Offline tests can
        # construct the workflow without reading .env or creating real clients.
        self.llm = llm
        self.amap_api_key = amap_api_key
        self.harness = harness or TravelAgentHarness()
        self.amap_tool = None
        self.weather_agent = None
        self.attraction_agent = None
        self.hotel_agent = None
        self.planner_agent = None
        self.workflow = None

    async def initialize(self):
        """Initialize real production agents and compile the parent graph."""
        if self.workflow is not None:
            return

        if self.llm is None:
            from my_llm import llm

            self.llm = llm
        if self.amap_api_key is None:
            from env_utils import AMAP_API_KEY

            self.amap_api_key = AMAP_API_KEY
        if not self.amap_api_key:
            raise RuntimeError("缺少环境变量 AMAP_API_KEY，请在项目根目录的 .env 中配置")

        print("初始化多智能体旅行规划系统...")
        try:
            print("  - 创建共享MCP工具...")
            self.amap_tool = MultiServerMCPClient(
                {
                    "amap": {
                        "url": f"https://mcp.amap.com/mcp?key={self.amap_api_key}",
                        "transport": "streamable_http",
                    }
                }
            )
            tools = await self.amap_tool.get_tools()

            weather_tools = prepare_agent_tools(tools, "weather", self.harness)
            attraction_tools = prepare_agent_tools(tools, "attraction", self.harness)
            hotel_tools = prepare_agent_tools(tools, "hotel", self.harness)

            print("  - 创建天气查询Agent...")
            self.weather_agent = self.harness.instrument_agent(
                create_agent(
                    self.llm,
                    weather_tools,
                    system_prompt=WEATHER_AGENT_PROMPT,
                )
            )
            print("  - 创建景点搜索Agent...")
            self.attraction_agent = self.harness.instrument_agent(
                create_agent(
                    self.llm,
                    attraction_tools,
                    system_prompt=ATTRACTION_AGENT_PROMPT,
                )
            )
            print("  - 创建酒店推荐Agent...")
            self.hotel_agent = self.harness.instrument_agent(
                create_agent(
                    self.llm,
                    hotel_tools,
                    system_prompt=HOTEL_AGENT_PROMPT,
                )
            )
            print("  - 创建行程规划Agent...")
            self.planner_agent = self.harness.instrument_agent(
                create_agent(self.llm, system_prompt=PLANNER_AGENT_PROMPT)
            )

            print("  - 创建 LangGraph 验证与修复工作流...")
            self.workflow = self._build_workflow()

            print("✅ 多智能体系统初始化成功")
            print("   主流程: attraction -> weather -> hotel -> planner -> validator")
            print(f"   共享高德地图工具数量: {len(tools)} 个")
            print(f"   Weather tools: {[tool.name for tool in weather_tools]}")
            print(f"   Attraction tools: {[tool.name for tool in attraction_tools]}")
            print(f"   Hotel tools: {[tool.name for tool in hotel_tools]}")
            print("   Planner tools: []")
        except Exception as exc:
            print(f"❌ 多智能体系统初始化失败: {exc}")
            raise

    async def plan_trip(self, request: TripRequest) -> TripPlan:
        """Execute one isolated graph invocation and return its final TripPlan."""
        if self.workflow is None:
            raise RuntimeError("多智能体系统尚未初始化，请先调用 initialize()")

        print(f"\n{'=' * 60}")
        print("🚀 开始由 LangGraph 调度多智能体...")
        print(f"目的地: {request.city}")
        print(f"日期: {request.start_date} 至 {request.end_date}")
        print(f"天数: {request.travel_days}天")
        print(f"偏好: {', '.join(request.preferences) if request.preferences else '无'}")
        print(f"{'=' * 60}\n")

        result, summary = await self.harness.run_workflow(
            self.workflow,
            {"request": request},
        )
        trip_plan = result["trip_plan"]

        print(f"{'=' * 60}")
        print(f"✅ 旅行计划生成完成! 状态: {result.get('validation_status')}")
        print(
            f"   Harness: run_id={summary.run_id}, "
            f"tools={summary.total_tool_calls}, "
            f"llm_calls={summary.llm_calls}, "
            f"latency_ms={summary.total_latency_ms:.1f}"
        )
        print(f"{'=' * 60}\n")
        return trip_plan

    def _build_workflow(self):
        graph = StateGraph(TripPlannerState)
        graph.add_node(
            "attraction", self.harness.traced_node("attraction", self._attraction_node)
        )
        graph.add_node(
            "weather", self.harness.traced_node("weather", self._weather_node)
        )
        graph.add_node("hotel", self.harness.traced_node("hotel", self._hotel_node))
        graph.add_node(
            "planner", self.harness.traced_node("planner", self._planner_node)
        )
        graph.add_node(
            "validator", self.harness.traced_node("validator", self._validator_node)
        )
        graph.add_node(
            "repair_weather",
            self.harness.traced_node("repair_weather", self._repair_weather_node),
        )
        graph.add_node(
            "repair_hotel",
            self.harness.traced_node("repair_hotel", self._repair_hotel_node),
        )
        graph.add_node(
            "repair_attraction",
            self.harness.traced_node(
                "repair_attraction", self._repair_attraction_node
            ),
        )
        graph.add_node(
            "degrade_weather",
            self.harness.traced_node("degrade_weather", self._degrade_weather_node),
        )
        graph.add_node(
            "degrade_hotel",
            self.harness.traced_node("degrade_hotel", self._degrade_hotel_node),
        )
        graph.add_node(
            "degrade_attraction",
            self.harness.traced_node(
                "degrade_attraction", self._degrade_attraction_node
            ),
        )
        graph.add_node(
            "replan", self.harness.traced_node("replan", self._replan_node)
        )
        graph.add_node(
            "finalize", self.harness.traced_node("finalize", self._finalize_node)
        )

        graph.add_edge(START, "attraction")
        graph.add_edge("attraction", "weather")
        graph.add_edge("weather", "hotel")
        graph.add_edge("hotel", "planner")
        graph.add_edge("planner", "validator")

        graph.add_conditional_edges(
            "validator",
            self._route_after_validation,
            {
                "end": END,
                "repair_weather": "repair_weather",
                "repair_hotel": "repair_hotel",
                "repair_attraction": "repair_attraction",
                "degrade_weather": "degrade_weather",
                "degrade_hotel": "degrade_hotel",
                "degrade_attraction": "degrade_attraction",
                "replan": "replan",
                "finalize": "finalize",
            },
        )

        graph.add_edge("repair_weather", "replan")
        graph.add_edge("repair_hotel", "replan")
        graph.add_edge("repair_attraction", "replan")
        graph.add_edge("degrade_weather", "replan")
        graph.add_edge("degrade_hotel", "replan")
        graph.add_edge("degrade_attraction", "replan")
        graph.add_edge("replan", "validator")
        graph.add_edge("finalize", END)
        return graph.compile()

    async def _attraction_node(self, state: TripPlannerState):
        print("📍 Attraction Node: 搜索景点...")
        text, evidence = await self._research_attractions(state["request"])
        return {
            "attraction_text": text,
            "attraction_evidence": evidence,
            "attraction_retry_count": state.get("attraction_retry_count", 0),
            "unavailable_sources": state.get("unavailable_sources", []),
        }

    async def _weather_node(self, state: TripPlannerState):
        print("🌤️  Weather Node: 查询天气...")
        text, evidence = await self._research_weather(state["request"])
        return {
            "weather_text": text,
            "weather_evidence": evidence,
            "weather_retry_count": state.get("weather_retry_count", 0),
        }

    async def _hotel_node(self, state: TripPlannerState):
        print("🏨 Hotel Node: 搜索酒店...")
        text, evidence = await self._research_hotels(state["request"])
        return {
            "hotel_text": text,
            "hotel_evidence": evidence,
            "hotel_retry_count": state.get("hotel_retry_count", 0),
        }

    async def _planner_node(self, state: TripPlannerState):
        print("📋 Planner Node: 生成行程计划...")
        query = self._build_planner_query(
            state["request"],
            state["attraction_text"],
            state["weather_text"],
            state["hotel_text"],
        )
        response = await self.harness.invoke_agent(
            "planner",
            self.planner_agent,
            {"messages": [("user", query)]},
        )
        text = self._extract_text(response)
        return {
            "trip_plan": self._parse_response(text),
            "planner_retry_count": state.get("planner_retry_count", 0),
            "validation_errors": [],
            "validation_status": "pending",
        }

    async def _validator_node(self, state: TripPlannerState):
        print("[Validator] checking trip plan...")
        outcome = validate_trip_plan(
            request=state["request"],
            trip_plan=state["trip_plan"],
            attraction_evidence=state.get("attraction_evidence", AttractionEvidence()),
            weather_evidence=state.get(
                "weather_evidence",
                WeatherEvidence(requested_dates=requested_dates(state["request"])),
            ),
            hotel_evidence=state.get("hotel_evidence", HotelEvidence()),
            unavailable_sources=state.get("unavailable_sources", []),
        )

        for code in outcome.repaired_codes:
            print(f"[Python Repair] {code}")

        if outcome.issues:
            print("[Validator] failed:")
            for issue in outcome.issues:
                print(f"  - {issue.source}:{issue.code}")
            status = "failed"
        else:
            status = "degraded" if state.get("unavailable_sources") else "passed"
            print(f"[Validator] {status}")

        self.harness.record_validation(
            status,
            [issue.code for issue in outcome.issues],
        )

        return {
            "trip_plan": outcome.trip_plan,
            "validation_errors": outcome.issues,
            "validation_status": status,
            "last_route": "validator",
        }

    def _route_after_validation(self, state: TripPlannerState) -> str:
        issues = state.get("validation_errors", [])
        if not issues:
            return self._select_route("end")

        unavailable = set(state.get("unavailable_sources", []))
        priorities = (
            ("weather", "weather_retry_count", self.MAX_WEATHER_RETRIES),
            ("hotel", "hotel_retry_count", self.MAX_HOTEL_RETRIES),
            ("attraction", "attraction_retry_count", self.MAX_ATTRACTION_RETRIES),
        )
        for source, counter_name, maximum in priorities:
            if source in unavailable or not any(issue.source == source for issue in issues):
                continue
            route = (
                f"repair_{source}"
                if state.get(counter_name, 0) < maximum
                else f"degrade_{source}"
            )
            return self._select_route(route)

        if any(issue.source == "planner" for issue in issues):
            route = (
                "replan"
                if state.get("planner_retry_count", 0) < self.MAX_PLANNER_RETRIES
                else "finalize"
            )
            return self._select_route(route)

        return self._select_route("finalize")

    def _select_route(self, route: str) -> str:
        print(f"[Router] route -> {'END' if route == 'end' else route}")
        self.harness.record_route(route)
        return route

    async def _repair_weather_node(self, state: TripPlannerState):
        self.harness.record_repair("weather")
        retry_count = state.get("weather_retry_count", 0) + 1
        print(f"[Retry] weather {retry_count}/{self.MAX_WEATHER_RETRIES}")
        issues = self._issues_for_source(state, "weather")
        try:
            text, evidence = await self._research_weather(state["request"], issues)
        except Exception as exc:
            print(f"[Repair Weather] failed: {exc}")
            text = state.get("weather_text", "")
            evidence = state.get("weather_evidence", WeatherEvidence())
        print("[Repair Weather] completed")
        return {
            "weather_text": text,
            "weather_evidence": evidence,
            "weather_retry_count": retry_count,
            "last_route": "repair_weather",
        }

    async def _repair_hotel_node(self, state: TripPlannerState):
        self.harness.record_repair("hotel")
        retry_count = state.get("hotel_retry_count", 0) + 1
        print(f"[Retry] hotel {retry_count}/{self.MAX_HOTEL_RETRIES}")
        issues = self._issues_for_source(state, "hotel")
        try:
            text, evidence = await self._research_hotels(state["request"], issues)
        except Exception as exc:
            print(f"[Repair Hotel] failed: {exc}")
            text = state.get("hotel_text", "")
            evidence = state.get("hotel_evidence", HotelEvidence())
        print("[Repair Hotel] completed")
        return {
            "hotel_text": text,
            "hotel_evidence": evidence,
            "hotel_retry_count": retry_count,
            "last_route": "repair_hotel",
        }

    async def _repair_attraction_node(self, state: TripPlannerState):
        self.harness.record_repair("attraction")
        retry_count = state.get("attraction_retry_count", 0) + 1
        print(f"[Retry] attraction {retry_count}/{self.MAX_ATTRACTION_RETRIES}")
        issues = self._issues_for_source(state, "attraction")
        try:
            text, evidence = await self._research_attractions(state["request"], issues)
        except Exception as exc:
            print(f"[Repair Attraction] failed: {exc}")
            text = state.get("attraction_text", "")
            evidence = state.get("attraction_evidence", AttractionEvidence())
        print("[Repair Attraction] completed")
        return {
            "attraction_text": text,
            "attraction_evidence": evidence,
            "attraction_retry_count": retry_count,
            "last_route": "repair_attraction",
        }

    async def _degrade_weather_node(self, state: TripPlannerState):
        return self._mark_unavailable(state, "weather")

    async def _degrade_hotel_node(self, state: TripPlannerState):
        return self._mark_unavailable(state, "hotel")

    async def _degrade_attraction_node(self, state: TripPlannerState):
        return self._mark_unavailable(state, "attraction")

    def _mark_unavailable(self, state: TripPlannerState, source: str):
        unavailable = list(state.get("unavailable_sources", []))
        if source not in unavailable:
            unavailable.append(source)
        print(f"[Retry] {source} retry limit reached; marked unavailable")
        return {
            "unavailable_sources": unavailable,
            "last_route": f"degrade_{source}",
        }

    async def _replan_node(self, state: TripPlannerState):
        retry_count = state.get("planner_retry_count", 0)
        if retry_count >= self.MAX_PLANNER_RETRIES:
            print(
                f"[Replan] skipped: planner retry limit "
                f"{retry_count}/{self.MAX_PLANNER_RETRIES}"
            )
            return {"last_route": "replan_skipped"}

        retry_count += 1
        self.harness.record_replan()
        print(f"[Replan] planner retry {retry_count}/{self.MAX_PLANNER_RETRIES}")
        query = self._build_replan_query(state)
        try:
            response = await self.harness.invoke_agent(
                "planner",
                self.planner_agent,
                {"messages": [("user", query)]},
            )
            text = self._extract_text(response)
            plan = self._parse_response(text)
        except Exception as exc:
            print(f"[Replan] failed: {exc}")
            plan = state["trip_plan"]
        return {
            "trip_plan": plan,
            "planner_retry_count": retry_count,
            "last_route": "replan",
        }

    async def _finalize_node(self, state: TripPlannerState):
        plan = state["trip_plan"].model_copy(deep=True)
        codes = [issue.code for issue in state.get("validation_errors", [])]
        if codes:
            notice = f"未能自动修复的约束: {', '.join(codes)}"
            if notice not in plan.overall_suggestions:
                plan.overall_suggestions += f"\n{notice}"
        print("[Router] workflow finalized with degraded result")
        return {
            "trip_plan": plan,
            "validation_status": "degraded",
            "last_route": "finalize",
        }

    async def _research_attractions(
        self,
        request: TripRequest,
        repair_issues: list[ValidationIssue] | None = None,
    ) -> tuple[str, AttractionEvidence]:
        response = await self.harness.invoke_agent(
            "attraction",
            self.attraction_agent,
            self._build_attraction_query(request, repair_issues),
        )
        text = self._extract_text(response)
        evidence = parse_attraction_evidence(text)
        print(f"景点 Evidence: {evidence.count} 个候选")
        return text, evidence

    async def _research_weather(
        self,
        request: TripRequest,
        repair_issues: list[ValidationIssue] | None = None,
    ) -> tuple[str, WeatherEvidence]:
        response = await self.harness.invoke_agent(
            "weather",
            self.weather_agent,
            self._build_weather_query(request, repair_issues),
        )
        text = self._extract_text(response)
        evidence = parse_weather_evidence(text, requested_dates(request))
        print(
            f"天气 Evidence: {len(evidence.available_dates)}/"
            f"{len(evidence.requested_dates)} 天"
        )
        return text, evidence

    async def _research_hotels(
        self,
        request: TripRequest,
        repair_issues: list[ValidationIssue] | None = None,
    ) -> tuple[str, HotelEvidence]:
        response = await self.harness.invoke_agent(
            "hotel",
            self.hotel_agent,
            self._build_hotel_query(request, repair_issues),
        )
        text = self._extract_text(response)
        evidence = parse_hotel_evidence(text)
        print(f"酒店 Evidence: {evidence.count} 个候选")
        return text, evidence

    def _build_attraction_query(
        self,
        request: TripRequest,
        repair_issues: list[ValidationIssue] | None = None,
    ) -> dict:
        preferences = ", ".join(request.preferences) if request.preferences else "经典景点"
        repair_context = self._repair_context(repair_issues)
        prompt = f"""请搜索{request.city}适合{request.travel_days}天游玩的{preferences}，推荐8-12个热门景点。
{repair_context}
请在工具调用后只返回 JSON：
{{"summary":"简短总结","attractions":[{{"name":"","address":"","location":{{"longitude":0,"latitude":0}},"description":"","ticket_price":0,"poi_id":""}}]}}
不要编造工具结果中不存在的景点。"""
        return {"messages": [("user", prompt)]}

    def _build_weather_query(
        self,
        request: TripRequest,
        repair_issues: list[ValidationIssue] | None = None,
    ) -> dict:
        dates = requested_dates(request)
        repair_context = self._repair_context(repair_issues)
        prompt = f"""请查询{request.city}从{request.start_date}到{request.end_date}的天气。
{repair_context}
请在工具调用后只返回 JSON：
{{"requested_dates":{json.dumps(dates, ensure_ascii=False)},"available_dates":[],"complete":false,"weather_items":[{{"date":"YYYY-MM-DD","day_weather":"","night_weather":"","day_temp":0,"night_temp":0,"wind_direction":"","wind_power":""}}]}}
如果工具无法覆盖某些日期，保持 complete=false，不得编造天气。"""
        return {"messages": [("user", prompt)]}

    def _build_hotel_query(
        self,
        request: TripRequest,
        repair_issues: list[ValidationIssue] | None = None,
    ) -> dict:
        repair_context = self._repair_context(repair_issues)
        prompt = f"""请在{request.city}搜索{request.accommodation}，推荐6-10家位置方便、价格适中的酒店。
{repair_context}
请在工具调用后只返回 JSON：
{{"summary":"简短总结","hotels":[{{"name":"","address":"","location":{{"longitude":0,"latitude":0}},"price_range":"","rating":"","distance":"","type":"","estimated_cost":0}}]}}
不要编造工具结果中不存在的酒店。"""
        return {"messages": [("user", prompt)]}

    def _build_planner_query(
        self,
        request: TripRequest,
        attractions: str,
        weather: str,
        hotels: str,
    ) -> str:
        query = f"""请根据以下信息生成{request.city}的{request.travel_days}天旅行计划:

**基本信息:**
- 城市: {request.city}
- 日期: {request.start_date} 至 {request.end_date}
- 天数: {request.travel_days}天
- 交通方式: {request.transportation}
- 住宿: {request.accommodation}
- 偏好: {', '.join(request.preferences) if request.preferences else '无'}

**景点 Evidence 原文:**
{attractions}

**天气 Evidence 原文:**
{weather}

**酒店 Evidence 原文:**
{hotels}

**要求:**
1. 每天安排2-3个景点
2. 每天必须包含早中晚三餐
3. 每天从 Hotel Evidence 中选择一个具体酒店
4. 只能使用 Evidence 中存在的景点、天气和酒店
5. 返回完整的JSON格式数据
"""
        if request.free_text_input:
            query += f"\n**额外要求:** {request.free_text_input}"
        return query

    def _build_replan_query(self, state: TripPlannerState) -> str:
        request = state["request"]
        errors = [issue.model_dump() for issue in state.get("validation_errors", [])]
        attraction_evidence = state.get("attraction_evidence", AttractionEvidence())
        weather_evidence = state.get("weather_evidence", WeatherEvidence())
        hotel_evidence = state.get("hotel_evidence", HotelEvidence())
        unavailable = state.get("unavailable_sources", [])
        return f"""请修复当前旅行计划，不要重新随意生成新事实。

**原始用户请求:**
{request.model_dump_json(indent=2)}

**当前 TripPlan:**
{state['trip_plan'].model_dump_json(indent=2)}

**Validation Errors:**
{json.dumps(errors, ensure_ascii=False, indent=2)}

**Attraction Evidence:**
{attraction_evidence.model_dump_json(indent=2)}

**Weather Evidence:**
{weather_evidence.model_dump_json(indent=2)}

**Hotel Evidence:**
{hotel_evidence.model_dump_json(indent=2)}

**Unavailable Sources:**
{json.dumps(unavailable, ensure_ascii=False)}

只修复 Validation Errors 指出的问题。必须基于 Evidence，不得编造 Evidence 中不存在的事实。
对于 Unavailable Sources，明确提示暂不可获取，不得填充虚构数据。
直接返回符合 TripPlan Schema 的完整 JSON。
"""

    @staticmethod
    def _repair_context(issues: list[ValidationIssue] | None) -> str:
        if not issues:
            return ""
        details = "\n".join(f"- {issue.code}: {issue.message}" for issue in issues)
        return f"上一次 Evidence 存在以下问题，请定向补充:\n{details}"

    @staticmethod
    def _issues_for_source(
        state: TripPlannerState,
        source: str,
    ) -> list[ValidationIssue]:
        return [
            issue
            for issue in state.get("validation_errors", [])
            if issue.source == source
        ]

    @staticmethod
    def _parse_response(response: str) -> TripPlan:
        data = extract_json_object(response)
        if not data:
            raise ValueError("未在 Planner 响应中找到有效 JSON 对象")
        return TripPlan.model_validate(data)

    @staticmethod
    def _extract_text(response) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, dict) and "messages" in response:
            messages = response["messages"]
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    content = last_message.content
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        return "".join(
                            item.get("text", "")
                            for item in content
                            if item.get("type") == "text"
                        )
        return str(response)[:500]


_multi_agent_planner = None


def get_trip_planner_agent() -> MultiAgentTripPlanner:
    global _multi_agent_planner
    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner()
    return _multi_agent_planner


async def main():
    """Manual production smoke test; never invoked by the offline test suite."""
    planner = MultiAgentTripPlanner()
    await planner.initialize()
    request = TripRequest(
        city="北京",
        start_date="2026-10-01",
        end_date="2026-10-03",
        travel_days=3,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化", "美食"],
    )
    trip_plan = await planner.plan_trip(request)
    print(trip_plan.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
