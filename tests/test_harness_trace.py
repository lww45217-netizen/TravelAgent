import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import ToolMessage

from trip_planner.harness import TravelAgentHarness


class HarnessTraceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="harness-trace-")
        self.addCleanup(self.temp_dir.cleanup)
        self.harness = TravelAgentHarness(log_dir=Path(self.temp_dir.name))

    async def test_trace_contains_run_and_node_lifecycle_with_latency(self):
        harness = self.harness

        class FakeWorkflow:
            async def ainvoke(self, state):
                async def node():
                    return {"ok": True}

                await harness.run_traced("weather", node)
                return {"validation_status": "passed"}

        _, summary = await harness.run_workflow(FakeWorkflow(), {})
        trace_path = harness.trace_writer.path_for(summary.run_id)
        rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
        ]
        event_types = [row["event_type"] for row in rows]

        self.assertIn("run_start", event_types)
        self.assertIn("node_start", event_types)
        self.assertIn("node_end", event_types)
        self.assertIn("run_end", event_types)
        self.assertTrue(all(row["run_id"] == summary.run_id for row in rows))
        node_end = next(row for row in rows if row["event_type"] == "node_end")
        self.assertGreaterEqual(node_end["latency_ms"], 0)

    async def test_node_error_and_failed_run_are_traced(self):
        harness = self.harness

        class FailingWorkflow:
            async def ainvoke(self, state):
                async def node():
                    raise ValueError("fake failure")

                return await harness.run_traced("planner", node)

        with self.assertRaises(ValueError):
            await harness.run_workflow(FailingWorkflow(), {})

        paths = list(Path(self.temp_dir.name).glob("*.jsonl"))
        self.assertEqual(len(paths), 1)
        rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
        self.assertIn("node_error", {row["event_type"] for row in rows})
        run_end = next(row for row in rows if row["event_type"] == "run_end")
        self.assertEqual(run_end["status"], "error")

    async def test_sensitive_metadata_is_redacted(self):
        context, token = self.harness.start_run()
        try:
            self.harness.record(
                "test",
                "metadata",
                "success",
                metadata={
                    "api_key": "secret-key-value",
                    "Authorization": "Bearer secret-token",
                    "prompt": "private prompt",
                    "tool": "maps_weather",
                },
            )
            self.harness.finish_run(context, success=True, final_status="passed")
        finally:
            self.harness.reset_run(token)

        content = self.harness.trace_writer.path_for(context.run_id).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("secret-key-value", content)
        self.assertNotIn("secret-token", content)
        self.assertNotIn("private prompt", content)
        self.assertIn("[REDACTED]", content)
        self.assertIn("maps_weather", content)

    async def test_tool_message_error_status_is_traced_as_error(self):
        context, token = self.harness.start_run()
        try:
            result = await self.harness.call_tool(
                "weather",
                "maps_weather",
                lambda: self._error_tool_result(),
            )
            self.harness.finish_run(context, success=True, final_status="passed")
        finally:
            self.harness.reset_run(token)

        self.assertEqual(result.status, "error")
        tool_end = next(
            event for event in context.events if event.event_type == "tool_call_end"
        )
        self.assertEqual(tool_end.status, "error")

    @staticmethod
    async def _error_tool_result():
        return ToolMessage(
            content="fake error",
            tool_call_id="fake-call",
            status="error",
        )
