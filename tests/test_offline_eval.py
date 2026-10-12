import unittest
from unittest.mock import patch

from eval.run_offline_eval import run_offline_eval


class OfflineEvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_nine_cases_and_metrics_pass_offline(self):
        with patch(
            "socket.socket.connect",
            side_effect=AssertionError("network forbidden in offline eval test"),
        ):
            report = await run_offline_eval()

        self.assertTrue(report["all_passed"], report)
        self.assertEqual(len(report["cases"]), 9)
        self.assertTrue(all(report["cases"].values()))
        self.assertTrue(all(value == 1.0 for value in report["metrics"].values()))


if __name__ == "__main__":
    unittest.main()
