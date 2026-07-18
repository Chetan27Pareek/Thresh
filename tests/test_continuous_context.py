from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thresh.cli import build_parser, run_continuous_context  # noqa: E402
from thresh.continuous_context import (  # noqa: E402
    CONTEXT_WARNING,
    build_continuous_context_report,
    load_continuous_context_report,
    report_to_markdown,
)


class ContinuousContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_path = ROOT / "data" / "soy_corn_continuous_history.csv"
        cls.history = pd.read_csv(cls.data_path)

    def test_builds_corn_context_from_normalized_history(self) -> None:
        report = build_continuous_context_report(self.history, "CORN_CC1")

        self.assertEqual(report.symbol, "CORN_CC1")
        self.assertEqual(report.latest_date, "2026-06-23")
        self.assertEqual(report.latest_close, 412.0)
        self.assertEqual(report.one_day_change, 0.5)
        self.assertEqual(report.twenty_day_change, -51.25)
        self.assertEqual(report.fifty_day_change, -29.0)
        self.assertEqual(report.current_percentile, 48.8)
        self.assertEqual(report.trend_label, "Weak")
        self.assertEqual(report.volume, 19302.0)
        self.assertIsNone(report.open_interest)
        self.assertEqual(report.warning, CONTEXT_WARNING)

    def test_loads_soybean_context_from_csv(self) -> None:
        report = load_continuous_context_report(self.data_path, "SOYBEANS_SC1")

        self.assertEqual(report.symbol, "SOYBEANS_SC1")
        self.assertEqual(report.latest_date, "2026-06-23")
        self.assertEqual(report.latest_close, 1117.5)
        self.assertEqual(report.trend_label, "Weak")
        self.assertEqual(report.volume, 15991.0)
        self.assertIsNone(report.open_interest)

    def test_markdown_includes_context_warning(self) -> None:
        report = build_continuous_context_report(self.history, "CORN_CC1")
        markdown = report_to_markdown(report)

        self.assertIn("# Continuous Context Report", markdown)
        self.assertIn("**Symbol:** CORN_CC1", markdown)
        self.assertIn(CONTEXT_WARNING, markdown)

    def test_cli_can_emit_json(self) -> None:
        args = build_parser().parse_args(
            [
                "continuous-context",
                "--symbol",
                "CORN_CC1",
                "--data",
                str(self.data_path),
                "--format",
                "json",
            ]
        )

        payload = json.loads(run_continuous_context(args))

        self.assertEqual(payload["symbol"], "CORN_CC1")
        self.assertEqual(payload["trend_label"], "Weak")
        self.assertEqual(payload["warning"], CONTEXT_WARNING)


if __name__ == "__main__":
    unittest.main()
