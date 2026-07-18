from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thresh.cli import build_parser, run_seasonal_agent
from thresh.continuous_context import load_continuous_context_report
from thresh.seasonal_agent import (
    MARKET_CONTEXT_NOTE,
    SeasonalAgent,
    SeasonalAgentConfig,
    report_to_json,
    report_to_markdown,
)


class SeasonalAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_path = ROOT / "data" / "sample_corn_july_dec_spread.csv"
        cls.continuous_data_path = ROOT / "data" / "soy_corn_continuous_history.csv"
        cls.spread_history = pd.read_csv(cls.data_path)
        cls.continuous_context = load_continuous_context_report(
            cls.continuous_data_path,
            "CORN_CC1",
        )

    def test_analyze_returns_structured_report(self) -> None:
        agent = SeasonalAgent(
            spread_history=self.spread_history,
            config=SeasonalAgentConfig(
                commodity="corn",
                spread_id="CORN_NZ",
                horizon_days=30,
                min_years=10,
            ),
        )

        report = agent.analyze("2024-06-07")

        self.assertEqual(report.commodity, "corn")
        self.assertEqual(report.spread_id, "CORN_NZ")
        self.assertEqual(report.analysis_date, "2024-06-07")
        self.assertEqual(report.historical_years_used, 19)
        self.assertIn(report.seasonal_signal, {"Bullish", "Bearish", "Neutral"})
        self.assertEqual(len(report.top_analog_years), 5)
        self.assertIn("Seasonal Agent view", report.narrative)

    def test_report_json_is_machine_readable(self) -> None:
        agent = SeasonalAgent(
            spread_history=self.spread_history,
            config=SeasonalAgentConfig(commodity="corn", spread_id="CORN_NZ"),
        )

        payload = json.loads(agent.analyze("2024-06-07").to_json())

        self.assertEqual(payload["spread_id"], "CORN_NZ")
        self.assertIsNone(payload["continuous_context"])
        self.assertIsInstance(payload["top_analog_years"], list)
        self.assertIn("move_already_priced", payload)

    def test_accepts_processed_spread_history_contract_year_schema(self) -> None:
        processed_schema = self.spread_history.rename(columns={"year": "contract_year"})
        processed_schema = processed_schema.drop(columns=["year"], errors="ignore")
        agent = SeasonalAgent(
            spread_history=processed_schema,
            config=SeasonalAgentConfig(commodity="corn", spread_id="CORN_NZ"),
        )

        report = agent.analyze("2024-06-07")

        self.assertEqual(report.analysis_date, "2024-06-07")
        self.assertEqual(report.historical_years_used, 19)
        self.assertIsNotNone(report.move_already_priced)

    def test_warns_when_fewer_than_min_years_exist(self) -> None:
        limited_history = self.spread_history[self.spread_history["year"].between(2020, 2024)]
        agent = SeasonalAgent(
            spread_history=limited_history,
            config=SeasonalAgentConfig(
                commodity="corn",
                spread_id="CORN_NZ",
                min_years=10,
            ),
        )

        report = agent.analyze("2024-06-07")

        self.assertEqual(report.historical_years_used, 4)
        self.assertIn("Only 4 historical analog years found", report.warning)

    def test_markdown_can_include_continuous_context(self) -> None:
        agent = SeasonalAgent(
            spread_history=self.spread_history,
            config=SeasonalAgentConfig(commodity="corn", spread_id="CORN_NZ"),
        )

        report = agent.analyze(
            "2024-06-07",
            continuous_context=self.continuous_context,
        )
        markdown = report_to_markdown(report)

        self.assertIn("## Market Context", markdown)
        self.assertIn(MARKET_CONTEXT_NOTE, markdown)
        self.assertIn("**Symbol:** CORN_CC1", markdown)
        self.assertIn("**Trend:** Weak", markdown)

    def test_json_includes_continuous_context_without_changing_signal(self) -> None:
        agent = SeasonalAgent(
            spread_history=self.spread_history,
            config=SeasonalAgentConfig(commodity="corn", spread_id="CORN_NZ"),
        )

        baseline = agent.analyze("2024-06-07")
        with_context = agent.analyze(
            "2024-06-07",
            continuous_context=self.continuous_context,
        )
        payload = json.loads(report_to_json(with_context))

        self.assertEqual(with_context.seasonal_signal, baseline.seasonal_signal)
        self.assertEqual(with_context.median_move, baseline.median_move)
        self.assertEqual(with_context.historical_years_used, baseline.historical_years_used)
        self.assertEqual(payload["continuous_context"]["symbol"], "CORN_CC1")
        self.assertEqual(payload["continuous_context"]["latest_date"], "2026-06-23")
        self.assertEqual(
            payload["continuous_context"]["seasonal_signal_note"],
            MARKET_CONTEXT_NOTE,
        )

    def test_future_years_are_not_used_as_analogs(self) -> None:
        future_rows = self.spread_history[self.spread_history["year"] == 2023].copy()
        future_rows["year"] = 2025
        future_rows["date"] = pd.to_datetime(future_rows["date"]) + pd.DateOffset(years=2)
        history_with_future = pd.concat(
            [self.spread_history, future_rows],
            ignore_index=True,
        )
        agent = SeasonalAgent(
            spread_history=history_with_future,
            config=SeasonalAgentConfig(commodity="corn", spread_id="CORN_NZ"),
        )

        report = agent.analyze("2024-06-07")

        self.assertEqual(report.historical_years_used, 19)
        self.assertNotIn(2025, {a["year"] for a in report.top_analog_years})

    def test_single_contract_year_explains_missing_analogs(self) -> None:
        single_contract_year = self.spread_history.copy()
        single_contract_year["year"] = 2026
        agent = SeasonalAgent(
            spread_history=single_contract_year,
            config=SeasonalAgentConfig(commodity="corn", spread_id="CORN_NZ"),
        )

        with self.assertRaisesRegex(ValueError, "No prior contract years"):
            agent.analyze("2026-06-16")

    def test_cli_can_emit_json(self) -> None:
        parser = [
            "seasonal",
            "--data",
            str(self.data_path),
            "--analysis-date",
            "2024-06-07",
            "--format",
            "json",
        ]

        args = build_parser().parse_args(parser)
        payload = json.loads(run_seasonal_agent(args))

        self.assertEqual(payload["analysis_date"], "2024-06-07")
        self.assertEqual(payload["spread_id"], "CORN_NZ")
        self.assertFalse(payload["analysis_date_is_latest"])
        self.assertEqual(payload["analysis_date_message"], "Analysis date: 2024-06-07")

    def test_cli_uses_explicit_analysis_date(self) -> None:
        args = build_parser().parse_args(
            [
                "seasonal",
                "--data",
                str(self.data_path),
                "--analysis-date",
                "2024-06-07",
                "--format",
                "markdown",
            ]
        )

        markdown = run_seasonal_agent(args)

        self.assertIn("Analysis date: 2024-06-07", markdown)
        self.assertNotIn("(latest available data)", markdown)

    def test_cli_uses_latest_available_date_when_analysis_date_is_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            processed_path = Path(tmp) / "corn_july_dec_spread_history.csv"
            processed_schema = self.spread_history.rename(columns={"year": "contract_year"})
            processed_schema = processed_schema.drop(columns=["year"], errors="ignore")
            processed_schema = processed_schema[processed_schema["date"] <= "2024-06-07"]
            processed_schema.to_csv(processed_path, index=False)
            args = build_parser().parse_args(
                [
                    "seasonal",
                    "--spread-id",
                    "CORN_NZ",
                    "--data",
                    str(processed_path),
                    "--format",
                    "json",
                ]
            )

            payload = json.loads(run_seasonal_agent(args))

            self.assertEqual(payload["analysis_date"], "2024-06-07")
            self.assertTrue(payload["analysis_date_is_latest"])
            self.assertEqual(
                payload["analysis_date_message"],
                "Analysis date: 2024-06-07 (latest available data)",
            )

    def test_cli_empty_dataset_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty_path = Path(tmp) / "empty_spread_history.csv"
            pd.DataFrame(
                columns=["date", "contract_year", "commodity", "spread_id", "spread_value"]
            ).to_csv(empty_path, index=False)
            args = build_parser().parse_args(
                [
                    "seasonal",
                    "--data",
                    str(empty_path),
                    "--format",
                    "json",
                ]
            )

            with self.assertRaisesRegex(ValueError, "Spread history dataset is empty"):
                run_seasonal_agent(args)

    def test_cli_seasonal_can_attach_continuous_context_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            processed_path = Path(tmp) / "corn_july_dec_spread_history.csv"
            processed_schema = self.spread_history.rename(columns={"year": "contract_year"})
            processed_schema = processed_schema.drop(columns=["year"], errors="ignore")
            processed_schema.to_csv(processed_path, index=False)
            args = build_parser().parse_args(
                [
                    "seasonal",
                    "--spread-id",
                    "CORN_NZ",
                    "--analysis-date",
                    "2024-06-07",
                    "--data",
                    str(processed_path),
                    "--continuous-context",
                    str(self.continuous_data_path),
                    "--format",
                    "json",
                ]
            )

            payload = json.loads(run_seasonal_agent(args))

            self.assertEqual(payload["spread_id"], "CORN_NZ")
            self.assertEqual(payload["continuous_context"]["symbol"], "CORN_CC1")
            self.assertIn("move_already_priced", payload)
            self.assertEqual(payload["continuous_context"]["latest_date"], "2026-06-23")


if __name__ == "__main__":
    unittest.main()
