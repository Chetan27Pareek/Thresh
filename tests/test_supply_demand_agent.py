from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thresh.cli import build_parser, run_supply_demand_agent  # noqa: E402
from thresh.supply_demand_agent import (  # noqa: E402
    SupplyDemandAgent,
    SupplyDemandAgentConfig,
    report_to_markdown,
)


class SupplyDemandAgentTests(unittest.TestCase):
    def test_corn_report_combines_wasde_and_crop_condition(self) -> None:
        agent = SupplyDemandAgent(
            wasde_data=self._wasde_fixture(),
            crop_condition_data=self._crop_fixture(),
            config=SupplyDemandAgentConfig(commodity="corn"),
        )

        report = agent.analyze("latest")

        self.assertEqual(report.commodity, "corn")
        self.assertEqual(report.analysis_date, "2026-06-15")
        self.assertEqual(report.marketing_year, "2026/27")
        self.assertEqual(report.stocks_situation, "Tight")
        self.assertEqual(report.wasde_revision_trend, "Tightening")
        self.assertEqual(report.crop_condition_good_excellent_pct, 65.0)
        self.assertEqual(report.crop_condition_trajectory, "Deteriorating")
        self.assertEqual(report.fundamental_bias, "Bullish")
        self.assertEqual(report.warnings, [])

    def test_soybeans_continue_with_wasde_only_when_crop_missing(self) -> None:
        agent = SupplyDemandAgent(
            wasde_data=self._wasde_fixture(),
            crop_condition_data=self._crop_fixture(),
            config=SupplyDemandAgentConfig(commodity="soybeans"),
        )

        report = agent.analyze("latest")

        self.assertEqual(report.analysis_date, "2026-06-10")
        self.assertEqual(report.stocks_situation, "Ample")
        self.assertEqual(report.wasde_revision_trend, "Loosening")
        self.assertEqual(report.crop_condition_good_excellent_pct, None)
        self.assertEqual(report.crop_condition_trajectory, "Unavailable")
        self.assertEqual(report.fundamental_bias, "Bearish")
        self.assertIn("Crop condition data unavailable for soybeans", report.warnings[0])

    def test_markdown_output_contains_required_sections(self) -> None:
        agent = SupplyDemandAgent(
            wasde_data=self._wasde_fixture(),
            crop_condition_data=self._crop_fixture(),
            config=SupplyDemandAgentConfig(commodity="corn"),
        )

        markdown = report_to_markdown(agent.analyze("2026-06-15"))

        self.assertIn("# Supply/Demand Agent Report", markdown)
        self.assertIn("**Fundamental Bias:** Bullish", markdown)
        self.assertIn("**Stocks Situation:** Tight", markdown)
        self.assertIn("**Crop Condition Trajectory:** Deteriorating", markdown)

    def test_json_output_is_machine_readable(self) -> None:
        agent = SupplyDemandAgent(
            wasde_data=self._wasde_fixture(),
            crop_condition_data=self._crop_fixture(),
            config=SupplyDemandAgentConfig(commodity="corn"),
        )

        payload = json.loads(agent.analyze("latest").to_json())

        self.assertEqual(payload["commodity"], "corn")
        self.assertEqual(payload["fundamental_bias"], "Bullish")
        self.assertIn("stocks_to_use_percentile", payload)

    def test_cli_supply_demand_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wasde_path = tmp_path / "wasde.csv"
            crop_path = tmp_path / "crop_conditions.csv"
            self._wasde_fixture().to_csv(wasde_path, index=False)
            self._crop_fixture().to_csv(crop_path, index=False)
            args = build_parser().parse_args(
                [
                    "supply-demand",
                    "--commodity",
                    "corn",
                    "--analysis-date",
                    "latest",
                    "--wasde-data",
                    str(wasde_path),
                    "--crop-condition-data",
                    str(crop_path),
                    "--format",
                    "json",
                ]
            )

            payload = json.loads(run_supply_demand_agent(args))

            self.assertEqual(payload["commodity"], "corn")
            self.assertEqual(payload["analysis_date"], "2026-06-15")
            self.assertEqual(payload["fundamental_bias"], "Bullish")

    @staticmethod
    def _wasde_fixture() -> pd.DataFrame:
        rows = []
        for commodity, stock_values in {
            "corn": [16.0, 12.0, 10.0, 8.0],
            "soybeans": [4.0, 8.0, 10.0, 12.0],
        }.items():
            for date, stocks_to_use in zip(
                ["2026-03-10", "2026-04-10", "2026-05-10", "2026-06-10"],
                stock_values,
            ):
                total_use = 1000.0
                rows.append(
                    {
                        "date": date,
                        "commodity": commodity,
                        "marketing_year": "2026/27",
                        "production": 1200.0,
                        "exports": 300.0,
                        "ending_stocks": total_use * stocks_to_use / 100,
                        "total_use": total_use,
                        "stocks_to_use": stocks_to_use,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _crop_fixture() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "date": "2026-06-01",
                    "commodity": "corn",
                    "good_pct": 60,
                    "excellent_pct": 15,
                    "good_excellent_pct": 75,
                },
                {
                    "date": "2026-06-08",
                    "commodity": "corn",
                    "good_pct": 57,
                    "excellent_pct": 13,
                    "good_excellent_pct": 70,
                },
                {
                    "date": "2026-06-15",
                    "commodity": "corn",
                    "good_pct": 53,
                    "excellent_pct": 12,
                    "good_excellent_pct": 65,
                },
            ]
        )


if __name__ == "__main__":
    unittest.main()
