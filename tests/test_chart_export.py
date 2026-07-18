from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thresh.chart_export import (
    ChartCalendarSpreadSpec,
    normalize_chart_calendar_spread_csv,
    summarize_spread_series,
    write_normalized_calendar_spread,
)
from thresh.cli import build_parser, run_normalize_calendar


class ChartExportTests(unittest.TestCase):
    def test_normalizes_calendar_spread_chart_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = self._write_raw_export(Path(tmp) / "raw.csv")
            spec = ChartCalendarSpreadSpec(
                commodity="corn",
                spread_id="CORN_NZ",
                symbol_prefix="ZC",
                nearby_month="N",
                deferred_month="Z",
                contract_year=2026,
            )

            normalized = normalize_chart_calendar_spread_csv(raw_path, spec)

            self.assertEqual(len(normalized), 3)
            self.assertEqual(set(normalized["year"]), {2026})
            self.assertEqual(set(normalized["nearby_symbol"]), {"ZCN26"})
            self.assertEqual(set(normalized["deferred_symbol"]), {"ZCZ26"})
            self.assertEqual(normalized.iloc[-1]["spread_value"], 9.25)
            self.assertEqual(normalized.iloc[-1]["volume"], 12)
            self.assertEqual(normalized.iloc[-1]["ema_20"], 10.0)

    def test_writes_normalized_file_and_summarizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = self._write_raw_export(tmp_path / "raw.csv")
            output_path = tmp_path / "normalized.csv"
            spec = ChartCalendarSpreadSpec(
                commodity="corn",
                spread_id="CORN_NZ",
                symbol_prefix="ZC",
                nearby_month="N",
                deferred_month="Z",
                contract_year=2026,
            )

            normalized = write_normalized_calendar_spread(raw_path, output_path, spec)
            summary = summarize_spread_series(normalized)

            self.assertTrue(output_path.exists())
            self.assertEqual(summary.date_start, "2026-06-12")
            self.assertEqual(summary.date_end, "2026-06-16")
            self.assertEqual(summary.latest_close, 9.25)
            self.assertEqual(summary.previous_close, 10.75)
            self.assertEqual(summary.one_day_change, -1.5)
            self.assertEqual(summary.trend_state, "weak below major averages")

    def test_cli_normalize_calendar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = self._write_raw_export(tmp_path / "raw.csv")
            output_path = tmp_path / "normalized.csv"
            args = build_parser().parse_args(
                [
                    "normalize-calendar",
                    "--input",
                    str(raw_path),
                    "--output",
                    str(output_path),
                    "--contract-year",
                    "2026",
                ]
            )

            message = run_normalize_calendar(args)

            self.assertIn("Wrote 3 rows", message)
            self.assertIn("Latest close: 9.25 cents", message)
            self.assertTrue(output_path.exists())

    @staticmethod
    def _write_raw_export(path: Path) -> Path:
        pd.DataFrame(
            {
                "Timestamp (UTC)": ["2026-06-12", "2026-06-15", "2026-06-16"],
                "Open": [12.25, 11.0, 10.0],
                "High": [12.25, 11.0, 10.0],
                "Low": [12.0, 10.5, 9.0],
                "Close": [12.25, 10.75, 9.25],
                "ZC Jul26-Dec26 Calendar: Volume": [4, 8, 12],
                "ZC Jul26-Dec26 Calendar: MA ma (20,C,ema,0)": [11.5, 10.75, 10.0],
                "ZC Jul26-Dec26 Calendar: MA ma (50,C,ema,0)": [12.5, 11.75, 11.0],
                "ZC Jul26-Dec26 Calendar: MA ma (200,C,ema,0)": [14.5, 13.75, 13.0],
                "ZC Jul26-Dec26 Calendar: Bollinger Bands Top (20,C,2,ma,y)": [13, 12, 11],
                "ZC Jul26-Dec26 Calendar: Bollinger Bands Median (20,C,2,ma,y)": [12, 11, 10],
                "ZC Jul26-Dec26 Calendar: Bollinger Bands Bottom (20,C,2,ma,y)": [11, 10, 9],
            }
        ).to_csv(path, index=False)
        return path


if __name__ == "__main__":
    unittest.main()
